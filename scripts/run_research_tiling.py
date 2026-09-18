"""Collect the predeclared GPU tiling interaction without local model computation."""
import fcntl
import io
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import modal

from coveragecv.artifacts import file_digest, read_json, safe_child, write_json
from coveragecv.training.cloud_budget import reserve_research_tiling

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/research_v2"


def collect(case, spec):
    folder = OUT / "tiled" / case
    folder.mkdir(parents=True, exist_ok=True)
    request = {"case": case, "protocol_sha256": file_digest(OUT / "tiling_protocol.json")}
    checkpoint, baseline = Path(spec["checkpoint"]), Path(spec["baseline"])
    if file_digest(checkpoint) != spec["checkpoint_sha256"] or file_digest(baseline) != spec["baseline_sha256"]:
        raise ValueError("Declared model or single-pass baseline changed")
    with (folder / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (folder / "size_gated.json").exists():
            if (folder / "call.json").exists():
                saved = read_json(folder / "call.json")
                if saved["request"] != request:
                    raise ValueError("Saved tiling call belongs to another protocol")
                call = modal.FunctionCall.from_id(saved["call_id"])
            else:
                if (folder / "reservation.json").exists():
                    raise ValueError("Orphan tiling reservation; resolve the existing call before retry")
                reservation = reserve_research_tiling()
                write_json(folder / "reservation.json", {"reservation_id": reservation, "request": request})
                remote = modal.Function.from_name("coveragecv-research-tiling", "evaluate")
                call = remote.spawn(**request, checkpoint=checkpoint.read_bytes(), baseline=read_json(baseline))
                write_json(folder / "call.json", {"request": request, "call_id": call.object_id})
                print({"case": case, "status": "submitted", "call_id": call.object_id}, flush=True)
            returned = call.get()
            with zipfile.ZipFile(io.BytesIO(returned)) as archive:
                if set(archive.namelist()) != {"full_frame.json", "full_frame_nms.json", "tiled.json",
                                               "comparison.json", "size_gated.json"}:
                    raise ValueError("Unexpected tiling output inventory")
                for name in archive.namelist():
                    safe_child(folder, name).write_bytes(archive.read(name))
        result = read_json(folder / "size_gated.json")
        if (result["checkpoint_sha256"] != spec["checkpoint_sha256"]
                or result["research_tiling_protocol_sha256"] != request["protocol_sha256"]
                or result["device"] != "cuda" or not result["per_class_routing_audit"]):
            raise ValueError("Tiled evaluation provenance mismatch")
        print({"case": case, "AP": result["metrics"]["AP"]}, flush=True)
        return {k: v for k, v in result.items() if k != "predictions"}


def main():
    protocol = read_json(OUT / "tiling_protocol.json")
    results, failures = {}, {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(collect, case, spec): case for case, spec in protocol["cases"].items()}
        for future in as_completed(futures):
            case = futures[future]
            try:
                results[case] = future.result()
            except Exception as error:  # noqa: BLE001 -- preserve every independent evaluation outcome
                failures[case] = {"type": type(error).__name__, "message": str(error)}
                print({"case": case, "status": "failed", "type": type(error).__name__}, flush=True)
            write_json(OUT / "tiled_results.json", {"status": "running", "results": results, "failures": failures})
    write_json(OUT / "tiled_results.json", {"status": "completed" if not failures else "completed_with_failures",
                                          "results": results, "failures": failures})


if __name__ == "__main__":
    main()
