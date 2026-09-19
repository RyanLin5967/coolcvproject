"""Submit once or reconnect the fixed, cloud-only geometry pilot."""
import fcntl
import io
import json
import zipfile
from pathlib import Path

import modal

from coveragecv.artifacts import file_digest, read_json, safe_child, write_json
from coveragecv.training.cloud_budget import reserve_geometry_consistency

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/continuous/geometry"


def main():
    request = {"protocol_sha256": file_digest(OUT / "protocol.json")}
    with (OUT / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (OUT / "result.json").exists():
            if (OUT / "call.json").exists():
                saved = read_json(OUT / "call.json")
                if saved["request"] != request:
                    raise ValueError("Saved call has a different protocol")
                call = modal.FunctionCall.from_id(saved["call_id"])
            else:
                if (OUT / "reservation.json").exists():
                    raise RuntimeError("Orphan reservation: reconcile before any new submission")
                reservation = reserve_geometry_consistency()
                write_json(OUT / "reservation.json", {"reservation_id": reservation, "request": request})
                call = modal.Function.from_name("coveragecv-geometry-consistency", "experiment").spawn(**request)
                write_json(OUT / "call.json", {"reservation_id": reservation, "request": request, "call_id": call.object_id})
                print({"status": "submitted", "call_id": call.object_id}, flush=True)
            with zipfile.ZipFile(io.BytesIO(call.get())) as archive:
                if not {"result.json", "train_selection.json"}.issubset(archive.namelist()):
                    raise ValueError("Incomplete pilot return")
                # Publish result.json last so an interrupted download remains resumable.
                for name in sorted(archive.namelist(), key=lambda n: n == "result.json"):
                    if not name.endswith(".json"):
                        raise ValueError("Unexpected returned file")
                    target = safe_child(OUT, name)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    write_json(target, json.loads(archive.read(name)))
        result = read_json(OUT / "result.json")
        if result["protocol_sha256"] != request["protocol_sha256"]:
            raise ValueError("Pilot result protocol mismatch")
        print({"status": result["status"], "gate": result["gate"], "evaluations": result["evaluations"],
               "failures": result["failures"]}, flush=True)


if __name__ == "__main__":
    main()
