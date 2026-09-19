"""Collect the one authorized contextual zoom pilot without duplicate submissions."""
import fcntl
import io
import json
import zipfile
from pathlib import Path

import modal

from coveragecv.artifacts import file_digest, read_json, safe_child, write_json
from coveragecv.training.cloud_budget import reserve_context_zoom

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/continuous/context_zoom"


def main():
    request = {"protocol_sha256": file_digest(OUT / "protocol.json")}
    with (OUT / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (OUT / "result.json").exists():
            if (OUT / "call.json").exists():
                saved = read_json(OUT / "call.json")
                if saved["request"] != request:
                    raise ValueError("Saved call belongs to another protocol")
                call = modal.FunctionCall.from_id(saved["call_id"])
            else:
                if (OUT / "reservation.json").exists():
                    raise RuntimeError("Orphan reservation: reconcile before resubmission")
                reservation = reserve_context_zoom()
                write_json(OUT / "reservation.json", {"request": request, "reservation_id": reservation})
                call = modal.Function.from_name("coveragecv-context-zoom", "experiment").spawn(**request)
                write_json(OUT / "call.json", {"request": request, "reservation_id": reservation, "call_id": call.object_id})
                print({"status": "submitted", "call_id": call.object_id}, flush=True)
            with zipfile.ZipFile(io.BytesIO(call.get())) as archive:
                if not {"result.json", "train_selection.json"}.issubset(archive.namelist()):
                    raise ValueError("Pilot did not return its gate evidence")
                for name in sorted(archive.namelist(), key=lambda n: n == "result.json"):
                    if not name.endswith(".json"):
                        raise ValueError("Unexpected artifact")
                    target = safe_child(OUT, name)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    write_json(target, json.loads(archive.read(name)))
        result = read_json(OUT / "result.json")
        if result["protocol_sha256"] != request["protocol_sha256"]:
            raise ValueError("Result protocol mismatch")
        print({k: result.get(k) for k in ("status", "gate", "evaluations", "failures")}, flush=True)


if __name__ == "__main__":
    main()
