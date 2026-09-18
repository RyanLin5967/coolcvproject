"""Submit once or resume the existing cloud-only segmentation pilot."""
import fcntl
import io
import zipfile
from pathlib import Path

import modal

from coveragecv.artifacts import file_digest, read_json, safe_child, write_json
from coveragecv.training.cloud_budget import reserve_segment_refinement

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/segment_refinement"


def main():
    request = {"protocol_sha256": file_digest(OUT / "protocol.json")}
    with (OUT / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (OUT / "result.json").exists():
            if (OUT / "call.json").exists():
                saved = read_json(OUT / "call.json")
                if saved["request"] != request:
                    raise ValueError("Saved pilot call belongs to a different protocol")
                call = modal.FunctionCall.from_id(saved["call_id"])
            else:
                if (OUT / "reservation.json").exists():
                    raise ValueError("Orphan pilot reservation; never automatically retry")
                reservation = reserve_segment_refinement()
                write_json(OUT / "reservation.json", {"reservation_id": reservation, "request": request})
                call = modal.Function.from_name("coveragecv-segment-refinement", "experiment").spawn(**request)
                write_json(OUT / "call.json", {"request": request, "call_id": call.object_id})
                print({"status": "submitted", "call_id": call.object_id}, flush=True)
            with zipfile.ZipFile(io.BytesIO(call.get())) as archive:
                if not {"result.json", "train_selection.json"}.issubset(archive.namelist()):
                    raise ValueError("Incomplete pilot return")
                for name in archive.namelist():
                    if not name.endswith(".json"):
                        raise ValueError("Unexpected pilot file")
                    safe_child(OUT, name).write_bytes(archive.read(name))
        result = read_json(OUT / "result.json")
        if result["protocol_sha256"] != request["protocol_sha256"]:
            raise ValueError("Pilot result protocol mismatch")
        print({k: v for k, v in result.items() if k != "weight_files"}, flush=True)


if __name__ == "__main__":
    main()
