"""Cloud collection must resume paid calls without silently spending again."""
import importlib.util
import io
import zipfile
from pathlib import Path

import pytest

from coveragecv.artifacts import file_digest, read_json, write_json


@pytest.fixture
def collection(tmp_path, monkeypatch):
    path = Path(__file__).parents[1] / "scripts/run_research_v2.py"
    module_spec = importlib.util.spec_from_file_location("research_collection", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    monkeypatch.setattr(module, "OUT", tmp_path)
    spec = {"task": "pawns", "arm": "aware", "alpha": 3, "exclusive_groups": [["black", "white"]],
            "steps": 2000, "seed": 17}
    parent = {"checkpoint_sha256": "parent", "view_digest": "learner"}
    protocol = {"datasets": {"pawns": {"parents": {"aware": parent}}}, "cases": {"case": spec}}
    write_json(tmp_path / "protocol.json", protocol)
    folder = tmp_path / "runs/case"
    folder.mkdir(parents=True)
    request = {"case": "case", "protocol_sha256": file_digest(tmp_path / "protocol.json")}
    # Reserving or spawning is always wrong in these saved-call recovery scenarios.
    def unexpected(*args, **kwargs):
        raise AssertionError("Would have submitted or reserved another paid call")
    monkeypatch.setattr(module, "reserve_research_v2_run", unexpected)
    monkeypatch.setattr(module.modal.Function, "from_name", unexpected)
    return module, spec, protocol, folder, request


def test_orphan_reservation_never_retries_cloud_training(collection):
    module, spec, protocol, folder, request = collection
    write_json(folder / "reservation.json", {"reservation_id": "paid", "request": request})
    with pytest.raises(RuntimeError, match="Orphan"):
        module.collect("case", spec, protocol)


def test_saved_receipt_cannot_be_rebound_to_a_different_protocol(collection):
    module, spec, protocol, folder, _ = collection
    write_json(folder / "receipt.json", {"case": "case", "spec": spec, "protocol_sha256": "old"})
    with pytest.raises(ValueError, match="different protocol"):
        module.collect("case", spec, protocol)


def test_saved_call_download_is_durable_before_scoring(collection, monkeypatch):
    module, spec, protocol, folder, request = collection
    write_json(folder / "call.json", {"call_id": "existing", "request": request})
    checkpoint = folder / "source.pt"
    checkpoint.write_bytes(b"test checkpoint")
    run = {**spec, "status": "completed", "detector_sha256": file_digest(checkpoint),
           "warm_start_sha256": "parent", "view_digest": "learner", "elapsed_seconds": 30}
    write_json(folder / "source.json", run)
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.write(checkpoint, "detector.pt")
        archive.write(folder / "source.json", "run.json")
        archive.writestr("progress.json", "[]")
    downloads = []

    class ExistingCall:
        def get(self):
            downloads.append(1)
            return payload.getvalue()

    monkeypatch.setattr(module.modal.FunctionCall, "from_id", lambda _: ExistingCall())
    receipt = module.collect("case", spec, protocol)
    assert receipt == read_json(folder / "receipt.json")
    assert receipt["checkpoint_sha256"] == file_digest(folder / "detector.pt")
    assert not (folder / "evaluation.json").exists()
    assert module.collect("case", spec, protocol) == receipt
    assert downloads == [1]
