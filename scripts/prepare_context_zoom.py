"""Freeze one contextual zoom hypothesis; no local training, inference or selection."""
import shutil
from pathlib import Path

from coveragecv.artifacts import file_digest, read_json, verify, write_json
from coveragecv.training.context_zoom import POLICY

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/continuous/context_zoom"


def main():
    if (OUT / "protocol.json").exists():
        raise ValueError("Context zoom protocol is already frozen")
    OUT.mkdir(parents=True, exist_ok=True)
    previous = read_json(ROOT / "artifacts/continuous/geometry/protocol.json")
    spec = read_json(ROOT / "artifacts/full_chess/experiment.json")
    payload = ROOT / "artifacts/continuous/geometry/payload.zip"
    if file_digest(payload) != previous["payload_sha256"]:
        raise ValueError("Reusable images/checkpoints changed")
    shutil.copy2(ROOT / "cloud-requirements.txt", OUT / "requirements.txt")
    source = OUT / "source/coveragecv"
    shutil.copytree(ROOT / "src/coveragecv", source, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    protocol = {"status": "frozen_before_gpu_gate", "policy": POLICY,
                "payload_sha256": previous["payload_sha256"], "checkpoints": previous["checkpoints"],
                "view_digest": verify(Path(spec["partial_view"]))["digest"],
                "reference_bundle_digest": previous["reference_bundle_digest"],
                "requirements_sha256": file_digest(OUT / "requirements.txt"),
                "source_files": {str(p.relative_to(source)): file_digest(p) for p in source.rglob("*") if p.is_file()},
                "budget": {"maximum_calls": 1, "maximum_usd": .60, "cash": 0, "timeout_seconds": 900},
                "selection": "One fixed wide-context crop rule; fresh actual candidate predictions on image-hashed20% observed TRAIN subset; reject before validation on failure. No margin/blend search.",
                "mechanism": "Allocate more feature-grid samples per object with the SAME detector and annotation convention. Cropping adds no native pixel information.",
                "limitations": "Detector saw train holdout during detector training. Published zoom papers use different training/architectures; this is an original exploratory inference adapter, not a reproduction.",
                "sources": ["https://arxiv.org/abs/2303.15390", "https://github.com/tchittesh/lzu",
                            "https://arxiv.org/abs/2303.08747", "https://github.com/akhilpm/DroneDetectron2"]}
    write_json(OUT / "protocol.json", protocol)
    print({"status": "frozen", "protocol_sha256": file_digest(OUT / "protocol.json")})


if __name__ == "__main__":
    main()
