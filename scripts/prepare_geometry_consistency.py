"""Freeze observed training proposals, fixed geometry policy and unchanged controls."""
import shutil
import zipfile
from pathlib import Path

from coveragecv.artifacts import file_digest, read_json, verify, write_json
from coveragecv.training.geometry_consistency import POLICY

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/continuous/geometry"


def main():
    if (OUT / "protocol.json").exists():
        raise ValueError("Geometry protocol is already frozen")
    OUT.mkdir(parents=True, exist_ok=True)
    spec = read_json(ROOT / "artifacts/full_chess/experiment.json")
    view, reference = Path(spec["partial_view"]), Path(spec["complete_bundle"])
    proposals = ROOT / "artifacts/localization_v2/all-pieces/aware/proposals.json"
    proposal_data = read_json(proposals)
    checkpoints = {"aware": ROOT / "artifacts/research_v2/runs/all-pieces-exclusive/detector.pt",
                   "complete": ROOT / "artifacts/research_v2/runs/all-pieces-full_power/detector.pt"}
    with zipfile.ZipFile(OUT / "payload.zip", "w", zipfile.ZIP_STORED) as archive:
        for prefix, path in (("view", view), ("reference", reference)):
            manifest = verify(path)
            for name in ("manifest.json", *manifest["files"]):
                archive.write(path / name, f"{prefix}/{name}")
        archive.write(proposals, "proposals.json")
        for case, path in checkpoints.items():
            run = read_json(path.parent / "run.json")
            if run["status"] != "completed" or run["detector_sha256"] != file_digest(path):
                raise ValueError("Invalid evaluation checkpoint")
            archive.write(path, f"checkpoints/{case}.pt")
    shutil.copy2(ROOT / "cloud-requirements.txt", OUT / "requirements.txt")
    source = OUT / "source/coveragecv"
    shutil.copytree(ROOT / "src/coveragecv", source, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    protocol = {"status": "frozen_before_gate", "policy": POLICY, "payload_sha256": file_digest(OUT / "payload.zip"),
                "proposals_sha256": file_digest(proposals),
                "proposal_checkpoint_sha256": proposal_data["binding"]["checkpoint_sha256"],
                "source_files": {str(p.relative_to(source)): file_digest(p) for p in source.rglob("*") if p.is_file()},
                "requirements_sha256": file_digest(OUT / "requirements.txt"),
                "reference_bundle_digest": verify(reference)["digest"],
                "checkpoints": {k: {"path": str(v), "checkpoint_sha256": file_digest(v)} for k, v in checkpoints.items()},
                "budget": {"maximum_calls": 1, "maximum_usd": .40, "cash": 0, "timeout_seconds": 600},
                "comparison": "Both aware/full use identical fixed geometric rule; preserve original scores and detection cardinality. Paired CUDA original/flip inference controls floating-point batching differences.",
                "gate_limits": "TRAIN predictions were saved on MPS before local compute prohibition. Gate executes in cloud. Detector already saw the train holdout. No validation thresholds are tuned.",
                "selection": "One fixed method; reject before validation unless mean IoU improves .003 and strict .90IoU match count does not fall.",
                "sources": ["https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/test_time_augmentation.py",
                            "https://arxiv.org/abs/1910.13302", "https://arxiv.org/abs/2206.10107"]}
    write_json(OUT / "protocol.json", protocol)
    print({"status": "frozen", "protocol_sha256": file_digest(OUT / "protocol.json")})


if __name__ == "__main__":
    main()
