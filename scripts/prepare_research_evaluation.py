"""Package evaluation references into a separate image inaccessible to training."""
import shutil
import zipfile
from pathlib import Path

from coveragecv.artifacts import file_digest, read_json, verify, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/research_v2"


def main():
    if (OUT / "evaluation_protocol.json").exists():
        raise ValueError("Evaluation protocol is already frozen")
    training = read_json(OUT / "protocol.json")
    destination = OUT / "references"
    destination.mkdir(exist_ok=True)
    source = OUT / "evaluation-source/coveragecv"
    shutil.copytree(ROOT / "src/coveragecv", source, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    source_files = {str(p.relative_to(source)): file_digest(p) for p in source.rglob("*") if p.is_file()}
    references = {}
    for task, data in training["datasets"].items():
        bundle = Path(data["local_reference"])
        manifest = verify(bundle)
        payload = destination / f"{task}.zip"
        with zipfile.ZipFile(payload, "w", zipfile.ZIP_STORED) as archive:
            for name in ("manifest.json", *manifest["files"]):
                archive.write(bundle / name, name)
        references[task] = {"payload_sha256": file_digest(payload), "bundle_digest": manifest["digest"]}
    write_json(OUT / "evaluation_protocol.json", {
        "references": references, "source_files": source_files,
        "training_protocol_sha256": file_digest(OUT / "protocol.json"),
        "split": "valid", "device": "cuda", "training_image_has_reference_bundle": False,
        "comparisons": "Score twelve final checkpoints and all six parent checkpoints with the same GPU evaluator",
        "budget": {"max_calls": 18, "reservation_per_call": .2, "total_reservation": 3.6, "cash": 0},
    })
    print({"reference_archives": len(references), "total_bytes": sum(p.stat().st_size for p in destination.glob('*.zip'))})


if __name__ == "__main__":
    main()
