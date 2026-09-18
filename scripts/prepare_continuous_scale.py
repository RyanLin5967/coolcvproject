"""Freeze the next loop iteration without importing or running a local model."""
import shutil
import zipfile
from pathlib import Path

from coveragecv.artifacts import file_digest, read_json, verify, write_json
from coveragecv.training.exposure import build_exposure_plan

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/continuous/scale"


def main():
    if (OUT / "protocol.json").exists():
        raise ValueError("Scale protocol is already frozen")
    previous = read_json(ROOT / "artifacts/research_v2/protocol.json")["datasets"]["construction"]
    parents, cases = {}, {}
    OUT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT / "inputs.zip", "w", zipfile.ZIP_STORED) as archive:
        for arm, case in (("aware", "construction-exclusive"), ("complete_reference", "construction-full_power")):
            view = Path(previous["parents"][arm]["local_view"])
            manifest = verify(view)
            parent = ROOT / "artifacts/research_v2/runs" / case
            run = read_json(parent / "run.json")
            if (run["status"] != "completed" or run["arm"] != arm or run["view_digest"] != manifest["digest"]
                    or run["detector_sha256"] != file_digest(parent / "detector.pt")):
                raise ValueError("Parent is not a verified completed learner checkpoint")
            for name in ("manifest.json", *manifest["files"]):
                archive.write(view / name, f"{arm}/view/{name}")
            for name in ("run.json", "detector.pt"):
                archive.write(parent / name, f"{arm}/parent/{name}")
            parents[arm] = {"view_digest": manifest["digest"], "checkpoint_sha256": run["detector_sha256"],
                            "local_parent": str(parent), "local_view": str(view)}
    partial = Path(parents["aware"]["local_view"])
    plans = {name: build_exposure_plan(partial, samples=8000, seed=20260919, repeat=repeat)
             for name, repeat in (("uniform", False), ("repeat", True))}
    for name, arm, resolution, sampling in (
        ("aware-control", "aware", 512, "uniform"), ("aware-scale", "aware", 640, "uniform"),
        ("aware-repeat", "aware", 512, "repeat"), ("aware-combined", "aware", 640, "repeat"),
        ("complete-control", "complete_reference", 512, "uniform"),
        ("complete-combined", "complete_reference", 640, "repeat")):
        cases[name] = {"arm": arm, "resolution": resolution, "sampling": sampling,
                       "exclusive_groups": previous["exclusive_groups"] if arm == "aware" else [],
                       "alpha": 1, "steps": 2000, "seed": 20260919}
    source = OUT / "source/coveragecv"
    shutil.copytree(ROOT / "src/coveragecv", source, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    reference = ROOT / "artifacts/research_v2/references/construction.zip"
    protocol = {"kind": "full_frame_scale_and_exposure", "status": "frozen_before_training",
                "parents": parents, "cases": cases, "exposure_plans": plans,
                "source_files": {str(p.relative_to(source)): file_digest(p) for p in source.rglob("*") if p.is_file()},
                "inputs_sha256": file_digest(OUT / "inputs.zip"), "reference_payload_sha256": file_digest(reference),
                "reference_bundle_digest": verify(Path(previous["local_reference"]))["digest"],
                "budget": {"training": 6, "evaluation": .72, "smoke": .20, "total_ceiling": 6.92, "cash": 0},
                "controls": "Full and partial arms replay exactly the same partial-derived exposure sequence. Same update count, final EMA, LR, Alpha1 and stock augmentation. No validation during training.",
                "scale": "All source train images are640x640; native640 preserves input detail otherwise downsampled to512. Positional parameterization unchanged.",
                "sampling": "LVIS class/image repeat factors, t=.20, weighted replacement fixed in advance. Uniform uses replacement too. Not a byte-for-byte Detectron2 sampler reproduction.",
                "promotion": "Report all six outcomes. Require aggregate gain >=1 AP over matched control to call material; independent seed/test required for generalization claims.",
                "sources": ["https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/samplers/distributed_sampler.py",
                            "https://rfdetr.roboflow.com/latest/learn/train/training-parameters/"]}
    write_json(OUT / "protocol.json", protocol)
    print({"status": "frozen", "cases": len(cases), "protocol_sha256": file_digest(OUT / "protocol.json")})


if __name__ == "__main__":
    main()
