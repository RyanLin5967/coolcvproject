"""Freeze selected/control recipes before reading construction test annotations."""
import shutil
from pathlib import Path

from coveragecv.artifacts import file_digest, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/continuous/confirmation"


def main():
    if (OUT / "protocol.json").exists():
        raise ValueError("Confirmation protocol is already frozen")
    OUT.mkdir(parents=True, exist_ok=True)
    cross = read_json(ROOT / "artifacts/continuous/cross_resolution/protocol.json")
    tile = read_json(ROOT / "artifacts/research_v2/tiling_protocol.json")
    cases = {}
    for name in ("construction-aware-control", "construction-aware-scale",
                 "construction-full-control", "construction-full-combined"):
        binding = cross["checkpoints"][name]
        parent = Path(binding["local_parent"])
        baseline = parent / "evaluation.json"
        result = read_json(baseline)
        if result["checkpoint_sha256"] != binding["checkpoint_sha256"] or result["split"] != "valid":
            raise ValueError("Saved validation baseline does not bind its checkpoint")
        target = OUT / "baselines" / f"{name}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(baseline, target)
        cases[name] = {**binding, "baseline_sha256": file_digest(target)}
    shutil.copy2(ROOT / "cloud-requirements.txt", OUT / "requirements.txt")
    source = OUT / "source/coveragecv"
    shutil.copytree(ROOT / "src/coveragecv", source, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    protocol = {"status": "frozen_before_test_scoring", "cases": cases,
                "payload_sha256": cross["payload_sha256"], "reference": cross["references"]["construction"],
                "tile_policy": tile["tile_policy"], "class_gate": tile["class_gate"],
                "source_files": {str(p.relative_to(source)): file_digest(p) for p in source.rglob("*") if p.is_file()},
                "requirements_sha256": file_digest(OUT / "requirements.txt"),
                "primary": "Single-pass COCO AP on the original reserved construction TEST split. No test labels inspected to select recipes; no test tuning or checkpoint changes after this declaration.",
                "secondary": "Apply the unchanged original384/stride256/NMS.5/edge2 validation tile policy and original partial-TRAIN class gate to all four models. Keep every outcome, including regressions.",
                "interpretation": "One seed. Validation selected aware native640 and full combined640. Within-label controls have same parent and updates; full combined also changes sampling. Vendor test is not a new external dataset.",
                "budget": {"calls": 4, "maximum_per_call": .12, "maximum": .48, "cash": 0},
                "test_access": "Preparation reads existing validation receipts and immutable manifest hashes only; first construction test metric computation occurs remotely after this protocol is frozen."}
    write_json(OUT / "protocol.json", protocol)
    print({"cases": len(cases), "protocol_sha256": file_digest(OUT / "protocol.json")})


if __name__ == "__main__":
    main()
