"""Freeze observed-label pawn assignment ablation without importing a local model."""
import ast
import shutil
import zipfile
from pathlib import Path

from coveragecv.artifacts import file_digest, read_json, verify, write_json
from coveragecv.training.exposure import build_exposure_plan

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/continuous/stable_assignment"


def main():
    if (OUT / "protocol.json").exists():
        raise ValueError("Stable-assignment protocol is already frozen")
    OUT.mkdir(parents=True, exist_ok=True)
    dataset = read_json(ROOT / "artifacts/research_v2/protocol.json")["datasets"]["pawns"]
    bindings = read_json(ROOT / "artifacts/continuous/cross_resolution/protocol.json")["checkpoints"]
    parents = {}
    with zipfile.ZipFile(OUT / "inputs.zip", "w", zipfile.ZIP_STORED) as archive:
        for arm, key in (("aware", "pawns-aware"), ("complete_reference", "pawns-full")):
            binding = bindings[key]
            parent = Path(binding["local_parent"])
            view = Path(dataset["parents"][arm]["local_view"])
            manifest, run = verify(view), read_json(parent / "run.json")
            sha = file_digest(parent / "detector.pt")
            if (run["status"] != "completed" or run["arm"] != arm or run["detector_sha256"] != sha
                    or sha != binding["checkpoint_sha256"] or run["view_digest"] != manifest["digest"]):
                raise ValueError("Parent/view contract mismatch")
            for name in ("manifest.json", *manifest["files"]):
                archive.write(view / name, f"{arm}/view/{name}")
            for name in ("run.json", "detector.pt"):
                archive.write(parent / name, f"{arm}/parent/{name}")
            parents[arm] = {"view_digest": manifest["digest"], "checkpoint_sha256": sha,
                            "local_parent": str(parent), "local_view": str(view)}
    plan = build_exposure_plan(Path(parents["aware"]["local_view"]), samples=8000, seed=20260920, repeat=False)
    cases = {}
    for name, arm, targets, matching in (
        ("aware-control", "aware", False, False), ("aware-targets", "aware", True, False),
        ("aware-matching", "aware", False, True), ("aware-both", "aware", True, True),
        ("complete-control", "complete_reference", False, False),
        ("complete-both", "complete_reference", True, True)):
        cases[name] = {"arm": arm, "steps": 2000, "seed": 20260920, "alpha": 1, "exclusive_groups": [],
                       "resolution": 512, "sampling": "uniform", "stable_assignment": {
                           "position_targets": targets, "position_matching": matching}}
    source = OUT / "source/coveragecv"
    shutil.copytree(ROOT / "src/coveragecv", source, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(ROOT / "cloud-requirements.txt", OUT / "requirements.txt")
    shutil.copy2(ROOT / "tests/test_stable_assignment.py", OUT / "test_stable_assignment.py")
    tree = ast.parse((source / "training/stable_assignment.py").read_text())
    policy = ast.literal_eval(next(node.value for node in tree.body if isinstance(node, ast.Assign)
                                  and any(isinstance(t, ast.Name) and t.id == "POLICY" for t in node.targets)))
    reference = ROOT / "artifacts/research_v2/references/pawns.zip"
    protocol = {"kind": "coverage_aware_stable_assignment", "status": "frozen_before_training", "parents": parents,
                "cases": cases, "policy": policy, "exposure_plans": {"uniform": plan},
                "inputs_sha256": file_digest(OUT / "inputs.zip"), "reference_payload_sha256": file_digest(reference),
                "reference_bundle_digest": verify(Path(dataset["local_reference"]))["digest"],
                "source_files": {str(p.relative_to(source)): file_digest(p) for p in source.rglob("*") if p.is_file()},
                "requirements_sha256": file_digest(OUT / "requirements.txt"),
                "tests_sha256": file_digest(OUT / "test_stable_assignment.py"), "pytest_version": "9.1.1",
                "budget": {"training": 6, "evaluation": .72, "smoke": .20, "total_ceiling": 6.92, "cash": 0},
                "controls": "Aware2x2(target/matcher changes); full control/both. Same starting weights within arm; same image sequence,2000updates,batch4,LR,finalEMA,no train-timeVALID. Stock bbox objectives preserved.",
                "selection": "No validation threshold search; all six cases scored with unchanged stock postprocessing. No TEST scoring. Aggregate AP50:95 plus AP75 and recall; report every outcome.",
                "limitations": "Original integration of Stable-DINO-inspired matching and RF-DETR existing position-target branch; not a full paper reproduction. Coverage is retained but improvements can also benefit full labels. One continuation seed from existing seed19parents.",
                "promotion": "Require≥1AP gain vs matched continued control to justify next-task replication; no promotion based only on a lower control or device variation.",
                "sources": ["https://arxiv.org/abs/2304.04742", "https://github.com/IDEA-Research/Stable-DINO",
                            "https://github.com/roboflow/rf-detr/tree/1.10.1"]}
    write_json(OUT / "protocol.json", protocol)
    print({"cases": len(cases), "protocol_sha256": file_digest(OUT / "protocol.json")})


if __name__ == "__main__":
    main()
