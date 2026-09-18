"""Freeze three learner pairs, real parent checkpoints, and the twelve-run protocol."""
import shutil
import zipfile
from pathlib import Path

from coveragecv.artifacts import file_digest, read_json, verify, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/research_v2"


def main():
    if (OUT / "protocol.json").exists():
        raise ValueError("Research protocol already frozen")
    inputs = OUT / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    datasets = {}
    for task, filename in (("pawns", "chess_experiment.json"), ("all-pieces", "full_chess/experiment.json"),
                           ("construction", "construction/experiment.json")):
        spec = read_json(ROOT / "artifacts" / filename)
        if task == "construction":
            acquisition = read_json(ROOT / "artifacts/acquisition/protocol.json")
            partial = Path(acquisition["views"]["guided"]["path"])
            complete = Path(acquisition["views"]["complete_standard"]["path"])
            parents = {"aware": ROOT / "artifacts/acquisition/construction/20260917/guided",
                       "complete_reference": ROOT / "artifacts/acquisition/construction/20260917/complete_standard"}
        else:
            partial, complete = Path(spec["partial_view"]), Path(spec["complete_view"])
            if task == "pawns":
                parents = {"aware": ROOT / "artifacts/improved/pawns/20260917/aware_augmented_512",
                           "complete_reference": ROOT / "artifacts/improved/pawns/20260917/complete_augmented_512"}
            else:
                parents = {arm: ROOT / "artifacts/capacity/all-pieces/20260917" / f"{arm}_large_704_ema"
                           for arm in ("aware", "complete_reference")}
        classes = read_json(partial / "ontology.json")["classes"]
        groups = ([classes] if task == "pawns" else [[name for name in classes if name != "bishop"]]
                  if task == "all-pieces" else [["helmet", "no-helmet"], ["vest", "no-vest"]])
        if any(name not in classes for group in groups for name in group):
            raise ValueError("Invalid declared ontology group")
        records = {}
        payload = inputs / f"{task}.zip"
        with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as archive:
            for arm, view in (("aware", partial), ("complete_reference", complete)):
                manifest = verify(view)
                parent = parents[arm]
                run = read_json(parent / "run.json")
                sha = file_digest(parent / "detector.pt")
                if run["detector_sha256"] != sha or run["view_digest"] != manifest["digest"] or run["arm"] != arm:
                    raise ValueError("Parent model does not bind its declared learner")
                for name in ("manifest.json", *manifest["files"]):
                    archive.write(view / name, f"{arm}/view/{name}")
                for name in ("run.json", "detector.pt"):
                    archive.write(parent / name, f"{arm}/parent/{name}")
                records[arm] = {"view_digest": manifest["digest"], "checkpoint_sha256": sha,
                                "local_parent": str(parent), "local_view": str(view)}
        datasets[task] = {"payload_sha256": file_digest(payload), "classes": classes,
                          "exclusive_groups": groups, "parents": records,
                          "local_reference": spec["complete_bundle"]}
    cases = {}
    for task, data in datasets.items():
        for name, alpha, use_groups, arm in (("control", 1, False, "aware"),
                ("exclusive", 1, True, "aware"), ("exclusive_power", 3, True, "aware"),
                ("full_power", 3, False, "complete_reference")):
            cases[f"{task}-{name}"] = {"task": task, "arm": arm, "alpha": alpha,
                "exclusive_groups": data["exclusive_groups"] if use_groups else [], "steps": 2000, "seed": 20260918}
    source = OUT / "source/coveragecv"
    shutil.copytree(ROOT / "src/coveragecv", source, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    source_files = {str(p.relative_to(source)): file_digest(p) for p in source.rglob("*") if p.is_file()}
    protocol = {"status": "frozen_before_training", "datasets": datasets, "cases": cases,
                "source_files": source_files, "resources": {"gpu": "L40S", "timeout_seconds": 1600,
                "max_concurrency": 8, "persistent_volumes": False, "retries": 0},
                "budget": {"user_reported_credits": 20.31, "max_run_reservation": 1.2,
                           "max_calls": 12, "total_reserved_ceiling": 14.4, "cash_budget": 0},
                "selection": "Retain all outcomes; no validation during training; fixed final EMA checkpoint",
                "ontology_evidence": "Pawn colors are exclusive for one object. Chess piece/color identities "
                "are exclusive; generic bishop is excluded because it overlaps color-specific bishops. "
                "Helmet/no-helmet and vest/no-vest are exclusive attributes; person stays independent.",
                "construction_budget": "Guided learner has 265 previously acquired published train boxes; "
                "all aware variants have identical annotations. Full reference retains all published train boxes."}
    write_json(OUT / "protocol.json", protocol)
    print({"cases": len(cases), "total_payload_bytes": sum(p.stat().st_size for p in inputs.glob('*.zip')),
           "protocol_sha256": file_digest(OUT / "protocol.json")})


if __name__ == "__main__":
    main()
