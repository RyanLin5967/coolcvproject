"""Freeze a factorial resolution audit and a matched pawn inference comparison."""
import shutil
import zipfile
from pathlib import Path

from coveragecv.artifacts import file_digest, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/continuous/cross_resolution"


def main():
    if (OUT / "protocol.json").exists():
        raise ValueError("Cross-resolution protocol is already frozen")
    OUT.mkdir(parents=True, exist_ok=True)
    scale = ROOT / "artifacts/continuous/scale/runs"
    v2 = ROOT / "artifacts/research_v2/runs"
    parents = {"construction-parent-aware": v2 / "construction-exclusive",
               "construction-parent-full": v2 / "construction-full_power",
               "construction-aware-control": scale / "aware-control",
               "construction-aware-scale": scale / "aware-scale",
               "construction-full-control": scale / "complete-control",
               "construction-full-combined": scale / "complete-combined",
               "pawns-aware": ROOT / "artifacts/improved/pawns/20260919/aware_augmented_512",
               "pawns-full": ROOT / "artifacts/improved/pawns/20260919/complete_augmented_512"}
    checkpoints = {}
    with zipfile.ZipFile(OUT / "payload.zip", "w", zipfile.ZIP_STORED) as archive:
        for name, parent in parents.items():
            run = read_json(parent / "run.json")
            sha = file_digest(parent / "detector.pt")
            if run["status"] != "completed" or run["detector_sha256"] != sha:
                raise ValueError("Expected verified completed checkpoint")
            archive.write(parent / "detector.pt", f"checkpoints/{name}.pt")
            checkpoints[name] = {"checkpoint_sha256": sha, "local_parent": str(parent),
                                 "checkpoint_resolution": run["model_config"]["resolution"],
                                 "view_digest": run["view_digest"], "arm": run["arm"]}
    cases = {}
    for name in parents:
        task = "pawns" if name.startswith("pawns") else "construction"
        resolutions = (512, 640) if task == "pawns" else (
            (512,) if name in ("construction-aware-scale", "construction-full-combined") else (640,))
        for resolution in resolutions:
            cases[f"{name}-at{resolution}"] = {"checkpoint": name, "task": task, "resolution": resolution}
    references = read_json(ROOT / "artifacts/research_v2/evaluation_protocol.json")["references"]
    shutil.copy2(ROOT / "cloud-requirements.txt", OUT / "requirements.txt")
    source = OUT / "source/coveragecv"
    shutil.copytree(ROOT / "src/coveragecv", source, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    protocol = {"status": "frozen_before_evaluation", "cases": cases, "checkpoints": checkpoints,
                "references": {k: references[k] for k in ("construction", "pawns")},
                "source_files": {str(p.relative_to(source)): file_digest(p) for p in source.rglob("*") if p.is_file()},
                "payload_sha256": file_digest(OUT / "payload.zip"),
                "requirements_sha256": file_digest(OUT / "requirements.txt"),
                "selection": "Construction off-diagonal512/640 cells separate inference and training effects. Pawns compare512/640 on strongest prior aware seed19 and its same-seed full control. All outcomes retained; no threshold search.",
                "inference": "Only configured image resolution changes; original weights, positional parameterization, evaluator, class list, confidence scores and complete reference are preserved.",
                "budget": {"calls": len(cases), "reservation_per_call": .12, "maximum": 1.20, "cash": 0}}
    if len(cases) != 10:
        raise ValueError("Unexpected case count")
    write_json(OUT / "protocol.json", protocol)
    print({"cases": len(cases), "protocol_sha256": file_digest(OUT / "protocol.json")})


if __name__ == "__main__":
    main()
