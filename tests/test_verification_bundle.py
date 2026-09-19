"""The published evidence bundle must keep reproducing the numbers the site shows.

These checks read only committed files, so they run without artifacts, credentials,
a GPU or any model code.
"""
import json
import struct
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/coveragecv/workbench/static/verify"
PUBLIC = ROOT / "public-demo/static/verify"
MANIFEST = json.loads((SOURCE / "manifest.json").read_text())
RUNS = [(cohort, run) for cohort in MANIFEST["cohorts"] for run in cohort["runs"]]


def test_bundle_is_not_empty():
    assert MANIFEST["cohorts"], "an empty evidence bundle is not a passing state"
    assert len(RUNS) >= 12, f"only {len(RUNS)} verifiable runs are published"


@pytest.mark.parametrize("cohort,run", RUNS, ids=[run["id"] for _, run in RUNS])
def test_committed_predictions_match_their_digest(cohort, run):
    import hashlib
    blob = (SOURCE / run["predictions"]["path"]).read_bytes()
    assert hashlib.sha256(blob).hexdigest() == run["predictions"]["sha256"]
    assert len(blob) == run["predictions"]["bytes"]
    magic, version, count, _ = struct.unpack("<4sIII", blob[:16])
    assert magic == b"CVB1" and version == 1
    assert count == run["predictions"]["count"]
    assert len(blob) == 16 + 28 * count, "header and array lengths disagree"


@pytest.mark.parametrize("cohort,run", RUNS, ids=[run["id"] for _, run in RUNS])
def test_every_run_carries_a_published_expectation(cohort, run):
    expected = run["expected"]
    assert 0 <= expected["AP"] <= 1 and 0 <= expected["AP50"] <= 1
    assert run["bundle_digest"] in MANIFEST["ground_truth"]
    assert len(run["checkpoint_sha256"]) == 64
    assert run["role"] in {"naive", "aware", "complete_reference"}
    # The reduction claim only means something if it is stated per run.
    assert run["detections_saved"] >= run["predictions"]["count"]


def test_reference_labels_are_committed_and_hashed():
    import hashlib
    for entry in MANIFEST["ground_truth"].values():
        body = (SOURCE / entry["path"]).read_bytes()
        assert hashlib.sha256(body).hexdigest() == entry["sha256"]
        labels = json.loads(body)
        assert len(labels["images"]) == entry["images"]
        assert len(labels["annotations"]) == entry["boxes"]
        # The browser evaluator refuses crowd/ignore boxes, so none may appear here.
        assert not any(a.get("iscrowd") or a.get("ignore") for a in labels["annotations"])


def test_public_demo_publishes_the_same_bytes():
    assert (PUBLIC / "manifest.json").read_bytes() == (SOURCE / "manifest.json").read_bytes()
    for cohort, run in RUNS:
        name = run["predictions"]["path"]
        assert (PUBLIC / name).read_bytes() == (SOURCE / name).read_bytes()
    for entry in MANIFEST["ground_truth"].values():
        assert (PUBLIC / entry["path"]).read_bytes() == (SOURCE / entry["path"]).read_bytes()


def test_site_headline_scores_are_verifiable():
    """The scores the Benchmarks page shows must be among the recomputable ones."""
    snapshot = json.loads((ROOT / "public-demo/data/snapshot.json").read_text())
    published = {round(run["metrics"]["AP"], 12)
                 for task in snapshot["routes"]["/research"]["tasks"].values()
                 for run in task.get("runs", [])}
    verifiable = {round(run["expected"]["AP"], 12) for _, run in RUNS}
    assert verifiable <= published, "the bundle claims a score the results ledger does not contain"


def test_browser_evaluator_matches_python_exactly():
    """The page's JavaScript COCO evaluator must reproduce pycocotools bit for bit.

    Without this, the site could agree with itself while both halves were wrong.
    """
    import shutil
    import subprocess
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed; run scripts/check_verification_parity.mjs to check parity")
    result = subprocess.run([node, str(ROOT / "scripts/check_verification_parity.mjs")],
                            capture_output=True, text=True, timeout=600, cwd=ROOT, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"{len(RUNS)}/{len(RUNS)} runs reproduced exactly" in result.stdout, result.stdout
    assert "Largest deviation across every compared field: 0 " in result.stdout, result.stdout


def test_published_assets_are_not_stale():
    """public-demo must carry the current shared UI, not a previous build of it.

    Editing a file under workbench/static and forgetting to rebuild silently ships an
    older page, which is how a verified local fix reaches visitors as the old bug.
    """
    source = ROOT / "src/coveragecv/workbench/static"
    published = ROOT / "public-demo/static"
    names = ("app.js", "benchmark-view.js", "merge-story.js", "verify-view.js", "coco-eval.js",
             "style.css", "merge-story.css", "verify.css")
    stale = [name for name in names
             if (source / name).read_bytes() != (published / name).read_bytes()]
    assert not stale, f"run scripts/build_public_demo.py; stale: {', '.join(stale)}"
