"""Bounded matched continuations after simulated annotation acquisition."""
import io
import json
import math
import time
import zipfile
from pathlib import Path

import modal

from coveragecv.artifacts import file_digest, read_json, safe_child, verify, write_json

ROOT = Path(__file__).resolve().parents[3]
CASES = {"guided": "aware", "random": "aware", "complete_standard": "complete_reference"}
app = modal.App("coveragecv-acquisition")


def build_image():
    from coveragecv.training.cloud_budget import require_cloud_execution
    require_cloud_execution("coveragecv-acquisition")
    return (modal.Image.debian_slim(python_version="3.12")
            .apt_install("libgl1", "libglib2.0-0")
            .pip_install_from_requirements(str(ROOT / "cloud-requirements.txt"))
            .add_local_dir(str(ROOT / "artifacts/modal/acquisition-source/coveragecv"), "/opt/coveragecv", copy=True)
            .add_local_file(str(ROOT / "artifacts/acquisition/input.zip"), "/input/dataset.zip", copy=True)
            .add_local_file(str(ROOT / "artifacts/acquisition/protocol.json"), "/input/protocol.json", copy=True)
            .add_local_file(str(Path.home() / ".cache/coveragecv/rf-detr-nano.pth"),
                            "/root/.cache/coveragecv/rf-detr-nano.pth", copy=True)
            .env({"PYTHONPATH": "/opt", "OMP_NUM_THREADS": "4", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}))


image = build_image() if modal.is_local() else modal.Image.debian_slim()


@app.function(image=image, gpu="L40S", cpu=(4, 4), memory=(8192, 24576), timeout=1500,
              max_containers=3, min_containers=0, scaledown_window=2, retries=0, include_source=False)
def experiment(case: str):
    import tempfile

    import torch

    from coveragecv.training.runner import create_initialization, run_training

    if case not in CASES:
        raise ValueError("case is outside the fixed acquisition comparison")
    protocol = read_json(Path("/input/protocol.json"))
    arm = CASES[case]
    parent = protocol["parents"][arm]
    torch.set_num_threads(4)
    started = time.monotonic()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        if file_digest(Path("/input/dataset.zip")) != protocol["dataset_sha256"]:
            raise ValueError("dataset archive identity changed")
        with zipfile.ZipFile("/input/dataset.zip") as archive:
            for name in archive.namelist():
                target = safe_child(root, name)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(name))
        # Reconnect to a completed parent; this does not rerun parent training.
        returned = modal.FunctionCall.from_id(parent["call_id"]).get()
        with zipfile.ZipFile(io.BytesIO(returned)) as archive:
            warm = root / "parent.pt"
            warm.write_bytes(archive.read("detector.pt"))
            previous = json.loads(archive.read("run.json"))
        if (file_digest(warm) != parent["checkpoint_sha256"] or previous["steps"] != 4000
                or previous["status"] != "completed" or previous["arm"] != arm
                or previous["seed"] != 20260917 or previous["recipe"] != "augmented_fresh"):
            raise ValueError("warm-start checkpoint differs from the fixed parent contract")
        view = root / case
        manifest = verify(view)
        if manifest["digest"] != protocol["views"][case]["digest"]:
            raise ValueError("learner view differs from the frozen acquisition protocol")
        initialization = create_initialization(view, root / "initialization", seed=20260917)
        count = len(read_json(view / "train/_annotations.coco.json")["images"])
        output = root / "output"
        run = run_training(view, initialization, output, arm=arm, recipe="augmented", seed=20260917,
                           epochs=math.ceil(2000/max(1, count//4)), batch=4, max_steps=2000,
                           device="cuda", timeout_seconds=1250, warm_start=warm,
                           validation_during_training=False)
        if run["status"] != "completed":
            raise RuntimeError("acquisition comparison did not complete all requested updates")
        run.update(method=case, total_training_steps=6000, parent_checkpoint_sha256=parent["checkpoint_sha256"],
                   experiment_protocol_sha256=file_digest(Path("/input/protocol.json")),
                   annotation_acquisition=protocol["views"][case].get("acquisition"))
        write_json(output / "run.json", run)
        write_json(output / "compute.json", {"status": "trained", "gpu": torch.cuda.get_device_name(),
            "compute_seconds": time.monotonic()-started, "evaluation": "Separate local stage after checkpoint collection"})
        packed = io.BytesIO()
        with zipfile.ZipFile(packed, "w", zipfile.ZIP_DEFLATED) as archive:
            for name in ("run.json", "detector.pt", "progress.json", "compute.json"):
                archive.write(output / name, name)
        return packed.getvalue()
