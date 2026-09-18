"""Bounded Large/704px/EMA experiment, separately versioned from running Nano jobs."""
import hashlib
import io
import math
import time
import zipfile
from pathlib import Path

import modal

app = modal.App("coveragecv-capacity")
ROOT = Path(__file__).resolve().parents[3]


def build_image():
    from coveragecv.training.cloud_budget import require_cloud_execution
    require_cloud_execution("coveragecv-capacity")
    return (modal.Image.debian_slim(python_version="3.12")
            .apt_install("libgl1", "libglib2.0-0")
            .pip_install_from_requirements(str(ROOT / "cloud-requirements.txt"))
            .add_local_dir(str(ROOT / "artifacts/modal/capacity-source/coveragecv"), "/opt/coveragecv", copy=True)
            .add_local_dir(str(ROOT / "artifacts/modal/inputs"), "/input", copy=True)
            .add_local_file(str(Path.home() / ".cache/coveragecv/rf-detr-large-2026.pth"),
                            "/root/.cache/coveragecv/rf-detr-large-2026.pth", copy=True)
            .env({"PYTHONPATH": "/opt", "OMP_NUM_THREADS": "4", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}))


image = build_image() if modal.is_local() else modal.Image.debian_slim()


@app.function(image=image, gpu="L40S", cpu=(4, 4), memory=(8192, 24576), timeout=1500,
              max_containers=3, min_containers=0, scaledown_window=2, retries=0, include_source=False)
def experiment(dataset_id: str, arm: str, seed: int = 20260917, steps: int = 2000):
    import tempfile

    import torch

    from coveragecv.artifacts import read_json, safe_child, verify, write_json
    from coveragecv.training.evaluate import evaluate_checkpoint
    from coveragecv.training.runner import create_initialization, run_training

    if arm not in ("aware", "naive", "complete_reference") or steps != 2000 or seed != 20260917:
        raise ValueError("outside the predeclared capacity protocol")
    if len(dataset_id) != 64 or any(c not in "0123456789abcdef" for c in dataset_id):
        raise ValueError("expected immutable dataset SHA-256")
    torch.set_num_threads(4)
    started = time.monotonic()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        payload = (Path("/input") / f"{dataset_id}.zip").read_bytes()
        if hashlib.sha256(payload).hexdigest() != dataset_id:
            raise ValueError("dataset archive hash mismatch")
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            for name in archive.namelist():
                path = safe_child(root, name)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(archive.read(name))
        view = root / ("complete" if arm == "complete_reference" else "partial")
        verify(view)
        initial = create_initialization(view, root / "initialization", seed=seed, variant="large")
        count = len(read_json(view / "train/_annotations.coco.json")["images"])
        output = root / "output"
        run = run_training(view, initial, output, arm=arm, recipe="large_fresh", seed=seed,
                           epochs=math.ceil(steps/max(1, count//4)), batch=4,
                           max_steps=steps, device="cuda", timeout_seconds=1250)
        if run["status"] != "completed":
            raise RuntimeError("Large run did not complete the requested updates")
        run.update(method=f"{arm}_large_704_ema")
        write_json(output / "run.json", run)
        # Use the same CPU evaluator as existing experiments to preserve the evaluation protocol.
        evaluation = evaluate_checkpoint(output / "detector.pt", root / "reference", output / "evaluation.json")
        write_json(output / "compute.json", {"provider": "modal", "gpu": torch.cuda.get_device_name(),
            "elapsed_seconds": time.monotonic()-started, "persistent_volumes": False,
            "model_metrics": evaluation["metrics"]})
        packed = io.BytesIO()
        with zipfile.ZipFile(packed, "w", zipfile.ZIP_DEFLATED) as archive:
            for name in ("run.json", "evaluation.json", "compute.json", "progress.json", "detector.pt"):
                archive.write(output / name, name)
            archive.write(root / "initialization/initialization.json", "initialization.json")
        return packed.getvalue()
