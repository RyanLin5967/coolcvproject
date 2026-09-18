"""One bounded, human-only crop-refinement experiment with shared paired controls."""
import hashlib
import io
import os
import time
import zipfile
from pathlib import Path

import modal

app = modal.App("coveragecv-refinement")
ROOT = Path(__file__).resolve().parents[3]


def build_image():
    from coveragecv.training.cloud_budget import require_cloud_execution
    require_cloud_execution("coveragecv-refinement")
    return (modal.Image.debian_slim(python_version="3.12")
            .apt_install("libgl1", "libglib2.0-0")
            .pip_install_from_requirements(str(ROOT / "cloud-requirements.txt"))
            .add_local_dir(str(ROOT / os.environ.get("COVERAGECV_REFINER_SOURCE", "artifacts/modal/refinement-source/coveragecv")), "/opt/coveragecv", copy=True)
            .add_local_dir(str(ROOT / "artifacts/modal/inputs"), "/input", copy=True)
            .add_local_file(str(Path.home() / ".cache/torch/hub/checkpoints/resnet18-f37072fd.pth"),
                            "/root/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth", copy=True)
            .env({"PYTHONPATH": "/opt", "OMP_NUM_THREADS": "4"}))


image = build_image() if modal.is_local() else modal.Image.debian_slim()


@app.function(image=image, gpu="L40S", cpu=(4, 4), memory=(8192, 24576), timeout=1200,
              max_containers=1, min_containers=0, scaledown_window=2, retries=0, include_source=False)
def experiment(dataset_id: str, parents: list[dict]):
    import tempfile

    import torch

    from coveragecv.artifacts import safe_child, verify, write_json
    from coveragecv.training.refiner import train_refiner

    if dataset_id != "fb5c4fa70b4db2ec71e2bf321c0249f6bff60ad50efd4a782a117e4e0b9cd96b":
        raise ValueError("refinement is limited to the predeclared all-pieces dataset")
    expected = {"naive_augmented_512": "naive", "aware_augmented_512": "aware",
                "complete_augmented_512": "complete_reference"}
    if len(parents) != 3 or {p["method"] for p in parents} != set(expected):
        raise ValueError("the complete matched control set is required")
    torch.set_num_threads(4)
    started = time.monotonic()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        payload = (Path("/input") / f"{dataset_id}.zip").read_bytes()
        if hashlib.sha256(payload).hexdigest() != dataset_id:
            raise ValueError("dataset hash mismatch")
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            for name in archive.namelist():
                path = safe_child(root, name)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(archive.read(name))
        view = root / "partial"
        verify(view)
        output = root / "output"
        receipt = train_refiner(view, output, device="cuda", seed=20260917,
                                steps=1000, batch_size=32, max_seconds=950)
        write_json(output / "compute.json", {"status": "trained", "checkpoint_sha256": receipt["checkpoint_sha256"],
            "compute_seconds": time.monotonic()-started, "gpu": torch.cuda.get_device_name(),
            "evaluation": "Separate local stage after durable checkpoint collection",
            "parent_detectors": parents})
        packed = io.BytesIO()
        with zipfile.ZipFile(packed, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(output.iterdir()):
                if path.is_file():
                    archive.write(path, path.name)
        return packed.getvalue()
