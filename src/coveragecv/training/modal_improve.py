"""Stage-two controlled experiments. Reuse cloud checkpoints without reuploading."""
import hashlib
import io
import math
import os
import time
import zipfile
from pathlib import Path

import modal

APP_NAME = os.environ.get("COVERAGECV_MODAL_APP", "coveragecv-improvement")
app = modal.App(APP_NAME)
WORKSPACE = Path(__file__).resolve().parents[3]
METHODS = {
    "aware_continue_384": ("aware", "continued", False),
    "aware_augmented_512": ("aware", "augmented", False),
    "aware_teacher_512": ("aware", "augmented", True),
    "naive_augmented_512": ("naive", "augmented", False),
    "complete_augmented_512": ("complete_reference", "augmented", False),
}
ABLATIONS = {"aware_teacher_cautious_512": ("aware", "augmented", True),
             "aware_teacher_classonly_512": ("aware", "augmented", True)}


def build_image():
    from coveragecv.training.cloud_budget import require_cloud_execution
    require_cloud_execution(APP_NAME)
    return (modal.Image.debian_slim(python_version="3.12")
            .apt_install("libgl1", "libglib2.0-0")
            .pip_install_from_requirements(str(WORKSPACE / "cloud-requirements.txt"))
            .add_local_dir(str(WORKSPACE / os.environ.get("COVERAGECV_MODAL_SOURCE", "artifacts/modal/improvement-source/coveragecv")),
                           "/opt/coveragecv", copy=True)
            .add_local_dir(str(WORKSPACE / "artifacts/modal/inputs"), "/input", copy=True)
            .add_local_file(str(Path.home() / ".cache/coveragecv/rf-detr-nano.pth"),
                            "/root/.cache/coveragecv/rf-detr-nano.pth", copy=True)
            .env({"PYTHONPATH": "/opt", "OMP_NUM_THREADS": "4", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}))


image = build_image() if modal.is_local() else modal.Image.debian_slim()


@app.function(image=image, gpu="L40S", cpu=(4, 4), memory=(8192, 24576), timeout=1800,
              max_containers=6, min_containers=0, scaledown_window=2, retries=0, include_source=False)
def fresh_experiment(dataset_id: str, arm: str, seed: int, steps: int = 4000):
    import tempfile

    import torch

    from coveragecv.artifacts import read_json, safe_child, verify, write_json
    from coveragecv.training.evaluate import evaluate_checkpoint
    from coveragecv.training.runner import create_initialization, run_training
    if arm not in ("naive", "aware", "complete_reference") or not 1 <= steps <= 4000:
        raise ValueError("request outside independent-domain experiment protocol")
    if len(dataset_id) != 64 or any(c not in "0123456789abcdef" for c in dataset_id):
        raise ValueError("expected frozen dataset SHA-256")
    torch.set_num_threads(4)
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
        view = root / ("complete" if arm == "complete_reference" else "partial")
        verify(view)
        initialization = create_initialization(view, root / "initialization", seed=seed)
        output = root / "output"
        batch = 4
        count = len(read_json(view / "train/_annotations.coco.json")["images"])
        started = time.monotonic()
        run = run_training(view, initialization, output, arm=arm, recipe="augmented_fresh", seed=seed,
                           epochs=math.ceil(steps/max(1,count//batch)), batch=batch,
                           max_steps=steps, device="cuda", timeout_seconds=1600)
        if run["status"] != "completed":
            raise RuntimeError("independent-domain run did not complete its update budget")
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


@app.function(image=image, cpu=2, memory=4096, timeout=300, max_containers=1,
              min_containers=0, scaledown_window=2, retries=0, include_source=False)
def hosted_export(parent_call: str, expected_sha: str, upload_url: str):
    """Transfer a class-correct export cloud-to-cloud using a short-lived signed URL."""
    import tempfile

    import requests

    from coveragecv.artifacts import file_digest
    from coveragecv.training.deploy import export_for_hosted
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        returned = modal.FunctionCall.from_id(parent_call).get(timeout=120)
        with zipfile.ZipFile(io.BytesIO(returned)) as archive:
            checkpoint = root / "detector.pt"
            checkpoint.write_bytes(archive.read("detector.pt"))
        if file_digest(checkpoint) != expected_sha:
            raise ValueError("checkpoint hash mismatch")
        output = root / "export"
        metadata = export_for_hosted(checkpoint, output)
        archive_path = root / "roboflow_deploy.zip"
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for name in ("weights.pt", "class_names.txt"):
                archive.write(output / name, name)
        # Never log the signed URL or an exception string that might contain it.
        try:
            with archive_path.open("rb") as stream:
                response = requests.put(upload_url, data=stream, timeout=(15, 120))
        except requests.RequestException as exc:
            raise RuntimeError(f"Cloud upload failed: {type(exc).__name__}") from None
        if not 200 <= response.status_code < 300:
            raise RuntimeError(f"Cloud upload HTTP {response.status_code}")
        return {"status": "uploaded", "http_status": response.status_code,
                "archive_sha256": file_digest(archive_path), "bytes": archive_path.stat().st_size,
                "export": metadata}


@app.function(image=image, gpu="L40S", cpu=(2, 4), memory=(8192, 24576), timeout=1200,
              max_containers=6, min_containers=0, scaledown_window=2, retries=0, include_source=False)
def improve(dataset_id: str, parent_call: str, parent_sha: str, method: str, seed: int, steps: int = 2000):
    import tempfile

    import torch

    from coveragecv.artifacts import file_digest, read_json, safe_child, verify, write_json
    from coveragecv.training.evaluate import evaluate_checkpoint
    from coveragecv.training.pseudo import mine_training_labels
    from coveragecv.training.runner import create_initialization, run_training

    recipes = METHODS | ABLATIONS
    if method not in recipes or not 1 <= steps <= 2000:
        raise ValueError("request is outside the declared improvement protocol")
    if len(dataset_id) != 64 or any(c not in "0123456789abcdef" for c in dataset_id):
        raise ValueError("expected frozen dataset SHA-256")
    torch.set_num_threads(4)
    arm, recipe, use_teacher = recipes[method]
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
        parent = root / "parent"
        parent.mkdir()
        returned = modal.FunctionCall.from_id(parent_call).get(timeout=120)
        with zipfile.ZipFile(io.BytesIO(returned)) as archive:
            for name in ("detector.pt", "run.json"):
                (parent / name).write_bytes(archive.read(name))
        del returned
        parent_run = read_json(parent / "run.json")
        checkpoint = parent / "detector.pt"
        original_view = root / ("complete" if arm == "complete_reference" else "partial")
        original_manifest = verify(original_view)
        if (file_digest(checkpoint) != parent_sha or parent_run["detector_sha256"] != parent_sha
                or parent_run["status"] != "completed" or parent_run["arm"] != arm
                or parent_run["seed"] != seed or parent_run["steps"] != 2000
                or parent_run["view_digest"] != original_manifest["digest"]):
            raise ValueError("parent checkpoint does not match the declared stage-one experiment")
        view = original_view
        if use_teacher:
            view = mine_training_labels(view, checkpoint, root / "pseudo-views", device="cuda")
            torch.cuda.empty_cache()
        initialization = create_initialization(view, root / "initialization", seed=seed)
        batch = 4
        count = len(read_json(view / "train/_annotations.coco.json")["images"])
        output = root / "output"
        run = run_training(view, initialization, output, arm=arm, recipe=recipe, warm_start=checkpoint,
                           max_steps=steps, epochs=math.ceil(steps/max(1,count//batch)), batch=batch,
                           device="cuda", seed=seed, timeout_seconds=950,
                           pseudo_box_weight={"aware_teacher_cautious_512": .1,
                                              "aware_teacher_classonly_512": 0.}.get(method, 1.))
        if run["status"] != "completed":
            raise RuntimeError("improvement run did not complete its requested updates")
        run.update(method=method, stage1_steps=parent_run["steps"], total_training_steps=parent_run["steps"]+run["steps"],
                   parent_call_id=parent_call, original_view_digest=original_manifest["digest"],
                   teacher_mining=use_teacher, pretrained_parameter_digest=parent_run["initial_parameter_digest"])
        write_json(output / "run.json", run)
        evaluation = evaluate_checkpoint(output / "detector.pt", root / "reference", output / "evaluation.json")
        write_json(output / "compute.json", {"provider": "modal", "gpu": torch.cuda.get_device_name(),
                   "elapsed_seconds": time.monotonic()-started, "persistent_volumes": False,
                   "model_metrics": evaluation["metrics"], "method": method})
        write_json(output / "learner_manifest.json", verify(view))
        if use_teacher:
            write_json(output / "pseudo-labels.json", read_json(view / "pseudo-labels.json"))
            write_json(output / "derived-train.coco.json", read_json(view / "train/_annotations.coco.json"))
        packed = io.BytesIO()
        with zipfile.ZipFile(packed, "w", zipfile.ZIP_DEFLATED) as archive:
            for name in ("run.json", "evaluation.json", "compute.json", "progress.json", "detector.pt",
                         "learner_manifest.json", "pseudo-labels.json", "derived-train.coco.json"):
                if (output / name).exists():
                    archive.write(output / name, name)
            archive.write(parent / "run.json", "parent-run.json")
        return packed.getvalue()


def run_improvement_batch(task="pawns", *, seeds=(20260917, 20260918, 20260919), methods=None,
                          app_name="coveragecv-improvement"):
    """Idempotent submission, reattachment and validated local artifact collection."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from coveragecv.artifacts import file_digest, read_json, safe_child, write_json

    if task not in ("pawns", "all-pieces"):
        raise ValueError("unknown frozen task")
    methods = list(METHODS) if methods is None else list(methods)
    recipes = METHODS | ABLATIONS
    if set(methods)-set(recipes) or set(seeds)-{20260917, 20260918, 20260919}:
        raise ValueError("unsupported experimental matrix")
    dataset_id = {"pawns": "d7cf01e2c897dd2ef67b2ed658e56ab371c41d248d5ca4e6341293f00f4e2da7",
                  "all-pieces": "fb5c4fa70b4db2ec71e2bf321c0249f6bff60ad50efd4a782a117e4e0b9cd96b"}[task]
    remote = modal.Function.from_name(app_name, "improve")
    base = WORKSPACE / "artifacts/gpu" / task
    root = WORKSPACE / "artifacts/improved" / task

    def execute(seed, method):
        arm = recipes[method][0]
        parent = base / str(seed) / arm
        if not (parent / "receipt.json").exists():
            return {"seed": seed, "method": method, "status": "waiting_for_stage_one"}
        receipt = read_json(parent / "receipt.json")
        call_id = read_json(parent / "call.json")["call_id"]
        folder = root / str(seed) / method
        folder.mkdir(parents=True, exist_ok=True)
        request = {"dataset_id": dataset_id, "parent_call": call_id, "parent_sha": receipt["checkpoint_sha256"],
                   "method": method, "seed": seed, "steps": 2000}
        if (folder / "receipt.json").exists():
            saved = read_json(folder / "run.json")
            if (saved["method"] != method or saved["seed"] != seed or saved["steps"] != 2000
                    or saved["warm_start_sha256"] != request["parent_sha"]
                    or file_digest(folder / "detector.pt") != saved["detector_sha256"]):
                raise ValueError("saved improvement does not match its request")
            return {"seed": seed, "method": method, "status": "reused"}
        if (folder / "call.json").exists():
            submitted = read_json(folder / "call.json")
            if submitted["request"] != request:
                raise ValueError("existing submitted improvement has different parameters")
            call = modal.FunctionCall.from_id(submitted["call_id"])
        else:
            from coveragecv.training.cloud_budget import require_cloud_execution
            require_cloud_execution(app_name)
            call = remote.spawn(**request)
            write_json(folder / "call.json", {"call_id": call.object_id, "request": request})
        result = call.get()
        with zipfile.ZipFile(io.BytesIO(result)) as archive:
            for name in archive.namelist():
                safe_child(folder, name).write_bytes(archive.read(name))
        saved = read_json(folder / "run.json")
        evaluation = read_json(folder / "evaluation.json")
        sha = file_digest(folder / "detector.pt")
        if (saved["method"] != method or saved["seed"] != seed or saved["steps"] != 2000
                or saved["status"] != "completed" or saved["warm_start_sha256"] != request["parent_sha"]
                or saved["detector_sha256"] != sha or evaluation["checkpoint_sha256"] != sha):
            raise ValueError("returned experiment does not match the submitted request")
        result = {"seed": seed, "method": method, "status": "completed", "checkpoint_sha256": sha,
                  "AP": evaluation["metrics"]["AP"]}
        write_json(folder / "receipt.json", result)
        return result

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(execute, seed, method) for seed in seeds for method in methods]
        for future in as_completed(futures):
            print(future.result(), flush=True)
