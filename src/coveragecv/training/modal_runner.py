"""Bounded multi-seed GPU experiments. Ephemeral workers; artifacts return locally."""
import hashlib
import io
import time
import zipfile
from pathlib import Path

import modal

app = modal.App("coveragecv-benchmark")
WORKSPACE = Path(__file__).resolve().parents[3]


def build_image():
    from coveragecv.training.cloud_budget import require_cloud_execution
    require_cloud_execution("coveragecv-benchmark")
    snapshot = WORKSPACE / "artifacts/modal/source/coveragecv"
    return (modal.Image.debian_slim(python_version="3.12")
            .apt_install("libgl1", "libglib2.0-0")
            .pip_install_from_requirements(str(WORKSPACE / "cloud-requirements.txt"))
            .add_local_dir(str(snapshot), "/opt/coveragecv", copy=True)
            .add_local_dir(str(WORKSPACE / "artifacts/modal/inputs"), "/input", copy=True)
            .add_local_file(str(Path.home() / ".cache/coveragecv/rf-detr-nano.pth"),
                            "/root/.cache/coveragecv/rf-detr-nano.pth", copy=True)
            .env({"PYTHONPATH": "/opt", "OMP_NUM_THREADS": "4", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}))


image = build_image() if modal.is_local() else modal.Image.debian_slim()


@app.function(image=image, gpu="L40S", cpu=(2, 4), memory=(8192, 24576), timeout=1800,
              max_containers=3, min_containers=0, scaledown_window=2, retries=0, include_source=False)
def gpu_experiment(dataset_id: str, arm: str, seed: int, steps: int):
    import math
    import tempfile

    import torch

    from coveragecv.artifacts import read_json, safe_child, verify, write_json
    from coveragecv.training.evaluate import evaluate_checkpoint
    from coveragecv.training.runner import create_initialization, run_training
    if arm not in ("naive", "aware", "complete_reference") or not 1 <= steps <= 3000:
        raise ValueError("experiment request is outside the approved recipe")
    if len(dataset_id) != 64 or any(c not in "0123456789abcdef" for c in dataset_id):
        raise ValueError("expected a frozen dataset SHA-256")
    payload = (Path("/input") / f"{dataset_id}.zip").read_bytes()
    if hashlib.sha256(payload).hexdigest() != dataset_id:
        raise ValueError("frozen dataset payload was modified")
    torch.set_num_threads(4)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            for name in archive.namelist():
                path = safe_child(root, name)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(archive.read(name))
        view = root / ("complete" if arm == "complete_reference" else "partial")
        verify(view)
        reference = root / "reference"
        verify(reference)
        output = root / "output"
        init = create_initialization(view, root / "initialization", seed=seed)
        batch = 4
        count = len(read_json(view / "train/_annotations.coco.json")["images"])
        epochs = math.ceil(steps/max(1, count//batch))
        started = time.monotonic()
        run = run_training(view, init, output, arm=arm, epochs=epochs, batch=batch, max_steps=steps,
                           device="cuda", seed=seed, timeout_seconds=1600)
        if run["status"] != "completed":
            raise RuntimeError("GPU run did not reach its requested update count")
        evaluation = evaluate_checkpoint(output / "detector.pt", reference, output / "evaluation.json")
        write_json(output / "compute.json", {"provider": "modal", "gpu": torch.cuda.get_device_name(),
            "elapsed_seconds": time.monotonic()-started, "persistent_volumes": False,
            "model_metrics": evaluation["metrics"]})
        packed = io.BytesIO()
        with zipfile.ZipFile(packed, "w", zipfile.ZIP_DEFLATED) as archive:
            for name in ("run.json", "evaluation.json", "compute.json", "progress.json", "detector.pt"):
                archive.write(output / name, name)
            archive.write(root / "initialization/initialization.json", "initialization.json")
        return packed.getvalue()


def pack_experiment(experiment: Path):
    from coveragecv.artifacts import read_json, verify
    ex = read_json(experiment)
    packed = io.BytesIO()
    with zipfile.ZipFile(packed, "w", zipfile.ZIP_DEFLATED) as archive:
        for key, prefix in (("partial_view", "partial"), ("complete_view", "complete"),
                            ("complete_bundle", "reference")):
            root = Path(ex[key])
            manifest = verify(root)
            for name in ("manifest.json", *manifest["files"]):
                archive.write(root / name, f"{prefix}/{name}")
    return packed.getvalue()


def run_batch(experiment: str = "artifacts/chess_experiment.json", output: str = "artifacts/gpu/pawns",
         steps: int = 2000, seeds: int = 3):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from coveragecv.artifacts import file_digest, read_json, safe_child, write_json
    if not 1 <= seeds <= 3 or not 1 <= steps <= 3000:
        raise ValueError("approved batch: up to 3 seeds, 3 arms, 3000 updates each")
    output_root = Path(output)
    output_root.mkdir(parents=True, exist_ok=True)
    payload = pack_experiment(Path(experiment))
    dataset_id = hashlib.sha256(payload).hexdigest()
    if not (WORKSPACE / "artifacts/modal/inputs" / f"{dataset_id}.zip").exists():
        raise ValueError("Freeze the input artifact and deploy this snapshot before launching its batch")
    remote = modal.Function.from_name("coveragecv-benchmark", "gpu_experiment")
    jobs = [(20260917+i, arm) for i in range(seeds) for arm in ("naive", "aware", "complete_reference")]
    # Upper bound uses the full function timeout, 4 CPU cores and 24 GiB RAM at published list rates.
    max_gross = len(jobs)*1800*(0.000542+4*0.0000131+24*0.00000222)
    write_json(output_root / "batch.json", {"steps": steps, "seeds": seeds, "jobs": len(jobs),
        "planned_max_runtime_resource_dollars": max_gross, "gpu": "L40S", "max_concurrent": 3,
        "per_call_timeout_seconds": 1800, "volumes": [], "cash_spend_limit": 0,
        "pricing_source": "https://modal.com/pricing", "source_experiment": str(Path(experiment).resolve())})

    def run(job):
        seed, arm = job
        folder = output_root / str(seed) / arm
        if (folder / "receipt.json").exists():
            receipt = read_json(folder / "receipt.json")
            saved = read_json(folder / "run.json")
            if (saved["steps"] != steps or saved["seed"] != seed or saved["arm"] != arm or
                    file_digest(folder / "detector.pt") != receipt["checkpoint_sha256"]):
                raise ValueError("existing receipt does not match the requested experiment")
            return {"seed": seed, "arm": arm, "status": "reused"}
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / "call.json").exists():
            submitted = read_json(folder / "call.json")
            if submitted["seed"] != seed or submitted["arm"] != arm or submitted["steps"] != steps:
                raise ValueError("existing submitted call has different parameters")
            call = modal.FunctionCall.from_id(submitted["call_id"])
        else:
            from coveragecv.training.cloud_budget import require_cloud_execution
            require_cloud_execution("coveragecv-benchmark")
            call = remote.spawn(dataset_id, arm, seed, steps)
            write_json(folder / "call.json", {"call_id": call.object_id, "seed": seed, "arm": arm,
                                              "steps": steps, "dataset_id": dataset_id})
        result = call.get()
        with zipfile.ZipFile(io.BytesIO(result)) as archive:
            for name in archive.namelist():
                path = safe_child(folder, name)
                path.write_bytes(archive.read(name))
        ledger = read_json(folder / "run.json")
        evaluation = read_json(folder / "evaluation.json")
        if ledger["arm"] != arm or ledger["steps"] != steps or ledger["status"] != "completed":
            raise ValueError("returned run does not match its submitted job")
        sha = file_digest(folder / "detector.pt")
        if ledger["detector_sha256"] != sha or evaluation["checkpoint_sha256"] != sha:
            raise ValueError("returned checkpoint and metrics are not bound")
        receipt = {"seed": seed, "arm": arm, "status": "completed", "checkpoint_sha256": sha}
        write_json(folder / "receipt.json", receipt)
        return receipt

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(run, job) for job in jobs]
        for future in as_completed(futures):
            print(future.result(), flush=True)


def freeze_input(experiment: str):
    payload = pack_experiment(Path(experiment))
    identity = hashlib.sha256(payload).hexdigest()
    folder = WORKSPACE / "artifacts/modal/inputs"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{identity}.zip").write_bytes(payload)
    return identity


@app.local_entrypoint()
def main(experiment: str = "artifacts/chess_experiment.json", output: str = "artifacts/gpu/pawns",
         steps: int = 2000, seeds: int = 3):
    run_batch(experiment, output, steps, seeds)
