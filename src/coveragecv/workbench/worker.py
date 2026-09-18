"""One isolated, restart-surviving local job. No shell and no cloud execution."""
import math
import os
import signal
import sys
from pathlib import Path

from coveragecv.artifacts import read_json, verify, write_json
from coveragecv.compiler import compile_bundle, materialize_view
from coveragecv.schema import DiagnosticError


class Cancelled(BaseException):
    pass


def local_hardware():
    """Probe local acceleration without creating a model or a device tensor."""
    try:
        import torch
    except (ImportError, OSError):
        return {"devices": {"cpu": True, "mps": False},
                "mps_unavailable_reason": "The training dependencies are not installed on this machine."}
    available = torch.backends.mps.is_built() and torch.backends.mps.is_available()
    return {"devices": {"cpu": True, "mps": available},
            "mps_unavailable_reason": None if available else
                "Apple GPU acceleration (MPS) is not available on this machine. Choose CPU."}


def validate_local_device(device):
    if device not in ("cpu", "mps"):
        raise ValueError("The local workbench supports CPU or Apple GPU (MPS) only.")
    if device == "mps":
        hardware = local_hardware()
        if not hardware["devices"]["mps"]:
            raise ValueError(hardware["mps_unavailable_reason"])


def execute(job, root, output):
    payload = job["payload"]
    if job["kind"] == "compile":
        spec = payload["spec"]
        # Freeze absolute source locations; never resolve against the worker's CWD.
        for source in spec["sources"]:
            for key in ("annotations", "images"):
                source[key] = str((Path(payload["base_dir"]) / source[key]).resolve())
        spec_path = output / "spec.json"
        write_json(spec_path, spec)
        bundle = compile_bundle(spec_path, root / "bundles")
        view = materialize_view(bundle, root / "views")
        return {"bundle": str(bundle), "view": str(view), "manifest": verify(bundle),
                "diagnostics": read_json(bundle / "diagnostics.json")}
    if job["kind"] != "train":
        raise ValueError("unsupported job type")
    device = payload.get("device", "cpu")
    validate_local_device(device)
    recipe = payload.get("recipe", "pilot")
    variant = "large" if recipe == "large_fresh" else "nano"
    from coveragecv.training.evaluate import evaluate_checkpoint
    from coveragecv.training.runner import create_initialization, run_training
    view, reference = Path(payload["view"]), Path(payload["reference"])
    manifest = verify(view)
    if manifest["digest"] != payload["view_digest"]:
        raise ValueError("queued dataset view no longer matches its immutable identity")
    initialization = create_initialization(view, root / "initializations" /
                                           f"{manifest['ontology_digest']}-{variant}-{payload['seed']}",
                                           seed=payload["seed"], variant=variant)
    train_images = len(read_json(view / "train/_annotations.coco.json")["images"])
    epochs = math.ceil(payload["steps"] / max(1, train_images // payload["batch"]))
    arms = {}
    for arm in payload["arms"]:
        folder = output / arm
        selected_view = Path(payload["complete_view"]) if arm == "complete_reference" else view
        write_json(output / "live.json", {"phase": "training", "arm": arm, "completed_arms": list(arms)})
        run = run_training(selected_view, initialization, folder, arm=arm, epochs=epochs,
                           max_steps=payload["steps"], batch=payload["batch"], seed=payload["seed"],
                           timeout_seconds=payload["timeout_seconds"], recipe=recipe, device=device)
        if run["status"] != "completed":
            raise ValueError("training reached its local time limit before completing the requested updates")
        write_json(output / "live.json", {"phase": "evaluating", "arm": arm, "completed_arms": list(arms)})
        evaluation = evaluate_checkpoint(folder / "detector.pt", reference, folder / "evaluation.json")
        arms[arm] = {"run": run, "metrics": evaluation["metrics"], "evaluation_path": str(folder / "evaluation.json"),
                     "checkpoint_path": str(folder / "detector.pt")}
        write_json(output / "partial_result.json", {"arms": arms})
    return {"arms": arms, "reference_bundle": str(reference), "view_digest": manifest["digest"]}


def main(job_path: Path):
    job = read_json(job_path)
    root, output = Path(job["root"]), job_path.parent
    from .store import Store
    Store(root).set_pid(job["id"], os.getpid())

    def cancel(_signum, _frame):
        raise Cancelled()

    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)
    try:
        result = {"status": "completed", "result": execute(job, root, output)}
    except Cancelled:
        result = {"status": "cancelled", "error": {"message": "Cancelled by the user; completed artifacts remain."}}
    except DiagnosticError as exc:
        result = {"status": "failed", "error": exc.as_dict()}
    except Exception as exc:  # noqa: BLE001 -- worker failures are persisted, never converted to successes
        result = {"status": "failed", "error": {"code": type(exc).__name__, "message": str(exc)}}
    write_json(output / "result.json", result)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
