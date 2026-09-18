"""Persistent local kill switch; a provider dashboard limit is not a billing guarantee."""
import fcntl
import uuid
from decimal import Decimal
from pathlib import Path

from coveragecv.artifacts import read_json, write_json


def require_cloud_execution(app_name):
    lock = Path.home() / ".config/coveragecv/cloud_lock.json"
    if lock.exists():
        config = read_json(lock)
        allowed = config.get("allowed_apps", [config["allowed_app"]] if config.get("allowed_app") else None)
        if config.get("blocked") or (allowed is not None and app_name not in allowed):
            raise RuntimeError("This cloud app is outside the current credit authorization. Do not launch compute.")


def reserve_capacity_run():
    """Reserve worst-case cost before a new call; never release on a lagging billing read."""
    return _reserve("coveragecv-capacity", "capacity", None)


def reserve_refinement_run():
    return _reserve("coveragecv-refinement", "refinement", "refiner_authorization")


def reserve_object_crop_run():
    return _reserve("coveragecv-object-crops", "object-crops", "object_crop_authorization")


def reserve_acquisition_run():
    return _reserve("coveragecv-acquisition", "acquisition", "acquisition_authorization")


def reserve_research_v2_run():
    return _reserve("coveragecv-research-v2", "research-v2", "research_v2_authorization")


def reserve_research_evaluation():
    return _reserve("coveragecv-research-evaluation", "research-evaluation", "research_evaluation_authorization")


def reserve_research_tiling():
    return _reserve("coveragecv-research-tiling", "research-tiling", "research_tiling_authorization")


def reserve_segment_refinement():
    return _reserve("coveragecv-segment-refinement", "segment-refinement", "segment_refinement_authorization")


def reserve_continuous_scale(stage):
    if stage not in ("train", "evaluate", "smoke"):
        raise ValueError("Unknown continuous scale stage")
    return _reserve("coveragecv-continuous-scale", f"continuous-scale-{stage}", f"continuous_scale_{stage}")


def _reserve(app_name, ledger_name, section):
    require_cloud_execution(app_name)
    root = Path.home() / ".config/coveragecv"
    config = read_json(root / "cloud_lock.json")
    if section:
        config = config[section]
    with (root / f"{ledger_name}-reservations.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root / f"{ledger_name}-reservations.json"
        reservations = read_json(path) if path.exists() else {}
        cost = config["reservation_per_run_usd"]
        committed = sum((Decimal(str(value)) for value in reservations.values()), Decimal(0))
        if (len(reservations) >= config["allowed_new_runs"] or
                committed+Decimal(str(cost)) > Decimal(str(config["new_compute_reservation_limit_usd"]))):
            raise RuntimeError(f"No unreserved credit remains for another {ledger_name} run")
        identity = uuid.uuid4().hex
        reservations[identity] = cost
        write_json(path, reservations)
        path.chmod(0o600)
        return identity
