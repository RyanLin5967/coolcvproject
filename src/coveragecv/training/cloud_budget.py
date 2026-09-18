"""Persistent local kill switch; a provider dashboard limit is not a billing guarantee."""
import fcntl
import uuid
from pathlib import Path

from coveragecv.artifacts import read_json, write_json


def require_cloud_execution(app_name):
    lock = Path.home() / ".config/coveragecv/cloud_lock.json"
    if lock.exists():
        config = read_json(lock)
        if config.get("blocked") or (config.get("allowed_app") and config["allowed_app"] != app_name):
            raise RuntimeError("This cloud app is outside the current credit authorization. Continue locally.")


def reserve_capacity_run():
    """Reserve worst-case cost before a new call; never release on a lagging billing read."""
    require_cloud_execution("coveragecv-capacity")
    root = Path.home() / ".config/coveragecv"
    config = read_json(root / "cloud_lock.json")
    with (root / "capacity-reservations.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root / "capacity-reservations.json"
        reservations = read_json(path) if path.exists() else {}
        cost = config["reservation_per_run_usd"]
        if (len(reservations) >= config["allowed_new_runs"] or
                sum(reservations.values())+cost > config["new_compute_reservation_limit_usd"]):
            raise RuntimeError("No unreserved credit remains for another capacity run")
        identity = uuid.uuid4().hex
        reservations[identity] = cost
        write_json(path, reservations)
        path.chmod(0o600)
        return identity
