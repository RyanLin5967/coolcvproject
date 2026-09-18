from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from coveragecv.artifacts import read_json, write_json
from coveragecv.training.cloud_budget import (
    require_cloud_execution,
    reserve_acquisition_run,
    reserve_capacity_run,
)


def test_credit_reservations_are_atomic_and_cannot_exceed_cohort(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = tmp_path / ".config/coveragecv"
    write_json(root / "cloud_lock.json", {"blocked": False, "allowed_app": "coveragecv-capacity",
        "allowed_new_runs": 3, "reservation_per_run_usd": 1.1, "new_compute_reservation_limit_usd": 3.5})
    with pytest.raises(RuntimeError, match="authorization"):
        require_cloud_execution("coveragecv-ablation")

    def reserve():
        try:
            return reserve_capacity_run()
        except RuntimeError:
            return None
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: reserve(), range(8)))
    assert len([x for x in results if x]) == 3
    saved = read_json(root / "capacity-reservations.json")
    assert len(saved) == 3 and sum(saved.values()) == pytest.approx(3.3)
    write_json(root / "cloud_lock.json", {"blocked": True})
    with pytest.raises(RuntimeError, match="authorization"):
        reserve_capacity_run()


def test_acquisition_exact_decimal_budget_and_separate_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = tmp_path / ".config/coveragecv"
    write_json(root / "cloud_lock.json", {"blocked": False, "allowed_apps": ["coveragecv-acquisition"],
        "acquisition_authorization": {"allowed_new_runs": 4, "reservation_per_run_usd": 1.10,
                                    "new_compute_reservation_limit_usd": 3.30}})
    assert len({reserve_acquisition_run() for _ in range(3)}) == 3
    with pytest.raises(RuntimeError, match="unreserved credit"):
        reserve_acquisition_run()
    assert len(read_json(root / "acquisition-reservations.json")) == 3
    assert not (root / "capacity-reservations.json").exists()
