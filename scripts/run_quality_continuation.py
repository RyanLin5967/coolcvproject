"""Fixed-budget local continuation with a strict-localization loss and matched controls."""
import argparse
import math
import time
from pathlib import Path

import torch
from pytorch_lightning import seed_everything
from rfdetr.training import RFDETRDataModule, build_trainer
from rfdetr.training.callbacks.ema import RFDETREMACallback

from coveragecv.artifacts import file_digest, read_json, verify, write_json
from coveragecv.training.evaluate import evaluate_checkpoint
from coveragecv.training.power_localization import with_power_localization
from coveragecv.training.runner import (
    CoverageModelModule,
    RunEvidence,
    assert_data_contract,
    checkpoint_config,
    configs,
    save_detector,
)

ROOT = Path(__file__).resolve().parents[1]


def run(task, arm, alpha, steps, output):
    spec = read_json(ROOT / "artifacts" / ("chess_experiment.json" if task == "pawns"
                                         else "full_chess/experiment.json"))
    view = Path(spec["complete_view" if arm == "complete_reference" else "partial_view"])
    manifest = verify(view)
    if task == "pawns":
        method = {"aware": "aware_augmented_512", "naive": "naive_augmented_512",
                  "complete_reference": "complete_augmented_512"}[arm]
        parent = ROOT / "artifacts/improved/pawns/20260917" / method
    else:
        parent = ROOT / "artifacts/capacity/all-pieces/20260917" / f"{arm}_large_704_ema"
    checkpoint = parent / "detector.pt"
    source = read_json(parent / "run.json")
    parent_sha = file_digest(checkpoint)
    if source["arm"] != arm or source["status"] != "completed" or source["detector_sha256"] != parent_sha:
        raise ValueError("Parent detector contract mismatch")
    if source.get("view_digest") != manifest["digest"]:
        raise ValueError("Parent detector trained on a different learner view")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "run.json").exists():
        saved = read_json(output / "run.json")
        if (saved["status"] != "completed" or saved["steps"] != steps or saved["alpha"] != alpha
                or saved["warm_start_sha256"] != parent_sha
                or saved["detector_sha256"] != file_digest(output / "detector.pt")):
            raise ValueError("Existing attempt must not be overwritten")
        return saved
    torch.set_num_threads(4)
    seed_everything(20260918, workers=True)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    mc = checkpoint_config(state, device="mps")
    batch = 4
    count = len(read_json(view / "train/_annotations.coco.json")["images"])
    epochs = math.ceil(steps/(count//batch))+1
    _, tc = configs(view, output, batch=batch, epochs=epochs, device="mps", seed=20260918,
                    recipe="large_fresh" if task == "all-pieces" else "augmented")
    tc.lr, tc.lr_encoder, tc.use_ema = 2e-5, 2e-6, True
    module = CoverageModelModule(mc, tc, view=view, aware=arm == "aware")
    module.model.load_state_dict(state["model"], strict=True)
    module.criterion = with_power_localization(module.criterion, alpha)
    del state
    dm = RFDETRDataModule(mc, tc)
    dm.setup("fit")
    assert_data_contract(dm, view)
    protocol = {"kind": "alpha_giou_continuation", "task": task, "arm": arm, "alpha": alpha,
                "seed": 20260918, "requested_steps": steps, "parent_steps": source.get("total_training_steps", source["steps"]),
                "warm_start_sha256": parent_sha, "view_digest": manifest["digest"], "device": "mps",
                "batch": batch, "train_config": tc.model_dump(mode="json"),
                "model_config": mc.model_dump(mode="json"), "validation_during_training": False,
                "checkpoint_policy": "fixed final-step EMA; no validation checkpoint selection",
                "loss_source_sha256": file_digest(ROOT / "src/coveragecv/training/power_localization.py"),
                "cash_cost": 0, "status": "running"}
    write_json(output / "run.json", protocol)
    ema = RFDETREMACallback(decay=tc.ema_decay, tau=tc.ema_tau)
    trainer = build_trainer(tc, mc, accelerator="mps", devices=1, precision="32-true",
        include_training_callbacks=False, callbacks=[RunEvidence(output), ema], enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0,
        max_steps=steps, limit_val_batches=0, max_time={"seconds": 3600}, log_every_n_steps=1,
        deterministic="warn")
    started = time.monotonic()
    try:
        trainer.fit(module, datamodule=dm)
        if trainer.global_step != steps:
            raise RuntimeError("Fixed continuation budget did not complete")
        # EMA callback leaves the detector in its final averaged state.
        save_detector(module, mc, tc, output / "detector.pt")
        protocol.update(status="completed", steps=trainer.global_step,
                        total_training_steps=protocol["parent_steps"]+trainer.global_step,
                        elapsed_seconds=time.monotonic()-started,
                        detector_sha256=file_digest(output / "detector.pt"))
    except BaseException as exc:
        protocol.update(status="failed", error_type=type(exc).__name__, elapsed_seconds=time.monotonic()-started)
        raise
    finally:
        write_json(output / "run.json", protocol)
    del trainer, module, dm
    torch.mps.empty_cache()
    result = evaluate_checkpoint(output / "detector.pt", Path(spec["complete_bundle"]),
                                 output / "evaluation.json", device="cpu")
    print({"task": task, "arm": arm, "alpha": alpha, "steps": steps, "AP": result["metrics"]["AP"]}, flush=True)
    return protocol


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["pawns", "all-pieces"], default="pawns")
    parser.add_argument("--arm", choices=["aware", "naive", "complete_reference"], default="aware")
    parser.add_argument("--alpha", choices=[1, 3], type=int, default=3)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.task, args.arm, args.alpha, args.steps, args.output)
