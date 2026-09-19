"""Fixed-budget continuation for explicit object exclusivity and precision losses."""
import math
import time
from pathlib import Path

import torch
from pytorch_lightning import seed_everything
from rfdetr.training import RFDETRDataModule, build_trainer
from rfdetr.training.callbacks.ema import RFDETREMACallback

from coveragecv.artifacts import file_digest, read_json, verify, write_json
from coveragecv.training.criterion import CoverageSetCriterion, coverage_table
from coveragecv.training.power_localization import with_power_localization
from coveragecv.training.runner import (
    CoverageModelModule,
    RunEvidence,
    assert_data_contract,
    checkpoint_config,
    configs,
    save_detector,
)


def train(view: Path, parent: Path, output: Path, *, arm, steps=2000, seed=20260918,
          alpha=1, exclusive_groups=(), device="cuda", max_seconds=1300,
          resolution=None, exposure_plan=None, stable_assignment=None):
    if arm not in ("aware", "complete_reference") or not 1 <= steps <= 2000:
        raise ValueError("Outside the bounded research continuation")
    manifest = verify(view)
    classes = read_json(view / "ontology.json")["classes"]
    data = read_json(view / "train/_annotations.coco.json")
    if any(ann.get("is_pseudo", False) for ann in data["annotations"]):
        raise ValueError("Research v2 uses observed human annotations only")
    if arm == "complete_reference":
        allowed, valid, _ = coverage_table(view)
        if not allowed[valid].all():
            raise ValueError("Full-label reference has incomplete coverage")
    if (output / "run.json").exists():
        raise ValueError("Refusing to overwrite an existing training attempt")
    parent_run = read_json(parent / "run.json")
    checkpoint = parent / "detector.pt"
    sha = file_digest(checkpoint)
    if (parent_run["status"] != "completed" or parent_run["arm"] != arm
            or parent_run["detector_sha256"] != sha or parent_run["view_digest"] != manifest["digest"]):
        raise ValueError("Parent checkpoint, arm, or learner view mismatch")
    indices = []
    for group in exclusive_groups:
        if any(name not in classes for name in group):
            raise ValueError("Exclusive ontology names must exist in the learner")
        indices.append([classes.index(name) for name in group])
    torch.set_num_threads(4)
    seed_everything(seed, workers=True)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state["class_names"] != classes:
        raise ValueError("Parent class layout mismatch")
    mc = checkpoint_config(state, device=device)
    if resolution is not None:
        if resolution not in (512, 640, 768) or state.get("model_variant", "nano") != "nano":
            raise ValueError("The scale study supports explicit Nano512/640/768 only")
        # Resolution changes image/feature sampling, not checkpoint tensor shapes.
        mc.resolution = resolution
    epochs = math.ceil(steps/max(1, len(data["images"])//4))+1
    _, tc = configs(view, output, batch=4, epochs=epochs, device=device, seed=seed,
                    recipe="large_fresh" if state.get("model_variant") == "large" else "augmented")
    tc.lr, tc.lr_encoder, tc.use_ema = 2e-5, 2e-6, True
    if exposure_plan is not None:
        if len(exposure_plan["sample_ids"]) != steps * 4:
            raise ValueError("Exposure sequence must match the exact update budget")
        tc.epochs = 1
    module = CoverageModelModule(mc, tc, view=view, aware=arm == "aware")
    module.model.load_state_dict(state["model"], strict=True)
    del state
    if indices and arm == "aware":
        original = module.criterion
        module.criterion = CoverageSetCriterion.from_stock(original, original.negative_allowed,
            original.valid_ids, contract_digest=original.contract_digest, exclusive_groups=indices)
    module.criterion = with_power_localization(module.criterion, alpha)
    if stable_assignment is not None:
        from coveragecv.training.stable_assignment import with_stable_assignment
        module.criterion = with_stable_assignment(module.criterion, **stable_assignment)
    dm = RFDETRDataModule(mc, tc)
    dm.setup("fit")
    assert_data_contract(dm, view)
    if exposure_plan is not None:
        from coveragecv.training.exposure import bind_exposure
        bind_exposure(dm, view, exposure_plan)
    protocol = {"kind": "ontology_aware_continuation", "arm": arm, "alpha": alpha,
                "exclusive_groups": list(map(list, exclusive_groups)), "classes": classes,
                "seed": seed, "requested_steps": steps, "batch": 4,
                "parent_steps": parent_run.get("total_training_steps", parent_run["steps"]),
                "warm_start_sha256": sha, "view_digest": manifest["digest"], "device": device,
                "train_config": tc.model_dump(mode="json"), "model_config": mc.model_dump(mode="json"),
                "validation_during_training": False, "checkpoint_policy": "final fixed-step EMA",
                "annotation_count": len(data["annotations"]), "status": "running"}
    if exposure_plan is not None:
        protocol.update(kind="full_frame_exposure_continuation", exposure_plan_digest=exposure_plan["digest"],
                        sampling=exposure_plan["sampling"], resolution=mc.resolution)
    if stable_assignment is not None:
        protocol.update(kind="stable_assignment_continuation", stable_assignment=stable_assignment)
    write_json(output / "run.json", protocol)
    trainer = build_trainer(tc, mc, accelerator="gpu" if device == "cuda" else device, devices=1,
        precision="32-true", include_training_callbacks=False,
        callbacks=[RunEvidence(output), RFDETREMACallback(decay=tc.ema_decay, tau=tc.ema_tau)],
        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
        num_sanity_val_steps=0, max_steps=steps, limit_val_batches=0,
        max_time={"seconds": max_seconds}, log_every_n_steps=1, deterministic="warn")
    started = time.monotonic()
    try:
        trainer.fit(module, datamodule=dm)
        if trainer.global_step != steps:
            raise RuntimeError("Training timed out before its fixed budget completed")
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
    return protocol
