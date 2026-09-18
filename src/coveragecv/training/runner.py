"""Run the upstream Lightning loop with a narrowly scoped coverage criterion."""
import hashlib
import math
import os
import tempfile
import time
from pathlib import Path

import torch
from pytorch_lightning import Callback, seed_everything
from rfdetr.assets.model_weights import download_pretrain_weights
from rfdetr.config import RFDETRLargeConfig, RFDETRNanoConfig, TrainConfig
from rfdetr.training import RFDETRDataModule, RFDETRModelModule, build_trainer

from coveragecv.artifacts import file_digest, read_json, verify, write_json
from coveragecv.training.criterion import CoverageSetCriterion, coverage_table


def configs(view, output, *, pretrained=None, batch=2, epochs=10, device="cpu", seed=20260917,
            recipe="pilot"):
    if recipe not in ("pilot", "continued", "augmented", "augmented_fresh", "large_fresh"):
        raise ValueError("unknown training recipe")
    classes = read_json(Path(view) / "ontology.json")["classes"]
    improved = recipe in ("augmented", "augmented_fresh", "large_fresh")
    large = recipe == "large_fresh"
    model = (RFDETRLargeConfig if large else RFDETRNanoConfig)(
        num_classes=len(classes), pretrain_weights=pretrained, device=device, compile=False,
        resolution=704 if large else (512 if improved else 384), positional_encoding_size=44 if large else 24)
    train = TrainConfig(dataset_dir=str(view), output_dir=str(output), dataset_file="roboflow", batch_size=batch,
                        epochs=epochs, grad_accum_steps=1, use_ema=large, multi_scale=False, expanded_scales=False,
                        scale_jitter=improved, aug_config=None if improved else {}, square_resize_div_64=True,
                        augmentation_backend="torchvision", num_workers=0, pack_targets=False,
                        tensorboard=False, wandb=False, mlflow=False, early_stopping=False, progress_bar=None,
                        seed=seed, compute_val_loss=True, checkpoint_interval=max(1, epochs),
                        lr=1e-4 if recipe in ("pilot", "augmented_fresh", "large_fresh") else 5e-5,
                        lr_encoder=1e-5 if recipe in ("pilot", "augmented_fresh", "large_fresh") else 5e-6,
                        lr_scheduler="step" if recipe == "pilot" else "cosine",
                        lr_drop=max(1, epochs), warmup_epochs=0)
    return model, train


def checkpoint_config(state, *, device="cpu"):
    """Reconstruct the recorded architecture, never silently choose a different family."""
    variant = state.get("model_variant", "nano")
    if variant not in ("nano", "large"):
        raise ValueError("unsupported checkpoint model variant")
    cls = RFDETRLargeConfig if variant == "large" else RFDETRNanoConfig
    return cls(**{**state["model_config"], "pretrain_weights": None, "device": device})


def parameter_digest(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode()+b"\0")
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def reset_task_heads(model, seed=20260917):
    seen = set()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for head in [model.class_embed, *model.transformer.enc_out_class_embed]:
            if id(head) in seen:
                continue
            seen.add(id(head))
            torch.nn.init.xavier_uniform_(head.weight)
            torch.nn.init.constant_(head.bias, -math.log(99))
    return len(seen)


def create_initialization(view: Path, output: Path, *, seed=20260917, variant="nano"):
    if variant not in ("nano", "large"):
        raise ValueError("unsupported model variant")
    manifest = verify(view)
    classes = read_json(view / "ontology.json")["classes"]
    output.mkdir(parents=True, exist_ok=True)
    target = output / "initialization.pt"
    if target.exists():
        metadata = read_json(output / "initialization.json")
        if file_digest(target) != metadata["checkpoint_sha256"] or metadata["seed"] != seed:
            raise ValueError("existing initialization does not match its manifest")
        saved = torch.load(target, map_location="cpu", weights_only=True)
        if saved.get("model_variant", "nano") != variant:
            raise ValueError("existing initialization uses a different model variant")
        if saved["classes"] != classes:
            raise ValueError("existing initialization uses a different ordered ontology")
        return target
    seed_everything(seed, workers=True)
    pretrained = Path.home() / ".cache/coveragecv" / (
        "rf-detr-large-2026.pth" if variant == "large" else "rf-detr-nano.pth")
    pretrained.parent.mkdir(parents=True, exist_ok=True)
    download_pretrain_weights(str(pretrained))
    mc, tc = configs(view, output, pretrained=str(pretrained), seed=seed,
                     recipe="large_fresh" if variant == "large" else "pilot")
    module = RFDETRModelModule(mc, tc)
    heads = reset_task_heads(module.model, seed)
    state = {k: v.detach().cpu() for k, v in module.model.state_dict().items()}
    # Keep the serialization basename stable, publish the checkpoint only when fully written.
    with tempfile.TemporaryDirectory(dir=output, prefix=".initialization-") as tmp:
        staged = Path(tmp) / target.name
        torch.save({"model": state, "classes": classes, "seed": seed, "model_variant": variant,
                    "model_config": mc.model_dump(mode="json"),
                    "parameter_digest": parameter_digest(module.model)}, staged)
        write_json(output / "initialization.json", {"seed": seed, "reset_heads": heads, "model_variant": variant,
            "classes": classes, "ontology_digest": manifest["ontology_digest"],
            "policy": "Xavier uniform weights; prior-0.01 bias on all K+1 main/encoder rows",
            "pretrained_sha256": file_digest(pretrained), "checkpoint_sha256": file_digest(staged),
            "parameter_digest": parameter_digest(module.model)})
        os.replace(staged, target)
    return target


class CoverageModelModule(RFDETRModelModule):
    def __init__(self, model_config, train_config, *, view: Path, aware: bool, pseudo_box_weight=1.):
        table, valid, manifest = coverage_table(view)
        super().__init__(model_config, train_config)
        self.coverage_manifest = manifest
        if aware:
            self.criterion = CoverageSetCriterion.from_stock(self.criterion, table, valid,
                contract_digest=manifest["digest"], pseudo_box_weight=pseudo_box_weight)

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint["coveragecv"] = {"view_digest": self.coverage_manifest["digest"],
                                    "ontology_digest": self.coverage_manifest["ontology_digest"]}

    def on_load_checkpoint(self, checkpoint):
        binding = checkpoint.get("coveragecv")
        if binding is None or binding["view_digest"] != self.coverage_manifest["digest"]:
            raise ValueError("checkpoint is not bound to the selected coverage view")
        super().on_load_checkpoint(checkpoint)


class RunEvidence(Callback):
    def __init__(self, output):
        self.output, self.started, self.rows = Path(output), time.monotonic(), []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % 10 == 0:
            loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
            value = float(loss.detach().cpu()) if isinstance(loss, torch.Tensor) else None
            write_json(self.output / "live.json", {"step": trainer.global_step,
                "epoch": trainer.current_epoch, "elapsed_seconds": time.monotonic()-self.started,
                "loss": value if value is None or math.isfinite(value) else None})

    def on_train_epoch_end(self, trainer, pl_module):
        row = {"epoch": trainer.current_epoch, "step": trainer.global_step,
               "elapsed_seconds": time.monotonic()-self.started,
               "metrics": {k: float(v.detach().cpu()) for k, v in trainer.callback_metrics.items()
                           if isinstance(v, torch.Tensor) and v.numel() == 1}}
        self.rows.append(row)
        write_json(self.output / "progress.json", self.rows)
        print(f"coveragecv epoch={row['epoch']} step={row['step']} elapsed={row['elapsed_seconds']:.1f}s", flush=True)


def run_training(view: Path, initialization: Path, output: Path, *, arm, epochs=10, batch=2,
                 max_steps=-1, device="cpu", seed=20260917, timeout_seconds=3600,
                 recipe="pilot", warm_start: Path | None = None, pseudo_box_weight=1.,
                 object_crop_plan: dict | None = None, crop_source_view: Path | None = None,
                 validation_during_training: bool | None = None):
    if arm not in ("naive", "aware", "complete_reference"):
        raise ValueError("unknown experiment arm")
    if not 0 <= pseudo_box_weight <= 1 or (pseudo_box_weight != 1 and arm != "aware"):
        raise ValueError("pseudo box weighting is an explicitly coverage-aware experiment")
    if device == "cpu":
        torch.set_num_threads(min(8, torch.get_num_threads()))
    manifest = verify(view)
    classes = read_json(view / "ontology.json")["classes"]
    if arm == "complete_reference":
        table, valid, _ = coverage_table(view)
        if not torch.all(table[valid]):
            raise ValueError("complete-reference training requires complete coverage")
    if output.exists() and (output / "run.json").exists():
        raise ValueError("run directory already has a ledger; use a new attempt directory")
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(seed, workers=True)
    mc, tc = configs(view, output, batch=batch, epochs=epochs, device=device, seed=seed, recipe=recipe)
    module = CoverageModelModule(mc, tc, view=view, aware=arm == "aware", pseudo_box_weight=pseudo_box_weight)
    initial = torch.load(initialization, map_location="cpu", weights_only=True)
    variant = "large" if recipe == "large_fresh" else "nano"
    if initial.get("model_variant", "nano") != variant:
        raise ValueError("initialization architecture does not match the training recipe")
    if initial["classes"] != classes:
        raise ValueError("initialization and learner ontology differ")
    module.model.load_state_dict(initial["model"], strict=True)
    sha = parameter_digest(module.model)
    if sha != initial["parameter_digest"]:
        raise ValueError("initial model parameters differ from the shared state")
    start_sha = sha
    if warm_start is not None:
        warm = torch.load(warm_start, map_location="cpu", weights_only=True)
        if warm.get("class_names") != classes:
            raise ValueError("warm-start checkpoint and learner ontology differ")
        module.model.load_state_dict(warm["model"], strict=True)
        start_sha = parameter_digest(module.model)
    dm = RFDETRDataModule(mc, tc)
    dm.setup("fit")
    assert_data_contract(dm, view)
    if (view / "pseudo-labels.json").exists():
        from coveragecv.training.provenance import PreservePseudoProvenance
        dm._dataset_train.prepare = PreservePseudoProvenance(dm._dataset_train.prepare)
    elif pseudo_box_weight != 1:
        raise ValueError("pseudo box weighting requires an audited pseudo-label learner view")
    if object_crop_plan is not None:
        if crop_source_view is None or max_steps < 1 or epochs != 1:
            raise ValueError("object crops require an explicit source view and one fixed-step epoch")
        if (view / "pseudo-labels.json").exists():
            raise ValueError("object crops currently require human-only supervision")
        from coveragecv.training.object_crops import install_object_crops
        install_object_crops(dm, view, crop_source_view, object_crop_plan)
        write_json(output / "object_crop_plan.json", object_crop_plan)
    elif crop_source_view is not None:
        raise ValueError("crop source supplied without a crop plan")
    run_validation = not tc.use_ema if validation_during_training is None else validation_during_training
    spec = {"arm": arm, "seed": seed, "classes": classes, "epochs": epochs, "batch": batch, "max_steps": max_steps,
            "view_digest": manifest["digest"], "initialization_sha256": file_digest(initialization),
            "initial_parameter_digest": sha, "model_config": mc.model_dump(mode="json"),
            "train_config": tc.model_dump(mode="json"), "status": "running",
            "device": device, "timeout_seconds": timeout_seconds, "torch_num_threads": torch.get_num_threads(),
            "recipe": recipe, "start_parameter_digest": start_sha,
            "warm_start_sha256": file_digest(warm_start) if warm_start else None,
            "optimizer_restarted": warm_start is not None, "pseudo_box_weight": pseudo_box_weight,
            "model_variant": variant, "export_weights": "ema" if tc.use_ema else "last",
            "validation_during_training": run_validation,
            "object_crop_plan_digest": object_crop_plan["digest"] if object_crop_plan is not None else None}
    spec["determinism"] = f"warn_on_nondeterministic_{device}_ops" if device in ("cuda", "mps") else "deterministic"
    write_json(output / "run.json", spec)
    started = time.monotonic()
    evidence = RunEvidence(output)
    callbacks = [evidence]
    if tc.use_ema:
        from rfdetr.training.callbacks.ema import RFDETREMACallback
        callbacks.append(RFDETREMACallback(decay=tc.ema_decay, tau=tc.ema_tau,
                                           update_interval_steps=tc.ema_update_interval))
    trainer = build_trainer(tc, mc, accelerator="gpu" if device == "cuda" else device,
        devices=1, precision="32-true", include_training_callbacks=False,
        callbacks=callbacks, enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False, num_sanity_val_steps=0, max_steps=max_steps,
        limit_val_batches=1.0 if run_validation else 0,
        max_time={"seconds": timeout_seconds}, log_every_n_steps=1,
        deterministic="warn" if device in ("cuda", "mps") else True)
    try:
        trainer.fit(module, datamodule=dm)
        trainer.save_checkpoint(output / "resume.ckpt")
        completed = trainer.global_step >= max_steps if max_steps > 0 else trainer.current_epoch >= epochs
        save_detector(module, mc, tc, output / "detector.pt")
        spec.update(status="completed" if completed else "incomplete", steps=trainer.global_step,
                    elapsed_seconds=time.monotonic()-started, final_parameter_digest=parameter_digest(module.model),
                    detector_sha256=file_digest(output / "detector.pt"))
    except BaseException as exc:
        spec.update(status="failed", error_type=type(exc).__name__, elapsed_seconds=time.monotonic()-started)
        raise
    finally:
        write_json(output / "run.json", spec)
    return spec


def save_detector(module, mc, tc, path):
    from rfdetr._namespace import _namespace_from_configs
    args = vars(_namespace_from_configs(mc, tc))
    classes = read_json(Path(tc.dataset_dir) / "ontology.json")["classes"]
    variant = "large" if isinstance(mc, RFDETRLargeConfig) else "nano"
    model_name = "RFDETRLarge" if variant == "large" else "RFDETRNano"
    args.update(model_name=f"rf-detr-{variant}", class_names=classes, num_classes=len(classes))
    # Plain detector state + metadata; no custom criterion object is serialized.
    # The public from_checkpoint loader needs its canonical top-level variant name.
    # model_config restores custom resolution and PE grids without caller overrides.
    torch.save({"model": {k: v.detach().cpu() for k, v in module.model.state_dict().items()},
                "args": args, "class_names": classes, "model_variant": variant, "model_name": model_name,
                "model_config": mc.model_dump(mode="json")}, path)


def assert_data_contract(dm, view):
    for split, dataset in (("train", dm._dataset_train), ("valid", dm._dataset_val)):
        expected = read_json(view / split / "_annotations.coco.json")
        mapping = {c["id"]: i for i, c in enumerate(expected["categories"])}
        assert dataset.cat2label == mapping, f"unexpected {split} category mapping"
        assert set(dataset.ids) == {im["id"] for im in expected["images"]}
        for image in expected["images"]:
            actual = dataset.coco.imgs[image["id"]]
            assert actual["file_name"] == image["file_name"]
