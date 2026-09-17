"""Run the upstream Lightning loop with a narrowly scoped coverage criterion."""
import hashlib
import math
import time
from pathlib import Path

import torch
from pytorch_lightning import Callback, seed_everything
from rfdetr.assets.model_weights import download_pretrain_weights
from rfdetr.config import RFDETRNanoConfig, TrainConfig
from rfdetr.training import RFDETRDataModule, RFDETRModelModule, build_trainer

from coveragecv.artifacts import file_digest, read_json, verify, write_json
from coveragecv.training.criterion import CoverageSetCriterion, coverage_table


def configs(view, output, *, pretrained=None, batch=2, epochs=10, device="cpu", seed=20260917):
    model = RFDETRNanoConfig(num_classes=2, pretrain_weights=pretrained, device=device, compile=False)
    train = TrainConfig(dataset_dir=str(view), output_dir=str(output), dataset_file="roboflow", batch_size=batch,
                        epochs=epochs, grad_accum_steps=1, use_ema=False, multi_scale=False, expanded_scales=False,
                        scale_jitter=False, aug_config={}, square_resize_div_64=True,
                        augmentation_backend="torchvision", num_workers=0, pack_targets=False,
                        tensorboard=False, wandb=False, mlflow=False, early_stopping=False, progress_bar=None,
                        seed=seed, compute_val_loss=True, checkpoint_interval=max(1, epochs),
                        lr=1e-4, lr_encoder=1e-5, lr_drop=max(1, epochs), warmup_epochs=0)
    return model, train


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


def create_initialization(view: Path, output: Path, *, seed=20260917):
    output.mkdir(parents=True, exist_ok=True)
    target = output / "initialization.pt"
    if target.exists():
        metadata = read_json(output / "initialization.json")
        if file_digest(target) != metadata["checkpoint_sha256"] or metadata["seed"] != seed:
            raise ValueError("existing initialization does not match its manifest")
        return target
    seed_everything(seed, workers=True)
    pretrained = Path.home() / ".cache/coveragecv/rf-detr-nano.pth"
    pretrained.parent.mkdir(parents=True, exist_ok=True)
    download_pretrain_weights(str(pretrained))
    mc, tc = configs(view, output, pretrained=str(pretrained), seed=seed)
    module = RFDETRModelModule(mc, tc)
    heads = reset_task_heads(module.model, seed)
    state = {k: v.detach().cpu() for k, v in module.model.state_dict().items()}
    torch.save({"model": state, "classes": ["black-pawn", "white-pawn"], "seed": seed,
                "model_config": mc.model_dump(mode="json"), "parameter_digest": parameter_digest(module.model)}, target)
    write_json(output / "initialization.json", {"seed": seed, "reset_heads": heads,
        "policy": "Xavier uniform weights; prior-0.01 bias on all K+1 main/encoder rows",
        "pretrained_sha256": file_digest(pretrained), "checkpoint_sha256": file_digest(target),
        "parameter_digest": parameter_digest(module.model)})
    return target


class CoverageModelModule(RFDETRModelModule):
    def __init__(self, model_config, train_config, *, view: Path, aware: bool):
        table, valid, manifest = coverage_table(view)
        super().__init__(model_config, train_config)
        self.coverage_manifest = manifest
        if aware:
            self.criterion = CoverageSetCriterion.from_stock(self.criterion, table, valid,
                                                            contract_digest=manifest["digest"])

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

    def on_train_epoch_end(self, trainer, pl_module):
        row = {"epoch": trainer.current_epoch, "step": trainer.global_step,
               "elapsed_seconds": time.monotonic()-self.started,
               "metrics": {k: float(v.detach().cpu()) for k, v in trainer.callback_metrics.items()
                           if isinstance(v, torch.Tensor) and v.numel() == 1}}
        self.rows.append(row)
        write_json(self.output / "progress.json", self.rows)
        print(f"coveragecv epoch={row['epoch']} step={row['step']} elapsed={row['elapsed_seconds']:.1f}s", flush=True)


def run_training(view: Path, initialization: Path, output: Path, *, arm, epochs=10, batch=2,
                 max_steps=-1, device="cpu", seed=20260917, timeout_seconds=3600):
    if arm not in ("naive", "aware", "complete_reference"):
        raise ValueError("unknown experiment arm")
    if device == "cpu":
        torch.set_num_threads(min(8, torch.get_num_threads()))
    manifest = verify(view)
    if read_json(view / "ontology.json")["classes"] != ["black-pawn", "white-pawn"]:
        raise ValueError("the MVP training recipe requires the ordered black-pawn / white-pawn ontology")
    if output.exists() and (output / "run.json").exists():
        raise ValueError("run directory already has a ledger; use a new attempt directory")
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(seed, workers=True)
    mc, tc = configs(view, output, batch=batch, epochs=epochs, device=device, seed=seed)
    module = CoverageModelModule(mc, tc, view=view, aware=arm == "aware")
    initial = torch.load(initialization, map_location="cpu", weights_only=True)
    module.model.load_state_dict(initial["model"], strict=True)
    sha = parameter_digest(module.model)
    if sha != initial["parameter_digest"]:
        raise ValueError("initial model parameters differ from the shared state")
    dm = RFDETRDataModule(mc, tc)
    dm.setup("fit")
    assert_data_contract(dm, view)
    spec = {"arm": arm, "seed": seed, "epochs": epochs, "batch": batch, "max_steps": max_steps,
            "view_digest": manifest["digest"], "initialization_sha256": file_digest(initialization),
            "initial_parameter_digest": sha, "model_config": mc.model_dump(mode="json"),
            "train_config": tc.model_dump(mode="json"), "status": "running",
            "device": device, "timeout_seconds": timeout_seconds, "torch_num_threads": torch.get_num_threads()}
    write_json(output / "run.json", spec)
    started = time.monotonic()
    evidence = RunEvidence(output)
    trainer = build_trainer(tc, mc, accelerator="gpu" if device == "cuda" else device,
        devices=1, precision="32-true", include_training_callbacks=False,
        callbacks=[evidence], enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False, num_sanity_val_steps=0, max_steps=max_steps,
        max_time={"seconds": timeout_seconds}, log_every_n_steps=1, deterministic=True)
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
    args.update(model_name="rf-detr-nano", class_names=["black-pawn", "white-pawn"], num_classes=2)
    # Plain detector state + metadata; no custom criterion object is serialized.
    torch.save({"model": {k: v.detach().cpu() for k, v in module.model.state_dict().items()},
                "args": args, "class_names": ["black-pawn", "white-pawn"]}, path)


def assert_data_contract(dm, view):
    for split, dataset in (("train", dm._dataset_train), ("valid", dm._dataset_val)):
        expected = read_json(view / split / "_annotations.coco.json")
        assert dataset.cat2label == {1: 0, 2: 1}, f"unexpected {split} category mapping"
        assert set(dataset.ids) == {im["id"] for im in expected["images"]}
        for image in expected["images"]:
            actual = dataset.coco.imgs[image["id"]]
            assert actual["file_name"] == image["file_name"]
