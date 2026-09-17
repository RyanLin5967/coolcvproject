"""Small account-free CLI; GPU/provider dependencies are imported only when used."""
import json
from pathlib import Path

import typer

from .artifacts import read_json, verify
from .compiler import compile_bundle, materialize_view
from .schema import CompileSpec, DiagnosticError

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False,
                  help="Preserve annotation coverage from datasets to RF-DETR training.")


def _result(fn, *args, **kwargs):
    try:
        value = fn(*args, **kwargs)
        typer.echo(json.dumps(value, indent=2, default=str))
    except DiagnosticError as e:
        typer.echo(json.dumps(e.as_dict()), err=True)
        raise typer.Exit(2) from None


@app.command()
def compile(spec: Path, output: Path = Path("artifacts/bundles")):
    """Validate source images, labels and coverage; publish an immutable bundle."""
    _result(compile_bundle, spec, output)


@app.command()
def validate(spec: Path):
    """Check manifest schema. Full data validation occurs during compilation."""
    _result(lambda: CompileSpec.model_validate_json(spec.read_text()).model_dump())


@app.command()
def inspect(bundle: Path):
    """Verify every payload, then show coverage and split counts."""
    _result(lambda: {"manifest": verify(bundle), "diagnostics": read_json(bundle / "diagnostics.json")})


@app.command()
def view(bundle: Path, output: Path = Path("artifacts/views")):
    """Materialize the actual RF-DETR layout without test/hidden oracle files."""
    _result(materialize_view, bundle, output)


@app.command()
def prepare(dataset: str = typer.Argument("chess"), root: Path = Path("data/chess"), output: Path = Path("artifacts"),
            download: bool = True):
    """Fetch and verify the pinned public pilot, then build paired training views."""
    if dataset != "chess":
        raise typer.BadParameter("only the pinned chess pilot is supported")
    from .datasets import prepare_chess
    _result(prepare_chess, root, output, download=download)


@app.command()
def doctor():
    """Show installed versions/device availability without reading credentials."""
    import importlib.metadata
    import platform
    versions = {}
    for name in ("coveragecv", "pydantic", "rfdetr", "torch", "torchvision", "modal", "roboflow"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    typer.echo(json.dumps({"python": platform.python_version(), "platform": platform.platform(),
                           "versions": versions}, indent=2))


@app.command()
def train(view: Path, initialization: Path, output: Path, arm: str = "aware", epochs: int = 1,
          max_steps: int = 100, batch: int = 2, device: str = "cpu"):
    """Run the pinned two-pawn pilot using the actual upstream Lightning loop."""
    from .training.runner import run_training
    _result(run_training, view, initialization, output, arm=arm, epochs=epochs, max_steps=max_steps,
            batch=batch, device=device)


@app.command()
def evaluate(checkpoint: Path, bundle: Path, output: Path, split: str = "valid"):
    """Evaluate against complete reference labels with common stock postprocessing."""
    from .training.evaluate import evaluate_checkpoint
    _result(lambda: evaluate_checkpoint(checkpoint, bundle, output, split=split)["metrics"])


@app.command()
def report(experiment: Path = Path("artifacts/chess_experiment.json"), runs: Path = Path("artifacts/mvp"),
           output: Path = Path("artifacts/demo/index.html")):
    """Build an offline interactive report from saved artifacts and measurements."""
    from .report import build_report
    _result(build_report, experiment, runs, output)


@app.command()
def demo(root: Path = Path("data/chess"), output: Path = Path("artifacts"), steps: int = 100,
         download: bool = True):
    """Reproduce the grouped local three-arm MVP. Existing completed runs are reused."""
    from .audit import group_chess
    from .datasets import prepare_chess
    from .report import build_report
    from .training.evaluate import evaluate_checkpoint
    from .training.runner import create_initialization, run_training
    prepare_chess(root, output, download=download)
    ex = group_chess(root, output)
    init = create_initialization(Path(ex["partial_view"]), output / "initialization")
    for arm in ("naive", "aware", "complete_reference"):
        folder = output / "mvp" / arm
        view = Path(ex["complete_view"] if arm == "complete_reference" else ex["partial_view"])
        if (folder / "run.json").exists():
            saved = read_json(folder / "run.json")
            if saved["arm"] != arm or saved["status"] != "completed" or saved["max_steps"] != steps or saved["view_digest"] != verify(view)["digest"]:
                raise typer.BadParameter("existing run does not match requested experiment; choose a new output root")
        else:
            run_training(view, init, folder, arm=arm, max_steps=steps, epochs=max(1, (steps+99)//100))
        if not (folder / "evaluation.json").exists():
            evaluate_checkpoint(folder / "detector.pt", Path(ex["complete_bundle"]), folder / "evaluation.json")
    _result(build_report, output / "chess_experiment.json", output / "mvp", output / "demo/index.html")


@app.command("roboflow-upload")
def roboflow_upload(view: Path, output: Path = Path("artifacts/provider")):
    """Upload observed train/validation data to the dedicated Roboflow demo project."""
    from .providers import upload_demo_view
    _result(lambda: {k: v for k, v in upload_demo_view(view, output).items() if k != "images"})


@app.command("roboflow-verify-export")
def roboflow_verify_export(view: Path, exported: Path,
                           output: Path = Path("artifacts/provider/roundtrip_verification.json")):
    """Verify every image, split, class and box in a downloaded Roboflow COCO export."""
    from .providers import verify_roboflow_export
    _result(verify_roboflow_export, view, exported, output)


if __name__ == "__main__":
    app()
