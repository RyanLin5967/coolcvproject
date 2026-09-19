"""Copy the shared UI into the committed, credential-free static demo."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/coveragecv/workbench/static"
OUTPUT = ROOT / "public-demo"


def main():
    snapshot = json.loads((OUTPUT / "data/snapshot.json").read_text())
    if not all(route in snapshot["routes"] for route in ("/state", "/research")):
        raise ValueError("Export the verified public snapshot before building the demo")
    static = OUTPUT / "static"
    static.mkdir(exist_ok=True)
    for name in ("app.js", "benchmark-view.js", "merge-story.js", "verify-view.js", "coco-eval.js",
                 "style.css", "merge-story.css", "verify.css", "prediction-highlights.json"):
        (static / name).write_bytes((SOURCE / name).read_bytes())
    # The verifiable evidence bundle: saved predictions, reference labels and digests.
    verify_source, verify_output = SOURCE / "verify", static / "verify"
    if not (verify_source / "manifest.json").exists():
        raise ValueError("Build the verification bundle before building the demo")
    verify_output.mkdir(exist_ok=True)
    published = {"manifest.json"}
    manifest = json.loads((verify_source / "manifest.json").read_text())
    for entry in manifest["ground_truth"].values():
        published.add(entry["path"])
    for cohort in manifest["cohorts"]:
        for run in cohort["runs"]:
            published.add(run["predictions"]["path"])
    for name in sorted(published):
        (verify_output / name).write_bytes((verify_source / name).read_bytes())
    for stale in verify_output.iterdir():
        if stale.name not in published:
            stale.unlink()
    index = (SOURCE / "index.html").read_text().replace('<html lang="en">', '<html lang="en" data-demo="true">')
    (OUTPUT / "index.html").write_text(index)
    (OUTPUT / "_headers").write_text(
        "/*\n"
        "  X-Content-Type-Options: nosniff\n"
        "  Referrer-Policy: strict-origin-when-cross-origin\n"
        "  X-Frame-Options: DENY\n"
        "  Permissions-Policy: camera=(), microphone=(), geolocation=()\n"
        "  Content-Security-Policy: default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; media-src 'self'; connect-src 'self'; object-src 'none'; "
        "base-uri 'self'; frame-ancestors 'none'; form-action 'none'\n"
    )
    print("Built public-demo: static recorded-results app; no backend or credentials required.")


if __name__ == "__main__":
    main()
