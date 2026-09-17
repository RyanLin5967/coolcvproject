"""Controlled source-level probe, not a claim about production museum data.

Uses downloaded upstream crop, scaffold and GBIF export functions unchanged.
Creates a tiny synthetic two-square image and supplied detector predictions.
No model inference, real specimen records, or external services are invoked.
"""
import csv
import importlib.util
import json
import tempfile
import sys
import types
from pathlib import Path
from PIL import Image, ImageDraw

root = Path(__file__).parent
spec = importlib.util.spec_from_file_location(
    "upstream_scaffold_locations", root / "customers_drawerdissect_locations.py"
)
upstream = importlib.util.module_from_spec(spec)
spec.loader.exec_module(upstream)

logs = types.ModuleType("logging_utils")
logs.log = logs.log_found = logs.log_progress = lambda *a, **k: None
sys.modules["logging_utils"] = logs

def load_upstream(name, filename):
    spec = importlib.util.spec_from_file_location(name, root / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

cropper = load_upstream("upstream_cropper", "customers_drawerdissect_crop_specimens.py")
exporter = load_upstream("upstream_gbif", "customers_drawerdissect_gbif.py")

with tempfile.TemporaryDirectory(prefix="roboflow-customer-probe-") as folder:
    folder = Path(folder)
    trays = folder / "trays"
    resized = folder / "resized_trays"
    coordinates = resized / "coordinates"
    trays.mkdir()
    coordinates.mkdir(parents=True)
    tray_image = Image.new("RGB", (300, 120), "white")
    draw = ImageDraw.Draw(tray_image)
    draw.rectangle((80, 40, 120, 80), fill=(255, 0, 0))
    draw.rectangle((180, 40, 220, 80), fill=(0, 255, 0))
    tray_image.save(trays / "drawer_tray_01.png")
    box_a = {"x": 100, "y": 60, "width": 40, "height": 40}
    box_b = {"x": 200, "y": 60, "width": 40, "height": 40}

    def run_actual_cropper(predictions, run_name):
        output = folder / run_name
        (coordinates / "drawer_tray_01_1000.json").write_text(json.dumps({
            "predictions": predictions, "image": {"width": 300, "height": 120}
        }))
        succeeded = cropper.process_tray((
            str(trays), str(resized), str(output), str(resized),
            "drawer_tray_01_1000.jpg", 1, 1,
        ))
        assert succeeded
        identity = {}
        for crop in sorted(output.rglob("*.png")):
            with Image.open(crop) as image:
                rgb = image.getpixel((image.width // 2, image.height // 2))
                identity[crop.stem] = "A" if rgb == (255, 0, 0) else "B" if rgb == (0, 255, 0) else "unknown"
        return identity

    before = run_actual_cropper([box_b], "old_specimens")
    after = run_actual_cropper([box_a, box_b], "new_specimens")
    curation = folder / "curation.csv"
    source = folder / "new.csv"
    with curation.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["specimen_id", "country", "approved"])
        writer.writeheader()
        writer.writerow({"specimen_id": "drawer_tray_01_spec_001", "country": "Canada", "approved": "yes"})
    with source.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["specimen_id", "tray", "country"])
        writer.writeheader()
        writer.writerows([
            {"specimen_id": "drawer_tray_01_spec_001", "tray": "drawer_tray_01", "country": "Mexico"},
            {"specimen_id": "drawer_tray_01_spec_002", "tray": "drawer_tray_01", "country": "Canada"},
        ])
    existing = upstream.load_existing_ids(curation)
    new_rows, counts, source_ids = upstream.build_new_rows(source, [], existing)
    orphan_ids = upstream.find_orphans(curation, source_ids)
    master = folder / "master_specimens.csv"
    with master.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["full_id", "drawer_id", "tray_id", "full_taxonomy"])
        writer.writeheader()
        writer.writerows([
            {"full_id": sid, "drawer_id": "drawer", "tray_id": "drawer_tray_01", "full_taxonomy": "Synthetic specimen"}
            for sid in after
        ])
    old_approved = {"specimen_id": "drawer_tray_01_spec_001", "country": "Canada", "approved": "yes"}
    export_rows, pending, _ = exporter.process_specimens(
        master, {}, {"drawer_tray_01_spec_001": {"FMNH-INS#": "SYNTHETIC-B"}},
        {"drawer_tray_01_spec_001": old_approved}, [], "local-fixture/{full_id}.png"
    )
    report = {
        "scope": "Controlled synthetic-image/prediction scenario; upstream crop, CSV scaffold and GBIF export functions called unchanged; only logging stubbed. Separate output directories emulate regenerated outputs. No model inference or production records involved.",
        "before_id_to_physical_specimen": before,
        "after_id_to_physical_specimen": after,
        "new_row_ids": [r["specimen_id"] for r in new_rows],
        "orphan_ids": orphan_ids,
        "old_approved_row_is_retained": "drawer_tray_01_spec_001" not in [r["specimen_id"] for r in new_rows],
        "exported_rows": export_rows,
        "pending_rows": pending,
        "conclusion": "For these fabricated inputs, actual crop numbering reassigns spec_001 from B to A; unchanged ID suppresses re-curation; no orphan is reported; GBIF export joins retained B metadata to current A crop path. Supported --rerun clears crop outputs without clearing top-level curation. This establishes a controlled mechanism, not its prevalence or any real incident."
    }
    (root / "customers_identity_probe_result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
