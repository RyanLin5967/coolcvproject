"""Wait for the corrected import, then check labels and boxes against its native checkpoint."""
import time
from collections import Counter
from pathlib import Path

import requests
from roboflow.core.training import TrainedModel

from coveragecv.artifacts import read_json, write_json
from coveragecv.providers import credentials, redact, safe_sdk
from coveragecv.training.evaluate import iou

ROOT = Path(__file__).resolve().parents[1]
provider = ROOT / "artifacts/provider"
folder = provider / "model/semantic-corrected"
c = credentials()
deployment = read_json(folder / "deployment.json")
deadline = time.monotonic()+1800
while time.monotonic() < deadline:
    response = requests.get(
        f"https://api.roboflow.com/{c['ROBOFLOW_WORKSPACE']}/coveragecv-chess-mvp/3/v2/trainings/get",
        params={"api_key": c["ROBOFLOW_API_KEY"], "trainingId": deployment["training_id"]}, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f"Roboflow training status HTTP {response.status_code}")
    details = redact(response.json())
    write_json(provider / "corrected_training_details.json", details)
    deployment["status"] = details["status"]
    write_json(folder / "deployment.json", deployment)
    print("Corrected hosted import:", details["status"], flush=True)
    if details["status"] == "finished":
        break
    if details["status"] in ("failed", "cancelled", "error"):
        raise RuntimeError("Provider did not complete the corrected import")
    time.sleep(30)
else:
    raise RuntimeError("Hosted import remains pending; no semantic parity claim")

experiment = read_json(ROOT / "artifacts/chess_experiment.json")
reference = Path(experiment["complete_bundle"])
data = read_json(reference / "splits/valid.coco.json")
classes = read_json(reference / "ontology.json")["classes"]
evaluation = read_json(ROOT / "artifacts/gpu/pawns/20260917/aware/evaluation.json")
assert evaluation["checkpoint_sha256"] == deployment["export"]["native_checkpoint_sha256"]
selected = [im for im in data["images"]
            if {a["category_id"] for a in data["annotations"] if a["image_id"] == im["id"]} == {1, 2}][:3]
model = TrainedModel(c["ROBOFLOW_API_KEY"], c["ROBOFLOW_WORKSPACE"], "coveragecv-chess-mvp",
                     deployment["model_id"], model_type="rfdetr-nano")
results, matched_classes = [], Counter()
for im in selected:
    with safe_sdk():
        hosted = model.predict(str(reference / im["file_name"]), confidence=40, overlap=50).json()
    native = [p for p in evaluation["predictions"] if p["image_id"] == im["id"] and p["score"] >= .25]
    matches, used, failures = [], set(), []
    for prediction in hosted["predictions"]:
        if prediction["confidence"] < .6:
            continue
        category = prediction["class_id"]
        if not 0 <= category < len(classes) or classes[category] != prediction["class"]:
            failures.append("semantic class index/name mismatch")
            continue
        box = [prediction["x"]-prediction["width"]/2, prediction["y"]-prediction["height"]/2,
               prediction["width"], prediction["height"]]
        best, index = max(((iou(box, p["bbox"]), j) for j, p in enumerate(native)
                          if j not in used and p["category_id"] == category+1), default=(0, -1))
        if best < .85:
            failures.append(f"hosted {prediction['class']} has no native same-class box at IoU >=0.85")
        else:
            used.add(index)
            matched_classes[prediction["class"]] += 1
            matches.append({"class": prediction["class"], "iou": best,
                            "hosted_confidence": prediction["confidence"], "native_confidence": native[index]["score"]})
    # Each class with a confident native prediction must also be present in hosted output.
    native_classes = {classes[p["category_id"]-1] for p in native if p["score"] >= .6}
    if native_classes != {m["class"] for m in matches}:
        failures.append("hosted output dropped a confident semantic class")
    results.append({"image_id": im["id"], "response": hosted, "matches": matches, "failures": failures})
passed = len(results) == 3 and all(not r["failures"] for r in results) and all(matched_classes[c] >= 3 for c in classes)
evidence = {"status": "passed" if passed else "failed", "model_id": deployment["model_id"],
            "native_checkpoint_sha256": evaluation["checkpoint_sha256"], "classes": classes,
            "criteria": {"hosted_confidence": .6, "native_candidate_confidence": .25,
                         "minimum_same_class_box_iou": .85, "images": 3, "minimum_matches_per_class": 3},
            "matched_classes": dict(matched_classes), "images": results,
            "scope": "Semantic/geometry smoke parity; not bitwise numerical equivalence or an exhaustive hosted evaluation."}
write_json(folder / "semantic_parity.json", redact(evidence))
print("Hosted parity:", evidence["status"], dict(matched_classes), flush=True)
if not passed:
    raise SystemExit(1)
