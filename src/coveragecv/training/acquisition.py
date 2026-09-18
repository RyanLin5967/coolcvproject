"""Hash-bound, train-only simulated annotation acquisition.

Selection has no reference/oracle argument. Application reveals only the frozen
image/class queries from published training annotations. This spends additional
annotation review units; it is not new human work or an equal-label-budget loss
comparison. Published annotations can themselves contain errors.
"""
import copy
import math
import re
import shutil
from collections import Counter
from pathlib import Path

from coveragecv.artifacts import (
    canonical,
    digest,
    file_digest,
    publish,
    read_json,
    safe_child,
    stage,
    verify,
    write_json,
    write_jsonl,
)
from coveragecv.compiler import _jsonl
from coveragecv.schema import NEGATIVE_ALLOWED, STATES

ELIGIBLE = frozenset(("unknown", "positive_only"))
PROTOCOL = "published-train-coverage-acquisition-v1"


def _integer(value):
    return type(value) is int


def _sha(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _view(path):
    path = Path(path)
    manifest = verify(path)
    if manifest.get("kind") != "training_view":
        raise ValueError("acquisition requires a verified training_view")
    if any(name.split("/")[0] == "test" or name.startswith("splits/")
           for name in manifest["files"]):
        raise ValueError("learner view must not include test or reference split files")
    classes = read_json(path / "ontology.json")["classes"]
    if not classes or len(set(classes)) != len(classes):
        raise ValueError("ontology classes must be nonempty and unique")
    data = read_json(path / "train/_annotations.coco.json")
    categories = data["categories"]
    expected = {i + 1: name for i, name in enumerate(classes)}
    if (len(categories) != len(expected) or any(not _integer(c.get("id")) for c in categories)
            or [c["id"] for c in categories] != list(expected)
            or {c["id"]: c["name"] for c in categories} != expected):
        raise ValueError("train category IDs/names must match the ontology")
    images = {}
    filenames = set()
    for image in data["images"]:
        iid = image["id"]
        if (not _integer(iid) or not 0 <= iid < manifest["index_space_size"]
                or iid in images or image["file_name"] in filenames
                or any(not _integer(image[k]) or image[k] <= 0 for k in ("width", "height"))):
            raise ValueError("train images need unique integer IDs, filenames and positive dimensions")
        name = f"train/{image['file_name']}"
        safe_child(path, name)
        if name not in manifest["files"]:
            raise ValueError("train image is missing from the verified file inventory")
        images[iid] = image
        filenames.add(image["file_name"])
    rows = _jsonl(path / "coverage.jsonl")
    coverage = {}
    for row in rows:
        if row["split"] != "train":
            if row["split"] != "valid":
                raise ValueError("learner coverage must not contain test records")
            continue
        iid, states = row["image_id"], row["states"]
        if (not _integer(iid) or iid not in images or iid in coverage or len(states) != len(classes)
                or any(state not in STATES for state in states)):
            raise ValueError("train coverage must uniquely cover the image and category ID space")
        coverage[iid] = row
    if set(coverage) != set(images):
        raise ValueError("train coverage is missing image records")
    return path, manifest, classes, data, images, coverage, rows


def _box(row, images, class_count, *, prediction=False):
    if row.get("is_pseudo", False):
        raise ValueError("pseudo labels/predictions are not admissible annotation evidence")
    iid, category = row.get("image_id"), row.get("category_id")
    if (not _integer(iid) or iid not in images or not _integer(category)
            or not 1 <= category <= class_count):
        raise ValueError("annotation or prediction refers to an unknown image/category")
    box = row.get("bbox")
    if (not isinstance(box, list) or len(box) != 4
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in box)):
        raise ValueError("bbox must contain four finite geometry values")
    x, y, w, h = box
    image = images[iid]
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x+w > image["width"]+1e-5 or y+h > image["height"]+1e-5:
        raise ValueError("bbox geometry must be positive-area and within the image")
    if row.get("iscrowd", 0) or row.get("ignore", 0) or row.get("segmentation"):
        raise ValueError("crowd/ignore/segmentation annotations are unsupported")
    if prediction:
        score = row.get("score")
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("prediction score must be finite and between zero and one")
    return iid, category


def _label_key(ann):
    return canonical({"image_id": ann["image_id"], "category_id": ann["category_id"],
                      "bbox": [float(v) for v in ann["bbox"]]})


def _observed(data, images, coverage, class_count):
    seen_ids, labels = set(), {}
    for ann in data["annotations"]:
        iid, category = _box(ann, images, class_count)
        if not _integer(ann.get("id")) or ann["id"] in seen_ids:
            raise ValueError("observed annotation IDs must be unique integers")
        seen_ids.add(ann["id"])
        key = _label_key(ann)
        if key in labels:
            raise ValueError("duplicate observed box requires source correction")
        if coverage[iid]["states"][category-1] == "verified_absent":
            raise ValueError("observed box contradicts verified absence")
        labels[key] = ann
    return labels


def plan_reviews(partial_view, predictions, mode="guided", per_class=30, seed=20260917,
                 checkpoint_sha256=None):
    """Select equal image/class quotas using observed TRAIN information only.

    ``predictions`` is a sequence of original-pixel COCO xywh detection dicts.
    Guided ranks descending maximum detection confidence for each eligible pair.
    Random ranks by a seeded SHA-256 permutation. Guided ties use that same
    permutation. Return a canonical JSON-compatible plan with a content digest;
    changing its queries is rejected by ``apply_reviews``.
    """
    if mode not in ("guided", "random"):
        raise ValueError("mode must be guided or random")
    if not _integer(per_class) or per_class <= 0 or not _integer(seed):
        raise ValueError("per_class must be a positive integer and seed must be an integer")
    _sha(checkpoint_sha256, "checkpoint_sha256")
    path, manifest, classes, data, images, coverage, _ = _view(partial_view)
    _observed(data, images, coverage, len(classes))
    predictions = list(predictions)
    maxima = {}
    for row in predictions:
        pair = _box(row, images, len(classes), prediction=True)
        maxima[pair] = max(maxima.get(pair, 0.0), float(row["score"]))
    queries, eligible_counts = [], {}
    for category, name in enumerate(classes, 1):
        eligible = [iid for iid in images if coverage[iid]["states"][category-1] in ELIGIBLE]
        eligible_counts[str(category)] = len(eligible)
        if len(eligible) < per_class:
            raise ValueError(f"class {name} has only {len(eligible)} eligible pairs for {per_class} reviews")

        def rank(iid, category=category):
            tie = digest({"seed": seed, "image_id": iid, "category_id": category})
            return (-maxima.get((iid, category), 0.0) if mode == "guided" else 0, tie)

        for rank_index, iid in enumerate(sorted(eligible, key=rank)[:per_class], 1):
            queries.append({"image_id": iid, "category_id": category, "class_name": name,
                            "previous_state": coverage[iid]["states"][category-1],
                            "rank_within_class": rank_index,
                            "max_prediction_confidence": maxima.get((iid, category), 0.0)})
    body = {"schema_version": 1, "protocol": PROTOCOL, "kind": "coverage_review_plan",
            "mode": mode, "seed": seed, "per_class": per_class,
            "query_count": len(queries), "classes": classes,
            "partial_view_digest": manifest["digest"], "ontology_digest": manifest["ontology_digest"],
            "train_annotations_sha256": file_digest(path / "train/_annotations.coco.json"),
            "checkpoint_sha256": checkpoint_sha256,
            "prediction_sha256": digest(sorted(predictions, key=canonical)),
            "prediction_count": len(predictions), "eligible_pairs_per_class": eligible_counts,
            "review_units": "image-class pairs; equal units do not imply equal acquired box counts",
            "selection_information": "observed train labels, coverage, images and detector predictions only",
            "reference_annotations_used_for_selection": False,
            "validation_or_test_labels_used": False, "queries": queries}
    return {**body, "digest": digest(body)}


def _check_plan(plan, manifest, classes, coverage):
    body = {key: value for key, value in plan.items() if key != "digest"}
    if digest(body) != plan.get("digest"):
        raise ValueError("review plan digest mismatch")
    if (plan.get("schema_version") != 1 or plan.get("protocol") != PROTOCOL
            or plan.get("kind") != "coverage_review_plan" or plan.get("mode") not in ("guided", "random")
            or plan.get("classes") != classes or plan.get("partial_view_digest") != manifest["digest"]
            or plan.get("ontology_digest") != manifest["ontology_digest"]
            or plan.get("train_annotations_sha256") != manifest["files"]["train/_annotations.coco.json"]):
        raise ValueError("review plan is not bound to this train view/ontology/protocol")
    _sha(plan.get("checkpoint_sha256"), "checkpoint_sha256")
    if (not _integer(plan.get("per_class")) or plan["per_class"] <= 0
            or plan.get("reference_annotations_used_for_selection") is not False
            or plan.get("validation_or_test_labels_used") is not False):
        raise ValueError("review plan violates the frozen train-only selection protocol")
    selected, counts = set(), Counter()
    for row in plan["queries"]:
        iid, category = row["image_id"], row["category_id"]
        if (not _integer(iid) or iid not in coverage or not _integer(category)
                or not 1 <= category <= len(classes) or (iid, category) in selected
                or row["class_name"] != classes[category-1]
                or row["previous_state"] not in ELIGIBLE
                or coverage[iid]["states"][category-1] != row["previous_state"]):
            raise ValueError("review plan contains duplicate, ineligible or invalid image/category queries")
        selected.add((iid, category))
        counts[category] += 1
    if (len(selected) != plan["query_count"] or len(selected) != len(classes)*plan["per_class"]
            or any(counts[c] != plan["per_class"] for c in range(1, len(classes)+1))):
        raise ValueError("review plan must have exactly the frozen equal per-class quota")
    return selected


def apply_reviews(partial_view, complete_training_view, plan, output_root):
    """Reveal frozen queries from a published TRAIN annotation oracle.

    Unrequested reference boxes are neither validated nor used. Validation files
    are copied byte-for-byte from the original learner and never parsed. All
    observed annotations survive; incompatible queried reference labels fail
    closed. Returns a new immutable training-view path with a detailed audit.
    """
    partial, manifest, classes, data, images, coverage, rows = _view(partial_view)
    observed = _observed(data, images, coverage, len(classes))
    selected = _check_plan(plan, manifest, classes, coverage)
    _, ref_manifest, ref_classes, ref_data, ref_images, ref_coverage, _ = _view(complete_training_view)
    if classes != ref_classes or manifest["ontology_digest"] != ref_manifest["ontology_digest"]:
        raise ValueError("reference and observed ontology/category IDs differ")
    if images != ref_images:
        raise ValueError("reference and observed image IDs, names or dimensions differ")
    for image in images.values():
        name = f"train/{image['file_name']}"
        if manifest["files"][name] != ref_manifest["files"][name]:
            raise ValueError("reference and observed train image content hashes differ")
    oracle, oracle_ids = {}, set()
    for ann in ref_data["annotations"]:
        if (ann.get("image_id"), ann.get("category_id")) not in selected:
            continue
        pair = _box(ann, images, len(classes))
        if not _integer(ann.get("id")) or ann["id"] in oracle_ids:
            raise ValueError("queried reference annotation IDs must be unique integers")
        oracle_ids.add(ann["id"])
        key = _label_key(ann)
        if key in oracle:
            raise ValueError("duplicate queried reference box requires source correction")
        if ref_coverage[pair[0]]["states"][pair[1]-1] == "verified_absent":
            raise ValueError("queried reference box contradicts verified absence")
        oracle[key] = ann
    for iid, category in selected:
        if ref_coverage[iid]["states"][category-1] not in NEGATIVE_ALLOWED:
            raise ValueError("queried reference class lacks exhaustive/verified-absent coverage")
    for key, ann in observed.items():
        if (ann["image_id"], ann["category_id"]) in selected and key not in oracle:
            raise ValueError("queried reference is missing or altered an observed label")
    additions = [ann for key, ann in oracle.items() if key not in observed]
    combined = [(ann, False) for ann in data["annotations"]] + [(ann, True) for ann in additions]
    combined.sort(key=lambda item: (item[0]["image_id"], item[0]["category_id"], _label_key(item[0])))
    acquired_ids, id_map, new_annotations = [], [], []
    for new_id, (ann, added) in enumerate(combined, 1):
        new_ann = copy.deepcopy(ann)
        new_ann["id"] = new_id
        if added:
            # This is provenance, not a confidence weight or a pseudo-label flag.
            new_ann["annotation_origin"] = "published_training_annotation_oracle"
            acquired_ids.append(new_id)
        else:
            id_map.append({"original_id": ann["id"], "new_id": new_id})
        new_annotations.append(new_ann)
    data["annotations"] = new_annotations
    oracle_counts = Counter((ann["image_id"], ann["category_id"]) for ann in oracle.values())
    query_audit = []
    for row in plan["queries"]:
        iid, category = row["image_id"], row["category_id"]
        previous = coverage[iid]["states"][category-1]
        state = "exhaustive" if oracle_counts[iid, category] else "verified_absent"
        coverage[iid]["states"][category-1] = state
        query_audit.append({**row, "previous_state": previous, "new_state": state,
                            "published_reference_boxes": oracle_counts[iid, category],
                            "added_boxes": sum(ann["image_id"] == iid and ann["category_id"] == category
                                               for ann in additions)})
    before = Counter(ann["category_id"] for ann in observed.values())
    after = Counter(ann["category_id"] for ann in new_annotations)
    audit = {"schema_version": 1, "protocol": PROTOCOL, "kind": "simulated_annotation_acquisition",
             "simulation": True, "actual_new_human_annotation": False,
             "annotation_source": "published training annotations; known source errors may remain",
             "equal_review_units_do_not_imply_equal_added_boxes": True,
             "same_label_budget_as_original_experiment": False,
             "partial_view_digest": manifest["digest"], "plan_digest": plan["digest"],
             "reference_training_annotations_sha256": ref_manifest["files"]["train/_annotations.coco.json"],
             "reference_view_digest": ref_manifest["digest"], "ontology_digest": manifest["ontology_digest"],
             "train_images": len(images), "review_units": len(selected),
             "reviewed_unique_images": len({iid for iid, _ in selected}),
             "boxes_before": len(observed), "boxes_after": len(new_annotations),
             "added_boxes": len(additions), "observed_labels_preserved": True,
             "unrequested_reference_labels_used": False, "validation_or_test_labels_used": False,
             "per_class": [{"category_id": i, "class_name": name, "review_units": plan["per_class"],
                            "boxes_before": before[i], "boxes_after": after[i],
                            "added_boxes": after[i]-before[i]} for i, name in enumerate(classes, 1)],
             "acquired_annotation_ids": acquired_ids, "observed_annotation_id_map": id_map,
             "queries": query_audit}
    output_root = Path(output_root)
    staging = stage(output_root)
    try:
        for name in manifest["files"]:
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(partial / name, target)
        write_json(staging / "train/_annotations.coco.json", data)
        write_jsonl(staging / "coverage.jsonl", rows)
        write_json(staging / "acquisition-plan.json", plan)
        write_json(staging / "acquisition-audit.json", audit)
        metadata = {key: value for key, value in manifest.items() if key not in ("files", "digest")}
        metadata.update(original_view_digest=manifest["digest"], acquisition_plan_digest=plan["digest"],
                        reference_training_annotations_sha256=audit["reference_training_annotations_sha256"],
                        supervision="observed-plus-simulated-published-training-annotation-acquisition")
        return publish(staging, output_root, metadata)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
