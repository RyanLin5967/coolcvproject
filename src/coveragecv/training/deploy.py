"""Explicit native-to-Roboflow serving label layout conversion.

RF-DETR 1.10.1's training labels are semantic-first with the reserved slot last.
The observed Roboflow hosted import pipeline consumes a background-first layout.
The export reorders classifier rows only; native checkpoints are never modified.
"""
import re
from pathlib import Path

import torch

from coveragecv.artifacts import file_digest, write_json


def reorder_classifier_rows(state, classes):
    count = len(classes)
    permutation = [count, *range(count)]
    pattern = re.compile(r"^(class_embed|transformer\.enc_out_class_embed\.\d+)\.(weight|bias)$")
    result, changed = dict(state), []
    for name, tensor in state.items():
        if not pattern.match(name):
            continue
        if tensor.shape[0] != count+1:
            raise ValueError("classifier dimensions disagree with the declared ontology")
        result[name] = tensor[permutation].clone()
        changed.append(name)
    if "class_embed.weight" not in changed or "class_embed.bias" not in changed:
        raise ValueError("unsupported checkpoint: expected RF-DETR main classifier parameters")
    if not any(name.startswith("transformer.enc_out_class_embed.") for name in changed):
        raise ValueError("unsupported checkpoint: missing encoder classifiers")
    return result, sorted(changed)


def export_for_hosted(checkpoint: Path, output: Path):
    from argparse import Namespace
    native = torch.load(checkpoint, map_location="cpu", weights_only=True)
    classes = native["class_names"]
    if "background_class83422" in classes:
        raise ValueError("expected a native semantic-first checkpoint, not an already converted export")
    state, changed = reorder_classifier_rows(native["model"], classes)
    args = dict(native["args"])
    args.update(dataset_dir=None, output_dir="output", resume="", class_names=classes)
    output.mkdir(parents=True, exist_ok=True)
    # Match the stock Roboflow export metadata type, while documenting the row permutation separately.
    torch.save({"model": state, "args": Namespace(**args)}, output / "weights.pt")
    (output / "class_names.txt").write_text("\n".join(["background_class83422", *classes])+"\n")
    metadata = {"native_checkpoint_sha256": file_digest(checkpoint),
                "export_checkpoint_sha256": file_digest(output / "weights.pt"), "classes": classes,
                "native_indices": {**{name: i for i, name in enumerate(classes)}, "reserved": len(classes)},
                "hosted_indices": {"reserved": 0, **{name: i+1 for i, name in enumerate(classes)}},
                "converted_parameters": changed, "native_checkpoint_modified": False,
                "compatibility_evidence": "Hosted import shifted native class 1 to class 0; observed 2026-09-17"}
    write_json(output / "export.json", metadata)
    return metadata
