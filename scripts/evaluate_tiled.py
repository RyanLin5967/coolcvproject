"""Freeze and run one construction slicing cohort; never tune against its scores."""
from datetime import UTC, datetime
from pathlib import Path

from coveragecv.artifacts import digest, file_digest, read_json, write_json
from coveragecv.training.tiled import PROTOCOL, device_parity_smoke, evaluate_tiled

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / "artifacts/tiled/construction/20260917"
    bundle = Path(read_json(ROOT / "artifacts/construction/experiment.json")["complete_bundle"])
    parents = ROOT / "artifacts/gpu/construction/20260917"
    methods = ["naive", "aware", "complete_reference"]
    # Four-image CPU/MPS preflight found material score/threshold differences.
    # Preserve that failed evidence and use the historical evaluator's CPU device.
    device = "cpu"
    sources = [ROOT / "src/coveragecv/training/tiled.py", Path(__file__)]
    binding = {"protocol": PROTOCOL, "device": device, "methods": methods,
               "bundle_digest": read_json(bundle / "manifest.json")["digest"],
               "source_files": {p.relative_to(ROOT).as_posix(): file_digest(p) for p in sources},
               "parents": {method: {"checkpoint_sha256": file_digest(parents / method / "detector.pt"),
                                    "baseline_sha256": file_digest(parents / method / "evaluation.json")}
                           for method in methods}}
    protocol_file = output / "protocol.json"
    if protocol_file.exists():
        if read_json(protocol_file)["binding"] != binding:
            raise ValueError("frozen protocol, code, or checkpoint changed; do not overwrite existing experiment")
    else:
        write_json(protocol_file, {"frozen_at": datetime.now(UTC).isoformat(),
                                   "binding_sha256": digest(binding), "binding": binding})
    print({"frozen_protocol": str(protocol_file.relative_to(ROOT)), "device": device}, flush=True)
    if device != "cpu" and not (output / "device_parity.json").exists():
        print(device_parity_smoke(parents / "aware/detector.pt", bundle, output / "device_parity.json",
                                  device=device), flush=True)
    if (output / "device_parity.json").exists() and not read_json(output / "device_parity.json")["passed"]:
        raise ValueError("previous device parity failed")
    results = []
    for method in methods:
        folder = output / method
        if (folder / "comparison.json").exists():
            result = read_json(folder / "comparison.json")
        else:
            print({"starting": method}, flush=True)
            result = evaluate_tiled(parents / method / "detector.pt", bundle, folder,
                                    baseline_evaluation=parents / method / "evaluation.json", device=device)
        results.append({"method": method, **result})
        write_json(output / "results.json", {"protocol_sha256": file_digest(protocol_file), "runs": results})
        print({"method": method, "full_AP": result["full_frame_metrics"]["AP"]*100,
               "full_NMS_AP": result["full_frame_nms_metrics"]["AP"]*100,
               "tiled_AP": result["tiled_metrics"]["AP"]*100,
               "delta_points": result["tiled_minus_fresh_full_AP_points"]}, flush=True)


if __name__ == "__main__":
    main()
