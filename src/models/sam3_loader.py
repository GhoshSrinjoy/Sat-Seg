"""Inspect and load the existing Hugging Face SAM 3 cache without network access."""

import argparse
import json
from pathlib import Path

from src.utils.configuration import load_config


def find_snapshot(settings: dict) -> Path:
    """Resolve the configured revision within a Hugging Face cache directory."""
    repository = Path(settings["cache_dir"]) / ("models--" + settings["model_id"].replace("/", "--"))
    revision = settings.get("revision", "main")
    ref_file = repository / "refs" / revision
    if ref_file.is_file():
        revision = ref_file.read_text(encoding="utf-8").strip()
    return repository / "snapshots" / revision


def inspect_snapshot(snapshot: Path) -> dict:
    """Inspect configs and weight shards; existence alone is not a load test."""
    report = {"snapshot": str(snapshot), "complete": False, "missing": []}
    required = ["config.json", "tokenizer_config.json"]
    report["missing"] = [name for name in required if not (snapshot / name).is_file()]
    if not any((snapshot / name).is_file() for name in ("tokenizer.json", "vocab.json")):
        report["missing"].append("tokenizer.json or vocab.json")
    if not any((snapshot / name).is_file() for name in ("processor_config.json", "preprocessor_config.json")):
        report["missing"].append("processor_config.json or preprocessor_config.json")
    index_file = snapshot / "model.safetensors.index.json"
    if index_file.is_file():
        index = json.loads(index_file.read_text(encoding="utf-8"))
        weights = sorted(set(index["weight_map"].values()))
    else:
        weights = ["model.safetensors"]
    weight_files = []
    for name in weights:
        path = snapshot / name
        if not path.is_file() or path.stat().st_size == 0:
            report["missing"].append(name)
        else:
            weight_files.append({"name": name, "bytes": path.stat().st_size,
                                 "resolved_path": str(path.resolve())})
    config_file = snapshot / "config.json"
    if config_file.is_file():
        config = json.loads(config_file.read_text(encoding="utf-8"))
        report["model_type"] = config.get("model_type")
        report["architectures"] = config.get("architectures")
    report["weights"] = weight_files
    report["complete"] = not report["missing"]
    return report


def load_sam3(settings: dict | None = None):
    """Load the image model and processor strictly from local cached files.

    Transformers supports extracting Sam3Model from the composite video
    checkpoint using its detector_config and detector_model weight prefix.
    Missing/mismatched image weights are treated as errors.
    """
    import torch
    from transformers import Sam3Config, Sam3Model, Sam3Processor

    settings = settings or load_config()["sam3"]
    snapshot = find_snapshot(settings)
    report = inspect_snapshot(snapshot)
    if not report["complete"]:
        raise FileNotFoundError(f"Incomplete SAM 3 cache: {report['missing']}. Run --inspect first.")
    if report.get("model_type") not in {"sam3", "sam3_video"}:
        raise ValueError(f"Unsupported checkpoint model_type: {report.get('model_type')}")
    device = settings.get("device", "cuda")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("The configuration requests CUDA, but CUDA is unavailable.")
    dtype_name = settings.get("dtype", "float32")
    if dtype_name not in {"float32", "float16", "bfloat16"}:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    checkpoint_config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    image_config = Sam3Config.from_dict(
        checkpoint_config["detector_config"] if report["model_type"] == "sam3_video" else checkpoint_config
    )
    model, info = Sam3Model.from_pretrained(
        str(snapshot), local_files_only=True, dtype=getattr(torch, dtype_name),
        config=image_config, output_loading_info=True,
    )
    failures = {key: info[key] for key in ("missing_keys", "mismatched_keys", "error_msgs") if info.get(key)}
    if failures:
        raise RuntimeError(f"SAM image detector weights were not loaded completely: {failures}")
    model = model.to(device).eval()
    processor = Sam3Processor.from_pretrained(str(snapshot), local_files_only=True)
    return model, processor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--inspect", action="store_true")
    group.add_argument("--load", action="store_true")
    group.add_argument("--smoke-inference", action="store_true")
    group.add_argument("--download-missing", action="store_true")
    args = parser.parse_args()
    settings = load_config(args.config)["sam3"]
    report = inspect_snapshot(find_snapshot(settings))
    print(json.dumps(report, indent=2), flush=True)
    if args.download_missing:
        if report["complete"]:
            print("Complete cache already exists; no model files downloaded.")
        else:
            from huggingface_hub import snapshot_download
            print("Downloading missing SAM 3 files to the configured cache.", flush=True)
            snapshot_download(
                repo_id=settings["model_id"], revision=settings.get("revision", "main"),
                cache_dir=settings["cache_dir"],
                allow_patterns=["*.json", "*.safetensors", "*.txt", "LICENSE*", "README.md"],
            )
            report = inspect_snapshot(find_snapshot(settings))
            print(json.dumps(report, indent=2))
    if not report["complete"]:
        raise SystemExit("SAM 3 cache is incomplete; no automatic download was performed.")
    if args.load or args.smoke_inference:
        model, processor = load_sam3(settings)
        print(f"SAM 3 image detector loaded on {model.device}; dtype={model.dtype}", flush=True)
        if args.smoke_inference:
            import time
            import torch
            from PIL import Image, ImageDraw
            image = Image.new("RGB", (128, 128), (40, 100, 45))
            ImageDraw.Draw(image).rectangle((35, 35, 90, 90), fill=(170, 170, 170))
            inputs = processor(images=image, text="building", return_tensors="pt").to(model.device)
            if model.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                outputs = model(**inputs)
            if model.device.type == "cuda":
                torch.cuda.synchronize()
            results = processor.post_process_instance_segmentation(
                outputs, threshold=settings.get("threshold", 0.5),
                mask_threshold=0.5, target_sizes=inputs["original_sizes"].tolist(),
            )[0]
            if not torch.isfinite(outputs.pred_masks).all():
                raise RuntimeError("SAM returned nonfinite mask logits.")
            print(json.dumps({
                "smoke_inference": "PASS", "image": "synthetic; not an accuracy benchmark",
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "objects": len(results["masks"]),
                "peak_allocated_gpu_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3)
                if model.device.type == "cuda" else None,
            }, indent=2))


if __name__ == "__main__":
    main()
