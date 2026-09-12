"""Real GPU smoke checks; synthetic classification data is NOT an accuracy benchmark."""

import csv
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from PIL import Image

from src.geo.providers import resolve_feature
from src.inference.segmentation import segment_image
from src.inference.change_detection import compare_images
from src.training.train_classifier import train_classifier, classify_image
from src.utils.configuration import PROJECT_ROOT


def main():
    output = PROJECT_ROOT / "outputs/logs" / ("app_verification_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    output.mkdir(parents=True)
    rng = np.random.default_rng(42)
    rows = []
    for split in ("train", "val", "test"):
        for label in ("synthetic_green", "synthetic_red"):
            for index in range(2):
                pixels = rng.integers(0, 30, (64,64,3), dtype=np.uint8)
                pixels[:, :, 1 if label.endswith("green") else 0] += 150
                path = output / f"{split}_{label}_{index}.png"
                Image.fromarray(pixels).save(path)
                rows.append({"path": path.name, "label": label, "split": split, "group": f"{split}_scene"})
    manifest = output / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    trained = train_classifier(manifest, output / "classifier", epochs=2, batch_size=2, image_size=64)
    classified = classify_image(trained["checkpoint"], output / rows[-1]["path"])
    image = PROJECT_ROOT / "data/raw/examples/uc_berkeley.tif"
    segmented = segment_image(image, enrich=True)
    compared = compare_images(image, image)
    assert compared["metadata"]["added_pixels"] == 0 and compared["metadata"]["removed_pixels"] == 0
    selected = resolve_feature("W", 23733659)
    assert selected["features"] and selected["features"][0]["geometry"]["type"] in {"Polygon", "MultiPolygon"}
    report = {"synthetic_training_completed": True, "checkpoint_reload": classified,
              "segmentation": {"run_id": segmented["run_id"], **segmented["metadata"]},
              "spatial_matches": sum(bool(f["properties"].get("map_evidence")) for f in segmented["geojson"]["features"]),
              "same_image_change_test": compared["metadata"],
              "live_address_footprint": selected["features"][0]["properties"],
              "accuracy_benchmark": False}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"verification_output": str(output), "report": report}, indent=2))


if __name__ == "__main__":
    main()
