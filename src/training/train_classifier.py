"""Train an RGB tile classifier from an explicit geographically split CSV manifest."""

import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.models.baseline_classifier import build_classifier
from src.training.classification_loops import run_epoch

NORMALIZATION = {"mean": [.485, .456, .406], "std": [.229, .224, .225]}


def image_transform(image_size=224, training=False):
    steps = [transforms.Resize((image_size, image_size))]
    if training:
        steps += [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
    return transforms.Compose(steps + [transforms.ToTensor(), transforms.Normalize(**NORMALIZATION)])


def read_manifest(path):
    path = Path(path).resolve()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"path", "label", "split", "group"} <= set(reader.fieldnames or []):
            raise ValueError("Manifest CSV needs path,label,split,group columns; group identifies a geographic area/scene.")
        rows = list(reader)
    groups, hashes = {}, {}
    for row in rows:
        if row["split"] not in {"train", "val", "test"} or not row["label"] or not row["group"]:
            raise ValueError("Every row needs a label, geographic group and train/val/test split.")
        image_path = (path.parent / row["path"]).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Training image not found: {image_path}")
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            raise ValueError("Training expects RGB PNG/JPEG tiles. Prepare and review multiband raster tiles first.")
        row["path"] = str(image_path)
        previous = groups.setdefault(row["group"], row["split"])
        if previous != row["split"]:
            raise ValueError(f"Geographic leakage: group {row['group']!r} appears in multiple splits.")
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        previous = hashes.setdefault(digest, row["split"])
        if previous != row["split"]:
            raise ValueError("Identical image files appear in multiple splits.")
    if not rows or not all(any(r["split"] == split for r in rows) for split in ("train", "val")):
        raise ValueError("Provide nonempty train and val splits; test is optional but recommended.")
    classes = sorted({r["label"] for r in rows if r["split"] == "train"})
    if len(classes) < 2 or any(r["label"] not in classes for r in rows):
        raise ValueError("Training must contain at least two classes and every evaluation class.")
    return rows, classes


class ManifestDataset(Dataset):
    def __init__(self, rows, classes, split, image_size):
        self.rows = [row for row in rows if row["split"] == split]
        self.class_to_index = {name: index for index, name in enumerate(classes)}
        self.transform = image_transform(image_size, split == "train")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.class_to_index[row["label"]]


def train_classifier(manifest, output, epochs=10, batch_size=16, learning_rate=.001,
                     device="cuda", image_size=224, seed=42):
    if epochs < 1 or batch_size < 2 or learning_rate <= 0 or image_size < 64:
        raise ValueError("Use epochs >= 1, batch_size >= 2, learning_rate > 0 and image_size >= 64.")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    rows, classes = read_manifest(manifest)
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty output directory to preserve existing training runs.")
    output.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    datasets = {split: ManifestDataset(rows, classes, split, image_size) for split in ("train", "val", "test")}
    if len(datasets["train"]) < 2:
        raise ValueError("Training needs at least two examples.")
    loaders = {split: DataLoader(dataset, batch_size=batch_size, shuffle=split == "train", num_workers=0)
               for split, dataset in datasets.items() if len(dataset)}
    model = build_classifier(len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history, best_loss = [], float("inf")
    config = {"architecture": "resnet18", "classes": classes, "image_size": image_size,
              "normalization": NORMALIZATION, "seed": seed, "epochs": epochs, "batch_size": batch_size,
              "learning_rate": learning_rate, "initialization": "random; no pretrained weights",
              "manifest": str(Path(manifest).resolve()), "split_counts": {s: len(d) for s, d in datasets.items()}}
    (output / "training_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    with (output / "resolved_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "split", "group"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    for epoch in range(1, epochs+1):
        train_metrics = run_epoch(model, loaders["train"], device, optimizer)
        val_metrics = run_epoch(model, loaders["val"], device)
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        print(json.dumps(record), flush=True)
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            torch.save({"model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "config": config, "epoch": epoch, "val": val_metrics}, output / "best.pt")
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    report = {"selected_epoch": checkpoint["epoch"], "best_val_loss": best_loss}
    for split in ("val", "test"):
        if split not in loaders:
            continue
        targets, predictions = [], []
        with torch.inference_mode():
            for images, labels in loaders[split]:
                predictions.extend(model(images.to(device)).argmax(1).cpu().tolist())
                targets.extend(labels.tolist())
        report[split] = {"classification_report": classification_report(targets, predictions,
            labels=list(range(len(classes))), target_names=classes, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(targets, predictions, labels=list(range(len(classes)))).tolist()}
    (output / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"checkpoint": str(output / "best.pt"), "evaluation": report}


def classify_image(checkpoint_path, image_path, device="cuda"):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    if config["architecture"] != "resnet18":
        raise ValueError("Unsupported classifier architecture.")
    model = build_classifier(len(config["classes"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    with Image.open(image_path) as image:
        tensor = image_transform(config["image_size"])(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        scores = model(tensor).softmax(1)[0].cpu().tolist()
    ranking = sorted(zip(config["classes"], scores), key=lambda pair: pair[1], reverse=True)
    return {"predicted_class": ranking[0][0], "scores": dict(ranking),
            "note": "Whole-image class prediction; scores are not calibrated probabilities or map identities."}
