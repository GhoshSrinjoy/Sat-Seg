"""Idempotently add the package/map/training workflow to the exploration notebook."""

from pathlib import Path
import shutil

import nbformat

from src.utils.configuration import PROJECT_ROOT


def main():
    path = PROJECT_ROOT / "notebooks/01_environment_and_raster_exploration.ipynb"
    notebook = nbformat.read(path, as_version=4)
    backup = PROJECT_ROOT / "outputs/logs/exploration_before_package_cells.ipynb"
    if not backup.exists():
        shutil.copy2(path, backup)
    additions = [
        ("package-help", "markdown", """## Use these predictions in the map application

The next cell exports the **existing** masks to WGS84 GeoJSON, including a
geohash per object. It does not run SAM a second time. Open the map interface in
a terminal with `conda run --no-capture-output -n sat_clas python -m src.cli serve`,
then visit **http://127.0.0.1:8765** and import the exported GeoJSON in Explore.

The app also accepts addresses, common questions, coordinates, geohashes, image
uploads, and before/after GeoTIFFs. See `docs/training_and_map_workflows.md`.
"""),
        ("package-export", "code", """from src.inference.segmentation import vectorize_masks

if globals().get("masks") is None or "run_dir" not in globals():
    raise RuntimeError("Run the image, inference and export cells above first.")
map_profile = profile if profile is not None and profile.get("crs") is not None else None
map_features = vectorize_masks(masks, scores, map_profile, PROMPT)
map_output_path = run_dir / "features.geojson"
map_output_path.write_text(json.dumps(map_features, indent=2), encoding="utf-8")
print("Map export:", map_output_path)
print("Georeferenced objects:", len(map_features["features"]))
if map_profile is None:
    print("This image has no map coordinates. The image preview remains available.")
"""),
        ("map-evidence-help", "markdown", """## Optional: attach map names, addresses and tags

Set `FETCH_MAP_EVIDENCE=True` to query the image window using OpenStreetMap.
This sends only the geographic area to the map provider. Names/addresses remain
source-backed candidates; missing records stay unknown. Multiple tenants can
occupy one building. This step uses the existing masks and does not train a model.
"""),
        ("map-evidence", "code", """FETCH_MAP_EVIDENCE = False

if FETCH_MAP_EVIDENCE:
    from rasterio.transform import array_bounds
    from rasterio.warp import transform_bounds
    from src.geo.providers import query_features
    from src.geo.matching import attach_map_evidence

    if map_profile is None:
        raise ValueError("A georeferenced image is required for map-data matching.")
    area_bounds = transform_bounds(map_profile["crs"], "EPSG:4326", *array_bounds(
        image.height, image.width, map_profile["transform"]))
    mapped_records = query_features(list(area_bounds), category="all")
    enriched_features = attach_map_evidence(map_features, mapped_records)
    enriched_path = run_dir / "features_with_map_evidence.geojson"
    enriched_path.write_text(json.dumps(enriched_features, indent=2), encoding="utf-8")
    print("Map evidence export:", enriched_path)
    display(pd.DataFrame([{
        "object_id": f["properties"]["object_id"],
        "candidate_name": f["properties"].get("candidate_name"),
        "candidate_address": f["properties"].get("candidate_address"),
        "map_candidates": len(f["properties"]["map_evidence"]),
    } for f in enriched_features["features"]]))
else:
    print("Map lookup is optional. Enable FETCH_MAP_EVIDENCE or use the app's Imagery tab.")
"""),
        ("training-help", "markdown", """## Train and reload a tile classifier

SAM inference above already uses trained local weights. The following optional
workflow trains a separate **ResNet-18 RGB tile classifier** on your reviewed
labels; it does not fine-tune SAM or learn building addresses.

Create a CSV with `path,label,split,group` columns. Splits are `train`, `val`,
and optionally `test`; groups must identify separated geographic areas/scenes.
See the complete dataset example in `docs/training_and_map_workflows.md`.
No real labeled dataset has been supplied, so this section starts disabled.
"""),
        ("training-settings", "code", """TRAINING_MANIFEST = None  # e.g. Path(r"D:/data/my_dataset/manifest.csv")
TRAINING_EPOCHS = 20
TRAINING_BATCH_SIZE = 16
NEW_TILE_PATH = None  # RGB PNG/JPEG to classify after training
trained_classifier = None
"""),
        ("training-run", "code", """from src.training.train_classifier import train_classifier, classify_image

if TRAINING_MANIFEST is None:
    print("Set TRAINING_MANIFEST to your labeled dataset CSV, then run this cell.")
else:
    training_output = PROJECT_ROOT / "outputs/training" / (
        "notebook_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    trained_classifier = train_classifier(
        manifest=TRAINING_MANIFEST,
        output=training_output,
        epochs=TRAINING_EPOCHS,
        batch_size=TRAINING_BATCH_SIZE,
        device="cuda",
    )
    print("Best checkpoint:", trained_classifier["checkpoint"])
    print("Evaluation:", json.dumps(trained_classifier["evaluation"], indent=2))
"""),
        ("training-predict", "code", """if NEW_TILE_PATH is None:
    print("Set NEW_TILE_PATH to classify a tile after training, or use the app's Training tab.")
elif trained_classifier is None:
    raise RuntimeError("Train a classifier above before predicting a new tile.")
else:
    tile_prediction = classify_image(trained_classifier["checkpoint"], NEW_TILE_PATH)
    print(json.dumps(tile_prediction, indent=2))
"""),
    ]
    ids = {item[0] for item in additions}
    notebook.cells = [cell for cell in notebook.cells if cell.get("id") not in ids]
    # Remove only empty trailing placeholders; preserve user code and outputs.
    while notebook.cells and notebook.cells[-1].cell_type == "code" and not notebook.cells[-1].source.strip():
        notebook.cells.pop()
    for cell_id, kind, source in additions:
        cell = nbformat.v4.new_markdown_cell(source) if kind == "markdown" else nbformat.v4.new_code_cell(source)
        cell["id"] = cell_id
        notebook.cells.append(cell)
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    print(f"Updated {path}: {len(notebook.cells)} cells")


if __name__ == "__main__":
    main()
