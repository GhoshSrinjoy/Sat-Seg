"""Bounded local SAM inference, vector footprints and correctly projected previews."""

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageOps
from rasterio.enums import ColorInterp, Resampling
from rasterio.features import shapes
from rasterio.transform import from_bounds, array_bounds
from rasterio.warp import transform_geom, transform_bounds, calculate_default_transform, reproject
from rasterio.windows import Window
from shapely.geometry import shape, mapping
from shapely.ops import unary_union

from src.data.preprocessing import rgb_preview
from src.data.raster_loading import read_raster
from src.geo.providers import feature_collection, geohash_properties, validate_bbox
from src.utils.configuration import PROJECT_ROOT, load_config
from src.inference.runtime import gpu_work

_model_lock = threading.Lock()
_model_pair = None
_model_key = None


def unload_sam():
    global _model_pair, _model_key
    with _model_lock:
        if _model_pair is not None:
            import gc
            import torch
            _model_pair = None
            _model_key = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def model_status():
    return {"loaded": _model_pair is not None, "configuration": _model_key}


def choose_rgb(dataset, bands=None):
    if bands is not None:
        if len(bands) != 3 or any(i not in dataset.indexes for i in bands):
            raise ValueError("RGB bands must be three valid one-based indexes.")
        return bands
    interpretations = list(dataset.colorinterp)
    if all(color in interpretations for color in (ColorInterp.red, ColorInterp.green, ColorInterp.blue)):
        return [interpretations.index(color)+1 for color in (ColorInterp.red, ColorInterp.green, ColorInterp.blue)]
    raise ValueError("This raster does not declare RGB bands. Enter their one-based indexes, e.g. 4,3,2.")


def image_from_pixels(pixels):
    valid = ~np.ma.getmaskarray(pixels).any(axis=0) & np.isfinite(pixels.filled(0)).all(axis=0)
    if not valid.any():
        raise ValueError("The selected image window contains no valid pixels.")
    if pixels.dtype == np.uint8:
        rgb = np.moveaxis(pixels.filled(0), 0, -1).copy()
        rgb[~valid] = 0
        image = Image.fromarray(rgb)
    else:
        image = rgb_preview(pixels)
    return image, valid


def read_image(path, bands=None, bounds=None, window_origin=None, size=1008):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    if not 64 <= size <= 1008:
        raise ValueError("Window size must be 64–1008 pixels for this interactive interface.")
    profile = None
    if path.suffix.lower() in {".tif", ".tiff"}:
        with rasterio.open(path) as dataset:
            selected_bands = choose_rgb(dataset, bands)
            width, height = dataset.width, dataset.height
        col, row = window_origin or (max(0, (width-size)//2), max(0, (height-size)//2))
        if col < 0 or row < 0 or col >= width or row >= height or int(col) != col or int(row) != row:
            raise ValueError("The pixel window origin must lie inside the raster.")
        pixels, profile = read_raster(path, selected_bands, Window(col, row, min(size, width-col), min(size, height-row)))
        image, valid = image_from_pixels(pixels)
        if not profile.get("crs"):
            profile = None
        metadata = {"rgb_bands": selected_bands, "window": [col, row, image.width, image.height]}
    else:
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            raise ValueError("Supported images: GeoTIFF, TIFF, PNG, JPEG.")
        with Image.open(path) as opened:
            original = ImageOps.exif_transpose(opened)
            width, height = original.size
            col, row = window_origin or (max(0, (width-size)//2), max(0, (height-size)//2))
            if not (0 <= col < width and 0 <= row < height) or int(col) != col or int(row) != row:
                raise ValueError("The pixel window origin must lie inside the image.")
            image = original.crop((col, row, min(col+size, width), min(row+size, height))).convert("RGB")
        valid = np.ones((image.height, image.width), dtype=bool)
        metadata = {"rgb_bands": [1, 2, 3], "window": [col, row, image.width, image.height]}
        if bounds:
            from affine import Affine
            west, south, east, north = validate_bbox(bounds)
            profile = {"crs": rasterio.CRS.from_epsg(4326), "width": image.width, "height": image.height,
                       "transform": from_bounds(west, south, east, north, width, height) * Affine.translation(col, row)}
            metadata["georeferencing"] = "User-provided bounds of the full north-up, orthorectified image"
    metadata["source_file"] = path.name
    metadata["georeferenced"] = profile is not None
    return image, valid, profile, metadata


def predict_masks(image, valid, prompt, threshold=None, settings=None):
    global _model_pair, _model_key
    from src.models.sam3_loader import load_sam3
    config = load_config()
    settings = settings or config["sam3"]
    threshold = settings["threshold"] if threshold is None else threshold
    if not prompt.strip() or len(prompt) > 200 or not 0 <= threshold <= 1:
        raise ValueError("Use a visual concept of 1–200 characters and threshold between zero and one.")
    # Serialize GPU work and load once per server process.
    key = {k: settings[k] for k in ("model_id", "cache_dir", "revision", "device", "dtype")}
    with gpu_work():
        if _model_key != key:
            unload_sam()
        with _model_lock:
            if _model_pair is None:
                from src.language.ollama import unload_ollama
                unload_ollama(config["ollama"])
                _model_pair = load_sam3(settings)
                _model_key = key
            return _predict_locked(image, valid, prompt, threshold, settings)


def _predict_locked(image, valid, prompt, threshold, settings):
    import torch
    model, processor = _model_pair
    inputs = processor(images=image, text=prompt.strip(), return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model(**inputs)
        result = processor.post_process_instance_segmentation(outputs, threshold=threshold,
            mask_threshold=settings["mask_threshold"], target_sizes=inputs["original_sizes"].tolist())[0]
    masks = result["masks"].detach().cpu().numpy().astype(bool)
    scores = result["scores"].detach().cpu().numpy()
    if len(masks) == 0:
        masks = np.zeros((0, image.height, image.width), dtype=bool)
    masks &= valid[None, :, :]
    keep = masks.reshape(len(masks), image.height*image.width).sum(axis=1) >= settings["min_object_pixels"]
    return masks[keep], scores[keep]


def mask_geometry(mask, profile):
    parts = [shape(geometry) for geometry, value in shapes(mask.astype("uint8"), mask=mask,
             transform=profile["transform"]) if value == 1]
    if not parts:
        return None
    geometry = unary_union(parts)
    if not geometry.is_valid:
        geometry = geometry.buffer(0)
    if geometry.is_empty:
        return None
    return transform_geom(profile["crs"], "EPSG:4326", mapping(geometry), precision=9)


def vectorize_masks(masks, scores, profile, prompt):
    features = []
    if profile is None:
        return feature_collection([], warning="Image has no georeferencing; results are in the image preview only.")
    for index, (mask, score) in enumerate(zip(masks, scores)):
        geometry = mask_geometry(mask, profile)
        if geometry:
            features.append({"type": "Feature", "geometry": geometry, "properties": {
                "object_id": index+1, "visual_class": prompt, "score": float(score),
                "area_pixels": int(mask.sum()), "source": "local SAM 3 prediction", "name": None,
                "address": None, "match_status": "unmatched", **geohash_properties(geometry)}})
    return feature_collection(features)


def new_run():
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    directory = PROJECT_ROOT / "outputs/predictions/map_app" / run_id
    directory.mkdir(parents=True)
    return run_id, directory


def save_preview(image, masks, profile, directory):
    image.save(directory / "image.png")
    union = masks.any(axis=0) if len(masks) else np.zeros((image.height, image.width), dtype=bool)
    overlay = np.array(image).copy()
    overlay[union] = (.55*overlay[union] + .45*np.array([50, 240, 160])).astype("uint8")
    Image.fromarray(overlay).save(directory / "overlay.png")
    bounds = None
    if profile:
        extent = array_bounds(image.height, image.width, profile["transform"])
        # Warp to Web Mercator so Leaflet's rectangular overlay aligns with the basemap.
        transform, width, height = calculate_default_transform(profile["crs"], "EPSG:3857",
            image.width, image.height, *extent)
        if width * height > 4_000_000:
            from affine import Affine
            factor = (4_000_000 / (width*height)) ** .5
            smaller_width, smaller_height = max(1, int(width*factor)), max(1, int(height*factor))
            transform = transform * Affine.scale(width/smaller_width, height/smaller_height)
            width, height = smaller_width, smaller_height
        destination = np.zeros((3, height, width), dtype="uint8")
        for index in range(3):
            reproject(np.array(image)[:, :, index], destination[index], src_transform=profile["transform"],
                src_crs=profile["crs"], dst_transform=transform, dst_crs="EPSG:3857", resampling=Resampling.bilinear)
        Image.fromarray(np.moveaxis(destination, 0, -1)).save(directory / "map_image.png")
        west, south, east, north = transform_bounds("EPSG:3857", "EPSG:4326", *array_bounds(height, width, transform))
        bounds = [[south, west], [north, east]]
    return bounds


def segment_image(path, prompt=None, threshold=None, bands=None, bounds=None, enrich=False, window_origin=None):
    settings = load_config()["sam3"]
    prompt = prompt or settings["prompt"]
    threshold = settings["threshold"] if threshold is None else threshold
    image, valid, profile, metadata = read_image(path, bands, bounds, window_origin)
    masks, scores = predict_masks(image, valid, prompt, threshold)
    collection = vectorize_masks(masks, scores, profile, prompt)
    warnings = []
    if enrich and profile and collection["features"]:
        from src.geo.providers import query_features
        from src.geo.matching import attach_map_evidence
        west, south, east, north = transform_bounds(profile["crs"], "EPSG:4326",
            *array_bounds(image.height, image.width, profile["transform"]))
        try:
            collection = attach_map_evidence(collection, query_features([west, south, east, north], "all"))
        except (RuntimeError, ValueError) as error:
            warnings.append(f"Segmentation succeeded; map enrichment unavailable: {error}")
    run_id, directory = new_run()
    map_bounds = save_preview(image, masks, profile, directory)
    np.savez_compressed(directory / "instances.npz", masks=masks, scores=scores)
    metadata.update(prompt=prompt, threshold=threshold, objects=len(masks), warnings=warnings)
    (directory / "features.geojson").write_text(json.dumps(collection), encoding="utf-8")
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"run_id": run_id, "geojson": collection, "metadata": metadata, "map_bounds": map_bounds,
            "preview": f"/results/{run_id}/overlay.png", "map_image": f"/results/{run_id}/map_image.png" if map_bounds else None}
