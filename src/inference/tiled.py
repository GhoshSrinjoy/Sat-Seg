"""Complete selected-area inference with bounded windows and seam merging."""

import json

import numpy as np
import rasterio
from affine import Affine
from PIL import Image, ImageOps, ImageDraw
from rasterio.features import shapes, rasterize
from rasterio.transform import from_bounds
from rasterio.warp import transform_geom
from shapely.affinity import affine_transform
from shapely.geometry import shape, mapping, box
from shapely.ops import unary_union

from src.geo.imagery import fetch_view, window_offsets
from src.geo.providers import validate_bbox, feature_collection, geohash_properties
from src.inference.segmentation import predict_masks, choose_rgb, image_from_pixels, new_run, save_preview
from src.utils.configuration import load_config
from src.inference.runtime import gpu_work


def read_full_image(path, bands=None, bounds=None, max_pixels=None):
    limit = max_pixels or load_config()["imagery"]["max_pixels"]
    profile = None
    if str(path).lower().endswith((".tif", ".tiff")):
        with rasterio.open(path) as source:
            if source.width*source.height > limit:
                raise ValueError(f"Image exceeds the {limit:,}-pixel interactive limit. Crop the area first.")
            pixels = source.read(choose_rgb(source, bands), masked=True)
            image, valid = image_from_pixels(pixels)
            valid &= source.dataset_mask() > 0
            if source.crs:
                profile = source.profile.copy()
    else:
        with Image.open(path) as opened:
            if opened.width*opened.height > limit:
                raise ValueError(f"Image exceeds the {limit:,}-pixel interactive limit. Crop or resize it first.")
            image = ImageOps.exif_transpose(opened).convert("RGBA")
            valid = np.array(image.getchannel("A")) > 0
            image = image.convert("RGB")
    if bounds:
        if profile:
            raise ValueError("This image already has georeferencing; do not override it with entered bounds.")
        profile = {"crs": "EPSG:4326", "transform": from_bounds(*validate_bbox(bounds), image.width, image.height),
                   "width": image.width, "height": image.height}
    if not valid.any():
        raise ValueError("Image has no valid pixels.")
    return image, valid, profile, {"georeferenced": profile is not None, "width": image.width, "height": image.height,
                                   "source_file": str(path).replace("\\", "/").split("/")[-1], "coverage": "full image"}


def merge_instances(instances, overlap):
    """Merge same-concept instances representing the same object across windows.

    Intersection over the smaller object handles an object truncated by a tile
    edge. Disjoint neighbours and different visual concepts stay separate.
    """
    merged = []
    for item in sorted(instances, key=lambda i: i["score"], reverse=True):
        item = dict(item)
        candidates = []
        for index, other in enumerate(merged):
            if other["concept"] != item["concept"]:
                continue
            a, b = item["geometry"], other["geometry"]
            if not a.intersects(b):
                continue
            smaller = min(a.area, b.area)
            if smaller and a.intersection(b).area / smaller >= overlap:
                candidates.append(index)
        if candidates:
            item["geometry"] = unary_union([item["geometry"]] + [merged[i]["geometry"] for i in candidates])
            item["score"] = max([item["score"]] + [merged[i]["score"] for i in candidates])
            for index in reversed(candidates):
                merged.pop(index)
        merged.append(item)
    return merged


def annotate(image, pixel_features, map_features=()):
    output = image.copy()
    draw = ImageDraw.Draw(output)
    properties = {f["properties"].get("object_id"): f["properties"] for f in map_features}
    for feature in pixel_features:
        p = {**feature["properties"], **properties.get(feature["properties"].get("object_id"), {})}
        if p.get("rejected"):
            continue
        geometry = shape(feature["geometry"])
        parts = list(geometry.geoms) if geometry.geom_type == "MultiPolygon" else [geometry]
        for part in parts:
            if part.geom_type == "Polygon":
                draw.line(list(part.exterior.coords), fill=(30, 240, 170), width=2)
        anchor = geometry.representative_point()
        label = p.get("reviewed_label") or p.get("label") or p.get("visual_class", "Object")
        bbox = draw.textbbox((anchor.x, anchor.y), label)
        draw.rectangle(bbox, fill=(15, 42, 50))
        draw.text((anchor.x, anchor.y), label, fill=(255, 255, 255))
    return output


def infer_instances(image, valid, concepts, config, progress=None):
    sam = config["sam3"]
    offsets = [(x, y) for y in window_offsets(image.height, sam["tile_size"], sam["tile_overlap"])
               for x in window_offsets(image.width, sam["tile_size"], sam["tile_overlap"])]
    if len(offsets) > config["imagery"]["max_tiles"]:
        raise ValueError("Too many inference windows. Choose a smaller image or area.")
    instances = []
    for index, (x, y) in enumerate(offsets):
        tile = image.crop((x, y, min(x+sam["tile_size"], image.width), min(y+sam["tile_size"], image.height)))
        tile_valid = valid[y:y+tile.height, x:x+tile.width]
        for concept_index, concept in enumerate(concepts):
            if progress:
                progress(f"Finding {concept} · window {index+1}/{len(offsets)}",
                         .35 + .5*(index*len(concepts)+concept_index)/(len(offsets)*len(concepts)))
            if not tile_valid.any():
                continue
            masks, scores = predict_masks(tile, tile_valid, concept, sam["threshold"], sam)
            for mask, score in zip(masks, scores):
                parts = [shape(g) for g, value in shapes(mask.astype("uint8"), mask=mask,
                         transform=Affine.translation(x, y)) if value == 1]
                if parts:
                    instances.append({"geometry": unary_union(parts), "score": float(score), "concept": concept})
                if len(instances) > 10000:
                    raise ValueError("Too many detections. Draw a smaller box or increase the detection threshold.")
    if progress:
        progress("Merging overlapping detections", .87)
    return merge_instances(instances, sam["merge_overlap"])


def analyse_image(image, valid, profile, metadata, prompt, config, enrich=True, progress=None, mapped_sites=None):
    concepts = list(dict.fromkeys(p.strip() for p in prompt.split(",") if p.strip()))
    if not concepts or len(concepts) > 4 or len(prompt) > 200:
        raise ValueError("Enter one to four comma-separated visual concepts (up to 200 characters).")
    with gpu_work():
        instances = infer_instances(image, valid, concepts, config, progress)
    clip = box(0, 0, image.width, image.height)
    features, image_features, polygons = [], [], []
    sites = [shape(f["geometry"]) for f in mapped_sites["features"]] if mapped_sites else []
    for item in instances:
        geometry = item["geometry"].intersection(clip)
        if geometry.is_empty:
            continue
        geographic = None
        if profile:
            t = profile["transform"]
            projected = affine_transform(geometry, [t.a, t.b, t.d, t.e, t.c, t.f])
            geographic = transform_geom(profile["crs"], "EPSG:4326", mapping(projected), precision=9)
            if sites and not any(shape(geographic).intersects(site) for site in sites):
                continue
        properties = {"object_id": len(image_features)+1, "visual_class": item["concept"],
                      "label": f"{item['concept'].capitalize()} {len(image_features)+1}", "score": item["score"],
                      "area_pixels": round(geometry.area), "source": "local SAM 3 prediction",
                      "name": None, "address": None, "match_status": "unmatched"}
        polygons.append(geometry)
        image_features.append({"type": "Feature", "geometry": mapping(geometry), "properties": dict(properties)})
        if profile:
            features.append({"type": "Feature", "geometry": geographic,
                             "properties": {**properties, **geohash_properties(geographic)}})
    attribution = metadata.get("source", {}).get("attribution", "User-provided imagery · local SAM 3")
    collection = feature_collection(features, attribution=attribution)
    warnings = metadata.setdefault("warnings", [])
    if (enrich or mapped_sites) and features:
        if progress:
            progress("Matching names and addresses from map records", .92)
        from src.geo.providers import query_features
        from src.geo.matching import attach_map_evidence
        extent = unary_union([shape(f["geometry"]) for f in features]).bounds
        try:
            collection = attach_map_evidence(collection, mapped_sites or query_features(list(extent), "all"))
            collection["attribution"] = attribution + " · © OpenStreetMap contributors (ODbL)"
        except (RuntimeError, ValueError) as error:
            warnings.append(f"Detections are ready; map evidence is unavailable: {error}")
    union = rasterize([(mapping(p), 1) for p in polygons], out_shape=(image.height, image.width),
                      fill=0, dtype="uint8").astype(bool) if polygons else np.zeros(valid.shape, bool)
    run_id, directory = new_run()
    map_bounds = save_preview(image, union[None], profile, directory)
    # Preserve no-data transparency in the exact analysed Web Mercator mosaic.
    if profile and str(profile["crs"]).upper() == "EPSG:3857":
        rgba = image.convert("RGBA")
        rgba.putalpha(Image.fromarray((valid*255).astype("uint8")))
        rgba.save(directory / "map_image.png")
    metadata.update(objects=len(image_features), prompt=prompt, threshold=config["sam3"]["threshold"],
                    mask_threshold=config["sam3"]["mask_threshold"], georeferenced=profile is not None,
                    model={k: config["sam3"][k] for k in ("model_id", "revision", "dtype")})
    pixel_collection = {"type": "FeatureCollection", "coordinate_space": "image_pixels", "features": image_features}
    (directory / "features.geojson").write_text(json.dumps(collection), encoding="utf-8")
    (directory / "image_features.json").write_text(json.dumps(pixel_collection), encoding="utf-8")
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    Image.fromarray((union*255).astype("uint8")).save(directory / "mask.png")
    annotate(image, image_features, collection["features"]).save(directory / "annotated.png")
    return {"run_id": run_id, "geojson": collection, "image_features": pixel_collection, "metadata": metadata,
            "map_bounds": map_bounds, "preview": f"/results/{run_id}/overlay.png",
            "annotated_preview": f"/results/{run_id}/annotated.png",
            "map_image": f"/results/{run_id}/map_image.png" if map_bounds else None}


def segment_view(bounds, prompt=None, enrich=None, config=None, progress=None):
    config = config or load_config()
    image, valid, profile, metadata = fetch_view(bounds, config, progress)
    return analyse_image(image, valid, profile, metadata, prompt or config["sam3"]["prompt"], config,
                         config["map"]["enrich"] if enrich is None else enrich, progress)


def segment_sites(bounds, category, tag=None, config=None, progress=None):
    from src.geo.providers import query_features
    config = config or load_config()
    if progress:
        progress("Locating mapped sites", .02)
    sites = query_features(bounds, category, tag)
    if not sites["features"]:
        raise ValueError("No matching mapped sites were found in this area.")
    image, valid, profile, metadata = fetch_view(bounds, config, progress)
    metadata["scope"] = f"Buildings intersecting mapped {category} records; site identity comes from map evidence"
    return analyse_image(image, valid, profile, metadata, "building", config, True, progress, sites)


def segment_full_image(path, prompt=None, threshold=None, bands=None, bounds=None, enrich=False, progress=None):
    config = load_config()
    if threshold is not None:
        config["sam3"]["threshold"] = threshold
    image, valid, profile, metadata = read_full_image(path, bands, bounds, config["imagery"]["max_pixels"])
    return analyse_image(image, valid, profile, metadata, prompt or config["sam3"]["prompt"], config, enrich, progress)
