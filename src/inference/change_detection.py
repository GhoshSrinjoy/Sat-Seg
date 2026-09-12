"""Align ordinary images or georasters, then compare masks on valid overlap."""

import json

import cv2
import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import ColorInterp, Resampling
from rasterio.vrt import WarpedVRT
from rasterio.features import shapes
from shapely.geometry import shape

from src.geo.providers import feature_collection, geohash_properties
from src.geo.imagery import window_offsets
from src.inference.segmentation import (choose_rgb, image_from_pixels, predict_masks,
                                        mask_geometry, new_run, save_preview)
from src.inference.tiled import read_full_image
from src.inference.runtime import gpu_work
from src.utils.configuration import load_config


def registration(before, after, before_valid, after_valid, mode="auto", points=None):
    """Warp before to after; manual points are normalized [[bx,by],[ax,ay]] pairs."""
    a, b = np.array(before), np.array(after)
    if mode == "same_grid":
        if before.size != after.size:
            raise ValueError("Already aligned images must have the same dimensions. Use automatic or manual alignment.")
        return before, before_valid & after_valid, {"method": "User-declared same pixel grid"}
    if mode == "auto" and a.shape == b.shape and np.array_equal(a, b):
        return before, before_valid & after_valid, {"method": "Identical image grids", "registration_error_px": 0}
    if mode == "manual":
        values = np.asarray(points, dtype="float64")
        if values.ndim != 3 or values.shape[1:] != (2, 2) or not 4 <= len(values) <= 30 or not np.isfinite(values).all():
            raise ValueError("Add at least four matching point pairs in the two image previews.")
        if (values < 0).any() or (values > 1).any():
            raise ValueError("Manual points must lie inside the images.")
        source = values[:, 0] * np.array(before.size)
        target = values[:, 1] * np.array(after.size)
        matrix, inliers = cv2.findHomography(source, target, cv2.RANSAC, 4)
    elif mode == "auto":
        def features(image, valid):
            factor = min(1, 1600/max(image.size))
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            size = (round(image.width*factor), round(image.height*factor))
            gray = cv2.resize(gray, size)
            mask = cv2.resize((valid*255).astype("uint8"), size, interpolation=cv2.INTER_NEAREST)
            keys, descriptors = cv2.ORB_create(nfeatures=6000).detectAndCompute(gray, mask)
            return keys, descriptors, factor
        ka, da, sa = features(before, before_valid)
        kb, db, sb = features(after, after_valid)
        if da is None or db is None:
            raise ValueError("Automatic alignment found too few landmarks. Add manual point pairs or select Already aligned.")
        matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
        good = [pair[0] for pair in matches if len(pair) == 2 and pair[0].distance < .72*pair[1].distance]
        if len(good) < 12:
            raise ValueError("Automatic alignment could not match enough landmarks. Use manual alignment.")
        source = np.float64([ka[m.queryIdx].pt for m in good]) / sa
        target = np.float64([kb[m.trainIdx].pt for m in good]) / sb
        matrix, inliers = cv2.findHomography(source, target, cv2.RANSAC, 3)
        if inliers is None or inliers.sum() < 10 or inliers.mean() < .4:
            raise ValueError("Alignment is unreliable. Add well-spaced manual points before comparing.")
    else:
        raise ValueError("Alignment must be auto, same_grid or manual.")
    if matrix is None or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-8:
        raise ValueError("Alignment is degenerate. Choose well-spaced landmarks in both images.")
    corners = np.float32([[[0, 0], [before.width, 0], [before.width, before.height], [0, before.height]]])
    warped_corners = cv2.perspectiveTransform(corners, matrix)[0]
    ratio = abs(cv2.contourArea(warped_corners)) / (after.width*after.height)
    if not cv2.isContourConvex(warped_corners) or not .1 < ratio < 10:
        raise ValueError("Alignment distorts the image excessively. Choose another image pair or better landmarks.")
    projected = cv2.perspectiveTransform(source.reshape(1, -1, 2), matrix)[0]
    keep = inliers.ravel().astype(bool)
    error = float(np.median(np.linalg.norm(projected[keep]-target[keep], axis=1)))
    coverage = cv2.contourArea(cv2.convexHull(target[keep].astype("float32"))) / (after.width*after.height)
    if coverage < .015:
        raise ValueError("Alignment landmarks are too close together. Spread manual points across the image.")
    aligned = cv2.warpPerspective(a, matrix, after.size, flags=cv2.INTER_LINEAR)
    valid = cv2.warpPerspective((before_valid*255).astype("uint8"), matrix, after.size,
                               flags=cv2.INTER_NEAREST) > 0
    valid &= after_valid
    if valid.mean() < .15:
        raise ValueError("The images have less than 15% usable overlap after alignment.")
    return Image.fromarray(aligned), valid, {"method": "Manual landmarks" if mode == "manual" else "Automatic landmark matching",
                "registration_error_px": round(error, 3), "matched_landmarks": int(keep.sum()),
                "transform_before_to_after": matrix.tolist()}


def prepare_pair(before_path, after_path, bands=None, alignment=None, bounds=None, points=None):
    config = load_config()
    alignment = alignment or config["comparison"]["alignment"]
    after, after_valid, profile, metadata = read_full_image(after_path, bands, bounds)
    before, before_valid, before_profile, _ = read_full_image(before_path, bands)
    if before_profile and profile and alignment == "auto":
        with rasterio.open(before_path) as source:
            with WarpedVRT(source, crs=profile["crs"], transform=profile["transform"], width=after.width,
                           height=after.height, resampling=Resampling.bilinear,
                           add_alpha=ColorInterp.alpha not in source.colorinterp) as aligned:
                pixels = aligned.read(choose_rgb(source, bands), masked=True)
                alpha_valid = aligned.dataset_mask() > 0
        before, before_valid = image_from_pixels(pixels)
        valid = before_valid & after_valid & alpha_valid
        details = {"method": "Georeferenced common grid"}
    else:
        before, valid, details = registration(before, after, before_valid, after_valid, alignment, points)
    if not valid.any():
        raise ValueError("The images have no valid overlapping pixels.")
    metadata.update(alignment=details, valid_overlap_pixels=int(valid.sum()),
                    overlap_fraction=round(float(valid.mean()), 4))
    return before, after, valid, profile, metadata


def alignment_preview(before_path, after_path, bands=None, alignment=None, bounds=None, points=None):
    before, after, valid, profile, metadata = prepare_pair(before_path, after_path, bands, alignment, bounds, points)
    run_id, directory = new_run()
    before.save(directory / "aligned_before.png")
    after.save(directory / "aligned_after.png")
    Image.fromarray((valid*255).astype("uint8")).save(directory / "overlap.png")
    pixels = (.5*np.array(before)+.5*np.array(after)).astype("uint8")
    pixels[~valid] = (40, 45, 55)
    Image.fromarray(pixels).save(directory / "alignment.png")
    return {"run_id": run_id, "before_preview": f"/results/{run_id}/aligned_before.png",
            "after_preview": f"/results/{run_id}/aligned_after.png", "preview": f"/results/{run_id}/alignment.png",
            "metadata": metadata}


def _union(image, valid, prompt, threshold, config, progress=None, label="image"):
    sam = config["sam3"]
    offsets = [(x, y) for y in window_offsets(image.height, sam["tile_size"], sam["tile_overlap"])
               for x in window_offsets(image.width, sam["tile_size"], sam["tile_overlap"])]
    if len(offsets) > config["imagery"]["max_tiles"]:
        raise ValueError("Too many comparison windows; use smaller images.")
    result = np.zeros(valid.shape, bool)
    for index, (x, y) in enumerate(offsets):
        if progress:
            progress(f"Comparing {label} · window {index+1}/{len(offsets)}",
                     (.15 if label == "before" else .5) + .3*index/len(offsets))
        tile = image.crop((x, y, min(image.width, x+sam["tile_size"]), min(image.height, y+sam["tile_size"])))
        mask = valid[y:y+tile.height, x:x+tile.width]
        if mask.any():
            masks, _ = predict_masks(tile, mask, prompt, threshold)
            result[y:y+tile.height, x:x+tile.width] |= masks.any(axis=0)
    return result


def remove_small(mask, minimum):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype("uint8"), 8)
    keep = np.zeros(count, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= minimum
    return keep[labels]


def compare_images(before_path, after_path, prompt=None, threshold=None, bands=None, alignment=None,
                   bounds=None, points=None, before_date="", after_date="", progress=None):
    config = load_config()
    prompt = prompt or config["sam3"]["prompt"]
    threshold = config["sam3"]["threshold"] if threshold is None else threshold
    if progress:
        progress("Aligning images", .05)
    before, after, valid, profile, metadata = prepare_pair(before_path, after_path, bands, alignment, bounds, points)
    with gpu_work():
        before_union = _union(before, valid, prompt, threshold, config, progress, "before")
        after_union = _union(after, valid, prompt, threshold, config, progress, "after")
    tolerance = config["comparison"]["edge_tolerance"]
    kernel = np.ones((2*tolerance+1, 2*tolerance+1), "uint8")
    before_expanded = cv2.dilate(before_union.astype("uint8"), kernel).astype(bool)
    after_expanded = cv2.dilate(after_union.astype("uint8"), kernel).astype(bool)
    minimum = config["comparison"]["min_change_pixels"]
    added = remove_small(after_union & ~before_expanded & valid, minimum)
    removed = remove_small(before_union & ~after_expanded & valid, minimum)
    features, image_features = [], []
    for label, mask in (("added candidate", added), ("removed candidate", removed)):
        for geometry, value in shapes(mask.astype("uint8"), mask=mask):
            if value != 1:
                continue
            pixels = shape(geometry)
            props = {"change": label, "visual_class": prompt, "area_pixels": round(pixels.area),
                     "source": "difference between SAM 3 masks on a common grid"}
            image_features.append({"type": "Feature", "geometry": geometry, "properties": props})
        if profile:
            geometry = mask_geometry(mask, profile)
            if geometry:
                features.append({"type": "Feature", "geometry": geometry, "properties": {
                    "change": label, "visual_class": prompt, "area_pixels": int(mask.sum()),
                    "source": "difference between SAM 3 masks on a common grid", **geohash_properties(geometry)}})
    run_id, directory = new_run()
    map_bounds = save_preview(after, after_union[None], profile, directory)
    before.save(directory / "before.png")
    overlay = np.array(after).copy()
    overlay[added] = (.4*overlay[added]+.6*np.array([45, 230, 125])).astype("uint8")
    overlay[removed] = (.4*overlay[removed]+.6*np.array([255, 80, 90])).astype("uint8")
    overlay[~valid] = (40, 45, 55)
    Image.fromarray(overlay).save(directory / "changes.png")
    collection = feature_collection(features, attribution="User-provided before/after images · local SAM 3")
    metadata.update(prompt=prompt, threshold=threshold, before_date=before_date, after_date=after_date,
                    added_pixels=int(added.sum()), removed_pixels=int(removed.sum()),
                    edge_tolerance=tolerance, min_change_pixels=minimum,
                    interpretation="Candidate changes. Review image alignment, seasons and detection errors before assigning a cause.")
    (directory / "features.geojson").write_text(json.dumps(collection), encoding="utf-8")
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    pixel_collection = {"type": "FeatureCollection", "coordinate_space": "image_pixels", "features": image_features}
    (directory / "image_features.json").write_text(json.dumps(pixel_collection), encoding="utf-8")
    return {"run_id": run_id, "geojson": collection, "image_features": pixel_collection, "metadata": metadata,
            "map_bounds": map_bounds, "preview": f"/results/{run_id}/changes.png",
            "before_preview": f"/results/{run_id}/before.png", "after_preview": f"/results/{run_id}/image.png",
            "map_image": f"/results/{run_id}/map_image.png" if map_bounds else None}
