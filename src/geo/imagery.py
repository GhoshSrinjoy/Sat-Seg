"""Small-area orthophoto acquisition on an explicit Web Mercator pixel grid.

Built-ins: public-domain USGS NAIP (CONUS) and Berlin DOP 2025 (DL-DE-Zero).
The same export services supply browser tiles and inference mosaics. No OSM or
Esri World Imagery basemap is captured or downloaded for inference.
"""

import hashlib
import io
import json
import math
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import requests
from PIL import Image, UnidentifiedImageError
from rasterio.transform import from_bounds, array_bounds
from rasterio.warp import transform_bounds
from affine import Affine

from src.geo.providers import validate_bbox
from src.utils.configuration import PROJECT_ROOT, load_config

PROVIDERS = {
    "naip": {"id": "naip", "name": "USGS NAIP · contiguous US", "kind": "arcgis",
             "url": "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer/exportImage",
             "bounds": [-125, 24, -66, 50], "resolution_m": .6, "acquisition": "Mosaic; acquisition dates vary by location",
             "attribution": "USGS / USDA NAIP · public domain",
             "source_url": "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer"},
    "berlin": {"id": "berlin", "name": "Berlin orthophotos · spring 2025", "kind": "wms",
               "url": "https://gdi.berlin.de/services/wms/dop_2025_fruehjahr", "layer": "dop_2025",
               "bounds": [13.08, 52.32, 13.77, 52.69], "resolution_m": .2, "acquisition": "Spring 2025",
               "attribution": "Geoportal Berlin / DOP 2025 · DL-DE-Zero 2.0",
               "source_url": "https://daten.berlin.de/datensaetze/digitale-farbige-orthophotos-2025-dop20rgbi-wms-6529de5a"},
}
_cache_lock = threading.Lock()
_fetch_slots = threading.BoundedSemaphore(3)


def source_for(bounds, settings=None):
    settings = settings or load_config()["imagery"]
    west, south, east, north = validate_bbox(bounds)
    if south < -85.05112878 or north > 85.05112878:
        raise ValueError("Satellite views support latitudes between -85.05 and 85.05 degrees.")
    selection = settings["provider"]
    if selection == "custom_wms":
        if not settings["custom_url"] or not settings["custom_layer"]:
            raise ValueError("Configure the custom WMS URL and imagery layer in Settings.")
        return {"id": selection, "name": "Custom orthophoto WMS", "kind": "wms", "url": settings["custom_url"],
                "layer": settings["custom_layer"], "token": settings["custom_token"],
                "resolution_m": settings["resolution_m"], "acquisition": "Not supplied by custom provider",
                "attribution": settings["custom_attribution"], "source_url": settings["custom_url"]}
    for key in ([selection] if selection != "auto" else ["berlin", "naip"]):
        source = PROVIDERS[key]
        w, s, e, n = source["bounds"]
        if w <= west < east <= e and s <= south < north <= n:
            return dict(source)
    raise ValueError("No built-in imagery coverage for this area. Select an area in Berlin or the contiguous US, "
                     "or configure an imagery WMS for this region in Settings.")


def window_offsets(length, size, overlap):
    if length <= size:
        return [0]
    offsets = list(range(0, length-size+1, size-overlap))
    if offsets[-1] != length-size:
        offsets.append(length-size)
    return offsets


def plan_view(bounds, config=None):
    config = config or load_config()
    bounds = validate_bbox(bounds, config["map"]["max_area_km2"])
    source = source_for(bounds, config["imagery"])
    extent = transform_bounds("EPSG:4326", "EPSG:3857", *bounds)
    # Web Mercator metres are enlarged by sec(latitude); the preference is ground metres.
    ground_resolution = max(config["imagery"]["resolution_m"], source["resolution_m"])
    pixel_size = ground_resolution / math.cos(math.radians((bounds[1]+bounds[3])/2))
    width = max(1, math.ceil((extent[2]-extent[0])/pixel_size))
    height = max(1, math.ceil((extent[3]-extent[1])/pixel_size))
    if width < 64 or height < 64:
        raise ValueError("Draw a wider area: both image dimensions must contain at least 64 pixels.")
    if width*height > config["imagery"]["max_pixels"]:
        raise ValueError(f"This area needs {width*height:,} pixels at {ground_resolution:g} m/pixel. "
                         "Draw a smaller box or choose a coarser imagery resolution in Settings.")
    sam = config["sam3"]
    tiles = len(window_offsets(width, sam["tile_size"], sam["tile_overlap"])) * len(
        window_offsets(height, sam["tile_size"], sam["tile_overlap"]))
    if tiles > config["imagery"]["max_tiles"]:
        raise ValueError(f"This selection needs {tiles} inference windows; limit is {config['imagery']['max_tiles']}. Draw a smaller box.")
    return {"bounds": bounds, "extent": list(extent), "width": width, "height": height,
            "resolution_m": ground_resolution, "inference_tiles": tiles,
            "source": {k: v for k, v in source.items() if k not in {"token", "url"}}}


def export_image(source, extent, width, height, settings):
    if source["kind"] == "arcgis":
        params = {"f": "image", "bbox": ",".join(map(str, extent)), "bboxSR": 3857, "imageSR": 3857,
                  "size": f"{width},{height}", "format": "png32", "transparent": "true",
                  "adjustAspectRatio": "false", "interpolation": "RSP_BilinearInterpolation",
                  "renderingRule": json.dumps({"rasterFunction": "NaturalColor"})}
    else:
        params = {"service": "WMS", "version": "1.1.1", "request": "GetMap", "layers": source["layer"],
                  "styles": "", "srs": "EPSG:3857", "bbox": ",".join(map(str, extent)),
                  "width": width, "height": height, "format": "image/png", "transparent": "true"}
        if source.get("token"):
            params["token"] = source["token"]
    key = hashlib.sha256(json.dumps([source["url"], params], sort_keys=True).encode()).hexdigest()
    cache = PROJECT_ROOT / "data/interim/map_cache/imagery"
    path = cache / (key + ".png")
    with _cache_lock:
        if path.is_file() and time.time()-path.stat().st_mtime < settings["cache_seconds"]:
            try:
                with Image.open(path) as opened:
                    return opened.convert("RGBA")
            except (UnidentifiedImageError, OSError):
                path.unlink(missing_ok=True)
    with _fetch_slots:
        try:
            with requests.get(source["url"], params=params, timeout=(5, 45), stream=True,
                              headers={"User-Agent": "sat-clas/0.3 interactive imagery"}) as response:
                response.raise_for_status()
                content = bytearray()
                for chunk in response.iter_content(65536):
                    content.extend(chunk)
                    if len(content) > 16*1024*1024:
                        raise ValueError("Imagery provider response exceeds 16 MB per tile.")
            with Image.open(io.BytesIO(content)) as opened:
                if opened.size != (width, height):
                    raise ValueError("Imagery provider returned unexpected image dimensions; map placement would be incorrect.")
                result = opened.convert("RGBA")
        except (requests.RequestException, UnidentifiedImageError) as error:
            raise RuntimeError(f"{source['name']} did not return imagery. Retry or check the provider in Settings.") from error
    if settings["cache_seconds"]:
        with _cache_lock:
            cache.mkdir(parents=True, exist_ok=True)
            temporary = cache / (key + "." + uuid.uuid4().hex + ".tmp")
            try:
                result.save(temporary, format="PNG")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
            files = sorted(cache.glob("*.png"), key=lambda p: p.stat().st_mtime)
            total = sum(p.stat().st_size for p in files)
            for entry in files:
                if total <= settings["cache_mb"] * 1024**2:
                    break
                total -= entry.stat().st_size
                entry.unlink(missing_ok=True)
    return result


def fetch_view(bounds, config=None, progress=None):
    config = config or load_config()
    plan = plan_view(bounds, config)
    source = source_for(bounds, config["imagery"])
    width, height = plan["width"], plan["height"]
    transform = from_bounds(*plan["extent"], width, height)
    mosaic = Image.new("RGBA", (width, height))
    columns, rows = range(0, width, 1024), range(0, height, 1024)
    total = len(columns)*len(rows)
    done = 0
    for row in rows:
        for col in columns:
            if progress:
                progress(f"Fetching imagery {done+1}/{total}", .05 + .3*done/total)
            w, h = min(1024, width-col), min(1024, height-row)
            extent = array_bounds(h, w, transform * Affine.translation(col, row))
            tile = export_image(source, extent, w, h, config["imagery"])
            mosaic.paste(tile, (col, row))
            done += 1
    valid = np.array(mosaic.getchannel("A")) > 0
    if valid.mean() < .1:
        raise ValueError("The imagery source has no usable coverage in most of this selection. Choose another area or provider.")
    profile = {"crs": "EPSG:3857", "transform": transform, "width": width, "height": height}
    plan.update(georeferenced=True, coverage_fraction=float(valid.mean()),
                warnings=[] if valid.all() else ["Some selected pixels have no imagery; detections exclude those pixels."])
    return mosaic.convert("RGB"), valid, profile, plan


def browser_tile(z, x, y):
    if not 8 <= z <= 20 or not (0 <= x < 2**z and 0 <= y < 2**z):
        raise ValueError("Invalid imagery tile coordinates.")
    half = 20037508.342789244
    span = 2*half / 2**z
    extent = [-half+x*span, half-(y+1)*span, -half+(x+1)*span, half-y*span]
    bounds = transform_bounds("EPSG:3857", "EPSG:4326", *extent)
    settings = load_config()["imagery"]
    try:
        source = source_for(bounds, settings)
    except ValueError:
        return Image.new("RGBA", (256, 256))
    return export_image(source, extent, 256, 256, settings)
