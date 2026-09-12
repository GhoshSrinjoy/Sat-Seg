"""Small, cached interactive Photon/Overpass queries; no planet-wide downloads."""

import hashlib
import json
import math
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import osm2geojson
import pygeohash
import requests
from shapely.geometry import shape, mapping, box

from src.utils.configuration import PROJECT_ROOT, load_config

ATTRIBUTION = "© OpenStreetMap contributors (ODbL)"
CATEGORIES = {
    "buildings": ['["building"]'],
    "hospitals": ['["amenity"="hospital"]', '["healthcare"="hospital"]'],
    "schools": ['["amenity"="school"]'],
    "parks": ['["leisure"="park"]'],
    "water": ['["natural"="water"]', '["waterway"]'],
    "roads": ['["highway"]'],
    "hiking": ['["route"="hiking"]', '["route"="foot"]'],
    "restaurants": ['["amenity"="restaurant"]'],
    "shops": ['["shop"]'],
    "trees": ['["natural"="tree"]', '["natural"="wood"]', '["landuse"="forest"]'],
    "all": ['["building"]', '["amenity"]', '["shop"]', '["leisure"]',
            '["natural"]', '["waterway"]', '["highway"]', '["landuse"]',
            '["railway"]', '["tourism"]', '["route"]', '["addr:housenumber"]'],
}
_network_lock = threading.Lock()
_last_request = 0.0


def feature_collection(features=(), **metadata):
    return {"type": "FeatureCollection", "features": list(features),
            "attribution": ATTRIBUTION, **metadata}


def validate_bbox(bounds, max_area_km2=None):
    if bounds is None or len(bounds) != 4:
        raise ValueError("Choose a map area: west, south, east, north.")
    west, south, east, north = map(float, bounds)
    if not all(math.isfinite(v) for v in (west, south, east, north)):
        raise ValueError("Map coordinates must be finite.")
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError("Invalid bounds. Split areas crossing the date line into two queries.")
    area = (east-west) * 111.32 * math.cos(math.radians((south+north)/2)) * (north-south) * 111.32
    if max_area_km2 is not None and area > max_area_km2:
        raise ValueError(f"Zoom in: this area is about {area:.1f} km²; interactive limit is {max_area_km2:g} km².")
    return [west, south, east, north]


def bounds_around(lon, lat, radius_m=150):
    if not (-180 <= lon <= 180 and -89.9 < lat < 89.9):
        raise ValueError("Coordinates are outside the supported map range.")
    dy = radius_m / 111320
    dx = dy / max(math.cos(math.radians(lat)), .001)
    return [max(-180, lon-dx), max(-90, lat-dy), min(180, lon+dx), min(90, lat+dy)]


def geohash_properties(geometry):
    point = shape(geometry).representative_point()
    return {"geohash": pygeohash.encode(point.y, point.x, precision=9)}


def geohash_area(value):
    value = value.strip().lower()
    if not 1 <= len(value) <= 12 or any(c not in "0123456789bcdefghjkmnpqrstuvwxyz" for c in value):
        raise ValueError("Enter a valid geohash of 1–12 characters.")
    lat, lon, lat_error, lon_error = pygeohash.decode_exactly(value)
    return feature_collection([{"type": "Feature", "geometry": mapping(box(
        lon-lon_error, lat-lat_error, lon+lon_error, lat+lat_error)),
        "properties": {"name": value, "geohash": value, "source": "geohash cell"}}])


def _request_json(url, *, params=None, data=None):
    global _last_request
    settings = load_config().get("map", {})
    key = hashlib.sha256(json.dumps([url, params, data], sort_keys=True).encode()).hexdigest()
    cache = PROJECT_ROOT / "data/interim/map_cache" / f"{key}.json"
    with _network_lock:
        if cache.is_file() and time.time()-cache.stat().st_mtime < settings.get("cache_seconds", 86400):
            return json.loads(cache.read_text(encoding="utf-8"))
        time.sleep(max(0, 1.1-(time.monotonic()-_last_request)))
        _last_request = time.monotonic()
        for attempt in range(2):
            try:
                response = requests.request("POST" if data else "GET", url, params=params, data=data,
                    headers={"User-Agent": settings.get("user_agent", "sat-clas/0.2 local-research")},
                    timeout=(10, 45))
                response.raise_for_status()
                result = response.json()
                break
            except (requests.RequestException, ValueError) as error:
                status = getattr(getattr(error, "response", None), "status_code", None)
                if attempt == 0 and (status in {429, 502, 503, 504} or isinstance(error, requests.Timeout)):
                    time.sleep(2)
                    _last_request = time.monotonic()
                    continue
                raise RuntimeError(f"Map provider unavailable ({url}): {error}") from error
        if result.get("remark"):
            raise RuntimeError(f"Map provider returned an incomplete result: {result['remark']}")
        result["sat_clas_fetched_at"] = datetime.now(timezone.utc).isoformat()
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_suffix(".part")
        temporary.write_text(json.dumps(result), encoding="utf-8")
        temporary.replace(cache)
        return result


def search_address(query, bounds=None):
    if not query.strip():
        raise ValueError("Enter an address or place name, preferably including city and country.")
    params = {"q": query, "limit": 6}
    if bounds:
        west, south, east, north = validate_bbox(bounds)
        params.update(lon=(west+east)/2, lat=(south+north)/2)
    url = load_config().get("map", {}).get("photon_url", "https://photon.komoot.io").rstrip("/")
    result = _request_json(url + "/api/", params=params)
    for feature in result.get("features", []):
        p = feature.setdefault("properties", {})
        p.update(source="OpenStreetMap via Photon", fetched_at=result["sat_clas_fetched_at"],
                 match_status="search candidate; select to inspect", **geohash_properties(feature["geometry"]))
        p["address"] = ", ".join(str(p[k]) for k in
            ("housenumber", "street", "postcode", "city", "state", "country") if p.get(k)) or None
    return feature_collection(result.get("features", []), query=query)


def _overpass(query):
    settings = load_config().get("map", {})
    raw = _request_json(settings.get("overpass_url", "https://overpass-api.de/api/interpreter"), data={"data": query})
    converted = osm2geojson.json2geojson(raw)
    features = []
    for feature in converted.get("features", []):
        if not feature.get("geometry"):
            continue
        geometry = shape(feature["geometry"])
        if geometry.is_empty:
            continue
        p = feature.get("properties", {})
        tags = p.get("tags", {})
        osm_type, osm_id = p.get("type"), p.get("id")
        feature["properties"] = {
            "osm_type": osm_type, "osm_id": osm_id, "tags": tags,
            "name": tags.get("name"), "address": tags.get("addr:full") or ", ".join(
                tags[k] for k in ("addr:housenumber", "addr:street", "addr:postcode", "addr:city", "addr:country")
                if tags.get(k)) or None,
            "source": "OpenStreetMap via Overpass", "fetched_at": raw["sat_clas_fetched_at"],
            "source_url": f"https://www.openstreetmap.org/{osm_type}/{osm_id}",
            "match_status": "mapped feature", **geohash_properties(feature["geometry"]),
        }
        features.append(feature)
    return feature_collection(features, fetched_at=raw["sat_clas_fetched_at"],
                              completeness="Available OSM records; coverage is not guaranteed.")


def query_features(bounds, category="buildings", tag=None):
    import re
    settings = load_config().get("map", {})
    west, south, east, north = validate_bbox(bounds, settings.get("max_area_km2", 25))
    if tag:
        if not re.fullmatch(r"[A-Za-z0-9_:.-]+=[A-Za-z0-9_: .;-]+", tag):
            raise ValueError("Custom tag must be key=value, e.g. amenity=pharmacy.")
        key, value = tag.split("=", 1)
        filters = [f'[{json.dumps(key)}={json.dumps(value)}]']
    elif category in CATEGORIES:
        filters = CATEGORIES[category]
    else:
        raise ValueError(f"Choose a supported category or an OSM key=value tag: {list(CATEGORIES)}")
    bounds_text = f"{south},{west},{north},{east}"
    selections = "".join(f"nwr{condition}({bounds_text});" for condition in filters)
    # out geom includes relation members; osm2geojson assembles multipolygons/routes.
    return _overpass(f"[out:json][timeout:40];({selections});out body geom;")


def resolve_feature(osm_type, osm_id):
    osm_type = {"N": "node", "W": "way", "R": "relation"}.get(osm_type, osm_type)
    if osm_type not in {"node", "way", "relation"} or not isinstance(osm_id, int) or osm_id < 1:
        raise ValueError("Invalid OpenStreetMap feature identifier.")
    result = _overpass(f"[out:json][timeout:30];{osm_type}({osm_id});out body geom;")
    # An address point may sit inside a building. Keep evidence separate from identity.
    if len(result["features"]) == 1 and result["features"][0]["geometry"]["type"] == "Point":
        point = shape(result["features"][0]["geometry"])
        try:
            buildings = query_features(bounds_around(point.x, point.y), "buildings")
            for feature in buildings["features"]:
                polygon = shape(feature["geometry"])
                if polygon.geom_type in {"Polygon", "MultiPolygon"} and polygon.covers(point):
                    feature["properties"]["match_status"] = "contains selected address/place point"
                    feature["properties"]["selected_point"] = result["features"][0]["properties"]
                    result["features"].append(feature)
        except RuntimeError as error:
            result["warning"] = str(error)
    return result
