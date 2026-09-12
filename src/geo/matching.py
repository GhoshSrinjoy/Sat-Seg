"""Attach spatial evidence without inventing a building's identity or use."""

from copy import deepcopy

from pyproj import CRS, Transformer
from shapely.geometry import shape
from shapely.ops import transform
from shapely.strtree import STRtree


def attach_map_evidence(detections, mapped_features):
    result = deepcopy(detections)
    records = [f for f in mapped_features.get("features", []) if f.get("geometry")]
    geometries = [shape(f["geometry"]) for f in records]
    tree = STRtree(geometries)
    for feature in result["features"]:
        polygon = shape(feature["geometry"])
        point = polygon.representative_point()
        # A local equal-area projection makes polygon overlap meaningful worldwide.
        local = CRS.from_proj4(f"+proj=laea +lat_0={point.y} +lon_0={point.x} +datum=WGS84 +units=m")
        project = Transformer.from_crs("EPSG:4326", local, always_xy=True).transform
        projected = transform(project, polygon)
        evidence = []
        for index in tree.query(polygon, predicate="intersects"):
            geometry = geometries[index]
            p = deepcopy(records[index]["properties"])
            if geometry.geom_type in {"Polygon", "MultiPolygon"}:
                other = transform(project, geometry)
                intersection = projected.intersection(other).area
                union = projected.union(other).area
                overlap = intersection / union if union else 0
                if overlap < .05:
                    continue
                p.update(spatial_relation="overlapping polygon", intersection_over_union=round(overlap, 4))
            elif geometry.geom_type == "Point" and polygon.covers(geometry):
                p.update(spatial_relation="contained point; may be a tenant or entrance")
            else:
                continue
            evidence.append(p)
        evidence.sort(key=lambda p: p.get("intersection_over_union", 0), reverse=True)
        properties = feature.setdefault("properties", {})
        properties["map_evidence"] = evidence
        properties["match_status"] = "spatial candidates; identity unverified" if evidence else "unmatched"
        properties["name"] = None
        properties["address"] = None
        # A single dominant mapped footprint is a useful tentative label, not ground truth.
        strong = [p for p in evidence if p.get("intersection_over_union", 0) >= .5]
        if len(strong) == 1:
            properties["candidate_name"] = strong[0].get("name")
            properties["candidate_address"] = strong[0].get("address")
        properties["label_note"] = "Names/addresses come from map evidence, not SAM; review spatial matches."
    return result
