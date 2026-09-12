"""Offline regression tests for spatial identity, parsing and data leakage."""

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_origin
from shapely.geometry import box, mapping, Point

from src.geo.matching import attach_map_evidence
from src.geo.providers import feature_collection, geohash_area, validate_bbox
from src.inference.segmentation import read_image, vectorize_masks
from src.language.query_parser import parse_query
from src.training.train_classifier import read_manifest


def feature(geometry, **properties):
    return {"type": "Feature", "geometry": mapping(geometry), "properties": properties}


class GeographicWorkflows(unittest.TestCase):
    def test_question_and_address_parsing(self):
        plan = parse_query("mark all hospitals in Berlin")
        self.assertEqual((plan.intent, plan.category, plan.area), ("features", "hospitals", "Berlin"))
        plan = parse_query("where are schools in this area?")
        self.assertEqual((plan.intent, plan.area), ("features", None))
        self.assertEqual(parse_query("where is house number 5 Starmansrteen").query, "5 Starmansrteen")
        self.assertEqual(parse_query("mark hiking routes").category, "hiking")
        self.assertEqual(parse_query("directions from Berlin to Potsdam").intent, "route")
        self.assertEqual(parse_query("52.52,13.405").intent, "coordinates")

    def test_geohash_and_area_validation(self):
        self.assertEqual(geohash_area("u33dc1")["features"][0]["geometry"]["type"], "Polygon")
        for bounds in ([170, 0, -170, 10], [-180, -80, 180, 80], [0, 0, float("nan"), 1]):
            with self.assertRaises(ValueError):
                validate_bbox(bounds, 25)

    def test_spatial_match_keeps_tenants_and_unknown_identity(self):
        footprint = box(13.4, 52.5, 13.4002, 52.5002)
        detections = feature_collection([feature(footprint, visual_class="building")])
        mapped = feature_collection([feature(footprint, name="Mapped building", address="5 Example Road"),
            feature(Point(13.4001, 52.5001), name="Tenant A"), feature(Point(13.40015, 52.5001), name="Tenant B")])
        result = attach_map_evidence(detections, mapped)["features"][0]["properties"]
        self.assertIsNone(result["name"])
        self.assertEqual(result["candidate_name"], "Mapped building")
        self.assertEqual(len(result["map_evidence"]), 3)
        self.assertNotIn("map_evidence", detections["features"][0]["properties"])
        unmatched = attach_map_evidence(detections, feature_collection())["features"][0]["properties"]
        self.assertEqual(unmatched["match_status"], "unmatched")

    def test_raster_window_uses_its_own_transform(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.tif"
            transform = from_origin(13.4, 52.5, .00001, .00001)
            with rasterio.open(path, "w", driver="GTiff", width=150, height=150, count=3,
                               dtype="uint8", crs="EPSG:4326", transform=transform) as dst:
                dst.write(np.ones((3, 150, 150), dtype="uint8")*150)
                dst.colorinterp = (rasterio.enums.ColorInterp.red, rasterio.enums.ColorInterp.green, rasterio.enums.ColorInterp.blue)
            image, valid, profile, metadata = read_image(path, window_origin=[20, 30], size=64)
            self.assertEqual(image.size, (64, 64))
            self.assertEqual(profile["transform"] * (0, 0), transform * (20, 30))
            masks = np.zeros((1, 64, 64), dtype=bool)
            masks[0, 10:20, 10:20] = True
            result = vectorize_masks(masks, [.9], profile, "building")
            self.assertEqual(len(result["features"]), 1)
            self.assertEqual(result["features"][0]["properties"]["area_pixels"], 100)
            image_path = Path(directory) / "unlocated.png"
            image.save(image_path)
            self.assertIsNone(read_image(image_path, size=64)[2])

    def test_training_rejects_geographic_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(4):
                Image.new("RGB", (64, 64), (index*30, 80, 20)).save(root / f"{index}.png")
            rows = [dict(path=f"{i}.png", label=str(i%2), split="train" if i<2 else "val", group="same_scene") for i in range(4)]
            manifest = root / "manifest.csv"
            with manifest.open("w", newline="") as handle:
                writer=csv.DictWriter(handle, fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "Geographic leakage"):
                read_manifest(manifest)

    def test_http_input_errors_and_coordinate_search(self):
        from fastapi.testclient import TestClient
        from src.interface.app import app
        client = TestClient(app)
        self.assertEqual(client.get("/").status_code, 200)
        self.assertEqual(client.post("/api/features", json={"bounds": [-180,-80,180,80]}).status_code, 400)
        result = client.post("/api/search", json={"query": "52.52,13.405"}).json()
        self.assertEqual(result["geojson"]["features"][0]["geometry"]["coordinates"], [13.405,52.52])
        self.assertEqual(client.post("/api/segment", files={"image": ("bad.txt", b"not an image")}).status_code, 400)

    def test_comparison_respects_existing_alpha_and_nodata(self):
        from src.inference.change_detection import compare_images
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rgba.tif"
            pixels = np.ones((4, 80, 80), dtype="uint8")*150
            pixels[3] = 255
            pixels[3, :10] = 0
            with rasterio.open(path, "w", driver="GTiff", width=80, height=80, count=4,
                               dtype="uint8", crs="EPSG:4326", transform=from_origin(13.4, 52.5, .00001, .00001)) as dst:
                dst.write(pixels)
                dst.colorinterp = (rasterio.enums.ColorInterp.red, rasterio.enums.ColorInterp.green,
                                   rasterio.enums.ColorInterp.blue, rasterio.enums.ColorInterp.alpha)
            captured_valid = []
            def empty_prediction(image, valid, prompt, threshold):
                captured_valid.append(valid)
                return np.zeros((0, image.height, image.width), dtype=bool), np.array([])
            with patch("src.inference.change_detection.predict_masks", side_effect=empty_prediction), \
                 patch("src.inference.change_detection.new_run", return_value=("test", Path(directory))), \
                 patch("src.inference.change_detection.save_preview", return_value=None):
                result = compare_images(path, path)
            self.assertEqual(result["metadata"]["valid_overlap_pixels"], 70*80)
            self.assertEqual(result["metadata"]["added_pixels"], 0)
            self.assertFalse(captured_valid[0][:10].any())


if __name__ == "__main__":
    unittest.main()
