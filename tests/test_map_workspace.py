"""Geographic and API regression coverage for the shared map workspace."""

import io
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image
from affine import Affine
from rasterio.transform import from_bounds
from shapely.geometry import box

from src.geo.imagery import fetch_view, plan_view, window_offsets
from src.inference.tiled import merge_instances, infer_instances, read_full_image
from src.inference.change_detection import registration, compare_images
from src.utils.configuration import load_config, save_settings
from src.utils.settings import public_settings


class MapWorkspace(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.environment = patch.dict(os.environ, {"SAT_CLAS_SETTINGS": str(self.root / "settings.yaml")})
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        self.temporary.cleanup()

    def test_settings_persist_validate_and_redact_token(self):
        save_settings({"sam3": {"threshold": .75, "mask_threshold": .65},
                       "imagery": {"custom_token": "private-test-value"}, "ollama": {"model": "local-example"}})
        loaded = load_config()
        self.assertEqual(loaded["sam3"]["threshold"], .75)
        self.assertEqual(loaded["sam3"]["mask_threshold"], .65)
        self.assertNotIn("custom_token", public_settings(loaded)["imagery"])
        self.assertTrue(public_settings(loaded)["imagery"]["has_custom_token"])
        with self.assertRaises(ValueError):
            save_settings({"sam3": {"threshold": 2}})
        self.assertEqual(load_config()["sam3"]["threshold"], .75)
        with self.assertRaises(ValueError):
            save_settings({"imagery": {"provider": "custom_wms", "custom_url": "file:///tmp/data"}})

    def test_imagery_coverage_and_pixel_guards(self):
        cfg = load_config()
        berlin = plan_view([13.41, 52.52, 13.412, 52.521], cfg)
        self.assertEqual(berlin["source"]["id"], "berlin")
        self.assertGreater(berlin["width"], 200)
        self.assertEqual(plan_view([-122.26, 37.871, -122.258, 37.872], cfg)["source"]["id"], "naip")
        for extent in ([10, 48, 10.001, 48.001], [179, 10, -179, 11], [-180, -80, 180, 80]):
            with self.assertRaises(ValueError):
                plan_view(extent, cfg)
        cfg["imagery"]["max_pixels"] = 100
        with self.assertRaisesRegex(ValueError, "pixels"):
            plan_view([13.41, 52.52, 13.412, 52.521], cfg)

    def test_mosaic_preserves_mercator_grid_and_fetches_all_edges(self):
        cfg = load_config()
        extent = [13.41, 52.52, 13.424, 52.527]
        requests = []
        def export(source, bounds, width, height, settings):
            requests.append((bounds, width, height))
            return Image.new("RGBA", (width, height), (60, 100, 120, 255))
        with patch("src.geo.imagery.export_image", side_effect=export):
            image, valid, profile, plan = fetch_view(extent, cfg)
        self.assertGreater(len(requests), 1)
        self.assertEqual(profile["crs"], "EPSG:3857")
        np.testing.assert_allclose(profile["transform"] * (0, 0), [plan["extent"][0], plan["extent"][3]])
        np.testing.assert_allclose(profile["transform"] * image.size, [plan["extent"][2], plan["extent"][1]])
        self.assertEqual(sum(w*h for _, w, h in requests), image.width*image.height)
        self.assertTrue(valid.all())

    def test_tiles_cover_full_selection_and_merge_boundary_object(self):
        cfg = load_config()
        cfg["sam3"].update(tile_size=256, tile_overlap=64, min_object_pixels=1)
        canvas = np.zeros((320, 450, 3), "uint8")
        canvas[80:160, 210:280] = 255  # crosses the first tile boundary
        canvas[285:310, 420:445] = 255  # outside the first tile entirely
        calls = []
        def masks(image, valid, prompt, threshold, settings):
            detected = np.array(image)[:, :, 0] > 100
            count, labels = cv2.connectedComponents(detected.astype("uint8"))
            calls.append(image.size)
            return np.stack([labels == i for i in range(1, count)]) if count > 1 else np.zeros((0, image.height, image.width), bool), np.full(count-1, .9)
        with patch("src.inference.tiled.predict_masks", side_effect=masks):
            result = infer_instances(Image.fromarray(canvas), np.ones(canvas.shape[:2], bool), ["building"], cfg)
        self.assertGreater(len(calls), 1)
        self.assertEqual(len(result), 2)
        self.assertEqual(sorted(round(i["geometry"].area) for i in result), [625, 5600])
        instances = [{"geometry": box(0, 0, 10, 10), "score": .9, "concept": "building"},
                     {"geometry": box(11, 0, 20, 10), "score": .8, "concept": "building"}]
        self.assertEqual(len(merge_instances(instances, .6)), 2)

    def test_uploaded_png_full_extent_and_alpha(self):
        image = Image.new("RGBA", (1200, 80), (255, 0, 0, 255))
        image.putpixel((1199, 0), (0, 0, 0, 0))
        path = self.root / "scene.png"
        image.save(path)
        data, valid, profile, metadata = read_full_image(path, bounds=[13.4, 52.5, 13.42, 52.501])
        self.assertEqual(data.size, (1200, 80))
        self.assertFalse(valid[0, -1])
        np.testing.assert_allclose(profile["transform"] * (1200, 80), [13.42, 52.5])

    def test_automatic_translation_alignment_and_failed_alignment(self):
        rng = np.random.default_rng(42)
        pixels = rng.integers(0, 255, (400, 500, 3), dtype="uint8")
        moved = cv2.warpAffine(pixels, np.float32([[1, 0, 12], [0, 1, 7]]), (500, 400))
        valid = np.ones((400, 500), bool)
        aligned, overlap, details = registration(Image.fromarray(pixels), Image.fromarray(moved), valid, valid)
        self.assertGreater(overlap.mean(), .9)
        self.assertLess(details["registration_error_px"], 2)
        np.testing.assert_allclose(np.array(details["transform_before_to_after"])[:2, 2], [12, 7], atol=2)
        with self.assertRaisesRegex(ValueError, "landmarks"):
            registration(Image.new("RGB", (100, 100)), Image.new("RGB", (100, 100), "white"),
                         np.ones((100, 100), bool), np.ones((100, 100), bool))

    def test_manual_registration(self):
        image = Image.new("RGB", (200, 200))
        valid = np.ones((200, 200), bool)
        pairs = [[[x,y],[x,y]] for x,y in ((.1,.1),(.9,.1),(.9,.9),(.1,.9))]
        _, overlap, details = registration(image, image, valid, valid, "manual", pairs)
        self.assertTrue(overlap.all())
        self.assertEqual(details["method"], "Manual landmarks")

    def test_png_comparison_without_georeferencing_detects_addition(self):
        before = np.zeros((180, 220, 3), "uint8")
        after = before.copy(); after[50:90, 60:110, 0] = 255
        first, second = self.root / "before.png", self.root / "after.png"
        Image.fromarray(before).save(first); Image.fromarray(after).save(second)
        def masks(image, valid, prompt, threshold):
            found = (np.array(image)[:, :, 0] > 100) & valid
            return found[None], np.array([.9])
        with patch("src.inference.change_detection.predict_masks", side_effect=masks), \
             patch("src.inference.change_detection.new_run", return_value=("test", self.root)):
            result = compare_images(first, second, alignment="same_grid")
        self.assertEqual(result["metadata"]["added_pixels"], 2000)
        self.assertEqual(result["metadata"]["removed_pixels"], 0)
        self.assertIsNone(result["map_bounds"])
        self.assertEqual(result["geojson"]["features"], [])
        self.assertEqual(result["image_features"]["coordinate_space"], "image_pixels")

    def test_api_settings_ollama_and_background_map_job(self):
        from fastapi.testclient import TestClient
        from src.interface.app import app
        client = TestClient(app)
        self.assertEqual(client.get("/settings").status_code, 200)
        self.assertEqual(client.put("/api/settings", json={"sam3": {"threshold": .7}}).status_code, 200)
        self.assertEqual(client.get("/api/settings").json()["sam3"]["threshold"], .7)
        with patch("src.language.ollama.requests.get", side_effect=__import__("requests").ConnectionError):
            self.assertFalse(client.post("/api/ollama/models", json={}).json()["connected"])
        result = client.post("/api/search", json={"query": "detect buildings and trees here"}).json()
        self.assertEqual(result["kind"], "segment")
        self.assertEqual(result["plan"]["prompt"], "building, tree")
        with patch("src.inference.tiled.segment_view", return_value={"run_id": "fake", "geojson": {"features": []}}):
            response = client.post("/api/segment-view", json={"bounds": [13.41,52.52,13.412,52.521]})
            self.assertEqual(response.status_code, 200)
            identifier = response.json()["job_id"]
            for _ in range(40):
                result = client.get(f"/api/jobs/{identifier}").json()
                if result["status"] in {"completed", "failed"}:
                    break
                time.sleep(.05)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["result"]["run_id"], "fake")


if __name__ == "__main__":
    unittest.main()
