"""Browser regression for map selection, settings, image uploads and alignment."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright, expect

from src.utils.configuration import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8766")
    args = parser.parse_args()
    output = PROJECT_ROOT / "outputs/figures/workspace"
    output.mkdir(parents=True, exist_ok=True)
    fixture = output / "alignment_fixture.png"
    Image.fromarray(np.random.default_rng(7).integers(0, 255, (300, 400, 3), dtype="uint8")).save(fixture)
    errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.url, wait_until="domcontentloaded")
        page.wait_for_function("settings !== null")
        expect(page.locator("#concept")).to_have_value("building")
        page.locator("#auto-analyse").uncheck()
        page.locator("#draw-area").click()
        rect = page.locator("#map").bounding_box()
        x, y = rect["x"]+rect["width"]*.45, rect["y"]+rect["height"]*.35
        page.mouse.move(x,y); page.mouse.down(); page.mouse.move(x+120,y+100,steps=12); page.mouse.up()
        page.wait_for_function("selectedBounds !== null")
        expect(page.locator("#satellite-map")).to_have_attribute("aria-pressed", "true")
        expect(page.locator(".selection-handle")).to_have_count(4)
        page.wait_for_function("document.getElementById('selection-info').textContent.includes('pixels')")
        prior = page.evaluate("arrayBounds(selectedBounds)")
        handle = page.locator(".selection-handle").first.bounding_box()
        page.mouse.move(handle["x"]+6,handle["y"]+6); page.mouse.down(); page.mouse.move(handle["x"]-20,handle["y"]-15,steps=8); page.mouse.up()
        assert page.evaluate("arrayBounds(selectedBounds)") != prior
        page.screenshot(path=str(output / "map_selection.png"), full_page=True)
        page.locator('[data-panel="settings-panel"]').click()
        expect(page).to_have_url(args.url+"/settings")
        expect(page.locator("#settings-form")).to_be_visible()
        expect(page.locator('[data-setting="sam3.mask_threshold"]')).to_have_value("0.5")
        page.locator("#refresh-models").click()
        expect(page.locator("#ollama-connection")).not_to_have_text("")
        page.screenshot(path=str(output / "settings.png"), full_page=True)
        page.locator('[data-panel="compare-panel"]').click()
        page.locator("#before-file").set_input_files(str(fixture))
        page.locator("#after-file").set_input_files(str(fixture))
        page.wait_for_function("beforeImage !== null && afterImage !== null")
        page.locator("#preview-alignment").click()
        expect(page.locator("#compare-submit")).to_be_enabled(timeout=45000)
        expect(page.locator("#alignment-result")).to_contain_text("100.0% overlap")
        page.locator("#swipe-slider").fill("75")
        expect(page.locator("#swipe-before")).to_have_css("clip-path", "inset(0px 25% 0px 0px)")
        page.screenshot(path=str(output / "image_comparison.png"), full_page=True)
        page.locator('[data-panel="search-panel"]').click()
        page.locator("#query").fill("52.52,13.405")
        page.locator('#search-form button[type="submit"]').click()
        expect(page.locator("#status")).to_contain_text("Coordinate location", timeout=15000)
        page.locator("#clear-selection").click()
        expect(page.locator(".selection-handle")).to_have_count(0)
        page.set_viewport_size({"width": 390,"height":844})
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.screenshot(path=str(output / "mobile.png"), full_page=True)
        assert not errors, errors
        report = {"map_draw_resize": "PASS", "satellite_switch": "PASS", "settings_page": "PASS",
                  "ollama_discovery_ui": "PASS", "png_alignment_preview": "PASS", "comparison_swipe": "PASS",
                  "coordinate_search": "PASS", "mobile_overflow": "PASS", "javascript_errors": errors}
        (output / "browser_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        browser.close()


if __name__ == "__main__":
    main()
