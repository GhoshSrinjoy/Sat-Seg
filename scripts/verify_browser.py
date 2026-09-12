"""Exercise the running local UI using installed Edge; requires the dev extra."""

import json
from playwright.sync_api import sync_playwright, expect
from src.utils.configuration import PROJECT_ROOT


def main():
    errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto("http://127.0.0.1:8765", wait_until="networkidle")
        expect(page.locator("#gpu")).to_contain_text("3090")
        page.locator("#query").fill("geohash: u33dc1")
        page.locator('#search-form button[type="submit"]').click()
        expect(page.locator("#status")).to_contain_text("Geohash search cell", timeout=15000)
        page.locator("#query").fill("1600 Amphitheatre Parkway, Mountain View, California")
        page.locator('#search-form button[type="submit"]').click()
        expect(page.locator("#candidates button").first).to_be_visible(timeout=30000)
        page.locator("#candidates button").first.click()
        expect(page.locator("#status")).to_contain_text("Selected source feature", timeout=110000)
        page.screenshot(path=str(PROJECT_ROOT / "outputs/figures/address_map_verified.png"), full_page=True)
        page.locator('[data-panel="imagery-panel"]').click()
        page.locator("#demo").click()
        expect(page.locator("#image-results")).to_be_visible(timeout=150000)
        expect(page.locator("#status")).not_to_have_class("error")
        page.wait_for_function("Array.from(document.querySelectorAll('#previews img')).every(image => image.complete && image.naturalWidth > 0)")
        page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")
        assert page.evaluate("map.getSize().y === document.getElementById('map').clientHeight")
        page.screenshot(path=str(PROJECT_ROOT / "outputs/figures/imagery_map_verified.png"), full_page=True)
        page.locator('[data-panel="training-panel"]').click()
        expect(page.locator("#train-form")).to_be_visible()
        assert not errors, errors
        report = {"browser": "Microsoft Edge headless", "address_search": "PASS", "selected_footprint": "PASS",
                  "geohash_search": "PASS", "imagery_demo": "PASS", "training_form": "PASS",
                  "image_status": page.locator("#status").inner_text(), "javascript_errors": errors}
        (PROJECT_ROOT / "outputs/logs/browser_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        browser.close()


if __name__ == "__main__":
    main()
