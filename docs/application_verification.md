# Local application verification — 2026-09-06

Verified with the existing Python 3.11 `sat_clas` environment and RTX 3090.
No SAM or Ollama model weights were downloaded for this application work.

- Seven offline regression tests pass: query/address parsing, geohash/bounds,
  spatial candidate matching and multiple tenants, window transforms, geographic
  training leakage, HTTP inputs, and before/after alpha/nodata handling.
- The exploration notebook runs all 22 cells in a fresh `sat_clas` kernel.
  Real-data training and extra map lookups remain disabled until configured.
- A real Berkeley imagery window produces 15 SAM detections. Live OSM lookup
  supplies spatial candidates for all 15 and tentative names for seven.
  These are execution/matching observations, not independently verified identities.
- Comparing the Berkeley image with itself produces zero added/removed pixels.
  This is a consistency check, not an accuracy assessment on historical changes.
- Two epochs of synthetic RGB classification run on CUDA. The best checkpoint
  reloads and classifies a tile. A separate test exercises the HTTP training job,
  subprocess completion, checkpoint listing and classification endpoint.
- Live address search returns Google campus candidates for the Mountain View
  example; selecting OSM way 23733659 retrieves Google Building 41's polygon,
  name and address. One public Overpass 504 occurred during testing; a later
  request succeeded. The client now makes one bounded transient-error retry.
- Microsoft Edge browser checks pass for geohash search, address candidates,
  footprint selection, the live imagery demo and the training form, with no
  JavaScript errors. Screenshots are in `outputs/figures/`.
- A distributable wheel contains the Python modules, static web interface,
  default configuration and `sat-clas` entry point.

Evidence: `outputs/logs/browser_verification.json`,
`outputs/logs/notebook_with_package_verified.ipynb`,
`outputs/logs/app_verification_20260906T175047Z/classifier/`, and dated
`outputs/predictions/map_app/` runs. The first combined smoke script stopped on
the Overpass timeout after its GPU checks; the failed provider request and
map enrichment subsequently passed in separate live/browser checks.

No real labeled dataset, SAM fine-tuning run, multilingual LLM benchmark,
turn-by-turn routing or production-scale global coverage has been validated.
