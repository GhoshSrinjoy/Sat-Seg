# Geo Explorer workspace

Run `sat-clas serve` (or `python -m src.cli serve`) and open
http://127.0.0.1:8765. The map, selected area and result set are shared by the
Explore, Compare and Settings views. Advanced retains the existing training tools.

## Search and label

1. Search an address or location. Select a candidate when the name is ambiguous.
2. Click **Draw area** and drag a rectangle. Aerial imagery switches on. Corner
   handles resize the rectangle; the centre handle moves it. **Use visible area**
   selects the current viewport. Escape cancels drawing.
3. SAM runs after drawing by default. Change this with the checkbox or Settings.
   Resizing a selection requires clicking **Analyse selected area** again.
4. Enter a visible concept, or up to four comma-separated concepts such as
   `building, tree`. Inspect labelled footprints and the result list.
5. Select a detection to edit its reviewed label or reject it. Changes persist in
   that run's GeoJSON, pixel-label JSON and annotated image. Export the filtered
   GeoJSON/CSV or download the annotated image and image-coordinate labels.

Progress and cancellation appear under the map. The map remains interactive.
Cancellation takes effect between imagery requests/inference windows. A maximum
of three analysis jobs can be queued. Reloading the browser reconnects to its
active job; server restarts end in-memory jobs. Finished results remain on disk.

The analysis uses every pixel in the selection, rather than a centre crop.
Overlapping windows are merged by same-concept spatial overlap. This reduces
duplicate detections at tile edges but is not an accuracy guarantee. Default limits
are 25 km², 4 million pixels and 25 inference windows. Smaller neighbourhood boxes
are appropriate for high-resolution building detection.

## Imagery sources

| Source | Coverage | Acquisition | Use |
| --- | --- | --- | --- |
| USGS NAIP | Contiguous United States | Mosaic; dates vary | Public-domain imagery from USGS/USDA |
| Geoportal Berlin DOP 2025 | Berlin | Spring 2025 | DL-DE-Zero 2.0 |
| Custom WMS | Supplied by the configured service | Provider-dependent | Configure an imagery layer permitting your analysis |

Auto selects a built-in provider when the complete box is within its coverage.
Areas outside coverage return an actionable message. Displayed imagery and
inference pixels use the same provider. Both browser tiles and mosaics are fetched
through the local server, without browser canvas capture. Mosaics retain their
EPSG:3857 affine transform; only output vectors are converted to WGS84.

WMS settings require a service URL, layer name, attribution and optionally a token.
The service must support WMS 1.1.1 GetMap, EPSG:3857 and PNG. A token is sent as the
`token` query parameter, stored locally, and omitted from the settings response.
Blank token fields preserve the saved value; the clear checkbox removes it.
Use an imagery layer with analysis rights, rather than a rendered street map.

Imagery cache: `data/interim/map_cache/imagery`, keyed by service and request grid,
with configurable age and size limits. Missing/transparent pixels are excluded
from inference. Exports retain provider attribution and acquisition information
when available; a basemap mosaic is not a date-specific historical image.

Provider references:

- [USGS NAIP service](https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer)
- [Berlin orthophotos and licence](https://daten.berlin.de/datensaetze/digitale-farbige-orthophotos-2025-dop20rgbi-wms-6529de5a)
- [Esri World Imagery extraction conditions](https://www.esri.com/arcgis-blog/products/arcgis-living-atlas/imagery/learn-to-use-ai-to-extract-information-from-world-imagery): its public basemap is not a built-in source for standalone SAM extraction.

## Ollama and map evidence

Settings discovers models from the configured Ollama server, shows connection
status, and tests the selected model with a schema-validated geographic request.
Local text-generation models are used; embedding and cloud-only entries cannot
be selected from the picker. OCR/specialized models may not pass a planning test.
No weights are downloaded automatically.

The language model chooses supported operations: address/place search, map
features, image segmentation, map-evidence enrichment, filtering and comparison.
The request includes selected bounds, the current place, the previous query,
result count and visual concept. Coordinates and explicit geohashes bypass the
language model. Explicit Address/place mode also avoids a model round trip.

Examples:

- `detect buildings and trees here` — segment imagery.
- `find hospitals in this area` — retrieve mapped hospital records.
- `find hospitals using satellite imagery` — use mapped sites to select building
  detections; a building must intersect a returned site/point.
- `label these buildings with addresses` — attach map evidence to current results.
- `only show buildings without a mapped address` — filter existing results.

SAM's visible class and score do not establish a building's identity or use.
Source names/addresses and tentative spatial matches remain separate. Missing
records stay unknown. Reviewed labels do not overwrite the original evidence.

One GPU owner is allowed at a time in the server. SAM is unloaded before an Ollama
query; the selected Ollama model is released before loading SAM. Training and
classification also release these models before using the GPU. External programs
and other server processes are outside this scheduler. Changes to checkpoint,
device or precision invalidate the cached SAM model. Settings cannot be applied
while background analysis or training is active.

Settings precedence: built-in defaults → `configs/default.yaml` (or
`SAT_CLAS_CONFIG`) → `configs/local.yaml` (or `SAT_CLAS_SETTINGS`). Local preferences
are written atomically. `SAT_CLAS_HOME` sets the project data/output root.

## Compare PNG/JPEG images

Add Before and After images of the same scene. Dates are optional labels.

- **Automatic** matches landmarks using ORB and a robust projective transform.
  Two georeferenced rasters are reprojected to the After grid instead.
- **Already aligned** requires equal dimensions and your declaration that the
  pixels correspond. Equal dimensions alone do not prove alignment.
- **Manual** uses at least four matching landmark pairs, spread across both
  images. Click Before and then After for each pair.

Preview alignment first. Inspect the swipe and blended images, then run the
comparison. Failed/degenerate registration is rejected. Comparison is limited to
valid overlapping pixels. Minimum region size and edge tolerance are configurable.
Added/removed regions are change candidates, not verified construction/demolition.

PNG/JPEG comparison works without coordinates and exports image-space regions.
Map placement is optional and uses the After image's georeferencing or entered
north-up orthorectified extent. A map box does not rectify an arbitrary photo.
GeoTIFF input is still accepted; RGB bands can be specified in Advanced options.

## API and verification

New APIs include:

| Endpoint | Function |
| --- | --- |
| `GET/PUT /api/settings` | Read/update validated persistent settings |
| `POST /api/ollama/models`, `/api/ollama/test` | Model discovery and planning test |
| `GET /api/sam/status`, `POST /api/sam/reload` | Inspect/load local SAM |
| `POST /api/imagery/plan` | Check coverage, dimensions and limits |
| `POST /api/segment-view` | Queue full selected-area segmentation |
| `GET /api/jobs/{id}`, `POST /api/jobs/{id}/cancel` | Progress and cancellation |
| `POST /api/compare/align` | Image alignment preview |
| `POST /api/compare` | PNG/JPEG/GeoTIFF comparison |
| `POST /api/enrich` | Match map evidence to an existing result set |
| `PATCH /api/results/{run}/labels` | Save reviewed labels and rejection flags |

Upload APIs accept `background=true` to return a job ID. The original synchronous
API remains available. API schemas are exposed at `/docs`.

```powershell
conda run --no-capture-output -n sat_clas python -m unittest discover -s tests -v
node --check src/interface/static/app.js
conda run --no-capture-output -n sat_clas python -m scripts.verify_workspace_browser --url http://127.0.0.1:8765
```

Browser verification uses the installed Edge browser via Playwright. Reports and
screenshots go to `outputs/figures/workspace/`. Tests cover settings persistence,
provider/size guards, pixel transforms, tile seams, PNG alpha, automatic/manual
registration, ordinary-image comparison and background HTTP jobs.
