# Working with the map application

See the [map workspace guide](map_workspace.md) for drawing a selection, imagery
coverage, model Settings and ordinary-image comparison.

From the project directory:

```powershell
conda run --no-capture-output -n sat_clas python -m src.cli serve
```

Open **http://127.0.0.1:8765**. The installed equivalent is `sat-clas serve`
after `conda activate sat_clas`. API documentation is at `/docs`. The server
binds to your local machine. Use Ctrl+C in its terminal to stop it.
For the background instance started during setup, use
`powershell -NoProfile -File scripts/stop_map.ps1`.

## Inputs and outputs

| Input | How to use it | Result |
| --- | --- | --- |
| Address / place | `1600 Amphitheatre Parkway, Mountain View, California` | Select a geocoder candidate, then retrieve its OSM geometry. An address node can also highlight containing building polygons. |
| Natural-language category query | `Where are the schools in this area?` | Mapped school records within the current viewport. |
| Named area | `hospitals in Berlin` | Choose the intended Berlin candidate, zoom to a neighborhood and click **Search this map area**. |
| Coordinates | `52.52,13.405` | Marker; input order is latitude, longitude. GeoJSON uses longitude, latitude. |
| Geohash | `geohash: u33dc1` | The geohash cell displayed as a polygon. |
| Viewport / bounding box | Pan and zoom, choose a category | Buildings, hospitals, schools, parks, roads, water, trees, shops, restaurants, hiking routes or common map elements. |
| Another mapped category | Enter `amenity=pharmacy`, `tourism=museum` or another key=value | Available records matching that tag. |
| Drawn map box + visual prompt | Explore; `building`, `tree`, `water` etc. | Fetch supported aerial imagery, process the complete selection, label map footprints. |
| GeoTIFF + visual prompt | Advanced → Segment an uploaded image | SAM scores, WGS84 polygons, preview and optional map evidence. |
| PNG/JPEG + visual prompt | Advanced → Segment an uploaded image | Image-space masks and labels. Optional full-image WGS84 bounds locate a north-up orthorectified image. |
| GeoJSON | Advanced → Import GeoJSON | Inspect supplied geometries and attributes on the map. |
| Before/after PNG, JPEG or GeoTIFFs | Compare | Preview registration; compare masks on valid overlap; export added/removed candidate regions. |
| Labeled RGB tiles | Training CSV manifest | Train/evaluate ResNet-18, save a checkpoint and classify new tiles. |

Global means queries can target locations worldwide. It does not mean the
application downloaded the planet or that every object has a record. Interactive
map-data requests are limited to 25 km²; zoom further in for dense areas. Map
selections and uploaded images use overlapping windows covering their full extent,
subject to pixel and window-count limits. The CLI/notebook retains the original
explicit crop option for larger rasters.

The model uses existing local SAM 3 weights. Changing its visual prompt requires
no training. Hospital/school identity is primarily a map-data query: appearances
alone do not reliably distinguish a hospital, office, school or residential building.

## Names, addresses and geographic evidence

Photon supplies address/place candidates. Overpass supplies OSM geometries and
tags, including names, addresses, category tags and source identifiers. Every map
record retains its source URL and fetch time. Click map features to inspect their
properties; named records have map labels. Downloads are GeoJSON.

When segmenting imagery, the application retrieves map features in the image
window once and attaches spatial candidates to detections. Polygon overlap is
measured in a local equal-area projection; contained points remain separate
tenant/entrance/place evidence. One dominant footprint can provide a **candidate**
name/address. These fields never become verified identity just because they overlap.
Absent names/addresses remain null. Several businesses can occupy one building;
a hospital or school may cover a campus with several buildings.

Geohashes are search/index cells, not unique building IDs. OSM type+ID is the
source identifier. An ambiguous address such as `5 Starmansrteen` needs a better
spelling, city/country, viewport context or a selected search candidate. The
application cannot certify an exact postal address missing from its source data.

Public Photon and Overpass services are used for small interactive requests;
responses are cached in `data/interim/map_cache` for 24 hours, with serialized
requests and one retry for transient server failures. No tile scraping or bulk
reverse-geocoding is performed. Map tiles load in the browser with OSM attribution.
For sustained use, configure your own/contracted endpoints in Settings.
Images and model inference stay local; search strings and requested geographic
areas are sent to the configured map providers. Browser basemap tiles use the
visible map area. Built-in aerial sources cover Berlin and the contiguous US;
custom WMS supports other regions. Historical comparisons use images you supply.

## Language processing

Common requests use a deterministic parser. SAM 3 already contains the text
encoder used to match visual concepts. No additional encoder or NER model is
required for the implemented workflows.

For broader phrasing, choose an installed model in Settings and use **Test selected
model**. The adapter validates a structured geographic plan; names and addresses
come from provider records. No model is downloaded automatically. Language and
SAM inference share a scheduler and release model memory when changing workloads.
See the workspace guide for the available operations and GPU behaviour.

Separate NER becomes useful only if you need a dedicated, measured extractor for
large volumes of address/place text. A general LLM is useful for composing complex
queries, not as the authoritative geographic database.

## Train a classifier with your data

Create reviewed RGB PNG/JPEG tiles and a CSV manifest. Paths are relative to the
CSV unless absolute. Labels must describe what you intend the classifier to learn.
Do not automatically treat unreviewed SAM predictions or OSM overlaps as ground truth.

```csv
path,label,split,group
tiles/region_a_roof_01.png,building,train,region_a
tiles/region_a_forest_01.png,forest,train,region_a
tiles/region_b_roof_01.png,building,val,region_b
tiles/region_b_forest_01.png,forest,val,region_b
tiles/region_c_roof_01.png,building,test,region_c
tiles/region_c_forest_01.png,forest,test,region_c
```

This is a format illustration, not an adequate dataset. Hold out geographic areas,
acquisition scenes and preferably dates. Neighboring/overlapping tiles must not
cross splits. The loader checks group separation and identical-file leakage;
you still need to define spatially separated groups correctly. All classes must
exist in the training split. RGB band order must be consistent across sources.

```powershell
sat-clas train --manifest D:/data/my_dataset/manifest.csv --output outputs/training/my_first_model --epochs 20 --batch-size 16
sat-clas classify D:/data/my_dataset/new_tile.png --checkpoint outputs/training/my_first_model/best.pt
```

Both commands also work as `python -m src.cli ...`. In the browser, enter the
manifest path in Training, choose epochs/batch size, and start the job. Progress
is polled, the job can be cancelled, and completed checkpoints appear in the
classifier selector. Choose an empty output directory for each CLI run.

Outputs: `best.pt`, `history.json`, `training_config.json`, `resolved_manifest.csv`
and `evaluation.json`. The best epoch is selected by validation loss; evaluation
includes class precision/recall/F1 and confusion matrices, plus a test split when
provided. Scores on new tiles are not calibrated probabilities. The baseline
starts with random weights; representative data and training are required before
it is useful. Training has been smoke-tested with synthetic tiles only, so no
real-world classifier accuracy is claimed.

This trainer trains **ResNet-18 image classification**, not the SAM 3 mask decoder.
Actual SAM fine-tuning needs reviewed instance masks/polygons, concept labels,
dataset conversion, geographic splits and a separate training configuration.
Meta's native training instructions use its own dependencies and Python 3.12+
workflow. Do not alter the working Python 3.11 inference environment blindly.
The next SAM-specific step is to select the annotated dataset, reproduce a small
native training run in a separate environment, and measure mask IoU/AP against
the current zero-shot baseline before scaling it.

## Package and Python API

```powershell
python -m pip install --no-deps -e .
python -m pip wheel --no-deps . --wheel-dir dist
sat-clas --help
```

The wheel bundles Python modules, the browser interface and a default config.
Install CUDA PyTorch first on another machine, then install the wheel and point
`SAT_CLAS_CONFIG` to that machine's YAML. Set `SAT_CLAS_HOME` to a writable project
directory for data/results when using the wheel. Model weights and data are external.
The existing `src` imports are retained so the notebook continues to work.

```python
from src.geo.providers import search_address, query_features
from src.inference.segmentation import segment_image

candidates = search_address("1600 Amphitheatre Parkway, Mountain View, California")
schools = query_features([-122.27, 37.87, -122.26, 37.88], category="schools")
result = segment_image("my_rgb_scene.tif", prompt="building", enrich=True)
```

Inference outputs live under `outputs/predictions/map_app/<run_id>/`: GeoJSON,
metadata, image/overlay previews and instance arrays for single-image segmentation.
The two-date workflow reports candidate mask changes, not verified construction
or demolition; registration, resolution, shadows and seasons affect results.

## What remains beyond this prototype

Turn-by-turn hiking route calculation is not implemented; existing mapped hiking
routes can be displayed. Production global coverage needs scalable map providers,
imagery licensing/acquisition, regional caching, full-raster tiling and deduplication,
better identity reconciliation, and held-out evaluation. Historical map-data
queries are separate from comparing supplied historical imagery. Text-only queries
do not automatically fetch satellite scenes or execute multi-stage GIS analyses.

## Verification and sources

```powershell
python -m unittest discover -s tests -v
python scripts/verify_application.py
```

The second command runs real GPU smoke checks and a few live provider queries;
it requires the cached Berkeley demo and network availability. Its synthetic
training scores are infrastructure checks, not model-quality results.

- [Photon API and public service usage](https://github.com/komoot/photon)
- [Photon address and location-bias parameters](https://github.com/komoot/photon/blob/master/docs/api-v1.md)
- [Overpass API](https://wiki.openstreetmap.org/wiki/Overpass_API)
- [OSM tile usage policy](https://operations.osmfoundation.org/policies/tiles/)
- [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs)
- [Transformers SAM 3](https://huggingface.co/docs/transformers/model_doc/sam3)
- [Meta SAM 3 training](https://github.com/facebookresearch/sam3/blob/main/README_TRAIN.md)
