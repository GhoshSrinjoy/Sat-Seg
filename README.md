# sat_clas

Local satellite classification and SAM 3 segmentation starter for Windows,
Python 3.11 and an NVIDIA RTX 3090. Processing uses PyTorch; SAM 3 weights live
outside this repository in `D:/data/models`. Ollama remains a separate service.

## Open the map interface

```powershell
conda run --no-capture-output -n sat_clas python -m src.cli serve
```

Open **http://127.0.0.1:8765**. Search a location, click **Draw area**, and drag a
box to find buildings, trees, water or other visible concepts with SAM. The map
switches to aerial imagery automatically and displays labelled footprints. Edit
or reject detections, match names/addresses from map records, and export labels.

**Settings** at **http://127.0.0.1:8765/settings** contains the Ollama model picker,
connection and query tests, local SAM checkpoint/device, thresholds, imagery
providers, cache limits and comparison preferences. Settings persist separately in
`configs/local.yaml` (ignored by version control). Models are never downloaded by
the app. Start the separately installed Ollama service to discover its local models.

**Compare** accepts before/after **PNG or JPEG images**, with an alignment preview,
manual landmark matching and a swipe viewer. Map coordinates are optional; TIFFs
remain supported. **Advanced** contains image uploads, GeoJSON import and classifier
training. The classifier trainer reloads saved checkpoints.

Built-in aerial imagery covers **Berlin** (open 2025 orthophotos) and the
**contiguous US** (public-domain NAIP). Other regions need a configured WMS imagery
source that permits analysis. Vector place/address searches remain worldwide.
The whole selected box is processed in overlapping windows, subject to area,
pixel and window-count limits. No street-map screenshots are used for SAM.

See [the map workspace guide](docs/map_workspace.md) for the UI, imagery coverage,
model settings, image alignment, API and verification commands.
The installed CLI is `sat-clas`; API documentation is at `/docs`.

See [training, input types, names/addresses and packaging](docs/training_and_map_workflows.md)
for runnable examples, dataset format and current limitations. No real labeled
training dataset has been supplied; the training pipeline has only been smoke-tested.

## Layout

```text
sat_clas/
  data/                 raw/, interim/, processed/
  notebooks/            01_environment_and_raster_exploration.ipynb
  src/
    data/               raster_loading.py, preprocessing.py
    models/             baseline_classifier.py, sam3_loader.py
    training/           classification_loops.py
                        train_classifier.py
    geo/                provider queries, geohashes, spatial evidence matching
    language/           common-query parser and optional Ollama adapter
    inference/          segmentation, projected exports, change candidates
    interface/          FastAPI server and browser map
    cli.py              installed sat-clas command
    utils/              configuration.py, verify_environment.py
  configs/              default.yaml
  outputs/              logs/, figures/, predictions/
  scripts/              setup, folder creation and environment export
  docs/                 installation and model inspection evidence
  environment.yml       unpinned environment specification
  requirements.txt      unpinned supporting packages
  pyproject.toml         editable/wheel installation and CLI entry point
```

## Create the environment (run once)

Run these commands in PowerShell or an Anaconda Prompt, from this project:

```powershell
Set-Location 'D:\src\sat_sam_clasfictn\sat_clas'
powershell -NoProfile -File .\scripts\initialize_project.ps1
conda create --name sat_clas --override-channels --channel conda-forge python=3.11 pip --yes
conda run --no-capture-output -n sat_clas python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
conda run --no-capture-output -n sat_clas python -m pip install -r requirements.txt
conda run --no-capture-output -n sat_clas python -m pip install --no-deps -e .
conda run --no-capture-output -n sat_clas python -m ipykernel install --user --name sat_clas --display-name 'Python (sat_clas)'
conda run --no-capture-output -n sat_clas python -m src.utils.verify_environment
conda run --no-capture-output -n sat_clas python -m src.models.sam3_loader --inspect
conda run --no-capture-output -n sat_clas python scripts/export_environment.py
```

Alternatively, `powershell -NoProfile -File .\scripts\setup_environment.ps1`
executes that sequence, refusing to modify an existing `sat_clas` environment.
Choose one installation method. To create from YAML instead, replace the conda
create and the two dependency-install commands with:

```powershell
conda env create --file environment.yml
```

`cu130` selects the CUDA 13.0 binary channel, not a PyTorch version. The resolver
selects stable PyTorch/torchvision releases without exact version pins. The
installed NVIDIA driver supports this runtime. A separate CUDA toolkit install
is unnecessary for these binary wheels.

The YAML deliberately resolves fresh package versions. `scripts/export_environment.py`
also records the actual resolved environment for this Windows machine; see
`docs/environment.resolved.yml` for a repeatable snapshot and
`docs/installed_packages.json` for the installed version inventory. The editable
project itself is installed separately with `pip install --no-deps -e .`.
To restore the recorded versions on another compatible Windows machine, use
`conda env create --file docs/environment.resolved.yml` when `sat_clas` does not
already exist, then install this checkout with the editable-install command.

## Start working

```powershell
conda activate sat_clas
python -m src.utils.verify_environment
python -m src.models.sam3_loader --load
python -m jupyterlab notebooks
```

Select the **Python (sat_clas)** notebook kernel. If conda activation is not
initialized in this shell, prefix Python commands with
`conda run --no-capture-output -n sat_clas` as above; no shell profile changes
are needed. In the notebook, use **Restart Kernel -> Run All** for the complete
image-loading, SAM inference, visualization and export workflow. It downloads a
12.7 MB Berkeley example once into `data/raw/examples/` and reuses local SAM
weights. Set `IMAGE_PATH` in its Settings cell to use your own GeoTIFF or PNG/JPEG.
Results are saved in separate run directories under `outputs/predictions/notebook_demo/`.

## SAM 3 cache

The existing `facebook/sam3` cache was found under
`D:/data/models/models--facebook--sam3`. Its snapshot revision is recorded in
`configs/default.yaml`. Setup reuses existing weights without downloading a
duplicate. The original cache includes the video model and its image detector;
the loader selects the image detector for satellite still-image work.

```powershell
python -m src.models.sam3_loader --inspect
python -m src.models.sam3_loader --load
python -m src.models.sam3_loader --smoke-inference
```

Loading is local-only by default and checks for missing or mismatched image
detector weights. The smoke inference uses a synthetic image to verify execution;
it does not measure satellite accuracy. If restoring on a different machine,
inspect first, then explicitly request missing files:

```powershell
python -m src.models.sam3_loader --inspect
# Only after inspecting the result and obtaining access to facebook/sam3:
hf auth login
python -m src.models.sam3_loader --download-missing
```

Model access can require accepting Meta's conditions on Hugging Face. Never put
access tokens into this repository. The Transformers implementation is used to
support the requested Python 3.11 environment and the existing checkpoint format.
The native Meta SAM repository documents a Python 3.12+ installation workflow.

## Scope and next decisions

Map record queries work worldwide within small areas. Names and addresses come
from available map records; image matches retain candidate status. Unlocated images
can be analysed and compared in image coordinates. Placing an ordinary photograph
on a map still needs correct georeferencing. A drawn box supplies the extent only
for an already north-up, orthorectified image.

Map selections and full-image uploads use overlapping inference windows. The
legacy CLI/notebook window option remains available. Existing hiking routes are
supported; turn-by-turn route calculation is not implemented.

To train on real data, supply reviewed RGB tiles, classes and geographic splits.
The runnable classifier trainer is separate from SAM fine-tuning, which still needs
a labeled segmentation dataset and a native training environment. The optional
Ollama query adapter uses a local model selected and tested through Settings.

## References

- [PyTorch installation](https://pytorch.org/get-started/locally/)
- [Transformers SAM 3](https://huggingface.co/docs/transformers/model_doc/sam3)
- [SAM 3 model repository](https://huggingface.co/facebook/sam3)
- [Native Meta implementation](https://github.com/facebookresearch/sam3)
