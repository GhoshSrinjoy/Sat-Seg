# Completed setup

Project: `D:/src/sat_sam_clasfictn/sat_clas`

Environment: `C:/Users/Srinjoy Ghosh/.conda/envs/sat_clas`

- Python resolved to **3.11.16**.
- PyTorch resolved to **2.14.0+cu130** and torchvision to **0.29.0+cu130**.
- `torch.cuda.is_available()` returned **True**.
- GPU: **NVIDIA GeForce RTX 3090**, 24 GB VRAM.
- CUDA matrix multiplication and torchvision CUDA NMS passed.
- All requested supporting packages imported successfully.
- `pip check` reported **No broken requirements found**.
- The project is installed in editable mode and the **Python (sat_clas)**
  Jupyter kernel is registered.

## Existing models

The existing Hugging Face SAM 3 cache was reused. No model weights were
downloaded, and the Ollama installation/model inventory was not modified.
The 3,439,938,512-byte weights file passed its SHA-256 check against the cache
blob identifier; details are in `model_inventory.json`.

The image detector loaded from the composite video checkpoint with no missing
or mismatched image weights. With Hugging Face networking disabled, a synthetic
image inference completed on CUDA with finite output masks. This verifies
execution only; no satellite classification accuracy has been measured.

## Starter verification

- GeoTIFF read/write preserved CRS, pixel values, nodata and window transforms,
  including a requested window crossing the image edge.
- RGB preview conversion respected masked pixels.
- The untrained classification baseline completed a GPU forward/backward pass,
  changed its parameters during training, and completed evaluation.
- The exploration notebook executed successfully with the `sat_clas` kernel.
- Python syntax and dependency-manifest consistency checks passed.

The optional executed notebook and SAM diagnostic log are under `outputs/logs/`.
The Jupyter run emitted Windows event-loop/transport notices and still completed.

## Reproduction

`environment.yml` and the README installation commands leave package versions
unpinned (apart from the requested Python 3.11 series). The separate
`environment.resolved.yml` records the versions selected during this installation
for restoration on a compatible Windows/NVIDIA system. No other conda environment
was modified.

No additional user decision was needed for this setup. Dataset selection,
classification labels and an Ollama model remain future application decisions.
