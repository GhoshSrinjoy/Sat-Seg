"""FastAPI local map explorer. Run with sat-clas serve."""

import json
import subprocess
import sys
import uuid
import io
from typing import Literal
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import UnidentifiedImageError
from rasterio.errors import RasterioIOError
from requests import RequestException
from starlette.middleware.trustedhost import TrustedHostMiddleware

from src.geo import providers
from src.language.query_parser import parse_query
from src.utils.configuration import PROJECT_ROOT, load_config, save_settings
from src.utils.settings import AppSettings, OllamaSettings, public_settings
from src.interface import jobs
from src.inference.runtime import gpu_work, set_training, training_active

app = FastAPI(title="Geo Explorer", version="0.3.0")
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"])
STATIC = Path(__file__).parent / "static"
RESULTS = PROJECT_ROOT / "outputs/predictions/map_app"
RESULTS.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.mount("/results", StaticFiles(directory=RESULTS), name="results")
_training_jobs = {}


@app.exception_handler(ValueError)
def bad_input(request, error):
    return JSONResponse({"detail": str(error)}, status_code=400)


@app.exception_handler(FileNotFoundError)
def missing_input(request, error):
    return JSONResponse({"detail": str(error)}, status_code=400)


@app.exception_handler(RuntimeError)
def runtime_problem(request, error):
    return JSONResponse({"detail": str(error)}, status_code=503)


@app.exception_handler(RequestException)
def language_provider_problem(request, error):
    return JSONResponse({"detail": f"Language model service unavailable: {error}"}, status_code=503)


@app.exception_handler(UnidentifiedImageError)
@app.exception_handler(RasterioIOError)
def malformed_image(request, error):
    return JSONResponse({"detail": f"Cannot read this image: {error}"}, status_code=400)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/settings")
def settings_page():
    return FileResponse(STATIC / "index.html")


@app.get("/api/settings")
def get_settings():
    return public_settings(load_config())


@app.put("/api/settings")
def put_settings(values: dict):
    if jobs.active() or training_active():
        raise ValueError("Wait for analysis/training to finish or cancel it before applying settings.")
    if any(key not in AppSettings.model_fields for key in values):
        raise ValueError("Unknown settings group.")
    imagery = values.get("imagery", {})
    imagery.pop("has_custom_token", None)
    # Blank password field preserves an existing token; explicit clearing is separate.
    if imagery.get("custom_token") == "":
        imagery.pop("custom_token")
    if imagery.pop("clear_custom_token", False):
        imagery["custom_token"] = ""
    with gpu_work():
        old = load_config()["sam3"]
        result = save_settings(values)
        if any(old[k] != result["sam3"][k] for k in ("model_id", "revision", "cache_dir", "device", "dtype")):
            from src.inference.segmentation import unload_sam
            unload_sam()
    return public_settings(load_config())


@app.post("/api/ollama/models")
def ollama_models(settings: OllamaSettings):
    from src.language.ollama import list_models
    return list_models(settings.model_dump())


class ModelTest(BaseModel):
    settings: OllamaSettings
    query: str = Field("Find buildings in the selected box using satellite imagery", min_length=1, max_length=1000)


@app.post("/api/ollama/test")
def ollama_test(request: ModelTest):
    from src.language.ollama import query_plan
    if jobs.active():
        raise ValueError("Wait for the active analysis before testing another model.")
    plan = query_plan(request.query, request.settings.model_dump(), {"has_selection": True})
    return {"message": "Model returned a valid geographic plan.", "plan": plan.model_dump()}


@app.get("/api/sam/status")
def sam_status():
    from src.models.sam3_loader import find_snapshot, inspect_snapshot
    from src.inference.segmentation import model_status
    return {**model_status(), **inspect_snapshot(find_snapshot(load_config()["sam3"]))}


@app.post("/api/sam/reload")
def sam_reload():
    if jobs.active():
        raise ValueError("Wait for the active analysis before reloading SAM.")
    def load(progress):
        from src.inference.segmentation import unload_sam, predict_masks
        from PIL import Image
        import numpy as np
        with gpu_work():
            unload_sam()
            progress("Loading and checking SAM on the configured device", .3)
            predict_masks(Image.new("RGB", (64, 64)), np.ones((64, 64), bool), "building", 1)
        return {"kind": "model", "message": "SAM loaded and inference check passed."}
    return jobs.submit(load)


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    result = jobs.get_job(job_id)
    if result is None:
        raise HTTPException(404, "Unknown analysis job.")
    return result


@app.post("/api/jobs/{job_id}/cancel")
def cancel_analysis(job_id: str):
    return jobs.cancel_job(job_id)


@app.get("/api/imagery/tiles/{z}/{x}/{y}.png")
def imagery_tile(z: int, x: int, y: int):
    from src.geo.imagery import browser_tile
    buffer = io.BytesIO()
    browser_tile(z, x, y).save(buffer, format="PNG")
    return Response(buffer.getvalue(), media_type="image/png", headers={"Cache-Control": "no-cache"})


class ViewRequest(BaseModel):
    bounds: list[float] = Field(min_length=4, max_length=4)
    prompt: str | None = Field(None, min_length=1, max_length=200)
    enrich: bool | None = None
    site_category: str | None = Field(None, max_length=80)
    site_tag: str | None = Field(None, max_length=100)


@app.post("/api/imagery/plan")
def imagery_plan(request: ViewRequest):
    from src.geo.imagery import plan_view
    return plan_view(request.bounds)


@app.post("/api/segment-view")
def segment_view(request: ViewRequest):
    from src.geo.imagery import plan_view
    from src.inference.tiled import segment_view as run
    config = load_config()
    plan_view(request.bounds, config)
    if training_active():
        raise ValueError("Wait for training to finish or cancel it before analysing imagery.")
    if request.site_category:
        from src.inference.tiled import segment_sites
        if request.site_category not in providers.CATEGORIES:
            raise ValueError("Unknown site category.")
        return jobs.submit(lambda progress: segment_sites(request.bounds, request.site_category, request.site_tag, config, progress))
    return jobs.submit(lambda progress: run(request.bounds, request.prompt, request.enrich, config, progress))


@app.get("/api/status")
def status():
    import torch
    config = load_config()
    return {"cuda": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "categories": list(providers.CATEGORIES), "ollama_enabled": bool(config["ollama"]["model"]),
            "ollama_model": config["ollama"]["model"], "training": training_active(),
            "area_limit_km2": config["map"]["max_area_km2"]}


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    bounds: list[float] | None = None
    use_ollama: bool = False
    mode: str = "auto"
    source: Literal["auto", "maps", "imagery", "combined"] = "auto"
    context: dict = Field(default_factory=dict)


@app.post("/api/search")
def search(request: SearchRequest):
    plan = parse_query(request.query, request.use_ollama and request.mode != "address", request.context)
    if request.mode == "address":
        plan.intent, plan.query = "address", request.query
    elif request.mode != "auto":
        raise ValueError("Search mode must be auto or address.")
    if request.source != "auto":
        plan.source = request.source
    if request.source == "maps" and plan.intent == "segment":
        plan.intent = "features"
        plan.category = plan.category or {"building": "buildings", "tree": "trees", "road": "roads", "water": "water"}.get(plan.prompt, "buildings")
    if plan.source in {"imagery", "combined"} and plan.intent == "features" and plan.category in {"buildings", "trees", "water", "roads"}:
        plan.intent = "segment"
        plan.prompt = plan.prompt or {"buildings": "building", "trees": "tree", "water": "water", "roads": "road"}[plan.category]
    result = {"plan": plan.model_dump(), "geojson": providers.feature_collection()}
    if plan.intent in {"enrich", "filter"}:
        return {**result, "kind": plan.intent, "message": "Updating existing results."}
    if plan.intent == "segment":
        if plan.area:
            return {**result, "kind": "segment_candidates", "geojson": providers.search_address(plan.area),
                    "message": "Select a place, then draw the area to analyse."}
        return {**result, "kind": "segment", "message": "Analysing imagery in the selected area."}
    if plan.intent == "address":
        result.update(kind="candidates", geojson=providers.search_address(plan.query, request.bounds),
                      message="Select the matching place to retrieve its mapped geometry. Names may have multiple matches.")
    elif plan.intent == "features":
        if plan.area:
            result.update(kind="area_candidates", geojson=providers.search_address(plan.area),
                          message="Choose the intended area, zoom to a neighborhood, then use Search this map area.")
        else:
            result.update(kind="features", geojson=providers.query_features(request.bounds, plan.category, plan.tag),
                          message="Available mapped records in this area. Coverage depends on OpenStreetMap.")
    elif plan.intent == "geohash":
        result.update(kind="features", geojson=providers.geohash_area(plan.query), message="Geohash search cell.")
    elif plan.intent == "coordinates":
        lat, lon = map(float, plan.query.split(","))
        providers.bounds_around(lon, lat)
        geometry = {"type": "Point", "coordinates": [lon, lat]}
        result.update(kind="features", message="Coordinate location; latitude, longitude input.",
                      geojson=providers.feature_collection([{"type": "Feature", "geometry": geometry,
                      "properties": {"name": plan.query, "source": "user coordinates", **providers.geohash_properties(geometry)}}]))
    elif plan.intent == "compare":
        result.update(kind="compare", message="Open Compare and add before/after PNG or JPEG images.")
    else:
        result.update(kind="route", message="Use the Hiking routes category to display mapped routes. Turn-by-turn route calculation is not implemented.")
    return result


class FeatureRequest(BaseModel):
    bounds: list[float]
    category: str = "buildings"
    tag: str | None = None


@app.post("/api/features")
def features(request: FeatureRequest):
    return providers.query_features(request.bounds, request.category, request.tag)


@app.get("/api/resolve/{osm_type}/{osm_id}")
def resolve(osm_type: str, osm_id: int):
    return providers.resolve_feature(osm_type, osm_id)


def save_upload(upload):
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        raise ValueError("Upload a PNG, JPEG or TIFF image.")
    directory = PROJECT_ROOT / "data/interim/uploads"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (uuid.uuid4().hex + suffix)
    count = 0
    try:
        with path.open("wb") as handle:
            while chunk := upload.file.read(1024*1024):
                count += len(chunk)
                if count > 128*1024*1024:
                    raise ValueError("Interactive uploads are limited to 128 MB. Use the CLI for larger local rasters.")
                handle.write(chunk)
        if not count:
            raise ValueError("The uploaded file is empty.")
        return path
    except Exception:
        path.unlink(missing_ok=True)
        raise


def parse_numbers(value, count, cast):
    if not value.strip():
        return None
    values = [cast(part.strip()) for part in value.split(",")]
    if len(values) != count:
        raise ValueError(f"Expected {count} comma-separated numbers.")
    return values


@app.post("/api/segment")
def segment(image: UploadFile = File(...), prompt: str | None = Form(None), threshold: float | None = Form(None),
            rgb_bands: str = Form(""), bounds: str = Form(""), window_origin: str = Form(""), enrich: bool = Form(True),
            background: bool = Form(False)):
    from src.inference.segmentation import segment_image
    from src.inference.tiled import segment_full_image
    path = save_upload(image)
    def run(progress=None):
        bands = parse_numbers(rgb_bands, 3, int)
        extent = parse_numbers(bounds, 4, float)
        if window_origin:
            return segment_image(path, prompt, threshold, bands, extent, enrich, parse_numbers(window_origin, 2, int))
        return segment_full_image(path, prompt, threshold, bands, extent, enrich, progress)
    try:
        if background:
            return jobs.submit(run, lambda: path.unlink(missing_ok=True))
        return run()
    except Exception:
        path.unlink(missing_ok=True)
        raise
    finally:
        if not background:
            path.unlink(missing_ok=True)


@app.post("/api/demo")
def demo():
    from src.inference.segmentation import segment_image
    path = PROJECT_ROOT / "data/raw/examples/uc_berkeley.tif"
    if not path.is_file():
        raise ValueError("Run the exploration notebook once to cache the Berkeley example, or upload your own image.")
    return segment_image(path, enrich=True)


@app.post("/api/compare")
@app.post("/api/compare/align")
def compare(request: Request, before: UploadFile = File(...), after: UploadFile = File(...),
            prompt: str | None = Form(None), threshold: float | None = Form(None), rgb_bands: str = Form(""),
            alignment: str | None = Form(None), bounds: str = Form(""), points: str = Form(""),
            before_date: str = Form(""), after_date: str = Form(""), background: bool = Form(False)):
    from src.inference.change_detection import compare_images, alignment_preview
    first = save_upload(before)
    second = None
    try:
        second = save_upload(after)
        kwargs = {"bands": parse_numbers(rgb_bands, 3, int), "alignment": alignment,
                  "bounds": parse_numbers(bounds, 4, float), "points": json.loads(points) if points else None}
        preview = request.url.path.endswith("/align")
        def run(progress=None):
            if preview:
                return alignment_preview(first, second, **kwargs)
            return compare_images(first, second, prompt, threshold, before_date=before_date,
                                  after_date=after_date, progress=progress, **kwargs)
        def cleanup():
            first.unlink(missing_ok=True)
            second.unlink(missing_ok=True)
        if background:
            return jobs.submit(run, cleanup)
        return run()
    except Exception:
        first.unlink(missing_ok=True)
        if second:
            second.unlink(missing_ok=True)
        raise
    finally:
        if not background:
            first.unlink(missing_ok=True)
            if second:
                second.unlink(missing_ok=True)


class EvidenceRequest(BaseModel):
    collection: dict
    bounds: list[float] = Field(min_length=4, max_length=4)


@app.post("/api/enrich")
def enrich_results(request: EvidenceRequest):
    from src.geo.matching import attach_map_evidence
    from shapely.geometry import shape
    collection = request.collection
    if collection.get("type") != "FeatureCollection" or not isinstance(collection.get("features"), list):
        raise ValueError("Expected a geographic feature collection.")
    if len(collection["features"]) > 10000:
        raise ValueError("Too many features for interactive evidence matching.")
    for feature in collection["features"]:
        geometry = shape(feature["geometry"])
        if geometry.is_empty or not geometry.is_valid:
            raise ValueError("Evidence matching needs valid geographic geometries.")
    return attach_map_evidence(collection, providers.query_features(request.bounds, "all"))


class LabelEdit(BaseModel):
    object_id: int
    reviewed_label: str = Field(max_length=200)
    rejected: bool = False


@app.patch("/api/results/{run_id}/labels")
def save_labels(run_id: str, edits: list[LabelEdit]):
    directory = (RESULTS / run_id).resolve()
    if directory.parent != RESULTS.resolve() or not directory.is_dir():
        raise ValueError("Unknown result run.")
    if len(edits) > 10000:
        raise ValueError("Too many label edits.")
    for filename in ("features.geojson", "image_features.json"):
        path = directory / filename
        if not path.is_file():
            continue
        collection = json.loads(path.read_text(encoding="utf-8"))
        by_id = {e.object_id: e for e in edits}
        for feature in collection["features"]:
            edit = by_id.get(feature["properties"].get("object_id"))
            if edit:
                feature["properties"].update(reviewed_label=edit.reviewed_label, rejected=edit.rejected)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(collection), encoding="utf-8")
        temporary.replace(path)
    pixel_path = directory / "image_features.json"
    image_path = directory / "image.png"
    if pixel_path.is_file() and image_path.is_file():
        from src.inference.tiled import annotate
        from PIL import Image
        pixels = json.loads(pixel_path.read_text(encoding="utf-8"))
        with Image.open(image_path) as image:
            annotate(image.convert("RGB"), pixels["features"]).save(directory / "annotated.png")
    return {"message": "Reviewed labels saved to this result run."}


class TrainingRequest(BaseModel):
    manifest: str = Field(min_length=1)
    epochs: int = Field(default=10, ge=1, le=10000)
    batch_size: int = Field(default=16, ge=2, le=128)


@app.post("/api/train")
def train(request: TrainingRequest):
    from src.training.train_classifier import read_manifest
    manifest = Path(request.manifest).resolve()
    if jobs.active():
        raise ValueError("Wait for imagery analysis to finish or cancel it before starting training.")
    read_manifest(manifest)  # Validate splits and labels before starting a GPU job.
    if any(job["process"].poll() is None for job in _training_jobs.values()):
        raise ValueError("A training job is already running; wait for it or cancel it first.")
    job_id = uuid.uuid4().hex[:12]
    output = PROJECT_ROOT / "outputs/training" / job_id
    log_dir = PROJECT_ROOT / "outputs/logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"training_{job_id}.log"
    command = [sys.executable, "-m", "src.cli", "train", "--manifest", str(manifest), "--output", str(output),
               "--epochs", str(request.epochs), "--batch-size", str(request.batch_size)]
    with gpu_work():
        from src.inference.segmentation import unload_sam
        from src.language.ollama import unload_ollama
        unload_sam()
        unload_ollama(load_config()["ollama"])
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        set_training(process)
    _training_jobs[job_id] = {"process": process, "output": output, "log": log_path}
    return {"job_id": job_id, "output": str(output)}


@app.get("/api/train/{job_id}")
def training_status(job_id: str):
    job = _training_jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown training job.")
    code = job["process"].poll()
    return {"job_id": job_id, "status": "running" if code is None else "completed" if code == 0 else "failed",
            "output": str(job["output"]), "log": job["log"].read_text(encoding="utf-8", errors="replace")[-16000:]}


@app.post("/api/train/{job_id}/cancel")
def cancel_training(job_id: str):
    job = _training_jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown training job.")
    if job["process"].poll() is None:
        job["process"].terminate()
    return {"message": "Cancellation requested; previously saved best checkpoint is retained."}


@app.get("/api/checkpoints")
def checkpoints():
    root = PROJECT_ROOT / "outputs/training"
    return {"checkpoints": [str(path.relative_to(root)).replace("\\", "/") for path in root.glob("*/best.pt")]}


@app.post("/api/classify")
def classify(image: UploadFile = File(...), checkpoint: str = Form(...)):
    from src.training.train_classifier import classify_image
    root = (PROJECT_ROOT / "outputs/training").resolve()
    path = (root / checkpoint).resolve()
    if not path.is_relative_to(root) or not path.is_file() or path.name != "best.pt":
        raise ValueError("Select a trained checkpoint from outputs/training.")
    uploaded = save_upload(image)
    try:
        with gpu_work():
            from src.inference.segmentation import unload_sam
            from src.language.ollama import unload_ollama
            unload_sam()
            unload_ollama(load_config()["ollama"])
            return classify_image(path, uploaded)
    finally:
        uploaded.unlink(missing_ok=True)
