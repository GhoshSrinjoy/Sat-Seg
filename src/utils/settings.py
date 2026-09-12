"""Validated settings shared by the API, CLI and inference workers."""

from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SettingsSection(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


def http_url(value):
    if not value:
        return value
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Enter an HTTP(S) URL without embedded credentials.")
    return value.rstrip("/")


class OllamaSettings(SettingsSection):
    base_url: str = "http://localhost:11434"
    model: str | None = None
    enabled: bool = False
    temperature: float = Field(0, ge=0, le=2)
    timeout: int = Field(120, ge=10, le=600)
    context_length: int = Field(4096, ge=1024, le=32768)
    unload_after_query: bool = True

    _url = field_validator("base_url")(http_url)

    @field_validator("model")
    @classmethod
    def model_name(cls, value):
        if value is not None and len(value) > 200:
            raise ValueError("Model name is too long.")
        return value.strip() or None if value else None


class SamSettings(SettingsSection):
    model_id: str = "facebook/sam3"
    cache_dir: str = "D:/data/models"
    revision: str = "3c879f39826c281e95690f02c7821c4de09afae7"
    local_files_only: Literal[True] = True
    device: Literal["cuda", "cpu"] = "cuda"
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    prompt: str = Field("building", min_length=1, max_length=200)
    threshold: float = Field(.5, ge=0, le=1)
    mask_threshold: float = Field(.5, ge=0, le=1)
    tile_size: int = Field(1008, ge=256, le=1008)
    tile_overlap: int = Field(96, ge=16, le=256)
    merge_overlap: float = Field(.6, ge=.1, le=1)
    min_object_pixels: int = Field(16, ge=1, le=100000)

    @model_validator(mode="after")
    def check_overlap(self):
        if self.tile_overlap >= self.tile_size / 2:
            raise ValueError("Tile overlap must be less than half the tile size.")
        if not self.prompt.strip():
            raise ValueError("Enter a default visual concept.")
        return self


class ImagerySettings(SettingsSection):
    provider: Literal["auto", "naip", "berlin", "custom_wms"] = "auto"
    resolution_m: float = Field(.6, ge=.2, le=20)
    max_pixels: int = Field(4_000_000, ge=65536, le=16_000_000)
    max_tiles: int = Field(25, ge=1, le=64)
    cache_seconds: int = Field(86400, ge=0, le=2592000)
    cache_mb: int = Field(512, ge=16, le=4096)
    custom_url: str = ""
    custom_layer: str = ""
    custom_attribution: str = "User-configured imagery"
    custom_token: str = ""

    _url = field_validator("custom_url")(http_url)

    @model_validator(mode="after")
    def custom_source(self):
        if self.provider == "custom_wms" and (not self.custom_url or not self.custom_layer.strip()):
            raise ValueError("Custom WMS needs a service URL and imagery layer name.")
        return self


class MapSettings(SettingsSection):
    photon_url: str = "https://photon.komoot.io"
    overpass_url: str = "https://overpass-api.de/api/interpreter"
    max_area_km2: float = Field(25, gt=0, le=25)
    cache_seconds: int = Field(86400, ge=0, le=2592000)
    user_agent: str = "sat-clas/0.3 local-research (interactive map explorer)"
    default_lat: float = Field(37.8716, ge=-85, le=85)
    default_lon: float = Field(-122.2727, ge=-180, le=180)
    default_zoom: int = Field(16, ge=2, le=20)
    show_labels: bool = True
    enrich: bool = True
    auto_analyse: bool = True

    _url = field_validator("photon_url", "overpass_url")(http_url)


class CompareSettings(SettingsSection):
    alignment: Literal["auto", "same_grid", "manual"] = "auto"
    min_change_pixels: int = Field(25, ge=1, le=100000)
    edge_tolerance: int = Field(1, ge=0, le=10)


class AppSettings(SettingsSection):
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    sam3: SamSettings = Field(default_factory=SamSettings)
    imagery: ImagerySettings = Field(default_factory=ImagerySettings)
    map: MapSettings = Field(default_factory=MapSettings)
    comparison: CompareSettings = Field(default_factory=CompareSettings)


def default_sections():
    return AppSettings().model_dump()


def public_settings(config):
    result = AppSettings.model_validate({k: config[k] for k in AppSettings.model_fields}).model_dump()
    result["imagery"]["has_custom_token"] = bool(result["imagery"].pop("custom_token"))
    return result
