"""Local model discovery and schema-validated geographic planning."""

import requests

from src.inference.runtime import gpu_work


def list_models(settings):
    try:
        response = requests.get(settings["base_url"].rstrip("/") + "/api/tags", timeout=(3, 5))
        response.raise_for_status()
        models = []
        for model in response.json().get("models", []):
            remote = bool(model.get("remote_host") or model.get("remote_model") or "cloud" in model["name"])
            embedding = "embed" in model["name"].lower()
            models.append({"name": model["name"], "size": model.get("size", 0), "details": model.get("details", {}),
                           "selectable": not remote and not embedding,
                           "note": "Cloud model; choose a local model" if remote else "Embedding model; cannot plan queries" if embedding else ""})
        return {"connected": True, "models": models, "message": f"Connected. {len(models)} models found; choose a local text model."}
    except (requests.RequestException, ValueError, KeyError):
        return {"connected": False, "models": [],
                "message": "Ollama is unreachable. Start Ollama, check the server URL, then refresh models."}


def unload_ollama(settings):
    if not settings.get("model"):
        return
    try:
        response = requests.post(settings["base_url"].rstrip("/") + "/api/generate",
                                 json={"model": settings["model"], "keep_alive": 0}, timeout=(3, 30))
        response.raise_for_status()
    except requests.ConnectionError:
        # An offline daemon holds no app-owned model on this host.
        return
    except requests.RequestException as error:
        raise RuntimeError("Could not release the configured Ollama model before SAM inference.") from error


def query_plan(query, settings, context=None):
    from src.language.query_parser import QueryPlan
    from src.inference.segmentation import unload_sam
    import json

    if not settings.get("model"):
        raise ValueError("Choose an installed Ollama model in Settings first.")
    if "cloud" in settings["model"] or "embed" in settings["model"].lower():
        raise ValueError("Choose a local text generation model for geographic planning.")
    instructions = (
        "Translate the geographic request into the provided schema. Do not invent coordinates, names or addresses. "
        "intent: address, features, coordinates, geohash, compare, route, segment, enrich, filter. "
        "Use segment for visual detection, SAM, satellite, aerial imagery or outlining visible objects; "
        "prompt is a short visible concept, e.g. building/tree/water. Multiple concepts may be comma-separated. "
        "Use features for map records, including hospitals, schools, restaurants and businesses; "
        "visual appearance alone cannot establish a building's use. "
        "Categories: buildings,hospitals,schools,parks,water,roads,hiking,restaurants,shops,trees,all. "
        "Use tag key=value for another OSM category. source is auto, maps, imagery or combined. "
        "area is only an explicitly named area, otherwise null (this area/here/selected box refer to context). "
        "Keep the complete address in query for address intent. radius_m is an explicitly requested distance "
        "around a named location, between 50 and 2000 metres. "
        "Existing hiking routes use features/hiking; directions use route. "
        "For a follow-up to attach map names/addresses to existing detections use enrich. "
        "For a follow-up asking for results without addresses use filter with filter=missing_address; "
        "filter=has_address or all also supported. Use compare for before/after images."
    )
    with gpu_work():
        unload_sam()
        try:
            response = requests.post(settings["base_url"].rstrip("/") + "/api/chat", json={
                "model": settings["model"], "stream": False, "think": False, "format": QueryPlan.model_json_schema(),
                "options": {"temperature": settings.get("temperature", 0),
                            "num_ctx": settings.get("context_length", 4096), "num_predict": 768},
                "keep_alive": 0 if settings.get("unload_after_query", True) else "5m",
                "messages": [{"role": "system", "content": instructions},
                             {"role": "user", "content": json.dumps({"query": query, "context": context or {}})}],
            }, timeout=(5, settings.get("timeout", 120)))
            response.raise_for_status()
            return QueryPlan.model_validate_json(response.json()["message"]["content"])
        except requests.RequestException as error:
            raise RuntimeError("Ollama query failed. Check the model and connection in Settings, or turn off Ollama for basic search.") from error
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError("This model did not return a valid geographic plan. Test another installed model in Settings.") from error
