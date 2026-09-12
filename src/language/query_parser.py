"""Parse frequent map requests, optionally using an explicitly configured Ollama model."""

import re
from typing import Literal

from pydantic import BaseModel, Field

from src.utils.configuration import load_config


class QueryPlan(BaseModel):
    intent: Literal["address", "features", "coordinates", "geohash", "compare", "route", "segment", "enrich", "filter"]
    query: str = Field(max_length=1000)
    category: str | None = None
    area: str | None = None
    tag: str | None = None
    explanation: str = ""
    source: Literal["auto", "maps", "imagery", "combined"] = "auto"
    prompt: str | None = Field(None, max_length=200)
    radius_m: float | None = Field(None, ge=50, le=2000)
    filter: Literal["all", "missing_address", "has_address"] = "all"


def parse_query(query, use_ollama=False, context=None):
    query = query.strip()
    if not query or len(query) > 1000:
        raise ValueError("Enter a query of 1–1000 characters.")
    lower = query.lower()
    # Coordinates and explicit geohashes do not need a model round trip.
    if re.match(r"^(?:geohash|hash)\s*[:= ]", lower):
        return QueryPlan(intent="geohash", query=re.sub(r"^\w+\s*[:= ]\s*", "", lower))
    if re.fullmatch(r"\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*", query):
        return QueryPlan(intent="coordinates", query=query, explanation="Input order is latitude, longitude.")
    if use_ollama:
        from src.language.ollama import query_plan
        settings = load_config().get("ollama", {})
        return query_plan(query, settings, context)
    if re.search(r"\b(without|missing|no)\s+(?:a |an |mapped )?addresses?\b", lower):
        return QueryPlan(intent="filter", query=query, filter="missing_address")
    if re.search(r"\b(label|enrich|attach|match)\b", lower) and re.search(r"\b(addresses|names)\b", lower):
        return QueryPlan(intent="enrich", query=query)
    if re.search(r"\b(compare|change|changes|past|before|after|transition)\b", lower):
        return QueryPlan(intent="compare", query=query)
    if re.search(r"\b(from .+ to|directions|navigate)\b", lower):
        return QueryPlan(intent="route", query=query)
    categories = {
        "hospitals": r"hospitals?|hospilarts|hospitals", "schools": r"schools?",
        "hiking": r"hiking|hike|walking routes?", "buildings": r"buildings?|buldings?",
        "parks": r"parks?", "water": r"water|rivers?|lakes?", "roads": r"roads?|streets?",
        "trees": r"trees?|forests?", "restaurants": r"restaurants?", "shops": r"shops?|stores?",
        "all": r"all (?:map )?(?:features|elements)",
    }
    tag_match = re.search(r"\b([a-zA-Z_:]+=[a-zA-Z0-9_:;-]+)\b", query)
    category = next((name for name, pattern in categories.items() if re.search(rf"\b(?:{pattern})\b", lower)), None)
    visual = bool(re.search(r"\b(sam|segment|satellite|imagery|aerial|outline|detect)\b", lower))
    if visual:
        area_match = re.search(r"\b(?:in|near|around)\s+(.+?)[?.!]*$", query, re.I)
        area = area_match.group(1).strip() if area_match else None
        if area and re.fullmatch(r"(?:(?:this|the|current|selected)\s+)?(?:area|map|view|box)|here", area, re.I):
            area = None
        concepts = [singular for plural, singular in (("buildings", "building"), ("trees", "tree"),
                    ("water", "water"), ("roads", "road")) if re.search(rf"\b(?:{categories[plural]})\b", lower)]
        concept = ", ".join(concepts) or re.sub(r"^(?:please\s+)?(?:segment|detect|outline)\s+", "", query, flags=re.I)
        if area_match:
            concept = concept.split(" in ")[0].split(" near ")[0].split(" around ")[0]
        if category in {"hospitals", "schools", "restaurants", "shops", "parks"}:
            return QueryPlan(intent="features", query=query, category=category, area=area, source="combined", prompt="building")
        return QueryPlan(intent="segment", query=query, area=area, prompt=concept[:200], source="imagery")
    # House-number searches win over category words in street/place names.
    address_hint = bool(re.search(r"\b\d+[a-z]?\b", lower))
    if (category or tag_match) and not address_hint:
        area_match = re.search(r"\b(?:in|near|around)\s+(.+?)[?.!]*$", query, re.I)
        area = area_match.group(1).strip() if area_match else None
        if area and re.fullmatch(r"(?:this|the|current)\s+(?:area|map|view)|here", area, re.I):
            area = None
        return QueryPlan(intent="features", query=query, category=category or "all", area=area,
                         tag=tag_match.group(1) if tag_match else None)
    address = re.sub(r"^(?:ok[, ]+)?(?:where\s+is|find|locate|show(?:\s+me)?|mark)\s+", "", query, flags=re.I)
    address = re.sub(r"^(?:the\s+)?house\s+numbers?\s+", "", address, flags=re.I).strip(" ?!")
    return QueryPlan(intent="address", query=address)
