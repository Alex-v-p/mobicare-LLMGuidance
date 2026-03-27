from __future__ import annotations

from typing import Any, Callable

from inference.clinical.config_repository import load_drug_dosing_catalog_payload


CatalogLoader = Callable[[], dict[str, Any]]


def drug_dosing_catalog(*, loader: CatalogLoader = load_drug_dosing_catalog_payload) -> dict[str, Any]:
    return loader()


def family_priority(*, loader: CatalogLoader = load_drug_dosing_catalog_payload) -> dict[str, int]:
    raw = drug_dosing_catalog(loader=loader).get("family_priority") or {}
    return {str(key): int(value) for key, value in raw.items()}


def default_agent(family: str, *, loader: CatalogLoader = load_drug_dosing_catalog_payload) -> str:
    raw = drug_dosing_catalog(loader=loader).get("default_agents") or {}
    value = raw.get(family)
    return str(value or family)


def family_query_order(*, loader: CatalogLoader = load_drug_dosing_catalog_payload) -> tuple[str, ...]:
    raw = drug_dosing_catalog(loader=loader).get("family_query_order") or []
    return tuple(str(item) for item in raw)


def family_keywords(*, loader: CatalogLoader = load_drug_dosing_catalog_payload) -> dict[str, set[str]]:
    families = drug_dosing_catalog(loader=loader).get("families") or {}
    return {
        str(family): {str(keyword).lower() for keyword in (spec.get("keywords") or [])}
        for family, spec in families.items()
    }


def family_query_template(family: str, *, loader: CatalogLoader = load_drug_dosing_catalog_payload) -> str:
    families = drug_dosing_catalog(loader=loader).get("families") or {}
    spec = families.get(family) or {}
    return str(spec.get("query_template") or "")
