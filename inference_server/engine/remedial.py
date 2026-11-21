"""
Remedial-Measures Lookup
========================
Serves remedial / management recommendations alongside Expert View results.

Currently backed by a placeholder JSON (``data/remedial_measures.json``); the
agronomy-reviewed Q1 "remedial measures database" content replaces that file
without any code change.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any

from inference_server import config


logger = logging.getLogger("inference_server.remedial")


@lru_cache(maxsize=1)
def _load_db() -> dict[str, Any]:
    path = config.REMEDIAL_DB_PATH
    if not path.exists():
        logger.warning("Remedial DB not found at %s; serving empty recommendations.", path)
        return {"_meta": {"status": "MISSING"}, "diseases": {}}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_remedial(disease_name: str) -> dict[str, Any]:
    """Return remedial content for a disease, falling back to a generic entry."""
    db = _load_db()
    diseases = db.get("diseases", {})
    entry = diseases.get(disease_name) or diseases.get("_default") or {}
    return {
        "status": db.get("_meta", {}).get("status", "UNKNOWN"),
        "disease": disease_name,
        **entry,
    }
