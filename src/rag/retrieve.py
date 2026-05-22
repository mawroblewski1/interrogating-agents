"""
Two-channel retrieval from the shared ChromaDB index.

Channel 1 — retrieve_techniques:
  Filters by phase suitability (early/middle/late/any) before semantic search.
  Returns a list of TechniqueCard dicts.

Channel 2 — retrieve_arguments:
  Filters by {type: argument, topic: <id>, side: <for|against>}.
  Returns a list of argument strings ranked by relevance to the query.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

# Prevent sentence-transformers from making network calls on every load.
# The model is already cached from the initial index build; offline mode
# ensures a dropped WiFi connection can never crash a running experiment.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import chromadb
import yaml
from chromadb.utils import embedding_functions

BASE        = Path(__file__).parent.parent.parent
CHROMA_PATH = BASE / "chroma_db"
COLLECTION  = "interrogation_corpus"
EMBED_MODEL = "all-MiniLM-L6-v2"

_PHASE_MAP = {
    "early":  {"early", "any"},
    "middle": {"middle", "any"},
    "late":   {"late", "any"},
}

# Module-level cache: load the embedding model and open the ChromaDB
# collection exactly once per process instead of once per retrieve call.
_collection_cache: chromadb.Collection | None = None


@dataclass
class TechniqueCard:
    name: str
    phase: str
    stem: str
    body: str
    triggers: list[str] = field(default_factory=list)
    pairs_with: list[str] = field(default_factory=list)
    in_tension_with: list[str] = field(default_factory=list)
    out_of_bounds: list[str] = field(default_factory=list)


def _get_collection() -> chromadb.Collection:
    global _collection_cache
    if _collection_cache is None:
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        )
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        _collection_cache = client.get_collection(COLLECTION, embedding_function=ef)
    return _collection_cache


def _load_technique_meta(stem: str) -> dict:
    """Load the full front-matter from a technique card file."""
    path = BASE / "data" / "techniques" / f"{stem}.md"
    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    if match:
        return yaml.safe_load(match.group(1))
    return {}


def retrieve_techniques(
    query: str,
    turn: int,
    budget: int,
    k: int = 5,
) -> list[TechniqueCard]:
    """Return up to k technique cards suited to the current conversation phase.

    Phase is inferred from turn / budget:
      early  — turn <= budget // 3
      late   — turn > (2 * budget) // 3
      middle — otherwise
    """
    if turn <= budget // 3:
        phase = "early"
    elif turn > (2 * budget) // 3:
        phase = "late"
    else:
        phase = "middle"

    allowed_phases = list(_PHASE_MAP[phase])

    collection = _get_collection()
    results = collection.query(
        query_texts=[query],
        n_results=min(k * 3, 30),   # fetch extra, then filter by phase
        where={"type": "technique"},
    )

    cards: list[TechniqueCard] = []
    docs      = results["documents"][0]
    metas     = results["metadatas"][0]

    for doc, meta in zip(docs, metas):
        if meta["phase"] not in allowed_phases:
            continue
        full_meta = _load_technique_meta(meta["stem"])
        cards.append(TechniqueCard(
            name=meta["name"],
            phase=meta["phase"],
            stem=meta["stem"],
            body=doc,
            triggers=full_meta.get("triggers", []),
            pairs_with=full_meta.get("pairs_with", []),
            in_tension_with=full_meta.get("in_tension_with", []),
            out_of_bounds=full_meta.get("out_of_bounds", []),
        ))
        if len(cards) >= k:
            break

    return cards


def retrieve_arguments(
    topic_id: str,
    side: str,
    query: str,
    k: int = 3,
) -> list[str]:
    """Return up to k arguments for the given topic and side, ranked by relevance.

    Args:
        topic_id: e.g. "housing_prop_123"
        side:     "for" or "against" — the interrogator's side
        query:    the current conversational moment (suspect's latest utterance + context)
        k:        max arguments to return
    """
    assert side in ("for", "against"), f"side must be 'for' or 'against', got {side!r}"

    collection = _get_collection()
    results = collection.query(
        query_texts=[query],
        n_results=k,
        where={"$and": [
            {"type": {"$eq": "argument"}},
            {"topic": {"$eq": topic_id}},
            {"side": {"$eq": side}},
        ]},
    )

    return results["documents"][0] if results["documents"] else []
