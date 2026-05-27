"""
Build and persist the ChromaDB embeddings index from technique cards and topic arguments.

Usage:
    python -m src.rag.index

Rebuilds from scratch every run. The index is stored in chroma_db/ (gitignored).
Two document types share one collection:
  type=technique  — one document per technique card (full prose body)
  type=argument   — one document per key_argument string, tagged by topic + side
"""
from __future__ import annotations

import re
from pathlib import Path

import chromadb
import yaml
from chromadb.utils import embedding_functions

BASE = Path(__file__).parent.parent.parent
TECHNIQUES_DIR = BASE / "data" / "techniques"
TOPICS_DIR     = BASE / "data" / "topics"
CHROMA_PATH    = BASE / "chroma_db"
COLLECTION     = "interrogation_corpus"
EMBED_MODEL    = "all-MiniLM-L6-v2"


def _parse_technique_card(path: Path) -> dict:
    """Split a technique .md file into front-matter dict and prose body."""
    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)
    if not match:
        raise ValueError(f"Technique card missing YAML front-matter: {path}")
    meta = yaml.safe_load(match.group(1))
    body = match.group(2).strip()
    return {"meta": meta, "body": body, "path": path}


def build_index() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    # Always rebuild from scratch.
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    collection = client.create_collection(COLLECTION, embedding_function=ef)

    documents: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    # --- Technique cards ---
    technique_paths = sorted(TECHNIQUES_DIR.glob("*.md"))
    print(f"Indexing {len(technique_paths)} technique cards...")
    for path in technique_paths:
        card = _parse_technique_card(path)
        meta = card["meta"]
        doc_id = f"technique::{path.stem}"
        documents.append(card["body"])
        metadatas.append({
            "type":     "technique",
            "name":     meta.get("name", path.stem),
            "phase":    meta.get("phase", "any"),
            "stem":     path.stem,
        })
        ids.append(doc_id)

    # --- Topic arguments ---
    topic_paths = sorted(TOPICS_DIR.glob("*.yaml"))
    print(f"Indexing arguments from {len(topic_paths)} topic cards...")
    for path in topic_paths:
        with open(path, encoding="utf-8") as f:
            card = yaml.safe_load(f)
        topic_id = card["id"]
        for side, section_key in [("for", "supporters"), ("against", "opponents")]:
            for i, arg in enumerate(card[section_key]["key_arguments"]):
                doc_id = f"argument::{topic_id}::{side}::{i}"
                documents.append(arg)
                metadatas.append({
                    "type":  "argument",
                    "topic": topic_id,
                    "side":  side,
                })
                ids.append(doc_id)

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print(f"Index built: {len(ids)} documents ({len(technique_paths)} techniques, "
          f"{len(ids) - len(technique_paths)} arguments).")
    return collection


if __name__ == "__main__":
    build_index()
