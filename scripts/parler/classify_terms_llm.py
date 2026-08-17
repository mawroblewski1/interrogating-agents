"""
Classify raw scraped terms (from mediawiki_scrape.py, or any newline/ndjson term list)
into your lexicon's topic/signal buckets, producing a DRAFT file for manual review --
this never writes directly to lexicon.txt.

Backend-agnostic: --backend selects which classifier actually does the work. See
classifier_backends.py for details on each.

  --backend anthropic (default)  needs internet + ANTHROPIC_API_KEY, best accuracy on
                                  nuanced/adjacent-category calls, gives a real reason
  --backend local                offline zero-shot NLI, no API key/cost, weaker on
                                  nuanced calls, no free-text reason
                                  (pip install transformers torch --break-system-packages)
  --backend ollama               offline generative LLM via a locally-run model, no
                                  API key/cost, real free-text reason, quality depends
                                  on the local model (needs `ollama serve` running)

Most scraped terms will legitimately be "not_extremist" -- that's expected and fine,
it's the classifier correctly recognizing ordinary vocabulary.

Example:
  export ANTHROPIC_API_KEY=sk-...
  python3 classify_terms_llm.py --terms wiktionary_terms.ndjson --backend anthropic \
      --out draft_lexicon_candidates.tsv --batch-size 40

  python3 classify_terms_llm.py --terms wiktionary_terms.ndjson --backend local \
      --out draft_lexicon_candidates.tsv

  ollama pull llama3.1   # once
  python3 classify_terms_llm.py --terms wiktionary_terms.ndjson --backend ollama \
      --out draft_lexicon_candidates.tsv

Then MANUALLY REVIEW draft_lexicon_candidates.tsv (same pattern as shortlist.tsv):
delete/edit rows you disagree with, then:
  python3 build_lexicon_from_review.py --draft draft_lexicon_candidates.tsv \
      --lexicon lexicon.txt
"""
import argparse
import json
import sys
import time
import classifier_backends

# ---------- defaults (edit these to change built-in behavior; overridable on the CLI) ----------
DEFAULT_OUT = "draft_lexicon_candidates.tsv"
DEFAULT_BACKEND = "anthropic"    # one of classifier_backends.BACKEND_REGISTRY: anthropic/local/ollama/fixed
DEFAULT_BATCH_SIZE = 40          # terms per API call (anthropic/ollama only; local/fixed do one at a time)
DEFAULT_MIN_CONFIDENCE = 0.0     # drop non-"not_extremist" rows below this; 0 = keep everything for review


def load_terms(path: str) -> list[str]:
    terms = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                terms.append(json.loads(line)["term"])
            else:
                terms.append(line)
    return terms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terms", required=True,
                     help="ndjson from mediawiki_scrape.py, or a plain newline-delimited term list")
    ap.add_argument("--out", default=DEFAULT_OUT, help="draft TSV to write (term/label/confidence/reason)")
    ap.add_argument("--backend", default=DEFAULT_BACKEND, choices=list(classifier_backends.BACKEND_REGISTRY),
                     help="which classifier does the work (see classifier_backends.py)")
    ap.add_argument("--model", default=None,
                     help="override the backend's default model (e.g. a specific Anthropic model "
                          "string, a HuggingFace zero-shot model for --backend local, or an "
                          "Ollama model name for --backend ollama); ignored by --backend fixed")
    ap.add_argument("--ollama-host", default=None,
                     help="override Ollama's default http://localhost:11434 (--backend ollama only)")
    ap.add_argument("--label", default=None,
                     help="the single bucket label to assign every term to (REQUIRED for --backend "
                          "fixed, ignored by every other backend, e.g. anti_vaccine, "
                          "signal:trollish, not_extremist)")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                     help="terms per call (anthropic/ollama backends only -- local/fixed classify "
                          "one term at a time regardless of this value)")
    ap.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE,
                     help="drop rows below this confidence (0 = keep everything for review); "
                          "not_extremist rows are always kept regardless of confidence")
    a = ap.parse_args()

    terms = load_terms(a.terms)
    # de-dupe while preserving order
    seen = set()
    terms = [t for t in terms if not (t.lower() in seen or seen.add(t.lower()))]
    print(f"[info] {len(terms)} unique term(s) to classify using backend={a.backend!r}")

    backend_kwargs = {"model": a.model}
    if a.backend == "ollama" and a.ollama_host:
        backend_kwargs["host"] = a.ollama_host
    if a.backend == "fixed":
        backend_kwargs["label"] = a.label
    backend = classifier_backends.get_backend(a.backend, **backend_kwargs)

    rows = []
    for i in range(0, len(terms), a.batch_size):
        batch = terms[i:i + a.batch_size]
        print(f"[info] classifying batch {i // a.batch_size + 1} "
              f"({i + 1}-{min(i + len(batch), len(terms))} of {len(terms)})", file=sys.stderr)
        results = backend.classify_batch(batch)
        by_term = {r.get("term", "").lower(): r for r in results if isinstance(r, dict)}
        for t in batch:
            r = by_term.get(t.lower(), {"label": "not_extremist", "confidence": 0.0,
                                         "reason": "MISSING_FROM_RESPONSE -- review manually"})
            rows.append((t, r.get("label", "not_extremist"), r.get("confidence", 0.0), r.get("reason", "")))
        time.sleep(0.3)

    n_before = len(rows)
    if a.min_confidence > 0:
        rows = [r for r in rows if r[1] == "not_extremist" or r[2] >= a.min_confidence]

    n_flagged = sum(1 for r in rows if r[1] != "not_extremist")
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("term\tlabel\tconfidence\treason\n")
        for term, label, conf, reason in rows:
            f.write(f"{term}\t{label}\t{conf:.2f}\t{reason}\n")

    print(f"[done] {len(rows)}/{n_before} row(s) written ({n_flagged} flagged as potentially "
          f"extremist, rest are not_extremist) -> {a.out}")
    print("[next] MANUALLY REVIEW this file -- delete/correct rows you disagree with, then:")
    print(f"       python3 build_lexicon_from_review.py --draft {a.out} --lexicon lexicon.txt")


if __name__ == "__main__":
    main()
