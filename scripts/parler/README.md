# Parler Persona Pipeline

Turns a raw Parler dump (~4M users / ~183M posts across zip file(s) of NDJSON records)
into two things:

1. ~100 manually curated extremist user **personas**, each with a full post history —
   used to simulate "suspects" and personalize an interrogator system to each user.
2. A gzip-compressed **reserve corpus** of extremist posts, held in case of later
   fine-tuning.

The dataset is too large to load into memory, and one user's posts are scattered across
files, so the pipeline is a funnel: stream the data in cheap passes, narrow the pool at
each stage, and only pull full text for the ~100 survivors.

Only Python stdlib is required for the core funnel and lexicon-building tools. The one
exception is `06_bucket_frequency.py`'s optional `--chart` flag, which uses `matplotlib`
to render a static, sorted horizontal bar chart (highest-scoring bucket on top, like a
ranked scoreboard rather than an animated race) — everything else in that script (and
every other file) works without it. Each run produces one chart; `--chart-metric`
picks which of the three columns (raw or `@weight`-normalized) it plots — see
"Bucket-frequency analysis" below.

## Active files

| File | Role |
|---|---|
| `parler_io.py` | Library — streaming zip/NDJSON reader (handles truncated downloads too) |
| `01_pass1_stats_corpus.py` | Stage 1 — full stream, per-user stats + reserve corpus |
| `02_select_candidates.py` | Stage 2 — in-memory, threshold filter |
| `03_fetch_candidate_posts.py` | Stage 3 — full stream, sample posts per candidate |
| `04_classify_select.py` | Stage 4 — in-memory, topic/signal scoring + diverse shortlist |
| `05_pass2_histories.py` | Stage 5 — full stream, final full histories |
| `lexicon.txt` | The keyword/topic-bucket data itself |
| `lexicon_io.py` | Library — shared lexicon parser (used by everything below) |
| `validate_lexicon.py` | Lints `lexicon.txt` before it's used |
| `mediawiki_scrape.py` | Scrapes candidate terms from Wiktionary/Wikipedia URLs |
| `classifier_backends.py` | Library — swappable term-classification backends |
| `classify_terms_llm.py` | Classifies a candidate term list into buckets (draft output) |
| `build_lexicon_from_review.py` | Merges a *reviewed* draft into `lexicon.txt` |
| `06_bucket_frequency.py` | Standalone analysis — bucket hit-frequency + leaderboard chart |
| `run_pipeline.sh` | Orchestrator — both the numbered-stage funnel and lexicon-building |
| `smoke_test_lexicon_tools.sh` | Standalone test of the lexicon-building tool chain |

## Quick start

```bash
chmod +x run_pipeline.sh
./run_pipeline.sh --input "/path/to/parler_folder" --limit 40000   # smoke test
./run_pipeline.sh --input "/path/to/parler_folder"                 # full run, stages 1-4
#   review shortlist.tsv by hand, save kept creator ids to final_users.txt
./run_pipeline.sh --input "/path/to/parler_folder" --from 5 --to 5 # final histories pass
```

`run_pipeline.sh` sets up its own `.venv` on first run. It validates `lexicon.txt`
automatically before any stage that uses it, and refuses to run Stage 5 until
`final_users.txt` exists.

## The core five-stage funnel

| Stage | Reads | Writes | Scope |
|---|---|---|---|
| 1 | the zip(s) | `user_stats.tsv.gz`, `extremist_corpus.ndjson.gz` | full stream |
| 2 | `user_stats.tsv.gz` | `candidates.tsv` | in-memory |
| 3 | the zip(s) + `candidates.tsv` | `candidate_posts.ndjson.gz` | full stream |
| 4 | `candidate_posts.ndjson.gz` + `lexicon.txt` | `shortlist.tsv` | in-memory |
| — | `shortlist.tsv` (manual review) | `final_users.txt` | human |
| 5 | the zip(s) + `final_users.txt` | `histories/<creator>.ndjson` | full stream |

Key `run_pipeline.sh` options: `--from`/`--to`/`--stage` to run a subset,
`--limit` (applies to Stages 1 and 3, for fast smoke tests), `--outdir`, `--lexicon`,
`--min-text-posts`, `--min-hits`, `--max-candidates`, `--sample`, `--per-topic`.
Every script's own defaults are listed as named constants near the top of the file, and
every flag has `--help` text — run any script with `--help` for the full list.

### How each stage actually works

**Stage 1** — single streaming pass. Maintains a hash map keyed by creator id, holding a
running aggregate per user (post count, text-post count, total chars, min/max
timestamp, cumulative lexicon-hit count, max followers seen). For each record, updates
that user's aggregate in place; `ext_hits` is *presence*-based, not occurrence-based —
each lexicon term contributes at most 1 per post regardless of repetition, summed
across a user's whole history. Any post with at least one hit is also opportunistically
written into the reserve corpus (Output B), capped at `--corpus-cap`. Complexity is
O(records × lexicon terms) for the substring scan, O(unique creators) memory — the
reason Stage 1 needs a real-memory machine even though nothing else in the funnel does.

**Stage 2** — pure in-memory filter + sort, no streaming. Loads the Stage 1 stats table,
keeps rows clearing both `--min-text-posts` and `--min-hits`, sorts survivors by
`(ext_hits, total_chars)` descending, truncates to `--max-candidates`.

**Stage 3** — a second streaming pass, but now with a small candidate-id set held in
memory. For each record, an O(1) hash lookup checks if the creator is a candidate who
hasn't yet hit their `--sample` cap; if so, the post is trimmed and written out. Only
the candidate set needs to be scanned against — this is why Stage 3 is a full stream
(it needs to see the whole dataset once to gather candidates' scattered posts) but stays
fast relative to Stage 1.

**Stage 4** — in-memory scoring, the most algorithmically involved stage; see
"Classifying vocabulary" below for the full weighting mechanics.

**Stage 5** — a third and final streaming pass, symmetric to Stage 3 but against the
much smaller final-approved-user set, writing one output file per user as it goes.

### Classifying vocabulary — two related but distinct systems

The pipeline classifies vocabulary in two different places, for two different purposes.
It's easy to conflate them since both ultimately assign a "bucket" to something, so
worth being explicit about which is which:

**1. Post-level classification (Stage 4, `score_user()`)** — decides which topic a
*candidate user* belongs to, based on their sampled posts:
  - Concatenate a candidate's sampled posts into one lowercase blob.
  - For every **topic** bucket, sum `(raw substring count of each of its terms) ×
    (that term's resolved weight for this bucket)`. A term unique to one bucket
    effectively has weight 1.0 there; a term listed in multiple buckets (e.g.
    `globalist` in both `anti_government` and `antisemitic`) has its weight split —
    explicit `@weight` values are honored as given, and any bucket left unannotated for
    that term absorbs an even share of whatever weight remains.
  - The topic bucket with the highest weighted-hit total becomes the user's label
    (ties go to whichever section appears first in `lexicon.txt`); if every bucket
    scores zero, the label is `other`. `strength` is the sum of all topic-bucket
    scores divided by the number of sampled posts.
  - **Signal** buckets are scored identically (raw counts, no weighting) but are
    computed entirely separately — they never enter the topic-hit comparison or the
    strength calculation, just get reported as their own columns alongside the label.
  - Candidates are grouped by their resulting label, ranked by `strength` within each
    group, and only the top `--per-topic` per group survive into `shortlist.tsv` — the
    mechanism that keeps the final persona set topically diverse rather than dominated
    by whichever bucket happened to have the most candidates.

**2. Lexicon-term classification (`classify_terms_llm.py` + `classifier_backends.py`)**
— decides which bucket a *candidate vocabulary term* belongs to, when you're building
or extending `lexicon.txt` itself. This is a deliberately two-part design:
  - **Part 1 — backend classification.** Whichever backend you chose (`anthropic`,
    `local`, `ollama`, `fixed`) receives a batch of terms plus the shared label
    definitions, and returns one best-fit label per term. `anthropic`/`ollama` produce
    a real free-text reason via generative reasoning; `local` produces a bare
    entailment score with no reasoning (zero-shot NLI, not generation); `fixed`
    performs no classification at all — every term gets the one label you specified,
    confidence 1.0, by construction.
  - **Part 2 — human review, then merge.** The backend's output is never trusted
    directly or written to `lexicon.txt` — it lands in a draft TSV instead. A human
    edits that file, removing or correcting rows they disagree with. Only what survives
    that edit (and isn't labeled `not_extremist`) gets merged: `build_lexicon_from_review.py`
    skips anything already present (case-insensitive dedup against the real lexicon)
    and appends the rest into the matching bracketed section, creating that section if
    it doesn't exist yet. This second part is the actual gate — even a `fixed`-backend
    batch, which skips classification *uncertainty* entirely, still stops here by
    default, and `--auto-merge` is refused for anything that went through a model.

## The lexicon system

`lexicon.txt` uses bracketed sections:

```
[anti_vaccine]
vaccine
vax

[signal:trollish]
clown world
🤡

[antisemitic]
globalist @0.6
```

- `[topic_name]` sections compete for a user's topic label in Stage 4.
- `[signal:name]` sections are scored separately and never compete for the label
  (used for ambiguous markers like trollish/meme content).
- Terms before any header count toward Stage 1's coarse prefilter only.
- `@weight` splits a term's contribution across buckets it appears in (auto-split
  evenly if unannotated) so a shared term like "globalist" doesn't get double-counted
  between `anti_government` and `antisemitic`.

Run `python3 validate_lexicon.py lexicon.txt` any time you edit it — it catches
malformed headers, duplicate/empty sections, and bad weight values before a real run.

## Building the lexicon from scraped/candidate terms

```bash
# 1. scrape terms from a list of URLs you write yourself
cat > sources.txt << 'EOF'
https://en.wiktionary.org/wiki/Category:English_offensive_terms
https://en.wikipedia.org/wiki/Category:American_white_nationalists @1
EOF
python3 mediawiki_scrape.py --urls sources.txt --out scraped_terms.ndjson --max-depth 2

# 2. classify (pick a backend: anthropic / local / ollama / fixed)
export ANTHROPIC_API_KEY=sk-...
python3 classify_terms_llm.py --terms scraped_terms.ndjson --backend anthropic --out draft.tsv

# 3. manually review draft.tsv — delete/correct rows you disagree with

# 4. merge only the approved rows
python3 build_lexicon_from_review.py --draft draft.tsv --lexicon lexicon.txt
python3 validate_lexicon.py lexicon.txt
```

A trailing ` @N` on a `sources.txt` line overrides `--max-depth` for just that one URL
(e.g. `@1` above recurses that category only 1 level deep, regardless of the `--max-depth`
passed on the command line) — unrelated to `lexicon.txt`'s `@weight` despite reusing the
same character. They live in different files and are read by different code
(`mediawiki_scrape.py` vs. `lexicon_io.py`); there's no actual ambiguity in practice, but
worth knowing `@` means two different things depending on which file you're editing.

Or do steps 1–4 in one command via `run_pipeline.sh`'s second mode:

```bash
# per-term LLM classification, always stops for manual review
./run_pipeline.sh --scrape-lexicon --urls sources.txt --classify-backend anthropic

# whole list is one known bucket, no model call, still stops for review by default
./run_pipeline.sh --scrape-lexicon --urls sources.txt --bucket antisemitic

# same, but skip the review pause entirely (only allowed with --bucket)
./run_pipeline.sh --scrape-lexicon --urls sources.txt --bucket antisemitic --auto-merge
```

`--backend fixed --label <bucket>` (or the wrapper's `--bucket`) skips classification
uncertainty for lists you already know the answer to; it does not skip review unless
you also pass `--auto-merge`.

## Bucket-frequency analysis

A separate streaming pass, independent of the funnel and the lexicon-building tools:

```bash
python3 06_bucket_frequency.py --input "/path/to/parler_folder" --lexicon lexicon.txt \
    --out bucket_counts.tsv --chart leaderboard_raw.png --limit 500000
```

`bucket_counts.tsv` is plain TSV, directly readable in R (`read.delim(...)`) or pandas
(`read_csv(..., sep="\t")`) with no conversion step. It includes both raw weighted hit
counts and counts normalized by each bucket's `@weight`-adjusted effective term count
(`n_terms_effective`, `hits_per_effective_term`), so a bucket padded with more lexicon
terms isn't automatically over-represented.

The chart shows exactly one of those metrics per run — `--chart-metric` picks which:

```bash
# raw hit counts (default) -- can make bigger buckets look over-represented
python3 06_bucket_frequency.py --input "..." --chart leaderboard_raw.png \
    --chart-metric weighted_hits

# normalized by @weight-adjusted effective term count -- the fairer cross-bucket comparison
python3 06_bucket_frequency.py --input "..." --chart leaderboard_normalized.png \
    --chart-metric hits_per_effective_term
```

Run it twice with different `--chart`/`--chart-metric` values (as above) to get both a
raw and a normalized chart side by side — each invocation writes one PNG.

## Testing

```bash
chmod +x smoke_test_lexicon_tools.sh
./smoke_test_lexicon_tools.sh
```

Exercises the scrape → classify → merge → frequency chain against synthetic data.
Degrades gracefully without internet or `ANTHROPIC_API_KEY` (substitutes small synthetic
stand-ins so the rest of the chain still gets real coverage). Runs in a scratch temp
directory and never touches your real `lexicon.txt`.

## Extending to other lexicon sources

Nothing about `classify_terms_llm.py` is Wiktionary/Wikipedia-specific — it accepts
**any** newline-delimited term list, or ndjson with a `"term"` key, regardless of where
it came from. `mediawiki_scrape.py` is just one way to produce that input.

To pull terms from a different source — a CSV export, a research paper's term
appendix, another API — write a small script that emits the same shape and feed it
straight into the existing classify → review → merge chain:

```python
# example: convert a CSV column of terms into the ndjson format classify_terms_llm.py expects
import csv, json

with open("my_terms.csv") as f_in, open("my_scraped_terms.ndjson", "w") as f_out:
    for row in csv.DictReader(f_in):
        f_out.write(json.dumps({
            "term": row["term_column"],
            "source": "my_terms.csv",
            "kind": "custom_import",
        }) + "\n")
```

```bash
python3 classify_terms_llm.py --terms my_scraped_terms.ndjson --backend anthropic --out draft.tsv
# review draft.tsv, then merge as usual
python3 build_lexicon_from_review.py --draft draft.tsv --lexicon lexicon.txt
```

Or skip the ndjson step entirely — a plain text file, one term per line, works directly:

```bash
python3 classify_terms_llm.py --terms my_plain_wordlist.txt --backend anthropic --out draft.tsv
```

Adding a new *classification* backend (e.g. a different local model, a different
hosted API) means subclassing `ClassifierBackend` in `classifier_backends.py` and
registering it in `BACKEND_REGISTRY` — `classify_terms_llm.py` and `run_pipeline.sh`
don't need to change at all.

## Extending to other social media platforms

The pipeline's Parler-specific surface is narrower than it might look — it's
concentrated in two places:

1. **`parler_io.py`** — handles Parler's specific file format (zip archives of
   `*.ndjson`, plus a raw-deflate fallback for truncated zip downloads). A different
   platform's dump will need its own file-format handling here.
2. **Field names hardcoded inline in Stages 1, 3, and 5** — `creator`, `body`,
   `createdAt`, `followers` are referenced directly in those three scripts, not
   abstracted through `parler_io.py`.

Everything else — the lexicon system, validator, weighting, scraping/classification
tools, and the frequency analysis — already only assumes "a post is text with a
creator and a timestamp," and needs no changes.

**To add a platform:** write an adapter matching `parler_io.py`'s interface
(`find_inputs(path)` and `iter_records(paths)`, yielding dicts normalized to the same
`creator`/`body`/`createdAt`/`followers` shape), then point Stages 1/3/5 at it instead
of `parler_io`.

```python
# reddit_io.py -- sketch of an adapter matching parler_io.py's interface
import json, glob, os

def find_inputs(path: str) -> list[str]:
    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, "*.ndjson")))  # Reddit dumps: often plain/zst NDJSON, not zip
    return [path]

def iter_records(paths):
    for p in paths:
        with open(p, encoding="utf-8") as f:      # swap for a zstd stream reader if compressed
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                # Reddit submissions and comments are DIFFERENT shapes -- normalize both:
                body = r.get("body") or r.get("selftext") or r.get("title") or ""
                yield {
                    "creator": r.get("author"),
                    "body": body,
                    "createdAt": r.get("created_utc"),   # Unix timestamp, not ISO 8601 like Parler
                    "followers": r.get("author_karma"),  # Reddit has no follower count -- karma or
                                                          # account age is the closest substitute;
                                                          # consider dropping this column's meaning
                                                          # instead of forcing a bad proxy
                }
```

Platform-specific gotchas worth knowing before building an adapter:

- **Reddit** has no real equivalent of a follower count; submissions and comments are
  different object shapes (a submission has `title`/`selftext`, a comment has `body`);
  timestamps are Unix epoch (`created_utc`), not ISO 8601. Official API access is
  free at low volume via PRAW for research use, but bulk access outside the API has
  real ToS exposure — plan around paid/official access rather than unofficial scraping.
- **X/Twitter** maps more directly (tweet text, author, `created_at`, and a genuine
  per-account follower count all exist), but meaningful-volume API access is paid-tier
  only, and scraping the site directly is against its ToS with real legal precedent
  against scrapers. Budget for API cost rather than planning around scraping.

## Design notes

- Stage 4's `score_user()` is an explicit **placeholder** keyword scorer — swap in a
  real classifier or LLM call for actual research use, keeping the
  `(strength, topic, signal_hits)` return contract.
- Manual human review is a deliberate gate at two points: `shortlist.tsv` →
  `final_users.txt`, and every LLM-classified lexicon draft before merging. `--auto-merge`
  is refused for anything that went through model classification rather than a
  pre-known `--bucket`.
- `creator` is a hash; report aggregates, never re-identify individuals.
- Matching throughout is case-insensitive substring matching — no stemming, no
  punctuation normalization.
