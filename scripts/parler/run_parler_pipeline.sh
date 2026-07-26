#!/usr/bin/env bash
#
# run_parler_pipeline.sh — driver for the Parler persona pipeline (Kubuntu / bash)
#
# Sets up a venv, then runs stages 1-5 of the pipeline (see README.md), with
# a --from/--to range so you can (re)run just part of it — e.g. after the
# manual shortlist review, or after a crash mid-stream on the big zip.
#
# Usage:
#   ./run_parler_pipeline.sh --input /path/to/parler_folder [options]
#
# Stage numbers:
#   1  01_pass1_stats_corpus.py      (full stream)
#   2  02_select_candidates.py       (in-memory)
#   3  03_fetch_candidate_posts.py   (full stream)
#   4  04_classify_select.py         (in-memory)
#   5  05_pass2_histories.py         (full stream — requires final_users.txt)
#
# Examples:
#   ./run_parler_pipeline.sh --input "/mnt/d/Parler" --limit 40000        # smoke test, stages 1-4
#   ./run_parler_pipeline.sh --input "/mnt/d/Parler"                      # full run, stages 1-4 (stops before manual review)
#   ./run_parler_pipeline.sh --input "/mnt/d/Parler" --from 5 --to 5      # just the final histories pass
#   ./run_parler_pipeline.sh --input "/mnt/d/Parler" --stage 3            # rerun just step 3
#
set -euo pipefail

# ---------- defaults ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT=""
OUTDIR="out"
LEXICON="lexicon.txt"
CANDIDATES="candidates.tsv"
CANDIDATE_POSTS="candidate_posts.ndjson.gz"
SHORTLIST="shortlist.tsv"
FINAL_USERS="final_users.txt"
HISTORIES_DIR="histories"

CORPUS_CAP=300000
MIN_TEXT_POSTS=50
MIN_HITS=3
MAX_CANDIDATES=5000
SAMPLE=25
PER_TOPIC=60
LIMIT=0

FROM=1
TO=4          # deliberately stops before stage 5 by default — manual review sits between 4 and 5
VENV_DIR=".venv"

# ---------- arg parsing ----------
usage() { grep '^#' "$0" | sed -e 's/^#//' -e 's/^ //'; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input) INPUT="$2"; shift 2 ;;
    --outdir) OUTDIR="$2"; shift 2 ;;
    --lexicon) LEXICON="$2"; shift 2 ;;
    --corpus-cap) CORPUS_CAP="$2"; shift 2 ;;
    --min-text-posts) MIN_TEXT_POSTS="$2"; shift 2 ;;
    --min-hits) MIN_HITS="$2"; shift 2 ;;
    --max-candidates) MAX_CANDIDATES="$2"; shift 2 ;;
    --sample) SAMPLE="$2"; shift 2 ;;
    --per-topic) PER_TOPIC="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --from) FROM="$2"; shift 2 ;;
    --to) TO="$2"; shift 2 ;;
    --stage) FROM="$2"; TO="$2"; shift 2 ;;
    --venv) VENV_DIR="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "Unknown option: $1" >&2; usage ;;
  esac
done

if [[ -z "$INPUT" && "$FROM" -le 3 ]]; then
  # stages 1 and 3 (and 5) need --input; 2 and 4 don't
  if [[ "$FROM" -eq 1 || "$TO" -ge 1 && "$FROM" -le 1 ]] || \
     [[ "$FROM" -le 3 && "$TO" -ge 3 ]]; then
    echo "[error] --input <folder> is required for stages 1 and 3" >&2
    exit 1
  fi
fi

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ---------- venv setup ----------
if [[ ! -d "$SCRIPT_DIR/$VENV_DIR" ]]; then
  log "Creating venv at $VENV_DIR (stdlib-only pipeline, but isolating anyway)"
  python3 -m venv "$SCRIPT_DIR/$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$SCRIPT_DIR/$VENV_DIR/bin/activate"
log "Using $(python3 --version) at $(command -v python3)"

cd "$SCRIPT_DIR"

run_stage() {
  local n="$1"; shift
  if (( n < FROM || n > TO )); then
    return 0
  fi
  log "=== Stage $n: $* ==="
  time "$@"
  log "=== Stage $n done ==="
}

# ---------- stage 5 manual-review guard ----------
if (( TO >= 5 )); then
  if [[ ! -f "$FINAL_USERS" ]]; then
    echo "[error] Stage 5 requires '$FINAL_USERS' (manually curated creator ids from $SHORTLIST)." >&2
    echo "        Run stages 1-4 first, review $SHORTLIST by hand, save kept ids to $FINAL_USERS," >&2
    echo "        then rerun with --from 5 --to 5." >&2
    exit 1
  fi
fi

# ---------- pre-flight: validate lexicon before any stage that uses it (1, 4) ----------
if (( (FROM <= 1 && TO >= 1) || (FROM <= 4 && TO >= 4) )); then
  log "Validating $LEXICON before running..."
  if ! python3 validate_lexicon.py "$LEXICON"; then
    echo "[error] lexicon validation failed — fix $LEXICON before running the pipeline (see errors above)" >&2
    exit 1
  fi
fi

# ---------- stages ----------
LIMIT_ARGS=()
if [[ "$LIMIT" -gt 0 ]]; then
  LIMIT_ARGS=(--limit "$LIMIT")
fi

run_stage 1 python3 01_pass1_stats_corpus.py \
  --input "$INPUT" --outdir "$OUTDIR" --lexicon "$LEXICON" \
  --corpus-cap "$CORPUS_CAP" "${LIMIT_ARGS[@]}"

run_stage 2 python3 02_select_candidates.py \
  --stats "$OUTDIR/user_stats.tsv.gz" --out "$CANDIDATES" \
  --min-text-posts "$MIN_TEXT_POSTS" --min-hits "$MIN_HITS" \
  --max-candidates "$MAX_CANDIDATES"

run_stage 3 python3 03_fetch_candidate_posts.py \
  --input "$INPUT" --candidates "$CANDIDATES" \
  --out "$CANDIDATE_POSTS" --sample "$SAMPLE" "${LIMIT_ARGS[@]}"

run_stage 4 python3 04_classify_select.py \
  --posts "$CANDIDATE_POSTS" --lexicon "$LEXICON" --out "$SHORTLIST" --per-topic "$PER_TOPIC"

if (( TO == 4 && FROM <= 4 )); then
  log "Stopped after stage 4 by default. Manually review '$SHORTLIST', save kept ids to"
  log "'$FINAL_USERS' (one creator id per line), then run: $0 --input \"$INPUT\" --from 5 --to 5"
fi

run_stage 5 python3 05_pass2_histories.py \
  --input "$INPUT" --users "$FINAL_USERS" --outdir "$HISTORIES_DIR"

log "Pipeline range [$FROM,$TO] complete."
