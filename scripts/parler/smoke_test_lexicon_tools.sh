#!/usr/bin/env bash
#
# smoke_test_lexicon_tools.sh -- exercises the lexicon-building/analysis tooling
# (mediawiki_scrape.py, classify_terms_llm.py, build_lexicon_from_review.py,
# 06_bucket_frequency.py) end to end with small/synthetic inputs, so you can confirm
# they work on YOUR machine before trusting them on real data. Not wired into
# run_parler_pipeline.sh -- run standalone:
#
#   chmod +x smoke_test_lexicon_tools.sh
#   ./smoke_test_lexicon_tools.sh
#
# Live-network steps (Wiktionary scrape, real LLM classification) are OPTIONAL and
# skipped automatically if you don't have internet / an API key set -- the rest of the
# chain still gets tested using a small hand-written stand-in file instead, so you still
# get real coverage of build_lexicon_from_review.py and 06_bucket_frequency.py either way.
#
set -uo pipefail  # no -e: we want to report failures per-step, not abort on the first one

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT
echo "[info] scratch dir: $WORKDIR"

PASS=0
FAIL=0
pass() { echo "[PASS] $1"; PASS=$((PASS+1)); }
fail() { echo "[FAIL] $1"; FAIL=$((FAIL+1)); }

# ---------------------------------------------------------------------------
echo
echo "=== 1/5: mediawiki_scrape.py ==="
cat > "$WORKDIR/urls.txt" << 'EOF'
https://en.wiktionary.org/wiki/Category:English_swear_words @0
EOF
if python3 mediawiki_scrape.py --urls "$WORKDIR/urls.txt" --max-terms 5 \
    --out "$WORKDIR/scraped_terms.ndjson" 2>"$WORKDIR/scrape.log"; then
  n=$(wc -l < "$WORKDIR/scraped_terms.ndjson" 2>/dev/null || echo 0)
  if [[ "$n" -gt 0 ]]; then
    pass "mediawiki_scrape.py: scraped $n term(s) from a live category"
  else
    fail "mediawiki_scrape.py ran but produced 0 terms (check $WORKDIR/scrape.log)"
  fi
else
  echo "[skip] no internet reachable, or Wiktionary structure changed -- see $WORKDIR/scrape.log"
  echo "       writing a synthetic stand-in so the rest of the chain can still be tested"
fi
# always ensure a usable terms file exists for step 2, live or synthetic
if [[ ! -s "$WORKDIR/scraped_terms.ndjson" ]]; then
  cat > "$WORKDIR/scraped_terms.ndjson" << 'EOF'
{"term": "globalist", "source": "synthetic", "kind": "test", "depth": 0}
{"term": "table", "source": "synthetic", "kind": "test", "depth": 0}
EOF
fi

# ---------------------------------------------------------------------------
echo
echo "=== 2/5: classify_terms_llm.py ==="
if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
  if python3 classify_terms_llm.py --terms "$WORKDIR/scraped_terms.ndjson" --backend anthropic \
      --out "$WORKDIR/draft.tsv" 2>"$WORKDIR/classify.log"; then
    n=$(($(wc -l < "$WORKDIR/draft.tsv") - 1))
    pass "classify_terms_llm.py (anthropic): classified $n term(s), see $WORKDIR/draft.tsv"
  else
    fail "classify_terms_llm.py (anthropic) failed -- see $WORKDIR/classify.log"
  fi
else
  echo "[skip] ANTHROPIC_API_KEY not set -- skipping live classification"
  echo "       writing a synthetic draft so the rest of the chain can still be tested"
  cat > "$WORKDIR/draft.tsv" << 'EOF'
term	label	confidence	reason
globalist	globalist_elites	0.80	synthetic test row
table	not_extremist	0.00	synthetic test row
EOF
fi

# ---------------------------------------------------------------------------
echo
echo "=== 3/5: build_lexicon_from_review.py ==="
cp lexicon.txt "$WORKDIR/lexicon_test.txt"
before_terms=$(grep -vc '^\s*#\|^\s*$\|^\[' "$WORKDIR/lexicon_test.txt")
if python3 build_lexicon_from_review.py --draft "$WORKDIR/draft.tsv" \
    --lexicon "$WORKDIR/lexicon_test.txt" 2>"$WORKDIR/merge.log"; then
  after_terms=$(grep -vc '^\s*#\|^\s*$\|^\[' "$WORKDIR/lexicon_test.txt")
  if [[ "$after_terms" -ge "$before_terms" ]]; then
    pass "build_lexicon_from_review.py: merged draft into a scratch copy of lexicon.txt " \
         "($before_terms -> $after_terms term lines); YOUR REAL lexicon.txt was NOT touched"
  else
    fail "build_lexicon_from_review.py ran but term count didn't increase -- see $WORKDIR/merge.log"
  fi
else
  fail "build_lexicon_from_review.py failed -- see $WORKDIR/merge.log"
fi

echo
echo "--- validating the merged scratch lexicon ---"
if python3 validate_lexicon.py "$WORKDIR/lexicon_test.txt" > "$WORKDIR/validate.log" 2>&1; then
  pass "validate_lexicon.py: merged scratch lexicon is well-formed"
else
  fail "validate_lexicon.py: merged scratch lexicon has errors -- see $WORKDIR/validate.log"
fi

# ---------------------------------------------------------------------------
echo
echo "=== 4/5: 06_bucket_frequency.py ==="
mkdir -p "$WORKDIR/testdata"
python3 - "$WORKDIR" << 'PYEOF'
import sys, zipfile, json
workdir = sys.argv[1]
recs = [
    json.dumps({"creator": "u1", "body": "the deep state and tyranny and globalist elites",
                "createdAt": "2021-01-01T00:00:00Z"}),
    json.dumps({"creator": "u2", "body": "rothschild family controls banks",
                "createdAt": "2021-01-01T00:00:00Z"}),
]
with zipfile.ZipFile(f"{workdir}/testdata/sample.zip", "w") as z:
    z.writestr("a.ndjson", "\n".join(recs) + "\n")
PYEOF
if python3 06_bucket_frequency.py --input "$WORKDIR/testdata" --lexicon lexicon.txt \
    --out "$WORKDIR/bucket_counts.tsv" 2>"$WORKDIR/freq.log"; then
  if [[ -s "$WORKDIR/bucket_counts.tsv" ]]; then
    pass "06_bucket_frequency.py: produced bucket_counts.tsv against synthetic data"
  else
    fail "06_bucket_frequency.py ran but output file is empty"
  fi
else
  fail "06_bucket_frequency.py failed -- see $WORKDIR/freq.log"
fi

echo
echo "--- optional: chart generation (needs matplotlib) ---"
if python3 06_bucket_frequency.py --input "$WORKDIR/testdata" --lexicon lexicon.txt \
    --out "$WORKDIR/bucket_counts2.tsv" --chart "$WORKDIR/leaderboard.png" \
    2>"$WORKDIR/chart.log"; then
  if [[ -s "$WORKDIR/leaderboard.png" ]]; then
    pass "chart generation: leaderboard.png created"
  else
    echo "[skip] matplotlib likely not installed -- see $WORKDIR/chart.log " \
         "(pip install matplotlib --break-system-packages)"
  fi
fi

# ---------------------------------------------------------------------------
echo
echo "=== 5/5: classifier_backends.py error handling (no live calls needed) ==="
if python3 -c "
import sys
sys.path.insert(0, '.')
import classifier_backends as cb
backend = cb.OllamaBackend(host='http://localhost:1')  # deliberately unreachable
try:
    backend.classify_batch(['test'])
    sys.exit(1)  # should NOT reach here
except SystemExit as e:
    sys.exit(0 if e.code == 1 else 2)
" 2>"$WORKDIR/ollama_err.log"; then
  pass "classifier_backends.py: OllamaBackend fails gracefully when unreachable"
else
  fail "classifier_backends.py: OllamaBackend error handling didn't behave as expected"
fi

# ---------------------------------------------------------------------------
echo
echo "=================================================="
echo "  $PASS passed, $FAIL failed"
echo "=================================================="
if [[ "$FAIL" -gt 0 ]]; then
  exit 1
fi
