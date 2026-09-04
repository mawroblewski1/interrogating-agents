"""
STAGE 6 (separate, optional). A standalone streaming pass over the zip(s) that tallies
WEIGHTED keyword hits per topic bucket, for a "leaderboard" bar chart of which extremism
topics are most represented in the raw dataset (or a --limit subset of it).

This is deliberately a separate pass from Stage 1 -- it answers a different question
(how much of each topic exists in the data overall) than Stage 1's per-user stats do,
and keeping it separate means you can re-run it against a subset (--limit) quickly
without re-running the whole Stage 1/2/3/4 candidate-selection funnel.

Uses the same lexicon_io weighting as Stage 4 (compute_term_weights), so a term shared
across topic buckets is split the same way here as it is during classification --
the leaderboard numbers stay consistent with how Stage 4 actually scores users.

NORMALIZATION: raw weighted-hit totals favor buckets with more lexicon terms, regardless
of real prevalence in the data. Two different corrections are tracked, answering two
different questions:

  - n_terms_effective / hits_per_effective_term: corrects for @weight splitting --
    a term worth 1.0 in a bucket (unshared, or no split) counts fully; a term split
    across multiple buckets contributes only its fractional share. This is a property
    of the LEXICON's structure, independent of the data.

  - n_terms_found / hits_per_found_term: corrects for terms that never actually
    appeared in the scanned data at all. A property of the DATA, not the lexicon.

hits_per_raw_term (dividing by plain unique-term count, ignoring both corrections) is
kept too for reference.

The console output also prints, per bucket, how many of its terms were ever found:
  anti_vaccine: 5/7 terms present in scanned data

PER-TERM COUNTS (--term-counts-json): optionally writes a JSON file structured just
like lexicon.txt itself -- same bracketed sections, same terms -- except every term now
carries its RAW (unweighted) appearance count from the scanned data instead of an
optional @weight. A term listed under two buckets in lexicon.txt appears under both
buckets here too, with the same count in each (its raw occurrence count doesn't depend
on which bucket you're looking at it from; only its WEIGHTED contribution to a bucket's
score does -- see hits_per_effective_term above for that). Terms that never matched
anything are still included, with count 0, so the file is a complete mirror of the
lexicon rather than only the terms that happened to hit:

  {
    "topics": {
      "anti_vaccine": {"vaccine": 42, "vax": 10, "jab": 0, ...},
      "anti_government": {"deep state": 5, ...}
    },
    "signals": {
      "trollish": {"clown world": 3, ...}
    }
  }

OUTPUT FORMAT: plain TSV, directly readable in both R and Python with no extra tooling
required -- e.g. read.delim("bucket_counts.tsv") in base R, or
pd.read_csv("bucket_counts.tsv", sep="\t") in pandas -- so the same file can be graphed
in RStudio or in Python without needing any conversion step. The JSON output is likewise
just standard json.load()/jsonlite::fromJSON() on either side.

Signal buckets are tallied too but reported separately from topic buckets (bucket_type
column), since they don't compete for weight the way topic buckets do -- signal-bucket
effective term counts currently equal their raw term counts (no cross-signal weight
splitting is implemented, only topic-topic, matching Stage 4's scope).

Example:
  python3 06_bucket_frequency.py --input "D:\\Parler" --lexicon lexicon.txt \
      --out bucket_counts.tsv --chart bucket_leaderboard.png --limit 500000
  python3 06_bucket_frequency.py --input "D:\\Parler" --term-counts-json term_counts.json ...
"""
import argparse
import json
from collections import defaultdict
import parler_io
import lexicon_io

# ---------- defaults (edit these to change built-in behavior; overridable on the CLI) ----------
DEFAULT_LEXICON = "lexicon.txt"
DEFAULT_OUT = "bucket_counts.tsv"
DEFAULT_TERM_COUNTS_JSON = ""          # empty string = skip; per-term counts, lexicon.txt-shaped
DEFAULT_CHART = ""                     # empty string = skip chart generation entirely
DEFAULT_CHART_METRIC = "weighted_hits"   # one of: weighted_hits, hits_per_raw_term,
                                          # hits_per_effective_term, hits_per_found_term
DEFAULT_MIN_BODY_CHARS = 15
DEFAULT_LIMIT = 0                      # stop after N records scanned; 0 = no limit


def bucket_term_stats(bucket_lex: dict, term_weights: dict = None):
    """Returns {bucket: (n_terms_raw, n_terms_effective)}. term_weights (from
    lexicon_io.compute_term_weights) makes n_terms_effective respect @weight splits;
    pass None to make effective == raw (used for signal buckets, which aren't split)."""
    stats = {}
    for bucket, terms in bucket_lex.items():
        unique_terms = sorted({t for t, _w in terms})
        n_raw = len(unique_terms)
        if term_weights:
            n_eff = sum(term_weights.get(t, {}).get(bucket, 1.0) for t in unique_terms)
        else:
            n_eff = float(n_raw)
        stats[bucket] = (n_raw, n_eff)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="folder containing the Parler zip(s), or a single file")
    ap.add_argument("--lexicon", default=DEFAULT_LEXICON, help="bracketed-section lexicon (see lexicon_io.py)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="TSV to write bucket counts/stats to")
    ap.add_argument("--term-counts-json", default=DEFAULT_TERM_COUNTS_JSON,
                     help="if set, write per-term raw appearance counts here, shaped like lexicon.txt")
    ap.add_argument("--chart", default=DEFAULT_CHART, help="if set, write a leaderboard bar chart PNG here")
    ap.add_argument("--chart-metric", default=DEFAULT_CHART_METRIC,
                     choices=["weighted_hits", "hits_per_raw_term", "hits_per_effective_term",
                              "hits_per_found_term"],
                     help="which column drives the chart's bar ordering/values")
    ap.add_argument("--min-body-chars", type=int, default=DEFAULT_MIN_BODY_CHARS,
                     help="minimum post length to be scanned for keyword hits")
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="stop after N records scanned (0 = all)")
    a = ap.parse_args()

    topic_lex, signal_lex = lexicon_io.load_topic_and_signal(a.lexicon)
    topic_weights = lexicon_io.compute_term_weights(topic_lex)
    topic_stats = bucket_term_stats(topic_lex, topic_weights)
    signal_stats = bucket_term_stats(signal_lex, term_weights=None)

    topic_counts: dict[str, float] = defaultdict(float)     # bucket -> weighted hit total
    signal_counts: dict[str, float] = defaultdict(float)
    # bucket -> term -> RAW (unweighted) occurrence count, initialized to 0 for every term
    # so absent terms are still present in the output rather than silently missing
    topic_term_counts: dict[str, dict[str, int]] = {
        bucket: {t: 0 for t, _w in terms} for bucket, terms in topic_lex.items()
    }
    signal_term_counts: dict[str, dict[str, int]] = {
        bucket: {t: 0 for t, _w in terms} for bucket, terms in signal_lex.items()
    }
    n_posts_matched = 0

    paths = parler_io.find_inputs(a.input)
    print(f"[info] inputs: {paths}")

    n = 0
    for rec in parler_io.iter_records(paths):
        n += 1
        body = rec.get("body") or ""
        if len(body) >= a.min_body_chars:
            text = body.lower()
            matched_this_post = False
            for bucket, kws in topic_lex.items():
                for k, _w in kws:
                    c = text.count(k)
                    if c:
                        topic_counts[bucket] += c * topic_weights.get(k, {}).get(bucket, 1.0)
                        topic_term_counts[bucket][k] += c
                        matched_this_post = True
            for bucket, kws in signal_lex.items():
                for k, _w in kws:
                    c = text.count(k)
                    if c:
                        signal_counts[bucket] += c
                        signal_term_counts[bucket][k] += c
                        matched_this_post = True
            if matched_this_post:
                n_posts_matched += 1
        if a.limit and n >= a.limit:
            print(f"[info] stopped at --limit {a.limit:,} records scanned")
            break
        if n % 1_000_000 == 0:
            print(f"[info] {n:,} records scanned so far...")

    print(f"[info] {n:,} records scanned, {n_posts_matched:,} post(s) matched at least one term")

    print("[info] terms present in scanned data, per bucket:")
    for bucket, counts in topic_term_counts.items():
        n_raw = len(counts)
        n_found = sum(1 for c in counts.values() if c > 0)
        print(f"       {bucket}: {n_found}/{n_raw} terms present")
    for bucket, counts in signal_term_counts.items():
        n_raw = len(counts)
        n_found = sum(1 for c in counts.values() if c > 0)
        print(f"       signal:{bucket}: {n_found}/{n_raw} terms present")

    def build_rows(bucket_lex, counts, stats, term_counts):
        rows = []
        for bucket in bucket_lex:
            n_raw, n_eff = stats[bucket]
            n_found = sum(1 for c in term_counts.get(bucket, {}).values() if c > 0)
            hits = counts.get(bucket, 0.0)
            hits_per_raw = (hits / n_raw) if n_raw else 0.0
            hits_per_eff = (hits / n_eff) if n_eff else 0.0
            hits_per_found = (hits / n_found) if n_found else 0.0
            rows.append((bucket, n_raw, n_eff, n_found, hits, hits_per_raw, hits_per_eff, hits_per_found))
        rows.sort(key=lambda r: r[4], reverse=True)  # default sort: raw weighted_hits desc
        return rows

    topic_rows = build_rows(topic_lex, topic_counts, topic_stats, topic_term_counts)
    signal_rows = build_rows(signal_lex, signal_counts, signal_stats, signal_term_counts)

    with open(a.out, "w", encoding="utf-8") as f:
        f.write("bucket_type\tbucket\tn_terms_raw\tn_terms_effective\tn_terms_found\tweighted_hits\t"
                 "hits_per_raw_term\thits_per_effective_term\thits_per_found_term\n")
        for bucket, n_raw, n_eff, n_found, hits, hpr, hpe, hpf in topic_rows:
            f.write(f"topic\t{bucket}\t{n_raw}\t{n_eff:.3f}\t{n_found}\t{hits:.2f}\t"
                     f"{hpr:.3f}\t{hpe:.3f}\t{hpf:.3f}\n")
        for bucket, n_raw, n_eff, n_found, hits, hpr, hpe, hpf in signal_rows:
            f.write(f"signal\t{bucket}\t{n_raw}\t{n_eff:.3f}\t{n_found}\t{hits:.2f}\t"
                     f"{hpr:.3f}\t{hpe:.3f}\t{hpf:.3f}\n")

    print(f"[done] {a.out}  (readable directly in R via read.delim(), or pandas via "
          f"read_csv(sep='\\t'))")

    if a.term_counts_json:
        with open(a.term_counts_json, "w", encoding="utf-8") as f:
            json.dump({"topics": topic_term_counts, "signals": signal_term_counts}, f,
                      indent=2, ensure_ascii=False, sort_keys=True)
        print(f"[out] {a.term_counts_json}")

    if a.chart:
        _write_chart(topic_rows, a.chart, a.chart_metric)
        print(f"[out] {a.chart}")


_METRIC_COL = {"weighted_hits": 4, "hits_per_raw_term": 5, "hits_per_effective_term": 6,
               "hits_per_found_term": 7}
_METRIC_LABEL = {
    "weighted_hits": "Weighted keyword hits",
    "hits_per_raw_term": "Weighted hits per lexicon term (raw count)",
    "hits_per_effective_term": "Weighted hits per lexicon term (effective, @weight-adjusted)",
    "hits_per_found_term": "Weighted hits per term actually found in the data",
}


def _write_chart(topic_rows, out_path: str, metric: str):
    """Static sorted horizontal 'leaderboard' bar chart of topic buckets."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not installed -- skipping chart. "
              "Install with: pip install matplotlib --break-system-packages")
        return

    col = _METRIC_COL[metric]
    buckets = [r[0] for r in topic_rows]
    values = [r[col] for r in topic_rows]
    order = sorted(range(len(buckets)), key=lambda i: values[i])  # ascending for barh (top = highest)
    buckets_sorted = [buckets[i] for i in order]
    values_sorted = [values[i] for i in order]

    fig_height = max(3, 0.5 * len(buckets_sorted))
    fig, ax = plt.subplots(figsize=(9, fig_height))
    bars = ax.barh(buckets_sorted, values_sorted, color="#c0392b")
    ax.set_xlabel(_METRIC_LABEL[metric])
    ax.set_title("Topic bucket leaderboard")
    for bar, v in zip(bars, values_sorted):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f" {v:,.2f}",
                 va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
