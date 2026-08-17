"""
STAGE 3-4. Scores each candidate for extremism strength + a TOPIC label, then makes a
DIVERSE shortlist via per-topic quotas (so the persona set is topically varied -> robust interrogator).

Topic and signal keyword buckets now live in lexicon.txt (see lexicon_io.py for the format)
instead of being hardcoded here, so the same lexicon drives both Stage 1's coarse prefilter
and Stage 4's topic classification. SIGNAL buckets (e.g. [signal:trollish]) are scored
separately and reported alongside the topic, but never compete for the topic label itself.

The scorer below is a KEYWORD PLACEHOLDER. For the paper, replace score_user() with a real
hate-speech classifier or an LLM call (keep the (strength, topic, signal_hits) return contract).

Example:
  python 04_classify_select.py --posts candidate_posts.ndjson.gz --lexicon lexicon.txt \
      --out shortlist.tsv --per-topic 60
Then MANUALLY REVIEW shortlist.tsv and save the kept creator ids (one per line) to final_users.txt.
"""
import argparse, gzip, json
from collections import defaultdict
import lexicon_io

# ---------- defaults (edit these to change built-in behavior; all overridable on the CLI) ----------
DEFAULT_LEXICON = "lexicon.txt"
DEFAULT_OUT = "shortlist.tsv"
DEFAULT_PER_TOPIC = 60   # max users kept per topic bucket


def score_user(posts, topic_lex, signal_lex, topic_weights):
    """Return (extremism_strength: float, topic: str, signal_hits: dict). PLACEHOLDER heuristic.
    strength/topic are driven only by topic_lex buckets; signal_lex buckets never affect topic.
    A term shared across multiple topic buckets splits its weight between them (topic_weights,
    from lexicon_io.compute_term_weights — even split by default, or per-bucket '@weight'
    overrides from lexicon.txt), so it doesn't count fully toward every bucket it appears in."""
    text = " ".join(p["body"] for p in posts).lower()
    topic_hits = {
        t: sum(text.count(k) * topic_weights.get(k, {}).get(t, 1.0) for k, _w in kws)
        for t, kws in topic_lex.items()
    }
    signal_hits = {s: sum(text.count(k) for k, _w in kws) for s, kws in signal_lex.items()}
    topic = max(topic_hits, key=topic_hits.get) if any(topic_hits.values()) else "other"
    strength = sum(topic_hits.values()) / max(1, len(posts))
    return strength, topic, signal_hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posts", required=True, help="candidate_posts.ndjson.gz from step 03")
    ap.add_argument("--lexicon", default=DEFAULT_LEXICON, help="bracketed-section lexicon (see lexicon_io.py)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="shortlist TSV to write")
    ap.add_argument("--per-topic", type=int, default=DEFAULT_PER_TOPIC, help="max users kept per topic bucket")
    a = ap.parse_args()

    topic_lex, signal_lex = lexicon_io.load_topic_and_signal(a.lexicon)
    topic_weights = lexicon_io.compute_term_weights(topic_lex)
    signal_names = sorted(signal_lex.keys())   # stable column order in the output

    byuser = defaultdict(list)
    with gzip.open(a.posts, "rt", encoding="utf-8") as f:
        for ln in f:
            r = json.loads(ln)
            byuser[r["creator"]].append(r)

    scored = []
    for cid, posts in byuser.items():
        strength, topic, signal_hits = score_user(posts, topic_lex, signal_lex, topic_weights)
        signal_cols = [signal_hits.get(name, 0) for name in signal_names]
        scored.append((cid, round(strength, 4), topic, len(posts), signal_cols))

    bytopic = defaultdict(list)
    for row in scored:
        bytopic[row[2]].append(row)

    shortlist = []
    for topic, rows in bytopic.items():
        rows.sort(key=lambda r: r[1], reverse=True)     # strongest first within topic
        shortlist += rows[:a.per_topic]

    with open(a.out, "w", encoding="utf-8") as o:
        header = ["creator", "strength", "topic", "n_sample_posts"] + [f"signal_{n}" for n in signal_names]
        o.write("\t".join(header) + "\n")
        for cid, strength, topic, n_posts, signal_cols in shortlist:
            row = [cid, strength, topic, n_posts] + signal_cols
            o.write("\t".join(map(str, row)) + "\n")
    print(f"[done] shortlist={len(shortlist)} across {len(bytopic)} topics -> {a.out}")
    if signal_names:
        print(f"[info] signal columns included: {', '.join('signal_' + n for n in signal_names)}")
    print("[next] MANUAL REVIEW shortlist.tsv: keep genuine, ARGUMENTATIVE extremists "
          "(a coherent stance, not just slurs). Save kept creator ids to final_users.txt.")


if __name__ == "__main__":
    main()
