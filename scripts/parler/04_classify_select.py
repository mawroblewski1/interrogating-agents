"""
STAGE 3-4. Scores each candidate against EVERY topic bucket (not just their winner) and
every signal bucket, then builds a shortlist that both caps each topic at --per-topic
(diversity -> no single bucket dominates) and attempts to guarantee at least
--min-per-topic candidates per bucket (coverage -> no bucket goes empty just because it
wasn't anyone's single strongest bucket).

Topic and signal keyword buckets live in lexicon.txt (see lexicon_io.py for the format)
instead of being hardcoded here, so the same lexicon drives both Stage 1's coarse prefilter
and Stage 4's topic classification. SIGNAL buckets (e.g. [signal:trollish]) are scored
separately and reported alongside the topic, but never compete for the topic label itself.

OUTPUT COLUMNS: creator, strength, topic (the winning/primary bucket), n_sample_posts,
then one topic_<bucket> column per topic bucket in lexicon.txt (that user's per-post-
normalized score in THAT bucket specifically -- these sum to `strength`), then one
signal_<name> column per signal bucket, then included_via (why this row is in the
shortlist: their own primary bucket, and/or which underfilled bucket(s) backfilled them
in). Exposing every bucket's score, not just the winner, is what makes overlap between
buckets visible per user instead of only seeing whichever bucket "won".

MINIMUM-GUARANTEE BACKFILL: after the normal per-topic-quota selection (candidates
grouped by their own top-scoring bucket, capped at --per-topic), any bucket that still
has fewer than --min-per-topic members gets backfilled -- candidates who scored nonzero
in that bucket, even though it wasn't their own top pick, ranked by their score in that
specific bucket and added until the minimum is reached. This is an ATTEMPT, not a
guarantee in the literal sense: it cannot invent candidates that don't exist in the
sampled data. If a bucket still falls short after backfill (e.g. it has few or no
matching lexicon terms, or genuinely no candidates ever mention it), that's printed as
a warning rather than silently failing -- worth treating as a signal that bucket's
lexicon terms may need attention, not necessarily a bug in this script.

The scorer below is a KEYWORD PLACEHOLDER. For the paper, replace score_user() with a real
hate-speech classifier or an LLM call (keep the (strength, topic, topic_hits, signal_hits)
return contract).

Example:
  python 04_classify_select.py --posts candidate_posts.ndjson.gz --lexicon lexicon.txt \
      --out shortlist.tsv --per-topic 60 --min-per-topic 5
Then MANUALLY REVIEW shortlist.tsv and save the kept creator ids (one per line) to final_users.txt.
"""
import argparse, gzip, json
from collections import defaultdict
import lexicon_io

# ---------- defaults (edit these to change built-in behavior; all overridable on the CLI) ----------
DEFAULT_LEXICON = "lexicon.txt"
DEFAULT_OUT = "shortlist.tsv"
DEFAULT_PER_TOPIC = 60     # max users kept per topic bucket (via their own top-scoring bucket)
DEFAULT_MIN_PER_TOPIC = 5  # attempt to backfill each bucket up to at least this many candidates


def score_user(posts, topic_lex, signal_lex, topic_weights):
    """Return (strength, topic, topic_hits, signal_hits). PLACEHOLDER heuristic.
    topic_hits is {bucket: per-post-normalized score for EVERY topic bucket}, not just
    the winner -- this is what lets the caller see a user's full cross-bucket profile,
    not only whichever bucket they scored highest in. strength = sum(topic_hits.values()),
    so it's the total across all buckets, same definition as before this file exposed
    per-bucket columns. signal_hits works the same as always, never affects the topic label.
    A term shared across multiple topic buckets splits its weight between them (topic_weights,
    from lexicon_io.compute_term_weights — even split by default, or per-bucket '@weight'
    overrides from lexicon.txt), so it doesn't count fully toward every bucket it appears in."""
    text = " ".join(p["body"] for p in posts).lower()
    n_posts = max(1, len(posts))
    raw_topic_hits = {
        t: sum(text.count(k) * topic_weights.get(k, {}).get(t, 1.0) for k, _w in kws)
        for t, kws in topic_lex.items()
    }
    topic_hits = {t: v / n_posts for t, v in raw_topic_hits.items()}  # per-post normalized
    signal_hits = {s: sum(text.count(k) for k, _w in kws) for s, kws in signal_lex.items()}
    topic = max(raw_topic_hits, key=raw_topic_hits.get) if any(raw_topic_hits.values()) else "other"
    strength = sum(topic_hits.values())
    return strength, topic, topic_hits, signal_hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posts", required=True, help="candidate_posts.ndjson.gz from step 03")
    ap.add_argument("--lexicon", default=DEFAULT_LEXICON, help="bracketed-section lexicon (see lexicon_io.py)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="shortlist TSV to write")
    ap.add_argument("--per-topic", type=int, default=DEFAULT_PER_TOPIC,
                     help="max users kept per topic bucket, via their own top-scoring bucket")
    ap.add_argument("--min-per-topic", type=int, default=DEFAULT_MIN_PER_TOPIC,
                     help="attempt to backfill each bucket up to at least this many candidates "
                          "(0 disables backfill entirely, same behavior as before this option existed)")
    a = ap.parse_args()

    topic_lex, signal_lex = lexicon_io.load_topic_and_signal(a.lexicon)
    topic_weights = lexicon_io.compute_term_weights(topic_lex)
    topic_names = list(topic_lex.keys())        # stable column order, same order as lexicon.txt
    signal_names = sorted(signal_lex.keys())    # stable column order in the output

    byuser = defaultdict(list)
    with gzip.open(a.posts, "rt", encoding="utf-8") as f:
        for ln in f:
            r = json.loads(ln)
            byuser[r["creator"]].append(r)

    scored = []
    for cid, posts in byuser.items():
        strength, topic, topic_hits, signal_hits = score_user(posts, topic_lex, signal_lex, topic_weights)
        scored.append({
            "cid": cid, "strength": round(strength, 4), "topic": topic,
            "n_posts": len(posts), "topic_hits": topic_hits, "signal_hits": signal_hits,
        })

    # ---------- stage A: normal per-topic-quota selection, via each user's OWN top bucket ----------
    byprimary = defaultdict(list)
    for s in scored:
        byprimary[s["topic"]].append(s)
    for topic in byprimary:
        byprimary[topic].sort(key=lambda s: s["strength"], reverse=True)

    selected: dict[str, dict] = {}       # cid -> row
    included_via: dict[str, list] = defaultdict(list)   # cid -> ["primary:x", "backfill:y", ...]
    for topic, rows in byprimary.items():
        for s in rows[:a.per_topic]:
            selected[s["cid"]] = s
            included_via[s["cid"]].append(f"primary:{topic}")

    # ---------- stage B: backfill each topic bucket up to --min-per-topic ----------
    backfill_warnings = []
    if a.min_per_topic > 0:
        for topic in topic_names:
            current_count = sum(1 for s in selected.values() if s["topic"] == topic)
            if current_count >= a.min_per_topic:
                continue
            needed = a.min_per_topic - current_count
            pool = [s for s in scored
                    if s["topic"] != topic and s["topic_hits"].get(topic, 0) > 0]
            pool.sort(key=lambda s: s["topic_hits"].get(topic, 0), reverse=True)
            n_added = 0
            for s in pool:
                if n_added >= needed:
                    break
                selected[s["cid"]] = s   # no-op if already selected via another bucket
                included_via[s["cid"]].append(f"backfill:{topic}")
                n_added += 1
            total_after = current_count + n_added
            if total_after < a.min_per_topic:
                backfill_warnings.append(
                    f"[warn] bucket '{topic}': only {total_after}/{a.min_per_topic} candidates "
                    f"available even after backfill (few or no matching lexicon terms / candidates)")

    # ---------- write output ----------
    shortlist = sorted(selected.values(), key=lambda s: (s["topic"], -s["strength"]))

    with open(a.out, "w", encoding="utf-8") as o:
        header = (["creator", "strength", "topic", "n_sample_posts"]
                   + [f"topic_{n}" for n in topic_names]
                   + [f"signal_{n}" for n in signal_names]
                   + ["included_via"])
        o.write("\t".join(header) + "\n")
        for s in shortlist:
            topic_cols = [round(s["topic_hits"].get(n, 0.0), 4) for n in topic_names]
            signal_cols = [s["signal_hits"].get(n, 0) for n in signal_names]
            reason = ";".join(included_via[s["cid"]])
            row = [s["cid"], s["strength"], s["topic"], s["n_posts"]] + topic_cols + signal_cols + [reason]
            o.write("\t".join(map(str, row)) + "\n")

    n_primary_topics = len({s["topic"] for s in selected.values()})
    print(f"[done] shortlist={len(shortlist)} across {n_primary_topics} primary topics "
          f"({len(topic_names)} topic bucket(s) total) -> {a.out}")
    print(f"[info] topic columns included: {', '.join('topic_' + n for n in topic_names)}")
    if signal_names:
        print(f"[info] signal columns included: {', '.join('signal_' + n for n in signal_names)}")
    for w in backfill_warnings:
        print(w)
    print("[next] MANUAL REVIEW shortlist.tsv: keep genuine, ARGUMENTATIVE extremists "
          "(a coherent stance, not just slurs). Save kept creator ids to final_users.txt.")


if __name__ == "__main__":
    main()
