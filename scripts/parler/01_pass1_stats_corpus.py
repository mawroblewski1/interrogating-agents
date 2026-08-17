"""
PASS 1 (single stream over all zip(s)). Emits TWO outputs:
  A) user_stats.tsv.gz          -> per-creator lightweight stats (drives persona selection)
  B) extremist_corpus.ndjson.gz -> GZIP-COMPRESSED pool of extremist posts (optional fine-tuning reserve)

Both outputs are compressed. Point --input at the FOLDER that holds the zip (filename not needed).

Example:
  python 01_pass1_stats_corpus.py --input "D:\\Parler" --outdir out --lexicon lexicon.txt --corpus-cap 300000

Note: the stats dict holds ~4M creators in RAM (a few GB). On a memory-limited machine,
lower coverage with --limit for testing, or run on a box with >=16 GB.
"""
import argparse, gzip, json, os, sys, time
from collections import defaultdict
import parler_io
import lexicon_io

# ---------- defaults (edit these to change built-in behavior; all overridable on the CLI) ----------
DEFAULT_OUTDIR = "out"
DEFAULT_LEXICON = "lexicon.txt"
DEFAULT_CORPUS_CAP = 300000       # max posts written to the reserve corpus (Output B)
DEFAULT_MIN_BODY_CHARS = 15       # minimum post length to count as a "text post"
DEFAULT_LIMIT = 0                 # stop after N records scanned; 0 = no limit (process everything)


def load_lexicon(path):
    """Flat set of every term in lexicon.txt (all sections + any unheaded terms).
    See lexicon_io.py for the bracketed-section format."""
    if not path or not os.path.exists(path):
        return set()
    return lexicon_io.load_flat_terms(path)


def ext_hits(text_lower, terms):
    return sum(1 for t in terms if t in text_lower) if terms else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="folder containing the Parler zip(s), or a single file")
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR, help="where to write user_stats.tsv.gz and extremist_corpus.ndjson.gz")
    ap.add_argument("--lexicon", default=DEFAULT_LEXICON, help="one extremism term per line")
    ap.add_argument("--corpus-cap", type=int, default=DEFAULT_CORPUS_CAP, help="max posts written to corpus B")
    ap.add_argument("--min-body-chars", type=int, default=DEFAULT_MIN_BODY_CHARS,
                     help="minimum post length to count as a 'text post' in the stats")
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="smoke test: stop after N records (0 = all)")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    terms = load_lexicon(a.lexicon)
    if not terms:
        print("[warn] no lexicon loaded -> extremism hit-count is 0 and corpus B stays empty. "
              "Supply --lexicon with real terms.", file=sys.stderr)
    paths = parler_io.find_inputs(a.input)
    print(f"[info] inputs: {paths}")

    # creator -> [n_posts, n_text_posts, total_chars, min_ts, max_ts, ext_hits, followers_max]
    stats = defaultdict(lambda: [0, 0, 0, "", "", 0, 0])
    corpus_path = os.path.join(a.outdir, "extremist_corpus.ndjson.gz")
    corpus = gzip.open(corpus_path, "wt", encoding="utf-8")

    n = 0
    n_corpus = 0
    t0 = time.time()
    for rec in parler_io.iter_records(paths):
        n += 1
        cid = rec.get("creator")
        if cid:
            body = rec.get("body") or ""
            ts = rec.get("createdAt") or ""
            s = stats[cid]
            s[0] += 1
            has_text = len(body) >= a.min_body_chars
            if has_text:
                s[1] += 1
                s[2] += len(body)
            if ts:
                if not s[3] or ts < s[3]:
                    s[3] = ts
                if not s[4] or ts > s[4]:
                    s[4] = ts
            fol = rec.get("followers")
            if isinstance(fol, int) and fol > s[6]:
                s[6] = fol
            h = ext_hits(body.lower(), terms) if (body and terms) else 0
            if h:
                s[5] += h
                if has_text and n_corpus < a.corpus_cap:
                    corpus.write(json.dumps(
                        {"creator": cid, "createdAt": ts, "body": body}, ensure_ascii=False) + "\n")
                    n_corpus += 1
        if a.limit and n >= a.limit:
            break
        if n % 1_000_000 == 0:
            rate = n / max(1e-9, time.time() - t0)
            print(f"[info] {n:,} recs | {len(stats):,} users | corpus {n_corpus:,} | {rate:,.0f} rec/s")

    corpus.close()
    stats_path = os.path.join(a.outdir, "user_stats.tsv.gz")
    with gzip.open(stats_path, "wt", encoding="utf-8") as out:
        out.write("creator\tn_posts\tn_text_posts\ttotal_chars\tmin_ts\tmax_ts\text_hits\tfollowers\n")
        for cid, s in stats.items():
            out.write(f"{cid}\t{s[0]}\t{s[1]}\t{s[2]}\t{s[3]}\t{s[4]}\t{s[5]}\t{s[6]}\n")

    print(f"[done] records={n:,} users={len(stats):,} corpus_posts={n_corpus:,}")
    print(f"[out] {stats_path}")
    print(f"[out] {corpus_path}  (gzip-compressed)")


if __name__ == "__main__":
    main()
