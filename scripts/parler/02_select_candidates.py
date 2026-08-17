"""
STAGE 2 (coarse filter). Reads user_stats.tsv.gz, keeps information-rich + likely-extremist
creators, ranks them, and writes a small candidate list.

Example:
  python 02_select_candidates.py --stats out/user_stats.tsv.gz --out candidates.tsv \
      --min-text-posts 50 --min-hits 3 --max-candidates 5000
"""
import argparse, gzip

# ---------- defaults (edit these to change built-in behavior; all overridable on the CLI) ----------
DEFAULT_OUT = "candidates.tsv"
DEFAULT_MIN_TEXT_POSTS = 50    # "information-rich" threshold
DEFAULT_MIN_HITS = 3           # min extremism keyword hits
DEFAULT_MAX_CANDIDATES = 5000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", required=True, help="path to user_stats.tsv.gz from Stage 1")
    ap.add_argument("--out", default=DEFAULT_OUT, help="candidate list to write")
    ap.add_argument("--min-text-posts", type=int, default=DEFAULT_MIN_TEXT_POSTS, help="information-rich threshold")
    ap.add_argument("--min-hits", type=int, default=DEFAULT_MIN_HITS, help="min extremism keyword hits")
    ap.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES,
                     help="hard cap on candidate list size after ranking")
    a = ap.parse_args()

    rows = []
    with gzip.open(a.stats, "rt", encoding="utf-8") as f:
        f.readline()  # header
        for ln in f:
            c, n_posts, n_text, total_chars, mn, mx, eh, fol = ln.rstrip("\n").split("\t")
            n_text, eh, fol, total_chars = int(n_text), int(eh), int(fol), int(total_chars)
            if n_text >= a.min_text_posts and eh >= a.min_hits:
                rows.append((c, n_text, eh, fol, total_chars))

    rows.sort(key=lambda r: (r[2], r[4]), reverse=True)   # ext_hits, then total_chars
    rows = rows[:a.max_candidates]

    with open(a.out, "w", encoding="utf-8") as o:
        o.write("creator\tn_text_posts\text_hits\tfollowers\ttotal_chars\n")
        for r in rows:
            o.write("\t".join(map(str, r)) + "\n")
    print(f"[done] candidates={len(rows)} -> {a.out}")


if __name__ == "__main__":
    main()
