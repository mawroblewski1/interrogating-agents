"""
FETCH PASS. Streams the zip(s) once and pulls up to --sample text posts per CANDIDATE creator,
so the next stage can classify + topic-tag them without reading the whole dataset again.
Output is gzip-compressed.

Example:
  python 03_fetch_candidate_posts.py --input "D:\\Parler" --candidates candidates.tsv \
      --out candidate_posts.ndjson.gz --sample 25
"""
import argparse, gzip, json
from collections import defaultdict
import parler_io


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", default="candidate_posts.ndjson.gz")
    ap.add_argument("--sample", type=int, default=25, help="max posts kept per candidate")
    ap.add_argument("--min-body-chars", type=int, default=15)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: stop after N records scanned (0 = all)")
    a = ap.parse_args()

    want = set()
    with open(a.candidates, encoding="utf-8") as f:
        next(f, None)  # header
        for ln in f:
            want.add(ln.split("\t", 1)[0])
    print(f"[info] {len(want)} candidate creators")

    kept = defaultdict(int)
    paths = parler_io.find_inputs(a.input)
    n = 0
    with gzip.open(a.out, "wt", encoding="utf-8") as o:
        for rec in parler_io.iter_records(paths):
            n += 1
            cid = rec.get("creator")
            if cid in want and kept[cid] < a.sample:
                body = rec.get("body") or ""
                if len(body) >= a.min_body_chars:
                    o.write(json.dumps(
                        {"creator": cid, "createdAt": rec.get("createdAt", ""), "body": body},
                        ensure_ascii=False) + "\n")
                    kept[cid] += 1
            if a.limit and n >= a.limit:
                print(f"[info] stopped early at --limit {a.limit:,} records scanned")
                break
    print(f"[done] wrote samples for {sum(1 for v in kept.values() if v)} creators -> {a.out}")


if __name__ == "__main__":
    main()
