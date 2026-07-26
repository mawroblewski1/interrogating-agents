"""
PASS 2 (final). Streams the zip(s) once more and writes the FULL post history for each
final (manually reviewed) creator to histories/<creator>.ndjson. Small output.

Example:
  python 05_pass2_histories.py --input "D:\\Parler" --users final_users.txt --outdir histories
final_users.txt = one creator id per line (from your manual review of shortlist.tsv).
"""
import argparse, json, os
from collections import defaultdict
import parler_io


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--users", required=True, help="one creator id per line")
    ap.add_argument("--outdir", default="histories")
    ap.add_argument("--min-body-chars", type=int, default=15)
    a = ap.parse_args()

    want = set()
    for ln in open(a.users, encoding="utf-8"):
        cid = ln.strip().split("\t")[0]
        if cid and not cid.startswith("#"):
            want.add(cid)
    os.makedirs(a.outdir, exist_ok=True)
    files = {cid: open(os.path.join(a.outdir, f"{cid}.ndjson"), "w", encoding="utf-8") for cid in want}
    kept = defaultdict(int)

    for rec in parler_io.iter_records(parler_io.find_inputs(a.input)):
        cid = rec.get("creator")
        if cid in want:
            body = rec.get("body") or ""
            if len(body) >= a.min_body_chars:
                files[cid].write(json.dumps(
                    {"createdAt": rec.get("createdAt", ""), "body": body}, ensure_ascii=False) + "\n")
                kept[cid] += 1

    for fh in files.values():
        fh.close()
    got = sum(1 for v in kept.values() if v)
    print(f"[done] histories for {got}/{len(want)} users -> {a.outdir}/  "
          f"(total posts: {sum(kept.values()):,})")


if __name__ == "__main__":
    main()
