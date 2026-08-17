"""
Merge a manually-reviewed draft candidates file (from classify_terms_llm.py, after you've
edited it -- deleted rows you disagree with, corrected labels, etc.) into lexicon.txt.

Only rows with label != not_extremist are considered. Terms already present anywhere in
lexicon.txt (case-insensitive) are skipped, not duplicated. New terms are appended under
their matching [topic_name] or [signal:name] section; if that section doesn't exist yet
in lexicon.txt, it's created at the end of the file.

This NEVER classifies anything itself -- it only merges what a human has already reviewed
and approved in the draft file. Always inspect the diff before committing lexicon.txt.

Example:
  python3 build_lexicon_from_review.py --draft draft_lexicon_candidates.tsv --lexicon lexicon.txt
"""
import argparse
import sys
import lexicon_io

# ---------- defaults (edit these to change built-in behavior; overridable on the CLI) ----------
DEFAULT_LEXICON = "lexicon.txt"


def load_draft_rows(path: str):
    rows = []
    with open(path, encoding="utf-8") as f:
        header = f.readline()
        for line_no, ln in enumerate(f, start=2):
            ln = ln.rstrip("\n")
            if not ln.strip():
                continue
            parts = ln.split("\t")
            if len(parts) < 2:
                print(f"[warn] skipping malformed draft line {line_no}: {ln!r}", file=sys.stderr)
                continue
            term, label = parts[0], parts[1]
            rows.append((term, label))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", required=True, help="reviewed output of classify_terms_llm.py")
    ap.add_argument("--lexicon", default=DEFAULT_LEXICON, help="lexicon.txt to merge approved terms into")
    ap.add_argument("--dry-run", action="store_true", help="report what would change, write nothing")
    a = ap.parse_args()

    rows = load_draft_rows(a.draft)
    approved = [(t, l) for t, l in rows if l != "not_extremist"]
    print(f"[info] {len(rows)} row(s) in draft, {len(approved)} approved (label != not_extremist)")

    existing = lexicon_io.load_parsed(a.lexicon)
    existing_terms_lower = lexicon_io.flatten_all_terms(existing)

    # bucket_name -> [term, ...] to append; bucket_name is the RAW section header text
    # (e.g. "anti_vaccine" or "signal:trollish")
    to_append: dict[str, list[str]] = {}
    n_skipped_dupe = 0
    for term, label in approved:
        if term.lower() in existing_terms_lower:
            n_skipped_dupe += 1
            continue
        section = label  # label already matches the section-header naming convention
        to_append.setdefault(section, []).append(term)

    if n_skipped_dupe:
        print(f"[info] {n_skipped_dupe} term(s) already present in {a.lexicon}, skipped")

    if not to_append:
        print("[done] nothing new to add")
        return

    for section, terms in to_append.items():
        print(f"[info] +{len(terms)} term(s) -> [{section}]")

    if a.dry_run:
        print("[dry-run] no changes written")
        return

    # figure out which sections already exist in the file, in their original case
    existing_section_names = {
        (s.lower() if s is not None else None): s
        for s in existing.sections.keys()
    }

    with open(a.lexicon, encoding="utf-8") as f:
        original_lines = f.read().splitlines()

    lines = list(original_lines)
    for section, terms in to_append.items():
        real_name = existing_section_names.get(section.lower())
        if real_name is not None:
            # insert right after the last term of that existing section, before the next header
            header_idx = None
            for i, line in enumerate(lines):
                if line.strip() == f"[{real_name}]":
                    header_idx = i
                    break
            insert_at = len(lines)
            if header_idx is not None:
                insert_at = header_idx + 1
                for i in range(header_idx + 1, len(lines)):
                    if lines[i].strip().startswith("[") and lines[i].strip().endswith("]"):
                        insert_at = i
                        break
                    insert_at = i + 1
            for t in terms:
                lines.insert(insert_at, t)
                insert_at += 1
        else:
            # brand-new section, appended at the end of the file
            if lines and lines[-1].strip():
                lines.append("")
            lines.append(f"[{section}]")
            lines.extend(terms)

    with open(a.lexicon, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[done] {a.lexicon} updated. Run validate_lexicon.py before using it.")


if __name__ == "__main__":
    main()
