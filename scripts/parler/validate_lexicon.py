"""
Validate lexicon.txt's structure and contents before running the pipeline.

Checks:
  - file decodes as UTF-8
  - malformed section headers (unclosed/extra brackets, empty name, bad 'signal:' name)
  - duplicate section names
  - duplicate terms (within a section, or split across sections)
  - empty topic sections (no terms yet -> that topic can never be assigned)
  - stray leading/trailing whitespace on term lines
  - malformed or invalid '@weight' annotations (unparseable, <= 0, or > 1.0)
  - explicit weights for a shared term summing to more than 1.0

Exit code 0 = no errors (warnings are fine, informational only).
Exit code 1 = at least one error found; fix before running the pipeline.

Example:
  python3 validate_lexicon.py lexicon.txt
"""
import argparse
import sys
import lexicon_io

# ---------- defaults (edit these to change built-in behavior; overridable on the CLI) ----------
DEFAULT_LEXICON_PATH = "lexicon.txt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=DEFAULT_LEXICON_PATH, help="lexicon.txt file to validate")
    a = ap.parse_args()

    try:
        parsed = lexicon_io.load_parsed(a.path)
    except FileNotFoundError:
        print(f"[fail] file not found: {a.path}", file=sys.stderr)
        sys.exit(1)
    except UnicodeDecodeError as e:
        print(f"[fail] {a.path} is not valid UTF-8: {e}", file=sys.stderr)
        sys.exit(1)

    topics, signals = lexicon_io.split_topic_signal(parsed)
    all_terms = lexicon_io.flatten_all_terms(parsed)

    n_sections = sum(1 for k in parsed.sections if k is not lexicon_io.UNHEADED_KEY)
    print(f"[ok] {n_sections} section(s) parsed, {len(all_terms)} unique term(s) total "
          f"({len(topics)} topic bucket(s), {len(signals)} signal bucket(s))")

    extra_warns = 0
    for name, terms in topics.items():
        if not terms:
            print(f"[warn] topic section '[{name}]' is empty — no user can ever be "
                  f"classified into this topic")
            extra_warns += 1
    for name, terms in signals.items():
        if not terms:
            print(f"[warn] signal section '[signal:{name}]' is empty")
            extra_warns += 1

    unheaded = parsed.sections.get(lexicon_io.UNHEADED_KEY, [])
    if unheaded:
        print(f"[info] {len(unheaded)} term(s) appear before any [section] header — "
              f"these count toward Stage 1's ext_hits only, and are excluded from every "
              f"Stage 4 topic/signal bucket")

    topic_weights = lexicon_io.compute_term_weights(topics)
    shared_terms = sorted(t for t, buckets in topic_weights.items() if len(buckets) > 1)
    if shared_terms:
        print(f"[info] {len(shared_terms)} term(s) shared across topic buckets — Stage 4 "
              f"splits their weight instead of counting them fully in each bucket:")
        for t in shared_terms:
            bucket_weights = topic_weights[t]
            parts = ", ".join(f"'{b}'={w:.3f}" for b, w in sorted(bucket_weights.items()))
            print(f"       '{t}' -> {parts}")
            explicit_sum = sum(
                w for name, terms in topics.items() for term, w in terms
                if term == t and w is not None
            )
            if explicit_sum > 1.0001:
                extra_warns += 1
                print(f"       [warn] explicit weights for '{t}' sum to {explicit_sum:.3f} "
                      f"(> 1.0) — likely a typo unless intentionally overweighting this term")

    n_errors = sum(1 for i in parsed.issues if i.level == "error")
    n_warns = sum(1 for i in parsed.issues if i.level == "warn") + extra_warns

    for issue in sorted(parsed.issues, key=lambda i: i.line_no):
        tag = "error" if issue.level == "error" else "warn"
        print(f"[{tag}] line {issue.line_no}: {issue.message}")

    if n_errors:
        print(f"[fail] {n_errors} error(s), {n_warns} warning(s) — fix errors before running the pipeline",
              file=sys.stderr)
        sys.exit(1)

    print(f"[pass] 0 errors, {n_warns} warning(s)")
    sys.exit(0)


if __name__ == "__main__":
    main()
