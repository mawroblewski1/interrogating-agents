"""
Shared parser for lexicon.txt's bracketed-section format. Used by:
  - 01_pass1_stats_corpus.py  (needs: flat set of every term)
  - 04_classify_select.py     (needs: topic buckets + signal buckets, kept separate, weighted)
  - validate_lexicon.py       (needs: structural + content warnings)

FORMAT
------
  # comment lines start with '#', ignored
  [topic_name]        -> a topic bucket; terms here compete for the Stage-4 topic label
  [signal:name]        -> a signal bucket; terms here are scored separately and never
                          compete for the topic label (e.g. trollish/meme markers that
                          aren't themselves evidence of a specific extremism topic)
  bare term lines      -> one term per line, no quoting; a multi-word phrase or an emoji
                          is just the literal line content
  term @weight          -> optional trailing '@<number>' pins this term's weight in THIS
                          bucket specifically, e.g. 'globalist @0.6'. Only meaningful for
                          a term that also appears in another bucket -- see WEIGHTING below.
  terms before any header (or in a header-less file) still count toward Stage 1's flat
  ext_hits set, but are excluded from every Stage-4 topic/signal bucket.

WEIGHTING
---------
A term that appears in only one bucket always has weight 1.0 there, whether or not you
annotate it. A term shared across multiple TOPIC buckets (e.g. 'globalist' in both
anti_government and antisemitic) has its weight split so a single occurrence contributes
a total of ~1.0 across all the buckets it's in, instead of counting fully in each
(which would double-count and inflate strength/topic scores). By default the split is
EVEN across however many buckets share the term. Use '@weight' to override the split for
specific buckets, e.g.:

    [anti_government]
    globalist @0.6

    [antisemitic]
    globalist @0.4

Any bucket left unannotated absorbs the remaining weight (1.0 minus the sum of explicit
weights for that term), split evenly among however many buckets were left unannotated.
See compute_term_weights() for the exact algorithm. Signal buckets are NOT weight-split
against topic buckets or against each other -- they're independent, always full weight.

Kept deliberately dumb otherwise: no regex term matching, no stemming, no punctuation
normalization. Substring matching (case-insensitive) happens in the callers.
"""
from __future__ import annotations
import re
from collections import OrderedDict
from dataclasses import dataclass, field

_HEADER_RE = re.compile(r"^\[([^\[\]]*)\]$")
_WEIGHT_RE = re.compile(r"^(.*\S)\s+@([0-9]*\.?[0-9]+)\s*$")

UNHEADED_KEY = None  # sentinel: terms that appeared before any [section]


@dataclass
class LexiconIssue:
    level: str      # "error" | "warn"
    line_no: int
    message: str


@dataclass
class ParsedLexicon:
    # section name -> list of (term, explicit_weight_or_None, line_no); UNHEADED_KEY holds pre-header terms
    sections: "OrderedDict[str | None, list[tuple[str, float | None, int]]]" = field(default_factory=OrderedDict)
    issues: list[LexiconIssue] = field(default_factory=list)


def _parse_term_line(stripped_line: str, line_no: int, issues: list[LexiconIssue]):
    """Split 'term @weight' -> (term, weight_or_None). Validates the weight if present."""
    m = _WEIGHT_RE.match(stripped_line)
    if not m:
        return stripped_line, None
    term, weight_str = m.group(1), m.group(2)
    try:
        weight = float(weight_str)
    except ValueError:
        issues.append(LexiconIssue("error", line_no, f"unparseable weight in {stripped_line!r}"))
        return term, None
    if weight <= 0:
        issues.append(LexiconIssue(
            "error", line_no, f"weight must be > 0, got {weight} for term {term!r}"))
        return term, None
    if weight > 1:
        issues.append(LexiconIssue(
            "warn", line_no,
            f"weight {weight} for term {term!r} is > 1.0 — unusual but allowed if intentional "
            f"(a term counting for more than a full point toward this bucket)"))
    return term, weight


def parse_lexicon_text(text: str) -> ParsedLexicon:
    result = ParsedLexicon()
    current = UNHEADED_KEY
    result.sections[current] = []
    seen_headers_lower: dict[str, int] = {}          # lowercased header -> first line_no
    # term.lower() -> [(section, line_no, term, explicit_weight)]
    seen_terms_lower: dict[str, list[tuple[str, int, str, float | None]]] = {}

    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("["):
            m = _HEADER_RE.match(line)
            if not m:
                result.issues.append(LexiconIssue(
                    "error", line_no,
                    f"malformed section header: {raw!r} (expected '[name]' with a single "
                    f"matching closing bracket and nothing else on the line)"))
                continue
            name = m.group(1).strip()
            if not name:
                result.issues.append(LexiconIssue("error", line_no, "empty section header '[]'"))
                continue
            if name.startswith("signal:") and name == "signal:":
                result.issues.append(LexiconIssue(
                    "error", line_no, "signal section has no name after the colon: '[signal:]'"))
                continue

            key_lower = name.lower()
            if key_lower in seen_headers_lower:
                result.issues.append(LexiconIssue(
                    "warn", line_no,
                    f"duplicate section '[{name}]' (first seen at line {seen_headers_lower[key_lower]}); "
                    f"terms will be merged into the same bucket"))
            else:
                seen_headers_lower[key_lower] = line_no

            current = name
            result.sections.setdefault(current, [])
            continue

        # a plain term line, possibly with a trailing '@weight'
        if raw != raw.strip():
            result.issues.append(LexiconIssue(
                "warn", line_no, f"line {raw!r} had leading/trailing whitespace (trimmed automatically)"))

        term, explicit_weight = _parse_term_line(line, line_no, result.issues)

        result.sections[current].append((term, explicit_weight, line_no))

        tl = term.lower()
        seen_terms_lower.setdefault(tl, []).append((current, line_no, term, explicit_weight))

    # duplicate-term detection (case-insensitive), across and within sections
    for tl, occurrences in seen_terms_lower.items():
        if len(occurrences) < 2:
            continue
        sections_involved = sorted({str(sec) for sec, _, _, _ in occurrences})
        if len(sections_involved) > 1:
            locs = ", ".join(f"'{sec}':L{ln}" for sec, ln, _, _ in occurrences)
            result.issues.append(LexiconIssue(
                "warn", occurrences[0][1],
                f"term '{tl}' appears in multiple sections ({locs}) — its weight will be "
                f"split across those buckets (see compute_term_weights)"))
        else:
            # same section, repeated (possibly different case) -> just redundant
            locs = ", ".join(f"L{ln}" for _, ln, _, _ in occurrences)
            result.issues.append(LexiconIssue(
                "warn", occurrences[0][1],
                f"term '{tl}' repeated within the same section ({locs}) — redundant, no effect"))

    return result


def load_parsed(path: str) -> ParsedLexicon:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return parse_lexicon_text(text)


def flatten_all_terms(parsed: ParsedLexicon) -> set[str]:
    """Every term from every section (and unheaded terms), lowercased. Used by Stage 1."""
    out = set()
    for terms in parsed.sections.values():
        for term, _weight, _line in terms:
            out.add(term.lower())
    return out


def split_topic_signal(parsed: ParsedLexicon):
    """Returns (topic_lexicons, signal_lexicons), each {bucket_name: [(term, explicit_weight_or_None), ...]},
    lowercased terms. Unheaded terms are excluded from both (they still count in Stage 1's flat set only)."""
    topics: "OrderedDict[str, list[tuple[str, float | None]]]" = OrderedDict()
    signals: "OrderedDict[str, list[tuple[str, float | None]]]" = OrderedDict()
    for section, terms in parsed.sections.items():
        if section is UNHEADED_KEY:
            continue
        term_list = [(t.lower(), w) for t, w, _line in terms]
        if section.startswith("signal:"):
            signal_name = section[len("signal:"):]
            signals.setdefault(signal_name, []).extend(term_list)
        else:
            topics.setdefault(section, []).extend(term_list)
    return topics, signals


def compute_term_weights(bucket_lexicons: dict) -> dict:
    """Given {bucket_name: [(term, explicit_weight_or_None), ...]}, return
    {term: {bucket_name: weight}}.

    For each term, across the buckets it appears in:
      - buckets with an explicit '@weight' use that value as-is
      - the remaining weight (1.0 - sum of explicit weights, floored at 0.0) is split
        evenly across whichever buckets for that term had NO explicit weight
      - a term in only one bucket with no explicit weight gets 1.0 there (unchanged
        from the simple even-split behavior)

    This prevents a shared term (e.g. 'globalist' in both anti_government and
    antisemitic) from counting fully toward every bucket it's in, while still letting
    you hand-tune the split (e.g. 60/40) per term.
    """
    # term -> {bucket_name: explicit_weight_or_None}
    term_to_buckets: dict[str, dict[str, float | None]] = {}
    for name, terms in bucket_lexicons.items():
        for t, w in terms:
            slot = term_to_buckets.setdefault(t, {})
            # last occurrence wins if a term is repeated within one bucket (already
            # flagged separately as a lint warning -- this is just a deterministic tiebreak)
            slot[name] = w

    weights: dict[str, dict[str, float]] = {}
    for term, bucket_weights in term_to_buckets.items():
        explicit = {b: w for b, w in bucket_weights.items() if w is not None}
        implicit_buckets = [b for b, w in bucket_weights.items() if w is None]
        explicit_sum = sum(explicit.values())
        remaining = max(0.0, 1.0 - explicit_sum)
        implicit_weight = (remaining / len(implicit_buckets)) if implicit_buckets else 0.0

        weights[term] = {}
        for b in bucket_weights:
            weights[term][b] = explicit[b] if b in explicit else implicit_weight

    return weights


def load_flat_terms(path: str) -> set[str]:
    """Convenience for Stage 1: parse + flatten in one call, ignoring lint issues."""
    return flatten_all_terms(load_parsed(path))


def load_topic_and_signal(path: str):
    """Convenience for Stage 4: parse + split in one call, ignoring lint issues."""
    return split_topic_signal(load_parsed(path))
