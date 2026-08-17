"""
Swappable term-classifier backends. Every backend implements the same interface:

    classify_batch(terms: list[str]) -> list[dict]
    # each dict: {"term": ..., "label": ..., "confidence": 0.0-1.0, "reason": ...}

so classify_terms_llm.py doesn't need to know which backend is doing the work.

BACKENDS:
  anthropic  -- calls the Anthropic API (claude-sonnet-5 by default). Needs internet +
                ANTHROPIC_API_KEY. Best accuracy on nuanced/adjacent-category calls
                (e.g. globalist vs. antisemitic), gives a real free-text reason.
  local      -- zero-shot NLI classification (facebook/bart-large-mnli by default) via
                the HuggingFace `transformers` library. Runs fully offline after the
                one-time model download (~1.6GB), no per-call cost, no API key. Treats
                classification as textual entailment ("does this term relate to
                <label>?") rather than reasoning about it -- no free-text reason, and
                weaker on nuanced/adjacent-category distinctions than an LLM.
                Requires: pip install transformers torch --break-system-packages
  ollama     -- calls a locally-run model via Ollama's HTTP API (llama3.1 by default).
                Fully offline once the model is pulled, no API key, real free-text
                reasoning like the anthropic backend (same system prompt), but quality
                depends on which local model you run -- generally a real generative
                model reasoning about the term, so usually stronger than the local NLI
                backend, but still typically weaker than Claude on nuanced calls.
                Requires: Ollama installed + running (`ollama serve`), a model pulled
                (`ollama pull llama3.1`). Nothing leaves your machine.

Add a new backend by subclassing ClassifierBackend and registering it in BACKEND_REGISTRY.

NOT A CLI SCRIPT -- this is a library module, imported by classify_terms_llm.py, not run
directly (no argparse, no __main__ block). Its "arguments" are the constructor kwargs
each backend class takes -- see each class's DEFAULT_MODEL/DEFAULT_HOST and __init__ for
what's configurable. Typical usage from another script:

  import classifier_backends
  backend = classifier_backends.get_backend("anthropic", model=None)   # or "local"/"ollama"/"fixed"
  results = backend.classify_batch(["globalist", "table"])
  # -> [{"term": "globalist", "label": "globalist_elites", "confidence": 0.8, "reason": "..."}, ...]
"""
import json
import os
import sys
import urllib.error
import urllib.request

# Keep this in sync with lexicon.txt's actual sections.
LABEL_DEFINITIONS = {
    "anti_vaccine": "vaccine opposition, pharma distrust, health misinformation",
    "patriot_militia": "sovereign citizen, militia, anti-federal-authority, tax protestor content",
    "globalist_elites": "New World Order / Great Reset / hidden-global-control narratives",
    "antisemitic": "explicit antisemitic slurs, symbols, or tropes (be conservative -- only "
                    "truly unambiguous terms, not generically anti-elite language)",
    "manosphere": "misogynistic subculture content, looksmaxxing, incel-adjacent",
    "immigrant_minority_crime": "content framing immigrants/minorities as criminal threats",
    "anti_dei_racial_grievance": "anti-DEI, anti-affirmative-action, racial-grievance content",
    "anti_model_minority": 'racism specifically targeting "model minority" groups',
    "election_conspiracy": "stolen-election / voter-fraud narratives",
    "child_trafficking_conspiracy": "QAnon-style elite child-trafficking claims",
    "anti_lgbt": '"groomer" discourse, anti-LGBT content',
    "anti_abortion_radical": "violence-justifying anti-abortion content specifically (not "
                              "ordinary pro-life vocabulary)",
    "signal:trollish": "nihilist/meme/troll markers, ambiguous on their own",
    "signal:christian_nationalism": "Christian-nationalist framing/vocabulary",
    "signal:incitement": 'explicit calls to violence or "justified" violence framing',
    "not_extremist": "ordinary vocabulary, default for anything not clearly one of the above",
}
ALL_LABELS = list(LABEL_DEFINITIONS.keys())


def build_system_prompt() -> str:
    lines = [
        "You are classifying individual words/phrases for a research lexicon used to "
        "detect extremist content in social media posts. For EACH term given, decide "
        "which single label best fits, or \"not_extremist\" if the term is ordinary "
        "vocabulary with no reliable extremism signal on its own (this will be the "
        "correct answer for MOST terms -- do not force a classification).",
        "",
        "Valid labels (choose exactly one per term):",
    ]
    for label, desc in LABEL_DEFINITIONS.items():
        lines.append(f"- {label}: {desc}")
    lines.append("")
    lines.append(
        "Respond with ONLY a JSON array, no other text, no markdown fences. One object "
        "per input term, in the same order given:")
    lines.append('[{"term": "...", "label": "...", "confidence": 0.0-1.0, "reason": "one short phrase"}]')
    return "\n".join(lines)


class ClassifierBackend:
    def classify_batch(self, terms: list[str]) -> list[dict]:
        raise NotImplementedError


class AnthropicBackend(ClassifierBackend):
    API_URL = "https://api.anthropic.com/v1/messages"
    DEFAULT_MODEL = "claude-sonnet-5"

    def __init__(self, model: str = None, api_key: str = None):
        self.model = model or self.DEFAULT_MODEL      # Anthropic model string; None -> DEFAULT_MODEL
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")  # falls back to env var
        if not self.api_key:
            print("[error] ANTHROPIC_API_KEY environment variable is not set", file=sys.stderr)
            sys.exit(1)
        self.system_prompt = build_system_prompt()

    def classify_batch(self, terms: list[str]) -> list[dict]:
        user_msg = "Classify these terms:\n" + "\n".join(f"- {t}" for t in terms)
        body = json.dumps({
            "model": self.model,
            "max_tokens": 4000,
            "system": self.system_prompt,
            "messages": [{"role": "user", "content": user_msg}],
        }).encode("utf-8")

        req = urllib.request.Request(
            self.API_URL, data=body, method="POST",
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            })
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        text_blocks = [b["text"] for b in data.get("content", []) if b.get("type") == "text"]
        raw = "\n".join(text_blocks).strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"[warn] could not parse model output as JSON: {e}\nraw: {raw[:500]}", file=sys.stderr)
            return [{"term": t, "label": "not_extremist", "confidence": 0.0,
                      "reason": "PARSE_ERROR -- review manually"} for t in terms]


class LocalZeroShotBackend(ClassifierBackend):
    """Zero-shot NLI classification via HuggingFace transformers. Fully offline after
    the one-time model download. No free-text reasoning -- 'reason' is a fixed label
    naming the method, not an explanation of the classification."""
    DEFAULT_MODEL = "facebook/bart-large-mnli"

    def __init__(self, model: str = None):
        self.model_name = model or self.DEFAULT_MODEL  # any HuggingFace zero-shot-classification model
        self._pipeline = None  # lazy-loaded so `transformers` is only required if this backend is used

    def _ensure_loaded(self):
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline
        except ImportError:
            print("[error] the 'local' backend requires transformers + torch:\n"
                  "        pip install transformers torch --break-system-packages", file=sys.stderr)
            sys.exit(1)
        print(f"[info] loading local model {self.model_name!r} (first run downloads it, "
              f"then it's cached and fully offline)...", file=sys.stderr)
        self._pipeline = pipeline("zero-shot-classification", model=self.model_name)

    def classify_batch(self, terms: list[str]) -> list[dict]:
        self._ensure_loaded()
        out = []
        for term in terms:
            result = self._pipeline(
                term, candidate_labels=ALL_LABELS,
                hypothesis_template="This term is an example of: {}.")
            top_label = result["labels"][0]
            top_score = float(result["scores"][0])
            out.append({
                "term": term, "label": top_label, "confidence": round(top_score, 4),
                "reason": f"zero-shot NLI ({self.model_name})",
            })
        return out


class OllamaBackend(ClassifierBackend):
    """Calls a locally-run model via Ollama's HTTP API (http://localhost:11434 by default).
    Same generative-LLM approach as AnthropicBackend -- real free-text reasoning, same
    system prompt -- but fully offline and free once you've pulled a model. Quality
    depends entirely on which local model you run; a small local model will generally
    be weaker than Claude on the nuanced/adjacent-category calls (globalist vs.
    antisemitic, coded vs. overt), the same caveat as the zero-shot backend but usually
    less severe since it's still a generative model reasoning about the term, not just
    scoring entailment.

    Requires Ollama installed and running (`ollama serve`) with a model pulled, e.g.:
        ollama pull llama3.1
    This talks to localhost only -- no internet connection needed once the model is
    pulled, and nothing here ever leaves your machine."""
    DEFAULT_MODEL = "llama3.1"
    DEFAULT_HOST = "http://localhost:11434"

    def __init__(self, model: str = None, host: str = None):
        self.model = model or self.DEFAULT_MODEL              # a model name you've already `ollama pull`ed
        self.host = (host or self.DEFAULT_HOST).rstrip("/")    # base URL of the running `ollama serve`
        self.system_prompt = build_system_prompt()

    def classify_batch(self, terms: list[str]) -> list[dict]:
        user_msg = "Classify these terms:\n" + "\n".join(f"- {t}" for t in terms)
        body = json.dumps({
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_msg},
            ],
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.host}/api/chat", data=body, method="POST",
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            print(f"[error] couldn't reach Ollama at {self.host}: {e}\n"
                  f"        is it running? try: ollama serve   (and: ollama pull {self.model})",
                  file=sys.stderr)
            sys.exit(1)

        raw = data.get("message", {}).get("content", "").strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"[warn] could not parse model output as JSON: {e}\nraw: {raw[:500]}", file=sys.stderr)
            return [{"term": t, "label": "not_extremist", "confidence": 0.0,
                      "reason": "PARSE_ERROR -- review manually"} for t in terms]


class FixedLabelBackend(ClassifierBackend):
    """Assigns ONE label to every term, no model call at all. For when you already know
    the whole scraped list belongs to a single known bucket (e.g. every name under a
    'Category:Antisemitic conspiracy theorists' page is going in [antisemitic]) and per-
    term LLM/NLI classification would just be unnecessary uncertainty. Still produces a
    normal draft file, so manual review before merging into lexicon.txt still applies --
    this skips the classification step, not the review step."""

    def __init__(self, label: str = None, model: str = None):
        # label: REQUIRED -- the one bucket every term will be assigned to.
        # model: accepted but ignored; kept only so every backend shares one constructor
        #        shape and get_backend() can pass the same kwargs regardless of --backend.
        if not label:
            print("[error] the 'fixed' backend requires --label <bucket_name>", file=sys.stderr)
            sys.exit(1)
        if label not in ALL_LABELS:
            print(f"[warn] {label!r} isn't one of the labels in LABEL_DEFINITIONS "
                  f"({list(LABEL_DEFINITIONS)}) -- proceeding anyway in case it's a bucket "
                  f"you've added to lexicon.txt since this list was last updated", file=sys.stderr)
        self.label = label

    def classify_batch(self, terms: list[str]) -> list[dict]:
        return [{"term": t, "label": self.label, "confidence": 1.0,
                  "reason": "manually assigned bucket -- no classification performed"} for t in terms]


BACKEND_REGISTRY = {
    "anthropic": AnthropicBackend,
    "local": LocalZeroShotBackend,
    "ollama": OllamaBackend,
    "fixed": FixedLabelBackend,
}


def get_backend(name: str, **kwargs) -> ClassifierBackend:
    if name not in BACKEND_REGISTRY:
        print(f"[error] unknown backend {name!r}, choose from: {list(BACKEND_REGISTRY)}", file=sys.stderr)
        sys.exit(1)
    return BACKEND_REGISTRY[name](**kwargs)
