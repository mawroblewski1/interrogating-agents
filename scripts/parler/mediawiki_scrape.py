"""
Scrape term/name lists from a set of Wiktionary or Wikipedia URLs you specify directly --
category pages (recursed into subcategories) and "List of X" articles (their outgoing
links, as a proxy for the list's entries) are both supported, on either site.

WHY URLS INSTEAD OF AUTO-DISCOVERY: every category and list-article has its own
addressable URL, so you don't need special include/exclude logic to scrape just one
subcategory of a larger tree -- list the exact URL of the node you want, at whatever
level of the hierarchy suits you, and it's scraped independently of its siblings. List
both a parent and a specific child if you want both; the child isn't excluded by the
parent or vice versa.

INPUT FILE FORMAT (--urls, one entry per line):
  https://en.wiktionary.org/wiki/Category:English_offensive_terms
  https://en.wiktionary.org/wiki/Category:English_anti-LGBTQ_slurs @1
  https://en.wikipedia.org/wiki/Category:American_white_nationalists
  https://en.wikipedia.org/wiki/List_of_white_nationalist_organizations
  # comment lines and blank lines are ignored

  Trailing '@N' overrides --max-depth for that one entry. Category URLs recurse into
  subcategories up to that depth. List-article URLs (non-Category: pages) are NOT
  recursed -- they yield their outgoing article links at depth 0 only, since a list
  article isn't a tree the way a category is.

CAUTION ON PEOPLE-NAME LISTS: a name matching in a post is much weaker evidence of
extremism than a coded term or symbol is -- the post could be discussing that person
critically, neutrally, or dismissively. Treat scraped names as a lower-confidence
signal (e.g. require co-occurrence with other lexicon terms, or a lower weight) rather
than a standalone strong hit, and apply extra scrutiny in manual review given the
reputational stakes of tagging a real name as an extremism marker.

Example:
  python3 mediawiki_scrape.py --urls sources.txt --out scraped_terms.ndjson --max-depth 2
"""
import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request

# ---------- defaults (edit these to change built-in behavior; overridable on the CLI) ----------
USER_AGENT = "parler-pipeline-research-scraper/1.0 (contact: set-your-contact-here)"
REQUEST_DELAY_SEC = 0.5     # politeness delay between API calls
DEFAULT_MAX_DEPTH = 2       # default subcategory recursion depth (category URLs only)
DEFAULT_MAX_TERMS = 0       # stop after N terms total; 0 = no limit
DEFAULT_OUT = "scraped_terms.ndjson"

_SITE_API = {
    "en.wiktionary.org": "https://en.wiktionary.org/w/api.php",
    "en.wikipedia.org": "https://en.wikipedia.org/w/api.php",
}


def _api_get(api_base: str, params: dict) -> dict:
    params = {**params, "format": "json"}
    url = api_base + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def parse_source_url(url: str):
    """Return (api_base, kind, title). kind is 'category' or 'article'."""
    parsed = urllib.parse.urlparse(url)
    host = parsed.netloc
    if host not in _SITE_API:
        raise ValueError(f"unsupported site {host!r} (only en.wiktionary.org / en.wikipedia.org)")
    api_base = _SITE_API[host]

    # path is like /wiki/Category:English_offensive_terms or /wiki/List_of_X
    m = re.match(r"^/wiki/(.+)$", parsed.path)
    if not m:
        raise ValueError(f"couldn't parse a page title out of {url!r}")
    raw_title = urllib.parse.unquote(m.group(1)).replace("_", " ")

    if raw_title.startswith("Category:"):
        return api_base, "category", raw_title[len("Category:"):]
    return api_base, "article", raw_title


def _category_members(api_base: str, category_title: str):
    cmcontinue = None
    while True:
        params = {
            "action": "query", "list": "categorymembers",
            "cmtitle": f"Category:{category_title}", "cmlimit": "500",
            "cmtype": "page|subcat",
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        data = _api_get(api_base, params)
        for m in data.get("query", {}).get("categorymembers", []):
            yield m["title"], m["ns"]
        cmcontinue = data.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            break
        time.sleep(REQUEST_DELAY_SEC)


def _article_links(api_base: str, article_title: str):
    """Outgoing links (namespace 0 = other articles) from a 'List of X'-style page,
    used as a proxy for that list's entries."""
    plcontinue = None
    while True:
        params = {
            "action": "query", "prop": "links", "titles": article_title,
            "pllimit": "500", "plnamespace": "0",
        }
        if plcontinue:
            params["plcontinue"] = plcontinue
        data = _api_get(api_base, params)
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            for link in page.get("links", []):
                yield link["title"]
        plcontinue = data.get("continue", {}).get("plcontinue")
        if not plcontinue:
            break
        time.sleep(REQUEST_DELAY_SEC)


def _scrape_category(api_base: str, root_title: str, max_depth: int, max_terms: int, n_so_far: int):
    seen_categories = set()
    queue = [(root_title, 0)]
    n = n_so_far
    while queue:
        category, depth = queue.pop(0)
        key = category.lower()
        if key in seen_categories:
            continue
        seen_categories.add(key)

        print(f"[info] scraping Category:{category} (depth {depth}) via {api_base}", file=sys.stderr)
        try:
            members = list(_category_members(api_base, category))
        except Exception as e:
            print(f"[warn] failed to fetch Category:{category}: {e}", file=sys.stderr)
            continue
        time.sleep(REQUEST_DELAY_SEC)

        for title, ns in members:
            if ns == 14:
                subcat_name = title.split(":", 1)[1] if ":" in title else title
                if depth < max_depth:
                    queue.append((subcat_name, depth + 1))
            elif ns == 0:
                yield {"term": title, "source": f"Category:{category}", "kind": "category_member", "depth": depth}
                n += 1
                if max_terms and n >= max_terms:
                    return


def _scrape_article(api_base: str, title: str, max_terms: int, n_so_far: int):
    print(f"[info] scraping article {title!r} (outgoing links) via {api_base}", file=sys.stderr)
    n = n_so_far
    try:
        links = _article_links(api_base, title)
    except Exception as e:
        print(f"[warn] failed to fetch article {title!r}: {e}", file=sys.stderr)
        return
    for link_title in links:
        yield {"term": link_title, "source": f"Article:{title}", "kind": "list_link", "depth": 0}
        n += 1
        if max_terms and n >= max_terms:
            return
    time.sleep(REQUEST_DELAY_SEC)


def load_url_entries(path: str):
    """Yield (url, per_entry_max_depth_or_None)."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r"^(\S+)\s+@(\d+)$", line)
            if m:
                yield m.group(1), int(m.group(2))
            else:
                yield line, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", required=True, help="file with one Wiktionary/Wikipedia URL per line")
    ap.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH,
                     help="default subcategory recursion depth (category URLs only; "
                          "override per-line with a trailing '@N')")
    ap.add_argument("--max-terms", type=int, default=DEFAULT_MAX_TERMS,
                     help="stop after N terms total across all URLs (0 = no limit)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="ndjson file to write scraped terms to")
    a = ap.parse_args()

    n = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for url, depth_override in load_url_entries(a.urls):
            if a.max_terms and n >= a.max_terms:
                print(f"[info] stopped at --max-terms {a.max_terms}", file=sys.stderr)
                break
            try:
                api_base, kind, title = parse_source_url(url)
            except ValueError as e:
                print(f"[warn] skipping {url!r}: {e}", file=sys.stderr)
                continue

            depth = depth_override if depth_override is not None else a.max_depth
            if kind == "category":
                gen = _scrape_category(api_base, title, depth, a.max_terms, n)
            else:
                gen = _scrape_article(api_base, title, a.max_terms, n)

            for rec in gen:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1

    print(f"[done] {n} term(s)/name(s) scraped -> {a.out}")
    print(f"[next] python3 classify_terms_llm.py --terms {a.out} --out draft_lexicon_candidates.tsv")


if __name__ == "__main__":
    main()
