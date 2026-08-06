"""Sync a public market-news headline feed from a small set of financial
news RSS feeds.

Each feed is fetched and parsed independently -- if one feed's URL has
moved or its format changed, that feed is skipped (logged to stderr)
rather than failing the whole run, so a single stale RSS URL doesn't
take the section down. Add/remove feeds by editing FEEDS below; no code
changes needed elsewhere.

Run standalone: `python sync_market_news.py`
"""
import csv
import os
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from urllib.request import Request, urlopen

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV = os.path.join(BASE_DIR, "public_market_news.csv")

USER_AGENT = os.environ.get("SEC_USER_AGENT", "AIInsider/1.0 (hello@aiinsider.store)")
REQUEST_TIMEOUT = 15
REQUEST_DELAY_SECONDS = 0.2
MAX_ITEMS_PER_FEED = 20
MAX_ROWS_KEPT = 100

# (display name, RSS/Atom URL). Public feeds can move; a feed failing here
# just gets skipped for that run, so it's safe to add/remove entries freely.
FEEDS = [
    ("MarketWatch", "https://www.marketwatch.com/rss/topstories"),
    ("MarketWatch Markets", "https://www.marketwatch.com/rss/marketpulse"),
    ("CNBC Markets", "https://www.cnbc.com/id/20910258/device/rss/rss.html"),
    ("Investing.com", "https://www.investing.com/rss/news_25.rss"),
]

COLUMNS = ["published", "source", "title", "link"]

ATOM_NS = {"a": "http://www.w3.org/2005/Atom"}


def fetch(url):
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        return resp.read()


def parse_feed(xml_bytes, source_name):
    root = ET.fromstring(xml_bytes)
    items = []

    rss_items = root.findall(".//item")
    if rss_items:
        for it in rss_items[:MAX_ITEMS_PER_FEED]:
            title = (it.findtext("title") or "").strip()
            link = (it.findtext("link") or "").strip()
            published = (it.findtext("pubDate") or "").strip()
            if title and link:
                items.append({"published": published, "source": source_name, "title": title[:220], "link": link})
        return items

    for it in root.findall("a:entry", ATOM_NS)[:MAX_ITEMS_PER_FEED]:
        title_el = it.find("a:title", ATOM_NS)
        link_el = it.find("a:link", ATOM_NS)
        updated_el = it.find("a:updated", ATOM_NS)
        title = (title_el.text or "").strip() if title_el is not None and title_el.text else ""
        link = link_el.get("href") if link_el is not None else ""
        published = (updated_el.text or "").strip() if updated_el is not None and updated_el.text else ""
        if title and link:
            items.append({"published": published, "source": source_name, "title": title[:220], "link": link})
    return items


def _parse_date(raw):
    for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S %Z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def main():
    all_items = []
    for source_name, url in FEEDS:
        try:
            all_items.extend(parse_feed(fetch(url), source_name))
        except Exception as exc:  # noqa: BLE001 -- one bad/moved feed must never take the others down
            print(f"Skipping feed {source_name}: {exc}", file=sys.stderr)
        time.sleep(REQUEST_DELAY_SECONDS)

    if not all_items:
        print("No market news items fetched this run; leaving existing CSV untouched.")
        return 0

    seen, deduped = set(), []
    for item in all_items:
        key = item["link"]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    def sort_key(item):
        parsed = _parse_date(item.get("published", ""))
        return parsed.timestamp() if parsed else 0

    deduped.sort(key=sort_key, reverse=True)
    deduped = deduped[:MAX_ROWS_KEPT]

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in deduped:
            writer.writerow({c: row.get(c, "") for c in COLUMNS})

    print(f"Wrote {len(deduped)} market news item(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
