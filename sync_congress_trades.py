"""Sync a public feed of congressional stock trading disclosures.

Source: the open-source House Stock Watcher / Senate Stock Watcher
datasets, which parse the official House Clerk and Senate eFD periodic
transaction reports (PTRs) into JSON. These are community-maintained
projects, not an official government API -- unlike sync_market_insider_
trades.py's SEC EDGAR source, the exact JSON field names here have
shifted over the years, so every field is read defensively through a
list of known/likely key names rather than a single hardcoded key, and
a malformed or reshaped record is skipped rather than crashing the run.

Run standalone: `python sync_congress_trades.py`
"""
import csv
import json
import os
import sys
import time
from urllib.request import Request, urlopen

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV = os.path.join(BASE_DIR, "public_congress_trades.csv")

USER_AGENT = os.environ.get("SEC_USER_AGENT", "AIInsider/1.0 (hello@aiinsider.store)")
HOUSE_URL = "https://house-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json"
SENATE_URL = "https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com/aggregate/all_transactions.json"
REQUEST_TIMEOUT = 30
MAX_ROWS_KEPT = 500

COLUMNS = [
    "disclosure_date", "transaction_date", "chamber", "member",
    "ticker", "asset_description", "transaction_type", "amount_range",
    "state_or_district",
]


def fetch_json(url):
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        return json.loads(resp.read())


def _first(d, keys, default=""):
    if not isinstance(d, dict):
        return default
    for k in keys:
        v = d.get(k)
        if v not in (None, ""):
            return v
    return default


def _normalize_type(raw):
    text = str(raw or "").strip().lower()
    if not text:
        return "Unknown"
    if "purchase" in text or text in ("buy", "p"):
        return "Purchase"
    if "sale" in text or "sell" in text:
        if "partial" in text:
            return "Sale (partial)"
        if "full" in text:
            return "Sale (full)"
        return "Sale"
    if "exchange" in text:
        return "Exchange"
    return str(raw).strip().title()[:30]


def _clean_ticker(raw):
    ticker = str(raw or "").upper().strip()
    if not ticker or ticker in ("--", "N/A", "NONE", "NA"):
        return ""
    return ticker[:12]


def house_rows(records):
    rows = []
    for r in records if isinstance(records, list) else []:
        ticker = _clean_ticker(_first(r, ["ticker"]))
        if not ticker:
            continue
        rows.append({
            "disclosure_date": _first(r, ["disclosure_date"]),
            "transaction_date": _first(r, ["transaction_date"]),
            "chamber": "House",
            "member": str(_first(r, ["representative", "member"]))[:120],
            "ticker": ticker,
            "asset_description": str(_first(r, ["asset_description", "asset"]))[:160],
            "transaction_type": _normalize_type(_first(r, ["type", "transaction_type"])),
            "amount_range": str(_first(r, ["amount"]))[:60],
            "state_or_district": str(_first(r, ["district", "state"]))[:20],
        })
    return rows


def senate_rows(records):
    """Senate Stock Watcher has historically sometimes nested individual
    line-items of one PTR filing under a "transactions" list rather than
    one row per transaction -- handle both shapes."""
    rows = []
    for r in records if isinstance(records, list) else []:
        if not isinstance(r, dict):
            continue
        nested = r.get("transactions")
        entries = nested if isinstance(nested, list) and nested else [r]
        senator = str(_first(r, ["senator", "member"]))[:120]
        disclosure_date = _first(r, ["disclosure_date", "report_date"])
        state = str(_first(r, ["state"]))[:20]
        for e in entries:
            ticker = _clean_ticker(_first(e, ["ticker", "symbol"]))
            if not ticker:
                continue
            rows.append({
                "disclosure_date": disclosure_date,
                "transaction_date": _first(e, ["transaction_date"]),
                "chamber": "Senate",
                "member": senator,
                "ticker": ticker,
                "asset_description": str(_first(e, ["asset_description", "asset_name", "asset"]))[:160],
                "transaction_type": _normalize_type(_first(e, ["type", "transaction_type"])),
                "amount_range": str(_first(e, ["amount"]))[:60],
                "state_or_district": state,
            })
    return rows


def main():
    all_rows = []
    for label, url, parser in (("House", HOUSE_URL, house_rows), ("Senate", SENATE_URL, senate_rows)):
        try:
            payload = fetch_json(url)
            records = payload if isinstance(payload, list) else payload.get("data", [])
            all_rows.extend(parser(records))
        except Exception as exc:  # noqa: BLE001 -- one dataset's schema drift must never take the other down
            print(f"Could not fetch/parse the {label} dataset: {exc}", file=sys.stderr)
        time.sleep(0.2)

    if not all_rows:
        print("No congress trades parsed this run; leaving existing CSV untouched.")
        return 0

    seen, deduped = set(), []
    for row in all_rows:
        key = (row["chamber"], row["member"], row["ticker"], row["transaction_date"], row["amount_range"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    deduped.sort(key=lambda r: (r.get("transaction_date") or "", r.get("disclosure_date") or ""), reverse=True)
    deduped = deduped[:MAX_ROWS_KEPT]

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in deduped:
            writer.writerow({c: row.get(c, "") for c in COLUMNS})

    print(f"Wrote {len(deduped)} congress trade row(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
