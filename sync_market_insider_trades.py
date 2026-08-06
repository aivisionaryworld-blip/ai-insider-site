"""Sync a public, unfiltered feed of real-world open-market insider trades
from SEC EDGAR.

This is intentionally separate from public_signal_logger.py: that script
only records the signals AI Insider's own scoring model has already
ranked. This script pulls the raw, market-wide Form 4 activity stream
straight from SEC EDGAR -- every open-market buy and sell any insider at
any public company files, whether or not it ever clears AI Insider's bar.
It powers the "all real insider trades" banner, shown alongside (never
instead of) the scored signal feed so visitors can see the difference
between raw noise and a filtered signal.

SEC's fair-access policy requires a descriptive User-Agent identifying the
requester (name + contact) -- see https://www.sec.gov/os/webmaster-faq#developers.
Set SEC_USER_AGENT in the environment to override the default below.

Run standalone: `python sync_market_insider_trades.py`
"""
import csv
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV = os.path.join(BASE_DIR, "public_market_insider_trades.csv")

USER_AGENT = os.environ.get("SEC_USER_AGENT", "AIInsider/1.0 (hello@aiinsider.store)")
FEED_URL = (
    "https://www.sec.gov/cgi-bin/browse-edgar"
    "?action=getcurrent&type=4&company=&dateb=&owner=include&count=100&output=atom"
)
MAX_NEW_FILINGS_PER_RUN = 40
MAX_ROWS_KEPT = 600
REQUEST_TIMEOUT = 15
REQUEST_DELAY_SECONDS = 0.25  # well under SEC's 10 req/sec fair-access limit

# Only these transaction codes read as an actual "trade" to a visitor --
# grants, option exercises, tax withholding, and gifts are excluded so the
# banner stays meaningful instead of full of routine compensation noise.
TRANSACTION_LABELS = {"P": "Open-market buy", "S": "Open-market sell"}

COLUMNS = [
    "filed_date", "transaction_date", "ticker", "company",
    "insider_name", "insider_title", "transaction_code", "transaction_type",
    "shares", "price", "value", "acquired_disposed", "accession_no",
]


# SEC accession numbers are always 10-digit-filer + 2-digit-year +
# 6-digit-sequence = 18 digits with no dashes. Matching on that fixed-width
# digit run (rather than an exact URL shape) survives EDGAR link-format
# variations between the classic flat index and the newer nested directory
# style, since both embed "data/{cik}/{18 digits}" as a substring.
FILING_URL_RE = re.compile(r"/Archives/edgar/data/(\d+)/(\d{18})")


def fetch(url):
    req = Request(url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"})
    with urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        return resp.read()


def load_existing_rows():
    if not os.path.exists(OUTPUT_CSV):
        return [], set()
    with open(OUTPUT_CSV, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    seen = {row["accession_no"] for row in rows if row.get("accession_no")}
    return rows, seen


def parse_feed_entries(atom_bytes):
    """Every Form 4 filed market-wide in SEC's current-filings feed."""
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(atom_bytes)
    entries = []
    for entry in root.findall("a:entry", ns):
        link_el = entry.find("a:link", ns)
        href = link_el.get("href") if link_el is not None else None
        updated_el = entry.find("a:updated", ns)
        filed_date = (updated_el.text or "")[:10] if updated_el is not None and updated_el.text else ""
        if not href:
            continue
        m = FILING_URL_RE.search(href)
        if not m:
            continue
        cik, accession_no = m.groups()
        entries.append({
            "cik": cik,
            "accession_no": accession_no,
            "filed_date": filed_date,
        })
    return entries


def find_primary_xml_url(cik, accession_no):
    """The raw ownership XML document for a filing, via EDGAR's directory
    listing JSON -- more reliable than scraping the index HTML page."""
    index_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no}/index.json"
    payload = fetch(index_url)
    import json
    listing = json.loads(payload)
    items = listing.get("directory", {}).get("item", [])
    names = [item.get("name", "") for item in items]
    primary = next((n for n in names if n.lower() == "primary_doc.xml"), None)
    if not primary:
        primary = next(
            (n for n in names if n.lower().endswith(".xml") and not n.lower().startswith("xsl")),
            None,
        )
    if not primary:
        return None
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no}/{primary}"


def _text(node, path):
    el = node.find(path)
    return el.text.strip() if el is not None and el.text else None


def parse_form4_xml(xml_bytes):
    """Parse one Form 4 ownershipDocument into issuer/owner + transactions.

    Only the first reportingOwner is used (the common case for an
    open-market Form 4); joint-filer edge cases are a known simplification.
    Returns None if the document doesn't look like a Form 4 at all.
    """
    root = ET.fromstring(xml_bytes)
    if root.tag != "ownershipDocument":
        return None

    ticker = _text(root, ".//issuer/issuerTradingSymbol")
    company = _text(root, ".//issuer/issuerName")
    owner = root.find(".//reportingOwner")
    insider_name = _text(owner, "reportingOwnerId/rptOwnerName") if owner is not None else None
    title = _text(owner, "reportingOwnerRelationship/officerTitle") if owner is not None else None
    is_director = (_text(owner, "reportingOwnerRelationship/isDirector") == "1") if owner is not None else False
    if not title and is_director:
        title = "Director"

    transactions = []
    for txn in root.findall(".//nonDerivativeTable/nonDerivativeTransaction"):
        code = _text(txn, "transactionCoding/transactionCode")
        date = _text(txn, "transactionDate/value")
        shares = _text(txn, "transactionAmounts/transactionShares/value")
        price = _text(txn, "transactionAmounts/transactionPricePerShare/value")
        ad_code = _text(txn, "transactionAmounts/transactionAcquiredDisposedCode/value")
        if not code or not date:
            continue
        transactions.append({
            "transaction_date": date,
            "transaction_code": code,
            "shares": shares,
            "price": price,
            "acquired_disposed": ad_code,
        })

    return {
        "ticker": ticker,
        "company": company,
        "insider_name": insider_name,
        "insider_title": title,
        "transactions": transactions,
    }


def build_rows(filing, parsed):
    rows = []
    ticker = (parsed.get("ticker") or "").upper().strip()
    if not ticker or not re.fullmatch(r"[A-Z0-9.\-]{1,12}", ticker):
        return rows
    for txn in parsed["transactions"]:
        code = (txn.get("transaction_code") or "").upper().strip()
        label = TRANSACTION_LABELS.get(code)
        if not label:
            continue  # not a real open-market buy/sell -- skip grants, exercises, etc.
        try:
            shares = float(txn["shares"]) if txn.get("shares") else None
        except (TypeError, ValueError):
            shares = None
        try:
            price = float(txn["price"]) if txn.get("price") else None
        except (TypeError, ValueError):
            price = None
        value = round(shares * price, 2) if shares is not None and price is not None else None
        rows.append({
            "filed_date": filing["filed_date"],
            "transaction_date": txn.get("transaction_date") or "",
            "ticker": ticker,
            "company": (parsed.get("company") or "").strip()[:120],
            "insider_name": (parsed.get("insider_name") or "").strip()[:120],
            "insider_title": (parsed.get("insider_title") or "").strip()[:120],
            "transaction_code": code,
            "transaction_type": label,
            "shares": shares if shares is not None else "",
            "price": price if price is not None else "",
            "value": value if value is not None else "",
            "acquired_disposed": txn.get("acquired_disposed") or "",
            "accession_no": filing["accession_no"],
        })
    return rows


def main():
    existing_rows, seen_accessions = load_existing_rows()
    try:
        entries = parse_feed_entries(fetch(FEED_URL))
    except (HTTPError, URLError, ET.ParseError) as exc:
        print(f"Could not fetch/parse the EDGAR current-filings feed: {exc}", file=sys.stderr)
        return 0  # leave the existing CSV untouched; not a hard failure

    new_entries = [e for e in entries if e["accession_no"] not in seen_accessions]
    new_entries = new_entries[:MAX_NEW_FILINGS_PER_RUN]

    new_rows = []
    for filing in new_entries:
        time.sleep(REQUEST_DELAY_SECONDS)
        try:
            xml_url = find_primary_xml_url(filing["cik"], filing["accession_no"])
            if not xml_url:
                continue
            time.sleep(REQUEST_DELAY_SECONDS)
            parsed = parse_form4_xml(fetch(xml_url))
            if not parsed:
                continue
            new_rows.extend(build_rows(filing, parsed))
        except (HTTPError, URLError, ET.ParseError, KeyError, ValueError) as exc:
            print(f"Skipping filing {filing['accession_no']}: {exc}", file=sys.stderr)
            continue

    if not new_rows:
        print("No new open-market Form 4 trades found this run.")
        return 0

    combined = new_rows + existing_rows
    # de-duplicate on accession_no + transaction_date + shares, most recent first
    dedup, order = {}, []
    for row in combined:
        key = (row["accession_no"], row.get("transaction_date", ""), row.get("shares", ""))
        if key not in dedup:
            order.append(key)
        dedup[key] = row
    ordered = [dedup[k] for k in order]
    ordered.sort(key=lambda r: (r.get("transaction_date", ""), r.get("filed_date", "")), reverse=True)
    ordered = ordered[:MAX_ROWS_KEPT]

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in ordered:
            writer.writerow({c: row.get(c, "") for c in COLUMNS})

    print(f"Added {len(new_rows)} new trade row(s); {len(ordered)} total kept.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
