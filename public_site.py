"""
public_site.py — PUBLIC-FACING business site. Separate from dashboard.py
(your private trading console). UX pattern modeled on modern insider-trading
SaaS sites: hero claim, resolved-trade proof cards, raw-data-vs-analyzed
comparison, pricing tiers (shown as "coming soon" for the ads-first phase),
FAQ. Copy and branding here are original — not copied from any competitor.

SAFETY BOUNDARY (do not remove): this file never imports portfolio
functions, v23_portfolio.csv, trade_log.csv position sizes, or cash/account
state. It reads ONLY signal + backtest CSVs, and strips personal fields
(allocation %, shares, $ amounts) before rendering. Visitors see the
SIGNAL and REASONING, never "put X% of your money into this" — that's the
line between publishing research and giving personalized advice.

Run locally to preview:
    pip install flask
    python public_site.py
Open http://localhost:5001

This does NOT deploy anywhere by running it. Real hosting (Render/Railway/
Fly.io) + a domain are separate manual steps. AdSense requires a live
public URL and your own Google signup — placeholders are marked below.

Disclaimer/Privacy/Terms text is DRAFT boilerplate, not legal advice —
have a securities attorney review before real launch or any paid tier.
"""

import hashlib
import json
import os
import re
import sys
from datetime import datetime
from functools import lru_cache

import numpy as np
import pandas as pd
from flask import Flask, render_template_string, abort, jsonify, request

from returns_engine import compute_all_returns, METRIC_EXPLANATIONS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ADSENSE_CLIENT_ID = os.environ.get("ADSENSE_CLIENT_ID", "").strip()
PUBLIC_CURRENT_SIGNALS = os.path.join(BASE_DIR, "public_signals_current.csv")
PUBLIC_SIGNAL_HISTORY = os.path.join(BASE_DIR, "public_signal_history.csv")
PUBLIC_SIGNAL_OUTCOMES = os.path.join(BASE_DIR, "public_signal_outcomes.csv")
PUBLIC_YTD_PERFORMANCE = os.path.join(BASE_DIR, "public_ytd_performance.json")
SYNCED_BOT_TRADE_LEDGER = os.path.join(BASE_DIR, "synced_bot_trade_ledger.json")
LEGACY_CURRENT_SIGNALS = os.path.join(BASE_DIR, "v23_t212_aggressive_alloc_confirmed_basket_executable.csv")


def current_signals_path():
    """Prefer the stripped public snapshot; retain the local legacy input only
    as a backwards-compatible fallback for the private development machine."""
    return PUBLIC_CURRENT_SIGNALS if os.path.exists(PUBLIC_CURRENT_SIGNALS) else LEGACY_CURRENT_SIGNALS


# ── data loading (public-safe fields only) ────────────────────────

def load_filings_table(limit=30):
    """Real recent filings for a dense, professional data-table view —
    same public-safe fields as everywhere else, richer since the logger
    now captures insider identity + market data."""
    path = os.path.join(BASE_DIR, "public_signal_history.csv")
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return []
    if df.empty:
        return []
    recent = df.sort_values("signal_date", ascending=False).head(limit)
    cols = ["ticker", "rating", "signal_date", "entry_price", "insider_name",
            "insider_title", "total_amount", "market_cap"]
    for c in cols:
        if c not in recent.columns:
            recent[c] = None
    return recent[cols].to_dict("records")


def load_weekly_signals(days=7, limit=100):
    """Signals actually filed within the last N days -- a real date-range
    filter, not just a row-count cap on load_filings_table(). Honest empty
    state if nothing qualifies this week; never backfilled with older rows
    to avoid looking busier than it is.
    """
    path = os.path.join(BASE_DIR, "public_signal_history.csv")
    if not os.path.exists(path):
        return [], None
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return [], None
    if df.empty:
        return [], None
    dates = pd.to_datetime(df["signal_date"], errors="coerce")
    cutoff = pd.Timestamp(datetime.now().date()) - pd.Timedelta(days=days - 1)
    mask = dates >= cutoff
    if not mask.any():
        return [], cutoff.date().isoformat()
    recent = df[mask].copy()
    recent["_sort"] = dates[mask]
    recent = recent.sort_values("_sort", ascending=False).head(limit)
    cols = ["ticker", "rating", "signal_date", "entry_price", "insider_name",
            "insider_title", "total_amount", "market_cap"]
    for c in cols:
        if c not in recent.columns:
            recent[c] = None
    return recent[cols].to_dict("records"), cutoff.date().isoformat()


def load_trust_stats():
    """Real aggregate numbers from actual history — never fabricated.
    Returns None (honest empty state) if there's not enough history yet."""
    path = os.path.join(BASE_DIR, "public_signal_history.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None
    if df.empty:
        return None
    return {
        "total_filings": len(df),
        "unique_tickers": df["ticker"].nunique(),
        "unique_insiders": df["insider_name"].nunique() if "insider_name" in df.columns else 0,
        "days_tracked": df["signal_date"].nunique(),
    }


def load_ticker_items(limit=15):
    """Real recent signals for the scrolling ticker strip — pulls from the
    permanent history log (public-safe fields only, same as everywhere
    else on the site), not fabricated."""
    path = os.path.join(BASE_DIR, "public_signal_history.csv")
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return []
    if df.empty:
        return []
    recent = df.sort_values("signal_date", ascending=False).head(limit)
    return recent[["ticker", "rating", "signal_date"]].to_dict("records")


def load_market_trades(limit=25):
    """Real, unfiltered, market-wide open-market Form 4 trades pulled
    straight from SEC EDGAR by sync_market_insider_trades.py.

    Distinct from load_public_signals()/load_ticker_items(): those only
    ever show trades that already passed AI Insider's own scoring model.
    This is the raw stream the model screens, for the "every real insider
    trade" banner. Honest empty state (an empty list) until the sync
    workflow has produced data — never fabricated.
    """
    path = os.path.join(BASE_DIR, "public_market_insider_trades.csv")
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return []
    if df.empty:
        return []
    recent = df.sort_values(["transaction_date", "filed_date"], ascending=False).head(limit)
    cols = ["transaction_date", "ticker", "company", "insider_name",
            "insider_title", "transaction_type", "shares", "price", "value"]
    for c in cols:
        if c not in recent.columns:
            recent[c] = None
    return recent[cols].to_dict("records")


def load_congress_trades(limit=200):
    """Real congressional stock-trading disclosures synced from the House
    Stock Watcher / Senate Stock Watcher datasets by
    sync_congress_trades.py. Honest empty state until the sync has
    produced data — never fabricated.
    """
    path = os.path.join(BASE_DIR, "public_congress_trades.csv")
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return []
    if df.empty:
        return []
    recent = df.sort_values(["transaction_date", "disclosure_date"], ascending=False).head(limit)
    cols = ["disclosure_date", "transaction_date", "chamber", "member", "ticker",
            "asset_description", "transaction_type", "amount_range", "state_or_district"]
    for c in cols:
        if c not in recent.columns:
            recent[c] = None
    return recent[cols].to_dict("records")


def load_market_news(limit=40):
    """Real market-news headlines synced from public RSS feeds by
    sync_market_news.py. Honest empty state until the sync has produced
    data — never fabricated.
    """
    path = os.path.join(BASE_DIR, "public_market_news.csv")
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return []
    if df.empty:
        return []
    recent = df.head(limit)  # already sorted newest-first by the sync script
    cols = ["published", "source", "title", "link"]
    for c in cols:
        if c not in recent.columns:
            recent[c] = None
    return recent[cols].to_dict("records")


def _newest_signal_date(path):
    """Newest signal_date in a public feed CSV, or None."""
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except (pd.errors.EmptyDataError, OSError):
        return None
    if df.empty:
        return None
    for column in ("signal_date", "trade_date", "date"):
        if column in df.columns:
            parsed = pd.to_datetime(df[column], errors="coerce").dropna()
            if not parsed.empty:
                return parsed.max().date()
    return None


def load_scan_status():
    """Honest status derived from the DATA, not the file's mtime.

    This previously used os.path.getmtime() on the feed. A git checkout — which
    is exactly what a Render deploy does — rewrites file mtimes, so the site
    reported the DEPLOY time as the scan time and therefore always looked
    fresh, no matter how old the underlying signals really were.

    We can only honestly know when the newest SIGNAL is dated; the data does
    not record when a scan ran. So that is all we claim. If the current feed is
    legitimately empty (a successful scan that found nothing qualifying), fall
    back to the permanent history so the page still shows a real date.
    """
    latest = _newest_signal_date(current_signals_path())
    if latest is None:
        latest = _newest_signal_date(PUBLIC_SIGNAL_HISTORY)
    if latest is None:
        return {"available": False}
    days_ago = (datetime.now().date() - latest).days
    return {
        "available": True,
        "last_signal_date": latest.strftime("%Y-%m-%d"),
        "days_ago": days_ago,
        "fresh": days_ago <= 3,
    }


def load_public_signals():
    path = current_signals_path()
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return []
    rows = []
    for _, r in df.iterrows():
        def clean_text(value):
            return "" if value is None or pd.isna(value) else str(value).strip()

        entry_price = r.get("confirmed_entry", r.get("current_price", ""))
        low_52w = r.get("low_52w")
        high_52w = r.get("high_52w")
        take_profit = r.get("take_profit")
        stop_loss = r.get("stop_loss")
        max_hold = r.get("max_hold")
        model_allocation_pct = r.get("model_allocation_pct")
        range_position_pct = None
        target_pct = None
        try:
            entry_num = float(entry_price)
            low_num = float(low_52w)
            high_num = float(high_52w)
            if np.isfinite(entry_num) and np.isfinite(low_num) and np.isfinite(high_num) and high_num > low_num:
                range_position_pct = round(max(0, min(100, (entry_num - low_num) / (high_num - low_num) * 100)), 1)
        except (TypeError, ValueError):
            pass
        try:
            target_num = float(take_profit)
            if np.isfinite(entry_num) and entry_num > 0 and np.isfinite(target_num):
                target_pct = round((target_num / entry_num - 1) * 100, 1)
        except (NameError, TypeError, ValueError):
            pass
        risk_reward = None
        try:
            stop_num = float(stop_loss)
            reward = target_num - entry_num
            risk = entry_num - stop_num
            if np.isfinite(risk) and np.isfinite(reward) and risk > 0 and reward > 0:
                risk_reward = round(reward / risk, 1)
        except (NameError, TypeError, ValueError):
            pass
        try:
            model_allocation_pct = float(model_allocation_pct)
            if not np.isfinite(model_allocation_pct):
                model_allocation_pct = None
            else:
                model_allocation_pct = round(max(0, min(100, model_allocation_pct)), 2)
        except (TypeError, ValueError):
            model_allocation_pct = None
        rows.append({
            "ticker": r.get("ticker", ""),
            "company": clean_text(r.get("company", "")),
            "rating": r.get("rating", ""),
            "conviction": clean_text(r.get("conviction", "")),
            "hold_mode": clean_text(r.get("hold_mode", "")),
            "signal_date": clean_text(r.get("signal_date", "")) or clean_text(r.get("trade_date", "")) or clean_text(r.get("transaction_date", "")) or clean_text(r.get("date", "")),
            "trade_date": r.get("trade_date", ""),
            "entry_price": entry_price,
            "take_profit": None if pd.isna(take_profit) else take_profit,
            "target_pct": target_pct,
            "risk_reward": risk_reward,
            "max_hold": None if pd.isna(max_hold) else max_hold,
            "stop_loss": None if pd.isna(stop_loss) else stop_loss,
            "model_allocation_pct": model_allocation_pct,
            "reason": clean_text(r.get("reason", ""))[:900],
            "red_flags": clean_text(r.get("red_flags", ""))[:700],
            "insider_name": r.get("insider_name", ""),
            "insider_title": r.get("insider_title", ""),
            "total_amount": r.get("total_amount", 0) or 0,
            # market data — fully public financial info, already fetched
            # during the scan, just carried through here
            "market_cap": r.get("market_cap"),
            "rsi14": r.get("rsi14"),
            "low_52w": low_52w,
            "high_52w": high_52w,
            "range_position_pct": range_position_pct,
            "ret_20d_pct": r.get("ret_20d_pct"),
            "avg_volume_20d": r.get("avg_volume_20d"),
            # NOT included: private cash allocation or share quantity.
        })
    return rows


def load_resolved_examples(limit=3):
    """Historical BACKTEST trades with a resolved outcome — honestly labeled
    as backtest, not live results, for the proof-cards section."""
    path = os.path.join(BASE_DIR, "bt_v24_executed_trades.csv")
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return []
    if df.empty or "pnl_%" not in df.columns:
        return []
    top = df.nlargest(limit, "pnl_%")
    out = []
    for _, r in top.iterrows():
        out.append({
            "ticker": r.get("ticker", ""),
            "rating": r.get("rating", ""),
            "pnl": round(float(r.get("pnl_%", 0)), 1),
            "entry_date": r.get("entry_date", ""),
            "days_held": r.get("days_held", ""),
        })
    return out


def load_backtest_summary():
    path = os.path.join(BASE_DIR, "bt_v24_summary.csv")
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path).iloc[0].to_dict()
    except Exception:
        return None


def load_real_returns():
    """PRIORITY 0 FIX: real XIRR + time-weighted return computed live from
    bt_v24_equity.csv — never a hardcoded 'annualized' figure. Returns None
    if the equity file isn't present, so the page can show an honest
    'not available yet' state instead of a wrong number."""
    path = os.path.join(BASE_DIR, "bt_v24_equity.csv")
    if not os.path.exists(path):
        return None
    try:
        return compute_all_returns(path)
    except Exception:
        return None


def load_performance_comparison():
    """Build a public, cash-flow-neutral strategy index alongside SPY.

    The strategy line removes deposits before chaining each day's return, so
    contributions cannot make the backtest appear to outperform. Both series
    are joined on the same trading dates and rebased in the browser for the
    selected 1D / 1W / 1M / 3M / YTD / Max window.
    """
    equity_path = os.path.join(BASE_DIR, "bt_v24_equity.csv")
    benchmark_path = os.path.join(BASE_DIR, "spy_benchmark.csv")
    if not os.path.exists(equity_path) or not os.path.exists(benchmark_path):
        return None
    try:
        return _load_performance_comparison_cached(
            os.path.getmtime(equity_path),
            os.path.getmtime(benchmark_path),
        )
    except Exception:
        return None


def _finite_number(value, digits=2):
    """Return a rounded public-safe number, or None for invalid input."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return round(number, digits)


def _public_allocation_pct(value):
    """Return a public model-allocation percentage without exposing dollars."""
    number = _finite_number(value)
    if number is None or number < 0 or number > 100:
        return None
    return number


def _normalized_public_date(value):
    """Normalize dates used by the public ledger to YYYY-MM-DD."""
    if value is None:
        return ""
    try:
        parsed = pd.to_datetime(value, errors="coerce")
    except (TypeError, ValueError):
        return ""
    if pd.isna(parsed):
        return ""
    return parsed.strftime("%Y-%m-%d")


def load_public_ytd_snapshot():
    """Load the owner-supplied YTD headline, curve and trade events.

    The JSON is deliberately a one-way privacy boundary: only dates, tickers,
    statuses and percentage returns are accepted. Account values, share counts,
    transaction IDs and dollar profit never enter the public page.
    """
    if not os.path.exists(PUBLIC_YTD_PERFORMANCE):
        return None
    try:
        with open(PUBLIC_YTD_PERFORMANCE, "r", encoding="utf-8") as source:
            raw = json.load(source)
    except (OSError, ValueError, TypeError):
        return None

    try:
        year = int(raw.get("year"))
    except (TypeError, ValueError):
        return None
    ytd_return = _finite_number(raw.get("ytd_return_pct"), 1)
    as_of = _normalized_public_date(raw.get("as_of"))
    if not as_of or ytd_return is None:
        return None

    events = []
    for raw_event in raw.get("trade_events", []):
        if not isinstance(raw_event, dict):
            continue
        date = _normalized_public_date(raw_event.get("date"))
        ticker = str(raw_event.get("ticker") or "").strip().upper()[:12]
        status = str(raw_event.get("status") or "").strip().upper()
        return_pct = _finite_number(raw_event.get("return_pct"))
        if not date or not ticker or status not in {"WIN", "LOSS"} or return_pct is None:
            continue
        events.append({
            "key": f"t212|{date}|{ticker}",
            "date": date,
            "ticker": ticker,
            "rating": "",
            "status": status,
            "return_pct": return_pct,
            "source": "TRADING_212_CLOSED",
            "source_label": "Trading 212 closed",
        })

    owner_curve_by_date = {}
    for raw_point in raw.get("owner_curve", []):
        if not isinstance(raw_point, dict):
            continue
        date = _normalized_public_date(raw_point.get("date"))
        return_pct = _finite_number(raw_point.get("return_pct"), 2)
        if not date or not date.startswith(f"{year}-") or return_pct is None:
            continue
        owner_curve_by_date[date] = {
            "date": date,
            "return_pct": return_pct,
        }
    owner_curve = [owner_curve_by_date[date] for date in sorted(owner_curve_by_date)]
    if owner_curve and owner_curve[-1]["date"] <= as_of:
        owner_curve_by_date[as_of] = {
            "date": as_of,
            "return_pct": ytd_return,
        }
        owner_curve = [owner_curve_by_date[date] for date in sorted(owner_curve_by_date)]

    return {
        "period": "YTD",
        "year": year,
        "as_of": as_of,
        "ytd_return_pct": ytd_return,
        "return_label": "Owner-reported Trading 212 YTD",
        "methodology_note": str(raw.get("methodology_note") or "").strip(),
        "curve_method": str(raw.get("curve_method") or "").strip(),
        "owner_curve": owner_curve,
        # First dated point of the reconstructed curve. `as_of` is where the
        # curve ENDS, so it must never be presented as where history begins.
        "curve_start": owner_curve[0]["date"] if owner_curve else None,
        "trade_events": events,
    }


def _load_public_signal_outcomes():
    """Return public signal outcomes keyed by signal date and ticker."""
    if not os.path.exists(PUBLIC_SIGNAL_OUTCOMES):
        return {}
    try:
        outcomes = pd.read_csv(PUBLIC_SIGNAL_OUTCOMES)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return {}
    result = {}
    for _, row in outcomes.iterrows():
        date = _normalized_public_date(row.get("signal_date"))
        ticker = str(row.get("ticker") or "").strip().upper()[:12]
        status = str(row.get("status") or "OPEN").strip().upper()
        if not date or not ticker or status not in {"OPEN", "WIN", "LOSS"}:
            continue
        result[(date, ticker)] = {
            "status": "OPEN SIGNAL" if status == "OPEN" else status,
            "return_pct": _finite_number(row.get("pnl_pct")) if status != "OPEN" else None,
            "resolved_date": _normalized_public_date(row.get("resolved_date")) if status != "OPEN" else "",
        }
    return result


def _load_synced_bot_trade_events(year):
    """Load the stable, public-only bot ledger synchronized by GitHub Actions."""
    if not os.path.exists(SYNCED_BOT_TRADE_LEDGER):
        return []
    try:
        with open(SYNCED_BOT_TRADE_LEDGER, "r", encoding="utf-8") as source:
            raw = json.load(source)
    except (OSError, ValueError, TypeError):
        return []
    result = []
    for raw_event in raw.get("events", []):
        if not isinstance(raw_event, dict):
            continue
        date = _normalized_public_date(raw_event.get("date"))
        ticker = str(raw_event.get("ticker") or "").strip().upper()[:12]
        rating = str(raw_event.get("rating") or "").strip().upper()[:4]
        status = str(raw_event.get("status") or "OPEN SIGNAL").strip().upper()
        if not date.startswith(f"{year}-") or not ticker or status not in {"OPEN SIGNAL", "WIN", "LOSS"}:
            continue
        result.append({
            "key": f"bot|{date}|{ticker}",
            "date": date,
            "ticker": ticker,
            "rating": rating,
            "status": status,
            "return_pct": _finite_number(raw_event.get("return_pct")) if status != "OPEN SIGNAL" else None,
            "model_allocation_pct": _public_allocation_pct(raw_event.get("model_allocation_pct")),
            "resolved_date": _normalized_public_date(raw_event.get("resolved_date")) if status != "OPEN SIGNAL" else "",
            "source": "AI_INSIDER_BOT",
            "source_label": "AI Insider bot",
        })
    return result


def load_ytd_trade_ledger(snapshot=None):
    """Combine percentage-only account closes with current public bot signals."""
    snapshot = snapshot or load_public_ytd_snapshot()
    if not snapshot:
        return None
    year = snapshot["year"]
    events = list(snapshot["trade_events"])
    outcomes = _load_public_signal_outcomes()
    signal_rows = []

    if os.path.exists(PUBLIC_SIGNAL_HISTORY):
        try:
            history = pd.read_csv(PUBLIC_SIGNAL_HISTORY)
            signal_rows.extend(history.to_dict("records"))
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
            pass
    signal_rows.extend(load_public_signals())

    bot_events = {}
    for row in signal_rows:
        date = _normalized_public_date(
            row.get("signal_date") or row.get("trade_date") or row.get("transaction_date") or row.get("date")
        )
        ticker = str(row.get("ticker") or "").strip().upper()[:12]
        if not date or not ticker or not date.startswith(f"{year}-"):
            continue
        key = f"bot|{date}|{ticker}"
        if key in bot_events:
            continue
        outcome = outcomes.get(
            (date, ticker),
            {"status": "OPEN SIGNAL", "return_pct": None, "resolved_date": ""},
        )
        rating = str(row.get("rating") or "").strip().upper()[:4]
        bot_events[key] = {
            "key": key,
            "date": date,
            "ticker": ticker,
            "rating": rating,
            "status": outcome["status"],
            "return_pct": outcome["return_pct"],
            "model_allocation_pct": _public_allocation_pct(row.get("model_allocation_pct")),
            "resolved_date": outcome.get("resolved_date", ""),
            "source": "AI_INSIDER_BOT",
            "source_label": "AI Insider bot",
        }

    # The synchronized file is newer than Render's bundled CSVs, so it wins
    # when both sources contain the same signal and an outcome has changed.
    for synced_event in _load_synced_bot_trade_events(year):
        local_event = bot_events.get(synced_event["key"], {})
        bot_events[synced_event["key"]] = {
            **local_event,
            **synced_event,
            "model_allocation_pct": (
                synced_event["model_allocation_pct"]
                if synced_event["model_allocation_pct"] is not None
                else local_event.get("model_allocation_pct")
            ),
            "resolved_date": synced_event["resolved_date"] or local_event.get("resolved_date", ""),
        }
    events.extend(bot_events.values())

    events.sort(key=lambda item: (item["date"], item["source"], item["ticker"]), reverse=True)
    closed = [event for event in events if event["status"] in {"WIN", "LOSS"}]
    wins = sum(event["status"] == "WIN" for event in closed)
    open_count = sum(event["status"] == "OPEN SIGNAL" for event in events)
    closed_returns = [event["return_pct"] for event in closed if event["return_pct"] is not None]
    latest_date = max(
        (event.get("resolved_date") or event["date"] for event in events),
        default=snapshot["as_of"],
    )
    return {
        "events": events,
        "activity_as_of": max(snapshot["as_of"], latest_date),
        "closed_count": len(closed),
        "wins": wins,
        "losses": len(closed) - wins,
        "open_count": open_count,
        "win_rate": round(wins / len(closed) * 100, 1) if closed else None,
        "average_closed_return": round(sum(closed_returns) / len(closed_returns), 2) if closed_returns else None,
    }


def build_owner_portfolio_continuation(snapshot, ledger):
    """Continue an owner-reported YTD baseline with resolved public bot signals.

    This is a percentage-only model continuation, not a brokerage balance feed.
    Open signals are chart markers and have no performance impact. A resolved
    signal changes the curve by its published allocation multiplied by its
    percentage return.
    """
    if not snapshot or not ledger:
        return []
    baseline_date = snapshot["as_of"]
    baseline_return = float(snapshot["ytd_return_pct"])
    baseline_growth = 1.0 + baseline_return / 100.0
    continuation_growth = 1.0
    points = []
    for historical_point in snapshot.get("owner_curve") or []:
        points.append({
            "date": historical_point["date"],
            "return_pct": historical_point["return_pct"],
            "ticker": "",
            "status": "BASELINE" if historical_point["date"] == baseline_date else "HISTORY",
            "allocation_pct": None,
            "signal_return_pct": None,
        })
    if not points or points[-1]["date"] != baseline_date:
        points.append({
            "date": baseline_date,
            "return_pct": round(baseline_return, 2),
            "ticker": "",
            "status": "BASELINE",
            "allocation_pct": None,
            "signal_return_pct": None,
        })

    bot_events = [
        event for event in ledger["events"]
        if event.get("source") == "AI_INSIDER_BOT" and event.get("date", "") >= baseline_date
    ]
    bot_events.sort(key=lambda event: (
        event.get("resolved_date") or event.get("date") or "",
        event.get("date") or "",
        event.get("ticker") or "",
    ))
    for event in bot_events:
        status = event.get("status")
        allocation = _public_allocation_pct(event.get("model_allocation_pct"))
        signal_return = _finite_number(event.get("return_pct"))
        if status in {"WIN", "LOSS"} and allocation is not None and signal_return is not None:
            continuation_growth *= max(0.0, 1.0 + (allocation / 100.0) * (signal_return / 100.0))
        continuation_return = (baseline_growth * continuation_growth - 1.0) * 100.0
        points.append({
            "date": event.get("resolved_date") or event["date"],
            "return_pct": round(continuation_return, 2),
            "ticker": event["ticker"],
            "status": status,
            "allocation_pct": allocation,
            "signal_return_pct": signal_return,
        })
    return points


@lru_cache(maxsize=4)
def _load_performance_comparison_cached(_equity_mtime, _benchmark_mtime):
    equity_path = os.path.join(BASE_DIR, "bt_v24_equity.csv")
    benchmark_path = os.path.join(BASE_DIR, "spy_benchmark.csv")

    equity = pd.read_csv(equity_path)
    benchmark = pd.read_csv(benchmark_path)
    required_equity = {"date", "equity_$", "deposited_$"}
    required_benchmark = {"date", "adjusted_close"}
    if not required_equity.issubset(equity.columns) or not required_benchmark.issubset(benchmark.columns):
        return None

    equity = equity[["date", "equity_$", "deposited_$"]].copy()
    equity["date"] = pd.to_datetime(equity["date"], errors="coerce").dt.normalize()
    equity["equity_$"] = pd.to_numeric(equity["equity_$"], errors="coerce")
    equity["deposited_$"] = pd.to_numeric(equity["deposited_$"], errors="coerce")
    equity = equity.dropna().sort_values("date").drop_duplicates("date", keep="last")

    benchmark = benchmark[["date", "adjusted_close"]].copy()
    benchmark["date"] = pd.to_datetime(benchmark["date"], errors="coerce").dt.normalize()
    benchmark["adjusted_close"] = pd.to_numeric(benchmark["adjusted_close"], errors="coerce")
    benchmark = benchmark.dropna().sort_values("date").drop_duplicates("date", keep="last")
    benchmark = benchmark[benchmark["adjusted_close"] > 0]
    if len(equity) < 2 or len(benchmark) < 2:
        return None

    benchmark_year = int(benchmark["date"].dt.year.max())
    benchmark_year_start = pd.Timestamp(year=benchmark_year, month=1, day=1)
    benchmark_prior = benchmark[benchmark["date"] < benchmark_year_start].tail(1)
    benchmark_ytd = benchmark[benchmark["date"] >= benchmark_year_start].copy()
    if benchmark_ytd.empty:
        return None
    benchmark_base_row = benchmark_prior.iloc[-1] if not benchmark_prior.empty else benchmark_ytd.iloc[0]
    benchmark_base = float(benchmark_base_row["adjusted_close"])
    benchmark_ytd_points = [[
        benchmark_base_row["date"].strftime("%Y-%m-%d"),
        0.0,
    ]]
    benchmark_ytd_points.extend([
        [
            row.date.strftime("%Y-%m-%d"),
            round((float(row.adjusted_close) / benchmark_base - 1.0) * 100.0, 4),
        ]
        for row in benchmark_ytd.itertuples(index=False)
    ])

    strategy_index = [100.0]
    for index in range(1, len(equity)):
        previous = equity.iloc[index - 1]
        current = equity.iloc[index]
        previous_equity = float(previous["equity_$"])
        deposit_change = float(current["deposited_$"] - previous["deposited_$"])
        if previous_equity <= 0:
            strategy_index.append(strategy_index[-1])
            continue
        daily_return = (float(current["equity_$"]) - deposit_change - previous_equity) / previous_equity
        growth_factor = max(0.000001, 1.0 + daily_return)
        strategy_index.append(strategy_index[-1] * growth_factor)

    strategy = pd.DataFrame({
        "date": equity["date"].to_numpy(),
        "strategy_index": strategy_index,
    })
    comparison = benchmark.merge(strategy, on="date", how="inner")
    comparison = comparison.sort_values("date")
    if len(comparison) < 2:
        return None

    points = [
        [
            row.date.strftime("%Y-%m-%d"),
            round(float(row.strategy_index), 6),
            round(float(row.adjusted_close), 6),
        ]
        for row in comparison.itertuples(index=False)
    ]
    summary = load_backtest_summary() or {}
    trade_count = summary.get("positions")
    win_rate = summary.get("win_rate_%")
    try:
        trade_count = int(float(trade_count))
    except (TypeError, ValueError):
        trade_count = None
    try:
        win_rate = round(float(win_rate), 1)
    except (TypeError, ValueError):
        win_rate = None

    return {
        "points": points,
        "benchmark_ytd_points": benchmark_ytd_points,
        "benchmark_ytd_as_of": benchmark_ytd_points[-1][0],
        "start_date": points[0][0],
        "as_of": points[-1][0],
        "trade_count": trade_count,
        "win_rate": win_rate,
        "benchmark_label": "S&P 500 (SPY proxy)",
    }


def load_live_track_record():
    path = os.path.join(BASE_DIR, "trade_log.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None
    if df.empty or "pnl_%" not in df.columns or len(df) < 5:
        return None
    return {
        "n": len(df),
        "win_rate": round((df["pnl_%"] > 0).mean() * 100, 1),
        "avg_pnl": round(df["pnl_%"].mean(), 2),
    }


# ── shared design system ────────────────────────────────────────

BASE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* ============================================================
   AI INSIDER — design system
   One token set, one cascade. Void-black terminal aesthetic:
   glass surfaces, a signal-triad accent palette (cyan / green /
   lime) that matches the chart-drawing JS exactly, tabular-nums
   mono for every number, one restrained Inter weight scale for display type.
   ============================================================ */

:root {
  /* surface */
  --void: #05060a;
  --bg: #05060a;
  --bg-elevated: #090c13;
  --surface: rgba(15, 19, 28, 0.58);
  --surface-solid: #10141c;
  --surface-2: rgba(20, 25, 36, 0.68);
  --surface-3: #161b26;
  --border: rgba(180, 200, 224, 0.11);
  --border-strong: rgba(0, 224, 255, 0.32);
  --border-hair: rgba(255, 255, 255, 0.08);

  /* text */
  --text: #f1f5fb;
  --text-2: #a4b0c2;
  --text-3: #7b849c;
  --muted: #a4b0c2;

  /* brand / signal triad — kept in exact sync with the canvas chart JS */
  --accent: #00e0ff;
  --accent-dim: rgba(0, 224, 255, 0.1);
  --lime: #d7ff7d;
  --lime-dim: rgba(215, 255, 125, 0.1);
  --green: #00ec9f;
  --green-dim: rgba(0, 236, 159, 0.1);
  --blue: #4ba3ff;
  --blue-dim: rgba(75, 163, 255, 0.1);
  --violet: #9b8cff;
  --amber: #ffc85a;
  --amber-dim: rgba(255, 200, 90, 0.12);
  --red: #ff5c72;
  --red-dim: rgba(255, 92, 114, 0.1);

  /* rating-tier aliases */
  --aplus: var(--amber);
  --a: var(--green);
  --b: #8aa7d8;

  /* radius — concentric scale, child always one step below parent */
  --r-xs: 8px;
  --r-sm: 12px;
  --r-md: 16px;
  --r-lg: 22px;
  --r-xl: 30px;
  --r-2xl: 40px;
  --radius: var(--r-md);

  /* glass material tiers */
  --glass-thin: blur(18px) saturate(180%);
  --glass-regular: blur(28px) saturate(180%);
  --glass-thick: blur(44px) saturate(200%);
  --glass-fill: linear-gradient(158deg, rgba(255,255,255,.07) 0%, rgba(255,255,255,.022) 42%, rgba(255,255,255,.01) 68%, rgba(255,255,255,.04) 100%);
  --rim: inset 0 1px 0 rgba(255,255,255,.22), inset 0 -1px 0 rgba(255,255,255,.045), inset 1px 0 0 rgba(255,255,255,.06), inset -1px 0 0 rgba(255,255,255,.06);

  /* elevation */
  --lift-1: 0 2px 10px rgba(0,0,0,.28);
  --lift-2: 0 12px 36px rgba(0,0,0,.36);
  --lift-3: 0 26px 70px rgba(0,0,0,.46);
  --shadow: var(--lift-2);
  --glow-accent: 0 0 0 1px rgba(0,224,255,.22), 0 8px 28px rgba(0,224,255,.14);
  --glow-green: 0 0 0 1px rgba(0,236,159,.22), 0 8px 28px rgba(0,236,159,.14);

  /* motion */
  --ease: cubic-bezier(.22,.9,.32,1);
  --ease-out: cubic-bezier(.16,1,.3,1);

  /* moving background blobs — the site's original three-color mesh */
  --bg-blob-1: #a8e063;
  --bg-blob-2: #edbd62;
  --bg-blob-3: #5f7de8;

  --nav-bg: rgba(6,8,13,.72);
}

/* Light theme — same institutional system, same accent family, every
   color darkened enough off white to hold AA text contrast. Toggled via
   [data-theme] on <html>, set before first paint by THEME_INIT so there
   is no flash of the wrong theme. */
:root[data-theme="light"] {
  --void: #e3e9f2;
  --bg: #e7ecf4;
  --bg-elevated: #f6f8fc;
  --surface: rgba(255, 255, 255, 0.6);
  --surface-solid: #f6f8fc;
  --surface-2: rgba(221, 228, 240, 0.92);
  --surface-3: #dfe6f0;
  --border: rgba(15, 23, 42, 0.14);
  --border-strong: rgba(0, 120, 145, 0.42);
  --border-hair: rgba(15, 23, 42, 0.09);

  --text: #0c1524;
  --text-2: #45516a;
  --text-3: #5a6578;
  --muted: #45516a;

  --accent: #006e89;
  --accent-dim: rgba(0, 137, 171, 0.1);
  --lime: #68680e;
  --lime-dim: rgba(122, 122, 16, 0.12);
  --green: #067351;
  --green-dim: rgba(6, 122, 86, 0.1);
  --blue: #1b60c9;
  --blue-dim: rgba(28, 100, 209, 0.1);
  --violet: #644adf;
  --amber: #8c5a09;
  --amber-dim: rgba(162, 104, 10, 0.12);
  --red: #bd2942;
  --red-dim: rgba(198, 43, 69, 0.1);

  --aplus: var(--amber);
  --a: var(--green);
  --b: #446697;

  --glass-fill: linear-gradient(158deg, rgba(255,255,255,.7) 0%, rgba(255,255,255,.3) 42%, rgba(255,255,255,.14) 68%, rgba(255,255,255,.4) 100%);
  --rim: inset 0 1px 0 rgba(255,255,255,.85), inset 0 -1px 0 rgba(15,23,42,.05), inset 1px 0 0 rgba(15,23,42,.03), inset -1px 0 0 rgba(15,23,42,.03);

  --lift-1: 0 2px 8px rgba(15,23,42,.07);
  --lift-2: 0 12px 28px rgba(15,23,42,.09);
  --lift-3: 0 24px 56px rgba(15,23,42,.12);
  --shadow: var(--lift-2);
  --glow-accent: 0 0 0 1px rgba(0,137,171,.25), 0 8px 24px rgba(0,137,171,.12);
  --glow-green: 0 0 0 1px rgba(6,122,86,.25), 0 8px 24px rgba(6,122,86,.12);

  --nav-bg: rgba(255,255,255,.72);
}

*, *::before, *::after { box-sizing: border-box; }
html { color-scheme: dark; }
html[data-theme="light"] { color-scheme: light; }
@media (prefers-reduced-motion: no-preference) { html { scroll-behavior: smooth; } }
section[id] { scroll-margin-top: 132px; }
body {
  background: var(--bg); color: var(--text);
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  margin: 0; line-height: 1.55; font-size: 16px;
  position: relative; overflow-x: hidden;
  -webkit-font-smoothing: antialiased;
  text-rendering: optimizeLegibility;
}
.mono, .mono * { font-family: 'JetBrains Mono', ui-monospace, monospace; font-variant-numeric: tabular-nums; }
h1, h2, h3 { font-family: 'Inter', -apple-system, sans-serif; font-weight: 700; letter-spacing: -.02em; margin: 0; }
h1 { letter-spacing: -.03em; }
h2 { letter-spacing: -.022em; }
h4 { margin: 0; font-family: 'Inter', sans-serif; }
p { margin: 0; }
a { color: inherit; text-decoration: none; }
ul { margin: 0; }
::selection { background: rgba(0,224,255,.28); color: #fff; }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; border-radius: 4px; }
.wrap { max-width: 1120px; margin: 0 auto; padding: 0 24px; position: relative; z-index: 1; }

/* ------------------------------------------------------------
   background system — vignette + fine instrument grid + aurora
   ------------------------------------------------------------ */
body::before {
  content: ""; position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background: var(--bg);
}
body::after {
  content: ""; position: fixed; inset: 0; z-index: 0; pointer-events: none; opacity: .5;
  background-image:
    linear-gradient(rgba(255,255,255,.035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,.035) 1px, transparent 1px);
  background-size: 64px 64px;
  mask-image: radial-gradient(1100px 700px at 50% 0%, #000 0%, transparent 72%);
  -webkit-mask-image: radial-gradient(1100px 700px at 50% 0%, #000 0%, transparent 72%);
}
.bg-mesh { position: fixed; inset: 0; z-index: 0; pointer-events: none; overflow: hidden; }
.bg-mesh span { position: absolute; border-radius: 50%; filter: blur(90px); opacity: .18; }
.bg-mesh .b1 { width: 480px; height: 480px; background: var(--bg-blob-1); top: -160px; left: -100px; animation: drift1 22s ease-in-out infinite; }
.bg-mesh .b2 { width: 420px; height: 420px; background: var(--bg-blob-2); top: 20%; right: -140px; animation: drift2 26s ease-in-out infinite; }
.bg-mesh .b3 { width: 380px; height: 380px; background: var(--bg-blob-3); bottom: -120px; left: 30%; animation: drift3 30s ease-in-out infinite; }
:root[data-theme="light"] body::before {
  background:
    radial-gradient(1100px 650px at 15% -8%, rgba(168,224,99,.16), transparent 60%),
    radial-gradient(950px 620px at 100% 6%, rgba(237,189,98,.14), transparent 55%),
    radial-gradient(850px 850px at 50% 112%, rgba(95,125,232,.13), transparent 60%),
    var(--bg);
}
:root[data-theme="light"] body::after {
  background-image:
    linear-gradient(rgba(15,23,42,.05) 1px, transparent 1px),
    linear-gradient(90deg, rgba(15,23,42,.05) 1px, transparent 1px);
}
:root[data-theme="light"] .bg-mesh span { opacity: .24; }
@keyframes drift1 { 0%,100% { transform: translate(0,0); } 50% { transform: translate(60px,80px); } }
@keyframes drift2 { 0%,100% { transform: translate(0,0); } 50% { transform: translate(-70px,50px); } }
@keyframes drift3 { 0%,100% { transform: translate(0,0); } 50% { transform: translate(50px,-60px); } }

/* ------------------------------------------------------------
   buttons
   ------------------------------------------------------------ */
.cta {
  display: inline-flex; align-items: center; gap: 6px;
  background: linear-gradient(135deg, var(--accent), var(--green));
  color: #04151a; font-weight: 700; font-size: 14.5px;
  padding: 10px 20px; border-radius: 999px; text-decoration: none; border: none; cursor: pointer;
  transition: transform .18s var(--ease), box-shadow .25s var(--ease);
  box-shadow: 0 0 0 rgba(0,224,255,0);
}
.cta:hover { transform: translateY(-1px); box-shadow: var(--glow-accent); }
.cta:active { transform: translateY(0); }
.cta.ghost {
  background: var(--surface-2); border: 1px solid var(--border); color: var(--text);
  backdrop-filter: var(--glass-thin); -webkit-backdrop-filter: var(--glass-thin);
  box-shadow: none;
}
.cta.ghost:hover { border-color: var(--border-strong); box-shadow: var(--lift-1); }

/* ------------------------------------------------------------
   nav / status bar / ticker
   ------------------------------------------------------------ */
nav {
  position: sticky; top: 0; z-index: 30;
  background: var(--nav-bg);
  backdrop-filter: var(--glass-regular); -webkit-backdrop-filter: var(--glass-regular);
  border-bottom: 1px solid var(--border-hair);
}
.status-bar {
  border-bottom: 1px solid var(--border-hair);
  padding: 7px 0; font-size: 13.5px; color: var(--text-3);
  font-family: 'JetBrains Mono', monospace;
}
.status-bar .wrap { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.status-dot {
  width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0;
  background: var(--green); box-shadow: 0 0 8px var(--green);
  animation: statuspulse 2.2s ease-in-out infinite;
}
.status-dot.stale { background: var(--text-3); box-shadow: none; animation: none; }
@keyframes statuspulse { 0%,100% { opacity: 1; } 50% { opacity: .3; } }
.live-watch {
  margin-left: auto; display: inline-flex; align-items: center; gap: 7px;
  color: var(--text-3); font-size: 12.5px; letter-spacing: .03em; text-transform: uppercase;
}
.live-watch-dot {
  width: 6px; height: 6px; border-radius: 50%; background: var(--accent);
  box-shadow: 0 0 8px var(--accent); animation: statuspulse 2.2s ease-in-out infinite;
}
.live-watch.checking .live-watch-dot { background: var(--amber); box-shadow: 0 0 8px var(--amber); }
.live-watch.reconnecting .live-watch-dot { background: var(--red); box-shadow: 0 0 8px var(--red); animation: statuspulse .9s ease-in-out infinite; }

.ticker-bar {
  background: var(--bg-elevated); border-bottom: 1px solid var(--border-hair);
  overflow: hidden; white-space: nowrap; position: relative;
  mask-image: linear-gradient(90deg, transparent, #000 5%, #000 95%, transparent);
  -webkit-mask-image: linear-gradient(90deg, transparent, #000 5%, #000 95%, transparent);
}
.ticker-track { display: inline-flex; gap: 34px; padding: 9px 0; animation: ticker-scroll 48s linear infinite; }
.ticker-bar:hover .ticker-track { animation-play-state: paused; }
@keyframes ticker-scroll { 0% { transform: translateX(0); } 100% { transform: translateX(-50%); } }
.ticker-item { font-family: 'JetBrains Mono', monospace; font-size: 13.5px; color: var(--text-3); display: inline-flex; align-items: center; gap: 7px; flex-shrink: 0; }
.ticker-item .t-ticker { color: var(--text); font-weight: 700; }
.ticker-item .t-rating { font-weight: 700; }
.ticker-item .t-rating.Aplus { color: var(--amber); }
.ticker-item .t-rating.A { color: var(--green); }
.ticker-item .t-rating.Bplus { color: #9fb3d4; }
:root[data-theme="light"] .ticker-item .t-rating.Bplus { color: #3f5c85; }
.ticker-item .t-rating.B { color: var(--b); }

.nav-shell { display: flex; justify-content: space-between; align-items: center; padding: 14px 24px; }
.brand { font-family: 'Inter', -apple-system, sans-serif; font-weight: 700; font-size: 18px; display: flex; align-items: center; gap: 9px; letter-spacing: -.01em; }
.brand .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--accent); box-shadow: 0 0 12px var(--accent); }
.brand-mark {
  display: inline-flex; align-items: center; justify-content: center;
  width: 26px; height: 26px; border-radius: 8px; font-size: 12.5px; font-weight: 800;
  background: linear-gradient(135deg, var(--accent), var(--green)); color: #04151a;
  box-shadow: var(--glow-accent);
}
.brand-build {
  font-family: 'JetBrains Mono', monospace; font-size: 11.5px; font-weight: 500;
  color: var(--text-3); border: 1px solid var(--border); border-radius: 999px;
  padding: 2px 8px; margin-left: 2px; letter-spacing: .04em;
}
.nav-links {
  display: flex; gap: 22px; align-items: center; flex: 1 1 auto; min-width: 0;
  overflow-x: auto; padding: 3px 0; -webkit-overflow-scrolling: touch; scrollbar-width: none;
  mask-image: linear-gradient(90deg, transparent, #000 3%, #000 97%, transparent);
  -webkit-mask-image: linear-gradient(90deg, transparent, #000 3%, #000 97%, transparent);
}
.nav-links::-webkit-scrollbar { display: none; }
.nav-links a { font-size: 14.5px; color: var(--text-2); position: relative; padding: 2px 0; white-space: nowrap; transition: color .15s; flex-shrink: 0; }
.nav-links a:hover { color: var(--text); }
.nav-links a::after {
  content: ""; position: absolute; left: 0; right: 100%; bottom: -3px; height: 1px;
  background: var(--accent); transition: right .22s var(--ease-out);
}
.nav-links a:hover::after { right: 0; }

.nav-pinned { display: flex; align-items: center; gap: 14px; flex-shrink: 0; margin-left: 20px; }
.nav-waitlist { font-size: 14px; color: var(--text-3); white-space: nowrap; transition: color .15s; }
.nav-waitlist:hover { color: var(--text-2); }

@media (max-width: 640px) { .nav-waitlist { display: none; } }

.theme-toggle {
  flex-shrink: 0; width: 34px; height: 34px; border-radius: 50%;
  display: inline-flex; align-items: center; justify-content: center;
  background: var(--surface-2); border: 1px solid var(--border); color: var(--text-2);
  cursor: pointer; transition: border-color .18s, color .18s, transform .18s var(--ease);
}
.theme-toggle:hover { color: var(--text); border-color: var(--border-strong); transform: translateY(-1px); }
.theme-icon { width: 17px; height: 17px; }
.theme-icon-moon { display: none; }
:root[data-theme="light"] .theme-icon-sun { display: none; }
:root[data-theme="light"] .theme-icon-moon { display: block; }

.nav-hamburger {
  display: none; flex-shrink: 0; width: 34px; height: 34px; border-radius: 50%;
  align-items: center; justify-content: center; gap: 4px; flex-direction: column;
  background: var(--surface-2); border: 1px solid var(--border); cursor: pointer; padding: 0;
  transition: border-color .18s;
}
.nav-hamburger:hover { border-color: var(--border-strong); }
.nav-hamburger span { display: block; width: 15px; height: 1.6px; background: var(--text-2); border-radius: 1px; transition: transform .22s var(--ease), opacity .18s; }
.nav-hamburger[aria-expanded="true"] span:nth-child(1) { transform: translateY(5.5px) rotate(45deg); }
.nav-hamburger[aria-expanded="true"] span:nth-child(2) { opacity: 0; }
.nav-hamburger[aria-expanded="true"] span:nth-child(3) { transform: translateY(-5.5px) rotate(-45deg); }

@media (max-width: 768px) {
  .nav-hamburger { display: inline-flex; }
  .nav-links {
    position: fixed; top: 0; left: 0; right: 0; z-index: 40;
    height: 100vh; height: 100dvh;
    flex-direction: column; align-items: stretch; gap: 2px; padding: 88px 22px 24px;
    background: var(--bg-elevated); overflow-y: auto; mask-image: none; -webkit-mask-image: none;
    transform: translateX(100%); transition: transform .28s var(--ease-out);
  }
  .nav-links.mobile-open { transform: translateX(0); }
  .nav-links a { padding: 15px 4px; border-bottom: 1px solid var(--border-hair); font-size: 16px; }
  .nav-links a::after { display: none; }
  body.nav-locked { overflow: hidden; }
}

@media (max-width: 480px) {
  .nav-shell { padding: 12px 16px; }
  .brand-build { display: none; }
  .nav-pinned { gap: 8px; margin-left: 10px; }
  .nav-pinned .cta { padding: 8px 13px; font-size: 12.5px; }
}

/* signal toast */
.signal-toast-region {
  position: fixed; top: 84px; right: 18px; z-index: 60;
  display: flex; flex-direction: column; gap: 10px; width: min(340px, calc(100vw - 36px));
  pointer-events: none;
}
.signal-toast {
  pointer-events: auto; display: flex; gap: 0; overflow: hidden;
  background: var(--surface-2); border: 1px solid var(--border-strong); border-radius: var(--r-md);
  backdrop-filter: var(--glass-thick); -webkit-backdrop-filter: var(--glass-thick);
  box-shadow: var(--lift-3);
  opacity: 0; transform: translateY(-10px) scale(.98);
  transition: opacity .28s var(--ease-out), transform .28s var(--ease-out);
}
.signal-toast.show { opacity: 1; transform: translateY(0) scale(1); }
.signal-toast-accent { width: 4px; flex-shrink: 0; background: linear-gradient(180deg, var(--accent), var(--green)); }
.signal-toast-body { padding: 14px 16px; flex: 1; min-width: 0; }
.signal-toast-label { font-size: 11.5px; text-transform: uppercase; letter-spacing: .07em; color: var(--accent); font-weight: 700; }
.signal-toast-body h3 { font-family: 'Inter', -apple-system, sans-serif; font-size: 16px; margin: 6px 0 4px; }
.signal-toast-body p { font-size: 13.5px; color: var(--text-2); line-height: 1.5; }
.signal-toast-actions { display: flex; gap: 12px; align-items: center; margin-top: 10px; }
.signal-toast-link { color: var(--accent); font-size: 13.5px; font-weight: 700; }
.signal-toast-close { background: none; border: none; color: var(--text-3); font-size: 13px; cursor: pointer; padding: 0; }
.signal-toast-close:hover { color: var(--text); }

/* ------------------------------------------------------------
   hero
   ------------------------------------------------------------ */
.hero { padding: 96px 0 64px; text-align: center; }
.hero .eyebrow {
  display: inline-flex; align-items: center; gap: 8px;
  color: var(--accent); font-size: 13.5px; font-weight: 600; text-transform: uppercase;
  letter-spacing: .1em; margin-bottom: 20px;
  animation: eyebrow-in .6s var(--ease-out);
}
.eyebrow-rule { width: 20px; height: 1px; background: var(--accent); box-shadow: 0 0 6px var(--accent); }
@keyframes eyebrow-in { from { opacity: 0; transform: translateY(-6px); } to { opacity: 1; transform: translateY(0); } }
.hero h1 {
  font-size: 56px; line-height: 1.06; max-width: 820px; margin: 0 auto;
  animation: hero-in .7s var(--ease-out) .08s both;
}
@keyframes hero-in { from { opacity: 0; transform: translateY(18px); } to { opacity: 1; transform: translateY(0); } }
.hero h1 .grad {
  background: linear-gradient(100deg, var(--accent) 4%, var(--green) 50%, var(--lime) 96%);
  background-size: 220% auto;
  -webkit-background-clip: text; background-clip: text; color: transparent;
  animation: shimmer 5s ease-in-out infinite;
}
@keyframes shimmer { 0%,100% { background-position: 0% center; } 50% { background-position: 100% center; } }
.hero p { color: var(--text-2); font-size: 18px; max-width: 580px; margin: 22px auto 0; animation: hero-in .7s var(--ease-out) .16s both; }
.hero .ctas { margin-top: 34px; display: flex; gap: 12px; justify-content: center; animation: hero-in .7s var(--ease-out) .24s both; }
.hero .cta { padding: 14px 28px; font-size: 15px; }
.hero-note { display: flex; gap: 20px; justify-content: center; flex-wrap: wrap; margin-top: 28px; font-size: 13.5px; color: var(--text-3); }
.hero-note span { position: relative; padding-left: 14px; }
.hero-note span::before { content: ""; position: absolute; left: 0; top: 50%; width: 4px; height: 4px; border-radius: 50%; background: var(--green); transform: translateY(-50%); }
@media (prefers-reduced-motion: reduce) {
  .hero .eyebrow, .hero h1, .hero p, .hero .ctas, .hero h1 .grad { animation: none; }
}

/* home hero — split layout with live signal monitor */
.home-hero { padding-top: 56px; text-align: left; }
.hero-layout { display: grid; grid-template-columns: 1.15fr .85fr; gap: 40px; align-items: center; }
.home-hero .eyebrow, .home-hero .ctas, .home-hero .hero-note { justify-content: flex-start; }
.home-hero h1 { max-width: 560px; margin: 0; font-size: 46px; }
.home-hero p { max-width: 500px; margin: 20px 0 0; }
@media (max-width: 940px) {
  .hero-layout { grid-template-columns: 1fr; }
  .home-hero { text-align: center; padding-top: 40px; }
  .home-hero .eyebrow, .home-hero .ctas, .home-hero .hero-note { justify-content: center; }
  .home-hero h1, .home-hero p { margin-left: auto; margin-right: auto; }
}

.signal-monitor {
  background: var(--surface); border: 1px solid var(--border); border-radius: var(--r-lg);
  backdrop-filter: var(--glass-regular); -webkit-backdrop-filter: var(--glass-regular);
  background-image: var(--glass-fill); box-shadow: var(--rim), var(--lift-2);
  padding: 20px; text-align: left;
}
.monitor-head { display: flex; justify-content: space-between; align-items: center; padding-bottom: 14px; border-bottom: 1px solid var(--border-hair); margin-bottom: 8px; }
.monitor-kicker { display: block; font-size: 11.5px; text-transform: uppercase; letter-spacing: .08em; color: var(--text-3); }
.monitor-head strong { font-family: 'Inter', -apple-system, sans-serif; font-size: 17px; }
.monitor-live {
  font-size: 11.5px; font-weight: 700; text-transform: uppercase; letter-spacing: .06em;
  color: var(--green); background: var(--green-dim); border: 1px solid rgba(0,236,159,.28);
  padding: 4px 9px; border-radius: 999px;
}
.monitor-row { display: flex; justify-content: space-between; align-items: center; gap: 12px; padding: 12px 4px; border-bottom: 1px solid var(--border-hair); }
.monitor-row:last-of-type { border-bottom: none; }
.monitor-company { display: flex; flex-direction: column; gap: 2px; min-width: 0; }
.monitor-company strong { font-family: 'JetBrains Mono', monospace; font-size: 15px; }
.monitor-company span { font-size: 13px; color: var(--text-3); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 150px; }
.monitor-price { text-align: right; }
.monitor-price strong { font-family: 'JetBrains Mono', monospace; font-size: 15px; display: block; }
.monitor-price span { font-size: 12px; color: var(--text-3); }
.monitor-rating, .trade-badge {
  display: inline-flex; align-items: center; gap: 5px; font-size: 12.5px; font-weight: 700; text-transform: uppercase;
  letter-spacing: .05em; padding: 4px 9px; border-radius: 999px; white-space: nowrap;
}
.monitor-rating::before, .trade-badge::before { content: ""; width: 5px; height: 5px; border-radius: 50%; background: currentColor; }
.monitor-rating.Aplus, .trade-badge.Aplus { background: var(--amber-dim); color: var(--amber); border: 1px solid rgba(255,200,90,.28); }
.monitor-rating.A, .trade-badge.A { background: var(--green-dim); color: var(--green); border: 1px solid rgba(0,236,159,.28); }
.monitor-rating.Bplus, .trade-badge.Bplus { background: var(--blue-dim); color: #9fb3d4; border: 1px solid rgba(75,163,255,.24); }
:root[data-theme="light"] .monitor-rating.Bplus, :root[data-theme="light"] .trade-badge.Bplus { color: #3f5c85; }
.monitor-rating.B, .trade-badge.B { background: rgba(138,167,216,.1); color: var(--b); border: 1px solid rgba(138,167,216,.24); }
.monitor-empty { color: var(--text-3); font-size: 14px; padding: 20px 4px; text-align: center; }
.monitor-foot { display: flex; flex-direction: column; gap: 3px; padding-top: 12px; margin-top: 4px; border-top: 1px solid var(--border-hair); }
.monitor-foot span { font-size: 11.5px; text-transform: uppercase; letter-spacing: .06em; color: var(--text-3); }
.monitor-foot strong { font-size: 13px; font-weight: 500; color: var(--text-2); }

/* ------------------------------------------------------------
   sections
   ------------------------------------------------------------ */
section { padding: 76px 0; border-top: 1px solid var(--border-hair); position: relative; }
.section-head { text-align: center; max-width: 620px; margin: 0 auto 46px; }
.section-head .eyebrow { color: var(--accent); font-size: 13.5px; font-weight: 600; text-transform: uppercase; letter-spacing: .1em; }
.section-head h2 { font-size: 33px; margin-top: 10px; }
.section-head p { color: var(--text-2); font-size: 16px; margin-top: 12px; }

/* ------------------------------------------------------------
   performance dashboard — flagship "terminal" panel
   ------------------------------------------------------------ */
.performance-terminal { padding-top: 52px; }
.performance-shell {
  position: relative; overflow: hidden;
  background: var(--surface); border: 1px solid var(--border); border-radius: var(--r-xl);
  backdrop-filter: var(--glass-regular); -webkit-backdrop-filter: var(--glass-regular);
  background-image: var(--glass-fill); box-shadow: var(--rim), var(--lift-3);
  padding: 32px 34px 30px;
}
.performance-shell::before {
  content: ""; position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, var(--accent), var(--green) 55%, var(--lime), transparent);
}
.performance-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 20px; flex-wrap: wrap; margin-bottom: 26px; }
.performance-heading .eyebrow { color: var(--accent); font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: .1em; }
.performance-heading h2 { font-size: 25px; margin-top: 8px; font-family: 'Inter', -apple-system, sans-serif; }
.performance-heading p { color: var(--text-2); font-size: 15px; margin-top: 8px; max-width: 480px; }
.performance-badges { display: flex; gap: 8px; flex-wrap: wrap; align-items: flex-start; }
.performance-badge {
  font-family: 'JetBrains Mono', monospace; font-size: 12.5px; color: var(--text-2);
  background: var(--surface-2); border: 1px solid var(--border); border-radius: 999px; padding: 6px 12px;
}
.performance-badge.live { color: var(--green); border-color: rgba(0,236,159,.3); background: var(--green-dim); }

.performance-kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px; background: var(--border-hair); border: 1px solid var(--border-hair); border-radius: var(--r-md); overflow: hidden; margin-bottom: 26px; }
.performance-kpis.owner-only { grid-template-columns: 1fr; }
.performance-kpi { background: var(--surface-solid); padding: 18px 20px; display: flex; flex-direction: column; gap: 6px; }
.performance-kpi span { font-size: 12px; text-transform: uppercase; letter-spacing: .07em; color: var(--text-3); }
.performance-kpi strong { font-family: 'JetBrains Mono', monospace; font-size: 25px; font-weight: 700; font-variant-numeric: tabular-nums; }
.performance-kpi small { font-size: 12.5px; color: var(--text-3); line-height: 1.4; }
.performance-kpi.owner strong { color: var(--lime); }
.performance-kpi strong.pos, .performance-kpi strong[data-strategy-return].pos { color: var(--green); }
.performance-kpi strong.neg { color: var(--red); }
@media (max-width: 760px) { .performance-kpis { grid-template-columns: repeat(2, 1fr); } }

.performance-chart-panel { }
.performance-chart-toolbar { display: flex; justify-content: space-between; align-items: center; gap: 16px; flex-wrap: wrap; margin-bottom: 18px; }
.performance-window-label { display: flex; flex-direction: column; gap: 2px; }
.performance-window-label span { font-size: 11.5px; text-transform: uppercase; letter-spacing: .06em; color: var(--text-3); }
.performance-window-label strong { font-family: 'Inter', -apple-system, sans-serif; font-size: 16px; }
.performance-ranges { display: inline-flex; gap: 3px; background: var(--surface-solid); border: 1px solid var(--border); border-radius: 999px; padding: 3px; }
.performance-range {
  font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 600; color: var(--text-2);
  background: transparent; border: none; border-radius: 999px; padding: 7px 14px; cursor: pointer;
  transition: background .18s, color .18s;
}
.performance-range:hover { color: var(--text); }
.performance-range.active { background: linear-gradient(135deg, var(--accent), var(--green)); color: #04151a; }

.performance-plot { position: relative; background: var(--surface-solid); border: 1px solid var(--border-hair); border-radius: var(--r-md); padding: 6px; }
.performance-canvas { width: 100%; height: 320px; display: block; }
.performance-tooltip {
  position: absolute; pointer-events: none; opacity: 0; transform: translateY(4px);
  background: var(--bg-elevated); border: 1px solid var(--border-strong); border-radius: var(--r-sm);
  padding: 10px 13px; font-size: 13px; box-shadow: var(--lift-2); transition: opacity .15s, transform .15s;
  min-width: 150px; z-index: 5;
}
.performance-tooltip.show { opacity: 1; transform: translateY(0); }
.performance-tooltip time { display: block; font-family: 'JetBrains Mono', monospace; color: var(--text-3); font-size: 12px; margin-bottom: 6px; }
.performance-tooltip div { display: flex; justify-content: space-between; gap: 14px; padding: 2px 0; color: var(--text-2); }
.performance-tooltip strong { font-family: 'JetBrains Mono', monospace; font-variant-numeric: tabular-nums; }
.performance-tooltip .owner-value { color: var(--lime); }
.performance-tooltip .strategy-value { color: var(--green); }
.performance-tooltip .benchmark-value { color: var(--blue); }

.performance-legend { display: flex; gap: 18px; flex-wrap: wrap; justify-content: center; margin-top: 16px; font-size: 13.5px; color: var(--text-2); }
.performance-legend span { display: inline-flex; align-items: center; gap: 7px; }
.performance-legend i { width: 10px; height: 2.5px; border-radius: 2px; background: var(--green); display: inline-block; }
.performance-legend .owner i { background: var(--lime); }
.performance-legend .benchmark i { background: var(--blue); }
.performance-legend.owner-mode [data-backtest-legend] { display: none; }

.performance-foot { display: flex; justify-content: space-between; gap: 20px; flex-wrap: wrap; margin-top: 18px; padding-top: 16px; border-top: 1px solid var(--border-hair); }
.performance-risk, .performance-asof { font-size: 13px; color: var(--text-3); line-height: 1.6; margin: 0; max-width: 480px; }
.performance-risk strong { color: var(--text-2); }
.performance-asof { text-align: right; }
@media (max-width: 760px) {
  .performance-chart-toolbar { flex-direction: column; align-items: flex-start; }
  .performance-foot { flex-direction: column; }
  .performance-asof { text-align: left; }
  .performance-canvas { height: 260px; }
}

/* ytd trade ledger table (inside dashboard) */
.ytd-ledger { margin-top: 30px; padding-top: 26px; border-top: 1px solid var(--border-hair); }
.ytd-ledger-head { display: flex; justify-content: space-between; align-items: flex-end; gap: 16px; flex-wrap: wrap; margin-bottom: 14px; }
.ytd-ledger-head h3 { font-family: 'Inter', -apple-system, sans-serif; font-size: 19px; }
.ytd-ledger-head p { font-size: 14px; color: var(--text-3); margin-top: 6px; max-width: 420px; }
.ytd-ledger-updated { text-align: right; font-size: 12px; color: var(--text-3); text-transform: uppercase; letter-spacing: .04em; }
.ytd-ledger-updated strong { display: block; font-family: 'JetBrains Mono', monospace; color: var(--text-2); font-size: 13.5px; text-transform: none; letter-spacing: 0; margin-top: 2px; }
.ytd-ledger-scroll { overflow: auto; max-height: 620px; border: 1px solid var(--border-hair); border-radius: var(--r-md); background: var(--surface-solid); -webkit-overflow-scrolling: touch; }
.ytd-ledger-table { width: 100%; border-collapse: collapse; font-family: 'JetBrains Mono', monospace; font-size: 14px; min-width: 560px; }
.ytd-ledger-table thead th {
  position: sticky; top: 0; z-index: 2;
  text-align: left; padding: 11px 16px; font-size: 12px; text-transform: uppercase; letter-spacing: .06em;
  color: var(--text-3); border-bottom: 1px solid var(--border-hair); font-weight: 600;
  background: var(--surface-solid); backdrop-filter: var(--glass-thin); -webkit-backdrop-filter: var(--glass-thin);
}
.ytd-ledger-table tbody td { padding: 10px 16px; border-bottom: 1px solid var(--border-hair); }
.ytd-ledger-table tbody tr:last-child td { border-bottom: none; }
.ytd-ledger-table tbody tr { transition: background .15s; }
.ytd-ledger-table tbody tr:hover { background: rgba(255,255,255,.025); }
:root[data-theme="light"] .ytd-ledger-table tbody tr:hover { background: rgba(15,23,42,.03); }
.trade-date { color: var(--text-3); }
.trade-symbol { color: var(--text); font-weight: 700; }
.trade-rating { margin-left: 7px; font-size: 11.5px; font-weight: 700; color: var(--accent); background: var(--accent-dim); border-radius: 5px; padding: 1px 5px; }
.ytd-status {
  display: inline-flex; align-items: center; gap: 5px; font-size: 11.5px; font-weight: 700; text-transform: uppercase;
  letter-spacing: .05em; padding: 3px 9px; border-radius: 999px;
}
.ytd-status::before { content: ""; width: 5px; height: 5px; border-radius: 50%; background: currentColor; }
.ytd-status.open-signal, .ytd-status.open { color: var(--accent); background: var(--accent-dim); }
.ytd-status.closed, .ytd-status.win, .ytd-status.resolved { color: var(--green); background: var(--green-dim); }
.ytd-status.loss { color: var(--red); background: var(--red-dim); }
.ytd-trade-return { text-align: right; font-variant-numeric: tabular-nums; }
.ytd-trade-return.pos { color: var(--green); }
.ytd-trade-return.neg { color: var(--red); }
.ytd-ledger-note { font-size: 12.5px; color: var(--text-3); margin-top: 12px; line-height: 1.6; }

.performance-methodology { display: flex; gap: 12px; margin-top: 22px; padding-top: 20px; border-top: 1px solid var(--border-hair); }
.performance-methodology-mark { width: 6px; height: 6px; border-radius: 50%; background: var(--accent); box-shadow: 0 0 8px var(--accent); flex-shrink: 0; margin-top: 6px; }
.performance-methodology p { font-size: 13px; color: var(--text-3); line-height: 1.7; }
.performance-methodology strong { color: var(--text-2); }

/* ------------------------------------------------------------
   trust bar
   ------------------------------------------------------------ */
.trust-bar { padding: 30px 0; border-top: 1px solid var(--border-hair); border-bottom: 1px solid var(--border-hair); }
.trust-stat { text-align: center; }
.trust-stat .n { font-family: 'JetBrains Mono', monospace; font-size: 27px; font-weight: 800; color: var(--text); font-variant-numeric: tabular-nums; }
.trust-stat .l { color: var(--text-3); font-size: 13px; text-transform: uppercase; letter-spacing: .07em; margin-top: 5px; }

/* ------------------------------------------------------------
   market activity banner — raw, unfiltered SEC feed (distinct from
   the AI-scored signal ticker in nav: amber accent, not accent/green)
   ------------------------------------------------------------ */
.market-activity-banner { padding: 40px 0 44px; }
.market-activity-head { max-width: 640px; margin: 0 auto 22px; text-align: center; }
.market-activity-head .eyebrow { color: var(--amber); font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: .09em; }
.market-activity-head h2 { font-size: 24px; margin-top: 10px; }
.market-activity-head p { color: var(--text-2); font-size: 14.5px; margin-top: 10px; line-height: 1.6; }
.market-activity-scroll {
  overflow: hidden; white-space: nowrap; position: relative;
  border: 1px solid var(--border); border-radius: var(--r-md); background: var(--surface-solid);
  mask-image: linear-gradient(90deg, transparent, #000 4%, #000 96%, transparent);
  -webkit-mask-image: linear-gradient(90deg, transparent, #000 4%, #000 96%, transparent);
}
.market-activity-track { display: inline-flex; gap: 0; animation: ticker-scroll 60s linear infinite; }
.market-activity-scroll:hover .market-activity-track { animation-play-state: paused; }
.market-activity-item {
  display: inline-flex; align-items: center; gap: 10px; flex-shrink: 0;
  padding: 14px 20px; border-right: 1px solid var(--border-hair);
  font-family: 'JetBrains Mono', monospace; font-size: 13.5px; color: var(--text-2);
}
.ma-ticker { color: var(--text); font-weight: 700; }
.ma-type { font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: .04em; padding: 2px 8px; border-radius: 999px; }
.ma-type.buy { color: var(--green); background: var(--green-dim); }
.ma-type.sell { color: var(--red); background: var(--red-dim); }
.ma-insider { color: var(--text-3); max-width: 220px; overflow: hidden; text-overflow: ellipsis; }
.ma-value { color: var(--amber); }
.ma-date { color: var(--text-3); }
@media (max-width: 640px) { .ma-insider { max-width: 120px; } }

/* ------------------------------------------------------------
   filings table
   ------------------------------------------------------------ */
.filings-tabs { display: inline-flex; gap: 3px; background: var(--surface-solid); border: 1px solid var(--border); border-radius: 999px; padding: 3px; margin-bottom: 20px; }
.filings-tab {
  font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 600; color: var(--text-2);
  background: transparent; border: none; border-radius: 999px; padding: 8px 18px; cursor: pointer;
  transition: background .18s, color .18s;
}
.filings-tab:hover { color: var(--text); }
.filings-tab.active { background: linear-gradient(135deg, var(--accent), var(--green)); color: #04151a; }
.filings-table-wrap { overflow: auto; max-height: 680px; border: 1px solid var(--border); border-radius: var(--r-lg); background: var(--surface-solid); -webkit-overflow-scrolling: touch; }
table.filings { width: 100%; border-collapse: collapse; font-family: 'JetBrains Mono', monospace; font-size: 14.5px; min-width: 740px; }
table.filings thead th {
  position: sticky; top: 0; z-index: 2;
  text-align: left; padding: 13px 18px; font-size: 12.5px; text-transform: uppercase; letter-spacing: .05em;
  color: var(--text-3); border-bottom: 1px solid var(--border); font-weight: 600; background: var(--surface-2);
  backdrop-filter: var(--glass-thin); -webkit-backdrop-filter: var(--glass-thin);
}
table.filings tbody td { padding: 12px 18px; border-bottom: 1px solid var(--border-hair); }
table.filings tbody tr:last-child td { border-bottom: none; }
table.filings tbody tr { transition: background .15s; }
table.filings tbody tr:hover { background: rgba(255,255,255,.025); }
:root[data-theme="light"] table.filings tbody tr:hover { background: rgba(15,23,42,.03); }
table.filings .tk { font-weight: 700; color: var(--text); }
table.filings .role { color: var(--text-3); font-size: 13.5px; margin-top: 2px; }
table.filings .amt { text-align: right; }

/* ------------------------------------------------------------
   market dashboard (tradingview + snapshots)
   ------------------------------------------------------------ */
.ticker-search {
  display: flex; align-items: center; gap: 10px; max-width: 460px; margin: 0 auto 14px;
  background: var(--surface-solid); border: 1px solid var(--border); border-radius: 999px;
  padding: 6px 8px 6px 18px; transition: border-color .18s;
}
.ticker-search:focus-within { border-color: var(--border-strong); }
.ticker-search-icon { width: 17px; height: 17px; color: var(--text-3); flex-shrink: 0; }
.ticker-search input {
  flex: 1; min-width: 0; background: none; border: none; outline: none;
  color: var(--text); font-family: 'JetBrains Mono', monospace; font-size: 14px; font-weight: 600;
  letter-spacing: .02em;
}
.ticker-search input::placeholder { color: var(--text-3); font-family: 'Inter', sans-serif; font-weight: 400; letter-spacing: normal; }
.ticker-search button {
  flex-shrink: 0; background: linear-gradient(135deg, var(--accent), var(--green)); color: #04151a;
  font-weight: 700; font-size: 13px; border: none; border-radius: 999px; padding: 10px 20px; cursor: pointer;
  transition: transform .18s var(--ease), box-shadow .25s var(--ease);
}
.ticker-search button:hover { transform: translateY(-1px); box-shadow: var(--glow-accent); }
.ticker-search-error { text-align: center; color: var(--amber); font-size: 12.5px; margin: -6px 0 14px; }
.ticker-chips { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin: 0 0 28px; }
.ticker-chip {
  font-family: 'JetBrains Mono', monospace; font-size: 12.5px; font-weight: 600; color: var(--text-2);
  background: var(--surface-2); border: 1px solid var(--border); border-radius: 999px; padding: 6px 14px;
  transition: border-color .18s, color .18s, background .18s;
}
.ticker-chip:hover { color: var(--text); border-color: var(--border-strong); }
.ticker-chip.active { color: var(--accent); border-color: var(--border-strong); background: var(--accent-dim); }
@media (max-width: 480px) { .ticker-search { border-radius: var(--r-lg); flex-wrap: wrap; padding: 12px; } .ticker-search input { flex-basis: 100%; padding: 6px 4px; } .ticker-search button { flex: 1; } }

.market-dashboard { display: grid; grid-template-columns: 1.8fr 1fr; gap: 22px; align-items: start; }
@media (max-width: 900px) { .market-dashboard { grid-template-columns: 1fr; } }

.market-chart-panel {
  background: var(--surface); border: 1px solid var(--border); border-radius: var(--r-lg);
  backdrop-filter: var(--glass-thin); -webkit-backdrop-filter: var(--glass-thin);
  overflow: hidden;
}
.market-panel-head { display: flex; justify-content: space-between; align-items: center; gap: 12px; padding: 16px 20px; border-bottom: 1px solid var(--border-hair); flex-wrap: wrap; }
.market-panel-head span { display: block; font-size: 11.5px; text-transform: uppercase; letter-spacing: .06em; color: var(--text-3); }
.market-panel-head strong { font-family: 'Inter', -apple-system, sans-serif; font-size: 16px; }
.market-delay-note { font-size: 12px; color: var(--text-3); max-width: 220px; text-align: right; }
.tradingview-widget-container { min-height: 820px; }
.tradingview-widget-container__widget { min-height: 820px; }
.tradingview-widget-copyright { font-size: 12px; color: var(--text-3); padding: 4px 20px; }
.tradingview-widget-copyright a { color: var(--text-3); }

.market-snapshot-list { display: flex; flex-direction: column; gap: 14px; }
@media (max-width: 900px) { .market-snapshot-list { display: grid; grid-template-columns: 1fr 1fr; } }
@media (max-width: 620px) { .market-snapshot-list { grid-template-columns: 1fr; } }
.market-snapshot-card {
  background: var(--surface); border: 1px solid var(--border); border-radius: var(--r-md);
  backdrop-filter: var(--glass-thin); -webkit-backdrop-filter: var(--glass-thin);
  padding: 16px 18px; transition: border-color .2s, transform .2s;
}
.market-snapshot-card:hover { border-color: var(--border-strong); transform: translateY(-2px); }
.market-snapshot-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 10px; }
.market-snapshot-symbol strong { font-family: 'JetBrains Mono', monospace; font-size: 15.5px; display: block; }
.market-snapshot-symbol span { font-size: 13px; color: var(--text-3); }
.market-snapshot-price { display: flex; align-items: baseline; gap: 10px; margin-top: 12px; flex-wrap: wrap; }
.market-data-label { font-size: 11.5px; text-transform: uppercase; letter-spacing: .05em; color: var(--text-3); width: 100%; }
.market-snapshot-price strong { font-family: 'JetBrains Mono', monospace; font-size: 20px; }
.market-change { font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 600; }
.market-change.pos { color: var(--green); }
.market-change.neg { color: var(--red); }
.market-metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 14px; padding-top: 12px; border-top: 1px solid var(--border-hair); }
.market-metric span { display: block; font-size: 11px; text-transform: uppercase; letter-spacing: .04em; color: var(--text-3); }
.market-metric strong { font-family: 'JetBrains Mono', monospace; font-size: 13.5px; }
.market-range { margin-top: 14px; }
.market-range-head { display: flex; justify-content: space-between; font-size: 12px; color: var(--text-3); margin-bottom: 6px; }
.market-range-head strong { color: var(--text-2); font-family: 'JetBrains Mono', monospace; }
.market-range-track { position: relative; height: 4px; border-radius: 999px; background: var(--border); }
.market-range-fill { position: absolute; inset: 0; width: var(--range-position, 0%); border-radius: 999px; background: linear-gradient(90deg, var(--accent), var(--green)); }
.market-range-pin { position: absolute; top: 50%; left: var(--range-position, 0%); width: 8px; height: 8px; border-radius: 50%; background: #fff; box-shadow: 0 0 0 2px var(--bg), 0 0 8px var(--accent); transform: translate(-50%, -50%); }
:root[data-theme="light"] .market-range-pin { background: var(--text); }
.market-range-labels { display: flex; justify-content: space-between; font-size: 11.5px; color: var(--text-3); margin-top: 6px; font-family: 'JetBrains Mono', monospace; }
.market-source-note { font-size: 11.5px; color: var(--text-3); margin-top: 12px; }

/* ------------------------------------------------------------
   trade cards / proof grid
   ------------------------------------------------------------ */
.trade-grid { display: grid; grid-template-columns: repeat(auto-fit,minmax(280px,1fr)); gap: 18px; }
.trade-card {
  background: var(--surface); border: 1px solid var(--border); border-radius: var(--r-lg); padding: 22px;
  backdrop-filter: var(--glass-thin); -webkit-backdrop-filter: var(--glass-thin);
  background-image: var(--glass-fill); box-shadow: var(--rim), var(--lift-1);
  transition: border-color .2s, transform .2s var(--ease), box-shadow .2s;
}
.trade-card:hover { border-color: var(--border-strong); transform: translateY(-4px); box-shadow: var(--rim), var(--lift-2), var(--glow-accent); }
.trade-top { display: flex; justify-content: space-between; align-items: flex-start; }
.trade-avatar {
  width: 38px; height: 38px; border-radius: 50%; background: var(--surface-2);
  display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 14px;
  color: var(--text-2); border: 1px solid var(--border); font-family: 'JetBrains Mono', monospace;
}
.trade-gain { font-family: 'JetBrains Mono', monospace; font-size: 24px; font-weight: 800; font-variant-numeric: tabular-nums; }
.trade-gain.pos { color: var(--green); }
.trade-gain.neg { color: var(--red); }
.trade-ticker { font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 16px; margin-top: 12px; }
.trade-meta { color: var(--text-3); font-size: 14px; margin-top: 4px; }
.trade-narrative { font-size: 15px; color: var(--text-2); margin-top: 14px; line-height: 1.6; }

/* Finviz-style compact market data strip */
.mkt-strip { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px 10px; margin-top: 14px; padding-top: 12px; border-top: 1px solid var(--border-hair); font-family: 'JetBrains Mono', monospace; }
.mkt-cell { font-size: 14px; }
.mkt-cell .l { color: var(--text-3); text-transform: uppercase; letter-spacing: .04em; font-size: 12.5px; display: block; }
.mkt-cell .v { color: var(--text); font-weight: 600; font-size: 14px; }
.mkt-cell .v.hi { color: var(--red); }
.mkt-cell .v.lo { color: var(--green); }

/* ------------------------------------------------------------
   featured signal alert
   ------------------------------------------------------------ */
.new-trade-alert {
  position: relative; overflow: hidden; margin-bottom: 22px;
  background: var(--surface-2); border: 1px solid var(--border-strong); border-radius: var(--r-xl); padding: 28px 30px;
  backdrop-filter: var(--glass-regular); -webkit-backdrop-filter: var(--glass-regular);
  background-image: var(--glass-fill); box-shadow: var(--rim), var(--lift-2);
}
.new-trade-alert::before {
  content: ""; position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, var(--accent), var(--lime));
}
.new-trade-alert-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; flex-wrap: wrap; }
.new-trade-kicker { display: inline-block; font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; color: var(--accent); }
.new-trade-alert-head h3 { font-family: 'Inter', -apple-system, sans-serif; font-size: 21px; margin-top: 8px; }
.new-trade-alert-head p { font-size: 14px; color: var(--text-3); margin-top: 6px; }
.new-trade-alert-badges { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.conviction-pill {
  font-size: 12.5px; font-weight: 700; text-transform: uppercase; letter-spacing: .04em;
  color: var(--lime); background: var(--lime-dim); border: 1px solid rgba(215,255,125,.28);
  padding: 4px 10px; border-radius: 999px;
}
.new-trade-alert-main { display: flex; gap: 30px; flex-wrap: wrap; margin-top: 24px; padding-top: 22px; border-top: 1px solid var(--border-hair); }
.new-trade-symbol { display: flex; flex-direction: column; gap: 3px; }
.new-trade-symbol span { font-size: 11.5px; text-transform: uppercase; letter-spacing: .06em; color: var(--text-3); }
.new-trade-symbol strong { font-family: 'JetBrains Mono', monospace; font-size: 30px; }
.new-trade-symbol small { font-size: 13px; color: var(--text-3); }
.new-trade-stats { display: grid; grid-template-columns: repeat(4, auto); gap: 24px; flex: 1; }
.new-trade-stat { display: flex; flex-direction: column; gap: 3px; }
.new-trade-stat span { font-size: 11.5px; text-transform: uppercase; letter-spacing: .06em; color: var(--text-3); }
.new-trade-stat strong { font-family: 'JetBrains Mono', monospace; font-size: 18px; }
.new-trade-stat small { font-size: 12.5px; color: var(--text-3); }
.new-trade-stat.allocation strong { color: var(--lime); }
@media (max-width: 760px) { .new-trade-stats { grid-template-columns: repeat(2, 1fr); } }

.new-trade-copy { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 22px; }
@media (max-width: 640px) { .new-trade-copy { grid-template-columns: 1fr; } }
.new-trade-copy-block { background: var(--surface-solid); border: 1px solid var(--border-hair); border-radius: var(--r-sm); padding: 16px 18px; }
.new-trade-copy-label { font-size: 11.5px; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; color: var(--accent); }
.new-trade-copy-block.watch .new-trade-copy-label { color: var(--amber); }
.new-trade-copy-block p { font-size: 14.5px; color: var(--text-2); margin-top: 8px; line-height: 1.6; }

.new-trade-risk { display: flex; gap: 12px; margin-top: 20px; padding: 14px 16px; background: var(--red-dim); border: 1px solid rgba(255,92,114,.28); border-radius: var(--r-sm); }
.new-trade-risk-icon {
  flex-shrink: 0; width: 22px; height: 22px; border-radius: 50%; background: var(--red); color: #2a0710;
  display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 14px;
}
.new-trade-risk strong { color: var(--red); font-size: 14px; }
.new-trade-risk p { font-size: 13.5px; color: var(--text-2); margin-top: 4px; line-height: 1.5; }

.signal-list-label { font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: .07em; color: var(--text-3); margin: 32px 0 16px; }
.signal-brief-grid { display: grid; grid-template-columns: repeat(auto-fit,minmax(340px,1fr)); gap: 18px; }
.signal-brief-card { padding: 24px; }
.signal-brief-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; flex-wrap: wrap; }
.signal-brief-symbol { display: flex; gap: 12px; align-items: center; }
.signal-brief-symbol .trade-ticker { margin-top: 0; }
.signal-company { font-size: 13px; color: var(--text-3); }
.signal-brief-badges { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
.signal-framework { display: grid; grid-template-columns: repeat(auto-fit,minmax(120px,1fr)); gap: 12px; margin-top: 18px; padding-top: 16px; border-top: 1px solid var(--border-hair); }
.signal-framework-cell { display: flex; flex-direction: column; gap: 3px; }
.signal-framework-cell span { font-size: 11px; text-transform: uppercase; letter-spacing: .04em; color: var(--text-3); }
.signal-framework-cell strong { font-family: 'JetBrains Mono', monospace; font-size: 14.5px; }
.signal-framework-cell small { font-size: 12px; color: var(--text-3); }
.signal-brief-body { margin-top: 16px; }
.signal-insider-line { font-size: 13.5px; color: var(--text-2); padding-bottom: 12px; margin-bottom: 12px; border-bottom: 1px solid var(--border-hair); }
.signal-track-line { font-size: 12.5px; color: var(--text-2); margin-bottom: 12px; }
.signal-track-line strong { color: var(--green); font-weight: 700; }
.signal-track-line span { color: var(--text-3); }
.signal-thesis-block { margin-top: 12px; }
.signal-thesis-label { font-size: 11.5px; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; color: var(--accent); }
.signal-thesis-block.signal-watch .signal-thesis-label { color: var(--amber); }
.signal-thesis-text { font-size: 14.5px; color: var(--text-2); margin-top: 6px; line-height: 1.6; }
.signal-sizing-note { font-size: 12px; color: var(--text-3); margin-top: 12px; line-height: 1.5; }

/* ------------------------------------------------------------
   compare grid
   ------------------------------------------------------------ */
.compare-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
@media (max-width: 720px) { .compare-grid { grid-template-columns: 1fr; } }
.compare-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--r-lg); padding: 28px; backdrop-filter: var(--glass-thin); -webkit-backdrop-filter: var(--glass-thin); }
.compare-card.highlight { border-color: var(--border-strong); background: var(--surface-2); background-image: linear-gradient(180deg, rgba(0,224,255,.06), transparent 60%); box-shadow: var(--glow-accent); }
.compare-card h3 { font-family: 'Inter', -apple-system, sans-serif; font-size: 19px; margin-bottom: 16px; }
.compare-card ul { list-style: none; padding: 0; margin: 0; }
.compare-card li { padding: 9px 0; color: var(--text-2); font-size: 15px; border-top: 1px solid var(--border-hair); }
.compare-card li:first-child { border-top: none; }
.compare-card.highlight li { color: var(--text); }
.compare-card.highlight li::before { content: "\\2713  "; color: var(--accent); font-weight: 700; }

/* ------------------------------------------------------------
   pricing
   ------------------------------------------------------------ */
.pricing-grid { display: grid; grid-template-columns: repeat(auto-fit,minmax(280px,1fr)); gap: 20px; max-width: 740px; margin: 0 auto; }
.price-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--r-lg); padding: 28px; position: relative; backdrop-filter: var(--glass-thin); -webkit-backdrop-filter: var(--glass-thin); }
.price-card.rec { border-color: var(--border-strong); box-shadow: var(--glow-accent); }
.price-card .rec-tag {
  position: absolute; top: -12px; right: 20px; background: linear-gradient(135deg, var(--accent), var(--green)); color: #04151a;
  font-size: 13px; font-weight: 700; padding: 4px 11px; border-radius: 999px;
}
.price-card h3 { font-family: 'Inter', -apple-system, sans-serif; font-size: 19px; }
.price-card .desc { color: var(--text-3); font-size: 14px; margin-top: 4px; }
.price-card .price { font-family: 'JetBrains Mono', monospace; font-size: 32px; font-weight: 700; margin-top: 18px; }
.price-card .price .per { font-size: 15px; color: var(--text-3); font-weight: 400; }
.price-card ul { list-style: none; padding: 0; margin: 20px 0 0; }
.price-card li { padding: 7px 0; font-size: 14.5px; color: var(--text-2); }
.price-card li::before { content: "\\2713  "; color: var(--green); }
.coming-soon-tag {
  display: inline-block; background: var(--surface-2); border: 1px solid var(--border); color: var(--text-3);
  font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: .06em;
  padding: 4px 10px; border-radius: 999px; margin-top: 16px;
}

/* ------------------------------------------------------------
   faq / ad / disclaimer / empty
   ------------------------------------------------------------ */
.faq-item { border-bottom: 1px solid var(--border-hair); padding: 20px 0; }
.faq-item h4 { font-size: 16px; font-weight: 600; }
.faq-item p { color: var(--text-2); font-size: 15px; margin-top: 10px; line-height: 1.6; }

.ad-slot {
  border: 1px dashed var(--border); color: var(--text-3); font-size: 13.5px;
  text-align: center; padding: 36px 12px; margin: 8px 0 0; border-radius: var(--r-sm);
  text-transform: uppercase; letter-spacing: .06em;
}

.disclaimer-box {
  background: var(--red-dim); border: 1px solid rgba(255,92,114,.24);
  border-radius: var(--r-sm); padding: 16px 20px; font-size: 14.5px; color: var(--text-2); margin-top: 24px;
}
.disclaimer-box strong { color: var(--red); }

.empty { color: var(--text-3); padding: 26px; text-align: center; border: 1px dashed var(--border); border-radius: var(--r-md); font-size: 14.5px; }

.fade-in { opacity: 0; transform: translateY(14px); transition: opacity .5s var(--ease-out), transform .5s var(--ease-out); }
.fade-in.in-view { opacity: 1; transform: translateY(0); }

/* ------------------------------------------------------------
   footer
   ------------------------------------------------------------ */
footer { padding: 52px 0 60px; border-top: 1px solid var(--border-hair); position: relative; }
footer::before { content: ""; position: absolute; top: 0; left: 0; right: 0; height: 1px; background: linear-gradient(90deg, transparent, var(--border-strong) 40%, transparent 60%, transparent); }
.footer-grid { display: flex; justify-content: space-between; flex-wrap: wrap; gap: 32px; }
.footer-col a { display: block; color: var(--text-3); text-decoration: none; font-size: 14.5px; margin-bottom: 9px; transition: color .15s; }
.footer-col a:hover { color: var(--text); }
.footer-note { color: var(--text-3); font-size: 13.5px; margin-top: 42px; max-width: 660px; line-height: 1.6; }

/* ------------------------------------------------------------
   generic stat grid (track-record / leaderboard pages)
   ------------------------------------------------------------ */
.stat-grid { display: grid; grid-template-columns: repeat(auto-fit,minmax(150px,1fr)); gap: 16px; }
.stat { background: var(--surface); border: 1px solid var(--border); border-radius: var(--r-md); padding: 20px; backdrop-filter: var(--glass-thin); -webkit-backdrop-filter: var(--glass-thin); transition: border-color .2s, transform .2s; }
.stat:hover { border-color: var(--border-strong); transform: translateY(-2px); }
.stat .n { font-family: 'JetBrains Mono', monospace; font-size: 28px; font-weight: 700; font-variant-numeric: tabular-nums; }
.stat .l { color: var(--text-3); font-size: 13px; text-transform: uppercase; letter-spacing: .05em; margin-top: 6px; }

@media (prefers-reduced-motion: reduce) {
  .bg-mesh span, .ticker-track, .status-dot, .live-watch-dot,
  .hero .eyebrow, .hero h1, .hero p, .hero .ctas, .hero h1 .grad {
    animation: none;
  }
  .trade-card, .compare-card, .price-card, .signal-brief-card, .market-snapshot-card, .stat, .cta, .fade-in {
    transition: none;
  }
  .trade-card:hover, .compare-card:hover, .price-card:hover,
  .signal-brief-card:hover, .market-snapshot-card:hover, .stat:hover, .cta:hover { transform: none; }
  .fade-in { opacity: 1; transform: none; }
}
"""

@app.context_processor
def inject_shared_data():
    """Makes ticker_items and scan_status available in EVERY template
    automatically, since the ticker/status bar appears in NAV on every
    page — avoids having to remember to pass these in each route."""
    return {
        "ticker_items": load_ticker_items(),
        "scan_status": load_scan_status(),
    }


THEME_INIT = """<script>(function(){try{var t=localStorage.getItem('aiinsider_theme');if(t!=='light'&&t!=='dark'){t=window.matchMedia('(prefers-color-scheme: light)').matches?'light':'dark';}document.documentElement.setAttribute('data-theme',t);}catch(e){}})();</script>"""

NAV = """
<div class="bg-mesh"><span class="b1"></span><span class="b2"></span><span class="b3"></span></div>
<nav aria-label="Primary navigation">
  <div class="status-bar"><div class="wrap">
    {% if scan_status.available %}
      <span class="status-dot {{ '' if scan_status.fresh else 'stale' }}"></span>
      Latest signal: {{ scan_status.last_signal_date }} ({{ scan_status.days_ago }}d ago) &middot; periodic scan, not real-time
    {% else %}
      <span class="status-dot stale"></span> Scanner status not available yet
    {% endif %}
    <span class="live-watch" data-signal-monitor-status><span class="live-watch-dot"></span><span data-signal-monitor-copy>Auto-check every 30s</span></span>
  </div></div>
  {% if ticker_items %}
  <div class="ticker-bar"><div class="ticker-track">
    {% for item in ticker_items + ticker_items %}
    <span class="ticker-item">
      <span class="t-ticker">${{ item.ticker }}</span>
      <span class="t-rating {{ item.rating|replace('+','plus') }}">{{ item.rating }}</span>
      <span>{{ item.signal_date }}</span>
    </span>
    {% endfor %}
  </div></div>
  {% endif %}
  <div class="wrap nav-shell">
    <a class="brand" href="/" aria-label="AI Insider home">
      <span class="brand-mark">AI</span>
      <span>INSIDER</span>
      <span class="brand-build">INTEL // 24</span>
    </a>
    <div class="nav-links" id="nav-links-panel">
      <a href="/#performance">Dashboard</a>
      <a href="/#signals">Signals</a>
      <a href="/#market-data">Markets</a>
      <a href="/insider-trades">Insider Trades</a>
      <a href="/market-news">Market News</a>
      <a href="/congress">Congress</a>
      <a href="/track-record">Track Record</a>
      <a href="/leaderboard">Leaderboard</a>
      <a href="/how-it-works">Methodology</a>
      <a href="/#pricing">Pricing</a>
    </div>
    <div class="nav-pinned">
    <a class="nav-waitlist" href="mailto:hello@aiinsider.store?subject=Waitlist">Join waitlist</a>
    <a class="cta" href="/#signals">Start Free</a>
    <button type="button" class="theme-toggle" data-theme-toggle aria-label="Switch between light and dark theme" title="Switch theme">
      <svg class="theme-icon theme-icon-sun" viewBox="0 0 24 24" fill="none" aria-hidden="true"><circle cx="12" cy="12" r="4.3" stroke="currentColor" stroke-width="1.7"/><path d="M12 2.6v2.2M12 19.2v2.2M4.3 4.3l1.6 1.6M18.1 18.1l1.6 1.6M2.6 12h2.2M19.2 12h2.2M4.3 19.7l1.6-1.6M18.1 5.9l1.6-1.6" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/></svg>
      <svg class="theme-icon theme-icon-moon" viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M20 14.3A8.4 8.4 0 1 1 9.7 4a6.9 6.9 0 0 0 10.3 10.3Z" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round"/></svg>
    </button>
    <button type="button" class="nav-hamburger" data-nav-toggle aria-label="Open menu" aria-expanded="false" aria-controls="nav-links-panel">
      <span></span><span></span><span></span>
    </button>
    </div>
  </div>
</nav>
<div id="signal-toast-region" class="signal-toast-region" aria-live="polite" aria-atomic="true"></div>
"""

FOOTER = """
<footer><div class="wrap">
  <div class="footer-grid">
    <div class="footer-col">
      <div class="brand" style="margin-bottom:14px;"><span class="dot"></span>AI INSIDER</div>
      <p style="color:var(--muted);font-size:13.5px;max-width:220px;">Scored insider buying, straight from SEC filings.</p>
    </div>
    <div class="footer-col">
      <a href="/">Signals</a>
      <a href="/track-record">Track Record</a>
      <a href="/how-it-works">How It Works</a>
    </div>
    <div class="footer-col">
      <a href="/disclaimer">Disclaimer</a>
      <a href="/privacy">Privacy</a>
      <a href="/about">About</a>
    </div>
  </div>
  <p class="footer-note">AI INSIDER tracks publicly available SEC Form 4 filings. This is not insider trading &mdash; it's public information. Nothing on this site is investment advice. All investing involves risk. Past performance, live or backtested, does not guarantee future results.</p>
  <p style="color:var(--muted);font-size:13px;margin-top:16px;">&copy; {{ year }} AI INSIDER</p>
</div></footer>
<script>
(() => {
  const STORAGE_KEY = 'aiinsider_theme';
  const toggle = document.querySelector('[data-theme-toggle]');
  const metaThemeColor = document.querySelector('meta[name="theme-color"]');
  const themeColors = { dark: '#03070c', light: '#f4f6fa' };
  const applyThemeColorMeta = theme => {
    if (metaThemeColor) metaThemeColor.setAttribute('content', themeColors[theme] || themeColors.dark);
  };
  applyThemeColorMeta(document.documentElement.getAttribute('data-theme') === 'light' ? 'light' : 'dark');
  if (toggle) {
    toggle.addEventListener('click', () => {
      const current = document.documentElement.getAttribute('data-theme') === 'light' ? 'light' : 'dark';
      const next = current === 'light' ? 'dark' : 'light';
      document.documentElement.setAttribute('data-theme', next);
      try { localStorage.setItem(STORAGE_KEY, next); } catch (_) {}
      applyThemeColorMeta(next);
      window.dispatchEvent(new CustomEvent('aiinsider:themechange', { detail: { theme: next } }));
    });
  }
})();

(() => {
  const hamburger = document.querySelector('[data-nav-toggle]');
  const panel = document.getElementById('nav-links-panel');
  if (!hamburger || !panel) return;
  const closeMenu = () => {
    panel.classList.remove('mobile-open');
    hamburger.setAttribute('aria-expanded', 'false');
    document.body.classList.remove('nav-locked');
  };
  const openMenu = () => {
    panel.classList.add('mobile-open');
    hamburger.setAttribute('aria-expanded', 'true');
    document.body.classList.add('nav-locked');
  };
  hamburger.addEventListener('click', () => {
    if (panel.classList.contains('mobile-open')) closeMenu(); else openMenu();
  });
  panel.querySelectorAll('a').forEach(link => link.addEventListener('click', closeMenu));
  document.addEventListener('keydown', e => { if (e.key === 'Escape') closeMenu(); });
  window.addEventListener('resize', () => { if (window.innerWidth > 768) closeMenu(); });
})();

(() => {
  const tabs = document.querySelectorAll('[data-filings-tab]');
  if (!tabs.length) return;
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const target = tab.getAttribute('data-filings-tab');
      tabs.forEach(t => {
        const active = t === tab;
        t.classList.toggle('active', active);
        t.setAttribute('aria-selected', active ? 'true' : 'false');
      });
      document.querySelectorAll('[data-filings-panel]').forEach(panel => {
        panel.hidden = panel.getAttribute('data-filings-panel') !== target;
      });
    });
  });
})();

if (!window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
  // stagger cards within the same grid so they reveal one after another
  document.querySelectorAll('.trade-grid').forEach(grid => {
    Array.from(grid.children).forEach((card, i) => {
      card.style.transitionDelay = (i * 60) + 'ms';
    });
  });
  const io = new IntersectionObserver((entries) => {
    entries.forEach(e => { if (e.isIntersecting) { e.target.classList.add('in-view'); io.unobserve(e.target); } });
  }, {threshold: 0.15});
  document.querySelectorAll('.fade-in').forEach(el => io.observe(el));
} else {
  document.querySelectorAll('.fade-in').forEach(el => el.classList.add('in-view'));
}

// Public-safe on-site alerting. This polls the website's own API and never
// requests browser notification permission or exposes private portfolio data.
(() => {
  const endpoint = '/api/signals';
  const storageKey = 'aiinsider_seen_public_signals_v1';
  const region = document.getElementById('signal-toast-region');
  const statusNodes = Array.from(document.querySelectorAll('[data-signal-monitor-status]'));
  const copyNodes = Array.from(document.querySelectorAll('[data-signal-monitor-copy]'));
  const ytdLedgerBody = document.querySelector('[data-ytd-ledger-body]');
  const ytdOpenCountNodes = Array.from(document.querySelectorAll('[data-ytd-open-count]'));
  const ytdActivityAsOfNodes = Array.from(document.querySelectorAll('[data-ytd-activity-asof]'));
  let initialized = false;
  let seen = new Set();

  try {
    const saved = JSON.parse(localStorage.getItem(storageKey) || '[]');
    if (Array.isArray(saved)) seen = new Set(saved);
  } catch (_) {
    seen = new Set();
  }

  const setMonitorState = (state, copy) => {
    statusNodes.forEach(node => {
      node.classList.remove('checking', 'reconnecting');
      if (state) node.classList.add(state);
    });
    copyNodes.forEach(node => { node.textContent = copy; });
  };

  const saveSeen = () => {
    const ids = Array.from(seen).slice(-200);
    seen = new Set(ids);
    try { localStorage.setItem(storageKey, JSON.stringify(ids)); } catch (_) {}
  };

  const upsertYtdLedgerSignal = signal => {
    if (!ytdLedgerBody || !signal) return;
    const ticker = String(signal.ticker || '').trim().toUpperCase().slice(0, 12);
    const rawDate = String(signal.signal_date || signal.trade_date || '').trim();
    const dateMatch = rawDate.match(/^\\d{4}-\\d{2}-\\d{2}/);
    const date = dateMatch ? dateMatch[0] : '';
    if (!ticker || !date) return;
    const key = `bot|${date}|${ticker}`;
    const exists = Array.from(ytdLedgerBody.querySelectorAll('[data-ledger-key]'))
      .some(row => row.dataset.ledgerKey === key);
    if (exists) return;

    const row = document.createElement('tr');
    row.dataset.ledgerKey = key;
    const dateCell = document.createElement('td');
    dateCell.className = 'trade-date';
    dateCell.textContent = date;
    const tickerCell = document.createElement('td');
    tickerCell.className = 'trade-symbol';
    tickerCell.textContent = `$${ticker}`;
    const rating = String(signal.rating || '').trim().toUpperCase().slice(0, 4);
    if (rating) {
      const ratingNode = document.createElement('span');
      ratingNode.className = 'trade-rating';
      ratingNode.textContent = rating;
      tickerCell.appendChild(ratingNode);
    }
    const sourceCell = document.createElement('td');
    sourceCell.textContent = 'AI Insider bot';
    const statusCell = document.createElement('td');
    const status = document.createElement('span');
    status.className = 'ytd-status open-signal';
    status.textContent = 'OPEN SIGNAL';
    statusCell.appendChild(status);
    const returnCell = document.createElement('td');
    returnCell.className = 'ytd-trade-return';
    returnCell.textContent = '—';
    row.append(dateCell, tickerCell, sourceCell, statusCell, returnCell);
    ytdLedgerBody.prepend(row);

    ytdOpenCountNodes.forEach(node => {
      const current = Number.parseInt(node.textContent, 10);
      node.textContent = String((Number.isFinite(current) ? current : 0) + 1);
    });
    ytdActivityAsOfNodes.forEach(node => {
      if (!node.textContent.trim() || date > node.textContent.trim()) node.textContent = date;
    });
  };

  const dismissToast = toast => {
    toast.classList.remove('show');
    window.setTimeout(() => toast.remove(), 280);
  };

  const showToast = (signals) => {
    if (!region || !signals.length) return;
    region.replaceChildren();
    const signal = signals[0];
    const toast = document.createElement('div');
    toast.className = 'signal-toast';
    toast.setAttribute('role', 'status');

    const accent = document.createElement('div');
    accent.className = 'signal-toast-accent';
    const body = document.createElement('div');
    body.className = 'signal-toast-body';
    const label = document.createElement('div');
    label.className = 'signal-toast-label';
    label.textContent = signals.length > 1 ? `${signals.length} new qualified signals` : 'New qualified signal';
    const title = document.createElement('h3');
    title.textContent = `${signal.rating || 'Rated'} · $${signal.ticker || '—'}`;
    const detail = document.createElement('p');
    const buyer = signal.insider_name || 'Open-market insider purchase';
    const price = signal.entry_price ? ` near $${signal.entry_price}` : '';
    const allocation = signal.model_allocation_pct !== null && signal.model_allocation_pct !== undefined
      ? ` Model allocation: ${Number(signal.model_allocation_pct).toLocaleString()}%.`
      : '';
    detail.textContent = `${buyer}${price}.${allocation} Published research only — not a recommendation.`;
    const actions = document.createElement('div');
    actions.className = 'signal-toast-actions';
    const link = document.createElement('a');
    link.className = 'signal-toast-link';
    link.href = '/#signals';
    link.textContent = signals.length > 1 ? 'View signals' : 'View signal';
    const close = document.createElement('button');
    close.className = 'signal-toast-close';
    close.type = 'button';
    close.textContent = 'Dismiss';
    close.addEventListener('click', () => dismissToast(toast));

    actions.append(link, close);
    body.append(label, title, detail, actions);
    toast.append(accent, body);
    region.appendChild(toast);
    requestAnimationFrame(() => toast.classList.add('show'));
    window.setTimeout(() => {
      if (toast.isConnected) dismissToast(toast);
    }, 14000);
  };

  const pollSignals = async () => {
    if (document.hidden) return;
    setMonitorState('checking', 'Checking public feed');
    try {
      const response = await fetch(endpoint, { cache: 'no-store', headers: { 'Accept': 'application/json' } });
      if (!response.ok) throw new Error('signal feed unavailable');
      const payload = await response.json();
      const signals = Array.isArray(payload.signals) ? payload.signals : [];
      signals.forEach(upsertYtdLedgerSignal);
      const unseen = signals.filter(signal => signal.signal_id && !seen.has(signal.signal_id));

      if (!initialized && seen.size === 0) {
        signals.forEach(signal => { if (signal.signal_id) seen.add(signal.signal_id); });
      } else if (unseen.length) {
        showToast(unseen);
        unseen.forEach(signal => seen.add(signal.signal_id));
      }
      signals.forEach(signal => { if (signal.signal_id) seen.add(signal.signal_id); });
      saveSeen();
      initialized = true;
      setMonitorState('', 'Auto-check every 30s');
    } catch (_) {
      setMonitorState('reconnecting', 'Feed reconnecting');
    }
  };

  pollSignals();
  window.setInterval(pollSignals, 30000);
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) pollSignals();
  });
})();
</script>
"""


def format_mcap(val):
    """1234567890 -> '$1.2B', 45000000 -> '$45.0M'. Honest '—' for missing data."""
    if val is None or (isinstance(val, float) and np.isnan(val)) or val == "":
        return "—"
    try:
        v = float(val)
    except (TypeError, ValueError):
        return "—"
    if v >= 1e9:
        return f"${v/1e9:.1f}B"
    if v >= 1e6:
        return f"${v/1e6:.0f}M"
    if v > 0:
        return f"${v:,.0f}"
    return "—"


app.jinja_env.filters["mcap"] = format_mcap


def ad_slot(label="Advertisement"):
    if ADSENSE_CLIENT_ID:
        return f'<div class="ad-slot"><!-- AdSense unit, client={ADSENSE_CLIENT_ID} --></div>'
    return f'<div class="ad-slot">{label} &middot; connect AdSense once live on a real domain</div>'


PUBLIC_SIGNAL_API_FIELDS = (
    "ticker", "company", "rating", "conviction", "hold_mode", "signal_date",
    "trade_date", "entry_price", "take_profit", "target_pct", "max_hold",
    "stop_loss", "model_allocation_pct", "reason", "red_flags",
    "insider_name", "insider_title", "total_amount", "market_cap",
    "rsi14", "low_52w", "high_52w", "range_position_pct",
    "ret_20d_pct", "avg_volume_20d",
)


def _public_json_value(value):
    """Convert dataframe scalar values into strict JSON-safe values."""
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    return value


def build_public_signal_api_payload():
    """Public allowlist only; never expose account or position-sizing data."""
    payload = []
    for signal in load_public_signals():
        safe_signal = {
            field: _public_json_value(signal.get(field))
            for field in PUBLIC_SIGNAL_API_FIELDS
        }
        identity = "|".join(str(safe_signal.get(field) or "") for field in (
            "ticker", "rating", "signal_date", "entry_price",
            "insider_name", "total_amount",
        ))
        safe_signal["signal_id"] = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
        payload.append(safe_signal)
    return payload


@app.route("/api/signals")
def api_signals():
    response = jsonify({
        "signals": build_public_signal_api_payload(),
        "scan_status": load_scan_status(),
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "poll_after_seconds": 30,
    })
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response


@app.route("/api/ytd-ledger")
def api_ytd_ledger():
    """Public percentage-only ledger used to synchronize VPS data to Render."""
    snapshot = load_public_ytd_snapshot()
    ledger = load_ytd_trade_ledger(snapshot)
    if not snapshot or not ledger:
        response = jsonify({"year": None, "activity_as_of": None, "events": []})
        response.status_code = 503
    else:
        bot_events = [
            {
                "date": event["date"],
                "ticker": event["ticker"],
                "rating": event["rating"],
                "status": event["status"],
                "return_pct": event["return_pct"],
                "model_allocation_pct": event.get("model_allocation_pct"),
                "resolved_date": event.get("resolved_date") or None,
                "source": "AI_INSIDER_BOT",
            }
            for event in ledger["events"]
            if event["source"] == "AI_INSIDER_BOT"
        ]
        response = jsonify({
            "year": snapshot["year"],
            "activity_as_of": ledger["activity_as_of"],
            "events": bot_events,
        })
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response


# ── home page ────────────────────────────────────────────────────

HOME_TEMPLATE = """
<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#03070c">
""" + THEME_INIT + """
<title>AI INSIDER &mdash; Scored insider buying from SEC filings</title>
<meta name="description" content="Every open-market insider purchase, scored against two decades of backtested history. Free, public, updated daily from SEC EDGAR.">
<style>""" + BASE_CSS + """</style>
</head><body>
""" + NAV + """

{% if performance or (ytd_snapshot and ytd_ledger) %}
<section id="performance" class="performance-terminal" aria-labelledby="performance-title"><div class="wrap">
  <article class="performance-shell">
    <header class="performance-head">
      <div class="performance-heading">
        <span class="eyebrow">Strategy dashboard</span>
        <h2 id="performance-title">{% if performance %}Owner portfolio, rules backtest &amp; S&amp;P 500{% else %}2026 performance &amp; trade activity{% endif %}</h2>
        <p>{% if performance %}A single dashboard with each record labeled by what it actually measures.{% else %}The owner-reported YTD snapshot and public trade ledger remain available while comparison-chart data is being refreshed.{% endif %}</p>
      </div>
      <div class="performance-badges" aria-label="Performance dataset details">
        {% if performance and performance.trade_count %}<span class="performance-badge">{{ performance.trade_count }} backtested trades</span>{% endif %}
        {% if performance and performance.win_rate is not none %}<span class="performance-badge">{{ performance.win_rate }}% historical win rate</span>{% endif %}
        <span class="performance-badge live">{{ signals|length }} open signal{{ '' if signals|length == 1 else 's' }}</span>
      </div>
    </header>

    {% if performance %}
    <div class="performance-kpis" aria-live="polite">
      {% if ytd_snapshot %}
      <div class="performance-kpi owner">
        <span>Owner portfolio</span>
        <strong data-owner-return>{{ '+' if ytd_snapshot.ytd_return_pct >= 0 else '' }}{{ ytd_snapshot.ytd_return_pct }}%</strong>
        <small>Selected-period return &middot; reconstructed from the owner ledger</small>
      </div>
      {% endif %}
      <div class="performance-kpi">
        <span>Rules backtest</span>
        <strong data-strategy-return>&mdash;</strong>
        <small>Separate hypothetical YTD time-weighted return</small>
      </div>
      <div class="performance-kpi">
        <span>S&amp;P 500</span>
        <strong data-benchmark-return>&mdash;</strong>
        <small>SPY adjusted-close proxy</small>
      </div>
      <div class="performance-kpi">
        <span data-alpha-label>Backtest excess</span>
        <strong data-alpha-return>&mdash;</strong>
        <small data-alpha-help>Rules backtest minus benchmark</small>
      </div>
    </div>
    {% elif ytd_snapshot %}
    <div class="performance-kpis owner-only">
      <div class="performance-kpi owner">
        <span>Owner account YTD</span>
        <strong>{{ '+' if ytd_snapshot.ytd_return_pct >= 0 else '' }}{{ ytd_snapshot.ytd_return_pct }}%</strong>
        <small>Owner-reported &middot; as of {{ ytd_snapshot.as_of }}</small>
      </div>
    </div>
    {% endif %}

    <div class="performance-chart-panel{% if not performance %} ledger-only{% endif %}">
      {% if performance %}
      <div class="performance-chart-toolbar">
        <div class="performance-window-label">
          <span>Selected performance view</span>
          <strong data-performance-window>Year to date</strong>
        </div>
        <div class="performance-ranges" role="group" aria-label="Performance chart period">
          <button class="performance-range" type="button" data-range="1D" aria-pressed="false">1D</button>
          <button class="performance-range" type="button" data-range="1W" aria-pressed="false">1W</button>
          <button class="performance-range" type="button" data-range="1M" aria-pressed="false">1M</button>
          <button class="performance-range" type="button" data-range="3M" aria-pressed="false">3M</button>
          <button class="performance-range active" type="button" data-range="YTD" aria-pressed="true">YTD</button>
        </div>
      </div>
      <div class="performance-plot">
        <canvas class="performance-canvas" data-performance-canvas role="img" aria-label="Owner portfolio compared with the S&P 500 for the selected period"></canvas>
        <div class="performance-tooltip" data-performance-tooltip>
          <time data-tooltip-date></time>
          <div data-tooltip-owner-row hidden><span>Owner portfolio</span><strong class="owner-value" data-tooltip-owner></strong></div>
          <div data-tooltip-backtest-row><span>Rules backtest</span><strong class="strategy-value" data-tooltip-strategy></strong></div>
          <div data-tooltip-benchmark-row><span>S&amp;P 500</span><strong class="benchmark-value" data-tooltip-benchmark></strong></div>
        </div>
      </div>
      <div class="performance-legend" data-performance-legend aria-hidden="true">
        {% if ytd_snapshot %}<span class="owner"><i></i>Owner portfolio</span>{% endif %}
        <span data-backtest-legend><i></i>Rules backtest</span>
        <span class="benchmark"><i></i>S&amp;P 500 (SPY)</span>
      </div>
      <div class="performance-foot">
        <p class="performance-risk" data-performance-risk><strong>Portfolio comparison:</strong> owner-reported +85% snapshot and forward bot continuation versus SPY adjusted-close YTD.</p>
        <p class="performance-asof" data-performance-asof>Owner history {% if ytd_snapshot.curve_start %}{{ ytd_snapshot.curve_start }} &rarr; {% endif %}{{ ytd_snapshot.as_of }}<br>SPY curve covers full YTD</p>
      </div>
      {% endif %}

      {% if ytd_ledger and ytd_ledger.events %}
      <section class="ytd-ledger" aria-labelledby="ytd-ledger-title">
        <header class="ytd-ledger-head">
          <div>
            <h3 id="ytd-ledger-title">2026 trade activity</h3>
            <p>Percentage-only closed activity from the supplied account export, followed by every new public bot signal.</p>
          </div>
          <span class="ytd-ledger-updated">Updated through<br><strong data-ytd-activity-asof>{{ ytd_ledger.activity_as_of }}</strong></span>
        </header>
        <div class="ytd-ledger-scroll">
          <table class="ytd-ledger-table">
            <thead><tr><th>Date</th><th>Ticker</th><th>Source</th><th>Status</th><th style="text-align:right;">Return</th></tr></thead>
            <tbody data-ytd-ledger-body>
              {% for trade in ytd_ledger.events %}
              <tr data-ledger-key="{{ trade.key }}">
                <td class="trade-date">{{ trade.date }}</td>
                <td class="trade-symbol">${{ trade.ticker }}{% if trade.rating %}<span class="trade-rating">{{ trade.rating }}</span>{% endif %}</td>
                <td>{{ trade.source_label }}</td>
                <td><span class="ytd-status {{ trade.status|lower|replace(' ', '-') }}">{{ trade.status }}</span></td>
                {% if trade.return_pct is not none %}
                <td class="ytd-trade-return {{ 'pos' if trade.return_pct >= 0 else 'neg' }}">{{ '+' if trade.return_pct >= 0 else '' }}{{ trade.return_pct }}%</td>
                {% else %}
                <td class="ytd-trade-return">&mdash;</td>
                {% endif %}
              </tr>
              {% endfor %}
            </tbody>
          </table>
        </div>
        <p class="ytd-ledger-note"><strong>Important:</strong> bot signals are research alerts, not proof that a trade was executed. Open signals do not change the closed-performance figures. Not financial advice. Trade at your own risk.</p>
      </section>
      {% endif %}

      {% if ytd_snapshot %}
      <div class="performance-methodology" id="performance-methodology">
        <span class="performance-methodology-mark" aria-hidden="true"></span>
        <p><strong>How to read the chart:</strong> the lime line reconstructs the owner portfolio path from dated closed-trade results and dividends in the supplied Trading 212 export, normalized to the owner-reported {{ ytd_snapshot.ytd_return_pct }}% YTD result on {{ ytd_snapshot.as_of }}. It is not a daily brokerage equity curve. New bot signals are added as markers. Open signals do not change performance; after a signal resolves, its published model-allocation percentage multiplied by its percentage gain or loss updates the continuation. Signals without a published allocation remain markers only. The blue line is SPY. Not financial advice. Trade at your own risk.</p>
      </div>
      {% endif %}
    </div>
    {% if performance %}
    <script type="application/json" id="performance-data">{{ performance|tojson }}</script>
    {% endif %}
  </article>
</div></section>
{% endif %}

<div class="hero home-hero"><div class="wrap">
  <div class="hero-layout">
    <div class="hero-copy">
      <span class="eyebrow"><span class="eyebrow-rule"></span>Independent research on public SEC filings</span>
      <h1>Thousands of insider filings. The signals that <span class="grad">actually mattered</span>.</h1>
      <p>AI Insider screens every open-market purchase by executives and directors against a rules-based model backtested across 20+ years of Form 4 outcomes &mdash; then explains, in plain English, why each signal cleared the bar.</p>
      <div class="ctas">
        <a class="cta" href="#signals">Start free &mdash; view signals</a>
        <a class="cta ghost" href="/how-it-works">Read the methodology &rarr;</a>
      </div>
      <div class="hero-note" aria-label="Research principles">
        <span>Open-market buys only</span>
        <span>Periodic SEC scan</span>
        <span>Research, not advice</span>
      </div>
    </div>
    <aside class="signal-monitor" aria-label="Latest qualified signals">
      <div class="monitor-head">
        <div><span class="monitor-kicker">Latest qualified signals</span><strong>Signal feed</strong></div>
        <span class="monitor-live">{{ 'Updated' if scan_status.available else 'Awaiting scan' }}</span>
      </div>
      {% if signals %}
        {% for s in signals[:3] %}
        <div class="monitor-row">
          <div class="monitor-company"><strong>${{ s.ticker }}</strong><span>{{ s.insider_name or 'Open-market purchase' }}</span></div>
          <div class="monitor-price"><strong>${{ s.entry_price }}</strong><span>signal price</span></div>
          <span class="monitor-rating {{ s.rating|replace('+','plus') }}">{{ s.rating }}</span>
        </div>
        {% endfor %}
      {% else %}
        <div class="monitor-empty">No new filing cleared the signal threshold in the latest scan. That is a normal outcome, not missing data.</div>
      {% endif %}
      <div class="monitor-foot"><span>Screening factors</span><strong>Buyer seniority &middot; ownership change &middot; purchase size &middot; price setup</strong></div>
    </aside>
  </div>
</div></div>

{% if performance and performance.trade_count %}
<div class="trust-bar"><div class="wrap" style="display:flex;justify-content:center;gap:48px;flex-wrap:wrap;">
  <div class="trust-stat"><div class="n">20+</div><div class="l">Years SEC Data</div></div>
  <div class="trust-stat"><div class="n">{{ performance.trade_count }}</div><div class="l">Backtested Trades</div></div>
  {% if performance.win_rate is not none %}<div class="trust-stat"><div class="n">{{ performance.win_rate }}%</div><div class="l">Historical Win Rate</div></div>{% endif %}
  <div class="trust-stat"><div class="n">AI</div><div class="l">Ranked Insider Signals</div></div>
</div></div>
{% elif trust_stats %}
<div class="trust-bar"><div class="wrap" style="display:flex;justify-content:center;gap:48px;flex-wrap:wrap;">
  <div class="trust-stat"><div class="n">{{ trust_stats.total_filings }}</div><div class="l">Filings Logged</div></div>
  <div class="trust-stat"><div class="n">{{ trust_stats.unique_tickers }}</div><div class="l">Tickers Tracked</div></div>
  <div class="trust-stat"><div class="n">{{ trust_stats.unique_insiders }}</div><div class="l">Insiders Followed</div></div>
  <div class="trust-stat"><div class="n">{{ trust_stats.days_tracked }}</div><div class="l">Days Tracked</div></div>
</div></div>
{% endif %}

{% if market_trades %}
<section id="market-activity" class="market-activity-banner"><div class="wrap">
  <div class="market-activity-head">
    <span class="eyebrow">Unfiltered &middot; market-wide &middot; not our signals</span>
    <h2>Every real insider trade, not just the ones we flag</h2>
    <p>Every open-market Form 4 buy or sell filed across the entire market, pulled straight from SEC EDGAR &mdash; raw and unscored. This is the noise AI Insider's model screens down to the signals above. <a href="/insider-trades" style="color:var(--amber);">View the full feed &rarr;</a></p>
  </div>
  <div class="market-activity-scroll"><div class="market-activity-track">
    {% for t in market_trades + market_trades %}
    <span class="market-activity-item">
      <span class="ma-ticker">${{ t.ticker }}</span>
      <span class="ma-type {{ 'sell' if t.transaction_type == 'Open-market sell' else 'buy' }}">{{ t.transaction_type }}</span>
      <span class="ma-insider">{{ t.insider_name or 'Insider' }}{% if t.insider_title %} &middot; {{ t.insider_title }}{% endif %}</span>
      {% if t.value %}<span class="ma-value">${{ '{:,.0f}'.format(t.value) }}</span>{% endif %}
      <span class="ma-date">{{ t.transaction_date }}</span>
    </span>
    {% endfor %}
  </div></div>
</div></section>
{% endif %}

{% macro filings_rows(rows) %}
  {% for f in rows %}
  <tr>
    <td class="tk">${{ f.ticker }}</td>
    <td>{{ f.insider_name or '—' }}{% if f.insider_title %}<div class="role">{{ f.insider_title }}</div>{% endif %}</td>
    <td>{{ f.signal_date }}</td>
    <td><span class="trade-badge {{ f.rating|replace('+','plus') }}">{{ f.rating }}</span></td>
    <td>${{ f.entry_price }}</td>
    <td>{{ f.market_cap|mcap }}</td>
    <td class="amt">{{ '${:,.0f}'.format(f.total_amount) if f.total_amount else '—' }}</td>
  </tr>
  {% endfor %}
{% endmacro %}
<section id="filings"><div class="wrap">
  <div class="section-head">
    <span class="eyebrow">01 / Signal ledger</span>
    <h2>Insider filings</h2>
    <p>A dense, timestamped view of every logged signal. Public SEC Form 4 data, most recent first.{% if trust_stats and performance %} {{ trust_stats.total_filings }} filings logged across {{ trust_stats.unique_tickers }} tickers and {{ trust_stats.unique_insiders }} insiders over {{ trust_stats.days_tracked }} days tracked.{% endif %}</p>
  </div>
  <div class="filings-tabs" role="tablist">
    <button type="button" class="filings-tab active" data-filings-tab="week" role="tab" aria-selected="true">This Week</button>
    <button type="button" class="filings-tab" data-filings-tab="all" role="tab" aria-selected="false">All Time</button>
  </div>
  <div class="filings-panel" data-filings-panel="week">
    {% if weekly_signals %}
    <div class="filings-table-wrap">
      <table class="filings">
        <thead><tr>
          <th>Ticker</th><th>Insider</th><th>Filed</th><th>Rating</th><th>Price</th><th>Mkt Cap</th><th class="amt">Amount</th>
        </tr></thead>
        <tbody>{{ filings_rows(weekly_signals) }}</tbody>
      </table>
    </div>
    {% else %}
    <div class="empty">No new signals filed since {{ weekly_since }} &mdash; the model didn't find anything that cleared the bar this week. Shown honestly empty rather than backfilled with older rows &mdash; check the All Time tab for full history.</div>
    {% endif %}
  </div>
  <div class="filings-panel" data-filings-panel="all" hidden>
    {% if filings %}
    <div class="filings-table-wrap">
      <table class="filings">
        <thead><tr>
          <th>Ticker</th><th>Insider</th><th>Filed</th><th>Rating</th><th>Price</th><th>Mkt Cap</th><th class="amt">Amount</th>
        </tr></thead>
        <tbody>{{ filings_rows(filings) }}</tbody>
      </table>
    </div>
    {% else %}
    <div class="empty">No filings logged yet — this table fills in as daily scans run.</div>
    {% endif %}
  </div>
</div></section>

<section id="market-data"><div class="wrap">
  <div class="section-head">
    <span class="eyebrow">02 / Market context</span>
    <h2>Price action around every signal</h2>
    <p>Explore the chart, then compare it with the market conditions captured by the bot during its latest scan.</p>
  </div>
  <form class="ticker-search" action="/#market-data" method="get" role="search" aria-label="Search a stock ticker">
    <svg class="ticker-search-icon" viewBox="0 0 24 24" fill="none" aria-hidden="true"><circle cx="11" cy="11" r="7" stroke="currentColor" stroke-width="1.8"/><path d="M20 20l-3.8-3.8" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>
    <input type="text" name="symbol" placeholder="Search any ticker &mdash; AAPL, TSLA, NVDA..." value="{{ chart_symbol or '' }}" maxlength="6" autocomplete="off" spellcheck="false" autocapitalize="characters" aria-label="Ticker symbol">
    <button type="submit">Search</button>
  </form>
  {% if requested_symbol and not chart_symbol %}
  <p class="ticker-search-error">"{{ requested_symbol }}" doesn't look like a valid ticker &mdash; showing the default chart instead.</p>
  {% endif %}
  <div class="ticker-chips">
    {% for t in ['SPY', 'AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL'] %}
    <a href="/?symbol={{ t }}#market-data" class="ticker-chip{{ ' active' if chart_symbol == t else '' }}">{{ t }}</a>
    {% endfor %}
  </div>
  <div class="market-dashboard">
    <div class="market-chart-panel">
      <div class="market-panel-head">
        <div><span>Interactive chart</span><strong>Daily price and volume</strong></div>
        <div class="market-delay-note">TradingView exchange data may be delayed. Display only, not for execution.</div>
      </div>
      <div class="tradingview-widget-container">
        <div class="tradingview-widget-container__widget"></div>
        <div class="tradingview-widget-copyright"><a href="https://www.tradingview.com/markets/stocks-usa/" rel="noopener nofollow" target="_blank">US market data</a> by TradingView</div>
        <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
        {
          "allow_symbol_change": true,
          "calendar": false,
          "details": false,
          "hide_side_toolbar": true,
          "hide_top_toolbar": false,
          "hide_legend": false,
          "hide_volume": false,
          "hotlist": false,
          "interval": "D",
          "locale": "en",
          "save_image": false,
          "style": "1",
          "symbol": "{{ chart_symbol or (signals[0].ticker if signals else 'AMEX:SPY') }}",
          "theme": "dark",
          "timezone": "Etc/UTC",
          "backgroundColor": "rgba(8, 12, 16, 0)",
          "gridColor": "rgba(255, 255, 255, 0.05)",
          "withdateranges": true,
          "width": "100%",
          "height": 820
        }
        </script>
      </div>
    </div>
    <div class="market-snapshot-list">
      {% if signals %}
        {% for s in signals[:4] %}
        <article class="market-snapshot-card">
          <div class="market-snapshot-head">
            <div class="market-snapshot-symbol"><strong>${{ s.ticker }}</strong><span>{{ s.company or 'Company name unavailable' }}</span></div>
            <span class="trade-badge {{ s.rating|replace('+','plus') }}">{{ s.rating }}</span>
          </div>
          <div class="market-snapshot-price">
            <span class="market-data-label">Price at scan</span>
            <strong>${{ s.entry_price }}</strong>
            {% if s.ret_20d_pct is not none %}<span class="market-change {{ 'pos' if s.ret_20d_pct >= 0 else 'neg' }}">{{ '%+.1f'|format(s.ret_20d_pct) }}% / 20D</span>{% endif %}
          </div>
          <div class="market-metrics">
            <div class="market-metric"><span>Market cap</span><strong>{{ s.market_cap|mcap }}</strong></div>
            <div class="market-metric"><span>RSI 14</span><strong>{{ '%.1f'|format(s.rsi14) if s.rsi14 is not none else '—' }}</strong></div>
            <div class="market-metric"><span>Avg volume</span><strong>{{ '{:,.0f}'.format(s.avg_volume_20d) if s.avg_volume_20d is not none else '—' }}</strong></div>
          </div>
          {% if s.range_position_pct is not none %}
          <div class="market-range" style="--range-position: {{ s.range_position_pct }}%;">
            <div class="market-range-head"><span>52-week range</span><strong>{{ '%.0f'|format(s.range_position_pct) }}% from low</strong></div>
            <div class="market-range-track"><span class="market-range-fill"></span><span class="market-range-pin"></span></div>
            <div class="market-range-labels"><span>${{ '%.2f'|format(s.low_52w) }}</span><span>${{ '%.2f'|format(s.high_52w) }}</span></div>
          </div>
          {% endif %}
          <p class="market-source-note">Snapshot from the latest bot scan, not a streaming quote.</p>
        </article>
        {% endfor %}
      {% else %}
        <div class="empty">Market snapshots will appear when the next signal clears the scanner.</div>
      {% endif %}
    </div>
  </div>
</div></section>

<section id="proof"><div class="wrap">
  <div class="section-head">
    <span class="eyebrow">03 / Historical proof</span>
    <h2>What these signals have caught before</h2>
    <p>Resolved examples from our historical simulation &mdash; labeled honestly as backtest, not live results. Dollar amounts are a $10,000 hypothetical position, for illustration only.</p>
  </div>
  {% if examples %}
  <div class="trade-grid">
    {% for e in examples %}
    <div class="trade-card fade-in">
      <div class="trade-top">
        <div class="trade-avatar">{{ e.ticker[:2] }}</div>
        <div class="trade-gain {{ 'pos' if e.pnl >= 0 else 'neg' }}">{{ '+' if e.pnl >= 0 else '' }}{{ e.pnl }}%</div>
      </div>
      <div class="trade-ticker">${{ e.ticker }}</div>
      <div class="trade-meta">Entered {{ e.entry_date }} &middot; held {{ e.days_held }} days</div>
      {% if e.pnl >= 0 %}
      <div style="color:var(--accent);font-weight:700;font-size:14px;margin-top:8px;">A $10,000 position would have made ${{ "{:,.0f}".format(10000 * e.pnl / 100) }}</div>
      {% endif %}
      <div class="trade-narrative">A {{ e.rating }}-rated signal on ${{ e.ticker }} &mdash; in the backtest, holding from entry to exit returned {{ e.pnl }}%.</div>
      <span class="trade-badge {{ e.rating|replace('+','plus') }}">{{ e.rating }} rated</span>
    </div>
    {% endfor %}
  </div>
  {% else %}
  <div class="empty">Backtest examples not published yet.</div>
  {% endif %}
</div></section>

<section id="signals"><div class="wrap">
  <div class="section-head">
    <span class="eyebrow">04 / Current output</span>
    <h2>Today's scored signals</h2>
    <p>Each qualifying filing with its conviction, model framework, full thesis and the risks that could break it.</p>
  </div>
  {% if signals %}
  {% set latest = signals[0] %}
  <article class="new-trade-alert fade-in" aria-labelledby="new-trade-title">
    <div class="new-trade-alert-head">
      <div>
        <span class="new-trade-kicker">New trade signal</span>
        <h3 id="new-trade-title">A new model setup has qualified</h3>
        <p>Published {{ latest.signal_date or latest.trade_date or 'after the latest scan' }} &middot; Review the complete thesis before acting.</p>
      </div>
      <div class="new-trade-alert-badges">
        {% if latest.conviction %}<span class="conviction-pill">{{ latest.conviction }} conviction</span>{% endif %}
        <span class="trade-badge {{ latest.rating|replace('+','plus') }}">{{ latest.rating }} rated</span>
      </div>
    </div>
    <div class="new-trade-alert-main">
      <div class="new-trade-symbol">
        <span>Ticker</span>
        <strong>${{ latest.ticker }}</strong>
        <small>{{ latest.company or 'Company name unavailable' }}</small>
      </div>
      <div class="new-trade-stats" aria-label="New trade setup">
        <div class="new-trade-stat"><span>Entry</span><strong>${{ latest.entry_price }}</strong><small>Reference price at scan</small></div>
        <div class="new-trade-stat"><span>Target</span><strong>{{ '$%.2f'|format(latest.take_profit) if latest.take_profit is not none else 'Price action' }}</strong><small>{{ '%+.1f'|format(latest.target_pct) + '%' if latest.target_pct is not none else 'No fixed target' }}</small></div>
        <div class="new-trade-stat"><span>Maximum hold</span><strong>{{ '%.0f'|format(latest.max_hold) + ' days' if latest.max_hold is not none else 'Open' }}</strong><small>{{ latest.hold_mode|title if latest.hold_mode else 'Model review window' }}</small></div>
        <div class="new-trade-stat allocation"><span>Model allocation</span><strong>{{ '%.2f'|format(latest.model_allocation_pct)|replace('.00', '') + '%' if latest.model_allocation_pct is not none else 'Not set' }}</strong><small>Percentage of model portfolio</small></div>
      </div>
    </div>
    <div class="new-trade-copy">
      <div class="new-trade-copy-block">
        <span class="new-trade-copy-label">Why</span>
        <p>{{ latest.reason or 'The scanner qualified this filing, but a written thesis is not available for this record.' }}</p>
      </div>
      <div class="new-trade-copy-block watch">
        <span class="new-trade-copy-label">Watch</span>
        <p>{{ latest.red_flags or 'Review the SEC filing, company disclosures and current market conditions independently.' }}</p>
      </div>
    </div>
    <div class="new-trade-risk" role="note" aria-label="Trading risk warning">
      <span class="new-trade-risk-icon" aria-hidden="true">!</span>
      <div>
        <strong>Trade at your own risk.</strong>
        <p>This is research generated from public information, not financial advice or a guarantee of profit. Verify the filing and decide whether the risk is appropriate for you.</p>
      </div>
    </div>
  </article>
  <div class="signal-list-label">Full research cards</div>
  <div class="signal-brief-grid">
    {% for s in signals %}
    <article class="trade-card signal-brief-card fade-in">
      <div class="signal-brief-head">
        <div class="signal-brief-symbol">
          <div class="trade-avatar">{{ s.ticker[:2]|upper }}</div>
          <div><div class="trade-ticker">${{ s.ticker }}</div><div class="signal-company">{{ s.company or 'Company name unavailable' }}</div></div>
        </div>
        <div class="signal-brief-badges">
          {% if s.conviction %}<span class="conviction-pill">{{ s.conviction }} conviction</span>{% endif %}
          <span class="trade-badge {{ s.rating|replace('+','plus') }}">{{ s.rating }} rated</span>
        </div>
      </div>
      <div class="signal-framework" aria-label="Signal framework">
        <div class="signal-framework-cell"><span>Entry reference</span><strong>${{ s.entry_price }}</strong><small>Price at scan</small></div>
        <div class="signal-framework-cell"><span>Model target</span><strong>{{ '$%.2f'|format(s.take_profit) if s.take_profit is not none else 'Not set' }}</strong><small>{{ '%+.1f'|format(s.target_pct) + '%' if s.target_pct is not none else 'No target' }}</small></div>
        <div class="signal-framework-cell"><span>Risk / reward</span><strong>{{ s.risk_reward|string + ':1' if s.risk_reward is not none else 'Not set' }}</strong><small>Target vs. stop</small></div>
        <div class="signal-framework-cell"><span>Research window</span><strong>{{ '%.0f'|format(s.max_hold) + ' days' if s.max_hold is not none else 'Open' }}</strong><small>{{ s.hold_mode|title if s.hold_mode else 'Signal review' }}</small></div>
        <div class="signal-framework-cell"><span>Exit risk rule</span><strong>{{ '$%.2f'|format(s.stop_loss) if s.stop_loss is not none else 'No fixed stop' }}</strong><small>Model parameter</small></div>
        <div class="signal-framework-cell"><span>Model allocation</span><strong>{{ '%.2f'|format(s.model_allocation_pct)|replace('.00', '') + '%' if s.model_allocation_pct is not none else 'Not set' }}</strong><small>Model portfolio</small></div>
      </div>
      <div class="signal-brief-body">
        {% if s.insider_name %}<div class="signal-insider-line">{{ s.insider_name }}{% if s.insider_title %} &middot; {{ s.insider_title }}{% endif %}{% if s.total_amount %} &middot; ${{ "{:,.0f}".format(s.total_amount) }} purchase{% endif %}{% if s.trade_date %} &middot; {{ s.trade_date }}{% endif %}</div>{% endif %}
        {% if rating_performance and s.rating in rating_performance %}
        <div class="signal-track-line">Historically, {{ s.rating }}-rated signals: <strong>{{ rating_performance[s.rating].win_rate }}% win rate</strong>{% if rating_performance[s.rating].avg_return is not none %}, <strong>{{ '%+.1f'|format(rating_performance[s.rating].avg_return) }}% avg return</strong>{% endif %} <span>({{ rating_performance[s.rating].n }} resolved)</span></div>
        {% endif %}
        <div class="signal-thesis-block">
          <div class="signal-thesis-label">Why it qualified</div>
          <p class="signal-thesis-text">{{ s.reason or 'The scanner qualified this filing, but a written thesis is not available for this record.' }}</p>
        </div>
        <div class="signal-thesis-block signal-watch">
          <div class="signal-thesis-label">Watch before acting</div>
          <p class="signal-thesis-text">{{ s.red_flags or 'No separate red-flag note was recorded. Review the filing and current company disclosures independently.' }}</p>
        </div>
        <div class="mkt-strip">
        <div class="mkt-cell"><span class="l">Price</span><span class="v">${{ s.entry_price }}</span></div>
        <div class="mkt-cell"><span class="l">Mkt Cap</span><span class="v">{{ s.market_cap|mcap }}</span></div>
        <div class="mkt-cell"><span class="l">RSI (14)</span><span class="v {{ 'hi' if s.rsi14 and s.rsi14 >= 70 else ('lo' if s.rsi14 and s.rsi14 <= 30 else '') }}">{{ s.rsi14 if s.rsi14 else '—' }}</span></div>
        <div class="mkt-cell"><span class="l">52W Low</span><span class="v">{{ '$%.2f'|format(s.low_52w) if s.low_52w else '—' }}</span></div>
        <div class="mkt-cell"><span class="l">52W High</span><span class="v">{{ '$%.2f'|format(s.high_52w) if s.high_52w else '—' }}</span></div>
        <div class="mkt-cell"><span class="l">20D Chg</span><span class="v {{ 'hi' if s.ret_20d_pct and s.ret_20d_pct < 0 else ('lo' if s.ret_20d_pct and s.ret_20d_pct > 0 else '') }}">{{ '%+.1f'|format(s.ret_20d_pct) + '%' if s.ret_20d_pct is not none else '—' }}</span></div>
        </div>
        <p class="signal-sizing-note"><strong>Model allocation:</strong> a research-framework percentage, not a personalized amount. Private account size, dollar allocation and share count are never published.</p>
      </div>
    </article>
    {% endfor %}
  </div>
  {% else %}
  <div class="empty">No new qualifying filings today &mdash; insider buying is naturally uneven. Check the track record while you wait.</div>
  {% endif %}
  """ + ad_slot() + """
</div></section>

<section><div class="wrap">
  <div class="section-head">
    <span class="eyebrow">Why this exists</span>
    <h2>Filings vs. filtered signal</h2>
    <p>Raw filings are free everywhere. The scoring is the actual work.</p>
  </div>
  <div class="compare-grid">
    <div class="compare-card">
      <h3>SEC EDGAR &amp; OpenInsider</h3>
      <ul>
        <li>Every Form 4, unfiltered, in filing order</li>
        <li>Thousands of rows a day &mdash; most are noise</li>
        <li>No context on which purchases historically mattered</li>
      </ul>
    </div>
    <div class="compare-card highlight">
      <h3>AI INSIDER</h3>
      <ul>
        <li>Filtered against a 20+ year backtest of scoring rules</li>
        <li>C-suite involvement, ownership increase, price setup all weighed</li>
        <li>Plain-English reasoning on every signal that qualifies</li>
      </ul>
    </div>
  </div>
</div></section>

<section id="pricing"><div class="wrap">
  <div class="section-head">
    <span class="eyebrow">Roadmap</span>
    <h2>Free today. More coming.</h2>
    <p>The signal feed stays free. A paid tier for real-time alerts and full history is planned &mdash; not live yet.</p>
  </div>
  <div class="pricing-grid">
    <div class="price-card">
      <h3>Free</h3>
      <div class="desc">What you're looking at right now</div>
      <div class="price">$0</div>
      <ul>
        <li>Daily scored signals</li>
        <li>Public track record</li>
        <li>Methodology &amp; backtest results</li>
      </ul>
      <span class="coming-soon-tag">Active</span>
    </div>
    <div class="price-card rec">
      <span class="rec-tag">Planned</span>
      <h3>Alerts</h3>
      <div class="desc">The moment a signal fires</div>
      <div class="price" style="font-size:20px;"><a href="mailto:hello@aiinsider.store?subject=Waitlist" style="color:var(--accent);text-decoration:none;">Join waitlist &rarr;</a></div>
      <ul>
        <li>Real-time delivery on qualifying filings</li>
        <li>Full historical signal archive</li>
        <li>Priority support</li>
      </ul>
      <span class="coming-soon-tag">Coming soon</span>
    </div>
  </div>
</div></section>

<section><div class="wrap">
  <div class="section-head">
    <span class="eyebrow">FAQ</span>
    <h2>Common questions</h2>
  </div>
  <div style="max-width:680px;margin:0 auto;">
    <div class="faq-item">
      <h4>Where does the data come from?</h4>
      <p>Every signal starts with an SEC Form 4 &mdash; the disclosure insiders are legally required to file within two business days of trading their own company's stock. All public record.</p>
    </div>
    <div class="faq-item">
      <h4>Is following insider trades legal?</h4>
      <p>Yes. Insiders disclose these trades publicly by law. Reading a public filing and deciding what to do with that information is completely legal. Trading on private, non-public information is the illegal kind &mdash; that's not what this is.</p>
    </div>
    <div class="faq-item">
      <h4>What's the difference between this and OpenInsider?</h4>
      <p>OpenInsider shows you every filing, unfiltered. We run each purchase through rules validated on two decades of historical data and only surface the ones that pass.</p>
    </div>
    <div class="faq-item">
      <h4>Is this investment advice?</h4>
      <p>No. This is published research on public filings, for informational purposes. See the full <a href="/disclaimer" style="color:var(--accent);">disclaimer</a>.</p>
    </div>
  </div>
</div></section>

<script>
(() => {
  const source = document.getElementById('performance-data');
  const canvas = document.querySelector('[data-performance-canvas]');
  if (!source || !canvas) return;

  let payload;
  try { payload = JSON.parse(source.textContent); } catch (_) { return; }
  const allRows = (payload.points || []).map(point => ({
    date: new Date(point[0] + 'T00:00:00Z'),
    dateLabel: point[0],
    strategy: Number(point[1]),
    benchmark: Number(point[2])
  })).filter(row => Number.isFinite(row.strategy) && Number.isFinite(row.benchmark));
  if (allRows.length < 2) return;
  const ownerSnapshotRaw = payload.owner_snapshot || {};
  const ownerSnapshotReturn = Number(ownerSnapshotRaw.return_pct);
  const ownerCurve = (payload.owner_curve || []).map(point => ({
    date: new Date(String(point.date || '') + 'T00:00:00Z'),
    dateLabel: String(point.date || ''),
    returnPct: Number(point.return_pct),
    ticker: String(point.ticker || ''),
    status: String(point.status || ''),
    allocationPct: point.allocation_pct === null || point.allocation_pct === undefined ? null : Number(point.allocation_pct),
    signalReturnPct: point.signal_return_pct === null || point.signal_return_pct === undefined ? null : Number(point.signal_return_pct)
  })).filter(point => Number.isFinite(point.date.getTime()) && Number.isFinite(point.returnPct));
  const benchmarkYtdRows = (payload.benchmark_ytd_points || []).map(point => ({
    date: new Date(String(point[0] || '') + 'T00:00:00Z'),
    dateLabel: String(point[0] || ''),
    returnPct: Number(point[1])
  })).filter(point => Number.isFinite(point.date.getTime()) && Number.isFinite(point.returnPct));
  const benchmarkYtdLatest = benchmarkYtdRows.length ? benchmarkYtdRows[benchmarkYtdRows.length - 1] : null;
  const latestPointAtOrBefore = (rows, date) => {
    let selected = null;
    for (const row of rows) {
      if (row.date.getTime() <= date.getTime()) selected = row;
      else break;
    }
    return selected;
  };
  const comparisonDates = new Set(benchmarkYtdRows.map(row => row.dateLabel));
  if (benchmarkYtdLatest) {
    ownerCurve.forEach(point => {
      if (point.date.getTime() <= benchmarkYtdLatest.date.getTime()) comparisonDates.add(point.dateLabel);
    });
  }
  const ownerBenchmarkRows = Array.from(comparisonDates).sort().map(dateLabel => {
    const date = new Date(dateLabel + 'T00:00:00Z');
    const ownerPoint = latestPointAtOrBefore(ownerCurve, date);
    const benchmarkPoint = latestPointAtOrBefore(benchmarkYtdRows, date);
    if (!ownerPoint || !benchmarkPoint) return null;
    return {
      date,
      dateLabel,
      ownerCumulative: ownerPoint.returnPct,
      benchmarkCumulative: benchmarkPoint.returnPct,
      ownerEvent: ownerPoint.dateLabel === dateLabel
    };
  }).filter(Boolean);

  const CHART_COLORS = {
    dark: {
      axisLabel: '#66727b', xLabel: '#647078',
      gridMajor: 'rgba(255,255,255,.13)', gridMinor: 'rgba(255,255,255,.055)',
      backtestFillFrom: 'rgba(0, 236, 159, .16)', backtestFillTo: 'rgba(0, 236, 159, 0)',
      ownerFillFrom: 'rgba(215, 255, 125, .16)', ownerFillTo: 'rgba(215, 255, 125, 0)',
      benchmark: '#4ba3ff', strategy: '#00ec9f', owner: '#d7ff7d', benchmarkLabel: '#75baff',
      crosshair: 'rgba(255,255,255,.18)', hoverDotFill: '#071014',
    },
    light: {
      axisLabel: '#5b6472', xLabel: '#5b6472',
      gridMajor: 'rgba(15,23,42,.14)', gridMinor: 'rgba(15,23,42,.06)',
      backtestFillFrom: 'rgba(6, 122, 86, .14)', backtestFillTo: 'rgba(6, 122, 86, 0)',
      ownerFillFrom: 'rgba(122, 122, 16, .14)', ownerFillTo: 'rgba(122, 122, 16, 0)',
      benchmark: '#1c64d1', strategy: '#067a56', owner: '#7a7a10', benchmarkLabel: '#1c64d1',
      crosshair: 'rgba(15,23,42,.22)', hoverDotFill: '#ffffff',
    },
  };
  const chartTheme = () => (document.documentElement.getAttribute('data-theme') === 'light' ? CHART_COLORS.light : CHART_COLORS.dark);

  const ctx = canvas.getContext('2d');
  const tooltip = document.querySelector('[data-performance-tooltip]');
  const tooltipDate = document.querySelector('[data-tooltip-date]');
  const tooltipOwner = document.querySelector('[data-tooltip-owner]');
  const tooltipOwnerRow = document.querySelector('[data-tooltip-owner-row]');
  const tooltipBacktestRow = document.querySelector('[data-tooltip-backtest-row]');
  const tooltipBenchmarkRow = document.querySelector('[data-tooltip-benchmark-row]');
  const tooltipStrategy = document.querySelector('[data-tooltip-strategy]');
  const tooltipBenchmark = document.querySelector('[data-tooltip-benchmark]');
  const strategyMetric = document.querySelector('[data-strategy-return]');
  const benchmarkMetric = document.querySelector('[data-benchmark-return]');
  const alphaMetric = document.querySelector('[data-alpha-return]');
  const alphaLabel = document.querySelector('[data-alpha-label]');
  const alphaHelp = document.querySelector('[data-alpha-help]');
  const ownerMetric = document.querySelector('[data-owner-return]');
  const windowLabel = document.querySelector('[data-performance-window]');
  const performanceLegend = document.querySelector('[data-performance-legend]');
  const performanceRisk = document.querySelector('[data-performance-risk]');
  const performanceAsOf = document.querySelector('[data-performance-asof]');
  const buttons = Array.from(document.querySelectorAll('[data-range]'));
  const dateFormatter = new Intl.DateTimeFormat('en-US', { month: 'short', day: 'numeric', year: 'numeric', timeZone: 'UTC' });
  const rangeNames = { '1D': 'One trading day', '1W': 'One week', '1M': 'One month', '3M': 'Three months', 'YTD': 'Year to date' };
  let activeRange = 'YTD';
  let visibleRows = [];
  let hoverIndex = null;

  const signedPercent = value => {
    const decimals = Math.abs(value) >= 100 ? 1 : 2;
    return `${value >= 0 ? '+' : ''}${value.toFixed(decimals)}%`;
  };
  const axisPercent = value => {
    const abs = Math.abs(value);
    if (abs >= 1000) return `${(value / 1000).toFixed(abs >= 10000 ? 0 : 1)}k%`;
    if (abs >= 100) return `${value.toFixed(0)}%`;
    return `${value.toFixed(abs < 10 ? 1 : 0)}%`;
  };
  const paintMetric = (node, value) => {
    node.textContent = signedPercent(value);
    node.classList.toggle('pos', value >= 0);
    node.classList.toggle('neg', value < 0);
  };

  const rangeStartIndex = (rows, range) => {
    if (range === '1D') return Math.max(0, rows.length - 2);
    const end = rows[rows.length - 1].date;
    let cutoff;
    if (range === 'YTD') {
      cutoff = new Date(Date.UTC(end.getUTCFullYear(), 0, 1));
    } else {
      const days = { '1W': 7, '1M': 30, '3M': 90 }[range] || 0;
      cutoff = new Date(end.getTime() - days * 86400000);
    }
    let index = 0;
    for (let i = 0; i < rows.length; i += 1) {
      if (rows[i].date <= cutoff) index = i;
      else break;
    }
    return index;
  };

  const selectBacktestRows = range => {
    const sourceRows = allRows.slice(rangeStartIndex(allRows, range));
    const strategyBase = sourceRows[0].strategy;
    const benchmarkBase = sourceRows[0].benchmark;
    return sourceRows.map(row => ({
      ...row,
      strategyReturn: (row.strategy / strategyBase - 1) * 100,
      benchmarkReturn: (row.benchmark / benchmarkBase - 1) * 100,
      ownerComparison: false
    }));
  };

  const selectRows = range => {
    if (ownerBenchmarkRows.length >= 2) {
      const sourceRows = ownerBenchmarkRows.slice(rangeStartIndex(ownerBenchmarkRows, range));
      const ownerBase = 1 + sourceRows[0].ownerCumulative / 100;
      const benchmarkBase = 1 + sourceRows[0].benchmarkCumulative / 100;
      return sourceRows.map(row => ({
        ...row,
        strategyReturn: ((1 + row.ownerCumulative / 100) / ownerBase - 1) * 100,
        benchmarkReturn: ((1 + row.benchmarkCumulative / 100) / benchmarkBase - 1) * 100,
        ownerComparison: true
      }));
    }
    return selectBacktestRows(range);
  };
  const backtestYtdRows = selectBacktestRows('YTD');
  const backtestYtdReturn = backtestYtdRows.length
    ? backtestYtdRows[backtestYtdRows.length - 1].strategyReturn
    : null;

  const drawSeries = (rows, xFor, yFor, color, width, dash = []) => {
    ctx.save();
    ctx.beginPath();
    rows.forEach((row, index) => {
      const x = xFor(index);
      const y = yFor(row);
      if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.setLineDash(dash);
    ctx.stroke();
    ctx.restore();
  };

  const drawChart = () => {
    if (!visibleRows.length) return;
    const theme = chartTheme();
    const width = Math.max(320, canvas.clientWidth);
    const height = Math.max(230, canvas.clientHeight);
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    const padding = { top: 20, right: 17, bottom: 34, left: width < 520 ? 48 : 58 };
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;
    const ownerMode = visibleRows[0]?.ownerComparison === true;
    const values = visibleRows.flatMap(row => [row.strategyReturn, row.benchmarkReturn]);
    let minValue = Math.min(0, ...values);
    let maxValue = Math.max(0, ...values);
    const comparisonSpread = Math.max(1, maxValue - minValue);
    minValue -= comparisonSpread * .12;
    maxValue += comparisonSpread * .12;
    const spread = Math.max(1, maxValue - minValue);
    const startTime = visibleRows[0].date.getTime();
    const dataEndTime = visibleRows[visibleRows.length - 1].date.getTime();
    const endTime = dataEndTime;
    const timeSpan = Math.max(1, endTime - startTime);
    const xForDate = date => padding.left + (date.getTime() - startTime) / timeSpan * plotWidth;
    const xFor = index => xForDate(visibleRows[index].date);
    const yValue = value => padding.top + (maxValue - value) / (maxValue - minValue) * plotHeight;
    const strategyY = row => yValue(row.strategyReturn);
    const benchmarkY = row => yValue(row.benchmarkReturn);

    ctx.save();
    ctx.font = "12.5px 'JetBrains Mono', monospace";
    ctx.fillStyle = theme.axisLabel;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let line = 0; line <= 4; line += 1) {
      const ratio = line / 4;
      const y = padding.top + ratio * plotHeight;
      const value = maxValue - ratio * (maxValue - minValue);
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(width - padding.right, y);
      ctx.strokeStyle = Math.abs(value) < spread / 8 ? theme.gridMajor : theme.gridMinor;
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.fillText(axisPercent(value), padding.left - 8, y);
    }
    ctx.restore();

    if (!ownerMode) {
      const zeroY = yValue(0);
      const fill = ctx.createLinearGradient(0, padding.top, 0, padding.top + plotHeight);
      fill.addColorStop(0, theme.backtestFillFrom);
      fill.addColorStop(1, theme.backtestFillTo);
      ctx.beginPath();
      visibleRows.forEach((row, index) => {
        const x = xFor(index);
        const y = strategyY(row);
        if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.lineTo(xFor(visibleRows.length - 1), zeroY);
      ctx.lineTo(xFor(0), zeroY);
      ctx.closePath();
      ctx.fillStyle = fill;
      ctx.fill();

      drawSeries(visibleRows, xFor, benchmarkY, theme.benchmark, 1.7, [5, 4]);
      drawSeries(visibleRows, xFor, strategyY, theme.strategy, 2.2);
    } else {
      const zeroY = yValue(0);
      const ownerFill = ctx.createLinearGradient(0, padding.top, 0, padding.top + plotHeight);
      ownerFill.addColorStop(0, theme.ownerFillFrom);
      ownerFill.addColorStop(1, theme.ownerFillTo);
      ctx.beginPath();
      visibleRows.forEach((row, index) => {
        const x = xFor(index);
        const y = strategyY(row);
        if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.lineTo(xFor(visibleRows.length - 1), zeroY);
      ctx.lineTo(xFor(0), zeroY);
      ctx.closePath();
      ctx.fillStyle = ownerFill;
      ctx.fill();

      drawSeries(visibleRows, xFor, benchmarkY, theme.benchmark, 2.1);
      drawSeries(visibleRows, xFor, strategyY, theme.owner, 2.5);
    }

    const labelIndexes = [0, Math.floor((visibleRows.length - 1) / 2), visibleRows.length - 1];
    ctx.save();
    ctx.font = "11.5px 'JetBrains Mono', monospace";
    ctx.fillStyle = theme.xLabel;
    ctx.textBaseline = 'bottom';
    labelIndexes.forEach((index, position) => {
      ctx.textAlign = position === 0 ? 'left' : (position === 2 ? 'right' : 'center');
      ctx.fillText(visibleRows[index].dateLabel, xFor(index), height - 8);
    });
    ctx.restore();

    if (ownerMode) {
      visibleRows.forEach((row, index) => {
        if (!row.ownerEvent || index === 0) return;
        ctx.beginPath();
        ctx.arc(xFor(index), strategyY(row), 2.6, 0, Math.PI * 2);
        ctx.fillStyle = theme.hoverDotFill;
        ctx.fill();
        ctx.strokeStyle = theme.owner;
        ctx.lineWidth = 1.4;
        ctx.stroke();
      });

      const latestOwner = visibleRows[visibleRows.length - 1];
      const latestX = xFor(visibleRows.length - 1);
      const latestY = strategyY(latestOwner);
      ctx.save();
      ctx.font = "600 13px 'JetBrains Mono', monospace";
      ctx.fillStyle = theme.owner;
      ctx.textAlign = 'right';
      ctx.textBaseline = 'bottom';
      ctx.fillText(`Owner ${signedPercent(latestOwner.strategyReturn)}`, latestX - 8, Math.max(14, latestY - 8));
      ctx.restore();

      const benchmarkLatestX = latestX;
      const benchmarkLatestY = benchmarkY(latestOwner);
      const labelsAreClose = Math.abs(latestY - benchmarkLatestY) < 24;
      ctx.save();
      ctx.font = "600 13px 'JetBrains Mono', monospace";
      ctx.fillStyle = theme.benchmarkLabel;
      ctx.textAlign = 'right';
      ctx.textBaseline = labelsAreClose ? 'top' : 'bottom';
      ctx.fillText(
        `S&P 500 ${signedPercent(latestOwner.benchmarkReturn)}`,
        benchmarkLatestX - 8,
        labelsAreClose ? Math.min(height - 34, benchmarkLatestY + 8) : Math.max(14, benchmarkLatestY - 8)
      );
      ctx.restore();
    }

    if (hoverIndex !== null && visibleRows[hoverIndex]) {
      const row = visibleRows[hoverIndex];
      const x = xFor(hoverIndex);
      ctx.beginPath();
      ctx.moveTo(x, padding.top);
      ctx.lineTo(x, padding.top + plotHeight);
      ctx.strokeStyle = theme.crosshair;
      ctx.lineWidth = 1;
      ctx.stroke();
      [[strategyY(row), ownerMode ? theme.owner : theme.strategy], [benchmarkY(row), theme.benchmark]].forEach(([y, color]) => {
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fillStyle = theme.hoverDotFill;
        ctx.fill();
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.stroke();
      });
    }
  };

  const renderRange = range => {
    activeRange = range;
    hoverIndex = null;
    if (tooltip) tooltip.classList.remove('show');
    visibleRows = selectRows(range);
    const latest = visibleRows[visibleRows.length - 1];
    const strategyReturn = latest.strategyReturn;
    const benchmarkReturn = latest.benchmarkReturn;
    const ownerMode = latest.ownerComparison === true;
    if (Number.isFinite(backtestYtdReturn)) paintMetric(strategyMetric, backtestYtdReturn);
    if (ownerMode) {
      if (ownerMetric) paintMetric(ownerMetric, strategyReturn);
      paintMetric(benchmarkMetric, benchmarkReturn);
      paintMetric(alphaMetric, strategyReturn - benchmarkReturn);
      if (alphaLabel) alphaLabel.textContent = 'Portfolio vs S&P';
      if (alphaHelp) alphaHelp.textContent = 'Owner portfolio minus SPY for the selected period';
      windowLabel.textContent = `${rangeNames[range]} comparison · ${visibleRows[0].dateLabel} to ${latest.dateLabel}`;
      canvas.setAttribute('aria-label', `For ${rangeNames[range]}, the reconstructed owner portfolio returned ${signedPercent(strategyReturn)} compared with ${signedPercent(benchmarkReturn)} for the S&P 500.`);
      if (performanceRisk) performanceRisk.textContent = `Portfolio comparison: ${rangeNames[range]} return from the reconstructed Trading 212 closed-trade path versus SPY. The owner path is anchored to the owner-reported ${signedPercent(ownerSnapshotReturn)} YTD result and is not a daily brokerage equity curve.`;
      if (performanceAsOf) performanceAsOf.textContent = `Selected range ${visibleRows[0].dateLabel} to ${latest.dateLabel} · common owner and SPY dates`;
    } else {
      paintMetric(strategyMetric, strategyReturn);
      paintMetric(benchmarkMetric, benchmarkReturn);
      paintMetric(alphaMetric, strategyReturn - benchmarkReturn);
      if (alphaLabel) alphaLabel.textContent = 'Backtest excess';
      if (alphaHelp) alphaHelp.textContent = 'Rules backtest minus benchmark';
      windowLabel.textContent = `${rangeNames[range]} · ${visibleRows[0].dateLabel} to ${latest.dateLabel}`;
      canvas.setAttribute('aria-label', `For ${rangeNames[range]}, the rules backtest returned ${signedPercent(strategyReturn)} compared with ${signedPercent(benchmarkReturn)} for the S&P 500.`);
      if (performanceRisk) performanceRisk.textContent = 'Hypothetical rules backtest: this line is not the owner account and is not a guarantee of future returns.';
      if (performanceAsOf) performanceAsOf.textContent = `Comparable data · ${visibleRows[0].dateLabel} to ${latest.dateLabel}`;
    }
    if (performanceLegend) performanceLegend.classList.toggle('owner-mode', ownerMode);
    if (tooltipOwnerRow) tooltipOwnerRow.hidden = !ownerMode;
    if (tooltipBacktestRow) tooltipBacktestRow.hidden = ownerMode;
    if (tooltipBenchmarkRow) tooltipBenchmarkRow.hidden = false;
    buttons.forEach(button => {
      const active = button.dataset.range === range;
      button.classList.toggle('active', active);
      button.setAttribute('aria-pressed', String(active));
    });
    drawChart();
  };

  buttons.forEach(button => button.addEventListener('click', () => renderRange(button.dataset.range)));
  canvas.addEventListener('pointermove', event => {
    const rect = canvas.getBoundingClientRect();
    const leftPadding = rect.width < 520 ? 48 : 58;
    const plotWidth = rect.width - leftPadding - 17;
    const localX = Math.max(0, Math.min(plotWidth, event.clientX - rect.left - leftPadding));
    const ownerMode = visibleRows[0]?.ownerComparison === true;
    const hoverRows = visibleRows;
    const startTime = hoverRows[0].date.getTime();
    const endTime = hoverRows[hoverRows.length - 1].date.getTime();
    const hoveredTime = startTime + (localX / Math.max(1, plotWidth)) * Math.max(1, endTime - startTime);
    hoverIndex = hoverRows.reduce((bestIndex, candidate, index) => (
      Math.abs(candidate.date.getTime() - hoveredTime) < Math.abs(hoverRows[bestIndex].date.getTime() - hoveredTime)
        ? index
        : bestIndex
    ), 0);
    const row = hoverRows[hoverIndex];
    if (!row) return;
    tooltipDate.textContent = dateFormatter.format(row.date);
    if (ownerMode) {
      tooltipOwner.textContent = signedPercent(row.strategyReturn);
      tooltipBenchmark.textContent = signedPercent(row.benchmarkReturn);
    } else {
      tooltipStrategy.textContent = signedPercent(row.strategyReturn);
      tooltipBenchmark.textContent = signedPercent(row.benchmarkReturn);
    }
    const tooltipWidth = 184;
    const preferredLeft = leftPadding + (row.date.getTime() - startTime) / Math.max(1, endTime - startTime) * plotWidth + 12;
    tooltip.style.left = `${Math.min(rect.width - tooltipWidth - 10, Math.max(10, preferredLeft))}px`;
    tooltip.style.top = '16px';
    tooltip.classList.add('show');
    drawChart();
  });
  canvas.addEventListener('pointerleave', () => {
    hoverIndex = null;
    if (tooltip) tooltip.classList.remove('show');
    drawChart();
  });

  if ('ResizeObserver' in window) {
    new ResizeObserver(() => drawChart()).observe(canvas.parentElement);
  } else {
    window.addEventListener('resize', drawChart, { passive: true });
  }
  window.addEventListener('aiinsider:themechange', () => drawChart());
  renderRange(activeRange);
})();
</script>

""" + FOOTER.replace("{{ year }}", str(datetime.now().year)) + """
</body></html>
"""


@app.route("/")
def home():
    requested_symbol = request.args.get("symbol", "").strip().upper()
    chart_symbol = requested_symbol if re.fullmatch(r"[A-Z]{1,6}(\.[A-Z]{1,2})?", requested_symbol) else None
    ytd_snapshot = load_public_ytd_snapshot()
    ytd_ledger = load_ytd_trade_ledger(ytd_snapshot)
    performance = load_performance_comparison()
    if performance and ytd_snapshot:
        owner_curve = build_owner_portfolio_continuation(ytd_snapshot, ytd_ledger)
        performance = {
            **performance,
            "owner_snapshot": {
                "as_of": ytd_snapshot["as_of"],
                "return_pct": ytd_snapshot["ytd_return_pct"],
                "label": ytd_snapshot["return_label"],
            },
            "owner_curve": owner_curve,
        }
    weekly_signals, weekly_since = load_weekly_signals()
    return render_template_string(
        HOME_TEMPLATE,
        signals=load_public_signals(),
        examples=load_resolved_examples(),
        filings=load_filings_table(),
        weekly_signals=weekly_signals,
        weekly_since=weekly_since,
        trust_stats=load_trust_stats(),
        performance=performance,
        ytd_snapshot=ytd_snapshot,
        ytd_ledger=ytd_ledger,
        market_trades=load_market_trades(),
        chart_symbol=chart_symbol,
        requested_symbol=requested_symbol,
        rating_performance=load_rating_performance(),
    )


# ── track record ─────────────────────────────────────────────────

TRACK_RECORD_TEMPLATE = """
<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#03070c">
""" + THEME_INIT + """
<title>Track Record &mdash; AI INSIDER</title>
<style>""" + BASE_CSS + """</style>
</head><body>
""" + NAV + """
<div class="hero"><div class="wrap">
  <span class="eyebrow">Two numbers, labeled honestly</span>
  <h1>The <span class="grad">backtest</span>, and the <span class="grad">real thing</span>.</h1>
  <p>Backtests are optimistic by nature. The live numbers are what actually happened, as they accumulate.</p>
</div></div>

<section><div class="wrap">
  <div class="section-head" style="text-align:left;margin:0 0 28px;">
    <span class="eyebrow">2021&ndash;2026 simulation &middot; Hypothetical backtest</span>
    <h2 style="font-size:26px;">20+ year backtest</h2>
  </div>
  {% if returns %}
  <div class="stat-grid">
    <div class="stat" title="{{ metric_help.cumulative_roi_pct }}">
      <div class="n">{{ returns.cumulative_roi_pct }}%</div>
      <div class="l">Cumulative ROI &nbsp;<span style="opacity:0.6;cursor:help;">?</span></div>
    </div>
    <div class="stat" title="{{ metric_help.xirr_pct }}">
      <div class="n">{{ returns.xirr_pct }}%</div>
      <div class="l">XIRR (money-weighted) &nbsp;<span style="opacity:0.6;cursor:help;">?</span></div>
    </div>
    <div class="stat" title="{{ metric_help.twr_annualized_pct }}">
      <div class="n">{{ returns.twr_annualized_pct }}%</div>
      <div class="l">Time-weighted CAGR &nbsp;<span style="opacity:0.6;cursor:help;">?</span></div>
    </div>
    <div class="stat"><div class="n">${{ "{:,.0f}".format(returns.total_deposited) }}</div><div class="l">Total deposited</div></div>
    <div class="stat"><div class="n">${{ "{:,.0f}".format(returns.profit) }}</div><div class="l">Profit</div></div>
    {% if backtest %}
    <div class="stat"><div class="n">{{ backtest.get('win_rate_%','\u2014') }}%</div><div class="l">Win rate</div></div>
    <div class="stat"><div class="n">{{ backtest.get('max_drawdown_%','\u2014') }}%</div><div class="l">Max drawdown</div></div>
    <div class="stat"><div class="n">{{ backtest.get('positions','\u2014') }}</div><div class="l">Trades simulated</div></div>
    {% endif %}
  </div>
  <p style="color:var(--muted);font-size:13px;margin-top:16px;">Hover any metric for what it means. XIRR and time-weighted CAGR are computed live from the daily equity ledger &mdash; not hardcoded.</p>
  {% else %}
  <div class="empty">Return calculations not available &mdash; equity ledger not published yet.</div>
  {% endif %}
  <div class="disclaimer-box"><strong>Hypothetical backtest, not investor performance:</strong> this is a simulation, not a live track record. It can carry survivorship bias and does not capture every real-world cost or friction. See <a href="/how-it-works" style="color:var(--accent);">methodology</a> for assumptions on fees, slippage, and entry timing.</div>
</div></section>

<section><div class="wrap">
  <div class="section-head" style="text-align:left;margin:0 0 28px;">
    <span class="eyebrow">Out of sample</span>
    <h2 style="font-size:26px;">Live results</h2>
  </div>
  {% if live %}
  <div class="stat-grid">
    <div class="stat"><div class="n">{{ live.n }}</div><div class="l">Closed trades</div></div>
    <div class="stat"><div class="n">{{ live.win_rate }}%</div><div class="l">Win rate</div></div>
    <div class="stat"><div class="n">{{ live.avg_pnl }}%</div><div class="l">Avg P&amp;L</div></div>
  </div>
  {% else %}
  <div class="empty">Live track record is still building. This updates automatically as real trades close &mdash; shown honestly empty rather than hidden.</div>
  {% endif %}
  """ + ad_slot() + """
</div></section>

""" + FOOTER.replace("{{ year }}", str(datetime.now().year)) + """
</body></html>
"""


def load_rating_performance():
    """Real win-rate + average return per rating tier (A+/A/B+/B), computed
    from actual resolved outcomes only -- never estimated or backfilled.
    Skips any tier with fewer than 3 resolved trades so a single lucky (or
    unlucky) signal can't produce a misleadingly precise stat. Returns {}
    (honest empty state) if outcomes data isn't available yet.
    """
    path = os.path.join(BASE_DIR, "public_signal_outcomes.csv")
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return {}
    if df.empty or "status" not in df.columns or "rating" not in df.columns:
        return {}
    resolved = df[df["status"].isin(["WIN", "LOSS"])].copy()
    if resolved.empty:
        return {}
    out = {}
    for rating, group in resolved.groupby("rating"):
        if len(group) < 3:
            continue
        out[rating] = {
            "win_rate": round((group["status"] == "WIN").mean() * 100, 1),
            "avg_return": round(group["pnl_pct"].mean(), 1) if "pnl_pct" in group.columns else None,
            "n": int(len(group)),
        }
    return out


def load_leaderboard_data():
    """Reads REAL resolved outcomes only — never estimates or projects an
    unresolved signal's eventual result. Splits into week/month/YTD windows."""
    path = os.path.join(BASE_DIR, "public_signal_outcomes.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None
    if df.empty or "status" not in df.columns:
        return None

    df["resolved_date"] = pd.to_datetime(df["resolved_date"], errors="coerce")
    resolved = df[df["status"].isin(["WIN", "LOSS"]) & df["resolved_date"].notna()].copy()
    if resolved.empty:
        return {"empty": True, "open_count": int((df["status"] == "OPEN").sum())}

    now = pd.Timestamp.now()
    windows = {
        "week": resolved[resolved["resolved_date"] >= now - pd.Timedelta(days=7)],
        "month": resolved[resolved["resolved_date"] >= now - pd.Timedelta(days=30)],
        "ytd": resolved[resolved["resolved_date"] >= pd.Timestamp(year=now.year, month=1, day=1)],
    }
    out = {"empty": False, "open_count": int((df["status"] == "OPEN").sum())}
    for name, w in windows.items():
        top = w.nlargest(10, "pnl_pct") if not w.empty else pd.DataFrame()
        out[name] = {
            "trades": top.to_dict("records"),
            "n_resolved": len(w),
            "win_rate": round((w["status"] == "WIN").mean() * 100, 1) if len(w) else None,
        }
    return out


@app.route("/leaderboard")
def leaderboard():
    data = load_leaderboard_data()
    return render_template_string(LEADERBOARD_TEMPLATE, data=data)


LEADERBOARD_TEMPLATE = """
<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#03070c">
""" + THEME_INIT + """
<title>Leaderboard &mdash; AI INSIDER</title>
<style>""" + BASE_CSS + """</style>
</head><body>
""" + NAV + """
<div class="hero"><div class="wrap">
  <span class="eyebrow">Best resolved signals &middot; real outcomes only</span>
  <h1>The <span class="grad">best-performing</span> signals.</h1>
  <p>Ranked from signals that actually resolved (hit target or expired) &mdash; never an in-progress guess. This is signal performance, not a user leaderboard; nothing here is self-reported.</p>
</div></div>

<section><div class="wrap">
  {% if data is none %}
  <div class="empty">Leaderboard data not published yet.</div>
  {% elif data.empty %}
  <div class="empty">No signals have resolved yet ({{ data.open_count }} still in progress &mdash; even the fastest signals take up to 30 days to resolve). Check back soon.</div>
  {% else %}

  {% for window, label in [('week','This Week'), ('month','This Month'), ('ytd','Year to Date')] %}
  <div class="section-head" style="text-align:left;margin:0 0 20px;">
    <span class="eyebrow">{{ label }}</span>
    <h2 style="font-size:22px;">
      {% if data[window].n_resolved %}{{ data[window].n_resolved }} resolved &middot; {{ data[window].win_rate }}% win rate{% else %}No resolved signals in this window yet{% endif %}
    </h2>
  </div>
  {% if data[window].trades %}
  <div class="trade-grid" style="margin-bottom:40px;">
    {% for t in data[window].trades[:6] %}
    <div class="trade-card">
      <div class="trade-top">
        <div class="trade-avatar">{{ t.ticker[:2] }}</div>
        <div class="trade-gain {{ 'pos' if t.pnl_pct >= 0 else 'neg' }}">{{ '+' if t.pnl_pct >= 0 else '' }}{{ t.pnl_pct }}%</div>
      </div>
      <div class="trade-ticker">${{ t.ticker }}</div>
      <div class="trade-meta">Signaled {{ t.signal_date }} &middot; resolved {{ t.resolved_date.strftime('%Y-%m-%d') if t.resolved_date is not string else t.resolved_date }}</div>
      <span class="trade-badge {{ t.rating|replace('+','plus') }}">{{ t.rating }}</span>
    </div>
    {% endfor %}
  </div>
  {% else %}
  <div class="empty" style="margin-bottom:40px;">Nothing resolved in this window yet.</div>
  {% endif %}
  {% endfor %}

  {% endif %}
  """ + ad_slot("Ad — leaderboard") + """
</div></section>

""" + FOOTER.replace("{{ year }}", str(datetime.now().year)) + """
</body></html>
"""



@app.route("/track-record")
def track_record():
    return render_template_string(
        TRACK_RECORD_TEMPLATE,
        backtest=load_backtest_summary(), live=load_live_track_record(),
        returns=load_real_returns(), metric_help=METRIC_EXPLANATIONS,
    )


# ── how it works ─────────────────────────────────────────────────

HOW_IT_WORKS_TEMPLATE = """
<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#03070c">
""" + THEME_INIT + """
<title>How It Works &mdash; AI INSIDER</title>
<style>""" + BASE_CSS + """</style>
</head><body>
""" + NAV + """
<div class="hero"><div class="wrap">
  <span class="eyebrow">Methodology</span>
  <h1>What we score, and <span class="grad">why</span>.</h1>
</div></div>
<section><div class="wrap" style="max-width:660px;">
  <p style="color:#c3c6d1;">Every signal starts with an SEC Form 4 filing &mdash; disclosure required within two business days of an insider buying or selling their own company's stock. We only look at open-market purchases, never option exercises, grants, or automatic sales.</p>
  <p style="color:#c3c6d1;">Each qualifying purchase is scored on: how much it grows the insider's existing stake, whether a CEO or CFO was involved, whether multiple insiders bought together, price relative to the 52-week range, and market capitalization. Purchases that already ran too far past the insider's own entry price are excluded.</p>
  <div class="disclaimer-box"><strong>What this is not:</strong> a recommendation to buy or sell any security, personalized advice, or a guarantee of anything. It's a rules-based screen of public filings, published for informational purposes. Full <a href="/disclaimer" style="color:var(--accent);">disclaimer</a>.</div>
</div></section>
""" + FOOTER.replace("{{ year }}", str(datetime.now().year)) + """
</body></html>
"""


@app.route("/how-it-works")
def how_it_works():
    return render_template_string(HOW_IT_WORKS_TEMPLATE)


# ── insider trades (full market-wide feed) ─────────────────────────

INSIDER_TRADES_TEMPLATE = """
<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#03070c">
""" + THEME_INIT + """
<title>Insider Trades &mdash; every real open-market Form 4 &mdash; AI INSIDER</title>
<meta name="description" content="Every open-market insider buy or sell filed market-wide with the SEC, unfiltered and unscored -- separate from AI Insider's own scored signal feed.">
<style>""" + BASE_CSS + """</style>
</head><body>
""" + NAV + """
<div class="hero"><div class="wrap">
  <span class="eyebrow">Unfiltered &middot; market-wide &middot; not our signals</span>
  <h1>Every real insider trade, <span class="grad">not just the ones we flag</span>.</h1>
  <p>Every open-market Form 4 buy or sell filed anywhere in the market, pulled straight from SEC EDGAR. This is the raw stream AI Insider's model screens down to the <a href="/#signals" style="color:var(--accent);">scored signals</a> above.</p>
</div></div>

<section><div class="wrap">
  <div class="section-head" style="text-align:left;margin:0 0 28px;">
    <span class="eyebrow">Live feed</span>
    <h2 style="font-size:26px;">{{ trades|length }} recent trade{{ '' if trades|length == 1 else 's' }}</h2>
    <p style="margin-top:8px;">Synced hourly from SEC EDGAR's current Form 4 filings. Only genuine open-market buys and sells &mdash; grants, option exercises, and gifts are excluded.</p>
  </div>
  {% if trades %}
  <div class="filings-table-wrap">
    <table class="filings">
      <thead><tr>
        <th>Date</th><th>Ticker</th><th>Company</th><th>Insider</th><th>Type</th><th class="amt">Shares</th><th class="amt">Price</th><th class="amt">Value</th>
      </tr></thead>
      <tbody>
        {% for t in trades %}
        <tr>
          <td>{{ t.transaction_date }}</td>
          <td class="tk">${{ t.ticker }}</td>
          <td>{{ t.company or '—' }}</td>
          <td>{{ t.insider_name or '—' }}{% if t.insider_title %}<div class="role">{{ t.insider_title }}</div>{% endif %}</td>
          <td><span class="ma-type {{ 'sell' if t.transaction_type == 'Open-market sell' else 'buy' }}">{{ t.transaction_type }}</span></td>
          <td class="amt">{{ '{:,.0f}'.format(t.shares) if t.shares else '—' }}</td>
          <td class="amt">{{ '${:,.2f}'.format(t.price) if t.price else '—' }}</td>
          <td class="amt">{{ '${:,.0f}'.format(t.value) if t.value else '—' }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
  {% else %}
  <div class="empty">No market-wide trades synced yet &mdash; this fills in automatically once the hourly SEC EDGAR sync has run. Shown honestly empty rather than faked.</div>
  {% endif %}
  <p class="signal-sizing-note" style="margin-top:16px;">This feed is informational only, not a recommendation. It reflects public SEC filings and may be incomplete or delayed.</p>
</div></section>

""" + FOOTER.replace("{{ year }}", str(datetime.now().year)) + """
</body></html>
"""


@app.route("/insider-trades")
def insider_trades():
    return render_template_string(INSIDER_TRADES_TEMPLATE, trades=load_market_trades(limit=200))


# ── congress trades ─────────────────────────────────────────────────

CONGRESS_TEMPLATE = """
<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#03070c">
""" + THEME_INIT + """
<title>Congress Trading &mdash; House &amp; Senate stock disclosures &mdash; AI INSIDER</title>
<meta name="description" content="Every stock trade disclosed by members of the House and Senate under the STOCK Act, synced from the House Stock Watcher and Senate Stock Watcher public datasets.">
<style>""" + BASE_CSS + """</style>
</head><body>
""" + NAV + """
<div class="hero"><div class="wrap">
  <span class="eyebrow">STOCK Act disclosures &middot; House &amp; Senate</span>
  <h1>What <span class="grad">Congress</span> is trading.</h1>
  <p>Every periodic transaction report members of the House and Senate are required to file when they buy or sell stock, synced from the House Stock Watcher and Senate Stock Watcher public datasets. Disclosures are legally allowed to lag the actual trade by up to 45 days &mdash; these are reported dates, not real-time.</p>
</div></div>

<section><div class="wrap">
  <div class="section-head" style="text-align:left;margin:0 0 28px;">
    <span class="eyebrow">Live feed</span>
    <h2 style="font-size:26px;">{{ trades|length }} recent disclosure{{ '' if trades|length == 1 else 's' }}</h2>
    <p style="margin-top:8px;">Synced every 6 hours from the House Stock Watcher and Senate Stock Watcher datasets, which parse the official House Clerk and Senate eFD filings.</p>
  </div>
  {% if trades %}
  <div class="filings-table-wrap">
    <table class="filings">
      <thead><tr>
        <th>Disclosed</th><th>Traded</th><th>Chamber</th><th>Member</th><th>Ticker</th><th>Type</th><th class="amt">Amount</th>
      </tr></thead>
      <tbody>
        {% for t in trades %}
        <tr>
          <td>{{ t.disclosure_date or '—' }}</td>
          <td>{{ t.transaction_date or '—' }}</td>
          <td>{{ t.chamber }}</td>
          <td>{{ t.member or '—' }}{% if t.state_or_district %}<div class="role">{{ t.state_or_district }}</div>{% endif %}</td>
          <td class="tk">${{ t.ticker }}</td>
          <td><span class="ma-type {{ 'sell' if 'Sale' in (t.transaction_type or '') else 'buy' }}">{{ t.transaction_type }}</span></td>
          <td class="amt">{{ t.amount_range or '—' }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
  {% else %}
  <div class="empty">No congressional disclosures synced yet &mdash; this fills in automatically once the sync has run. Shown honestly empty rather than faked.</div>
  {% endif %}
  <p class="signal-sizing-note" style="margin-top:16px;">Congressional trading data is sourced from third-party open datasets that parse official disclosures; it is informational only, may lag or contain gaps, and is not a recommendation.</p>
</div></section>

""" + FOOTER.replace("{{ year }}", str(datetime.now().year)) + """
</body></html>
"""


@app.route("/congress")
def congress_trades():
    return render_template_string(CONGRESS_TEMPLATE, trades=load_congress_trades(limit=200))


# ── market news ──────────────────────────────────────────────────

MARKET_NEWS_TEMPLATE = """
<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#03070c">
""" + THEME_INIT + """
<title>Market News &mdash; AI INSIDER</title>
<meta name="description" content="Financial market headlines aggregated from public RSS feeds.">
<style>""" + BASE_CSS + """</style>
</head><body>
""" + NAV + """
<div class="hero"><div class="wrap">
  <span class="eyebrow">Aggregated headlines</span>
  <h1>Market <span class="grad">news</span>, in one place.</h1>
  <p>Financial headlines pulled from public market-news feeds, refreshed hourly. General market context, not part of AI Insider's own research.</p>
</div></div>

<section><div class="wrap">
  <div class="section-head" style="text-align:left;margin:0 0 28px;">
    <span class="eyebrow">Live feed</span>
    <h2 style="font-size:26px;">{{ news|length }} recent headline{{ '' if news|length == 1 else 's' }}</h2>
  </div>
  {% if news %}
  <div style="max-width:820px;">
    {% for n in news %}
    <div class="faq-item">
      <h4><a href="{{ n.link }}" target="_blank" rel="noopener nofollow" style="color:var(--text);">{{ n.title }}</a></h4>
      <p>{{ n.source }}{% if n.published %} &middot; {{ n.published }}{% endif %}</p>
    </div>
    {% endfor %}
  </div>
  {% else %}
  <div class="empty">No market news synced yet &mdash; this fills in automatically once the sync has run. Shown honestly empty rather than faked.</div>
  {% endif %}
</div></section>

""" + FOOTER.replace("{{ year }}", str(datetime.now().year)) + """
</body></html>
"""


@app.route("/market-news")
def market_news():
    return render_template_string(MARKET_NEWS_TEMPLATE, news=load_market_news(limit=60))


# ── simple pages ─────────────────────────────────────────────────

SIMPLE_PAGE = """
<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#03070c">
""" + THEME_INIT + """
<title>{{ title }} &mdash; AI INSIDER</title>
<style>""" + BASE_CSS + """</style>
</head><body>
""" + NAV + """
<div class="hero"><div class="wrap"><h1>{{ title }}</h1></div></div>
<section><div class="wrap" style="color:var(--muted);font-size:14px;max-width:680px;">
{{ body|safe }}
</div></section>
""" + FOOTER.replace("{{ year }}", str(datetime.now().year)) + """
</body></html>
"""

DISCLAIMER_BODY = """
<p>AI INSIDER publishes information derived from public SEC filings for informational and educational purposes only. Nothing on this site constitutes investment advice, a recommendation, or a solicitation to buy or sell any security. We are not a registered investment adviser or broker-dealer.</p>
<p>Past performance, whether backtested or live, is not indicative of future results. All investing involves risk, including possible loss of principal. Do your own research and consult a licensed financial professional before making investment decisions.</p>
<p>Backtested results shown on this site are hypothetical, have inherent limitations, and do not represent actual trading.</p>
<p>Market data and charts may be delayed, incomplete, or unavailable. They are provided for context only and must not be used for order execution or price verification.</p>
"""

PRIVACY_BODY = """
<p>This site does not currently collect personal information beyond standard web server logs. If ads are enabled, third-party providers (e.g. Google AdSense) may use cookies to serve relevant ads &mdash; see their own privacy policies.</p>
<p>The market chart is provided by TradingView. Loading the embedded chart connects your browser directly to TradingView, and that connection is governed by TradingView's privacy policy.</p>
"""

ABOUT_BODY = """
<p>AI INSIDER is built and run by one person who got tired of scrolling insider-trading tables by hand. It started as a personal tool; this public version shares the same signal-scoring engine, minus any personal position sizing.</p>
"""


@app.route("/disclaimer")
def disclaimer():
    return render_template_string(SIMPLE_PAGE, title="Disclaimer", body=DISCLAIMER_BODY)


@app.route("/privacy")
def privacy():
    return render_template_string(SIMPLE_PAGE, title="Privacy", body=PRIVACY_BODY)


@app.route("/about")
def about():
    return render_template_string(SIMPLE_PAGE, title="About", body=ABOUT_BODY)


if __name__ == "__main__":
    print("Public site starting (preview only)...")
    print("Open http://localhost:5001")
    # 0.0.0.0 = accept connections from anywhere (needed on a real server).
    # If you're ever running this on your own PC again for local testing,
    # change back to "127.0.0.1" so it's not exposed to your home network.
    app.run(host="0.0.0.0", port=5001, debug=False)
