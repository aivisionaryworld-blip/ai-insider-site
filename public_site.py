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
import sys
from datetime import datetime
from functools import lru_cache

import numpy as np
import pandas as pd
from flask import Flask, render_template_string, abort, jsonify

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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
  --bg: #08090d;
  --surface: #111319;
  --surface-2: #171a22;
  --border: #22252e;
  --text: #e7e9ee;
  --muted: #8b8fa3;
  --accent: #4ee2a6;
  --accent-dim: rgba(78,226,166,0.1);
  --red: #ef5f6f;
  --amber: #e8b04b;
  --aplus: var(--amber);
  --a: var(--accent);
  --b: #7c93c9;
}
* { box-sizing: border-box; }
body {
  background: var(--bg); color: var(--text);
  font-family: 'Inter', -apple-system, sans-serif;
  margin: 0; line-height: 1.5; font-size: 15px;
  position: relative; overflow-x: hidden;
}
.mono { font-family: 'JetBrains Mono', monospace; }
h1, h2, h3 { font-weight: 700; letter-spacing: -0.02em; margin: 0; }
a { color: inherit; }
.wrap { max-width: 1080px; margin: 0 auto; padding: 0 24px; position: relative; z-index: 1; }

/* animated gradient mesh background — the "more energy" layer */
.bg-mesh {
  position: fixed; inset: 0; z-index: 0; pointer-events: none; overflow: hidden;
}
.bg-mesh span {
  position: absolute; border-radius: 50%; filter: blur(90px); opacity: 0.16;
}
.bg-mesh .b1 {
  width: 480px; height: 480px; background: var(--accent);
  top: -160px; left: -100px; animation: drift1 22s ease-in-out infinite;
}
.bg-mesh .b2 {
  width: 420px; height: 420px; background: var(--amber);
  top: 20%; right: -140px; animation: drift2 26s ease-in-out infinite;
}
.bg-mesh .b3 {
  width: 380px; height: 380px; background: #5f7de8;
  bottom: -120px; left: 30%; animation: drift3 30s ease-in-out infinite;
}
@keyframes drift1 { 0%,100% { transform: translate(0,0); } 50% { transform: translate(60px,80px); } }
@keyframes drift2 { 0%,100% { transform: translate(0,0); } 50% { transform: translate(-70px,50px); } }
@keyframes drift3 { 0%,100% { transform: translate(0,0); } 50% { transform: translate(50px,-60px); } }
@media (prefers-reduced-motion: reduce) {
  .bg-mesh span { animation: none; }
}

nav {
  position: sticky; top: 0; z-index: 10;
  background: rgba(8,9,13,0.85); backdrop-filter: blur(10px);
}
nav .wrap { display: flex; justify-content: space-between; align-items: center; }
.brand { font-weight: 800; font-size: 17px; display: flex; align-items: center; gap: 8px; }
.brand .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--accent); box-shadow: 0 0 12px var(--accent); }
.nav-links { display: flex; gap: 28px; align-items: center; }
.nav-links a { font-size: 14px; color: var(--muted); text-decoration: none; }
.nav-links a:hover { color: var(--text); }
.cta {
  background: var(--accent); color: #06110c; font-weight: 700; font-size: 13.5px;
  padding: 9px 18px; border-radius: 7px; text-decoration: none; border: none; cursor: pointer;
}
.cta.ghost {
  background: transparent; border: 1px solid var(--border); color: var(--text);
}

.hero { padding: 88px 0 64px; text-align: center; }
.hero .eyebrow {
  color: var(--accent); font-size: 13px; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.1em; margin-bottom: 18px; display: inline-block;
  animation: eyebrow-in 0.6s ease-out;
}
@keyframes eyebrow-in { from { opacity: 0; transform: translateY(-6px); } to { opacity: 1; transform: translateY(0); } }
.hero h1 {
  font-size: 52px; line-height: 1.08; max-width: 780px; margin: 0 auto;
  animation: hero-in 0.7s ease-out 0.1s both;
}
@keyframes hero-in { from { opacity: 0; transform: translateY(16px); } to { opacity: 1; transform: translateY(0); } }
.hero h1 .grad {
  background: linear-gradient(90deg, var(--accent), #7de8c7, var(--accent));
  background-size: 200% auto;
  -webkit-background-clip: text; background-clip: text; color: transparent;
  animation: shimmer 4s ease-in-out infinite;
}
@keyframes shimmer { 0%,100% { background-position: 0% center; } 50% { background-position: 100% center; } }
@media (prefers-reduced-motion: reduce) {
  .hero .eyebrow, .hero h1 { animation: none; }
  .hero h1 .grad { animation: none; }
}
.hero p { color: var(--muted); font-size: 17px; max-width: 560px; margin: 20px auto 0; animation: hero-in 0.7s ease-out 0.2s both; }
.hero .ctas { margin-top: 32px; display: flex; gap: 12px; justify-content: center; animation: hero-in 0.7s ease-out 0.3s both; }
.hero .cta { padding: 13px 26px; font-size: 14px; }
.hero .cta:not(.ghost) { box-shadow: 0 0 0 rgba(78,226,166,0.4); transition: box-shadow 0.25s, transform 0.15s; }
.hero .cta:not(.ghost):hover { box-shadow: 0 0 26px rgba(78,226,166,0.35); transform: translateY(-1px); }

section { padding: 72px 0; border-top: 1px solid var(--border); }
.section-head { text-align: center; max-width: 600px; margin: 0 auto 44px; }
.section-head .eyebrow { color: var(--accent); font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; }
.section-head h2 { font-size: 32px; margin-top: 10px; }
.section-head p { color: var(--muted); font-size: 15px; margin-top: 10px; }

.trade-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px,1fr)); gap: 18px; }
.trade-card {
  background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 22px;
  transition: border-color 0.2s, transform 0.2s, box-shadow 0.2s;
}
.trade-card:hover {
  border-color: var(--accent); transform: translateY(-4px);
  box-shadow: 0 12px 32px rgba(0,0,0,0.35), 0 0 20px rgba(78,226,166,0.08);
}
.trade-top { display: flex; justify-content: space-between; align-items: flex-start; }
.trade-avatar {
  width: 38px; height: 38px; border-radius: 50%; background: var(--surface-2);
  display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 13.5px;
  color: var(--muted); border: 1px solid var(--border);
}
.trade-gain { font-size: 24px; font-weight: 800; font-family: 'JetBrains Mono', monospace; }
.trade-gain.pos { color: var(--accent); }
.trade-gain.neg { color: var(--red); }
.trade-ticker { font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 15px; margin-top: 12px; }
.trade-meta { color: var(--muted); font-size: 13.5px; margin-top: 4px; }
.trade-narrative { font-size: 14px; color: #c3c6d1; margin-top: 14px; line-height: 1.6; }

/* Finviz-style compact market data strip */
.mkt-strip {
  display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px 10px;
  margin-top: 14px; padding-top: 12px; border-top: 1px solid var(--border);
  font-family: 'JetBrains Mono', monospace;
}
.mkt-cell { font-size: 13px; }
.mkt-cell .l { color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; font-size: 12px; display: block; }
.mkt-cell .v { color: var(--text); font-weight: 600; font-size: 13px; }
.mkt-cell .v.hi { color: var(--red); }
.mkt-cell .v.lo { color: var(--accent); }

/* trust stats bar — real numbers only */
.trust-bar {
  display: flex; justify-content: center; gap: 48px; flex-wrap: wrap;
  padding: 28px 0; border-top: 1px solid var(--border); border-bottom: 1px solid var(--border);
}
.trust-stat { text-align: center; }
.trust-stat .n { font-family: 'JetBrains Mono', monospace; font-size: 26px; font-weight: 800; color: #fff; }
.trust-stat .l { color: var(--muted); font-size: 13px; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 4px; }

/* professional filings data table */
.filings-table-wrap { overflow-x: auto; border: 1px solid var(--border); border-radius: 12px; background: var(--surface); }
table.filings {
  width: 100%; border-collapse: collapse; font-family: 'JetBrains Mono', monospace; font-size: 13.5px; min-width: 720px;
}
table.filings thead th {
  text-align: left; padding: 12px 16px; font-size: 12.5px; text-transform: uppercase; letter-spacing: 0.05em;
  color: var(--muted); border-bottom: 1px solid var(--border); font-weight: 600; background: var(--surface-2);
}
table.filings tbody td { padding: 11px 16px; border-bottom: 1px solid var(--border); }
table.filings tbody tr:last-child td { border-bottom: none; }
table.filings tbody tr { transition: background 0.15s; }
table.filings tbody tr:hover { background: rgba(255,255,255,0.02); }
table.filings .tk { font-weight: 700; color: var(--text); }
table.filings .role { color: var(--muted); font-size: 13px; }
table.filings .amt { text-align: right; }
.trade-badge {
  display: inline-block; font-size: 12.5px; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.06em; padding: 3px 8px; border-radius: 5px; margin-top: 12px;
}
.trade-badge.Aplus { background: rgba(232,176,75,0.12); color: var(--amber); }
.trade-badge.A { background: var(--accent-dim); color: var(--accent); }
.trade-badge.Bplus { background: rgba(124,147,201,0.18); color: #9fb3d4; }
.trade-badge.B { background: rgba(124,147,201,0.12); color: var(--b); }

.compare-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
@media (max-width: 720px) { .compare-grid { grid-template-columns: 1fr; } }
.compare-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 28px; }
.compare-card.highlight { border-color: var(--accent); background: linear-gradient(180deg, rgba(78,226,166,0.05), transparent); }
.compare-card h3 { font-size: 18px; margin-bottom: 16px; }
.compare-card ul { list-style: none; padding: 0; margin: 0; }
.compare-card li { padding: 8px 0; color: var(--muted); font-size: 14px; border-top: 1px solid var(--border); }
.compare-card li:first-child { border-top: none; }
.compare-card.highlight li { color: var(--text); }
.compare-card.highlight li::before { content: "\\2713  "; color: var(--accent); font-weight: 700; }

.stat-grid { display: grid; grid-template-columns: repeat(auto-fit,minmax(140px,1fr)); gap: 24px; }
.stat { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 20px; transition: border-color 0.2s, transform 0.2s; }
.stat:hover { border-color: var(--accent); transform: translateY(-2px); }
.stat .n { font-family: 'JetBrains Mono', monospace; font-size: 30px; font-weight: 700; }
.stat .l { color: var(--muted); font-size: 13px; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 6px; }

.pricing-grid { display: grid; grid-template-columns: repeat(auto-fit,minmax(280px,1fr)); gap: 20px; max-width: 720px; margin: 0 auto; }
.price-card { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 28px; position: relative; }
.price-card.rec { border-color: var(--accent); }
.price-card .rec-tag {
  position: absolute; top: -12px; right: 20px; background: var(--accent); color: #06110c;
  font-size: 13px; font-weight: 700; padding: 4px 10px; border-radius: 6px;
}
.price-card h3 { font-size: 18px; }
.price-card .desc { color: var(--muted); font-size: 13.5px; margin-top: 4px; }
.price-card .price { font-family: 'JetBrains Mono', monospace; font-size: 32px; font-weight: 700; margin-top: 18px; }
.price-card .price .per { font-size: 14px; color: var(--muted); font-weight: 400; }
.price-card ul { list-style: none; padding: 0; margin: 20px 0 0; }
.price-card li { padding: 7px 0; font-size: 14px; color: #c3c6d1; }
.price-card li::before { content: "\\2713  "; color: var(--accent); }
.coming-soon-tag {
  display: inline-block; background: var(--surface-2); border: 1px solid var(--border); color: var(--muted);
  font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em;
  padding: 4px 10px; border-radius: 6px; margin-top: 16px;
}

.faq-item { border-bottom: 1px solid var(--border); padding: 20px 0; }
.faq-item h4 { font-size: 15px; font-weight: 600; margin: 0; }
.faq-item p { color: var(--muted); font-size: 14px; margin: 10px 0 0; }

.ad-slot {
  border: 1px dashed var(--border); color: var(--muted); font-size: 13px;
  text-align: center; padding: 36px 12px; margin: 8px 0 0; border-radius: 10px;
  text-transform: uppercase; letter-spacing: 0.06em;
}

.disclaimer-box {
  background: rgba(239,95,111,0.06); border: 1px solid rgba(239,95,111,0.25);
  border-radius: 10px; padding: 16px 20px; font-size: 13.5px; color: var(--muted); margin-top: 24px;
}
.disclaimer-box strong { color: var(--red); }

footer { padding: 48px 0 60px; border-top: 1px solid var(--border); }
.footer-grid { display: flex; justify-content: space-between; flex-wrap: wrap; gap: 32px; }
.footer-col a { display: block; color: var(--muted); text-decoration: none; font-size: 13.5px; margin-bottom: 8px; }
.footer-col a:hover { color: var(--text); }
.footer-note { color: var(--muted); font-size: 13px; margin-top: 40px; max-width: 640px; }

/* live ticker strip */
.ticker-bar {
  background: #0c0d11; border-bottom: 1px solid var(--border);
  overflow: hidden; white-space: nowrap; position: relative;
}
.ticker-track {
  display: inline-flex; gap: 32px; padding: 9px 0;
  animation: ticker-scroll 45s linear infinite;
}
.ticker-bar:hover .ticker-track { animation-play-state: paused; }
@keyframes ticker-scroll {
  0% { transform: translateX(0); }
  100% { transform: translateX(-50%); }
}
.ticker-item { font-size: 13px; color: var(--muted); display: inline-flex; align-items: center; gap: 6px; flex-shrink: 0; }
.ticker-item .t-ticker { color: var(--text); font-weight: 700; }
.ticker-item .t-rating { font-weight: 700; }
.ticker-item .t-rating.Aplus { color: var(--amber); }
.ticker-item .t-rating.A { color: var(--accent); }
.ticker-item .t-rating.B { color: var(--b); }

/* live scan status */
.status-bar {
  background: var(--surface); border-bottom: 1px solid var(--border);
  padding: 7px 0; font-size: 13px; color: var(--muted);
}
.status-bar .wrap { display: flex; align-items: center; gap: 8px; }
.status-dot {
  width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0;
  background: var(--accent); box-shadow: 0 0 6px var(--accent);
  animation: statuspulse 2s ease-in-out infinite;
}
.status-dot.stale { background: var(--muted); box-shadow: none; animation: none; }
@keyframes statuspulse { 0%,100% { opacity: 1; } 50% { opacity: 0.35; } }

/* scroll-in animation, respects reduced motion */
.fade-in { opacity: 0; transform: translateY(14px); transition: opacity 0.5s ease, transform 0.5s ease; }
.fade-in.in-view { opacity: 1; transform: translateY(0); }
@media (prefers-reduced-motion: reduce) {
  .fade-in { opacity: 1; transform: none; transition: none; }
  .ticker-track { animation: none; }
  .status-dot { animation: none; }
}

.empty { color: var(--muted); padding: 24px; text-align: center; border: 1px dashed var(--border); border-radius: 10px; }

/* ------------------------------------------------------------------
   FUTURE INTELLIGENCE TERMINAL — 2026 visual system
   A restrained, institutional take on a futuristic finance product.
   ------------------------------------------------------------------ */
:root {
  --bg: #03070c;
  --surface: rgba(9, 17, 27, 0.82);
  --surface-2: rgba(15, 27, 41, 0.88);
  --surface-3: #111f2e;
  --border: rgba(137, 193, 225, 0.14);
  --border-strong: rgba(0, 224, 255, 0.34);
  --text: #edf8ff;
  --muted: #8395a8;
  --accent: #00e0ff;
  --accent-2: #6fffc1;
  --accent-dim: rgba(0, 224, 255, 0.09);
  --violet: #8f7cff;
  --red: #ff7182;
  --amber: #ffc85a;
  --aplus: var(--amber);
  --a: var(--accent-2);
  --b: #8aa7d8;
  --radius: 18px;
  --shadow: 0 24px 80px rgba(0, 0, 0, 0.38);
}

html { scroll-behavior: smooth; background: var(--bg); }
body {
  min-height: 100vh;
  background:
    radial-gradient(circle at 14% 8%, rgba(0, 224, 255, 0.09), transparent 32rem),
    radial-gradient(circle at 88% 24%, rgba(143, 124, 255, 0.08), transparent 28rem),
    linear-gradient(180deg, #03070c 0%, #050a10 42%, #03070c 100%);
  color: var(--text);
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  letter-spacing: -0.005em;
}
body::before {
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  z-index: 0;
  opacity: 0.24;
  background-image:
    linear-gradient(rgba(133, 190, 222, 0.055) 1px, transparent 1px),
    linear-gradient(90deg, rgba(133, 190, 222, 0.055) 1px, transparent 1px);
  background-size: 72px 72px;
  mask-image: linear-gradient(to bottom, black 0%, transparent 78%);
}
body::after {
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  z-index: 0;
  opacity: 0.018;
  background: repeating-linear-gradient(0deg, #fff 0, #fff 1px, transparent 1px, transparent 4px);
}

::selection { background: rgba(0, 224, 255, 0.28); color: #fff; }
a, button { -webkit-tap-highlight-color: transparent; }
a:focus-visible, button:focus-visible {
  outline: 2px solid var(--accent);
  outline-offset: 4px;
  border-radius: 6px;
}
.wrap { max-width: 1180px; padding-left: 28px; padding-right: 28px; }
h1, h2, h3 { letter-spacing: -0.045em; }

.bg-mesh span { filter: blur(120px); opacity: 0.1; }
.bg-mesh .b1 { background: var(--accent); width: 540px; height: 540px; }
.bg-mesh .b2 { background: var(--violet); opacity: 0.075; }
.bg-mesh .b3 { background: var(--accent-2); opacity: 0.06; }

nav {
  background: rgba(3, 7, 12, 0.78);
  backdrop-filter: blur(22px) saturate(145%);
  border-bottom: 1px solid rgba(137, 193, 225, 0.1);
}
.status-bar {
  background: rgba(4, 10, 16, 0.94);
  border-bottom: 1px solid rgba(137, 193, 225, 0.1);
  padding: 6px 0;
  font-family: 'JetBrains Mono', monospace;
  font-size: 12.5px;
  letter-spacing: 0.055em;
  text-transform: uppercase;
}
.status-bar .wrap::before {
  content: "SYSTEM // ";
  color: rgba(0, 224, 255, 0.62);
}
.status-dot {
  width: 7px;
  height: 7px;
  background: var(--accent-2);
  box-shadow: 0 0 0 3px rgba(111, 255, 193, 0.08), 0 0 14px var(--accent-2);
}
.ticker-bar { background: rgba(6, 13, 21, 0.88); }
.ticker-track { padding: 7px 0; gap: 38px; }
.ticker-item {
  font-family: 'JetBrains Mono', monospace;
  font-size: 12.5px;
  letter-spacing: 0.04em;
}
.ticker-item::before { content: "//"; color: rgba(0, 224, 255, 0.38); }
.ticker-item .t-ticker { color: #dff8ff; }
.ticker-item .t-rating.A { color: var(--accent-2); }
.ticker-item .t-rating.Bplus { color: #a7bdeb; }

.nav-shell {
  min-height: 74px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 28px;
}
.brand {
  flex-shrink: 0;
  color: var(--text);
  text-decoration: none;
  font-family: 'JetBrains Mono', monospace;
  font-size: 15px;
  letter-spacing: 0.08em;
}
.brand-mark {
  width: 34px;
  height: 34px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 1px solid var(--border-strong);
  border-radius: 9px;
  color: var(--accent);
  background: linear-gradient(145deg, rgba(0, 224, 255, 0.12), rgba(0, 224, 255, 0.02));
  box-shadow: inset 0 0 18px rgba(0, 224, 255, 0.08), 0 0 22px rgba(0, 224, 255, 0.07);
  font-size: 13px;
  letter-spacing: -0.03em;
}
.brand-build {
  color: var(--muted);
  font-size: 11px;
  letter-spacing: 0.1em;
  border-left: 1px solid var(--border);
  padding-left: 10px;
}
.nav-links { gap: 6px; }
.nav-links a {
  position: relative;
  color: #8ea2b5;
  font-family: 'JetBrains Mono', monospace;
  font-size: 12.5px;
  letter-spacing: 0.035em;
  text-transform: uppercase;
  padding: 10px 11px;
  border-radius: 8px;
  transition: color .2s ease, background .2s ease;
}
.nav-links a:hover { color: var(--text); background: rgba(137, 193, 225, 0.07); }
.nav-links .cta.ghost {
  border: 1px solid var(--border-strong);
  color: var(--accent);
  background: rgba(0, 224, 255, 0.055);
  padding: 10px 14px;
}

.hero { padding: 96px 0 78px; }
.home-hero { padding: 82px 0 70px; text-align: left; }
.hero-grid {
  display: grid;
  grid-template-columns: minmax(0, 1.08fr) minmax(360px, .92fr);
  gap: 72px;
  align-items: center;
}
.hero-copy { position: relative; }
.hero .eyebrow, .section-head .eyebrow {
  color: var(--accent);
  font-family: 'JetBrains Mono', monospace;
  font-size: 12.5px;
  font-weight: 600;
  letter-spacing: 0.14em;
}
.hero .eyebrow {
  border: 1px solid rgba(0, 224, 255, 0.18);
  background: rgba(0, 224, 255, 0.045);
  border-radius: 999px;
  padding: 7px 11px;
}
.eyebrow-dot {
  display: inline-block;
  width: 6px;
  height: 6px;
  border-radius: 50%;
  margin-right: 7px;
  vertical-align: 1px;
  background: var(--accent-2);
  box-shadow: 0 0 10px var(--accent-2);
}
.hero h1 {
  max-width: 920px;
  font-size: clamp(44px, 6vw, 76px);
  line-height: .98;
  text-wrap: balance;
}
.home-hero h1 { margin: 28px 0 0; max-width: 690px; }
.hero h1 .grad {
  background: linear-gradient(100deg, #effcff 3%, #62edff 46%, #70ffc1 96%);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
  animation: none;
}
.hero p {
  color: #93a7ba;
  font-size: 17px;
  line-height: 1.75;
  max-width: 650px;
}
.home-hero p { margin: 24px 0 0; max-width: 610px; }
.hero .ctas { justify-content: flex-start; margin-top: 34px; }
.cta {
  border-radius: 10px;
  background: linear-gradient(135deg, var(--accent), #5fffd0);
  color: #021015;
  box-shadow: 0 10px 34px rgba(0, 224, 255, 0.14);
}
.cta:hover { transform: translateY(-2px); }
.cta.ghost {
  color: #c8d8e5;
  background: rgba(137, 193, 225, 0.045);
  border-color: rgba(137, 193, 225, 0.2);
  box-shadow: none;
}
.system-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 9px;
  margin-top: 30px;
}
.system-chip {
  color: #8da2b6;
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: .07em;
  padding: 7px 9px;
  border: 1px solid var(--border);
  border-radius: 7px;
  background: rgba(7, 14, 22, 0.7);
}
.system-chip::before { content: "✓"; color: var(--accent-2); margin-right: 6px; }

.signal-terminal {
  position: relative;
  overflow: hidden;
  border: 1px solid rgba(0, 224, 255, 0.22);
  border-radius: 22px;
  background: linear-gradient(160deg, rgba(13, 25, 38, 0.95), rgba(5, 11, 18, 0.96));
  box-shadow: var(--shadow), inset 0 1px 0 rgba(255,255,255,.04), 0 0 70px rgba(0,224,255,.06);
}
.signal-terminal::after {
  content: "";
  position: absolute;
  left: 0;
  right: 0;
  top: -20%;
  height: 28%;
  pointer-events: none;
  background: linear-gradient(180deg, transparent, rgba(0, 224, 255, .075), transparent);
  animation: terminal-scan 5s linear infinite;
}
@keyframes terminal-scan { to { transform: translateY(520px); } }
.terminal-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 17px;
  border-bottom: 1px solid var(--border);
  background: rgba(255,255,255,.018);
  font-family: 'JetBrains Mono', monospace;
  color: #7890a5;
  font-size: 12px;
  letter-spacing: .1em;
  text-transform: uppercase;
}
.terminal-lights { display: flex; gap: 6px; }
.terminal-lights span { width: 6px; height: 6px; border-radius: 50%; background: #293847; }
.terminal-lights span:nth-child(1) { background: var(--red); }
.terminal-lights span:nth-child(2) { background: var(--amber); }
.terminal-lights span:nth-child(3) { background: var(--accent-2); box-shadow: 0 0 10px rgba(111,255,193,.7); }
.terminal-body { padding: 11px 20px 19px; }
.terminal-row {
  display: grid;
  grid-template-columns: 76px minmax(0,1fr) auto;
  align-items: center;
  gap: 14px;
  padding: 16px 0;
  border-bottom: 1px solid rgba(137, 193, 225, 0.1);
  font-family: 'JetBrains Mono', monospace;
}
.terminal-row:last-of-type { border-bottom: none; }
.terminal-step { color: #60758a; font-size: 12px; letter-spacing: .1em; }
.terminal-main strong { display: block; color: #dff6ff; font-size: 13px; letter-spacing: .03em; }
.terminal-main span { display: block; color: #647a8e; font-size: 12px; margin-top: 3px; }
.terminal-state {
  color: var(--accent-2);
  font-size: 11px;
  letter-spacing: .08em;
  border: 1px solid rgba(111, 255, 193, .2);
  background: rgba(111, 255, 193, .055);
  border-radius: 5px;
  padding: 5px 7px;
}
.terminal-output {
  margin-top: 10px;
  padding: 18px;
  border: 1px solid rgba(0, 224, 255, .18);
  border-radius: 13px;
  background: rgba(0, 224, 255, .045);
}
.terminal-output-label {
  display: flex;
  align-items: center;
  justify-content: space-between;
  color: #648095;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  letter-spacing: .1em;
  text-transform: uppercase;
}
.terminal-output-value {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 20px;
  margin-top: 9px;
}
.terminal-output-value strong { color: var(--accent); font-family: 'JetBrains Mono', monospace; font-size: 18px; }
.terminal-output-value span { color: #8298aa; font-size: 12.5px; }

.trust-bar {
  position: relative;
  border-color: rgba(137, 193, 225, 0.12);
  background: rgba(6, 13, 21, 0.52);
  backdrop-filter: blur(14px);
  padding: 0;
}
.trust-bar .wrap { padding-top: 30px; padding-bottom: 30px; }
.trust-stat { min-width: 130px; }
.trust-stat .n {
  color: #effcff;
  font-size: 25px;
  letter-spacing: -.05em;
  text-shadow: 0 0 22px rgba(0, 224, 255, .16);
}
.trust-stat .l { font-family: 'JetBrains Mono', monospace; font-size: 12px; letter-spacing: .1em; }

section { padding: 88px 0; border-color: rgba(137, 193, 225, 0.11); }
.section-head { max-width: 650px; margin-bottom: 46px; }
.section-head h2 { font-size: clamp(30px, 4vw, 44px); line-height: 1.08; margin-top: 14px; }
.section-head p { color: #7f94a7; font-size: 15px; line-height: 1.7; }

.trade-grid { gap: 16px; }
.trade-card, .compare-card, .stat, .price-card, .filings-table-wrap {
  position: relative;
  overflow: hidden;
  background: linear-gradient(155deg, rgba(16, 29, 43, .82), rgba(6, 13, 21, .86));
  border-color: rgba(137, 193, 225, .14);
  box-shadow: inset 0 1px 0 rgba(255,255,255,.025);
  backdrop-filter: blur(12px);
}
.trade-card::before, .compare-card::before, .stat::before, .price-card::before {
  content: "";
  position: absolute;
  left: 0;
  top: 0;
  width: 42px;
  height: 1px;
  background: var(--accent);
  box-shadow: 0 0 12px var(--accent);
  opacity: .65;
}
.trade-card { border-radius: var(--radius); padding: 23px; }
.trade-card:hover {
  border-color: rgba(0, 224, 255, .32);
  transform: translateY(-5px);
  box-shadow: 0 22px 55px rgba(0,0,0,.28), 0 0 34px rgba(0,224,255,.055);
}
.trade-avatar {
  border-radius: 10px;
  border-color: rgba(0, 224, 255, .18);
  background: rgba(0,224,255,.055);
  color: var(--accent);
  font-family: 'JetBrains Mono', monospace;
}
.trade-ticker { color: #e9fbff; font-size: 17px; letter-spacing: -.02em; }
.trade-meta { color: #6f8599; font-family: 'JetBrains Mono', monospace; font-size: 12.5px; }
.trade-narrative { color: #a9bac8; }
.mkt-strip { border-color: rgba(137,193,225,.12); gap: 9px; }
.mkt-cell {
  padding: 7px 8px;
  border-radius: 7px;
  background: rgba(137,193,225,.035);
}
.mkt-cell .l { color: #5f7487; }
.mkt-cell .v { color: #cbe3ee; }
.trade-badge { border: 1px solid currentColor; background: transparent !important; opacity: .92; }
.trade-badge.Aplus { color: var(--amber); }
.trade-badge.A { color: var(--accent-2); }
.trade-badge.Bplus { color: #aabcf0; }

.filings-table-wrap { border-radius: var(--radius); }
table.filings { font-size: 13px; }
table.filings thead th {
  padding-top: 14px;
  padding-bottom: 14px;
  color: #6c8295;
  background: rgba(8, 16, 25, .94);
  border-color: rgba(137, 193, 225, .12);
  letter-spacing: .09em;
}
table.filings tbody td { padding-top: 14px; padding-bottom: 14px; border-color: rgba(137,193,225,.08); }
table.filings tbody tr:hover { background: rgba(0,224,255,.035); }
table.filings .tk { color: var(--accent); }

.compare-card, .price-card { border-radius: var(--radius); }
.compare-card.highlight, .price-card.rec {
  border-color: rgba(0,224,255,.34);
  background: linear-gradient(155deg, rgba(0,224,255,.075), rgba(6,13,21,.88));
}
.compare-card li, .faq-item, .mkt-strip { border-color: rgba(137,193,225,.1); }
.stat { border-radius: 14px; }
.stat .n { color: #eafdff; font-size: 28px; letter-spacing: -.06em; }
.stat .l { font-family: 'JetBrains Mono', monospace; font-size: 12px; letter-spacing: .09em; }
.price-card .rec-tag {
  border-radius: 0 0 0 9px;
  top: 0;
  right: 0;
  background: var(--accent);
  font-family: 'JetBrains Mono', monospace;
}
.faq-item { padding: 23px 0; }
.faq-item h4 { color: #dcecf5; font-size: 16px; }
.faq-item h4::before { content: "+"; color: var(--accent); margin-right: 12px; font-family: 'JetBrains Mono', monospace; }
.empty {
  border-color: rgba(137,193,225,.18);
  border-radius: var(--radius);
  background: rgba(9,17,27,.54);
  color: #70869a;
  font-family: 'JetBrains Mono', monospace;
  font-size: 13px;
}
.ad-slot { border-color: rgba(137,193,225,.13); background: rgba(6,13,21,.42); }
.disclaimer-box {
  border-radius: 13px;
  background: rgba(255,113,130,.045);
  border-color: rgba(255,113,130,.2);
}
footer { position: relative; background: rgba(2,6,10,.54); border-color: rgba(137,193,225,.11); }
.footer-note { line-height: 1.7; }

@media (max-width: 940px) {
  .hero-grid { grid-template-columns: 1fr; gap: 46px; }
  .hero-copy { text-align: center; }
  .home-hero h1, .home-hero p { margin-left: auto; margin-right: auto; }
  .hero .ctas, .system-chips { justify-content: center; }
  .signal-terminal { max-width: 620px; margin: 0 auto; width: 100%; }
  .nav-shell { align-items: flex-start; flex-direction: column; gap: 4px; padding-top: 14px; padding-bottom: 11px; }
  .nav-links { width: 100%; overflow-x: auto; justify-content: flex-start; padding: 5px 0; scrollbar-width: none; }
  .nav-links::-webkit-scrollbar { display: none; }
  .nav-links a { white-space: nowrap; }
}
@media (max-width: 640px) {
  .wrap { padding-left: 18px; padding-right: 18px; }
  .status-bar .wrap { white-space: nowrap; overflow-x: auto; }
  .brand-build { display: none; }
  .nav-links .cta.ghost { display: none; }
  .home-hero { padding-top: 58px; }
  .hero { padding: 68px 0 55px; }
  .hero h1 { font-size: clamp(39px, 12vw, 56px); }
  .hero p { font-size: 15px; }
  .hero .ctas { flex-direction: column; }
  .hero .cta { width: 100%; text-align: center; }
  .system-chips { gap: 6px; }
  .system-chip { font-size: 11px; }
  .terminal-body { padding: 8px 14px 14px; }
  .terminal-row { grid-template-columns: 55px minmax(0,1fr); gap: 9px; }
  .terminal-state { display: none; }
  .terminal-output-value { align-items: flex-start; flex-direction: column; gap: 4px; }
  section { padding: 66px 0; }
  .section-head { text-align: left; margin-bottom: 30px; }
  .trade-grid { grid-template-columns: 1fr; }
  .mkt-strip { grid-template-columns: repeat(2,1fr); }
  .trust-bar .wrap { display: grid !important; grid-template-columns: repeat(2,1fr); gap: 24px !important; }
  .compare-grid, .pricing-grid { grid-template-columns: 1fr; }
}
@media (prefers-reduced-motion: reduce) {
  html { scroll-behavior: auto; }
  .signal-terminal::after { animation: none; display: none; }
}

/* ------------------------------------------------------------------
   INSTITUTIONAL RESEARCH UI
   Quiet, data-led, and intentionally free of sci-fi dashboard motifs.
   ------------------------------------------------------------------ */
:root {
  --bg: #0b0d0f;
  --surface: #111417;
  --surface-2: #161a1e;
  --surface-3: #1b2025;
  --border: #272c32;
  --border-strong: #3a424b;
  --text: #f2f4f5;
  --muted: #939ba4;
  --accent: #a8e063;
  --accent-2: #a8e063;
  --accent-dim: rgba(168, 224, 99, .09);
  --violet: #9b9cf5;
  --red: #ff7b86;
  --amber: #edbd62;
  --b: #91a8ca;
  --radius: 12px;
  --shadow: 0 24px 60px rgba(0, 0, 0, .22);
}

body {
  background: var(--bg);
  color: var(--text);
  letter-spacing: 0;
}
body::before, body::after, .bg-mesh { display: none; }
.wrap { max-width: 1160px; }
h1, h2, h3 { letter-spacing: -.035em; }

nav {
  background: rgba(11, 13, 15, .96);
  backdrop-filter: blur(14px);
  border-bottom: 1px solid var(--border);
}
.status-bar {
  background: #0b0d0f;
  border-color: var(--border);
  color: #747d86;
  font-size: 12px;
  letter-spacing: .075em;
}
.status-bar .wrap::before { content: ""; }
.status-dot {
  background: var(--accent);
  box-shadow: 0 0 0 3px rgba(168, 224, 99, .08);
}
.ticker-bar { background: #0e1113; border-color: var(--border); }
.ticker-track { padding: 6px 0; }
.ticker-item { color: #6f7881; }
.ticker-item::before { content: ""; }
.ticker-item .t-ticker { color: #c7ccd1; }

.nav-shell {
  min-height: 68px;
  padding-top: 0;
  padding-bottom: 0;
  flex-direction: row;
  align-items: center;
}
.brand {
  gap: 9px;
  font-family: 'Inter', sans-serif;
  font-size: 14px;
  font-weight: 750;
  letter-spacing: -.01em;
}
.brand-mark {
  width: auto;
  height: auto;
  padding-right: 9px;
  border: 0;
  border-right: 1px solid #424850;
  border-radius: 0;
  color: var(--accent);
  background: none;
  box-shadow: none;
  font-family: 'JetBrains Mono', monospace;
  font-size: 13px;
}
.brand-build { display: none; }
.nav-links { gap: 2px; }
.nav-links a {
  color: #929aa2;
  font-family: 'Inter', sans-serif;
  font-size: 13px;
  font-weight: 550;
  letter-spacing: 0;
  text-transform: none;
  padding: 9px 10px;
}
.nav-links a:hover { color: #fff; background: #171b1f; }
.nav-links .cta.ghost {
  color: #e8ebed;
  border-color: #363d44;
  background: #171b1f;
  padding: 9px 13px;
}

.hero { padding: 78px 0 58px; }
.home-hero { padding: 76px 0 68px; text-align: left; }
.hero-layout {
  display: grid;
  grid-template-columns: minmax(0, 1.05fr) minmax(390px, .95fr);
  gap: 82px;
  align-items: center;
}
.hero-copy { text-align: left; }
.hero .eyebrow {
  padding: 0;
  border: 0;
  border-radius: 0;
  background: none;
  color: var(--accent);
  font-size: 12px;
  letter-spacing: .13em;
}
.eyebrow-rule {
  display: inline-block;
  width: 22px;
  height: 1px;
  margin-right: 9px;
  vertical-align: 3px;
  background: var(--accent);
}
.hero h1 {
  max-width: 800px;
  font-size: clamp(42px, 5vw, 64px);
  line-height: 1.03;
  letter-spacing: -.055em;
}
.home-hero h1 { max-width: 650px; margin: 23px 0 0; }
.hero h1 .grad {
  background: none;
  color: inherit;
  -webkit-text-fill-color: currentColor;
}
.hero p {
  color: #a1a8af;
  font-size: 16px;
  line-height: 1.7;
}
.home-hero p { max-width: 600px; margin: 22px 0 0; }
.hero .ctas { margin-top: 30px; gap: 18px; }
.hero .cta {
  padding: 12px 18px;
  border-radius: 7px;
  font-size: 13.5px;
  box-shadow: none;
}
.cta {
  background: var(--text);
  color: #0b0d0f;
}
.cta:hover { transform: none; background: #fff; }
.cta.ghost {
  padding-left: 0 !important;
  padding-right: 0 !important;
  border: 0;
  color: #b2bac1;
  background: transparent;
}
.cta.ghost:hover { color: #fff; background: transparent; }
.hero-note {
  display: flex;
  gap: 15px;
  flex-wrap: wrap;
  margin-top: 24px;
  color: #656e77;
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  letter-spacing: .035em;
  text-transform: uppercase;
}
.hero-note span + span::before { content: "·"; margin-right: 15px; color: #41474d; }

.signal-monitor {
  border: 1px solid var(--border);
  border-radius: 13px;
  background: #101316;
  box-shadow: var(--shadow);
  overflow: hidden;
}
.monitor-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 20px;
  padding: 20px 21px;
  border-bottom: 1px solid var(--border);
}
.monitor-kicker {
  display: block;
  color: #6f7881;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  letter-spacing: .09em;
  text-transform: uppercase;
}
.monitor-head strong { display: block; margin-top: 4px; font-size: 15px; letter-spacing: -.02em; }
.monitor-live {
  color: #8e986f;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: .07em;
  white-space: nowrap;
}
.monitor-live::before {
  content: "";
  display: inline-block;
  width: 6px;
  height: 6px;
  margin-right: 6px;
  border-radius: 50%;
  background: var(--accent);
}
.monitor-row {
  display: grid;
  grid-template-columns: minmax(0,1fr) auto auto;
  align-items: center;
  gap: 18px;
  padding: 17px 21px;
  border-bottom: 1px solid #22272c;
}
.monitor-company strong, .monitor-price strong {
  display: block;
  font-family: 'JetBrains Mono', monospace;
  font-size: 13.5px;
}
.monitor-company span, .monitor-price span {
  display: block;
  margin-top: 3px;
  color: #737c85;
  font-size: 12.5px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.monitor-price { text-align: right; }
.monitor-rating {
  min-width: 33px;
  padding: 6px 7px;
  border: 1px solid #394047;
  border-radius: 6px;
  color: #cfd4d8;
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  font-weight: 700;
  text-align: center;
}
.monitor-rating.Aplus { border-color: rgba(237,189,98,.35); color: var(--amber); }
.monitor-rating.A { border-color: rgba(168,224,99,.34); color: var(--accent); }
.monitor-rating.Bplus, .monitor-rating.B { color: #9eb1cd; }
.monitor-empty { padding: 34px 21px; color: #7a838b; font-size: 13.5px; line-height: 1.6; }
.monitor-foot {
  padding: 15px 21px 17px;
  background: #0e1113;
}
.monitor-foot span {
  display: block;
  color: #606970;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: .08em;
}
.monitor-foot strong { display: block; margin-top: 5px; color: #9199a1; font-size: 12.5px; font-weight: 500; }

.trust-bar { background: #0e1113; backdrop-filter: none; }
.trust-bar .wrap { justify-content: stretch !important; gap: 0 !important; }
.trust-stat {
  flex: 1;
  min-width: 140px;
  padding: 0 26px;
  text-align: left;
  border-left: 1px solid var(--border);
}
.trust-stat:first-child { border-left: 0; padding-left: 0; }
.trust-stat .n { color: #e7eaec; font-size: 23px; text-shadow: none; }
.trust-stat .l { color: #68717a; }

section { padding: 78px 0; border-color: var(--border); }
.section-head { max-width: 680px; margin: 0 0 36px; text-align: left; }
.section-head .eyebrow { color: #77818a; font-size: 12px; }
.section-head h2 { font-size: clamp(28px, 3.5vw, 39px); }
.section-head p { color: #8f979f; max-width: 610px; }

.trade-card, .compare-card, .stat, .price-card, .filings-table-wrap {
  background: var(--surface);
  border-color: var(--border);
  box-shadow: none;
  backdrop-filter: none;
}
.trade-card::before, .compare-card::before, .stat::before, .price-card::before { display: none; }
.trade-card { border-radius: 10px; }
.trade-card:hover { transform: translateY(-2px); border-color: #3a424a; box-shadow: 0 14px 32px rgba(0,0,0,.16); }
.trade-avatar { color: #b9c0c6; background: #1a1f23; border-color: #30363c; }
.trade-ticker { color: #f1f3f4; }
.trade-meta { color: #747d85; }
.trade-narrative { color: #aeb5bb; }
.mkt-cell { background: #15191d; }
.mkt-cell .v { color: #cbd0d4; }
.trade-badge { border-radius: 5px; }

.filings-table-wrap { border-radius: 10px; }
table.filings thead th { background: #0e1113; color: #747d85; }
table.filings tbody tr:hover { background: #15191d; }
table.filings .tk { color: #e9edef; }

.compare-card, .price-card { border-radius: 10px; }
.compare-card.highlight, .price-card.rec {
  background: #121618;
  border-color: #41494f;
}
.stat { border-radius: 9px; }
.stat .n { color: #f0f2f3; }
.faq-item h4::before { color: #6e777f; }
.empty { background: #0f1214; border-color: #30363c; color: #737c84; }
.ad-slot { background: #0e1113; border-color: #2a3035; }
footer { background: #090b0d; }

.hero:not(.home-hero) { text-align: left; padding: 72px 0 54px; }
.hero:not(.home-hero) h1 { margin-left: 0; max-width: 800px; }
.hero:not(.home-hero) p { margin-left: 0; max-width: 650px; }

@media (max-width: 940px) {
  .nav-shell { flex-direction: row; align-items: center; }
  .nav-links { width: auto; }
  .nav-links a:nth-child(4), .nav-links a:nth-child(5), .nav-links a:nth-child(6), .nav-links .cta.ghost { display: none; }
  .hero-layout { grid-template-columns: 1fr; gap: 44px; }
  .hero-copy { text-align: left; }
  .home-hero h1, .home-hero p { margin-left: 0; margin-right: 0; }
  .hero .ctas { justify-content: flex-start; }
}
@media (max-width: 640px) {
  .ticker-bar { display: none; }
  .nav-shell { min-height: 62px; }
  .nav-links a:nth-child(3) { display: none; }
  .nav-links a { padding-left: 7px; padding-right: 7px; }
  .home-hero { padding-top: 56px; }
  .hero h1 { font-size: clamp(38px, 11vw, 49px); }
  .hero .ctas { flex-direction: row; align-items: center; }
  .hero .cta { width: auto; }
  .hero-note { gap: 9px; line-height: 1.7; }
  .hero-note span + span::before { margin-right: 9px; }
  .monitor-row { grid-template-columns: minmax(0,1fr) auto; }
  .monitor-price { display: none; }
  .trust-bar .wrap { grid-template-columns: repeat(2,1fr); }
  .trust-stat { padding: 10px 14px; border-left: 0; }
  .trust-stat:first-child { padding-left: 14px; }
}

/* Apple-inspired ambient glass layer: light behind the product, not over it. */
body {
  background:
    linear-gradient(180deg, rgba(8, 12, 17, .84), rgba(7, 10, 13, .97)),
    #080b0e;
}
.bg-mesh {
  display: block;
  position: fixed;
  inset: 0;
  z-index: 0;
  pointer-events: none;
  overflow: hidden;
}
.bg-mesh span {
  display: block;
  position: absolute;
  border-radius: 50%;
  filter: blur(105px);
  mix-blend-mode: screen;
  will-change: transform;
}
.bg-mesh .b1 {
  width: 680px;
  height: 680px;
  top: -300px;
  left: -240px;
  background: #087cff;
  opacity: .23;
  animation: glass-blue-drift 16s ease-in-out infinite;
}
.bg-mesh .b2 {
  width: 560px;
  height: 560px;
  top: 15%;
  right: -250px;
  background: #00ec9f;
  opacity: .17;
  animation: glass-green-drift 19s ease-in-out infinite;
}
.bg-mesh .b3 {
  width: 580px;
  height: 580px;
  bottom: -320px;
  left: 22%;
  background: #00b8ff;
  opacity: .12;
  animation: glass-lower-drift 23s ease-in-out infinite;
}
@keyframes glass-blue-drift {
  0%, 100% { transform: translate3d(-35px, -20px, 0) scale(.96); }
  25% { transform: translate3d(145px, 70px, 0) scale(1.08); }
  50% { transform: translate3d(235px, 185px, 0) scale(1.16); }
  75% { transform: translate3d(55px, 235px, 0) scale(1.03); }
}
@keyframes glass-green-drift {
  0%, 100% { transform: translate3d(45px, -40px, 0) scale(1.02); }
  25% { transform: translate3d(-120px, 55px, 0) scale(1.12); }
  50% { transform: translate3d(-235px, 175px, 0) scale(.98); }
  75% { transform: translate3d(-80px, 290px, 0) scale(1.14); }
}
@keyframes glass-lower-drift {
  0%, 100% { transform: translate3d(-90px, 80px, 0) scale(1); }
  25% { transform: translate3d(80px, -60px, 0) scale(1.1); }
  50% { transform: translate3d(260px, -150px, 0) scale(.94); }
  75% { transform: translate3d(115px, 30px, 0) scale(1.08); }
}

nav {
  background: rgba(9, 12, 16, .7);
  border-color: rgba(255, 255, 255, .075);
  backdrop-filter: blur(26px) saturate(145%);
  -webkit-backdrop-filter: blur(26px) saturate(145%);
  box-shadow: 0 1px 0 rgba(255, 255, 255, .025);
}
.status-bar, .ticker-bar {
  background: rgba(7, 10, 13, .62);
  border-color: rgba(255, 255, 255, .065);
  backdrop-filter: blur(20px) saturate(135%);
  -webkit-backdrop-filter: blur(20px) saturate(135%);
}

.signal-monitor,
.trade-card,
.compare-card,
.stat,
.price-card,
.filings-table-wrap,
.empty {
  background: linear-gradient(145deg, rgba(27, 34, 41, .7), rgba(12, 16, 21, .72));
  border-color: rgba(255, 255, 255, .1);
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, .075),
    inset 0 -1px 0 rgba(255, 255, 255, .018),
    0 24px 70px rgba(0, 0, 0, .2);
  backdrop-filter: blur(24px) saturate(135%);
  -webkit-backdrop-filter: blur(24px) saturate(135%);
}
/* The glass-card rules above group .filings-table-wrap with the cards and set
   `overflow: hidden`, which silently overrode the `overflow-x: auto` declared
   where the wrapper is first defined. The table was therefore CLIPPED, not
   scrollable: at 375px wide, 382px of it — over half — was unreachable.
   The cards need the clipping for their ::before accent bar; the table wrapper
   has no ::before, so it never did. Restore horizontal scrolling here, after
   the card rules, and keep the vertical axis clipped for the border radius. */
.filings-table-wrap {
  overflow-x: auto;
  overflow-y: hidden;
  -webkit-overflow-scrolling: touch;   /* momentum scrolling on iOS */
}
.trade-card:hover {
  border-color: rgba(119, 217, 255, .2);
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,.09),
    0 22px 55px rgba(0,0,0,.24),
    0 0 40px rgba(0, 139, 255, .035);
}
.monitor-head, .monitor-row { border-color: rgba(255, 255, 255, .075); }
.monitor-foot, table.filings thead th, .mkt-cell {
  background: rgba(8, 12, 16, .46);
}
.trust-bar {
  background: rgba(11, 15, 19, .56);
  border-color: rgba(255, 255, 255, .075);
  backdrop-filter: blur(22px) saturate(140%);
  -webkit-backdrop-filter: blur(22px) saturate(140%);
}
section, footer { border-color: rgba(255, 255, 255, .075); }
footer {
  background: rgba(6, 9, 12, .65);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
}
.nav-links .cta.ghost {
  background: rgba(255, 255, 255, .055);
  border-color: rgba(255, 255, 255, .115);
  box-shadow: inset 0 1px 0 rgba(255,255,255,.055);
}

.live-watch {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  margin-left: auto;
  color: #77818a;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  letter-spacing: .075em;
  text-transform: uppercase;
  white-space: nowrap;
}
.live-watch-dot {
  width: 5px;
  height: 5px;
  border-radius: 50%;
  background: var(--accent);
  box-shadow: 0 0 8px rgba(168, 224, 99, .55);
}
.live-watch.checking .live-watch-dot { animation: live-watch-pulse 1s ease-in-out infinite; }
.live-watch.reconnecting { color: var(--amber); }
.live-watch.reconnecting .live-watch-dot { background: var(--amber); box-shadow: none; }
@keyframes live-watch-pulse { 50% { opacity: .28; transform: scale(.75); } }

.signal-toast-region {
  position: fixed;
  right: 22px;
  bottom: 22px;
  z-index: 80;
  width: min(390px, calc(100vw - 32px));
  pointer-events: none;
}
.signal-toast {
  opacity: 0;
  transform: translateY(16px) scale(.985);
  pointer-events: auto;
  overflow: hidden;
  border: 1px solid rgba(142, 255, 209, .22);
  border-radius: 14px;
  background: linear-gradient(145deg, rgba(24, 34, 40, .9), rgba(9, 14, 18, .92));
  box-shadow: inset 0 1px 0 rgba(255,255,255,.1), 0 24px 80px rgba(0,0,0,.42), 0 0 54px rgba(0,236,159,.08);
  backdrop-filter: blur(28px) saturate(145%);
  -webkit-backdrop-filter: blur(28px) saturate(145%);
  transition: opacity .28s ease, transform .28s ease;
}
.signal-toast.show { opacity: 1; transform: translateY(0) scale(1); }
.signal-toast-accent { height: 2px; background: linear-gradient(90deg, #087cff, #00ec9f); }
.signal-toast-body { padding: 18px 19px 17px; }
.signal-toast-label {
  color: var(--accent);
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  letter-spacing: .1em;
  text-transform: uppercase;
}
.signal-toast h3 { margin-top: 7px; color: #f4f6f7; font-size: 20px; letter-spacing: -.035em; }
.signal-toast p { margin: 7px 0 0; color: #a1a9b0; font-size: 13px; line-height: 1.55; }
.signal-toast-actions { display: flex; align-items: center; gap: 14px; margin-top: 15px; }
.signal-toast-link {
  color: #0b0d0f;
  background: #f2f4f5;
  border-radius: 6px;
  padding: 8px 11px;
  font-size: 13px;
  font-weight: 700;
  text-decoration: none;
}
.signal-toast-close {
  border: 0;
  padding: 7px 0;
  color: #808991;
  background: transparent;
  font: inherit;
  font-size: 13px;
  cursor: pointer;
}
.signal-toast-close:hover { color: #fff; }

.market-dashboard {
  display: grid;
  grid-template-columns: minmax(0, 1.45fr) minmax(310px, .55fr);
  gap: 18px;
  align-items: stretch;
}
.market-chart-panel,
.market-snapshot-card {
  overflow: hidden;
  border: 1px solid rgba(255, 255, 255, .1);
  border-radius: 14px;
  background: linear-gradient(145deg, rgba(25, 33, 40, .72), rgba(10, 15, 19, .75));
  box-shadow: inset 0 1px 0 rgba(255,255,255,.075), 0 24px 70px rgba(0,0,0,.2);
  backdrop-filter: blur(24px) saturate(135%);
  -webkit-backdrop-filter: blur(24px) saturate(135%);
}
.market-panel-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 20px;
  min-height: 66px;
  padding: 17px 19px;
  border-bottom: 1px solid rgba(255,255,255,.075);
}
.market-panel-head span,
.market-data-label {
  display: block;
  color: #707a83;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  letter-spacing: .09em;
  text-transform: uppercase;
}
.market-panel-head strong { display: block; margin-top: 4px; font-size: 14px; }
.market-delay-note { max-width: 240px; color: #737c85; font-size: 12.5px; line-height: 1.45; text-align: right; }
.tradingview-widget-container { width: 100%; height: 510px; }
.tradingview-widget-container__widget { width: 100%; height: calc(100% - 26px); }
.tradingview-widget-copyright { padding: 4px 12px 7px; color: #68717a; font-size: 12px; text-align: right; }
.tradingview-widget-copyright a { color: #8d969e; text-decoration: none; }

.market-snapshot-list { display: grid; gap: 12px; }
.market-snapshot-card { padding: 19px; }
.market-snapshot-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 14px; }
.market-snapshot-symbol strong {
  display: block;
  color: #f1f4f5;
  font-family: 'JetBrains Mono', monospace;
  font-size: 17px;
}
.market-snapshot-symbol span {
  display: block;
  max-width: 220px;
  margin-top: 3px;
  overflow: hidden;
  color: #7c858d;
  font-size: 12.5px;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.market-snapshot-price { margin-top: 18px; }
.market-snapshot-price strong { display: inline-block; margin-top: 4px; color: #edf1f3; font-size: 26px; letter-spacing: -.045em; }
.market-change { margin-left: 8px; font-family: 'JetBrains Mono', monospace; font-size: 12.5px; }
.market-change.pos { color: var(--accent); }
.market-change.neg { color: var(--red); }
.market-metrics {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 7px;
  margin-top: 15px;
}
.market-metric { padding: 9px 8px; border-radius: 7px; background: rgba(7, 11, 14, .38); }
.market-metric span { display: block; color: #66717a; font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: .07em; text-transform: uppercase; }
.market-metric strong { display: block; margin-top: 4px; color: #c5cbd0; font-family: 'JetBrains Mono', monospace; font-size: 12.5px; font-weight: 500; }
.market-range { margin-top: 15px; }
.market-range-head, .market-range-labels { display: flex; justify-content: space-between; align-items: center; }
.market-range-head { color: #717b84; font-size: 12px; }
.market-range-head strong { color: #abb3ba; font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 500; }
.market-range-track { position: relative; height: 4px; margin-top: 9px; border-radius: 99px; background: rgba(255,255,255,.09); }
.market-range-fill { position: absolute; inset: 0 auto 0 0; width: var(--range-position); border-radius: inherit; background: linear-gradient(90deg, #087cff, #00ec9f); }
.market-range-pin { position: absolute; top: 50%; left: var(--range-position); width: 8px; height: 8px; border: 2px solid #dfffee; border-radius: 50%; background: #101519; transform: translate(-50%, -50%); box-shadow: 0 0 10px rgba(0,236,159,.38); }
.market-range-labels { margin-top: 7px; color: #646e77; font-family: 'JetBrains Mono', monospace; font-size: 11px; }
.market-source-note { margin: 14px 0 0; color: #69737c; font-size: 12.5px; line-height: 1.55; }

.new-trade-alert {
  position: relative;
  overflow: hidden;
  margin-bottom: 28px;
  border: 1px solid rgba(64, 226, 164, .27);
  border-radius: 14px;
  background:
    radial-gradient(circle at 88% 8%, rgba(0, 236, 159, .12), transparent 34%),
    radial-gradient(circle at 8% 100%, rgba(8, 124, 255, .1), transparent 36%),
    linear-gradient(145deg, rgba(16, 25, 31, .96), rgba(6, 12, 16, .96));
  box-shadow: 0 24px 70px rgba(0, 0, 0, .28), inset 0 1px rgba(255,255,255,.035);
}
.new-trade-alert::before {
  content: '';
  position: absolute;
  inset: 0 auto 0 0;
  width: 3px;
  background: linear-gradient(180deg, #00ec9f, #087cff);
  box-shadow: 0 0 22px rgba(0, 236, 159, .6);
}
.new-trade-alert-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 24px;
  padding: 24px 26px 20px;
  border-bottom: 1px solid rgba(255,255,255,.08);
}
.new-trade-kicker {
  display: inline-flex;
  align-items: center;
  gap: 9px;
  color: #86f2c5;
  font-family: 'JetBrains Mono', monospace;
  font-size: 12.5px;
  font-weight: 700;
  letter-spacing: .13em;
  text-transform: uppercase;
}
.new-trade-kicker::before {
  content: '';
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: #00ec9f;
  box-shadow: 0 0 12px rgba(0, 236, 159, .85);
}
.new-trade-alert-head h3 { margin: 9px 0 0; color: #f3f7f5; font-size: 30px; letter-spacing: -.045em; }
.new-trade-alert-head p { margin: 5px 0 0; color: #7f8a92; font-size: 13px; }
.new-trade-alert-badges { display: flex; align-items: center; justify-content: flex-end; gap: 8px; flex-wrap: wrap; }
.new-trade-alert-main {
  display: grid;
  grid-template-columns: minmax(190px, .72fr) minmax(0, 2.28fr);
  gap: 24px;
  padding: 25px 26px;
}
.new-trade-symbol {
  display: flex;
  flex-direction: column;
  justify-content: center;
  min-height: 128px;
  padding: 20px;
  border: 1px solid rgba(255,255,255,.075);
  border-radius: 10px;
  background: rgba(3, 8, 11, .4);
}
.new-trade-symbol span, .new-trade-stat span, .new-trade-copy-label {
  color: #69757d;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  letter-spacing: .1em;
  text-transform: uppercase;
}
.new-trade-symbol strong { margin-top: 8px; color: #f5f8f7; font-size: 43px; letter-spacing: -.055em; line-height: 1; }
.new-trade-symbol small { margin-top: 8px; color: #8a949b; font-size: 13px; line-height: 1.4; }
.new-trade-stats { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
.new-trade-stat {
  min-width: 0;
  padding: 16px 17px;
  border: 1px solid rgba(255,255,255,.07);
  border-radius: 9px;
  background: rgba(3, 8, 11, .32);
}
.new-trade-stat strong { display: block; margin-top: 7px; color: #e9edeb; font-family: 'JetBrains Mono', monospace; font-size: 18px; }
.new-trade-stat small { display: block; margin-top: 4px; color: #707b82; font-size: 12px; }
.new-trade-stat.allocation { border-color: rgba(0,236,159,.22); background: rgba(0, 100, 69, .1); }
.new-trade-stat.allocation strong { color: #6ff0bd; }
.new-trade-copy {
  display: grid;
  grid-template-columns: minmax(0, 1.45fr) minmax(260px, .55fr);
  gap: 12px;
  padding: 0 26px 24px;
}
.new-trade-copy-block {
  padding: 18px 19px;
  border: 1px solid rgba(255,255,255,.07);
  border-radius: 10px;
  background: rgba(3, 8, 11, .28);
}
.new-trade-copy-block p { margin: 9px 0 0; color: #c5cbc8; font-size: 13.5px; line-height: 1.72; }
.new-trade-copy-block.watch { border-color: rgba(255, 153, 102, .2); background: rgba(82, 35, 23, .14); }
.new-trade-copy-block.watch .new-trade-copy-label { color: #efa479; }
.new-trade-copy-block.watch p { color: #c6b5ac; }
.new-trade-risk {
  display: flex;
  align-items: flex-start;
  gap: 14px;
  margin: 0 26px 26px;
  padding: 16px 18px;
  border: 1px solid rgba(255, 112, 92, .35);
  border-radius: 10px;
  background: linear-gradient(90deg, rgba(106, 26, 22, .28), rgba(70, 30, 18, .16));
}
.new-trade-risk-icon {
  display: grid;
  flex: 0 0 auto;
  width: 28px;
  height: 28px;
  place-items: center;
  border: 1px solid rgba(255, 132, 105, .45);
  border-radius: 50%;
  color: #ff9a7e;
  font-family: 'JetBrains Mono', monospace;
  font-size: 14px;
  font-weight: 800;
}
.new-trade-risk strong { display: block; color: #ffb19c; font-size: 15px; letter-spacing: -.01em; }
.new-trade-risk p { margin: 4px 0 0; color: #bea9a2; font-size: 13px; line-height: 1.55; }
.signal-list-label { margin: 0 0 13px; color: #7a858d; font-family: 'JetBrains Mono', monospace; font-size: 12px; letter-spacing: .1em; text-transform: uppercase; }
.signal-brief-grid { display: grid; grid-template-columns: 1fr; gap: 18px; }
.signal-brief-card { overflow: hidden; padding: 0; }
.signal-brief-card:hover { transform: translateY(-2px); }
.signal-brief-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 22px;
  padding: 22px 23px 18px;
  border-bottom: 1px solid rgba(255,255,255,.075);
}
.signal-brief-symbol { display: flex; align-items: center; gap: 13px; }
.signal-brief-symbol .trade-avatar { flex: 0 0 auto; }
.signal-brief-symbol .trade-ticker { margin-top: 0; font-size: 19px; }
.signal-company { margin-top: 4px; color: #747f87; font-size: 13px; }
.signal-brief-badges { display: flex; align-items: center; justify-content: flex-end; gap: 8px; flex-wrap: wrap; }
.signal-brief-badges .trade-badge { margin-top: 0; }
.conviction-pill {
  padding: 4px 8px;
  border: 1px solid rgba(255,255,255,.11);
  border-radius: 5px;
  color: #929ba3;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  letter-spacing: .06em;
  text-transform: uppercase;
}
.signal-framework {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(145px, 1fr));
  border-bottom: 1px solid rgba(255,255,255,.075);
  background: rgba(7, 11, 14, .24);
}
.signal-framework-cell { min-width: 0; padding: 15px 20px; border-right: 1px solid rgba(255,255,255,.065); }
.signal-framework-cell:last-child { border-right: 0; }
.signal-framework-cell span { display: block; color: #68727a; font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: .08em; text-transform: uppercase; }
.signal-framework-cell strong { display: block; margin-top: 5px; color: #e6e9eb; font-family: 'JetBrains Mono', monospace; font-size: 13.5px; }
.signal-framework-cell small { display: block; margin-top: 3px; color: #68727a; font-size: 11.5px; }
.signal-brief-body { padding: 22px 23px 23px; }
.signal-insider-line { margin-bottom: 18px; color: #7c858d; font-family: 'JetBrains Mono', monospace; font-size: 12px; }
.signal-thesis-block + .signal-thesis-block { margin-top: 13px; }
.signal-thesis-label { display: flex; align-items: center; gap: 8px; color: #7ce1b4; font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: .1em; text-transform: uppercase; }
.signal-thesis-label::before { content: ''; width: 5px; height: 5px; border-radius: 50%; background: currentColor; box-shadow: 0 0 10px currentColor; }
.signal-thesis-text { margin: 8px 0 0; color: #bdc4ca; font-size: 13.5px; line-height: 1.72; }
.signal-watch {
  padding: 14px 15px;
  border: 1px solid rgba(255, 153, 102, .15);
  border-radius: 8px;
  background: rgba(82, 35, 23, .14);
}
.signal-watch .signal-thesis-label { color: #e3a076; }
.signal-watch .signal-thesis-text { color: #b9aaa2; font-size: 13px; }
.signal-brief-card .mkt-strip { margin-top: 19px; }
.signal-sizing-note { margin: 15px 0 0; color: #89939a; font-size: 13px; line-height: 1.6; }
.signal-sizing-note strong { color: #c1c8cc; font-weight: 650; }

/* Seeking Alpha-style performance dashboard, using public backtest data. */
.performance-terminal {
  position: relative;
  padding: 28px 0 18px;
  border-bottom: 0;
}
.performance-shell {
  position: relative;
  overflow: hidden;
  border: 1px solid rgba(255, 255, 255, .105);
  border-radius: 18px;
  background:
    radial-gradient(circle at 5% 0%, rgba(8, 124, 255, .12), transparent 34%),
    radial-gradient(circle at 100% 100%, rgba(0, 236, 159, .08), transparent 36%),
    linear-gradient(145deg, rgba(18, 25, 31, .86), rgba(6, 11, 15, .91));
  box-shadow: inset 0 1px 0 rgba(255,255,255,.075), 0 30px 90px rgba(0,0,0,.28);
  backdrop-filter: blur(28px) saturate(135%);
  -webkit-backdrop-filter: blur(28px) saturate(135%);
}
.performance-shell::before {
  content: '';
  position: absolute;
  inset: 0 0 auto;
  height: 2px;
  background: linear-gradient(90deg, #087cff, #00ec9f 58%, transparent);
  opacity: .82;
}
.performance-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 28px;
  padding: 27px 29px 22px;
  border-bottom: 1px solid rgba(255,255,255,.075);
}
.performance-heading .eyebrow { margin-bottom: 10px; }
.performance-heading h2 { color: #f3f6f7; font-size: clamp(25px, 3vw, 36px); letter-spacing: -.045em; }
.performance-heading p { max-width: 690px; margin: 9px 0 0; color: #88939b; font-size: 13px; line-height: 1.65; }
.performance-badges { display: flex; align-items: center; justify-content: flex-end; gap: 8px; flex-wrap: wrap; }
.performance-badge {
  padding: 7px 10px;
  border: 1px solid rgba(255,255,255,.1);
  border-radius: 999px;
  color: #9aa4ab;
  background: rgba(255,255,255,.035);
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  letter-spacing: .065em;
  text-transform: uppercase;
  white-space: nowrap;
}
.performance-badge.live { border-color: rgba(0,236,159,.2); color: #75e9b9; background: rgba(0,105,73,.1); }
.ytd-live-panel {
  display: grid;
  grid-template-columns: minmax(245px, .75fr) minmax(0, 1.6fr);
  border-bottom: 1px solid rgba(255,255,255,.075);
  background: linear-gradient(110deg, rgba(0,236,159,.075), rgba(8,124,255,.035) 44%, transparent 78%);
}
.ytd-return-card {
  position: relative;
  padding: 25px 29px 23px;
  border-right: 1px solid rgba(255,255,255,.075);
}
.ytd-return-label { color: #72dcb2; font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: .09em; text-transform: uppercase; }
.ytd-return-value { display: block; margin-top: 7px; color: #69efb7; font-family: 'JetBrains Mono', monospace; font-size: clamp(38px, 5vw, 58px); line-height: 1; letter-spacing: -.065em; text-shadow: 0 0 28px rgba(0,236,159,.16); }
.ytd-return-period { display: block; margin-top: 8px; color: #a6b1b7; font-size: 13px; }
.ytd-return-asof { display: block; margin-top: 5px; color: #68747c; font-family: 'JetBrains Mono', monospace; font-size: 11px; text-transform: uppercase; letter-spacing: .06em; }
.ytd-live-detail { display: flex; flex-direction: column; justify-content: center; padding: 23px 29px; }
.ytd-live-detail h3 { color: #e7edef; font-size: 16px; letter-spacing: -.025em; }
.ytd-live-detail > p { max-width: 720px; margin: 7px 0 0; color: #849098; font-size: 13px; line-height: 1.65; }
.ytd-live-stats { display: flex; flex-wrap: wrap; gap: 9px 24px; margin-top: 16px; }
.ytd-live-stat span { display: block; color: #66727a; font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: .08em; text-transform: uppercase; }
.ytd-live-stat strong { display: block; margin-top: 4px; color: #c7d0d4; font-family: 'JetBrains Mono', monospace; font-size: 13.5px; }
.performance-kpis {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  border-bottom: 1px solid rgba(255,255,255,.075);
}
.performance-kpis.owner-only { grid-template-columns: minmax(0, 1fr); }
.performance-kpi { padding: 19px 29px 18px; border-right: 1px solid rgba(255,255,255,.075); }
.performance-kpi:last-child { border-right: 0; }
.performance-kpi span { display: block; color: #6f7b84; font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: .085em; text-transform: uppercase; }
.performance-kpi strong { display: block; margin-top: 8px; color: #ecf1f2; font-family: 'JetBrains Mono', monospace; font-size: clamp(22px, 3vw, 31px); letter-spacing: -.045em; }
.performance-kpi strong.pos { color: #63e8b2; }
.performance-kpi strong.neg { color: #ff8278; }
.performance-kpi small { display: block; margin-top: 4px; color: #67727a; font-size: 12px; }
.performance-kpi.owner { background: linear-gradient(135deg, rgba(199,255,103,.085), rgba(0,236,159,.025)); }
.performance-kpi.owner strong { color: #d7ff7d; text-shadow: 0 0 22px rgba(199,255,103,.16); }
.performance-chart-panel { padding: 22px 29px 25px; }
.performance-chart-toolbar { display: flex; align-items: center; justify-content: space-between; gap: 20px; margin-bottom: 14px; }
.performance-window-label span { display: block; color: #67727b; font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: .08em; text-transform: uppercase; }
.performance-window-label strong { display: block; margin-top: 4px; color: #aeb7bd; font-family: 'JetBrains Mono', monospace; font-size: 12.5px; font-weight: 500; }
.performance-ranges { display: inline-flex; align-items: center; gap: 3px; padding: 4px; border: 1px solid rgba(255,255,255,.075); border-radius: 8px; background: rgba(4, 8, 11, .38); }
.performance-range {
  min-width: 42px;
  border: 0;
  border-radius: 5px;
  padding: 7px 9px;
  color: #7d8890;
  background: transparent;
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  transition: color .18s ease, background .18s ease, box-shadow .18s ease;
}
.performance-range:hover { color: #dce2e4; }
.performance-range:focus-visible { outline: 2px solid rgba(92, 189, 255, .8); outline-offset: 2px; }
.performance-range.active { color: #ecffff; background: linear-gradient(135deg, rgba(8,124,255,.28), rgba(0,236,159,.18)); box-shadow: inset 0 0 0 1px rgba(91,223,188,.17); }
.performance-plot { position: relative; height: 330px; border: 1px solid rgba(255,255,255,.055); border-radius: 11px; background: rgba(2, 7, 10, .25); }
.performance-canvas { display: block; width: 100%; height: 100%; }
.performance-tooltip {
  position: absolute;
  z-index: 3;
  display: none;
  min-width: 174px;
  padding: 11px 12px;
  border: 1px solid rgba(255,255,255,.13);
  border-radius: 8px;
  background: rgba(5, 10, 14, .92);
  box-shadow: 0 18px 46px rgba(0,0,0,.38);
  backdrop-filter: blur(16px);
  pointer-events: none;
}
.performance-tooltip.show { display: block; }
.performance-tooltip [hidden] { display: none; }
.performance-tooltip time { display: block; color: #7c8790; font-family: 'JetBrains Mono', monospace; font-size: 11px; }
.performance-tooltip div { display: flex; align-items: center; justify-content: space-between; gap: 18px; margin-top: 7px; color: #aeb6bc; font-size: 12.5px; }
.performance-tooltip strong { font-family: 'JetBrains Mono', monospace; font-size: 12.5px; }
.performance-tooltip .owner-value { color: #d7ff7d; }
.performance-tooltip .strategy-value { color: #65edb5; }
.performance-tooltip .benchmark-value { color: #75baff; }
.performance-legend { display: flex; align-items: center; gap: 20px; margin-top: 13px; color: #828c94; font-size: 12.5px; }
.performance-legend span { display: inline-flex; align-items: center; gap: 7px; }
.performance-legend i { width: 20px; height: 2px; border-radius: 99px; background: #00ec9f; box-shadow: 0 0 8px rgba(0,236,159,.25); }
.performance-legend .benchmark i { background: #4ba3ff; box-shadow: none; }
.performance-legend .owner i { background: #d7ff7d; box-shadow: 0 0 9px rgba(199,255,103,.42); }
.performance-legend.owner-mode [data-backtest-legend] { display: none; }
.performance-foot {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 24px;
  margin-top: 16px;
  padding-top: 16px;
  border-top: 1px solid rgba(255,255,255,.065);
  color: #69747c;
  font-size: 12.5px;
  line-height: 1.6;
}
.performance-risk { max-width: 590px; color: #a89289; }
.performance-risk strong { color: #e9a88d; }
.performance-asof { flex: 0 0 auto; font-family: 'JetBrains Mono', monospace; text-align: right; }
.ytd-ledger {
  margin-top: 19px;
  overflow: hidden;
  border: 1px solid rgba(255,255,255,.065);
  border-radius: 11px;
  background: rgba(2,7,10,.24);
}
.ytd-ledger-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 24px; padding: 18px 18px 15px; border-bottom: 1px solid rgba(255,255,255,.065); }
.ytd-ledger-head h3 { color: #dce3e5; font-size: 15px; letter-spacing: -.02em; }
.ytd-ledger-head p { margin: 5px 0 0; color: #6f7b83; font-size: 12.5px; line-height: 1.55; }
.ytd-ledger-updated { flex: 0 0 auto; color: #6c7880; font-family: 'JetBrains Mono', monospace; font-size: 11px; text-align: right; text-transform: uppercase; letter-spacing: .055em; }
.ytd-ledger-scroll { max-height: 390px; overflow: auto; scrollbar-color: #29343b transparent; }
.ytd-ledger-table { width: 100%; border-collapse: collapse; }
.ytd-ledger-table th { position: sticky; z-index: 1; top: 0; padding: 10px 14px; color: #65717a; background: rgba(8,14,18,.97); font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 500; letter-spacing: .08em; text-align: left; text-transform: uppercase; }
.ytd-ledger-table td { padding: 12px 14px; border-top: 1px solid rgba(255,255,255,.045); color: #89949b; font-size: 12.5px; }
.ytd-ledger-table tbody tr:first-child td { border-top: 0; }
.ytd-ledger-table tbody tr:hover { background: rgba(255,255,255,.018); }
.ytd-ledger-table .trade-date { color: #7d8991; font-family: 'JetBrains Mono', monospace; }
.ytd-ledger-table .trade-symbol { color: #e0e7e9; font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 700; }
.ytd-ledger-table .trade-rating { margin-left: 6px; color: #79bdff; font-size: 11px; }
.ytd-status { display: inline-flex; align-items: center; gap: 6px; padding: 5px 7px; border: 1px solid rgba(255,255,255,.08); border-radius: 999px; color: #9ba6ac; background: rgba(255,255,255,.025); font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: .055em; text-transform: uppercase; white-space: nowrap; }
.ytd-status::before { content: ''; width: 5px; height: 5px; border-radius: 50%; background: currentColor; }
.ytd-status.win { border-color: rgba(0,236,159,.16); color: #70dfb0; background: rgba(0,126,85,.08); }
.ytd-status.loss { border-color: rgba(255,105,96,.16); color: #ef8c84; background: rgba(121,35,31,.09); }
.ytd-status.open-signal { border-color: rgba(75,163,255,.18); color: #7fbcfa; background: rgba(26,84,143,.1); }
.ytd-trade-return { color: #aeb8bd; font-family: 'JetBrains Mono', monospace; font-weight: 600; text-align: right; }
.ytd-trade-return.pos { color: #69e9b4; }
.ytd-trade-return.neg { color: #f1877e; }
.ytd-ledger-note { padding: 12px 18px; border-top: 1px solid rgba(255,255,255,.055); color: #8e7971; font-size: 12px; line-height: 1.55; }
.ytd-ledger-note strong { color: #dda087; }
.performance-methodology { display: flex; align-items: flex-start; gap: 12px; margin-top: 17px; padding-top: 16px; border-top: 1px solid rgba(255,255,255,.065); }
.performance-methodology-mark { flex: 0 0 8px; width: 8px; height: 8px; margin-top: 5px; border-radius: 1px; background: #d7ff7d; box-shadow: 0 0 9px rgba(199,255,103,.35); transform: rotate(45deg); }
.performance-methodology p { margin: 0; color: #7c878e; font-size: 12.5px; line-height: 1.65; }
.performance-methodology strong { color: #cfd7da; font-weight: 650; }
.performance-chart-panel.ledger-only .ytd-ledger { margin-top: 0; }

@media (max-width: 760px) {
  .performance-terminal { padding-top: 18px; }
  .performance-head { flex-direction: column; padding: 23px 19px 18px; }
  .performance-badges { justify-content: flex-start; }
  .ytd-live-panel { grid-template-columns: 1fr; }
  .ytd-return-card { padding: 21px 19px 19px; border-right: 0; border-bottom: 1px solid rgba(255,255,255,.065); }
  .ytd-live-detail { padding: 19px; }
  .ytd-live-stats { gap: 12px 22px; }
  .performance-kpis { grid-template-columns: 1fr; }
  .performance-kpi { display: grid; grid-template-columns: 1fr auto; align-items: center; padding: 14px 19px; border-right: 0; border-bottom: 1px solid rgba(255,255,255,.065); }
  .performance-kpi:last-child { border-bottom: 0; }
  .performance-kpi strong { grid-column: 2; grid-row: 1 / span 2; margin: 0; font-size: 22px; }
  .performance-kpi small { grid-column: 1; }
  .performance-chart-panel { padding: 18px 12px 20px; }
  .performance-chart-toolbar { align-items: flex-start; flex-direction: column; gap: 12px; }
  .performance-ranges { width: 100%; overflow-x: auto; }
  .performance-range { flex: 1 0 42px; }
  .performance-plot { height: 270px; }
  .performance-foot { flex-direction: column; gap: 10px; }
  .performance-asof { text-align: left; }
  .ytd-ledger-head { flex-direction: column; gap: 8px; }
  .ytd-ledger-updated { text-align: left; }
  .ytd-ledger-scroll { max-height: 440px; }
  .ytd-ledger-table th, .ytd-ledger-table td { padding: 11px 10px; }
  .ytd-ledger-table th:nth-child(3), .ytd-ledger-table td:nth-child(3) { display: none; }
}

@media (prefers-reduced-motion: reduce) {
  .bg-mesh span { animation: none !important; }
  .live-watch.checking .live-watch-dot { animation: none; }
  .signal-toast { transition: none; }
}
@media (max-width: 640px) {
  .live-watch { display: none; }
  .signal-toast-region { right: 16px; bottom: 16px; }
  .market-metrics { grid-template-columns: repeat(2, 1fr); }
  .tradingview-widget-container { height: 440px; }
  .signal-brief-head { padding: 19px 17px 16px; }
  .signal-brief-body { padding: 19px 17px 20px; }
  .new-trade-alert-head { padding: 21px 18px 18px; }
  .new-trade-alert-head h3 { font-size: 25px; }
  .new-trade-alert-main { grid-template-columns: 1fr; gap: 10px; padding: 18px; }
  .new-trade-symbol { min-height: 0; }
  .new-trade-symbol strong { font-size: 36px; }
  .new-trade-copy { grid-template-columns: 1fr; padding: 0 18px 18px; }
  .new-trade-risk { margin: 0 18px 20px; }
  .signal-framework { grid-template-columns: repeat(2, 1fr); }
  .signal-framework-cell:nth-child(2) { border-right: 0; }
  .signal-framework-cell:nth-child(-n+2) { border-bottom: 1px solid rgba(255,255,255,.065); }
}
@media (max-width: 900px) {
  .market-dashboard { grid-template-columns: 1fr; }
  .market-snapshot-list { grid-template-columns: repeat(2, minmax(0,1fr)); }
}
@media (max-width: 620px) {
  .market-snapshot-list { grid-template-columns: 1fr; }
  .market-delay-note { display: none; }
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
    <div class="nav-links">
      <a href="/#performance">Dashboard</a>
      <a href="/#signals">Signals</a>
      <a href="/#market-data">Market</a>
      <a href="/track-record">Track Record</a>
      <a href="/leaderboard">Leaderboard</a>
      <a href="/how-it-works">How It Works</a>
      <a href="/#pricing">Pricing</a>
      <a class="cta ghost" href="mailto:hello@aiinsider.store?subject=Waitlist">Join waitlist</a>
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
      <h1>Insider buying, filtered for what matters.</h1>
      <p>AI Insider screens open-market purchases by executives and directors, then publishes the setups that clear a rules-based process tested across more than two decades of Form 4 history.</p>
      <div class="ctas">
        <a class="cta" href="#signals">View latest signals</a>
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

{% if trust_stats %}
<div class="trust-bar"><div class="wrap" style="display:flex;justify-content:center;gap:48px;flex-wrap:wrap;">
  <div class="trust-stat"><div class="n">{{ trust_stats.total_filings }}</div><div class="l">Filings Logged</div></div>
  <div class="trust-stat"><div class="n">{{ trust_stats.unique_tickers }}</div><div class="l">Tickers Tracked</div></div>
  <div class="trust-stat"><div class="n">{{ trust_stats.unique_insiders }}</div><div class="l">Insiders Followed</div></div>
  <div class="trust-stat"><div class="n">{{ trust_stats.days_tracked }}</div><div class="l">Days Tracked</div></div>
</div></div>
{% endif %}

<section id="filings"><div class="wrap">
  <div class="section-head">
    <span class="eyebrow">01 / Signal ledger</span>
    <h2>Latest insider filings</h2>
    <p>A dense, timestamped view of every logged signal. Public SEC Form 4 data, most recent first.</p>
  </div>
  {% if filings %}
  <div class="filings-table-wrap">
    <table class="filings">
      <thead><tr>
        <th>Ticker</th><th>Insider</th><th>Filed</th><th>Rating</th><th>Price</th><th>Mkt Cap</th><th class="amt">Amount</th>
      </tr></thead>
      <tbody>
        {% for f in filings %}
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
      </tbody>
    </table>
  </div>
  {% else %}
  <div class="empty">No filings logged yet — this table fills in as daily scans run.</div>
  {% endif %}
</div></section>

<section id="market-data"><div class="wrap">
  <div class="section-head">
    <span class="eyebrow">02 / Market context</span>
    <h2>Price action around every signal</h2>
    <p>Explore the chart, then compare it with the market conditions captured by the bot during its latest scan.</p>
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
          "symbol": "{{ signals[0].ticker if signals else 'AMEX:SPY' }}",
          "theme": "dark",
          "timezone": "Etc/UTC",
          "backgroundColor": "rgba(8, 12, 16, 0)",
          "gridColor": "rgba(255, 255, 255, 0.05)",
          "withdateranges": true,
          "autosize": true
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
        <div class="signal-framework-cell"><span>Research window</span><strong>{{ '%.0f'|format(s.max_hold) + ' days' if s.max_hold is not none else 'Open' }}</strong><small>{{ s.hold_mode|title if s.hold_mode else 'Signal review' }}</small></div>
        <div class="signal-framework-cell"><span>Exit risk rule</span><strong>{{ '$%.2f'|format(s.stop_loss) if s.stop_loss is not none else 'No fixed stop' }}</strong><small>Model parameter</small></div>
        <div class="signal-framework-cell"><span>Model allocation</span><strong>{{ '%.2f'|format(s.model_allocation_pct)|replace('.00', '') + '%' if s.model_allocation_pct is not none else 'Not set' }}</strong><small>Model portfolio</small></div>
      </div>
      <div class="signal-brief-body">
        {% if s.insider_name %}<div class="signal-insider-line">{{ s.insider_name }}{% if s.insider_title %} &middot; {{ s.insider_title }}{% endif %}{% if s.total_amount %} &middot; ${{ "{:,.0f}".format(s.total_amount) }} purchase{% endif %}{% if s.trade_date %} &middot; {{ s.trade_date }}{% endif %}</div>{% endif %}
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
    ctx.font = "11.5px 'JetBrains Mono', monospace";
    ctx.fillStyle = '#66727b';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let line = 0; line <= 4; line += 1) {
      const ratio = line / 4;
      const y = padding.top + ratio * plotHeight;
      const value = maxValue - ratio * (maxValue - minValue);
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(width - padding.right, y);
      ctx.strokeStyle = Math.abs(value) < spread / 8 ? 'rgba(255,255,255,.13)' : 'rgba(255,255,255,.055)';
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.fillText(axisPercent(value), padding.left - 8, y);
    }
    ctx.restore();

    if (!ownerMode) {
      const zeroY = yValue(0);
      const fill = ctx.createLinearGradient(0, padding.top, 0, padding.top + plotHeight);
      fill.addColorStop(0, 'rgba(0, 236, 159, .16)');
      fill.addColorStop(1, 'rgba(0, 236, 159, 0)');
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

      drawSeries(visibleRows, xFor, benchmarkY, '#4ba3ff', 1.7, [5, 4]);
      drawSeries(visibleRows, xFor, strategyY, '#00ec9f', 2.2);
    } else {
      const zeroY = yValue(0);
      const ownerFill = ctx.createLinearGradient(0, padding.top, 0, padding.top + plotHeight);
      ownerFill.addColorStop(0, 'rgba(215, 255, 125, .16)');
      ownerFill.addColorStop(1, 'rgba(215, 255, 125, 0)');
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

      drawSeries(visibleRows, xFor, benchmarkY, '#4ba3ff', 2.1);
      drawSeries(visibleRows, xFor, strategyY, '#d7ff7d', 2.5);
    }

    const labelIndexes = [0, Math.floor((visibleRows.length - 1) / 2), visibleRows.length - 1];
    ctx.save();
    ctx.font = "10.5px 'JetBrains Mono', monospace";
    ctx.fillStyle = '#647078';
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
        ctx.fillStyle = '#071014';
        ctx.fill();
        ctx.strokeStyle = '#d7ff7d';
        ctx.lineWidth = 1.4;
        ctx.stroke();
      });

      const latestOwner = visibleRows[visibleRows.length - 1];
      const latestX = xFor(visibleRows.length - 1);
      const latestY = strategyY(latestOwner);
      ctx.save();
      ctx.font = "600 12px 'JetBrains Mono', monospace";
      ctx.fillStyle = '#d7ff7d';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'bottom';
      ctx.fillText(`Owner ${signedPercent(latestOwner.strategyReturn)}`, latestX - 8, Math.max(14, latestY - 8));
      ctx.restore();

      const benchmarkLatestX = latestX;
      const benchmarkLatestY = benchmarkY(latestOwner);
      const labelsAreClose = Math.abs(latestY - benchmarkLatestY) < 24;
      ctx.save();
      ctx.font = "600 12px 'JetBrains Mono', monospace";
      ctx.fillStyle = '#75baff';
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
      ctx.strokeStyle = 'rgba(255,255,255,.18)';
      ctx.lineWidth = 1;
      ctx.stroke();
      [[strategyY(row), ownerMode ? '#d7ff7d' : '#00ec9f'], [benchmarkY(row), '#4ba3ff']].forEach(([y, color]) => {
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fillStyle = '#071014';
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
  renderRange(activeRange);
})();
</script>

""" + FOOTER.replace("{{ year }}", str(datetime.now().year)) + """
</body></html>
"""


@app.route("/")
def home():
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
    return render_template_string(
        HOME_TEMPLATE,
        signals=load_public_signals(),
        examples=load_resolved_examples(),
        filings=load_filings_table(),
        trust_stats=load_trust_stats(),
        performance=performance,
        ytd_snapshot=ytd_snapshot,
        ytd_ledger=ytd_ledger,
    )


# ── track record ─────────────────────────────────────────────────

TRACK_RECORD_TEMPLATE = """
<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#03070c">
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


# ── simple pages ─────────────────────────────────────────────────

SIMPLE_PAGE = """
<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#03070c">
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
