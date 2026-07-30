"""
public_signal_resolver.py — checks every signal in public_signal_history.csv
against REAL current/historical prices and resolves it: still open, hit
target (WIN), or expired without hitting target (LOSS/EXPIRED).

Uses the SAME exit rules already validated in the backtest — no new rules
invented here:
    A+  : no fixed stop, target = entry * 1.15 minimum, 120-day cap
    A/B : target = entry * 1.10, 30-day cap, no fixed stop

Run this WEEKLY (not daily — resolution takes days/weeks, no need to check
more often). Needs internet + yfinance, so run it on your PC, not anywhere
without network access.

Output: public_signal_outcomes.csv — upload this to GitHub for the
leaderboard to have real data.
"""

import os
import time
from datetime import datetime, timedelta

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE = os.path.join(BASE_DIR, "public_signal_history.csv")
OUTCOMES_FILE = os.path.join(BASE_DIR, "public_signal_outcomes.csv")

TARGET_PCT_STANDARD = 0.10
TARGET_PCT_RUNNER = 0.15
MAX_HOLD_STANDARD = 30
MAX_HOLD_RUNNER = 120


def resolve_one(ticker, rating, entry_price, signal_date_str):
    """Returns dict with status/pnl, or None if price data unavailable."""
    if yf is None:
        return None
    try:
        signal_date = datetime.strptime(signal_date_str, "%Y-%m-%d")
        entry_price = float(entry_price)
        if entry_price <= 0:
            return None

        is_runner = str(rating).strip() == "A+"
        target_mult = 1 + (TARGET_PCT_RUNNER if is_runner else TARGET_PCT_STANDARD)
        max_days = MAX_HOLD_RUNNER if is_runner else MAX_HOLD_STANDARD
        target_price = entry_price * target_mult

        days_elapsed = (datetime.now() - signal_date).days
        end_check = min(datetime.now(), signal_date + timedelta(days=max_days))

        hist = yf.Ticker(ticker).history(
            start=signal_date.strftime("%Y-%m-%d"),
            end=(end_check + timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=False,
        )
        if hist is None or hist.empty:
            return None

        hit_target_date = None
        for date, bar in hist.iterrows():
            if float(bar.get("High", 0)) >= target_price:
                hit_target_date = date
                break

        if hit_target_date is not None:
            pnl_pct = (target_price - entry_price) / entry_price * 100
            return {"status": "WIN", "pnl_pct": round(pnl_pct, 2),
                    "resolved_date": hit_target_date.strftime("%Y-%m-%d"),
                    "exit_price": round(target_price, 4)}

        if days_elapsed >= max_days:
            last_close = float(hist["Close"].iloc[-1])
            pnl_pct = (last_close - entry_price) / entry_price * 100
            status = "WIN" if pnl_pct > 0 else "LOSS"
            return {"status": status, "pnl_pct": round(pnl_pct, 2),
                    "resolved_date": end_check.strftime("%Y-%m-%d"),
                    "exit_price": round(last_close, 4)}

        current_price = float(hist["Close"].iloc[-1])
        pnl_pct = (current_price - entry_price) / entry_price * 100
        return {"status": "OPEN", "pnl_pct": round(pnl_pct, 2),
                "resolved_date": None, "exit_price": None}

    except Exception as e:
        print(f"  {ticker}: error resolving — {e}")
        return None


def main():
    if yf is None:
        print("yfinance not installed — run: pip install yfinance")
        return
    if not os.path.exists(HISTORY_FILE):
        print(f"{HISTORY_FILE} not found — run public_signal_logger.py for a "
              f"few days first to build up signal history.")
        return

    history = pd.read_csv(HISTORY_FILE)
    if history.empty:
        print("Signal history is empty — nothing to resolve yet.")
        return

    print(f"Resolving {len(history)} historical signals against real prices...")
    rows = []
    for i, r in history.iterrows():
        result = resolve_one(r["ticker"], r["rating"], r["entry_price"], r["signal_date"])
        time.sleep(0.3)
        if result is None:
            continue
        rows.append({
            "ticker": r["ticker"], "rating": r["rating"],
            "signal_date": r["signal_date"], "entry_price": r["entry_price"],
            **result,
        })
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(history)} processed...")

    out = pd.DataFrame(rows)
    out.to_csv(OUTCOMES_FILE, index=False)

    resolved = out[out["status"] != "OPEN"]
    print(f"\nDone. {len(out)} signals checked, {len(resolved)} resolved "
          f"({len(out) - len(resolved)} still open/in progress).")
    if not resolved.empty:
        print(f"Resolved win rate so far: {(resolved['status']=='WIN').mean()*100:.1f}%")
    print(f"Saved -> {OUTCOMES_FILE}")
    print("Upload this file to GitHub to update the public leaderboard.")


if __name__ == "__main__":
    main()
