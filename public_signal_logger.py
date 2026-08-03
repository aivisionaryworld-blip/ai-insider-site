"""
public_signal_logger.py — appends today's PUBLIC-SAFE signals to a
permanent, growing history file. Without this, the public site only ever
shows "today," and nothing (a leaderboard, a real track record) can be
built on top of it.

Run this once per day, right after your normal scan
(run_and_send_aggressive_fixed_v2.py). Safe to run multiple times on the
same day — it won't create duplicate rows for the same ticker+date.

Output: public_signal_history.csv — this is the file you upload to GitHub
alongside the others (same drag-and-drop process you already know).
"""

import os
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIGNALS_FILE = os.path.join(BASE_DIR, "v23_t212_aggressive_alloc_confirmed_basket_executable.csv")
HISTORY_FILE = os.path.join(BASE_DIR, "public_signal_history.csv")


def main():
    if not os.path.exists(SIGNALS_FILE):
        print("No signal file found from today's scan — nothing to log.")
        return

    try:
        today = pd.read_csv(SIGNALS_FILE)
    except pd.errors.EmptyDataError:
        print("Today's signal file is empty — nothing to log.")
        return

    if today.empty:
        print("No signals today — nothing to log.")
        return

    today_date = datetime.now().strftime("%Y-%m-%d")

    rows = []
    for _, r in today.iterrows():
        rows.append({
            "ticker": r.get("ticker", ""),
            "rating": r.get("rating", ""),
            "signal_date": today_date,
            "entry_price": r.get("confirmed_entry", r.get("current_price", "")),
            "reason": str(r.get("reason", ""))[:400],
            "insider_name": r.get("insider_name", ""),
            "insider_title": r.get("insider_title", ""),
            "total_amount": r.get("total_amount", 0) or 0,
            "market_cap": r.get("market_cap"),
            "rsi14": r.get("rsi14"),
            "low_52w": r.get("low_52w"),
            "high_52w": r.get("high_52w"),
            "ret_20d_pct": r.get("ret_20d_pct"),
            # deliberately NOT included: allocation_pct, basket_allocation_$,
            # basket_shares — same public-safety rule as the website itself
        })
    new_rows = pd.DataFrame(rows)

    if os.path.exists(HISTORY_FILE):
        history = pd.read_csv(HISTORY_FILE)
        # de-dupe: same ticker + same signal_date already logged
        existing_keys = set(zip(history["ticker"], history["signal_date"]))
        new_rows = new_rows[~new_rows.apply(
            lambda r: (r["ticker"], r["signal_date"]) in existing_keys, axis=1)]
        combined = pd.concat([history, new_rows], ignore_index=True)
    else:
        combined = new_rows

    combined.to_csv(HISTORY_FILE, index=False)
    print(f"Logged {len(new_rows)} new signal(s). "
          f"Total history: {len(combined)} signals across all days.")
    print(f"Saved -> {HISTORY_FILE}")
    print("Remember to upload this file to GitHub to update the public site.")


if __name__ == "__main__":
    main()
