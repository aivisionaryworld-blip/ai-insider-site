"""Append the current public-safe signal snapshot to permanent history.

The model allocation is a percentage only; account cash, dollar allocation
and share quantity are never accepted by this public logger.
"""

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
SIGNALS_FILE = BASE_DIR / "public_signals_current.csv"
HISTORY_FILE = BASE_DIR / "public_signal_history.csv"


def main():
    if not SIGNALS_FILE.exists():
        print("No public signal snapshot found - nothing to log.")
        return
    try:
        current = pd.read_csv(SIGNALS_FILE)
    except pd.errors.EmptyDataError:
        current = pd.DataFrame()
    if current.empty:
        print("No current public signals - history left unchanged.")
        return

    current = current.rename(columns={"confirmed_entry": "entry_price"})
    if HISTORY_FILE.exists():
        try:
            history = pd.read_csv(HISTORY_FILE)
        except pd.errors.EmptyDataError:
            history = pd.DataFrame()
    else:
        history = pd.DataFrame()

    combined = pd.concat([history, current], ignore_index=True, sort=False)
    combined = combined.drop_duplicates(
        subset=["ticker", "signal_date"], keep="last"
    )
    combined.to_csv(HISTORY_FILE, index=False)
    print(f"History now contains {len(combined)} public signal(s).")


if __name__ == "__main__":
    main()
