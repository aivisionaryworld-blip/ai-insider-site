"""
returns_engine.py — replaces the misleading "annualized_simple_roi_%" figure
with two properly-computed, industry-standard metrics, both derived live
from bt_v24_equity.csv (never hardcoded):

  XIRR  — money-weighted return. Correctly accounts for WHEN each deposit
          happened (a dollar deposited in month 1 has more time to compound
          than one deposited in month 59 — simple ROI/years ignores this
          entirely, which is exactly why the old "216.74%" figure was wrong).

  TWR   — time-weighted return, annualized. Chain-links daily returns and
          strips out the effect of deposit TIMING/SIZE, isolating the
          performance of the STRATEGY itself, independent of your cash-flow
          schedule. This is the number that's actually comparable to a
          benchmark like SPY.

Both are computed from real cash-flow events reconstructed from the equity
CSV's own deposited_$ column (start capital + every month it increases),
not assumed or hardcoded.
"""

from datetime import datetime, date
import csv


def _parse_date(s):
    return datetime.strptime(s, "%Y-%m-%d").date()


def load_equity_series(path):
    """Returns list of dicts: date, cash, invested, equity, deposited."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "date": _parse_date(r["date"]),
                "equity": float(r["equity_$"]),
                "deposited": float(r["deposited_$"]),
            })
    rows.sort(key=lambda x: x["date"])
    return rows


def reconstruct_cashflows(equity_series):
    """Rebuild the actual deposit events from the cumulative deposited_$
    column — a cash outflow (from the investor's perspective, negative)
    on every day the cumulative deposited total increases, plus a final
    positive inflow of the ending equity (a hypothetical "cash out" on the
    last day, standard XIRR convention for an ongoing position)."""
    flows = []
    last_deposited = 0.0
    for row in equity_series:
        delta = row["deposited"] - last_deposited
        if delta > 0.01:
            flows.append((row["date"], -delta))
            last_deposited = row["deposited"]
    if equity_series:
        flows.append((equity_series[-1]["date"], equity_series[-1]["equity"]))
    return flows


def xirr(cashflows, guess=0.15):
    """Standard XIRR via Newton-Raphson on the flows' actual dates.
    cashflows: list of (date, amount) — outflows negative, inflows positive.
    Pure Python, no scipy dependency (keeps deployment simple)."""
    if not cashflows:
        return None
    t0 = cashflows[0][0]

    def npv(rate):
        total = 0.0
        for d, amt in cashflows:
            years = (d - t0).days / 365.25
            total += amt / ((1 + rate) ** years)
        return total

    def dnpv(rate):
        total = 0.0
        for d, amt in cashflows:
            years = (d - t0).days / 365.25
            if years == 0:
                continue
            total += -years * amt / ((1 + rate) ** (years + 1))
        return total

    rate = guess
    for _ in range(100):
        f = npv(rate)
        df = dnpv(rate)
        if abs(df) < 1e-12:
            break
        new_rate = rate - f / df
        if abs(new_rate - rate) < 1e-8:
            return new_rate
        rate = new_rate
        if rate <= -0.999:  # guard against divergence
            rate = -0.9
    return rate


def time_weighted_return(equity_series):
    """Daily-linked TWR, annualized. Strips out deposit size/timing so this
    reflects the STRATEGY's return, comparable to a benchmark like SPY."""
    if len(equity_series) < 2:
        return None
    chain = 1.0
    for i in range(1, len(equity_series)):
        prev = equity_series[i - 1]
        curr = equity_series[i]
        deposit_today = curr["deposited"] - prev["deposited"]
        # sub-period return excludes today's deposit from the gain calc
        denom = prev["equity"]
        if denom <= 0:
            continue
        r = (curr["equity"] - deposit_today - denom) / denom
        chain *= (1 + r)
    total_days = (equity_series[-1]["date"] - equity_series[0]["date"]).days
    years = total_days / 365.25
    if years <= 0:
        return None
    cumulative = chain - 1
    annualized = chain ** (1 / years) - 1
    return {"cumulative_pct": cumulative * 100, "annualized_pct": annualized * 100, "years": years}


def compute_all_returns(equity_csv_path):
    """Single entry point the website calls. Returns a dict of clean,
    separately-labeled metrics — never a single conflated 'annualized' figure."""
    series = load_equity_series(equity_csv_path)
    if len(series) < 2:
        return None

    flows = reconstruct_cashflows(series)
    money_weighted = xirr(flows)
    time_weighted = time_weighted_return(series)

    total_deposited = series[-1]["deposited"]
    final_equity = series[-1]["equity"]
    profit = final_equity - total_deposited
    cumulative_roi = (profit / total_deposited * 100) if total_deposited > 0 else None

    return {
        "final_equity": round(final_equity, 2),
        "total_deposited": round(total_deposited, 2),
        "profit": round(profit, 2),
        "cumulative_roi_pct": round(cumulative_roi, 2) if cumulative_roi is not None else None,
        "xirr_pct": round(money_weighted * 100, 2) if money_weighted is not None else None,
        "twr_annualized_pct": round(time_weighted["annualized_pct"], 2) if time_weighted else None,
        "twr_cumulative_pct": round(time_weighted["cumulative_pct"], 2) if time_weighted else None,
        "years": round(time_weighted["years"], 2) if time_weighted else None,
    }


METRIC_EXPLANATIONS = {
    "cumulative_roi_pct": "Total profit divided by total deposited, over the whole window. Doesn't account for WHEN money was deposited — a simple headline number, not a rate of return.",
    "xirr_pct": "Money-weighted annual return. Accounts for the exact date and size of every deposit, same method used for real portfolio performance with contributions over time. This is the closest thing to 'what rate of return did the money actually earn.'",
    "twr_annualized_pct": "Time-weighted annual return. Strips out the effect of deposit timing entirely, isolating the STRATEGY's own performance. This is the number that's fair to compare against a benchmark like SPY, since benchmarks don't have your deposit schedule.",
}


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "bt_v24_equity.csv"
    result = compute_all_returns(path)
    if result is None:
        print(f"Could not compute — check that {path} exists and has >= 2 rows.")
    else:
        for k, v in result.items():
            print(f"{k}: {v}")
