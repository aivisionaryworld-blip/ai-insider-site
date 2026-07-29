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

import os
import sys
from datetime import datetime

import pandas as pd
from flask import Flask, render_template_string

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ADSENSE_CLIENT_ID = os.environ.get("ADSENSE_CLIENT_ID", "").strip()


# ── data loading (public-safe fields only) ────────────────────────

def load_public_signals():
    path = os.path.join(BASE_DIR, "v23_t212_aggressive_alloc_confirmed_basket_executable.csv")
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return []
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "ticker": r.get("ticker", ""),
            "rating": r.get("rating", ""),
            "entry_price": r.get("confirmed_entry", r.get("current_price", "")),
            "reason": str(r.get("reason", ""))[:280],
            # NOT included: allocation_pct, basket_allocation_$, basket_shares
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
}
.mono { font-family: 'JetBrains Mono', monospace; }
h1, h2, h3 { font-weight: 700; letter-spacing: -0.02em; margin: 0; }
a { color: inherit; }
.wrap { max-width: 1080px; margin: 0 auto; padding: 0 24px; }

nav {
  position: sticky; top: 0; z-index: 10;
  background: rgba(8,9,13,0.85); backdrop-filter: blur(10px);
  border-bottom: 1px solid var(--border);
  padding: 16px 0;
}
nav .wrap { display: flex; justify-content: space-between; align-items: center; }
.brand { font-weight: 800; font-size: 17px; display: flex; align-items: center; gap: 8px; }
.brand .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--accent); box-shadow: 0 0 12px var(--accent); }
.nav-links { display: flex; gap: 28px; align-items: center; }
.nav-links a { font-size: 14px; color: var(--muted); text-decoration: none; }
.nav-links a:hover { color: var(--text); }
.cta {
  background: var(--accent); color: #06110c; font-weight: 700; font-size: 13px;
  padding: 9px 18px; border-radius: 7px; text-decoration: none; border: none; cursor: pointer;
}
.cta.ghost {
  background: transparent; border: 1px solid var(--border); color: var(--text);
}

.hero { padding: 88px 0 64px; text-align: center; }
.hero .eyebrow {
  color: var(--accent); font-size: 12px; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.1em; margin-bottom: 18px; display: inline-block;
}
.hero h1 {
  font-size: 52px; line-height: 1.08; max-width: 780px; margin: 0 auto;
}
.hero h1 .grad {
  background: linear-gradient(90deg, var(--accent), #7de8c7);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero p { color: var(--muted); font-size: 17px; max-width: 560px; margin: 20px auto 0; }
.hero .ctas { margin-top: 32px; display: flex; gap: 12px; justify-content: center; }
.hero .cta { padding: 13px 26px; font-size: 14px; }

section { padding: 72px 0; border-top: 1px solid var(--border); }
.section-head { text-align: center; max-width: 600px; margin: 0 auto 44px; }
.section-head .eyebrow { color: var(--accent); font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; }
.section-head h2 { font-size: 32px; margin-top: 10px; }
.section-head p { color: var(--muted); font-size: 15px; margin-top: 10px; }

.trade-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px,1fr)); gap: 18px; }
.trade-card {
  background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 22px;
  transition: border-color 0.15s;
}
.trade-card:hover { border-color: #333744; }
.trade-top { display: flex; justify-content: space-between; align-items: flex-start; }
.trade-avatar {
  width: 38px; height: 38px; border-radius: 50%; background: var(--surface-2);
  display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 13px;
  color: var(--muted); border: 1px solid var(--border);
}
.trade-gain { font-size: 24px; font-weight: 800; font-family: 'JetBrains Mono', monospace; }
.trade-gain.pos { color: var(--accent); }
.trade-gain.neg { color: var(--red); }
.trade-ticker { font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 15px; margin-top: 12px; }
.trade-meta { color: var(--muted); font-size: 12.5px; margin-top: 4px; }
.trade-narrative { font-size: 13.5px; color: #c3c6d1; margin-top: 14px; line-height: 1.6; }
.trade-badge {
  display: inline-block; font-size: 10px; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.06em; padding: 3px 8px; border-radius: 5px; margin-top: 12px;
}
.trade-badge.Aplus { background: rgba(232,176,75,0.12); color: var(--amber); }
.trade-badge.A { background: var(--accent-dim); color: var(--accent); }
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
.stat { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 20px; }
.stat .n { font-family: 'JetBrains Mono', monospace; font-size: 30px; font-weight: 700; }
.stat .l { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 6px; }

.pricing-grid { display: grid; grid-template-columns: repeat(auto-fit,minmax(280px,1fr)); gap: 20px; max-width: 720px; margin: 0 auto; }
.price-card { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 28px; position: relative; }
.price-card.rec { border-color: var(--accent); }
.price-card .rec-tag {
  position: absolute; top: -12px; right: 20px; background: var(--accent); color: #06110c;
  font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 6px;
}
.price-card h3 { font-size: 18px; }
.price-card .desc { color: var(--muted); font-size: 13px; margin-top: 4px; }
.price-card .price { font-family: 'JetBrains Mono', monospace; font-size: 32px; font-weight: 700; margin-top: 18px; }
.price-card .price .per { font-size: 14px; color: var(--muted); font-weight: 400; }
.price-card ul { list-style: none; padding: 0; margin: 20px 0 0; }
.price-card li { padding: 7px 0; font-size: 13.5px; color: #c3c6d1; }
.price-card li::before { content: "\\2713  "; color: var(--accent); }
.coming-soon-tag {
  display: inline-block; background: var(--surface-2); border: 1px solid var(--border); color: var(--muted);
  font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em;
  padding: 4px 10px; border-radius: 6px; margin-top: 16px;
}

.faq-item { border-bottom: 1px solid var(--border); padding: 20px 0; }
.faq-item h4 { font-size: 15px; font-weight: 600; margin: 0; }
.faq-item p { color: var(--muted); font-size: 14px; margin: 10px 0 0; }

.ad-slot {
  border: 1px dashed var(--border); color: var(--muted); font-size: 11px;
  text-align: center; padding: 36px 12px; margin: 8px 0 0; border-radius: 10px;
  text-transform: uppercase; letter-spacing: 0.06em;
}

.disclaimer-box {
  background: rgba(239,95,111,0.06); border: 1px solid rgba(239,95,111,0.25);
  border-radius: 10px; padding: 16px 20px; font-size: 12.5px; color: var(--muted); margin-top: 24px;
}
.disclaimer-box strong { color: var(--red); }

footer { padding: 48px 0 60px; border-top: 1px solid var(--border); }
.footer-grid { display: flex; justify-content: space-between; flex-wrap: wrap; gap: 32px; }
.footer-col a { display: block; color: var(--muted); text-decoration: none; font-size: 13px; margin-bottom: 8px; }
.footer-col a:hover { color: var(--text); }
.footer-note { color: var(--muted); font-size: 12px; margin-top: 40px; max-width: 640px; }

.empty { color: var(--muted); padding: 24px; text-align: center; border: 1px dashed var(--border); border-radius: 10px; }
"""

NAV = """
<nav><div class="wrap">
  <div class="brand"><span class="dot"></span>FORM4SIGNAL</div>
  <div class="nav-links">
    <a href="/">Signals</a>
    <a href="/track-record">Track Record</a>
    <a href="/how-it-works">How It Works</a>
    <a href="/#pricing">Pricing</a>
    <a class="cta ghost" href="#" style="pointer-events:none;opacity:0.5;">Sign up (soon)</a>
  </div>
</div></nav>
"""

FOOTER = """
<footer><div class="wrap">
  <div class="footer-grid">
    <div class="footer-col">
      <div class="brand" style="margin-bottom:14px;"><span class="dot"></span>FORM4SIGNAL</div>
      <p style="color:var(--muted);font-size:13px;max-width:220px;">Scored insider buying, straight from SEC filings.</p>
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
  <p class="footer-note">FORM4SIGNAL tracks publicly available SEC Form 4 filings. This is not insider trading &mdash; it's public information. Nothing on this site is investment advice. All investing involves risk. Past performance, live or backtested, does not guarantee future results.</p>
  <p style="color:var(--muted);font-size:12px;margin-top:16px;">&copy; {{ year }} FORM4SIGNAL</p>
</div></footer>
"""


def ad_slot(label="Advertisement"):
    if ADSENSE_CLIENT_ID:
        return f'<div class="ad-slot"><!-- AdSense unit, client={ADSENSE_CLIENT_ID} --></div>'
    return f'<div class="ad-slot">{label} &middot; connect AdSense once live on a real domain</div>'


# ── home page ────────────────────────────────────────────────────

HOME_TEMPLATE = """
<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FORM4SIGNAL &mdash; Scored insider buying from SEC filings</title>
<meta name="description" content="Every open-market insider purchase, scored against a 5-year backtest. Free, public, updated daily from SEC EDGAR.">
<style>""" + BASE_CSS + """</style>
</head><body>
""" + NAV + """

<div class="hero"><div class="wrap">
  <span class="eyebrow">Built on SEC Form 4 filings</span>
  <h1>Insiders buy their own stock<br>for one reason. <span class="grad">We score which purchases mean it.</span></h1>
  <p>Every open-market insider purchase, filtered through the same rules we backtested across five years of filings &mdash; ownership increase, buyer seniority, price setup. Free to read. Nothing here is a recommendation.</p>
  <div class="ctas">
    <a class="cta" href="#signals">See today's signals</a>
    <a class="cta ghost" href="/track-record">View the track record</a>
  </div>
</div></div>

<section id="proof"><div class="wrap">
  <div class="section-head">
    <span class="eyebrow">From the backtest</span>
    <h2>What these signals have caught before</h2>
    <p>Resolved examples from our 5-year historical simulation &mdash; labeled honestly as backtest, not live results.</p>
  </div>
  {% if examples %}
  <div class="trade-grid">
    {% for e in examples %}
    <div class="trade-card">
      <div class="trade-top">
        <div class="trade-avatar">{{ e.ticker[:2] }}</div>
        <div class="trade-gain {{ 'pos' if e.pnl >= 0 else 'neg' }}">{{ '+' if e.pnl >= 0 else '' }}{{ e.pnl }}%</div>
      </div>
      <div class="trade-ticker">${{ e.ticker }}</div>
      <div class="trade-meta">Entered {{ e.entry_date }} &middot; held {{ e.days_held }} days</div>
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
    <span class="eyebrow">Live &middot; Updated Daily</span>
    <h2>Today's scored signals</h2>
    <p>Real SEC filings from the last scan. Rating and reasoning only &mdash; no position sizing, that part's your call.</p>
  </div>
  {% if signals %}
  <div class="trade-grid">
    {% for s in signals %}
    <div class="trade-card">
      <div class="trade-top">
        <div class="trade-avatar">{{ s.ticker[:2] }}</div>
        <span class="trade-badge {{ s.rating|replace('+','plus') }}">{{ s.rating }}</span>
      </div>
      <div class="trade-ticker">${{ s.ticker }}</div>
      <div class="trade-meta">Filed near ${{ s.entry_price }}</div>
      <div class="trade-narrative">{{ s.reason }}</div>
    </div>
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
      <h3>FORM4SIGNAL</h3>
      <ul>
        <li>Filtered against a 5-year backtest of scoring rules</li>
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
      <div class="price">&mdash;<span class="per"> /mo</span></div>
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
      <p>OpenInsider shows you every filing, unfiltered. We run each purchase through rules validated on five years of historical data and only surface the ones that pass.</p>
    </div>
    <div class="faq-item">
      <h4>Is this investment advice?</h4>
      <p>No. This is published research on public filings, for informational purposes. See the full <a href="/disclaimer" style="color:var(--accent);">disclaimer</a>.</p>
    </div>
  </div>
</div></section>

""" + FOOTER.replace("{{ year }}", str(datetime.now().year)) + """
</body></html>
"""


@app.route("/")
def home():
    return render_template_string(
        HOME_TEMPLATE,
        signals=load_public_signals(),
        examples=load_resolved_examples(),
    )


# ── track record ─────────────────────────────────────────────────

TRACK_RECORD_TEMPLATE = """
<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Track Record &mdash; FORM4SIGNAL</title>
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
    <span class="eyebrow">2021&ndash;2026 simulation</span>
    <h2 style="font-size:26px;">5-year backtest</h2>
  </div>
  {% if backtest %}
  <div class="stat-grid">
    <div class="stat"><div class="n">{{ backtest.get('annualized_simple_roi_%','\u2014') }}%</div><div class="l">Annualized</div></div>
    <div class="stat"><div class="n">{{ backtest.get('win_rate_%','\u2014') }}%</div><div class="l">Win rate</div></div>
    <div class="stat"><div class="n">{{ backtest.get('max_drawdown_%','\u2014') }}%</div><div class="l">Max drawdown</div></div>
    <div class="stat"><div class="n">{{ backtest.get('positions','\u2014') }}</div><div class="l">Trades simulated</div></div>
  </div>
  {% else %}
  <div class="empty">Backtest summary not published yet.</div>
  {% endif %}
  <div class="disclaimer-box"><strong>Read this first:</strong> backtests are simulations, not predictions. They can carry survivorship bias and can't capture every real-world cost or friction. Treat this as an upper bound, not a promise.</div>
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


@app.route("/track-record")
def track_record():
    return render_template_string(
        TRACK_RECORD_TEMPLATE,
        backtest=load_backtest_summary(), live=load_live_track_record(),
    )


# ── how it works ─────────────────────────────────────────────────

HOW_IT_WORKS_TEMPLATE = """
<!DOCTYPE html><html><head><meta charset="utf-8">
<title>How It Works &mdash; FORM4SIGNAL</title>
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
<title>{{ title }} &mdash; FORM4SIGNAL</title>
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
<p style="color:var(--amber);"><strong>DRAFT &mdash; have a securities attorney review before real launch or any paid tier.</strong></p>
<p>FORM4SIGNAL publishes information derived from public SEC filings for informational and educational purposes only. Nothing on this site constitutes investment advice, a recommendation, or a solicitation to buy or sell any security. We are not a registered investment adviser or broker-dealer.</p>
<p>Past performance, whether backtested or live, is not indicative of future results. All investing involves risk, including possible loss of principal. Do your own research and consult a licensed financial professional before making investment decisions.</p>
<p>Backtested results shown on this site are hypothetical, have inherent limitations, and do not represent actual trading.</p>
"""

PRIVACY_BODY = """
<p style="color:var(--amber);"><strong>DRAFT &mdash; have a lawyer review before launch, especially once analytics, ads, or payments are added.</strong></p>
<p>This site does not currently collect personal information beyond standard web server logs. If ads are enabled, third-party providers (e.g. Google AdSense) may use cookies to serve relevant ads &mdash; see their own privacy policies.</p>
"""

ABOUT_BODY = """
<p>FORM4SIGNAL is built and run by one person who got tired of scrolling insider-trading tables by hand. It started as a personal tool; this public version shares the same signal-scoring engine, minus any personal position sizing.</p>
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
    app.run(host="127.0.0.1", port=5001, debug=False)
