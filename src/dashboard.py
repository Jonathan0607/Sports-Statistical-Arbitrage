import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import psycopg2
import os
import sys
import logging
import math
from datetime import datetime, timezone

# Ensure root directory is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Dashboard")

# ══════════════════════════════════════════════════════════
# 1. STREAMLIT PAGE CONFIGURATION & THEMING
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="StatArb Trading Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Institutional Cyber-Quant CSS Injection
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=JetBrains+Mono:wght@400;700&display=swap');
        
        /* Global Font Overrides */
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        code, pre, .api-status {
            font-family: 'JetBrains Mono', monospace !important;
        }

        /* Deep Institutional Background */
        .stApp {
            background-color: #06090e;
            color: #f8fafc;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #0a0f18 !important;
            border-right: 1px solid rgba(59, 130, 246, 0.15);
        }

        /* High-Tech Metric Cards */
        div[data-testid="metric-container"] {
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.8) 0%, rgba(15, 23, 42, 0.4) 100%);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-top: 2px solid #3b82f6;
            border-radius: 8px;
            padding: 20px 24px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
            backdrop-filter: blur(12px);
            transition: all 0.3s ease;
        }
        div[data-testid="metric-container"]:hover {
            border-top: 2px solid #10b981;
            box-shadow: 0 10px 30px rgba(16, 185, 129, 0.15);
            transform: translateY(-2px);
        }

        /* Metric Text Overrides */
        div[data-testid="metric-container"] label {
            color: #64748b !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            font-size: 0.75rem !important;
            letter-spacing: 0.08em;
        }
        
        /* Custom Headers */
        .terminal-header {
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(135deg, #f8fafc 0%, #94a3b8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 2px;
            letter-spacing: -0.03em;
        }

        .terminal-subheader {
            color: #3b82f6;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 30px;
        }

        .section-title {
            font-size: 1.2rem;
            font-weight: 600;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            padding-bottom: 10px;
            margin-top: 20px;
            margin-bottom: 20px;
            color: #e2e8f0;
        }

        /* Animated Pulsing Status Indicators */
        .pulse-green {
            display: inline-block;
            width: 8px;
            height: 8px;
            background-color: #10b981;
            border-radius: 50%;
            margin-right: 12px;
            box-shadow: 0 0 8px #10b981;
            animation: pulse 2s infinite;
        }
        .pulse-amber {
            display: inline-block;
            width: 8px;
            height: 8px;
            background-color: #f59e0b;
            border-radius: 50%;
            margin-right: 12px;
            box-shadow: 0 0 8px #f59e0b;
        }
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
            70% { box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
            100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
        }

        .api-status {
            font-size: 0.85rem;
            color: #94a3b8;
            margin-bottom: 14px;
            display: flex;
            align-items: center;
        }
        
        .api-status strong {
            color: #e2e8f0;
            margin-right: 6px;
        }

        /* Style pandas dataframe in streamlit */
        .stDataFrame {
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 6px;
        }
    </style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# 2. DATA UTILITIES & CONNECTIONS
# ══════════════════════════════════════════════════════════
DB_URI = os.getenv("POSTGRES_URI", "postgresql://postgres:password@localhost:5432/sports_db")

def get_redis_connection_status() -> bool:
    """Check if Redis server is reachable."""
    try:
        from infrastructure.redis_client import RedisClient
        return RedisClient().ping()
    except Exception:
        return False

def get_postgres_connection_status() -> bool:
    """Check if Postgres server is reachable."""
    try:
        conn = psycopg2.connect(DB_URI, connect_timeout=3)
        conn.close()
        return True
    except Exception:
        return False

def fetch_live_signals() -> tuple[pd.DataFrame, bool]:
    """Fetches live arbitrage signals from Redis cache."""
    try:
        from infrastructure.redis_client import RedisClient
        r = RedisClient().get_client()
        keys = r.keys("signal:*")
        
        signals = []
        for k in keys:
            data = r.hgetall(k)
            if data:
                def get_field(d, *possible_keys, default=""):
                    for pk in possible_keys:
                        if pk in d: return d[pk]
                        if pk.lower() in d: return d[pk.lower()]
                        underscore_k = pk.replace(" ", "_")
                        if underscore_k in d: return d[underscore_k]
                        if underscore_k.lower() in d: return d[underscore_k.lower()]
                    return default

                signals.append({
                    "Player": get_field(data, "Player"),
                    "Market": get_field(data, "Market", "market"),
                    "Side": get_field(data, "Side", "side"),
                    "Sharp Line": float(get_field(data, "Sharp Line", default=0.0)),
                    "Retail Line": float(get_field(data, "Retail Line", default=0.0)),
                    "True Prob": float(get_field(data, "True Prob", default=0.0)),
                    "EV Edge": float(get_field(data, "EV Edge", default=0.0)),
                    "Raw Kelly": float(get_field(data, "Raw Kelly", default=0.0)),
                    "Masked Size": float(get_field(data, "Masked Size", default=0.0)),
                    "Book": get_field(data, "Book"),
                    "Sharp Book": get_field(data, "Sharp Book"),
                    "Anchor Liquidity USD": get_field(data, "Anchor Liquidity USD"),
                    "timestamp": get_field(data, "timestamp")
                })
        if signals:
            return pd.DataFrame(signals), False
    except Exception as e:
        logger.error(f"Redis Live Signals fetch failed: {e}")
        
    return pd.DataFrame(columns=[
        "Player", "Market", "Side", "Sharp Line", "Retail Line", 
        "True Prob", "EV Edge", "Raw Kelly", "Masked Size", 
        "Book", "Sharp Book", "Anchor Liquidity USD", "timestamp"
    ]), True

def fetch_historical_signals() -> tuple[pd.DataFrame, bool]:
    """Queries historical arb signals from PostgreSQL hypertable."""
    try:
        import sqlalchemy
        adjusted_uri = DB_URI
        if adjusted_uri.startswith("postgres://"):
            adjusted_uri = adjusted_uri.replace("postgres://", "postgresql+psycopg2://", 1)
        engine = sqlalchemy.create_engine(adjusted_uri)
        query = """
            SELECT 
                timestamp, player_name as "Player", sharp_book as "Sharp Book",
                stale_book as "Book", sharp_line as "Sharp Line", stale_line as "Retail Line",
                expected_value as "EV Edge", kelly_fraction as "Raw Kelly", anchor_liquidity_usd as "Anchor Liquidity USD"
            FROM arb_signals ORDER BY timestamp ASC;
        """
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        if not df.empty:
            df["EV Edge"] = df["EV Edge"].astype(float)
            df["Raw Kelly"] = df["Raw Kelly"].astype(float)
            if "Anchor Liquidity USD" in df.columns:
                df["Anchor Liquidity USD"] = pd.to_numeric(df["Anchor Liquidity USD"], errors='coerce')
            return df, False
    except Exception as e:
        logger.error(f"Postgres Historical Signals fetch failed: {e}")
        
    return pd.DataFrame(columns=[
        "timestamp", "Player", "Sharp Book", "Book", "Sharp Line", 
        "Retail Line", "EV Edge", "Raw Kelly", "Anchor Liquidity USD"
    ]), True

def fetch_portfolio_metrics() -> pd.DataFrame:
    """Queries settled wagers from trade_execution table in PostgreSQL."""
    try:
        import sqlalchemy
        adjusted_uri = DB_URI
        if adjusted_uri.startswith("postgres://"):
            adjusted_uri = adjusted_uri.replace("postgres://", "postgresql+psycopg2://", 1)
        engine = sqlalchemy.create_engine(adjusted_uri)
        query = """
            SELECT 
                trade_id, timestamp, player_name, bookmaker, bet_type, line_value, odds,
                stake_amount, expected_value, status, settlement_outcome, pnl, settled_at
            FROM trade_execution WHERE status = 'SETTLED' ORDER BY settled_at ASC;
        """
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        if not df.empty:
            df["stake_amount"] = df["stake_amount"].astype(float)
            df["pnl"] = df["pnl"].astype(float)
            df["expected_value"] = df["expected_value"].astype(float)
        return df
    except Exception as e:
        logger.error(f"Postgres settled wagers fetch failed: {e}")
        return pd.DataFrame()

def compute_portfolio_metrics(df: pd.DataFrame, alpha: float = 30.0) -> dict:
    """Computes net pnl, profit factor, win rate, and Bayesian-shrunk Sharpe ratio."""
    if df.empty:
        return {"net_pnl": 0.0, "sharpe_ratio": 0.0, "profit_factor": 1.0, "total_trades": 0, "win_rate": 0.0}
    
    net_pnl = df["pnl"].sum()
    pnl_rates = df["pnl"] / df["stake_amount"]
    mean_rate = pnl_rates.mean()
    std_rate = pnl_rates.std()
    empirical_SR = mean_rate / std_rate if std_rate != 0 and pd.notna(std_rate) else 0.0
        
    n = len(df)
    shrunk_SR = (n / (n + alpha)) * empirical_SR if n > 0 else 0.0
    
    gross_profits = df[df["pnl"] > 0]["pnl"].sum()
    gross_losses = abs(df[df["pnl"] < 0]["pnl"].sum())
    profit_factor = gross_profits / gross_losses if gross_losses > 0 else (gross_profits if gross_profits > 0 else 1.0)
    
    settled_decisions = df[df["settlement_outcome"].isin(["WIN", "LOSS"])]
    win_rate = len(settled_decisions[settled_decisions["settlement_outcome"] == "WIN"]) / len(settled_decisions) if not settled_decisions.empty else 0.0
        
    return {
        "net_pnl": net_pnl, "sharpe_ratio": shrunk_SR, "profit_factor": profit_factor,
        "total_trades": n, "win_rate": win_rate
    }

def get_venue_segmentation(df: pd.DataFrame) -> pd.DataFrame:
    """Groups settled wagers into Prediction Markets vs Retail Sportsbooks."""
    if df.empty:
        return pd.DataFrame(columns=["Venue", "Trades", "Volume", "Net PnL", "Win Rate", "Profit Factor", "Sharpe (Shrunk)"])
        
    def categorize_venue(book):
        if str(book).lower() in ["kalshi", "polymarket"]: return "Prediction Markets"
        return "Retail Sportsbooks"
        
    df_cat = df.copy()
    df_cat["Venue"] = df_cat["bookmaker"].apply(categorize_venue)
    
    segments = []
    for name, group in df_cat.groupby("Venue"):
        alpha = 50.0 if name == "Prediction Markets" else 15.0
        metrics = compute_portfolio_metrics(group, alpha=alpha)
        segments.append({
            "Venue": name, "Trades": metrics["total_trades"],
            "Volume": f"${group['stake_amount'].sum():,.2f}",
            "Net PnL": f"${metrics['net_pnl']:+,.2f}",
            "Win Rate": f"{metrics['win_rate']:.1%}",
            "Profit Factor": f"{metrics['profit_factor']:.2f}",
            "Sharpe (Shrunk)": f"{metrics['sharpe_ratio']:.2f}"
        })
    return pd.DataFrame(segments)


# ══════════════════════════════════════════════════════════
# 3. APP LAYOUT & CONTROL SIDEBAR
# ══════════════════════════════════════════════════════════

st.markdown('<div class="terminal-header">STATARB KINETIC</div>', unsafe_allow_html=True)
st.markdown('<div class="terminal-subheader">Lead Quantitative Dashboard | HFT Orderflow & Analytics</div>', unsafe_allow_html=True)

# Sidebar System Health Status
st.sidebar.markdown("## ⚙️ Infrastructure")

with st.sidebar:
    @st.fragment(run_every="10s")
    def render_infrastructure_status():
        redis_live = get_redis_connection_status()
        pg_live = get_postgres_connection_status()

        r_status = '<span class="pulse-green"></span> Online' if redis_live else '<span class="pulse-amber"></span> Disconnected'
        p_status = '<span class="pulse-green"></span> Online' if pg_live else '<span class="pulse-amber"></span> Disconnected'

        st.markdown(f"""
        <div class="api-status"><strong>Redis Cache:</strong> {r_status}</div>
        <div class="api-status"><strong>TimescaleDB:</strong> {p_status}</div>
        """, unsafe_allow_html=True)

    render_infrastructure_status()

st.sidebar.markdown("---")

# Sidebar Data Feed Telemetry
st.sidebar.markdown("## 📡 Data Feed Telemetry")
st.sidebar.markdown("""
<div class="api-status"><span class="pulse-green"></span> <strong>PrizePicks</strong> [Tier 1 REST]</div>
<div class="api-status"><span class="pulse-green"></span> <strong>Pinnacle</strong> [Sharp Anchor]</div>
<div class="api-status"><span class="pulse-green"></span> <strong>Kalshi</strong> [CLOB WSS]</div>
<div class="api-status"><span class="pulse-green"></span> <strong>Polymarket</strong> [CLOB WSS]</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# Auto refresh control panel
st.sidebar.markdown("## ⏱️ Engine Control")
auto_refresh_enabled = st.sidebar.checkbox("Enable Live Orderbook Sync (30s)", value=True)

# Information banner
st.sidebar.markdown("---")

# Fetch datasets strictly from live Redis / Postgres
live_df, live_empty = fetch_live_signals()
hist_df, hist_empty = fetch_historical_signals()


# ══════════════════════════════════════════════════════════
# 4. DASHBOARD LAYOUT & TABS
# ══════════════════════════════════════════════════════════

tab_live, tab_hist, tab_monte_carlo = st.tabs(["⚡ Live Execution Board", "📈 Historical Analytics", "🎯 Live Monte Carlo Radar"])

# ──────────────────────────────────────────────────────────
# TAB 1: LIVE EXECUTION BOARD
# ──────────────────────────────────────────────────────────
with tab_live:
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    
    with m_col1:
        st.metric(label="Active Arbs Detected", value=f"{len(live_df)} entries", delta="Live Redis Cache")
    with m_col2:
        avg_ev = live_df["EV Edge"].mean() * 100 if not live_df.empty else 0.0
        st.metric(label="Average EV Edge", value=f"+{avg_ev:.2f}%", delta="Optimal Profile")
    with m_col3:
        max_ev = live_df["EV Edge"].max() * 100 if not live_df.empty else 0.0
        st.metric(label="Max EV Edge", value=f"+{max_ev:.2f}%", delta="Alpha Target")
    with m_col4:
        st.metric(label="Last Engine Sync", value=datetime.now().strftime("%H:%M:%S"), delta="Connected" if not live_empty else "Awaiting Stream")
        
    st.markdown('<div class="section-title">⚡ Live +EV Arbitrage Orderbook</div>', unsafe_allow_html=True)
    
    refresh_rate = "30s" if auto_refresh_enabled else None
    
    @st.fragment(run_every=refresh_rate)
    def render_live_scanner_table():
        display_df = fetch_live_signals()[0] if auto_refresh_enabled else live_df
            
        if display_df.empty:
            st.info("Scanner Active: Awaiting live +EV orderflow from HFP...")
            return

        styled_df = display_df.copy()
        styled_df["EV Edge %"] = (styled_df["EV Edge"] * 100).map(lambda x: f"+{x:.2f}%")
        styled_df["True Prob %"] = (styled_df["True Prob"] * 100).map(lambda x: f"{x:.1f}%")
        styled_df["Masked Kelly Stake"] = styled_df["Masked Size"].map(lambda x: f"${x:,.2f}")
        
        def parse_liquidity(row):
            sb = str(row.get("Sharp Book", "")).lower()
            liq = row.get("Anchor Liquidity USD", "")
            if sb == "pinnacle": return "3-Hr Baseline"
            try:
                if liq and str(liq).strip(): return f"${float(liq):,.2f}"
            except ValueError: pass
            return "Thin/Unknown"
            
        styled_df["Anchor Liquidity"] = styled_df.apply(parse_liquidity, axis=1)
        
        styled_df = styled_df.rename(columns={
            "Market": "Stat",
            "Retail Line": "Line",
            "Book": "Target Book",
            "Sharp Book": "Anchor Source",
            "EV Edge %": "EV %",
            "Masked Kelly Stake": "Recommended Stake"
        })
        
        column_selection = [
            "Player", "Stat", "Side", "Line", "Target Book", 
            "Anchor Source", "EV %", "Recommended Stake", "Anchor Liquidity"
        ]
        
        st.dataframe(styled_df[column_selection], use_container_width=True, hide_index=True)
        
    render_live_scanner_table()

# ──────────────────────────────────────────────────────────
# TAB 2: HISTORICAL ANALYTICS
# ──────────────────────────────────────────────────────────
with tab_hist:
    wagers_df = fetch_portfolio_metrics()
    wagers_metrics = compute_portfolio_metrics(wagers_df)
    
    st.markdown('<div class="section-title">💼 Portfolio Performance Telemetry</div>', unsafe_allow_html=True)
    if not wagers_df.empty:
        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
        with p_col1: st.metric(label="Net PnL", value=f"${wagers_metrics['net_pnl']:+,.2f}", delta="Realized")
        with p_col2: st.metric(label="Sharpe Ratio (Shrunk)", value=f"{wagers_metrics['sharpe_ratio']:.2f}", delta=f"n = {wagers_metrics['total_trades']}")
        with p_col3: st.metric(label="Profit Factor", value=f"{wagers_metrics['profit_factor']:.2f}", delta="Gross Win/Loss")
        with p_col4: st.metric(label="Win Rate", value=f"{wagers_metrics['win_rate']:.1%}", delta="Hit Rate")
            
        pnl_chart_df = wagers_df.copy().sort_values("settled_at")
        pnl_chart_df["Cumulative PnL ($)"] = pnl_chart_df["pnl"].cumsum()
        
        fig_pnl = px.line(pnl_chart_df, x="settled_at", y="Cumulative PnL ($)", template="plotly_dark")
        fig_pnl.update_traces(line=dict(color="#3b82f6", width=3))
        fig_pnl.update_layout(
            xaxis=dict(gridcolor="rgba(255, 255, 255, 0.05)"),
            yaxis=dict(gridcolor="rgba(255, 255, 255, 0.05)"),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=40, r=40, t=30, b=40), height=300
        )
        st.plotly_chart(fig_pnl, use_container_width=True)
    else:
        st.info("Awaiting settled wagers in database to populate telemetry.")

    st.markdown('<div class="section-title">🏛️ Venue Segmentation Analysis</div>', unsafe_allow_html=True)
    if not wagers_df.empty:
        st.dataframe(get_venue_segmentation(wagers_df), use_container_width=True, hide_index=True)
    else:
        st.info("Awaiting wagers for venue segmentation.")

    st.markdown('<div class="section-title">📊 Historical Signal Exhaust (TimescaleDB)</div>', unsafe_allow_html=True)
    if hist_df.empty:
        st.info("Awaiting historical arbitrage signals...")
    else:
        h_col1, h_col2, h_col3 = st.columns(3)
        with h_col1: st.metric(label="Total Logged Signals", value=f"{len(hist_df)} logs", delta="TimescaleDB")
        with h_col2: st.metric(label="Cumulative EV Detected", value=f"+{hist_df['EV Edge'].sum() * 100:.1f}%", delta="Edge Mass")
        with h_col3: st.metric(label="Historical Average EV", value=f"+{hist_df['EV Edge'].mean() * 100:.2f}%", delta="Mean Profile")
            
        styled_hist = hist_df.copy().sort_values("timestamp", ascending=False)
        styled_hist["EV %"] = (styled_hist["EV Edge"] * 100).map(lambda x: f"+{x:.2f}%")
        styled_hist["Kelly %"] = (styled_hist["Raw Kelly"] * 100).map(lambda x: f"{x:.1f}%")
        styled_hist["Timestamp"] = pd.to_datetime(styled_hist["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        
        cols = ["Timestamp", "Player", "Anchor Source", "Target Book", "Sharp Line", "Retail Line", "EV %", "Kelly %"]
        st.dataframe(styled_hist.rename(columns={"Sharp Book": "Anchor Source", "Book": "Target Book"})[cols].head(50), use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────────────────
# TAB 3: LIVE MONTE CARLO RADAR
# ──────────────────────────────────────────────────────────
with tab_monte_carlo:
    st.markdown('<div class="section-title">🎯 ZINB Monte Carlo Simulator</div>', unsafe_allow_html=True)
    st.write("Run 10,000-Path Monte Carlo simulations in real-time. Pulls current lines from PrizePicks and projects them via the XGBoost pipeline.")
    
    if st.button("Initialize 10,000-Path Simulation"):
        with st.spinner("Connecting to endpoints & simulating discrete paths..."):
            try:
                import asyncio
                from research.live_monte_carlo import run_live_board_simulation
                
                def run_async(coro):
                    try: return asyncio.run(coro)
                    except RuntimeError: return asyncio.get_event_loop().run_until_complete(coro)
                
                sim_data = run_async(run_live_board_simulation())
                
                if not sim_data:
                    st.warning("No active NBA projection lines found on the board.")
                else:
                    df = pd.DataFrame(sim_data)
                    def highlight_probs(val):
                        try:
                            if float(val) > 0.60: return 'background-color: rgba(16, 185, 129, 0.15); color: #10b981; font-weight: bold;'
                        except: pass
                        return ''
                    
                    st.dataframe(
                        df.style.map(highlight_probs, subset=["Over Prob", "Under Prob"])
                                .format({"Over Prob": "{:.1%}", "Under Prob": "{:.1%}", "Mean": "{:.2f}", "Line": "{:.1f}"}),
                        use_container_width=True, hide_index=True
                    )
                    st.success("Simulation complete. 10,000 universes explored per player.")
            except Exception as e:
                st.error(f"Failed to execute Live Monte Carlo Simulation: {e}")
