import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import psycopg2
import os
import sys
import logging
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
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark premium CSS injection for glassmorphic styling
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono:wght@400;700&display=swap');
        
        /* Font overrides */
        html, body, [class*="css"] {
            font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        code, pre {
            font-family: 'JetBrains Mono', monospace !important;
        }

        /* Dark premium container background */
        .stApp {
            background-color: #0b0f19;
            color: #f1f5f9;
        }

        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #0e1526 !important;
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }

        /* Glassmorphic Metric Cards */
        div[data-testid="metric-container"] {
            background: rgba(17, 24, 39, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 18px 24px;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
            backdrop-filter: blur(10px);
            transition: all 0.3s ease;
        }
        div[data-testid="metric-container"]:hover {
            border-color: rgba(16, 185, 129, 0.3);
            box-shadow: 0 4px 30px rgba(16, 185, 129, 0.1);
            transform: translateY(-2px);
        }

        /* Metric text overrides */
        div[data-testid="metric-container"] label {
            color: #94a3b8 !important;
            font-weight: 500 !important;
            text-transform: uppercase;
            font-size: 0.8rem !important;
            letter-spacing: 0.05em;
        }
        
        /* Custom elements */
        .terminal-header {
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #10b981 0%, #3b82f6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 5px;
            letter-spacing: -0.02em;
        }

        .terminal-subheader {
            color: #64748b;
            font-size: 1.0rem;
            margin-bottom: 25px;
        }

        .status-badge {
            background-color: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.3);
            color: #10b981;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            display: inline-block;
        }

        .status-badge-offline {
            background-color: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.3);
            color: #ef4444;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            display: inline-block;
        }
        
        .section-title {
            font-size: 1.4rem;
            font-weight: 600;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            padding-bottom: 8px;
            margin-top: 15px;
            margin-bottom: 20px;
            color: #f1f5f9;
        }

        /* Style pandas dataframe in streamlit */
        .stDataFrame {
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            background-color: rgba(17, 24, 39, 0.3);
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

# Caching Redis fetches to prevent blocking Streamlit rendering loops
def fetch_live_signals() -> tuple[pd.DataFrame, bool]:
    """
    Fetches live arbitrage signals from Redis cache.
    Returns: (DataFrame, is_simulated)
    """
    try:
        from infrastructure.redis_client import RedisClient
        r = RedisClient().get_client()
        keys = r.keys("signal:*")
        
        signals = []
        for k in keys:
            data = r.hgetall(k)
            if data:
                # Helper for robust deserialization (case-insensitive and alias-aware)
                def get_field(d, *possible_keys, default=""):
                    for pk in possible_keys:
                        if pk in d:
                            return d[pk]
                        # check lowercase
                        if pk.lower() in d:
                            return d[pk.lower()]
                        # check underscores replacing spaces
                        underscore_k = pk.replace(" ", "_")
                        if underscore_k in d:
                            return d[underscore_k]
                        if underscore_k.lower() in d:
                            return d[underscore_k.lower()]
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
        
    # Return empty layout matching the schema
    return pd.DataFrame(columns=[
        "Player", "Market", "Side", "Sharp Line", "Retail Line", 
        "True Prob", "EV Edge", "Raw Kelly", "Masked Size", 
        "Book", "Sharp Book", "Anchor Liquidity USD", "timestamp"
    ]), True

def fetch_historical_signals() -> tuple[pd.DataFrame, bool]:
    """
    Queries historical arb signals from PostgreSQL hypertable.
    Returns: (DataFrame, is_simulated)
    """
    try:
        conn = psycopg2.connect(DB_URI)
        query = """
            SELECT 
                timestamp,
                player_name as "Player",
                sharp_book as "Sharp Book",
                stale_book as "Book",
                sharp_line as "Sharp Line",
                stale_line as "Retail Line",
                expected_value as "EV Edge",
                kelly_fraction as "Raw Kelly",
                anchor_liquidity_usd as "Anchor Liquidity USD"
            FROM arb_signals
            ORDER BY timestamp ASC;
        """
        df = pd.read_sql(query, conn)
        conn.close()
        if not df.empty:
            # Ensure correct types
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


# ══════════════════════════════════════════════════════════
# 3. PURGED FALLBACK GENERATORS (STRICT LIVE DATA BINDING)
# ══════════════════════════════════════════════════════════
# Hardcoded arrays, dummy JSON objects, and simulated logs
# have been eradicated to enforce frontend data integrity.


# ══════════════════════════════════════════════════════════
# 4. APP LAYOUT & CONTROL SIDEBAR
# ══════════════════════════════════════════════════════════

# Header Section
st.markdown('<div class="terminal-header">StatArb Trading Terminal</div>', unsafe_allow_html=True)
st.markdown('<div class="terminal-subheader">Lead Quantitative Dashboard | Live Arbitrage Orderflow & Analytics</div>', unsafe_allow_html=True)

# Sidebar System Health Status
st.sidebar.markdown("## 🖥️ System Integration Status")

redis_live = get_redis_connection_status()
pg_live = get_postgres_connection_status()

if redis_live:
    st.sidebar.markdown('<div class="status-badge">● Redis Cache Online</div>', unsafe_allow_html=True)
else:
    st.sidebar.markdown('<div class="status-badge-offline">○ Redis Cache Offline</div>', unsafe_allow_html=True)

st.sidebar.markdown("")

if pg_live:
    st.sidebar.markdown('<div class="status-badge">● TimescaleDB Online</div>', unsafe_allow_html=True)
else:
    st.sidebar.markdown('<div class="status-badge-offline">○ TimescaleDB Offline</div>', unsafe_allow_html=True)

st.sidebar.markdown("---")

# Auto refresh control panel
st.sidebar.markdown("## ⏱️ Control Panel")
auto_refresh_enabled = st.sidebar.checkbox("Enable Auto-Refresh (30s)", value=True)

# Information banner
st.sidebar.markdown("---")
st.sidebar.info(
    "**StatArb Risk Engine**\n\n"
    "This dashboard pulls live arbitrage signals from local memory structures (Redis) and historical analytics from TimescaleDB.\n\n"
    "It enforces Level 2 orderbook checks (Best Bid/Ask >= $100.00) on prediction markets and falls back to Pinnacle's 3-Hour Cache when necessary."
)


# Fetch datasets strictly from live Redis / Postgres
live_df, live_empty = fetch_live_signals()
hist_df, hist_empty = fetch_historical_signals()

# Bind directly to live pipelines
live_df_to_show = live_df
live_source_type = "Live Redis Cache"

hist_df_to_show = hist_df
hist_source_type = "Live PostgreSQL hypertable"


# ══════════════════════════════════════════════════════════
# 5. DASHBOARD LAYOUT & TABS
# ══════════════════════════════════════════════════════════

tab_live, tab_hist = st.tabs(["⚡ Live Execution Board", "📈 Historical Analytics"])

# ──────────────────────────────────────────────────────────
# TAB 1: LIVE EXECUTION BOARD
# ──────────────────────────────────────────────────────────
with tab_live:
    # Top metrics row
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    
    with m_col1:
        st.metric(
            label="Active Arbs Detected", 
            value=f"{len(live_df_to_show)} opportunities",
            delta=f"Feed: {live_source_type}"
        )
    with m_col2:
        if not live_df_to_show.empty:
            avg_ev = live_df_to_show["EV Edge"].mean() * 100
        else:
            avg_ev = 0.0
        st.metric(
            label="Average EV Edge", 
            value=f"+{avg_ev:.2f}%", 
            delta="Optimal Edge Profile"
        )
    with m_col3:
        if not live_df_to_show.empty:
            max_ev = live_df_to_show["EV Edge"].max() * 100
        else:
            max_ev = 0.0
        st.metric(
            label="Max EV Edge", 
            value=f"+{max_ev:.2f}%", 
            delta="High Priority Signal"
        )
    with m_col4:
        st.metric(
            label="Last Refresh Sync", 
            value=datetime.now().strftime("%H:%M:%S"),
            delta="Connected to Poller Feed" if not live_empty else "Graceful Fallback Mode"
        )
        
    st.markdown('<div class="section-title">⚡ Live +EV Arbitrage Signals Scanner</div>', unsafe_allow_html=True)
    
    # Auto-refresh mechanism inside tab
    # Fragment enables re-rendering just the signal scanner table every 30 seconds
    refresh_rate = "30s" if auto_refresh_enabled else None
    
    @st.fragment(run_every=refresh_rate)
    def render_live_scanner_table():
        # Fetch fresh data if auto refreshing
        if auto_refresh_enabled:
            current_live_df, current_empty = fetch_live_signals()
            display_df = current_live_df
        else:
            display_df = live_df_to_show
            
        if display_df.empty:
            st.info("System Active: Awaiting live arbitrage signals...")
            return

        # Format and polish the dataframe columns
        styled_df = display_df.copy()
        
        # 1. Format probabilities and edges
        styled_df["EV Edge %"] = (styled_df["EV Edge"] * 100).map(lambda x: f"+{x:.2f}%")
        styled_df["True Prob %"] = (styled_df["True Prob"] * 100).map(lambda x: f"{x:.1f}%")
        
        # 2. Format Kelly stake as currency
        styled_df["Masked Kelly Stake"] = styled_df["Masked Size"].map(lambda x: f"${x:,.2f}")
        
        # 3. Handle dynamic Liquidity Indicator
        # Display exact dollar amount if prediction market (Kalshi/Polymarket), else '3-Hr Baseline' for Pinnacle
        def parse_liquidity(row):
            sb = str(row.get("Sharp Book", "")).lower()
            liq = row.get("Anchor Liquidity USD", "")
            
            if sb == "pinnacle":
                return "3-Hr Baseline (Pinnacle)"
            
            try:
                if liq and str(liq).strip():
                    val = float(liq)
                    return f"${val:,.2f}"
            except ValueError:
                pass
                
            return "Thin/Unknown"
            
        styled_df["Anchor Liquidity"] = styled_df.apply(parse_liquidity, axis=1)
        
        # Rename columns to match institutional layout requirements
        styled_df = styled_df.rename(columns={
            "Market": "Stat",
            "Retail Line": "Line",
            "Book": "Target Book",
            "Sharp Book": "Anchor Source",
            "EV Edge %": "EV %",
            "Masked Kelly Stake": "Recommended Kelly Stake"
        })
        
        # Select clean columns as per prompt requirements
        column_selection = [
            "Player", "Stat", "Side", "Line", "Target Book", 
            "Anchor Source", "EV %", "Recommended Kelly Stake", "Anchor Liquidity"
        ]
        
        # Display polished dataframe
        st.dataframe(
            styled_df[column_selection],
            use_container_width=True,
            hide_index=True
        )
        
        # Add visual helper
        st.caption("ℹ️ *Note: Columns show real-time limits. Prediction market prices are mapped to continuous variables (e.g. Yes Ask/Bid). Pinnacle anchor utilizes the 3-hour cache TTL.*")
        
    render_live_scanner_table()

# ──────────────────────────────────────────────────────────
# TAB 2: HISTORICAL ANALYTICS
# ──────────────────────────────────────────────────────────
with tab_hist:
    if hist_df_to_show.empty:
        st.info("System Active: Awaiting historical arbitrage signals from TimescaleDB...")
        st.stop()

    h_col1, h_col2, h_col3 = st.columns(3)
    
    with h_col1:
        st.metric(
            label="Total Logged Signals", 
            value=f"{len(hist_df_to_show)} logs",
            delta="TimescaleDB Hypertable"
        )
    with h_col2:
        total_ev = hist_df_to_show["EV Edge"].sum() * 100
        st.metric(
            label="Cumulative EV Edge Detected", 
            value=f"+{total_ev:.1f}%", 
            delta="Captured Quantitative Spread"
        )
    with h_col3:
        avg_ev = hist_df_to_show["EV Edge"].mean() * 100
        st.metric(
            label="Historical Average EV", 
            value=f"+{avg_ev:.2f}%", 
            delta="Robust Edge Profile"
        )
        
    st.markdown('<div class="section-title">📈 Performance Metrics & Distribution Analytics</div>', unsafe_allow_html=True)
    
    col_chart_l, col_chart_r = st.columns([7, 5])
    
    with col_chart_l:
        st.markdown("### Cumulative Expected Value Detected Over Time")
        
        # Compute cumulative expected value over time
        chart_df = hist_df_to_show.copy()
        chart_df = chart_df.sort_values("timestamp")
        chart_df["Cumulative EV Edge (%)"] = (chart_df["EV Edge"].cumsum()) * 100
        
        fig_line = px.line(
            chart_df,
            x="timestamp",
            y="Cumulative EV Edge (%)",
            labels={"timestamp": "Signal Log Time", "Cumulative EV Edge (%)": "Cumulative EV Edge (%)"},
            template="plotly_dark"
        )
        
        fig_line.update_traces(
            line=dict(color="#10b981", width=3),
            hovertemplate="Time: %{x}<br>Cumulative EV: +%{y:.2f}%<extra></extra>"
        )
        
        fig_line.update_layout(
            xaxis=dict(
                gridcolor="rgba(255, 255, 255, 0.05)",
                color="#94a3b8"
            ),
            yaxis=dict(
                gridcolor="rgba(255, 255, 255, 0.05)",
                color="#94a3b8"
            ),
            plot_bgcolor="rgba(17, 24, 39, 0.4)",
            paper_bgcolor="rgba(0, 0, 0, 0)",
            margin=dict(l=40, r=40, t=30, b=40),
            height=450
        )
        
        st.plotly_chart(fig_line, use_container_width=True)
        
    with col_chart_r:
        st.markdown("### Anchor Source Distribution")
        
        # Group count of sharp books
        distribution_df = hist_df_to_show.groupby("Sharp Book").size().reset_index(name="Count")
        
        fig_pie = px.pie(
            distribution_df,
            names="Sharp Book",
            values="Count",
            color="Sharp Book",
            color_discrete_map={
                "Pinnacle": "#3b82f6",     # Blue
                "Kalshi": "#10b981",       # Emerald
                "Polymarket": "#f59e0b"    # Amber
            },
            template="plotly_dark"
        )
        
        fig_pie.update_traces(
            hole=0.4, 
            hoverinfo="label+percent+value",
            textinfo="percent+label",
            textfont=dict(size=12, color="#ffffff")
        )
        
        fig_pie.update_layout(
            plot_bgcolor="rgba(0, 0, 0, 0)",
            paper_bgcolor="rgba(0, 0, 0, 0)",
            margin=dict(l=40, r=40, t=30, b=40),
            legend=dict(font=dict(color="#94a3b8")),
            height=450
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)

    # Detailed logs table
    st.markdown('<div class="section-title">📋 Historical Arbitrage Log Feed (TimescaleDB)</div>', unsafe_allow_html=True)
    
    styled_hist = hist_df_to_show.copy()
    styled_hist = styled_hist.sort_values("timestamp", ascending=False)
    
    styled_hist["EV Edge %"] = (styled_hist["EV Edge"] * 100).map(lambda x: f"+{x:.2f}%")
    styled_hist["Raw Kelly %"] = (styled_hist["Raw Kelly"] * 100).map(lambda x: f"{x:.1f}%")
    
    def parse_hist_liquidity(val):
        if pd.isna(val) or val == 0.0:
            return "3-Hr Baseline (Pinnacle)"
        return f"${val:,.2f}"
        
    styled_hist["Anchor Liquidity"] = styled_hist["Anchor Liquidity USD"].apply(parse_hist_liquidity)
    styled_hist["Timestamp"] = pd.to_datetime(styled_hist["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    
    # Handle missing Market column gracefully if loaded from database
    if "Market" not in styled_hist.columns:
        styled_hist["Market"] = "N/A"
        
    # Rename columns to match institutional layout requirements
    styled_hist = styled_hist.rename(columns={
        "Market": "Stat",
        "Retail Line": "Line",
        "Book": "Target Book",
        "Sharp Book": "Anchor Source",
        "EV Edge %": "EV %",
        "Raw Kelly %": "Recommended Kelly Stake"
    })
    
    column_selection_hist = [
        "Timestamp", "Player", "Stat", "Anchor Source", "Target Book", 
        "Sharp Line", "Line", "EV %", "Recommended Kelly Stake", "Anchor Liquidity"
    ]
    
    st.dataframe(
        styled_hist[column_selection_hist].head(50),
        use_container_width=True,
        hide_index=True
    )
