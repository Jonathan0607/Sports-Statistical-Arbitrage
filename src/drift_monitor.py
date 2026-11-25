import os
import sys
import requests
from datetime import datetime, timedelta

# Ensure src is in PYTHONPATH if run externally
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.historical_replay_core import HistoricalReplayCore

def send_discord_alert(webhook_url, message):
    if not webhook_url:
        print("No DISCORD_WEBHOOK_URL set. Skipping alert.")
        return
    
    payload = {
        "content": message
    }
    try:
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
        print("Discord alert sent successfully.")
    except Exception as e:
        print(f"Failed to send Discord alert: {e}")

def main():
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=14)
    
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    
    print(f"Running WNBA Drift Monitor for dates {start_date_str} to {end_date_str}")
    
    db_uri = os.getenv("POSTGRES_URI", "postgresql+psycopg2://postgres:password@localhost:5432/sports_db")
    
    backtester = HistoricalReplayCore(
        db_uri=db_uri,
        league="WNBA"
    )
    
    try:
        results = backtester.run_backtest(
            start_date=start_date_str,
            end_date=end_date_str,
            stat_categories=["Rebounds", "Assists"],
            return_summary=True
        )
    except Exception as e:
        print(f"Failed to run backtest: {e}")
        return
        
    summary = results.get("summary", [])
    
    if not summary:
        print("No predictions found for the last 14 days.")
        return
        
    discord_webhook = os.getenv("DISCORD_WEBHOOK_URL")
    
    for row in summary:
        category = row["Category"]
        meta_mae = row["Meta MAE"]
        ewma_mae = row["EWMA MAE"]
        count = row["Count"]
        
        if ewma_mae == 0:
            continue
            
        edge_pct = ((ewma_mae - meta_mae) / ewma_mae) * 100.0
        
        print(f"[{category}] Count: {count} | Meta MAE: {meta_mae:.3f} | EWMA MAE: {ewma_mae:.3f} | Edge: {edge_pct:.2f}%")
        
        # Trigger alert if Edge Percentage drops below 1.0%
        if edge_pct < 1.0:
            msg = (
                f"🚨 **WNBA Model Drift Alert** 🚨\n"
                f"**Category:** {category}\n"
                f"**Degradation:** Edge has dropped below 1.0% threshold (Current: `{edge_pct:.2f}%`).\n"
                f"**Metrics:** Meta MAE `{meta_mae:.3f}` vs EWMA MAE `{ewma_mae:.3f}` over the last 14 days ({count} samples).\n"
                f"⚠️ **ACTION REQUIRED:** Halt automated betting on {category} and recalibrate model."
            )
            print(f"Triggering alert for {category}!")
            send_discord_alert(discord_webhook, msg)
        else:
            print(f"✅ {category} model is healthy.")

if __name__ == "__main__":
    main()
