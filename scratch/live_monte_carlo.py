import sys
import os
import logging
import asyncio
from prettytable import PrettyTable
from curl_cffi.requests import AsyncSession

# Dynamic path resolution to load your custom modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scrapers.hfp_poller import UnifiedRestPoller
from src.models import PlayerProjectionSimulator
from infrastructure.entity_resolver import clean_player_name

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("LiveMonteCarlo")

async def run_live_board_simulation():
    print("\n" + "="*70)
    print("🏀 LIVE PRIZEPICKS MONTE CARLO PROJECTION (10,000 Paths/Player) 🏀")
    print("="*70 + "\n")
    
    logger.info("Initializing Live API Poller...")
    poller = UnifiedRestPoller()
    
    logger.info("Fetching active NBA lines from PrizePicks...")
    live_board_dict = {}
    try:
        import random
        fingerprint = random.choice(["chrome110", "chrome116", "chrome120", "edge101", "safari15_5"])
        proxy_url = os.environ.get("RESIDENTIAL_PROXY_URL")
        proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else None
        
        # Establish the async session required by your fetch_pp method
        async with AsyncSession(impersonate=fingerprint, proxies=proxies, headers=poller.headers) as session:
            live_board_dict = await poller.fetch_pp(session)
    except Exception as e:
        logger.error(f"Failed to fetch live board: {e}")

    if live_board_dict is None:
        live_board_dict = {}

    if not live_board_dict:
        if getattr(poller, 'rate_limited', False):
            logger.warning("PP API call was blocked (WAF/Rate Limit). Attempting SQLite board snapshot fallback...")
        else:
            logger.warning("PP API returned 200 OK but has no active projections for NBA (league_id=7) right now. Attempting SQLite board snapshot fallback...")
        sqlite_successful = False
        try:
            import sqlite3
            from datetime import datetime, timezone
            sqlite_db = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "market_data.db")
            if os.path.exists(sqlite_db):
                conn = sqlite3.connect(sqlite_db)
                cur = conn.cursor()
                cur.execute("SELECT MAX(Timestamp) FROM board_snapshots WHERE Book = 'PrizePicks';")
                latest_ts = cur.fetchone()[0]
                if latest_ts:
                    # Parse timestamp (handles ISO format with offset e.g. +00:00)
                    snapshot_dt = datetime.fromisoformat(latest_ts)
                    now_dt = datetime.now(timezone.utc)
                    age_seconds = (now_dt - snapshot_dt).total_seconds()
                    age_minutes = age_seconds / 60.0
                    
                    if age_minutes > 15.0:
                        logger.warning(f"SQLite fallback data is extremely stale ({age_minutes:.1f} minutes old).")
                    else:
                        if age_minutes > 3.0:
                            print("\n" + "!"*70)
                            print(f"⚠️  WARNING: SQLite Snapshot is stale ({age_minutes:.1f} minutes old).")
                            print("Running live simulations on stale lines can lead to catastrophic execution")
                            print("risks if player news or injuries broke while the scraper was blocked.")
                            print("!"*70 + "\n")
                        else:
                            logger.info(f"SQLite snapshot age: {age_minutes:.1f} minutes. Line data is fresh.")
                            
                        cur.execute("""
                            SELECT Player, Stat, Line, Timestamp 
                            FROM board_snapshots 
                            WHERE Book = 'PrizePicks' AND Timestamp = ?;
                        """, (latest_ts,))
                        rows = cur.fetchall()
                        for i, row in enumerate(rows):
                            live_board_dict[f"sqlite_{i}"] = {
                                'player': row[0],
                                'prop': row[1],
                                'line': row[2],
                                'odds_type': 'standard',
                                'start_time': row[3]
                            }
                        logger.info(f"Successfully loaded {len(live_board_dict)} cached PrizePicks lines from SQLite snapshot.")
                        sqlite_successful = True
                else:
                    logger.error("No PrizePicks snapshots found in SQLite database.")
                conn.close()
            else:
                logger.error(f"SQLite database file not found at {sqlite_db}")
        except Exception as se:
            logger.error(f"Failed to read from SQLite cache: {se}")

        if not sqlite_successful:
            logger.warning("SQLite fallback is unavailable or stale. Attempting Sharp Cache fallback...")
            try:
                import json
                from datetime import datetime, timezone
                sharp_cache_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scrapers", "sharp_cache.json")
                if os.path.exists(sharp_cache_path):
                    with open(sharp_cache_path, 'r') as f:
                        cache_data = json.load(f)
                    
                    fetched_at_str = cache_data.get("fetched_at")
                    if fetched_at_str:
                        sharp_dt = datetime.fromisoformat(fetched_at_str)
                        now_dt = datetime.now(timezone.utc)
                        age_seconds = (now_dt - sharp_dt).total_seconds()
                        age_minutes = age_seconds / 60.0
                        
                        if age_minutes > 180.0:
                            logger.critical(f"FATAL: Sharp cache fallback data is extremely stale ({age_minutes:.1f} minutes old).")
                            logger.critical("Refusing to run Monte Carlo risk simulations on high-latency lines.")
                            return
                        
                        if age_minutes > 45.0:
                            print("\n" + "!"*70)
                            print(f"⚠️  WARNING: Sharp Cache is stale ({age_minutes:.1f} minutes old).")
                            print("Running live simulations on stale lines can lead to catastrophic execution")
                            print("risks if player news or injuries broke while the provider was blocked.")
                            print("!"*70 + "\n")
                        else:
                            logger.info(f"Sharp cache age: {age_minutes:.1f} minutes. Line data is fresh.")
                        
                        lines = cache_data.get("lines", {})
                        count = 0
                        for player_key, stats_dict in lines.items():
                            for stat_type, lines_dict in stats_dict.items():
                                for line_str, line_info in lines_dict.items():
                                    line_val = line_info.get("pinnacle_line") or line_info.get("dk_line") or float(line_str)
                                    unique_key = f"sharp_{player_key}_{stat_type}_{line_str}"
                                    live_board_dict[unique_key] = {
                                        'player': player_key,
                                        'prop': stat_type,
                                        'line': line_val,
                                        'odds_type': 'standard',
                                        'start_time': fetched_at_str
                                    }
                                    count += 1
                        logger.info(f"Successfully loaded {count} cached player prop lines from Sharp Cache.")
                    else:
                        logger.error("Sharp cache missing 'fetched_at' timestamp.")
                else:
                    logger.error(f"Sharp cache file not found at {sharp_cache_path}")
            except Exception as sce:
                logger.error(f"Failed to read from Sharp Cache: {sce}")
            
            if not live_board_dict:
                logger.critical("FATAL: Both SQLite snapshot and Sharp Cache fallbacks are stale, missing, or empty.")
                logger.critical("Refusing to run Monte Carlo risk simulations on high-latency lines.")
                return

    if not live_board_dict:
        logger.warning("No live or cached NBA lines found on the board.")
        return

    # Convert your dictionary output into a list of props we can iterate through
    live_board = list(live_board_dict.values())
    
    sim_results = []
    
    table = PrettyTable()
    table.field_names = ["Player", "Stat", "PP Line", "Simulated Mean", "Over Prob", "Under Prob", "Fair Odds (Over)", "5% Floor", "95% Ceiling"]
    table.align = "l"

    logger.info(f"Simulating up to 10 active base-stat legs...")
    compact_count = 0

    # DO NOT slice [:10] here. We must scan the whole board to find 10 valid base stats.
    for index, leg in enumerate(live_board):
        raw_player = leg.get("player", "Unknown")
        stat_type = leg.get("prop", "Unknown")
        
        # FILTER 1: Skip composite/combo players
        if "+" in raw_player:
            continue
            
        # FILTER 2: Skip composite stats so we can see the DB models work
        if stat_type.lower() not in ['points', 'rebounds', 'assists', 'turnovers']:
            continue
            
        # FILTER 3: Skip promotional / boosted lines
        odds_type = leg.get("odds_type", "standard")
        is_promo = leg.get("is_promo", False) or leg.get("promo", False) or leg.get("flash_sale", False)
        if odds_type != "standard" or is_promo:
            continue
            
        pp_line = float(leg.get("line", 0.0) or 0.0)
        if pp_line == 0.0:
            continue
            
        player_name = clean_player_name(raw_player)
        
        # 1. Resolve the Database ID
        raw_pp_name = leg.get("player", "Unknown")
        master_id = poller.resolver.resolve_player("PrizePicks", raw_pp_name)
        if not master_id:
            master_id = player_name 
            
        # 2. Fetch Historical Logs
        history_df = await poller.get_player_history_async(master_id)        
        
        # 3. Map the Stat Column
        mapped_stat = poller._map_stat_type(stat_type)
        
        # 4. Pull the true model-implied mean (mu) from the production pipeline
        true_mean = pp_line # Default fallback
        if mapped_stat and not history_df.empty:
            try:
                # Calls your ZINB + XGBoost modeling stack
                _, fitted_zinb = poller.calculate_theoretical_prob(
                    player_name=player_name, 
                    stat_col=mapped_stat, 
                    line=pp_line, 
                    master_id=master_id,
                    history_df=history_df
                )
                true_mean = fitted_zinb.mu
            except Exception as me:
                logger.warning(f"Model fitting failed for {player_name}. Error: {me}")
        
        # 5. Calculate Dynamic Variance based on actual game logs
        dynamic_factor = 1.25 # Standard baseline
        if mapped_stat and not history_df.empty and mapped_stat in history_df.columns:
            historical_array = history_df[mapped_stat].dropna().values
            dynamic_factor = PlayerProjectionSimulator.calculate_dynamic_overdispersion(historical_array)

        # 6. Run the Monte Carlo Simulation (using asyncio.to_thread to avoid blocking the event loop)
        results = await asyncio.to_thread(
            PlayerProjectionSimulator.simulate_stat_distribution,
            projected_mean=true_mean,
            sportsbook_line=pp_line,
            overdispersion_factor=dynamic_factor,
            n_simulations=10000
        )

        GREEN = "\033[32m"
        RESET = "\033[0m"

        if results['prob_over'] > 0.60:
            over_prob_str = f"{GREEN}{results['prob_over']:.1%}{RESET}"
        else:
            over_prob_str = f"{results['prob_over']:.1%}"

        if results['prob_under'] > 0.60:
            under_prob_str = f"{GREEN}{results['prob_under']:.1%}{RESET}"
        else:
            under_prob_str = f"{results['prob_under']:.1%}"

        # Store in list of dicts
        sim_results.append({
            "Player": player_name,
            "Stat": stat_type,
            "Line": pp_line,
            "Mean": true_mean,
            "Over Prob": results['prob_over'],
            "Under Prob": results['prob_under'],
            "Fair Odds": results['fair_odds_over'],
            "5% Floor": results['5th_percentile_floor'],
            "95% Ceiling": results['95th_percentile_ceiling']
        })

        table.add_row([
            player_name,
            stat_type,
            pp_line,
            f"{true_mean:.1f}",
            over_prob_str,
            under_prob_str,
            results['fair_odds_over'],
            results['5th_percentile_floor'],
            results['95th_percentile_ceiling']
        ])
        
        # 7. Increment counter and break if we hit 10
        compact_count += 1
        if compact_count >= 10:
            break

    print("\n")
    print(table)
    print("\n" + "="*70)
    print("Simulation Complete. 10,000 Universes explored per player.")
    print("="*70)
    
    return sim_results

if __name__ == "__main__":
    asyncio.run(run_live_board_simulation())