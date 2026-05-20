import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("MiddleScanner")

class MiddleScanner:
    @staticmethod
    def scan_for_arbitrage(market_data: Dict[str, Dict]) -> List[Dict]:
        """
        Scans a dictionary of market lines from different retail books and flags risk-free middles and pure arbitrage.
        
        Expected structure of market_data:
        {
            'DraftKings': {'line': 8.5, 'over_odds': -120, 'under_odds': +100},
            'FanDuel':    {'line': 10.5, 'over_odds': +110, 'under_odds': -130},
            'PrizePicks': {'line': 9.0, 'over_odds': -137, 'under_odds': -137}
        }
        """
        opportunities = []
        books = list(market_data.keys())
        
        for i in range(len(books)):
            for j in range(i + 1, len(books)):
                book_a = books[i]
                book_b = books[j]
                
                data_a = market_data[book_a]
                data_b = market_data[book_b]
                
                line_a = data_a.get('line')
                line_b = data_b.get('line')
                
                if line_a is None or line_b is None:
                    continue
                    
                # 1. Check for Standard "Middles" (Different Lines)
                # We want the highest Over line to be lower than the Under line
                if line_a < line_b:
                    opp = MiddleScanner._evaluate_cross_book(book_a, data_a, book_b, data_b, 'over_odds', 'under_odds', line_a, line_b)
                    if opp: opportunities.append(opp)
                elif line_b < line_a:
                    opp = MiddleScanner._evaluate_cross_book(book_b, data_b, book_a, data_a, 'over_odds', 'under_odds', line_b, line_a)
                    if opp: opportunities.append(opp)
                    
                # 2. Check for Pure Arbitrage (Same Lines, Inefficient Odds)
                if line_a == line_b:
                    opp1 = MiddleScanner._evaluate_cross_book(book_a, data_a, book_b, data_b, 'over_odds', 'under_odds', line_a, line_b, is_arb=True)
                    opp2 = MiddleScanner._evaluate_cross_book(book_b, data_b, book_a, data_a, 'over_odds', 'under_odds', line_b, line_a, is_arb=True)
                    if opp1: opportunities.append(opp1)
                    if opp2: opportunities.append(opp2)
                    
        return opportunities

    @staticmethod
    def _evaluate_cross_book(book_over: str, data_over: dict, book_under: str, data_under: dict, 
                             over_key: str, under_key: str, line_over: float, line_under: float, is_arb=False) -> Optional[Dict]:
        """
        Evaluates if taking the Over on book_over and the Under on book_under creates a profitable scenario.
        """
        odds_over = data_over.get(over_key)
        odds_under = data_under.get(under_key)
        
        if odds_over is None or odds_under is None:
            return None
            
        # Convert American odds to implied probability
        def _to_implied(american: int) -> float:
            dec = (american / 100.0) + 1.0 if american > 0 else (100.0 / abs(american)) + 1.0
            return 1.0 / dec
            
        implied_over = _to_implied(odds_over)
        implied_under = _to_implied(odds_under)
        
        sum_implied = implied_over + implied_under
        
        # A true arbitrage exists if the sum of implied probabilities is < 1.0
        # A "middle" exists if the lines are different and the vig is low enough that the risk is subsidized
        
        is_profitable_middle = False
        if not is_arb and line_over < line_under:
            # For middles, we tolerate a slight negative expected sum if the gap is wide enough.
            # e.g., if we risk losing 3% but have a 10% chance to hit both bets.
            # A rigorous model would use the ZINB distribution to calculate the exact probability of hitting the middle gap.
            # For this basic scanner, we flag it if the gap >= 1.0 unit.
            if (line_under - line_over) >= 1.0 and sum_implied < 1.08:
                is_profitable_middle = True
                
        is_profitable_arb = is_arb and sum_implied < 1.0
        
        if is_profitable_arb or is_profitable_middle:
            return {
                'type': 'PURE_ARBITRAGE' if is_profitable_arb else 'MIDDLE',
                'buy_over': {'book': book_over, 'line': line_over, 'odds': odds_over},
                'buy_under': {'book': book_under, 'line': line_under, 'odds': odds_under},
                'implied_sum': round(sum_implied, 4),
                'guaranteed_roi': round((1.0 - sum_implied) * 100, 2) if is_profitable_arb else None,
                'middle_gap': line_under - line_over if not is_arb else 0.0
            }
            
        return None
