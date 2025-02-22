from typing import List, Tuple, Optional
import numpy as np
from dataclasses import dataclass

@dataclass
class RiskMetrics:
    vix_percentile: float
    std_dev: float
    mean_price: float
    current_deviation: float

class RiskIndicators:
    def __init__(self, lookback_period: int = 20):
        self.lookback_period = lookback_period

    def calculate_metrics(self, prices: np.ndarray, vix: float, vix_history: List[float]) -> RiskMetrics:
        """
        Calculate risk metrics for mean reversion strategy
        
        Args:
            prices: Historical price data
            vix: Current VIX value
            vix_history: Historical VIX values
        Returns:
            RiskMetrics: Calculated risk metrics
        """
        mean_price = np.mean(prices[-self.lookback_period:])
        std_dev = np.std(prices[-self.lookback_period:])
        current_deviation = (prices[-1] - mean_price) / std_dev
        vix_percentile = np.percentile(vix_history, vix)
        
        return RiskMetrics(
            vix_percentile=vix_percentile,
            std_dev=std_dev,
            mean_price=mean_price,
            current_deviation=current_deviation
        )

    def should_enter_trade(self, metrics: RiskMetrics) -> bool:
        """Check if entry conditions are met"""
        return (abs(metrics.current_deviation) > 2 and 
                metrics.vix_percentile < 75)

    def should_exit_trade(self, metrics: RiskMetrics) -> bool:
        """Check if exit conditions are met"""
        return (abs(metrics.current_deviation) < 0.5 or 
                metrics.vix_percentile > 75) 