from typing import List, Tuple, Optional, Callable
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

        return RiskMetrics(
            std_dev=std_dev,
            mean_price=mean_price,
            current_deviation=current_deviation
        )
                
    def calculate_var_es(self, 
                         simulated_returns: Optional[np.ndarray] = None,
                         z_score: float = None,
                         std: float = None) -> Tuple[float, float]:
        """
        Calculate Value at Risk (VaR) and Expected Shortfall (ES) at a specified confidence level over a given period
        using Parametric VaR method (also called Variance-Covariance Method).
        
        Args:
            simulated_returns: Optional pre-computed simulated returns with shape (iterations, T)
            confidence_level: Confidence level for VaR and ES (e.g., 0.01 for 1%)
        
        Returns:
            Tuple[float, float]: VaR and ES at the specified confidence level
            
        Raises:
            ValueError: If variance is not provided
            ValueError: If simulated_returns is not provided or empty
        """
        if std is None:
            raise ValueError("standard deviation parameter is required for calculating VaR and ES")
            
        if simulated_returns is None or len(simulated_returns) == 0:
            raise ValueError("simulated_returns must be provided and cannot be empty")
                        
        # Z-score for confidence level multiplied by the latest standard deviation
        VaR = z_score * std
        
        # Calculate ES at confidence_level %
        # Use the mask to filter the array and get only values <= -VaR
        filtered_returns = [x for x in simulated_returns if x <= -VaR]
        
        # Calculate ES as the mean of filtered returns
        ES = np.mean(filtered_returns) if len(filtered_returns) > 0 else VaR
         
        return VaR, ES 