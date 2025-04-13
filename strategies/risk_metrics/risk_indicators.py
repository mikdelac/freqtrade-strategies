from typing import List, Tuple, Optional, Callable
import numpy as np
from scipy.stats import norm, t
from dataclasses import dataclass
from strategies.risk_metrics.volatility_models import GARCHModel

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
                
    def calculate_var_es(self, 
                         simulated_returns: Optional[np.ndarray] = None,
                         confidence_level: float = 0.01,
                         variance_update_callback: Optional[Callable[[float, float], float]] = None) -> Tuple[float, float]:
        """
        Calculate Value at Risk (VaR) and Expected Shortfall (ES) at a specified confidence level over a given period
        using Monte Carlo simulation.
        
        Args:
            simulated_returns: Optional pre-computed simulated returns with shape (iterations, T)
            confidence_level: Confidence level for VaR and ES (e.g., 0.01 for 1%)
            variance_update_callback: Callback function to update conditional variance with signature (sigma2_t, R_t) -> new_sigma2_t
        
        Returns:
            Tuple[float, float]: VaR and ES at the specified confidence level
            
        Raises:
            ValueError: If no variance_update_callback is provided
        """
        if variance_update_callback is None:
            raise ValueError("variance_update_callback is required for calculating VaR and ES")
            
        if simulated_returns is None or len(simulated_returns) == 0:
            raise ValueError("simulated_returns must be provided and cannot be empty")
            
        # Initialize GARCH model with desired parameters
        garch = GARCHModel(omega=0.000005, alpha=0.1, beta=0.85)
        
        # Apply conditional variance update using the provided callback
        latest_variance = garch.variance  # Start with initial variance from GARCH model
        for R_t in simulated_returns:
            latest_variance = variance_update_callback(R_t)
        
        # Calculate VaR using the latest conditional standard deviation
        latest_std = np.sqrt(latest_variance)
        print(f"latest_std: {latest_std}")
        # Z-score for confidence level multiplied by the latest standard deviation
        VaR_N = norm.ppf(1 - confidence_level) * latest_std
        print(f"VaR with normal distribution z-score: {VaR_N}")
        df = 5  # Can be adjusted based on empirical data
        VaR_Z = t.ppf(1 - confidence_level, df) * latest_std
        print(f"VaR with Student-t distribution (df={df}): {VaR_Z}")
        VaR = VaR_N
        
        # Calculate ES at confidence_level %
        # Use the mask to filter the array and get only values <= -VaR
        filtered_returns = [x for x in simulated_returns if x <= -VaR]
        #print(f"filtered_returns: {filtered_returns}")
        # Calculate ES as the mean of filtered returns
        ES = np.mean(filtered_returns) if len(filtered_returns) > 0 else VaR
        
        print(f"ES: {ES}")
 
        return VaR, ES 