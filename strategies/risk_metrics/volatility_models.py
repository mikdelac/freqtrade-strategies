from typing import List, Tuple, Optional, Callable
import numpy as np
from enum import Enum
import pandas as pd

class VolatilityRegime(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class VolatilityModel:
    def __init__(self, 
                 low_threshold: float = 0.01,
                 medium_threshold: float = 0.015,
                 risk_multipliers: Optional[dict] = None,
                 atr_period: int = 14):
        """
        Initialize VolatilityModel with thresholds and risk multipliers.
        
        Args:
            low_threshold: Threshold for low volatility
            medium_threshold: Threshold for medium volatility
            risk_multipliers: Dict mapping regimes to risk multipliers
            atr_period: Period for ATR calculation
        """
        self.low_threshold = low_threshold
        self.medium_threshold = medium_threshold
        self.risk_multipliers = risk_multipliers or {
            VolatilityRegime.LOW: 1.0,
            VolatilityRegime.MEDIUM: 0.8,
            VolatilityRegime.HIGH: 0.5
        }
        self.atr_period = atr_period

    def get_regime_and_multiplier(self, vol: float) -> Tuple[str, float]:
        """
        Determine volatility regime and corresponding risk multiplier
        
        Args:
            vol: Volatility value
            
        Returns:
            Tuple of (regime value, risk multiplier)
        """
        regime = self.detect_regime(vol)
        return regime.value, self.risk_multipliers[regime]

    def detect_regime(self, volatility: float) -> VolatilityRegime:
        """
        Detect current volatility regime
        
        Args:
            volatility: Volatility value
        Returns:
            VolatilityRegime: Current volatility regime
        """
        if pd.isna(volatility):
            return VolatilityRegime.MEDIUM
            
        if volatility <= self.low_threshold:
            return VolatilityRegime.LOW
        elif volatility <= self.medium_threshold:
            return VolatilityRegime.MEDIUM
        return VolatilityRegime.HIGH

    def calculate_volatility(self, returns: np.ndarray, garch_model: 'GARCHModel' = None) -> float:
        """
        Calculate volatility using GARCH model if provided, otherwise use simple historical volatility.
        
        Args:
            returns: Array of log return values
            garch_model: Optional GARCHModel instance to use for volatility calculation
            
        Returns:
            float: Estimated volatility
        """
        if garch_model is not None:
            # Use GARCH model for volatility calculation
            garch_model.reset_variance()
            return garch_model.calculate_volatility(returns)
        else:
            # Fallback to simple historical volatility
            if len(returns) == 0:
                return float('nan')
            return np.std(returns)
    
    def calculate_atr(self, dataframe: pd.DataFrame) -> pd.Series:
        """
        Calculate the Average True Range (ATR) for the given dataframe.
        
        Args:
            dataframe: DataFrame containing 'high', 'low', and 'close' price columns.
            
        Returns:
            pd.Series: ATR values.
        """
        try:
            import talib
            # Use TA-Lib ATR function
            atr = pd.Series(
                talib.ATR(
                    dataframe['high'].values,
                    dataframe['low'].values,
                    dataframe['close'].values,
                    timeperiod=self.atr_period
                ),
                index=dataframe.index
            )
        except ImportError:
            # Fallback to manual calculation if TA-Lib is not available
            high = dataframe['high']
            low = dataframe['low']
            close = dataframe['close']
            
            true_range = pd.DataFrame({
                'TR1': high - low,
                'TR2': np.abs(high - close.shift(1)),
                'TR3': np.abs(low - close.shift(1))
            }).max(axis=1)
            
            atr = true_range.rolling(window=self.atr_period, min_periods=1).mean()
            
        # Store current ATR value
        return atr

class GARCHModel():
    def __init__(self, 
                 omega: float = 0.000005,
                 alpha: float = 0.1,
                 beta: float = 0.85,
                 theta: float = 0.0,
                 risk_multipliers: Optional[dict] = None):
        """
        Initialize GARCHModel with specific parameters for GARCH(1,1) and NGARCH.

        Args:
            omega: Constant term in the GARCH equation.
            alpha: Coefficient for the squared return.
            beta: Coefficient for the lagged variance.
            theta: Leverage effect parameter for NGARCH.
            risk_multipliers: Dict mapping regimes to risk multipliers.
        """
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.theta = theta

        # Validate parameters to ensure stationarity
        if self.alpha + self.beta >= 1:
            raise ValueError("The sum of alpha and beta must be less than 1 for the GARCH(1,1) model to be stationary.")

        # Initialize variance with long-term variance
        self.variance = self.omega / (1 - self.alpha - self.beta)
        
    def reset_variance(self):
        """
        Reset the model's variance to its long-term value.
        
        This function resets the conditional variance of the GARCH model to its 
        unconditional (long-term) variance, which is calculated as:
        σ²_unconditional = ω / (1 - α - β)
        
        Returns:
            float: The reset variance value
        """
        self.variance = self.omega / (1 - self.alpha - self.beta)
        return self.variance

    def update_conditional_variance(self, R_t: float) -> float:
        """
        Update the conditional variance using the GARCH(1,1) model.
        σ²(t+1) = ω + α(R_t)² + βσ²_t

        Args:
            R_t: Return at time t.

        Returns:
            float: Updated variance at time t+1.
        """
        self.variance = self.omega + self.alpha * (R_t ** 2) + self.beta * self.variance
        return self.variance

    def update_conditional_variance_ngarch(self, R_t: float) -> float:
        """
        Update the conditional variance using the NGARCH model.
        σ²(t+1) = ω + α(R_t - θσ_t)² + βσ²_t

        Args:
            R_t: Return at time t.

        Returns:
            float: Updated variance at time t+1.
        """
        sigma_t = np.sqrt(self.variance)
        self.variance = self.omega + self.alpha * (R_t - self.theta * sigma_t)**2 + self.beta * self.variance
        return self.variance

    def calculate_volatility(self, returns: np.ndarray) -> float:
        """
        Calculate volatility using GARCH(1,1) model.
        Takes log returns and iteratively updates the variances using the GARCH(1,1) formula.
        Finally, returns the square root of the final estimated variance (standard deviation).
        
        Args:
            returns: Array of log return values (not prices).
            
        Returns:
            float: Estimated volatility (standard deviation) from GARCH model.
        """
        
        if len(returns) == 0:
            return float('nan')
        
        # Check for invalid data
        if np.isnan(returns).any() or np.isinf(returns).any():
            return float('nan')
                
        # Iteratively update variances using GARCH(1,1) formula
        for R_t in returns:
            self.update_conditional_variance(R_t)
        
        # Return the square root of the final variance (volatility/standard deviation)
        return np.sqrt(self.variance)

def garch_bbands(
    close: pd.Series,
    window: int = 20,
    k: float = 2.0,
    model: Optional[GARCHModel] = None,
    multiplicative: bool = True,
    use_ngarch: bool = False,
) -> pd.DataFrame:
    """
    Compute forward-looking Bollinger Bands using one-step-ahead GARCH volatility forecasts.

    This replaces the rolling standard deviation in traditional Bollinger Bands with the
    conditional standard deviation forecast from a GARCH model.

    Two formulations are supported:
      - multiplicative=True (preferred, lognormal):
          upper = SMA * exp(k * sigma_{t+1})
          lower = SMA * exp(-k * sigma_{t+1})
      - multiplicative=False (additive approximation):
          upper = SMA + k * sigma_{t+1} * SMA
          lower = SMA - k * sigma_{t+1} * SMA

    Args:
        close: Price close series indexed by time.
        window: Moving average window for the Bollinger midline.
        k: Band width multiplier (analogous to standard deviations).
        model: Optional pre-configured GARCHModel. If None, a default GARCHModel is created.
        multiplicative: Use the lognormal (multiplicative) bands if True, else additive approximation.
        use_ngarch: If True, uses the NGARCH update with leverage (theta) from the provided model.

    Returns:
        pd.DataFrame with columns: 'bb_mid_garch', 'bb_upper_garch', 'bb_lower_garch', 'garch_sigma_forecast'.
    """
    assert isinstance(close, pd.Series), "close must be a pandas Series"
    assert window > 0, "window must be positive"
    assert k >= 0, "k must be non-negative"

    if model is None:
        model = GARCHModel()

    # Prepare log prices and returns (length N-1)
    log_prices = np.log(close.astype(float))
    log_returns = np.diff(log_prices.to_numpy())

    # Initialize forecasts aligned to price index; first value has no forecast
    n = len(close)
    sigma_forecast = np.full(n, np.nan)

    # Reset to unconditional variance before forecasting
    model.reset_variance()

    # Iterate returns: r[i] produces variance forecast for time index i+1
    for i in range(len(log_returns)):
        r_t = log_returns[i]
        if not np.isfinite(r_t):
            continue
        if use_ngarch:
            sigma_sq_t1 = model.update_conditional_variance_ngarch(r_t)
        else:
            sigma_sq_t1 = model.update_conditional_variance(r_t)
        sigma_forecast[i + 1] = np.sqrt(sigma_sq_t1)

    # Midline via simple moving average over price (classic BB mid)
    middle_band = close.rolling(window=window, min_periods=window).mean()

    if multiplicative:
        upper_band = middle_band * np.exp(k * sigma_forecast)
        lower_band = middle_band * np.exp(-k * sigma_forecast)
    else:
        # Small-sigma additive approximation
        upper_band = middle_band + (k * sigma_forecast * middle_band)
        lower_band = middle_band - (k * sigma_forecast * middle_band)

    return pd.DataFrame(
        {
            'bb_mid_garch': middle_band,
            'bb_upper_garch': upper_band,
            'bb_lower_garch': lower_band,
            'garch_sigma_forecast': sigma_forecast,
        },
        index=close.index,
    )

def run_garch_examples() -> None:
    """
    Run GARCH examples and risk calculations for demonstration purposes.
    """
    from .monte_carlo import MonteCarloSimulator
    from ..risk_metrics.risk_indicators import RiskIndicators
    from scipy.stats import norm, t
    
    print("--------------------------------")
    print("--------------------------------")
    print("Begin Default GARCH")        
    print("---")

    # Exemple d'utilisation
    garch_model = GARCHModel()
    risk_indicators = RiskIndicators()

    # Example 1: Basic GARCH(1,1) with default parameters
    simulated_returns = [0.07, 0.06, 0.05, 0.09]
    np_array = np.array(simulated_returns)
    # Calculate log returns on arithmetics returns
    log_returns = np.log(1 + np_array)
    # Calculate log returns based on price levels
    #log_returns = np.log(prices[1:] / prices[:-1])

    #df = 5  # Can be adjusted based on empirical data
    #VaR_Z = t.ppf(1 - confidence_level, df) * std
    confidence_level = 0.01
    VaR_1_percent, ES_1_percent = risk_indicators.calculate_var_es(
        simulated_returns=log_returns,
        z_score=norm.ppf(1 - confidence_level),
        std=garch_model.calculate_volatility(log_returns)
    )
    garch_model.reset_variance()
    print(f"VaR à 1% sur 35 jours (simulation GARCH sans Monte Carlo): {VaR_1_percent:.4f} ({VaR_1_percent * 100:.2f}%)")
    print(f"ES à 1% sur 35 jours (simulation GARCH sans Monte Carlo): {ES_1_percent:.4f} ({ES_1_percent * 100:.2f}%)")

    print("--------------------------------")
    print("Begin Monte Carlo with GARCH")        
    print("---")

    # Example 2: Monte Carlo simulation with GARCH using the new MonteCarloSimulator
    monte_carlo = MonteCarloSimulator(garch_model)
    simulated_returns = monte_carlo.simulate_with_garch(T=3, iterations=1000)
    confidence_level = 0.01
    VaR_1_percent, ES_1_percent = risk_indicators.calculate_var_es(
        simulated_returns=simulated_returns,
        z_score=norm.ppf(1 - confidence_level),
        std=garch_model.calculate_volatility(simulated_returns)
    )
    print(f"VaR à 1% sur 3 jours avec 1000 simulations (GARCH avec Monte Carlo): {VaR_1_percent:.4f} ({VaR_1_percent * 100:.2f}%)")
    print(f"ES à 1% sur 3 jours avec 1000 simulations (GARCH avec Monte Carlo): {ES_1_percent:.4f} ({ES_1_percent * 100:.2f}%)")
    
    # Example 3: Monte Carlo with custom distribution (Student's t)
    print("--------------------------------")
    print("Begin Monte Carlo with Student's t-distribution")        
    print("---")
    
    simulated_returns_t = monte_carlo.simulate_with_custom_distribution(T=3, iterations=1000)
    VaR_1_percent_t, ES_1_percent_t = risk_indicators.calculate_var_es(
        simulated_returns=simulated_returns_t,
        z_score=norm.ppf(1 - confidence_level),
        std=garch_model.calculate_volatility(simulated_returns_t)
    )
    print(f"VaR à 1% sur 3 jours avec distribution t de Student: {VaR_1_percent_t:.4f} ({VaR_1_percent_t * 100:.2f}%)")
    print(f"ES à 1% sur 3 jours avec distribution t de Student: {ES_1_percent_t:.4f} ({ES_1_percent_t * 100:.2f}%)")
    
    # Display simulation statistics
    stats = monte_carlo.get_simulation_statistics(simulated_returns)
    print(f"Statistiques de simulation - Moyenne: {stats['mean']:.4f}, Écart-type: {stats['std']:.4f}")
    print(f"Skewness: {stats['skewness']:.4f}, Kurtosis: {stats['kurtosis']:.4f}")