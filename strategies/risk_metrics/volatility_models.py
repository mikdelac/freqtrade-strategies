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