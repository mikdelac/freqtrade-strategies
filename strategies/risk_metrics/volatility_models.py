from typing import List, Tuple, Optional
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
                 atr_period: int = 14 #ATR period usually 14 days
                 ):
        self.low_threshold = low_threshold
        self.medium_threshold = medium_threshold
        self.risk_multipliers = risk_multipliers or {
            VolatilityRegime.LOW: 1.0,
            VolatilityRegime.MEDIUM: 0.8,
            VolatilityRegime.HIGH: 0.5
        }
        self.atr_period = atr_period
        self.current_atr = None  # To store the latest ATR value

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

    def calculate_volatility(self, prices: np.ndarray) -> float:
        """Calculate historical volatility"""
        returns = np.log(prices[1:] / prices[:-1])
        return np.std(returns)
    
    def calculate_atr(self, dataframe: pd.DataFrame) -> pd.Series:
        """
        Calculate the Average True Range (ATR) for the given dataframe.
        
        Args:
            dataframe: DataFrame containing 'high', 'low', and 'close' price columns.
            
        Returns:
            pd.Series: ATR values.
        """
        high = dataframe['high']
        low = dataframe['low']
        close = dataframe['close']
        
        true_range = pd.DataFrame({
            'TR1': high - low,
            'TR2': np.abs(high - close.shift(1)),
            'TR3': np.abs(low - close.shift(1))
        }).max(axis=1)
        
        atr = true_range.rolling(window=self.atr_period, min_periods=1).mean()
        self.current_atr = atr.iloc[-1]
        return atr

    def get_atr(self) -> Optional[float]:
        """Retrieve the latest ATR value."""
        return self.current_atr 