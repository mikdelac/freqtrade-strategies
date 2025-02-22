from typing import List, Tuple, Optional
import numpy as np
from enum import Enum

class VolatilityRegime(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class VolatilityModel:
    def __init__(self, window_size: int = 20):
        self.window_size = window_size

    def detect_regime(self, prices: np.ndarray) -> VolatilityRegime:
        """
        Detect current volatility regime
        
        Args:
            prices: Array of historical prices
        Returns:
            VolatilityRegime: Current volatility regime
        """
        volatility = self.calculate_volatility(prices)
        if volatility <= 15:
            return VolatilityRegime.LOW
        elif volatility <= 25:
            return VolatilityRegime.MEDIUM
        return VolatilityRegime.HIGH

    def calculate_volatility(self, prices: np.ndarray) -> float:
        """Calculate historical volatility"""
        returns = np.log(prices[1:] / prices[:-1])
        return np.std(returns) * np.sqrt(252)  # Annualized volatility 