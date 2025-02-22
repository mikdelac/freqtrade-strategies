from typing import Optional
import numpy as np
from .volatility_models import VolatilityRegime

class PositionSizer:
    def __init__(self, max_position: float = 0.10):
        self.max_position = max_position

    def calculate_position_size(
        self, 
        vix_level: float, 
        regime: VolatilityRegime,
        vix_trend: float
    ) -> float:
        """
        Calculate position size based on VIX and volatility regime
        
        Args:
            vix_level: Current VIX value
            regime: Current volatility regime
            vix_trend: VIX trend indicator
        Returns:
            float: Position size as percentage of portfolio
        """
        base_size = self.max_position * (1 / (1 + vix_level/20))  # Inverse scaling with VIX
        
        # Apply regime-based adjustments
        if regime == VolatilityRegime.HIGH:
            base_size *= 0.5
        elif regime == VolatilityRegime.MEDIUM:
            base_size *= 0.75
            
        # Reduce size if VIX is trending up
        if vix_trend > 0:
            base_size *= 0.8
            
        return min(base_size, self.max_position) 