from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum

class TrendDirection(Enum):
    UP = "up"
    DOWN = "down"
    SIDEWAYS = "sideways"

@dataclass
class Trendline:
    start_index: int
    end_index: int
    slope: float
    intercept: float
    direction: TrendDirection
    strength: float  # R-squared value
    price_type: str  # 'high' or 'low'
    
class TrendAnalysis:
    def __init__(self, 
                min_points: int = 5,
                min_slope: float = 0.0001,
                min_strength: float = 0.8, #was 0.8
                angle_threshold: float = 45):
        """
        Initialize TrendAnalysis with parameters for trendline detection.
        
        Args:
            min_points: Minimum number of points required to form a trendline
            min_slope: Minimum absolute slope to consider for a trendline
            min_strength: Minimum R-squared value to consider a valid trendline
            angle_threshold: Maximum angle in degrees for valid trendlines
        """
        self.min_points = min_points
        self.min_slope = min_slope
        self.min_strength = min_strength
        self.angle_threshold = angle_threshold
        
    def _calculate_linear_regression(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate linear regression parameters and R-squared value.
        
        Args:
            x: x-coordinates (time indices)
            y: y-coordinates (price values)
            
        Returns:
            Tuple of (slope, intercept, r_squared)
        """
        if len(x) < 2:
            return 0, 0, 0
            
        # Normalize x to be relative to the start of the window
        x_norm = x - x[0]
            
        coeffs = np.polyfit(x_norm, y, 1)
        slope, intercept = coeffs
        
        # Calculate R-squared
        y_pred = slope * x_norm + intercept
        r_squared = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))
        
        return slope, intercept, r_squared
        
    def _is_valid_trendline(self, 
                         slope: float, 
                         strength: float, 
                         points: int) -> bool:
        """
        Check if a potential trendline meets validity criteria.
        
        Args:
            slope: Slope of the trendline
            strength: R-squared value of the fit
            points: Number of points used to create the trendline
            
        Returns:
            bool: Whether the trendline is valid
        """
        if points < self.min_points:
            print("points < self.min_points")
            return False
            
        if abs(slope) < self.min_slope:
            print("abs(slope) < self.min_slope")
            return False
            
        if strength < self.min_strength:
            print("strength < self.min_strength")
            return False
            
        # Check angle is not too steep
        angle = abs(np.degrees(np.arctan(slope)))
        if angle > self.angle_threshold:
            print("angle > self.angle_threshold")
            return False
            
        return True
        
    def find_trendlines(self, 
                     dataframe: pd.DataFrame, 
                     window: int = 20,
                     price_type: str = 'close') -> List[Trendline]:
        """
        Find potential trendlines in the given price data.
        
        Args:
            dataframe: DataFrame with price data
            window: Rolling window size for trendline detection
            price_type: Which price to use ('high', 'low', 'close')
            
        Returns:
            List of detected Trendline objects
        """
        trendlines = []
        prices = dataframe[price_type].values
        indices = np.arange(len(prices))
        
        for i in range(len(prices) - window + 1):
            window_prices = prices[i:i+window]
            window_indices = indices[i:i+window]
            
            # Calculate trendline parameters
            slope, intercept, strength = self._calculate_linear_regression(
                window_indices, 
                window_prices
            )
            
            if self._is_valid_trendline(slope, strength, len(window_prices)):
                direction = (TrendDirection.UP if slope > 0 
                           else TrendDirection.DOWN if slope < 0 
                           else TrendDirection.SIDEWAYS)
                
                trendline = Trendline(
                    start_index=i,
                    end_index=i+window-1,
                    slope=slope,
                    intercept=intercept,
                    direction=direction,
                    strength=strength,
                    price_type=price_type
                )
                trendlines.append(trendline)
        
        return trendlines
        
    def get_current_trendline(self, 
                           dataframe: pd.DataFrame,
                           lookback_period: int = 20,
                           price_type: str = 'close') -> Optional[Trendline]:
        """
        Get the most recent valid trendline.
        
        Args:
            dataframe: DataFrame with price data
            lookback_period: Number of candles to look back
            price_type: Which price to use ('high', 'low', 'close')
            
        Returns:
            Most recent valid Trendline or None if no valid trendline found
        """
        print("lookback_period: ", lookback_period)
        if len(dataframe) < lookback_period:
            return None
        # Get recent data
        recent_data = dataframe.iloc[-lookback_period:]
        prices = recent_data[price_type].values
        indices = np.arange(len(prices))
        # Calculate trendline parameters
        slope, intercept, strength = self._calculate_linear_regression(indices, prices)
        print("slope: ", slope)
        print("intercept: ", intercept)
        print("strength: ", strength)
        if self._is_valid_trendline(slope, strength, len(prices)):
            direction = (TrendDirection.UP if slope > 0 
                       else TrendDirection.DOWN if slope < 0 
                       else TrendDirection.SIDEWAYS)
            
            return Trendline(
                start_index=len(dataframe) - lookback_period,
                end_index=len(dataframe) - 1,
                slope=slope,
                intercept=intercept,
                direction=direction,
                strength=strength,
                price_type=price_type
            )
        
        return None
        
    def extrapolate_trendline(self, 
                           trendline: Trendline, 
                           steps_forward: int = 10) -> np.ndarray:
        """
        Extrapolate a trendline forward in time.
        
        Args:
            trendline: Trendline object to extrapolate
            steps_forward: Number of steps to project forward
            
        Returns:
            Array of projected price values
        """
        future_indices = np.arange(
            trendline.end_index + 1,
            trendline.end_index + steps_forward + 1
        )
        return trendline.slope * future_indices + trendline.intercept
        
    def detect_breakout(self, 
                     trendline: Trendline,
                     current_price: float,
                     threshold: float = 0.02) -> bool:
        """
        Detect if current price represents a breakout from trendline.
        
        Args:
            trendline: Current trendline
            current_price: Latest price
            threshold: Minimum percentage deviation to consider a breakout
            
        Returns:
            bool: True if price has broken out of trendline
        """
        # Calculate expected price at current point
        current_idx = trendline.end_index + 1
        expected_price = trendline.slope * current_idx + trendline.intercept
        
        # Calculate percentage deviation
        deviation = abs(current_price - expected_price) / expected_price
        
        return deviation > threshold 