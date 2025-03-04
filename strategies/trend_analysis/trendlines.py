from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from scipy.signal import find_peaks

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
    validation_points: List[Tuple[int, float]] = None
    
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
        
    def _find_swing_points(self, 
                    prices: np.ndarray, 
                    window: int = 5,
                    price_type: str = 'high',
                    min_points: int = 3,
                    distance: int = None) -> List[Tuple[int, float]]:
        """
        Find swing high or low points in the price array.
        
        Args:
            prices: Array of price values
            window: Window size for peak detection (used if distance is None)
            price_type: Type of price to examine ('high' or 'low')
            min_points: Minimum number of points to identify, used only for recursive calls
            distance: Minimum horizontal distance between peaks (if None, derived from window)

        Returns:
            List of tuples containing (index, price) of swing points
        """
        from scipy.signal import find_peaks
        
        if len(prices) < window:
            return []

        # Use provided distance or derive from window
        if distance is None:
            distance = max(1, window // 2)
        
        # Calculate prominence as a percentage of price range
        price_range = np.max(prices) - np.min(prices)
        prominence = price_range * 0.01  # 1% of price range as minimum prominence
        
        if price_type == 'high':
            # Find peaks (local maxima)
            peaks, _ = find_peaks(prices, distance=distance, prominence=prominence)
            swing_points = [(int(idx), float(prices[idx])) for idx in peaks]
            print("high swing_points: ", swing_points)
        else:  # For 'low' prices
            # For valleys (local minima), invert the signal but keep original price values
            peaks, _ = find_peaks(-prices, distance=distance, prominence=prominence)
            swing_points = [(int(idx), float(prices[idx])) for idx in peaks]
        
        # If we don't have enough points, try with smaller distance
        if len(swing_points) < min_points and distance > 1:
            smaller_distance = max(1, distance - 1)
            return self._find_swing_points(prices, window, price_type, min_points, smaller_distance)
            
        # Sort points by time index
        return sorted(swing_points, key=lambda x: x[0])

    def find_trendlines(self, 
                     dataframe: pd.DataFrame, 
                     window: int = 20,
                     price_type: str = 'close',
                     min_points: int = 3) -> List[Trendline]:
        """
        Find potential trendlines by connecting local maxima/minima.
        
        Args:
            dataframe: DataFrame with price data
            window: Rolling window size for trendline detection
            price_type: Which price to use ('high', 'low', 'close')
            min_points: Minimum number of points to form a trendline
            
        Returns:
            List of detected Trendline objects
        """
        trendlines = []
        prices = dataframe[price_type].values
        indices = np.arange(len(prices))
        
        # Find swing points
        swing_points = self._find_swing_points(prices, window=5, price_type=price_type, min_points=min_points)
        
        if len(swing_points) < min_points:
            return []
            
        # Try connecting different combinations of swing points
        for i in range(len(swing_points) - 1):
            for j in range(i + 1, len(swing_points)):
                start_idx, start_price = swing_points[i]
                end_idx, end_price = swing_points[j]
                
                # Calculate trendline parameters
                x = np.array([start_idx, end_idx])
                y = np.array([start_price, end_price])
                slope, intercept = np.polyfit(x, y, 1)
                
                # Check angle
                angle = abs(np.degrees(np.arctan(slope)))
                if angle > self.angle_threshold:
                    continue
                    
                # Find validation points between these two points
                validation_points = []
                for k in range(len(swing_points)):
                    if k != i and k != j:  # Skip the endpoints
                        idx, price = swing_points[k]
                        if start_idx < idx < end_idx:  # Only consider points between endpoints
                            expected_price = slope * idx + intercept
                            deviation = abs(price - expected_price) / price
                            if deviation <= 0.02:  # 2% tolerance
                                validation_points.append((idx, price))
                
                # Calculate strength based on number of validation points
                strength = len(validation_points) / (end_idx - start_idx)
                
                if strength >= self.min_strength:
                    direction = (TrendDirection.UP if slope > 0 
                               else TrendDirection.DOWN if slope < 0 
                               else TrendDirection.SIDEWAYS)
                    
                    trendline = Trendline(
                        start_index=start_idx,
                        end_index=end_idx,
                        slope=slope,
                        intercept=intercept,
                        direction=direction,
                        strength=strength,
                        price_type=price_type,
                        validation_points=validation_points
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