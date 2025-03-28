from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from scipy.signal import find_peaks
import talib.abstract as ta  # Add this import

class TrendAnalysis:
    def __init__(self, 
                min_points: int = 5,
                min_slope: float = 0.001,
                min_strength: float = 0.8, #was 0.8
                angle_threshold: float = 5,
                atr_threshold: float = 0.02):  # New ATR threshold parameter
        """
        Initialize TrendAnalysis with parameters for trendline detection.
        
        Args:
            min_points: Minimum number of points required to form a trendline
            min_slope: Minimum absolute slope to consider for a trendline
            min_strength: Minimum R-squared value to consider a valid trendline
            angle_threshold: Maximum angle in degrees for valid trendlines
            atr_threshold: Minimum ATR value to confirm significant swing points
        """
        self.min_points = min_points
        self.min_slope = min_slope
        self.min_strength = min_strength
        self.angle_threshold = angle_threshold
        self.atr_threshold = atr_threshold  # Store ATR threshold
        
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
                    price_type: str = 'high',
                    min_points: int = 2,
                    distance: int = 5,
                    atr_values: np.ndarray = None) -> List[Tuple[int, float]]:
        """
        Find swing high or low points in the price array using scipy.signal.find_peaks.
        
        Args:
            prices: Array of price values
            window: Window size for peak detection (used if distance is None)
            price_type: Type of price to examine ('high' or 'low')
            min_points: Minimum number of points to identify, used only for recursive calls
            distance: Minimum horizontal distance between peaks (if None, derived from window)
            atr_values: Array of ATR values corresponding to each price point for significance filtering

        Returns:
            List of tuples containing (index, price) of swing points
        """
        from scipy.signal import find_peaks
                
        # Calculate prominence as a percentage of price range
        price_range = np.max(prices) - np.min(prices)
        prominence = price_range * 0.04  # 2% of price range as minimum prominence
        
        if price_type == 'high':
            # Find peaks (local maxima)
            peaks, _ = find_peaks(prices, distance=distance, prominence=prominence)
            swing_points = [(int(idx), float(prices[idx])) for idx in peaks]
        else:  # For 'low' prices
            # For valleys (local minima), invert the signal but keep original price values
            peaks, _ = find_peaks(-prices, distance=distance, prominence=prominence)
            swing_points = [(int(idx), float(prices[idx])) for idx in peaks]
        
        # If we don't have enough points, try with smaller distance
        if len(swing_points) < min_points and distance > 1:
            smaller_distance = max(1, distance - 1)
            return self._find_swing_points(prices, price_type, min_points, smaller_distance, atr_values)
        
        # Filter by ATR significance if provided
        if atr_values is not None and len(atr_values) > 0:
            swing_points = self._filter_by_atr_significance(prices, swing_points, atr_values, price_type)
            
        # Sort points by time index
        return sorted(swing_points, key=lambda x: x[0])
        
    def _filter_by_atr_significance(self, 
                              prices: np.ndarray, 
                              swing_points: List[Tuple[int, float]], 
                              atr_values: np.ndarray,
                              price_type: str,
                              atr_multiplier: float = 1.0) -> List[Tuple[int, float]]:
        """
        Filter swing points to include only those with price movements significant compared to their candle's ATR.
        
        Args:
            prices: Original price array
            swing_points: List of (index, price) tuples representing swing points
            atr_values: Array of ATR values for each candle
            price_type: Type of price ('high' or 'low')
            atr_multiplier: Multiplier for ATR threshold (default: 1.0)
            
        Returns:
            Filtered list of swing points with significant price swings
        """
        if not swing_points or len(atr_values) == 0:
            return swing_points
            
        significant_points = []
        
        for i, (idx, price) in enumerate(swing_points):
            # Ensure index is valid for ATR array
            if idx >= len(atr_values):
                continue
                
            # Get ATR value for this specific candle
            current_atr = atr_values[idx]
            if current_atr <= 0:
                continue
                
            min_price_change = current_atr * atr_multiplier
            
            # Check significance against neighbors
            is_significant = True
            
            # Check left neighbors (previous price points)
            if idx > 0:
                # Calculate how many points to look back
                lookback = min(5, idx)  # Look back up to 5 points or to the start
                left_min = np.min(prices[idx-lookback:idx])
                left_max = np.max(prices[idx-lookback:idx])
                
                if price_type == 'high':
                    # For high points, check if it's significantly higher than nearby points
                    if price - left_max < min_price_change:
                        is_significant = False
                else:
                    # For low points, check if it's significantly lower than nearby points
                    if left_min - price < min_price_change:
                        is_significant = False
            
            # Check right neighbors (next price points)
            if idx < len(prices) - 1:
                # Calculate how many points to look ahead
                lookahead = min(5, len(prices) - idx - 1)  # Look ahead up to 5 points or to the end
                right_min = np.min(prices[idx+1:idx+lookahead+1])
                right_max = np.max(prices[idx+1:idx+lookahead+1])
                
                if price_type == 'high':
                    # For high points, check if it's significantly higher than nearby points
                    if price - right_max < min_price_change:
                        is_significant = False
                else:
                    # For low points, check if it's significantly lower than nearby points
                    if right_min - price < min_price_change:
                        is_significant = False
            
            if is_significant:
                significant_points.append((idx, price))
        
        return significant_points


    def calculate_talib_linearreg(self, dataframe: pd.DataFrame, timeperiod: int = 14, 
                                 price_field: str = 'close') -> pd.DataFrame:
        """
        Calculate and visualize a linear regression trendline using TA-Lib's LINEARREG functions.
        
        This method creates a true straight line representing the linear regression for the most
        recent data window. Unlike the default TA-Lib behavior which calculates a new regression
        value at each point, this method:
        
        1. Uses TA-Lib's LINEARREG functions to calculate regression parameters (slope, intercept)
        2. Identifies the most recent valid regression point
        3. Draws a single straight line covering the most recent timeperiod candles
        4. Adds a forecast line extending a few candles into the future
        
        This approach results in a clean, straight line on the chart that clearly shows the current
        trend direction as determined by linear regression.
        
        Args:
            dataframe: Price dataframe with OHLCV data
            timeperiod: Lookback period for calculating linear regression (default: 14)
            price_field: Which price to use for regression (default: 'close')
            
        Returns:
            DataFrame with added linear regression indicators and a straight trendline
        """
        # Make a copy to avoid modifying the original dataframe
        result = dataframe.copy()
        
        # Get the price series for regression
        price_series = result[price_field]
        
        # Calculate linear regression line endpoint values
        # This gives the projected value of the regression line at each point
        result['linear_reg'] = ta.LINEARREG(price_series, timeperiod=timeperiod)
        
        # Calculate slope of the regression line
        result['linear_reg_slope'] = ta.LINEARREG_SLOPE(price_series, timeperiod=timeperiod)
        
        # Calculate angle of the regression line (in degrees)
        result['linear_reg_angle'] = ta.LINEARREG_ANGLE(price_series, timeperiod=timeperiod)
        
        # Calculate intercept of the regression line
        result['linear_reg_intercept'] = ta.LINEARREG_INTERCEPT(price_series, timeperiod=timeperiod)
        
        # Calculate linear regression-based forecast (TSF) for one period ahead
        result['linear_reg_forecast'] = ta.TSF(price_series, timeperiod=timeperiod)
        
        # Initialize column for the linear regression line
        result['linear_reg_line'] = np.nan
        
        # We'll draw a proper straight line using only the latest valid regression data
        # Find the last valid regression point
        last_valid_idx = None
        last_valid_slope = None
        last_valid_intercept = None
        
        # Start from the end of the dataframe and find the last valid regression point
        for i in range(len(result)-1, timeperiod-2, -1):
            if not (np.isnan(result.loc[i, 'linear_reg_slope']) or np.isnan(result.loc[i, 'linear_reg_intercept'])):
                last_valid_idx = i
                last_valid_slope = result.loc[i, 'linear_reg_slope']
                last_valid_intercept = result.loc[i, 'linear_reg_intercept']
                break
        
        # If we found a valid point, draw the line
        if last_valid_idx is not None:
            # Calculate line for the most recent timeperiod candles
            # This will ensure we have a single straight line on the chart
            start_idx = max(0, last_valid_idx - timeperiod + 1)
            end_idx = last_valid_idx
            
            # Draw the line for these candles
            for i in range(start_idx, end_idx + 1):
                # Calculate x relative to the start of the segment (0-based)
                x_value = i - start_idx
                # Calculate y using the linear regression formula: y = mx + b
                line_value = last_valid_slope * x_value + last_valid_intercept
                # Store the line value
                result.loc[i, 'linear_reg_line'] = line_value
            
            # Add a forecast line extending a few candles into the future
            forecast_periods = 5
            for i in range(1, forecast_periods + 1):
                if end_idx + i < len(result):
                    forecast_x = timeperiod - 1 + i
                    forecast_value = last_valid_slope * forecast_x + last_valid_intercept
                    result.loc[end_idx + i, 'linear_reg_forecast'] = forecast_value
        
        return result

