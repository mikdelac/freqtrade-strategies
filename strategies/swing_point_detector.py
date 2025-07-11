"""
Swing Point Detector - Market Extrema Detection

This module provides swing point detection functionality for identifying
local highs and lows in price data using scipy.signal.find_peaks.

Author: Extracted from TrendAnalysis for modular design
"""

import numpy as np
import pandas as pd
from typing import List, Tuple
from scipy.signal import find_peaks


class SwingPointDetector:
    """
    A class for detecting swing points (local extrema) in price data.
    
    This class provides methods to identify swing highs and lows using
    configurable parameters for distance and prominence.
    """
    
    def __init__(self, distance: int = 10, prominence: float = 0.04):
        """
        Initialize SwingPointDetector with default parameters.
        
        Args:
            distance: Minimum horizontal distance between peaks
            prominence: Minimum prominence percentage for peaks
        """
        self.distance = distance
        self.prominence = prominence
    
    def find_swing_points(self, 
                         prices: np.ndarray, 
                         price_type: str = 'high',
                         distance: int = None,
                         prominence: float = None) -> List[Tuple[int, float]]:
        """
        Find swing high or low points in the price array using scipy.signal.find_peaks.
        
        Args:
            prices: Array of price values
            price_type: Type of price to examine ('high' or 'low')
            distance: Minimum horizontal distance between peaks (overrides default)
            prominence: Minimum prominence percentage for peaks (overrides default)

        Returns:
            List of tuples containing (index, price) of swing points
        """
        # Use provided parameters or fall back to instance defaults
        distance = distance if distance is not None else self.distance
        prominence = prominence if prominence is not None else self.prominence
        
        price_range = np.max(prices) - np.min(prices)
        prominence_value = price_range * prominence
        
        if price_type == 'high':
            # Find peaks (local maxima)
            peaks, _ = find_peaks(prices, distance=distance, prominence=prominence_value)
            swing_points = [(int(idx), float(prices[idx])) for idx in peaks]
        else:  # For 'low' prices
            # For valleys (local minima), invert the signal but keep original price values
            peaks, _ = find_peaks(-prices, distance=distance, prominence=prominence_value)
            swing_points = [(int(idx), float(prices[idx])) for idx in peaks]
                    
        # Sort points by time index
        return sorted(swing_points, key=lambda x: x[0])
    
    def find_and_map_swing_points(self, dataframe: pd.DataFrame, 
                                 recent_data: pd.DataFrame = None,
                                 distance: int = None,
                                 prominence: float = None) -> Tuple[List, List]:
        """
        Find swing points and map them to dataframe columns.
        
        Args:
            dataframe: Full dataframe to store swing points in
            recent_data: Recent data to analyze (if None, uses full dataframe)
            distance: Minimum horizontal distance between peaks (overrides default)
            prominence: Minimum prominence percentage for peaks (overrides default)
            
        Returns:
            Tuple[List, List]: (mapped_highs, mapped_lows) for the full dataframe
        """
        # Use recent_data if provided, otherwise use full dataframe
        analysis_data = recent_data if recent_data is not None else dataframe
        
        # Find swing highs and lows
        highs = self.find_swing_points(analysis_data['high'].values, 'high', distance, prominence)
        lows = self.find_swing_points(analysis_data['low'].values, 'low', distance, prominence)
        
        # Initialize the swing point columns with NaN values
        dataframe.loc[:, 'all_highs'] = np.nan
        dataframe.loc[:, 'all_lows'] = np.nan
        
        # Calculate the offset to map analysis_data indices to dataframe indices
        offset = len(dataframe) - len(analysis_data)
        
        # Map swing highs
        for idx, price in highs:
            if 0 <= idx < len(analysis_data):
                mapped_index = offset + idx
                if 0 <= mapped_index < len(dataframe):
                    dataframe.loc[dataframe.index[mapped_index], 'all_highs'] = price
        
        # Map swing lows
        for idx, price in lows:
            if 0 <= idx < len(analysis_data):
                mapped_index = offset + idx
                if 0 <= mapped_index < len(dataframe):
                    dataframe.loc[dataframe.index[mapped_index], 'all_lows'] = price
        
        # Return the mapped arrays for backward compatibility
        mapped_highs = dataframe['all_highs'].values.tolist()
        mapped_lows = dataframe['all_lows'].values.tolist()
        
        return mapped_highs, mapped_lows
    
    def get_swing_highs_lows_series(self, dataframe: pd.DataFrame,
                                   distance: int = None,
                                   prominence: float = None) -> Tuple[pd.Series, pd.Series]:
        """
        Get swing points as pandas Series for easy use with other functions.
        
        Args:
            dataframe: DataFrame with OHLCV data
            distance: Minimum horizontal distance between peaks (overrides default)
            prominence: Minimum prominence percentage for peaks (overrides default)
            
        Returns:
            Tuple[pd.Series, pd.Series]: (swing_highs_series, swing_lows_series)
        """
        # Find swing points
        highs = self.find_swing_points(dataframe['high'].values, 'high', distance, prominence)
        lows = self.find_swing_points(dataframe['low'].values, 'low', distance, prominence)
        
        # Create series with NaN values
        swing_highs = pd.Series(index=dataframe.index, dtype=float)
        swing_lows = pd.Series(index=dataframe.index, dtype=float)
        
        # Fill in swing points
        for idx, price in highs:
            if 0 <= idx < len(dataframe):
                swing_highs.iloc[idx] = price
        
        for idx, price in lows:
            if 0 <= idx < len(dataframe):
                swing_lows.iloc[idx] = price
        
        return swing_highs, swing_lows 