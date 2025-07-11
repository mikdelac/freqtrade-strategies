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
    
    def __init__(self, distance: int = 10, prominence: float = 0.12):
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
        Find swing high or low points based on price changes using scipy.signal.find_peaks.
        
        This method detects abrupt price movements by analyzing the differences between
        consecutive prices rather than the raw price values.
        
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
        
        # Calculate price changes (differences)
        price_changes = np.diff(prices)
        
        # Handle edge case where we have less than 2 prices
        if len(price_changes) == 0:
            return []
        
        # Calculate prominence based on the range of price changes
        change_range = np.max(price_changes) - np.min(price_changes)
        prominence_value = change_range * prominence
        
        if price_type == 'high':
            # Find peaks in positive price changes (abrupt upward movements)
            peaks, _ = find_peaks(price_changes, distance=distance, prominence=prominence_value)
            # Add 1 to peak indices because price_changes is one element shorter than prices
            # The peak at index i in price_changes corresponds to the candle at index i+1 in prices
            swing_points = [(int(idx + 1), float(prices[idx + 1])) for idx in peaks if idx + 1 < len(prices)]
        else:  # For 'low' prices
            # Find peaks in negative price changes (abrupt downward movements)
            # Invert the price changes to find valleys as peaks
            peaks, _ = find_peaks(-price_changes, distance=distance, prominence=prominence_value)
            # Add 1 to peak indices because price_changes is one element shorter than prices
            swing_points = [(int(idx + 1), float(prices[idx + 1])) for idx in peaks if idx + 1 < len(prices)]
                    
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
    
