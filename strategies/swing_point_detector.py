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
    
    def __init__(self, distance: int = 10, prominence: float = 0.05, wlen: int = None, width: int = None):
        """
        Initialize SwingPointDetector with default parameters.
        
        Args:
            distance: Minimum horizontal distance between peaks
            prominence: Minimum prominence percentage for peaks
            wlen: Minimum width of peaks
            width: Minimum width of peaks
        """
        self.distance = distance
        self.prominence = prominence
        self.wlen = wlen
        self.width = width

    def find_swing_points(self, 
                         prices: np.ndarray, 
                         price_type: str = 'high',
                         distance: int = None,
                         prominence: float = None,
                         width: int = None,
                         wlen: int = None) -> List[Tuple[int, float]]:
        """
        Find swing high or low points based on prices using scipy.signal.find_peaks.
        
        This method detects local maxima and minima directly from the price values.
        
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
        width = width if width is not None else self.width
        wlen = wlen if wlen is not None else self.wlen

        # Handle edge case where we have less than 2 prices
        if len(prices) < 2:
            return []
        
        # Calculate prominence based on the price range
        price_range = np.max(prices) - np.min(prices)
        prominence_value = price_range * prominence
        
        if price_type == 'high':
            # Find peaks in prices (local maxima)
            peaks, _ = find_peaks(prices, distance=distance, prominence=prominence_value, width=width, wlen=wlen)
            swing_points = [(int(idx), float(prices[idx])) for idx in peaks]
        else:  # For 'low' prices
            # Find valleys in prices (local minima) by inverting the signal
            peaks, _ = find_peaks(-prices, distance=distance, prominence=prominence_value, width=width, wlen=wlen)
            swing_points = [(int(idx), float(prices[idx])) for idx in peaks]
                    
        # Sort points by time index
        return sorted(swing_points, key=lambda x: x[0])
    
    def find_and_map_swing_points(self, dataframe: pd.DataFrame) -> Tuple[List, List]:
        """
        Find swing points and map them to dataframe columns.
        
        Args:
            dataframe: Full dataframe to store swing points in
            distance: Minimum horizontal distance between peaks (overrides default)
            prominence: Minimum prominence percentage for peaks (overrides default)
            
        Returns:
            Tuple[List, List]: (mapped_highs, mapped_lows) for the full dataframe
        """
        
        # Find swing highs and lows
        highs = self.find_swing_points(dataframe['high'].values, 'high')
        lows = self.find_swing_points(dataframe['low'].values, 'low')
                
        # Initialize the swing point columns with NaN values
        dataframe.loc[:, 'all_highs'] = np.nan
        dataframe.loc[:, 'all_lows'] = np.nan
        
        # Calculate the offset to map analysis_data indices to dataframe indices
        offset = 0

        analysis_data = dataframe
        
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
    
