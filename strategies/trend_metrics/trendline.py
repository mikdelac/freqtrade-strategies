"""
defines trendline based indicator logic
based on
https://github.com/dysonance/Trendy
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Union, Tuple, Dict, Any, List
from scipy import stats
import random
import time


class Trendline:
    """
    A class to store and manage trendline information.
    
    This class encapsulates all relevant information about a trendline including
    its type, temporal boundaries, mathematical equation parameters, and lifecycle data.
    """
    
    def __init__(self, 
                 trendline_type: str,
                 start_time: Union[datetime, pd.Timestamp],
                 end_time: Union[datetime, pd.Timestamp],
                 slope: float,
                 start_price: float,
                 r_squared: float,
                 bounce_count: int = -1,
                 creation_time: Optional[Union[datetime, pd.Timestamp]] = None,
                 bounce_timestamps: Optional[list] = None):
        """
        Initialize a Trendline object.
        
        Args:
            trendline_type: Type of trendline ('support' or 'resistance')
            start_time: Timestamp when the trendline begins
            end_time: Timestamp when the trendline ends
            slope: Slope of the trendline (price change per time unit)
            start_price: Price at the start point
            r_squared: R-squared coefficient indicating trendline fit quality (0.0 to 1.0)
            bounce_count: Number of times price has bounced off this trendline (default: -1)
            creation_time: When this trendline object was created (defaults to current time)
            bounce_timestamps: List of timestamps where bounces occurred (defaults to empty list)
        """
        self.trendline_type = trendline_type.lower()
        self.start_time = pd.Timestamp(start_time)
        self.end_time = pd.Timestamp(end_time)
        self.slope = slope
        self.start_price = start_price
        self.r_squared = r_squared
        self.bounce_count = bounce_count
        self.creation_time = pd.Timestamp(creation_time) if creation_time else pd.Timestamp.now()
        self.bounce_timestamps = bounce_timestamps or []
        
        # Validation
        if self.trendline_type not in ['support', 'resistance']:
            raise ValueError("trendline_type must be either 'support' or 'resistance'")
        
        if self.start_time >= self.end_time:
            raise ValueError("start_time must be before end_time")
            
        if self.r_squared < 0.0 or self.r_squared > 1.0:
            raise ValueError("r_squared must be between 0.0 and 1.0")
    
    @property
    def duration(self) -> timedelta:
        """
        Calculate the duration of the trendline.
        
        Returns:
            timedelta: Duration between start and end time
        """
        return self.end_time - self.start_time
    
    @property
    def age(self) -> timedelta:
        """
        Calculate how long this trendline has been active since creation.
        
        Returns:
            timedelta: Time since the trendline was created
        """
        return pd.Timestamp.now() - self.creation_time
    
    @property
    def duration_hours(self) -> float:
        """
        Get duration in hours.
        
        Returns:
            float: Duration in hours
        """
        return self.duration.total_seconds() / 3600
    
    @property
    def age_hours(self) -> float:
        """
        Get age in hours.
        
        Returns:
            float: Age in hours
        """
        return self.age.total_seconds() / 3600
    
    def get_price_at_time(self, timestamp: Union[datetime, pd.Timestamp]) -> float:
        """
        Calculate the trendline price at a specific timestamp using the linear equation.
        
        The slope is in units of price change per candle period, so we need to convert
        time differences to candle periods for proper calculation.
        
        Args:
            timestamp: The timestamp to calculate price for
            
        Returns:
            float: Price at the given timestamp
        """
        timestamp = pd.Timestamp(timestamp)
        
        # Calculate time difference in seconds
        time_diff_seconds = (timestamp - self.start_time).total_seconds()
        
        # Assume 1-minute candles for time period conversion (most common timeframe)
        # This converts time difference to number of candles since start_time
        candle_periods = time_diff_seconds / 60.0  # 60 seconds per 1-minute candle
        
        # Use linear equation: price = slope * candle_periods + start_price
        # where slope is price change per candle period
        return self.slope * candle_periods + self.start_price
    
    def is_active_at_time(self, timestamp: Union[datetime, pd.Timestamp]) -> bool:
        """
        Check if the trendline is active (valid) at a specific timestamp.
        
        Args:
            timestamp: The timestamp to check
            
        Returns:
            bool: True if trendline is active at the given time
        """
        timestamp = pd.Timestamp(timestamp)
        return self.start_time <= timestamp <= self.end_time
    
    def extend_end_time(self, new_end_time: Union[datetime, pd.Timestamp]) -> None:
        """
        Extend the trendline's end time.
        
        Args:
            new_end_time: New end time for the trendline
        """
        new_end_time = pd.Timestamp(new_end_time)
        if new_end_time <= self.end_time:
            raise ValueError("new_end_time must be after current end_time")
        
        self.end_time = new_end_time
    
    def to_dict(self) -> dict:
        """
        Convert trendline to dictionary representation.
        
        Returns:
            dict: Dictionary containing all trendline information
        """
        return {
            'trendline_type': self.trendline_type,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'slope': self.slope,
            'start_price': self.start_price,
            'bounce_count': self.bounce_count,
            'creation_time': self.creation_time,
            'duration_hours': self.duration_hours,
            'age_hours': self.age_hours,
            'bounce_timestamps': self.bounce_timestamps,
            'r_squared': self.r_squared
        }
    
    def __str__(self) -> str:
        """String representation of the trendline."""
        return (f"Trendline({self.trendline_type.title()}: "
                f"{self.start_time.strftime('%Y-%m-%d %H:%M')} to "
                f"{self.end_time.strftime('%Y-%m-%d %H:%M')}, "
                f"slope={self.slope:.6f}, bounces={self.bounce_count}, "
                f"duration={self.duration_hours:.1f}h, R²={self.r_squared:.3f})")
    
    def __repr__(self) -> str:
        """Detailed representation of the trendline."""
        return (f"Trendline(type='{self.trendline_type}', "
                f"start_time='{self.start_time}', end_time='{self.end_time}', "
                f"slope={self.slope}, start_price={self.start_price}, "
                f"bounce_count={self.bounce_count}, r_squared={self.r_squared})")


def generate_bounce_conditions(close_data, level_data, direction: str, tolerance: float = 0.00005, pivot_highs=None, pivot_lows=None):
    """
    Generate bounce conditions for support or resistance levels.
    This is the centralized bounce detection logic used throughout the system.
    
    Args:
        close_data: Close price series
        level_data: Trendline level series
        direction: Either 'long' for support bounce or 'short' for resistance bounce
        tolerance: Proximity tolerance (default: 0.00005 or 0.005%)
        pivot_highs: Pre-calculated pivot high points (pandas Series with same index) - REQUIRED
        pivot_lows: Pre-calculated pivot low points (pandas Series with same index) - REQUIRED
        
    Returns:
        pandas.Series: Boolean series indicating bounce conditions
    """
    if direction == 'long':
        if pivot_lows is None:
            raise ValueError("pivot_lows must be provided for long direction bounce detection")
            
        # Long entry: Bounce off support using pre-calculated pivot lows
        bounce_conditions = (
            # Current close is above support
            (close_data > level_data) &
            # Previous candle had a pivot low (swing low extrema)
            (~pivot_lows.shift(1).isna()) &
            # Pivot low must be above or at the support level
            (pivot_lows.shift(1) >= level_data.shift(1)) &
            # Current close is higher than previous close (upward movement)
            (close_data > close_data.shift(1)) 
        )
    elif direction == 'short':
        if pivot_highs is None:
            raise ValueError("pivot_highs must be provided for short direction bounce detection")
            
        # Short entry: Bounce off resistance using pre-calculated pivot highs
        bounce_conditions = (
            # Current close is below resistance
            (close_data < level_data) &
            # Previous candle had a pivot high (swing high extrema)
            (~pivot_highs.shift(1).isna()) &
            # Pivot high must be below or at the resistance level
            (pivot_highs.shift(1) <= level_data.shift(1)) &
            # Current close is lower than previous close (downward movement)
            (close_data < close_data.shift(1)) 
        )
    else:
        return pd.Series([False] * len(close_data), index=close_data.index)
        
    return bounce_conditions


def gentrends(dataframe, field="close", window=1 / 3.0, charts=False):
    """
    Returns a Pandas dataframe with support and resistance lines.

    :param dataframe: incoming data matrix
    :param field: for which column would you like to generate the trendline
    :param window: How long the trendlines should be. If window < 1, then it
                   will be taken as a percentage of the size of the data
    :param charts: Boolean value saying whether to print chart to screen
    """

    import numpy as np
    import pandas as pd

    # Use high values for resistance (Max Line)
    x_high = np.array(dataframe["high"])
    # Use low values for support (Min Line)
    x_low = np.array(dataframe["low"])
    # Use the specified field for Data
    x_data = np.array(dataframe[field])

    if window < 1:
        window = int(window * len(x_data))

    # Find max and min points using high and low data respectively
    max1 = np.where(x_high == max(x_high))[0][0]  # find the index of the abs max in high
    min1 = np.where(x_low == min(x_low))[0][0]  # find the index of the abs min in low

    # First the max
    if max1 + window >= len(x_high):
        max2 = max(x_high[0 : (max1 - window)])
    else:
        max2 = max(x_high[(max1 + window) :])

    # Now the min
    if min1 - window <= 0:
        min2 = min(x_low[(min1 + window) :])
    else:
        min2 = min(x_low[0 : (min1 - window)])

    # Now find the indices of the secondary extrema
    max2 = np.where(x_high == max2)[0][0]  # find the index of the 2nd max
    min2 = np.where(x_low == min2)[0][0]  # find the index of the 2nd min

    # Create & extend the lines
    maxslope = (x_high[max1] - x_high[max2]) / (max1 - max2)  # slope between max points
    minslope = (x_low[min1] - x_low[min2]) / (min1 - min2)  # slope between min points
    a_max = x_high[max1] - (maxslope * max1)  # y-intercept for max trendline
    a_min = x_low[min1] - (minslope * min1)  # y-intercept for min trendline
    b_max = x_high[max1] + (maxslope * (len(x_high) - max1))  # extend to last data pt
    b_min = x_low[min1] + (minslope * (len(x_low) - min1))  # extend to last data point
    maxline = np.linspace(a_max, b_max, len(x_data))  # Y values between max's
    minline = np.linspace(a_min, b_min, len(x_data))  # Y values between min's

    # OUTPUT
    trends = np.transpose(np.array((x_data, maxline, minline)))
    trends = pd.DataFrame(
        trends, index=np.arange(0, len(x_data)), columns=["Data", "Max Line", "Min Line"]
    )
    
    # Add slope information to the DataFrame
    trends['Max Slope'] = maxslope
    trends['Min Slope'] = minslope

    if charts:
        from matplotlib.pyplot import close, grid, plot, savefig

        plot(trends)
        grid()

        if isinstance(charts, str):
            savefig(f"{charts}.png")
        else:
            savefig(f"{x_data[0]}_{x_data[len(x_data) - 1]}.png")
        close()

    return trends


def segtrends(dataframe, field="close", segments=2, charts=False):
    """
    Turn minitrends to iterative process more easily adaptable to
    implementation in simple trading systems; allows backtesting functionality.

    :param dataframe: incoming data matrix
    :param field: for which column would you like to generate the trendline
    :param segments: Number of  Trend line segments to generate
    :param charts: Boolean value saying whether to print chart to screen
    """

    x = dataframe[field]
    import numpy as np

    y = np.array(x)
    high_values = np.array(dataframe["high"])
    low_values = np.array(dataframe["low"])

    # Implement trendlines
    segments = int(segments)
    maxima = np.ones(segments)
    minima = np.ones(segments)
    segsize = int(len(y) / segments)
    for i in range(1, segments + 1):
        ind2 = i * segsize
        ind1 = ind2 - segsize
        maxima[i - 1] = max(high_values[ind1:ind2])
        minima[i - 1] = min(low_values[ind1:ind2])

    # Find the indexes of these maxima in the data
    x_maxima = np.ones(segments)
    x_minima = np.ones(segments)
    for i in range(0, segments):
        x_maxima[i] = np.where(high_values == maxima[i])[0][0]
        x_minima[i] = np.where(low_values == minima[i])[0][0]

    if charts:
        import matplotlib.pyplot as plt

        plt.plot(y)
        plt.grid(True)

    # Store all trendlines
    all_maxlines = {}
    all_minlines = {}

    for i in range(0, segments - 1):
        maxslope = (maxima[i + 1] - maxima[i]) / (x_maxima[i + 1] - x_maxima[i])
        a_max = maxima[i] - (maxslope * x_maxima[i])
        b_max = maxima[i] + (maxslope * (len(y) - x_maxima[i]))
        maxline = np.linspace(a_max, b_max, len(y))
        all_maxlines[f'Max_Line_{i}'] = maxline

        minslope = (minima[i + 1] - minima[i]) / (x_minima[i + 1] - x_minima[i])
        a_min = minima[i] - (minslope * x_minima[i])
        b_min = minima[i] + (minslope * (len(y) - x_minima[i]))
        minline = np.linspace(a_min, b_min, len(y))
        all_minlines[f'Min_Line_{i}'] = minline

        if charts:
            plt.plot(maxline, "g")
            plt.plot(minline, "r")

    if charts:
        plt.show()

    import pandas as pd

    # Create a DataFrame with the original data
    trends = pd.DataFrame(
        y, index=np.arange(0, len(x)), columns=["Data"]
    )
    
    # Add all maxlines and minlines to the DataFrame
    for i in range(segments - 1):
        if f'Max_Line_{i}' in all_maxlines:
            trends[f'Max_Line_{i}'] = all_maxlines[f'Max_Line_{i}']
        if f'Min_Line_{i}' in all_minlines:
            trends[f'Min_Line_{i}'] = all_minlines[f'Min_Line_{i}']
    
    # Add standard Max Line and Min Line for backward compatibility
    if len(all_maxlines) > 0:
        trends['Max Line'] = all_maxlines[f'Max_Line_{segments-2}']  # Last segment
    if len(all_minlines) > 0:
        trends['Min Line'] = all_minlines[f'Min_Line_{segments-2}']  # Last segment
    
    return trends


def rank_trendlines(trends, trendline_objects):
    """
    Ranks trendlines based on bounce count from Trendline objects, with R-squared as tiebreaker.
    
    Ranking System:
    1. Primary: Number of bounces (higher is better)
    2. Tiebreaker: R-squared value (higher is better)
    
    Args:
        trends: DataFrame containing price data and trendlines
        trendline_objects: List of Trendline objects with pre-calculated bounce counts (REQUIRED)
        
    Returns:
        Dictionary with ranked maxlines and minlines based on bounce count and R-squared
    """
    import pandas as pd
    import numpy as np
    
    if len(trends) == 0:
        # print("ERROR: Empty trends DataFrame")
        return {"ranked_maxlines": {}, "ranked_minlines": {}}
    
    # print(f"=== BOUNCE COUNT + R-SQUARED TRENDLINE RANKING ===")
    
    # Trendline objects are now required since bounces are always pre-calculated
    if not trendline_objects:
        # print("ERROR: trendline_objects is required - bounces should be pre-calculated")
        return {"ranked_maxlines": {}, "ranked_minlines": {}}
    
    # print(f"Using pre-calculated bounce counts and R-squared from {len(trendline_objects)} Trendline objects")
    
    # Define trendline categories with enhanced scoring
    trendline_categories = {
        'resistance': {
            'columns': [col for col in trends.columns if col.startswith("Max_Line_") or col == "Max Line"],
            'trendlines': []  # Store (column, bounce_count, r_squared, trendline_obj)
        },
        'support': {
            'columns': [col for col in trends.columns if col.startswith("Min_Line_") or col == "Min Line"],
            'trendlines': []  # Store (column, bounce_count, r_squared, trendline_obj)
        }
    }
    
    # Map trendline objects to their corresponding columns and extract bounce counts + R-squared
    for trendline_obj in trendline_objects:
        if trendline_obj.trendline_type == 'resistance':
            # Find the corresponding resistance column (usually "Max Line")
            for col in trendline_categories['resistance']['columns']:
                if col in trends.columns and not trends[col].isna().all():
                    trendline_categories['resistance']['trendlines'].append((
                        col, 
                        trendline_obj.bounce_count, 
                        trendline_obj.r_squared,
                        trendline_obj
                    ))
                    # print(f"Resistance line '{col}': {trendline_obj.bounce_count} bounces, R²={trendline_obj.r_squared:.3f}")
                    break
        elif trendline_obj.trendline_type == 'support':
            # Find the corresponding support column (usually "Min Line")
            for col in trendline_categories['support']['columns']:
                if col in trends.columns and not trends[col].isna().all():
                    trendline_categories['support']['trendlines'].append((
                        col, 
                        trendline_obj.bounce_count, 
                        trendline_obj.r_squared,
                        trendline_obj
                    ))
                    # print(f"Support line '{col}': {trendline_obj.bounce_count} bounces, R²={trendline_obj.r_squared:.3f}")
                    break
    
    # Sort trendlines by bounce count (descending), then by R-squared (descending)
    def sort_key(trendline_tuple):
        col, bounce_count, r_squared, trendline_obj = trendline_tuple
        return (-bounce_count, -r_squared)  # Negative for descending order
    
    # Sort and create ranked results
    resistance_sorted = sorted(trendline_categories['resistance']['trendlines'], key=sort_key)
    support_sorted = sorted(trendline_categories['support']['trendlines'], key=sort_key)
    
    # Create the ranked results dictionary with bounce counts as scores (for backward compatibility)
    ranked_results = {
        "ranked_maxlines": {col: bounce_count for col, bounce_count, r_squared, trendline_obj in resistance_sorted},
        "ranked_minlines": {col: bounce_count for col, bounce_count, r_squared, trendline_obj in support_sorted}
    }
    
    # Print final rankings with enhanced information
    # print(f"\n=== FINAL RANKINGS (Bounce Count + R-squared Tiebreaker) ===")
    # print("RESISTANCE LINES:")
    # for i, (col, bounce_count, r_squared, trendline_obj) in enumerate(resistance_sorted, 1):
    #     if bounce_count > 0:
    #         print(f"  {i}. {col}: {bounce_count} bounces, R²={r_squared:.3f}")
    
    # print("SUPPORT LINES:")
    # for i, (col, bounce_count, r_squared, trendline_obj) in enumerate(support_sorted, 1):
    #     if bounce_count > 0:
    #         print(f"  {i}. {col}: {bounce_count} bounces, R²={r_squared:.3f}")
    
    return ranked_results


def generate_trendlines_for_period(recent_data: pd.DataFrame, random_period: int, 
                                 mc_recalc_interval_minutes: int, 
                                 trendline_proximity_threshold: float) -> Dict[str, Any]:
    """
    Generate trendlines and create Trendline objects for a specific lookback period.
    
    Args:
        recent_data: DataFrame with recent OHLCV data (must already have all_highs and all_lows columns)
        random_period: Lookback period to test
        mc_recalc_interval_minutes: Monte Carlo recalculation interval in minutes
        trendline_proximity_threshold: Threshold for trendline proximity scoring
            
    Returns:
        Dict containing trendlines and Trendline objects (without scores)
    """
    # Generate trends for this period
    trends = gentrends(recent_data, field='close', window=1/3.0)
    
    # Extract slope from the trends dataframe - gentrends now provides these columns
    resistance_slope = trends['Max Slope'].iloc[-1] if 'Max Slope' in trends.columns else 0.0
    support_slope = trends['Min Slope'].iloc[-1] if 'Min Slope' in trends.columns else 0.0
    
    # Create Trendline objects immediately after gentrends
    trendline_objects = []
    
    # Set start_time as the last candle in the lookback period (when the trendline becomes active)
    start_time = recent_data['date'].iloc[-1]  # Last candle in the lookback period
    # Set end_time as start_time plus the Monte Carlo recalculation interval
    # Ensure minimum duration of at least 1 minute to avoid start_time == end_time errors
    min_duration_minutes = mc_recalc_interval_minutes
    end_time = start_time + pd.Timedelta(minutes=min_duration_minutes)
    
    # Create resistance trendline object (Max Line)
    if 'Max Line' in trends.columns and not trends['Max Line'].isna().all():
        # Get the resistance price at the start_time (last candle)
        resistance_start_price = trends['Max Line'].iloc[-1]
        
        # Calculate R-squared for resistance trendline
        resistance_r_squared = calculate_r_squared(
            price_series=recent_data['high'],
            trendline_series=trends['Max Line']
        )
        
        resistance_trendline = Trendline(
            trendline_type='resistance',
            start_time=start_time,
            end_time=end_time,
            slope=resistance_slope,
            start_price=resistance_start_price,
            r_squared=resistance_r_squared,
            bounce_count=0
        )
        trendline_objects.append(resistance_trendline)
    
    # Create support trendline object (Min Line)
    if 'Min Line' in trends.columns and not trends['Min Line'].isna().all():
        # Get the support price at the start_time (last candle)
        support_start_price = trends['Min Line'].iloc[-1]
        
        # Calculate R-squared for support trendline
        support_r_squared = calculate_r_squared(
            price_series=recent_data['low'],
            trendline_series=trends['Min Line']
        )
        
        support_trendline = Trendline(
            trendline_type='support',
            start_time=start_time,
            end_time=end_time,
            slope=support_slope,
            start_price=support_start_price,
            r_squared=support_r_squared,
            bounce_count=0
        )
        trendline_objects.append(support_trendline)
    
    return {
        'trends': trends,
        'resistance_slope': resistance_slope,
        'support_slope': support_slope,
        'trendline_objects': trendline_objects,  # List of Trendline objects
        'start_time': start_time,  # Store the calculated start time
        'end_time': end_time,      # Store the calculated end time
        'lookback_period': random_period,  # Store the period used
        'recent_data': recent_data  # Include the data for later use
    }


def calculate_r_squared(price_series: Union[pd.Series, np.ndarray], 
                       trendline_series: Union[pd.Series, np.ndarray]) -> float:
    """
    Calculate the R-squared value for a trendline fit to price data using scipy.stats.linregress.
    
    Args:
        price_series: Actual price values (e.g., high or low prices)
        trendline_series: Calculated trendline values
        
    Returns:
        float: R-squared value (0.0 to 1.0). Returns 0.0 if calculation fails.
    """
    try:
        # Convert to numpy arrays for consistent handling
        y_actual = np.array(price_series)
        y_predicted = np.array(trendline_series)
        
        # Remove any NaN values
        valid_mask = ~(np.isnan(y_actual) | np.isnan(y_predicted))
        if not np.any(valid_mask):
            return 0.0
            
        y_actual = y_actual[valid_mask]
        y_predicted = y_predicted[valid_mask]
        
        # Need at least 2 data points for linear regression
        if len(y_actual) < 2:
            return 0.0
            
        # Create x values as indices (time points)
        x = np.arange(len(y_actual))
        
        # Calculate linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y_actual)
        
        # Return R-squared value (square of correlation coefficient)
        return max(0.0, min(1.0, r_value * r_value))
        
    except Exception as e:
        return 0.0


def create_trendline_object(recent_data: pd.DataFrame, trends: pd.DataFrame, 
                           trendline_type: str, slope: float, start_time: pd.Timestamp, 
                           end_time: pd.Timestamp, trendline_proximity_threshold: float) -> Optional[Trendline]:
    """
    Create a single Trendline object (either support or resistance).
    
    Args:
        recent_data: Recent price data
        trends: Generated trendline data
        trendline_type: Either 'support' or 'resistance'
        slope: Slope of the trendline
        start_time: Start time for trendline
        end_time: End time for trendline
        trendline_proximity_threshold: Threshold for trendline proximity scoring
        
    Returns:
        Trendline object or None if creation fails
    """
    # Define column and price series based on trendline type
    if trendline_type == 'resistance':
        column = 'Max Line'
        price_series = recent_data['high']
        direction = 'short'
    elif trendline_type == 'support':
        column = 'Min Line'
        price_series = recent_data['low']
        direction = 'long'
    else:
        return None
    
    # Check if the trendline column exists and has valid data
    if column not in trends.columns or trends[column].isna().all():
        return None
    
    # Get the price at the start_time (last candle)
    start_price = trends[column].iloc[-1]
    
    # Calculate R-squared for trendline
    r_squared = calculate_r_squared(
        price_series=price_series,
        trendline_series=trends[column]
    )
    
    # Generate bounce conditions
    bounce_conditions = generate_bounce_conditions(
        close_data=recent_data['close'],
        level_data=trends[column].reindex(recent_data.index, method='ffill'),
        direction=direction,
        tolerance=trendline_proximity_threshold,
        pivot_highs=recent_data['all_highs'],
        pivot_lows=recent_data['all_lows']
    )
    
    # Capture bounce timestamps
    bounce_indices = bounce_conditions[bounce_conditions].index
    bounce_timestamps = recent_data.loc[bounce_indices, 'date'].tolist()
    
    # Create trendline object
    trendline = Trendline(
        trendline_type=trendline_type,
        start_time=start_time,
        end_time=end_time,
        slope=slope,
        start_price=start_price,
        r_squared=r_squared,
        bounce_count=bounce_conditions.sum(),
        bounce_timestamps=bounce_timestamps
    )
    
    return trendline


def output_trendlines_info(trendline_objects: List[Trendline]) -> None:
    """
    Output information about trendlines to the console.
    
    Args:
        trendline_objects: List of Trendline objects
    """
    try:
        print("\n=== STORED TRENDLINES SUMMARY ===")
        
        # Count trendlines from stored_trendlines list
        total_trendlines = len(trendline_objects)
        resistance_trendlines = sum(1 for tl in trendline_objects if tl.trendline_type == 'resistance')
        support_trendlines = sum(1 for tl in trendline_objects if tl.trendline_type == 'support')
                
        # Display trendline count summary
        print(f"\n--- TRENDLINE COUNT SUMMARY ---")
        print(f"  Total Saved Trendlines: {total_trendlines}")
        print(f"  Resistance Trendlines: {resistance_trendlines}")
        print(f"  Support Trendlines: {support_trendlines}")
        
        # Display all stored trendlines
        if total_trendlines > 0:
            print(f"\n--- ALL STORED TRENDLINES ---")
            for i, trendline in enumerate(trendline_objects, 1):
                print(f"  {i}. {trendline.trendline_type.upper()} TRENDLINE")
                print(f"     Start Time: {trendline.start_time}")
                print(f"     End Time: {trendline.end_time}")
                print(f"     Duration: {trendline.duration_hours:.2f} hours")
                print(f"     Age: {trendline.age_hours:.2f} hours")
                print(f"     Slope: {trendline.slope:.8f}")
                print(f"     R-squared: {trendline.r_squared:.4f}")
                print(f"     Start Price: {trendline.start_price:.6f}")
                print(f"     Bounce Count: {trendline.bounce_count}")
                
                # Display bounce timestamps if available
                if hasattr(trendline, 'bounce_timestamps') and trendline.bounce_timestamps:
                    print(f"     Bounce Timestamps ({len(trendline.bounce_timestamps)}):")
                    for j, timestamp in enumerate(trendline.bounce_timestamps, 1):
                        print(f"       {j}. {timestamp}")
                else:
                    print(f"     Bounce Timestamps: None")
                
                print("")
        else:
            print(f"  No trendlines stored during this execution")
        
        
        print(f"=== END TRENDLINES SUMMARY ===\n")
        
    except Exception as e:
        print(f"Error outputting stored trendlines: {e}")


def generate_lookback_periods(total_candles: int, mc_iterations: int) -> List[int]:
    """
    Generate lookback periods for Monte Carlo optimization.
    
    Args:
        total_candles: Total number of candles available
        mc_iterations: Number of Monte Carlo iterations to run
        
    Returns:
        List of lookback periods to test
    """
    # Generate core periods by dividing total_candles into segments
    core_periods = []
    
    # Create divisions of total_candles
    for divisor in [1.1, 1.2, 1.4, 1.6, 1.8, 2, 3, 4, 5, 6, 8]:
        period = int(total_candles // divisor)
        core_periods.append(period)
    
    # Remove duplicates and sort
    core_periods = sorted(list(set(core_periods)))
    
    # Generate additional random periods to reach mc_iterations
    remaining_iterations = mc_iterations - len(core_periods)
    random_periods = []
    
    if remaining_iterations > 0:
        for _ in range(remaining_iterations):
            random_period = random.randint(0, total_candles)
            random_periods.append(random_period)
    
    # Combine core periods with random periods
    lookback_periods = core_periods + random_periods
    
    # Shuffle to randomize the order
    random.shuffle(lookback_periods)
    
    return lookback_periods


def evaluate_lookback_period_for_trendlines(dataframe: pd.DataFrame, random_period: int,
                              recalc_interval_minutes: int, trendline_proximity_threshold: float) -> Tuple[List[Trendline], float, float]:
    """
    Evaluate a single lookback period and return trendline objects and scores.
    
    Args:
        dataframe: Full price dataframe
        random_period: Lookback period to test
        recalc_interval_minutes: Recalculation interval in minutes
        trendline_proximity_threshold: Threshold for trendline proximity scoring
        
    Returns:
        Tuple of (trendline_objects, resistance_score, support_score)
    """
    # Test this period
    recent_data = dataframe.tail(random_period).copy()
    
    # Generate trends for this period directly
    trends = gentrends(recent_data, field='close', window=1/3.0)
    
    # Extract slopes from the trends dataframe
    resistance_slope = trends['Max Slope'].iloc[-1] if 'Max Slope' in trends.columns else 0.0
    support_slope = trends['Min Slope'].iloc[-1] if 'Min Slope' in trends.columns else 0.0
    
    # Set start_time as the last candle in the lookback period (when the trendline becomes active)
    start_time = recent_data['date'].iloc[-1]  # Last candle in the lookback period
    # Set end_time as start_time plus the Monte Carlo recalculation interval
    min_duration_minutes = recalc_interval_minutes
    end_time = start_time + pd.Timedelta(minutes=min_duration_minutes)
    
    # Create trendline objects individually
    trendline_objects = []
    
    # Create resistance trendline
    resistance_trendline = create_trendline_object(
        recent_data, trends, 'resistance', resistance_slope, start_time, end_time, trendline_proximity_threshold
    )
    if resistance_trendline:
        trendline_objects.append(resistance_trendline)
    
    # Create support trendline
    support_trendline = create_trendline_object(
        recent_data, trends, 'support', support_slope, start_time, end_time, trendline_proximity_threshold
    )
    if support_trendline:
        trendline_objects.append(support_trendline)
    
    # Calculate scores using rank_trendlines
    main_lines_score = rank_trendlines(trends, trendline_objects)
    
    current_resistance_score = main_lines_score["ranked_maxlines"].get("Max Line", 0)
    current_support_score = main_lines_score["ranked_minlines"].get("Min Line", 0)
    
    return trendline_objects, current_resistance_score, current_support_score


def monte_carlo_period_optimization(dataframe: pd.DataFrame, pair: str = "UNKNOWN",
                                   recalc_interval_minutes: int = 120, mc_iterations: int = 1000,
                                   trendline_proximity_threshold: float = 0.00005):
    """
    Perform Monte Carlo period optimization for trendline analysis.
    
    Args:
        dataframe: Full price dataframe
        pair: Trading pair name for logging
        recalc_interval_minutes: Recalculation interval in minutes
        mc_iterations: Number of Monte Carlo iterations to run
        trendline_proximity_threshold: Threshold for trendline proximity scoring
        
    Returns:
        Dictionary containing optimization results
    """
    # Set MAX_LOOKBACK_PERIOD dynamically based on available data
    total_candles = len(dataframe)
            
    # Seed random number generator for reproducible results with some variability
    random.seed(int(time.time() * 1000) % 10000)  # Use current time for seed
    
    print(f"=== Monte Carlo Period Optimization with Core Periods for {pair} ===")
    
    # Generate lookback periods
    lookback_periods = generate_lookback_periods(total_candles, mc_iterations)

    # Initialize best scores and trendlines
    best_resistance_score = 0.0
    best_support_score = 0.0
    best_resistance_period = 0
    best_support_period = 0
    best_resistance_trendline = None
    best_support_trendline = None
    
    for iteration, random_period in enumerate(lookback_periods):
        try:
            # Evaluate this period
            trendline_objects, current_resistance_score, current_support_score = evaluate_lookback_period_for_trendlines(
                dataframe, random_period, recalc_interval_minutes, trendline_proximity_threshold
            )
            
            # Update best trendlines if current scores are better
            if current_resistance_score > best_resistance_score:
                best_resistance_score = current_resistance_score
                best_resistance_period = random_period
                best_resistance_trendline = next((obj for obj in trendline_objects if obj.trendline_type == 'resistance'), None)
                print(f"New best resistance found for {pair} at iteration {iteration + 1}: period {random_period}, score {current_resistance_score:.4f}, bounces {best_resistance_trendline.bounce_count if best_resistance_trendline else 'N/A'}")
            
            if current_support_score > best_support_score:
                best_support_score = current_support_score
                best_support_period = random_period
                best_support_trendline = next((obj for obj in trendline_objects if obj.trendline_type == 'support'), None)
                print(f"New best support found for {pair} at iteration {iteration + 1}: period {random_period}, score {current_support_score:.4f}, bounces {best_support_trendline.bounce_count if best_support_trendline else 'N/A'}")
            
            # Progress reporting with more details
            if (iteration + 1) % 100 == 0:
                print(f"Monte Carlo progress for {pair}: {iteration + 1}/{len(lookback_periods)} completed")
                print(f"Current best - Resistance: {best_resistance_score:.4f} (period {best_resistance_period}), Support: {best_support_score:.4f} (period {best_support_period})")
            
        except Exception as e:
            print(f"Error in Monte Carlo iteration {iteration} for {pair}: {e}")
            continue
            
    return {
        'resistance_score': best_resistance_score,
        'support_score': best_support_score,
        'best_resistance_trendline': best_resistance_trendline,
        'best_support_trendline': best_support_trendline
    }


def project_trendlines_forward(dataframe: pd.DataFrame, candle_index: int, 
                              resistance_trendline: Optional[Trendline], support_trendline: Optional[Trendline]) -> Tuple[float, float]:
    """
    Project trendlines forward using slopes and apply results to dataframe.
    
    Args:
        dataframe: The dataframe to update
        candle_index: Current candle index
        resistance_trendline: Resistance trendline object
        support_trendline: Support trendline object
        
    Returns:
        Tuple of (new_resistance_value, new_support_value)
    """
    pandas_index = dataframe.index[candle_index]
    
    # Calculate resistance projection
    if resistance_trendline:
        resistance_value = resistance_trendline.get_price_at_time(dataframe['date'].iloc[candle_index])
        dataframe.loc[pandas_index, 'MC_Optimal_Resistance'] = resistance_value
        dataframe.loc[pandas_index, 'MC_Resistance_Score'] = resistance_trendline.bounce_count
    else:
        dataframe.loc[pandas_index, 'MC_Optimal_Resistance'] = np.nan
        dataframe.loc[pandas_index, 'MC_Resistance_Score'] = 0.0
    
    # Calculate support projection
    if support_trendline:
        support_value = support_trendline.get_price_at_time(dataframe['date'].iloc[candle_index])
        dataframe.loc[pandas_index, 'MC_Optimal_Support'] = support_value
        dataframe.loc[pandas_index, 'MC_Support_Score'] = support_trendline.bounce_count
    else:
        dataframe.loc[pandas_index, 'MC_Optimal_Support'] = np.nan
        dataframe.loc[pandas_index, 'MC_Support_Score'] = 0.0
    
    return (resistance_value if resistance_trendline else np.nan, 
            support_value if support_trendline else np.nan)
