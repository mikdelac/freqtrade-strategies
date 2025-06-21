"""
defines trendline based indicator logic
based on
https://github.com/dysonance/Trendy
"""

import numpy as np
import pandas as pd

def generate_bounce_conditions(close_data, high_data, low_data, level_data, direction: str, tolerance: float = 0.00005):
    """
    Generate bounce conditions for support or resistance levels.
    This is the centralized bounce detection logic used throughout the system.
    
    Args:
        close_data: Close price series
        high_data: High price series  
        low_data: Low price series
        level_data: Trendline level series
        direction: Either 'long' for support bounce or 'short' for resistance bounce
        tolerance: Proximity tolerance (default: 0.00005 or 0.005%)
        
    Returns:
        pandas.Series: Boolean series indicating bounce conditions
    """
    if direction == 'long':
        # Long entry: Bounce off support using swing low extrema
        bounce_conditions = (
            # Current close is above support
            (close_data > level_data) &
            # Previous low was at or near support but ABOVE it (within tolerance, rejected)
            (low_data.shift(1) >= level_data.shift(1)) &  # Low is above or at support
            (abs(low_data.shift(1) - level_data.shift(1)) <= 
             level_data.shift(1) * tolerance) &  # But close enough to be considered a test
            # Previous low was lower than the low 2 candles ago (swing low pattern)
            (low_data.shift(1) <= low_data.shift(2)) &
            # Previous low was lower than current low (confirming bounce)
            (low_data.shift(1) < low_data) &
            # Current close is higher than previous close (upward movement)
            (close_data > close_data.shift(1)) &
            # Level data is valid
            (~level_data.isna()) &
            (~level_data.shift(1).isna())
        )
    elif direction == 'short':
        # Short entry: Bounce off resistance using swing high extrema
        bounce_conditions = (
            # Current close is below resistance
            (close_data < level_data) &
            # Previous high was at or near resistance but BELOW it (within tolerance, rejected)
            (high_data.shift(1) <= level_data.shift(1)) &  # High is below or at resistance
            (abs(high_data.shift(1) - level_data.shift(1)) <= 
             level_data.shift(1) * tolerance) &  # But close enough to be considered a test
            # Previous high was higher than the high 2 candles ago (swing high pattern)
            (high_data.shift(1) >= high_data.shift(2)) &
            # Previous high was higher than current high (confirming bounce)
            (high_data.shift(1) > high_data) &
            # Current close is lower than previous close (downward movement)
            (close_data < close_data.shift(1)) &
            # Level data is valid
            (~level_data.isna()) &
            (~level_data.shift(1).isna())
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


def rank_trendlines(trends, price_field="Data", threshold=0.01, max_prefix="Max_Line_", min_prefix="Min_Line_", all_highs=None, all_lows=None, pivot_bonus=5.0):
    """
    Ranks trendlines based on the number of bounces off the trendline using the same
    bounce detection logic as generate_bounce_conditions from RiskMetrics.py.
    
    A bounce is defined as:
    1. Price approaches the trendline (within threshold)
    2. Price forms a swing point (pivot) near the trendline
    3. Price then moves away from the trendline in the expected direction
    
    For resistance lines (maxlines): 
    - Price approaches from below, forms a swing high near the line, then moves back down
    
    For support lines (minlines):
    - Price approaches from above, forms a swing low near the line, then moves back up
    
    This scoring method is consistent with the bounce detection used in signal generation.
    
    :param trends: DataFrame containing price data and trendlines
    :param price_field: Column name for price data (default: "Data")
    :param threshold: Proximity threshold as a percentage (default: 0.01 or 1%)
    :param max_prefix: Prefix for maxline columns (default: "Max_Line_")
    :param min_prefix: Prefix for minline columns (default: "Min_Line_")
    :param all_highs: Series or DataFrame column with high pivot points (default: None)
    :param all_lows: Series or DataFrame column with low pivot points (default: None)
    :param pivot_bonus: Multiplier for bonus points when bounces occur at pivot points (default: 5.0)
    :return: Dictionary with ranked maxlines and minlines
    """
    import pandas as pd
    import numpy as np
    
    # --- Initialization ---
    trendlines = {
        'resistance': {
            'columns': [col for col in trends.columns if col.startswith(max_prefix) or col == "Max Line"],
            'scores': {},
            'pivot_points': all_highs,  # Resistance lines should use high pivots
        },
        'support': {
            'columns': [col for col in trends.columns if col.startswith(min_prefix) or col == "Min Line"],
            'scores': {},
            'pivot_points': all_lows,  # Support lines should use low pivots
        }
    }
    
    # Total number of datapoints
    n_points = len(trends)
    
    # Calculate recency weights - newer bounces are more valuable
    recency_weights = np.linspace(1.0, 3.0, n_points)
    
    # --- Prepare pivot point arrays ---
    def prepare_pivot_array(pivot_points, n_points):
        """Helper function to prepare pivot point arrays with consistent length"""
        if pivot_points is None or pivot_points.isna().all():
            return None
            
        array = pivot_points.reset_index(drop=True).values
        
        if len(array) > n_points:
            return array[:n_points]
        elif len(array) < n_points:
            padding = np.full(n_points - len(array), np.nan)
            return np.concatenate([array, padding])
        return array
    
    # Prepare pivot arrays once
    prepared_pivots = {
        'highs': prepare_pivot_array(all_highs, n_points),
        'lows': prepare_pivot_array(all_lows, n_points)
    }
    
    # --- Calculate bounce scores for each trendline ---
    # Generate bounce conditions using the existing helper function
    for trendline_type, config in trendlines.items():
        columns = config['columns']
        scores = config['scores']
        
        # Get the appropriate pivot points for this trendline type
        pivot_points = prepared_pivots['highs'] if trendline_type == 'resistance' else prepared_pivots['lows']
        
        for col in columns:
            # Skip columns with all NaN values
            if trends[col].isna().all():
                continue
                
            # Get price and trendline data
            price_data = trends[price_field]
            trendline_data = trends[col]
            
            # We need high and low data for bounce detection
            # If not available in trends DataFrame, use price_data as approximation
            if 'high' in trends.columns and 'low' in trends.columns:
                high_data = trends['high']
                low_data = trends['low']
            else:
                # Fallback: use price_data for both high and low
                high_data = price_data
                low_data = price_data
            
            # Generate bounce conditions using the centralized function
            if trendline_type == 'resistance':
                bounce_conditions = generate_bounce_conditions(
                    price_data, high_data, low_data, trendline_data, 'short', threshold
                )
            else:  # support
                bounce_conditions = generate_bounce_conditions(
                    price_data, high_data, low_data, trendline_data, 'long', threshold
                )
            
            # Calculate score based on bounces
            total_score = 0
            bounce_indices = bounce_conditions[bounce_conditions].index
            
            # === MAJOR SCORING COMPONENT: Count Respecting Pivots ===
            # This is the most important scoring factor - how many pivots respect this trendline
            respecting_pivots_score = 0
            respecting_pivots_count = 0
            
            if pivot_points is not None:
                valid_pivot_indices = np.where(~np.isnan(pivot_points))[0]
                
                for pivot_idx in valid_pivot_indices:
                    if pivot_idx >= len(trendline_data):
                        continue
                        
                    pivot_value = pivot_points[pivot_idx]
                    trendline_value = trendline_data.iloc[pivot_idx]
                    
                    if pd.isna(trendline_value) or pd.isna(pivot_value) or pivot_value <= 0:
                        continue
                    
                    # Check if this pivot respects the trendline
                    if trendline_type == 'resistance':
                        # For resistance: pivot high should be BELOW the resistance line
                        if pivot_value <= trendline_value:
                            # Calculate how close the pivot is to the resistance
                            distance_pct = abs(trendline_value - pivot_value) / trendline_value
                            
                            # Award points based on proximity (closer = more points)
                            proximity_factor = max(0, 1.0 - (distance_pct / (threshold * 3)))  # Extended range for pivots
                            
                            # Base points for respecting pivot
                            base_pivot_points = 50.0  # Much higher than bounce points
                            
                            # Bonus for very close pivots (within threshold)
                            if distance_pct <= threshold:
                                proximity_bonus = 25.0 * (1.0 - distance_pct / threshold)
                            else:
                                proximity_bonus = 0
                            
                            # Apply recency weighting
                            recency_factor = recency_weights[min(pivot_idx, len(recency_weights)-1)]
                            
                            pivot_score = (base_pivot_points + proximity_bonus) * proximity_factor * recency_factor
                            respecting_pivots_score += pivot_score
                            respecting_pivots_count += 1
                            
                    else:  # support
                        # For support: pivot low should be ABOVE the support line
                        if pivot_value >= trendline_value:
                            # Calculate how close the pivot is to the support
                            distance_pct = abs(pivot_value - trendline_value) / trendline_value
                            
                            # Award points based on proximity (closer = more points)
                            proximity_factor = max(0, 1.0 - (distance_pct / (threshold * 3)))  # Extended range for pivots
                            
                            # Base points for respecting pivot
                            base_pivot_points = 50.0  # Much higher than bounce points
                            
                            # Bonus for very close pivots (within threshold)
                            if distance_pct <= threshold:
                                proximity_bonus = 25.0 * (1.0 - distance_pct / threshold)
                            else:
                                proximity_bonus = 0
                            
                            # Apply recency weighting
                            recency_factor = recency_weights[min(pivot_idx, len(recency_weights)-1)]
                            
                            pivot_score = (base_pivot_points + proximity_bonus) * proximity_factor * recency_factor
                            respecting_pivots_score += pivot_score
                            respecting_pivots_count += 1
            
            # === MULTIPLIER BONUS: More respecting pivots = exponential bonus ===
            if respecting_pivots_count > 0:
                # Exponential multiplier based on number of respecting pivots
                pivot_count_multiplier = 1.0 + (respecting_pivots_count * 0.5)  # +50% per additional pivot
                respecting_pivots_score *= pivot_count_multiplier
                
                print(f"  {col}: {respecting_pivots_count} respecting pivots, base score: {respecting_pivots_score/pivot_count_multiplier:.2f}, multiplied: {respecting_pivots_score:.2f}")
            
            # === SECONDARY SCORING: Actual Bounces ===
            bounce_score = 0
            for idx in bounce_indices:
                # Convert pandas index to integer position
                pos = bounce_conditions.index.get_loc(idx)
                
                # Base score for the bounce (much lower than pivot score)
                base_bounce_score = 10.0  # Base points for any confirmed bounce
                
                # Calculate bounce strength based on price movement
                if pos < len(price_data) - 1:
                    if trendline_type == 'resistance':
                        # For resistance: measure downward movement after bounce
                        current_price = price_data.iloc[pos]
                        next_price = price_data.iloc[pos + 1]
                        strength = max(0, (current_price - next_price) / current_price)
                    else:
                        # For support: measure upward movement after bounce
                        current_price = price_data.iloc[pos]
                        next_price = price_data.iloc[pos + 1]
                        strength = max(0, (next_price - current_price) / current_price)
                    
                    strength_bonus = strength * 100  # Convert percentage to points
                else:
                    strength_bonus = 0
                
                # Calculate proximity bonus
                trendline_value = trendline_data.iloc[pos]
                if trendline_type == 'resistance':
                    relevant_price = high_data.iloc[pos]
                else:
                    relevant_price = low_data.iloc[pos]
                
                if not pd.isna(trendline_value) and trendline_value > 0:
                    distance_pct = abs(relevant_price - trendline_value) / trendline_value
                    proximity_bonus = max(0, (threshold - distance_pct) / threshold * 5.0)
                else:
                    proximity_bonus = 0
                
                # Apply recency weighting
                recency_factor = recency_weights[min(pos, len(recency_weights)-1)]
                
                # Calculate individual bounce score
                individual_bounce_score = (base_bounce_score + strength_bonus + proximity_bonus) * recency_factor
                bounce_score += individual_bounce_score
            
            # === TOTAL SCORE CALCULATION ===
            # Pivot score is the dominant factor, bounce score is secondary
            total_score = respecting_pivots_score + bounce_score
            
            # Store the final score
            scores[col] = total_score
            
            # Enhanced debug output
            if respecting_pivots_count > 0 or len(bounce_indices) > 0:
                print(f"{col} ({trendline_type}):")
                print(f"  Respecting pivots: {respecting_pivots_count} (score: {respecting_pivots_score:.2f})")
                print(f"  Confirmed bounces: {len(bounce_indices)} (score: {bounce_score:.2f})")
                print(f"  TOTAL SCORE: {total_score:.2f}")
                print(f"  ---")
    
    # --- Sort and return results ---
    ranked_results = {
        "ranked_maxlines": {k: v for k, v in sorted(trendlines['resistance']['scores'].items(), 
                                                   key=lambda item: item[1], 
                                                   reverse=True)},
        "ranked_minlines": {k: v for k, v in sorted(trendlines['support']['scores'].items(), 
                                                   key=lambda item: item[1], 
                                                   reverse=True)}
    }
    
    return ranked_results