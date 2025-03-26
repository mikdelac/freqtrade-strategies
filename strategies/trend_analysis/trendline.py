"""
defines trendline based indicator logic
based on
https://github.com/dysonance/Trendy
"""


def gentrends(dataframe, field="close", window=1 / 3.0, charts=False):
    """
    Returns a Pandas dataframe with support and resistance lines.

    :param dataframe: incoming data matrix
    :param field: for which column would you like to generate the trendline
    :param window: How long the trendlines should be. If window < 1, then it
                   will be taken as a percentage of the size of the data
    :param charts: Boolean value saying whether to print chart to screen
    """

    x = dataframe[field]

    import numpy as np
    import pandas as pd

    x = np.array(x)

    if window < 1:
        window = int(window * len(x))

    max1 = np.where(x == max(x))[0][0]  # find the index of the abs max
    min1 = np.where(x == min(x))[0][0]  # find the index of the abs min

    # First the max
    if max1 + window >= len(x):
        max2 = max(x[0 : (max1 - window)])
    else:
        max2 = max(x[(max1 + window) :])

    # Now the min
    if min1 - window <= 0:
        min2 = min(x[(min1 + window) :])
    else:
        min2 = min(x[0 : (min1 - window)])

    # Now find the indices of the secondary extrema
    max2 = np.where(x == max2)[0][0]  # find the index of the 2nd max
    min2 = np.where(x == min2)[0][0]  # find the index of the 2nd min

    # Create & extend the lines
    maxslope = (x[max1] - x[max2]) / (max1 - max2)  # slope between max points
    minslope = (x[min1] - x[min2]) / (min1 - min2)  # slope between min points
    a_max = x[max1] - (maxslope * max1)  # y-intercept for max trendline
    a_min = x[min1] - (minslope * min1)  # y-intercept for min trendline
    b_max = x[max1] + (maxslope * (len(x) - max1))  # extend to last data pt
    b_min = x[min1] + (minslope * (len(x) - min1))  # extend to last data point
    maxline = np.linspace(a_max, b_max, len(x))  # Y values between max's
    minline = np.linspace(a_min, b_min, len(x))  # Y values between min's

    # OUTPUT
    trends = np.transpose(np.array((x, maxline, minline)))
    trends = pd.DataFrame(
        trends, index=np.arange(0, len(x)), columns=["Data", "Max Line", "Min Line"]
    )

    if charts:
        from matplotlib.pyplot import close, grid, plot, savefig

        plot(trends)
        grid()

        if isinstance(charts, str):
            savefig(f"{charts}.png")
        else:
            savefig(f"{x[0]}_{x[len(x) - 1]}.png")
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


def rank_trendlines(trends, price_field="Data", threshold=0.01, touch_weight=2.0, max_prefix="Max_Line_", min_prefix="Min_Line_", all_highs=None, all_lows=None, pivot_bonus=5.0):
    """
    Ranks trendlines based on how often price closes near them.
    
    This function evaluates the strength and reliability of trendlines by measuring how frequently
    price interacts with them. It assigns scores to each trendline based on:
    1. Near misses: When price comes within a threshold percentage of the trendline
    2. Touches: When price comes very close to the trendline (within half the threshold)
    
    Points that are nearer to the present time are weighted more heavily, using a linear
    scaling factor that increases as we approach the most recent datapoints.
    
    Additionally, this function now gives bonus points to:
    - Maxlines (resistance) that are close to all_lows datapoints
    - Minlines (support) that are close to all_highs datapoints
    
    This rewards trendlines that connect significant swing points and are respected by price action.
    
    Touches are weighted more heavily than near misses using the touch_weight parameter.
    The function returns dictionaries of ranked maxlines (resistance) and minlines (support)
    sorted by their scores in descending order.
    
    For maxlines (resistance):
    - A near miss occurs when price is below but within threshold% of the line
    - A touch occurs when price is below but within (threshold/2)% of the line
    
    For minlines (support):
    - A near miss occurs when price is above but within threshold% of the line
    - A touch occurs when price is above but within (threshold/2)% of the line
    
    :param trends: DataFrame containing price data and trendlines
    :param price_field: Column name for price data (default: "Data")
    :param threshold: Proximity threshold as a percentage (default: 0.01 or 1%)
    :param touch_weight: Weight multiplier for touches vs near misses (default: 2.0)
    :param max_prefix: Prefix for maxline columns (default: "Max_Line_")
    :param min_prefix: Prefix for minline columns (default: "Min_Line_")
    :param all_highs: Series or DataFrame column with high pivot points (default: None)
    :param all_lows: Series or DataFrame column with low pivot points (default: None)
    :param pivot_bonus: Multiplier for bonus points when trendlines are near pivots (default: 5.0)
    :return: Dictionary with ranked maxlines and minlines
    """
    import pandas as pd
    import numpy as np
    
    # Initialize dictionaries to store scores
    maxline_scores = {}
    minline_scores = {}
    
    # Get all maxline and minline columns
    maxline_cols = [col for col in trends.columns if col.startswith(max_prefix) or col == "Max Line"]
    minline_cols = [col for col in trends.columns if col.startswith(min_prefix) or col == "Min Line"]
    
    # Total number of datapoints
    n_points = len(trends)
    
    # Calculate recency weights - linear increase from oldest to newest point
    recency_weights = np.linspace(1.0, 3.0, n_points)  # Newest point is 3x more valuable than oldest
    
    # Prepare highs and lows for matching with trends
    # Convert from Series with potentially different indices to numpy arrays for positional matching
    all_highs_array = None
    all_lows_array = None
    
    if all_highs is not None and not all_highs.isna().all():
        # Extract valid values and reset to positional indexing
        all_highs_array = all_highs.reset_index(drop=True).values
        # If the arrays are different lengths, trim or pad 
        if len(all_highs_array) > n_points:
            # Trim to match trends length
            all_highs_array = all_highs_array[:n_points]
        elif len(all_highs_array) < n_points:
            # Pad with NaN to match trends length
            padding = np.full(n_points - len(all_highs_array), np.nan)
            all_highs_array = np.concatenate([all_highs_array, padding])
    
    if all_lows is not None and not all_lows.isna().all():
        # Extract valid values and reset to positional indexing
        all_lows_array = all_lows.reset_index(drop=True).values
        # If the arrays are different lengths, trim or pad
        if len(all_lows_array) > n_points:
            # Trim to match trends length
            all_lows_array = all_lows_array[:n_points]
        elif len(all_lows_array) < n_points:
            # Pad with NaN to match trends length
            padding = np.full(n_points - len(all_lows_array), np.nan)
            all_lows_array = np.concatenate([all_lows_array, padding])
    
    # Calculate scores for each maxline
    for col in maxline_cols:
        # Skip columns with all NaN values
        if trends[col].isna().all():
            continue
            
        # Calculate proximity score
        # For maxlines: price is near when price is close to but not above the line
        price_data = trends[price_field]
        trendline_data = trends[col]
        
        # Calculate distance as percentage of price
        distance = (trendline_data - price_data) / price_data
        
        # Identify near misses and touches with boolean masks
        near_misses = (distance >= 0) & (distance <= threshold)
        touches = (distance >= 0) & (distance <= threshold/2)
        
        # Apply recency weighting to each interaction
        weighted_near_misses = np.where(near_misses, recency_weights, 0)
        weighted_touches = np.where(touches, recency_weights * touch_weight, 0)
        
        # Calculate base score
        base_score = np.sum(weighted_near_misses) + np.sum(weighted_touches)
        
        # Add bonus for proximity to all_lows if provided
        low_pivot_bonus = 0
        if all_lows_array is not None:
            # Get non-nan indices in all_lows_array
            low_indices = np.where(~np.isnan(all_lows_array))[0]
            
            for idx in low_indices:
                # Skip if index is out of bounds for trendline data
                if idx >= len(trendline_data):
                    continue
                    
                # Calculate distance between the trendline and the low point
                low_value = all_lows_array[idx]
                trendline_value = trendline_data.iloc[idx]
                
                # Distance as percentage
                if not pd.isna(trendline_value) and not pd.isna(low_value) and low_value > 0:
                    dist_pct = (trendline_value - low_value) / low_value
                    # Award bonus points if the trendline is close to the low point
                    if 0 <= abs(dist_pct) <= threshold:
                        # More points for closer proximity
                        proximity_factor = 1.0 - (abs(dist_pct) / threshold)  # 1.0 for exact match, 0.0 for threshold
                        
                        # Apply recency weighting to the bonus
                        recency_factor = recency_weights[min(idx, len(recency_weights)-1)]
                        low_pivot_bonus += pivot_bonus * proximity_factor * recency_factor
        
        # Add the low pivot bonus to the score
        total_score = base_score + low_pivot_bonus
        print(f"Total score for {col}: {total_score} and base score: {base_score} and low pivot bonus: {low_pivot_bonus}")
        
        # Store score
        maxline_scores[col] = total_score
    
    # Calculate scores for each minline
    for col in minline_cols:
        # Skip columns with all NaN values
        if trends[col].isna().all():
            continue
            
        # Calculate proximity score
        # For minlines: price is near when price is close to but not below the line
        price_data = trends[price_field]
        trendline_data = trends[col]
        
        # Calculate distance as percentage of price
        distance = (price_data - trendline_data) / price_data
        
        # Identify near misses and touches with boolean masks
        near_misses = (distance >= 0) & (distance <= threshold)
        touches = (distance >= 0) & (distance <= threshold/2)
        
        # Apply recency weighting to each interaction
        weighted_near_misses = np.where(near_misses, recency_weights, 0)
        weighted_touches = np.where(touches, recency_weights * touch_weight, 0)
        
        # Calculate base score
        base_score = np.sum(weighted_near_misses) + np.sum(weighted_touches)
        
        # Add bonus for proximity to all_highs if provided
        high_pivot_bonus = 0
        if all_highs_array is not None:
            # Get non-nan indices in all_highs_array
            high_indices = np.where(~np.isnan(all_highs_array))[0]
            
            for idx in high_indices:
                # Skip if index is out of bounds for trendline data
                if idx >= len(trendline_data):
                    continue
                    
                # Calculate distance between the trendline and the high point
                high_value = all_highs_array[idx]
                trendline_value = trendline_data.iloc[idx]
                
                # Distance as percentage
                if not pd.isna(trendline_value) and not pd.isna(high_value) and high_value > 0:
                    dist_pct = (high_value - trendline_value) / high_value
                    print(f"dist_pct: {dist_pct} and threshold: {threshold}")
                    
                    # Award bonus points if the trendline is close to the high point
                    if 0 <= abs(dist_pct) <= threshold:
                        # More points for closer proximity
                        proximity_factor = 1.0 - (abs(dist_pct) / threshold)  # 1.0 for exact match, 0.0 for threshold
                        
                        # Apply recency weighting to the bonus
                        recency_factor = recency_weights[min(idx, len(recency_weights)-1)]
                        high_pivot_bonus += pivot_bonus * proximity_factor * recency_factor
        
        # Add the high pivot bonus to the score
        total_score = base_score + high_pivot_bonus
        
        # Store score
        minline_scores[col] = total_score
    
    # Sort dictionaries by score in descending order
    ranked_maxlines = {k: v for k, v in sorted(maxline_scores.items(), key=lambda item: item[1], reverse=True)}
    ranked_minlines = {k: v for k, v in sorted(minline_scores.items(), key=lambda item: item[1], reverse=True)}
    
    # Create result dictionary with ranked lines and their scores
    result = {
        "ranked_maxlines": ranked_maxlines,
        "ranked_minlines": ranked_minlines
    }
    
    return result