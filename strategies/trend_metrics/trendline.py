"""
defines trendline based indicator logic
based on
https://github.com/dysonance/Trendy
"""

import numpy as np
import pandas as pd

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
            # Previous pivot low was close to support (allows piercing)
            (abs(pivot_lows.shift(1) - level_data.shift(1)) <= 
             level_data.shift(1) * tolerance) &  # Close enough to be considered a test
            # Current close is higher than previous close (upward movement)
            (close_data > close_data.shift(1)) &
            # Level data is valid
            (~level_data.isna()) &
            (~level_data.shift(1).isna())
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
            # Previous pivot high was close to resistance (allows piercing)
            (abs(pivot_highs.shift(1) - level_data.shift(1)) <= 
             level_data.shift(1) * tolerance) &  # Close enough to be considered a test
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


def rank_trendlines(trends, price_field="Data", threshold=0.01, max_prefix="Max_Line_", min_prefix="Min_Line_", all_highs=None, all_lows=None, pivot_bonus=5.0):
    """
    Ranks trendlines based purely on bounce quality using generate_bounce_conditions.
    
    Scoring System based on professional trendline analysis:
    - Number of bounces: More bounces = higher score
    - Precision of bounces: Closer to trendline = higher score
    - Distribution of bounces: Well-distributed bounces across trendline length = higher score
    
    Args:
        trends: DataFrame containing price data and trendlines
        price_field: Column name for price data (default: "Data")
        threshold: Proximity threshold as a percentage (default: 0.01 or 1%)
        max_prefix: Prefix for maxline columns (default: "Max_Line_")
        min_prefix: Prefix for minline columns (default: "Min_Line_")
        all_highs: Series with high pivot points (REQUIRED)
        all_lows: Series with low pivot points (REQUIRED)
        pivot_bonus: Not used in this simplified version
        
    Returns:
        Dictionary with ranked maxlines and minlines based on bounce quality
    """
    import pandas as pd
    import numpy as np
    
    # Validate required inputs
    if all_highs is None or all_lows is None:
        print("ERROR: all_highs and all_lows pivot points are required for bounce analysis")
        return {"ranked_maxlines": {}, "ranked_minlines": {}}
    
    if len(trends) == 0:
        print("ERROR: Empty trends DataFrame")
        return {"ranked_maxlines": {}, "ranked_minlines": {}}
    
    # Get price data
    if price_field not in trends.columns:
        print(f"ERROR: Price field '{price_field}' not found in trends DataFrame")
        return {"ranked_maxlines": {}, "ranked_minlines": {}}
    
    price_data = trends[price_field]
    n_points = len(price_data)
    
    print(f"=== BOUNCE-BASED TRENDLINE RANKING ===")
    print(f"Analyzing {n_points} data points with threshold {threshold:.3f}")
    
    # Prepare pivot points as pandas Series
    def prepare_pivot_series(pivot_points, n_points):
        """Convert pivot points to properly indexed pandas Series"""
        if pivot_points is None:
            return pd.Series([np.nan] * n_points, index=price_data.index)
        
        # Ensure we have a pandas Series with the right index
        if hasattr(pivot_points, 'reindex'):
            return pivot_points.reindex(price_data.index, fill_value=np.nan)
        else:
            # Convert to Series if it's not already
            series = pd.Series(pivot_points, index=price_data.index[:len(pivot_points)])
            return series.reindex(price_data.index, fill_value=np.nan)
    
    pivot_highs_series = prepare_pivot_series(all_highs, n_points)
    pivot_lows_series = prepare_pivot_series(all_lows, n_points)
    
    # Define trendline categories
    trendline_categories = {
        'resistance': {
            'columns': [col for col in trends.columns if col.startswith(max_prefix) or col == "Max Line"],
            'scores': {},
            'direction': 'short'  # Bounce off resistance = short signal
        },
        'support': {
            'columns': [col for col in trends.columns if col.startswith(min_prefix) or col == "Min Line"],
            'scores': {},
            'direction': 'long'   # Bounce off support = long signal
        }
    }
    
    # === BOUNCE ANALYSIS FOR EACH TRENDLINE ===
    for trendline_type, config in trendline_categories.items():
        columns = config['columns']
        scores = config['scores']
        direction = config['direction']
        
        print(f"\n--- Analyzing {trendline_type.upper()} Lines ---")
        
        for col in columns:
            if col not in trends.columns:
                continue
                
            trendline_data = trends[col]
            
            # Skip trendlines with all NaN values
            if trendline_data.isna().all():
                scores[col] = 0.0
                continue
            
            print(f"\nEvaluating {col}:")
            
            # === STEP 1: DETECT BOUNCES ===
            try:
                bounce_conditions = generate_bounce_conditions(
                    close_data=price_data,
                    level_data=trendline_data,
                    direction=direction,
                    tolerance=threshold,
                    pivot_highs=pivot_highs_series,
                    pivot_lows=pivot_lows_series
                )
            except Exception as e:
                print(f"  ERROR detecting bounces: {e}")
                scores[col] = 0.0
                continue
            
            # Get bounce indices
            bounce_indices = bounce_conditions[bounce_conditions].index
            bounce_count = len(bounce_indices)
            
            if bounce_count == 0:
                print(f"  No bounces detected")
                scores[col] = 0.0
                continue
            
            print(f"  Found {bounce_count} bounces")
            
            # === STEP 2: CALCULATE BOUNCE QUALITY SCORES ===
            total_score = 0.0
            bounce_positions = []  # For distribution analysis
            
            for i, bounce_idx in enumerate(bounce_indices):
                # Get position in dataframe
                pos = bounce_conditions.index.get_loc(bounce_idx)
                bounce_positions.append(pos)
                
                # Get trendline and price values at bounce
                trendline_value = trendline_data.iloc[pos]
                price_value = price_data.iloc[pos]
                
                if pd.isna(trendline_value) or pd.isna(price_value):
                    continue
                
                # === PRECISION SCORE ===
                # How close was the bounce to the trendline?
                distance_pct = abs(price_value - trendline_value) / trendline_value
                
                # Precision factor: closer = higher score (1.0 at exact touch, decreases with distance)
                if distance_pct <= threshold:
                    precision_factor = 1.0 - (distance_pct / threshold)  # 1.0 to 0.0
                else:
                    precision_factor = max(0.0, 1.0 - (distance_pct / (threshold * 3)))  # Extended range
                
                # === BASE BOUNCE SCORE ===
                base_bounce_score = 100.0  # Base points per bounce
                
                # === STRENGTH SCORE ===
                # How strong was the reaction after the bounce?
                strength_bonus = 0.0
                if pos < len(price_data) - 1:
                    current_price = price_data.iloc[pos]
                    next_price = price_data.iloc[pos + 1]
                    
                    if direction == 'short':  # Resistance bounce - expect downward movement
                        if next_price < current_price:
                            strength_pct = (current_price - next_price) / current_price
                            strength_bonus = strength_pct * 50.0  # Up to 50 bonus points
                    else:  # Support bounce - expect upward movement
                        if next_price > current_price:
                            strength_pct = (next_price - current_price) / current_price
                            strength_bonus = strength_pct * 50.0  # Up to 50 bonus points
                
                # === RECENCY FACTOR ===
                # More recent bounces are slightly more valuable (higher pos = more recent)
                # Scale from 0.8 (oldest) to 1.2 (most recent) for reasonable weighting
                recency_factor = 0.8 + (0.4 * pos / len(price_data))  # 0.8 to 1.2
                
                # === INDIVIDUAL BOUNCE SCORE ===
                individual_score = (base_bounce_score + strength_bonus) * precision_factor * recency_factor
                total_score += individual_score
                
                print(f"    Bounce {i+1} at pos {pos}: precision={precision_factor:.3f}, strength_bonus={strength_bonus:.1f}, score={individual_score:.1f}")
            
            # === STEP 3: DISTRIBUTION BONUS ===
            distribution_bonus = calculate_distribution_bonus(bounce_positions, n_points)
            
            # === STEP 4: FINAL SCORE CALCULATION ===
            # Base score from individual bounces
            base_score = total_score
            
            # Count multiplier: More bounces = exponential benefit
            count_multiplier = 1.0 + (bounce_count - 1) * 0.3  # +30% per additional bounce
            
            # Distribution multiplier: Well-distributed bounces get bonus
            distribution_multiplier = 1.0 + distribution_bonus
            
            # Final score
            final_score = base_score * count_multiplier * distribution_multiplier
            scores[col] = final_score
            
            print(f"  FINAL SCORE: {final_score:.1f}")
            print(f"    Base score: {base_score:.1f}")
            print(f"    Count multiplier: {count_multiplier:.2f}x ({bounce_count} bounces)")
            print(f"    Distribution multiplier: {distribution_multiplier:.2f}x")
    
    # === STEP 5: SORT AND RETURN RESULTS ===
    ranked_results = {
        "ranked_maxlines": {k: v for k, v in sorted(trendline_categories['resistance']['scores'].items(), 
                                                   key=lambda item: item[1], 
                                                   reverse=True)},
        "ranked_minlines": {k: v for k, v in sorted(trendline_categories['support']['scores'].items(), 
                                                   key=lambda item: item[1], 
                                                   reverse=True)}
    }
    
    # Print final rankings
    print(f"\n=== FINAL RANKINGS ===")
    print("RESISTANCE LINES:")
    for i, (line, score) in enumerate(ranked_results["ranked_maxlines"].items(), 1):
        if score > 0:
            print(f"  {i}. {line}: {score:.1f}")
    
    print("SUPPORT LINES:")
    for i, (line, score) in enumerate(ranked_results["ranked_minlines"].items(), 1):
        if score > 0:
            print(f"  {i}. {line}: {score:.1f}")
    
    return ranked_results


def calculate_distribution_bonus(bounce_positions, total_length):
    """
    Calculate distribution bonus based on how well bounces are spread across trendline.
    
    Args:
        bounce_positions: List of positions where bounces occurred
        total_length: Total length of the data
        
    Returns:
        float: Distribution bonus (0.0 to 1.0)
    """
    if len(bounce_positions) <= 1:
        return 0.0
    
    # Sort positions
    positions = sorted(bounce_positions)
    
    # Calculate coverage ratio (how much of the trendline is covered)
    coverage_span = positions[-1] - positions[0]
    coverage_ratio = coverage_span / max(total_length - 1, 1)
    
    # Calculate distribution uniformity
    gaps = np.diff(positions)
    if len(gaps) > 0:
        ideal_gap = coverage_span / len(gaps)
        if ideal_gap > 0:
            gap_deviations = [abs(gap - ideal_gap) / ideal_gap for gap in gaps]
            uniformity = max(0.0, 1.0 - np.mean(gap_deviations))
        else:
            uniformity = 0.0
    else:
        uniformity = 0.0
    
    # Combine coverage and uniformity
    distribution_bonus = (coverage_ratio * 0.6 + uniformity * 0.4) * 0.5  # Max 50% bonus
    
    return distribution_bonus