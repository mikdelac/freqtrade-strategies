from typing import List, Tuple, Optional, Callable, Dict, Any
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import random
import time
from .volatility_models import GARCHModel

# Import trendline functions and Trendline class from trend_metrics using absolute imports
try:
    from trend_metrics.trendline import gentrends, rank_trendlines, Trendline, generate_bounce_conditions, calculate_r_squared, generate_trendlines_for_period, apply_monte_carlo_results
except ImportError:
    # Fallback for different import structures
    try:
        from strategies.trend_metrics.trendline import gentrends, rank_trendlines, Trendline, generate_bounce_conditions, calculate_r_squared, generate_trendlines_for_period, apply_monte_carlo_results
    except ImportError:
        print("Warning: Could not import trendline functions. Some functionality may be limited.")
        # Define dummy functions to prevent errors
        def gentrends(*args, **kwargs):
            return pd.DataFrame()
        def rank_trendlines(*args, **kwargs):
            return {"ranked_maxlines": {}, "ranked_minlines": {}}
        def generate_bounce_conditions(*args, **kwargs):
            return pd.Series([False] * 100)  # Return dummy series
        def calculate_r_squared(*args, **kwargs):
            return 0.0  # Return dummy R-squared
        
        # Define dummy Trendline class
        class Trendline:
            def __init__(self, *args, **kwargs):
                pass

class MonteCarloSimulator:
    """
    Monte Carlo simulator for financial risk analysis.
    
    This class provides Monte Carlo simulation capabilities for generating
    financial return scenarios using various volatility models including GARCH.
    """
    
    def __init__(self, volatility_model: Optional[GARCHModel] = None):
        """
        Initialize Monte Carlo simulator.
        
        Args:
            volatility_model: Optional volatility model (defaults to GARCH(1,1))
        """
        self.volatility_model = volatility_model or GARCHModel()
    
    def simulate_returns(self, 
                        T: int, 
                        iterations: int = 1000,
                        calculate_variance_fn: Optional[Callable[[float, float], float]] = None,
                        random_generator_fn: Optional[Callable[[float], float]] = None,
                        initial_variance: Optional[float] = None) -> np.ndarray:
        """
        Perform Monte Carlo simulation for financial returns.

        Args:
            T: Number of time steps (days) to simulate.
            iterations: Number of simulation paths.
            calculate_variance_fn: Optional callback function to calculate variance with signature (sigma2_t, R_t) -> new_sigma2_t.
                                 If None, uses the volatility model's update_conditional_variance method.
            random_generator_fn: Optional callback function to generate random returns with signature (sigma) -> return.
                               If None, uses normal distribution (sigma * N(0,1)).
            initial_variance: Optional initial variance value to use. If None, uses the model's long-term variance.

        Returns:
            np.ndarray: Simulated returns with shape (iterations, T).
            
        Raises:
            ValueError: If T or iterations are not positive.
        """
        # Validate input parameters
        if T <= 0 or iterations <= 0:
            raise ValueError("T and iterations must be positive")
            
        # If calculate_variance_fn is None, use the volatility model's method
        if calculate_variance_fn is None:
            calculate_variance_fn = lambda sigma2_t, R_t: self.volatility_model.update_conditional_variance(R_t)
        
        # If random_generator_fn is None, use normal distribution
        if random_generator_fn is None:
            random_generator_fn = lambda sigma: sigma * np.random.normal()
            
        # Initialize variance with long-term variance if not provided
        if initial_variance is None:
            initial_variance = self.volatility_model.omega / (1 - self.volatility_model.alpha - self.volatility_model.beta)
            
        # Initialize array to store simulated returns
        R = np.zeros((iterations, T))
        
        # Perform Monte Carlo simulation
        for i in range(iterations):
            # Reset model variance for each simulation path
            self.volatility_model.variance = initial_variance
            sigma2_t = initial_variance  # Start with initial variance
            
            for t in range(T):
                # Generate return using the provided random generator function
                sigma_t = np.sqrt(sigma2_t)
                R[i, t] = random_generator_fn(sigma_t)
                
                # Update variance for next step
                sigma2_t = calculate_variance_fn(sigma2_t, R[i, t])
        
        return R
    
    def simulate_cumulative_returns(self, 
                                   T: int, 
                                   iterations: int = 1000,
                                   calculate_variance_fn: Optional[Callable[[float, float], float]] = None,
                                   random_generator_fn: Optional[Callable[[float], float]] = None,
                                   initial_variance: Optional[float] = None) -> np.ndarray:
        """
        Perform Monte Carlo simulation and return cumulative returns for each path.

        Args:
            T: Number of time steps (days) to simulate.
            iterations: Number of simulation paths.
            calculate_variance_fn: Optional callback function to calculate variance.
            random_generator_fn: Optional callback function to generate random returns.
            initial_variance: Optional initial variance value to use.

        Returns:
            np.ndarray: Cumulative returns for each simulation path with shape (iterations,).
        """
        # Get the full return matrix
        returns_matrix = self.simulate_returns(
            T=T,
            iterations=iterations,
            calculate_variance_fn=calculate_variance_fn,
            random_generator_fn=random_generator_fn,
            initial_variance=initial_variance
        )
        
        # Calculate the sum of returns for each simulation path
        return np.sum(returns_matrix, axis=1)
    
    def simulate_with_garch(self, 
                           T: int, 
                           iterations: int = 1000) -> np.ndarray:
        """
        Convenience method for GARCH-based Monte Carlo simulation.
        
        Args:
            T: Number of time steps to simulate.
            iterations: Number of simulation paths.
            
        Returns:
            np.ndarray: Cumulative returns for each simulation path.
        """
        return self.simulate_cumulative_returns(T=T, iterations=iterations)
    
    def simulate_with_custom_distribution(self, 
                                        T: int, 
                                        iterations: int = 1000,
                                        distribution_fn: Optional[Callable[[float], float]] = None) -> np.ndarray:
        """
        Simulate returns using a custom distribution function.
        
        Args:
            T: Number of time steps to simulate.
            iterations: Number of simulation paths.
            distribution_fn: Function that takes volatility and returns a random draw.
                           If None, uses Student's t-distribution with 5 degrees of freedom.
            
        Returns:
            np.ndarray: Cumulative returns for each simulation path.
        """
        if distribution_fn is None:
            # Default to Student's t-distribution for fat tails
            from scipy.stats import t
            distribution_fn = lambda sigma: sigma * t.rvs(df=5)
        
        return self.simulate_cumulative_returns(
            T=T,
            iterations=iterations,
            random_generator_fn=distribution_fn
        )
    
    def get_simulation_statistics(self, returns: np.ndarray) -> dict:
        """
        Calculate statistics from simulation results.
        
        Args:
            returns: Array of simulated returns.
            
        Returns:
            dict: Dictionary containing simulation statistics.
        """
        return {
            'mean': np.mean(returns),
            'std': np.std(returns),
            'min': np.min(returns),
            'max': np.max(returns),
            'percentile_1': np.percentile(returns, 1),
            'percentile_5': np.percentile(returns, 5),
            'percentile_95': np.percentile(returns, 95),
            'percentile_99': np.percentile(returns, 99),
            'skewness': self._calculate_skewness(returns),
            'kurtosis': self._calculate_kurtosis(returns)
        }
    
    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """Calculate skewness of returns."""
        mean = np.mean(returns)
        std = np.std(returns)
        return np.mean(((returns - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, returns: np.ndarray) -> float:
        """Calculate excess kurtosis of returns."""
        mean = np.mean(returns)
        std = np.std(returns)
        return np.mean(((returns - mean) / std) ** 4) - 3


class TrendlineMonteCarloOptimizer:
    """
    Contains the core Monte Carlo optimization logic for finding optimal trendline periods.
    
    This class handles the heavy computational work of running Monte Carlo simulations
    to find the best lookback periods for trendline analysis.
    """
    
    def __init__(self, mc_iterations: int, min_lookback_period: int, 
                 trendline_proximity_threshold: float):
        """
        Initialize the Monte Carlo optimizer.
        
        Args:
            mc_iterations: Number of Monte Carlo iterations to run
            min_lookback_period: Minimum lookback period to test
            trendline_proximity_threshold: Threshold for trendline proximity scoring
        """
        self.mc_iterations = mc_iterations
        self.min_lookback_period = min_lookback_period
        self.trendline_proximity_threshold = trendline_proximity_threshold
    
    def monte_carlo_period_optimization(self, dataframe: pd.DataFrame, pair: str = "UNKNOWN",
                                       recalc_interval_minutes: int = 120):
        # Set MAX_LOOKBACK_PERIOD dynamically based on available data
        total_candles = len(dataframe)
        max_lookback_period = total_candles
        
        if total_candles < self.min_lookback_period:
            print(f"Not enough data for Monte Carlo optimization. Need at least {self.min_lookback_period} candles, got {total_candles}.")
            return {
                'optimal_resistance_period': self.min_lookback_period,
                'optimal_support_period': self.min_lookback_period,
                'resistance_score': 0.0,
                'support_score': 0.0,
                'resistance_line': np.full(len(dataframe), np.nan),
                'support_line': np.full(len(dataframe), np.nan)
            }
        
        # Seed random number generator for reproducible results with some variability
        random.seed(int(time.time() * 1000) % 10000)  # Use current time for seed
        
        # Create core periods by dividing total candles
        print(f"=== Monte Carlo Period Optimization with Core Periods for {pair} ===")
        
        # Generate core periods by dividing total_candles into segments
        min_lookback_period = max(self.min_lookback_period, 50)
        core_periods = []
        
        # Create divisions of total_candles
        for divisor in [1.1, 1.2, 1.4, 1.6, 1.8, 2, 3, 4, 5, 6, 8]:
            period = int(total_candles // divisor)
            if period >= min_lookback_period:
                core_periods.append(period)
        
        # Remove duplicates and sort
        core_periods = sorted(list(set(core_periods)))
        
        # Generate additional random periods to reach mc_iterations
        remaining_iterations = self.mc_iterations - len(core_periods)
        random_periods = []
        
        if remaining_iterations > 0:
            for _ in range(remaining_iterations):
                random_period = random.randint(min_lookback_period, max_lookback_period)
                random_periods.append(random_period)
        
        # Combine core periods with random periods
        lookback_periods = core_periods + random_periods
        
        # Ensure we have exactly mc_iterations periods
        if len(lookback_periods) > self.mc_iterations:
            lookback_periods = lookback_periods[:self.mc_iterations]
        elif len(lookback_periods) < self.mc_iterations:
            # Pad with repeated core periods if needed
            while len(lookback_periods) < self.mc_iterations:
                lookback_periods.extend(core_periods[:self.mc_iterations - len(lookback_periods)])
        
        # Shuffle to randomize the order
        random.shuffle(lookback_periods)
        
        print(f"Generated {len(lookback_periods)} lookback periods by dividing total candles ({total_candles}):")
        print(f"  Range: {min(lookback_periods)} to {max(lookback_periods)} candles")
        print(f"  Mean: {np.mean(lookback_periods):.1f}, Std: {np.std(lookback_periods):.1f}")
        print(f"  Core periods from divisions: {core_periods}")
        
        print(f"Starting Monte Carlo period optimization for {pair} with {len(lookback_periods)} iterations...")
        print(f"Testing periods from {min(lookback_periods)} to {max(lookback_periods)} candles (total data: {total_candles})")
        
        best_resistance_score = 0.0
        best_support_score = 0.0
        best_resistance_period = self.min_lookback_period
        best_support_period = self.min_lookback_period
        best_resistance_line = np.full(len(dataframe), np.nan)
        best_support_line = np.full(len(dataframe), np.nan)
        best_resistance_slope = 0.0
        best_support_slope = 0.0
        best_resistance_trendline = None
        best_support_trendline = None
        
        # Store all tested periods and scores for analysis
        tested_periods = []
        resistance_scores = []
        support_scores = []
        
        # Track period distribution for debugging
        period_counts = {}
        
        for iteration, random_period in enumerate(lookback_periods):
            # Track period distribution
            period_range = (random_period // 100) * 100  # Group by hundreds
            period_counts[period_range] = period_counts.get(period_range, 0) + 1
            
            try:
                # Test this period
                recent_data = dataframe.tail(random_period).copy()
                
                # Use the helper method to generate trendlines and create Trendline objects
                trendline_results = generate_trendlines_for_period(
                    recent_data=recent_data,
                    random_period=random_period,
                    mc_recalc_interval_minutes=recalc_interval_minutes,
                    trendline_proximity_threshold=self.trendline_proximity_threshold
                )
                
                trends = trendline_results['trends']
                trendline_objects = trendline_results.get('trendline_objects', [])
                recent_data = trendline_results['recent_data']
                
                # Generate bounce conditions and update Trendline objects with bounce counts
                trendline_config = {
                    'resistance': {
                        'line_key': 'Max Line',
                        'direction': 'short',
                        'line_data': trends.get('Max Line', pd.Series())
                    },
                    'support': {
                        'line_key': 'Min Line', 
                        'direction': 'long',
                        'line_data': trends.get('Min Line', pd.Series())
                    }
                }
                
                for trendline_obj in trendline_objects:
                    config = trendline_config.get(trendline_obj.trendline_type)
                    if config and config['line_key'] in trends.columns:
                        try:
                            # Ensure indices match by reindexing the trendline data
                            level_line = config['line_data'].copy()
                            level_line.index = recent_data.index
                            
                            bounce_conditions = generate_bounce_conditions(
                                close_data=recent_data['close'],
                                level_data=level_line,
                                direction=config['direction'],
                                tolerance=self.trendline_proximity_threshold,
                                pivot_highs=recent_data['all_highs'],
                                pivot_lows=recent_data['all_lows']
                            )
                            # Store bounce count in the Trendline object
                            trendline_obj.bounce_count = bounce_conditions.sum()
                            
                            # Capture bounce timestamps
                            bounce_indices = bounce_conditions[bounce_conditions].index
                            trendline_obj.bounce_timestamps = recent_data.loc[bounce_indices, 'date'].tolist()
                            
                        except Exception as e:
                            print(f"Error generating {trendline_obj.trendline_type} bounce conditions: {e}")
                            trendline_obj.bounce_count = 0
                            trendline_obj.bounce_timestamps = []
                
                # Calculate scores using rank_trendlines - pass the trendline objects so it can use their bounce counts
                main_lines_score = rank_trendlines(
                    trends,
                    trendline_objects  # Pass the trendline objects with bounce counts
                )
                
                current_resistance_score = main_lines_score["ranked_maxlines"].get("Max Line", 0)
                current_support_score = main_lines_score["ranked_minlines"].get("Min Line", 0)
                
                # Store results
                tested_periods.append(random_period)
                resistance_scores.append(current_resistance_score)
                support_scores.append(current_support_score)
                
                # Check if this is the best resistance line so far
                if current_resistance_score > best_resistance_score:
                    best_resistance_score = current_resistance_score
                    best_resistance_period = random_period
                    
                    # Store the resistance line
                    resistance_line = np.full(len(dataframe), np.nan)
                    last_idx = len(dataframe) - len(recent_data)
                    for i in range(len(trends)):
                        current_idx = last_idx + i
                        if current_idx < len(dataframe):
                            resistance_line[current_idx] = trends['Max Line'].iloc[i]
                    best_resistance_line = resistance_line
                    
                    best_resistance_slope = trendline_results['resistance_slope']
                    
                    # Store the best resistance trendline object
                    for trendline_obj in trendline_objects:
                        if trendline_obj.trendline_type == 'resistance':
                            best_resistance_trendline = trendline_obj
                            break
                    
                    print(f"New best resistance found for {pair} at iteration {iteration + 1}: period {random_period}, score {current_resistance_score:.4f}, bounces {best_resistance_trendline.bounce_count if best_resistance_trendline else 'N/A'}")
                
                # Check if this is the best support line so far
                if current_support_score > best_support_score:
                    best_support_score = current_support_score
                    best_support_period = random_period
                    
                    # Store the support line
                    support_line = np.full(len(dataframe), np.nan)
                    last_idx = len(dataframe) - len(recent_data)
                    for i in range(len(trends)):
                        current_idx = last_idx + i
                        if current_idx < len(dataframe):
                            support_line[current_idx] = trends['Min Line'].iloc[i]
                    best_support_line = support_line
                    
                    best_support_slope = trendline_results['support_slope']
                    
                    # Store the best support trendline object
                    for trendline_obj in trendline_objects:
                        if trendline_obj.trendline_type == 'support':
                            best_support_trendline = trendline_obj
                            break
                    
                    print(f"New best support found for {pair} at iteration {iteration + 1}: period {random_period}, score {current_support_score:.4f}, bounces {best_support_trendline.bounce_count if best_support_trendline else 'N/A'}")
                
                # Progress reporting with more details
                if (iteration + 1) % 100 == 0:
                    print(f"Monte Carlo progress for {pair}: {iteration + 1}/{len(lookback_periods)} completed")
                    print(f"Current best - Resistance: {best_resistance_score:.4f} (period {best_resistance_period}), Support: {best_support_score:.4f} (period {best_support_period})")
                    print(f"Last 5 tested periods: {tested_periods[-5:] if len(tested_periods) >= 5 else tested_periods}")
                
            except Exception as e:
                print(f"Error in Monte Carlo iteration {iteration} for {pair}: {e}")
                continue
        
        print(f"Core-period Monte Carlo optimization completed for {pair}!")
        print(f"Optimal resistance period: {best_resistance_period} (score: {best_resistance_score:.4f})")
        print(f"Optimal support period: {best_support_period} (score: {best_support_score:.4f})")
        print(f"Tested periods range: {min(tested_periods) if tested_periods else 'N/A'} to {max(tested_periods) if tested_periods else 'N/A'} candles")
        
        # Print period distribution for debugging
        print(f"Period distribution for {pair} (grouped by hundreds):")
        for period_range in sorted(period_counts.keys()):
            print(f"  {period_range}-{period_range+99}: {period_counts[period_range]} times")
        
        # Print some statistics about the tested periods
        if tested_periods:
            print(f"Period statistics for {pair} - Min: {min(tested_periods)}, Max: {max(tested_periods)}, Mean: {np.mean(tested_periods):.1f}")
            print(f"Unique periods tested: {len(set(tested_periods))} out of {len(tested_periods)} total")
        
        return {
            'optimal_resistance_period': best_resistance_period,
            'optimal_support_period': best_support_period,
            'resistance_score': best_resistance_score,
            'support_score': best_support_score,
            'resistance_line': best_resistance_line,
            'support_line': best_support_line,
            'tested_periods': tested_periods,
            'all_resistance_scores': resistance_scores,
            'all_support_scores': support_scores,
            'max_lookback_period': max_lookback_period,
            'period_distribution': period_counts,
            'best_resistance_slope': best_resistance_slope,
            'best_support_slope': best_support_slope,
            'best_resistance_trendline': best_resistance_trendline,
            'best_support_trendline': best_support_trendline
        }
