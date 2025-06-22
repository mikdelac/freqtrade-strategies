from typing import List, Tuple, Optional, Callable, Dict, Any
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import random
import time
from .volatility_models import GARCHModel

# Import trendline functions from trend_metrics using absolute imports
try:
    from trend_metrics.trendline import gentrends, rank_trendlines
except ImportError:
    # Fallback for different import structures
    try:
        from strategies.trend_metrics.trendline import gentrends, rank_trendlines
    except ImportError:
        print("Warning: Could not import trendline functions. Some functionality may be limited.")
        # Define dummy functions to prevent errors
        def gentrends(*args, **kwargs):
            return pd.DataFrame()
        def rank_trendlines(*args, **kwargs):
            return {"ranked_maxlines": {}, "ranked_minlines": {}}

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


class MonteCarloCache:
    """
    Manages per-pair caching of Monte Carlo results with timing logic.
    
    This class encapsulates all caching functionality for Monte Carlo optimization results,
    including when to recalculate, storing results per trading pair, and retrieving them.
    """
    
    def __init__(self, recalc_interval_minutes: int):
        """
        Initialize Monte Carlo cache.
        
        Args:
            recalc_interval_minutes: Interval in minutes between recalculations
        """
        self.recalc_interval_minutes = recalc_interval_minutes
        self.last_mc_recalc_time_per_pair: Dict[str, Optional[datetime]] = {}
        self.cached_mc_results_per_pair: Dict[str, Dict[str, Any]] = {}
    
    def should_recalculate(self, pair: str, current_time: Optional[datetime] = None) -> bool:
        """
        Determine if Monte Carlo results should be recalculated for a pair.
        
        Args:
            pair: Trading pair name
            current_time: Current time (uses current UTC time if None)
            
        Returns:
            bool: True if recalculation is needed
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)
        
        if pair not in self.last_mc_recalc_time_per_pair:
            return True
        
        last_time = self.last_mc_recalc_time_per_pair[pair]
        if last_time is None:
            return True
        
        time_diff = (current_time - last_time).total_seconds() / 60
        return time_diff >= self.recalc_interval_minutes
    
    def cache_results(self, pair: str, results: Dict[str, Any], timestamp: Optional[datetime] = None) -> None:
        """
        Cache Monte Carlo results for a pair.
        
        Args:
            pair: Trading pair name
            results: Monte Carlo optimization results
            timestamp: Timestamp of the results (uses current UTC time if None)
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        self.cached_mc_results_per_pair[pair] = results
        self.last_mc_recalc_time_per_pair[pair] = timestamp
    
    def get_results(self, pair: str) -> Dict[str, Any]:
        """
        Get cached Monte Carlo results for a pair.
        
        Args:
            pair: Trading pair name
            
        Returns:
            Dict containing Monte Carlo results, or empty dict if not available
        """
        return self.cached_mc_results_per_pair.get(pair, {})
    
    def has_results(self, pair: str) -> bool:
        """
        Check if Monte Carlo results are available for a pair.
        
        Args:
            pair: Trading pair name
            
        Returns:
            bool: True if results are cached for the pair
        """
        return pair in self.cached_mc_results_per_pair and bool(self.cached_mc_results_per_pair[pair])
    
    def clear_cache(self, pair: str) -> None:
        """
        Clear Monte Carlo cache for a specific pair.
        
        Args:
            pair: Trading pair name
        """
        if pair in self.cached_mc_results_per_pair:
            del self.cached_mc_results_per_pair[pair]
        if pair in self.last_mc_recalc_time_per_pair:
            del self.last_mc_recalc_time_per_pair[pair]
        print(f"Cleared Monte Carlo cache for {pair}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the Monte Carlo cache across all pairs.
        
        Returns:
            Dict containing cache statistics
        """
        stats = {
            'total_pairs_cached': len(self.cached_mc_results_per_pair),
            'pairs_with_results': [],
            'cache_ages_minutes': {},
            'last_recalc_times': {}
        }
        
        current_time = datetime.now(timezone.utc)
        
        for pair in self.cached_mc_results_per_pair:
            if self.cached_mc_results_per_pair[pair]:
                stats['pairs_with_results'].append(pair)
                
            if pair in self.last_mc_recalc_time_per_pair and self.last_mc_recalc_time_per_pair[pair]:
                last_time = self.last_mc_recalc_time_per_pair[pair]
                age_minutes = (current_time - last_time).total_seconds() / 60
                stats['cache_ages_minutes'][pair] = age_minutes
                stats['last_recalc_times'][pair] = last_time.isoformat()
        
        return stats
    
    def print_cache_summary(self) -> None:
        """
        Print a summary of the Monte Carlo cache status for all pairs.
        """
        stats = self.get_cache_stats()
        
        print("=== Monte Carlo Cache Summary ===")
        print(f"Total pairs with cache: {stats['total_pairs_cached']}")
        print(f"Pairs with valid results: {len(stats['pairs_with_results'])}")
        
        if stats['pairs_with_results']:
            print("Pairs with cached results:")
            for pair in stats['pairs_with_results']:
                age = stats['cache_ages_minutes'].get(pair, 0)
                print(f"  - {pair}: {age:.1f} minutes old")
        
        if len(stats['cache_ages_minutes']) != len(stats['pairs_with_results']):
            stale_pairs = set(self.cached_mc_results_per_pair.keys()) - set(stats['pairs_with_results'])
            if stale_pairs:
                print(f"Pairs with stale/empty cache: {list(stale_pairs)}")


class TrendlineMonteCarloOptimizer:
    """
    Contains the core Monte Carlo optimization logic for finding optimal trendline periods.
    
    This class handles the heavy computational work of running Monte Carlo simulations
    to find the best lookback periods for trendline analysis, completely separate from
    caching or how the results are used.
    """
    
    def __init__(self, mc_iterations: int, min_lookback_period: int, 
                 trendline_proximity_threshold: float, trend_analyzer):
        """
        Initialize the Monte Carlo optimizer.
        
        Args:
            mc_iterations: Number of Monte Carlo iterations to run
            min_lookback_period: Minimum lookback period to test
            trendline_proximity_threshold: Threshold for trendline proximity scoring
            trend_analyzer: TrendAnalysis instance for swing point detection
        """
        self.mc_iterations = mc_iterations
        self.min_lookback_period = min_lookback_period
        self.trendline_proximity_threshold = trendline_proximity_threshold
        self.trend_analyzer = trend_analyzer
    
    def generate_fixed_lookback_periods(self, dataframe: pd.DataFrame) -> List[int]:
        """
        Generate fixed lookback periods for Monte Carlo optimization.
        This replaces the volatility-based period generation to properly separate 
        GARCH volatility estimation from lookback period selection.
        
        Args:
            dataframe: DataFrame with OHLCV data
            
        Returns:
            List[int]: List of fixed lookback periods for testing
        """
        total_candles = len(dataframe)
        max_lookback_period = total_candles  # Use all available data
        min_lookback_period = max(self.min_lookback_period, 50)  # Use fixed min
        
        # Define core fixed periods that cover different time horizons
        core_periods = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000]
        
        # Filter periods based on available data
        valid_core_periods = [p for p in core_periods if min_lookback_period <= p <= max_lookback_period]
        
        # Generate additional random periods to reach MC_ITERATIONS
        remaining_iterations = self.mc_iterations - len(valid_core_periods)
        random_periods = []
        
        if remaining_iterations > 0:
            # Generate random periods to fill the remaining iterations
            for _ in range(remaining_iterations):
                random_period = random.randint(min_lookback_period, max_lookback_period)
                random_periods.append(random_period)
        
        # Combine core periods with random periods
        all_periods = valid_core_periods + random_periods
        
        # Ensure we have exactly MC_ITERATIONS periods
        if len(all_periods) > self.mc_iterations:
            all_periods = all_periods[:self.mc_iterations]
        elif len(all_periods) < self.mc_iterations:
            # Pad with repeated core periods if needed
            while len(all_periods) < self.mc_iterations:
                all_periods.extend(valid_core_periods[:self.mc_iterations - len(all_periods)])
        
        # Shuffle to randomize the order
        random.shuffle(all_periods)
        
        print(f"Generated {len(all_periods)} fixed lookback periods:")
        print(f"  Range: {min(all_periods)} to {max(all_periods)} candles")
        print(f"  Mean: {np.mean(all_periods):.1f}, Std: {np.std(all_periods):.1f}")
        print(f"  Core periods included: {valid_core_periods}")
        
        return all_periods
    
    def _generate_trendlines_for_period(self, recent_data: pd.DataFrame, random_period: int) -> Dict[str, Any]:
        """
        Generate trendlines and calculate scores for a specific lookback period.
        
        Args:
            recent_data: DataFrame with recent OHLCV data
            random_period: Lookback period to test
            
        Returns:
            Dict containing trendlines and scores
        """
        # Find swing points for this period with adaptive parameters
        high_swing_points = self.trend_analyzer._find_swing_points(
            prices=recent_data['high'].values,
            price_type='high',
            min_points=max(3, random_period // 50),  # Adaptive min_points
            distance=max(5, random_period // 100)    # Adaptive distance
        )
        
        low_swing_points = self.trend_analyzer._find_swing_points(
            prices=recent_data['low'].values,
            price_type='low',
            min_points=max(3, random_period // 50),
            distance=max(5, random_period // 100)
        )
        
        # Initialize swing point columns
        recent_data['all_highs'] = np.nan
        recent_data['all_lows'] = np.nan
        
        # Map swing points
        for idx, price in high_swing_points:
            if idx < len(recent_data):
                recent_data.iloc[idx, recent_data.columns.get_loc('all_highs')] = price
        
        for idx, price in low_swing_points:
            if idx < len(recent_data):
                recent_data.iloc[idx, recent_data.columns.get_loc('all_lows')] = price
        
        # Generate trends for this period
        trends = gentrends(recent_data, field='close', window=1/3.0)
        
        # Calculate scores for the main lines
        main_lines_score = rank_trendlines(
            trends,
            price_field="Data", 
            threshold=self.trendline_proximity_threshold,
            all_highs=recent_data['all_highs'],
            all_lows=recent_data['all_lows'],
            pivot_bonus=10.0  # Higher bonus for MC optimization
        )
        
        resistance_score = main_lines_score["ranked_maxlines"].get("Max Line", 0)
        support_score = main_lines_score["ranked_minlines"].get("Min Line", 0)
        
        return {
            'trends': trends,
            'resistance_score': resistance_score,
            'support_score': support_score,
            'high_swing_points': high_swing_points,
            'low_swing_points': low_swing_points
        }
    
    def monte_carlo_period_optimization(self, dataframe: pd.DataFrame, pair: str = "UNKNOWN") -> Dict[str, Any]:
        """
        Use Monte Carlo simulation to test different lookback periods and find
        the ones that produce the highest scoring resistance and support lines.
        Now uses fixed lookback periods
        
        Args:
            dataframe: DataFrame with OHLCV data
            pair: Trading pair name for logging
            
        Returns:
            Dict containing optimal periods and their scores
        """
        # Set MAX_LOOKBACK_PERIOD dynamically based on available data
        total_candles = len(dataframe)
        max_lookback_period = min(total_candles, 1000)  # Fixed max instead of garch_max_candles
        
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
        
        # Generate fixed lookback periods
        print(f"=== Monte Carlo Period Optimization with Fixed Periods for {pair} ===")
        lookback_periods = self.generate_fixed_lookback_periods(dataframe)
        
        print(f"Starting Monte Carlo period optimization for {pair} with {len(lookback_periods)} iterations...")
        print(f"Testing periods from {min(lookback_periods)} to {max(lookback_periods)} candles (total data: {total_candles})")
        
        best_resistance_score = 0.0
        best_support_score = 0.0
        best_resistance_period = self.min_lookback_period
        best_support_period = self.min_lookback_period
        best_resistance_line = np.full(len(dataframe), np.nan)
        best_support_line = np.full(len(dataframe), np.nan)
        
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
                
                # Use the helper method to generate trendlines and scores
                trendline_results = self._generate_trendlines_for_period(recent_data, random_period)
                
                current_resistance_score = trendline_results['resistance_score']
                current_support_score = trendline_results['support_score']
                trends = trendline_results['trends']
                
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
                    
                    print(f"New best resistance found for {pair} at iteration {iteration + 1}: period {random_period}, score {current_resistance_score:.4f}")
                
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
                    
                    print(f"New best support found for {pair} at iteration {iteration + 1}: period {random_period}, score {current_support_score:.4f}")
                
                # Progress reporting with more details
                if (iteration + 1) % 100 == 0:
                    print(f"Monte Carlo progress for {pair}: {iteration + 1}/{len(lookback_periods)} iterations completed")
                    print(f"Current best - Resistance: {best_resistance_score:.4f} (period {best_resistance_period}), Support: {best_support_score:.4f} (period {best_support_period})")
                    print(f"Last 5 tested periods: {tested_periods[-5:] if len(tested_periods) >= 5 else tested_periods}")
                
            except Exception as e:
                print(f"Error in Monte Carlo iteration {iteration} for {pair}: {e}")
                continue
        
        print(f"Fixed-period Monte Carlo optimization completed for {pair}!")
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
            'period_distribution': period_counts
        }


class MonteCarloManager:
    """
    Coordinates Monte Carlo optimization with caching and result application.
    
    This class acts as the public-facing interface that coordinates the MonteCarloCache
    and TrendlineMonteCarloOptimizer to execute the full process: check the cache,
    run the optimization if needed, and apply the results to the dataframe.
    """
    
    def __init__(self, mc_iterations: int, min_lookback_period: int, 
                 recalc_interval_minutes: int, trendline_proximity_threshold: float,
                 trend_analyzer, volatility_model, garch_model):
        """
        Initialize the Monte Carlo manager.
        
        Args:
            mc_iterations: Number of Monte Carlo iterations
            min_lookback_period: Minimum lookback period
            recalc_interval_minutes: Cache recalculation interval in minutes
            trendline_proximity_threshold: Threshold for trendline proximity scoring
            trend_analyzer: TrendAnalysis instance
            volatility_model: VolatilityModel instance
            garch_model: GARCHModel instance
        """
        self.cache = MonteCarloCache(recalc_interval_minutes)
        self.optimizer = TrendlineMonteCarloOptimizer(
            mc_iterations, min_lookback_period, 
            trendline_proximity_threshold, trend_analyzer
        )
        self.volatility_model = volatility_model
        self.garch_model = garch_model
    
    def estimate_current_volatility_regime(self, dataframe: pd.DataFrame) -> Tuple[str, float, float]:
        """
        Estimate current volatility regime using GARCH model for risk management purposes.
        This is separated from lookback period generation to ensure proper use of GARCH.
        
        Args:
            dataframe: DataFrame with OHLCV data
            
        Returns:
            Tuple[str, float, float]: (regime, volatility, risk_multiplier)
        """
        # Calculate log returns for GARCH volatility estimation
        log_returns = np.log(dataframe['close'].pct_change() + 1).dropna().values
        
        # Use a reasonable window for volatility estimation
        volatility_window = min(len(log_returns), 500)  # Fixed window for volatility estimation
        volatility_window = max(volatility_window, 100)   # Minimum window for reliable estimation
        
        # Calculate current volatility using GARCH model
        recent_returns = log_returns[-volatility_window:]
        
        # Clean data and calculate volatility
        if len(recent_returns) > 0 and not np.isnan(recent_returns).all() and not np.isinf(recent_returns).any():
            # Use VolatilityModel's calculate_volatility method with GARCHModel
            current_volatility = self.volatility_model.calculate_volatility(recent_returns, self.garch_model)
            print(f"  GARCH volatility result: {current_volatility:.6f}")
        else:
            print(f"  Invalid data for GARCH calculation, using fallback")
            # Fallback to simple standard deviation using VolatilityModel
            current_volatility = self.volatility_model.calculate_volatility(recent_returns)
            
        regime, risk_multiplier = self.volatility_model.get_regime_and_multiplier(current_volatility)
        
        print(f"Current volatility regime: {regime} (volatility: {current_volatility:.6f}, risk multiplier: {risk_multiplier:.2f})")
        
        return regime, current_volatility, risk_multiplier
    
    def execute_monte_carlo_optimization(self, dataframe: pd.DataFrame, pair: str,
                                       enable_mc_optimization: bool = True) -> Dict[str, Any]:
        """
        Execute Monte Carlo optimization with caching logic.
        
        Args:
            dataframe: DataFrame with OHLCV data
            pair: Trading pair name
            enable_mc_optimization: Whether MC optimization is enabled
            
        Returns:
            Dict containing Monte Carlo results
        """
        if not enable_mc_optimization:
            return {}
        
        # Check if we need to recalculate
        should_recalc = self.cache.should_recalculate(pair)
        
        if should_recalc:
            print(f"Running Monte Carlo optimization for {pair}...")
            
            # Run the optimization
            mc_results = self.optimizer.monte_carlo_period_optimization(dataframe, pair)
            
            # Add volatility regime information
            regime, current_volatility, risk_multiplier = self.estimate_current_volatility_regime(dataframe)
            mc_results.update({
                'volatility_regime': regime,
                'current_volatility': current_volatility,
                'risk_multiplier': risk_multiplier
            })
            
            # Cache the results
            self.cache.cache_results(pair, mc_results)
            
            print(f"Monte Carlo optimization completed and cached for {pair}")
            return mc_results
        else:
            # Use cached results
            cached_results = self.cache.get_results(pair)
            if cached_results:
                print(f"Using cached Monte Carlo results for {pair}")
                return cached_results
            else:
                # No cached results available, run optimization
                print(f"No cached results found for {pair}, running optimization...")
                return self.execute_monte_carlo_optimization(dataframe, pair, enable_mc_optimization)
    
    def apply_monte_carlo_results(self, dataframe: pd.DataFrame, mc_results: Dict[str, Any]) -> None:
        """
        Apply Monte Carlo results to the dataframe.
        
        Args:
            dataframe: DataFrame to apply results to
            mc_results: Monte Carlo optimization results
        """
        if not mc_results:
            return
        
        # Apply the optimal lines to the dataframe using proper pandas assignment
        # Handle potential length mismatches when new candles appear
        if 'resistance_line' in mc_results:
            resistance_line = mc_results['resistance_line']
            if len(resistance_line) != len(dataframe):
                # Handle length mismatch - resize the array to match current dataframe length
                if len(resistance_line) < len(dataframe):
                    # Dataframe grew (new candles added) - extend the array with NaN values
                    extended_line = np.full(len(dataframe), np.nan)
                    extended_line[:len(resistance_line)] = resistance_line
                    resistance_line = extended_line
                else:
                    # Dataframe shrunk (unlikely but handle it) - truncate the array
                    resistance_line = resistance_line[:len(dataframe)]
            dataframe.loc[:, 'MC_Optimal_Resistance'] = resistance_line
            
        if 'support_line' in mc_results:
            support_line = mc_results['support_line']
            if len(support_line) != len(dataframe):
                # Handle length mismatch - resize the array to match current dataframe length
                if len(support_line) < len(dataframe):
                    # Dataframe grew (new candles added) - extend the array with NaN values
                    extended_line = np.full(len(dataframe), np.nan)
                    extended_line[:len(support_line)] = support_line
                    support_line = extended_line
                else:
                    # Dataframe shrunk (unlikely but handle it) - truncate the array
                    support_line = support_line[:len(dataframe)]
            dataframe.loc[:, 'MC_Optimal_Support'] = support_line
        
        # Apply scores using proper pandas assignment
        dataframe.loc[:, 'MC_Resistance_Score'] = mc_results.get('resistance_score', 0.0)
        dataframe.loc[:, 'MC_Support_Score'] = mc_results.get('support_score', 0.0)
        dataframe.loc[:, 'MC_Optimal_Period'] = mc_results.get('optimal_resistance_period', 0.0)
    
    # Expose cache methods for external access
    def get_pair_results(self, pair: str) -> Dict[str, Any]:
        """Get Monte Carlo results for a specific pair."""
        return self.cache.get_results(pair)
    
    def has_pair_results(self, pair: str) -> bool:
        """Check if Monte Carlo results are available for a pair."""
        return self.cache.has_results(pair)
    
    def clear_pair_cache(self, pair: str) -> None:
        """Clear Monte Carlo cache for a specific pair."""
        self.cache.clear_cache(pair)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the Monte Carlo cache."""
        return self.cache.get_cache_stats()
    
    def print_cache_summary(self) -> None:
        """Print a summary of the Monte Carlo cache status."""
        self.cache.print_cache_summary() 