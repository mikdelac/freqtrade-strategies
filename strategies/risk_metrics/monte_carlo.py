from typing import List, Tuple, Optional, Callable, Dict, Any
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import random
import time
from .volatility_models import GARCHModel

# Import trendline functions and Trendline class from trend_metrics using absolute imports
try:
    from trend_metrics.trendline import gentrends, rank_trendlines, Trendline, generate_bounce_conditions, calculate_r_squared, generate_trendlines_for_period, create_trendline_object
except ImportError:
    # Fallback for different import structures
    try:
        from strategies.trend_metrics.trendline import gentrends, rank_trendlines, Trendline, generate_bounce_conditions, calculate_r_squared, generate_trendlines_for_period, create_trendline_object
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
        def create_trendline_object(*args, **kwargs):
            return None  # Return dummy trendline object
        
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
