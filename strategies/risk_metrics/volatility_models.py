from typing import List, Tuple, Optional, Callable
import numpy as np
from enum import Enum
import pandas as pd

class VolatilityRegime(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class VolatilityModel:
    def __init__(self, 
                 low_threshold: float = 0.01,
                 medium_threshold: float = 0.015,
                 risk_multipliers: Optional[dict] = None,
                 atr_period: int = 14):
        """
        Initialize VolatilityModel with thresholds and risk multipliers.
        
        Args:
            low_threshold: Threshold for low volatility
            medium_threshold: Threshold for medium volatility
            risk_multipliers: Dict mapping regimes to risk multipliers
            atr_period: Period for ATR calculation
        """
        self.low_threshold = low_threshold
        self.medium_threshold = medium_threshold
        self.risk_multipliers = risk_multipliers or {
            VolatilityRegime.LOW: 1.0,
            VolatilityRegime.MEDIUM: 0.8,
            VolatilityRegime.HIGH: 0.5
        }
        self.atr_period = atr_period

    def get_regime_and_multiplier(self, vol: float) -> Tuple[str, float]:
        """
        Determine volatility regime and corresponding risk multiplier
        
        Args:
            vol: Volatility value
            
        Returns:
            Tuple of (regime value, risk multiplier)
        """
        regime = self.detect_regime(vol)
        return regime.value, self.risk_multipliers[regime]

    def detect_regime(self, volatility: float) -> VolatilityRegime:
        """
        Detect current volatility regime
        
        Args:
            volatility: Volatility value
        Returns:
            VolatilityRegime: Current volatility regime
        """
        if pd.isna(volatility):
            return VolatilityRegime.MEDIUM
            
        if volatility <= self.low_threshold:
            return VolatilityRegime.LOW
        elif volatility <= self.medium_threshold:
            return VolatilityRegime.MEDIUM
        return VolatilityRegime.HIGH

    def calculate_volatility(self, prices: np.ndarray) -> float:
        """Calculate historical volatility"""
        returns = np.log(prices[1:] / prices[:-1])
        return np.std(returns)
    
    def calculate_atr(self, dataframe: pd.DataFrame) -> pd.Series:
        """
        Calculate the Average True Range (ATR) for the given dataframe.
        
        Args:
            dataframe: DataFrame containing 'high', 'low', and 'close' price columns.
            
        Returns:
            pd.Series: ATR values.
        """
        try:
            import talib
            # Use TA-Lib ATR function
            atr = pd.Series(
                talib.ATR(
                    dataframe['high'].values,
                    dataframe['low'].values,
                    dataframe['close'].values,
                    timeperiod=self.atr_period
                ),
                index=dataframe.index
            )
        except ImportError:
            # Fallback to manual calculation if TA-Lib is not available
            high = dataframe['high']
            low = dataframe['low']
            close = dataframe['close']
            
            true_range = pd.DataFrame({
                'TR1': high - low,
                'TR2': np.abs(high - close.shift(1)),
                'TR3': np.abs(low - close.shift(1))
            }).max(axis=1)
            
            atr = true_range.rolling(window=self.atr_period, min_periods=1).mean()
            
        # Store current ATR value
        return atr

class GARCHModel():
    def __init__(self, 
                 omega: float = 0.000005,
                 alpha: float = 0.1,
                 beta: float = 0.85,
                 theta: float = 0.0,
                 risk_multipliers: Optional[dict] = None):
        """
        Initialize GARCHModel with specific parameters for GARCH(1,1) and NGARCH.

        Args:
            omega: Constant term in the GARCH equation.
            alpha: Coefficient for the squared return.
            beta: Coefficient for the lagged variance.
            theta: Leverage effect parameter for NGARCH.
            risk_multipliers: Dict mapping regimes to risk multipliers.
        """
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.theta = theta

        # Validate parameters to ensure stationarity
        if self.alpha + self.beta >= 1:
            raise ValueError("The sum of alpha and beta must be less than 1 for the GARCH(1,1) model to be stationary.")

        # Initialize variance with long-term variance
        self.variance = self.omega / (1 - self.alpha - self.beta)
        
    def reset_variance(self):
        """
        Reset the model's variance to its long-term value.
        
        This function resets the conditional variance of the GARCH model to its 
        unconditional (long-term) variance, which is calculated as:
        σ²_unconditional = ω / (1 - α - β)
        
        Returns:
            float: The reset variance value
        """
        self.variance = self.omega / (1 - self.alpha - self.beta)
        return self.variance

    def update_conditional_variance(self, R_t: float) -> float:
        """
        Update the conditional variance using the GARCH(1,1) model.
        σ²(t+1) = ω + α(R_t)² + βσ²_t

        Args:
            R_t: Return at time t.

        Returns:
            float: Updated variance at time t+1.
        """
        self.variance = self.omega + self.alpha * (R_t ** 2) + self.beta * self.variance
        return self.variance

    def update_conditional_variance_ngarch(self, R_t: float) -> float:
        """
        Update the conditional variance using the NGARCH model.
        σ²(t+1) = ω + α(R_t - θσ_t)² + βσ²_t

        Args:
            R_t: Return at time t.

        Returns:
            float: Updated variance at time t+1.
        """
        sigma_t = np.sqrt(self.variance)
        self.variance = self.omega + self.alpha * (R_t - self.theta * sigma_t)**2 + self.beta * self.variance
        return self.variance

    def calculate_volatility(self, prices: np.ndarray) -> float:
        """
        Calculate volatility using GARCH(1,1) model.
        First get the log returns, then iteratively update the variances using the GARCH(1,1) formula.
        Finally, return the square root of the final estimated variance (standard deviation).
        
        Args:
            prices: Array of price values.
            
        Returns:
            float: Estimated volatility (standard deviation) from GARCH model.
        """
        
        if len(prices) == 0:
            return float('nan')
                
        # Iteratively update variances using GARCH(1,1) formula
        for R_t in prices:
            self.update_conditional_variance(R_t)
        
        # Return volatility as square root of variance (standard deviation)
        return np.sqrt(self.variance)

    def monte_carlo_simulation(self, 
                      T: int, 
                      iterations: Optional[int] = 1000,
                      calculate_variance_fn: Optional[Callable[[float, float], float]] = None,
                      random_generator_fn: Optional[Callable[[float], float]] = None,
                      initial_variance: Optional[float] = None) -> np.ndarray:
        """
        Perform a Monte Carlo simulation for financial returns.

        Args:
            T: Number of time steps (days) to simulate.
            iterations: Optional number of simulation paths, defaults to 1000.
            calculate_variance_fn: Optional callback function to calculate variance with signature (sigma2_t, R_t) -> new_sigma2_t.
                                 If None, uses this model's update_conditional_variance method.
            random_generator_fn: Optional callback function to generate random returns with signature (sigma) -> return.
                               If None, uses normal distribution (sigma * N(0,1)).
            initial_variance: Optional initial variance value to use. If None, uses the model's long-term variance.

        Returns:
            np.ndarray: Simulated returns with shape (iterations, T).
        """
        # If iterations is None, use default value of 1000
        if iterations is None:
            iterations = 1000
            
        # If calculate_variance_fn is None, use the model's own method
        if calculate_variance_fn is None:
            calculate_variance_fn = lambda sigma2_t, R_t: self.update_conditional_variance(R_t)
        
        # If random_generator_fn is None, use normal distribution
        if random_generator_fn is None:
            random_generator_fn = lambda sigma: sigma * np.random.normal()
            
        # Initialize variance with long-term variance if not provided
        if initial_variance is None:
            initial_variance = self.omega / (1 - self.alpha - self.beta)
            
        # Initialize array to store simulated returns
        R = np.zeros((iterations, T))
        
        # Perform Monte Carlo simulation
        for i in range(iterations):
            sigma2_t = initial_variance  # Start with initial variance
            for t in range(T):
                # Generate return using the provided random generator function
                sigma_t = np.sqrt(sigma2_t)
                R[i, t] = random_generator_fn(sigma_t)
                
                # Update variance for next step
                sigma2_t = calculate_variance_fn(sigma2_t, R[i, t])
        
        # Calculate the sum of returns for each simulation path
        R_sum = np.zeros(iterations)
        for i in range(iterations):
            R_sum[i] = np.sum(R[i])
        return R_sum
        
    def simulate_returns(self, 
                      T: int, 
                      iterations: Optional[int] = 1000,
                      calculate_variance_fn: Optional[Callable[[float, float], float]] = None) -> np.ndarray:
        """
        Simulate returns using the GARCH(1,1) model. This is a wrapper around monte_carlo_simulation.

        Args:
            T: Number of time steps (days) to simulate.
            iterations: Optional number of simulation paths, defaults to 1000.
            calculate_variance_fn: Optional callback function to calculate variance with signature (sigma2_t, R_t) -> new_sigma2_t.
                                 If None, uses this model's update_conditional_variance method.

        Returns:
            np.ndarray: Simulated returns with shape (iterations, T).
        """
        # Use monte_carlo_simulation with default random generator (normal distribution)
        return self.monte_carlo_simulation(
            T=T,
            iterations=iterations,
            calculate_variance_fn=calculate_variance_fn
        )