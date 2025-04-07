from typing import List, Tuple, Optional
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
                 risk_multipliers: Optional[dict] = None):
        """
        Initialize GARCHModel with specific parameters for GARCH(1,1).

        Args:
            omega: Constant term in the GARCH equation.
            alpha: Coefficient for the squared return.
            beta: Coefficient for the lagged variance.
            atr_period: Period for ATR calculation.
            risk_multipliers: Dict mapping regimes to risk multipliers.
        """
        self.omega = omega
        self.alpha = alpha
        self.beta = beta

        # Validate parameters to ensure stationarity
        if self.alpha + self.beta >= 1:
            raise ValueError("The sum of alpha and beta must be less than 1 for the GARCH(1,1) model to be stationary.")

        # Initialize variance with long-term variance
        self.variance = self.omega / (1 - self.alpha - self.beta)

    def update_conditional_variance(self, R_t: float) -> float:
        """
        Update the conditional variance using the GARCH(1,1) model.

        Args:
            R_t: Return at time t.

        Returns:
            float: Updated variance at time t+1.
        """
        self.variance = self.omega + self.alpha * (R_t ** 2) + self.beta * self.variance
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
        # Calculate log returns
        returns = np.log(prices[1:] / prices[:-1])
        
        if len(returns) == 0:
            return float('nan')
        
        # Initialize variance with long-term variance
        self.variance = self.omega / (1 - self.alpha - self.beta)
        
        # Iteratively update variances using GARCH(1,1) formula
        for R_t in returns:
            self.update_conditional_variance(R_t)
        
        # Return volatility as square root of variance (standard deviation)
        return np.sqrt(self.variance)

    def simulate_returns(self, T: int, iterations: int) -> np.ndarray:
        """
        Simulate returns using the GARCH(1,1) model.

        Args:
            T: Number of time steps to simulate.
            iterations: Number of simulation paths.

        Returns:
            np.ndarray: Simulated returns.
        """
        R = np.zeros((iterations, T))
        for i in range(iterations):
            sigma2_t = self.variance  # Start with current variance
            for t in range(T):
                z_t = np.random.normal()  # Random shock
                R[i, t] = np.sqrt(sigma2_t) * z_t  # Calculate expected return at t
                sigma2_t = self.update_conditional_variance(R[i, t])
        return R

    def calculate_var_es(self, T: int = 3, iterations: int = 1000, confidence_level: float = 0.01) -> Tuple[float, float]:
        """
        Calculer la Value at Risk (VaR) et l'Expected Shortfall (ES) à un niveau de confiance spécifié sur une période donnée.
        
        Args:
            T (int): Nombre de jours pour le calcul de la VaR (ex: 3 jours).
            iterations (int): Nombre de chemins de simulation.
            confidence_level (float): Niveau de confiance pour la VaR et l'ES (ex: 0.01 pour 1%).
        
        Returns:
            Tuple[float, float]: VaR et ES aux niveaux de confiance spécifiés.
        """
        # Simuler les rendements sur T jours avec le nombre d'itérations
        simulated_returns = self.simulate_returns(T=T, iterations=iterations)
        
        # Calculer le rendement cumulé sur T jours pour chaque simulation
        R_sum = np.sum(simulated_returns, axis=1)
        
        # Calculer l'écart type de R_sum
        std_R_sum = np.std(R_sum)
        print(f"Standard deviation of cumulative returns (R_sum): {std_R_sum:.6f}")
        
        # Calculer la VaR à confidence_level (%)
        VaR = np.percentile(R_sum, confidence_level * 100)
        
        # Calculer l'ES à confidence_level %
        ES = R_sum[R_sum <= VaR].mean()
        
        return VaR, ES