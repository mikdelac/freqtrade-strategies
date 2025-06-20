# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Dict, Optional, Union, Tuple, List
from functools import reduce
from scipy.stats import norm, t
import random


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,  # @informative decorator
    # Hyperopt Parameters
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    # timeframe helpers
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    # Strategy helper functions
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)

# --------------------------------
# Add your lib to import here
from risk_metrics.volatility_models import VolatilityModel, VolatilityRegime, GARCHModel
from risk_metrics.risk_indicators import RiskIndicators
from risk_metrics.monte_carlo import MonteCarloSimulator
from trend_metrics.trend_analysis import TrendAnalysis
from trend_metrics.trendline import gentrends, segtrends, rank_trendlines
from technical.util import resample_to_interval, resampled_merge

class RiskMetrics(IStrategy):
    """
    RiskMetrics strategy using HAR-RV (Heterogeneous Autoregression Realized Volatility) model
    for volatility forecasting and risk management.
    
    Features:
    - Volatility forecasting using HAR-RV model
    - Risk-adjusted position sizing based on volatility regime
    - Trendline analysis with support and resistance identification
    - Trendline ranking based on price proximity and touch frequency
    - Monte Carlo optimization for optimal trendline periods
    - Linear regression trendlines using TA-Lib's LINEARREG functions
      * Provides straight-line trendlines using the least squares method
      * Visualizes slope, angle, and projected forecasts
      * Useful for identifying short to medium-term trends
    - Dynamic timeframe selection based on available data
      * Automatically selects the highest appropriate timeframe
      * Adapts analysis based on available historical data length
    - Monte Carlo Score Convergence Detection and Risk Management
      * Detects when support and resistance scores are similar
      * Implements adaptive position sizing during convergence periods
      * Switches between bounce trading and breakout modes
    """
    INTERFACE_VERSION = 3

    # Timeframe settings
    timeframe = "5m"
    MINUTES_IN_DAY = 24 * 60
    MINUTES_PER_CANDLE = 5
    CANDLES_PER_DAY = MINUTES_IN_DAY // MINUTES_PER_CANDLE  # 288 5-min candles per day
    TRADING_DAYS_PER_YEAR = 252
    WEEKS_PER_MONTH = 4.33
    TRADING_DAYS_PER_WEEK = 5
    TRADING_DAYS_PER_MONTH = 22
    HOURS_PER_DAY = 24
    WEEKS_PER_YEAR = 52
    MONTHS_PER_YEAR = 12
    
    # Volatility calculation constants
    DAILY_CANDLES = CANDLES_PER_DAY  # Target: 288 candles
    WEEKLY_CANDLES = CANDLES_PER_DAY * TRADING_DAYS_PER_WEEK  # Target: 1440 candles
    MONTHLY_CANDLES = CANDLES_PER_DAY * TRADING_DAYS_PER_MONTH  # Target: 6336 candles
    ONE_HOUR_CANDLES = 12  # 12 candles = 60 minutes
    
    # Monte Carlo period optimization settings
    MC_ITERATIONS = 1000
    MIN_LOOKBACK_PERIOD = 50
    # MAX_LOOKBACK_PERIOD will be set dynamically based on available data
    
    # Timeframe thresholds for resampling
    # Minimum number of candles needed for each timeframe
    TIMEFRAME_THRESHOLDS = {
        '1d': CANDLES_PER_DAY,           # Need at least 1 day of data
        '3d': CANDLES_PER_DAY * 3,       # Need at least 3 days of data
        '1w': WEEKLY_CANDLES,            # Need at least 1 week of data
        '1M': MONTHLY_CANDLES,           # Need at least 1 month of data
    }
    
    # Supported higher timeframes in order of preference (highest first)
    HIGHER_TIMEFRAMES = ['1M', '1w', '3d', '1d']
    
    # Risk thresholds - Note: These thresholds are now in terms of non-annualized volatility
    LOW_VOL_THRESHOLD = 0.01
    MEDIUM_VOL_THRESHOLD = 0.015
    
    # RSI settings
    RSI_PERIOD = 14
    
    # Trading parameters
    can_short: bool = False
    
    # Risk parameters
    risk_reduction_high = DecimalParameter(0.3, 0.7, default=0.5, space="buy", optimize=True)
    risk_reduction_medium = DecimalParameter(0.6, 0.9, default=0.8, space="buy", optimize=True)
    high_vol_threshold_1h = DecimalParameter(0.01, 0.05, default=0.02, space="buy", optimize=True)
    rv_1h_change_threshold = DecimalParameter(0.05, 0.10, default=0.01, space="buy", optimize=True)
    
    # Trendline parameters
    trendline_proximity_threshold = DecimalParameter(0.005, 0.02, default=0.002, space="buy", optimize=True)
    trendline_touch_weight = DecimalParameter(1.5, 3.0, default=2.5, space="buy", optimize=True)
    
    # Linear Regression parameters
    linearreg_timeperiod = IntParameter(10, 500, default=200, space="buy", optimize=True)
    linearreg_price_field = CategoricalParameter(['close', 'open', 'high', 'low'], default='close', space="buy", optimize=False)

    # Monte Carlo optimization parameters
    enable_mc_optimization = BooleanParameter(default=True, space="buy", optimize=False)

    # GARCH Volatility-based period sampling parameters
    garch_min_candles = IntParameter(100, 300, default=100, space="buy", optimize=True)
    garch_max_candles = IntParameter(500, 1000, default=1000, space="buy", optimize=True)
    
    # Volatility regime sampling distribution parameters (Beta distribution)
    # Low volatility regime: favor longer lookback periods (small alpha, large beta pushes toward end of range)
    low_vol_beta_a = DecimalParameter(1.0, 2.0, default=1.0, space="buy", optimize=True)
    low_vol_beta_b = DecimalParameter(2.0, 4.0, default=3.0, space="buy", optimize=True)
    
    # Medium volatility regime: balanced distribution
    medium_vol_beta_a = DecimalParameter(0.8, 1.2, default=1.0, space="buy", optimize=True)
    medium_vol_beta_b = DecimalParameter(0.8, 1.2, default=1.0, space="buy", optimize=True)
    
    # High volatility regime: favor shorter lookback periods (large alpha, small beta pushes toward start of range)
    high_vol_beta_a = DecimalParameter(2.0, 4.0, default=3.0, space="buy", optimize=True)
    high_vol_beta_b = DecimalParameter(1.0, 2.0, default=1.5, space="buy", optimize=True)
    
    # Enable volatility-based period sampling
    enable_volatility_sampling = BooleanParameter(default=True, space="buy", optimize=False)

    # Monte Carlo Score Convergence Parameters
    score_convergence_high_threshold = DecimalParameter(0.85, 0.95, default=0.90, space="buy", optimize=True)
    score_convergence_medium_threshold = DecimalParameter(0.70, 0.85, default=0.80, space="buy", optimize=True)
    score_convergence_low_threshold = DecimalParameter(0.50, 0.70, default=0.60, space="buy", optimize=True)
    
    # Convergence risk multipliers
    convergence_high_penalty = DecimalParameter(0.1, 0.3, default=0.2, space="buy", optimize=True)
    convergence_medium_penalty = DecimalParameter(0.4, 0.6, default=0.5, space="buy", optimize=True)
    convergence_low_penalty = DecimalParameter(0.7, 0.9, default=0.8, space="buy", optimize=True)
    
    # Enable convergence detection
    enable_convergence_detection = BooleanParameter(default=True, space="buy", optimize=False)

    # Minimal ROI designed for the strategy.
    minimal_roi = {
        "360": 0.15,  # Exit after 6 hours if profit is 15%
        "240": 0.10,  # Exit after 4 hours if profit is 10%
        "120": 0.07,  # Exit after 2 hours if profit is 7%
        "60": 0.05,   # Exit after 1 hour if profit is 5%
        "30": 0.03,   # Exit after 30 min if profit is 3%
        "0": 0.02     # Exit immediately if profit is 2%
    }

    # Disable stoploss since we're using ROI-based exits only
    stoploss = -0.1  # Effectively disabled
    trailing_stop = False
    use_exit_signal = False  # Disable exit signals since we're using ROI
    exit_profit_only = False  # Only exit in profit
    ignore_roi_if_entry_signal = False  # Don't ignore ROI even if we have a new entry signal

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

    # Order settings
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": True
    }

    # Optional order time in force.
    order_time_in_force = {
        "entry": "GTC",
        "exit": "GTC"
    }

    @property
    def plot_config(self):
        # Basic configuration with default plots
        plot_config = {
            "main_plot": {
                "all_highs": {"color": "red", "type": "scatter", "symbol": "triangle-down", "size": 12, "fillcolor": "red"},
                "all_lows": {"color": "green", "type": "scatter", "symbol": "triangle-up", "size": 12, "fillcolor": "green"},
                "Resistance Line": {"color": "red", "width": 2.0},
                "Support Line": {"color": "green", "width": 2.0},
                "Highest_Scored_Line": {"color": "purple", "width": 3.0},
                "linear_reg_line": {"color": "blue", "width": 3.0},
                "MC_Optimal_Resistance": {"color": "darkred", "width": 4.0, "dash": "dot"},
                "MC_Optimal_Support": {"color": "darkgreen", "width": 4.0, "dash": "dot"}
            },
            "subplots": {
                "ATR": {
                    "atr": {"color": "blue"}
                },
                "Mean Scores": {
                    "Resistance_Mean_Score": {"color": "red", "type": "line", "width": 2.0},
                    "Support_Mean_Score": {"color": "green", "type": "line", "width": 2.0},
                    "Highest_Line_Score": {"color": "purple", "type": "line", "width": 2.5}
                },
                "Total Line Scores": {
                    "Resistance_Line_Score": {"color": "red", "type": "line", "width": 2.0},
                    "Support_Line_Score": {"color": "green", "type": "line", "width": 2.0}
                },
                "Monte Carlo Optimization": {
                    "MC_Resistance_Score": {"color": "darkred", "type": "line", "width": 3.0},
                    "MC_Support_Score": {"color": "darkgreen", "type": "line", "width": 3.0},
                    "MC_Optimal_Period": {"color": "orange", "type": "line", "width": 1.5}
                },
                "Score Convergence Analysis": {
                    "score_convergence_ratio": {"color": "purple", "type": "line", "width": 2.0},
                    "convergence_multiplier": {"color": "orange", "type": "line", "width": 2.0}
                },
                "Trading Mode": {
                    "trading_mode_indicator": {"color": "blue", "type": "line", "width": 2.0}
                },
                "Timeframes": {
                    "highest_timeframe_indicator": {"color": "blue", "type": "line", "width": 2.0}
                },
                "Risk Metrics": {
                    "risk_multiplier": {"color": "orange", "type": "line", "width": 2.0}
                }
            }
        }
        
        # Add volatility regime subplot if enabled
        if hasattr(self, 'enable_volatility_sampling') and self.enable_volatility_sampling.value:
            plot_config["subplots"]["Volatility Analysis"] = {
                "volatility": {"color": "purple", "type": "line", "width": 2.0},
                "risk_multiplier": {"color": "orange", "type": "line", "width": 2.0}
            }
        
        # Dynamically add higher timeframe plots based on available timeframes
        # These won't be added until determine_highest_timeframe is called
        if hasattr(self, 'highest_timeframe') and self.highest_timeframe != self.timeframe:
            suffix = f"_{self.highest_timeframe}"
            
            # Add higher timeframe volatility to subplots
            if "Risk Metrics" not in plot_config["subplots"]:
                plot_config["subplots"]["Risk Metrics"] = {}
                
            plot_config["subplots"]["Risk Metrics"][f"rv{suffix}"] = {
                "color": "red", 
                "type": "line", 
                "width": 2.0
            }
            
            # Add higher timeframe price data to main plot with semi-transparent color
            for field in ['open', 'high', 'low', 'close']:
                field_name = f"{field}{suffix}"
                plot_config["main_plot"][field_name] = {
                    "color": "rgba(100, 100, 255, 0.3)",  # Semi-transparent blue
                    "width": 1.0
                }
        
        return plot_config
    
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        # Initialize volatility model with thresholds and risk multipliers
        self.volatility_model = VolatilityModel(
            low_threshold=self.LOW_VOL_THRESHOLD,
            medium_threshold=self.MEDIUM_VOL_THRESHOLD,
            risk_multipliers={
                VolatilityRegime.LOW: 1.0,
                VolatilityRegime.MEDIUM: self.risk_reduction_medium.value,
                VolatilityRegime.HIGH: self.risk_reduction_high.value
            }
        )
        self.trend_analyzer = TrendAnalysis(
            min_points=2,  # Reduced minimum points
            min_slope=0.00001,  # Reduced minimum slope
            min_strength=0.2,  # Reduced strength requirement
            angle_threshold=90  # Increased angle threshold
        )
        self.har_model = None
        self.last_fit = None
        # Initialize highest timeframe as None - will be determined dynamically
        self.highest_timeframe = None
        self.available_timeframes = []
        
        # Initialize GARCH model for volatility-based period optimization
        self.garch_model = GARCHModel()
        
        print(f"RiskMetrics strategy initialized with volatility-based period sampling:")
        print(f"  GARCH candles range: {self.garch_min_candles.value} to {self.garch_max_candles.value}")
        print(f"  Volatility sampling enabled: {self.enable_volatility_sampling.value}")
        print(f"  Low vol Beta params: ({self.low_vol_beta_a.value:.2f}, {self.low_vol_beta_b.value:.2f})")
        print(f"  Medium vol Beta params: ({self.medium_vol_beta_a.value:.2f}, {self.medium_vol_beta_b.value:.2f})")
        print(f"  High vol Beta params: ({self.high_vol_beta_a.value:.2f}, {self.high_vol_beta_b.value:.2f})")
        print(f"  Score convergence detection enabled: {self.enable_convergence_detection.value}")
        print(f"  Convergence thresholds: High={self.score_convergence_high_threshold.value:.2f}, Medium={self.score_convergence_medium_threshold.value:.2f}, Low={self.score_convergence_low_threshold.value:.2f}")

    def calculate_score_convergence_ratio(self, support_score: float, resistance_score: float) -> float:
        """
        Calculate the convergence ratio between MC support and resistance scores.
        Returns the ratio of the smaller score to the larger score (0.0 to 1.0).
        
        Args:
            support_score: MC_Support_Score value
            resistance_score: MC_Resistance_Score value
            
        Returns:
            float: Convergence ratio (0.0 = completely different, 1.0 = identical)
        """
        if support_score <= 0 or resistance_score <= 0:
            return 0.0
        
        # Calculate the ratio of smaller to larger score
        min_score = min(support_score, resistance_score)
        max_score = max(support_score, resistance_score)
        
        return min_score / max_score

    def get_convergence_multiplier(self, convergence_ratio: float) -> float:
        """
        Calculate position size multiplier based on score convergence ratio.
        Higher convergence = lower position size due to increased uncertainty.
        
        Args:
            convergence_ratio: Score convergence ratio (0.0 to 1.0)
            
        Returns:
            float: Position size multiplier (0.0 to 1.0)
        """
        if not self.enable_convergence_detection.value:
            return 1.0
        
        if convergence_ratio >= self.score_convergence_high_threshold.value:
            # High convergence - significant risk reduction
            return self.convergence_high_penalty.value
        elif convergence_ratio >= self.score_convergence_medium_threshold.value:
            # Medium convergence - moderate risk reduction
            return self.convergence_medium_penalty.value
        elif convergence_ratio >= self.score_convergence_low_threshold.value:
            # Low convergence - slight risk reduction
            return self.convergence_low_penalty.value
        
        # No significant convergence - no penalty
        return 1.0

    def get_trading_mode(self, convergence_ratio: float) -> str:
        """
        Determine current trading mode based on score convergence ratio.
        
        Args:
            convergence_ratio: Score convergence ratio (0.0 to 1.0)
            
        Returns:
            str: Trading mode identifier
        """
        if not self.enable_convergence_detection.value:
            return "NORMAL_BOUNCE"
        
        if convergence_ratio >= self.score_convergence_high_threshold.value:
            return "HIGH_CONVERGENCE"    # Avoid trading or prepare for breakout
        elif convergence_ratio >= self.score_convergence_medium_threshold.value:
            return "MEDIUM_CONVERGENCE"  # Reduced size + breakout watch
        elif convergence_ratio >= self.score_convergence_low_threshold.value:
            return "LOW_CONVERGENCE"     # Slight caution
        
        return "NORMAL_BOUNCE"  # Standard bounce trading

    def should_enter_trade_with_convergence(self, support_score: float, resistance_score: float) -> Tuple[bool, str]:
        """
        Enhanced entry logic considering score convergence.
        
        Args:
            support_score: MC_Support_Score value
            resistance_score: MC_Resistance_Score value
            
        Returns:
            Tuple[bool, str]: (should_enter, reason)
        """
        if not self.enable_convergence_detection.value:
            return True, "CONVERGENCE_DISABLED"
        
        convergence_ratio = self.calculate_score_convergence_ratio(support_score, resistance_score)
        trading_mode = self.get_trading_mode(convergence_ratio)
        
        if trading_mode == "HIGH_CONVERGENCE":
            return False, f"HIGH_CONVERGENCE_DETECTED ({convergence_ratio:.3f})"
        elif trading_mode == "MEDIUM_CONVERGENCE":
            return True, f"MEDIUM_CONVERGENCE_CAUTION ({convergence_ratio:.3f})"
        elif trading_mode == "LOW_CONVERGENCE":
            return True, f"LOW_CONVERGENCE_SLIGHT_CAUTION ({convergence_ratio:.3f})"
        
        return True, f"NORMAL_BOUNCE_MODE ({convergence_ratio:.3f})"

    def detect_breakout_scenario(self, dataframe: DataFrame) -> Dict[str, any]:
        """
        Detect when market is in breakout mode due to score convergence.
        
        Args:
            dataframe: DataFrame with MC scores
            
        Returns:
            Dict containing breakout analysis
        """
        if len(dataframe) == 0:
            return {"mode": "INSUFFICIENT_DATA", "ratio": 0.0, "multiplier": 1.0}
        
        current_candle = dataframe.iloc[-1]
        support_score = current_candle.get('MC_Support_Score', 0)
        resistance_score = current_candle.get('MC_Resistance_Score', 0)
        
        convergence_ratio = self.calculate_score_convergence_ratio(support_score, resistance_score)
        convergence_multiplier = self.get_convergence_multiplier(convergence_ratio)
        trading_mode = self.get_trading_mode(convergence_ratio)
        
        return {
            "mode": trading_mode,
            "ratio": convergence_ratio,
            "multiplier": convergence_multiplier,
            "support_score": support_score,
            "resistance_score": resistance_score,
            "recommendation": self._get_trading_recommendation(trading_mode, convergence_ratio)
        }

    def _get_trading_recommendation(self, trading_mode: str, convergence_ratio: float) -> str:
        """
        Get trading recommendation based on convergence analysis.
        
        Args:
            trading_mode: Current trading mode
            convergence_ratio: Score convergence ratio
            
        Returns:
            str: Trading recommendation
        """
        recommendations = {
            "HIGH_CONVERGENCE": f"AVOID bounce trades. Scores too similar ({convergence_ratio:.3f}). Wait for breakout or clear divergence.",
            "MEDIUM_CONVERGENCE": f"CAUTION: Reduced position size. Monitor for breakout signals. Convergence ratio: {convergence_ratio:.3f}",
            "LOW_CONVERGENCE": f"SLIGHT CAUTION: Minor convergence detected ({convergence_ratio:.3f}). Normal trading with reduced risk.",
            "NORMAL_BOUNCE": f"NORMAL bounce trading conditions. Clear score differentiation ({convergence_ratio:.3f})."
        }
        
        return recommendations.get(trading_mode, f"Unknown mode: {trading_mode}")

    def determine_highest_timeframe(self, dataframe: DataFrame) -> str:
        """
        Determine the highest possible timeframe based on available data length.
        
        Args:
            dataframe: DataFrame with current timeframe's OHLCV data
            
        Returns:
            str: The highest timeframe that can be used ('1M', '1w', '3d', '1d' or base timeframe)
        """
        data_length = len(dataframe)
        returned_tf = []

        # Log available data
        print(f"Available data: {data_length} candles at {self.timeframe} timeframe")
                
        # Check each higher timeframe from highest to lowest
        for tf in self.HIGHER_TIMEFRAMES:
            threshold = self.TIMEFRAME_THRESHOLDS.get(tf, 0)
            print(f"Threshold for {tf}: {threshold}")
            if data_length >= threshold:
                # Add to available timeframes
                self.available_timeframes.append(tf)
                
                if returned_tf == []:
                    # Return the highest timeframe (first one that matches)
                    returned_tf = tf
        
        if returned_tf == []:
            # If no higher timeframe has enough data, return the base timeframe
            print(f"Not enough data for higher timeframes. Using base timeframe: {self.timeframe}")
            return self.timeframe
        else:
            return returned_tf

    def resample_to_higher_timeframes(self, dataframe: DataFrame) -> Dict[str, DataFrame]:
        """
        Resample the dataframe to all available higher timeframes.
        
        Args:
            dataframe: DataFrame with current timeframe's OHLCV data
            
        Returns:
            Dict[str, DataFrame]: Dictionary of resampled dataframes keyed by timeframe
        """
        resampled_dfs = {}
        
        # Skip if no data
        if len(dataframe) == 0:
            return resampled_dfs
        
        print(f"Available timeframes: {self.available_timeframes}")
        # For each available higher timeframe, resample the data
        for tf in self.available_timeframes:
            if tf == self.timeframe:
                continue  # Skip the base timeframe
                
            try:
                # Use resample_to_interval to convert to higher timeframe
                resampled = resample_to_interval(dataframe, self.TIMEFRAME_THRESHOLDS[tf])
                
                # Add to our dictionary
                resampled_dfs[tf] = resampled
                
                print(f"Successfully resampled to {tf} timeframe: {len(resampled)} candles")
            except Exception as e:
                print(f"Error resampling to {tf}: {e}")
        
        return resampled_dfs

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        # We'll dynamically determine the informative pairs based on the highest timeframe
        # This is done at runtime in populate_indicators
        return []

    def _group_by_day(self, timestamps: pd.Series) -> pd.Series:
        """
        Group timestamps by trading day
        
        Args:
            timestamps: Series of timestamps
            
        Returns:
            Series with day grouping
        """
        return pd.to_datetime(timestamps).dt.date

    # Define scaling factors for different frequencies
    SCALING_FACTORS = {
        'daily': TRADING_DAYS_PER_YEAR,
        'weekly': WEEKS_PER_YEAR,
        'monthly': MONTHS_PER_YEAR
    }

    def generate_volatility_based_lookback_periods(self, dataframe: DataFrame) -> List[int]:
        """
        Generate lookback periods based on current volatility regime using GARCH model
        and Beta distribution sampling.
        
        Args:
            dataframe: DataFrame with OHLCV data
            
        Returns:
            List[int]: List of lookback periods optimized for current volatility regime
        """
        if not self.enable_volatility_sampling.value:
            # Fall back to uniform random sampling if volatility sampling is disabled
            total_candles = len(dataframe)
            max_lookback_period = min(total_candles, self.garch_max_candles.value)
            return [random.randint(self.MIN_LOOKBACK_PERIOD, max_lookback_period) 
                   for _ in range(self.MC_ITERATIONS)]
        
        # Calculate log returns for GARCH volatility estimation
        log_returns = np.log(dataframe['close'].pct_change() + 1).dropna().values
        
        # Determine number of candles for volatility calculation based on regime
        volatility_candles = min(len(log_returns), self.garch_max_candles.value)
        volatility_candles = max(volatility_candles, self.garch_min_candles.value)
        
        # Calculate current volatility using GARCH model
        recent_returns = log_returns[-volatility_candles:]
        
        # Clean data and calculate volatility
        if len(recent_returns) > 0 and not np.isnan(recent_returns).all() and not np.isinf(recent_returns).any():
            # Use VolatilityModel's calculate_volatility method with GARCHModel
            current_volatility = self.volatility_model.calculate_volatility(recent_returns, self.garch_model)
            print(f"  GARCH volatility result: {current_volatility}")
        else:
            print(f"  Invalid data for GARCH calculation, using fallback")
            # Fallback to simple standard deviation using VolatilityModel
            current_volatility = self.volatility_model.calculate_volatility(recent_returns)
            
        regime, risk_multiplier = self.volatility_model.get_regime_and_multiplier(current_volatility)
        
        print(f"Current volatility regime: {regime} (volatility: {current_volatility:.6f}, risk multiplier: {risk_multiplier:.2f})")
        
        # Set Beta distribution parameters based on volatility regime
        if regime == 'low':
            a, b = self.low_vol_beta_a.value, self.low_vol_beta_b.value
            print(f"Using low volatility sampling: Beta({a:.2f}, {b:.2f}) - favors longer periods")
        elif regime == 'medium':
            a, b = self.medium_vol_beta_a.value, self.medium_vol_beta_b.value
            print(f"Using medium volatility sampling: Beta({a:.2f}, {b:.2f}) - balanced distribution")
        else:  # high volatility
            a, b = self.high_vol_beta_a.value, self.high_vol_beta_b.value
            print(f"Using high volatility sampling: Beta({a:.2f}, {b:.2f}) - favors shorter periods")
        
        # Generate Beta-distributed samples
        samples = np.random.beta(a, b, self.MC_ITERATIONS)
        
        # Scale samples to lookback period range
        total_candles = len(dataframe)
        max_lookback_period = min(total_candles, self.garch_max_candles.value)
        min_lookback_period = max(self.MIN_LOOKBACK_PERIOD, self.garch_min_candles.value)
        
        scaled_samples = samples * (max_lookback_period - min_lookback_period) + min_lookback_period
        lookback_periods = np.round(scaled_samples).astype(int).tolist()
        
        # Ensure all periods are within valid range
        lookback_periods = [max(min_lookback_period, min(p, max_lookback_period)) for p in lookback_periods]
        
        # Add some fixed key periods for comprehensive coverage
        fixed_periods = [50, 100, 200, 300, 500]
        fixed_periods = [p for p in fixed_periods if min_lookback_period <= p <= max_lookback_period]
        
        # Replace some random periods with fixed periods (up to 20% of total iterations)
        num_fixed = min(len(fixed_periods), self.MC_ITERATIONS // 5)
        if num_fixed > 0:
            # Replace the first num_fixed periods with fixed periods
            lookback_periods[:num_fixed] = fixed_periods[:num_fixed]
        
        # Shuffle to randomize the order
        random.shuffle(lookback_periods)
        
        print(f"Generated {len(lookback_periods)} lookback periods:")
        print(f"  Range: {min(lookback_periods)} to {max(lookback_periods)} candles")
        print(f"  Mean: {np.mean(lookback_periods):.1f}, Std: {np.std(lookback_periods):.1f}")
        print(f"  Fixed periods included: {fixed_periods[:num_fixed] if num_fixed > 0 else 'None'}")
        
        return lookback_periods

    def monte_carlo_period_optimization(self, dataframe: DataFrame) -> Dict[str, any]:
        """
        Use Monte Carlo simulation to test different lookback periods and find
        the ones that produce the highest scoring resistance and support lines.
        Now uses volatility-based period sampling for improved optimization.
        
        Args:
            dataframe: DataFrame with OHLCV data
            
        Returns:
            Dict containing optimal periods and their scores
        """
        # Set MAX_LOOKBACK_PERIOD dynamically based on available data and GARCH parameters
        total_candles = len(dataframe)
        max_lookback_period = min(total_candles, self.garch_max_candles.value)
        
        if total_candles < self.MIN_LOOKBACK_PERIOD:
            print(f"Not enough data for Monte Carlo optimization. Need at least {self.MIN_LOOKBACK_PERIOD} candles, got {total_candles}.")
            return {
                'optimal_resistance_period': self.MIN_LOOKBACK_PERIOD,
                'optimal_support_period': self.MIN_LOOKBACK_PERIOD,
                'resistance_score': 0.0,
                'support_score': 0.0,
                'resistance_line': np.full(len(dataframe), np.nan),
                'support_line': np.full(len(dataframe), np.nan)
            }
        
        # Seed random number generator for reproducible results with some variability
        import time
        random.seed(int(time.time() * 1000) % 10000)  # Use current time for seed
        
        # Generate volatility-based lookback periods
        print("=== Volatility-Based Monte Carlo Period Optimization ===")
        lookback_periods = self.generate_volatility_based_lookback_periods(dataframe)
        
        print(f"Starting Monte Carlo period optimization with {len(lookback_periods)} iterations...")
        print(f"Testing periods from {min(lookback_periods)} to {max(lookback_periods)} candles (total data: {total_candles})")
        
        best_resistance_score = 0.0
        best_support_score = 0.0
        best_resistance_period = self.MIN_LOOKBACK_PERIOD
        best_support_period = self.MIN_LOOKBACK_PERIOD
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
                
                # Find swing points for this period
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
                    threshold=self.trendline_proximity_threshold.value,
                    touch_weight=self.trendline_touch_weight.value,
                    all_highs=recent_data['all_highs'],
                    all_lows=recent_data['all_lows'],
                    pivot_bonus=10.0  # Higher bonus for MC optimization
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
                    
                    print(f"New best resistance found at iteration {iteration + 1}: period {random_period}, score {current_resistance_score:.4f}")
                
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
                    
                    print(f"New best support found at iteration {iteration + 1}: period {random_period}, score {current_support_score:.4f}")
                
                # Progress reporting with more details
                if (iteration + 1) % 100 == 0:
                    print(f"Monte Carlo progress: {iteration + 1}/{len(lookback_periods)} iterations completed")
                    print(f"Current best - Resistance: {best_resistance_score:.4f} (period {best_resistance_period}), Support: {best_support_score:.4f} (period {best_support_period})")
                    print(f"Last 5 tested periods: {tested_periods[-5:] if len(tested_periods) >= 5 else tested_periods}")
                
            except Exception as e:
                print(f"Error in Monte Carlo iteration {iteration}: {e}")
                continue
        
        print(f"Volatility-based Monte Carlo optimization completed!")
        print(f"Optimal resistance period: {best_resistance_period} (score: {best_resistance_score:.4f})")
        print(f"Optimal support period: {best_support_period} (score: {best_support_score:.4f})")
        print(f"Tested periods range: {min(tested_periods) if tested_periods else 'N/A'} to {max(tested_periods) if tested_periods else 'N/A'} candles")
        
        # Print period distribution for debugging
        print("Period distribution (grouped by hundreds):")
        for period_range in sorted(period_counts.keys()):
            print(f"  {period_range}-{period_range+99}: {period_counts[period_range]} times")
        
        # Print some statistics about the tested periods
        if tested_periods:
            print(f"Period statistics - Min: {min(tested_periods)}, Max: {max(tested_periods)}, Mean: {np.mean(tested_periods):.1f}")
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
            'volatility_regime': self.volatility_model.get_regime_and_multiplier(
                self.volatility_model.calculate_volatility(
                    np.log(dataframe['close'].pct_change() + 1).dropna().values[-self.garch_max_candles.value:],
                    self.garch_model
                ) if len(dataframe) > self.garch_min_candles.value else np.nan
            )[0] if len(dataframe) > self.garch_min_candles.value else 'insufficient_data'
        }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame, including ATR.
        """
        if len(dataframe) == 0:
            return dataframe
            
        # Determine the highest timeframe we can use based on available data
        self.highest_timeframe = self.determine_highest_timeframe(dataframe)
        
        # Add an indicator showing which highest timeframe was selected
        dataframe['highest_timeframe_indicator'] = 1.0
        dataframe['highest_timeframe'] = self.highest_timeframe
        print(f"Highest timeframe: {self.highest_timeframe}")
        # Resample to higher timeframes if possible
        resampled_dfs = self.resample_to_higher_timeframes(dataframe)
        print(f"Resampled dataframes: {resampled_dfs}")

        # Calculate ATR and store it for visualization
        dataframe['atr'] = self.volatility_model.calculate_atr(dataframe)

        # Calculate and store volatility regime information
        if len(dataframe) >= self.garch_min_candles.value:
            # Calculate log returns for volatility estimation
            log_returns = np.log(dataframe['close'].pct_change() + 1).dropna().values
            
            # Use appropriate number of candles for volatility calculation
            volatility_candles = min(len(log_returns), self.garch_max_candles.value)
            volatility_candles = max(volatility_candles, self.garch_min_candles.value)
            
            # Calculate current volatility and regime
            recent_returns = log_returns[-volatility_candles:]
            
            # Debug information
            print(f"Volatility calculation debug:")
            print(f"  Total log returns: {len(log_returns)}")
            print(f"  Volatility candles used: {volatility_candles}")
            print(f"  Recent returns shape: {recent_returns.shape}")
            print(f"  Recent returns stats: mean={np.mean(recent_returns):.6f}, std={np.std(recent_returns):.6f}")
            print(f"  Recent returns has NaN: {np.isnan(recent_returns).any()}")
            print(f"  Recent returns has Inf: {np.isinf(recent_returns).any()}")
            
            # Clean data and calculate volatility
            if len(recent_returns) > 0 and not np.isnan(recent_returns).all() and not np.isinf(recent_returns).any():
                # Use VolatilityModel's calculate_volatility method with GARCHModel
                current_volatility = self.volatility_model.calculate_volatility(recent_returns, self.garch_model)
                print(f"  GARCH volatility result: {current_volatility}")
            else:
                print(f"  Invalid data for GARCH calculation, using fallback")
                # Fallback to simple standard deviation using VolatilityModel
                current_volatility = self.volatility_model.calculate_volatility(recent_returns)
            
            # Get regime and multiplier
            regime, risk_multiplier = self.volatility_model.get_regime_and_multiplier(current_volatility)
            
            # Store volatility information in dataframe
            dataframe['volatility'] = current_volatility
            dataframe['volatility_regime'] = regime
            dataframe['risk_multiplier'] = risk_multiplier
            
            print(f"Current volatility regime: {regime} (volatility: {current_volatility:.6f}, risk multiplier: {risk_multiplier:.2f})")
        else:
            # Insufficient data for volatility calculation
            dataframe['volatility'] = np.nan
            dataframe['volatility_regime'] = 'insufficient_data'
            dataframe['risk_multiplier'] = 1.0
            print(f"Insufficient data for volatility calculation. Need at least {self.garch_min_candles.value} candles, got {len(dataframe)}")

        # Initialize marker columns for support and resistance points
        dataframe['all_highs'] = np.nan
        dataframe['all_lows'] = np.nan
        
        # Initialize columns for main trend lines
        dataframe['Resistance Line'] = np.nan
        dataframe['Support Line'] = np.nan
        
        # Initialize Monte Carlo optimal lines
        dataframe['MC_Optimal_Resistance'] = np.nan
        dataframe['MC_Optimal_Support'] = np.nan
        dataframe['MC_Resistance_Score'] = 0.0
        dataframe['MC_Support_Score'] = 0.0
        dataframe['MC_Optimal_Period'] = 0.0
        
        # Initialize new columns for highest scored line
        dataframe['Highest_Scored_Line'] = np.nan
        dataframe['Highest_Set_Mean'] = np.nan
        dataframe['Highest_Line_Score'] = np.nan
        dataframe['Highest_Line_Type'] = ""  # Will be "Resistance" or "Support"
        dataframe['Highest_Line_Text'] = ""  # For displaying text on the chart
        
        # Define the maximum number of segments we'll use
        max_segments = 10
        
        # Initialize columns for all individual segment trend lines
        for i in range(max_segments):
            dataframe[f'Max_Line_{i}'] = np.nan
            dataframe[f'Min_Line_{i}'] = np.nan

        # Monte Carlo Period Optimization
        if self.enable_mc_optimization.value and len(dataframe) >= self.MIN_LOOKBACK_PERIOD:
            print("=== Monte Carlo Period Optimization ===")
            mc_results = self.monte_carlo_period_optimization(dataframe)
            
            # Store the optimal lines
            dataframe['MC_Optimal_Resistance'] = mc_results['resistance_line']
            dataframe['MC_Optimal_Support'] = mc_results['support_line']
            
            # Get the raw scores
            raw_resistance_score = mc_results['resistance_score']
            raw_support_score = mc_results['support_score']
                                    
            # Store the raw scores so they show up on the graph
            dataframe['MC_Resistance_Score'] = raw_resistance_score
            dataframe['MC_Support_Score'] = raw_support_score
            dataframe['MC_Optimal_Period'] = max(mc_results['optimal_resistance_period'], 
                                                mc_results['optimal_support_period'])
            
            print(f"Monte Carlo results stored in dataframe")
            print(f"Max lookback period used: {mc_results.get('max_lookback_period', 'N/A')} candles")
            print(f"MC Resistance Score: {raw_resistance_score:.6f} (raw)")
            print(f"MC Support Score: {raw_support_score:.6f} (raw)")
            
            # Additional debug info about the dataframe columns
            print(f"Dataframe shape: {dataframe.shape}")
            print(f"MC_Resistance_Score column stats: min={dataframe['MC_Resistance_Score'].min()}, max={dataframe['MC_Resistance_Score'].max()}")
            print(f"MC_Support_Score column stats: min={dataframe['MC_Support_Score'].min()}, max={dataframe['MC_Support_Score'].max()}")
            
            # === Monte Carlo Score Convergence Analysis ===
            print("=== Monte Carlo Score Convergence Analysis ===")
            
            # Calculate convergence ratio between MC scores
            convergence_ratio = self.calculate_score_convergence_ratio(raw_support_score, raw_resistance_score)
            convergence_multiplier = self.get_convergence_multiplier(convergence_ratio)
            trading_mode = self.get_trading_mode(convergence_ratio)
            
            # Convert trading mode to numeric indicator for plotting
            trading_mode_mapping = {
                "NORMAL_BOUNCE": 1.0,
                "LOW_CONVERGENCE": 2.0,
                "MEDIUM_CONVERGENCE": 3.0,
                "HIGH_CONVERGENCE": 4.0
            }
            trading_mode_indicator = trading_mode_mapping.get(trading_mode, 0.0)
            
            # Store convergence metrics in dataframe
            dataframe['score_convergence_ratio'] = convergence_ratio
            dataframe['convergence_multiplier'] = convergence_multiplier
            dataframe['trading_mode_indicator'] = trading_mode_indicator
            dataframe['trading_mode'] = trading_mode
            
            # Log convergence analysis results
            print(f"Score Convergence Analysis Results:")
            print(f"  Support Score: {raw_support_score:.6f}")
            print(f"  Resistance Score: {raw_resistance_score:.6f}")
            print(f"  Convergence Ratio: {convergence_ratio:.3f}")
            print(f"  Convergence Multiplier: {convergence_multiplier:.3f}")
            print(f"  Trading Mode: {trading_mode}")
            print(f"  Trading Mode Indicator: {trading_mode_indicator}")
            
            # Get detailed breakout scenario analysis
            breakout_analysis = self.detect_breakout_scenario(dataframe)
            print(f"  Breakout Analysis: {breakout_analysis['recommendation']}")
            
        else:
            print("Monte Carlo optimization skipped (disabled or insufficient data)")
            print(f"Available candles: {len(dataframe)}, minimum required: {self.MIN_LOOKBACK_PERIOD}")
            
            # Initialize with zeros when optimization is skipped
            dataframe['MC_Resistance_Score'] = 0.0
            dataframe['MC_Support_Score'] = 0.0
            
            # Initialize convergence metrics with default values
            dataframe['score_convergence_ratio'] = 0.0
            dataframe['convergence_multiplier'] = 1.0
            dataframe['trading_mode_indicator'] = 0.0
            dataframe['trading_mode'] = "INSUFFICIENT_DATA"

        # Calculate trendlines using local maxima/minima
        lookback = 200  # Use last 200 candles for trendline calculation

        # Don't calculate trendlines if we don't have enough data
        if len(dataframe) < 30:
            return dataframe
            
        recent_data = dataframe.tail(lookback).copy()
        
        # Calculate linear regression trendlines using TA-Lib
        # These will provide straight line trendlines using the least squares method
        try:
            # Get appropriate timeperiod for linear regression
            timeperiod = self.linearreg_timeperiod.value
            price_field = self.linearreg_price_field.value
            
            # The linear regression trendline is a true straight line calculated using
            # the least squares fit method over the specified timeperiod. Unlike the other
            # trendlines that connect pivot points, this line represents the statistical
            # best fit line through the price data and can help identify the trend
            # direction and strength. A steeper slope indicates a stronger trend.
            linearreg_data = self.trend_analyzer.calculate_talib_linearreg(
                dataframe, 
                timeperiod=timeperiod,
                price_field=price_field
            )
            
            # Copy the linear regression columns back to the original dataframe
            dataframe['linear_reg'] = linearreg_data['linear_reg']
            dataframe['linear_reg_slope'] = linearreg_data['linear_reg_slope']
            dataframe['linear_reg_angle'] = linearreg_data['linear_reg_angle']
            dataframe['linear_reg_intercept'] = linearreg_data['linear_reg_intercept']
            dataframe['linear_reg_line'] = linearreg_data['linear_reg_line']
                    
        except Exception as e:
            print(f"Error calculating linear regression: {e}")

        # Mark high and low points for visualization
        try:
            # Use the TrendAnalysis._find_swing_points function to find significant swing points
            # This provides better identification of true support and resistance levels
            high_swing_points = self.trend_analyzer._find_swing_points(
                prices=recent_data['high'].values,
                price_type='high',
                min_points=5,
                distance=10
            )
            
            low_swing_points = self.trend_analyzer._find_swing_points(
                prices=recent_data['low'].values,
                price_type='low',
                min_points=5,
                distance=10
            )
            
            # Initialize all_highs and all_lows in recent_data with NaN values
            recent_data['all_highs'] = np.nan
            recent_data['all_lows'] = np.nan
            
            # Map swing high points to the dataframe
            for idx, price in high_swing_points:
                if idx < len(recent_data):
                    actual_idx = len(dataframe) - len(recent_data) + idx
                    if 0 <= actual_idx < len(dataframe):
                        dataframe.loc[actual_idx, 'all_highs'] = price
                        recent_data.iloc[idx, recent_data.columns.get_loc('all_highs')] = price
            
            # Map swing low points to the dataframe
            for idx, price in low_swing_points:
                if idx < len(recent_data):
                    actual_idx = len(dataframe) - len(recent_data) + idx
                    if 0 <= actual_idx < len(dataframe):
                        dataframe.loc[actual_idx, 'all_lows'] = price
                        recent_data.iloc[idx, recent_data.columns.get_loc('all_lows')] = price
            
        except Exception as e:
            print(f"Error finding swing points: {e}")

        # Find global trendlines using gentrends
        try:
            # Generate trends using the gentrends function
            # This gives us the main max and min lines (resistance and support)
            trends = gentrends(recent_data, field='close', window=1/3.0)

            # Map the trend lines to the dataframe
            last_idx = len(dataframe) - len(recent_data)
            for i in range(len(trends)):
                current_idx = last_idx + i
                if current_idx < len(dataframe):
                    dataframe.loc[current_idx, 'Resistance Line'] = trends['Max Line'].iloc[i]
                    dataframe.loc[current_idx, 'Support Line'] = trends['Min Line'].iloc[i]
        except Exception as e:
            # Fail gracefully if gentrends fails
            print(f"Error in gentrends: {e}")
        
        # Use segtrends with different segment counts to find multiple resistance and support lines
        # Try with max_segments + 1 segments to get max_segments maxlines and minlines
        try:
            # Generate segmented trends
            seg_trends = segtrends(recent_data, field='close', segments=max_segments + 1)
            
            # Map all the individual max and min lines to the dataframe
            last_idx = len(dataframe) - len(recent_data)
            for i in range(len(seg_trends)):
                current_idx = last_idx + i
                if current_idx < len(dataframe):
                    # Copy all available Max_Line_X and Min_Line_X columns
                    for j in range(max_segments):
                        max_col = f'Max_Line_{j}'
                        min_col = f'Min_Line_{j}'
                        
                        if max_col in seg_trends.columns:
                            dataframe.loc[current_idx, max_col] = seg_trends[max_col].iloc[i]
                        
                        if min_col in seg_trends.columns:
                            dataframe.loc[current_idx, min_col] = seg_trends[min_col].iloc[i]
        except Exception as e:
            # Fail gracefully if segtrends fails
            print(f"Error in segtrends: {e}")
        
        # Rank trendlines based on proximity to price
        try:
            # Use the rank_trendlines function to score and rank the trendlines
            trendline_rankings = rank_trendlines(
                seg_trends, 
                price_field="Data", 
                threshold=self.trendline_proximity_threshold.value,
                touch_weight=self.trendline_touch_weight.value,
                max_prefix="Max_Line_", 
                min_prefix="Min_Line_",
                all_highs=recent_data['all_highs'],
                all_lows=recent_data['all_lows'],
                pivot_bonus=8.0  # Increased pivot bonus to emphasize swing points
            )
            
            # Store the top ranked maxlines and minlines in the dataframe
            # Add columns for the rankings
            ranked_maxlines = trendline_rankings["ranked_maxlines"]
            ranked_minlines = trendline_rankings["ranked_minlines"]
            
            # Calculate mean scores for each set and select the highest scored line
            scores_result = self.trend_analyzer.calculate_trendline_set_scores(ranked_maxlines, ranked_minlines)
            
            # Store mean scores in the dataframe
            resistance_mean_score = scores_result["max_mean_score"]
            support_mean_score = scores_result["min_mean_score"]
            highest_set = scores_result["highest_set"]
            highest_line_name = scores_result["highest_line_name"]
            highest_line_score = scores_result["highest_line_score"]
            
            # Add indicators for visualization
            dataframe['Resistance_Mean_Score'] = resistance_mean_score
            dataframe['Support_Mean_Score'] = support_mean_score
            
            # Calculate and store scores for the main Max Line and Min Line
            main_lines_score = rank_trendlines(
                trends,  # Use the gentrends output that has Resistance Line and Support Line
                price_field="Data", 
                threshold=self.trendline_proximity_threshold.value,
                touch_weight=self.trendline_touch_weight.value,
                all_highs=recent_data['all_highs'],
                all_lows=recent_data['all_lows'],
                pivot_bonus=9.0  # Increased pivot bonus to emphasize swing points
            )
            
            # Extract the scores for Resistance Line and Support Line
            resistance_line_score = main_lines_score["ranked_maxlines"].get("Max Line", 0)
            support_line_score = main_lines_score["ranked_minlines"].get("Min Line", 0)
            
            # Store the scores in the dataframe
            dataframe['Resistance_Line_Score'] = resistance_line_score
            dataframe['Support_Line_Score'] = support_line_score
                        
            # Copy the highest scored line to the Highest_Scored_Line column
            if highest_line_name is not None:
                last_idx = len(dataframe) - len(recent_data)
                for i in range(len(seg_trends)):
                    current_idx = last_idx + i
                    if current_idx < len(dataframe):
                        dataframe.loc[current_idx, 'Highest_Scored_Line'] = seg_trends[highest_line_name].iloc[i]
                        dataframe.loc[current_idx, 'Highest_Set_Mean'] = resistance_mean_score if highest_set == "max" else support_mean_score
                        dataframe.loc[current_idx, 'Highest_Line_Score'] = highest_line_score
                        dataframe.loc[current_idx, 'Highest_Line_Type'] = "Resistance" if highest_set == "max" else "Support"
                        
        except Exception as e:
            # Fail gracefully if ranking fails
            print(f"Error in ranking trendlines: {e}")

        print("--------------------------------")
        print("--------------------------------")
        print("Begin Default GARCH")        
        print("---")

        # Exemple d'utilisation
        garch_model = GARCHModel()
        risk_indicators = RiskIndicators()

        # Example 1: Basic GARCH(1,1) with default parameters

        simulated_returns = [0.07, 0.06, 0.05, 0.09]
        np_array = np.array(simulated_returns)
        # Calculate log returns on arithmetics returns
        log_returns = np.log(1 + np_array)
        # Calculate log returns based on price levels
        #log_returns = np.log(prices[1:] / prices[:-1])

        #df = 5  # Can be adjusted based on empirical data
        #VaR_Z = t.ppf(1 - confidence_level, df) * std
        confidence_level = 0.01
        VaR_1_percent, ES_1_percent = risk_indicators.calculate_var_es(
            simulated_returns=log_returns,
            z_score=norm.ppf(1 - confidence_level),
            std=garch_model.calculate_volatility(log_returns)
        )
        garch_model.reset_variance()
        print(f"VaR à 1% sur 35 jours (simulation GARCH sans Monte Carlo): {VaR_1_percent:.4f} ({VaR_1_percent * 100:.2f}%)")
        print(f"ES à 1% sur 35 jours (simulation GARCH sans Monte Carlo): {ES_1_percent:.4f} ({ES_1_percent * 100:.2f}%)")

        print("--------------------------------")
        print("Begin Monte Carlo with GARCH")        
        print("---")

        # Example 2: Monte Carlo simulation with GARCH using the new MonteCarloSimulator
        monte_carlo = MonteCarloSimulator(garch_model)
        simulated_returns = monte_carlo.simulate_with_garch(T=3, iterations=1000)
        confidence_level = 0.01
        VaR_1_percent, ES_1_percent = risk_indicators.calculate_var_es(
            simulated_returns=simulated_returns,
            z_score=norm.ppf(1 - confidence_level),
            std=garch_model.calculate_volatility(simulated_returns)
        )
        print(f"VaR à 1% sur 3 jours avec 1000 simulations (GARCH avec Monte Carlo): {VaR_1_percent:.4f} ({VaR_1_percent * 100:.2f}%)")
        print(f"ES à 1% sur 3 jours avec 1000 simulations (GARCH avec Monte Carlo): {ES_1_percent:.4f} ({ES_1_percent * 100:.2f}%)")
        
        # Example 3: Monte Carlo with custom distribution (Student's t)
        print("--------------------------------")
        print("Begin Monte Carlo with Student's t-distribution")        
        print("---")
        
        simulated_returns_t = monte_carlo.simulate_with_custom_distribution(T=3, iterations=1000)
        VaR_1_percent_t, ES_1_percent_t = risk_indicators.calculate_var_es(
            simulated_returns=simulated_returns_t,
            z_score=norm.ppf(1 - confidence_level),
            std=garch_model.calculate_volatility(simulated_returns_t)
        )
        print(f"VaR à 1% sur 3 jours avec distribution t de Student: {VaR_1_percent_t:.4f} ({VaR_1_percent_t * 100:.2f}%)")
        print(f"ES à 1% sur 3 jours avec distribution t de Student: {ES_1_percent_t:.4f} ({ES_1_percent_t * 100:.2f}%)")
        
        # Display simulation statistics
        stats = monte_carlo.get_simulation_statistics(simulated_returns)
        print(f"Statistiques de simulation - Moyenne: {stats['mean']:.4f}, Écart-type: {stats['std']:.4f}")
        print(f"Skewness: {stats['skewness']:.4f}, Kurtosis: {stats['kurtosis']:.4f}")
            
        return dataframe

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                          current_rate: float, current_profit: float,
                          min_stake: Optional[float], max_stake: float,
                          current_entry_rate: float, current_exit_rate: float,
                          current_entry_profit: float, current_exit_profit: float,
                          **kwargs) -> Optional[float]:
        """
        Adjust position size based on volatility regime and score convergence analysis.
        
        Position sizing factors:
        1. Volatility-based risk multiplier
        2. Score convergence multiplier (reduces size when scores are similar)
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if len(dataframe) > 0:
            current_candle = dataframe.iloc[-1].squeeze()
            
            # Get base risk multiplier from volatility regime
            base_risk_multiplier = current_candle.get('risk_multiplier', 1.0)
            
            # Get convergence multiplier
            convergence_multiplier = current_candle.get('convergence_multiplier', 1.0)
            
            # Combine both multipliers
            combined_multiplier = base_risk_multiplier * convergence_multiplier
            
            # Apply combined multiplier to position size
            adjusted_stake = max_stake * combined_multiplier
            
            # Log position sizing decision
            support_score = current_candle.get('MC_Support_Score', 0)
            resistance_score = current_candle.get('MC_Resistance_Score', 0)
            convergence_ratio = current_candle.get('score_convergence_ratio', 0)
            trading_mode = current_candle.get('trading_mode', 'UNKNOWN')
            
            print(f"Position Sizing for {trade.pair}:")
            print(f"  Base Risk Multiplier (volatility): {base_risk_multiplier:.3f}")
            print(f"  Convergence Multiplier: {convergence_multiplier:.3f}")
            print(f"  Combined Multiplier: {combined_multiplier:.3f}")
            print(f"  Trading Mode: {trading_mode}")
            print(f"  Convergence Ratio: {convergence_ratio:.3f}")
            print(f"  Support Score: {support_score:.6f}")
            print(f"  Resistance Score: {resistance_score:.6f}")
            print(f"  Max Stake: {max_stake:.2f}")
            print(f"  Adjusted Stake: {adjusted_stake:.2f}")
            
            return adjusted_stake
            
        return max_stake

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Bounce Trading Strategy Implementation using Price Extrema with Convergence Detection:
        - Long entry: Price creates a LOW extrema near support, then moves up (bounce off support)
        - Short entry: Price creates a HIGH extrema near resistance, then moves down (bounce off resistance)
        - Enhanced with Monte Carlo Score Convergence Detection for risk management
        
        Following bounce trading methodology from RebelsFunding with convergence risk management
        """
        # Initialize entry signals
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0
        
        # Long entry: Bounce off support using swing low extrema with convergence check
        if 'MC_Optimal_Support' in dataframe.columns and 'MC_Support_Score' in dataframe.columns and 'MC_Resistance_Score' in dataframe.columns:
            # Basic bounce conditions
            bounce_long_conditions = (
                # Current close is above support
                (dataframe['close'] > dataframe['MC_Optimal_Support']) &
                # Previous low was at or near support (within small tolerance)
                (abs(dataframe['low'].shift(1) - dataframe['MC_Optimal_Support'].shift(1)) <= 
                 dataframe['MC_Optimal_Support'].shift(1) * 0.002) &  # 0.2% tolerance
                # Previous low was lower than the low 2 candles ago (swing low pattern)
                (dataframe['low'].shift(1) <= dataframe['low'].shift(2)) &
                # Previous low was lower than current low (confirming bounce)
                (dataframe['low'].shift(1) < dataframe['low']) &
                # Current close is higher than previous close (upward movement)
                (dataframe['close'] > dataframe['close'].shift(1)) &
                # Support data is valid
                (~dataframe['MC_Optimal_Support'].isna()) &
                (~dataframe['MC_Optimal_Support'].shift(1).isna())
            )
            
            # Add convergence filter for long entries
            if self.enable_convergence_detection.value:
                # Calculate convergence ratios for each row
                convergence_ratios = dataframe.apply(
                    lambda row: self.calculate_score_convergence_ratio(
                        row.get('MC_Support_Score', 0), 
                        row.get('MC_Resistance_Score', 0)
                    ), axis=1
                )
                
                # Allow entry only if convergence ratio is below high threshold
                convergence_filter = convergence_ratios < self.score_convergence_high_threshold.value
                
                # Apply convergence filter to long conditions
                dataframe.loc[bounce_long_conditions & convergence_filter, 'enter_long'] = 1
                
                print(f"Long entries with convergence filter applied")
            else:
                # Apply basic bounce conditions without convergence filter
                dataframe.loc[bounce_long_conditions, 'enter_long'] = 1
        
        # Short entry: Bounce off resistance using swing high extrema with convergence check
        if 'MC_Optimal_Resistance' in dataframe.columns and 'MC_Support_Score' in dataframe.columns and 'MC_Resistance_Score' in dataframe.columns:
            # Basic bounce conditions
            bounce_short_conditions = (
                # Current close is below resistance
                (dataframe['close'] < dataframe['MC_Optimal_Resistance']) &
                # Previous high was at or near resistance (within small tolerance)
                (abs(dataframe['high'].shift(1) - dataframe['MC_Optimal_Resistance'].shift(1)) <= 
                 dataframe['MC_Optimal_Resistance'].shift(1) * 0.002) &  # 0.2% tolerance
                # Previous high was higher than the high 2 candles ago (swing high pattern)
                (dataframe['high'].shift(1) >= dataframe['high'].shift(2)) &
                # Previous high was higher than current high (confirming bounce)
                (dataframe['high'].shift(1) > dataframe['high']) &
                # Current close is lower than previous close (downward movement)
                (dataframe['close'] < dataframe['close'].shift(1)) &
                # Resistance data is valid
                (~dataframe['MC_Optimal_Resistance'].isna()) &
                (~dataframe['MC_Optimal_Resistance'].shift(1).isna())
            )
            
            # Add convergence filter for short entries
            if self.enable_convergence_detection.value:
                # Calculate convergence ratios for each row (reuse from long entry if available)
                if 'convergence_ratios' not in locals():
                    convergence_ratios = dataframe.apply(
                        lambda row: self.calculate_score_convergence_ratio(
                            row.get('MC_Support_Score', 0), 
                            row.get('MC_Resistance_Score', 0)
                        ), axis=1
                    )
                
                # Allow entry only if convergence ratio is below high threshold
                convergence_filter = convergence_ratios < self.score_convergence_high_threshold.value
                
                # Apply convergence filter to short conditions
                dataframe.loc[bounce_short_conditions & convergence_filter, 'enter_short'] = 1
                
                print(f"Short entries with convergence filter applied")
            else:
                # Apply basic bounce conditions without convergence filter
                dataframe.loc[bounce_short_conditions, 'enter_short'] = 1
        
        # Log entry signal summary
        long_signals = dataframe['enter_long'].sum()
        short_signals = dataframe['enter_short'].sum()
        print(f"Entry signals generated: {long_signals} long, {short_signals} short")
        
        if self.enable_convergence_detection.value and len(dataframe) > 0:
            # Log convergence analysis for the most recent candle
            current_candle = dataframe.iloc[-1]
            support_score = current_candle.get('MC_Support_Score', 0)
            resistance_score = current_candle.get('MC_Resistance_Score', 0)
            
            should_enter, reason = self.should_enter_trade_with_convergence(support_score, resistance_score)
            print(f"Current convergence status: {reason}")
            print(f"Entry allowed: {should_enter}")
        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Exit signals for bounce trading:
        - Exit long when support is broken with conviction (close below support)
        - Exit short when resistance is broken with conviction (close above resistance)
        """
        # Initialize exit signals
        dataframe.loc[:, 'exit_long'] = 0
        dataframe.loc[:, 'exit_short'] = 0
        
        # Exit long when support is definitively broken
        # Price closes below support with conviction
        if 'MC_Optimal_Support' in dataframe.columns:
            dataframe.loc[
                (dataframe['close'] < dataframe['MC_Optimal_Support']) &
                (dataframe['close'].shift(1) >= dataframe['MC_Optimal_Support'].shift(1)) &
                # Add conviction: close is significantly below support
                (dataframe['close'] < dataframe['MC_Optimal_Support'] * 0.998) &  # 0.2% below
                (~dataframe['MC_Optimal_Support'].isna()) &
                (~dataframe['MC_Optimal_Support'].shift(1).isna()),
                'exit_long'
            ] = 1
        
        # Exit short when resistance is definitively broken
        # Price closes above resistance with conviction
        if 'MC_Optimal_Resistance' in dataframe.columns:
            dataframe.loc[
                (dataframe['close'] > dataframe['MC_Optimal_Resistance']) &
                (dataframe['close'].shift(1) <= dataframe['MC_Optimal_Resistance'].shift(1)) &
                # Add conviction: close is significantly above resistance
                (dataframe['close'] > dataframe['MC_Optimal_Resistance'] * 1.002) &  # 0.2% above
                (~dataframe['MC_Optimal_Resistance'].isna()) &
                (~dataframe['MC_Optimal_Resistance'].shift(1).isna()),
                'exit_short'
            ] = 1
        
        return dataframe

    def get_market_condition_description(self, dataframe: DataFrame) -> Dict[str, str]:
        """
        Generate a human-readable description of current market conditions
        based on the timeframe analysis.
        
        Args:
            dataframe: The analyzed dataframe with indicators
            
        Returns:
            A dictionary containing market condition descriptions
        """
        if len(dataframe) < 10:
            return {"error": "Not enough data for market condition analysis"}
            
        # Get the most recent candle
        current_candle = dataframe.iloc[-1].squeeze()
        
        # Get the highest timeframe used
        highest_tf = current_candle.get('highest_timeframe', self.timeframe)
        
        # Get volatility regime
        volatility_regime = current_candle.get('volatility_regime', 'Unknown')
        
        # Get risk multiplier
        risk_multiplier = current_candle.get('risk_multiplier', 1.0)
        
        # Get linear regression angle to determine trend direction
        trend_angle = current_candle.get('linear_reg_angle', 0)
        
        # Determine trend direction based on angle
        if trend_angle > 45:
            trend_direction = "Strong uptrend"
        elif trend_angle > 20:
            trend_direction = "Moderate uptrend"
        elif trend_angle > 5:
            trend_direction = "Mild uptrend"
        elif trend_angle > -5:
            trend_direction = "Sideways"
        elif trend_angle > -20:
            trend_direction = "Mild downtrend"
        elif trend_angle > -45:
            trend_direction = "Moderate downtrend"
        else:
            trend_direction = "Strong downtrend"
        
        # Get highest scored line type (support/resistance)
        strongest_level_type = current_candle.get('Highest_Line_Type', 'Unknown')
        
        # Format the percentage of risk based on the multiplier
        risk_percentage = risk_multiplier * 100
        
        # Create description
        description = {
            "timeframe": f"Analysis based on {highest_tf} data",
            "volatility": f"{volatility_regime} volatility environment",
            "trend": trend_direction,
            "important_level": f"Most significant level: {strongest_level_type}",
            "risk_assessment": f"Recommended position size: {risk_percentage:.1f}% of maximum",
            "summary": (
                f"Market is in a {trend_direction.lower()} with {volatility_regime.lower()} volatility. "
                f"Position sizing set to {risk_percentage:.1f}% based on {highest_tf} analysis."
            )
        }
        
        return description