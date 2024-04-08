#####################
# NECESSARY IMPORTS #
#####################

import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from .financial_indicators import simple_moving_average, exponential_moving_average, full_stochastic

################################################################################
# MARKET ANALYSIS
#
#
# These are the functions necessary to analyze the general trend of the market
# and current price status in terms of oversold7overbought conditions to allow
# the user to select which stocks are recommended to screen. One can also plot
# the closing price of the S&P500 alongside with several moving averages and the
# corresponding stochastic indicator.
################################################################################

class Market_analysis:

    spx_df = yf.Ticker("^GSPC").history(period="200d").drop(["Dividends","Stock Splits"], axis = 1)

    uptrend = False
    downtrend = False
    oversold = False
    overbought = False
    long_bias = False
    short_bias = False

################################################################################
# Description: Verifies if the market is currently on an uptrend or downtrend
# by comparing moving averages with different periods and saves two boolean
# variables describing the trend
#
# Inputs: None
#
# Outputs: None
################################################################################

    @classmethod
    def verify_market_trend(cls):

        for period in [50, 100, 200]:
            cls.spx_df = simple_moving_average(cls.spx_df, period)

        for period in [20, 40]:
            cls.spx_df = exponential_moving_average(cls.spx_df, period)

        short_uptrend = (cls.spx_df.tail(1).Close > cls.spx_df.tail(1).EMA20).bool() and (cls.spx_df.tail(1).EMA20 > cls.spx_df.tail(1).EMA40).bool()
        short_downtrend = (cls.spx_df.tail(1).Close < cls.spx_df.tail(1).EMA20).bool() and (cls.spx_df.tail(1).EMA20 < cls.spx_df.tail(1).EMA40).bool()

        long_uptrend = (cls.spx_df.tail(1).Close > cls.spx_df.tail(1).SMA50).bool() and (cls.spx_df.tail(1).SMA50 > cls.spx_df.tail(1).SMA100).bool() and (cls.spx_df.tail(1).SMA100 > cls.spx_df.tail(1).SMA200).bool()
        long_downtrend = (cls.spx_df.tail(1).Close < cls.spx_df.tail(1).SMA50).bool() and (cls.spx_df.tail(1).SMA50 < cls.spx_df.tail(1).SMA100).bool() and (cls.spx_df.tail(1).SMA100 < cls.spx_df.tail(1).SMA200).bool()

        cls.uptrend = short_uptrend and long_uptrend
        cls.downtrend = short_downtrend and long_downtrend

################################################################################
# Description: Verifies if the market is currently oversold or overbought by
# analyzing the full stochastic oscillator with periods 5, 3, 3 and saves the
# result in two boolean variables
#
# Inputs: None
#
# Outputs: None
################################################################################

    @classmethod
    def verify_market_stochastic(cls):

        cls.spx_df = full_stochastic(cls.spx_df,5,3,3)

        cls.oversold = (cls.spx_df.tail(1)["Fast K"] < 20).bool() and (cls.spx_df.tail(1)["Slow K"] < 20).bool()
        cls.overbought = (cls.spx_df.tail(1)["Fast K"] > 80).bool() and (cls.spx_df.tail(1)["Slow K"] > 80).bool()

################################################################################
# Description: Analyzes the market trend and the price status through the full
# stochastic oscillator, informs the user which type of positions (long/short)
# are recommended in the current market conditions, and caves a list that
# contains either strong or weak stocks depending on the market being bullish
# or bearish, respectively
#
# Inputs: None
#
# Outputs: None
################################################################################

    @classmethod
    def analyze_market(cls, plot: bool = True):

        cls.verify_market_trend()
        cls.verify_market_stochastic()

        cls.long_bias = cls.uptrend and not cls.overbought
        cls.short_bias = cls.downtrend and not cls.oversold

        if cls.long_bias:
            print("Market is currently good for long positions.")

        elif cls.short_bias:
            print("Market is currently good for short positions.")

        else:
            print("Market is currently indecisive.")

        if plot:
            cls.plot_market_conditions()

################################################################################
# Description: Produces two plots of the current market conditions, the first
# showing the closing price alongside with four moving averages of interest, and
# the second showing the fast K and slow K stochastic oscillators
#
# Inputs: None
#
# Outputs: None
################################################################################

    @classmethod
    def plot_market_conditions(cls):

        for period in [50, 100, 200]:
            cls.spx_df = simple_moving_average(cls.spx_df, period)

        for period in [20, 40]:
            cls.spx_df = exponential_moving_average(cls.spx_df, period)

        cls.spx_df = full_stochastic(cls.spx_df,5,3,3)

        # First plot: SPX and moving averages

        plt.figure(figsize=(10, 5))
        plt.plot(cls.spx_df.tail(30).Close, 'k.-', label='S&P500')
        plt.plot(cls.spx_df.tail(30).EMA20, 'r--', label='EMA20')
        plt.plot(cls.spx_df.tail(30).EMA40, 'b--', label='EMA40')
        plt.plot(cls.spx_df.tail(30).SMA50, 'g-', label='SMA50')
        plt.plot(cls.spx_df.tail(30).SMA100, 'm-', label='SMA100')
        plt.grid(linestyle=':')
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.show()

        # Second plot: fast K and slow K stochastic

        plt.figure(figsize=(10, 5))
        plt.plot(cls.spx_df.tail(30)["Fast K"], 'k-', label='Fast %K')
        plt.plot(cls.spx_df.tail(30)["Slow K"], 'r-', label='Slow %K')
        plt.axhline(y=80, color='g', linestyle='--')
        plt.axhline(y=20, color='g', linestyle='--')
        plt.grid(linestyle=':')
        plt.ylim(0,100)
        plt.xlabel("Date")
        plt.ylabel("Full Stochastic")
        plt.legend()
        plt.show()
