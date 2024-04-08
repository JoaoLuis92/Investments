#####################
# NECESSARY IMPORTS #
#####################

import numpy as np
import yfinance as yf
import pandas as pd
from .market_analysis import Market_analysis
from .financial_indicators import check_MACD

################################################################################
# STRATEGY SCREENERS
#
# These are the functions that filter through lists of tickers with the goal
# of finding which stocks currently satisfy a given set of requirements. Each
# screener follows its own rules to filter through the stocks.
################################################################################

class Screener():

    screened_stocks = {}
    screened_stocks_list = []


################################################################################
# Description: Filters through a list of stocks and verifies which satisfy a
# given set of requirements that describe a trading strategy. The following
# strategies have been implemented:
#
## MACD crossover: MACD
#
# Inputs:
# tickers: list of tickers to be screened
#
# Outputs:
# signal_list: list of tickers for which a signal has been detected
################################################################################

    @classmethod
    def screener_MACD_crossover_long(cls, tickers: list) -> list:

        if not Market_analysis.long_bias:
            print("It is not recommended to follow this strategy in the current market condition.")

        signal_list = []

        for ticker in tickers:

            df_ticker = yf.Ticker(ticker).history(period="200d").drop(["Dividends","Stock Splits"], axis = 1)

            signal = check_MACD(df_ticker, 12, 26, 9, True)

            if signal.is_bullish:
                signal_list.append(ticker)

        cls.screened_stocks["MACD crossover bullish"] = signal_list
        cls.screened_stocks_list.extend(signal_list)

    @classmethod
    def screener_MACD_crossover_short(cls, tickers: list) -> list:

        if not Market_analysis.short_bias:
            print("It is not recommended to follow this strategy in the current market condition.")

        signal_list = []

        for ticker in tickers:

            df_ticker = yf.Ticker(ticker).history(period="200d").drop(["Dividends","Stock Splits"], axis = 1)

            signal = check_MACD(df_ticker, 12, 26, 9, True)

            if signal.is_bearish:
                signal_list.append(ticker)

        cls.screened_stocks["MACD crossover bearish"] = signal_list
        cls.screened_stocks_list.extend(signal_list)
