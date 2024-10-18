#####################
# NECESSARY IMPORTS #
#####################

import numpy as np
import yfinance as yf
import pandas as pd
from .financial_indicators import log_returns

################################################################################
# STOCK LISTS
#
#
# These are the functions necessary for filtering through lists of stock tickers
# and produce lists of high volume stocks and strong/weak relative strength
# stocks in comparison with the S&P500. These also include functions to import
# and export these lists from/to .csv data files
################################################################################

class Stock_lists():

    stock_tickers = []
    high_volume_stocks = []
    strong_stocks = []
    weak_stocks = []
    screened_stocks = []

################################################################################
# Description: Imports the tickers from the Nasdaq and New York Stock Exchange
# from data files and saves a list containing these tickers
#
# Inputs: None
#
# Outputs: None
################################################################################

    @classmethod
    def import_stock_tickers(cls):

        tickers_nasdaq = pd.read_csv("stock_lists/tickers_nasdaq.csv").Ticker.to_list()
        tickers_nyse = pd.read_csv("stock_lists/tickers_nyse.csv").Ticker.to_list()

        cls.stock_tickers = tickers_nasdaq + tickers_nyse

################################################################################
# Description: Imports the high volume, strong, and weak stock lists from csv
# data files and saves them on lists containing these stocks
#
# Inputs: None
#
# Outputs: None
################################################################################

    @classmethod
    def import_stock_lists(cls):

        cls.high_volume_stocks = pd.read_csv("stock_lists/high_volume_stocks.csv").Ticker.to_list()
        cls.strong_stocks = pd.read_csv("stock_lists/strong_stocks.csv").Ticker.to_list()
        cls.weak_stocks = pd.read_csv("stock_lists/weak_stocks.csv").Ticker.to_list()

################################################################################
# Description: Filters through a list of stocks to select those with a volume
# larger than a certain threshold of interest and saves the result in a list
#
# Inputs:
# volume: minimum volume for a high volume stock
# export: boolean variable to export the result to a file
#
# Outputs: None
################################################################################

    @classmethod
    def produce_high_volume_list(cls, volume: int = 200000, export: bool = True):

        cls.import_stock_tickers()

        assert len(cls.stock_tickers) > 0, "List of stock tickers is empty"
        assert volume > 0, f"Volume {volume} must be greater than zero"

        work_list = []

        for ticker in cls.stock_tickers:

            try:
                if yf.Ticker(ticker).history(period="3mo").Volume.mean() > volume:
                    work_list.append(ticker)

            except AttributeError:
                continue

        cls.high_volume_stocks = work_list

        if export:
            cls.export_high_volume_stocks()


################################################################################
# Description: Filters through a list of tickers, compares the relative change
# of each ticker with the relative change of the S&P500, and returns two lists
# containing the tickers that are stronger and weaker with respect to the S&P500
#
# Inputs:
# period: time interval to be analyzed
# export: boolean variable to export the result to a file
#
# Outputs: None
################################################################################

    @classmethod
    def produce_relative_strength_lists(cls, period: int = 30, export: bool = True):

        assert len(cls.high_volume_stocks) > 0, "List of high volume stocks is empty"
        assert period > 0, f"Period {period} must be greater than zero"

        work_strong_list = []
        work_weak_list = []
        period_str = str(period) + "d"

        spx_change = log_returns(yf.Ticker("^GSPC").history(period=period_str).drop(["Dividends","Stock Splits"], axis = 1)).tail(1).CLR.iloc[0]

        for ticker in cls.high_volume_stocks:

            try:
                stock_change = log_returns(yf.Ticker(ticker).history(period=period_str)).tail(1).CLR.iloc[0]

                if stock_change > spx_change:
                    work_strong_list.append(ticker)
                else:
                    work_weak_list.append(ticker)
            except (AttributeError, IndexError):
                continue

        cls.strong_stocks = work_strong_list
        cls.weak_stocks = work_weak_list

        if export:
            cls.export_relative_strength_stocks()


################################################################################
# Description: Exports the list of high volume, strong, and weak stocks to a csv
# file
#
# Inputs: None
#
# Outputs: None
################################################################################

    @classmethod
    def export_high_volume_stocks(cls):

        pd.DataFrame(Stock_lists.high_volume_stocks,columns=["Ticker"]).to_csv("stock_lists/high_volume_stocks.csv")

    @classmethod
    def export_relative_strength_stocks(cls):

        pd.DataFrame(Stock_lists.strong_stocks,columns=["Ticker"]).to_csv("stock_lists/strong_stocks.csv")
        pd.DataFrame(Stock_lists.weak_stocks,columns=["Ticker"]).to_csv("stock_lists/weak_stocks.csv")
