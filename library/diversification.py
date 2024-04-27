#####################
# NECESSARY IMPORTS #
#####################

import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from .market_analysis import Market_analysis
from .financial_indicators import log_returns
from .strategy_screeners import Screener

################################################################################
# DIVERSIFICATION
#
# These are the functions necessary to verify if there exists any type of
# correlation between the stocks that have been screened. These functions allow
# one to check if the screened stocks belong to the same sectors and also if
# there has been a correlation in the closing prices in psat data
################################################################################

def remove_duplicates(tickers: list) -> list:

    unique_list = []

    for ticker in tickers:
        if ticker not in unique_list:
            unique_list.append(ticker)

    return unique_list

def closing_data(tickers: list) -> pd.core.frame.DataFrame:

    df_work = pd.DataFrame()

    for ticker in tickers:

        df_ticker = yf.Ticker(ticker).history(period="2000d").drop(["Dividends","Stock Splits"], axis = 1)

        df_work[ticker] = df_ticker.Close

    return df_work

def compare_stocks(tickers: list):

    df_work = closing_data(tickers + ["^GSPC"]).tail(30)

    df_work = np.log(df_work / df_work.shift(1)).cumsum().apply(np.exp)

    plt.figure(figsize=(10, 5))

    for ticker in tickers:
        plt.plot(df_work[ticker], label=ticker)

    plt.plot(df_work["^GSPC"], "k", label="S&P 500")

    plt.grid(linestyle=':')
    plt.xlabel("Date")
    plt.ylabel("Cumulative log returns")
    plt.legend()
    plt.show()

class Diversification():

    stocks_to_diversify = []
    diversified_stocks = []
    correlation_matrix = pd.DataFrame()
    covariance_matrix = pd.DataFrame()
    sector_dict = {}

    @classmethod
    def import_screened_stocks(cls):
        cls.stocks_to_diversify = Screener.screened_stocks_list

    @classmethod
    def correlation_and_covariance(cls, plot_cor: bool = False, plot_cov: bool = False) -> pd.core.frame.DataFrame:

        df_work = pd.DataFrame()

        for ticker in cls.stocks_to_diversify:

            df_ticker = yf.Ticker(ticker).history(period="200d").drop(["Dividends","Stock Splits"], axis = 1)
            df_ticker = log_returns(df_ticker)

            df_work[ticker] = df_ticker.LR

        cls.correlation_matrix = df_work.corr()
        cls.covariance_matrix = df_work.cov() * 252 #annualized

        if plot_cor:

            plt.figure(figsize = (20,10))
            sn.diverging_palette(145, 300, s=60, as_cmap=True)
            sn.heatmap(cls.covariance_matrix, annot=True, cmap = "PiYG")
            plt.show()

        if plot_cov:

            plt.figure(figsize = (20,10))
            sn.diverging_palette(145, 300, s=60, as_cmap=True)
            sn.heatmap(cls.correlation_matrix, annot=True, cmap = "PiYG")
            plt.show()

    @classmethod
    def diversify_by_correlation(cls, type: str = "Long"):

        cls.correlation_and_covariance()

        df_work = closing_data(cls.correlation_matrix.index).tail(30)
        df_work = np.log(df_work / df_work.shift(1)).mean()

        for ticker in cls.correlation_matrix.index:

            if type == "Long":
                cls.diversified_stocks.append(df_work.loc[cls.correlation_matrix[cls.correlation_matrix[ticker]>0.3].index.to_list()].idxmax())

            elif type == "Short":
                cls.diversified_stocks.append(df_work.loc[cls.correlation_matrix[cls.correlation_matrix[ticker]>0.3].index.to_list()].idxmin())

        cls.diversified_stocks = remove_duplicates(cls.diversified_stocks)

    @classmethod
    def verify_sector(cls):

        for ticker in cls.diversified_stocks:

            try:
                ticker_object = yf.Ticker(ticker)
                ticker_sector = ticker_object.info["sector"]

                if ticker_sector not in cls.sector_dict:

                    cls.sector_dict[ticker_sector] = [ticker]

                else:

                    cls.sector_dict[ticker_sector].append(ticker)

            except KeyError:
                continue

    @classmethod
    def diversify_by_sector(cls, type: str = "Long"):

        cls.verify_sector()

        df_work = closing_data(cls.diversified_stocks).tail(30)
        df_work = np.log(df_work / df_work.shift(1)).mean()

        for sector in cls.sector_dict.keys():

            if type == "Long":
                cls.sector_dict[sector] = df_work.loc[cls.sector_dict[sector]].idxmax()

            elif type == "Short":
                cls.sector_dict[sector] = df_work.loc[cls.sector_dict[sector]].idxmin()

        cls.diversified_stocks = list(cls.sector_dict.values())

    @classmethod
    def complete_diversify(cls, type: str = "Long"):

        cls.import_screened_stocks()
        cls.diversify_by_correlation(type)
        cls.diversify_by_sector(type)
