#####################
# NECESSARY IMPORTS #
#####################

import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from .market_analysis import Market_analysis

################################################################################
# DIVERSIFICATION
#
# These are the functions necessary to verify if there exists any type of
# correlation between the stocks that have been screened. These functions allow
# one to check if the screened stocks belong to the same sectors and also if
# there has been a correlation in the closing prices in psat data
################################################################################

def closing_data(tickers: list) -> pd.core.frame.DataFrame:

    df_work = pd.DataFrame()

    for ticker in tickers:

        df_ticker = yf.Ticker(ticker).history(period="200d").drop(["Dividends","Stock Splits"], axis = 1)

        df_work[ticker] = df_ticker.Close

    return df_work

def compare_stocks(tickers: list):

    df_work = closing_data(tickers + ["^GSPC"])

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

def verify_correlation(tickers: list) -> pd.core.frame.DataFrame:

    df_close = pd.DataFrame()

    for ticker in tickers:

        df_ticker = yf.Ticker(ticker).history(period="200d").drop(["Dividends","Stock Splits"], axis = 1)

        df_close[ticker] = df_ticker.Close

    correlation_matrix = df_close.corr()

    plt.figure(figsize = (20,10))
    sn.diverging_palette(145, 300, s=60, as_cmap=True)
    sn.heatmap(correlation_matrix, annot=True, cmap = "PiYG")
    plt.show()

    return correlation_matrix

def verify_sector(tickers: list) -> dict:

    sector_dict = {}

    for ticker in tickers:

        try:
            ticker_object = yf.Ticker(ticker)
            ticker_sector = ticker_object.info["sector"]

            if ticker_sector not in sector_dict:

                sector_dict[ticker_sector] = [ticker]

            else:

                sector_dict[ticker_sector].append(ticker)

        except KeyError:
            continue

    return sector_dict
