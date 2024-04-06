#####################
# NECESSARY IMPORTS #
#####################

import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

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

        tickers_nasdaq = pd.read_csv("tickers_nasdaq.csv").Ticker.to_list()
        tickers_nyse = pd.read_csv("tickers_nyse.csv").Ticker.to_list()

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

        cls.high_volume_stocks = pd.read_csv("high_volume_stocks.csv").Ticker.to_list()
        cls.strong_stocks = pd.read_csv("strong_stocks.csv").Ticker.to_list()
        cls.weak_stocks = pd.read_csv("weak_stocks.csv").Ticker.to_list()

################################################################################
# Description: Filters through a list of stocks to select those with a volume
# larger than a certain threshold of interest and saves the result in a list
#
# Inputs:
# volume: minimum volume for a high volume stock
#
# Outputs: None
################################################################################

    @classmethod
    def produce_high_volume_list(cls, volume: int = 200000):

        assert len(cls.stock_tickers) > 0, "List of stock tickers is empty"
        assert volume > 0, f"Volume {volume} must be greater than zero"

        work_list = []

        for ticker in cls.stock_tickers:

            try:
                if yf.Ticker(ticker).history(period="3mo").Volume.mean() > volume:
                    work_list.append(ticker)

            except:
                continue

        cls.high_volume_stocks = work_list

################################################################################
# Description: Computes the relative change of the closing price of a stock in
# a given time period of interest
#
# Inputs:
# df: dataframe containing the stock data
# period: time interval to be analyzed
#
# Outputs:
# relative_change: value of the relative change of the stock
################################################################################

    @staticmethod
    def stock_relative_change(df: pd.core.frame.DataFrame, period: int) -> float:

        assert period > 0, f"Period {period} must be greater than zero"

        return (df.tail(period).iloc[period-1].Close - df.tail(period).iloc[0].Close) / df.tail(period).iloc[0].Close

################################################################################
# Description: Filters through a list of tickers, compares the relative change
# of each ticker with the relative change of the S&P500, and returns two lists
# containing the tickers that are stronger and weaker with respect to the S&P500
#
# Inputs:
# period: time interval to be analyzed
#
# Outputs: None
################################################################################

    @classmethod
    def produce_relative_strength_lists(cls, period: int = 30) -> (list, list):

        assert len(cls.high_volume_stocks) > 0, "List of high volume stocks is empty"
        assert period > 0, f"Period {period} must be greater than zero"

        work_strong_list = []
        work_weak_list = []

        spx_change = cls.stock_relative_change(Market_analysis.spx_df, period)

        for ticker in cls.high_volume_stocks:

            try:
                stock_data = yf.Ticker(ticker).history(period="3mo")
                stock_change = cls.stock_relative_change(stock_data, period)

                if stock_change > spx_change:
                    work_strong_list.append(ticker)
                else:
                    work_weak_list.append(ticker)
            except:
                continue

        cls.strong_stocks = work_strong_list
        cls.weak_stocks = work_weak_list

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

        pd.DataFrame(Stock_lists.high_volume_stocks,columns=["Ticker"]).to_csv("high_volume_stocks.csv")

    @classmethod
    def export_relative_strength_stocks(cls):

        pd.DataFrame(Stock_lists.strong_stocks,columns=["Ticker"]).to_csv("strong_stocks.csv")
        pd.DataFrame(Stock_lists.weak_stocks,columns=["Ticker"]).to_csv("weak_stocks.csv")

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

    stocks_to_screen = []

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
    def analyze_market(cls):

        cls.verify_market_trend()
        cls.verify_market_stochastic()

        cls.long_bias = cls.uptrend and not cls.overbought
        cls.short_bias = cls.downtrend and not cls.oversold

        if cls.long_bias:
            print("Market is currently good for long positions.")
            cls.stocks_to_screen = Stock_lists.strong_stocks

        elif cls.short_bias:
            print("Market is currently good for short positions.")
            cls.stocks_to_screen = Stock_lists.weak_stocks

        else:
            print("Market is currently indecisive.")

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

################################################################################
# FINANCIAL INDICATORS
#
# These are the functions necessary to process the data and compute the values
# of different financial indicators. Furthermore, some of these functions are
# used to apply the financial indicator to the data and verify is a signal has
# been detected.
################################################################################

################################################################################
# Description: Computes the simple or exponentially weighted moving average with
# a given period of the data input
#
# Inputs:
# df: dataframe containing stock data
# period: time interval of the moving average
#
# Outputs:
# df_work: dataframe with additional column with moving average
################################################################################

def simple_moving_average(df: pd.core.frame.DataFrame, period: int) -> pd.core.frame.DataFrame:

    assert period > 0, f"Period {period} must be greater than zero"

    df_work = df.copy()
    df_work["SMA"+str(period)] = df_work["Close"].rolling(period).mean()

    return df_work

def exponential_moving_average(df: pd.core.frame.DataFrame, period: int) -> pd.core.frame.DataFrame:

    assert period > 0, f"Period {period} must be greater than zero"

    df_work = df.copy()
    df_work["EMA"+str(period)] = df_work['Close'].ewm(span = period).mean()

    return df_work

################################################################################
# Description: Computes the full stochastic oscillator with gien fast K, slow K
# and slow D periods of the data input
#
# Inputs:
# df: dataframe containing stock data
# fk_period: time interval of fast K period
# sk_period: time interval of slow K (fast D) period
# sd_period: time interval of slow D period
#
# Outputs:
# df_work: dataframe with additional columns with fast K, slow K, and slow D
################################################################################

def full_stochastic(df: pd.core.frame.DataFrame, fk_period: int = 5, sk_period: int = 5, sd_period: int = 3) -> pd.core.frame.DataFrame:

    assert fk_period > 0, f"Fast-K period {fk_period} must be greater than zero"
    assert sk_period > 0, f"Slow-K period {sk_period} must be greater than zero"
    assert sd_period > 0, f"Slow-D period {sd_period} must be greater than zero"

    df_work = df.copy()
    fast_k_list = []

    for i in range(len(df)):
        low = df_work.iloc[i]['Low']
        high = df_work.iloc[i]['High']

        if i >= fk_period:

            for n in range(fk_period):

                if df_work.iloc[i-n]['High'] >= high:
                    high = df_work.iloc[i-n]['High']
                elif df_work.iloc[i-n]['Low'] < low:
                    low = df_work.iloc[i-n]['Low']
        if high != low:
            fast_k = 100 * (df_work.iloc[i]['Close'] - low) / (high - low)
        else:
            fast_k = 0

        fast_k_list.append(fast_k)

    df_work["Fast K"] = fast_k_list
    df_work["Slow K"] = df_work["Fast K"].rolling(sk_period).mean()
    df_work["Slow D"] = df_work["Slow K"].rolling(sd_period).mean()

    return df_work

################################################################################
# Description: Computes the moving average convergence-divergence indicator with
# the given time periods of the data input
#
# Inputs:
# df: dataframe containing stock data
# macd_period_1: time interval of first exponential moving average
# macd_period_2: time interval of second exponential moving average
# signal_period: time interval of moving average for signal
#
# Outputs:
# df_work: dataframe with additional columns with MACD and signal values
################################################################################

def moving_average_convergence_divergence(df: pd.core.frame.DataFrame, macd_period_1: int = 12, macd_period_2: int = 26, signal_period: int = 9) -> pd.core.frame.DataFrame:

    assert macd_period_1 > 0, f"First MACD period {macd_period_1} must be greater than zero"
    assert macd_period_2 > 0, f"Second MACD Period {macd_period_2} must be greater than zero"
    assert signal_period > 0, f"Signal-line period {signal_period} must be greater than zero"

    df_work = df.copy()

    df_work = exponential_moving_average(df_work, macd_period_1)
    df_work = exponential_moving_average(df_work, macd_period_2)

    df_work["MACD"] = df_work["EMA" + str(macd_period_1)] - df_work["EMA" + str(macd_period_2)]
    df_work["Signal"] = (df_work['MACD']).ewm(span = signal_period).mean()

    df_work = df_work.drop(columns = ["EMA" + str(macd_period_1), "EMA" + str(macd_period_2)])

    return df_work

################################################################################
# Description: Computes the log returns and the cummulative log returns for the
# data given as input
#
# Inputs:
# df: dataframe containing stock data
#
# Outputs:
# df_work: dataframe with additional columns with LR and CLR
################################################################################

def log_returns(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:

    df_work = df.copy()

    df_work["LR"] = np.log( df_work["Close"] / df_work["Close"].shift(1))
    df_work["CLR"] = df_work["LR"].cumsum().apply(np.exp)

    return df_work

################################################################################
# Description: Computes the average true range indicator with the given time
# period of the data input
#
# Inputs:
# df: dataframe containing stock data
# period: time interval to compute the ATR
#
# Outputs:
# df_work: dataframe with additional columns with ATR
################################################################################

def average_true_range(df: pd.core.frame.DataFrame, period: int = 14) -> pd.core.frame.DataFrame:

    assert period > 0, f"Period {period} must be greater than zero"

    df_work = df.copy()

    df_work["TR1"] = df_work.High - df_work.Low
    df_work["TR2"] = df_work.High - df_work.Close.shift()
    df_work["TR3"] = df_work.Low - df_work.Close.shift()

    df_work["TR"] = df_work[['TR1', 'TR2', 'TR3']].max(axis=1)
    df_work["ATR"] = df_work.TR.ewm(alpha = 1 / period, adjust=False).mean()

    df_work = df_work.drop(columns = ["TR1", "TR2", "TR3", "TR"])

    return df_work

################################################################################
# Description: Verifies the current trend of the stock data given as input based
# on the analysis of four major simple moving averages
#
# Inputs:
# df: dataframe containing stock data
#
# Outputs:
# uptrend: boolean variable that is true is the stock is uptrending
# downtrend: boolean variable that is true is the stock is downtrending
################################################################################

def check_trend(df: pd.core.frame.DataFrame) -> (bool, bool):

    df_work = df.copy()

    for period in [20, 40, 100, 200]:
        df_work = simple_moving_average(df_work, period)

    uptrend = (df_work.tail(1)["SMA20"] > df_work.tail(1)["SMA40"]).bool() and (df_work.tail(1)["SMA40"] > df_work.tail(1)["SMA100"]).bool() and (df_work.tail(1)["SMA100"] > df_work.tail(1)["SMA200"]).bool()
    downtrend = (df_work.tail(1)["SMA20"] < df_work.tail(1)["SMA40"]).bool() and (df_work.tail(1)["SMA40"] < df_work.tail(1)["SMA100"]).bool() and (df_work.tail(1)["SMA100"] < df_work.tail(1)["SMA200"]).bool()

    return uptrend, downtrend

################################################################################
# Description: Verifies the current price status, i.e., oversold of overbought,
# of the strock data input based on the full stochastic oscillator
#
# Inputs:
# df: dataframe containing stock data
# fk_period: time interval of fast K period
# sk_period: time interval of slow K (fast D) period
# sd_period: time interval of slow D period
#
# Outputs:
# oversold: boolean variable that is true is the stock is oversold
# overbought: boolean variable that is true is the stock is overbought
################################################################################

def check_stochastic(df: pd.core.frame.DataFrame, fk_period: int = 5, sk_period: int = 3, sd_period: int = 3) -> (bool, bool):

    df_work = df.copy()
    df_work = full_stochastic(df_work, fk_period, sk_period, sd_period)

    oversold = (df_work.tail(1)["Fast K"] < 30).bool() and (df_work.tail(1)["Slow K"] < 30).bool()
    overbought = (df_work.tail(1)["Fast K"] > 70).bool() and (df_work.tail(1)["Slow K"] > 70).bool()

    return oversold, overbought

################################################################################
# Description: Verifies is the moving average convergence-divergence indicator
# is currently indicating a bullish or bearish signal. Optionally, it checks if
# a crossover has happened from the previous day
#
# Inputs:
# df: dataframe containing stock data
# macd_period_1: time interval of first exponential moving average
# macd_period_2: time interval of second exponential moving average
# signal_period: time interval of moving average for signal
#
# Outputs:
# uptrend: boolean variable that is true is the stock is uptrending
# downtrend: boolean variable that is true is the stock is downtrending
################################################################################

def check_MACD(df: pd.core.frame.DataFrame, macd_period_1: int = 12, macd_period_2: int = 26, signal_period: int = 9, crossover: bool = False) -> (bool, bool):

    df_work = df.copy()
    df_work = moving_average_convergence_divergence(df_work, macd_period_1, macd_period_2, signal_period)

    macd_bullish = (df_work.tail(1)["MACD"] > df_work.tail(1)["Signal"]).bool()
    macd_bearish = (df_work.tail(1)["MACD"] < df_work.tail(1)["Signal"]).bool()

    if crossover:

        bullish_cross = macd_bullish and df_work.tail(2).iloc[0]["MACD"] < df_work.tail(2).iloc[0]["Signal"]
        bearish_cross = macd_bearish and df_work.tail(2).iloc[0]["MACD"] > df_work.tail(2).iloc[0]["Signal"]

        return bullish_cross, bearish_cross

    else:

        return macd_bullish, macd_bearish

################################################################################
# CANDLESTICK PATTERNS
#
# These are the functions necesssary to process the data and verify if any
# candlestick pattern has been detected. The confirmation argument is used to
# guarantee that a candlestick with the expected behavior follows the pattern
################################################################################

################################################################################
# Description: Computes the body and range of the candlesticks of the data
# input
#
# Inputs:
# df: dataframe containing stock data
#
# Outputs:
# df_work: dataframe with additional columns containing body and range
################################################################################

def body_and_range(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:

    df_work = df.copy()

    df_work["Body"] = np.absolute(df_work["Open"] - df_work["Close"])
    df_work["Range"] = df_work["High"] - df_work["Low"]

    return df_work

################################################################################
# Description: Verifies if a confirmation candle, i.e., a candlestick with a
# behavior following the prediction from a previous candlestick pattern, follows
# immediately after the pattern has been detected
#
# Inputs:
# df: dataframe containing stock data
#
# Outputs:
# confirmation: boolean variable that is true if the pattern is confirmed
################################################################################

def pattern_confirmation_bullish(df: pd.core.frame.DataFrame) -> bool:

    confirmation_day = df.iloc[len(df)-1]
    last_pattern_day = df.iloc[len(df)-2]

    confirmation_1 = confirmation_day.Low > last_pattern_day.Low
    confirmation_2 = confirmation_day.Close > last_pattern_day.High
    confirmation_3 = confirmation_day.Close > confirmation_day.Open

    return confirmation_1 and confirmation_2 and confirmation_3

def pattern_confirmation_bearish(df: pd.core.frame.DataFrame) -> bool:

    confirmation_day = df.iloc[len(df)-1]
    last_pattern_day = df.iloc[len(df)-2]

    confirmation_1 = confirmation_day.High < last_pattern_day.High
    confirmation_2 = confirmation_day.Close < last_pattern_day.Low
    confirmation_3 = confirmation_day.Close < confirmation_day.Open

    return confirmation_1 and confirmation_2 and confirmation_3

################################################################################
# Description: Verifies if any of the reversal candlestick patterns implemented
# has been detected in the data input
#
# Inputs:
# df: dataframe containing stock data
# confirm: bool to include a verification of confirmation after the pattern
#
# Outputs:
# signal: boolean variable that is true if a pattern has been detected
################################################################################

def detect_bullish_pattern(df: pd.core.frame.DataFrame, confirm: bool = True) -> bool:

    signal_1 = pattern_bullish_pinbar(df, confirmation=confirm)
    signal_2 = pattern_white_soldier(df, confirmation=confirm)
    signal_3 = pattern_morning_star(df, confirmation=confirm)
    signal_4 = pattern_bullish_engulfing(df, confirmation=confirm)

    return signal_1 or signal_2 or signal_3 or signal_4

def detect_bearish_pattern(df: pd.core.frame.DataFrame, confirm: bool = True) -> bool:

    signal_1 = pattern_bearish_pinbar(df, confirmation=confirm)
    signal_2 = pattern_black_crow(df, confirmation=confirm)
    signal_3 = pattern_evening_star(df, confirmation=confirm)
    signal_4 = pattern_bearish_engulfing(df, confirmation=confirm)

    return signal_1 or signal_2 or signal_3 or signal_4

################################################################################
# Description: Verifies if a given candlestick pattern has been detected in the
# data input. The following candlestick patterns have been implemented:
#
## Bullish pinbar / Bearish pinbar
## One white soldier / One black crow
## Morning star / Evening star
## Bullish engulfin / bearish engulfing
#
# Inputs:
# df: dataframe containing stock data
# confirm: bool to include a verification of confirmation after the pattern
#
# Outputs:
# signal: boolean variable that is true if the pattern has been detected
################################################################################

def pattern_bullish_pinbar(df: pd.core.frame.DataFrame, confirmation: bool = True) -> bool:

    df_work = df.tail(2)
    df_work = body_and_range(df_work)

    if confirmation:

        pattern_day_1 = df_work.iloc[len(df_work)-2]
        confirmation_day = df_work.iloc[len(df_work)-1]

    else:

        pattern_day_1 = df_work.iloc[len(df_work)-1]

    requirement_1 = pattern_day_1.Body < pattern_day_1.Range / 3
    requirement_2 = min(pattern_day_1.Open, pattern_day_1.Close) > pattern_day_1.Low + (2/3) * pattern_day_1.Range

    if confirmation:

        requirement_3 = pattern_confirmation_bullish(df_work)
        return requirement_1 and requirement_2 and requirement_3

    else:

        return requirement_1 and requirement_2

def pattern_bearish_pinbar(df: pd.core.frame.DataFrame, confirmation: bool = True) -> bool:

    df_work = df.tail(2)
    df_work = body_and_range(df_work)

    if confirmation:

        pattern_day_1 = df_work.iloc[len(df_work)-2]
        confirmation_day = df_work.iloc[len(df_work)-1]

    else:

        pattern_day_1 = df_work.iloc[len(df_work)-1]

    requirement_1 = pattern_day_1.Body < pattern_day_1.Range / 3
    requirement_2 = max(pattern_day_1.Open, pattern_day_1.Close) < pattern_day_1.High - (2/3) * pattern_day_1.Range

    if confirmation:

        requirement_3 = pattern_confirmation_bearish(df_work)
        return requirement_1 and requirement_2 and requirement_3

    else:

        return requirement_1 and requirement_2

def pattern_white_soldier(df: pd.core.frame.DataFrame, confirmation: bool = True) -> bool:

    df_work = df.tail(3)

    if confirmation:

        pattern_day_1 = df_work.iloc[len(df_work)-3]
        pattern_day_2 = df_work.iloc[len(df_work)-2]
        confirmation_day = df_work.iloc[len(df_work)-1]

    else:

        pattern_day_1 = df_work.iloc[len(df_work)-2]
        pattern_day_2 = df_work.iloc[len(df_work)-1]

    requirement_1 = pattern_day_2.Low > pattern_day_1.Low
    requirement_2 = pattern_day_2.Open > pattern_day_1.Close
    requirement_3 = pattern_day_2.Close > pattern_day_1.High
    requirement_4 = pattern_day_1.Close < pattern_day_1.Open
    requirement_5 = pattern_day_2.Close > pattern_day_2.Open

    if confirmation:

        requirement_6 = pattern_confirmation_bullish(df_work)
        return requirement_1 and requirement_2 and requirement_3 and requirement_4 and requirement_5 and requirement_6

    else:

        return requirement_1 and requirement_2 and requirement_3 and requirement_4 and requirement_5

def pattern_black_crow(df: pd.core.frame.DataFrame, confirmation: bool = True) -> bool:

    df_work = df.tail(3)

    if confirmation:

        pattern_day_1 = df_work.iloc[len(df_work)-3]
        pattern_day_2 = df_work.iloc[len(df_work)-2]
        confirmation_day = df_work.iloc[len(df_work)-1]

    else:

        pattern_day_1 = df_work.iloc[len(df_work)-2]
        pattern_day_2 = df_work.iloc[len(df_work)-1]

    requirement_1 = pattern_day_2.High < pattern_day_1.High
    requirement_2 = pattern_day_2.Open < pattern_day_1.Close
    requirement_3 = pattern_day_2.Close < pattern_day_1.Low
    requirement_4 = pattern_day_1.Close > pattern_day_1.Open
    requirement_5 = pattern_day_2.Close < pattern_day_2.Open

    if confirmation:

        requirement_6 = pattern_confirmation_bearish(df_work)
        return requirement_1 and requirement_2 and requirement_3 and requirement_4 and requirement_5 and requirement_6

    else:

        return requirement_1 and requirement_2 and requirement_3 and requirement_4 and requirement_5

def pattern_morning_star(df: pd.core.frame.DataFrame, confirmation: bool = True) -> bool:

    df_work = df.tail(4)
    df_work = body_and_range(df_work)

    if confirmation:

        pattern_day_1 = df_work.iloc[len(df_work)-4]
        pattern_day_2 = df_work.iloc[len(df_work)-3]
        pattern_day_3 = df_work.iloc[len(df_work)-2]
        confirmation_day = df_work.iloc[len(df_work)-1]

    else:

        pattern_day_1 = df_work.iloc[len(df_work)-3]
        pattern_day_2 = df_work.iloc[len(df_work)-2]
        pattern_day_3 = df_work.iloc[len(df_work)-1]

    requirement_1 = pattern_day_2.Body < 0.5 * pattern_day_1.Body
    requirement_2 = pattern_day_2.Body < 0.5 * pattern_day_3.Body
    requirement_3 = max(pattern_day_2.Open,pattern_day_2.Close) < pattern_day_1.Close
    requirement_4 = max(pattern_day_2.Open,pattern_day_2.Close) < pattern_day_3.Open
    requirement_5 = pattern_day_1.Close < pattern_day_1.Open
    requirement_6 = pattern_day_3.Close > pattern_day_3.Open
    requirement_7 = pattern_day_3.Close > pattern_day_1.Close + 0.5 * pattern_day_1.Body

    if confirmation:

        requirement_8 = pattern_confirmation_bullish(df_work)
        return requirement_1 and requirement_2 and requirement_3 and requirement_4 and requirement_5 and requirement_6 and requirement_7 and requirement_8

    else:

        return requirement_1 and requirement_2 and requirement_3 and requirement_4 and requirement_5 and requirement_6 and requirement_7

def pattern_evening_star(df: pd.core.frame.DataFrame, confirmation: bool = True) -> bool:

    df_work = df.tail(4)
    df_work = body_and_range(df_work)

    if confirmation:

        pattern_day_1 = df_work.iloc[len(df_work)-4]
        pattern_day_2 = df_work.iloc[len(df_work)-3]
        pattern_day_3 = df_work.iloc[len(df_work)-2]
        confirmation_day = df_work.iloc[len(df_work)-1]

    else:

        pattern_day_1 = df_work.iloc[len(df_work)-3]
        pattern_day_2 = df_work.iloc[len(df_work)-2]
        pattern_day_3 = df_work.iloc[len(df_work)-1]

    requirement_1 = pattern_day_2.Body < 0.5 * pattern_day_1.Body
    requirement_2 = pattern_day_2.Body < 0.5 * pattern_day_3.Body
    requirement_3 = min(pattern_day_2.Open,pattern_day_2.Close) > pattern_day_1.Close
    requirement_4 = min(pattern_day_2.Open,pattern_day_2.Close) > pattern_day_3.Open
    requirement_5 = pattern_day_1.Close > pattern_day_1.Open
    requirement_6 = pattern_day_3.Close < pattern_day_3.Open
    requirement_7 = pattern_day_3.Close < pattern_day_1.Close - 0.5 * pattern_day_1.Body

    if confirmation:

        requirement_8 = pattern_confirmation_bearish(df_work)
        return requirement_1 and requirement_2 and requirement_3 and requirement_4 and requirement_5 and requirement_6 and requirement_7 and requirement_8

    else:

        return requirement_1 and requirement_2 and requirement_3 and requirement_4 and requirement_5 and requirement_6 and requirement_7

def pattern_bullish_engulfing(df: pd.core.frame.DataFrame, confirmation: bool = True) -> bool:

    df_work = df.tail(3)

    if confirmation:

        pattern_day_1 = df_work.iloc[len(df_work)-3]
        pattern_day_2 = df_work.iloc[len(df_work)-2]
        confirmation_day = df_work.iloc[len(df_work)-1]

    else:

        pattern_day_1 = df_work.iloc[len(df_work)-2]
        pattern_day_2 = df_work.iloc[len(df_work)-1]

    requirement_1 = pattern_day_1.Close < pattern_day_1.Open
    requirement_2 = pattern_day_2.Close > pattern_day_2.Open
    requirement_3 = pattern_day_2.Open < pattern_day_1.Close
    requirement_4 = pattern_day_2.Close > pattern_day_1.High

    if confirmation:

        requirement_5 = pattern_confirmation_bullish(df_work)
        return requirement_1 and requirement_2 and requirement_3 and requirement_4 and requirement_5

    else:

        return requirement_1 and requirement_2 and requirement_3 and requirement_4

def pattern_bearish_engulfing(df: pd.core.frame.DataFrame, confirmation: bool = True) -> bool:

    df_work = df.tail(3)

    if confirmation:

        pattern_day_1 = df_work.iloc[len(df_work)-3]
        pattern_day_2 = df_work.iloc[len(df_work)-2]
        confirmation_day = df_work.iloc[len(df_work)-1]

    else:

        pattern_day_1 = df_work.iloc[len(df_work)-2]
        pattern_day_2 = df_work.iloc[len(df_work)-1]

    requirement_1 = pattern_day_1.Close > pattern_day_1.Open
    requirement_2 = pattern_day_2.Close < pattern_day_2.Open
    requirement_3 = pattern_day_2.Open > pattern_day_1.Close
    requirement_4 = pattern_day_2.Close < pattern_day_1.Low

    if confirmation:

        requirement_5 = pattern_confirmation_bearish(df_work)
        return requirement_1 and requirement_2 and requirement_3 and requirement_4 and requirement_5

    else:

        return requirement_1 and requirement_2 and requirement_3 and requirement_4

################################################################################
# STRATEGY SCREENERS
#
# These are the functions that filter through lists of tickers with the goal
# of finding which stocks currently satisfy a given set of requirements. Each
# screener follows its own rules to filter through the stocks.
################################################################################

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

def screener_MACD_crossover_long(tickers: list) -> list:

    if not Market_analysis.long_bias:
        print("It is not recommended to follow this strategy in the current market condition.")

    signal_list = []

    for ticker in tickers:

            try:
                df_ticker = yf.Ticker(ticker).history(period="200d").drop(["Dividends","Stock Splits"], axis = 1)

                signal, _ = check_MACD(df_ticker, 12, 26, 9, True)

                if signal:
                    signal_list.append(ticker)

            except:
                continue

    Stock_lists.screened_stocks = signal_list

    return signal_list

def screener_MACD_crossover_short(tickers: list) -> list:

    if not Market_analysis.short_bias:
        print("It is not recommended to follow this strategy in the current market condition.")

    signal_list = []

    for ticker in tickers:

            try:
                df_ticker = yf.Ticker(ticker).history(period="200d").drop(["Dividends","Stock Splits"], axis = 1)

                _, signal = check_MACD(df_ticker, 12, 26, 9, True)

                if signal:
                    signal_list.append(ticker)

            except:
                continue

    Stock_lists.screened_stocks = signal_list

    return signal_list


# MAYBE CREATE ANOTHER PACKAGE FOR VISUALIZATION?

def closing_data(tickers: list) -> pd.core.frame.DataFrame:

    df_work = pd.DataFrame()

    for ticker in tickers:

        df_ticker = yf.Ticker(ticker).history(period="200d").drop(["Dividends","Stock Splits"], axis = 1)

        df_work[ticker] = df_ticker.Close

    return df_work

def compare_stocks(tickers: list):

    df_work = closing_data(tickers)

    df_work = np.log(df_work / df_work.shift(1)).cumsum().apply(np.exp)

    plt.figure(figsize=(10, 5))

    for ticker in tickers:

        plt.plot(df_work[ticker], label=ticker)

    plt.grid(linestyle=':')
    plt.xlabel("Date")
    plt.ylabel("Cumulative log returns")
    plt.legend()
    plt.show()

################################################################################
# ENTRY RULES
#
# These are the functions necessary to calculate the prices at which to place
# the buy and sell (both stop and limit) orders for a given stock after a signal
# has been detected
################################################################################

################################################################################
# Description: prints the details of a given order, namely the prices for the
# buy/sell orders, and the position sizing details
#
# Inputs:
# buy_stop: price at which to place buy stop order
# buy_limit: price at which to place buy limit order
# sell_stop: price at which to place sell stop order
# sell_limit: price at which to place sell limit order
# number_shares: number of shares to buy
# position_size: total size of the position
# total_risk: total risk of the position
#
# Outputs: None
################################################################################

def print_order_details(buy_stop: float, buy_limit: float, sell_stop: float, sell_limit: float, number_shares: float, position_size: float, total_risk: float):

    print("The buy and sell orders should be placed at the following prices:")
    print(f"Buy stop: {buy_stop}")
    print(f"Buy limit: {buy_limit}")
    print(f"Sell stop: {sell_stop}")
    print(f"Sell limit: {sell_limit}")
    print(" ")
    print(f"Number of shares to buy: {number_shares}")
    print(f"Total position size: {position_size}")
    print(f"Total risk of position: {total_risk}")
    print(" ")

################################################################################
# Description: calculates the adequate price margin to be considered when
# placing an order for a stock with a given closing price
#
# Inputs:
# price: latest closing price of the stock to be bought/sold
#
# Outputs:
# order margin: price margin to be used in the orders
################################################################################

def calculate_margin(price: float) -> float:

    if price <= 5:
        order_margin = 0.01

    elif price <= 10:
        order_margin = 0.02

    elif price <= 50:
        order_margin = 0.03

    elif price <= 100:
        order_margin = 0.05

    else:
        order_margin = 0.1

    return order_margin

################################################################################
# Description: calculates the prices at which the buy/sell stop/limit orders
# should be placed based on the prices of the latest two days and the ATR
#
# Inputs:
# df: dataframe containing stock data
# profit_ratio: number of times the profit should surpass the risk
#
# Outputs:
# buy_stop: price at which to place buy stop order
# buy_limit: price at which to place buy limit order
# sell_stop: price at which to place sell stop order
# sell_limit: price at which to place sell limit order
################################################################################

def entry_rules_long(df: pd.core.frame.DataFrame, profit_ratio: float) -> (float, float, float, float):

    df_work = df.copy()

    df_work = average_true_range(df_work)
    df_work["1D range"] = df_work.High - df_work.Low
    df_work["2D range"] = df_work.High - df_work.shift().Low

    df_work = df_work.tail(2)

    order_margin = calculate_margin(df_work.iloc[1].Close)

    buy_stop = df_work.iloc[1].High + order_margin
    buy_limit = df_work.iloc[1].High + 2 * order_margin

    if df_work.iloc[1]["1D range"] < df_work.iloc[1].ATR and df_work.iloc[1]["2D range"] < df_work.iloc[1].ATR * 1.5:
        sell_stop = df_work.iloc[0].Low - order_margin
    else:
        sell_stop = df_work.iloc[1].Low - order_margin

    sell_limit = buy_stop + profit_ratio * (buy_stop - sell_stop)

    return buy_stop, buy_limit, sell_stop, sell_limit

def entry_rules_short(df: pd.core.frame.DataFrame, profit_ratio: float) -> (float, float, float, float):

    df_work = df.copy()

    df_work = average_true_range(df_work)
    df_work["1D range"] = df_work.High - df_work.Low
    df_work["2D range"] = df_work.shift().High - df_work.Low

    df_work = df_work.tail(2)

    order_margin = calculate_margin(df_work.iloc[1].Close)

    sell_stop = df_work.iloc[1].Low - order_margin
    sell_limit = df_work.iloc[1].Low - 2 * order_margin

    if df_work.iloc[1]["1D range"] < df_work.iloc[1].ATR and df_work.iloc[1]["2D range"] < df_work.iloc[1].ATR * 1.5:
        buy_stop = df_work.iloc[0].High + order_margin
    else:
        buy_stop = df_work.iloc[1].High + order_margin

    buy_limit = sell_stop - profit_ratio * (buy_stop - sell_stop)

    return buy_stop, buy_limit, sell_stop, sell_limit

################################################################################
# Description: calculates the number of shares that should be bought/sell to
# obtain a position with a certain risk and profit ratios for a given capital
#
# Inputs:
# df: dataframe containing stock data
# total_capital: total capital available for the position
# profit_ratio: number of times the profit should surpass the risk
# risk_ratio: percentage of the total capital to be risked
#
# Outputs:
# number_shares: number of shares to buy
# position_size: total size of the position
# total_risk: total risk of the position
################################################################################

def position_sizing_long(df: pd.core.frame.DataFrame, total_capital: float, profit_ratio: float, risk_ratio: float) -> (float, float, float):

    df_work = df.copy()

    buy_stop, _, sell_stop, _ = entry_rules_long(df_work, profit_ratio)

    number_shares = int(total_capital * 0.01 * risk_ratio / (buy_stop - sell_stop))
    position_size = number_shares * buy_stop
    total_risk = 0.01 * risk_ratio * total_capital

    return number_shares, position_size, total_risk

def position_sizing_short(df: pd.core.frame.DataFrame, total_capital: float, profit_ratio: float, risk_ratio: float) -> (float, float, float):

    df_work = df.copy()

    buy_stop, _, sell_stop, _ = entry_rules_short(df_work, profit_ratio)

    number_shares = int(total_capital * 0.01 * risk_ratio / (buy_stop - sell_stop))
    position_size = number_shares * sell_stop
    total_risk = 0.01 * risk_ratio * total_capital

    return number_shares, position_size, total_risk

################################################################################
# Description: calculates the prices at which the buy/sell stop/limit orders
# should be placed, as well as the number of shares that should be bought/sold,
# the total position size, and the total risk. Then, these details are printed
#
# Inputs:
# ticker: ticker for the stock under analysis
# total_capital: total capital available for the position
# profit_ratio: number of times the profit should surpass the risk
# risk_ratio: percentage of the total capital to be risked
#
# Outputs: None
################################################################################

def order_details_long(ticker: str, total_capital: float, profit_ratio: float, risk_ratio: float):

    df_work = yf.Ticker(ticker).history(period="400d").drop(["Dividends","Stock Splits"], axis = 1)

    buy_stop, buy_limit, sell_stop, sell_limit = entry_rules_long(df_work, profit_ratio)
    number_shares, position_size, total_risk = position_sizing_long(df_work, total_capital, profit_ratio, risk_ratio)

    print_order_details(buy_stop, buy_limit, sell_stop, sell_limit, number_shares, position_size, total_risk)

def order_details_short(ticker: str, total_capital: float, profit_ratio: float, risk_ratio: float):

    df_work = yf.Ticker(ticker).history(period="400d").drop(["Dividends","Stock Splits"], axis = 1)

    buy_stop, buy_limit, sell_stop, sell_limit = entry_rules_short(df_work, profit_ratio)
    number_shares, position_size, total_risk = position_sizing_short(df_work, total_capital, profit_ratio, risk_ratio)

    print_order_details(buy_stop, buy_limit, sell_stop, sell_limit, number_shares, position_size, total_risk)

################################################################################
# Description: iterates through a list of tickers, ideally those who have been
# previously screened, and returns the entry rules for each of the tickers in
# the list.
#
# Inputs:
# tickers: list containing stock tickers
# total_capital: total capital available for the position
# profit_ratio: number of times the profit should surpass the risk
# risk_ratio: percentage of the total capital to be risked
#
# Outputs: None
################################################################################

def order_details_long_list(tickers: list, total_capital: float, profit_ratio: float, risk_ratio: float):

    for ticker in tickers:

        order_details_long(ticker, total_capital, profit_ratio, risk_ratio)

def order_details_short_list(tickers: list, total_capital: float, profit_ratio: float, risk_ratio: float):

    for ticker in tickers:

        order_details_short(ticker, total_capital, profit_ratio, risk_ratio)

################################################################################
# BACKTESTING
#
# These are the functions necessary to backtest the strategies implemented and
# calculate their success rate
################################################################################

def backtest_MACD_crossover_ticker(ticker: str, initial_capital: float, risk_ratio: float, profit_ratio: float, current_win_streak: int = 0,  current_loss_streak: int = 0,  current_longest_win_streak: int = 0,  current_longest_loss_streak: int = 0) -> (int, int, int, float, int, int, int, int):

    total_capital = initial_capital
    number_trades = 0

    wins = 0
    losses = 0

    win_streak = current_win_streak
    loss_streak = current_loss_streak

    longest_win_streak = current_longest_win_streak
    longest_loss_streak = current_longest_loss_streak


    longing = False
    shorting = False
    buy_order = False
    sell_order = False

    df_ticker = yf.Ticker(ticker).history(period="400d").drop(["Dividends","Stock Splits"], axis = 1)

    for i in range(200):

        try:

            if longing:

                if df_ticker.shift(200-i).iloc[400-1].Close > sell_limit:

                    longing = False
                    number_trades += 1
                    wins += 1
                    total_capital += number_shares * (sell_limit - buy_stop)

                    if loss_streak > longest_loss_streak:
                        longest_loss_streak = loss_streak

                    win_streak += 1
                    loss_streak = 0


                elif df_ticker.shift(200-i).iloc[400-1].Close < sell_stop:

                    longing = False
                    number_trades += 1
                    losses += 1
                    total_capital -= number_shares * (buy_stop - sell_stop)

                    if win_streak > longest_win_streak:
                        longest_win_streak = win_streak

                    win_streak = 0
                    loss_streak += 1

            if buy_order and i > order_instant + 2:
                buy_order = False

            if buy_order:

                if df_ticker.shift(200-i).iloc[400-1].Close > buy_stop:

                    buy_order = False
                    longing = True

            if not longing and not shorting and not buy_order and not sell_order:

                macd_signal, _ = check_MACD(df_ticker.shift(200-i), 12, 26, 9, True)

                if macd_signal:

                    buy_order = True
                    buy_stop, buy_limit, sell_stop, sell_limit = entry_rules_long(df_ticker.shift(200-i), profit_ratio)
                    number_shares, _, _ = position_sizing_long(df_ticker.shift(200-i), total_capital, profit_ratio, risk_ratio)
                    order_instant = i

        except:
            continue

    return number_trades, wins, losses, total_capital, win_streak, loss_streak, longest_win_streak, longest_loss_streak

def backtest_MACD_crossover(tickers: list, initial_capital: float, risk_ratio: float, profit_ratio: float):

    total_capital = initial_capital
    number_trades = 0
    wins = 0
    losses = 0

    win_streak = 0
    loss_streak = 0

    longest_win_streak = 0
    longest_loss_streak = 0

    for ticker in tickers:

        ticker_trades, ticker_wins, ticker_losses, ticker_capital, new_win_streak, new_loss_streak, new_longest_win_streak, new_longest_loss_streak = backtest_MACD_crossover_ticker(ticker, total_capital, risk_ratio, profit_ratio, win_streak, loss_streak, longest_win_streak, longest_loss_streak)
        number_trades += ticker_trades
        wins += ticker_wins
        losses += ticker_losses
        total_capital = ticker_capital
        win_streak = new_win_streak
        loss_streak = new_loss_streak
        longest_win_streak = new_longest_win_streak
        longest_loss_streak = new_longest_loss_streak

    percentage_profit = (total_capital - initial_capital) / initial_capital * 100
    success_rate = wins / number_trades * 100
    average_profit = (profit_ratio * wins - risk_ratio * losses) / number_trades

    print("Results:")
    print(" ")
    print(f"Number of trades: {number_trades}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Longest winning streak: {longest_win_streak}")
    print(f"Longest losing streak: {longest_loss_streak}")
    print(f"Success_rate: {success_rate}%")
    print(f"Average profit per trade: {average_profit}%")
    print(" ")
    print(f"Initial capital: {initial_capital}")
    print(f"Final capital: {total_capital}")
    print(f"Profit percentage: {percentage_profit.round(1)}%")

################################################################################
# DIVERSIFICATION
#
# These are the functions necessary to verify if there exists any type of
# correlation between the stocks that have been screened. These functions allow
# one to check if the screened stocks belong to the same sectors and also if
# there has been a correlation in the closing prices in psat data
################################################################################

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
