#####################
# NECESSARY IMPORTS #
#####################

import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

################################################################################
# GENERAL FUNCTIONS
#
#
# These are the functions necessary for importaing data, filtering through
# stock lists, and analyzing the general behavior of the market to help
# deciding which strategies are the most adequate at a given time.
################################################################################

################################################################################
# Description: Imports the tickers from the Nasdaq and New York Stock Exchange
# from data files and returns a list containing these tickers
#
# Inputs: None
#
# Outputs:
# list_of_tickers: list containing tickers
################################################################################

def import_stock_tickers() -> list:

    tickers_nasdaq = pd.read_csv("tickers_nasdaq.csv").Ticker.to_list()
    tickers_nyse = pd.read_csv("tickers_nyse.csv").Ticker.to_list()

    return tickers_nasdaq + tickers_nyse

################################################################################
# Description: Filters through a list of stocks to select those with a volume
# larger than a certain threshold of interest
#
# Inputs:
# tickers: list of tickers to be filtered
# volume: minimum volume for a high volume stock
#
# Outputs:
# high_volume_list: list containing the tickers of high volume stocks
################################################################################

def produce_high_volume_list(tickers: list, volume: int = 200000) -> list:

    high_volume_list = []

    for ticker in tickers:

        try:
            if yf.Ticker(ticker).history(period="3mo").Volume.mean() > volume:
                high_volume_list.append(ticker)

        except:
            continue

    return high_volume_list

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

def stock_relative_change(df: pd.core.frame.DataFrame, period: int) -> float:

    return (df.tail(period).iloc[period-1].Close - df.tail(period).iloc[0].Close) / df.tail(period).iloc[0].Close

################################################################################
# Description: Filters through a list of tickers, compares the relative change
# of each ticker with the relative change of the S&P500, and returns two lists
# containing the tickers that are stronger and weaker with respect to the S&P500
#
# Inputs:
# tickers: list of tickers to be filtered
# period: time interval to be analyzed
#
# Outputs:
# relative_change: value of the relative change of the stock
################################################################################

def produce_relative_strength_lists(tickers: list, period: int = 30) -> (list, list):

    strong_stocks_list = []
    weak_stocks_list = []

    spx_df = yf.Ticker("^GSPC").history(period="200d").drop(["Dividends","Stock Splits"], axis = 1)
    spx_change = stock_relative_change(spx_df, period)

    for ticker in tickers:

        try:
            stock_data = yf.Ticker(ticker).history(period="3mo")
            stock_change = stock_relative_change(stock_data, period)

            if stock_change > spx_change:
                strong_stocks_list.append(ticker)
            else:
                weak_stocks_list.append(ticker)
        except:
            continue

    return strong_stocks_list, weak_stocks_list

################################################################################
# Description: Verifies if the market is currently on an uptrend or downtrend
# by comparing moving averages with different periods and returns two boolean
# variables describing the trend
#
# Inputs: None
#
# Outputs:
# uptrend: boolean variable that is true if the market is uptrending
# downtrend: boolean variable that is true if the market is downtrending
################################################################################

def verify_market_trend() -> (bool, bool):

    spx_df = yf.Ticker("^GSPC").history(period="200d").drop(["Dividends","Stock Splits"], axis = 1)

    for period in [50, 100, 200]:
        spx_df = simple_moving_average(spx_df, period)

    for period in [20, 40]:
        spx_df = exponential_moving_average(spx_df, period)

    short_uptrend = (spx_df.tail(1).Close > spx_df.tail(1).EMA20).bool() and (spx_df.tail(1).EMA20 > spx_df.tail(1).EMA40).bool()
    short_downtrend = (spx_df.tail(1).Close < spx_df.tail(1).EMA20).bool() and (spx_df.tail(1).EMA20 < spx_df.tail(1).EMA40).bool()

    long_uptrend = (spx_df.tail(1).Close > spx_df.tail(1).SMA50).bool() and (spx_df.tail(1).SMA50 > spx_df.tail(1).SMA100).bool() and (spx_df.tail(1).SMA100 > spx_df.tail(1).SMA200).bool()
    long_downtrend = (spx_df.tail(1).Close < spx_df.tail(1).SMA50).bool() and (spx_df.tail(1).SMA50 < spx_df.tail(1).SMA100).bool() and (spx_df.tail(1).SMA100 < spx_df.tail(1).SMA200).bool()

    uptrend = short_uptrend and long_uptrend
    downtrend = short_downtrend and long_downtrend

    return uptrend, downtrend

################################################################################
# Description: Verifies if the market is currently oversold or overbought by
# analyzing the full stochastic oscillator with periods 5, 3, 3
#
# Inputs: None
#
# Outputs:
# oversold: boolean variable that is true if the market is oversold
# overbought: boolean variable that is true if the market is overbought
################################################################################

def verify_market_stochastic() -> (bool, bool):

    spx_df = yf.Ticker("^GSPC").history(period="200d").drop(["Dividends","Stock Splits"], axis = 1)

    spx_df = full_stochastic(spx_df,5,3,3)

    oversold = (spx_df.tail(1)["Fast K"] < 20).bool() and (spx_df.tail(1)["Slow K"] < 20).bool()
    overbought = (spx_df.tail(1)["Fast K"] > 80).bool() and (spx_df.tail(1)["Slow K"] > 80).bool()

    return oversold, overbought

################################################################################
# Description: Analyzes the market trend and the price status through the full
# stochastic oscillator, informs the user which type of positions (long/short)
# are recommended in the current market conditions, and returns a list that
# contains either strong or weak stocks depending on the market being bullish
# or bearish, respectively
#
# Inputs:
# strong_stocks_list: list containing tickers of strong stocks
# weak_stocks_list: list containing tickers of weak stocks
#
# Outputs:
# market_bias: string containing either "Long" or "Short" describing market bias
# stocks_list: list containing either strong or weak stocks depending on bias
################################################################################

def analyze_market(strong_stocks_list: list, weak_stocks_list: list) -> (str, list):

    uptrend, downtrend = verify_market_trend()
    oversold, overbought = verify_market_stochastic()

    long_bias = uptrend and not overbought
    short_bias = downtrend and not oversold

    if long_bias:
        print("Market is currently good for long positions.")
        return "Long", strong_stocks_list

    elif short_bias:
        print("Market is currently good for short positions.")
        return "Short", weak_stocks_list

    else:
        print("Market is currently indecisive.")
        return "Indecisive", []

################################################################################
# Description: Produces two plots of the current market conditions, the first
# showing the closing price alongside with four moving averages of interest, and
# the second showing the fast K and slow K stochastic oscillators
#
# Inputs: None
#
# Outputs: None
################################################################################

def plot_market_conditions() -> None:

    spx_df = yf.Ticker("^GSPC").history(period="200d").drop(["Dividends","Stock Splits"], axis = 1)

    for period in [50, 100, 200]:
        spx_df = simple_moving_average(spx_df, period)

    for period in [20, 40]:
        spx_df = exponential_moving_average(spx_df, period)

    spx_df = full_stochastic(spx_df,5,3,3)

    # First plot: SPX and moving averages

    plt.figure(figsize=(10, 5))
    plt.plot(spx_df.tail(30).Close, 'k.-', label='S&P500')
    plt.plot(spx_df.tail(30).EMA20, 'r--', label='EMA20')
    plt.plot(spx_df.tail(30).EMA40, 'b--', label='EMA40')
    plt.plot(spx_df.tail(30).SMA50, 'g-', label='SMA50')
    plt.plot(spx_df.tail(30).SMA100, 'm-', label='SMA100')
    plt.grid(linestyle=':')
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

    # Second plot: fast K and slow K stochastic

    plt.figure(figsize=(10, 5))
    plt.plot(spx_df.tail(30)["Fast K"], 'k-', label='Fast %K')
    plt.plot(spx_df.tail(30)["Slow K"], 'r-', label='Slow %K')
    plt.axhline(y=80, color='g', linestyle='--')
    plt.axhline(y=20, color='g', linestyle='--')
    plt.grid(linestyle=':')
    plt.ylim(0,100)
    plt.xlabel("Date")
    plt.ylabel("Full Stochastic")
    plt.legend()
    plt.show()

    return None

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

    df_work = df.copy()
    df_work["SMA"+str(period)] = df_work["Close"].rolling(period).mean()

    return df_work

def exponential_moving_average(df: pd.core.frame.DataFrame, period: int) -> pd.core.frame.DataFrame:

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

    df_work = df.copy()

    df_work = exponential_moving_average(df_work, macd_period_1)
    df_work = exponential_moving_average(df_work, macd_period_2)

    df_work["MACD"] = df_work["EMA" + str(macd_period_1)] - df_work["EMA" + str(macd_period_2)]
    df_work["Signal"] = (df_work['MACD']).ewm(span = signal_period).mean()

    df_work = df_work.drop(columns = ["EMA" + str(macd_period_1), "EMA" + str(macd_period_2)])

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
## Basic: trend, stochastic, MACD, and candlestick pattern
#
# Inputs:
# tickers: list of tickers to be screened
#
# Outputs:
# signal_list: list of tickers for which a signal has been detected
################################################################################

def screener_basic_long(tickers: list) -> list:

    signal_list = []

    for ticker in tickers:

            try:
                df_ticker = yf.Ticker(ticker).history(period="200d").drop(["Dividends","Stock Splits"], axis = 1)

                trend_signal, _ = check_trend(df_ticker)
                stochastic_signal, _ = check_stochastic(df_ticker, 5, 5, 3)
                macd_signal, _ = check_MACD(df_ticker, 12, 26, 9, False)
                pattern_signal = detect_bullish_pattern(df_ticker, confirm = True)

                signal = trend_signal and stochastic_signal and macd_signal and pattern_signal

                if signal:
                    signal_list.append(ticker)

            except:
                continue

    return signal_list

def screener_basic_short(tickers: list) -> list:

    signal_list = []

    for ticker in tickers:

            try:
                df_ticker = yf.Ticker(ticker).history(period="200d").drop(["Dividends","Stock Splits"], axis = 1)

                _, trend_signal = check_trend(df_ticker)
                _, stochastic_signal = check_stochastic(df_ticker, 5, 5, 3)
                _, macd_signal = check_MACD(df_ticker, 12, 26, 9, False)
                pattern_signal = detect_bearish_pattern(df_ticker, confirm = True)

                signal = trend_signal and stochastic_signal and macd_signal and pattern_signal

                if signal:
                    signal_list.append(ticker)

            except:
                continue

    return signal_list
