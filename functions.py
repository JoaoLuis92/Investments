import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

#####################
# GENERAL FUNCTIONS #
#####################

# These are the functions necessary for importaing data, filtering through stock
# lists, and analyzing the general behavior of the market to help deciding which
# strategies are the most adequate at a given time.

# Imports the tickers from the Nasdaq and New York Stock Exchange from data files and returns them in a list

def import_stock_tickers():

    tickers_nasdaq = pd.read_csv("tickers_nasdaq.csv").Ticker.to_list()
    tickers_nyse = pd.read_csv("tickers_nyse.csv").Ticker.to_list()

    return tickers_nasdaq + tickers_nyse

# Filters through a list of tickers and returns a list of the tickers corresponding to high volume stocks

def produce_high_volume_list(tickers, volume = 200000):

    high_volume_list = []

    for ticker in tickers:

        try:
            if yf.Ticker(ticker).history(period="3mo").Volume.mean() > volume:
                high_volume_list.append(ticker)

        except:
            continue

    return high_volume_list

# Calculates the relative change in the price of a given stock

def stock_relative_change(df, period):

    return (df.tail(period).iloc[period-1].Close - df.tail(period).iloc[0].Close) / df.tail(period).iloc[0].Close

# Filters through a list of tickers and returns two lists corresponding to strong and weak stocks in comparison with the S&P500

def produce_relative_strength_lists(tickers, period = 30):

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

# Verifies if the market is currently on an uptrend or a downtrend using moving averaged

def verify_market_trend():

    spx_df = yf.Ticker("^GSPC").history(period="200d").drop(["Dividends","Stock Splits"], axis = 1)

    for period in [50, 100, 200]:
        simple_moving_average(spx_df, period)

    for period in [20, 40]:
        exponential_moving_average(spx_df, period)

    short_uptrend = (spx_df.tail(1).Close > spx_df.tail(1).EMA20).bool() and (spx_df.tail(1).EMA20 > spx_df.tail(1).EMA40).bool()
    short_downtrend = (spx_df.tail(1).Close < spx_df.tail(1).EMA20).bool() and (spx_df.tail(1).EMA20 < spx_df.tail(1).EMA40).bool()

    long_uptrend = (spx_df.tail(1).Close > spx_df.tail(1).SMA50).bool() and (spx_df.tail(1).SMA50 > spx_df.tail(1).SMA100).bool() and (spx_df.tail(1).SMA100 > spx_df.tail(1).SMA200).bool()
    long_downtrend = (spx_df.tail(1).Close < spx_df.tail(1).SMA50).bool() and (spx_df.tail(1).SMA50 < spx_df.tail(1).SMA100).bool() and (spx_df.tail(1).SMA100 < spx_df.tail(1).SMA200).bool()

    uptrend = short_uptrend and long_uptrend
    downtrend = short_downtrend and long_downtrend

    return uptrend, downtrend

# Verifies is the market is overbought or oversold using the RSI stochastic indicators

def verify_market_stochastic():

    spx_df = yf.Ticker("^GSPC").history(period="200d").drop(["Dividends","Stock Splits"], axis = 1)

    full_stochastic(spx_df,5,3,3)

    oversold = (spx_df.tail(1)["Fast K"] < 20).bool() and (spx_df.tail(1)["Slow K"] < 20).bool()
    overbought = (spx_df.tail(1)["Fast K"] > 80).bool() and (spx_df.tail(1)["Slow K"] > 80).bool()

    return oversold, overbought

# Analyzes the market, prints the result, and returns a list of recommended stocks to Analyzes

def analyze_market(strong_stocks_list, weak_stocks_list):

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

# Produces the plots of market trend and stockastic indicators

def plot_market_conditions():

    spx_df = yf.Ticker("^GSPC").history(period="200d").drop(["Dividends","Stock Splits"], axis = 1)

    for period in [50, 100, 200]:
        simple_moving_average(spx_df, period)

    for period in [20, 40]:
        exponential_moving_average(spx_df, period)

    full_stochastic(spx_df,5,3,3)

    # Plots the SPX and respective moving averages

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

# Plots the RSI indicator of the SPX

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

########################
# FINANCIAL INDICATORS #
########################

# These are the functions necessary to process the data and compute the values
# of different financial indicators. Furthermore, some of these functions are
# used to apply the financial indicator to the data and verify is a signal has
# been detected.

# Computes the simple moving average of a given period

def simple_moving_average(df, period):

    df["SMA"+str(period)] = df["Close"].rolling(period).mean()

# Computes the exponential moving average of a given period

def exponential_moving_average(df, period):

    df["EMA"+str(period)] = df['Close'].ewm(span = period).mean()

# Computes the full stochastic indicator (RSI) for the given periods

def full_stochastic(df, fk_period, sk_period, sd_period):

    fast_k_list = []

    for i in range(len(df)):
        low = df.iloc[i]['Low']
        high = df.iloc[i]['High']

        if i >= fk_period:

            for n in range(fk_period):

                if df.iloc[i-n]['High'] >= high:
                    high = df.iloc[i-n]['High']
                elif df.iloc[i-n]['Low'] < low:
                    low = df.iloc[i-n]['Low']
        if high != low:
            fast_k = 100 * (df.iloc[i]['Close'] - low) / (high - low)
        else:
            fast_k = 0

        fast_k_list.append(fast_k)

    df["Fast K"] = fast_k_list
    df["Slow K"] = df["Fast K"].rolling(sk_period).mean()
    df["Slow D"] = df["Slow K"].rolling(sd_period).mean()

########################
# CANDLESTICK PATTERNS #
########################

# These are the functions necesssary to process the data and verify if any
# candlestick pattern has been detected. The confirmation argument is used to
# guarantee that a candlestick with the expected behavior follows the pattern

# Computes the body and the range of the candlesticks

def body_and_range(df):

    df["Body"] = np.absolute(df["Open"] - df["Close"])
    df["Range"] = df["High"] - df["Low"]

# Verifies if the bullish pattern has been confirmed by the following candlestick

def pattern_confirmation_bullish(df):

    confirmation_day = df.iloc[len(df)-1]
    last_pattern_day = df.iloc[len(df)-2]

    confirmation_1 = confirmation_day.Low > last_pattern_day.Low
    confirmation_2 = confirmation_day.Close > last_pattern_day.High
    confirmation_3 = confirmation_day.Close > confirmation_day.Open

    return confirmation_1 and confirmation_2 and confirmation_3

# Verifies if the bearish pattern has been confirmed by the following candlestick

def pattern_confirmation_bearish(df):

    confirmation_day = df.iloc[len(df)-1]
    last_pattern_day = df.iloc[len(df)-2]

    confirmation_1 = confirmation_day.High < last_pattern_day.High
    confirmation_2 = confirmation_day.Close < last_pattern_day.Low
    confirmation_3 = confirmation_day.Close < confirmation_day.Open

    return confirmation_1 and confirmation_2 and confirmation_3

# Verifies if any of the bullish candlestick patterns has been detected

def detect_bullish_pattern(df, confirm=True):

    signal_1 = pattern_bullish_pinbar(df, confirmation=confirm)
    signal_2 = pattern_white_soldier(df, confirmation=confirm)
    signal_3 = pattern_morning_star(df, confirmation=confirm)
    signal_4 = pattern_bullish_engulfing(df, confirmation=confirm)

    return signal_1 or signal_2 or signal_3 or signal_4

# Verifies if any of the bearish candlestick patterns has been detected

def detect_bearish_pattern(df, confirm=True):

    signal_1 = pattern_bearish_pinbar(df, confirmation=confirm)
    signal_2 = pattern_black_crow(df, confirmation=confirm)
    signal_3 = pattern_evening_star(df, confirmation=confirm)
    signal_4 = pattern_bearish_engulfing(df, confirmation=confirm)

    return signal_1 or signal_2 or signal_3 or signal_4

# The following functions analyze each of the candlestick patterns independently

## Bullish pinbar

def pattern_bullish_pinbar(df, confirmation=True):

    df_work = df.tail(2)
    body_and_range(df_work)

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

## Bearish pinbar

def pattern_bearish_pinbar(df, confirmation=True):

    df_work = df.tail(2)
    body_and_range(df_work)

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

## One white soldier

def pattern_white_soldier(df, confirmation=True):

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

## One black crow

def pattern_black_crow(df, confirmation=True):

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

## Morning star

def pattern_morning_star(df, confirmation=True):

    df_work = df.tail(4)
    body_and_range(df_work)

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

## Evening star

def pattern_evening_star(df, confirmation=True):

    df_work = df.tail(4)
    body_and_range(df_work)

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

## Bullish engulfing

def pattern_bullish_engulfing(df, confirmation=True):

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

## Bearish engulfing

def pattern_bearish_engulfing(df, confirmation=True):

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
