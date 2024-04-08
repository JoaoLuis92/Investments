#####################
# NECESSARY IMPORTS #
#####################

import numpy as np
import yfinance as yf
import pandas as pd

################################################################################
# FINANCIAL INDICATORS
#
# These are the functions necessary to process the data and compute the values
# of different financial indicators. Furthermore, some of these functions are
# used to apply the financial indicator to the data and verify is a signal has
# been detected.
################################################################################

class Signal():

    def __init__(self, is_bullish: bool, is_bearish: bool):

        self.is_bullish = is_bullish
        self.is_bearish = is_bearish


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

    return Signal(uptrend, downtrend)

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

    return Signal(oversold, overbought)

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

        return Signal(bullish_cross, bearish_cross)

    else:

        return Signal(macd_bullish, macd_bearish)

