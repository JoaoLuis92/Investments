#####################
# NECESSARY IMPORTS #
#####################

import numpy as np
import yfinance as yf
import pandas as pd
from .financial_indicators import Signal

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

def detect_candlestick_pattern(df: pd.core.frame.DataFrame, confirm: bool = True) -> (bool, bool):

    bullish = detect_bullish_pattern(df,confirm)
    bearish = detect_bearish_pattern(df,confirm)

    return Signal(bullish, bearish)

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
