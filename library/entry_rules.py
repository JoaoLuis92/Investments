#####################
# NECESSARY IMPORTS #
#####################

import numpy as np
import yfinance as yf
import pandas as pd
from .financial_indicators import average_true_range

################################################################################
# ENTRY RULES
#
# These are the functions necessary to calculate the prices at which to place
# the buy and sell (both stop and limit) orders for a given stock after a signal
# has been detected
################################################################################

class Entry_rules():

    def __init__(self, buy_stop: float, buy_limit: float, sell_stop: float, sell_limit: float):

        self.buy_stop = buy_stop
        self.buy_limit = buy_limit
        self.sell_stop = sell_stop
        self.sell_limit = sell_limit

class Position_sizing():

    def __init__(self, number_shares: int, position_size: float, total_risk: float):

        self.number_shares = number_shares
        self.position_size = position_size
        self.total_risk = total_risk

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

def print_order_details(entry_rules: Entry_rules, position_sizing: Position_sizing):

    print(f"""
    Buy stop: {round(entry_rules.buy_stop,2)}
    Buy limit: {round(entry_rules.buy_limit,2)}
    Sell stop: {round(entry_rules.sell_stop,2)}
    Sell limit: {round(entry_rules.sell_limit,2)}

    Number of shares: {position_sizing.number_shares}
    Position size: {round(position_sizing.position_size,2)}
    Total risk: {round(position_sizing.total_risk,2)}
    """)

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

    return Entry_rules(buy_stop, buy_limit, sell_stop, sell_limit)

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

    return Entry_rules(buy_stop, buy_limit, sell_stop, sell_limit)

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

    entry_rules = entry_rules_long(df_work, profit_ratio)

    number_shares = int(total_capital * 0.01 * risk_ratio / (entry_rules.buy_stop - entry_rules.sell_stop))
    position_size = number_shares * entry_rules.buy_stop
    total_risk = 0.01 * risk_ratio * total_capital

    return Position_sizing(number_shares, position_size, total_risk)

def position_sizing_short(df: pd.core.frame.DataFrame, total_capital: float, profit_ratio: float, risk_ratio: float) -> (float, float, float):

    df_work = df.copy()

    entry_rules = entry_rules_short(df_work, profit_ratio)

    number_shares = int(total_capital * 0.01 * risk_ratio / (entry_rules.buy_stop - entry_rules.sell_stop))
    position_size = number_shares * entry_rules.sell_stop
    total_risk = 0.01 * risk_ratio * total_capital

    return Position_sizing(number_shares, position_size, total_risk)

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

    entry_rules = entry_rules_long(df_work, profit_ratio)
    position_sizing = position_sizing_long(df_work, total_capital, profit_ratio, risk_ratio)

    print_order_details(entry_rules, position_sizing)

def order_details_short(ticker: str, total_capital: float, profit_ratio: float, risk_ratio: float):

    df_work = yf.Ticker(ticker).history(period="400d").drop(["Dividends","Stock Splits"], axis = 1)

    entry_rules = entry_rules_short(df_work, profit_ratio)
    position_sizing = position_sizing_short(df_work, total_capital, profit_ratio, risk_ratio)

    print_order_details(entry_rules, position_sizing)

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

        print("Stock: ", ticker)
        order_details_long(ticker, total_capital, profit_ratio, risk_ratio)

def order_details_short_list(tickers: list, total_capital: float, profit_ratio: float, risk_ratio: float):

    for ticker in tickers:

        print("Stock: ", ticker)
        order_details_short(ticker, total_capital, profit_ratio, risk_ratio)
