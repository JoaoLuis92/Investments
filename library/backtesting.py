#####################
# NECESSARY IMPORTS #
#####################

import numpy as np
import yfinance as yf
import pandas as pd
from .entry_rules import Entry_rules, Position_sizing, entry_rules_long, entry_rules_short, position_sizing_long, position_sizing_short
from .financial_indicators import check_MACD

################################################################################
# BACKTESTING
#
# These are the functions necessary to backtest the strategies implemented and
# calculate their success rate
################################################################################

class Backtest_results():

    def __init__(self, number_trades: int, wins: int, losses: int, total_capital: float, win_streak: int, loss_streak: int, longest_win_streak: int, longest_loss_streak: int):

        self.number_trades = number_trades
        self.wins = wins
        self.losses = losses
        self.total_capital = total_capital
        self.win_streak = win_streak
        self.loss_streak = loss_streak
        self.longest_win_streak = longest_win_streak
        self.longest_loss_streak = longest_loss_streak

################################################################################
# Description:
#
# Inputs:
# ticker:
# initia:capital:
# risk_ratio:
# profit_ratio:
# current_win_streak:
# current_loss_streak:
# current_longest_win_streak:
# current_longest_loss_streak:
#
# Outputs: None
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

                if df_ticker.shift(200-i).iloc[400-1].Close > entry_rules.sell_limit:

                    longing = False
                    number_trades += 1
                    wins += 1
                    total_capital += position_sizing.number_shares * (entry_rules.sell_limit - entry_rules.buy_stop)

                    if loss_streak > longest_loss_streak:
                        longest_loss_streak = loss_streak

                    win_streak += 1
                    loss_streak = 0


                elif df_ticker.shift(200-i).iloc[400-1].Close < entry_rules.sell_stop:

                    longing = False
                    number_trades += 1
                    losses += 1
                    total_capital -= position_sizing.number_shares * (entry_rules.buy_stop - entry_rules.sell_stop)

                    if win_streak > longest_win_streak:
                        longest_win_streak = win_streak

                    win_streak = 0
                    loss_streak += 1

            if buy_order and i > order_instant + 2:
                buy_order = False

            if buy_order:

                if df_ticker.shift(200-i).iloc[400-1].Close > entry_rules.buy_stop:

                    buy_order = False
                    longing = True

            if not longing and not shorting and not buy_order and not sell_order:

                signal = check_MACD(df_ticker.shift(200-i), 12, 26, 9, True)

                if signal.is_bullish:

                    buy_order = True
                    entry_rules = entry_rules_long(df_ticker.shift(200-i), profit_ratio)
                    position_sizing = position_sizing_long(df_ticker.shift(200-i), total_capital, profit_ratio, risk_ratio)
                    order_instant = i

        except IndexError:
            continue

    return Backtest_results(number_trades, wins, losses, total_capital, win_streak, loss_streak, longest_win_streak, longest_loss_streak)

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

        results = backtest_MACD_crossover_ticker(ticker, total_capital, risk_ratio, profit_ratio, win_streak, loss_streak, longest_win_streak, longest_loss_streak)
        number_trades += results.number_trades
        wins += results.wins
        losses += results.losses
        total_capital = results.total_capital
        win_streak = results.win_streak
        loss_streak = results.loss_streak
        longest_win_streak = results.longest_win_streak
        longest_loss_streak = results.longest_loss_streak

    percentage_profit = (total_capital - initial_capital) / initial_capital * 100
    success_rate = wins / number_trades * 100
    average_profit = (profit_ratio * wins - risk_ratio * losses) / number_trades

    print(f"""
    Backtesting results:

    Number of trades: {number_trades}
    Wins: {wins}
    Losses: {losses}
    Longest winning streak: {longest_win_streak}
    Longest losing streak: {longest_loss_streak}
    Success_rate: {round(success_rate,1)}%
    Average profit per trade: {round(average_profit,1)}%

    Initial capital: {initial_capital}
    Final capital: {round(total_capital,2)}
    Profit percentage: {round(percentage_profit,1)}%
    """)
