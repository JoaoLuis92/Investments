#####################
# NECESSARY IMPORTS #
#####################

import numpy as np
import yfinance as yf
import pandas as pd
import scipy.stats as scs
import statsmodels.api as sm
from pylab import mpl, plt
from .financial_indicators import log_returns
from .diversification import closing_data
import scipy.optimize as sco
import scipy.interpolate as sci

################################################################################
# STATISTICS
#
# These are the functions necessary to analyze a wide variety of statistical
# properties, e.g. normality tests, quantile-quantile plots, among others
################################################################################

def normality_tests(df: pd.core.frame.DataFrame, plots = True):

    df_work = df.copy()
    df_work = log_returns(df_work).dropna()

    if plots:
        df_work["LR"].hist(bins=10, figsize=(5, 5))
        sm.qqplot(df_work["LR"], line="s")

    print(f"""
    Skew:         {round(scs.skew(df_work["LR"]),2)}
    Kurt:         {round(scs.kurtosis(df_work["LR"]),2)}
    Skew p-value: {round(scs.skewtest(df_work["LR"])[1],2)}
    Kurt p-value: {round(scs.kurtosistest(df_work["LR"])[1],2)}
    Norm p-value: {round(scs.normaltest(df_work["LR"])[1],2)}
    """)

class Portfolio():

    def __init__(self, tickers: list, rf: float = 0):

        self.tickers = tickers
        self.rf = rf
        self.noa = len(tickers)
        self.data = closing_data(tickers)
        self.rets = np.log(self.data / self.data.shift(1))
        self.prets = []
        self.pvols = []
        self.trets = []
        self.tvols = []
        self.i_min = 0
        self.weights = []
        self.eweights = np.array(self.noa * [1 / self.noa,])
        self.bounds = tuple((0,1) for x in range(self.noa))
        self.consts = ()
        self.opts = 0
        self.optv = 0
        self.optf = 0
        self.optm = 0
        self.interpol = 0

    def portfolio_returns(self, weights):
        return np.sum(self.rets.mean() * weights) * 252

    def portfolio_volatility(self, weights):
        return np.sqrt(np.dot(weights.T, np.dot(self.rets.cov() * 252, weights)))

    def negative_sharpe(self, weights):
        return - self.portfolio_returns(weights) / self.portfolio_volatility(weights)

    def monte_carlo_simulation(self):

        for p in range(5000):

            self.weights = np.random.random(self.noa)
            self.weights /= self.weights.sum()

            self.prets.append(self.portfolio_returns(self.weights))
            self.pvols.append(self.portfolio_volatility(self.weights))

        self.prets = np.array(self.prets)
        self.pvols = np.array(self.pvols)

    def optimize_sharpe(self):

        self.consts =({"type": "eq", "fun": lambda x: np.sum(x) - 1})
        self.opts = sco.minimize(self.negative_sharpe, self.eweights, method="SLSQP", bounds = self.bounds, constraints = self.consts)

        print(f"""
        Sharpe optimization:
        Optimized weights:    {self.opts["x"].round(3)}
        Optimized returns:    {self.portfolio_returns(self.opts["x"]).round(3)}
        Optimized volatility: {self.portfolio_volatility(self.opts["x"]).round(3)}
        Optimized Sharpe:     {round(self.portfolio_returns(self.opts["x"]) / self.portfolio_volatility(self.opts["x"]),3)}
        """)

    def optimize_risk(self):

        self.consts =({"type": "eq", "fun": lambda x: np.sum(x) - 1})
        self.optv = sco.minimize(self.portfolio_volatility, self.eweights, method="SLSQP", bounds = self.bounds, constraints = self.consts)

        print(f"""
        Risk optimization:
        Optimized weights:    {self.optv["x"].round(3)}
        Optimized returns:    {self.portfolio_returns(self.optv["x"]).round(3)}
        Optimized volatility: {self.portfolio_volatility(self.optv["x"]).round(3)}
        Optimized Sharpe:     {round(self.portfolio_returns(self.optv["x"]) / self.portfolio_volatility(self.optv["x"]),3)}
        """)

    def efficient_frontier(self):

        self.consts = ({"type": "eq", "fun": lambda x: np.sum(x) - 1},
                       {"type": "eq", "fun": lambda x: self.portfolio_returns(x) - tret})

        self.trets = np.linspace(self.prets.min(), self.prets.max(), 50)

        for tret in self.trets:

            self.optf = sco.minimize(self.portfolio_volatility, self.eweights, method="SLSQP", bounds = self.bounds, constraints = self.consts)
            self.tvols.append(self.optf["fun"])

        self.tvols = np.array(self.tvols)

    def f(self, x):
        return sci.splev(x, self.interpol, der=0)

    def df(self, x):
        return sci.splev(x, self.interpol, der=1)

    def equations(self, p: list):

        eq1 = self.rf - p[0]
        eq2 = self.rf + p[1] * p[2] - self.f(p[2])
        eq3 = p[1] - self.df(p[2])

        return eq1, eq2, eq3

    def capital_market_line(self):

        self.i_min = np.argmin(self.tvols)
        self.interpol = sci.splrep(self.tvols[self.i_min:], self.trets[self.i_min:])

        self.optm = sco.fsolve(self.equations, [self.rf, 0.5, self.portfolio_volatility(self.opts["x"]).round(3)])

    def plot_optimization(self):

        plt.figure(figsize=(10,5))
        plt.scatter(self.pvols, self.prets, c = (self.prets - self.rf) / self.pvols, marker=".", cmap = "coolwarm")
        plt.plot(self.tvols,  self.trets, "b", lw=3)
        cx = np.linspace(self.optm[2]-0.01, self.optm[2]+0.01)
        #cx = np.linspace(self.tvols[self.i_min:].min()-0.005, self.tvols[self.i_min:].max()+0.005)
        plt.plot(cx, self.optm[0] + self.optm[1] * cx, "r", lw=1.5)
        plt.plot(self.portfolio_volatility(self.opts["x"]), self.portfolio_returns(self.opts["x"]), "y*", markersize=15)
        plt.plot(self.portfolio_volatility(self.optv["x"]), self.portfolio_returns(self.optv["x"]), "y*", markersize=15)
        plt.plot(self.optm[2], self.f(self.optm[2]), "y*", markersize=15)
        plt.xlabel("expected volatility")
        plt.ylabel("expected returns")
        plt.colorbar(label="sharpe ratio")
        plt.grid()

    def optimize_portfolio(self, plot = True):

        self.monte_carlo_simulation()
        self.optimize_sharpe()
        self.optimize_risk()
        self.efficient_frontier()
        self.capital_market_line()

        if plot:

            self.plot_optimization()
