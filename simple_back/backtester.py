from simple_back.price_providers import DailyPriceProvider, YahooFinanceProvider
import pandas as pd
import pandas_market_calendars as mcal
from dateutil.relativedelta import relativedelta
from datetime import date
from abc import ABC, abstractmethod
import multiprocessing
import numpy as np
import copy
import math
import json

from .fees import NoFee
from .metrics import MaxDrawdown, AnnualReturn, PortfolioValue, ProfitLoss, TotalValue

class LongShortLiquidationError(Exception):
    def __init__(self, message):
        self.message = message

class Position:
    def __init__(self, bt, symbol, date, event, num_shares):
        self.symbol = symbol
        self.date = date
        self.event = event
        self.num_shares_int = num_shares
        self.init_price = bt.price(symbol)
        self.bt = bt
        self.frozen = False

    def __repr__(self):
        if self.short:
            t = 'short'
        if self.long:
            t = 'long'
        result = {
            'symbol': self.symbol,
            'date & event': str(self.date) + " " + self.event,
            'type': t,
            'shares': self.num_shares,
            'profit/loss (abs)': f'{self.profit_loss_abs:.2f}',
            'profit/loss (%)': f'{self.profit_loss_pct:.2f}',
        }
        if self.frozen:
            result['end date & event'] = str(self.end_date) + " " + self.end_event
        return json.dumps(result, sort_keys=True, indent=2)

    @property
    def short(self):
        return self.num_shares_int < 0

    @property
    def long(self):
        return self.num_shares_int > 0

    @property
    def value(self):
        if self.short:
            old_val = self.initial_value
            cur_val = self.num_shares * self.bt.price(self.symbol)
            return old_val + (old_val - cur_val)
        if self.long:
            return self.num_shares * self.bt.price(self.symbol)

    @property
    def price(self):
        if self.frozen:
            return self.bt.prices[self.symbol,self.end_date,self.end_event]
        else:
            return self.bt.price(self.symbol)

    @property
    def value_pershare(self):
        if self.long:
            return self.bt.price(self.symbol)
        if self.short:
            return self.init_price + (self.init_price - self.bt.price(self.symbol))

    @property
    def initial_value(self):
        if self.short:
            return self.num_shares * self.init_price
        if self.long:
            return self.num_shares * self.init_price

    @property
    def profit_loss_pct(self):
            return self.value / self.initial_value - 1

    @property
    def profit_loss_abs(self):
        return self.value - self.initial_value

    @property
    def num_shares(self):
        return abs(self.num_shares_int)

    def _remove_shares(self, n):
        if self.short:
            self.num_shares_int += n
        if self.long:
            self.num_shares_int -= n

    def _freeze(self):
        self.frozen = True
        self.end_date = self.bt.current_date
        self.end_event = self.bt.event

class Portfolio:
    def __init__(self, bt, positions=[]):
        self.positions = positions
        self.bt = bt
    
    @property
    def value(self):
        val = 0
        for pos in self.positions:
            val += pos.value
        return val

    def __repr__(self):
        return self.positions.__repr__()

    def liquidate(self, num_shares=-1):
        is_long = False
        is_short = False
        for pos in self.positions:
            if pos.long:
                is_long = True
            if pos.short:
                is_short = True
        if is_long and is_short:
            raise LongShortLiquidationError("liquidating a mix of long and short positions is not possible")
        for pos in self.positions:
            if num_shares == -1 or num_shares > pos.num_shares:
                self.bt.available_capital += pos.value
                self.bt.portfolio._remove(pos)
                
                pos._freeze()
                self.bt.trades._add(pos)
                
                num_shares -= pos.num_shares
            elif num_shares > 0 and num_shares < pos.num_shares:
                self.bt.available_capital += pos.value_pershare * num_shares
                pos._remove_shares(num_shares)
                
                # TODO: test how much this slows everything down, maybe make conditional
                hist = copy.copy(pos)
                hist._freeze()
                if hist.short:
                    hist.num_shares_int = (-1) * num_shares
                if hist.long:
                    hist.num_shares_int = num_shares
                self.bt.trades._add(hist)
                
                break

    def _add(self, position):
        self.positions.append(position)

    def _remove(self, position):
        self.positions.remove(position)

    def __getitem__(self, symbol):
        new_pos = []
        # TODO: add possibility for gt, lt...
        if type(symbol) == str:
            for pos in self.positions:
                if pos.symbol == symbol:
                    new_pos.append(pos)
        return Portfolio(self.bt, new_pos)

    @property
    def short(self):
        new_pos = []
        for pos in self.positions:
            if pos.short:
                new_pos.append(pos)
        return Portfolio(self.bt, new_pos)

    @property
    def long(self):
        new_pos = []
        for pos in self.positions:
            if pos.long:
                new_pos.append(pos)
        return Portfolio(self.bt, new_pos)

    def __len__(self):
        return len(self.positions)

class Backtester:
    def __init__(
        self,
        start_capital,
        prices=None,
        strategy=None,
        # list of dates to be used
        dates=None,
        # if list of dates is not supplied, use pandas_market_calendars
        market_calendar=None,
        cal=None,
        start=None,
        start_date=None,
        end_date=None,
        end=None,
        trade_cost_function=NoFee,
        metrics=[MaxDrawdown(), AnnualReturn(), PortfolioValue(), TotalValue(), ProfitLoss()],
        # this slows down testing, but leads to more accurate metrics
        # and can be nice for the plotly candlestick chart
        use_high_low=False,
        highlow=False,
        live_chart=True,
    ):
        self.start_capital = start_capital
        self.capital = start_capital
        self.available_capital = start_capital

        if prices is None:
            prices = YahooFinanceProvider()
        prices.bt = self
        if dates is not None:
            self.dates = dates
        else:
            if start is not None:
                start_date = start
                end_date = end
            if cal is not None:
                market_calendar = cal
            cal = mcal.get_calendar(market_calendar)
            if start_date is not None:
                if end_date is None:
                    end_date = date.today() - relativedelta(days=1)
                if type(start_date) == relativedelta:
                    start_date = date.today() - start_date
                if type(end_date) == relativedelta:
                    end_date = date.today() - end_date
                sched = cal.schedule(start_date=start_date, end_date=end_date)
            self.dates = mcal.date_range(sched, frequency="1D")
            self.dates = [d.date() for d in self.dates]
        if strategy is not None:
            self.strategy = strategy
        self.prices = prices
        self.portfolio = Portfolio(self)
        self.trades = copy.deepcopy(Portfolio(self))
        self.value_ot = pd.DataFrame(columns=["date", "event", "value"])
        self.value_ot["value"] = self.value_ot["value"].astype(float)
        self.trade_cost = trade_cost_function

        self.metrics = {}
        for m in metrics:
            m.bt = self
            self.metrics[m.name] = m

        self.use_high_low = use_high_low or highlow

        self.live_chart = live_chart

    def run(self):
        for i, date in enumerate(self.dates):
            self.i = i
            self.current_date = date

            self.event = "open"
            self.update()
            self.strategy.run(self.current_date, self.event, self)

            self.event = "close"
            self.update()
            self.strategy.run(self.current_date, self.event, self)

            if self.live_chart:
                self.live_plot()
        return self

    def __iter__(self):
        self.i = -1
        self.event = "close"
        return self

    def __next__(self):
        if self.event == "open":
            self.event = "close"
        elif self.event == "close":
            try:
                self.i += 1
                self.current_date = self.dates[self.i]
                self.event = "open"
            except IndexError:
                self.i -= 1
                for metric in self.metrics.values():
                    if metric._single:
                        metric(write=True)
                if self.live_chart:
                    self.live_plot()
                raise StopIteration
        self.update()
        return self.current_date, self.event, self

    def __len__(self):
        return len(self.dates) * 2

    def price(self, symbol):
        return self.prices[symbol, self.current_date, self.event]

    def update(self):
        for metric in self.metrics.values():
            if metric._series:
                metric(write=True)
        self.capital = self.available_capital + self.metrics["Portfolio Value"][-1]
        if self.live_chart and self.i % 10 == 0:
            self.live_plot()

    def _order(self, symbol, capital, as_percent=False):
        if capital < 0:
            short = True
            capital = (-1) * capital
        else:
            short = False
        if not as_percent:
            if capital > self.available_capital:
                raise Exception("not enough capital available")
        else:
            if capital * self.capital > self.available_capital:
                if not math.isclose(capital * self.capital, self.available_capital):
                    raise Exception(
                        f"""
                        not enough capital available:
                        ordered {capital} * {self.capital} with only {self.available_capital} available
                        """
                    )
        current_price = self.price(symbol)
        if as_percent:
            capital = capital * self.capital
        total, num_shares = self.trade_cost(current_price, capital)
        if short:
            num_shares *= -1
        self.available_capital -= total
        self.portfolio._add(Position(
            self,
            symbol,
            self.current_date,
            self.event,
            num_shares)
        )
    
    def order_pct(self, symbol, capital):
        self._order(symbol, capital, as_percent=True)

    def order_abs(self, symbol, capital):
        self._order(symbol, capital, as_percent=False)

    def live_plot(self):
        try:
            try:
                _ = self.live_plot_first
                self.live_plot_first = False
            except:
                self.live_plot_first = True
            if self.live_plot_first:
                from IPython import display
                import pylab as pl
                import matplotlib.pyplot as plt

                self.ext = {}
                self.ext["plt"] = plt
                self.ext["display"] = display
                self.ext["pl"] = pl
            self.ext["plt"].plot(self.metrics['Total Value'].value)
            #self.ext["plt"].xlim([self.dates[0], self.dates[-1]])
            self.ext["display"].clear_output(wait=True)
            self.ext["display"].display(self.ext["pl"].gcf())
            self.ext["plt"].close()
        except ImportError:
            raise ("Live plots only work in Jupyter Lab or Notebook")

    @property
    def plotly(self):
        try:
            import plotly.graph_objs as go

            date_vals = self.value_ot["date"].unique()
            open_vals = self.value_ot[self.value_ot["event"] == "open"].reset_index()[
                "value"
            ]
            close_vals = self.value_ot[self.value_ot["event"] == "close"].reset_index()[
                "value"
            ]
            if self.prices.hasHighLow:
                high_vals = self.value_ot[
                    self.value_ot["event"] == "high"
                ].reset_index()["value"]
                low_vals = self.value_ot[self.value_ot["event"] == "low"].reset_index()[
                    "value"
                ]
            else:
                high_vals = [max(v) for v in zip(open_vals, close_vals)]
                low_vals = [min(v) for v in zip(open_vals, close_vals)]
            all_dates = pd.date_range(self.dates[0], self.dates[-1])
            bt_dates = pd.Series(self.dates)
            fig = go.Figure(
                data=[
                    go.Candlestick(
                        x=date_vals,
                        open=open_vals,
                        close=close_vals,
                        high=high_vals,
                        low=low_vals,
                    )
                ]
            )
            fig.update_xaxes(
                rangebreaks=[
                    dict(values=all_dates[~all_dates.isin(bt_dates)]),  # hide weekends
                ]
            )
            return fig
        except ImportError:
            raise ("Please install plotly for charting to work.")
