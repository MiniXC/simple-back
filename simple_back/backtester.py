import pandas as pd
import pandas_market_calendars as mcal
from dateutil.relativedelta import relativedelta
from datetime import date
import numpy as np
import copy
import math
import json
from typing import Union, List, Tuple, Callable, overload
from warnings import warn
import os
from IPython.display import clear_output
from dataclasses import dataclass
from collections.abc import MutableSequence

from .price_providers import DailyPriceProvider, YahooFinanceProvider, DailyDataProvider
from .fees import NoFee
from .metrics import (
    MaxDrawdown,
    AnnualReturn,
    PortfolioValue,
    ProfitLoss,
    TotalValue,
    Metric,
)

try:
    from IPython import display
    import pylab as pl
    import matplotlib.pyplot as plt

    plt_exists = True
except ImportError:
    plt_exists = False


def _cls():
    clear_output(wait=True)
    os.system("cls" if os.name == "nt" else "clear")


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
        self.df_cols = [
            "symbol",
            "date",
            "event",
            "order_type",
            "profit_loss_abs",
            "profit_loss_pct",
            "price",
        ]

    def __repr__(self):
        t = self.order_type
        result = {
            "symbol": self.symbol,
            "date & event": str(self.date) + " " + self.event,
            "type": t,
            "shares": self.num_shares,
            "profit/loss (abs)": f"{self.profit_loss_abs:.2f}",
            "profit/loss (%)": f"{self.profit_loss_pct:.2f}",
            "price": f"{self.price:.2f}",
        }
        if self.frozen:
            result["end date & event"] = str(self.end_date) + " " + self.end_event
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
            return self.bt.prices[self.symbol, self.end_date, self.end_event]
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

    @property
    def order_type(self):
        t = None
        if self.short:
            t = "short"
        if self.long:
            t = "long"
        return t

    def _remove_shares(self, n):
        if self.short:
            self.num_shares_int += n
        if self.long:
            self.num_shares_int -= n

    def _freeze(self):
        self.frozen = True
        self.end_date = self.bt.current_date
        self.end_event = self.bt.event


class Portfolio(MutableSequence):
    def __init__(self, bt, positions: List[Position] = []):
        self.positions = positions
        self.bt = bt

    @property
    def value(self):
        val = 0
        for pos in self.positions:
            val += pos.value
        return val

    @property
    def df(self):
        pos_dict = {}
        for pos in self.positions:
            for col in pos.df_cols:
                if col not in pos_dict:
                    pos_dict[col] = []
                pos_dict[col].append(getattr(pos, col))
        return pd.DataFrame(pos_dict)

    def __repr__(self):
        return self.df.__repr__()

    def liquidate(self, num_shares=-1):
        is_long = False
        is_short = False
        for pos in self.positions:
            if pos.long:
                is_long = True
            if pos.short:
                is_short = True
        if is_long and is_short:
            raise LongShortLiquidationError(
                "liquidating a mix of long and short positions is not possible"
            )
        for pos in self.positions:
            if num_shares == -1 or num_shares >= pos.num_shares:
                self.bt._available_capital += pos.value
                self.bt.portfolio._remove(pos)

                pos._freeze()
                self.bt.trades._add(pos)

                num_shares -= pos.num_shares
            elif num_shares > 0 and num_shares < pos.num_shares:
                self.bt._available_capital += pos.value_pershare * num_shares
                pos._remove_shares(num_shares)

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

    @overload
    def __getitem__(self, index: Union[int, slice]):
        ...

    @overload
    def __getitem__(self, index: str):
        ...

    @overload
    def __getitem__(self, index: Union[np.ndarray, pd.Series, List[bool]]):
        ...

    def __getitem__(self, index):
        if isinstance(index, (int, slice)):
            return Portfolio(self.bt, copy.copy(self.positions[index]))
        if isinstance(index, str):
            new_pos = []
            for pos in self.positions:
                if pos.symbol == index:
                    new_pos.append(pos)
            return Portfolio(self.bt, new_pos)
        if isinstance(index, (np.ndarray, pd.Series, List[bool])):
            if len(index) > 0:
                new_pos = list(np.array(self.bt.portfolio.positions)[index])
            else:
                new_pos = []
            return Portfolio(self.bt, new_pos)

    def __setitem__(self, index, value):
        self.positions[index] = value

    def __delitem__(self, index: Union[int, slice]) -> None:
        del self.positions[index]

    def __len__(self):
        return len(self.positions)

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

    def insert(self, index: int, value: Position) -> None:
        self.positions.insert(index, value)


class BacktesterBuilder:
    def __init__(self):
        self.bt = copy.deepcopy(Backtester())

    def balance(self, amount: int) -> "BacktesterBuilder":
        self.bt._capital = amount
        self.bt._available_capital = amount
        self.bt._start_capital = amount
        return self

    def prices(self, prices: DailyPriceProvider) -> "BacktesterBuilder":
        self.bt.prices = prices
        self.bt.prices.bt = self.bt
        return self

    def data(self, data: DailyDataProvider) -> "BacktesterBuilder":
        self.bt.data.append(data)
        data.bt = self.bt
        return self

    def trade_cost(
        self, trade_cost: Callable[[float, float], Tuple[float, int]]
    ) -> "BacktesterBuilder":
        self.bt._trade_cost = trade_cost
        return self

    def metrics(self, metrics: Union[Metric, List[Metric]]) -> "BacktesterBuilder":
        if type(metrics) == list:
            for m in metrics:
                for m in metrics:
                    m.bt = self.bt
                    self.bt.metrics[m.name] = m
        else:
            metrics.bt = self.bt
            self.bt.metrics[metrics.name] = metrics
        return self

    def clear_metrics(self) -> "BacktesterBuilder":
        metrics = [PortfolioValue()]
        self.bt.metrics = {}
        self.bt.metrics(metrics)
        return self

    def calendar(self, calendar: str) -> "BacktesterBuilder":
        self.bt._calendar = calendar
        return self

    def live_metrics(self, every: int = 10) -> "BacktesterBuilder":
        if self.bt._live_plot:
            warn(
                """
                live plotting and metrics cannot be used together,
                 setting live plotting to false
                """
            )
            self.bt._live_plot = False
        self.bt._live_metrics = True
        self.bt._live_metrics_every = every
        return self

    def live_plot(
        self,
        every: int = 10,
        metric: str = "Total Value",
        compare: List[str] = [],
        event: str = "open",
    ) -> "BacktesterBuilder":
        if self.bt._live_metrics:
            warn(
                """
                live metrics and plotting cannot be used together,
                 setting live metrics to false
                """
            )
            self.bt._live_metrics = False
        self.bt._live_plot = True
        self.bt._live_plot_every = every
        self.bt._live_plot_metric = metric
        self.bt._live_plot_compare = compare
        self.bt._live_plot_event = event
        return self

    def build(self) -> "Backtester":
        return self.bt


class Backtester:
    def __getitem__(self, date_range: slice) -> "Backtester":
        if self.assume_nyse:
            self._calendar = "NYSE"
        if date_range.start is not None:
            start_date = date_range.start
        else:
            raise ValueError("a date range without a start value is not allowed")
        if date_range.stop is not None:
            end_date = date_range.stop
        else:
            end_date = date.today() - relativedelta(days=1)
        cal = mcal.get_calendar(self._calendar)
        if type(start_date) == relativedelta:
            start_date = date.today() + start_date
        if type(end_date) == relativedelta:
            end_date = date.today() + end_date
        sched = cal.schedule(start_date=start_date, end_date=end_date)
        self.dates = mcal.date_range(sched, frequency="1D")
        self.dates = [d.date() for d in self.dates]
        return self

    def __init__(self):
        self.dates = None
        self.assume_nyse = False

        self.prices = YahooFinanceProvider()
        self.prices.bt = self

        self.portfolio = Portfolio(self)
        self.trades = copy.deepcopy(Portfolio(self))

        self._trade_cost = NoFee

        metrics = [
            MaxDrawdown(),
            AnnualReturn(),
            PortfolioValue(),
            TotalValue(),
            ProfitLoss(),
        ]
        self.metrics = {}
        for m in metrics:
            m.bt = self
            self.metrics[m.name] = m

        self.data = []

        self._start_capital = None
        self._available_capital = None
        self._capital = None

        self._live_plot = False
        self._live_metrics = False

        self._siblings = []

    def _set_self(self):
        self.portfolio.bt = self
        self.trades.bt = self
        self.prices.bt = self

        for m in self.metrics.values():
            m.bt = self

    def _init_iter(self, bt=None):
        if bt is None:
            bt = self
        if bt.assume_nyse:
            warn("no market calendar specified, assuming NYSE calendar")
        if bt._available_capital is None or bt._capital is None:
            raise ValueError(
                "initial balance not specified, you can do so using .balance"
            )
        if bt.dates is None or len(bt.dates) == 0:
            raise ValueError(
                "no dates selected, you can select dates using [start_date:end_date]"
            )
        bt.i = -1
        bt.event = "close"
        return self

    def _next_iter(self, bt=None):
        no_bt = False
        if bt is None:
            bt = self
            no_bt = True
        if bt.event == "open":
            bt.event = "close"
        elif bt.event == "close":
            try:
                bt.i += 1
                bt.current_date = bt.dates[bt.i]
                bt.event = "open"
            except IndexError:
                bt.i -= 1
                for metric in bt.metrics.values():
                    if metric._single:
                        metric(write=True)
                if no_bt:
                    if bt._live_metrics:
                        bt._show_live_metrics()
                    if bt._live_plot:
                        bt._show_live_plot()
                raise StopIteration
        bt._update(no_bt)
        return bt.current_date, bt.event, bt

    def __iter__(self):
        return self._init_iter()

    def __next__(self):
        return self._next_iter()

    def __len__(self):
        return len(self.dates) * 2

    def _show_live_metrics(self, bts=None):
        _cls()
        for mkey in self.metrics.keys():
            if bts is None:
                metric = self.metrics[mkey]
                if str(metric) == "None":
                    metric = f"{metric():.2f}"
                print(f"{mkey:20} {metric}")
            else:
                metrics = []
                for bt in bts:
                    metric = bt.metrics[mkey]
                    if str(metric) == "None":
                        metric = f"{metric():.2f}"
                    metric = f"{str(metric):15}"
                    metrics.append(metric)
                print(f"{mkey:20} {''.join(metrics)}")

    def _show_live_plot(self, bts=None):
        if not plt_exists:
            warn("matplotlib not installed, setting live plotting to false")
            self._live_plot = False
        plot_df = pd.DataFrame()
        if bts is None:
            metric = self.metrics[self._live_plot_metric].df[self._live_plot_event]
            plot_df["Backtest"] = metric
        else:
            for i, bt in enumerate(bts):
                metric = bt.metrics[self._live_plot_metric].df[self._live_plot_event]
                plot_df[f"Backtest {i}"] = metric
        for ticker in self._live_plot_compare:
            comp = self.prices[ticker].loc[plot_df.index][self._live_plot_event]
            comp = comp * (self.balance.start / comp.iloc[0])
            plot_df[ticker] = comp
        plot_df.plot()
        plt.xlim([self.dates[0], self.dates[-1]])
        display.clear_output(wait=True)
        display.display(pl.gcf())
        plt.close()

    def _update(self, no_bt):
        for metric in self.metrics.values():
            if metric._series:
                metric(write=True)
        self._capital = self._available_capital + self.metrics["Portfolio Value"][-1]
        if no_bt:
            if self._live_metrics and self.i % self._live_metrics_every == 0:
                self._show_live_metrics()
            if self._live_plot and self.i % self._live_plot_every == 0:
                self._show_live_plot()

    def _order(self, symbol, capital, as_percent=False):
        if capital < 0:
            short = True
            capital = (-1) * capital
        else:
            short = False
        if not as_percent:
            if capital > self._available_capital:
                raise Exception("not enough capital available")
        else:
            if capital * self._capital > self._available_capital:
                if not math.isclose(capital * self._capital, self._available_capital):
                    raise Exception(
                        f"""
                        not enough capital available:
                        ordered {capital} * {self._capital}
                        with only {self._available_capital} available
                        """
                    )
        current_price = self.price(symbol)
        if as_percent:
            capital = capital * self._capital
        total, num_shares = self._trade_cost(current_price, capital)
        if short:
            num_shares *= -1
        if num_shares != 0:
            self._available_capital -= total
            self.portfolio._add(
                Position(self, symbol, self.current_date, self.event, num_shares)
            )
        else:
            raise Exception(
                f"""
                not enough capital specified to order a single share of {symbol}:
                tried to order {capital} of {symbol}
                with {symbol} price at {current_price}
                """
            )

    def order_pct(self, symbol, capital):
        self._order(symbol, capital, as_percent=True)

    def order_abs(self, symbol, capital):
        self._order(symbol, capital, as_percent=False)

    def price(self, symbol):
        try:
            return self.prices[symbol, self.current_date][self.event]
        except KeyError:
            self.prices.rem_cache(symbol)
            return self.prices[symbol, self.current_date][self.event]

    @property
    def balance(self):
        @dataclass
        class Balance:
            start: float = self._start_capital
            current: float = self._available_capital

        return Balance()

    @property
    def pf(self):
        return self.portfolio.df

    def run(self, strategies: List[Callable[["Date", str, "Backtester"], None]]):
        for i in range(len(strategies)):
            new_bt = copy.deepcopy(self)
            new_bt._set_self()
            self._init_iter(new_bt)
            self._siblings.append(new_bt)

        for i in range(len(self)):
            for i, bt in enumerate(self._siblings):
                strategies[i](*self._next_iter(bt))
                sib_i = bt.i
            if self._live_plot and sib_i % self._live_plot_every == 0:
                self._show_live_plot(self._siblings)
            if self._live_metrics and sib_i % self._live_metrics_every == 0:
                self._show_live_metrics(self._siblings)

        return self._siblings
