import pandas as pd
import pandas_market_calendars as mcal
from dateutil.relativedelta import relativedelta
import datetime
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
import multiprocessing
import threading
from collections.abc import MutableSequence

from .data_providers import (
    DailyPriceProvider,
    YahooFinanceProvider,
    DailyDataProvider,
    PriceUnavailableError,
    DataProvider,
)
from .fees import NoFee, Fee, InsufficientCapitalError
from .metrics import (
    MaxDrawdown,
    AnnualReturn,
    PortfolioValue,
    DailyProfitLoss,
    TotalValue,
    Metric,
)
from .strategy import Strategy, BuyAndHold

try:
    from IPython import display
    import pylab as pl
    import matplotlib.pyplot as plt

    plt_exists = True
except ImportError:
    plt_exists = False

try:
    from tqdm import tqdm

    tqdm_exists = True
except ImportError:
    tqdm_exists = False


class BacktestRunException(Exception):
    def __init__(self, message):
        self.message = message

class LongShortLiquidationError(Exception):
    def __init__(self, message):
        self.message = message

class NegativeValueError(Exception):
    def __init__(self, message):
        self.message = message

class StrategySequence:
    """A sequence of strategies than can be accessed by name or :class:`int` index.\
    Returned by :py:obj:`.Backtester.strategies` and should not be used elsewhere.

    Examples:

        Access by :class:`str`::

            bt.strategies['Some Strategy Name']

        Access by :class:`int`::

            bt.strategies[0]

        Use as iterator::

            for strategy in bt.strategies:
                # do something
    """

    def __init__(self, bt):
        self.bt = bt
        self.i = 0

    def __getitem__(self, index: Union[str, int]):
        if isinstance(index, int):
            bt = self.bt._get_bts()[index]
            bt._from_sequence = True
            return bt
        elif isinstance(index, str):
            for i, bt in enumerate(self.bt._get_bts()):
                if bt.name is not None:
                    if bt.name == index:
                        bt._from_sequence = True
                        return bt
                else:
                    if f"Backtest {i}" == index:
                        bt._from_sequence = True
                        return bt
            raise IndexError

    def __iter__(self):
        return self

    def __next__(self):
        i = self.i
        self.i += 1
        return self[i]

    def __len__(self):
        return len(self.bt._get_bts())


def _cls():
    clear_output(wait=True)
    os.system("cls" if os.name == "nt" else "clear")


class Position:
    def __init__(self, bt, symbol, date, event, num_shares):
        self.symbol = symbol
        self.date = date
        self.event = event
        self.num_shares_int = num_shares
        self.start_price = bt.price(symbol)
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
            cur_val = self.num_shares * self.price
            return old_val + (old_val - cur_val)
        if self.long:
            return self.num_shares * self.price

    @property
    def price(self):
        if self.frozen:
            return self.bt.prices[self.symbol, self.end_date][self.end_event]
        else:
            return self.bt.price(self.symbol)

    @property
    def value_pershare(self):
        if self.long:
            return self.price
        if self.short:
            return self.start_price + (self.start_price - self.price)

    @property
    def initial_value(self):
        if self.short:
            return self.num_shares * self.start_price
        if self.long:
            return self.num_shares * self.start_price

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
        self.df_cols += ['start_price', 'end_date', 'end_event']


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
            self.bt._graceful_stop()
            raise LongShortLiquidationError(
                "liquidating a mix of long and short positions is not possible"
            )
        for pos in copy.copy(self.positions):
            if num_shares == -1 or num_shares >= pos.num_shares:
                self.bt._available_capital += pos.value
                if self.bt._available_capital < 0:
                    self.bt._graceful_stop()
                    raise NegativeValueError(
                        f'Tried to liquidate position resulting in negative capital {self.bt._available_capital}.'
                    )
                self.bt.portfolio._remove(pos)

                pos._freeze()
                self.bt.trades._add(copy.copy(pos))

                if num_shares != -1:
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

    def __bool__(self):
        return len(self) != 0

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

    def filter(self, func: Callable[[Position], bool]):
        new_pos = []
        for pos in self.positions:
            if func(pos):
                new_pos.append(pos)
        return Portfolio(self.bt, new_pos)

    def attr(self, attribute):
        result = [getattr(pos, attribute) for pos in self.positions]
        if len(result) == 0:
            return None
        elif len(result) == 1:
            return result[0]
        else:
            return result



class BacktesterBuilder:
    """
    Used to configure a :class:`.Backtester`
    and then creating it with :meth:`.build`

    Example:
        Create a new :class:`.Backtester` with 10,000 starting balance
        which runs on days the `NYSE`_ is open::

            bt = BacktesterBuilder().balance(10_000).calendar('NYSE').build()

    .. _NYSE:
       https://www.nyse.com/index
    """

    def __init__(self):
        self.bt = copy.deepcopy(Backtester())

    def name(self, name: str) -> "BacktesterBuilder":
        """**Optional**, name will be set to "Backtest 0" if not specified.

        Set the name of the strategy run using the :class:`.Backtester` iterator.

        Args:
            name: The strategy name.
        """
        self.bt.name = name
        return self

    def balance(self, amount: int) -> "BacktesterBuilder":
        """**Required**, set the starting balance for all :class:`.Strategy` objects
        run with the :class:`.Backtester`

        Args:
            amount: The starting balance.
        """
        self.bt._capital = amount
        self.bt._available_capital = amount
        self.bt._start_capital = amount
        return self

    def prices(self, prices: DailyPriceProvider) -> "BacktesterBuilder":
        """**Optional**, set the :class:`.DailyPriceProvider` used to get prices during
        a backtest. If this is not called, :class:`.YahooPriceProvider`
        is used.

        Args:
            prices: The price provider.
        """
        self.bt.prices = prices
        self.bt.prices.bt = self.bt
        return self

    def data(self, data: DataProvider) -> "BacktesterBuilder":
        """**Optional**, add a :class:`.DataProvider` to use external data without time leaks.

        Args:
            data: The data provider.
        """
        self.bt.data[data.name] = data
        data.bt = self.bt
        return self

    def trade_cost(
        self, trade_cost: Union[Fee, Callable[[float, float], Tuple[float, int]]]
    ) -> "BacktesterBuilder":
        """**Optional**, set a :class:`.Fee` to be applied when buying shares.
        When not set, :class:`.NoFee` is used.
        """
        self.bt._trade_cost = trade_cost
        return self

    def metrics(self, metrics: Union[Metric, List[Metric]]) -> "BacktesterBuilder":
        if isinstance(metrics, list):
            for m in metrics:
                for m in metrics:
                    m.bt = self.bt
                    self.bt.metric[m.name] = m
        else:
            metrics.bt = self.bt
            self.bt.metric[metrics.name] = metrics
        return self

    def clear_metrics(self) -> "BacktesterBuilder":
        metrics = [PortfolioValue()]
        self.bt.metric = {}
        self.bt.metric(metrics)
        return self

    def calendar(self, calendar: str) -> "BacktesterBuilder":
        self.bt._calendar = calendar
        return self

    def live_metrics(self, every: int = 10) -> "BacktesterBuilder":
        if self.bt._live_plot:
            self.bt._warn.append(
                """
                live plotting and metrics cannot be used together,
                 setting live plotting to false
                """
            )
            self.bt._live_plot = False
        self.bt._live_metrics = True
        self.bt._live_metrics_every = every
        return self

    def no_live_metrics(self) -> "BacktesterBuilder":
        scopy = copy.copy(self)
        scopy.bt._live_metrics = False
        return scopy

    def live_plot(
        self,
        every: int = 10,
        metric: str = "Total Value",
        figsize: Tuple[float, float] = None,
    ) -> "BacktesterBuilder":
        if self.bt._live_metrics:
            self.bt._warn.append(
                """
                live metrics and plotting cannot be used together,
                 setting live metrics to false
                """
            )
            self.bt._live_metrics = False
        self.bt._live_plot = True
        self.bt._live_plot_every = every
        self.bt._live_plot_metric = metric
        self.bt._live_plot_figsize = figsize
        return self

    def no_live_plot(self) -> "BacktesterBuilder":
        scopy = copy.copy(self)
        scopy.bt._live_plot = False
        return scopy

    def live_progress(self, every: int = 10) -> "BacktesterBuilder":
        self.bt._live_progress = True
        self.bt._live_progress_every = every
        return self

    def no_live_progress(self) -> "BacktesterBuilder":
        scopy = copy.copy(self)
        scopy.bt._live_progress = False
        return scopy

    def compare(
        self,
        strategies: List[
            Union[Callable[["datetime.date", str, "Backtester"], None], Strategy, str]
        ],
    ):
        return self.strategies(strategies)

    def strategies(
        self,
        strategies: List[
            Union[Callable[["datetime.date", str, "Backtester"], None], Strategy, str]
        ],
    ) -> "BacktesterBuilder":
        strats = []
        for strat in strategies:
            if isinstance(strat, str):
                strats.append(BuyAndHold(strat))
            else:
                strats.append(strat)
        self.bt._temp_strategies = strats
        self.bt._has_strategies = True
        return self

    def build(self) -> "Backtester":
        self.bt._builder = self
        return copy.deepcopy(self.bt)


class Backtester:
    def __getitem__(self, date_range: slice) -> "Backtester":
        if self._run_before:
            raise BacktestRunException(
                "Backtest has already run, build a new backtester to run again."
            )
        self._run_before = True
        if self.assume_nyse:
            self._calendar = "NYSE"
        if date_range.start is not None:
            start_date = date_range.start
        else:
            raise ValueError("a date range without a start value is not allowed")
        if date_range.stop is not None:
            end_date = date_range.stop
        else:
            self._warn.append(
                "backtests with no end date can lead to non-replicable results"
            )
            end_date = datetime.date.today() - relativedelta(days=1)
        cal = mcal.get_calendar(self._calendar)
        if isinstance(start_date, relativedelta):
            start_date = datetime.date.today() + start_date
        if isinstance(end_date, relativedelta):
            end_date = datetime.date.today() + end_date
        sched = cal.schedule(start_date=start_date, end_date=end_date)
        self._schedule = sched
        self.dates = mcal.date_range(sched, frequency="1D")
        self.datetimes = []
        self.dates = [d.date() for d in self.dates]
        for date in self.dates:
            self.datetimes += [
                sched.loc[date]["market_open"],
                sched.loc[date]["market_close"],
            ]

        if self._has_strategies:
            self._set_strategies(self._temp_strategies)

        return self

    def __init__(self):
        self.dates = []
        self.assume_nyse = False

        self.prices = YahooFinanceProvider()
        self.prices.bt = self

        self.portfolio = Portfolio(self)
        self.trades = copy.deepcopy(Portfolio(self))

        self._trade_cost = NoFee()

        metrics = [
            MaxDrawdown(),
            AnnualReturn(),
            PortfolioValue(),
            TotalValue(),
            DailyProfitLoss(),
        ]
        self.metric = {}
        for m in metrics:
            m.bt = self
            self.metric[m.name] = m

        self.data = {}

        self._start_capital = None
        self._available_capital = None
        self._capital = None

        self._live_plot = False
        self._live_plot_figsize = None
        self._live_metrics = False
        self._live_progress = False

        self._strategies = []
        self._temp_strategies = []
        self._has_strategies = False
        self.name = None

        self._no_iter = False

        self._schedule = None
        self._warn = []

        self._add_metrics = {}
        self._add_metrics_lines = []

        self.datetimes = None
        self.add_metric_exists = False

        self._run_before = False

        self._last_thread = None

        self._from_sequence = False

    def _set_self(self, new_self=None):
        if new_self is not None:
            self = new_self
        self.portfolio.bt = self
        self.trades.bt = self
        self.prices.bt = self

        for m in self.metric.values():
            m.__init__()
            m.bt = self

    def _init_iter(self, bt=None):
        global _live_progress_pbar
        if bt is None:
            bt = self
        if bt.assume_nyse:
            self._warn.append("no market calendar specified, assuming NYSE calendar")
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
        if self._live_progress:
            _live_progress_pbar = tqdm(total=len(self))
            _cls()
        return self

    def _next_iter(self, bt=None):
        if bt is None:
            bt = self
        if bt.i == len(self):
            bt._init_iter()
        if bt.event == "open":
            bt.event = "close"
            bt.i += 1
        elif bt.event == "close":
            try:
                bt.i += 1
                bt.current_date = bt.dates[bt.i // 2]
                bt.event = "open"
            except IndexError:
                bt.i -= 1
                for metric in bt.metric.values():
                    if metric._single:
                        metric(write=True)
                if self._has_strategies:
                    for strat in self._strategies:
                        for metric in strat.metric.values():
                            if metric._single:
                                metric(write=True)
                self.plot(self._get_bts(), last=True)
                for w in self._warn:
                    warn(w)
                raise StopIteration
        bt._update()
        return bt.current_date, bt.event, bt

    def add_metric(self, key, value):
        if key not in self._add_metrics:
            self._add_metrics[key] = (
                np.repeat(np.nan, len(self)),
                np.repeat(True, len(self)),
            )
        self._add_metrics[key][0][self.i] = value
        self._add_metrics[key][1][self.i] = False

    def add_line(self, **kwargs):
        self._add_metrics_lines.append((self.timestamp, kwargs))

    @property
    def timestamp(self):
        return self._schedule.loc[self.current_date][f"market_{self.event}"]

    def __iter__(self):
        return self._init_iter()

    def __next__(self):
        if len(self._strategies) > 0:
            result = self._next_iter()
            self._run_once()
            self.plot([self] + self._strategies)
        else:
            result = self._next_iter()
            self.plot([self])
        return result

    def __len__(self):
        return len(self.dates) * 2

    def _show_live_metrics(self, bts=None):
        _cls()
        lines = []
        bt_names = []
        if bts is not None:
            if not self._no_iter and self not in bts:
                bts = [self] + bts
            for i, bt in enumerate(bts):
                if bt.name is None:
                    name = f"Backtest {i}"
                else:
                    name = bt.name
                name = f"{name:20}"
                if len(name) > 20:
                    name = name[:18]
                    name += ".."
                bt_names.append(name)
            lines.append(f"{'':20} {''.join(bt_names)}")
        if bts is None:
            bts = [self]
        for mkey in self.metric.keys():
            metrics = []
            for bt in bts:
                metric = bt.metric[mkey]
                if str(metric) == "None":
                    metric = f"{metric():.2f}"
                metric = f"{str(metric):20}"
                metrics.append(metric)
            lines.append(f"{mkey:20} {''.join(metrics)}")
        for line in lines:
            print(line)
        if self._live_progress:
            print()
            print(self._show_live_progress())

    def _show_live_plot(self, bts=None, start_end=None):
        if not plt_exists:
            self._warn.append(
                "matplotlib not installed, setting live plotting to false"
            )
            self._live_plot = False
            return None
        plot_df = pd.DataFrame()
        plot_df["Date"] = self.datetimes
        plot_df = plot_df.set_index("Date")
        plot_add_df = plot_df.copy()
        add_metric_exists = False
        for i, bt in enumerate(bts):
            metric = bt.metric[self._live_plot_metric].values
            name = f"Backtest {i}"
            if bt.name is not None:
                name = bt.name
            plot_df[name] = metric
            for mkey in bt._add_metrics.keys():
                add_metric = bt._add_metrics[mkey]
                plot_add_df[mkey] = np.ma.masked_where(add_metric[1], add_metric[0])
                if not self.add_metric_exists:
                    self.add_metric_exists = True

        if self._live_plot_figsize is None:
            if add_metric_exists:
                self._live_plot_figsize = (10, 13)
            else:
                self._live_plot_figsize = (10, 6.5)

        if self.add_metric_exists:
            fig, axes = plt.subplots(2, 1, sharex=True, figsize=self._live_plot_figsize)
        else:
            fig, axes = plt.subplots(1, 1, sharex=True, figsize=self._live_plot_figsize)
            axes = [axes]

        if self._live_progress:
            axes[0].set_title(str(self._show_live_progress()))
        plot_df.plot(ax=axes[0])
        axes[0].set_ylim(bottom=0)

        if self.add_metric_exists:
            try:
                interp_df = plot_add_df.interpolate(method="linear")
                interp_df.plot(ax=axes[1], cmap="Accent")
                for line in self._add_metrics_lines:
                    plt.axvline(line[0], **line[1])
            except TypeError:
                pass

        fig.autofmt_xdate()
        if start_end is not None:
            plt.xlim([start_end[0], start_end[1]])
        else:
            plt.xlim([self.dates[0], self.dates[-1]])

        display.clear_output(wait=True)
        plt.draw()
        plt.pause(0.001)
        
    def _show_live_progress(self):
        _live_progress_pbar.n = self.i + 1
        return _live_progress_pbar

    def _update(self):
        for metric in self.metric.values():
            if metric._series:
                metric(write=True)
        self._capital = self._available_capital + self.metric["Portfolio Value"][-1]

    def _graceful_stop(self):
        if self._last_thread is not None:
            self._last_thread.join()
        self.plot(self._get_bts(), last=True)

    def _order(self, symbol, capital, as_percent=False, as_percent_available=False, shares=None):
        self._capital = self._available_capital + self.metric["Portfolio Value"]()
        if capital < 0:
            short = True
            capital = (-1) * capital
        else:
            short = False
        if not as_percent and not as_percent_available:
            if capital > self._available_capital:
                self._graceful_stop()
                raise InsufficientCapitalError("not enough capital available")
        elif as_percent:
            if abs(capital * self._capital) > self._available_capital:
                if not math.isclose(capital * self._capital, self._available_capital):
                    self._graceful_stop()
                    raise InsufficientCapitalError(
                        f"""
                        not enough capital available:
                        ordered {capital} * {self._capital}
                        with only {self._available_capital} available
                        """
                    )
        elif as_percent_available:
            if abs(capital * self._available_capital) > self._available_capital:
                if not math.isclose(capital * self._available_capital, self._available_capital):
                    self._graceful_stop()
                    raise InsufficientCapitalError(
                        f"""
                        not enough capital available:
                        ordered {capital} * {self._available_capital}
                        with only {self._available_capital} available
                        """
                    )
        current_price = self.price(symbol)
        if as_percent:
            capital = capital * self._capital
        if as_percent_available:
            capital = capital * self._available_capital
        try:
            if shares is None:
                num_shares, total = self._trade_cost(current_price, capital)
            else:
                num_shares, total = self._trade_cost(current_price, self._available_capital, num_shares=shares)
        except Exception as e:
            self._graceful_stop()
            raise e
        if short:
            num_shares *= -1
        if num_shares != 0:
            self._available_capital -= total
            self.portfolio._add(
                Position(self, symbol, self.current_date, self.event, num_shares)
            )
        else:
            _cls()
            raise Exception(
                f"""
                not enough capital specified to order a single share of {symbol}:
                tried to order {capital} of {symbol}
                with {symbol} price at {current_price}
                """
            )

    #def order_pct(self, symbol, capital):
    #    self._order(symbol, capital, as_percent=True)

    #def order_abs(self, symbol, capital):
    #    self._order(symbol, capital, as_percent=False)

    def long(self, symbol, **kwargs):
        if 'percent' in kwargs:
            self._order(symbol, kwargs['percent'], as_percent=True)
        if 'absolute' in kwargs:
            self._order(symbol, kwargs['absolute'])
        if 'percent_available' in kwargs:
            self._order(symbol, kwargs['percent_available'], as_percent_available=True)
        if 'nshares' in kwargs:
            self._order(symbol, 1, shares=kwargs['nshares'])

    def short(self, symbol, **kwargs):
        if 'percent' in kwargs:
            self._order(symbol, -kwargs['percent'], as_percent=True)
        if 'absolute' in kwargs:
            self._order(symbol, -kwargs['absolute'])
        if 'percent_available' in kwargs:
            self._order(symbol, -kwargs['percent_available'], as_percent_available=True)
        if 'nshares' in kwargs:
            self._order(symbol, -1, shares=kwargs['nshares'])

    def price(self, symbol):
        try:
            price = self.prices[symbol, self.current_date][self.event]
        except KeyError:
            try:
                self.prices.rem_cache(symbol)
                price = self.prices[symbol, self.current_date][self.event]
            except KeyError:
                self._graceful_stop()
                raise PriceUnavailableError(
                    symbol,
                    self.current_date,
                    f"""
                    Price for {symbol} on {self.current_date} could not be found.
                    """.strip(),
                )
        if math.isnan(price) or price is None:
            self._graceful_stop()
            raise PriceUnavailableError(
                symbol,
                self.current_date,
                f"""
                    Price for {symbol} on {self.current_date} is nan or None.
                    """.strip(),
            )
        return price

    @property
    def balance(self):
        @dataclass
        class Balance:
            start: float = self._start_capital
            current: float = self._available_capital

        return Balance()

    def _get_bts(self):
        bts = [self]
        if self._has_strategies:
            if self._no_iter:
                bts = self._strategies
            else:
                bts = bts + self._strategies
        return bts

    @property
    def metrics(self):
        bts = self._get_bts()
        dfs = []
        for i, bt in enumerate(bts):
            df = pd.DataFrame()
            df["Event"] = np.tile(["open", "close"], len(bt) // 2 + 1)[: len(bt)]
            df["Date"] = np.repeat(bt.dates, 2)
            if self._has_strategies:
                if bt.name is not None:
                    df["Backtest"] = np.repeat(bt.name, len(bt))
                else:
                    df["Backtest"] = np.repeat(f"Backtest {i}", len(bt))
            for key in bt.metric.keys():
                metric = bt.metric[key]
                if metric._series:
                    df[key] = metric.values
                if metric._single:
                    df[key] = np.repeat(metric.value, len(bt))
            dfs.append(df)
        if self._has_strategies:
            return pd.concat(dfs).set_index(["Backtest", "Date", "Event"])
        else:
            return pd.concat(dfs).set_index(["Date", "Event"])

    @property
    def summary(self):
        bts = self._get_bts()
        dfs = []
        for i, bt in enumerate(bts):
            df = pd.DataFrame()
            if self._has_strategies:
                if bt.name is not None:
                    df["Backtest"] = [bt.name]
                else:
                    df["Backtest"] = [f"Backtest {i}"]
            for key in bt.metric.keys():
                metric = bt.metric[key]
                if metric._series:
                    df[f"{key} (Last Value)"] = [metric[-1]]
                if metric._single:
                    df[key] = [metric.value]
            dfs.append(df)
        if self._has_strategies:
            return pd.concat(dfs).set_index(["Backtest"])
        else:
            return df

    @property
    def strategies(self):
        if self._has_strategies:
            return StrategySequence(self)

    @property
    def pf(self):
        return self.portfolio

    def _set_strategies(
        self, strategies: List[Callable[["Date", str, "Backtester"], None]]
    ):
        self._strategies_call = copy.deepcopy(strategies)
        for strat in strategies:
            new_bt = copy.deepcopy(self)
            new_bt._set_self()
            new_bt.name = strat.name
            new_bt._has_strategies = False
            self._init_iter(new_bt)
            self._strategies.append(new_bt)

    def _run_once(self):
        for i, bt in enumerate(self._strategies):
            self._strategies_call[i](*self._next_iter(bt))

    def plot(self, bts, last=False):
        try:
            if self._live_plot and (self.i % self._live_plot_every == 0 or last):
                if self._last_thread is None or not self._last_thread.is_alive():
                    thr = threading.Thread(target=self._show_live_plot, args=(bts,))
                    thr.start()
                    self._last_thread = thr
                if last:
                    self._last_thread.join()
                    self._show_live_plot(bts)
            if self._live_metrics and (self.i % self._live_metrics_every == 0 or last):
                self._show_live_metrics(bts)
            if not (self._live_metrics or self._live_plot) and (
                self.i % self._live_progress_every == 0 or last
            ):
                _cls()
                print(self._show_live_progress())
        except:
            pass

    def show(self, start=None, end=None):
        bts = self._get_bts()
        if self._from_sequence:
            bts = [self]
        if start is not None or end is not None:
            self._show_live_plot(bts, [start, end])
        else:
            self._show_live_plot(bts)

    def run(self):
        self._init_iter()
        self._no_iter = True

        for _ in range(len(self)):
            self._run_once()
            self.i = self._strategies[-1].i
            self.plot(self._strategies)

        self.plot(self._strategies, last=True)

        for strat in self._strategies:
            for metric in strat.metric.values():
                if metric._single:
                    metric(write=True)
