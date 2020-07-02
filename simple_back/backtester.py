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
from IPython.display import clear_output, display, HTML, display_html
from dataclasses import dataclass
import multiprocessing
import threading
from collections.abc import MutableSequence
import uuid
import time
import tabulate

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
    TotalReturn,
    Metric,
)
from .strategy import Strategy, BuyAndHold
from .exceptions import BacktestRunError, LongShortLiquidationError, NegativeValueError
from .utils import is_notebook, _cls

# matplotlib is not a strict requirement, only needed for live_plot
try:
    import pylab as pl
    import matplotlib.pyplot as plt
    import matplotlib

    plt_exists = True
except ImportError:
    plt_exists = False

# tqdm is not a strict requirement
try:
    from tqdm import tqdm

    tqdm_exists = True
except ImportError:
    tqdm_exists = False

# from https://stackoverflow.com/a/44923103, https://stackoverflow.com/a/50899244
def display_side_by_side(bts):
    html_str = ""
    for bt in bts:
        styler = bt.logs.style.set_table_attributes(
            "style='display:inline'"
        ).set_caption(bt.name)
        html_str += styler._repr_html_()
    display_html(html_str, raw=True)


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


class Position:
    """Tracks a single position in a portfolio or trade history.
    """

    def __init__(
        self,
        bt: "Backtester",
        symbol: str,
        date: datetime.date,
        event: str,
        nshares: int,
        uid: str,
        fee: float,
        slippage: float = None,
    ):
        self.symbol = symbol
        self.date = date
        self.event = event
        self._nshares_int = nshares
        self.start_price = bt.price(symbol)
        if slippage is not None:
            if nshares < 0:
                self.start_price *= 1 + slippage
            if nshares > 0:
                self.start_price *= 1 - slippage
        self._slippage = slippage
        self._bt = bt
        self._frozen = False
        self._uid = uid
        self.fee = fee

    def _attr(self):
        return [attr for attr in dir(self) if not attr.startswith("_")]

    def __repr__(self) -> str:
        result = {}
        for attr in self._attr():
            val = getattr(self, attr)
            if isinstance(val, float):
                result[attr] = f"{val:.2f}"
            else:
                result[attr] = str(val)
        return json.dumps(result, sort_keys=True, indent=2)

    @property
    def _short(self) -> bool:
        """True if this is a short position.
        """
        return self._nshares_int < 0

    @property
    def _long(self) -> bool:
        """True if this is a long position.
        """
        return self._nshares_int > 0

    @property
    def value(self) -> float:
        """Returns the current market value of the position.
        """
        if self._short:
            old_val = self.initial_value
            cur_val = self.nshares * self.price
            return old_val + (old_val - cur_val)
        if self._long:
            return self.nshares * self.price

    @property
    def price(self) -> float:
        """Returns the current price if the position is held in a portfolio.
        Returns the last price if the position was liquidated and is part of a trade history.
        """
        if self._frozen:
            result = self._bt.prices[self.symbol, self.end_date][self.end_event]
        else:
            result = self._bt.price(self.symbol)
        if self._slippage is not None:
            if self._short:
                result *= 1 - self._slippage
            if self._long:
                result *= 1 + self._slippage
        return result

    @property
    def value_pershare(self) -> float:
        """Returns the value of the position per share.
        """
        if self._long:
            return self.price
        if self._short:
            return self.start_price + (self.start_price - self.price)

    @property
    def initial_value(self) -> float:
        """Returns the initial value of the position, including fees.
        """
        if self._short:
            return self.nshares * self.start_price + self.fee
        if self._long:
            return self.nshares * self.start_price + self.fee

    @property
    def profit_loss_pct(self) -> float:
        """Returns the profit/loss associated with the position (not including commission)
        in relative terms.
        """
        return self.value / self.initial_value - 1

    @property
    def profit_loss_abs(self) -> float:
        """Returns the profit/loss associated with the position (not including commission)
        in absolute terms.
        """
        return self.value - self.initial_value

    @property
    def nshares(self) -> int:
        """Returns the number of shares in the position.
        """
        return abs(self._nshares_int)

    @property
    def order_type(self) -> str:
        """Returns "long" or "short" based on the position type.
        """
        t = None
        if self._short:
            t = "short"
        if self._long:
            t = "long"
        return t

    def _remove_shares(self, n):
        if self._short:
            self._nshares_int += n
        if self._long:
            self._nshares_int -= n

    def _freeze(self):
        self._frozen = True
        self.end_date = self._bt.current_date
        self.end_event = self._bt.event


class Portfolio(MutableSequence):
    """A portfolio is a collection of :class:`.Position` objects,
    and can be used to :meth:`.liquidate` a subset of them.
    """

    def __init__(self, bt, positions: List[Position] = []):
        self.positions = positions
        self.bt = bt

    @property
    def total_value(self) -> float:
        """Returns the total value of the portfolio.
        """
        val = 0
        for pos in self.positions:
            val += pos.value
        return val

    @property
    def df(self) -> pd.DataFrame:
        pos_dict = {}
        for pos in self.positions:
            for col in pos._attr():
                if col not in pos_dict:
                    pos_dict[col] = []
                pos_dict[col].append(getattr(pos, col))
        return pd.DataFrame(pos_dict)

    def _get_by_uid(self, uid) -> Position:
        for pos in self.positions:
            if pos._uid == uid:
                return pos

    def __repr__(self) -> str:
        return self.positions.__repr__()

    def liquidate(self, nshares: int = -1, _bt: "Backtester" = None):
        """Liquidate all positions in the current "view" of the portfolio.
        If no view is given using `['some_ticker']`, :meth:`.filter`,
        :meth:`.Portfolio.long` or :meth:`.Portfolio.short`,
        an attempt to liquidate all positions is made.

        Args:
            nshares:
                The number of shares to be liquidated.
                This should only be used when a ticker is selected using `['some_ticker']`.

        Examples:
            Select all `MSFT` positions and liquidate them::

                bt.portfolio['MSFT'].liquidate()

            Liquidate 10 `MSFT` shares::

                bt.portfolio['MSFT'].liquidate(nshares=10)

            Liquidate all long positions::

                bt.portfolio.long.liquidate()

            Liquidate all positions that have lost more than 5% in value.
            We can either use :meth:`.filter` or the dataframe as indexer
            (in this case in combination with the pf shorthand)::

                bt.pf[bt.pf.df['profit_loss_pct'] < -0.05].liquidate()
                # or
                bt.pf.filter(lambda x: x.profit_loss_pct < -0.05)
        """
        bt = _bt
        if bt is None:
            bt = self.bt
            if bt._slippage is not None:
                self.liquidate(nshares, bt.lower_bound)
        is_long = False
        is_short = False
        for pos in self.positions:
            if pos._long:
                is_long = True
            if pos._short:
                is_short = True
        if is_long and is_short:
            bt._graceful_stop()
            raise LongShortLiquidationError(
                "liquidating a mix of long and short positions is not possible"
            )
        for pos in copy.copy(self.positions):
            pos = bt.pf._get_by_uid(pos._uid)
            if nshares == -1 or nshares >= pos.nshares:
                bt._available_capital += pos.value
                if bt._available_capital < 0:
                    bt._graceful_stop()
                    raise NegativeValueError(
                        f"Tried to liquidate position resulting in negative capital {bt._available_capital}."
                    )
                bt.portfolio._remove(pos)

                pos._freeze()
                bt.trades._add(copy.copy(pos))

                if nshares != -1:
                    nshares -= pos.nshares
            elif nshares > 0 and nshares < pos.nshares:
                bt._available_capital += pos.value_pershare * nshares
                pos._remove_shares(nshares)

                hist = copy.copy(pos)
                hist._freeze()
                if hist._short:
                    hist._nshares_int = (-1) * nshares
                if hist._long:
                    hist._nshares_int = nshares
                bt.trades._add(hist)

                break

    def _add(self, position):
        self.positions.append(position)

    def _remove(self, position):
        self.positions = [pos for pos in self.positions if pos._uid != position._uid]

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

    def insert(self, index: int, value: Position) -> None:
        self.positions.insert(index, value)

    @property
    def short(self) -> "Portfolio":
        """Returns a view of the portfolio (which can be treated as its own :class:`.Portfolio`)
        containing all *short* positions.
        """
        new_pos = []
        for pos in self.positions:
            if pos._short:
                new_pos.append(pos)
        return Portfolio(self.bt, new_pos)

    @property
    def long(self) -> "Portfolio":
        """Returns a view of the portfolio (which can be treated as its own :class:`.Portfolio`)
        containing all *long* positions.
        """
        new_pos = []
        for pos in self.positions:
            if pos._long:
                new_pos.append(pos)
        return Portfolio(self.bt, new_pos)

    def filter(self, func: Callable[[Position], bool]) -> "Portfolio":
        """Filters positions using any :class`.Callable`

        Args:
            func: The function/callable to do the filtering.
        """
        new_pos = []
        for pos in self.positions:
            if func(pos):
                new_pos.append(pos)
        return Portfolio(self.bt, new_pos)

    def attr(self, attribute: str) -> List:
        """Get a list of values for a certain value for all posititions

        Args:
            attribute:
                String name of the attribute to get.
                Can be any attribute of :class:`.Position`.
        """
        self.bt._warn.append(
            f"""
        .attr will be removed in 0.7
        you can use b.portfolio.df[{attribute}]
        instead of b.portfolio.attr('{attribute}')
        """
        )
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
        self = copy.deepcopy(self)
        self.bt.name = name
        return self

    def balance(self, amount: int) -> "BacktesterBuilder":
        """**Required**, set the starting balance for all :class:`.Strategy` objects
        run with the :class:`.Backtester`

        Args:
            amount: The starting balance.
        """
        self = copy.deepcopy(self)
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
        self = copy.deepcopy(self)
        self.bt.prices = prices
        self.bt.prices.bt = self.bt
        return self

    def data(self, data: DataProvider) -> "BacktesterBuilder":
        """**Optional**, add a :class:`.DataProvider` to use external data without time leaks.

        Args:
            data: The data provider.
        """
        self = copy.deepcopy(self)
        self.bt.data[data.name] = data
        data.bt = self.bt
        return self

    def trade_cost(
        self, trade_cost: Union[Fee, Callable[[float, float], Tuple[float, int]]]
    ) -> "BacktesterBuilder":
        """**Optional**, set a :class:`.Fee` to be applied when buying shares.
        When not set, :class:`.NoFee` is used.

        Args:
            trade_cost: one ore more :class:`.Fee` objects or callables.
        """
        self = copy.deepcopy(self)
        self.bt._trade_cost = trade_cost
        return self

    def metrics(self, metrics: Union[Metric, List[Metric]]) -> "BacktesterBuilder":
        """**Optional**, set additional :class:`.Metric` objects to be used.

        Args:
            metrics: one or more :class:`.Metric` objects
        """
        self = copy.deepcopy(self)
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
        """**Optional**, remove all default metrics,
        except :class:`.PortfolioValue`, which is needed internally.
        """
        self = copy.deepcopy(self)
        metrics = [PortfolioValue()]
        self.bt.metric = {}
        self.bt.metric(metrics)
        return self

    def calendar(self, calendar: str) -> "BacktesterBuilder":
        """**Optional**, set a `pandas market calendar`_ to be used.
        If not called, "NYSE" is used.

        Args:
            calendar: the calendar identifier

        .. _pandas market calendar:
           https://pandas-market-calendars.readthedocs.io/en/latest/calendars.html
        """
        self = copy.deepcopy(self)
        self.bt._calendar = calendar
        return self

    def live_metrics(self, every: int = 10) -> "BacktesterBuilder":
        """**Optional**, shows all metrics live in output. This can be useful
        when running simple-back from terminal.

        Args:
            every: how often metrics should be updated
            (in events, e.g. 10 = 5 days)
        """
        self = copy.deepcopy(self)
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

    def no_live_metrics(self) -> "BacktesterBuilder":
        """Disables showing live metrics.
        """
        self = copy.deepcopy(self)
        self.bt._live_metrics = False
        return self

    def live_plot(
        self,
        every: int = None,
        metric: str = "Total Value",
        figsize: Tuple[float, float] = None,
        min_y: int = 0,
        blocking: bool = False,
    ) -> "BacktesterBuilder":
        """**Optional**, shows the backtest results live using matplotlib.
        Can only be used in notebooks.

        Args:
            every:
                how often metrics should be updated
                (in events, e.g. 10 = 5 days)
                the regular default is 10,
                blocking default is 100
            metric: which metric to plot
            figsize: size of the plot
            min_y:
                minimum value on the y axis, set to `None`
                for no lower limit
            blocking:
                will disable threading for plots and
                allow live plotting in terminal,
                this will slow down the backtester
                significantly
        """
        self = copy.deepcopy(self)
        if self.bt._live_metrics:
            warn(
                """
                live metrics and plotting cannot be used together,
                 setting live metrics to false
                """
            )
            self.bt._live_metrics = False
        if is_notebook():
            if every is None:
                every = 10
        elif not blocking:
            warn(
                """
                live plots use threading which is not supported
                with matplotlib outside notebooks. to disable
                threading for live plots, you can call
                live_plot with ``blocking = True``.

                live_plot set to false.
                """
            )
            return self
        elif blocking:
            self.bt._live_plot_blocking = True
            if every is None:
                every = 100

        self.bt._live_plot = True
        self.bt._live_plot_every = every
        self.bt._live_plot_metric = metric
        self.bt._live_plot_figsize = figsize
        self.bt._live_plot_min = min_y

        return self

    def no_live_plot(self) -> "BacktesterBuilder":
        """Disables showing live plots.
        """
        self = copy.deepcopy(self)
        self.bt._live_plot = False
        return self

    def live_progress(self, every: int = 10) -> "BacktesterBuilder":
        """**Optional**, shows a live progress bar using :class:`.tqdm`, either
        as port of a plot or as text output.
        """
        self = copy.deepcopy(self)
        self.bt._live_progress = True
        self.bt._live_progress_every = every
        return self

    def no_live_progress(self) -> "BacktesterBuilder":
        """Disables the live progress bar.
        """
        self = copy.deepcopy(self)
        self.bt._live_progress = False
        return self

    def compare(
        self,
        strategies: List[
            Union[Callable[["datetime.date", str, "Backtester"], None], Strategy, str]
        ],
    ):
        """**Optional**, alias for :meth:`.BacktesterBuilder.strategies`,
        should be used when comparing to :class:`.BuyAndHold` of a ticker instead of other strategies.

        Args:
            strategies:
                should be the string of the ticker to compare to,
                but :class:`.Strategy` objects can be passed as well
        """
        self = copy.deepcopy(self)
        return self.strategies(strategies)

    def strategies(
        self,
        strategies: List[
            Union[Callable[["datetime.date", str, "Backtester"], None], Strategy, str]
        ],
    ) -> "BacktesterBuilder":
        """**Optional**, sets :class:`.Strategy` objects to run.

        Args:
            strategies:
                list of :class:`.Strategy` objects or tickers to :class:`.BuyAndHold`
        """
        self = copy.deepcopy(self)
        strats = []
        for strat in strategies:
            if isinstance(strat, str):
                strats.append(BuyAndHold(strat))
            else:
                strats.append(strat)
        self.bt._temp_strategies = strats
        self.bt._has_strategies = True
        return self

    def slippage(self, slippage: int = 0.0005):
        """**Optional**, sets slippage which will create a (lower bound) strategy.
        The orginial strategies will run without slippage.

        Args:
            slippage:
                the slippage in percent of the base price,
                default is equivalent to quantopian default for US Equities
        """
        self = copy.deepcopy(self)
        self.bt._slippage = slippage
        return self

    def build(self) -> "Backtester":
        """Build a :class:`.Backtester` given the previous configuration.
        """
        self = copy.deepcopy(self)
        self.bt._builder = self
        return copy.deepcopy(self.bt)


class Backtester:
    """The :class:`.Backtester` object is yielded alongside
    the current day and event (open or close)
    when it is called with a date range,
    which can be of the following forms.
    The :class:`.Backtester` object stores information
    about the backtest after it has completed.

    Examples:

        Initialize with dates as strings::

            bt['2010-1-1','2020-1-1'].run()
            # or
            for day, event, b in bt['2010-1-1','2020-1-1']:
                ...
        
        Initialize with dates as :class:`.datetime.date` objects::

            bt[datetime.date(2010,1,1),datetime.date(2020,1,1)]

        Initialize with dates as :class:`.int`::

            bt[-100:] # backtest 100 days into the past
    """

    def __getitem__(self, date_range: slice) -> "Backtester":
        if self._run_before:
            raise BacktestRunError(
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

    def _init_slippage(self, bt=None):
        if bt is None:
            bt = self
        lower_bound = copy.deepcopy(bt)
        lower_bound._strategies = []
        lower_bound._set_self()
        lower_bound.name += " (lower bound)"
        lower_bound._has_strategies = False
        lower_bound._slippage_percent = (-1) * self._slippage
        lower_bound._slippage = None
        lower_bound._init_iter(lower_bound)
        bt.lower_bound = lower_bound
        self._strategies.append(lower_bound)

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
            TotalReturn(),
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
        self._live_plot_metric = "Total Value"
        self._live_plot_figsize = None
        self._live_plot_min = None
        self._live_plot_axes = None
        self._live_plot_blocking = False
        self._live_metrics = False
        self._live_progress = False

        self._strategies = []
        self._temp_strategies = []
        self._has_strategies = False
        self.name = "Backtest"

        self._no_iter = False

        self._schedule = None
        self._warn = []
        self._log = []

        self._add_metrics = {}
        self._add_metrics_lines = []

        self.datetimes = None
        self.add_metric_exists = False

        self._run_before = False

        self._last_thread = None

        self._from_sequence = False

        self._slippage = None
        self._slippage_percent = None

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
        if self._slippage is not None and not self._no_iter:
            self._init_slippage(self)
            self._has_strategies = True
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
                self._plot(self._get_bts(), last=True)
                raise StopIteration
        bt._update()
        return bt.current_date, bt.event, bt

    def add_metric(self, key: str, value: float):
        """Called inside the backtest, adds a metric that is visually tracked.

        Args:
            key: the metric name
            value: the numerical value of the metric
        """
        if key not in self._add_metrics:
            self._add_metrics[key] = (
                np.repeat(np.nan, len(self)),
                np.repeat(True, len(self)),
            )
        self._add_metrics[key][0][self.i] = value
        self._add_metrics[key][1][self.i] = False

    def add_line(self, **kwargs):
        """Adds a vertical line on the plot on the current date + event.
        """
        self._add_metrics_lines.append((self.timestamp, kwargs))

    def log(self, text: str):
        """Adds a log text on the current day and event that can be accessed using :obj:`.logs`
        after the backtest has completed.

        Args:
            text: text to log
        """
        self._log.append([self.current_date, self.event, text])

    @property
    def timestamp(self):
        """Returns the current timestamp, which includes the correct open/close time,
        depending on the calendar that was set using :meth:`.BacktesterBuilder.calendar`
        """
        return self._schedule.loc[self.current_date][f"market_{self.event}"]

    def __iter__(self):
        return self._init_iter()

    def __next__(self):
        if len(self._strategies) > 0:
            result = self._next_iter()
            self._run_once()
            self._plot([self] + self._strategies)
        else:
            result = self._next_iter()
            self._plot([self])
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
        main_col = []
        bound_col = []
        for i, bt in enumerate(bts):
            metric = bt.metric[self._live_plot_metric].values
            name = f"Backtest {i}"
            if bt.name is not None:
                name = bt.name
            if bt._slippage_percent is not None:
                bound_col.append(name)
            else:
                main_col.append(name)
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
            fig, axes = plt.subplots(
                2, 1, sharex=True, figsize=self._live_plot_figsize, num=0
            )
        else:
            fig, axes = plt.subplots(
                1, 1, sharex=True, figsize=self._live_plot_figsize, num=0
            )
            axes = [axes]

        if self._live_progress:
            axes[0].set_title(str(self._show_live_progress()))
        plot_df[main_col].plot(ax=axes[0])
        try:
            if self._slippage is not None:
                for col in main_col:
                    axes[0].fill_between(
                        plot_df.index,
                        plot_df[f"{col} (lower bound)"],
                        plot_df[f"{col}"],
                        alpha=0.1,
                    )
        except KeyError:
            pass
        if self._live_plot_min is not None:
            axes[0].set_ylim(bottom=self._live_plot_min)
        plt.tight_layout()

        if self.add_metric_exists:
            try:
                interp_df = plot_add_df.interpolate(method="linear")
                interp_df.plot(ax=axes[1], cmap="Accent")
                for bt in bts:
                    for line in bt._add_metrics_lines:
                        plt.axvline(line[0], **line[1])
            except TypeError:
                pass

        fig.autofmt_xdate()
        if start_end is not None:
            plt.xlim([start_end[0], start_end[1]])
        else:
            plt.xlim([self.dates[0], self.dates[-1]])

        clear_output(wait=True)

        plt.draw()
        plt.pause(0.001)
        if self._live_plot_blocking:
            plt.clf()  # needed to prevent overlapping tick labels

        captions = []

        for bt in bts:
            captions.append(bt.name)

        has_logs = False

        for bt in bts:
            if len(bt._log) > 0:
                has_logs = True

        if has_logs:
            display_side_by_side(bts)

        for w in self._warn:
            warn(w)

    def _show_live_progress(self):
        _live_progress_pbar.n = self.i + 1
        return _live_progress_pbar

    def _update(self):
        for metric in self.metric.values():
            if metric._series:
                try:
                    metric(write=True)
                except PriceUnavailableError as e:
                    if self.event == "close":
                        self.i -= 2
                    if self.event == "open":
                        self.i -= 1

                    self._warn.append(
                        f"{e.symbol} discontinued on {self.current_date}, liquidating at previous day's {self.event} price"
                    )

                    self.current_date = self.dates[(self.i // 2)]

                    self.portfolio[e.symbol].liquidate()
                    metric(write=True)

                    if self.event == "close":
                        self.i += 2
                    if self.event == "open":
                        self.i += 1
                    self.current_date = self.dates[(self.i // 2)]

        self._capital = self._available_capital + self.metric["Portfolio Value"][-1]

    def _graceful_stop(self):
        if self._last_thread is not None:
            self._last_thread.join()
            del self._last_thread
        self._plot(self._get_bts(), last=True)

    def _order(
        self,
        symbol,
        capital,
        as_percent=False,
        as_percent_available=False,
        shares=None,
        uid=None,
    ):
        if uid is None:
            uid = uuid.uuid4()
        if self._slippage is not None:
            self.lower_bound._order(
                symbol, capital, as_percent, as_percent_available, shares, uid
            )

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
                if not math.isclose(
                    capital * self._available_capital, self._available_capital
                ):
                    self._graceful_stop()
                    raise InsufficientCapitalError(
                        f"""
                        not enough capital available:
                        ordered {capital} * {self._available_capital}
                        with only {self._available_capital} available
                        """
                    )
        current_price = self.price(symbol)

        if self._slippage_percent is not None:
            if short:
                current_price *= 1 + self._slippage_percent
            else:
                current_price *= 1 - self._slippage_percent

        if as_percent:
            capital = capital * self._capital
        if as_percent_available:
            capital = capital * self._available_capital
        try:
            if shares is None:
                fee_dict = self._trade_cost(current_price, capital)
                nshares, total, fee = (
                    fee_dict["nshares"],
                    fee_dict["total"],
                    fee_dict["fee"],
                )
            else:
                fee_dict = self._trade_cost(
                    current_price, self._available_capital, nshares=shares
                )
                nshares, total, fee = (
                    fee_dict["nshares"],
                    fee_dict["total"],
                    fee_dict["fee"],
                )
        except Exception as e:
            self._graceful_stop()
            raise e
        if short:
            nshares *= -1
        if nshares != 0:
            self._available_capital -= total
            pos = Position(
                self,
                symbol,
                self.current_date,
                self.event,
                nshares,
                uid,
                fee,
                self._slippage_percent,
            )
            self.portfolio._add(pos)
        else:
            _cls()
            raise Exception(
                f"""
                not enough capital specified to order a single share of {symbol}:
                tried to order {capital} of {symbol}
                with {symbol} price at {current_price}
                """
            )

    def long(self, symbol: str, **kwargs):
        """Enter a long position of the given symbol.

        Args:
            symbol: the ticker to buy
            kwargs:
                one of either
                "percent" as a percentage of total value (cash + positions),
                "absolute" as an absolute value,
                "percent_available" as a percentage of remaining funds (excluding positions)
                "nshares" as a number of shares
        """
        if "percent" in kwargs:
            self._order(symbol, kwargs["percent"], as_percent=True)
        if "absolute" in kwargs:
            self._order(symbol, kwargs["absolute"])
        if "percent_available" in kwargs:
            self._order(symbol, kwargs["percent_available"], as_percent_available=True)
        if "nshares" in kwargs:
            self._order(symbol, 1, shares=kwargs["nshares"])

    def short(self, symbol: str, **kwargs):
        """Enter a short position of the given symbol.

        Args:
            symbol: the ticker to short
            kwargs:
                one of either
                "percent" as a percentage of total value (cash + positions),
                "absolute" as an absolute value,
                "percent_available" as a percentage of remaining funds (excluding positions)
                "nshares" as a number of shares
        """
        if "percent" in kwargs:
            self._order(symbol, -kwargs["percent"], as_percent=True)
        if "absolute" in kwargs:
            self._order(symbol, -kwargs["absolute"])
        if "percent_available" in kwargs:
            self._order(symbol, -kwargs["percent_available"], as_percent_available=True)
        if "nshares" in kwargs:
            self._order(symbol, -1, shares=kwargs["nshares"])

    def price(self, symbol: str) -> float:
        """Get the current price of a given symbol.

        Args:
            symbol: the ticker
        """
        try:
            price = self.prices[symbol, self.current_date][self.event]
        except KeyError:
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
    def balance(self) -> "Balance":
        """Get the current or starting balance.

        Examples:

            Get the current balance::

                bt.balance.current

            Get the starting balance::

                bt.balance.start
        """

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
    def metrics(self) -> pd.DataFrame:
        """Get a dataframe of all metrics collected during the backtest(s).
        """
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
    def summary(self) -> pd.DataFrame:
        """Get a dataframe showing the last and overall values of all metrics
        collected during the backtest.
        This can be helpful for comparing backtests at a glance.
        """
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
        """Provides access to sub-strategies, returning a :class:`.StrategySequence`.
        """
        if self._has_strategies:
            return StrategySequence(self)

    @property
    def pf(self) -> Portfolio:
        """Shorthand for `portfolio`, returns the backtesters portfolio.
        """
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
            if self._slippage is not None:
                self._init_slippage(new_bt)

            # this is bad but not bad enough to
            # do anything other than this hotfix
            self._no_iter = True
            self._init_iter(new_bt)
            self._no_iter = False

            self._strategies.append(new_bt)

    def _run_once(self):
        no_slip_strats = [
            strat for strat in self._strategies if strat._slippage_percent is None
        ]
        slip_strats = [
            strat for strat in self._strategies if strat._slippage_percent is not None
        ]
        for bt in slip_strats:
            self._next_iter(bt)
        for i, bt in enumerate(no_slip_strats):
            self._strategies_call[i](*self._next_iter(bt))

    def _plot(self, bts, last=False):
        try:
            if self._live_plot and (self.i % self._live_plot_every == 0 or last):
                if not self._live_plot_blocking:
                    if self._last_thread is None or not self._last_thread.is_alive():
                        thr = threading.Thread(target=self._show_live_plot, args=(bts,))
                        thr.start()
                        self._last_thread = thr
                    if last:
                        self._last_thread.join()
                        self._show_live_plot(bts)
                else:
                    self._show_live_plot(bts)
            if self._live_metrics and (self.i % self._live_metrics_every == 0 or last):
                self._show_live_metrics(bts)
            if (
                not (self._live_metrics or self._live_plot)
                and self._live_progress
                and (self.i % self._live_progress_every == 0 or last)
            ):
                _cls()
                print(self._show_live_progress())
                for l in self._log[-20:]:
                    print(l)
                if len(self._log) > 20:
                    print("... more logs stored in Backtester.logs")
                for w in self._warn:
                    warn(w)
        except:
            pass

    @property
    def logs(self) -> pd.DataFrame:
        """Returns a :class:`.pd.DataFrame` for logs collected during the backtest.
        """
        df = pd.DataFrame(self._log, columns=["date", "event", "log"])
        df = df.set_index(["date", "event"])
        return df

    def show(self, start=None, end=None):
        """Show the backtester as a plot.
        
        Args:
            start: the start date
            end: the end date
        """
        bts = self._get_bts()
        if self._from_sequence:
            bts = [self]
        if start is not None or end is not None:
            self._show_live_plot(bts, [start, end])
        else:
            self._show_live_plot(bts)
        if not is_notebook():
            plt.show()

    def run(self):
        """Run the backtesters strategies without using an iterator.
        This is only possible if strategies have been set using :meth:`.BacktesterBuilder.strategies`.
        """
        self._no_iter = True
        self._init_iter()

        for _ in range(len(self)):
            self._run_once()
            self.i = self._strategies[-1].i
            self._plot(self._strategies)

        self._plot(self._strategies, last=True)

        for strat in self._strategies:
            for metric in strat.metric.values():
                if metric._single:
                    metric(write=True)
