from abc import ABC, abstractmethod
from yahoo_fin.stock_info import get_data
from typing import Union, List, Optional, Tuple
import pandas as pd
from dateutil.relativedelta import relativedelta
from datetime import date
import numpy as np
import diskcache as dc


class TimeLeakError(Exception):
    def __init__(self, current_date, requested_date, message):
        self.current_date = current_date
        self.requested_date = (requested_date,)
        self.message = message

class DailyDataProvider(ABC):
    def __init__(self):
        self.current_date = date.today()
        self.current_event = self.columns[np.argmax(self.columns_order)]
        self.cache = dc.Cache(".simple-back")

    def _remove_leaky_vals(self, df, cols, date):
        if isinstance(df, pd.DataFrame):
            if type(date) == str:
                date = pd.to_datetime(date)
            istoday = type(date) != slice and date == self.current_date
            if istoday or (df is not None and self.current_date in df.index):
                cur_order = self.columns.index(self.current_event)
                if type(cols) == str:
                    cols = [cols]
                for i, col in enumerate(cols):
                    if self.columns_order[i] > cur_order:
                        if istoday:
                            df[col] = None
                        else:
                            df.at[self.current_date, col] = None
            if type(date) == slice and not df.empty:
                sum_recent = (df.index.date > self.current_date).sum()
                if sum_recent > 0:
                    raise TimeLeakError(
                        self.current_date,
                        df.index.date[-1],
                        f"""
                        {sum_recent} dates in index
                        more recent than {self.current_date}
                        """,
                    )
            elif type(date) != slice:
                if date > self.current_date:
                    raise TimeLeakError(
                        self.current_date,
                        date,
                        f"""
                        {date} is more recent than {self.current_date},
                        resulting in time leak
                        """,
                    )
        return df

    def _get_order(self, event):
        return self.columns_order[self.columns.index(event)]

    @property
    def _max_order(self):
        max_order = min(self.columns_order)
        if type(self.current_event) == list:
            for event in self.current_event:
                order = self._get_order(event)
                if order > max_order:
                    max_order = order
        if type(self.current_event) == str:
            max_order = self._get_order(self.current_event)
        return max_order

    def __getitem__(
        self,
        symbol_date_event: Union[
            str,
            List[str],
            Tuple[
                Union[List[str], str],
                Optional[Union[slice, object]],
                Optional[Union[List[str], str]],
            ],
        ],
    ) -> pd.DataFrame:
        """
        Expects a tuple of (ticker_symbol, date, 'open' or 'close')
        and returns the price
        for said symbol at that point in time.
        ````
        my_price_provider['AAPL', date(2015,1,1), 'open']
        ````
        """
        try:
            self.current_date = self.bt.current_date
            self.current_event = self.bt.event
        except AttributeError:
            pass
        if type(symbol_date_event) is not str:
            len_t = len(symbol_date_event)
        else:
            len_t = 0

        if len_t >= 0:
            symbol = symbol_date_event
            date = slice(None, self.current_date)
            event = self.columns
        if len_t >= 1:
            symbol = symbol_date_event[0]
        if len_t >= 2:
            date = symbol_date_event[1]
            if type(date) is date:
                if date > self.current_date:
                    raise TimeLeakError(
                        self.current_date,
                        date,
                        f"""
                        {date} is more recent than {self.current_date},
                        resulting in time leak
                        """,
                    )
            if type(date) == slice:
                if date == slice(None, None):
                    date = slice(None, self.current_date)
                if date.stop is None:
                    date = slice(date.start, self.current_date, date.step)
                stop_date = date.stop
                if type(date.stop) == relativedelta:
                    date = slice(date.start, self.current_date + date.stop, date.step)
                if type(date.start) == relativedelta:
                    if type(stop_date) == str:
                        stop_date = pd.to_datetime(stop_date)
                    date = slice(stop_date + date.start, date.stop, date.step)
                stop_date = date.stop
                if type(date.stop) == int and date.stop <= 0:
                    date = slice(
                        date.start,
                        self.current_date - relativedelta(days=-1 * date.stop),
                        date.step,
                    )
                if type(date.start) == int and date.start <= 0:
                    stop_date = date.stop
                    if type(stop_date) == str:
                        stop_date = pd.to_datetime(stop_date)
                    date = slice(
                        stop_date - relativedelta(days=-1 * date.start),
                        date.stop,
                        date.step,
                    )
        data = self._get_cached(symbol, date, event)
        return self._remove_leaky_vals(data, event, date)

    def _get_cached(self, *args) -> pd.DataFrame:
        if args in self.cache:
            return self.cache.get(args)
        else:
            result = self.get(args[0], args[1], args[2])
            self.cache.set(args, result)
            return result

    @abstractmethod
    def get(
        self, symbol: str, date: Union[slice, date], event: Union[str, List[str]]
    ) -> pd.DataFrame:
        pass

    @property
    @abstractmethod
    def columns(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def columns_order(self) -> List[int]:
        pass

    def get_cache(self, key):
        return self.cache[str(key) + str(type(self))]

    def in_cache(self, key):
        return str(key) + str(type(self)) in self.cache

    def set_cache(self, key, val):
        self.cache.set(str(key) + str(type(self)), val)

    def rem_cache(self, key):
        del self.cache[str(key) + str(type(self))]

    def clear_cache(self):
        self.cache.clear()


class DailyPriceProvider(DailyDataProvider):
    def __init__(self, highlow=True):
        self.highlow = highlow
        super().__init__()

    @property
    def columns(self):
        if self.highlow:
            return ["open", "close", "high", "low"]
        else:
            return ["open", "close"]

    @property
    def columns_order(self):
        if self.highlow:
            return [0, 1, 1, 1]
        else:
            return [0, 1]

    @abstractmethod
    def get(
        self, symbol: str, date: Union[slice, date], event: Union[str, List[str]]
    ) -> pd.DataFrame:
        pass


class YahooFinanceProvider(DailyPriceProvider):
    def __init__(self, highlow=False, adjust_prices=True):
        self.adjust_prices = adjust_prices
        self.highlow = highlow
        super().__init__()

    def get(
        self, symbol: str, date: Union[slice, date], event: Union[str, List[str]]
    ) -> pd.DataFrame:

        if not self.in_cache(symbol):
            self.set_cache(symbol, get_data(symbol))
        df = self.get_cache(symbol)
        entry = df.loc[date].copy()
        adj = entry["adjclose"] / entry["close"]
        if self.adjust_prices:
            entry["open"] = adj * entry["open"]
            entry["close"] = entry["adjclose"]
            entry["high"] = adj * entry["high"]
            entry["low"] = adj * entry["low"]
        return entry[event]
