from abc import ABC, abstractmethod
from yahoo_fin.stock_info import get_data
from typing import Union, List, Optional, Tuple
import pandas as pd
from dateutil.relativedelta import relativedelta
from datetime import date, datetime
import datetime
import numpy as np
import diskcache as dc
import pytz
import requests
import re
from urllib.parse import urlencode
from bs4 import BeautifulSoup
from memoization import cached
from json import JSONDecodeError

from .exceptions import TimeLeakError, PriceUnavailableError


def _get_arg_key(self, *args):
    return str(args)


class CachedProvider(ABC):
    def __init__(self, debug=False):
        self.cache = dc.Cache(".simple-back")
        self.mem_cache = {}
        self.debug = debug

    def get_key(self, key: str) -> str:
        return str(key) + self.name

    def get_cache(self, key):
        nkey = self.get_key(key)
        if self.in_cache(key):
            return self.mem_cache[nkey]
        else:
            raise KeyError(f"{nkey} not in cache")

    def in_cache(self, key) -> bool:
        if not self.debug:
            key = self.get_key(key)
            if key in self.mem_cache:
                return True
            if key in self.cache:
                self.mem_cache[key] = self.cache[key]
                return True
        return False

    def set_cache(self, key, val, expire_days=None):
        key = self.get_key(key)
        self.mem_cache[key] = val
        if expire_days is None:
            self.cache.set(key, val)
        else:
            self.cache.set(key, val, expire=expire_days * 60 * 60 * 24)

    def rem_cache(self, key):
        key = self.get_key(key)
        del self.mem_cache[key]
        del self.cache[key]

    def clear_cache(self):
        for key in self.cache.iterkeys():
            if key.endswith(self.name):
                del self.cache[key]
        self.mem_cache = {}

    @property
    @abstractmethod
    def name(self):
        pass


class DataProvider(CachedProvider):
    def __init__(self, debug=False):
        self.current_datetime = pd.Timestamp(
            datetime.datetime.utcnow(), tzinfo=pytz.utc
        )
        super().__init__(debug=debug)

    def __getitem__(self, symbol_datetime=None) -> pd.DataFrame:
        try:
            self.current_datetime = self.bt.timestamp
        except AttributeError:
            pass
        if isinstance(symbol_datetime, tuple):
            symbol = symbol_datetime[0]
            date = symbol_datetime[1]
        elif isinstance(symbol_datetime, str):
            symbol = symbol_datetime
            date = self.current_datetime
        elif isinstance(symbol_datetime, pd.Timestamp) or symbol_datetime is None:
            symbol = None
            date = self.current_datetime
        if date > self.current_datetime:
            raise TimeLeakError(
                self.current_datetime,
                date,
                f"""
                {date} is more recent than {self.current_datetime},
                resulting in time leak
                """,
            )
        if not self.debug:
            return self._get_cached(self.name, date, symbol)
        else:
            return self.get(date, symbol)

    def __call__(self, datetime=None):
        try:
            if datetime is None:
                self.current_datetime = self.bt.timestamp
            else:
                self.current_datetime = datetime
        except AttributeError:
            pass
        if not self.debug:
            return self._get_cached(self.name, self.current_datetime, None)
        else:
            return self.get(self.current_datetime, None)

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def dates(self, symbol):
        pass

    @abstractmethod
    def get(self, datetime: pd.Timestamp, symbol: str = None):
        pass

    @cached(thread_safe=False, custom_key_maker=_get_arg_key)
    def _get_cached(self, *args) -> pd.DataFrame:
        return self.get(args[1], args[2])


class WikipediaProvider(DataProvider):
    def get_revisions(self, title):
        url = (
            "https://en.wikipedia.org/w/api.php?action=query&format=xml&prop=revisions&rvlimit=500&"
            + title
        )
        revisions = []
        next_params = ""

        if self.in_cache(title):
            results = self.get_cache(title)
        else:
            while True:
                response = requests.get(url + next_params).text
                revisions += re.findall("<rev [^>]*>", response)
                cont = re.search('<continue rvcontinue="([^"]+)"', response)
                if not cont:
                    break
                next_params = "&rvcontinue=" + cont.group(1)

            results = [
                (
                    pd.Timestamp(re.findall('timestamp="([^"]+)', r)[0]),
                    re.findall('id="([^"]+)', r)[0],
                )
                for r in revisions
            ]

            self.set_cache(title, results, 1)

        return results

    @property
    def name(self):
        return "Wikipedia Provider"

    def get(self, datetime: pd.Timestamp, symbol: str):
        new_symbol = self.transform_symbol(symbol)
        if new_symbol is None:
            new_symbol = symbol
        titles = urlencode({"titles": new_symbol})
        title = urlencode({"title": new_symbol})
        rev = self.get_revisions(titles)
        for r in rev:
            if r[0] <= datetime:
                if self.debug:
                    print(r[0])
                if self.in_cache(title + r[1]):
                    html = self.get_cache(title + r[1])
                else:
                    url = f"https://en.wikipedia.org/w/index.php?{title}&oldid={r[1]}"
                    if self.debug:
                        print(url)
                    html = requests.get(url).text
                    self.set_cache(title + r[1], html)
                return self.process_html(html, symbol)

    def dates(self, symbol):
        new_symbol = self.transform_symbol(symbol)
        if new_symbol is None:
            new_symbol = symbol
        titles = urlencode({"titles": new_symbol})
        rev = self.get_revisions(titles)
        revs = [r[0] for r in rev]
        revs.reverse()
        return revs

    def transform_symbol(self, symbol):
        return symbol

    @abstractmethod
    def process_html(self, html, symbol):
        pass


class SpProvider(WikipediaProvider):
    def transform_symbol(self, symbol):
        if symbol == "S&P_500":
            return "List_of_S&P_500_companies"
        return symbol

    def process_html(self, html, symbol):
        bs_object = BeautifulSoup(html, "html.parser")
        if symbol == "S&P_100":
            table = bs_object.find("table", {"class": "wikitable sortable"})
            td_i = 0
        if symbol == "S&P_500":
            table = bs_object.find("table", {"id": "constituents"})
            td_i = 0
            if table is None:
                table = bs_object.find("table", {"class": "wikitable sortable"})
            if table is None:
                table = bs_object.find("table", {"class": "wikitable"})
                td_i = 1
            if table is None:
                return None
        tickers = []
        try:
            for row in table.findAll("tr")[1:]:
                ticker = row.findAll("td")[td_i].text.strip()
                tickers.append(ticker)
            return pd.Series(tickers)
        except:
            return None

    @property
    def name(self):
        return "S&P"


class DailyDataProvider(CachedProvider):
    def __init__(self, debug=False):
        self.current_date = date.today()
        self.current_event = self.columns[np.argmax(self.columns_order)]
        self._leak_allowed = False
        super().__init__(debug=debug)

    def _remove_leaky_vals(self, df, cols, date):
        if isinstance(df, pd.DataFrame):
            if isinstance(date, str):
                date = pd.to_datetime(date)
            istoday = (not isinstance(date, slice)) and date == self.current_date
            if istoday or (df is not None and self.current_date in df.index):
                cur_order = self.columns.index(self.current_event)
                if isinstance(cols, str):
                    cols = [cols]
                for i, col in enumerate(cols):
                    if self.columns_order[i] > cur_order:
                        if istoday:
                            df[col] = None
                        else:
                            df.at[self.current_date, col] = None
            if isinstance(date, slice) and not df.empty:
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
            elif not isinstance(date, slice):
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
        if isinstance(self.current_event, list):
            for event in self.current_event:
                order = self._get_order(event)
                if order > max_order:
                    max_order = order
        if isinstance(self.current_event, str):
            max_order = self._get_order(self.current_event)
        return max_order

    def set_date_event(self, date, event):
        self.current_date = date
        self.current_event = event

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
        if not isinstance(symbol_date_event, str):
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
            if isinstance(date, datetime.date):
                if date > self.current_date and not self._leak_allowed:
                    raise TimeLeakError(
                        self.current_date,
                        date,
                        f"""
                        {date} is more recent than {self.current_date},
                        resulting in time leak
                        """,
                    )
            if isinstance(date, slice):
                if date == slice(None, None):
                    date = slice(None, self.current_date)
                if date.stop is None:
                    date = slice(date.start, self.current_date, date.step)
                stop_date = date.stop
                if isinstance(date.stop, relativedelta):
                    date = slice(date.start, self.current_date + date.stop, date.step)
                if isinstance(date.start, relativedelta):
                    if isinstance(stop_date, str):
                        stop_date = pd.to_datetime(stop_date)
                    date = slice(stop_date + date.start, date.stop, date.step)
                stop_date = date.stop
                if isinstance(date.stop, int) and date.stop <= 0:
                    date = slice(
                        date.start,
                        self.current_date - relativedelta(days=-1 * date.stop),
                        date.step,
                    )
                if isinstance(date.start, int) and date.start <= 0:
                    stop_date = date.stop
                    if isinstance(stop_date, slice):
                        stop_date = stop_date.stop
                    if isinstance(stop_date, str):
                        stop_date = pd.to_datetime(stop_date)
                    date = slice(
                        stop_date - relativedelta(days=-1 * date.start),
                        date.stop,
                        date.step,
                    )
        if not self.debug:
            data = self._get_cached(self.name, symbol, date, event)
        else:
            data = self.get(symbol, date, event)
        if self._leak_allowed:
            return data
        return self._remove_leaky_vals(data, event, date)

    @cached(thread_safe=False, custom_key_maker=_get_arg_key)
    def _get_cached(self, *args) -> pd.DataFrame:
        return self.get(args[1], args[2], args[3])

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


class DailyPriceProvider(DailyDataProvider):
    def __init__(self, highlow=True, debug=False):
        self.highlow = highlow
        super().__init__(debug=debug)

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
    def __init__(self, highlow=False, adjust_prices=True, debug=False):
        self.adjust_prices = adjust_prices
        self.highlow = highlow
        super().__init__(debug=debug)

    @property
    def name(self):
        return "Yahoo Finance Prices"

    def get(
        self, symbol: str, date: Union[slice, date], event: Union[str, List[str]]
    ) -> pd.DataFrame:
        if not self.in_cache(symbol):
            try:
                df = get_data(symbol)
            except AssertionError:
                raise PriceUnavailableError(
                    symbol, date, f"Price for {symbol} could not be found."
                )
            self.set_cache(symbol, df)
        else:
            df = self.get_cache(symbol)
        if df.isna().any().any():
            df.dropna(inplace=True, axis=0)
        entry = df.loc[date].copy()
        adj = entry["adjclose"] / entry["close"]
        if self.adjust_prices:
            entry["open"] = adj * entry["open"]
            entry["close"] = entry["adjclose"]
            entry["high"] = adj * entry["high"]
            entry["low"] = adj * entry["low"]
        return entry[event]
