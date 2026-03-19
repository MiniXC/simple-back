from abc import ABC, abstractmethod
try:
    import yfinance as yf
    _has_yfinance = True
except ImportError:
    _has_yfinance = False
from typing import Union, List, Optional, Tuple
import pandas as pd
from dateutil.relativedelta import relativedelta
from datetime import date, datetime
import datetime
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
        if isinstance(self.current_date, str):
            self.current_date = datetime.date.fromisoformat(self.current_date)
        # Normalise Timestamp/str → datetime.date so comparisons work (pandas 2.x+)
        if isinstance(self.current_date, pd.Timestamp):
            self.current_date = self.current_date.date()
        if isinstance(date, pd.Timestamp):
            date = date.date()
        elif isinstance(date, str):
            date = datetime.date.fromisoformat(date)
        if isinstance(df, pd.DataFrame):
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
    """Daily price provider using the *yfinance* package.

    Args:
        highlow: Include high/low columns (default False).
        adjust_prices: Use adjusted close to scale OHLC (default True).
        impersonate: curl_cffi browser fingerprint to use (default
            ``"chrome120"``).  The latest ``"chrome"`` target is blocked
            by Yahoo from many datacenter IPs; older targets work.
        debug: Disable caching for debugging.
    """

    def __init__(self, highlow=False, adjust_prices=True, impersonate="chrome120", debug=False):
        if not _has_yfinance:
            raise ImportError(
                "yfinance is required for YahooFinanceProvider. "
                "Install with: pip install 'simple-back[yahoo]'"
            )
        self.adjust_prices = adjust_prices
        self.highlow = highlow
        self._impersonate = impersonate
        super().__init__(debug=debug)

    def _make_session(self):
        """Build a curl_cffi session with a working impersonate target."""
        from curl_cffi.requests import Session as CffiSession
        return CffiSession(impersonate=self._impersonate)

    @property
    def name(self):
        return "Yahoo Finance Prices"

    def get(
        self, symbol: str, date: Union[slice, date], event: Union[str, List[str]]
    ) -> pd.DataFrame:
        if not self.in_cache(symbol):
            try:
                session = self._make_session()
                tk = yf.Ticker(symbol, session=session)
                df = tk.history(period="max", auto_adjust=False)
                if df.empty:
                    raise ValueError("empty dataframe")
                # Normalise column names to lowercase
                df.columns = [c.lower() for c in df.columns]
                # Ensure tz-naive date index for consistent slicing
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
            except Exception as exc:
                raise PriceUnavailableError(
                    symbol, date, f"Price for {symbol} could not be found: {exc}"
                )
            self.set_cache(symbol, df)
        else:
            df = self.get_cache(symbol)
        df.dropna(subset=["open", "close"], inplace=True)
        # Convert date to Timestamp for .loc[] (pandas 2.x compat)
        if isinstance(date, slice):
            date = slice(
                pd.Timestamp(date.start) if date.start is not None else None,
                pd.Timestamp(date.stop) if date.stop is not None else None,
                date.step,
            )
        else:
            date = pd.Timestamp(date)
        entry = df.loc[date].copy()
        _fields = entry.columns if isinstance(entry, pd.DataFrame) else entry.index
        if self.adjust_prices and "adj close" in _fields:
            adj = entry["adj close"] / entry["close"]
            entry["open"] = adj * entry["open"]
            entry["close"] = entry["adj close"]
            entry["high"] = adj * entry["high"]
            entry["low"] = adj * entry["low"]
        return entry[event]


class AlpacaProvider(DailyPriceProvider):
    """Daily price provider using Alpaca Markets historical bars API.

    Requires ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables,
    or pass them directly.

    Args:
        api_key: Alpaca API key (falls back to ALPACA_API_KEY env var).
        secret_key: Alpaca secret key (falls back to ALPACA_SECRET_KEY env var).
        paper: Use paper trading endpoint (default True).
        highlow: Include high/low columns (default True).
        debug: Disable caching for debugging.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        paper: bool = True,
        highlow: bool = True,
        debug: bool = False,
    ):
        import os as _os

        self._api_key = api_key or _os.environ.get("ALPACA_API_KEY", "")
        self._secret_key = secret_key or _os.environ.get("ALPACA_SECRET_KEY", "")
        if not self._api_key or not self._secret_key:
            raise ValueError(
                "Alpaca credentials required. Set ALPACA_API_KEY and "
                "ALPACA_SECRET_KEY environment variables or pass them directly."
            )
        self._paper = paper
        self.highlow = highlow
        super().__init__(highlow=highlow, debug=debug)

    @property
    def name(self):
        return "Alpaca Prices"

    def _fetch_bars(self, symbol: str) -> pd.DataFrame:
        """Fetch all available daily bars for *symbol* from Alpaca."""
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        client = StockHistoricalDataClient(self._api_key, self._secret_key)
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            # Alpaca keeps ~7 years of daily bars; omit start for max history
            start=datetime.datetime(2015, 1, 1),
        )
        bars = client.get_stock_bars(request)
        df = bars.df
        # bars.df has a MultiIndex (symbol, timestamp) – drop symbol level
        if isinstance(df.index, pd.MultiIndex):
            df = df.droplevel("symbol")
        # Normalise index to date-only (tz-naive, midnight) so Timestamp lookups work
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        # Rename to match DailyPriceProvider conventions
        df = df.rename(
            columns={
                "open": "open",
                "close": "close",
                "high": "high",
                "low": "low",
                "volume": "volume",
                "vwap": "vwap",
            }
        )
        return df

    def get(
        self, symbol: str, date: Union[slice, date], event: Union[str, List[str]]
    ) -> pd.DataFrame:
        if not self.in_cache(symbol):
            try:
                df = self._fetch_bars(symbol)
            except Exception as exc:
                raise PriceUnavailableError(
                    symbol, date, f"Alpaca: price for {symbol} unavailable – {exc}"
                )
            self.set_cache(symbol, df, expire_days=1)
        else:
            df = self.get_cache(symbol)

        df.dropna(subset=["open", "close"], inplace=True)
        # Convert date to Timestamp for .loc[] (pandas 2.x compat)
        if isinstance(date, slice):
            date = slice(
                pd.Timestamp(date.start) if date.start is not None else None,
                pd.Timestamp(date.stop) if date.stop is not None else None,
                date.step,
            )
        else:
            date = pd.Timestamp(date)
        entry = df.loc[date].copy()
        return entry[event]
