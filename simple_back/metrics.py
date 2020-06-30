from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
import pandas as pd

from .exceptions import MissingMetricsError


class Metric(ABC):
    @property
    def requires(self) -> Optional[List[type]]:
        return None

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        if (self._single and self.value is None) or (
            self._series and self.values is None
        ):
            return "None"
        if self._single:
            return f"{self.value:.2f}"
        if self._series:
            return f"{self[-1]:.2f}"
        else:
            return f"{self.name}"

    def __init__(self):
        self._single = False
        self._series = False
        self.value = None
        self.values = None
        self.current_event = "open"
        self.bt = None

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    def set_values(self, bt):
        if self._single:
            self.value = self.get_value(bt)
        if self._series:
            self.all_values[self.i] = self.get_value(bt)

    def __call__(self, write=False):
        if not write:
            return self.get_value(self.bt)

        self.current_event = self.bt.event
        self.i = self.bt.i
        if self._series and (self.bt.i == 0 and self.bt.event == "open"):
            self.all_values = np.repeat(np.nan, len(self.bt))
        if self.requires is None:
            self.set_values(self.bt)
        else:
            all_requires_present = True
            missing = ""
            for req in self.requires:
                if req not in self.bt.metric.keys():
                    all_requires_present = False
                    missing = req
                    break
            if all_requires_present:
                self.set_values(self.bt)
            else:
                raise MissingMetricsError(
                    self.requires,
                    f"""
                    The following metric required by {type(self)} is missing:
                    {missing}
                    """,
                )

    @abstractmethod
    def get_value(self, bt):
        pass


class SingleMetric(Metric):
    def __init__(self, name: Optional[str] = None):
        self._single = True
        self._series = False
        self.value = None

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def get_value(self, bt):
        pass


class SeriesMetric(Metric):
    def __init__(self, name: Optional[str] = None):
        self._single = False
        self._series = True
        self.value_open = []
        self.value_close = []
        self.i = 0
        self.last_len = 0
        self.all_values = []

    @property
    def values(self):
        return self.all_values

    @property
    def df(self):
        df = pd.DataFrame()
        df["date"] = self.bt.dates[: self.i // 2 + 1]
        df["open"] = self.all_values[0::2][: self.i // 2 + 1]
        df["close"] = self.all_values[1::2][: self.i // 2 + 1]
        if self.current_event == "open":
            df.at[-1, "close"] = None
        return df.set_index("date").dropna(how="all")

    @property
    @abstractmethod
    def name(self):
        pass

    def __len__(self):
        return self.i + 1

    def __getitem__(self, i):
        return self.all_values[: self.i + 1][i]

    @abstractmethod
    def get_value(self, bt):
        pass


class MaxDrawdown(SingleMetric):
    @property
    def name(self):
        return "Max Drawdown (%)"

    def get_value(self, bt):
        highest_peaks = bt.metrics["Total Value"].cummax()
        actual_value = bt.metrics["Total Value"]
        md = np.min(((actual_value - highest_peaks) / highest_peaks).values) * 100
        return md


class AnnualReturn(SingleMetric):
    @property
    def name(self):
        return "Annual Return"

    @property
    def requires(self):
        return ["Portfolio Value"]

    def get_value(self, bt):
        vals = bt.metric["Total Value"]
        year = 1 / ((bt.dates[-1] - bt.dates[0]).days / 365.25)
        return (vals[-1] / vals[0]) ** year


class PortfolioValue(SeriesMetric):
    @property
    def name(self):
        return "Portfolio Value"

    def get_value(self, bt):
        return bt.portfolio.total_value


class DailyProfitLoss(SeriesMetric):
    @property
    def name(self):
        return "Daily Profit/Loss"

    @property
    def requires(self):
        return ["Total Value"]

    def get_value(self, bt):
        try:
            return bt.metric["Total Value"][-1] - bt.metric["Total Value"][-3]
        except IndexError:
            return 0


class TotalValue(SeriesMetric):
    @property
    def name(self):
        return "Total Value"

    @property
    def requires(self):
        return ["Portfolio Value"]

    def get_value(self, bt):
        return bt.metric["Portfolio Value"]() + bt._available_capital


class TotalReturn(SeriesMetric):
    @property
    def name(self):
        return "Total Return (%)"

    @property
    def requires(self):
        return ["Total Value"]

    def get_value(self, bt):
        return ((bt.metric["Total Value"][-1] / bt.balance.start) - 1) * 100
