__version__ = '0.2.0'

from .backtester import Backtester, Strategy
from .price_providers import DailyPriceProvider, YahooFinanceProvider


__all__ = [
    "Backtester",
    "DailyPriceProvider",
    "YahooFinanceProvider",
    "Strategy"
]