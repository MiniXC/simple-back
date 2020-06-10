"""
A backtester with minimal setup and sensible defaults.

**simple_back** is a backtester providing easy ways to test trading
strategies, often ones that use external data, with minimal amount of
code, while avoiding time leaks.
"""


__version__ = "0.4.4"

from . import backtester
from . import strategy
from . import fees
from . import data_providers
from . import metrics

__all__ = ["backtester", "strategy", "fees", "data_providers", "metrics"]
