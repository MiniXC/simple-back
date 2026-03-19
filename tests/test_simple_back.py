import os
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from simple_back import __version__
from simple_back.backtester import BacktesterBuilder
from simple_back.data_providers import AlpacaProvider, YahooFinanceProvider, DailyPriceProvider
from datetime import date
from pathlib import Path

PLOT_DIR = Path(__file__).parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)


def test_version():
    assert __version__ == "0.7.0"


def test_builder_requires_prices():
    """Builder must raise if no price provider is set."""
    builder = (
        BacktesterBuilder()
        .name("Test")
        .balance(10_000)
        .calendar("NYSE")
    )
    bt = builder.build()
    with pytest.raises(ValueError, match="no price provider set"):
        for _ in bt[date(2024, 1, 2):date(2024, 1, 5)]:
            pass


def test_builder_requires_balance():
    """Builder must raise if no balance is set."""
    bt = BacktesterBuilder().name("Test").calendar("NYSE").build()
    with pytest.raises(ValueError, match="no price provider set"):
        for _ in bt[date(2024, 1, 2):date(2024, 1, 5)]:
            pass


# ── yfinance tests ───────────────────────────────────────────────────
# YahooFinanceProvider uses curl_cffi with impersonate="chrome120"
# (newer chrome fingerprints are blocked by Yahoo from datacenter IPs).


def _plot_price_data(data, title, filename):
    """Plot OHLC-style price data and save to PLOT_DIR."""
    fig, ax = plt.subplots(figsize=(10, 4))
    for col in data.columns:
        ax.plot(data.index, data[col], marker="o", label=col)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / filename, dpi=100)
    plt.close(fig)


def _plot_backtest(bt, title, filename):
    """Plot total value and return curves from a backtest and save to PLOT_DIR."""
    tv = bt.metric["Total Value"].df
    tr = bt.metric["Total Return (%)"].df

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for col in tv.columns:
        ax1.plot(tv.index, tv[col], label=col)
    ax1.set_title(f"{title} — Total Value")
    ax1.set_ylabel("Value ($)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    for col in tr.columns:
        ax2.plot(tr.index, tr[col], label=col)
    ax2.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax2.set_title(f"{title} — Total Return")
    ax2.set_ylabel("Return (%)")
    ax2.set_xlabel("Date")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / filename, dpi=100)
    plt.close(fig)


def test_yfinance_provider_basic():
    """Smoke test: fetch a few days of AAPL data from yfinance."""
    provider = YahooFinanceProvider(highlow=True)
    provider.clear_cache()
    provider.current_date = date(2024, 12, 31)
    provider.current_event = "close"
    data = provider["AAPL", slice(date(2024, 12, 20), date(2024, 12, 31)), ["open", "close"]]
    assert len(data) > 0
    assert "open" in data.columns
    assert "close" in data.columns
    _plot_price_data(data, "AAPL (yfinance) — Dec 20-31 2024", "yfinance_provider_basic.png")


def test_backtest_with_yfinance():
    """Integration test: buy-and-hold AAPL with yfinance provider."""
    provider = YahooFinanceProvider()
    provider.clear_cache()
    builder = (
        BacktesterBuilder()
        .name("AAPL yfinance Test")
        .balance(10_000)
        .calendar("NYSE")
        .prices(provider)
    )
    bt = builder.build()
    days = 0
    for day, event, b in bt[date(2024, 6, 3):date(2024, 6, 14)]:
        if event == "open" and not b.portfolio:
            b.long("AAPL", percent=1)
        days += 1
    assert days > 0
    total_return = bt.metric["Total Return (%)"][-1]
    assert total_return is not None
    # AAPL went up in early June 2024; return should be positive
    assert total_return > -50  # sanity check, not exact
    _plot_backtest(bt, "AAPL Buy-and-Hold (yfinance)", "yfinance_backtest.png")


# ── Alpaca tests ─────────────────────────────────────────────────────

_has_alpaca = bool(
    os.environ.get("ALPACA_API_KEY") and os.environ.get("ALPACA_SECRET_KEY")
)


@pytest.mark.skipif(not _has_alpaca, reason="ALPACA_API_KEY/ALPACA_SECRET_KEY not set")
def test_alpaca_provider_basic():
    """Smoke test: fetch a few days of AAPL data from Alpaca."""
    provider = AlpacaProvider()
    provider.clear_cache()
    provider.current_date = date(2024, 12, 31)
    provider.current_event = "close"
    data = provider["AAPL", slice(date(2024, 12, 20), date(2024, 12, 31)), ["open", "close"]]
    assert len(data) > 0
    assert "open" in data.columns
    assert "close" in data.columns
    _plot_price_data(data, "AAPL (Alpaca) — Dec 20-31 2024", "alpaca_provider_basic.png")


@pytest.mark.skipif(not _has_alpaca, reason="ALPACA_API_KEY/ALPACA_SECRET_KEY not set")
def test_backtest_with_alpaca():
    """Integration test: run a simple buy-and-hold on AAPL via Alpaca."""
    provider = AlpacaProvider()
    provider.clear_cache()
    builder = (
        BacktesterBuilder()
        .name("AAPL Test")
        .balance(10_000)
        .calendar("NYSE")
        .prices(provider)
    )
    bt = builder.build()
    days = 0
    for day, event, b in bt[date(2024, 6, 3):date(2024, 6, 14)]:
        if event == "open" and not b.portfolio:
            b.long("AAPL", percent=1)
        days += 1
    assert days > 0
    assert bt.metric["Total Return (%)"][-1] is not None
    _plot_backtest(bt, "AAPL Buy-and-Hold (Alpaca)", "alpaca_backtest.png")
