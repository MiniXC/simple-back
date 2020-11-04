from simple_back import __version__
from simple_back.backtester import BacktesterBuilder
from datetime import date


def test_version():
    assert __version__ == "0.6.3"


def test_compare_quantopian():
    builder = (
        BacktesterBuilder()
        .name("TSLA Strategy")
        .balance(10_000)
        .calendar("NYSE")
        # .live_metrics()
        .slippage(0.0005)
    )
    bt = builder.build()
    bt.prices.clear_cache()
    for _, _, b in bt[
        date.fromisoformat("2017-01-01") : date.fromisoformat("2020-01-01")
    ]:
        if not b.pf or True:
            b.pf.liquidate()
            print(b._schedule.index)
            b.long("TSLA", percent=1)
    quant_no_slippage = 105.97  # https://www.quantopian.com/posts/test-without-slippage-to-compare-with-simple-back
    quant_slippage = (
        -52.8
    )  # https://www.quantopian.com/posts/test-with-slippage-to-compare-with-simple-back
    # within 10%/15% of both bounds
    assert (
        abs(bt.summary.iloc[0]["Total Return (%) (Last Value)"] - quant_no_slippage)
        <= 15
    )
    assert (
        abs(bt.summary.iloc[1]["Total Return (%) (Last Value)"] - quant_slippage) <= 10
    )
