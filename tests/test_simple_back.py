from simple_back import __version__
from simple_back.backtester import BacktesterBuilder


def test_version():
    assert __version__ == "0.6.0"


def test_compare_quantopian():
    builder = (
        BacktesterBuilder()
        .name("TSLA Strategy")
        .balance(10_000)
        .calendar("NYSE")
        .live_metrics()
        .slippage(0.0005)
    )
    bt = builder.build()
    for _, _, b in bt["2017-1-1":"2020-1-1"]:
        if not b.pf or True:
            b.pf.liquidate()
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
