from simple_back.backtester import BacktesterBuilder

builder = (
    BacktesterBuilder()
    .name("My First Strategy")
    .balance(10_000)  # define your starting balance
    .live_progress()  # show a progress bar (requires tqdm)
    # .live_plot()      # shows a live plot when working in a notebook
    .compare(["MSFT"])  # compare to buying and holding MSFT
    .calendar("NYSE")  # trade on days the NYSE is open
)

bt = builder.build()  # build the backtester

# you can now use bt like you would any iterator
# specify a date range, and the code inside will be run
# on open and close of every trading day in that range
for day, event, b in bt["2019-1-1":"2020-1-1"]:

    # you can get the current prices of securities
    b.price("MSFT")  # the current price of MSFT
    b.prices["MSFT", -30:]["open"].mean()
    # ^ gets the mean open price over the last 30 days

    # you can now order stocks using
    b.long("MSFT", percent=0.5)  # allocate .5 of your funds to MSFT
    b.long(
        "MSFT", percent_available=0.1
    )  # allocate .1 of your cash still available to MSFT
    b.long("MSFT", absolute=1_000)  # buy 1,000 worth of MSFT
    b.long("MSFT", nshares=1)  # buy 1 MSFT share

    # you can access your protfolio using
    b.portfolio
    b.pf
    # or in dataframe form using
    b.portfolio.df
    b.pf.df

    # you can use the portfolio object to get out of positions
    b.pf.liquidate()  # liquidate all positions
    b.pf.long.liquidate()  # liquidate all long positions
    b.pf["MSFT"].liquidate()  # liquidate all MSFT positions

    # you can also filter the portfolio using
    # the portfolio dataframe, or using .filter
    b.pf[b.pf.df.profit_loss_pct < -0.05].liquidate()
    b.pf.filter(lambda x: x.profit_loss_pct < -0.05).liquidate()
    # ^ both liquidate all positions that have lost more than 5%

    # you can use the metric dict to get current metrics
    b.metric["Portfolio Value"][-1]  # gets the last computed value
    b.metric["Portfolio Value"]()  # computes the current value


# AFTER THE BACKTEST

# get all metrics in dataframe form
bt.metrics

# get a summary of the backtest as a dataframe
bt.summary
# ^ this is useful for comparison when running several strategies
