# 📈📉&nbsp;&nbsp;simple-back
![build](https://github.com/MiniXC/simple-back/workflows/build/badge.svg)
![PyPI](https://img.shields.io/pypi/v/simple-back)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![codecov](https://codecov.io/gh/MiniXC/simple-back/branch/master/graph/badge.svg)](https://codecov.io/gh/MiniXC/simple-back)
[![Documentation Status](https://readthedocs.org/projects/simple-back/badge/?version=latest)](https://simple-back.readthedocs.io/en/latest/?badge=latest)


## Installation
````
pip install simple_back
````
## Quickstart
> The following is a simple crossover strategy. For a full tutorial on how to build a strategy using **simple-back**, visit [the quickstart tutorial](https://simple-back.readthedocs.io/en/latest/intro/quickstart.html)

````python
from simple_back.backtester import BacktesterBuilder

bt = (
   BacktesterBuilder()
   .name('JNUG 20-Day Crossover')
   .balance(10_000)
   .calendar('NYSE')
   .compare(['JNUG']) # strategies to compare with
   .live_plot() # we assume we are running this in a Jupyter Notebook
   .build()
)

for day, event, b in bt['2019-1-1':'2020-1-1']:
    if event == 'open':
        jnug_ma = b.prices['JNUG',-20:]['close'].mean()

        if b.price('JNUG') > jnug_ma:
            if not b.portfolio['JNUG'].long: # check if we already are long JNUG
                b.portfolio['JNUG'].short.liquidate() # liquidate any/all short JNUG positions
                b.order_pct('JNUG', 1) # long JNUG

        if b.price('JNUG') < jnug_ma:
            if not b.portfolio['JNUG'].short: # check if we already are short JNUG
                b.portfolio['JNUG'].long.liquidate() # liquidate any/all long JNUG positions
                b.order_pct('JNUG', -1) # short JNUG
````


## Why another python backtester?
There are many backtesters out there, but this is the first one built for rapid prototyping in Jupyter Notebooks.

### Built for Jupyter Notebooks
Get live feedback on your backtests (live plotting, progress and metrics) *in your notebook* to immediatly notice if something is off about your strategy.

### Sensible Defaults
Many backtesters need a great deal of configuration and setup before they can be used. 
Not so this one.  At it's core you only need one loop, as this backtester can be used like any iterator.
A default provider for prices is included, and caches all its data on your disk to minimize the number of requests needed.

### Extensibile
This is intended to be a lean framework where, e.g. adding crypto data is as easy as extending the ``DailyPriceProvider`` class.
