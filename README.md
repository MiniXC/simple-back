# 📈📉&nbsp;&nbsp;simple-back
![build](https://github.com/MiniXC/simple-back/workflows/build/badge.svg)
![PyPI](https://img.shields.io/pypi/v/simple-back)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![codecov](https://codecov.io/gh/MiniXC/simple-back/branch/master/graph/badge.svg)](https://codecov.io/gh/MiniXC/simple-back)



## Installation
````
pip install simple_back
````
## Quickstart
> The following buys QQQ everytime it is above its 30-day average and shorts when it is below. It only does this on market open.

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
There are many backtesters out there, so inevitably, this is just another one of them.

This utility is not built for intra-day trading strategies, as most free data sources only give access to open and close prices.

Other things that make this backtester different are:

### Defaults over Configuration
Many backtesters need a great deal of configuration and setup before they can be used. 
Not so this one. 
At it's core you only need one loop, as this backtester can be used like any iterator.

I'm currently trying to figure out if I can make this iterator use multiprocessing, but until then you will have to use a [strategy object](https://render.githubusercontent.com/view/ipynb?commit=503b7c54ed188028d3ed4abd60c72b724b034d70&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f4d696e6958432f73696d706c652d6261636b2f353033623763353465643138383032386433656434616264363063373262373234623033346437302f6578616d706c65732f75736167655f6578616d706c65732e6970796e62&nwo=MiniXC%2Fsimple-back&path=examples%2Fusage_examples.ipynb&repository_id=265216728&repository_type=Repository#Strategy-Class) if you want faster backtests.

### Exensibilty over Trying-to-do-it-all
This is inteded to be a lean framework where, e.g. adding crypto data is as easy as extending the ``DailyPriceProvider`` class.

### Prototyping over Production
This should go without saying, beware before using this in any production systems. This is intended for rapid prototyping of strategies and nothing more.
