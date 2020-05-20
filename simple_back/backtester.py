from simple_back.price_providers import DailyPriceProvider
import pandas as pd
import pandas_market_calendars as mcal
from dateutil.relativedelta import relativedelta
from abc import ABC, abstractmethod
import multiprocessing
import numpy as np
import copy

class Strategy(ABC):

    @abstractmethod
    def open_close(self, day, bt):
        pass

    @abstractmethod
    def open(self, day, bt):
        pass

    @abstractmethod
    def close(self, day, bt):
        pass

class Backtester():
    
    def __init__(
        self, 
        start_capital, 
        prices,

        strategy=None,

        # list of dates to be used
        dates=None, 

        # if list of dates is not supplied, use pandas_market_calendars
        market_calendar=None,
        start_date=None,
        end_date=None,

        trade_cost_function=None,
    ):
        self.start_capital = start_capital
        self.capital = start_capital
        self.available_capital = start_capital
        if dates is not None:
            self.dates = dates
        else:
            cal = mcal.get_calendar(market_calendar)
            sched = cal.schedule(start_date=start_date, end_date=end_date)
            self.dates = mcal.date_range(sched, frequency='1D')
            self.dates = [d.date() for d in self.dates]
        if strategy is not None:
            self.strategy = strategy
        self.prices = prices
        self.portfolio = pd.DataFrame(columns=['symbol', 'date', 'event', 'number', 'price'])
        self.value_ot = pd.DataFrame(columns=['date','event','value'])
        self.value_ot['value'] = self.value_ot['value'].astype(float)

    def run_worker(self, num):
        return self.workers[num].run(0)

    def run(self, num_workers=-1):
        if num_workers == -1:
            num_workers = multiprocessing.cpu_count()
        if num_workers == 0:
            for date in self.dates:
                self.current_date = date
                
                self.event = 'open'
                self.update()
                self.strategy.open(self.current_date, self)
                self.strategy.open_close(self.current_date, self)

                self.event = 'close'
                self.update()
                self.strategy.close(self.current_date, self)
                self.strategy.open_close(self.current_date, self)

            return self
        else:
            self.workers = []
            date_splits = np.array_split(self.dates, num_workers)
            for i in range(num_workers):
                worker_bt = copy.deepcopy(self)
                # share the prices object
                worker_bt.prices = self.prices
                # set dates
                worker_bt.dates = date_splits[i]
                self.workers.append(worker_bt)
                # start multiprocessing
            pool = multiprocessing.Pool(processes=num_workers)
            backtesters = pool.map(self.run_worker, range(num_workers))
            new_value_ot = None
            for i, bt in enumerate(backtesters):
                if i == 0:
                    new_value_ot = bt.value_ot.copy()
                else:
                    temp_value_ot = bt.value_ot.copy()
                    temp_value_ot['value'] = bt.value_ot['value'] * (new_value_ot['value'].iloc[-1]/self.start_capital)
                    new_value_ot = new_value_ot.append(temp_value_ot)
            self.value_ot = new_value_ot

    def __iter__(self):
        self.i = 0
        self.event = 'close'
        return self

    def __next__(self):
        if self.event == 'open':
            self.event = 'close'
        elif self.event == 'close':
            try:
                self.current_date = self.dates[self.i]
                self.i += 1
                self.event = 'open'
            except:
                raise StopIteration
        self.update()
        return self.current_date, self.event, self

    def __len__(self):
        return len(self.dates)*2-1

    def price(self, symbol, lookback=None):
        if lookback is None:
            return self.prices[symbol,self.current_date,self.event]
        if lookback > 0:
            start = self.current_date-relativedelta(days=lookback)
            end = self.current_date
            return self.prices[symbol,start:end,self.event]

    def update(self):
        self.capital = self.available_capital
        for _, pos in self.portfolio.iterrows():
            if pos['number'] > 0:
                self.capital += pos['number']*self.price(pos['symbol'])
            else:
                cur_val = abs(pos['number'])*self.price(pos['symbol'])
                old_val = abs(pos['number'])*pos['price']
                self.capital += (old_val - cur_val)
        self.value_ot.loc[len(self.value_ot)] = [self.current_date, self.event, self.capital]

    def order(self, symbol, capital, short=False, as_percent=False):
        if not as_percent:
            if capital > self.available_capital:
                raise Exception('not enough capital available')
        else:
            if capital > 1:
                raise Exception('not enough capital available')
        current_price = self.prices[symbol, self.current_date, self.event]
        if as_percent:
            capital = capital*self.available_capital
        num_shares = capital // current_price
        if short:
            num_shares *= -1
        else:
            self.available_capital -= current_price * num_shares
        self.portfolio.loc[len(self.portfolio)] = [symbol, self.current_date, self.event, num_shares, current_price]

    @property
    def values(self):
        vals = self.value_ot.drop(columns=['event']).groupby('date').mean().rename(columns={'value':'Backtest'})
        vals.index = pd.to_datetime(vals.index)
        return vals

    @property
    def profit_loss(self):
        return self.values.pct_change()[1:]

    def compare(self, symbol, vals=None):
        if type(symbol) == list:
            first = symbol[0]
            symbols = symbol[1:]
            symbol = first
        c = self.prices[symbol,self.dates,'open']
        c = c/c[0]*self.start_capital
        if vals is None:
            vals = self.values.copy()
        vals[symbol] = pd.DataFrame(c, columns=[symbol]) 
        if len(symbols) is not 0:
            vals = self.compare(symbols, vals)
        return vals

    def compare_pl(self, symbol):
        comps = self.compare(symbol)
        return [comps[col].pct_change()[1:] for col in comps.columns]

    def liquidate(self, pos_index):
        for _, pos in self.portfolio[pos_index].iterrows():
            if pos['number'] > 0:
                self.available_capital += pos['number']*self.price(pos['symbol'])
            else:
                cur_val = abs(pos['number'])*self.price(pos['symbol'])
                old_val = abs(pos['number'])*pos['price']
                self.available_capital += old_val - cur_val
        self.portfolio = self.portfolio[~pos_index]