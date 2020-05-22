from simple_back.price_providers import DailyPriceProvider, YahooFinanceProvider
import pandas as pd
import pandas_market_calendars as mcal
from dateutil.relativedelta import relativedelta
from datetime import date
from abc import ABC, abstractmethod
import multiprocessing
import numpy as np
import copy


class Fee(ABC):

    @abstractmethod
    def __call__(self, price, num_shares, order_type):
        pass

class FlatPerTrade(Fee):

    def __init__(self, fee):
        self.fee = fee

    def __call__(self, price, capital):
        num_shares = (capital - self.fee) // price
        return price * num_shares + self.fee, num_shares

class FlatPerShare(Fee):

    def __init__(self, fee):
        self.fee = fee

    def __call__(self, price, capital):
        num_shares = capital // (self.fee + price)
        return (price + self.fee) * num_shares, num_shares

class NoFee(Fee):

    def __call__(self, price, capital):
        num_shares = capital // price
        return num_shares * price, num_shares

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
        prices=None,

        strategy=None,

        # list of dates to be used
        dates=None, 

        # if list of dates is not supplied, use pandas_market_calendars
        market_calendar=None,
        cal=None,
        start=None,
        start_date=None,
        end_date=None,
        end=None,

        trade_cost_function=NoFee(),
    ):
        self.start_capital = start_capital
        self.capital = start_capital
        self.available_capital = start_capital

        if prices is None:
            prices = YahooFinanceProvider()
        if dates is not None:
            self.dates = dates
        else:
            if start is not None:
                start_date = start
                end_date = end
            if cal is not None:
                market_calendar = cal
            cal = mcal.get_calendar(market_calendar)
            if start_date is not None:
                if end_date is None:
                    end_date = date.today()-relativedelta(days=1)
                if type(start_date) == relativedelta:
                    start_date = date.today()-start_date
                if type(end_date) == relativedelta:
                    end_date = date.today()-end_date
                sched = cal.schedule(start_date=start_date, end_date=end_date)
            self.dates = mcal.date_range(sched, frequency='1D')
            self.dates = [d.date() for d in self.dates]
        if strategy is not None:
            self.strategy = strategy
        self.prices = prices
        self.portfolio = pd.DataFrame(columns=['symbol', 'date', 'event', 'num_shares', 'price'])
        self.value_ot = pd.DataFrame(columns=['date','event','value'])
        self.value_ot['value'] = self.value_ot['value'].astype(float)
        self.trade_cost = trade_cost_function

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
            if pos['num_shares'] > 0:
                self.capital += pos['num_shares']*self.price(pos['symbol'])
            else:
                cur_val = abs(pos['num_shares'])*self.price(pos['symbol'])
                old_val = abs(pos['num_shares'])*pos['price']
                self.capital += (old_val - cur_val)
        self.value_ot.at[len(self.value_ot)] = [self.current_date, self.event, self.capital]

    def order(self, symbol, capital, short=False, as_percent=False):
        if not as_percent:
            if capital > self.available_capital:
                raise Exception('not enough capital available')
        else:
            if capital * self.capital > self.available_capital:
                raise Exception(
                    f"""
                    not enough capital available:
                    ordered {capital} * {self.capital} with only {self.available_capital} available
                    """
                )
        current_price = self.prices[symbol, self.current_date, self.event]
        if as_percent:
            capital = capital * self.available_capital
        total, num_shares = self.trade_cost(current_price, capital)
        if short:
            num_shares *= -1
        else:
            self.available_capital -= total
        self.portfolio.at[len(self.portfolio)] = [symbol, self.current_date, self.event, num_shares, current_price]

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

    def liquidate(self, symbol, num_shares=None, short=False):
        if type(symbol) == list:
            if num_shares == None:
                for sym in symbol:
                    self.liquidate(sym,None)
            else:
                for sym, num in zip(symbol,num_shares):
                    self.liquidate(sym,num)
        else:
            drop_i = []
            for i, pos in self.portfolio[self.portfolio['symbol']==symbol].iterrows():
                if pos['num_shares'] > 0 and not short:
                    if num_shares is None or pos['num_shares'] <= num_shares:
                        self.available_capital += pos['num_shares'] * self.price(pos['symbol'])
                        drop_i.append(i)
                        if num_shares is not None:
                            num_shares -= pos['num_shares']
                        if num_shares == 0:
                            break
                    if num_shares is not None and pos['num_shares'] > num_shares:
                        self.available_capital += num_shares * self.price(pos['symbol'])
                        self.portfolio.at[i,'num_shares'] -= num_shares
                        break
                if pos['num_shares'] < 0 and short:
                    if num_shares is None or abs(pos['num_shares']) <= num_shares:
                        cur_val = abs(pos['num_shares'])*self.price(pos['symbol'])
                        old_val = abs(pos['num_shares'])*pos['price']
                        self.available_capital += old_val - cur_val
                        drop_i.append(i)
                        if num_shares is not None:
                            num_shares -= pos['num_shares']
                        if num_shares == 0:
                            break
                    if num_shares is not None and abs(pos['num_shares']) > num_shares:
                        cur_val = num_shares * self.price(pos['symbol'])
                        old_val = num_shares * pos['price']
                        self.available_capital += old_val - cur_val
                        self.portfolio.at[i,'num_shares'] += num_shares
                        break
            self.portfolio = self.portfolio.drop(drop_i)

    def liquidateIndex(self, pos_index=None):
        if pos_index == None:
            iter_port = self.portfolio
        else:
            iter_port = self.portfolio[pos_index]
        for _, pos in iter_port.iterrows():
            if pos['num_shares'] > 0:
                self.available_capital += pos['num_shares']*self.price(pos['symbol'])
            else:
                cur_val = abs(pos['num_shares'])*self.price(pos['symbol'])
                old_val = abs(pos['num_shares'])*pos['price']
                self.available_capital += old_val - cur_val
        if pos_index == None:
            self.portfolio = self.portfolio.iloc[0:0]
        else:
            self.portfolio = self.portfolio[~pos_index]
