from simple_back.price_providers import DailyPriceProvider
import pandas as pd
import pandas_market_calendars as mcal
from dateutil.relativedelta import relativedelta

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
            self.strategy.bt = self
        self.prices = prices
        self.portfolio = pd.DataFrame(columns=['symbol', 'date', 'event', 'number', 'price'])
        self.value_ot = pd.DataFrame(columns=['date','event','value'])
        self.value_ot['value'] = self.value_ot['value'].astype(float)

    def run(self):
        for date in self.dates:
            self.current_date = date

            self.event = 'open'
            self.strategy.open()

            self.event = 'close'
            self.strategy.close()

    def __iter__(self):
        self.current_date = self.dates[0]
        self.i = 0
        self.event = 'open'
        return self

    def __next__(self):
        self.update()
        if self.event == 'open':
            self.event = 'close'
        elif self.event == 'close':
            try:
                self.i += 1
                self.current_date = self.dates[self.i]
                self.event = 'open'
            except:
                raise StopIteration
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