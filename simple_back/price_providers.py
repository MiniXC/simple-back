from abc import ABC, abstractmethod
from yahoo_fin.stock_info import get_data

class DailyPriceProvider(ABC):

    def __getitem__(self, symbol_date_event):
        """
        Expects a tuple of (ticker_symbol, date, 'open' or 'close') and returns the price
        for said symbol at that point in time.
        ````
        my_price_provider['AAPL', date(2015,1,1), 'open']
        ````
        """
        if type(symbol_date_event) is not str:
            return self.get_price(*symbol_date_event)
        else:
            return self.get_price(symbol_date_event)
        
    @abstractmethod
    def get_price(self, symbol, date=None, event=None):
        pass


class YahooFinanceProvider(DailyPriceProvider):

    def __init__(self, adjust_prices=True):
        self.adjust_prices = adjust_prices
        self.symbols = {}
        super().__init__()

    def get_price(self, symbol, date=None, event=None):
        if symbol not in self.symbols:
            self.symbols[symbol] = get_data(symbol)
        df = self.symbols[symbol]
        if date is not None:
            entry = df.loc[date]
        else:
            entry = df
        if event == 'open':
            if self.adjust_prices:
                return entry['adjclose']/entry['close']*entry['open']
            else:
                return entry['open']
        if event == 'close':
            if self.adjust_prices:
                return entry['adjclose']
            else:
                return entry['close']
        if event is None:
            return entry
        raise Exception('event was neither "open" nor "close", but was still given')
