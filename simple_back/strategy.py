from abc import ABC, abstractmethod


class Strategy(ABC):
    def __call__(self, day, event, bt):
        self.run(day, event, bt)

    @property
    def name(self):
        return None

    @abstractmethod
    def run(self, day, event, bt):
        pass


class BuyAndHold(Strategy):
    def __init__(self, ticker):
        self.ticker = ticker
        self.is_bought = False

    def run(self, day, event, bt):
        if not self.is_bought:
            bt.order_pct(self.ticker, 1)
            self.is_bought = True

    @property
    def name(self):
        return f"{self.ticker} (Buy & Hold)"


class SellAndHold(Strategy):
    def __init__(self, ticker):
        self.ticker = ticker
        self.is_bought = False

    def run(self, day, event, bt):
        if not self.is_bought:
            bt.order_pct(self.ticker, -1)
            self.is_bought = True

    @property
    def name(self):
        return f"{self.ticker} (Sell & Hold)"
