from abc import ABC, abstractmethod


class Fee(ABC):
    """
    Abstract ``Callable`` that calculates the total cost and number of shares
    given an asset price and the capital to be allocated to it.
    It returns the total cost and the number of shares aquired for that cost.
    """

    def __call__(self, price: float, capital: float):
        shares = self.num_shares(price, capital)
        cost = self.cost(price, shares)
        if cost > capital:
            raise ValueError("Cost cannot be higher than capital.")
        return shares, cost

    @abstractmethod
    def num_shares(self, price, capital):
        pass

    @abstractmethod
    def cost(self, price, num_shares):
        pass


class FlatPerTrade(Fee):
    def __init__(self, fee):
        self.fee = fee

    def num_shares(self, price, capital):
        return (capital - self.fee) // price

    def cost(self, price, num_shares):
        return price * num_shares + self.fee


class FlatPerShare(Fee):
    def __init__(self, fee):
        self.fee = fee

    def num_shares(self, price, capital):
        return capital // (self.fee + price)

    def cost(self, price, num_shares):
        return (price + self.fee) * num_shares


class NoFee(Fee):
    """
    Returns the number of shares possible to buy with given capital,
    and calculates to total cost of buying said shares.

    Example:
        How many shares of an asset costing 10 can be bought using 415,
        and what is the total cost::

            >>> NoFee(10, 415)
            41, 410
    """

    def num_shares(self, price, capital):
        return capital // price

    def cost(self, price, num_shares):
        return price * num_shares
