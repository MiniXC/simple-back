from abc import ABC, abstractmethod

from .exceptions import InsufficientCapitalError


class Fee(ABC):
    """
    Abstract ``Callable`` that calculates the total cost and number of shares
    given an asset price and the capital to be allocated to it.
    It returns the total cost and the number of shares aquired for that cost.
    """

    def __call__(self, price: float, capital: float = None, nshares: int = None):
        if nshares is None:
            shares = self.nshares(price, capital)
        else:
            shares = nshares
        cost = self.cost(price, shares)
        fee = cost - (price * shares)
        if cost > capital:
            raise InsufficientCapitalError(
                f"Tried to buy {shares} shares at {price} with only {capital}."
            )
        return {
            "nshares": shares,
            "total": cost,
            "fee": fee,
        }

    @abstractmethod
    def nshares(self, price, capital):
        pass

    @abstractmethod
    def cost(self, price, nshares):
        pass


class FlatPerTrade(Fee):
    def __init__(self, fee):
        self.fee = fee

    def nshares(self, price, capital):
        return (capital - self.fee) // price

    def cost(self, price, nshares):
        return price * nshares + self.fee


class FlatPerShare(Fee):
    def __init__(self, fee):
        self.fee = fee

    def nshares(self, price, capital):
        return capital // (self.fee + price)

    def cost(self, price, nshares):
        return (price + self.fee) * nshares


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

    def nshares(self, price, capital):
        return capital // price

    def cost(self, price, nshares):
        return price * nshares
