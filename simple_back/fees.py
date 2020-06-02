class FlatPerTrade:
    def __init__(self, fee):
        self.fee = fee

    def __call__(self, price, capital):
        num_shares = (capital - self.fee) // price
        return price * num_shares + self.fee, num_shares


class FlatPerShare:
    def __init__(self, fee):
        self.fee = fee

    def __call__(self, price, capital):
        num_shares = capital // (self.fee + price)
        return (price + self.fee) * num_shares, num_shares


def NoFee(price, capital):
    num_shares = capital // price
    return num_shares * price, num_shares
