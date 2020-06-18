class BacktestRunError(Exception):
    def __init__(self, message):
        self.message = message


class LongShortLiquidationError(Exception):
    def __init__(self, message):
        self.message = message


class NegativeValueError(Exception):
    def __init__(self, message):
        self.message = message


class TimeLeakError(Exception):
    def __init__(self, current_date, requested_date, message):
        self.current_date = current_date
        self.requested_date = requested_date
        self.message = message


class PriceUnavailableError(Exception):
    def __init__(self, symbol, requested_date, message):
        self.symbol = symbol
        self.requested_date = requested_date
        self.message = message


class InsufficientCapitalError(Exception):
    def __init__(self, message):
        self.message = message


class MissingMetricsError(Exception):
    def __init__(self, metrics, message):
        self.metrics = metrics
        self.message = message
