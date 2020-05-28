from abc import ABC, abstractmethod


class Strategy(ABC):
    @abstractmethod
    def run(self, day, event, bt):
        pass
