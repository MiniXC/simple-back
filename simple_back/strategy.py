from abc import ABC, abstractmethod


class Strategy(ABC):
    def __call__(self, day, event, bt):
        self.run(day, event, bt)

    @abstractmethod
    def run(self, day, event, bt):
        pass
