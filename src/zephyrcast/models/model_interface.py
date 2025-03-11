
from abc import abstractmethod, ABC
import pandas as pd


class ModelInterface(ABC):
    @property
    @abstractmethod
    def context(self):
        pass

    @abstractmethod
    def train(self, data_train: pd.DataFrame):
        pass

    @abstractmethod
    def predict(self, last_window: pd.DataFrame):
        pass