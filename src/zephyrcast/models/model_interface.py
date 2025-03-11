
from abc import abstractmethod, ABC
import pandas as pd


class ModelInterface(ABC):
    @property
    @abstractmethod
    def window_size(self):
        pass

    @property
    @abstractmethod
    def is_trained(self):
        pass

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def train(self, data_train: pd.DataFrame):
        pass

    @abstractmethod
    def predict(self, last_window: pd.DataFrame):
        pass