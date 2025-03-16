import pandas as pd
from zephyrcast.models.model_interface import ModelInterface


class BaselineModel(ModelInterface):
    def __init__(self, steps: int, target: str):
        self._steps = steps
        self._target = target
        self._context = 1

    @property
    def window_size(self):
        return self._context
    
    @property
    def is_trained(self):
        return True

    @property
    def name(self):
        return f"baseline_X{self._target}"
    
    def train(self, data_train: pd.DataFrame):
        pass

    def predict(self, last_window: pd.DataFrame) -> pd.DataFrame:
        predictions_index = pd.Index([last_window.index[-1] + (i + 1)* last_window.index.freq  for i in range(self._steps)])

        last_known_value = last_window.iloc[-1][self._target]
        return pd.Series([last_known_value] * self._steps, index=predictions_index)
