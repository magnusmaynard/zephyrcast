from datetime import datetime
from typing import Literal

from click import UsageError

from zephyrcast.data.utils import get_train_test_data
from zephyrcast.models.baseline_model import BaselineModel
from zephyrcast.models.multivar_forecast_model import MultiVariantForecastModel
from zephyrcast.models.seq2seq_model import Seq2SeqModel


class ModelHarness:
    def __init__(
        self, arch: Literal["baseline", "multivar", "seq2seq"], steps: int, target: str
    ):
        print("Initialising model")
        self._steps = steps
        self._target = target
        if arch == "multivar":
            self._model = MultiVariantForecastModel(steps=steps, target=target)
        elif arch == "seq2seq":
            self._model = Seq2SeqModel(steps=steps, target=target)
        else:
            self._model = BaselineModel(steps=steps, target=target)

        self._data_filename = "rocky_gully_near_6_features.csv"
        self._data_split_date = datetime.strptime("2025-02-07 23:59:00", "%Y-%m-%d %H:%M:%S")
        self._data_train, self._data_test = get_train_test_data(
            filename=self._data_filename, train_split_date=self._data_split_date
        )

    def train(self):
        print("Training...")
        self._model.train(data_train=self._data_train)

    def evaluate(self):
        print("Evaluating...")
        self._model.evaluate(data_test=self._data_test, save_plots=True)

    def predict(
        self,
        start_date: datetime
    ):
        print("Predicting...")
        if start_date <= self._data_split_date:
            raise UsageError(
                f"The testing date {start_date.strftime('%Y-%m-%d %H:%M:%S')} is within the training range {self._data_split_date}. Please choose a later date."
            )

        self._model.predict(data_test=self._data_test, start_date=start_date)
