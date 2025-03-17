import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from zephyrcast import project_config
from zephyrcast.models.model_interface import ModelInterface
from torch.utils.data import DataLoader, TensorDataset

from zephyrcast.data.utils import find_latest_model_path, load_data_from_csv

class WeatherDataset(Dataset):
    def __init__(self, data, target, window_size, steps):
        self._data = data
        self._target = target
        self._window_size = window_size
        self._steps = steps

    def __len__(self):
        return len(self._data) - self._window_size - self._steps

    def __getitem__(self, idx):
        # input
        x = self._data.iloc[idx : idx + self._window_size][
            self._target
        ].values.astype(np.float32)
        # output
        y = self._data.iloc[idx + self._window_size : idx + self._window_size + self._steps][
            self._target
        ].values.astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


class Encoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True
        )
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x


class LSTMModel(ModelInterface):
    def __init__(self, steps: int, target: str):

        self._steps = 1#steps
        self._target = target
        self._window_size = 1 # 24
        self._is_trained = False

        # nearby_stations = 6
        # features_per_nearby_station = 15
        # features_per_target_station = 17
        # input_size = (
        #     nearby_stations * features_per_nearby_station + features_per_target_station
        # )
        input_size = 1
        # output_size = 3  # 0_wind_avg, 0_wind_bearing, 0_wind_gust
        hidden_size = 48
        self._batch_size = 8
        learning_rate = 0.001

        self._encoder = Encoder(input_size=input_size, hidden_size=hidden_size)
        self._criterion = nn.MSELoss()
        self._optimizer = torch.optim.Adam(self._encoder.parameters(),
            lr=learning_rate,
        )

        self._load_latest_model()

    @property
    def window_size(self):
        return self._window_size

    @property
    def is_trained(self):
        return self._is_trained

    @property
    def name(self):
        return f"lstm_X{self._target}"

    def _load_latest_model(self):
        try:
            latest_model_path = find_latest_model_path(suffix=".pth")
            checkpoint = torch.load(latest_model_path)

            self._encoder.load_state_dict(checkpoint["encoder"])

            print(f"Loaded LSTM model: {latest_model_path}")
            self._is_trained = True
        except Exception as ex:
            print(f"Failed to load model: {ex}")
            self._is_trained = False


    def _save_model(self):
        models_dir = project_config["models_dir"]
        model_path = os.path.join(models_dir, f"{self.name}.pth")
        torch.save(
            {
                "encoder": self._encoder.state_dict(),
            },
            model_path,
        )


    def train(self, data_train: pd.DataFrame, epochs=5):
        dataset = WeatherDataset(
            data=data_train, target=self._target, window_size=self._window_size, steps=self._steps
        )
        loader = DataLoader(dataset, batch_size=self._batch_size, shuffle=True)

        for epoch in range(epochs):
            self._encoder.train()
            epoch_loss = 0
            for x_batch, y_batch in loader:
                y_pred = self._encoder(x_batch)
                loss = self._criterion(y_pred, y_batch)
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                epoch_loss += loss.item()            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(loader)}")
        

        self._save_model()
        self._is_trained = True

    def predict(self, last_window: pd.DataFrame) -> pd.DataFrame:
        last_window_torch =  torch.from_numpy(last_window[self._target].to_numpy().astype(np.float32)).unsqueeze(-1)
        self._encoder.eval()
        with torch.no_grad():
            predictions_torch = self._encoder(last_window_torch)
        predictions = pd.DataFrame(predictions_torch.cpu().numpy())
        start_index = last_window.index[-1] + last_window.index.freq
        new_index = pd.date_range(
            start=start_index, periods=len(predictions), freq=last_window.index.freq
        )
        predictions.index = new_index
        return predictions