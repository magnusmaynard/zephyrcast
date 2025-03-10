# zephyrcast
Local, short-term weather forecasting specifically for paragliding safety, using solely weather station data from [Zephyr](https://zephyrapp.nz/).

Work in progress: This proof of concept aims to develop an early warning system to alert paragliders of approaching dangerous conditions.

## Setup
This project uses [poetry](https://python-poetry.org/docs/#installation), ensure this is installed.
```
poetry install
poetry run keyring set zephyr_api_key zephyrcast
```

## Usage
First train the model on all of the downloaded data:
```
poetry run zcast fetch
poetry run zcast prepare
poetry run zcast train
```

Predict on downloaded dataset:
```
poetry run zcast predict --date "2025-02-15 12:30:00"
```

See help for more details on the available commands:
```
poetry run zcast --help
```


## Links
- [zephyr-model repo](https://github.com/lewinfox/zephyr-model) - Used for initial experimentation and as an example how to use Zephyr API and data.
- [zephyr repo](https://github.com/kyzh0/zephyr) - Implementation details on Zephyr API.e