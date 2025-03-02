# zephyrcast
Local, short-term weather forecasting specifically for paragliding safety, using solely weather station data from [Zephyr](https://zephyrapp.nz/).

Work in progress: This proof of concept aims to develop an early warning system to alert paragliders of approaching dangerous conditions.

## Setup
This project uses [poetry](https://python-poetry.org/docs/#installation), ensure this is install first.
```
poetry install
poetry run keyring set zephyr_api_key zephyrcast
```

## Run
```
poetry run fetch
poetry run prepare
poetry run train
```

## Links
- [zephyr-model repo](https://github.com/lewinfox/zephyr-model) - Used for initial experimentation and as an example how to use Zephyr API and data.
- [zephyr repo](https://github.com/kyzh0/zephyr) - Implementation details on Zephyr API.