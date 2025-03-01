# zephyrcast
Local, short term forecasting using only weather station data from Zephyr. Experimentation and work in progress.

## Setup
Requires poetry to be installed:
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

