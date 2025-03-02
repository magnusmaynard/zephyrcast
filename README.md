# zephyrcast
Local, short term forecasting using only weather station data from [Zephyr](https://zephyrapp.nz/). 

_Experimental and work in progress._

## Setup
Requires [poetry](https://python-poetry.org/docs/#installation) to be installed:
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

## Notes
- Repo used for initial experimentation with Zephyr API and data: https://github.com/lewinfox/
- Zephyr repo: https://github.com/kyzh0/zephyr