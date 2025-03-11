#!/usr/bin/env python3
import click
from datetime import datetime

from zephyrcast.models.model_harness import ModelHarness
from zephyrcast.data.fetch import fetch_data
from zephyrcast.data.prepare import prepare_data

DEFAULT_ARCH = "multivar"
ARCH_OPTIONS = ["baseline", "multivar", "seq2seq"]
TARGET = "0_wind_avg"
STEPS = 6


def get_model(arch=DEFAULT_ARCH):
    return ModelHarness(arch=arch, steps=STEPS, target=TARGET)


@click.group(
    help="Command line tool for using the zephyrcast model. This model is for local, short-term weather forecasting specifically for paragliding safety, using solely weather station data from Zephyr."
)
def cli():
    pass


@cli.command(help="Fetch all raw data from Zephyr API and store locally.")
def fetch():
    fetch_data()


@cli.command(help="Prepare and preprocess the data for training.")
def prepare():
    prepare_data()


@cli.command(help="Train zephyrcast model on prepared data.")
@click.option(
    "--arch",
    type=click.Choice(ARCH_OPTIONS),
    default=DEFAULT_ARCH,
    help="Model architecture to use for training.",
)
def train(arch):
    model = get_model(arch)
    model.train()


@cli.command(help="Measure performance of zephyrcast model using backtesting, on prepared data.")
@click.option(
    "--arch",
    type=click.Choice(ARCH_OPTIONS),
    default=DEFAULT_ARCH,
    help="Model architecture to use for evaluation.",
)
def evaluate(arch):
    model = get_model(arch)
    model.evaluate()


@cli.command(help="Run predictions using the trained zephyrcast model on prepared data.")
@click.option(
    "--live", is_flag=True, help="Predict on live data from the Zephyr API."
)
@click.option(
    "--date",
    type=click.DateTime(
        formats=["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"]
    ),
    help="Start datetime for prediction (format: YYYY-MM-DD, YYYY-MM-DDThh:mm:ss, or YYYY-MM-DD hh:mm:ss).",
)
@click.option(
    "--arch",
    type=click.Choice(ARCH_OPTIONS),
    default=DEFAULT_ARCH,
    help="Model architecture to use for prediction.",
)
def predict(live, date, arch):
    model = get_model(arch)
    if date:
        start_datetime = (
            date if isinstance(date, datetime) else datetime.fromisoformat(date)
        )
        model.predict(date=start_datetime)
    elif live:
        raise NotImplementedError("Live predictions are not implemented")
    else:
        raise click.UsageError("No prediction option specified.")


@cli.command(help="Clean any data and models.")
def clean():
    raise NotImplementedError("Clean is not implemented")


if __name__ == "__main__":
    cli()
