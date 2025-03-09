#!/usr/bin/env python3
import click
from datetime import datetime


@click.group(
    help="Command line tool for using the zephyrcast model. This model is for local, short-term weather forecasting specifically for paragliding safety, using solely weather station data from Zephyr."
)
def cli():
    pass


@cli.command(help="Fetch all raw data from Zephyr API and store locally.")
def fetch():
    from zephyrcast.fetch import fetch

    fetch()


@cli.command(help="Prepare and preprocess the data for training.")
def prepare():
    from zephyrcast.prepare import prepare

    prepare()


@cli.command(help="Train zephyrcast model on prepared data.")
def train():
    from zephyrcast.train import train

    train()


@cli.command(help="Run predictions using the trained zephyrcast model on new data.")
@click.option("--live", is_flag=True, help="Predict on live data from the Zephyr API.")
@click.option(
    "--date",
    type=click.DateTime(formats=["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"]),
    help="Start datetime for prediction (format: YYYY-MM-DD, YYYY-MM-DDThh:mm:ss, or YYYY-MM-DD hh:mm:ss).",
)
def predict(live, date):
    from zephyrcast.predict import predict_files, predict_live

    if date:
        start_datetime = (
            date if isinstance(date, datetime) else datetime.fromisoformat(date)
        )

        predict_files(start_date=start_datetime)

        click.echo(f"Using start datetime: {start_datetime}")

    elif live:
        click.echo("Running in live mode for predictions")
        predict_live()
    else:
        raise click.UsageError("No prediction option specified.")


@cli.command(help="Clean any data and models.")
def clean():
    from zephyrcast.clean import clean

    clean()


if __name__ == "__main__":
    cli()
