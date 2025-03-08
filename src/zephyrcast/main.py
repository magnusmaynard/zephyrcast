#!/usr/bin/env python3
import click


@click.group(help="Command line tool for using the zephyrcast model. This model is for local, short-term weather forecasting specifically for paragliding safety, using solely weather station data from Zephyr.")
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
    "--files",
    "-f",
    multiple=True,
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, readable=True, path_type=str
    ),
    help="Predict on JSON file(s).",
)
def predict(live, files):
    from zephyrcast.predict import predict_live, predict_files

    if live:
        click.echo("Running in monitoring mode for predictions")
        if files:
            click.echo("--files option is ignored in live mode")

        predict_live()

    elif files:
        click.echo(f"Processing {len(files)} file(s) for prediction:")

        predict_files()
    else:
        raise click.UsageError("No prediction option specified.")


@cli.command(help="Clean any data and models.")
def clean():
    from zephyrcast.clean import clean

    clean()

if __name__ == "__main__":
    cli()
