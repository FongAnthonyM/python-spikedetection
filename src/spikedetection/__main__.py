"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """Python Spikedetection."""


if __name__ == "__main__":
    main(prog_name="python-spikedetection")  # pragma: no cover
