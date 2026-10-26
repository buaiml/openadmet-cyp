"""Interface every training backend implements."""

import argparse


class Backend:
    """One trainable model family: its extra CLI flags and its run function."""

    #: `chemprop`-style CLIs forward flags they don't know to the underlying tool.
    passthrough = False
    #: Per-backend defaults for the shared flags (epochs, ...).
    defaults: dict[str, object] = {}

    def __init__(self, name: str) -> None:
        self.name = name

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add this backend's own flags to the shared parser."""

    def validate(self, args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
        """Reject flag combinations that make no sense for this backend."""

    def run(self, args: argparse.Namespace, extra_args: list[str]) -> int:
        """Train; return a process exit code."""
        raise NotImplementedError
