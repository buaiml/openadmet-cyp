"""CLI for managing trained foundation models (chemprop grid runs)."""

import argparse
import os
import re
import sys
from pathlib import Path

from rich import box
from rich.console import Console
from rich.table import Table

DEFAULT_HOME = Path("results/chemprop")
HOME_ENV = "FOUNDATION_MODEL_HOME"

console = Console()


def _home() -> Path:
    """Model directory: $FOUNDATION_MODEL_HOME if set, else results/chemprop."""
    return Path(os.environ.get(HOME_ENV) or DEFAULT_HOME)


def list_models(home: Path) -> list[tuple[str, int, int]]:
    """Return (name, seeds with a trained best.pt, total seed dirs) for each model under home.

    A model is any directory (at any depth) holding seed<N> subdirectories; its name is its path relative to home.
    """
    by_model: dict[Path, list[Path]] = {}
    for path in home.rglob("seed*"):
        if path.is_dir() and re.fullmatch(r"seed\d+", path.name):
            by_model.setdefault(path.parent, []).append(path)
    return [
        (
            str(model_dir.relative_to(home)),
            sum((seed / "model_0" / "best.pt").is_file() for seed in seeds),
            len(seeds),
        )
        for model_dir, seeds in sorted(by_model.items())
    ]


def _print_models(home: Path, models: list[tuple[str, int, int]]) -> None:
    table = Table(box=box.SIMPLE_HEAD, header_style="bold cyan", title=str(home), title_style="bold")
    table.add_column("Model", style="bold", no_wrap=True)
    table.add_column("Seeds", justify="right")
    for name, trained, total in models:
        seeds = str(total) if trained == total else f"[yellow]{trained}/{total}[/] [dim]trained[/]"
        table.add_row(name, seeds)
    console.print(table)


def main() -> None:
    """Entry point for the foundation-model CLI."""
    parser = argparse.ArgumentParser(
        description=f"Manage trained foundation models. Looks in ${HOME_ENV}, defaulting to {DEFAULT_HOME}.",
    )
    parser.add_argument("--list", action="store_true", help="List the available models with their seed counts")
    args = parser.parse_args()

    if not args.list:
        parser.print_help()
        return

    home = _home()
    if not home.is_dir():
        print(f"Foundation model home {home} does not exist (set ${HOME_ENV} to change it)", file=sys.stderr)
        sys.exit(1)
    models = list_models(home)
    if not models:
        print(f"No models found in {home}", file=sys.stderr)
        sys.exit(1)
    _print_models(home, models)


if __name__ == "__main__":
    main()
