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


def list_models(home: Path) -> list[tuple[str, list[Path]]]:
    """Return (name, weights path of each seed) for each model under home.

    A model is any directory (at any depth) holding seed<N> subdirectories; its name is its path relative to home.
    Seeds are in numeric order, and the path is the seed's model_0/best.pt whether or not it exists yet.
    """
    by_model: dict[Path, list[Path]] = {}
    for path in home.rglob("seed*"):
        if path.is_dir() and re.fullmatch(r"seed\d+", path.name):
            by_model.setdefault(path.parent, []).append(path)
    return [
        (
            str(model_dir.relative_to(home)),
            [seed / "model_0" / "best.pt" for seed in sorted(seeds, key=lambda p: int(p.name.removeprefix("seed")))],
        )
        for model_dir, seeds in sorted(by_model.items())
    ]


def _print_models(home: Path, models: list[tuple[str, list[Path]]]) -> None:
    table = Table(box=box.SIMPLE_HEAD, header_style="bold cyan", title=str(home), title_style="bold")
    table.add_column("Model", style="bold", no_wrap=True)
    table.add_column("Seeds", justify="right")
    for name, weights in models:
        trained = sum(w.is_file() for w in weights)
        seeds = str(len(weights)) if trained == len(weights) else f"[yellow]{trained}/{len(weights)}[/] [dim]trained[/]"
        table.add_row(name, seeds)
    console.print(table)


def _print_paths(models: list[tuple[str, list[Path]]]) -> None:
    """Print each model's per-seed weights paths as plain lines, ready to paste into `--input chemprop[...]`."""
    for name, weights in models:
        print(name)
        for w in weights:
            print(f"  {w}" + ("" if w.is_file() else "  (missing)"))


def main() -> None:
    """Entry point for the foundation-model CLI."""
    parser = argparse.ArgumentParser(
        description=f"Manage trained foundation models. Looks in ${HOME_ENV}, defaulting to {DEFAULT_HOME}.",
    )
    parser.add_argument("--list", action="store_true", help="List the available models with their seed counts")
    parser.add_argument("--paths", action="store_true", help="Show each model's per-seed best.pt path instead of the seed count")
    args = parser.parse_args()

    if not (args.list or args.paths):
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
    if args.paths:
        _print_paths(models)
    else:
        _print_models(home, models)


if __name__ == "__main__":
    main()
