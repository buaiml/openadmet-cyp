"""CLI for inspecting and filling the molecular-representation feature cache."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from data_tools.inputs import INPUT_REGISTRY, featurize, is_known_input
from data_tools.load import load_test, load_train

console = Console()


def _molecules(data_dir: Path) -> pl.DataFrame:
    """Unique SMILES of the challenge train and blinded test sets: what evaluate-models and generate-results featurize."""
    frames = [load_train(data_dir / "train.csv").select("SMILES"), load_test(data_dir / "test.csv").select("SMILES")]
    return pl.concat(frames).unique(subset=["SMILES"], keep="first", maintain_order=True)


def _size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    raise AssertionError


def _check(cache_dir: Path, smiles: set[str] | None) -> None:
    """Print one row per cached representation, plus registered ones that have no cache yet."""
    table = Table(box=box.SIMPLE_HEAD, header_style="bold cyan", pad_edge=False)
    table.add_column("Representation", style="bold", overflow="fold")
    table.add_column("Molecules", justify="right")
    table.add_column("Dim", justify="right")
    table.add_column("Size", justify="right")
    table.add_column("Updated", no_wrap=True)
    if smiles is not None:
        table.add_column("Covers", justify="right")

    cached: set[str] = set()
    for npy in sorted(cache_dir.glob("*.npy")) if cache_dir.exists() else []:
        idx = npy.with_suffix(".idx")
        if not idx.exists():
            table.add_row(escape(npy.stem), "[red]no .idx[/]", "", _size(npy.stat().st_size), "")
            continue
        cached.add(npy.stem)
        indexed = idx.read_text().splitlines()
        shape = np.load(npy, mmap_mode="r").shape
        stale = "[red]mismatch[/]" if shape[0] != len(indexed) else str(shape[0])
        row = [
            escape(npy.stem),
            stale,
            str(shape[1]) if len(shape) > 1 else "?",
            _size(npy.stat().st_size + idx.stat().st_size),
            time.strftime("%Y-%m-%d %H:%M", time.localtime(npy.stat().st_mtime)),
        ]
        if smiles is not None:
            have = len(smiles.intersection(indexed))
            colour = "green" if have == len(smiles) else "yellow" if have else "red"
            row.append(f"[{colour}]{have}/{len(smiles)}[/]")
        table.add_row(*row)

    for name in INPUT_REGISTRY:
        if name not in cached:
            table.add_row(f"[dim]{name}[/]", "[dim]not cached[/]", "", "", "", *([""] if smiles is not None else []))
    console.print(f"[dim]cache dir:[/] {cache_dir}  [dim]Covers = cached molecules out of {len(smiles)} train+test[/]" if smiles is not None else f"[dim]cache dir:[/] {cache_dir}")
    console.print(table)


def main() -> None:
    """Entry point for the representations CLI."""
    parser = argparse.ArgumentParser(description="Inspect or fill the representation feature cache.")
    parser.add_argument("--check", action="store_true", help="Show which representations are cached and how complete they are")
    parser.add_argument(
        "--embed",
        nargs="+",
        metavar="REPRESENTATION",
        help="Compute and cache these representations for the train and test molecules "
        "(a registered name, chemprop[/path/to/weights], or 'all'). Only uncached molecules are computed.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing train.csv and test.csv")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/features"), help="Directory for cached feature matrices")
    args = parser.parse_args()

    if not args.check and not args.embed:
        parser.error("pass --check and/or --embed")

    if args.embed:
        names = list(INPUT_REGISTRY) if args.embed == ["all"] else args.embed
        unknown = [n for n in names if not is_known_input(n)]
        if unknown:
            print(f"Unknown representation(s): {unknown}. Available: {list(INPUT_REGISTRY)} or chemprop[/path]", file=sys.stderr)
            sys.exit(1)
        molecules = _molecules(args.data_dir)
        console.print(f"Embedding {len(molecules)} train+test molecules")
        for name in names:
            try:
                features = featurize(molecules, [name], args.cache_dir)
            except Exception as exc:  # one failing representation should not stop the rest
                print(f"[{name}] failed: {exc}", file=sys.stderr)
                continue
            bad = int((~np.isfinite(features)).any(axis=1).sum())
            console.print(f"[green]✓[/] {escape(name)}: {features.shape[1]} dims" + (f", [yellow]{bad} molecules with non-finite values[/]" if bad else ""))

    if args.check:
        smiles = set(_molecules(args.data_dir)["SMILES"].to_list()) if (args.data_dir / "train.csv").exists() else None
        _check(args.cache_dir, smiles)


if __name__ == "__main__":
    main()
