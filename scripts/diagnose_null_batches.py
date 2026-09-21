"""Find the all-unlabelled batches that silently kill a chemprop run.

Chemprop reduces its loss as `total_loss / num_samples`, where `num_samples`
counts *valid* (non-null) targets in the batch -- `chemprop/nn/metrics.py`. A
batch in which no row carries a single non-null target therefore evaluates
0 / 0 = NaN, the NaN reaches the gradients, and every weight in the model
becomes NaN. Training then runs to completion producing nothing: the best
checkpoint stays wherever it was before the NaN, and chemprop scores the test
set with that checkpoint. The run looks finished and reports a plausible
number.

`build-union` makes this easy to hit. Auxiliary-only molecules are kept as
rows so that selecting an auxiliary head brings its molecules along, so a
configuration that selects *few* auxiliary heads trains on a table that is
mostly unlabelled -- 10.5% labelled rows for the no-auxiliary baseline of the
2026-09-14 search. At that density a 64-row batch is empty with probability
1.6e-4, which over 30 epochs of ~584 batches is ~2.8 expected hits.

The batch sequence is reproducible, which is what makes this diagnosable
rather than merely probable: the training dataloader shuffles with
`--data-seed` (default 0), *not* with `--pytorch-seed`. Repeating a run with
three pytorch seeds therefore repeats the identical batch partition three
times and kills all three at the same step, which is why dead configurations
report a seed spread near zero.

This script replays that exact sampler (`numpy.random.default_rng(data_seed)`,
shuffled once per epoch, chunked by batch size) over a union table and reports,
per head configuration, the epoch at which the first empty batch appears.

Usage:
    python scripts/diagnose_null_batches.py --data-path data/union_train.csv
    python scripts/diagnose_null_batches.py --data-path data/union_train.csv --epochs 30 --batch-size 64
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

_SCORED_DEFAULT = [
    "CYP1A2_pIC50_direct_inhibition",
    "CYP2C9_pIC50_direct_inhibition",
    "CYP2D6_pIC50_direct_inhibition",
    "CYP3A4_pIC50_direct_inhibition",
]
_SKIP = {"SMILES", "inchikey_block", "inchikey_full", "split", "PUBCHEM_CID"}


def labelled_mask(df: pl.DataFrame, columns: list[str]) -> np.ndarray:
    """Boolean mask: does this row carry at least one non-null value in `columns`?"""
    return df.select(pl.any_horizontal([pl.col(c).is_not_null() for c in columns]))[:, 0].to_numpy()


def batch_sequence(n_rows: int, batch_size: int, epochs: int, data_seed: int) -> list[list[np.ndarray]]:
    """Replay chemprop's SeededSampler: one in-place shuffle per epoch, then chunk.

    The same sequence serves every configuration, because the sampler depends
    only on row count and `--data-seed` -- not on which columns are targets.
    """
    rng = np.random.default_rng(data_seed)
    indices = np.arange(n_rows)
    sequence = []
    for _ in range(epochs):
        rng.shuffle(indices)
        sequence.append([indices[i : i + batch_size].copy() for i in range(0, n_rows, batch_size)])
    return sequence


def first_empty_batch(mask: np.ndarray, sequence: list[list[np.ndarray]]) -> tuple[int, int] | None:
    """Return (epoch, global step) of the first batch with no labelled row, or None."""
    step = 0
    for epoch, batches in enumerate(sequence):
        for batch in batches:
            if not mask[batch].any():
                return epoch, step
            step += 1
    return None


def main() -> None:
    """Entry point: report labelled density and time-to-NaN for every solo config."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-path", type=Path, default=Path("data/union_train.csv"))
    parser.add_argument("--scored-columns", nargs="+", default=_SCORED_DEFAULT)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--data-seed", type=int, default=0, help="chemprop's --data-seed (default 0, as chemprop's)")
    args = parser.parse_args()

    df = pl.read_csv(args.data_path, infer_schema_length=None)
    if "split" in df.columns:
        df = df.filter(pl.col("split") == "train")
    scored = [c for c in args.scored_columns if c in df.columns]
    aux = [c for c in df.columns if c not in set(scored) | _SKIP and df[c].dtype.is_numeric()]

    sequence = batch_sequence(len(df), args.batch_size, args.epochs, args.data_seed)
    base = labelled_mask(df, scored)

    candidates: dict[str, list[str]] = {}
    for column in aux:
        candidates.setdefault(column.split("_")[0], []).append(column)

    print(
        f"{args.data_path}: {len(df)} train rows, {args.epochs} epochs x "
        f"{len(sequence[0])} batches of {args.batch_size}\n",
        file=sys.stderr,
    )
    header = f"{'config':12s} {'labelled':>9s} {'P(empty batch)':>15s} {'first empty batch':>19s}"
    print(header)
    print("-" * len(header))

    def report(name: str, mask: np.ndarray) -> None:
        hit = first_empty_batch(mask, sequence)
        fraction = 1 - mask.mean()
        where = f"epoch {hit[0]}, step {hit[1]}" if hit else "none in range"
        print(f"{name:12s} {mask.mean():8.1%} {fraction ** args.batch_size:15.2e} {where:>19s}")

    report("baseline", base)
    for name, columns in sorted(candidates.items()):
        report(name, base | labelled_mask(df, columns))


if __name__ == "__main__":
    main()
