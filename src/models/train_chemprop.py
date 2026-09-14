"""CLI wrapper to train a Chemprop D-MPNN on any SMILES/target CSV.

Built for joint multi-task training rather than pretrain-then-finetune: pass
every head at once via --target-columns (see data_tools.build_union, which
emits the wide challenge + auxiliary table). One shared encoder learns from
all sources; each head keeps its own scale, so no cross-assay calibration is
required. Chemprop masks missing targets in the loss automatically, so the
sparse block structure of a multi-source union costs nothing -- molecules
measured by only one source simply contribute to that head.

Thin subprocess wrapper around the `chemprop train` CLI so all of chemprop's
own flags (--batch-size, --num-workers, --accelerator, --split-type,
--task-weights, ...) stay available via passthrough, instead of being
re-declared here.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def build_command(
    data_path: Path,
    smiles_column: str,
    target_columns: list[str],
    output_dir: Path,
    epochs: int,
    seed: int,
    from_foundation: str | None,
    extra_args: list[str],
) -> list[str]:
    """Build the `chemprop train` argv for the given dataset and targets."""
    cmd = [
        "chemprop",
        "train",
        "--data-path",
        str(data_path),
        "--smiles-columns",
        smiles_column,
        "--target-columns",
        *target_columns,
        "--task-type",
        "regression",
        "--output-dir",
        str(output_dir),
        "--epochs",
        str(epochs),
        "--pytorch-seed",
        str(seed),
    ]
    if from_foundation is not None:
        value = "CHEMELEON" if from_foundation.lower() == "chemeleon" else from_foundation
        cmd += ["--from-foundation", value]
    return cmd + extra_args


def main() -> None:
    """Entry point for the train-chemprop CLI."""
    parser = argparse.ArgumentParser(
        description="Train a Chemprop D-MPNN on any SMILES/target CSV (passes unknown flags through to `chemprop train`).",
    )
    parser.add_argument("--data-path", type=Path, required=True, help="CSV with a SMILES column and target column(s)")
    parser.add_argument("--smiles-column", default="SMILES", help="Name of the SMILES column (default: SMILES)")
    parser.add_argument(
        "--target-columns",
        nargs="+",
        required=True,
        help="Name(s) of the target column(s) to train on. Pass every head for joint multi-task "
        "training; missing values are masked in the loss by chemprop.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for chemprop outputs (default: results/chemprop/<data stem>_<targets>)",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument("--seed", type=int, default=42, help="PyTorch seed (default: 42)")
    parser.add_argument(
        "--from-foundation",
        default=None,
        help="Foundation checkpoint to initialize message passing from: 'chemeleon' or a path to a local .pt file",
    )
    args, extra_args = parser.parse_known_args()

    if shutil.which("chemprop") is None:
        print("chemprop CLI not found on PATH. Install it with: pip install chemprop", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir
    if output_dir is None:
        target_tag = "+".join(args.target_columns)
        output_dir = Path("results/chemprop") / f"{args.data_path.stem}_{target_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_command(
        args.data_path,
        args.smiles_column,
        args.target_columns,
        output_dir,
        args.epochs,
        args.seed,
        args.from_foundation,
        extra_args,
    )
    print(f"Running: {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd, check=False)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
