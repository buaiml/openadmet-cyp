"""CLI wrapper to train a Chemprop D-MPNN on any SMILES/target CSV.

Built for joint multi-task training: pass every head at once (see
data_tools.build_union, or --datasets below). One shared encoder learns from
all sources; each head keeps its own scale, so no cross-assay calibration is
required. Chemprop masks missing targets in the loss automatically, so the
sparse block structure of a multi-source union costs nothing -- molecules
measured by only one source simply contribute to that head.

Thin subprocess wrapper around the `chemprop train` CLI so all of chemprop's
own flags (--batch-size, --num-workers, --accelerator, --task-weights, ...)
stay available via passthrough, instead of being re-declared here. Passthrough
flags apply to every stage.

Data comes from either a ready table (--data-path) or from the named sources,
merged at run time into one wide table with a head per source x isoform
(CYP2D6-BindingDB, CYP3A4-TDI, ...). See `--list-datasets`.

Stages. `--datasets A B C` trains everything jointly in one stage. Repeat
`--stage` to pretrain first: stages run in order and each one starts from the
previous stage's message-passing encoder and aggregation (the heads are fresh,
so stages can have different heads and even different task types).

  # one stage, everything jointly
  train-chemprop --datasets tdc_hlm tdc_rlm pubchem:CYP1A2 --loss-function mae

  # pretrain on stability, then train on the challenge plus auxiliary heads
  train-chemprop --stage tdc_hlm tdc_rlm --stage pubchem:CYP1A2 bindingdb:CYP2C8

The challenge (scored heads and the pinned split) is always part of the last
stage. Earlier stages train only on the datasets they list, on molecules in
the pinned train split, so no val/test molecule is seen during pretraining;
they use chemprop's own random split of those rows for early stopping.
A stage whose heads are all binary is trained as classification. A stage that
mixes binary and regression heads is trained as regression and the 0/1 labels
are treated as numbers.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import polars as pl

from data_tools.merge_datasets import describe, merge, parse_dataset_specs


def build_command(
    data_path: Path,
    smiles_column: str,
    target_columns: list[str],
    output_dir: Path,
    epochs: int,
    seed: int,
    from_foundation: str | None,
    extra_args: list[str],
    task_type: str = "regression",
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
        task_type,
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


def find_best_model(output_dir: Path) -> Path:
    """Return the best checkpoint a finished chemprop run wrote (first model if several)."""
    found = sorted(output_dir.rglob("best*.pt"))
    if not found:
        raise FileNotFoundError(f"no best*.pt under {output_dir}; did chemprop finish?")
    if len(found) > 1:
        print(f"  {len(found)} models under {output_dir}; chaining from {found[0]}", file=sys.stderr)
    return found[0]


def _slug(specs: list[str]) -> str:
    return "+".join(specs).replace(":", "-").replace(",", "_")


def main() -> None:
    """Entry point for the train-chemprop CLI."""
    parser = argparse.ArgumentParser(
        description="Train a Chemprop D-MPNN on any SMILES/target CSV (passes unknown flags through to `chemprop train`).",
        epilog="Chemprop's own flags (--depth, --batch-size, --accelerator, ...) pass through to every stage: "
        "see `chemprop train --help`. Use --list-datasets for the sources.",
    )
    parser.add_argument("--data-path", type=Path, default=None, help="CSV with a SMILES column and target column(s)")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        metavar="NAME[:ISO,ISO]",
        help="Train these sources jointly in one stage (challenge is always included). Heads are named "
        "{isoform}-{source}, e.g. bindingdb:CYP2D6,CYP1A2 tdi pubchem",
    )
    parser.add_argument(
        "--stage",
        nargs="+",
        action="append",
        default=None,
        metavar="NAME[:ISO,ISO]",
        help="One training stage; repeat for several. Each stage starts from the previous stage's encoder. "
        "The challenge joins the last stage automatically.",
    )
    parser.add_argument(
        "--stage-epochs",
        nargs="+",
        type=int,
        default=None,
        help="Epochs per stage, in order (default: --epochs for every stage)",
    )
    parser.add_argument("--list-datasets", action="store_true", help="List selectable datasets and exit")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Where the source CSVs live (default: data)")
    parser.add_argument("--split-strategy", default="scaffold", choices=["scaffold", "butina", "random"])
    parser.add_argument("--split-seed", type=int, default=0, help="Seed for the merged table's split (default: 0)")
    parser.add_argument("--smiles-column", default="SMILES", help="Name of the SMILES column (default: SMILES)")
    parser.add_argument(
        "--target-columns",
        nargs="+",
        default=None,
        help="Name(s) of the target column(s) to train on. Required with --data-path; with --datasets/--stage "
        "the default is every head of the stage. Missing values are masked in the loss by chemprop.",
    )
    parser.add_argument(
        "--loss-function",
        default=None,
        help="Loss for regression stages (e.g. mae, mse). Classification stages keep chemprop's default.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for outputs (default: results/chemprop/<datasets>). Several stages write to "
        "<output-dir>/stage1, stage2, ...",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument("--seed", type=int, default=42, help="PyTorch seed (default: 42)")
    parser.add_argument(
        "--from-foundation",
        default=None,
        help="Foundation checkpoint to initialize the first stage's message passing from: 'chemeleon' or a "
        "path to a local model file. Later stages start from the previous stage instead.",
    )
    args, extra_args = parser.parse_known_args()

    if args.list_datasets:
        print(describe())
        return
    if args.datasets is not None and args.stage is not None:
        parser.error("--datasets and --stage are alternatives; use --stage once per stage")
    stage_specs = [args.datasets] if args.datasets is not None else args.stage
    if (args.data_path is None) == (stage_specs is None):
        parser.error("pass exactly one of --data-path, --datasets or --stage")
    if args.data_path is not None and not args.target_columns:
        parser.error("--target-columns is required with --data-path")
    if args.stage_epochs is not None and stage_specs is not None and len(args.stage_epochs) != len(stage_specs):
        parser.error(f"--stage-epochs needs one value per stage ({len(stage_specs)})")

    if shutil.which("chemprop") is None:
        print("chemprop CLI not found on PATH. Install it with: pip install chemprop", file=sys.stderr)
        sys.exit(1)

    if stage_specs is None:  # a ready table: one plain run
        output_dir = args.output_dir or Path("results/chemprop") / f"{args.data_path.stem}_{'+'.join(args.target_columns)}"
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_command(
            args.data_path, args.smiles_column, args.target_columns, output_dir, args.epochs, args.seed,
            args.from_foundation, _with_loss(args.loss_function, "regression") + extra_args,
        )
        print(f"Running: {' '.join(cmd)}", file=sys.stderr)
        sys.exit(subprocess.run(cmd, check=False).returncode)

    try:
        parsed = [parse_dataset_specs(s, with_challenge=False) for s in stage_specs]
    except ValueError as exc:
        parser.error(str(exc))
    n_stages = len(parsed)
    output_dir = args.output_dir or Path("results/chemprop") / "__".join(_slug(s) for s in stage_specs)
    output_dir.mkdir(parents=True, exist_ok=True)

    foundation = args.from_foundation
    for index, specs in enumerate(parsed):
        final = index == n_stages - 1
        label = f"stage {index + 1}/{n_stages}"
        stage_dir = output_dir if n_stages == 1 else output_dir / f"stage{index + 1}"
        stage_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== {label}: {', '.join(specs) or 'challenge only'} ===", file=sys.stderr)
        table, heads, scored, tasks = merge(
            {"challenge": None, **specs},  # the challenge supplies the scored heads and the pinned split
            data_dir=args.data_dir,
            split_strategy=args.split_strategy,
            split_seed=args.split_seed,
        )
        stage_heads = [h for h in heads if final or h not in scored]
        unknown = [c for c in args.target_columns or [] if c not in stage_heads]
        if unknown:
            parser.error(f"--target-columns not among {label} heads {stage_heads}: {unknown}")
        target_columns = args.target_columns or stage_heads

        if final:
            extra = ["--splits-column", "split", *extra_args]
        else:
            # Pretraining must not see molecules held out for evaluation.
            table = table.filter(pl.col("split") == "train")
            extra = list(extra_args)
        table = table.filter(pl.any_horizontal([pl.col(h).is_not_null() for h in target_columns]))

        kinds = {tasks[h] for h in target_columns}
        task_type = "classification" if kinds == {"classification"} else "regression"
        if kinds == {"classification", "regression"}:
            print("  stage mixes binary and regression heads: training as regression, 0/1 labels as numbers", file=sys.stderr)

        data_path = stage_dir / "merged.csv"
        table.select(["SMILES", *target_columns, "split"]).write_csv(data_path)
        print(
            f"  {len(table)} molecules, {len(target_columns)} heads, {task_type}; wrote {data_path}",
            file=sys.stderr,
        )

        epochs = args.stage_epochs[index] if args.stage_epochs else args.epochs
        cmd = build_command(
            data_path, args.smiles_column, target_columns, stage_dir, epochs, args.seed, foundation,
            _with_loss(args.loss_function, task_type) + extra, task_type,
        )
        print(f"Running: {' '.join(cmd)}", file=sys.stderr)
        code = subprocess.run(cmd, check=False).returncode
        if code != 0:
            print(f"{label} failed with exit code {code}", file=sys.stderr)
            sys.exit(code)
        if not final:
            foundation = str(find_best_model(stage_dir))
            print(f"  next stage starts from {foundation}", file=sys.stderr)

    print(f"\nDone. Final model: {find_best_model(stage_dir)}", file=sys.stderr)


def _with_loss(loss_function: str | None, task_type: str) -> list[str]:
    """Loss flag for a stage; only regression stages take --loss-function."""
    return ["--loss-function", loss_function] if loss_function and task_type == "regression" else []


if __name__ == "__main__":
    main()
