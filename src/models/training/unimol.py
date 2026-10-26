"""Uni-Mol backend: fine-tune jointly on the challenge plus any auxiliary datasets.

One model per (learning rate, seed) with every head at once. Missing labels are
masked in the loss (unimol_tools' multilabel_regression uses a NaN-masked MAE),
so molecules measured by only one source still train that head. Auxiliary-only
molecules always land in train, so no validation molecule is seen in training.

  train-model --model unimol --datasets pubchem:CYP1A2 bindingdb:CYP1A2,CYP2C8
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from data_tools.merge_datasets import parse_dataset_specs
from models.training.base import Backend
from models.training.common import prepare_joint_split, slug, write_val_outputs


def set_seed(seed: int) -> None:
    """Seed the main random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class UnimolBackend(Backend):
    defaults = {"epochs": 30}

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--learning-rate", type=float, default=2e-4)
        parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")

    def run(self, args: argparse.Namespace, extra_args: list[str]) -> int:
        from unimol_tools import MolPredict, MolTrain  # heavy and optional: only needed for this backend

        try:
            specs = parse_dataset_specs(args.datasets or [], with_challenge=False)
        except ValueError as exc:
            args.parser.error(str(exc))
        set_seed(args.seed)

        train_df, val_df, target_columns, eval_heads = prepare_joint_split(specs, args, args.parser)
        datasets = args.datasets or []
        use_cuda = torch.cuda.is_available() and not args.cpu

        base_dir = args.output_dir or Path("results/unimol") / slug(datasets)
        output_dir = base_dir / f"lr_{args.learning_rate}_seed_{args.seed}"
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Datasets: {', '.join(datasets) or 'challenge only'}")
        print(f"Heads: {len(target_columns)} ({len(eval_heads)} scored)")
        print(f"Training molecules: {len(train_df)}")
        print(f"Validation molecules: {len(val_df)}")
        print(f"Learning rate: {args.learning_rate}")
        print(f"Seed: {args.seed}")
        print(f"Device: {'CUDA' if use_cuda else 'CPU'}")

        trainer = MolTrain(
            task="multilabel_regression",  # NaN-masked loss
            data_type="molecule",
            model_name="unimolv1",
            smiles_col="SMILES",
            target_cols=target_columns,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            save_path=str(output_dir),
            use_cuda=use_cuda,
            kfold=1,
            split_seed=args.seed,
            seed=args.seed,
            metrics="none",
        )
        # No official validation rows are passed to the trainer.
        trainer.fit(train_df[["SMILES", *target_columns]])

        predictor = MolPredict(load_model=str(output_dir))
        predictions = np.asarray(predictor.predict(val_df[["SMILES"]])).reshape(len(val_df), len(target_columns))
        if not np.isfinite(predictions).all():
            raise RuntimeError("Non-finite predictions returned by Uni-Mol.")

        write_val_outputs(
            output_dir,
            f"unimol_{slug(datasets)}_lr{args.learning_rate}_seed{args.seed}",
            val_df,
            train_df,
            dict(zip(target_columns, predictions.T)),
            eval_heads,
            {"learning_rate": args.learning_rate, "seed": args.seed},
        )
        return 0
