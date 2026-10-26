"""Train any supported model on the challenge plus auxiliary datasets.

  train-model --model chemprop  --datasets pubchem:CYP1A2 bindingdb:CYP2C8 --loss-function mae
  train-model --model chemeleon --datasets pubchem:CYP1A2
  train-model --model unimol    --datasets pubchem:CYP1A2 --learning-rate 2e-4

Repeat --datasets for several configs and add --n-seeds N (or --seeds 0 1 2) to
train every config once per seed (configs x seeds models):

  train-model --model unimol --n-seeds 3 \\
      --datasets challenge \\
      --datasets pubchem:CYP1A2 bindingdb:CYP2C8 \\
      --datasets no_challenge pubchem:CYP1A2 bindingdb:CYP2C8

Datasets, split and head selection are shared (models.training.common); each
backend adds its own flags. `--model chemeleon` is chemprop started from the
CheMeleon foundation weights. Flags a chemprop-based backend doesn't know are
passed through to `chemprop train`. See models.training to add a backend.
"""

import argparse
import copy
import sys

from data_tools.merge_datasets import describe
from models.training import BACKENDS
from models.training.common import add_common_arguments, slug

_MARKERS = {"challenge", "no_challenge"}  # dataset-config tokens, not sources


def main(argv: list[str] | None = None, model: str | None = None) -> None:
    """Entry point for the train-model CLI (`model` pins the backend for the legacy aliases)."""
    argv = sys.argv[1:] if argv is None else argv
    if model is None:
        pre = argparse.ArgumentParser(add_help=False)
        pre.add_argument("--model", choices=sorted(BACKENDS))
        known, _ = pre.parse_known_args(argv)
        model = known.model
    if model is None:
        if "--list-datasets" in argv:
            print(describe())
            return
        print(f"usage: train-model --model {{{','.join(sorted(BACKENDS))}}} [...]  (add --help after --model)", file=sys.stderr)
        sys.exit(2)

    backend = BACKENDS[model]
    parser = argparse.ArgumentParser(
        prog=f"train-model --model {model}",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", choices=sorted(BACKENDS), default=model, help="Which model to train")
    add_common_arguments(parser)
    backend.add_arguments(parser)
    parser.set_defaults(**backend.defaults)
    args, extra = parser.parse_known_args(argv)
    args.parser = parser

    if args.list_datasets:
        print(describe())
        return
    if extra and not backend.passthrough:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")

    seeds = args.seeds or (list(range(args.seed, args.seed + args.n_seeds)) if args.n_seeds else [args.seed])
    runs = [(group, seed) for group in args.datasets or [None] for seed in seeds]
    failed = []
    for index, (group, seed) in enumerate(runs, 1):
        run = copy.copy(args)
        run.seed = seed
        run.no_challenge_labels = False
        run.run_label = "challenge"
        if group is not None:
            tokens = [t for t in group if t not in _MARKERS]
            run.no_challenge_labels = "no_challenge" in group
            if run.no_challenge_labels and not backend.supports_no_challenge:
                parser.error(f"--model {model} cannot train without the challenge ('no_challenge')")
            if run.no_challenge_labels and not tokens:
                parser.error("'no_challenge' needs at least one auxiliary dataset in the same --datasets")
            run.datasets = tokens
            run.run_label = ("aux_only+" if run.no_challenge_labels else "") + slug(tokens)
        if len(runs) > 1:
            base = args.output_dir or backend.results_dir
            run.output_dir = base / run.run_label / f"seed{seed}"
        backend.validate(run, parser)
        print(f"\n##### run {index}/{len(runs)}: {run.run_label} seed {seed}", file=sys.stderr)
        code = backend.run(run, extra)
        if code != 0:
            failed.append(f"{run.run_label} seed {seed}")
    if failed:
        print(f"\n{len(failed)}/{len(runs)} runs failed: {', '.join(failed)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
