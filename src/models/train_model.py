"""Train any supported model on the challenge plus auxiliary datasets.

  train-model --model chemprop  --datasets pubchem:CYP1A2 bindingdb:CYP2C8 --loss-function mae
  train-model --model chemeleon --datasets pubchem:CYP1A2
  train-model --model unimol    --datasets pubchem:CYP1A2 --learning-rate 2e-4

Datasets, split and head selection are shared (models.training.common); each
backend adds its own flags. `--model chemeleon` is chemprop started from the
CheMeleon foundation weights. Flags a chemprop-based backend doesn't know are
passed through to `chemprop train`. See models.training to add a backend.
"""

import argparse
import sys

from data_tools.merge_datasets import describe
from models.training import BACKENDS
from models.training.common import add_common_arguments


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
    backend.validate(args, parser)
    sys.exit(backend.run(args, extra))


if __name__ == "__main__":
    main()
