"""Create and save a development dataset for the ensemble smoke test.

Split challenge training data into scaffold-based train/validation/test
populations, validate molecule identifiers, and preserve existing output.
"""

from pathlib import Path

import polars as pl

from data_tools.prepare_ensemble import (
    SCORED_TARGETS,
    prepare_ensemble_data,
)
from data_tools.split_column import assign_splits


def main():
    output_path = Path("data/ensemble_smoke.csv")
    if output_path.exists():
        raise FileExistsError(f"Preserving existing splits: {output_path}")

    data = pl.read_csv("data/train.csv", infer_schema_length=None)

    splits = assign_splits(
        data,
        scored_columns=SCORED_TARGETS,
        strategy="scaffold",
        val_fraction=0.1,
        test_fraction=0.1,
        seed=42,
    )

    # Adds/verifies identifiers and rejects connectivity blocks
    # that cross splits.
    prepared = prepare_ensemble_data(data.with_columns(splits))
    prepared.write_csv(output_path)

    print(prepared.group_by("split").len().sort("split"))
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()