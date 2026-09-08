"""Download CYP challenge dataset splits from HuggingFace to a local data folder."""

import argparse
from pathlib import Path

import polars as pl

_HF_BASE = "hf://datasets/openadmet/cyp-challenge-train-test/"
_HF_TRAIN = "cyp-challenge-TRAIN_inhibition.csv"
_HF_TEST = "cyp-challenge-TEST-BLINDED.csv"
_HF_TDI_TRAIN = "cyp-challenge-TRAIN_TDI.csv"
_HF_EMAX_TRAIN = "cyp-challenge-TRAIN_Emax.csv"
_HF_SINGLE_CONC_TRAIN = "cyp-challenge-single-concentration-TRAIN.csv"

_TRAIN_FILE = "train.csv"
_TEST_FILE = "test.csv"
_TDI_TRAIN_FILE = "tdi_train.csv"
_EMAX_TRAIN_FILE = "emax_train.csv"
_SINGLE_CONC_TRAIN_FILE = "single_concentration_train.csv"

_DEFAULT_DATA_DIR = Path("data")


def _download_split(hf_path: str, local_path: Path) -> None:
    if local_path.exists():
        print(f"Skipping (already exists): {local_path}")
        return
    local_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {hf_path} ...")
    df = pl.read_csv(hf_path)
    df.write_csv(local_path)
    print(f"Saved {len(df)} rows -> {local_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CYP challenge dataset splits.")
    parser.add_argument("--no-train", action="store_true", help="Skip the direct-inhibition train split.")
    parser.add_argument("--no-test", action="store_true", help="Skip the blinded test split.")
    parser.add_argument("--no-tdi", action="store_true", help="Skip the time-dependent inhibition (TDI) train split.")
    parser.add_argument("--no-emax", action="store_true", help="Skip the Emax train split.")
    parser.add_argument("--no-single-concentration", action="store_true", help="Skip the single-concentration train split.")
    parser.add_argument(
        "--train-location",
        type=Path,
        default=_DEFAULT_DATA_DIR / _TRAIN_FILE,
        help="Destination path for the direct-inhibition train CSV.",
    )
    parser.add_argument(
        "--test-location",
        type=Path,
        default=_DEFAULT_DATA_DIR / _TEST_FILE,
        help="Destination path for the blinded test CSV.",
    )
    parser.add_argument(
        "--tdi-location",
        type=Path,
        default=_DEFAULT_DATA_DIR / _TDI_TRAIN_FILE,
        help="Destination path for the TDI train CSV.",
    )
    parser.add_argument(
        "--emax-location",
        type=Path,
        default=_DEFAULT_DATA_DIR / _EMAX_TRAIN_FILE,
        help="Destination path for the Emax train CSV.",
    )
    parser.add_argument(
        "--single-concentration-location",
        type=Path,
        default=_DEFAULT_DATA_DIR / _SINGLE_CONC_TRAIN_FILE,
        help="Destination path for the single-concentration train CSV.",
    )
    args = parser.parse_args()

    if not args.no_train:
        _download_split(_HF_BASE + _HF_TRAIN, args.train_location)

    if not args.no_test:
        _download_split(_HF_BASE + _HF_TEST, args.test_location)

    if not args.no_tdi:
        _download_split(_HF_BASE + _HF_TDI_TRAIN, args.tdi_location)

    if not args.no_emax:
        _download_split(_HF_BASE + _HF_EMAX_TRAIN, args.emax_location)

    if not args.no_single_concentration:
        _download_split(_HF_BASE + _HF_SINGLE_CONC_TRAIN, args.single_concentration_location)


if __name__ == "__main__":
    main()
