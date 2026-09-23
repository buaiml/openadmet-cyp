"""Measure how close each auxiliary head's molecules sit to the evaluation set.

The question this answers: head-search says some aux heads improve macro
RMSE (CYP1A2 by ~0.036 in the fixed 2026-09-22 run). Is that transfer, or is the aux data just carrying molecules that look
like the held-out molecules the RMSE is scored on?

head-search trains via `chemprop --splits-column split`, which is a *row-wise*
split: a held-out molecule is held out entirely, every label with it. So there
is no channel by which an eval molecule's own auxiliary label reaches the
encoder. What remains is proximity -- other molecules, near the eval ones,
that the aux head drags into training -- and it comes in two forms:

  1. Scaffold leakage. add-split-column builds scaffold groups over scored
     rows only (data_tools.split_column.assign_splits) and forces
     auxiliary-only molecules to train. So an aux-only molecule sharing a
     Bemis-Murcko scaffold with a test molecule lands in train anyway --
     the scaffold split protects the scored table and does nothing for the
     aux table. Reported as `scaffold_overlap_frac`.

  2. Nearest-neighbour proximity. For each eval molecule, the maximum ECFP4
     Tanimoto to any *training* molecule carrying this head's label. Max, not
     mean: leakage is a nearest-neighbour property. One close analogue of a
     test molecule is what lets the model predict it, and that analogue is
     invisible in a mean over every pair. Reported as `nn_tanimoto_*`, with
     the `frac_ge_*` tail columns counting eval molecules that have such an
     analogue at all.

`eval_in_aux_frac` is reported too, but it is composition, not leakage: the
share of eval molecules that also carry this head's label and were held out
along with it. It says how far the eval set is drawn from the same library as
the aux head, which is context for reading the two numbers above.

Set size is the obvious confounder: a head with 16k molecules covers chemical
space better than one with 800 for reasons that have nothing to do with the
isoform. `--match-n` repeats the nearest-neighbour measurement on random
subsamples of a common size so proximity can be read independently of volume.

Two reference rows are emitted alongside the candidates:
  __train_scored__  molecules already in train via a scored label (what the
                    baseline model saw) -- the floor any aux head must beat
  __train_all__     every training molecule, aux or not -- the ceiling

Dependencies are deliberately just the stdlib, numpy and rdkit. The SCC
compute nodes lack the AVX2/FMA/BMI CPU features the packaged polars wheel is
built against, so importing it there dies with SIGILL; for a table this size
polars bought nothing worth that risk.

Usage (SCC):
    python scripts/aux_similarity.py \
        --data-path data/union_train.csv \
        --results-path results/head_search.json \
        --out results/aux_similarity.csv

The proximity columns depend only on the union and its split; the gain columns
depend only on head-search. When head-search is re-run on the same union,
`--refresh-gains` re-joins the new gains onto an existing --out table without
the union or rdkit:
    python scripts/aux_similarity.py --refresh-gains \
        --results-path results/head_search.json --out results/aux_similarity.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_TANIMOTO_THRESHOLDS = (0.4, 0.7)

_SCORED_DEFAULT = [
    "CYP1A2_pIC50_direct_inhibition",
    "CYP2C9_pIC50_direct_inhibition",
    "CYP2D6_pIC50_direct_inhibition",
    "CYP3A4_pIC50_direct_inhibition",
]

# Mirrors models.head_search.discover_candidates. Duplicated rather than
# imported because that module imports polars at module scope, which is fatal
# on the SCC compute nodes (see module docstring). Keep the two in sync: if
# head-search changes how it groups columns into candidates, this analysis
# stops describing the run it claims to explain.
_NON_LABEL_COLUMNS = {"SMILES", "inchikey_block", "inchikey_full", "split", "PUBCHEM_CID"}


def read_table(path: Path) -> tuple[list[str], dict[str, list[str]]]:
    """Read a CSV into (column order, {column: values as strings})."""
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            raise SystemExit(f"{path} is empty")
        columns: dict[str, list[str]] = {name: [] for name in header}
        for row in reader:
            for name, value in zip(header, row):
                columns[name].append(value)
            for name in header[len(row):]:
                columns[name].append("")
    return header, columns


def _is_numeric(values: list[str]) -> bool:
    """True if every non-empty entry parses as a float (and at least one does)."""
    seen = False
    for value in values:
        if value == "":
            continue
        try:
            float(value)
        except ValueError:
            return False
        seen = True
    return seen


def discover_candidates(header: list[str], columns: dict[str, list[str]], scored_columns: list[str], granularity: str) -> dict[str, list[str]]:
    """Group auxiliary label columns into named candidates, as head-search does.

    Column names are `{protein}_{readout}_{source}`, so the leading token is
    the protein. 'protein' bundles every readout of one protein into a single
    candidate; 'source' keeps protein and source apart; 'readout' treats every
    column separately.
    """
    skip = set(scored_columns) | _NON_LABEL_COLUMNS
    aux = [c for c in header if c not in skip and _is_numeric(columns[c])]

    if granularity == "readout":
        return {c: [c] for c in aux}

    candidates: dict[str, list[str]] = {}
    for column in aux:
        parts = column.split("_")
        if granularity == "source" and len(parts) >= 3:
            name = f"{parts[0]}_{parts[-1]}"
        else:
            name = parts[0]
        candidates.setdefault(name, []).append(column)
    return candidates


def _rows_with_any(columns: dict[str, list[str]], names: list[str], n_rows: int) -> list[int]:
    """Row indices where at least one of `names` is non-empty."""
    return [i for i in range(n_rows) if any(columns[name][i] != "" for name in names)]


def _fingerprints(smiles: list[str]) -> tuple[list, list[int]]:
    """Return ECFP4 fingerprints and the row indices they came from, skipping unparseable SMILES."""
    from rdkit import RDLogger
    from rdkit.Chem import MolFromSmiles
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

    RDLogger.DisableLog("rdApp.*")
    morgan = GetMorganGenerator(radius=2, fpSize=2048)

    fps, kept = [], []
    for idx, smi in enumerate(smiles):
        mol = MolFromSmiles(smi) if smi else None
        if mol is None:
            continue
        fps.append(morgan.GetFingerprint(mol))
        kept.append(idx)

    n_bad = len(smiles) - len(kept)
    if n_bad:
        print(f"  {n_bad} SMILES failed to parse and were dropped", file=sys.stderr)
    return fps, kept


def _scaffolds(smiles: list[str]) -> list[str | None]:
    """Return the Bemis-Murcko scaffold SMILES for each input, None where parsing fails."""
    from rdkit import RDLogger
    from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

    RDLogger.DisableLog("rdApp.*")
    out: list[str | None] = []
    for smi in smiles:
        try:
            out.append(MurckoScaffoldSmiles(smi, includeChirality=False))
        except Exception:  # pylint: disable=broad-except
            out.append(None)
    return out


def _nn_similarity(eval_fps: list, ref_fps: list, exclude: list[set[int]] | None = None) -> np.ndarray:
    """Max Tanimoto from each eval fingerprint to the reference set.

    `exclude` gives, per eval molecule, reference positions to ignore. The
    split is row-wise, so an eval molecule is never in the training reference
    set and this is normally a no-op; it is kept as a guard for tables where
    the same structure survives under two rows. Returns 0.0 where the
    reference set is empty for that molecule.
    """
    from rdkit import DataStructs

    if not ref_fps:
        return np.zeros(len(eval_fps))

    out = np.zeros(len(eval_fps))
    for i, fp in enumerate(eval_fps):
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, ref_fps))
        if exclude is not None and exclude[i]:
            sims[list(exclude[i])] = -1.0
        out[i] = sims.max() if sims.size else 0.0
    return np.clip(out, 0.0, 1.0)


def _summarize(nn: np.ndarray, prefix: str = "nn_tanimoto") -> dict:
    """Reduce a nearest-neighbour similarity vector to the reported summary stats."""
    if nn.size == 0:
        return {f"{prefix}_mean": float("nan"), f"{prefix}_median": float("nan")}
    stats = {
        f"{prefix}_mean": float(nn.mean()),
        f"{prefix}_median": float(np.median(nn)),
    }
    for threshold in _TANIMOTO_THRESHOLDS:
        stats[f"{prefix}_frac_ge_{threshold}"] = float((nn >= threshold).mean())
    return stats


def load_gains(results_path: Path) -> tuple[float, dict[str, float]]:
    """Return (baseline macro RMSE, {candidate: solo macro RMSE}) from a head-search JSON.

    Only single-candidate configurations are used. Round 2+ entries measure a
    set, not a head, and their spread sits inside the seed noise anyway.
    """
    entries = json.loads(results_path.read_text())

    baseline = next((e["macro_rmse_mean"] for e in entries if not e["candidates"]), None)
    if baseline is None:
        raise SystemExit(f"{results_path} has no baseline entry (one with an empty 'candidates')")

    solo = {e["candidates"][0]: e["macro_rmse_mean"] for e in entries if len(e["candidates"]) == 1}
    print(f"{results_path}: baseline {baseline:.4f}, {len(solo)} solo heads", file=sys.stderr)
    return baseline, solo


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation between two equal-length vectors."""
    return float(np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1])


def main() -> None:
    """Entry point: join head-search gains to eval-set chemical proximity per aux head."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-path", type=Path, default=Path("data/union_train.csv"))
    parser.add_argument("--results-path", type=Path, default=Path("results/head_search.json"))
    parser.add_argument("--out", type=Path, default=Path("results/aux_similarity.csv"))
    parser.add_argument(
        "--eval-split",
        default="test",
        choices=["test", "val", "test+val"],
        help="Which split rows the RMSE is scored on (head-search reports test/)",
    )
    parser.add_argument(
        "--granularity",
        default="protein",
        choices=["protein", "source", "readout"],
        help="Must match the head-search run being explained",
    )
    parser.add_argument("--scored-columns", nargs="+", default=_SCORED_DEFAULT, help="Columns that count as scored labels")
    parser.add_argument(
        "--match-n",
        type=int,
        default=500,
        help="Subsample every aux set to this many molecules for the size-matched comparison (0 disables)",
    )
    parser.add_argument("--match-repeats", type=int, default=5, help="Subsample repeats for --match-n")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--refresh-gains",
        action="store_true",
        help="Only re-join gains from --results-path onto the existing --out table (no union, no rdkit)",
    )
    args = parser.parse_args()

    if args.refresh_gains:
        refresh_gains(args.out, args.results_path)
        return

    header, columns = read_table(args.data_path)
    if "split" not in columns:
        raise SystemExit(
            f"{args.data_path} has no 'split' column -- this analysis has to score proximity "
            "against the same held-out molecules head-search used. Run add-split-column first.",
        )
    if "SMILES" not in columns:
        raise SystemExit(f"{args.data_path} has no 'SMILES' column")

    split = columns["split"]
    n_rows = len(split)
    wanted = {"test", "val"} if args.eval_split == "test+val" else {args.eval_split}
    eval_rows = [i for i in range(n_rows) if split[i] in wanted]
    train_rows = [i for i in range(n_rows) if split[i] == "train"]
    print(f"{args.data_path}: {n_rows} rows, {len(train_rows)} train, {len(eval_rows)} eval ({args.eval_split})", file=sys.stderr)
    if not eval_rows:
        raise SystemExit(f"No rows with split in {sorted(wanted)}")

    candidates = discover_candidates(header, columns, args.scored_columns, args.granularity)
    baseline_rmse, solo_rmse = load_gains(args.results_path)

    unmatched = sorted(set(solo_rmse) - set(candidates))
    if unmatched:
        print(f"WARNING: in head-search but not in this table: {unmatched}", file=sys.stderr)

    smiles = columns["SMILES"]
    print("Fingerprinting...", file=sys.stderr)
    all_fps, fp_rows = _fingerprints(smiles)
    fp_of_row = dict(zip(fp_rows, all_fps))

    print("Computing scaffolds...", file=sys.stderr)
    scaffolds = _scaffolds(smiles)

    eval_fp_rows = [r for r in eval_rows if r in fp_of_row]
    eval_fps = [fp_of_row[r] for r in eval_fp_rows]
    eval_scaffolds = [scaffolds[r] for r in eval_fp_rows]
    n_eval = len(eval_fp_rows)

    # Reference sets: what the baseline model already had, and everything in train.
    scored_present = [c for c in args.scored_columns if c in columns]
    if not scored_present:
        raise SystemExit(f"None of the scored columns {args.scored_columns} are in {args.data_path}")
    scored_rows = set(_rows_with_any(columns, scored_present, n_rows))
    reference_sets = {
        "__train_scored__": [r for r in train_rows if r in scored_rows],
        "__train_all__": list(train_rows),
    }

    rng = np.random.default_rng(args.seed)
    records: list[dict] = []

    for name in list(reference_sets) + sorted(candidates):
        if name in reference_sets:
            member_rows = reference_sets[name]
            member_columns: list[str] = []
        else:
            member_columns = candidates[name]
            member_rows = _rows_with_any(columns, member_columns, n_rows)

        member_set = set(member_rows)
        train_members = [r for r in member_rows if split[r] == "train"]

        # Composition, not leakage: these rows are held out with their aux
        # labels, so this only says how far eval is drawn from the same library.
        eval_in_aux = sum(1 for r in eval_fp_rows if r in member_set)

        # Channel 1: scaffold shared with a training molecule carrying this head.
        train_scaffolds = {scaffolds[r] for r in train_members if scaffolds[r] is not None}
        scaffold_hits = sum(1 for s in eval_scaffolds if s is not None and s in train_scaffolds)

        # Channel 2: nearest-neighbour proximity over the head's training rows.
        ref_rows = [r for r in train_members if r in fp_of_row]
        ref_fps = [fp_of_row[r] for r in ref_rows]
        position_of_row = {r: i for i, r in enumerate(ref_rows)}
        exclude = [{position_of_row[r]} if r in position_of_row else set() for r in eval_fp_rows]
        nn = _nn_similarity(eval_fps, ref_fps, exclude)

        record = {
            "name": name,
            "n_columns": len(member_columns),
            "n_molecules": len(member_rows),
            "n_train_molecules": len(train_members),
            "eval_in_aux_frac": eval_in_aux / n_eval if n_eval else float("nan"),
            "scaffold_overlap_frac": scaffold_hits / n_eval if n_eval else float("nan"),
        }
        record.update(_summarize(nn))

        # Size-matched control, so proximity is readable independently of volume.
        if args.match_n and len(ref_fps) >= args.match_n:
            means = []
            for _ in range(args.match_repeats):
                pick = rng.choice(len(ref_fps), size=args.match_n, replace=False)
                picked_fps = [ref_fps[i] for i in pick]
                picked_exclude = [{j for j, p in enumerate(pick) if ref_rows[p] == r} for r in eval_fp_rows]
                means.append(_nn_similarity(eval_fps, picked_fps, picked_exclude).mean())
            record["nn_tanimoto_matched_mean"] = float(np.mean(means))
            record["nn_tanimoto_matched_sd"] = float(np.std(means))
        else:
            record["nn_tanimoto_matched_mean"] = float("nan")
            record["nn_tanimoto_matched_sd"] = float("nan")

        if name in solo_rmse:
            record["solo_macro_rmse"] = solo_rmse[name]
            record["gain"] = baseline_rmse - solo_rmse[name]
        else:
            record["solo_macro_rmse"] = float("nan")
            record["gain"] = float("nan")

        records.append(record)
        gain_text = f"{record['gain']:+.3f}" if name in solo_rmse else "    --"
        print(
            f"  {name:<18} n={len(member_rows):>6}  self={record['eval_in_aux_frac']:.2f}  "
            f"scaf={record['scaffold_overlap_frac']:.2f}  nn={record['nn_tanimoto_mean']:.3f}  "
            f"gain={gain_text}",
            file=sys.stderr,
        )

    write_records(records, args.out)
    report_correlations(records)


def write_records(records: list[dict], out: Path) -> None:
    """Sort by gain, best first (heads without one last), and write the table."""
    records.sort(key=lambda r: (np.isnan(r["gain"]), -r["gain"] if not np.isnan(r["gain"]) else 0.0))

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"\nWrote {out} ({len(records)} rows)", file=sys.stderr)


def refresh_gains(out: Path, results_path: Path) -> None:
    """Replace the gain columns of an existing table with those of a new head-search run."""
    _, columns = read_table(out)
    names = columns["name"]
    baseline_rmse, solo_rmse = load_gains(results_path)

    unmatched = sorted(set(solo_rmse) - set(names))
    if unmatched:
        print(f"WARNING: in head-search but not in {out}: {unmatched}", file=sys.stderr)

    records: list[dict] = []
    for i, name in enumerate(names):
        record: dict = {key: values[i] for key, values in columns.items()}
        if name in solo_rmse:
            record["solo_macro_rmse"] = solo_rmse[name]
            record["gain"] = baseline_rmse - solo_rmse[name]
        else:
            record["solo_macro_rmse"] = float("nan")
            record["gain"] = float("nan")
        records.append(record)

    write_records(records, out)
    report_correlations(records)


def report_correlations(records: list[dict]) -> None:
    """Print Spearman rho of each proximity column against gain."""
    scored = [r for r in records if not np.isnan(float(r["gain"]))]
    if len(scored) >= 3:
        print(f"\nSpearman rho vs. gain (n={len(scored)}):", file=sys.stderr)
        gains = np.array([float(r["gain"]) for r in scored])
        for column in ("eval_in_aux_frac", "scaffold_overlap_frac", "nn_tanimoto_mean", "nn_tanimoto_matched_mean", "n_molecules"):
            values = np.array([float(r[column]) for r in scored])
            ok = ~np.isnan(values)
            if ok.sum() < 3:
                print(f"  {column:<28} n/a  (fewer than 3 heads)", file=sys.stderr)
            elif np.std(values[ok]) == 0:
                # Happens at --granularity protein when several heads share one
                # source table and so cover exactly the same molecules. A rank
                # correlation against a constant is an artefact, not a result.
                print(f"  {column:<28} n/a  (constant across heads)", file=sys.stderr)
            else:
                print(f"  {column:<28} {_spearman(values[ok], gains[ok]):+.3f}  (n={ok.sum()})", file=sys.stderr)


if __name__ == "__main__":
    main()
