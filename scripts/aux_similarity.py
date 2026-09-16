"""Measure how close each auxiliary head's molecules sit to the evaluation set.

The question this answers: head-search says one aux head is worth ~0.21 macro
RMSE. Is that transfer, or is the aux data just carrying molecules that look
like the held-out molecules the RMSE is scored on?

Three distinct channels can produce a gain that is not transfer, and they are
not the same thing, so this reports all three separately:

  1. Self-supervision. The union table is one row per molecule. A held-out
     molecule whose *scored* label is hidden can still appear in training
     through an auxiliary column on the same row. The encoder sees that exact
     structure, with a correlated label, every epoch. Reported as
     `eval_in_aux_frac`.

  2. Scaffold leakage. add-split-column builds scaffold groups over scored
     rows only (data_tools.split_column.assign_splits) and forces
     auxiliary-only molecules to train. So an aux-only molecule sharing a
     Bemis-Murcko scaffold with a test molecule lands in train anyway --
     the scaffold split does not protect against the aux table. Reported as
     `scaffold_overlap_frac`.

  3. Plain nearest-neighbour proximity. For each eval molecule, the maximum
     ECFP4 Tanimoto to any *training* molecule carrying this head's label,
     with the eval molecule itself excluded so channel 1 does not leak into
     this number. Reported as `nn_tanimoto_*`.

Set size is the obvious confounder: a head with 16k molecules covers chemical
space better than one with 800 for reasons that have nothing to do with the
isoform. `--match-n` repeats the nearest-neighbour measurement on random
subsamples of a common size so proximity can be read independently of volume.

Two reference rows are emitted alongside the candidates:
  __train_scored__  molecules already in train via a scored label (what the
                    baseline model saw) -- the floor any aux head must beat
  __train_all__     every training molecule, aux or not -- the ceiling

Usage (SCC):
    python scripts/aux_similarity.py \
        --data-path data/union_train.csv \
        --results-path results/head_search.json \
        --out results/aux_similarity.csv
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from models.head_search import _SCORED_DEFAULT, discover_candidates  # noqa: E402

_TANIMOTO_THRESHOLDS = (0.4, 0.7)


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

    `exclude` gives, per eval molecule, reference positions to ignore -- used to
    drop the eval molecule's own row so self-supervision does not inflate
    what is meant to be a neighbour distance. Returns 0.0 where the reference
    set is empty for that molecule.
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
        raise ValueError(f"{results_path} has no baseline entry (one with an empty 'candidates')")

    solo = {e["candidates"][0]: e["macro_rmse_mean"] for e in entries if len(e["candidates"]) == 1}
    print(f"{results_path}: baseline {baseline:.4f}, {len(solo)} solo heads", file=sys.stderr)
    return baseline, solo


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
    parser.add_argument(
        "--scored-columns",
        nargs="+",
        default=_SCORED_DEFAULT,
        help="Columns that count as scored labels",
    )
    parser.add_argument(
        "--match-n",
        type=int,
        default=500,
        help="Subsample every aux set to this many molecules for the size-matched comparison (0 disables)",
    )
    parser.add_argument("--match-repeats", type=int, default=5, help="Subsample repeats for --match-n")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    df = pl.read_csv(args.data_path, infer_schema_length=None)
    if "split" not in df.columns:
        raise SystemExit(
            f"{args.data_path} has no 'split' column -- this analysis has to score proximity "
            "against the same held-out molecules head-search used. Run add-split-column first.",
        )

    wanted = {"test", "val"} if args.eval_split == "test+val" else {args.eval_split}
    split = df["split"].to_list()
    eval_rows = [i for i, s in enumerate(split) if s in wanted]
    train_rows = [i for i, s in enumerate(split) if s == "train"]
    print(f"{args.data_path}: {len(df)} rows, {len(train_rows)} train, {len(eval_rows)} eval ({args.eval_split})", file=sys.stderr)
    if not eval_rows:
        raise SystemExit(f"No rows with split in {sorted(wanted)}")

    candidates = discover_candidates(df, args.scored_columns, args.granularity)
    baseline_rmse, solo_rmse = load_gains(args.results_path)

    unmatched = sorted(set(solo_rmse) - set(candidates))
    if unmatched:
        print(f"WARNING: in head-search but not in this table: {unmatched}", file=sys.stderr)

    smiles = df["SMILES"].to_list()
    print("Fingerprinting...", file=sys.stderr)
    all_fps, fp_rows = _fingerprints(smiles)
    fp_of_row = {row: fp for row, fp in zip(fp_rows, all_fps)}

    print("Computing scaffolds...", file=sys.stderr)
    scaffolds = _scaffolds(smiles)

    eval_fp_rows = [r for r in eval_rows if r in fp_of_row]
    eval_fps = [fp_of_row[r] for r in eval_fp_rows]
    eval_scaffolds = [scaffolds[r] for r in eval_fp_rows]
    n_eval = len(eval_fp_rows)

    # Reference sets: what the baseline model already had, and everything in train.
    scored_present = [c for c in args.scored_columns if c in df.columns]
    has_scored = df.select(pl.any_horizontal([pl.col(c).is_not_null() for c in scored_present]))[:, 0].to_list()
    reference_sets = {
        "__train_scored__": [r for r in train_rows if has_scored[r]],
        "__train_all__": list(train_rows),
    }

    rng = np.random.default_rng(args.seed)
    records: list[dict] = []

    for name in list(reference_sets) + sorted(candidates):
        if name in reference_sets:
            member_rows = reference_sets[name]
            columns: list[str] = []
        else:
            columns = candidates[name]
            mask = df.select(pl.any_horizontal([pl.col(c).is_not_null() for c in columns]))[:, 0].to_list()
            member_rows = [i for i, flag in enumerate(mask) if flag]

        member_set = set(member_rows)
        train_members = [r for r in member_rows if split[r] == "train"]

        # Channel 1: eval molecules that are themselves supervised by this head.
        eval_in_aux = sum(1 for r in eval_fp_rows if r in member_set)

        # Channel 2: scaffold shared with a training molecule carrying this head.
        train_scaffolds = {scaffolds[r] for r in train_members if scaffolds[r] is not None}
        scaffold_hits = sum(1 for s in eval_scaffolds if s is not None and s in train_scaffolds)

        # Channel 3: nearest-neighbour proximity, self-matches removed.
        ref_rows = [r for r in train_members if r in fp_of_row]
        ref_fps = [fp_of_row[r] for r in ref_rows]
        position_of_row = {r: i for i, r in enumerate(ref_rows)}
        exclude = [{position_of_row[r]} if r in position_of_row else set() for r in eval_fp_rows]
        nn = _nn_similarity(eval_fps, ref_fps, exclude)

        record = {
            "name": name,
            "n_columns": len(columns),
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

    out = pl.DataFrame(records).sort("gain", descending=True, nulls_last=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(args.out)
    print(f"\nWrote {args.out} ({len(out)} rows)", file=sys.stderr)

    scored = out.filter(pl.col("gain").is_not_nan() & pl.col("gain").is_not_null())
    if len(scored) >= 3:
        print(f"\nSpearman rho vs. gain (n={len(scored)}):", file=sys.stderr)
        gains = scored["gain"].to_numpy()
        for column in ("eval_in_aux_frac", "scaffold_overlap_frac", "nn_tanimoto_mean", "nn_tanimoto_matched_mean", "n_molecules"):
            values = scored[column].to_numpy().astype(float)
            ok = ~np.isnan(values)
            if ok.sum() < 3:
                print(f"  {column:<28} n/a  (fewer than 3 heads)", file=sys.stderr)
                continue
            if np.std(values[ok]) == 0:
                # Happens at --granularity protein when several heads share one
                # source table and so cover exactly the same molecules. A rank
                # correlation against a constant is an artefact, not a result.
                print(f"  {column:<28} n/a  (constant across heads)", file=sys.stderr)
                continue
            ranked_x = np.argsort(np.argsort(values[ok]))
            ranked_y = np.argsort(np.argsort(gains[ok]))
            rho = np.corrcoef(ranked_x, ranked_y)[0, 1]
            print(f"  {column:<28} {rho:+.3f}  (n={ok.sum()})", file=sys.stderr)


if __name__ == "__main__":
    main()
