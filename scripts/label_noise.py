"""Score how noisy each auxiliary head's labels are, and whether noise tracks head-search gain.

The question: sample count, chemical proximity and enzyme similarity explain
little of which aux heads help (reports/head_search_analysis.md,
reports/protein_similarity_analysis.md). Section 4c of the head-search report
lists label quality as untested. If noisy heads help less, the noise measures
below should correlate negatively with solo gain.

Four families of measure, each per candidate at head-search's 'protein'
granularity:

  1. Replicate disagreement. Molecules measured more than once for the same
     gene by the same source. Reported as the pooled within-molecule SD,
     sqrt(sum (x - group mean)^2 / sum (n - 1)), over every replicated
     molecule. NULL (an empty cell) when the head has no replicated molecule:
     a head with no replicates has unknown noise, not zero noise.
       - ChEMBL and BindingDB: from the per-record files build-family-datasets
         writes (data/families/), before build-family-heads averages them.
         Filtered to the same affinity types as build-family-heads (IC50 by
         default). Records repeating an identical value (to 0.01 log units)
         within one source are dropped first by default: they are almost
         always one measurement deposited twice, and counting them would
         report agreement that was never measured.
       - AID 1851 qHTS: the same structure screened under more than one
         PubChem substance id. build-pubchem-aux keeps only the last of these
         rather than averaging, so the model trains on one draw from this
         spread.
     `rep_sd_pic50` pools every pIC50-scale source, and is the one replicate
     measure defined on the same scale for all 20 heads.

  2. ChEMBL vs BindingDB agreement. Molecules with a value from both sources
     for the same gene (each source averaged over its own replicates). Pairs
     whose values match to 0.01 are counted as mirrored -- BindingDB imports
     ChEMBL-curated data -- and excluded before the RMSD, bias and Spearman
     are computed. NULL below --min-pairs independent pairs.

  3. qHTS curve quality (the five AID 1851 isoforms only; NULL elsewhere),
     from the NCGC curve class: fraction inactive (class 4), and among actives
     the fraction with a complete curve (|class| 1.x), the median fit R^2,
     the fraction of fitted potencies extrapolated beyond the top tested
     concentration, and the fraction that are activators (positive class).

  4. Dynamic range, on the labels head-search actually trains on (train
     split of the union): per-column SD, 5th-95th percentile span, and
     `mode_frac`, the share of labels sitting on the single most common value.
     A large mode_frac is a pile-up at a censoring limit -- BindingDB's
     10 uM cutoff, for example, arrives as an exact 5.0 once its '>' is
     stripped. Per-column detail goes to --columns-out. AID 1851 potencies sit
     on a ~0.1 log-unit grid, so their mode_frac (~5-10%) is quantization, not
     censoring; compare it across qHTS columns, not against family heads.

Every measure is then correlated with solo gain from head-search (Spearman,
all heads and family tier only). With --similarity-path, a partial Spearman
controlling for log molecule count and nearest-neighbour Tanimoto is added:
noisy heads may simply be small or chemically distant, and without the
control this analysis would re-measure hypotheses 1 and 2.

Dependencies are the stdlib, numpy, scipy and rdkit, for the same SCC reason
as scripts/aux_similarity.py (polars dies with SIGILL on the compute nodes).

Usage (SCC, where data/families/ lives):
    python scripts/label_noise.py \
        --families-dir data/families \
        --pubchem-raw data/pubchem_aid1851_raw.csv \
        --data-path data/union_train.csv \
        --results-path results/head_search.json \
        --similarity-path results/aux_similarity.csv \
        --out results/label_noise.csv \
        --columns-out results/label_noise_columns.csv
"""

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aux_similarity import discover_candidates, load_gains, read_table  # noqa: E402

_SCORED_DEFAULT = [
    "CYP1A2_pIC50_direct_inhibition",
    "CYP2C9_pIC50_direct_inhibition",
    "CYP2D6_pIC50_direct_inhibition",
    "CYP3A4_pIC50_direct_inhibition",
]

# Mirrors data_tools.pubchem._PANEL_TO_ISOFORM. Not imported: that module is
# polars-free, but keeping the analysis self-describing matches aux_similarity.
_PANEL_TO_ISOFORM = {
    "p450-cyp1a2": "CYP1A2",
    "p450-cyp2c9": "CYP2C9",
    "p450-cyp2c19": "CYP2C19",
    "p450-cyp2d6": "CYP2D6",
    "p450-cyp3a4": "CYP3A4",
}
_PUBCHEM_PREAMBLE = {"RESULT_TYPE", "RESULT_DESCR", "RESULT_UNIT", "RESULT_ATTR_CONC_MICROMOL"}
_DUPLICATE_TOL = 0.005  # two values closer than this (log units) are the same number at 0.01 precision

# (column, direction a noise-hurts hypothesis predicts vs. gain)
_MEASURES = [
    ("rep_sd_pic50", "-"),
    ("rep_sd_chembl", "-"),
    ("rep_sd_bindingdb", "-"),
    ("rep_sd_qhts_pic50", "-"),
    ("rep_sd_qhts_pctinhib", "-"),
    ("rep_frac_range_gt_1", "-"),
    ("xsource_rmsd", "-"),
    ("xsource_spearman", "+"),
    ("qhts_inactive_frac", "-"),
    ("qhts_complete_curve_frac", "+"),
    ("qhts_median_r2", "+"),
    ("qhts_extrapolated_frac", "-"),
    ("label_sd_pic50", "+"),
    ("mode_frac_max", "-"),
]


class _KeyCache:
    """Memoized SMILES -> InChIKey connectivity block, the repo's join key."""

    def __init__(self) -> None:
        from data_tools.standardize import connectivity_key

        self._fn = connectivity_key
        self._cache: dict[str, str | None] = {}

    def __call__(self, smiles: str) -> str | None:
        if smiles not in self._cache:
            self._cache[smiles] = self._fn(smiles) if smiles else None
        return self._cache[smiles]


def _dedupe(values: list[float]) -> tuple[list[float], int]:
    """Drop values within _DUPLICATE_TOL of one already kept; return (kept, n dropped)."""
    kept: list[float] = []
    for value in values:
        if all(abs(value - k) >= _DUPLICATE_TOL for k in kept):
            kept.append(value)
    return kept, len(values) - len(kept)


def pooled_replicate_sd(groups: list[list[float]]) -> tuple[float | None, int]:
    """Pooled within-group SD over groups with >= 2 values; (None, 0) if there are none."""
    ss = 0.0
    dof = 0
    n_groups = 0
    for values in groups:
        if len(values) < 2:
            continue
        arr = np.asarray(values, dtype=float)
        ss += float(((arr - arr.mean()) ** 2).sum())
        dof += len(arr) - 1
        n_groups += 1
    if n_groups == 0:
        return None, 0
    return math.sqrt(ss / dof), n_groups


def _frac_range_gt(groups: list[list[float]], threshold: float) -> float | None:
    """Share of replicated molecules whose max-min spread exceeds threshold; None if none replicated."""
    spreads = [max(v) - min(v) for v in groups if len(v) >= 2]
    if not spreads:
        return None
    return sum(s > threshold for s in spreads) / len(spreads)


def load_family_records(
    families_dir: Path,
    genes: set[str],
    affinity_types: set[str],
    keep_exact_duplicates: bool,
    key_of: _KeyCache,
) -> tuple[dict[str, dict[str, dict[str, list[float]]]], dict[str, int]]:
    """Return ({gene: {source: {key: [values]}}}, {gene: exact duplicates dropped}) from build-family-datasets."""
    manifest = families_dir / "manifest.csv"
    if not manifest.exists():
        print(f"WARNING: no {manifest}; ChEMBL/BindingDB measures will be NULL", file=sys.stderr)
        return {}, {}

    with manifest.open(newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r.get("gene") in genes]

    records: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        path = families_dir / row["output_file"]
        if not path.exists():
            print(f"  warning: missing {path}, skipping", file=sys.stderr)
            continue
        with path.open(newline="") as handle:
            for record in csv.DictReader(handle):
                if (record.get("affinity_type") or "").strip().upper() not in affinity_types:
                    continue
                try:
                    value = float(record.get("pAffinity") or "")
                except ValueError:
                    continue
                key = key_of((record.get("SMILES") or "").strip())
                if key is None:
                    continue
                source = (record.get("source") or "unknown").strip()
                records[row["gene"]][source][key].append(value)

    dropped: dict[str, int] = defaultdict(int)
    if not keep_exact_duplicates:
        for gene, by_source in records.items():
            for by_key in by_source.values():
                for key, values in by_key.items():
                    by_key[key], n = _dedupe(values)
                    dropped[gene] += n
    return records, dropped


def load_qhts(raw_path: Path, key_of: _KeyCache) -> tuple[dict[str, dict], float]:
    """Parse AID 1851 into per-isoform replicate groups and curve-quality rows; also return top conc (uM)."""
    if not raw_path.exists():
        print(f"WARNING: no {raw_path}; qHTS measures will be NULL", file=sys.stderr)
        return {}, float("nan")

    per_iso: dict[str, dict] = {}
    with raw_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        concs = [float(c.split()[2]) for c in reader.fieldnames or [] if c.startswith("Activity at ")]
        top_conc = max(concs) if concs else float("nan")
        for record in reader:
            tag = (record.get("PUBCHEM_RESULT_TAG") or "").strip()
            if tag in _PUBCHEM_PREAMBLE or not tag.isdigit():
                continue
            isoform = _PANEL_TO_ISOFORM.get((record.get("Panel Name") or "").strip().lower())
            if isoform is None:
                continue
            key = key_of((record.get("PUBCHEM_EXT_DATASOURCE_SMILES") or "").strip())
            if key is None:
                continue
            entry = per_iso.setdefault(
                isoform,
                {"pic50": defaultdict(list), "pctinhib": defaultdict(list), "rows": []},
            )
            potency = _float(record.get("Potency"))
            if potency is not None and potency > 0:
                entry["pic50"][key].append(6.0 - math.log10(potency))
            max_response = _float(record.get("Max_Response"))
            if max_response is not None:
                entry["pctinhib"][key].append(-max_response)
            entry["rows"].append(
                (
                    _float(record.get("Fit_CurveClass")),
                    _float(record.get("Fit_R2")),
                    potency,
                    (record.get("PUBCHEM_ACTIVITY_OUTCOME") or "").strip(),
                )
            )
    return per_iso, top_conc


def _float(text: str | None) -> float | None:
    """Parse a numeric cell, None for blank or non-numeric."""
    try:
        return float((text or "").strip())
    except ValueError:
        return None


def qhts_curve_quality(rows: list[tuple], top_conc: float) -> dict[str, float | None]:
    """Curve-class summary for one isoform's AID 1851 rows (NCGC classes: 1 complete .. 4 inactive)."""
    classes = [c for c, _, _, _ in rows if c is not None]
    if not classes:
        return {}
    active = [c for c in classes if c != 4]
    fitted = [(r2, p) for c, r2, p, _ in rows if c is not None and c != 4 and p is not None]
    r2s = [r2 for r2, _ in fitted if r2 is not None]
    return {
        "qhts_inactive_frac": sum(c == 4 for c in classes) / len(classes),
        "qhts_complete_curve_frac": sum(1 <= abs(c) < 2 for c in active) / len(active) if active else None,
        "qhts_activator_frac": sum(c > 0 for c in active) / len(active) if active else None,
        "qhts_median_r2": float(np.median(r2s)) if r2s else None,
        "qhts_extrapolated_frac": sum(p > top_conc for _, p in fitted) / len(fitted) if fitted else None,
        "qhts_inconclusive_frac": sum(o == "Inconclusive" for _, _, _, o in rows) / len(rows),
    }


def cross_source(chembl: dict[str, list[float]], bindingdb: dict[str, list[float]], min_pairs: int) -> dict:
    """ChEMBL vs BindingDB agreement on shared molecules, mirrored (identical) pairs excluded."""
    shared = sorted(set(chembl) & set(bindingdb))
    diffs, xs, ys = [], [], []
    mirrored = 0
    for key in shared:
        c, b = float(np.mean(chembl[key])), float(np.mean(bindingdb[key]))
        if abs(c - b) < _DUPLICATE_TOL:
            mirrored += 1
            continue
        diffs.append(b - c)
        xs.append(c)
        ys.append(b)
    out: dict = {"xsource_n_shared": len(shared), "xsource_n_mirrored": mirrored, "xsource_n_pairs": len(diffs)}
    if len(diffs) < min_pairs:
        out.update({"xsource_rmsd": None, "xsource_bias": None, "xsource_spearman": None})
        return out
    arr = np.asarray(diffs)
    out["xsource_rmsd"] = float(np.sqrt((arr**2).mean()))
    out["xsource_bias"] = float(arr.mean())
    out["xsource_spearman"] = _spearman(np.asarray(xs), np.asarray(ys))[0]
    return out


def column_ranges(columns: dict[str, list[str]], names: list[str], rows: list[int]) -> list[dict]:
    """Dynamic-range stats for each label column over the given rows."""
    out = []
    for name in names:
        values = np.array([float(columns[name][i]) for i in rows if columns[name][i] != ""])
        if values.size == 0:
            continue
        rounded, counts = np.unique(np.round(values, 2), return_counts=True)
        top = int(counts.argmax())
        p5, p95 = np.percentile(values, [5, 95])
        out.append(
            {
                "column": name,
                "n": int(values.size),
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)) if values.size > 1 else None,
                "p5": float(p5),
                "p95": float(p95),
                "span_5_95": float(p95 - p5),
                "mode_value": float(rounded[top]),
                "mode_frac": float(counts[top] / values.size),
            }
        )
    return out


def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Spearman rho and two-sided p."""
    from scipy.stats import spearmanr

    result = spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def _ranks(x: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    return rankdata(x)


def partial_spearman(x: np.ndarray, y: np.ndarray, controls: np.ndarray) -> tuple[float, float]:
    """Spearman of x and y with the ranks of `controls` (n x k) regressed out of both; (rho, p)."""
    from scipy.stats import t as t_dist

    design = np.column_stack([np.ones(len(x))] + [_ranks(c) for c in controls.T])
    resid = []
    for v in (x, y):
        r = _ranks(v)
        beta, *_ = np.linalg.lstsq(design, r, rcond=None)
        resid.append(r - design @ beta)
    rho = float(np.corrcoef(resid[0], resid[1])[0, 1])
    dof = len(x) - 2 - controls.shape[1]
    if dof <= 0 or abs(rho) >= 1:
        return rho, float("nan")
    t_stat = rho * math.sqrt(dof / (1 - rho**2))
    return rho, float(2 * t_dist.sf(abs(t_stat), dof))


def report_correlations(records: list[dict], similarity: dict[str, dict]) -> None:
    """Print Spearman (all, family-only) and, with similarity data, partial Spearman of each measure vs gain."""
    scored = [r for r in records if r.get("gain") is not None]
    print(f"\nSpearman rho vs. solo gain (expected sign if noise hurts in brackets):", file=sys.stderr)
    header = f"  {'measure':<30} {'all':>20}  {'family only':>20}"
    if similarity:
        header += f"  {'partial (log n, NN Tanimoto)':>30}"
    print(header, file=sys.stderr)

    for column, sign in _MEASURES:
        cells = []
        for subset in (scored, [r for r in scored if r["tier"] == "family"]):
            pairs = [(r[column], r["gain"]) for r in subset if r.get(column) is not None]
            cells.append(_format_rho(pairs))
        line = f"  {column + ' (' + sign + ')':<30} {cells[0]:>20}  {cells[1]:>20}"

        if similarity:
            rows = [
                r
                for r in scored
                if r.get(column) is not None
                and r["name"] in similarity
                and not math.isnan(similarity[r["name"]]["nn_tanimoto_mean"])
            ]
            if len(rows) >= 6:
                x = np.array([r[column] for r in rows])
                y = np.array([r["gain"] for r in rows])
                controls = np.array(
                    [
                        [math.log10(similarity[r["name"]]["n_molecules"]), similarity[r["name"]]["nn_tanimoto_mean"]]
                        for r in rows
                    ]
                )
                if np.std(x) > 0:
                    rho, p = partial_spearman(x, y, controls)
                    line += f"  {f'{rho:+.2f} (p={p:.3f}, n={len(rows)})':>30}"
                else:
                    line += f"  {'n/a (constant)':>30}"
            else:
                line += f"  {'n/a (< 6 heads)':>30}"
        print(line, file=sys.stderr)


def _format_rho(pairs: list[tuple[float, float]]) -> str:
    if len(pairs) < 3:
        return "n/a (< 3 heads)"
    x = np.array([p[0] for p in pairs])
    y = np.array([p[1] for p in pairs])
    if np.std(x) == 0:
        return "n/a (constant)"
    rho, p = _spearman(x, y)
    return f"{rho:+.2f} (p={p:.3f}, n={len(pairs)})"


def load_similarity(path: Path | None) -> dict[str, dict]:
    """{head: {n_molecules, nn_tanimoto_mean}} from aux_similarity.py output, or {} if absent."""
    if path is None:
        return {}
    if not path.exists():
        print(f"WARNING: {path} not found; skipping partial correlations", file=sys.stderr)
        return {}
    out = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                out[row["name"]] = {
                    "n_molecules": float(row["n_molecules"]),
                    "nn_tanimoto_mean": float(row["nn_tanimoto_mean"]),
                }
            except (KeyError, ValueError):
                continue
    return out


def _write(path: Path, rows: list[dict]) -> None:
    """Write dict rows; None becomes an empty (NULL) cell."""
    fieldnames: list[str] = []
    for row in rows:
        fieldnames += [k for k in row if k not in fieldnames]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if row.get(k) is None else row[k] for k in fieldnames})
    print(f"Wrote {path} ({len(rows)} rows)", file=sys.stderr)


def main() -> None:
    """Entry point: score label noise and dynamic range per aux head and correlate with gain."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--families-dir", type=Path, default=Path("data/families"))
    parser.add_argument("--pubchem-raw", type=Path, default=Path("data/pubchem_aid1851_raw.csv"))
    parser.add_argument("--data-path", type=Path, default=Path("data/union_train.csv"))
    parser.add_argument("--results-path", type=Path, default=Path("results/head_search.json"))
    parser.add_argument(
        "--similarity-path",
        type=Path,
        default=None,
        help="aux_similarity.py output; enables partial Spearman controlling for size and proximity",
    )
    parser.add_argument("--out", type=Path, default=Path("results/label_noise.csv"))
    parser.add_argument("--columns-out", type=Path, default=Path("results/label_noise_columns.csv"))
    parser.add_argument("--scored-columns", nargs="+", default=_SCORED_DEFAULT)
    parser.add_argument("--granularity", default="protein", choices=["protein", "source", "readout"])
    parser.add_argument(
        "--affinity-types",
        nargs="+",
        default=["IC50"],
        help="Family record types to score; match the build-family-heads run (default: IC50)",
    )
    parser.add_argument(
        "--keep-exact-duplicates",
        action="store_true",
        help="Count identical repeated values within a source as replicates (default: drop them)",
    )
    parser.add_argument(
        "--min-pairs", type=int, default=10, help="Min non-mirrored ChEMBL/BindingDB pairs (default 10)"
    )
    args = parser.parse_args()

    header, columns = read_table(args.data_path)
    if "split" not in columns:
        raise SystemExit(f"{args.data_path} has no 'split' column; run add-split-column first")
    train_rows = [i for i, s in enumerate(columns["split"]) if s == "train"]
    candidates = discover_candidates(header, columns, args.scored_columns, args.granularity)
    baseline_rmse, solo_rmse = load_gains(args.results_path)
    missing = sorted(set(solo_rmse) - set(candidates))
    if missing:
        print(f"WARNING: in head-search but not in {args.data_path}: {missing}", file=sys.stderr)

    genes = {name.split("_")[0] for name in candidates}
    key_of = _KeyCache()
    print("Reading family records...", file=sys.stderr)
    family, dropped = load_family_records(
        args.families_dir, genes, {t.upper() for t in args.affinity_types}, args.keep_exact_duplicates, key_of
    )
    print("Reading AID 1851...", file=sys.stderr)
    qhts, top_conc = load_qhts(args.pubchem_raw, key_of)
    if not math.isnan(top_conc):
        print(f"  top tested concentration {top_conc} uM (pIC50 floor {6 - math.log10(top_conc):.2f})", file=sys.stderr)

    records: list[dict] = []
    column_rows: list[dict] = []
    for name in sorted(candidates):
        gene = name.split("_")[0]
        cols = candidates[name]
        by_source = family.get(gene, {})
        chembl = by_source.get("chembl", {})
        bindingdb = by_source.get("bindingdb", {})
        iso = qhts.get(gene) if any(c.endswith("_aid1851") for c in cols) else None

        chembl_groups = list(chembl.values())
        bindingdb_groups = list(bindingdb.values())
        qhts_pic50_groups = list(iso["pic50"].values()) if iso else []
        qhts_pct_groups = list(iso["pctinhib"].values()) if iso else []
        pic50_groups = chembl_groups + bindingdb_groups + qhts_pic50_groups

        record: dict = {"name": name, "tier": "qHTS" if iso else "family"}
        record["gain"] = baseline_rmse - solo_rmse[name] if name in solo_rmse else None
        for label, groups in (
            ("pic50", pic50_groups),
            ("chembl", chembl_groups),
            ("bindingdb", bindingdb_groups),
            ("qhts_pic50", qhts_pic50_groups),
            ("qhts_pctinhib", qhts_pct_groups),
        ):
            sd, n_rep = pooled_replicate_sd(groups)
            record[f"rep_sd_{label}"] = sd
            record[f"rep_n_{label}"] = n_rep
        record["rep_frac_range_gt_1"] = _frac_range_gt(pic50_groups, 1.0)
        record["exact_duplicates_dropped"] = dropped.get(gene, 0)
        record.update(cross_source(chembl, bindingdb, args.min_pairs))
        if iso:
            record.update(qhts_curve_quality(iso["rows"], top_conc))

        ranges = column_ranges(columns, cols, train_rows)
        for r in ranges:
            column_rows.append({"name": name, **r})
        pic50_ranges = [r for r in ranges if "_pIC50_" in r["column"] and r["sd"] is not None]
        n_pic50 = sum(r["n"] for r in pic50_ranges)
        record["label_sd_pic50"] = sum(r["sd"] * r["n"] for r in pic50_ranges) / n_pic50 if n_pic50 else None
        record["mode_frac_max"] = max((r["mode_frac"] for r in ranges), default=None)
        records.append(record)

        gain_text = f"{record['gain']:+.3f}" if record["gain"] is not None else "   --"
        rep_text = f"{record['rep_sd_pic50']:.2f}" if record["rep_sd_pic50"] is not None else "NULL"
        x_text = f"{record['xsource_rmsd']:.2f}" if record["xsource_rmsd"] is not None else "NULL"
        print(
            f"  {name:<10} {record['tier']:<6} rep_sd={rep_text:>5} (n={record['rep_n_pic50']:>5})  "
            f"xsource_rmsd={x_text:>5} (pairs={record['xsource_n_pairs']:>5})  gain={gain_text}",
            file=sys.stderr,
        )

    records.sort(key=lambda r: (r["gain"] is None, -(r["gain"] or 0.0)))
    _write(args.out, records)
    _write(args.columns_out, column_rows)
    report_correlations(records, load_similarity(args.similarity_path))


if __name__ == "__main__":
    main()
