"""Print non-null example counts per isoform/head/data-type in a union table.

Answers "how many examples do we have, of what type, for each isoform" --
breaks every column in data/union_train.csv down into (gene, data type,
source) and reports non-null row counts, so head-search results can be read
against the actual data volume behind each head.

Usage:
    python scripts/data_counts.py [--data-path data/union_train.csv]
"""

import argparse
from pathlib import Path

import polars as pl

_SUFFIXES = [
    ("_pIC50_direct_inhibition", "pIC50", "direct_inhibition (challenge)"),
    ("_pctinhib_aid1851", "pctinhib", "aid1851 (PubChem qHTS)"),
    ("_pIC50_aid1851", "pIC50", "aid1851 (PubChem qHTS)"),
    ("_pIC50_chembl", "pIC50", "chembl (family)"),
    ("_pIC50_bindingdb", "pIC50", "bindingdb (family)"),
]


def _classify(col: str) -> tuple[str, str, str] | None:
    """Split a column name into (gene, readout, source_label), longest suffix first."""
    for suffix, readout, source_label in sorted(_SUFFIXES, key=lambda s: -len(s[0])):
        if col.endswith(suffix):
            return col[: -len(suffix)], readout, source_label
    return None


def main() -> None:
    """Entry point: load the union table and print per-gene/readout/source counts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=Path("data/union_train.csv"))
    args = parser.parse_args()

    df = pl.read_csv(args.data_path, infer_schema_length=None)
    print(f"{args.data_path}: {len(df)} total rows\n")

    rows: list[tuple[str, str, str, int]] = []
    for col in df.columns:
        if col in ("SMILES", "split"):
            continue
        parsed = _classify(col)
        if parsed is None:
            continue
        gene, readout, source_label = parsed
        n = df[col].drop_nulls().len()
        rows.append((gene, readout, source_label, n))

    rows.sort(key=lambda r: (r[0], r[2], r[1]))

    gene_w = max((len(r[0]) for r in rows), default=4)
    readout_w = max((len(r[1]) for r in rows), default=7)
    source_w = max((len(r[2]) for r in rows), default=6)

    header = f"{'gene':<{gene_w}}  {'readout':<{readout_w}}  {'source':<{source_w}}  count"
    print(header)
    print("-" * len(header))
    current_gene = None
    gene_total = 0
    for gene, readout, source_label, n in rows:
        if current_gene is not None and gene != current_gene:
            print(f"{'':<{gene_w}}  {'':<{readout_w}}  {'TOTAL':<{source_w}}  {gene_total}")
            print()
            gene_total = 0
        current_gene = gene
        gene_total += n
        print(f"{gene:<{gene_w}}  {readout:<{readout_w}}  {source_label:<{source_w}}  {n}")
    if current_gene is not None:
        print(f"{'':<{gene_w}}  {'':<{readout_w}}  {'TOTAL':<{source_w}}  {gene_total}")


if __name__ == "__main__":
    main()
