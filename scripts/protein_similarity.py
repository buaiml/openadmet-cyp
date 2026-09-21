"""Test whether protein similarity between isoforms explains auxiliary-head gain.

`reports/head_search_analysis.md` left the split unexplained: some aux heads buy
~0.2 macro RMSE and others buy nothing, and row count does not predict which.
`chemspace.py` attacks the ligand side of that question. This one attacks the
protein side: if an aux head helps because its enzyme is *like* the scored
enzymes -- same fold, same substrate-binding residues, therefore transferable
structure-activity signal -- then gain should rise with similarity to
CYP{1A2,2C9,2D6,3A4}.

Two independent similarity axes, because they disagree across the P450 family:

  sequence    Needleman-Wunsch global alignment (BLOSUM62, -11/-1) of the
              canonical UniProt sequences, percent identity over the shorter
              sequence. Sensitive to the residue-level differences that set
              substrate specificity apart within a subfamily.
  structure   TM-score from TM-align over the AlphaFold DB models. Fold-level,
              length-normalised, and nearly saturated inside P450s -- every
              member shares the same triangular-prism fold, so the dynamic
              range here is small by construction and a flat trend on this
              axis is weaker evidence than a flat trend on sequence.

Both are measured against each of the four scored isoforms separately, then
aggregated per aux head as the mean over the four (the metric is a macro
average over those same four tasks) and as the max (a head could help by being
close to just one). Self-pairs are excluded from the aggregate: an aux head for
a scored isoform is 100% identical to itself, which is a statement about the
experiment design, not about transfer. Those four heads are drawn as stars, as
in `plot_volume_vs_gain.py`, and are confounded anyway -- they carry a
supervision channel no external head has.

Gains are read from `results/head_search_results.csv`: solo-head configs only
(a multi-head config measures a set, not a head), `baseline - config`, so
positive means the head helped. Per-task gains come from the same rows.

Usage:
    python scripts/protein_similarity.py
    python scripts/protein_similarity.py --results-path results/head_search_results.csv

Network: UniProt REST (sequences) and AlphaFold DB (structures), both cached
under --cache-dir, so a second run is offline. Requires biopython and tmtools
on top of the project requirements.
"""

import argparse
import json
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from Bio.Align import PairwiseAligner, substitution_matrices
from scipy.stats import spearmanr
from tmtools import tm_align

_UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
_AFDB_PREDICTION_URL = "https://alphafold.ebi.ac.uk/api/prediction"
_AFDB_FILE_URL = "https://alphafold.ebi.ac.uk/files"
_TIMEOUT_S = 60

_SCORED = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
_QHTS = {"CYP1A2", "CYP2C9", "CYP2C19", "CYP2D6", "CYP3A4"}
_COLORS = {"qHTS": "#1f77b4", "family": "#ff7f0e"}

# Three-letter -> one-letter, for reading a sequence back off a PDB model.
_AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}


@dataclass
class Protein:
    """One isoform: its UniProt sequence and its AlphaFold CA trace."""

    gene: str
    accession: str
    sequence: str
    ca_coords: np.ndarray
    ca_sequence: str


# --------------------------------------------------------------------------
# fetching
# --------------------------------------------------------------------------


def _http_get(url: str) -> bytes:
    """GET url, returning the raw body."""
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:
        return response.read()


def fetch_uniprot(genes: list[str], cache_dir: Path) -> dict[str, tuple[str, str]]:
    """Return {gene: (accession, sequence)} for reviewed human entries, cached as JSON.

    Queried by exact gene name rather than hardcoded accessions so a renamed or
    merged entry fails loudly instead of silently aligning the wrong protein.
    """
    cache = cache_dir / "uniprot_human_p450.json"
    known: dict[str, tuple[str, str]] = {}
    if cache.exists():
        known = {g: tuple(v) for g, v in json.loads(cache.read_text()).items()}

    for gene in genes:
        if gene in known:
            continue
        query = f"gene_exact:{gene} AND organism_id:9606 AND reviewed:true"
        url = f"{_UNIPROT_SEARCH_URL}?query={urllib.parse.quote(query)}&fields=accession,sequence&format=json&size=5"
        results = json.loads(_http_get(url)).get("results", [])
        if not results:
            raise SystemExit(f"UniProt has no reviewed human entry for gene {gene}")
        entry = results[0]
        known[gene] = (entry["primaryAccession"], entry["sequence"]["value"])
        print(f"  {gene}: {known[gene][0]} ({len(known[gene][1])} aa)", file=sys.stderr)

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({g: list(v) for g, v in known.items()}, indent=2))
    return {g: known[g] for g in genes}


def fetch_structure(accession: str, cache_dir: Path) -> Path:
    """Return a local path to the AlphaFold DB model for accession, downloading once.

    The file version is read from the prediction API rather than assumed: AFDB
    retires old `model_vN.pdb` URLs when it re-releases, so a pinned version
    turns into a 404 on the next rebuild.
    """
    path = cache_dir / f"AF-{accession}.pdb"
    if path.exists():
        return path

    meta = json.loads(_http_get(f"{_AFDB_PREDICTION_URL}/{accession}"))
    version = meta[0]["latestVersion"]
    url = f"{_AFDB_FILE_URL}/AF-{accession}-F1-model_v{version}.pdb"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_http_get(url))
    print(f"  {accession}: AlphaFold model v{version}", file=sys.stderr)
    return path


def read_ca_trace(path: Path) -> tuple[np.ndarray, str]:
    """Return (N x 3 CA coordinates, one-letter sequence) from a single-chain PDB."""
    coords: list[list[float]] = []
    residues: list[str] = []
    for line in path.read_text().splitlines():
        if not line.startswith("ATOM") or line[12:16].strip() != "CA":
            continue
        resname = line[17:20].strip()
        if resname not in _AA3TO1:
            continue
        coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        residues.append(_AA3TO1[resname])
    return np.array(coords), "".join(residues)


# --------------------------------------------------------------------------
# similarity
# --------------------------------------------------------------------------


def _aligner() -> PairwiseAligner:
    """Global BLOSUM62 aligner with standard affine gap penalties."""
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -11
    aligner.extend_gap_score = -1
    # End gaps unpenalised: several P450s differ only by a longer N-terminal
    # membrane anchor, which is not a specificity difference.
    aligner.end_gap_score = 0.0
    return aligner


def sequence_identity(seq_a: str, seq_b: str, aligner: PairwiseAligner) -> float:
    """Percent identity of the best global alignment, normalised by the shorter sequence."""
    alignment = aligner.align(seq_a, seq_b)[0]
    a, b = alignment[0], alignment[1]
    identities = sum(1 for x, y in zip(a, b) if x == y and x != "-")
    return 100.0 * identities / min(len(seq_a), len(seq_b))


def tm_score(a: Protein, b: Protein) -> float:
    """Mean of the two TM-align normalisations between two structures.

    TM-score is asymmetric (normalised by either chain's length). P450 lengths
    differ by at most ~15%, so the two values are close; the mean avoids
    picking a direction and is symmetric, which the plot needs.
    """
    result = tm_align(a.ca_coords, b.ca_coords, a.ca_sequence, b.ca_sequence)
    return 0.5 * (result.tm_norm_chain1 + result.tm_norm_chain2)


# --------------------------------------------------------------------------
# gains
# --------------------------------------------------------------------------


def load_gains(results_path: Path) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Return ({gene: macro gain}, {gene: {scored isoform: gain}}) from head-search.

    Solo configs only. Gain is `baseline - config`, so positive means the head
    helped.
    """
    rows = [line.split(",") for line in results_path.read_text().strip().splitlines()]
    header, body = rows[0], rows[1:]
    index = {name: i for i, name in enumerate(header)}

    baseline = next((r for r in body if r[0] == "baseline"), None)
    if baseline is None:
        raise SystemExit(f"{results_path} has no 'baseline' config row")

    macro: dict[str, float] = {}
    per_task: dict[str, dict[str, float]] = {}
    for row in body:
        config = row[0]
        if config == "baseline" or "+" in config:
            continue
        macro[config] = float(baseline[index["macroRMSE"]]) - float(row[index["macroRMSE"]])
        per_task[config] = {
            task: float(baseline[index[task]]) - float(row[index[task]])
            for task in _SCORED
            if task in index
        }

    print(
        f"{results_path}: baseline macro RMSE {float(baseline[index['macroRMSE']]):.4f}, "
        f"{len(macro)} solo heads",
        file=sys.stderr,
    )
    return macro, per_task


# --------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------


def _annotate_correlation(
    ax: plt.Axes, x: list[float], y: list[float], label: str, xy: tuple[float, float]
) -> None:
    """Write a Spearman rho + p for one subset at an empty spot in the axes."""
    if len(x) < 3:
        return
    rho, p = spearmanr(x, y)
    ax.text(
        xy[0],
        xy[1],
        f"{label}: rho = {rho:+.2f}, p = {p:.3f}, n = {len(x)}",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        color="#333333",
    )


def plot_axis(
    axis_name: str,
    unit: str,
    similarity: dict[str, dict[str, float]],
    macro: dict[str, float],
    per_task: dict[str, dict[str, float]],
    out_path: Path,
) -> None:
    """Render the two-panel gain-vs-similarity figure for one similarity axis."""
    genes = sorted(macro, key=lambda g: -macro[g])
    fig, (ax_macro, ax_task) = plt.subplots(1, 2, figsize=(13.5, 5.8))

    # Panel 1: one point per aux head, similarity averaged over the four scored
    # isoforms (self-pairs dropped -- see module docstring).
    subsets: dict[str, tuple[list[float], list[float]]] = {"qHTS": ([], []), "family": ([], [])}
    for position, gene in enumerate(genes):
        others = [similarity[gene][s] for s in _SCORED if s != gene]
        x, y = float(np.mean(others)), macro[gene]
        tier = "qHTS" if gene in _QHTS else "family"
        subsets[tier][0].append(x)
        subsets[tier][1].append(y)
        self_aux = gene in _SCORED
        ax_macro.scatter(
            x,
            y,
            color=_COLORS[tier],
            marker="*" if self_aux else "o",
            s=170 if self_aux else 55,
            zorder=4 if self_aux else 3,
            edgecolor="black" if self_aux else "white",
            linewidth=0.6,
            label=f"{tier}, self-aux" if self_aux else tier,
        )
        # The inert heads pile up on the y = 0 line; cycling the label corner
        # keeps that cluster readable.
        offset = [(6, 5), (6, -13), (-44, 5), (-44, -13), (6, 17), (-44, 17)][position % 6]
        ax_macro.annotate(gene, (x, y), textcoords="offset points", xytext=offset, fontsize=7.5, color="#333333")

    all_x = subsets["qHTS"][0] + subsets["family"][0]
    all_y = subsets["qHTS"][1] + subsets["family"][1]
    _annotate_correlation(ax_macro, all_x, all_y, "all heads", (0.33, 0.50))
    _annotate_correlation(ax_macro, *subsets["family"], "family tier only", (0.33, 0.45))

    ax_macro.axhline(0, color="#999999", linewidth=0.8, zorder=1)
    ax_macro.set_xlabel(f"Mean {axis_name} similarity to the four scored isoforms ({unit})")
    ax_macro.set_ylabel("Macro RMSE improvement over baseline (0.9430)")
    ax_macro.set_title(f"Solo aux-head gain vs. {axis_name} similarity")
    handles, labels = ax_macro.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax_macro.legend(unique.values(), unique.keys(), title="Tier", fontsize=8, loc="center left")
    ax_macro.grid(True, alpha=0.25, zorder=0)

    # Panel 2: every (aux head, scored task) pair, so a head that helps only the
    # isoform it resembles is visible instead of averaged away.
    for task, marker in zip(_SCORED, ("o", "s", "^", "D")):
        x = [similarity[g][task] for g in genes if g != task]
        y = [per_task[g][task] for g in genes if g != task]
        ax_task.scatter(x, y, s=34, marker=marker, alpha=0.85, edgecolor="white", linewidth=0.4, label=task)

    pair_x = [similarity[g][t] for g in genes for t in _SCORED if g != t]
    pair_y = [per_task[g][t] for g in genes for t in _SCORED if g != t]
    _annotate_correlation(ax_task, pair_x, pair_y, "all pairs", (0.42, 0.60))

    ax_task.axhline(0, color="#999999", linewidth=0.8, zorder=1)
    ax_task.set_xlabel(f"{axis_name.capitalize()} similarity, aux head vs. that one task ({unit})")
    ax_task.set_ylabel("Per-task RMSE improvement over baseline")
    ax_task.set_title(f"Per-task gain vs. {axis_name} similarity (self-pairs excluded)")
    ax_task.legend(title="Scored task", fontsize=8)
    ax_task.grid(True, alpha=0.25, zorder=0)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def write_table(
    genes: list[str],
    proteins: dict[str, Protein],
    identity: dict[str, dict[str, float]],
    tm: dict[str, dict[str, float]],
    macro: dict[str, float],
    per_task: dict[str, dict[str, float]],
    out_path: Path,
) -> None:
    """Write the per-gene similarity/gain table the figures are drawn from."""
    header = (
        ["gene", "accession", "tier", "self_aux", "macro_gain"]
        + [f"gain_{t}" for t in _SCORED]
        + [f"identity_{t}" for t in _SCORED]
        + [f"tm_{t}" for t in _SCORED]
        + ["identity_mean", "identity_max", "tm_mean", "tm_max"]
    )
    lines = [",".join(header)]
    for gene in sorted(genes, key=lambda g: -macro[g]):
        ident_others = [identity[gene][t] for t in _SCORED if t != gene]
        tm_others = [tm[gene][t] for t in _SCORED if t != gene]
        row = (
            [gene, proteins[gene].accession, "qHTS" if gene in _QHTS else "family", str(gene in _SCORED), f"{macro[gene]:.4f}"]
            + [f"{per_task[gene][t]:.4f}" for t in _SCORED]
            + [f"{identity[gene][t]:.2f}" for t in _SCORED]
            + [f"{tm[gene][t]:.4f}" for t in _SCORED]
            + [f"{np.mean(ident_others):.2f}", f"{max(ident_others):.2f}",
               f"{np.mean(tm_others):.4f}", f"{max(tm_others):.4f}"]
        )
        lines.append(",".join(row))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")


def main() -> None:
    """Entry point: fetch proteins, measure both similarities, plot against gain."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-path", type=Path, default=Path("results/head_search_results.csv"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/proteins"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    parser.add_argument("--table-path", type=Path, default=Path("results/protein_similarity.csv"))
    args = parser.parse_args()

    macro, per_task = load_gains(args.results_path)
    genes = sorted(macro)

    print(f"Resolving {len(genes)} genes against UniProt", file=sys.stderr)
    uniprot = fetch_uniprot(genes, args.cache_dir)

    print("Fetching AlphaFold models", file=sys.stderr)
    proteins: dict[str, Protein] = {}
    for gene in genes:
        accession, sequence = uniprot[gene]
        coords, ca_sequence = read_ca_trace(fetch_structure(accession, args.cache_dir))
        proteins[gene] = Protein(gene, accession, sequence, coords, ca_sequence)

    aligner = _aligner()
    identity: dict[str, dict[str, float]] = {}
    tm: dict[str, dict[str, float]] = {}
    for gene in genes:
        identity[gene] = {
            task: sequence_identity(proteins[gene].sequence, proteins[task].sequence, aligner)
            for task in _SCORED
        }
        tm[gene] = {task: tm_score(proteins[gene], proteins[task]) for task in _SCORED}
        print(
            f"  {gene}: identity {min(identity[gene].values()):.1f}-{max(identity[gene].values()):.1f}%, "
            f"TM {min(tm[gene].values()):.3f}-{max(tm[gene].values()):.3f}",
            file=sys.stderr,
        )

    write_table(genes, proteins, identity, tm, macro, per_task, args.table_path)
    plot_axis("sequence", "% identity", identity, macro, per_task, args.out_dir / "similarity_vs_gain_sequence.png")
    plot_axis("structure", "TM-score", tm, macro, per_task, args.out_dir / "similarity_vs_gain_structure.png")


if __name__ == "__main__":
    main()
