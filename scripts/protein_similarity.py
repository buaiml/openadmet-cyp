"""Test whether protein similarity between isoforms explains auxiliary-head gain.

`reports/head_search_analysis.md` leaves the ranking unexplained: some aux heads
buy a measurable macro RMSE gain and others buy nothing, and row count does not
predict which.
`chemspace.py` attacks the ligand side of that question. This one attacks the
protein side: if an aux head helps because its enzyme is *like* the scored
enzymes -- same fold, same substrate-binding residues, therefore transferable
structure-activity signal -- then gain should rise with similarity to
CYP{1A2,2C9,2D6,3A4}.

Four similarity axes, two whole-protein and two restricted to the heme pocket:

  sequence    Needleman-Wunsch global alignment (BLOSUM62, -11/-1) of the
              canonical UniProt sequences, percent identity over the shorter
              sequence.
  structure   TM-score from TM-align over the AlphaFold DB models. Fold-level,
              length-normalised, and nearly saturated inside P450s -- every
              member shares the same triangular-prism fold, so the dynamic
              range here is small by construction and a flat trend on this
              axis is weaker evidence than a flat trend on sequence.
  pocket seq  Percent identity over the scored isoform's active-site residues
              only, read off the TM-align structural correspondence. Whole-
              protein identity is dominated by the conserved core and the
              I-helix; what decides whether two P450s turn over the same
              chemistry is the ~20 residues lining the cavity above the heme,
              and those are exactly what the two global axes average away.
  pocket shape  CA RMSD over the same positions after a Kabsch fit on the
              pocket pairs alone, so it measures cavity geometry rather than
              whether the two folds superpose (they always do). Lower is more
              similar, unlike the other three.

The pocket is defined per scored isoform, not by a sequence motif: a
ligand-bound crystal structure (CYP1A2 2HI4, CYP2C9 1R9O, CYP2D6 4WNV, CYP3A4
1TQN) is superposed onto that isoform's AlphaFold model, its heme is carried
into the model frame, and the pocket is every residue with a heavy atom within
--pocket-cutoff A of the heme *on the distal side* of the porphyrin plane. The
distal filter matters: the proximal shell is the cysteine ligand loop, which is
invariant across the whole superfamily and would flatten the axis by
construction. On CYP3A4 the default 6 A picks out R105, S119, I120, F302, A305,
T309, I369, A370, R372, L373, E374 -- the textbook substrate-recognition
positions -- so the definition is doing what it claims.

All four are measured against each of the four scored isoforms separately, then
aggregated per aux head as the mean over the four (the metric is a macro
average over those same four tasks) and as the max (a head could help by being
close to just one). Self-pairs are excluded from the aggregate: an aux head for
a scored isoform is 100% identical to itself, which is a statement about the
experiment design, not about transfer. Those four heads are drawn as stars, as
in `plot_volume_vs_gain.py`, and are confounded anyway -- they carry a
supervision channel no external head has.

Gains are read from the head-search JSON (`results/head_search.json`): solo-head
configs only (a multi-head config measures a set, not a head),
`baseline - config`, so positive means the head helped. Per-task gains come
from the same entries.

Usage:
    python scripts/protein_similarity.py
    python scripts/protein_similarity.py --results-path results/head_search.json

Network: UniProt REST (sequences), AlphaFold DB (models) and RCSB (the four
holo structures), all cached under --cache-dir, so a second run is offline.
Requires biopython and tmtools on top of the project requirements.
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
_RCSB_FILE_URL = "https://files.rcsb.org/download"
_TIMEOUT_S = 60

# Ligand-bound crystal structure per scored isoform, as (PDB id, chain). Each
# one is only ever used as a heme donor, so the requirement is a well-resolved
# HEM in a chain that superposes cleanly on the AlphaFold model.
_HOLO = {
    "CYP1A2": ("2HI4", "A"),   # alpha-naphthoflavone
    "CYP2C9": ("1R9O", "A"),   # flurbiprofen
    "CYP2D6": ("4WNV", "A"),   # thioridazine
    "CYP3A4": ("1TQN", "A"),   # ligand-free, heme present
}

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
class Residue:
    """One modelled residue: its letter, its CA, and all its heavy atoms."""

    letter: str
    number: int
    ca: np.ndarray
    atoms: np.ndarray


@dataclass
class Protein:
    """One isoform: its UniProt sequence and its AlphaFold model."""

    gene: str
    accession: str
    sequence: str
    residues: list[Residue]

    @property
    def ca_coords(self) -> np.ndarray:
        """N x 3 CA trace of the model."""
        return np.array([r.ca for r in self.residues])

    @property
    def ca_sequence(self) -> str:
        """One-letter sequence of the modelled residues, in model order."""
        return "".join(r.letter for r in self.residues)


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


def fetch_pdb(pdb_id: str, cache_dir: Path) -> Path:
    """Return a local path to an RCSB entry, downloading once."""
    path = cache_dir / f"{pdb_id}.pdb"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_http_get(f"{_RCSB_FILE_URL}/{pdb_id}.pdb"))
        print(f"  {pdb_id}: RCSB entry", file=sys.stderr)
    return path


def read_residues(path: Path, chain: str | None = None) -> tuple[list[Residue], dict[str, np.ndarray]]:
    """Return (protein residues, heme atoms) from a PDB file.

    Hydrogens and alternate locations beyond the first are dropped; residues
    without a CA are dropped, since every downstream step needs one. The heme
    dictionary is keyed by PDB atom name (FE, NA, NB, ...) and is empty for an
    AlphaFold model, which has no cofactor.
    """
    collected: dict[str, dict] = {}
    order: list[str] = []
    heme: dict[str, np.ndarray] = {}

    for line in path.read_text().splitlines():
        record = line[:6].strip()
        if record not in ("ATOM", "HETATM") or (chain and line[21] != chain):
            continue
        if line[16] not in (" ", "A") or line[76:78].strip() == "H":
            continue
        name, resname = line[12:16].strip(), line[17:20].strip()
        xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])

        if resname == "HEM":
            heme.setdefault(name, xyz)
            continue
        if record != "ATOM" or resname not in _AA3TO1:
            continue

        key = line[21] + line[22:27]
        if key not in collected:
            collected[key] = {"letter": _AA3TO1[resname], "number": int(line[22:26]), "atoms": [], "ca": None}
            order.append(key)
        collected[key]["atoms"].append(xyz)
        if name == "CA":
            collected[key]["ca"] = xyz

    residues = [
        Residue(collected[k]["letter"], collected[k]["number"], collected[k]["ca"], np.array(collected[k]["atoms"]))
        for k in order
        if collected[k]["ca"] is not None
    ]
    return residues, heme


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


def _kabsch_rmsd(p: np.ndarray, q: np.ndarray) -> float:
    """RMSD between two equal-length point sets after optimal superposition."""
    p, q = p - p.mean(0), q - q.mean(0)
    v, _, wt = np.linalg.svd(p.T @ q)
    if np.linalg.det(v @ wt) < 0:
        v[:, -1] *= -1
    return float(np.sqrt(((p @ (v @ wt) - q) ** 2).sum() / len(p)))


def transfer_heme(model: Protein, holo_path: Path, chain: str) -> tuple[dict[str, np.ndarray], float]:
    """Carry a crystal structure's heme into an AlphaFold model's frame.

    Returns (heme atoms in model coordinates, superposition RMSD). The donor is
    the same protein as the model, so the fit is a sanity check rather than a
    free parameter -- a large RMSD here means the wrong chain was picked.
    """
    holo_residues, heme = read_residues(holo_path, chain)
    if not heme:
        raise SystemExit(f"{holo_path} chain {chain} has no HEM")

    holo_ca = np.array([r.ca for r in holo_residues])
    holo_seq = "".join(r.letter for r in holo_residues)
    result = tm_align(holo_ca, model.ca_coords, holo_seq, model.ca_sequence)
    rotation, translation = np.array(result.u), np.array(result.t)
    return {name: rotation @ xyz + translation for name, xyz in heme.items()}, result.rmsd


def pocket_indices(model: Protein, heme: dict[str, np.ndarray], cutoff: float) -> list[int]:
    """Residue indices lining the substrate cavity above the heme.

    Distal side only. The porphyrin plane is fit through the four pyrrole
    nitrogens and oriented away from the axial cysteine, so the conserved
    proximal ligand loop -- identical across the superfamily, and therefore
    pure noise for this question -- is excluded.
    """
    iron = heme["FE"]
    heme_atoms = np.array(list(heme.values()))
    pyrrole = np.array([heme[n] for n in ("NA", "NB", "NC", "ND")])
    normal = np.linalg.svd(pyrrole - pyrrole.mean(0))[2][2]
    normal = normal / np.linalg.norm(normal)

    cysteine_atoms = np.vstack([r.atoms for r in model.residues if r.letter == "C"])
    axial = cysteine_atoms[np.argmin(np.linalg.norm(cysteine_atoms - iron, axis=1))]
    if (axial - iron) @ normal > 0:
        normal = -normal

    indices = []
    for i, residue in enumerate(model.residues):
        distal = residue.atoms[(residue.atoms - iron) @ normal > 0]
        if len(distal) == 0:
            continue
        if np.linalg.norm(distal[:, None, :] - heme_atoms[None], axis=2).min() <= cutoff:
            indices.append(i)
    return indices


def structure_metrics(aux: Protein, scored: Protein, pocket: list[int]) -> dict[str, float]:
    """TM-score plus pocket identity and pocket CA RMSD for one aux/scored pair.

    One TM-align call serves both: its alignment gives the residue
    correspondence the pocket metrics are read off, so the pocket comparison
    inherits a structural alignment rather than a sequence one -- necessary at
    20-30% identity, where a sequence alignment of the cavity is not reliable.
    """
    result = tm_align(aux.ca_coords, scored.ca_coords, aux.ca_sequence, scored.ca_sequence)

    # Walk the gapped alignment to map scored-model index -> aux-model index.
    partner: dict[int, int] = {}
    i = j = 0
    for a, b in zip(result.seqxA, result.seqyA):
        if a != "-" and b != "-":
            partner[j] = i
        i += a != "-"
        j += b != "-"

    matched = [(k, partner[k]) for k in pocket if k in partner]
    identical = sum(1 for k, m in matched if scored.residues[k].letter == aux.residues[m].letter)
    rmsd = (
        _kabsch_rmsd(
            np.array([aux.residues[m].ca for _, m in matched]),
            np.array([scored.residues[k].ca for k, _ in matched]),
        )
        if len(matched) >= 3
        else float("nan")
    )

    return {
        "tm": 0.5 * (result.tm_norm_chain1 + result.tm_norm_chain2),
        "pocket_identity": 100.0 * identical / len(pocket),
        "pocket_rmsd": rmsd,
        "pocket_coverage": len(matched) / len(pocket),
    }


# --------------------------------------------------------------------------
# gains
# --------------------------------------------------------------------------


def load_gains(results_path: Path) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Return ({gene: macro gain}, {gene: {scored isoform: gain}}) from head-search.

    Solo configs only. Gain is `baseline - config`, so positive means the head
    helped.
    """
    entries = json.loads(results_path.read_text())

    baseline = next((e for e in entries if not e["candidates"]), None)
    if baseline is None:
        raise SystemExit(f"{results_path} has no baseline entry (one with an empty 'candidates')")

    base_heads = baseline.get("per_head_rmse_mean", {})
    macro: dict[str, float] = {}
    per_task: dict[str, dict[str, float]] = {}
    for entry in entries:
        if len(entry["candidates"]) != 1:
            continue
        gene = entry["candidates"][0]
        solo_heads = entry.get("per_head_rmse_mean", {})
        macro[gene] = baseline["macro_rmse_mean"] - entry["macro_rmse_mean"]
        per_task[gene] = {
            task: base_heads[f"{task}_pIC50_direct_inhibition"] - solo_heads[f"{task}_pIC50_direct_inhibition"]
            for task in _SCORED
            if f"{task}_pIC50_direct_inhibition" in base_heads and f"{task}_pIC50_direct_inhibition" in solo_heads
        }

    print(
        f"{results_path}: baseline macro RMSE {baseline['macro_rmse_mean']:.4f}, {len(macro)} solo heads",
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
    ax_macro.set_ylabel("Macro RMSE improvement over no-aux baseline")
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
    axes: dict[str, dict[str, dict[str, float]]],
    macro: dict[str, float],
    per_task: dict[str, dict[str, float]],
    out_path: Path,
) -> None:
    """Write the per-gene similarity/gain table the figures are drawn from."""
    names = list(axes)
    header = (
        ["gene", "accession", "tier", "self_aux", "macro_gain"]
        + [f"gain_{t}" for t in _SCORED]
        + [f"{name}_{t}" for name in names for t in _SCORED]
        + [f"{name}_{stat}" for name in names for stat in ("mean", "max", "min")]
    )
    lines = [",".join(header)]
    for gene in sorted(genes, key=lambda g: -macro[g]):
        others = {name: [axes[name][gene][t] for t in _SCORED if t != gene] for name in names}
        row = (
            [gene, proteins[gene].accession, "qHTS" if gene in _QHTS else "family", str(gene in _SCORED), f"{macro[gene]:.4f}"]
            + [f"{per_task[gene][t]:.4f}" for t in _SCORED]
            + [f"{axes[name][gene][t]:.3f}" for name in names for t in _SCORED]
            + [f"{f(others[name]):.3f}" for name in names for f in (np.mean, max, min)]
        )
        lines.append(",".join(row))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")


def main() -> None:
    """Entry point: fetch proteins, measure all four similarities, plot against gain."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-path", type=Path, default=Path("results/head_search.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/proteins"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    parser.add_argument("--table-path", type=Path, default=Path("results/protein_similarity.csv"))
    parser.add_argument(
        "--pocket-cutoff",
        type=float,
        default=6.0,
        help="A from any heme atom, distal side, for a residue to count as pocket (default 6.0)",
    )
    args = parser.parse_args()

    macro, per_task = load_gains(args.results_path)
    genes = sorted(macro)

    print(f"Resolving {len(genes)} genes against UniProt", file=sys.stderr)
    uniprot = fetch_uniprot(genes, args.cache_dir)

    print("Fetching AlphaFold models", file=sys.stderr)
    proteins: dict[str, Protein] = {}
    for gene in genes:
        accession, sequence = uniprot[gene]
        residues, _ = read_residues(fetch_structure(accession, args.cache_dir))
        proteins[gene] = Protein(gene, accession, sequence, residues)

    print(f"Defining heme pockets ({args.pocket_cutoff} A, distal side)", file=sys.stderr)
    pockets: dict[str, list[int]] = {}
    for task in _SCORED:
        pdb_id, chain = _HOLO[task]
        heme, fit_rmsd = transfer_heme(proteins[task], fetch_pdb(pdb_id, args.cache_dir), chain)
        pockets[task] = pocket_indices(proteins[task], heme, args.pocket_cutoff)
        lining = " ".join(
            f"{proteins[task].residues[i].letter}{proteins[task].residues[i].number}" for i in pockets[task]
        )
        print(f"  {task}: {pdb_id} fit {fit_rmsd:.2f} A, {len(pockets[task])} residues: {lining}", file=sys.stderr)

    aligner = _aligner()
    axes: dict[str, dict[str, dict[str, float]]] = {
        "identity": {}, "tm": {}, "pocket_identity": {}, "pocket_rmsd": {}, "pocket_coverage": {}
    }
    for gene in genes:
        axes["identity"][gene] = {
            task: sequence_identity(proteins[gene].sequence, proteins[task].sequence, aligner)
            for task in _SCORED
        }
        for task in _SCORED:
            metrics = structure_metrics(proteins[gene], proteins[task], pockets[task])
            for key in ("tm", "pocket_identity", "pocket_rmsd", "pocket_coverage"):
                axes[key].setdefault(gene, {})[task] = metrics[key]
        print(
            f"  {gene}: identity {min(axes['identity'][gene].values()):.1f}-{max(axes['identity'][gene].values()):.1f}%, "
            f"TM {min(axes['tm'][gene].values()):.3f}-{max(axes['tm'][gene].values()):.3f}, "
            f"pocket identity {min(axes['pocket_identity'][gene].values()):.0f}-{max(axes['pocket_identity'][gene].values()):.0f}%, "
            f"pocket RMSD {min(axes['pocket_rmsd'][gene].values()):.2f}-{max(axes['pocket_rmsd'][gene].values()):.2f} A",
            file=sys.stderr,
        )

    write_table(genes, proteins, axes, macro, per_task, args.table_path)
    plot_axis("sequence", "% identity", axes["identity"], macro, per_task,
              args.out_dir / "similarity_vs_gain_sequence.png")
    plot_axis("structure", "TM-score", axes["tm"], macro, per_task,
              args.out_dir / "similarity_vs_gain_structure.png")
    plot_axis("pocket sequence", "% identity", axes["pocket_identity"], macro, per_task,
              args.out_dir / "similarity_vs_gain_pocket_sequence.png")
    plot_axis("pocket shape", "CA RMSD, A (lower = more similar)", axes["pocket_rmsd"], macro, per_task,
              args.out_dir / "similarity_vs_gain_pocket_shape.png")


if __name__ == "__main__":
    main()
