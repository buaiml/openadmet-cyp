"""Project every dataset in the union onto a shared chemical-space map.

The question this answers is the one `aux_similarity.py` can only summarise:
why some auxiliary heads buy a measurable macro RMSE gain and others buy
nothing. Row count and nearest-neighbour Tanimoto to the eval set are scalar
summaries (and every head sits *further* from the eval molecules than the
baseline's own training molecules already did). This one draws the map
instead, so the shape of each dataset -- and where the challenge molecules
fall inside it -- is visible rather than averaged away.

Datasets are kept separate the way head-search keeps them separate:

  challenge/{train,val,test}   scored pIC50 rows, split by the pinned `split`
                               column. `test` is what head-search reports RMSE
                               on; `val` is the second holdout.
  challenge/blind             data/test.csv -- the 750 submission molecules.
                               No IC50 exists for these, so they appear on the
                               map but never in any gain.
  qhts/{isoform}              PubChem AID 1851, one group per isoform.
  family/{gene}               ChEMBL + BindingDB affinity heads, one per gene.

qHTS and family are split apart even where they name the same protein, because
they are different libraries: AID 1851 is a single 17k-compound screening deck
run across five isoforms, while the family heads are heterogeneous med-chem
series accumulated per target. Pooling them would hide exactly the difference
this map is meant to show. Gains, however, are only measured at the
protein-level candidate head-search actually ran, so both groups of one
protein carry that protein's single gain -- stated on the figures.

Two colourings, as separate figures:

  --by-isoform   each dataset highlighted against a grey backdrop of every
                 molecule, with the challenge holdouts overlaid in every panel
                 so "where does the test set land" is answerable per dataset.
                 Small multiples rather than one 20-colour scatter: past three
                 series a categorical scatter palette stops being separable
                 for colour-blind readers, and a 40k-point overplot stops
                 being readable for everyone.

  --by-gain      the same small multiples, but each dataset painted a flat
                 colour for the macro RMSE improvement its head bought in
                 head-search. Per-dataset rather than per-molecule because a
                 qHTS molecule carries all five isoform heads at once, so any
                 per-molecule reduction either saturates or averages the
                 question away. With --gain-metric per-head an extra pooled
                 figure is drawn, one panel per scored isoform coloured by
                 that head's own RMSE delta -- "this isoform's contribution to
                 the improvement on the 4 challenge isoforms", literally.

Dimensionality reduction is run several ways because no single projection is
trustworthy on its own: PCA is faithful to global variance and useless for
local structure, t-SNE is the reverse, UMAP sits between them. A cluster that
survives all three is a real cluster. t-SNE and UMAP are seeded from a 50-
component PCA of the fingerprint matrix, which is standard practice and keeps
runtime linear in molecules rather than quadratic.

Dependencies are the stdlib, numpy, scikit-learn, rdkit and matplotlib. Polars
is deliberately not imported: the SCC compute nodes lack the AVX2/FMA CPU
features its packaged wheel is built against and importing it dies with SIGILL.
UMAP is optional and skipped with a warning if umap-learn is not installed.

Usage (SCC):
    python scripts/chemspace.py \
        --data-path data/union_train.csv \
        --test-path data/test.csv \
        --results-path results/head_search.json \
        --out-dir reports \
        --methods pca tsne umap

After a head-search re-run, redraw only the gain figures from the saved dump:
    python scripts/chemspace.py --from-embedding results/chemspace_embedding.csv \
        --results-path results/head_search.json --gain-metric per-head
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

# Must precede the pyplot import: the default cache lands in $HOME, which is
# over quota on the SCC, and matplotlib only warns before falling back slowly.
os.environ.setdefault("MPLCONFIGDIR", os.environ.get("TMPDIR", "/tmp") + "/mpl-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.colors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# Validated three-slot categorical set (all-pairs, light surface): blue,
# orange, aqua. Three is the cap for a scatter -- a fourth slot puts yellow
# beside orange and fails colour-blind separation on a point cloud. Everything
# past three is separated by faceting, not hue.
_ACCENT = "#2a78d6"
_TEST = "#eb6834"
_VAL = "#1baf7a"
_BACKDROP = "#d8d7d2"
_INK = "#0b0b0b"
_INK_SOFT = "#52514e"

# Sequential blue ramp, light -> dark, for the gain colouring.
_SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

_SCORED_DEFAULT = [
    "CYP1A2_pIC50_direct_inhibition",
    "CYP2C9_pIC50_direct_inhibition",
    "CYP2D6_pIC50_direct_inhibition",
    "CYP3A4_pIC50_direct_inhibition",
]

_NON_LABEL_COLUMNS = {"SMILES", "inchikey_block", "inchikey_full", "split", "PUBCHEM_CID"}

# Source token (the trailing part of `{protein}_{readout}_{source}`) that marks
# a column as coming from the PubChem qHTS panel rather than a family head.
_QHTS_SOURCE = "aid1851"

# Caption notes for projections whose caveat does not depend on the data. PCA's
# does (how much variance two components hold) and is written in embed().
_NOTES = {
    "tsne": "local structure only — cluster sizes and between-cluster distances are not meaningful",
    "umap": "local structure preferred over global — between-cluster distances are only weakly meaningful",
}

_CHALLENGE_GROUPS = ["challenge/train", "challenge/val", "challenge/test", "challenge/blind"]


# --------------------------------------------------------------------------
# table loading
# --------------------------------------------------------------------------


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


def discover_groups(
    header: list[str],
    columns: dict[str, list[str]],
    scored_columns: list[str],
) -> dict[str, list[str]]:
    """Map dataset group name -> the auxiliary columns that define membership.

    Column names are `{protein}_{readout}_{source}`, so the leading token is
    the protein and the trailing one the source. Grouping is by
    (source class, protein) so the qHTS deck and the family affinity data of
    the same protein stay apart -- see the module docstring.
    """
    skip = set(scored_columns) | _NON_LABEL_COLUMNS
    groups: dict[str, list[str]] = {}
    for column in header:
        if column in skip or not _is_numeric(columns[column]):
            continue
        parts = column.split("_")
        protein = parts[0]
        source = parts[-1] if len(parts) >= 3 else ""
        prefix = "qhts" if source == _QHTS_SOURCE else "family"
        groups.setdefault(f"{prefix}/{protein}", []).append(column)
    return groups


def load_gains(results_path: Path, scored_columns: list[str]) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Return ({candidate: macro gain}, {candidate: {scored head: gain}}) from head-search.

    Only single-candidate entries are read. A round-2+ entry measures a set,
    not a head, so it cannot be attributed to one dataset. Gain is
    `baseline - solo`, so positive means the head helped.
    """
    entries = json.loads(results_path.read_text())

    baseline = next((e for e in entries if not e["candidates"]), None)
    if baseline is None:
        raise SystemExit(f"{results_path} has no baseline entry (one with an empty 'candidates')")

    macro: dict[str, float] = {}
    per_head: dict[str, dict[str, float]] = {}
    for entry in entries:
        if len(entry["candidates"]) != 1:
            continue
        name = entry["candidates"][0]
        macro[name] = baseline["macro_rmse_mean"] - entry["macro_rmse_mean"]
        base_heads = baseline.get("per_head_rmse_mean", {})
        solo_heads = entry.get("per_head_rmse_mean", {})
        per_head[name] = {
            head: base_heads[head] - solo_heads[head]
            for head in scored_columns
            if head in base_heads and head in solo_heads
        }

    print(
        f"{results_path}: baseline macro RMSE {baseline['macro_rmse_mean']:.4f}, {len(macro)} solo heads",
        file=sys.stderr,
    )
    return macro, per_head


# --------------------------------------------------------------------------
# molecules
# --------------------------------------------------------------------------


def _connectivity_keys(smiles: list[str]) -> list[str | None]:
    """InChIKey connectivity block per SMILES, None where parsing fails.

    Same key `build_union` and `check-overlap` join on, so a molecule that
    reaches this map through two sources collapses to one point even when the
    sources spell it as a salt, a stereoisomer or a different charge state.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from data_tools.standardize import connectivity_key

    return [connectivity_key(smi) if smi else None for smi in smiles]


def _fingerprint_matrix(smiles: list[str]) -> tuple[np.ndarray, list[int]]:
    """ECFP4 (radius 2, 2048 bits) as a uint8 matrix, plus the rows kept.

    Same generator and parameters as `load.py::_butina_split` and
    `split_column.py::_butina_groups`, so distances on this map are comparable
    to the clustering the pinned split was built from.
    """
    from rdkit import RDLogger
    from rdkit.Chem import MolFromSmiles
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

    RDLogger.DisableLog("rdApp.*")
    morgan = GetMorganGenerator(radius=2, fpSize=2048)

    rows, kept = [], []
    for idx, smi in enumerate(smiles):
        mol = MolFromSmiles(smi) if smi else None
        if mol is None:
            continue
        rows.append(np.frombuffer(morgan.GetFingerprintAsNumPy(mol).tobytes(), dtype=np.uint8))
        kept.append(idx)

    n_bad = len(smiles) - len(kept)
    if n_bad:
        print(f"  {n_bad} SMILES failed to parse and were dropped", file=sys.stderr)
    return np.vstack(rows) if rows else np.zeros((0, 2048), dtype=np.uint8), kept


def collect_molecules(
    data_path: Path,
    test_path: Path | None,
    scored_columns: list[str],
) -> tuple[list[str], dict[str, np.ndarray], dict[str, list[str]]]:
    """Build the deduplicated molecule list and a boolean membership mask per group.

    Returns (smiles, {group: bool mask over molecules}, {group: source columns}).
    """
    header, columns = read_table(data_path)
    if "split" not in columns:
        raise SystemExit(
            f"{data_path} has no 'split' column -- this map has to mark the same held-out "
            "molecules head-search used. Run add-split-column first.",
        )
    if "SMILES" not in columns:
        raise SystemExit(f"{data_path} has no 'SMILES' column")

    scored_present = [c for c in scored_columns if c in columns]
    if not scored_present:
        raise SystemExit(f"None of the scored columns {scored_columns} are in {data_path}")

    aux_groups = discover_groups(header, columns, scored_columns)
    n_qhts = sum(1 for g in aux_groups if g.startswith("qhts/"))
    n_family = sum(1 for g in aux_groups if g.startswith("family/"))
    print(
        f"{data_path}: {len(columns['SMILES'])} rows, {n_qhts} qHTS groups, {n_family} family groups",
        file=sys.stderr,
    )
    if not n_family:
        print(
            "WARNING: no family (chembl/bindingdb) columns found. This union was built from the "
            "qHTS source alone, so the family datasets -- the ones the gain question is about -- "
            "will be missing from the map. Rebuild with build-family-heads + build-union.",
            file=sys.stderr,
        )

    # Row -> molecule identity, then merge duplicate structures across sources.
    row_smiles = columns["SMILES"]
    row_keys = columns.get("inchikey_block")
    if row_keys is None or any(k == "" for k in row_keys):
        print("Computing connectivity keys...", file=sys.stderr)
        computed = _connectivity_keys(row_smiles)
        row_keys = [k if k else f"__row{i}__" for i, k in enumerate(computed)]

    index_of: dict[str, int] = {}
    smiles: list[str] = []
    row_to_molecule: list[int] = []
    for key, smi in zip(row_keys, row_smiles):
        if key not in index_of:
            index_of[key] = len(smiles)
            smiles.append(smi)
        row_to_molecule.append(index_of[key])

    n_rows = len(row_smiles)
    masks: dict[str, np.ndarray] = {}

    def _blank_mask() -> np.ndarray:
        return np.zeros(len(smiles), dtype=bool)

    # Challenge splits: scored rows only, so the qHTS-only rows that share the
    # split column do not masquerade as challenge molecules.
    split = columns["split"]
    for name in ("train", "val", "test"):
        mask = _blank_mask()
        for i in range(n_rows):
            if split[i] == name and any(columns[c][i] != "" for c in scored_present):
                mask[row_to_molecule[i]] = True
        masks[f"challenge/{name}"] = mask

    for group, group_columns in aux_groups.items():
        mask = _blank_mask()
        for i in range(n_rows):
            if any(columns[c][i] != "" for c in group_columns):
                mask[row_to_molecule[i]] = True
        masks[group] = mask

    # The blind submission set is a separate file with no labels at all. Its
    # molecules are appended, not joined, because build_union deliberately
    # keeps them out of the training table.
    blind = _blank_mask()
    if test_path is not None and test_path.exists():
        _, test_columns = read_table(test_path)
        if "SMILES" not in test_columns:
            raise SystemExit(f"{test_path} has no 'SMILES' column")
        test_smiles = test_columns["SMILES"]
        test_keys = _connectivity_keys(test_smiles)
        blind = list(blind)
        for key, smi in zip(test_keys, test_smiles):
            key = key if key else f"__blind{len(smiles)}__"
            if key not in index_of:
                index_of[key] = len(smiles)
                smiles.append(smi)
                blind.append(True)
                for name, mask in masks.items():
                    masks[name] = np.append(mask, False)
            else:
                blind[index_of[key]] = True
        blind = np.asarray(blind, dtype=bool)
        overlap = int(blind.sum()) - len(test_smiles)
        print(
            f"{test_path}: {len(test_smiles)} blind molecules "
            f"({int(blind.sum())} unique after keying)",
            file=sys.stderr,
        )
        if overlap > 0:
            print(
                f"  NOTE: {overlap} blind molecules already appear in the union -- "
                "build_union is supposed to drop test overlap, so check check-overlap.",
                file=sys.stderr,
            )
    else:
        print(f"WARNING: {test_path} not found -- the blind test set will be missing", file=sys.stderr)
    masks["challenge/blind"] = blind

    group_columns = {g: aux_groups.get(g, []) for g in masks}
    return smiles, masks, group_columns


# --------------------------------------------------------------------------
# projections
# --------------------------------------------------------------------------


def embed(
    fingerprints: np.ndarray,
    methods: list[str],
    seed: int,
    pca_components: int,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Return ({method: (n, 2) coordinates}, {method: caption note}) per projection.

    The note carries whatever a reader needs to not over-trust the picture --
    for PCA, how little of the fingerprint variance two components actually
    hold, which on a 2048-bit ECFP4 matrix is a few percent.
    """
    from sklearn.decomposition import PCA

    out: dict[str, np.ndarray] = {}
    notes: dict[str, str] = {}
    features = fingerprints.astype(np.float32)

    n_components = min(pca_components, features.shape[0], features.shape[1])
    print(f"PCA to {n_components} components...", file=sys.stderr)
    pca = PCA(n_components=n_components, random_state=seed)
    reduced = pca.fit_transform(features)

    for method in methods:
        if method == "pca":
            explained = pca.explained_variance_ratio_[:2].sum()
            print(f"  pca: first two components explain {explained:.1%} of variance", file=sys.stderr)
            out["pca"] = reduced[:, :2]
            notes["pca"] = f"PC1+PC2 hold {explained:.1%} of fingerprint variance — read relative position, not distance"

        elif method == "tsne":
            from sklearn.manifold import TSNE

            print(f"  tsne on {reduced.shape[0]} molecules (this is the slow one)...", file=sys.stderr)
            notes["tsne"] = _NOTES["tsne"]
            out["tsne"] = TSNE(
                n_components=2,
                init="pca",
                perplexity=min(30.0, max(5.0, (reduced.shape[0] - 1) / 3.0)),
                random_state=seed,
            ).fit_transform(reduced)

        elif method == "umap":
            try:
                from umap import UMAP
            except ImportError as exc:
                print(
                    f"  umap: umap-learn is not importable ({exc}) -- skipping. "
                    "`pip install umap-learn` in the cyp env to include it.",
                    file=sys.stderr,
                )
                continue
            except Exception as exc:  # pylint: disable=broad-except
                # umap-learn imports numba, which on a cluster fails in ways
                # that are not ImportError -- an llvmlite/numpy ABI mismatch, or
                # a numba cache it cannot write. Those must not take the whole
                # run down after t-SNE has already been paid for, and they must
                # not be reported as "not installed", which sends you to
                # reinstall a package that is already there.
                print(
                    f"  umap: umap-learn is installed but failed to import "
                    f"({type(exc).__name__}: {exc}) -- skipping.",
                    file=sys.stderr,
                )
                continue
            print(f"  umap on {reduced.shape[0]} molecules...", file=sys.stderr)
            notes["umap"] = _NOTES["umap"]
            out["umap"] = UMAP(n_components=2, random_state=seed).fit_transform(reduced)

        elif method == "mds":
            from sklearn.manifold import MDS

            print(f"  mds on {reduced.shape[0]} molecules...", file=sys.stderr)
            out["mds"] = MDS(n_components=2, random_state=seed, normalized_stress="auto").fit_transform(reduced)

        elif method == "isomap":
            from sklearn.manifold import Isomap

            print(f"  isomap on {reduced.shape[0]} molecules...", file=sys.stderr)
            out["isomap"] = Isomap(n_components=2).fit_transform(reduced)

        else:
            raise SystemExit(f"Unknown method '{method}'")

    return out, notes


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------


def _order_groups(masks: dict[str, np.ndarray], gains: dict[str, float]) -> list[str]:
    """Challenge groups first, then qHTS and family datasets by descending gain."""
    datasets = [g for g in masks if g not in _CHALLENGE_GROUPS]

    def key(group: str) -> tuple:
        prefix, protein = group.split("/", 1)
        gain = gains.get(protein)
        return (0 if prefix == "qhts" else 1, -(gain if gain is not None else -9.9), protein)

    return [g for g in _CHALLENGE_GROUPS if g in masks] + sorted(datasets, key=key)


def _style_axis(ax, title: str, subtitle: str | None = None) -> None:
    """Strip an axis down to a titled frame -- projection units carry no meaning."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#e4e3de")
    ax.set_title(title, fontsize=8.5, color=_INK, pad=4)
    if subtitle:
        # Inside the frame, not above it: a second line over the title collides
        # with it at small-multiple sizes, and the corner is empty anyway.
        ax.text(
            0.03, 0.97, subtitle, transform=ax.transAxes, ha="left", va="top",
            fontsize=7, color=_INK_SOFT, zorder=5,
        )


def _backdrop(ax, coords: np.ndarray) -> None:
    """Every molecule, recessive, so each panel shares one frame of reference."""
    ax.scatter(coords[:, 0], coords[:, 1], s=1.5, c=_BACKDROP, linewidth=0, rasterized=True, zorder=1)


def figure_overview(
    coords_by_method: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    notes: dict[str, str],
    out_path: Path,
) -> None:
    """One column per projection: source classes on top, challenge holdouts below."""
    methods = list(coords_by_method)
    fig, axes = plt.subplots(2, len(methods), figsize=(4.4 * len(methods), 9.0), squeeze=False)

    qhts = np.zeros_like(masks["challenge/blind"])
    family = np.zeros_like(qhts)
    for group, mask in masks.items():
        if group.startswith("qhts/"):
            qhts |= mask
        elif group.startswith("family/"):
            family |= mask
    challenge = np.zeros_like(qhts)
    for group in _CHALLENGE_GROUPS:
        challenge |= masks.get(group, np.zeros_like(qhts))

    for column, method in enumerate(methods):
        coords = coords_by_method[method]

        ax = axes[0][column]
        # Largest set first and partly transparent: these three overlap heavily,
        # and whichever is drawn last opaque simply erases the other two.
        layers = sorted(
            ((family, _VAL, "family"), (qhts, _ACCENT, "qHTS"), (challenge, _TEST, "challenge")),
            key=lambda layer: int(layer[0].sum()),
            reverse=True,
        )
        for mask, color, label in layers:
            ax.scatter(
                coords[mask, 0], coords[mask, 1], s=2.5, c=color, alpha=0.55,
                linewidth=0, rasterized=True, label=label, zorder=2,
            )
        _style_axis(ax, f"{method.upper()} — where each source sits")
        legend = ax.legend(loc="upper right", fontsize=7.5, frameon=True, markerscale=4, labelcolor=_INK_SOFT)
        legend.get_frame().set_edgecolor("#e4e3de")

        ax = axes[1][column]
        _backdrop(ax, coords)
        for group, color, size in (
            ("challenge/val", _VAL, 5),
            ("challenge/test", _ACCENT, 5),
            ("challenge/blind", _TEST, 6),
        ):
            mask = masks.get(group)
            if mask is None or not mask.any():
                continue
            ax.scatter(
                coords[mask, 0], coords[mask, 1], s=size, c=color, linewidth=0.25,
                edgecolor="#fcfcfb", rasterized=True, label=f"{group.split('/')[1]}  n={int(mask.sum())}", zorder=3,
            )
        _style_axis(ax, f"{method.upper()} — where the held-out molecules land")
        legend = ax.legend(loc="upper right", fontsize=7.5, frameon=True, markerscale=3, labelcolor=_INK_SOFT)
        legend.get_frame().set_edgecolor("#e4e3de")

    fig.suptitle(
        "Chemical space of every CYP dataset in the union (ECFP4, radius 2, 2048 bits)",
        fontsize=12, color=_INK,
    )
    caveats = "\n".join(f"{m.upper()}: {notes[m]}" for m in methods if notes.get(m))
    fig.text(
        0.5, 0.005,
        "Grey = all molecules. 'challenge/blind' is data/test.csv, the submission set, which carries no IC50."
        + (f"\n{caveats}" if caveats else ""),
        ha="center", fontsize=8, color=_INK_SOFT,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}", file=sys.stderr)


def figure_by_isoform(
    method: str,
    coords: np.ndarray,
    masks: dict[str, np.ndarray],
    gains: dict[str, float],
    out_path: Path,
    n_columns: int,
    note: str = "",
) -> None:
    """Small multiples: one panel per dataset, holdouts overlaid in every panel."""
    order = _order_groups(masks, gains)
    n_rows = -(-len(order) // n_columns)
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(2.5 * n_columns, 2.7 * n_rows), squeeze=False)

    for position, group in enumerate(order):
        ax = axes[position // n_columns][position % n_columns]
        mask = masks[group]
        _backdrop(ax, coords)

        # The scored holdout goes under every dataset, not over it. Drawn on
        # top at equal weight it simply covers the smaller family decks -- the
        # question is where the dataset sits relative to the test molecules,
        # so the dataset has to be the mark you read.
        holdout = masks.get("challenge/test")
        if holdout is not None and holdout.any() and group not in _CHALLENGE_GROUPS:
            ax.scatter(
                coords[holdout, 0], coords[holdout, 1],
                s=2.0, c=_TEST, alpha=0.45, linewidth=0, rasterized=True, zorder=2,
            )

        ax.scatter(coords[mask, 0], coords[mask, 1], s=3.0, c=_ACCENT, linewidth=0, rasterized=True, zorder=3)

        protein = group.split("/", 1)[1] if "/" in group else group
        gain = gains.get(protein)
        subtitle = f"n={int(mask.sum())}"
        if group not in _CHALLENGE_GROUPS:
            subtitle += f"   gain {gain:+.3f}" if gain is not None else "   gain n/a"
        _style_axis(ax, group, subtitle)

    for position in range(len(order), n_rows * n_columns):
        axes[position // n_columns][position % n_columns].axis("off")

    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=5, color=_ACCENT, label="this dataset"),
        Line2D([], [], marker="o", linestyle="", markersize=5, color=_TEST, label="challenge/test (scored holdout)"),
        Line2D([], [], marker="o", linestyle="", markersize=5, color=_BACKDROP, label="all molecules"),
    ]
    legend = fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.012),
        ncol=3, fontsize=8.5, frameon=True, labelcolor=_INK_SOFT,
    )
    legend.get_frame().set_edgecolor("#e4e3de")

    fig.suptitle(f"{method.upper()} — each dataset against the shared chemical space", fontsize=12, color=_INK)
    fig.text(
        0.5, 0.001,
        "Datasets ordered by the macro RMSE gain their head bought in head-search (qHTS first, then family). "
        "Gain is measured per protein-level candidate, so a protein's qHTS and family panels share one number. "
        "challenge/val is its own panel rather than an overlay."
        + (f"\n{method.upper()}: {note}" if note else ""),
        ha="center", fontsize=8, color=_INK_SOFT,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.975))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}", file=sys.stderr)


def _gain_per_molecule(
    masks: dict[str, np.ndarray],
    gains: dict[str, float],
    n_molecules: int,
    aggregate: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-molecule gain and a mask of which molecules have one at all.

    A molecule usually carries several heads' labels, so it has several gains.
    `mean` reads as "the average head this molecule feeds"; `max` as "the best
    one it feeds". Neither is the truth -- gain was measured per head, not per
    molecule -- and `max` in particular saturates, because every molecule in
    the qHTS deck carries all five isoform heads and so inherits the best of
    them. `mean` is the default for that reason; the per-dataset panels below
    are the colouring that needs no aggregation at all.
    """
    total = np.zeros(n_molecules)
    count = np.zeros(n_molecules)
    best = np.full(n_molecules, -np.inf)

    for group, mask in masks.items():
        if group in _CHALLENGE_GROUPS:
            continue
        gain = gains.get(group.split("/", 1)[1])
        if gain is None:
            continue
        total[mask] += gain
        count[mask] += 1
        best[mask] = np.maximum(best[mask], gain)

    has_gain = count > 0
    values = np.where(has_gain, best if aggregate == "max" else np.divide(total, np.maximum(count, 1)), np.nan)
    return values, has_gain


def figure_by_gain(
    method: str,
    coords: np.ndarray,
    masks: dict[str, np.ndarray],
    gains: dict[str, float],
    out_path: Path,
    n_columns: int,
    note: str = "",
) -> None:
    """Small multiples with each dataset painted a flat colour for its own gain.

    This is the honest version of "colour the points by contribution". Pooling
    the datasets into one scatter and colouring molecule-wise cannot work:
    every molecule in the qHTS deck belongs to all five isoform heads at once,
    so any per-molecule reduction either saturates (max) or averages the
    question away (mean). One panel per dataset keeps the gain attached to the
    thing it was actually measured on.
    """
    order = [g for g in _order_groups(masks, gains) if g not in _CHALLENGE_GROUPS]
    if not order:
        print(f"  no datasets with a gain -- skipping {out_path}", file=sys.stderr)
        return

    values = [gains.get(g.split("/", 1)[1]) for g in order]
    finite = [v for v in values if v is not None]
    ramp = matplotlib.colors.LinearSegmentedColormap.from_list("gain", _SEQUENTIAL)
    norm = matplotlib.colors.Normalize(vmin=min(finite + [0.0]), vmax=max(finite + [0.0]))

    n_rows = -(-len(order) // n_columns)
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(2.5 * n_columns, 2.7 * n_rows), squeeze=False)

    for position, (group, gain) in enumerate(zip(order, values)):
        ax = axes[position // n_columns][position % n_columns]
        mask = masks[group]
        _backdrop(ax, coords)

        holdout = masks.get("challenge/test")
        if holdout is not None and holdout.any():
            ax.scatter(
                coords[holdout, 0], coords[holdout, 1],
                s=2.0, c=_TEST, alpha=0.45, linewidth=0, rasterized=True, zorder=2,
            )

        color = "#bdbcb6" if gain is None else ramp(norm(gain))
        ax.scatter(coords[mask, 0], coords[mask, 1], s=3.0, color=color, linewidth=0, rasterized=True, zorder=3)
        _style_axis(ax, group, f"n={int(mask.sum())}" + (f"   {gain:+.3f}" if gain is not None else "   gain n/a"))

    for position in range(len(order), n_rows * n_columns):
        axes[position // n_columns][position % n_columns].axis("off")

    bar = fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=ramp),
        ax=axes.ravel().tolist(), fraction=0.015, pad=0.012,
    )
    bar.set_label("Macro RMSE improvement bought by this dataset's head", fontsize=9, color=_INK_SOFT)
    bar.outline.set_edgecolor("#e4e3de")

    # No legend box. A colorbar-bearing figure cannot use tight_layout, so
    # nothing reserves a bottom margin, and a floating legend lands on the
    # caption at whatever height this grid happens to take. The one non-ramp
    # colour is named in the caption instead, which keeps identity off colour
    # alone without an artist that has to be positioned blind.
    fig.suptitle(
        f"{method.upper()} — chemical space coloured by each dataset's contribution",
        fontsize=12, color=_INK,
    )
    fig.text(
        0.5, 0.001,
        "Every panel is the same map; only the highlighted dataset changes. Its colour is the RMSE that "
        "dataset's head bought as a solo addition in head-search, over the no-aux baseline. "
        "Grey = all molecules; orange = challenge/test, the scored holdout."
        + (f"\n{method.upper()}: {note}" if note else ""),
        ha="center", fontsize=8, color=_INK_SOFT,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}", file=sys.stderr)


def figure_gain_pooled(
    method: str,
    coords: np.ndarray,
    masks: dict[str, np.ndarray],
    per_head_gains: dict[str, dict[str, float]],
    scored_columns: list[str],
    aggregate: str,
    out_path: Path,
) -> None:
    """One panel per scored isoform: the map coloured by that head's own RMSE delta.

    This is the literal "contribution to the improvement on the 4 challenge
    isoforms" reading -- CYP1A2's panel shows which chemistry sits behind the
    heads that improved the CYP1A2 prediction specifically, which need not be
    the same chemistry that improved CYP3A4.
    """
    panels = []
    for head in scored_columns:
        gains = {c: g[head] for c, g in per_head_gains.items() if head in g}
        if gains:
            panels.append((head.split("_")[0], gains))
    if not panels:
        print(f"  head-search has no per-head RMSE -- skipping {out_path}", file=sys.stderr)
        return

    ramp = matplotlib.colors.LinearSegmentedColormap.from_list("gain", _SEQUENTIAL)
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 5.2), squeeze=False)

    pooled = [_gain_per_molecule(masks, g, coords.shape[0], aggregate)[0] for _, g in panels]
    finite = np.concatenate([v[~np.isnan(v)] for v in pooled]) if pooled else np.array([0.0])
    vmin, vmax = float(finite.min()), float(finite.max())

    points = None
    for column, ((label, _), values) in enumerate(zip(panels, pooled)):
        ax = axes[0][column]
        has_gain = ~np.isnan(values)

        # No-gain molecules are the scored challenge rows themselves: the thing
        # being improved, not a contributor.
        ax.scatter(coords[~has_gain, 0], coords[~has_gain, 1], s=2.0, c=_BACKDROP, linewidth=0, rasterized=True, zorder=1)
        ordering = np.argsort(np.where(has_gain, values, -np.inf))
        ordering = ordering[has_gain[ordering]]
        points = ax.scatter(
            coords[ordering, 0], coords[ordering, 1], c=values[ordering],
            cmap=ramp, vmin=vmin, vmax=vmax, s=3.0, linewidth=0, rasterized=True, zorder=2,
        )

        mask = masks.get("challenge/test")
        if mask is not None and mask.any():
            ax.scatter(
                coords[mask, 0], coords[mask, 1], s=8, facecolor="none",
                edgecolor=_TEST, linewidth=0.5, rasterized=True, label="challenge/test", zorder=3,
            )
        _style_axis(ax, f"{label} — what improved this head")
        legend = ax.legend(loc="upper right", fontsize=7.5, frameon=True, markerscale=2, labelcolor=_INK_SOFT)
        legend.get_frame().set_edgecolor("#e4e3de")

    bar = fig.colorbar(points, ax=axes.ravel().tolist(), fraction=0.02, pad=0.012)
    bar.set_label(
        ("Best" if aggregate == "max" else "Mean") + " RMSE improvement over this molecule's dataset heads",
        fontsize=9, color=_INK_SOFT,
    )
    bar.outline.set_edgecolor("#e4e3de")

    fig.suptitle(f"{method.upper()} — per-isoform contribution to each scored challenge head", fontsize=12, color=_INK)
    fig.text(
        0.5, 0.005,
        f"A molecule usually feeds several heads, so its colour is the {aggregate} over them; "
        "the per-dataset figure is the version that needs no such reduction.",
        ha="center", fontsize=8, color=_INK_SOFT,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}", file=sys.stderr)


# --------------------------------------------------------------------------


def write_embedding(
    out_path: Path,
    smiles: list[str],
    masks: dict[str, np.ndarray],
    coords_by_method: dict[str, np.ndarray],
) -> None:
    """Dump coordinates and group membership so the figures can be redrawn without recomputing."""
    order = sorted(masks)
    fieldnames = ["SMILES", "groups"] + [f"{m}_{axis}" for m in coords_by_method for axis in ("x", "y")]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for i, smi in enumerate(smiles):
            record = {"SMILES": smi, "groups": ";".join(g for g in order if masks[g][i])}
            for method, coords in coords_by_method.items():
                record[f"{method}_x"] = f"{coords[i, 0]:.4f}"
                record[f"{method}_y"] = f"{coords[i, 1]:.4f}"
            writer.writerow(record)
    print(f"Wrote {out_path} ({len(smiles)} molecules)", file=sys.stderr)


def read_embedding(path: Path) -> tuple[list[str], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Inverse of write_embedding: (smiles, {group: mask}, {method: (n, 2) coordinates})."""
    _, columns = read_table(path)
    smiles = columns["SMILES"]
    memberships = [set(g for g in groups.split(";") if g) for groups in columns["groups"]]

    groups = sorted(set().union(*memberships))
    masks = {g: np.array([g in m for m in memberships]) for g in groups}

    methods = [c[: -len("_x")] for c in columns if c.endswith("_x")]
    coords = {
        m: np.column_stack([np.asarray(columns[f"{m}_x"], dtype=float), np.asarray(columns[f"{m}_y"], dtype=float)])
        for m in methods
    }
    print(f"{path}: {len(smiles)} molecules, {len(groups)} datasets, projections {methods}", file=sys.stderr)
    return smiles, masks, coords


def main() -> None:
    """Entry point: project every dataset onto one map and draw both colourings."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-path", type=Path, default=Path("data/union_train.csv"))
    parser.add_argument("--test-path", type=Path, default=Path("data/test.csv"), help="Blind submission set (no labels)")
    parser.add_argument("--results-path", type=Path, default=Path("results/head_search.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    parser.add_argument(
        "--embedding-out",
        type=Path,
        default=Path("results/chemspace_embedding.csv"),
        help="Where to dump coordinates + group membership so figures can be redrawn; pass '' to skip",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["pca", "tsne", "umap"],
        choices=["pca", "tsne", "umap", "mds", "isomap"],
        help="Projections to run. umap is skipped with a warning if umap-learn is missing.",
    )
    parser.add_argument(
        "--gain-metric",
        default="macro",
        choices=["macro", "per-head"],
        help="'macro' colours by macro RMSE gain; 'per-head' draws one panel per scored isoform",
    )
    parser.add_argument(
        "--gain-aggregate",
        default="mean",
        choices=["mean", "max"],
        help="How to reduce several datasets' gains onto one molecule",
    )
    parser.add_argument("--scored-columns", nargs="+", default=_SCORED_DEFAULT)
    parser.add_argument("--pca-components", type=int, default=50, help="PCA dimensions fed to t-SNE/UMAP")
    parser.add_argument("--panel-columns", type=int, default=6, help="Columns in the small-multiples grid")
    parser.add_argument(
        "--max-molecules",
        type=int,
        default=0,
        help="Subsample to this many molecules before projecting (0 = all). Challenge and holdout "
        "molecules are always kept; only the auxiliary decks are thinned.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--from-embedding",
        type=Path,
        help="Redraw the gain-dependent figures from a saved --embedding-out dump instead of re-projecting. "
        "Use after a head-search re-run: no union, fingerprints or t-SNE needed. The overview is gain-free "
        "and is left alone.",
    )
    args = parser.parse_args()

    macro_gains, per_head_gains = load_gains(args.results_path, args.scored_columns)

    if args.from_embedding:
        _, masks, coords_by_method = read_embedding(args.from_embedding)
        notes = {m: _NOTES.get(m, "") for m in coords_by_method}
        notes["pca"] = "read relative position, not distance"
        draw_gain_figures(args, coords_by_method, masks, notes, macro_gains, per_head_gains)
        return

    smiles, masks, _ = collect_molecules(args.data_path, args.test_path, args.scored_columns)

    unmatched = sorted({g.split("/", 1)[1] for g in masks if g not in _CHALLENGE_GROUPS} - set(macro_gains))
    if unmatched:
        print(f"WARNING: datasets in the union with no head-search gain: {unmatched}", file=sys.stderr)

    print(f"{len(smiles)} unique molecules across {len(masks)} datasets", file=sys.stderr)
    print("Fingerprinting...", file=sys.stderr)
    fingerprints, kept = _fingerprint_matrix(smiles)
    smiles = [smiles[i] for i in kept]
    masks = {g: m[kept] for g, m in masks.items()}

    if args.max_molecules and len(smiles) > args.max_molecules:
        protected = np.zeros(len(smiles), dtype=bool)
        for group in _CHALLENGE_GROUPS:
            if group in masks:
                protected |= masks[group]
        budget = args.max_molecules - int(protected.sum())
        if budget <= 0:
            raise SystemExit(
                f"--max-molecules {args.max_molecules} is below the {int(protected.sum())} protected "
                "challenge molecules; raise it or drop the flag",
            )
        candidates = np.flatnonzero(~protected)
        picked = np.random.default_rng(args.seed).choice(candidates, size=min(budget, candidates.size), replace=False)
        keep = np.sort(np.concatenate([np.flatnonzero(protected), picked]))
        print(f"Subsampled to {keep.size} molecules ({int(protected.sum())} challenge molecules kept)", file=sys.stderr)
        smiles = [smiles[i] for i in keep]
        fingerprints = fingerprints[keep]
        masks = {g: m[keep] for g, m in masks.items()}

    coords_by_method, notes = embed(fingerprints, args.methods, args.seed, args.pca_components)
    if not coords_by_method:
        raise SystemExit("No projections were produced")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    figure_overview(coords_by_method, masks, notes, args.out_dir / "chemspace_overview.png")
    draw_gain_figures(args, coords_by_method, masks, notes, macro_gains, per_head_gains)

    # Path("") is Path("."), which is truthy and a directory -- an empty value
    # has to be checked as a string or the dump tries to write over the cwd.
    if str(args.embedding_out) not in ("", "."):
        write_embedding(args.embedding_out, smiles, masks, coords_by_method)


def draw_gain_figures(
    args: argparse.Namespace,
    coords_by_method: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    notes: dict[str, str],
    macro_gains: dict[str, float],
    per_head_gains: dict[str, dict[str, float]],
) -> None:
    """Draw every figure that depends on head-search gains, one set per projection."""
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for method, coords in coords_by_method.items():
        figure_by_isoform(
            method, coords, masks, macro_gains,
            args.out_dir / f"chemspace_by_isoform_{method}.png", args.panel_columns, notes.get(method, ""),
        )
        figure_by_gain(
            method, coords, masks, macro_gains,
            args.out_dir / f"chemspace_by_gain_{method}.png", args.panel_columns, notes.get(method, ""),
        )
        if args.gain_metric == "per-head":
            figure_gain_pooled(
                method, coords, masks, per_head_gains, args.scored_columns,
                args.gain_aggregate, args.out_dir / f"chemspace_gain_perhead_{method}.png",
            )


if __name__ == "__main__":
    main()
