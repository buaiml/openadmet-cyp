# src/representations/cyp_pharmacophore.py
"""CYP-specific pharmacophore features: heme-coordination motifs, isoform
preference descriptors, and mechanism-based inactivation alerts.

The motif table below is the single source of truth. `src/data_tools/motif_report.py`
generates `reports/pharmacophore_features.md` from it, so the published report and
the computed features cannot drift, and a motif without a literature `source`
raises rather than emitting a blank cell.

Every SMARTS here was checked against named compounds before being committed;
see `tests/test_cyp_pharmacophore.py`. Sources are catalogued in
`docs/superpowers/specs/2026-10-08-cyp-pharmacophore-features-design.md` section 12.

DO NOT add a motif because it improved a validation score. A motif added that way
is a fitted parameter. Fix the list from the literature first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, ClassVar

import numpy as np
import polars as pl
from rdkit import Chem
from tqdm import tqdm

from representations.base import Representation

# Bump when any motif's SMARTS, order, or membership changes. The registered
# name carries this version, so the content-addressed cache in
# `data_tools.inputs._featurize_one` cannot serve stale columns under a new table.
TABLE_VERSION = "v1"


@dataclass(frozen=True)
class Motif:
    """One SMARTS-counted substructure feature."""

    name: str
    smarts: str
    block: str  # "heme" | "tdi"
    source: str  # literature citation; empty blocks the report generator
    rollup: str = ""  # T6 donor label, for the 2D/3D comparison
    note: str = ""


# Donor nitrogen: aromatic, two-connected, no hydrogen. Covers pyridine,
# pyrimidine, imidazole N3, triazole/tetrazole N, thiazole N, oxazole N.
_DONOR = "[nX2H0]"
# Hindered = the donor has a substituted (or ring-fused) aromatic neighbour.
_DONOR_HINDERED = "[nX2H0;$(n:[cH0])]"
_DONOR_UNHINDERED = "[nX2H0;!$(n:[cH0])]"

_MONK_2020 = (
    "Monk et al. 2020, Biochim Biophys Acta Proteins Proteom — azoles carry an "
    "imidazole, triazole or tetrazole whose ring nitrogen coordinates as the "
    "sixth axial ligand to the heme iron."
)
_MOHAMED_2023 = (
    "Mohamed et al. 2023, J Inorg Biochem — imidazole- and pyridine-based ligands "
    "coordinate directly to ferric heme-thiolate as type II nitrogen donors."
)
_PERRIN_2009 = (
    "Perrin et al. 2009, Metallomics — inhibition potency depends on accessibility "
    "of the imidazolyl nitrogen lone pair, tuned by steric hindrance near the ring."
)
_PODGORSKI_2020 = (
    "Podgorski et al. 2020, Biochemistry — crystal structures: 4-(pyridin-3-yl)"
    "benzoic acid coordinates the heme iron; 4-(pyridin-2-yl)benzoic acid does not."
)
_PAINE_2003 = (
    "Paine et al. 2003, J Biol Chem — CYP2D6 substrates typically contain a basic "
    "nitrogen atom; Glu-216 and Asp-301 define that specificity."
)
_MIRZAEI_2021 = (
    "Mirzaei et al. 2021, Chem Res Toxicol — review of mechanism-based "
    "inactivation latent groups: thiophene, furan, alkylamines, terminal acetylene."
)
_MAO_2025 = (
    "Mao 2025, J Pharm Anal — furan, thiophene, acetylene and methylenedioxyphenyl "
    "drive CYP3A4 mechanism-based inactivation; MDP, alkylamine and hydrazine drive "
    "quasi-irreversible inactivation."
)
_DANG_2017 = (
    "Dang et al. 2017, Chem Res Toxicol — furan and thiophene alerts are highly "
    "conditional: each can form reactive metabolites through certain pathways but "
    "not always. Expect weak signal on direct-inhibition heads."
)

_MOTIFS: tuple[Motif, ...] = (
    # --- ring systems bearing a coordinating nitrogen ---
    Motif("heme_imidazole", "[nX2H0]1c[nX3]cc1", "heme", _MONK_2020, "imidazole"),
    Motif("heme_triazole_124", "n1ncnc1", "heme", _MONK_2020, "triazole",
          "1,2,4-triazole ring; ring count, not donor count"),
    Motif("heme_triazole_123", "n1nncc1", "heme", _MONK_2020, "triazole"),
    Motif("heme_tetrazole", "n1nnnc1", "heme", _MONK_2020, "triazole",
          "Tetrazole nitrogen also ligates heme (Monk 2020)."),
    Motif("heme_pyridine", "n1ccccc1", "heme", _MOHAMED_2023, "pyridine"),
    Motif("heme_pyrimidine", "n1cnccc1", "heme", _MOHAMED_2023, "pyridine"),
    Motif("heme_thiazole", "s1cncc1", "heme", _MONK_2020, "imidazole",
          "Five-membered azole; donor N resembles imidazole N3 more than pyridine."),
    Motif("heme_oxazole", "o1cncc1", "heme", _MONK_2020, "imidazole"),
    # --- donor accessibility ---
    Motif("heme_donor_total", _DONOR, "heme", _MOHAMED_2023, "",
          "All aromatic sp2 donor nitrogens, hindered or not."),
    Motif("heme_donor_unhindered", _DONOR_UNHINDERED, "heme", _PODGORSKI_2020, "",
          "Encodes the steric half of lone-pair accessibility only; Perrin 2009 "
          "shows intramolecular H-bonding also tunes it, and that is not captured."),
    Motif("heme_donor_hindered", _DONOR_HINDERED, "heme", _PERRIN_2009, ""),
    # --- aliphatic amines ---
    Motif("heme_amine_primary",
          "[NX3;H2;!$(NC=O);!$(NS(=O)=O);!$(N[a]);!$(N=*)]",
          "heme", _PAINE_2003, "aliphatic amine",
          "Excludes amides, sulfonamides and anilines, which are not donors."),
    Motif("heme_amine_basic",
          "[NX3;H0,H1;!$(NC=O);!$(NS(=O)=O);!$(N[a]);!$(N=*);!$(N#*)]",
          "heme", _PAINE_2003, "aliphatic amine",
          "Secondary and tertiary basic aliphatic amines."),
    # --- mechanism-based inactivation (time-dependent, not direct, inhibition) ---
    Motif("tdi_mdp", "c1ccc2c(c1)OCO2", "tdi", _MAO_2025, "",
          "Methylenedioxyphenyl / benzodioxole."),
    Motif("tdi_terminal_alkyne", "[CH1X2]#[CX2]", "tdi", _MIRZAEI_2021, "",
          "Terminal only; internal alkynes are not alerts."),
    Motif("tdi_furan", "o1cccc1", "tdi", _DANG_2017, ""),
    Motif("tdi_thiophene", "s1cccc1", "tdi", _DANG_2017, ""),
    Motif("tdi_hydrazine", "[NX3;!$(N=*)][NX3;!$(N=*)]", "tdi", _MAO_2025, "",
          "Also matches hydrazides, which Mao 2025 includes."),
    Motif("tdi_alkylamine_tertiary",
          "[NX3;H0;!$(NC=O);!$(NS(=O)=O);!$(N[a]);!$(N=*);!$(N#*)]",
          "tdi", _MIRZAEI_2021, "",
          "Tertiary alkylamines; quasi-irreversible via nitroso/iminium."),
)


def _parse(smi: str | None) -> Chem.Mol | None:
    """Return a usable molecule, or None.

    Chem.MolFromSmiles("") returns a VALID zero-atom molecule rather than None,
    which would silently produce an all-zeros feature row indistinguishable from
    a real molecule with no matches. Both guards are required.
    """
    if smi is None or not smi.strip():
        return None
    mol = Chem.MolFromSmiles(smi.strip())
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    return mol


class _SmartsCountBlock(Representation):
    """Counts each motif in one block. Invalid SMILES give an all-NaN row;
    a valid molecule with no matches gives a zeros row."""

    block: ClassVar[str] = ""
    _desc: ClassVar[str] = "SMARTS motifs"

    @property
    def motifs(self) -> list[Motif]:
        return [m for m in _MOTIFS if m.block == self.block]

    @property
    def feature_names(self) -> list[str]:
        return [m.name for m in self.motifs]

    def transform(self, smiles: pl.Series) -> np.ndarray:
        patterns = [Chem.MolFromSmarts(m.smarts) for m in self.motifs]
        out = np.full((len(smiles), len(patterns)), np.nan, dtype=np.float64)
        for i, smi in enumerate(tqdm(smiles.to_list(), desc=self._desc, unit="mol", leave=True)):
            mol = _parse(smi)
            if mol is None:
                continue
            out[i] = [float(len(mol.GetSubstructMatches(p))) for p in patterns]
        return out


class HemeCoordinationMotifs(_SmartsCountBlock):
    """Substructure counts for nitrogens able to ligate the heme iron, plus a
    steric-accessibility split. The 2D control for round-1 T6."""

    name: ClassVar[str] = f"cyp_heme_{TABLE_VERSION}"
    block: ClassVar[str] = "heme"
    _desc: ClassVar[str] = "heme motifs"


class MechanismBasedAlerts(_SmartsCountBlock):
    """Structural alerts for time-dependent inactivation.

    Emitted as a separate registered block because these concern TDI rather than
    direct inhibition, so little signal is expected on the four scored heads.
    Round-1 T1 adds TDI heads and will want them.
    """

    name: ClassVar[str] = f"cyp_tdi_{TABLE_VERSION}"
    block: ClassVar[str] = "tdi"
    _desc: ClassVar[str] = "TDI alerts"
