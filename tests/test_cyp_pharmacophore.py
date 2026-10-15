# tests/test_cyp_pharmacophore.py
"""Synthetic contract checks for the CYP pharmacophore blocks, not accuracy tests."""

import numpy as np
import polars as pl
import pytest

from representations.cyp_pharmacophore import (
    _MOTIFS,
    HemeCoordinationMotifs,
)

HEME = HemeCoordinationMotifs()


def _row(block, smiles):
    """Return the single feature row for one SMILES as a dict keyed by column name."""
    arr = block.transform(pl.Series([smiles]))
    return dict(zip(block.feature_names, arr[0]))


def test_every_motif_smarts_compiles():
    from rdkit import Chem

    bad = [m.name for m in _MOTIFS if Chem.MolFromSmarts(m.smarts) is None]
    assert bad == []


def test_every_motif_rollup_is_a_t6_donor_label():
    allowed = {"imidazole", "triazole", "pyridine", "aliphatic amine"}
    for m in _MOTIFS:
        if m.block == "heme" and m.rollup:
            assert m.rollup in allowed, f"{m.name} has rollup {m.rollup!r}"


def test_feature_names_match_matrix_width():
    arr = HEME.transform(pl.Series(["c1ccccc1"]))
    assert arr.shape == (1, len(HEME.feature_names))


# --- the Podgorski pair: the sharpest available test of the hindrance term ---
# 4-(pyridin-3-yl)benzoic acid coordinates heme iron; 4-(pyridin-2-yl)benzoic
# acid does not (an aqua ligand remains). Crystallographically resolved.
# Podgorski et al. 2020, Biochemistry — spec section 12.1.
def test_podgorski_pair_separates_on_hindrance():
    coordinates = _row(HEME, "OC(=O)c1ccc(cc1)-c1cccnc1")   # pyridin-3-yl
    does_not = _row(HEME, "OC(=O)c1ccc(cc1)-c1ccccn1")      # pyridin-2-yl

    assert coordinates["heme_donor_unhindered"] == 1.0
    assert coordinates["heme_donor_hindered"] == 0.0

    assert does_not["heme_donor_unhindered"] == 0.0
    assert does_not["heme_donor_hindered"] == 1.0

    # both are pyridines; only accessibility differs
    assert coordinates["heme_pyridine"] == does_not["heme_pyridine"] == 1.0


def test_ortho_disubstituted_pyridine_is_hindered():
    r = _row(HEME, "Cc1cccc(C)n1")  # 2,6-dimethylpyridine
    assert r["heme_donor_hindered"] == 1.0
    assert r["heme_donor_unhindered"] == 0.0


def test_known_type_ii_inhibitors_have_an_unhindered_donor():
    keto = _row(HEME, "CC(=O)N1CCN(CC1)c1ccc(OC[C@H]2CO[C@@](Cn3ccnc3)(O2)c2ccc(Cl)cc2Cl)cc1")
    assert keto["heme_imidazole"] == 1.0
    assert keto["heme_donor_unhindered"] >= 1.0

    flu = _row(HEME, "OC(Cn1cncn1)(Cn1cncn1)c1ccc(F)cc1F")
    assert flu["heme_triazole_124"] == 2.0
    assert flu["heme_donor_unhindered"] == 4.0


def test_tetrazole_is_counted():
    losartan = _row(HEME, "CCCCc1nc(Cl)c(CO)n1Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1")
    assert losartan["heme_tetrazole"] == 1.0


def test_pyrazole_is_not_counted_as_imidazole():
    assert _row(HEME, "c1cc[nH]n1")["heme_imidazole"] == 0.0
    assert _row(HEME, "c1c[nH]cn1")["heme_imidazole"] == 1.0


def test_primary_aliphatic_amine_counted_but_not_amide_or_aniline():
    assert _row(HEME, "CC(N)Cc1ccccc1")["heme_amine_primary"] == 1.0   # amphetamine
    assert _row(HEME, "CC(N)=O")["heme_amine_primary"] == 0.0          # acetamide
    assert _row(HEME, "Nc1ccccc1")["heme_amine_primary"] == 0.0        # aniline


# --- Review Focus items ---
def test_no_match_molecule_gives_zeros_not_nan():
    """Critical: NaN here would make drop_nan_rows delete every non-azole molecule."""
    arr = HEME.transform(pl.Series(["c1ccccc1"]))
    assert not np.isnan(arr).any()
    assert (arr == 0).all()


@pytest.mark.parametrize("bad", [None, "", "   ", "not_a_smiles"])
def test_invalid_null_and_blank_smiles_give_all_nan(bad):
    """MolFromSmiles('') returns a VALID zero-atom mol, so 'mol is not None' is not enough."""
    arr = HEME.transform(pl.Series([bad], dtype=pl.Utf8))
    assert np.isnan(arr).all()


def test_empty_input_returns_zero_rows_with_full_width():
    arr = HEME.transform(pl.Series([], dtype=pl.Utf8))
    assert arr.shape == (0, len(HEME.feature_names))


def test_duplicate_smiles_return_one_row_each_in_input_order():
    arr = HEME.transform(pl.Series(["c1ccccc1", "c1ccncc1", "c1ccccc1"]))
    assert arr.shape[0] == 3
    np.testing.assert_array_equal(arr[0], arr[2])
    assert arr[1][HEME.feature_names.index("heme_pyridine")] == 1.0


def test_transform_is_deterministic():
    s = pl.Series(["c1ccncc1", "CC(N)Cc1ccccc1"])
    np.testing.assert_array_equal(HEME.transform(s), HEME.transform(s))


def test_dtype_is_float64():
    assert HEME.transform(pl.Series(["c1ccncc1"])).dtype == np.float64


# --- mechanism-based inactivation (cyp_tdi_v1) ---
from representations.cyp_pharmacophore import MechanismBasedAlerts

TDI = MechanismBasedAlerts()


def test_tdi_block_is_separate_from_heme_block():
    assert set(TDI.feature_names).isdisjoint(HEME.feature_names)
    assert TDI.name != HEME.name


def test_methylenedioxyphenyl_alert():
    assert _row(TDI, "C=CCc1ccc2OCOc2c1")["tdi_mdp"] == 1.0  # safrole
    assert _row(TDI, "Fc1ccc(cc1)[C@@H]1CCNC[C@H]1COc1ccc2OCOc2c1")["tdi_mdp"] == 1.0  # paroxetine
    assert _row(TDI, "c1ccccc1")["tdi_mdp"] == 0.0


def test_terminal_alkyne_alert_excludes_internal_alkynes():
    ethinyl = "C[C@]12CC[C@H]3[C@@H](CCc4cc(O)ccc34)[C@@H]1CC[C@]2(O)C#C"
    assert _row(TDI, ethinyl)["tdi_terminal_alkyne"] == 1.0  # ethinylestradiol
    assert _row(TDI, "CC#C")["tdi_terminal_alkyne"] == 1.0   # propyne
    assert _row(TDI, "CC#CC")["tdi_terminal_alkyne"] == 0.0  # 2-butyne, internal


def test_furan_and_thiophene_alerts():
    assert _row(TDI, "c1ccoc1")["tdi_furan"] == 1.0
    assert _row(TDI, "Clc1ccccc1CN1CCc2sccc2C1")["tdi_thiophene"] == 1.0  # ticlopidine
    assert _row(TDI, "c1ccccc1")["tdi_furan"] == 0.0


def test_hydrazine_alert():
    assert _row(TDI, "NNC(=O)c1ccncc1")["tdi_hydrazine"] == 1.0  # isoniazid


def test_tdi_no_match_gives_zeros_and_invalid_gives_nan():
    assert not np.isnan(TDI.transform(pl.Series(["c1ccccc1"]))).any()
    assert np.isnan(TDI.transform(pl.Series([""], dtype=pl.Utf8))).all()


# --- isoform-preference descriptors (cyp_isoform_v1) ---
from representations.cyp_pharmacophore import (
    _ISOFORM_FEATURES,
    IsoformPreferenceDescriptors,
)

ISO = IsoformPreferenceDescriptors()


def test_every_isoform_descriptor_is_cited():
    uncited = [d.name for d in _ISOFORM_FEATURES if not d.source.strip()]
    assert uncited == []


def test_isoform_width_matches_names():
    arr = ISO.transform(pl.Series(["c1ccncc1"]))
    assert arr.shape == (1, len(ISO.feature_names))


def test_basic_amine_count_tracks_basicity():
    """CYP2D6 ligands characteristically carry a basic nitrogen (Paine 2003)."""
    assert _row(ISO, "CC(C)NCC(O)COc1cccc2ccccc12")["iso_basic_amine_count"] == 1.0  # propranolol
    assert _row(ISO, "c1ccccc1")["iso_basic_amine_count"] == 0.0


def test_acidic_group_count_tracks_anionic_ligands():
    """CYP2C9 prefers anionic ligands via Arg108 (Wester 2004, Dickmann 2004)."""
    assert _row(ISO, "CC(C)Cc1ccc(cc1)C(C)C(=O)O")["iso_acidic_group_count"] == 1.0  # ibuprofen
    assert _row(ISO, "c1ccccc1")["iso_acidic_group_count"] == 0.0


def test_aromatic_fraction_is_a_planarity_proxy():
    """CYP1A2's cavity is adapted to large planar substrates (Sansen 2007)."""
    flat = _row(ISO, "c1ccc2ccccc2c1")["iso_aromatic_frac"]      # naphthalene
    saturated = _row(ISO, "CC1CCCCCCC1")["iso_aromatic_frac"]    # cyclooctane
    assert flat == 1.0
    assert saturated == 0.0


def test_size_and_lipophilicity_present_and_finite():
    r = _row(ISO, "CC(C)NCC(O)COc1cccc2ccccc12")
    assert 250.0 < r["iso_mw"] < 270.0
    assert np.isfinite(r["iso_clogp"])
    assert r["iso_aromatic_rings"] == 2.0


def test_isoform_invalid_smiles_gives_all_nan():
    for bad in [None, "", "   ", "not_a_smiles"]:
        assert np.isnan(ISO.transform(pl.Series([bad], dtype=pl.Utf8))).all()


def test_isoform_hbond_and_polarity_columns_are_not_transposed():
    """iso_hba and iso_hbd are asymmetric for propranolol, so a swap is caught."""
    prop = _row(ISO, "CC(C)NCC(O)COc1cccc2ccccc12")
    assert prop["iso_hba"] == 3.0
    assert prop["iso_hbd"] == 2.0
    assert prop["iso_tpsa"] == pytest.approx(41.49, abs=0.01)
    assert prop["iso_fraction_csp3"] == pytest.approx(0.375, abs=0.001)

    flat = _row(ISO, "c1ccccc1")
    assert flat["iso_hba"] == 0.0
    assert flat["iso_hbd"] == 0.0
    assert flat["iso_tpsa"] == 0.0
    assert flat["iso_fraction_csp3"] == 0.0


from representations.cyp_pharmacophore import ERG_N_FEATURES, ErGFingerprint

ERG = ErGFingerprint()


def test_erg_width_is_315():
    assert ERG_N_FEATURES == 315
    arr = ERG.transform(pl.Series(["CC(C)NCC(O)COc1cccc2ccccc12"]))
    assert arr.shape == (1, 315)
    assert len(ERG.feature_names) == 315


def test_erg_is_not_all_zero_for_a_real_molecule():
    arr = ERG.transform(pl.Series(["CC(C)NCC(O)COc1cccc2ccccc12"]))
    assert (arr != 0).sum() > 0


def test_erg_invalid_smiles_gives_all_nan():
    for bad in [None, "", "   ", "not_a_smiles"]:
        assert np.isnan(ERG.transform(pl.Series([bad], dtype=pl.Utf8))).all()


def test_erg_empty_input_keeps_full_width():
    assert ERG.transform(pl.Series([], dtype=pl.Utf8)).shape == (0, 315)


def test_erg_is_deterministic():
    s = pl.Series(["CC(C)NCC(O)COc1cccc2ccccc12"])
    np.testing.assert_array_equal(ERG.transform(s), ERG.transform(s))
