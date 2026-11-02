"""Build a solvated CYP3A4 (1TQN) + ligand Amber system.

Steps: clean protein/heme from 1TQN -> RDKit-embed ligand and place it in the distal
pocket (rigid-body search, clash-free) -> antechamber GAFF2/AM1-BCC -> tleap.

Needs on PATH: antechamber, parmchk2, tleap, ambmask (AmberTools). Python: numpy, rdkit.
Heme parameters (not in AmberTools): Shahrokh et al. 2012 IC6 set (HEM.mol2, CYP.mol2, IC6.frcmod)
in scripts/amber/heme_params/IC6, or pass --heme-dir.

Usage: python build_system.py --name OCNT-2310988 --smiles '<smiles>' --out DIR \
         [--heme-dir DIR] [--pdb data/proteins/1TQN.pdb]
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

CYS_RES = 442  # heme-ligating Cys in 1TQN (verified below by Fe-SG distance)


def run(cmd, cwd):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"FAILED: {' '.join(cmd)}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return r.stdout


def clean_pdb(src, dst):
    """Chain A protein + HEM, no waters/altlocs. Cys442 -> CYP. Returns (xyz heavy, Fe, normal, n_res)."""
    prot, hem = [], []
    for ln in open(src):
        if ln[:6] not in ("ATOM  ", "HETATM") or ln[21] != "A":
            continue
        if ln[16] not in " A":
            continue
        res = ln[17:20]
        if res == "HOH":
            continue
        (hem if res == "HEM" else prot).append(ln)
    prot = [ln.replace("CYS A 442", "CYP A 442") if int(ln[22:26]) == CYS_RES else ln for ln in prot]
    with open(dst, "w") as f:
        f.writelines(prot + ["TER\n"] + hem + ["END\n"])
    xyz = lambda lns: np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in lns])
    names = lambda lns: [l[12:16].strip() for l in lns]
    hx, hn = xyz(hem), names(hem)
    fe = hx[hn.index("FE")]
    sg = xyz([l for l in prot if int(l[22:26]) == CYS_RES and l[12:16].strip() == "SG"])[0]
    assert np.linalg.norm(fe - sg) < 2.8, "Cys442 SG not bound to Fe"
    pyr = hx[[hn.index(n) for n in ("NA", "NB", "NC", "ND")]]
    n = np.cross(pyr[1] - pyr[0], pyr[2] - pyr[1])
    n /= np.linalg.norm(n)
    if np.dot(n, sg - fe) > 0:  # point to the distal side (away from Cys)
        n = -n
    heavy = np.array([x for x, l in zip(xyz(prot), prot) if l[76:78].strip() != "H"])
    return heavy, fe, n


def rand_rot(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    a, b, c, d = q
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a-b*b+c*c-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a-b*b-c*c+d*d]])


def place(mol, heavy_prot, fe, n, seed=0, n_conf=20, n_try=4000):
    """Find clash-free pose with COM near Fe + 7 A along distal normal. Heme heavy atoms are in heavy_prot."""
    rng = np.random.default_rng(seed)
    cids = AllChem.EmbedMultipleConfs(mol, n_conf, randomSeed=seed)
    AllChem.MMFFOptimizeMoleculeConfs(mol)
    heavy_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    target = fe + 7.0 * n
    best = None  # (score, cid, R, shift)
    for thr in (3.0, 2.6, 2.2):
        for cid in cids:
            X = mol.GetConformer(cid).GetPositions()
            Xc = X - X[heavy_idx].mean(0)
            for _ in range(n_try // n_conf):
                R = rand_rot(rng)
                com = target + rng.uniform(-3, 3, 3)
                P = Xc[heavy_idx] @ R.T + com
                d = np.linalg.norm(P[:, None] - heavy_prot[None], axis=2).min()
                if d < thr:
                    continue
                score = np.linalg.norm(com - target)
                if best is None or score < best[0]:
                    best = (score, cid, R, com, Xc)
        if best:
            break
    if best is None:
        sys.exit("no clash-free placement; pocket may be too closed - increase sampling or pick another structure")
    _, cid, R, com, Xc = best
    conf = mol.GetConformer(cid)
    newpos = Xc @ R.T + com
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, newpos[i].tolist())
    return cid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--smiles", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pdb", default=str(Path(__file__).resolve().parents[2] / "data/proteins/1TQN.pdb"))
    ap.add_argument("--heme-dir", default=str(Path(__file__).resolve().parent / "heme_params/IC6"),
                    help="dir with HEM.mol2, CYP.mol2, IC6.frcmod (Shahrokh 2012, ferric high-spin penta-coordinate)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    heavy, fe, n = clean_pdb(a.pdb, out / "protein_heme.pdb")
    mol = Chem.AddHs(Chem.MolFromSmiles(a.smiles))
    charge = Chem.GetFormalCharge(mol)
    cid = place(mol, heavy, fe, n, a.seed)
    Chem.MolToPDBFile(mol, str(out / "lig_placed.pdb"), confId=cid)

    run(["antechamber", "-i", "lig_placed.pdb", "-fi", "pdb", "-o", "lig.mol2", "-fo", "mol2",
         "-c", "bcc", "-nc", str(charge), "-at", "gaff2", "-rn", "LIG", "-pf", "y"], out)
    run(["parmchk2", "-i", "lig.mol2", "-f", "mol2", "-o", "lig.frcmod", "-s", "gaff2"], out)

    # residue count of protein part (CYP counts as one residue): heme is last -> not needed by number, use mask
    heme = Path(a.heme_dir).resolve()
    for f in ("HEM.mol2", "CYP.mol2", "IC6.frcmod"):
        assert (heme / f).exists(), f"missing {heme / f}"
    leap = f"""source leaprc.protein.ff14SB
source leaprc.gaff2
source leaprc.water.tip3p
loadamberparams {heme}/IC6.frcmod
HEM = loadmol2 {heme}/HEM.mol2
CYP = loadmol2 {heme}/CYP.mol2
set CYP head CYP.1.N
set CYP tail CYP.1.C
loadamberparams lig.frcmod
LIG = loadmol2 lig.mol2
prot = loadpdb protein_heme.pdb
bond prot.{CYS_RES}.SG prot.HEMRES.FE
sys = combine {{prot LIG}}
solvateoct sys TIP3PBOX 10.0
addIonsRand sys Na+ 0
addIonsRand sys Cl- 0
saveamberparm sys system.parm7 system.rst7
savepdb sys system.pdb
quit
"""
    # HEM residue number = last protein residue + 1; find from the cleaned PDB
    seen = []
    for ln in open(out / "protein_heme.pdb"):
        if ln[:6] in ("ATOM  ", "HETATM"):
            k = (ln[22:26], ln[17:20])
            if not seen or seen[-1] != k:
                seen.append(k)
    hem_idx = [i for i, k in enumerate(seen, 1) if k[1] == "HEM"][0]
    leap = leap.replace("HEMRES", str(hem_idx))
    (out / "tleap.in").write_text(leap)
    run(["tleap", "-f", "tleap.in"], out)
    assert (out / "system.parm7").exists(), "tleap produced no topology"

    # atom numbers for the pull restraint (COM of ligand heavy atoms vs Fe)
    def atoms(mask):
        res = run(["ambmask", "-p", "system.parm7", "-c", "system.rst7", "-find", mask], out)
        return [int(l[6:11]) for l in res.splitlines() if l.startswith(("ATOM", "HETATM"))]

    lig = atoms(":LIG & !@H=")
    fe_i = atoms(":HEM@FE")
    assert lig and len(fe_i) == 1, (len(lig), fe_i)
    (out / "pull.disang").write_text(
        " &rst iat=-1,-1, igr1=" + ",".join(map(str, lig)) + f", igr2={fe_i[0]},\n"
        "   r1=0.0, r2=0.0, r3=6.0, r4=7.0, rk2=10.0, rk3=10.0, /\n")
    print(f"built {out}  lig_heavy={len(lig)} fe={fe_i[0]} charge={charge}")


if __name__ == "__main__":
    main()
