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
import urllib.request
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


def add_chain_breaks(lines, max_cn=2.0):
    """Insert TER where consecutive residues are not bonded (C..N > max_cn A), so tleap does not
    bridge the gaps in 1TQN with a stretched peptide bond; gap ends get ordinary termini."""
    res = []  # (key, [lines])
    for l in lines:
        key = (l[22:27], l[17:20])
        if not res or res[-1][0] != key:
            res.append((key, []))
        res[-1][1].append(l)

    def xyz(grp, name):
        for l in grp:
            if l[12:16].strip() == name:
                return np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])
        return None

    out = []
    for i, (key, grp) in enumerate(res):
        if i:
            c, n_ = xyz(res[i - 1][1], "C"), xyz(grp, "N")
            if c is not None and n_ is not None and np.linalg.norm(c - n_) > max_cn:
                out.append("TER\n")
        out.extend(grp)
    return out


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
    # tleap addresses residues by PDB number, and 1TQN numbering has gaps: renumber 1..N so the
    # index used in the tleap `bond` command equals the residue's position in the chain.
    def renumber(lns, start):
        out, prev, k = [], None, start - 1
        for l in lns:
            if l.startswith("TER"):
                out.append(l)
                continue
            key = (l[22:27], l[17:20])
            if key != prev:
                k += 1
                prev = key
            out.append(f"{l[:22]}{k:4d} {l[27:]}")
        return out, k

    prot_r, last = renumber(add_chain_breaks(prot), 1)
    hem_r, _ = renumber(hem, last + 1)
    with open(dst, "w") as f:
        f.writelines(prot_r + ["TER\n"] + hem_r + ["END\n"])
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


def clearance_grid(heavy_prot, center, half=24.0, step=0.75):
    """Distance from each grid point to the nearest protein/heme heavy atom."""
    ax = np.arange(-half, half + 1e-6, step)
    G = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1) + center
    flat = G.reshape(-1, 3)
    d = np.empty(len(flat))
    for i in range(0, len(flat), 2000):
        d[i:i + 2000] = np.linalg.norm(flat[i:i + 2000, None] - heavy_prot[None], axis=2).min(1)
    return d.reshape(G.shape[:3]), ax, center


def place(mol, heavy_prot, fe, n, seed=0, n_conf=30, n_pose=20000, want=3.0, accept=2.0):
    """Clash-minimizing rigid-body placement in the distal pocket.

    Samples conformers x random rotations x centre-of-mass positions on free grid cells
    near Fe, scores each pose by a soft clash penalty from a precomputed clearance grid,
    and keeps the best. Fails only if the best pose still has a heavy atom < `accept` A
    from the protein (minimization removes the remaining mild overlaps).
    """
    rng = np.random.default_rng(seed)
    cids = list(AllChem.EmbedMultipleConfs(mol, n_conf, randomSeed=seed))
    if not cids:
        sys.exit("RDKit could not embed the ligand")
    AllChem.MMFFOptimizeMoleculeConfs(mol)
    heavy_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    grid, ax, c0 = clearance_grid(heavy_prot, fe)
    step = ax[1] - ax[0]
    gshape = np.array(grid.shape)

    def clear_at(P):  # P (..., 3) -> clearance, nearest grid cell
        idx = np.rint((P - c0 - ax[0]) / step).astype(int)
        inside = ((idx >= 0) & (idx < gshape)).all(-1)
        idx = np.clip(idx, 0, gshape - 1)
        return np.where(inside, grid[idx[..., 0], idx[..., 1], idx[..., 2]], 10.0)

    # candidate centre-of-mass sites: free cells within 11 A of Fe, biased to the distal side
    cells = np.argwhere(grid >= 2.5)
    sites = cells * step + ax[0] + c0
    rel = sites - fe
    keep = (np.linalg.norm(rel, axis=1) < 11.0) & (rel @ n > 0.0)
    sites = sites[keep]
    if len(sites) == 0:
        sys.exit("no free space on the distal side of the heme in this structure")
    target = fe + 6.0 * n

    best = (np.inf, None)
    for cid in cids:
        X = mol.GetConformer(cid).GetPositions()
        Xh = X[heavy_idx] - X[heavy_idx].mean(0)
        Rs = np.stack([rand_rot(rng) for _ in range(n_pose)])            # (N,3,3)
        com = sites[rng.integers(0, len(sites), n_pose)]                  # (N,3)
        P = np.einsum("nij,aj->nai", Rs, Xh) + com[:, None, :]            # (N,A,3)
        c = clear_at(P)                                                   # (N,A)
        pen = (np.clip(want - c, 0, None) ** 2).sum(1) + 0.02 * np.linalg.norm(com - target, axis=1)
        k = int(pen.argmin())
        if pen[k] < best[0]:
            best = (pen[k], (cid, Rs[k], com[k], c[k].min()))
    cid, R, com, cmin = best[1]
    if cmin < accept:
        sys.exit(f"no acceptable placement: best pose has a heavy atom {cmin:.2f} A from the protein "
                 f"(need >= {accept}); the pocket may be too closed - try another structure")
    print(f"placed ligand: conformer {cid}, closest protein contact {cmin:.2f} A")
    X = mol.GetConformer(cid).GetPositions()
    X = (X - X[heavy_idx].mean(0)) @ R.T + com
    conf = mol.GetConformer(cid)
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, X[i].tolist())
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
    pdb = Path(a.pdb)
    if not pdb.exists():  # data/ is git-ignored, so fetch the crystal structure on first use
        pdb.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://files.rcsb.org/download/{pdb.stem}.pdb"
        print(f"fetching {url}")
        urllib.request.urlretrieve(url, pdb)
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    heavy, fe, n = clean_pdb(pdb, out / "protein_heme.pdb")
    mol = Chem.AddHs(Chem.MolFromSmiles(a.smiles))
    charge = Chem.GetFormalCharge(mol)
    cid = place(mol, heavy, fe, n, a.seed)
    Chem.MolToMolFile(mol, str(out / "lig_placed.mol"), confId=cid)  # keeps bond orders (PDB loses them)

    run(["antechamber", "-i", "lig_placed.mol", "-fi", "mdl", "-o", "lig.mol2", "-fo", "mol2",
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
bond prot.CYSRES.SG prot.HEMRES.FE
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
    cys_idx = [i for i, k in enumerate(seen, 1) if k[1] == "CYP"][0]
    leap = leap.replace("HEMRES", str(hem_idx)).replace("CYSRES", str(cys_idx))
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
