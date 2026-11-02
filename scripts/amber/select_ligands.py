"""Write ligands.csv: 5 strong and 5 non-inhibitors of CYP3A4 from the challenge train set.

Inhibitors: pIC50 >= 6.5 (includes 4 heme-binding azoles + 1 non-azole sulfonamide control).
Non-inhibitors: pIC50 < 2.0 (mostly carboxylic acids; 1 triazole that does NOT inhibit).
Usage: python scripts/amber/select_ligands.py
"""
import csv
from pathlib import Path

COL = "CYP3A4_pIC50_direct_inhibition"
INHIB = ["OCNT-2310988", "OCNT-2312554", "OCNT-2395590", "OCNT-2309846", "OCNT-2310172"]
NONINHIB = ["OCNT-2313450", "OCNT-2317086", "OCNT-2308894", "OCNT-2313078", "OCNT-2316472"]

root = Path(__file__).resolve().parents[2]
rows = {r["Molecule_Name"]: r for r in csv.DictReader(open(root / "data/train.csv"))}
out = root / "scripts/amber/ligands.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["name", "label", "pIC50", "smiles"])
    for label, ids in (("inhibitor", INHIB), ("noninhibitor", NONINHIB)):
        for i in ids:
            p = float(rows[i][COL])
            assert (p >= 6.5) if label == "inhibitor" else (p < 2.0), (i, p)
            w.writerow([i, label, f"{p:.2f}", rows[i]["SMILES"]])
print("wrote", out)
