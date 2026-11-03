#!/bin/bash
# Run one ligand end to end: build -> minimize -> heat -> equil -> pull (restraint on) -> release (free) -> analysis.
# Usage: run_md.sh NAME SMILES OUTROOT
set -euo pipefail
NAME=$1; SMILES=$2; ROOT=$3
D="$ROOT/$NAME"; mkdir -p "$D"
HERE="$(cd "$(dirname "$0")" && pwd)"
HEME_DIR="${HEME_DIR:-$HERE/heme_params/IC6}"
PMEMD="${PMEMD:-pmemd.cuda}"

[ -f "$D/pull.disang" ] || python "$HERE/build_system.py" --name "$NAME" --smiles "$SMILES" --out "$D" \
    --heme-dir "$HEME_DIR"
cd "$D"


cat > min1.in <<EOF
min1: restrain solute
&cntrl
 imin=1, maxcyc=3000, ncyc=1000, ntb=1, cut=9.0, ntpr=500,
 ntr=1, restraint_wt=10.0, restraintmask='!:WAT,Na+,Cl-',
/
EOF
cat > min2.in <<EOF
min2: free
&cntrl
 imin=1, maxcyc=5000, ncyc=2000, ntb=1, cut=9.0, ntpr=500,
/
EOF
cat > heat.in <<EOF
heat NVT 0->300K, backbone+ligand-heavy restrained, pull restraint on
&cntrl
 imin=0, irest=0, ntx=1, nstlim=50000, dt=0.002, ntpr=5000, ntwx=0, ntwr=25000,
 ntb=1, cut=9.0, ntc=2, ntf=2, iwrap=1,
 ntt=3, gamma_ln=2.0, tempi=10.0, temp0=300.0, ig=-1,
 ntr=1, restraint_wt=5.0, restraintmask='@CA,C,N,O | (:LIG & !@H=)',
 nmropt=1,
/
&wt type='TEMP0', istep1=0, istep2=40000, value1=10.0, value2=300.0, /
&wt type='END', /
DISANG=pull.disang
EOF
cat > equil.in <<EOF
equil NPT, backbone restrained, pull restraint on
&cntrl
 imin=0, irest=1, ntx=5, nstlim=100000, dt=0.002, ntpr=5000, ntwx=0, ntwr=50000,
 ntb=2, ntp=1, taup=2.0, cut=9.0, ntc=2, ntf=2, iwrap=1,
 ntt=3, gamma_ln=2.0, temp0=300.0, ig=-1,
 ntr=1, restraint_wt=1.0, restraintmask='@CA,C,N,O',
 nmropt=1,
/
&wt type='END', /
DISANG=pull.disang
EOF
cat > pull.in <<EOF
pull: 1 ns NPT, COM(lig)-Fe flat-bottom restraint pulls ligand into pocket (<6 A), no positional restraints
&cntrl
 imin=0, irest=1, ntx=5, nstlim=500000, dt=0.002, ntpr=5000, ntwx=5000, ntwr=100000,
 ntb=2, ntp=1, taup=2.0, cut=9.0, ntc=2, ntf=2, iwrap=1, ioutfm=1,
 ntt=3, gamma_ln=2.0, temp0=300.0, ig=-1,
 nmropt=1,
/
&wt type='END', /
DISANG=pull.disang
EOF
cat > release.in <<EOF
release: 4 ns NPT, ALL restraints off; does the ligand stay or leave?
&cntrl
 imin=0, irest=1, ntx=5, nstlim=2000000, dt=0.002, ntpr=5000, ntwx=5000, ntwr=100000,
 ntb=2, ntp=1, taup=2.0, cut=9.0, ntc=2, ntf=2, iwrap=1, ioutfm=1,
 ntt=3, gamma_ln=2.0, temp0=300.0, ig=-1,
/
EOF

step() { # name, prev_rst, exe
    local n=$1 prev=$2 exe=$3
    [ -f $n.rst7 ] && return 0
    $exe -O -i $n.in -p system.parm7 -c $prev -ref $prev -o $n.out -r $n.rst7 -x $n.nc -inf $n.mdinfo
}
step min1    system.rst7 "$PMEMD"
step min2    min1.rst7   "$PMEMD"
step heat    min2.rst7   "$PMEMD"
step equil   heat.rst7   "$PMEMD"
step pull    equil.rst7  "$PMEMD"
step release pull.rst7   "$PMEMD"

# analysis: Fe-ligand distances, closest approach, contacts, ligand RMSD after protein fit
cat > analysis.cpptraj <<EOF
parm system.parm7
trajin pull.nc
trajin release.nc
autoimage
distance com_fe   :LIG&!@H= :HEM@FE out dist_com_fe.dat
nativecontacts :LIG&!@H= :1-500&!@H= distance 4.5 byresidue out contacts_byres.dat resout contacts_res.dat
rms fit  @CA,C,N,O,FE&:1-999 first
rms lig  :LIG&!@H= nofit first out lig_rmsd.dat
run
EOF
cpptraj -i analysis.cpptraj > analysis.log 2>&1 || echo "cpptraj failed, see analysis.log"
echo "done $NAME"
