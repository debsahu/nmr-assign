# 04. Structure calculation (not started)

Goal: assigned shifts plus NOESY peaks in, structure ensemble out.

## Approach (draft)
1. Dihedral restraints from backbone shifts (own TALOS-like predictor
   trained on BMRB/PDB pairs; TALOS itself is not used).
2. NOESY peak assignment: ambiguous distance restraints from shift matching,
   iteratively filtered against intermediate structures.
3. Structure generation: simulated annealing in torsion or Cartesian space
   (OpenMM, MIT/LGPL components to be checked) seeded optionally by an
   AlphaFold2 model (Apache-2.0 code; check weight terms).
4. Ensemble selection and validation (restraint violations, Ramachandran).

## Clean-room
CYANA, ARIA, Xplor-NIH, and CNS are not used or read; methods only from
publications.

## Evaluation
Backbone RMSD to deposited PDB structures for proteins with BMRB + PDB
entries; restraint violation statistics.
