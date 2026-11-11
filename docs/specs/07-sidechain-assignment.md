# 07. Sidechain assignment (not started)

Goal: extend backbone assignment (03) to all 1H, 13C, 15N sidechain shifts.

## Experiments
HCCH-TOCSY, HCCH-COSY, (H)CC(CO)NH-TOCSY, H(CCCO)NH-TOCSY, 13C-HSQC,
aromatic 13C-HSQC; 15N/13C-edited NOESY as supporting evidence.

## Approach (draft)
1. Fix backbone assignments with confidence >= 0.8 as anchors.
2. Expected-peak generation per residue type from each experiment's
   magnetization transfer pathway (topology tables derived from the PDB
   Chemical Component Dictionary, not from any commercial residue library).
3. Same scoring family as 03: LLR against per-atom BMRB priors plus
   cross-spectrum consistency, with candidates for ambiguous peaks.
4. Joint optimization over backbone + sidechain so sidechain evidence can
   correct weak backbone calls.

## Evaluation
Per-atom-type accuracy against reference assignments (05); completeness of
methyl and aromatic assignments reported separately.
