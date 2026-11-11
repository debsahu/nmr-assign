# 02. Peak picking (not started)

Goal: processed spectra in, peak lists (01 TSV) out, for HSQC and backbone
triple-resonance spectra first; NOESY later.

## Stage A: classical baseline
1. Noise estimate per spectrum (robust MAD of empty regions).
2. Local maxima above k x noise (k tunable, default 5), with sign for
   phase-sensitive experiments (HNCACB).
3. Lineshape fit (Lorentzian / Gaussian mix) to refine centres and split
   overlapping peaks.
4. Restrict triple-resonance peaks to H/N strips under HSQC roots.

## Stage B: neural picker
- 3D convolutional network trained on synthetic spectra rendered from BMRB
  shift tables via the simulator (05): lineshapes, noise, t1 noise,
  truncation artefacts, overlap, missing and weak peaks.
- Output: peak probability map plus sub-voxel offsets.
- Clean-room: DEEP Picker (GPL) may be read as a paper only, not as code.

## Evaluation
- Ground truth: expected peak positions predicted from the reference
  assignment (05).
- Metrics: recall, precision, position error per dimension.
- Final metric: downstream assignment accuracy (03) using picked peaks
  versus simulated peaks.

## Open questions (maintainer)
- Which spectra are available with the completed assignment, and processed
  how (Bruker pdata, NMRPipe)?
- Acceptable false-positive rate given the assignment engine tolerates
  spurious peaks?
