# External Adversarial Review Protocol (`Claude` + `scripts/review_pack.sh`)

Every atomic task (`T08.1`–`T14.4` in `docs/roadmap/MASTER_ROADMAP.md`) and every phase milestone is designed to be audited independently in an external LLM session (such as **Claude**) before or immediately after landing on `main`.

`scripts/review_pack.sh` packages a self-contained Markdown review bundle in one command so you can copy-paste or upload a single file into Claude with zero manual prompt assembly.

---

## 1. Quickstart Commands

### Review a Single Atomic Task (`--task Txx.y`)
```sh
# Write a self-contained review bundle for task T08.3 across the latest commit:
sh scripts/review_pack.sh --task T08.3 HEAD~1..HEAD --out /tmp/review_T08.3.md

# macOS one-liner: copy the bundle directly to the clipboard and paste into Claude:
sh scripts/review_pack.sh --task T08.3 HEAD~1..HEAD | pbcopy
```

### Review an Unmerged Feature/Swarm Branch Against `origin/main`
```sh
# Default <git_range> is origin/main..HEAD when omitted:
sh scripts/review_pack.sh --task T09.4 --out review_pack.md
```

### Review an Entire Phase Milestone (`v0.3.0..HEAD`)
```sh
# Omit --task to review a multi-task phase range:
sh scripts/review_pack.sh v0.3.0..HEAD --out /tmp/review_phase08.md
```

> **Note:** Keep generated `review_pack.md` files in `/tmp/` or `data/private/` rather than committing them to git.

---

## 2. What `scripts/review_pack.sh` Includes

Each generated bundle contains five structured sections:

1. **Pre-Written Adversarial Review Prompt (for Claude)**:
   - Instructs the reviewer to act as a skeptical NMR spectroscopist, numerical algorithms engineer, and clean-room IP auditor — actively hunting subtle physical, statistical, numerical, and cross-platform bugs rather than rubber-stamping code.
   - Requests a structured verdict (`APPROVE`, `APPROVE WITH MINOR FOLLOW-UPS`, or `BLOCK`) with `P0`/`P1`/`P2` severity tags and exact `file:line` citations.
2. **Target Task Card (`docs/roadmap/MASTER_ROADMAP.md`)**:
   - Automatically extracted when `--task Txx.y` is supplied, giving the reviewer the exact **Owned Files/Modules**, **Clean-Room Literature Citations**, **User/CLI/GUI Contract**, and **Quantitative Done Gate**.
3. **Automated Pre-Flight Hygiene Scan + `AGENTS.md` 8-Point Audit Checklist**:
   - Runs automated checks over `<git_range>` (out-of-scope file touches, hardcoded host user-directory paths, tracked `data/private/` files, `Cargo.toml`/`THIRD_PARTY.md` dependency touches, and `Co-Authored-By` trailers) and embeds the full scientific checklist below.
4. **Commit History & Diffstat (`git log --stat <git_range>`)**.
5. **Complete Unified Diff (`git diff <git_range>`)**.

---

## 3. Mandatory Clean-Room & Scientific Audit Rubric

Every internal verifier and external Claude review evaluates changes against eight non-negotiable failure modes learned from real-spectrum development:

| # | Audit Dimension | What Must Hold | Failure Mode to Catch |
|---|---|---|---|
| **1** | **Clean-Room & Licences (`AGENTS.md`)** | Every algorithm module starts with a docstring citing peer-reviewed papers or public format specs; dependencies restricted to `MIT`, `BSD-2/3`, `Apache-2.0`, `ISC`, `Zlib` (`THIRD_PARTY.md` updated); compatibility claims phrased strictly as true comparisons. | Forbidden product names in module/file/symbol names; unattributed magic tables; copyleft (`GPL`/`AGPL`) or proprietary concepts copied without literature derivation. |
| **2** | **Field-Strength Invariance (`Hz` vs `ppm`)** | Scalar couplings ($^3J_{\text{HNH}\alpha} \approx 9.5\text{ Hz}$), digital resolution gates (`Hz/pt`), sinc sidelobe spacings, and linewidths (`Hz`) convert to `ppm` dynamically via `axis.obs_mhz`. | Hardcoded `ppm` cutoffs tuned at `500 MHz` that over-merge or under-merge peaks at `800–950 MHz`. |
| **3** | **Unlabelled-Only Reference Offsets** | Global chemical-shift reference offsets `(dH, dN, dCA, dCB, dC)` are estimated strictly from **unlabelled** empirical shift sets vs sequence-weighted priors (or unlabelled deposited shift sets in `score`), invariant to arbitrary residue-label permutations. | Fitting residuals on paired `(pred_k, truth_k)` assignments, which artificially inflates `CA`/`CB`/`C'` shift agreement on misassigned residues. |
| **4** | **Zero Structural or Homolog Leakage** | Backbone assignment (`assign`, `pipeline`, `benchmark`, `real-benchmark`) runs **sequence-only** (no `--pdb`); any empirical fragment database (`T12.3`) operates strictly post-assignment and excludes `< 35%` Needleman–Wunsch identity + any shared 7-mer peptide during benchmarks. | Silently conditioning sequence-only assignment benchmarks on deposited 3D structures or homologous BMRB entries. |
| **5** | **Confirmation-Bias Guard in Picking & NUS** | Peak picking (`T08.5`) and hypercomplex IST/FISTA NUS reconstruction (`T08.3`) depend **only** on experimental signal cubes and observed high-SNR spectral anchors — never on sequence or BMRB shift priors. | Hallucinating noise bumps at expected BMRB chemical shifts and reporting inflated assignment confidence. |
| **6** | **Cross-Platform, Linux Case & WebGPU Safety** | Spectrometer file discovery uses case-insensitive directory traversal (`ACQUS`/`SER` vs `acqus`/`ser`) so Linux `ext4` matches macOS APFS; `nmr-core` + `nmr-gui` compile cleanly on `wasm32-unknown-unknown`; explicit `--engine gpu` fails loudly when unsupported rather than silently falling back to CPU. | Literal lowercase `.join("acqus")` breaking uppercase UXNMR archives on Linux CI; silent CPU fallback masking broken WGSL shaders. |
| **7** | **Confidence Calibration & 5-Seed Gates** | High-confidence (`>= 0.80`) assignments account for whole-fragment register shifts on unanchored short stretches; solver changes pass 5-seed `--realistic` Ubiquitin and 5-seed `CISD3` (`standard` + `evo-graph`) without accuracy or HC-precision regression. | Single-spin marginals reporting `conf = 0.99` on floating 2–4 residue islands shifted by $\pm 1$ register; single-seed cherry-picking. |
| **8** | **Held-Out Benchmark Firewall** | All algorithm development, hyperparameter tuning, and ablation studies use synthetic suites and the frozen `dev` split (`real_manifest.tsv`); `heldout` is run **once** per milestone strictly for reporting. | Inspecting `heldout` errors and adjusting picker/solver constants afterward, turning held-out evaluation into a second `dev` set. |

---

## 4. Triage & Swarm Remediation Workflow

When Claude returns a review on a task or phase bundle:

1. **`P0` (Correctness, Clean-Room, Data Leakage, Crash/Panic) & `P1` (Calibration/Accuracy Regression, Cross-Platform Bug)**:
   - Paste Claude's findings directly into the swarm coordinator prompt referencing the Task ID (`Txx.y`).
   - Fix on a dedicated branch, add a regression unit/integration test reproducing the exact edge case, re-run `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo check --release -p nmr-gui --target wasm32-unknown-unknown`, and `cargo test --release --workspace`, and re-generate `scripts/review_pack.sh --task Txx.y` to verify closure.
2. **`P2` (Future Regime Enhancement, Non-Blocking UX Polish, Constant Review for Maintainer)**:
   - Record under the appropriate atomic task in `TODO.md` / `docs/roadmap/MASTER_ROADMAP.md` (or as `- [ ] (M)` if it touches `docs/specs/03-assignment.md` §3–§4 scoring constants owned by the maintainer).
3. **Release Tagging (`AGENTS.md`)**:
   - Once all tasks in a phase pass CI + external review, bump `[workspace.package] version` in `Cargo.toml`, record verified 5-seed metrics in `CHANGELOG.md`, and create an annotated `v0.Y.Z` git tag.
