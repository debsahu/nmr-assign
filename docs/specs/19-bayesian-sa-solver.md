# 19. Bayesian Pseudo-Energy & Specific-Heat ($C_v(T)$) Adaptive Simulated Annealing Backbone Solver

Status: implemented in `crates/nmr-core/src/bayes_sa.rs`, `crates/nmr-gpu/src/bayes_sa_gpu.rs`, and `crates/nmr-gpu/src/shaders/bayes_sa.wgsl`. Leaves `docs/specs/03-assignment.md`, `crates/nmr-core/src/solver.rs`, and `crates/nmr-core/src/evo_graph.rs` untouched while providing an independent clean-room Bayesian statistical-mechanics assignment engine compatible with `nmr_core::solver::AssignResult` and `nmr_core::qc::DatasetQcReport`.

## 1. Provenance and Clean-Room Scope

Every mathematical equation, probability model, Markov proposal, and GPU compute kernel in this solver is derived 100% clean-room from peer-reviewed statistical mechanics and biomolecular NMR publications together with our CC0 empirical BMRB prior corpus (`data/priors/bmrb_priors.json`):

1. **Bishop, K. P., Torres-Montalvo, G., Kotaru, S., Mimun, L. C. & Wand, A. J.** (2023). "Robust automated backbone triple resonance NMR assignments of proteins using Bayesian-based simulated annealing." *Nature Communications* **14**, 1556. DOI: `10.1038/s41467-023-37219-z` — joint $(i, i-1)$ amino-acid $\times$ secondary-structure Bayesian shift likelihoods, sequential overlap log-Bayes factors normalized against background shift coincidence density, null/dummy spin-system chemical potentials, and specific-heat ($C_v(T)$) adaptive simulated annealing across the cooperative assignment phase transition.
2. **Kirkpatrick, S., Gelatt, C. D. & Vecchi, M. P.** (1983). "Optimization by Simulated Annealing." *Science* **220**(4598), 671–680 — Metropolis–Hastings Boltzmann sampling, thermal energy variance, and specific-heat phase-transition diagnostics.
3. **Wishart, D. S., Bigam, C. G., Holm, A., Hodges, R. S. & Sykes, B. D.** (1995). *J. Biomol. NMR* **5**, 67–81 & **6**, 135–140 — primary-sequence covalent neighbor chemical-shift corrections (`nmr_core::priors::delta_covalent`) and secondary-structure chemical-shift displacements across $\alpha$-helix, $\beta$-strand, and random coil conformations.

No external C++/Python source repositories or external parameter tables were consulted.

## 2. Inputs, Outputs, and Public API

### 2.1 CPU (`nmr_core::bayes_sa`)

- `BayesSaConfig`: solver configuration (`replicas = 32` CPU default, `128` via `BayesSaConfig::default_gpu()`, `CoolingScheduleKind::{AdaptiveSpecificHeat, FixedGeometric}`, `t_start`, `t_min`, `delta_beta_0`, `specific_heat_eta`, `steps_per_stage_per_active`, `fixed_total_steps`, `outlier_floor_eps`, channel tolerances `sigma_{ca,cb,c,ha}`, `link_floor`, `support_bonus`, `mu_null_override`, `use_hn`, `auto_reference`, `enable_weak_root_penalty`, `apply_qc_calibration`, `locked_assignments: Vec<LockedAssignment>`, and `seed`).
- `BayesSaMatrices`: precomputed `(e_type, e_unary, e_link, weak_penalties, mu_null, applied_offsets, detailed_offsets, effective_priors, resolved_locks, active_slots)`.
- `BayesSaTempStep` & `BayesSaReport`: per-stage thermodynamic telemetry (`temperature`, `mean_energy`, intra-replica `energy_variance`, `ensemble_spread_variance`, `specific_heat_cv`, `acceptance_rate`, `min_replica_energy`) plus detected critical temperature $T_{\text{crit}}$, peak $C_v(T_{\text{crit}})$, `mu_null`, applied `ShiftOffsets` / `DetailedReferenceOffsets`, and `DatasetQcReport`.
- `compute_bayes_energy_matrices(systems, sequence, priors, cfg) -> BayesSaMatrices`.
- `evaluate_state_energy(at, matrices) -> Result<f64, String>` and `evaluate_position_energy(position, matrices) -> Result<f64, String>`: exact state validation and dimensionless nats pseudo-energy evaluation.
- `solve_bayes_sa`, `solve_bayes_sa_with_peaks`, and `solve_bayes_sa_from_peaks`: multi-threaded CPU solver (`std::thread::scope`) returning `(AssignResult, BayesSaReport)` (and assembled `Vec<SpinSystem>` for `solve_bayes_sa_from_peaks`).

### 2.2 WebGPU (`nmr_gpu::bayes_sa_gpu` & `shaders/bayes_sa.wgsl`)

- `BayesSaGpuSolver`: reusable WebGPU compute context compiling 7 entry points (`compute_ln_z_bg`, `compute_e_type`, `compute_e_link`, `eval_state_energies`, `anneal_bayes_256`, `anneal_bayes_1024`, `anneal_bayes_global`) within a single portable `@group(0)` layout (`1` uniform + `8` storage buffers, compliant with `wgpu::Limits::default()`), tracking `wgpu::DeviceType` (`is_software_cpu_adapter()`).
- `BayesGpuKernelTier` & `select_bayes_gpu_kernel_tier(max_dim, force_global) -> (BayesGpuKernelTier, u32)`: deterministic tier selector (`Sram512`, `Sram1024`, `GlobalStorage`).
- `validate_bayes_gpu_buffer_budget(device, s_len, k_len, replicas) -> Result<(), String>`: validates every scoring, `Q27.36` state-evaluation, move-table, and replica state storage buffer allocation plus compute workgroup counts against `device.limits().max_buffer_size`, `max_storage_buffer_binding_size`, and `max_compute_workgroups_per_dimension`.
- `compute_bayes_energy_matrices_gpu`, `evaluate_states_energy_gpu`, `solve_bayes_sa_gpu`, `solve_bayes_sa_gpu_with_peaks`, and `solve_bayes_sa_gpu_from_peaks`.

## 3. Clean-Room Bayesian Pseudo-Energy Formulation

Let $\mathcal{S} = \{0, \dots, S-1\}$ be the assembled amide-rooted spin systems and $\mathcal{K} = \{0, \dots, N_{\text{seq}}-1\}$ be the primary sequence positions. Residues lacking backbone amide protons ($k = 0$ N-terminus and all `PRO` positions) belong to $\mathcal{K}_{\text{forbid}}$ and are strictly forbidden from holding any spin system ($x_k = -1$).

A valid assignment state $\mathbf{x} = (x_0, \dots, x_{N_{\text{seq}}-1}) \in (\{-1\} \cup \mathcal{S})^{N_{\text{seq}}}$ is a partial injection ($x_k = -1$ denotes an unassigned/null sequence slot; every spin system $s \in \mathcal{S}$ appears at most once) satisfying all user `LockedAssignment` constraints. Its dimensionless total Bayesian pseudo-energy (in nats, at unit canonical scale $T = 1.0$) is:

$$E(\mathbf{x}) = \sum_{\substack{k \in \mathcal{K} \\ x_k \ge 0}} E_{\text{unary}}(x_k, k) + \sum_{\substack{k = 1 \\ x_{k-1} \ge 0,\, x_k \ge 0}}^{N_{\text{seq}}-1} E_{\text{link}}(x_{k-1}, x_k)$$

where $E_{\text{unary}}(s, k) = E_{\text{type}}(s, k) + E_{\text{weak}}(s) + \mu_{\text{null}}$ for unlocked admissible pairs ($\mu_{\text{null}} > 0$ is the calibrated positive assignment cost relative to the $E = 0$ unassigned/null baseline; exact user locks retain clamped raw $E_{\text{type}}(s, k)$). Null slots ($x_k = -1$) contribute $0$ unary energy and $0$ sequential link energy across gaps, naturally isolating `PRO` breaks and missing amides without cascading frame-shifts.

### 3.1 Reference Offset Registration & Conformer Shift Priors

When `cfg.auto_reference` is enabled, `estimate_reference_offsets_detailed(systems, sequence, priors)` estimates global chemical-shift referencing offsets $(\Delta\delta_{\text{CA}}, \Delta\delta_{\text{CB}}, \Delta\delta_{\text{C'}}, \Delta\delta_{\text{N}}, \Delta\delta_{\text{H}})$ and shifts the CC0 `Priors` table via `priors.with_offsets(applied_offsets)`.

Because empirical BMRB distributions represent marginal mixtures across secondary structures, each amino acid $r \in \{0, \dots, 19\}$ and backbone conformation class $c \in \{\alpha\text{-helix}, \beta\text{-strand}, \text{coil}\}$ is parameterized by biophysical secondary-shift displacements $\Delta\mu_{r, c, a}$ (preserving the empirical BMRB mean across amino-acid-specific propensities $P(c \mid r)$ and decomposing total BMRB variance into conformer-mean dispersion plus intra-class thermal variance), augmented at sequence position $k$ by primary-sequence covalent neighbor corrections `delta_covalent(sequence, k)`. Adjacent backbone conformations $(c_k, c_{k-1})$ follow a cooperative first-order Markov persistence transition $P(c_k \mid c_{k-1})$ (`PRO` forced to coil/turn).

### 3.2 Heavy-Tail Student-$t(\nu = 5)$ + Uniform Outlier Mixture & Exact Neutral Missing Channels

For each channel $a \in \{\text{CA}, \text{CB}, \text{C'}, \text{H}^{\text{N}}, \text{N}, \text{HA}\}$, observed intra-strip candidates $\{(\delta_{s, u, j}, \text{sup}_{s, u, j})\}$ (`sup > 0`) are weighted by corroboration support $w_j \propto 0.62^j \exp(\gamma_{\text{sup}}(\text{sup}_j - 1))$ ($\sum_j w_j = 1$) and evaluated under a Student-$t(\nu = 5)$ + uniform spectral-window mixture:

$$p(x \mid \mu, \sigma_t, R_a) = (1 - \epsilon_{\text{out}}) \frac{\Gamma(3)}{\sqrt{5\pi}\,\Gamma(5/2)\,\sigma_t} \left(1 + \frac{(x - \mu)^2}{5\sigma_t^2}\right)^{-3} + \frac{\epsilon_{\text{out}}}{R_a}$$

1. **Outlier robustness**: Even an extreme $15\text{ ppm}$ mispicked artifact or ring-current shift is bounded below by $\epsilon_{\text{out}} / R_a$, so a single corrupted channel never vetoes the true residue when remaining channels corroborate it.
2. **Exact `0.0` missing-channel invariance**: Unobserved channels (`cnt == 0`, e.g., `HNCA`-only datasets or `CA`-less paramagnetic strips) contribute `0.0` log-Bayes factor identically in both numerator and denominator, yielding exact $E_{\text{type}}(s, k) = 0.0$ when all channels are absent. Conversely, observing a real `CB(0)` or `CB(-1)` candidate against a `GLY` hypothesis incurs a soft mispick penalty ($\ln 0.02 \approx -3.912$ nats log-Bayes factor, i.e. $+3.912$ nats in $E_{\text{type}}$) rather than hard `-inf`.
3. **Marginal $(i, i-1)$ log-Bayes factor**:

$$E_{\text{type}}(s, k) = -\ln \frac{\sum_{c_k, c_{k-1}} P(c_k, c_{k-1} \mid \text{aa}_k, \text{aa}_{k-1}) \, \mathcal{B}_i(s \mid k, c_k) \, \mathcal{B}_{i-1}(s \mid k-1, c_{k-1})}{\sum_{r, r', c, c'} \pi(r)\pi_{\text{prev}}(r') P(c, c' \mid r, r') \, \mathcal{B}_i(s \mid r, c) \, \mathcal{B}_{i-1}(s \mid r', c')}$$

### 3.3 Multi-Pathway Rare-Shift Sequential Connectivity $E_{\text{link}}(s, t)$ & Null Chemical Potential $\mu_{\text{null}}$

Across all shared pathways $a \in \{\text{CA}, \text{CB}, \text{C'}, \text{HA}\}$ where strip $s$ provides $(a, 0)$ and strip $t$ provides $(a, -1)$, background shift-coincidence density $p_{\text{rand}, a}(\bar{\delta})$ is evaluated by weighting the 20-amino-acid $\times$ 3-conformer BMRB Student-$t(\nu = 5)$ + uniform heavy-tail mixture densities using sequence amino-acid counts Dirichlet-shrunk toward natural abundance `NAT_AA_PROB` ($\alpha_0 = 16.0$). Candidate pairs $(u, v)$ compete via support-normalized weights $w_u^{(0)} w_v^{(-1)}$:

$$\text{BF}_a(s, t) = (1 - p_{\text{leak}}) \sum_{u, v} w_u^{(0)} w_v^{(-1)} \frac{\mathcal{N}(\delta_u^{(0)} - \delta_v^{(-1)} \mid 0, 2\sigma_a^2)}{p_{\text{rand}, a}(\bar{\delta}_{u, v})} + p_{\text{leak}}, \qquad E_{\text{link}}(s, t) = -\sum_a \ln \text{BF}_a(s, t)$$

Because $p_{\text{rand}, a}(\bar{\delta})$ is small in rare chemical-shift wings (`Gly CA` $\approx 45\text{ ppm}$, `Ser/Thr CB` $\approx 63\text{–}70\text{ ppm}$) and high in crowded central bands (`CA` $\approx 56\text{ ppm}$, `CB` $\approx 31\text{ ppm}$), rare-shift matches yield substantially stronger negative sequential link energies than crowded-band matches at identical $\Delta\delta = 0$.

The null/dummy chemical potential $\mu_{\text{null}}$ is calibrated automatically from dataset completeness ($S_{\text{clean}} / N_{\text{assignable}}$) and strip quality, paired with weak-root penalties $E_{\text{weak}}(s)$ for uncorroborated single-experiment noise strips, sub-stoichiometric minor-conformer shadows, and water-band artifacts.

## 4. Reversible Markov Moves & Specific-Heat $C_v(T)$ Adaptive Cooling

### 4.1 Balanced Metropolis–Hastings Move Kernel

Each replica maintains bijective state arrays `at[k]` and `pos[s]` over unlocked slots `free_k` and unlocked admissible spin systems `free_s`, executing three reversible move families with exact local $\mathcal{O}(1)$ / $\mathcal{O}(W)$ energy deltas:

- **Move (a) (`16%`)**: Symmetric two-position spin/dummy swap between unlocked sequence slots $(k_1, k_2)$ (55% local sequence-window distance $\le 12$, 45% global).
- **Move (b) (`56%`)**: Exact Metropolis–Hastings spin-to-slot proposal $(k, s)$ combining global `free_s`, top-$M$ unary/walk-guided slot candidates `cand_spins`, and sequential neighbor extensions `succ8`/`pred8`, handling 5 reversible sub-cases with exact forward/reverse Hastings weights $\ln(w_{\text{rev}} / w_{\text{fwd}})$:
  1.Occupied self-deletion ($x_k = s \to -1$),
  2. Unassigned spin insertion into empty slot ($x_k = -1, \text{pos}[s] = -1 \to x_k = s$),
  3. Unassigned spin replacing occupied slot ($x_k = t \to s$), either evicting $t \to -1$ or bijectively relocating $t \to m$ (`RestoreEvict` guided by `top_k8`),
  4. Assigned spin relocating from $k_s \to k$ into an empty slot,
  5. Assigned spin targeting an occupied slot ($x_k = t, x_{k_s} = s$), performing either a 2-swap or bijective `Evict` ($s \to k, t \to -1$ paired with Case 3's `RestoreEvict`).
- **Move (c) (`28%`)**: Contiguous `free_k` window moves skipping `PRO` and locked slots cleanly:
  1. Register-preserving symmetric $+1 / -1$ cyclic window shift ($W \in [2, L_{\max}]$),
  2. Exact-MH non-overlapping contiguous block swap ($L \in [2, L_{\max}]$) guided by `succ8`/`pred8` stitching and `top_k8` island relocation.

### 4.2 Specific-Heat $C_v(T_m)$ Schedule & Post-Annealing Ensemble Consolidation

At stage $m$ with temperature $T_m$, each replica equilibrates for an intra-stage burn-in (75% on stage 0 from 20% sparse seeding; 50% on subsequent stages) and accumulates online Welford energy mean and variance $(\mu_{E, r}, M_{2, r})$. Ensemble intra-replica thermal variance $\sigma_E^2(T_m) = \frac{1}{R} \sum_r \sigma_{E, r}^2$ defines the dimensionless specific heat:

$$C_v(T_m) = \frac{\langle E^2 \rangle_{T_m} - \langle E \rangle_{T_m}^2}{T_m^2} = \frac{\sigma_E^2(T_m)}{T_m^2} \ge 0$$

The next stage temperature updates adaptively via:

$$T_{m+1} = \max\!\left(T_{\min},\; T_m \exp\!\left(-\frac{\Delta\beta_0}{\sqrt{1 + \eta \, C_v(T_m) / N_{\text{active}}}}\right)\right)$$

decelerating automatically across the cooperative assignment phase transition around $T_{\text{crit}}$ (where chain nucleation causes $C_v(T)$ to peak) and accelerating through high-$T$ paramagnetic and low-$T$ frozen regimes, terminating early once post-peak $C_v(T_m) / N_{\text{active}} \le \text{tol}_{C_v}$ and acceptance rate $\le \text{tol}_{\text{acc}}$ stabilize.

Across all $R$ replicas, `finalize_bayes_sa_from_replicas` synthesizes a Minimum-Bayes-Risk (MBR) consensus candidate from Laplace-smoothed Boltzmann votes, applies deterministic zero-temperature ($T = 0$) coordinate descent over single-site, 2-swap, and connected-stretch relocations, computes unit-temperature marginal assignment posteriors (incorporating cooperative multi-residue chain corroboration across Proline-delimited sequential islands on large proteins), applies `DatasetQcReport` calibration (`evaluate_dataset_qc` + `apply_dataset_qc_to_assignments`), and enforces hard `LockedAssignment` invariants (`confidence = 1.0`).

## 5. WebGPU Compute Architecture (`nmr-gpu::bayes_sa_gpu`)

1. **Portable 8-Storage-Buffer Binding Layout**: All 7 WGSL entry points share one `@group(0)` layout (`@binding(0)` uniform parameters + `@binding(1..=8)` storage buffers), satisfying baseline `wgpu::Limits::default()` (`max_storage_buffers_per_shader_stage == 8`).
2. **Three-Tier `i16`-Paired `16 KiB` Workgroup SRAM & Global Storage-Buffer Execution**:
   - Each `u32` word of `var<workgroup> wg_packed: array<u32, 4096>` (`16,384 B` = exact portable `16 KiB` WebGPU SRAM limit) packs signed `i16` `(at[i], pos[i])` halves without device-memory read-write cache thrashing, using zero-mutation candidate energy evaluation (`site_e_cand_*`, `two_site_e_cand_*`) and mathematically exact Lazy Hastings upper-bound rejection short-circuiting so rejected moves never mutate state or scan `cand_spins` / `succ8` / `pred8`:
     - $\max(S, N_{\text{seq}}) \le 512$ (including `480 aa` proteins): `anneal_bayes_256` (`@workgroup_size(8, 1, 1)`, 8 replicas/workgroup in `16 KiB` SRAM).
     - $513 \le \max(S, N_{\text{seq}}) \le 1024$ (including `600–800 aa` proteins): `anneal_bayes_1024` (`@workgroup_size(4, 1, 1)`, 4 replicas/workgroup in `16 KiB` SRAM).
   - Ultra-large proteins ($\max(S, N_{\text{seq}}) > 1024$) and explicit global-buffer tests execute `anneal_bayes_global` (`@workgroup_size(32, 1, 1)`) directly over global storage state buffers (`at_buf`, `pos_buf`), supporting arbitrary sequence lengths up to `device.limits().max_storage_buffer_binding_size`.
3. **TDR-Safe Chunking & Device Limit Guards**: Long MCMC stages are split into chunks of at most `MAX_BAYES_GPU_CHUNK_STEPS = 4,096` proposals per GPU submission with persistent `xoshiro128++` PRNG and Welford accumulators across chunks (verified bit-for-bit identical between `512` and `4,096` chunk sizes). Every buffer size is validated upfront against `device.limits().max_buffer_size` and `max_storage_buffer_binding_size`.
4. **Fast-Math-Immune `Q27.36` Fixed-Point State Energy Parity**: Because shader compilers (e.g. Metal MSL fast-math) algebraically fold floating-point `TwoSum`/Kahan error terms, `eval_state_energies` encodes `f64` matrix entries as 64-bit two's-complement `Q27.36` fixed-point integers ($2^{36}$ scale, $\approx 1.45 \times 10^{-11}\text{ nats}$ resolution) and accumulates exact 64-bit integer carries (`vec2<u32>`), achieving $< 10^{-4}$ CPU vs. WebGPU parity on `E_type`, `E_unary`, `E_link`, and arbitrary state total energies $E(\mathbf{x})$ even at $|E| > 3,000\text{ nats}$ (`480+ aa`).
