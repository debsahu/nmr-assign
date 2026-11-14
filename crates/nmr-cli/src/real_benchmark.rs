//! Frozen real-protein raw time-domain (FID) benchmark runner (`nmrassign real-benchmark`).
//!
//! Provenance: Original clean-room implementation of `docs/specs/05-benchmark.md` and
//! `docs/specs/11-fid-processing.md` sections 5.2–5.3, citing:
//! - Public spectrometer parameter specifications: Bruker JCAMP-DX v5.0 (`acqus`/`acqu2s`/`acqu3s`,
//!   `pulseprogram`, `pdata/1/title`) and Varian/Agilent VnmrJ Data File Reference (`procpar`
//!   ASCII dictionary: `pslabel`, `seqfil`, `comment`, `sw`/`sw1`/`sw2`, `sfrq`/`dfrq`/`dfrq2`).
//! - IUPAC/IUBMB indirect chemical-shift ranges for triple-resonance protein NMR experiments:
//!   Wishart, D. S. et al., *J. Biomol. NMR* **1995**, 6, 135–140.
//! - Strictly unlabelled empirical global reference-offset estimation (`fit_unlabelled_global_offsets`
//!   in `crates/nmr-cli/src/main.rs`) — fitting ONE scalar offset per atom type (`H`, `N`, `CA`,
//!   `CB`, `C`) per entry from unlabelled marginal/2D point clouds without ever inspecting matched
//!   `(predicted, truth)` residue pairs.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use nmr_core::evaluate::to_reference;
use nmr_core::peaks::Peak;
use nmr_core::picker::{
    corroborated_re_pick_3d, pick_hsqc_2d, CorroboratedRepickConfig, PickerConfig, StripBank3D,
};
use nmr_core::priors::{DetailedReferenceOffsets, NucleusOffsetFit, OffsetFitStatus, ShiftOffsets};
use nmr_core::process::{process_fid_dir_2d, process_fid_dir_3d};
use nmr_core::qc::{DatasetQcReport, ProteinReliabilityCurve, STANDARD_CPU_CALIBRATED_RUNS};
use nmr_core::solver::ScoreConfig;
use nmr_core::spectrum::{Spectrum2D, Spectrum3D};
use nmr_core::spinsystems::{build_spin_systems, Tolerances};

use crate::{
    evaluate_prediction_against_truth, execute_assignment_workflow, load_priors,
    parse_bounded_runs, parse_bounded_steps, parse_engine, parse_solver, parse_truth_auto,
    register_3d_hn_to_hsqc_adaptive, resolve_solver_engine, version_string, EngineKind, SolverKind,
};

// ============================================================================
// Manifest Types & Parsing (`data/benchmark/real_manifest.tsv`)
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BenchmarkSplitFilter {
    Dev,
    Heldout,
    All,
}

impl BenchmarkSplitFilter {
    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "dev" => Ok(Self::Dev),
            "heldout" | "holdout" | "held-out" => Ok(Self::Heldout),
            "all" => Ok(Self::All),
            other => Err(format!(
                "unknown --split {other:?}; expected dev or heldout (or all)"
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Dev => "dev",
            Self::Heldout => "heldout",
            Self::All => "all",
        }
    }

    pub fn matches(self, entry_split: &str) -> bool {
        match self {
            Self::All => true,
            Self::Dev => entry_split.trim().eq_ignore_ascii_case("dev"),
            Self::Heldout => {
                let s = entry_split.trim();
                s.eq_ignore_ascii_case("heldout")
                    || s.eq_ignore_ascii_case("holdout")
                    || s.eq_ignore_ascii_case("held-out")
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RealManifestEntry {
    pub bmrb_id: String,
    pub protein: String,
    pub residues: usize,
    pub field_mhz: f64,
    pub vendor: String,
    pub experiments: Vec<String>,
    pub split: String,
    pub cohort: String,
}

pub fn parse_real_manifest_str(text: &str) -> Result<Vec<RealManifestEntry>, String> {
    let mut out = Vec::new();
    for (line_idx, raw_line) in text.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols
            .first()
            .is_some_and(|c| c.trim().eq_ignore_ascii_case("bmrb_id"))
        {
            continue;
        }
        if cols.len() < 7 {
            return Err(format!(
                "real_manifest.tsv line {} has {} columns (expected 7 or 8: bmrb_id, protein, residues, field_mhz, vendor, experiments, split[, cohort])",
                line_idx + 1,
                cols.len()
            ));
        }
        let bmrb_id = cols[0].trim().to_string();
        let protein = cols[1].trim().to_string();
        let residues: usize = cols[2].trim().parse().map_err(|e| {
            format!(
                "invalid residues {:?} on line {} ({bmrb_id}): {e}",
                cols[2],
                line_idx + 1
            )
        })?;
        let field_first = cols[3].split([',', '/']).next().unwrap_or("").trim();
        let field_mhz: f64 = field_first.parse().map_err(|e| {
            format!(
                "invalid field_mhz {:?} on line {} ({bmrb_id}): {e}",
                cols[3],
                line_idx + 1
            )
        })?;
        let vendor = cols[4].trim().to_string();
        let experiments: Vec<String> = cols[5]
            .split([',', ';', '+'])
            .map(canonicalize_exp_token)
            .filter(|s| !s.is_empty() && s != "*" && s != "AUTO")
            .collect();
        let split = cols[6].trim().to_ascii_lowercase();
        let cohort = cols
            .get(7)
            .map(|c| c.trim().to_ascii_lowercase())
            .filter(|c| !c.is_empty())
            .unwrap_or_else(|| split.clone());
        out.push(RealManifestEntry {
            bmrb_id,
            protein,
            residues,
            field_mhz,
            vendor,
            experiments,
            split,
            cohort,
        });
    }
    Ok(out)
}

fn canonicalize_exp_token(raw: &str) -> String {
    let cleaned: String = raw
        .trim()
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect::<String>()
        .to_ascii_uppercase();
    match cleaned.as_str() {
        "15NHSQC" | "NHSQC" | "HNHSQC" | "HSQC" | "TROSY" | "15NTROSY" => "HSQC".to_string(),
        "HNCOCACB" | "CBCACONH" | "CBCACOCANH" => "CBCACONH".to_string(),
        "CBCANH" | "HNCACB" => "HNCACB".to_string(),
        other => other.to_string(),
    }
}

// ============================================================================
// Linux Case-Safe Filesystem & Local Entry Resolution Helpers
// ============================================================================

/// Deterministically list directory children sorted by exact UTF-8/byte filename.
fn sorted_dir_entries(dir: &Path) -> Vec<(String, PathBuf)> {
    let Ok(rd) = fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut items: Vec<(String, PathBuf)> = rd
        .filter_map(|r| r.ok())
        .map(|e| (e.file_name().to_string_lossy().into_owned(), e.path()))
        .collect();
    items.sort_by(|a, b| a.0.cmp(&b.0));
    items
}

/// Case-insensitive child lookup returning the exact filesystem path (safe on Linux `ext4`).
pub fn find_child_ci(dir: &Path, name_ci: &str) -> Option<PathBuf> {
    for (fname, path) in sorted_dir_entries(dir) {
        if fname.eq_ignore_ascii_case(name_ci) {
            return Some(path);
        }
    }
    None
}

/// Resolve the main git repository checkout root when running inside a git worktree so gitignored
/// `data/ubiquitin` and `data/private/` are discovered when `--data-root` uses its default.
fn resolve_git_main_checkout_root() -> Option<PathBuf> {
    let dot_git = Path::new(".git");
    if dot_git.is_file() {
        let txt = fs::read_to_string(dot_git).ok()?;
        for line in txt.lines() {
            if let Some(rest) = line.strip_prefix("gitdir:") {
                let gd = PathBuf::from(rest.trim());
                // Worktree gitdir has shape `<main_repo>/.git/worktrees/<worktree_name>`.
                let worktrees_dir = gd.parent()?;
                if worktrees_dir.file_name().and_then(|s| s.to_str()) == Some("worktrees") {
                    let main_git_dir = worktrees_dir.parent()?;
                    if main_git_dir.file_name().and_then(|s| s.to_str()) == Some(".git") {
                        return main_git_dir.parent().map(Path::to_path_buf);
                    }
                }
            }
        }
    }
    None
}

#[derive(Clone, Debug)]
pub struct ResolvedEntryLocations {
    pub entry_dir: PathBuf,
    pub truth_path: Option<PathBuf>,
}

/// Resolve `(entry_dir, optional_truth_path)` for a manifest entry.
/// Returns `Err(expected_display_path)` when the local dataset directory is absent on disk.
pub fn resolve_local_entry_paths(
    bmrb_id: &str,
    data_root: &Path,
    allow_repo_fallbacks: bool,
) -> Result<ResolvedEntryLocations, String> {
    let id_trim = bmrb_id.trim();
    let mut base_roots: Vec<PathBuf> = Vec::new();
    // When `--data-root` has shape `<repo>/data/private/bmrb` or `<repo>/data/real_bmrb`, include
    // `<repo>/data` so sibling `<repo>/data/ubiquitin` and `<repo>/data/private/bmrb51439_cisd3` resolve cleanly too.
    if data_root
        .file_name()
        .and_then(|s| s.to_str())
        .is_some_and(|s| s.eq_ignore_ascii_case("bmrb"))
    {
        if let Some(priv_parent) = data_root.parent() {
            if priv_parent
                .file_name()
                .and_then(|s| s.to_str())
                .is_some_and(|s| s.eq_ignore_ascii_case("private"))
            {
                if let Some(data_anc) = priv_parent.parent() {
                    base_roots.push(data_anc.to_path_buf());
                }
            }
        }
    } else if data_root
        .file_name()
        .and_then(|s| s.to_str())
        .is_some_and(|s| s.eq_ignore_ascii_case("real_bmrb"))
    {
        if let Some(data_anc) = data_root.parent() {
            base_roots.push(data_anc.to_path_buf());
        }
    }
    if allow_repo_fallbacks {
        base_roots.push(PathBuf::from("data"));
        if let Some(env_data) = env::var_os("NMR_REAL_DATA_DIR") {
            base_roots.push(PathBuf::from(env_data));
        }
        if let Some(main_repo) = resolve_git_main_checkout_root() {
            base_roots.push(main_repo.join("data"));
        }
    }

    if id_trim.eq_ignore_ascii_case("ubiquitin_local") {
        // 1. Check `<data-root>/ubiquitin_local` first (supports custom isolated test roots)
        if let Some(custom_ubq) = find_child_ci(data_root, "ubiquitin_local") {
            if custom_ubq.is_dir() {
                let truth = find_truth_file_in_or_near(&custom_ubq, "ubiquitin_local");
                return Ok(ResolvedEntryLocations {
                    entry_dir: custom_ubq,
                    truth_path: truth,
                });
            }
        }
        for droot in &base_roots {
            if let Some(ubq_dir) = find_child_ci(droot, "ubiquitin") {
                if ubq_dir.is_dir() {
                    let truth = find_child_ci(droot, "ubiquitin_backbone_assignment_191104.txt")
                        .or_else(|| find_truth_file_in_or_near(&ubq_dir, "ubiquitin_local"));
                    return Ok(ResolvedEntryLocations {
                        entry_dir: ubq_dir,
                        truth_path: truth,
                    });
                }
            }
        }
        return Err("data/ubiquitin".to_string());
    }

    let id_digits = id_trim
        .strip_prefix("bmr")
        .or_else(|| id_trim.strip_prefix("BMR"))
        .unwrap_or(id_trim);

    // Special local CISD3 (`51439`) paths + `<data-root>/<id>`
    let mut dir_candidates: Vec<PathBuf> = Vec::new();
    if let Some(c1) = find_child_ci(data_root, id_digits) {
        dir_candidates.push(c1);
    }
    if let Some(c2) = find_child_ci(data_root, &format!("bmr{id_digits}")) {
        dir_candidates.push(c2);
    }

    if !base_roots.is_empty() {
        if id_digits == "51439" {
            for droot in &base_roots {
                if let Some(priv_dir) = find_child_ci(droot, "private") {
                    if let Some(cisd3) = find_child_ci(&priv_dir, "bmrb51439_cisd3") {
                        dir_candidates.push(cisd3);
                    }
                    if let Some(bmrb_sub) = find_child_ci(&priv_dir, "bmrb") {
                        if let Some(c) = find_child_ci(&bmrb_sub, "51439") {
                            dir_candidates.push(c);
                        }
                    }
                }
                if let Some(real_bmrb) = find_child_ci(droot, "real_bmrb") {
                    if let Some(c) = find_child_ci(&real_bmrb, "51439") {
                        dir_candidates.push(c);
                    }
                }
            }
        } else {
            for droot in &base_roots {
                if let Some(priv_dir) = find_child_ci(droot, "private") {
                    if let Some(bmrb_sub) = find_child_ci(&priv_dir, "bmrb") {
                        if let Some(c1) = find_child_ci(&bmrb_sub, id_digits) {
                            dir_candidates.push(c1);
                        }
                        if let Some(c2) = find_child_ci(&bmrb_sub, &format!("bmr{id_digits}")) {
                            dir_candidates.push(c2);
                        }
                    }
                }
                if let Some(real_bmrb) = find_child_ci(droot, "real_bmrb") {
                    if let Some(c1) = find_child_ci(&real_bmrb, id_digits) {
                        dir_candidates.push(c1);
                    }
                    if let Some(c2) = find_child_ci(&real_bmrb, &format!("bmr{id_digits}")) {
                        dir_candidates.push(c2);
                    }
                }
            }
        }
    }

    for cand in dir_candidates {
        if cand.is_dir() {
            let truth = find_truth_file_in_or_near(&cand, id_digits);
            return Ok(ResolvedEntryLocations {
                entry_dir: cand,
                truth_path: truth,
            });
        }
    }

    Err(data_root.join(id_digits).display().to_string())
}

fn find_truth_file_in_or_near(entry_dir: &Path, id_digits: &str) -> Option<PathBuf> {
    let target_3 = format!("bmr{id_digits}_3.str");
    let target_plain = format!("bmr{id_digits}.str");

    if let Some(p) = find_child_ci(entry_dir, &target_3) {
        if p.is_file() {
            return Some(p);
        }
    }
    if let Some(p) = find_child_ci(entry_dir, &target_plain) {
        if p.is_file() {
            return Some(p);
        }
    }
    if let Some(p) = find_child_ci(entry_dir, "ubiquitin_backbone_assignment_191104.txt") {
        if p.is_file() {
            return Some(p);
        }
    }

    // Any `.str` at top level or 1 subdirectory deep
    for (fname, path) in sorted_dir_entries(entry_dir) {
        if path.is_file() && fname.to_ascii_lowercase().ends_with(".str") {
            return Some(path);
        }
    }
    for (_, sub) in sorted_dir_entries(entry_dir) {
        if sub.is_dir() {
            for (fname, path) in sorted_dir_entries(&sub) {
                if path.is_file() && fname.to_ascii_lowercase().ends_with(".str") {
                    return Some(path);
                }
            }
        }
    }

    // Fallback to public `data/benchmark/bmr<id>.str` if present
    let bench_dir = Path::new("data/benchmark");
    if let Some(p) = find_child_ci(bench_dir, &target_plain) {
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

// ============================================================================
// Deterministic Raw FID Experiment Auto-Discovery (Bruker & Varian)
// ============================================================================

#[derive(Clone, Debug)]
pub struct DiscoveredFidExperiment {
    pub exp_type: String,
    pub dir: PathBuf,
    pub rel_path: String,
    pub vendor: String,
    pub dims: usize,
    pub pulse_program: String,
    pub title: String,
    pub h_mhz: f64,
    pub indirect_td_product: usize,
    pub raw_file_bytes: u64,
    pub quad_mode_summary: String,
    pub nus_summary: String,
}

#[derive(Clone, Debug)]
pub struct CandidateFidInspection {
    pub dir: PathBuf,
    pub rel_path: String,
    pub vendor: String,
    pub dims: usize,
    pub pulse_program: String,
    pub title: String,
    pub h_mhz: f64,
    pub direct_nuc: String,
    pub indirect_nucs: Vec<String>,
    pub c_carrier_ppm: Option<f64>,
    pub c_sw_ppm: Option<f64>,
    pub n_sw_ppm: Option<f64>,
    pub indirect_td_product: usize,
    pub raw_file_bytes: u64,
    pub quad_mode_summary: String,
    pub nus_summary: String,
    pub classified_exp: Option<String>,
}

/// Infer canonical experiment type (`HSQC`, `HNCO`, `HNCACO`, `HNCA`, `HNCOCA`, `HNCACB`, `CBCACONH`)
/// from a single Bruker (`acqus`/`ser`) or Varian (`procpar`/`fid`) directory.
pub fn infer_single_fid_experiment(dir: &Path) -> Option<String> {
    let rel = dir.file_name().and_then(|s| s.to_str()).unwrap_or(".");
    inspect_single_raw_fid_dir(dir, rel).and_then(|c| c.classified_exp)
}

/// Inspect `<entry_dir>` recursively (up to depth 6, skipping `pdata` and hidden folders) and
/// deterministically classify raw Bruker (`acqus`/`ser`) and Varian (`procpar`/`fid`) backbone
/// experiments (`HSQC`, `HNCO`, `HNCACO`, `HNCA`, `HNCOCA`, `HNCACB`, `CBCACONH`).
pub fn discover_entry_experiments(
    entry_dir: &Path,
    hint_field_mhz: Option<f64>,
) -> Result<(Vec<DiscoveredFidExperiment>, Vec<CandidateFidInspection>), String> {
    let mut raw_dirs = Vec::new();
    collect_raw_fid_dirs_recursive(entry_dir, 0, 6, &mut raw_dirs);

    let mut inspections: Vec<CandidateFidInspection> = Vec::new();
    for dir in raw_dirs {
        let rel = dir
            .strip_prefix(entry_dir)
            .unwrap_or(&dir)
            .to_string_lossy()
            .replace('\\', "/");
        let rel_clean = if rel.is_empty() { ".".to_string() } else { rel };
        if let Some(insp) = inspect_single_raw_fid_dir(&dir, &rel_clean) {
            inspections.push(insp);
        }
    }

    // Group classified candidates by canonical experiment type
    let mut by_exp: BTreeMap<String, Vec<CandidateFidInspection>> = BTreeMap::new();
    for insp in &inspections {
        if let Some(exp) = &insp.classified_exp {
            if insp.raw_file_bytes > 0 {
                by_exp.entry(exp.clone()).or_default().push(insp.clone());
            }
        }
    }

    // Anchor session parent directories: parent directories of uniquely-identified 3D experiments
    // (or all 3D triple-resonance experiments when unambiguous) plus median 3D `1H` spectrometer MHz.
    let mut anchor_parents = BTreeSet::new();
    let mut triple_mhz_vals: Vec<f64> = Vec::new();
    for (exp, cands) in &by_exp {
        if exp != "HSQC" {
            for c in cands {
                if c.h_mhz > 100.0 {
                    triple_mhz_vals.push(c.h_mhz);
                }
            }
            if cands.len() == 1 {
                if let Some(parent) = Path::new(&cands[0].rel_path).parent() {
                    let ps = parent.to_string_lossy().into_owned();
                    if !ps.is_empty() && ps != "." {
                        anchor_parents.insert(ps);
                    }
                }
            }
        }
    }
    triple_mhz_vals.sort_by(|a, b| a.total_cmp(b));
    let target_mhz = if !triple_mhz_vals.is_empty() {
        Some(triple_mhz_vals[triple_mhz_vals.len() / 2])
    } else {
        hint_field_mhz.filter(|&m| m > 100.0)
    };

    let exp_order = [
        "HSQC", "HNCO", "HNCACO", "HNCA", "HNCOCA", "HNCACB", "CBCACONH",
    ];
    let mut selected = Vec::new();
    for &exp_name in &exp_order {
        let Some(mut cands) = by_exp.remove(exp_name) else {
            continue;
        };
        cands.sort_by(|a, b| {
            let sa = candidate_selection_score(a, &anchor_parents, target_mhz);
            let sb = candidate_selection_score(b, &anchor_parents, target_mhz);
            sb.cmp(&sa).then_with(|| a.rel_path.cmp(&b.rel_path))
        });
        let best = &cands[0];
        // Once a 3D experiment picks its best session directory, add its parent to anchors too
        // so HSQC / companion 3Ds co-locate deterministically.
        if exp_name != "HSQC" {
            if let Some(parent) = Path::new(&best.rel_path).parent() {
                let ps = parent.to_string_lossy().into_owned();
                if !ps.is_empty() && ps != "." {
                    anchor_parents.insert(ps);
                }
            }
        }
        selected.push(DiscoveredFidExperiment {
            exp_type: exp_name.to_string(),
            dir: best.dir.clone(),
            rel_path: best.rel_path.clone(),
            vendor: best.vendor.clone(),
            dims: best.dims,
            pulse_program: best.pulse_program.clone(),
            title: best.title.clone(),
            h_mhz: best.h_mhz,
            indirect_td_product: best.indirect_td_product,
            raw_file_bytes: best.raw_file_bytes,
            quad_mode_summary: best.quad_mode_summary.clone(),
            nus_summary: best.nus_summary.clone(),
        });
    }

    // Re-evaluate HSQC selection if 3D tie-breaking added a new anchor parent (`selected[0]` is HSQC)
    if let Some(hsqc_pos) = selected.iter().position(|s| s.exp_type == "HSQC") {
        let mut hsqc_cands: Vec<CandidateFidInspection> = inspections
            .iter()
            .filter(|c| c.classified_exp.as_deref() == Some("HSQC") && c.raw_file_bytes > 0)
            .cloned()
            .collect();
        if hsqc_cands.len() > 1 {
            hsqc_cands.sort_by(|a, b| {
                let sa = candidate_selection_score(a, &anchor_parents, target_mhz);
                let sb = candidate_selection_score(b, &anchor_parents, target_mhz);
                sb.cmp(&sa).then_with(|| a.rel_path.cmp(&b.rel_path))
            });
            let best = &hsqc_cands[0];
            selected[hsqc_pos] = DiscoveredFidExperiment {
                exp_type: "HSQC".to_string(),
                dir: best.dir.clone(),
                rel_path: best.rel_path.clone(),
                vendor: best.vendor.clone(),
                dims: best.dims,
                pulse_program: best.pulse_program.clone(),
                title: best.title.clone(),
                h_mhz: best.h_mhz,
                indirect_td_product: best.indirect_td_product,
                raw_file_bytes: best.raw_file_bytes,
                quad_mode_summary: best.quad_mode_summary.clone(),
                nus_summary: best.nus_summary.clone(),
            };
        }
    }

    Ok((selected, inspections))
}

fn candidate_selection_score(
    c: &CandidateFidInspection,
    anchor_parents: &BTreeSet<String>,
    target_mhz: Option<f64>,
) -> i64 {
    let mut score = 0i64;
    if let Some(parent) = Path::new(&c.rel_path).parent() {
        let ps = parent.to_string_lossy();
        if anchor_parents.contains(ps.as_ref()) {
            score += 10_000;
        }
    }
    if let Some(tmhz) = target_mhz {
        let diff = (c.h_mhz - tmhz).abs();
        if diff <= 5.0 {
            score += 5_000;
        } else if diff <= 25.0 {
            score += 1_000;
        } else {
            score -= (diff as i64) * 10;
        }
    }
    let t_low = c.title.to_ascii_lowercase();
    let p_low = c.pulse_program.to_ascii_lowercase();
    let rel_low = c.rel_path.to_ascii_lowercase();
    // Deprioritize aligned media / RDC / paramagnetic / BIRD / D2O H-D exchange / NH2-only variants
    // when an isotropic H2O backbone spectrum exists
    if t_low.contains("align")
        || t_low.contains("allign")
        || t_low.contains("acrylamid")
        || t_low.contains("polyphosphazine")
        || t_low.contains("bicelle")
        || t_low.contains("phage")
        || t_low.contains("paramag")
        || t_low.contains("d2o")
        || t_low.contains("hdex")
        || rel_low.contains("d2o")
        || rel_low.contains("hdex")
        || rel_low.contains("nh2only")
        || p_low.contains("bird")
    {
        score -= 3_000;
    }
    // Prefer standard HSQC over TROSY when at moderate field unless TROSY is the only 2D
    if c.classified_exp.as_deref() == Some("HSQC") && !p_low.contains("trosy") {
        score += 200;
    }
    score += (c.indirect_td_product.min(4_096)) as i64;
    score
}

fn collect_raw_fid_dirs_recursive(
    dir: &Path,
    depth: usize,
    max_depth: usize,
    out: &mut Vec<PathBuf>,
) {
    if depth > max_depth {
        return;
    }
    let has_bruker_acqu =
        find_child_ci(dir, "acqus").is_some() || find_child_ci(dir, "acqu").is_some();
    let has_bruker_raw = find_child_ci(dir, "ser").is_some_and(|p| p.is_file())
        || find_child_ci(dir, "fid").is_some_and(|p| p.is_file());
    let has_varian_procpar = find_child_ci(dir, "procpar").is_some_and(|p| p.is_file());
    let has_varian_fid = find_child_ci(dir, "fid").is_some_and(|p| p.is_file());

    if (has_bruker_acqu && has_bruker_raw) || (has_varian_procpar && has_varian_fid) {
        out.push(dir.to_path_buf());
        return;
    }

    for (fname, path) in sorted_dir_entries(dir) {
        if !path.is_dir() {
            continue;
        }
        if fname.starts_with('.')
            || fname.eq_ignore_ascii_case("pdata")
            || fname.eq_ignore_ascii_case("__macosx")
        {
            continue;
        }
        collect_raw_fid_dirs_recursive(&path, depth + 1, max_depth, out);
    }
}

fn inspect_single_raw_fid_dir(dir: &Path, rel_path: &str) -> Option<CandidateFidInspection> {
    if find_child_ci(dir, "acqus").is_some() || find_child_ci(dir, "acqu").is_some() {
        inspect_bruker_dir(dir, rel_path)
    } else if find_child_ci(dir, "procpar").is_some() {
        inspect_varian_dir(dir, rel_path)
    } else {
        None
    }
}

// ============================================================================
// Bruker JCAMP-DX Acquisition Metadata Inspector
// ============================================================================

fn parse_jcamp_map(path: &Path) -> HashMap<String, String> {
    let mut map = HashMap::new();
    let Ok(bytes) = fs::read(path) else {
        return map;
    };
    let text = String::from_utf8_lossy(&bytes);
    let mut pending_array_key: Option<String> = None;
    for raw_line in text.lines() {
        let line = raw_line.trim();
        if let Some(key) = pending_array_key.take() {
            if !line.starts_with("##") {
                map.insert(key, line.trim_matches(['<', '>']).trim().to_string());
                continue;
            }
        }
        if let Some(rest) = line.strip_prefix("##$") {
            if let Some((k, v)) = rest.split_once('=') {
                let key = k.trim().to_string();
                let val = v.trim();
                if val.starts_with('(') && val.contains("..") && !val.ends_with('>') {
                    // JCAMP array header `##$KEY= (0..7)` with values on next line
                    pending_array_key = Some(key);
                } else {
                    map.insert(key, val.trim_matches(['<', '>']).trim().to_string());
                }
            }
        }
    }
    map
}

fn read_bruker_dim_params(
    dir: &Path,
    primary: &str,
    fallback: &str,
) -> Option<HashMap<String, String>> {
    let p1 = find_child_ci(dir, primary);
    let p2 = find_child_ci(dir, fallback);
    if p1.is_none() && p2.is_none() {
        return None;
    }
    let mut merged = p2.map(|p| parse_jcamp_map(&p)).unwrap_or_default();
    if let Some(p) = p1 {
        for (k, v) in parse_jcamp_map(&p) {
            merged.insert(k, v);
        }
    }
    Some(merged)
}

fn normalize_nucleus_code(raw: &str, sfo_mhz: f64, h_mhz: f64) -> String {
    let up: String = raw
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect::<String>()
        .to_ascii_uppercase();
    match up.as_str() {
        "1H" | "H1" | "PROTON" => return "1H".to_string(),
        "15N" | "N15" => return "15N".to_string(),
        "13C" | "C13" => return "13C".to_string(),
        "2H" | "H2" | "D" => return "2H".to_string(),
        "31P" | "P31" => return "31P".to_string(),
        "19F" | "F19" => return "19F".to_string(),
        _ => {}
    }
    if h_mhz > 100.0 && sfo_mhz > 1.0 {
        let ratio = sfo_mhz / h_mhz;
        if (0.90..=1.10).contains(&ratio) {
            return "1H".to_string();
        }
        if (0.085..=0.125).contains(&ratio) {
            return "15N".to_string();
        }
        if (0.23..=0.27).contains(&ratio) {
            return "13C".to_string();
        }
    } else if sfo_mhz >= 250.0 {
        return "1H".to_string();
    }
    up
}

fn read_bruker_title_and_pp_header(dir: &Path) -> (String, String) {
    let mut title = String::new();
    if let Some(pdata) = find_child_ci(dir, "pdata") {
        if let Some(p1) = find_child_ci(&pdata, "1") {
            if let Some(tp) = find_child_ci(&p1, "title") {
                if let Ok(bytes) = fs::read(tp) {
                    title = String::from_utf8_lossy(&bytes)
                        .lines()
                        .map(str::trim)
                        .filter(|l| !l.is_empty())
                        .collect::<Vec<_>>()
                        .join(" | ");
                }
            }
        }
    }
    let mut pp_comment = String::new();
    if let Some(pp) =
        find_child_ci(dir, "pulseprogram").or_else(|| find_child_ci(dir, "pulseprogram.precomp"))
    {
        if let Ok(bytes) = fs::read(pp) {
            let head = String::from_utf8_lossy(&bytes[..bytes.len().min(2048)]);
            pp_comment = head
                .lines()
                .take(20)
                .filter(|l| l.trim_start().starts_with(';'))
                .collect::<Vec<_>>()
                .join(" ");
        }
    }
    (title, pp_comment)
}

fn inspect_bruker_dir(dir: &Path, rel_path: &str) -> Option<CandidateFidInspection> {
    let d1 = read_bruker_dim_params(dir, "acqus", "acqu")?;
    let d2_opt = read_bruker_dim_params(dir, "acqu2s", "acqu2");
    let d3_opt = read_bruker_dim_params(dir, "acqu3s", "acqu3");

    let ser_file = find_child_ci(dir, "ser").filter(|p| p.is_file());
    let fid_file = find_child_ci(dir, "fid").filter(|p| p.is_file());
    let raw_file_bytes = ser_file
        .as_ref()
        .or(fid_file.as_ref())
        .and_then(|p| fs::metadata(p).ok())
        .map(|m| m.len())
        .unwrap_or(0);

    let get_usize = |m: &HashMap<String, String>, k: &str| -> usize {
        m.get(k).and_then(|v| v.parse::<usize>().ok()).unwrap_or(0)
    };
    let get_f64 = |m: &HashMap<String, String>, k: &str| -> f64 {
        m.get(k).and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0)
    };
    let get_i32 = |m: &HashMap<String, String>, k: &str| -> Option<i32> {
        m.get(k).and_then(|v| v.parse::<i32>().ok())
    };

    let pulprog = d1.get("PULPROG").cloned().unwrap_or_default();
    let sfo1 = get_f64(&d1, "SFO1");
    let bf1 = get_f64(&d1, "BF1");
    let h_mhz_guess = if sfo1 >= 200.0 {
        sfo1
    } else if bf1 >= 200.0 {
        bf1
    } else {
        sfo1
    };
    let direct_nuc = normalize_nucleus_code(
        d1.get("NUC1").map(String::as_str).unwrap_or(""),
        sfo1,
        h_mhz_guess,
    );

    let td2 = d2_opt.as_ref().map(|m| get_usize(m, "TD")).unwrap_or(0);
    let nustd2 = d2_opt.as_ref().map(|m| get_usize(m, "NusTD")).unwrap_or(0);
    let td3 = d3_opt.as_ref().map(|m| get_usize(m, "TD")).unwrap_or(0);
    let nustd3 = d3_opt.as_ref().map(|m| get_usize(m, "NusTD")).unwrap_or(0);

    let dims = if ser_file.is_some() && d3_opt.is_some() && (td3 > 1 || nustd3 > 1) {
        3
    } else if ser_file.is_some() && d2_opt.is_some() && (td2 > 1 || nustd2 > 1) {
        2
    } else {
        1
    };

    let h_mhz = if direct_nuc == "1H" {
        sfo1.max(bf1)
    } else {
        0.0
    };
    let mut indirect_nucs = Vec::new();
    let mut c_carrier_ppm = None;
    let mut c_sw_ppm = None;
    let mut n_sw_ppm = None;

    let mut inspect_ind = |m: &HashMap<String, String>| {
        let sfo = get_f64(m, "SFO1");
        let bf = get_f64(m, "BF1");
        let sw_ppm = {
            let sw = get_f64(m, "SW");
            if sw > 0.0 {
                sw
            } else if sfo > 0.0 {
                get_f64(m, "SW_h") / sfo
            } else {
                0.0
            }
        };
        let nuc =
            normalize_nucleus_code(m.get("NUC1").map(String::as_str).unwrap_or(""), sfo, h_mhz);
        if nuc == "13C" {
            c_sw_ppm = Some(sw_ppm);
            if bf > 0.0 && sfo > 0.0 {
                c_carrier_ppm = Some(((sfo - bf) / bf) * 1.0e6);
            }
        } else if nuc == "15N" {
            n_sw_ppm = Some(sw_ppm);
        }
        indirect_nucs.push(nuc);
    };

    if dims >= 2 {
        if let Some(m2) = &d2_opt {
            inspect_ind(m2);
        }
    }
    if dims >= 3 {
        if let Some(m3) = &d3_opt {
            inspect_ind(m3);
        }
    }

    let aq_mod = get_i32(&d1, "AQ_mod").unwrap_or(-1);
    let fn2 = d2_opt.as_ref().and_then(|m| get_i32(m, "FnMODE"));
    let fn3 = d3_opt.as_ref().and_then(|m| get_i32(m, "FnMODE"));
    let quad_mode_summary = match dims {
        3 => format!(
            "AQ_mod={aq_mod},F2:FnMODE={},F1:FnMODE={}",
            fn2.map(|v| v.to_string()).unwrap_or_else(|| "?".into()),
            fn3.map(|v| v.to_string()).unwrap_or_else(|| "?".into())
        ),
        2 => format!(
            "AQ_mod={aq_mod},F1:FnMODE={}",
            fn2.map(|v| v.to_string()).unwrap_or_else(|| "?".into())
        ),
        _ => format!("AQ_mod={aq_mod}"),
    };

    let fn_type = get_i32(&d1, "FnTYPE").unwrap_or(0);
    let has_nuslist = find_child_ci(dir, "nuslist").is_some_and(|p| p.is_file());
    let nus_summary = if fn_type == 2 || (fn_type > 0 && has_nuslist) {
        format!(
            "NUS(FnTYPE={fn_type},nuslist={})",
            if has_nuslist { "yes" } else { "missing" }
        )
    } else {
        format!("uniform(FnTYPE={fn_type})")
    };

    let eff_td2 = td2.max(nustd2).max(1);
    let eff_td3 = if dims == 3 { td3.max(nustd3).max(1) } else { 1 };
    let indirect_td_product = if dims >= 2 { eff_td2 * eff_td3 } else { 0 };

    let (title, pp_comment) = read_bruker_title_and_pp_header(dir);
    let dir_leaf = dir.file_name().and_then(|s| s.to_str()).unwrap_or("");
    let classified_exp = classify_backbone_experiment(
        dims,
        &direct_nuc,
        &indirect_nucs,
        &pulprog,
        &pp_comment,
        &title,
        dir_leaf,
        c_carrier_ppm,
        c_sw_ppm,
        n_sw_ppm,
    );

    Some(CandidateFidInspection {
        dir: dir.to_path_buf(),
        rel_path: rel_path.to_string(),
        vendor: "Bruker".to_string(),
        dims,
        pulse_program: pulprog,
        title,
        h_mhz,
        direct_nuc,
        indirect_nucs,
        c_carrier_ppm,
        c_sw_ppm,
        n_sw_ppm,
        indirect_td_product,
        raw_file_bytes,
        quad_mode_summary,
        nus_summary,
        classified_exp,
    })
}

// ============================================================================
// Varian / Agilent `procpar` Acquisition Metadata Inspector
// ============================================================================

#[derive(Default)]
struct VarianProcparSummary {
    strings: HashMap<String, Vec<String>>,
    reals: HashMap<String, Vec<f64>>,
}

impl VarianProcparSummary {
    fn first_str(&self, key: &str) -> Option<&str> {
        self.strings
            .get(key)
            .and_then(|v| v.first())
            .map(String::as_str)
    }

    fn first_f64(&self, key: &str) -> Option<f64> {
        self.reals.get(key).and_then(|v| v.first()).copied()
    }
}

fn parse_varian_procpar_summary(path: &Path) -> VarianProcparSummary {
    let mut out = VarianProcparSummary::default();
    let Ok(bytes) = fs::read(path) else {
        return out;
    };
    let text = String::from_utf8_lossy(&bytes);
    let lines: Vec<&str> = text.lines().collect();
    let mut idx = 0usize;

    while idx < lines.len() {
        let hdr = lines[idx].trim();
        if hdr.is_empty() {
            idx += 1;
            continue;
        }
        let hdr_toks: Vec<&str> = hdr.split_whitespace().collect();
        // Standard VnmrJ `procpar` parameter header has 11 tokens: `name subtype basictype ...`
        if hdr_toks.len() >= 11
            && hdr_toks[1].parse::<i32>().is_ok()
            && matches!(hdr_toks[2], "0" | "1" | "2")
        {
            let name = hdr_toks[0].to_string();
            let basictype = hdr_toks[2];
            idx += 1;
            if idx >= lines.len() {
                break;
            }
            let val_line = lines[idx].trim();
            if basictype == "2" {
                // String array: `<count> "first" ...` followed by `count - 1` quoted lines
                let (cnt_str, rest) = val_line
                    .split_once(char::is_whitespace)
                    .unwrap_or((val_line, ""));
                let count = cnt_str.parse::<usize>().unwrap_or(1);
                let mut vals = Vec::with_capacity(count);
                vals.push(unquote_procpar_str(rest.trim()));
                idx += 1;
                for _ in 1..count {
                    if idx < lines.len() {
                        vals.push(unquote_procpar_str(lines[idx].trim()));
                        idx += 1;
                    }
                }
                out.strings.insert(name, vals);
            } else if basictype == "1" {
                let toks: Vec<&str> = val_line.split_whitespace().collect();
                let count = toks
                    .first()
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(0);
                let mut nums: Vec<f64> = toks
                    .iter()
                    .skip(1)
                    .filter_map(|s| s.parse::<f64>().ok())
                    .collect();
                idx += 1;
                while nums.len() < count && idx < lines.len() {
                    let more: Vec<f64> = lines[idx]
                        .split_whitespace()
                        .filter_map(|s| s.parse::<f64>().ok())
                        .collect();
                    if more.is_empty() {
                        break;
                    }
                    nums.extend(more);
                    idx += 1;
                }
                out.reals.insert(name, nums);
            } else {
                idx += 1;
            }
            // Skip trailing enum specification line (`0` or `N "opt1" "opt2" ...`)
            if idx < lines.len() {
                idx += 1;
            }
        } else {
            idx += 1;
        }
    }
    out
}

fn unquote_procpar_str(s: &str) -> String {
    let t = s.trim();
    if t.len() >= 2 && t.starts_with('"') && t.ends_with('"') {
        t[1..t.len() - 1].to_string()
    } else {
        t.trim_matches('"').to_string()
    }
}

fn inspect_varian_dir(dir: &Path, rel_path: &str) -> Option<CandidateFidInspection> {
    let procpar_path = find_child_ci(dir, "procpar")?;
    let fid_path = find_child_ci(dir, "fid").filter(|p| p.is_file());
    let raw_file_bytes = fid_path
        .as_ref()
        .and_then(|p| fs::metadata(p).ok())
        .map(|m| m.len())
        .unwrap_or(0);

    let pp = parse_varian_procpar_summary(&procpar_path);
    let pslabel = pp.first_str("pslabel").unwrap_or("");
    let seqfil = pp.first_str("seqfil").unwrap_or("");
    let pulse_program = if !pslabel.is_empty() {
        pslabel.to_string()
    } else {
        seqfil.to_string()
    };
    let comment = pp.first_str("comment").unwrap_or("").to_string();

    let sfrq = pp.first_f64("sfrq").unwrap_or(0.0);
    let dfrq = pp.first_f64("dfrq").unwrap_or(0.0);
    let dfrq2 = pp.first_f64("dfrq2").unwrap_or(0.0);
    let dfrq3 = pp.first_f64("dfrq3").unwrap_or(0.0);

    let tn = pp.first_str("tn").unwrap_or("");
    let dn = pp.first_str("dn").unwrap_or("");
    let dn2 = pp.first_str("dn2").unwrap_or("");
    let dn3 = pp.first_str("dn3").unwrap_or("");

    let direct_nuc = normalize_nucleus_code(tn, sfrq, sfrq);
    let h_mhz = if direct_nuc == "1H" { sfrq } else { 0.0 };

    let ni = pp.first_f64("ni").unwrap_or(0.0) as usize;
    let ni2 = pp.first_f64("ni2").unwrap_or(0.0) as usize;
    let ni3 = pp.first_f64("ni3").unwrap_or(0.0) as usize;

    let mut active_ind_slots: Vec<usize> = Vec::new();
    if ni > 1 {
        active_ind_slots.push(1);
    }
    if ni2 > 1 {
        active_ind_slots.push(2);
    }
    if ni3 > 1 && active_ind_slots.len() < 2 {
        active_ind_slots.push(3);
    }
    let dims = 1 + active_ind_slots.len();

    let axis_str = pp.first_str("axis").unwrap_or("");
    let axis_chars: Vec<char> = axis_str.chars().collect();

    let mut indirect_nucs = Vec::new();
    let mut c_carrier_ppm = None;
    let mut c_sw_ppm = None;
    let mut n_sw_ppm = None;

    for &slot in &active_ind_slots {
        let sw_key = match slot {
            1 => "sw1",
            2 => "sw2",
            _ => "sw3",
        };
        let reffrq_key = match slot {
            1 => "reffrq1",
            2 => "reffrq2",
            _ => "reffrq3",
        };
        let sw_hz = pp.first_f64(sw_key).unwrap_or(0.0);
        let reffrq = pp.first_f64(reffrq_key).unwrap_or(0.0);
        let axis_ch = axis_chars.get(slot).copied();

        // Determine channel (`d` -> dn/dfrq, `2` -> dn2/dfrq2, `3` -> dn3/dfrq3).
        // Prefer hardware synthesizer frequency `chan_frq` (`dfrq`/`dfrq2`) over `reffrq*`,
        // which in legacy VnmrJ `procpar` files is occasionally left at the 1H frequency.
        let (chan_nuc_raw, chan_frq) = match axis_ch {
            Some('d') | Some('1') => (dn, dfrq),
            Some('2') => (dn2, dfrq2),
            Some('3') => (dn3, dfrq3),
            _ => {
                // BioPack standard: in 2D `gNhsqc`, `ni` (`sw1 ~ 1200..3500 Hz`) is `N15` (`dn2`) when `dn=C13`,
                // whereas in 3D (`ghncacb`, `ghnco`), `ni` (`sw1`) is `C13` (`dn`) and `ni2` (`sw2`) is `N15` (`dn2`).
                if dims == 2 {
                    let n_cand = normalize_nucleus_code(dn2, dfrq2, h_mhz);
                    let d_cand = normalize_nucleus_code(dn, dfrq, h_mhz);
                    if d_cand == "15N" {
                        (dn, dfrq)
                    } else if n_cand == "15N" && sw_hz <= 4800.0 {
                        (dn2, dfrq2)
                    } else {
                        (dn, dfrq)
                    }
                } else if slot == 1 {
                    (dn, dfrq)
                } else {
                    (dn2, dfrq2)
                }
            }
        };

        let nuc = normalize_nucleus_code(
            chan_nuc_raw,
            if chan_frq > 1.0 { chan_frq } else { reffrq },
            h_mhz,
        );
        let obs = if chan_frq > 1.0 {
            chan_frq
        } else if reffrq > 1.0 {
            reffrq
        } else if nuc == "13C" {
            h_mhz * 0.25144953
        } else if nuc == "15N" {
            h_mhz * 0.10132912
        } else {
            0.0
        };
        let sw_ppm = if obs > 0.0 { sw_hz / obs } else { 0.0 };
        if nuc == "13C" {
            c_sw_ppm = Some(sw_ppm);
            if h_mhz > 100.0 && chan_frq > 1.0 {
                let nu0_h = h_mhz / (1.0 + 4.766e-6);
                let nu0_c = nu0_h * 0.25144953;
                c_carrier_ppm = Some(((chan_frq - nu0_c) / nu0_c) * 1.0e6);
            }
        } else if nuc == "15N" {
            n_sw_ppm = Some(sw_ppm);
        }
        indirect_nucs.push(nuc);
    }

    let fmt_phase = |key: &str| -> String {
        match pp.reals.get(key) {
            Some(v) if !v.is_empty() => v
                .iter()
                .map(|x| format!("{}", *x as i32))
                .collect::<Vec<_>>()
                .join(","),
            _ => "none".to_string(),
        }
    };
    let fcoef_info = if pp.first_str("fcoef").is_some_and(|s| !s.is_empty())
        || pp.first_str("f1coef").is_some_and(|s| !s.is_empty())
        || pp.first_str("f2coef").is_some_and(|s| !s.is_empty())
    {
        "+fcoef"
    } else {
        ""
    };
    let quad_mode_summary = match dims {
        3 => format!(
            "phase=[{}],phase2=[{}]{fcoef_info}",
            fmt_phase("phase"),
            fmt_phase("phase2")
        ),
        2 => format!("phase=[{}]{fcoef_info}", fmt_phase("phase")),
        _ => "1D".to_string(),
    };

    let sampling = pp.first_str("sampling").unwrap_or("");
    let sparse = pp.first_str("sparse").unwrap_or("");
    let has_sch = find_child_ci(dir, "sampling.sch").is_some_and(|p| p.is_file())
        || find_child_ci(dir, "nuslist").is_some_and(|p| p.is_file());
    let explicit_uniform = sampling.eq_ignore_ascii_case("s") && !sparse.eq_ignore_ascii_case("y");
    let nus_summary = if sampling.eq_ignore_ascii_case("sparse")
        || sparse.eq_ignore_ascii_case("y")
        || (!explicit_uniform && sampling.is_empty() && has_sch)
    {
        format!(
            "NUS(sampling={sampling:?},sparse={sparse:?},sch={})",
            if has_sch { "yes" } else { "missing" }
        )
    } else {
        "uniform".to_string()
    };

    let indirect_td_product = active_ind_slots
        .iter()
        .map(|&s| match s {
            1 => ni.max(1),
            2 => ni2.max(1),
            _ => ni3.max(1),
        })
        .product::<usize>();

    let dir_leaf = dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .trim_end_matches(".fid");
    let seq_both = if !pslabel.is_empty() && !seqfil.is_empty() && pslabel != seqfil {
        format!("{pslabel} {seqfil}")
    } else {
        pulse_program.clone()
    };

    let classified_exp = classify_backbone_experiment(
        dims,
        &direct_nuc,
        &indirect_nucs,
        &seq_both,
        "",
        &comment,
        dir_leaf,
        c_carrier_ppm,
        c_sw_ppm,
        n_sw_ppm,
    );

    Some(CandidateFidInspection {
        dir: dir.to_path_buf(),
        rel_path: rel_path.to_string(),
        vendor: "Varian".to_string(),
        dims,
        pulse_program,
        title: comment,
        h_mhz,
        direct_nuc,
        indirect_nucs,
        c_carrier_ppm,
        c_sw_ppm,
        n_sw_ppm,
        indirect_td_product,
        raw_file_bytes,
        quad_mode_summary,
        nus_summary,
        classified_exp,
    })
}

// ============================================================================
// Pure Deterministic Experiment Classifier (`HSQC` & 3D Triple-Resonance)
// ============================================================================

#[allow(clippy::too_many_arguments)]
fn classify_backbone_experiment(
    dims: usize,
    direct_nuc: &str,
    indirect_nucs: &[String],
    primary_prog: &str,
    pp_header_comment: &str,
    title_or_comment: &str,
    dir_leaf_fallback: &str,
    c_carrier_ppm: Option<f64>,
    c_sw_ppm: Option<f64>,
    n_sw_ppm: Option<f64>,
) -> Option<String> {
    if direct_nuc != "1H" {
        return None;
    }

    let norm_alnum = |s: &str| -> String {
        s.chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .collect::<String>()
            .to_ascii_lowercase()
    };
    let p_norm = norm_alnum(primary_prog);
    let t_norm = norm_alnum(title_or_comment);
    let d_norm = norm_alnum(dir_leaf_fallback);
    let cmt_norm = norm_alnum(pp_header_comment);

    if dims == 2 {
        if indirect_nucs.first().map(String::as_str) != Some("15N") {
            return None;
        }
        if let Some(nsw) = n_sw_ppm {
            if nsw > 0.0 && !(5.0..=100.0).contains(&nsw) {
                return None;
            }
        }
        // Reject heteronuclear relaxation / T1 / T2 / NOE / CPMG / CEST / TOCSY / HMBC / CON 2D experiments
        for bad in [
            "hsqct1",
            "hsqct2",
            "hsqcnoe",
            "hsqnoef",
            "invit1",
            "invit2",
            "invinoe",
            "trosyt1",
            "trosyt2",
            "trosynoe",
            "gnhsqct1",
            "gnhsqct2",
            "gnhsqcnoe",
            "cpmg",
            "t1rho",
            "hetnoe",
            "relax",
            "cest",
            "tocsy",
            "dipsi",
            "roesy",
            "cosy",
            "hmbc",
            "ipap",
        ] {
            if p_norm.contains(bad) || t_norm.contains(bad) {
                return None;
            }
        }
        if title_or_comment.to_ascii_lowercase().contains("(1h) noe")
            || title_or_comment.to_ascii_lowercase().contains("15n-noe")
        {
            return None;
        }
        for src in [&p_norm, &cmt_norm, &t_norm, &d_norm] {
            if src.contains("hsqc")
                || src.contains("trosy")
                || src.contains("sfhmqc")
                || src.contains("sofast")
            {
                return Some("HSQC".to_string());
            }
        }
        return None;
    }

    if dims == 3 {
        let has_15n = indirect_nucs.iter().any(|n| n == "15N");
        let has_13c = indirect_nucs.iter().any(|n| n == "13C");
        if !has_15n || !has_13c {
            return None;
        }

        // Reject sidechain / NOESY / TOCSY 3D sequences first
        for sidechain_or_noe in [
            "hccconh",
            "ccconh",
            "hcch",
            "noesy",
            "tocsy",
            "dipsi",
            "hbhaconh",
            "hbhacbcanh",
            "hnha",
            "hnhb",
            "hncannh",
            "hncocannh",
        ] {
            if p_norm.contains(sidechain_or_noe) {
                return None;
            }
        }

        let match_3d_token = |tok: &str| -> Option<&'static str> {
            if tok.contains("cbcaconh")
                || tok.contains("cbcacocanh")
                || tok.contains("cbcacogp")
                || tok.contains("cbcacoca")
                || tok.contains("hncocacb")
                || tok.contains("cbcaconnh")
            {
                Some("CBCACONH")
            } else if tok.contains("hncacb") || tok.contains("cbcanh") {
                Some("HNCACB")
            } else if tok.contains("hncoca") {
                Some("HNCOCA")
            } else if tok.contains("hncaco") {
                Some("HNCACO")
            } else if tok.contains("hnca") {
                Some("HNCA")
            } else if tok.contains("hnco") {
                Some("HNCO")
            } else {
                None
            }
        };

        let cand = match_3d_token(&p_norm)
            .or_else(|| match_3d_token(&cmt_norm))
            .or_else(|| match_3d_token(&t_norm))
            .or_else(|| match_3d_token(&d_norm))?;

        // Physics sanity check against 13C carrier / spectral width when present
        if let Some(cc) = c_carrier_ppm {
            match cand {
                // Carbonyl C' carrier ~ 170..182 ppm (allow generous 135..215 ppm unless folded)
                "HNCO" | "HNCACO"
                    if !(135.0..=215.0).contains(&cc) && c_sw_ppm.is_some_and(|sw| sw > 42.0) =>
                {
                    return None;
                }
                // Aliphatic CA / CACB carrier ~ 35..60 ppm (never at 176 ppm carbonyl!)
                "HNCACB" | "CBCACONH" | "HNCA" | "HNCOCA" if cc > 130.0 => {
                    return None;
                }
                _ => {}
            }
        }
        return Some(cand.to_string());
    }

    None
}

// ============================================================================
// Benchmark Execution, Macro/Micro Aggregation & JSON Reporting
// ============================================================================

#[derive(Clone, Debug)]
pub struct EntryFailureDiagnostic {
    pub stage: String,
    pub experiment: Option<String>,
    pub vendor: String,
    pub quad_mode: String,
    pub nus_status: String,
    pub reason: String,
}

#[derive(Clone, Debug)]
pub struct EntrySuccessMetrics {
    pub seeds: usize,
    pub discovered_experiments: Vec<(String, String)>,
    pub n_hsqc_roots: usize,
    pub n_total_peaks: usize,
    pub assignable_per_seed: usize,
    pub total_assignable: usize,
    pub total_assigned: usize,
    pub total_correct: usize,
    pub total_hc_assigned: usize,
    pub total_hc_correct: usize,
    pub mean_accuracy: f64,
    pub mean_recall: f64,
    pub mean_hc_precision: f64,
    pub mean_hc_recall: f64,
    pub global_offsets: ShiftOffsets,
    pub detailed_offsets: DetailedReferenceOffsets,
    pub prior_detailed_offsets: DetailedReferenceOffsets,
    pub qc_report: DatasetQcReport,
    pub reliability: ProteinReliabilityCurve,
    pub predictions: Vec<(f64, bool)>,
    pub elapsed_secs: f64,
}

#[derive(Clone, Debug)]
pub enum EntryBenchmarkOutcome {
    SkippedMissingData { expected_path: String },
    Failed(EntryFailureDiagnostic),
    Succeeded(Box<EntrySuccessMetrics>),
}

#[derive(Clone, Debug)]
pub struct EntryBenchmarkRecord {
    pub manifest_entry: RealManifestEntry,
    pub outcome: EntryBenchmarkOutcome,
}

#[derive(Clone, Debug)]
pub struct RealBenchmarkSummary {
    pub split: String,
    pub solver_label: String,
    pub seeds: usize,
    pub hc_threshold: f64,
    pub records: Vec<EntryBenchmarkRecord>,
    pub n_hc_active_proteins: usize,
    pub n_hc_abstained_proteins: usize,
    pub worst_protein_hc_precision: Option<(String, String, f64)>,
    pub median_protein_hc_precision: Option<f64>,
    pub mean_protein_ece: f64,
    pub mean_protein_ece_10: f64,
    pub mean_protein_debiased_ece_10: f64,
    pub mean_protein_ece_10_bernoulli_floor: f64,
    pub mean_protein_ece_12: f64,
    pub pooled_reliability: ProteinReliabilityCurve,
    pub macro_accuracy: f64,
    pub macro_recall: f64,
    pub macro_hc_precision: f64,
    pub macro_hc_recall: f64,
    pub micro_assignable: usize,
    pub micro_assigned: usize,
    pub micro_correct: usize,
    pub micro_hc_assigned: usize,
    pub micro_hc_correct: usize,
    pub micro_accuracy: f64,
    pub micro_recall: f64,
    pub micro_hc_precision: f64,
    pub micro_hc_recall: f64,
}

#[derive(Clone, Debug)]
pub struct RealBenchmarkConfig {
    pub manifest_path: PathBuf,
    pub split: BenchmarkSplitFilter,
    pub solver: SolverKind,
    pub engine_opt: Option<EngineKind>,
    pub seeds: usize,
    pub base_seed: u64,
    pub data_root: PathBuf,
    pub allow_repo_fallbacks: bool,
    pub priors_path: Option<String>,
    pub runs_opt: Option<usize>,
    pub steps_opt: Option<usize>,
    pub hc_threshold: f64,
    pub no_crop: bool,
    pub lp: bool,
    pub json_out: Option<PathBuf>,
    pub verbose: bool,
}

/// Execute `process -> pick -> assign -> score` across all manifest entries matching `cfg.split`.
pub fn run_real_benchmark_with_loaders(
    cfg: &RealBenchmarkConfig,
    load_fid_2d: &dyn Fn(&Path) -> Result<Spectrum2D, String>,
    load_fid_3d: &dyn Fn(&Path, &str) -> Result<Spectrum3D, String>,
) -> Result<RealBenchmarkSummary, String> {
    let _validated_engine = resolve_solver_engine(cfg.solver, cfg.engine_opt)?;
    let manifest_txt = fs::read_to_string(&cfg.manifest_path)
        .map_err(|e| format!("failed to read {}: {e}", cfg.manifest_path.display()))?;
    let all_entries = parse_real_manifest_str(&manifest_txt)?;
    let entries: Vec<RealManifestEntry> = all_entries
        .into_iter()
        .filter(|e| cfg.split.matches(&e.split))
        .collect();

    if entries.is_empty() {
        return Err(format!(
            "no entries matching --split {} in {}",
            cfg.split.as_str(),
            cfg.manifest_path.display()
        ));
    }

    let priors = load_priors(cfg.priors_path.as_deref())?;
    let score_cfg = ScoreConfig::default();
    let picker_cfg = PickerConfig::default();

    let mut records: Vec<EntryBenchmarkRecord> = Vec::with_capacity(entries.len());
    let mut solver_label = match cfg.solver {
        SolverKind::Standard => "standard".to_string(),
        SolverKind::EvoGraph => "evo-graph".to_string(),
    };

    for entry in entries {
        let loc = match resolve_local_entry_paths(
            &entry.bmrb_id,
            &cfg.data_root,
            cfg.allow_repo_fallbacks,
        ) {
            Ok(l) => l,
            Err(expected_path) => {
                records.push(EntryBenchmarkRecord {
                    manifest_entry: entry,
                    outcome: EntryBenchmarkOutcome::SkippedMissingData { expected_path },
                });
                continue;
            }
        };

        let t_entry = Instant::now();

        // 1. Load deposited ground-truth shifts
        let Some(truth_path) = loc.truth_path else {
            records.push(EntryBenchmarkRecord {
                outcome: EntryBenchmarkOutcome::Failed(EntryFailureDiagnostic {
                    stage: "truth".to_string(),
                    experiment: None,
                    vendor: entry.vendor.clone(),
                    quad_mode: "n/a".to_string(),
                    nus_status: "n/a".to_string(),
                    reason: format!(
                        "missing NMR-STAR / shift truth file in {}",
                        loc.entry_dir.display()
                    ),
                }),
                manifest_entry: entry,
            });
            continue;
        };

        let (truth_ref, _excluded_q, _fmt) = match parse_truth_auto(&truth_path) {
            Ok(t) => t,
            Err(e) => {
                records.push(EntryBenchmarkRecord {
                    outcome: EntryBenchmarkOutcome::Failed(EntryFailureDiagnostic {
                        stage: "truth".to_string(),
                        experiment: None,
                        vendor: entry.vendor.clone(),
                        quad_mode: "n/a".to_string(),
                        nus_status: "n/a".to_string(),
                        reason: e,
                    }),
                    manifest_entry: entry,
                });
                continue;
            }
        };

        // 2. Auto-discover raw FID experiments inside `entry_dir`
        let (discovered, all_inspected) =
            match discover_entry_experiments(&loc.entry_dir, Some(entry.field_mhz)) {
                Ok(d) => d,
                Err(e) => {
                    records.push(EntryBenchmarkRecord {
                        outcome: EntryBenchmarkOutcome::Failed(EntryFailureDiagnostic {
                            stage: "discovery".to_string(),
                            experiment: None,
                            vendor: entry.vendor.clone(),
                            quad_mode: "n/a".to_string(),
                            nus_status: "n/a".to_string(),
                            reason: e,
                        }),
                        manifest_entry: entry,
                    });
                    continue;
                }
            };

        let fallback_vendor = all_inspected
            .first()
            .map(|c| c.vendor.clone())
            .unwrap_or_else(|| entry.vendor.clone());
        let fallback_quad = all_inspected
            .first()
            .map(|c| c.quad_mode_summary.clone())
            .unwrap_or_else(|| "unknown".to_string());
        let fallback_nus = all_inspected
            .first()
            .map(|c| c.nus_summary.clone())
            .unwrap_or_else(|| "unknown".to_string());

        let Some(hsqc_exp) = discovered.iter().find(|d| d.exp_type == "HSQC") else {
            records.push(EntryBenchmarkRecord {
                outcome: EntryBenchmarkOutcome::Failed(EntryFailureDiagnostic {
                    stage: "discovery".to_string(),
                    experiment: Some("HSQC".to_string()),
                    vendor: fallback_vendor,
                    quad_mode: fallback_quad,
                    nus_status: fallback_nus,
                    reason: format!(
                        "no 2D 1H-15N HSQC raw FID discovered in {} ({} candidate FID dirs inspected)",
                        loc.entry_dir.display(),
                        all_inspected.len()
                    ),
                }),
                manifest_entry: entry,
            });
            continue;
        };

        // Check that any experiment explicitly listed in `real_manifest.tsv` is present
        let required_set: BTreeSet<&str> = entry.experiments.iter().map(String::as_str).collect();
        let mut missing_required: Vec<String> = Vec::new();
        for &req in &required_set {
            if !discovered.iter().any(|d| d.exp_type == req) {
                missing_required.push(req.to_string());
            }
        }
        if !missing_required.is_empty() {
            records.push(EntryBenchmarkRecord {
                outcome: EntryBenchmarkOutcome::Failed(EntryFailureDiagnostic {
                    stage: "discovery".to_string(),
                    experiment: Some(missing_required.join(",")),
                    vendor: hsqc_exp.vendor.clone(),
                    quad_mode: hsqc_exp.quad_mode_summary.clone(),
                    nus_status: hsqc_exp.nus_summary.clone(),
                    reason: format!(
                        "manifest-declared experiment(s) [{}] not discovered in {} (found: {})",
                        missing_required.join(", "),
                        loc.entry_dir.display(),
                        discovered
                            .iter()
                            .map(|d| format!("{}:{}", d.exp_type, d.rel_path))
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                }),
                manifest_entry: entry,
            });
            continue;
        }

        let triple_exps: Vec<&DiscoveredFidExperiment> =
            discovered.iter().filter(|d| d.exp_type != "HSQC").collect();
        if triple_exps.is_empty() {
            records.push(EntryBenchmarkRecord {
                outcome: EntryBenchmarkOutcome::Failed(EntryFailureDiagnostic {
                    stage: "discovery".to_string(),
                    experiment: None,
                    vendor: hsqc_exp.vendor.clone(),
                    quad_mode: hsqc_exp.quad_mode_summary.clone(),
                    nus_status: hsqc_exp.nus_summary.clone(),
                    reason: format!(
                        "no 3D triple-resonance raw FIDs discovered alongside HSQC in {}",
                        loc.entry_dir.display()
                    ),
                }),
                manifest_entry: entry,
            });
            continue;
        }

        // 3. Process 2D HSQC raw FID & pick roots
        let hsqc_spec = match load_fid_2d(&hsqc_exp.dir) {
            Ok(s) => s,
            Err(reason) => {
                records.push(EntryBenchmarkRecord {
                    outcome: EntryBenchmarkOutcome::Failed(EntryFailureDiagnostic {
                        stage: "process".to_string(),
                        experiment: Some("HSQC".to_string()),
                        vendor: hsqc_exp.vendor.clone(),
                        quad_mode: hsqc_exp.quad_mode_summary.clone(),
                        nus_status: hsqc_exp.nus_summary.clone(),
                        reason,
                    }),
                    manifest_entry: entry,
                });
                continue;
            }
        };

        let roots = pick_hsqc_2d(&hsqc_spec, &picker_cfg);
        let mut all_peaks: Vec<Peak> = roots.clone();
        let mut used_exps: Vec<(String, String)> =
            vec![("HSQC".to_string(), hsqc_exp.rel_path.clone())];
        let mut entry_failed_diag: Option<EntryFailureDiagnostic> = None;

        // 4. Process each 3D triple-resonance raw FID one cube at a time & pick strips
        let mut strip_bank = StripBank3D::new();
        for texp in &triple_exps {
            let is_required =
                required_set.is_empty() || required_set.contains(texp.exp_type.as_str());
            match load_fid_3d(&texp.dir, &texp.exp_type) {
                Ok(mut spec3d) => {
                    // Align 3D HN plane to HSQC via two-stage rigid (dH, dN) registration
                    let (dh, dn) = register_3d_hn_to_hsqc_adaptive(&mut spec3d, &hsqc_spec);
                    if dh.abs() > 1e-6 || dn.abs() > 1e-6 {
                        eprintln!(
                            "[register] {}: aligned {} to HSQC: dH={:+.4} ppm, dN={:+.3} ppm",
                            entry.bmrb_id, texp.exp_type, dh, dn
                        );
                    }
                    let pks = strip_bank.pick_and_record_3d(&spec3d, &roots, &picker_cfg);
                    all_peaks.extend(pks);
                    used_exps.push((texp.exp_type.clone(), texp.rel_path.clone()));
                }
                Err(reason) => {
                    if is_required {
                        entry_failed_diag = Some(EntryFailureDiagnostic {
                            stage: "process".to_string(),
                            experiment: Some(texp.exp_type.clone()),
                            vendor: texp.vendor.clone(),
                            quad_mode: texp.quad_mode_summary.clone(),
                            nus_status: texp.nus_summary.clone(),
                            reason,
                        });
                        break;
                    } else if cfg.verbose {
                        eprintln!(
                            "[real-benchmark] {}: skipping optional unlisted {} ({}) after error: {}",
                            entry.bmrb_id, texp.exp_type, texp.rel_path, reason
                        );
                    }
                }
            }
        }

        if let Some(diag) = entry_failed_diag {
            records.push(EntryBenchmarkRecord {
                manifest_entry: entry,
                outcome: EntryBenchmarkOutcome::Failed(diag),
            });
            continue;
        }

        if used_exps.len() >= 3 && !roots.is_empty() {
            all_peaks.extend(corroborated_re_pick_3d(
                &strip_bank,
                &roots,
                &all_peaks,
                &picker_cfg,
                &CorroboratedRepickConfig::default(),
            ));
        }

        let all_systems_for_eval =
            match build_spin_systems(&all_peaks, &Tolerances::default(), 3, false) {
                Ok(s) => s,
                Err(reason) => {
                    records.push(EntryBenchmarkRecord {
                        outcome: EntryBenchmarkOutcome::Failed(EntryFailureDiagnostic {
                            stage: "pick".to_string(),
                            experiment: None,
                            vendor: hsqc_exp.vendor.clone(),
                            quad_mode: hsqc_exp.quad_mode_summary.clone(),
                            nus_status: hsqc_exp.nus_summary.clone(),
                            reason,
                        }),
                        manifest_entry: entry,
                    });
                    continue;
                }
            };

        // 5. Run solver across `0..cfg.seeds` & score strictly with `fit_unlabelled_global_offsets_detailed`
        let num_seeds = cfg.seeds.max(1);
        let mut tot_assignable = 0usize;
        let mut tot_assigned = 0usize;
        let mut tot_correct = 0usize;
        let mut tot_hc_assigned = 0usize;
        let mut tot_hc_correct = 0usize;
        let mut assignable_per_seed = 0usize;
        let mut reported_offsets = ShiftOffsets::default();
        let mut reported_detailed_offsets = DetailedReferenceOffsets::default();
        let mut reported_prior_detailed_offsets = DetailedReferenceOffsets::default();
        let mut reported_qc_report = DatasetQcReport::default();
        let mut entry_predictions: Vec<(f64, bool)> = Vec::new();
        let mut assign_err: Option<String> = None;

        for s_idx in 0..num_seeds {
            let cur_seed = cfg.base_seed + (s_idx as u64);
            let wf = match execute_assignment_workflow(
                cfg.solver,
                cfg.engine_opt,
                &all_peaks,
                &truth_ref.sequence,
                &priors,
                &score_cfg,
                cfg.runs_opt,
                cfg.steps_opt,
                STANDARD_CPU_CALIBRATED_RUNS,
                cur_seed,
            ) {
                Ok(w) => w,
                Err(e) => {
                    assign_err = Some(e);
                    break;
                }
            };
            solver_label = wf.engine_label.clone();

            let (pred_ref, conf_by_k) = to_reference(
                &wf.result,
                &wf.systems,
                &truth_ref.sequence,
                &truth_ref.seq_ids,
                0.0,
                Some(&wf.effective_priors),
            );
            let mut conf_by_seq_id = BTreeMap::new();
            for (k, c) in conf_by_k {
                conf_by_seq_id.insert(truth_ref.seq_ids[k], c);
            }

            let eval = evaluate_prediction_against_truth(
                &pred_ref,
                &truth_ref,
                &conf_by_seq_id,
                Some(&all_systems_for_eval),
                cfg.hc_threshold,
                true,
            );
            if s_idx == 0 {
                assignable_per_seed = eval.assignable;
                reported_offsets = eval.global_offsets;
                reported_detailed_offsets = eval.detailed_offsets;
                reported_prior_detailed_offsets = wf.detailed_offsets;
                reported_qc_report = wf.qc_report;
            }
            entry_predictions.extend_from_slice(&eval.placed_predictions);
            tot_assignable += eval.assignable;
            tot_assigned += eval.assigned;
            tot_correct += eval.correct;
            tot_hc_assigned += eval.hc_assigned;
            tot_hc_correct += eval.hc_correct;
        }

        if let Some(reason) = assign_err {
            records.push(EntryBenchmarkRecord {
                outcome: EntryBenchmarkOutcome::Failed(EntryFailureDiagnostic {
                    stage: "assign".to_string(),
                    experiment: None,
                    vendor: hsqc_exp.vendor.clone(),
                    quad_mode: hsqc_exp.quad_mode_summary.clone(),
                    nus_status: hsqc_exp.nus_summary.clone(),
                    reason,
                }),
                manifest_entry: entry,
            });
            continue;
        }

        let mean_accuracy = if tot_assigned > 0 {
            (tot_correct as f64) / (tot_assigned as f64)
        } else {
            0.0
        };
        let mean_recall = if tot_assignable > 0 {
            (tot_correct as f64) / (tot_assignable as f64)
        } else {
            0.0
        };
        let mean_hc_precision = if tot_hc_assigned > 0 {
            (tot_hc_correct as f64) / (tot_hc_assigned as f64)
        } else {
            0.0
        };
        let mean_hc_recall = if tot_assignable > 0 {
            (tot_hc_correct as f64) / (tot_assignable as f64)
        } else {
            0.0
        };
        let reliability = ProteinReliabilityCurve::from_predictions_with_repeats(
            &entry_predictions,
            cfg.hc_threshold,
            num_seeds as f64,
        );

        records.push(EntryBenchmarkRecord {
            manifest_entry: entry,
            outcome: EntryBenchmarkOutcome::Succeeded(Box::new(EntrySuccessMetrics {
                seeds: num_seeds,
                discovered_experiments: used_exps,
                n_hsqc_roots: roots.len(),
                n_total_peaks: all_peaks.len(),
                assignable_per_seed,
                total_assignable: tot_assignable,
                total_assigned: tot_assigned,
                total_correct: tot_correct,
                total_hc_assigned: tot_hc_assigned,
                total_hc_correct: tot_hc_correct,
                mean_accuracy,
                mean_recall,
                mean_hc_precision,
                mean_hc_recall,
                global_offsets: reported_offsets,
                detailed_offsets: reported_detailed_offsets,
                prior_detailed_offsets: reported_prior_detailed_offsets,
                qc_report: reported_qc_report,
                reliability,
                predictions: entry_predictions,
                elapsed_secs: t_entry.elapsed().as_secs_f64(),
            })),
        });
    }

    let summary = summarize_real_benchmark_records(
        cfg.split.as_str(),
        solver_label,
        cfg.seeds.max(1),
        cfg.hc_threshold,
        records,
    );

    print_real_benchmark_report(&summary);

    if let Some(json_p) = &cfg.json_out {
        if let Some(parent) = json_p.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)
                    .map_err(|e| format!("failed creating {}: {e}", parent.display()))?;
            }
        }
        let json = render_summary_json(&summary);
        fs::write(json_p, json).map_err(|e| format!("failed writing {}: {e}", json_p.display()))?;
        println!("Wrote JSON benchmark report -> {}", json_p.display());
    }

    let n_failed = summary
        .records
        .iter()
        .filter(|r| matches!(r.outcome, EntryBenchmarkOutcome::Failed(_)))
        .count();
    if n_failed > 0 {
        return Err(format!(
            "{n_failed} present real-benchmark entry(s) failed processing or scoring (see diagnostics above)"
        ));
    }

    Ok(summary)
}

/// Aggregate macro/micro metrics, per-protein HC abstention vs wrong-HC statistics,
/// worst/median per-protein HC precision, and pooled + per-protein reliability curves (`P1-1`).
pub fn summarize_real_benchmark_records(
    split: &str,
    solver_label: String,
    seeds: usize,
    hc_threshold: f64,
    records: Vec<EntryBenchmarkRecord>,
) -> RealBenchmarkSummary {
    let mut n_ok = 0usize;
    let mut n_hc_active_proteins = 0usize;
    let mut n_hc_abstained_proteins = 0usize;
    let mut sum_acc = 0.0f64;
    let mut sum_rec = 0.0f64;
    let mut sum_active_hcp = 0.0f64;
    let mut sum_hcr = 0.0f64;
    let mut sum_ece = 0.0f64;
    let mut sum_ece_10 = 0.0f64;
    let mut sum_debiased_ece_10 = 0.0f64;
    let mut sum_ece_10_floor = 0.0f64;
    let mut sum_ece_12 = 0.0f64;

    let mut active_hc_precisions: Vec<(String, String, f64)> = Vec::new();
    let mut pooled_predictions: Vec<(f64, bool)> = Vec::new();

    let mut micro_assignable = 0usize;
    let mut micro_assigned = 0usize;
    let mut micro_correct = 0usize;
    let mut micro_hc_assigned = 0usize;
    let mut micro_hc_correct = 0usize;

    for rec in &records {
        if let EntryBenchmarkOutcome::Succeeded(m) = &rec.outcome {
            n_ok += 1;
            sum_acc += m.mean_accuracy;
            sum_rec += m.mean_recall;
            sum_hcr += m.mean_hc_recall;
            sum_ece += m.reliability.ece;
            sum_ece_10 += m.reliability.ece_10;
            sum_debiased_ece_10 += m.reliability.debiased_ece_10;
            sum_ece_10_floor += m.reliability.ece_10_bernoulli_floor;
            sum_ece_12 += m.reliability.ece_12;
            pooled_predictions.extend_from_slice(&m.predictions);

            if let Some(hcp) = m.reliability.hc_precision {
                n_hc_active_proteins += 1;
                sum_active_hcp += hcp;
                active_hc_precisions.push((
                    rec.manifest_entry.bmrb_id.clone(),
                    rec.manifest_entry.protein.clone(),
                    hcp,
                ));
            } else {
                n_hc_abstained_proteins += 1;
            }

            micro_assignable += m.total_assignable;
            micro_assigned += m.total_assigned;
            micro_correct += m.total_correct;
            micro_hc_assigned += m.total_hc_assigned;
            micro_hc_correct += m.total_hc_correct;
        }
    }

    active_hc_precisions.sort_by(|a, b| a.2.total_cmp(&b.2).then_with(|| a.0.cmp(&b.0)));
    let worst_protein_hc_precision = active_hc_precisions.first().cloned();
    let median_protein_hc_precision = if active_hc_precisions.is_empty() {
        None
    } else {
        let mid = active_hc_precisions.len() / 2;
        if active_hc_precisions.len().is_multiple_of(2) {
            Some(0.5 * (active_hc_precisions[mid - 1].2 + active_hc_precisions[mid].2))
        } else {
            Some(active_hc_precisions[mid].2)
        }
    };

    let denom_ok = if n_ok > 0 { n_ok as f64 } else { 1.0 };
    let pooled_reliability = ProteinReliabilityCurve::from_predictions_with_repeats(
        &pooled_predictions,
        hc_threshold,
        seeds.max(1) as f64,
    );

    RealBenchmarkSummary {
        split: split.to_string(),
        solver_label,
        seeds,
        hc_threshold,
        records,
        n_hc_active_proteins,
        n_hc_abstained_proteins,
        worst_protein_hc_precision,
        median_protein_hc_precision,
        mean_protein_ece: if n_ok > 0 { sum_ece / denom_ok } else { 0.0 },
        mean_protein_ece_10: if n_ok > 0 { sum_ece_10 / denom_ok } else { 0.0 },
        mean_protein_debiased_ece_10: if n_ok > 0 {
            sum_debiased_ece_10 / denom_ok
        } else {
            0.0
        },
        mean_protein_ece_10_bernoulli_floor: if n_ok > 0 {
            sum_ece_10_floor / denom_ok
        } else {
            0.0
        },
        mean_protein_ece_12: if n_ok > 0 { sum_ece_12 / denom_ok } else { 0.0 },
        pooled_reliability,
        macro_accuracy: if n_ok > 0 { sum_acc / denom_ok } else { 0.0 },
        macro_recall: if n_ok > 0 { sum_rec / denom_ok } else { 0.0 },
        macro_hc_precision: if n_hc_active_proteins > 0 {
            sum_active_hcp / (n_hc_active_proteins as f64)
        } else {
            0.0
        },
        macro_hc_recall: if n_ok > 0 { sum_hcr / denom_ok } else { 0.0 },
        micro_assignable,
        micro_assigned,
        micro_correct,
        micro_hc_assigned,
        micro_hc_correct,
        micro_accuracy: if micro_assigned > 0 {
            (micro_correct as f64) / (micro_assigned as f64)
        } else {
            0.0
        },
        micro_recall: if micro_assignable > 0 {
            (micro_correct as f64) / (micro_assignable as f64)
        } else {
            0.0
        },
        micro_hc_precision: if micro_hc_assigned > 0 {
            (micro_hc_correct as f64) / (micro_hc_assigned as f64)
        } else {
            0.0
        },
        micro_hc_recall: if micro_assignable > 0 {
            (micro_hc_correct as f64) / (micro_assignable as f64)
        } else {
            0.0
        },
    }
}

fn format_reliability_bins_inline(rel: &ProteinReliabilityCurve) -> String {
    rel.bins
        .iter()
        .enumerate()
        .map(|(idx, b)| {
            let close = if idx + 1 == rel.bins.len() { ']' } else { ')' };
            if b.count > 0 {
                format!(
                    "[{:.1}-{:.1}{close}: {}/{} ({:.1}%)",
                    b.lower,
                    b.upper,
                    b.correct,
                    b.count,
                    b.empirical_acc * 100.0
                )
            } else {
                format!("[{:.1}-{:.1}{close}: 0/0 (-)", b.lower, b.upper)
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

pub fn format_real_benchmark_report(summary: &RealBenchmarkSummary) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let _ = writeln!(
        out,
        "=== Real Time-Domain Benchmark (split={}, solver={}, seeds={}) ===",
        summary.split, summary.solver_label, summary.seeds
    );
    let _ = writeln!(
        out,
        "{:<16} {:<14} {:>4} {:<7} {:<7} | {:>8} {:>8} {:>8} | {:<40}",
        "BMRB",
        "Protein",
        "Res",
        "Vendor",
        "QC",
        "Accuracy",
        "Recall",
        "HC-Prec",
        "Global Offset (pred - truth) / Status"
    );
    let _ = writeln!(out, "{}", "-".repeat(132));

    let mut n_ok = 0usize;
    let mut n_skip = 0usize;
    let mut n_fail = 0usize;

    for rec in &summary.records {
        let e = &rec.manifest_entry;
        match &rec.outcome {
            EntryBenchmarkOutcome::SkippedMissingData { expected_path } => {
                n_skip += 1;
                let _ = writeln!(
                    out,
                    "{:<16} {:<14} {:>4} {:<7} {:<7} | {:>8} {:>8} {:>8} | SKIPPED (missing local data: {})",
                    e.bmrb_id,
                    truncate_col(&e.protein, 14),
                    e.residues,
                    truncate_col(&e.vendor, 7),
                    "-",
                    "-",
                    "-",
                    "-",
                    expected_path
                );
            }
            EntryBenchmarkOutcome::Failed(diag) => {
                n_fail += 1;
                let _ = writeln!(
                    out,
                    "{:<16} {:<14} {:>4} {:<7} {:<7} | {:>8} {:>8} {:>8} | FAILED [{}]",
                    e.bmrb_id,
                    truncate_col(&e.protein, 14),
                    e.residues,
                    truncate_col(&diag.vendor, 7),
                    "ERR",
                    "ERR",
                    "ERR",
                    "ERR",
                    diag.stage
                );
            }
            EntryBenchmarkOutcome::Succeeded(m) => {
                n_ok += 1;
                let hc_col = if m.reliability.is_abstained() {
                    format!("{:>8}", "ABSTAIN")
                } else {
                    format!("{:>7.1}%", m.mean_hc_precision * 100.0)
                };
                let off_str = format!(
                    "{} ({:.1}s)",
                    m.detailed_offsets.format_compact(),
                    m.elapsed_secs
                );
                let _ = writeln!(
                    out,
                    "{:<16} {:<14} {:>4} {:<7} {:<7} | {:>7.1}% {:>7.1}% {} | {}",
                    e.bmrb_id,
                    truncate_col(&e.protein, 14),
                    e.residues,
                    truncate_col(&e.vendor, 7),
                    m.qc_report.verdict.as_str(),
                    m.mean_accuracy * 100.0,
                    m.mean_recall * 100.0,
                    hc_col,
                    off_str
                );
            }
        }
    }

    let _ = writeln!(out, "{}", "-".repeat(132));
    if n_ok > 0 {
        let macro_hc_col = if summary.n_hc_active_proteins == 0 {
            format!("{:>8}", "ABSTAIN")
        } else {
            format!("{:>7.1}%", summary.macro_hc_precision * 100.0)
        };
        let micro_hc_col = if summary.micro_hc_assigned == 0 {
            format!("{:>8}", "ABSTAIN")
        } else {
            format!("{:>7.1}%", summary.micro_hc_precision * 100.0)
        };
        let _ = writeln!(
            out,
            "MACRO MEAN ({n_ok} evaluated)                           | {:>7.1}% {:>7.1}% {} | HC-Recall={:.1}% (HC-active={}/{n_ok}, abstained={})",
            summary.macro_accuracy * 100.0,
            summary.macro_recall * 100.0,
            macro_hc_col,
            summary.macro_hc_recall * 100.0,
            summary.n_hc_active_proteins,
            summary.n_hc_abstained_proteins,
        );
        let _ = writeln!(
            out,
            "MICRO POOL ({}/{:<5} correct/assigned)              | {:>7.1}% {:>7.1}% {} | HC={}/{} (assignable={})",
            summary.micro_correct,
            summary.micro_assigned,
            summary.micro_accuracy * 100.0,
            summary.micro_recall * 100.0,
            micro_hc_col,
            summary.micro_hc_correct,
            summary.micro_hc_assigned,
            summary.micro_assignable,
        );

        let mut cohort_order: Vec<String> = Vec::new();
        for rec in &summary.records {
            if !cohort_order.contains(&rec.manifest_entry.cohort) {
                cohort_order.push(rec.manifest_entry.cohort.clone());
            }
        }
        if cohort_order.len() > 1 {
            let _ = writeln!(out, "\nCohort Breakdown ({}):", cohort_order.join(" vs "));
            for coh in &cohort_order {
                let sub_recs: Vec<EntryBenchmarkRecord> = summary
                    .records
                    .iter()
                    .filter(|r| &r.manifest_entry.cohort == coh)
                    .cloned()
                    .collect();
                let sub = summarize_real_benchmark_records(
                    coh,
                    summary.solver_label.clone(),
                    summary.seeds,
                    summary.hc_threshold,
                    sub_recs,
                );
                let sub_ok = sub
                    .records
                    .iter()
                    .filter(|r| matches!(r.outcome, EntryBenchmarkOutcome::Succeeded(_)))
                    .count();
                let sub_hc = if sub.n_hc_active_proteins == 0 {
                    "ABSTAIN".to_string()
                } else {
                    format!("{:.1}%", sub.macro_hc_precision * 100.0)
                };
                let _ = writeln!(
                    out,
                    "  - cohort={:<12} ({}/{} evaluated): Macro Acc={:.1}%, Recall={:.1}%, HC-Prec={}, HC-Recall={:.1}%, Mean ECE4/10/12={:.3}/{:.3}/{:.3}, Pooled ECE4/10/12={:.3}/{:.3}/{:.3}",
                    coh,
                    sub_ok,
                    sub.records.len(),
                    sub.macro_accuracy * 100.0,
                    sub.macro_recall * 100.0,
                    sub_hc,
                    sub.macro_hc_recall * 100.0,
                    sub.mean_protein_ece,
                    sub.mean_protein_ece_10,
                    sub.mean_protein_ece_12,
                    sub.pooled_reliability.ece,
                    sub.pooled_reliability.ece_10,
                    sub.pooled_reliability.ece_12,
                );
            }
        }

        let _ = writeln!(
            out,
            "\nPer-Protein Confidence Calibration & Reliability Curves (threshold >= {:.2}):",
            summary.hc_threshold
        );
        match (
            &summary.worst_protein_hc_precision,
            summary.median_protein_hc_precision,
        ) {
            (Some((w_id, w_prot, w_hcp)), Some(med_hcp)) => {
                let _ = writeln!(
                    out,
                    "  Per-protein HC-Prec : worst={} ({}) {:.1}%, median={:.1}% across {} HC-active protein(s); abstained={}/{n_ok}",
                    w_prot,
                    w_id,
                    w_hcp * 100.0,
                    med_hcp * 100.0,
                    summary.n_hc_active_proteins,
                    summary.n_hc_abstained_proteins,
                );
            }
            _ => {
                let _ = writeln!(
                    out,
                    "  Per-protein HC-Prec : all {n_ok} evaluated protein(s) ABSTAINED (0 HC calls >= {:.2})",
                    summary.hc_threshold
                );
            }
        }
        let _ = writeln!(
            out,
            "  Calibration (ECE)   : mean per-protein ECE4/10/12={:.3}/{:.3}/{:.3} (debiased ECE10={:.3}, Bernoulli floor={:.3}), pooled ECE4/10/12={:.3}/{:.3}/{:.3}",
            summary.mean_protein_ece,
            summary.mean_protein_ece_10,
            summary.mean_protein_ece_12,
            summary.mean_protein_debiased_ece_10,
            summary.mean_protein_ece_10_bernoulli_floor,
            summary.pooled_reliability.ece,
            summary.pooled_reliability.ece_10,
            summary.pooled_reliability.ece_12,
        );
        let _ = writeln!(
            out,
            "  Pooled reliability  : {}",
            format_reliability_bins_inline(&summary.pooled_reliability)
        );
        for rec in &summary.records {
            if let EntryBenchmarkOutcome::Succeeded(m) = &rec.outcome {
                let qc_note = if m.qc_report.reasons.is_empty() {
                    String::new()
                } else {
                    format!(" | QC notes: {}", m.qc_report.reasons.join("; "))
                };
                let _ = writeln!(
                    out,
                    "  - {:<16} ({:<12}): QC={:<7} | HC={:<22} | ECE={:.3} (ECE10={:.3}, ECE12={:.3}) | {}{}",
                    rec.manifest_entry.bmrb_id,
                    truncate_col(&rec.manifest_entry.protein, 12),
                    m.qc_report.verdict.as_str(),
                    m.reliability.hc_status,
                    m.reliability.ece,
                    m.reliability.ece_10,
                    m.reliability.ece_12,
                    format_reliability_bins_inline(&m.reliability),
                    qc_note
                );
            }
        }
    } else {
        let _ = writeln!(
            out,
            "SUMMARY: 0 evaluated, {n_skip} SKIPPED (missing local data), {n_fail} failed"
        );
    }

    if n_fail > 0 {
        let _ = writeln!(out, "\nFailure Diagnostics:");
        for rec in &summary.records {
            if let EntryBenchmarkOutcome::Failed(d) = &rec.outcome {
                let _ = writeln!(
                    out,
                    "  - {} ({}): stage={}, exp={}, vendor={}, quad_mode={}, nus={} -> {}",
                    rec.manifest_entry.bmrb_id,
                    rec.manifest_entry.protein,
                    d.stage,
                    d.experiment.as_deref().unwrap_or("-"),
                    d.vendor,
                    d.quad_mode,
                    d.nus_status,
                    d.reason
                );
            }
        }
    }
    out
}

pub fn print_real_benchmark_report(summary: &RealBenchmarkSummary) {
    print!("{}", format_real_benchmark_report(summary));
}

fn truncate_col(s: &str, width: usize) -> String {
    if s.chars().count() <= width {
        s.to_string()
    } else {
        s.chars().take(width).collect()
    }
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 4);
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn offset_status_slug(status: OffsetFitStatus) -> &'static str {
    match status {
        OffsetFitStatus::Fitted => "fitted",
        OffsetFitStatus::NotSignificant => "not_significant",
        OffsetFitStatus::TooFewValues => "too_few_values",
    }
}

fn render_nucleus_offset_json(label: &str, fit: &NucleusOffsetFit) -> String {
    format!(
        "{{ \"status\": \"{}\", \"compact\": \"{}\", \"applied_ppm\": {:.4}, \"candidate_ppm\": {:.4}, \"log_likelihood_gain\": {:.3}, \"n_values\": {} }}",
        offset_status_slug(fit.status),
        json_escape(&fit.format_compact(label)),
        fit.applied_ppm,
        fit.candidate_ppm,
        fit.log_likelihood_gain,
        fit.n_values,
    )
}

pub(crate) fn render_detailed_offsets_json(d: &DetailedReferenceOffsets, indent: usize) -> String {
    let pad = " ".repeat(indent);
    let pad2 = " ".repeat(indent + 2);
    format!(
        "{{\n\
         {pad2}\"compact\": \"{}\",\n\
         {pad2}\"H\": {},\n\
         {pad2}\"N\": {},\n\
         {pad2}\"CA\": {},\n\
         {pad2}\"CB\": {},\n\
         {pad2}\"C\": {}\n\
         {pad}}}",
        json_escape(&d.format_compact()),
        render_nucleus_offset_json("dH", &d.h),
        render_nucleus_offset_json("dN", &d.n),
        render_nucleus_offset_json("dCA", &d.ca),
        render_nucleus_offset_json("dCB", &d.cb),
        render_nucleus_offset_json("dC", &d.c),
    )
}

pub(crate) fn render_qc_report_json(qc: &DatasetQcReport, indent: usize) -> String {
    let pad = " ".repeat(indent);
    let pad2 = " ".repeat(indent + 2);
    let reasons_json = qc
        .reasons
        .iter()
        .map(|r| format!("\"{}\"", json_escape(r)))
        .collect::<Vec<_>>()
        .join(", ");
    let cal_warn_json = match &qc.calibration_warning {
        Some(w) => format!("\"{}\"", json_escape(w)),
        None => "null".to_string(),
    };
    format!(
        "{{\n\
         {pad2}\"verdict\": \"{}\",\n\
         {pad2}\"confidence_calibrated\": {},\n\
         {pad2}\"warn_sub_hc_uncalibrated\": {},\n\
         {pad2}\"calibration_warning\": {},\n\
         {pad2}\"confidence_scale\": {:.4},\n\
         {pad2}\"confidence_cap\": {:.4},\n\
         {pad2}\"already_applied\": {},\n\
         {pad2}\"n_assignable_amides\": {},\n\
         {pad2}\"n_raw_hsqc_roots\": {},\n\
         {pad2}\"n_amide_hsqc_roots\": {},\n\
         {pad2}\"hsqc_3d_support_frac\": {:.6},\n\
         {pad2}\"supported_roots_per_assignable\": {:.6},\n\
         {pad2}\"aliphatic_support_frac\": {:.6},\n\
         {pad2}\"hsqc_3d_rms_dh_ppm\": {:.6},\n\
         {pad2}\"hsqc_3d_rms_dn_ppm\": {:.6},\n\
         {pad2}\"hsqc_multiplicity_ratio\": {:.6},\n\
         {pad2}\"raw_hsqc_multiplicity_ratio\": {:.6},\n\
         {pad2}\"offset_consistency_ok\": {},\n\
         {pad2}\"connectivity_frac\": {:.6},\n\
         {pad2}\"corroborated_connectivity_frac\": {:.6},\n\
         {pad2}\"solution_margin\": {:.6},\n\
         {pad2}\"hc_call_fraction\": {:.6},\n\
         {pad2}\"protein_hc_gate_triggered\": {},\n\
         {pad2}\"reasons\": [{}],\n\
         {pad2}\"detailed_offsets\": {}\n\
         {pad}}}",
        qc.verdict.as_str(),
        qc.confidence_calibrated,
        qc.warn_sub_hc_uncalibrated,
        cal_warn_json,
        qc.confidence_scale,
        qc.confidence_cap,
        qc.already_applied,
        qc.n_assignable_amides,
        qc.n_raw_hsqc_roots,
        qc.n_amide_hsqc_roots,
        qc.hsqc_3d_support_frac,
        qc.supported_roots_per_assignable,
        qc.aliphatic_support_frac,
        qc.hsqc_3d_rms_dh_ppm,
        qc.hsqc_3d_rms_dn_ppm,
        qc.hsqc_multiplicity_ratio,
        qc.raw_hsqc_multiplicity_ratio,
        qc.offset_consistency_ok,
        qc.connectivity_frac,
        qc.corroborated_connectivity_frac,
        qc.solution_margin,
        qc.hc_call_fraction,
        qc.protein_hc_gate_triggered,
        reasons_json,
        render_detailed_offsets_json(&qc.detailed_offsets, indent + 2),
    )
}

pub(crate) fn render_reliability_curve_json(
    rel: &ProteinReliabilityCurve,
    indent: usize,
) -> String {
    let pad = " ".repeat(indent);
    let pad2 = " ".repeat(indent + 2);
    let pad4 = " ".repeat(indent + 4);
    let hc_prec_json = match rel.hc_precision {
        Some(v) => format!("{v:.6}"),
        None => "null".to_string(),
    };
    let bins_json = rel
        .bins
        .iter()
        .enumerate()
        .map(|(idx, b)| {
            let close = if idx + 1 == rel.bins.len() { ']' } else { ')' };
            format!(
                "{pad4}{{ \"range\": \"[{:.1}, {:.1}{close}\", \"lower\": {:.2}, \"upper\": {:.2}, \"count\": {}, \"correct\": {}, \"mean_conf\": {:.6}, \"empirical_acc\": {:.6} }}",
                b.lower, b.upper, b.lower, b.upper, b.count, b.correct, b.mean_conf, b.empirical_acc
            )
        })
        .collect::<Vec<_>>()
        .join(",\n");
    format!(
        "{{\n\
         {pad2}\"total_predictions\": {},\n\
         {pad2}\"ece\": {:.6},\n\
         {pad2}\"ece_10\": {:.6},\n\
         {pad2}\"debiased_ece_10\": {:.6},\n\
         {pad2}\"ece_10_bernoulli_floor\": {:.6},\n\
         {pad2}\"ece_12\": {:.6},\n\
         {pad2}\"hc_threshold\": {:.4},\n\
         {pad2}\"abstained\": {},\n\
         {pad2}\"hc_assigned\": {},\n\
         {pad2}\"hc_correct\": {},\n\
         {pad2}\"hc_precision\": {},\n\
         {pad2}\"hc_status\": \"{}\",\n\
         {pad2}\"bins\": [\n{bins_json}\n{pad2}]\n\
         {pad}}}",
        rel.total_predictions,
        rel.ece,
        rel.ece_10,
        rel.debiased_ece_10,
        rel.ece_10_bernoulli_floor,
        rel.ece_12,
        rel.hc_threshold,
        rel.is_abstained(),
        rel.hc_assigned,
        rel.hc_correct,
        hc_prec_json,
        json_escape(&rel.hc_status),
    )
}

pub fn render_summary_json(summary: &RealBenchmarkSummary) -> String {
    let n_total = summary.records.len();
    let n_ok = summary
        .records
        .iter()
        .filter(|r| matches!(r.outcome, EntryBenchmarkOutcome::Succeeded(_)))
        .count();
    let n_skip = summary
        .records
        .iter()
        .filter(|r| matches!(r.outcome, EntryBenchmarkOutcome::SkippedMissingData { .. }))
        .count();
    let n_fail = summary
        .records
        .iter()
        .filter(|r| matches!(r.outcome, EntryBenchmarkOutcome::Failed(_)))
        .count();

    let macro_hcp_json = if summary.n_hc_active_proteins > 0 {
        format!("{:.6}", summary.macro_hc_precision)
    } else {
        "null".to_string()
    };
    let micro_hcp_json = if summary.micro_hc_assigned > 0 {
        format!("{:.6}", summary.micro_hc_precision)
    } else {
        "null".to_string()
    };
    let worst_hcp_json = match &summary.worst_protein_hc_precision {
        Some((bid, prot, val)) => format!(
            "{{ \"bmrb_id\": \"{}\", \"protein\": \"{}\", \"hc_precision\": {:.6} }}",
            json_escape(bid),
            json_escape(prot),
            val
        ),
        None => "null".to_string(),
    };
    let median_hcp_json = match summary.median_protein_hc_precision {
        Some(val) => format!("{val:.6}"),
        None => "null".to_string(),
    };

    let mut j = String::new();
    j.push_str("{\n");
    j.push_str(&format!(
        "  \"tool\": \"{}\",\n  \"split\": \"{}\",\n  \"solver\": \"{}\",\n  \"seeds\": {},\n  \"hc_threshold\": {:.4},\n",
        json_escape(&version_string()),
        json_escape(&summary.split),
        json_escape(&summary.solver_label),
        summary.seeds,
        summary.hc_threshold,
    ));
    j.push_str(&format!(
        "  \"counts\": {{ \"total\": {n_total}, \"evaluated\": {n_ok}, \"hc_active_proteins\": {}, \"hc_abstained_proteins\": {}, \"skipped_missing_local_data\": {n_skip}, \"failed\": {n_fail} }},\n",
        summary.n_hc_active_proteins,
        summary.n_hc_abstained_proteins,
    ));
    j.push_str(&format!(
        "  \"macro_average\": {{ \"accuracy\": {:.6}, \"recall\": {:.6}, \"hc_precision\": {}, \"hc_recall\": {:.6}, \"hc_active_proteins\": {}, \"hc_abstained_proteins\": {} }},\n",
        summary.macro_accuracy,
        summary.macro_recall,
        macro_hcp_json,
        summary.macro_hc_recall,
        summary.n_hc_active_proteins,
        summary.n_hc_abstained_proteins,
    ));
    j.push_str(&format!(
        "  \"micro_average\": {{ \"accuracy\": {:.6}, \"recall\": {:.6}, \"hc_precision\": {}, \"hc_recall\": {:.6}, \"assignable\": {}, \"assigned\": {}, \"correct\": {}, \"hc_assigned\": {}, \"hc_correct\": {} }},\n",
        summary.micro_accuracy,
        summary.micro_recall,
        micro_hcp_json,
        summary.micro_hc_recall,
        summary.micro_assignable,
        summary.micro_assigned,
        summary.micro_correct,
        summary.micro_hc_assigned,
        summary.micro_hc_correct,
    ));
    j.push_str(&format!(
        "  \"calibration_summary\": {{\n    \"hc_active_proteins\": {},\n    \"hc_abstained_proteins\": {},\n    \"worst_protein_hc_precision\": {},\n    \"median_protein_hc_precision\": {},\n    \"mean_protein_ece\": {:.6},\n    \"mean_protein_ece_10\": {:.6},\n    \"mean_protein_debiased_ece_10\": {:.6},\n    \"mean_protein_ece_10_bernoulli_floor\": {:.6},\n    \"mean_protein_ece_12\": {:.6},\n    \"pooled_reliability_curve\": {}\n  }},\n",
        summary.n_hc_active_proteins,
        summary.n_hc_abstained_proteins,
        worst_hcp_json,
        median_hcp_json,
        summary.mean_protein_ece,
        summary.mean_protein_ece_10,
        summary.mean_protein_debiased_ece_10,
        summary.mean_protein_ece_10_bernoulli_floor,
        summary.mean_protein_ece_12,
        render_reliability_curve_json(&summary.pooled_reliability, 4),
    ));
    let mut cohort_order: Vec<String> = Vec::new();
    for rec in &summary.records {
        if !cohort_order.contains(&rec.manifest_entry.cohort) {
            cohort_order.push(rec.manifest_entry.cohort.clone());
        }
    }
    let cohort_json_items: Vec<String> = cohort_order
        .iter()
        .map(|coh| {
            let sub_recs: Vec<EntryBenchmarkRecord> = summary
                .records
                .iter()
                .filter(|r| &r.manifest_entry.cohort == coh)
                .cloned()
                .collect();
            let sub = summarize_real_benchmark_records(
                coh,
                summary.solver_label.clone(),
                summary.seeds,
                summary.hc_threshold,
                sub_recs,
            );
            let sub_ok = sub
                .records
                .iter()
                .filter(|r| matches!(r.outcome, EntryBenchmarkOutcome::Succeeded(_)))
                .count();
            let sub_hcp = if sub.n_hc_active_proteins > 0 {
                format!("{:.6}", sub.macro_hc_precision)
            } else {
                "null".to_string()
            };
            format!(
                "    {{ \"cohort\": \"{}\", \"total\": {}, \"evaluated\": {}, \"macro_accuracy\": {:.6}, \"macro_recall\": {:.6}, \"macro_hc_precision\": {}, \"macro_hc_recall\": {:.6}, \"hc_active_proteins\": {}, \"hc_abstained_proteins\": {}, \"mean_protein_ece\": {:.6}, \"mean_protein_ece_10\": {:.6}, \"mean_protein_debiased_ece_10\": {:.6}, \"mean_protein_ece_10_bernoulli_floor\": {:.6}, \"mean_protein_ece_12\": {:.6}, \"pooled_ece\": {:.6}, \"pooled_ece_10\": {:.6}, \"pooled_ece_12\": {:.6} }}",
                json_escape(coh),
                sub.records.len(),
                sub_ok,
                sub.macro_accuracy,
                sub.macro_recall,
                sub_hcp,
                sub.macro_hc_recall,
                sub.n_hc_active_proteins,
                sub.n_hc_abstained_proteins,
                sub.mean_protein_ece,
                sub.mean_protein_ece_10,
                sub.mean_protein_debiased_ece_10,
                sub.mean_protein_ece_10_bernoulli_floor,
                sub.mean_protein_ece_12,
                sub.pooled_reliability.ece,
                sub.pooled_reliability.ece_10,
                sub.pooled_reliability.ece_12,
            )
        })
        .collect();
    j.push_str(&format!(
        "  \"cohort_summaries\": [\n{}\n  ],\n",
        cohort_json_items.join(",\n")
    ));
    j.push_str("  \"entries\": [\n");
    for (idx, rec) in summary.records.iter().enumerate() {
        let comma = if idx + 1 < summary.records.len() {
            ","
        } else {
            ""
        };
        let e = &rec.manifest_entry;
        j.push_str("    {\n");
        j.push_str(&format!(
            "      \"bmrb_id\": \"{}\",\n      \"protein\": \"{}\",\n      \"residues\": {},\n      \"field_mhz\": {:.2},\n      \"vendor\": \"{}\",\n      \"split\": \"{}\",\n      \"cohort\": \"{}\",\n",
            json_escape(&e.bmrb_id),
            json_escape(&e.protein),
            e.residues,
            e.field_mhz,
            json_escape(&e.vendor),
            json_escape(&e.split),
            json_escape(&e.cohort),
        ));
        match &rec.outcome {
            EntryBenchmarkOutcome::SkippedMissingData { expected_path } => {
                j.push_str(&format!(
                    "      \"status\": \"SKIPPED (missing local data)\",\n      \"expected_path\": \"{}\"\n",
                    json_escape(expected_path)
                ));
            }
            EntryBenchmarkOutcome::Failed(d) => {
                let exp_json = match &d.experiment {
                    Some(ex) => format!("\"{}\"", json_escape(ex)),
                    None => "null".to_string(),
                };
                j.push_str(&format!(
                    "      \"status\": \"FAILED\",\n      \"failure\": {{ \"stage\": \"{}\", \"experiment\": {}, \"vendor\": \"{}\", \"quad_mode\": \"{}\", \"nus\": \"{}\", \"reason\": \"{}\" }}\n",
                    json_escape(&d.stage),
                    exp_json,
                    json_escape(&d.vendor),
                    json_escape(&d.quad_mode),
                    json_escape(&d.nus_status),
                    json_escape(&d.reason),
                ));
            }
            EntryBenchmarkOutcome::Succeeded(m) => {
                let exps_json = m
                    .discovered_experiments
                    .iter()
                    .map(|(k, p)| {
                        format!(
                            "{{\"exp\":\"{}\",\"path\":\"{}\"}}",
                            json_escape(k),
                            json_escape(p)
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                let entry_hcp_json = match m.reliability.hc_precision {
                    Some(v) => format!("{v:.6}"),
                    None => "null".to_string(),
                };
                j.push_str(&format!(
                    "      \"status\": \"OK\",\n      \"qc_verdict\": \"{}\",\n      \"discovered_experiments\": [{}],\n      \"hsqc_roots\": {},\n      \"total_peaks\": {},\n      \"assignable_per_seed\": {},\n      \"accuracy\": {:.6},\n      \"recall\": {:.6},\n      \"hc_abstained\": {},\n      \"hc_precision\": {},\n      \"hc_status\": \"{}\",\n      \"hc_recall\": {:.6},\n      \"global_offsets_ppm\": {{ \"H\": {:.4}, \"N\": {:.4}, \"CA\": {:.4}, \"CB\": {:.4}, \"C\": {:.4} }},\n      \"detailed_offsets_compact\": \"{}\",\n      \"detailed_offsets\": {},\n      \"prior_detailed_offsets\": {},\n      \"qc_report\": {},\n      \"reliability_curve\": {},\n      \"elapsed_secs\": {:.3}\n",
                    m.qc_report.verdict.as_str(),
                    exps_json,
                    m.n_hsqc_roots,
                    m.n_total_peaks,
                    m.assignable_per_seed,
                    m.mean_accuracy,
                    m.mean_recall,
                    m.reliability.is_abstained(),
                    entry_hcp_json,
                    json_escape(&m.reliability.hc_status),
                    m.mean_hc_recall,
                    m.global_offsets.h,
                    m.global_offsets.n,
                    m.global_offsets.ca,
                    m.global_offsets.cb,
                    m.global_offsets.c,
                    json_escape(&m.detailed_offsets.format_compact()),
                    render_detailed_offsets_json(&m.detailed_offsets, 6),
                    render_detailed_offsets_json(&m.prior_detailed_offsets, 6),
                    render_qc_report_json(&m.qc_report, 6),
                    render_reliability_curve_json(&m.reliability, 6),
                    m.elapsed_secs,
                ));
            }
        }
        j.push_str(&format!("    }}{comma}\n"));
    }
    j.push_str("  ]\n}\n");
    j
}

pub fn parse_real_benchmark_args(args: &[String]) -> Result<RealBenchmarkConfig, String> {
    let mut manifest_path = PathBuf::from("data/benchmark/real_manifest.tsv");
    let mut split_opt: Option<BenchmarkSplitFilter> = None;
    let mut solver = SolverKind::EvoGraph;
    let mut engine_opt: Option<EngineKind> = None;
    let mut seeds = 5usize;
    let mut base_seed = 0u64;
    let mut data_root: Option<PathBuf> = None;
    let mut priors_path: Option<String> = None;
    let mut runs_opt: Option<usize> = None;
    let mut steps_opt: Option<usize> = None;
    let mut hc_threshold = 0.8f64;
    let mut no_crop = false;
    let mut lp = false;
    let mut json_out: Option<PathBuf> = None;
    let mut verbose = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--manifest" => {
                i += 1;
                manifest_path = PathBuf::from(args.get(i).ok_or("missing value for --manifest")?);
            }
            "--split" => {
                i += 1;
                split_opt = Some(BenchmarkSplitFilter::parse(
                    args.get(i).ok_or("missing value for --split")?,
                )?);
            }
            "--solver" => {
                i += 1;
                solver = parse_solver(args.get(i).ok_or("missing value for --solver")?)?;
            }
            "--engine" => {
                i += 1;
                engine_opt = Some(parse_engine(
                    args.get(i).ok_or("missing value for --engine")?,
                )?);
            }
            "--seeds" => {
                i += 1;
                seeds = args
                    .get(i)
                    .ok_or("missing value for --seeds")?
                    .parse::<usize>()
                    .map_err(|e| format!("{e}"))?
                    .max(1);
            }
            "--seed" => {
                i += 1;
                base_seed = args
                    .get(i)
                    .ok_or("missing value for --seed")?
                    .parse::<u64>()
                    .map_err(|e| format!("{e}"))?;
            }
            "--data-root" => {
                i += 1;
                data_root = Some(PathBuf::from(
                    args.get(i).ok_or("missing value for --data-root")?,
                ));
            }
            "--priors" => {
                i += 1;
                priors_path = args.get(i).cloned();
            }
            "--runs" => {
                i += 1;
                runs_opt = Some(parse_bounded_runs(
                    args.get(i).ok_or("missing value for --runs")?,
                )?);
            }
            "--steps" => {
                i += 1;
                steps_opt = Some(parse_bounded_steps(
                    args.get(i).ok_or("missing value for --steps")?,
                )?);
            }
            "--hc" => {
                i += 1;
                hc_threshold = args
                    .get(i)
                    .ok_or("missing value for --hc")?
                    .parse()
                    .map_err(|e| format!("{e}"))?;
            }
            "--json-out" => {
                i += 1;
                json_out = Some(PathBuf::from(
                    args.get(i).ok_or("missing value for --json-out")?,
                ));
            }
            "--no-crop" => no_crop = true,
            "--lp" => lp = true,
            "-v" | "--verbose" => verbose = true,
            other => return Err(format!("unknown real-benchmark flag {other:?}")),
        }
        i += 1;
    }

    let split = split_opt.ok_or("real-benchmark requires --split dev|heldout")?;
    let allow_repo_fallbacks = data_root.is_none();
    let effective_data_root = data_root.unwrap_or_else(|| PathBuf::from("data/private/bmrb"));

    Ok(RealBenchmarkConfig {
        manifest_path,
        split,
        solver,
        engine_opt,
        seeds,
        base_seed,
        data_root: effective_data_root,
        allow_repo_fallbacks,
        priors_path,
        runs_opt,
        steps_opt,
        hc_threshold,
        no_crop,
        lp,
        json_out,
        verbose,
    })
}

pub fn cmd_real_benchmark(args: &[String]) -> Result<(), String> {
    let cfg = parse_real_benchmark_args(args)?;
    let no_crop = cfg.no_crop;
    let lp = cfg.lp;
    run_real_benchmark_with_loaders(
        &cfg,
        &|dir| process_fid_dir_2d(dir, no_crop, lp).map(|(s, _)| s),
        &|dir, exp| process_fid_dir_3d(dir, exp, no_crop, lp).map(|(s, _)| s),
    )
    .map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;
    use nmr_core::nmrstar::read_star;
    use nmr_core::simulate::{
        render_hsqc_2d, render_triple_3d, simulate, SimConfig, SpectrumRenderConfig,
    };
    use std::process;

    #[test]
    fn test_manifest_parsing_and_missing_local_data_skips_cleanly() {
        let dir = env::temp_dir().join(format!("nmr_real_bench_skip_{}", process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        let manifest = dir.join("real_manifest.tsv");
        fs::write(
            &manifest,
            "bmrb_id\tprotein\tresidues\tfield_mhz\tvendor\texperiments\tsplit\n\
             ubiquitin_local\tUbiquitin\t76\t500\tBruker\tHSQC,HNCACB,CBCACONH,HNCO,HNCACO\tdev\n\
             51439\tCISD3\t63\t900\tBruker\tHSQC,HNCO,HNCA,HNCOCA,CBCACONH\tdev\n\
             15477\tTargetX\t90\t600\tVarian\tHSQC,HNCO,HNCACB,CBCACONH\theldout\n",
        )
        .unwrap();

        let json_out = dir.join("out.json");
        let empty_root = dir.join("empty_data_root");
        let args = vec![
            "--manifest".to_string(),
            manifest.to_string_lossy().into_owned(),
            "--split".to_string(),
            "dev".to_string(),
            "--solver".to_string(),
            "evo-graph".to_string(),
            "--data-root".to_string(),
            empty_root.to_string_lossy().into_owned(),
            "--json-out".to_string(),
            json_out.to_string_lossy().into_owned(),
        ];

        // When local directories are absent, `real-benchmark` marks them `SKIPPED (missing local data)` and returns `Ok(())`.
        cmd_real_benchmark(&args).unwrap();

        let json_txt = fs::read_to_string(&json_out).unwrap();
        assert!(json_txt.contains("\"skipped_missing_local_data\": 2"));
        assert!(json_txt.contains("\"failed\": 0"));
        assert!(json_txt.contains("SKIPPED (missing local data)"));

        // Verify sibling `<repo>/data/ubiquitin` & `<repo>/data/private/bmrb51439_cisd3`
        // resolve when `--data-root` has shape `<repo>/data/private/bmrb`
        let fake_repo_data = dir.join("fake_repo/data");
        let fake_bmrb_root = fake_repo_data.join("private/bmrb");
        fs::create_dir_all(fake_repo_data.join("ubiquitin")).unwrap();
        fs::write(
            fake_repo_data.join("ubiquitin_backbone_assignment_191104.txt"),
            "truth",
        )
        .unwrap();
        fs::create_dir_all(fake_repo_data.join("private/bmrb51439_cisd3")).unwrap();
        fs::create_dir_all(&fake_bmrb_root).unwrap();
        let ubq_sib = resolve_local_entry_paths("ubiquitin_local", &fake_bmrb_root, false).unwrap();
        assert_eq!(ubq_sib.entry_dir, fake_repo_data.join("ubiquitin"));
        assert_eq!(
            ubq_sib.truth_path,
            Some(fake_repo_data.join("ubiquitin_backbone_assignment_191104.txt"))
        );
        let cisd_sib = resolve_local_entry_paths("51439", &fake_bmrb_root, false).unwrap();
        assert_eq!(
            cisd_sib.entry_dir,
            fake_repo_data.join("private/bmrb51439_cisd3")
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_varian_and_uppercase_bruker_case_safe_fid_discovery_and_e2e_scoring() {
        let dir = env::temp_dir().join(format!("nmr_real_bench_disc_{}", process::id()));
        let _ = fs::remove_dir_all(&dir);
        let data_root = dir.join("bmrb");
        let varian_entry = data_root.join("77777");
        fs::create_dir_all(&varian_entry).unwrap();

        let fixture_star = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../data/fixtures/synthetic_ubiquitin.str");
        fs::copy(&fixture_star, varian_entry.join("bmr77777_3.str")).unwrap();

        // Write Varian BioPack `procpar` + non-empty `fid` subdirectories (`gNhsqc`, `ghnco`, `ghncacb`, `ghn_co_cacb`)
        let write_varian_exp =
            |sub: &str, pslabel: &str, ni: usize, ni2: usize, sw1: f64, sw2: f64| {
                let p = varian_entry.join(sub);
                fs::create_dir_all(&p).unwrap();
                let procpar = format!(
                    "pslabel 2 2 8192 0 0 2 1 1 1 64\n\
                 1 \"{pslabel}\"\n\
                 0\n\
                 tn 2 2 8192 0 0 2 1 1 1 64\n\
                 1 \"H1\"\n\
                 0\n\
                 dn 2 2 8192 0 0 2 1 1 1 64\n\
                 1 \"C13\"\n\
                 0\n\
                 dn2 2 2 8192 0 0 2 1 1 1 64\n\
                 1 \"N15\"\n\
                 0\n\
                 sfrq 1 1 10000 0 0 2 1 1 1 64\n\
                 1 600.133\n\
                 0\n\
                 dfrq 1 1 10000 0 0 2 1 1 1 64\n\
                 1 150.903\n\
                 0\n\
                 dfrq2 1 1 10000 0 0 2 1 1 1 64\n\
                 1 60.818\n\
                 0\n\
                 ni 1 1 10000 0 0 2 1 1 1 64\n\
                 1 {ni}\n\
                 0\n\
                 ni2 1 1 10000 0 0 2 1 1 1 64\n\
                 1 {ni2}\n\
                 0\n\
                 sw1 1 1 100000 0 0 2 1 1 1 64\n\
                 1 {sw1}\n\
                 0\n\
                 sw2 1 1 100000 0 0 2 1 1 1 64\n\
                 1 {sw2}\n\
                 0\n\
                 phase 1 1 10 0 0 2 1 1 1 64\n\
                 2 1 2\n\
                 0\n\
                 phase2 1 1 10 0 0 2 1 1 1 64\n\
                 2 1 2\n\
                 0\n"
                );
                fs::write(p.join("procpar"), procpar).unwrap();
                fs::write(p.join("fid"), b"dummy_raw_fid_payload").unwrap();
            };

        write_varian_exp("exp1_root.fid", "gNhsqc", 128, 1, 2000.0, 0.0);
        write_varian_exp("exp2_co.fid", "ghnco", 48, 32, 2200.0, 2000.0);
        write_varian_exp("exp3_cacb.fid", "ghncacb", 64, 32, 11000.0, 2000.0);
        write_varian_exp("exp4_inter.fid", "ghn_co_cacb", 64, 32, 11000.0, 2000.0);
        // Relaxation distractor that must be ignored
        write_varian_exp("exp5_t1.fid", "gNhsqcT1", 128, 1, 2000.0, 0.0);

        let (disc, _) = discover_entry_experiments(&varian_entry, Some(600.0)).unwrap();
        let names: Vec<&str> = disc.iter().map(|d| d.exp_type.as_str()).collect();
        assert_eq!(names, vec!["HSQC", "HNCO", "HNCACB", "CBCACONH"]);

        // Also verify uppercase UXNMR Bruker (`ACQUS`, `ACQU2S`, `SER`) case-safe discovery
        let bruker_up = data_root.join("88888/EXP_UPPER");
        fs::create_dir_all(&bruker_up).unwrap();
        fs::write(
            bruker_up.join("ACQUS"),
            "##$PULPROG= <hsqcetfpf3gpsi>\n##$NUC1= <1H>\n##$SFO1= 500.13\n##$SW= 14.0\n##$TD= 1024\n##$AQ_mod= 3\n",
        )
        .unwrap();
        fs::write(
            bruker_up.join("ACQU2S"),
            "##$NUC1= <15N>\n##$SFO1= 50.68\n##$SW= 34.0\n##$TD= 128\n##$FnMODE= 6\n",
        )
        .unwrap();
        fs::write(bruker_up.join("SER"), b"raw_bruker_ser").unwrap();
        let (disc_b, _) =
            discover_entry_experiments(&data_root.join("88888"), Some(500.0)).unwrap();
        assert_eq!(disc_b.len(), 1);
        assert_eq!(disc_b[0].exp_type, "HSQC");

        // End-to-end `run_real_benchmark_with_loaders` test with systematic `-2.75 ppm` 13C offset
        let ref_data = read_star(&fixture_star, None).unwrap();
        let (sim_peaks, _) = simulate(
            &ref_data,
            &SimConfig {
                experiments: vec![
                    "HSQC".into(),
                    "HNCO".into(),
                    "HNCACB".into(),
                    "CBCACONH".into(),
                ],
                p_missing_peak: 0.0,
                spurious_rate: 0.0,
                ref_offset_ca: -2.75,
                ref_offset_cb: -2.75,
                ref_offset_c: -2.75,
                seed: 11,
                ..SimConfig::default()
            },
        );
        let rcfg = SpectrumRenderConfig::default();
        let s_hsqc = render_hsqc_2d(&sim_peaks, &rcfg);
        let s_hnco = render_triple_3d("HNCO", &sim_peaks, &rcfg);
        let s_hncacb = render_triple_3d("HNCACB", &sim_peaks, &rcfg);
        let s_cbcaconh = render_triple_3d("CBCACONH", &sim_peaks, &rcfg);

        let manifest = dir.join("manifest.tsv");
        fs::write(
            &manifest,
            "bmrb_id\tprotein\tresidues\tfield_mhz\tvendor\texperiments\tsplit\n\
             77777\tUbqVarian\t76\t600\tVarian\tHSQC,HNCO,HNCACB,CBCACONH\tdev\n",
        )
        .unwrap();

        let cfg = RealBenchmarkConfig {
            manifest_path: manifest,
            split: BenchmarkSplitFilter::Dev,
            solver: SolverKind::Standard,
            engine_opt: Some(EngineKind::Cpu),
            seeds: 2,
            base_seed: 0,
            data_root: data_root.clone(),
            allow_repo_fallbacks: false,
            priors_path: None,
            runs_opt: Some(8),
            steps_opt: Some(30_000),
            hc_threshold: 0.8,
            no_crop: false,
            lp: false,
            json_out: Some(dir.join("summary.json")),
            verbose: false,
        };

        let summary =
            run_real_benchmark_with_loaders(&cfg, &|_| Ok(s_hsqc.clone()), &|_, exp| match exp {
                "HNCO" => Ok(s_hnco.clone()),
                "HNCACB" => Ok(s_hncacb.clone()),
                "CBCACONH" => Ok(s_cbcaconh.clone()),
                other => Err(format!("unexpected exp {other}")),
            })
            .unwrap();

        assert_eq!(summary.records.len(), 1);
        match &summary.records[0].outcome {
            EntryBenchmarkOutcome::Succeeded(m) => {
                assert!(m.mean_accuracy >= 0.85, "acc={}", m.mean_accuracy);
                assert!(
                    (m.global_offsets.ca - (-2.75)).abs() < 0.35,
                    "expected ~ -2.75 ppm unlabelled dCA offset, got {}",
                    m.global_offsets.ca
                );
                assert_eq!(m.detailed_offsets.ca.status, OffsetFitStatus::Fitted);
                assert!(!m.reliability.is_abstained());
            }
            other => panic!("expected Succeeded, got {other:?}"),
        }

        let summary_json_txt = fs::read_to_string(dir.join("summary.json")).unwrap();
        for required_key in [
            "\"warn_sub_hc_uncalibrated\":",
            "\"aliphatic_support_frac\":",
            "\"corroborated_connectivity_frac\":",
            "\"hc_call_fraction\":",
            "\"protein_hc_gate_triggered\":",
            "\"debiased_ece_10\":",
            "\"ece_10_bernoulli_floor\":",
            "\"mean_protein_debiased_ece_10\":",
            "\"mean_protein_ece_10_bernoulli_floor\":",
        ] {
            assert!(
                summary_json_txt.contains(required_key),
                "summary.json missing Phase 13 key {required_key}"
            );
        }

        // Also smoke-check discovery on local Ubiquitin & CISD3 if present on developer machine
        if let Ok(ubq_loc) = resolve_local_entry_paths("ubiquitin_local", &data_root, true) {
            let (ubq_disc, _) =
                discover_entry_experiments(&ubq_loc.entry_dir, Some(500.0)).unwrap();
            let ubq_map: BTreeMap<String, String> = ubq_disc
                .into_iter()
                .map(|d| (d.exp_type, d.rel_path))
                .collect();
            assert_eq!(
                ubq_map.get("HSQC").map(String::as_str),
                Some("Ubiquitin_090309/2")
            );
            assert_eq!(
                ubq_map.get("HNCACB").map(String::as_str),
                Some("Ubiquitin_090309/3")
            );
            assert_eq!(
                ubq_map.get("CBCACONH").map(String::as_str),
                Some("Ubiquitin_090309/4")
            );
            assert_eq!(
                ubq_map.get("HNCO").map(String::as_str),
                Some("Ubiquitin_111909/10")
            );
            assert_eq!(
                ubq_map.get("HNCACO").map(String::as_str),
                Some("Ubiquitin_111909/11")
            );
        }
        if let Ok(cisd_loc) = resolve_local_entry_paths("51439", &data_root, true) {
            let (cisd_disc, _) =
                discover_entry_experiments(&cisd_loc.entry_dir, Some(900.0)).unwrap();
            let cisd_exps: Vec<&str> = cisd_disc.iter().map(|d| d.exp_type.as_str()).collect();
            assert_eq!(
                cisd_exps,
                vec!["HSQC", "HNCO", "HNCA", "HNCOCA", "CBCACONH"]
            );
        }

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_committed_real_manifest_tsv_schema_splits_and_disjointness() {
        let manifest_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../data/benchmark/real_manifest.tsv");
        let raw = fs::read_to_string(&manifest_path).unwrap();
        let entries = parse_real_manifest_str(&raw).unwrap();
        let n_dev = entries.iter().filter(|e| e.split == "dev").count();
        let n_held = entries.iter().filter(|e| e.split == "heldout").count();
        assert_eq!(
            n_dev, 13,
            "expected 13 dev entries (10 dev + 3 dev_from_p10)"
        );

        // Verify unique BMRB IDs across all entries
        let unique_ids: BTreeSet<&str> = entries.iter().map(|e| e.bmrb_id.as_str()).collect();
        assert_eq!(unique_ids.len(), entries.len());

        // Verify Q1 + P1-7 governance transitions:
        // - 5329, 6318, 15476, 16385, 53280 in `dev` (`cohort=dev`),
        // - 16072, 16782, 17092 promoted from `cold_p10` to `dev` (`cohort=dev_from_p10`),
        // - 5845, 17030, 18145, 51351, 16099, 18489 tagged `seen_p8_p9` (`heldout`),
        // - Former `cold_p12` entries relabeled `seen_p12` (`heldout`, spent at `v0.8.0` per `P1-7`).
        for promoted_id in ["5329", "6318", "15476", "16385", "53280"] {
            let e = entries
                .iter()
                .find(|e| e.bmrb_id == promoted_id)
                .unwrap_or_else(|| panic!("missing promoted entry {promoted_id}"));
            assert_eq!(e.split, "dev", "expected {promoted_id} in dev split");
            assert_eq!(e.cohort, "dev", "expected {promoted_id} in dev cohort");
        }
        for dev_p10_id in ["16072", "16782", "17092"] {
            let e = entries
                .iter()
                .find(|e| e.bmrb_id == dev_p10_id)
                .unwrap_or_else(|| panic!("missing dev_from_p10 entry {dev_p10_id}"));
            assert_eq!(e.split, "dev", "expected {dev_p10_id} in dev split");
            assert_eq!(
                e.cohort, "dev_from_p10",
                "expected {dev_p10_id} in dev_from_p10 cohort"
            );
            assert!((50..=160).contains(&e.residues));
        }
        let n_seen = entries
            .iter()
            .filter(|e| e.split == "heldout" && e.cohort == "seen_p8_p9")
            .count();
        let n_seen_p12 = entries
            .iter()
            .filter(|e| e.split == "heldout" && e.cohort == "seen_p12")
            .count();
        let n_cold_p12 = entries.iter().filter(|e| e.cohort == "cold_p12").count();
        assert_eq!(n_seen, 6);
        assert_eq!(
            n_cold_p12, 0,
            "cold_p12 was evaluated at v0.8.0 and must be relabeled seen_p12 per P1-7"
        );
        assert!(
            (6..=8).contains(&n_seen_p12),
            "expected 6..=8 seen_p12 entries, got {n_seen_p12}"
        );
        assert_eq!(n_held, n_seen + n_seen_p12);
        for seen_id in ["5845", "17030", "18145", "51351", "16099", "18489"] {
            let e = entries
                .iter()
                .find(|e| e.bmrb_id == seen_id)
                .unwrap_or_else(|| panic!("missing seen heldout entry {seen_id}"));
            assert_eq!(e.split, "heldout");
            assert_eq!(e.cohort, "seen_p8_p9");
        }
        for e in entries
            .iter()
            .filter(|e| e.split == "heldout" && e.cohort == "seen_p12")
        {
            let seen_p12_id = e.bmrb_id.as_str();
            assert!((50..=160).contains(&e.residues));
            // Header-only verification (strictly zero FID FFT, peak picking, or solver runs!)
            if let Ok(loc) =
                resolve_local_entry_paths(seen_p12_id, Path::new("data/private/bmrb"), true)
            {
                assert!(
                    loc.truth_path.as_ref().is_some_and(|p| p.is_file()),
                    "seen_p12 entry {seen_p12_id} must have NMR-STAR file present"
                );
                let ws = nmr_core::classify::classify_nmr_workspace(&loc.entry_dir).unwrap();
                assert!(
                    ws.problem_regime == nmr_core::classify::ProblemRegime::GlobularBackbone
                        || ws
                            .secondary_regimes
                            .contains(&nmr_core::classify::ProblemRegime::GlobularBackbone),
                    "seen_p12 entry {seen_p12_id} must support GlobularBackbone (got primary {:?}, secondary {:?})",
                    ws.problem_regime,
                    ws.secondary_regimes
                );
                let (disc, _) =
                    discover_entry_experiments(&loc.entry_dir, Some(e.field_mhz)).unwrap();
                let disc_exps: BTreeSet<&str> = disc.iter().map(|d| d.exp_type.as_str()).collect();
                println!("  [seen_p12 header-only] {seen_p12_id} -> discovered={disc_exps:?}");
                for req_exp in &e.experiments {
                    assert!(
                        disc_exps.contains(req_exp.as_str()),
                        "seen_p12 entry {seen_p12_id} missing manifest experiment {req_exp}; discovered {disc_exps:?}"
                    );
                }
            }
        }

        // Verify zero overlap with Phase-1 `data/benchmark/manifest.tsv`
        let p1_raw = fs::read_to_string(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/benchmark/manifest.tsv"),
        )
        .unwrap();
        for line in p1_raw.lines().skip(1) {
            if let Some(col0) = line.split('\t').next() {
                let id_num = col0.trim().trim_start_matches("bmr");
                assert!(
                    !entries.iter().any(|e| e.bmrb_id == id_num),
                    "real_manifest entry {id_num} overlaps Phase-1 manifest.tsv"
                );
            }
        }

        // Verify zero overlap with empirical prior training set in `data/priors/PROVENANCE.md`
        let prov_raw = fs::read_to_string(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/priors/PROVENANCE.md"),
        )
        .unwrap();
        for tok in prov_raw
            .split(|c: char| !c.is_ascii_alphanumeric())
            .filter(|t| t.starts_with("bmr") && t.len() > 3)
        {
            let id_num = tok.trim_start_matches("bmr");
            if id_num.chars().all(|c| c.is_ascii_digit()) {
                assert!(
                    !entries.iter().any(|e| e.bmrb_id == id_num),
                    "real_manifest entry {id_num} overlaps empirical prior training set in data/priors/PROVENANCE.md"
                );
            }
        }

        // Verify `data/benchmark/PROVENANCE.md` pre-registration record embeds the exact
        // FIPS 180-4 SHA-256 digest of `data/benchmark/real_manifest.tsv` (Q1 governance).
        let (manifest_sha256, _) = crate::fetch_bmrb::sha256_file_and_size(&manifest_path).unwrap();
        let bench_prov_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/benchmark/PROVENANCE.md");
        let bench_prov_raw = fs::read_to_string(&bench_prov_path)
            .expect("data/benchmark/PROVENANCE.md must exist with pre-registered manifest SHA-256");
        assert!(
            bench_prov_raw.contains(&manifest_sha256),
            "data/benchmark/PROVENANCE.md does not contain current real_manifest.tsv SHA-256 ({manifest_sha256})"
        );

        // Verify P1-6 default solver is `evo-graph` while `--solver standard` remains selectable
        let cfg_default =
            parse_real_benchmark_args(&["--split".to_string(), "dev".to_string()]).unwrap();
        assert_eq!(cfg_default.solver, SolverKind::EvoGraph);
        let cfg_std = parse_real_benchmark_args(&[
            "--split".to_string(),
            "dev".to_string(),
            "--solver".to_string(),
            "standard".to_string(),
        ])
        .unwrap();
        assert_eq!(cfg_std.solver, SolverKind::Standard);
    }

    #[test]
    fn test_calibration_summary_distinguishes_abstain_from_zero_percent_hc_and_detailed_offsets() {
        use nmr_core::qc::DatasetQcVerdict;

        let mk_entry = |id: &str, name: &str| RealManifestEntry {
            bmrb_id: id.to_string(),
            protein: name.to_string(),
            residues: 70,
            field_mhz: 600.0,
            vendor: "Bruker".to_string(),
            experiments: vec!["HSQC".into(), "HNCO".into(), "HNCACB".into()],
            split: "dev".to_string(),
            cohort: "dev".to_string(),
        };

        let det_mixed = DetailedReferenceOffsets {
            h: NucleusOffsetFit {
                status: OffsetFitStatus::NotSignificant,
                applied_ppm: 0.0,
                candidate_ppm: 0.01,
                log_likelihood_gain: 0.4,
                n_values: 60,
            },
            n: NucleusOffsetFit {
                status: OffsetFitStatus::NotSignificant,
                applied_ppm: 0.0,
                candidate_ppm: -0.05,
                log_likelihood_gain: 0.6,
                n_values: 60,
            },
            ca: NucleusOffsetFit {
                status: OffsetFitStatus::Fitted,
                applied_ppm: -2.75,
                candidate_ppm: -2.75,
                log_likelihood_gain: 42.5,
                n_values: 58,
            },
            cb: NucleusOffsetFit {
                status: OffsetFitStatus::TooFewValues,
                applied_ppm: 0.0,
                candidate_ppm: 0.0,
                log_likelihood_gain: 0.0,
                n_values: 0,
            },
            c: NucleusOffsetFit {
                status: OffsetFitStatus::NotSignificant,
                applied_ppm: 0.0,
                candidate_ppm: 0.08,
                log_likelihood_gain: 1.1,
                n_values: 55,
            },
        };

        // 1. Clean high-accuracy protein (`PASS`, 19/20 = 95.0% HC precision)
        let mut preds_good: Vec<(f64, bool)> = (0..19).map(|_| (0.95, true)).collect();
        preds_good.push((0.85, false));
        let rel_good = ProteinReliabilityCurve::from_predictions(&preds_good, 0.8);
        let rec_good = EntryBenchmarkRecord {
            manifest_entry: mk_entry("11111", "CleanProt"),
            outcome: EntryBenchmarkOutcome::Succeeded(Box::new(EntrySuccessMetrics {
                seeds: 1,
                discovered_experiments: vec![("HSQC".into(), "1".into())],
                n_hsqc_roots: 62,
                n_total_peaks: 310,
                assignable_per_seed: 20,
                total_assignable: 20,
                total_assigned: 20,
                total_correct: 19,
                total_hc_assigned: 20,
                total_hc_correct: 19,
                mean_accuracy: 0.95,
                mean_recall: 0.95,
                mean_hc_precision: 0.95,
                mean_hc_recall: 0.95,
                global_offsets: det_mixed.to_shift_offsets(),
                detailed_offsets: det_mixed,
                prior_detailed_offsets: det_mixed,
                qc_report: DatasetQcReport {
                    verdict: DatasetQcVerdict::Pass,
                    detailed_offsets: det_mixed,
                    ..DatasetQcReport::default()
                },
                reliability: rel_good,
                predictions: preds_good,
                elapsed_secs: 0.5,
            })),
        };

        // 2. Abstained dataset (`ABSTAIN`, all confidences capped <= 0.45 -> 0 HC calls)
        let preds_abstain: Vec<(f64, bool)> = (0..10).map(|i| (0.35, i < 2)).collect();
        let rel_abstain = ProteinReliabilityCurve::from_predictions(&preds_abstain, 0.8);
        assert!(rel_abstain.is_abstained());
        let rec_abstain = EntryBenchmarkRecord {
            manifest_entry: mk_entry("22222", "AbstainProt"),
            outcome: EntryBenchmarkOutcome::Succeeded(Box::new(EntrySuccessMetrics {
                seeds: 1,
                discovered_experiments: vec![("HSQC".into(), "1".into())],
                n_hsqc_roots: 60,
                n_total_peaks: 80,
                assignable_per_seed: 20,
                total_assignable: 20,
                total_assigned: 10,
                total_correct: 2,
                total_hc_assigned: 0,
                total_hc_correct: 0,
                mean_accuracy: 0.20,
                mean_recall: 0.10,
                mean_hc_precision: 0.0,
                mean_hc_recall: 0.0,
                global_offsets: ShiftOffsets::default(),
                detailed_offsets: DetailedReferenceOffsets::default(),
                prior_detailed_offsets: DetailedReferenceOffsets::default(),
                qc_report: DatasetQcReport {
                    verdict: DatasetQcVerdict::Abstain,
                    confidence_scale: 0.0,
                    confidence_cap: 0.45,
                    reasons: vec!["catastrophic 3D strip support (12.0% < 25.0%)".into()],
                    ..DatasetQcReport::default()
                },
                reliability: rel_abstain,
                predictions: preds_abstain,
                elapsed_secs: 0.3,
            })),
        };

        // 3. Miscalibrated dataset with wrong HC calls (`WARN`, 0/5 = 0.0% HC precision, NOT abstained!)
        let preds_wrong_hc: Vec<(f64, bool)> = (0..5).map(|_| (0.86, false)).collect();
        let rel_wrong_hc = ProteinReliabilityCurve::from_predictions(&preds_wrong_hc, 0.8);
        assert!(!rel_wrong_hc.is_abstained());
        assert_eq!(rel_wrong_hc.hc_precision, Some(0.0));
        let rec_wrong_hc = EntryBenchmarkRecord {
            manifest_entry: mk_entry("33333", "WrongHcProt"),
            outcome: EntryBenchmarkOutcome::Succeeded(Box::new(EntrySuccessMetrics {
                seeds: 1,
                discovered_experiments: vec![("HSQC".into(), "1".into())],
                n_hsqc_roots: 60,
                n_total_peaks: 200,
                assignable_per_seed: 10,
                total_assignable: 10,
                total_assigned: 5,
                total_correct: 0,
                total_hc_assigned: 5,
                total_hc_correct: 0,
                mean_accuracy: 0.0,
                mean_recall: 0.0,
                mean_hc_precision: 0.0,
                mean_hc_recall: 0.0,
                global_offsets: ShiftOffsets::default(),
                detailed_offsets: DetailedReferenceOffsets::default(),
                prior_detailed_offsets: DetailedReferenceOffsets::default(),
                qc_report: DatasetQcReport {
                    verdict: DatasetQcVerdict::Warn,
                    confidence_scale: 0.78,
                    confidence_cap: 1.0,
                    reasons: vec!["elevated HSQC root multiplicity (1.60x)".into()],
                    ..DatasetQcReport::default()
                },
                reliability: rel_wrong_hc,
                predictions: preds_wrong_hc,
                elapsed_secs: 0.4,
            })),
        };

        let summary = summarize_real_benchmark_records(
            "dev",
            "CPU (f64, evo-graph)".to_string(),
            1,
            0.8,
            vec![rec_good, rec_abstain, rec_wrong_hc],
        );

        assert_eq!(summary.n_hc_active_proteins, 2);
        assert_eq!(summary.n_hc_abstained_proteins, 1);
        assert!(
            (summary.macro_hc_precision - 0.475).abs() < 1e-9,
            "macro HC precision must average over the 2 HC-active proteins (0.95 and 0.0 -> 0.475), got {}",
            summary.macro_hc_precision
        );
        assert_eq!(
            summary.worst_protein_hc_precision,
            Some(("33333".to_string(), "WrongHcProt".to_string(), 0.0))
        );
        assert!(
            (summary.median_protein_hc_precision.unwrap() - 0.475).abs() < 1e-9,
            "got {:?}",
            summary.median_protein_hc_precision
        );

        let table = format_real_benchmark_report(&summary);
        assert!(table.contains("PASS") && table.contains("ABSTAIN") && table.contains("WARN"));
        assert!(table.contains("dCA=-2.75(ΔLL=42.5)"), "table:\n{table}");
        assert!(table.contains("dCB=n<5(0)"), "table:\n{table}");
        assert!(table.contains("dC=ns(ΔLL=1.1)"), "table:\n{table}");
        assert!(
            table.contains("Abstained (0 HC calls)"),
            "expected explicit abstained status in table:\n{table}"
        );
        assert!(
            table.contains("0.0% (0/5)"),
            "expected explicit 0.0% (0/5) wrong-HC status in table:\n{table}"
        );
        assert!(
            table.contains("[0.0-0.5):")
                && table.contains("[0.5-0.8):")
                && table.contains("[0.8-0.9):")
                && table.contains("[0.9-1.0]:"),
            "expected 4-bin reliability curves in table:\n{table}"
        );

        let json = render_summary_json(&summary);
        assert!(json.contains("\"hc_active_proteins\": 2"));
        assert!(json.contains("\"hc_abstained_proteins\": 1"));
        assert!(json.contains("\"hc_abstained\": true,\n      \"hc_precision\": null"));
        assert!(json.contains("\"hc_abstained\": false,\n      \"hc_precision\": 0.000000"));
        assert!(json.contains("\"status\": \"fitted\""));
        assert!(json.contains("\"status\": \"not_significant\""));
        assert!(json.contains("\"status\": \"too_few_values\""));
        assert!(json.contains("\"calibration_summary\":"));
    }

    #[test]
    #[ignore]
    fn dev_split_lopo_calibration_and_minor_conformer_audit() {
        use nmr_core::evo_graph::{assign_evo_graph_from_peaks_with_minor_damping, EvoGraphConfig};
        use nmr_core::priors::sync_detailed_reference_offsets;
        use nmr_core::qc::{
            evaluate_seed_predictions_with_gate, fit_isotonic_monotone_knots,
            fit_isotonic_monotone_knots_beta, fit_isotonic_monotone_knots_shrunk_toward,
            fit_protein_hc_gate_config, has_weak_likelihood_large_offset,
            is_unconverged_solution_margin, select_gpu_shrinkage_weight_nested_lopo,
            DatasetQcVerdict, ProteinHcTrainingSeed, ProteinReliabilityCurve,
            CANONICAL_CALIBRATION_GRID_12, CANONICAL_CALIBRATION_LAPLACE_PRIOR,
            CANONICAL_GPU_CPU_SHRINKAGE_WEIGHT, CANONICAL_PROTEIN_HC_GATE,
            QC_STRONG_BELIEF_THRESHOLD, STANDARD_CPU_PRIOR_ALPHA, STANDARD_CPU_PRIOR_BETA,
        };
        use nmr_core::solver::{assign_uncalibrated, prepare_scoring_priors, AnnealConfig};
        use nmr_gpu::GpuEngine;

        let gpu_engine = GpuEngine::new_headless()
            .expect("WebGPU headless adapter required for STANDARD_GPU LOPO calibration");
        println!(
            "  [GPU ADAPTER] name={:?}, backend={:?}",
            gpu_engine.adapter_name, gpu_engine.backend_name
        );

        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        env::set_current_dir(&repo_root).expect("chdir repo_root");
        let manifest_path = PathBuf::from("data/benchmark/real_manifest.tsv");
        let manifest_txt = fs::read_to_string(&manifest_path).expect("read real_manifest.tsv");
        let entries: Vec<RealManifestEntry> = parse_real_manifest_str(&manifest_txt)
            .expect("parse real_manifest.tsv")
            .into_iter()
            .filter(|e| e.split == "dev")
            .collect();
        assert_eq!(entries.len(), 13, "expected exact 13 dev entries");
        for e in &entries {
            assert_eq!(e.split, "dev", "STRICT HELDOUT FIREWALL");
        }

        let priors = load_priors(None).expect("load priors");
        let score_cfg = ScoreConfig::default();
        let picker_cfg = PickerConfig::default();
        let data_root = PathBuf::from("data/private/bmrb");

        let make_training_seed =
            |qc: &DatasetQcReport, raw_predictions: Vec<(f64, bool)>| -> ProteinHcTrainingSeed {
                ProteinHcTrainingSeed {
                    n_assignable_amides: qc.n_assignable_amides,
                    hsqc_3d_support_frac: qc.hsqc_3d_support_frac,
                    supported_roots_per_assignable: qc.supported_roots_per_assignable,
                    aliphatic_support_frac: qc.aliphatic_support_frac,
                    hsqc_multiplicity_ratio: qc.hsqc_multiplicity_ratio,
                    hsqc_3d_rms_dh_ppm: qc.hsqc_3d_rms_dh_ppm,
                    hsqc_3d_rms_dn_ppm: qc.hsqc_3d_rms_dn_ppm,
                    offset_consistency_ok: qc.offset_consistency_ok,
                    weak_large_offset: has_weak_likelihood_large_offset(
                        &qc.detailed_offsets,
                        CANONICAL_PROTEIN_HC_GATE.min_large_offset_gain_per_obs,
                    )
                    .is_some(),
                    connectivity_frac: qc.connectivity_frac,
                    corroborated_connectivity_frac: qc.corroborated_connectivity_frac,
                    raw_predictions,
                }
            };

        // Fixed dataset convergence specification (`(0.65, 0.07, 1.60, 0.16)`) identical to
        // `nmr_core::qc::is_unconverged_solution_margin` / `evaluate_dataset_qc`: excludes
        // datasets whose solver ensemble failed to converge (`Abstain`) from the PAV training pool.
        let is_trainable_prot = |seeds: &[ProteinHcTrainingSeed]| -> bool {
            let tot_n: usize = seeds.iter().map(|s| s.raw_predictions.len()).sum();
            if tot_n == 0 {
                return false;
            }
            let s0 = &seeds[0];
            let strong: usize = seeds
                .iter()
                .map(|s| {
                    s.raw_predictions
                        .iter()
                        .filter(|&&(p, _)| p >= QC_STRONG_BELIEF_THRESHOLD)
                        .count()
                })
                .sum();
            let m = (strong as f64) / (tot_n as f64);
            let n_per_seed = tot_n / seeds.len().max(1);
            !(s0.hsqc_3d_support_frac < 0.25
                || s0.aliphatic_support_frac < 0.20
                || (n_per_seed >= 6 && s0.connectivity_frac < 0.15)
                || is_unconverged_solution_margin(tot_n, m, s0.hsqc_multiplicity_ratio)
                || ((s0.hsqc_3d_rms_dh_ppm > 0.065 || s0.hsqc_3d_rms_dn_ppm > 0.65)
                    && s0.hsqc_3d_support_frac < 0.45)
                || (s0.hsqc_3d_support_frac < 0.45
                    && s0.supported_roots_per_assignable < 0.35
                    && m < 0.20))
        };

        type ProtRecord = (String, String, DatasetQcVerdict, Vec<ProteinHcTrainingSeed>);
        let mut evo_by_prot: Vec<ProtRecord> = Vec::new();
        let mut std_by_prot: Vec<ProtRecord> = Vec::new();
        let mut std_gpu_by_prot: Vec<ProtRecord> = Vec::new();

        // P1-5 minor-conformer audit accumulators across all 10 dev proteins
        let mut tot_true_roots_1seed = 0usize;
        let mut tot_weak_true_roots_1seed = 0usize;
        let mut tot_weak_ht_true_roots_1seed = 0usize;
        let mut weak_with_assigned = 0usize;
        let mut weak_with_correct = 0usize;
        let mut weak_without_assigned = 0usize;
        let mut weak_without_correct = 0usize;
        let mut weak_total_5seed = 0usize;
        let mut all_with_assigned = 0usize;
        let mut all_with_correct = 0usize;
        let mut all_without_assigned = 0usize;
        let mut all_without_correct = 0usize;
        let mut per_prot_audit_lines: Vec<String> = Vec::new();

        let only_filter = env::var("ONLY_ENTRY").ok();
        for entry in &entries {
            if let Some(ref f) = only_filter {
                if !f.split(',').any(|tok| tok.trim() == entry.bmrb_id) {
                    continue;
                }
            }
            let loc = resolve_local_entry_paths(&entry.bmrb_id, &data_root, true)
                .unwrap_or_else(|p| panic!("missing dev entry {}: {p:?}", entry.bmrb_id));
            let truth_path = loc.truth_path.expect("truth STAR path");
            let (truth_ref, _, _) = parse_truth_auto(&truth_path).expect("parse truth STAR");
            let (discovered, _) = discover_entry_experiments(&loc.entry_dir, Some(entry.field_mhz))
                .expect("discover");
            let hsqc_exp = discovered
                .iter()
                .find(|d| d.exp_type == "HSQC")
                .expect("HSQC");
            let triple_exps: Vec<&DiscoveredFidExperiment> =
                discovered.iter().filter(|d| d.exp_type != "HSQC").collect();

            let (hsqc_spec, _) =
                process_fid_dir_2d(&hsqc_exp.dir, false, false).expect("load 2D HSQC");
            let roots = pick_hsqc_2d(&hsqc_spec, &picker_cfg);
            let mut all_peaks: Vec<Peak> = roots.clone();
            let mut used_exps = vec![("HSQC".to_string(), hsqc_exp.rel_path.clone())];
            let mut strip_bank = StripBank3D::new();
            for texp in &triple_exps {
                if let Ok((mut spec3d, _)) =
                    process_fid_dir_3d(&texp.dir, &texp.exp_type, false, false)
                {
                    let _ = register_3d_hn_to_hsqc_adaptive(&mut spec3d, &hsqc_spec);
                    let pks = strip_bank.pick_and_record_3d(&spec3d, &roots, &picker_cfg);
                    all_peaks.extend(pks);
                    used_exps.push((texp.exp_type.clone(), texp.rel_path.clone()));
                }
            }
            if used_exps.len() >= 3 && !roots.is_empty() {
                all_peaks.extend(corroborated_re_pick_3d(
                    &strip_bank,
                    &roots,
                    &all_peaks,
                    &picker_cfg,
                    &CorroboratedRepickConfig::default(),
                ));
            }
            let all_systems_for_eval =
                build_spin_systems(&all_peaks, &Tolerances::default(), 3, false)
                    .expect("build_spin_systems");

            let mut evo_samples = Vec::new();
            let mut std_samples = Vec::new();
            let mut std_gpu_samples = Vec::new();
            let mut evo_verdict = DatasetQcVerdict::Pass;
            let mut std_verdict = DatasetQcVerdict::Pass;
            let mut std_gpu_verdict = DatasetQcVerdict::Pass;
            let mut evo_qc_reasons = String::new();
            let mut std_qc_reasons = String::new();

            let mut p_weak_1seed = 0usize;
            let mut p_weak_ht_1seed = 0usize;
            let mut p_true_1seed = 0usize;
            let mut p_weak_w_ass = 0usize;
            let mut p_weak_w_cor = 0usize;
            let mut p_weak_wo_ass = 0usize;
            let mut p_weak_wo_cor = 0usize;

            // Precompute standard spin systems & offsets once per protein
            let std_base_det = nmr_core::priors::estimate_reference_offsets_detailed(
                &all_systems_for_eval,
                &truth_ref.sequence,
                &priors,
            );
            let (std_eff_priors, std_offs) = prepare_scoring_priors(
                &all_systems_for_eval,
                &truth_ref.sequence,
                &priors,
                &score_cfg,
            );
            let std_synced_det = sync_detailed_reference_offsets(&std_offs, &std_base_det);

            for seed in 0..5u64 {
                // 1. Evo-graph WITH minor-conformer damping (`apply_qc_calibration = false`;
                // zero empirical knot tables exist upstream of Pass 2!)
                let evo_cfg_raw = EvoGraphConfig {
                    score_cfg: score_cfg.clone(),
                    auto_reference: true,
                    apply_qc_calibration: false,
                    seed,
                    ..EvoGraphConfig::default()
                };
                let (evo_sys, evo_res_raw, evo_rep) =
                    assign_evo_graph_from_peaks_with_minor_damping(
                        &all_peaks,
                        &truth_ref.sequence,
                        &priors,
                        &evo_cfg_raw,
                        true,
                    );
                let qc_evo = &evo_rep.qc_report;
                if seed == 0 {
                    evo_verdict = qc_evo.verdict;
                    evo_qc_reasons = qc_evo.reasons.join("; ");
                }
                let eff_evo_priors = evo_rep.shifted_priors(&priors);
                let (pred_evo, conf_k_evo) = to_reference(
                    &evo_res_raw,
                    &evo_sys,
                    &truth_ref.sequence,
                    &truth_ref.seq_ids,
                    0.0,
                    Some(&eff_evo_priors),
                );
                let mut conf_seq_evo = BTreeMap::new();
                for (k, c) in conf_k_evo {
                    conf_seq_evo.insert(truth_ref.seq_ids[k], c);
                }
                let eval_evo = evaluate_prediction_against_truth(
                    &pred_evo,
                    &truth_ref,
                    &conf_seq_evo,
                    Some(&all_systems_for_eval),
                    0.80,
                    true,
                );
                evo_samples.push(make_training_seed(
                    qc_evo,
                    eval_evo.placed_predictions.clone(),
                ));
                all_with_assigned += eval_evo.assigned;
                all_with_correct += eval_evo.correct;

                // Map `evo_sys` 1-to-1 to `eval_evo.truth_for_cmp` (`tol_h = 0.05, tol_n = 0.50`)
                let mut true_roots = Vec::new();
                for k in 1..eval_evo.truth_for_cmp.sequence.len() {
                    if eval_evo.truth_for_cmp.sequence[k] == "PRO" {
                        continue;
                    }
                    if let (Some(&th), Some(&tn)) = (
                        eval_evo.truth_for_cmp.shifts.get(&(k, "H".to_string())),
                        eval_evo.truth_for_cmp.shifts.get(&(k, "N".to_string())),
                    ) {
                        true_roots.push((k, th, tn));
                    }
                }
                let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
                for (si, ss) in evo_sys.iter().enumerate() {
                    for (ti, &(_, th, tn)) in true_roots.iter().enumerate() {
                        let dh = (ss.h - th).abs();
                        let dn = (ss.n - tn).abs();
                        if dh <= 0.05 && dn <= 0.50 {
                            pairs.push(((dh / 0.05).powi(2) + (dn / 0.50).powi(2), si, ti));
                        }
                    }
                }
                pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
                let mut used_s = vec![false; evo_sys.len()];
                let mut used_t = vec![false; true_roots.len()];
                let mut evo_ss_truth = vec![-1i32; evo_sys.len()];
                for &(_, si, ti) in &pairs {
                    if !used_s[si] && !used_t[ti] {
                        used_s[si] = true;
                        used_t[ti] = true;
                        evo_ss_truth[si] = true_roots[ti].0 as i32;
                    }
                }

                // 2. Evo-graph WITHOUT minor-conformer damping (`enable_minor_damping = false`)
                let (_evo_sys_no, evo_res_no, evo_rep_no) =
                    assign_evo_graph_from_peaks_with_minor_damping(
                        &all_peaks,
                        &truth_ref.sequence,
                        &priors,
                        &evo_cfg_raw,
                        false,
                    );
                let eff_no_priors = evo_rep_no.shifted_priors(&priors);
                let (pred_no, conf_k_no) = to_reference(
                    &evo_res_no,
                    &evo_sys,
                    &truth_ref.sequence,
                    &truth_ref.seq_ids,
                    0.0,
                    Some(&eff_no_priors),
                );
                let mut conf_seq_no = BTreeMap::new();
                for (k, c) in conf_k_no {
                    conf_seq_no.insert(truth_ref.seq_ids[k], c);
                }
                let eval_no = evaluate_prediction_against_truth(
                    &pred_no,
                    &truth_ref,
                    &conf_seq_no,
                    Some(&all_systems_for_eval),
                    0.80,
                    true,
                );
                all_without_assigned += eval_no.assigned;
                all_without_correct += eval_no.correct;

                for (si, ss) in evo_sys.iter().enumerate() {
                    let tk = evo_ss_truth[si];
                    if tk < 0 {
                        continue;
                    }
                    if seed == 0 {
                        p_true_1seed += 1;
                        if ss.rel_root_height.is_some_and(|rh| rh < 0.15) {
                            p_weak_ht_1seed += 1;
                        }
                    }
                    if ss
                        .rel_root_intensity
                        .or(ss.rel_root_height)
                        .is_some_and(|r| r < 0.15)
                    {
                        if seed == 0 {
                            p_weak_1seed += 1;
                        }
                        weak_total_5seed += 1;
                        if evo_res_raw.position[si] >= 0 {
                            p_weak_w_ass += 1;
                            if evo_res_raw.position[si] == tk {
                                p_weak_w_cor += 1;
                            }
                        }
                        if evo_res_no.position[si] >= 0 {
                            p_weak_wo_ass += 1;
                            if evo_res_no.position[si] == tk {
                                p_weak_wo_cor += 1;
                            }
                        }
                    }
                }

                // 3. Standard CPU (`runs = 16`) raw pre-QC confidences
                let anneal_cfg = AnnealConfig {
                    runs: 16,
                    steps_per_system: 1500,
                    seed,
                    ..AnnealConfig::default()
                };
                let std_res_raw = assign_uncalibrated(
                    &all_systems_for_eval,
                    &truth_ref.sequence,
                    &priors,
                    &score_cfg,
                    &anneal_cfg,
                );
                let qc_std = nmr_core::qc::evaluate_dataset_qc(
                    Some(&all_peaks),
                    &all_systems_for_eval,
                    &truth_ref.sequence,
                    &priors,
                    &score_cfg,
                    &std_res_raw,
                    Some(&std_synced_det),
                );
                if seed == 0 {
                    std_verdict = qc_std.verdict;
                    std_qc_reasons = qc_std.reasons.join("; ");
                }
                let (pred_std, conf_k_std) = to_reference(
                    &std_res_raw,
                    &all_systems_for_eval,
                    &truth_ref.sequence,
                    &truth_ref.seq_ids,
                    0.0,
                    Some(&std_eff_priors),
                );
                let mut conf_seq_std = BTreeMap::new();
                for (k, c) in conf_k_std {
                    conf_seq_std.insert(truth_ref.seq_ids[k], c);
                }
                let eval_std = evaluate_prediction_against_truth(
                    &pred_std,
                    &truth_ref,
                    &conf_seq_std,
                    Some(&all_systems_for_eval),
                    0.80,
                    true,
                );
                std_samples.push(make_training_seed(
                    &qc_std,
                    eval_std.placed_predictions.clone(),
                ));
                if only_filter.is_some() && seed == 0 {
                    let n_hi_ok = eval_std
                        .placed_predictions
                        .iter()
                        .filter(|&&(p, ok)| p >= 0.90 && ok)
                        .count();
                    let n_hi_bad = eval_std
                        .placed_predictions
                        .iter()
                        .filter(|&&(p, ok)| p >= 0.90 && !ok)
                        .count();
                    println!(
                        "    [SEED0 {}] acc={}/{} ({:.1}%), eval_offs={}, qc={}, conn={:.3}, margin={:.3}, p>=0.90 ok={} bad={}",
                        entry.bmrb_id,
                        eval_std.correct,
                        eval_std.assigned,
                        100.0 * (eval_std.correct as f64) / (eval_std.assigned.max(1) as f64),
                        eval_std.global_offsets,
                        qc_std.verdict.as_str(),
                        qc_std.connectivity_frac,
                        qc_std.solution_margin,
                        n_hi_ok,
                        n_hi_bad,
                    );
                }
                // 4. Standard WebGPU (`runs = 64`, matching `execute_assignment_workflow` on GPU)
                let gpu_anneal_cfg = AnnealConfig {
                    runs: 64,
                    steps_per_system: 1500,
                    seed,
                    ..AnnealConfig::default()
                };
                let std_gpu_res_raw = gpu_engine
                    .assign_gpu_uncalibrated(
                        &all_systems_for_eval,
                        &truth_ref.sequence,
                        &priors,
                        &score_cfg,
                        &gpu_anneal_cfg,
                    )
                    .expect("assign_gpu_uncalibrated");
                let qc_std_gpu = nmr_core::qc::evaluate_dataset_qc(
                    Some(&all_peaks),
                    &all_systems_for_eval,
                    &truth_ref.sequence,
                    &priors,
                    &score_cfg,
                    &std_gpu_res_raw,
                    Some(&std_synced_det),
                );
                if seed == 0 {
                    std_gpu_verdict = qc_std_gpu.verdict;
                }
                let (pred_std_gpu, conf_k_std_gpu) = to_reference(
                    &std_gpu_res_raw,
                    &all_systems_for_eval,
                    &truth_ref.sequence,
                    &truth_ref.seq_ids,
                    0.0,
                    Some(&std_eff_priors),
                );
                let mut conf_seq_std_gpu = BTreeMap::new();
                for (k, c) in conf_k_std_gpu {
                    conf_seq_std_gpu.insert(truth_ref.seq_ids[k], c);
                }
                let eval_std_gpu = evaluate_prediction_against_truth(
                    &pred_std_gpu,
                    &truth_ref,
                    &conf_seq_std_gpu,
                    Some(&all_systems_for_eval),
                    0.80,
                    true,
                );
                std_gpu_samples.push(make_training_seed(
                    &qc_std_gpu,
                    eval_std_gpu.placed_predictions.clone(),
                ));
            }

            tot_true_roots_1seed += p_true_1seed;
            tot_weak_true_roots_1seed += p_weak_1seed;
            tot_weak_ht_true_roots_1seed += p_weak_ht_1seed;
            weak_with_assigned += p_weak_w_ass;
            weak_with_correct += p_weak_w_cor;
            weak_without_assigned += p_weak_wo_ass;
            weak_without_correct += p_weak_wo_cor;

            per_prot_audit_lines.push(format!(
                "  {:16} {:14} | QC(evo={}, std={}) | true_roots={} | r_vol<0.15={} (r_ht<0.15={}) | 5-seed weak correct: WITH={}/{} (ass={}), WITHOUT={}/{} (ass={}) | evo_notes=[{}] std_notes=[{}]",
                entry.bmrb_id,
                entry.protein,
                evo_verdict.as_str(),
                std_verdict.as_str(),
                p_true_1seed,
                p_weak_1seed,
                p_weak_ht_1seed,
                p_weak_w_cor,
                p_weak_1seed * 5,
                p_weak_w_ass,
                p_weak_wo_cor,
                p_weak_1seed * 5,
                p_weak_wo_ass,
                evo_qc_reasons,
                std_qc_reasons,
            ));

            evo_by_prot.push((
                entry.bmrb_id.clone(),
                entry.protein.clone(),
                evo_verdict,
                evo_samples,
            ));
            std_by_prot.push((
                entry.bmrb_id.clone(),
                entry.protein.clone(),
                std_verdict,
                std_samples,
            ));
            std_gpu_by_prot.push((
                entry.bmrb_id.clone(),
                entry.protein.clone(),
                std_gpu_verdict,
                std_gpu_samples,
            ));
        }

        println!("\n================ P1-5 MINOR-CONFORMER (< 0.15) 13-PROTEIN DEV AUDIT ================");
        for line in &per_prot_audit_lines {
            println!("{line}");
        }
        println!(
            "  TOTAL (13 dev proteins): true_roots={} | r_vol<0.15={} ({:.2}%) [vs r_height<0.15={} ({:.2}%)]",
            tot_true_roots_1seed,
            tot_weak_true_roots_1seed,
            100.0 * (tot_weak_true_roots_1seed as f64) / (tot_true_roots_1seed.max(1) as f64),
            tot_weak_ht_true_roots_1seed,
            100.0 * (tot_weak_ht_true_roots_1seed as f64) / (tot_true_roots_1seed.max(1) as f64),
        );
        println!(
            "  r_vol<0.15 5-seed pool (N={}): WITH damping acc={:.2}% ({}/{}), recall={:.2}% ({}/{}) | WITHOUT damping acc={:.2}% ({}/{}), recall={:.2}% ({}/{})",
            weak_total_5seed,
            100.0 * (weak_with_correct as f64) / (weak_with_assigned.max(1) as f64),
            weak_with_correct,
            weak_with_assigned,
            100.0 * (weak_with_correct as f64) / (weak_total_5seed.max(1) as f64),
            weak_with_correct,
            weak_total_5seed,
            100.0 * (weak_without_correct as f64) / (weak_without_assigned.max(1) as f64),
            weak_without_correct,
            weak_without_assigned,
            100.0 * (weak_without_correct as f64) / (weak_total_5seed.max(1) as f64),
            weak_without_correct,
            weak_total_5seed,
        );
        println!(
            "  Overall 13-dev micro accuracy: WITH damping={:.2}% ({}/{}) | WITHOUT damping={:.2}% ({}/{})",
            100.0 * (all_with_correct as f64) / (all_with_assigned.max(1) as f64),
            all_with_correct,
            all_with_assigned,
            100.0 * (all_without_correct as f64) / (all_without_assigned.max(1) as f64),
            all_without_correct,
            all_without_assigned,
        );

        struct LopoGateCheck {
            solver_name: String,
            oof_mean_ece: f64,
            oof_mean_ece_10: f64,
            oof_mean_debiased_ece_10: f64,
            oof_pool_ece_10: f64,
            oof_micro_hcp: f64,
            oof_macro_hcp: f64,
            full_knots: Vec<(f64, f64)>,
        }
        let mut gate_checks: Vec<LopoGateCheck> = Vec::new();
        let identity_knots = [(0.0, 0.0), (1.0, 1.0)];

        // Strict 13-fold Leave-One-Protein-Out (LOPO) cross-validation (`P1-7` & `P1-8`):
        // inside every outer fold `hold_idx`, BOTH the isotonic PAV knots AND the protein-level
        // HC gate (`fit_protein_hc_gate_config`) are fitted strictly on the 12 training proteins
        // `j != hold_idx`, with zero target leakage from `by_prot[hold_idx]`.
        let mut run_lopo_calibration = |solver_name: &str, by_prot: &[ProtRecord]| {
            let canonical_grid = CANONICAL_CALIBRATION_GRID_12;
            let canonical_laplace = CANONICAL_CALIBRATION_LAPLACE_PRIOR;
            let fit_knots_for_solver = |pairs: &[(f64, bool)]| -> Vec<(f64, f64)> {
                if solver_name == "STANDARD_CPU" {
                    fit_isotonic_monotone_knots_beta(
                        pairs,
                        canonical_grid,
                        STANDARD_CPU_PRIOR_ALPHA,
                        STANDARD_CPU_PRIOR_BETA,
                    )
                } else {
                    fit_isotonic_monotone_knots(pairs, canonical_grid, canonical_laplace)
                }
            };

            let collect_train_pairs = |skip_idx: Option<usize>| -> Vec<(f64, bool)> {
                let mut pairs = Vec::new();
                for (idx, (_, _, _, seeds)) in by_prot.iter().enumerate() {
                    if skip_idx == Some(idx) {
                        continue;
                    }
                    if is_trainable_prot(seeds) {
                        for s in seeds {
                            pairs.extend_from_slice(&s.raw_predictions);
                        }
                    }
                }
                pairs
            };

            println!(
                "\n================ CANONICAL 13-FOLD LOPO CALIBRATION: {solver_name} ================"
            );
            // 1. Report UNCALIBRATED (identity map) baseline
            {
                let mut prot_eces = Vec::new();
                let mut prot_eces_10 = Vec::new();
                let mut prot_eces_12 = Vec::new();
                let mut prot_hcps = Vec::new();
                let mut pooled = Vec::new();
                for (_, _, _, seeds) in by_prot {
                    let mut preds = Vec::new();
                    for s in seeds {
                        preds.extend(evaluate_seed_predictions_with_gate(
                            s,
                            &identity_knots,
                            &CANONICAL_PROTEIN_HC_GATE,
                        ));
                    }
                    let rel =
                        ProteinReliabilityCurve::from_predictions_with_repeats(&preds, 0.80, 5.0);
                    prot_eces.push(rel.ece);
                    prot_eces_10.push(rel.ece_10);
                    prot_eces_12.push(rel.ece_12);
                    if let Some(hcp) = rel.hc_precision {
                        prot_hcps.push(hcp);
                    }
                    pooled.extend(preds);
                }
                let pool_rel =
                    ProteinReliabilityCurve::from_predictions_with_repeats(&pooled, 0.80, 5.0);
                let mean_ece = prot_eces.iter().sum::<f64>() / (prot_eces.len().max(1) as f64);
                let mean_ece_10 =
                    prot_eces_10.iter().sum::<f64>() / (prot_eces_10.len().max(1) as f64);
                let mean_ece_12 =
                    prot_eces_12.iter().sum::<f64>() / (prot_eces_12.len().max(1) as f64);
                let macro_hcp = prot_hcps.iter().sum::<f64>() / (prot_hcps.len().max(1) as f64);
                println!(
                    "  [Identity T=1.0 baseline] mean_ECE4/10/12={mean_ece:.4}/{mean_ece_10:.4}/{mean_ece_12:.4}, pooled_ECE4/10/12={:.4}/{:.4}/{:.4}, macro_HCP={:.2}% (n_hc_prot={}), micro_HCP={:.2}% ({}/{})",
                    pool_rel.ece,
                    pool_rel.ece_10,
                    pool_rel.ece_12,
                    macro_hcp * 100.0,
                    prot_hcps.len(),
                    pool_rel.hc_precision.unwrap_or(0.0) * 100.0,
                    pool_rel.hc_correct,
                    pool_rel.hc_assigned,
                );
            }

            // 2. Strict 13-fold LOPO: fit both PAV knots and `ProteinHcGateConfig` inside each fold
            let mut oof_eces = Vec::with_capacity(by_prot.len());
            let mut oof_eces_10 = Vec::with_capacity(by_prot.len());
            let mut oof_deb_eces_10 = Vec::with_capacity(by_prot.len());
            let mut oof_floor_10 = Vec::with_capacity(by_prot.len());
            let mut oof_eces_12 = Vec::with_capacity(by_prot.len());
            let mut oof_hcps = Vec::with_capacity(by_prot.len());
            let mut oof_pooled = Vec::new();

            for (hold_idx, (_, _, _, held_seeds)) in by_prot.iter().enumerate() {
                let train_pairs = collect_train_pairs(Some(hold_idx));
                let fold_knots = fit_knots_for_solver(&train_pairs);
                let train_prots: Vec<Vec<ProteinHcTrainingSeed>> = by_prot
                    .iter()
                    .enumerate()
                    .filter(|&(j, _)| j != hold_idx)
                    .map(|(_, (_, _, _, s))| s.clone())
                    .collect();
                let fold_gate_cfg = fit_protein_hc_gate_config(&train_prots, &fold_knots);
                let mut held_preds = Vec::new();
                for s in held_seeds {
                    held_preds.extend(evaluate_seed_predictions_with_gate(
                        s,
                        &fold_knots,
                        &fold_gate_cfg,
                    ));
                }
                let rel =
                    ProteinReliabilityCurve::from_predictions_with_repeats(&held_preds, 0.80, 5.0);
                oof_eces.push(rel.ece);
                oof_eces_10.push(rel.ece_10);
                oof_deb_eces_10.push(rel.debiased_ece_10);
                oof_floor_10.push(rel.ece_10_bernoulli_floor);
                oof_eces_12.push(rel.ece_12);
                if let Some(hcp) = rel.hc_precision {
                    oof_hcps.push(hcp);
                }
                oof_pooled.extend(held_preds);
            }
            let oof_pool =
                ProteinReliabilityCurve::from_predictions_with_repeats(&oof_pooled, 0.80, 5.0);
            let oof_mean_ece = oof_eces.iter().sum::<f64>() / (oof_eces.len().max(1) as f64);
            let oof_mean_ece_10 =
                oof_eces_10.iter().sum::<f64>() / (oof_eces_10.len().max(1) as f64);
            let oof_mean_deb_ece_10 =
                oof_deb_eces_10.iter().sum::<f64>() / (oof_deb_eces_10.len().max(1) as f64);
            let oof_mean_floor_10 =
                oof_floor_10.iter().sum::<f64>() / (oof_floor_10.len().max(1) as f64);
            let oof_mean_ece_12 =
                oof_eces_12.iter().sum::<f64>() / (oof_eces_12.len().max(1) as f64);
            let oof_macro_hcp = oof_hcps.iter().sum::<f64>() / (oof_hcps.len().max(1) as f64);
            let oof_micro_hcp = oof_pool.hc_precision.unwrap_or(0.0);

            // 3. Fit full-13 `dev` estimator for deployment
            let all_train = collect_train_pairs(None);
            let full_knots = fit_knots_for_solver(&all_train);
            let all_prots: Vec<Vec<ProteinHcTrainingSeed>> =
                by_prot.iter().map(|(_, _, _, s)| s.clone()).collect();
            let full_gate_cfg = fit_protein_hc_gate_config(&all_prots, &full_knots);

            println!(
                "\n  >>> 13-FOLD IN-FOLD-REFIT LOPO {solver_name} (full_gate={full_gate_cfg:?})"
            );
            println!(
                "  OOF LOPO: mean_ECE4/10/12={oof_mean_ece:.4}/{oof_mean_ece_10:.4}/{oof_mean_ece_12:.4} (debiased_ECE10={oof_mean_deb_ece_10:.4}, Bernoulli_floor={oof_mean_floor_10:.4}), pooled_ECE4/10/12={:.4}/{:.4}/{:.4}, macro_HCP={:.2}%, micro_HCP={:.2}% ({}/{})",
                oof_pool.ece,
                oof_pool.ece_10,
                oof_pool.ece_12,
                oof_macro_hcp * 100.0,
                oof_micro_hcp * 100.0,
                oof_pool.hc_correct,
                oof_pool.hc_assigned,
            );
            for ((((id, name, verd, seeds), &pece), &pece10), &pece12) in by_prot
                .iter()
                .zip(oof_eces.iter())
                .zip(oof_eces_10.iter())
                .zip(oof_eces_12.iter())
            {
                let tot_s: usize = seeds.iter().map(|s| s.raw_predictions.len()).sum();
                let n_ok: usize = seeds
                    .iter()
                    .map(|s| s.raw_predictions.iter().filter(|&&(_, ok)| ok).count())
                    .sum();
                let acc = 100.0 * (n_ok as f64) / (tot_s.max(1) as f64);
                let mut preds_p = Vec::new();
                for s in seeds {
                    preds_p.extend(evaluate_seed_predictions_with_gate(
                        s,
                        &full_knots,
                        &full_gate_cfg,
                    ));
                }
                let rel_p =
                    ProteinReliabilityCurve::from_predictions_with_repeats(&preds_p, 0.80, 5.0);
                println!(
                    "    per-protein {id:16} ({name:12}, QC={:4}): acc={acc:5.1}%, OOF_ECE4/10/12={pece:.4}/{pece10:.4}/{pece12:.4}, INS_HC={}",
                    verd.as_str(),
                    rel_p.hc_status,
                );
            }
            print!("  RUST KNOTS {solver_name}: &[");
            for (i, &(x, y)) in full_knots.iter().enumerate() {
                if i > 0 {
                    print!(", ");
                }
                print!("({x:.4}, {y:.4})");
            }
            println!("];");
            if solver_name != "STANDARD_GPU_UNSHRUNK" {
                gate_checks.push(LopoGateCheck {
                    solver_name: solver_name.to_string(),
                    oof_mean_ece,
                    oof_mean_ece_10,
                    oof_mean_debiased_ece_10: oof_mean_deb_ece_10,
                    oof_pool_ece_10: oof_pool.ece_10,
                    oof_micro_hcp,
                    oof_macro_hcp,
                    full_knots,
                });
            }
        };

        run_lopo_calibration("EVO_GRAPH", &evo_by_prot);
        run_lopo_calibration("STANDARD_CPU", &std_by_prot);
        run_lopo_calibration("STANDARD_GPU_UNSHRUNK", &std_gpu_by_prot);

        // P1-8: Strict NESTED 13x12 LOPO selection of `CANONICAL_GPU_CPU_SHRINKAGE_WEIGHT` for
        // `STANDARD_GPU` (selecting `w_i` strictly inside the 12 training proteins of outer fold `i`
        // via 12-fold inner LOPO minimizing `ECE_10`, holding out `hold_idx` from BOTH GPU and CPU!).
        let full_selected_gpu_w = {
            println!(
                "\n================ P1-8 NESTED 13x12 LOPO GPU SHRINKAGE SELECTION ================"
            );
            let mut oof_eces = Vec::new();
            let mut oof_eces_10 = Vec::new();
            let mut oof_deb_eces_10 = Vec::new();
            let mut oof_floor_10 = Vec::new();
            let mut oof_eces_12 = Vec::new();
            let mut oof_hcps = Vec::new();
            let mut oof_pooled = Vec::new();
            let mut outer_chosen_ws = Vec::new();

            for (hold_idx, (_, _, _, held_gpu_seeds)) in std_gpu_by_prot.iter().enumerate() {
                let mut tr_gpu_prots = Vec::new();
                let mut tr_cpu_raw = Vec::new();
                let mut tr_gpu_ok = Vec::new();
                let mut tr_cpu_ok = Vec::new();
                let mut gpu_train_flat = Vec::new();
                let mut cpu_train_flat = Vec::new();

                for idx in 0..std_gpu_by_prot.len() {
                    if idx == hold_idx {
                        continue;
                    }
                    let g_seeds = &std_gpu_by_prot[idx].3;
                    let c_seeds = &std_by_prot[idx].3;
                    let g_ok = is_trainable_prot(g_seeds);
                    let c_ok = is_trainable_prot(c_seeds);
                    let c_pairs: Vec<(f64, bool)> = c_seeds
                        .iter()
                        .flat_map(|s| s.raw_predictions.iter().copied())
                        .collect();
                    if g_ok {
                        for s in g_seeds {
                            gpu_train_flat.extend_from_slice(&s.raw_predictions);
                        }
                    }
                    if c_ok {
                        cpu_train_flat.extend_from_slice(&c_pairs);
                    }
                    tr_gpu_prots.push(g_seeds.clone());
                    tr_cpu_raw.push(c_pairs);
                    tr_gpu_ok.push(g_ok);
                    tr_cpu_ok.push(c_ok);
                }

                // Inner LOPO selector inside the 12 training proteins of outer fold `hold_idx`
                let init_knots = fit_isotonic_monotone_knots_shrunk_toward(
                    &gpu_train_flat,
                    &cpu_train_flat,
                    0.50,
                    CANONICAL_CALIBRATION_GRID_12,
                    CANONICAL_CALIBRATION_LAPLACE_PRIOR,
                );
                let init_gate = fit_protein_hc_gate_config(&tr_gpu_prots, &init_knots);
                let (w_i, _) = select_gpu_shrinkage_weight_nested_lopo(
                    &tr_gpu_prots,
                    &tr_cpu_raw,
                    &tr_gpu_ok,
                    &tr_cpu_ok,
                    &init_gate,
                );
                outer_chosen_ws.push(w_i);

                let gpu_fold_knots = fit_isotonic_monotone_knots_shrunk_toward(
                    &gpu_train_flat,
                    &cpu_train_flat,
                    w_i,
                    CANONICAL_CALIBRATION_GRID_12,
                    CANONICAL_CALIBRATION_LAPLACE_PRIOR,
                );
                let fold_gate = fit_protein_hc_gate_config(&tr_gpu_prots, &gpu_fold_knots);
                let mut held_preds = Vec::new();
                for s in held_gpu_seeds {
                    held_preds.extend(evaluate_seed_predictions_with_gate(
                        s,
                        &gpu_fold_knots,
                        &fold_gate,
                    ));
                }
                let rel =
                    ProteinReliabilityCurve::from_predictions_with_repeats(&held_preds, 0.80, 5.0);
                oof_eces.push(rel.ece);
                oof_eces_10.push(rel.ece_10);
                oof_deb_eces_10.push(rel.debiased_ece_10);
                oof_floor_10.push(rel.ece_10_bernoulli_floor);
                oof_eces_12.push(rel.ece_12);
                if let Some(hcp) = rel.hc_precision {
                    oof_hcps.push(hcp);
                }
                oof_pooled.extend(held_preds);
            }

            let pool =
                ProteinReliabilityCurve::from_predictions_with_repeats(&oof_pooled, 0.80, 5.0);
            let m_ece4 = oof_eces.iter().sum::<f64>() / (oof_eces.len().max(1) as f64);
            let m_ece10 = oof_eces_10.iter().sum::<f64>() / (oof_eces_10.len().max(1) as f64);
            let m_deb10 =
                oof_deb_eces_10.iter().sum::<f64>() / (oof_deb_eces_10.len().max(1) as f64);
            let m_flr10 = oof_floor_10.iter().sum::<f64>() / (oof_floor_10.len().max(1) as f64);
            let m_ece12 = oof_eces_12.iter().sum::<f64>() / (oof_eces_12.len().max(1) as f64);
            let m_hcp = oof_hcps.iter().sum::<f64>() / (oof_hcps.len().max(1) as f64);
            let micro_hcp = pool.hc_precision.unwrap_or(0.0);

            println!(
                "  [NESTED 13x12 LOPO STANDARD_GPU] outer_ws={outer_chosen_ws:?}\n  OOF mean_ECE4/10/12={m_ece4:.4}/{m_ece10:.4}/{m_ece12:.4} (debiased_ECE10={m_deb10:.4}, Bernoulli_floor={m_flr10:.4}), pooled_ECE4/10/12={:.4}/{:.4}/{:.4}, macro_HCP={:.2}%, micro_HCP={:.2}% ({}/{})",
                pool.ece,
                pool.ece_10,
                pool.ece_12,
                m_hcp * 100.0,
                micro_hcp * 100.0,
                pool.hc_correct,
                pool.hc_assigned,
            );

            // Full-13 inner LOPO selection of `CANONICAL_GPU_CPU_SHRINKAGE_WEIGHT`
            let full_gpu_prots: Vec<Vec<ProteinHcTrainingSeed>> = std_gpu_by_prot
                .iter()
                .map(|(_, _, _, s)| s.clone())
                .collect();
            let full_cpu_raw: Vec<Vec<(f64, bool)>> = std_by_prot
                .iter()
                .map(|(_, _, _, s)| {
                    s.iter()
                        .flat_map(|sd| sd.raw_predictions.iter().copied())
                        .collect()
                })
                .collect();
            let full_gpu_ok: Vec<bool> = std_gpu_by_prot
                .iter()
                .map(|(_, _, _, s)| is_trainable_prot(s))
                .collect();
            let full_cpu_ok: Vec<bool> = std_by_prot
                .iter()
                .map(|(_, _, _, s)| is_trainable_prot(s))
                .collect();
            let all_gpu: Vec<(f64, bool)> = std_gpu_by_prot
                .iter()
                .filter(|(_, _, _, s)| is_trainable_prot(s))
                .flat_map(|(_, _, _, s)| s.iter().flat_map(|sd| sd.raw_predictions.iter().copied()))
                .collect();
            let all_cpu: Vec<(f64, bool)> = std_by_prot
                .iter()
                .filter(|(_, _, _, s)| is_trainable_prot(s))
                .flat_map(|(_, _, _, s)| s.iter().flat_map(|sd| sd.raw_predictions.iter().copied()))
                .collect();

            let full_init_knots = fit_isotonic_monotone_knots_shrunk_toward(
                &all_gpu,
                &all_cpu,
                0.50,
                CANONICAL_CALIBRATION_GRID_12,
                CANONICAL_CALIBRATION_LAPLACE_PRIOR,
            );
            let full_init_gate = fit_protein_hc_gate_config(&full_gpu_prots, &full_init_knots);
            let (sel_w, cand_stats) = select_gpu_shrinkage_weight_nested_lopo(
                &full_gpu_prots,
                &full_cpu_raw,
                &full_gpu_ok,
                &full_cpu_ok,
                &full_init_gate,
            );
            println!(
                "  [FULL-13 INNER LOPO] selected CANONICAL_GPU_CPU_SHRINKAGE_WEIGHT = {sel_w:.2}:"
            );
            for (cw, cece10, chcp) in &cand_stats {
                println!(
                    "    w={cw:4.2}: inner_OOF_ECE10={cece10:.5}, inner_OOF_micro_HCP={:.2}%",
                    chcp * 100.0
                );
            }

            let full_gpu_shrunk = fit_isotonic_monotone_knots_shrunk_toward(
                &all_gpu,
                &all_cpu,
                sel_w,
                CANONICAL_CALIBRATION_GRID_12,
                CANONICAL_CALIBRATION_LAPLACE_PRIOR,
            );
            print!("  RUST KNOTS STANDARD_GPU (w={sel_w:.2}): &[");
            for (i, &(x, y)) in full_gpu_shrunk.iter().enumerate() {
                if i > 0 {
                    print!(", ");
                }
                print!("({x:.4}, {y:.4})");
            }
            println!("];");

            gate_checks.push(LopoGateCheck {
                solver_name: "STANDARD_GPU".to_string(),
                oof_mean_ece: m_ece4,
                oof_mean_ece_10: m_ece10,
                oof_mean_debiased_ece_10: m_deb10,
                oof_pool_ece_10: pool.ece_10,
                oof_micro_hcp: micro_hcp,
                oof_macro_hcp: m_hcp,
                full_knots: full_gpu_shrunk,
            });
            sel_w
        };

        if only_filter.is_none() {
            let is_reference_gpu_adapter = gpu_engine.adapter_name.contains("Apple M5 Pro");
            if is_reference_gpu_adapter {
                assert!(
                    (full_selected_gpu_w - CANONICAL_GPU_CPU_SHRINKAGE_WEIGHT).abs() < 1e-9,
                    "full-13 inner LOPO selected w={full_selected_gpu_w:.2} != CANONICAL_GPU_CPU_SHRINKAGE_WEIGHT={CANONICAL_GPU_CPU_SHRINKAGE_WEIGHT:.2}"
                );
            }
            for LopoGateCheck {
                solver_name,
                oof_mean_ece,
                oof_mean_ece_10,
                oof_mean_debiased_ece_10,
                oof_pool_ece_10,
                oof_micro_hcp,
                oof_macro_hcp,
                full_knots,
            } in gate_checks
            {
                assert!(
                    oof_pool_ece_10 <= 0.050,
                    "{solver_name} OOF pooled_ECE_10 {oof_pool_ece_10:.4} exceeded primary 0.050 gate"
                );
                assert!(
                    oof_micro_hcp >= 0.94,
                    "{solver_name} OOF micro_HCP {:.2}% fell below primary 94.0% gate",
                    oof_micro_hcp * 100.0
                );
                assert!(
                    oof_macro_hcp >= 0.95,
                    "{solver_name} OOF macro_HCP {:.2}% fell below 95.0% gate",
                    oof_macro_hcp * 100.0
                );
                assert!(
                    oof_mean_ece <= 0.065,
                    "{solver_name} OOF mean_ECE_4 {oof_mean_ece:.4} exceeded 0.065 gate"
                );
                assert!(
                    oof_mean_ece_10 <= 0.105,
                    "{solver_name} OOF mean_ECE_10 {oof_mean_ece_10:.4} exceeded finite-sample 0.105 gate"
                );
                assert!(
                    oof_mean_debiased_ece_10 <= 0.065,
                    "{solver_name} OOF mean_debiased_ECE_10 {oof_mean_debiased_ece_10:.4} exceeded 0.065 gate"
                );
                let committed = match solver_name.as_str() {
                    "EVO_GRAPH" => nmr_core::qc::EVO_GRAPH_CALIBRATION_KNOTS,
                    "STANDARD_CPU" => nmr_core::qc::STANDARD_CALIBRATION_KNOTS,
                    "STANDARD_GPU" => nmr_core::qc::STANDARD_GPU_CALIBRATION_KNOTS,
                    other => panic!("unexpected solver_name {other}"),
                };
                assert_eq!(
                    committed.len(),
                    full_knots.len(),
                    "{solver_name} committed knot count mismatch"
                );
                let tol = if solver_name == "STANDARD_GPU" && !is_reference_gpu_adapter {
                    0.05
                } else {
                    2e-4
                };
                for ((cx, cy), (fx, fy)) in committed.iter().zip(full_knots.iter()) {
                    assert!(
                        (cx - fx).abs() <= tol && (cy - fy).abs() <= tol,
                        "{solver_name} committed knot ({cx:.4}, {cy:.4}) drifted from fitted ({fx:.4}, {fy:.4}) beyond tol={tol} (adapter={:?})",
                        gpu_engine.adapter_name
                    );
                }
            }
        }
    }
}
