//! Raw time-domain (FID) readers for Bruker (`fid` / `ser` + `acqus`) and Varian/Agilent (`fid` + `procpar`),
//! including automatic uniform vs Non-Uniform Sampling (NUS) detection and hypercomplex grid decoding.
//!
//! Provenance: Clean-room original implementation of `docs/specs/11-fid-processing.md` section 1
//! from published methods and public format specifications:
//! - Hypercomplex quadrature & Echo-Antiecho recombination: States, Haberkorn & Ruben,
//!   *J. Magn. Reson.* 1982, 48, 286–292; Marion et al., *J. Magn. Reson.* 1989, 85, 393–399;
//!   Palmer et al., *J. Magn. Reson.* 1991, 93, 151–170; Kay, Keifer & Saarinen,
//!   *J. Am. Chem. Soc.* 1992, 114, 10663–10665.
//! - Non-uniform sampling (NUS) hypercomplex time-grid representation for compressed-sensing / IST
//!   reconstruction: Stern, Donoho & Hoch, *J. Magn. Reson.* 2007, 188, 295–300;
//!   Holland et al., *Angew. Chem. Int. Ed.* 2011, 50, 6548–6551;
//!   Hyberts et al., *J. Biomol. NMR* 2012, 52, 315–327.
//! - Bruker digital filter group delay: Moskau, *Concepts Magn. Reson.* 2002, 15, 164–176;
//!   Westler & Abildgaard (1996, NMRFAM technical note) `(DSPFVS, DECIM)` lookup table.
//! - IUPAC indirect chemical shift referencing: Wishart et al., *J. Biomol. NMR* 1995, 6, 135–140.
//! - Public file formats: Bruker JCAMP-DX v5.0 (`acqus`/`acqu2s`/`acqu3s`, `procs`, `nuslist`,
//!   `fid`/`ser`) and Varian/Agilent VnmrJ Data File Reference (`datafilehead`, `datablockhead`,
//!   `procpar`, `sampling.sch`).

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Spectrometer vendor format for a raw time-domain dataset.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FidVendor {
    Bruker,
    Varian,
}

/// Quadrature / acquisition mode for a time-domain axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuadMode {
    /// Simultaneous / digital quadrature detection on the direct axis (`AQ_mod = 3` or `4`, or Varian complex).
    Dqd,
    /// Single-channel / magnitude / QF mode (`FnMODE = 1` or Varian `phase = 1`).
    Qf,
    /// Hypercomplex cosine/sine indirect quadrature (`FnMODE = 4` or Varian `phase = 1,2`).
    States,
    /// Hypercomplex States-TPPI (`FnMODE = 5`, $180^\circ$ odd-increment alternation).
    StatesTppi,
    /// Sensitivity-enhanced gradient Echo-Antiecho / Rance-Kay (`FnMODE = 6` or Varian `fcoef`).
    EchoAntiecho,
}

/// Stored apodization hint extracted from Bruker `procs` or Varian `procpar`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum StoredWindowHint {
    None,
    Exponential { lb_hz: f64 },
    SineBell { offset_pi_frac: f64 },
    SquaredSineBell { offset_pi_frac: f64 },
}

/// Optional stored processing defaults discovered alongside the raw FID (`procs` or `procpar`).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct StoredDimProc {
    pub p0_deg: Option<f64>,
    pub p1_deg: Option<f64>,
    pub window: Option<StoredWindowHint>,
    pub si: Option<usize>,
    pub td_eff: Option<usize>,
    pub fcor: Option<f64>,
    pub reverse: bool,
    pub bc_mod: Option<i32>,
    pub bcfw_ppm: Option<f64>,
}

/// Metadata for a single time-domain axis.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FidAxisMeta {
    pub label: String,
    /// Raw scalar word count on the full Nyquist grid (`TD` on uniform Bruker, `NusTD` on NUS Bruker, `np` or `2 * ni` on Varian).
    pub td_raw: usize,
    /// Number of complex time-domain increments (`td_raw / 2` for complex/hypercomplex, `td_raw` for `QF` indirect).
    pub complex_points: usize,
    /// Spectral sweep width in Hz.
    pub sw_hz: f64,
    /// Spectrometer reference frequency (`0 ppm` denominator) in MHz.
    pub obs_mhz: f64,
    /// Transmitter carrier frequency in MHz.
    pub carrier_mhz: f64,
    /// Chemical shift (ppm) at frequency-domain index `0` (downfield / high-ppm edge).
    pub first_ppm: f64,
    pub quad_mode: QuadMode,
    /// Digital filter group delay in complex points (`0.0` on indirect axes).
    pub group_delay: f64,
    /// Sign convention for complex DFT (`true` = conjugate time signal before standard forward DFT,
    /// matching Bruker DQD `FT_mod = 6` and States/Echo-Antiecho descending-ppm order).
    pub conjugate_for_fft: bool,
    pub stored_proc: StoredDimProc,
}

/// Complex `f64` sample `(re, im)`.
pub type Complex64 = [f64; 2];

/// Raw 1D FID (`[n0]` complex points).
#[derive(Clone, Debug)]
pub struct Fid1D {
    pub name: String,
    pub vendor: FidVendor,
    pub pulse_program: String,
    pub axis_x: FidAxisMeta,
    pub data: Vec<Complex64>,
}

/// Raw 2D FID normalized into uniform hypercomplex `[n1_cplx * 2 * n0]` traces:
/// for indirect increment `k1 in 0..n1_cplx`, `comp = 0` is the cosine trace and `comp = 1` is the sine trace.
#[derive(Clone, Debug)]
pub struct Fid2D {
    pub name: String,
    pub vendor: FidVendor,
    pub pulse_program: String,
    /// Direct acquisition axis (`1H`).
    pub axis_x: FidAxisMeta,
    /// Indirect axis (`F1` in 2D, e.g. `15N` or `13C`).
    pub axis_y: FidAxisMeta,
    pub data: Vec<Complex64>,
    /// `Some` of length `axis_y.complex_points` (`true` at sampled `k1`, `false` and zero-filled in `data`
    /// at unsampled `k1`) when non-uniformly sampled (NUS), or `None` when uniformly sampled.
    pub nus_mask: Option<Vec<bool>>,
}

impl Fid2D {
    #[inline]
    pub fn trace(&self, k1: usize, sin_comp: usize) -> &[Complex64] {
        let n0 = self.axis_x.complex_points;
        let base = (k1 * 2 + sin_comp) * n0;
        &self.data[base..base + n0]
    }
}

/// Raw 3D FID normalized into uniform hypercomplex `[n2_cplx * 2 * n1_cplx * 2 * n0]` traces:
/// `axis_x` = direct (`F3`, `1H`), `axis_y` = `F2` (`acqu2s` / `ni`), `axis_z` = `F1` (`acqu3s` / `ni2`).
#[derive(Clone, Debug)]
pub struct Fid3D {
    pub name: String,
    pub vendor: FidVendor,
    pub pulse_program: String,
    pub axis_x: FidAxisMeta,
    pub axis_y: FidAxisMeta,
    pub axis_z: FidAxisMeta,
    pub data: Vec<Complex64>,
    /// `Some` of length `axis_z.complex_points * axis_y.complex_points` (indexed `k2 * n1_cplx + k1`,
    /// `true` where sampled and zero-filled in `data` where unsampled) when NUS, or `None` when uniform.
    pub nus_mask: Option<Vec<bool>>,
}

impl Fid3D {
    #[inline]
    pub fn trace(&self, k2: usize, c2: usize, k1: usize, c1: usize) -> &[Complex64] {
        let n0 = self.axis_x.complex_points;
        let n1 = self.axis_y.complex_points;
        let idx = (((k2 * 2 + c2) * n1 + k1) * 2 + c1) * n0;
        &self.data[idx..idx + n0]
    }
}

/// Sum types returned by `read_fid_auto`.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug)]
pub enum RawFid {
    Dim1(Fid1D),
    Dim2(Fid2D),
    Dim3(Fid3D),
}

// ============================================================================
// Bruker Digital Filter Group Delay Lookup (Westler & Abildgaard 1996 / Moskau 2002)
// ============================================================================

/// Look up legacy Bruker digital filter group delay (in complex points) from `(DSPFVS, DECIM)`
/// when explicit `GRPDLY` is absent or negative.
pub fn bruker_dsp_group_delay(dspfvs: i32, decim: i32) -> Result<f64, String> {
    if decim <= 1 || dspfvs <= 0 {
        return Ok(0.0);
    }
    let val = match dspfvs {
        10 => match decim {
            2 => Some(44.7500),
            3 => Some(33.5000),
            4 => Some(66.6250),
            6 => Some(59.083333333333336),
            8 => Some(68.5625),
            12 => Some(60.3750),
            16 => Some(69.53125),
            24 => Some(61.020833333333336),
            32 => Some(70.015625),
            48 => Some(61.34375),
            64 => Some(70.2578125),
            96 => Some(61.505208333333336),
            128 => Some(70.37890625),
            192 => Some(61.5859375),
            256 => Some(70.439453125),
            384 => Some(61.626302083333336),
            512 => Some(70.4697265625),
            768 => Some(61.646484375),
            1024 => Some(70.48486328125),
            1536 => Some(61.656575520833336),
            2048 => Some(70.492431640625),
            _ => None,
        },
        11 => match decim {
            2 => Some(46.0000),
            3 => Some(36.5000),
            4 => Some(48.0000),
            6 => Some(50.166666666666664),
            8 => Some(53.2500),
            12 => Some(69.5000),
            16 => Some(72.2500),
            24 => Some(70.16666666666667),
            32 => Some(72.7500),
            48 => Some(70.5000),
            64 => Some(73.0000),
            96 => Some(70.66666666666667),
            128 => Some(72.5000),
            192 => Some(71.33333333333333),
            256 => Some(72.2500),
            384 => Some(71.66666666666667),
            512 => Some(72.1250),
            768 => Some(71.83333333333333),
            1024 => Some(72.0625),
            1536 => Some(71.91666666666667),
            2048 => Some(72.03125),
            _ => None,
        },
        12 => match decim {
            2 => Some(46.3110),
            3 => Some(36.5300),
            4 => Some(47.8700),
            6 => Some(50.2290),
            8 => Some(53.2890),
            12 => Some(69.5510),
            16 => Some(71.6000),
            24 => Some(70.1840),
            32 => Some(72.1380),
            48 => Some(70.5280),
            64 => Some(72.3480),
            96 => Some(70.7000),
            128 => Some(72.5240),
            192 => Some(71.33333333333333),
            256 => Some(72.2500),
            384 => Some(71.66666666666667),
            512 => Some(72.1250),
            768 => Some(71.83333333333333),
            1024 => Some(72.0625),
            1536 => Some(71.91666666666667),
            2048 => Some(72.03125),
            _ => None,
        },
        13 => match decim {
            2 => Some(2.7500),
            3 => Some(2.8333333333333335),
            4 => Some(2.8750),
            6 => Some(2.9166666666666665),
            8 => Some(2.9375),
            12 => Some(2.9583333333333335),
            16 => Some(2.96875),
            24 => Some(2.9791666666666665),
            32 => Some(2.984375),
            48 => Some(2.9895833333333335),
            64 => Some(2.9921875),
            _ => None,
        },
        _ => None,
    };
    val.ok_or_else(|| {
        format!(
            "Bruker digital filter group delay unknown: GRPDLY missing/negative and (DSPFVS={dspfvs}, DECIM={decim}) not in lookup table"
        )
    })
}

// ============================================================================
// Auto-detection Entrypoints
// ============================================================================

/// Locate a child entry inside `dir` by ASCII case-insensitive filename match (preferring exact
/// case when multiple entries collide on case-sensitive filesystems such as Linux `ext4`), returning
/// the actual on-disk `PathBuf` when found.
pub fn find_child_ci(dir: &Path, name: &str) -> Option<PathBuf> {
    let entries = fs::read_dir(dir).ok()?;
    let mut ci_match: Option<PathBuf> = None;
    for entry in entries.flatten() {
        let fname = entry.file_name();
        if let Some(s) = fname.to_str() {
            if s == name {
                return Some(entry.path());
            }
            if ci_match.is_none() && s.eq_ignore_ascii_case(name) {
                ci_match = Some(entry.path());
            }
        }
    }
    ci_match
}

fn find_file_ci(dir: &Path, name: &str) -> Option<PathBuf> {
    find_child_ci(dir, name).filter(|p| p.is_file())
}

fn find_dir_ci(dir: &Path, name: &str) -> Option<PathBuf> {
    find_child_ci(dir, name).filter(|p| p.is_dir())
}

fn find_relpath_file_ci(dir: &Path, rel: &str) -> Option<PathBuf> {
    let mut cur = dir.to_path_buf();
    for comp in Path::new(rel).components() {
        match comp {
            std::path::Component::Normal(os) => {
                let seg = os.to_str()?;
                cur = find_child_ci(&cur, seg)?;
            }
            std::path::Component::CurDir => {}
            _ => return None,
        }
    }
    cur.is_file().then_some(cur)
}

/// Detect vendor and dimensionality of a raw FID directory (`acqus` -> Bruker, `procpar` -> Varian)
/// and load the dataset (automatically detecting uniform vs NUS schedules).
pub fn read_fid_auto(dir: impl AsRef<Path>) -> Result<RawFid, String> {
    let p = resolve_fid_dir(dir.as_ref())?;
    if find_file_ci(&p, "acqus").is_some() || find_file_ci(&p, "acqu").is_some() {
        let has_3 = find_file_ci(&p, "acqu3s").is_some() || find_file_ci(&p, "acqu3").is_some();
        let has_2 = find_file_ci(&p, "acqu2s").is_some() || find_file_ci(&p, "acqu2").is_some();
        let has_ser = find_file_ci(&p, "ser").is_some();
        if has_3 && has_ser {
            Ok(RawFid::Dim3(read_bruker_fid_3d(&p)?))
        } else if has_2 && has_ser {
            Ok(RawFid::Dim2(read_bruker_fid_2d(&p)?))
        } else {
            Ok(RawFid::Dim1(read_bruker_fid_1d(&p)?))
        }
    } else if find_file_ci(&p, "procpar").is_some() {
        read_varian_fid_auto(&p)
    } else {
        Err(format!(
            "unrecognized FID directory {}: neither Bruker acqus nor Varian procpar found",
            p.display()
        ))
    }
}

/// Read a 2D raw FID directory (Bruker `ser` + `acqus`/`acqu2s` or Varian `fid` + `procpar`).
pub fn read_fid_2d(dir: impl AsRef<Path>) -> Result<Fid2D, String> {
    match read_fid_auto(dir)? {
        RawFid::Dim2(f) => Ok(f),
        RawFid::Dim1(_) => Err("expected 2D FID dataset, found 1D FID".to_string()),
        RawFid::Dim3(_) => Err("expected 2D FID dataset, found 3D FID".to_string()),
    }
}

/// Read a 3D raw FID directory (Bruker `ser` + `acqus`/`acqu2s`/`acqu3s` or Varian `fid` + `procpar`).
pub fn read_fid_3d(dir: impl AsRef<Path>) -> Result<Fid3D, String> {
    match read_fid_auto(dir)? {
        RawFid::Dim3(f) => Ok(f),
        RawFid::Dim1(_) => Err("expected 3D FID dataset, found 1D FID".to_string()),
        RawFid::Dim2(_) => Err("expected 3D FID dataset, found 2D FID".to_string()),
    }
}

fn resolve_fid_dir(path: &Path) -> Result<PathBuf, String> {
    if path.is_file() {
        if let Some(parent) = path.parent() {
            return Ok(parent.to_path_buf());
        }
    }
    if path.is_dir() {
        return Ok(path.to_path_buf());
    }
    Err(format!("FID path does not exist: {}", path.display()))
}

// ============================================================================
// NUS Schedule File Helpers (Bruker `nuslist` & Varian `sampling.sch`)
// ============================================================================

fn parse_nus_schedule_1d(path: &Path, n1_cplx: usize) -> Result<Vec<usize>, String> {
    let text = fs::read_to_string(path)
        .map_err(|e| format!("failed to read NUS schedule {}: {e}", path.display()))?;
    let mut coords = Vec::new();
    for (line_no, raw_line) in text.lines().enumerate() {
        let clean = raw_line.split(['#', ';']).next().unwrap_or("").trim();
        if clean.is_empty() {
            continue;
        }
        let toks: Vec<&str> = clean.split_whitespace().collect();
        if toks.len() != 1 {
            return Err(format!(
                "invalid 2D NUS schedule {} at line {}: expected 1 integer coordinate, got '{}'",
                path.display(),
                line_no + 1,
                clean
            ));
        }
        let k1: usize = toks[0].parse().map_err(|e| {
            format!(
                "invalid integer '{}' in 2D NUS schedule {} at line {}: {e}",
                toks[0],
                path.display(),
                line_no + 1
            )
        })?;
        if k1 >= n1_cplx {
            return Err(format!(
                "out-of-bounds 2D NUS coordinate k1={k1} in {} at line {} (full grid n1_cplx={n1_cplx})",
                path.display(),
                line_no + 1
            ));
        }
        coords.push(k1);
    }
    if coords.is_empty() {
        return Err(format!("NUS schedule file {} is empty", path.display()));
    }
    Ok(coords)
}

/// Parse a 2-column 3D NUS schedule (`col0 col1` per line).
/// When `col0_is_y` is `true` (Bruker `AQSEQ == 0` `3-2-1` and Varian `ni, ni2`), returns `(k1_y, k2_z)`;
/// when `col0_is_y` is `false` (Bruker `AQSEQ == 1` `3-1-2`), returns `(k1_y, k2_z)` with columns swapped.
fn parse_nus_schedule_2d(
    path: &Path,
    n1_cplx: usize,
    n2_cplx: usize,
    col0_is_y: bool,
) -> Result<Vec<(usize, usize)>, String> {
    let text = fs::read_to_string(path)
        .map_err(|e| format!("failed to read 3D NUS schedule {}: {e}", path.display()))?;
    let mut coords = Vec::new();
    for (line_no, raw_line) in text.lines().enumerate() {
        let clean = raw_line.split(['#', ';']).next().unwrap_or("").trim();
        if clean.is_empty() {
            continue;
        }
        let toks: Vec<&str> = clean.split_whitespace().collect();
        if toks.len() != 2 {
            return Err(format!(
                "invalid 3D NUS schedule {} at line {}: expected 2 integer coordinates, got '{}'",
                path.display(),
                line_no + 1,
                clean
            ));
        }
        let c0: usize = toks[0].parse().map_err(|e| {
            format!(
                "invalid integer '{}' in 3D NUS schedule {} at line {}: {e}",
                toks[0],
                path.display(),
                line_no + 1
            )
        })?;
        let c1: usize = toks[1].parse().map_err(|e| {
            format!(
                "invalid integer '{}' in 3D NUS schedule {} at line {}: {e}",
                toks[1],
                path.display(),
                line_no + 1
            )
        })?;
        let (k1, k2) = if col0_is_y { (c0, c1) } else { (c1, c0) };
        if k1 >= n1_cplx || k2 >= n2_cplx {
            return Err(format!(
                "out-of-bounds 3D NUS coordinate (k1={k1}, k2={k2}) in {} at line {} (full grid n1_cplx={n1_cplx}, n2_cplx={n2_cplx})",
                path.display(),
                line_no + 1
            ));
        }
        coords.push((k1, k2));
    }
    if coords.is_empty() {
        return Err(format!("3D NUS schedule file {} is empty", path.display()));
    }
    Ok(coords)
}

fn find_bruker_nuslist_file(dir: &Path, acqus: &HashMap<String, String>) -> Option<PathBuf> {
    if let Some(nl_name) = acqus.get("NUSLIST") {
        let trimmed = nl_name.trim();
        if !trimmed.is_empty()
            && !trimmed.eq_ignore_ascii_case("automatic")
            && !trimmed.eq_ignore_ascii_case("none")
        {
            if let Some(candidate) = find_relpath_file_ci(dir, trimmed) {
                if fs::metadata(&candidate).map(|m| m.len()).unwrap_or(0) > 0 {
                    return Some(candidate);
                }
            }
        }
    }
    if let Some(default_nl) = find_file_ci(dir, "nuslist") {
        if fs::metadata(&default_nl).map(|m| m.len()).unwrap_or(0) > 0 {
            return Some(default_nl);
        }
    }
    None
}

fn find_varian_schedule_file(dir: &Path, pp: &HashMap<String, ProcparVal>) -> Option<PathBuf> {
    if let Some(custom) = pp.get("sparse_file").and_then(ProcparVal::first_str) {
        let trimmed = custom.trim();
        if !trimmed.is_empty() {
            if let Some(p) = find_relpath_file_ci(dir, trimmed) {
                if fs::metadata(&p).map(|m| m.len()).unwrap_or(0) > 0 {
                    return Some(p);
                }
            }
        }
    }
    for name in ["sampling.sch", "nuslist"] {
        if let Some(p) = find_file_ci(dir, name) {
            if fs::metadata(&p).map(|m| m.len()).unwrap_or(0) > 0 {
                return Some(p);
            }
        }
    }
    None
}

// ============================================================================
// Bruker Raw FID (`fid` / `ser`) Reader
// ============================================================================

fn parse_jcamp_file(path: &Path) -> Result<HashMap<String, String>, String> {
    let text =
        fs::read_to_string(path).map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    Ok(parse_jcamp_str(&text))
}

fn parse_jcamp_str(text: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    let mut lines = text.lines().peekable();
    while let Some(line) = lines.next() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("##$") {
            if let Some((k, v)) = rest.split_once('=') {
                let key = k.trim().to_string();
                let rhs = v.trim();
                if rhs.starts_with('(') && rhs.contains("..") {
                    let mut arr_tokens = Vec::new();
                    while let Some(&next_line) = lines.peek() {
                        let nt = next_line.trim();
                        if nt.starts_with("##") || nt.starts_with("$$") {
                            break;
                        }
                        arr_tokens.push(nt);
                        lines.next();
                    }
                    map.insert(key, arr_tokens.join(" "));
                } else {
                    let clean_v = rhs.trim_matches(|c| c == '<' || c == '>').trim();
                    map.insert(key, clean_v.to_string());
                }
            }
        }
    }
    map
}

fn read_merged_bruker_params(
    dir: &Path,
    primary: &str,
    fallback: &str,
) -> Result<HashMap<String, String>, String> {
    let mut merged = HashMap::new();
    if let Some(fb_path) = find_file_ci(dir, fallback) {
        merged.extend(parse_jcamp_file(&fb_path)?);
    }
    if let Some(pr_path) = find_file_ci(dir, primary) {
        merged.extend(parse_jcamp_file(&pr_path)?);
    }
    if merged.is_empty() {
        return Err(format!(
            "missing Bruker parameter file {} (or {}) in {}",
            primary,
            fallback,
            dir.display()
        ));
    }
    Ok(merged)
}

fn read_optional_bruker_proc(dir: &Path, proc_name: &str) -> Option<HashMap<String, String>> {
    if let Some(pdata_dir) = find_dir_ci(dir, "pdata") {
        if let Some(d1) = find_dir_ci(&pdata_dir, "1") {
            if let Some(p1) = find_file_ci(&d1, proc_name) {
                return parse_jcamp_file(&p1).ok();
            }
        }
    }
    if let Some(p_direct) = find_file_ci(dir, proc_name) {
        return parse_jcamp_file(&p_direct).ok();
    }
    None
}

fn extract_bruker_stored_proc(proc_map: Option<&HashMap<String, String>>) -> StoredDimProc {
    let Some(pm) = proc_map else {
        return StoredDimProc::default();
    };
    let ph_mod = pm
        .get("PH_mod")
        .and_then(|s| s.parse::<i32>().ok())
        .unwrap_or(1);
    let p0_deg =
        pm.get("PHC0")
            .and_then(|s| s.parse::<f64>().ok())
            .map(|v| if ph_mod == 0 { 0.0 } else { v });
    let p1_deg =
        pm.get("PHC1")
            .and_then(|s| s.parse::<f64>().ok())
            .map(|v| if ph_mod == 0 { 0.0 } else { v });

    let wdw = pm
        .get("WDW")
        .and_then(|s| s.parse::<i32>().ok())
        .unwrap_or(0);
    let ssb = pm
        .get("SSB")
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(2.0);
    let lb = pm
        .get("LB")
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0);
    let offset_pi_frac = if ssb >= 2.0 { 1.0 / ssb } else { 0.0 };
    let window = Some(match wdw {
        0 => StoredWindowHint::None,
        1 => StoredWindowHint::Exponential { lb_hz: lb },
        3 => StoredWindowHint::SineBell { offset_pi_frac },
        4 => StoredWindowHint::SquaredSineBell { offset_pi_frac },
        _ => StoredWindowHint::SquaredSineBell {
            offset_pi_frac: 0.5,
        },
    });

    let si_raw = pm
        .get("SI")
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v > 0);
    let ftsize = pm
        .get("FTSIZE")
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v > 0);
    // When TopSpin 4 strip-transform (`STSI < FTSIZE`) crops `SI`, `FTSIZE` holds the true uncropped FFT size.
    let si = match (si_raw, ftsize) {
        (Some(s), Some(ft)) => Some(s.max(ft)),
        (Some(s), None) => Some(s),
        (None, Some(ft)) => Some(ft),
        (None, None) => None,
    };

    let td_eff = pm
        .get("TDeff")
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v > 0);
    let fcor = pm.get("FCOR").and_then(|s| s.parse::<f64>().ok());
    let reverse = pm
        .get("REVERSE")
        .is_some_and(|v| v.eq_ignore_ascii_case("yes") || v == "1");
    let bc_mod = pm.get("BC_mod").and_then(|s| s.parse::<i32>().ok());
    let bcfw_ppm = pm.get("BCFW").and_then(|s| s.parse::<f64>().ok());

    StoredDimProc {
        p0_deg,
        p1_deg,
        window,
        si,
        td_eff,
        fcor,
        reverse,
        bc_mod,
        bcfw_ppm,
    }
}

fn bruker_proc_first_ppm(
    proc_map: Option<&HashMap<String, String>>,
    sfo1: f64,
    obs_mhz: f64,
    sw_hz: f64,
) -> f64 {
    if let Some(p) = proc_map {
        let sref = p
            .get("SREF_mod")
            .and_then(|v| v.parse::<i32>().ok())
            .unwrap_or(1);
        if sref != 0 {
            if let Some(offset) = p.get("OFFSET").and_then(|v| v.parse::<f64>().ok()) {
                let si = p
                    .get("SI")
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(0);
                let ftsize = p
                    .get("FTSIZE")
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(0);
                let stsr = p
                    .get("STSR")
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(0);
                if ftsize > si && stsr > 0 && obs_mhz > 0.0 {
                    return offset + (stsr as f64) * (sw_hz / obs_mhz) / (ftsize as f64);
                }
                return offset;
            }
        }
    }
    ((sfo1 - obs_mhz) * 1.0e6 + 0.5 * sw_hz) / obs_mhz
}

fn normalize_nucleus_label(raw: &str, fallback: &str) -> String {
    let up = raw
        .trim()
        .trim_matches('<')
        .trim_matches('>')
        .to_ascii_uppercase();
    match up.as_str() {
        "" | "OFF" | "NONE" => fallback.to_string(),
        "H1" | "1H" | "HN" => "1H".to_string(),
        "N15" | "15N" => "15N".to_string(),
        "C13" | "13C" => "13C".to_string(),
        other => other.to_string(),
    }
}

fn validate_bruker_fntype(
    dir: &Path,
    acqus: &HashMap<String, String>,
    indirect_maps: &[&HashMap<String, String>],
) -> Result<bool, String> {
    let mut is_fntype_nus = false;
    for (idx, m) in std::iter::once(&acqus)
        .chain(indirect_maps.iter())
        .enumerate()
    {
        if let Some(ft) = m.get("FnTYPE").and_then(|s| s.parse::<i32>().ok()) {
            match ft {
                0 => {}
                2 => {
                    if idx == 0 && indirect_maps.is_empty() {
                        return Err(format!(
                            "invalid Bruker 1D dataset in {}: FnTYPE=2 (NUS) is not valid for 1D FID",
                            dir.display()
                        ));
                    }
                    is_fntype_nus = true;
                }
                other => {
                    return Err(format!(
                        "unsupported Bruker non-traditional sampling mode in {}: FnTYPE={other} (expected 0=uniform or 2=NUS)",
                        dir.display()
                    ));
                }
            }
        }
    }
    Ok(is_fntype_nus)
}

fn parse_td_indirect_entry(acqus: &HashMap<String, String>, idx: usize) -> Option<usize> {
    acqus
        .get("TD_INDIRECT")
        .and_then(|s| s.split_whitespace().nth(idx))
        .and_then(|tok| tok.parse::<usize>().ok())
        .filter(|&v| v > 0)
}

fn resolve_bruker_nus_full_td(
    acqu_ind: &HashMap<String, String>,
    acqus: &HashMap<String, String>,
    td_indirect_idx: usize,
    unflattened_td_fallback: Option<usize>,
    dim_label: &str,
    dir: &Path,
) -> Result<usize, String> {
    if let Some(nus_td) = acqu_ind
        .get("NusTD")
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v > 0)
    {
        return Ok(nus_td);
    }
    if let Some(td_ind) = parse_td_indirect_entry(acqus, td_indirect_idx) {
        return Ok(td_ind);
    }
    if let Some(fb) = unflattened_td_fallback.filter(|&v| v > 1) {
        return Ok(fb);
    }
    Err(format!(
        "Bruker NUS dataset in {} is missing NusTD (and TD_INDIRECT[{td_indirect_idx}]) full-grid size on {dim_label}",
        dir.display()
    ))
}

fn parse_bruker_direct_axis(
    acqus: &HashMap<String, String>,
    procs: Option<&HashMap<String, String>>,
) -> Result<(FidAxisMeta, i32, i32), String> {
    let td: usize = acqus
        .get("TD")
        .ok_or("missing ##$TD in Bruker acqus")?
        .parse()
        .map_err(|e| format!("invalid TD in acqus: {e}"))?;
    if td < 2 {
        return Err(format!("invalid Bruker direct TD={td}"));
    }
    let bytorda: i32 = acqus
        .get("BYTORDA")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let dtypa: i32 = acqus.get("DTYPA").and_then(|v| v.parse().ok()).unwrap_or(0);
    if !matches!(dtypa, 0..=2) {
        return Err(format!(
            "unsupported Bruker DTYPA={dtypa} (expected 0=int32, 1=float32, 2=float64)"
        ));
    }

    let aq_mod: i32 = acqus
        .get("AQ_mod")
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    let quad_mode = match aq_mod {
        1 => QuadMode::Qf,
        3 | 4 => QuadMode::Dqd,
        other => {
            return Err(format!(
                "unsupported Bruker direct quadrature mode AQ_mod={other} (expected 1=QF, 3=DQD, 4=QSIM)"
            ));
        }
    };

    let sfo1: f64 = acqus
        .get("SFO1")
        .and_then(|v| v.parse().ok())
        .or_else(|| acqus.get("BF1").and_then(|v| v.parse().ok()))
        .ok_or("missing ##$SFO1 / ##$BF1 in Bruker acqus")?;
    let bf1: f64 = acqus
        .get("BF1")
        .and_then(|v| v.parse().ok())
        .unwrap_or(sfo1);
    let sw_ppm_acq: Option<f64> = acqus
        .get("SW")
        .and_then(|v| v.parse().ok())
        .filter(|&v| v > 0.0);
    let sw_hz: f64 = acqus
        .get("SW_h")
        .and_then(|v| v.parse().ok())
        .filter(|&v| v > 0.0)
        .or_else(|| sw_ppm_acq.map(|sw| sw * sfo1))
        .ok_or("missing ##$SW_h / ##$SW in Bruker acqus")?;

    let proc_sf = procs
        .and_then(|p| p.get("SF"))
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|&v| v > 0.0);
    let obs_mhz = proc_sf.unwrap_or(bf1);
    let first_ppm = bruker_proc_first_ppm(procs, sfo1, obs_mhz, sw_hz);

    let grpdly_raw = acqus
        .get("GRPDLY")
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(-1.0);
    let dspfvs = acqus
        .get("DSPFVS")
        .and_then(|v| v.parse::<i32>().ok())
        .unwrap_or(0);
    let decim = acqus
        .get("DECIM")
        .and_then(|v| v.parse::<i32>().ok())
        .unwrap_or(1);

    let pknl_enabled = procs
        .and_then(|p| p.get("PKNL"))
        .map(|v| !v.eq_ignore_ascii_case("no") && v != "0")
        .unwrap_or(true);
    let group_delay = if !pknl_enabled {
        0.0
    } else if grpdly_raw > 0.0 {
        grpdly_raw
    } else {
        bruker_dsp_group_delay(dspfvs, decim)?
    };

    let raw_nuc = acqus
        .get("NUC1")
        .map(String::as_str)
        .or_else(|| procs.and_then(|p| p.get("AXNUC").map(String::as_str)))
        .unwrap_or("1H");
    let label = normalize_nucleus_label(raw_nuc, "1H");

    let mut stored_proc = extract_bruker_stored_proc(procs);
    if let Some(ref mut td_eff) = stored_proc.td_eff {
        // Convert TopSpin scalar word count TDeff to complex points
        *td_eff = (*td_eff / 2).min(td / 2).max(1);
    }

    Ok((
        FidAxisMeta {
            label,
            td_raw: td,
            complex_points: td / 2,
            sw_hz,
            obs_mhz,
            carrier_mhz: sfo1,
            first_ppm,
            quad_mode,
            group_delay,
            conjugate_for_fft: true,
            stored_proc,
        },
        bytorda,
        dtypa,
    ))
}

fn parse_bruker_indirect_axis(
    acqu_ind: &HashMap<String, String>,
    proc_ind: Option<&HashMap<String, String>>,
    fallback_label: &str,
    td_override: Option<usize>,
) -> Result<FidAxisMeta, String> {
    let td = match td_override {
        Some(v) => v,
        None => acqu_ind
            .get("TD")
            .ok_or_else(|| format!("missing ##$TD in Bruker indirect ({fallback_label}) acqus"))?
            .parse()
            .map_err(|e| format!("invalid indirect TD: {e}"))?,
    };
    if td < 1 {
        return Err(format!("invalid Bruker indirect TD={td}"));
    }

    let fnmode: i32 = acqu_ind
        .get("FnMODE")
        .ok_or_else(|| format!("missing ##$FnMODE in Bruker indirect ({fallback_label}) acqus"))?
        .parse()
        .map_err(|e| format!("invalid FnMODE: {e}"))?;

    let quad_mode = match fnmode {
        1 => QuadMode::Qf,
        4 => QuadMode::States,
        5 => QuadMode::StatesTppi,
        6 => QuadMode::EchoAntiecho,
        other => {
            return Err(format!(
                "unsupported Bruker indirect FnMODE={other} on {fallback_label} (expected 1=QF, 4=States, 5=States-TPPI, 6=Echo-Antiecho; never guessing)"
            ));
        }
    };

    if quad_mode != QuadMode::Qf && td < 2 {
        return Err(format!(
            "Bruker indirect {fallback_label} has hypercomplex FnMODE={fnmode} but TD={td} < 2"
        ));
    }

    let sfo1: f64 = acqu_ind
        .get("SFO1")
        .and_then(|v| v.parse().ok())
        .or_else(|| acqu_ind.get("BF1").and_then(|v| v.parse().ok()))
        .ok_or_else(|| format!("missing ##$SFO1 / ##$BF1 in Bruker indirect ({fallback_label})"))?;
    let bf1: f64 = acqu_ind
        .get("BF1")
        .and_then(|v| v.parse().ok())
        .unwrap_or(sfo1);

    // In TopSpin 2.x/3.x/4.x indirect acqu2s/acqu3s, SW (ppm) * SFO1 (MHz) is authoritative
    // because SW_h may retain a default 1D template value or be omitted.
    let sw_ppm_acq = acqu_ind
        .get("SW")
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|&v| v > 0.0);
    let sw_hz = sw_ppm_acq
        .map(|sw| sw * sfo1)
        .or_else(|| {
            acqu_ind
                .get("SW_h")
                .and_then(|v| v.parse::<f64>().ok())
                .filter(|&v| v > 0.0)
        })
        .ok_or_else(|| format!("missing ##$SW / ##$SW_h in Bruker indirect ({fallback_label})"))?;

    let proc_sf = proc_ind
        .and_then(|p| p.get("SF"))
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|&v| v > 0.0);
    let obs_mhz = proc_sf.unwrap_or(bf1);
    let first_ppm = bruker_proc_first_ppm(proc_ind, sfo1, obs_mhz, sw_hz);

    let raw_nuc = acqu_ind
        .get("NUC1")
        .map(String::as_str)
        .or_else(|| proc_ind.and_then(|p| p.get("AXNUC").map(String::as_str)))
        .unwrap_or(fallback_label);
    let label = normalize_nucleus_label(raw_nuc, fallback_label);

    let complex_points = if quad_mode == QuadMode::Qf {
        td
    } else {
        td / 2
    };
    let mut stored_proc = extract_bruker_stored_proc(proc_ind);
    if let Some(ref mut td_eff) = stored_proc.td_eff {
        *td_eff = if quad_mode == QuadMode::Qf {
            (*td_eff).min(complex_points).max(1)
        } else {
            (*td_eff / 2).min(complex_points).max(1)
        };
    }

    Ok(FidAxisMeta {
        label,
        td_raw: td,
        complex_points,
        sw_hz,
        obs_mhz,
        carrier_mhz: sfo1,
        first_ppm,
        quad_mode,
        group_delay: 0.0,
        conjugate_for_fft: true,
        stored_proc,
    })
}

fn decode_bruker_trace(
    slice: &[u8],
    td_words: usize,
    big_endian: bool,
    dtypa: i32,
) -> Vec<Complex64> {
    let n_cplx = td_words / 2;
    let mut out = Vec::with_capacity(n_cplx);
    match dtypa {
        2 => {
            for k in 0..n_cplx {
                let b0 = k * 16;
                let r_arr: [u8; 8] = slice[b0..b0 + 8].try_into().unwrap();
                let i_arr: [u8; 8] = slice[b0 + 8..b0 + 16].try_into().unwrap();
                let re = if big_endian {
                    f64::from_be_bytes(r_arr)
                } else {
                    f64::from_le_bytes(r_arr)
                };
                let im = if big_endian {
                    f64::from_be_bytes(i_arr)
                } else {
                    f64::from_le_bytes(i_arr)
                };
                out.push([re, im]);
            }
        }
        1 => {
            for k in 0..n_cplx {
                let b0 = k * 8;
                let r_arr: [u8; 4] = slice[b0..b0 + 4].try_into().unwrap();
                let i_arr: [u8; 4] = slice[b0 + 4..b0 + 8].try_into().unwrap();
                let re = if big_endian {
                    f32::from_be_bytes(r_arr) as f64
                } else {
                    f32::from_le_bytes(r_arr) as f64
                };
                let im = if big_endian {
                    f32::from_be_bytes(i_arr) as f64
                } else {
                    f32::from_le_bytes(i_arr) as f64
                };
                out.push([re, im]);
            }
        }
        _ => {
            for k in 0..n_cplx {
                let b0 = k * 8;
                let r_arr: [u8; 4] = slice[b0..b0 + 4].try_into().unwrap();
                let i_arr: [u8; 4] = slice[b0 + 4..b0 + 8].try_into().unwrap();
                let re = if big_endian {
                    i32::from_be_bytes(r_arr) as f64
                } else {
                    i32::from_le_bytes(r_arr) as f64
                };
                let im = if big_endian {
                    i32::from_be_bytes(i_arr) as f64
                } else {
                    i32::from_le_bytes(i_arr) as f64
                };
                out.push([re, im]);
            }
        }
    }
    out
}

fn determine_bruker_ser_stride(
    file_len: usize,
    n_traces: usize,
    td_words: usize,
    word_bytes: usize,
    allow_trailing_blocks: bool,
    dir: &Path,
) -> Result<usize, String> {
    let raw_trace_bytes = td_words * word_bytes;
    let stride_1024 = raw_trace_bytes.div_ceil(1024) * 1024;
    let stride_256w = td_words.div_ceil(256) * 256 * word_bytes;

    if file_len == n_traces * stride_1024 {
        Ok(stride_1024)
    } else if file_len == n_traces * stride_256w {
        Ok(stride_256w)
    } else if allow_trailing_blocks
        && file_len > n_traces * stride_1024
        && file_len.is_multiple_of(stride_1024)
    {
        Ok(stride_1024)
    } else if file_len < n_traces * raw_trace_bytes {
        Err(format!(
            "Bruker ser in {} is smaller than expected trace count ({file_len} B < {n_traces} traces x {stride_1024} B)",
            dir.display()
        ))
    } else {
        Err(format!(
            "Bruker ser size mismatch in {}: got {file_len} B for {n_traces} traces (expected {} B with 1024-B block stride or {} B with 256-word block stride)",
            dir.display(),
            n_traces * stride_1024,
            n_traces * stride_256w
        ))
    }
}

/// Recombine an indirect pair of complex direct FIDs `(z0, z1)` at grid coordinate `k`
/// into uniform `States` `(z_cos, z_sin)` traces.
///
/// Algebra (under Bruker conjugate direct-FT convention `z_conj = re - i im`, `W = FT(z_conj) * ph`):
/// - `States`: `z_cos = z0`, `z_sin = z1`.
/// - `StatesTppi`: `z_cos = (-1)^k * z0`, `z_sin = (-1)^k * z1` (undoes $180^\circ$ acquisition alternation at grid coordinate `k`).
/// - `EchoAntiecho`: `z_cos_conj = -i * (z0_conj + z1_conj)`, `z_sin_conj = -(z0_conj - z1_conj)`
///   (`out_cos = [-(im0 + im1), re0 + re1]`, `out_sin = [re1 - re0, im1 - im0]`), matching Bruker TopSpin
///   `MC2 = 5` / Rance-Kay sensitivity enhancement (`Re(FT(z_cos)*ph)` and `Re(FT(z_sin)*ph)` in pure absorption
///   at the stored `procs.PHC0` phase with correct P/N frequency discrimination).
#[inline]
fn recombine_quad_pair(
    mode: QuadMode,
    k: usize,
    z0: &[Complex64],
    z1: &[Complex64],
    out_cos: &mut [Complex64],
    out_sin: &mut [Complex64],
) {
    match mode {
        QuadMode::States | QuadMode::Dqd => {
            out_cos.copy_from_slice(z0);
            out_sin.copy_from_slice(z1);
        }
        QuadMode::StatesTppi => {
            let sign = if k % 2 == 1 { -1.0 } else { 1.0 };
            for i in 0..z0.len() {
                out_cos[i] = [sign * z0[i][0], sign * z0[i][1]];
                out_sin[i] = [sign * z1[i][0], sign * z1[i][1]];
            }
        }
        QuadMode::EchoAntiecho => {
            for i in 0..z0.len() {
                let (re0, im0) = (z0[i][0], z0[i][1]);
                let (re1, im1) = (z1[i][0], z1[i][1]);
                out_cos[i] = [-(im0 + im1), re0 + re1];
                out_sin[i] = [re1 - re0, im1 - im0];
            }
        }
        QuadMode::Qf => {
            out_cos.copy_from_slice(z0);
            for v in out_sin.iter_mut() {
                *v = [0.0, 0.0];
            }
        }
    }
}

#[inline]
fn add_complex_slice(dst: &mut [Complex64], src: &[Complex64]) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        d[0] += s[0];
        d[1] += s[1];
    }
}

#[inline]
fn scale_complex_slice(dst: &mut [Complex64], scale: f64) {
    for d in dst.iter_mut() {
        d[0] *= scale;
        d[1] *= scale;
    }
}

/// Read a Bruker 1D `fid` directory (`acqus` + `fid`).
pub fn read_bruker_fid_1d(dir: impl AsRef<Path>) -> Result<Fid1D, String> {
    let dir = resolve_fid_dir(dir.as_ref())?;
    let acqus = read_merged_bruker_params(&dir, "acqus", "acqu")?;
    validate_bruker_fntype(&dir, &acqus, &[])?;
    let procs = read_optional_bruker_proc(&dir, "procs");
    let (axis_x, bytorda, dtypa) = parse_bruker_direct_axis(&acqus, procs.as_ref())?;

    let fid_path = find_file_ci(&dir, "fid")
        .ok_or_else(|| format!("missing 1D fid binary file in {}", dir.display()))?;
    let bytes =
        fs::read(&fid_path).map_err(|e| format!("failed to read {}: {e}", fid_path.display()))?;
    let word_bytes = if dtypa == 2 { 8 } else { 4 };
    let needed = (axis_x.complex_points * 2) * word_bytes;
    if bytes.len() < needed {
        return Err(format!(
            "Bruker 1D fid truncated: need {needed} bytes, got {}",
            bytes.len()
        ));
    }

    let data = decode_bruker_trace(
        &bytes[..needed],
        axis_x.complex_points * 2,
        bytorda != 0,
        dtypa,
    );
    let pulse_program = acqus.get("PULPROG").cloned().unwrap_or_default();
    let name = dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("fid")
        .to_string();

    Ok(Fid1D {
        name,
        vendor: FidVendor::Bruker,
        pulse_program,
        axis_x,
        data,
    })
}

fn read_optional_bruker_pulseprogram(dir: &Path) -> Option<String> {
    find_file_ci(dir, "pulseprogram")
        .or_else(|| find_file_ci(dir, "pulseprogram.precomp"))
        .and_then(|p| fs::read_to_string(p).ok())
}

fn contains_word_token(line: &str, word: &str) -> bool {
    let bytes = line.as_bytes();
    let wb = word.as_bytes();
    if wb.is_empty() || bytes.len() < wb.len() {
        return false;
    }
    let is_ident = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    let mut pos = 0usize;
    while pos + wb.len() <= bytes.len() {
        if let Some(rel) = line[pos..].find(word) {
            let start = pos + rel;
            let end = start + wb.len();
            let left_ok = start == 0 || !is_ident(bytes[start - 1]);
            let right_ok = end == bytes.len() || !is_ident(bytes[end]);
            if left_ok && right_ok {
                return true;
            }
            pos = start + 1;
        } else {
            break;
        }
    }
    false
}

/// Extract `(reg_num, step_sign)` from `ip<k>` (`+1`) or `dp<k>` (`-1`) tokens WITHOUT `*`
/// (`ip<m>*2` / `dp<m>*2` are `180 deg` axial-peak toggles rather than `90 deg` States quadrature).
fn extract_90deg_ip_dp_token(line: &str) -> Option<(usize, i32)> {
    for raw_tok in line.split(|c: char| {
        c.is_whitespace() || matches!(c, ',' | '(' | ')' | ';' | ':' | '+' | '-' | '=')
    }) {
        if raw_tok.contains('*') {
            continue;
        }
        if let Some(num_str) = raw_tok.strip_prefix("ip") {
            if !num_str.is_empty() && num_str.bytes().all(|b| b.is_ascii_digit()) {
                if let Ok(reg) = num_str.parse::<usize>() {
                    return Some((reg, 1));
                }
            }
        } else if let Some(num_str) = raw_tok.strip_prefix("dp") {
            if !num_str.is_empty() && num_str.bytes().all(|b| b.is_ascii_digit()) {
                if let Ok(reg) = num_str.parse::<usize>() {
                    return Some((reg, -1));
                }
            }
        }
    }
    None
}

/// Extract `(reg_num, step_sign)` from TopSpin `F1PH(calph(ph<k>, +90), ...)` or `F2PH(...)`.
fn extract_calph_90deg_macro(text: &str, macro_prefix: &str) -> Option<(usize, i32)> {
    let idx = text.find(macro_prefix)?;
    let tail = &text[idx + macro_prefix.len()..];
    let cidx = tail.find("calph(")?;
    let inner = &tail[cidx + "calph(".len()..];
    let end = inner.find(')')?;
    let args = &inner[..end];
    let (ph_part, deg_part) = args.split_once(',')?;
    let reg = ph_part
        .trim()
        .strip_prefix("ph")?
        .trim()
        .parse::<usize>()
        .ok()?;
    let deg: i32 = deg_part.trim().trim_start_matches('+').parse().ok()?;
    match deg.rem_euclid(360) {
        90 => Some((reg, 1)),
        270 => Some((reg, -1)),
        _ => None,
    }
}

/// Infer Bruker indirect `##$REVERSE` from spin-physics phase incrementation in `pulseprogram`
/// when `pdata/1/proc2s` or `proc3s` is absent (States, Haberkorn & Ruben *J. Magn. Reson.* **1982**,
/// 48, 286–292; Marion et al. *J. Magn. Reson.* **1989**, 85, 393–399):
/// - `f_dim == 1`: Bruker `F1` (`acqu2s` in 2D, `acqu3s` in 3D; evolution delay `d0`).
/// - `f_dim == 2`: Bruker `F2` (`acqu2s` in 3D; evolution delay `d10`).
pub fn infer_bruker_raw_indirect_reverse(
    pp_text: Option<&str>,
    pulprog: &str,
    f_dim: usize,
    quad_mode: QuadMode,
) -> bool {
    let rev_tag = if f_dim == 1 {
        "f1 reverse:"
    } else {
        "f2 reverse:"
    };
    if let Some(txt) = pp_text {
        for raw_line in txt.lines() {
            if let Some((_, cmt)) = raw_line.split_once(';') {
                let low = cmt.trim().to_ascii_lowercase();
                if let Some(rest) = low.strip_prefix(rev_tag) {
                    let val = rest.trim();
                    if val.starts_with("true") || val.starts_with("yes") {
                        return true;
                    }
                    if val.starts_with("false") || val.starts_with("no") {
                        return false;
                    }
                }
            }
        }
    }

    if !matches!(quad_mode, QuadMode::States | QuadMode::StatesTppi) {
        return false;
    }

    if let Some(txt) = pp_text {
        let mut rf_lines: Vec<&str> = Vec::new();
        for raw_line in txt.lines() {
            let code = raw_line.split(';').next().unwrap_or("").trim();
            if code.is_empty() || code.starts_with('#') || code.starts_with('"') {
                continue;
            }
            if let Some(rest) = code.strip_prefix("ph") {
                let digits_len = rest.bytes().take_while(|b| b.is_ascii_digit()).count();
                if digits_len > 0 && rest[digits_len..].trim_start().starts_with('=') {
                    break;
                }
            }
            rf_lines.push(code);
        }

        let macro_name = if f_dim == 1 { "F1PH" } else { "F2PH" };
        let inc_tok = if f_dim == 1 { "id0" } else { "id10" };
        let precomp_loop = if f_dim == 1 { "t1loop" } else { "t2loop" };

        let quad_reg_sign = extract_calph_90deg_macro(txt, macro_name)
            .or_else(|| {
                // Precompiled TopSpin `d0 = d0orig + t1loop * in0` followed by `MCWRK ip<k> + phval<k>`
                let lines: Vec<&str> = txt.lines().collect();
                for (idx, line) in lines.iter().enumerate() {
                    if line.contains(precomp_loop) {
                        for follow in lines.iter().skip(idx + 1).take(6) {
                            let code = follow.split(';').next().unwrap_or("");
                            if let Some(found) = extract_90deg_ip_dp_token(code) {
                                return Some(found);
                            }
                        }
                    }
                }
                None
            })
            .or_else(|| {
                // Classic Avance loop: 2-step quadrature `ip<k>` / `dp<k>` appears immediately before `id0` / `id10`
                for (idx, line) in rf_lines.iter().enumerate() {
                    if contains_word_token(line, inc_tok) {
                        for prev in (0..idx).rev().take(4) {
                            if contains_word_token(
                                rf_lines[prev],
                                if f_dim == 1 { "id10" } else { "id0" },
                            ) {
                                break;
                            }
                            if let Some(found) = extract_90deg_ip_dp_token(rf_lines[prev]) {
                                return Some(found);
                            }
                        }
                    }
                }
                None
            });

        if let Some((reg, step_sign)) = quad_reg_sign {
            let ph_tok = format!("ph{reg}");
            let evo_tok = if f_dim == 1 { "d0" } else { "d10" };
            let dec_tok = if f_dim == 1 { "d20" } else { "d30" };

            let ph_line = rf_lines
                .iter()
                .position(|l| !l.contains("calph") && contains_word_token(l, &ph_tok));
            let evo_line = rf_lines.iter().position(|l| {
                !l.starts_with("define ")
                    && !l.contains("caldel")
                    && contains_word_token(l, evo_tok)
            });

            if let (Some(p_idx), Some(e_idx)) = (ph_line, evo_line) {
                let mut net_sign = step_sign;
                if p_idx > e_idx {
                    // Quadrature readout pulse occurs AFTER evolution `t_1`/`t_2`: `cos(omega t + phi)`
                    net_sign = -net_sign;
                } else {
                    // Quadrature excitation pulse occurs BEFORE evolution: check whether a decremented
                    // semi-constant-time delay (`d30` / `d20`) and `180 deg` refocusing pulse precede `d10`/`d0`
                    let has_dec_before_evo = rf_lines[p_idx + 1..e_idx]
                        .iter()
                        .any(|l| contains_word_token(l, dec_tok));
                    if has_dec_before_evo {
                        net_sign = -net_sign;
                    }
                }
                return net_sign < 0;
            }
        }
    }

    f_dim == 2 && pulprog.to_ascii_lowercase().contains("gpwg3d")
}

/// Read a Bruker 2D `ser` directory (`acqus`, `acqu2s`, and `ser`), automatically detecting uniform vs NUS.
pub fn read_bruker_fid_2d(dir: impl AsRef<Path>) -> Result<Fid2D, String> {
    let dir = resolve_fid_dir(dir.as_ref())?;
    let acqus = read_merged_bruker_params(&dir, "acqus", "acqu")?;
    let acqu2s = read_merged_bruker_params(&dir, "acqu2s", "acqu2")?;
    let is_fntype_nus = validate_bruker_fntype(&dir, &acqus, &[&acqu2s])?;

    let procs = read_optional_bruker_proc(&dir, "procs");
    let proc2s = read_optional_bruker_proc(&dir, "proc2s");

    let (axis_x, bytorda, dtypa) = parse_bruker_direct_axis(&acqus, procs.as_ref())?;

    let ser_path = find_file_ci(&dir, "ser")
        .ok_or_else(|| format!("missing 2D ser binary file in {}", dir.display()))?;
    let bytes =
        fs::read(&ser_path).map_err(|e| format!("failed to read {}: {e}", ser_path.display()))?;
    let word_bytes = if dtypa == 2 { 8 } else { 4 };
    let raw_stride_1024 = (axis_x.td_raw * word_bytes).div_ceil(1024) * 1024;

    let td_acq_y: usize = acqu2s
        .get("TD")
        .ok_or("missing ##$TD in Bruker acqu2s")?
        .parse()
        .map_err(|e| format!("invalid TD in acqu2s: {e}"))?;

    let sched_file = find_bruker_nuslist_file(&dir, &acqus);
    let nus_td_opt = acqu2s
        .get("NusTD")
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .or_else(|| parse_td_indirect_entry(&acqus, 1));

    // TopSpin 4 uniform guard: FnTYPE == 0 with ser matching td_acq_y (and NusTD <= td_acq_y when no smaller nuslist) is uniform.
    let implicit_nus = !is_fntype_nus
        && sched_file.is_some()
        && (nus_td_opt.is_some_and(|ntd| ntd > td_acq_y)
            || bytes.len() < td_acq_y * raw_stride_1024);
    let is_nus = is_fntype_nus || implicit_nus;

    let td_full_y = if is_nus {
        let unflat = (bytes.len() < td_acq_y * raw_stride_1024).then_some(td_acq_y);
        resolve_bruker_nus_full_td(&acqu2s, &acqus, 1, unflat, "F1 (acqu2s)", &dir)?
    } else {
        td_acq_y
    };

    let mut axis_y = parse_bruker_indirect_axis(&acqu2s, proc2s.as_ref(), "15N", Some(td_full_y))?;
    if proc2s.as_ref().and_then(|p| p.get("REVERSE")).is_none() {
        let pp_txt = read_optional_bruker_pulseprogram(&dir);
        let pulprog = acqus.get("PULPROG").map(String::as_str).unwrap_or("");
        axis_y.stored_proc.reverse =
            infer_bruker_raw_indirect_reverse(pp_txt.as_deref(), pulprog, 1, axis_y.quad_mode);
    }
    let q1_mult = if axis_y.quad_mode == QuadMode::Qf {
        1
    } else {
        2
    };
    let n0 = axis_x.complex_points;
    let n1 = axis_y.complex_points;
    let big_endian = bytorda != 0;
    let trace_byte_len = n0 * 2 * word_bytes;
    let mut data = vec![[0.0, 0.0]; n1 * 2 * n0];

    let nus_mask = if is_nus {
        let sched_path = sched_file.ok_or_else(|| {
            format!(
                "Bruker 2D NUS dataset in {} is missing a non-empty nuslist schedule file",
                dir.display()
            )
        })?;
        let coords = parse_nus_schedule_1d(&sched_path, n1)?;
        let n_acq_traces = coords.len() * q1_mult;
        if td_acq_y != n_acq_traces && td_acq_y != td_full_y {
            return Err(format!(
                "Bruker 2D NUS trace count mismatch in {}: acqu2s.TD={td_acq_y}, but nuslist has {} points ({n_acq_traces} traces)",
                dir.display(),
                coords.len()
            ));
        }
        let stride = determine_bruker_ser_stride(
            bytes.len(),
            n_acq_traces,
            axis_x.td_raw,
            word_bytes,
            false,
            &dir,
        )?;
        let mut counts = vec![0usize; n1];
        let mut tmp_cos = vec![[0.0, 0.0]; n0];
        let mut tmp_sin = vec![[0.0, 0.0]; n0];

        for (row_idx, &k1) in coords.iter().enumerate() {
            if axis_y.quad_mode == QuadMode::Qf {
                let off = row_idx * stride;
                let tr = decode_bruker_trace(
                    &bytes[off..off + trace_byte_len],
                    n0 * 2,
                    big_endian,
                    dtypa,
                );
                add_complex_slice(&mut data[(k1 * 2) * n0..(k1 * 2 + 1) * n0], &tr);
            } else {
                let off0 = (2 * row_idx) * stride;
                let off1 = (2 * row_idx + 1) * stride;
                let z0 = decode_bruker_trace(
                    &bytes[off0..off0 + trace_byte_len],
                    n0 * 2,
                    big_endian,
                    dtypa,
                );
                let z1 = decode_bruker_trace(
                    &bytes[off1..off1 + trace_byte_len],
                    n0 * 2,
                    big_endian,
                    dtypa,
                );
                recombine_quad_pair(axis_y.quad_mode, k1, &z0, &z1, &mut tmp_cos, &mut tmp_sin);
                add_complex_slice(&mut data[(k1 * 2) * n0..(k1 * 2 + 1) * n0], &tmp_cos);
                add_complex_slice(&mut data[(k1 * 2 + 1) * n0..(k1 * 2 + 2) * n0], &tmp_sin);
            }
            counts[k1] += 1;
        }
        for (k1, &cnt) in counts.iter().enumerate() {
            if cnt > 1 {
                scale_complex_slice(
                    &mut data[(k1 * 2) * n0..(k1 * 2 + 2) * n0],
                    1.0 / (cnt as f64),
                );
            }
        }
        Some(counts.into_iter().map(|c| c > 0).collect())
    } else {
        let stride = determine_bruker_ser_stride(
            bytes.len(),
            td_acq_y,
            axis_x.td_raw,
            word_bytes,
            true,
            &dir,
        )?;
        if axis_y.quad_mode == QuadMode::Qf {
            for k1 in 0..n1 {
                let off = k1 * stride;
                let tr = decode_bruker_trace(
                    &bytes[off..off + trace_byte_len],
                    n0 * 2,
                    big_endian,
                    dtypa,
                );
                data[(k1 * 2) * n0..(k1 * 2 + 1) * n0].copy_from_slice(&tr);
            }
        } else {
            for k1 in 0..n1 {
                let off0 = (2 * k1) * stride;
                let off1 = (2 * k1 + 1) * stride;
                let z0 = decode_bruker_trace(
                    &bytes[off0..off0 + trace_byte_len],
                    n0 * 2,
                    big_endian,
                    dtypa,
                );
                let z1 = decode_bruker_trace(
                    &bytes[off1..off1 + trace_byte_len],
                    n0 * 2,
                    big_endian,
                    dtypa,
                );
                let (cos_slice, sin_slice) =
                    data[(k1 * 2) * n0..(k1 * 2 + 2) * n0].split_at_mut(n0);
                recombine_quad_pair(axis_y.quad_mode, k1, &z0, &z1, cos_slice, sin_slice);
            }
        }
        None
    };

    let pulse_program = acqus.get("PULPROG").cloned().unwrap_or_default();
    let name = dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("ser")
        .to_string();

    Ok(Fid2D {
        name,
        vendor: FidVendor::Bruker,
        pulse_program,
        axis_x,
        axis_y,
        data,
        nus_mask,
    })
}

/// Recombine the `q1_mult * q2_mult` indirect traces for a single `(k1, k2)` 3D grid point
/// and accumulate into `data` at `(k2, k1)`.
#[allow(clippy::too_many_arguments)]
fn accumulate_bruker_3d_point<F>(
    axis_y_qm: QuadMode,
    axis_z_qm: QuadMode,
    k1: usize,
    k2: usize,
    n0: usize,
    n1: usize,
    mut fetch_sub_trace: F,
    z00_cos: &mut [Complex64],
    z00_sin: &mut [Complex64],
    z01_cos: &mut [Complex64],
    z01_sin: &mut [Complex64],
    out_c2_cos: &mut [Complex64],
    out_c2_sin: &mut [Complex64],
    data: &mut [Complex64],
) where
    F: FnMut(usize, usize) -> Vec<Complex64>,
{
    if axis_y_qm == QuadMode::Qf {
        let t_e = fetch_sub_trace(0, 0);
        z00_cos.copy_from_slice(&t_e);
        z00_sin.fill([0.0, 0.0]);
        if axis_z_qm != QuadMode::Qf {
            let t_o = fetch_sub_trace(0, 1);
            z01_cos.copy_from_slice(&t_o);
            z01_sin.fill([0.0, 0.0]);
        }
    } else {
        let t00 = fetch_sub_trace(0, 0);
        let t10 = fetch_sub_trace(1, 0);
        recombine_quad_pair(axis_y_qm, k1, &t00, &t10, z00_cos, z00_sin);

        if axis_z_qm != QuadMode::Qf {
            let t01 = fetch_sub_trace(0, 1);
            let t11 = fetch_sub_trace(1, 1);
            recombine_quad_pair(axis_y_qm, k1, &t01, &t11, z01_cos, z01_sin);
        }
    }

    let base_c2_0_c1_0 = (((k2 * 2) * n1 + k1) * 2) * n0;
    let base_c2_1_c1_0 = (((k2 * 2 + 1) * n1 + k1) * 2) * n0;

    if axis_z_qm == QuadMode::Qf {
        add_complex_slice(&mut data[base_c2_0_c1_0..base_c2_0_c1_0 + n0], z00_cos);
        add_complex_slice(
            &mut data[base_c2_0_c1_0 + n0..base_c2_0_c1_0 + 2 * n0],
            z00_sin,
        );
    } else {
        recombine_quad_pair(axis_z_qm, k2, z00_cos, z01_cos, out_c2_cos, out_c2_sin);
        add_complex_slice(&mut data[base_c2_0_c1_0..base_c2_0_c1_0 + n0], out_c2_cos);
        add_complex_slice(&mut data[base_c2_1_c1_0..base_c2_1_c1_0 + n0], out_c2_sin);

        if axis_y_qm != QuadMode::Qf {
            recombine_quad_pair(axis_z_qm, k2, z00_sin, z01_sin, out_c2_cos, out_c2_sin);
            add_complex_slice(
                &mut data[base_c2_0_c1_0 + n0..base_c2_0_c1_0 + 2 * n0],
                out_c2_cos,
            );
            add_complex_slice(
                &mut data[base_c2_1_c1_0 + n0..base_c2_1_c1_0 + 2 * n0],
                out_c2_sin,
            );
        }
    }
}

/// Read a Bruker 3D `ser` directory (`acqus`, `acqu2s`, `acqu3s`, and `ser`), automatically detecting uniform vs NUS.
pub fn read_bruker_fid_3d(dir: impl AsRef<Path>) -> Result<Fid3D, String> {
    let dir = resolve_fid_dir(dir.as_ref())?;
    let acqus = read_merged_bruker_params(&dir, "acqus", "acqu")?;
    let acqu2s = read_merged_bruker_params(&dir, "acqu2s", "acqu2")?;
    let acqu3s = read_merged_bruker_params(&dir, "acqu3s", "acqu3")?;
    let is_fntype_nus = validate_bruker_fntype(&dir, &acqus, &[&acqu2s, &acqu3s])?;

    let procs = read_optional_bruker_proc(&dir, "procs");
    let proc2s = read_optional_bruker_proc(&dir, "proc2s");
    let proc3s = read_optional_bruker_proc(&dir, "proc3s");

    let (axis_x, bytorda, dtypa) = parse_bruker_direct_axis(&acqus, procs.as_ref())?;

    let aqseq: i32 = acqus.get("AQSEQ").and_then(|v| v.parse().ok()).unwrap_or(0);
    if !matches!(aqseq, 0 | 1) {
        return Err(format!(
            "unsupported Bruker 3D AQSEQ={aqseq} in {} (expected 0 for 3-2-1 or 1 for 3-1-2)",
            dir.display()
        ));
    }

    let ser_path = find_file_ci(&dir, "ser")
        .ok_or_else(|| format!("missing 3D ser binary file in {}", dir.display()))?;
    let bytes =
        fs::read(&ser_path).map_err(|e| format!("failed to read {}: {e}", ser_path.display()))?;
    let word_bytes = if dtypa == 2 { 8 } else { 4 };
    let raw_stride_1024 = (axis_x.td_raw * word_bytes).div_ceil(1024) * 1024;

    let td_acq_y: usize = acqu2s
        .get("TD")
        .ok_or("missing ##$TD in Bruker acqu2s")?
        .parse()
        .map_err(|e| format!("invalid TD in acqu2s: {e}"))?;
    let td_acq_z: usize = acqu3s
        .get("TD")
        .ok_or("missing ##$TD in Bruker acqu3s")?
        .parse()
        .map_err(|e| format!("invalid TD in acqu3s: {e}"))?;
    let total_acq_param = td_acq_y * td_acq_z;

    let sched_file = find_bruker_nuslist_file(&dir, &acqus);
    let nus_td_y_opt = acqu2s
        .get("NusTD")
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .or_else(|| parse_td_indirect_entry(&acqus, 2));
    let nus_td_z_opt = acqu3s
        .get("NusTD")
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .or_else(|| parse_td_indirect_entry(&acqus, 1));

    // TopSpin 4 uniform guard: FnTYPE == 0 (e.g. CISD3_HNCA with NUSLIST=<automatic>, NusAMOUNT=100,
    // TD=40x80, NusTD=40x80) is uniform when ser matches the uniform grid!
    let full_grid_est = nus_td_y_opt.unwrap_or(td_acq_y) * nus_td_z_opt.unwrap_or(td_acq_z);
    let implicit_nus = !is_fntype_nus
        && sched_file.is_some()
        && (full_grid_est > total_acq_param || bytes.len() < full_grid_est * raw_stride_1024);
    let is_nus = is_fntype_nus || implicit_nus;

    let (td_full_y, td_full_z) = if is_nus {
        let unflat = td_acq_y > 1 && td_acq_z > 1;
        (
            resolve_bruker_nus_full_td(
                &acqu2s,
                &acqus,
                2,
                unflat.then_some(td_acq_y),
                "F2 (acqu2s)",
                &dir,
            )?,
            resolve_bruker_nus_full_td(
                &acqu3s,
                &acqus,
                1,
                unflat.then_some(td_acq_z),
                "F1 (acqu3s)",
                &dir,
            )?,
        )
    } else {
        (td_acq_y, td_acq_z)
    };

    let mut axis_y = parse_bruker_indirect_axis(&acqu2s, proc2s.as_ref(), "15N", Some(td_full_y))?;
    let mut axis_z = parse_bruker_indirect_axis(&acqu3s, proc3s.as_ref(), "13C", Some(td_full_z))?;
    if proc2s.as_ref().and_then(|p| p.get("REVERSE")).is_none()
        || proc3s.as_ref().and_then(|p| p.get("REVERSE")).is_none()
    {
        let pp_txt = read_optional_bruker_pulseprogram(&dir);
        let pulprog = acqus.get("PULPROG").map(String::as_str).unwrap_or("");
        if proc2s.as_ref().and_then(|p| p.get("REVERSE")).is_none() {
            axis_y.stored_proc.reverse =
                infer_bruker_raw_indirect_reverse(pp_txt.as_deref(), pulprog, 2, axis_y.quad_mode);
        }
        if proc3s.as_ref().and_then(|p| p.get("REVERSE")).is_none() {
            axis_z.stored_proc.reverse =
                infer_bruker_raw_indirect_reverse(pp_txt.as_deref(), pulprog, 1, axis_z.quad_mode);
        }
    }

    let q1_mult = if axis_y.quad_mode == QuadMode::Qf {
        1
    } else {
        2
    };
    let q2_mult = if axis_z.quad_mode == QuadMode::Qf {
        1
    } else {
        2
    };

    let n0 = axis_x.complex_points;
    let n1 = axis_y.complex_points;
    let n2 = axis_z.complex_points;
    let big_endian = bytorda != 0;
    let trace_byte_len = n0 * 2 * word_bytes;

    let mut data = vec![[0.0, 0.0]; n2 * 2 * n1 * 2 * n0];
    let mut z00_cos = vec![[0.0, 0.0]; n0];
    let mut z00_sin = vec![[0.0, 0.0]; n0];
    let mut z01_cos = vec![[0.0, 0.0]; n0];
    let mut z01_sin = vec![[0.0, 0.0]; n0];
    let mut out_c2_cos = vec![[0.0, 0.0]; n0];
    let mut out_c2_sin = vec![[0.0, 0.0]; n0];

    let nus_mask = if is_nus {
        let sched_path = sched_file.ok_or_else(|| {
            format!(
                "Bruker 3D NUS dataset in {} is missing a non-empty nuslist schedule file",
                dir.display()
            )
        })?;
        // AQSEQ == 0 (3-2-1): col 0 is F2 (acqu2s), col 1 is F1 (acqu3s).
        // AQSEQ == 1 (3-1-2): col 0 is F1 (acqu3s), col 1 is F2 (acqu2s).
        let coords = parse_nus_schedule_2d(&sched_path, n1, n2, aqseq == 0)?;
        let traces_per_row = q1_mult * q2_mult;
        let expected_acq_traces = coords.len() * traces_per_row;
        if total_acq_param != expected_acq_traces && total_acq_param != td_full_y * td_full_z {
            return Err(format!(
                "Bruker 3D NUS trace count mismatch in {}: acqu2s.TD*acqu3s.TD={total_acq_param}, but nuslist has {} points ({expected_acq_traces} traces)",
                dir.display(),
                coords.len()
            ));
        }
        let stride = determine_bruker_ser_stride(
            bytes.len(),
            expected_acq_traces,
            axis_x.td_raw,
            word_bytes,
            false,
            &dir,
        )?;
        let mut counts = vec![0usize; n2 * n1];

        for (row_idx, &(k1, k2)) in coords.iter().enumerate() {
            let row_base = row_idx * traces_per_row;
            let fetch_sub = |iy_sub: usize, iz_sub: usize| -> Vec<Complex64> {
                let sub_idx = if aqseq == 0 {
                    // F2 (`iy_sub`) is inner / fastest, F1 (`iz_sub`) is outer / slowest
                    iz_sub * q1_mult + iy_sub
                } else {
                    // F1 (`iz_sub`) is inner / fastest, F2 (`iy_sub`) is outer / slowest
                    iy_sub * q2_mult + iz_sub
                };
                let off = (row_base + sub_idx) * stride;
                decode_bruker_trace(&bytes[off..off + trace_byte_len], n0 * 2, big_endian, dtypa)
            };
            accumulate_bruker_3d_point(
                axis_y.quad_mode,
                axis_z.quad_mode,
                k1,
                k2,
                n0,
                n1,
                fetch_sub,
                &mut z00_cos,
                &mut z00_sin,
                &mut z01_cos,
                &mut z01_sin,
                &mut out_c2_cos,
                &mut out_c2_sin,
                &mut data,
            );
            counts[k2 * n1 + k1] += 1;
        }

        for k2 in 0..n2 {
            for k1 in 0..n1 {
                let cnt = counts[k2 * n1 + k1];
                if cnt > 1 {
                    let inv = 1.0 / (cnt as f64);
                    for c2 in 0..2 {
                        let base = (((k2 * 2 + c2) * n1 + k1) * 2) * n0;
                        scale_complex_slice(&mut data[base..base + 2 * n0], inv);
                    }
                }
            }
        }
        Some(counts.into_iter().map(|c| c > 0).collect())
    } else {
        let stride = determine_bruker_ser_stride(
            bytes.len(),
            total_acq_param,
            axis_x.td_raw,
            word_bytes,
            true,
            &dir,
        )?;
        for k2 in 0..n2 {
            for k1 in 0..n1 {
                let fetch_sub = |iy_sub: usize, iz_sub: usize| -> Vec<Complex64> {
                    let iy = k1 * q1_mult + iy_sub;
                    let iz = k2 * q2_mult + iz_sub;
                    let tidx = if aqseq == 0 {
                        iz * td_full_y + iy
                    } else {
                        iy * td_full_z + iz
                    };
                    let off = tidx * stride;
                    decode_bruker_trace(
                        &bytes[off..off + trace_byte_len],
                        n0 * 2,
                        big_endian,
                        dtypa,
                    )
                };
                accumulate_bruker_3d_point(
                    axis_y.quad_mode,
                    axis_z.quad_mode,
                    k1,
                    k2,
                    n0,
                    n1,
                    fetch_sub,
                    &mut z00_cos,
                    &mut z00_sin,
                    &mut z01_cos,
                    &mut z01_sin,
                    &mut out_c2_cos,
                    &mut out_c2_sin,
                    &mut data,
                );
            }
        }
        None
    };

    let pulse_program = acqus.get("PULPROG").cloned().unwrap_or_default();
    let name = dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("ser")
        .to_string();

    Ok(Fid3D {
        name,
        vendor: FidVendor::Bruker,
        pulse_program,
        axis_x,
        axis_y,
        axis_z,
        data,
        nus_mask,
    })
}

// ============================================================================
// Varian / Agilent (`fid` + `procpar`) Reader
// ============================================================================

#[derive(Clone, Debug)]
enum ProcparVal {
    Numbers(Vec<f64>),
    Strings(Vec<String>),
}

impl ProcparVal {
    fn first_f64(&self) -> Option<f64> {
        match self {
            ProcparVal::Numbers(v) => v.first().copied(),
            ProcparVal::Strings(v) => v.first().and_then(|s| s.parse().ok()),
        }
    }

    fn first_str(&self) -> Option<&str> {
        match self {
            ProcparVal::Strings(v) => v.first().map(String::as_str),
            ProcparVal::Numbers(_) => None,
        }
    }

    fn numbers(&self) -> Option<&[f64]> {
        match self {
            ProcparVal::Numbers(v) => Some(v.as_slice()),
            ProcparVal::Strings(_) => None,
        }
    }
}

fn parse_procpar_str(text: &str) -> Result<HashMap<String, ProcparVal>, String> {
    let mut map = HashMap::new();
    let lines: Vec<&str> = text.lines().collect();
    let mut idx = 0usize;

    while idx < lines.len() {
        let hdr = lines[idx].trim();
        idx += 1;
        if hdr.is_empty() {
            continue;
        }
        let hdr_parts: Vec<&str> = hdr.split_whitespace().collect();
        if hdr_parts.len() < 11 {
            continue;
        }
        let name = hdr_parts[0].to_string();
        let basictype: i32 = match hdr_parts[2].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        if idx >= lines.len() {
            break;
        }

        if basictype == 2 {
            // String array: line starts with `<count> "str0"`, followed by `count - 1` lines of `"str_i"`
            let first_val_line = lines[idx].trim();
            idx += 1;
            let (count_str, rem) = first_val_line
                .split_once(char::is_whitespace)
                .unwrap_or((first_val_line, ""));
            let count: usize = count_str.parse().unwrap_or(1);
            let mut strs = Vec::with_capacity(count);
            if !rem.trim().is_empty() {
                strs.push(unquote_procpar(rem.trim()));
            }
            while strs.len() < count && idx < lines.len() {
                let sline = lines[idx].trim();
                idx += 1;
                strs.push(unquote_procpar(sline));
            }
            map.insert(name, ProcparVal::Strings(strs));
        } else {
            // Real/integer array (`basictype == 1` or `0`): `<count> v1 v2 ...` possibly wrapping lines
            let mut nums_raw = Vec::new();
            let first_tokens: Vec<&str> = lines[idx].split_whitespace().collect();
            idx += 1;
            if let Some(cnt_tok) = first_tokens.first() {
                let count: usize = cnt_tok.parse().unwrap_or(0);
                for &t in &first_tokens[1..] {
                    if let Ok(v) = t.parse::<f64>() {
                        nums_raw.push(v);
                    }
                }
                while nums_raw.len() < count && idx < lines.len() {
                    let more: Vec<&str> = lines[idx].split_whitespace().collect();
                    idx += 1;
                    if more.is_empty() {
                        break;
                    }
                    for &t in &more {
                        if let Ok(v) = t.parse::<f64>() {
                            nums_raw.push(v);
                        }
                    }
                }
            }
            map.insert(name, ProcparVal::Numbers(nums_raw));
        }

        // Skip trailing enumerated-values definition line (`0` or `<n_enum> "e1" "e2" ...`)
        if idx < lines.len() {
            idx += 1;
        }
    }

    Ok(map)
}

fn unquote_procpar(s: &str) -> String {
    let t = s.trim();
    if t.len() >= 2
        && ((t.starts_with('"') && t.ends_with('"')) || (t.starts_with('\'') && t.ends_with('\'')))
    {
        t[1..t.len() - 1].to_string()
    } else {
        t.to_string()
    }
}

const VARIAN_S_DATA: u16 = 0x0001;
const VARIAN_S_SPEC: u16 = 0x0002;
const VARIAN_S_32: u16 = 0x0004;
const VARIAN_S_FLOAT: u16 = 0x0008;

#[derive(Clone, Debug)]
struct VarianBinaryFid {
    np: usize,
    traces: Vec<Vec<Complex64>>,
}

fn read_varian_fid_binary(path: &Path) -> Result<VarianBinaryFid, String> {
    let bytes =
        fs::read(path).map_err(|e| format!("failed to read Varian fid {}: {e}", path.display()))?;
    if bytes.len() < 32 {
        return Err(format!(
            "Varian fid file {} shorter than 32-byte header",
            path.display()
        ));
    }

    let nblocks = i32::from_be_bytes(bytes[0..4].try_into().unwrap()).max(0) as usize;
    let ntraces = i32::from_be_bytes(bytes[4..8].try_into().unwrap()).max(0) as usize;
    let np = i32::from_be_bytes(bytes[8..12].try_into().unwrap()).max(0) as usize;
    let ebytes = i32::from_be_bytes(bytes[12..16].try_into().unwrap()).max(0) as usize;
    let _tbytes = i32::from_be_bytes(bytes[16..20].try_into().unwrap());
    let bbytes = i32::from_be_bytes(bytes[20..24].try_into().unwrap()).max(0) as usize;
    let status = u16::from_be_bytes(bytes[26..28].try_into().unwrap());
    let nbheaders = i32::from_be_bytes(bytes[28..32].try_into().unwrap()).max(1) as usize;

    if (status & VARIAN_S_DATA) == 0 {
        return Err(format!(
            "Varian fid {} has S_DATA=0 (empty file)",
            path.display()
        ));
    }
    if (status & VARIAN_S_SPEC) != 0 {
        return Err(format!(
            "Varian file {} has S_SPEC=1 (already Fourier-transformed spectrum, expected raw FID)",
            path.display()
        ));
    }
    if np < 2 || !np.is_multiple_of(2) {
        return Err(format!("invalid Varian np={np} in {}", path.display()));
    }
    let total_needed = 32 + nblocks * bbytes;
    if bytes.len() < total_needed {
        return Err(format!(
            "Varian fid {} truncated: header declares {total_needed} B ({nblocks} blocks x {bbytes} B), got {} B",
            path.display(),
            bytes.len()
        ));
    }

    let n_cplx = np / 2;
    let mut traces = Vec::with_capacity(nblocks * ntraces);
    let hdr_bytes_per_block = nbheaders * 28;

    for b in 0..nblocks {
        let b_off = 32 + b * bbytes;
        let blk_status = if nbheaders >= 1 && b_off + 4 <= bytes.len() {
            u16::from_be_bytes(bytes[b_off + 2..b_off + 4].try_into().unwrap())
        } else {
            status
        };
        let is_float = (blk_status & VARIAN_S_FLOAT) != 0 || (status & VARIAN_S_FLOAT) != 0;
        let is_32 = (blk_status & VARIAN_S_32) != 0 || (status & VARIAN_S_32) != 0 || ebytes == 4;

        let data_start = b_off + hdr_bytes_per_block;
        for tr in 0..ntraces {
            let t_off = data_start + tr * np * ebytes;
            let mut cvec = Vec::with_capacity(n_cplx);
            if is_float && ebytes == 4 {
                for k in 0..n_cplx {
                    let p = t_off + k * 8;
                    let re = f32::from_be_bytes(bytes[p..p + 4].try_into().unwrap()) as f64;
                    let im = f32::from_be_bytes(bytes[p + 4..p + 8].try_into().unwrap()) as f64;
                    cvec.push([re, im]);
                }
            } else if is_32 && ebytes == 4 {
                for k in 0..n_cplx {
                    let p = t_off + k * 8;
                    let re = i32::from_be_bytes(bytes[p..p + 4].try_into().unwrap()) as f64;
                    let im = i32::from_be_bytes(bytes[p + 4..p + 8].try_into().unwrap()) as f64;
                    cvec.push([re, im]);
                }
            } else if ebytes == 2 {
                for k in 0..n_cplx {
                    let p = t_off + k * 4;
                    let re = i16::from_be_bytes(bytes[p..p + 2].try_into().unwrap()) as f64;
                    let im = i16::from_be_bytes(bytes[p + 2..p + 4].try_into().unwrap()) as f64;
                    cvec.push([re, im]);
                }
            } else {
                return Err(format!(
                    "unsupported Varian sample format (status=0x{status:04x}, ebytes={ebytes}) in {}",
                    path.display()
                ));
            }
            traces.push(cvec);
        }
    }

    Ok(VarianBinaryFid { np, traces })
}

fn is_varian_explicit_nus(pp: &HashMap<String, ProcparVal>) -> bool {
    let samp_sparse = pp
        .get("sampling")
        .and_then(ProcparVal::first_str)
        .is_some_and(|s| s.eq_ignore_ascii_case("sparse"));
    let sparse_y = pp
        .get("sparse")
        .and_then(ProcparVal::first_str)
        .is_some_and(|s| s.eq_ignore_ascii_case("y"));
    samp_sparse || sparse_y
}

/// IUPAC $\Xi$ ratios relative to ${}^{1}\text{H}$ DSS (Wishart et al., *J. Biomol. NMR* 1995).
fn iupac_xi_ratio(nuc_label: &str) -> f64 {
    match nuc_label {
        "13C" => 0.251_449_530,
        "15N" => 0.101_329_118,
        _ => 1.0,
    }
}

/// Compute Varian/Agilent chemical-shift axis `(obs_mhz, first_ppm)` directly from hardware
/// spectrometer frequencies (`sfrq`, `dfrq`, `dfrq2`), sample temperature (`temp` in C, default 25 C),
/// and IUPAC indirect $\Xi$ ratios anchored to ${}^{1}\text{H}$ DSS (Wishart et al., *J. Biomol. NMR* 1995),
/// ignoring VnmrJ interactive display cursor offsets (`rfl*`, `rfp*`, `reffrq*`) stored in `procpar`.
fn varian_axis_ppm(
    pp: &HashMap<String, ProcparVal>,
    nuc_label: &str,
    sw_hz: f64,
    carrier_mhz: f64,
    sfrq_1h: f64,
) -> (f64, f64) {
    let temp_c = pp
        .get("temp")
        .and_then(ProcparVal::first_f64)
        .filter(|&t| (-20.0..=100.0).contains(&t))
        .unwrap_or(25.0);
    let h2o_ppm = 4.766 - 0.0119 * (temp_c - 25.0);
    let bf_1h_hw = sfrq_1h / (1.0 + h2o_ppm * 1.0e-6);
    // Accept calibrated `H1reffrq` only if consistent within 2.5 ppm of hardware `sfrq_1h`
    let bf_1h = pp
        .get("H1reffrq")
        .and_then(ProcparVal::first_f64)
        .filter(|&v| v > 0.0 && ((v - bf_1h_hw) / bf_1h_hw.max(1.0)).abs() <= 2.5e-6)
        .unwrap_or(bf_1h_hw);
    let obs_mhz = bf_1h * iupac_xi_ratio(nuc_label);
    let first_ppm = ((carrier_mhz - obs_mhz) * 1.0e6 + 0.5 * sw_hz) / obs_mhz;
    (obs_mhz, first_ppm)
}

fn parse_varian_fcoef(raw: Option<&str>) -> Result<Option<[f64; 8]>, String> {
    let Some(s) = raw else {
        return Ok(None);
    };
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    let vals: Result<Vec<f64>, _> = trimmed.split_whitespace().map(str::parse::<f64>).collect();
    let vals = vals.map_err(|e| format!("invalid Varian fcoef '{trimmed}': {e}"))?;
    if vals.len() != 8 {
        return Err(format!(
            "invalid Varian fcoef '{trimmed}': expected 8 floats, got {}",
            vals.len()
        ));
    }
    let mut out = [0.0; 8];
    out.copy_from_slice(&vals);
    Ok(Some(out))
}

/// Apply 8-element VnmrJ `fcoef` `[c0..c7]` to a complex FID pair `(z1, z2)`:
/// With Varian standard forward DFT (`conjugate_for_fft = false`, where `W = RR + i RI = FT(z)`):
/// `Re(FT((a - i b) z)) = a * RR + b * RI`.
#[inline]
fn apply_varian_fcoef_pair(
    coef: [f64; 8],
    z1: &[Complex64],
    z2: &[Complex64],
    out_cos: &mut [Complex64],
    out_sin: &mut [Complex64],
) {
    let [c0, c1, c2, c3, c4, c5, c6, c7] = coef;
    // `(a - i b) * (r + i m) = (a*r + b*m) + i*(a*m - b*r)`
    for idx in 0..z1.len() {
        let (r1, m1) = (z1[idx][0], z1[idx][1]);
        let (r2, m2) = (z2[idx][0], z2[idx][1]);
        out_cos[idx] = [
            (c0 * r1 + c1 * m1) + (c2 * r2 + c3 * m2),
            (c0 * m1 - c1 * r1) + (c2 * m2 - c3 * r2),
        ];
        out_sin[idx] = [
            (c4 * r1 + c5 * m1) + (c6 * r2 + c7 * m2),
            (c4 * m1 - c5 * r1) + (c6 * m2 - c7 * r2),
        ];
    }
}

fn parse_varian_phase_array(
    pp: &HashMap<String, ProcparVal>,
    key: &str,
    fcoef: Option<[f64; 8]>,
) -> Result<(QuadMode, [f64; 8]), String> {
    let arr = pp.get(key).and_then(ProcparVal::numbers).ok_or_else(|| {
        format!("missing `{key}` quadrature array in Varian procpar (never guessing)")
    })?;

    if arr.len() == 1 && (arr[0] - 1.0).abs() < 1e-6 {
        return Ok((QuadMode::Qf, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
    }
    if arr.len() == 2 && (arr[0] - 1.0).abs() < 1e-6 && (arr[1] - 2.0).abs() < 1e-6 {
        if let Some(c) = fcoef {
            let is_ea =
                c[1].abs() > 0.0 || c[3].abs() > 0.0 || c[5].abs() > 0.0 || c[7].abs() > 0.0;
            let qm = if is_ea {
                QuadMode::EchoAntiecho
            } else {
                QuadMode::States
            };
            return Ok((qm, c));
        }
        return Ok((QuadMode::States, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]));
    }

    Err(format!(
        "unsupported Varian `{key}` quadrature array {:?} (expected [1, 2] hypercomplex or [1] QF)",
        arr
    ))
}

fn infer_varian_indirect_channel(
    pp: &HashMap<String, ProcparVal>,
    dim_idx: usize,
    sw_hz: f64,
) -> (String, f64) {
    let axis_str = pp.get("axis").and_then(ProcparVal::first_str).unwrap_or("");
    let ch_char = axis_str.chars().nth(dim_idx);

    let dn = normalize_nucleus_label(
        pp.get("dn")
            .and_then(ProcparVal::first_str)
            .unwrap_or("13C"),
        "13C",
    );
    let dn2 = normalize_nucleus_label(
        pp.get("dn2")
            .and_then(ProcparVal::first_str)
            .unwrap_or("15N"),
        "15N",
    );
    let dfrq = pp
        .get("dfrq")
        .and_then(ProcparVal::first_f64)
        .unwrap_or(125.0);
    let dfrq2 = pp
        .get("dfrq2")
        .and_then(ProcparVal::first_f64)
        .unwrap_or(50.0);

    match ch_char {
        Some('d') => return (dn, dfrq),
        Some('2') => return (dn2, dfrq2),
        Some('p') => {
            let sfrq = pp
                .get("sfrq")
                .and_then(ProcparVal::first_f64)
                .unwrap_or(500.0);
            return ("1H".to_string(), sfrq);
        }
        _ => {}
    }

    let sw_ppm_dn2 = sw_hz / dfrq2.max(1.0);
    let is_2d_only = pp.get("ni2").and_then(ProcparVal::first_f64).unwrap_or(0.0) <= 1.0;
    if dim_idx == 2 || (dn2 == "15N" && (18.0..=50.0).contains(&sw_ppm_dn2) && is_2d_only) {
        (dn2, dfrq2)
    } else {
        (dn, dfrq)
    }
}

#[allow(clippy::too_many_arguments)]
fn accumulate_varian_3d_point<'a, F>(
    qm1: QuadMode,
    coef1: [f64; 8],
    qm2: QuadMode,
    coef2: [f64; 8],
    k1: usize,
    k2: usize,
    n0: usize,
    ni_full: usize,
    mut fetch_sub: F,
    y_e_cos: &mut [Complex64],
    y_e_sin: &mut [Complex64],
    y_o_cos: &mut [Complex64],
    y_o_sin: &mut [Complex64],
    z_cos: &mut [Complex64],
    z_sin: &mut [Complex64],
    data: &mut [Complex64],
) where
    F: FnMut(usize, usize) -> &'a [Complex64],
{
    if qm1 == QuadMode::Qf {
        y_e_cos.copy_from_slice(fetch_sub(0, 0));
        y_e_sin.fill([0.0, 0.0]);
        if qm2 != QuadMode::Qf {
            y_o_cos.copy_from_slice(fetch_sub(0, 1));
            y_o_sin.fill([0.0, 0.0]);
        }
    } else {
        apply_varian_fcoef_pair(coef1, fetch_sub(0, 0), fetch_sub(1, 0), y_e_cos, y_e_sin);
        if qm2 != QuadMode::Qf {
            apply_varian_fcoef_pair(coef1, fetch_sub(0, 1), fetch_sub(1, 1), y_o_cos, y_o_sin);
        }
    }

    let base_c2_0 = (((k2 * 2) * ni_full + k1) * 2) * n0;
    let base_c2_1 = (((k2 * 2 + 1) * ni_full + k1) * 2) * n0;

    if qm2 == QuadMode::Qf {
        add_complex_slice(&mut data[base_c2_0..base_c2_0 + n0], y_e_cos);
        add_complex_slice(&mut data[base_c2_0 + n0..base_c2_0 + 2 * n0], y_e_sin);
    } else {
        apply_varian_fcoef_pair(coef2, y_e_cos, y_o_cos, z_cos, z_sin);
        add_complex_slice(&mut data[base_c2_0..base_c2_0 + n0], z_cos);
        add_complex_slice(&mut data[base_c2_1..base_c2_1 + n0], z_sin);

        apply_varian_fcoef_pair(coef2, y_e_sin, y_o_sin, z_cos, z_sin);
        add_complex_slice(&mut data[base_c2_0 + n0..base_c2_0 + 2 * n0], z_cos);
        add_complex_slice(&mut data[base_c2_1 + n0..base_c2_1 + 2 * n0], z_sin);
    }
}

fn non_empty_procpar_str<'a>(pp: &'a HashMap<String, ProcparVal>, key: &str) -> Option<&'a str> {
    pp.get(key)
        .and_then(ProcparVal::first_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
}

fn read_varian_fid_auto(dir: &Path) -> Result<RawFid, String> {
    let procpar_path = find_file_ci(dir, "procpar")
        .ok_or_else(|| format!("missing Varian procpar in {}", dir.display()))?;
    let procpar_txt = fs::read_to_string(&procpar_path)
        .map_err(|e| format!("failed to read {}: {e}", procpar_path.display()))?;
    let pp = parse_procpar_str(&procpar_txt)?;

    let fid_path = find_file_ci(dir, "fid")
        .ok_or_else(|| format!("missing Varian fid binary in {}", dir.display()))?;
    let bin = read_varian_fid_binary(&fid_path)?;

    let sfrq = pp
        .get("sfrq")
        .and_then(ProcparVal::first_f64)
        .ok_or("missing `sfrq` in Varian procpar")?;
    let sw = pp
        .get("sw")
        .and_then(ProcparVal::first_f64)
        .ok_or("missing `sw` in Varian procpar")?;
    let tn = normalize_nucleus_label(
        pp.get("tn").and_then(ProcparVal::first_str).unwrap_or("1H"),
        "1H",
    );
    let (obs_x, first_ppm_x) = varian_axis_ppm(&pp, &tn, sw, sfrq, sfrq);

    let rp = pp.get("rp").and_then(ProcparVal::first_f64);
    let lp = pp.get("lp").and_then(ProcparVal::first_f64);
    let fn_x = pp
        .get("fn")
        .and_then(ProcparVal::first_f64)
        .map(|v| (v.round() as usize) / 2)
        .filter(|&v| v > 0);
    let lsfid = pp
        .get("lsfid")
        .and_then(ProcparVal::first_f64)
        .unwrap_or(0.0)
        .max(0.0);

    let axis_x = FidAxisMeta {
        label: tn,
        td_raw: bin.np,
        complex_points: bin.np / 2,
        sw_hz: sw,
        obs_mhz: obs_x,
        carrier_mhz: sfrq,
        first_ppm: first_ppm_x,
        quad_mode: QuadMode::Dqd,
        group_delay: lsfid,
        // Varian receiver phase convention uses standard negative-exponent forward FFT without conjugation
        conjugate_for_fft: false,
        stored_proc: StoredDimProc {
            p0_deg: rp,
            p1_deg: lp,
            window: Some(StoredWindowHint::SquaredSineBell {
                offset_pi_frac: 0.5,
            }),
            si: fn_x,
            td_eff: None,
            fcor: Some(0.5),
            reverse: false,
            bc_mod: None,
            bcfw_ppm: None,
        },
    };

    let ni_raw = pp
        .get("ni")
        .and_then(ProcparVal::first_f64)
        .unwrap_or(0.0)
        .round() as usize;
    let nimax = pp
        .get("nimax")
        .and_then(ProcparVal::first_f64)
        .map(|v| v.round() as usize)
        .filter(|&v| v > 1);
    let ni2_raw = pp
        .get("ni2")
        .and_then(ProcparVal::first_f64)
        .unwrap_or(0.0)
        .round() as usize;
    let ni2max = pp
        .get("ni2max")
        .and_then(ProcparVal::first_f64)
        .map(|v| v.round() as usize)
        .filter(|&v| v > 1);

    let pulse_program = pp
        .get("seqfil")
        .and_then(ProcparVal::first_str)
        .unwrap_or("")
        .to_string();
    let name = dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("varian.fid")
        .to_string();

    // Note: standard uniform BioPack sets `sampling = "s"` (sequential); only `sampling = "sparse"`
    // or `sparse = "y"` explicitly flags NUS (or a non-empty schedule with fewer traces than uniform).
    let explicit_nus = is_varian_explicit_nus(&pp);
    let explicit_uniform = !explicit_nus
        && pp
            .get("sampling")
            .and_then(ProcparVal::first_str)
            .is_some_and(|s| s.eq_ignore_ascii_case("s") || s.eq_ignore_ascii_case("standard"));
    let sched_file = if explicit_uniform {
        None
    } else {
        find_varian_schedule_file(dir, &pp)
    };

    // Only consult `nimax` / `ni2max` when NUS is active (on uniform BioPack datasets, `nimax` may hold
    // a template upper bound larger than the acquired uniform `ni`, while on flattened NUS datasets
    // `ni_raw` may equal the schedule row count and `nimax` holds the true Nyquist grid size).
    let prelim_has_nus = explicit_nus || sched_file.is_some();
    let ni = if prelim_has_nus {
        nimax.unwrap_or(ni_raw)
    } else {
        ni_raw
    };
    let ni2 = if prelim_has_nus {
        ni2max.unwrap_or(ni2_raw)
    } else {
        ni2_raw
    };

    if ni <= 1 && ni2 <= 1 {
        if explicit_nus {
            return Err(format!(
                "invalid Varian 1D dataset in {}: procpar declares sparse NUS with ni<=1",
                dir.display()
            ));
        }
        let data = bin
            .traces
            .into_iter()
            .next()
            .ok_or("Varian 1D fid contains 0 traces")?;
        return Ok(RawFid::Dim1(Fid1D {
            name,
            vendor: FidVendor::Varian,
            pulse_program,
            axis_x,
            data,
        }));
    }

    let sw1 = pp
        .get("sw1")
        .and_then(ProcparVal::first_f64)
        .ok_or("missing `sw1` in Varian 2D/3D procpar")?;
    // VnmrJ/BioPack 3D (`ni2 > 1`) uses `f1coef` + `f2coef` in `ft3d`, whereas `procpar` templated
    // from 2D `gNfhsqc` frequently retains a stale non-empty 2D Echo-Antiecho `fcoef`.
    // Prefer non-empty `f1coef` before falling back to `fcoef` when `ni2 > 1` (and `fcoef` -> `f1coef` for 2D).
    let raw_fcoef_1 = if ni2 > 1 {
        non_empty_procpar_str(&pp, "f1coef").or_else(|| non_empty_procpar_str(&pp, "fcoef"))
    } else {
        non_empty_procpar_str(&pp, "fcoef").or_else(|| non_empty_procpar_str(&pp, "f1coef"))
    };
    let fcoef_1 = parse_varian_fcoef(raw_fcoef_1)?;
    let (qm1, coef1) = parse_varian_phase_array(&pp, "phase", fcoef_1)?;
    let (label_1, carrier_1) = infer_varian_indirect_channel(&pp, 1, sw1);
    let (obs_1, first_ppm_1) = varian_axis_ppm(&pp, &label_1, sw1, carrier_1, sfrq);
    let fn1 = pp
        .get("fn1")
        .and_then(ProcparVal::first_f64)
        .map(|v| (v.round() as usize) / 2)
        .filter(|&v| v > 0);

    let p1_mult = if qm1 == QuadMode::Qf { 1 } else { 2 };
    let axis_y = FidAxisMeta {
        label: label_1,
        td_raw: ni * p1_mult,
        complex_points: ni,
        sw_hz: sw1,
        obs_mhz: obs_1,
        carrier_mhz: carrier_1,
        first_ppm: first_ppm_1,
        quad_mode: qm1,
        group_delay: 0.0,
        // VnmrJ `wft2da`/`ft3d` transforms explicit `fcoef` coefficients with `+i` indirect FT,
        // whereas default States without `fcoef` folds imaginary negation into `[1,0,0,0,0,0,+1,0]`.
        conjugate_for_fft: fcoef_1.is_some(),
        stored_proc: StoredDimProc {
            p0_deg: pp.get("rp1").and_then(ProcparVal::first_f64),
            p1_deg: pp.get("lp1").and_then(ProcparVal::first_f64),
            window: Some(StoredWindowHint::SquaredSineBell {
                offset_pi_frac: 0.5,
            }),
            si: fn1,
            td_eff: None,
            fcor: Some(0.5),
            reverse: false,
            bc_mod: None,
            bcfw_ppm: None,
        },
    };

    let n0 = axis_x.complex_points;

    if ni2 <= 1 {
        let uniform_traces = ni * p1_mult;
        let is_nus = explicit_nus || (sched_file.is_some() && bin.traces.len() < uniform_traces);

        let mut data = vec![[0.0, 0.0]; ni * 2 * n0];
        let nus_mask = if is_nus {
            let sched_path = sched_file.ok_or_else(|| {
                format!(
                    "Varian 2D NUS dataset in {} is missing sampling.sch / nuslist schedule file",
                    dir.display()
                )
            })?;
            let coords = parse_nus_schedule_1d(&sched_path, ni)?;
            let expected_acq = coords.len() * p1_mult;
            if bin.traces.len() != expected_acq {
                return Err(format!(
                    "Varian 2D NUS trace count mismatch in {}: fid has {} traces, but schedule {} has {} points ({expected_acq} traces)",
                    dir.display(),
                    bin.traces.len(),
                    sched_path.display(),
                    coords.len()
                ));
            }
            let mut counts = vec![0usize; ni];
            let mut tmp_cos = vec![[0.0, 0.0]; n0];
            let mut tmp_sin = vec![[0.0, 0.0]; n0];
            for (row_idx, &k1) in coords.iter().enumerate() {
                if qm1 == QuadMode::Qf {
                    add_complex_slice(
                        &mut data[(k1 * 2) * n0..(k1 * 2 + 1) * n0],
                        &bin.traces[row_idx],
                    );
                } else {
                    apply_varian_fcoef_pair(
                        coef1,
                        &bin.traces[2 * row_idx],
                        &bin.traces[2 * row_idx + 1],
                        &mut tmp_cos,
                        &mut tmp_sin,
                    );
                    add_complex_slice(&mut data[(k1 * 2) * n0..(k1 * 2 + 1) * n0], &tmp_cos);
                    add_complex_slice(&mut data[(k1 * 2 + 1) * n0..(k1 * 2 + 2) * n0], &tmp_sin);
                }
                counts[k1] += 1;
            }
            for (k1, &cnt) in counts.iter().enumerate() {
                if cnt > 1 {
                    scale_complex_slice(
                        &mut data[(k1 * 2) * n0..(k1 * 2 + 2) * n0],
                        1.0 / (cnt as f64),
                    );
                }
            }
            Some(counts.into_iter().map(|c| c > 0).collect())
        } else {
            if let Some(sp) = sched_file {
                // If a schedule file is present on a full-size 2D dataset, verify whether it matches;
                // if it declares fewer points than uniform_traces, report inconsistency rather than silently ignoring.
                let coords = parse_nus_schedule_1d(&sp, ni)?;
                if coords.len() * p1_mult != bin.traces.len() {
                    return Err(format!(
                        "Varian 2D schedule {} has {} points ({} traces) inconsistent with fid trace count {}",
                        sp.display(),
                        coords.len(),
                        coords.len() * p1_mult,
                        bin.traces.len()
                    ));
                }
            }
            if bin.traces.len() < uniform_traces {
                return Err(format!(
                    "Varian 2D fid in {} has {} traces, expected {uniform_traces} (missing sampling.sch for NUS or incomplete run)",
                    dir.display(),
                    bin.traces.len()
                ));
            }
            if qm1 == QuadMode::Qf {
                for k1 in 0..ni {
                    data[(k1 * 2) * n0..(k1 * 2 + 1) * n0].copy_from_slice(&bin.traces[k1]);
                }
            } else {
                for k1 in 0..ni {
                    let (cos_s, sin_s) = data[(k1 * 2) * n0..(k1 * 2 + 2) * n0].split_at_mut(n0);
                    apply_varian_fcoef_pair(
                        coef1,
                        &bin.traces[2 * k1],
                        &bin.traces[2 * k1 + 1],
                        cos_s,
                        sin_s,
                    );
                }
            }
            None
        };

        return Ok(RawFid::Dim2(Fid2D {
            name,
            vendor: FidVendor::Varian,
            pulse_program,
            axis_x,
            axis_y,
            data,
            nus_mask,
        }));
    }

    // 3D Varian dataset (`ni` and `ni2`)
    let sw2 = pp
        .get("sw2")
        .and_then(ProcparVal::first_f64)
        .ok_or("missing `sw2` in Varian 3D procpar")?;
    let fcoef_2 = parse_varian_fcoef(non_empty_procpar_str(&pp, "f2coef"))?;
    let (qm2, coef2) = parse_varian_phase_array(&pp, "phase2", fcoef_2)?;
    let (label_2, carrier_2) = infer_varian_indirect_channel(&pp, 2, sw2);
    let (obs_2, first_ppm_2) = varian_axis_ppm(&pp, &label_2, sw2, carrier_2, sfrq);
    let fn2 = pp
        .get("fn2")
        .and_then(ProcparVal::first_f64)
        .map(|v| (v.round() as usize) / 2)
        .filter(|&v| v > 0);

    let p2_mult = if qm2 == QuadMode::Qf { 1 } else { 2 };
    let axis_z = FidAxisMeta {
        label: label_2,
        td_raw: ni2 * p2_mult,
        complex_points: ni2,
        sw_hz: sw2,
        obs_mhz: obs_2,
        carrier_mhz: carrier_2,
        first_ppm: first_ppm_2,
        quad_mode: qm2,
        group_delay: 0.0,
        conjugate_for_fft: fcoef_2.is_some(),
        stored_proc: StoredDimProc {
            p0_deg: pp.get("rp2").and_then(ProcparVal::first_f64),
            p1_deg: pp.get("lp2").and_then(ProcparVal::first_f64),
            window: Some(StoredWindowHint::SquaredSineBell {
                offset_pi_frac: 0.5,
            }),
            si: fn2,
            td_eff: None,
            fcor: Some(0.5),
            reverse: false,
            bc_mod: None,
            bcfw_ppm: None,
        },
    };

    let array_str = pp
        .get("array")
        .and_then(ProcparVal::first_str)
        .unwrap_or("phase,phase2")
        .replace([' ', '(', ')'], "");
    let phase2_fastest = array_str == "phase,phase2";
    if !array_str.is_empty()
        && array_str != "phase,phase2"
        && array_str != "phase2,phase"
        && qm1 != QuadMode::Qf
        && qm2 != QuadMode::Qf
    {
        return Err(format!(
            "unsupported Varian 3D `array`='{array_str}' in {} (expected 'phase,phase2' or 'phase2,phase')",
            dir.display()
        ));
    }

    let uniform_traces = ni * p1_mult * ni2 * p2_mult;
    let is_nus = explicit_nus || (sched_file.is_some() && bin.traces.len() < uniform_traces);

    let mut data = vec![[0.0, 0.0]; ni2 * 2 * ni * 2 * n0];
    let mut y_e_cos = vec![[0.0, 0.0]; n0];
    let mut y_e_sin = vec![[0.0, 0.0]; n0];
    let mut y_o_cos = vec![[0.0, 0.0]; n0];
    let mut y_o_sin = vec![[0.0, 0.0]; n0];
    let mut z_cos = vec![[0.0, 0.0]; n0];
    let mut z_sin = vec![[0.0, 0.0]; n0];
    let traces_per_point = p1_mult * p2_mult;

    let nus_mask = if is_nus {
        let sched_path = sched_file.ok_or_else(|| {
            format!(
                "Varian 3D NUS dataset in {} is missing sampling.sch / nuslist schedule file",
                dir.display()
            )
        })?;
        let coords = parse_nus_schedule_2d(&sched_path, ni, ni2, true)?;
        let expected_acq = coords.len() * traces_per_point;
        if bin.traces.len() != expected_acq {
            return Err(format!(
                "Varian 3D NUS trace count mismatch in {}: fid has {} traces, but schedule {} has {} points ({expected_acq} traces)",
                dir.display(),
                bin.traces.len(),
                sched_path.display(),
                coords.len()
            ));
        }
        let mut counts = vec![0usize; ni2 * ni];
        for (row_idx, &(k1, k2)) in coords.iter().enumerate() {
            let base_tr = row_idx * traces_per_point;
            let fetch_sub = |p1: usize, p2: usize| -> &[Complex64] {
                let inner = if phase2_fastest {
                    p1 * p2_mult + p2
                } else {
                    p2 * p1_mult + p1
                };
                &bin.traces[base_tr + inner]
            };
            accumulate_varian_3d_point(
                qm1,
                coef1,
                qm2,
                coef2,
                k1,
                k2,
                n0,
                ni,
                fetch_sub,
                &mut y_e_cos,
                &mut y_e_sin,
                &mut y_o_cos,
                &mut y_o_sin,
                &mut z_cos,
                &mut z_sin,
                &mut data,
            );
            counts[k2 * ni + k1] += 1;
        }
        for k2 in 0..ni2 {
            for k1 in 0..ni {
                let cnt = counts[k2 * ni + k1];
                if cnt > 1 {
                    let inv = 1.0 / (cnt as f64);
                    for c2 in 0..2 {
                        let base = (((k2 * 2 + c2) * ni + k1) * 2) * n0;
                        scale_complex_slice(&mut data[base..base + 2 * n0], inv);
                    }
                }
            }
        }
        Some(counts.into_iter().map(|c| c > 0).collect())
    } else {
        if let Some(sp) = sched_file {
            let coords = parse_nus_schedule_2d(&sp, ni, ni2, true)?;
            if coords.len() * traces_per_point != bin.traces.len() {
                return Err(format!(
                    "Varian 3D schedule {} has {} points ({} traces) inconsistent with fid trace count {}",
                    sp.display(),
                    coords.len(),
                    coords.len() * traces_per_point,
                    bin.traces.len()
                ));
            }
        }
        if bin.traces.len() < uniform_traces {
            return Err(format!(
                "Varian 3D fid in {} has {} traces, expected {uniform_traces} (missing sampling.sch for NUS or incomplete run)",
                dir.display(),
                bin.traces.len()
            ));
        }

        for k2 in 0..ni2 {
            for k1 in 0..ni {
                let base_tr = (k2 * ni + k1) * traces_per_point;
                let fetch_sub = |p1: usize, p2: usize| -> &[Complex64] {
                    let inner = if phase2_fastest {
                        p1 * p2_mult + p2
                    } else {
                        p2 * p1_mult + p1
                    };
                    &bin.traces[base_tr + inner]
                };
                accumulate_varian_3d_point(
                    qm1,
                    coef1,
                    qm2,
                    coef2,
                    k1,
                    k2,
                    n0,
                    ni,
                    fetch_sub,
                    &mut y_e_cos,
                    &mut y_e_sin,
                    &mut y_o_cos,
                    &mut y_o_sin,
                    &mut z_cos,
                    &mut z_sin,
                    &mut data,
                );
            }
        }
        None
    };

    Ok(RawFid::Dim3(Fid3D {
        name,
        vendor: FidVendor::Varian,
        pulse_program,
        axis_x,
        axis_y,
        axis_z,
        data,
        nus_mask,
    }))
}
