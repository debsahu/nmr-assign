//! General `pulseprogram` & acquisition-physics interpreter (`T13.0`) and empirical `{sign, fold}`
//! 2D-HSQC strip-matching likelihood verification (`Q1`).
//!
//! # Clean-Room Provenance
//! Original implementation grounded exclusively in peer-reviewed spin-physics literature and
//! public spectrometer parameter file specifications:
//! - Cavanagh, J., Fairbrother, W. J., Palmer, A. G., Rance, M., & Skelton, N. J., *Protein NMR
//!   Spectroscopy: Principles and Practice*, 2nd ed., Academic Press (2007), Ch. 3, 4, 6 & 7 —
//!   hypercomplex quadrature, constant-time vs real-time evolution, coherence-order selection,
//!   and triple-resonance magnetization-transfer topologies.
//! - States, D. J., Haberkorn, R. A., & Ruben, D. J. (*J. Magn. Reson.* **1982**, 48, 286–292) —
//!   two-dimensional pure-absorption hypercomplex quadrature detection (`States`).
//! - Marion, D., & Wüthrich, K. (*Biochem. Biophys. Res. Commun.* **1983**, 113, 967–974) —
//!   time-proportional phase incrementation (`TPPI`) and `States-TPPI` axial-peak displacement.
//! - Marion, D., Ikura, M., Tschudin, R., & Bax, A. (*J. Magn. Reson.* **1989**, 85, 393–399) —
//!   initial indirect sampling delay $t_1(0) = \Delta t_1 / 2$, linear phase `(PHC0, PHC1) = (-90°, 180°)`,
//!   first-point `FCOR = 1.0` weighting, and folded-peak sign behavior under `States` vs `States-TPPI`.
//! - Kay, L. E., Keifer, P., & Saarinen, T. (*J. Am. Chem. Soc.* **1992**, 114, 10663–10665);
//!   Schleucher, J., et al. (*J. Biomol. NMR* **1994**, 4, 301–306) — pulsed-field-gradient
//!   sensitivity-enhanced Rance–Kay `Echo-Antiecho` coherence selection and $\gamma_H / \gamma_X$
//!   gradient ratio polarity.
//! - Stejskal, E. O., & Tanner, J. E. (*J. Chem. Phys.* **1965**, 42, 288–292); Wu, D. H., Chen, A. D.,
//!   & Johnson, C. S. (*J. Magn. Reson. A* **1995**, 115, 260–264) — bipolar-gradient LED (`BPP-LED`)
//!   and stimulated-echo (`DOSY`) diffusion encoding.
//! - Moskau, D. (*Concepts Magn. Reson.* **2002**, 15, 164–176) — Bruker digital filter group delay
//!   (`GRPDLY` / `DSPFVS` / `DECIM`) phase characteristics.
//! - Public Bruker JCAMP-DX v5.0 (`acqus`/`acqu2s`/`acqu3s`/`acqu4s`, experiment-local executed
//!   `pulseprogram`/`pulseprogram.precomp`, `nuslist`, `vdlist`/`vclist`/`fqlist`/`difflist`) and
//!   public Varian/Agilent `procpar` ASCII parameter format specifications.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::f64::consts::PI;
use std::fmt;
use std::fs;
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::classify::{PulseSequenceClass, QuadratureKind};
use crate::fid::QuadMode;
use crate::peaks::Peak;
use crate::spectrum::{AxisPpm, Spectrum2D, Spectrum3D};

// ============================================================================
// Core Error & Physics Data Types
// ============================================================================

/// Structured error returned by the `pulprog` sandbox preprocessor, parameter parser, or schedule decoder.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PulprogError {
    /// Underlying filesystem read failure inside the experiment directory.
    Io(String),
    /// Rejected `#include` directive attempting `..` parent traversal, absolute path, drive prefix, or symlink escape.
    UnsafeIncludePath(String),
    /// Recursive `#include` cycle detected among local include files.
    IncludeCycle(String),
    /// Local `#include` nesting exceeded the safety bound.
    IncludeDepthExceeded(usize),
    /// Total `#include` count, preprocessed line count, file byte size, or `procpar` array length exceeded safety cap.
    ResourceLimitExceeded(String),
    /// Malformed `#ifdef`/`#ifndef`/`#else`/`#endif` conditional structure.
    PreprocessorSyntax(String),
    /// Required spectrometer acquisition header (`acqus` or `procpar`) is missing or invalid.
    MissingParameters(String),
    /// Invalid or out-of-bounds NUS coordinate schedule.
    InvalidNusSchedule(String),
}

impl fmt::Display for PulprogError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(s) => write!(f, "pulprog I/O error: {s}"),
            Self::UnsafeIncludePath(p) => {
                write!(
                    f,
                    "unsafe pulseprogram #include path rejected (experiment-local relative filenames only): '{p}'"
                )
            }
            Self::IncludeCycle(p) => {
                write!(f, "pulseprogram local #include cycle detected at '{p}'")
            }
            Self::IncludeDepthExceeded(d) => {
                write!(f, "pulseprogram local #include depth exceeded limit ({d})")
            }
            Self::ResourceLimitExceeded(s) => {
                write!(f, "pulprog resource safety limit exceeded: {s}")
            }
            Self::PreprocessorSyntax(s) => write!(f, "pulseprogram preprocessor syntax error: {s}"),
            Self::MissingParameters(s) => write!(f, "missing acquisition parameters: {s}"),
            Self::InvalidNusSchedule(s) => write!(f, "invalid NUS schedule: {s}"),
        }
    }
}

impl std::error::Error for PulprogError {}

/// Spectrometer vendor family for raw acquisition interpretation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PulprogVendor {
    Bruker,
    Varian,
}

/// Decoded quadrature mode on an indirect dimension.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecodedQuadMode {
    States,
    StatesTppi,
    Tppi,
    EchoAntiecho,
    Qf,
    Qseq,
    Dqd,
}

impl DecodedQuadMode {
    /// Convert to `fid::QuadMode` when directly supported by hypercomplex/QF time-domain storage.
    pub fn to_fid_quad_mode(self) -> Option<QuadMode> {
        match self {
            Self::States => Some(QuadMode::States),
            Self::StatesTppi => Some(QuadMode::StatesTppi),
            Self::EchoAntiecho => Some(QuadMode::EchoAntiecho),
            Self::Qf => Some(QuadMode::Qf),
            Self::Dqd => Some(QuadMode::Dqd),
            Self::Tppi | Self::Qseq => None,
        }
    }

    /// Convert to `classify::QuadratureKind`.
    pub fn to_quadrature_kind(self) -> QuadratureKind {
        match self {
            Self::States => QuadratureKind::States,
            Self::StatesTppi => QuadratureKind::StatesTppi,
            Self::Tppi => QuadratureKind::Tppi,
            Self::EchoAntiecho => QuadratureKind::EchoAntiecho,
            Self::Qf | Self::Qseq => QuadratureKind::Qf,
            Self::Dqd => QuadratureKind::Dqd,
        }
    }
}

/// Provenance of the decoded quadrature mode for an indirect dimension.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuadSource {
    /// Standard Bruker `##$FnMODE` integer code in `acqu2s`/`acqu3s`/`acqu4s`.
    HeaderFnMode(i32),
    /// Manual hypercomplex loop (`lo to <lbl> times 2` with `ip*`/`dp*`/`igrad`) overriding `FnMODE = 0` or `1 (QF)`.
    SynthesizedManualLoop { header_fnmode: i32 },
    /// Varian `procpar` `phase`/`phase2`/`phase3` + `f1coef`/`f2coef`/`f3coef` array.
    VarianPhaseArray,
}

/// Sensitivity-enhanced Rance–Kay / Echo-Antiecho linear combination matrix and gradient polarity.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RanceKayMatrix {
    /// Canonical 8-element `[re0_c, im0_c, re1_c, im1_c, re0_s, im0_s, re1_s, im1_s]` coefficients
    /// mapping raw trace pair `(z0, z1)` to `(cos, sin)` components.
    pub coef: [f64; 8],
    /// `true` when P-type and N-type echo/antiecho selection pathways are swapped relative to standard
    /// same-sign `GPZ` Rance–Kay convention (e.g., opposite-sign `GPZ[enc]` vs `GPZ[dec]` or `*EA*-1`).
    pub ea_polarity_inverted: bool,
    /// Encoding gradient amplitude (`%` for Bruker `GPZ`, DAC units for Varian `gzlvl`) when identified.
    pub enc_grad_level: Option<f64>,
    /// Decoding gradient amplitude (`%` for Bruker `GPZ`, DAC units for Varian `gzlvl`) when identified.
    pub dec_grad_level: Option<f64>,
    /// Signed encoding-to-decoding gradient ratio $G_{\text{enc}} / G_{\text{dec}}$ (`~ +9.87` for $^{15}\text{N}/^1\text{H}$, `~ +3.98` for $^{13}\text{C}/^1\text{H}$).
    pub grad_ratio: Option<f64>,
}

/// Spectral aliasing/folding behavior across the indirect Nyquist boundary (Marion et al. 1989).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FoldingConvention {
    /// Circular wrap across Nyquist edge without sign inversion (`States` / `Echo-Antiecho` without TPPI).
    SameSignWrap,
    /// Nyquist edge shifted by $SW/2$ (`States-TPPI` / `TPPI`); aliased peaks fold/invert sign under `PHC1 = 180°` or TPPI.
    MirrorInvertedWrap,
    /// Constant-time indirect evolution with cosinusoidal `J_CC` modulation / CT mirror-image behavior.
    ConstantTimeMirror,
    /// Non-phase-sensitive single-channel / diffusion-attenuated axis (`QF` / `DOSY`).
    Magnitude,
}

/// Multi-dimensional indirect loop ordering in 3D/4D experiments.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AcquisitionOrder {
    Dim1D,
    Dim2D,
    /// Canonical Bruker `AQSEQ = 0` (`3-2-1`: `F3` direct, `F2`/`acqu2s` inner loop, `F1`/`acqu3s` outer loop)
    /// or Varian `array = "phase,phase2"` (`d2`/`ni` inner, `d3`/`ni2` outer, `phase2` fastest).
    Aqseq321,
    /// Transposed Bruker `AQSEQ = 1` (`3-1-2`: `F3` direct, `F1`/`acqu3s` inner loop, `F2`/`acqu2s` outer loop).
    Aqseq312,
    /// Varian `array = "phase2,phase"` where `(ni, ni2)` grid order holds with `phase` fastest inside `(k1, k2)`.
    VarianPhase1Fastest,
    Dim4D,
}

/// Storage layout of pseudo-3D relaxation, CPMG, CEST, hetNOE, or DOSY series.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Pseudo3DStorageOrder {
    /// Pure frequency-evolved 1D/2D/3D/4D dataset (no pseudo-dimension list loop).
    NotPseudo3D,
    /// Complete 2D planes acquired sequentially: relaxation/exchange variable (`vd`/`vc`/`fq`/`difflist`
    /// or Varian `array = "relaxT,phase"` / `"relaxT,d2,phase"`) advances in the OUTER loop outside $t_1$
    /// (`[plane_0(all t_1), plane_1(all t_1), ...]`).
    PlaneQueued,
    /// Interleaved at every $t_1$ increment: relaxation/exchange/reference toggle (`vd`/`vc`/`fq`/`l*` or
    /// Varian `array = "d2,relaxT,phase"`) advances INSIDE the $t_1$ loop (`[t_1(0) all delays, t_1(1) all delays, ...]`).
    T1Interleaved,
}

/// Detected index origin (`0`-based vs `1`-based) of a `nuslist` or `sampling.sch` schedule.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NusIndexBase {
    ZeroBased,
    OneBased,
}

/// Decoded non-uniform sampling schedule normalized to 0-based canonical indirect coordinates.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecodedNusSchedule {
    pub index_base: NusIndexBase,
    /// `true` when raw file columns `(col0, col1)` were transposed from `AQSEQ = 312` `(F1, F2)` into `(F2, F1)`.
    pub aqseq_transposed: bool,
    /// Normalized 0-based grid coordinates: `[k1]` in 2D, `[k1_f2, k2_f1]` in 3D, `[k1, k2, k3]` in 4D.
    pub points: Vec<Vec<usize>>,
}

/// Symbolic loop node extracted from `lo to <lbl> times <count>` or `mc #0 to <lbl>` directives.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoopNode {
    pub label: String,
    pub count_expr: String,
    pub resolved_count: Option<usize>,
    pub start_line: usize,
    pub end_line: usize,
    pub phase_ops: Vec<String>,
    pub delay_ops: Vec<String>,
    pub grad_ops: Vec<String>,
    pub list_ops: Vec<String>,
    pub children: Vec<LoopNode>,
}

/// Deduced indirect-dimension spin-physics descriptor.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IndirectDimPhysics {
    /// Sequence evolution index (`1` = `F1`/`d0` or Varian `d2`/`ni`; `2` = `F2`/`d10` or Varian `d3`/`ni2`).
    pub f_dim: usize,
    /// Source parameter block (`"acqu2s"`, `"acqu3s"`, `"procpar:d2"`, `"procpar:d3"`, etc.).
    pub acqu_source: String,
    /// Canonical isotope label (`"15N"`, `"13C"`, `"1H"`, etc.).
    pub nucleus: String,
    pub carrier_ppm: f64,
    pub sw_ppm: f64,
    pub sw_hz: f64,
    pub obs_mhz: f64,
    pub td_raw: usize,
    pub complex_points: usize,
    pub quad_mode: DecodedQuadMode,
    pub quad_source: QuadSource,
    /// `true` when indirect precession phase requires frequency reversal (`REVERSE = yes`).
    pub reverse_frequency: bool,
    pub rance_kay: Option<RanceKayMatrix>,
    pub constant_time: bool,
    /// Effective initial evolution delay $t_1(0)$ in seconds (including $4 p_{90} / \pi$ compensation when uncompensated).
    pub initial_delay_s: f64,
    /// Full complex dwell increment $\Delta t = 1 / SW_{\text{Hz}}$ in seconds.
    pub increment_s: f64,
    /// `true` when $t_1(0) \approx \Delta t / 2$ (`Marion et al. 1989` half-dwell sampling or Varian `f1180 = 'y'`).
    pub half_dwell_sampling: bool,
    /// Predicted zero-order indirect phase in degrees (`0.0` or `-90.0`).
    pub predicted_phc0_deg: f64,
    /// Predicted first-order indirect phase in degrees (`0.0` or `180.0`).
    pub predicted_phc1_deg: f64,
    pub folding: FoldingConvention,
}

/// Spin pool node in the deduced magnetization-transfer topology.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum SpinPoolNode {
    HAmide,
    HAliphatic,
    HAromatic,
    NAmide,
    CAlpha,
    CBeta,
    CAliphatic,
    CCarbonyl,
    CAromatic,
    Deuterium,
}

/// Physical coupling or encoding mechanism connecting two spin pools.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum TransferMechanism {
    IneptJCoupling { delay_ms: f64 },
    ConstantTimeJCoupling { delay_ms: f64 },
    IsotropicTocsyMixing,
    DipolarNoesyMixing { mix_ms: f64 },
    DiffusionStejskalTanner { delta_ms: f64, bipolar: bool },
    RelaxationT1Delay,
    RelaxationT2Echo,
    CpmgDispersion,
    CestSaturation,
    HetNoeSaturation,
}

/// Directed magnetization-transfer or encoding step deduced from RF/gradient instructions.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TransferEdge {
    pub from: SpinPoolNode,
    pub to: SpinPoolNode,
    pub mechanism: TransferMechanism,
}

/// Rich pulse-sequence family deduced from magnetization-transfer physics (includes `Dosy` and `N15Trosy`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PulprogSequenceFamily {
    N15Hsqc,
    N15Trosy,
    C13Hsqc,
    C13Hmqc,
    Hnco,
    Hncaco,
    Hnca,
    Hncoca,
    Hncacb,
    Cbcaconh,
    Hbhaconh,
    Ccconh,
    Hccconh,
    HcchTocsy,
    N15Noesy,
    C13Noesy,
    Con2D,
    Caco2D,
    Canco3D,
    Cbcaco3D,
    RelaxR1,
    RelaxR2,
    HetNoe,
    Cpmg,
    Cest,
    Dosy,
    OneDim,
    Unknown,
}

impl PulprogSequenceFamily {
    /// Map to `classify::PulseSequenceClass` (preserving `Unknown` for `Dosy`/`OneDim`/`Hbhaconh`
    /// so non-assignment decoys are never auto-selected as backbone assignment spectra).
    pub fn to_pulse_sequence_class(self) -> PulseSequenceClass {
        match self {
            Self::N15Hsqc | Self::N15Trosy => PulseSequenceClass::N15Hsqc,
            Self::C13Hsqc => PulseSequenceClass::C13Hsqc,
            Self::C13Hmqc => PulseSequenceClass::C13Hmqc,
            Self::Hnco => PulseSequenceClass::Hnco,
            Self::Hncaco => PulseSequenceClass::Hncaco,
            Self::Hnca => PulseSequenceClass::Hnca,
            Self::Hncoca => PulseSequenceClass::Hncoca,
            Self::Hncacb => PulseSequenceClass::Hncacb,
            Self::Cbcaconh => PulseSequenceClass::Cbcaconh,
            Self::Ccconh => PulseSequenceClass::Ccconh,
            Self::Hccconh => PulseSequenceClass::Hccconh,
            Self::HcchTocsy => PulseSequenceClass::HcchTocsy,
            Self::N15Noesy => PulseSequenceClass::N15Noesy,
            Self::C13Noesy => PulseSequenceClass::C13Noesy,
            Self::Con2D => PulseSequenceClass::Con2D,
            Self::Caco2D => PulseSequenceClass::Caco2D,
            Self::Canco3D => PulseSequenceClass::Canco3D,
            Self::Cbcaco3D => PulseSequenceClass::Cbcaco3D,
            Self::RelaxR1 => PulseSequenceClass::RelaxR1,
            Self::RelaxR2 => PulseSequenceClass::RelaxR2,
            Self::HetNoe => PulseSequenceClass::HetNoe,
            Self::Cpmg => PulseSequenceClass::Cpmg,
            Self::Cest => PulseSequenceClass::Cest,
            Self::Dosy | Self::Hbhaconh | Self::OneDim | Self::Unknown => {
                PulseSequenceClass::Unknown
            }
        }
    }
}

/// Deduced magnetization-transfer graph and active channel physics.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MagnetizationTransferGraph {
    pub active_nuclei: Vec<String>,
    pub carrier_jumps_ppm: Vec<(String, f64)>,
    pub evolved_pools: Vec<(usize, SpinPoolNode)>,
    pub detected_pool: SpinPoolNode,
    pub trosy_selected: bool,
    pub sensitivity_enhanced: bool,
    pub diffusion_encoded: bool,
    pub edges: Vec<TransferEdge>,
    pub deduced_family: PulprogSequenceFamily,
}

/// Complete spin-physics interpretation of a Bruker or Varian experiment directory.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PulprogInterpretation {
    pub vendor: PulprogVendor,
    pub pulse_program_name: String,
    pub executed_source_present: bool,
    pub active_defines: BTreeMap<String, String>,
    pub resolved_local_includes: Vec<String>,
    pub unresolved_standard_includes: Vec<String>,
    pub preprocessed_lines: usize,
    pub dimensionality: usize,
    pub aqseq: AcquisitionOrder,
    pub pseudo_3d_order: Pseudo3DStorageOrder,
    pub pseudo_3d_list_kind: Option<String>,
    /// When `pseudo_3d_order == T1Interleaved` and explicit loops / array nesting exist,
    /// `Some(true)` if the `vd`/`vc`/`fq` plane list advances *inner* to the `times 2` quadrature loop
    /// (`((k1 * 2 + q) * n_planes) + p`), and `Some(false)` if the `times 2` quadrature loop is innermost
    /// (`((k1 * n_planes + p) * 2) + q`). `None` when `mc` macros determine nesting from `acqu2s`/`acqu3s`.
    pub pseudo_3d_list_inner_to_quad: Option<bool>,
    pub varian_array_nesting: Vec<String>,
    pub direct_nucleus: String,
    pub direct_group_delay_pts: f64,
    pub cnst: Vec<f64>,
    pub delays_s: Vec<f64>,
    pub increments_s: Vec<f64>,
    pub loop_counters: Vec<usize>,
    pub gpz_pct: Vec<f64>,
    pub spnam: Vec<String>,
    pub cpdprg: Vec<String>,
    pub loops: Vec<LoopNode>,
    pub indirect_dims: Vec<IndirectDimPhysics>,
    pub nus_schedule: Option<DecodedNusSchedule>,
    pub transfer_graph: MagnetizationTransferGraph,
    pub warnings: Vec<String>,
}

// ============================================================================
// Section 1: Preprocessor (`ZGOPTNS`, `#ifdef`/`#ifndef`/`#else`/`#endif`, Safe Local `#include`)
// ============================================================================

const MAX_INCLUDE_DEPTH: usize = 8;
const MAX_TOTAL_INCLUDES: usize = 64;
const MAX_PREPROCESSED_LINES: usize = 65_536;
const MAX_TEXT_FILE_BYTES: u64 = 4 * 1024 * 1024;
const MAX_PROCPAR_VALUES: usize = 1_000_000;

/// Parse Bruker `##$ZGOPTNS` (e.g. `"-DTROSY -DLABEL_CN -DMODE=2 -UNOE"`) into a macro map.
pub fn parse_zgoptns(zgoptns: &str) -> BTreeMap<String, String> {
    let mut defs = BTreeMap::new();
    let trimmed = zgoptns.trim().trim_matches('<').trim_matches('>');
    for tok in trimmed.split_whitespace() {
        let tok = tok.trim_matches('"').trim_matches('\'');
        if let Some(rest) = tok.strip_prefix("-D") {
            if let Some((k, v)) = rest.split_once('=') {
                let k = k.trim();
                if !k.is_empty() {
                    defs.insert(k.to_string(), v.trim().to_string());
                }
            } else if !rest.is_empty() {
                defs.insert(rest.to_string(), "1".to_string());
            }
        } else if let Some(rest) = tok.strip_prefix("-U") {
            defs.remove(rest.trim());
        }
    }
    defs
}

/// Strictly validate a `#include` target path:
/// - Rejects empty strings, leading `/` or `\`, `~`, Windows/URI `:` prefixes, backslashes, and ANY `..` component!
/// - Permits only safe relative paths within the experiment directory.
pub fn validate_local_include_path(raw: &str) -> Result<String, PulprogError> {
    let t = raw.trim();
    if t.is_empty()
        || t.starts_with('/')
        || t.starts_with('\\')
        || t.starts_with('~')
        || t.contains('\\')
        || t.contains(':')
        || t.contains('\0')
    {
        return Err(PulprogError::UnsafeIncludePath(raw.to_string()));
    }
    let p = Path::new(t);
    for comp in p.components() {
        match comp {
            Component::Normal(os) => {
                let s = os.to_string_lossy();
                if s == ".." || s.is_empty() {
                    return Err(PulprogError::UnsafeIncludePath(raw.to_string()));
                }
            }
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(PulprogError::UnsafeIncludePath(raw.to_string()));
            }
        }
    }
    Ok(t.to_string())
}

/// Output of preprocessing an executed Bruker `pulseprogram`.
#[derive(Clone, Debug, Default)]
pub struct PreprocessedPulprog {
    pub lines: Vec<String>,
    pub defines: BTreeMap<String, String>,
    pub resolved_includes: Vec<String>,
    pub unresolved_standard_includes: Vec<String>,
}

/// Strip C-style `/* ... */` block comments across the entire source while preserving line breaks.
fn strip_c_block_comments_preserving_lines(src: &str) -> String {
    let bytes = src.as_bytes();
    let mut out = String::with_capacity(src.len());
    let mut i = 0usize;
    let mut in_block = false;
    let mut in_quotes = false;

    while i < bytes.len() {
        if in_block {
            if i + 1 < bytes.len() && bytes[i] == b'*' && bytes[i + 1] == b'/' {
                in_block = false;
                i += 2;
            } else {
                if bytes[i] == b'\n' {
                    out.push('\n');
                }
                i += 1;
            }
        } else {
            if bytes[i] == b'"' {
                in_quotes = !in_quotes;
                out.push('"');
                i += 1;
                continue;
            }
            if !in_quotes && i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'*' {
                in_block = true;
                i += 2;
                continue;
            }
            let ch = src[i..].chars().next().unwrap();
            out.push(ch);
            i += ch.len_utf8();
        }
    }
    out
}

/// Strip Bruker `;` comments outside `"..."` quotes on a single line.
fn strip_bruker_semicolon_comment(line: &str) -> &str {
    let mut in_quotes = false;
    for (idx, b) in line.bytes().enumerate() {
        if b == b'"' {
            in_quotes = !in_quotes;
        } else if b == b';' && !in_quotes {
            return &line[..idx];
        }
    }
    line
}

/// Evaluate a `#if` / `#elif` boolean preprocessor expression against `defines`.
fn eval_cpp_expr(expr: &str, defines: &BTreeMap<String, String>) -> bool {
    let s = expr.trim();
    if s.is_empty() {
        return false;
    }
    // Split top-level `||`
    if let Some((lhs, rhs)) = split_top_level_binop(s, "||") {
        return eval_cpp_expr(lhs, defines) || eval_cpp_expr(rhs, defines);
    }
    // Split top-level `&&`
    if let Some((lhs, rhs)) = split_top_level_binop(s, "&&") {
        return eval_cpp_expr(lhs, defines) && eval_cpp_expr(rhs, defines);
    }
    // Outer parentheses
    if s.starts_with('(') && s.ends_with(')') && matching_outer_parens(s) {
        return eval_cpp_expr(&s[1..s.len() - 1], defines);
    }
    // Unary `!`
    if let Some(rest) = s.strip_prefix('!') {
        return !eval_cpp_expr(rest, defines);
    }
    // `defined(SYM)` or `defined SYM`
    if let Some(rest) = s.strip_prefix("defined") {
        let r = rest.trim();
        if let Some(inner) = r.strip_prefix('(').and_then(|x| x.strip_suffix(')')) {
            return defines.contains_key(inner.trim());
        }
        if !r.is_empty() && r.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_') {
            return defines.contains_key(r);
        }
    }
    // `==` / `!=`
    if let Some((lhs, rhs)) = split_top_level_binop(s, "==") {
        return resolve_cpp_atom(lhs, defines) == resolve_cpp_atom(rhs, defines);
    }
    if let Some((lhs, rhs)) = split_top_level_binop(s, "!=") {
        return resolve_cpp_atom(lhs, defines) != resolve_cpp_atom(rhs, defines);
    }
    let val = resolve_cpp_atom(s, defines);
    val != 0
}

fn matching_outer_parens(s: &str) -> bool {
    let mut depth = 0i32;
    for (idx, b) in s.bytes().enumerate() {
        if b == b'(' {
            depth += 1;
        } else if b == b')' {
            depth -= 1;
            if depth == 0 && idx + 1 < s.len() {
                return false;
            }
        }
    }
    depth == 0
}

fn split_top_level_binop<'a>(s: &'a str, op: &str) -> Option<(&'a str, &'a str)> {
    let bytes = s.as_bytes();
    let ob = op.as_bytes();
    let mut depth = 0i32;
    let mut i = 0usize;
    while i + ob.len() <= bytes.len() {
        if bytes[i] == b'(' {
            depth += 1;
        } else if bytes[i] == b')' {
            depth -= 1;
        } else if depth == 0 && &bytes[i..i + ob.len()] == ob {
            return Some((&s[..i], &s[i + ob.len()..]));
        }
        i += 1;
    }
    None
}

fn resolve_cpp_atom(atom: &str, defines: &BTreeMap<String, String>) -> i64 {
    let t = atom.trim();
    if let Ok(v) = t.parse::<i64>() {
        return v;
    }
    if let Some(dv) = defines.get(t) {
        return dv.trim().parse::<i64>().unwrap_or(1);
    }
    0
}

#[derive(Clone, Copy, Debug)]
struct CondFrame {
    parent_active: bool,
    branch_taken: bool,
    currently_active: bool,
    seen_else: bool,
}

/// Preprocess Bruker `pulseprogram` source in memory against `zgoptns` and an optional map of
/// experiment-local include files (`filename -> source`).
pub fn preprocess_bruker_pulseprogram(
    source: &str,
    zgoptns: &str,
    local_includes: &HashMap<String, String>,
) -> Result<PreprocessedPulprog, PulprogError> {
    preprocess_with_resolver(source, zgoptns, &|inc_rel| {
        if let Some(txt) = local_includes.get(inc_rel) {
            return Ok(Some(txt.clone()));
        }
        // Case-insensitive single-filename match within `local_includes`
        for (k, v) in local_includes {
            if k.eq_ignore_ascii_case(inc_rel) {
                return Ok(Some(v.clone()));
            }
        }
        Ok(None)
    })
}

///Canonicalize `exp_dir` fail-closed so symlink containment checks can never be bypassed.
fn canonicalize_exp_dir(exp_dir: &Path) -> Result<PathBuf, PulprogError> {
    fs::canonicalize(exp_dir).map_err(|e| {
        PulprogError::Io(format!(
            "failed to canonicalize experiment directory {}: {e}",
            exp_dir.display()
        ))
    })
}

/// Read an experiment-local text file fail-closed:
/// 1. `fs::canonicalize(path)` must succeed and reside strictly inside `canon_root` (rejects any escaping symlink!).
/// 2. Enforces `MAX_TEXT_FILE_BYTES` (`4 MiB`) before and during reading.
/// 3. Decodes ISO-8859-1 / Latin-1 vendor header comments cleanly via `String::from_utf8_lossy`.
fn read_sandboxed_text_lossy(
    canon_root: &Path,
    path: &Path,
    display_rel: &str,
) -> Result<String, PulprogError> {
    let target_c = fs::canonicalize(path)
        .map_err(|_| PulprogError::UnsafeIncludePath(display_rel.to_string()))?;
    if !target_c.starts_with(canon_root) {
        return Err(PulprogError::UnsafeIncludePath(display_rel.to_string()));
    }
    let meta = fs::metadata(&target_c)
        .map_err(|e| PulprogError::Io(format!("{}: {e}", path.display())))?;
    if meta.len() > MAX_TEXT_FILE_BYTES {
        return Err(PulprogError::ResourceLimitExceeded(format!(
            "file '{}' ({} bytes) exceeds {} byte safety cap",
            display_rel,
            meta.len(),
            MAX_TEXT_FILE_BYTES
        )));
    }
    use std::io::Read;
    let f = fs::File::open(&target_c)
        .map_err(|e| PulprogError::Io(format!("{}: {e}", path.display())))?;
    let mut buf = Vec::with_capacity((meta.len() as usize).min(65_536));
    f.take(MAX_TEXT_FILE_BYTES + 1)
        .read_to_end(&mut buf)
        .map_err(|e| PulprogError::Io(format!("{}: {e}", path.display())))?;
    if (buf.len() as u64) > MAX_TEXT_FILE_BYTES {
        return Err(PulprogError::ResourceLimitExceeded(format!(
            "file '{}' exceeds {} byte safety cap",
            display_rel, MAX_TEXT_FILE_BYTES
        )));
    }
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

/// Preprocess Bruker `pulseprogram` inside `exp_dir`, resolving local `#include` files strictly
/// inside `exp_dir` (never accessing external TopSpin/system include directories and rejecting
/// symlinks or paths that escape `exp_dir` fail-closed).
pub fn preprocess_bruker_pulseprogram_in_dir(
    source: &str,
    zgoptns: &str,
    exp_dir: &Path,
) -> Result<PreprocessedPulprog, PulprogError> {
    let canon_root = canonicalize_exp_dir(exp_dir)?;
    preprocess_with_resolver(source, zgoptns, &|inc_rel| {
        let Some(found_path) = find_relative_child_ci(exp_dir, inc_rel) else {
            return Ok(None);
        };
        let txt = read_sandboxed_text_lossy(&canon_root, &found_path, inc_rel)?;
        Ok(Some(txt))
    })
}

fn find_relative_child_ci(base: &Path, rel: &str) -> Option<PathBuf> {
    let is_file_or_symlink =
        |p: &Path| p.is_file() || fs::symlink_metadata(p).is_ok_and(|m| m.file_type().is_symlink());
    let direct = base.join(rel);
    if is_file_or_symlink(&direct) {
        return Some(direct);
    }
    let mut cur = base.to_path_buf();
    for seg in rel.split('/') {
        let entries = fs::read_dir(&cur).ok()?;
        let mut matched: Option<PathBuf> = None;
        for ent in entries.flatten() {
            if ent.file_name().to_string_lossy().eq_ignore_ascii_case(seg) {
                matched = Some(ent.path());
                break;
            }
        }
        cur = matched?;
    }
    is_file_or_symlink(&cur).then_some(cur)
}

fn preprocess_with_resolver<F>(
    source: &str,
    zgoptns: &str,
    resolve_local: &F,
) -> Result<PreprocessedPulprog, PulprogError>
where
    F: Fn(&str) -> Result<Option<String>, PulprogError>,
{
    if (source.len() as u64) > MAX_TEXT_FILE_BYTES {
        return Err(PulprogError::ResourceLimitExceeded(format!(
            "pulseprogram source ({} bytes) exceeds {} byte safety cap",
            source.len(),
            MAX_TEXT_FILE_BYTES
        )));
    }
    let mut out = PreprocessedPulprog {
        lines: Vec::new(),
        defines: parse_zgoptns(zgoptns),
        resolved_includes: Vec::new(),
        unresolved_standard_includes: Vec::new(),
    };
    let mut stack: Vec<CondFrame> = Vec::new();
    let mut include_chain: Vec<String> = Vec::new();
    let mut total_includes: usize = 0;

    preprocess_recursive(
        source,
        resolve_local,
        &mut out,
        &mut stack,
        &mut include_chain,
        &mut total_includes,
    )?;

    if !stack.is_empty() {
        return Err(PulprogError::PreprocessorSyntax(
            "unterminated #ifdef / #ifndef / #if block at end of pulseprogram".to_string(),
        ));
    }
    Ok(out)
}

fn preprocess_recursive<F>(
    source: &str,
    resolve_local: &F,
    out: &mut PreprocessedPulprog,
    stack: &mut Vec<CondFrame>,
    include_chain: &mut Vec<String>,
    total_includes: &mut usize,
) -> Result<(), PulprogError>
where
    F: Fn(&str) -> Result<Option<String>, PulprogError>,
{
    if (source.len() as u64) > MAX_TEXT_FILE_BYTES {
        return Err(PulprogError::ResourceLimitExceeded(format!(
            "pulseprogram include ({} bytes) exceeds {} byte safety cap",
            source.len(),
            MAX_TEXT_FILE_BYTES
        )));
    }
    let cleaned = strip_c_block_comments_preserving_lines(source);

    for raw_line in cleaned.lines() {
        let trimmed = raw_line.trim_start();
        if let Some(after_hash) = trimmed.strip_prefix('#') {
            let directive_full = strip_bruker_semicolon_comment(after_hash).trim();
            if directive_full.is_empty() {
                continue;
            }
            // Precompiled TopSpin `cpp` linemarker: `# 24 "/opt/topspin/..." 2` -> ignore!
            if directive_full.as_bytes()[0].is_ascii_digit() {
                continue;
            }

            let (kw, rest) = split_first_word(directive_full);
            let parent_active = stack.last().map(|f| f.currently_active).unwrap_or(true);

            match kw {
                "ifdef" => {
                    let sym = rest.split_whitespace().next().unwrap_or("");
                    let cond = parent_active && out.defines.contains_key(sym);
                    stack.push(CondFrame {
                        parent_active,
                        branch_taken: cond,
                        currently_active: cond,
                        seen_else: false,
                    });
                }
                "ifndef" => {
                    let sym = rest.split_whitespace().next().unwrap_or("");
                    let cond = parent_active && !out.defines.contains_key(sym);
                    stack.push(CondFrame {
                        parent_active,
                        branch_taken: cond,
                        currently_active: cond,
                        seen_else: false,
                    });
                }
                "if" => {
                    let cond = parent_active && eval_cpp_expr(rest, &out.defines);
                    stack.push(CondFrame {
                        parent_active,
                        branch_taken: cond,
                        currently_active: cond,
                        seen_else: false,
                    });
                }
                "elif" => {
                    let Some(top) = stack.last_mut() else {
                        return Err(PulprogError::PreprocessorSyntax(
                            "#elif without matching #if/#ifdef".to_string(),
                        ));
                    };
                    if top.seen_else {
                        return Err(PulprogError::PreprocessorSyntax(
                            "#elif after #else".to_string(),
                        ));
                    }
                    if !top.parent_active || top.branch_taken {
                        top.currently_active = false;
                    } else {
                        let cond = eval_cpp_expr(rest, &out.defines);
                        top.currently_active = cond;
                        top.branch_taken = cond;
                    }
                }
                "else" => {
                    let Some(top) = stack.last_mut() else {
                        return Err(PulprogError::PreprocessorSyntax(
                            "#else without matching #if/#ifdef".to_string(),
                        ));
                    };
                    if top.seen_else {
                        return Err(PulprogError::PreprocessorSyntax(
                            "duplicate #else in conditional block".to_string(),
                        ));
                    }
                    top.seen_else = true;
                    top.currently_active = top.parent_active && !top.branch_taken;
                    top.branch_taken = true;
                }
                "endif" => {
                    if stack.pop().is_none() {
                        return Err(PulprogError::PreprocessorSyntax(
                            "#endif without matching #if/#ifdef".to_string(),
                        ));
                    }
                }
                "define" => {
                    if parent_active {
                        let (sym, val) = split_first_word(rest);
                        if !sym.is_empty() {
                            // Ignore Bruker `define delay DELTA` / `define list<gradient>` (those do NOT start with `#`,
                            // whereas `#define FOO 1` does).
                            let v = if val.is_empty() { "1" } else { val };
                            out.defines.insert(sym.to_string(), v.to_string());
                        }
                    }
                }
                "undef" => {
                    if parent_active {
                        let sym = rest.split_whitespace().next().unwrap_or("");
                        out.defines.remove(sym);
                    }
                }
                "include" => {
                    let raw_target = parse_include_target(rest)?;
                    // Always validate include path safety and local symlink containment even in inactive
                    // conditional branches so path-traversal or escaping-symlink payloads fail closed!
                    let safe_rel = validate_local_include_path(&raw_target)?;
                    if parent_active {
                        if include_chain.len() >= MAX_INCLUDE_DEPTH {
                            return Err(PulprogError::IncludeDepthExceeded(MAX_INCLUDE_DEPTH));
                        }
                        if include_chain
                            .iter()
                            .any(|p| p.eq_ignore_ascii_case(&safe_rel))
                        {
                            return Err(PulprogError::IncludeCycle(safe_rel));
                        }
                        match resolve_local(&safe_rel)? {
                            Some(inc_src) => {
                                if *total_includes >= MAX_TOTAL_INCLUDES {
                                    return Err(PulprogError::ResourceLimitExceeded(format!(
                                        "total local #include expansions exceeded limit ({MAX_TOTAL_INCLUDES})"
                                    )));
                                }
                                *total_includes += 1;
                                out.resolved_includes.push(safe_rel.clone());
                                include_chain.push(safe_rel);
                                preprocess_recursive(
                                    &inc_src,
                                    resolve_local,
                                    out,
                                    stack,
                                    include_chain,
                                    total_includes,
                                )?;
                                include_chain.pop();
                            }
                            None => {
                                out.unresolved_standard_includes.push(safe_rel);
                            }
                        }
                    } else {
                        let _ = resolve_local(&safe_rel)?;
                    }
                }
                _ => {
                    // Other vendor pragmas (e.g. `#pragma`) are ignored when safe.
                }
            }
            continue;
        }

        let active = stack.last().map(|f| f.currently_active).unwrap_or(true);
        if active {
            if out.lines.len() >= MAX_PREPROCESSED_LINES {
                return Err(PulprogError::ResourceLimitExceeded(format!(
                    "preprocessed line count exceeded limit ({MAX_PREPROCESSED_LINES})"
                )));
            }
            out.lines.push(raw_line.to_string());
        }
    }
    Ok(())
}

fn split_first_word(s: &str) -> (&str, &str) {
    let s = s.trim();
    for (idx, ch) in s.char_indices() {
        if ch.is_whitespace() || ch == '(' || ch == '"' || ch == '<' {
            if idx == 0 {
                break;
            }
            return (&s[..idx], s[idx..].trim());
        }
    }
    (s, "")
}

fn parse_include_target(rest: &str) -> Result<String, PulprogError> {
    let t = rest.trim();
    if let Some(inner) = t.strip_prefix('"') {
        if let Some(end) = inner.find('"') {
            return Ok(inner[..end].to_string());
        }
    }
    if let Some(inner) = t.strip_prefix('<') {
        if let Some(end) = inner.find('>') {
            return Ok(inner[..end].to_string());
        }
    }
    Err(PulprogError::PreprocessorSyntax(format!(
        "malformed #include target: '{t}'"
    )))
}

// ============================================================================
// Section 2: Bruker JCAMP-DX Acquisition Header & Moskau (2002) Group Delay Parser
// ============================================================================

/// Parsed Bruker JCAMP-DX scalar and array map (`KEY -> raw value string`).
pub fn parse_bruker_jcamp_text(text: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    let mut current_key: Option<String> = None;
    let mut current_val = String::new();

    for raw_line in text.lines() {
        let line = raw_line.split("$$").next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix("##") {
            if let Some(prev_k) = current_key.take() {
                map.insert(prev_k, clean_jcamp_scalar(&current_val));
                current_val.clear();
            }
            if let Some((k_part, v_part)) = rest.split_once('=') {
                let key = k_part.trim().trim_start_matches('$').to_string();
                if key != "END" {
                    current_key = Some(key);
                    current_val.push_str(v_part.trim());
                }
            }
        } else if current_key.is_some() {
            if !current_val.is_empty() {
                current_val.push(' ');
            }
            current_val.push_str(line);
        }
    }
    if let Some(prev_k) = current_key.take() {
        map.insert(prev_k, clean_jcamp_scalar(&current_val));
    }
    map
}

fn clean_jcamp_scalar(raw: &str) -> String {
    let t = raw.trim();
    if t.starts_with('<') && t.ends_with('>') && !t[1..t.len() - 1].contains('>') {
        return t[1..t.len() - 1].to_string();
    }
    t.to_string()
}

/// Extract numeric array `(0..N) v0 v1 ...` from a parsed Bruker JCAMP value string.
pub fn parse_jcamp_f64_vec(map: &HashMap<String, String>, key: &str, min_len: usize) -> Vec<f64> {
    let mut out = Vec::new();
    if let Some(raw) = map.get(key) {
        let body = if let Some(idx) = raw.find(')') {
            if raw.trim_start().starts_with('(') {
                &raw[idx + 1..]
            } else {
                raw.as_str()
            }
        } else {
            raw.as_str()
        };
        for tok in body.split_whitespace() {
            if let Ok(v) = tok.parse::<f64>() {
                out.push(v);
            }
        }
    }
    if out.len() < min_len {
        out.resize(min_len, 0.0);
    }
    out
}

/// Extract integer array `(0..N) v0 v1 ...` from a parsed Bruker JCAMP value string.
pub fn parse_jcamp_usize_vec(
    map: &HashMap<String, String>,
    key: &str,
    min_len: usize,
) -> Vec<usize> {
    let mut out = Vec::new();
    if let Some(raw) = map.get(key) {
        let body = if let Some(idx) = raw.find(')') {
            if raw.trim_start().starts_with('(') {
                &raw[idx + 1..]
            } else {
                raw.as_str()
            }
        } else {
            raw.as_str()
        };
        for tok in body.split_whitespace() {
            if let Ok(v) = tok.parse::<isize>() {
                out.push(v.max(0) as usize);
            }
        }
    }
    if out.len() < min_len {
        out.resize(min_len, 0);
    }
    out
}

/// Extract `<...>` bracketed string array `(0..N) <s0> <s1> ...` from a Bruker JCAMP value string.
pub fn parse_jcamp_string_vec(map: &HashMap<String, String>, key: &str) -> Vec<String> {
    let mut out = Vec::new();
    let Some(raw) = map.get(key) else {
        return out;
    };
    let mut rest = raw.as_str();
    while let Some(open) = rest.find('<') {
        let after = &rest[open + 1..];
        if let Some(close) = after.find('>') {
            out.push(after[..close].to_string());
            rest = &after[close + 1..];
        } else {
            break;
        }
    }
    out
}

/// Compute Bruker digital filter group delay in complex points from `GRPDLY`, falling back to the
/// published DSP firmware table (`DSPFVS` 10/11/12 & `DECIM`, Moskau *Concepts Magn. Reson.* 2002).
pub fn resolve_bruker_group_delay(acqus: &HashMap<String, String>) -> f64 {
    if let Some(gd) = acqus
        .get("GRPDLY")
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|&g| g > 0.0)
    {
        return gd;
    }
    let dspfvs = acqus
        .get("DSPFVS")
        .and_then(|s| s.parse::<i32>().ok())
        .unwrap_or(0);
    let decim = acqus
        .get("DECIM")
        .and_then(|s| s.parse::<i32>().ok())
        .unwrap_or(1);
    if dspfvs <= 0 || decim <= 1 {
        return 0.0;
    }
    // Moskau (2002) Table 1 exact phase-delay values for Avance DSPFVS 10, 11, 12
    match (dspfvs, decim) {
        (10, 2) => 44.7500,
        (10, 3) => 33.5000,
        (10, 4) => 66.6250,
        (10, 6) => 59.0833,
        (10, 8) => 68.5625,
        (10, 12) => 60.3750,
        (10, 16) => 69.5313,
        (10, 24) => 61.0208,
        (10, 32) => 70.0156,
        (10, 48) => 61.3438,
        (10, 64) => 70.2578,
        (10, 96) => 61.5052,
        (10, 128) => 70.3789,
        (10, 192) => 61.5859,
        (10, 256) => 70.4395,
        (10, 384) => 61.6263,
        (10, 512) => 70.4697,
        (10, 768) => 61.6465,
        (10, 1024) => 70.4849,
        (10, 1536) => 61.6566,
        (10, 2048) => 70.4924,
        (11, 2) => 46.0000,
        (11, 3) => 36.5000,
        (11, 4) => 48.0000,
        (11, 6) => 50.1667,
        (11, 8) => 53.2500,
        (11, 12) => 69.5000,
        (11, 16) => 72.2500,
        (11, 24) => 70.1667,
        (11, 32) => 72.7500,
        (11, 48) => 70.5000,
        (11, 64) => 73.0000,
        (11, 96) => 70.6667,
        (11, 128) => 72.5000,
        (11, 192) => 71.3333,
        (11, 256) => 72.2500,
        (11, 384) => 71.6667,
        (11, 512) => 72.1250,
        (11, 768) => 71.8333,
        (11, 1024) => 72.0625,
        (11, 1536) => 71.9167,
        (11, 2048) => 72.0313,
        (12, 2) => 46.311,
        (12, 3) => 36.530,
        (12, 4) => 47.870,
        (12, 6) => 50.229,
        (12, 8) => 53.289,
        (12, 12) => 69.551,
        (12, 16) => 71.600,
        (12, 24) => 70.184,
        (12, 32) => 72.138,
        (12, 48) => 70.528,
        (12, 64) => 72.348,
        (12, 96) => 70.700,
        (12, 128) => 72.524,
        _ => 0.0,
    }
}

/// Normalize Bruker/Varian nucleus token (`<1H>`, `N15`, `C13`, `PROTON`, etc.) to canonical IUPAC mass-prefix form.
pub fn canonicalize_nucleus(raw: &str, obs_mhz: Option<f64>, h1_mhz: Option<f64>) -> String {
    let alnum: String = raw
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect::<String>()
        .to_ascii_uppercase();
    match alnum.as_str() {
        "1H" | "H1" | "PROTON" | "H" => return "1H".to_string(),
        "15N" | "N15" | "N" => return "15N".to_string(),
        "13C" | "C13" | "C" => return "13C".to_string(),
        "2H" | "H2" | "D" | "DEUTERIUM" => return "2H".to_string(),
        "19F" | "F19" => return "19F".to_string(),
        "31P" | "P31" => return "31P".to_string(),
        _ => {}
    }
    if let (Some(f), Some(h)) = (obs_mhz, h1_mhz) {
        if h > 40.0 && f > 1.0 {
            let r = f / h;
            if (0.92..=1.08).contains(&r) {
                return "1H".to_string();
            }
            if (0.090..=0.115).contains(&r) {
                return "15N".to_string();
            }
            if (0.235..=0.265).contains(&r) {
                return "13C".to_string();
            }
            if (0.145..=0.165).contains(&r) {
                return "2H".to_string();
            }
        }
    }
    if obs_mhz.is_some_and(|f| f >= 250.0) {
        return "1H".to_string();
    }
    if alnum.is_empty() {
        "1H".to_string()
    } else {
        alnum
    }
}

// ============================================================================
// Section 3: `nuslist` & Varian `sampling.sch` Decoder (0-based vs 1-based + `AQSEQ`)
// ============================================================================

/// Parse and normalize a 1D/2D/3D NUS schedule (`nuslist` or `sampling.sch`), automatically
/// detecting **0-based vs 1-based** indexing and transposing `(col0, col1)` when `aqseq == Aqseq312`.
///
/// # 0-based vs 1-based Detection Invariants
/// 1. Explicit header comment (`1-based`, `one-based`, `index_base=1`, `base 1`) with all coordinates in `1..=grid_dims[d]` -> `OneBased`.
/// 2. Zero (`0`) never appears anywhere in the schedule AND at least one coordinate equals `grid_dims[d]` (with all coordinates in `1..=grid_dims[d]`) -> `OneBased` (mathematically impossible in 0-based!).
/// 3. Zero (`0`) never appears anywhere in `>= 8` rows, the initial `t=0` origin row is `[1, 1, ...]`, and all coordinates lie in `1..=grid_dims[d]` -> `OneBased`.
/// 4. Otherwise `ZeroBased` (validating `k < grid_dims[d]`, so mixed invalid schedules like `0` + `n_cplx` are strictly rejected as out-of-bounds!).
pub fn decode_nuslist_str(
    text: &str,
    grid_dims: &[usize],
    aqseq: AcquisitionOrder,
) -> Result<DecodedNusSchedule, PulprogError> {
    let nd = grid_dims.len();
    if nd == 0 {
        return Err(PulprogError::InvalidNusSchedule(
            "empty indirect grid dimensions".to_string(),
        ));
    }
    let aqseq_transposed = nd == 2 && aqseq == AcquisitionOrder::Aqseq312;
    // File column grid bounds before `AQSEQ` swap:
    let file_bounds: Vec<usize> = if aqseq_transposed {
        vec![grid_dims[1], grid_dims[0]]
    } else {
        grid_dims.to_vec()
    };

    let mut explicit_one_based = false;
    let mut explicit_zero_based = false;
    let mut raw_rows: Vec<(usize, Vec<usize>)> = Vec::new();

    for (line_idx, raw_line) in text.lines().enumerate() {
        if let Some(cmt_idx) = raw_line.find(['#', ';']) {
            let cmt = raw_line[cmt_idx + 1..].to_ascii_lowercase();
            if cmt.contains("1-based")
                || cmt.contains("one-based")
                || cmt.contains("index_base=1")
                || cmt.contains("index_base = 1")
            {
                explicit_one_based = true;
            } else if cmt.contains("0-based")
                || cmt.contains("zero-based")
                || cmt.contains("index_base=0")
            {
                explicit_zero_based = true;
            }
        }
        let clean = raw_line.split(['#', ';']).next().unwrap_or("").trim();
        if clean.is_empty() {
            continue;
        }
        let toks: Vec<&str> = clean.split_whitespace().collect();
        if toks.len() != nd {
            return Err(PulprogError::InvalidNusSchedule(format!(
                "line {}: expected {nd} integer coordinate(s), got '{}'",
                line_idx + 1,
                clean
            )));
        }
        let mut row = Vec::with_capacity(nd);
        for tok in toks {
            let val = tok.parse::<usize>().map_err(|e| {
                PulprogError::InvalidNusSchedule(format!(
                    "line {}: invalid integer '{tok}': {e}",
                    line_idx + 1
                ))
            })?;
            row.push(val);
        }
        raw_rows.push((line_idx + 1, row));
    }

    if raw_rows.is_empty() {
        return Err(PulprogError::InvalidNusSchedule(
            "NUS schedule is empty".to_string(),
        ));
    }

    let any_zero = raw_rows.iter().any(|(_, r)| r.contains(&0));
    let all_within_one_based = !any_zero
        && raw_rows.iter().all(|(_, r)| {
            r.iter()
                .zip(file_bounds.iter())
                .all(|(&v, &b)| v >= 1 && v <= b)
        });
    let any_equals_upper = raw_rows.iter().any(|(_, r)| {
        r.iter()
            .zip(file_bounds.iter())
            .any(|(&v, &b)| b > 0 && v == b)
    });
    let first_is_unit_origin = raw_rows[0].1.iter().all(|&v| v == 1);

    let index_base = if !explicit_zero_based
        && all_within_one_based
        && (explicit_one_based || any_equals_upper || (raw_rows.len() >= 8 && first_is_unit_origin))
    {
        NusIndexBase::OneBased
    } else {
        NusIndexBase::ZeroBased
    };

    let mut points = Vec::with_capacity(raw_rows.len());
    for (line_no, mut r) in raw_rows {
        if index_base == NusIndexBase::OneBased {
            for v in &mut r {
                *v -= 1;
            }
        }
        if aqseq_transposed {
            r.swap(0, 1);
        }
        for (d, (&k, &bound)) in r.iter().zip(grid_dims.iter()).enumerate() {
            if k >= bound {
                return Err(PulprogError::InvalidNusSchedule(format!(
                    "out-of-bounds NUS coordinate dim {d}: k={k} at line {line_no} (full grid n_cplx={bound})"
                )));
            }
        }
        points.push(r);
    }

    Ok(DecodedNusSchedule {
        index_base,
        aqseq_transposed,
        points,
    })
}

// ============================================================================
// Section 4: Executed Bruker `pulseprogram` Symbolic Loop & Physics Interpreter
// ============================================================================

#[derive(Clone, Debug, Default)]
struct RawLoopSpan {
    label: String,
    count_expr: String,
    resolved_count: Option<usize>,
    start_line: usize,
    end_line: usize,
    phase_ops: Vec<String>,
    delay_ops: Vec<String>,
    grad_ops: Vec<String>,
    list_ops: Vec<String>,
}

#[derive(Clone, Debug, Default)]
struct SymbolicDimLoopInfo {
    synthesized_quad: Option<DecodedQuadMode>,
    quad_reg_sign: Option<(usize, i32)>,
    paired_post_evo_echo_quad: bool,
    comment_reverse_override: Option<bool>,
    has_tppi_toggle: bool,
    has_igrad_ea: bool,
    ea_explicit_mult: f64,
    ea_dur_ratio: f64,
    ea_grad_idx: Option<usize>,
    dec_grad_idx: Option<usize>,
    constant_time: bool,
    evo_decremented: bool,
    sct_inc_before_dec: Option<bool>,
    loop_nesting_rank: Option<usize>,
    per_scan_fq_sfo_mhz: Option<(String, f64)>,
}

/// Inputs for in-memory Bruker experiment physics interpretation.
#[derive(Clone, Debug, Default)]
pub struct BrukerExperimentInputs<'a> {
    pub acqus_text: &'a str,
    pub acqu2s_text: Option<&'a str>,
    pub acqu3s_text: Option<&'a str>,
    pub acqu4s_text: Option<&'a str>,
    pub pulseprogram_text: Option<&'a str>,
    pub nuslist_text: Option<&'a str>,
    pub vdlist_text: Option<&'a str>,
    pub vclist_text: Option<&'a str>,
    pub fqlist_text: Option<&'a str>,
    pub difflist_text: Option<&'a str>,
    pub local_includes: HashMap<String, String>,
}

fn parse_bruker_time_literal_s(raw: &str) -> Option<f64> {
    let t = raw.trim().trim_matches('"');
    if t.is_empty() {
        return None;
    }
    let last = t.as_bytes()[t.len() - 1];
    let (num_s, scale) = match last {
        b's' if t.ends_with("ms") => (&t[..t.len() - 2], 1e-3),
        b's' if t.ends_with("us") => (&t[..t.len() - 2], 1e-6),
        b's' if t.ends_with("ns") => (&t[..t.len() - 2], 1e-9),
        b's' => (&t[..t.len() - 1], 1.0),
        b'm' => (&t[..t.len() - 1], 1e-3),
        b'u' => (&t[..t.len() - 1], 1e-6),
        b'n' => (&t[..t.len() - 1], 1e-9),
        _ => (t, 1.0),
    };
    num_s.trim().parse::<f64>().ok().map(|v| v * scale)
}

fn contains_word(haystack: &str, needle: &str) -> bool {
    first_word_pos(haystack, needle).is_some()
}

fn first_word_pos(haystack: &str, needle: &str) -> Option<usize> {
    let hb = haystack.as_bytes();
    let nb = needle.as_bytes();
    if nb.is_empty() || hb.len() < nb.len() {
        return None;
    }
    let is_ident = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    let mut pos = 0usize;
    while pos + nb.len() <= hb.len() {
        if let Some(rel) = haystack[pos..].find(needle) {
            let s = pos + rel;
            let e = s + nb.len();
            let l_ok = s == 0 || !is_ident(hb[s - 1]);
            let r_ok = e == hb.len() || !is_ident(hb[e]);
            if l_ok && r_ok {
                return Some(s);
            }
            pos = s + 1;
        } else {
            break;
        }
    }
    None
}

fn extract_loop_ops_from_line(
    code: &str,
    phase_ops: &mut Vec<String>,
    delay_ops: &mut Vec<String>,
    grad_ops: &mut Vec<String>,
    list_ops: &mut Vec<String>,
) {
    let toks: Vec<&str> = code
        .split(|c: char| c.is_whitespace() || matches!(c, ',' | '(' | ')' | ':' | ';'))
        .filter(|s| !s.is_empty())
        .collect();
    let mut i = 0usize;
    while i < toks.len() {
        let tok = toks[i];
        let low = tok.to_ascii_lowercase();
        if let Some(rest) = low
            .strip_prefix("ip")
            .or_else(|| low.strip_prefix("dp"))
            .or_else(|| low.strip_prefix("rp"))
        {
            let digits = rest.bytes().take_while(|b| b.is_ascii_digit()).count();
            if digits > 0 {
                phase_ops.push(low.clone());
            }
        } else if let Some(rest) = low
            .strip_prefix("id")
            .or_else(|| low.strip_prefix("dd"))
            .or_else(|| low.strip_prefix("rd"))
        {
            if !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit()) {
                delay_ops.push(low.clone());
            }
        } else if low == "igrad" || low == "setgrad" || low == "dgrad" || low == "rgrad" {
            let arg = toks.get(i + 1).copied().unwrap_or("");
            grad_ops.push(if arg.is_empty() {
                low.clone()
            } else {
                format!("{low} {arg}")
            });
        } else if matches!(
            low.as_str(),
            "vd" | "vc" | "va" | "vp" | "ivd" | "ivc" | "difflist"
        ) || (low.starts_with("fq")
            && low.len() <= 4
            && low[2..].bytes().all(|b| b.is_ascii_digit()))
            || low.starts_with("fq=")
        {
            // Note: `fq1:f2` inside loop increments vs RF body — we record `fq` list advances (`fq1list` or `fq1` without `=cnst`).
            if !low.starts_with("fq=cnst") {
                list_ops.push(low);
            }
        }
        i += 1;
    }
}

fn resolve_loop_count_expr(expr: &str, loop_counters: &[usize], td_ind: &[usize]) -> Option<usize> {
    let clean = expr.trim().trim_matches(|c: char| c == '(' || c == ')');
    if let Ok(v) = clean.parse::<usize>() {
        return Some(v);
    }
    let low = clean.to_ascii_lowercase();
    if let Some(rest) = low.strip_prefix('l') {
        if let Ok(idx) = rest.parse::<usize>() {
            return loop_counters.get(idx).copied().filter(|&v| v > 0);
        }
    }
    if low == "td1" {
        return td_ind.first().copied().filter(|&v| v > 0);
    }
    if low == "td2" {
        return td_ind.get(1).copied().filter(|&v| v > 0);
    }
    if low == "td3" {
        return td_ind.get(2).copied().filter(|&v| v > 0);
    }
    None
}

fn nest_loop_spans(mut spans: Vec<RawLoopSpan>) -> Vec<LoopNode> {
    // Sort outer-first: earlier `start_line`, or larger `end_line` on identical `start_line`!
    // Notice in Bruker Avance:
    //   2 d1 ...
    //     ip3
    //     lo to 2 times 2     (start=10, end=20 -> inner!)
    //     id0
    //     lo to 2 times l0    (start=10, end=23 -> outer!)
    spans.sort_by(|a, b| {
        a.start_line
            .cmp(&b.start_line)
            .then_with(|| b.end_line.cmp(&a.end_line))
    });

    fn insert_into_tree(tree: &mut Vec<LoopNode>, node: LoopNode) {
        for existing in tree.iter_mut() {
            if node.start_line >= existing.start_line && node.end_line <= existing.end_line {
                insert_into_tree(&mut existing.children, node);
                return;
            }
        }
        tree.push(node);
    }

    let mut roots = Vec::new();
    for sp in spans {
        insert_into_tree(
            &mut roots,
            LoopNode {
                label: sp.label,
                count_expr: sp.count_expr,
                resolved_count: sp.resolved_count,
                start_line: sp.start_line,
                end_line: sp.end_line,
                phase_ops: sp.phase_ops,
                delay_ops: sp.delay_ops,
                grad_ops: sp.grad_ops,
                list_ops: sp.list_ops,
                children: Vec::new(),
            },
        );
    }
    roots
}

/// Interpret an in-memory Bruker experiment (`acqus`, `acqu2s`/`3s`/`4s`, executed `pulseprogram`,
/// `nuslist`, and optional variable lists).
pub fn interpret_bruker_experiment(
    inp: &BrukerExperimentInputs<'_>,
) -> Result<PulprogInterpretation, PulprogError> {
    interpret_bruker_experiment_inner(inp, None)
}

/// Interpret a Bruker experiment directory on disk, reading ONLY experiment-local `acqus`/`acqu*s`,
/// `pulseprogram`/`pulseprogram.precomp`, `nuslist`, and variable schedule files (fail-closed
/// against escaping symlinks and oversized files, with ISO-8859-1 / Latin-1 lossy decoding).
pub fn interpret_bruker_dir(dir: impl AsRef<Path>) -> Result<PulprogInterpretation, PulprogError> {
    let dir = dir.as_ref();
    let canon_root = canonicalize_exp_dir(dir)?;
    let acqus_text =
        read_merged_jcamp_pair_in_dir(dir, &canon_root, "acqus", "acqu")?.ok_or_else(|| {
            PulprogError::MissingParameters(format!("missing acqus/acqu in {}", dir.display()))
        })?;
    let acqu2s_text = read_merged_jcamp_pair_in_dir(dir, &canon_root, "acqu2s", "acqu2")?;
    let acqu3s_text = read_merged_jcamp_pair_in_dir(dir, &canon_root, "acqu3s", "acqu3")?;
    let acqu4s_text = read_merged_jcamp_pair_in_dir(dir, &canon_root, "acqu4s", "acqu4")?;

    let pp_path = find_relative_child_ci(dir, "pulseprogram")
        .or_else(|| find_relative_child_ci(dir, "pulseprogram.precomp"));
    let pulseprogram_text = match pp_path {
        Some(p) => {
            let rel_name = p
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "pulseprogram".to_string());
            Some(read_sandboxed_text_lossy(&canon_root, &p, &rel_name)?)
        }
        None => None,
    };

    let read_opt_list = |names: &[&str]| -> Result<Option<String>, PulprogError> {
        for &nm in names {
            if let Some(p) = find_relative_child_ci(dir, nm) {
                let s = read_sandboxed_text_lossy(&canon_root, &p, nm)?;
                if !s.trim().is_empty() {
                    return Ok(Some(s));
                }
            }
        }
        Ok(None)
    };

    let nuslist_text = read_opt_list(&["nuslist"])?;
    let vdlist_text = read_opt_list(&["vdlist"])?;
    let vclist_text = read_opt_list(&["vclist"])?;
    let fqlist_text = read_opt_list(&[
        "fqlist", "fq1list", "fq2list", "fq3list", "fq4list", "fq5list", "fq6list", "fq7list",
        "fq8list",
    ])?;
    let difflist_text = read_opt_list(&["difflist"])?;

    let inputs = BrukerExperimentInputs {
        acqus_text: &acqus_text,
        acqu2s_text: acqu2s_text.as_deref(),
        acqu3s_text: acqu3s_text.as_deref(),
        acqu4s_text: acqu4s_text.as_deref(),
        pulseprogram_text: pulseprogram_text.as_deref(),
        nuslist_text: nuslist_text.as_deref(),
        vdlist_text: vdlist_text.as_deref(),
        vclist_text: vclist_text.as_deref(),
        fqlist_text: fqlist_text.as_deref(),
        difflist_text: difflist_text.as_deref(),
        local_includes: HashMap::new(),
    };
    interpret_bruker_experiment_inner(&inputs, Some(dir))
}

fn read_merged_jcamp_pair_in_dir(
    dir: &Path,
    canon_root: &Path,
    status_name: &str,
    base_name: &str,
) -> Result<Option<String>, PulprogError> {
    let p_status = find_relative_child_ci(dir, status_name);
    let p_base = find_relative_child_ci(dir, base_name);
    match (p_base, p_status) {
        (Some(b), Some(s)) => {
            let mut txt = read_sandboxed_text_lossy(canon_root, &b, base_name)?;
            txt.push('\n');
            txt.push_str(&read_sandboxed_text_lossy(canon_root, &s, status_name)?);
            Ok(Some(txt))
        }
        (None, Some(s)) => Ok(Some(read_sandboxed_text_lossy(
            canon_root,
            &s,
            status_name,
        )?)),
        (Some(b), None) => Ok(Some(read_sandboxed_text_lossy(canon_root, &b, base_name)?)),
        (None, None) => Ok(None),
    }
}

#[allow(clippy::too_many_lines)]
fn interpret_bruker_experiment_inner(
    inp: &BrukerExperimentInputs<'_>,
    exp_dir: Option<&Path>,
) -> Result<PulprogInterpretation, PulprogError> {
    let a1 = parse_bruker_jcamp_text(inp.acqus_text);
    if a1.is_empty() {
        return Err(PulprogError::MissingParameters(
            "empty Bruker acqus parameter map".to_string(),
        ));
    }
    let a2 = inp.acqu2s_text.map(parse_bruker_jcamp_text);
    let a3 = inp.acqu3s_text.map(parse_bruker_jcamp_text);
    let a4 = inp.acqu4s_text.map(parse_bruker_jcamp_text);

    let parmode = a1.get("PARMODE").and_then(|s| s.parse::<usize>().ok());
    let mut dimensionality = match parmode {
        Some(p) => p + 1,
        None => {
            if a4.is_some() {
                4
            } else if a3.is_some() {
                3
            } else if a2.is_some() {
                2
            } else {
                1
            }
        }
    };
    if dimensionality >= 2 && a2.is_none() {
        dimensionality = 1;
    } else if dimensionality >= 3 && a3.is_none() {
        dimensionality = 2;
    } else if dimensionality >= 4 && a4.is_none() {
        dimensionality = 3;
    }

    let pulprog_name = a1.get("PULPROG").cloned().unwrap_or_default();
    let zgoptns_raw = a1.get("ZGOPTNS").map(String::as_str).unwrap_or("");

    let cnst = parse_jcamp_f64_vec(&a1, "CNST", 64);
    let mut delays_s = parse_jcamp_f64_vec(&a1, "D", 64);
    let mut increments_s = parse_jcamp_f64_vec(&a1, "IN", 64);
    let inp_vec = parse_jcamp_f64_vec(&a1, "INP", 64);
    for (idx, &v) in inp_vec.iter().enumerate().take(64) {
        if increments_s[idx] <= 0.0 && v > 0.0 {
            increments_s[idx] = v * 1e-6;
        }
    }
    let pulses_us = parse_jcamp_f64_vec(&a1, "P", 64);
    let mut loop_counters = parse_jcamp_usize_vec(&a1, "L", 32);
    let gpz_pct = parse_jcamp_f64_vec(&a1, "GPZ", 32);
    let spnam = parse_jcamp_string_vec(&a1, "SPNAM");
    let cpdprg = parse_jcamp_string_vec(&a1, "CPDPRG");
    let td_indirect_arr = parse_jcamp_usize_vec(&a1, "TD_INDIRECT", 8);

    // Channel nuclei `:f1`..`:f4`
    let sfo1_ch = a1.get("SFO1").and_then(|s| s.parse::<f64>().ok());
    let bf1_ch = a1
        .get("BF1")
        .and_then(|s| s.parse::<f64>().ok())
        .or(sfo1_ch);
    let direct_nucleus = canonicalize_nucleus(
        a1.get("NUC1").map(String::as_str).unwrap_or("1H"),
        sfo1_ch,
        bf1_ch.filter(|&b| b >= 100.0),
    );
    let h1_base_mhz = if direct_nucleus == "1H" {
        bf1_ch
    } else {
        (1..=4).find_map(|ch| {
            let n = a1
                .get(&format!("NUC{ch}"))
                .map(String::as_str)
                .unwrap_or("");
            let bf = a1
                .get(&format!("BF{ch}"))
                .or_else(|| a1.get(&format!("SFO{ch}")))
                .and_then(|s| s.parse::<f64>().ok())?;
            (canonicalize_nucleus(n, Some(bf), None) == "1H").then_some(bf)
        })
    };

    let mut ch_nuclei: [String; 5] = [
        String::new(),
        direct_nucleus.clone(),
        String::new(),
        String::new(),
        String::new(),
    ];
    let mut ch_carrier_ppm: [Option<f64>; 5] = [None; 5];
    for ch in 1..=4 {
        let sfo = a1
            .get(&format!("SFO{ch}"))
            .and_then(|s| s.parse::<f64>().ok());
        let bf = a1
            .get(&format!("BF{ch}"))
            .and_then(|s| s.parse::<f64>().ok())
            .or(sfo);
        if let Some(raw_n) = a1.get(&format!("NUC{ch}")) {
            if !raw_n.eq_ignore_ascii_case("off") && !raw_n.trim().is_empty() {
                ch_nuclei[ch] = canonicalize_nucleus(raw_n, sfo, h1_base_mhz);
            }
        }
        if let (Some(s), Some(b)) = (sfo, bf) {
            if b > 1.0 {
                let c_ppm = (s - b) * 1e6 / b;
                if c_ppm.abs() > 1e-3 {
                    ch_carrier_ppm[ch] = Some(c_ppm);
                }
            }
        }
        if ch_carrier_ppm[ch].is_none() {
            if let (Some(o_hz), Some(b)) = (
                a1.get(&format!("O{ch}"))
                    .and_then(|s| s.parse::<f64>().ok()),
                bf,
            ) {
                if b > 1.0 && o_hz.abs() > 1e-3 {
                    ch_carrier_ppm[ch] = Some(o_hz / b);
                }
            }
        }
    }

    let direct_group_delay_pts = resolve_bruker_group_delay(&a1);

    // Preprocess executed `pulseprogram` if present
    let pre = match inp.pulseprogram_text {
        Some(pp_src) => {
            if let Some(dir) = exp_dir {
                Some(preprocess_bruker_pulseprogram_in_dir(
                    pp_src,
                    zgoptns_raw,
                    dir,
                )?)
            } else {
                Some(preprocess_bruker_pulseprogram(
                    pp_src,
                    zgoptns_raw,
                    &inp.local_includes,
                )?)
            }
        }
        None => None,
    };

    let mut warnings = Vec::new();
    let executed_source_present = pre.is_some();
    let active_defines = pre
        .as_ref()
        .map(|p| p.defines.clone())
        .unwrap_or_else(|| parse_zgoptns(zgoptns_raw));
    let resolved_local_includes = pre
        .as_ref()
        .map(|p| p.resolved_includes.clone())
        .unwrap_or_default();
    let unresolved_standard_includes = pre
        .as_ref()
        .map(|p| p.unresolved_standard_includes.clone())
        .unwrap_or_default();
    let preprocessed_lines = pre.as_ref().map(|p| p.lines.len()).unwrap_or(0);

    // Indirect TD vector (`F1` = outermost `acqu3s` in 3D / `acqu2s` in 2D; `F2` = `acqu2s` in 3D)
    let td_f1 = match dimensionality {
        2 => a2
            .as_ref()
            .and_then(|m| m.get("TD"))
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0),
        3 => a3
            .as_ref()
            .and_then(|m| m.get("TD"))
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0),
        4 => a4
            .as_ref()
            .and_then(|m| m.get("TD"))
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0),
        _ => 0,
    };
    let td_f2 = match dimensionality {
        3 => a2
            .as_ref()
            .and_then(|m| m.get("TD"))
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0),
        4 => a3
            .as_ref()
            .and_then(|m| m.get("TD"))
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0),
        _ => 0,
    };
    let td_seq = [td_f1, td_f2, 0];
    let sw_h_of = |amap: Option<&HashMap<String, String>>| -> Option<f64> {
        let m = amap?;
        m.get("SW_h")
            .and_then(|s| s.parse::<f64>().ok())
            .filter(|&v| v > 0.0)
            .or_else(|| {
                let sw = m.get("SW")?.parse::<f64>().ok()?;
                let sfo = m.get("SFO1")?.parse::<f64>().ok()?;
                (sw > 0.0 && sfo > 0.0).then_some(sw * sfo)
            })
    };
    let inf1_s = match dimensionality {
        2 => sw_h_of(a2.as_ref()).map(|v| 1.0 / v),
        3 => sw_h_of(a3.as_ref()).map(|v| 1.0 / v),
        4 => sw_h_of(a4.as_ref()).map(|v| 1.0 / v),
        _ => None,
    };
    let inf2_s = match dimensionality {
        3 => sw_h_of(a2.as_ref()).map(|v| 1.0 / v),
        4 => sw_h_of(a3.as_ref()).map(|v| 1.0 / v),
        _ => None,
    };
    if let Some(ref p) = pre {
        if increments_s[0] <= 0.0 {
            if let Some(dw1) = inf1_s {
                let raw_d0_uses = p
                    .lines
                    .iter()
                    .filter(|l| {
                        let c = strip_bruker_semicolon_comment(l).trim();
                        !c.starts_with('"')
                            && !c.contains("caldel")
                            && !c.contains("MCWRK")
                            && contains_word(c, "d0")
                    })
                    .count();
                increments_s[0] = if raw_d0_uses >= 2 { dw1 * 0.5 } else { dw1 };
            }
        }
        if increments_s[10] <= 0.0 {
            if let Some(dw2) = inf2_s {
                let raw_d10_uses = p
                    .lines
                    .iter()
                    .filter(|l| {
                        let c = strip_bruker_semicolon_comment(l).trim();
                        !c.starts_with('"')
                            && !c.contains("caldel")
                            && !c.contains("MCWRK")
                            && contains_word(c, "d10")
                    })
                    .count();
                increments_s[10] = if raw_d10_uses >= 2 { dw2 * 0.5 } else { dw2 };
            }
        }
    }

    // Symbolic analysis of preprocessed lines
    let mut pp_aqseq_override: Option<AcquisitionOrder> = None;
    let mut label_lines: HashMap<String, usize> = HashMap::new();
    let mut exec_lines: Vec<(usize, String)> = Vec::new();
    let mut raw_loops: Vec<RawLoopSpan> = Vec::new();
    let mut mc_lines: Vec<String> = Vec::new();
    let mut named_delays_s: HashMap<String, f64> = HashMap::new();
    let mut comment_rev_overrides: [Option<bool>; 3] = [None, None, None];

    if let Some(ref p) = pre {
        let mut in_phase_section = false;
        for (idx, raw) in p.lines.iter().enumerate() {
            if let Some((_, cmt)) = raw.split_once(';') {
                let cmt_low = cmt.trim().to_ascii_lowercase();
                for (f_d, prefix) in [(1usize, "f1 reverse:"), (2usize, "f2 reverse:")] {
                    if let Some(rest) = cmt_low.strip_prefix(prefix) {
                        let val = rest.trim();
                        if val.starts_with("true") || val.starts_with("yes") {
                            comment_rev_overrides[f_d] = Some(true);
                        } else if val.starts_with("false") || val.starts_with("no") {
                            comment_rev_overrides[f_d] = Some(false);
                        }
                    }
                }
            }
            let code = strip_bruker_semicolon_comment(raw).trim();
            if code.is_empty() {
                continue;
            }
            // Quoted inline parameter equation `"d0=3u"`, `"d0=(in0/2-p1-...)"`, `"l3=(td1/2)"`, `"d26=2.3m"`, etc.
            if code.starts_with('"') && code.ends_with('"') && code.len() >= 3 {
                let eq = &code[1..code.len() - 1];
                if let Some((lhs, rhs)) = eq.split_once('=') {
                    let lhs_t = lhs.trim();
                    let lhs_low = lhs_t.to_ascii_lowercase();
                    let rhs_t = rhs.trim();
                    let rhs_low = rhs_t.to_ascii_lowercase();
                    if let Some(d_idx) = lhs_low
                        .strip_prefix('d')
                        .and_then(|x| x.parse::<usize>().ok())
                    {
                        let is_runtime_in_override = (d_idx == 0 && rhs_low.contains("in0"))
                            || (d_idx == 10 && rhs_low.contains("in10"));
                        let is_multi_half_dwell =
                            rhs_low.contains("cnst") && rhs_low.contains("*2+1)*in");
                        if d_idx < 64
                            && (delays_s[d_idx] <= 0.0
                                || is_multi_half_dwell
                                || is_runtime_in_override)
                        {
                            if let Some(sec) = eval_simple_delay_expr(
                                rhs_t,
                                &delays_s,
                                &increments_s,
                                &pulses_us,
                                &cnst,
                            ) {
                                delays_s[d_idx] = sec;
                            }
                        }
                    } else if let Some(in_idx) = lhs_low
                        .strip_prefix("in")
                        .and_then(|x| x.parse::<usize>().ok())
                    {
                        if in_idx < 64 {
                            let clean_r = rhs_low.replace(' ', "");
                            let clean_r = clean_r.trim_matches(|c| c == '(' || c == ')');
                            if let Some(after) = clean_r.strip_prefix("inf1") {
                                if let Some(dw1) = inf1_s {
                                    let div = after
                                        .strip_prefix('/')
                                        .and_then(|x| x.parse::<f64>().ok())
                                        .unwrap_or(1.0);
                                    if div > 0.0 {
                                        increments_s[in_idx] = dw1 / div;
                                    }
                                }
                            } else if let Some(after) = clean_r.strip_prefix("inf2") {
                                if let Some(dw2) = inf2_s {
                                    let div = after
                                        .strip_prefix('/')
                                        .and_then(|x| x.parse::<f64>().ok())
                                        .unwrap_or(1.0);
                                    if div > 0.0 {
                                        increments_s[in_idx] = dw2 / div;
                                    }
                                }
                            } else if let Some(sec) = eval_simple_delay_expr(
                                rhs_t,
                                &delays_s,
                                &increments_s,
                                &pulses_us,
                                &cnst,
                            ) {
                                increments_s[in_idx] = sec;
                            }
                        }
                    } else if let Some(l_idx) = lhs_low
                        .strip_prefix('l')
                        .and_then(|x| x.parse::<usize>().ok())
                    {
                        if l_idx < 32 && loop_counters[l_idx] == 0 {
                            if rhs_t.contains("td1/2") && td_f1 >= 2 {
                                loop_counters[l_idx] = td_f1 / 2;
                            } else if rhs_t.contains("td2/2") && td_f2 >= 2 {
                                loop_counters[l_idx] = td_f2 / 2;
                            } else if let Ok(v) = rhs_t
                                .trim_matches(|c| c == '(' || c == ')')
                                .parse::<usize>()
                            {
                                loop_counters[l_idx] = v;
                            }
                        }
                    } else if let Some(sec) =
                        eval_simple_delay_expr(rhs_t, &delays_s, &increments_s, &pulses_us, &cnst)
                    {
                        named_delays_s.insert(lhs_t.to_string(), sec);
                    }
                }
                continue;
            }

            let low = code.to_ascii_lowercase();
            if let Some(rest) = low.strip_prefix("aqseq") {
                let tok = rest.split_whitespace().next().unwrap_or("");
                if tok == "312" {
                    pp_aqseq_override = Some(AcquisitionOrder::Aqseq312);
                } else if tok == "321" {
                    pp_aqseq_override = Some(AcquisitionOrder::Aqseq321);
                }
                continue;
            }
            if low.starts_with("prosol ") || low.starts_with("define ") {
                continue;
            }

            // Phase table `phK = ...`
            if let Some(rest) = low.strip_prefix("ph") {
                let d_len = rest.bytes().take_while(|b| b.is_ascii_digit()).count();
                if d_len > 0 {
                    let after_d = rest[d_len..].trim_start();
                    if after_d.starts_with('=') {
                        in_phase_section = true;
                        continue;
                    }
                }
            }
            if in_phase_section {
                continue;
            }

            // Leading label (`1 ze`, `2 d1`, `lbl: ...`)
            let mut stmt = code;
            let first_tok = stmt.split_whitespace().next().unwrap_or("");
            let is_symbolic_lbl = (first_tok.ends_with(':') || first_tok.ends_with(','))
                && first_tok.len() > 1
                && first_tok[..first_tok.len() - 1]
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'_');
            if !first_tok.is_empty()
                && (first_tok.bytes().all(|b| b.is_ascii_digit()) || is_symbolic_lbl)
            {
                let lbl = first_tok.trim_end_matches([':', ',']).to_string();
                label_lines.entry(lbl).or_insert(exec_lines.len());
                stmt = stmt[first_tok.len()..].trim_start();
            }

            let cur_exec_idx = exec_lines.len();
            exec_lines.push((idx + 1, stmt.to_string()));

            let stmt_low = stmt.to_ascii_lowercase();
            if stmt_low.contains("mc #") || stmt_low.starts_with("mc ") {
                mc_lines.push(stmt.to_string());
            }

            // Parse `lo to <lbl> times <count>`
            if let Some(lo_pos) = first_word_pos(&stmt_low, "lo") {
                let tail = stmt[lo_pos + 2..].trim_start();
                if let Some(after_to) = tail
                    .strip_prefix("to ")
                    .or_else(|| tail.strip_prefix("TO "))
                {
                    let mut parts = after_to.split_whitespace();
                    let target_lbl = parts
                        .next()
                        .unwrap_or("")
                        .trim_end_matches([':', ','])
                        .to_string();
                    let times_kw = parts.next().unwrap_or("");
                    if times_kw.eq_ignore_ascii_case("times") {
                        let cnt_expr: String = parts.collect::<Vec<_>>().join("");
                        let start_exec = label_lines.get(&target_lbl).copied().unwrap_or(0);
                        let start_src_line = exec_lines.get(start_exec).map(|x| x.0).unwrap_or(1);
                        // Collect ops strictly in this loop's own increment tail (after any
                        // already-closed inner `lo to` loop ending at or after `start_src_line`,
                        // whether jumping to the same numeric label `2` or nested TopSpin `mc`
                        // labels `LBLSTS2, LBLF2, LBLSTS1, LBLF1,`).
                        let prev_same_end = raw_loops
                            .iter()
                            .rev()
                            .find(|prev_l| prev_l.end_line >= start_src_line)
                            .map(|prev_l| {
                                exec_lines
                                    .iter()
                                    .position(|(ln, _)| *ln == prev_l.end_line)
                                    .map(|p_idx| p_idx + 1)
                                    .unwrap_or(start_exec)
                            })
                            .unwrap_or(start_exec);

                        let mut phase_ops = Vec::new();
                        let mut delay_ops = Vec::new();
                        let mut grad_ops = Vec::new();
                        let mut list_ops = Vec::new();
                        for (_, body_stmt) in &exec_lines[prev_same_end..=cur_exec_idx] {
                            extract_loop_ops_from_line(
                                body_stmt,
                                &mut phase_ops,
                                &mut delay_ops,
                                &mut grad_ops,
                                &mut list_ops,
                            );
                        }
                        let resolved_count =
                            resolve_loop_count_expr(&cnt_expr, &loop_counters, &td_seq);
                        raw_loops.push(RawLoopSpan {
                            label: target_lbl,
                            count_expr: cnt_expr,
                            resolved_count,
                            start_line: start_src_line,
                            end_line: idx + 1,
                            phase_ops,
                            delay_ops,
                            grad_ops,
                            list_ops,
                        });
                    }
                }
            }
        }
    }

    let loops = nest_loop_spans(raw_loops.clone());

    // Resolve `AQSEQ` (`321` vs `312`):
    // 1. Manual `lo to` loops (`mc_lines.is_empty()`) execute hard-coded hardware loop order
    //    regardless of template `##$AQSEQ` (`0` or `1`) in `acqus` (e.g., XWIN-NMR 2.x `hncof.s`
    //    with `##$AQSEQ= 1` while `l4` (`d10`, F2) runs inner to `l6` (`d0`, F1) = `Aqseq321`,
    //    or `d0` (F1) inner to `d10` (F2) under `##$AQSEQ= 0` = `Aqseq312`).
    // 2. Otherwise explicit `##$AQSEQ= 1` -> `Aqseq312`, `##$AQSEQ= 0` -> `Aqseq321`.
    // 3. When `##$AQSEQ` is omitted, fall back to `aqseq 312`/`aqseq 321` directive or loop order.
    let header_aqseq = a1.get("AQSEQ").and_then(|s| s.parse::<i32>().ok());
    let symbolic_loop_aqseq = detect_symbolic_loop_aqseq(&raw_loops);
    let aqseq = match dimensionality {
        1 => AcquisitionOrder::Dim1D,
        2 => AcquisitionOrder::Dim2D,
        3 => {
            if mc_lines.is_empty() {
                if let Some(loop_order) = symbolic_loop_aqseq {
                    if (header_aqseq == Some(0) && loop_order == AcquisitionOrder::Aqseq312)
                        || (header_aqseq == Some(1) && loop_order == AcquisitionOrder::Aqseq321)
                    {
                        warnings.push(format!(
                            "manual `lo to` loop nesting specifies `{loop_order:?}` while acqus has ##$AQSEQ={}; honoring executed hardware loop order",
                            header_aqseq.unwrap_or(-1)
                        ));
                    }
                    loop_order
                } else {
                    match header_aqseq {
                        Some(1) => AcquisitionOrder::Aqseq312,
                        Some(0) => {
                            if pp_aqseq_override == Some(AcquisitionOrder::Aqseq312) {
                                warnings.push(
                                    "pulseprogram specifies `aqseq 312` while acqus ##$AQSEQ=0"
                                        .to_string(),
                                );
                            }
                            AcquisitionOrder::Aqseq321
                        }
                        _ => pp_aqseq_override.unwrap_or(AcquisitionOrder::Aqseq321),
                    }
                }
            } else {
                match header_aqseq {
                    Some(1) => AcquisitionOrder::Aqseq312,
                    Some(0) => {
                        if pp_aqseq_override == Some(AcquisitionOrder::Aqseq312) {
                            warnings.push(
                                "pulseprogram specifies `aqseq 312` while acqus ##$AQSEQ=0"
                                    .to_string(),
                            );
                        }
                        AcquisitionOrder::Aqseq321
                    }
                    _ => pp_aqseq_override
                        .or(symbolic_loop_aqseq)
                        .unwrap_or(AcquisitionOrder::Aqseq321),
                }
            }
        }
        _ => AcquisitionOrder::Dim4D,
    };

    // Analyze per-indirect-dimension symbolic loop quadrature, `EA`/`GPZ` polarity, and SCT order
    let mut dim_sym_f1 = analyze_bruker_symbolic_dim(
        1,
        &exec_lines,
        &raw_loops,
        &mc_lines,
        &gpz_pct,
        &pulses_us,
        &ch_nuclei,
    );
    dim_sym_f1.comment_reverse_override = comment_rev_overrides[1];
    let mut dim_sym_f2 = analyze_bruker_symbolic_dim(
        2,
        &exec_lines,
        &raw_loops,
        &mc_lines,
        &gpz_pct,
        &pulses_us,
        &ch_nuclei,
    );
    dim_sym_f2.comment_reverse_override = comment_rev_overrides[2];

    let mut fq_extra_jumps_ppm: Vec<(String, f64)> = Vec::new();
    let (fq_sfo_f1, mut fq_jumps_f1) = resolve_bruker_per_scan_fq_sfo(
        1,
        &exec_lines,
        &raw_loops,
        &a1,
        &ch_nuclei,
        inp.fqlist_text,
        exp_dir,
    )?;
    dim_sym_f1.per_scan_fq_sfo_mhz = fq_sfo_f1;
    fq_extra_jumps_ppm.append(&mut fq_jumps_f1);
    let (fq_sfo_f2, mut fq_jumps_f2) = resolve_bruker_per_scan_fq_sfo(
        2,
        &exec_lines,
        &raw_loops,
        &a1,
        &ch_nuclei,
        inp.fqlist_text,
        exp_dir,
    )?;
    dim_sym_f2.per_scan_fq_sfo_mhz = fq_sfo_f2;
    fq_extra_jumps_ppm.append(&mut fq_jumps_f2);

    // Build `IndirectDimPhysics` in acquisition parameter file order:
    // - 2D: `[acqu2s (F1)]`
    // - 3D: `[acqu2s (F2), acqu3s (F1)]` (matching `fid.rs` `axis_y = acqu2s`, `axis_z = acqu3s`)
    // - 4D: `[acqu2s (F3), acqu3s (F2), acqu4s (F1)]`
    let mut indirect_dims = Vec::new();
    match dimensionality {
        2 => {
            if let Some(ref m2) = a2 {
                indirect_dims.push(build_bruker_indirect_dim_physics(
                    1,
                    "acqu2s",
                    &a1,
                    m2,
                    h1_base_mhz,
                    &delays_s,
                    &increments_s,
                    &pulses_us,
                    &exec_lines,
                    &dim_sym_f1,
                    &pulprog_name,
                )?);
            }
        }
        3 => {
            // In a pseudo-3D series (`vdlist`/`vclist`/`fqlist`/`difflist` present and no `d10`
            // second chemical-shift evolution delay), one indirect axis is a `QF` delay/counter
            // ramp (`FnMODE == 1`) and the other is the single frequency-evolved `t_1` (`d0`) axis!
            let has_pseudo_list = inp.vdlist_text.is_some_and(|s| !s.trim().is_empty())
                || inp.vclist_text.is_some_and(|s| !s.trim().is_empty())
                || inp.fqlist_text.is_some_and(|s| !s.trim().is_empty())
                || inp.difflist_text.is_some_and(|s| !s.trim().is_empty());
            let has_d10_evo = exec_lines.iter().any(|(_, s)| {
                contains_word(s, "d10") || contains_word(s, "id10") || contains_word(s, "dd10")
            });
            let fn2 = a2
                .as_ref()
                .and_then(|m| m.get("FnMODE"))
                .and_then(|s| s.parse::<i32>().ok())
                .unwrap_or(0);
            let fn3 = a3
                .as_ref()
                .and_then(|m| m.get("FnMODE"))
                .and_then(|s| s.parse::<i32>().ok())
                .unwrap_or(0);
            let empty_sym = SymbolicDimLoopInfo::default();
            let (sym_for_a2, sym_for_a3): (&SymbolicDimLoopInfo, &SymbolicDimLoopInfo) =
                if has_pseudo_list && !has_d10_evo {
                    if fn3 == 1 && fn2 >= 4 {
                        (&dim_sym_f1, &empty_sym)
                    } else if fn2 == 1 && fn3 >= 4 {
                        (&empty_sym, &dim_sym_f1)
                    } else {
                        (&dim_sym_f2, &dim_sym_f1)
                    }
                } else {
                    (&dim_sym_f2, &dim_sym_f1)
                };

            if let Some(ref m2) = a2 {
                indirect_dims.push(build_bruker_indirect_dim_physics(
                    if std::ptr::eq(sym_for_a2, &dim_sym_f1) {
                        1
                    } else {
                        2
                    },
                    "acqu2s",
                    &a1,
                    m2,
                    h1_base_mhz,
                    &delays_s,
                    &increments_s,
                    &pulses_us,
                    &exec_lines,
                    sym_for_a2,
                    &pulprog_name,
                )?);
            }
            if let Some(ref m3) = a3 {
                indirect_dims.push(build_bruker_indirect_dim_physics(
                    1,
                    "acqu3s",
                    &a1,
                    m3,
                    h1_base_mhz,
                    &delays_s,
                    &increments_s,
                    &pulses_us,
                    &exec_lines,
                    sym_for_a3,
                    &pulprog_name,
                )?);
            }
        }
        4 => {
            let dummy_sym = SymbolicDimLoopInfo::default();
            if let Some(ref m2) = a2 {
                indirect_dims.push(build_bruker_indirect_dim_physics(
                    3,
                    "acqu2s",
                    &a1,
                    m2,
                    h1_base_mhz,
                    &delays_s,
                    &increments_s,
                    &pulses_us,
                    &exec_lines,
                    &dummy_sym,
                    &pulprog_name,
                )?);
            }
            if let Some(ref m3) = a3 {
                indirect_dims.push(build_bruker_indirect_dim_physics(
                    2,
                    "acqu3s",
                    &a1,
                    m3,
                    h1_base_mhz,
                    &delays_s,
                    &increments_s,
                    &pulses_us,
                    &exec_lines,
                    &dim_sym_f2,
                    &pulprog_name,
                )?);
            }
            if let Some(ref m4) = a4 {
                indirect_dims.push(build_bruker_indirect_dim_physics(
                    1,
                    "acqu4s",
                    &a1,
                    m4,
                    h1_base_mhz,
                    &delays_s,
                    &increments_s,
                    &pulses_us,
                    &exec_lines,
                    &dim_sym_f1,
                    &pulprog_name,
                )?);
            }
        }
        _ => {}
    }

    // Detect Pseudo-3D storage order (`PlaneQueued` vs `T1Interleaved`) and inner loop nesting
    let (pseudo_3d_order, pseudo_3d_list_kind) = detect_bruker_pseudo3d_order(
        &raw_loops,
        &exec_lines,
        &mc_lines,
        &indirect_dims,
        inp,
        &pulprog_name,
    );
    let pseudo_3d_list_inner_to_quad = if pseudo_3d_order == Pseudo3DStorageOrder::T1Interleaved {
        let list_lp = raw_loops.iter().find(|l| {
            l.list_ops
                .iter()
                .any(|o| matches!(o.as_str(), "ivd" | "ivc") || o.starts_with("ifq"))
        });
        let quad_lp = raw_loops.iter().find(|l| {
            l.resolved_count == Some(2) && (!l.phase_ops.is_empty() || !l.grad_ops.is_empty())
        });
        match (list_lp, quad_lp) {
            (Some(ll), Some(ql)) => Some(ll.end_line < ql.end_line),
            _ => None,
        }
    } else {
        None
    };

    // Decode `nuslist` if present
    let nus_schedule = if let Some(nus_txt) = inp.nuslist_text {
        let mut grid_dims = Vec::new();
        for (idx, d) in indirect_dims.iter().enumerate() {
            if d.quad_mode == DecodedQuadMode::Qf {
                continue;
            }
            let amap = match idx {
                0 => a2.as_ref(),
                1 => a3.as_ref(),
                _ => a4.as_ref(),
            };
            let nustd = amap
                .and_then(|m| m.get("NusTD"))
                .and_then(|s| s.parse::<usize>().ok())
                .filter(|&v| v > 1)
                .or_else(|| td_indirect_arr.get(idx + 1).copied().filter(|&v| v > 1))
                .unwrap_or(d.td_raw);
            grid_dims.push((nustd / 2).max(d.complex_points).max(1));
        }
        if !grid_dims.is_empty() {
            Some(decode_nuslist_str(nus_txt, &grid_dims, aqseq)?)
        } else {
            None
        }
    } else {
        None
    };

    // Deduce active nuclei, `CNST` carrier jumps, and `MagnetizationTransferGraph`
    let transfer_graph = build_bruker_transfer_graph(
        dimensionality,
        &direct_nucleus,
        &ch_nuclei,
        &ch_carrier_ppm,
        &fq_extra_jumps_ppm,
        &cnst,
        &delays_s,
        &named_delays_s,
        &spnam,
        &cpdprg,
        &gpz_pct,
        &indirect_dims,
        &exec_lines,
        &raw_loops,
        pseudo_3d_order,
        pseudo_3d_list_kind.as_deref(),
        &active_defines,
        &pulprog_name,
    );

    Ok(PulprogInterpretation {
        vendor: PulprogVendor::Bruker,
        pulse_program_name: pulprog_name,
        executed_source_present,
        active_defines,
        resolved_local_includes,
        unresolved_standard_includes,
        preprocessed_lines,
        dimensionality,
        aqseq,
        pseudo_3d_order,
        pseudo_3d_list_kind,
        pseudo_3d_list_inner_to_quad,
        varian_array_nesting: Vec::new(),
        direct_nucleus,
        direct_group_delay_pts,
        cnst,
        delays_s,
        increments_s,
        loop_counters,
        gpz_pct,
        spnam,
        cpdprg,
        loops,
        indirect_dims,
        nus_schedule,
        transfer_graph,
        warnings,
    })
}

fn eval_simple_delay_expr(
    rhs: &str,
    delays_s: &[f64],
    increments_s: &[f64],
    pulses_us: &[f64],
    cnst: &[f64],
) -> Option<f64> {
    let t = rhs.trim();
    if let Some(s) = parse_bruker_time_literal_s(t) {
        return Some(s);
    }
    // `1s/(cnst4*4)` or `1s/(cnst4*2)`
    let low = t.replace(' ', "").to_ascii_lowercase();
    if let Some(inner) = low.strip_prefix("1s/(").and_then(|s| s.strip_suffix(')')) {
        if let Some((a, b)) = inner.split_once('*') {
            let val_a = resolve_cnst_or_num(a, cnst)?;
            let val_b = resolve_cnst_or_num(b, cnst)?;
            let denom = val_a * val_b;
            if denom.abs() > 1e-9 {
                return Some(1.0 / denom);
            }
        } else if let Some(v) = resolve_cnst_or_num(inner, cnst) {
            if v.abs() > 1e-9 {
                return Some(1.0 / v);
            }
        }
    }
    let mut outer_div = 1.0_f64;
    let mut expr_body = low.as_str();
    if expr_body.starts_with('(') {
        if let Some(pos) = expr_body.rfind(")/") {
            if let Ok(dv) = expr_body[pos + 2..].parse::<f64>() {
                if dv > 0.0 {
                    outer_div = dv;
                    expr_body = &expr_body[1..pos];
                }
            }
        }
    }
    let clean = expr_body.trim_start_matches('(').trim_end_matches(')');
    // Pre-compensated multi-half-dwell runtime expression `"d0=((cnst0*2+1)*in0-...)/2"`
    if let Some(rest_cnst) = clean.strip_prefix("cnst") {
        let k_len = rest_cnst.bytes().take_while(|b| b.is_ascii_digit()).count();
        if k_len > 0 {
            if let Ok(c_idx) = rest_cnst[..k_len].parse::<usize>() {
                let after_k = &rest_cnst[k_len..];
                if let Some(after_mul) = after_k.strip_prefix("*2+1)*") {
                    for (in_tok, in_idx) in [("in10", 10usize), ("in0", 0usize)] {
                        if after_mul.starts_with(in_tok) {
                            if let (Some(&c_val), Some(&inc_s)) =
                                (cnst.get(c_idx), increments_s.get(in_idx))
                            {
                                if (0.0..=4.0).contains(&c_val) && inc_s > 0.0 {
                                    return Some(((2.0 * c_val + 1.0) * inc_s) / outer_div);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // Half-dwell runtime expression `"d0=(in0/2-p1-(p21*2/3.14159))"` or `"(in0-...)/2"`
    for (in_tok, in_idx) in [("in10", 10usize), ("in0", 0usize)] {
        if let Some(after_in) = clean.strip_prefix(in_tok) {
            if let Some(&inc_s) = increments_s.get(in_idx) {
                if inc_s > 0.0 {
                    if let Some(after_div) = after_in.strip_prefix('/') {
                        let d_len = after_div
                            .bytes()
                            .take_while(|b| b.is_ascii_digit() || *b == b'.')
                            .count();
                        if d_len > 0 {
                            if let Ok(dv) = after_div[..d_len].parse::<f64>() {
                                if dv > 0.0 {
                                    return Some(inc_s / (dv * outer_div));
                                }
                            }
                        }
                    }
                    return Some(inc_s / outer_div);
                }
            }
        }
    }
    // Leading `d<K>` term (e.g. `"d23-d21-p26"`, `"d23/2-p14/2"`)
    let first_atom = clean.split(['+', '-']).next().unwrap_or("");
    if let Some((base_d, div)) = first_atom.split_once('/') {
        if let (Some(d_idx), Ok(dv)) = (
            base_d
                .strip_prefix('d')
                .and_then(|x| x.parse::<usize>().ok()),
            div.parse::<f64>(),
        ) {
            if d_idx < delays_s.len() && delays_s[d_idx] > 0.0 && dv > 0.0 {
                return Some(delays_s[d_idx] / dv);
            }
        }
    } else if let Some(d_idx) = first_atom
        .strip_prefix('d')
        .and_then(|x| x.parse::<usize>().ok())
    {
        if d_idx < delays_s.len() && delays_s[d_idx] > 0.0 {
            return Some(delays_s[d_idx]);
        }
    } else if let Some(p_idx) = first_atom
        .strip_prefix('p')
        .and_then(|x| x.parse::<usize>().ok())
    {
        if p_idx < pulses_us.len() && pulses_us[p_idx] > 0.0 {
            return Some(pulses_us[p_idx] * 1e-6);
        }
    }
    None
}

fn resolve_cnst_or_num(s: &str, cnst: &[f64]) -> Option<f64> {
    if let Ok(v) = s.parse::<f64>() {
        return Some(v);
    }
    if let Some(idx) = s.strip_prefix("cnst").and_then(|x| x.parse::<usize>().ok()) {
        return cnst.get(idx).copied().filter(|&v| v.abs() > 1e-9);
    }
    None
}

fn detect_symbolic_loop_aqseq(raw_loops: &[RawLoopSpan]) -> Option<AcquisitionOrder> {
    let mut f1_end = None;
    let mut f2_end = None;
    for lp in raw_loops {
        if lp.delay_ops.iter().any(|d| d == "id0" || d == "dd0") {
            f1_end = Some(lp.end_line);
        }
        if lp.delay_ops.iter().any(|d| d == "id10" || d == "dd10") {
            f2_end = Some(lp.end_line);
        }
    }
    match (f1_end, f2_end) {
        (Some(e1), Some(e2)) if e1 < e2 => Some(AcquisitionOrder::Aqseq312),
        (Some(e1), Some(e2)) if e2 < e1 => Some(AcquisitionOrder::Aqseq321),
        _ => None,
    }
}

/// Extract a canonical lowercase 180° refocusing pulse token (e.g. `"p14:sp3"`, `"cab_180"`,
/// `"c13_180"`, `"p14"`, `"p22"`, `"p2"`) from an executable Bruker statement inside a spin echo.
fn extract_180_pulse_token(stmt: &str) -> Option<String> {
    let low = stmt.to_ascii_lowercase();
    for raw_tok in low.split(|c: char| c.is_whitespace() || matches!(c, '(' | ')' | ',')) {
        let tok = raw_tok.trim();
        if tok.is_empty() {
            continue;
        }
        if tok.contains("180") {
            return Some(tok.split(':').next().unwrap_or(tok).to_string());
        }
        if tok.starts_with("p14:sp") || tok.starts_with("p22:sp") || tok.starts_with("p2:sp") {
            return Some(tok.to_string());
        }
        if matches!(tok, "p14" | "p22" | "p2" | "p8" | "p4") {
            return Some(tok.to_string());
        }
    }
    None
}

fn analyze_bruker_symbolic_dim(
    f_dim: usize,
    exec_lines: &[(usize, String)],
    raw_loops: &[RawLoopSpan],
    mc_lines: &[String],
    gpz_pct: &[f64],
    pulses_us: &[f64],
    ch_nuclei: &[String; 5],
) -> SymbolicDimLoopInfo {
    let mut info = SymbolicDimLoopInfo {
        ea_explicit_mult: 1.0,
        ea_dur_ratio: 1.0,
        ..SymbolicDimLoopInfo::default()
    };

    let (inc_d, dec_d, alt_dec_evo, alt_inc_comp, ph_macro, ea_macro, evo_tok, dec_tok) =
        if f_dim == 1 {
            ("id0", "dd20", "dd0", "id20", "F1PH", "F1EA", "d0", "d20")
        } else {
            ("id10", "dd30", "dd10", "id30", "F2PH", "F2EA", "d10", "d30")
        };

    // 1. Check `mc` macro lines (`F1PH`, `F2PH`, `F1EA`, `F2EA`)
    let joined_mc = mc_lines.join(" ");
    if let Some((reg, sign)) = extract_calph_macro_90(&joined_mc, ph_macro) {
        info.quad_reg_sign = Some((reg, sign));
    }
    if joined_mc.contains(ea_macro) {
        info.has_igrad_ea = true;
    }
    if joined_mc.contains(&format!("caldel({dec_tok}, -")) {
        info.constant_time = true;
    }

    // 2. Check precompiled `MCWRK` lines (`t1loop` / `t2loop`) BEFORE `raw_loops` so NUS `SPARSELIST`
    // precompiled sequences associate the exact `ph` register per indirect dimension.
    let precomp_tok = if f_dim == 1 { "t1loop" } else { "t2loop" };
    for (idx, (_, stmt)) in exec_lines.iter().enumerate() {
        if contains_word(stmt, dec_d) && contains_word(stmt, inc_d) {
            info.constant_time = true;
        }
        if contains_word(stmt, alt_dec_evo) && contains_word(stmt, alt_inc_comp) {
            info.constant_time = true;
            info.evo_decremented = true;
        }
        if stmt.contains(precomp_tok) && info.quad_reg_sign.is_none() {
            for (_, follow) in exec_lines.iter().skip(idx + 1).take(6) {
                if let Some(found) = follow
                    .split(|c: char| c.is_whitespace() || matches!(c, ',' | '(' | ')' | ':' | '+'))
                    .find_map(parse_90deg_ip_dp)
                {
                    info.quad_reg_sign = Some(found);
                    break;
                }
            }
        }
    }

    // 3. Check explicit `lo to <lbl> times ...` loops for this indirect dimension
    for (lp_idx, lp) in raw_loops.iter().enumerate() {
        let touches_dim = lp
            .delay_ops
            .iter()
            .any(|op| op == inc_d || op == alt_dec_evo);
        if !touches_dim {
            continue;
        }
        info.loop_nesting_rank = Some(lp_idx);
        if lp.delay_ops.iter().any(|op| op == dec_d) {
            info.constant_time = true;
        }
        if lp.delay_ops.iter().any(|op| op == alt_dec_evo) {
            info.evo_decremented = true;
            if lp.delay_ops.iter().any(|op| op == alt_inc_comp) {
                info.constant_time = true;
            }
        }
        let outer_has_tppi = lp
            .phase_ops
            .iter()
            .any(|op| op.ends_with("*2") && (op.starts_with("ip") || op.starts_with("dp")));
        info.has_tppi_toggle |= outer_has_tppi;

        // Look for an immediately preceding 2-step quadrature loop (`resolved_count == Some(2)` or `count_expr == "2"`)!
        if lp_idx > 0 {
            let prev = &raw_loops[lp_idx - 1];
            let is_two_step = prev.resolved_count == Some(2) || prev.count_expr.trim() == "2";
            if is_two_step && prev.end_line < lp.end_line {
                let prev_igrad = !prev.grad_ops.is_empty();
                let prev_90_all: Vec<(usize, i32)> = prev
                    .phase_ops
                    .iter()
                    .filter_map(|op| parse_90deg_ip_dp(op))
                    .collect();
                if prev_igrad {
                    info.has_igrad_ea = true;
                    info.synthesized_quad = Some(DecodedQuadMode::EchoAntiecho);
                } else if let Some(&(reg, sign)) = prev_90_all.first() {
                    info.quad_reg_sign = Some((reg, sign));
                    let (_, rec_qturns) = phase_ops_quarter_turns(&lp.phase_ops, 31);
                    let rec_toggles_180 = rec_qturns.rem_euclid(4) == 2;
                    let (rf_reset, rf_qturns) = phase_ops_quarter_turns(&lp.phase_ops, reg);
                    // Executing `ip<reg>`/`dp<reg>` twice in `prev` advances `ph<reg>` by 180° per `t1`
                    // step unless `lp` cancels it (`rp<reg>` or `ip<reg>*2`/`dp<reg>*2`). When `ph<reg>`
                    // IS uncancelled AND `ph31` ALSO toggles by 180° (`2 * ip31`), `(-1)_RF * (-1)_rec = +1`
                    // = pure `States` (`States-TPPI` requires `rec_toggles_180 && rf_cancelled_in_outer`).
                    let rf_cancelled_in_outer = rf_reset || rf_qturns.rem_euclid(4) == 2;
                    info.synthesized_quad = Some(if rec_toggles_180 && rf_cancelled_in_outer {
                        DecodedQuadMode::StatesTppi
                    } else {
                        DecodedQuadMode::States
                    });

                    // Paired post-evolution readout echo `(90_{phA}) - tau - (180_{+x}) - tau - (90_{phB})`
                    // (e.g. Abildgaard `hncacbf.s` / `hncocacbf.s` `ip16` + `ip17`, vs TopSpin `hncacbgp3d`
                    // which places an additional `(p14:sp3 ph1):f2` 180° refocusing pulse between the last
                    // `d0` and `ph9`!): count parity `(1 + pre_refocus_cnt) % 2 == 1` of the echo's own
                    // 180° refocusing pulse token between the last `evo_tok` (`d0`/`d10`) and `phA`.
                    if prev_90_all.len() >= 2 && prev_90_all[0].0 != prev_90_all[1].0 {
                        let r_a = format!("ph{}", prev_90_all[0].0);
                        let r_b = format!("ph{}", prev_90_all[1].0);
                        let last_evo_l = exec_lines.iter().rposition(|(_, s)| {
                            !s.contains("MCWRK")
                                && !s.contains("caldel")
                                && !s.trim_start().starts_with('"')
                                && !s.starts_with("define ")
                                && contains_word(s, evo_tok)
                        });
                        let a_l = exec_lines.iter().position(|(_, s)| {
                            !s.trim_start().starts_with('"') && contains_word(s, &r_a)
                        });
                        let b_l = exec_lines.iter().position(|(_, s)| {
                            !s.trim_start().starts_with('"') && contains_word(s, &r_b)
                        });
                        if let (Some(el), Some(al), Some(bl)) = (last_evo_l, a_l, b_l) {
                            if al > el && bl > al {
                                let echo_180_tok = exec_lines[al + 1..bl]
                                    .iter()
                                    .find_map(|(_, s)| extract_180_pulse_token(s));
                                if let Some(ref tok180) = echo_180_tok {
                                    let pre_refocus_cnt = exec_lines[el + 1..al]
                                        .iter()
                                        .filter(|(_, s)| {
                                            s.to_ascii_lowercase().contains(tok180.as_str())
                                        })
                                        .count();
                                    if (1 + pre_refocus_cnt) % 2 == 1 {
                                        info.paired_post_evo_echo_quad = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } else {
            // Single-loop TPPI (`ipK` + `id0` in the same `td1` loop without a `times 2` inner loop)
            if let Some((reg, sign)) = lp.phase_ops.iter().find_map(|op| parse_90deg_ip_dp(op)) {
                info.quad_reg_sign.get_or_insert((reg, sign));
            }
        }
    }

    let all_exec = exec_lines
        .iter()
        .map(|(_, s)| s.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    let no_space_exec = all_exec.replace(' ', "").to_ascii_lowercase();
    let no_space_mc = mc_lines.join(" ").replace(' ', "").to_ascii_lowercase();
    let sparselist_ct = if f_dim == 1 {
        no_space_exec.contains("-t1loop*in20")
            || no_space_exec.contains("d20=d20orig-t1loop")
            || no_space_exec.contains("caldel(d20,-in20)")
            || no_space_mc.contains("caldel(d20,-in20)")
    } else {
        no_space_exec.contains("-t2loop*in30")
            || no_space_exec.contains("d30=d30orig-t2loop")
            || no_space_exec.contains("caldel(d30,-in30)")
            || no_space_mc.contains("caldel(d30,-in30)")
    };
    if (contains_word(&all_exec, inc_d) && contains_word(&all_exec, dec_d)) || sparselist_ct {
        info.constant_time = true;
        // Find positions in RF body (excluding `MCWRK` / `caldel` / quoted equations)
        let rf_only = exec_lines
            .iter()
            .map(|(_, s)| s.as_str())
            .filter(|s| {
                !s.contains("MCWRK")
                    && !s.contains("caldel")
                    && !s.trim_start().starts_with('"')
                    && !contains_word(s, inc_d)
            })
            .collect::<Vec<_>>()
            .join("\n");
        if let (Some(pe), Some(pd)) = (
            first_word_pos(&rf_only, evo_tok),
            first_word_pos(&rf_only, dec_tok),
        ) {
            info.sct_inc_before_dec = Some(pe < pd);
        }
    }

    // 4. Analyze `gp<enc>*EA` (including bipolar `gp<enc>*EA ... 180 ... gp<enc>*-1*EA` pairs!)
    //    and decoding gradient `gp<dec>` polarity
    let ea_tag = if f_dim == 2 && all_exec.contains("EA2") {
        "EA2"
    } else {
        "EA"
    };
    let mut enc_events: Vec<(usize, usize, f64, f64)> = Vec::new(); // (line_idx, gp_idx, dur_us, mult)
    for (l_idx, (_, stmt)) in exec_lines.iter().enumerate() {
        if let Some((gp_i, dur_us, mult)) = parse_gp_instruction(stmt, Some(ea_tag), pulses_us) {
            enc_events.push((l_idx, gp_i, dur_us, mult));
        }
    }
    if let Some(&(first_line, enc_gp, _, first_mult)) = enc_events.first() {
        let last_line = enc_events
            .iter()
            .filter(|&&(_, g, _, _)| g == enc_gp)
            .map(|&(l, _, _, _)| l)
            .max()
            .unwrap_or(first_line);
        let total_enc_dur: f64 = enc_events
            .iter()
            .filter(|&&(_, g, _, _)| g == enc_gp)
            .map(|&(_, _, d, _)| d)
            .sum();
        info.has_igrad_ea = true;
        info.ea_grad_idx = Some(enc_gp);
        info.ea_explicit_mult = first_mult;

        // Expected coherence-selection ratio `|gamma_H / gamma_X|` (`~9.87` for 15N, `~3.98` for 13C)
        let has_15n = ch_nuclei.iter().any(|n| n == "15N");
        let has_13c = ch_nuclei.iter().any(|n| n == "13C");
        let enc_val = gpz_pct.get(enc_gp).copied().unwrap_or(0.0);
        if enc_val.abs() > 1e-6 {
            let mut best_dec: Option<(usize, f64, f64)> = None;
            for (_, stmt) in exec_lines.iter().skip(last_line + 1) {
                if contains_word(stmt, "go") || stmt.starts_with("go=") {
                    break;
                }
                if let Some((dec_gp, dec_dur, _)) = parse_gp_instruction(stmt, None, pulses_us) {
                    if dec_gp == enc_gp {
                        continue;
                    }
                    let dec_val = gpz_pct.get(dec_gp).copied().unwrap_or(0.0);
                    if dec_val.abs() <= 1e-6 {
                        continue;
                    }
                    let dur_ratio = total_enc_dur.max(1.0) / dec_dur.max(1.0);
                    let area_ratio = (enc_val.abs() / dec_val.abs()) * dur_ratio;
                    // Match `~9.87` (or `~4.93` when bipolar double-decoding gradient) for 15N,
                    // or `~3.98` (or `~1.99`) for 13C!
                    let mut dist = 999.0_f64;
                    if has_15n {
                        dist = dist
                            .min((area_ratio - 9.87).abs())
                            .min((area_ratio * 2.0 - 9.87).abs());
                    }
                    if has_13c {
                        dist = dist
                            .min((area_ratio - 3.98).abs())
                            .min((area_ratio * 2.0 - 3.98).abs());
                    }
                    if !has_15n && !has_13c {
                        dist = (area_ratio - 9.87).abs().min((area_ratio - 3.98).abs());
                    }
                    if dist < 2.6 && best_dec.is_none_or(|(_, b, _)| dist < b) {
                        best_dec = Some((dec_gp, dist, dur_ratio));
                    }
                }
            }
            if let Some((dec_gp, _, dur_ratio)) = best_dec {
                info.dec_grad_idx = Some(dec_gp);
                info.ea_dur_ratio = dur_ratio;
            }
        }
    }

    info
}

fn parse_gp_instruction(
    stmt: &str,
    require_ea_tag: Option<&str>,
    pulses_us: &[f64],
) -> Option<(usize, f64, f64)> {
    for tok in stmt.split_whitespace() {
        let Some((dur_part, gp_part)) = tok.split_once(":gp") else {
            continue;
        };
        if let Some(tag) = require_ea_tag {
            if !gp_part.contains(tag) {
                continue;
            }
        } else if gp_part.contains("EA") {
            continue;
        }
        let digits = gp_part.bytes().take_while(|b| b.is_ascii_digit()).count();
        if digits == 0 {
            continue;
        }
        let gp_idx = gp_part[..digits].parse::<usize>().ok()?;
        let mut mult = 1.0_f64;
        for factor in gp_part[digits..].split('*').skip(1) {
            let f = factor
                .trim_end_matches(|c: char| !c.is_ascii_alphanumeric() && c != '-' && c != '.');
            if f == "-1" || f == "-1.0" {
                mult *= -1.0;
            } else if let Ok(v) = f.parse::<f64>() {
                if v != 0.0 {
                    mult *= v.signum();
                }
            }
        }
        let dur_clean = dur_part.trim_start_matches('(');
        let dur_us = if let Some(p_idx) = dur_clean
            .to_ascii_lowercase()
            .strip_prefix('p')
            .and_then(|x| x.parse::<usize>().ok())
        {
            pulses_us
                .get(p_idx)
                .copied()
                .filter(|&v| v > 0.0)
                .unwrap_or(1000.0)
        } else {
            parse_bruker_time_literal_s(dur_clean)
                .map(|s| s * 1e6)
                .unwrap_or(1000.0)
        };
        return Some((gp_idx, dur_us, mult));
    }
    None
}

fn parse_90deg_ip_dp(tok: &str) -> Option<(usize, i32)> {
    if tok.contains('*') {
        return None;
    }
    let low = tok.to_ascii_lowercase();
    if let Some(rest) = low.strip_prefix("ip") {
        if !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit()) {
            return Some((rest.parse().ok()?, 1));
        }
    } else if let Some(rest) = low.strip_prefix("dp") {
        if !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit()) {
            return Some((rest.parse().ok()?, -1));
        }
    }
    None
}

fn phase_ops_quarter_turns(ops: &[String], target_reg: usize) -> (bool, i32) {
    let mut has_reset = false;
    let mut qturns = 0_i32;
    for op in ops {
        let low = op.to_ascii_lowercase();
        if let Some(rest) = low.strip_prefix("rp") {
            if rest.parse::<usize>().ok() == Some(target_reg) {
                has_reset = true;
                qturns = 0;
            }
        } else if let Some((sign, rest)) = low
            .strip_prefix("ip")
            .map(|r| (1_i32, r))
            .or_else(|| low.strip_prefix("dp").map(|r| (-1_i32, r)))
        {
            let (reg_s, mult_s) = rest
                .split_once('*')
                .map(|(a, b)| (a, Some(b)))
                .unwrap_or((rest, None));
            if reg_s.parse::<usize>().ok() == Some(target_reg) {
                let m = mult_s.and_then(|s| s.parse::<i32>().ok()).unwrap_or(1);
                qturns += sign * m;
            }
        }
    }
    (has_reset, qturns)
}

#[allow(clippy::type_complexity)]
fn resolve_bruker_per_scan_fq_sfo(
    f_dim: usize,
    exec_lines: &[(usize, String)],
    raw_loops: &[RawLoopSpan],
    acqus: &HashMap<String, String>,
    ch_nuclei: &[String; 5],
    fallback_fqlist_text: Option<&str>,
    exp_dir: Option<&Path>,
) -> Result<(Option<(String, f64)>, Vec<(String, f64)>), PulprogError> {
    if raw_loops
        .iter()
        .any(|l| l.list_ops.iter().any(|o| o.starts_with("ifq")))
    {
        return Ok((None, Vec::new()));
    }
    let evo_tok = if f_dim == 1 { "d0" } else { "d10" };
    let Some(evo_idx) = exec_lines.iter().position(|(_, s)| {
        !s.contains("MCWRK")
            && !s.contains("caldel")
            && !s.trim_start().starts_with('"')
            && !s.starts_with("define ")
            && contains_word(s, evo_tok)
    }) else {
        return Ok((None, Vec::new()));
    };

    // Scan `exec_lines` for intra-scan `fq<k>:f<ch>` switches (`1 <= k <= 8`, `1 <= ch <= 4`)
    for k in 1..=8usize {
        for (ch, nuc) in ch_nuclei.iter().enumerate().take(5).skip(1) {
            if nuc.is_empty() {
                continue;
            }
            let pat = format!("fq{k}:f{ch}");
            let mut before_evo = 0usize;
            let mut total_calls = 0usize;
            for (idx, (_, stmt)) in exec_lines.iter().enumerate() {
                if stmt.trim_start().starts_with('"') {
                    continue;
                }
                let low = stmt.to_ascii_lowercase();
                let n_hits = low
                    .split(|c: char| c.is_whitespace() || matches!(c, ',' | '(' | ')' | ';'))
                    .filter(|t| *t == pat)
                    .count();
                if n_hits > 0 {
                    total_calls += n_hits;
                    if idx < evo_idx {
                        before_evo += n_hits;
                    }
                }
            }
            if before_evo == 0 || total_calls < 2 {
                continue;
            }
            // Locate the corresponding `fq*list` file (`fq<k>list`, XWIN-NMR 0-indexed `fq<k-1>list`, or `fqlist`)
            let fq_txt_owned: Option<String> = if let Some(dir) = exp_dir {
                let canon_root = canonicalize_exp_dir(dir)?;
                let mut candidates = vec![format!("fq{k}list")];
                if k >= 1 {
                    candidates.push(format!("fq{}list", k - 1));
                }
                candidates.push("fqlist".to_string());
                let mut loaded = None;
                for cand in &candidates {
                    if let Some(p) = find_relative_child_ci(dir, cand) {
                        let s = read_sandboxed_text_lossy(&canon_root, &p, cand)?;
                        if !s.trim().is_empty() {
                            loaded = Some(s);
                            break;
                        }
                    }
                }
                loaded.or_else(|| fallback_fqlist_text.map(String::from))
            } else {
                fallback_fqlist_text.map(String::from)
            };

            let Some(txt) = fq_txt_owned else {
                continue;
            };
            let ch_bf = acqus
                .get(&format!("BF{ch}"))
                .or_else(|| acqus.get(&format!("SFO{ch}")))
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(150.9);
            let Some(entries_mhz) = parse_bruker_fqlist_mhz(&txt, ch_bf) else {
                continue;
            };
            if entries_mhz.len() != total_calls {
                continue;
            }
            let active_sfo = entries_mhz[(before_evo - 1) % entries_mhz.len()];
            let mut extra_jumps = Vec::new();
            if entries_mhz.len() >= 3 && ch_bf > 1.0 {
                for &mhz in &entries_mhz {
                    let ppm = (mhz - ch_bf) * 1e6 / ch_bf;
                    extra_jumps.push((nuc.clone(), ppm));
                }
            }
            return Ok((Some((nuc.clone(), active_sfo)), extra_jumps));
        }
    }
    Ok((None, Vec::new()))
}

fn parse_bruker_fqlist_mhz(text: &str, default_bf_mhz: f64) -> Option<Vec<f64>> {
    #[derive(Clone, Copy)]
    enum FqUnit {
        DefaultBfHz(f64),
        HzOffset(f64),
        PpmOffset(f64),
        AbsHz,
    }
    let mut unit = FqUnit::DefaultBfHz(default_bf_mhz);
    let mut entries = Vec::new();
    for raw in text.lines() {
        let t = raw.split(['#', ';']).next().unwrap_or("").trim();
        if t.is_empty() {
            continue;
        }
        let mut parts = t.split_whitespace();
        let first = parts.next().unwrap_or("");
        if first.eq_ignore_ascii_case("O") || first.eq_ignore_ascii_case("B") {
            if let Some(base) = parts.next().and_then(|s| s.parse::<f64>().ok()) {
                if base > 1.0 {
                    unit = FqUnit::HzOffset(base);
                }
            }
            continue;
        }
        if first.eq_ignore_ascii_case("P") {
            if let Some(base) = parts.next().and_then(|s| s.parse::<f64>().ok()) {
                if base > 1.0 {
                    unit = FqUnit::PpmOffset(base);
                }
            }
            continue;
        }
        if first.eq_ignore_ascii_case("SFO") {
            if let Some(kind) = parts.next() {
                if kind.eq_ignore_ascii_case("hz") {
                    unit = FqUnit::AbsHz;
                }
            }
            continue;
        }
        if let Ok(val) = first.parse::<f64>() {
            let mhz = match unit {
                FqUnit::HzOffset(base) => base + val * 1e-6,
                FqUnit::PpmOffset(base) => base * (1.0 + val * 1e-6),
                FqUnit::AbsHz => val * 1e-6,
                FqUnit::DefaultBfHz(bf) => {
                    if val > 1.0e7 {
                        val * 1e-6
                    } else {
                        bf + val * 1e-6
                    }
                }
            };
            entries.push(mhz);
        }
    }
    (!entries.is_empty()).then_some(entries)
}

fn extract_calph_macro_90(text: &str, macro_prefix: &str) -> Option<(usize, i32)> {
    let idx = text.find(macro_prefix)?;
    let tail = &text[idx + macro_prefix.len()..];
    let cidx = tail.find("calph(")?;
    let inner = &tail[cidx + "calph(".len()..];
    let end = inner.find(')')?;
    let (ph_s, deg_s) = inner[..end].split_once(',')?;
    let reg = ph_s
        .trim()
        .strip_prefix("ph")?
        .trim()
        .parse::<usize>()
        .ok()?;
    let deg: i32 = deg_s.trim().trim_start_matches('+').parse().ok()?;
    match deg.rem_euclid(360) {
        90 => Some((reg, 1)),
        270 => Some((reg, -1)),
        _ => None,
    }
}

#[allow(clippy::too_many_arguments)]
fn build_bruker_indirect_dim_physics(
    f_dim: usize,
    acqu_source: &str,
    acqus: &HashMap<String, String>,
    acqu_ind: &HashMap<String, String>,
    h1_base_mhz: Option<f64>,
    delays_s: &[f64],
    _increments_s: &[f64],
    pulses_us: &[f64],
    exec_lines: &[(usize, String)],
    sym: &SymbolicDimLoopInfo,
    pulprog_name: &str,
) -> Result<IndirectDimPhysics, PulprogError> {
    let td_raw = acqu_ind
        .get("TD")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1)
        .max(1);
    let header_fnmode = acqu_ind
        .get("FnMODE")
        .and_then(|s| s.parse::<i32>().ok())
        .unwrap_or(0);

    // Resolve quadrature mode:
    // When `FnMODE == 0 (Undefined)` or `FnMODE == 1 (QF)` AND the executed `pulseprogram`
    // explicitly synthesizes a 2-step hypercomplex loop (`ip*`/`dp*` -> `States`/`States-TPPI`,
    // or `igrad` -> `Echo-Antiecho`), override to the synthesized hypercomplex mode!
    let (quad_mode, quad_source) = match header_fnmode {
        4 => (DecodedQuadMode::States, QuadSource::HeaderFnMode(4)),
        5 => (DecodedQuadMode::StatesTppi, QuadSource::HeaderFnMode(5)),
        6 => (DecodedQuadMode::EchoAntiecho, QuadSource::HeaderFnMode(6)),
        3 => (DecodedQuadMode::Tppi, QuadSource::HeaderFnMode(3)),
        2 => (DecodedQuadMode::Qseq, QuadSource::HeaderFnMode(2)),
        0 | 1 => {
            if let Some(synth) = sym.synthesized_quad {
                (synth, QuadSource::SynthesizedManualLoop { header_fnmode })
            } else if header_fnmode == 1 {
                (DecodedQuadMode::Qf, QuadSource::HeaderFnMode(1))
            } else {
                // `FnMODE == 0` with `F1PH`/`F2PH`/`F1EA`/`F2EA` macro in `mc`:
                if sym.has_igrad_ea {
                    (DecodedQuadMode::EchoAntiecho, QuadSource::HeaderFnMode(0))
                } else {
                    (DecodedQuadMode::Qf, QuadSource::HeaderFnMode(0))
                }
            }
        }
        other => (DecodedQuadMode::Qf, QuadSource::HeaderFnMode(other)),
    };

    let sfo_ind = acqu_ind
        .get("SFO1")
        .and_then(|s| s.parse::<f64>().ok())
        .or_else(|| acqu_ind.get("BF1").and_then(|s| s.parse::<f64>().ok()))
        .unwrap_or(60.0);
    let bf_ind = acqu_ind
        .get("BF1")
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(sfo_ind);
    let nucleus = canonicalize_nucleus(
        acqu_ind.get("NUC1").map(String::as_str).unwrap_or("15N"),
        Some(sfo_ind),
        h1_base_mhz,
    );

    // Sync active channel SFO from `acqus` if `acqu2s`/`acqu3s` kept template `SFO1 == BF1`
    let obs_mhz = bf_ind.max(1e-6);
    let mut active_sfo = sfo_ind;
    for ch in 1..=4 {
        let ch_n = acqus
            .get(&format!("NUC{ch}"))
            .map(String::as_str)
            .unwrap_or("");
        let ch_bf = acqus
            .get(&format!("BF{ch}"))
            .and_then(|s| s.parse::<f64>().ok());
        let ch_sfo = acqus
            .get(&format!("SFO{ch}"))
            .and_then(|s| s.parse::<f64>().ok());
        if canonicalize_nucleus(ch_n, ch_sfo, h1_base_mhz) == nucleus {
            if let (Some(cb), Some(cs)) = (ch_bf, ch_sfo) {
                if (cb - bf_ind).abs() < 0.05 && (sfo_ind - bf_ind).abs() < 1e-6 {
                    active_sfo = cs;
                }
            }
        }
    }
    if let Some((ref fq_nuc, fq_sfo)) = sym.per_scan_fq_sfo_mhz {
        if *fq_nuc == nucleus && fq_sfo > 1.0 {
            active_sfo = fq_sfo;
        }
    }
    let carrier_ppm = (active_sfo - obs_mhz) * 1e6 / obs_mhz;

    let sw_ppm = acqu_ind
        .get("SW")
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|&v| v > 0.0)
        .or_else(|| {
            acqu_ind
                .get("SW_h")
                .and_then(|s| s.parse::<f64>().ok())
                .filter(|&v| v > 0.0)
                .map(|swh| swh / active_sfo.max(1e-6))
        })
        .unwrap_or(30.0);
    let sw_hz = sw_ppm * active_sfo.max(1e-6);
    let complex_points = match quad_mode {
        DecodedQuadMode::Qf | DecodedQuadMode::Tppi | DecodedQuadMode::Qseq => td_raw,
        _ => (td_raw / 2).max(1),
    };

    // Rance–Kay combination matrix & `GPZ` polarity for `EchoAntiecho`
    let gpz_vec = parse_jcamp_f64_vec(acqus, "GPZ", 32);
    let rance_kay = if quad_mode == DecodedQuadMode::EchoAntiecho {
        let enc_lvl = sym
            .ea_grad_idx
            .and_then(|i| gpz_vec.get(i).copied())
            .map(|v| v * sym.ea_explicit_mult);
        let dec_lvl = sym.dec_grad_idx.and_then(|i| gpz_vec.get(i).copied());
        let ratio = match (enc_lvl, dec_lvl) {
            (Some(e), Some(d)) if d.abs() > 1e-6 => Some((e / d) * sym.ea_dur_ratio.max(1e-6)),
            _ => None,
        };
        let ea_polarity_inverted = ratio.is_some_and(|r| r < 0.0);
        let coef = if ea_polarity_inverted {
            // Swapped `(z0, z1)` echo/antiecho selection negates the sine combination relative to cos!
            [0.0, -1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 0.0]
        } else {
            [0.0, -1.0, 0.0, -1.0, -1.0, 0.0, 1.0, 0.0]
        };
        Some(RanceKayMatrix {
            coef,
            ea_polarity_inverted,
            enc_grad_level: enc_lvl,
            dec_grad_level: dec_lvl,
            grad_ratio: ratio,
        })
    } else {
        None
    };

    // Deduce `reverse_frequency` (`REVERSE = yes`):
    let reverse_frequency = infer_symbolic_reverse(
        f_dim,
        quad_mode,
        sym,
        exec_lines,
        pulprog_name,
        rance_kay.as_ref().is_some_and(|rk| rk.ea_polarity_inverted),
    );

    // Initial $t_1(0)$ sampling delay (`d0` for `F1`, `d10` for `F2`), dwell `1 / SW_hz`, and Marion et al. (1989) `(PHC0, PHC1)`
    let d_idx = if f_dim == 1 {
        0
    } else if f_dim == 2 {
        10
    } else {
        20
    };
    let raw_d0_s = delays_s.get(d_idx).copied().unwrap_or(0.0);
    let dwell_s = 1.0 / sw_hz.max(1e-6);

    // Check whether `d0` appears twice (`d0 ... 180_pulse ... d0` -> total initial evolution = `2 * d0 + (4/pi)*p90`)
    // vs once (`d0` -> initial evolution = `d0`), and whether constant-time sets `t(0) = 0`!
    let evo_tok = if f_dim == 1 { "d0" } else { "d10" };
    let d0_occurrences = exec_lines
        .iter()
        .filter(|(_, s)| !s.contains("MCWRK") && !s.contains("caldel") && contains_word(s, evo_tok))
        .count();
    let double_d0 = d0_occurrences >= 2;
    let p90_s = if nucleus == "15N" {
        pulses_us.get(21).copied().unwrap_or(0.0) * 1e-6
    } else if nucleus == "13C" {
        // Hard `13C` 90° pulse is `P[3]` (`p3`); `P[13]` (`p13`) in Bruker triple-resonance
        // sequences (`hncacbgp3d`, `hncogp3d`) is a shaped pulse (`> 100 us`) compensated by `sp8`.
        pulses_us.get(3).copied().unwrap_or(0.0) * 1e-6
    } else {
        pulses_us.get(1).copied().unwrap_or(0.0) * 1e-6
    };

    let initial_delay_s = if sym.constant_time && raw_d0_s > 2.5e-3 {
        // Constant-time where `d10 = T_CT / 2` and `d30 = T_CT / 2`: `t(0) = d10 - d30 = 0`!
        0.0
    } else if double_d0 {
        // `d0 - 180 - d0` with `in0 = DW / 2`: when `raw_d0_s <= 10 us` (`d0 = 3u` minimal
        // switching delay), initial sampling is zero-delay (`PHC1 = 0°`, `FCOR = 0.5`).
        // Only add `(4/pi) * p90` finite-pulse evolution when `raw_d0_s > 10 us` reflects an
        // explicit pre-compensated half-dwell formula (`d0 = in0/2 - p1 - 2*p90/pi`).
        let base = 2.0 * raw_d0_s;
        if raw_d0_s <= 10e-6 || base > 0.25 * dwell_s {
            base
        } else {
            base + (4.0 / PI) * p90_s.min(40e-6)
        }
    } else if raw_d0_s > 0.0 {
        if raw_d0_s <= 10e-6 || raw_d0_s > 0.25 * dwell_s {
            raw_d0_s
        } else {
            raw_d0_s + (4.0 / PI) * p90_s.min(40e-6)
        }
    } else {
        0.0
    };

    let frac_dwell = if dwell_s > 0.0 {
        initial_delay_s / dwell_s
    } else {
        0.0
    };
    let half_dwell_sampling = !sym.constant_time && (0.34..=0.68).contains(&frac_dwell);
    let multi_half_dwell = !sym.constant_time
        && (1.34..=4.68).contains(&frac_dwell)
        && ((frac_dwell - 0.5) - (frac_dwell - 0.5).round()).abs() <= 0.18;
    let (predicted_phc0_deg, predicted_phc1_deg) = if half_dwell_sampling {
        (-90.0, 180.0)
    } else if multi_half_dwell {
        let dwell_mult = (frac_dwell - 0.5).round() + 0.5;
        let raw_p0 = -180.0 * dwell_mult;
        let wrapped_p0 = (raw_p0 + 180.0).rem_euclid(360.0) - 180.0;
        (wrapped_p0, 360.0 * dwell_mult)
    } else {
        (0.0, 0.0)
    };

    let folding = match quad_mode {
        DecodedQuadMode::Qf | DecodedQuadMode::Qseq => FoldingConvention::Magnitude,
        _ if sym.constant_time => FoldingConvention::ConstantTimeMirror,
        DecodedQuadMode::StatesTppi | DecodedQuadMode::Tppi => {
            FoldingConvention::MirrorInvertedWrap
        }
        DecodedQuadMode::States | DecodedQuadMode::EchoAntiecho | DecodedQuadMode::Dqd => {
            FoldingConvention::SameSignWrap
        }
    };

    Ok(IndirectDimPhysics {
        f_dim,
        acqu_source: acqu_source.to_string(),
        nucleus,
        carrier_ppm,
        sw_ppm,
        sw_hz,
        obs_mhz,
        td_raw,
        complex_points,
        quad_mode,
        quad_source,
        reverse_frequency,
        rance_kay,
        constant_time: sym.constant_time,
        initial_delay_s,
        increment_s: dwell_s,
        half_dwell_sampling,
        predicted_phc0_deg,
        predicted_phc1_deg,
        folding,
    })
}

fn infer_symbolic_reverse(
    f_dim: usize,
    quad_mode: DecodedQuadMode,
    sym: &SymbolicDimLoopInfo,
    exec_lines: &[(usize, String)],
    pulprog_name: &str,
    ea_polarity_inverted: bool,
) -> bool {
    if let Some(explicit_rev) = sym.comment_reverse_override {
        return explicit_rev;
    }
    if quad_mode == DecodedQuadMode::EchoAntiecho {
        let base_rev = sym.sct_inc_before_dec.unwrap_or(false);
        return base_rev ^ ea_polarity_inverted;
    }
    if !matches!(
        quad_mode,
        DecodedQuadMode::States | DecodedQuadMode::StatesTppi
    ) {
        return false;
    }
    if let Some((reg, step_sign)) = sym.quad_reg_sign {
        let ph_tok = format!("ph{reg}");
        let evo_tok = if f_dim == 1 { "d0" } else { "d10" };
        let dec_tok = if f_dim == 1 { "d20" } else { "d30" };
        let mut evo_line = None;
        let mut first_ph_line = None;
        for (idx, (_, stmt)) in exec_lines.iter().enumerate() {
            if stmt.contains("MCWRK")
                || stmt.contains("caldel")
                || stmt.trim_start().starts_with('"')
            {
                continue;
            }
            if evo_line.is_none() && !stmt.starts_with("define ") && contains_word(stmt, evo_tok) {
                evo_line = Some(idx);
            }
            if first_ph_line.is_none() && !stmt.contains("calph(") && contains_word(stmt, &ph_tok) {
                first_ph_line = Some(idx);
            }
        }
        if let (Some(p_idx), Some(el)) = (first_ph_line, evo_line) {
            // Note: an initial phase-table offset such as `ph9 = 3 1` (`270°` / `90°`) multiplies
            // the complex FID by a constant unit phasor `exp(+-i * phi_0)` (`PHC0` shift), never
            // conjugating `exp(-i * omega * t)` <-> `exp(+i * omega * t)`. Only the quadrature
            // increment sign (`ip` vs `dp`), readout-after-evolution (`p_idx > el`), and semi-CT
            // decremented-delay refocusing (`has_dec_before_evo`) determine `REVERSE`.
            let mut net_sign = step_sign;
            if p_idx > el {
                net_sign = -net_sign;
            } else {
                let has_dec_before_evo = exec_lines[p_idx + 1..el].iter().any(|(_, s)| {
                    !s.contains("MCWRK")
                        && !s.contains("caldel")
                        && !s.trim_start().starts_with('"')
                        && contains_word(s, dec_tok)
                });
                if has_dec_before_evo {
                    net_sign = -net_sign;
                }
            }
            if sym.evo_decremented {
                net_sign = -net_sign;
            }
            if sym.paired_post_evo_echo_quad {
                net_sign = -net_sign;
            }
            return net_sign < 0;
        }
    }
    let low = pulprog_name.to_ascii_lowercase();
    (f_dim == 2 && low.contains("gpwg3d"))
        || (f_dim == 1 && (low.contains("noesy") || low.contains("tocsy") || low.contains("roesy")))
}

fn detect_bruker_pseudo3d_order(
    raw_loops: &[RawLoopSpan],
    exec_lines: &[(usize, String)],
    mc_lines: &[String],
    indirect_dims: &[IndirectDimPhysics],
    inp: &BrukerExperimentInputs<'_>,
    pulprog_name: &str,
) -> (Pseudo3DStorageOrder, Option<String>) {
    // 1. DOSY (`dstebpgp3s`, `ledbpgp2s`, `difflist`, or bipolar diffusion gradient ramp with QF indirect)
    let low_pp = pulprog_name.to_ascii_lowercase();
    let has_difflist = inp.difflist_text.is_some()
        || raw_loops
            .iter()
            .any(|l| l.list_ops.iter().any(|o| o == "difflist"));
    let all_ind_qf = !indirect_dims.is_empty()
        && indirect_dims
            .iter()
            .all(|d| d.quad_mode == DecodedQuadMode::Qf);
    if has_difflist
        || (all_ind_qf
            && (low_pp.contains("dste")
                || low_pp.contains("ledbp")
                || low_pp.contains("dosy")
                || low_pp.contains("bpp")))
    {
        return (
            Pseudo3DStorageOrder::PlaneQueued,
            Some("difflist".to_string()),
        );
    }

    // 2. Check symbolic loop nesting between `t1` (`id0`) and `vd`/`vc`/`fq`/toggle counter (`l*`)!
    let t1_loop_end = raw_loops
        .iter()
        .find(|l| l.delay_ops.iter().any(|d| d == "id0" || d == "dd0"))
        .map(|l| l.end_line);
    let list_loop = raw_loops.iter().find(|l| {
        l.list_ops
            .iter()
            .any(|o| matches!(o.as_str(), "ivd" | "ivc") || o.starts_with("ifq"))
    });

    let list_kind_from_inputs = if inp.vdlist_text.is_some()
        || exec_lines.iter().any(|(_, s)| contains_word(s, "vd"))
    {
        Some("vdlist".to_string())
    } else if inp.vclist_text.is_some() || exec_lines.iter().any(|(_, s)| contains_word(s, "vc")) {
        Some("vclist".to_string())
    } else if inp.fqlist_text.is_some() {
        Some("fqlist".to_string())
    } else {
        None
    };

    if let (Some(t1_end), Some(ll)) = (t1_loop_end, list_loop) {
        let kind = if ll.list_ops.iter().any(|o| o.contains("vd")) {
            "vdlist"
        } else if ll.list_ops.iter().any(|o| o.contains("vc")) {
            "vclist"
        } else {
            "fqlist"
        };
        if ll.end_line < t1_end {
            return (Pseudo3DStorageOrder::T1Interleaved, Some(kind.to_string()));
        }
        return (Pseudo3DStorageOrder::PlaneQueued, Some(kind.to_string()));
    }

    // 3. TopSpin `mc #0 to 2 F1QF(calclist(vd, +1)) F2PH(...)` vs `F1PH(...) F2QF(calclist(vd, +1))`:
    //    In Bruker `mc` with `AQSEQ=321` (`F2` inner, `F1` outer):
    //    - `F1QF` (`acqu3s` outer QF, `F2PH` `acqu2s` inner `t1`): `vd` is in OUTER `F1` -> `PlaneQueued`!
    //    - `F2QF` (`acqu2s` inner QF, `F1PH` `acqu3s` outer `t1`): `vd` is in INNER `F2` -> `T1Interleaved`!
    let joined_mc = mc_lines.join(" ");
    if joined_mc.contains("F2QF") && (joined_mc.contains("F1PH") || joined_mc.contains("F1EA")) {
        return (
            Pseudo3DStorageOrder::T1Interleaved,
            list_kind_from_inputs.or_else(|| Some("qf_interleaved".to_string())),
        );
    }
    if joined_mc.contains("F1QF") && (joined_mc.contains("F2PH") || joined_mc.contains("F2EA")) {
        return (
            Pseudo3DStorageOrder::PlaneQueued,
            list_kind_from_inputs.or_else(|| Some("qf_planes".to_string())),
        );
    }

    // 4. Check indirect dimensions' `QF` placement (`indirect_dims[0]` = `acqu2s` inner, `indirect_dims[1]` = `acqu3s` outer)
    if indirect_dims.len() == 2 {
        let inner_qf = indirect_dims[0].quad_mode == DecodedQuadMode::Qf;
        let outer_qf = indirect_dims[1].quad_mode == DecodedQuadMode::Qf;
        if inner_qf && !outer_qf {
            return (
                Pseudo3DStorageOrder::T1Interleaved,
                list_kind_from_inputs.or_else(|| Some("qf_interleaved".to_string())),
            );
        }
        if outer_qf && !inner_qf {
            return (
                Pseudo3DStorageOrder::PlaneQueued,
                list_kind_from_inputs.or_else(|| Some("qf_planes".to_string())),
            );
        }
        // If BOTH indirect dimensions are phase-sensitive hypercomplex (`!inner_qf && !outer_qf`)
        // AND the pulse sequence evolves two chemical-shift axes (`d0` and `d10`/`id10`/`dd10`)
        // with no `ivd`/`ivc`/`ifq` list-increment loop, any `fq*list` on disk is an intra-scan
        // carrier-switching list (e.g., XWIN-NMR 2.x `hncocacbf.s` / `hncof.s`), NOT pseudo-3D!
        let has_d10_shift_evo = exec_lines.iter().any(|(_, s)| {
            contains_word(s, "d10") || contains_word(s, "id10") || contains_word(s, "dd10")
        });
        if !inner_qf && !outer_qf && has_d10_shift_evo {
            return (Pseudo3DStorageOrder::NotPseudo3D, None);
        }
    }

    if let Some(kind) = list_kind_from_inputs {
        // Default Bruker `hsqct1etf3gpsi3d` / `hsqct2etf3gpsi3d`: if `ivd`/`vd` appears inside an inner loop before `id0`,
        // `T1Interleaved`; otherwise `PlaneQueued`.
        let vd_line = exec_lines
            .iter()
            .position(|(_, s)| contains_word(s, "ivd") || contains_word(s, "ivc"));
        let id0_line = exec_lines.iter().position(|(_, s)| contains_word(s, "id0"));
        if let (Some(vl), Some(il)) = (vd_line, id0_line) {
            if vl < il {
                return (Pseudo3DStorageOrder::T1Interleaved, Some(kind));
            }
        }
        return (Pseudo3DStorageOrder::PlaneQueued, Some(kind));
    }

    (Pseudo3DStorageOrder::NotPseudo3D, None)
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn build_bruker_transfer_graph(
    dimensionality: usize,
    direct_nucleus: &str,
    ch_nuclei: &[String; 5],
    ch_carrier_ppm: &[Option<f64>; 5],
    extra_carrier_jumps_ppm: &[(String, f64)],
    cnst: &[f64],
    delays_s: &[f64],
    named_delays_s: &HashMap<String, f64>,
    spnam: &[String],
    cpdprg: &[String],
    gpz_pct: &[f64],
    indirect_dims: &[IndirectDimPhysics],
    exec_lines: &[(usize, String)],
    raw_loops: &[RawLoopSpan],
    pseudo_3d_order: Pseudo3DStorageOrder,
    pseudo_3d_list_kind: Option<&str>,
    active_defines: &BTreeMap<String, String>,
    pulprog_name: &str,
) -> MagnetizationTransferGraph {
    let mut active_set: BTreeSet<String> = BTreeSet::new();
    active_set.insert(direct_nucleus.to_string());
    for d in indirect_dims {
        if d.quad_mode != DecodedQuadMode::Qf || dimensionality <= 2 {
            active_set.insert(d.nucleus.clone());
        }
    }

    let mut carrier_jumps_ppm: Vec<(String, f64)> = Vec::new();
    let mut visited_co = false;
    let mut visited_ca = false;
    let mut visited_cacb = false;
    let mut has_1h_15n_inept = false;
    let mut has_1h_13c_inept = false;
    let mut has_n_c_transfer = false;
    let mut has_co_ca_transfer = false;
    let mut has_ca_cb_transfer = false;
    let mut has_tocsy_spinlock = false;
    let mut has_noesy_mixing_ms: Option<f64> = None;
    let mut has_diffusion = false;
    let mut has_bipolar_grad = false;
    let mut diffusion_delta_ms = 50.0_f64;
    let mut has_cest_sat = false;
    let mut has_hetnoe_sat = false;
    let mut has_r1_vd = false;
    let mut has_r2_echo = false;
    let mut has_cpmg_ct = false;

    let all_exec_str = exec_lines
        .iter()
        .map(|(_, s)| s.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    let low_exec = all_exec_str.to_ascii_lowercase();

    // Detect active RF channels used in executable lines (`:f1`, `:f2`, `:f3`, `:f4`)
    for (ch, nuc) in ch_nuclei.iter().enumerate().take(5).skip(1) {
        let tag = format!(":f{ch}");
        if low_exec.contains(&tag) && !nuc.is_empty() {
            active_set.insert(nuc.clone());
        }
    }

    // Track `fq=cnstK(...)` carrier jumps AND `CNST` chemical shift constants (`165..190` CO, `49..65` CA, `35..48.5` CACB)
    for (_, stmt) in exec_lines {
        let low = stmt.to_ascii_lowercase();
        if let Some(pos) = low.find("fq=cnst") {
            let after = &low[pos + "fq=cnst".len()..];
            let d_len = after.bytes().take_while(|b| b.is_ascii_digit()).count();
            if d_len > 0 {
                if let Ok(c_idx) = after[..d_len].parse::<usize>() {
                    if let Some(&ppm) = cnst.get(c_idx) {
                        if ppm > 1.0 {
                            let ch_nuc = if low.contains(":f2") {
                                ch_nuclei[2].as_str()
                            } else if low.contains(":f3") {
                                ch_nuclei[3].as_str()
                            } else {
                                "13C"
                            };
                            let nuc = if ch_nuc.is_empty() { "13C" } else { ch_nuc };
                            carrier_jumps_ppm.push((nuc.to_string(), ppm));
                            if (165.0..=190.0).contains(&ppm) {
                                visited_co = true;
                            } else if (49.0..=66.0).contains(&ppm) {
                                visited_ca = true;
                            } else if (35.0..=48.9).contains(&ppm) {
                                visited_cacb = true;
                            }
                        }
                    }
                }
            }
        }
    }
    // Also inspect `CNST[20..31]` when `spoff` off-resonance shaped pulses jump between `cnst` offsets!
    for &idx in &[20usize, 21, 22, 23, 24, 25, 26] {
        if let Some(&ppm) = cnst.get(idx) {
            if (165.0..=190.0).contains(&ppm) && !spnam.is_empty() {
                // Only mark `visited_co` if `:f2` (`13C`) actually fires shaped pulses (`:sp`)!
                if low_exec.contains(":sp") && active_set.contains("13C") {
                    visited_co = true;
                }
            } else if (49.0..=66.0).contains(&ppm)
                && low_exec.contains(":sp")
                && active_set.contains("13C")
            {
                visited_ca = true;
            } else if (35.0..=48.9).contains(&ppm)
                && low_exec.contains(":sp")
                && active_set.contains("13C")
            {
                visited_cacb = true;
            }
        }
    }
    for &(ref nuc, ppm) in extra_carrier_jumps_ppm {
        carrier_jumps_ppm.push((nuc.clone(), ppm));
        if (165.0..=190.0).contains(&ppm) {
            visited_co = true;
        } else if (49.0..=66.0).contains(&ppm) {
            visited_ca = true;
        } else if (35.0..=48.9).contains(&ppm) {
            visited_cacb = true;
        }
    }
    for ch in 1..=4 {
        if let Some(c_ppm) = ch_carrier_ppm[ch] {
            if ch_nuclei[ch] == "13C" {
                let evolves_co = indirect_dims
                    .iter()
                    .any(|d| d.nucleus == "13C" && d.carrier_ppm > 145.0);
                if (165.0..=190.0).contains(&c_ppm) {
                    visited_co = true;
                } else if (49.0..=66.0).contains(&c_ppm) && !evolves_co {
                    visited_ca = true;
                } else if (35.0..=48.9).contains(&c_ppm) && !evolves_co {
                    visited_cacb = true;
                }
            }
        }
    }

    // Collect referenced delay durations in executable body
    let mut used_delays_ms: Vec<(String, f64)> = Vec::new();
    for (d_idx, &val_s) in delays_s.iter().enumerate().take(64).skip(1) {
        if val_s <= 0.0 {
            continue;
        }
        let tok = format!("d{d_idx}");
        if contains_word(&low_exec, &tok) {
            used_delays_ms.push((tok, val_s * 1e3));
        }
    }
    for (k, &val_s) in named_delays_s {
        if val_s > 0.0 && contains_word(&all_exec_str, k) {
            used_delays_ms.push((k.clone(), val_s * 1e3));
        }
    }

    // Identify physical J-coupling & mixing delays:
    for (name, ms) in &used_delays_ms {
        if name == "d1" || name == "d11" || name == "d12" || name == "d16" {
            continue;
        }
        if (2.05..=2.95).contains(ms) && active_set.contains("15N") && active_set.contains("1H") {
            has_1h_15n_inept = true;
        }
        if (1.45..=2.02).contains(ms) && active_set.contains("13C") && active_set.contains("1H") {
            has_1h_13c_inept = true;
        }
        if (10.0..=29.0).contains(ms) && active_set.contains("15N") && active_set.contains("13C") {
            has_n_c_transfer = true;
        }
        if (3.1..=4.9).contains(ms) && active_set.contains("13C") {
            has_co_ca_transfer = true;
        }
        if (6.2..=15.2).contains(ms) && active_set.contains("13C") && !visited_co {
            has_ca_cb_transfer = true;
        }
        if (6.2..=8.2).contains(ms) && visited_cacb {
            has_ca_cb_transfer = true;
        }
        if (name == "d8" || name == "d9") && *ms >= 35.0 {
            has_noesy_mixing_ms = Some(*ms);
        }
        if name == "d20" && *ms >= 15.0 {
            diffusion_delta_ms = *ms;
        }
    }
    // Also check `d26` (`1H-15N` INEPT), `d23` (`N-CO`/`N-CA`), `d22` (`CO-CA`), `d28`/`d24` (`CA-CB`) standard registers
    if delays_s[26] >= 2.0e-3 && delays_s[26] <= 3.0e-3 && active_set.contains("15N") {
        has_1h_15n_inept = true;
    }
    if delays_s[23] >= 10.0e-3 && delays_s[23] <= 30.0e-3 && active_set.contains("13C") {
        has_n_c_transfer = true;
    }
    if delays_s[22] >= 3.0e-3 && delays_s[22] <= 5.0e-3 && active_set.contains("13C") {
        has_co_ca_transfer = true;
    }
    if (delays_s[28] >= 6.0e-3 && delays_s[28] <= 15.5e-3)
        || (delays_s[27] >= 6.0e-3 && delays_s[27] <= 15.5e-3 && visited_cacb)
    {
        has_ca_cb_transfer = true;
    }

    // Isotropic TOCSY (`dipsi` / `mlev` / `flopsy` spin-lock in RF body or invoked `CPDPRG[k]`)
    let invoked_tocsy_cpd = cpdprg.iter().enumerate().any(|(k, prg)| {
        let p_low = prg.to_ascii_lowercase();
        let is_iso = p_low.contains("dipsi")
            || p_low.contains("mlev")
            || p_low.contains("flopsy")
            || p_low.contains("tocsy");
        is_iso
            && (contains_word(&low_exec, &format!("cpd{k}"))
                || low_exec.contains(&format!("cpd{k}:")))
    });
    if invoked_tocsy_cpd
        || low_exec.contains("(d9 cpd")
        || low_exec.contains("(d19 cpd")
        || low_exec.contains("dipsi")
        || low_exec.contains("mlev")
        || low_exec.contains("flopsy")
        || low_exec.contains("p6*0.64")
        || low_exec.contains("p6*1.4")
    {
        has_tocsy_spinlock = true;
    }

    // DOSY diffusion detection (`p30:gp6` / bipolar `gp6` + `gp7` / `difflist` / `setgrad` with `QF` indirect)
    let all_ind_qf = !indirect_dims.is_empty()
        && indirect_dims
            .iter()
            .all(|d| d.quad_mode == DecodedQuadMode::Qf);
    let has_p30_grad = low_exec.contains("p30:gp") || low_exec.contains("p30*0.5:gp");
    if pseudo_3d_list_kind == Some("difflist")
        || (all_ind_qf && (has_p30_grad || (delays_s[20] >= 0.015 && low_exec.contains(":gp"))))
    {
        has_diffusion = true;
        // Bipolar gradient pair (`gp6` and `gp7` with opposite signs or `-1` factor around `180°` pulse)
        has_bipolar_grad = low_exec.contains("*-1")
            || gpz_pct
                .windows(2)
                .any(|w| w[0].abs() > 5.0 && (w[0] + w[1]).abs() < 1e-3);
    }

    // Dynamics detection (`RelaxR1`, `RelaxR2`, `Cpmg`, `Cest`, `HetNoe`)
    if pseudo_3d_list_kind == Some("vdlist") || contains_word(&low_exec, "vd") {
        has_r1_vd = true;
    }
    if pseudo_3d_list_kind == Some("vclist") || contains_word(&low_exec, "vc") {
        // Distinguish `CPMG` (constant-time `T_relax` where `vc` varies echo count at fixed total time,
        // or `d` echo spacing `>= 0.5 ms` variable) vs `RelaxR2` (total relaxation time = `vc * echo_unit`)
        if low_exec.contains("time_t2")
            || low_exec.contains("t_cpmg")
            || low_exec.contains("ncyc")
            || active_defines.contains_key("CPMG")
        {
            has_cpmg_ct = true;
        } else {
            has_r2_echo = true;
        }
    }
    if pseudo_3d_list_kind == Some("fqlist")
        && (delays_s.iter().any(|&d| (0.15..=1.2).contains(&d)) || low_exec.contains("cw"))
    {
        has_cest_sat = true;
    }
    // `{1H}-15N hetNOE`: proton `120°`/`90°` saturation loop (`lo to ... times l*` with `~5 ms` inter-pulse delay before `15N` `p21` excitation, no `1H->15N` initial INEPT!)
    let has_proton_sat_loop = raw_loops.iter().any(|l| {
        l.resolved_count.is_some_and(|c| c >= 50)
            && l.delay_ops.is_empty()
            && l.phase_ops.is_empty()
    });
    if has_proton_sat_loop && active_set.contains("15N") && !has_1h_15n_inept {
        has_hetnoe_sat = true;
    }

    let trosy_selected = active_defines.contains_key("TROSY")
        || pulprog_name.to_ascii_lowercase().contains("trosy")
        || pulprog_name.to_ascii_lowercase().starts_with("trhn")
        || (has_1h_15n_inept
            && !low_exec.contains("cpd3:f3")
            && !low_exec.contains("cpds3:f3")
            && active_set.contains("15N")
            && direct_nucleus == "1H"
            && !exec_lines.is_empty());
    let sensitivity_enhanced = indirect_dims
        .iter()
        .any(|d| d.quad_mode == DecodedQuadMode::EchoAntiecho);

    let detected_pool = if direct_nucleus == "13C" {
        if ch_carrier_ppm[1].is_some_and(|c| c > 140.0) {
            SpinPoolNode::CCarbonyl
        } else {
            SpinPoolNode::CAlpha
        }
    } else if has_diffusion || (!active_set.contains("15N") && active_set.contains("13C")) {
        SpinPoolNode::HAliphatic
    } else {
        SpinPoolNode::HAmide
    };

    // Map each evolved indirect dimension to its physical `SpinPoolNode`
    let mut evolved_pools: Vec<(usize, SpinPoolNode)> = Vec::new();
    for d in indirect_dims {
        if d.quad_mode == DecodedQuadMode::Qf {
            continue;
        }
        let pool = match d.nucleus.as_str() {
            "15N" => SpinPoolNode::NAmide,
            "13C" => {
                if d.carrier_ppm > 145.0 {
                    SpinPoolNode::CCarbonyl
                } else if d.sw_ppm >= 48.0 || (34.0..=48.9).contains(&d.carrier_ppm) || visited_cacb
                {
                    SpinPoolNode::CAliphatic
                } else if (49.0..=68.0).contains(&d.carrier_ppm) {
                    SpinPoolNode::CAlpha
                } else if (110.0..=145.0).contains(&d.carrier_ppm) {
                    SpinPoolNode::CAromatic
                } else {
                    SpinPoolNode::CAliphatic
                }
            }
            "1H" => SpinPoolNode::HAliphatic,
            "2H" => SpinPoolNode::Deuterium,
            _ => SpinPoolNode::HAmide,
        };
        evolved_pools.push((d.f_dim, pool));
    }

    // Populate `edges`
    let mut edges = Vec::new();
    if has_diffusion {
        edges.push(TransferEdge {
            from: detected_pool,
            to: detected_pool,
            mechanism: TransferMechanism::DiffusionStejskalTanner {
                delta_ms: diffusion_delta_ms,
                bipolar: has_bipolar_grad,
            },
        });
    }
    if has_1h_15n_inept {
        edges.push(TransferEdge {
            from: SpinPoolNode::HAmide,
            to: SpinPoolNode::NAmide,
            mechanism: TransferMechanism::IneptJCoupling { delay_ms: 2.4 },
        });
    }
    if has_1h_13c_inept {
        edges.push(TransferEdge {
            from: SpinPoolNode::HAliphatic,
            to: SpinPoolNode::CAliphatic,
            mechanism: TransferMechanism::IneptJCoupling { delay_ms: 1.8 },
        });
    }
    if has_n_c_transfer {
        let evolves_ca = evolved_pools
            .iter()
            .any(|(_, p)| *p == SpinPoolNode::CAlpha);
        let target_c =
            if visited_co && ((!visited_ca && !visited_cacb) || (visited_ca && evolves_ca)) {
                SpinPoolNode::CCarbonyl
            } else {
                SpinPoolNode::CAlpha
            };
        edges.push(TransferEdge {
            from: SpinPoolNode::NAmide,
            to: target_c,
            mechanism: TransferMechanism::ConstantTimeJCoupling { delay_ms: 12.0 },
        });
    }
    if has_co_ca_transfer {
        edges.push(TransferEdge {
            from: SpinPoolNode::CCarbonyl,
            to: SpinPoolNode::CAlpha,
            mechanism: TransferMechanism::IneptJCoupling { delay_ms: 4.0 },
        });
    }
    if has_ca_cb_transfer {
        edges.push(TransferEdge {
            from: SpinPoolNode::CAlpha,
            to: SpinPoolNode::CBeta,
            mechanism: TransferMechanism::IneptJCoupling { delay_ms: 7.0 },
        });
    }
    if has_tocsy_spinlock {
        edges.push(TransferEdge {
            from: SpinPoolNode::CAliphatic,
            to: SpinPoolNode::CAliphatic,
            mechanism: TransferMechanism::IsotropicTocsyMixing,
        });
    }
    if let Some(mix_ms) = has_noesy_mixing_ms {
        edges.push(TransferEdge {
            from: SpinPoolNode::HAliphatic,
            to: detected_pool,
            mechanism: TransferMechanism::DipolarNoesyMixing { mix_ms },
        });
    }
    if has_r1_vd {
        edges.push(TransferEdge {
            from: SpinPoolNode::NAmide,
            to: SpinPoolNode::NAmide,
            mechanism: TransferMechanism::RelaxationT1Delay,
        });
    }
    if has_r2_echo {
        edges.push(TransferEdge {
            from: SpinPoolNode::NAmide,
            to: SpinPoolNode::NAmide,
            mechanism: TransferMechanism::RelaxationT2Echo,
        });
    }
    if has_cpmg_ct {
        edges.push(TransferEdge {
            from: SpinPoolNode::NAmide,
            to: SpinPoolNode::NAmide,
            mechanism: TransferMechanism::CpmgDispersion,
        });
    }
    if has_cest_sat {
        edges.push(TransferEdge {
            from: SpinPoolNode::NAmide,
            to: SpinPoolNode::NAmide,
            mechanism: TransferMechanism::CestSaturation,
        });
    }
    if has_hetnoe_sat {
        edges.push(TransferEdge {
            from: SpinPoolNode::HAmide,
            to: SpinPoolNode::NAmide,
            mechanism: TransferMechanism::HetNoeSaturation,
        });
    }

    let has_explicit_co_jump = carrier_jumps_ppm
        .iter()
        .any(|(_, ppm)| (165.0..=190.0).contains(ppm));
    let has_explicit_ca_jump = carrier_jumps_ppm
        .iter()
        .any(|(_, ppm)| (35.0..=66.0).contains(ppm));
    let n_short_cc_relays = used_delays_ms
        .iter()
        .filter(|(name, ms)| {
            name != "d1"
                && name != "d11"
                && name != "d12"
                && name != "d16"
                && (3.0..=5.0).contains(ms)
        })
        .count();
    let has_dual_cc_relays = n_short_cc_relays >= 2
        || ((3.0e-3..=5.0e-3).contains(&delays_s[22]) && (3.0e-3..=5.0e-3).contains(&delays_s[28]));

    // Deduce `PulprogSequenceFamily` from first-principles transfer graph!
    let deduced_family = deduce_family_from_graph(
        dimensionality,
        direct_nucleus,
        &evolved_pools,
        pseudo_3d_order,
        has_diffusion,
        has_cest_sat,
        has_cpmg_ct,
        has_r2_echo,
        has_r1_vd,
        has_hetnoe_sat,
        has_tocsy_spinlock,
        has_noesy_mixing_ms.is_some(),
        visited_co,
        has_explicit_co_jump,
        visited_ca,
        has_explicit_ca_jump,
        visited_cacb,
        has_co_ca_transfer,
        has_ca_cb_transfer,
        has_dual_cc_relays,
        has_1h_13c_inept,
        trosy_selected,
        sensitivity_enhanced,
        &low_exec,
    );

    MagnetizationTransferGraph {
        active_nuclei: active_set.into_iter().collect(),
        carrier_jumps_ppm,
        evolved_pools,
        detected_pool,
        trosy_selected,
        sensitivity_enhanced,
        diffusion_encoded: has_diffusion,
        edges,
        deduced_family,
    }
}

#[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
fn deduce_family_from_graph(
    dimensionality: usize,
    direct_nucleus: &str,
    evolved_pools: &[(usize, SpinPoolNode)],
    pseudo_3d_order: Pseudo3DStorageOrder,
    has_diffusion: bool,
    has_cest_sat: bool,
    has_cpmg_ct: bool,
    has_r2_echo: bool,
    has_r1_vd: bool,
    has_hetnoe_sat: bool,
    has_tocsy_spinlock: bool,
    has_noesy_mixing: bool,
    visited_co: bool,
    has_explicit_co_jump: bool,
    visited_ca: bool,
    has_explicit_ca_jump: bool,
    visited_cacb: bool,
    has_co_ca_transfer: bool,
    _has_ca_cb_transfer: bool,
    has_dual_cc_relays: bool,
    has_1h_13c_inept: bool,
    trosy_selected: bool,
    sensitivity_enhanced: bool,
    low_exec: &str,
) -> PulprogSequenceFamily {
    if dimensionality == 1 {
        return PulprogSequenceFamily::OneDim;
    }
    if has_diffusion {
        return PulprogSequenceFamily::Dosy;
    }
    if has_cest_sat {
        return PulprogSequenceFamily::Cest;
    }
    if has_cpmg_ct {
        return PulprogSequenceFamily::Cpmg;
    }
    if has_r2_echo {
        return PulprogSequenceFamily::RelaxR2;
    }
    if has_r1_vd && pseudo_3d_order != Pseudo3DStorageOrder::NotPseudo3D {
        return PulprogSequenceFamily::RelaxR1;
    }
    if has_hetnoe_sat {
        return PulprogSequenceFamily::HetNoe;
    }

    // 13C-detected protonless experiments (`CON`, `CACO`, `CANCO`, `CBCACO`)
    if direct_nucleus == "13C" {
        if evolved_pools.len() == 1 {
            return match evolved_pools[0].1 {
                SpinPoolNode::NAmide => PulprogSequenceFamily::Con2D,
                _ => PulprogSequenceFamily::Caco2D,
            };
        }
        if evolved_pools.len() == 2 {
            let has_cacb = evolved_pools
                .iter()
                .any(|(_, p)| *p == SpinPoolNode::CAliphatic);
            return if has_cacb || visited_cacb {
                PulprogSequenceFamily::Cbcaco3D
            } else {
                PulprogSequenceFamily::Canco3D
            };
        }
    }

    // 2D 1H-detected experiments
    if evolved_pools.len() == 1 && pseudo_3d_order == Pseudo3DStorageOrder::NotPseudo3D {
        return match evolved_pools[0].1 {
            SpinPoolNode::NAmide => {
                if trosy_selected {
                    PulprogSequenceFamily::N15Trosy
                } else {
                    PulprogSequenceFamily::N15Hsqc
                }
            }
            SpinPoolNode::CAlpha
            | SpinPoolNode::CAliphatic
            | SpinPoolNode::CAromatic
            | SpinPoolNode::CCarbonyl => {
                // Distinguish HMQC (multiple-quantum `H90 - d2 - C90 - d0 - H180 - d0 - C90`: no Rance-Kay
                // sensitivity enhancement, no 13C 180° refocusing `p4`/`:sp`, and only a single `p1` 90° pulse)
                // from HSQC (Cavanagh et al. 2007 §6.2.2).
                let p1_count = low_exec.matches("(p1 ").count() + low_exec.matches(" p1 ").count();
                let has_c180 = low_exec.contains("(p4 ")
                    || low_exec.contains(" p4 ")
                    || low_exec.contains(":sp");
                let c90_hmqc = !sensitivity_enhanced
                    && !has_c180
                    && p1_count <= 1
                    && (low_exec.contains("(p2 ") || low_exec.contains(" p2 "))
                    && low_exec.contains("d0");
                if c90_hmqc {
                    PulprogSequenceFamily::C13Hmqc
                } else {
                    PulprogSequenceFamily::C13Hsqc
                }
            }
            _ => PulprogSequenceFamily::Unknown,
        };
    }

    // 3D 1H-detected experiments (`evolved_pools.len() == 2`)
    if evolved_pools.len() == 2 {
        let has_n = evolved_pools
            .iter()
            .any(|(_, p)| *p == SpinPoolNode::NAmide);
        let has_h_ind = evolved_pools
            .iter()
            .any(|(_, p)| *p == SpinPoolNode::HAliphatic);
        let c_pool = evolved_pools.iter().find_map(|(_, p)| match p {
            SpinPoolNode::CCarbonyl | SpinPoolNode::CAlpha | SpinPoolNode::CAliphatic => Some(*p),
            _ => None,
        });

        if has_noesy_mixing && has_h_ind {
            return if has_n {
                PulprogSequenceFamily::N15Noesy
            } else {
                PulprogSequenceFamily::C13Noesy
            };
        }
        if has_tocsy_spinlock {
            if has_n {
                return if has_h_ind {
                    PulprogSequenceFamily::Hccconh
                } else {
                    PulprogSequenceFamily::Ccconh
                };
            }
            return PulprogSequenceFamily::HcchTocsy;
        }
        if has_n && has_h_ind && visited_co {
            return PulprogSequenceFamily::Hbhaconh;
        }

        if has_n {
            if let Some(cp) = c_pool {
                match cp {
                    SpinPoolNode::CCarbonyl => {
                        return if visited_ca && (has_co_ca_transfer || has_explicit_ca_jump) {
                            PulprogSequenceFamily::Hncaco
                        } else {
                            PulprogSequenceFamily::Hnco
                        };
                    }
                    SpinPoolNode::CAlpha => {
                        return if visited_co && (has_co_ca_transfer || has_explicit_co_jump) {
                            PulprogSequenceFamily::Hncoca
                        } else {
                            PulprogSequenceFamily::Hnca
                        };
                    }
                    SpinPoolNode::CAliphatic => {
                        let first_f2 = low_exec.find("(p3 ").or_else(|| low_exec.find("):f2"));
                        let first_f3 = low_exec.find("(p21 ").or_else(|| low_exec.find("):f3"));
                        let c_before_n =
                            matches!((first_f2, first_f3), (Some(c), Some(n)) if c < n);
                        return if has_explicit_co_jump
                            || (visited_co && c_before_n)
                            || (has_1h_13c_inept && has_co_ca_transfer)
                            || (visited_co && visited_ca && visited_cacb && has_dual_cc_relays)
                        {
                            PulprogSequenceFamily::Cbcaconh
                        } else {
                            PulprogSequenceFamily::Hncacb
                        };
                    }
                    _ => {}
                }
            }
        }
    }

    PulprogSequenceFamily::Unknown
}

// ============================================================================
// Section 5: Varian / Agilent `procpar` Interpreter & Unified Entry Points
// ============================================================================

#[derive(Clone, Debug)]
struct VarianParam {
    values: Vec<String>,
}

impl VarianParam {
    fn first_str(&self) -> Option<&str> {
        self.values.first().map(String::as_str)
    }
    fn first_f64(&self) -> Option<f64> {
        self.first_str()?.parse::<f64>().ok()
    }
    fn first_usize(&self) -> Option<usize> {
        self.first_f64().map(|v| v.max(0.0).round() as usize)
    }
}

fn parse_varian_procpar_text(text: &str) -> Result<HashMap<String, VarianParam>, PulprogError> {
    if (text.len() as u64) > MAX_TEXT_FILE_BYTES {
        return Err(PulprogError::ResourceLimitExceeded(format!(
            "Varian procpar ({} bytes) exceeds {} byte safety cap",
            text.len(),
            MAX_TEXT_FILE_BYTES
        )));
    }
    let mut map = HashMap::new();
    let mut lines = text.lines().peekable();

    while let Some(raw) = lines.next() {
        let header = raw.trim();
        if header.is_empty() {
            continue;
        }
        let toks: Vec<&str> = header.split_whitespace().collect();
        if toks.len() < 11 {
            continue;
        }
        let name = toks[0].to_string();
        let Ok(_subtype) = toks[1].parse::<i32>() else {
            continue;
        };
        let Ok(basic_type) = toks[2].parse::<i32>() else {
            continue;
        };

        let Some(val_line) = lines.next() else {
            break;
        };
        let mut vt = val_line.trim();
        let count_str = vt.split_whitespace().next().unwrap_or("0");
        let Ok(n_vals) = count_str.parse::<usize>() else {
            continue;
        };
        if n_vals > MAX_PROCPAR_VALUES {
            return Err(PulprogError::ResourceLimitExceeded(format!(
                "Varian procpar parameter '{name}' value count ({n_vals}) exceeds limit ({MAX_PROCPAR_VALUES})"
            )));
        }
        vt = vt[count_str.len()..].trim_start();

        let mut values = Vec::with_capacity(n_vals.min(1024));
        if basic_type == 2 {
            // Quoted string values (first on `val_line`, subsequent `n_vals - 1` on following lines)
            if n_vals > 0 {
                values.push(unquote_procpar(vt));
            }
            for _ in 1..n_vals {
                let Some(nl) = lines.next() else {
                    break;
                };
                values.push(unquote_procpar(nl.trim()));
            }
        } else {
            for tok in vt.split_whitespace() {
                if values.len() < n_vals {
                    values.push(tok.to_string());
                }
            }
            while values.len() < n_vals {
                let Some(nl) = lines.next() else {
                    break;
                };
                for tok in nl.split_whitespace() {
                    if values.len() < n_vals {
                        values.push(tok.to_string());
                    }
                }
            }
        }
        // Skip enum constraint line (`0` or `N "opt1" ...`)
        let _ = lines.next();
        map.insert(name, VarianParam { values });
    }
    Ok(map)
}

fn unquote_procpar(s: &str) -> String {
    let t = s.trim();
    if let Some(inner) = t.strip_prefix('"') {
        if let Some(end) = inner.rfind('"') {
            return inner[..end].to_string();
        }
    }
    t.to_string()
}

/// Interpret Varian/Agilent `procpar` text (and optional `sampling.sch`) into a unified `PulprogInterpretation`.
#[allow(clippy::too_many_lines)]
pub fn interpret_varian_procpar(
    procpar_text: &str,
    sampling_sch_text: Option<&str>,
) -> Result<PulprogInterpretation, PulprogError> {
    let pp = parse_varian_procpar_text(procpar_text)?;
    if pp.is_empty() {
        return Err(PulprogError::MissingParameters(
            "empty or invalid Varian procpar".to_string(),
        ));
    }

    let seqfil = pp
        .get("seqfil")
        .and_then(VarianParam::first_str)
        .filter(|s| !s.is_empty())
        .or_else(|| pp.get("pslabel").and_then(VarianParam::first_str))
        .unwrap_or("")
        .to_string();
    let seq_low = seqfil.to_ascii_lowercase();

    let sfrq = pp
        .get("sfrq")
        .and_then(VarianParam::first_f64)
        .unwrap_or(600.0);
    let dfrq = pp.get("dfrq").and_then(VarianParam::first_f64);
    let dfrq2 = pp.get("dfrq2").and_then(VarianParam::first_f64);
    let dfrq3 = pp.get("dfrq3").and_then(VarianParam::first_f64);

    let tn = canonicalize_nucleus(
        pp.get("tn")
            .and_then(VarianParam::first_str)
            .unwrap_or("H1"),
        Some(sfrq),
        None,
    );
    let dn = pp
        .get("dn")
        .and_then(VarianParam::first_str)
        .map(|s| canonicalize_nucleus(s, dfrq, Some(sfrq)))
        .unwrap_or_default();
    let dn2 = pp
        .get("dn2")
        .and_then(VarianParam::first_str)
        .map(|s| canonicalize_nucleus(s, dfrq2, Some(sfrq)))
        .unwrap_or_default();
    let dn3 = pp
        .get("dn3")
        .and_then(VarianParam::first_str)
        .map(|s| canonicalize_nucleus(s, dfrq3, Some(sfrq)))
        .unwrap_or_default();

    let ni = pp.get("ni").and_then(VarianParam::first_usize).unwrap_or(0);
    let ni2 = pp
        .get("ni2")
        .and_then(VarianParam::first_usize)
        .unwrap_or(0);
    let ni3 = pp
        .get("ni3")
        .and_then(VarianParam::first_usize)
        .unwrap_or(0);

    let array_raw = pp
        .get("array")
        .and_then(VarianParam::first_str)
        .unwrap_or("")
        .replace(['(', ')'], "");
    let varian_array_nesting: Vec<String> = array_raw
        .split([',', ';', ' '])
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect();

    // Check arrayed `gzlvl1` / `gzlvl` DOSY diffusion!
    let gzlvl1_vals = pp.get("gzlvl1").map(|p| p.values.len()).unwrap_or(0);
    let is_dosy = gzlvl1_vals > 1
        || varian_array_nesting
            .iter()
            .any(|v| v.to_ascii_lowercase().starts_with("gzlvl"))
        || seq_low.starts_with("dbppste")
        || seq_low.starts_with("doneshot")
        || seq_low.contains("dosy");

    let mut indirect_dims = Vec::new();
    let resolve_var_ind = |slot: usize,
                           ni_pts: usize,
                           sw_key: &str,
                           phase_key: &str,
                           fcoef_key: &str,
                           f180_key: &str,
                           refsrc_key: &str,
                           rfl_key: &str,
                           rfp_key: &str|
     -> Option<IndirectDimPhysics> {
        if ni_pts <= 1 {
            return None;
        }
        let sw_hz = pp
            .get(sw_key)
            .and_then(VarianParam::first_f64)
            .unwrap_or(2000.0);
        let refsrc = pp
            .get(refsrc_key)
            .and_then(VarianParam::first_str)
            .unwrap_or("");
        let (nuc, obs_mhz) = match refsrc {
            "dfrq" if !dn.is_empty() => (dn.clone(), dfrq.unwrap_or(150.0)),
            "dfrq2" if !dn2.is_empty() => (dn2.clone(), dfrq2.unwrap_or(60.0)),
            "dfrq3" if !dn3.is_empty() => (dn3.clone(), dfrq3.unwrap_or(60.0)),
            _ => {
                if slot == 1 && ni2 <= 1 {
                    if seq_low.contains("nhsqc")
                        || seq_low.contains("n15")
                        || seq_low.contains("15n")
                        || seq_low.contains("gt1")
                        || seq_low.contains("gt2")
                    {
                        (
                            if dn2.is_empty() {
                                "15N".to_string()
                            } else {
                                dn2.clone()
                            },
                            dfrq2.unwrap_or(sfrq * 0.1013),
                        )
                    } else if let Some(df2) = dfrq2 {
                        if df2 > 10.0
                            && (15.0..=45.0).contains(&(sw_hz / df2))
                            && !seq_low.contains("chsqc")
                        {
                            (
                                if dn2.is_empty() {
                                    "15N".to_string()
                                } else {
                                    dn2.clone()
                                },
                                df2,
                            )
                        } else {
                            (
                                if dn.is_empty() {
                                    "13C".to_string()
                                } else {
                                    dn.clone()
                                },
                                dfrq.unwrap_or(sfrq * 0.2514),
                            )
                        }
                    } else {
                        (
                            if dn.is_empty() {
                                "13C".to_string()
                            } else {
                                dn.clone()
                            },
                            dfrq.unwrap_or(sfrq * 0.2514),
                        )
                    }
                } else if slot == 1 {
                    if seq_low.contains("noesynhsqc")
                        || seq_low.contains("hc_co_nh")
                        || seq_low.contains("hccconh")
                    {
                        (tn.clone(), sfrq)
                    } else {
                        (
                            if dn.is_empty() {
                                "13C".to_string()
                            } else {
                                dn.clone()
                            },
                            dfrq.unwrap_or(sfrq * 0.2514),
                        )
                    }
                } else if slot == 2 {
                    if seq_low.contains("hcchtocsy") || seq_low.contains("noesyc") {
                        (tn.clone(), sfrq)
                    } else {
                        (
                            if dn2.is_empty() {
                                "15N".to_string()
                            } else {
                                dn2.clone()
                            },
                            dfrq2.unwrap_or(sfrq * 0.1013),
                        )
                    }
                } else {
                    (
                        if dn3.is_empty() {
                            tn.clone()
                        } else {
                            dn3.clone()
                        },
                        dfrq3.unwrap_or(sfrq),
                    )
                }
            }
        };

        let sw_ppm = sw_hz / obs_mhz.max(1e-6);
        let carrier_ppm = match (
            pp.get(rfl_key).and_then(VarianParam::first_f64),
            pp.get(rfp_key).and_then(VarianParam::first_f64),
        ) {
            (Some(rfl), Some(rfp)) => ((sw_hz - rfl + rfp) - 0.5 * sw_hz) / obs_mhz.max(1e-6),
            _ => {
                if nuc == "15N" {
                    119.0
                } else if nuc == "13C" {
                    if seq_low.contains("hnco") || seq_low.contains("hncaco") {
                        176.0
                    } else if seq_low.contains("hncacb") || seq_low.contains("cbca") {
                        43.0
                    } else {
                        55.0
                    }
                } else {
                    4.7
                }
            }
        };

        let fcoef_opt = pp
            .get(fcoef_key)
            .or_else(|| (slot == 1).then(|| pp.get("fcoef")).flatten())
            .and_then(VarianParam::first_str)
            .and_then(parse_fcoef_8);
        let phase_vals: Vec<i32> = pp
            .get(phase_key)
            .map(|p| {
                p.values
                    .iter()
                    .filter_map(|v| v.parse::<f64>().ok().map(|x| x.round() as i32))
                    .collect()
            })
            .unwrap_or_default();

        let (quad_mode, rance_kay, reverse_frequency) = decode_varian_quad_and_rk(
            &phase_vals,
            fcoef_opt,
            pp.get("gzlvl1").and_then(VarianParam::first_f64),
            pp.get("gzlvl2").and_then(VarianParam::first_f64),
        );

        let f180 = pp
            .get(f180_key)
            .and_then(VarianParam::first_str)
            .is_some_and(|s| s.eq_ignore_ascii_case("y"));
        let dwell_s = 1.0 / sw_hz.max(1e-6);
        let initial_delay_s = if f180 { 0.5 * dwell_s } else { 0.0 };
        let (phc0, phc1) = if f180 { (-90.0, 180.0) } else { (0.0, 0.0) };
        let folding = match quad_mode {
            DecodedQuadMode::Qf | DecodedQuadMode::Qseq => FoldingConvention::Magnitude,
            DecodedQuadMode::StatesTppi | DecodedQuadMode::Tppi => {
                FoldingConvention::MirrorInvertedWrap
            }
            _ => FoldingConvention::SameSignWrap,
        };
        let mult = if matches!(quad_mode, DecodedQuadMode::Qf | DecodedQuadMode::Tppi) {
            1
        } else {
            2
        };

        Some(IndirectDimPhysics {
            f_dim: slot,
            acqu_source: format!("procpar:d{}", slot + 1),
            nucleus: nuc,
            carrier_ppm,
            sw_ppm,
            sw_hz,
            obs_mhz,
            td_raw: ni_pts * mult,
            complex_points: ni_pts,
            quad_mode,
            quad_source: QuadSource::VarianPhaseArray,
            reverse_frequency,
            rance_kay,
            constant_time: seq_low.contains("ct") || slot == 2,
            initial_delay_s,
            increment_s: dwell_s,
            half_dwell_sampling: f180,
            predicted_phc0_deg: phc0,
            predicted_phc1_deg: phc1,
            folding,
        })
    };

    if let Some(d1) = resolve_var_ind(
        1,
        ni,
        "sw1",
        "phase",
        "f1coef",
        "f1180",
        "refsource1",
        "rfl1",
        "rfp1",
    ) {
        indirect_dims.push(d1);
    }
    if let Some(d2) = resolve_var_ind(
        2,
        ni2,
        "sw2",
        "phase2",
        "f2coef",
        "f2180",
        "refsource2",
        "rfl2",
        "rfp2",
    ) {
        indirect_dims.push(d2);
    }
    if let Some(d3) = resolve_var_ind(
        3,
        ni3,
        "sw3",
        "phase3",
        "f3coef",
        "f3180",
        "refsource3",
        "rfl3",
        "rfp3",
    ) {
        indirect_dims.push(d3);
    }

    let non_phase_vars: Vec<&str> = varian_array_nesting
        .iter()
        .map(String::as_str)
        .filter(|s| {
            !s.starts_with("phase")
                && !s.eq_ignore_ascii_case("izar")
                && !matches!(*s, "d2" | "d3" | "d4")
        })
        .collect();

    let dimensionality = if is_dosy && indirect_dims.is_empty() {
        2
    } else if !non_phase_vars.is_empty() && indirect_dims.len() == 1 {
        3
    } else {
        1 + indirect_dims.len()
    };

    let aqseq = match indirect_dims.len() {
        0 => AcquisitionOrder::Dim1D,
        1 => AcquisitionOrder::Dim2D,
        2 => {
            if array_raw.replace(' ', "") == "phase2,phase" {
                AcquisitionOrder::VarianPhase1Fastest
            } else {
                AcquisitionOrder::Aqseq321
            }
        }
        _ => AcquisitionOrder::Dim4D,
    };

    let (pseudo_3d_order, pseudo_3d_list_kind, pseudo_3d_list_inner_to_quad) = if is_dosy {
        (
            Pseudo3DStorageOrder::PlaneQueued,
            Some("gzlvl1".to_string()),
            None,
        )
    } else if let Some(&var) = non_phase_vars.first() {
        // In Varian VnmrJ `psg`, implicit `d2` (`ni`) runs *inside* explicit non-phase array variables
        // (`array = "relaxT,phase"` or `"relaxT,d2,phase"` acquires plane-by-plane -> `PlaneQueued`);
        // interleaving `var` inside `t1` requires explicit `array = "d2,relaxT,phase"` (`dp < var_pos` -> `T1Interleaved`).
        let d2_pos = varian_array_nesting.iter().position(|v| v == "d2");
        let var_pos = varian_array_nesting
            .iter()
            .position(|v| v == var)
            .unwrap_or(0);
        let phase_pos = varian_array_nesting
            .iter()
            .position(|v| v.starts_with("phase"));
        if d2_pos.is_some_and(|dp| dp < var_pos) {
            let inner_to_quad = phase_pos.map(|pp_idx| pp_idx < var_pos);
            (
                Pseudo3DStorageOrder::T1Interleaved,
                Some(var.to_string()),
                inner_to_quad,
            )
        } else {
            (
                Pseudo3DStorageOrder::PlaneQueued,
                Some(var.to_string()),
                None,
            )
        }
    } else {
        (Pseudo3DStorageOrder::NotPseudo3D, None, None)
    };

    let nus_schedule = if let Some(sch_txt) = sampling_sch_text {
        let grid_dims: Vec<usize> = indirect_dims.iter().map(|d| d.complex_points).collect();
        if !grid_dims.is_empty() {
            Some(decode_nuslist_str(
                sch_txt,
                &grid_dims,
                AcquisitionOrder::Aqseq321,
            )?)
        } else {
            None
        }
    } else {
        None
    };

    // Varian magnetization-transfer graph
    let mut active_nuclei: BTreeSet<String> = BTreeSet::new();
    active_nuclei.insert(tn.clone());
    for d in &indirect_dims {
        active_nuclei.insert(d.nucleus.clone());
    }
    let evolved_pools: Vec<(usize, SpinPoolNode)> = indirect_dims
        .iter()
        .map(|d| {
            let pool = match d.nucleus.as_str() {
                "15N" => SpinPoolNode::NAmide,
                "13C" => {
                    if d.carrier_ppm > 145.0 {
                        SpinPoolNode::CCarbonyl
                    } else if d.sw_ppm >= 48.0 || (34.0..=48.9).contains(&d.carrier_ppm) {
                        SpinPoolNode::CAliphatic
                    } else {
                        SpinPoolNode::CAlpha
                    }
                }
                _ => SpinPoolNode::HAliphatic,
            };
            (d.f_dim, pool)
        })
        .collect();

    let deduced_family =
        deduce_varian_family(&seq_low, is_dosy, &non_phase_vars, &tn, &evolved_pools, &pp);

    let gzlvl1 = pp
        .get("gzlvl1")
        .and_then(VarianParam::first_f64)
        .unwrap_or(0.0);
    let gzlvl2 = pp
        .get("gzlvl2")
        .and_then(VarianParam::first_f64)
        .unwrap_or(0.0);
    let sensitivity_enhanced = indirect_dims
        .iter()
        .any(|d| d.quad_mode == DecodedQuadMode::EchoAntiecho);

    Ok(PulprogInterpretation {
        vendor: PulprogVendor::Varian,
        pulse_program_name: seqfil,
        executed_source_present: false,
        active_defines: BTreeMap::new(),
        resolved_local_includes: Vec::new(),
        unresolved_standard_includes: Vec::new(),
        preprocessed_lines: 0,
        dimensionality,
        aqseq,
        pseudo_3d_order,
        pseudo_3d_list_kind,
        pseudo_3d_list_inner_to_quad,
        varian_array_nesting,
        direct_nucleus: tn,
        direct_group_delay_pts: 0.0,
        cnst: Vec::new(),
        delays_s: Vec::new(),
        increments_s: Vec::new(),
        loop_counters: Vec::new(),
        gpz_pct: vec![gzlvl1, gzlvl2],
        spnam: Vec::new(),
        cpdprg: Vec::new(),
        loops: Vec::new(),
        indirect_dims,
        nus_schedule,
        transfer_graph: MagnetizationTransferGraph {
            active_nuclei: active_nuclei.into_iter().collect(),
            carrier_jumps_ppm: Vec::new(),
            evolved_pools,
            detected_pool: if is_dosy {
                SpinPoolNode::HAliphatic
            } else {
                SpinPoolNode::HAmide
            },
            trosy_selected: seq_low.contains("trosy"),
            sensitivity_enhanced,
            diffusion_encoded: is_dosy,
            edges: Vec::new(),
            deduced_family,
        },
        warnings: Vec::new(),
    })
}

fn parse_fcoef_8(raw: &str) -> Option<[f64; 8]> {
    let vals: Vec<f64> = raw
        .split_whitespace()
        .filter_map(|t| t.parse::<f64>().ok())
        .collect();
    if vals.len() == 8 {
        let mut arr = [0.0; 8];
        arr.copy_from_slice(&vals);
        Some(arr)
    } else {
        None
    }
}

fn decode_varian_quad_and_rk(
    phase_vals: &[i32],
    fcoef: Option<[f64; 8]>,
    gzlvl1: Option<f64>,
    gzlvl2: Option<f64>,
) -> (DecodedQuadMode, Option<RanceKayMatrix>, bool) {
    if let Some(c) = fcoef {
        // Off-diagonal cross-mixing (`c[2]` or `c[3]` nonzero AND `c[5]` or `c[7]` nonzero) = Rance–Kay Echo-Antiecho!
        let is_ea = (c[0].abs() + c[1].abs()) > 0.0
            && (c[2].abs() + c[3].abs()) > 0.0
            && (c[4].abs() + c[5].abs()) > 0.0
            && (c[6].abs() + c[7].abs()) > 0.0;
        if is_ea {
            // Standard BioPack Rance–Kay: `"1 0 -1 0 0 1 0 1"` or `"1 0 -1 0 0 -1 0 -1"`.
            // Note: negating the sine row (`0 -1 0 -1` vs `0 +1 0 +1`) flips indirect frequency sign (`reverse_frequency`)!
            let sin_sign = c[5] + c[7];
            let cos_diff = c[0] - c[2];
            let inverted = (sin_sign < 0.0) ^ (cos_diff < 0.0);
            let ratio = match (gzlvl1, gzlvl2) {
                (Some(g1), Some(g2)) if g2.abs() > 1e-6 => Some(g1 / g2),
                _ => None,
            };
            return (
                DecodedQuadMode::EchoAntiecho,
                Some(RanceKayMatrix {
                    coef: c,
                    ea_polarity_inverted: inverted,
                    enc_grad_level: gzlvl1,
                    dec_grad_level: gzlvl2,
                    grad_ratio: ratio,
                }),
                c[5] < 0.0,
            );
        }
        let rev = c[4] < 0.0 || c[6] < 0.0;
        if phase_vals == [1, 3] {
            return (DecodedQuadMode::StatesTppi, None, rev);
        }
        return (DecodedQuadMode::States, None, rev);
    }

    match phase_vals {
        [1, 2] => (DecodedQuadMode::States, None, false),
        [1, 3] => (DecodedQuadMode::StatesTppi, None, false),
        [1, 2, 3, 4] => (DecodedQuadMode::Tppi, None, false),
        _ => (DecodedQuadMode::Qf, None, false),
    }
}

fn deduce_varian_family(
    seq_low: &str,
    is_dosy: bool,
    non_phase_vars: &[&str],
    tn: &str,
    evolved_pools: &[(usize, SpinPoolNode)],
    pp: &HashMap<String, VarianParam>,
) -> PulprogSequenceFamily {
    if is_dosy {
        return PulprogSequenceFamily::Dosy;
    }
    if let Some(&v) = non_phase_vars.first() {
        let vl = v.to_ascii_lowercase();
        if vl.contains("ncyc") || vl.contains("cpmg") {
            return PulprogSequenceFamily::Cpmg;
        }
        if vl.contains("sat") || vl.contains("frq") {
            return PulprogSequenceFamily::Cest;
        }
        if seq_low.contains("t2") || vl.contains("t2") {
            return PulprogSequenceFamily::RelaxR2;
        }
        if seq_low.contains("noe") && !seq_low.contains("noesy") {
            return PulprogSequenceFamily::HetNoe;
        }
        return PulprogSequenceFamily::RelaxR1;
    }
    if seq_low == "ghnha"
        || seq_low == "hnha"
        || seq_low.starts_with("hnhagp")
        || seq_low.contains("caha")
    {
        return PulprogSequenceFamily::Unknown;
    }
    let has_ind_n = evolved_pools
        .iter()
        .any(|(_, p)| *p == SpinPoolNode::NAmide);
    let has_ind_c = evolved_pools.iter().any(|(_, p)| {
        matches!(
            *p,
            SpinPoolNode::CCarbonyl | SpinPoolNode::CAlpha | SpinPoolNode::CAliphatic
        )
    });
    let triple_res_axes_ok = evolved_pools.len() < 2 || (has_ind_n && has_ind_c && tn == "1H");
    if triple_res_axes_ok {
        if (seq_low.contains("cbca") && seq_low.contains("co"))
            || seq_low.contains("hncocacb")
            || seq_low.contains("hn_co_cacb")
        {
            return PulprogSequenceFamily::Cbcaconh;
        }
        if seq_low.contains("hncacb") || seq_low.contains("hncab") {
            return PulprogSequenceFamily::Hncacb;
        }
        if seq_low.contains("hncaco") || seq_low.contains("hn_ca_co") {
            return PulprogSequenceFamily::Hncaco;
        }
        if seq_low.contains("hncoca") || seq_low.contains("hn_co_ca") {
            return PulprogSequenceFamily::Hncoca;
        }
        if seq_low.contains("hnca") {
            return PulprogSequenceFamily::Hnca;
        }
        if seq_low.contains("hnco") {
            return PulprogSequenceFamily::Hnco;
        }
    }
    if seq_low.contains("nhsqc")
        || (evolved_pools.len() == 1 && evolved_pools[0].1 == SpinPoolNode::NAmide && tn == "1H")
    {
        return if seq_low.contains("trosy") {
            PulprogSequenceFamily::N15Trosy
        } else {
            PulprogSequenceFamily::N15Hsqc
        };
    }
    if seq_low.contains("chmqc") || seq_low.contains("hmqc") {
        return PulprogSequenceFamily::C13Hmqc;
    }
    if seq_low.contains("chsqc") || seq_low == "ghsqc" || (evolved_pools.len() == 1 && tn == "1H") {
        // Note: `ghsqc` with `15N` indirect is `N15Hsqc`, with `13C` indirect is `C13Hsqc`!
        if evolved_pools
            .first()
            .is_some_and(|(_, p)| *p == SpinPoolNode::NAmide)
        {
            return PulprogSequenceFamily::N15Hsqc;
        }
        return PulprogSequenceFamily::C13Hsqc;
    }
    if evolved_pools.len() == 2 {
        let has_n = evolved_pools
            .iter()
            .any(|(_, p)| *p == SpinPoolNode::NAmide);
        let c_pool = evolved_pools.iter().find_map(|(_, p)| match p {
            SpinPoolNode::CCarbonyl | SpinPoolNode::CAlpha | SpinPoolNode::CAliphatic => Some(*p),
            _ => None,
        });
        if has_n {
            if let Some(cp) = c_pool {
                let dof_co = pp
                    .get("dof")
                    .and_then(VarianParam::first_f64)
                    .is_some_and(|v| v.abs() > 5000.0);
                return match cp {
                    SpinPoolNode::CCarbonyl => PulprogSequenceFamily::Hnco,
                    SpinPoolNode::CAlpha => PulprogSequenceFamily::Hnca,
                    SpinPoolNode::CAliphatic => {
                        if dof_co && seq_low.contains("co") {
                            PulprogSequenceFamily::Cbcaconh
                        } else {
                            PulprogSequenceFamily::Hncacb
                        }
                    }
                    _ => PulprogSequenceFamily::Unknown,
                };
            }
        }
    }
    PulprogSequenceFamily::Unknown
}

/// Interpret a Varian/Agilent `.fid` directory on disk (`procpar` + optional `sampling.sch` / `nuslist`),
/// enforcing sandbox containment, file size caps, and ISO-8859-1 / Latin-1 lossy decoding.
pub fn interpret_varian_dir(dir: impl AsRef<Path>) -> Result<PulprogInterpretation, PulprogError> {
    let dir = dir.as_ref();
    let canon_root = canonicalize_exp_dir(dir)?;
    let procpar_path = find_relative_child_ci(dir, "procpar").ok_or_else(|| {
        PulprogError::MissingParameters(format!("missing Varian procpar in {}", dir.display()))
    })?;
    let procpar_text = read_sandboxed_text_lossy(&canon_root, &procpar_path, "procpar")?;
    let sch_text = match find_relative_child_ci(dir, "sampling.sch")
        .map(|p| (p, "sampling.sch"))
        .or_else(|| find_relative_child_ci(dir, "nuslist").map(|p| (p, "nuslist")))
    {
        Some((p, nm)) => {
            let s = read_sandboxed_text_lossy(&canon_root, &p, nm)?;
            (!s.trim().is_empty()).then_some(s)
        }
        None => None,
    };
    interpret_varian_procpar(&procpar_text, sch_text.as_deref())
}

/// Unified auto-detecting experiment directory interpreter (Bruker `acqus` vs Varian `procpar`).
pub fn interpret_experiment_dir(
    dir: impl AsRef<Path>,
) -> Result<PulprogInterpretation, PulprogError> {
    let dir = dir.as_ref();
    if find_relative_child_ci(dir, "acqus").is_some()
        || find_relative_child_ci(dir, "acqu").is_some()
    {
        interpret_bruker_dir(dir)
    } else if find_relative_child_ci(dir, "procpar").is_some() {
        interpret_varian_dir(dir)
    } else {
        Err(PulprogError::MissingParameters(format!(
            "no Bruker acqus or Varian procpar found in {}",
            dir.display()
        )))
    }
}

// ============================================================================
// Section 6: Empirical `{sign, fold}` HSQC Strip-Matching Verification (Q1)
// ============================================================================

/// Klein four-group ($\mathbb{Z}_2 \times \mathbb{Z}_2$) discrete frequency-bin transformation
/// on an indirect axis of a processed 3D spectrum.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AxisTransformOp {
    /// Identity (`j -> j`).
    Canonical,
    /// Frequency-sign reversal around carrier `N/2` and Nyquist `0` (`j -> (N - j) % N`, i.e. `REVERSE` toggle).
    SignFlip,
    /// Circular half-spectral-width shift (`j -> (j + N/2) % N`, i.e. `States` <-> `States-TPPI` `(-1)^k` toggle).
    HalfSwShift,
    /// Combined sign reversal + half-SW shift (`TPPI` / mirror-folded axis).
    MirrorFold,
}

impl AxisTransformOp {
    /// Map destination bin `dst` (`0..n`) in the transformed spectrum to source bin `src` (`0..n`) in the input spectrum.
    #[inline]
    pub fn src_for_dst(self, n: usize, dst: usize) -> usize {
        if n == 0 {
            return 0;
        }
        let half = n / 2;
        match self {
            Self::Canonical => dst % n,
            Self::SignFlip => (n - (dst % n)) % n,
            Self::HalfSwShift => (dst + half) % n,
            Self::MirrorFold => {
                let shifted = (dst + half) % n;
                (n - shifted) % n
            }
        }
    }

    /// Inverse of [`Self::src_for_dst`]: given a peak at `src` (`0..n`) in the input spectrum,
    /// return its destination bin `dst` (`0..n`) after applying `self`.
    #[inline]
    pub fn dst_for_src(self, n: usize, src: usize) -> usize {
        if n == 0 {
            return 0;
        }
        let half = n / 2;
        let s = src % n;
        match self {
            Self::Canonical => s,
            Self::SignFlip => (n - s) % n,
            Self::HalfSwShift => (s + n - half) % n,
            Self::MirrorFold => {
                let t = (n - s) % n;
                (t + n - half) % n
            }
        }
    }
}

/// Named indirect-axis `{sign, fold}` hypothesis evaluated against a reference `2D 15N-HSQC`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NamedAxisHypothesis {
    Canonical,
    SignFlipN,
    HalfSwShiftN,
    MirrorFoldN,
    SignFlipC,
    HalfSwShiftC,
    MirrorFoldC,
    Joint {
        op_n: AxisTransformOp,
        op_c: AxisTransformOp,
    },
}

impl NamedAxisHypothesis {
    pub fn from_ops(op_n: AxisTransformOp, op_c: AxisTransformOp) -> Self {
        match (op_n, op_c) {
            (AxisTransformOp::Canonical, AxisTransformOp::Canonical) => Self::Canonical,
            (AxisTransformOp::SignFlip, AxisTransformOp::Canonical) => Self::SignFlipN,
            (AxisTransformOp::HalfSwShift, AxisTransformOp::Canonical) => Self::HalfSwShiftN,
            (AxisTransformOp::MirrorFold, AxisTransformOp::Canonical) => Self::MirrorFoldN,
            (AxisTransformOp::Canonical, AxisTransformOp::SignFlip) => Self::SignFlipC,
            (AxisTransformOp::Canonical, AxisTransformOp::HalfSwShift) => Self::HalfSwShiftC,
            (AxisTransformOp::Canonical, AxisTransformOp::MirrorFold) => Self::MirrorFoldC,
            _ => Self::Joint { op_n, op_c },
        }
    }

    pub fn ops(self) -> (AxisTransformOp, AxisTransformOp) {
        match self {
            Self::Canonical => (AxisTransformOp::Canonical, AxisTransformOp::Canonical),
            Self::SignFlipN => (AxisTransformOp::SignFlip, AxisTransformOp::Canonical),
            Self::HalfSwShiftN => (AxisTransformOp::HalfSwShift, AxisTransformOp::Canonical),
            Self::MirrorFoldN => (AxisTransformOp::MirrorFold, AxisTransformOp::Canonical),
            Self::SignFlipC => (AxisTransformOp::Canonical, AxisTransformOp::SignFlip),
            Self::HalfSwShiftC => (AxisTransformOp::Canonical, AxisTransformOp::HalfSwShift),
            Self::MirrorFoldC => (AxisTransformOp::Canonical, AxisTransformOp::MirrorFold),
            Self::Joint { op_n, op_c } => (op_n, op_c),
        }
    }
}

/// Quantitative log-likelihood & root-recovery breakdown for a single `{sign, fold}` hypothesis.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AxisHypothesisScore {
    pub hypothesis: NamedAxisHypothesis,
    pub op_n: AxisTransformOp,
    pub op_c: AxisTransformOp,
    pub ln_likelihood_n: f64,
    pub ln_likelihood_c: f64,
    pub ln_prior: f64,
    pub ln_posterior: f64,
    pub matched_roots: usize,
    pub total_roots: usize,
    pub matched_root_fraction: f64,
    pub best_dh_ppm: f64,
    pub best_dn_ppm: f64,
}

/// Summary returned by [`verify_3d_axis_hypotheses_against_hsqc`] and [`evaluate_3d_axis_hypotheses_against_hsqc`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AxisHypothesisVerification {
    pub selected: NamedAxisHypothesis,
    pub applied: bool,
    pub polarity_flipped: bool,
    /// Log-Bayes-factor gain $\ln P(H_{\text{selected}} \mid D) - \ln P(H_{\text{canonical}} \mid D)$ in nats.
    pub ln_bayes_factor_vs_canonical: f64,
    pub canonical_matched_fraction: f64,
    pub selected_matched_fraction: f64,
    pub candidates: Vec<AxisHypothesisScore>,
}

/// Minimum log-posterior (complexity-penalized log-Bayes-factor) gain in nats required to override `Canonical`.
pub const MIN_AXIS_RESCUE_LN_BF: f64 = 12.0;

/// Apply `(op_n, op_c)` indirect voxel permutations in place on `spec_3d` (`Y = 15N`, `Z = 13C`),
/// followed by experiment-appropriate polarity reconciliation when `reconcile_polarity` is `true`.
pub fn apply_indirect_axis_transform_3d(
    spec_3d: &mut Spectrum3D,
    op_n: AxisTransformOp,
    op_c: AxisTransformOp,
    exp_override: Option<&str>,
    reconcile_polarity: bool,
) -> bool {
    let nx = spec_3d.axis_x.size;
    let ny = spec_3d.axis_y.size;
    let nz = spec_3d.axis_z.size;
    if nx == 0 || ny == 0 || nz == 0 || spec_3d.data.len() != nz * ny * nx {
        return false;
    }

    if op_n != AxisTransformOp::Canonical && ny >= 2 {
        let plane_elems = ny * nx;
        let mut scratch_plane = vec![0.0_f32; plane_elems];
        for iz in 0..nz {
            let p_off = iz * plane_elems;
            scratch_plane.copy_from_slice(&spec_3d.data[p_off..p_off + plane_elems]);
            for dst_y in 0..ny {
                let src_y = op_n.src_for_dst(ny, dst_y);
                spec_3d.data[p_off + dst_y * nx..p_off + (dst_y + 1) * nx]
                    .copy_from_slice(&scratch_plane[src_y * nx..(src_y + 1) * nx]);
            }
        }
    }

    if op_c != AxisTransformOp::Canonical && nz >= 2 {
        let plane_elems = ny * nx;
        let old_data = spec_3d.data.clone();
        for dst_z in 0..nz {
            let src_z = op_c.src_for_dst(nz, dst_z);
            spec_3d.data[dst_z * plane_elems..(dst_z + 1) * plane_elems]
                .copy_from_slice(&old_data[src_z * plane_elems..(src_z + 1) * plane_elems]);
        }
    }

    if reconcile_polarity
        && (op_n != AxisTransformOp::Canonical || op_c != AxisTransformOp::Canonical)
    {
        let exp_owned = exp_override
            .map(String::from)
            .unwrap_or_else(|| spec_3d.exp.clone());
        reconcile_3d_polarity_in_place(spec_3d, &exp_owned)
    } else {
        false
    }
}

fn reconcile_3d_polarity_in_place(spec_3d: &mut Spectrum3D, exp: &str) -> bool {
    let norm = exp.trim().replace(['(', ')'], "").to_ascii_uppercase();
    let nx = spec_3d.axis_x.size;
    let ny = spec_3d.axis_y.size;
    let nz = spec_3d.axis_z.size;
    if nx == 0 || ny == 0 || nz == 0 {
        return false;
    }

    if norm == "HNCACB" {
        // Ensure `CA` core band (`50.5..=58.0 ppm`, excluding Ser/Thr CB `>= 58.5 ppm`) has positive
        // net strong energy, while guarding `CA`-bleached paramagnetic `HNCACB` (where `CA` is empty
        // and `CB` in `15.5..=43.0 ppm` is already negative) against false negation!
        let mut ca_pos_e = 0.0_f64;
        let mut ca_neg_e = 0.0_f64;
        let mut cb_pos_e = 0.0_f64;
        let mut cb_neg_e = 0.0_f64;
        let mut max_abs = 0.0_f32;
        for &v in &spec_3d.data {
            let a = v.abs();
            if a > max_abs {
                max_abs = a;
            }
        }
        let thr = (0.08 * max_abs) as f64;
        if thr <= 0.0 {
            return false;
        }
        for iz in 0..nz {
            let c_ppm = spec_3d.axis_z.index_to_ppm(iz as f64);
            let in_ca = (50.5..=58.0).contains(&c_ppm);
            let in_cb = (15.5..=43.0).contains(&c_ppm);
            if !in_ca && !in_cb {
                continue;
            }
            let plane = &spec_3d.data[iz * ny * nx..(iz + 1) * ny * nx];
            for &v in plane {
                let vf = v as f64;
                if vf > thr {
                    if in_ca {
                        ca_pos_e += vf * vf;
                    } else {
                        cb_pos_e += vf * vf;
                    }
                } else if vf < -thr {
                    if in_ca {
                        ca_neg_e += vf * vf;
                    } else {
                        cb_neg_e += vf * vf;
                    }
                }
            }
        }
        let cb_already_canonical =
            cb_neg_e > cb_pos_e * 2.0 && (ca_pos_e + ca_neg_e) < 0.20 * cb_neg_e;
        if !cb_already_canonical && ca_neg_e > ca_pos_e * 1.25 {
            for v in &mut spec_3d.data {
                *v = -*v;
            }
            return true;
        }
        false
    } else {
        // All-positive triple-resonance experiments (`HNCO`, `HNCA`, `CBCACONH`, `HNCOCA`, `HNCACO`)
        let mut max_abs = 0.0_f32;
        for &v in &spec_3d.data {
            let a = v.abs();
            if a > max_abs {
                max_abs = a;
            }
        }
        let thr = (0.12 * max_abs) as f64;
        if thr <= 0.0 {
            return false;
        }
        let mut pos_e = 0.0_f64;
        let mut neg_e = 0.0_f64;
        for &v in &spec_3d.data {
            let vf = v as f64;
            if vf > thr {
                pos_e += vf * vf;
            } else if vf < -thr {
                neg_e += vf * vf;
            }
        }
        if neg_e > pos_e * 1.35 {
            for v in &mut spec_3d.data {
                *v = -*v;
            }
            return true;
        }
        false
    }
}

/// Extract distinct positive 2D `(1H, 15N)` root peaks `(h_ppm, n_ppm, weight)` from a processed `2D HSQC`.
pub fn extract_hsqc_2d_roots_for_verification(hsqc_2d: &Spectrum2D) -> Vec<(f64, f64, f64)> {
    let nx = hsqc_2d.axis_x.size;
    let ny = hsqc_2d.axis_y.size;
    if nx < 4 || ny < 4 || hsqc_2d.data.len() != ny * nx {
        return Vec::new();
    }
    let max2 = hsqc_2d
        .data
        .iter()
        .map(|&v| v as f64)
        .fold(0.0_f64, f64::max);
    if max2 <= 0.0 {
        return Vec::new();
    }
    let mut scratch: Vec<f64> = hsqc_2d.data.iter().map(|&v| v as f64).collect();
    let med2 = quick_median(&mut scratch);
    for v in &mut scratch {
        *v = (*v - med2).abs();
    }
    let sigma2 = (1.4826 * quick_median(&mut scratch)).max(0.015 * max2);
    let thresh2 = (med2 + 4.5 * sigma2).max(0.05 * max2);

    let mut raw_pts: Vec<(f64, f64, f64)> = Vec::new();
    for iy in 1..ny - 1 {
        let n_ppm = hsqc_2d.axis_y.index_to_ppm(iy as f64);
        for ix in 1..nx - 1 {
            let h_ppm = hsqc_2d.axis_x.index_to_ppm(ix as f64);
            // Focus on backbone amide proton envelope (`5.4..11.8 ppm`) to avoid water-ridge artifacts
            if !(5.4..=11.8).contains(&h_ppm) {
                continue;
            }
            let v = hsqc_2d.data[iy * nx + ix] as f64;
            if v <= thresh2 {
                continue;
            }
            if v >= hsqc_2d.data[(iy - 1) * nx + ix] as f64
                && v >= hsqc_2d.data[(iy + 1) * nx + ix] as f64
                && v >= hsqc_2d.data[iy * nx + ix - 1] as f64
                && v >= hsqc_2d.data[iy * nx + ix + 1] as f64
            {
                let snr = (v - med2) / sigma2;
                let w = (snr / 8.0).clamp(0.6, 1.4);
                raw_pts.push((snr, h_ppm, n_ppm));
                let _ = w;
            }
        }
    }
    raw_pts.sort_by(|a, b| b.0.total_cmp(&a.0));

    // Non-maximum suppression within `(0.03 ppm 1H, 0.30 ppm 15N)` so split/shoulder pixels don't double-count
    let mut roots: Vec<(f64, f64, f64)> = Vec::new();
    for (snr, h, n) in raw_pts {
        if roots
            .iter()
            .any(|&(rh, rn, _)| (rh - h).abs() < 0.03 && (rn - n).abs() < 0.30)
        {
            continue;
        }
        let w = (snr / 10.0).clamp(0.65, 1.35);
        roots.push((h, n, w));
        if roots.len() >= 240 {
            break;
        }
    }
    roots
}

/// Evaluate all `{sign, fold}` indirect-axis hypotheses of `spec_3d` against `hsqc_2d` without mutating `spec_3d`.
pub fn evaluate_3d_axis_hypotheses_against_hsqc(
    spec_3d: &Spectrum3D,
    hsqc_2d: &Spectrum2D,
    exp_override: Option<&str>,
) -> AxisHypothesisVerification {
    let weighted_roots = extract_hsqc_2d_roots_for_verification(hsqc_2d);
    evaluate_3d_axis_hypotheses_with_weighted_roots(spec_3d, &weighted_roots, exp_override)
}

/// Evaluate all `{sign, fold}` indirect-axis hypotheses of `spec_3d` against picked `(H, N)` `Peak` roots without mutating `spec_3d`.
pub fn evaluate_3d_axis_hypotheses_against_roots(
    spec_3d: &Spectrum3D,
    hsqc_roots: &[Peak],
    exp_override: Option<&str>,
) -> AxisHypothesisVerification {
    let weighted: Vec<(f64, f64, f64)> = hsqc_roots
        .iter()
        .filter(|p| (5.4..=11.8).contains(&p.h))
        .take(240)
        .map(|p| (p.h, p.n, 1.0))
        .collect();
    evaluate_3d_axis_hypotheses_with_weighted_roots(spec_3d, &weighted, exp_override)
}

/// Verify and automatically rescue inverted (`SignFlip`), half-SW-shifted (`States` <-> `States-TPPI` `HalfSwShift`),
/// or mirror-folded (`MirrorFold`) indirect `15N`/`13C` axes of `spec_3d` against a `2D 15N-HSQC` when the
/// log-Bayes-factor gain exceeds [`MIN_AXIS_RESCUE_LN_BF`]. Never mutates clean canonical spectra.
pub fn verify_3d_axis_hypotheses_against_hsqc(
    spec_3d: &mut Spectrum3D,
    hsqc_2d: &Spectrum2D,
    exp_override: Option<&str>,
) -> AxisHypothesisVerification {
    let mut ver = evaluate_3d_axis_hypotheses_against_hsqc(spec_3d, hsqc_2d, exp_override);
    if ver.selected != NamedAxisHypothesis::Canonical {
        let (op_n, op_c) = ver.selected.ops();
        let flipped = apply_indirect_axis_transform_3d(spec_3d, op_n, op_c, exp_override, true);
        ver.applied = true;
        ver.polarity_flipped = flipped;
    }
    ver
}

/// Same as [`verify_3d_axis_hypotheses_against_hsqc`], accepting pre-picked `2D HSQC` `Peak` roots directly.
pub fn verify_3d_axis_hypotheses_against_roots(
    spec_3d: &mut Spectrum3D,
    hsqc_roots: &[Peak],
    exp_override: Option<&str>,
) -> AxisHypothesisVerification {
    let mut ver = evaluate_3d_axis_hypotheses_against_roots(spec_3d, hsqc_roots, exp_override);
    if ver.selected != NamedAxisHypothesis::Canonical {
        let (op_n, op_c) = ver.selected.ops();
        let flipped = apply_indirect_axis_transform_3d(spec_3d, op_n, op_c, exp_override, true);
        ver.applied = true;
        ver.polarity_flipped = flipped;
    }
    ver
}

#[allow(clippy::too_many_lines)]
fn evaluate_3d_axis_hypotheses_with_weighted_roots(
    spec_3d: &Spectrum3D,
    roots: &[(f64, f64, f64)],
    exp_override: Option<&str>,
) -> AxisHypothesisVerification {
    let nx = spec_3d.axis_x.size;
    let ny = spec_3d.axis_y.size;
    let nz = spec_3d.axis_z.size;
    let total_roots = roots.len();

    if nx < 4 || ny < 4 || nz < 4 || total_roots < 5 || spec_3d.data.len() != nz * ny * nx {
        return AxisHypothesisVerification {
            selected: NamedAxisHypothesis::Canonical,
            applied: false,
            polarity_flipped: false,
            ln_bayes_factor_vs_canonical: 0.0,
            canonical_matched_fraction: 0.0,
            selected_matched_fraction: 0.0,
            candidates: Vec::new(),
        };
    }

    // 1. Build 2D `(1H, 15N)` max-abs sky-projection `proj[iy * nx + ix]` and 3D noise floor
    let plane_elems = ny * nx;
    let mut proj = vec![0.0_f64; plane_elems];
    for plane in spec_3d.data.chunks_exact(plane_elems).take(nz) {
        for (dst, &s) in proj.iter_mut().zip(plane.iter()) {
            let a = (s as f64).abs();
            if a > *dst {
                *dst = a;
            }
        }
    }
    let max_proj = proj.iter().copied().fold(0.0_f64, f64::max);
    if max_proj <= 0.0 {
        return AxisHypothesisVerification {
            selected: NamedAxisHypothesis::Canonical,
            applied: false,
            polarity_flipped: false,
            ln_bayes_factor_vs_canonical: 0.0,
            canonical_matched_fraction: 0.0,
            selected_matched_fraction: 0.0,
            candidates: Vec::new(),
        };
    }

    let mut scratch = proj.clone();
    let med_p = quick_median(&mut scratch);
    for v in &mut scratch {
        *v = (*v - med_p).abs();
    }
    let sigma_p = (1.4826 * quick_median(&mut scratch)).max(0.015 * max_proj);

    // Estimate 3D per-voxel thermal noise `sigma_3d` from a strided subsample of `spec_3d.data`
    let step_sub = (spec_3d.data.len() / 8192).max(1);
    let mut sub_vox: Vec<f64> = spec_3d
        .data
        .iter()
        .step_by(step_sub)
        .map(|&v| (v as f64).abs())
        .collect();
    let sigma_3d = (1.4826 * quick_median(&mut sub_vox)).max(0.01 * max_proj);

    // Empirical background box hit probability `p_bg` on the amide region of `proj`
    let h_lo = spec_3d.axis_x.min_ppm().max(6.0);
    let h_hi = spec_3d.axis_x.max_ppm().min(10.5);
    let n_lo = spec_3d.axis_y.min_ppm();
    let n_hi = spec_3d.axis_y.max_ppm();
    let mut bg_hits = 0usize;
    let mut bg_tot = 0usize;
    if h_hi > h_lo && n_hi > n_lo {
        for gh in 0..16 {
            let h_q = h_lo + (gh as f64 + 0.5) / 16.0 * (h_hi - h_lo);
            for gn in 0..24 {
                let n_q = n_lo + (gn as f64 + 0.5) / 24.0 * (n_hi - n_lo);
                bg_tot += 1;
                let snr_q = box_max_snr_on_proj(
                    &proj,
                    &spec_3d.axis_x,
                    &spec_3d.axis_y,
                    AxisTransformOp::Canonical,
                    h_q,
                    n_q,
                    0.038,
                    0.36,
                    med_p,
                    sigma_p,
                );
                if snr_q >= 3.5 {
                    bg_hits += 1;
                }
            }
        }
    }
    let p_bg = if bg_tot > 0 {
        (bg_hits as f64 / bg_tot as f64).clamp(0.025, 0.45)
    } else {
        0.10
    };
    let p_sig = 0.82_f64.max(p_bg + 0.25).min(0.97);
    let ln_hit_odds = (p_sig / p_bg).ln();
    let ln_miss_odds = ((1.0 - p_sig) / (1.0 - p_bg)).ln();

    // 2. Evaluate all 4 `op_n` operations (`Canonical`, `SignFlip`, `HalfSwShift`, `MirrorFold`),
    // optimizing small cross-experiment carrier drift `(dh, dn)` in `[-0.12, +0.12] x [-0.60, +0.60] ppm`!
    let ops_all = [
        AxisTransformOp::Canonical,
        AxisTransformOp::SignFlip,
        AxisTransformOp::HalfSwShift,
        AxisTransformOp::MirrorFold,
    ];

    #[derive(Clone)]
    struct NResult {
        ll_n: f64,
        matched_roots: usize,
        dh: f64,
        dn: f64,
        /// Raw `(ix_src, iy_src)` coordinates in `spec_3d` for each matched root!
        matched_raw_coords: Vec<(usize, usize)>,
    }

    let mut n_results: Vec<(AxisTransformOp, NResult)> = Vec::with_capacity(4);
    for &op_n in &ops_all {
        let mut best_ll = f64::NEG_INFINITY;
        let mut best_dh = 0.0_f64;
        let mut best_dn = 0.0_f64;

        // Coarse `(dh, dn)` scan (`9 x 13` = 117 points, ultra-fast on `<= 240` roots)
        for ih in -4..=4 {
            let dh = ih as f64 * 0.03;
            for in_i in -6..=6 {
                let dn = in_i as f64 * 0.10;
                let mut ll = 0.0_f64;
                for &(hr, nr, wr) in roots {
                    let snr = box_max_snr_on_proj(
                        &proj,
                        &spec_3d.axis_x,
                        &spec_3d.axis_y,
                        op_n,
                        hr - dh,
                        nr - dn,
                        0.038,
                        0.36,
                        med_p,
                        sigma_p,
                    );
                    if snr >= 3.5 {
                        ll += wr * (ln_hit_odds + 0.35 * (snr - 3.5).min(12.0));
                    } else {
                        ll += wr * ln_miss_odds;
                    }
                }
                if ll > best_ll {
                    best_ll = ll;
                    best_dh = dh;
                    best_dn = dn;
                }
            }
        }

        // Collect matched roots and exact apex `(ix, iy_src)` at `(best_dh, best_dn)`
        let mut matched_roots = 0usize;
        let mut matched_raw_coords = Vec::new();
        let mut refined_ll = 0.0_f64;
        for &(hr, nr, wr) in roots {
            let (snr, best_ix_iy) = box_max_snr_and_raw_coord_on_proj(
                &proj,
                &spec_3d.axis_x,
                &spec_3d.axis_y,
                op_n,
                hr - best_dh,
                nr - best_dn,
                0.038,
                0.36,
                med_p,
                sigma_p,
            );
            if snr >= 3.5 {
                matched_roots += 1;
                refined_ll += wr * (ln_hit_odds + 0.35 * (snr - 3.5).min(12.0));
                if let Some(coord) = best_ix_iy {
                    matched_raw_coords.push(coord);
                }
            } else {
                refined_ll += wr * ln_miss_odds;
            }
        }

        n_results.push((
            op_n,
            NResult {
                ll_n: refined_ll,
                matched_roots,
                dh: best_dh,
                dn: best_dn,
                matched_raw_coords,
            },
        ));
    }

    // 3. For each `op_n`, extract `13C` strip extrema at `matched_raw_coords` and score all 4 `op_c` hypotheses!
    let exp_str = exp_override.unwrap_or(&spec_3d.exp);
    let c13_regime = classify_c13_regime(exp_str, &spec_3d.axis_z);
    let mut candidates: Vec<AxisHypothesisScore> = Vec::with_capacity(16);
    let mut strip_peaks_by_n: Vec<(AxisTransformOp, Vec<StripPeak13C>)> = Vec::with_capacity(4);

    for (op_n, ref n_res) in &n_results {
        let strip_peaks =
            extract_13c_strip_extrema_at_coords(spec_3d, &n_res.matched_raw_coords, sigma_3d);
        for &op_c in &ops_all {
            let ll_c = score_13c_axis_prior(&strip_peaks, &spec_3d.axis_z, op_c, exp_str);
            let op_prior = |op: AxisTransformOp| -> f64 {
                match op {
                    AxisTransformOp::Canonical => 0.0,
                    // Single physical flag mismatch (`REVERSE` sign flip OR `States <-> States-TPPI` toggle)
                    AxisTransformOp::SignFlip | AxisTransformOp::HalfSwShift => -6.0,
                    // Compound 2-flag mismatch (`SignFlip + HalfSwShift`) carries an Occam complexity penalty
                    // so unimodal symmetric axes (e.g. `HNCO` `13CO`) never invent a gratuitous sign flip!
                    AxisTransformOp::MirrorFold => -9.5,
                }
            };
            let ln_prior = op_prior(*op_n) + op_prior(op_c);
            let ln_posterior = n_res.ll_n + ll_c + ln_prior;
            let matched_root_fraction = (n_res.matched_roots as f64) / (total_roots as f64);

            candidates.push(AxisHypothesisScore {
                hypothesis: NamedAxisHypothesis::from_ops(*op_n, op_c),
                op_n: *op_n,
                op_c,
                ln_likelihood_n: n_res.ll_n,
                ln_likelihood_c: ll_c,
                ln_prior,
                ln_posterior,
                matched_roots: n_res.matched_roots,
                total_roots,
                matched_root_fraction,
                best_dh_ppm: n_res.dh,
                best_dn_ppm: n_res.dn,
            });
        }
        strip_peaks_by_n.push((*op_n, strip_peaks));
    }

    let canon_score = candidates
        .iter()
        .find(|c| c.hypothesis == NamedAxisHypothesis::Canonical)
        .cloned()
        .unwrap();

    // Sort candidates descending by `ln_posterior` and select the highest-posterior
    // non-canonical candidate that passes BOTH the Bayes-factor threshold and physical
    // plausibility gates (`accept_n` and `accept_c`). Evaluating across `sorted` ensures
    // that if `(SignFlipN, SignFlipC)` fails `accept_c` on a `CA`-bleached paramagnetic
    // `HNCACB`, `(SignFlipN, CanonicalC)` is cleanly selected instead of aborting `SignFlipN`!
    let mut sorted = candidates.clone();
    sorted.sort_by(|a, b| b.ln_posterior.total_cmp(&a.ln_posterior));

    let accepted_cand = sorted.iter().find(|cand| {
        if cand.hypothesis == NamedAxisHypothesis::Canonical {
            return false;
        }
        let bf_gain = cand.ln_posterior - canon_score.ln_posterior;
        if bf_gain < MIN_AXIS_RESCUE_LN_BF {
            return false;
        }
        let accept_n = if cand.op_n != AxisTransformOp::Canonical {
            cand.matched_roots >= 5
                && cand.matched_root_fraction >= 0.40
                && cand.matched_root_fraction >= canon_score.matched_root_fraction + 0.20
        } else {
            true
        };
        if !accept_n {
            return false;
        }
        if cand.op_c != AxisTransformOp::Canonical {
            let same_n_canon_c = candidates
                .iter()
                .find(|c| c.op_n == cand.op_n && c.op_c == AxisTransformOp::Canonical)
                .unwrap_or(&canon_score);
            let cand_n_peaks = strip_peaks_by_n
                .iter()
                .find(|(op, _)| *op == cand.op_n)
                .map(|(_, pks)| pks.as_slice())
                .unwrap_or(&[]);
            let canon_c_diag = compute_c13_band_diag(
                cand_n_peaks,
                &spec_3d.axis_z,
                AxisTransformOp::Canonical,
                c13_regime,
            );
            let cand_c_diag =
                compute_c13_band_diag(cand_n_peaks, &spec_3d.axis_z, cand.op_c, c13_regime);
            cand.matched_roots >= 5
                && (cand.ln_likelihood_c - same_n_canon_c.ln_likelihood_c) >= 18.0
                && is_13c_axis_rescue_physically_justified(
                    c13_regime,
                    cand.op_c,
                    &canon_c_diag,
                    &cand_c_diag,
                )
        } else {
            true
        }
    });

    let (selected, selected_frac, ln_bf) = if let Some(win) = accepted_cand {
        (
            win.hypothesis,
            win.matched_root_fraction,
            win.ln_posterior - canon_score.ln_posterior,
        )
    } else {
        (
            NamedAxisHypothesis::Canonical,
            canon_score.matched_root_fraction,
            0.0,
        )
    };

    AxisHypothesisVerification {
        selected,
        applied: false,
        polarity_flipped: false,
        ln_bayes_factor_vs_canonical: ln_bf,
        canonical_matched_fraction: canon_score.matched_root_fraction,
        selected_matched_fraction: selected_frac,
        candidates,
    }
}

#[allow(clippy::too_many_arguments)]
fn box_max_snr_on_proj(
    proj: &[f64],
    ax_x: &AxisPpm,
    ax_y: &AxisPpm,
    op_n: AxisTransformOp,
    h_ppm: f64,
    n_ppm: f64,
    tol_h: f64,
    tol_n: f64,
    med_p: f64,
    sigma_p: f64,
) -> f64 {
    box_max_snr_and_raw_coord_on_proj(
        proj, ax_x, ax_y, op_n, h_ppm, n_ppm, tol_h, tol_n, med_p, sigma_p,
    )
    .0
}

#[allow(clippy::too_many_arguments)]
fn box_max_snr_and_raw_coord_on_proj(
    proj: &[f64],
    ax_x: &AxisPpm,
    ax_y: &AxisPpm,
    op_n: AxisTransformOp,
    h_ppm: f64,
    n_ppm: f64,
    tol_h: f64,
    tol_n: f64,
    med_p: f64,
    sigma_p: f64,
) -> (f64, Option<(usize, usize)>) {
    let nx = ax_x.size;
    let ny = ax_y.size;
    if h_ppm < ax_x.min_ppm() - tol_h
        || h_ppm > ax_x.max_ppm() + tol_h
        || n_ppm < ax_y.min_ppm() - tol_n
        || n_ppm > ax_y.max_ppm() + tol_n
    {
        return (0.0, None);
    }
    let ix_a = ax_x.ppm_to_index(h_ppm + tol_h);
    let ix_b = ax_x.ppm_to_index(h_ppm - tol_h);
    let iy_a = ax_y.ppm_to_index(n_ppm + tol_n);
    let iy_b = ax_y.ppm_to_index(n_ppm - tol_n);

    let x0 = ix_a.min(ix_b).floor().clamp(0.0, (nx - 1) as f64) as usize;
    let x1 = ix_a.max(ix_b).ceil().clamp(x0 as f64, (nx - 1) as f64) as usize;
    let y0 = iy_a.min(iy_b).floor().clamp(0.0, (ny - 1) as f64) as usize;
    let y1 = iy_a.max(iy_b).ceil().clamp(y0 as f64, (ny - 1) as f64) as usize;

    let mut best_v = med_p;
    let mut best_coord = None;
    for dst_y in y0..=y1 {
        let src_y = op_n.src_for_dst(ny, dst_y);
        let row = &proj[src_y * nx..(src_y + 1) * nx];
        for (ix, &v) in row.iter().enumerate().take(x1 + 1).skip(x0) {
            if v > best_v {
                best_v = v;
                best_coord = Some((ix, src_y));
            }
        }
    }
    let snr = ((best_v - med_p) / sigma_p).max(0.0);
    (snr, best_coord)
}

/// Extracted 1D `13C` strip peak `(raw_iz, signed_val, weight)`.
#[derive(Clone, Copy, Debug)]
struct StripPeak13C {
    raw_iz: usize,
    signed_val: f64,
    weight: f64,
}

fn extract_13c_strip_extrema_at_coords(
    spec_3d: &Spectrum3D,
    raw_coords: &[(usize, usize)],
    sigma_3d: f64,
) -> Vec<StripPeak13C> {
    let nx = spec_3d.axis_x.size;
    let ny = spec_3d.axis_y.size;
    let nz = spec_3d.axis_z.size;
    let plane_elems = ny * nx;
    let thr = 4.0 * sigma_3d;

    let mut out = Vec::new();
    let mut trace = vec![0.0_f64; nz];
    for &(ix_c, iy_c) in raw_coords {
        let x0 = ix_c.saturating_sub(1);
        let x1 = (ix_c + 1).min(nx - 1);
        let y0 = iy_c.saturating_sub(1);
        let y1 = (iy_c + 1).min(ny - 1);

        for (iz, tr_dst) in trace.iter_mut().enumerate() {
            let p_off = iz * plane_elems;
            let mut best_signed = 0.0_f64;
            for iy in y0..=y1 {
                for ix in x0..=x1 {
                    let v = spec_3d.data[p_off + iy * nx + ix] as f64;
                    if v.abs() > best_signed.abs() {
                        best_signed = v;
                    }
                }
            }
            *tr_dst = best_signed;
        }

        // Pick up to 4 local extrema along `0..nz` (including wrap-around boundary bins `0` and `nz-1`
        // so Nyquist-split peaks in half-SW-shifted cubes are captured faithfully!)
        let mut strip_cands: Vec<(f64, usize, f64)> = Vec::new();
        for iz in 0..nz {
            let v = trace[iz];
            let av = v.abs();
            if av < thr {
                continue;
            }
            let prev_a = if iz == 0 { 0.0 } else { trace[iz - 1].abs() };
            let next_a = if iz + 1 == nz {
                0.0
            } else {
                trace[iz + 1].abs()
            };
            if av >= prev_a && av >= next_a {
                strip_cands.push((av, iz, v));
            }
        }
        strip_cands.sort_by(|a, b| b.0.total_cmp(&a.0));
        for (av, iz, v) in strip_cands.into_iter().take(4) {
            let w = (av / (6.0 * sigma_3d)).clamp(0.5, 2.0);
            out.push(StripPeak13C {
                raw_iz: iz,
                signed_val: v,
                weight: w,
            });
        }
    }
    out
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum C13Regime {
    Carbonyl,
    CAlphaOnly,
    AliphaticCaCb { biphasic: bool },
    Uninformative,
}

fn classify_c13_regime(exp: &str, ax_z: &AxisPpm) -> C13Regime {
    let norm = exp.trim().replace(['(', ')'], "").to_ascii_uppercase();
    let sw_ppm = ax_z.step_ppm.abs() * (ax_z.size as f64);
    let mid_ppm = 0.5 * (ax_z.min_ppm() + ax_z.max_ppm());

    match norm.as_str() {
        "HNCO" | "HNCACO" => {
            if sw_ppm >= 7.5 {
                C13Regime::Carbonyl
            } else {
                C13Regime::Uninformative
            }
        }
        "HNCA" | "HNCOCA" => {
            if sw_ppm >= 22.0 {
                C13Regime::CAlphaOnly
            } else {
                C13Regime::Uninformative
            }
        }
        "HNCACB" => {
            if sw_ppm >= 44.0 {
                C13Regime::AliphaticCaCb { biphasic: true }
            } else {
                C13Regime::Uninformative
            }
        }
        "CBCACONH" => {
            if sw_ppm >= 44.0 {
                C13Regime::AliphaticCaCb { biphasic: false }
            } else {
                C13Regime::Uninformative
            }
        }
        _ => {
            if mid_ppm > 145.0 && sw_ppm >= 7.5 {
                C13Regime::Carbonyl
            } else if (34.0..=52.0).contains(&mid_ppm) && sw_ppm >= 44.0 {
                C13Regime::AliphaticCaCb { biphasic: false }
            } else if (48.0..=66.0).contains(&mid_ppm) && sw_ppm >= 22.0 {
                C13Regime::CAlphaOnly
            } else {
                C13Regime::Uninformative
            }
        }
    }
}

fn score_13c_axis_prior(
    peaks: &[StripPeak13C],
    ax_z: &AxisPpm,
    op_c: AxisTransformOp,
    exp: &str,
) -> f64 {
    let nz = ax_z.size;
    if peaks.len() < 6 || nz < 4 {
        return 0.0;
    }
    let regime = classify_c13_regime(exp, ax_z);
    if regime == C13Regime::Uninformative {
        return 0.0;
    }

    let sw_ppm = (ax_z.step_ppm.abs() * (nz as f64)).max(1.0);
    let ln_uniform = -(sw_ppm.ln());

    // Optimize `13C` referencing offset `dc in [-2.8, +2.8] ppm` so TMS-vs-DSS (`~2.7 ppm`) or
    // carrier offsets never bias `op_c` against `Canonical`:
    let dc_grid: &[f64] = if sw_ppm >= 11.0 {
        &[-2.8, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 2.8]
    } else {
        &[-0.8, -0.4, 0.0, 0.4, 0.8]
    };

    let mut best_ll = f64::NEG_INFINITY;
    for &dc in dc_grid {
        let mut ll = 0.0_f64;
        let mut ca_signed_sum = 0.0_f64;
        let mut ca_abs_sum = 0.0_f64;
        let mut cb_signed_sum = 0.0_f64;
        let mut cb_abs_sum = 0.0_f64;
        let mut edge_weight = 0.0_f64;
        let mut center_weight = 0.0_f64;
        let mut tot_weight = 0.0_f64;

        let lo_edge = ax_z.min_ppm() + 0.14 * sw_ppm;
        let hi_edge = ax_z.max_ppm() - 0.14 * sw_ppm;

        for pk in peaks {
            let dst_iz = op_c.dst_for_src(nz, pk.raw_iz);
            let c_raw = ax_z.index_to_ppm(dst_iz as f64);
            let c = c_raw - dc;
            tot_weight += pk.weight;
            if c_raw <= lo_edge || c_raw >= hi_edge {
                edge_weight += pk.weight;
            } else if (c_raw - 0.5 * (ax_z.min_ppm() + ax_z.max_ppm())).abs() <= 0.22 * sw_ppm {
                center_weight += pk.weight;
            }

            let ln_p = match regime {
                C13Regime::Carbonyl => {
                    // BMRB 13CO mixture (`175.8 +- 2.35 ppm` + Gly `173.4 +- 2.8 ppm`), allowing
                    // minor `+- SW` Nyquist folding in narrow (`< 16 ppm`) carbonyl windows so a
                    // `10-15%` folded Gly/beta-sheet tail on a TMS-referenced sample never mimics `HalfSwShiftC`.
                    let base = 0.88 * gauss_pdf(c, 175.8, 2.35) + 0.12 * gauss_pdf(c, 173.4, 2.80);
                    let folded = if sw_ppm < 16.0 {
                        let f_lo = 0.88 * gauss_pdf(c - sw_ppm, 175.8, 2.35)
                            + 0.12 * gauss_pdf(c - sw_ppm, 173.4, 2.80);
                        let f_hi = 0.88 * gauss_pdf(c + sw_ppm, 175.8, 2.35)
                            + 0.12 * gauss_pdf(c + sw_ppm, 173.4, 2.80);
                        0.30 * f_lo.max(f_hi)
                    } else {
                        0.0
                    };
                    (base.max(folded) + 1e-4).ln() - ln_uniform
                }
                C13Regime::CAlphaOnly => {
                    if !(41.5..=68.0).contains(&c) {
                        -6.5
                    } else {
                        let p_main = 0.90 * gauss_pdf(c, 56.6, 3.4);
                        let p_gly = 0.10 * gauss_pdf(c, 45.1, 1.4);
                        (p_main + p_gly + 1e-4).ln() - ln_uniform
                    }
                }
                C13Regime::AliphaticCaCb { .. } => {
                    if !(13.5..=72.2).contains(&c) {
                        -7.5
                    } else {
                        let p_ca = 0.44 * gauss_pdf(c, 56.5, 3.4);
                        let p_gly = 0.05 * gauss_pdf(c, 45.1, 1.4);
                        let p_cb_mid = 0.31 * gauss_pdf(c, 33.5, 5.2);
                        let p_cb_up = 0.12 * gauss_pdf(c, 20.2, 2.6);
                        let p_cb_st = 0.08 * gauss_pdf(c, 65.2, 3.2);
                        (p_ca + p_gly + p_cb_mid + p_cb_up + p_cb_st + 1e-4).ln() - ln_uniform
                    }
                }
                C13Regime::Uninformative => 0.0,
            };
            ll += pk.weight * ln_p;

            if matches!(regime, C13Regime::AliphaticCaCb { biphasic: true }) {
                if (51.0..=60.0).contains(&c) {
                    ca_signed_sum += pk.signed_val;
                    ca_abs_sum += pk.signed_val.abs();
                } else if (15.5..=42.0).contains(&c) {
                    cb_signed_sum += pk.signed_val;
                    cb_abs_sum += pk.signed_val.abs();
                }
            }
        }

        // Nyquist-wrap bimodal edge-pileup penalty on Carbonyl & CAlphaOnly (catches narrow-SW `HalfSwShift`
        // even when carrier is near `176 ppm` and `SW` is `10..14 ppm`!):
        if matches!(regime, C13Regime::Carbonyl | C13Regime::CAlphaOnly) && tot_weight > 0.0 {
            let edge_frac = edge_weight / tot_weight;
            let center_frac = center_weight / tot_weight;
            if edge_frac > 0.45 && center_frac < 0.25 {
                ll -= tot_weight * 1.4 * (edge_frac - center_frac);
            }
        }

        // Biphasic `CA` vs `CB` opposite-sign coherence reward for `HNCACB` (invariant to global +-1 flip!),
        // plus single-band negative-only `CA` penalty so `CA`-bleached paramagnetic `CB`-only cubes
        // never gain likelihood by reflecting negative `CB` (`~30 ppm`) into the positive `CA` (`~56 ppm`) band!
        if matches!(regime, C13Regime::AliphaticCaCb { biphasic: true }) {
            if ca_abs_sum > 0.0 && cb_abs_sum > 0.08 * ca_abs_sum && ca_abs_sum > 0.08 * cb_abs_sum
            {
                let mean_ca_sign = ca_signed_sum / ca_abs_sum;
                let mean_cb_sign = cb_signed_sum / cb_abs_sum;
                // Under true `HNCACB` C-axis: `mean_ca_sign ~ +1, mean_cb_sign ~ -1` (or `-1` and `+1`) -> `sep ~ 2.0`!
                let sep = (mean_ca_sign - mean_cb_sign).abs();
                ll += 0.35 * tot_weight * (sep - 1.0);
            } else if ca_abs_sum > 0.0 && cb_abs_sum <= 0.08 * ca_abs_sum {
                let mean_ca_sign = ca_signed_sum / ca_abs_sum;
                if mean_ca_sign < -0.35 {
                    ll -= 0.90 * tot_weight;
                }
            }
        }

        if ll > best_ll {
            best_ll = ll;
        }
    }

    best_ll
}

#[derive(Clone, Copy, Debug, Default)]
struct C13BandDiag {
    oob_frac: f64,
    edge_frac: f64,
    lo_edge_frac: f64,
    hi_edge_frac: f64,
    center_frac: f64,
    low_aliph_frac: f64,
    ca_core_frac: f64,
    ser_thr_frac: f64,
    biphasic_sep: f64,
    signed_biphasic_sep: f64,
}

fn compute_c13_band_diag(
    peaks: &[StripPeak13C],
    ax_z: &AxisPpm,
    op_c: AxisTransformOp,
    regime: C13Regime,
) -> C13BandDiag {
    let nz = ax_z.size;
    if peaks.is_empty() || nz < 4 {
        return C13BandDiag::default();
    }
    let sw_ppm = (ax_z.step_ppm.abs() * (nz as f64)).max(1.0);
    let mid_ppm = 0.5 * (ax_z.min_ppm() + ax_z.max_ppm());
    let lo_edge = ax_z.min_ppm() + 0.14 * sw_ppm;
    let hi_edge = ax_z.max_ppm() - 0.14 * sw_ppm;

    let mut tot_w = 0.0_f64;
    let mut oob_w = 0.0_f64;
    let mut edge_w = 0.0_f64;
    let mut lo_edge_w = 0.0_f64;
    let mut hi_edge_w = 0.0_f64;
    let mut center_w = 0.0_f64;
    let mut low_aliph_w = 0.0_f64;
    let mut ca_core_w = 0.0_f64;
    let mut ser_thr_w = 0.0_f64;
    let mut ca_signed = 0.0_f64;
    let mut ca_abs = 0.0_f64;
    let mut cb_signed = 0.0_f64;
    let mut cb_abs = 0.0_f64;

    for pk in peaks {
        let dst_iz = op_c.dst_for_src(nz, pk.raw_iz);
        let c_raw = ax_z.index_to_ppm(dst_iz as f64);
        tot_w += pk.weight;

        let is_oob = match regime {
            C13Regime::Carbonyl => !(166.5..=184.5).contains(&c_raw),
            C13Regime::CAlphaOnly => !(41.0..=68.5).contains(&c_raw),
            C13Regime::AliphaticCaCb { .. } => !(13.5..=72.2).contains(&c_raw),
            C13Regime::Uninformative => false,
        };
        if is_oob {
            oob_w += pk.weight;
        }
        if c_raw <= lo_edge {
            edge_w += pk.weight;
            lo_edge_w += pk.weight;
        } else if c_raw >= hi_edge {
            edge_w += pk.weight;
            hi_edge_w += pk.weight;
        }
        if (c_raw - mid_ppm).abs() <= 0.22 * sw_ppm {
            center_w += pk.weight;
        }
        if (15.5..=43.5).contains(&c_raw) {
            low_aliph_w += pk.weight;
            cb_signed += pk.signed_val;
            cb_abs += pk.signed_val.abs();
        } else if (50.0..=60.5).contains(&c_raw) {
            ca_core_w += pk.weight;
            ca_signed += pk.signed_val;
            ca_abs += pk.signed_val.abs();
        } else if (60.5..=72.2).contains(&c_raw) {
            ser_thr_w += pk.weight;
        }
    }

    if tot_w <= 0.0 {
        return C13BandDiag::default();
    }
    let signed_biphasic_sep = if ca_abs > 0.0 && cb_abs > 0.0 {
        ca_signed / ca_abs - cb_signed / cb_abs
    } else {
        0.0
    };
    let biphasic_sep = signed_biphasic_sep.abs();
    C13BandDiag {
        oob_frac: oob_w / tot_w,
        edge_frac: edge_w / tot_w,
        lo_edge_frac: lo_edge_w / tot_w,
        hi_edge_frac: hi_edge_w / tot_w,
        center_frac: center_w / tot_w,
        low_aliph_frac: low_aliph_w / tot_w,
        ca_core_frac: ca_core_w / tot_w,
        ser_thr_frac: ser_thr_w / tot_w,
        biphasic_sep,
        signed_biphasic_sep,
    }
}

fn is_13c_axis_rescue_physically_justified(
    regime: C13Regime,
    op_c: AxisTransformOp,
    canon_d: &C13BandDiag,
    best_d: &C13BandDiag,
) -> bool {
    match regime {
        C13Regime::Uninformative => false,
        C13Regime::Carbonyl | C13Regime::CAlphaOnly => {
            // For unimodal `13CO` (`166.5..184.5 ppm`) and `13CA` (`41.0..68.5 ppm`), `SignFlip` or
            // `MirrorFold` is forbidden when `Canonical` already places peaks inside the physical
            // envelope (preventing false mirroring of e.g. `<13CO> ≈ 172 ppm` β-sheet/PCS/TMS `HNCO`!).
            let oob_rescue =
                canon_d.oob_frac >= 0.25 && (canon_d.oob_frac - best_d.oob_frac) >= 0.20;
            let nyquist_split_rescue = canon_d.edge_frac >= 0.45
                && canon_d.lo_edge_frac >= 0.18
                && canon_d.hi_edge_frac >= 0.18
                && canon_d.center_frac <= 0.16
                && best_d.center_frac >= 0.50
                && (canon_d.edge_frac - best_d.edge_frac) >= 0.25
                && (best_d.center_frac - canon_d.center_frac) >= 0.30
                && best_d.oob_frac <= canon_d.oob_frac + 0.05;
            match op_c {
                AxisTransformOp::Canonical => true,
                AxisTransformOp::SignFlip | AxisTransformOp::MirrorFold => oob_rescue,
                AxisTransformOp::HalfSwShift => oob_rescue || nyquist_split_rescue,
            }
        }
        C13Regime::AliphaticCaCb { biphasic } => {
            let oob_rescue =
                canon_d.oob_frac >= 0.20 && (canon_d.oob_frac - best_d.oob_frac) >= 0.15;
            let nyquist_split_rescue = canon_d.edge_frac >= 0.28
                && canon_d.lo_edge_frac >= 0.06
                && canon_d.hi_edge_frac >= 0.06
                && (canon_d.edge_frac - best_d.edge_frac) >= 0.15
                && best_d.oob_frac <= canon_d.oob_frac + 0.05;
            // Two-band swap requires BOTH `CB` (`15.5..43.5`) and `CA`/`high-aliphatic` to be
            // substantively populated under `Canonical` AND under `best` (with `CA` core `50..60.5`
            // populated under `Canonical` for `SignFlip` so `CA`-bleached paramagnetic `CB`-only
            // cubes with or without Ser/Thr `CB` are never reflected into `CA`!), never increasing
            // OOB peaks, and either recovering a depleted `CA` core (`>= 1.35x` and `+0.10`,
            // matching `is_mirrored_aliphatic_cacb_3d`) or fixing inverted biphasic `CA`/`CB` signs.
            let canon_upper_ok = if op_c == AxisTransformOp::SignFlip {
                canon_d.ca_core_frac >= 0.15
            } else {
                (canon_d.ca_core_frac + canon_d.ser_thr_frac) >= 0.18
            };
            let ca_core_gain = best_d.ca_core_frac >= canon_d.ca_core_frac * 1.35
                && (best_d.ca_core_frac - canon_d.ca_core_frac) >= 0.10;
            let swap_justified = if biphasic {
                best_d.biphasic_sep >= 1.30 && (canon_d.signed_biphasic_sep < 0.60 || ca_core_gain)
            } else {
                ca_core_gain
            };
            let two_band_swap_rescue = canon_d.low_aliph_frac >= 0.18
                && canon_upper_ok
                && best_d.low_aliph_frac >= 0.18
                && best_d.ca_core_frac >= 0.20
                && best_d.oob_frac <= canon_d.oob_frac + 0.03
                && swap_justified;
            oob_rescue || nyquist_split_rescue || two_band_swap_rescue
        }
    }
}

#[inline]
fn gauss_pdf(x: f64, mu: f64, sigma: f64) -> f64 {
    let z = (x - mu) / sigma;
    (-0.5 * z * z).exp() / ((2.0 * PI).sqrt() * sigma)
}

fn quick_median(vals: &mut [f64]) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    let mid = vals.len() / 2;
    let (_, &mut m, _) = vals.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    m
}
