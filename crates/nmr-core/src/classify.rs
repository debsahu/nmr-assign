//! Spectrometer folder & archive auto-classifier (`nmrassign inspect`, `/api/v1/inspect`,
//! and `wasm32`-compatible drag-and-drop inventory engine).
//!
//! Provenance: Clean-room original implementation of `docs/specs/12-workbench-ux.md` §1,
//! grounded in published multidimensional NMR pulse-sequence physics and open format specs:
//! - Kay, L. E., Ikura, M., Tschudin, R., & Bax, A. (*J. Magn. Reson.* 1990, 89, 496–514) —
//!   triple-resonance `HNCO`, `HNCA`, and `HN(CO)CA` magnetization transfer and axis windows.
//! - Grzesiek, S., & Bax, A. (*J. Am. Chem. Soc.* 1992, 114, 6291–6293) — `CBCA(CO)NH` and
//!   `HNCACB` aliphatic `13CA`/`13CB` correlation windows.
//! - Bermel, W., Bertini, I., Felli, I. C., Lee, Y.-M., Luchinat, C., & Pierattelli, R.
//!   (*J. Am. Chem. Soc.* 2009, 131, 15339–15345) — protonless `13C`-detected `CON`, `CACO`,
//!   `CANCO`, and `CBCACO` experiments for intrinsically disordered and paramagnetic proteins.
//! - Tugarinov, V., Hwang, P. M., Ollerenshaw, J. E., & Kay, L. E. (*J. Am. Chem. Soc.* 2003,
//!   125, 10120–10121) — `1H-13C` methyl-TROSY `HMQC` spectroscopy.
//! - Delaglio, F., et al. (*J. Biomol. NMR* 1995, 6, 277–293) — NMRPipe 512-float header layout.
//! - Deutsch, P. (RFC 1951 & RFC 1952, May 1996) and IEEE Std 1003.1-1988 (`ustar`) — DEFLATE,
//!   GZIP, and POSIX tar archive header formats.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs::{self, File};
use std::io::{self, BufReader, Cursor, Read, Seek, SeekFrom};
use std::path::Path;

use serde::{Deserialize, Serialize};

/// Maximum header bytes read from text parameter/schedule files (`acqus`, `procpar`, `nuslist`).
const MAX_TEXT_HEADER_BYTES: usize = 256 * 1024;
/// Maximum header bytes read from `.str`, `.nef`, or `.fasta` files during workspace discovery.
const MAX_TABLE_SNIFF_BYTES: usize = 16 * 1024;
/// UCSF Sparky v2 header prefix (`180` global + up to `4 * 128` axis bytes = `692` bytes).
const MAX_UCSF_HEADER_BYTES: usize = 1024;
/// NMRPipe 512-float header byte length.
const NMRPIPE_HEADER_BYTES: usize = 2048;

/// `wasm32`-compatible virtual file entry supplying a relative path, bounded header bytes,
/// and total byte footprint without loading bulk binary FID or spectrum payloads.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VirtualEntry {
    /// Forward-slash relative path inside the workspace or archive.
    pub relative_path: String,
    /// Bounded header bytes (empty for skipped binary signal payloads such as `ser`/`fid`/`3rrr`).
    #[serde(default)]
    pub header_bytes: Vec<u8>,
    /// Full file size in bytes (`header_bytes.len()` when omitted or zero on non-empty `header_bytes`).
    #[serde(default)]
    pub total_bytes: u64,
}

impl VirtualEntry {
    /// Construct an in-memory `VirtualEntry` whose full contents are in `header_bytes`.
    pub fn new(relative_path: impl Into<String>, header_bytes: Vec<u8>) -> Self {
        let total_bytes = header_bytes.len() as u64;
        Self {
            relative_path: relative_path.into(),
            header_bytes,
            total_bytes,
        }
    }

    /// Construct a header-only `VirtualEntry` recording `total_bytes` without retaining payload bytes.
    pub fn header_only(
        relative_path: impl Into<String>,
        header_bytes: Vec<u8>,
        total_bytes: u64,
    ) -> Self {
        let effective = total_bytes.max(header_bytes.len() as u64);
        Self {
            relative_path: relative_path.into(),
            header_bytes,
            total_bytes: effective,
        }
    }

    /// Effective byte length on disk or in archive.
    pub fn effective_bytes(&self) -> u64 {
        self.total_bytes.max(self.header_bytes.len() as u64)
    }
}

/// Detected spectrometer or interchange vendor format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ExperimentVendor {
    Bruker,
    Varian,
    NmrPipe,
    SparkyUcsf,
    SequenceFasta,
    StarOrNef,
}

impl ExperimentVendor {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bruker => "Bruker",
            Self::Varian => "Varian",
            Self::NmrPipe => "NmrPipe",
            Self::SparkyUcsf => "SparkyUcsf",
            Self::SequenceFasta => "SequenceFasta",
            Self::StarOrNef => "StarOrNef",
        }
    }
}

impl fmt::Display for ExperimentVendor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Experiment dimensionality, distinguishing true frequency-evolved 3D cubes from delay/counter `Pseudo3D` series.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ExperimentDimensionality {
    #[serde(rename = "1D")]
    Dim1D,
    #[serde(rename = "2D")]
    Dim2D,
    #[serde(rename = "3D")]
    Dim3D,
    #[serde(rename = "Pseudo3D")]
    Pseudo3D,
    #[serde(rename = "4D")]
    Dim4D,
}

impl ExperimentDimensionality {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Dim1D => "1D",
            Self::Dim2D => "2D",
            Self::Dim3D => "3D",
            Self::Pseudo3D => "Pseudo3D",
            Self::Dim4D => "4D",
        }
    }
}

impl fmt::Display for ExperimentDimensionality {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Time-domain vs frequency-domain availability for an experiment.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum DataState {
    RawFid,
    Processed,
    Both,
}

impl DataState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::RawFid => "RawFid",
            Self::Processed => "Processed",
            Self::Both => "Both",
        }
    }
}

impl fmt::Display for DataState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Quadrature detection mode per spectral axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum QuadratureKind {
    #[serde(rename = "DQD")]
    Dqd,
    #[serde(rename = "States")]
    States,
    #[serde(rename = "States-TPPI")]
    StatesTppi,
    #[serde(rename = "Echo-Antiecho")]
    EchoAntiecho,
    #[serde(rename = "QF")]
    Qf,
    #[serde(rename = "TPPI")]
    Tppi,
    #[serde(rename = "Processed")]
    Processed,
    #[serde(rename = "Unknown")]
    Unknown,
}

impl QuadratureKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Dqd => "DQD",
            Self::States => "States",
            Self::StatesTppi => "States-TPPI",
            Self::EchoAntiecho => "Echo-Antiecho",
            Self::Qf => "QF",
            Self::Tppi => "TPPI",
            Self::Processed => "Processed",
            Self::Unknown => "Unknown",
        }
    }
}

impl fmt::Display for QuadratureKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Non-uniform sampling (NUS) summary and sampling density.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NusSummary {
    pub is_nus: bool,
    pub sampled_points: Option<usize>,
    pub full_grid_points: Option<usize>,
    pub sampling_density: Option<f64>,
    pub badge: String,
}

impl Default for NusSummary {
    fn default() -> Self {
        Self {
            is_nus: false,
            sampled_points: None,
            full_grid_points: None,
            sampling_density: None,
            badge: "Uniform".to_string(),
        }
    }
}

/// Per-axis physical metadata extracted from acquisition or processed headers.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClassifiedAxis {
    /// Axis slot label (`F3`/`F2`/`F1`, `X`/`Y`/`Z`, etc.).
    pub label: String,
    /// Canonical isotope label (`1H`, `15N`, `13C`, `31P`, `2H`, `Delay`, etc.).
    pub nucleus: String,
    /// Time-domain or processed point count along this axis.
    pub points: usize,
    /// Spectral width in ppm when frequency-evolved.
    pub sw_ppm: Option<f64>,
    /// Spectral center / carrier position in ppm when available.
    pub carrier_ppm: Option<f64>,
    /// Channel observe frequency in MHz.
    pub obs_mhz: Option<f64>,
    /// Axis quadrature detection mode.
    pub quadrature: QuadratureKind,
}

/// Canonical NMR pulse-sequence classification across `1H`-detected, `13C`-detected,
/// methyl-TROSY, sidechain TOCSY/NOESY, relaxation, exchange, and titration experiments.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum PulseSequenceClass {
    N15Hsqc,
    C13Hsqc,
    C13Hmqc,
    Hnco,
    Hncaco,
    Hnca,
    Hncoca,
    Hncacb,
    Cbcaconh,
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
    TitrationSeries,
    Unknown,
}

impl PulseSequenceClass {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::N15Hsqc => "N15Hsqc",
            Self::C13Hsqc => "C13Hsqc",
            Self::C13Hmqc => "C13Hmqc",
            Self::Hnco => "Hnco",
            Self::Hncaco => "Hncaco",
            Self::Hnca => "Hnca",
            Self::Hncoca => "Hncoca",
            Self::Hncacb => "Hncacb",
            Self::Cbcaconh => "Cbcaconh",
            Self::Ccconh => "Ccconh",
            Self::Hccconh => "Hccconh",
            Self::HcchTocsy => "HcchTocsy",
            Self::N15Noesy => "N15Noesy",
            Self::C13Noesy => "C13Noesy",
            Self::Con2D => "Con2D",
            Self::Caco2D => "Caco2D",
            Self::Canco3D => "Canco3D",
            Self::Cbcaco3D => "Cbcaco3D",
            Self::RelaxR1 => "RelaxR1",
            Self::RelaxR2 => "RelaxR2",
            Self::HetNoe => "HetNoe",
            Self::Cpmg => "Cpmg",
            Self::Cest => "Cest",
            Self::TitrationSeries => "TitrationSeries",
            Self::Unknown => "Unknown",
        }
    }

    /// Canonical `nmrassign` solver experiment token when directly supported by the peak/assignment pipeline.
    pub fn canonical_solver_exp(self) -> Option<&'static str> {
        match self {
            Self::N15Hsqc => Some("HSQC"),
            Self::C13Hsqc | Self::C13Hmqc => Some("C13HSQC"),
            Self::Hnco => Some("HNCO"),
            Self::Hncaco => Some("HNCACO"),
            Self::Hnca => Some("HNCA"),
            Self::Hncoca => Some("HNCOCA"),
            Self::Hncacb => Some("HNCACB"),
            Self::Cbcaconh => Some("CBCACONH"),
            Self::Ccconh => Some("CCCONH"),
            Self::Hccconh => Some("HCCCONH"),
            Self::HcchTocsy => Some("HCCH_TOCSY"),
            Self::N15Noesy => Some("N15_NOESY"),
            Self::C13Noesy => Some("C13_NOESY"),
            _ => None,
        }
    }

    pub fn is_1h_backbone_triple(self) -> bool {
        matches!(
            self,
            Self::Hnco | Self::Hncaco | Self::Hnca | Self::Hncoca | Self::Hncacb | Self::Cbcaconh
        )
    }

    pub fn is_13c_detected(self) -> bool {
        matches!(
            self,
            Self::Con2D | Self::Caco2D | Self::Canco3D | Self::Cbcaco3D
        )
    }

    pub fn is_sidechain_or_noesy(self) -> bool {
        matches!(
            self,
            Self::Ccconh | Self::Hccconh | Self::HcchTocsy | Self::N15Noesy | Self::C13Noesy
        )
    }

    pub fn is_dynamics(self) -> bool {
        matches!(
            self,
            Self::RelaxR1 | Self::RelaxR2 | Self::HetNoe | Self::Cpmg | Self::Cest
        )
    }
}

impl fmt::Display for PulseSequenceClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Overall scientific problem regime inferred from the workspace inventory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ProblemRegime {
    GlobularBackbone,
    SidechainNoesy,
    Idp13cDetected,
    MethylTrosy,
    Paramagnetic,
    RelaxationDynamics,
    TitrationCsp,
}

impl ProblemRegime {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::GlobularBackbone => "GlobularBackbone",
            Self::SidechainNoesy => "SidechainNoesy",
            Self::Idp13cDetected => "Idp13cDetected",
            Self::MethylTrosy => "MethylTrosy",
            Self::Paramagnetic => "Paramagnetic",
            Self::RelaxationDynamics => "RelaxationDynamics",
            Self::TitrationCsp => "TitrationCsp",
        }
    }
}

impl fmt::Display for ProblemRegime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Classified NMR experiment or auxiliary sequence/shift table file.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClassifiedExperiment {
    /// Relative path from workspace or archive root.
    pub relative_path: String,
    /// Short display name (`CISD3_CBCACONH`, `HSQC.ucsf`, `v6/HNCA`, etc.).
    pub display_name: String,
    /// Vendor or interchange format.
    pub vendor: ExperimentVendor,
    /// Dimensionality (`1D`, `2D`, `3D`, `Pseudo3D`, `4D`).
    pub dimensionality: ExperimentDimensionality,
    /// Raw vs processed state.
    pub data_state: DataState,
    /// Nominal `1H` spectrometer base frequency in MHz (e.g. `600.13`, `800.13`, `900.21`).
    pub spectrometer_mhz: Option<f64>,
    /// Ordered spectral axes (direct dimension first).
    pub axes: Vec<ClassifiedAxis>,
    /// Compact indirect quadrature summary (`Echo-Antiecho`, `States-TPPI / Echo-Antiecho`, `QF`, `Processed`).
    pub quadrature_badge: String,
    /// Non-uniform sampling badge & sampling density.
    pub nus: NusSummary,
    /// Detected pulse-sequence class.
    pub pulse_class: PulseSequenceClass,
    /// Raw pulse program identifier (`PULPROG` or `seqfil`) when available.
    pub pulse_program: Option<String>,
    /// Experiment title/comment line when available.
    pub title: Option<String>,
    /// Byte length of time-domain FID (`ser`/`fid`) without reading its payload.
    pub raw_signal_bytes: u64,
    /// Byte length of processed spectral matrices (`2rr`, `3rrr`, `.ucsf`, `.ft2`, `.ft3`) without reading payloads.
    pub processed_bytes: u64,
    /// Whether this entry should be checked by default in the ingest matrix.
    pub selected_by_default: bool,
    /// Diagnostic notes (e.g., `Pseudo3D` delay list kind, paramagnetic sweep hint, unacquired shell).
    pub notes: Vec<String>,
}

/// Auxiliary FASTA sequence or NMR-STAR / NEF file discovered in the workspace.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuxiliaryFileRecord {
    pub relative_path: String,
    pub vendor: ExperimentVendor,
    pub total_bytes: u64,
    pub summary: String,
}

/// Complete classified inventory for a spectrometer folder, archive, or drag-and-drop bundle.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WorkspaceInventory {
    /// Root folder or archive display name.
    pub root_name: String,
    /// Discovered NMR spectral experiments sorted deterministically by `relative_path`.
    pub experiments: Vec<ClassifiedExperiment>,
    /// Discovered FASTA/sequence files and NMR-STAR / NEF files.
    pub auxiliary_files: Vec<AuxiliaryFileRecord>,
    /// Primary recommended scientific workflow regime.
    pub problem_regime: ProblemRegime,
    /// Additional regimes represented in multi-purpose workspaces (e.g. relaxation + 13C-detected sub-studies).
    pub secondary_regimes: Vec<ProblemRegime>,
    /// Actionable warnings for missing sequence, missing 2D root, or sparse experiment sets.
    pub missing_input_hints: Vec<String>,
    /// Classification wall-clock time in milliseconds (`0.0` on `wasm32`).
    pub elapsed_ms: f64,
}

impl WorkspaceInventory {
    /// Serialize this inventory as pretty-printed JSON.
    pub fn to_json_pretty(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| format!("failed to serialize WorkspaceInventory JSON: {e}"))
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn measure_ms<F, R>(f: F) -> (R, f64)
where
    F: FnOnce() -> R,
{
    let t0 = std::time::Instant::now();
    let res = f();
    (res, t0.elapsed().as_secs_f64() * 1000.0)
}

#[cfg(target_arch = "wasm32")]
fn measure_ms<F, R>(f: F) -> (R, f64)
where
    F: FnOnce() -> R,
{
    (f(), 0.0)
}

// ============================================================================
// Pure-Rust RFC 1951 DEFLATE + RFC 1952 GZIP Streaming Reader & POSIX Tar Walker
// ============================================================================

const FIXED_LEN_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
const FIXED_LEN_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
const FIXED_DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const FIXED_DIST_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];
const CLEN_ORDER: [usize; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

struct HuffTable {
    max_bits: u8,
    /// Packed `(len << 12) | sym`; `0` means invalid code.
    lut: Vec<u32>,
}

impl HuffTable {
    fn from_lengths(lengths: &[u8]) -> Result<Self, String> {
        let mut max_bits = 0u8;
        let mut bl_count = [0u16; 16];
        for &l in lengths {
            if l > 15 {
                return Err("invalid Huffman code length > 15".to_string());
            }
            if l > 0 {
                bl_count[l as usize] += 1;
                max_bits = max_bits.max(l);
            }
        }
        if max_bits == 0 {
            return Ok(Self {
                max_bits: 1,
                lut: vec![0; 2],
            });
        }
        let mut next_code = [0u16; 16];
        let mut code = 0u32;
        for bits in 1..=15 {
            code = (code + bl_count[bits - 1] as u32) << 1;
            next_code[bits] = code as u16;
        }
        let table_size = 1usize << (max_bits as usize);
        let mut lut = vec![0u32; table_size];
        for (sym, &len) in lengths.iter().enumerate() {
            if len == 0 {
                continue;
            }
            let c = next_code[len as usize];
            next_code[len as usize] = c.wrapping_add(1);
            // Reverse `len` bits for LSB-first DEFLATE bitstreams
            let mut rev = 0usize;
            for b in 0..(len as usize) {
                if (c >> b) & 1 != 0 {
                    rev |= 1 << ((len as usize) - 1 - b);
                }
            }
            let packed = ((len as u32) << 12) | (sym as u32);
            let step = 1usize << (len as usize);
            let mut idx = rev;
            while idx < table_size {
                lut[idx] = packed;
                idx += step;
            }
        }
        Ok(Self { max_bits, lut })
    }

    fn fixed_lit_len() -> Self {
        let mut lens = [0u8; 288];
        lens[0..=143].fill(8);
        lens[144..=255].fill(9);
        lens[256..=279].fill(7);
        lens[280..=287].fill(8);
        Self::from_lengths(&lens).expect("valid RFC 1951 fixed literal table")
    }

    fn fixed_dist() -> Self {
        let lens = [5u8; 32];
        Self::from_lengths(&lens).expect("valid RFC 1951 fixed distance table")
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum DeflatePhase {
    NeedGzipHeader,
    NeedBlockHeader,
    StoredBlock {
        remaining: usize,
    },
    CompressedBlock {
        final_block: bool,
    },
    CopyMatch {
        dist: usize,
        remaining: usize,
        final_block: bool,
    },
    NeedGzipTrailer,
    Done,
}

const MAX_DISK_INFLATE_BYTES: u64 = 16 * 1024 * 1024 * 1024; // 16 GiB bound for streamed/discarded archive payloads
const MAX_RETAINED_HEADER_BYTES: usize = 64 * 1024 * 1024; // 64 MiB bound on retained VirtualEntry.header_bytes memory

const fn build_crc32_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0usize;
    while i < 256 {
        let mut crc = i as u32;
        let mut bit = 0;
        while bit < 8 {
            if (crc & 1) != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
            bit += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
}

const CRC32_TABLE: [u32; 256] = build_crc32_table();

/// Streaming RFC 1952 GZIP + RFC 1951 DEFLATE reader using a fixed 32 KiB LZ77 ring buffer.
struct GzipStreamDecoder<R: Read> {
    inner: BufReader<R>,
    bit_buf: u64,
    bit_cnt: u8,
    pending_id1: Option<u8>,
    window: [u8; 32768],
    win_pos: usize,
    crc32_state: u32,
    member_out_bytes: u64,
    total_out_bytes: u64,
    max_inflate_bytes: u64,
    phase: DeflatePhase,
    final_block_pending: bool,
    lit_table: HuffTable,
    dist_table: HuffTable,
}

impl<R: Read> GzipStreamDecoder<R> {
    fn new(reader: R, max_inflate_bytes: u64) -> Self {
        Self {
            inner: BufReader::with_capacity(32 * 1024, reader),
            bit_buf: 0,
            bit_cnt: 0,
            pending_id1: None,
            window: [0u8; 32768],
            win_pos: 0,
            crc32_state: 0xFFFF_FFFF,
            member_out_bytes: 0,
            total_out_bytes: 0,
            max_inflate_bytes,
            phase: DeflatePhase::NeedGzipHeader,
            final_block_pending: false,
            lit_table: HuffTable::fixed_lit_len(),
            dist_table: HuffTable::fixed_dist(),
        }
    }

    #[inline]
    fn push_out_byte(&mut self, out: &mut [u8], written: &mut usize, b: u8) -> io::Result<()> {
        if self.total_out_bytes >= self.max_inflate_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "compressed archive exceeds maximum uncompressed byte limit",
            ));
        }
        self.window[self.win_pos] = b;
        self.win_pos = (self.win_pos + 1) & 0x7FFF;
        let idx = ((self.crc32_state ^ (b as u32)) & 0xFF) as usize;
        self.crc32_state = (self.crc32_state >> 8) ^ CRC32_TABLE[idx];
        self.member_out_bytes += 1;
        self.total_out_bytes += 1;
        out[*written] = b;
        *written += 1;
        Ok(())
    }

    fn next_raw_byte(&mut self) -> io::Result<Option<u8>> {
        if self.bit_cnt >= 8 {
            let b = (self.bit_buf & 0xFF) as u8;
            self.bit_buf >>= 8;
            self.bit_cnt -= 8;
            return Ok(Some(b));
        }
        self.bit_buf = 0;
        self.bit_cnt = 0;
        let mut buf = [0u8; 1];
        let n = self.inner.read(&mut buf)?;
        if n == 0 {
            Ok(None)
        } else {
            Ok(Some(buf[0]))
        }
    }

    fn require_raw_byte(&mut self) -> io::Result<u8> {
        self.next_raw_byte()?
            .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "truncated gzip stream"))
    }

    fn align_byte(&mut self) {
        let rem = self.bit_cnt & 7;
        if rem > 0 {
            self.bit_buf >>= rem;
            self.bit_cnt -= rem;
        }
    }

    fn ensure_bits(&mut self, n: u8) -> io::Result<()> {
        while self.bit_cnt < n {
            let mut b = [0u8; 1];
            let got = self.inner.read(&mut b)?;
            if got == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "unexpected EOF in DEFLATE bitstream",
                ));
            }
            self.bit_buf |= (b[0] as u64) << self.bit_cnt;
            self.bit_cnt += 8;
        }
        Ok(())
    }

    fn read_bits(&mut self, n: u8) -> io::Result<u32> {
        if n == 0 {
            return Ok(0);
        }
        self.ensure_bits(n)?;
        let mask = (1u64 << n) - 1;
        let val = (self.bit_buf & mask) as u32;
        self.bit_buf >>= n;
        self.bit_cnt -= n;
        Ok(val)
    }

    fn decode_sym_lit(&mut self) -> io::Result<u16> {
        let max_b = self.lit_table.max_bits;
        while self.bit_cnt < max_b {
            let mut b = [0u8; 1];
            if self.inner.read(&mut b)? == 0 {
                break;
            }
            self.bit_buf |= (b[0] as u64) << self.bit_cnt;
            self.bit_cnt += 8;
        }
        let mask = (1u64 << max_b) - 1;
        let idx = (self.bit_buf & mask) as usize;
        let entry = self.lit_table.lut[idx];
        let len = (entry >> 12) as u8;
        if len == 0 || len > self.bit_cnt {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid literal/length Huffman symbol",
            ));
        }
        self.bit_buf >>= len;
        self.bit_cnt -= len;
        Ok((entry & 0x0FFF) as u16)
    }

    fn decode_sym_dist(&mut self) -> io::Result<u16> {
        let max_b = self.dist_table.max_bits;
        while self.bit_cnt < max_b {
            let mut b = [0u8; 1];
            if self.inner.read(&mut b)? == 0 {
                break;
            }
            self.bit_buf |= (b[0] as u64) << self.bit_cnt;
            self.bit_cnt += 8;
        }
        let mask = (1u64 << max_b) - 1;
        let idx = (self.bit_buf & mask) as usize;
        let entry = self.dist_table.lut[idx];
        let len = (entry >> 12) as u8;
        if len == 0 || len > self.bit_cnt {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid distance Huffman symbol",
            ));
        }
        self.bit_buf >>= len;
        self.bit_cnt -= len;
        Ok((entry & 0x0FFF) as u16)
    }

    fn parse_gzip_header(&mut self) -> io::Result<bool> {
        self.align_byte();
        let next_id1 = match self.pending_id1.take() {
            Some(b) => Some(b),
            None => self.next_raw_byte()?,
        };
        let Some(id1) = next_id1 else {
            return Ok(false);
        };
        let id2 = self.require_raw_byte()?;
        if id1 != 0x1f || id2 != 0x8b {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid GZIP magic bytes (expected 0x1f 0x8b)",
            ));
        }
        let cm = self.require_raw_byte()?;
        if cm != 8 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unsupported GZIP compression method (expected DEFLATE=8)",
            ));
        }
        let flg = self.require_raw_byte()?;
        // Skip MTIME (4) + XFL (1) + OS (1)
        for _ in 0..6 {
            let _ = self.require_raw_byte()?;
        }
        if flg & 0x04 != 0 {
            let xlen_lo = self.require_raw_byte()? as usize;
            let xlen_hi = self.require_raw_byte()? as usize;
            let xlen = xlen_lo | (xlen_hi << 8);
            for _ in 0..xlen {
                let _ = self.require_raw_byte()?;
            }
        }
        if flg & 0x08 != 0 {
            while self.require_raw_byte()? != 0 {}
        }
        if flg & 0x10 != 0 {
            while self.require_raw_byte()? != 0 {}
        }
        if flg & 0x02 != 0 {
            let _ = self.require_raw_byte()?;
            let _ = self.require_raw_byte()?;
        }
        Ok(true)
    }

    fn load_dynamic_tables(&mut self) -> io::Result<()> {
        let hlit = (self.read_bits(5)? as usize) + 257;
        let hdist = (self.read_bits(5)? as usize) + 1;
        let hclen = (self.read_bits(4)? as usize) + 4;

        let mut clen_lens = [0u8; 19];
        for i in 0..hclen {
            clen_lens[CLEN_ORDER[i]] = self.read_bits(3)? as u8;
        }
        let clen_table = HuffTable::from_lengths(&clen_lens)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let total_codes = hlit + hdist;
        let mut combined = Vec::with_capacity(total_codes);
        while combined.len() < total_codes {
            let max_b = clen_table.max_bits;
            while self.bit_cnt < max_b {
                let mut b = [0u8; 1];
                if self.inner.read(&mut b)? == 0 {
                    break;
                }
                self.bit_buf |= (b[0] as u64) << self.bit_cnt;
                self.bit_cnt += 8;
            }
            let idx = (self.bit_buf & ((1u64 << max_b) - 1)) as usize;
            let entry = clen_table.lut[idx];
            let bl = (entry >> 12) as u8;
            if bl == 0 || bl > self.bit_cnt {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "invalid dynamic code-length symbol",
                ));
            }
            self.bit_buf >>= bl;
            self.bit_cnt -= bl;
            let sym = (entry & 0x0FFF) as u8;
            match sym {
                0..=15 => combined.push(sym),
                16 => {
                    let prev = *combined.last().ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidData, "repeat code 16 at start")
                    })?;
                    let rep = (self.read_bits(2)? as usize) + 3;
                    for _ in 0..rep {
                        if combined.len() < total_codes {
                            combined.push(prev);
                        }
                    }
                }
                17 => {
                    let rep = (self.read_bits(3)? as usize) + 3;
                    for _ in 0..rep {
                        if combined.len() < total_codes {
                            combined.push(0);
                        }
                    }
                }
                18 => {
                    let rep = (self.read_bits(7)? as usize) + 11;
                    for _ in 0..rep {
                        if combined.len() < total_codes {
                            combined.push(0);
                        }
                    }
                }
                _ => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "unknown code-length symbol",
                    ))
                }
            }
        }

        self.lit_table = HuffTable::from_lengths(&combined[..hlit])
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        self.dist_table = HuffTable::from_lengths(&combined[hlit..])
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        Ok(())
    }
}

impl<R: Read> Read for GzipStreamDecoder<R> {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        let mut written = 0usize;
        while written < out.len() {
            match self.phase {
                DeflatePhase::Done => break,
                DeflatePhase::NeedGzipHeader => {
                    if self.parse_gzip_header()? {
                        self.crc32_state = 0xFFFF_FFFF;
                        self.member_out_bytes = 0;
                        self.phase = DeflatePhase::NeedBlockHeader;
                    } else {
                        self.phase = DeflatePhase::Done;
                    }
                }
                DeflatePhase::NeedGzipTrailer => {
                    self.align_byte();
                    let mut tr = [0u8; 8];
                    for slot in &mut tr {
                        *slot = self.require_raw_byte()?;
                    }
                    let expected_crc = u32::from_le_bytes([tr[0], tr[1], tr[2], tr[3]]);
                    let expected_isize = u32::from_le_bytes([tr[4], tr[5], tr[6], tr[7]]);
                    let actual_crc = !self.crc32_state;
                    let actual_isize = self.member_out_bytes as u32;
                    if actual_crc != expected_crc || actual_isize != expected_isize {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "gzip trailer CRC32 or ISIZE mismatch",
                        ));
                    }
                    // Peek at remaining bytes after the 8-byte trailer:
                    // - EOF or all-zero trailing padding -> finish cleanly (`Done`).
                    // - Non-zero byte -> concatenated next gzip member (`NeedGzipHeader`).
                    match self.next_raw_byte()? {
                        None => {
                            self.phase = DeflatePhase::Done;
                        }
                        Some(0x00) => {
                            let mut pad_bytes: u64 = 1;
                            if self.total_out_bytes.saturating_add(pad_bytes)
                                > self.max_inflate_bytes
                            {
                                return Err(io::Error::new(
                                    io::ErrorKind::InvalidData,
                                    "compressed archive exceeds maximum uncompressed byte limit",
                                ));
                            }
                            while self.bit_cnt >= 8 {
                                let b = (self.bit_buf & 0xFF) as u8;
                                self.bit_buf >>= 8;
                                self.bit_cnt -= 8;
                                if b != 0 {
                                    return Err(io::Error::new(
                                        io::ErrorKind::InvalidData,
                                        "invalid GZIP magic bytes (expected 0x1f 0x8b)",
                                    ));
                                }
                                pad_bytes += 1;
                                if self.total_out_bytes.saturating_add(pad_bytes)
                                    > self.max_inflate_bytes
                                {
                                    return Err(io::Error::new(
                                        io::ErrorKind::InvalidData,
                                        "compressed archive exceeds maximum uncompressed byte limit",
                                    ));
                                }
                            }
                            self.bit_buf = 0;
                            self.bit_cnt = 0;
                            loop {
                                let chunk = std::io::BufRead::fill_buf(&mut self.inner)?;
                                if chunk.is_empty() {
                                    break;
                                }
                                if chunk.iter().any(|&b| b != 0) {
                                    return Err(io::Error::new(
                                        io::ErrorKind::InvalidData,
                                        "invalid GZIP magic bytes (expected 0x1f 0x8b)",
                                    ));
                                }
                                let n = chunk.len();
                                pad_bytes = pad_bytes.saturating_add(n as u64);
                                if self.total_out_bytes.saturating_add(pad_bytes)
                                    > self.max_inflate_bytes
                                {
                                    return Err(io::Error::new(
                                        io::ErrorKind::InvalidData,
                                        "compressed archive exceeds maximum uncompressed byte limit",
                                    ));
                                }
                                std::io::BufRead::consume(&mut self.inner, n);
                            }
                            self.phase = DeflatePhase::Done;
                        }
                        Some(b) => {
                            self.pending_id1 = Some(b);
                            self.phase = DeflatePhase::NeedGzipHeader;
                        }
                    }
                }
                DeflatePhase::NeedBlockHeader => {
                    let bfinal = self.read_bits(1)? == 1;
                    let btype = self.read_bits(2)?;
                    self.final_block_pending = bfinal;
                    match btype {
                        0 => {
                            self.align_byte();
                            let lo = self.require_raw_byte()? as u16;
                            let hi = self.require_raw_byte()? as u16;
                            let len = lo | (hi << 8);
                            let nlo = self.require_raw_byte()? as u16;
                            let nhi = self.require_raw_byte()? as u16;
                            let nlen = nlo | (nhi << 8);
                            if len != !nlen {
                                return Err(io::Error::new(
                                    io::ErrorKind::InvalidData,
                                    "stored DEFLATE block LEN/NLEN mismatch",
                                ));
                            }
                            self.phase = DeflatePhase::StoredBlock {
                                remaining: len as usize,
                            };
                        }
                        1 => {
                            self.lit_table = HuffTable::fixed_lit_len();
                            self.dist_table = HuffTable::fixed_dist();
                            self.phase = DeflatePhase::CompressedBlock {
                                final_block: bfinal,
                            };
                        }
                        2 => {
                            self.load_dynamic_tables()?;
                            self.phase = DeflatePhase::CompressedBlock {
                                final_block: bfinal,
                            };
                        }
                        _ => {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                "reserved DEFLATE BTYPE=3",
                            ))
                        }
                    }
                }
                DeflatePhase::StoredBlock { mut remaining } => {
                    while remaining > 0 && written < out.len() {
                        let b = self.require_raw_byte()?;
                        self.push_out_byte(out, &mut written, b)?;
                        remaining -= 1;
                    }
                    if remaining == 0 {
                        self.phase = if self.final_block_pending {
                            DeflatePhase::NeedGzipTrailer
                        } else {
                            DeflatePhase::NeedBlockHeader
                        };
                    } else {
                        self.phase = DeflatePhase::StoredBlock { remaining };
                    }
                }
                DeflatePhase::CopyMatch {
                    dist,
                    mut remaining,
                    final_block,
                } => {
                    while remaining > 0 && written < out.len() {
                        let src = (self.win_pos.wrapping_sub(dist)) & 0x7FFF;
                        let b = self.window[src];
                        self.push_out_byte(out, &mut written, b)?;
                        remaining -= 1;
                    }
                    if remaining == 0 {
                        self.phase = DeflatePhase::CompressedBlock { final_block };
                    } else {
                        self.phase = DeflatePhase::CopyMatch {
                            dist,
                            remaining,
                            final_block,
                        };
                    }
                }
                DeflatePhase::CompressedBlock { final_block } => {
                    let sym = self.decode_sym_lit()?;
                    if sym < 256 {
                        self.push_out_byte(out, &mut written, sym as u8)?;
                    } else if sym == 256 {
                        self.phase = if final_block {
                            DeflatePhase::NeedGzipTrailer
                        } else {
                            DeflatePhase::NeedBlockHeader
                        };
                    } else {
                        let lidx = (sym as usize).saturating_sub(257);
                        if lidx >= FIXED_LEN_BASE.len() {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                "out-of-range DEFLATE length symbol",
                            ));
                        }
                        let extra_l = self.read_bits(FIXED_LEN_EXTRA[lidx])? as usize;
                        let match_len = (FIXED_LEN_BASE[lidx] as usize) + extra_l;
                        let dsym = self.decode_sym_dist()? as usize;
                        if dsym >= FIXED_DIST_BASE.len() {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                "out-of-range DEFLATE distance symbol",
                            ));
                        }
                        let extra_d = self.read_bits(FIXED_DIST_EXTRA[dsym])? as usize;
                        let dist = (FIXED_DIST_BASE[dsym] as usize) + extra_d;
                        if dist == 0 || (dist as u64) > self.member_out_bytes {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                "DEFLATE match distance exceeds output history",
                            ));
                        }
                        self.phase = DeflatePhase::CopyMatch {
                            dist,
                            remaining: match_len,
                            final_block,
                        };
                    }
                }
            }
        }
        Ok(written)
    }
}

/// File-reading strategy determined strictly from relative path BEFORE touching file bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FileReadAction {
    /// Bulk time-domain or frequency-domain signal payload: record `total_bytes` ONLY, read `0` bytes!
    MetadataOnlySignal,
    /// Binary UCSF v2 spectrum: read at most `MAX_UCSF_HEADER_BYTES` (`1024 B`).
    BoundedBinary(usize),
    /// Text acquisition/processing/schedule/sequence/STAR file: read at most `limit` bytes.
    BoundedText(usize),
    /// Unrelated binary/auxiliary file (`spnam*`, `shimvalues`, `.pdf`, `.ps`, `.png`, etc.): skip.
    Ignore,
}

fn classify_file_read_action(rel_path: &str) -> FileReadAction {
    let norm = normalize_rel_path(rel_path);
    if norm.is_empty() || is_noise_path(&norm) {
        return FileReadAction::Ignore;
    }
    let fname = norm.rsplit('/').next().unwrap_or(&norm);
    let lower = fname.to_ascii_lowercase();

    // 1. Bulk binary signal payloads: NEVER read payload bytes!
    if matches!(
        lower.as_str(),
        "ser"
            | "fid"
            | "origfid"
            | "fidc.dat"
            | "1r"
            | "1i"
            | "2rr"
            | "2ri"
            | "2ir"
            | "2ii"
            | "3rrr"
            | "3rri"
            | "3rir"
            | "3iir"
            | "4rrrr"
    ) {
        return FileReadAction::MetadataOnlySignal;
    }

    // 2. Bruker JCAMP-DX parameter, pulse-program, title, and pseudo-3D / NUS schedule files
    if matches!(
        lower.as_str(),
        "acqus"
            | "acqu"
            | "acqu2s"
            | "acqu2"
            | "acqu3s"
            | "acqu3"
            | "acqu4s"
            | "acqu4"
            | "procs"
            | "proc"
            | "proc2s"
            | "proc2"
            | "proc3s"
            | "proc3"
            | "proc4s"
            | "proc4"
            | "title"
            | "pulseprogram"
            | "pulseprogram.precomp"
            | "pulprog"
            | "nuslist"
            | "difflist"
            | "vdlist"
            | "vclist"
            | "valist"
            | "vplist"
            | "fqlist"
            | "fq1list"
            | "fq2list"
            | "fq3list"
            | "fq4list"
            | "fq5list"
            | "fq6list"
            | "fq7list"
            | "fq8list"
            | "procpar"
            | "text"
            | "sampling.sch"
    ) {
        return FileReadAction::BoundedText(MAX_TEXT_HEADER_BYTES);
    }

    if lower.ends_with(".incl") {
        return FileReadAction::BoundedText(MAX_TEXT_HEADER_BYTES);
    }

    if lower.ends_with(".sch") && lower.contains("nus") {
        return FileReadAction::BoundedText(MAX_TEXT_HEADER_BYTES);
    }

    // 3. UCSF Sparky v2 processed spectrum
    if lower.ends_with(".ucsf") {
        return FileReadAction::BoundedBinary(MAX_UCSF_HEADER_BYTES);
    }

    // 4. NMRPipe processed spectra (`.ft1`, `.ft2`, `.ft3`, `.ft4`, `.ft`)
    if lower.ends_with(".ft1")
        || lower.ends_with(".ft2")
        || lower.ends_with(".ft3")
        || lower.ends_with(".ft4")
        || lower.ends_with(".ft")
    {
        return FileReadAction::BoundedBinary(NMRPIPE_HEADER_BYTES);
    }

    // 5. Polymer sequence (`.fasta`, `.fa`, `.seq`) & shift/session interchange (`.str`, `.star`, `.bmrb`, `.nef`)
    if lower.ends_with(".fasta") || lower.ends_with(".faa") || lower.ends_with(".seq") {
        return FileReadAction::BoundedText(MAX_TABLE_SNIFF_BYTES);
    }
    if lower.ends_with(".str")
        || lower.ends_with(".star")
        || lower.ends_with(".bmrb")
        || lower.ends_with(".nef")
    {
        return FileReadAction::BoundedText(MAX_TABLE_SNIFF_BYTES);
    }

    FileReadAction::Ignore
}

fn normalize_rel_path(raw: &str) -> String {
    let replaced = raw.replace('\\', "/");
    let mut parts = Vec::new();
    for seg in replaced.split('/') {
        let trimmed = seg.trim();
        if trimmed.is_empty() || trimmed == "." {
            continue;
        }
        if trimmed == ".." {
            parts.pop();
            continue;
        }
        parts.push(trimmed);
    }
    parts.join("/")
}

fn is_noise_path(norm: &str) -> bool {
    for seg in norm.split('/') {
        if seg == "__MACOSX" || seg == ".git" || seg == ".DS_Store" || seg.starts_with("._") {
            return true;
        }
    }
    false
}

fn is_tar_path_lower(lower: &str) -> bool {
    lower.ends_with(".tar")
}

fn is_targz_path_lower(lower: &str) -> bool {
    lower.ends_with(".tar.gz") || lower.ends_with(".tgz")
}

fn parse_tar_size(field: &[u8]) -> u64 {
    if field.is_empty() {
        return 0;
    }
    // GNU base-256 big-endian binary size encoding
    if field[0] & 0x80 != 0 {
        let mut acc = 0u64;
        for &b in &field[1..] {
            acc = (acc << 8) | (b as u64);
        }
        return acc;
    }
    let s = std::str::from_utf8(field).unwrap_or("");
    let clean = s.trim_matches(|c: char| c == '\0' || c.is_ascii_whitespace());
    u64::from_str_radix(clean, 8).unwrap_or(0)
}

fn parse_tar_cstr(field: &[u8]) -> String {
    let end = field.iter().position(|&b| b == 0).unwrap_or(field.len());
    String::from_utf8_lossy(&field[..end]).trim().to_string()
}

fn read_exact_or_eof<R: Read>(r: &mut R, buf: &mut [u8]) -> io::Result<bool> {
    let mut off = 0usize;
    while off < buf.len() {
        let n = r.read(&mut buf[off..])?;
        if n == 0 {
            if off == 0 {
                return Ok(false);
            }
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "truncated tar block",
            ));
        }
        off += n;
    }
    Ok(true)
}

fn discard_bytes_read<R: Read>(r: &mut R, mut count: u64) -> io::Result<()> {
    let mut scratch = [0u8; 8192];
    while count > 0 {
        let chunk = (count.min(scratch.len() as u64)) as usize;
        let n = r.read(&mut scratch[..chunk])?;
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "truncated tar entry payload",
            ));
        }
        count -= n as u64;
    }
    Ok(())
}

fn validate_tar_header_checksum(hdr: &[u8; 512]) -> bool {
    let chk_field = &hdr[148..156];
    let s = std::str::from_utf8(chk_field).unwrap_or("");
    let clean = s.trim_matches(|c: char| c == '\0' || c.is_ascii_whitespace());
    let Ok(stored) = i64::from_str_radix(clean, 8) else {
        return false;
    };
    let mut unsigned_sum = 0i64;
    let mut signed_sum = 0i64;
    for (idx, &b) in hdr.iter().enumerate() {
        if (148..156).contains(&idx) {
            unsigned_sum += 0x20;
            signed_sum += 0x20;
        } else {
            unsigned_sum += b as i64;
            signed_sum += (b as i8) as i64;
        }
    }
    stored == unsigned_sum || stored == signed_sum
}

/// Walk a POSIX.1-1988 `ustar` stream using `skip_fn` to advance over unread payload tails
/// (`Seek::seek` on `.tar` files/cursors in `O(1)`, or bounded scratch discard on `.tar.gz`),
/// while enforcing `max_retained_header_bytes` across all retained `VirtualEntry.header_bytes`.
fn scan_tar_with_skipper<R, S>(
    reader: &mut R,
    mut skip_fn: S,
    max_retained_header_bytes: usize,
    out: &mut Vec<VirtualEntry>,
) -> Result<(), String>
where
    R: Read,
    S: FnMut(&mut R, u64) -> io::Result<()>,
{
    let mut hdr = [0u8; 512];
    let mut zero_blocks = 0u8;
    let mut gnu_long_name: Option<String> = None;
    let mut pax_path: Option<String> = None;
    let mut pax_size: Option<u64> = None;
    let mut retained_header_bytes: usize = out.iter().map(|e| e.header_bytes.len()).sum();

    loop {
        let has_block =
            read_exact_or_eof(reader, &mut hdr).map_err(|e| format!("tar read error: {e}"))?;
        if !has_block {
            break;
        }
        if hdr.iter().all(|&b| b == 0) {
            zero_blocks += 1;
            if zero_blocks >= 2 {
                break;
            }
            continue;
        }
        zero_blocks = 0;

        if !validate_tar_header_checksum(&hdr) {
            return Err("invalid POSIX tar header checksum".to_string());
        }

        let typeflag = hdr[156];
        let raw_size = pax_size
            .take()
            .unwrap_or_else(|| parse_tar_size(&hdr[124..136]));
        let padded_size = raw_size
            .checked_add(511)
            .ok_or_else(|| "tar entry size overflow".to_string())?
            & !511;

        if typeflag == b'L' {
            // GNU LongLink path override for next entry
            let cap = (raw_size.min(8192)) as usize;
            let mut name_buf = vec![0u8; cap];
            if cap > 0 {
                read_exact_or_eof(reader, &mut name_buf)
                    .map_err(|e| format!("tar GNU LongLink read error: {e}"))?;
            }
            let rem = padded_size.saturating_sub(cap as u64);
            if rem > 0 {
                skip_fn(reader, rem).map_err(|e| format!("tar skip error: {e}"))?;
            }
            gnu_long_name = Some(parse_tar_cstr(&name_buf));
            continue;
        }

        if typeflag == b'x' || typeflag == b'g' {
            // POSIX PAX extended header
            let cap = (raw_size.min(16384)) as usize;
            let mut pax_buf = vec![0u8; cap];
            if cap > 0 {
                read_exact_or_eof(reader, &mut pax_buf)
                    .map_err(|e| format!("tar PAX header read error: {e}"))?;
            }
            let rem = padded_size.saturating_sub(cap as u64);
            if rem > 0 {
                skip_fn(reader, rem).map_err(|e| format!("tar skip error: {e}"))?;
            }
            if typeflag == b'x' {
                if let Ok(txt) = std::str::from_utf8(&pax_buf) {
                    for line in txt.lines() {
                        let kv = line.split_once(' ').map(|(_, rest)| rest).unwrap_or(line);
                        if let Some(p) = kv.strip_prefix("path=") {
                            pax_path = Some(p.trim().to_string());
                        } else if let Some(sz) = kv.strip_prefix("size=") {
                            pax_size = sz.trim().parse::<u64>().ok();
                        }
                    }
                }
            }
            continue;
        }

        let name = if let Some(p) = pax_path.take() {
            gnu_long_name = None;
            p
        } else if let Some(g) = gnu_long_name.take() {
            g
        } else {
            let base = parse_tar_cstr(&hdr[0..100]);
            // Strictly require POSIX.1-1988 `ustar\0` before reading `hdr[345..500]` as prefix,
            // since GNU tar (`ustar  \0`) stores binary `atime`/`ctime` at offsets 345..369.
            let is_posix_ustar = &hdr[257..263] == b"ustar\0";
            let prefix = if is_posix_ustar {
                parse_tar_cstr(&hdr[345..500])
            } else {
                String::new()
            };
            if prefix.is_empty() {
                base
            } else {
                format!("{prefix}/{base}")
            }
        };

        let is_regular = typeflag == 0 || typeflag == b'0' || typeflag == b'7';
        if !is_regular {
            if padded_size > 0 {
                skip_fn(reader, padded_size).map_err(|e| format!("tar skip error: {e}"))?;
            }
            continue;
        }

        let rel = normalize_rel_path(&name);
        match classify_file_read_action(&rel) {
            FileReadAction::Ignore => {
                if padded_size > 0 {
                    skip_fn(reader, padded_size).map_err(|e| format!("tar skip error: {e}"))?;
                }
            }
            FileReadAction::MetadataOnlySignal => {
                out.push(VirtualEntry::header_only(rel, Vec::new(), raw_size));
                if padded_size > 0 {
                    skip_fn(reader, padded_size).map_err(|e| format!("tar skip error: {e}"))?;
                }
            }
            FileReadAction::BoundedBinary(limit) | FileReadAction::BoundedText(limit) => {
                let to_read = (raw_size.min(limit as u64)) as usize;
                let next_retained = retained_header_bytes.saturating_add(to_read);
                if next_retained > max_retained_header_bytes {
                    return Err(format!(
                        "archive exceeded maximum retained header byte limit ({max_retained_header_bytes} bytes)"
                    ));
                }
                let mut prefix_bytes = vec![0u8; to_read];
                if to_read > 0 {
                    read_exact_or_eof(reader, &mut prefix_bytes)
                        .map_err(|e| format!("tar header entry read error: {e}"))?;
                }
                let rem = padded_size.saturating_sub(to_read as u64);
                if rem > 0 {
                    skip_fn(reader, rem).map_err(|e| format!("tar skip error: {e}"))?;
                }
                retained_header_bytes = next_retained;
                out.push(VirtualEntry::header_only(rel, prefix_bytes, raw_size));
            }
        }
    }
    Ok(())
}

fn scan_tar_seekable_bounded<R: Read + Seek>(
    reader: &mut R,
    max_retained_header_bytes: usize,
    out: &mut Vec<VirtualEntry>,
) -> Result<(), String> {
    scan_tar_with_skipper(
        reader,
        |r, n| {
            let step = i64::try_from(n)
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "tar entry > i64::MAX"))?;
            r.seek(SeekFrom::Current(step)).map(|_| ())
        },
        max_retained_header_bytes,
        out,
    )
}

fn scan_tar_seekable<R: Read + Seek>(
    reader: &mut R,
    out: &mut Vec<VirtualEntry>,
) -> Result<(), String> {
    scan_tar_seekable_bounded(reader, MAX_RETAINED_HEADER_BYTES, out)
}

fn scan_targz_reader_bounded<R: Read>(
    reader: R,
    max_inflate_bytes: u64,
    max_retained_header_bytes: usize,
    out: &mut Vec<VirtualEntry>,
) -> Result<(), String> {
    let mut gz = GzipStreamDecoder::new(reader, max_inflate_bytes);
    scan_tar_with_skipper(&mut gz, discard_bytes_read, max_retained_header_bytes, out)?;
    let mut scratch = [0u8; 4096];
    while gz
        .read(&mut scratch)
        .map_err(|e| format!("gzip trailer verification error: {e}"))?
        > 0
    {}
    Ok(())
}

fn scan_targz_reader<R: Read>(
    reader: R,
    max_inflate_bytes: u64,
    out: &mut Vec<VirtualEntry>,
) -> Result<(), String> {
    scan_targz_reader_bounded(reader, max_inflate_bytes, MAX_RETAINED_HEADER_BYTES, out)
}

/// Inspect an in-memory `.tar` or `.tar.gz` / `.tgz` byte buffer into `VirtualEntry` headers
/// with explicit upper bounds on total decompressed stream bytes (`max_inflate_bytes`) and
/// total retained header memory (`max_retained_header_bytes`).
pub fn extract_archive_virtual_entries_bounded(
    archive_name: &str,
    archive_bytes: &[u8],
    max_inflate_bytes: u64,
    max_retained_header_bytes: usize,
) -> Result<Vec<VirtualEntry>, String> {
    let lower = archive_name.to_ascii_lowercase();
    let is_gz = is_targz_path_lower(&lower)
        || (archive_bytes.len() >= 2 && archive_bytes[0] == 0x1f && archive_bytes[1] == 0x8b);
    let mut entries = Vec::new();
    if is_gz {
        scan_targz_reader_bounded(
            Cursor::new(archive_bytes),
            max_inflate_bytes,
            max_retained_header_bytes,
            &mut entries,
        )?;
    } else {
        if (archive_bytes.len() as u64) > max_inflate_bytes {
            return Err(format!(
                "archive exceeded maximum uncompressed byte limit ({max_inflate_bytes} bytes)"
            ));
        }
        let mut cur = Cursor::new(archive_bytes);
        scan_tar_seekable_bounded(&mut cur, max_retained_header_bytes, &mut entries)?;
    }
    Ok(entries)
}

/// Inspect an in-memory `.tar` or `.tar.gz` / `.tgz` byte buffer into `VirtualEntry` headers
/// without retaining bulk binary signal payloads (`ser`/`fid` payloads are streamed and discarded
/// up to `16 GiB` while retained `VirtualEntry.header_bytes` memory is bounded to `64 MiB`).
pub fn extract_archive_virtual_entries(
    archive_name: &str,
    archive_bytes: &[u8],
) -> Result<Vec<VirtualEntry>, String> {
    extract_archive_virtual_entries_bounded(
        archive_name,
        archive_bytes,
        MAX_DISK_INFLATE_BYTES,
        MAX_RETAINED_HEADER_BYTES,
    )
}

fn read_prefix_from_disk(path: &Path, limit: usize) -> io::Result<Vec<u8>> {
    let mut f = File::open(path)?;
    let mut first = [0u8; 16384];
    let cap0 = first.len().min(limit);
    let n0 = f.read(&mut first[..cap0])?;
    if n0 < cap0 || limit <= first.len() {
        return Ok(first[..n0].to_vec());
    }
    let mut out = Vec::with_capacity(32768.min(limit));
    out.extend_from_slice(&first[..n0]);
    while out.len() < limit {
        let want = (limit - out.len()).min(first.len());
        let n = f.read(&mut first[..want])?;
        if n == 0 {
            break;
        }
        out.extend_from_slice(&first[..n]);
    }
    Ok(out)
}

fn scan_pdata_procno_dir(
    root: &Path,
    pdata_dir: &Path,
    has_acqus_parent: bool,
    entries: &mut Vec<VirtualEntry>,
) {
    // Ultra-fast path when `acqus` already exists in the parent Bruker experiment folder:
    // probe `pdata/1/{3rrr,2rr,1r,4rrrr}` directly via `fs::metadata` without `opendir`/`readdir`!
    let proc1 = pdata_dir.join("1");
    if has_acqus_parent {
        for bin_name in ["3rrr", "2rr", "1r", "4rrrr"] {
            let bin_path = proc1.join(bin_name);
            if let Ok(meta) = fs::metadata(&bin_path) {
                if meta.is_file() {
                    if let Ok(rel_os) = bin_path.strip_prefix(root) {
                        let rel = normalize_rel_path(&rel_os.to_string_lossy());
                        entries.push(VirtualEntry::header_only(rel, Vec::new(), meta.len()));
                    }
                    return;
                }
            }
        }
        return;
    }

    let chosen_proc = if proc1.is_dir() {
        Some(proc1)
    } else {
        fs::read_dir(pdata_dir).ok().and_then(|rd| {
            let mut procnos: Vec<(usize, std::path::PathBuf)> = rd
                .flatten()
                .filter_map(|e| {
                    let s = e.file_name().into_string().ok()?;
                    let num = s.parse::<usize>().ok()?;
                    e.file_type().ok()?.is_dir().then_some((num, e.path()))
                })
                .collect();
            procnos.sort_by_key(|(n, _)| *n);
            procnos.into_iter().next().map(|(_, p)| p)
        })
    };

    let Some(proc_dir) = chosen_proc else {
        return;
    };
    let Ok(rd) = fs::read_dir(&proc_dir) else {
        return;
    };
    for fe in rd.flatten() {
        let fname_os = fe.file_name();
        let Some(fname) = fname_os.to_str() else {
            continue;
        };
        let low = fname.to_ascii_lowercase();
        let is_proc_bin = matches!(low.as_str(), "1r" | "2rr" | "3rrr" | "4rrrr");
        let is_proc_text = matches!(
            low.as_str(),
            "procs"
                | "proc"
                | "proc2s"
                | "proc2"
                | "proc3s"
                | "proc3"
                | "proc4s"
                | "proc4"
                | "title"
        );
        if !is_proc_text && !is_proc_bin {
            continue;
        }
        let p = fe.path();
        let Ok(rel_os) = p.strip_prefix(root) else {
            continue;
        };
        let rel = normalize_rel_path(&rel_os.to_string_lossy());
        if is_proc_bin {
            let sz = fe.metadata().map(|m| m.len()).unwrap_or(0);
            entries.push(VirtualEntry::header_only(rel, Vec::new(), sz));
        } else if let Ok(prefix) = read_prefix_from_disk(&p, MAX_TEXT_HEADER_BYTES) {
            let sz = prefix.len() as u64;
            entries.push(VirtualEntry::header_only(rel, prefix, sz));
        }
    }
}

/// Fast recursive directory header walker (`< 50 ms` per protein bundle).
///
/// Key optimizations:
/// 1. Never opens or reads `ser`/`fid`/`2rr`/`3rrr` payloads (uses `DirEntry::metadata().len()`).
/// 2. Detects Bruker/Varian experiment roots from the already-read `read_dir` slice (zero duplicate
///    `read_dir` calls), inspects only `pdata/1` and `ft/` (single `stat` + plane count), and skips
///    hundreds of converted slice files in `fid/` or `ft/`.
/// 3. Defers child `.tar`/`.tar.gz`/`.tgz` archives and skips them when unpacked NMR experiments
///    already exist in the directory tree (avoiding redundant 300–400 MB archive decompression).
fn collect_virtual_entries_from_dir(
    root: &Path,
    cur: &Path,
    inside_vendor_exp: bool,
    entries: &mut Vec<VirtualEntry>,
    deferred_archives: &mut Vec<std::path::PathBuf>,
) -> Result<(), String> {
    let rd = fs::read_dir(cur)
        .map_err(|e| format!("failed to read directory {}: {e}", cur.display()))?;
    let mut children: Vec<fs::DirEntry> = rd.flatten().collect();
    children.sort_by_key(|e| e.file_name());

    let mut child_names_lower = BTreeSet::new();
    for e in &children {
        if let Some(n) = e.file_name().to_str() {
            child_names_lower.insert(n.to_ascii_lowercase());
        }
    }
    let has_acqus = child_names_lower.contains("acqus") || child_names_lower.contains("acqu");
    let is_exp_root = inside_vendor_exp || has_acqus || child_names_lower.contains("procpar");

    let mut subdirs = Vec::new();
    let mut recorded_ft_inside_exp = false;

    for entry in children {
        let fname_os = entry.file_name();
        let Some(fname) = fname_os.to_str() else {
            continue;
        };
        if fname == ".git" || fname == "__MACOSX" || fname == ".DS_Store" || fname.starts_with("._")
        {
            continue;
        }
        let Ok(ft) = entry.file_type() else {
            continue;
        };
        if ft.is_dir() {
            let lower_dir = fname.to_ascii_lowercase();
            if is_exp_root {
                if lower_dir == "pdata" {
                    scan_pdata_procno_dir(root, &entry.path(), has_acqus, entries);
                } else if lower_dir == "ft" {
                    if let Ok(ft_rd) = fs::read_dir(entry.path()) {
                        let mut plane_count = 0u64;
                        let mut first_plane_size = 0u64;
                        let mut sample_rel = None;
                        for fe in ft_rd.flatten() {
                            if let Some(n) = fe.file_name().to_str() {
                                let nl = n.to_ascii_lowercase();
                                if nl.ends_with(".ft2")
                                    || nl.ends_with(".ft3")
                                    || nl.ends_with(".ft4")
                                {
                                    plane_count += 1;
                                    if sample_rel.is_none() {
                                        first_plane_size =
                                            fe.metadata().map(|m| m.len()).unwrap_or(0);
                                        if let Ok(rel) = fe.path().strip_prefix(root) {
                                            sample_rel =
                                                Some(normalize_rel_path(&rel.to_string_lossy()));
                                        }
                                    }
                                }
                            }
                        }
                        if let Some(rel) = sample_rel {
                            entries.push(VirtualEntry::header_only(
                                rel,
                                Vec::new(),
                                first_plane_size.saturating_mul(plane_count),
                            ));
                        }
                    }
                }
                // Skip all other subdirectories inside an experiment root (e.g. `fid/` slice dirs)
                continue;
            }
            subdirs.push(entry.path());
            continue;
        }

        if !ft.is_file() {
            continue;
        }

        let lower = fname.to_ascii_lowercase();
        // Skip duplicate Bruker non-status parameter files (`acqu`, `acqu2`, `acqu3`, `acqu4`)
        // when the status parameter file (`acqus`, `acqu2s`, ...) is present, and skip duplicate
        // `pulseprogram.precomp`/`pulprog` when executed `pulseprogram` is already present!
        if is_exp_root
            && ((lower == "acqu" && child_names_lower.contains("acqus"))
                || (lower == "acqu2" && child_names_lower.contains("acqu2s"))
                || (lower == "acqu3" && child_names_lower.contains("acqu3s"))
                || (lower == "acqu4" && child_names_lower.contains("acqu4s"))
                || ((lower == "pulseprogram.precomp" || lower == "pulprog")
                    && child_names_lower.contains("pulseprogram")))
        {
            continue;
        }
        let path = entry.path();
        if is_tar_path_lower(&lower) || is_targz_path_lower(&lower) {
            deferred_archives.push(path);
            continue;
        }

        let Ok(rel_os) = path.strip_prefix(root) else {
            continue;
        };
        let rel = normalize_rel_path(&rel_os.to_string_lossy());

        if is_exp_root
            && (lower.ends_with(".ft1")
                || lower.ends_with(".ft2")
                || lower.ends_with(".ft3")
                || lower.ends_with(".ft4"))
        {
            if !recorded_ft_inside_exp {
                let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
                entries.push(VirtualEntry::header_only(rel, Vec::new(), size));
                recorded_ft_inside_exp = true;
            }
            continue;
        }

        match classify_file_read_action(&rel) {
            FileReadAction::Ignore => {}
            FileReadAction::MetadataOnlySignal => {
                let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
                entries.push(VirtualEntry::header_only(rel, Vec::new(), size));
            }
            FileReadAction::BoundedBinary(limit) | FileReadAction::BoundedText(limit) => {
                if let Ok(prefix) = read_prefix_from_disk(&path, limit) {
                    let size = if prefix.len() < limit {
                        prefix.len() as u64
                    } else {
                        entry
                            .metadata()
                            .map(|m| m.len())
                            .unwrap_or(prefix.len() as u64)
                    };
                    entries.push(VirtualEntry::header_only(rel, prefix, size));
                }
            }
        }
    }

    for sub_path in subdirs {
        collect_virtual_entries_from_dir(root, &sub_path, false, entries, deferred_archives)?;
    }
    Ok(())
}

// ============================================================================
// Vendor Header Parsers (Bruker JCAMP-DX, Varian procpar, UCSF v2, NMRPipe)
// ============================================================================

fn canonical_nucleus(raw: &str) -> String {
    let cleaned = raw
        .trim()
        .trim_matches(|c: char| c == '<' || c == '>' || c == '"' || c == '\'')
        .trim();
    let up = cleaned.to_ascii_uppercase();
    match up.as_str() {
        "" | "OFF" | "NONE" | "UNKNOWN" => String::new(),
        "1H" | "H1" | "H" | "HN" | "PROTON" => "1H".to_string(),
        "15N" | "N15" | "N" | "NH" => "15N".to_string(),
        "13C" | "C13" | "C" | "CA" | "CB" | "CO" | "CACB" => "13C".to_string(),
        "2H" | "H2" | "D" => "2H".to_string(),
        "19F" | "F19" | "F" => "19F".to_string(),
        "31P" | "P31" | "P" => "31P".to_string(),
        other => other.to_string(),
    }
}

/// Proton-equivalent spectrometer base frequency in MHz derived from `(nucleus, obs_mhz)`
/// using IUPAC gyromagnetic ratios when `1H` channel `SFO1` is not directly observed (e.g. `13C`-detected `CON`).
fn infer_proton_base_mhz(nucleus: &str, obs_mhz: f64) -> Option<f64> {
    if obs_mhz <= 1.0 || !obs_mhz.is_finite() {
        return None;
    }
    let mhz = match nucleus {
        "1H" => obs_mhz,
        "13C" => obs_mhz * 3.976_933,
        "15N" => obs_mhz * 9.869_062,
        "31P" => obs_mhz * 2.470_376,
        "19F" => obs_mhz * 1.062_918,
        "2H" => obs_mhz * 6.514_399,
        _ => return None,
    };
    Some((mhz * 100.0).round() / 100.0)
}

fn parse_jcamp_map(bytes: &[u8]) -> BTreeMap<String, String> {
    let text = String::from_utf8_lossy(bytes);
    let mut map = BTreeMap::new();
    let mut current_array_key: Option<String> = None;
    for raw_line in text.lines() {
        let line = raw_line.trim();
        if let Some(rest) = line.strip_prefix("##$") {
            current_array_key = None;
            if let Some((k, v)) = rest.split_once('=') {
                let key = k.trim().to_string();
                let val = v.trim();
                if val.starts_with('(') && val.contains("..") {
                    if let Some((_spec, after)) = val.split_once(')') {
                        let tail = after.trim();
                        if !tail.is_empty() {
                            map.insert(key, tail.to_string());
                        } else {
                            current_array_key = Some(key);
                        }
                    }
                } else {
                    map.insert(key, val.to_string());
                }
            }
        } else if line.starts_with("##") || line.starts_with("$$") {
            current_array_key = None;
        } else if let Some(key) = current_array_key.take() {
            if !line.is_empty() {
                map.insert(key, line.to_string());
            }
        }
    }
    map
}

fn jcamp_f64(map: &BTreeMap<String, String>, key: &str) -> Option<f64> {
    map.get(key)?
        .split_whitespace()
        .next()?
        .parse::<f64>()
        .ok()
        .filter(|v| v.is_finite())
}

fn jcamp_usize(map: &BTreeMap<String, String>, key: &str) -> Option<usize> {
    map.get(key)?
        .split_whitespace()
        .next()?
        .parse::<isize>()
        .ok()
        .map(|v| v.max(0) as usize)
}

fn jcamp_str(map: &BTreeMap<String, String>, key: &str) -> Option<String> {
    let v = map.get(key)?.trim();
    let unquoted = v
        .trim_matches(|c: char| c == '<' || c == '>' || c == '"')
        .trim();
    if unquoted.is_empty() {
        None
    } else {
        Some(unquoted.to_string())
    }
}

fn bruker_fnmode_quadrature(fnmode: Option<usize>, is_direct: bool) -> QuadratureKind {
    match fnmode {
        Some(0) if is_direct => QuadratureKind::Dqd,
        Some(1) => QuadratureKind::Qf,
        Some(2) => QuadratureKind::Dqd,
        Some(3) => QuadratureKind::Tppi,
        Some(4) => QuadratureKind::States,
        Some(5) => QuadratureKind::StatesTppi,
        Some(6) => QuadratureKind::EchoAntiecho,
        _ if is_direct => QuadratureKind::Dqd,
        _ => QuadratureKind::Unknown,
    }
}

fn count_nonempty_schedule_lines(bytes: &[u8]) -> usize {
    let text = String::from_utf8_lossy(bytes);
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#') && !l.starts_with(';'))
        .count()
}

/// Parse Varian / Agilent `procpar` text into `(name -> Vec<tokens>)`.
fn parse_varian_procpar_map(bytes: &[u8]) -> BTreeMap<String, Vec<String>> {
    let text = String::from_utf8_lossy(bytes);
    let mut lines = text.lines().peekable();
    let mut map = BTreeMap::new();

    while let Some(header_line) = lines.next() {
        let h = header_line.trim();
        if h.is_empty() {
            continue;
        }
        let mut h_parts = h.split_whitespace();
        let Some(name) = h_parts.next() else {
            continue;
        };
        let Some(_subtype) = h_parts.next().and_then(|s| s.parse::<i32>().ok()) else {
            continue;
        };
        let Some(basictype) = h_parts.next().and_then(|s| s.parse::<i32>().ok()) else {
            continue;
        };
        if !(0..=2).contains(&basictype) {
            continue;
        }

        let Some(val_line) = lines.next() else {
            break;
        };
        let vtrim = val_line.trim();
        let mut count_split = vtrim.splitn(2, char::is_whitespace);
        let nvals = count_split
            .next()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        let mut rest = count_split.next().unwrap_or("").trim().to_string();

        let mut vals = Vec::with_capacity(nvals.min(32));
        if basictype == 2 {
            // Quoted string values (first on `val_line`, remaining `nvals - 1` on subsequent lines)
            if nvals > 0 {
                vals.push(unquote_varian(&rest));
                for _ in 1..nvals {
                    if let Some(next_s) = lines.next() {
                        vals.push(unquote_varian(next_s.trim()));
                    }
                }
            }
        } else {
            while vals.len() < nvals {
                for tok in rest.split_whitespace() {
                    vals.push(tok.to_string());
                    if vals.len() >= nvals {
                        break;
                    }
                }
                if vals.len() >= nvals {
                    break;
                }
                if let Some(more) = lines.next() {
                    rest = more.trim().to_string();
                } else {
                    break;
                }
            }
        }
        // Skip enum/protection line (`0` or `n ...`)
        let _ = lines.next();
        map.insert(name.to_string(), vals);
    }
    map
}

fn unquote_varian(s: &str) -> String {
    let t = s.trim();
    if let Some(stripped) = t.strip_prefix('"').and_then(|r| r.strip_suffix('"')) {
        stripped.to_string()
    } else {
        t.trim_matches('"').to_string()
    }
}

fn procpar_str(map: &BTreeMap<String, Vec<String>>, key: &str) -> Option<String> {
    map.get(key)?
        .first()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn procpar_f64(map: &BTreeMap<String, Vec<String>>, key: &str) -> Option<f64> {
    map.get(key)?
        .first()?
        .parse::<f64>()
        .ok()
        .filter(|v| v.is_finite())
}

fn procpar_usize(map: &BTreeMap<String, Vec<String>>, key: &str) -> Option<usize> {
    map.get(key)?
        .first()?
        .parse::<f64>()
        .ok()
        .map(|v| v.round().max(0.0) as usize)
}

fn varian_indirect_quadrature(
    map: &BTreeMap<String, Vec<String>>,
    phase_key: &str,
    coef_key: &str,
) -> QuadratureKind {
    if let Some(coef) = procpar_str(map, coef_key) {
        let vals: Vec<i32> = coef
            .split_whitespace()
            .filter_map(|t| t.parse::<i32>().ok())
            .collect();
        // Echo-Antiecho Rance-Kay recombination coefficients e.g. `"1 0 -1 0 0 1 0 1"` or `"1 0 1 0 0 -1 0 1"`
        if vals.len() == 8 && vals.iter().filter(|&&x| x != 0).count() == 4 {
            return QuadratureKind::EchoAntiecho;
        }
    }
    if let Some(ph) = map.get(phase_key) {
        if ph.len() >= 2 {
            if ph.iter().any(|v| v == "3") {
                return QuadratureKind::StatesTppi;
            }
            return QuadratureKind::States;
        }
        if ph.len() == 1 && ph[0] == "1" {
            return QuadratureKind::Qf;
        }
    }
    if let Some(arr) = procpar_str(map, "array") {
        if arr.contains(phase_key) {
            return QuadratureKind::States;
        }
    }
    QuadratureKind::States
}

/// Parse UCSF Sparky v2 header (`180` global + `ndim * 128` axis bytes).
fn parse_ucsf_header(
    bytes: &[u8],
) -> Result<(ExperimentDimensionality, Vec<ClassifiedAxis>, Option<f64>), String> {
    if bytes.len() < 180 || &bytes[0..8] != b"UCSF NMR" {
        return Err("not a UCSF v2 spectrum header".to_string());
    }
    let ndim = bytes[10] as usize;
    if !(1..=4).contains(&ndim) || bytes.len() < 180 + ndim * 128 {
        return Err(format!("invalid or truncated UCSF v2 ndim={ndim}"));
    }
    let dim = match ndim {
        1 => ExperimentDimensionality::Dim1D,
        2 => ExperimentDimensionality::Dim2D,
        3 => ExperimentDimensionality::Dim3D,
        _ => ExperimentDimensionality::Dim4D,
    };

    let mut axes = Vec::with_capacity(ndim);
    let mut base_mhz: Option<f64> = None;
    for d in 0..ndim {
        let off = 180 + d * 128;
        let raw_nuc_end = bytes[off..off + 6]
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(6);
        let raw_nuc = String::from_utf8_lossy(&bytes[off..off + raw_nuc_end]).to_string();
        let nuc = canonical_nucleus(&raw_nuc);
        let npts = i32::from_be_bytes(bytes[off + 8..off + 12].try_into().unwrap()).max(0) as usize;
        let obs_mhz = f32::from_be_bytes(bytes[off + 20..off + 24].try_into().unwrap()) as f64;
        let sw_hz = f32::from_be_bytes(bytes[off + 24..off + 28].try_into().unwrap()) as f64;
        let xmtr_ppm = f32::from_be_bytes(bytes[off + 28..off + 32].try_into().unwrap()) as f64;

        let sw_ppm = if obs_mhz > 1e-3 && sw_hz > 0.0 {
            Some(sw_hz / obs_mhz)
        } else {
            None
        };
        let carrier_ppm = xmtr_ppm.is_finite().then_some(xmtr_ppm);
        if nuc == "1H" && obs_mhz > 10.0 {
            base_mhz = Some((obs_mhz * 100.0).round() / 100.0);
        } else if base_mhz.is_none() {
            base_mhz = infer_proton_base_mhz(&nuc, obs_mhz);
        }
        axes.push(ClassifiedAxis {
            label: format!("w{}", d + 1),
            nucleus: if nuc.is_empty() { raw_nuc } else { nuc },
            points: npts,
            sw_ppm,
            carrier_ppm,
            obs_mhz: (obs_mhz > 1e-3).then_some(obs_mhz),
            quadrature: QuadratureKind::Processed,
        });
    }
    // In UCSF Sparky 3D/2D files, `w_ndim` or the `1H` axis is typically direct; put `1H` (or direct) first if needed
    if axes.len() >= 2 && axes[0].nucleus != "1H" {
        if let Some(h_pos) = axes.iter().rposition(|a| a.nucleus == "1H") {
            let h_ax = axes.remove(h_pos);
            axes.insert(0, h_ax);
        }
    }
    Ok((dim, axes, base_mhz))
}

/// Parse NMRPipe 512-float (`2048`-byte) header.
fn parse_nmrpipe_header(
    bytes: &[u8],
    plane_count_override: Option<usize>,
) -> Result<(ExperimentDimensionality, Vec<ClassifiedAxis>, Option<f64>), String> {
    if bytes.len() < NMRPIPE_HEADER_BYTES {
        return Err("NMRPipe header shorter than 2048 bytes".to_string());
    }
    let order_le = f32::from_le_bytes(bytes[8..12].try_into().unwrap());
    let order_be = f32::from_be_bytes(bytes[8..12].try_into().unwrap());
    let big_endian = if (order_le - 2.345).abs() < 1e-2 {
        false
    } else if (order_be - 2.345).abs() < 1e-2 {
        true
    } else {
        return Err("invalid NMRPipe FDORDER magic".to_string());
    };

    let mut hdr = [0.0f32; 512];
    for (i, chunk) in bytes[..NMRPIPE_HEADER_BYTES].chunks_exact(4).enumerate() {
        let arr: [u8; 4] = chunk.try_into().unwrap();
        hdr[i] = if big_endian {
            f32::from_be_bytes(arr)
        } else {
            f32::from_le_bytes(arr)
        };
    }

    let mut ndim = (hdr[9].round() as usize).clamp(1, 4);
    if plane_count_override.is_some_and(|c| c > 1) && ndim < 3 {
        ndim = 3;
    }
    let dim = match ndim {
        1 => ExperimentDimensionality::Dim1D,
        2 => ExperimentDimensionality::Dim2D,
        3 => ExperimentDimensionality::Dim3D,
        _ => ExperimentDimensionality::Dim4D,
    };

    const SW_IDX: [usize; 4] = [100, 229, 11, 29];
    const ORIG_IDX: [usize; 4] = [101, 249, 12, 30];
    const OBS_IDX: [usize; 4] = [119, 218, 10, 28];
    const LABEL_IDX: [usize; 4] = [16, 18, 20, 22];

    let sizes = [
        (hdr[99].round() as usize).max(1),
        (hdr[219].round() as usize).max(1),
        plane_count_override.unwrap_or_else(|| (hdr[15].round() as usize).max(1)),
        (hdr[32].round() as usize).max(1),
    ];

    let mut axes = Vec::with_capacity(ndim);
    let mut base_mhz: Option<f64> = None;
    for (d, &sz) in sizes.iter().enumerate().take(ndim) {
        let code = hdr[24 + d].round() as isize;
        let slot = match code {
            2 => 0,
            1 => 1,
            3 => 2,
            4 => 3,
            _ => d.min(3),
        };
        let lbl_start = LABEL_IDX[slot] * 4;
        let lbl_slice = &bytes[lbl_start..lbl_start + 8];
        let lbl_end = lbl_slice.iter().position(|&b| b == 0).unwrap_or(8);
        let raw_lbl = String::from_utf8_lossy(&lbl_slice[..lbl_end])
            .trim()
            .to_string();
        let mut nuc = canonical_nucleus(&raw_lbl);
        let obs_mhz = hdr[OBS_IDX[slot]] as f64;
        let sw_hz = hdr[SW_IDX[slot]] as f64;
        let orig_hz = hdr[ORIG_IDX[slot]] as f64;

        // Infer nucleus from gyromagnetic ratio if label is generic (`X`, `Y`, `Z`)
        if !matches!(nuc.as_str(), "1H" | "15N" | "13C" | "31P" | "19F" | "2H") {
            if let Some(b) = base_mhz {
                let ratio = obs_mhz / b;
                if (ratio - 1.0).abs() < 0.05 {
                    nuc = "1H".to_string();
                } else if (ratio - 0.25145).abs() < 0.03 {
                    nuc = "13C".to_string();
                } else if (ratio - 0.10133).abs() < 0.02 {
                    nuc = "15N".to_string();
                }
            } else if obs_mhz > 350.0 {
                nuc = "1H".to_string();
            }
        }
        if nuc == "1H" && obs_mhz > 10.0 {
            base_mhz = Some((obs_mhz * 100.0).round() / 100.0);
        } else if base_mhz.is_none() {
            base_mhz = infer_proton_base_mhz(&nuc, obs_mhz);
        }

        let sw_ppm = (obs_mhz > 1e-3 && sw_hz > 0.0).then_some(sw_hz / obs_mhz);
        let carrier_ppm = sw_ppm.map(|sw| (orig_hz / obs_mhz) + 0.5 * sw);
        axes.push(ClassifiedAxis {
            label: if raw_lbl.is_empty() {
                format!("F{}", d + 1)
            } else {
                raw_lbl
            },
            nucleus: nuc,
            points: sz,
            sw_ppm,
            carrier_ppm,
            obs_mhz: (obs_mhz > 1e-3).then_some(obs_mhz),
            quadrature: QuadratureKind::Processed,
        });
    }
    Ok((dim, axes, base_mhz))
}

// ============================================================================
// Pulse-Sequence & Axis-Physics Classifier
// ============================================================================

fn classify_pulse_token(
    raw_token: &str,
    dim: ExperimentDimensionality,
    axes: &[ClassifiedAxis],
) -> Option<PulseSequenceClass> {
    let lower = raw_token.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return None;
    }
    // Strip trailing vendor author/platform extensions (`.bv`, `.fp`, `.fp2`, `.fa`, `.sas`, `.sct`, `.be`, `.2`)
    let mut base = lower.as_str();
    for ext in [
        ".sas", ".sct", ".fp2", ".bv", ".fp", ".fa", ".be", ".1", ".2", ".3",
    ] {
        if let Some(s) = base.strip_suffix(ext) {
            base = s;
        }
    }
    let compact: String = base.chars().filter(|c| c.is_ascii_alphanumeric()).collect();

    if dim == ExperimentDimensionality::Dim1D
        || compact == "pulsecal"
        || compact == "p1cal"
        || compact.starts_with("1dp")
        || compact == "zg"
        || compact == "zg30"
        || compact == "zgpr"
        || compact == "zgesgp"
        || compact == "s2pul"
    {
        return Some(PulseSequenceClass::Unknown);
    }

    let direct_nuc = axes.first().map(|a| a.nucleus.as_str()).unwrap_or("");
    let has_15n = axes.iter().skip(1).any(|a| a.nucleus == "15N");
    let has_13c = axes.iter().skip(1).any(|a| a.nucleus == "13C");

    // 1. Exchange & Relaxation (`CEST`, `CPMG`, `HetNOE`, `R1`, `R2`)
    if compact.contains("cest") {
        return Some(PulseSequenceClass::Cest);
    }
    if compact.contains("cpmg") {
        return Some(PulseSequenceClass::Cpmg);
    }
    // Steady-state heteronuclear NOE (`hsqcnoef3gpsi`, `invinoe`, `c_hnco_noe`, `hetnoe`, `gNnoe`)
    // Distinguished from 3D/4D NOESY (`noesy`, `nohsgp`, `noehsqc`)
    if (compact.contains("hsqcnoe")
        || compact.contains("invinoe")
        || compact.contains("hetnoe")
        || compact.contains("nhnoe")
        || base.contains("_noe")
        || compact.contains("gnnoe"))
        && !compact.contains("noesy")
        && !compact.contains("nohsgp")
    {
        return Some(PulseSequenceClass::HetNoe);
    }
    // Longitudinal R1/T1 (`hsqct1`, `invit1`, `_t1_`, `gt1`, `nh_t1`, `r1_`) — NOT `.t1` evolution suffix on `hcacon`!
    if !compact.contains("hcacon")
        && (compact.contains("hsqct1")
            || compact.contains("invit1")
            || base.contains("_t1_")
            || base.ends_with("_t1")
            || base.contains("_r1_")
            || base.ends_with("_r1")
            || compact.starts_with("gt1")
            || compact.contains("nht1")
            || compact.contains("trosyt1"))
    {
        return Some(PulseSequenceClass::RelaxR1);
    }
    // Transverse R2/T2 / R1rho (`hsqct2`, `invit2`, `_t2_`, `gt2`, `r1rho`, `t1rho`)
    if compact.contains("hsqct2")
        || compact.contains("invit2")
        || base.contains("_t2_")
        || base.ends_with("_t2")
        || base.contains("_r2_")
        || base.ends_with("_r2")
        || compact.starts_with("gt2")
        || compact.contains("nht2")
        || compact.contains("trosyt2")
        || compact.contains("r1rho")
        || compact.contains("t1rho")
    {
        return Some(PulseSequenceClass::RelaxR2);
    }

    // 2. Protonless `13C`-detected suite (`CON`, `CACO`, `CANCO`, `CBCACO` — Bermel et al. 2009)
    // Guard against `1H`-detected `CBCACONH` (whose compact token starts with `"cbcaco"` + `"nh"`).
    if matches!(
        dim,
        ExperimentDimensionality::Dim3D | ExperimentDimensionality::Dim4D
    ) && direct_nuc != "1H"
        && !compact.contains("cbcaconh")
        && !compact.contains("conh")
    {
        if compact.contains("cbcaco") || compact.contains("cbcanco") {
            return Some(PulseSequenceClass::Cbcaco3D);
        }
        if compact.contains("canco") || (direct_nuc == "13C" && compact.contains("ncaco")) {
            return Some(PulseSequenceClass::Canco3D);
        }
    }
    if dim == ExperimentDimensionality::Dim2D {
        if compact.contains("hcacon")
            || base.starts_with("c_con")
            || compact == "con"
            || (direct_nuc == "13C" && has_15N_only(axes) && compact.contains("con"))
        {
            return Some(PulseSequenceClass::Con2D);
        }
        if base.starts_with("c_caco")
            || compact.contains("hcaco")
            || compact == "caco"
            || (direct_nuc == "13C" && has_13C_only(axes) && compact.contains("caco"))
        {
            return Some(PulseSequenceClass::Caco2D);
        }
    }

    // 3. 3D / 4D NOESY (`15N-NOESY` vs `13C-NOESY`)
    if compact.contains("noesy") || compact.contains("nohsgp") || compact.contains("noehsqc") {
        // Explicit isotope tokens in the pulse-sequence name take precedence over indirect axis defaults
        if compact.contains("n15")
            || compact.contains("15n")
            || compact.contains("nhsqc")
            || compact.contains("nohsgp19d")
        {
            return Some(PulseSequenceClass::N15Noesy);
        }
        if compact.contains("c13")
            || compact.contains("13c")
            || compact.contains("noesyc")
            || compact.contains("cnoesy")
            || compact.contains("c4dnoesy")
        {
            return Some(PulseSequenceClass::C13Noesy);
        }
        if has_15n && !has_13c {
            return Some(PulseSequenceClass::N15Noesy);
        }
        if has_13c && !has_15n {
            return Some(PulseSequenceClass::C13Noesy);
        }
        if has_15n {
            return Some(PulseSequenceClass::N15Noesy);
        }
        if has_13c {
            return Some(PulseSequenceClass::C13Noesy);
        }
    }

    // 4. Aliphatic Sidechain TOCSY (`HCCH-TOCSY`, `H(CCO)NH`, `CC(CO)NH`)
    if compact.contains("hcchtocsy")
        || compact.contains("hcchcosy")
        || compact.contains("hcchdipsi")
    {
        return Some(PulseSequenceClass::HcchTocsy);
    }
    if compact.contains("hccconh")
        || compact.contains("hcconh")
        || base.contains("ghc_co_nh")
        || compact.contains("hcctocsy")
        || compact.contains("hcccon")
    {
        return Some(PulseSequenceClass::Hccconh);
    }
    if compact.contains("ccconh")
        || compact.contains("cconh")
        || base.contains("gc_co_nh")
        || compact.contains("ccctocsy")
        || compact.contains("cccon")
    {
        return Some(PulseSequenceClass::Ccconh);
    }

    // 5. 3D Backbone Triple-Resonance (specific multi-step transfers checked BEFORE shorter prefixes!)
    if matches!(
        dim,
        ExperimentDimensionality::Dim3D | ExperimentDimensionality::Dim4D
    ) {
        if compact.contains("cbcaconh")
            || compact.contains("cbcaconnh")
            || compact.contains("cbcacogp")
        {
            return Some(PulseSequenceClass::Cbcaconh);
        }
        if compact.contains("hncacb") || compact.contains("cbcanh") {
            return Some(PulseSequenceClass::Hncacb);
        }
        if compact.contains("hncaco") {
            return Some(PulseSequenceClass::Hncaco);
        }
        if compact.contains("hncoca") {
            return Some(PulseSequenceClass::Hncoca);
        }
        if compact.contains("hnca") && !compact.contains("hncannh") {
            return Some(PulseSequenceClass::Hnca);
        }
        if compact.contains("hnco") {
            return Some(PulseSequenceClass::Hnco);
        }
    }

    // 6. 2D Heteronuclear Correlation (`15N-HSQC`, `13C-HSQC`, `13C-HMQC` Methyl-TROSY)
    // Decide methyl (`C13Hmqc`) primarily from `13C` carrier (`8.0..=28.0 ppm`) and `SW <= 36.0 ppm`,
    // treating `hmqc`/`hsqc` in `PULPROG`/`seqfil` only as a fallback hint when carrier/SW are missing.
    if matches!(
        dim,
        ExperimentDimensionality::Dim2D | ExperimentDimensionality::Pseudo3D
    ) {
        if compact.contains("hmqc") || compact.contains("methyltrosy") {
            if has_15n && !has_13c && !compact.contains("c13") && !compact.contains("13c") {
                return Some(PulseSequenceClass::N15Hsqc);
            }
            return Some(classify_2d_13c_heteronuclear(axes, Some(true)));
        }
        if compact.contains("hsqc")
            || compact.contains("trosy")
            || compact.contains("fhsqc")
            || compact.contains("hsgp")
            || compact.contains("inviet")
        {
            let explicit_methyl_tok = compact.contains("methyl");
            if has_13c && !has_15n {
                return Some(classify_2d_13c_heteronuclear(
                    axes,
                    Some(explicit_methyl_tok),
                ));
            }
            if has_15n && !has_13c {
                return Some(PulseSequenceClass::N15Hsqc);
            }
            if compact.contains("c13") || compact.contains("13c") || compact.contains("chsqc") {
                return Some(classify_2d_13c_heteronuclear(
                    axes,
                    Some(explicit_methyl_tok),
                ));
            }
            if compact.contains("n15") || compact.contains("15n") || compact.contains("nhsqc") {
                return Some(PulseSequenceClass::N15Hsqc);
            }
            // Default heteronuclear `hsqc`/`trosy` when axes lack explicit NUC labels
            return Some(PulseSequenceClass::N15Hsqc);
        }
    }

    None
}

/// Classify a 2D `1H`-`13C` heteronuclear correlation spectrum into methyl (`C13Hmqc`) vs
/// general aliphatic/aromatic (`C13Hsqc`), using `13C` indirect carrier (`8.0..=28.0 ppm`)
/// and sweep width (`SW <= 36.0 ppm`) as primary physical evidence and falling back to
/// `token_prefers_methyl` (`hmqc`/`methyltrosy` vs `hsqc`) only when carrier/SW are missing.
fn classify_2d_13c_heteronuclear(
    axes: &[ClassifiedAxis],
    token_prefers_methyl: Option<bool>,
) -> PulseSequenceClass {
    let c_ax = axes.iter().skip(1).find(|a| a.nucleus == "13C");
    let car = c_ax.and_then(|a| a.carrier_ppm);
    let sw = c_ax.and_then(|a| a.sw_ppm);

    if let (Some(c), Some(s)) = (car, sw) {
        return if (8.0..=28.0).contains(&c) && s <= 36.0 {
            PulseSequenceClass::C13Hmqc
        } else {
            PulseSequenceClass::C13Hsqc
        };
    }
    if car.is_some_and(|c| !(8.0..=28.0).contains(&c)) || sw.is_some_and(|s| s > 36.0) {
        return PulseSequenceClass::C13Hsqc;
    }
    match token_prefers_methyl {
        Some(true) => PulseSequenceClass::C13Hmqc,
        _ => PulseSequenceClass::C13Hsqc,
    }
}

#[allow(non_snake_case)]
fn has_15N_only(axes: &[ClassifiedAxis]) -> bool {
    axes.iter().skip(1).any(|a| a.nucleus == "15N")
        && !axes.iter().skip(1).any(|a| a.nucleus == "13C")
}

#[allow(non_snake_case)]
fn has_13C_only(axes: &[ClassifiedAxis]) -> bool {
    axes.iter().skip(1).any(|a| a.nucleus == "13C")
        && !axes.iter().skip(1).any(|a| a.nucleus == "15N")
}

/// Pure axis-physics fallback when pulse program and title strings are absent or generic.
fn infer_pulse_class_from_axis_physics(
    dim: ExperimentDimensionality,
    axes: &[ClassifiedAxis],
    pseudo_kind: Option<&str>,
) -> PulseSequenceClass {
    if dim == ExperimentDimensionality::Dim1D {
        return PulseSequenceClass::Unknown;
    }
    if dim == ExperimentDimensionality::Pseudo3D {
        return match pseudo_kind {
            Some("vclist") => PulseSequenceClass::Cpmg,
            Some("fqlist") => PulseSequenceClass::Cest,
            _ => PulseSequenceClass::RelaxR1,
        };
    }

    let direct = axes.first().map(|a| a.nucleus.as_str()).unwrap_or("");
    if dim == ExperimentDimensionality::Dim2D && axes.len() >= 2 {
        let ind = axes[1].nucleus.as_str();
        if direct == "1H" && ind == "15N" {
            return PulseSequenceClass::N15Hsqc;
        }
        if direct == "1H" && ind == "13C" {
            return classify_2d_13c_heteronuclear(axes, None);
        }
        if direct == "13C" && ind == "15N" {
            return PulseSequenceClass::Con2D;
        }
        if direct == "13C" && ind == "13C" {
            return PulseSequenceClass::Caco2D;
        }
    }

    if dim == ExperimentDimensionality::Dim3D && axes.len() >= 3 {
        let ind1 = axes[1].nucleus.as_str();
        let ind2 = axes[2].nucleus.as_str();
        if direct == "13C" && ((ind1 == "15N" && ind2 == "13C") || (ind1 == "13C" && ind2 == "15N"))
        {
            let c_ax = if ind1 == "13C" { &axes[1] } else { &axes[2] };
            return if c_ax.sw_ppm.unwrap_or(30.0) >= 48.0 {
                PulseSequenceClass::Cbcaco3D
            } else {
                PulseSequenceClass::Canco3D
            };
        }
        if direct == "1H" && ((ind1 == "15N" && ind2 == "1H") || (ind1 == "1H" && ind2 == "15N")) {
            return PulseSequenceClass::N15Noesy;
        }
        if direct == "1H" && ((ind1 == "13C" && ind2 == "1H") || (ind1 == "1H" && ind2 == "13C")) {
            return PulseSequenceClass::C13Noesy;
        }
        if direct == "1H" && ind1 == "13C" && ind2 == "13C" {
            return PulseSequenceClass::HcchTocsy;
        }
        if direct == "1H" && ((ind1 == "15N" && ind2 == "13C") || (ind1 == "13C" && ind2 == "15N"))
        {
            let c_ax = if ind1 == "13C" { &axes[1] } else { &axes[2] };
            let sw = c_ax.sw_ppm.unwrap_or(0.0);
            let car = c_ax.carrier_ppm.unwrap_or(0.0);
            if car > 155.0 || (sw > 0.0 && sw <= 22.0 && car == 0.0) {
                return PulseSequenceClass::Hnco;
            }
            if sw >= 48.0 || (car > 25.0 && car < 47.0 && sw >= 40.0) {
                return PulseSequenceClass::Hncacb;
            }
            if (45.0..=68.0).contains(&car) || (sw > 22.0 && sw < 48.0) {
                return PulseSequenceClass::Hnca;
            }
        }
    }

    PulseSequenceClass::Unknown
}

// ============================================================================
// Virtual Entry Grouping, Vendor Builders & Workspace Regime Inference
// ============================================================================

#[derive(Default)]
struct BrukerFolderGroup {
    acqus: Option<Vec<u8>>,
    acqu2s: Option<Vec<u8>>,
    acqu3s: Option<Vec<u8>>,
    acqu4s: Option<Vec<u8>>,
    procs: Option<Vec<u8>>,
    proc2s: Option<Vec<u8>>,
    proc3s: Option<Vec<u8>>,
    proc4s: Option<Vec<u8>>,
    title: Option<Vec<u8>>,
    pulseprogram: Option<Vec<u8>>,
    nuslist: Option<Vec<u8>>,
    difflist: Option<Vec<u8>>,
    vdlist: Option<Vec<u8>>,
    vclist: Option<Vec<u8>>,
    fqlist: Option<Vec<u8>>,
    local_includes: std::collections::HashMap<String, String>,
    raw_bytes: u64,
    processed_bytes: u64,
}

#[derive(Default)]
struct VarianFolderGroup {
    procpar: Option<Vec<u8>>,
    text: Option<Vec<u8>>,
    sampling_sch: Option<Vec<u8>>,
    raw_bytes: u64,
    processed_bytes: u64,
}

struct NmrPipeGroup {
    representative_path: String,
    stem_name: String,
    header_bytes: Vec<u8>,
    plane_count: usize,
    total_bytes: u64,
}

fn find_bruker_root_for_path(rel_path: &str) -> Option<(String, &str)> {
    let parts: Vec<&str> = rel_path.split('/').collect();
    if parts.is_empty() {
        return None;
    }
    let last_lower = parts.last().unwrap().to_ascii_lowercase();

    // Direct acquisition / schedule files inside `<exp_root>/<file>`
    if matches!(
        last_lower.as_str(),
        "acqus"
            | "acqu"
            | "acqu2s"
            | "acqu2"
            | "acqu3s"
            | "acqu3"
            | "acqu4s"
            | "acqu4"
            | "pulseprogram"
            | "pulseprogram.precomp"
            | "pulprog"
            | "nuslist"
            | "difflist"
            | "vdlist"
            | "vclist"
            | "valist"
            | "vplist"
            | "fqlist"
            | "fq1list"
            | "fq2list"
            | "fq3list"
            | "fq4list"
            | "fq5list"
            | "fq6list"
            | "fq7list"
            | "fq8list"
            | "ser"
    ) {
        let root = if parts.len() > 1 {
            parts[..parts.len() - 1].join("/")
        } else {
            ".".to_string()
        };
        return Some((root, parts.last().unwrap()));
    }

    // `pdata/<procno>/{procs,proc2s,proc3s,title,1r,2rr,3rrr,...}` -> `<exp_root>` is parent of `pdata`
    for (idx, seg) in parts.iter().enumerate() {
        if seg.eq_ignore_ascii_case("pdata") && idx + 2 < parts.len() {
            let root = if idx == 0 {
                ".".to_string()
            } else {
                parts[..idx].join("/")
            };
            return Some((root, parts.last().unwrap()));
        }
    }

    None
}

fn find_varian_root_for_path(rel_path: &str) -> Option<(String, &str)> {
    let parts: Vec<&str> = rel_path.split('/').collect();
    if parts.is_empty() {
        return None;
    }
    let last_lower = parts.last().unwrap().to_ascii_lowercase();
    if matches!(last_lower.as_str(), "procpar" | "sampling.sch") {
        let mut end = parts.len() - 1;
        if end > 0 && parts[end - 1].eq_ignore_ascii_case("acqfil") {
            end -= 1;
        }
        let root = if end == 0 {
            ".".to_string()
        } else {
            parts[..end].join("/")
        };
        return Some((root, parts.last().unwrap()));
    }
    None
}

fn split_nmrpipe_series_key(rel_path: &str) -> Option<(String, String)> {
    let (parent, fname) = match rel_path.rsplit_once('/') {
        Some((p, f)) => (p, f),
        None => ("", rel_path),
    };
    let (stem, ext) = fname.rsplit_once('.')?;
    let ext_l = ext.to_ascii_lowercase();
    if !matches!(ext_l.as_str(), "ft1" | "ft2" | "ft3" | "ft4" | "ft") {
        return None;
    }
    let digit_count = stem
        .chars()
        .rev()
        .take_while(|c| c.is_ascii_digit())
        .count();
    if digit_count >= 3 && stem.len() > digit_count {
        let alpha_prefix = &stem[..stem.len() - digit_count];
        let key = if parent.is_empty() {
            format!("{alpha_prefix}*.{ext_l}")
        } else {
            format!("{parent}/{alpha_prefix}*.{ext_l}")
        };
        return Some((key, alpha_prefix.to_string()));
    }
    None
}

fn format_quadrature_badge(axes: &[ClassifiedAxis]) -> String {
    if axes.len() <= 1 {
        return axes
            .first()
            .map(|a| a.quadrature.as_str().to_string())
            .unwrap_or_else(|| "Unknown".to_string());
    }
    let mut uniq = Vec::new();
    for ax in axes.iter().skip(1) {
        let s = ax.quadrature.as_str();
        if !uniq.contains(&s) {
            uniq.push(s);
        }
    }
    uniq.join(" / ")
}

fn display_name_from_rel(rel_path: &str, root_name: &str) -> String {
    if rel_path == "." || rel_path.is_empty() {
        return root_name.to_string();
    }
    let segs: Vec<&str> = rel_path.split('/').collect();
    if segs.len() >= 2 {
        format!("{}/{}", segs[segs.len() - 2], segs[segs.len() - 1])
    } else {
        segs[0].to_string()
    }
}

fn build_bruker_experiment(
    rel_root: &str,
    root_name: &str,
    grp: &BrukerFolderGroup,
) -> Option<ClassifiedExperiment> {
    if grp.acqus.is_none() && grp.procs.is_none() {
        return None;
    }

    let a1 = grp
        .acqus
        .as_deref()
        .map(parse_jcamp_map)
        .unwrap_or_default();
    let a2 = grp
        .acqu2s
        .as_deref()
        .map(parse_jcamp_map)
        .unwrap_or_default();
    let a3 = grp
        .acqu3s
        .as_deref()
        .map(parse_jcamp_map)
        .unwrap_or_default();
    let a4 = grp
        .acqu4s
        .as_deref()
        .map(parse_jcamp_map)
        .unwrap_or_default();

    let p1 = grp
        .procs
        .as_deref()
        .map(parse_jcamp_map)
        .unwrap_or_default();
    let p2 = grp
        .proc2s
        .as_deref()
        .map(parse_jcamp_map)
        .unwrap_or_default();
    let p3 = grp
        .proc3s
        .as_deref()
        .map(parse_jcamp_map)
        .unwrap_or_default();
    let p4 = grp
        .proc4s
        .as_deref()
        .map(parse_jcamp_map)
        .unwrap_or_default();

    let has_d4 = (!a4.is_empty() && jcamp_usize(&a4, "TD").unwrap_or(0) > 1)
        || (!p4.is_empty() && jcamp_usize(&p4, "SI").unwrap_or(0) > 1);
    let has_d3 = has_d4
        || (!a3.is_empty()
            && (jcamp_usize(&a3, "TD").unwrap_or(0) > 1
                || jcamp_usize(&a3, "NusTD").unwrap_or(0) > 1))
        || (!p3.is_empty() && jcamp_usize(&p3, "SI").unwrap_or(0) > 1);
    let has_d2 = has_d3
        || (!a2.is_empty()
            && (jcamp_usize(&a2, "TD").unwrap_or(0) > 1
                || jcamp_usize(&a2, "NusTD").unwrap_or(0) > 1))
        || (!p2.is_empty() && jcamp_usize(&p2, "SI").unwrap_or(0) > 1);

    let ndim_raw: usize = if has_d4 {
        4
    } else if has_d3 {
        3
    } else if has_d2 {
        2
    } else {
        1
    };

    let build_axis = |idx: usize,
                      amap: &BTreeMap<String, String>,
                      pmap: &BTreeMap<String, String>|
     -> ClassifiedAxis {
        let is_direct = idx == 0;
        let raw_nuc = jcamp_str(amap, "NUC1")
            .or_else(|| jcamp_str(pmap, "AXNUC"))
            .unwrap_or_default();
        let mut nuc = canonical_nucleus(&raw_nuc);
        let sfo1 = jcamp_f64(amap, "SFO1").or_else(|| jcamp_f64(pmap, "SF"));
        let bf1 = jcamp_f64(amap, "BF1").or(sfo1);
        let mut sw_ppm = jcamp_f64(amap, "SW");
        if sw_ppm.is_none() || sw_ppm == Some(0.0) {
            if let (Some(sw_hz), Some(sf)) = (
                jcamp_f64(amap, "SW_h").or_else(|| jcamp_f64(pmap, "SW_p")),
                sfo1,
            ) {
                if sf > 1e-3 {
                    sw_ppm = Some(sw_hz / sf);
                }
            }
        }
        let carrier_ppm = if let (Some(sfo), Some(bf)) = (sfo1, bf1) {
            if bf > 1.0 && (sfo - bf).abs() > 1e-9 {
                Some((sfo - bf) * 1.0e6 / bf)
            } else if let (Some(off), Some(sw)) = (jcamp_f64(pmap, "OFFSET"), sw_ppm) {
                Some(off - 0.5 * sw)
            } else {
                None
            }
        } else {
            None
        };

        let pts = if !amap.is_empty() {
            let td = jcamp_usize(amap, "TD").unwrap_or(0);
            let nustd = jcamp_usize(amap, "NusTD").unwrap_or(0);
            if td <= 1 && nustd > 1 {
                nustd
            } else {
                td
            }
        } else {
            jcamp_usize(pmap, "SI").unwrap_or(0)
        };

        let quad = if !amap.is_empty() {
            bruker_fnmode_quadrature(jcamp_usize(amap, "FnMODE"), is_direct)
        } else {
            QuadratureKind::Processed
        };
        if nuc.is_empty() && quad == QuadratureKind::Qf && !is_direct {
            nuc = "Delay".to_string();
        }

        let lbl = format!("F{}", ndim_raw.saturating_sub(idx));
        ClassifiedAxis {
            label: lbl,
            nucleus: nuc,
            points: pts,
            sw_ppm: sw_ppm.filter(|&v| v > 0.0),
            carrier_ppm,
            obs_mhz: sfo1.filter(|&v| v > 0.1),
            quadrature: quad,
        }
    };

    let mut axes = Vec::with_capacity(ndim_raw);
    axes.push(build_axis(0, &a1, &p1));
    if ndim_raw >= 2 {
        axes.push(build_axis(1, &a2, &p2));
    }
    if ndim_raw >= 3 {
        axes.push(build_axis(2, &a3, &p3));
    }
    if ndim_raw >= 4 {
        axes.push(build_axis(3, &a4, &p4));
    }

    let mut base_mhz: Option<f64> = None;
    // Check `SFO1..SFO4` on `acqus` for explicit `1H` channel first
    for ch in 1..=4 {
        let nkey = format!("NUC{ch}");
        let skey = format!("SFO{ch}");
        if let (Some(n), Some(sf)) = (jcamp_str(&a1, &nkey), jcamp_f64(&a1, &skey)) {
            if canonical_nucleus(&n) == "1H" && sf > 40.0 {
                base_mhz = Some((sf * 100.0).round() / 100.0);
                break;
            }
        }
    }
    if base_mhz.is_none() {
        for ax in &axes {
            if let Some(obs) = ax.obs_mhz {
                if let Some(m) = infer_proton_base_mhz(&ax.nucleus, obs) {
                    base_mhz = Some(m);
                    break;
                }
            }
        }
    }

    let pulprog = jcamp_str(&a1, "PULPROG").or_else(|| {
        grp.pulseprogram.as_deref().and_then(|b| {
            let txt = String::from_utf8_lossy(b);
            txt.lines()
                .map(str::trim)
                .find(|l| l.starts_with(';') && l.len() > 2)
                .map(|l| l.trim_start_matches(';').trim().to_string())
        })
    });

    let title = grp
        .title
        .as_deref()
        .map(|b| {
            String::from_utf8_lossy(b)
                .lines()
                .map(str::trim)
                .filter(|l| !l.is_empty())
                .collect::<Vec<_>>()
                .join(" | ")
        })
        .filter(|s| !s.is_empty());

    // Detect `Pseudo3D`:
    // 1. `vdlist` or `vclist` present (relaxation T1/T2 or CPMG counter stepped across planes), OR
    // 2. Any indirect axis (`acqu2s` or `acqu3s`) uses `FnMODE == 1 (QF)` whose `TD` matches
    //    `vdlist`/`vclist`/`fqlist` length (e.g. interleaved `acqu2s`-stepped 3D `CEST` `fq1list`),
    //    or `ndim_raw == 3` where the slowest axis (`acqu3s`) uses `FnMODE == 1 (QF)`, OR
    // 3. `ndim_raw == 2` with a multi-offset `fqlist` (`>= 4` offsets, e.g. CEST saturation sweep — NOT a 2-entry `CA`/`CO` pulse switch in hypercomplex 3D `hncaco`!).
    let vd_len = grp
        .vdlist
        .as_deref()
        .map(count_nonempty_schedule_lines)
        .unwrap_or(0);
    let vc_len = grp
        .vclist
        .as_deref()
        .map(count_nonempty_schedule_lines)
        .unwrap_or(0);
    let fq_len = grp
        .fqlist
        .as_deref()
        .map(count_nonempty_schedule_lines)
        .unwrap_or(0);
    let diff_len = grp
        .difflist
        .as_deref()
        .map(count_nonempty_schedule_lines)
        .unwrap_or(0);

    // Run `nmr-core::pulprog` symbolic pulseprogram & acquisition-physics interpreter when
    // `acqus` is present and either:
    // - `PULPROG` name heuristics cannot classify the sequence (custom/renamed laboratory sequence),
    // - any indirect axis has `QF` or `Unknown` `FnMODE` (manual `ip*`/`dp*` `States-TPPI`, pseudo-3D, or DOSY),
    // - variable delay/counter/frequency/diffusion lists are present, or
    // - `AQSEQ == 1` (`312` transposed) or 3D `AQSEQ` is omitted.
    let prelim_dim = match ndim_raw {
        1 => ExperimentDimensionality::Dim1D,
        2 => ExperimentDimensionality::Dim2D,
        3 => ExperimentDimensionality::Dim3D,
        _ => ExperimentDimensionality::Dim4D,
    };
    let prelim_pulprog_known = pulprog
        .as_deref()
        .and_then(|pp| classify_pulse_token(pp, prelim_dim, &axes))
        .is_some_and(|c| c != PulseSequenceClass::Unknown);
    let has_qf_or_unknown_indirect = axes
        .iter()
        .skip(1)
        .any(|a| matches!(a.quadrature, QuadratureKind::Qf | QuadratureKind::Unknown));
    let needs_pulprog_interp = grp.acqus.is_some()
        && (grp.difflist.is_some()
            || (grp.pulseprogram.is_some()
                && ((!prelim_pulprog_known && ndim_raw >= 2)
                    || has_qf_or_unknown_indirect
                    || vd_len > 0
                    || vc_len > 0
                    || fq_len > 0
                    || jcamp_usize(&a1, "AQSEQ") == Some(1)
                    || ndim_raw >= 3)));

    let pulprog_interp = if needs_pulprog_interp {
        let a1_s = grp
            .acqus
            .as_deref()
            .map(|b| String::from_utf8_lossy(b).into_owned())
            .unwrap_or_default();
        let a2_s = grp
            .acqu2s
            .as_deref()
            .map(|b| String::from_utf8_lossy(b).into_owned());
        let a3_s = grp
            .acqu3s
            .as_deref()
            .map(|b| String::from_utf8_lossy(b).into_owned());
        let a4_s = grp
            .acqu4s
            .as_deref()
            .map(|b| String::from_utf8_lossy(b).into_owned());
        let pp_s = grp
            .pulseprogram
            .as_deref()
            .map(|b| String::from_utf8_lossy(b).into_owned());
        let nl_s = grp
            .nuslist
            .as_deref()
            .map(|b| String::from_utf8_lossy(b).into_owned());
        let vd_s = grp
            .vdlist
            .as_deref()
            .map(|b| String::from_utf8_lossy(b).into_owned());
        let vc_s = grp
            .vclist
            .as_deref()
            .map(|b| String::from_utf8_lossy(b).into_owned());
        let fq_s = grp
            .fqlist
            .as_deref()
            .map(|b| String::from_utf8_lossy(b).into_owned());
        let df_s = grp
            .difflist
            .as_deref()
            .map(|b| String::from_utf8_lossy(b).into_owned());
        let inp = crate::pulprog::BrukerExperimentInputs {
            acqus_text: &a1_s,
            acqu2s_text: a2_s.as_deref(),
            acqu3s_text: a3_s.as_deref(),
            acqu4s_text: a4_s.as_deref(),
            pulseprogram_text: pp_s.as_deref(),
            nuslist_text: nl_s.as_deref(),
            vdlist_text: vd_s.as_deref(),
            vclist_text: vc_s.as_deref(),
            fqlist_text: fq_s.as_deref(),
            difflist_text: df_s.as_deref(),
            local_includes: grp.local_includes.clone(),
        };
        crate::pulprog::interpret_bruker_experiment(&inp).ok()
    } else {
        None
    };

    // Upgrade indirect axes whose manual `ip*`/`dp*`/`igrad` loops synthesize hypercomplex quadrature under `FnMODE = 0 / QF`
    if let Some(ref pi) = pulprog_interp {
        for (ind_idx, ind_phys) in pi.indirect_dims.iter().enumerate() {
            let ax_idx = ind_idx + 1;
            if ax_idx >= axes.len() {
                break;
            }
            if matches!(
                ind_phys.quad_source,
                crate::pulprog::QuadSource::SynthesizedManualLoop { .. }
            ) {
                let upgraded = match ind_phys.quad_mode {
                    crate::pulprog::DecodedQuadMode::States => Some(QuadratureKind::States),
                    crate::pulprog::DecodedQuadMode::StatesTppi => Some(QuadratureKind::StatesTppi),
                    crate::pulprog::DecodedQuadMode::EchoAntiecho => {
                        Some(QuadratureKind::EchoAntiecho)
                    }
                    _ => None,
                };
                if let Some(q) = upgraded {
                    axes[ax_idx].quadrature = q;
                    if axes[ax_idx].nucleus == "Delay" || axes[ax_idx].nucleus.is_empty() {
                        axes[ax_idx].nucleus = ind_phys.nucleus.clone();
                    }
                }
            }
        }
    }

    let is_dosy = diff_len > 0
        || pulprog_interp.as_ref().is_some_and(|pi| {
            pi.transfer_graph.deduced_family == crate::pulprog::PulprogSequenceFamily::Dosy
        })
        || pulprog.as_deref().is_some_and(|s| {
            let low = s.to_ascii_lowercase();
            low.starts_with("dstebpgp")
                || low.starts_with("ledbpgp")
                || low.contains("dosy")
                || (low.contains("ste") && low.contains("gp") && has_qf_or_unknown_indirect)
        });

    let d3_is_qf = ndim_raw == 3
        && axes
            .get(2)
            .is_some_and(|a| a.quadrature == QuadratureKind::Qf);
    let d2_inner_is_qf_in_3d = ndim_raw == 3
        && axes
            .get(1)
            .is_some_and(|a| a.quadrature == QuadratureKind::Qf)
        && axes
            .get(2)
            .is_some_and(|a| a.quadrature != QuadratureKind::Qf);
    let any_indirect_qf_matches = |sched_len: usize| -> bool {
        sched_len > 0
            && axes
                .iter()
                .skip(1)
                .any(|a| a.quadrature == QuadratureKind::Qf && a.points == sched_len)
    };
    let mut notes = Vec::new();
    if let Some(ref pi) = pulprog_interp {
        if pi.aqseq == crate::pulprog::AcquisitionOrder::Aqseq312 {
            notes.push("AQSEQ=312 transposed indirect acquisition order".to_string());
        }
        match pi.pseudo_3d_order {
            crate::pulprog::Pseudo3DStorageOrder::T1Interleaved => {
                notes.push("pseudo-3D t1-interleaved storage".to_string());
            }
            crate::pulprog::Pseudo3DStorageOrder::PlaneQueued if !is_dosy => {
                notes.push("pseudo-3D plane-queued storage".to_string());
            }
            _ => {}
        }
    } else if jcamp_usize(&a1, "AQSEQ") == Some(1) && ndim_raw == 3 {
        notes.push("AQSEQ=312 transposed indirect acquisition order".to_string());
    }

    let mut pseudo_kind: Option<&'static str> = None;
    let dimensionality = if is_dosy {
        notes.push("DOSY diffusion (Stejskal-Tanner / BPP-LED, QF)".to_string());
        match ndim_raw {
            1 => ExperimentDimensionality::Dim1D,
            2 => ExperimentDimensionality::Dim2D,
            _ => ExperimentDimensionality::Pseudo3D,
        }
    } else if vd_len > 1 || (vd_len > 0 && (d3_is_qf || any_indirect_qf_matches(vd_len))) {
        pseudo_kind = Some("vdlist");
        notes.push(format!("pseudo-3D variable delay list ({vd_len} planes)"));
        ExperimentDimensionality::Pseudo3D
    } else if vc_len > 1 || (vc_len > 0 && (d3_is_qf || any_indirect_qf_matches(vc_len))) {
        pseudo_kind = Some("vclist");
        notes.push(format!("pseudo-3D variable counter list ({vc_len} planes)"));
        ExperimentDimensionality::Pseudo3D
    } else if fq_len >= 4 && (ndim_raw == 2 || d3_is_qf || any_indirect_qf_matches(fq_len)) {
        pseudo_kind = Some("fqlist");
        notes.push(format!(
            "pseudo-3D frequency offset list ({fq_len} offsets)"
        ));
        ExperimentDimensionality::Pseudo3D
    } else if d3_is_qf || d2_inner_is_qf_in_3d {
        pseudo_kind = Some("qf_planes");
        notes.push("pseudo-3D QF plane stack".to_string());
        ExperimentDimensionality::Pseudo3D
    } else {
        match ndim_raw {
            1 => ExperimentDimensionality::Dim1D,
            2 => ExperimentDimensionality::Dim2D,
            3 => ExperimentDimensionality::Dim3D,
            _ => ExperimentDimensionality::Dim4D,
        }
    };

    // NUS detection: strictly `FnTYPE == 2` or `nuslist` present (avoids TopSpin `FnTYPE=0` template false positives!)
    let fn_type = jcamp_usize(&a1, "FnTYPE").unwrap_or(0);
    let nuslist_count = grp
        .nuslist
        .as_deref()
        .map(count_nonempty_schedule_lines)
        .unwrap_or(0);
    let is_nus = fn_type == 2 || nuslist_count > 0;
    let nus = if is_nus {
        let mut full_grid = 1usize;
        let mut have_grid = false;
        for (idx, amap) in [(1usize, &a2), (2usize, &a3), (3usize, &a4)] {
            if idx >= axes.len() {
                break;
            }
            if axes[idx].quadrature == QuadratureKind::Qf {
                continue;
            }
            let nustd = jcamp_usize(amap, "NusTD")
                .filter(|&v| v > 1)
                .unwrap_or(axes[idx].points);
            if nustd > 1 {
                let complex_n = (nustd / 2).max(1);
                full_grid = full_grid.saturating_mul(complex_n);
                have_grid = true;
            }
        }
        let sampled = if nuslist_count > 0 {
            Some(nuslist_count)
        } else if ndim_raw == 2 {
            jcamp_usize(&a2, "TD").map(|td| (td / 2).max(1))
        } else {
            None
        };
        let density = match (sampled, have_grid.then_some(full_grid)) {
            (Some(s), Some(fg)) if fg > 0 => Some((s as f64 / fg as f64).clamp(0.0, 1.0)),
            _ => jcamp_f64(&a1, "NusAMOUNT").map(|pct| (pct / 100.0).clamp(0.0, 1.0)),
        };
        let badge = match density {
            Some(d) => format!("NUS {:.1}%", d * 100.0),
            None => "NUS".to_string(),
        };
        NusSummary {
            is_nus: true,
            sampled_points: sampled,
            full_grid_points: have_grid.then_some(full_grid),
            sampling_density: density,
            badge,
        }
    } else {
        NusSummary::default()
    };

    // Pulse-sequence classification:
    // Priority 1: `PULPROG` + axis nuclei/dimensionality (blocked on DOSY diffusion!)
    // Priority 2: Symbolic `nmr-core::pulprog` `MagnetizationTransferGraph` (executed `pulseprogram` RF/CNST/transfer topology!)
    // Priority 3: `title` / folder name (ONLY when `dimensionality != 1D` and `PULPROG` + transfer graph were absent/generic)
    // Priority 4: Pure axis-physics `(nuclei, carrier_ppm, sw_ppm)`
    let disp = display_name_from_rel(rel_root, root_name);
    let mut pulse_class = if is_dosy {
        PulseSequenceClass::Unknown
    } else {
        pulprog
            .as_deref()
            .and_then(|pp| classify_pulse_token(pp, dimensionality, &axes))
            .unwrap_or(PulseSequenceClass::Unknown)
    };

    if !is_dosy
        && (pulse_class == PulseSequenceClass::Unknown
            || (dimensionality == ExperimentDimensionality::Pseudo3D && !pulse_class.is_dynamics()))
    {
        if let Some(ref pi) = pulprog_interp {
            let c = pi.transfer_graph.deduced_family.to_pulse_sequence_class();
            if c != PulseSequenceClass::Unknown
                && (dimensionality != ExperimentDimensionality::Pseudo3D || c.is_dynamics())
            {
                pulse_class = c;
            }
        }
    }

    if !is_dosy
        && pulse_class == PulseSequenceClass::Unknown
        && dimensionality != ExperimentDimensionality::Dim1D
    {
        if let Some(t) = title.as_deref() {
            if let Some(c) = classify_pulse_token(t, dimensionality, &axes) {
                if c != PulseSequenceClass::Unknown {
                    pulse_class = c;
                }
            }
        }
    }
    if !is_dosy
        && pulse_class == PulseSequenceClass::Unknown
        && dimensionality != ExperimentDimensionality::Dim1D
    {
        if let Some(c) = classify_pulse_token(&disp, dimensionality, &axes) {
            if c != PulseSequenceClass::Unknown {
                pulse_class = c;
            }
        }
    }
    let is_known_specialized_j = pulprog.as_deref().is_some_and(|s| {
        let sl = s.to_ascii_lowercase();
        sl == "ghnha" || sl == "hnha" || sl.starts_with("hnhagp")
    });
    if is_known_specialized_j {
        notes.push("3D HNHA scalar-coupling experiment".to_string());
    }
    if !is_dosy
        && ((!is_known_specialized_j && pulse_class == PulseSequenceClass::Unknown)
            || (dimensionality == ExperimentDimensionality::Pseudo3D && !pulse_class.is_dynamics()))
    {
        let phys = infer_pulse_class_from_axis_physics(dimensionality, &axes, pseudo_kind);
        if phys != PulseSequenceClass::Unknown {
            pulse_class = phys;
        }
    }

    let data_state = match (grp.raw_bytes > 0, grp.processed_bytes > 0) {
        (true, true) => DataState::Both,
        (false, true) => DataState::Processed,
        (true, false) => DataState::RawFid,
        (false, false) => {
            if grp.acqus.is_some() {
                notes.push("header-only (no binary fid/ser payload on disk)".to_string());
                DataState::RawFid
            } else {
                DataState::Processed
            }
        }
    };

    let quadrature_badge = format_quadrature_badge(&axes);
    let selected_by_default = pulse_class != PulseSequenceClass::Unknown
        && dimensionality != ExperimentDimensionality::Dim1D;

    Some(ClassifiedExperiment {
        relative_path: rel_root.to_string(),
        display_name: disp,
        vendor: ExperimentVendor::Bruker,
        dimensionality,
        data_state,
        spectrometer_mhz: base_mhz,
        axes,
        quadrature_badge,
        nus,
        pulse_class,
        pulse_program: pulprog,
        title,
        raw_signal_bytes: grp.raw_bytes,
        processed_bytes: grp.processed_bytes,
        selected_by_default,
        notes,
    })
}

fn build_varian_experiment(
    rel_root: &str,
    root_name: &str,
    grp: &VarianFolderGroup,
) -> Option<ClassifiedExperiment> {
    let procpar_bytes = grp.procpar.as_deref()?;
    let map = parse_varian_procpar_map(procpar_bytes);
    if map.is_empty() {
        return None;
    }

    let tn = canonical_nucleus(&procpar_str(&map, "tn").unwrap_or_else(|| "H1".to_string()));
    let dn = canonical_nucleus(&procpar_str(&map, "dn").unwrap_or_default());
    let dn2 = canonical_nucleus(&procpar_str(&map, "dn2").unwrap_or_default());
    let dn3 = canonical_nucleus(&procpar_str(&map, "dn3").unwrap_or_default());

    let sfrq = procpar_f64(&map, "sfrq");
    let dfrq = procpar_f64(&map, "dfrq");
    let dfrq2 = procpar_f64(&map, "dfrq2");
    let dfrq3 = procpar_f64(&map, "dfrq3");

    let np = procpar_usize(&map, "np").unwrap_or(0);
    let ni = procpar_usize(&map, "ni").unwrap_or(0);
    let ni2 = procpar_usize(&map, "ni2").unwrap_or(0);
    let ni3 = procpar_usize(&map, "ni3").unwrap_or(0);

    let seqfil = procpar_str(&map, "seqfil").or_else(|| procpar_str(&map, "pslabel"));
    let pslabel = procpar_str(&map, "pslabel");
    let array_str = procpar_str(&map, "array").unwrap_or_default();

    let title = grp
        .text
        .as_deref()
        .map(|b| {
            String::from_utf8_lossy(b)
                .lines()
                .map(str::trim)
                .filter(|l| !l.is_empty())
                .collect::<Vec<_>>()
                .join(" | ")
        })
        .filter(|s| !s.is_empty())
        .or_else(|| procpar_str(&map, "comment"));

    // Determine indirect axis channel assignment (`refsource1`, `refsource2`, `seqfil`, or `sw` physics).
    // Note: in classic Varian VNMR, `refsource1` and `refsource2` both default to `"sfrq"` unless
    // explicitly re-referenced; we therefore trust `"dfrq"`/`"dfrq2"`/`"dfrq3"` directly and use
    // pulse-program + sweep-width physics when `refsource*` is `"sfrq"` or unset.
    let resolve_varian_channel =
        |refsource_key: &str, sw_hz: Option<f64>, default_slot: u8| -> (String, Option<f64>) {
            if let Some(rs) = procpar_str(&map, refsource_key) {
                match rs.as_str() {
                    "dfrq" if !dn.is_empty() => return (dn.clone(), dfrq),
                    "dfrq2" if !dn2.is_empty() => return (dn2.clone(), dfrq2),
                    "dfrq3" if !dn3.is_empty() => return (dn3.clone(), dfrq3),
                    _ => {}
                }
            }
            let sf_low = seqfil.as_deref().unwrap_or("").to_ascii_lowercase();
            // Homonuclear `1H` indirect axes in 3D/4D Varian BioPack/ProteinPack experiments
            if default_slot == 1
                && ni2 > 1
                && (sf_low.contains("gnoesynhsqc")
                    || sf_low.contains("n15_noesy")
                    || sf_low.contains("ghc_co_nh")
                    || sf_low.contains("hccconh")
                    || sf_low.contains("hcconh")
                    || sf_low.contains("ghnha")
                    || sf_low == "hnha")
            {
                return (tn.clone(), sfrq);
            }
            if default_slot == 2 && sf_low.contains("hcchtocsy") {
                return (tn.clone(), sfrq);
            }
            if default_slot == 2
                && (sf_low.contains("noesyc")
                    || sf_low.contains("cnoesy")
                    || sf_low.contains("c4dnoesy"))
            {
                return (tn.clone(), sfrq);
            }
            // Classic VNMR 2D: disambiguate `15N` (`dn2`) vs `13C` (`dn`)
            if default_slot == 1 && ni2 <= 1 && !dn.is_empty() && !dn2.is_empty() {
                if sf_low.contains("nhsqc")
                    || sf_low.contains("n15")
                    || sf_low.contains("15n")
                    || sf_low.contains("gt1")
                    || sf_low.contains("gt2")
                    || sf_low.contains("gnoesy")
                {
                    return (dn2.clone(), dfrq2);
                }
                if sf_low.contains("chsqc")
                    || sf_low.contains("c13")
                    || sf_low.contains("13c")
                    || sf_low.contains("hmqc")
                {
                    return (dn.clone(), dfrq);
                }
                if let (Some(sw), Some(df2)) = (sw_hz, dfrq2) {
                    if df2 > 10.0 && (15.0..=45.0).contains(&(sw / df2)) {
                        return (dn2.clone(), dfrq2);
                    }
                }
            }
            match default_slot {
                1 => {
                    if !dn.is_empty() {
                        (dn.clone(), dfrq)
                    } else {
                        (dn2.clone(), dfrq2)
                    }
                }
                2 => {
                    if !dn2.is_empty() {
                        (dn2.clone(), dfrq2)
                    } else {
                        (dn.clone(), dfrq)
                    }
                }
                _ => {
                    if !dn3.is_empty() {
                        (dn3.clone(), dfrq3)
                    } else {
                        (tn.clone(), sfrq)
                    }
                }
            }
        };

    let sw0 = procpar_f64(&map, "sw");
    let sw1 = procpar_f64(&map, "sw1");
    let sw2 = procpar_f64(&map, "sw2");
    let sw3 = procpar_f64(&map, "sw3");

    let mut axes = Vec::new();
    axes.push(ClassifiedAxis {
        label: "t2".to_string(),
        nucleus: if tn.is_empty() {
            "1H".to_string()
        } else {
            tn.clone()
        },
        points: (np / 2).max(np),
        sw_ppm: match (sw0, sfrq) {
            (Some(sw), Some(sf)) if sf > 1e-3 => Some(sw / sf),
            _ => None,
        },
        carrier_ppm: match (
            procpar_f64(&map, "rfl"),
            procpar_f64(&map, "rfp"),
            sw0,
            sfrq,
        ) {
            (Some(rfl), Some(rfp), Some(sw), Some(sf)) if sf > 1e-3 => {
                Some(((sw - rfl + rfp) - 0.5 * sw) / sf)
            }
            _ => None,
        },
        obs_mhz: sfrq,
        quadrature: QuadratureKind::Dqd,
    });

    let varian_carrier =
        |rfl_key: &str, rfp_key: &str, sw_hz: Option<f64>, obs_mhz: Option<f64>| -> Option<f64> {
            match (
                procpar_f64(&map, rfl_key),
                procpar_f64(&map, rfp_key),
                sw_hz,
                obs_mhz,
            ) {
                (Some(rfl), Some(rfp), Some(sw), Some(sf)) if sf > 1e-3 => {
                    Some(((sw - rfl + rfp) - 0.5 * sw) / sf)
                }
                _ => None,
            }
        };

    if ni > 1 {
        let (nuc1, obs1) = resolve_varian_channel("refsource1", sw1, 1);
        axes.push(ClassifiedAxis {
            label: "t1".to_string(),
            nucleus: nuc1,
            points: ni,
            sw_ppm: match (sw1, obs1) {
                (Some(sw), Some(sf)) if sf > 1e-3 => Some(sw / sf),
                _ => None,
            },
            carrier_ppm: varian_carrier("rfl1", "rfp1", sw1, obs1),
            obs_mhz: obs1,
            quadrature: varian_indirect_quadrature(&map, "phase", "f1coef"),
        });
    }
    if ni2 > 1 {
        let (nuc2, obs2) = resolve_varian_channel("refsource2", sw2, 2);
        axes.push(ClassifiedAxis {
            label: "t2_ind".to_string(),
            nucleus: nuc2,
            points: ni2,
            sw_ppm: match (sw2, obs2) {
                (Some(sw), Some(sf)) if sf > 1e-3 => Some(sw / sf),
                _ => None,
            },
            carrier_ppm: varian_carrier("rfl2", "rfp2", sw2, obs2),
            obs_mhz: obs2,
            quadrature: varian_indirect_quadrature(&map, "phase2", "f2coef"),
        });
    }
    if ni3 > 1 {
        let (nuc3, obs3) = resolve_varian_channel("refsource3", sw3, 3);
        axes.push(ClassifiedAxis {
            label: "t3_ind".to_string(),
            nucleus: nuc3,
            points: ni3,
            sw_ppm: match (sw3, obs3) {
                (Some(sw), Some(sf)) if sf > 1e-3 => Some(sw / sf),
                _ => None,
            },
            carrier_ppm: varian_carrier("rfl3", "rfp3", sw3, obs3),
            obs_mhz: obs3,
            quadrature: varian_indirect_quadrature(&map, "phase3", "f3coef"),
        });
    }

    // Check Varian `array` for non-phase arrayed relaxation/CPMG/CEST/DOSY (`d2`, `relaxT`, `ncyc`, `satfrq`, `gzlvl1`, etc.)
    let non_phase_array: Vec<&str> = array_str
        .split([',', ';', ' '])
        .map(str::trim)
        .filter(|tok| {
            !tok.is_empty() && !tok.starts_with("phase") && !tok.eq_ignore_ascii_case("izar")
        })
        .collect();
    let is_varian_dosy = non_phase_array
        .iter()
        .any(|v| v.to_ascii_lowercase().starts_with("gzlvl"))
        || map.get("gzlvl1").is_some_and(|v| v.len() >= 4)
        || seqfil.as_deref().is_some_and(|s| {
            let sl = s.to_ascii_lowercase();
            sl.contains("dosy") || sl.contains("bppste") || sl.starts_with("dbpp")
        });

    let mut notes = Vec::new();
    let mut pseudo_kind: Option<&'static str> = None;
    let dimensionality = if is_varian_dosy {
        notes.push("DOSY diffusion (arrayed gzlvl1, QF)".to_string());
        if axes.len() <= 2 {
            ExperimentDimensionality::Dim2D
        } else {
            ExperimentDimensionality::Pseudo3D
        }
    } else if !non_phase_array.is_empty() && axes.len() == 2 {
        let var = non_phase_array[0];
        let arr_len = map.get(var).map(|v| v.len()).unwrap_or(0);
        let low_var = var.to_ascii_lowercase();
        if low_var.contains("ncyc") || low_var.contains("cpmg") {
            pseudo_kind = Some("vclist");
        } else if low_var.contains("sat") || low_var.contains("frq") || low_var.contains("offset") {
            pseudo_kind = Some("fqlist");
        } else {
            pseudo_kind = Some("vdlist");
        }
        notes.push(format!("pseudo-3D Varian array={var} ({arr_len} planes)"));
        // Record `T1Interleaved` vs `PlaneQueued` from Varian `array` token order (`d2,relaxT,phase` vs `relaxT,phase`)
        let arr_tokens: Vec<&str> = array_str
            .split([',', ';', ' '])
            .map(str::trim)
            .filter(|t| !t.is_empty())
            .collect();
        let d2_pos = arr_tokens.iter().position(|t| t.eq_ignore_ascii_case("d2"));
        let var_pos = arr_tokens.iter().position(|t| *t == var);
        if let (Some(dp), Some(vp)) = (d2_pos, var_pos) {
            if dp < vp {
                notes.push("pseudo-3D t1-interleaved storage".to_string());
            } else {
                notes.push("pseudo-3D plane-queued storage".to_string());
            }
        } else {
            notes.push("pseudo-3D plane-queued storage".to_string());
        }
        ExperimentDimensionality::Pseudo3D
    } else {
        match axes.len() {
            1 => ExperimentDimensionality::Dim1D,
            2 => ExperimentDimensionality::Dim2D,
            3 => ExperimentDimensionality::Dim3D,
            _ => ExperimentDimensionality::Dim4D,
        }
    };

    // Varian NUS (`sparse == "y"` or `sampling == "sparse"` or `sampling.sch` present)
    let sparse_flag = procpar_str(&map, "sparse")
        .map(|s| s.eq_ignore_ascii_case("y"))
        .unwrap_or(false)
        || procpar_str(&map, "sampling")
            .map(|s| s.eq_ignore_ascii_case("sparse") || s.eq_ignore_ascii_case("nus"))
            .unwrap_or(false);
    let sch_lines = grp
        .sampling_sch
        .as_deref()
        .map(count_nonempty_schedule_lines)
        .unwrap_or(0);
    let nus = if sparse_flag || sch_lines > 0 {
        let mut full_grid = 1usize;
        for ax in axes.iter().skip(1) {
            full_grid = full_grid.saturating_mul(ax.points.max(1));
        }
        let sampled = (sch_lines > 0)
            .then_some(sch_lines)
            .or_else(|| procpar_usize(&map, "nimax"));
        let density = sampled
            .filter(|_| full_grid > 1)
            .map(|s| (s as f64 / full_grid as f64).clamp(0.0, 1.0));
        let badge = match density {
            Some(d) => format!("NUS {:.1}%", d * 100.0),
            None => "NUS".to_string(),
        };
        NusSummary {
            is_nus: true,
            sampled_points: sampled,
            full_grid_points: (full_grid > 1).then_some(full_grid),
            sampling_density: density,
            badge,
        }
    } else {
        NusSummary::default()
    };

    let base_mhz = procpar_f64(&map, "h1freq")
        .filter(|&v| v > 40.0)
        .or_else(|| sfrq.and_then(|sf| infer_proton_base_mhz(&axes[0].nucleus, sf)));

    let disp = display_name_from_rel(rel_root, root_name);
    let mut pulse_class = if is_varian_dosy {
        PulseSequenceClass::Unknown
    } else {
        seqfil
            .as_deref()
            .and_then(|s| classify_pulse_token(s, dimensionality, &axes))
            .unwrap_or(PulseSequenceClass::Unknown)
    };
    if !is_varian_dosy && pulse_class == PulseSequenceClass::Unknown {
        if let Some(ps) = pslabel.as_deref() {
            if let Some(c) = classify_pulse_token(ps, dimensionality, &axes) {
                pulse_class = c;
            }
        }
    }
    if !is_varian_dosy && pulse_class == PulseSequenceClass::Unknown {
        let pp_txt = String::from_utf8_lossy(procpar_bytes);
        let sch_txt = grp
            .sampling_sch
            .as_deref()
            .map(|b| String::from_utf8_lossy(b).into_owned());
        if let Ok(vi) = crate::pulprog::interpret_varian_procpar(&pp_txt, sch_txt.as_deref()) {
            let c = vi.transfer_graph.deduced_family.to_pulse_sequence_class();
            if c != PulseSequenceClass::Unknown
                && (dimensionality != ExperimentDimensionality::Pseudo3D || c.is_dynamics())
            {
                pulse_class = c;
            }
        }
    }
    if !is_varian_dosy
        && pulse_class == PulseSequenceClass::Unknown
        && dimensionality != ExperimentDimensionality::Dim1D
    {
        if let Some(c) = classify_pulse_token(&disp, dimensionality, &axes) {
            pulse_class = c;
        }
    }
    let is_known_specialized_j = seqfil.as_deref().is_some_and(|s| {
        let sl = s.to_ascii_lowercase();
        sl == "ghnha" || sl == "hnha" || sl.starts_with("hnhagp")
    });
    if is_known_specialized_j {
        notes.push("3D HNHA scalar-coupling experiment".to_string());
    }
    if !is_varian_dosy
        && ((!is_known_specialized_j && pulse_class == PulseSequenceClass::Unknown)
            || (dimensionality == ExperimentDimensionality::Pseudo3D && !pulse_class.is_dynamics()))
    {
        let phys = infer_pulse_class_from_axis_physics(dimensionality, &axes, pseudo_kind);
        if phys != PulseSequenceClass::Unknown {
            pulse_class = phys;
        }
    }

    let data_state = match (grp.raw_bytes > 0, grp.processed_bytes > 0) {
        (true, true) => DataState::Both,
        (false, true) => DataState::Processed,
        _ => DataState::RawFid,
    };

    let quadrature_badge = format_quadrature_badge(&axes);
    let selected_by_default = pulse_class != PulseSequenceClass::Unknown
        && dimensionality != ExperimentDimensionality::Dim1D;

    Some(ClassifiedExperiment {
        relative_path: rel_root.to_string(),
        display_name: disp,
        vendor: ExperimentVendor::Varian,
        dimensionality,
        data_state,
        spectrometer_mhz: base_mhz,
        axes,
        quadrature_badge,
        nus,
        pulse_class,
        pulse_program: seqfil,
        title,
        raw_signal_bytes: grp.raw_bytes,
        processed_bytes: grp.processed_bytes,
        selected_by_default,
        notes,
    })
}

fn has_titration_marker(s: &str) -> bool {
    let low = s.to_ascii_lowercase();
    low.contains("titrat")
        || low.contains("csp")
        || low.contains("equiv")
        || low.contains("0eq")
        || low.contains("1eq")
        || low.contains("2eq")
        || low.contains("5eq")
        || low.contains("0.5eq")
        || low.contains("ligand")
        || low.contains("0um")
        || low.contains("50um")
        || low.contains("100um")
        || low.contains("200um")
        || low.contains("500um")
}

fn infer_regime_and_hints(
    experiments: &mut [ClassifiedExperiment],
    aux_files: &[AuxiliaryFileRecord],
) -> (ProblemRegime, Vec<ProblemRegime>, Vec<String>) {
    // Promote explicit 2D HSQC/HMQC titration series (`>= 3` titration-labeled 2D spectra)
    let titrat_indices: Vec<usize> = experiments
        .iter()
        .enumerate()
        .filter(|(_, e)| {
            e.dimensionality == ExperimentDimensionality::Dim2D
                && matches!(
                    e.pulse_class,
                    PulseSequenceClass::N15Hsqc
                        | PulseSequenceClass::C13Hsqc
                        | PulseSequenceClass::C13Hmqc
                        | PulseSequenceClass::TitrationSeries
                )
                && (has_titration_marker(&e.relative_path)
                    || e.title.as_deref().is_some_and(has_titration_marker))
        })
        .map(|(i, _)| i)
        .collect();

    let has_3d_backbone = experiments
        .iter()
        .any(|e| e.pulse_class.is_1h_backbone_triple());

    if titrat_indices.len() >= 3 && !has_3d_backbone {
        for idx in &titrat_indices {
            experiments[*idx].pulse_class = PulseSequenceClass::TitrationSeries;
        }
    }

    let has_paramag = experiments.iter().any(|e| {
        let p_low = e.relative_path.to_ascii_lowercase();
        let t_low = e.title.as_deref().unwrap_or("").to_ascii_lowercase();
        p_low.contains("paramag")
            || t_low.contains("paramag")
            || p_low.contains("_pcs")
            || t_low.contains("pseudocontact")
            || p_low.contains("_pre_")
            || e.axes
                .iter()
                .any(|a| a.nucleus == "1H" && a.sw_ppm.unwrap_or(0.0) > 28.0)
    });

    let count_1h_triple = experiments
        .iter()
        .filter(|e| e.pulse_class.is_1h_backbone_triple())
        .count();
    let count_13c_det = experiments
        .iter()
        .filter(|e| e.pulse_class.is_13c_detected())
        .count();
    let count_sidechain_noesy = experiments
        .iter()
        .filter(|e| e.pulse_class.is_sidechain_or_noesy())
        .count();
    let count_methyl_hmqc = experiments
        .iter()
        .filter(|e| e.pulse_class == PulseSequenceClass::C13Hmqc)
        .count();
    let count_dynamics = experiments
        .iter()
        .filter(|e| e.pulse_class.is_dynamics())
        .count();
    let count_titration = experiments
        .iter()
        .filter(|e| e.pulse_class == PulseSequenceClass::TitrationSeries)
        .count()
        .max(if titrat_indices.len() >= 3 {
            titrat_indices.len()
        } else {
            0
        });

    let mut candidates = Vec::new();
    if has_paramag {
        candidates.push(ProblemRegime::Paramagnetic);
    }
    if count_1h_triple > 0 && count_sidechain_noesy >= 2 {
        candidates.push(ProblemRegime::SidechainNoesy);
    }
    if count_1h_triple > 0 {
        candidates.push(ProblemRegime::GlobularBackbone);
    }
    if count_13c_det > 0 {
        candidates.push(ProblemRegime::Idp13cDetected);
    }
    if count_methyl_hmqc > 0 {
        candidates.push(ProblemRegime::MethylTrosy);
    }
    if count_dynamics > 0 {
        candidates.push(ProblemRegime::RelaxationDynamics);
    }
    if count_titration >= 3 {
        candidates.push(ProblemRegime::TitrationCsp);
    }

    let primary = candidates
        .first()
        .copied()
        .unwrap_or(ProblemRegime::GlobularBackbone);
    let secondary: Vec<ProblemRegime> = candidates.into_iter().skip(1).collect();

    // Build actionable missing-input hints
    let mut hints = Vec::new();
    let has_seq = !aux_files.is_empty();
    if !has_seq {
        hints.push(
            "No polymer sequence (.fasta, .str, or .nef) detected; paste or load sequence in Stage 1."
                .to_string(),
        );
    }
    let has_n15_hsqc = experiments
        .iter()
        .any(|e| e.pulse_class == PulseSequenceClass::N15Hsqc);
    if count_1h_triple > 0 && !has_n15_hsqc {
        hints.push(
            "3D amide triple-resonance spectra found without a 2D 15N-HSQC root spectrum."
                .to_string(),
        );
    }
    let has_cb = experiments.iter().any(|e| {
        matches!(
            e.pulse_class,
            PulseSequenceClass::Hncacb | PulseSequenceClass::Cbcaconh
        )
    });
    if count_1h_triple > 0 && !has_cb {
        hints.push(
            "Sparse CA/CO-only backbone suite (no HNCACB or CBCA(CO)NH detected); amino-acid CB typing unavailable."
                .to_string(),
        );
    }
    let has_con_2d = experiments
        .iter()
        .any(|e| e.pulse_class == PulseSequenceClass::Con2D);
    let has_13c_3d = experiments.iter().any(|e| {
        matches!(
            e.pulse_class,
            PulseSequenceClass::Canco3D | PulseSequenceClass::Cbcaco3D
        )
    });
    if has_13c_3d && !has_con_2d {
        hints.push(
            "3D 13C-detected CANCO/CBCACO spectra found without a 2D CON root spectrum."
                .to_string(),
        );
    }

    (primary, secondary, hints)
}

fn expand_inline_archives(entries: &[VirtualEntry]) -> Vec<VirtualEntry> {
    let has_unpacked = entries.iter().any(|e| {
        let low = e.relative_path.to_ascii_lowercase();
        !is_tar_path_lower(&low) && !is_targz_path_lower(&low)
    });
    if has_unpacked {
        return entries.to_vec();
    }
    let mut expanded = Vec::new();
    for e in entries {
        let low = e.relative_path.to_ascii_lowercase();
        if (is_tar_path_lower(&low) || is_targz_path_lower(&low)) && !e.header_bytes.is_empty() {
            if let Ok(sub) = extract_archive_virtual_entries(&e.relative_path, &e.header_bytes) {
                expanded.extend(sub);
                continue;
            }
        }
        expanded.push(e.clone());
    }
    expanded
}

/// Classify an in-memory slice of `VirtualEntry` records (`wasm32`-compatible, zero disk I/O).
pub fn classify_nmr_entries(entries: &[VirtualEntry]) -> WorkspaceInventory {
    classify_nmr_entries_named("workspace", entries)
}

fn classify_nmr_entries_named(root_name: &str, raw_entries: &[VirtualEntry]) -> WorkspaceInventory {
    let (
        (experiments, auxiliary_files, problem_regime, secondary_regimes, missing_input_hints),
        elapsed_ms,
    ) = measure_ms(|| {
        let expanded = expand_inline_archives(raw_entries);

        let mut bruker_groups: BTreeMap<String, BrukerFolderGroup> = BTreeMap::new();
        let mut varian_groups: BTreeMap<String, VarianFolderGroup> = BTreeMap::new();

        // First pass: discover Bruker & Varian experiment roots
        for e in &expanded {
            let rel = normalize_rel_path(&e.relative_path);
            if rel.is_empty() || is_noise_path(&rel) {
                continue;
            }
            if let Some((vroot, fname)) = find_varian_root_for_path(&rel) {
                let grp = varian_groups.entry(vroot).or_default();
                match fname.to_ascii_lowercase().as_str() {
                    "procpar" => grp.procpar = Some(e.header_bytes.clone()),
                    "sampling.sch" => grp.sampling_sch = Some(e.header_bytes.clone()),
                    _ => {}
                }
                continue;
            }
            if let Some((broot, fname)) = find_bruker_root_for_path(&rel) {
                let grp = bruker_groups.entry(broot).or_default();
                match fname.to_ascii_lowercase().as_str() {
                    "acqus" => grp.acqus = Some(e.header_bytes.clone()),
                    "acqu" if grp.acqus.is_none() => grp.acqus = Some(e.header_bytes.clone()),
                    "acqu2s" => grp.acqu2s = Some(e.header_bytes.clone()),
                    "acqu2" if grp.acqu2s.is_none() => grp.acqu2s = Some(e.header_bytes.clone()),
                    "acqu3s" => grp.acqu3s = Some(e.header_bytes.clone()),
                    "acqu3" if grp.acqu3s.is_none() => grp.acqu3s = Some(e.header_bytes.clone()),
                    "acqu4s" => grp.acqu4s = Some(e.header_bytes.clone()),
                    "acqu4" if grp.acqu4s.is_none() => grp.acqu4s = Some(e.header_bytes.clone()),
                    "procs" => grp.procs = Some(e.header_bytes.clone()),
                    "proc" if grp.procs.is_none() => grp.procs = Some(e.header_bytes.clone()),
                    "proc2s" => grp.proc2s = Some(e.header_bytes.clone()),
                    "proc2" if grp.proc2s.is_none() => grp.proc2s = Some(e.header_bytes.clone()),
                    "proc3s" => grp.proc3s = Some(e.header_bytes.clone()),
                    "proc3" if grp.proc3s.is_none() => grp.proc3s = Some(e.header_bytes.clone()),
                    "proc4s" => grp.proc4s = Some(e.header_bytes.clone()),
                    "proc4" if grp.proc4s.is_none() => grp.proc4s = Some(e.header_bytes.clone()),
                    "title" if grp.title.is_none() => grp.title = Some(e.header_bytes.clone()),
                    "pulseprogram" => grp.pulseprogram = Some(e.header_bytes.clone()),
                    "pulseprogram.precomp" | "pulprog" if grp.pulseprogram.is_none() => {
                        grp.pulseprogram = Some(e.header_bytes.clone())
                    }
                    "nuslist" => grp.nuslist = Some(e.header_bytes.clone()),
                    "difflist" => grp.difflist = Some(e.header_bytes.clone()),
                    "vdlist" => grp.vdlist = Some(e.header_bytes.clone()),
                    "vclist" | "valist" | "vplist" if grp.vclist.is_none() => {
                        grp.vclist = Some(e.header_bytes.clone())
                    }
                    "fqlist" | "fq1list" | "fq2list" | "fq3list" | "fq4list" | "fq5list"
                    | "fq6list" | "fq7list" | "fq8list" => {
                        let new_len = count_nonempty_schedule_lines(&e.header_bytes);
                        let cur_len = grp
                            .fqlist
                            .as_deref()
                            .map(count_nonempty_schedule_lines)
                            .unwrap_or(0);
                        if grp.fqlist.is_none() || new_len > cur_len {
                            grp.fqlist = Some(e.header_bytes.clone());
                        }
                    }
                    "ser" => grp.raw_bytes += e.effective_bytes(),
                    "1r" | "1i" | "2rr" | "2ri" | "2ir" | "2ii" | "3rrr" | "3rri" | "4rrrr" => {
                        grp.processed_bytes += e.effective_bytes()
                    }
                    _ => {}
                }
            }
        }

        // Retain only valid Bruker roots that actually carry `acqus` or `procs`
        bruker_groups.retain(|_, g| g.acqus.is_some() || g.procs.is_some());
        varian_groups.retain(|_, g| g.procpar.is_some());

        let vendor_roots: BTreeSet<String> = bruker_groups
            .keys()
            .chain(varian_groups.keys())
            .cloned()
            .collect();

        let find_enclosing_vendor = |rel: &str| -> Option<String> {
            if vendor_roots.contains(".") {
                return Some(".".to_string());
            }
            let mut cur = rel;
            while let Some((parent, _)) = cur.rsplit_once('/') {
                if vendor_roots.contains(parent) {
                    return Some(parent.to_string());
                }
                cur = parent;
            }
            None
        };

        let mut ucsf_entries: Vec<(String, Vec<u8>, u64)> = Vec::new();
        let mut pipe_series: BTreeMap<String, NmrPipeGroup> = BTreeMap::new();
        let mut pipe_singles: Vec<(String, Vec<u8>, u64)> = Vec::new();
        let mut aux_files: Vec<AuxiliaryFileRecord> = Vec::new();

        // Second pass: attach `fid`/`text`/`.ft*` inside vendor folders, and collect standalone `.ucsf`/`.ft*`/`.fasta`/`.str`/`.nef`
        for e in &expanded {
            let rel = normalize_rel_path(&e.relative_path);
            if rel.is_empty() || is_noise_path(&rel) {
                continue;
            }
            let fname = rel.rsplit('/').next().unwrap_or(&rel);
            let lower = fname.to_ascii_lowercase();

            // Sequence & STAR/NEF files anywhere in workspace
            if lower.ends_with(".fasta") || lower.ends_with(".faa") || lower.ends_with(".seq") {
                let txt = String::from_utf8_lossy(&e.header_bytes);
                let res_count: usize = txt
                    .lines()
                    .filter(|l| {
                        !l.trim_start().starts_with('>') && !l.trim_start().starts_with(';')
                    })
                    .flat_map(|l| l.chars())
                    .filter(|c| c.is_ascii_alphabetic())
                    .count();
                aux_files.push(AuxiliaryFileRecord {
                    relative_path: rel,
                    vendor: ExperimentVendor::SequenceFasta,
                    total_bytes: e.effective_bytes(),
                    summary: format!("FASTA sequence (~{res_count} aa)"),
                });
                continue;
            }
            if lower.ends_with(".str")
                || lower.ends_with(".star")
                || lower.ends_with(".bmrb")
                || lower.ends_with(".nef")
            {
                let kind = if lower.ends_with(".nef") {
                    "NEF v1.1 project/session"
                } else {
                    "NMR-STAR v3 entry"
                };
                aux_files.push(AuxiliaryFileRecord {
                    relative_path: rel,
                    vendor: ExperimentVendor::StarOrNef,
                    total_bytes: e.effective_bytes(),
                    summary: kind.to_string(),
                });
                continue;
            }

            if let Some(enclosing) = find_enclosing_vendor(&rel) {
                if let Some(vgrp) = varian_groups.get_mut(&enclosing) {
                    if lower == "fid" || lower == "origfid" {
                        vgrp.raw_bytes += e.effective_bytes();
                    } else if lower == "text" && vgrp.text.is_none() {
                        vgrp.text = Some(e.header_bytes.clone());
                    } else if lower.ends_with(".ft1")
                        || lower.ends_with(".ft2")
                        || lower.ends_with(".ft3")
                        || lower.ends_with(".ft4")
                    {
                        vgrp.processed_bytes += e.effective_bytes();
                    }
                    continue;
                }
                if let Some(bgrp) = bruker_groups.get_mut(&enclosing) {
                    if lower == "fid" || lower == "origfid" {
                        bgrp.raw_bytes += e.effective_bytes();
                    } else if lower.ends_with(".incl") {
                        let rel_in_exp = if enclosing == "." {
                            rel.clone()
                        } else {
                            rel.strip_prefix(&format!("{enclosing}/"))
                                .unwrap_or(&rel)
                                .to_string()
                        };
                        let inc_txt = String::from_utf8_lossy(&e.header_bytes).into_owned();
                        bgrp.local_includes.insert(rel_in_exp, inc_txt.clone());
                        bgrp.local_includes.insert(fname.to_string(), inc_txt);
                    } else if lower.ends_with(".ft1")
                        || lower.ends_with(".ft2")
                        || lower.ends_with(".ft3")
                        || lower.ends_with(".ft4")
                    {
                        bgrp.processed_bytes += e.effective_bytes();
                    }
                    continue;
                }
            }

            if lower.ends_with(".ucsf") {
                ucsf_entries.push((rel, e.header_bytes.clone(), e.effective_bytes()));
                continue;
            }

            if lower.ends_with(".ft1")
                || lower.ends_with(".ft2")
                || lower.ends_with(".ft3")
                || lower.ends_with(".ft4")
                || lower.ends_with(".ft")
            {
                if let Some((series_key, stem_name)) = split_nmrpipe_series_key(&rel) {
                    let entry =
                        pipe_series
                            .entry(series_key.clone())
                            .or_insert_with(|| NmrPipeGroup {
                                representative_path: series_key,
                                stem_name,
                                header_bytes: e.header_bytes.clone(),
                                plane_count: 0,
                                total_bytes: 0,
                            });
                    if entry.header_bytes.is_empty() && !e.header_bytes.is_empty() {
                        entry.header_bytes = e.header_bytes.clone();
                    }
                    entry.plane_count += 1;
                    entry.total_bytes += e.effective_bytes();
                } else {
                    pipe_singles.push((rel, e.header_bytes.clone(), e.effective_bytes()));
                }
            }
        }

        let mut experiments = Vec::new();
        for (rel_root, grp) in &bruker_groups {
            if let Some(exp) = build_bruker_experiment(rel_root, root_name, grp) {
                experiments.push(exp);
            }
        }
        for (rel_root, grp) in &varian_groups {
            if let Some(exp) = build_varian_experiment(rel_root, root_name, grp) {
                experiments.push(exp);
            }
        }
        for (rel, hdr_bytes, total_bytes) in ucsf_entries {
            if let Ok((dim, axes, base_mhz)) = parse_ucsf_header(&hdr_bytes) {
                let disp = display_name_from_rel(&rel, root_name);
                let stem = rel
                    .rsplit('/')
                    .next()
                    .unwrap_or(&rel)
                    .trim_end_matches(".ucsf")
                    .trim_end_matches(".UCSF");
                let pulse_class = classify_pulse_token(stem, dim, &axes)
                    .filter(|&c| c != PulseSequenceClass::Unknown)
                    .or_else(|| {
                        classify_pulse_token(&disp, dim, &axes)
                            .filter(|&c| c != PulseSequenceClass::Unknown)
                    })
                    .unwrap_or_else(|| infer_pulse_class_from_axis_physics(dim, &axes, None));
                let quadrature_badge = format_quadrature_badge(&axes);
                experiments.push(ClassifiedExperiment {
                    relative_path: rel,
                    display_name: disp,
                    vendor: ExperimentVendor::SparkyUcsf,
                    dimensionality: dim,
                    data_state: DataState::Processed,
                    spectrometer_mhz: base_mhz,
                    axes,
                    quadrature_badge,
                    nus: NusSummary::default(),
                    pulse_class,
                    pulse_program: None,
                    title: None,
                    raw_signal_bytes: 0,
                    processed_bytes: total_bytes,
                    selected_by_default: pulse_class != PulseSequenceClass::Unknown,
                    notes: Vec::new(),
                });
            }
        }
        for (_key, pgrp) in pipe_series {
            if let Ok((dim, axes, base_mhz)) = parse_nmrpipe_header(
                &pgrp.header_bytes,
                (pgrp.plane_count > 1).then_some(pgrp.plane_count),
            ) {
                let disp = display_name_from_rel(&pgrp.representative_path, root_name);
                let pulse_class = classify_pulse_token(&pgrp.stem_name, dim, &axes)
                    .filter(|&c| c != PulseSequenceClass::Unknown)
                    .unwrap_or_else(|| infer_pulse_class_from_axis_physics(dim, &axes, None));
                let quadrature_badge = format_quadrature_badge(&axes);
                experiments.push(ClassifiedExperiment {
                    relative_path: pgrp.representative_path,
                    display_name: disp,
                    vendor: ExperimentVendor::NmrPipe,
                    dimensionality: dim,
                    data_state: DataState::Processed,
                    spectrometer_mhz: base_mhz,
                    axes,
                    quadrature_badge,
                    nus: NusSummary::default(),
                    pulse_class,
                    pulse_program: None,
                    title: None,
                    raw_signal_bytes: 0,
                    processed_bytes: pgrp.total_bytes,
                    selected_by_default: pulse_class != PulseSequenceClass::Unknown,
                    notes: vec![format!("NMRPipe {} plane stack", pgrp.plane_count)],
                });
            }
        }
        for (rel, hdr_bytes, total_bytes) in pipe_singles {
            if let Ok((dim, axes, base_mhz)) = parse_nmrpipe_header(&hdr_bytes, None) {
                let disp = display_name_from_rel(&rel, root_name);
                let stem = rel
                    .rsplit('/')
                    .next()
                    .unwrap_or(&rel)
                    .split('.')
                    .next()
                    .unwrap_or(&rel);
                let pulse_class = classify_pulse_token(stem, dim, &axes)
                    .filter(|&c| c != PulseSequenceClass::Unknown)
                    .or_else(|| {
                        classify_pulse_token(&disp, dim, &axes)
                            .filter(|&c| c != PulseSequenceClass::Unknown)
                    })
                    .unwrap_or_else(|| infer_pulse_class_from_axis_physics(dim, &axes, None));
                let quadrature_badge = format_quadrature_badge(&axes);
                experiments.push(ClassifiedExperiment {
                    relative_path: rel,
                    display_name: disp,
                    vendor: ExperimentVendor::NmrPipe,
                    dimensionality: dim,
                    data_state: DataState::Processed,
                    spectrometer_mhz: base_mhz,
                    axes,
                    quadrature_badge,
                    nus: NusSummary::default(),
                    pulse_class,
                    pulse_program: None,
                    title: None,
                    raw_signal_bytes: 0,
                    processed_bytes: total_bytes,
                    selected_by_default: pulse_class != PulseSequenceClass::Unknown,
                    notes: Vec::new(),
                });
            }
        }

        // If at least one experiment in the workspace has real binary signal/processed bytes,
        // uncheck empty unacquired header shells (`raw_signal_bytes == 0 && processed_bytes == 0`).
        let any_binary = experiments
            .iter()
            .any(|e| e.raw_signal_bytes > 0 || e.processed_bytes > 0);
        if any_binary {
            for exp in &mut experiments {
                if exp.raw_signal_bytes == 0 && exp.processed_bytes == 0 {
                    exp.selected_by_default = false;
                }
            }
        }

        experiments.sort_by(|a, b| a.relative_path.cmp(&b.relative_path));
        aux_files.sort_by(|a, b| a.relative_path.cmp(&b.relative_path));

        let (regime, secondary, hints) = infer_regime_and_hints(&mut experiments, &aux_files);
        (experiments, aux_files, regime, secondary, hints)
    });

    WorkspaceInventory {
        root_name: root_name.to_string(),
        experiments,
        auxiliary_files,
        problem_regime,
        secondary_regimes,
        missing_input_hints,
        elapsed_ms,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn collect_virtual_entries_root(
    root: &Path,
    entries: &mut Vec<VirtualEntry>,
    deferred_archives: &mut Vec<std::path::PathBuf>,
) -> Result<(), String> {
    let rd = fs::read_dir(root)
        .map_err(|e| format!("failed to read directory {}: {e}", root.display()))?;
    let mut children: Vec<fs::DirEntry> = rd.flatten().collect();
    children.sort_by_key(|e| e.file_name());

    let is_single_exp = children.iter().any(|e| {
        e.file_name().to_str().is_some_and(|n| {
            n.eq_ignore_ascii_case("acqus")
                || n.eq_ignore_ascii_case("acqu")
                || n.eq_ignore_ascii_case("procpar")
        })
    });
    if is_single_exp {
        return collect_virtual_entries_from_dir(root, root, false, entries, deferred_archives);
    }

    let mut subdirs = Vec::new();
    for entry in children {
        let fname_os = entry.file_name();
        let Some(fname) = fname_os.to_str() else {
            continue;
        };
        if fname == ".git" || fname == "__MACOSX" || fname == ".DS_Store" || fname.starts_with("._")
        {
            continue;
        }
        let Ok(ft) = entry.file_type() else {
            continue;
        };
        if ft.is_dir() {
            subdirs.push(entry.path());
            continue;
        }
        if !ft.is_file() {
            continue;
        }
        let lower = fname.to_ascii_lowercase();
        let path = entry.path();
        if is_tar_path_lower(&lower) || is_targz_path_lower(&lower) {
            deferred_archives.push(path);
            continue;
        }
        let Ok(rel_os) = path.strip_prefix(root) else {
            continue;
        };
        let rel = normalize_rel_path(&rel_os.to_string_lossy());
        match classify_file_read_action(&rel) {
            FileReadAction::Ignore => {}
            FileReadAction::MetadataOnlySignal => {
                let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
                entries.push(VirtualEntry::header_only(rel, Vec::new(), size));
            }
            FileReadAction::BoundedBinary(limit) | FileReadAction::BoundedText(limit) => {
                if let Ok(prefix) = read_prefix_from_disk(&path, limit) {
                    let size = if prefix.len() < limit {
                        prefix.len() as u64
                    } else {
                        entry
                            .metadata()
                            .map(|m| m.len())
                            .unwrap_or(prefix.len() as u64)
                    };
                    entries.push(VirtualEntry::header_only(rel, prefix, size));
                }
            }
        }
    }

    // Expand one extra directory level for non-experiment bundle containers so large multi-bundle
    // workspaces (e.g. 140+ experiments nested two levels deep) balance evenly across worker threads.
    let mut work_dirs = Vec::new();
    for sub in subdirs {
        let Ok(sub_rd) = fs::read_dir(&sub) else {
            continue;
        };
        let mut sub_children: Vec<fs::DirEntry> = sub_rd.flatten().collect();
        sub_children.sort_by_key(|e| e.file_name());
        let sub_is_exp = sub_children.iter().any(|e| {
            e.file_name().to_str().is_some_and(|n| {
                n.eq_ignore_ascii_case("acqus")
                    || n.eq_ignore_ascii_case("acqu")
                    || n.eq_ignore_ascii_case("procpar")
            })
        });
        if sub_is_exp {
            work_dirs.push(sub);
            continue;
        }
        for entry in sub_children {
            let fname_os = entry.file_name();
            let Some(fname) = fname_os.to_str() else {
                continue;
            };
            if fname == ".git"
                || fname == "__MACOSX"
                || fname == ".DS_Store"
                || fname.starts_with("._")
            {
                continue;
            }
            let Ok(ft) = entry.file_type() else {
                continue;
            };
            if ft.is_dir() {
                work_dirs.push(entry.path());
            } else if ft.is_file() {
                let lower = fname.to_ascii_lowercase();
                let path = entry.path();
                if is_tar_path_lower(&lower) || is_targz_path_lower(&lower) {
                    deferred_archives.push(path);
                    continue;
                }
                if let Ok(rel_os) = path.strip_prefix(root) {
                    let rel = normalize_rel_path(&rel_os.to_string_lossy());
                    match classify_file_read_action(&rel) {
                        FileReadAction::Ignore => {}
                        FileReadAction::MetadataOnlySignal => {
                            let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
                            entries.push(VirtualEntry::header_only(rel, Vec::new(), size));
                        }
                        FileReadAction::BoundedBinary(limit)
                        | FileReadAction::BoundedText(limit) => {
                            if let Ok(prefix) = read_prefix_from_disk(&path, limit) {
                                let size = if prefix.len() < limit {
                                    prefix.len() as u64
                                } else {
                                    entry
                                        .metadata()
                                        .map(|m| m.len())
                                        .unwrap_or(prefix.len() as u64)
                                };
                                entries.push(VirtualEntry::header_only(rel, prefix, size));
                            }
                        }
                    }
                }
            }
        }
    }

    if work_dirs.len() < 4 {
        for sub in work_dirs {
            collect_virtual_entries_from_dir(root, &sub, false, entries, deferred_archives)?;
        }
        return Ok(());
    }

    type ScanBucketOutput = (Vec<VirtualEntry>, Vec<std::path::PathBuf>);
    let n_workers = work_dirs.len().min(8);
    let mut buckets: Vec<Vec<std::path::PathBuf>> = vec![Vec::new(); n_workers];
    for (idx, dir) in work_dirs.into_iter().enumerate() {
        buckets[idx % n_workers].push(dir);
    }
    let results: Vec<Result<ScanBucketOutput, String>> = std::thread::scope(|s| {
        let handles: Vec<_> = buckets
            .iter()
            .map(|bucket| {
                s.spawn(move || {
                    let mut loc_entries = Vec::new();
                    let mut loc_archives = Vec::new();
                    for sub in bucket {
                        collect_virtual_entries_from_dir(
                            root,
                            sub,
                            false,
                            &mut loc_entries,
                            &mut loc_archives,
                        )?;
                    }
                    Ok((loc_entries, loc_archives))
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| {
                h.join()
                    .unwrap_or_else(|_| Err("directory scanner worker panicked".to_string()))
            })
            .collect()
    });

    for res in results {
        let (loc_e, loc_a) = res?;
        entries.extend(loc_e);
        deferred_archives.extend(loc_a);
    }
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn collect_virtual_entries_root(
    root: &Path,
    entries: &mut Vec<VirtualEntry>,
    deferred_archives: &mut Vec<std::path::PathBuf>,
) -> Result<(), String> {
    collect_virtual_entries_from_dir(root, root, false, entries, deferred_archives)
}

/// Inspect and classify a spectrometer workspace directory, single spectrum/sequence file,
/// or `.tar` / `.tar.gz` / `.tgz` archive on disk without reading bulk binary FID or cube payloads.
pub fn classify_nmr_workspace(path: &Path) -> Result<WorkspaceInventory, String> {
    let meta = fs::metadata(path).map_err(|e| format!("cannot access {}: {e}", path.display()))?;
    let root_name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("workspace")
        .to_string();

    let (mut inv, total_ms) = measure_ms(|| -> Result<WorkspaceInventory, String> {
        let mut entries = Vec::new();
        if meta.is_dir() {
            let mut deferred_archives = Vec::new();
            collect_virtual_entries_root(path, &mut entries, &mut deferred_archives)?;
            // Only scan child `.tar`/`.tar.gz` archives if no unpacked spectral headers were found!
            let has_spectral = entries.iter().any(|e| {
                let low = e.relative_path.to_ascii_lowercase();
                !(low.ends_with(".str")
                    || low.ends_with(".star")
                    || low.ends_with(".bmrb")
                    || low.ends_with(".nef")
                    || low.ends_with(".fasta")
                    || low.ends_with(".faa")
                    || low.ends_with(".seq"))
            });
            if !has_spectral {
                for arch in deferred_archives {
                    let alow = arch.to_string_lossy().to_ascii_lowercase();
                    if is_targz_path_lower(&alow) {
                        let f = File::open(&arch)
                            .map_err(|e| format!("failed to open {}: {e}", arch.display()))?;
                        scan_targz_reader(f, MAX_DISK_INFLATE_BYTES, &mut entries)?;
                    } else if is_tar_path_lower(&alow) {
                        let mut f = File::open(&arch)
                            .map_err(|e| format!("failed to open {}: {e}", arch.display()))?;
                        scan_tar_seekable(&mut f, &mut entries)?;
                    }
                }
            }
        } else if meta.is_file() {
            let lower = root_name.to_ascii_lowercase();
            if is_targz_path_lower(&lower) {
                let f = File::open(path)
                    .map_err(|e| format!("failed to open archive {}: {e}", path.display()))?;
                scan_targz_reader(f, MAX_DISK_INFLATE_BYTES, &mut entries)?;
            } else if is_tar_path_lower(&lower) {
                let mut f = File::open(path)
                    .map_err(|e| format!("failed to open archive {}: {e}", path.display()))?;
                scan_tar_seekable(&mut f, &mut entries)?;
            } else {
                match classify_file_read_action(&root_name) {
                    FileReadAction::BoundedBinary(lim) | FileReadAction::BoundedText(lim) => {
                        let prefix = read_prefix_from_disk(path, lim)
                            .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
                        entries.push(VirtualEntry::header_only(
                            root_name.clone(),
                            prefix,
                            meta.len(),
                        ));
                    }
                    FileReadAction::MetadataOnlySignal => {
                        entries.push(VirtualEntry::header_only(
                            root_name.clone(),
                            Vec::new(),
                            meta.len(),
                        ));
                    }
                    FileReadAction::Ignore => {
                        return Err(format!(
                            "unsupported NMR workspace or archive file: {}",
                            path.display()
                        ));
                    }
                }
            }
        } else {
            return Err(format!("unsupported path kind: {}", path.display()));
        }
        Ok(classify_nmr_entries_named(&root_name, &entries))
    });

    if let Ok(ref mut inventory) = inv {
        inventory.elapsed_ms = total_ms;
    }
    inv
}
