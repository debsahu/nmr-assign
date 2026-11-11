//! Processed 2D and 3D NMR spectrum representation and Bruker `pdata`, NMRPipe, and UCSF Sparky I/O.
//!
//! Provenance: Original clean-room implementation of `docs/specs/01-data-io.md` from public
//! format specifications:
//! - NMRPipe 512-float header specification (Delaglio et al., *J. Biomol. NMR* 6, 277-293, 1995).
//! - Bruker TopSpin JCAMP-DX `procs`/`proc2s`/`proc3s` and submatrix `2rr`/`3rrr` binary layout.
//! - UCSF Sparky v2 `.ucsf` binary format specification (Goddard & Kneller, *SPARKY 3 Manual —
//!   UCSF NMR Data Format*, University of California, San Francisco: 180-byte global header,
//!   128-byte big-endian axis headers, and zero-padded big-endian IEEE 754 `f32` data tiles).

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

#[derive(Clone, Debug)]
pub struct AxisPpm {
    pub label: String,
    pub size: usize,
    /// Chemical shift (ppm) at index `0` (high-ppm edge).
    pub first_ppm: f64,
    /// Signed step per point in ppm (typically negative: `-sw_ppm / size`).
    pub step_ppm: f64,
    /// Spectrometer observation frequency in MHz.
    pub obs_mhz: f64,
}

impl AxisPpm {
    /// Construct a standard descending NMR axis spanning `[low_ppm, high_ppm]` across `size` points.
    pub fn from_bounds(
        label: impl Into<String>,
        size: usize,
        high_ppm: f64,
        low_ppm: f64,
        obs_mhz: f64,
    ) -> Self {
        let step_ppm = if size > 1 {
            (low_ppm - high_ppm) / ((size - 1) as f64)
        } else {
            -0.01
        };
        Self {
            label: label.into(),
            size,
            first_ppm: high_ppm,
            step_ppm,
            obs_mhz,
        }
    }

    #[inline]
    pub fn sw_ppm(&self) -> f64 {
        (self.step_ppm.abs()) * (self.size as f64)
    }

    #[inline]
    pub fn last_ppm(&self) -> f64 {
        self.index_to_ppm((self.size.saturating_sub(1)) as f64)
    }

    #[inline]
    pub fn max_ppm(&self) -> f64 {
        self.first_ppm.max(self.last_ppm())
    }

    #[inline]
    pub fn min_ppm(&self) -> f64 {
        self.first_ppm.min(self.last_ppm())
    }

    #[inline]
    pub fn index_to_ppm(&self, idx: f64) -> f64 {
        self.first_ppm + idx * self.step_ppm
    }

    #[inline]
    pub fn ppm_to_index(&self, ppm: f64) -> f64 {
        (ppm - self.first_ppm) / self.step_ppm
    }
}

/// 2D processed spectrum stored row-major `[y_indirect * nx + x_direct]`.
/// Typically `axis_x` is `1H` (direct) and `axis_y` is `15N` or `13C` (indirect).
#[derive(Clone, Debug)]
pub struct Spectrum2D {
    pub name: String,
    pub axis_x: AxisPpm,
    pub axis_y: AxisPpm,
    pub data: Vec<f32>,
}

impl Spectrum2D {
    #[inline]
    pub fn nx(&self) -> usize {
        self.axis_x.size
    }

    #[inline]
    pub fn ny(&self) -> usize {
        self.axis_y.size
    }

    #[inline]
    pub fn get(&self, ix: usize, iy: usize) -> f32 {
        self.data[iy * self.axis_x.size + ix]
    }

    pub fn max_abs(&self) -> f32 {
        self.data.iter().fold(0.0_f32, |m, &v| m.max(v.abs()))
    }
}

/// 3D processed spectrum stored `[iz * (ny * nx) + iy * nx + ix]`.
/// Typically `axis_x` is `1H`, `axis_y` is `15N`, and `axis_z` is `13C` (or `axis_y`/`axis_z` swapped).
#[derive(Clone, Debug)]
pub struct Spectrum3D {
    pub name: String,
    pub exp: String,
    pub axis_x: AxisPpm,
    pub axis_y: AxisPpm,
    pub axis_z: AxisPpm,
    pub data: Vec<f32>,
}

impl Spectrum3D {
    #[inline]
    pub fn nx(&self) -> usize {
        self.axis_x.size
    }

    #[inline]
    pub fn ny(&self) -> usize {
        self.axis_y.size
    }

    #[inline]
    pub fn nz(&self) -> usize {
        self.axis_z.size
    }

    #[inline]
    pub fn get(&self, ix: usize, iy: usize, iz: usize) -> f32 {
        let nx = self.axis_x.size;
        let ny = self.axis_y.size;
        self.data[(iz * ny + iy) * nx + ix]
    }

    /// Extract a 2D `(1H strip_width x 13C full_axis)` strip around `(h_ppm, n_ppm)`.
    /// Output has `axis_x` = 1H window and `axis_y` = 13C axis for vertical strip plotting.
    pub fn extract_strip(&self, h_ppm: f64, n_ppm: f64, half_width_h_ppm: f64) -> Spectrum2D {
        // Identify which indirect axis is 15N vs 13C by label or ppm range
        let y_is_n = self.axis_y.label.contains('N')
            || (self.axis_y.max_ppm() > 95.0
                && self.axis_y.min_ppm() < 145.0
                && !self.axis_z.label.contains('N'));

        let (n_axis, c_axis) = if y_is_n {
            (&self.axis_y, &self.axis_z)
        } else {
            (&self.axis_z, &self.axis_y)
        };

        let in_idx = n_axis
            .ppm_to_index(n_ppm)
            .round()
            .clamp(0.0, (n_axis.size.saturating_sub(1)) as f64) as usize;

        let idx_hi = self.axis_x.ppm_to_index(h_ppm + half_width_h_ppm);
        let idx_lo = self.axis_x.ppm_to_index(h_ppm - half_width_h_ppm);
        let x0 = idx_hi.min(idx_lo).floor().max(0.0) as usize;
        let x1 = idx_hi
            .max(idx_lo)
            .ceil()
            .min((self.axis_x.size.saturating_sub(1)) as f64) as usize;
        let sub_nx = (x1.saturating_sub(x0) + 1).max(1);
        let sub_ny = c_axis.size;

        let mut strip_data = vec![0.0f32; sub_ny * sub_nx];
        for ic in 0..sub_ny {
            for (dx, ix) in (x0..x0 + sub_nx).enumerate() {
                let v = if y_is_n {
                    self.get(ix, in_idx, ic)
                } else {
                    self.get(ix, ic, in_idx)
                };
                strip_data[ic * sub_nx + dx] = v;
            }
        }

        let strip_axis_x = AxisPpm {
            label: self.axis_x.label.clone(),
            size: sub_nx,
            first_ppm: self.axis_x.index_to_ppm(x0 as f64),
            step_ppm: self.axis_x.step_ppm,
            obs_mhz: self.axis_x.obs_mhz,
        };

        Spectrum2D {
            name: format!("{} strip", self.exp),
            axis_x: strip_axis_x,
            axis_y: c_axis.clone(),
            data: strip_data,
        }
    }
}

// ============================================================================
// Bruker TopSpin pdata reader & writer
// ============================================================================

fn parse_jcamp_params(text: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("##$") {
            if let Some((k, v)) = rest.split_once('=') {
                let clean_v = v.trim().trim_matches(|c| c == '<' || c == '>').trim();
                map.insert(k.trim().to_string(), clean_v.to_string());
            }
        }
    }
    map
}

fn parse_bruker_axis(
    params: &HashMap<String, String>,
    fallback_label: &str,
) -> Result<(AxisPpm, usize, i32, i32, i32), String> {
    let si: usize = params
        .get("SI")
        .ok_or("missing ##$SI in Bruker procs")?
        .parse()
        .map_err(|e| format!("invalid SI: {e}"))?;
    let xdim: usize = params
        .get("XDIM")
        .and_then(|v| v.parse().ok())
        .unwrap_or(si)
        .max(1);
    let sw_hz: f64 = params
        .get("SW_p")
        .ok_or("missing ##$SW_p in Bruker procs")?
        .parse()
        .map_err(|e| format!("invalid SW_p: {e}"))?;
    let sf_mhz: f64 = params
        .get("SF")
        .ok_or("missing ##$SF in Bruker procs")?
        .parse()
        .map_err(|e| format!("invalid SF: {e}"))?;
    let offset_ppm: f64 = params
        .get("OFFSET")
        .ok_or("missing ##$OFFSET in Bruker procs")?
        .parse()
        .map_err(|e| format!("invalid OFFSET: {e}"))?;
    let bytordp: i32 = params
        .get("BYTORDP")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let dtypp: i32 = params
        .get("DTYPP")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let nc_proc: i32 = params
        .get("NC_proc")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let label = params
        .get("AXNUC")
        .cloned()
        .filter(|s| !s.is_empty() && s != "off")
        .unwrap_or_else(|| fallback_label.to_string());

    let sw_ppm = sw_hz / sf_mhz.max(1e-9);
    let step_ppm = -sw_ppm / (si.max(1) as f64);

    Ok((
        AxisPpm {
            label,
            size: si,
            first_ppm: offset_ppm,
            step_ppm,
            obs_mhz: sf_mhz,
        },
        xdim,
        bytordp,
        dtypp,
        nc_proc,
    ))
}

fn decode_bruker_words(
    bytes: &[u8],
    bytordp: i32,
    dtypp: i32,
    nc_proc: i32,
    expected: usize,
) -> Result<Vec<f32>, String> {
    let scale = 2.0_f64.powi(nc_proc);
    let big_endian = bytordp != 0;
    if dtypp == 2 {
        // 64-bit float
        if bytes.len() < expected * 8 {
            return Err(format!(
                "Bruker data truncated: need {} bytes, got {}",
                expected * 8,
                bytes.len()
            ));
        }
        let mut out = Vec::with_capacity(expected);
        for chunk in bytes[..expected * 8].chunks_exact(8) {
            let arr: [u8; 8] = chunk.try_into().unwrap();
            let v = if big_endian {
                f64::from_be_bytes(arr)
            } else {
                f64::from_le_bytes(arr)
            };
            out.push((v * scale) as f32);
        }
        Ok(out)
    } else {
        // 32-bit int (dtypp == 0) or 32-bit float (dtypp == 1)
        if bytes.len() < expected * 4 {
            return Err(format!(
                "Bruker data truncated: need {} bytes, got {}",
                expected * 4,
                bytes.len()
            ));
        }
        let mut out = Vec::with_capacity(expected);
        for chunk in bytes[..expected * 4].chunks_exact(4) {
            let arr: [u8; 4] = chunk.try_into().unwrap();
            let v = if dtypp == 1 {
                let f = if big_endian {
                    f32::from_be_bytes(arr)
                } else {
                    f32::from_le_bytes(arr)
                };
                (f as f64) * scale
            } else {
                let i = if big_endian {
                    i32::from_be_bytes(arr)
                } else {
                    i32::from_le_bytes(arr)
                };
                (i as f64) * scale
            };
            out.push(v as f32);
        }
        Ok(out)
    }
}

/// Read a Bruker TopSpin 2D processed directory (containing `procs`, `proc2s`, and `2rr`).
pub fn read_bruker_pdata_2d(pdata_dir: impl AsRef<Path>) -> Result<Spectrum2D, String> {
    let dir = pdata_dir.as_ref();
    let procs_txt = fs::read_to_string(dir.join("procs"))
        .map_err(|e| format!("failed to read {}: {e}", dir.join("procs").display()))?;
    let proc2s_txt = fs::read_to_string(dir.join("proc2s"))
        .map_err(|e| format!("failed to read {}: {e}", dir.join("proc2s").display()))?;

    let p1 = parse_jcamp_params(&procs_txt);
    let p2 = parse_jcamp_params(&proc2s_txt);
    let (axis_x, xdim_x, bytordp, dtypp, nc_proc) = parse_bruker_axis(&p1, "1H")?;
    let (axis_y, xdim_y, ..) = parse_bruker_axis(&p2, "15N")?;

    let nx = axis_x.size;
    let ny = axis_y.size;
    let raw_bytes = fs::read(dir.join("2rr"))
        .map_err(|e| format!("failed to read {}: {e}", dir.join("2rr").display()))?;
    let raw = decode_bruker_words(&raw_bytes, bytordp, dtypp, nc_proc, nx * ny)?;

    let tx = xdim_x.min(nx).max(1);
    let ty = xdim_y.min(ny).max(1);
    let nbx = nx.div_ceil(tx);
    let nby = ny.div_ceil(ty);

    let mut data = vec![0.0f32; ny * nx];
    let mut idx = 0usize;
    for by in 0..nby {
        for bx in 0..nbx {
            for ly in 0..ty {
                for lx in 0..tx {
                    let gx = bx * tx + lx;
                    let gy = by * ty + ly;
                    if idx < raw.len() {
                        if gx < nx && gy < ny {
                            data[gy * nx + gx] = raw[idx];
                        }
                        idx += 1;
                    }
                }
            }
        }
    }

    Ok(Spectrum2D {
        name: dir
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("2rr")
            .to_string(),
        axis_x,
        axis_y,
        data,
    })
}

/// Write a Bruker TopSpin 2D processed directory (`procs`, `proc2s`, `2rr`) for round-trip testing and export.
pub fn write_bruker_pdata_2d(pdata_dir: impl AsRef<Path>, spec: &Spectrum2D) -> Result<(), String> {
    let dir = pdata_dir.as_ref();
    fs::create_dir_all(dir).map_err(|e| format!("failed to create {}: {e}", dir.display()))?;

    let max_abs = spec.max_abs().max(1e-12) as f64;
    let target_max = 1.0e8_f64;
    let nc_proc = (max_abs / target_max).log2().ceil() as i32;
    let inv_scale = 1.0 / 2.0_f64.powi(nc_proc);

    let write_proc = |path: &Path, ax: &AxisPpm| -> Result<(), String> {
        let sw_hz = ax.sw_ppm() * ax.obs_mhz;
        let content = format!(
            "##TITLE= nmrassign pdata\n\
             ##$SI= {}\n\
             ##$XDIM= {}\n\
             ##$SW_p= {:.6}\n\
             ##$SF= {:.6}\n\
             ##$OFFSET= {:.6}\n\
             ##$AXNUC= <{}>\n\
             ##$BYTORDP= 0\n\
             ##$DTYPP= 0\n\
             ##$NC_proc= {}\n\
             ##END=\n",
            ax.size, ax.size, sw_hz, ax.obs_mhz, ax.first_ppm, ax.label, nc_proc
        );
        fs::write(path, content).map_err(|e| format!("failed to write {}: {e}", path.display()))
    };

    write_proc(&dir.join("procs"), &spec.axis_x)?;
    write_proc(&dir.join("proc2s"), &spec.axis_y)?;

    // With XDIM == SI, single submatrix equals row-major array
    let mut bytes = Vec::with_capacity(spec.data.len() * 4);
    for &v in &spec.data {
        let q = ((v as f64) * inv_scale)
            .round()
            .clamp(i32::MIN as f64, i32::MAX as f64) as i32;
        bytes.extend_from_slice(&q.to_le_bytes());
    }
    fs::write(dir.join("2rr"), bytes).map_err(|e| format!("failed to write 2rr: {e}"))
}

/// Read a Bruker TopSpin 3D processed directory (`procs`, `proc2s`, `proc3s`, and `3rrr`).
pub fn read_bruker_pdata_3d(pdata_dir: impl AsRef<Path>, exp: &str) -> Result<Spectrum3D, String> {
    let dir = pdata_dir.as_ref();
    let p1 = parse_jcamp_params(
        &fs::read_to_string(dir.join("procs")).map_err(|e| format!("failed to read procs: {e}"))?,
    );
    let p2 = parse_jcamp_params(
        &fs::read_to_string(dir.join("proc2s"))
            .map_err(|e| format!("failed to read proc2s: {e}"))?,
    );
    let p3 = parse_jcamp_params(
        &fs::read_to_string(dir.join("proc3s"))
            .map_err(|e| format!("failed to read proc3s: {e}"))?,
    );

    let (axis_x, xdim_x, bytordp, dtypp, nc_proc) = parse_bruker_axis(&p1, "1H")?;
    let (axis_y, xdim_y, ..) = parse_bruker_axis(&p2, "15N")?;
    let (axis_z, xdim_z, ..) = parse_bruker_axis(&p3, "13C")?;

    let (nx, ny, nz) = (axis_x.size, axis_y.size, axis_z.size);
    let raw_bytes = fs::read(dir.join("3rrr")).map_err(|e| format!("failed to read 3rrr: {e}"))?;
    let raw = decode_bruker_words(&raw_bytes, bytordp, dtypp, nc_proc, nx * ny * nz)?;

    let tx = xdim_x.min(nx).max(1);
    let ty = xdim_y.min(ny).max(1);
    let tz = xdim_z.min(nz).max(1);
    let nbx = nx.div_ceil(tx);
    let nby = ny.div_ceil(ty);
    let nbz = nz.div_ceil(tz);

    let mut data = vec![0.0f32; nz * ny * nx];
    let mut idx = 0usize;
    for bz in 0..nbz {
        for by in 0..nby {
            for bx in 0..nbx {
                for lz in 0..tz {
                    for ly in 0..ty {
                        for lx in 0..tx {
                            let gx = bx * tx + lx;
                            let gy = by * ty + ly;
                            let gz = bz * tz + lz;
                            if idx < raw.len() {
                                if gx < nx && gy < ny && gz < nz {
                                    data[(gz * ny + gy) * nx + gx] = raw[idx];
                                }
                                idx += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(Spectrum3D {
        name: dir
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("3rrr")
            .to_string(),
        exp: exp.to_string(),
        axis_x,
        axis_y,
        axis_z,
        data,
    })
}

// ============================================================================
// NMRPipe .ft2 / .ft3 reader & writer (512-float 2048-byte header)
// ============================================================================

const PIPE_HEADER_FLOATS: usize = 512;
const PIPE_HEADER_BYTES: usize = 2048;

// Parameter slot indices for F2 (slot 0), F1 (slot 1), F3 (slot 2), F4 (slot 3)
const PIPE_SW_IDX: [usize; 4] = [100, 229, 11, 29];
const PIPE_ORIG_IDX: [usize; 4] = [101, 249, 12, 30];
const PIPE_OBS_IDX: [usize; 4] = [119, 218, 10, 28];
const PIPE_LABEL_IDX: [usize; 4] = [16, 18, 20, 22];

fn decode_pipe_header(bytes: &[u8]) -> Result<(Vec<f32>, bool), String> {
    if bytes.len() < PIPE_HEADER_BYTES {
        return Err("NMRPipe file shorter than 2048-byte header".to_string());
    }
    let order_le = f32::from_le_bytes(bytes[8..12].try_into().unwrap());
    let order_be = f32::from_be_bytes(bytes[8..12].try_into().unwrap());
    let big_endian = if (order_le - 2.345).abs() < 1e-2 {
        false
    } else if (order_be - 2.345).abs() < 1e-2 {
        true
    } else {
        return Err(format!(
            "unrecognized NMRPipe byte-order magic (le={order_le}, be={order_be})"
        ));
    };

    let mut hdr = Vec::with_capacity(PIPE_HEADER_FLOATS);
    for chunk in bytes[..PIPE_HEADER_BYTES].chunks_exact(4) {
        let arr: [u8; 4] = chunk.try_into().unwrap();
        hdr.push(if big_endian {
            f32::from_be_bytes(arr)
        } else {
            f32::from_le_bytes(arr)
        });
    }
    Ok((hdr, big_endian))
}

fn pipe_label(raw_header: &[u8], slot: usize) -> String {
    let start = PIPE_LABEL_IDX[slot] * 4;
    let slice = &raw_header[start..start + 8];
    let end = slice.iter().position(|&b| b == 0).unwrap_or(8);
    let s = String::from_utf8_lossy(&slice[..end]).trim().to_string();
    if s.is_empty() {
        format!("F{}", slot + 1)
    } else {
        s
    }
}

fn pipe_axis(hdr: &[f32], raw_header: &[u8], slot: usize, size: usize) -> AxisPpm {
    let sw_hz = (hdr[PIPE_SW_IDX[slot]] as f64).max(1e-3);
    let orig_hz = hdr[PIPE_ORIG_IDX[slot]] as f64;
    let obs_mhz = (hdr[PIPE_OBS_IDX[slot]] as f64).max(1e-3);
    let n = size.max(1) as f64;
    // In NMRPipe, ORIG is the frequency in Hz at the last point (size - 1).
    let first_hz = orig_hz + sw_hz * ((n - 1.0) / n);
    let first_ppm = first_hz / obs_mhz;
    let step_ppm = -sw_hz / (obs_mhz * n);
    AxisPpm {
        label: pipe_label(raw_header, slot),
        size,
        first_ppm,
        step_ppm,
        obs_mhz,
    }
}

fn pipe_slot_for_dim(hdr: &[f32], dim_idx: usize) -> usize {
    let v = hdr[24 + dim_idx].round() as isize;
    // NMRPipe FDDIMORDER stores 1-based parameter codes where 2->F2(slot 0), 1->F1(slot 1), 3->F3(slot 2), 4->F4(slot 3)
    match v {
        2 => 0,
        1 => 1,
        3 => 2,
        4 => 3,
        _ => dim_idx.min(3),
    }
}

/// Read an NMRPipe 2D spectrum (`.ft2` / `.dat`).
pub fn read_nmrpipe_2d(path: impl AsRef<Path>) -> Result<Spectrum2D, String> {
    let bytes = fs::read(path.as_ref())
        .map_err(|e| format!("failed to read {}: {e}", path.as_ref().display()))?;
    let (hdr, big_endian) = decode_pipe_header(&bytes)?;

    let nx = (hdr[99].round() as usize).max(1);
    let ny = (hdr[219].round() as usize).max(1);
    let slot_x = pipe_slot_for_dim(&hdr, 0);
    let slot_y = pipe_slot_for_dim(&hdr, 1);

    let axis_x = pipe_axis(&hdr, &bytes[..PIPE_HEADER_BYTES], slot_x, nx);
    let axis_y = pipe_axis(&hdr, &bytes[..PIPE_HEADER_BYTES], slot_y, ny);

    let expected = nx * ny;
    let body = &bytes[PIPE_HEADER_BYTES..];
    if body.len() < expected * 4 {
        return Err(format!(
            "NMRPipe 2D body truncated: expected {} floats, got {} bytes",
            expected,
            body.len()
        ));
    }

    let mut data = Vec::with_capacity(expected);
    for chunk in body[..expected * 4].chunks_exact(4) {
        let arr: [u8; 4] = chunk.try_into().unwrap();
        data.push(if big_endian {
            f32::from_be_bytes(arr)
        } else {
            f32::from_le_bytes(arr)
        });
    }

    Ok(Spectrum2D {
        name: path
            .as_ref()
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("nmrpipe.ft2")
            .to_string(),
        axis_x,
        axis_y,
        data,
    })
}

/// Write an NMRPipe 2D spectrum (`.ft2`) with complete frequency referencing.
pub fn write_nmrpipe_2d(path: impl AsRef<Path>, spec: &Spectrum2D) -> Result<(), String> {
    let mut hdr = [0.0f32; PIPE_HEADER_FLOATS];
    hdr[0] = 0.0;
    hdr[1] = f32::from_bits(0xeeee_eeee);
    hdr[2] = 2.345;
    hdr[9] = 2.0; // FDDIMCOUNT
    hdr[24] = 2.0; // dim 0 -> F2 (slot 0)
    hdr[25] = 1.0; // dim 1 -> F1 (slot 1)
    hdr[26] = 3.0;
    hdr[27] = 4.0;
    hdr[56] = 1.0; // FDQUADFLAG: real
    hdr[106] = 1.0;
    hdr[55] = 1.0;
    hdr[99] = spec.nx() as f32; // FDSIZE
    hdr[219] = spec.ny() as f32; // FDSPECNUM

    let set_slot = |hdr: &mut [f32; PIPE_HEADER_FLOATS], slot: usize, ax: &AxisPpm| {
        let n = ax.size.max(1) as f64;
        let sw_hz = ax.sw_ppm() * ax.obs_mhz;
        let last_ppm = ax.index_to_ppm(n - 1.0);
        let orig_hz = last_ppm * ax.obs_mhz;
        hdr[PIPE_SW_IDX[slot]] = sw_hz as f32;
        hdr[PIPE_ORIG_IDX[slot]] = orig_hz as f32;
        hdr[PIPE_OBS_IDX[slot]] = ax.obs_mhz as f32;
    };
    set_slot(&mut hdr, 0, &spec.axis_x);
    set_slot(&mut hdr, 1, &spec.axis_y);

    let mut bytes = Vec::with_capacity(PIPE_HEADER_BYTES + spec.data.len() * 4);
    for &w in &hdr {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    // Write 8-byte ASCII labels
    let write_label = |buf: &mut [u8], slot: usize, label: &str| {
        let start = PIPE_LABEL_IDX[slot] * 4;
        let mut raw = [0u8; 8];
        let b = label.as_bytes();
        let len = b.len().min(8);
        raw[..len].copy_from_slice(&b[..len]);
        buf[start..start + 8].copy_from_slice(&raw);
    };
    write_label(&mut bytes[..PIPE_HEADER_BYTES], 0, &spec.axis_x.label);
    write_label(&mut bytes[..PIPE_HEADER_BYTES], 1, &spec.axis_y.label);

    for &v in &spec.data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(path.as_ref(), bytes)
        .map_err(|e| format!("failed to write {}: {e}", path.as_ref().display()))
}

/// Read a single-stream NMRPipe 3D file (`.ft3`).
pub fn read_nmrpipe_3d(path: impl AsRef<Path>, exp: &str) -> Result<Spectrum3D, String> {
    let bytes = fs::read(path.as_ref())
        .map_err(|e| format!("failed to read {}: {e}", path.as_ref().display()))?;
    let (hdr, big_endian) = decode_pipe_header(&bytes)?;

    let nx = (hdr[99].round() as usize).max(1);
    let ny = (hdr[219].round() as usize).max(1);
    let nz = (hdr[15].round() as usize).max(1);

    let slot_x = pipe_slot_for_dim(&hdr, 0);
    let slot_y = pipe_slot_for_dim(&hdr, 1);
    let slot_z = pipe_slot_for_dim(&hdr, 2);

    let axis_x = pipe_axis(&hdr, &bytes[..PIPE_HEADER_BYTES], slot_x, nx);
    let axis_y = pipe_axis(&hdr, &bytes[..PIPE_HEADER_BYTES], slot_y, ny);
    let axis_z = pipe_axis(&hdr, &bytes[..PIPE_HEADER_BYTES], slot_z, nz);

    let expected = nx * ny * nz;
    let body = &bytes[PIPE_HEADER_BYTES..];
    if body.len() < expected * 4 {
        return Err(format!(
            "NMRPipe 3D body truncated: expected {} floats, got {} bytes",
            expected,
            body.len()
        ));
    }

    let mut data = Vec::with_capacity(expected);
    for chunk in body[..expected * 4].chunks_exact(4) {
        let arr: [u8; 4] = chunk.try_into().unwrap();
        data.push(if big_endian {
            f32::from_be_bytes(arr)
        } else {
            f32::from_le_bytes(arr)
        });
    }

    Ok(Spectrum3D {
        name: path
            .as_ref()
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("nmrpipe.ft3")
            .to_string(),
        exp: exp.to_string(),
        axis_x,
        axis_y,
        axis_z,
        data,
    })
}

// ============================================================================
// UCSF Sparky v2 .ucsf reader & writer (180-byte header + N*128-byte axis headers + BE f32 tiles)
// ============================================================================

const UCSF_FILE_HEADER_BYTES: usize = 180;
const UCSF_AXIS_HEADER_BYTES: usize = 128;
const UCSF_MAGIC_PREFIX: &[u8; 8] = b"UCSF NMR";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum UcsfNuc {
    H,
    N,
    C,
}

/// Check whether an in-memory buffer starts with the UCSF NMR magic signature (`"UCSF NMR\0\0"`).
pub fn is_ucsf_bytes(bytes: &[u8]) -> bool {
    bytes.len() >= 10 && &bytes[..8] == UCSF_MAGIC_PREFIX && bytes[8] == 0
}

/// Check whether `path` is a UCSF Sparky spectrum file by sniffing its 10-byte magic header
/// (or `.ucsf` extension if unreadable).
pub fn is_ucsf_file(path: impl AsRef<Path>) -> bool {
    let p = path.as_ref();
    if p.is_dir() {
        return false;
    }
    if let Ok(mut f) = File::open(p) {
        let mut magic = [0u8; 10];
        if f.read_exact(&mut magic).is_ok() {
            return is_ucsf_bytes(&magic);
        }
    }
    p.extension()
        .and_then(|s| s.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("ucsf"))
}

fn classify_ucsf_nucleus(ax: &AxisPpm) -> UcsfNuc {
    let up = ax.label.to_ascii_uppercase();
    if up.contains('N') {
        UcsfNuc::N
    } else if up.contains('C') {
        UcsfNuc::C
    } else if up.contains('H') || (ax.max_ppm() <= 20.0 && ax.min_ppm() >= -5.0) {
        UcsfNuc::H
    } else if ax.min_ppm() >= 90.0 && ax.max_ppm() <= 150.0 && ax.sw_ppm() < 60.0 {
        UcsfNuc::N
    } else {
        UcsfNuc::C
    }
}

fn parse_ucsf_file_header(hdr: &[u8], expected_ndim: u8) -> Result<(), String> {
    if hdr.len() < UCSF_FILE_HEADER_BYTES {
        return Err(format!(
            "UCSF file shorter than {UCSF_FILE_HEADER_BYTES}-byte global header ({} bytes)",
            hdr.len()
        ));
    }
    if !is_ucsf_bytes(hdr) {
        return Err("unrecognized UCSF magic header (expected 'UCSF NMR\\0\\0')".to_string());
    }
    let ndim = hdr[10];
    if ndim != expected_ndim {
        return Err(format!(
            "UCSF dimension mismatch: file has ndim={ndim}, expected {expected_ndim}"
        ));
    }
    let ncomp = hdr[11];
    if ncomp != 1 {
        return Err(format!(
            "unsupported UCSF ncomponents={ncomp} (expected 1 for real spectrum)"
        ));
    }
    let encoding = hdr[12];
    if encoding != 0 {
        return Err(format!(
            "unsupported UCSF encoding={encoding} (expected 0 for IEEE 754 float32)"
        ));
    }
    let version = hdr[13];
    if version != 2 {
        return Err(format!(
            "unsupported UCSF format version={version} (expected 2)"
        ));
    }
    Ok(())
}

fn parse_ucsf_axis_header(ax_hdr: &[u8], axis_idx: usize) -> Result<(AxisPpm, usize), String> {
    if ax_hdr.len() < UCSF_AXIS_HEADER_BYTES {
        return Err(format!(
            "UCSF axis[{axis_idx}] header shorter than {UCSF_AXIS_HEADER_BYTES} bytes"
        ));
    }
    let nuc_slice = &ax_hdr[0..6];
    let nuc_end = nuc_slice.iter().position(|&b| b == 0).unwrap_or(6);
    let raw_label = String::from_utf8_lossy(&nuc_slice[..nuc_end])
        .trim()
        .to_string();
    let label = if raw_label.is_empty() {
        format!("W{}", axis_idx + 1)
    } else {
        raw_label
    };

    let npoints_i32 = i32::from_be_bytes(ax_hdr[8..12].try_into().unwrap());
    if npoints_i32 <= 0 {
        return Err(format!(
            "invalid UCSF axis[{axis_idx}] npoints={npoints_i32}"
        ));
    }
    let npoints = npoints_i32 as usize;

    let tile_i32 = i32::from_be_bytes(ax_hdr[16..20].try_into().unwrap());
    if tile_i32 <= 0 {
        return Err(format!(
            "invalid UCSF axis[{axis_idx}] tile_size={tile_i32}"
        ));
    }
    let tile_size = tile_i32 as usize;

    let obs_mhz = f32::from_be_bytes(ax_hdr[20..24].try_into().unwrap()) as f64;
    let sw_hz = f32::from_be_bytes(ax_hdr[24..28].try_into().unwrap()) as f64;
    let xmtr_ppm = f32::from_be_bytes(ax_hdr[28..32].try_into().unwrap()) as f64;

    if !obs_mhz.is_finite() || obs_mhz <= 0.0 {
        return Err(format!(
            "invalid UCSF axis[{axis_idx}] spectrometer_mhz={obs_mhz}"
        ));
    }
    if !sw_hz.is_finite() || sw_hz <= 0.0 {
        return Err(format!(
            "invalid UCSF axis[{axis_idx}] spectral_width_hz={sw_hz}"
        ));
    }
    if !xmtr_ppm.is_finite() {
        return Err(format!("invalid UCSF axis[{axis_idx}] xmtr_ppm={xmtr_ppm}"));
    }

    // In UCSF v2, byte offset 28..32 stores the carrier/center chemical shift (xmtr_ppm),
    // so index 0 (high-ppm edge) sits at xmtr_ppm + 0.5 * (sw_hz / obs_mhz) with step -(sw_hz / obs_mhz) / npoints.
    let sw_ppm = sw_hz / obs_mhz;
    let first_ppm = xmtr_ppm + 0.5 * sw_ppm;
    let step_ppm = -sw_ppm / (npoints as f64);

    Ok((
        AxisPpm {
            label,
            size: npoints,
            first_ppm,
            step_ppm,
            obs_mhz,
        },
        tile_size,
    ))
}

fn encode_ucsf_file_header(ndim: u8) -> [u8; UCSF_FILE_HEADER_BYTES] {
    let mut hdr = [0u8; UCSF_FILE_HEADER_BYTES];
    hdr[..8].copy_from_slice(UCSF_MAGIC_PREFIX);
    hdr[8] = 0;
    hdr[9] = 0;
    hdr[10] = ndim;
    hdr[11] = 1; // ncomponents = 1 (real)
    hdr[12] = 0; // encoding = 0 (IEEE 754 float32)
    hdr[13] = 2; // format version = 2
    let comment = b"nmrassign UCSF v2 spectrum";
    hdr[46..46 + comment.len()].copy_from_slice(comment);
    hdr
}

fn encode_ucsf_axis_header(ax: &AxisPpm, tile_size: usize) -> [u8; UCSF_AXIS_HEADER_BYTES] {
    let mut buf = [0u8; UCSF_AXIS_HEADER_BYTES];
    let lbl_bytes = ax.label.as_bytes();
    let copy_len = lbl_bytes.len().min(5);
    buf[..copy_len].copy_from_slice(&lbl_bytes[..copy_len]);

    let npts = (ax.size.max(1)) as i32;
    let tsize = (tile_size.max(1)) as i32;
    let obs_mhz = (ax.obs_mhz.max(1e-3)) as f32;
    let sw_ppm = ax.sw_ppm();
    let sw_hz = (sw_ppm * ax.obs_mhz.max(1e-3)) as f32;
    let xmtr_ppm = (ax.first_ppm - 0.5 * sw_ppm) as f32;

    buf[8..12].copy_from_slice(&npts.to_be_bytes());
    buf[12..16].copy_from_slice(&npts.to_be_bytes());
    buf[16..20].copy_from_slice(&tsize.to_be_bytes());
    buf[20..24].copy_from_slice(&obs_mhz.to_be_bytes());
    buf[24..28].copy_from_slice(&sw_hz.to_be_bytes());
    buf[28..32].copy_from_slice(&xmtr_ppm.to_be_bytes());
    buf
}

/// Read a 2D UCSF Sparky spectrum (`.ucsf`), mapping `1H` to `axis_x` and the indirect
/// heteronucleus (`15N` / `13C`) to `axis_y`.
pub fn read_ucsf_2d(path: impl AsRef<Path>) -> Result<Spectrum2D, String> {
    let p = path.as_ref();
    let file = File::open(p).map_err(|e| format!("failed to open {}: {e}", p.display()))?;
    let mut reader = BufReader::with_capacity(256 * 1024, file);

    let mut hdr = [0u8; UCSF_FILE_HEADER_BYTES + 2 * UCSF_AXIS_HEADER_BYTES];
    reader
        .read_exact(&mut hdr)
        .map_err(|e| format!("failed to read UCSF 2D header from {}: {e}", p.display()))?;
    parse_ucsf_file_header(&hdr[..UCSF_FILE_HEADER_BYTES], 2)?;

    let (ax0, t0) = parse_ucsf_axis_header(
        &hdr[UCSF_FILE_HEADER_BYTES..UCSF_FILE_HEADER_BYTES + UCSF_AXIS_HEADER_BYTES],
        0,
    )?;
    let (ax1, t1) =
        parse_ucsf_axis_header(&hdr[UCSF_FILE_HEADER_BYTES + UCSF_AXIS_HEADER_BYTES..], 1)?;

    let n0 = ax0.size;
    let n1 = ax1.size;
    let nt0 = n0.div_ceil(t0);
    let nt1 = n1.div_ceil(t1);

    // Standard Sparky 2D layout has axis[0] = indirect (15N/13C) and axis[1] = direct (1H).
    // Transpose into Spectrum2D(X=1H, Y=indirect) if axis[0] is 1H and axis[1] is heteronuclear.
    let transpose =
        classify_ucsf_nucleus(&ax0) == UcsfNuc::H && classify_ucsf_nucleus(&ax1) != UcsfNuc::H;
    let (axis_x, axis_y) = if transpose { (ax0, ax1) } else { (ax1, ax0) };
    let nx = axis_x.size;
    let ny = axis_y.size;

    let mut data = vec![0.0f32; ny * nx];
    let mut tile_bytes = vec![0u8; t0 * t1 * 4];

    for bt0 in 0..nt0 {
        let base0 = bt0 * t0;
        let valid0 = (n0 - base0).min(t0);
        for bt1 in 0..nt1 {
            let base1 = bt1 * t1;
            let valid1 = (n1 - base1).min(t1);
            reader.read_exact(&mut tile_bytes).map_err(|e| {
                format!(
                    "UCSF 2D tile ({bt0},{bt1}) truncated in {}: {e}",
                    p.display()
                )
            })?;
            if !transpose {
                for l0 in 0..valid0 {
                    let gy = base0 + l0;
                    let dst_row = &mut data[gy * nx + base1..gy * nx + base1 + valid1];
                    let src_off = l0 * t1 * 4;
                    let src_slice = &tile_bytes[src_off..src_off + valid1 * 4];
                    for (dst, chunk) in dst_row.iter_mut().zip(src_slice.chunks_exact(4)) {
                        *dst = f32::from_be_bytes(chunk.try_into().unwrap());
                    }
                }
            } else {
                for l0 in 0..valid0 {
                    let gx = base0 + l0;
                    let src_off = l0 * t1 * 4;
                    let src_slice = &tile_bytes[src_off..src_off + valid1 * 4];
                    for (l1, chunk) in src_slice.chunks_exact(4).enumerate() {
                        let gy = base1 + l1;
                        data[gy * nx + gx] = f32::from_be_bytes(chunk.try_into().unwrap());
                    }
                }
            }
        }
    }

    Ok(Spectrum2D {
        name: p
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("spectrum.ucsf")
            .to_string(),
        axis_x,
        axis_y,
        data,
    })
}

/// Write a 2D UCSF Sparky spectrum (`.ucsf`) using standard tile sizes (`64 x 256` capped to axis size).
pub fn write_ucsf_2d(path: impl AsRef<Path>, spec: &Spectrum2D) -> Result<(), String> {
    let ty = spec.ny().clamp(1, 64);
    let tx = spec.nx().clamp(1, 256);
    write_ucsf_2d_tiled(path, spec, ty, tx)
}

/// Write a 2D UCSF Sparky spectrum (`.ucsf`) with explicit `(tile_y, tile_x)` dimensions,
/// zero-padding partial boundary tiles per the UCSF v2 format specification.
pub fn write_ucsf_2d_tiled(
    path: impl AsRef<Path>,
    spec: &Spectrum2D,
    tile_y: usize,
    tile_x: usize,
) -> Result<(), String> {
    let p = path.as_ref();
    let file = File::create(p).map_err(|e| format!("failed to create {}: {e}", p.display()))?;
    let mut writer = BufWriter::with_capacity(256 * 1024, file);

    let t0 = tile_y.max(1);
    let t1 = tile_x.max(1);
    writer
        .write_all(&encode_ucsf_file_header(2))
        .map_err(|e| format!("write error {}: {e}", p.display()))?;
    // UCSF convention: axis[0] = indirect (axis_y), axis[1] = direct (axis_x)
    writer
        .write_all(&encode_ucsf_axis_header(&spec.axis_y, t0))
        .map_err(|e| format!("write error {}: {e}", p.display()))?;
    writer
        .write_all(&encode_ucsf_axis_header(&spec.axis_x, t1))
        .map_err(|e| format!("write error {}: {e}", p.display()))?;

    let ny = spec.ny();
    let nx = spec.nx();
    let nt0 = ny.div_ceil(t0);
    let nt1 = nx.div_ceil(t1);
    let mut tile_bytes = vec![0u8; t0 * t1 * 4];

    for bt0 in 0..nt0 {
        let base0 = bt0 * t0;
        for bt1 in 0..nt1 {
            let base1 = bt1 * t1;
            tile_bytes.fill(0);
            let valid0 = (ny.saturating_sub(base0)).min(t0);
            let valid1 = (nx.saturating_sub(base1)).min(t1);
            for l0 in 0..valid0 {
                let gy = base0 + l0;
                let row_off = l0 * t1 * 4;
                for l1 in 0..valid1 {
                    let gx = base1 + l1;
                    let val = spec.data[gy * nx + gx];
                    let dst = row_off + l1 * 4;
                    tile_bytes[dst..dst + 4].copy_from_slice(&val.to_be_bytes());
                }
            }
            writer
                .write_all(&tile_bytes)
                .map_err(|e| format!("write error {}: {e}", p.display()))?;
        }
    }
    writer
        .flush()
        .map_err(|e| format!("flush error {}: {e}", p.display()))
}

fn choose_ucsf_3d_axis_perm(axes: &[AxisPpm; 3]) -> (usize, usize, usize) {
    let nucs = [
        classify_ucsf_nucleus(&axes[0]),
        classify_ucsf_nucleus(&axes[1]),
        classify_ucsf_nucleus(&axes[2]),
    ];
    let h_dims: Vec<usize> = (0..3).filter(|&i| nucs[i] == UcsfNuc::H).collect();
    let n_dims: Vec<usize> = (0..3).filter(|&i| nucs[i] == UcsfNuc::N).collect();
    let c_dims: Vec<usize> = (0..3).filter(|&i| nucs[i] == UcsfNuc::C).collect();

    if h_dims.len() == 1 && n_dims.len() == 1 && c_dims.len() == 1 {
        // Triple-resonance (1H, 15N, 13C) -> Spectrum3D(X=1H, Y=15N, Z=13C)
        (h_dims[0], n_dims[0], c_dims[0])
    } else if h_dims.len() == 2 && (n_dims.len() == 1 || c_dims.len() == 1) {
        // 3D NOESY / HCCCONH / HCCH-TOCSY -> X = direct 1H (highest axis index), Y = heavy (15N/13C), Z = indirect 1H
        let heavy = if n_dims.len() == 1 {
            n_dims[0]
        } else {
            c_dims[0]
        };
        let x_dim = *h_dims.iter().max().unwrap();
        let z_dim = *h_dims.iter().min().unwrap();
        (x_dim, heavy, z_dim)
    } else if h_dims.len() == 1 {
        // One 1H axis + two heavy axes (e.g. CCCONH or custom 3D) -> X = 1H, Y/Z preserve indirect order
        let x_dim = h_dims[0];
        let others: Vec<usize> = (0..3).filter(|&i| i != x_dim).collect();
        (x_dim, others[1], others[0])
    } else {
        (2, 1, 0)
    }
}

/// Read a 3D UCSF Sparky spectrum (`.ucsf`), assembling big-endian `f32` tiles (including
/// zero-padded partial boundary tiles) and ordering dimensions into `Spectrum3D(X=1H, Y=15N, Z=13C)`.
pub fn read_ucsf_3d(path: impl AsRef<Path>, exp: &str) -> Result<Spectrum3D, String> {
    let p = path.as_ref();
    let file = File::open(p).map_err(|e| format!("failed to open {}: {e}", p.display()))?;
    let mut reader = BufReader::with_capacity(1024 * 1024, file);

    let mut hdr = [0u8; UCSF_FILE_HEADER_BYTES + 3 * UCSF_AXIS_HEADER_BYTES];
    reader
        .read_exact(&mut hdr)
        .map_err(|e| format!("failed to read UCSF 3D header from {}: {e}", p.display()))?;
    parse_ucsf_file_header(&hdr[..UCSF_FILE_HEADER_BYTES], 3)?;

    let (ax0, t0) = parse_ucsf_axis_header(
        &hdr[UCSF_FILE_HEADER_BYTES..UCSF_FILE_HEADER_BYTES + UCSF_AXIS_HEADER_BYTES],
        0,
    )?;
    let (ax1, t1) = parse_ucsf_axis_header(
        &hdr[UCSF_FILE_HEADER_BYTES + UCSF_AXIS_HEADER_BYTES
            ..UCSF_FILE_HEADER_BYTES + 2 * UCSF_AXIS_HEADER_BYTES],
        1,
    )?;
    let (ax2, t2) = parse_ucsf_axis_header(
        &hdr[UCSF_FILE_HEADER_BYTES + 2 * UCSF_AXIS_HEADER_BYTES..],
        2,
    )?;

    let raw_axes = [ax0, ax1, ax2];
    let (x_dim, y_dim, z_dim) = choose_ucsf_3d_axis_perm(&raw_axes);
    let axis_x = raw_axes[x_dim].clone();
    let axis_y = raw_axes[y_dim].clone();
    let axis_z = raw_axes[z_dim].clone();

    let (n0, n1, n2) = (raw_axes[0].size, raw_axes[1].size, raw_axes[2].size);
    let (nt0, nt1, nt2) = (n0.div_ceil(t0), n1.div_ceil(t1), n2.div_ceil(t2));
    let (nx, ny, nz) = (axis_x.size, axis_y.size, axis_z.size);

    let mut data = vec![0.0f32; nz * ny * nx];
    let mut tile_bytes = vec![0u8; t0 * t1 * t2 * 4];
    let direct_zyx = (x_dim, y_dim, z_dim) == (2, 1, 0);

    for bt0 in 0..nt0 {
        let base0 = bt0 * t0;
        let valid0 = (n0 - base0).min(t0);
        for bt1 in 0..nt1 {
            let base1 = bt1 * t1;
            let valid1 = (n1 - base1).min(t1);
            for bt2 in 0..nt2 {
                let base2 = bt2 * t2;
                let valid2 = (n2 - base2).min(t2);
                reader.read_exact(&mut tile_bytes).map_err(|e| {
                    format!(
                        "UCSF 3D tile ({bt0},{bt1},{bt2}) truncated in {}: {e}",
                        p.display()
                    )
                })?;

                if direct_zyx {
                    for l0 in 0..valid0 {
                        let gz = base0 + l0;
                        for l1 in 0..valid1 {
                            let gy = base1 + l1;
                            let dst_start = (gz * ny + gy) * nx + base2;
                            let dst_row = &mut data[dst_start..dst_start + valid2];
                            let src_off = (l0 * t1 + l1) * t2 * 4;
                            let src_slice = &tile_bytes[src_off..src_off + valid2 * 4];
                            for (dst, chunk) in dst_row.iter_mut().zip(src_slice.chunks_exact(4)) {
                                *dst = f32::from_be_bytes(chunk.try_into().unwrap());
                            }
                        }
                    }
                } else {
                    for l0 in 0..valid0 {
                        let g0 = base0 + l0;
                        for l1 in 0..valid1 {
                            let g1 = base1 + l1;
                            let src_off = (l0 * t1 + l1) * t2 * 4;
                            let src_slice = &tile_bytes[src_off..src_off + valid2 * 4];
                            for (l2, chunk) in src_slice.chunks_exact(4).enumerate() {
                                let coords = [g0, g1, base2 + l2];
                                let gx = coords[x_dim];
                                let gy = coords[y_dim];
                                let gz = coords[z_dim];
                                data[(gz * ny + gy) * nx + gx] =
                                    f32::from_be_bytes(chunk.try_into().unwrap());
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(Spectrum3D {
        name: p
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("spectrum3d.ucsf")
            .to_string(),
        exp: exp.to_string(),
        axis_x,
        axis_y,
        axis_z,
        data,
    })
}

/// Write a 3D UCSF Sparky spectrum (`.ucsf`) using standard tile sizes (`32 x 8 x 128` capped to axis size).
pub fn write_ucsf_3d(path: impl AsRef<Path>, spec: &Spectrum3D) -> Result<(), String> {
    let tz = spec.nz().clamp(1, 32);
    let ty = spec.ny().clamp(1, 8);
    let tx = spec.nx().clamp(1, 128);
    write_ucsf_3d_tiled(path, spec, tz, ty, tx)
}

/// Write a 3D UCSF Sparky spectrum (`.ucsf`) with explicit `(tile_z, tile_y, tile_x)` dimensions,
/// zero-padding partial boundary tiles per the UCSF v2 format specification.
pub fn write_ucsf_3d_tiled(
    path: impl AsRef<Path>,
    spec: &Spectrum3D,
    tile_z: usize,
    tile_y: usize,
    tile_x: usize,
) -> Result<(), String> {
    let p = path.as_ref();
    let file = File::create(p).map_err(|e| format!("failed to create {}: {e}", p.display()))?;
    let mut writer = BufWriter::with_capacity(1024 * 1024, file);

    let (t0, t1, t2) = (tile_z.max(1), tile_y.max(1), tile_x.max(1));
    writer
        .write_all(&encode_ucsf_file_header(3))
        .map_err(|e| format!("write error {}: {e}", p.display()))?;
    // UCSF convention: axis[0] = Z (13C), axis[1] = Y (15N), axis[2] = X (1H)
    writer
        .write_all(&encode_ucsf_axis_header(&spec.axis_z, t0))
        .map_err(|e| format!("write error {}: {e}", p.display()))?;
    writer
        .write_all(&encode_ucsf_axis_header(&spec.axis_y, t1))
        .map_err(|e| format!("write error {}: {e}", p.display()))?;
    writer
        .write_all(&encode_ucsf_axis_header(&spec.axis_x, t2))
        .map_err(|e| format!("write error {}: {e}", p.display()))?;

    let (nz, ny, nx) = (spec.nz(), spec.ny(), spec.nx());
    let (nt0, nt1, nt2) = (nz.div_ceil(t0), ny.div_ceil(t1), nx.div_ceil(t2));
    let mut tile_bytes = vec![0u8; t0 * t1 * t2 * 4];

    for bt0 in 0..nt0 {
        let base0 = bt0 * t0;
        let valid0 = (nz.saturating_sub(base0)).min(t0);
        for bt1 in 0..nt1 {
            let base1 = bt1 * t1;
            let valid1 = (ny.saturating_sub(base1)).min(t1);
            for bt2 in 0..nt2 {
                let base2 = bt2 * t2;
                let valid2 = (nx.saturating_sub(base2)).min(t2);
                tile_bytes.fill(0);
                for l0 in 0..valid0 {
                    let gz = base0 + l0;
                    for l1 in 0..valid1 {
                        let gy = base1 + l1;
                        let dst_row_off = (l0 * t1 + l1) * t2 * 4;
                        let src_row_off = (gz * ny + gy) * nx + base2;
                        for l2 in 0..valid2 {
                            let val = spec.data[src_row_off + l2];
                            let dst = dst_row_off + l2 * 4;
                            tile_bytes[dst..dst + 4].copy_from_slice(&val.to_be_bytes());
                        }
                    }
                }
                writer
                    .write_all(&tile_bytes)
                    .map_err(|e| format!("write error {}: {e}", p.display()))?;
            }
        }
    }
    writer
        .flush()
        .map_err(|e| format!("flush error {}: {e}", p.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::process;

    #[test]
    fn test_ucsf_permuted_axis_orders_and_partial_tiles() {
        let dir = env::temp_dir().join(format!("nmr_ucsf_perm_test_{}", process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        // 1. 2D UCSF stored in transposed order: axis[0] = 1H (nx=13, t0=5), axis[1] = 15N (ny=9, t1=4)
        let ax_h = AxisPpm::from_bounds("1H", 13, 10.2, 6.0, 600.13);
        let ax_n = AxisPpm::from_bounds("15N", 9, 130.0, 106.0, 60.81);
        let (t0, t1) = (5usize, 4usize);
        let p2 = dir.join("transposed_2d.ucsf");
        {
            let mut f = File::create(&p2).unwrap();
            f.write_all(&encode_ucsf_file_header(2)).unwrap();
            f.write_all(&encode_ucsf_axis_header(&ax_h, t0)).unwrap();
            f.write_all(&encode_ucsf_axis_header(&ax_n, t1)).unwrap();
            let nt0 = ax_h.size.div_ceil(t0);
            let nt1 = ax_n.size.div_ceil(t1);
            for bt0 in 0..nt0 {
                for bt1 in 0..nt1 {
                    let mut tile = vec![0u8; t0 * t1 * 4];
                    for l0 in 0..t0 {
                        let gx = bt0 * t0 + l0;
                        for l1 in 0..t1 {
                            let gy = bt1 * t1 + l1;
                            let val = if gx < ax_h.size && gy < ax_n.size {
                                (gy * 100 + gx + 1) as f32
                            } else {
                                -999.0 // boundary padding sentinel must be ignored by reader
                            };
                            let off = (l0 * t1 + l1) * 4;
                            tile[off..off + 4].copy_from_slice(&val.to_be_bytes());
                        }
                    }
                    f.write_all(&tile).unwrap();
                }
            }
        }
        assert!(is_ucsf_file(&p2));
        let s2 = read_ucsf_2d(&p2).unwrap();
        assert_eq!(s2.nx(), 13);
        assert_eq!(s2.ny(), 9);
        assert_eq!(s2.axis_x.label, "1H");
        assert_eq!(s2.axis_y.label, "15N");
        for gy in 0..9 {
            for gx in 0..13 {
                assert_eq!(s2.get(gx, gy), (gy * 100 + gx + 1) as f32);
            }
        }

        // 2. 3D UCSF stored in non-standard order: axis[0] = 1H (nx=7, t0=3), axis[1] = 13C (nz=6, t1=4), axis[2] = 15N (ny=5, t2=2)
        let ax3_h = AxisPpm::from_bounds("1H", 7, 9.8, 7.0, 500.13);
        let ax3_c = AxisPpm::from_bounds("13C", 6, 70.0, 20.0, 125.76);
        let ax3_n = AxisPpm::from_bounds("15N", 5, 128.0, 108.0, 50.68);
        let (u0, u1, u2) = (3usize, 4usize, 2usize);
        let p3 = dir.join("permuted_3d.ucsf");
        {
            let mut f = File::create(&p3).unwrap();
            f.write_all(&encode_ucsf_file_header(3)).unwrap();
            f.write_all(&encode_ucsf_axis_header(&ax3_h, u0)).unwrap();
            f.write_all(&encode_ucsf_axis_header(&ax3_c, u1)).unwrap();
            f.write_all(&encode_ucsf_axis_header(&ax3_n, u2)).unwrap();
            for bt0 in 0..ax3_h.size.div_ceil(u0) {
                for bt1 in 0..ax3_c.size.div_ceil(u1) {
                    for bt2 in 0..ax3_n.size.div_ceil(u2) {
                        let mut tile = vec![0u8; u0 * u1 * u2 * 4];
                        for l0 in 0..u0 {
                            let gx = bt0 * u0 + l0;
                            for l1 in 0..u1 {
                                let gz = bt1 * u1 + l1;
                                for l2 in 0..u2 {
                                    let gy = bt2 * u2 + l2;
                                    let val =
                                        if gx < ax3_h.size && gz < ax3_c.size && gy < ax3_n.size {
                                            (gz * 1000 + gy * 10 + gx + 1) as f32
                                        } else {
                                            -777.0
                                        };
                                    let off = ((l0 * u1 + l1) * u2 + l2) * 4;
                                    tile[off..off + 4].copy_from_slice(&val.to_be_bytes());
                                }
                            }
                        }
                        f.write_all(&tile).unwrap();
                    }
                }
            }
        }
        let s3 = read_ucsf_3d(&p3, "HNCA").unwrap();
        assert_eq!((s3.nx(), s3.ny(), s3.nz()), (7, 5, 6));
        assert_eq!(s3.axis_x.label, "1H");
        assert_eq!(s3.axis_y.label, "15N");
        assert_eq!(s3.axis_z.label, "13C");
        for gz in 0..6 {
            for gy in 0..5 {
                for gx in 0..7 {
                    assert_eq!(s3.get(gx, gy, gz), (gz * 1000 + gy * 10 + gx + 1) as f32);
                }
            }
        }

        let _ = fs::remove_dir_all(&dir);
    }
}
