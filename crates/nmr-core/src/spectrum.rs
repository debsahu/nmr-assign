//! Processed 2D and 3D NMR spectrum representation and Bruker `pdata` / NMRPipe I/O.
//!
//! Provenance: Original clean-room implementation of `docs/specs/01-data-io.md` from public
//! format specifications:
//! - NMRPipe 512-float header specification (Delaglio et al., *J. Biomol. NMR* 6, 277-293, 1995).
//! - Bruker TopSpin JCAMP-DX `procs`/`proc2s`/`proc3s` and submatrix `2rr`/`3rrr` binary layout.

use std::collections::HashMap;
use std::fs;
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
