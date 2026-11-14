use std::f64::consts::PI;
use std::fs;
use std::path::{Path, PathBuf};

use nmr_core::fid::{
    bruker_dsp_group_delay, find_child_ci, read_bruker_fid_1d, read_bruker_fid_2d,
    read_bruker_fid_3d, read_fid_2d, read_fid_2d_opts, read_fid_3d, read_fid_3d_opts,
    read_fid_auto, Complex64, FidVendor, QuadMode, RawFid,
};
use nmr_core::process::{
    process_fid_dir_2d, process_fid_dir_2d_opts, process_fid_dir_3d, process_fid_dir_3d_opts,
};
use nmr_core::spectrum::{read_bruker_pdata_2d, read_bruker_pdata_3d};

fn temp_test_dir(tag: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("nmrassign_fid_test_{tag}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&p);
    fs::create_dir_all(&p).expect("create temp test dir");
    p
}

/// Build a big-endian Varian binary `fid` file with `nblocks` blocks (`ntraces = 1` per block)
/// of `np` `f32` scalar words (`np / 2` complex samples per trace).
fn build_varian_f32_fid(traces: &[Vec<[f32; 2]>]) -> Vec<u8> {
    let nblocks = traces.len() as i32;
    let ntraces: i32 = 1;
    let n_cplx = traces.first().map_or(0, Vec::len);
    let np = (n_cplx * 2) as i32;
    let ebytes: i32 = 4;
    let tbytes: i32 = np * ebytes;
    let bbytes: i32 = 28 + tbytes;
    let status: u16 = 0x0001 | 0x0004 | 0x0008; // S_DATA | S_32 | S_FLOAT

    let mut bin = Vec::with_capacity(32 + (nblocks as usize) * (bbytes as usize));
    bin.extend_from_slice(&nblocks.to_be_bytes());
    bin.extend_from_slice(&ntraces.to_be_bytes());
    bin.extend_from_slice(&np.to_be_bytes());
    bin.extend_from_slice(&ebytes.to_be_bytes());
    bin.extend_from_slice(&tbytes.to_be_bytes());
    bin.extend_from_slice(&bbytes.to_be_bytes());
    bin.extend_from_slice(&1i16.to_be_bytes()); // vers_id
    bin.extend_from_slice(&status.to_be_bytes());
    bin.extend_from_slice(&1i32.to_be_bytes()); // nbheaders

    for (b, tr) in traces.iter().enumerate() {
        assert_eq!(tr.len(), n_cplx);
        bin.extend_from_slice(&0i16.to_be_bytes()); // scale
        bin.extend_from_slice(&status.to_be_bytes()); // status
        bin.extend_from_slice(&((b + 1) as i16).to_be_bytes()); // index
        bin.extend_from_slice(&0i16.to_be_bytes()); // mode
        bin.extend_from_slice(&1i32.to_be_bytes()); // ctcount
        bin.extend_from_slice(&0.0f32.to_be_bytes()); // lpval
        bin.extend_from_slice(&0.0f32.to_be_bytes()); // rpval
        bin.extend_from_slice(&0.0f32.to_be_bytes()); // lvl
        bin.extend_from_slice(&0.0f32.to_be_bytes()); // tlt
        for &[re, im] in tr {
            bin.extend_from_slice(&re.to_be_bytes());
            bin.extend_from_slice(&im.to_be_bytes());
        }
    }
    bin
}

/// In-place radix-2 forward FFT (`sum_{k=0}^{N-1} x[k] exp(-2 pi i k m / N)`) with `fftshift` helper
/// for self-contained real-data quadrature verification against Bruker `pdata`.
fn fft_radix2_forward(buf: &mut [Complex64]) {
    let n = buf.len();
    assert!(
        n.is_power_of_two(),
        "radix-2 FFT requires power of two, got {n}"
    );
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while (j & bit) != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            buf.swap(i, j);
        }
    }
    let mut len = 2usize;
    while len <= n {
        let half = len / 2;
        let ang = -2.0 * PI / (len as f64);
        let wlen = [ang.cos(), ang.sin()];
        for start in (0..n).step_by(len) {
            let mut w = [1.0, 0.0];
            for k in 0..half {
                let u = buf[start + k];
                let v_raw = buf[start + k + half];
                let v = [
                    v_raw[0] * w[0] - v_raw[1] * w[1],
                    v_raw[0] * w[1] + v_raw[1] * w[0],
                ];
                buf[start + k] = [u[0] + v[0], u[1] + v[1]];
                buf[start + k + half] = [u[0] - v[0], u[1] - v[1]];
                w = [
                    w[0] * wlen[0] - w[1] * wlen[1],
                    w[0] * wlen[1] + w[1] * wlen[0],
                ];
            }
        }
        len <<= 1;
    }
}

fn fftshift_in_place(buf: &mut [Complex64]) {
    let half = buf.len() / 2;
    buf.rotate_left(half);
}

fn transform_bruker_direct_trace(
    trace: &[Complex64],
    si: usize,
    group_delay: f64,
    p0_deg: f64,
    p1_deg: f64,
) -> Vec<f64> {
    let n = trace.len();
    let shift_int = group_delay.floor() as usize;
    let frac = group_delay - (shift_int as f64);
    let mut z = vec![[0.0, 0.0]; si];
    for k in 0..n.min(si) {
        let src = trace[(k + shift_int) % n];
        // Conjugate for Bruker DQD (`FT_mod = 6`)
        z[k] = [src[0], -src[1]];
    }
    if z.len() > 1 {
        z[0][0] *= 0.5;
        z[0][1] *= 0.5;
    }
    fft_radix2_forward(&mut z);
    fftshift_in_place(&mut z);

    let mut out = Vec::with_capacity(si);
    for (m, c) in z.iter().enumerate() {
        let f_norm = ((m as f64) / (si as f64)) - 0.5; // [-0.5, +0.5)
        let gd_phase = 2.0 * PI * frac * f_norm;
        let usr_phase = (p0_deg + p1_deg * ((m as f64) / (si as f64))).to_radians();
        let phi = gd_phase + usr_phase;
        let (cp, sp) = (phi.cos(), phi.sin());
        out.push(c[0] * cp - c[1] * sp);
    }
    out
}

#[test]
fn bruker_dsp_lookup_table_and_unknown_rejection() {
    assert!((bruker_dsp_group_delay(10, 2).unwrap() - 44.75).abs() < 1e-9);
    assert!((bruker_dsp_group_delay(10, 4).unwrap() - 66.625).abs() < 1e-9);
    assert!((bruker_dsp_group_delay(11, 8).unwrap() - 53.25).abs() < 1e-9);
    assert!((bruker_dsp_group_delay(12, 2).unwrap() - 46.311).abs() < 1e-9);
    assert!((bruker_dsp_group_delay(12, 16).unwrap() - 71.600).abs() < 1e-9);
    assert!((bruker_dsp_group_delay(13, 4).unwrap() - 2.875).abs() < 1e-9);
    assert_eq!(bruker_dsp_group_delay(0, 1).unwrap(), 0.0);
    // Modern firmware (DSPFVS >= 20) without explicit GRPDLY must error rather than guess
    assert!(bruker_dsp_group_delay(20, 1600).is_err());
}

#[test]
fn bruker_1d_big_endian_float64_and_dspfvs_fallback() {
    let dir = temp_test_dir("bruker_1d_f64");
    let acqus = "\
##TITLE= synthetic 1D
##$TD= 16
##$BYTORDA= 1
##$DTYPA= 2
##$AQ_mod= 3
##$SFO1= 600.132860
##$BF1= 600.130000
##$SW_h= 7200.0
##$NUC1= <1H>
##$GRPDLY= -1
##$DSPFVS= 10
##$DECIM= 4
##$PULPROG= <zgpr>
##END=
";
    fs::write(dir.join("acqus"), acqus).unwrap();

    let mut fid_bytes = Vec::new();
    for k in 0..8 {
        let re = (k as f64) * 1.5;
        let im = -(k as f64) * 0.5;
        fid_bytes.extend_from_slice(&re.to_be_bytes());
        fid_bytes.extend_from_slice(&im.to_be_bytes());
    }
    fs::write(dir.join("fid"), &fid_bytes).unwrap();

    let fid = read_bruker_fid_1d(&dir).expect("read 1D float64 Bruker FID");
    assert_eq!(fid.vendor, FidVendor::Bruker);
    assert_eq!(fid.axis_x.label, "1H");
    assert_eq!(fid.axis_x.complex_points, 8);
    assert!((fid.axis_x.group_delay - 66.625).abs() < 1e-9);
    assert!((fid.data[3][0] - 4.5).abs() < 1e-12);
    assert!((fid.data[3][1] - (-1.5)).abs() < 1e-12);

    // 1D dataset with FnTYPE = 2 (NUS) must error cleanly
    fs::write(
        dir.join("acqus"),
        acqus.replace("##END=", "##$FnTYPE= 2\n##END="),
    )
    .unwrap();
    assert!(read_bruker_fid_1d(&dir).is_err());

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn bruker_2d_ser_padding_echo_antiecho_and_strip_transform_procs() {
    let dir = temp_test_dir("bruker_2d_pad_ea");
    // TD = 300 int32 words (150 complex points = 1200 bytes) -> padded to 512 words (2048 bytes) per trace!
    let acqus = "\
##TITLE= synthetic 2D EA
##$TD= 300
##$BYTORDA= 0
##$DTYPA= 0
##$AQ_mod= 3
##$SFO1= 500.132375
##$BF1= 500.130000
##$SW_h= 6000.0
##$NUC1= <1H>
##$GRPDLY= 67.984
##END=
";
    let acqu2s = "\
##TITLE= synthetic F1
##$TD= 4
##$FnMODE= 6
##$SFO1= 50.683713
##$BF1= 50.677733
##$SW= 36.0
##$NUC1= <15N>
##END=
";
    // TopSpin 4 strip-transform procs (`FTSIZE = 1024 > SI = 512`, `STSR = 128`)
    fs::create_dir_all(dir.join("pdata").join("1")).unwrap();
    let procs = "\
##$SI= 512
##$FTSIZE= 1024
##$STSR= 128
##$STSI= 512
##$SF= 500.130000
##$OFFSET= 10.500000
##$SREF_mod= 1
##END=
";
    fs::write(dir.join("acqus"), acqus).unwrap();
    fs::write(dir.join("acqu2s"), acqu2s).unwrap();
    fs::write(dir.join("pdata").join("1").join("procs"), procs).unwrap();

    let stride_words = 512usize; // 2 * 256 words = 2048 bytes
    let mut ser_bytes = Vec::with_capacity(4 * stride_words * 4);
    for trace_idx in 0..4i32 {
        for w in 0..stride_words {
            let val: i32 = if w < 300 {
                (trace_idx + 1) * 100 + (w as i32)
            } else {
                999_999 // padding garbage that must be ignored
            };
            ser_bytes.extend_from_slice(&val.to_le_bytes());
        }
    }
    fs::write(dir.join("ser"), &ser_bytes).unwrap();

    let fid2 = read_bruker_fid_2d(&dir).expect("read padded 2D ser");
    assert!(fid2.nus_mask.is_none());
    assert_eq!(fid2.axis_x.complex_points, 150);
    assert_eq!(fid2.axis_y.complex_points, 2);
    assert_eq!(fid2.axis_y.quad_mode, QuadMode::EchoAntiecho);
    assert_eq!(fid2.axis_x.stored_proc.si, Some(1024));
    let expected_uncropped_ppm = 10.5 + 128.0 * (6000.0 / 500.13) / 1024.0;
    assert!((fid2.axis_x.first_ppm - expected_uncropped_ppm).abs() < 1e-9);

    // Verified Bruker MC2=5 Echo-Antiecho recombination under conjugate_for_fft=true:
    // z0[0] = [100, 101], z1[0] = [200, 201]
    // cos[0] = [-(101 + 201), 100 + 200] = [-302, 300]
    // sin[0] = [200 - 100, 201 - 101] = [100, 100]
    let cos0 = fid2.trace(0, 0);
    let sin0 = fid2.trace(0, 1);
    assert_eq!(cos0[0], [-302.0, 300.0]);
    assert_eq!(sin0[0], [100.0, 100.0]);

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn bruker_topspin4_uniform_guard_with_automatic_nuslist_and_nustd() {
    let dir = temp_test_dir("bruker_ts4_uniform_guard");
    // Mirrors CISD3_HNCA: FnTYPE=0, NUSLIST=<automatic>, NusAMOUNT=100, TD == NusTD, no nuslist file.
    let acqus = "\
##$TD= 64
##$BYTORDA= 0
##$DTYPA= 0
##$AQ_mod= 3
##$FnTYPE= 0
##$NUSLIST= <automatic>
##$NusAMOUNT= 100
##$AQSEQ= 0
##$TD_INDIRECT= (0..7)
0 6 4 0 0 0 0 0
##$SFO1= 900.214228
##$BF1= 900.210000
##$SW_h= 12500.0
##$NUC1= <1H>
##$GRPDLY= 67.984
##END=
";
    let acqu2s = "\
##$TD= 4
##$NusTD= 4
##$FnMODE= 5
##$SFO1= 91.228
##$BF1= 91.217
##$SW= 35.0
##$NUC1= <15N>
##END=
";
    let acqu3s = "\
##$TD= 6
##$NusTD= 6
##$FnMODE= 5
##$SFO1= 226.369
##$BF1= 226.357
##$SW= 32.0
##$NUC1= <13C>
##END=
";
    fs::write(dir.join("acqus"), acqus).unwrap();
    fs::write(dir.join("acqu2s"), acqu2s).unwrap();
    fs::write(dir.join("acqu3s"), acqu3s).unwrap();
    // 4 * 6 = 24 traces, each 1024-byte padded
    fs::write(dir.join("ser"), vec![0u8; 24 * 1024]).unwrap();

    let fid3 = read_fid_3d(&dir).expect("TopSpin 4 uniform 3D must load as uniform");
    assert!(fid3.nus_mask.is_none());
    assert_eq!(fid3.axis_x.complex_points, 32);
    assert_eq!(fid3.axis_y.complex_points, 2);
    assert_eq!(fid3.axis_z.complex_points, 3);

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn bruker_2d_nus_states_tppi_and_weighted_duplicate_averaging() {
    let dir = temp_test_dir("bruker_2d_nus");
    // Full grid NusTD = 12 (n1_cplx = 6). Schedule has 4 rows: [0, 3, 3, 5] (k1=3 repeated twice).
    // acqu2s.TD = 8 (4 rows * 2 States-TPPI traces).
    let acqus = "\
##$TD= 16
##$BYTORDA= 0
##$DTYPA= 0
##$AQ_mod= 3
##$FnTYPE= 2
##$NUSLIST= <automatic>
##$SFO1= 600.133
##$BF1= 600.130
##$SW_h= 8000.0
##$NUC1= <1H>
##$GRPDLY= 68.0
##END=
";
    let acqu2s = "\
##$TD= 8
##$NusTD= 12
##$FnMODE= 5
##$SFO1= 60.818
##$BF1= 60.810
##$SW= 32.0
##$NUC1= <15N>
##END=
";
    fs::write(dir.join("acqus"), acqus).unwrap();
    fs::write(dir.join("acqu2s"), acqu2s).unwrap();
    fs::write(
        dir.join("nuslist"),
        "# 2D weighted NUS schedule\n0\n3 ; duplicate 1\n3 ; duplicate 2\n5\n",
    )
    .unwrap();

    // 8 traces, each padded to 1024 bytes (256 i32 words)
    let mut ser_bytes = Vec::with_capacity(8 * 1024);
    // Row 0 -> k1=0 (even, sign +1): cos=[10, -1], sin=[20, -2]
    // Row 1 -> k1=3 (odd, sign -1):  raw cos=[100, 0], sin=[200, 0]
    // Row 2 -> k1=3 (odd, sign -1):  raw cos=[300, 0], sin=[400, 0]  => avg raw=[200,0]/[300,0] => States-TPPI=[-200,0]/[-300,0]
    // Row 3 -> k1=5 (odd, sign -1):  raw cos=[50, 5], sin=[60, 6]    => States-TPPI=[-50,-5]/[-60,-6]
    let raw_pairs: [([i32; 2], [i32; 2]); 4] = [
        ([10, -1], [20, -2]),
        ([100, 0], [200, 0]),
        ([300, 0], [400, 0]),
        ([50, 5], [60, 6]),
    ];
    for (c_pt, s_pt) in raw_pairs {
        for pt in [c_pt, s_pt] {
            let mut block = vec![0u8; 1024];
            block[0..4].copy_from_slice(&pt[0].to_le_bytes());
            block[4..8].copy_from_slice(&pt[1].to_le_bytes());
            ser_bytes.extend_from_slice(&block);
        }
    }
    fs::write(dir.join("ser"), &ser_bytes).unwrap();

    let fid2 = read_fid_2d(&dir).expect("read Bruker 2D NUS");
    assert_eq!(fid2.axis_y.td_raw, 12);
    assert_eq!(fid2.axis_y.complex_points, 6);
    assert_eq!(
        fid2.nus_mask,
        Some(vec![true, false, false, true, false, true])
    );

    assert_eq!(fid2.trace(0, 0)[0], [10.0, -1.0]);
    assert_eq!(fid2.trace(0, 1)[0], [20.0, -2.0]);
    // Unsampled k1=1, 2, 4 must be strictly zero
    for unsampled in [1usize, 2, 4] {
        assert_eq!(fid2.trace(unsampled, 0)[0], [0.0, 0.0]);
        assert_eq!(fid2.trace(unsampled, 1)[0], [0.0, 0.0]);
    }
    // Averaged duplicate k1=3 with (-1)^3 = -1 States-TPPI parity
    assert_eq!(fid2.trace(3, 0)[0], [-200.0, 0.0]);
    assert_eq!(fid2.trace(3, 1)[0], [-300.0, 0.0]);
    // k1=5 with (-1)^5 = -1 States-TPPI parity
    assert_eq!(fid2.trace(5, 0)[0], [-50.0, -5.0]);
    assert_eq!(fid2.trace(5, 1)[0], [-60.0, -6.0]);

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn bruker_3d_nus_nustd_flattening_and_aqseq_321_vs_312() {
    for aqseq in [0i32, 1i32] {
        let dir = temp_test_dir(&format!("bruker_3d_nus_aqseq_{aqseq}"));
        // Full grid: F2 (`acqu2s`) NusTD = 8 (`n1_cplx = 4`), F1 (`acqu3s`) NusTD = 10 (`n2_cplx = 5`).
        // TopSpin flattened NUS acquisition: acqu2s.TD = 12 (`3` schedule rows * `4` traces), acqu3s.TD = 1.
        let acqus = format!(
            "\
##$TD= 16
##$BYTORDA= 0
##$DTYPA= 0
##$AQ_mod= 3
##$FnTYPE= 2
##$AQSEQ= {aqseq}
##$SFO1= 900.214
##$BF1= 900.210
##$SW_h= 12500.0
##$NUC1= <1H>
##$GRPDLY= 67.984
##END=
"
        );
        let acqu2s = "\
##$TD= 12
##$NusTD= 8
##$FnMODE= 5
##$SFO1= 91.228
##$BF1= 91.217
##$SW= 35.0
##$NUC1= <15N>
##END=
";
        let acqu3s = "\
##$TD= 1
##$NusTD= 10
##$FnMODE= 5
##$SFO1= 226.369
##$BF1= 226.357
##$SW= 65.0
##$NUC1= <13C>
##END=
";
        fs::write(dir.join("acqus"), acqus).unwrap();
        fs::write(dir.join("acqu2s"), acqu2s).unwrap();
        fs::write(dir.join("acqu3s"), acqu3s).unwrap();

        // Sampled (k1_y, k2_z): (1, 2), (3, 1), and repeated (1, 2) for duplicate averaging!
        // For AQSEQ=0 (3-2-1), nuslist is `k1_f2 k2_f1`. For AQSEQ=1 (3-1-2), nuslist is `k2_f1 k1_f2`.
        let nuslist = if aqseq == 0 {
            "1 2\n3 1\n1 2\n"
        } else {
            "2 1\n1 3\n2 1\n"
        };
        fs::write(dir.join("nuslist"), nuslist).unwrap();

        // Build 3 rows x 4 sub-traces = 12 padded 1024-byte traces.
        // Logical `(iy_sub, iz_sub)` values for each row:
        // Row 0 `(k1=1, k2=2)` -> (-1)^(1+2) = -1: raw (0,0)=10, (1,0)=20, (0,1)=30, (1,1)=40
        // Row 1 `(k1=3, k2=1)` -> (-1)^(3+1) = +1: raw (0,0)=7,  (1,0)=8,  (0,1)=9,  (1,1)=11
        // Row 2 `(k1=1, k2=2)` -> (-1)^(1+2) = -1: raw (0,0)=30, (1,0)=40, (0,1)=50, (1,1)=60
        // Average at `(k1=1, k2=2)` raw: (20, 30, 40, 50) -> after States-TPPI (-1): (-20, -30, -40, -50).
        let row_sub_vals: [[[i32; 2]; 2]; 3] = [
            [[10, 20], [30, 40]], // indexed [iz_sub][iy_sub]
            [[7, 8], [9, 11]],
            [[30, 40], [50, 60]],
        ];

        let mut ser_bytes = Vec::with_capacity(12 * 1024);
        for row in &row_sub_vals {
            for sub in 0..4usize {
                let (iy_sub, iz_sub) = if aqseq == 0 {
                    // F2 (`iy_sub`) inner, F1 (`iz_sub`) outer
                    (sub % 2, sub / 2)
                } else {
                    // F1 (`iz_sub`) inner, F2 (`iy_sub`) outer
                    (sub / 2, sub % 2)
                };
                let re = row[iz_sub][iy_sub];
                let mut block = vec![0u8; 1024];
                block[0..4].copy_from_slice(&re.to_le_bytes());
                ser_bytes.extend_from_slice(&block);
            }
        }
        fs::write(dir.join("ser"), &ser_bytes).unwrap();

        let fid3 = read_bruker_fid_3d(&dir).expect("read Bruker 3D NUS");
        assert_eq!(fid3.axis_y.complex_points, 4);
        assert_eq!(fid3.axis_z.complex_points, 5);
        let mask = fid3.nus_mask.as_ref().expect("NUS mask present");
        assert_eq!(mask.len(), 5 * 4);
        assert_eq!(mask.iter().filter(|&&b| b).count(), 2);
        assert!(mask[2 * 4 + 1]); // (k2=2, k1=1)
        assert!(mask[4 + 3]); // (k2=1, k1=3)
        assert!(!mask[0]);

        // Verify `(k2=2, k1=1)` averaged & negated (`(-1)^(1+2) = -1`):
        // trace(k2, c2, k1, c1) -> c2 corresponds to iz_sub, c1 corresponds to iy_sub
        assert_eq!(fid3.trace(2, 0, 1, 0)[0], [-20.0, 0.0]);
        assert_eq!(fid3.trace(2, 0, 1, 1)[0], [-30.0, 0.0]);
        assert_eq!(fid3.trace(2, 1, 1, 0)[0], [-40.0, 0.0]);
        assert_eq!(fid3.trace(2, 1, 1, 1)[0], [-50.0, 0.0]);

        // Verify `(k2=1, k1=3)` (`(-1)^(3+1) = +1`):
        assert_eq!(fid3.trace(1, 0, 3, 0)[0], [7.0, 0.0]);
        assert_eq!(fid3.trace(1, 0, 3, 1)[0], [8.0, 0.0]);
        assert_eq!(fid3.trace(1, 1, 3, 0)[0], [9.0, 0.0]);
        assert_eq!(fid3.trace(1, 1, 3, 1)[0], [11.0, 0.0]);

        let _ = fs::remove_dir_all(&dir);
    }
}

#[test]
fn bruker_error_cases_unsupported_fnmode_fntype_and_invalid_nuslist() {
    let dir = temp_test_dir("bruker_errors");
    let acqus_base = "\
##$TD= 64
##$BYTORDA= 0
##$DTYPA= 0
##$AQ_mod= 3
##$SFO1= 500.132
##$BF1= 500.130
##$SW_h= 6000.0
##$GRPDLY= 68.0
##END=
";
    // 1. Unsupported FnMODE = 0, 2 (QSEQ), 3 (TPPI) must error rather than guess
    for bad_fnmode in [0, 2, 3] {
        let acqu2s_bad = format!(
            "\
##$TD= 8
##$FnMODE= {bad_fnmode}
##$SFO1= 50.68
##$BF1= 50.67
##$SW= 30.0
##END=
"
        );
        fs::write(dir.join("acqus"), acqus_base).unwrap();
        fs::write(dir.join("acqu2s"), acqu2s_bad).unwrap();
        fs::write(dir.join("ser"), vec![0u8; 8 * 1024]).unwrap();
        let err = read_bruker_fid_2d(&dir).unwrap_err();
        assert!(
            err.contains(&format!("FnMODE={bad_fnmode}")),
            "unexpected error for FnMODE={bad_fnmode}: {err}"
        );
    }

    // 2. Unsupported FnTYPE = 1 (projection) must error
    fs::write(
        dir.join("acqus"),
        acqus_base.replace("##END=", "##$FnTYPE= 1\n##END="),
    )
    .unwrap();
    let acqu2s_nus = "\
##$TD= 4
##$NusTD= 8
##$FnMODE= 5
##$SFO1= 50.68
##$BF1= 50.67
##$SW= 30.0
##END=
";
    fs::write(dir.join("acqu2s"), acqu2s_nus).unwrap();
    assert!(read_bruker_fid_2d(&dir).unwrap_err().contains("FnTYPE=1"));

    // 3. FnTYPE = 2 (NUS) with missing nuslist must error
    fs::write(
        dir.join("acqus"),
        acqus_base.replace("##END=", "##$FnTYPE= 2\n##END="),
    )
    .unwrap();
    fs::write(dir.join("ser"), vec![0u8; 4 * 1024]).unwrap();
    let missing_err = read_bruker_fid_2d(&dir).unwrap_err();
    assert!(
        missing_err.contains("nuslist"),
        "expected missing nuslist error, got: {missing_err}"
    );

    // 4. Out-of-bounds nuslist coordinate (`k1 = 4` when `n1_cplx = 4`) must error
    fs::write(dir.join("nuslist"), "0\n4\n").unwrap();
    let oob_err = read_bruker_fid_2d(&dir).unwrap_err();
    assert!(
        oob_err.contains("out-of-bounds"),
        "expected out-of-bounds error, got: {oob_err}"
    );

    // 5. Trace count mismatch (`nuslist` has 3 rows = 6 traces, `acqu2s.TD = 4`) must error
    fs::write(dir.join("nuslist"), "0\n1\n3\n").unwrap();
    let mismatch_err = read_bruker_fid_2d(&dir).unwrap_err();
    assert!(
        mismatch_err.contains("mismatch"),
        "expected mismatch error, got: {mismatch_err}"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn varian_2d_and_3d_uniform_nus_fcoef_and_error_handling() {
    // --- Part A: Varian 2D uniform + 2D NUS (`sampling = "sparse"` + `sampling.sch`) ---
    let dir2 = temp_test_dir("varian_2d_nus");
    let procpar_2d = "\
sfrq 1 1 1000 0 0 2 1 1 1 1
1 599.7429
0
dfrq2 1 1 1000 0 0 2 1 1 1 2
1 60.7782
0
sw 1 1 100000 0 0 2 1 1 1 3
1 8000.0
0
sw1 1 1 100000 0 0 2 1 1 1 4
1 2000.0
0
np 1 1 65536 2 2 2 1 1 1 5
1 16
0
ni 1 1 4096 0 1 2 1 1 1 6
1 5
0
tn 2 2 8 0 0 2 1 1 1 7
1 \"H1\"
0
dn2 2 2 8 0 0 2 1 1 1 8
1 \"N15\"
0
axis 2 2 8 0 0 2 1 1 1 9
1 \"p2\"
0
phase 1 1 4 1 1 2 1 1 1 10
2 1 2
0
rfl 1 1 100000 -100000 0 2 1 1 1 11
1 4000.0
0
rfp 1 1 100000 -100000 0 2 1 1 1 12
1 2860.0
0
reffrq 1 1 1000 0 0 2 1 1 1 13
1 599.740
0
";
    fs::write(dir2.join("procpar"), procpar_2d).unwrap();

    // Uniform 2D: 2 * ni = 10 traces of 8 complex points
    let uniform_traces: Vec<Vec<[f32; 2]>> = (0..10)
        .map(|b| {
            (0..8)
                .map(|k| [(b as f32) * 10.0 + (k as f32), -(k as f32)])
                .collect()
        })
        .collect();
    fs::write(dir2.join("fid"), build_varian_f32_fid(&uniform_traces)).unwrap();

    let u2 = read_fid_2d(&dir2).expect("read Varian 2D uniform");
    assert_eq!(u2.vendor, FidVendor::Varian);
    assert!(u2.nus_mask.is_none());
    assert!(!u2.axis_x.conjugate_for_fft);
    assert_eq!(u2.axis_x.complex_points, 8);
    assert_eq!(u2.axis_y.complex_points, 5);
    assert_eq!(u2.trace(1, 0)[2], [22.0, -2.0]);
    assert_eq!(u2.trace(1, 1)[2], [32.0, -2.0]);

    // Standard BioPack uniform `sampling = "s"` with leftover `nimax = 64 > ni = 5` must stay uniform (ni=5)!
    let procpar_2d_samp_s = format!(
        "{procpar_2d}\nsampling 2 2 8 0 0 2 1 1 1 14\n1 \"s\"\n0\nnimax 1 1 4096 0 1 2 1 1 1 15\n1 64\n0\n"
    );
    fs::write(dir2.join("procpar"), &procpar_2d_samp_s).unwrap();
    let u2_s = read_fid_2d(&dir2).expect("Varian sampling=\"s\" must decode as uniform");
    assert!(u2_s.nus_mask.is_none());
    assert_eq!(u2_s.axis_y.complex_points, 5);

    // Convert to Varian 2D NUS: `sampling = "sparse"`, `ni = 5`, `sampling.sch` has 3 rows `[0, 2, 4]` -> 6 traces
    let procpar_2d_nus = format!("{procpar_2d}\nsampling 2 2 8 0 0 2 1 1 1 14\n1 \"sparse\"\n0\n");
    fs::write(dir2.join("procpar"), &procpar_2d_nus).unwrap();
    fs::write(dir2.join("fid"), build_varian_f32_fid(&uniform_traces[..6])).unwrap();

    // Missing sampling.sch must error
    assert!(read_fid_2d(&dir2).is_err());

    // Valid sampling.sch -> reconstructs sparse hypercomplex grid with nus_mask
    fs::write(dir2.join("sampling.sch"), "0\n2\n4\n").unwrap();
    let n2 = read_fid_2d(&dir2).expect("read Varian 2D NUS");
    assert_eq!(n2.nus_mask, Some(vec![true, false, true, false, true]));
    assert_eq!(n2.trace(0, 0)[1], [1.0, -1.0]);
    assert_eq!(n2.trace(1, 0)[1], [0.0, 0.0]);
    assert_eq!(n2.trace(2, 0)[1], [21.0, -1.0]);
    assert_eq!(n2.trace(4, 1)[1], [51.0, -1.0]);

    // Unsupported phase array `[1, 2, 3]` must error cleanly
    fs::write(
        dir2.join("procpar"),
        procpar_2d_nus.replace("2 1 2", "3 1 2 3"),
    )
    .unwrap();
    assert!(read_fid_2d(&dir2).is_err());
    let _ = fs::remove_dir_all(&dir2);

    // --- Part B: Varian 3D NUS (`sparse = "y"`, `fcoef` Echo-Antiecho, `array = "phase,phase2"`, Xi fallback) ---
    let dir3 = temp_test_dir("varian_3d_nus");
    let procpar_3d = "\
sfrq 1 1 1000 0 0 2 1 1 1 1
1 599.7429
0
dfrq 1 1 1000 0 0 2 1 1 1 2
1 150.805
0
dfrq2 1 1 1000 0 0 2 1 1 1 3
1 60.7782
0
sw 1 1 100000 0 0 2 1 1 1 4
1 8000.0
0
sw1 1 1 100000 0 0 2 1 1 1 5
1 9600.0
0
sw2 1 1 100000 0 0 2 1 1 1 6
1 1940.0
0
np 1 1 65536 2 2 2 1 1 1 7
1 8
0
ni 1 1 4096 0 1 2 1 1 1 8
1 4
0
ni2 1 1 4096 0 1 2 1 1 1 9
1 3
0
tn 2 2 8 0 0 2 1 1 1 10
1 \"H1\"
0
dn 2 2 8 0 0 2 1 1 1 11
1 \"C13\"
0
dn2 2 2 8 0 0 2 1 1 1 12
1 \"N15\"
0
axis 2 2 8 0 0 2 1 1 1 13
1 \"pd2\"
0
phase 1 1 4 1 1 2 1 1 1 14
2 1 2
0
phase2 1 1 4 1 1 2 1 1 1 15
2 1 2
0
f2coef 2 2 32 0 0 2 1 1 1 16
1 \"1 0 -1 0 0 1 0 1\"
0
array 2 2 32 0 0 2 1 1 1 17
1 \"phase,phase2\"
0
sparse 2 2 8 0 0 2 1 1 1 18
1 \"y\"
0
";
    fs::write(dir3.join("procpar"), procpar_3d).unwrap();
    // 2 schedule coordinates `(k1=1, k2=0)` and `(k1=3, k2=2)` -> 2 * 4 = 8 traces of 4 complex points
    fs::write(dir3.join("sampling.sch"), "1 0\n3 2\n").unwrap();
    let traces_3d: Vec<Vec<[f32; 2]>> = (0..8)
        .map(|t| (0..4).map(|k| [(t + 1) as f32, (k as f32) * 0.5]).collect())
        .collect();
    fs::write(dir3.join("fid"), build_varian_f32_fid(&traces_3d)).unwrap();

    let v3 = match read_fid_auto(&dir3).expect("read Varian 3D NUS") {
        RawFid::Dim3(f) => f,
        other => panic!("expected Dim3, got {other:?}"),
    };
    assert_eq!(v3.vendor, FidVendor::Varian);
    assert_eq!(v3.axis_x.label, "1H");
    assert_eq!(v3.axis_y.label, "13C");
    assert_eq!(v3.axis_z.label, "15N");
    assert_eq!(v3.axis_y.quad_mode, QuadMode::States);
    assert_eq!(v3.axis_z.quad_mode, QuadMode::EchoAntiecho);
    assert_eq!(v3.axis_y.complex_points, 4);
    assert_eq!(v3.axis_z.complex_points, 3);
    let mask3 = v3.nus_mask.as_ref().expect("Varian 3D NUS mask");
    assert_eq!(mask3.len(), 3 * 4);
    assert!(mask3[1]); // (k2=0, k1=1)
    assert!(mask3[2 * 4 + 3]); // (k2=2, k1=3)
    assert_eq!(mask3.iter().filter(|&&b| b).count(), 2);

    let _ = fs::remove_dir_all(&dir3);
}

#[test]
fn real_data_ubiquitin_and_cisd3_smoke_and_quadrature_verification() {
    // 1. Ubiquitin Echo-Antiecho (`FnMODE = 6`) verification against `pdata/1/2rr`
    let ubi_hsqc =
        Path::new("/Users/debsahu/Workspace/nmr_software/data/ubiquitin/Ubiquitin_090309/2");
    if ubi_hsqc.join("ser").is_file() && ubi_hsqc.join("pdata/1/2rr").is_file() {
        let fid2 = read_fid_2d(ubi_hsqc).expect("read Ubiquitin 15N-HSQC raw FID");
        assert!(fid2.nus_mask.is_none());
        assert_eq!(fid2.axis_y.quad_mode, QuadMode::EchoAntiecho);
        assert_eq!(fid2.axis_x.complex_points, 1024);
        assert_eq!(fid2.axis_y.complex_points, 128);

        let pdata2 = read_bruker_pdata_2d(ubi_hsqc.join("pdata/1")).expect("read Ubiquitin 2rr");
        let si_x = fid2.axis_x.stored_proc.si.unwrap_or(2048);
        let si_y = fid2.axis_y.stored_proc.si.unwrap_or(256);
        let p0_x = fid2.axis_x.stored_proc.p0_deg.unwrap_or(0.0);
        let p1_x = fid2.axis_x.stored_proc.p1_deg.unwrap_or(0.0);
        let p0_y = fid2.axis_y.stored_proc.p0_deg.unwrap_or(0.0);
        let p1_y = fid2.axis_y.stored_proc.p1_deg.unwrap_or(0.0);

        // Transform direct dimension for all 128 cos/sin pairs
        let n1 = fid2.axis_y.complex_points;
        let mut x_cos = Vec::with_capacity(n1);
        let mut x_sin = Vec::with_capacity(n1);
        for k1 in 0..n1 {
            x_cos.push(transform_bruker_direct_trace(
                fid2.trace(k1, 0),
                si_x,
                fid2.axis_x.group_delay,
                p0_x,
                p1_x,
            ));
            x_sin.push(transform_bruker_direct_trace(
                fid2.trace(k1, 1),
                si_x,
                fid2.axis_x.group_delay,
                p0_x,
                p1_x,
            ));
        }

        // Find global amide peak (`ix` in downfield half `0 .. si_x / 2`) in `pdata/1/2rr`
        let mut pdata_best = (0usize, 0usize, f32::NEG_INFINITY);
        for iy in 0..pdata2.ny() {
            for ix in (si_x / 8)..(si_x / 2) {
                let v = pdata2.get(ix, iy);
                if v > pdata_best.2 {
                    pdata_best = (ix, iy, v);
                }
            }
        }

        // Transform indirect `15N` column at `ix = pdata_best.0` and verify peak `iy` & positive sign!
        let ix_test = pdata_best.0;
        let mut col = vec![[0.0, 0.0]; si_y];
        for k1 in 0..n1 {
            col[k1] = [x_cos[k1][ix_test], -x_sin[k1][ix_test]];
        }
        col[0][0] *= 0.5;
        col[0][1] *= 0.5;
        fft_radix2_forward(&mut col);
        fftshift_in_place(&mut col);

        let mut fid_best_iy = 0usize;
        let mut fid_best_val = f64::NEG_INFINITY;
        for (iy, c) in col.iter().enumerate() {
            let phi = (p0_y + p1_y * ((iy as f64) / (si_y as f64))).to_radians();
            let val = c[0] * phi.cos() - c[1] * phi.sin();
            if val > fid_best_val {
                fid_best_val = val;
                fid_best_iy = iy;
            }
        }
        assert!(
            fid_best_val > 0.0 && fid_best_iy.abs_diff(pdata_best.1) <= 1,
            "Ubiquitin Echo-Antiecho peak mismatch: fid iy={fid_best_iy} (val={fid_best_val}), pdata iy={}",
            pdata_best.1
        );

        // Smoke-read Ubiquitin 3D uniform Echo-Antiecho `ser` (`3` HNCACB) if present
        let ubi_hncacb =
            Path::new("/Users/debsahu/Workspace/nmr_software/data/ubiquitin/Ubiquitin_090309/3");
        if ubi_hncacb.join("ser").is_file() {
            let f3 = read_fid_3d(ubi_hncacb).expect("read Ubiquitin 3D HNCACB ser");
            assert!(f3.nus_mask.is_none());
            assert_eq!(f3.axis_y.quad_mode, QuadMode::EchoAntiecho);
            assert_eq!(f3.axis_z.quad_mode, QuadMode::StatesTppi);
        }
    }

    // 2. CISD3 TopSpin 4 uniform guard (`CISD3_HNCA`) + 3D NUS (`CISD3_CBCACONH`) vs `pdata/1/3rrr`
    let cisd3_root =
        Path::new("/Users/debsahu/Workspace/nmr_software/data/private/bmrb51439_cisd3");
    let cisd3_hnca = cisd3_root.join("CISD3_HNCA");
    if cisd3_hnca.join("ser").is_file() {
        let hnca = read_fid_3d(&cisd3_hnca).expect("read CISD3_HNCA uniform TopSpin 4 3D");
        assert!(
            hnca.nus_mask.is_none(),
            "CISD3_HNCA (FnTYPE=0, NUSLIST=<automatic>, NusAMOUNT=100) must be uniform"
        );
        assert_eq!(hnca.axis_x.complex_points, 1024);
        assert_eq!(hnca.axis_y.complex_points, 20);
        assert_eq!(hnca.axis_z.complex_points, 40);
        // Strip-transform FTSIZE = 2048 preserved over cropped SI = 1024
        assert_eq!(hnca.axis_x.stored_proc.si, Some(2048));
    }

    let cisd3_cbca = cisd3_root.join("CISD3_CBCACONH");
    if cisd3_cbca.join("ser").is_file() && cisd3_cbca.join("pdata/1/3rrr").is_file() {
        let cbca = read_fid_3d(&cisd3_cbca).expect("read CISD3_CBCACONH 3D NUS");
        assert_eq!(cbca.axis_x.complex_points, 1024);
        assert_eq!(cbca.axis_y.complex_points, 20);
        assert_eq!(cbca.axis_z.complex_points, 40);
        let mask = cbca
            .nus_mask
            .as_ref()
            .expect("CISD3_CBCACONH must have NUS mask");
        assert_eq!(mask.len(), 40 * 20);
        assert_eq!(mask.iter().filter(|&&b| b).count(), 400);

        // Verify 3D States-TPPI parity & (c2, c1) trace ordering on CISD3_CBCACONH against `pdata/1/3rrr`
        let pdata3 =
            read_bruker_pdata_3d(cisd3_cbca.join("pdata/1"), "CBCACONH").expect("read 3rrr");
        let mut pdata_best = (0usize, 0usize, 0usize, f32::NEG_INFINITY);
        for iz in 0..pdata3.nz() {
            for iy in 0..pdata3.ny() {
                for ix in 200..800.min(pdata3.nx()) {
                    let v = pdata3.get(ix, iy, iz);
                    if v > pdata_best.3 {
                        pdata_best = (ix, iy, iz, v);
                    }
                }
            }
        }

        let si_x = cbca.axis_x.stored_proc.si.unwrap_or(2048); // uncropped 2048; STSR=0 aligns bin ix directly!
        let si_y = cbca.axis_y.stored_proc.si.unwrap_or(128);
        let si_z = cbca.axis_z.stored_proc.si.unwrap_or(256);
        let p0_x = cbca.axis_x.stored_proc.p0_deg.unwrap_or(0.0);
        let p1_x = cbca.axis_x.stored_proc.p1_deg.unwrap_or(0.0);
        let ix_peak = pdata_best.0;

        // Evaluate direct FT at `ix_peak` for all sampled `(k2, k1)` and zero-filled unsampled points
        let n1 = cbca.axis_y.complex_points;
        let n2 = cbca.axis_z.complex_points;
        let mut y_spec_c2_0 = vec![vec![[0.0, 0.0]; si_y]; n2];
        let mut y_spec_c2_1 = vec![vec![[0.0, 0.0]; si_y]; n2];

        let p0_y = cbca.axis_y.stored_proc.p0_deg.unwrap_or(0.0);
        let p1_y = cbca.axis_y.stored_proc.p1_deg.unwrap_or(0.0);
        let rev_y = cbca.axis_y.stored_proc.reverse;

        for k2 in 0..n2 {
            for (c2, dst_row) in [
                (0usize, &mut y_spec_c2_0[k2]),
                (1usize, &mut y_spec_c2_1[k2]),
            ] {
                let mut row = vec![[0.0, 0.0]; si_y];
                for k1 in 0..n1 {
                    if !mask[k2 * n1 + k1] {
                        continue;
                    }
                    let rc = transform_bruker_direct_trace(
                        cbca.trace(k2, c2, k1, 0),
                        si_x,
                        cbca.axis_x.group_delay,
                        p0_x,
                        p1_x,
                    )[ix_peak];
                    let rs = transform_bruker_direct_trace(
                        cbca.trace(k2, c2, k1, 1),
                        si_x,
                        cbca.axis_x.group_delay,
                        p0_x,
                        p1_x,
                    )[ix_peak];
                    row[k1] = [rc, -rs];
                }
                row[0][0] *= 0.5;
                row[0][1] *= 0.5;
                fft_radix2_forward(&mut row);
                fftshift_in_place(&mut row);
                for (iy, dst_cell) in dst_row.iter_mut().enumerate().take(si_y) {
                    let src_iy = if rev_y { (si_y - iy) % si_y } else { iy };
                    let phi = (p0_y + p1_y * ((src_iy as f64) / (si_y as f64))).to_radians();
                    let re = row[src_iy][0] * phi.cos() - row[src_iy][1] * phi.sin();
                    *dst_cell = [re, 0.0];
                }
            }
        }

        let p0_z = cbca.axis_z.stored_proc.p0_deg.unwrap_or(0.0);
        let p1_z = cbca.axis_z.stored_proc.p1_deg.unwrap_or(0.0);
        let rev_z = cbca.axis_z.stored_proc.reverse;

        let mut plane_best = (0usize, 0usize, f64::NEG_INFINITY);
        for iy in 0..si_y {
            let mut z_col = vec![[0.0, 0.0]; si_z];
            for k2 in 0..n2 {
                z_col[k2] = [y_spec_c2_0[k2][iy][0], -y_spec_c2_1[k2][iy][0]];
            }
            z_col[0][0] *= 0.5;
            z_col[0][1] *= 0.5;
            fft_radix2_forward(&mut z_col);
            fftshift_in_place(&mut z_col);
            for iz in 0..si_z {
                let src_iz = if rev_z { (si_z - iz) % si_z } else { iz };
                let phi = (p0_z + p1_z * ((src_iz as f64) / (si_z as f64))).to_radians();
                let val = z_col[src_iz][0] * phi.cos() - z_col[src_iz][1] * phi.sin();
                if val > plane_best.2 {
                    plane_best = (iy, iz, val);
                }
            }
        }

        assert!(
            plane_best.2 > 0.0
                && plane_best.0.abs_diff(pdata_best.1) <= 1
                && plane_best.1.abs_diff(pdata_best.2) <= 1,
            "CISD3_CBCACONH 3D NUS peak mismatch: zero-filled DFT (iy={}, iz={}, val={}) vs 3rrr (iy={}, iz={})",
            plane_best.0,
            plane_best.1,
            plane_best.2,
            pdata_best.1,
            pdata_best.2
        );
    }
}

#[test]
fn case_insensitive_child_and_uppercase_bruker_varian_lookup() {
    // 1. Uppercase legacy UXNMR Bruker 3D NUS (`ACQUS`, `ACQU2S`, `ACQU3S`, `SER`, `NUSLIST`, `PDATA/1/PROCS`)
    let dir_b = temp_test_dir("bruker_uppercase_ci");
    fs::create_dir_all(dir_b.join("PDATA").join("1")).unwrap();
    let acqus = "\
##$TD= 16
##$BYTORDA= 0
##$DTYPA= 0
##$AQ_mod= 3
##$FnTYPE= 2
##$AQSEQ= 0
##$SFO1= 600.133
##$BF1= 600.130
##$SW_h= 8000.0
##$NUC1= <1H>
##$GRPDLY= 67.99
##END=
";
    let acqu2s = "\
##$TD= 8
##$NusTD= 8
##$FnMODE= 4
##$SFO1= 60.818
##$BF1= 60.810
##$SW= 32.0
##$NUC1= <15N>
##END=
";
    let acqu3s = "\
##$TD= 1
##$NusTD= 8
##$FnMODE= 4
##$SFO1= 150.910
##$BF1= 150.900
##$SW= 60.0
##$NUC1= <13C>
##END=
";
    let procs = "\
##$SI= 512
##$PHC0= -42.5
##$PHC1= 12.0
##$PH_mod= 1
##END=
";
    fs::write(dir_b.join("ACQUS"), acqus).unwrap();
    fs::write(dir_b.join("ACQU2S"), acqu2s).unwrap();
    fs::write(dir_b.join("ACQU3S"), acqu3s).unwrap();
    fs::write(dir_b.join("NUSLIST"), "0 1\n2 3\n").unwrap();
    fs::write(dir_b.join("PDATA").join("1").join("PROCS"), procs).unwrap();
    fs::write(dir_b.join("SER"), vec![0u8; 8 * 1024]).unwrap();

    // Verify `find_child_ci` resolves exact on-disk uppercase `ACQUS` and `PDATA`
    let found_acqus = find_child_ci(&dir_b, "acqus").expect("find_child_ci acqus -> ACQUS");
    assert_eq!(found_acqus.file_name().unwrap().to_str().unwrap(), "ACQUS");
    let found_pdata = find_child_ci(&dir_b, "pdata").expect("find_child_ci pdata -> PDATA");
    assert_eq!(found_pdata.file_name().unwrap().to_str().unwrap(), "PDATA");
    assert!(find_child_ci(&dir_b, "nonexistent_file").is_none());

    let fid3 = read_fid_3d(&dir_b).expect("read uppercase Bruker 3D NUS");
    assert_eq!(fid3.axis_x.stored_proc.p0_deg, Some(-42.5));
    assert_eq!(fid3.axis_x.stored_proc.si, Some(512));
    assert_eq!(
        fid3.nus_mask
            .as_ref()
            .unwrap()
            .iter()
            .filter(|&&b| b)
            .count(),
        2
    );
    let _ = fs::remove_dir_all(&dir_b);

    // 2. Uppercase Varian 2D NUS (`PROCPAR`, `FID`, `SAMPLING.SCH`)
    let dir_v = temp_test_dir("varian_uppercase_ci");
    let procpar = "\
sfrq 1 1 1000 0 0 2 1 1 1 1
1 599.7429
0
dfrq2 1 1 1000 0 0 2 1 1 1 2
1 60.7782
0
sw 1 1 100000 0 0 2 1 1 1 3
1 8000.0
0
sw1 1 1 100000 0 0 2 1 1 1 4
1 2000.0
0
np 1 1 65536 2 2 2 1 1 1 5
1 8
0
ni 1 1 4096 0 1 2 1 1 1 6
1 4
0
tn 2 2 8 0 0 2 1 1 1 7
1 \"H1\"
0
dn2 2 2 8 0 0 2 1 1 1 8
1 \"N15\"
0
axis 2 2 8 0 0 2 1 1 1 9
1 \"p2\"
0
phase 1 1 4 1 1 2 1 1 1 10
2 1 2
0
sampling 2 2 8 0 0 2 1 1 1 11
1 \"sparse\"
0
";
    fs::write(dir_v.join("PROCPAR"), procpar).unwrap();
    fs::write(dir_v.join("SAMPLING.SCH"), "0\n3\n").unwrap();
    let traces: Vec<Vec<[f32; 2]>> = (0..4).map(|_| vec![[1.0, 0.0]; 4]).collect();
    fs::write(dir_v.join("FID"), build_varian_f32_fid(&traces)).unwrap();

    let v2 = read_fid_2d(&dir_v).expect("read uppercase Varian 2D NUS");
    assert_eq!(v2.nus_mask, Some(vec![true, false, false, true]));
    let _ = fs::remove_dir_all(&dir_v);
}

#[test]
fn varian_3d_f1coef_precedence_over_stale_2d_fcoef() {
    let dir = temp_test_dir("varian_f1coef_prec");
    // BioPack 3D procpar templated from 2D gNfhsqc:
    // - stale 2D `fcoef` = "1 0 -1 0 0 1 0 1" (Echo-Antiecho!)
    // - active 3D `f1coef` = "1 0 0 0 0 0 1 0" (States!)
    // - active 3D `f2coef` = "1 0 -1 0 0 1 0 1" (Echo-Antiecho!)
    let procpar_3d = "\
sfrq 1 1 1000 0 0 2 1 1 1 1
1 599.7429
0
dfrq 1 1 1000 0 0 2 1 1 1 2
1 150.805
0
dfrq2 1 1 1000 0 0 2 1 1 1 3
1 60.7782
0
sw 1 1 100000 0 0 2 1 1 1 4
1 8000.0
0
sw1 1 1 100000 0 0 2 1 1 1 5
1 9600.0
0
sw2 1 1 100000 0 0 2 1 1 1 6
1 1940.0
0
np 1 1 65536 2 2 2 1 1 1 7
1 8
0
ni 1 1 4096 0 1 2 1 1 1 8
1 2
0
ni2 1 1 4096 0 1 2 1 1 1 9
1 2
0
tn 2 2 8 0 0 2 1 1 1 10
1 \"H1\"
0
dn 2 2 8 0 0 2 1 1 1 11
1 \"C13\"
0
dn2 2 2 8 0 0 2 1 1 1 12
1 \"N15\"
0
axis 2 2 8 0 0 2 1 1 1 13
1 \"pd2\"
0
phase 1 1 4 1 1 2 1 1 1 14
2 1 2
0
phase2 1 1 4 1 1 2 1 1 1 15
2 1 2
0
fcoef 2 2 32 0 0 2 1 1 1 16
1 \"1 0 -1 0 0 1 0 1\"
0
f1coef 2 2 32 0 0 2 1 1 1 17
1 \"1 0 0 0 0 0 1 0\"
0
f2coef 2 2 32 0 0 2 1 1 1 18
1 \"1 0 -1 0 0 1 0 1\"
0
array 2 2 32 0 0 2 1 1 1 19
1 \"phase,phase2\"
0
";
    fs::write(dir.join("procpar"), procpar_3d).unwrap();
    // Uniform 3D: 2 * 2 * 4 = 16 traces of 4 complex points
    let traces_3d: Vec<Vec<[f32; 2]>> = (0..16).map(|t| vec![[(t + 1) as f32, 0.0]; 4]).collect();
    fs::write(dir.join("fid"), build_varian_f32_fid(&traces_3d)).unwrap();

    let v3 = read_fid_3d(&dir).expect("read Varian 3D with stale fcoef + active f1coef");
    // Must use `f1coef` (States), NOT stale 2D `fcoef` (Echo-Antiecho)!
    assert_eq!(v3.axis_y.quad_mode, QuadMode::States);
    assert_eq!(v3.axis_z.quad_mode, QuadMode::EchoAntiecho);
    // At (k2=0, k1=0), traces 0..4 have `array="phase,phase2"` -> (p1=0,p2=0)=1, (0,1)=2, (1,0)=3, (1,1)=4:
    // y_e (p2=0): States on (1, 3) -> cos=[1,0], sin=[3,0]
    // y_o (p2=1): States on (2, 4) -> cos=[2,0], sin=[4,0]
    // f2coef "1 0 -1 0 0 1 0 1" on (y_e_cos=[1,0], y_o_cos=[2,0]) -> z_cos = [1-2, 0] = [-1, 0], z_sin = [0, -1-2] = [0, -3]
    assert_eq!(v3.trace(0, 0, 0, 0)[0], [-1.0, 0.0]);
    assert_eq!(v3.trace(0, 1, 0, 0)[0], [0.0, -3.0]);

    // Conversely, for 2D (`ni2 = 0`), `fcoef` takes precedence over `f1coef`, AND an empty `fcoef = ""` falls back to `f1coef`!
    let procpar_2d_pref = procpar_3d.replace(
        "ni2 1 1 4096 0 1 2 1 1 1 9\n1 2",
        "ni2 1 1 4096 0 1 2 1 1 1 9\n1 0",
    );
    fs::write(dir.join("procpar"), &procpar_2d_pref).unwrap();
    fs::write(dir.join("fid"), build_varian_f32_fid(&traces_3d[..4])).unwrap();
    let v2 = read_fid_2d(&dir).expect("read Varian 2D with both fcoef and f1coef");
    assert_eq!(v2.axis_y.quad_mode, QuadMode::EchoAntiecho);

    let procpar_2d_empty_fcoef = procpar_2d_pref
        .replace(
            "fcoef 2 2 32 0 0 2 1 1 1 16\n1 \"1 0 -1 0 0 1 0 1\"",
            "fcoef 2 2 32 0 0 2 1 1 1 16\n1 \"\"",
        )
        .replace(
            "f1coef 2 2 32 0 0 2 1 1 1 17\n1 \"1 0 0 0 0 0 1 0\"",
            "f1coef 2 2 32 0 0 2 1 1 1 17\n1 \"1 0 -1 0 0 1 0 1\"",
        );
    fs::write(dir.join("procpar"), &procpar_2d_empty_fcoef).unwrap();
    let v2_fb = read_fid_2d(&dir).expect("read Varian 2D with empty fcoef + non-empty f1coef");
    assert_eq!(v2_fb.axis_y.quad_mode, QuadMode::EchoAntiecho);
    assert!(v2_fb.axis_y.conjugate_for_fft);

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn varian_hardware_xi_referencing_and_bruker_raw_pulseprogram_reverse_inference() {
    use nmr_core::fid::infer_bruker_raw_indirect_reverse;

    // 1. Varian hardware Xi referencing ignores stale VnmrJ interactive `rfl*`/`rfp*`/`reffrq*` cursor offsets
    let dir = temp_test_dir("varian_xi_hw");
    let procpar_stale_cursors = "\
sfrq 1 1 1000 0 0 2 1 1 1 1
1 599.7429
0
dfrq2 1 1 1000 0 0 2 1 1 1 2
1 60.7782
0
temp 1 1 200 -100 0 2 1 1 1 3
1 25.0
0
sw 1 1 100000 0 0 2 1 1 1 4
1 8000.0
0
sw1 1 1 100000 0 0 2 1 1 1 5
1 2000.0
0
np 1 1 65536 2 2 2 1 1 1 6
1 8
0
ni 1 1 4096 0 1 2 1 1 1 7
1 4
0
tn 2 2 8 0 0 2 1 1 1 8
1 \"H1\"
0
dn2 2 2 8 0 0 2 1 1 1 9
1 \"N15\"
0
axis 2 2 8 0 0 2 1 1 1 10
1 \"p2\"
0
phase 1 1 4 1 1 2 1 1 1 11
2 1 2
0
rfl 1 1 100000 -100000 0 2 1 1 1 12
1 9999.0
0
rfp 1 1 100000 -100000 0 2 1 1 1 13
1 -1234.0
0
reffrq 1 1 1000 0 0 2 1 1 1 14
1 599.880
0
rfl1 1 1 100000 -100000 0 2 1 1 1 15
1 8888.0
0
rfp1 1 1 100000 -100000 0 2 1 1 1 16
1 1234.0
0
reffrq1 1 1 1000 0 0 2 1 1 1 17
1 599.880
0
";
    fs::write(dir.join("procpar"), procpar_stale_cursors).unwrap();
    let traces: Vec<Vec<[f32; 2]>> = (0..8).map(|_| vec![[1.0, 0.0]; 4]).collect();
    fs::write(dir.join("fid"), build_varian_f32_fid(&traces)).unwrap();

    let v_no_fcoef = read_fid_2d(&dir).expect("read Varian 2D without fcoef");
    assert!(
        !v_no_fcoef.axis_y.conjugate_for_fft,
        "Default Varian States without fcoef uses [1,0,0,0,0,0,+1,0] (conjugate_for_fft = false)"
    );
    let bf_1h = 599.7429 / (1.0 + 4.766e-6);
    let expected_first_h = ((599.7429 - bf_1h) * 1.0e6 + 4000.0) / bf_1h;
    let obs_15n = bf_1h * 0.101_329_118;
    let expected_first_n = ((60.7782 - obs_15n) * 1.0e6 + 1000.0) / obs_15n;
    assert!((v_no_fcoef.axis_x.first_ppm - expected_first_h).abs() < 1e-6);
    assert!((v_no_fcoef.axis_y.first_ppm - expected_first_n).abs() < 1e-6);

    let procpar_with_fcoef = format!(
        "{procpar_stale_cursors}\nfcoef 2 2 32 0 0 2 1 1 1 18\n1 \"1 0 -1 0 0 -1 0 -1\"\n0\n"
    );
    fs::write(dir.join("procpar"), procpar_with_fcoef).unwrap();
    let v_with_fcoef = read_fid_2d(&dir).expect("read Varian 2D with explicit fcoef");
    assert!(
        v_with_fcoef.axis_y.conjugate_for_fft,
        "Explicit Varian fcoef requires +i indirect FT (conjugate_for_fft = true)"
    );
    let _ = fs::remove_dir_all(&dir);

    // 2. Bruker raw `ser` indirect `REVERSE` spin-physics inference
    let pp_post_ip9 = "\
  d0
  (p13:sp2 ph9):f2
  d10
  go=2 ph31
  MCWRK id10
  MCWRK rd10 MCWRK ip9 MCWRK ip10
  MCWRK id0 MCWRK dp9*2
ph9=3 1
";
    assert!(infer_bruker_raw_indirect_reverse(
        Some(pp_post_ip9),
        "hncacbgp3d",
        1,
        QuadMode::StatesTppi
    ));

    let pp_post_dp9 = pp_post_ip9.replace("ip9", "dp9");
    assert!(!infer_bruker_raw_indirect_reverse(
        Some(&pp_post_dp9),
        "hncacbgp3d",
        1,
        QuadMode::StatesTppi
    ));

    let pp_pre_ip4 = "\
  (p13:sp2 ph4):f2
  d0
  d10
  go=2 ph31
  MCWRK id10
  MCWRK rd10 MCWRK ip4
  MCWRK id0
ph4=0
";
    assert!(!infer_bruker_raw_indirect_reverse(
        Some(pp_pre_ip4),
        "hncogp3d",
        1,
        QuadMode::StatesTppi
    ));
    assert!(!infer_bruker_raw_indirect_reverse(
        Some(pp_post_ip9),
        "hncacbgp3d",
        2,
        QuadMode::EchoAntiecho
    ));

    let pp_sct_f2 = "\
  (p13:sp2 ph3):f2
  d0
  (p21 ph5):f3
  d30
  (p22 ph8):f3
  d10
  F1PH(calph(ph3, +90), caldel(d0, +in0))
  F2PH(calph(ph5, +90), caldel(d10, +in10) & caldel(d30, -in30))
ph3=0
";
    assert!(!infer_bruker_raw_indirect_reverse(
        Some(pp_sct_f2),
        "cbcacgp3d",
        1,
        QuadMode::StatesTppi
    ));
    assert!(infer_bruker_raw_indirect_reverse(
        Some(pp_sct_f2),
        "cbcacgp3d",
        2,
        QuadMode::StatesTppi
    ));
}

#[test]
fn bruker_xwin_nmr_unprocessed_pdata_channel_sfo_and_echo_antiecho_reverse() {
    use nmr_core::fid::infer_bruker_raw_indirect_reverse;

    // 1. Echo-Antiecho (`FnMODE = 6`) SCT delay ordering (`f_dim = 1`: `d0`/`d20`, `f_dim = 2`: `d10`/`d30`)
    let pp_ea_f1_d0_first = "\
  (p21 ph1):f3
  d0
  (p22 ph2):f3
  d20
  go=2 ph31
  MCWRK id0
  MCWRK dd20
";
    assert!(
        infer_bruker_raw_indirect_reverse(
            Some(pp_ea_f1_d0_first),
            "hsqcetf3gpsi",
            1,
            QuadMode::EchoAntiecho
        ),
        "2D Echo-Antiecho SCT with d0 before d20 evolves exp(-i*w*t1) and requires REVERSE = true"
    );
    let pp_ea_f1_d20_first = "\
  (p21 ph1):f3
  d20
  (p22 ph2):f3
  d0
  go=2 ph31
  MCWRK id0
  MCWRK dd20
";
    assert!(
        !infer_bruker_raw_indirect_reverse(
            Some(pp_ea_f1_d20_first),
            "hsqcetf3gpsi",
            1,
            QuadMode::EchoAntiecho
        ),
        "2D Echo-Antiecho SCT with d20 before d0 evolves exp(+i*w*t1) and requires REVERSE = false"
    );
    let pp_ea_d10_first = "\
  (p21 ph1):f3
  d10
  (p22 ph2):f3
  d30
  go=2 ph31
  MCWRK id10
  MCWRK dd30
";
    assert!(
        infer_bruker_raw_indirect_reverse(
            Some(pp_ea_d10_first),
            "hncacbgp3d",
            2,
            QuadMode::EchoAntiecho
        ),
        "3D F2 Echo-Antiecho SCT with d10 before d30 evolves exp(-i*w*t2) and requires REVERSE = true"
    );

    // 2. Synthetic XWIN-NMR 3D directory:
    // - `pdata/1/procs` has `FT_mod = 0`, `PH_mod = 0`, `PHC0 = 0`, bogus `OFFSET = 99.0`
    // - `pdata/1/proc2s` exists WITHOUT `proc3s` (incomplete 2D scout plane)
    // - `acqus` has `NUC3 = <15N>`, `BF3 = 60.8106`, `SFO3 = 60.8180` while `acqu2s` has stale `SFO1 = 60.8106`
    let dir = temp_test_dir("xwin_nmr_3d_guards");
    let acqus = "\
##TITLE= XWIN-NMR 3D test
##$BYTORDA= 0
##$DTYPA= 0
##$AQ_mod= 3
##$TD= 32
##$SW_h= 8000.0
##$SFO1= 600.1328
##$BF1= 600.1300
##$NUC1= <1H>
##$NUC2= <13C>
##$BF2= 150.9028
##$SFO2= 150.9112
##$NUC3= <15N>
##$BF3= 60.8106
##$SFO3= 60.8180
##$DECIM= 1
##$DSPFVS= 0
##$AQSEQ= 0
##$PULPROG= <hncogp3d>
##END=
";
    let acqu2s = "\
##$FnMODE= 6
##$TD= 8
##$SW= 32.0
##$SFO1= 60.8106
##$BF1= 60.8106
##$NUC1= <15N>
##END=
";
    let acqu3s = "\
##$FnMODE= 4
##$TD= 8
##$SW= 14.0
##$SFO1= 150.9293
##$BF1= 150.9028
##$NUC1= <13C>
##END=
";
    fs::write(dir.join("acqus"), acqus).unwrap();
    fs::write(dir.join("acqu2s"), acqu2s).unwrap();
    fs::write(dir.join("acqu3s"), acqu3s).unwrap();
    fs::write(dir.join("pulseprogram"), pp_ea_d10_first).unwrap();

    // 8 * 8 = 64 traces, each 1024-byte block (256 i32 words)
    let mut ser_bytes = Vec::with_capacity(64 * 1024);
    for t in 0..64i32 {
        for w in 0..256i32 {
            let val = if w < 32 { (t + 1) * 10 + w } else { 0 };
            ser_bytes.extend_from_slice(&val.to_le_bytes());
        }
    }
    fs::write(dir.join("ser"), &ser_bytes).unwrap();

    let pdata1 = dir.join("pdata").join("1");
    fs::create_dir_all(&pdata1).unwrap();
    // Unprocessed XWIN-NMR template (`FT_mod = 0`, `PH_mod = 0`) + incomplete 2D scout `proc2s`
    fs::write(
        pdata1.join("procs"),
        "##$FT_mod= 0\n##$PH_mod= 0\n##$PHC0= 0\n##$PHC1= 0\n##$OFFSET= 99.0\n##$SF= 599.0\n##$SI= 64\n##END=\n",
    )
    .unwrap();
    fs::write(
        pdata1.join("proc2s"),
        "##$FT_mod= 4\n##$PH_mod= 1\n##$PHC0= 123.0\n##$OFFSET= 250.0\n##$SF= 60.8106\n##$SI= 32\n##$REVERSE= no\n##END=\n",
    )
    .unwrap();

    let fid3 = read_fid_3d(&dir).expect("read XWIN-NMR 3D ser");
    assert!(
        fid3.axis_x.stored_proc.p0_deg.is_none() && fid3.axis_x.stored_proc.p1_deg.is_none(),
        "PH_mod = 0 / FT_mod = 0 must leave stored phases as None so blind autophasing activates"
    );
    assert!(
        (fid3.axis_x.obs_mhz - 600.1300).abs() < 1e-6 && fid3.axis_x.first_ppm < 20.0,
        "FT_mod = 0 bogus OFFSET/SF must not override acqus referencing: got {:?}",
        fid3.axis_x
    );
    assert!(
        fid3.axis_y.stored_proc.p0_deg.is_none(),
        "Incomplete 2D scout proc2s (without proc3s) must not leak into 3D indirect axis"
    );
    assert!(
        (fid3.axis_y.carrier_mhz - 60.8180).abs() < 1e-6,
        "Indirect 15N carrier must sync to active acqus SFO3 (60.8180) over stale acqu2s SFO1 (60.8106), got {}",
        fid3.axis_y.carrier_mhz
    );
    assert!(
        fid3.axis_y.stored_proc.reverse,
        "Echo-Antiecho with (d10 ... d30) must infer REVERSE = true when calibrated proc2s/proc3s is absent"
    );
    assert!(fid3.axis_y.states_recombined && fid3.axis_z.states_recombined);

    // 3. Unsupported Bruker `FnMODE` codes (`0 = Undefined`, `2 = QSEQ`, `3 = TPPI`) must fail explicitly
    for bad_mode in [0, 2, 3, 99] {
        fs::write(
            dir.join("acqu2s"),
            format!("##$FnMODE= {bad_mode}\n##$TD= 8\n##$SW= 32.0\n##$SFO1= 60.8180\n##$BF1= 60.8106\n##$NUC1= <15N>\n##END=\n"),
        )
        .unwrap();
        let err = read_fid_3d(&dir).expect_err("unsupported FnMODE must return Err");
        assert!(
            err.contains("FnMODE"),
            "expected FnMODE error message for mode {bad_mode}, got: {err}"
        );
    }

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn varian_3d_fcoef_sign_and_origfid_untouched_backup() {
    let dir = temp_test_dir("varian_3d_fcoef_origfid");
    let make_procpar = |f1coef_str: &str, f2coef_str: &str| -> String {
        format!(
            "\
sfrq 1 1 1000 0 0 2 1 1 1 1
1 499.915
0
dfrq 1 1 1000 0 0 2 1 1 1 2
1 125.708
0
dfrq2 1 1 1000 0 0 2 1 1 1 3
1 50.662
0
sw 1 1 100000 0 0 2 1 1 1 4
1 8000.0
0
sw1 1 1 100000 0 0 2 1 1 1 5
1 7542.0
0
sw2 1 1 100000 0 0 2 1 1 1 6
1 1347.0
0
np 1 1 65536 2 2 2 1 1 1 7
1 16
0
ni 1 1 4096 0 1 2 1 1 1 8
1 2
0
ni2 1 1 4096 0 1 2 1 1 1 9
1 2
0
tn 2 2 8 0 0 2 1 1 1 10
1 \"H1\"
0
dn 2 2 8 0 0 2 1 1 1 11
1 \"C13\"
0
dn2 2 2 8 0 0 2 1 1 1 12
1 \"N15\"
0
phase 1 1 4 1 1 2 1 1 1 13
2 1 2
0
phase2 1 1 4 1 1 2 1 1 1 14
2 1 2
0
array 2 2 32 0 0 2 1 1 1 15
1 \"phase,phase2\"
0
f1coef 2 2 32 0 0 2 1 1 1 16
1 \"{f1coef_str}\"
0
f2coef 2 2 32 0 0 2 1 1 1 17
1 \"{f2coef_str}\"
0
"
        )
    };

    // ni = 2, ni2 = 2, 4 traces per point -> 16 traces of 8 complex points
    let mut orig_traces: Vec<Vec<[f32; 2]>> = (0..16).map(|_| vec![[0.0, 0.0]; 8]).collect();
    orig_traces[0][0] = [10.0, 0.0]; // Echo
    orig_traces[1][0] = [6.0, 0.0]; // Antiecho
    let mut munged_traces = orig_traces.clone();
    munged_traces[0][0] = [999.0, -999.0]; // Pre-shuffled `fid` that disagrees with procpar

    fs::write(dir.join("fid"), build_varian_f32_fid(&munged_traces)).unwrap();
    fs::write(dir.join("origfid"), build_varian_f32_fid(&orig_traces)).unwrap();

    // Case A: `f1coef = ""` (default VnmrJ `"1 0 0 0 0 0 -1 0"` -> `false`), Rance-Kay `f2coef` -> `true`
    fs::write(dir.join("procpar"), make_procpar("", "1 0 -1 0 0 -1 0 -1")).unwrap();
    let f_default = read_fid_3d(&dir).expect("read Varian 3D with empty f1coef");
    assert!(
        !f_default.axis_y.conjugate_for_fft,
        "Empty Varian f1coef must use conjugate_for_fft = false"
    );
    assert!(
        f_default.axis_z.conjugate_for_fft,
        "Rance-Kay Echo-Antiecho f2coef must use conjugate_for_fft = true"
    );
    // Verify `origfid` was read (`z_cos = 10 - 6 = 4.0`, NOT `999 - 6`!)
    assert!(
        (f_default.trace(0, 0, 0, 0)[0][0] - 4.0).abs() < 1e-5,
        "Expected untouched `origfid` trace recombination (10 - 6 = 4.0), got {:?}",
        f_default.trace(0, 0, 0, 0)[0]
    );

    // Case B: explicit `f1coef = "1 0 0 0 0 0 -1 0"` -> `false`
    fs::write(
        dir.join("procpar"),
        make_procpar("1 0 0 0 0 0 -1 0", "1 0 -1 0 0 -1 0 -1"),
    )
    .unwrap();
    let f_neg = read_fid_3d(&dir).expect("read Varian 3D with -1 f1coef");
    assert!(!f_neg.axis_y.conjugate_for_fft);

    // Case C: explicit `f1coef = "1 0 0 0 0 0 1 0"` (`ghn_cacb` constant-time States) -> `true`
    fs::write(
        dir.join("procpar"),
        make_procpar("1 0 0 0 0 0 1 0", "1 0 -1 0 0 -1 0 -1"),
    )
    .unwrap();
    let f_pos = read_fid_3d(&dir).expect("read Varian 3D with +1 f1coef");
    assert!(
        f_pos.axis_y.conjugate_for_fft,
        "Same-sign States f1coef '1 0 0 0 0 0 1 0' must set conjugate_for_fft = true"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn f1_xwinnmr_missing_fnmode_pulprog_and_mc2_fallbacks_and_template_sf_guard() {
    let dir = temp_test_dir("f1_xwinnmr_mc2_sf_guard");
    let acqus = "\
##TITLE= Parameter file, XWIN-NMR 2.6
##JCAMPDX= 5.0
##$BYTORDA= 1
##$DTYPA= 0
##$AQ_mod= 3
##$TD= 32
##$SW_h= 8389.26
##$SFO1= 600.1328206
##$BF1= 600.1300000
##$O1= 2820.61
##$NUC1= <1H>
##$NUC3= <15N>
##$BF3= 60.8106630
##$SFO3= 60.8179603
##$O3= 7297.29
##$DECIM= 1
##$DSPFVS= 0
##$PULPROG= <fhsqc.s>
##END=
";
    // Legacy XWIN-NMR 2.x `acqu2s` omits `##$FnMODE` entirely!
    let acqu2s_no_fnmode = "\
##TITLE= Parameter file, XWIN-NMR 2.6
##JCAMPDX= 5.0
##$TD= 8
##$SW= 29.85
##$SFO1= 60.8179603
##$BF1= 60.8106630
##$NUC1= <15N>
##END=
";
    fs::write(dir.join("acqus"), acqus).unwrap();
    fs::write(dir.join("acqu2s"), acqu2s_no_fnmode).unwrap();

    // 8 traces of 256 big-endian i32 words (1024-byte blocks)
    let mut ser_bytes = Vec::with_capacity(8 * 1024);
    for t in 0..8i32 {
        for w in 0..256i32 {
            let v = if w < 32 { (t + 1) * 100 + w } else { 0 };
            ser_bytes.extend_from_slice(&v.to_be_bytes());
        }
    }
    fs::write(dir.join("ser"), &ser_bytes).unwrap();

    // Priority 3: neither `pulseprogram` nor `proc2s` `MC2` -> fail closed!
    let err_none = read_fid_2d(&dir).expect_err("missing FnMODE without pulprog or MC2 must fail");
    assert!(
        err_none.contains("FnMODE") && err_none.contains("MC2"),
        "unexpected error: {err_none}"
    );

    // Priority 2: `proc2s` `MC2` deterministic mapping (`0=>Qf`, `3=>States`, `4=>StatesTppi`, `5=>EchoAntiecho`)
    // AND template `SF = 600.13` (`1H` template on `15N` `BF1 = 60.810663`!) + `OFFSET = -898657.9` guard!
    let pdata1 = dir.join("pdata").join("1");
    fs::create_dir_all(&pdata1).unwrap();
    fs::write(
        pdata1.join("procs"),
        "##$FT_mod= 6\n##$PH_mod= 1\n##$PHC0= 0\n##$PHC1= 0\n##$SF= 600.13\n##$OFFSET= 11.68\n##$SI= 32\n##END=\n",
    )
    .unwrap();
    for (mc2_code, expected_qm) in [
        (0, QuadMode::Qf),
        (3, QuadMode::States),
        (4, QuadMode::StatesTppi),
        (5, QuadMode::EchoAntiecho),
    ] {
        fs::write(
            pdata1.join("proc2s"),
            format!(
                "##$MC2= {mc2_code}\n##$FT_mod= 4\n##$PH_mod= 1\n##$PHC0= 0\n##$PHC1= 0\n##$SF= 600.1300000\n##$OFFSET= -898657.9\n##$SI= 16\n##$REVERSE= no\n##END=\n"
            ),
        )
        .unwrap();
        let fid2 = read_fid_2d(&dir).expect("MC2 fallback should succeed");
        assert_eq!(fid2.axis_y.quad_mode, expected_qm);
        assert!(
            (fid2.axis_y.obs_mhz - 60.8106630).abs() < 1e-4,
            "proc2s template 1H SF=600.13 must be rejected on 15N axis (BF1=60.810663), got {}",
            fid2.axis_y.obs_mhz
        );
        assert!(
            fid2.axis_y.first_ppm > 100.0 && fid2.axis_y.first_ppm < 160.0,
            "proc2s template OFFSET=-898657.9 must be rejected, got first_ppm={}",
            fid2.axis_y.first_ppm
        );
    }

    // Unsupported `MC2` codes (`1 = QSEQ`, `2 = TPPI`, `99 = unknown`) must fail closed!
    for bad_mc2 in [1, 2, 99] {
        fs::write(
            pdata1.join("proc2s"),
            format!("##$MC2= {bad_mc2}\n##$SF= 60.8106630\n##$OFFSET= 134.9\n##END=\n"),
        )
        .unwrap();
        let err = read_fid_2d(&dir).expect_err("unsupported MC2 must fail closed");
        assert!(
            err.contains("MC2"),
            "expected MC2 error for {bad_mc2}, got: {err}"
        );
    }

    // Priority 1: `pulseprogram` `SynthesizedManualLoop` (`StatesTppi`) takes priority over `proc2s` `MC2 = 3` (`States`)!
    fs::write(
        pdata1.join("proc2s"),
        "##$MC2= 3\n##$SF= 60.8106630\n##$OFFSET= 134.9\n##END=\n",
    )
    .unwrap();
    let pp_manual_states_tppi = "\
1 ze
2 d1 do:f3
3 (p1 ph1)
  (p21 ph3):f3
  d0
  (p2 ph2)
  d0
  (p21 ph4):f3
  go=3 ph31 cpd3:f3
  d1 do:f3 wr #0 if #0 zd ip3
  lo to 3 times 2
  d1 id0 ip3*2 ip31*2
  lo to 2 times 4
exit
ph1=0
ph2=0
ph3=0 2
ph4=0
ph31=0 2
";
    fs::write(dir.join("pulseprogram"), pp_manual_states_tppi).unwrap();
    let fid_pp = read_fid_2d(&dir).expect("pulseprogram manual loop priority 1");
    assert_eq!(
        fid_pp.axis_y.quad_mode,
        QuadMode::StatesTppi,
        "Priority 1 pulseprogram SynthesizedManualLoop must take precedence over proc2s MC2"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn f2_truncated_fid_opt_in_zero_fill_bruker_and_varian() {
    // 1. Truncated Bruker 3D `ser` (`48 / 64` traces = 75.0% acquired)
    let bdir = temp_test_dir("f2_trunc_bruker_3d");
    let acqus = "\
##TITLE= Parameter file, TopSpin 3.2
##JCAMPDX= 5.0
##$BYTORDA= 0
##$DTYPA= 0
##$AQ_mod= 3
##$TD= 32
##$SW_h= 8000.0
##$SFO1= 600.1328
##$BF1= 600.1300
##$NUC1= <1H>
##$DECIM= 1
##$DSPFVS= 0
##$AQSEQ= 0
##$PULPROG= <hncogp3d>
##END=
";
    let acqu2s = "\
##$FnMODE= 4
##$TD= 8
##$SW= 32.0
##$SFO1= 60.8180
##$BF1= 60.8106
##$NUC1= <15N>
##END=
";
    let acqu3s = "\
##$FnMODE= 4
##$TD= 8
##$SW= 14.0
##$SFO1= 150.9293
##$BF1= 150.9028
##$NUC1= <13C>
##END=
";
    fs::write(bdir.join("acqus"), acqus).unwrap();
    fs::write(bdir.join("acqu2s"), acqu2s).unwrap();
    fs::write(bdir.join("acqu3s"), acqu3s).unwrap();

    // Write only 48 of the expected 64 1024-byte traces (+ 128 trailing partial bytes from mid-scan halt!)
    let mut ser_trunc = Vec::with_capacity(48 * 1024 + 128);
    for t in 0..48i32 {
        for w in 0..256i32 {
            let val = if w < 32 { (t + 1) * 10 + w } else { 0 };
            ser_trunc.extend_from_slice(&val.to_le_bytes());
        }
    }
    ser_trunc.extend_from_slice(&[0xAB; 128]);
    fs::write(bdir.join("ser"), &ser_trunc).unwrap();

    assert!(
        read_fid_3d(&bdir).is_err(),
        "default read_fid_3d must reject truncated Bruker ser"
    );
    assert!(
        process_fid_dir_3d(&bdir, "HNCO", false, false).is_err(),
        "default process_fid_dir_3d must reject truncated Bruker ser"
    );

    let bfid = read_fid_3d_opts(&bdir, true).expect("opt-in truncated Bruker 3D");
    assert_eq!(bfid.truncated_traces, Some((48, 64)));
    assert_eq!(bfid.trace(0, 0, 0, 0)[0], [10.0, 11.0]);
    // k2 = 3 (traces 48..63) must be zero-filled!
    assert_eq!(bfid.trace(3, 0, 0, 0), vec![[0.0, 0.0]; 16].as_slice());

    let (_bspec, brec) =
        process_fid_dir_3d_opts(&bdir, "HNCO", false, false, true).expect("process truncated 3D");
    assert!(brec.allow_truncated_fid);
    assert!(brec.truncated_fid_zero_filled);
    let bwarn = brec
        .truncated_fid_warning
        .as_deref()
        .expect("truncated_fid_warning recorded");
    assert!(bwarn.contains("48/64") && bwarn.contains("75.0%"));

    // Empty `ser` (`0` complete traces) must STILL fail closed even with `allow_truncated_fid = true`!
    fs::write(bdir.join("ser"), [0u8; 64]).unwrap();
    assert!(read_fid_3d_opts(&bdir, true).is_err());
    let _ = fs::remove_dir_all(&bdir);

    // 2. Truncated Varian 2D & 3D `fid` (including mid-hypercomplex-point halt `10 / 16` traces!)
    let vdir = temp_test_dir("f2_trunc_varian_3d");
    let procpar_3d = "\
sfrq 1 1 1000 0 0 2 1 1 1 1
1 599.812
0
dfrq 1 1 1000 0 0 2 1 1 1 2
1 150.822
0
dfrq2 1 1 1000 0 0 2 1 1 1 3
1 60.785
0
sw 1 1 100000 0 0 2 1 1 1 4
1 8000.0
0
sw1 1 1 100000 0 0 2 1 1 1 5
1 9000.0
0
sw2 1 1 100000 0 0 2 1 1 1 6
1 1800.0
0
np 1 1 65536 2 2 2 1 1 1 7
1 32
0
ni 1 1 4096 0 1 2 1 1 1 8
1 2
0
ni2 1 1 4096 0 1 2 1 1 1 9
1 2
0
tn 2 2 8 0 0 2 1 1 1 10
1 \"H1\"
0
dn 2 2 8 0 0 2 1 1 1 11
1 \"C13\"
0
dn2 2 2 8 0 0 2 1 1 1 12
1 \"N15\"
0
phase 1 1 4 1 1 2 1 1 1 13
2 1 2
0
phase2 1 1 4 1 1 2 1 1 1 14
2 1 2
0
array 2 2 32 0 0 2 1 1 1 15
1 \"phase,phase2\"
0
f1coef 2 2 32 0 0 2 1 1 1 16
1 \"1 0 0 0 0 0 -1 0\"
0
f2coef 2 2 32 0 0 2 1 1 1 17
1 \"1 0 -1 0 0 -1 0 -1\"
0
";
    fs::write(vdir.join("procpar"), procpar_3d).unwrap();
    // Expects 2 * 2 * 2 * 2 = 16 traces of 16 complex points; write only 10 blocks (+ truncated 11th block bytes)!
    let mut vtraces: Vec<Vec<[f32; 2]>> = (0..11).map(|_| vec![[0.0, 0.0]; 16]).collect();
    for (ti, tr) in vtraces.iter_mut().enumerate() {
        tr[0] = [(ti as f32) + 5.0, 1.0];
    }
    let mut vbytes = build_varian_f32_fid(&vtraces);
    // Chop off the last 40 bytes so block 11 is incomplete on disk (leaving 10 complete traces out of 16)!
    vbytes.truncate(vbytes.len() - 40);
    fs::write(vdir.join("fid"), &vbytes).unwrap();

    assert!(
        read_fid_3d(&vdir).is_err(),
        "default read_fid_3d must reject truncated Varian fid"
    );
    assert!(
        process_fid_dir_3d(&vdir, "CBCACONH", false, false).is_err(),
        "default process_fid_dir_3d must reject truncated Varian fid"
    );

    let vfid = read_fid_3d_opts(&vdir, true).expect("opt-in truncated Varian 3D");
    assert_eq!(vfid.truncated_traces, Some((10, 16)));
    let (_vspec, vrec) = process_fid_dir_3d_opts(&vdir, "CBCACONH", false, false, true)
        .expect("process truncated Varian 3D");
    assert!(vrec.allow_truncated_fid);
    assert!(vrec.truncated_fid_zero_filled);
    let vwarn = vrec
        .truncated_fid_warning
        .as_deref()
        .expect("Varian truncated_fid_warning");
    assert!(vwarn.contains("10/16") && vwarn.contains("62.5%"));

    // Also check Varian 2D (`ni2 = 0`) default reject vs opt-in zero-fill
    let procpar_2d = procpar_3d.replace("ni2 1 1 4096 0 1 2 1 1 1 9\n1 2\n0\n", "");
    fs::write(vdir.join("procpar"), procpar_2d).unwrap();
    let v2_traces: Vec<Vec<[f32; 2]>> = (0..3).map(|_| vec![[2.0, 0.5]; 16]).collect();
    fs::write(vdir.join("fid"), build_varian_f32_fid(&v2_traces)).unwrap();
    assert!(read_fid_2d(&vdir).is_err());
    assert!(process_fid_dir_2d(&vdir, true, false).is_err());
    let vfid2 = read_fid_2d_opts(&vdir, true).expect("opt-in truncated Varian 2D");
    assert_eq!(vfid2.truncated_traces, Some((3, 4)));
    let (_vspec2, vrec2) =
        process_fid_dir_2d_opts(&vdir, true, false, true).expect("process truncated Varian 2D");
    assert!(vrec2.truncated_fid_zero_filled);

    let _ = fs::remove_dir_all(&vdir);
}
