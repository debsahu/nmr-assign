//! Comprehensive unit, randomized property, ring-buffer eviction, and bit-exact NEF v1.1
//! session round-trip tests for `nmr_core::session` and `nmr_core::nef` (`T09.6`).

use std::collections::BTreeMap;
use std::f64::consts::{E, PI, SQRT_2};
use std::fs;
use std::path::PathBuf;

use nmr_core::nef::{read_nef_str, write_nef_str, NefDocument};
use nmr_core::nmrstar::read_star;
use nmr_core::peaks::Peak;
use nmr_core::process::{
    ApodizationWindow, BaselineConfig, DimRecipe, NusIstConfig, PhaseSource, ProcessingRecipe,
    SolventFilterConfig,
};
use nmr_core::qc::DatasetQcVerdict;
use nmr_core::rng::Rng;
use nmr_core::session::{
    read_session_nef_file, read_session_nef_str, write_session_nef_file, write_session_nef_str,
    CommandHistory, ResidueAssignmentState, SessionContourParams, WorkbenchCommand,
    WorkbenchSessionState, MAX_UNDO_STEPS,
};
use nmr_core::simulate::{simulate, SimConfig};

fn ubiquitin_fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/fixtures/synthetic_ubiquitin.str")
}

fn sample_processing_recipe(
    direct_label: &str,
    indirect_label: &str,
    nus: bool,
) -> ProcessingRecipe {
    ProcessingRecipe {
        dims: vec![
            DimRecipe {
                label: direct_label.to_string(),
                remove_digital_filter: true,
                group_delay: 67.984_375,
                solvent_filter: Some(SolventFilterConfig {
                    enabled: true,
                    half_width_points: 28,
                }),
                td_eff: Some(1024),
                linear_prediction: false,
                lp_target_points: None,
                lp_order: None,
                window: ApodizationWindow::SquaredSineBell {
                    offset_pi_frac: 0.487_654_321_098_765_4,
                },
                fcor: 0.5,
                si: 2048,
                conjugate_for_fft: false,
                reverse: false,
                hilbert: false,
                p0_deg: Some(-14.375_123_456_789),
                p1_deg: Some(0.25),
                phase_source: PhaseSource::Explicit,
                baseline: Some(BaselineConfig {
                    enabled: true,
                    poly_order: 2,
                    mad_sigma: 3.125,
                }),
                roi_ppm: Some([5.45, 11.55]),
            },
            DimRecipe {
                label: indirect_label.to_string(),
                remove_digital_filter: false,
                group_delay: 0.0,
                solvent_filter: None,
                td_eff: Some(96),
                linear_prediction: true,
                lp_target_points: Some(192),
                lp_order: Some(12),
                window: ApodizationWindow::Exponential { lb_hz: 4.75 },
                fcor: 1.0,
                si: 256,
                conjugate_for_fft: false,
                reverse: true,
                hilbert: true,
                p0_deg: Some(90.0),
                p1_deg: Some(-180.0),
                phase_source: PhaseSource::Stored,
                baseline: None,
                roi_ppm: None,
            },
        ],
        nus: nus.then_some(NusIstConfig {
            iterations: 48,
            initial_thresh_frac: 0.825_123_456_789_012_3,
            final_noise_sigma: 2.35,
            fista_momentum: true,
        }),
        enable_axis_rescue: true,
        axis_rescue_audit: None,
    }
}

#[test]
fn test_all_nine_commands_individual_apply_undo_redo_and_displacement() {
    let mut state = WorkbenchSessionState::from_fasta_sequence("TestProt", "MQIFVKTLTGK");
    let mut history = CommandHistory::new();
    assert_eq!(history.capacity(), MAX_UNDO_STEPS);
    assert_eq!(MAX_UNDO_STEPS, 256);

    // 1. AddPeak
    let s0 = state.clone();
    let p0 = Peak::hsqc(8.4123, 121.456).with_intensity(10500.0, Some(52000.0));
    let p1 = Peak::triple("HNCACB", 8.4123, 121.456, 56.321, 1).with_intensity(4200.0, None);
    let p2 = Peak::triple("HNCACB", 8.4123, 121.456, 31.876, -1).with_intensity(3100.0, None);
    let p3 = Peak::hsqc(7.8950, 117.220).with_intensity(8900.0, Some(41000.0));
    history.execute(&mut state, WorkbenchCommand::AddPeak(p0.clone()));
    assert_eq!(state.peaks.len(), 1);
    history.undo(&mut state).expect("undo AddPeak");
    assert!(state.bitwise_eq(&s0));
    history.redo(&mut state).expect("redo AddPeak");
    assert_eq!(state.peaks, vec![p0.clone()]);

    history.execute(&mut state, WorkbenchCommand::AddPeak(p1.clone()));
    history.execute(&mut state, WorkbenchCommand::AddPeak(p2.clone()));
    history.execute(&mut state, WorkbenchCommand::AddPeak(p3.clone()));
    assert_eq!(state.peaks.len(), 4);

    // 2. DeletePeaks with unsorted + duplicate + out-of-range indices
    let before_del = state.clone();
    history.execute(&mut state, WorkbenchCommand::DeletePeaks(vec![2, 0, 2, 99]));
    assert_eq!(state.peaks, vec![p1.clone(), p3.clone()]);
    let after_del = state.clone();
    history.undo(&mut state).expect("undo DeletePeaks");
    assert!(
        state.bitwise_eq(&before_del),
        "DeletePeaks undo must restore exact peak order & positions"
    );
    history.redo(&mut state).expect("redo DeletePeaks");
    assert!(state.bitwise_eq(&after_del));
    history.undo(&mut state).expect("restore 4 peaks");

    // 3. MovePeak
    let before_move = state.clone();
    history.execute(
        &mut state,
        WorkbenchCommand::MovePeak {
            index: 1,
            new_h: 8.4199,
            new_n: 121.512,
            new_c: Some(56.405),
        },
    );
    assert_eq!(state.peaks[1].h, 8.4199);
    assert_eq!(state.peaks[1].c, Some(56.405));
    let after_move = state.clone();
    history.undo(&mut state).expect("undo MovePeak");
    assert!(state.bitwise_eq(&before_move));
    history.redo(&mut state).expect("redo MovePeak");
    assert!(state.bitwise_eq(&after_move));

    // 4. RecenterPeak
    let before_recenter = state.clone();
    history.execute(
        &mut state,
        WorkbenchCommand::RecenterPeak {
            index: 2,
            fitted_h: 8.4131,
            fitted_n: 121.461,
            fitted_c: Some(31.889),
            new_height: Some(3255.75),
        },
    );
    assert_eq!(state.peaks[2].height, Some(3255.75));
    let after_recenter = state.clone();
    history.undo(&mut state).expect("undo RecenterPeak");
    assert!(state.bitwise_eq(&before_recenter));
    history.redo(&mut state).expect("redo RecenterPeak");
    assert!(state.bitwise_eq(&after_recenter));

    // Seed initial assignments with shifts & locks to test displacement thoroughly!
    let mut shifts_ss2 = BTreeMap::new();
    shifts_ss2.insert("CA".to_string(), 55.8125);
    shifts_ss2.insert("CB".to_string(), 29.4375);
    state.assignments.insert(
        3,
        ResidueAssignmentState::with_shifts(Some(2), shifts_ss2, 0.91),
    );
    state.locked_residues.insert(2, 3);

    let mut shifts_ss5 = BTreeMap::new();
    shifts_ss5.insert("CA".to_string(), 61.125);
    state.assignments.insert(
        6,
        ResidueAssignmentState::with_shifts(Some(5), shifts_ss5, 0.87),
    );
    state.locked_residues.insert(5, 6);
    state.validate_invariants().unwrap();

    // 5. AssignSpinSystem displacing BOTH spin_system=2 from residue 3 AND spin_system=5 at residue 6!
    let before_assign = state.clone();
    history.execute(
        &mut state,
        WorkbenchCommand::AssignSpinSystem {
            spin_system: 2,
            residue_seq_idx: 6,
        },
    );
    state.validate_invariants().unwrap();
    assert_eq!(state.assigned_residue_for_spin_system(2), Some(6));
    assert_eq!(state.assigned_residue_for_spin_system(5), None);
    assert!(!state.assignments.contains_key(&3));
    assert!(!state.is_spin_system_locked(2));
    assert!(!state.is_spin_system_locked(5));
    let after_assign = state.clone();

    history
        .undo(&mut state)
        .expect("undo displaced AssignSpinSystem");
    state.validate_invariants().unwrap();
    assert!(state.bitwise_eq(&before_assign));
    history
        .redo(&mut state)
        .expect("redo displaced AssignSpinSystem");
    state.validate_invariants().unwrap();
    assert!(state.bitwise_eq(&after_assign));
    history
        .undo(&mut state)
        .expect("restore two locked residues");

    // 6. LockResidue (`locked = true` with displacement, then `locked = false`)
    let before_lock = state.clone();
    history.execute(
        &mut state,
        WorkbenchCommand::LockResidue {
            spin_system: 2,
            residue_seq_idx: 6,
            locked: true,
        },
    );
    state.validate_invariants().unwrap();
    assert_eq!(state.locked_residue_for_spin_system(2), Some(6));
    assert_eq!(state.locked_spin_system_for_residue(6), Some(2));
    assert_eq!(state.locked_residue_for_spin_system(5), None);
    let after_lock = state.clone();

    history.undo(&mut state).expect("undo LockResidue(true)");
    state.validate_invariants().unwrap();
    assert!(state.bitwise_eq(&before_lock));
    history.redo(&mut state).expect("redo LockResidue(true)");
    assert!(state.bitwise_eq(&after_lock));

    let before_unlock = state.clone();
    history.execute(
        &mut state,
        WorkbenchCommand::LockResidue {
            spin_system: 2,
            residue_seq_idx: 6,
            locked: false,
        },
    );
    state.validate_invariants().unwrap();
    assert!(!state.is_residue_locked(6));
    assert_eq!(state.assigned_residue_for_spin_system(2), Some(6));
    let after_unlock = state.clone();
    history.undo(&mut state).expect("undo LockResidue(false)");
    assert!(state.bitwise_eq(&before_unlock));
    history.redo(&mut state).expect("redo LockResidue(false)");
    assert!(state.bitwise_eq(&after_unlock));
    // Re-lock so UnassignResidue also tests lock cleanup
    history.undo(&mut state).expect("re-lock 2 -> 6");
    assert!(state.is_residue_locked(6));

    // 7. UnassignResidue
    let before_unassign = state.clone();
    history.execute(
        &mut state,
        WorkbenchCommand::UnassignResidue { residue_seq_idx: 6 },
    );
    state.validate_invariants().unwrap();
    assert!(!state.assignments.contains_key(&6));
    assert!(!state.is_residue_locked(6));
    let after_unassign = state.clone();
    history.undo(&mut state).expect("undo UnassignResidue");
    state.validate_invariants().unwrap();
    assert!(state.bitwise_eq(&before_unassign));
    history.redo(&mut state).expect("redo UnassignResidue");
    assert!(state.bitwise_eq(&after_unassign));

    // 8. SetContourParams (insert new layer + update existing layer)
    let before_contour_insert = state.clone();
    let mut hsqc_layer = SessionContourParams::new("HSQC");
    hsqc_layer.base_level = 1.75;
    hsqc_layer.ppm_nudge = [0.015, -0.12];
    history.execute(
        &mut state,
        WorkbenchCommand::SetContourParams {
            layer_id: "HSQC".to_string(),
            params: hsqc_layer.clone(),
        },
    );
    assert_eq!(state.contour_layers.len(), 1);
    let after_contour_insert = state.clone();
    history
        .undo(&mut state)
        .expect("undo SetContourParams insert");
    assert!(state.bitwise_eq(&before_contour_insert));
    history
        .redo(&mut state)
        .expect("redo SetContourParams insert");
    assert!(state.bitwise_eq(&after_contour_insert));

    let before_contour_update = state.clone();
    hsqc_layer.level_factor = 1.55;
    hsqc_layer.show_negative = false;
    history.execute(
        &mut state,
        WorkbenchCommand::SetContourParams {
            layer_id: "HSQC".to_string(),
            params: hsqc_layer,
        },
    );
    assert_eq!(state.contour_layers.len(), 1);
    assert!(!state.contour_layers[0].show_negative);
    let after_contour_update = state.clone();
    history
        .undo(&mut state)
        .expect("undo SetContourParams update");
    assert!(state.bitwise_eq(&before_contour_update));
    history
        .redo(&mut state)
        .expect("redo SetContourParams update");
    assert!(state.bitwise_eq(&after_contour_update));

    // 9. SetReferenceOffset
    let before_offset = state.clone();
    let new_offsets = [0.018, -0.24, -1.92, -1.88, 0.35];
    history.execute(
        &mut state,
        WorkbenchCommand::SetReferenceOffset {
            offsets: new_offsets,
        },
    );
    assert_eq!(state.reference_offsets, new_offsets);
    let after_offset = state.clone();
    history.undo(&mut state).expect("undo SetReferenceOffset");
    assert!(state.bitwise_eq(&before_offset));
    history.redo(&mut state).expect("redo SetReferenceOffset");
    assert!(state.bitwise_eq(&after_offset));

    // Verify new `execute` clears redo tail
    history.undo(&mut state).expect("step back 1");
    assert!(history.can_redo());
    history.execute(
        &mut state,
        WorkbenchCommand::SetReferenceOffset {
            offsets: [0.0, 0.1, 0.2, 0.3, 0.4],
        },
    );
    assert!(
        !history.can_redo(),
        "new execute must clear the pending redo tail"
    );
    assert_eq!(history.redo_len(), 0);
}

#[test]
fn test_ring_buffer_256_step_bound_eviction_and_exact_replay() {
    let mut state = WorkbenchSessionState::from_fasta_sequence("RingProt", "MQIFVKTLTGKTITLEVEPS");
    let mut history = CommandHistory::new();
    let total_steps = 320usize; // Exceeds 256 by 64 steps

    let mut timeline: Vec<WorkbenchSessionState> = Vec::with_capacity(total_steps + 1);
    timeline.push(state.clone());

    for step in 0..total_steps {
        let cmd = match step % 5 {
            0 => WorkbenchCommand::AddPeak(
                Peak::hsqc(7.5 + (step as f64) * 0.005, 110.0 + (step as f64) * 0.08)
                    .with_intensity(1000.0 + (step as f64) * 13.5, None),
            ),
            1 => WorkbenchCommand::AssignSpinSystem {
                spin_system: step % 14,
                residue_seq_idx: (step * 3) % 20,
            },
            2 => WorkbenchCommand::LockResidue {
                spin_system: (step + 2) % 14,
                residue_seq_idx: (step * 7) % 20,
                locked: step % 2 == 0,
            },
            3 => WorkbenchCommand::SetReferenceOffset {
                offsets: [
                    (step as f64) * 1e-3,
                    -(step as f64) * 5e-3,
                    0.25,
                    -0.25,
                    (step as f64) * 2e-3,
                ],
            },
            _ => {
                let mut params = SessionContourParams::new(format!("L{}", step % 4));
                params.base_level = 1.0 + (step as f32) * 0.05;
                params.num_levels = 8 + ((step % 6) as u32);
                WorkbenchCommand::SetContourParams {
                    layer_id: params.layer_id.clone(),
                    params,
                }
            }
        };
        history.execute(&mut state, cmd);
        state.validate_invariants().unwrap();
        timeline.push(state.clone());
        assert!(history.undo_len() <= MAX_UNDO_STEPS);
    }

    assert_eq!(history.undo_len(), MAX_UNDO_STEPS);
    assert!(history.can_undo());
    assert!(!history.can_redo());

    // Undo all 256 retained steps -> must land bit-for-bit on timeline[320 - 256] == timeline[64]!
    for k in 1..=MAX_UNDO_STEPS {
        let desc = history.undo(&mut state);
        assert!(desc.is_some(), "undo #{k} should succeed");
        state.validate_invariants().unwrap();
        let expected_idx = total_steps - k;
        assert!(
            state.bitwise_eq(&timeline[expected_idx]),
            "mismatch after {k} undos (expected timeline[{expected_idx}])"
        );
    }

    // Step 257 must be None because steps 0..64 were evicted from the 256-step ring buffer
    assert!(!history.can_undo());
    assert_eq!(history.undo(&mut state), None);
    assert!(state.bitwise_eq(&timeline[total_steps - MAX_UNDO_STEPS]));
    assert_eq!(history.redo_len(), MAX_UNDO_STEPS);

    // Redo all 256 steps -> must land bit-for-bit back on timeline[320]!
    for k in 1..=MAX_UNDO_STEPS {
        let desc = history.redo(&mut state);
        assert!(desc.is_some(), "redo #{k} should succeed");
        state.validate_invariants().unwrap();
        let expected_idx = (total_steps - MAX_UNDO_STEPS) + k;
        assert!(
            state.bitwise_eq(&timeline[expected_idx]),
            "mismatch after {k} redos (expected timeline[{expected_idx}])"
        );
    }
    assert!(!history.can_redo());
    assert_eq!(history.redo(&mut state), None);
}

fn gen_random_command(rng: &mut Rng, state: &WorkbenchSessionState) -> WorkbenchCommand {
    let n_res = state.sequence.len().max(1);
    let n_ss = 18usize;
    let variant = (rng.next_u64() % 9) as usize;

    match variant {
        0 => {
            let exps = ["HSQC", "HNCA", "HNCACB", "CBCACONH", "HNCO", "NOESY15N"];
            let exp = exps[(rng.next_u64() as usize) % exps.len()];
            let h = rng.uniform(6.2, 10.4);
            let n = rng.uniform(104.0, 130.0);
            let mut pk = if exp == "HSQC" {
                Peak::hsqc(h, n)
            } else if exp == "NOESY15N" {
                Peak::noesy_3d(
                    "NOESY15N",
                    h,
                    n,
                    rng.uniform(1.0, 8.5),
                    rng.uniform(500.0, 9500.0),
                )
            } else {
                let sign = if exp == "HNCACB" && (rng.next_u64() & 1) == 1 {
                    -1
                } else {
                    1
                };
                Peak::triple(exp, h, n, rng.uniform(16.0, 176.0), sign)
            };
            if (rng.next_u64() & 1) == 0 {
                let vol = ((rng.next_u64() & 1) == 0).then(|| rng.uniform(1000.0, 51_000.0));
                pk = pk.with_intensity(rng.uniform(250.0, 10_250.0), vol);
            }
            WorkbenchCommand::AddPeak(pk)
        }
        1 => {
            let count = ((rng.next_u64() % 5) as usize) + 1;
            let bound = state.peaks.len() + 2; // Intentionally include occasional out-of-range & duplicate indices
            let mut idxs = Vec::with_capacity(count);
            for _ in 0..count {
                idxs.push((rng.next_u64() as usize) % bound.max(1));
            }
            WorkbenchCommand::DeletePeaks(idxs)
        }
        2 => {
            let bound = state.peaks.len() + 1;
            let index = (rng.next_u64() as usize) % bound.max(1);
            let new_c = (!rng.next_u64().is_multiple_of(3)).then(|| rng.uniform(18.0, 176.0));
            WorkbenchCommand::MovePeak {
                index,
                new_h: rng.uniform(6.0, 10.5),
                new_n: rng.uniform(103.0, 131.0),
                new_c,
            }
        }
        3 => {
            let bound = state.peaks.len() + 1;
            let index = (rng.next_u64() as usize) % bound.max(1);
            let fitted_c = ((rng.next_u64() & 1) == 0).then(|| rng.uniform(20.0, 80.0));
            let new_height = ((rng.next_u64() & 1) == 0).then(|| rng.uniform(100.0, 25_100.0));
            WorkbenchCommand::RecenterPeak {
                index,
                fitted_h: rng.uniform(6.5, 10.3),
                fitted_n: rng.uniform(105.0, 130.0),
                fitted_c,
                new_height,
            }
        }
        4 => WorkbenchCommand::AssignSpinSystem {
            spin_system: (rng.next_u64() as usize) % n_ss,
            residue_seq_idx: (rng.next_u64() as usize) % n_res,
        },
        5 => WorkbenchCommand::LockResidue {
            spin_system: (rng.next_u64() as usize) % n_ss,
            residue_seq_idx: (rng.next_u64() as usize) % n_res,
            locked: !rng.next_u64().is_multiple_of(3),
        },
        6 => WorkbenchCommand::UnassignResidue {
            residue_seq_idx: (rng.next_u64() as usize) % n_res,
        },
        7 => {
            let layer_names = ["HSQC", "C13HSQC", "HNCACB_slice", "HNCO_proj"];
            let lid = layer_names[(rng.next_u64() as usize) % layer_names.len()].to_string();
            let params = SessionContourParams {
                layer_id: lid.clone(),
                visible: !rng.next_u64().is_multiple_of(4),
                base_level: rng.uniform(0.5, 8.5) as f32,
                level_factor: rng.uniform(1.15, 1.80) as f32,
                num_levels: 4 + ((rng.next_u64() % 14) as u32),
                show_negative: (rng.next_u64() & 1) == 0,
                pos_rgba: [
                    rng.uniform(0.0, 1.0) as f32,
                    rng.uniform(0.0, 1.0) as f32,
                    rng.uniform(0.0, 1.0) as f32,
                    0.9,
                ],
                neg_rgba: [
                    rng.uniform(0.0, 1.0) as f32,
                    rng.uniform(0.0, 1.0) as f32,
                    rng.uniform(0.0, 1.0) as f32,
                    0.85,
                ],
                ppm_nudge: [
                    rng.uniform(-0.04, 0.04) as f32,
                    rng.uniform(-0.20, 0.20) as f32,
                ],
            };
            WorkbenchCommand::SetContourParams {
                layer_id: lid,
                params,
            }
        }
        _ => WorkbenchCommand::SetReferenceOffset {
            offsets: [
                rng.uniform(-0.1, 0.1),
                rng.uniform(-0.75, 0.75),
                rng.uniform(-1.5, 1.5),
                rng.uniform(-1.5, 1.5),
                rng.uniform(-1.0, 1.0),
            ],
        },
    }
}

#[test]
fn test_randomized_property_all_commands_apply_undo_redo_identity() {
    for seed in 0..16u64 {
        let mut rng = Rng::new(10_000 + seed * 97);
        let mut state = WorkbenchSessionState::from_fasta_sequence(
            format!("PropSeed{seed}"),
            "MQIFVKTLTGKTITLE",
        );
        // Seed initial peaks and shifted assignments so deletions/displacements hit rich state immediately
        for i in 0..8 {
            state.peaks.push(Peak::hsqc(
                8.0 + (i as f64) * 0.11,
                115.0 + (i as f64) * 1.3,
            ));
            let mut sh = BTreeMap::new();
            sh.insert("H".to_string(), 8.0 + (i as f64) * 0.11);
            sh.insert("N".to_string(), 115.0 + (i as f64) * 1.3);
            sh.insert("CA".to_string(), 52.5 + (i as f64) * 1.7);
            state.assignments.insert(
                i,
                ResidueAssignmentState::with_shifts(Some(i), sh, 0.75 + (i as f64) * 0.02),
            );
        }
        state.validate_invariants().unwrap();

        let mut history = CommandHistory::new();
        let mut checkpoints: Vec<WorkbenchSessionState> = vec![state.clone()];

        for step in 0..200usize {
            let before = state.clone();
            let cmd = gen_random_command(&mut rng, &state);
            history.execute(&mut state, cmd);
            state.validate_invariants().unwrap_or_else(|e| {
                panic!("seed {seed} step {step} forward invariant failed: {e}")
            });
            let after = state.clone();

            // Verify immediate `apply -> undo == identity`
            history.undo(&mut state).expect("immediate undo");
            state
                .validate_invariants()
                .unwrap_or_else(|e| panic!("seed {seed} step {step} undo invariant failed: {e}"));
            assert!(
                state.bitwise_eq(&before),
                "seed {seed} step {step}: apply -> undo violated bit-exact identity"
            );

            // Verify immediate `apply -> undo -> redo == apply`
            history.redo(&mut state).expect("immediate redo");
            state
                .validate_invariants()
                .unwrap_or_else(|e| panic!("seed {seed} step {step} redo invariant failed: {e}"));
            assert!(
                state.bitwise_eq(&after),
                "seed {seed} step {step}: apply -> undo -> redo violated bit-exact parity"
            );

            checkpoints.push(state.clone());

            // Every 25 steps, perform a multi-step random undo walk followed by partial redo + branch!
            if (step + 1) % 25 == 0 {
                let max_back = history.undo_len().min(checkpoints.len() - 1).min(18);
                let n_back = ((rng.next_u64() as usize) % max_back) + 1;
                let head_idx = checkpoints.len() - 1;
                for b in 1..=n_back {
                    history.undo(&mut state).expect("multi-step undo");
                    state.validate_invariants().unwrap();
                    assert!(
                        state.bitwise_eq(&checkpoints[head_idx - b]),
                        "seed {seed} step {step}: multi-step undo #{b} mismatch"
                    );
                }
                let n_fwd = (rng.next_u64() as usize) % (n_back + 1);
                for f in 1..=n_fwd {
                    history.redo(&mut state).expect("multi-step redo");
                    state.validate_invariants().unwrap();
                    assert!(
                        state.bitwise_eq(&checkpoints[head_idx - n_back + f]),
                        "seed {seed} step {step}: multi-step redo #{f} mismatch"
                    );
                }
                checkpoints.truncate(head_idx - n_back + n_fwd + 1);
                assert!(state.bitwise_eq(checkpoints.last().unwrap()));
            }
        }

        // Verify bit-exact NEF v1.1 round-trip on the final randomly mutated state of every seed!
        let nef_text = write_session_nef_str(&state);
        let nef_reloaded = read_session_nef_str(&nef_text)
            .unwrap_or_else(|e| panic!("seed {seed} NEF session round-trip failed: {e}"));
        assert!(
            nef_reloaded.bitwise_eq(&state),
            "seed {seed}: randomized NEF round-trip violated bit-exact parity"
        );
    }
}

#[test]
fn test_nef_session_bit_exact_roundtrip_and_standard_saveframes() {
    let ref_data = read_star(ubiquitin_fixture_path(), None).expect("read Ubiquitin fixture");
    let (mut peaks, _) = simulate(
        &ref_data,
        &SimConfig {
            seed: 42,
            ..SimConfig::default()
        },
    );

    // Interleave tricky peaks at end of list (irrational floats, negative zero, 3D NOESY h2, signed CB)
    peaks.push(
        Peak::hsqc(PI * E, 100.0 + E * 7.123_456_789_012_345)
            .with_intensity(12_345.678_901_234_567, Some(987_654.321_098_765_4)),
    );
    peaks.push(Peak::noesy_3d(
        "NOESY15N",
        8.123_456_789_012_346,
        119.987_654_321_098_76,
        4.321_098_765_432_109,
        4_567.890_123_456_789,
    ));
    peaks.push(Peak::triple(
        "HNCACB",
        7.654_321_098_765_432,
        114.321_098_765_432_1,
        28.765_432_109_876_544,
        -1,
    ));

    let mut state = WorkbenchSessionState::from_reference_and_peaks(
        "Human Ubiquitin (1D3Z) #1",
        &ref_data,
        None,
        &peaks,
    );
    state.chain_code = "B".to_string();
    state.seq_ids = (-2..(76 - 2)).collect(); // Negative starting author residue IDs!
    state.view_bounds_ppm = Some([
        10.625_123_456_789_01,
        5.875_987_654_321_098,
        103.251_111_111_111_1,
        133.759_999_999_999_9,
    ]);
    state.reference_offsets = [
        -0.0, // Explicitly verify IEEE-754 negative zero preservation!
        0.135_792_468_013_579_24,
        -1.987_654_321_098_765_4,
        -1.976_543_210_987_654_3,
        SQRT_2 * 0.25,
    ];
    state.qc_verdict = Some(DatasetQcVerdict::Warn);

    // Assign and lock specific spin systems + include a shift-only residue and a spin-system-only residue
    if let Some(entry1) = state.assignments.get_mut(&1) {
        entry1.spin_system = Some(0);
        entry1.confidence = 1.0;
        entry1
            .shifts
            .insert("CA".to_string(), 55.123_456_789_012_346);
    }
    state.locked_residues.insert(0, 1);

    if let Some(entry5) = state.assignments.get_mut(&5) {
        entry5.spin_system = Some(4);
        entry5.confidence = 0.934_567_890_123_456_7;
    }
    state.locked_residues.insert(4, 5);

    // Add a spin-system-only assignment with empty shifts map (e.g. interactive pre-shift assignment)
    state
        .assignments
        .insert(14, ResidueAssignmentState::from_spin_system(11, 1.0));
    state.locked_residues.insert(11, 14);

    state.confirmed_links = vec![(0, 2), (2, 4), (9, 11)];

    let mut layer1 = SessionContourParams::new("HSQC");
    layer1.base_level = 1.333_333_4;
    layer1.level_factor = std::f32::consts::SQRT_2;
    layer1.ppm_nudge = [-0.0, 0.0625];
    let mut layer2 = SessionContourParams::new("save_overlay_HNCACB"); // Reserved `save_*` prefix test!
    layer2.visible = false;
    layer2.show_negative = true;
    layer2.num_levels = 14;
    layer2.pos_rgba = [0.12, 0.84, 0.45, 0.77];
    layer2.neg_rgba = [0.95, 0.22, 0.61, 0.73];
    state.contour_layers = vec![layer1, layer2];

    // Include carbonyl label `"13C'"` containing a single quote inside JSON!
    state.processing_recipes.insert(
        "HNCO_3D".to_string(),
        sample_processing_recipe("1H", "13C'", true),
    );
    state.processing_recipes.insert(
        "HSQC_2D".to_string(),
        sample_processing_recipe("1H", "15N", false),
    );
    state.validate_invariants().unwrap();

    let nef_str_1 = write_session_nef_str(&state);

    // Verify canonical NEF v1.1 saveframes AND vendor-prefixed `save_nmrassign_workbench` tags/loops
    for expected in [
        "save_nef_nmr_meta_data",
        "save_nef_molecular_system",
        "save_nef_chemical_shift_list_default",
        "save_nef_nmr_spectrum_HSQC",
        "save_nef_nmr_spectrum_HNCACB",
        "save_nmrassign_workbench",
        "_nmrassign_workbench.sf_category     nmrassign_workbench",
        "_nmrassign_workbench.sf_framecode    nmrassign_workbench",
        "_nmrassign_workbench.shift_list_crc32",
        "_nmrassign_lock.spin_system_id",
        "_nmrassign_contour_layer.layer_id",
        "_nmrassign_processing_recipe.recipe_json",
    ] {
        assert!(
            nef_str_1.contains(expected),
            "expected NEF block/tag '{expected}' missing from session NEF output"
        );
    }

    // Generic NEF v1.1 reader must still read standard molecular system, shifts, and peak lists cleanly
    let generic_doc = read_nef_str(&nef_str_1).expect("generic NEF parse");
    let generic_ref = generic_doc.to_reference().expect("generic to_reference");
    assert_eq!(generic_ref.sequence.len(), 76);
    assert_eq!(generic_doc.to_peaks().len(), state.peaks.len());

    // Session reader must reconstruct bit-exact `WorkbenchSessionState`
    let reloaded = read_session_nef_str(&nef_str_1).expect("read_session_nef_str");
    assert_eq!(reloaded, state);
    assert!(
        reloaded.bitwise_eq(&state),
        "reloaded WorkbenchSessionState must match original bit-for-bit (IEEE-754 to_bits)"
    );
    assert!(reloaded.reference_offsets[0].is_sign_negative()); // `-0.0` preserved!
    assert!(reloaded.contour_layers[0].ppm_nudge[0].is_sign_negative()); // `-0.0f32` preserved!

    // Second serialization must be 100% byte-identical (`idempotent`)
    let nef_str_2 = write_session_nef_str(&reloaded);
    assert_eq!(nef_str_1, nef_str_2);

    // File I/O round-trip check
    let tmp_path = std::env::temp_dir().join(format!(
        "nmrassign_session_roundtrip_{}.nef",
        std::process::id()
    ));
    write_session_nef_file(&tmp_path, &state).expect("write_session_nef_file");
    let from_disk = read_session_nef_file(&tmp_path).expect("read_session_nef_file");
    let _ = fs::remove_file(&tmp_path);
    assert!(from_disk.bitwise_eq(&state));
}

#[test]
fn test_nef_session_edge_strings_verdicts_and_plain_external_nef_import() {
    // 1. Test edge-case strings (empty protein name, quotes, `#`, `stop_`, `loop_`) and all QC verdicts
    for verdict in [
        None,
        Some(DatasetQcVerdict::Pass),
        Some(DatasetQcVerdict::Warn),
        Some(DatasetQcVerdict::Abstain),
    ] {
        let mut s = WorkbenchSessionState::new("", Vec::<String>::new());
        s.qc_verdict = verdict;
        s.contour_layers.push(SessionContourParams::new("stop_"));
        s.contour_layers.push(SessionContourParams::new("loop_"));
        s.contour_layers
            .push(SessionContourParams::new("quote'and\"hash#layer"));
        s.processing_recipes.insert(
            "spec'with\"quotes".to_string(),
            sample_processing_recipe("1H 'amide'", "15N \"backbone\"", false),
        );
        let nef = write_session_nef_str(&s);
        let back = read_session_nef_str(&nef).expect("edge-case NEF session parse");
        assert!(back.bitwise_eq(&s));
    }

    // 2. Test importing a plain third-party NEF v1.1 file WITHOUT `save_nmrassign_workbench`
    let ref_data = read_star(ubiquitin_fixture_path(), None).expect("read Ubiquitin fixture");
    let plain_peaks = vec![
        Peak::hsqc(8.35, 120.5),
        Peak::triple("HNCA", 8.35, 120.5, 54.2, 1),
    ];
    let plain_doc = NefDocument::from_reference_and_peaks(
        "external_tool_export",
        &ref_data,
        None,
        &plain_peaks,
    );
    assert!(plain_doc.workbench_session.is_none());
    let plain_nef = write_nef_str(&plain_doc);
    assert!(!plain_nef.contains("save_nmrassign_workbench"));

    let imported = read_session_nef_str(&plain_nef).expect("import plain external NEF");
    assert_eq!(imported.protein_name, "external_tool_export");
    assert_eq!(imported.sequence, ref_data.sequence);
    assert_eq!(imported.peaks.len(), 2);
    assert!(imported.locked_residues.is_empty());
    imported.validate_invariants().unwrap();
}

#[test]
fn test_p2_4_and_p2_5_nef_multiline_quotes_legacy_category_and_shift_drift_adoption() {
    // 1. P2-4: Standard `fmt_val` + `NefDocument` must round-trip strings containing BOTH `'` and `"`
    //    AND strings > 80 characters (including leading/trailing whitespace) 100% losslessly via `;...;`
    //    multi-line STAR text blocks.
    let ref_data = read_star(ubiquitin_fixture_path(), None).expect("read Ubiquitin fixture");
    let tricky_both_quotes = "  Peak 'A' & \"B\" (13C' / 1H\" overlap #7)  ";
    let long_over_80 = "Long acquisition & curation note exceeding eighty ASCII characters without losing 'single' or \"double\" quotes!";
    assert!(long_over_80.len() > 80);

    let mut doc = NefDocument::from_reference_and_peaks(
        "p2_4_lossless_test",
        &ref_data,
        None,
        &[
            Peak::hsqc(8.41, 121.2),
            Peak::triple("HNCA", 8.41, 121.2, 55.4, 1),
        ],
    );
    doc.meta_data.coordinate_file_name = Some(long_over_80.to_string());
    doc.peak_lists[0].peaks[0].details = Some(tricky_both_quotes.to_string());

    let nef_txt = write_nef_str(&doc);
    assert!(
        nef_txt.contains(&format!("\n;{tricky_both_quotes}\n;\n")),
        "expected `;...;` STAR block for string with both quote types"
    );
    assert!(
        nef_txt.contains(&format!("\n;{long_over_80}\n;\n")),
        "expected `;...;` STAR block for >80-char string"
    );
    let reparsed_doc = read_nef_str(&nef_txt).expect("reparse NefDocument with `;...;` blocks");
    assert_eq!(
        reparsed_doc.meta_data.coordinate_file_name.as_deref(),
        Some(long_over_80)
    );
    assert_eq!(
        reparsed_doc.peak_lists[0].peaks[0].details.as_deref(),
        Some(tricky_both_quotes)
    );

    // 2. P2-4: `ProcessingRecipe` JSON in `save_nmrassign_workbench` is written as `;...;` block,
    //    `sf_category == sf_framecode == nmrassign_workbench`, and legacy `nmrassign_workbench_session`
    //    category still loads cleanly.
    let mut sess = WorkbenchSessionState::from_fasta_sequence("MiniUbq", ">u\nMQIFVK");
    sess.processing_recipes.insert(
        "HNCO_3D".to_string(),
        sample_processing_recipe("1H", "13C'", true),
    );
    let mut sh0 = BTreeMap::new();
    sh0.insert("H".to_string(), 8.412_345_678_9);
    sh0.insert("N".to_string(), 121.234_567_89);
    sh0.insert("CA".to_string(), 55.123_456_789);
    sh0.insert("CB".to_string(), 29.400_0);
    sess.assignments
        .insert(1, ResidueAssignmentState::with_shifts(Some(0), sh0, 0.95));
    sess.locked_residues.insert(0, 1);

    let sess_nef = write_session_nef_str(&sess);
    assert!(sess_nef.contains("_nmrassign_workbench.sf_category     nmrassign_workbench\n"));
    assert!(sess_nef.contains("_nmrassign_workbench.sf_framecode    nmrassign_workbench\n"));
    assert!(
        sess_nef.contains("\n;{\"dims\":["),
        "ProcessingRecipe JSON must be serialized as a `;...;` STAR text block"
    );
    let clean_reload = read_session_nef_str(&sess_nef).expect("clean session reload");
    assert!(clean_reload.load_warnings.is_empty());
    assert!(clean_reload.bitwise_eq(&sess));

    // Legacy `nmrassign_workbench_session` backward-compatibility read check
    let legacy_nef = sess_nef.replace(
        "_nmrassign_workbench.sf_category     nmrassign_workbench\n",
        "_nmrassign_workbench.sf_category     nmrassign_workbench_session\n",
    );
    let legacy_reload = read_session_nef_str(&legacy_nef).expect("legacy sf_category reload");
    assert!(legacy_reload.bitwise_eq(&sess));

    // 3. P2-5: Simulate external tool editing `save_nef_chemical_shift_list_default`:
    //    - Modify GLN-2 `CA` from `55.1235` -> `56.7500`
    //    - Delete GLN-2 `CB` (`29.4000`)
    //    - Add ILE-3 `CA` = `60.8500`
    let mut edited_doc = read_nef_str(&sess_nef).expect("parse session as NefDocument");
    let shifts_vec = &mut edited_doc.shift_lists[0].shifts;
    for s in shifts_vec.iter_mut() {
        if s.sequence_code == Some(2) && s.atom_name == "CA" {
            s.value = 56.75;
        }
    }
    shifts_vec.retain(|s| !(s.sequence_code == Some(2) && s.atom_name == "CB"));
    shifts_vec.push(nmr_core::nef::AssignedShift {
        chain_code: "A".to_string(),
        sequence_code: Some(3),
        residue_name: "ILE".to_string(),
        atom_name: "CA".to_string(),
        element: Some("C".to_string()),
        isotope_number: Some(13),
        value: 60.85,
        value_uncertainty: Some(0.10),
    });
    // Serialize `edited_doc` WITHOUT regenerating `workbench_session.shift_list_crc32` or `assignment_shifts`!
    let externally_edited_nef = write_nef_str(&edited_doc);
    let adopted =
        read_session_nef_str(&externally_edited_nef).expect("reload externally edited NEF");

    assert_eq!(
        adopted.load_warnings.len(),
        1,
        "expected 1 CRC32 drift warning, got {:?}",
        adopted.load_warnings
    );
    assert!(adopted.load_warnings[0].contains("CRC32"));
    // GLN-2 (res_idx = 1): `CA` adopted `56.75`, `CB` deleted, untouched `H` kept full IEEE-754 precision, lock preserved!
    let r1 = adopted.assignments.get(&1).expect("GLN-2 assignment kept");
    assert_eq!(r1.spin_system, Some(0));
    assert!(adopted.is_residue_locked(1));
    assert_eq!(r1.shifts.get("CA").copied(), Some(56.75));
    assert!(
        !r1.shifts.contains_key("CB"),
        "deleted CB shift must be pruned"
    );
    assert_eq!(
        r1.shifts.get("H").unwrap().to_bits(),
        8.412_345_678_9f64.to_bits(),
        "unedited atom H must retain bit-exact precision"
    );
    // ILE-3 (res_idx = 2): newly added `CA = 60.85` adopted!
    let r2 = adopted
        .assignments
        .get(&2)
        .expect("ILE-3 newly added shift adopted");
    assert_eq!(r2.shifts.get("CA").copied(), Some(60.85));
    adopted.validate_invariants().unwrap();
}
