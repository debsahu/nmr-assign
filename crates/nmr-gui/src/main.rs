//! Interactive WebGPU + `eframe`/`egui` NMR spectrum and backbone resonance assignment workbench.
//!
//! Provenance: Original scientific GUI built on `eframe`, `wgpu`, `nmr-core`, and `nmr-gpu`.

mod contour_painter;

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use contour_painter::{ContourCallback, ContourPipelineResource, ContourUniforms};
use eframe::{egui, egui_wgpu};
use nmr_core::evaluate::{score, to_reference, Metrics};
use nmr_core::nmrstar::{read_star, write_star, Reference};
use nmr_core::peaks::{read_peaks, Peak};
use nmr_core::picker::{estimate_noise_mad, pick_hsqc_2d, PickerConfig};
use nmr_core::priors::{one_to_three, three_to_one, Priors};
use nmr_core::rng::Rng;
use nmr_core::simulate::{
    render_hsqc_2d, render_triple_3d, simulate, SimConfig, SpectrumRenderConfig,
};
use nmr_core::solver::{
    assign_from_matrices, pair_matrix, unary_matrix, AnnealConfig, AssignResult, Matrix2D,
    ScoreConfig, FORBID,
};
use nmr_core::spectrum::{
    read_bruker_pdata_2d, read_nmrpipe_2d, Spectrum2D, Spectrum3D,
};
use nmr_core::spinsystems::{build_spin_systems, SpinSystem, Tolerances};
use nmr_gpu::GpuEngine;

const UBIQUITIN_SEQ: &str =
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG";

fn build_synthetic_ubiquitin() -> Reference {
    if let Ok(r) = read_star("data/fixtures/synthetic_ubiquitin.str", None) {
        return r;
    }
    let pri = Priors::provisional();
    let mut rng = Rng::new(1);
    let seq: Vec<String> = UBIQUITIN_SEQ
        .chars()
        .map(|c| one_to_three(c).to_string())
        .collect();
    let seq_ids: Vec<i32> = (1..=(seq.len() as i32)).collect();
    let mut r = Reference::new(seq.clone(), seq_ids);
    for (k, res) in seq.iter().enumerate() {
        if let Some(atoms) = pri.table.get(res) {
            for (atom, &(mu, sd)) in atoms {
                if atom == "H" && (k == 0 || res == "PRO") {
                    continue;
                }
                r.shifts.insert((k, atom.clone()), rng.normal(mu, sd));
            }
        }
    }
    r
}

struct NmrWorkbenchApp {
    gpu_engine: Option<GpuEngine>,
    use_gpu: bool,
    status_msg: String,
    last_solve_ms: Option<f64>,

    // Active dataset
    priors: Priors,
    reference: Option<Reference>,
    sequence: Vec<String>,
    seq_ids: Vec<i32>,
    peaks: Vec<Peak>,
    truth: Option<Vec<i32>>,
    systems: Vec<SpinSystem>,
    u_mat: Option<Matrix2D>,
    p_mat: Option<Matrix2D>,
    assign_res: Option<AssignResult>,
    pred_ref: Option<Reference>,
    pred_conf: BTreeMap<usize, f64>,
    metrics: Option<Metrics>,

    // Spectra for GPU contour & 3D strip visualization
    hsqc_spec: Option<Spectrum2D>,
    hsqc_arc: Arc<Vec<f32>>,
    hsqc_gen: u64,
    hncacb_3d: Option<Spectrum3D>,

    // 2D HSQC viewport & contour settings
    view_h_left: f32,
    view_h_right: f32,
    view_n_top: f32,
    view_n_bottom: f32,
    contour_base: f32,
    contour_factor: f32,
    contour_levels: u32,
    show_negative: bool,
    show_peak_labels: bool,

    // Solver & simulator controls
    replicas: usize,
    steps_per_sys: usize,
    assign_cost: f64,
    sim_missing: f64,
    sim_spurious: f64,
    sim_seed: u64,
    picker_snr: f32,

    // Selection & File I/O inputs
    selected_ss: Option<usize>,
    selected_res: Option<usize>,
    path_input: String,
    export_prefix: String,
}

impl NmrWorkbenchApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        cc.egui_ctx.set_theme(egui::Theme::Dark);
        let mut style = (*cc.egui_ctx.style_of(egui::Theme::Dark)).clone();
        style.visuals = egui::Visuals::dark();
        style.visuals.panel_fill = egui::Color32::from_rgb(16, 20, 28);
        style.visuals.window_fill = egui::Color32::from_rgb(20, 25, 35);
        cc.egui_ctx.set_style_of(egui::Theme::Dark, style);

        let mut gpu_engine = None;
        if let Some(rs) = &cc.wgpu_render_state {
            let contour_res = ContourPipelineResource::new(&rs.device, rs.target_format);
            rs.renderer
                .write()
                .callback_resources
                .insert(contour_res);

            let info = rs.adapter.get_info();
            gpu_engine = Some(GpuEngine::from_device(
                rs.device.clone(),
                rs.queue.clone(),
                info.name.clone(),
                format!("{:?}", info.backend),
            ));
        }

        let mut app = Self {
            use_gpu: gpu_engine.is_some(),
            gpu_engine,
            status_msg: String::new(),
            last_solve_ms: None,
            priors: Priors::provisional(),
            reference: None,
            sequence: Vec::new(),
            seq_ids: Vec::new(),
            peaks: Vec::new(),
            truth: None,
            systems: Vec::new(),
            u_mat: None,
            p_mat: None,
            assign_res: None,
            pred_ref: None,
            pred_conf: BTreeMap::new(),
            metrics: None,
            hsqc_spec: None,
            hsqc_arc: Arc::new(Vec::new()),
            hsqc_gen: 1,
            hncacb_3d: None,
            view_h_left: 10.4,
            view_h_right: 6.1,
            view_n_top: 103.0,
            view_n_bottom: 135.0,
            contour_base: 8.5,
            contour_factor: 1.42,
            contour_levels: 10,
            show_negative: true,
            show_peak_labels: true,
            replicas: 64,
            steps_per_sys: 1500,
            assign_cost: 0.0,
            sim_missing: 0.06,
            sim_spurious: 0.04,
            sim_seed: 0,
            picker_snr: 5.0,
            selected_ss: None,
            selected_res: None,
            path_input: "data/fixtures/synthetic_ubiquitin.str".to_string(),
            export_prefix: "result_gui".to_string(),
        };

        app.load_demo_and_solve();
        app
    }

    fn reset_view_to_spec(&mut self) {
        if let Some(spec) = &self.hsqc_spec {
            self.view_h_left = spec.axis_x.max_ppm() as f32;
            self.view_h_right = spec.axis_x.min_ppm() as f32;
            self.view_n_top = spec.axis_y.min_ppm() as f32;
            self.view_n_bottom = spec.axis_y.max_ppm() as f32;
            let noise = estimate_noise_mad(&spec.data);
            self.contour_base = (noise * 4.8).max(1.0);
        }
    }

    fn rebuild_systems_and_matrices(&mut self) {
        match build_spin_systems(&self.peaks, &Tolerances::default(), 3, false) {
            Ok(sys) => {
                let scfg = ScoreConfig {
                    assign_cost: self.assign_cost,
                    ..ScoreConfig::default()
                };
                let (u, p) = if self.use_gpu {
                    if let Some(gpu) = &self.gpu_engine {
                        gpu.score_matrices_gpu(&sys, &self.sequence, &self.priors, &scfg)
                            .unwrap_or_else(|_| {
                                (
                                    unary_matrix(&sys, &self.sequence, &self.priors, &scfg),
                                    pair_matrix(&sys, &scfg),
                                )
                            })
                    } else {
                        (
                            unary_matrix(&sys, &self.sequence, &self.priors, &scfg),
                            pair_matrix(&sys, &scfg),
                        )
                    }
                } else {
                    (
                        unary_matrix(&sys, &self.sequence, &self.priors, &scfg),
                        pair_matrix(&sys, &scfg),
                    )
                };
                self.u_mat = Some(u);
                self.p_mat = Some(p);
                self.systems = sys;
                self.selected_ss = (!self.systems.is_empty()).then_some(0);
            }
            Err(e) => {
                self.status_msg = format!("Spin system build error: {e}");
            }
        }
    }

    fn regenerate_synthetic_spectra(&mut self) {
        let rcfg = SpectrumRenderConfig {
            nx_h: 512,
            ny_n: 256,
            nz_c: 96,
            seed: self.sim_seed.wrapping_add(7),
            ..SpectrumRenderConfig::default()
        };
        let hsqc = render_hsqc_2d(&self.peaks, &rcfg);
        self.hsqc_arc = Arc::new(hsqc.data.clone());
        self.hsqc_gen = self.hsqc_gen.wrapping_add(1);
        self.hsqc_spec = Some(hsqc);
        self.hncacb_3d = Some(render_triple_3d("HNCACB", &self.peaks, &rcfg));
        self.reset_view_to_spec();
    }

    fn load_demo_and_solve(&mut self) {
        let ref_data = build_synthetic_ubiquitin();
        self.load_from_reference(ref_data);
        self.run_assignment();
    }

    fn load_from_reference(&mut self, ref_data: Reference) {
        let (peaks, truth) = simulate(
            &ref_data,
            &SimConfig {
                p_missing_peak: self.sim_missing,
                spurious_rate: self.sim_spurious,
                seed: self.sim_seed,
                ..SimConfig::default()
            },
        );
        self.sequence = ref_data.sequence.clone();
        self.seq_ids = ref_data.seq_ids.clone();
        self.reference = Some(ref_data);
        self.peaks = peaks;
        self.truth = Some(truth);
        self.assign_res = None;
        self.pred_ref = None;
        self.pred_conf.clear();
        self.metrics = None;
        self.regenerate_synthetic_spectra();
        self.rebuild_systems_and_matrices();
        self.status_msg = format!(
            "Simulated {} peaks & {} spin systems ({} aa) + 2D HSQC / 3D HNCACB spectra",
            self.peaks.len(),
            self.systems.len(),
            self.sequence.len()
        );
    }

    fn run_assignment(&mut self) {
        if self.systems.is_empty() || self.sequence.is_empty() {
            self.status_msg = "Need spin systems and sequence to run assignment".to_string();
            return;
        }
        let scfg = ScoreConfig {
            assign_cost: self.assign_cost,
            ..ScoreConfig::default()
        };
        let acfg = AnnealConfig {
            runs: self.replicas,
            steps_per_system: self.steps_per_sys,
            seed: self.sim_seed,
            ..AnnealConfig::default()
        };

        let t0 = Instant::now();
        let res_opt = if self.use_gpu {
            if let Some(gpu) = &self.gpu_engine {
                match gpu.assign_gpu(&self.systems, &self.sequence, &self.priors, &scfg, &acfg) {
                    Ok(r) => Some((r, "WebGPU")),
                    Err(e) => {
                        self.status_msg = format!("GPU solver error ({e}), falling back to CPU");
                        None
                    }
                }
            } else {
                None
            }
        } else {
            None
        };

        let (res, engine_tag) = res_opt.unwrap_or_else(|| {
            let u = unary_matrix(&self.systems, &self.sequence, &self.priors, &scfg);
            let p = self
                .p_mat
                .clone()
                .unwrap_or_else(|| pair_matrix(&self.systems, &scfg));
            self.u_mat = Some(u.clone());
            (assign_from_matrices(&u, &p, &acfg), "CPU-MT")
        });

        let dt_ms = t0.elapsed().as_secs_f64() * 1000.0;
        self.last_solve_ms = Some(dt_ms);

        let (pred, conf) = to_reference(
            &res,
            &self.systems,
            &self.sequence,
            &self.seq_ids,
            0.0,
            Some(&self.priors),
        );
        if let Some(truth) = &self.truth {
            if truth.len() == res.position.len() {
                self.metrics = Some(score(&res, truth, 0.8));
            } else {
                self.metrics = None;
            }
        }
        let n_hc = conf.values().filter(|&&c| c >= 0.8).count();
        self.status_msg = format!(
            "[{engine_tag}] Assigned {}/{} residues ({} high-conf >=0.8) in {:.0} ms",
            conf.len(),
            self.sequence.len(),
            n_hc,
            dt_ms
        );
        if let Some(ss) = self.selected_ss {
            if let Some(&k) = res.position.get(ss) {
                if k >= 0 {
                    self.selected_res = Some(k as usize);
                }
            }
        }
        self.pred_ref = Some(pred);
        self.pred_conf = conf;
        self.assign_res = Some(res);
    }

    fn load_path_auto(&mut self) {
        let path_str = self.path_input.trim().to_string();
        let p = Path::new(&path_str);
        if !p.exists() {
            self.status_msg = format!("Path not found: {}", p.display());
            return;
        }
        if p.is_dir() || p.file_name().and_then(|s| s.to_str()) == Some("2rr") {
            let dir = if p.is_dir() { p } else { p.parent().unwrap_or(p) };
            match read_bruker_pdata_2d(dir) {
                Ok(spec) => {
                    self.hsqc_arc = Arc::new(spec.data.clone());
                    self.hsqc_gen = self.hsqc_gen.wrapping_add(1);
                    self.hsqc_spec = Some(spec);
                    self.reset_view_to_spec();
                    self.status_msg = format!("Loaded Bruker 2D pdata from {}", dir.display());
                }
                Err(e) => self.status_msg = format!("Bruker load error: {e}"),
            }
            return;
        }

        let ext = p
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        match ext.as_str() {
            "str" | "star" => match read_star(p, None) {
                Ok(r) => self.load_from_reference(r),
                Err(e) => self.status_msg = format!("NMR-STAR error: {e}"),
            },
            "tsv" => match read_peaks(p) {
                Ok(pks) => {
                    self.peaks = pks;
                    self.truth = None;
                    self.metrics = None;
                    self.regenerate_synthetic_spectra();
                    self.rebuild_systems_and_matrices();
                    self.status_msg =
                        format!("Loaded {} peaks from {}", self.peaks.len(), p.display());
                }
                Err(e) => self.status_msg = format!("Peak TSV error: {e}"),
            },
            "ft2" | "dat" | "pipe" => match read_nmrpipe_2d(p) {
                Ok(spec) => {
                    self.hsqc_arc = Arc::new(spec.data.clone());
                    self.hsqc_gen = self.hsqc_gen.wrapping_add(1);
                    self.hsqc_spec = Some(spec);
                    self.reset_view_to_spec();
                    self.status_msg = format!("Loaded NMRPipe 2D spectrum {}", p.display());
                }
                Err(e) => self.status_msg = format!("NMRPipe load error: {e}"),
            },
            _ => {
                self.status_msg = format!(
                    "Unrecognized file extension .{ext} (expected .str, .tsv, .ft2, or Bruker pdata dir)"
                );
            }
        }
    }

    fn pick_2d_roots(&mut self) {
        let Some(spec) = &self.hsqc_spec else {
            self.status_msg = "No 2D HSQC spectrum loaded".to_string();
            return;
        };
        let picked = pick_hsqc_2d(
            spec,
            &PickerConfig {
                snr_threshold: self.picker_snr,
                ..PickerConfig::default()
            },
        );
        let n_picked = picked.len();
        // Keep existing triple-resonance peaks while replacing HSQC roots
        let mut combined = picked;
        combined.extend(self.peaks.iter().filter(|p| p.exp != "HSQC").cloned());
        self.peaks = combined;
        self.truth = None;
        self.metrics = None;
        self.rebuild_systems_and_matrices();
        self.status_msg = format!(
            "Classical picker found {} HSQC roots (SNR >= {:.1}) -> rebuilt {} spin systems",
            n_picked,
            self.picker_snr,
            self.systems.len()
        );
    }

    fn export_results(&mut self) {
        let Some(pred) = &self.pred_ref else {
            self.status_msg = "Run assignment first before exporting".to_string();
            return;
        };
        let prefix = Path::new(self.export_prefix.trim());
        let str_p = prefix.with_extension("str");
        let tsv_p = prefix.with_extension("tsv");
        if let Err(e) = write_star(&str_p, pred, Some(&self.pred_conf)) {
            self.status_msg = format!("Export failed: {e}");
            return;
        }
        let mut rows = vec!["seq_id\tresidue\tatom\tshift\tconfidence".to_string()];
        for (&(k, ref atom), &v) in &pred.shifts {
            let c = self.pred_conf.get(&k).copied().unwrap_or(0.0);
            rows.push(format!(
                "{}\t{}\t{}\t{:.3}\t{:.2}",
                self.seq_ids[k], self.sequence[k], atom, v, c
            ));
        }
        let _ = std::fs::write(&tsv_p, rows.join("\n") + "\n");
        self.status_msg = format!("Exported {} and {}", str_p.display(), tsv_p.display());
    }

    fn draw_top_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.heading(
                egui::RichText::new("nmr-assign")
                    .strong()
                    .color(egui::Color32::from_rgb(110, 220, 255)),
            );
            if let Some(gpu) = &self.gpu_engine {
                ui.label(
                    egui::RichText::new(format!("⚡ {} ({})", gpu.adapter_name, gpu.backend_name))
                        .small()
                        .color(egui::Color32::from_rgb(130, 230, 175)),
                );
            }
            ui.separator();

            if ui.button("🧬 Ubiquitin Demo").clicked() {
                self.load_demo_and_solve();
            }

            ui.label("Path:");
            ui.add(egui::TextEdit::singleline(&mut self.path_input).desired_width(210.0));
            if ui
                .button("📂 Load (.str / .tsv / pdata / .ft2)")
                .clicked()
            {
                self.load_path_auto();
            }

            ui.separator();
            ui.checkbox(&mut self.use_gpu, "WebGPU Compute");
            ui.add(egui::Slider::new(&mut self.replicas, 8..=256).text("Replicas"));
            ui.add(egui::Slider::new(&mut self.steps_per_sys, 400..=4000).text("Steps/SS"));
            if ui
                .add(
                    egui::Slider::new(&mut self.assign_cost, 0.0..=4.0)
                        .text("Assign cost")
                        .fixed_decimals(2),
                )
                .changed()
                && !self.systems.is_empty()
                && !self.sequence.is_empty()
            {
                let scfg = ScoreConfig {
                    assign_cost: self.assign_cost,
                    ..ScoreConfig::default()
                };
                self.u_mat = Some(unary_matrix(&self.systems, &self.sequence, &self.priors, &scfg));
            }

            let solve_btn = egui::Button::new(
                egui::RichText::new("▶ Run Assignment")
                    .strong()
                    .color(egui::Color32::from_rgb(15, 25, 20)),
            )
            .fill(egui::Color32::from_rgb(82, 224, 152));
            if ui.add(solve_btn).clicked() {
                self.run_assignment();
            }

            if ui.button("💾 Export").clicked() {
                self.export_results();
            }
        });

        ui.add_space(3.0);
        ui.horizontal_wrapped(|ui| {
            ui.label(egui::RichText::new("Simulator / Picker:").small().weak());
            let mut changed_sim = false;
            changed_sim |= ui
                .add(
                    egui::Slider::new(&mut self.sim_missing, 0.0..=0.25)
                        .text("Missing")
                        .fixed_decimals(2),
                )
                .changed();
            changed_sim |= ui
                .add(
                    egui::Slider::new(&mut self.sim_spurious, 0.0..=0.20)
                        .text("Spurious")
                        .fixed_decimals(2),
                )
                .changed();
            let _ = changed_sim;
            if ui.small_button("🎲 Re-simulate & Solve").clicked() {
                self.sim_seed = self.sim_seed.wrapping_add(1);
                if let Some(r) = self.reference.clone() {
                    self.load_from_reference(r);
                    self.run_assignment();
                }
            }
            ui.separator();
            ui.add(egui::Slider::new(&mut self.picker_snr, 3.0..=12.0).text("Pick SNR"));
            if ui.small_button("🔍 Pick 2D HSQC Peaks").clicked() {
                self.pick_2d_roots();
            }
            ui.separator();
            ui.label(
                egui::RichText::new(&self.status_msg)
                    .small()
                    .color(egui::Color32::from_rgb(190, 210, 235)),
            );
        });
    }

    fn draw_sequence_ribbon(&mut self, ui: &mut egui::Ui) {
        if self.sequence.is_empty() {
            return;
        }
        let mut res_to_ss: BTreeMap<usize, (usize, f64)> = BTreeMap::new();
        if let Some(res) = &self.assign_res {
            for (s, &k) in res.position.iter().enumerate() {
                if k >= 0 {
                    res_to_ss.insert(k as usize, (s, res.confidence[s]));
                }
            }
        }

        egui::ScrollArea::horizontal()
            .id_salt("seq_ribbon_scroll")
            .max_height(52.0)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 3.0;
                    for (k, three) in self.sequence.iter().enumerate() {
                        let one = three_to_one(three);
                        let seq_id = self.seq_ids.get(k).copied().unwrap_or((k + 1) as i32);
                        let is_pro_or_nterm = k == 0 || three == "PRO";
                        let assigned = res_to_ss.get(&k).copied();
                        let is_selected = self.selected_res == Some(k);

                        let fill = if is_pro_or_nterm {
                            egui::Color32::from_rgb(38, 43, 54)
                        } else if let Some((_, conf)) = assigned {
                            if conf >= 0.80 {
                                egui::Color32::from_rgb(28, 125, 82)
                            } else if conf >= 0.50 {
                                egui::Color32::from_rgb(148, 112, 28)
                            } else {
                                egui::Color32::from_rgb(145, 68, 42)
                            }
                        } else {
                            egui::Color32::from_rgb(48, 56, 74)
                        };

                        let stroke = if is_selected {
                            egui::Stroke::new(2.0, egui::Color32::from_rgb(120, 235, 255))
                        } else {
                            egui::Stroke::new(1.0, egui::Color32::from_rgb(65, 76, 98))
                        };

                        let label = format!("{one}{seq_id}");
                        let btn = egui::Button::new(
                            egui::RichText::new(label)
                                .monospace()
                                .size(11.5)
                                .color(egui::Color32::WHITE),
                        )
                        .fill(fill)
                        .stroke(stroke)
                        .min_size(egui::vec2(34.0, 28.0));

                        let resp = ui.add(btn);
                        if resp.clicked() {
                            self.selected_res = Some(k);
                            if let Some((ss_idx, _)) = assigned {
                                self.selected_ss = Some(ss_idx);
                            }
                        }
                        resp.on_hover_ui(|ui| {
                            ui.label(egui::RichText::new(format!("{three} {seq_id}")).strong());
                            if let Some((ss_idx, conf)) = assigned {
                                ui.label(format!("Spin system #{ss_idx} (conf {:.0}%)", conf * 100.0));
                            } else if is_pro_or_nterm {
                                ui.label("No amide H root (N-term / Proline)");
                            } else {
                                ui.label("Unassigned");
                            }
                            if let Some(pred) = &self.pred_ref {
                                for atom in ["H", "N", "CA", "CB", "C"] {
                                    if let Some(&v) = pred.shifts.get(&(k, atom.to_string())) {
                                        ui.monospace(format!("  {atom:2}: {v:7.3} ppm"));
                                    }
                                }
                            }
                        });
                    }
                });
            });
    }

    fn draw_hsqc_canvas(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("2D ¹H–¹⁵N HSQC (WebGPU Shader)").strong());
            ui.add(
                egui::Slider::new(&mut self.contour_base, 0.5..=60.0)
                    .logarithmic(true)
                    .text("Base"),
            );
            ui.add(egui::Slider::new(&mut self.contour_factor, 1.15..=2.0).text("Step"));
            ui.add(egui::Slider::new(&mut self.contour_levels, 4..=16).text("Levels"));
            ui.checkbox(&mut self.show_negative, "Neg");
            ui.checkbox(&mut self.show_peak_labels, "Labels");
            if ui.small_button("Reset Zoom").clicked() {
                self.reset_view_to_spec();
            }
        });

        let avail = ui.available_size();
        let canvas_size = egui::vec2(avail.x.max(200.0), (avail.y - 4.0).max(200.0));
        let (rect, response) =
            ui.allocate_exact_size(canvas_size, egui::Sense::click_and_drag());

        // Pan with drag
        if response.dragged() {
            let delta = response.drag_delta();
            let span_h = self.view_h_right - self.view_h_left;
            let span_n = self.view_n_bottom - self.view_n_top;
            let dh = -delta.x / rect.width().max(1.0) * span_h;
            let dn = -delta.y / rect.height().max(1.0) * span_n;
            self.view_h_left += dh;
            self.view_h_right += dh;
            self.view_n_top += dn;
            self.view_n_bottom += dn;
        }
        if response.double_clicked() {
            self.reset_view_to_spec();
        }

        // Scroll-wheel zoom around pointer
        if let Some(hover_pos) = response.hover_pos() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll.abs() > 0.1 {
                let zoom = (1.0 - scroll * 0.0025).clamp(0.75, 1.35);
                let u = ((hover_pos.x - rect.left()) / rect.width()).clamp(0.0, 1.0);
                let v = ((hover_pos.y - rect.top()) / rect.height()).clamp(0.0, 1.0);
                let anchor_h = self.view_h_left + u * (self.view_h_right - self.view_h_left);
                let anchor_n = self.view_n_top + v * (self.view_n_bottom - self.view_n_top);
                self.view_h_left = anchor_h + (self.view_h_left - anchor_h) * zoom;
                self.view_h_right = anchor_h + (self.view_h_right - anchor_h) * zoom;
                self.view_n_top = anchor_n + (self.view_n_top - anchor_n) * zoom;
                self.view_n_bottom = anchor_n + (self.view_n_bottom - anchor_n) * zoom;
            }
        }

        let painter = ui.painter_at(rect);
        if let Some(spec) = &self.hsqc_spec {
            let uniforms = ContourUniforms {
                view_ppm: [
                    self.view_h_left,
                    self.view_h_right,
                    self.view_n_top,
                    self.view_n_bottom,
                ],
                axis_ppm: [
                    spec.axis_x.first_ppm as f32,
                    spec.axis_x.step_ppm as f32,
                    spec.axis_y.first_ppm as f32,
                    spec.axis_y.step_ppm as f32,
                ],
                dims: [
                    spec.nx() as f32,
                    spec.ny() as f32,
                    rect.width(),
                    rect.height(),
                ],
                levels: [
                    self.contour_base,
                    self.contour_factor,
                    self.contour_levels as f32,
                    if self.show_negative { 1.0 } else { 0.0 },
                ],
            };
            let cb = egui_wgpu::Callback::new_paint_callback(
                rect,
                ContourCallback {
                    uniforms,
                    spec_gen: self.hsqc_gen,
                    spec_data: Arc::clone(&self.hsqc_arc),
                },
            );
            painter.add(cb);
        } else {
            painter.rect_filled(rect, 4.0, egui::Color32::from_rgb(14, 18, 25));
        }

        let ppm_to_screen = |h: f64, n: f64| -> egui::Pos2 {
            let u = ((h as f32) - self.view_h_left) / (self.view_h_right - self.view_h_left);
            let v = ((n as f32) - self.view_n_top) / (self.view_n_bottom - self.view_n_top);
            egui::pos2(rect.left() + u * rect.width(), rect.top() + v * rect.height())
        };

        // Click to select nearest spin system
        if response.clicked() {
            if let Some(click_pos) = response.interact_pointer_pos() {
                let mut best: Option<(usize, f32)> = None;
                for (s, ss) in self.systems.iter().enumerate() {
                    let pt = ppm_to_screen(ss.h, ss.n);
                    let dist = pt.distance(click_pos);
                    if dist <= 18.0 && best.map_or(true, |(_, bd)| dist < bd) {
                        best = Some((s, dist));
                    }
                }
                if let Some((s, _)) = best {
                    self.selected_ss = Some(s);
                    if let Some(res) = &self.assign_res {
                        if res.position[s] >= 0 {
                            self.selected_res = Some(res.position[s] as usize);
                        }
                    }
                }
            }
        }

        // Draw spin system peak markers and residue labels
        for (s, ss) in self.systems.iter().enumerate() {
            let pt = ppm_to_screen(ss.h, ss.n);
            if !rect.contains(pt) {
                continue;
            }
            let is_sel = self.selected_ss == Some(s);
            let (assigned_k, conf) = self
                .assign_res
                .as_ref()
                .map(|r| (r.position[s], r.confidence[s]))
                .unwrap_or((-1, 0.0));

            let marker_col = if assigned_k >= 0 {
                if conf >= 0.80 {
                    egui::Color32::from_rgb(95, 245, 165)
                } else {
                    egui::Color32::from_rgb(255, 205, 95)
                }
            } else {
                egui::Color32::from_rgb(155, 175, 205)
            };

            let sz = if is_sel { 6.5 } else { 3.8 };
            painter.line_segment(
                [egui::pos2(pt.x - sz, pt.y), egui::pos2(pt.x + sz, pt.y)],
                egui::Stroke::new(if is_sel { 2.0 } else { 1.2 }, marker_col),
            );
            painter.line_segment(
                [egui::pos2(pt.x, pt.y - sz), egui::pos2(pt.x, pt.y + sz)],
                egui::Stroke::new(if is_sel { 2.0 } else { 1.2 }, marker_col),
            );
            if is_sel {
                painter.circle_stroke(
                    pt,
                    10.5,
                    egui::Stroke::new(2.0, egui::Color32::from_rgb(255, 240, 120)),
                );
            }

            if self.show_peak_labels {
                let text = if assigned_k >= 0 && (assigned_k as usize) < self.sequence.len() {
                    let k = assigned_k as usize;
                    format!("{}{}", three_to_one(&self.sequence[k]), self.seq_ids[k])
                } else {
                    format!("#{s}")
                };
                painter.text(
                    egui::pos2(pt.x + 6.0, pt.y - 6.0),
                    egui::Align2::LEFT_BOTTOM,
                    text,
                    egui::FontId::monospace(10.5),
                    marker_col,
                );
            }
        }

        // Draw axis ticks and live cursor ppm readout
        painter.rect_stroke(
            rect,
            2.0,
            egui::Stroke::new(1.0, egui::Color32::from_rgb(65, 80, 105)),
            egui::StrokeKind::Inside,
        );
        for i in 0..=5 {
            let t = (i as f32) / 5.0;
            let h_val = self.view_h_left + t * (self.view_h_right - self.view_h_left);
            let x = rect.left() + t * rect.width();
            painter.text(
                egui::pos2(x, rect.bottom() - 3.0),
                egui::Align2::CENTER_BOTTOM,
                format!("{h_val:.2} ¹H"),
                egui::FontId::monospace(10.0),
                egui::Color32::from_rgb(170, 188, 215),
            );

            let n_val = self.view_n_top + t * (self.view_n_bottom - self.view_n_top);
            let y = rect.top() + t * rect.height();
            painter.text(
                egui::pos2(rect.left() + 4.0, y),
                egui::Align2::LEFT_CENTER,
                format!("{n_val:.1}"),
                egui::FontId::monospace(10.0),
                egui::Color32::from_rgb(170, 188, 215),
            );
        }

        if let Some(hp) = response.hover_pos() {
            let u = (hp.x - rect.left()) / rect.width();
            let v = (hp.y - rect.top()) / rect.height();
            let cur_h = self.view_h_left + u * (self.view_h_right - self.view_h_left);
            let cur_n = self.view_n_top + v * (self.view_n_bottom - self.view_n_top);
            painter.text(
                egui::pos2(rect.right() - 8.0, rect.top() + 6.0),
                egui::Align2::RIGHT_TOP,
                format!("¹H {cur_h:.3} ppm | ¹⁵N {cur_n:.2} ppm"),
                egui::FontId::monospace(11.0),
                egui::Color32::from_rgb(220, 235, 255),
            );
        }
    }

    fn draw_right_inspector(&mut self, ui: &mut egui::Ui) {
        // 1. Metrics summary card if reference truth is present
        if let Some(m) = &self.metrics {
            egui::Frame::group(ui.style())
                .fill(egui::Color32::from_rgb(22, 29, 40))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("Benchmark vs Truth").strong());
                        if let Some(ms) = self.last_solve_ms {
                            ui.label(
                                egui::RichText::new(format!("{ms:.0} ms"))
                                    .monospace()
                                    .color(egui::Color32::from_rgb(120, 225, 255)),
                            );
                        }
                    });
                    ui.columns(3, |cols| {
                        cols[0].vertical_centered(|ui| {
                            ui.label(egui::RichText::new("Accuracy").small().weak());
                            ui.label(
                                egui::RichText::new(format!("{:.1}%", m.accuracy() * 100.0))
                                    .strong()
                                    .size(16.0),
                            );
                        });
                        cols[1].vertical_centered(|ui| {
                            ui.label(egui::RichText::new("Recall").small().weak());
                            ui.label(
                                egui::RichText::new(format!("{:.1}%", m.recall() * 100.0))
                                    .strong()
                                    .size(16.0),
                            );
                        });
                        cols[2].vertical_centered(|ui| {
                            ui.label(egui::RichText::new("HC Prec (>=0.8)").small().weak());
                            ui.label(
                                egui::RichText::new(format!(
                                    "{:.1}% ({}/{})",
                                    m.hc_precision() * 100.0,
                                    m.hc_correct,
                                    m.hc_assigned
                                ))
                                .strong()
                                .size(15.0)
                                .color(egui::Color32::from_rgb(95, 240, 165)),
                            );
                        });
                    });
                });
            ui.add_space(6.0);
        }

        // 2. 3D Triple-Resonance Sequential Walk Strip Plot (k-1, k, k+1)
        ui.label(
            egui::RichText::new("3D HNCACB / CBCA(CO)NH Sequential Walk Strips")
                .strong(),
        );
        ui.label(
            egui::RichText::new("Cyan = +Cα, Coral = −Cβ, Gold ◆ = (i−1), Dashed = sequential tie")
                .small()
                .weak(),
        );

        let Some(sel_s) = self.selected_ss.filter(|&s| s < self.systems.len()) else {
            ui.label("Click any HSQC peak or assigned residue to inspect 3D strips.");
            return;
        };

        // Determine predecessor and successor spin systems (either from sequence assignment or best P[s,t])
        let (pred_s, succ_s) = self.find_walk_neighbors(sel_s);
        let strip_list: Vec<(Option<usize>, &str)> =
            vec![(pred_s, "i − 1"), (Some(sel_s), "Selected (i)"), (succ_s, "i + 1")];

        let strip_h = 275.0f32;
        let (strip_rect, _) = ui.allocate_exact_size(
            egui::vec2(ui.available_width().max(240.0), strip_h),
            egui::Sense::hover(),
        );
        self.paint_sequential_strips(ui.painter_at(strip_rect), strip_rect, &strip_list);

        ui.add_space(8.0);
        ui.separator();

        // 3. Detailed Spin System Candidate & LLR / Connectivity Inspector
        egui::ScrollArea::vertical()
            .id_salt("ss_inspector_scroll")
            .show(ui, |ui| {
                let ss = &self.systems[sel_s];
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new(format!(
                            "Spin System #{sel_s}  (¹H {:.3}, ¹⁵N {:.2})",
                            ss.h, ss.n
                        ))
                        .strong(),
                    );
                    if let Some(res) = &self.assign_res {
                        let k = res.position[sel_s];
                        if k >= 0 && (k as usize) < self.sequence.len() {
                            let ku = k as usize;
                            ui.label(
                                egui::RichText::new(format!(
                                    "→ {}{} ({:.0}%)",
                                    self.sequence[ku],
                                    self.seq_ids[ku],
                                    res.confidence[sel_s] * 100.0
                                ))
                                .color(egui::Color32::from_rgb(95, 240, 165))
                                .strong(),
                            );
                        }
                    }
                });

                egui::Grid::new("cands_grid")
                    .striped(true)
                    .min_col_width(58.0)
                    .show(ui, |ui| {
                        ui.strong("Atom");
                        ui.strong("Offset");
                        ui.strong("Candidates (ppm × support)");
                        ui.end_row();

                        for atom in ["CA", "CB", "C"] {
                            for (off, label) in [(0, "i"), (-1, "i − 1")] {
                                if let Some(cands) = ss.cands.get(&(atom.to_string(), off)) {
                                    ui.monospace(atom);
                                    ui.monospace(label);
                                    let text = cands
                                        .iter()
                                        .map(|(v, sup)| format!("{v:.2} (×{sup})"))
                                        .collect::<Vec<_>>()
                                        .join(", ");
                                    ui.monospace(text);
                                    ui.end_row();
                                }
                            }
                        }
                    });

                if let Some(u_mat) = &self.u_mat {
                    ui.add_space(6.0);
                    ui.label(egui::RichText::new("Top Residue Typing LLR U(s, k):").small().strong());
                    let mut ranked: Vec<(usize, f64)> = (0..u_mat.cols)
                        .map(|k| (k, u_mat.get(sel_s, k)))
                        .filter(|&(_, v)| v > FORBID * 0.5)
                        .collect();
                    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
                    let top_txt = if ranked.is_empty() {
                        "0 observed carbons (excluded from backbone)".to_string()
                    } else {
                        ranked
                            .iter()
                            .take(5)
                            .map(|&(k, sc)| format!("{}{}: {:+.1}", self.sequence[k], self.seq_ids[k], sc))
                            .collect::<Vec<_>>()
                            .join("  |  ")
                    };
                    ui.monospace(top_txt);
                }

                if let Some(p_mat) = &self.p_mat {
                    ui.add_space(4.0);
                    ui.label(
                        egui::RichText::new("Best Sequential Successors P(s → t):")
                            .small()
                            .strong(),
                    );
                    let mut succs: Vec<(usize, f64)> = (0..p_mat.cols)
                        .filter(|&t| t != sel_s)
                        .map(|t| (t, p_mat.get(sel_s, t)))
                        .filter(|&(_, v)| v > -20.0 && v.abs() > 1e-6)
                        .collect();
                    succs.sort_by(|a, b| b.1.total_cmp(&a.1));
                    for (t, sc) in succs.into_iter().take(4) {
                        if ui
                            .small_button(format!("Jump → SS #{t} (P = {sc:+.2})"))
                            .clicked()
                        {
                            self.selected_ss = Some(t);
                        }
                    }
                }
            });
    }

    fn find_walk_neighbors(&self, sel_s: usize) -> (Option<usize>, Option<usize>) {
        if let Some(res) = &self.assign_res {
            let k = res.position[sel_s];
            if k >= 0 {
                let pred = res
                    .position
                    .iter()
                    .position(|&pk| pk == k - 1);
                let succ = res
                    .position
                    .iter()
                    .position(|&pk| pk == k + 1);
                return (pred, succ);
            }
        }
        if let Some(p_mat) = &self.p_mat {
            let pred = (0..p_mat.rows)
                .filter(|&a| a != sel_s)
                .max_by(|&a, &b| p_mat.get(a, sel_s).total_cmp(&p_mat.get(b, sel_s)))
                .filter(|&a| p_mat.get(a, sel_s) > 0.0);
            let succ = (0..p_mat.cols)
                .filter(|&b| b != sel_s)
                .max_by(|&a, &b| p_mat.get(sel_s, a).total_cmp(&p_mat.get(sel_s, b)))
                .filter(|&b| p_mat.get(sel_s, b) > 0.0);
            return (pred, succ);
        }
        (None, None)
    }

    fn paint_sequential_strips(
        &self,
        painter: egui::Painter,
        rect: egui::Rect,
        strips: &[(Option<usize>, &str)],
    ) {
        painter.rect_filled(rect, 4.0, egui::Color32::from_rgb(13, 17, 24));
        let c_top = 13.0f64;
        let c_bot = 74.0f64;
        let top_pad = 22.0f32;
        let bot_pad = 8.0f32;
        let left_axis_w = 34.0f32;

        let plot_top = rect.top() + top_pad;
        let plot_bot = rect.bottom() - bot_pad;
        let c_to_y = |c_ppm: f64| -> f32 {
            let t = ((c_ppm - c_top) / (c_bot - c_top)) as f32;
            plot_top + t.clamp(0.0, 1.0) * (plot_bot - plot_top)
        };

        // 13C ppm scale on left
        for c_tick in [15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 72.0] {
            let y = c_to_y(c_tick);
            painter.text(
                egui::pos2(rect.left() + 3.0, y),
                egui::Align2::LEFT_CENTER,
                format!("{c_tick:.0}"),
                egui::FontId::monospace(9.5),
                egui::Color32::from_rgb(145, 165, 190),
            );
            painter.line_segment(
                [
                    egui::pos2(rect.left() + left_axis_w, y),
                    egui::pos2(rect.right() - 4.0, y),
                ],
                egui::Stroke::new(0.5, egui::Color32::from_rgb(28, 35, 48)),
            );
        }

        let total_w = (rect.width() - left_axis_w - 6.0).max(90.0);
        let col_w = total_w / (strips.len() as f32);

        let mut col_centers = Vec::with_capacity(strips.len());

        for (idx, &(ss_opt, role)) in strips.iter().enumerate() {
            let x0 = rect.left() + left_axis_w + (idx as f32) * col_w + 3.0;
            let x1 = x0 + col_w - 6.0;
            let cx = 0.5 * (x0 + x1);
            col_centers.push(cx);

            let strip_box = egui::Rect::from_min_max(egui::pos2(x0, plot_top), egui::pos2(x1, plot_bot));
            painter.rect_stroke(
                strip_box,
                2.0,
                egui::Stroke::new(
                    if idx == 1 { 1.5 } else { 1.0 },
                    if idx == 1 {
                        egui::Color32::from_rgb(95, 200, 255)
                    } else {
                        egui::Color32::from_rgb(50, 62, 82)
                    },
                ),
                egui::StrokeKind::Inside,
            );

            let Some(s) = ss_opt else {
                painter.text(
                    egui::pos2(cx, rect.top() + 10.0),
                    egui::Align2::CENTER_CENTER,
                    format!("{role}: —"),
                    egui::FontId::monospace(10.0),
                    egui::Color32::GRAY,
                );
                continue;
            };
            let ss = &self.systems[s];

            // Header title (assigned residue or SS#)
            let title = self
                .assign_res
                .as_ref()
                .and_then(|r| (r.position[s] >= 0).then(|| r.position[s] as usize))
                .map(|k| format!("{}{} (#{s})", three_to_one(&self.sequence[k]), self.seq_ids[k]))
                .unwrap_or_else(|| format!("SS #{s}"));
            painter.text(
                egui::pos2(cx, rect.top() + 10.0),
                egui::Align2::CENTER_CENTER,
                title,
                egui::FontId::monospace(10.5),
                egui::Color32::from_rgb(225, 238, 255),
            );

            // Render 3D HNCACB strip slice intensity heatmap if available
            if let Some(s3) = &self.hncacb_3d {
                let slice = s3.extract_strip(ss.h, ss.n, 0.04);
                let max_v = slice.max_abs().max(1.0);
                for ic in 0..slice.ny() {
                    let c_ppm = slice.axis_y.index_to_ppm(ic as f64);
                    if !(c_top..=c_bot).contains(&c_ppm) {
                        continue;
                    }
                    let y = c_to_y(c_ppm);
                    let mid_x = slice.nx() / 2;
                    let val = slice.get(mid_x, ic);
                    let norm = (val.abs() / max_v).clamp(0.0, 1.0);
                    if norm > 0.14 {
                        let col = if val >= 0.0 {
                            egui::Color32::from_rgba_unmultiplied(40, 195, 255, (norm * 130.0) as u8)
                        } else {
                            egui::Color32::from_rgba_unmultiplied(255, 95, 105, (norm * 130.0) as u8)
                        };
                        let half_bar = norm * (strip_box.width() * 0.36);
                        painter.line_segment(
                            [egui::pos2(cx - half_bar, y), egui::pos2(cx + half_bar, y)],
                            egui::Stroke::new(2.2, col),
                        );
                    }
                }
            }

            // Plot resolved spin-system candidates in strip: (i) solid circles, (i-1) diamonds
            for atom in ["CA", "CB"] {
                let base_col = if atom == "CA" {
                    egui::Color32::from_rgb(80, 230, 255)
                } else {
                    egui::Color32::from_rgb(255, 115, 120)
                };
                if let Some(cands_i) = ss.cands.get(&(atom.to_string(), 0)) {
                    if let Some(&(ppm, sup)) = cands_i.first() {
                        if sup > 0 {
                            let y = c_to_y(ppm);
                            painter.circle_filled(egui::pos2(cx - 8.0, y), 4.2, base_col);
                            painter.text(
                                egui::pos2(cx - 14.0, y),
                                egui::Align2::RIGHT_CENTER,
                                format!("{ppm:.1}"),
                                egui::FontId::monospace(8.8),
                                base_col,
                            );
                        }
                    }
                }
                if let Some(cands_im1) = ss.cands.get(&(atom.to_string(), -1)) {
                    if let Some(&(ppm, _)) = cands_im1.first() {
                        let y = c_to_y(ppm);
                        let d_col = egui::Color32::from_rgb(250, 205, 90);
                        let pts = vec![
                            egui::pos2(cx + 8.0, y - 4.2),
                            egui::pos2(cx + 12.2, y),
                            egui::pos2(cx + 8.0, y + 4.2),
                            egui::pos2(cx + 3.8, y),
                        ];
                        painter.add(egui::Shape::convex_polygon(
                            pts,
                            d_col,
                            egui::Stroke::NONE,
                        ));
                    }
                }
            }
        }

        // Draw sequential connectivity tie-lines between strip[w] (offset 0) and strip[w+1] (offset -1)
        for w in 0..strips.len().saturating_sub(1) {
            let (Some(sa), Some(sb)) = (strips[w].0, strips[w + 1].0) else {
                continue;
            };
            for atom in ["CA", "CB"] {
                let Some(va) = self.systems[sa]
                    .cands
                    .get(&(atom.to_string(), 0))
                    .and_then(|c| c.first())
                    .filter(|c| c.1 > 0)
                    .map(|c| c.0)
                else {
                    continue;
                };
                let Some(vb) = self.systems[sb]
                    .cands
                    .get(&(atom.to_string(), -1))
                    .and_then(|c| c.first())
                    .map(|c| c.0)
                else {
                    continue;
                };
                if (va - vb).abs() <= 0.45 {
                    let ya = c_to_y(va);
                    let yb = c_to_y(vb);
                    let x_start = col_centers[w] - 4.0;
                    let x_end = col_centers[w + 1] + 4.0;
                    let col = if atom == "CA" {
                        egui::Color32::from_rgba_unmultiplied(110, 245, 255, 210)
                    } else {
                        egui::Color32::from_rgba_unmultiplied(255, 145, 145, 210)
                    };
                    // Dashed horizontal connector
                    let segs = 9;
                    for j in (0..segs).step_by(2) {
                        let t0 = (j as f32) / (segs as f32);
                        let t1 = ((j + 1) as f32) / (segs as f32);
                        painter.line_segment(
                            [
                                egui::pos2(x_start + t0 * (x_end - x_start), ya + t0 * (yb - ya)),
                                egui::pos2(x_start + t1 * (x_end - x_start), ya + t1 * (yb - ya)),
                            ],
                            egui::Stroke::new(1.5, col),
                        );
                    }
                }
            }
        }
    }
}

impl eframe::App for NmrWorkbenchApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::Panel::top("top_controls_panel").show(ui, |ui| {
            self.draw_top_bar(ui);
            ui.add_space(4.0);
            self.draw_sequence_ribbon(ui);
        });

        egui::Panel::right("right_strip_inspector")
            .default_size(390.0)
            .min_size(320.0)
            .show(ui, |ui| {
                self.draw_right_inspector(ui);
            });

        egui::CentralPanel::default().show(ui, |ui| {
            self.draw_hsqc_canvas(ui);
        });
    }
}

fn main() -> eframe::Result {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("nmr-assign — WebGPU Protein NMR Spectrum & Assignment Workbench")
            .with_inner_size([1380.0, 860.0])
            .with_min_inner_size([960.0, 620.0]),
        ..Default::default()
    };
    eframe::run_native(
        "nmr-assign",
        native_options,
        Box::new(|cc| Ok(Box::new(NmrWorkbenchApp::new(cc)))),
    )
}
