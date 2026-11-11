# 06. WebGPU compute and interactive GUI (`nmr-gpu`, `nmr-gui`)

Replaces the earlier two-language Python + planned Julia design with a single
pure-Rust + WebGPU (`wgpu` / WGSL) stack that runs natively on Metal, Vulkan,
DX12, and browser WebGPU.

## 1. WebGPU compute (`crates/nmr-gpu`)
- **`scoring.wgsl`**: Computes `unary_matrix` `U(s, k)` and `pair_matrix` `P(s, t)` in parallel `(16 x 16)` workgroups. Parity-tested in `crates/nmr-gpu/tests/gpu_parity_tests.rs` against `nmr-core` (`f64`) within `< 1e-3` (`f32` precision at 176 ppm carbonyl shifts).
- **`anneal.wgsl`**: Runs tens to hundreds of independent Metropolis-Hastings simulated annealing replicas in parallel (`--engine gpu`). Per-replica `at` and `pos` arrays reside in on-chip `var<workgroup>` SRAM (`16 KB` per workgroup, within WebGPU baseline limits) to avoid global DRAM traffic in the inner loop; exact `f64` scores are re-evaluated on readback.
- **Shared device**: `GpuEngine::from_device` shares `eframe`'s active `wgpu::Device` and `wgpu::Queue` in the GUI; `GpuEngine::new_headless` runs standalone in `nmr-cli`.

## 2. Interactive workbench (`crates/nmr-gui`)
Built on `eframe` / `egui` + `egui_wgpu::CallbackTrait`:
- **2D GPU contour shader (`contour_shader.wgsl`)**: Bilinear interpolation + screen-space gradient anti-aliased multi-level positive (cyan) and negative (coral) contours at 60–120 fps, with interactive pan, cursor-anchored zoom, and spin-system selection.
- **Sequence & confidence ribbon**: Clickable per-residue tiles colored by consensus confidence (`>= 0.8` high, `0.5–0.8` medium, `< 0.5` low, unassigned, Pro/N-term).
- **3D sequential walk strip viewer**: Plots `i-1`, `i`, and `i+1` `HNCACB` / `CBCA(CO)NH` `13C` strips with slice intensity shading, `+CA` / `-CB` / `i-1` markers, and dashed horizontal connectivity tie-lines.
- **Direct spectrum & assignment workflow**: Load `.str`, `.tsv`, Bruker `pdata` (`2rr`), or NMRPipe (`.ft2`), run classical peak picking or WebGPU assignment live, and export `.str` + `.tsv`.
