# Third-party dependencies

| Dependency | Licence | Used for |
|---|---|---|
| `serde`, `serde_json` | MIT OR Apache-2.0 | JSON prior table serialization (`nmr-core`) |
| `wgpu` | MIT OR Apache-2.0 | Cross-platform WebGPU / Metal / Vulkan / DX12 compute & rendering (`nmr-gpu`, `nmr-gui`) |
| `eframe` (`egui`, `egui-wgpu`) | MIT OR Apache-2.0 | Interactive scientific GUI (`nmr-gui`) |
| `bytemuck` | Zlib OR Apache-2.0 OR MIT | Safe POD GPU buffer casting (`nmr-gpu`, `nmr-gui`) |
| `pollster` | MIT OR Apache-2.0 | Minimal async executor for headless `wgpu` initialization (`nmr-gpu`) |

Transitive crates pulled by `wgpu` and `eframe` use `MIT`, `Apache-2.0`, `BSD-2-Clause`, `BSD-3-Clause`, `ISC`, or `Zlib` (with `BSL-1.0` only on Windows clipboard bindings when enabled). Zero GPL, LGPL, AGPL, or MPL crates are used.

Data: BMRB entries used to build priors or fixtures must be cited by BMRB ID.
