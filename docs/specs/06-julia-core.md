# 06. Julia core (planned)

Port order, each gated on parity with the Python reference (same U and P
matrices to 1e-9, comparable benchmark accuracy):
1. Priors, `unary_matrix`, `pair_matrix`.
2. Annealer (hot loop; expected 50x to 100x speed-up, which converts into
   recall on noisy data).
3. Spin system builder; NMRTools.jl (MIT) for spectra.
4. Python wrapper via juliacall (MIT): `cli --engine julia`.

CI runs the benchmark with both engines.
