# PACT: Constitutive Transfer Through Persistent Material Realizations

This repository contains the non-manuscript research and reproducibility material associated with the PACT paper, **“PACT: Constitutive Transfer Through Persistent Material Realizations.”**

## Scope

The repository is intentionally **paper-free**: it does not contain the manuscript PDF, LaTeX manuscript source, or journal/arXiv submission files. It contains the self-contained matched-information elastoplastic transfer benchmark used to test constitutive transfer through persistent material realizations, together with the protocol, numerical implementation, verification records, result summaries, sensitivity sweep, and table/figure-generation code.

The benchmark is a designed computational mechanics test, not an external experimental validation dataset. Two compression-only elastoplastic elements are observation-equivalent in the source setting but carry different persistent internal states; an asymmetric target exposes the consequences of discarding those states. All compared methods receive identical information.

## Reproduce the benchmark

With Python 3, NumPy, SciPy, pandas and Matplotlib installed:

```bash
python3 benchmark/run_benchmark.py
python3 benchmark/build_tables.py
python3 benchmark/plot_results.py
```

`benchmark/environment.json` records the versions used for the archived run. `benchmark/protocol.json` records the fixed inputs, seed, methods, scores, noise levels, and sensitivity grid.

The benchmark scripts regenerate the detailed virtual-record CSV, trajectory archive, tables and publication figures. The repository keeps the compact result summaries and robustness sweep under version control; larger or redundant generated artefacts are reproducible from the declared protocol and source.

## Repository contents

- `benchmark/protocol.json` — benchmark design and scoring protocol.
- `benchmark/run_benchmark.py` — numerical mechanics, exact reference, inference, scoring and verification checks.
- `benchmark/build_tables.py` — LaTeX tables generated from archived summary results.
- `benchmark/plot_results.py` — publication figure generator.
- `benchmark/environment.json` — numerical software versions used for the archived run.
- `benchmark/results/summary.json` — headline numerical results and verification checks.
- `benchmark/results/robustness_sweep.csv` — complete 27-configuration robustness sweep for all methods.
- `benchmark/results/table_initial.tex` and `table_conditioned.tex` — generated numerical tables.
- `BENCHMARK_PROVENANCE.json` — provenance and verification record.

## Manuscript exclusion

No paper or manuscript source is stored here, by design. The arXiv/journal submission package is maintained separately.
