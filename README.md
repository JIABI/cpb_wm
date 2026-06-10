````markdown
# Certified Planning Bandwidths for Neural World-Model Agents

This repository contains the anonymized implementation and source tables for the paper:

**Empirically Certified Planning Bandwidths for Neural World-Model Agents**

The code implements **Certified Planning Bandwidths (CPB)**: a deployment-time calibration layer for fixed world models, fixed policies, and declared downstream interfaces. CPB estimates an interface-risk curve over planning stochasticities, constructs a conservative certified band using simultaneous one-sided upper-confidence bounds, and deploys by projecting a noise-matched stochasticity response into the certified band.

Here, **certified** means finite-sample inclusion on a calibration distribution under a fixed interface. It is not a worst-case runtime safety guarantee and not a zero-violation shield.

---

## What is included

The anonymized archive contains:

- CPB calibration and projection code.
- Common-random-numbers (CRN) paired rollout wrappers.
- Interface encoders for action-region, value-threshold, safety-event proxy, and Atari action/event interfaces.
- Empirical-Bernstein and Hoeffding upper-confidence certification routines.
- Controlled finite-state mechanism study.
- DreamerV3 / RSSM calibration and deployment scripts.
- DIAMOND and IRIS visual-world-model diagnostic replication scripts.
- Ablation and failure-mode scripts.
- Source tables for all main-text and supplementary figures/tables.
- Pair-seed lists, calibration grids, OOD perturbation grids, bootstrap diagnostic scripts, and plotting scripts.

The repository is anonymized for double-blind review. Non-anonymized public artifacts will be released after acceptance.

---

## Repository layout

```text
.
├── src/
│   └── cpb/
│       ├── calibration/        # certified-band construction
│       ├── confidence/         # Hoeffding and empirical-Bernstein bounds
│       ├── interfaces/         # downstream interface encoders
│       ├── projection/         # projection / rejection rules
│       ├── rollouts/           # imagined-realized CRN rollout wrappers
│       └── metrics/            # interface risk, recovery, OOD diagnostics
├── experiments/
│   ├── exp1_controlled/        # finite-state mechanism study
│   ├── exp2_dreamerv3/         # DreamerV3 / DM Control experiments
│   ├── exp3_visual/            # DIAMOND and IRIS diagnostic replications
│   └── exp4_ablations/         # ablations and failure modes
├── results/
│   ├── source_tables/          # seed-level CSV/Parquet source tables
│   ├── figures/                # generated PDF figures
│   └── runs/                   # experiment outputs
├── configs/                    # experiment configs and fixed grids
├── scripts/                    # plotting and table-generation scripts
├── tests/                      # unit and smoke tests
└── README.md
````

If your local unpacked archive uses slightly different module paths, use the corresponding script names in `experiments/` and `scripts/`. The source tables are the authoritative record for the paper’s numerical results.

---

## Installation

Create a clean Python environment first.

```bash
conda create -n cpb python=3.10 -y
conda activate cpb
```

Install the repository in editable mode:

```bash
pip install -e ".[dev]"
```

For the full experiment stack:

```bash
pip install -e ".[dev,control,visual]"
```

Optional extras are grouped as follows:

| Extra        | Purpose                                    |
| ------------ | ------------------------------------------ |
| `.[dev]`     | pytest, ruff, mypy, pre-commit, jupyterlab |
| `.[control]` | DM Control / Gymnasium wrappers            |
| `.[visual]`  | DIAMOND / IRIS diagnostic dependencies     |
| `.[plots]`   | plotting and table-generation utilities    |

GPU-enabled visual-world-model runs may require additional framework-specific installation steps. See `configs/visual/README.md` if included in the archive.

---

## Quick smoke test

Run the unit tests and a small calibration smoke test:

```bash
pytest -q
```

```bash
python -m experiments.exp1_controlled.run_bandwidth \
  --noise-levels 0.05 0.10 0.20 \
  --sigma-min 0.0 \
  --sigma-max 0.8 \
  --sigma-n 11 \
  --horizon 16 \
  --episodes 200 \
  --seeds 3 \
  --alpha 0.10 \
  --out results/runs/exp1_controlled/smoke.csv
```

Generate a smoke plot:

```bash
python -m experiments.exp1_controlled.plot_bandwidth \
  --csv results/runs/exp1_controlled/smoke.csv \
  --out results/figures/exp1_smoke.pdf
```

---

## Reproducing the main paper

### Experiment 1: controlled finite-state mechanism study

This experiment verifies the lower-endpoint, upper-endpoint, and projection mechanism in a finite-state process where the reference risk can be computed exactly.

```bash
python -m experiments.exp1_controlled.run_bandwidth \
  --noise-levels 0.00 0.05 0.10 0.20 0.30 \
  --sigma-min 0.0 \
  --sigma-max 1.0 \
  --sigma-n 41 \
  --horizon 16 \
  --episodes 2000 \
  --seeds 10 \
  --alpha 0.10 \
  --confidence hoeffding \
  --out results/runs/exp1_controlled/bandwidth.csv
```

```bash
python -m experiments.exp1_controlled.plot_main \
  --csv results/runs/exp1_controlled/bandwidth.csv \
  --out results/figures/fig2_controlled_mechanism.pdf
```

Expected output:

* Controlled risk curves.
* Conservative certified endpoints.
* Projection of the analytic response into the certified band.
* Source table for the controlled endpoint/projection table.

---

### Experiment 2: DreamerV3 / DM Control CPB deployment

This experiment evaluates CPB in a learned RSSM world-model agent on stochastic DM Control tasks.

The main settings are:

* Horizon: `H = 5`
* Tolerance: `alpha = 0.30`
* Ensemble size: `K = 3`
* Primary interface: `k = 8` action-cluster interface
* Certification rule: empirical-Bernstein simultaneous one-sided upper confidence
* Pairing protocol: CRN-paired imagined-realized rollouts
* Paired rollouts: `N = 200` per sigma-grid point within each uncertainty bin

Run calibration:

```bash
python -m experiments.exp2_dreamerv3.calibrate_cpb \
  --config configs/exp2_dreamerv3/main.yaml \
  --confidence empirical_bernstein \
  --out results/runs/exp2_dreamerv3/calibration/
```

Run deployment evaluation:

```bash
python -m experiments.exp2_dreamerv3.evaluate_deployment \
  --config configs/exp2_dreamerv3/main.yaml \
  --calibration results/runs/exp2_dreamerv3/calibration/ \
  --out results/runs/exp2_dreamerv3/deployment/
```

Generate figures and tables:

```bash
python -m scripts.make_fig3_dreamerv3 \
  --source results/source_tables/exp2_dreamerv3.csv \
  --out results/figures/fig3_dreamerv3.pdf
```

```bash
python -m scripts.make_table3_deployment \
  --source results/source_tables/exp2_deployment.csv \
  --out results/source_tables/table3_deployment.tex
```

Expected outputs:

* EB-certified risk curves and endpoints.
* Return–violation–recovery trade-off.
* Mid-episode noise-spike recovery curves.
* Table III deployment summary.
* Paired bootstrap diagnostic intervals for method differences.

Bootstrap resampling is used only for method-difference uncertainty reporting. It is **not** used to construct the certified set.

---

### OOD bandwidth collapse and recalibration-time warning

This experiment measures when the certified band becomes empty under OOD shifts. The warning is a **post-recalibration diagnostic** conditional on a recalibration pass being run. It is not an online action-selection detector.

```bash
python -m experiments.exp2_dreamerv3.run_ood_recalibration \
  --config configs/exp2_dreamerv3/ood.yaml \
  --calibration results/runs/exp2_dreamerv3/calibration/ \
  --confidence empirical_bernstein \
  --out results/runs/exp2_dreamerv3/ood/
```

```bash
python -m scripts.make_fig4_ood \
  --source results/source_tables/exp2_ood.csv \
  --out results/figures/fig4_ood_warning.pdf
```

Main OOD families:

* mass–friction shifts
* observation corruption
* action delay

The reported lead time is measured from the first recalibration pass at which the EB-certified band is empty to the first conventional detector threshold crossing. Recalibration latency is not included in the reported lead time.

---

### Experiment 3: DIAMOND / IRIS visual diagnostic replications

This experiment checks whether the CPB calibration protocol can be instantiated beyond RSSM world models. It is a compact diagnostic replication, not a full Atari benchmark.

```bash
python -m experiments.exp3_visual.calibrate_visual_cpb \
  --config configs/exp3_visual/main.yaml \
  --confidence empirical_bernstein \
  --out results/runs/exp3_visual/calibration/
```

```bash
python -m experiments.exp3_visual.evaluate_visual \
  --config configs/exp3_visual/main.yaml \
  --calibration results/runs/exp3_visual/calibration/ \
  --out results/runs/exp3_visual/deployment/
```

```bash
python -m scripts.make_fig5_visual \
  --source results/source_tables/exp3_visual.csv \
  --out results/figures/fig5_visual.pdf
```

Expected outputs:

* DIAMOND EB-certified diagnostic calibration curves.
* IRIS EB-certified diagnostic calibration curves.
* Empty-band rejection cases.
* Architecture-level diagnostic summary.
* Table IV visual model diagnostic summary.
* Table S6 per-game visual outcomes.

The CPB protocol is unchanged across architectures. Only the model-specific uncertainty estimator and rollout-sampling stochasticity differ.

---

### Experiment 4: ablations and failure modes

This experiment separates response matching, certification, and projection.

```bash
python -m experiments.exp4_ablations.run_ablations \
  --config configs/exp4_ablations/main.yaml \
  --confidence empirical_bernstein \
  --out results/runs/exp4_ablations/
```

```bash
python -m scripts.make_fig6_ablations \
  --source results/source_tables/exp4_ablations.csv \
  --out results/figures/fig6_ablations.pdf
```

Expected outputs:

* Match-only / cap-only / CPB-full component ablation.
* Response-scale misspecification sweep.
* Biased uncertainty-estimator stress test.
* Interface robustness across action-region, value-threshold, and safety-event proxy interfaces.
* Empty-band and disconnected-band diagnostics.
* Cheetah-run upper-endpoint saturation diagnostic.
* Table V method roles.
* Table S7 ablation summary.
* Table S8 safety-oriented comparator summary.

---

## Certification rules

CPB constructs the empirical conservative certified band

```math
\widehat{\mathcal B}^{\mathrm{WM}}_{H,\alpha}(z)
=
\left\{
\sigma\in\Sigma_z:
\widehat R_H^{+,\mathrm{WM}}(\sigma;z)\le \alpha
\right\}.
```

Two one-sided upper-confidence rules are implemented:

1. **Hoeffding bound**
   Used in the controlled mechanism study and in the proof-aligned construction.

2. **Empirical-Bernstein bound**
   Used in the main DreamerV3, OOD, visual, and ablation experiments.

The empirical-Bernstein rule is justified by its own simultaneous one-sided inclusion lemma. It is not justified by comparing its radius to the Hoeffding radius.

The endpoint-recovery theorem in the paper is proved for the Hoeffding construction under a connected-band endpoint-margin condition. The implemented empirical-Bernstein band has the same conservative inclusion role but does not inherit the Hoeffding endpoint-rate theorem.

---

## Common-random-numbers paired rollouts

CPB estimates risk using paired imagined-realized rollouts that share:

* the same recorded initial observation,
* the same planner-noise sequence,
* and, where applicable, the same wrapper-level environment seed.

The primary risk is the mean-per-step interface disagreement

```math
R_H^{\mathrm{WM}}(\sigma;z)
=
\mathbb E_{(s_0,\xi)\sim\mathcal D_z}
\left[
\frac{1}{H}
\sum_{t=1}^{H}
\mathbf 1\{I(\tau_t^{\mathrm{imag}})\neq I(\tau_t^{\mathrm{real}})\}
\right].
```

The repository includes a real-real CRN sanity check. The expected output is zero interface disagreement between two realized rollouts sharing the same recorded initial state and planner-noise sequence.

```bash
python -m experiments.exp2_dreamerv3.check_crn_real_real \
  --config configs/exp2_dreamerv3/main.yaml \
  --out results/runs/crn_real_real/
```

---

## Interfaces

CPB certifies relative to a declared downstream interface. The main interfaces are:

| Setting                       | Interface                                                   |
| ----------------------------- | ----------------------------------------------------------- |
| DM Control primary            | `k = 8` action-cluster interface                            |
| DM Control robustness         | `k = 16`, `k = 32` action-cluster interfaces                |
| DM Control value-threshold    | above/below critic-value quantile                           |
| DM Control safety-event proxy | binary boundary-event proxy; not a formal safety constraint |
| Atari visual                  | action label                                                |
| Atari event                   | score-change indicator                                      |

Safety-event proxy interfaces are used only as downstream binary decision labels for interface-risk calibration. They are not formal task-level safety guarantees.

---

## Source tables

The source tables are provided under:

```text
results/source_tables/
```

They include seed-level data for:

* Table III: main DM Control deployment comparison.
* Fig. 3: DreamerV3 calibration and deployment curves.
* Fig. 4: OOD bandwidth collapse and detector lead time.
* Table IV: visual model diagnostic summary.
* Fig. 5: DIAMOND / IRIS visual diagnostic curves.
* Table V and Fig. 6: ablations and failure modes.
* Supplementary Tables S1–S9.

The source tables are the preferred way to verify all reported means, standard deviations, paired bootstrap intervals, collapse thresholds, and empty-band rates.

---

## Regenerating all paper figures

```bash
python -m scripts.make_all_figures \
  --source-dir results/source_tables \
  --out-dir results/figures
```

Regenerating all LaTeX tables:

```bash
python -m scripts.make_all_tables \
  --source-dir results/source_tables \
  --out-dir results/generated_tables
```

---

## Runtime and compute budget

The full DM Control offline calibration uses approximately:

```text
N = 200 paired rollouts
× up to 21 sigma-grid points
× 3 uncertainty bins
× 5 environments
× 5 calibration seeds
≈ 3 × 10^5 paired imagined-realized rollouts.
```

On A100/H100-class hardware, this takes approximately 2–3 wall-clock hours per environment-seed calibration cell with a `K = 3` RSSM ensemble.

Online deployment is much cheaper. Algorithm 2 requires:

* one uncertainty estimate,
* lookup of the cached certified band,
* and projection into the band.

The projection cost is `O(|Σ_z|)` with a simple scan, or `O(log |Σ_z|)` when the certified band is stored as sorted connected components.

---

## Quality checks

Format and lint:

```bash
ruff format src tests experiments scripts
ruff check src tests experiments scripts
```

Type check core modules:

```bash
mypy --strict src/cpb
```

Run tests:

```bash
pytest -q
```

Run only smoke tests:

```bash
pytest -q tests/smoke
```

---

## Troubleshooting

### The certified band is empty

An empty band means that no tested stochasticity satisfies the certified interface-risk tolerance on the current calibration cell. CPB then returns `NOT-CERTIFIABLE`. The default evaluation fallback uses the deterministic actor mode only to keep the episode running. No admissibility claim is made for fallback timesteps.

### The OOD lead time looks shorter than the recalibration period

The reported OOD lead time is conditional on a recalibration pass having been run at that OOD severity. It is a post-recalibration diagnostic lead, not a guarantee that a periodic monitor will fire within that many online environment steps.

### The EB band differs from the Hoeffding band

This is expected. Hoeffding is used for the proof-aligned construction and controlled study. Empirical-Bernstein is used in the main implementation because it is a valid variance-adaptive simultaneous upper-confidence rule and is often empirically tighter.

### Bootstrap intervals do not match the certified set

Bootstrap intervals are used only for diagnostic method-difference uncertainty, such as paired differences in NDV, violation, and recovery. Bootstrap quantiles are not used to construct the certified band.

---

## Citation

During double-anonymous review, please cite the anonymized manuscript.

A public BibTeX entry will be added after acceptance.

```
```

