# Paper 6 - Cross-Species Transfer Learning for Osteosarcoma Survival

**Working manuscript title:**  
**When Cross-Species Compatibility Does Not Guarantee Transfer Utility in Event-Limited Survival Modeling**

**Status:** manuscript in preparation / internal review.

This repository contains the analysis code, frozen contracts, audit artifacts, and reproducibility workflow for Paper 6, a cross-species survival-transfer study using canine osteosarcoma as a source domain and human osteosarcoma as the target domain.

The project is designed around one central question:

> When target outcome information is scarce, does measurable cross-species molecular compatibility translate into useful and sufficiently safe prognostic transfer?

The study separates **architecture selection**, **post-result diagnostics**, **controlled mechanism experiments**, **outcome-blind biological characterization**, and **descriptive human evaluation** into chronologically distinct evidence layers.

---

## 1. Scientific overview

Rare-disease survival modeling creates a difficult transfer-learning setting: the target cohort may contain too few events to safely fit, tune, select, and evaluate a flexible transfer model using the same outcome data.

Paper 6 therefore uses:

- canine osteosarcoma as the molecular source domain;
- human osteosarcoma as the target domain;
- a frozen 50-Hallmark representation;
- a known-truth simulation benchmark as the primary environment for architecture-level safety evaluation;
- a separate new-seed mechanism experiment;
- an outcome-blind DOG²–TARGET biological-context analysis;
- a frozen descriptive TARGET-OS evaluation.

The main methodological emphasis is **negative-transfer protection and evidence chronology**, not retrospective selection of the best-performing human model.

---

## 2. Data sources

### Canine source

- **DOG² / GSE238110**
- 186 canine osteosarcoma expression samples
- source survival endpoint used only where authorized by the corresponding frozen analysis stage

### Human osteosarcoma

- **TARGET Osteosarcoma**
- 88 expression samples in the outcome-blind expression roster
- 86 complete primary overall-survival cases in the frozen TARGET evaluation
- 29 observed OS events and 57 censored observations
- one censored zero-time case retained unchanged

### Earlier premise cohort

- **GSE16091**
- used only in the earlier premise stage
- not used to tune the later simulation grid or select the final transfer architecture

### Reserved outcome cohorts

The outcome variables for the following cohorts remain outside the Paper 6 reported evaluation:

- **GSE21257**
- **GSE39055**

Their outcomes were not opened to explain, rescue, or reinterpret the TARGET results.

> Raw source datasets are not redistributed by this repository where source terms require users to obtain them directly from the original repository.

---

## 3. Cross-species representation

The primary dog-to-human bridge contains:

- **11,815 one-to-one aligned dog–human gene pairs**
- **50 MSigDB Hallmark gene sets**

Predictive preprocessing is partition-local.

For TARGET predictive evaluation:

1. gene-level means and population standard deviations are estimated on the current training partition only;
2. the fitted transform is applied unchanged to held-out patients;
3. Hallmark scores are constructed from the training-standardized genes;
4. Hallmark-level scaling is fitted on the same training partition;
5. no whole-TARGET or held-out-patient statistic enters predictive preprocessing.

Source and target domains are standardized independently.

---

## 4. Evidence architecture

The project intentionally distinguishes analysis stages by when their rules were fixed and what information was available.

### Layer 1 — Frozen known-truth benchmark

Primary architecture-selection environment.

- 180 simulation scenarios
- 21,600 replicates
- target event budgets: 5, 10, 15, 20, 29, 40
- six source–target transport regimes
- additional covariance-shift, censoring, mapping-error, and prior-quality stress axes
- equal-scenario weighting for the frozen selection summaries

Only A2 and A3 were eligible for final architecture selection.

Frozen negative-transfer definitions:

- negative transfer: `ΔC <= -0.02`
- catastrophic negative transfer: `ΔC <= -0.05`

Frozen protection requirements included:

- aggregate negative-transfer rate `<= 0.10`
- catastrophic-transfer rate `<= 0.05` in designated stress strata
- additional module-recovery requirements for A3

Frozen result:

`HOLD_NO_SAFE_SELECTABLE_ARCHITECTURE`

This decision is immutable for Paper 6.

---

### Layer 2 — Bounded post-HOLD diagnostics

These analyses were permitted to explain the frozen result but not to change it.

Examples include:

- implementation and metric replay;
- threshold-attainability oracles;
- compatibility-versus-utility diagnostics;
- comparator decomposition;
- event-count abstention;
- A6 feasibility testing.

The proposed A6 trust/abstention branch was closed under its own frozen opening rule.

---

### Layer 3 — Controlled new-seed mechanism experiment

A separate experiment used:

- 12 cells
- 3 event budgets: 10, 29, 40
- 4 transport regimes: R0, R2, R4, R5
- 500 replicates per cell
- **6,000 new seeds**
- zero overlap with the original 21,600 benchmark replicates

Seven interventions were evaluated (M0–M6), with four H1–H4 mechanism contrasts frozen before fitting the new replicates.

Later bounded diagnostics included:

- sign-inversion equivalence;
- discrete regime-shape analysis;
- fixed-residual truth-gate substitution (`M6'`).

These later diagnostics cannot rewrite H1–H4.

---

### Layer 4 — Secondary outcome-blind biological context

This stage was frozen after TARGET outcome opening but before completed TARGET model results were inspected.

Evidence status:

**SECONDARY / POST-TARGET-OPENING / PRE-TARGET-RESULT-INSPECTION / OUTCOME-BLIND**

The analysis uses:

- all 186 DOG² expression samples;
- all 88 TARGET expression samples;
- all 50 frozen Hallmark gene sets;
- four independently frozen molecular-program anchors: M34, M40, M11, M24;
- 1,000 exact-size matched random panels per tested set.

No survival outcome is used in this analysis.

Primary structural statistic:

- Pearson gene–gene correlation matrices within each cohort;
- identical within-set edge vectors;
- DOG²–TARGET Spearman edge concordance.

Matched controls are exact on a four-dimensional joint tertile structure based on expression level and variance in both cohorts.

Matched-control percentiles are descriptive and are **not p-values**.

---

### Layer 5 — Descriptive TARGET-OS evaluation

TARGET was deliberately assigned a descriptive, non-confirmatory role.

Frozen evaluation:

- 86 cases
- 29 OS events
- 9 models: T0–T2 and N0–N5
- 5-fold outer CV × 20 repeats
- event-stratified folds
- 5,000 valid patient-clustered bootstrap draws
- no model selection
- no post-opening hyperparameter search
- no reconstruction of real-data A3 / hardened A3

Final frozen TARGET interpretation:

`T-D`

The point-estimate pattern initially satisfied T-A, but the frozen uncertainty rules required downgrade to T-D because the source-orientation interval included 0.50 and the branch-defining paired intervals spanned the frozen neutral region.

TARGET therefore remained descriptively unresolved and did not alter the simulation HOLD.

---

## 5. Main findings

### Frozen architecture selection

Neither selectable architecture satisfied the frozen negative-transfer rule.

- A2 aggregate negative-transfer rate: ~0.215
- A3 aggregate negative-transfer rate: ~0.486
- frozen maximum: 0.10

A3 met its module-recovery requirement but still failed the safety rule.

Post-HOLD oracle analyses showed that the 0.10 threshold was attainable within the same known-truth environment, so the HOLD was not caused by an impossible criterion.

### Compatibility is not the same as transfer utility

At 29 target events, the compatibility statistic strongly identified the predefined mechanistic regimes but performed below chance for predicting independent-test transfer benefit versus harm.

The marginal compatibility–utility relationship was strongly affected by regime composition.

### Partial transportability was a distinct failure region

In the controlled new-seed experiment, head retargeting had its lowest transfer utility in the predefined partially transportable regime rather than in the fully nontransportable regime.

This is treated as a discrete generator-category result, not as a continuous biological U-shaped relationship.

### Global sign reversal was largely an orientation problem

In misleading-source R5 at 29 events, the retargeted head was effectively equivalent to reversing the frozen source score.

This bounded audit was classified as:

`SIGN_INVERSION_EQUIVALENT`

### Prediction-time gate hardening improved observed transfer behavior

Hardening the learned soft gate at prediction time:

- required no refitting;
- improved Uno-C;
- sharply reduced empirical negative transfer;
- increased risk-score dispersion.

Its original composite H3 machine status remains unchanged because one frozen rate-reduction requirement was structurally unattainable at the realized control rate.

### Biological preservation was heterogeneous but non-random in structure

Across the 50 Hallmarks:

- median DOG²–TARGET Spearman edge concordance: ~0.319
- median matched-control percentile: ~0.783
- strong heterogeneity across programs

The independently frozen strong-preservation anchors M34 and M40 occupied the extreme upper tail of their matched-control distributions, while M11 and M24 did not.

This analysis characterizes the cross-species biological setting; it is not treated as independent proof that structural concordance predicts prognostic transfer utility.

### TARGET remained unresolved

The largest TARGET Uno-C point estimates were observed for the most flexible neural procedures (N4 and N5), but the frozen branch-defining uncertainty remained wide.

The final state was T-D, and no TARGET result changed the frozen simulation decision.

---

## 6. Repository principles

This project uses versioned analysis contracts rather than mutable exploratory scripts.

Important rules:

1. **Do not edit a frozen artifact in place.**
2. If a technical correction is necessary, create a new versioned amendment.
3. Preserve the earlier artifact and its failure/provenance record.
4. Verify expected SHA-256 hashes before downstream execution.
5. Do not reopen sealed outcomes from reserved cohorts.
6. Do not reinterpret post-HOLD diagnostics as part of the original architecture-selection test.
7. Do not upgrade descriptive TARGET or biological results into confirmatory evidence.
8. Do not reconstruct real-data A3 or introduce a new transfer branch using opened TARGET outcomes.

The repository intentionally preserves failed technical attempts when they are relevant to the audit trail.

---

## 7. Key execution stages

The repository contains many versioned scripts. The exact dependency order is encoded by the frozen contracts and artifact hashes.

Important late-stage scripts include:

```text
scripts/
  05f3f_run_exact_target86_semantic_f3c_lock.py
  05g0_freeze_postopening_preresult_outcomeblind_biology_contract.py
  05g1_execute_frozen_outcomeblind_biology_context.py
```

### TARGET execution

```powershell
python scripts\05f3f_run_exact_target86_semantic_f3c_lock.py
```

Successful terminal state:

```text
PASS_TARGET86_POSTOPENING_AUDITED_FROZEN_DESCRIPTIVE_EVALUATION_COMPLETE
```

### Biological-context execution

`05g0` is the frozen contract and must not be modified or rerun to change the analysis specification.

Execute the already-frozen biological analysis with:

```powershell
python scripts\05g1_execute_frozen_outcomeblind_biology_context.py
```

Successful terminal state:

```text
PASS_FROZEN_SECONDARY_POSTOPENING_PRE_05F3F_RESULT_OUTCOMEBLIND_BIOLOGY_EXECUTION
```

---

## 8. Important output locations

Representative final outputs:

```text
results/
  human_posthold_descriptive/
    05f3f/
      ...
    05g1/
      hallmark50_structural_landscape.tsv
      paper4_anchor_structural_context.tsv
      matched_random_control_summary.tsv
      matched_random_control_draws.tsv
      matched_random_panel_membership.npz
      figure5_biological_context_data.tsv
      summary.json
```

> Some historical artifact filenames retain internal development labels such as `paper4`. These filenames are provenance identifiers only and are not manuscript terminology.

Simulation and mechanism outputs are stored in their corresponding frozen stage directories with manifests and versioned summaries.

---

## 9. Reproducibility notes

The analysis uses Python with packages including:

- NumPy
- pandas
- PyTorch
- scikit-learn
- scikit-survival

Exact package and platform versions for final analyses should be taken from the repository's frozen environment / software ledger rather than inferred from this README.

Execution devices differ by stage:

- large simulation/model-fitting stages may use GPU where defined by the frozen implementation;
- aggregation and many diagnostics use CPU;
- the final TARGET execution uses the frozen audited CPU path;
- the outcome-blind biological-context analysis uses CPU/BLAS plus multiprocessing.

Execution device is not a model-selection variable.

---

## 10. Figures and manuscript-facing outputs

The main manuscript currently uses five figures:

1. analysis chronology and outcome firewall;
2. frozen HOLD and threshold attainability;
3. compatibility, transfer utility, and discrete regime structure;
4. controlled mechanism and prediction-time gating analyses;
5. outcome-blind biological structural preservation.

The manuscript also reports:

- the frozen model-family map;
- controlled R5@29 mechanism results;
- artifact-label versus scientific-reading table;
- descriptive TARGET model results and paired contrasts.

---

## 11. Data and code availability

Public datasets should be downloaded from their original repositories.

Where redistribution is allowed, the released repository may include derived non-identifying intermediate artifacts required for exact reproduction. Where redistribution is restricted, users should obtain the original data directly and reproduce the downstream artifacts using the committed scripts and frozen contracts.

The public release is intended to include:

- versioned analysis scripts;
- frozen contracts;
- model and metric registries;
- random-seed manifests;
- derived reproducibility fixtures;
- machine-readable result summaries;
- artifact SHA-256 manifests;
- manuscript-facing tables and figure data where permitted.

---


## 12. Citation

The manuscript is currently under internal review. Citation metadata will be added after a stable public manuscript or preprint record is available.

Until then, please refer to the project by its working title:

> *When Cross-Species Compatibility Does Not Guarantee Transfer Utility in Event-Limited Survival Modeling*

---

## 14. Contact

For questions about the analysis or reproducibility workflow, please contact the corresponding authors listed in the manuscript.

---

## 15. Scope of this repository

This repository is intended to reproduce and audit **Paper 6 only**.

Some upstream artifacts originate from earlier independently frozen molecular-preservation work and are imported only where explicitly bound by the Paper 6 contracts. Their inclusion does not make those earlier studies part of the Paper 6 architecture-selection evidence.

The central rule for reading this repository is therefore:

> **Later evidence may explain an earlier frozen decision, but it may not rewrite it.**
