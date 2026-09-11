Paper 6 - 05h3c aggregation-input technical preflight

Status
------
PASS_READY_FOR_05H3D_AGGREGATION_SENSITIVITY

This stage computes no core-only or regime-balanced scientific estimate.

Resolved identities
-------------------
Scenario table:
results\simulation_phase_diagram\05d\scenario_model_metric_summary.tsv

CORE/STRESS:
Resolved only from literal existing scenario_id prefixes CORE_ and STRESS_.
No regex and no interpretation of the descriptive `family` column.
Counts: 36 CORE, 144 STRESS.

Descriptive family labels:
{
  "CORE_PHASE_DIAGRAM": 36,
  "MAPPING_PRIOR_STRESS": 72,
  "SHIFT_CENSOR_STRESS": 72
}

Regimes:
Resolved from the prefix before the first underscore in transfer_regime.
Counts: {'R0': 6, 'R1': 6, 'R2': 78, 'R3': 6, 'R4': 6, 'R5': 78}

Retained replicate arrays:
180 NPZ files, 21,600 frozen replicates total.
Required keys: replicate_seed, delta_c_vs_B0, negative_transfer, catastrophic_negative_transfer

Validated model axis:
B0, B4, A0, A1, A2, A3, A4, A3_NO_EVOLUTION_PRIOR, A3_NO_TARGET_OVERRIDE, A3_PERMUTED_PRIOR, A3_MAPPING_PERMUTATION, A3_SOURCE_OUTCOME_PERMUTATION
A2 index: 4
A3 index: 5

Threshold identity:
negative_transfer == (delta_c_vs_B0 <= -0.02): exact
catastrophic_negative_transfer == (delta_c_vs_B0 <= -0.05): exact

A2/A3 retained-array replay:
all 180 scenarios PASS.
max mean-delta difference = 1.572e-08
max NT-rate difference = 0.000e+00
max catastrophic-rate difference = 0.000e+00

Next
----
A new 05h3d computation script should read this contract and refuse to run if
any source hash, column binding, scenario membership, regime mapping, NPZ count,
model-axis identity, or frozen threshold identity changes.
