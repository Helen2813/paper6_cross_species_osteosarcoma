Paper 6 - 05h2b frozen all-model safety summary

Analysis role
-------------
POST-HOLD DESCRIPTIVE CONTEXT.

Scientific source
-----------------
The exporter is explicitly bound to the authoritative frozen 05d table:
results/simulation_phase_diagram/05d/scenario_model_metric_summary.tsv

No diagnostic table is used as the scientific source.

Existing scenario-level fields used directly
--------------------------------------------
mean_uno_c
mean_delta_c_vs_B0
negative_transfer_rate
catastrophic_negative_transfer_rate

The catastrophic quantity is therefore NOT reconstructed from a scenario flag.

Frozen model family
-------------------
B0, B4, A0, A1, A2, A3, A4.
B0 is the reference.
Only A2 and A3 remain eligible for the frozen primary architecture-selection rule.
This descriptive export cannot make another model selectable and cannot reopen
HOLD_NO_SAFE_SELECTABLE_ARCHITECTURE.

Frozen safety definitions
-------------------------
Negative transfer: delta Uno-C <= -0.02.
Catastrophic negative transfer: delta Uno-C <= -0.05.

Aggregation
-----------
The source table already contains within-scenario means/rates. 05h2b assigns equal
weight to each of the 180 frozen scenarios. Regime summaries assign equal weight
to each frozen scenario within its R0-R5 regime.

Structural checks
-----------------
180 total scenarios.
36 CORE scenarios.
144 STRESS scenarios.
R0/R1/R2/R3/R4/R5 counts = 6/6/78/6/6/78.
1260 rows for the seven frozen main models.
B0 delta-C, negative-transfer rate, and catastrophic-transfer rate are exactly zero
for every frozen scenario.

Frozen aggregate cross-check
----------------------------
Every exported all-model overall quantity is independently compared against the
pre-existing 05d model_aggregate_summary.tsv.

Accepted replay cross-checks
----------------------------
Previously accepted post-HOLD all-model Uno-C, negative-transfer rates, and R5
catastrophic rates must also be reproduced within the frozen rounding tolerance.

Key quantities after a PASS
---------------------------
A2 mean Uno-C: 0.569809
A2 negative-transfer rate: 0.215417
A3 mean Uno-C: 0.530067
A3 negative-transfer rate: 0.486333

Prohibited operations
---------------------
No model fitting.
No retuning.
No reading of TARGET, GSE21257, GSE39055, or DOG2 outcomes.
No recomputation of survival metrics from predictions.
No threshold change.
No architecture-selection change.
