Paper 6 - 05h4 threshold-definition sensitivity

Role
----
POST-HOLD SENSITIVITY OF THE ESTIMATED SAFETY PROFILE.
This is NOT a reopened pass/fail decision.

Primary definitions remain unchanged
------------------------------------
Negative transfer: delta Uno-C <= -0.02.
Catastrophic transfer: delta Uno-C <= -0.05.
The original frozen architecture-selection decision is never replaced.

Prespecified neighboring severity definitions from 05h0
-------------------------------------------------------
Negative transfer: -0.01, -0.02, -0.03.
Catastrophic transfer: -0.04, -0.05, -0.06.

Aggregation
-----------
For each fixed frozen scenario, the replicate-level rate is calculated at
the specified severity threshold. The 180 scenario rates are then averaged
with equal scenario weight, matching the frozen-original estimand.

Reference rate lines
--------------------
Negative-transfer descriptive reference lines: 0.05, 0.10, 0.15.
Catastrophic-transfer descriptive reference lines: 0.025, 0.05, 0.075.
These are presentation aids only; neighboring definitions/reference lines
cannot create a new PASS or replace the original frozen decision.

Results
-------
catastrophic_transfer / A2 / threshold -0.04: rate=0.106972; change_vs_primary=+0.035000
catastrophic_transfer / A2 / threshold -0.05: rate=0.071972; change_vs_primary=+0.000000 [FROZEN PRIMARY DEFINITION]
catastrophic_transfer / A2 / threshold -0.06: rate=0.047306; change_vs_primary=-0.024667
catastrophic_transfer / A3 / threshold -0.04: rate=0.375889; change_vs_primary=+0.049000
catastrophic_transfer / A3 / threshold -0.05: rate=0.326889; change_vs_primary=+0.000000 [FROZEN PRIMARY DEFINITION]
catastrophic_transfer / A3 / threshold -0.06: rate=0.281833; change_vs_primary=-0.045056
negative_transfer / A2 / threshold -0.01: rate=0.289611; change_vs_primary=+0.074194
negative_transfer / A2 / threshold -0.02: rate=0.215417; change_vs_primary=+0.000000 [FROZEN PRIMARY DEFINITION]
negative_transfer / A2 / threshold -0.03: rate=0.153389; change_vs_primary=-0.062028
negative_transfer / A3 / threshold -0.01: rate=0.546778; change_vs_primary=+0.060444
negative_transfer / A3 / threshold -0.02: rate=0.486333; change_vs_primary=+0.000000 [FROZEN PRIMARY DEFINITION]
negative_transfer / A3 / threshold -0.03: rate=0.429250; change_vs_primary=-0.057083

Technical provenance
--------------------
No model was refit.
No human outcome was read.
No schema discovery was performed.
All retained NPZ identities/model indices come from the PASS 05h3c contract.
The -0.02/-0.05 rows must exactly reproduce the frozen-original rates in 05h3d.
