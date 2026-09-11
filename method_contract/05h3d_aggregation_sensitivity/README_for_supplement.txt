Paper 6 - 05h3d frozen aggregation sensitivity

Analysis role
-------------
POST-HOLD ROBUSTNESS ANALYSIS.
The frozen-original 180-scenario equal-weight estimand remains PRIMARY and unchanged.
core_only and regime_balanced are sensitivity estimands only.

Technical binding
-----------------
05h3d performs no schema discovery and no label interpretation.
All source hashes, columns, scenario memberships, regime mappings, NPZ paths,
model-axis indices, and threshold identities come from the PASS 05h3c contract.

Estimands
---------
frozen_original: all 180 frozen scenarios, equal weight per scenario.
core_only: exact 36 CORE scenarios frozen by 05h3c, equal weight per scenario.
regime_balanced: all 180 scenarios; equal scenario weight within each R0-R5,
then equal 1/6 weight for each regime.

Metrics
-------
mean delta Uno-C vs B0.
negative-transfer rate at delta C <= -0.02.
catastrophic-transfer rate at delta C <= -0.05.

Monte Carlo uncertainty
-----------------------
4,000 deterministic within-scenario replicate bootstrap draws.
Seed: 2070713194, derived from the frozen 05h3c contract hash.
Scenarios are fixed and are never resampled.
The reported 2.5% and 97.5% quantiles quantify Monte Carlo uncertainty from
the retained finite replicate runs under the fixed scenario design. They are not
superpopulation confidence intervals over possible scenario designs.

Point estimates
---------------
frozen_original / A2: mean_delta_uno_c=0.014254; negative_transfer_rate_at_deltaC_le_-0.02=0.215417; catastrophic_transfer_rate_at_deltaC_le_-0.05=0.071972
frozen_original / A3: mean_delta_uno_c=-0.025487; negative_transfer_rate_at_deltaC_le_-0.02=0.486333; catastrophic_transfer_rate_at_deltaC_le_-0.05=0.326889
core_only / A2: mean_delta_uno_c=0.019287; negative_transfer_rate_at_deltaC_le_-0.02=0.209583; catastrophic_transfer_rate_at_deltaC_le_-0.05=0.085694
core_only / A3: mean_delta_uno_c=0.003248; negative_transfer_rate_at_deltaC_le_-0.02=0.324722; catastrophic_transfer_rate_at_deltaC_le_-0.05=0.153056
regime_balanced / A2: mean_delta_uno_c=0.015577; negative_transfer_rate_at_deltaC_le_-0.02=0.219263; catastrophic_transfer_rate_at_deltaC_le_-0.05=0.085374
regime_balanced / A3: mean_delta_uno_c=-0.001132; negative_transfer_rate_at_deltaC_le_-0.02=0.343718; catastrophic_transfer_rate_at_deltaC_le_-0.05=0.173547

Interpretation guardrail
------------------------
Neither sensitivity estimand can replace the frozen original estimand, alter
the -0.02/-0.05 definitions, reopen A2/A3 architecture selection, or alter
HOLD_NO_SAFE_SELECTABLE_ARCHITECTURE.
