Paper 6 ICDC parent-trial API probe
Script version: 02c-probe-icdc-parent-trial-api-v1-no-cli

Status
------
PASS_READY_FOR_ICDC_CASE_QUERY_DESIGN

Reason
------
ICDC exposes COTC021 and COTC022 with the expected public case counts (152 + 157 = 309), and GraphQL schema introspection succeeded. A case-level metadata query can now be frozen in the next script.

Why this stage exists
---------------------
The local Paper 4 repository does not contain a patient-level parent-trial
table large enough to audit selection into the 186-dog RNA cohort.

The ICDC publicly exposes the related COTC021 and COTC022 studies and clinical
metadata through its data commons/API. This probe verifies the live study
counts and discovers the current GraphQL query surface before any case-level
download logic is frozen.

No omics files are downloaded. No survival, treatment-effect, or molecular
model is fit.
