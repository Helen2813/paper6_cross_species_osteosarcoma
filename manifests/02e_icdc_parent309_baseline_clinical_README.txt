Paper 6 ICDC parent309 baseline clinical snapshot
Script version: 02e-fetch-icdc-parent309-baseline-clinical-v1-no-cli

Status
------
FAIL_CASE_FETCH_INCOMPLETE

Reason
------
309 of 309 ICDC case-detail requests failed after retries.

Scope
-----
This stage fetches only baseline/design metadata needed to audit selection from
the 309-case public ICDC parent population into the 186-dog RNA cohort:
study identity, demographic variables, study arm/cohort, enrollment site and
subgroup, and baseline diagnosis/site/pathology fields.

Outcome/response variables are deliberately not requested. In particular, the
public ICDC case-details object also exposes response-related information, but
this Paper 6 stage does not retrieve it.

No omics files are downloaded and no predictive/causal model is fit.
