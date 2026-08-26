Paper 6 DOG2 parent-trial -> RNA186 selection audit
Script version: 02b-audit-dog2-parent-trial-selection-v2-no-cli

Status
------
BLOCKED_NO_PARENT_TRIAL_PATIENT_TABLE

Reason
------
No local lightweight table with at least 250 unique COTC subjects was found. The 324/309 parent-trial -> RNA186 selection mechanism cannot yet be audited at patient level.

Question
--------
Can the 186-dog pretreatment RNA subset be treated as preserving enough of the
original randomized parent-trial structure to justify later treatment-arm
invariance analyses?

Interpretation
--------------
This script is deliberately conservative. It requires a patient-level parent
table with explicit COTC subject identifiers and strong linkage to the frozen
186-subject mapping from Script 02a.

Even a PASS-like descriptive result does not prove that RNA availability,
pathology review, tissue quality, or record completeness were independent of
all potential outcomes or modifiers. It only supports proceeding to a separate
randomization-protocol audit.

Published parent counts (324 randomized; 309
ITT) are reference values only. The script reports the locally observed
patient-level table rather than forcing these counts.

No molecular model, survival association, treatment effect, or treatment-effect
heterogeneity analysis is performed here.
