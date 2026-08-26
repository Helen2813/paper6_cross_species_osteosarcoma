Paper 6 DOG2 transcriptomic-selection audit
Script version: 02-audit-dog2-transcriptomic-selection-v1-no-cli

Question
--------
Does the 186-dog bulk-RNA subset preserve enough of the original COTC021/022
randomized parent cohort structure to justify later treatment-arm invariance
analyses?

Published design facts used only as reference
---------------------------------------------
- Randomized parent trial enrollment: 324 dogs.
- Intent-to-treat population reported in the trial publication: 309.
- Bulk-RNA subset: 186 dogs.

Current gate
------------
BLOCKED_SELECTED_ID_MAPPING_INCOMPLETE

Reason
------
Fewer than 170 of 186 transcriptomic dogs could be resolved to COTC021/022 subject IDs from the locked clinical table.

Important
---------
The original trial randomization applies to the randomized parent population.
Subsetting by tissue availability, RNA quality, pathology review, or medical
record completeness can induce selection bias. A balanced 93/93 RNA subset is
therefore not sufficient by itself to justify exact randomization inference.

This script does not fit any survival or molecular model and does not perform
treatment-effect discovery.
