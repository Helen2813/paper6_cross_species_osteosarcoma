Paper 6 human cohort inventory
Script version: 01-inventory-human-osteosarcoma-cohorts-v1-no-cli

This stage is a premise audit, not a modelling stage.

What this script does
---------------------
1. Re-verifies the Script-00 SHA-256 lock for locally available human inputs.
2. Audits expression/clinical sample-ID matching for TARGET-OS, GSE21257,
   and GSE39055.
3. Enumerates clinical columns and flags endpoint-like fields without fitting
   any outcome model.
4. Computes exact normalized sample-ID overlap across the locally available
   cohorts.
5. Writes a candidate registry of public human osteosarcoma datasets relevant
   to later endpoint and sample-lineage auditing.

Critical interpretation rule
----------------------------
Zero exact sample-ID overlap does NOT establish patient-level independence.
Different GEO accessions can rename the same patient/specimen. Independence
must be established from source publications, sample annotations, and, where
possible, patient/sample lineage identifiers.

The pooled-human-source comparator remains NOT READY after this script.
No scientific dataset is copied into the Paper 6 repository.
