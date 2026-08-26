Paper 6 ICDC parent-trial case fetch
Script version: 02d-fetch-icdc-parent-trial-cases-v1-no-cli

Status
------
BLOCKED_NO_EXACT_309_CASE_QUERY

Reason
------
No introspected direct study-linked case/clinical GraphQL field returned exactly 152 COTC021 and 157 COTC022 records.

Method
------
The script uses live GraphQL introspection rather than hard-coding an assumed
ICDC case query. It searches only direct query fields with a study-code
argument and case/subject/clinical semantics, introspects each return object,
requests only scalar fields, and selects a query only if it returns exactly
152 COTC021 and 157 COTC022 records.

No omics files are downloaded. The saved JSON/CSV files are small clinical/case
metadata snapshots used only to establish the parent population and identifier
linkage before the RNA186 selection audit.
