Paper 6 ICDC COTC021/COTC022 parent roster
Script version: 02d2-fetch-icdc-parent-roster-globalsearch-v1-no-cli

Status
------
PASS_PARENT309_ROSTER_LOCK_READY

Reason
------
ICDC globalSearch returned exactly 152 COTC021 and 157 COTC022 cases; all 309 case IDs map uniquely to COTC subjects, all 186 frozen RNA-profiled DOG2 subjects are contained in the parent roster, and exactly 123 parent cases are not in RNA186.

Method
------
The ICDC public frontend uses globalSearch(input, first, offset) to retrieve
case search results including case_id and clinical_study_designation. This
script queries each trial study code, strictly filters returned case rows to
the requested study designation, and requires the published/public ICDC counts
of 152 COTC021 and 157 COTC022 cases.

The parent roster is accepted only if:
- 309 total rows are returned;
- all 309 case IDs are unique;
- all 309 case IDs parse to unique COTC subject identifiers;
- all frozen 186 RNA-profiled DOG2 subjects occur in that parent roster; and
- exactly 123 parent subjects remain outside RNA186.

No omics data are downloaded and no outcome model is fit.
