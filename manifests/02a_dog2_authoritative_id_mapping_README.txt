Paper 6 authoritative DOG2 identifier bridge
Script version: 02a-resolve-dog2-authoritative-ids-v3-no-cli

Status
------
PASS

Reason
------
The locked 186-dog Paper 4 clinical cohort matches the 186 authoritative GSE238110/Paper 5 COTC subjects exactly through a globally unique numeric patient identifier bridge.

Why earlier versions failed
---------------------------
The processed Paper 4 clinical table does not store COTC subject strings or a
unique GSM identifier. Earlier scripts therefore attempted joins in identifier
spaces that were not actually present.

Version 3 uses the matching relationship that was used in Paper 4 itself:
the DOG2 clinical data contain a numeric Patient ID, while official GSE238110
COTC titles contain the same numeric patient identifier as the subject suffix.

This bridge is accepted only if:
1. all 186 authoritative COTC subjects have globally unique numeric suffixes;
2. one locked clinical identifier field contains exactly 186 unique IDs;
3. the two 186-ID sets are exactly equal; and
4. the resulting COTC set exactly equals the authoritative Paper 5 subject set.

If any condition fails, the script stops. It never falls back to row order.

No scientific dataset is copied. No outcome model or feature selection is run.
