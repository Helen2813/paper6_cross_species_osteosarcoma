Paper 6 upstream input lock
Script version: 00-lock-upstream-inputs-v1-no-cli

Purpose
-------
Paper 6 reads immutable upstream data directly from Paper 4 rather than copying
large scientific files into the Paper 6 repository.

This lock records:
- Paper-4-relative file path
- SHA-256
- file size
- lightweight row/column dimensions for tabular inputs
- verification against the Paper 4 frozen-program manifest where applicable

The absolute Paper 4 path is intentionally NOT written into the committed
manifest or lock JSON. This keeps machine-specific paths out of Git.

Paper 4 root resolution
-----------------------
Resolved via: sibling_repository

Locked assets
-------------
14

Scientific guardrails
---------------------
1. This script performs no model fitting.
2. This script performs no feature selection.
3. This script performs no outcome association testing.
4. This script copies no scientific datasets.
5. Downstream Paper 6 scripts should verify these hashes before analysis.
6. Paper 4 frozen assets are reference/provenance inputs; Paper 6 must not
   silently modify their scientific definitions.
