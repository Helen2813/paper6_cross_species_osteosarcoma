# Paper 6 CBM strengthening contract

**Contract version:** `cbm-strengthening-v1`  
**Script version:** `05h0-freeze-cbm-strengthening-contract-v1-no-cli`

## Purpose

Freeze a limited, post-HOLD strengthening program for submission to Computers in Biology and Medicine without reopening the primary benchmark, model selection, TARGET interpretation, or sealed external cohorts.

## Existing science remains locked

- Primary HOLD decision: **LOCKED AND UNCHANGED**
- Selectable architectures: **A2 and A3 only**
- Original architectures and hyperparameters: **LOCKED**
- Original benchmark predictions: **reuse only; no refitting**
- TARGET interpretation: **T-D remains unchanged**
- Primary negative-transfer rule: **Delta Uno-C <= -0.02; maximum allowed rate 0.10**
- Primary catastrophic-transfer rule: **Delta Uno-C <= -0.05; maximum allowed rate 0.05**
- GSE21257 outcomes: **SEALED**
- GSE39055 outcomes: **SEALED**

## Authorized strengthening work

1. Make the M34/M40/M11/M24 anchor provenance self-contained and machine-readable.
2. Report an all-model descriptive safety summary for B0, B4, A0, A1, A2, A3, and A4.
3. Recompute A2/A3 aggregate summaries under:
   - the unchanged original 180-scenario equal-scenario weighting;
   - the exact 36 pre-stress core scenarios;
   - a six-regime balanced estimand giving R0--R5 equal weight.
4. Report prespecified neighboring negative/catastrophic-transfer severity definitions as
   **sensitivity of the estimated safety profile**, never as a reopened pass/fail decision.
5. Add a post-opening no-covariate Kaplan-Meier IBS reference using only frozen outer-training
   outcomes and the existing TARGET evaluation mechanics.
6. Fix terminology and interpretation for prognostic compatibility, no-target-outcome transfer,
   AUROC 0.254, N4/N5, and T1 IBS.
7. Rebuild Figure 4, clean Figure 5B, verify recent literature, and convert the package to
   Computers in Biology and Medicine / Elsevier submission structure.

## Explicit prohibitions

- No new transfer architecture or domain-adaptation method.
- No original benchmark refitting.
- No retrospective A2/A3 eligibility changes.
- No retrospective threshold changes.
- No reopening TARGET T-D.
- No GSE21257/GSE39055 outcome access.
- No hiding an unfavorable sensitivity result.

## Reporting commitment

Every analysis authorized by this contract must be reported regardless of whether it strengthens, weakens, or leaves unchanged the current manuscript conclusions.
