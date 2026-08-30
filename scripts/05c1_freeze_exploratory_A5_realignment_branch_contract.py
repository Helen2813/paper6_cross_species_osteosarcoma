#!/usr/bin/env python3
"""
Paper 6 - freeze exploratory A5 realignment/rejection branch before 05d.

This contract is intentionally frozen AFTER the empirical GSE16091 premise
observation but BEFORE looking at 05d synthetic phase-diagram results.

Purpose
-------
A5 is NOT part of the frozen primary A2-vs-A3 architecture comparison.
It is an explicitly exploratory extension that may be opened only by a
prespecified 05d failure signature.

A5 concept
----------
Evolution-Conditioned Hierarchical Realignment-or-Rejection (A5-ECHRR)

For each module, the conceptual states are:
    DIRECT   : source representation/effect can be borrowed directly
    REALIGN  : source biology may be reusable after outcome-blind
               representation correction
    REJECT   : source prognostic information should not be borrowed

Identifiability constraint
--------------------------
Expression alone cannot generally distinguish REALIGN from REJECT when the
difference is prognostic-effect transportability. Therefore A5 is NOT allowed
to estimate independent REALIGN-vs-REJECT probabilities module by module from
29 target events. The REALIGN-vs-REJECT split is hierarchical:
    - shared across prespecified evolutionary-conservation strata;
    - target-outcome evidence updates stratum-level parameters;
    - module-specific deviations are strongly shrunk.

The representation realigner itself is outcome-blind.

No model fitting.
No network access.
No real outcome/expression values read.
No 05d results read.
No GSE16091 use by A5 is ever allowed.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


SCRIPT_VERSION = "05c1-freeze-exploratory-A5-realignment-branch-v1-no-cli"

ROOT = Path(__file__).resolve().parents[1]

A_DIR = ROOT / "results" / "simulation_contract" / "05a"
A_CONTRACT = A_DIR / "negative_transfer_simulation_contract.json"
A_SUMMARY = A_DIR / "summary.json"

C_DIR = ROOT / "results" / "simulation_model_matrix" / "05c"
C_SUMMARY = C_DIR / "summary.json"

C0_DIR = ROOT / "results" / "simulation_model_matrix_diagnostics" / "05c0"
C0_SUMMARY = C0_DIR / "summary.json"

OUT_DIR = ROOT / "results" / "simulation_extension_contract" / "05c1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONTRACT_JSON = OUT_DIR / "A5_ECHRR_branch_contract.json"
BRANCH_RULES_TSV = OUT_DIR / "A5_branch_rules.tsv"
IDENT_GRID_TSV = OUT_DIR / "A5_state_identifiability_event_grid.tsv"
COMPARATORS_TSV = OUT_DIR / "A5_comparator_registry.tsv"
SUMMARY_JSON = OUT_DIR / "summary.json"

# This is the planned 05d result root. We also scan more broadly below.
PLANNED_05D_SUMMARY = (
    ROOT / "results" / "simulation_phase_diagram" / "05d" / "summary.json"
)

EXPECTED_05A_STATUS = (
    "PASS_NEGATIVE_TRANSFER_SIMULATION_AND_AI_SELECTION_CONTRACT_FROZEN"
)
EXPECTED_05C_STATUS = (
    "PASS_CLOSED_SYNTHETIC_TRANSFER_MODEL_MATRIX_COMPLETE"
)
EXPECTED_05C0_STATUS = (
    "PASS_SYNTHETIC_MODEL_MATRIX_EXECUTION_COMPLETE"
)

# Reuse the scientifically meaningful 0.02 Uno-C unit already frozen in 05a
# for negative-transfer definition. No new data-dependent effect-size scale.
SHIFT_SPECIFIC_C_DROP = 0.02
N_REQUIRED_EVENT_SLICES = 2
PRESPECIFIED_SHIFT_EVENT_SLICES = [10, 15, 29, 40]

# Existing frozen A3 safety/recovery thresholds from 05a.
MAX_NEGATIVE_TRANSFER_RATE = 0.10
MAX_CATASTROPHIC_NEGATIVE_TRANSFER_RATE = 0.05
MIN_MODULE_RECOVERY_AUROC = 0.70
MAX_FALSE_BORROW_RATE = 0.20

# A5 identifiability study deliberately extends beyond the empirical target
# event budget to learn whether the third state is estimable at all.
A5_IDENTIFIABILITY_EVENT_GRID = [10, 15, 29, 40, 60, 100, 150]
A5_EMPIRICALLY_RELEVANT_EVENT_CEILING = 40
A5_MIN_STATE_DISCRIMINATION_AUROC = 0.70

# Closed exploratory implementation grid. If branch opens, A5a may select only
# among these values using NEW synthetic data; no empirical outcomes.
REALIGNMENT_RANK_GRID = [2, 4]
IDENTITY_PENALTY_GRID = [0.1, 1.0]
HIERARCHY_SHRINKAGE_GRID = [0.1, 1.0]
N_CONSERVATION_STRATA = 5


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_file(path: Path) -> Path:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Required frozen artifact missing: {path}")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def find_existing_05d_results() -> List[Path]:
    candidates: List[Path] = []

    if PLANNED_05D_SUMMARY.exists():
        candidates.append(PLANNED_05D_SUMMARY)

    # Conservative pre-registration guard: fail if any Paper-6 results folder
    # already contains a 05d-like summary. This prevents accidental freezing
    # after inspecting 05d under a slightly different directory name.
    results_root = ROOT / "results"
    if results_root.exists():
        for path in results_root.rglob("summary.json"):
            parts_lower = [p.lower() for p in path.parts]
            if any(
                part == "05d"
                or part.startswith("05d_")
                or part.startswith("05d-")
                for part in parts_lower
            ):
                if path not in candidates:
                    candidates.append(path)

    return sorted(set(candidates))


def main() -> None:
    started = now_utc()

    print("=" * 120)
    print("Paper 6 - freeze exploratory A5 realignment/rejection branch before 05d")
    print("=" * 120)
    print(f"Script version: {SCRIPT_VERSION}")
    print()
    print("Safety / chronology:")
    print("  Real DOG2/GSE16091/human outcome values read: NO")
    print("  Real expression values read: NO")
    print("  05d phase-diagram results read: NO")
    print("  A5 model fitting: NO")
    print("  Network access: NO")
    print("  GPU execution: NO")
    print("  A5 allowed to use GSE16091 now or later: NO")
    print()

    for path in [A_CONTRACT, A_SUMMARY, C_SUMMARY, C0_SUMMARY]:
        require_file(path)

    existing_05d = find_existing_05d_results()
    if existing_05d:
        joined = "\n  ".join(str(p) for p in existing_05d)
        raise RuntimeError(
            "A 05d-like summary already exists. This A5 contract must be frozen "
            "before inspection of 05d results. Existing files:\n  " + joined
        )

    a_summary = read_json(A_SUMMARY)
    c_summary = read_json(C_SUMMARY)
    c0_summary = read_json(C0_SUMMARY)

    if str(a_summary.get("scientific_status")) != EXPECTED_05A_STATUS:
        raise RuntimeError("05a is not in expected frozen PASS state.")
    if str(c_summary.get("scientific_status")) != EXPECTED_05C_STATUS:
        raise RuntimeError("05c is not in expected PASS state.")
    if str(c0_summary.get("scientific_status")) != EXPECTED_05C0_STATUS:
        raise RuntimeError("05c0 execution audit is not in expected PASS state.")

    # ------------------------------------------------------------------
    # Branch rules.
    # ------------------------------------------------------------------
    branch_rows = [
        {
            "rule_id": "A5_CLOSE_PRIMARY_SUCCESS",
            "priority": 1,
            "action": "CLOSE_A5",
            "condition": (
                "A3 satisfies ALL frozen 05a safety/recovery criteria AND "
                "A3 is selected by the frozen 05d A2-vs-A3 rule."
            ),
            "rationale": (
                "If A3 already solves safe selective borrowing under the frozen "
                "criteria, added architecture complexity is not justified."
            ),
        },
        {
            "rule_id": "A5_OPEN_SHIFT_SPECIFIC",
            "priority": 2,
            "action": "OPEN_A5",
            "condition": (
                "A3 preserves misleading-source rejection safety "
                "(catastrophic negative-transfer rate <=0.05 and false-borrow "
                "rate <=0.20), BUT in R2 partially transportable / correct-prior "
                "stress scenarios the matched A3-vs-B0 mean Uno-C advantage "
                "drops by >=0.02 under severe covariance shift OR severe mapping "
                "error relative to the matched no-shift/no-mapping condition in "
                "at least 2 of the prespecified event slices {10,15,29,40}; "
                "and A3 is not the frozen final selected architecture."
            ),
            "rationale": (
                "Open A5 only for the specific failure mode it was designed to "
                "address: representation/mapping shift despite preserved ability "
                "to reject misleading source information."
            ),
        },
        {
            "rule_id": "A5_CLOSE_GENERAL_FAILURE",
            "priority": 3,
            "action": "CLOSE_A5",
            "condition": (
                "A3 failure is not shift-specific, OR misleading-source rejection "
                "safety itself fails, OR the shift-specific >=0.02 deterioration "
                "criterion is seen in fewer than 2 prespecified event slices."
            ),
            "rationale": (
                "A representation-realignment extension is not a principled "
                "response to general transfer failure or inability to reject "
                "misleading biology."
            ),
        },
    ]
    branch_df = pd.DataFrame(branch_rows)
    branch_df.to_csv(BRANCH_RULES_TSV, sep="\t", index=False)

    # ------------------------------------------------------------------
    # Identifiability feasibility grid.
    # ------------------------------------------------------------------
    ident_df = pd.DataFrame(
        {
            "target_events": A5_IDENTIFIABILITY_EVENT_GRID,
            "empirically_relevant_for_current_human_setting": [
                int(x <= A5_EMPIRICALLY_RELEVANT_EVENT_CEILING)
                for x in A5_IDENTIFIABILITY_EVENT_GRID
            ],
            "primary_state_discrimination_target": [
                "REALIGN_vs_REJECT" for _ in A5_IDENTIFIABILITY_EVENT_GRID
            ],
            "minimum_AUROC_for_usable_state_discrimination": [
                A5_MIN_STATE_DISCRIMINATION_AUROC
                for _ in A5_IDENTIFIABILITY_EVENT_GRID
            ],
        }
    )
    ident_df.to_csv(IDENT_GRID_TSV, sep="\t", index=False)

    # ------------------------------------------------------------------
    # Mandatory comparators.
    # ------------------------------------------------------------------
    comparator_rows = [
        {
            "model": "B0",
            "role": "target-only baseline",
            "mandatory": True,
        },
        {
            "model": "B2",
            "role": "DOG2 zero-shot; mandatory no-realignment source benchmark",
            "mandatory": True,
        },
        {
            "model": "B3",
            "role": (
                "global outcome-blind CORAL+COX representation-alignment comparator"
            ),
            "mandatory": True,
        },
        {
            "model": "B4",
            "role": "classical residual-transfer comparator",
            "mandatory": True,
        },
        {
            "model": "A2",
            "role": "frozen primary selectable adapter comparator",
            "mandatory": True,
        },
        {
            "model": "A3",
            "role": "frozen primary selective-borrowing comparator",
            "mandatory": True,
        },
        {
            "model": "A5_ECHRR",
            "role": "exploratory hierarchical realignment-or-rejection extension",
            "mandatory": True,
        },
    ]
    comparator_df = pd.DataFrame(comparator_rows)
    comparator_df.to_csv(COMPARATORS_TSV, sep="\t", index=False)

    contract: Dict[str, Any] = {
        "script_version": SCRIPT_VERSION,
        "status": "PASS",
        "scientific_status": (
            "PASS_A5_EXPLORATORY_BRANCH_SPECIFICATION_FROZEN_BEFORE_05D"
        ),
        "created_utc": now_utc(),
        "chronology": {
            "developed_after_GSE16091_premise_observation": True,
            "frozen_before_05d_phase_diagram_results": True,
            "05d_results_existing_at_freeze": False,
            "primary_A2_vs_A3_registry_changed": False,
        },
        "A5": {
            "code": "A5_ECHRR",
            "name": (
                "Evolution-Conditioned Hierarchical Realignment-or-Rejection"
            ),
            "status": "EXPLORATORY_EXTENSION_NOT_PRIMARY",
            "states": ["DIRECT", "REALIGN", "REJECT"],
            "terminology_rule": (
                "Use realignment / representation correction, not translation, "
                "to avoid collision with translational-oncology terminology."
            ),
            "identifiability_principle": (
                "REALIGN versus REJECT cannot generally be identified from "
                "outcome-blind expression alone. Target outcomes may update only "
                "hierarchically pooled REALIGN-vs-REJECT parameters, not freely "
                "estimated independent per-module states."
            ),
            "hierarchy": {
                "n_evolutionary_conservation_strata": N_CONSERVATION_STRATA,
                "strata_definition": (
                    "Outcome-blind quintiles of the future frozen evolutionary "
                    "conservation score; ties resolved deterministically."
                ),
                "module_specific_deviations": (
                    "allowed only with strong hierarchical shrinkage"
                ),
            },
            "representation_realigner": {
                "uses_target_outcomes": False,
                "purpose": (
                    "correct cross-species representation geometry before "
                    "borrowing, while remaining near identity when conservation "
                    "and geometry support direct transport"
                ),
                "closed_rank_grid": REALIGNMENT_RANK_GRID,
                "closed_identity_penalty_grid": IDENTITY_PENALTY_GRID,
            },
            "outcome_gate": {
                "uses_target_outcomes": True,
                "level": (
                    "evolutionary-conservation stratum with shrunk module deviations"
                ),
                "closed_hierarchy_shrinkage_grid": HIERARCHY_SHRINKAGE_GRID,
            },
        },
        "branch_gate": {
            "prespecified_event_slices": PRESPECIFIED_SHIFT_EVENT_SLICES,
            "shift_specific_effect_size": SHIFT_SPECIFIC_C_DROP,
            "minimum_event_slices_meeting_shift_signature": (
                N_REQUIRED_EVENT_SLICES
            ),
            "rules_file": str(BRANCH_RULES_TSV.relative_to(ROOT)),
            "rule_priority": [
                "A5_CLOSE_PRIMARY_SUCCESS",
                "A5_OPEN_SHIFT_SPECIFIC",
                "A5_CLOSE_GENERAL_FAILURE",
            ],
        },
        "A5_simulation_if_opened": {
            "first_question": (
                "How many target events are required to distinguish REALIGN from "
                "REJECT under hierarchical pooling?"
            ),
            "identifiability_event_grid": A5_IDENTIFIABILITY_EVENT_GRID,
            "state_discrimination_metric": "AUROC_REALIGN_vs_REJECT",
            "minimum_state_discrimination_AUROC": (
                A5_MIN_STATE_DISCRIMINATION_AUROC
            ),
            "empirical_relevance_rule": (
                "If REALIGN-vs-REJECT AUROC first reaches 0.70 only above 40 "
                "target events, A5 is not eligible for empirical Paper-6 human "
                "evaluation and remains simulation/conceptual only."
            ),
            "mandatory_comparators": comparator_df[
                comparator_df["mandatory"]
            ]["model"].tolist(),
            "predictive_success_rule": (
                "Within the shift scenarios that opened A5, mean Uno-C improvement "
                "of A5 over A3 must be >=0.02, and A5 must not be inferior to both "
                "B2 zero-shot and B3 CORAL alignment. Safety must remain within "
                "the frozen 05a catastrophic-negative-transfer and false-borrow "
                "limits."
            ),
            "nonshift_safety_rule": (
                "In fully transportable/no-shift scenarios, A5 mean Uno C may not "
                "degrade by >0.01 versus A3."
            ),
            "stopping_rule": (
                "If identifiability or predictive/safety criteria fail, stop A5 "
                "development for Paper 6 and do not evaluate A5 on real human outcomes."
            ),
        },
        "real_data_firewall": {
            "GSE16091": "FORBIDDEN_FOR_A5_FOREVER",
            "reason": (
                "GSE16091 observation contributed to the A5 hypothesis; reuse "
                "would create a circular development/evaluation loop."
            ),
            "TARGET_GSE21257_GSE39055": (
                "May be evaluated only if A5 branch opens, A5 simulation criteria "
                "pass, implementation/hyperparameters are frozen, and reserved "
                "human outcomes remain unopened until that freeze."
            ),
        },
        "frozen_existing_thresholds_reused": {
            "max_negative_transfer_rate": MAX_NEGATIVE_TRANSFER_RATE,
            "max_catastrophic_negative_transfer_rate": (
                MAX_CATASTROPHIC_NEGATIVE_TRANSFER_RATE
            ),
            "min_module_recovery_AUROC": MIN_MODULE_RECOVERY_AUROC,
            "max_false_borrow_rate": MAX_FALSE_BORROW_RATE,
        },
        "provenance": {
            "05a_contract_sha256": sha256_file(A_CONTRACT),
            "05a_summary_sha256": sha256_file(A_SUMMARY),
            "05c_summary_sha256": sha256_file(C_SUMMARY),
            "05c0_summary_sha256": sha256_file(C0_SUMMARY),
            "this_script_sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    write_json(CONTRACT_JSON, contract)

    summary = {
        "script_version": SCRIPT_VERSION,
        "status": "PASS",
        "scientific_status": contract["scientific_status"],
        "run_started_utc": started,
        "run_finished_utc": now_utc(),
        "A5_primary_model": False,
        "A5_branch_currently_open": False,
        "A5_branch_decision_deferred_to_prespecified_05d_signature": True,
        "GSE16091_allowed_for_A5": False,
        "05d_results_read": False,
        "model_fitting": False,
        "network_access": False,
        "GPU_execution": False,
        "final_artifact_hashes": {
            "A5_ECHRR_branch_contract_json": sha256_file(CONTRACT_JSON),
            "A5_branch_rules_tsv": sha256_file(BRANCH_RULES_TSV),
            "A5_state_identifiability_event_grid_tsv": sha256_file(
                IDENT_GRID_TSV
            ),
            "A5_comparator_registry_tsv": sha256_file(COMPARATORS_TSV),
        },
        "next": (
            "Commit these artifacts before running 05d. Then run 05d and apply "
            "the branch rules mechanically."
        ),
    }
    write_json(SUMMARY_JSON, summary)

    print("-" * 120)
    print("A5 exploratory extension")
    print("-" * 120)
    print("Name: Evolution-Conditioned Hierarchical Realignment-or-Rejection")
    print("Primary A2-vs-A3 registry changed: NO")
    print("A5 branch open now: NO")
    print("GSE16091 allowed for A5: NO")
    print()
    print("05d branch rule:")
    print(
        "  CLOSE if A3 passes frozen safety/recovery and is the frozen selected architecture."
    )
    print(
        "  OPEN only for prespecified shift-specific degradation while misleading-source rejection remains safe."
    )
    print(
        "  CLOSE for general A3 failure or failure to reject misleading source biology."
    )
    print()
    print("A5 identifiability event grid:")
    print(f"  {A5_IDENTIFIABILITY_EVENT_GRID}")
    print(
        "  If REALIGN-vs-REJECT AUROC >=0.70 is not achieved by <=40 events, "
        "A5 cannot enter empirical Paper-6 human evaluation."
    )
    print()
    print("Mandatory comparators:")
    print("  B0, B2, B3, B4, A2, A3, A5")
    print()
    print("=" * 120)
    print("05c1 A5 exploratory branch contract: PASS")
    print("=" * 120)
    print()
    print("IMPORTANT: git-add/commit the 05c1 artifacts BEFORE running 05d.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 120, file=sys.stderr)
        print("05c1 A5 exploratory branch contract: FAIL", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print("=" * 120, file=sys.stderr)
        raise
