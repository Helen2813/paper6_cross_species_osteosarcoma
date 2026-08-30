#!/usr/bin/env python3
"""
Paper 6 - freeze operational A5 branch definitions before 05d.

This is an ADDENDUM to the already-passed 05c1 exploratory A5 contract.
It does not replace or reopen 05c1.

Why this addendum exists
------------------------
05c1 froze the scientific idea and high-level branch rule before 05d, but the
branch rule used several quantities that 05d will calculate. 05c1a removes the
remaining implementation freedom BEFORE 05d by freezing:

1. exact definitions of severe covariance shift and severe mapping error;
2. exact matched reference scenarios;
3. exact replicate/scenario aggregation for the shift-specific A5 trigger;
4. whether hits may be pooled across shift axes (they may NOT);
5. exact disposition if A5 closes;
6. exact conservation-stratum assignment algorithm;
7. interpretation of failure inside the deliberately small A5 hyperparameter grid.

No model fitting.
No 05d results read.
No real outcomes/expression read.
No network.
No GPU.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


SCRIPT_VERSION = "05c1a-freeze-A5-operational-branch-definitions-v1-no-cli"

ROOT = Path(__file__).resolve().parents[1]

A_DIR = ROOT / "results" / "simulation_contract" / "05a"
A_CONTRACT = A_DIR / "negative_transfer_simulation_contract.json"
A_SCENARIOS = A_DIR / "simulation_scenario_registry.tsv"
A_SUMMARY = A_DIR / "summary.json"

C1_DIR = ROOT / "results" / "simulation_extension_contract" / "05c1"
C1_CONTRACT = C1_DIR / "A5_ECHRR_branch_contract.json"
C1_RULES = C1_DIR / "A5_branch_rules.tsv"
C1_SUMMARY = C1_DIR / "summary.json"

OUT_DIR = ROOT / "results" / "simulation_extension_contract" / "05c1a"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONTRAST_REGISTRY = OUT_DIR / "A5_shift_contrast_registry.tsv"
OPERATIONAL_CONTRACT = OUT_DIR / "A5_operational_branch_definitions.json"
SUMMARY_JSON = OUT_DIR / "summary.json"

PLANNED_05D_SUMMARY = (
    ROOT / "results" / "simulation_phase_diagram" / "05d" / "summary.json"
)

EXPECTED_05A_STATUS = (
    "PASS_NEGATIVE_TRANSFER_SIMULATION_AND_AI_SELECTION_CONTRACT_FROZEN"
)
EXPECTED_05C1_STATUS = (
    "PASS_A5_EXPLORATORY_BRANCH_SPECIFICATION_FROZEN_BEFORE_05D"
)

EVENT_SLICES = [10, 15, 29, 40]
TRANSFER_REGIME = "R2_PARTIALLY_TRANSPORTABLE"
CORRECT_PRIOR = "P0_CORRECT"

SEVERE_COVARIANCE_LABEL = "S2_SEVERE"
SEVERE_COVARIANCE_STRENGTH = 0.50
NO_COVARIANCE_LABEL = "S0_NONE"
NO_COVARIANCE_STRENGTH = 0.00

SEVERE_MAPPING_LABEL = "M2_SEVERE"
SEVERE_MAPPING_FRACTION = 0.30
NO_MAPPING_LABEL = "M0_NONE"
NO_MAPPING_FRACTION = 0.00

MODERATE_COVARIANCE_LABEL = "S1_MODERATE"
MODERATE_COVARIANCE_STRENGTH = 0.25
MODERATE_CENSORING_LABEL = "C1_MODERATE"
MODERATE_CENSORING_FRACTION = 0.40

COVARIANCE_CENSOR_LEVELS = [
    ("C0_LOW", 0.20),
    ("C1_MODERATE", 0.40),
    ("C2_HIGH", 0.60),
]

SHIFT_DROP_THRESHOLD = 0.02
MIN_EVENT_SLICES_SAME_AXIS = 2

N_HALLMARK_MODULES = 50
N_CONSERVATION_STRATA = 5
MODULES_PER_STRATUM = 10

# These are the deliberately small closed 05c1 grids.
EXPECTED_REALIGNMENT_RANK_GRID = [2, 4]
EXPECTED_IDENTITY_PENALTY_GRID = [0.1, 1.0]
EXPECTED_HIERARCHY_SHRINKAGE_GRID = [0.1, 1.0]


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
                candidates.append(path)

    return sorted(set(candidates))


def one_row(frame: pd.DataFrame, **criteria: Any) -> pd.Series:
    mask = np.ones(len(frame), dtype=bool)

    for key, value in criteria.items():
        if isinstance(value, float):
            mask &= np.isclose(
                pd.to_numeric(frame[key], errors="raise").to_numpy(dtype=float),
                float(value),
                rtol=1e-12,
                atol=1e-12,
            )
        else:
            mask &= frame[key].astype(str).to_numpy() == str(value)

    subset = frame.loc[mask]

    if len(subset) != 1:
        raise RuntimeError(
            f"Expected exactly one scenario for {criteria}, found {len(subset)}."
        )

    return subset.iloc[0]


def verify_05a_axis_levels(contract: Dict[str, Any]) -> None:
    stress = contract.get("stress_axes") or {}

    shift_levels = {
        str(name): float(value)
        for name, value in stress.get("covariance_shift", [])
    }
    mapping_levels = {
        str(name): float(value)
        for name, value in stress.get("mapping_error", [])
    }

    expected_shift = {
        NO_COVARIANCE_LABEL: NO_COVARIANCE_STRENGTH,
        MODERATE_COVARIANCE_LABEL: MODERATE_COVARIANCE_STRENGTH,
        SEVERE_COVARIANCE_LABEL: SEVERE_COVARIANCE_STRENGTH,
    }
    expected_mapping = {
        NO_MAPPING_LABEL: NO_MAPPING_FRACTION,
        "M1_MODERATE": 0.10,
        SEVERE_MAPPING_LABEL: SEVERE_MAPPING_FRACTION,
    }

    if shift_levels != expected_shift:
        raise RuntimeError(
            f"05a covariance-shift levels changed: {shift_levels}"
        )

    if mapping_levels != expected_mapping:
        raise RuntimeError(
            f"05a mapping-error levels changed: {mapping_levels}"
        )


def main() -> None:
    started = now_utc()

    print("=" * 120)
    print("Paper 6 - freeze operational A5 branch definitions before 05d")
    print("=" * 120)
    print(f"Script version: {SCRIPT_VERSION}")
    print()
    print("Safety / chronology:")
    print("  05d results read: NO")
    print("  Real DOG2/GSE16091/human values read: NO")
    print("  Model fitting: NO")
    print("  Network access: NO")
    print("  GPU execution: NO")
    print("  Existing 05c1 scientific contract replaced: NO [ADDENDUM ONLY]")
    print()

    for path in [
        A_CONTRACT,
        A_SCENARIOS,
        A_SUMMARY,
        C1_CONTRACT,
        C1_RULES,
        C1_SUMMARY,
    ]:
        require_file(path)

    existing_05d = find_existing_05d_results()
    if existing_05d:
        joined = "\n  ".join(str(p) for p in existing_05d)
        raise RuntimeError(
            "05d-like results already exist. 05c1a must be frozen before 05d.\n  "
            + joined
        )

    a_contract = read_json(A_CONTRACT)
    a_summary = read_json(A_SUMMARY)
    c1_contract = read_json(C1_CONTRACT)
    c1_summary = read_json(C1_SUMMARY)
    scenarios = pd.read_csv(A_SCENARIOS, sep="\t")

    if str(a_summary.get("scientific_status")) != EXPECTED_05A_STATUS:
        raise RuntimeError("05a is not in the expected frozen PASS state.")

    if str(c1_summary.get("scientific_status")) != EXPECTED_05C1_STATUS:
        raise RuntimeError("05c1 is not in the expected frozen PASS state.")

    if bool(c1_summary.get("05d_results_read")):
        raise RuntimeError("05c1 summary indicates 05d results were read.")

    verify_05a_axis_levels(a_contract)

    if sha256_file(A_SCENARIOS) != str(
        a_contract.get("scenario_registry_sha256")
    ):
        raise RuntimeError("05a scenario registry SHA256 mismatch.")

    # Verify that the small A5 search space already frozen in 05c1 has not moved.
    a5 = c1_contract.get("A5") or {}
    realigner = a5.get("representation_realigner") or {}
    outcome_gate = a5.get("outcome_gate") or {}

    if realigner.get("closed_rank_grid") != EXPECTED_REALIGNMENT_RANK_GRID:
        raise RuntimeError("05c1 A5 realignment-rank grid changed.")
    if realigner.get("closed_identity_penalty_grid") != (
        EXPECTED_IDENTITY_PENALTY_GRID
    ):
        raise RuntimeError("05c1 A5 identity-penalty grid changed.")
    if outcome_gate.get("closed_hierarchy_shrinkage_grid") != (
        EXPECTED_HIERARCHY_SHRINKAGE_GRID
    ):
        raise RuntimeError("05c1 A5 hierarchy-shrinkage grid changed.")

    hierarchy = a5.get("hierarchy") or {}
    if int(hierarchy.get("n_evolutionary_conservation_strata")) != (
        N_CONSERVATION_STRATA
    ):
        raise RuntimeError("05c1 conservation-strata count changed.")

    # ------------------------------------------------------------------
    # Exact scenario contrasts.
    # ------------------------------------------------------------------
    contrast_rows: List[Dict[str, Any]] = []

    # Covariance-shift axis:
    # S0_NONE vs S2_SEVERE, with everything else matched, separately for
    # each event slice and censoring level. The event-level trigger averages
    # the THREE censoring-specific deterioration contrasts equally.
    for events in EVENT_SLICES:
        for censor_label, censor_fraction in COVARIANCE_CENSOR_LEVELS:
            baseline = one_row(
                scenarios,
                family="SHIFT_CENSOR_STRESS",
                target_events=events,
                transfer_regime=TRANSFER_REGIME,
                covariance_shift=NO_COVARIANCE_LABEL,
                covariance_shift_strength=NO_COVARIANCE_STRENGTH,
                censoring=censor_label,
                target_censor_fraction=float(censor_fraction),
                mapping_error=NO_MAPPING_LABEL,
                mapping_error_fraction=NO_MAPPING_FRACTION,
                source_prior_state=CORRECT_PRIOR,
            )
            severe = one_row(
                scenarios,
                family="SHIFT_CENSOR_STRESS",
                target_events=events,
                transfer_regime=TRANSFER_REGIME,
                covariance_shift=SEVERE_COVARIANCE_LABEL,
                covariance_shift_strength=SEVERE_COVARIANCE_STRENGTH,
                censoring=censor_label,
                target_censor_fraction=float(censor_fraction),
                mapping_error=NO_MAPPING_LABEL,
                mapping_error_fraction=NO_MAPPING_FRACTION,
                source_prior_state=CORRECT_PRIOR,
            )

            contrast_rows.append(
                {
                    "axis": "COVARIANCE_SHIFT",
                    "target_events": events,
                    "matched_nuisance_level": censor_label,
                    "reference_scenario_id": str(baseline["scenario_id"]),
                    "stress_scenario_id": str(severe["scenario_id"]),
                    "reference_definition": (
                        "S0_NONE; same events/R2/censoring/M0_NONE/P0_CORRECT"
                    ),
                    "stress_definition": (
                        "S2_SEVERE(0.50); same events/R2/censoring/M0_NONE/P0_CORRECT"
                    ),
                    "event_level_aggregation_weight": 1.0 / 3.0,
                }
            )

    # Mapping-error axis:
    # M0_NONE vs M2_SEVERE in the frozen MAPPING_PRIOR_STRESS family.
    # Note carefully: the family itself holds covariance at S1_MODERATE and
    # censoring at C1_MODERATE. Therefore the reference is "no mapping error",
    # NOT "no shift".
    for events in EVENT_SLICES:
        baseline = one_row(
            scenarios,
            family="MAPPING_PRIOR_STRESS",
            target_events=events,
            transfer_regime=TRANSFER_REGIME,
            covariance_shift=MODERATE_COVARIANCE_LABEL,
            covariance_shift_strength=MODERATE_COVARIANCE_STRENGTH,
            censoring=MODERATE_CENSORING_LABEL,
            target_censor_fraction=MODERATE_CENSORING_FRACTION,
            mapping_error=NO_MAPPING_LABEL,
            mapping_error_fraction=NO_MAPPING_FRACTION,
            source_prior_state=CORRECT_PRIOR,
        )
        severe = one_row(
            scenarios,
            family="MAPPING_PRIOR_STRESS",
            target_events=events,
            transfer_regime=TRANSFER_REGIME,
            covariance_shift=MODERATE_COVARIANCE_LABEL,
            covariance_shift_strength=MODERATE_COVARIANCE_STRENGTH,
            censoring=MODERATE_CENSORING_LABEL,
            target_censor_fraction=MODERATE_CENSORING_FRACTION,
            mapping_error=SEVERE_MAPPING_LABEL,
            mapping_error_fraction=SEVERE_MAPPING_FRACTION,
            source_prior_state=CORRECT_PRIOR,
        )

        contrast_rows.append(
            {
                "axis": "MAPPING_ERROR",
                "target_events": events,
                "matched_nuisance_level": MODERATE_CENSORING_LABEL,
                "reference_scenario_id": str(baseline["scenario_id"]),
                "stress_scenario_id": str(severe["scenario_id"]),
                "reference_definition": (
                    "M0_NONE; same events/R2/S1_MODERATE/C1_MODERATE/P0_CORRECT"
                ),
                "stress_definition": (
                    "M2_SEVERE(0.30); same events/R2/S1_MODERATE/C1_MODERATE/P0_CORRECT"
                ),
                "event_level_aggregation_weight": 1.0,
            }
        )

    contrast_df = pd.DataFrame(contrast_rows)
    contrast_df.to_csv(CONTRAST_REGISTRY, sep="\t", index=False)

    if len(contrast_df) != 16:
        raise RuntimeError(
            f"A5 contrast registry rows={len(contrast_df)}, expected=16."
        )

    # ------------------------------------------------------------------
    # Operational definitions.
    # ------------------------------------------------------------------
    operational: Dict[str, Any] = {
        "script_version": SCRIPT_VERSION,
        "status": "PASS",
        "scientific_status": (
            "PASS_A5_OPERATIONAL_BRANCH_DEFINITIONS_FROZEN_BEFORE_05D"
        ),
        "created_utc": now_utc(),
        "relationship_to_05c1": (
            "ADDENDUM_ONLY; 05c1 scientific branch concept remains unchanged"
        ),
        "chronology": {
            "05c1_already_frozen": True,
            "05d_results_existing_at_05c1a_freeze": False,
            "05d_results_read": False,
        },
        "exact_stress_definitions": {
            "severe_covariance_shift": {
                "label": SEVERE_COVARIANCE_LABEL,
                "strength": SEVERE_COVARIANCE_STRENGTH,
                "source": "05a frozen stress axis",
            },
            "no_covariance_shift_reference": {
                "label": NO_COVARIANCE_LABEL,
                "strength": NO_COVARIANCE_STRENGTH,
                "source": "05a frozen stress axis",
            },
            "severe_mapping_error": {
                "label": SEVERE_MAPPING_LABEL,
                "fraction": SEVERE_MAPPING_FRACTION,
                "source": "05a frozen stress axis",
            },
            "no_mapping_error_reference": {
                "label": NO_MAPPING_LABEL,
                "fraction": NO_MAPPING_FRACTION,
                "source": "05a frozen stress axis",
            },
        },
        "branch_metric": {
            "per_replicate_delta_c": (
                "UnoC_A3 - UnoC_B0, evaluated on the independent synthetic "
                "target-test population for that replicate"
            ),
            "scenario_delta_c": (
                "arithmetic mean of per-replicate (UnoC_A3 - UnoC_B0) over ALL "
                "replicates in the frozen scenario; no trimming, winsorization, "
                "bootstrap weighting, or exclusion"
            ),
            "pair_deterioration": (
                "scenario_delta_c(reference) - scenario_delta_c(severe); "
                "positive values mean the severe shift eroded A3's advantage"
            ),
            "covariance_event_slice_deterioration": (
                "equal-weight arithmetic mean of the three pair_deterioration "
                "values at C0_LOW, C1_MODERATE, C2_HIGH for the same event count"
            ),
            "mapping_event_slice_deterioration": (
                "the single M0_NONE-vs-M2_SEVERE pair_deterioration at "
                "S1_MODERATE/C1_MODERATE/P0_CORRECT for that event count"
            ),
            "branch_trigger_threshold": SHIFT_DROP_THRESHOLD,
            "threshold_comparison": ">=",
            "confidence_interval_or_pvalue_used_for_branch_trigger": False,
            "reason": (
                "The 05c1 branch gate is a prespecified practical-effect trigger, "
                "not a new inferential hypothesis test."
            ),
        },
        "same_axis_requirement": {
            "minimum_event_slices": MIN_EVENT_SLICES_SAME_AXIS,
            "event_slices": EVENT_SLICES,
            "rule": (
                "A shift-specific signature exists only if covariance deterioration "
                ">=0.02 occurs in >=2 event slices on the COVARIANCE axis OR mapping "
                "deterioration >=0.02 occurs in >=2 event slices on the MAPPING axis."
            ),
            "one_covariance_hit_plus_one_mapping_hit_counts_as_two": False,
        },
        "matched_reference_rule": {
            "covariance_axis": (
                "S0_NONE versus S2_SEVERE; same event count, R2, censoring level, "
                "M0_NONE, and P0_CORRECT. Three censoring-specific contrasts are "
                "averaged equally within each event slice."
            ),
            "mapping_axis": (
                "M0_NONE versus M2_SEVERE; same event count, R2, S1_MODERATE, "
                "C1_MODERATE, and P0_CORRECT. This is explicitly NOT a no-covariance-"
                "shift reference because the frozen mapping family uses S1_MODERATE."
            ),
        },
        "A5_open_rule_operationalized": {
            "condition_1": (
                "05d shows A3 preserves misleading-source rejection safety under "
                "the already-frozen 05a definitions: catastrophic negative-transfer "
                "rate <=0.05 and false-borrow rate <=0.20."
            ),
            "condition_2": (
                "The SAME-AXIS shift-specific signature defined in this addendum is met."
            ),
            "condition_3": (
                "A3 is not the final architecture selected by the frozen 05d A2-vs-A3 rule."
            ),
            "action_if_all_true": "OPEN_A5",
            "action_otherwise": "CLOSE_A5",
            "undefined_required_metric_action": (
                "HOLD_A5_BRANCH_DECISION_FOR_TECHNICAL_RESOLUTION; do not interpret "
                "an undefined metric as either an OPEN or CLOSE signal."
            ),
        },
        "A5_close_disposition": {
            "if_closed_because_A3_succeeds": (
                "Do not implement or evaluate A5 in Paper 6. Mention only as a "
                "preregistered-but-unopened future representation-realignment direction."
            ),
            "if_closed_because_failure_is_not_shift_specific": (
                "Do not implement or evaluate A5 in Paper 6. A broader architecture "
                "may be studied only as a separate future methods project with a new "
                "prospective contract."
            ),
            "if_closed_because_A3_cannot_reject_misleading_source": (
                "Do not use representation realignment as a rescue. Paper 6 reports "
                "the failure mode under the frozen A2/A3 framework."
            ),
            "real_human_A5_evaluation_after_close": "FORBIDDEN",
        },
        "conservation_strata_algorithm": {
            "n_modules_required": N_HALLMARK_MODULES,
            "n_strata": N_CONSERVATION_STRATA,
            "modules_per_stratum": MODULES_PER_STRATUM,
            "input": (
                "one finite, outcome-blind frozen evolutionary-conservation score "
                "for each of the 50 Hallmark modules"
            ),
            "sort_order": (
                "descending conservation score; exact ties broken by Hallmark module "
                "name in ascending lexicographic order"
            ),
            "assignment": (
                "rank positions 1-10=stratum 1 (most conserved), 11-20=stratum 2, "
                "21-30=stratum 3, 31-40=stratum 4, 41-50=stratum 5"
            ),
            "missing_score_rule": (
                "A5 empirical implementation cannot proceed unless all 50 modules "
                "receive finite conservation scores under a separately frozen, "
                "outcome-blind evolutionary-prior contract. Missing scores cannot "
                "be resolved using human outcomes."
            ),
        },
        "closed_hyperparameter_space_interpretation": {
            "realignment_rank_grid": EXPECTED_REALIGNMENT_RANK_GRID,
            "identity_penalty_grid": EXPECTED_IDENTITY_PENALTY_GRID,
            "hierarchy_shrinkage_grid": EXPECTED_HIERARCHY_SHRINKAGE_GRID,
            "total_combinations": (
                len(EXPECTED_REALIGNMENT_RANK_GRID)
                * len(EXPECTED_IDENTITY_PENALTY_GRID)
                * len(EXPECTED_HIERARCHY_SHRINKAGE_GRID)
            ),
            "failure_interpretation": (
                "Failure means A5 did not meet the frozen criteria WITHIN THIS "
                "prespecified implementation/search space. It does NOT prove that "
                "all possible realignment architectures are ineffective."
            ),
            "post_result_grid_expansion_within_Paper6": "FORBIDDEN",
            "broader_grid_if_scientifically_desired": (
                "new separately preregistered future study only"
            ),
        },
        "contrast_registry": {
            "path": str(CONTRAST_REGISTRY.relative_to(ROOT)),
            "sha256": sha256_file(CONTRAST_REGISTRY),
            "n_rows": len(contrast_df),
        },
        "provenance": {
            "05a_contract_sha256": sha256_file(A_CONTRACT),
            "05a_scenario_registry_sha256": sha256_file(A_SCENARIOS),
            "05c1_contract_sha256": sha256_file(C1_CONTRACT),
            "05c1_rules_sha256": sha256_file(C1_RULES),
            "05c1_summary_sha256": sha256_file(C1_SUMMARY),
            "this_script_sha256": sha256_file(Path(__file__).resolve()),
        },
    }

    write_json(OPERATIONAL_CONTRACT, operational)

    summary = {
        "script_version": SCRIPT_VERSION,
        "status": "PASS",
        "scientific_status": operational["scientific_status"],
        "run_started_utc": started,
        "run_finished_utc": now_utc(),
        "05d_results_read": False,
        "branch_trigger_fully_operationalized": True,
        "severe_covariance_definition": "S2_SEVERE=0.50",
        "severe_mapping_definition": "M2_SEVERE=0.30",
        "same_axis_two_event_slice_rule": True,
        "conservation_strata_assignment_frozen": True,
        "A5_close_disposition_frozen": True,
        "small_grid_failure_interpretation_frozen": True,
        "model_fitting": False,
        "network_access": False,
        "GPU_execution": False,
        "final_artifact_hashes": {
            "A5_shift_contrast_registry.tsv": sha256_file(CONTRAST_REGISTRY),
            "A5_operational_branch_definitions.json": sha256_file(
                OPERATIONAL_CONTRACT
            ),
        },
        "next": (
            "Commit 05c1 + 05c1a artifacts before 05d. Then 05d must use the "
            "contrast registry and formulas in 05c1a mechanically."
        ),
    }
    write_json(SUMMARY_JSON, summary)

    print("-" * 120)
    print("Exact A5 shift trigger")
    print("-" * 120)
    print(
        "Covariance: S0_NONE(0.00) vs S2_SEVERE(0.50), matched within "
        "event/censoring/R2/M0/P0; average equally across C0/C1/C2 per event slice."
    )
    print(
        "Mapping: M0_NONE(0.00) vs M2_SEVERE(0.30), matched within "
        "event/R2/S1_MODERATE/C1_MODERATE/P0."
    )
    print()
    print(
        "Per scenario: mean_r[UnoC(A3)-UnoC(B0)]. "
        "Deterioration = reference mean deltaC - severe mean deltaC."
    )
    print(
        "Trigger: deterioration >=0.02 in >=2 of {10,15,29,40} event slices "
        "ON THE SAME AXIS."
    )
    print("One covariance hit + one mapping hit: DOES NOT satisfy trigger.")
    print()

    print("A5 closure:")
    print("  CLOSED means no A5 implementation or real-human evaluation in Paper 6.")
    print("  It may remain a preregistered future direction in Discussion.")
    print()

    print("Conservation strata:")
    print("  deterministic 5 x 10-module rank strata from 50 finite frozen scores.")
    print()

    print("Closed A5 search-space interpretation:")
    print("  2 x 2 x 2 = 8 combinations.")
    print(
        "  Failure means no success in this frozen search space; it does not "
        "falsify every conceivable realignment architecture."
    )
    print("  Post-result grid expansion inside Paper 6: FORBIDDEN.")
    print()

    print("=" * 120)
    print("05c1a A5 operational branch definitions: PASS")
    print("=" * 120)
    print()
    print("IMPORTANT: git-add/commit 05c1 and 05c1a BEFORE running 05d.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 120, file=sys.stderr)
        print("05c1a A5 operational branch definitions: FAIL", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print("=" * 120, file=sys.stderr)
        raise
