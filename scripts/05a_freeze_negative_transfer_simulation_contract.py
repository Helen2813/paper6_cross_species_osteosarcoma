#!/usr/bin/env python3
"""
Paper 6 - freeze negative-transfer simulation / AI-selection contract.

Purpose
-------
Freeze the known-truth simulation environment that will determine whether the
closed Paper-6 transfer candidates can safely borrow cross-species information
under target event scarcity.

This contract is frozen AFTER the sacrificial GSE16091 premise test, therefore
it explicitly forbids using GSE16091 numerical performance to choose simulation
parameters, architecture hyperparameters, or selection thresholds.

The event grid and scientific stress axes below come from the already-planned
Paper-6 design:
    target events = 5, 10, 15, 20, 29, 40
    source/target effect concordance
    covariance/domain shift
    censoring
    mapping error
    source-prior correctness
    transferable-module fraction

Closed model registry (from 02i)
--------------------------------
A0 small Hallmark survival network from scratch
A1 frozen DOG2 module encoder + target head
A2 DOG2 module encoder + low-rank residual target adapter
A3 evolution-conditioned module-resolved selective borrowing
A4 unrestricted/full neural fine-tuning

Required classical anchors
--------------------------
B0 target-only ridge Cox
B4 residual/Trans-Cox-style classical transfer

Primary method-selection principle
----------------------------------
Simulation is the architecture-selection environment.
GSE16091 is NOT an architecture leaderboard.

The proposed method must improve negative-transfer protection, recover known
borrowable modules, and remain competitive in prediction/calibration.
If A2 and A3 are practically tied, select the simpler candidate.

No model fitting.
No outcome/expression values read.
No network access.
No CLI arguments.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


SCRIPT_VERSION = "05a-freeze-negative-transfer-simulation-contract-v1-no-cli"

ROOT = Path(__file__).resolve().parents[1]

I_DIR = ROOT / "results" / "revised_design" / "02i"
I_CONTRACT = I_DIR / "revised_model_benchmark_source_gate_contract.json"
I_SUMMARY = I_DIR / "summary.json"

C4_DIR = ROOT / "results" / "human_premise" / "04c"
C4_RESULT = C4_DIR / "classical_premise_test_results.json"
C4_SUMMARY = C4_DIR / "summary.json"
C4_PRIMARY = C4_DIR / "primary_premise_result.json"

OUT_DIR = ROOT / "results" / "simulation_contract" / "05a"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCENARIOS = OUT_DIR / "simulation_scenario_registry.tsv"
MODEL_REGISTRY = OUT_DIR / "closed_model_registry.tsv"
SELECTION_RULES = OUT_DIR / "architecture_selection_rules.tsv"
CONTRACT_JSON = OUT_DIR / "negative_transfer_simulation_contract.json"
SUMMARY_JSON = OUT_DIR / "summary.json"

TARGET_EVENT_GRID = [5, 10, 15, 20, 29, 40]

# Core known-truth transfer regimes.
TRANSFER_REGIMES = [
    {
        "regime": "R0_FULLY_TRANSPORTABLE",
        "transferable_fraction": 1.00,
        "effect_concordance": 0.90,
        "nontransferable_sign_flip_fraction": 0.00,
    },
    {
        "regime": "R1_MOSTLY_TRANSPORTABLE",
        "transferable_fraction": 0.75,
        "effect_concordance": 0.75,
        "nontransferable_sign_flip_fraction": 0.00,
    },
    {
        "regime": "R2_PARTIALLY_TRANSPORTABLE",
        "transferable_fraction": 0.50,
        "effect_concordance": 0.50,
        "nontransferable_sign_flip_fraction": 0.10,
    },
    {
        "regime": "R3_WEAKLY_TRANSPORTABLE",
        "transferable_fraction": 0.25,
        "effect_concordance": 0.25,
        "nontransferable_sign_flip_fraction": 0.25,
    },
    {
        "regime": "R4_NONTRANSPORTABLE",
        "transferable_fraction": 0.00,
        "effect_concordance": 0.00,
        "nontransferable_sign_flip_fraction": 0.50,
    },
    {
        "regime": "R5_MISLEADING_SOURCE",
        "transferable_fraction": 0.00,
        "effect_concordance": -0.50,
        "nontransferable_sign_flip_fraction": 1.00,
    },
]

SHIFT_LEVELS = [
    ("S0_NONE", 0.00),
    ("S1_MODERATE", 0.25),
    ("S2_SEVERE", 0.50),
]

CENSORING_LEVELS = [
    ("C0_LOW", 0.20),
    ("C1_MODERATE", 0.40),
    ("C2_HIGH", 0.60),
]

MAPPING_ERROR_LEVELS = [
    ("M0_NONE", 0.00),
    ("M1_MODERATE", 0.10),
    ("M2_SEVERE", 0.30),
]

PRIOR_STATES = [
    "P0_CORRECT",
    "P1_UNINFORMATIVE",
    "P2_MISLEADING",
]

N_MODULES = 50
N_CAUSAL_MODULES = 10

SOURCE_N = 186
SOURCE_EVENT_TARGET = 124

SIMULATION_REPLICATES_CORE = 200
SIMULATION_REPLICATES_STRESS = 100

BASE_SEED = 20260831

PRACTICAL_TIE_DELTA_C = 0.01
PRACTICAL_TIE_DELTA_IBS = 0.01

MAX_NEGATIVE_TRANSFER_RATE_PRIMARY = 0.10
MAX_CATASTROPHIC_NEGATIVE_TRANSFER_RATE = 0.05

MIN_TRUE_BORROWABLE_MODULE_AUROC = 0.70
MAX_FALSE_BORROW_RATE_MISLEADING = 0.20


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


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


def require_file(path: Path) -> Path:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Required frozen artifact missing: {path}")
    return path


def main() -> None:
    started = now_utc()

    print("=" * 120)
    print("Paper 6 - freeze negative-transfer simulation / AI-selection contract")
    print("=" * 120)
    print(f"Script version: {SCRIPT_VERSION}")
    print()
    print("Safety / scope:")
    print("  DOG2 outcome/expression values read: NO")
    print("  GSE16091 outcome/expression values read: NO")
    print("  TARGET/GSE21257/GSE39055 outcomes read: NO")
    print("  Model fitting: NO")
    print("  Network access: NO")
    print("  GSE16091 numerical results used to choose simulation parameters: NO")
    print()

    for path in [
        I_CONTRACT,
        I_SUMMARY,
        C4_RESULT,
        C4_SUMMARY,
        C4_PRIMARY,
    ]:
        require_file(path)

    i_contract = read_json(I_CONTRACT)
    c4_result = read_json(C4_RESULT)
    c4_summary = read_json(C4_SUMMARY)

    if clean(i_contract.get("status")) != "PASS":
        raise RuntimeError("02i contract is not PASS.")
    if clean(c4_result.get("status")) != "PASS":
        raise RuntimeError("04c final result is not PASS.")
    if clean(c4_summary.get("status")) != "PASS":
        raise RuntimeError("04c summary is not PASS.")

    if clean(c4_result.get("scientific_status")) not in {
        "SUPPORTS_CANINE_ADDED_VALUE",
        "INCONCLUSIVE_NEUTRAL",
        "ARGUES_AGAINST_CANINE_ADDED_VALUE",
    }:
        raise RuntimeError("04c premise state is not recognized.")

    # Verify closed 02i AI registry instead of silently inventing a new list.
    ai_registry = i_contract.get("ai_candidate_registry") or []
    ai_by_id = {
        clean(row.get("candidate_id")): row
        for row in ai_registry
    }

    expected_ai = {"A0", "A1", "A2", "A3", "A4"}
    if set(ai_by_id) != expected_ai:
        raise RuntimeError(
            f"02i closed AI registry changed: {sorted(ai_by_id)}"
        )

    benchmark_registry = i_contract.get("classical_benchmark_registry") or []
    benchmark_by_id = {
        clean(row.get("benchmark_id")): row
        for row in benchmark_registry
    }

    for required_id in ["B0", "B4"]:
        if required_id not in benchmark_by_id:
            raise RuntimeError(
                f"02i classical benchmark registry lacks {required_id}."
            )

    # ------------------------------------------------------------------
    # Scenario registry.
    #
    # We intentionally avoid a gigantic full Cartesian product.
    # Core grid:
    #   all target-event counts x all transfer regimes
    #   with moderate covariance shift, moderate censoring, no mapping error,
    #   correct source prior.
    #
    # Stress slices perturb one nuisance axis at a time around the central
    # partial-transport regime and around the misleading-source regime.
    # ------------------------------------------------------------------

    scenario_rows: List[Dict[str, Any]] = []

    scenario_id = 0

    for events in TARGET_EVENT_GRID:
        for regime in TRANSFER_REGIMES:
            scenario_rows.append(
                {
                    "scenario_id": f"CORE_{scenario_id:04d}",
                    "family": "CORE_PHASE_DIAGRAM",
                    "target_events": events,
                    "transfer_regime": regime["regime"],
                    "transferable_fraction": regime["transferable_fraction"],
                    "effect_concordance": regime["effect_concordance"],
                    "sign_flip_fraction": regime[
                        "nontransferable_sign_flip_fraction"
                    ],
                    "covariance_shift": "S1_MODERATE",
                    "covariance_shift_strength": 0.25,
                    "censoring": "C1_MODERATE",
                    "target_censor_fraction": 0.40,
                    "mapping_error": "M0_NONE",
                    "mapping_error_fraction": 0.00,
                    "source_prior_state": "P0_CORRECT",
                    "replicates": SIMULATION_REPLICATES_CORE,
                    "selection_environment": True,
                }
            )
            scenario_id += 1

    # Stress slices at representative event counts.
    stress_event_grid = [10, 15, 29, 40]
    stress_regimes = [
        next(r for r in TRANSFER_REGIMES if r["regime"] == "R2_PARTIALLY_TRANSPORTABLE"),
        next(r for r in TRANSFER_REGIMES if r["regime"] == "R5_MISLEADING_SOURCE"),
    ]

    for events in stress_event_grid:
        for regime in stress_regimes:
            for shift_name, shift_strength in SHIFT_LEVELS:
                for censor_name, censor_fraction in CENSORING_LEVELS:
                    scenario_rows.append(
                        {
                            "scenario_id": f"STRESS_{scenario_id:04d}",
                            "family": "SHIFT_CENSOR_STRESS",
                            "target_events": events,
                            "transfer_regime": regime["regime"],
                            "transferable_fraction": regime["transferable_fraction"],
                            "effect_concordance": regime["effect_concordance"],
                            "sign_flip_fraction": regime[
                                "nontransferable_sign_flip_fraction"
                            ],
                            "covariance_shift": shift_name,
                            "covariance_shift_strength": shift_strength,
                            "censoring": censor_name,
                            "target_censor_fraction": censor_fraction,
                            "mapping_error": "M0_NONE",
                            "mapping_error_fraction": 0.00,
                            "source_prior_state": "P0_CORRECT",
                            "replicates": SIMULATION_REPLICATES_STRESS,
                            "selection_environment": True,
                        }
                    )
                    scenario_id += 1

            for map_name, map_fraction in MAPPING_ERROR_LEVELS:
                for prior_state in PRIOR_STATES:
                    scenario_rows.append(
                        {
                            "scenario_id": f"STRESS_{scenario_id:04d}",
                            "family": "MAPPING_PRIOR_STRESS",
                            "target_events": events,
                            "transfer_regime": regime["regime"],
                            "transferable_fraction": regime["transferable_fraction"],
                            "effect_concordance": regime["effect_concordance"],
                            "sign_flip_fraction": regime[
                                "nontransferable_sign_flip_fraction"
                            ],
                            "covariance_shift": "S1_MODERATE",
                            "covariance_shift_strength": 0.25,
                            "censoring": "C1_MODERATE",
                            "target_censor_fraction": 0.40,
                            "mapping_error": map_name,
                            "mapping_error_fraction": map_fraction,
                            "source_prior_state": prior_state,
                            "replicates": SIMULATION_REPLICATES_STRESS,
                            "selection_environment": True,
                        }
                    )
                    scenario_id += 1

    scenarios = pd.DataFrame(scenario_rows)
    scenarios.to_csv(SCENARIOS, sep="\t", index=False)

    # ------------------------------------------------------------------
    # Frozen model registry.
    # ------------------------------------------------------------------

    model_rows = [
        {
            "model_id": "B0",
            "model": "TARGET_ONLY_RIDGE_COX",
            "category": "CLASSICAL_ANCHOR",
            "eligible_for_final_AI_selection": False,
            "required": True,
        },
        {
            "model_id": "B4",
            "model": "CLASSICAL_RESIDUAL_TRANSFER_COX",
            "category": "CLASSICAL_TRANSFER_ANCHOR",
            "eligible_for_final_AI_selection": False,
            "required": True,
        },
    ]

    for candidate_id in ["A0", "A1", "A2", "A3", "A4"]:
        row = ai_by_id[candidate_id]
        model_rows.append(
            {
                "model_id": candidate_id,
                "model": clean(row.get("model")),
                "category": "AI",
                "eligible_for_final_AI_selection": bool(
                    row.get("eligible_for_primary_selection")
                ),
                "required": True,
            }
        )

    models = pd.DataFrame(model_rows)
    models.to_csv(MODEL_REGISTRY, sep="\t", index=False)

    # ------------------------------------------------------------------
    # Architecture-selection rules.
    # ------------------------------------------------------------------

    selection_rows = [
        {
            "priority": 1,
            "criterion": "NEGATIVE_TRANSFER_RATE",
            "rule": (
                "For each scenario, negative transfer means model Uno C is "
                ">=0.02 below B0 target-only ridge Cox. Primary selectable "
                "candidate should have aggregate negative-transfer rate <=0.10."
            ),
        },
        {
            "priority": 2,
            "criterion": "CATASTROPHIC_NEGATIVE_TRANSFER",
            "rule": (
                "Catastrophic negative transfer means Uno C is >=0.05 below B0. "
                "Rate must be <=0.05 in misleading-source and severe-shift regimes."
            ),
        },
        {
            "priority": 3,
            "criterion": "TRUE_BORROWABLE_MODULE_RECOVERY",
            "rule": (
                "For models with a module borrowing gate, AUROC for identifying "
                "known truly transferable modules should be >=0.70 across "
                "partially transportable regimes."
            ),
        },
        {
            "priority": 4,
            "criterion": "FALSE_BORROWING_UNDER_MISLEADING_SOURCE",
            "rule": (
                "Mean fraction of nontransferable modules receiving a borrow "
                "weight >=0.5 should be <=0.20 in misleading-source regimes."
            ),
        },
        {
            "priority": 5,
            "criterion": "PREDICTIVE_PERFORMANCE",
            "rule": (
                "Among candidates satisfying protection criteria, compare Uno C, "
                "IBS, calibration slope/intercept, and event-scaled learning curves."
            ),
        },
        {
            "priority": 6,
            "criterion": "PRACTICAL_TIE",
            "rule": (
                "If A2 and A3 differ by <0.01 mean Uno C and <0.01 mean IBS "
                "while both satisfy protection criteria, select the simpler model A2."
            ),
        },
        {
            "priority": 7,
            "criterion": "EMPIRICAL_GSE16091_FIREWALL",
            "rule": (
                "GSE16091 performance cannot choose architecture, hyperparameters, "
                "gate thresholds, simulation scenarios, or tie-breaking. It is an "
                "empirical stress result reported only after simulation-based selection."
            ),
        },
    ]

    pd.DataFrame(selection_rows).to_csv(
        SELECTION_RULES,
        sep="\t",
        index=False,
    )

    contract = {
        "script_version": SCRIPT_VERSION,
        "status": "PASS",
        "scientific_status": (
            "PASS_NEGATIVE_TRANSFER_SIMULATION_AND_AI_SELECTION_CONTRACT_FROZEN"
        ),
        "created_utc": now_utc(),
        "simulation_objective": (
            "Determine when cross-species borrowing helps, becomes neutral, or "
            "causes negative transfer under target event scarcity, and select "
            "between the already-closed A2/A3 candidates using known truth."
        ),
        "generative_structure": {
            "n_modules": N_MODULES,
            "n_causal_modules": N_CAUSAL_MODULES,
            "source_n": SOURCE_N,
            "source_event_target": SOURCE_EVENT_TARGET,
            "target_event_grid": TARGET_EVENT_GRID,
            "module_covariance": (
                "Block-correlated latent Gaussian Hallmark-module factors; "
                "source/target covariance shift is imposed independently of "
                "outcome-effect transport."
            ),
            "survival_model": (
                "Proportional-hazards data generation with continuous event times; "
                "independent censoring calibrated to scenario target censor fraction."
            ),
            "transport_truth": (
                "Each causal module is labeled transferable/nontransferable. "
                "Transferable target effects are correlated with source effects; "
                "nontransferable effects may attenuate, disappear, or flip sign "
                "according to the frozen regime."
            ),
            "mapping_error": (
                "Outcome-independent permutation/dropout of the prespecified "
                "fraction of module-to-source correspondences."
            ),
        },
        "core_transfer_regimes": TRANSFER_REGIMES,
        "stress_axes": {
            "covariance_shift": SHIFT_LEVELS,
            "censoring": CENSORING_LEVELS,
            "mapping_error": MAPPING_ERROR_LEVELS,
            "source_prior_state": PRIOR_STATES,
        },
        "scenario_registry_sha256": sha256_file(SCENARIOS),
        "n_scenarios": int(len(scenarios)),
        "replicates": {
            "core": SIMULATION_REPLICATES_CORE,
            "stress": SIMULATION_REPLICATES_STRESS,
            "base_seed": BASE_SEED,
        },
        "closed_models": model_rows,
        "selection_thresholds": {
            "negative_transfer_definition_delta_c": -0.02,
            "catastrophic_negative_transfer_definition_delta_c": -0.05,
            "max_primary_negative_transfer_rate": MAX_NEGATIVE_TRANSFER_RATE_PRIMARY,
            "max_catastrophic_negative_transfer_rate": (
                MAX_CATASTROPHIC_NEGATIVE_TRANSFER_RATE
            ),
            "min_true_borrowable_module_AUROC": (
                MIN_TRUE_BORROWABLE_MODULE_AUROC
            ),
            "max_false_borrow_rate_misleading": (
                MAX_FALSE_BORROW_RATE_MISLEADING
            ),
            "practical_tie_delta_c": PRACTICAL_TIE_DELTA_C,
            "practical_tie_delta_IBS": PRACTICAL_TIE_DELTA_IBS,
        },
        "architecture_selection": {
            "eligible_candidates": ["A2", "A3"],
            "simulation_first": True,
            "GSE16091_is_architecture_leaderboard": False,
            "tie_rule": "A2 if practical tie",
            "selection_rules_sha256": sha256_file(SELECTION_RULES),
        },
        "required_metrics": [
            "Uno C",
            "integrated Brier score",
            "calibration slope",
            "calibration intercept",
            "delta Uno C versus B0",
            "negative-transfer indicator",
            "catastrophic-negative-transfer indicator",
            "true borrowable-module AUROC",
            "false borrow rate",
            "mean borrow weight by true module class",
            "event-scaled learning curve",
        ],
        "ablation_requirements": [
            "A3 without evolutionary prior",
            "A3 without target-overridable gate",
            "A3 with permuted source-module prior",
            "A3 with mapping permutation",
            "A3 source-outcome permutation",
        ],
        "04c_empirical_result_firewall": {
            "04c_result_sha256": sha256_file(C4_RESULT),
            "04c_summary_sha256": sha256_file(C4_SUMMARY),
            "04c_premise_state_recorded_for_provenance_only": clean(
                c4_result.get("scientific_status")
            ),
            "04c_numeric_model_performance_read_in_05a": False,
            "04c_numeric_model_performance_used_to_set_parameters": False,
            "event_grid_includes_15_by_prior_plan_not_by_04c_tuning": True,
            "AI_architecture_tuning_from_GSE16091": False,
        },
        "reserved_human_outcome_firewall": {
            "TARGET_OS": "CLOSED",
            "GSE21257": "CLOSED",
            "GSE39055": "CLOSED",
        },
        "execution_order": [
            "05b materialize known-truth simulation datasets/shards",
            "05c run closed classical/AI model matrix and ablations",
            "05d aggregate phase diagram and apply frozen architecture-selection rule",
            "freeze selected AI method/hyperparameters",
            "only then open TARGET-OS primary human transfer gate",
        ],
        "runtime": {
            "simulation_generation": "CPU appropriate",
            "AI model fitting": (
                "GPU preferred when available and scientifically useful; "
                "do not disturb another active long-running GPU project."
            ),
        },
    }

    write_json(CONTRACT_JSON, contract)

    summary = {
        "script_version": SCRIPT_VERSION,
        "status": "PASS",
        "scientific_status": contract["scientific_status"],
        "run_started_utc": started,
        "run_finished_utc": now_utc(),
        "target_event_grid": TARGET_EVENT_GRID,
        "n_scenarios": int(len(scenarios)),
        "core_replicates": SIMULATION_REPLICATES_CORE,
        "stress_replicates": SIMULATION_REPLICATES_STRESS,
        "closed_models": [row["model_id"] for row in model_rows],
        "selectable_candidates": ["A2", "A3"],
        "GSE16091_used_for_AI_tuning": False,
        "reserved_human_outcomes_read": False,
        "final_artifact_hashes": {
            "simulation_scenario_registry_tsv": sha256_file(SCENARIOS),
            "closed_model_registry_tsv": sha256_file(MODEL_REGISTRY),
            "architecture_selection_rules_tsv": sha256_file(SELECTION_RULES),
            "negative_transfer_simulation_contract_json": sha256_file(
                CONTRACT_JSON
            ),
        },
        "next": "05b materialize known-truth simulation datasets/shards",
    }

    write_json(SUMMARY_JSON, summary)

    print("-" * 120)
    print("Frozen simulation design")
    print("-" * 120)
    print(f"Target event grid: {TARGET_EVENT_GRID}")
    print(f"Core transfer regimes: {len(TRANSFER_REGIMES)}")
    print(f"Total scenarios: {len(scenarios)}")
    print(f"Core replicates/scenario: {SIMULATION_REPLICATES_CORE}")
    print(f"Stress replicates/scenario: {SIMULATION_REPLICATES_STRESS}")
    print()

    print("Closed model matrix:")
    for row in model_rows:
        print(
            f"  {row['model_id']}: {row['model']} "
            f"[selectable={row['eligible_for_final_AI_selection']}]"
        )

    print()
    print("Primary selectable AI candidates: A2 vs A3")
    print("Tie rule: choose simpler A2 if practical tie")
    print("GSE16091 numerical performance used for architecture selection: NO")
    print()

    print("=" * 120)
    print("05a NEGATIVE-TRANSFER SIMULATION CONTRACT SUMMARY")
    print("=" * 120)
    print("Human reserved outcomes read: NO")
    print("Model fitting: NO")
    print("Simulation-first architecture selection: YES")
    print("GSE16091 is architecture leaderboard: NO")
    print()
    print("Next: 05b materialize known-truth simulation datasets/shards.")
    print("=" * 120)
    print("05a negative-transfer simulation / AI-selection contract: PASS")
    print("=" * 120)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 120, file=sys.stderr)
        print(
            "05a negative-transfer simulation / AI-selection contract: FAIL",
            file=sys.stderr,
        )
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print("=" * 120, file=sys.stderr)
        raise
