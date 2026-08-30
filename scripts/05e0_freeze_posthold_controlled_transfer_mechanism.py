#!/usr/bin/env python3
"""
Paper 6 - freeze post-HOLD controlled transfer-mechanism experiment.

POST-RESULT chronology:
- 05d = HOLD_NO_SAFE_SELECTABLE_ARCHITECTURE
- 05d0c = STOP_A6_FOR_PAPER6
- 05d0g diagnostics complete

This stage does NOT rescue A2/A3 and does NOT reopen A6.
It freezes a NEW-SEED, focused, within-DGP controlled mechanism experiment
before any new model fitting.

Questions:
H1 HEAD RETARGETING:
    Does re-estimating only the target head on a frozen source encoder rescue
    sign-reversed source prognostic semantics?

H2 TARGET-HEAD CONSTRAINT:
    Does removing A1's L2-to-source-head constraint worsen robustness?

H3 SOFT-GATE LEAKAGE:
    Does prediction-only hardening of A3's learned soft gates improve R5 safety?

H4 GATE LIMITATION:
    Does a TRUE binary transportability gate rescue the same A3 source/residual
    construction?

No human outcomes.
No model fitting.
No GPU.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_VERSION = "05e0-freeze-posthold-controlled-transfer-mechanism-v1-no-cli"
CONTRACT_VERSION = "paper6-posthold-controlled-mechanism-v1"

ROOT = Path(__file__).resolve().parents[1]

A_DIR = ROOT / "results" / "simulation_contract" / "05a"
A_CONTRACT = A_DIR / "negative_transfer_simulation_contract.json"

B_DIR = ROOT / "simulations" / "05b"
B_CONTRACT = B_DIR / "generator_contract.json"
B_MANIFEST = B_DIR / "scenario_recipe_manifest.tsv"

C_DIR = ROOT / "results" / "simulation_model_matrix" / "05c"
C_IMPL = C_DIR / "model_implementation_contract.json"
C_SUMMARY = C_DIR / "summary.json"

D_DIR = ROOT / "results" / "simulation_phase_diagram" / "05d"
D_SUMMARY = D_DIR / "summary.json"

D0C_DIR = ROOT / "results" / "simulation_phase_diagram_diagnostics" / "05d0c"
D0C_SUMMARY = D0C_DIR / "summary.json"

D0G_DIR = ROOT / "results" / "simulation_phase_diagram_diagnostics" / "05d0g"
D0G_SUMMARY = D0G_DIR / "summary.json"

OUT_DIR = ROOT / "results" / "posthold_controlled_mechanism" / "05e0"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONTRACT_JSON = OUT_DIR / "controlled_mechanism_contract.json"
SCENARIO_REGISTRY = OUT_DIR / "controlled_mechanism_scenario_registry.tsv"
MODEL_REGISTRY = OUT_DIR / "controlled_mechanism_model_registry.tsv"
CONTRAST_REGISTRY = OUT_DIR / "controlled_mechanism_contrast_registry.tsv"
SUMMARY_JSON = OUT_DIR / "summary.json"

EXPECTED_05C_STATUS = "PASS_CLOSED_SYNTHETIC_TRANSFER_MODEL_MATRIX_COMPLETE"
EXPECTED_05D_STATUS = "HOLD_NO_SAFE_SELECTABLE_ARCHITECTURE"
EXPECTED_05D0C_STATUS = "STOP_A6_FOR_PAPER6"
EXPECTED_05D0G_STATUS = (
    "PASS_POSTHOC_UTILITY_COMPARATOR_WITHINSCENARIO_PROVENANCE_AUDIT_COMPLETE"
)

EVENT_BUDGETS = [10, 29, 40]
PRIMARY_EVENT_BUDGET = 29

REGIMES = [
    ("R0_FULLY_TRANSPORTABLE", 1.00, 0.90, 0.00),
    ("R2_PARTIALLY_TRANSPORTABLE", 0.50, 0.50, 0.10),
    ("R4_NONTRANSPORTABLE", 0.00, 0.00, 0.50),
    ("R5_MISLEADING_SOURCE", 0.00, -0.50, 1.00),
]

REPLICATES_PER_CELL = 500
TARGET_TEST_N = 500
NEW_SEED_NAMESPACE_BASE = 930_000_000

# Focus the mechanism experiment: remove other stress axes.
COVARIANCE_SHIFT = ("S0_NONE", 0.00)
CENSORING = ("C1_MODERATE", 0.40)
MAPPING_ERROR = ("M0_NONE", 0.00)
PRIOR_STATE = "P0_CORRECT"

# Exact 05c A1/A3 values to replay.
A1_EPOCHS = 80
A1_LR = 0.020
A1_L2_TO_SOURCE = 1e-2

A3_EPOCHS = 120
A3_LR = 0.020
A3_RESIDUAL_L2 = 0.01
A3_GATE_THRESHOLD = 0.50

# Prospective practical scales for THIS experiment only.
PRACTICAL_DELTA_C = 0.02
PRACTICAL_CAT_REDUCTION = 0.10

BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 930_777_001


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean(x: Any) -> str:
    return "" if x is None else str(x).strip()


def require_file(path: Path) -> None:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Required artifact missing: {path}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")


def main() -> None:
    started = now_utc()

    print("=" * 120)
    print("Paper 6 - freeze post-HOLD controlled transfer-mechanism experiment")
    print("=" * 120)
    print(f"Script version: {SCRIPT_VERSION}")
    print()
    print("Scope:")
    print("  New-seed controlled mechanism experiment: YES")
    print("  Original 05d HOLD changed: NO")
    print("  A6 reopened: NO")
    print("  Model fitting in 05e0: NO")
    print("  Human outcomes read: NO")
    print("  GPU execution: NO")
    print()

    for p in [
        A_CONTRACT, B_CONTRACT, B_MANIFEST,
        C_IMPL, C_SUMMARY, D_SUMMARY, D0C_SUMMARY, D0G_SUMMARY
    ]:
        require_file(p)

    c_summary = read_json(C_SUMMARY)
    d_summary = read_json(D_SUMMARY)
    d0c_summary = read_json(D0C_SUMMARY)
    d0g_summary = read_json(D0G_SUMMARY)
    c_impl = read_json(C_IMPL)

    if clean(c_summary.get("scientific_status")) != EXPECTED_05C_STATUS:
        raise RuntimeError("05c scientific state mismatch.")
    if clean(d_summary.get("scientific_status")) != EXPECTED_05D_STATUS:
        raise RuntimeError("05d is not in frozen HOLD state.")
    if clean(d0c_summary.get("scientific_status")) != EXPECTED_05D0C_STATUS:
        raise RuntimeError("05d0c is not STOP_A6_FOR_PAPER6.")
    if clean(d0g_summary.get("scientific_status")) != EXPECTED_05D0G_STATUS:
        raise RuntimeError("05d0g is not in expected completed PASS state.")

    neural = c_impl.get("neural") or {}
    expected_impl = {
        "A1_epochs": A1_EPOCHS,
        "A1_lr": A1_LR,
        "A1_head_l2_to_source": A1_L2_TO_SOURCE,
        "A3_epochs": A3_EPOCHS,
        "A3_lr": A3_LR,
        "A3_residual_l2": A3_RESIDUAL_L2,
    }
    for key, value in expected_impl.items():
        if key not in neural or not np.isclose(
            float(neural[key]), float(value), rtol=0, atol=1e-12
        ):
            raise RuntimeError(f"05c implementation identity mismatch at {key}.")

    rows = []
    idx = 0
    for events in EVENT_BUDGETS:
        for regime, tf, concordance, flip in REGIMES:
            seed_start = NEW_SEED_NAMESPACE_BASE + idx * 10_000
            rows.append({
                "cell_id": f"MECH_{idx:03d}",
                "cell_index": idx,
                "target_events": events,
                "transfer_regime": regime,
                "transferable_fraction": tf,
                "effect_concordance": concordance,
                "nontransferable_sign_flip_fraction": flip,
                "covariance_shift": COVARIANCE_SHIFT[0],
                "covariance_shift_value": COVARIANCE_SHIFT[1],
                "censoring": CENSORING[0],
                "censoring_fraction": CENSORING[1],
                "mapping_error": MAPPING_ERROR[0],
                "mapping_error_fraction": MAPPING_ERROR[1],
                "source_prior_state": PRIOR_STATE,
                "replicates": REPLICATES_PER_CELL,
                "target_test_n": TARGET_TEST_N,
                "seed_start": seed_start,
                "seed_end": seed_start + REPLICATES_PER_CELL - 1,
                "primary_event_budget": events == PRIMARY_EVENT_BUDGET,
            })
            idx += 1

    scenarios = pd.DataFrame(rows)

    # Prove zero overlap with old 05b seeds before freeze.
    old_manifest = pd.read_csv(B_MANIFEST, sep="\t")
    old_seeds = set()
    for row in old_manifest.itertuples(index=False):
        recipe = ROOT / str(row.recipe_path)
        require_file(recipe)
        with np.load(recipe, allow_pickle=False) as data:
            old_seeds.update(
                np.asarray(data["replicate_seed"], dtype=np.int64).tolist()
            )

    new_seeds = set()
    for row in scenarios.itertuples(index=False):
        new_seeds.update(range(int(row.seed_start), int(row.seed_end) + 1))

    overlap = old_seeds.intersection(new_seeds)
    if overlap:
        raise RuntimeError(
            f"New seed namespace overlaps original 05b seeds: {sorted(overlap)[:10]}"
        )

    scenarios.to_csv(SCENARIO_REGISTRY, sep="\t", index=False)

    models = pd.DataFrame([
        {
            "model": "M0_B0",
            "role": "reference",
            "definition": "Exact 05c B0 target-only ridge Cox replay.",
            "oracle": False,
        },
        {
            "model": "M1_SOURCE_NETWORK_ZERO_SHOT",
            "role": "head-adaptation control",
            "definition": (
                "Frozen source MLP encoder+source head applied to aligned target; "
                "no target update."
            ),
            "oracle": False,
        },
        {
            "model": "M2_A1_SOURCE_CENTERED_HEAD",
            "role": "head-retargeting intervention",
            "definition": (
                "Exact 05c A1: frozen source encoder; source-initialized target "
                "head updated with epochs=80, lr=.02, L2-to-source=.01."
            ),
            "oracle": False,
        },
        {
            "model": "M3_A1_FREE_HEAD",
            "role": "target-flexibility intervention",
            "definition": (
                "Same M2 encoder, initialization, epochs and lr; only "
                "L2-to-source-head is set to zero."
            ),
            "oracle": False,
        },
        {
            "model": "M4_A3_SOFT_LEARNED",
            "role": "selective-borrowing reference",
            "definition": "Exact 05c A3 replay.",
            "oracle": False,
        },
        {
            "model": "M5_A3_HARDENED_PREDICTION",
            "role": "continuous-leakage intervention",
            "definition": (
                "Fit M4 exactly; at prediction only replace g by I(g>=.5), "
                "reusing M4 residual coefficients without refit."
            ),
            "oracle": False,
        },
        {
            "model": "M6_A3_ORACLE_HARD_GATE",
            "role": "truth-gate upper bound",
            "definition": (
                "Use true transportable_mask as fixed 0/1 gate; no gate learning; "
                "train only A3 complementary residual with exact A3 lr/epochs/L2."
            ),
            "oracle": True,
        },
    ])
    models.to_csv(MODEL_REGISTRY, sep="\t", index=False)

    contrasts = pd.DataFrame([
        {
            "contrast_id": "H1_HEAD_RETARGET_R5",
            "event_budget": 29,
            "transfer_regime": "R5_MISLEADING_SOURCE",
            "treatment": "M2_A1_SOURCE_CENTERED_HEAD",
            "control": "M1_SOURCE_NETWORK_ZERO_SHOT",
            "primary_estimand": "paired mean UnoC difference",
            "supportive_threshold": PRACTICAL_DELTA_C,
            "extra_requirement": "same nonnegative direction at 10 and 40 events",
        },
        {
            "contrast_id": "H2_HEAD_CONSTRAINT_R5",
            "event_budget": 29,
            "transfer_regime": "R5_MISLEADING_SOURCE",
            "treatment": "M2_A1_SOURCE_CENTERED_HEAD",
            "control": "M3_A1_FREE_HEAD",
            "primary_estimand": "paired mean UnoC difference",
            "supportive_threshold": 0.01,
            "extra_requirement": "interpret with IBS and test-risk SD",
        },
        {
            "contrast_id": "H3_SOFT_LEAKAGE_R5",
            "event_budget": 29,
            "transfer_regime": "R5_MISLEADING_SOURCE",
            "treatment": "M5_A3_HARDENED_PREDICTION",
            "control": "M4_A3_SOFT_LEARNED",
            "primary_estimand": "paired mean UnoC difference",
            "supportive_threshold": PRACTICAL_DELTA_C,
            "extra_requirement": f"catastrophic-rate reduction >= {PRACTICAL_CAT_REDUCTION:.2f}",
        },
        {
            "contrast_id": "H4_ORACLE_GATE_R5",
            "event_budget": 29,
            "transfer_regime": "R5_MISLEADING_SOURCE",
            "treatment": "M6_A3_ORACLE_HARD_GATE",
            "control": "M4_A3_SOFT_LEARNED",
            "primary_estimand": "paired mean UnoC difference",
            "supportive_threshold": PRACTICAL_DELTA_C,
            "extra_requirement": f"catastrophic-rate reduction >= {PRACTICAL_CAT_REDUCTION:.2f}",
        },
    ])
    contrasts.to_csv(CONTRAST_REGISTRY, sep="\t", index=False)

    contract = {
        "script_version": SCRIPT_VERSION,
        "contract_version": CONTRACT_VERSION,
        "status": "PASS_FROZEN_BEFORE_NEW_CONTROLLED_SIMULATION_FITS",
        "created_utc": now_utc(),
        "scientific_role": (
            "post-HOLD confirmatory within-DGP controlled mechanism experiment; "
            "not architecture rescue, not model selection, not external DGP validation"
        ),
        "source_artifact_hashes": {
            "05a_contract": sha256_file(A_CONTRACT),
            "05b_generator_contract": sha256_file(B_CONTRACT),
            "05c_implementation_contract": sha256_file(C_IMPL),
            "05d_summary": sha256_file(D_SUMMARY),
            "05d0c_summary": sha256_file(D0C_SUMMARY),
            "05d0g_summary": sha256_file(D0G_SUMMARY),
        },
        "design": {
            "event_budgets": EVENT_BUDGETS,
            "primary_event_budget": PRIMARY_EVENT_BUDGET,
            "regimes": [r[0] for r in REGIMES],
            "fixed_covariance_shift": COVARIANCE_SHIFT,
            "fixed_censoring": CENSORING,
            "fixed_mapping_error": MAPPING_ERROR,
            "fixed_prior_state": PRIOR_STATE,
            "replicates_per_cell": REPLICATES_PER_CELL,
            "n_cells": len(scenarios),
            "total_new_replicates": int(scenarios["replicates"].sum()),
            "target_test_n": TARGET_TEST_N,
            "old_new_seed_overlap": 0,
        },
        "models": models.to_dict(orient="records"),
        "primary_contrasts": contrasts.to_dict(orient="records"),
        "metrics": {
            "primary": "Uno_C",
            "secondary": [
                "IBS",
                "test_risk_SD",
                "test_risk_q99_q01",
                "negative_transfer_rate",
                "catastrophic_negative_transfer_rate",
            ],
            "negative_transfer_delta_c_vs_B0": -0.02,
            "catastrophic_delta_c_vs_B0": -0.05,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "guardrails": {
            "no_model_selection": True,
            "no_A6_reopening": True,
            "no_posthoc_hyperparameter_tuning": True,
            "no_grid_expansion_after_results": True,
            "oracle_gate_label_nonimplementable": True,
            "same_DGP_new_seed_confirmation_not_external_validation": True,
            "reserved_human_outcomes_remain_closed": True,
        },
        "next": (
            "05e1 materialize new-seed recipes; 05e2 run frozen controlled branches; "
            "05e3 evaluate paired contrasts."
        ),
    }
    write_json(CONTRACT_JSON, contract)

    summary = {
        "script_version": SCRIPT_VERSION,
        "status": "PASS",
        "scientific_status": "PASS_POSTHOLD_CONTROLLED_MECHANISM_EXPERIMENT_FROZEN",
        "run_started_utc": started,
        "run_finished_utc": now_utc(),
        "n_cells": len(scenarios),
        "replicates_per_cell": REPLICATES_PER_CELL,
        "total_new_replicates": int(scenarios["replicates"].sum()),
        "old_new_seed_overlap": 0,
        "model_fitting": False,
        "human_outcomes_read": False,
        "GPU_execution": False,
        "artifact_hashes": {
            "contract": sha256_file(CONTRACT_JSON),
            "scenario_registry": sha256_file(SCENARIO_REGISTRY),
            "model_registry": sha256_file(MODEL_REGISTRY),
            "contrast_registry": sha256_file(CONTRAST_REGISTRY),
        },
    }
    write_json(SUMMARY_JSON, summary)

    print("=" * 120)
    print("05e0 CONTROLLED MECHANISM EXPERIMENT FREEZE")
    print("=" * 120)
    print(f"Cells: {len(scenarios)}")
    print(f"Replicates per cell: {REPLICATES_PER_CELL}")
    print(f"Total new replicates: {int(scenarios['replicates'].sum()):,}")
    print(f"Old/new seed overlap: {len(overlap)}")
    print()
    print(models[["model", "role"]].to_string(index=False))
    print()
    print(contrasts[["contrast_id", "treatment", "control"]].to_string(index=False))
    print()
    print("Original 05d HOLD changed: NO")
    print("A6 reopened: NO")
    print("Human outcomes read: NO")
    print("Model fitting: NO")
    print(f"Contract SHA256: {sha256_file(CONTRACT_JSON)}")
    print("=" * 120)
    print("05e0: PASS_POSTHOLD_CONTROLLED_MECHANISM_EXPERIMENT_FROZEN")
    print("=" * 120)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 120, file=sys.stderr)
        print("05e0 controlled mechanism contract freeze: FAIL", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print("=" * 120, file=sys.stderr)
        raise
