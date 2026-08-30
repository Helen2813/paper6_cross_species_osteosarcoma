#!/usr/bin/env python3
"""
Paper 6 - prioritized post-05d diagnostic of frozen AI failure.

This is a FOLLOW-UP to the already-run 05d0 v1 structural diagnostic.

The frozen scientific result remains:
    05d = HOLD_NO_SAFE_SELECTABLE_ARCHITECTURE
    A5  = CLOSE_A5

This stage is POST-RESULT and EXPLANATORY ONLY. It may diagnose a provable
implementation defect, threshold feasibility, or genuine architecture failure.
It may NOT retune, change thresholds, select A2/A3, or open human outcomes.

Priority follow-up questions (scope frozen after v1, before oracle/risk-scale read)
-----------------------------------------------------------
1. Units:
   false-borrow is per MODULE while catastrophic negative transfer is per
   REPLICATE. Quantify how module-level errors compose within a replicate.

2. Comparators:
   replay B0/B4/A0/A1/A2/A3/A4 from frozen 05d metrics and add one clearly
   POST-HOC diagnostic linear zero-shot source score reconstructed from frozen
   05c source coefficients. This diagnostic B2-like score is NOT added to the
   frozen 05d selection registry.

3. Feasibility:
   evaluate the true data-generating target-risk oracle and simple
   regime-abstention oracles. This does not alter the 05d threshold; it asks
   whether the frozen 0.10 negative-transfer target was attainable in the
   simulated world.

4. Risk amplitude / IBS:
   quantify A2 vs A3 train/test risk-score scale and compare saved IBS to a
   no-predictor Cox baseline on a deterministic subset.

5. Composition:
   relate replicate-level catastrophic failure to the NUMBER of broad false
   borrowing decisions and, more importantly, the number of causal sign-flipped
   modules borrowed.

6. Manual metric replay:
   independently recompute Uno C for fixed scenario/replicate/model examples
   directly with sksurv and compare with frozen 05d checkpoints.

A5 follow-up interpretation policy
------------------------------------
The 05d A5=CLOSE result may be re-evaluated exactly once ONLY if this diagnostic
(or a direct follow-up) proves a deterministic implementation defect that:
  a) violates the already-frozen intended A3 implementation semantics, AND
  b) materially changes A3 predictions/metrics used by the A5 trigger.
The scenario grid, scientific architecture, hyperparameters, thresholds and A5
rule must remain unchanged.

A poor but correctly implemented objective, risk-scale pathology inherent to
the frozen architecture, or the already-fixed scenario_id aggregation bug do
NOT qualify. If no qualifying implementation defect is proved, A5=CLOSE is
final for Paper 6.

No model fitting.
No tuning.
No real data.
No GPU.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from sksurv.metrics import concordance_index_ipcw
    from sksurv.util import Surv
except ImportError as exc:
    raise ImportError("05d0 v2 requires scikit-survival.") from exc


SCRIPT_VERSION = "05d0a-followup-units-oracle-riskscale-diagnostic-v1-no-cli"

ROOT = Path(__file__).resolve().parents[1]

# Frozen scenario / generator assets.
A_DIR = ROOT / "results" / "simulation_contract" / "05a"
A_SCENARIOS = A_DIR / "simulation_scenario_registry.tsv"
A_CONTRACT = A_DIR / "negative_transfer_simulation_contract.json"

B_DIR = ROOT / "simulations" / "05b"
B_GENERATOR_CONTRACT = B_DIR / "generator_contract.json"

# Frozen 05c model outputs.
C_DIR = ROOT / "results" / "simulation_model_matrix" / "05c"
C_MANIFEST = C_DIR / "scenario_output_manifest.tsv"
C_SUMMARY = C_DIR / "summary.json"
C_IMPLEMENTATION = C_DIR / "model_implementation_contract.json"

# Already-run bounded v1 diagnostic. This follow-up is explicitly AFTER it.
D0_V1_DIR = ROOT / "results" / "simulation_phase_diagram_diagnostics" / "05d0"
D0_V1_SUMMARY = D0_V1_DIR / "summary.json"

# Frozen 05d result.
D_DIR = ROOT / "results" / "simulation_phase_diagram" / "05d"
D_METRIC_CONTRACT = D_DIR / "metric_implementation_contract.json"
D_METRIC_MANIFEST = D_DIR / "scenario_metric_manifest.tsv"
D_METRIC_DIR = D_DIR / "scenario_metrics"
D_SCENARIO_SUMMARY = D_DIR / "scenario_model_metric_summary.tsv"
D_CANDIDATE_SUMMARY = D_DIR / "A2_A3_frozen_selection_summary.tsv"
D_ARCH_DECISION = D_DIR / "frozen_architecture_selection.json"
D_A5_DECISION = D_DIR / "A5_branch_decision.json"
D_SUMMARY = D_DIR / "summary.json"

# Exact original 05d implementation, reused only for frozen grid/IBS helpers.
ORIGINAL_05D_SCRIPT = (
    ROOT / "scripts" / "05d_aggregate_negative_transfer_phase_diagram.py"
)

OUT_DIR = ROOT / "results" / "simulation_phase_diagram_diagnostics" / "05d0a"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DIAGNOSTIC_CONTRACT = OUT_DIR / "diagnostic_interpretation_contract.json"
UNIT_COMPOSITION = OUT_DIR / "false_borrow_unit_composition.tsv"
COMPARATOR_SUMMARY = OUT_DIR / "frozen_and_diagnostic_comparator_summary.tsv"
ORACLE_SUMMARY = OUT_DIR / "threshold_feasibility_oracle_summary.tsv"
RISK_SCALE = OUT_DIR / "risk_scale_and_ibs_summary.tsv"
COUNT_CATASTROPHIC = OUT_DIR / "catastrophic_by_false_borrow_count.tsv"
MANUAL_METRIC = OUT_DIR / "manual_uno_metric_replay.tsv"
INVARIANT_AUDIT = OUT_DIR / "implementation_invariant_audit.tsv"
ABLATION_SUMMARY = OUT_DIR / "A3_ablation_failure_summary.tsv"
SCENARIO_DIAGNOSTICS = OUT_DIR / "scenario_failure_diagnostics.tsv"
SUMMARY_JSON = OUT_DIR / "summary.json"

EXPECTED_05C_STATUS = "PASS_CLOSED_SYNTHETIC_TRANSFER_MODEL_MATRIX_COMPLETE"
EXPECTED_05D_STATUS = "HOLD_NO_SAFE_SELECTABLE_ARCHITECTURE"
EXPECTED_05D0_V1_STATUS = (
    "PASS_NO_STRUCTURAL_IMPLEMENTATION_BUG_DETECTED_IN_BOUNDED_AUDIT"
)

EXPECTED_SCENARIOS = 180
EXPECTED_REPLICATES = 21600
EXPECTED_MODULES = 50
EXPECTED_TEST_N = 500

MAIN_MODELS = ["B0", "B4", "A0", "A1", "A2", "A3", "A4"]
ABLATIONS = [
    "A3_NO_EVOLUTION_PRIOR",
    "A3_NO_TARGET_OVERRIDE",
    "A3_PERMUTED_PRIOR",
    "A3_MAPPING_PERMUTATION",
    "A3_SOURCE_OUTCOME_PERMUTATION",
]
ALL_MODELS = MAIN_MODELS + ABLATIONS
MODEL_INDEX = {m: i for i, m in enumerate(ALL_MODELS)}
IBS_MODEL_INDEX = {m: i for i, m in enumerate(MAIN_MODELS)}

NEGATIVE_DELTA = -0.02
CATASTROPHIC_DELTA = -0.05
FROZEN_MAX_NT = 0.10

GATE_THRESHOLD = 0.50
NUM_TOL = 2e-6

# Deterministic subset for computationally heavier no-predictor IBS check.
NULL_IBS_REPS_PER_SCENARIO = 10

# Manual metric replay examples are fixed before diagnostic values are read.
MANUAL_REPLAY_SCENARIO_IDS = ["CORE_0000", "CORE_0015", "STRESS_0179"]
MANUAL_REPLAY_REPLICATE = 0
MANUAL_REPLAY_MODELS = ["B0", "B4", "A2", "A3"]

# Count bins frozen before read.
FALSE_COUNT_BINS = [
    (-0.5, 0.5, "0"),
    (0.5, 2.5, "1-2"),
    (2.5, 5.5, "3-5"),
    (5.5, 10.5, "6-10"),
    (10.5, 50.5, ">10"),
]
HARM_COUNT_BINS = [
    (-0.5, 0.5, "0"),
    (0.5, 1.5, "1"),
    (1.5, 3.5, "2-3"),
    (3.5, 10.5, ">=4"),
]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_file(path: Path) -> Path:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Required artifact missing: {path}")
    return path


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


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
        raise
    return module


def finite_mean(values: Iterable[float]) -> float:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if len(x) else float("nan")


def finite_median(values: Iterable[float]) -> float:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else float("nan")


def finite_q(values: Iterable[float], q: float) -> float:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    return float(np.quantile(x, q)) if len(x) else float("nan")


def make_surv(t: np.ndarray, e: np.ndarray) -> np.ndarray:
    return Surv.from_arrays(
        event=np.asarray(e, dtype=bool),
        time=np.asarray(t, dtype=float),
    )


def direct_tau(
    train_time: np.ndarray,
    train_event: np.ndarray,
    test_time: np.ndarray,
    q: float = 0.80,
) -> Optional[float]:
    events = np.asarray(train_time, dtype=float)[
        np.asarray(train_event, dtype=bool)
    ]
    if len(events) < 2:
        return None
    preferred = float(np.quantile(events, q))
    upper = min(
        preferred,
        float(np.max(train_time)),
        float(np.max(test_time)),
    )
    upper = float(np.nextafter(upper, -np.inf))
    lower = max(float(np.min(train_time)), float(np.min(test_time)))
    if upper <= lower or not np.isfinite(upper):
        return None
    return upper


def direct_uno(
    train_time: np.ndarray,
    train_event: np.ndarray,
    test_time: np.ndarray,
    test_event: np.ndarray,
    risk: np.ndarray,
) -> float:
    tau = direct_tau(train_time, train_event, test_time)
    if tau is None:
        return float("nan")
    try:
        return float(
            concordance_index_ipcw(
                make_surv(train_time, train_event),
                make_surv(test_time, test_event),
                np.asarray(risk, dtype=float),
                tau=tau,
            )[0]
        )
    except Exception:
        return float("nan")


def standardize_target(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.mean(x_train, axis=0, keepdims=True)
    sd = np.std(x_train, axis=0, ddof=0, keepdims=True)
    sd = np.where(sd > 1e-6, sd, 1.0)
    return (
        (x_train - mean) / sd,
        (x_test - mean) / sd,
    )


def mapped_source_beta(
    beta_source_fit: np.ndarray,
    mapping_index: np.ndarray,
) -> np.ndarray:
    out = np.zeros(EXPECTED_MODULES, dtype=float)
    for j in range(EXPECTED_MODULES):
        source_idx = int(mapping_index[j])
        if source_idx >= 0:
            out[j] = float(beta_source_fit[source_idx])
    return out


def bin_label(count: int, bins: List[Tuple[float, float, str]]) -> str:
    x = float(count)
    for lo, hi, label in bins:
        if lo < x <= hi:
            return label
    return "OUTSIDE"


def scenario_is_severe(scenario: pd.Series) -> bool:
    return (
        str(scenario["covariance_shift"]) == "S2_SEVERE"
        or str(scenario["mapping_error"]) == "M2_SEVERE"
    )


def main() -> None:
    started = now_utc()

    print("=" * 120)
    print("Paper 6 - follow-up 05d0a units/oracle/risk-scale diagnostic")
    print("=" * 120)
    print(f"Script version: {SCRIPT_VERSION}")
    print()
    print("Scope:")
    print("  Model fitting / retuning: NO")
    print("  Frozen 05d HOLD changed: NO")
    print("  A5 CLOSE changed by diagnostic alone: NO")
    print("  Human outcomes read: NO")
    print("  GPU execution: NO")
    print()

    for p in [
        D0_V1_SUMMARY,
        A_SCENARIOS,
        A_CONTRACT,
        B_GENERATOR_CONTRACT,
        C_MANIFEST,
        C_SUMMARY,
        C_IMPLEMENTATION,
        D_METRIC_CONTRACT,
        D_METRIC_MANIFEST,
        D_SCENARIO_SUMMARY,
        D_CANDIDATE_SUMMARY,
        D_ARCH_DECISION,
        D_A5_DECISION,
        D_SUMMARY,
        ORIGINAL_05D_SCRIPT,
    ]:
        require_file(p)

    # ------------------------------------------------------------------
    # Freeze FOLLOW-UP interpretation scope after v1 but before oracle/risk-scale read.
    # ------------------------------------------------------------------
    diagnostic_contract = {
        "script_version": SCRIPT_VERSION,
        "status": "FOLLOWUP_SCOPE_FROZEN_AFTER_V1_BEFORE_ORACLE_RISKSCALE_READ",
        "created_utc": now_utc(),
        "known_input_scientific_state": EXPECTED_05D_STATUS,
        "prior_diagnostic_v1_already_run": True,
        "prior_diagnostic_v1_status_required": EXPECTED_05D0_V1_STATUS,
        "chronology_note": (
            "Structural/ablation/harmful-module summaries from 05d0 v1 were "
            "already visible before this follow-up. This file does NOT claim "
            "pre-diagnostic preregistration. It only freezes the scope and A5 "
            "bug-reassessment rule before the not-yet-seen oracle/risk-scale/"
            "per-replicate composition results."
        ),
        "priority_order": [
            "1_UNIT_COMPOSITION",
            "2_FROZEN_AND_DIAGNOSTIC_COMPARATORS",
            "3_THRESHOLD_FEASIBILITY_ORACLES",
            "4_RISK_SCALE_AND_IBS",
            "5_CATASTROPHIC_VS_FALSE_BORROW_COUNTS",
            "6_MANUAL_UNO_REPLAY",
        ],
        "oracle_definitions": {
            "TRUE_TARGET_RISK_ORACLE": (
                "data-generating target linear predictor X_target_test @ beta_target_truth"
            ),
            "REGIME_ABSTENTION_A2": (
                "A2 risk in R0-R3; B0 risk in R4-R5"
            ),
            "REGIME_ABSTENTION_A3": (
                "A3 risk in R0-R3; B0 risk in R4-R5"
            ),
            "B2_DIAGNOSTIC_ZERO_SHOT_LINEAR": (
                "POST-HOC DIAGNOSTIC ONLY: frozen 05c source-linear coefficients "
                "mapped to target modules and applied to target-training-standardized "
                "target test expression. Not part of frozen 05d model registry."
            ),
        },
        "threshold_feasibility_interpretation": (
            "The frozen 0.10 threshold remains unchanged. B0 has zero negative-transfer "
            "by definition; oracle diagnostics ask whether informative transfer/fallback "
            "was feasible, not whether the threshold should be relaxed."
        ),
        "risk_scale_diagnostic": {
            "models": MAIN_MODELS,
            "statistics": [
                "test risk SD",
                "test risk 1st-99th percentile range",
                "test max absolute score",
                "saved IBS",
                "no-predictor IBS on deterministic first 10 replicates per scenario",
            ],
        },
        "A5_re_evaluation_policy": {
            "current_state": "CLOSE_A5",
            "may_re_evaluate_once_only_if": (
                "a deterministic implementation defect is proved that violates "
                "the frozen intended A3 semantics AND materially changes A3 "
                "predictions/metrics used in the A5 trigger"
            ),
            "must_remain_unchanged": [
                "05a scenarios",
                "05a thresholds",
                "A3 scientific architecture",
                "A3 hyperparameters",
                "05c1/05c1a A5 rule",
            ],
            "does_not_qualify": [
                "poor performance of correctly implemented A3",
                "risk-scale pathology inherent to frozen A3 objective",
                "05d scenario_id aggregation bug",
                "post-hoc desire to improve A3",
            ],
            "if_no_qualifying_bug": "A5_CLOSE_FINAL_FOR_PAPER6",
        },
        "new_architecture_rule": (
            "If no qualifying implementation defect is proved, any replacement "
            "architecture requires a NEW prospective contract and NEW untouched "
            "simulation scenarios. Existing 21,600 simulations become development/"
            "diagnostic evidence only for that architecture."
        ),
    }
    write_json(DIAGNOSTIC_CONTRACT, diagnostic_contract)
    diagnostic_contract_hash = sha256_file(DIAGNOSTIC_CONTRACT)

    print(
        "Diagnostic interpretation contract frozen before detailed read: PASS"
    )
    print(f"Diagnostic contract SHA256: {diagnostic_contract_hash}")
    print()

    # ------------------------------------------------------------------
    # Verify frozen state.
    # ------------------------------------------------------------------
    d0_v1_summary = read_json(D0_V1_SUMMARY)
    c_summary = read_json(C_SUMMARY)
    d_summary = read_json(D_SUMMARY)
    arch = read_json(D_ARCH_DECISION)
    a5 = read_json(D_A5_DECISION)
    generator_contract = read_json(B_GENERATOR_CONTRACT)
    metric_contract = read_json(D_METRIC_CONTRACT)

    if str(d0_v1_summary.get("scientific_status")) != EXPECTED_05D0_V1_STATUS:
        raise RuntimeError(
            "05d0 v1 is not in the expected completed structural-audit PASS state."
        )
    if str(c_summary.get("scientific_status")) != EXPECTED_05C_STATUS:
        raise RuntimeError("05c is not in expected PASS state.")
    if str(d_summary.get("scientific_status")) != EXPECTED_05D_STATUS:
        raise RuntimeError("05d is not in expected HOLD state.")
    if arch.get("selected_architecture") is not None:
        raise RuntimeError("05d unexpectedly selected an architecture.")
    if str(a5.get("branch_decision")) != "CLOSE_A5":
        raise RuntimeError("A5 is not CLOSE.")

    # Verify original 05d script identity against frozen metric contract.
    expected_05d_script_hash = str(metric_contract.get("script_sha256"))
    if sha256_file(ORIGINAL_05D_SCRIPT) != expected_05d_script_hash:
        raise RuntimeError("Original frozen 05d script SHA256 mismatch.")

    m05d = load_module(
        ORIGINAL_05D_SCRIPT,
        "paper6_frozen_05d_for_diagnostic_v2",
    )

    generator_path = ROOT / str(
        generator_contract["authoritative_generator_script"]
    )
    require_file(generator_path)
    if sha256_file(generator_path) != str(
        generator_contract["authoritative_generator_script_sha256"]
    ):
        raise RuntimeError("05b generator SHA256 mismatch.")

    gen05b = load_module(
        generator_path,
        "paper6_frozen_05b_generator_for_diagnostic_v2",
    )

    scenarios = pd.read_csv(A_SCENARIOS, sep="\t")
    c_manifest = pd.read_csv(C_MANIFEST, sep="\t")
    d_manifest = pd.read_csv(D_METRIC_MANIFEST, sep="\t")
    frozen_candidate = pd.read_csv(D_CANDIDATE_SUMMARY, sep="\t")

    if len(scenarios) != EXPECTED_SCENARIOS:
        raise RuntimeError("Scenario count changed.")
    if int(scenarios["replicates"].sum()) != EXPECTED_REPLICATES:
        raise RuntimeError("Replicate count changed.")

    scenario_by_id = scenarios.set_index("scenario_id", drop=False)
    c_by_id = {str(r.scenario_id): r for r in c_manifest.itertuples(index=False)}
    d_by_id = {str(r.scenario_id): r for r in d_manifest.itertuples(index=False)}

    # Aggregators.
    invariant_rows: List[Dict[str, Any]] = []
    unit_rows: List[Dict[str, Any]] = []
    scenario_rows: List[Dict[str, Any]] = []
    risk_rows: List[Dict[str, Any]] = []
    count_rows: List[Dict[str, Any]] = []
    manual_rows: List[Dict[str, Any]] = []
    oracle_rows: List[Dict[str, Any]] = []
    ablation_rows_raw: List[Dict[str, Any]] = []

    # Replayed global values.
    replay_by_model = {
        m: {
            "scenario_nt": [],
            "scenario_cat": [],
            "scenario_c": [],
            "scenario_delta": [],
            "misleading_cat": [],
            "severe_cat": [],
        }
        for m in ALL_MODELS
    }

    # Null IBS deterministic subset.
    null_ibs_values: List[float] = []

    # ------------------------------------------------------------------
    # Main scenario loop.
    # ------------------------------------------------------------------
    for s_idx, srow in enumerate(scenarios.itertuples(index=False), start=1):
        sid = str(srow.scenario_id)
        scenario = scenario_by_id.loc[sid]

        crow = c_by_id[sid]
        drow = d_by_id[sid]
        cpath = ROOT / str(crow.output_path)
        dpath = ROOT / str(drow.metric_output_path)

        require_file(cpath)
        require_file(dpath)

        if sha256_file(cpath) != str(crow.output_sha256):
            raise RuntimeError(f"{sid}: 05c output hash mismatch.")
        if sha256_file(dpath) != str(drow.metric_output_sha256):
            raise RuntimeError(f"{sid}: 05d metric hash mismatch.")

        with np.load(cpath, allow_pickle=False) as cdata, np.load(
            dpath, allow_pickle=False
        ) as ddata:
            seeds = np.asarray(cdata["replicate_seed"], dtype=np.int64)
            if not np.array_equal(seeds, ddata["replicate_seed"]):
                raise RuntimeError(f"{sid}: seed order mismatch.")

            n_rep = len(seeds)
            uno = np.asarray(ddata["uno_c"], dtype=float)
            delta = np.asarray(ddata["delta_c_vs_B0"], dtype=float)
            neg = np.asarray(ddata["negative_transfer"], dtype=np.uint8)
            cat = np.asarray(
                ddata["catastrophic_negative_transfer"],
                dtype=np.uint8,
            )
            ibs = np.asarray(ddata["ibs"], dtype=float)

            causal = np.asarray(cdata["causal_mask"], dtype=bool)
            transportable = np.asarray(
                cdata["transportable_mask"], dtype=bool
            )
            beta_source_truth = np.asarray(
                cdata["beta_source_truth"], dtype=float
            )
            beta_target_truth = np.asarray(
                cdata["beta_target_truth"], dtype=float
            )
            prior = np.asarray(cdata["prior_score"], dtype=float)
            mapping = np.asarray(cdata["mapping_index"], dtype=int)
            gate = np.asarray(cdata["gate_A3"], dtype=float)
            gate_no_override = np.asarray(
                cdata["gate_A3_NO_TARGET_OVERRIDE"],
                dtype=float,
            )

            # ----------------------------------------------------------
            # Cheap implementation invariants.
            # ----------------------------------------------------------
            b0i = MODEL_INDEX["B0"]
            invariant_rows.append(
                {
                    "scenario_id": sid,
                    "check": "B0_DELTA_ZERO",
                    "value": float(np.nanmax(np.abs(delta[:, b0i]))),
                    "pass": bool(
                        np.nanmax(np.abs(delta[:, b0i])) <= 1e-7
                    ),
                }
            )
            invariant_rows.append(
                {
                    "scenario_id": sid,
                    "check": "B0_FLAGS_ZERO",
                    "value": int(
                        np.sum(neg[:, b0i]) + np.sum(cat[:, b0i])
                    ),
                    "pass": bool(
                        np.sum(neg[:, b0i]) == 0
                        and np.sum(cat[:, b0i]) == 0
                    ),
                }
            )
            invariant_rows.append(
                {
                    "scenario_id": sid,
                    "check": "NO_OVERRIDE_GATE_EQUALS_PRIOR",
                    "value": float(
                        np.max(np.abs(gate_no_override - prior))
                    ),
                    "pass": bool(
                        np.max(np.abs(gate_no_override - prior))
                        <= NUM_TOL
                    ),
                }
            )

            # ----------------------------------------------------------
            # Frozen model replay + risk-scale summaries.
            # ----------------------------------------------------------
            for model in ALL_MODELS:
                mi = MODEL_INDEX[model]
                valid = np.isfinite(uno[:, mi])
                cmean = finite_mean(uno[:, mi])
                dmean = finite_mean(delta[:, mi])
                ntrate = (
                    float(np.mean(neg[valid, mi]))
                    if np.any(valid)
                    else float("nan")
                )
                catrate = (
                    float(np.mean(cat[valid, mi]))
                    if np.any(valid)
                    else float("nan")
                )

                replay_by_model[model]["scenario_c"].append(cmean)
                replay_by_model[model]["scenario_delta"].append(dmean)
                replay_by_model[model]["scenario_nt"].append(ntrate)
                replay_by_model[model]["scenario_cat"].append(catrate)

                if str(scenario["transfer_regime"]) == "R5_MISLEADING_SOURCE":
                    replay_by_model[model]["misleading_cat"].append(catrate)
                if scenario_is_severe(scenario):
                    replay_by_model[model]["severe_cat"].append(catrate)

                scenario_rows.append(
                    {
                        "scenario_id": sid,
                        "model": model,
                        "target_events": int(scenario["target_events"]),
                        "transfer_regime": str(scenario["transfer_regime"]),
                        "source_prior_state": str(
                            scenario["source_prior_state"]
                        ),
                        "covariance_shift": str(
                            scenario["covariance_shift"]
                        ),
                        "mapping_error": str(scenario["mapping_error"]),
                        "mean_uno_c": cmean,
                        "mean_delta_c_vs_B0": dmean,
                        "negative_transfer_rate": ntrate,
                        "catastrophic_rate": catrate,
                    }
                )

                if model in MAIN_MODELS:
                    rt = np.asarray(
                        cdata[f"risk_test_{model}"], dtype=float
                    )
                    rr_sd = np.std(rt, axis=1, ddof=0)
                    rr_qrange = np.quantile(
                        rt, 0.99, axis=1
                    ) - np.quantile(rt, 0.01, axis=1)
                    rr_max = np.max(np.abs(rt), axis=1)

                    model_ibs = ibs[:, IBS_MODEL_INDEX[model]]

                    for r in range(n_rep):
                        risk_rows.append(
                            {
                                "scenario_id": sid,
                                "replicate": r,
                                "model": model,
                                "transfer_regime": str(
                                    scenario["transfer_regime"]
                                ),
                                "target_events": int(
                                    scenario["target_events"]
                                ),
                                "test_risk_sd": float(rr_sd[r]),
                                "test_risk_q99_q01": float(
                                    rr_qrange[r]
                                ),
                                "test_risk_max_abs": float(
                                    rr_max[r]
                                ),
                                "ibs": float(model_ibs[r]),
                                "catastrophic": int(cat[r, mi]),
                            }
                        )

            # ----------------------------------------------------------
            # Units/composition for A3.
            # ----------------------------------------------------------
            a3i = MODEL_INDEX["A3"]
            broad_false_counts = np.zeros(n_rep, dtype=int)
            harmful_counts = np.zeros(n_rep, dtype=int)
            harmful_gate_mass = np.zeros(n_rep, dtype=float)
            harmful_gate_mean = np.full(n_rep, np.nan, dtype=float)
            attenuated_counts = np.zeros(n_rep, dtype=int)
            any_false = np.zeros(n_rep, dtype=bool)
            any_harmful = np.zeros(n_rep, dtype=bool)

            for r in range(n_rep):
                false_modules = ~transportable[r]
                borrowed = gate[r] >= GATE_THRESHOLD

                broad_false_counts[r] = int(
                    np.sum(false_modules & borrowed)
                )

                harmful = (
                    causal[r]
                    & (~transportable[r])
                    & (beta_source_truth[r] * beta_target_truth[r] < 0)
                    & (np.abs(beta_source_truth[r]) > 0)
                    & (np.abs(beta_target_truth[r]) > 0)
                )
                attenuated = (
                    causal[r]
                    & (~transportable[r])
                    & (np.abs(beta_source_truth[r]) > 0)
                    & (np.abs(beta_target_truth[r]) <= 1e-12)
                )

                harmful_counts[r] = int(np.sum(harmful & borrowed))
                if np.any(harmful):
                    harmful_gate_mass[r] = float(np.sum(gate[r, harmful]))
                    harmful_gate_mean[r] = float(np.mean(gate[r, harmful]))
                attenuated_counts[r] = int(
                    np.sum(attenuated & borrowed)
                )
                any_false[r] = broad_false_counts[r] > 0
                any_harmful[r] = harmful_counts[r] > 0

                count_rows.append(
                    {
                        "scenario_id": sid,
                        "replicate": r,
                        "transfer_regime": str(
                            scenario["transfer_regime"]
                        ),
                        "source_prior_state": str(
                            scenario["source_prior_state"]
                        ),
                        "target_events": int(
                            scenario["target_events"]
                        ),
                        "broad_false_borrow_count": int(
                            broad_false_counts[r]
                        ),
                        "harmful_signflip_borrow_count": int(
                            harmful_counts[r]
                        ),
                        "harmful_signflip_gate_mass": float(
                            harmful_gate_mass[r]
                        ),
                        "harmful_signflip_mean_gate": float(
                            harmful_gate_mean[r]
                        ),
                        "attenuated_causal_borrow_count": int(
                            attenuated_counts[r]
                        ),
                        "catastrophic_A3": int(cat[r, a3i]),
                        "delta_c_A3": float(delta[r, a3i]),
                    }
                )

            unit_rows.append(
                {
                    "scenario_id": sid,
                    "transfer_regime": str(
                        scenario["transfer_regime"]
                    ),
                    "source_prior_state": str(
                        scenario["source_prior_state"]
                    ),
                    "target_events": int(
                        scenario["target_events"]
                    ),
                    "mean_false_borrow_count_per_replicate": float(
                        np.mean(broad_false_counts)
                    ),
                    "fraction_replicates_any_false_borrow": float(
                        np.mean(any_false)
                    ),
                    "mean_harmful_signflip_borrow_count": float(
                        np.mean(harmful_counts)
                    ),
                    "mean_harmful_signflip_gate_mass": float(
                        np.mean(harmful_gate_mass)
                    ),
                    "mean_harmful_signflip_gate": finite_mean(
                        harmful_gate_mean
                    ),
                    "fraction_replicates_any_harmful_signflip_borrow": float(
                        np.mean(any_harmful)
                    ),
                    "A3_catastrophic_rate": float(
                        np.mean(cat[:, a3i])
                    ),
                }
            )

            # ----------------------------------------------------------
            # Frozen A3 ablations, scenario-level.
            # ----------------------------------------------------------
            a3_mean = finite_mean(uno[:, a3i])
            a3_nt = float(np.mean(neg[:, a3i]))
            a3_cat = float(np.mean(cat[:, a3i]))

            for abl in ABLATIONS:
                ai = MODEL_INDEX[abl]
                ablation_rows_raw.append(
                    {
                        "scenario_id": sid,
                        "ablation": abl,
                        "transfer_regime": str(
                            scenario["transfer_regime"]
                        ),
                        "source_prior_state": str(
                            scenario["source_prior_state"]
                        ),
                        "target_events": int(
                            scenario["target_events"]
                        ),
                        "delta_uno_ablation_minus_A3": (
                            finite_mean(uno[:, ai]) - a3_mean
                        ),
                        "change_nt_rate": (
                            float(np.mean(neg[:, ai])) - a3_nt
                        ),
                        "change_cat_rate": (
                            float(np.mean(cat[:, ai])) - a3_cat
                        ),
                    }
                )

            # ----------------------------------------------------------
            # Manual Uno replay fixed examples.
            # ----------------------------------------------------------
            if sid in MANUAL_REPLAY_SCENARIO_IDS:
                r = MANUAL_REPLAY_REPLICATE
                tr_t = np.asarray(
                    cdata["target_train_time"][r], dtype=float
                )
                tr_e = np.asarray(
                    cdata["target_train_event"][r], dtype=np.uint8
                )
                te_t = np.asarray(
                    cdata["target_test_time"][r], dtype=float
                )
                te_e = np.asarray(
                    cdata["target_test_event"][r], dtype=np.uint8
                )

                for model in MANUAL_REPLAY_MODELS:
                    mi = MODEL_INDEX[model]
                    risk = np.asarray(
                        cdata[f"risk_test_{model}"][r],
                        dtype=float,
                    )
                    observed = direct_uno(
                        tr_t, tr_e, te_t, te_e, risk
                    )
                    frozen = float(uno[r, mi])

                    manual_rows.append(
                        {
                            "scenario_id": sid,
                            "replicate": r,
                            "model": model,
                            "manual_uno_c": observed,
                            "frozen_05d_uno_c": frozen,
                            "absolute_difference": abs(
                                observed - frozen
                            ),
                            "pass": bool(
                                np.isfinite(observed)
                                and np.isfinite(frozen)
                                and abs(observed - frozen) <= 1e-8
                            ),
                        }
                    )

            # ----------------------------------------------------------
            # Regenerate exact synthetic arrays for oracle + diagnostic B2.
            # ----------------------------------------------------------
            oracle_c = np.full(n_rep, np.nan, dtype=float)
            b2diag_c = np.full(n_rep, np.nan, dtype=float)
            abstain_a2_c = np.full(n_rep, np.nan, dtype=float)
            abstain_a3_c = np.full(n_rep, np.nan, dtype=float)

            for r, seed in enumerate(seeds):
                generated = gen05b.generate_full_replicate(
                    int(seed),
                    scenario.to_dict(),
                    target_test_n=EXPECTED_TEST_N,
                )

                # Replay truth identity before oracle use.
                if not np.array_equal(
                    generated["beta_target"].astype(np.float32),
                    cdata["beta_target_truth"][r],
                ):
                    raise RuntimeError(
                        f"{sid} rep {r}: regenerated beta_target mismatch."
                    )
                if not np.array_equal(
                    generated["mapping_index"].astype(np.int16),
                    cdata["mapping_index"][r],
                ):
                    raise RuntimeError(
                        f"{sid} rep {r}: regenerated mapping mismatch."
                    )

                tr_t = np.asarray(
                    cdata["target_train_time"][r], dtype=float
                )
                tr_e = np.asarray(
                    cdata["target_train_event"][r], dtype=np.uint8
                )
                te_t = np.asarray(
                    cdata["target_test_time"][r], dtype=float
                )
                te_e = np.asarray(
                    cdata["target_test_event"][r], dtype=np.uint8
                )

                xtr = np.asarray(
                    generated["X_target_train"], dtype=float
                )
                xte = np.asarray(
                    generated["X_target_test"], dtype=float
                )
                _, xte_z = standardize_target(xtr, xte)

                true_risk = (
                    xte
                    @ np.asarray(
                        cdata["beta_target_truth"][r],
                        dtype=float,
                    )
                )
                oracle_c[r] = direct_uno(
                    tr_t, tr_e, te_t, te_e, true_risk
                )

                beta_prior = mapped_source_beta(
                    np.asarray(
                        cdata["beta_source_linear_fit"][r],
                        dtype=float,
                    ),
                    np.asarray(
                        cdata["mapping_index"][r],
                        dtype=int,
                    ),
                )
                b2risk = xte_z @ beta_prior
                b2diag_c[r] = direct_uno(
                    tr_t, tr_e, te_t, te_e, b2risk
                )

                # Regime-level oracle abstention: source-adaptive model only in
                # R0-R3, exact B0 fallback in R4/R5.
                regime = str(scenario["transfer_regime"])
                if regime in {
                    "R4_NONTRANSPORTABLE",
                    "R5_MISLEADING_SOURCE",
                }:
                    abstain_a2_c[r] = float(uno[r, MODEL_INDEX["B0"]])
                    abstain_a3_c[r] = float(uno[r, MODEL_INDEX["B0"]])
                else:
                    abstain_a2_c[r] = float(uno[r, MODEL_INDEX["A2"]])
                    abstain_a3_c[r] = float(uno[r, MODEL_INDEX["A3"]])

                # No-predictor IBS on deterministic first K reps/scenario.
                if r < min(NULL_IBS_REPS_PER_SCENARIO, n_rep):
                    zero_train = np.zeros_like(tr_t, dtype=float)
                    zero_test = np.zeros_like(te_t, dtype=float)
                    grid = m05d.ibs_time_grid(tr_t, tr_e, te_t)
                    null_ibs = m05d.safe_ibs(
                        tr_t,
                        tr_e,
                        zero_train,
                        te_t,
                        te_e,
                        zero_test,
                        grid,
                    )
                    if np.isfinite(null_ibs):
                        null_ibs_values.append(float(null_ibs))

            b0_c = uno[:, MODEL_INDEX["B0"]]

            for name, values in [
                ("TRUE_TARGET_RISK_ORACLE", oracle_c),
                ("B2_DIAGNOSTIC_ZERO_SHOT_LINEAR", b2diag_c),
                ("REGIME_ABSTENTION_A2", abstain_a2_c),
                ("REGIME_ABSTENTION_A3", abstain_a3_c),
            ]:
                valid = np.isfinite(values) & np.isfinite(b0_c)
                deltas = values - b0_c

                oracle_rows.append(
                    {
                        "scenario_id": sid,
                        "diagnostic_model": name,
                        "transfer_regime": str(
                            scenario["transfer_regime"]
                        ),
                        "target_events": int(
                            scenario["target_events"]
                        ),
                        "mean_uno_c": finite_mean(values),
                        "mean_delta_c_vs_B0": finite_mean(deltas),
                        "negative_transfer_rate": (
                            float(
                                np.mean(
                                    deltas[valid] <= NEGATIVE_DELTA
                                )
                            )
                            if np.any(valid)
                            else float("nan")
                        ),
                        "catastrophic_rate": (
                            float(
                                np.mean(
                                    deltas[valid]
                                    <= CATASTROPHIC_DELTA
                                )
                            )
                            if np.any(valid)
                            else float("nan")
                        ),
                    }
                )

        if s_idx % 20 == 0 or s_idx == EXPECTED_SCENARIOS:
            print(f"  diagnosed scenarios: {s_idx}/{EXPECTED_SCENARIOS}")

    # ------------------------------------------------------------------
    # Assemble outputs.
    # ------------------------------------------------------------------
    invariant_df = pd.DataFrame(invariant_rows)
    unit_df = pd.DataFrame(unit_rows)
    scenario_df = pd.DataFrame(scenario_rows)
    risk_df = pd.DataFrame(risk_rows)
    count_df = pd.DataFrame(count_rows)
    manual_df = pd.DataFrame(manual_rows)
    oracle_df = pd.DataFrame(oracle_rows)
    ablation_raw = pd.DataFrame(ablation_rows_raw)

    invariant_df.to_csv(INVARIANT_AUDIT, sep="\t", index=False)
    unit_df.to_csv(UNIT_COMPOSITION, sep="\t", index=False)
    scenario_df.to_csv(SCENARIO_DIAGNOSTICS, sep="\t", index=False)
    manual_df.to_csv(MANUAL_METRIC, sep="\t", index=False)

    # Frozen comparator summary.
    comparator_rows = []
    for model in MAIN_MODELS:
        x = scenario_df[scenario_df["model"] == model]
        comparator_rows.append(
            {
                "model": model,
                "status": "FROZEN_05C_MODEL",
                "equal_scenario_mean_uno_c": finite_mean(
                    x["mean_uno_c"]
                ),
                "aggregate_negative_transfer_rate": finite_mean(
                    x["negative_transfer_rate"]
                ),
                "catastrophic_rate_misleading": finite_mean(
                    x.loc[
                        x["transfer_regime"]
                        == "R5_MISLEADING_SOURCE",
                        "catastrophic_rate",
                    ]
                ),
                "catastrophic_rate_severe_shift": finite_mean(
                    x.loc[
                        (
                            (x["covariance_shift"] == "S2_SEVERE")
                            | (x["mapping_error"] == "M2_SEVERE")
                        ),
                        "catastrophic_rate",
                    ]
                ),
            }
        )

    # Add diagnostic B2-like zero-shot.
    b2x = oracle_df[
        oracle_df["diagnostic_model"]
        == "B2_DIAGNOSTIC_ZERO_SHOT_LINEAR"
    ].copy()
    # Join the frozen severe-shift labels by scenario id.
    severe_ids = set(
        scenarios.loc[
            (scenarios["covariance_shift"].astype(str) == "S2_SEVERE")
            | (scenarios["mapping_error"].astype(str) == "M2_SEVERE"),
            "scenario_id",
        ].astype(str)
    )
    comparator_rows.append(
        {
            "model": "B2_DIAGNOSTIC_ZERO_SHOT_LINEAR",
            "status": "POSTHOC_DIAGNOSTIC_NOT_FROZEN_SELECTION_MODEL",
            "equal_scenario_mean_uno_c": finite_mean(b2x["mean_uno_c"]),
            "aggregate_negative_transfer_rate": finite_mean(
                b2x["negative_transfer_rate"]
            ),
            "catastrophic_rate_misleading": finite_mean(
                b2x.loc[
                    b2x["transfer_regime"] == "R5_MISLEADING_SOURCE",
                    "catastrophic_rate",
                ]
            ),
            "catastrophic_rate_severe_shift": finite_mean(
                b2x.loc[
                    b2x["scenario_id"].astype(str).isin(severe_ids),
                    "catastrophic_rate",
                ]
            ),
        }
    )
    comparator_df = pd.DataFrame(comparator_rows)
    comparator_df.to_csv(COMPARATOR_SUMMARY, sep="\t", index=False)

    # Oracle summary.
    oracle_summary_rows = []
    for name, x in oracle_df.groupby("diagnostic_model"):
        oracle_summary_rows.append(
            {
                "diagnostic_model": name,
                "n_scenarios": len(x),
                "equal_scenario_mean_uno_c": finite_mean(x["mean_uno_c"]),
                "aggregate_negative_transfer_rate": finite_mean(
                    x["negative_transfer_rate"]
                ),
                "catastrophic_rate": finite_mean(x["catastrophic_rate"]),
                "meets_frozen_0_10_nt_target_descriptively": bool(
                    finite_mean(x["negative_transfer_rate"])
                    <= FROZEN_MAX_NT
                ),
            }
        )
    oracle_summary = pd.DataFrame(oracle_summary_rows)
    oracle_summary.to_csv(ORACLE_SUMMARY, sep="\t", index=False)

    # Risk scale summary overall and misleading subset.
    risk_summary_rows = []
    for subset_name, subset in [
        ("ALL", risk_df),
        (
            "R5_MISLEADING_SOURCE",
            risk_df[
                risk_df["transfer_regime"] == "R5_MISLEADING_SOURCE"
            ],
        ),
    ]:
        for model, x in subset.groupby("model"):
            risk_summary_rows.append(
                {
                    "subset": subset_name,
                    "model": model,
                    "n_replicates": len(x),
                    "median_test_risk_sd": finite_median(
                        x["test_risk_sd"]
                    ),
                    "q95_test_risk_sd": finite_q(
                        x["test_risk_sd"], 0.95
                    ),
                    "median_test_q99_q01": finite_median(
                        x["test_risk_q99_q01"]
                    ),
                    "median_test_max_abs": finite_median(
                        x["test_risk_max_abs"]
                    ),
                    "median_IBS": finite_median(x["ibs"]),
                    "median_IBS_catastrophic_reps": finite_median(
                        x.loc[x["catastrophic"] == 1, "ibs"]
                    ),
                }
            )
    risk_summary = pd.DataFrame(risk_summary_rows)

    null_ibs_median = finite_median(null_ibs_values)
    risk_summary["deterministic_subset_no_predictor_IBS_median"] = (
        null_ibs_median
    )
    risk_summary.to_csv(RISK_SCALE, sep="\t", index=False)

    # Catastrophic by count bins.
    count_summary_rows = []
    for subset_name, x in [
        ("ALL", count_df),
        (
            "R5_MISLEADING_SOURCE",
            count_df[
                count_df["transfer_regime"] == "R5_MISLEADING_SOURCE"
            ],
        ),
    ]:
        for count_col, bins in [
            ("broad_false_borrow_count", FALSE_COUNT_BINS),
            ("harmful_signflip_borrow_count", HARM_COUNT_BINS),
        ]:
            temp = x.copy()
            temp["count_bin"] = [
                bin_label(int(v), bins)
                for v in temp[count_col].to_numpy()
            ]
            for label, g in temp.groupby("count_bin"):
                count_summary_rows.append(
                    {
                        "subset": subset_name,
                        "count_metric": count_col,
                        "count_bin": label,
                        "n_replicates": len(g),
                        "mean_count": finite_mean(g[count_col]),
                        "catastrophic_rate_A3": finite_mean(
                            g["catastrophic_A3"]
                        ),
                        "mean_delta_c_A3": finite_mean(
                            g["delta_c_A3"]
                        ),
                    }
                )
    count_summary = pd.DataFrame(count_summary_rows)
    count_summary.to_csv(
        COUNT_CATASTROPHIC,
        sep="\t",
        index=False,
    )

    # Ablation summary.
    ablation_summary_rows = []
    for subset_name, mask_fn in [
        ("ALL", lambda x: np.ones(len(x), dtype=bool)),
        (
            "R5_MISLEADING_SOURCE",
            lambda x: (
                x["transfer_regime"] == "R5_MISLEADING_SOURCE"
            ).to_numpy(),
        ),
        (
            "P2_MISLEADING_PRIOR",
            lambda x: (
                x["source_prior_state"] == "P2_MISLEADING"
            ).to_numpy(),
        ),
    ]:
        for abl, x in ablation_raw.groupby("ablation"):
            mask = mask_fn(x)
            g = x.loc[mask]
            ablation_summary_rows.append(
                {
                    "subset": subset_name,
                    "ablation": abl,
                    "n_scenarios": len(g),
                    "mean_delta_uno_ablation_minus_A3": finite_mean(
                        g["delta_uno_ablation_minus_A3"]
                    ),
                    "mean_change_nt_rate": finite_mean(
                        g["change_nt_rate"]
                    ),
                    "mean_change_cat_rate": finite_mean(
                        g["change_cat_rate"]
                    ),
                }
            )
    ablation_summary = pd.DataFrame(ablation_summary_rows)
    ablation_summary.to_csv(
        ABLATION_SUMMARY,
        sep="\t",
        index=False,
    )

    # ------------------------------------------------------------------
    # Verify frozen candidate summary independently.
    # ------------------------------------------------------------------
    replay_candidate_rows = []
    for model in ["A2", "A3"]:
        x = scenario_df[scenario_df["model"] == model]
        replay_candidate_rows.append(
            {
                "model": model,
                "aggregate_nt": finite_mean(
                    x["negative_transfer_rate"]
                ),
                "cat_misleading": finite_mean(
                    x.loc[
                        x["transfer_regime"] == "R5_MISLEADING_SOURCE",
                        "catastrophic_rate",
                    ]
                ),
                "cat_severe": finite_mean(
                    x.loc[
                        (
                            (x["covariance_shift"] == "S2_SEVERE")
                            | (x["mapping_error"] == "M2_SEVERE")
                        ),
                        "catastrophic_rate",
                    ]
                ),
            }
        )
    replay_candidate = pd.DataFrame(replay_candidate_rows).set_index("model")
    frozen_candidate_idx = frozen_candidate.set_index("model")

    selection_replay_pass = True
    for model in ["A2", "A3"]:
        checks = [
            (
                replay_candidate.loc[model, "aggregate_nt"],
                frozen_candidate_idx.loc[
                    model, "aggregate_negative_transfer_rate"
                ],
            ),
            (
                replay_candidate.loc[model, "cat_misleading"],
                frozen_candidate_idx.loc[
                    model, "catastrophic_rate_misleading"
                ],
            ),
            (
                replay_candidate.loc[model, "cat_severe"],
                frozen_candidate_idx.loc[
                    model, "catastrophic_rate_severe_shift"
                ],
            ),
        ]
        selection_replay_pass &= all(
            np.isclose(float(a), float(b), atol=NUM_TOL, rtol=0)
            for a, b in checks
        )

    invariant_pass = bool(invariant_df["pass"].all())
    manual_pass = bool(manual_df["pass"].all())

    # Unit composition headline for R5.
    r5_units = unit_df[
        unit_df["transfer_regime"] == "R5_MISLEADING_SOURCE"
    ]
    r5_mean_false_count = finite_mean(
        r5_units["mean_false_borrow_count_per_replicate"]
    )
    r5_any_false = finite_mean(
        r5_units["fraction_replicates_any_false_borrow"]
    )
    r5_harm_count = finite_mean(
        r5_units["mean_harmful_signflip_borrow_count"]
    )
    r5_harm_gate_mass = finite_mean(
        r5_units["mean_harmful_signflip_gate_mass"]
    )
    r5_harm_gate_mean = finite_mean(
        r5_units["mean_harmful_signflip_gate"]
    )
    r5_any_harm = finite_mean(
        r5_units[
            "fraction_replicates_any_harmful_signflip_borrow"
        ]
    )
    r5_cat = finite_mean(r5_units["A3_catastrophic_rate"])

    # Risk-scale comparison.
    all_risk = risk_summary[risk_summary["subset"] == "ALL"].set_index("model")
    r5_risk = risk_summary[
        risk_summary["subset"] == "R5_MISLEADING_SOURCE"
    ].set_index("model")

    a3_vs_a2_sd_ratio = float(
        all_risk.loc["A3", "median_test_risk_sd"]
        / all_risk.loc["A2", "median_test_risk_sd"]
    )
    a3_vs_a2_r5_sd_ratio = float(
        r5_risk.loc["A3", "median_test_risk_sd"]
        / r5_risk.loc["A2", "median_test_risk_sd"]
    )

    # Feasibility headlines.
    oracle_idx = oracle_summary.set_index("diagnostic_model")
    true_oracle_nt = float(
        oracle_idx.loc[
            "TRUE_TARGET_RISK_ORACLE",
            "aggregate_negative_transfer_rate",
        ]
    )
    abstain_a2_nt = float(
        oracle_idx.loc[
            "REGIME_ABSTENTION_A2",
            "aggregate_negative_transfer_rate",
        ]
    )
    abstain_a3_nt = float(
        oracle_idx.loc[
            "REGIME_ABSTENTION_A3",
            "aggregate_negative_transfer_rate",
        ]
    )

    # Diagnostic classification is intentionally conservative.
    if not invariant_pass or not manual_pass or not selection_replay_pass:
        diagnostic_status = (
            "HOLD_TECHNICAL_METRIC_OR_INVARIANT_DISCREPANCY"
        )
    else:
        diagnostic_status = (
            "PASS_FROZEN_RESULT_REPRODUCED_DIAGNOSTIC_INTERPRETATION_REQUIRED"
        )

    summary = {
        "script_version": SCRIPT_VERSION,
        "status": (
            "PASS" if diagnostic_status.startswith("PASS") else "HOLD"
        ),
        "scientific_status": diagnostic_status,
        "run_started_utc": started,
        "run_finished_utc": now_utc(),
        "diagnostic_contract_sha256": diagnostic_contract_hash,
        "prior_05d0_v1_summary_sha256": sha256_file(D0_V1_SUMMARY),
        "prior_05d0_v1_status": EXPECTED_05D0_V1_STATUS,
        "frozen_05d_status_unchanged": EXPECTED_05D_STATUS,
        "A5_state_unchanged_by_this_diagnostic": "CLOSE_A5",
        "implementation_invariants_pass": invariant_pass,
        "manual_uno_replay_pass": manual_pass,
        "frozen_selection_metric_replay_pass": selection_replay_pass,
        "unit_composition_R5": {
            "mean_broad_false_borrowed_modules_per_replicate": r5_mean_false_count,
            "fraction_replicates_with_any_broad_false_borrow": r5_any_false,
            "mean_harmful_signflip_borrowed_modules_per_replicate": r5_harm_count,
            "mean_harmful_signflip_gate_mass_per_replicate": r5_harm_gate_mass,
            "mean_harmful_signflip_gate": r5_harm_gate_mean,
            "fraction_replicates_with_any_harmful_signflip_borrow": r5_any_harm,
            "A3_catastrophic_rate": r5_cat,
        },
        "risk_scale": {
            "A3_to_A2_median_test_risk_SD_ratio_all": a3_vs_a2_sd_ratio,
            "A3_to_A2_median_test_risk_SD_ratio_R5": a3_vs_a2_r5_sd_ratio,
            "no_predictor_IBS_median_deterministic_subset": null_ibs_median,
        },
        "threshold_feasibility": {
            "true_target_risk_oracle_NT_rate": true_oracle_nt,
            "regime_abstention_A2_NT_rate": abstain_a2_nt,
            "regime_abstention_A3_NT_rate": abstain_a3_nt,
            "frozen_target": FROZEN_MAX_NT,
        },
        "interpretation_guardrails": {
            "do_not_call_module_vs_prediction_decoupling_a_discovery_before_unit_analysis": True,
            "do_not_call_A3_bug_based_on_bad_performance_alone": True,
            "qualifying_bug_required_for_A5_one_time_re_evaluation": True,
            "no_threshold_relaxation": True,
            "no_human_outcomes": True,
        },
        "final_artifact_hashes": {
            "diagnostic_interpretation_contract.json": sha256_file(
                DIAGNOSTIC_CONTRACT
            ),
            "false_borrow_unit_composition.tsv": sha256_file(
                UNIT_COMPOSITION
            ),
            "frozen_and_diagnostic_comparator_summary.tsv": sha256_file(
                COMPARATOR_SUMMARY
            ),
            "threshold_feasibility_oracle_summary.tsv": sha256_file(
                ORACLE_SUMMARY
            ),
            "risk_scale_and_ibs_summary.tsv": sha256_file(RISK_SCALE),
            "catastrophic_by_false_borrow_count.tsv": sha256_file(
                COUNT_CATASTROPHIC
            ),
            "manual_uno_metric_replay.tsv": sha256_file(MANUAL_METRIC),
            "implementation_invariant_audit.tsv": sha256_file(
                INVARIANT_AUDIT
            ),
            "A3_ablation_failure_summary.tsv": sha256_file(
                ABLATION_SUMMARY
            ),
            "scenario_failure_diagnostics.tsv": sha256_file(
                SCENARIO_DIAGNOSTICS
            ),
        },
    }
    write_json(SUMMARY_JSON, summary)

    # ------------------------------------------------------------------
    # Compact console output.
    # ------------------------------------------------------------------
    print()
    print("=" * 120)
    print("05d0 PRIORITY 1 — UNITS / COMPOSITION")
    print("=" * 120)
    print(f"R5 mean false-borrowed modules/replicate: {r5_mean_false_count:.3f}")
    print(f"R5 fraction any false borrow:             {r5_any_false:.4f}")
    print(f"R5 mean harmful sign-flip borrows/rep:    {r5_harm_count:.3f}")
    print(f"R5 harmful sign-flip gate mass/rep:       {r5_harm_gate_mass:.3f}")
    print(f"R5 mean gate on harmful sign-flips:       {r5_harm_gate_mean:.4f}")
    print(f"R5 fraction any harmful sign-flip borrow: {r5_any_harm:.4f}")
    print(f"R5 A3 catastrophic rate:                  {r5_cat:.4f}")

    print()
    print("=" * 120)
    print("05d0 PRIORITY 2 — FROZEN / DIAGNOSTIC COMPARATORS")
    print("=" * 120)
    print(comparator_df.to_string(index=False))

    print()
    print("=" * 120)
    print("05d0 PRIORITY 3 — THRESHOLD FEASIBILITY")
    print("=" * 120)
    print(oracle_summary.to_string(index=False))

    print()
    print("=" * 120)
    print("05d0 PRIORITY 4 — RISK SCALE / IBS")
    print("=" * 120)
    display_risk = risk_summary[
        risk_summary["model"].isin(["B0", "B4", "A1", "A2", "A3", "A4"])
    ]
    print(display_risk.to_string(index=False))
    print()
    print(f"A3/A2 median risk-SD ratio [ALL]: {a3_vs_a2_sd_ratio:.3f}")
    print(f"A3/A2 median risk-SD ratio [R5]:  {a3_vs_a2_r5_sd_ratio:.3f}")
    print(f"No-predictor IBS median [fixed subset]: {null_ibs_median:.4f}")

    print()
    print("=" * 120)
    print("05d0 PRIORITY 5 — CATASTROPHIC VS BORROW COUNTS")
    print("=" * 120)
    print(
        count_summary[
            count_summary["subset"] == "R5_MISLEADING_SOURCE"
        ].to_string(index=False)
    )

    print()
    print("=" * 120)
    print("05d0 PRIORITY 6 — MANUAL UNO REPLAY")
    print("=" * 120)
    print(manual_df.to_string(index=False))

    print()
    print("=" * 120)
    print("A3 ABLATIONS — POST-HOC DIAGNOSTIC ONLY")
    print("=" * 120)
    print(ablation_summary.to_string(index=False))

    print()
    print("=" * 120)
    print("05d0a FOLLOW-UP DIAGNOSTIC SUMMARY")
    print("=" * 120)
    print(f"Status: {diagnostic_status}")
    print("Frozen 05d HOLD changed: NO")
    print("A5 CLOSE changed by this diagnostic: NO")
    print(
        "Implementation invariants: "
        + ("PASS" if invariant_pass else "FAIL")
    )
    print(
        "Manual Uno replay: "
        + ("PASS" if manual_pass else "FAIL")
    )
    print(
        "Frozen selection-metric replay: "
        + ("PASS" if selection_replay_pass else "FAIL")
    )
    print(
        f"True-target oracle NT rate: {true_oracle_nt:.4f} "
        f"(frozen target <= {FROZEN_MAX_NT:.2f})"
    )
    print(
        f"Regime-abstention A2 NT rate: {abstain_a2_nt:.4f}"
    )
    print(
        f"Regime-abstention A3 NT rate: {abstain_a3_nt:.4f}"
    )
    print()
    print(
        "Do not interpret the 05d HOLD mechanistically until the units, oracle, "
        "risk-scale and count-composition outputs above are reviewed."
    )
    print("=" * 120)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 120, file=sys.stderr)
        print("05d0a follow-up failure diagnostic: FAIL", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print("=" * 120, file=sys.stderr)
        raise
