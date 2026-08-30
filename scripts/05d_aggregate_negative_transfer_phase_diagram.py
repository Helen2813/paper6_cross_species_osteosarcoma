#!/usr/bin/env python3
"""
Paper 6 - aggregate frozen synthetic negative-transfer phase diagram and select A2/A3.

This is the scientific metric/selection stage for the already-completed 05c
synthetic model matrix.

Chronology
----------
05a  froze scenarios, candidates, metrics and selection thresholds.
05b  froze deterministic known-truth generator/recipes.
05c  fit the closed model matrix and saved predictions/gates.
05c0 verified every saved branch executed and is nondegenerate.
05c1/05c1a preregistered the exploratory A5 branch BEFORE 05d.

05d:
- DOES NOT fit or tune any predictive model.
- DOES NOT read any real DOG2/human outcome or expression values.
- computes frozen synthetic evaluation metrics from 05c predictions;
- applies the frozen A2-vs-A3 rule mechanically;
- applies the preregistered A5 OPEN/CLOSE rule mechanically.

The metric implementation contract is written/verified BEFORE the first 05c
scenario shard is opened.

No CLI arguments.
CPU only.
Restart-safe by scenario.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import roc_auc_score
except ImportError as exc:
    raise ImportError("05d requires scikit-learn.") from exc

try:
    from sksurv.metrics import concordance_index_ipcw, integrated_brier_score
    from sksurv.util import Surv
except ImportError as exc:
    raise ImportError(
        "05d requires scikit-survival in the active Paper-6 .venv."
    ) from exc


SCRIPT_VERSION = "05d-aggregate-negative-transfer-phase-diagram-v1-no-cli"
METRIC_IMPLEMENTATION_VERSION = "paper6-05d-synthetic-evaluation-v1"

ROOT = Path(__file__).resolve().parents[1]

# 05a
A_DIR = ROOT / "results" / "simulation_contract" / "05a"
A_CONTRACT = A_DIR / "negative_transfer_simulation_contract.json"
A_SCENARIOS = A_DIR / "simulation_scenario_registry.tsv"
A_SELECTION_RULES = A_DIR / "architecture_selection_rules.tsv"
A_SUMMARY = A_DIR / "summary.json"

# 05c
C_DIR = ROOT / "results" / "simulation_model_matrix" / "05c"
C_SUMMARY = C_DIR / "summary.json"
C_MANIFEST = C_DIR / "scenario_output_manifest.tsv"
C_IMPLEMENTATION = C_DIR / "model_implementation_contract.json"

# 05c0
C0_DIR = ROOT / "results" / "simulation_model_matrix_diagnostics" / "05c0"
C0_SUMMARY = C0_DIR / "summary.json"

# 05c1 / 05c1a
C1_DIR = ROOT / "results" / "simulation_extension_contract" / "05c1"
C1_CONTRACT = C1_DIR / "A5_ECHRR_branch_contract.json"
C1_SUMMARY = C1_DIR / "summary.json"

C1A_DIR = ROOT / "results" / "simulation_extension_contract" / "05c1a"
C1A_CONTRACT = C1A_DIR / "A5_operational_branch_definitions.json"
C1A_CONTRASTS = C1A_DIR / "A5_shift_contrast_registry.tsv"
C1A_SUMMARY = C1A_DIR / "summary.json"

# Output.
OUT_DIR = ROOT / "results" / "simulation_phase_diagram" / "05d"
CHECKPOINT_DIR = OUT_DIR / "scenario_metrics"
CHECKPOINT_MANIFEST_DIR = OUT_DIR / "scenario_manifests"

for directory in [OUT_DIR, CHECKPOINT_DIR, CHECKPOINT_MANIFEST_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

METRIC_CONTRACT = OUT_DIR / "metric_implementation_contract.json"
SCENARIO_SUMMARY = OUT_DIR / "scenario_model_metric_summary.tsv"
MODEL_SUMMARY = OUT_DIR / "model_aggregate_summary.tsv"
CANDIDATE_SUMMARY = OUT_DIR / "A2_A3_frozen_selection_summary.tsv"
LEARNING_CURVES = OUT_DIR / "event_scaled_learning_curves.tsv"
ABLATION_SUMMARY = OUT_DIR / "A3_ablation_summary.tsv"
MODULE_SUMMARY = OUT_DIR / "A3_module_recovery_summary.tsv"
A5_SHIFT_DIAGNOSTIC = OUT_DIR / "A5_shift_trigger_diagnostics.tsv"
A5_DECISION = OUT_DIR / "A5_branch_decision.json"
ARCHITECTURE_DECISION = OUT_DIR / "frozen_architecture_selection.json"
SCENARIO_METRIC_MANIFEST = OUT_DIR / "scenario_metric_manifest.tsv"
SUMMARY_JSON = OUT_DIR / "summary.json"

EXPECTED_05A_STATUS = (
    "PASS_NEGATIVE_TRANSFER_SIMULATION_AND_AI_SELECTION_CONTRACT_FROZEN"
)
EXPECTED_05C_STATUS = (
    "PASS_CLOSED_SYNTHETIC_TRANSFER_MODEL_MATRIX_COMPLETE"
)
EXPECTED_05C0_STATUS = (
    "PASS_SYNTHETIC_MODEL_MATRIX_EXECUTION_COMPLETE"
)
EXPECTED_05C1_STATUS = (
    "PASS_A5_EXPLORATORY_BRANCH_SPECIFICATION_FROZEN_BEFORE_05D"
)
EXPECTED_05C1A_STATUS = (
    "PASS_A5_OPERATIONAL_BRANCH_DEFINITIONS_FROZEN_BEFORE_05D"
)

EXPECTED_SCENARIOS = 180
EXPECTED_REPLICATES = 21600
EXPECTED_MODULES = 50

MAIN_MODELS = ["B0", "B4", "A0", "A1", "A2", "A3", "A4"]
ABLATIONS = [
    "A3_NO_EVOLUTION_PRIOR",
    "A3_NO_TARGET_OVERRIDE",
    "A3_PERMUTED_PRIOR",
    "A3_MAPPING_PERMUTATION",
    "A3_SOURCE_OUTCOME_PERMUTATION",
]
ALL_MODELS = MAIN_MODELS + ABLATIONS

IBS_MODELS = MAIN_MODELS
CALIBRATION_MODELS = MAIN_MODELS

GATE_KEYS = {
    "A3": "gate_A3",
    "A3_NO_EVOLUTION_PRIOR": "gate_A3_NO_EVOLUTION_PRIOR",
    "A3_NO_TARGET_OVERRIDE": "gate_A3_NO_TARGET_OVERRIDE",
    "A3_PERMUTED_PRIOR": "gate_A3_PERMUTED_PRIOR",
    "A3_MAPPING_PERMUTATION": "gate_A3_MAPPING_PERMUTATION",
    "A3_SOURCE_OUTCOME_PERMUTATION": "gate_A3_SOURCE_OUTCOME_PERMUTATION",
}

# Frozen 05a thresholds. They are verified against the 05a contract at runtime.
NEGATIVE_TRANSFER_DELTA_C = -0.02
CATASTROPHIC_DELTA_C = -0.05
MAX_NEGATIVE_TRANSFER_RATE = 0.10
MAX_CATASTROPHIC_RATE = 0.05
MIN_MODULE_RECOVERY_AUROC = 0.70
MAX_FALSE_BORROW_RATE = 0.20
PRACTICAL_TIE_C = 0.01
PRACTICAL_TIE_IBS = 0.01

BORROW_GATE_THRESHOLD = 0.50

# Metric implementation choices, frozen BEFORE 05d sees model performance.
UNO_TAU_EVENT_QUANTILE = 0.80

IBS_LOWER_EVENT_QUANTILE = 0.20
IBS_UPPER_EVENT_QUANTILE = 0.80
IBS_N_TIME_POINTS = 20
IBS_MIN_TIME_POINTS = 5

CALIBRATION_HORIZON_EVENT_QUANTILE = 0.50
CALIBRATION_SLOPE_MAX_ITER = 40
CALIBRATION_SLOPE_TOL = 1e-8
CALIBRATION_SLOPE_ABS_CAP = 20.0

EPS = 1e-10
RISK_EXP_CLIP = 30.0

# Equal scenario weighting prevents the 200-replicate core scenarios from
# automatically receiving twice the architecture-selection weight of the
# 100-replicate stress scenarios. Replicates remain equally weighted within
# each scenario.
AGGREGATION_POLICY = "EQUAL_SCENARIO_WEIGHT_AFTER_WITHIN_SCENARIO_REPLICATE_MEAN"

# Checkpoint schema.
SCENARIO_METRIC_SCHEMA_VERSION = 1


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


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


def atomic_savez_compressed(path: Path, **arrays: np.ndarray) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    tmp.replace(path)


def make_surv(time_values: np.ndarray, event_values: np.ndarray) -> np.ndarray:
    return Surv.from_arrays(
        event=np.asarray(event_values, dtype=bool),
        time=np.asarray(time_values, dtype=float),
    )


def metric_contract_payload(script_path: Path) -> Dict[str, Any]:
    return {
        "script_version": SCRIPT_VERSION,
        "metric_implementation_version": METRIC_IMPLEMENTATION_VERSION,
        "status": "PASS_FROZEN_BEFORE_FIRST_05D_SCENARIO_READ",
        "created_utc": now_utc(),
        "script_sha256": sha256_file(script_path),
        "primary_discrimination": {
            "metric": "Uno C via sksurv.metrics.concordance_index_ipcw",
            "risk_orientation": "larger score = higher hazard / shorter survival",
            "tau_rule": (
                "80th percentile of TARGET-TRAINING observed EVENT times, clipped "
                "strictly below common train/test observed follow-up support"
            ),
            "tau_event_quantile": UNO_TAU_EVENT_QUANTILE,
            "same_tau_for_all_models_within_replicate": True,
        },
        "integrated_brier_score": {
            "models": IBS_MODELS,
            "prediction": (
                "Cox-type survival curve obtained by Breslow baseline cumulative "
                "hazard estimated on the target-training outcomes using that "
                "model's saved target-training risk score"
            ),
            "time_grid": (
                "20 equally spaced points from the 20th to 80th percentile of "
                "target-training observed EVENT times, clipped to strict common "
                "train/test follow-up support"
            ),
            "lower_event_quantile": IBS_LOWER_EVENT_QUANTILE,
            "upper_event_quantile": IBS_UPPER_EVENT_QUANTILE,
            "n_time_points": IBS_N_TIME_POINTS,
            "minimum_time_points": IBS_MIN_TIME_POINTS,
            "implementation": "sksurv.metrics.integrated_brier_score",
        },
        "calibration": {
            "models": CALIBRATION_MODELS,
            "slope": (
                "univariate Cox calibration coefficient of TEST outcome on the "
                "unchanged saved model risk score; ideal=1"
            ),
            "slope_implementation": (
                "Breslow-tie Newton-Raphson on the one-dimensional partial likelihood"
            ),
            "intercept": (
                "descriptive calibration-in-the-large at the median target-training "
                "event-time horizon: cloglog(KM observed test survival) minus "
                "cloglog(mean predicted test survival); ideal=0"
            ),
            "used_for_architecture_selection": False,
        },
        "negative_transfer": {
            "delta_c": "UnoC(model) - UnoC(B0), same replicate",
            "negative_indicator": "delta_c <= -0.02",
            "catastrophic_indicator": "delta_c <= -0.05",
        },
        "gate_truth_metrics": {
            "true_borrowable_label": "05b/05c transportable_mask over all 50 modules",
            "module_recovery_AUROC": (
                "ROC AUROC of continuous borrowing gate versus transportable_mask; "
                "defined only when both truth classes are present"
            ),
            "false_borrow_rate": (
                "fraction of truly nontransportable modules with gate >=0.50"
            ),
            "borrow_gate_threshold": BORROW_GATE_THRESHOLD,
        },
        "aggregation": {
            "within_scenario": "arithmetic mean over all valid frozen replicates",
            "across_scenarios": AGGREGATION_POLICY,
            "selection_environment": "all 180 frozen 05a scenarios",
            "IBS_A2_A3_comparison": (
                "within each scenario use only replicates where BOTH A2 and A3 IBS "
                "are finite; then equal-weight scenario means"
            ),
        },
        "safety": {
            "negative_transfer": (
                "aggregate equal-scenario negative-transfer rate <=0.10"
            ),
            "catastrophic": (
                "catastrophic rate <=0.05 separately in R5 misleading-source "
                "scenarios AND in severe covariance/mapping-shift scenarios"
            ),
            "A3_module_recovery": (
                "mean equal-scenario AUROC >=0.70 across R1/R2/R3 scenarios"
            ),
            "A3_false_borrow": (
                "mean equal-scenario false-borrow rate <=0.20 across R5 scenarios"
            ),
            "gate_specific_criteria_apply_to_A2": False,
        },
        "A2_A3_selection": {
            "eligible": ["A2", "A3"],
            "if_only_one_passes_safety": "select the safe candidate",
            "if_neither_passes_safety": "HOLD_NO_SAFE_SELECTABLE_ARCHITECTURE",
            "if_both_pass": (
                "Uno C is primary. If abs(mean_C_A3-mean_C_A2)>=0.01, select higher "
                "mean Uno C. Otherwise compare common-replicate mean IBS: if "
                "abs(mean_IBS_A3-mean_IBS_A2)>=0.01, select lower IBS. If both "
                "differences are <0.01, select simpler A2."
            ),
            "if_required_A2_A3_IBS_is_not_assessable_and_C_is_practical_tie": (
                "HOLD_SELECTION_IBS_NOT_ASSESSABLE"
            ),
            "calibration_used_for_tie_break": False,
        },
        "A5_branch": {
            "source": "05c1 + 05c1a only",
            "05c1a_contrast_registry_sha256": sha256_file(C1A_CONTRASTS),
            "must_apply_mechanically": True,
        },
        "real_data_firewall": {
            "DOG2_values_read": False,
            "GSE16091_values_read": False,
            "TARGET_GSE21257_GSE39055_values_read": False,
        },
    }


def freeze_or_verify_metric_contract(payload: Dict[str, Any]) -> None:
    if METRIC_CONTRACT.exists():
        existing = read_json(METRIC_CONTRACT)

        # created_utc can differ only on first write; everything scientific must match.
        for key in [
            "script_version",
            "metric_implementation_version",
            "script_sha256",
            "primary_discrimination",
            "integrated_brier_score",
            "calibration",
            "negative_transfer",
            "gate_truth_metrics",
            "aggregation",
            "safety",
            "A2_A3_selection",
            "A5_branch",
            "real_data_firewall",
        ]:
            if existing.get(key) != payload.get(key):
                raise RuntimeError(
                    f"Existing 05d metric contract differs at {key!r}. "
                    "Do not continue with a modified implementation."
                )
        return

    write_json(METRIC_CONTRACT, payload)


def verify_frozen_thresholds(contract_05a: Dict[str, Any]) -> None:
    t = contract_05a.get("selection_thresholds") or {}

    checks = {
        "negative_transfer_definition_delta_c": NEGATIVE_TRANSFER_DELTA_C,
        "catastrophic_negative_transfer_definition_delta_c": CATASTROPHIC_DELTA_C,
        "max_primary_negative_transfer_rate": MAX_NEGATIVE_TRANSFER_RATE,
        "max_catastrophic_negative_transfer_rate": MAX_CATASTROPHIC_RATE,
        "min_true_borrowable_module_AUROC": MIN_MODULE_RECOVERY_AUROC,
        "max_false_borrow_rate_misleading": MAX_FALSE_BORROW_RATE,
        "practical_tie_delta_c": PRACTICAL_TIE_C,
        "practical_tie_delta_IBS": PRACTICAL_TIE_IBS,
    }

    for key, expected in checks.items():
        observed = float(t.get(key))
        if not np.isclose(observed, expected, rtol=0, atol=1e-12):
            raise RuntimeError(
                f"Frozen 05a threshold changed for {key}: {observed} vs {expected}."
            )


def strict_common_upper(
    train_time: np.ndarray,
    test_time: np.ndarray,
    preferred: float,
) -> Optional[float]:
    upper = min(
        float(preferred),
        float(np.max(train_time)),
        float(np.max(test_time)),
    )
    upper = float(np.nextafter(upper, -np.inf))

    lower = max(
        float(np.min(train_time)),
        float(np.min(test_time)),
    )

    if not np.isfinite(upper) or upper <= lower:
        return None
    return upper


def uno_tau(
    train_time: np.ndarray,
    train_event: np.ndarray,
    test_time: np.ndarray,
) -> Optional[float]:
    event_times = np.asarray(train_time)[np.asarray(train_event, dtype=bool)]
    if len(event_times) < 2:
        return None

    preferred = float(
        np.quantile(event_times, UNO_TAU_EVENT_QUANTILE)
    )
    return strict_common_upper(train_time, test_time, preferred)


def safe_uno_c(
    train_time: np.ndarray,
    train_event: np.ndarray,
    test_time: np.ndarray,
    test_event: np.ndarray,
    risk: np.ndarray,
    tau: Optional[float],
) -> float:
    if tau is None:
        return float("nan")

    try:
        y_train = make_surv(train_time, train_event)
        y_test = make_surv(test_time, test_event)
        value = float(
            concordance_index_ipcw(
                y_train,
                y_test,
                np.asarray(risk, dtype=float),
                tau=float(tau),
            )[0]
        )
        return value if np.isfinite(value) else float("nan")
    except Exception:
        return float("nan")


def breslow_cumulative_hazard_at(
    train_time: np.ndarray,
    train_event: np.ndarray,
    train_risk: np.ndarray,
    query_times: np.ndarray,
) -> Optional[np.ndarray]:
    t = np.asarray(train_time, dtype=float)
    e = np.asarray(train_event, dtype=bool)
    r = np.asarray(train_risk, dtype=float)
    q = np.asarray(query_times, dtype=float)

    if len(t) != len(e) or len(t) != len(r):
        return None
    if not np.isfinite(t).all() or not np.isfinite(r).all():
        return None

    event_times = np.unique(t[e])
    if len(event_times) == 0:
        return None

    exp_risk = np.exp(np.clip(r, -RISK_EXP_CLIP, RISK_EXP_CLIP))

    increments = np.empty(len(event_times), dtype=float)

    for i, event_time in enumerate(event_times):
        d = int(np.sum(e & np.isclose(t, event_time, rtol=0, atol=0)))
        denom = float(np.sum(exp_risk[t >= event_time]))

        if d <= 0 or not np.isfinite(denom) or denom <= 0:
            return None

        increments[i] = d / denom

    cumulative = np.cumsum(increments)
    idx = np.searchsorted(event_times, q, side="right") - 1

    out = np.zeros(len(q), dtype=float)
    valid = idx >= 0
    out[valid] = cumulative[idx[valid]]
    return out


def ibs_time_grid(
    train_time: np.ndarray,
    train_event: np.ndarray,
    test_time: np.ndarray,
) -> Optional[np.ndarray]:
    event_times = np.asarray(train_time, dtype=float)[
        np.asarray(train_event, dtype=bool)
    ]

    if len(event_times) < 2:
        return None

    lower_pref = float(
        np.quantile(event_times, IBS_LOWER_EVENT_QUANTILE)
    )
    upper_pref = float(
        np.quantile(event_times, IBS_UPPER_EVENT_QUANTILE)
    )

    lower = max(
        lower_pref,
        float(np.nextafter(np.min(test_time), np.inf)),
    )
    upper = strict_common_upper(train_time, test_time, upper_pref)

    if upper is None or upper <= lower:
        return None

    grid = np.linspace(
        lower,
        upper,
        IBS_N_TIME_POINTS,
        dtype=float,
    )
    grid = np.unique(grid[np.isfinite(grid)])

    if len(grid) < IBS_MIN_TIME_POINTS:
        return None
    return grid


def safe_ibs(
    train_time: np.ndarray,
    train_event: np.ndarray,
    train_risk: np.ndarray,
    test_time: np.ndarray,
    test_event: np.ndarray,
    test_risk: np.ndarray,
    grid: Optional[np.ndarray],
) -> float:
    if grid is None:
        return float("nan")

    try:
        h0 = breslow_cumulative_hazard_at(
            train_time,
            train_event,
            train_risk,
            grid,
        )
        if h0 is None:
            return float("nan")

        rr = np.exp(
            np.clip(
                np.asarray(test_risk, dtype=float),
                -RISK_EXP_CLIP,
                RISK_EXP_CLIP,
            )
        )
        survival = np.exp(-np.outer(rr, h0))
        survival = np.clip(survival, 1e-8, 1.0)

        y_train = make_surv(train_time, train_event)
        y_test = make_surv(test_time, test_event)

        value = float(
            integrated_brier_score(
                y_train,
                y_test,
                survival,
                np.asarray(grid, dtype=float),
            )
        )
        return value if np.isfinite(value) else float("nan")
    except Exception:
        return float("nan")


def cox_calibration_slope(
    time_values: np.ndarray,
    event_values: np.ndarray,
    predictor: np.ndarray,
) -> float:
    """
    One-dimensional Cox calibration slope with Breslow ties.

    The model is h(t | eta) = h0(t) exp(beta * eta).
    Ideal beta = 1 when eta is perfectly calibrated on the Cox scale.
    """
    t = np.asarray(time_values, dtype=float)
    e = np.asarray(event_values, dtype=bool)
    x = np.asarray(predictor, dtype=float)

    if len(t) < 3 or int(e.sum()) < 2:
        return float("nan")
    if not np.isfinite(t).all() or not np.isfinite(x).all():
        return float("nan")
    if np.std(x) <= EPS:
        return float("nan")

    order = np.argsort(t)[::-1]
    ts = t[order]
    es = e[order]
    xs = x[order]

    # Group-end index for each row in descending-time order, so tied times use
    # the same complete risk set under Breslow handling.
    group_end = np.empty(len(ts), dtype=int)
    start = 0
    while start < len(ts):
        end = start
        while end + 1 < len(ts) and ts[end + 1] == ts[start]:
            end += 1
        group_end[start:end + 1] = end
        start = end + 1

    event_positions = np.flatnonzero(es)
    if len(event_positions) < 2:
        return float("nan")

    beta = 1.0

    for _ in range(CALIBRATION_SLOPE_MAX_ITER):
        bx = np.clip(beta * xs, -RISK_EXP_CLIP, RISK_EXP_CLIP)
        w = np.exp(bx)

        s0 = np.cumsum(w)
        s1 = np.cumsum(w * xs)
        s2 = np.cumsum(w * xs * xs)

        ends = group_end[event_positions]
        denom = s0[ends]

        if np.any(denom <= 0) or not np.isfinite(denom).all():
            return float("nan")

        mean_x = s1[ends] / denom
        var_x = s2[ends] / denom - mean_x * mean_x
        var_x = np.maximum(var_x, 0.0)

        score = float(np.sum(xs[event_positions] - mean_x))
        info = float(np.sum(var_x))

        if not np.isfinite(score) or not np.isfinite(info) or info <= EPS:
            return float("nan")

        step = score / info
        beta_new = float(
            np.clip(
                beta + step,
                -CALIBRATION_SLOPE_ABS_CAP,
                CALIBRATION_SLOPE_ABS_CAP,
            )
        )

        if abs(beta_new - beta) < CALIBRATION_SLOPE_TOL:
            beta = beta_new
            break

        beta = beta_new

    return beta if np.isfinite(beta) else float("nan")


def km_survival_at(
    time_values: np.ndarray,
    event_values: np.ndarray,
    horizon: float,
) -> float:
    t = np.asarray(time_values, dtype=float)
    e = np.asarray(event_values, dtype=bool)

    if not np.isfinite(horizon):
        return float("nan")

    order = np.argsort(t)
    t = t[order]
    e = e[order]

    n_at_risk = len(t)
    survival = 1.0
    i = 0

    while i < len(t):
        current = t[i]
        j = i

        while j + 1 < len(t) and t[j + 1] == current:
            j += 1

        group_n = j - i + 1
        d = int(np.sum(e[i:j + 1]))

        if current <= horizon and d > 0:
            if n_at_risk <= 0:
                return float("nan")
            survival *= 1.0 - d / n_at_risk

        n_at_risk -= group_n
        i = j + 1

        if current > horizon:
            break

    return float(np.clip(survival, 1e-8, 1.0 - 1e-8))


def calibration_intercept_at_horizon(
    train_time: np.ndarray,
    train_event: np.ndarray,
    train_risk: np.ndarray,
    test_time: np.ndarray,
    test_event: np.ndarray,
    test_risk: np.ndarray,
) -> float:
    event_times = np.asarray(train_time, dtype=float)[
        np.asarray(train_event, dtype=bool)
    ]

    if len(event_times) < 2:
        return float("nan")

    horizon_pref = float(
        np.quantile(
            event_times,
            CALIBRATION_HORIZON_EVENT_QUANTILE,
        )
    )
    horizon = strict_common_upper(
        train_time,
        test_time,
        horizon_pref,
    )
    if horizon is None:
        return float("nan")

    h0 = breslow_cumulative_hazard_at(
        train_time,
        train_event,
        train_risk,
        np.asarray([horizon], dtype=float),
    )
    if h0 is None:
        return float("nan")

    rr = np.exp(
        np.clip(
            np.asarray(test_risk, dtype=float),
            -RISK_EXP_CLIP,
            RISK_EXP_CLIP,
        )
    )
    pred_surv = np.exp(-rr * float(h0[0]))
    mean_pred_surv = float(
        np.clip(
            np.mean(pred_surv),
            1e-8,
            1.0 - 1e-8,
        )
    )
    observed_surv = km_survival_at(
        test_time,
        test_event,
        horizon,
    )

    if not np.isfinite(observed_surv):
        return float("nan")

    # Complementary-log-log difference; ideal=0.
    obs = math.log(-math.log(observed_surv))
    pred = math.log(-math.log(mean_pred_surv))
    value = obs - pred
    return float(value) if np.isfinite(value) else float("nan")


def safe_module_auc(
    truth: np.ndarray,
    gate: np.ndarray,
) -> float:
    truth = np.asarray(truth, dtype=int)
    gate = np.asarray(gate, dtype=float)

    if len(np.unique(truth)) < 2:
        return float("nan")
    if not np.isfinite(gate).all():
        return float("nan")

    try:
        value = float(roc_auc_score(truth, gate))
        return value if np.isfinite(value) else float("nan")
    except Exception:
        return float("nan")


def scenario_metric_paths(scenario_id: str) -> Tuple[Path, Path]:
    return (
        CHECKPOINT_DIR / f"{scenario_id}.npz",
        CHECKPOINT_MANIFEST_DIR / f"{scenario_id}.json",
    )


def load_valid_metric_checkpoint(
    scenario_id: str,
    source_output_sha256: str,
    metric_contract_sha256: str,
) -> Optional[Dict[str, Any]]:
    metric_path, manifest_path = scenario_metric_paths(scenario_id)

    if not metric_path.exists() or not manifest_path.exists():
        return None

    manifest = read_json(manifest_path)

    if clean(manifest.get("status")) != "PASS":
        return None
    if clean(manifest.get("source_output_sha256")) != source_output_sha256:
        return None
    if clean(manifest.get("metric_contract_sha256")) != metric_contract_sha256:
        return None
    if clean(manifest.get("metric_output_sha256")) != sha256_file(metric_path):
        return None

    return manifest


def evaluate_scenario(
    source_path: Path,
) -> Dict[str, np.ndarray]:
    with np.load(source_path, allow_pickle=False) as data:
        n_rep = len(data["replicate_seed"])

        train_time = np.asarray(
            data["target_train_time"],
            dtype=float,
        )
        train_event = np.asarray(
            data["target_train_event"],
            dtype=np.uint8,
        )
        test_time = np.asarray(
            data["target_test_time"],
            dtype=float,
        )
        test_event = np.asarray(
            data["target_test_event"],
            dtype=np.uint8,
        )

        transportable = np.asarray(
            data["transportable_mask"],
            dtype=np.uint8,
        )

        uno = np.full(
            (n_rep, len(ALL_MODELS)),
            np.nan,
            dtype=np.float64,
        )
        ibs = np.full(
            (n_rep, len(IBS_MODELS)),
            np.nan,
            dtype=np.float64,
        )
        cal_slope = np.full(
            (n_rep, len(CALIBRATION_MODELS)),
            np.nan,
            dtype=np.float64,
        )
        cal_intercept = np.full(
            (n_rep, len(CALIBRATION_MODELS)),
            np.nan,
            dtype=np.float64,
        )

        gate_models = list(GATE_KEYS)
        module_auc = np.full(
            (n_rep, len(gate_models)),
            np.nan,
            dtype=np.float64,
        )
        false_borrow = np.full_like(module_auc, np.nan)
        mean_gate_true = np.full_like(module_auc, np.nan)
        mean_gate_false = np.full_like(module_auc, np.nan)

        for r in range(n_rep):
            tr_t = train_time[r]
            tr_e = train_event[r]
            te_t = test_time[r]
            te_e = test_event[r]

            tau = uno_tau(tr_t, tr_e, te_t)
            grid = ibs_time_grid(tr_t, tr_e, te_t)

            # Uno C for all main/ablation branches.
            for m_idx, model in enumerate(ALL_MODELS):
                risk_test = np.asarray(
                    data[f"risk_test_{model}"][r],
                    dtype=float,
                )
                uno[r, m_idx] = safe_uno_c(
                    tr_t,
                    tr_e,
                    te_t,
                    te_e,
                    risk_test,
                    tau,
                )

            # IBS + calibration for primary/main benchmark models.
            for m_idx, model in enumerate(IBS_MODELS):
                risk_train = np.asarray(
                    data[f"risk_train_{model}"][r],
                    dtype=float,
                )
                risk_test = np.asarray(
                    data[f"risk_test_{model}"][r],
                    dtype=float,
                )

                ibs[r, m_idx] = safe_ibs(
                    tr_t,
                    tr_e,
                    risk_train,
                    te_t,
                    te_e,
                    risk_test,
                    grid,
                )

                cal_slope[r, m_idx] = cox_calibration_slope(
                    te_t,
                    te_e,
                    risk_test,
                )
                cal_intercept[r, m_idx] = (
                    calibration_intercept_at_horizon(
                        tr_t,
                        tr_e,
                        risk_train,
                        te_t,
                        te_e,
                        risk_test,
                    )
                )

            # Known-truth borrowing metrics.
            truth = transportable[r].astype(int)

            for g_idx, model in enumerate(gate_models):
                gate = np.asarray(
                    data[GATE_KEYS[model]][r],
                    dtype=float,
                )

                module_auc[r, g_idx] = safe_module_auc(
                    truth,
                    gate,
                )

                true_mask = truth == 1
                false_mask = truth == 0

                if np.any(true_mask):
                    mean_gate_true[r, g_idx] = float(
                        np.mean(gate[true_mask])
                    )

                if np.any(false_mask):
                    mean_gate_false[r, g_idx] = float(
                        np.mean(gate[false_mask])
                    )
                    false_borrow[r, g_idx] = float(
                        np.mean(
                            gate[false_mask]
                            >= BORROW_GATE_THRESHOLD
                        )
                    )

        b0_idx = ALL_MODELS.index("B0")
        delta_c = uno - uno[:, [b0_idx]]
        negative = (
            np.isfinite(delta_c)
            & (delta_c <= NEGATIVE_TRANSFER_DELTA_C)
        ).astype(np.uint8)
        catastrophic = (
            np.isfinite(delta_c)
            & (delta_c <= CATASTROPHIC_DELTA_C)
        ).astype(np.uint8)

        return {
            "replicate_seed": np.asarray(
                data["replicate_seed"],
                dtype=np.int64,
            ),
            "uno_c": uno.astype(np.float32),
            "delta_c_vs_B0": delta_c.astype(np.float32),
            "negative_transfer": negative,
            "catastrophic_negative_transfer": catastrophic,
            "ibs": ibs.astype(np.float32),
            "calibration_slope": cal_slope.astype(np.float32),
            "calibration_intercept": cal_intercept.astype(np.float32),
            "module_recovery_auc": module_auc.astype(np.float32),
            "false_borrow_rate": false_borrow.astype(np.float32),
            "mean_gate_true": mean_gate_true.astype(np.float32),
            "mean_gate_false": mean_gate_false.astype(np.float32),
        }


def finite_mean(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if len(x) else float("nan")


def finite_median(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else float("nan")


def finite_fraction(values: np.ndarray) -> float:
    x = np.asarray(values)
    if x.size == 0:
        return float("nan")
    return float(np.mean(np.isfinite(x)))


def mean_bool_over_valid(
    indicator: np.ndarray,
    valid_mask: np.ndarray,
) -> float:
    indicator = np.asarray(indicator, dtype=float)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    if int(valid_mask.sum()) == 0:
        return float("nan")

    return float(np.mean(indicator[valid_mask]))


def aggregate_scenario(
    scenario: pd.Series,
    metric_path: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    with np.load(metric_path, allow_pickle=False) as data:
        uno = np.asarray(data["uno_c"], dtype=float)
        delta = np.asarray(data["delta_c_vs_B0"], dtype=float)
        negative = np.asarray(data["negative_transfer"], dtype=np.uint8)
        catastrophic = np.asarray(
            data["catastrophic_negative_transfer"],
            dtype=np.uint8,
        )
        ibs = np.asarray(data["ibs"], dtype=float)
        cal_slope = np.asarray(
            data["calibration_slope"],
            dtype=float,
        )
        cal_intercept = np.asarray(
            data["calibration_intercept"],
            dtype=float,
        )
        module_auc = np.asarray(
            data["module_recovery_auc"],
            dtype=float,
        )
        false_borrow = np.asarray(
            data["false_borrow_rate"],
            dtype=float,
        )
        mean_gate_true = np.asarray(
            data["mean_gate_true"],
            dtype=float,
        )
        mean_gate_false = np.asarray(
            data["mean_gate_false"],
            dtype=float,
        )

        rows: List[Dict[str, Any]] = []
        gate_models = list(GATE_KEYS)

        for model_idx, model in enumerate(ALL_MODELS):
            c_values = uno[:, model_idx]
            valid_c = np.isfinite(c_values)

            row: Dict[str, Any] = {
                "scenario_id": str(scenario["scenario_id"]),
                "family": str(scenario["family"]),
                "target_events": int(scenario["target_events"]),
                "transfer_regime": str(scenario["transfer_regime"]),
                "transferable_fraction": float(
                    scenario["transferable_fraction"]
                ),
                "covariance_shift": str(
                    scenario["covariance_shift"]
                ),
                "censoring": str(scenario["censoring"]),
                "mapping_error": str(scenario["mapping_error"]),
                "source_prior_state": str(
                    scenario["source_prior_state"]
                ),
                "replicates": int(scenario["replicates"]),
                "model": model,
                "valid_uno_fraction": finite_fraction(c_values),
                "mean_uno_c": finite_mean(c_values),
                "median_uno_c": finite_median(c_values),
                "mean_delta_c_vs_B0": finite_mean(delta[:, model_idx]),
                "negative_transfer_rate": mean_bool_over_valid(
                    negative[:, model_idx],
                    valid_c,
                ),
                "catastrophic_negative_transfer_rate": (
                    mean_bool_over_valid(
                        catastrophic[:, model_idx],
                        valid_c,
                    )
                ),
                "mean_ibs": float("nan"),
                "valid_ibs_fraction": float("nan"),
                "median_calibration_slope": float("nan"),
                "median_abs_calibration_intercept": float("nan"),
                "mean_module_recovery_auc": float("nan"),
                "mean_false_borrow_rate": float("nan"),
                "mean_gate_true": float("nan"),
                "mean_gate_false": float("nan"),
            }

            if model in IBS_MODELS:
                j = IBS_MODELS.index(model)
                row["mean_ibs"] = finite_mean(ibs[:, j])
                row["valid_ibs_fraction"] = finite_fraction(
                    ibs[:, j]
                )

                slope = cal_slope[:, j]
                intercept = cal_intercept[:, j]

                row["median_calibration_slope"] = (
                    finite_median(slope)
                )
                row["median_abs_calibration_intercept"] = (
                    finite_median(np.abs(intercept))
                )

            if model in gate_models:
                j = gate_models.index(model)
                row["mean_module_recovery_auc"] = (
                    finite_mean(module_auc[:, j])
                )
                row["mean_false_borrow_rate"] = (
                    finite_mean(false_borrow[:, j])
                )
                row["mean_gate_true"] = finite_mean(
                    mean_gate_true[:, j]
                )
                row["mean_gate_false"] = finite_mean(
                    mean_gate_false[:, j]
                )

            rows.append(row)

        # Pairwise-common A2/A3 IBS for the frozen practical-tie calculation.
        a2_ibs = ibs[:, IBS_MODELS.index("A2")]
        a3_ibs = ibs[:, IBS_MODELS.index("A3")]
        common = np.isfinite(a2_ibs) & np.isfinite(a3_ibs)

        common_summary = {
            "scenario_id": str(scenario["scenario_id"]),
            "n_common_A2_A3_IBS": int(common.sum()),
            "common_A2_mean_IBS": (
                float(np.mean(a2_ibs[common]))
                if common.any()
                else float("nan")
            ),
            "common_A3_mean_IBS": (
                float(np.mean(a3_ibs[common]))
                if common.any()
                else float("nan")
            ),
        }

        return rows, common_summary


def scenario_is_severe_shift(row: pd.Series) -> bool:
    return (
        str(row["covariance_shift"]) == "S2_SEVERE"
        or str(row["mapping_error"]) == "M2_SEVERE"
    )


def equal_scenario_mean(
    frame: pd.DataFrame,
    column: str,
) -> float:
    values = pd.to_numeric(
        frame[column],
        errors="coerce",
    ).to_numpy(dtype=float)
    return finite_mean(values)


def build_candidate_summary(
    scenario_summary: pd.DataFrame,
    common_ibs: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for model in ["A2", "A3"]:
        part = scenario_summary[
            scenario_summary["model"] == model
        ].copy()

        misleading = part[
            part["transfer_regime"] == "R5_MISLEADING_SOURCE"
        ].copy()

        severe_mask = (
            (part["covariance_shift"] == "S2_SEVERE")
            | (part["mapping_error"] == "M2_SEVERE")
        )
        severe = part[severe_mask].copy()

        row: Dict[str, Any] = {
            "model": model,
            "aggregate_negative_transfer_rate": (
                equal_scenario_mean(
                    part,
                    "negative_transfer_rate",
                )
            ),
            "catastrophic_rate_misleading": (
                equal_scenario_mean(
                    misleading,
                    "catastrophic_negative_transfer_rate",
                )
            ),
            "catastrophic_rate_severe_shift": (
                equal_scenario_mean(
                    severe,
                    "catastrophic_negative_transfer_rate",
                )
            ),
            "equal_scenario_mean_uno_c": (
                equal_scenario_mean(part, "mean_uno_c")
            ),
            "equal_scenario_mean_IBS": (
                equal_scenario_mean(part, "mean_ibs")
            ),
            "common_A2_A3_equal_scenario_mean_IBS": (
                float("nan")
            ),
            "module_recovery_AUROC_partial": float("nan"),
            "false_borrow_rate_misleading": float("nan"),
            "negative_transfer_safety_pass": False,
            "catastrophic_misleading_safety_pass": False,
            "catastrophic_severe_shift_safety_pass": False,
            "module_recovery_pass": np.nan,
            "false_borrow_pass": np.nan,
            "overall_protection_pass": False,
        }

        common_col = (
            "common_A2_mean_IBS"
            if model == "A2"
            else "common_A3_mean_IBS"
        )
        row["common_A2_A3_equal_scenario_mean_IBS"] = (
            equal_scenario_mean(common_ibs, common_col)
        )

        row["negative_transfer_safety_pass"] = bool(
            np.isfinite(
                row["aggregate_negative_transfer_rate"]
            )
            and row["aggregate_negative_transfer_rate"]
            <= MAX_NEGATIVE_TRANSFER_RATE
        )
        row["catastrophic_misleading_safety_pass"] = bool(
            np.isfinite(row["catastrophic_rate_misleading"])
            and row["catastrophic_rate_misleading"]
            <= MAX_CATASTROPHIC_RATE
        )
        row["catastrophic_severe_shift_safety_pass"] = bool(
            np.isfinite(
                row["catastrophic_rate_severe_shift"]
            )
            and row["catastrophic_rate_severe_shift"]
            <= MAX_CATASTROPHIC_RATE
        )

        base_pass = (
            row["negative_transfer_safety_pass"]
            and row["catastrophic_misleading_safety_pass"]
            and row[
                "catastrophic_severe_shift_safety_pass"
            ]
        )

        if model == "A3":
            partial = part[
                part["transfer_regime"].isin(
                    [
                        "R1_MOSTLY_TRANSPORTABLE",
                        "R2_PARTIALLY_TRANSPORTABLE",
                        "R3_WEAKLY_TRANSPORTABLE",
                    ]
                )
            ].copy()

            row["module_recovery_AUROC_partial"] = (
                equal_scenario_mean(
                    partial,
                    "mean_module_recovery_auc",
                )
            )
            row["false_borrow_rate_misleading"] = (
                equal_scenario_mean(
                    misleading,
                    "mean_false_borrow_rate",
                )
            )

            row["module_recovery_pass"] = bool(
                np.isfinite(
                    row["module_recovery_AUROC_partial"]
                )
                and row["module_recovery_AUROC_partial"]
                >= MIN_MODULE_RECOVERY_AUROC
            )
            row["false_borrow_pass"] = bool(
                np.isfinite(
                    row["false_borrow_rate_misleading"]
                )
                and row["false_borrow_rate_misleading"]
                <= MAX_FALSE_BORROW_RATE
            )

            row["overall_protection_pass"] = bool(
                base_pass
                and row["module_recovery_pass"]
                and row["false_borrow_pass"]
            )
        else:
            row["module_recovery_pass"] = np.nan
            row["false_borrow_pass"] = np.nan
            row["overall_protection_pass"] = bool(base_pass)

        rows.append(row)

    return pd.DataFrame(rows)


def select_architecture(
    candidate_summary: pd.DataFrame,
) -> Dict[str, Any]:
    a2 = candidate_summary.set_index("model").loc["A2"]
    a3 = candidate_summary.set_index("model").loc["A3"]

    safe2 = bool(a2["overall_protection_pass"])
    safe3 = bool(a3["overall_protection_pass"])

    decision: Dict[str, Any] = {
        "status": "PASS",
        "eligible_candidates": ["A2", "A3"],
        "A2_protection_pass": safe2,
        "A3_protection_pass": safe3,
        "selected_architecture": None,
        "decision_rule": None,
        "delta_mean_uno_c_A3_minus_A2": float("nan"),
        "delta_common_mean_IBS_A3_minus_A2": float("nan"),
    }

    if safe2 and not safe3:
        decision["selected_architecture"] = "A2"
        decision["decision_rule"] = "ONLY_A2_PASSES_FROZEN_PROTECTION"
        return decision

    if safe3 and not safe2:
        decision["selected_architecture"] = "A3"
        decision["decision_rule"] = "ONLY_A3_PASSES_FROZEN_PROTECTION"
        return decision

    if not safe2 and not safe3:
        decision["status"] = "HOLD"
        decision["decision_rule"] = (
            "HOLD_NO_SAFE_SELECTABLE_ARCHITECTURE"
        )
        return decision

    c2 = float(a2["equal_scenario_mean_uno_c"])
    c3 = float(a3["equal_scenario_mean_uno_c"])
    i2 = float(
        a2["common_A2_A3_equal_scenario_mean_IBS"]
    )
    i3 = float(
        a3["common_A2_A3_equal_scenario_mean_IBS"]
    )

    d_c = c3 - c2
    d_i = i3 - i2

    decision["delta_mean_uno_c_A3_minus_A2"] = d_c
    decision["delta_common_mean_IBS_A3_minus_A2"] = d_i

    if not np.isfinite(d_c):
        decision["status"] = "HOLD"
        decision["decision_rule"] = "HOLD_SELECTION_UNOC_NOT_ASSESSABLE"
        return decision

    if abs(d_c) >= PRACTICAL_TIE_C:
        decision["selected_architecture"] = (
            "A3" if d_c > 0 else "A2"
        )
        decision["decision_rule"] = (
            "PRIMARY_UNO_C_DIFFERENCE_EXCEEDS_PRACTICAL_TIE"
        )
        return decision

    if not np.isfinite(d_i):
        decision["status"] = "HOLD"
        decision["decision_rule"] = (
            "HOLD_SELECTION_IBS_NOT_ASSESSABLE"
        )
        return decision

    if abs(d_i) >= PRACTICAL_TIE_IBS:
        decision["selected_architecture"] = (
            "A3" if d_i < 0 else "A2"
        )
        decision["decision_rule"] = (
            "UNOC_PRACTICAL_TIE_IBS_DIFFERENCE_EXCEEDS_TIE"
        )
        return decision

    decision["selected_architecture"] = "A2"
    decision["decision_rule"] = (
        "PRACTICAL_TIE_SELECT_SIMPLER_A2"
    )
    return decision


def compute_a5_shift_diagnostic(
    scenario_summary: pd.DataFrame,
    contrast_registry: pd.DataFrame,
) -> pd.DataFrame:
    a3 = scenario_summary[
        scenario_summary["model"] == "A3"
    ].set_index("scenario_id")

    rows = []

    for row in contrast_registry.itertuples(index=False):
        ref = str(row.reference_scenario_id)
        stress = str(row.stress_scenario_id)

        if ref not in a3.index or stress not in a3.index:
            raise RuntimeError(
                f"A5 contrast scenario missing from 05d summary: {ref} / {stress}"
            )

        ref_delta = float(a3.loc[ref, "mean_delta_c_vs_B0"])
        stress_delta = float(
            a3.loc[stress, "mean_delta_c_vs_B0"]
        )
        deterioration = ref_delta - stress_delta

        rows.append(
            {
                "axis": str(row.axis),
                "target_events": int(row.target_events),
                "matched_nuisance_level": str(
                    row.matched_nuisance_level
                ),
                "reference_scenario_id": ref,
                "stress_scenario_id": stress,
                "reference_mean_delta_c_A3_vs_B0": ref_delta,
                "stress_mean_delta_c_A3_vs_B0": stress_delta,
                "pair_deterioration": deterioration,
                "event_level_aggregation_weight": float(
                    row.event_level_aggregation_weight
                ),
            }
        )

    pair_df = pd.DataFrame(rows)

    event_rows = []

    for (axis, events), group in pair_df.groupby(
        ["axis", "target_events"],
        sort=True,
    ):
        weights = group[
            "event_level_aggregation_weight"
        ].to_numpy(dtype=float)
        values = group["pair_deterioration"].to_numpy(
            dtype=float
        )

        if not np.isfinite(values).all():
            event_deterioration = float("nan")
        else:
            weights = weights / weights.sum()
            event_deterioration = float(
                np.sum(weights * values)
            )

        event_rows.append(
            {
                "axis": axis,
                "target_events": int(events),
                "event_slice_deterioration": (
                    event_deterioration
                ),
                "trigger_hit_ge_0_02": bool(
                    np.isfinite(event_deterioration)
                    and event_deterioration >= 0.02
                ),
                "n_pair_contrasts": len(group),
            }
        )

    return pd.DataFrame(event_rows)


def decide_a5_branch(
    a5_shift: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    architecture_decision: Dict[str, Any],
) -> Dict[str, Any]:
    a3 = candidate_summary.set_index("model").loc["A3"]

    misleading_safety = bool(
        np.isfinite(a3["catastrophic_rate_misleading"])
        and np.isfinite(a3["false_borrow_rate_misleading"])
        and float(a3["catastrophic_rate_misleading"])
        <= MAX_CATASTROPHIC_RATE
        and float(a3["false_borrow_rate_misleading"])
        <= MAX_FALSE_BORROW_RATE
    )

    if a5_shift["event_slice_deterioration"].isna().any():
        return {
            "status": "HOLD",
            "branch_decision": (
                "HOLD_A5_BRANCH_DECISION_FOR_TECHNICAL_RESOLUTION"
            ),
            "reason": "At least one preregistered A5 shift metric is undefined.",
            "misleading_source_rejection_safety_pass": misleading_safety,
        }

    axis_hits = (
        a5_shift.groupby("axis")["trigger_hit_ge_0_02"]
        .sum()
        .astype(int)
        .to_dict()
    )

    shift_signature = any(
        int(v) >= 2
        for v in axis_hits.values()
    )

    selected = architecture_decision.get(
        "selected_architecture"
    )
    a3_not_selected = selected != "A3"

    if selected == "A3":
        return {
            "status": "PASS",
            "branch_decision": "CLOSE_A5_PRIMARY_A3_SUCCESS",
            "reason": (
                "A3 is the frozen selected architecture; 05c1 priority rule closes A5."
            ),
            "misleading_source_rejection_safety_pass": misleading_safety,
            "shift_specific_signature": shift_signature,
            "axis_hits": axis_hits,
            "A3_not_selected": False,
        }

    if (
        misleading_safety
        and shift_signature
        and a3_not_selected
    ):
        return {
            "status": "PASS",
            "branch_decision": "OPEN_A5",
            "reason": (
                "Prespecified same-axis shift-specific degradation occurred while "
                "A3 retained misleading-source rejection safety and A3 was not selected."
            ),
            "misleading_source_rejection_safety_pass": True,
            "shift_specific_signature": True,
            "axis_hits": axis_hits,
            "A3_not_selected": True,
        }

    return {
        "status": "PASS",
        "branch_decision": "CLOSE_A5",
        "reason": (
            "The conjunction required by the preregistered 05c1/05c1a OPEN rule "
            "was not satisfied."
        ),
        "misleading_source_rejection_safety_pass": misleading_safety,
        "shift_specific_signature": shift_signature,
        "axis_hits": axis_hits,
        "A3_not_selected": a3_not_selected,
    }


def main() -> None:
    started = now_utc()

    print("=" * 120)
    print("Paper 6 - aggregate frozen synthetic negative-transfer phase diagram")
    print("=" * 120)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Metric implementation: {METRIC_IMPLEMENTATION_VERSION}")
    print()
    print("Safety / scope:")
    print("  Real DOG2 outcome/expression values read: NO")
    print("  Real human outcome/expression values read: NO")
    print("  Reserved TARGET/GSE21257/GSE39055 outcomes read: NO")
    print("  Model fitting / architecture retuning: NO")
    print("  05c saved synthetic predictions read: YES")
    print("  Architecture selection: frozen A2-vs-A3 rule only")
    print("  A5 decision: frozen 05c1/05c1a rule only")
    print("  GPU execution: NO")
    print()

    for path in [
        A_CONTRACT,
        A_SCENARIOS,
        A_SELECTION_RULES,
        A_SUMMARY,
        C_SUMMARY,
        C_MANIFEST,
        C_IMPLEMENTATION,
        C0_SUMMARY,
        C1_CONTRACT,
        C1_SUMMARY,
        C1A_CONTRACT,
        C1A_CONTRASTS,
        C1A_SUMMARY,
    ]:
        require_file(path)

    contract_05a = read_json(A_CONTRACT)
    summary_05a = read_json(A_SUMMARY)
    summary_05c = read_json(C_SUMMARY)
    summary_05c0 = read_json(C0_SUMMARY)
    summary_05c1 = read_json(C1_SUMMARY)
    summary_05c1a = read_json(C1A_SUMMARY)

    if clean(summary_05a.get("scientific_status")) != EXPECTED_05A_STATUS:
        raise RuntimeError("05a is not in expected PASS state.")
    if clean(summary_05c.get("scientific_status")) != EXPECTED_05C_STATUS:
        raise RuntimeError("05c is not in expected PASS state.")
    if clean(summary_05c0.get("scientific_status")) != EXPECTED_05C0_STATUS:
        raise RuntimeError("05c0 is not in expected PASS state.")
    if clean(summary_05c1.get("scientific_status")) != EXPECTED_05C1_STATUS:
        raise RuntimeError("05c1 is not in expected PASS state.")
    if clean(summary_05c1a.get("scientific_status")) != EXPECTED_05C1A_STATUS:
        raise RuntimeError("05c1a is not in expected PASS state.")

    verify_frozen_thresholds(contract_05a)

    scenarios = pd.read_csv(A_SCENARIOS, sep="\t")
    manifest = pd.read_csv(C_MANIFEST, sep="\t")
    contrast_registry = pd.read_csv(C1A_CONTRASTS, sep="\t")

    if len(scenarios) != EXPECTED_SCENARIOS:
        raise RuntimeError("05a scenario count changed.")
    if len(manifest) != EXPECTED_SCENARIOS:
        raise RuntimeError("05c manifest scenario count changed.")
    if int(manifest["replicates"].sum()) != EXPECTED_REPLICATES:
        raise RuntimeError("05c manifest replicate count changed.")

    # Freeze/verify the metric implementation BEFORE opening scenario outputs.
    metric_payload = metric_contract_payload(
        Path(__file__).resolve()
    )
    freeze_or_verify_metric_contract(metric_payload)
    metric_contract_hash = sha256_file(METRIC_CONTRACT)

    print(f"05d metric contract SHA256: {metric_contract_hash}")
    print("Metric implementation frozen before scenario read: PASS")
    print()

    manifest_by_id = {
        str(row.scenario_id): row
        for row in manifest.itertuples(index=False)
    }

    metric_manifest_rows: List[Dict[str, Any]] = []

    total = len(scenarios)
    reused = 0
    computed = 0
    wall_start = time.time()

    for i, scenario in scenarios.reset_index(drop=True).iterrows():
        scenario_id = str(scenario["scenario_id"])

        source_row = manifest_by_id.get(scenario_id)
        if source_row is None:
            raise RuntimeError(
                f"{scenario_id}: no 05c source manifest row."
            )

        source_path = ROOT / str(source_row.output_path)
        require_file(source_path)

        source_hash = sha256_file(source_path)
        if source_hash != str(source_row.output_sha256):
            raise RuntimeError(
                f"{scenario_id}: 05c source output SHA256 mismatch."
            )

        metric_path, metric_manifest_path = (
            scenario_metric_paths(scenario_id)
        )

        existing = load_valid_metric_checkpoint(
            scenario_id,
            source_hash,
            metric_contract_hash,
        )

        if existing is not None:
            reused += 1
            elapsed = float(existing.get("elapsed_seconds", 0.0))
        else:
            scenario_start = time.time()

            result = evaluate_scenario(source_path)
            atomic_savez_compressed(metric_path, **result)

            elapsed = time.time() - scenario_start

            scenario_manifest = {
                "status": "PASS",
                "created_utc": now_utc(),
                "scenario_id": scenario_id,
                "replicates": int(scenario["replicates"]),
                "source_output_path": str(source_path.relative_to(ROOT)),
                "source_output_sha256": source_hash,
                "metric_contract_sha256": metric_contract_hash,
                "metric_output_path": str(metric_path.relative_to(ROOT)),
                "metric_output_sha256": sha256_file(metric_path),
                "metric_output_size_bytes": metric_path.stat().st_size,
                "elapsed_seconds": elapsed,
            }
            write_json(metric_manifest_path, scenario_manifest)
            computed += 1

        manifest_payload = read_json(metric_manifest_path)

        metric_manifest_rows.append(
            {
                "scenario_id": scenario_id,
                "replicates": int(scenario["replicates"]),
                "status": "REUSED" if existing is not None else "COMPUTED",
                "metric_output_path": str(metric_path.relative_to(ROOT)),
                "metric_output_sha256": sha256_file(metric_path),
                "elapsed_seconds": elapsed,
            }
        )

        done = i + 1
        if existing is None:
            print(
                f"  {scenario_id}: PASS metrics "
                f"[{int(scenario['replicates'])} reps, {elapsed:.1f}s] "
                f"overall {done}/{total}"
            )

        if done % 10 == 0 or done == total:
            total_elapsed = time.time() - wall_start
            rate = done / max(total_elapsed, 1e-9)
            remaining = total - done
            eta_min = remaining / max(rate, 1e-9) / 60.0
            print(
                f"    progress: reused={reused}, computed={computed}, "
                f"approx remaining={eta_min:.1f} min"
            )

    metric_manifest_df = pd.DataFrame(metric_manifest_rows)
    metric_manifest_df.to_csv(
        SCENARIO_METRIC_MANIFEST,
        sep="\t",
        index=False,
    )

    # ------------------------------------------------------------------
    # Aggregate all scenario checkpoints.
    # ------------------------------------------------------------------
    scenario_rows: List[Dict[str, Any]] = []
    common_ibs_rows: List[Dict[str, Any]] = []

    scenario_by_id = scenarios.set_index("scenario_id")

    for row in metric_manifest_df.itertuples(index=False):
        scenario_id = str(row.scenario_id)
        metric_path = ROOT / str(row.metric_output_path)

        rows, common = aggregate_scenario(
            scenario_by_id.loc[scenario_id],
            metric_path,
        )
        scenario_rows.extend(rows)
        common_ibs_rows.append(common)

    scenario_summary = pd.DataFrame(scenario_rows)
    common_ibs = pd.DataFrame(common_ibs_rows)

    scenario_summary.to_csv(
        SCENARIO_SUMMARY,
        sep="\t",
        index=False,
    )

    # Model aggregate summary.
    model_rows = []

    for model in ALL_MODELS:
        part = scenario_summary[
            scenario_summary["model"] == model
        ]

        model_rows.append(
            {
                "model": model,
                "n_scenarios": len(part),
                "equal_scenario_mean_uno_c": (
                    equal_scenario_mean(part, "mean_uno_c")
                ),
                "equal_scenario_mean_delta_c_vs_B0": (
                    equal_scenario_mean(
                        part,
                        "mean_delta_c_vs_B0",
                    )
                ),
                "equal_scenario_negative_transfer_rate": (
                    equal_scenario_mean(
                        part,
                        "negative_transfer_rate",
                    )
                ),
                "equal_scenario_catastrophic_rate": (
                    equal_scenario_mean(
                        part,
                        "catastrophic_negative_transfer_rate",
                    )
                ),
                "equal_scenario_mean_IBS": (
                    equal_scenario_mean(part, "mean_ibs")
                ),
                "median_scenario_calibration_slope": (
                    finite_median(
                        pd.to_numeric(
                            part["median_calibration_slope"],
                            errors="coerce",
                        ).to_numpy(dtype=float)
                    )
                ),
                "median_scenario_abs_calibration_intercept": (
                    finite_median(
                        pd.to_numeric(
                            part[
                                "median_abs_calibration_intercept"
                            ],
                            errors="coerce",
                        ).to_numpy(dtype=float)
                    )
                ),
                "equal_scenario_module_recovery_AUROC": (
                    equal_scenario_mean(
                        part,
                        "mean_module_recovery_auc",
                    )
                ),
                "equal_scenario_false_borrow_rate": (
                    equal_scenario_mean(
                        part,
                        "mean_false_borrow_rate",
                    )
                ),
            }
        )

    model_summary = pd.DataFrame(model_rows)
    model_summary.to_csv(
        MODEL_SUMMARY,
        sep="\t",
        index=False,
    )

    # Event-scaled learning curves from CORE only.
    core = scenario_summary[
        scenario_summary["family"] == "CORE_PHASE_DIAGRAM"
    ].copy()

    learning = (
        core.groupby(
            ["target_events", "model"],
            as_index=False,
        )
        .agg(
            mean_uno_c=("mean_uno_c", "mean"),
            mean_delta_c_vs_B0=("mean_delta_c_vs_B0", "mean"),
            negative_transfer_rate=("negative_transfer_rate", "mean"),
            catastrophic_negative_transfer_rate=(
                "catastrophic_negative_transfer_rate",
                "mean",
            ),
            mean_ibs=("mean_ibs", "mean"),
        )
        .sort_values(["target_events", "model"])
    )
    learning.to_csv(
        LEARNING_CURVES,
        sep="\t",
        index=False,
    )

    # A3 ablation summary.
    abl = model_summary[
        model_summary["model"].isin(["A3"] + ABLATIONS)
    ].copy()
    abl.to_csv(
        ABLATION_SUMMARY,
        sep="\t",
        index=False,
    )

    # A3 module recovery by transport regime.
    gate_rows = []

    for (model, regime), part in scenario_summary[
        scenario_summary["model"].isin(list(GATE_KEYS))
    ].groupby(["model", "transfer_regime"]):
        gate_rows.append(
            {
                "model": model,
                "transfer_regime": regime,
                "n_scenarios": len(part),
                "mean_module_recovery_AUROC": (
                    equal_scenario_mean(
                        part,
                        "mean_module_recovery_auc",
                    )
                ),
                "mean_false_borrow_rate": (
                    equal_scenario_mean(
                        part,
                        "mean_false_borrow_rate",
                    )
                ),
                "mean_gate_true": (
                    equal_scenario_mean(
                        part,
                        "mean_gate_true",
                    )
                ),
                "mean_gate_false": (
                    equal_scenario_mean(
                        part,
                        "mean_gate_false",
                    )
                ),
            }
        )

    module_summary = pd.DataFrame(gate_rows)
    module_summary.to_csv(
        MODULE_SUMMARY,
        sep="\t",
        index=False,
    )

    # Frozen A2/A3 selection.
    candidate_summary = build_candidate_summary(
        scenario_summary,
        common_ibs,
    )
    candidate_summary.to_csv(
        CANDIDATE_SUMMARY,
        sep="\t",
        index=False,
    )

    architecture_decision = select_architecture(
        candidate_summary
    )
    architecture_decision.update(
        {
            "created_utc": now_utc(),
            "05a_contract_sha256": sha256_file(A_CONTRACT),
            "05d_metric_contract_sha256": metric_contract_hash,
            "candidate_summary_sha256": sha256_file(
                CANDIDATE_SUMMARY
            ),
            "GSE16091_used_for_selection": False,
            "reserved_human_outcomes_read": False,
        }
    )
    write_json(
        ARCHITECTURE_DECISION,
        architecture_decision,
    )

    # A5 preregistered branch rule.
    a5_shift = compute_a5_shift_diagnostic(
        scenario_summary,
        contrast_registry,
    )
    a5_shift.to_csv(
        A5_SHIFT_DIAGNOSTIC,
        sep="\t",
        index=False,
    )

    a5_decision = decide_a5_branch(
        a5_shift,
        candidate_summary,
        architecture_decision,
    )
    a5_decision.update(
        {
            "created_utc": now_utc(),
            "05c1_contract_sha256": sha256_file(
                C1_CONTRACT
            ),
            "05c1a_contract_sha256": sha256_file(
                C1A_CONTRACT
            ),
            "A5_shift_diagnostic_sha256": sha256_file(
                A5_SHIFT_DIAGNOSTIC
            ),
            "GSE16091_allowed_for_A5": False,
        }
    )
    write_json(A5_DECISION, a5_decision)

    # ------------------------------------------------------------------
    # Final status.
    # ------------------------------------------------------------------
    selected = architecture_decision.get(
        "selected_architecture"
    )

    if architecture_decision["status"] != "PASS":
        scientific_status = architecture_decision[
            "decision_rule"
        ]
        overall_status = "HOLD"
    elif a5_decision["status"] != "PASS":
        scientific_status = (
            "HOLD_A5_BRANCH_TECHNICAL_METRIC_UNRESOLVED"
        )
        overall_status = "HOLD"
    else:
        scientific_status = (
            f"PASS_FROZEN_AI_SELECTED_{selected}_"
            f"{a5_decision['branch_decision']}"
        )
        overall_status = "PASS"

    summary = {
        "script_version": SCRIPT_VERSION,
        "metric_implementation_version": METRIC_IMPLEMENTATION_VERSION,
        "status": overall_status,
        "scientific_status": scientific_status,
        "run_started_utc": started,
        "run_finished_utc": now_utc(),
        "n_scenarios": EXPECTED_SCENARIOS,
        "total_replicates": EXPECTED_REPLICATES,
        "selected_architecture": selected,
        "architecture_decision_rule": architecture_decision[
            "decision_rule"
        ],
        "A5_branch_decision": a5_decision[
            "branch_decision"
        ],
        "real_data_values_read": False,
        "reserved_human_outcomes_read": False,
        "GSE16091_used_for_selection": False,
        "model_fitting": False,
        "GPU_execution": False,
        "scenario_metric_checkpoints_reused_this_run": reused,
        "scenario_metric_checkpoints_computed_this_run": computed,
        "final_artifact_hashes": {
            "metric_implementation_contract_json": sha256_file(
                METRIC_CONTRACT
            ),
            "scenario_metric_manifest_tsv": sha256_file(
                SCENARIO_METRIC_MANIFEST
            ),
            "scenario_model_metric_summary_tsv": sha256_file(
                SCENARIO_SUMMARY
            ),
            "model_aggregate_summary_tsv": sha256_file(
                MODEL_SUMMARY
            ),
            "A2_A3_frozen_selection_summary_tsv": sha256_file(
                CANDIDATE_SUMMARY
            ),
            "event_scaled_learning_curves_tsv": sha256_file(
                LEARNING_CURVES
            ),
            "A3_ablation_summary_tsv": sha256_file(
                ABLATION_SUMMARY
            ),
            "A3_module_recovery_summary_tsv": sha256_file(
                MODULE_SUMMARY
            ),
            "A5_shift_trigger_diagnostics_tsv": sha256_file(
                A5_SHIFT_DIAGNOSTIC
            ),
            "frozen_architecture_selection_json": sha256_file(
                ARCHITECTURE_DECISION
            ),
            "A5_branch_decision_json": sha256_file(
                A5_DECISION
            ),
        },
        "next_if_PASS": (
            "Freeze the selected A2/A3 real-data implementation/hyperparameters, "
            "build the outcome-blind evolutionary conservation prior, and only "
            "then proceed toward the reserved human transfer gate. If A5=OPEN, "
            "run the separately preregistered A5 identifiability/simulation branch "
            "before any A5 real-human evaluation."
        ),
    }
    write_json(SUMMARY_JSON, summary)

    print()
    print("=" * 120)
    print("05d FROZEN A2-vs-A3 SELECTION")
    print("=" * 120)
    print(
        candidate_summary[
            [
                "model",
                "aggregate_negative_transfer_rate",
                "catastrophic_rate_misleading",
                "catastrophic_rate_severe_shift",
                "module_recovery_AUROC_partial",
                "false_borrow_rate_misleading",
                "equal_scenario_mean_uno_c",
                "common_A2_A3_equal_scenario_mean_IBS",
                "overall_protection_pass",
            ]
        ].to_string(index=False)
    )
    print()
    print(
        f"Selected architecture: "
        f"{architecture_decision.get('selected_architecture')}"
    )
    print(
        f"Selection rule: "
        f"{architecture_decision.get('decision_rule')}"
    )

    print()
    print("=" * 120)
    print("05d PREREGISTERED A5 BRANCH DECISION")
    print("=" * 120)
    print(a5_shift.to_string(index=False))
    print()
    print(
        f"A5 branch: {a5_decision['branch_decision']}"
    )
    print(f"Reason: {a5_decision['reason']}")

    print()
    print("=" * 120)
    print("05d NEGATIVE-TRANSFER PHASE DIAGRAM SUMMARY")
    print("=" * 120)
    print(f"Status: {overall_status}")
    print(f"Scientific status: {scientific_status}")
    print(f"Selected AI: {selected}")
    print(f"A5: {a5_decision['branch_decision']}")
    print("Real human outcomes read: NO")
    print("Model fitting in 05d: NO")
    print("GPU execution: NO")
    print("=" * 120)

    if overall_status != "PASS":
        raise RuntimeError(scientific_status)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 120, file=sys.stderr)
        print(
            "05d negative-transfer phase diagram / architecture selection: FAIL",
            file=sys.stderr,
        )
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print("=" * 120, file=sys.stderr)
        raise
