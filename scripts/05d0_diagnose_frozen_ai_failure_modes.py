#!/usr/bin/env python3
"""
Paper 6 - bounded post-05d diagnostic of frozen AI failure modes.

Scientific status entering this stage:
    05d = HOLD_NO_SAFE_SELECTABLE_ARCHITECTURE
    A5  = CLOSE_A5

This script is deliberately POST-RESULT and DIAGNOSTIC.
It does NOT create a new selection rule, retune any model, change any threshold,
open any human outcome, or rescue A2/A3.

Primary questions
-----------------
1. Is the 05d HOLD reproducible directly from the saved per-replicate metrics?
2. Is there any structural implementation invariant failure in the frozen 05c/05d
   artifacts that could explain the result?
3. Where does A2/A3 negative transfer concentrate (events, transfer regime,
   prior, covariance/mapping shift, censoring)?
4. Why can A3 recover broad transportability (AUROC) yet fail prediction safety?
   In particular, does the gate borrow the *causal harmful/sign-flipped* modules
   more often than suggested by the all-module false-borrow metric?
5. Which frozen A3 ablations help or hurt in the failure regimes?
6. Do simpler frozen comparators show the same failure topology?

Important distinction
---------------------
The existing 05d gate-recovery metric treats all 50 modules equally through
transportable_mask. Prediction harm, however, is generated only by modules that
actually carry source prognostic effect and may be attenuated/reversed in the
target. This diagnostic therefore adds POST-HOC explanatory summaries for:
    - causal & transportable modules;
    - causal, nontransportable, sign-flipped modules;
    - causal, nontransportable, attenuated-to-zero modules.
These summaries CANNOT alter the frozen 05d decision.

No model fitting.
No tuning.
No GPU.
No real data.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import roc_auc_score
except ImportError as exc:
    raise ImportError("05d0 requires scikit-learn.") from exc


SCRIPT_VERSION = "05d0-diagnose-frozen-ai-failure-modes-v1-no-cli"

ROOT = Path(__file__).resolve().parents[1]

# Frozen simulation registry.
A_DIR = ROOT / "results" / "simulation_contract" / "05a"
A_SCENARIOS = A_DIR / "simulation_scenario_registry.tsv"
A_CONTRACT = A_DIR / "negative_transfer_simulation_contract.json"

# Frozen model outputs.
C_DIR = ROOT / "results" / "simulation_model_matrix" / "05c"
C_MANIFEST = C_DIR / "scenario_output_manifest.tsv"
C_SUMMARY = C_DIR / "summary.json"

# Execution audit.
C0_DIR = ROOT / "results" / "simulation_model_matrix_diagnostics" / "05c0"
C0_SUMMARY = C0_DIR / "summary.json"

# Frozen 05d scientific outputs.
D_DIR = ROOT / "results" / "simulation_phase_diagram" / "05d"
D_METRIC_CONTRACT = D_DIR / "metric_implementation_contract.json"
D_METRIC_MANIFEST = D_DIR / "scenario_metric_manifest.tsv"
D_METRIC_DIR = D_DIR / "scenario_metrics"
D_SCENARIO_SUMMARY = D_DIR / "scenario_model_metric_summary.tsv"
D_MODEL_SUMMARY = D_DIR / "model_aggregate_summary.tsv"
D_CANDIDATE_SUMMARY = D_DIR / "A2_A3_frozen_selection_summary.tsv"
D_ARCH_DECISION = D_DIR / "frozen_architecture_selection.json"
D_A5_DECISION = D_DIR / "A5_branch_decision.json"
D_SUMMARY = D_DIR / "summary.json"

OUT_DIR = ROOT / "results" / "simulation_phase_diagram_diagnostics" / "05d0"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INVARIANT_AUDIT = OUT_DIR / "implementation_invariant_audit.tsv"
SELECTION_REPLAY = OUT_DIR / "independent_selection_metric_replay.tsv"
FAILURE_EVENT_REGIME = OUT_DIR / "failure_by_event_and_transfer_regime.tsv"
FAILURE_PRIOR_REGIME = OUT_DIR / "failure_by_prior_and_transfer_regime.tsv"
FAILURE_STRESS = OUT_DIR / "failure_by_stress_axis.tsv"
MODEL_COMPARATOR_SUMMARY = OUT_DIR / "all_model_failure_summary.tsv"
ABLATION_SUMMARY = OUT_DIR / "A3_ablation_failure_contrasts.tsv"
GATE_HARM_SCENARIO = OUT_DIR / "A3_gate_harmful_module_diagnostics_by_scenario.tsv"
GATE_COUPLING = OUT_DIR / "A3_gate_prediction_coupling.tsv"
GATE_HARM_GROUPED = OUT_DIR / "A3_gate_harmful_module_grouped_summary.tsv"
SUMMARY_JSON = OUT_DIR / "summary.json"

EXPECTED_SCENARIOS = 180
EXPECTED_REPLICATES = 21600
EXPECTED_MODULES = 50

EXPECTED_05C_STATUS = "PASS_CLOSED_SYNTHETIC_TRANSFER_MODEL_MATRIX_COMPLETE"
EXPECTED_05C0_STATUS = "PASS_SYNTHETIC_MODEL_MATRIX_EXECUTION_COMPLETE"
EXPECTED_05D_STATUS = "HOLD_NO_SAFE_SELECTABLE_ARCHITECTURE"

MAIN_MODELS = ["B0", "B4", "A0", "A1", "A2", "A3", "A4"]
ABLATIONS = [
    "A3_NO_EVOLUTION_PRIOR",
    "A3_NO_TARGET_OVERRIDE",
    "A3_PERMUTED_PRIOR",
    "A3_MAPPING_PERMUTATION",
    "A3_SOURCE_OUTCOME_PERMUTATION",
]
ALL_MODELS = MAIN_MODELS + ABLATIONS

MODEL_INDEX = {name: i for i, name in enumerate(ALL_MODELS)}
A3_GATE_INDEX = 0  # 05d gate model order starts with A3.

NEGATIVE_THRESHOLD = -0.02
CATASTROPHIC_THRESHOLD = -0.05

FROZEN_MAX_NEGATIVE_RATE = 0.10
FROZEN_MAX_CATASTROPHIC_RATE = 0.05
FROZEN_MIN_MODULE_AUC = 0.70
FROZEN_MAX_FALSE_BORROW = 0.20

NUM_TOL = 2e-6
GATE_TOL = 2e-6

# Post-hoc descriptive gate threshold; identical to frozen 05d false-borrow definition.
GATE_BORROW_THRESHOLD = 0.50


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_file(path: Path) -> Path:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Required artifact missing: {path}")
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


def finite_mean(values: Iterable[float]) -> float:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if len(x) else float("nan")


def finite_median(values: Iterable[float]) -> float:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else float("nan")


def rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)

    x = x[keep]
    y = y[keep]

    if len(x) < 5:
        return float("nan")
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")

    rx = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    ry = pd.Series(y).rank(method="average").to_numpy(dtype=float)

    if np.std(rx) <= 1e-12 or np.std(ry) <= 1e-12:
        return float("nan")

    return float(np.corrcoef(rx, ry)[0, 1])


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)

    if len(np.unique(labels)) < 2:
        return float("nan")
    if not np.isfinite(scores).all():
        return float("nan")

    try:
        return float(roc_auc_score(labels, scores))
    except Exception:
        return float("nan")


def exact_severe_mask(frame: pd.DataFrame) -> pd.Series:
    return (
        (frame["covariance_shift"].astype(str) == "S2_SEVERE")
        | (frame["mapping_error"].astype(str) == "M2_SEVERE")
    )


def compare_numeric(
    observed: float,
    expected: float,
    *,
    atol: float = NUM_TOL,
) -> bool:
    if not np.isfinite(observed) and not np.isfinite(expected):
        return True
    return bool(np.isclose(observed, expected, rtol=0, atol=atol))


def main() -> None:
    started = now_utc()

    print("=" * 120)
    print("Paper 6 - bounded post-05d frozen-AI failure-mode diagnostic")
    print("=" * 120)
    print(f"Script version: {SCRIPT_VERSION}")
    print()
    print("Scope:")
    print("  Model fitting: NO")
    print("  Hyperparameter tuning / rescue: NO")
    print("  Frozen 05d decision changed: NO")
    print("  Real DOG2/human values read: NO")
    print("  Reserved human outcomes read: NO")
    print("  GPU execution: NO")
    print("  Post-result explanatory diagnostics: YES")
    print()

    for path in [
        A_SCENARIOS,
        A_CONTRACT,
        C_MANIFEST,
        C_SUMMARY,
        C0_SUMMARY,
        D_METRIC_CONTRACT,
        D_METRIC_MANIFEST,
        D_SCENARIO_SUMMARY,
        D_MODEL_SUMMARY,
        D_CANDIDATE_SUMMARY,
        D_ARCH_DECISION,
        D_A5_DECISION,
        D_SUMMARY,
    ]:
        require_file(path)

    scenarios = pd.read_csv(A_SCENARIOS, sep="\t")
    c_manifest = pd.read_csv(C_MANIFEST, sep="\t")
    d_metric_manifest = pd.read_csv(D_METRIC_MANIFEST, sep="\t")
    d_scenario = pd.read_csv(D_SCENARIO_SUMMARY, sep="\t")
    d_candidate = pd.read_csv(D_CANDIDATE_SUMMARY, sep="\t")

    c_summary = read_json(C_SUMMARY)
    c0_summary = read_json(C0_SUMMARY)
    d_summary = read_json(D_SUMMARY)
    arch = read_json(D_ARCH_DECISION)
    a5 = read_json(D_A5_DECISION)

    if str(c_summary.get("scientific_status")) != EXPECTED_05C_STATUS:
        raise RuntimeError("05c is not in expected PASS state.")
    if str(c0_summary.get("scientific_status")) != EXPECTED_05C0_STATUS:
        raise RuntimeError("05c0 is not in expected PASS state.")
    if str(d_summary.get("scientific_status")) != EXPECTED_05D_STATUS:
        raise RuntimeError(
            f"05d scientific status is {d_summary.get('scientific_status')!r}, "
            f"expected {EXPECTED_05D_STATUS!r}."
        )
    if arch.get("selected_architecture") is not None:
        raise RuntimeError("05d unexpectedly selected an architecture.")
    if str(a5.get("branch_decision")) != "CLOSE_A5":
        raise RuntimeError("A5 is not in expected CLOSE state.")

    if len(scenarios) != EXPECTED_SCENARIOS:
        raise RuntimeError("Scenario registry is not 180 rows.")
    if int(scenarios["replicates"].astype(int).sum()) != EXPECTED_REPLICATES:
        raise RuntimeError("Frozen replicate total changed.")
    if len(c_manifest) != EXPECTED_SCENARIOS:
        raise RuntimeError("05c manifest is not 180 rows.")
    if len(d_metric_manifest) != EXPECTED_SCENARIOS:
        raise RuntimeError("05d metric manifest is not 180 rows.")

    # Verify the scientific hashes that matter here. We intentionally do not use
    # the technical-recovery JSON hash as a scientific dependency.
    final_hashes = d_summary.get("final_artifact_hashes") or {}
    scientific_hash_checks = {
        "scenario_model_metric_summary_tsv": D_SCENARIO_SUMMARY,
        "A2_A3_frozen_selection_summary_tsv": D_CANDIDATE_SUMMARY,
        "frozen_architecture_selection_json": D_ARCH_DECISION,
        "A5_branch_decision_json": D_A5_DECISION,
    }

    for key, path in scientific_hash_checks.items():
        expected = str(final_hashes.get(key, ""))
        if not expected:
            raise RuntimeError(f"05d summary lacks scientific artifact hash {key}.")
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(
                f"Scientific 05d artifact hash mismatch for {key}: {path}"
            )

    scenario_by_id = scenarios.set_index("scenario_id", drop=False)
    c_by_id = {
        str(row.scenario_id): row
        for row in c_manifest.itertuples(index=False)
    }
    dmetric_by_id = {
        str(row.scenario_id): row
        for row in d_metric_manifest.itertuples(index=False)
    }

    invariant_rows: List[Dict[str, Any]] = []
    replay_scenario_rows: List[Dict[str, Any]] = []
    gate_scenario_rows: List[Dict[str, Any]] = []
    coupling_rows: List[Dict[str, Any]] = []

    # For all-model post-hoc comparator replay.
    model_scenario_rows: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Read each frozen scenario exactly once.
    # ------------------------------------------------------------------
    for i, scenario_row in enumerate(scenarios.itertuples(index=False), start=1):
        scenario_id = str(scenario_row.scenario_id)
        scenario = scenario_by_id.loc[scenario_id]

        c_row = c_by_id.get(scenario_id)
        d_row = dmetric_by_id.get(scenario_id)

        if c_row is None or d_row is None:
            raise RuntimeError(f"{scenario_id}: missing source/metric manifest row.")

        c_path = ROOT / str(c_row.output_path)
        d_path = ROOT / str(d_row.metric_output_path)

        require_file(c_path)
        require_file(d_path)

        if sha256_file(c_path) != str(c_row.output_sha256):
            raise RuntimeError(f"{scenario_id}: 05c output hash mismatch.")
        if sha256_file(d_path) != str(d_row.metric_output_sha256):
            raise RuntimeError(f"{scenario_id}: 05d metric hash mismatch.")

        with np.load(c_path, allow_pickle=False) as cdata, np.load(
            d_path, allow_pickle=False
        ) as ddata:
            seeds_c = np.asarray(cdata["replicate_seed"], dtype=np.int64)
            seeds_d = np.asarray(ddata["replicate_seed"], dtype=np.int64)

            if not np.array_equal(seeds_c, seeds_d):
                raise RuntimeError(f"{scenario_id}: replicate seed order mismatch.")

            n_rep = len(seeds_c)
            p = EXPECTED_MODULES

            causal = np.asarray(cdata["causal_mask"], dtype=bool)
            transportable = np.asarray(cdata["transportable_mask"], dtype=bool)
            prior = np.asarray(cdata["prior_score"], dtype=float)
            mapping = np.asarray(cdata["mapping_index"], dtype=int)
            beta_source = np.asarray(cdata["beta_source_truth"], dtype=float)
            beta_target = np.asarray(cdata["beta_target_truth"], dtype=float)

            gate = np.asarray(cdata["gate_A3"], dtype=float)
            gate_no_override = np.asarray(
                cdata["gate_A3_NO_TARGET_OVERRIDE"],
                dtype=float,
            )

            uno = np.asarray(ddata["uno_c"], dtype=float)
            delta = np.asarray(ddata["delta_c_vs_B0"], dtype=float)
            neg = np.asarray(ddata["negative_transfer"], dtype=np.uint8)
            cat = np.asarray(
                ddata["catastrophic_negative_transfer"],
                dtype=np.uint8,
            )
            module_auc_saved = np.asarray(
                ddata["module_recovery_auc"],
                dtype=float,
            )
            false_borrow_saved = np.asarray(
                ddata["false_borrow_rate"],
                dtype=float,
            )

            # ----------------------------------------------------------
            # Structural invariants.
            # ----------------------------------------------------------
            b0_idx = MODEL_INDEX["B0"]
            b0_delta_max = float(np.nanmax(np.abs(delta[:, b0_idx])))
            b0_neg_n = int(np.sum(neg[:, b0_idx]))
            b0_cat_n = int(np.sum(cat[:, b0_idx]))

            invariant_rows.append(
                {
                    "scenario_id": scenario_id,
                    "check": "B0_DELTA_C_EXACT_ZERO",
                    "observed": b0_delta_max,
                    "expected": 0.0,
                    "pass": b0_delta_max <= 1e-7,
                }
            )
            invariant_rows.append(
                {
                    "scenario_id": scenario_id,
                    "check": "B0_NEGATIVE_FLAGS_ZERO",
                    "observed": b0_neg_n,
                    "expected": 0,
                    "pass": b0_neg_n == 0,
                }
            )
            invariant_rows.append(
                {
                    "scenario_id": scenario_id,
                    "check": "B0_CATASTROPHIC_FLAGS_ZERO",
                    "observed": b0_cat_n,
                    "expected": 0,
                    "pass": b0_cat_n == 0,
                }
            )

            # Frozen A3 no-target-override semantics: gate == prior exactly up to float32.
            gate_prior_maxdiff = float(
                np.max(np.abs(gate_no_override - prior))
            )
            invariant_rows.append(
                {
                    "scenario_id": scenario_id,
                    "check": "A3_NO_TARGET_OVERRIDE_GATE_EQUALS_PRIOR",
                    "observed": gate_prior_maxdiff,
                    "expected": 0.0,
                    "pass": gate_prior_maxdiff <= GATE_TOL,
                }
            )

            expected_transportable = int(
                round(float(scenario["transferable_fraction"]) * p)
            )
            observed_transportable = transportable.sum(axis=1)
            invariant_rows.append(
                {
                    "scenario_id": scenario_id,
                    "check": "TRANSPORTABLE_MASK_COUNT",
                    "observed": (
                        f"{int(observed_transportable.min())}-"
                        f"{int(observed_transportable.max())}"
                    ),
                    "expected": expected_transportable,
                    "pass": bool(
                        np.all(observed_transportable == expected_transportable)
                    ),
                }
            )

            identity = np.arange(p)[None, :]
            observed_mapping_changed = np.sum(
                mapping != identity,
                axis=1,
            )
            expected_mapping_changed = int(
                round(float(scenario["mapping_error_fraction"]) * p)
            )
            invariant_rows.append(
                {
                    "scenario_id": scenario_id,
                    "check": "MAPPING_CORRUPTION_COUNT",
                    "observed": (
                        f"{int(observed_mapping_changed.min())}-"
                        f"{int(observed_mapping_changed.max())}"
                    ),
                    "expected": expected_mapping_changed,
                    "pass": bool(
                        np.all(
                            observed_mapping_changed
                            == expected_mapping_changed
                        )
                    ),
                }
            )

            # Recompute frozen A3 broad gate metrics from 05c truth/gates and
            # compare against 05d stored metrics.
            recomputed_auc = np.full(n_rep, np.nan, dtype=float)
            recomputed_false = np.full(n_rep, np.nan, dtype=float)

            for r in range(n_rep):
                labels = transportable[r].astype(int)
                if len(np.unique(labels)) >= 2:
                    recomputed_auc[r] = safe_auc(labels, gate[r])

                false_mask = ~transportable[r]
                if np.any(false_mask):
                    recomputed_false[r] = float(
                        np.mean(
                            gate[r, false_mask] >= GATE_BORROW_THRESHOLD
                        )
                    )

            saved_auc = module_auc_saved[:, A3_GATE_INDEX]
            saved_false = false_borrow_saved[:, A3_GATE_INDEX]

            auc_diff = np.abs(recomputed_auc - saved_auc)
            auc_diff = auc_diff[np.isfinite(auc_diff)]
            false_diff = np.abs(recomputed_false - saved_false)
            false_diff = false_diff[np.isfinite(false_diff)]

            max_auc_diff = (
                float(np.max(auc_diff)) if len(auc_diff) else 0.0
            )
            max_false_diff = (
                float(np.max(false_diff)) if len(false_diff) else 0.0
            )

            invariant_rows.append(
                {
                    "scenario_id": scenario_id,
                    "check": "A3_MODULE_AUC_RECOMPUTE_MATCH",
                    "observed": max_auc_diff,
                    "expected": 0.0,
                    "pass": max_auc_diff <= GATE_TOL,
                }
            )
            invariant_rows.append(
                {
                    "scenario_id": scenario_id,
                    "check": "A3_FALSE_BORROW_RECOMPUTE_MATCH",
                    "observed": max_false_diff,
                    "expected": 0.0,
                    "pass": max_false_diff <= GATE_TOL,
                }
            )

            # ----------------------------------------------------------
            # Independent replay of selection inputs.
            # ----------------------------------------------------------
            for model in ["A2", "A3"]:
                idx = MODEL_INDEX[model]
                valid = np.isfinite(uno[:, idx])

                replay_scenario_rows.append(
                    {
                        "scenario_id": scenario_id,
                        "family": str(scenario["family"]),
                        "target_events": int(scenario["target_events"]),
                        "transfer_regime": str(scenario["transfer_regime"]),
                        "covariance_shift": str(scenario["covariance_shift"]),
                        "mapping_error": str(scenario["mapping_error"]),
                        "censoring": str(scenario["censoring"]),
                        "source_prior_state": str(
                            scenario["source_prior_state"]
                        ),
                        "model": model,
                        "mean_uno_c": finite_mean(uno[:, idx]),
                        "mean_delta_c_vs_B0": finite_mean(delta[:, idx]),
                        "negative_transfer_rate": (
                            float(np.mean(neg[valid, idx]))
                            if np.any(valid)
                            else float("nan")
                        ),
                        "catastrophic_rate": (
                            float(np.mean(cat[valid, idx]))
                            if np.any(valid)
                            else float("nan")
                        ),
                        "mean_module_auc": (
                            finite_mean(saved_auc)
                            if model == "A3"
                            else float("nan")
                        ),
                        "mean_false_borrow": (
                            finite_mean(saved_false)
                            if model == "A3"
                            else float("nan")
                        ),
                    }
                )

            # All frozen model branches for descriptive comparison.
            for model in ALL_MODELS:
                idx = MODEL_INDEX[model]
                valid = np.isfinite(uno[:, idx])

                model_scenario_rows.append(
                    {
                        "scenario_id": scenario_id,
                        "target_events": int(scenario["target_events"]),
                        "transfer_regime": str(scenario["transfer_regime"]),
                        "covariance_shift": str(scenario["covariance_shift"]),
                        "mapping_error": str(scenario["mapping_error"]),
                        "source_prior_state": str(
                            scenario["source_prior_state"]
                        ),
                        "model": model,
                        "mean_uno_c": finite_mean(uno[:, idx]),
                        "mean_delta_c_vs_B0": finite_mean(delta[:, idx]),
                        "negative_transfer_rate": (
                            float(np.mean(neg[valid, idx]))
                            if np.any(valid)
                            else float("nan")
                        ),
                        "catastrophic_rate": (
                            float(np.mean(cat[valid, idx]))
                            if np.any(valid)
                            else float("nan")
                        ),
                    }
                )

            # ----------------------------------------------------------
            # Post-hoc causal/harmful module gate diagnostics.
            # ----------------------------------------------------------
            a3_idx = MODEL_INDEX["A3"]
            a3_delta = delta[:, a3_idx]

            scenario_harmful_rate = np.full(n_rep, np.nan, dtype=float)
            scenario_harmful_gate = np.full(n_rep, np.nan, dtype=float)
            scenario_beneficial_rate = np.full(n_rep, np.nan, dtype=float)
            scenario_beneficial_gate = np.full(n_rep, np.nan, dtype=float)
            scenario_attenuated_rate = np.full(n_rep, np.nan, dtype=float)
            scenario_attenuated_gate = np.full(n_rep, np.nan, dtype=float)
            scenario_noncausal_false_rate = np.full(n_rep, np.nan, dtype=float)

            broad_auc = saved_auc.copy()
            broad_false = saved_false.copy()

            for r in range(n_rep):
                bs = beta_source[r]
                bt = beta_target[r]

                beneficial = (
                    causal[r]
                    & transportable[r]
                    & (bs * bt > 0)
                    & (np.abs(bs) > 0)
                    & (np.abs(bt) > 0)
                )
                harmful_flip = (
                    causal[r]
                    & (~transportable[r])
                    & (bs * bt < 0)
                    & (np.abs(bs) > 0)
                    & (np.abs(bt) > 0)
                )
                attenuated = (
                    causal[r]
                    & (~transportable[r])
                    & (np.abs(bs) > 0)
                    & (np.abs(bt) <= 1e-12)
                )
                noncausal_nontransportable = (
                    (~causal[r]) & (~transportable[r])
                )

                if np.any(beneficial):
                    scenario_beneficial_rate[r] = float(
                        np.mean(
                            gate[r, beneficial] >= GATE_BORROW_THRESHOLD
                        )
                    )
                    scenario_beneficial_gate[r] = float(
                        np.mean(gate[r, beneficial])
                    )

                if np.any(harmful_flip):
                    scenario_harmful_rate[r] = float(
                        np.mean(
                            gate[r, harmful_flip] >= GATE_BORROW_THRESHOLD
                        )
                    )
                    scenario_harmful_gate[r] = float(
                        np.mean(gate[r, harmful_flip])
                    )

                if np.any(attenuated):
                    scenario_attenuated_rate[r] = float(
                        np.mean(
                            gate[r, attenuated] >= GATE_BORROW_THRESHOLD
                        )
                    )
                    scenario_attenuated_gate[r] = float(
                        np.mean(gate[r, attenuated])
                    )

                if np.any(noncausal_nontransportable):
                    scenario_noncausal_false_rate[r] = float(
                        np.mean(
                            gate[r, noncausal_nontransportable]
                            >= GATE_BORROW_THRESHOLD
                        )
                    )

            gate_scenario_rows.append(
                {
                    "scenario_id": scenario_id,
                    "family": str(scenario["family"]),
                    "target_events": int(scenario["target_events"]),
                    "transfer_regime": str(scenario["transfer_regime"]),
                    "covariance_shift": str(scenario["covariance_shift"]),
                    "mapping_error": str(scenario["mapping_error"]),
                    "censoring": str(scenario["censoring"]),
                    "source_prior_state": str(
                        scenario["source_prior_state"]
                    ),
                    "mean_A3_delta_c": finite_mean(a3_delta),
                    "negative_transfer_rate_A3": float(
                        np.mean(
                            a3_delta[np.isfinite(a3_delta)]
                            <= NEGATIVE_THRESHOLD
                        )
                    ),
                    "catastrophic_rate_A3": float(
                        np.mean(
                            a3_delta[np.isfinite(a3_delta)]
                            <= CATASTROPHIC_THRESHOLD
                        )
                    ),
                    "broad_transportability_AUROC": finite_mean(broad_auc),
                    "broad_false_borrow_rate": finite_mean(broad_false),
                    "beneficial_causal_borrow_rate": finite_mean(
                        scenario_beneficial_rate
                    ),
                    "beneficial_causal_mean_gate": finite_mean(
                        scenario_beneficial_gate
                    ),
                    "harmful_signflip_borrow_rate": finite_mean(
                        scenario_harmful_rate
                    ),
                    "harmful_signflip_mean_gate": finite_mean(
                        scenario_harmful_gate
                    ),
                    "attenuated_causal_borrow_rate": finite_mean(
                        scenario_attenuated_rate
                    ),
                    "attenuated_causal_mean_gate": finite_mean(
                        scenario_attenuated_gate
                    ),
                    "noncausal_nontransportable_false_borrow_rate": finite_mean(
                        scenario_noncausal_false_rate
                    ),
                }
            )

            coupling_rows.append(
                {
                    "scenario_id": scenario_id,
                    "target_events": int(scenario["target_events"]),
                    "transfer_regime": str(scenario["transfer_regime"]),
                    "source_prior_state": str(
                        scenario["source_prior_state"]
                    ),
                    "rho_module_AUROC_vs_deltaC": rank_corr(
                        broad_auc,
                        a3_delta,
                    ),
                    "rho_broad_false_borrow_vs_deltaC": rank_corr(
                        broad_false,
                        a3_delta,
                    ),
                    "rho_harmful_signflip_borrow_vs_deltaC": rank_corr(
                        scenario_harmful_rate,
                        a3_delta,
                    ),
                    "rho_harmful_signflip_mean_gate_vs_deltaC": rank_corr(
                        scenario_harmful_gate,
                        a3_delta,
                    ),
                    "rho_beneficial_borrow_vs_deltaC": rank_corr(
                        scenario_beneficial_rate,
                        a3_delta,
                    ),
                }
            )

        if i % 20 == 0 or i == EXPECTED_SCENARIOS:
            print(f"  diagnosed scenarios: {i}/{EXPECTED_SCENARIOS}")

    invariant_df = pd.DataFrame(invariant_rows)
    replay_scenario = pd.DataFrame(replay_scenario_rows)
    gate_scenario = pd.DataFrame(gate_scenario_rows)
    coupling = pd.DataFrame(coupling_rows)
    model_scenario = pd.DataFrame(model_scenario_rows)

    invariant_df.to_csv(INVARIANT_AUDIT, sep="\t", index=False)
    gate_scenario.to_csv(GATE_HARM_SCENARIO, sep="\t", index=False)
    coupling.to_csv(GATE_COUPLING, sep="\t", index=False)

    # ------------------------------------------------------------------
    # Independent candidate aggregate replay.
    # ------------------------------------------------------------------
    replay_rows: List[Dict[str, Any]] = []

    for model in ["A2", "A3"]:
        part = replay_scenario[
            replay_scenario["model"] == model
        ].copy()
        misleading = part[
            part["transfer_regime"] == "R5_MISLEADING_SOURCE"
        ]
        severe = part[exact_severe_mask(part)]

        row: Dict[str, Any] = {
            "model": model,
            "aggregate_negative_transfer_rate_replayed": finite_mean(
                part["negative_transfer_rate"]
            ),
            "catastrophic_rate_misleading_replayed": finite_mean(
                misleading["catastrophic_rate"]
            ),
            "catastrophic_rate_severe_shift_replayed": finite_mean(
                severe["catastrophic_rate"]
            ),
            "equal_scenario_mean_uno_c_replayed": finite_mean(
                part["mean_uno_c"]
            ),
            "module_recovery_AUROC_partial_replayed": float("nan"),
            "false_borrow_rate_misleading_replayed": float("nan"),
        }

        if model == "A3":
            partial = part[
                part["transfer_regime"].isin(
                    [
                        "R1_MOSTLY_TRANSPORTABLE",
                        "R2_PARTIALLY_TRANSPORTABLE",
                        "R3_WEAKLY_TRANSPORTABLE",
                    ]
                )
            ]
            row["module_recovery_AUROC_partial_replayed"] = finite_mean(
                partial["mean_module_auc"]
            )
            row["false_borrow_rate_misleading_replayed"] = finite_mean(
                misleading["mean_false_borrow"]
            )

        replay_rows.append(row)

    replay_df = pd.DataFrame(replay_rows)

    # Compare against authoritative frozen candidate table.
    authoritative = d_candidate.set_index("model")
    all_selection_matches = True

    for idx, row in replay_df.iterrows():
        model = row["model"]
        auth = authoritative.loc[model]

        comparisons = [
            (
                row["aggregate_negative_transfer_rate_replayed"],
                float(auth["aggregate_negative_transfer_rate"]),
            ),
            (
                row["catastrophic_rate_misleading_replayed"],
                float(auth["catastrophic_rate_misleading"]),
            ),
            (
                row["catastrophic_rate_severe_shift_replayed"],
                float(auth["catastrophic_rate_severe_shift"]),
            ),
            (
                row["equal_scenario_mean_uno_c_replayed"],
                float(auth["equal_scenario_mean_uno_c"]),
            ),
        ]

        if model == "A3":
            comparisons.extend(
                [
                    (
                        row["module_recovery_AUROC_partial_replayed"],
                        float(auth["module_recovery_AUROC_partial"]),
                    ),
                    (
                        row["false_borrow_rate_misleading_replayed"],
                        float(auth["false_borrow_rate_misleading"]),
                    ),
                ]
            )

        model_match = all(
            compare_numeric(obs, exp)
            for obs, exp in comparisons
        )
        replay_df.loc[idx, "matches_frozen_05d_candidate_summary"] = (
            model_match
        )
        all_selection_matches &= model_match

    replay_df.to_csv(SELECTION_REPLAY, sep="\t", index=False)

    # ------------------------------------------------------------------
    # Failure topology tables.
    # ------------------------------------------------------------------
    event_regime = (
        replay_scenario.groupby(
            ["model", "target_events", "transfer_regime"],
            as_index=False,
        )
        .agg(
            n_scenarios=("scenario_id", "nunique"),
            mean_uno_c=("mean_uno_c", "mean"),
            mean_delta_c_vs_B0=("mean_delta_c_vs_B0", "mean"),
            negative_transfer_rate=("negative_transfer_rate", "mean"),
            catastrophic_rate=("catastrophic_rate", "mean"),
        )
        .sort_values(["model", "target_events", "transfer_regime"])
    )
    event_regime.to_csv(
        FAILURE_EVENT_REGIME,
        sep="\t",
        index=False,
    )

    prior_regime = (
        replay_scenario.groupby(
            ["model", "source_prior_state", "transfer_regime"],
            as_index=False,
        )
        .agg(
            n_scenarios=("scenario_id", "nunique"),
            mean_uno_c=("mean_uno_c", "mean"),
            mean_delta_c_vs_B0=("mean_delta_c_vs_B0", "mean"),
            negative_transfer_rate=("negative_transfer_rate", "mean"),
            catastrophic_rate=("catastrophic_rate", "mean"),
            mean_module_auc=("mean_module_auc", "mean"),
            mean_false_borrow=("mean_false_borrow", "mean"),
        )
        .sort_values(["model", "source_prior_state", "transfer_regime"])
    )
    prior_regime.to_csv(
        FAILURE_PRIOR_REGIME,
        sep="\t",
        index=False,
    )

    stress_rows: List[Dict[str, Any]] = []
    for model in ["A2", "A3"]:
        part = replay_scenario[
            replay_scenario["model"] == model
        ]

        for axis in [
            "covariance_shift",
            "mapping_error",
            "censoring",
            "source_prior_state",
        ]:
            for level, group in part.groupby(axis):
                stress_rows.append(
                    {
                        "model": model,
                        "axis": axis,
                        "level": str(level),
                        "n_scenarios": len(group),
                        "mean_delta_c_vs_B0": finite_mean(
                            group["mean_delta_c_vs_B0"]
                        ),
                        "negative_transfer_rate": finite_mean(
                            group["negative_transfer_rate"]
                        ),
                        "catastrophic_rate": finite_mean(
                            group["catastrophic_rate"]
                        ),
                    }
                )

    stress_df = pd.DataFrame(stress_rows)
    stress_df.to_csv(FAILURE_STRESS, sep="\t", index=False)

    # ------------------------------------------------------------------
    # Frozen comparator summary.
    # ------------------------------------------------------------------
    comparator_rows: List[Dict[str, Any]] = []

    for model, part in model_scenario.groupby("model"):
        misleading = part[
            part["transfer_regime"] == "R5_MISLEADING_SOURCE"
        ]
        severe = part[exact_severe_mask(part)]

        comparator_rows.append(
            {
                "model": model,
                "n_scenarios": len(part),
                "equal_scenario_mean_uno_c": finite_mean(
                    part["mean_uno_c"]
                ),
                "equal_scenario_mean_delta_c_vs_B0": finite_mean(
                    part["mean_delta_c_vs_B0"]
                ),
                "aggregate_negative_transfer_rate": finite_mean(
                    part["negative_transfer_rate"]
                ),
                "catastrophic_rate_misleading": finite_mean(
                    misleading["catastrophic_rate"]
                ),
                "catastrophic_rate_severe_shift": finite_mean(
                    severe["catastrophic_rate"]
                ),
            }
        )

    comparator_df = pd.DataFrame(comparator_rows).sort_values(
        "equal_scenario_mean_uno_c",
        ascending=False,
    )
    comparator_df.to_csv(
        MODEL_COMPARATOR_SUMMARY,
        sep="\t",
        index=False,
    )

    # ------------------------------------------------------------------
    # A3 ablation contrasts relative to A3.
    # ------------------------------------------------------------------
    base_a3 = model_scenario[
        model_scenario["model"] == "A3"
    ].set_index("scenario_id")

    ablation_rows: List[Dict[str, Any]] = []

    subsets = {
        "ALL": lambda x: x.index == x.index,
        "R5_MISLEADING_SOURCE": lambda x: (
            x["transfer_regime"] == "R5_MISLEADING_SOURCE"
        ),
        "R2_PARTIALLY_TRANSPORTABLE": lambda x: (
            x["transfer_regime"] == "R2_PARTIALLY_TRANSPORTABLE"
        ),
        "P2_MISLEADING_PRIOR": lambda x: (
            x["source_prior_state"] == "P2_MISLEADING"
        ),
        "SEVERE_SHIFT": lambda x: exact_severe_mask(x),
    }

    for ablation in ABLATIONS:
        abl = model_scenario[
            model_scenario["model"] == ablation
        ].set_index("scenario_id")

        joined = base_a3[
            [
                "target_events",
                "transfer_regime",
                "covariance_shift",
                "mapping_error",
                "source_prior_state",
                "mean_uno_c",
                "negative_transfer_rate",
                "catastrophic_rate",
            ]
        ].join(
            abl[
                [
                    "mean_uno_c",
                    "negative_transfer_rate",
                    "catastrophic_rate",
                ]
            ],
            lsuffix="_A3",
            rsuffix="_ABL",
            how="inner",
        )

        for subset_name, subset_fn in subsets.items():
            mask = np.asarray(subset_fn(joined), dtype=bool)
            part = joined.loc[mask].copy()

            if len(part) == 0:
                continue

            d_c = (
                part["mean_uno_c_ABL"]
                - part["mean_uno_c_A3"]
            )
            d_nt = (
                part["negative_transfer_rate_ABL"]
                - part["negative_transfer_rate_A3"]
            )
            d_cat = (
                part["catastrophic_rate_ABL"]
                - part["catastrophic_rate_A3"]
            )

            ablation_rows.append(
                {
                    "ablation": ablation,
                    "subset": subset_name,
                    "n_scenarios": len(part),
                    "mean_delta_uno_ablation_minus_A3": finite_mean(d_c),
                    "median_delta_uno_ablation_minus_A3": finite_median(d_c),
                    "fraction_scenarios_ablation_higher_uno": float(
                        np.mean(d_c > 0)
                    ),
                    "mean_change_negative_transfer_rate": finite_mean(d_nt),
                    "mean_change_catastrophic_rate": finite_mean(d_cat),
                }
            )

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(
        ABLATION_SUMMARY,
        sep="\t",
        index=False,
    )

    # ------------------------------------------------------------------
    # Grouped harmful-module gate summaries.
    # ------------------------------------------------------------------
    grouped_rows: List[Dict[str, Any]] = []

    grouping_specs = [
        (["transfer_regime"], "BY_TRANSFER_REGIME"),
        (["source_prior_state"], "BY_PRIOR"),
        (["target_events"], "BY_EVENT_COUNT"),
        (["transfer_regime", "source_prior_state"], "BY_REGIME_AND_PRIOR"),
    ]

    numeric_gate_cols = [
        "mean_A3_delta_c",
        "negative_transfer_rate_A3",
        "catastrophic_rate_A3",
        "broad_transportability_AUROC",
        "broad_false_borrow_rate",
        "beneficial_causal_borrow_rate",
        "beneficial_causal_mean_gate",
        "harmful_signflip_borrow_rate",
        "harmful_signflip_mean_gate",
        "attenuated_causal_borrow_rate",
        "attenuated_causal_mean_gate",
        "noncausal_nontransportable_false_borrow_rate",
    ]

    for group_cols, grouping_name in grouping_specs:
        for keys, group in gate_scenario.groupby(group_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)

            row = {
                "grouping": grouping_name,
                "n_scenarios": len(group),
            }
            for col, key in zip(group_cols, keys):
                row[col] = key

            for col in numeric_gate_cols:
                row[col] = finite_mean(group[col])

            grouped_rows.append(row)

    grouped_df = pd.DataFrame(grouped_rows)
    grouped_df.to_csv(
        GATE_HARM_GROUPED,
        sep="\t",
        index=False,
    )

    # ------------------------------------------------------------------
    # Summarize coupling.
    # ------------------------------------------------------------------
    coupling_summary = {
        "median_within_scenario_rho_module_AUROC_vs_deltaC": finite_median(
            coupling["rho_module_AUROC_vs_deltaC"]
        ),
        "median_within_scenario_rho_broad_false_borrow_vs_deltaC": finite_median(
            coupling["rho_broad_false_borrow_vs_deltaC"]
        ),
        "median_within_scenario_rho_harmful_signflip_borrow_vs_deltaC": finite_median(
            coupling["rho_harmful_signflip_borrow_vs_deltaC"]
        ),
        "median_within_scenario_rho_harmful_signflip_mean_gate_vs_deltaC": finite_median(
            coupling["rho_harmful_signflip_mean_gate_vs_deltaC"]
        ),
        "median_within_scenario_rho_beneficial_borrow_vs_deltaC": finite_median(
            coupling["rho_beneficial_borrow_vs_deltaC"]
        ),
    }

    # ------------------------------------------------------------------
    # Final diagnostic status.
    # ------------------------------------------------------------------
    invariant_pass = bool(invariant_df["pass"].all())

    a3_auth = d_candidate.set_index("model").loc["A3"]
    module_prediction_decoupling = bool(
        float(a3_auth["module_recovery_AUROC_partial"])
        >= FROZEN_MIN_MODULE_AUC
        and float(a3_auth["false_borrow_rate_misleading"])
        <= FROZEN_MAX_FALSE_BORROW
        and (
            float(a3_auth["aggregate_negative_transfer_rate"])
            > FROZEN_MAX_NEGATIVE_RATE
            or float(a3_auth["catastrophic_rate_misleading"])
            > FROZEN_MAX_CATASTROPHIC_RATE
            or float(a3_auth["catastrophic_rate_severe_shift"])
            > FROZEN_MAX_CATASTROPHIC_RATE
        )
    )

    if not invariant_pass or not all_selection_matches:
        status = "HOLD_TECHNICAL_INVARIANT_OR_SELECTION_REPLAY_FAILURE"
    else:
        status = "PASS_NO_STRUCTURAL_IMPLEMENTATION_BUG_DETECTED_IN_BOUNDED_AUDIT"

    summary = {
        "script_version": SCRIPT_VERSION,
        "status": "PASS" if status.startswith("PASS") else "HOLD",
        "scientific_status": status,
        "run_started_utc": started,
        "run_finished_utc": now_utc(),
        "05d_frozen_scientific_status_unchanged": EXPECTED_05D_STATUS,
        "A5_frozen_branch_unchanged": "CLOSE_A5",
        "model_fitting": False,
        "retuning": False,
        "real_data_values_read": False,
        "reserved_human_outcomes_read": False,
        "GPU_execution": False,
        "implementation_invariants_all_pass": invariant_pass,
        "independent_selection_metric_replay_pass": all_selection_matches,
        "module_recovery_prediction_safety_decoupling": (
            module_prediction_decoupling
        ),
        "coupling_summary": coupling_summary,
        "interpretation_guardrails": {
            "posthoc_diagnostics_may_not_rescue_A2_or_A3": True,
            "posthoc_diagnostics_may_not_change_05d_thresholds": True,
            "posthoc_diagnostics_may_not_open_A5": True,
            "harmful_module_gate_metrics_are_explanatory_only": True,
            "if_no_implementation_bug_is_found": (
                "Any new safe-transfer architecture requires a NEW prospective "
                "development contract and NEW untouched simulation scenarios; "
                "the existing 21,600 simulations may be used as development/"
                "diagnostic evidence but not as that architecture's final unbiased benchmark."
            ),
        },
        "final_artifact_hashes": {
            "implementation_invariant_audit.tsv": sha256_file(INVARIANT_AUDIT),
            "independent_selection_metric_replay.tsv": sha256_file(SELECTION_REPLAY),
            "failure_by_event_and_transfer_regime.tsv": sha256_file(
                FAILURE_EVENT_REGIME
            ),
            "failure_by_prior_and_transfer_regime.tsv": sha256_file(
                FAILURE_PRIOR_REGIME
            ),
            "failure_by_stress_axis.tsv": sha256_file(FAILURE_STRESS),
            "all_model_failure_summary.tsv": sha256_file(
                MODEL_COMPARATOR_SUMMARY
            ),
            "A3_ablation_failure_contrasts.tsv": sha256_file(
                ABLATION_SUMMARY
            ),
            "A3_gate_harmful_module_diagnostics_by_scenario.tsv": sha256_file(
                GATE_HARM_SCENARIO
            ),
            "A3_gate_prediction_coupling.tsv": sha256_file(GATE_COUPLING),
            "A3_gate_harmful_module_grouped_summary.tsv": sha256_file(
                GATE_HARM_GROUPED
            ),
        },
    }
    write_json(SUMMARY_JSON, summary)

    # ------------------------------------------------------------------
    # User-facing compact output.
    # ------------------------------------------------------------------
    print()
    print("-" * 120)
    print("Independent replay of frozen 05d selection inputs")
    print("-" * 120)
    print(replay_df.to_string(index=False))

    print()
    print("-" * 120)
    print("Structural implementation audit")
    print("-" * 120)
    audit_by_check = (
        invariant_df.groupby("check", as_index=False)
        .agg(
            n_scenarios=("scenario_id", "nunique"),
            n_pass=("pass", "sum"),
            all_pass=("pass", "all"),
        )
    )
    print(audit_by_check.to_string(index=False))

    print()
    print("-" * 120)
    print("Frozen model family: aggregate post-hoc comparison")
    print("-" * 120)
    print(
        comparator_df[
            [
                "model",
                "equal_scenario_mean_uno_c",
                "aggregate_negative_transfer_rate",
                "catastrophic_rate_misleading",
                "catastrophic_rate_severe_shift",
            ]
        ].to_string(index=False)
    )

    print()
    print("-" * 120)
    print("A3 ablations vs A3 [ALL and key failure subsets]")
    print("-" * 120)
    print(
        ablation_df[
            [
                "ablation",
                "subset",
                "n_scenarios",
                "mean_delta_uno_ablation_minus_A3",
                "mean_change_negative_transfer_rate",
                "mean_change_catastrophic_rate",
            ]
        ].to_string(index=False)
    )

    print()
    print("-" * 120)
    print("A3 broad transportability vs prognostically harmful borrowing")
    print("-" * 120)

    key_groups = grouped_df[
        (
            (grouped_df["grouping"] == "BY_TRANSFER_REGIME")
            & (
                grouped_df.get("transfer_regime", pd.Series(index=grouped_df.index))
                == "R5_MISLEADING_SOURCE"
            )
        )
        | (
            (grouped_df["grouping"] == "BY_PRIOR")
            & (
                grouped_df.get("source_prior_state", pd.Series(index=grouped_df.index))
                == "P2_MISLEADING"
            )
        )
    ].copy()

    display_cols = [
        c for c in [
            "grouping",
            "transfer_regime",
            "source_prior_state",
            "n_scenarios",
            "mean_A3_delta_c",
            "catastrophic_rate_A3",
            "broad_false_borrow_rate",
            "harmful_signflip_borrow_rate",
            "harmful_signflip_mean_gate",
            "noncausal_nontransportable_false_borrow_rate",
        ]
        if c in key_groups.columns
    ]
    print(key_groups[display_cols].to_string(index=False))

    print()
    print("Within-scenario rank-correlation summary:")
    for key, value in coupling_summary.items():
        print(f"  {key}: {value:.4f}" if np.isfinite(value) else f"  {key}: NA")

    print()
    print("=" * 120)
    print("05d0 FROZEN-AI FAILURE DIAGNOSTIC SUMMARY")
    print("=" * 120)
    print(f"Status: {status}")
    print(
        "Independent frozen-selection replay: "
        + ("PASS" if all_selection_matches else "FAIL")
    )
    print(
        "Structural implementation invariants: "
        + ("PASS" if invariant_pass else "FAIL")
    )
    print(
        "Module-recovery / prediction-safety decoupling: "
        + ("YES" if module_prediction_decoupling else "NO")
    )
    print("05d scientific HOLD changed: NO")
    print("A5 CLOSE changed: NO")
    print()
    if invariant_pass and all_selection_matches:
        print(
            "If the detailed tables do not reveal a separately provable implementation "
            "defect, the current evidence supports a genuine architecture/objective "
            "failure rather than a bookkeeping error."
        )
        print(
            "Any replacement architecture must enter a NEW prospective branch with "
            "NEW untouched simulations before human outcomes are opened."
        )
    else:
        print(
            "TECHNICAL HOLD: resolve the failed invariant/replay before drawing "
            "methodological conclusions."
        )
    print("=" * 120)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 120, file=sys.stderr)
        print("05d0 frozen-AI failure diagnostic: FAIL", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print("=" * 120, file=sys.stderr)
        raise
