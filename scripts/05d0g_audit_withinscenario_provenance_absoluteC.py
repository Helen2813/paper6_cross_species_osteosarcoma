#!/usr/bin/env python3
"""
Paper 6 - 05d0g:
utility-reference feasibility, comparator decomposition,
within-scenario correlation audit, threshold provenance erratum,
absolute-C sensitivity, and regime-specific safety.

Chronology
----------
This stage is AFTER the already-completed old 05d0e regime-semantics audit AND the already-completed 05d0f v1 utility/comparator/regime-safety audit. Both are preserved and required as inputs.

Frozen decisions remain unchanged:
- 05d  = HOLD_NO_SAFE_SELECTABLE_ARCHITECTURE
- A5   = CLOSE_A5
- 05d0c = STOP_A6_FOR_PAPER6

No result in this script may reopen A6.

IMPORTANT TERMINOLOGY / PROVENANCE CORRECTION
---------------------------------------------
The original frozen 05a simulation-selection contract did NOT impose a
positive mean utility threshold of +0.02.

The value 0.02 appears in several DISTINCT roles across Paper 6:
- 05a: NEGATIVE transfer boundary, deltaC <= -0.02.
- 05c1/05c1a: A5 shift/deterioration trigger, +0.02.
- 05d0c: POST-HOC operational positive label,
         beneficial if test deltaC(engine-B0) >= +0.02.
- planned 05f1 human relative-effect rule may also use +0.02, but if no 05f1
  artifact exists yet it MUST be reported as NOT_YET_MATERIALIZED/FROZEN.

05d0d used wording such as "frozen practical +0.02 gain" and 05d0c used a
column name "beneficial_regime" for R0-R3. Those labels are corrected here:
- +0.02 positive = POST-HOC operational reference, not frozen 05a utility rule.
- R0-R3 = MECHANISTICALLY COMPATIBLE generator regimes, not necessarily
  operationally beneficial regimes.

Questions
---------
1. Utility-reference ceiling:
   Within already-evaluated B0/A1/A2 predictions, what mean deltaC is attainable
   by an outcome-aware per-replicate oracle that selects on the SAME independent
   test outcome used for evaluation?

   This is deliberately an optimistic / winner's-curse upper bound.
   If even this bound is < +0.02, then +0.02 is unattainable within:
      (i) the evaluated B0/A1/A2 engine family,
      (ii) the frozen scenario grid,
      (iii) equal-scenario weighting.
   It does NOT prove global infeasibility for all possible methods.

2. Comparator decomposition:
   For each event budget:
      rho(compatibility, C_transfer)
      rho(compatibility, C_B0)
      rho(compatibility, deltaC)

3. Within-scenario decomposition:
   Compute the same three Spearman correlations WITHIN each scenario across
   its 100/200 replicates, then summarize their distribution. This prevents
   marginal regime/scenario mixture from masquerading as an individual-level
   association.

4. Absolute-C post-hoc sensitivity:
   05d0c operational labels depended on deltaC vs B0. As a comparator-free
   sensitivity ONLY, define:
      ABSOLUTE_HIGHER_DISCRIMINATION: C_transfer >= 0.55
      ABSOLUTE_ANTICONCORDANT:        C_transfer <= 0.50
      0.50 < C < 0.55:                excluded
   Compute AUROC of the SAME compatibility score for this label.
   This cannot change STOP_A6_FOR_PAPER6.

5. Exact R0-R5 outcomes and complete A1/A2 event-only abstention ladder with
   negative-transfer rate reported separately in:
      R0-R3 mechanistically compatible
      R4-R5 mechanistically incompatible.

No model fitting.
No new trust signal.
No threshold tuning.
No human outcomes.
No GPU.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import roc_auc_score
except ImportError as exc:
    raise ImportError("05d0f v2 requires scikit-learn.") from exc


SCRIPT_VERSION = (
    "05d0g-withinscenario-provenance-absoluteC-sensitivity-v1-no-cli"
)

ROOT = Path(__file__).resolve().parents[1]

# Frozen / completed inputs.
A_DIR = ROOT / "results" / "simulation_contract" / "05a"
A_CONTRACT = A_DIR / "negative_transfer_simulation_contract.json"

C1_DIR = ROOT / "results" / "simulation_extension_contract" / "05c1"
C1_CONTRACT = C1_DIR / "A5_ECHRR_branch_contract.json"

C1A_DIR = ROOT / "results" / "simulation_extension_contract" / "05c1a"
C1A_CONTRACT = C1A_DIR / "A5_operational_branch_definitions.json"

D0C_DIR = ROOT / "results" / "simulation_phase_diagram_diagnostics" / "05d0c"
D0C_REPS = D0C_DIR / "global_trust_replicate_statistics.tsv"
D0C_POLICY = D0C_DIR / "event_count_only_abstention_policies.tsv"
D0C_CONTRACT = D0C_DIR / "global_trust_feasibility_contract.json"
D0C_SUMMARY = D0C_DIR / "summary.json"

D0D_DIR = ROOT / "results" / "simulation_phase_diagram_diagnostics" / "05d0d"
D0D_SUMMARY = D0D_DIR / "summary.json"

# Already-run OLD 05d0e. Preserve it.
OLD_D0E_DIR = ROOT / "results" / "simulation_phase_diagram_diagnostics" / "05d0e"
OLD_D0E_SUMMARY = OLD_D0E_DIR / "summary.json"

# Already-run 05d0f v1 utility/comparator/regime-safety audit. Preserve it.
PREV_D0F_DIR = ROOT / "results" / "simulation_phase_diagram_diagnostics" / "05d0f"
PREV_D0F_SUMMARY = PREV_D0F_DIR / "summary.json"

OUT_DIR = ROOT / "results" / "simulation_phase_diagram_diagnostics" / "05d0g"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ERRATUM = OUT_DIR / "diagnostic_nomenclature_erratum.json"
THRESHOLD_PROVENANCE = OUT_DIR / "threshold_0p02_provenance.tsv"
UTILITY_ORACLE = OUT_DIR / "posthoc_plus002_utility_oracle_upper_bounds.tsv"
MARGINAL_CORR = OUT_DIR / "marginal_comparator_decomposition.tsv"
WITHIN_SCENARIO = OUT_DIR / "within_scenario_comparator_decomposition.tsv"
WITHIN_SCENARIO_SUMMARY = OUT_DIR / "within_scenario_correlation_summary.tsv"
ABSOLUTE_C_SENSITIVITY = OUT_DIR / "absolute_C_operational_sensitivity.tsv"
QUARTILES = OUT_DIR / "compatibility_quartile_comparator_decomposition.tsv"
EXACT_REGIMES = OUT_DIR / "exact_regime_A1_A2_outcomes.tsv"
FULL_LADDER = OUT_DIR / "complete_A1_A2_event_abstention_ladder_regime_specific.tsv"
SUMMARY_JSON = OUT_DIR / "summary.json"

EXPECTED_D0C_STATUS = "STOP_A6_FOR_PAPER6"
EXPECTED_D0D_STATUS = "PASS_POSTHOC_MECHANISTIC_OPERATIONAL_CROSSWALK_COMPLETE"
EXPECTED_OLD_D0E_STATUS = (
    "PASS_POSTHOC_REGIME_SEMANTICS_AUDIT_COMPLETE_A6_REMAINS_STOP"
)
EXPECTED_PREV_D0F_STATUS = (
    "PASS_POSTHOC_UTILITY_COMPARATOR_REGIME_SAFETY_AUDIT_COMPLETE"
)

ENGINES = ["A1", "A2"]
EVENT_BUDGETS = [5, 10, 15, 20, 29, 40]
EVENT_POLICY_THRESHOLDS = [5, 10, 15, 20, 29, 40, 41]
PRIMARY_EVENT_BUDGET = 29

# Frozen 05a negative-transfer definition.
NEGATIVE_TRANSFER_DELTA = -0.02

# POST-HOC 05d0c operational positive reference, NOT frozen 05a utility rule.
POSTHOC_POSITIVE_REFERENCE = +0.02

# Frozen BEFORE this sensitivity is computed. Descriptive only.
ABSOLUTE_C_HIGH = 0.55
ABSOLUTE_C_LOW = 0.50


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


def finite_mean(values: Iterable[float]) -> float:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if len(x) else float("nan")


def finite_median(values: Iterable[float]) -> float:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else float("nan")


def finite_quantile(values: Iterable[float], q: float) -> float:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    return float(np.quantile(x, q)) if len(x) else float("nan")


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]

    if len(x) < 5 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")

    rx = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    ry = pd.Series(y).rank(method="average").to_numpy(dtype=float)

    if np.std(rx) <= 1e-12 or np.std(ry) <= 1e-12:
        return float("nan")

    return float(np.corrcoef(rx, ry)[0, 1])


def safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)

    keep = np.isfinite(score) & np.isin(y, [0, 1])
    y = y[keep]
    score = score[keep]

    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")

    return float(roc_auc_score(y, score))


def mechanistic_group(regime: str) -> str:
    r = str(regime)
    if r.startswith(("R0_", "R1_", "R2_", "R3_")):
        return "MECHANISTICALLY_COMPATIBLE_R0_R3"
    if r.startswith(("R4_", "R5_")):
        return "MECHANISTICALLY_INCOMPATIBLE_R4_R5"
    raise RuntimeError(f"Unrecognized transfer regime: {regime}")


def scenario_equal_mean(frame: pd.DataFrame, value_col: str) -> float:
    vals = (
        frame.groupby("scenario_id", as_index=False)[value_col]
        .mean()[value_col]
        .to_numpy(dtype=float)
    )
    return finite_mean(vals)


def scenario_equal_rate(frame: pd.DataFrame, indicator_col: str) -> float:
    vals = (
        frame.groupby("scenario_id", as_index=False)[indicator_col]
        .mean()[indicator_col]
        .to_numpy(dtype=float)
    )
    return finite_mean(vals)


def walk_json(obj: Any, path: str = "$") -> List[Tuple[str, Any]]:
    out: List[Tuple[str, Any]] = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(walk_json(v, f"{path}.{k}"))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(walk_json(v, f"{path}[{i}]"))
    else:
        out.append((path, obj))

    return out


def exact_002_occurrences(label: str, path: Path) -> List[Dict[str, Any]]:
    """
    Record exact numeric +/-0.02 occurrences and string occurrences mentioning
    0.02. This is provenance, not semantic inference.
    """
    if not path.exists():
        return [{
            "artifact": label,
            "artifact_path": str(path.relative_to(ROOT)),
            "artifact_sha256": "",
            "json_path": "",
            "value_type": "MISSING_ARTIFACT",
            "value": "",
            "context": "NOT_MATERIALIZED",
        }]

    obj = read_json(path)
    rows: List[Dict[str, Any]] = []
    digest = sha256_file(path)

    for json_path, value in walk_json(obj):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if np.isclose(abs(float(value)), 0.02, atol=1e-12, rtol=0):
                rows.append({
                    "artifact": label,
                    "artifact_path": str(path.relative_to(ROOT)),
                    "artifact_sha256": digest,
                    "json_path": json_path,
                    "value_type": "NUMERIC_EXACT_ABS_0.02",
                    "value": repr(float(value)),
                    "context": "",
                })

        elif isinstance(value, str) and re.search(
            r"(?<![\d.])[-+]?(?:0\.02|\.02)(?!\d)",
            value,
        ):
            rows.append({
                "artifact": label,
                "artifact_path": str(path.relative_to(ROOT)),
                "artifact_sha256": digest,
                "json_path": json_path,
                "value_type": "STRING_MENTIONS_0.02",
                "value": value,
                "context": "",
            })

    if not rows:
        rows.append({
            "artifact": label,
            "artifact_path": str(path.relative_to(ROOT)),
            "artifact_sha256": digest,
            "json_path": "",
            "value_type": "NO_0.02_OCCURRENCE_FOUND",
            "value": "",
            "context": "",
        })

    return rows


def main() -> None:
    started = now_utc()

    print("=" * 120)
    print("Paper 6 - 05d0g within-scenario/provenance/absolute-C sensitivity audit")
    print("=" * 120)
    print(f"Script version: {SCRIPT_VERSION}")
    print()
    print("Scope:")
    print("  Old 05d0e preserved: YES")
    print("  Previous 05d0f v1 preserved: YES")
    print("  New model fitting: NO")
    print("  New trust signal: NO")
    print("  +0.02 treated as frozen 05a positive utility rule: NO")
    print("  Absolute-C sensitivity can reopen A6: NO")
    print("  A6 remains STOP: YES")
    print("  Human outcomes: NO")
    print("  GPU: NO")
    print()

    for p in [
        A_CONTRACT,
        D0C_REPS,
        D0C_POLICY,
        D0C_CONTRACT,
        D0C_SUMMARY,
        D0D_SUMMARY,
        OLD_D0E_SUMMARY,
        PREV_D0F_SUMMARY,
    ]:
        require_file(p)

    d0c = read_json(D0C_SUMMARY)
    d0d = read_json(D0D_SUMMARY)
    old_d0e = read_json(OLD_D0E_SUMMARY)
    prev_d0f = read_json(PREV_D0F_SUMMARY)

    if str(d0c.get("scientific_status")) != EXPECTED_D0C_STATUS:
        raise RuntimeError("05d0c is not STOP_A6_FOR_PAPER6.")
    if str(d0d.get("scientific_status")) != EXPECTED_D0D_STATUS:
        raise RuntimeError("05d0d is not complete.")
    if str(old_d0e.get("scientific_status")) != EXPECTED_OLD_D0E_STATUS:
        raise RuntimeError("Old 05d0e regime-semantics audit is not PASS.")
    if str(prev_d0f.get("scientific_status")) != EXPECTED_PREV_D0F_STATUS:
        raise RuntimeError("Already-run 05d0f v1 utility/comparator audit is not PASS.")

    reps = pd.read_csv(D0C_REPS, sep="\t")
    prior_policies = pd.read_csv(D0C_POLICY, sep="\t")

    if len(reps) != 21600:
        raise RuntimeError(
            f"Expected 21,600 replicate rows; observed {len(reps)}."
        )

    reps["mechanistic_group"] = [
        mechanistic_group(x)
        for x in reps["transfer_regime"].astype(str)
    ]

    # ------------------------------------------------------------------
    # 0. Explicit erratum + threshold provenance.
    # ------------------------------------------------------------------
    erratum = {
        "script_version": SCRIPT_VERSION,
        "created_utc": now_utc(),
        "status": "EXPLICIT_POSTRESULT_NOMENCLATURE_ERRATUM",
        "old_artifacts_are_not_modified": True,
        "prior_05d0e_preserved": True,
        "prior_05d0e_summary_sha256": sha256_file(OLD_D0E_SUMMARY),
        "prior_05d0f_v1_preserved": True,
        "prior_05d0f_v1_summary_sha256": sha256_file(PREV_D0F_SUMMARY),
        "corrections": [
            {
                "old_location": "05d0c event_count_only_abstention_policies.tsv",
                "old_term": "beneficial_regime_activation_fraction / mean_delta_in_beneficial_regimes",
                "correct_term": (
                    "R0_R3_mechanistically_compatible_activation_fraction / "
                    "R0_R3_mechanistically_compatible_mean_delta_c"
                ),
                "reason": (
                    "R0-R3 are generator-defined mechanistic compatibility regimes; "
                    "they are not guaranteed operationally beneficial for A1/A2."
                ),
            },
            {
                "old_location": "05d0d console/table wording",
                "old_term": "frozen practical +0.02 gain",
                "correct_term": (
                    "post-hoc +0.02 operational positive reference inherited "
                    "from the 05d0c benefit label"
                ),
                "reason": (
                    "No positive mean-gain +0.02 selection threshold is asserted "
                    "as part of frozen 05a here."
                ),
            },
        ],
        "numerical_results_changed": False,
        "05d_hold_changed": False,
        "05d0c_STOP_A6_changed": False,
    }
    write_json(ERRATUM, erratum)

    provenance_rows: List[Dict[str, Any]] = []

    for label, path in [
        ("05a_SIMULATION_SELECTION", A_CONTRACT),
        ("05c1_A5_BRANCH", C1_CONTRACT),
        ("05c1a_A5_OPERATIONAL", C1A_CONTRACT),
        ("05d0c_OPERATIONAL_FEASIBILITY", D0C_CONTRACT),
    ]:
        provenance_rows.extend(exact_002_occurrences(label, path))

    # 05f1 has not been run/materialized in the current workflow unless an
    # actual artifact exists. Record filesystem reality, do not claim freeze.
    human_candidates = list(
        (ROOT / "results").glob("**/*05f1*.json")
    ) + list(
        (ROOT / "scripts").glob("05f1*.py")
    )

    if human_candidates:
        for p in human_candidates:
            provenance_rows.append({
                "artifact": "05f1_HUMAN_GATE_CANDIDATE",
                "artifact_path": str(p.relative_to(ROOT)),
                "artifact_sha256": sha256_file(p),
                "json_path": "",
                "value_type": "EXISTING_FILE_NOT_PARSED_AS_FROZEN_JSON"
                if p.suffix.lower() != ".json"
                else "EXISTING_05f1_ARTIFACT",
                "value": "",
                "context": (
                    "Presence alone does not establish that a +0.02 role is frozen."
                ),
            })
    else:
        provenance_rows.append({
            "artifact": "05f1_HUMAN_GATE",
            "artifact_path": "",
            "artifact_sha256": "",
            "json_path": "",
            "value_type": "NOT_YET_MATERIALIZED_IN_CURRENT_WORKFLOW",
            "value": "",
            "context": (
                "Do not cite a future planned +0.02 human rule as already frozen."
            ),
        })

    provenance_df = pd.DataFrame(provenance_rows)
    provenance_df.to_csv(
        THRESHOLD_PROVENANCE,
        sep="\t",
        index=False,
    )

    # ------------------------------------------------------------------
    # 1. Outcome-aware utility-reference oracle upper bounds.
    # ------------------------------------------------------------------
    oracle_rows: List[Dict[str, Any]] = []

    def append_oracle(
        policy: str,
        engine_scope: str,
        chosen_delta: np.ndarray,
        activation: np.ndarray,
        select_a1: Optional[np.ndarray] = None,
        select_a2: Optional[np.ndarray] = None,
    ) -> None:
        temp = reps[["scenario_id"]].copy()
        temp["_delta"] = np.asarray(chosen_delta, dtype=float)
        temp["_active"] = np.asarray(activation, dtype=float)
        temp["_negative"] = (
            temp["_delta"] <= NEGATIVE_TRANSFER_DELTA
        ).astype(float)

        mean_delta = scenario_equal_mean(temp, "_delta")

        oracle_rows.append({
            "policy": policy,
            "engine_scope": engine_scope,
            "oracle_status": (
                "OPTIMISTIC_EVALUATION_OUTCOME_SELECTED_UPPER_BOUND_WINNERS_CURSE"
            ),
            "uses_same_independent_test_outcome_for_selection_and_evaluation": True,
            "scope_engine_family": "B0/A1/A2_ONLY",
            "scope_scenario_grid": "FROZEN_05a_180_SCENARIOS",
            "scope_weighting": "EQUAL_SCENARIO_WEIGHTING",
            "equal_scenario_mean_delta_c_vs_B0": mean_delta,
            "aggregate_negative_transfer_rate": scenario_equal_rate(
                temp, "_negative"
            ),
            "equal_scenario_activation_fraction": scenario_equal_mean(
                temp, "_active"
            ),
            "reaches_posthoc_mean_plus_0_02_reference": bool(
                mean_delta >= POSTHOC_POSITIVE_REFERENCE
            ),
            "selection_fraction_A1": (
                float(np.mean(select_a1))
                if select_a1 is not None
                else float("nan")
            ),
            "selection_fraction_A2": (
                float(np.mean(select_a2))
                if select_a2 is not None
                else float("nan")
            ),
        })

    d1 = reps["delta_c_A1"].to_numpy(dtype=float)
    d2 = reps["delta_c_A2"].to_numpy(dtype=float)

    for engine, d in [("A1", d1), ("A2", d2)]:
        append_oracle(
            f"{engine}_ALWAYS_TRANSFER_REPLAY",
            engine,
            d,
            np.ones(len(d), dtype=float),
        )
        append_oracle(
            f"{engine}_PERFECT_POSITIVE_DELTA_ORACLE",
            engine,
            np.where(d > 0.0, d, 0.0),
            (d > 0.0).astype(float),
        )
        append_oracle(
            f"{engine}_PERFECT_DELTA_GE_0_02_ORACLE",
            engine,
            np.where(
                d >= POSTHOC_POSITIVE_REFERENCE,
                d,
                0.0,
            ),
            (d >= POSTHOC_POSITIVE_REFERENCE).astype(float),
        )

    best_delta = np.maximum.reduce(
        [np.zeros(len(reps)), d1, d2]
    )
    choose_a1 = (d1 > 0) & (d1 >= d2)
    choose_a2 = (d2 > 0) & (d2 > d1)

    append_oracle(
        "BEST_OF_B0_A1_A2_PER_REPLICATE_ORACLE",
        "B0/A1/A2",
        best_delta,
        (best_delta > 0).astype(float),
        select_a1=choose_a1,
        select_a2=choose_a2,
    )

    best_delta02 = np.maximum.reduce([
        np.zeros(len(reps)),
        np.where(d1 >= POSTHOC_POSITIVE_REFERENCE, d1, 0.0),
        np.where(d2 >= POSTHOC_POSITIVE_REFERENCE, d2, 0.0),
    ])

    append_oracle(
        "BEST_OF_B0_A1_A2_DELTA_GE_0_02_ORACLE",
        "B0/A1/A2",
        best_delta02,
        (best_delta02 > 0).astype(float),
        select_a1=(
            (d1 >= POSTHOC_POSITIVE_REFERENCE) & (d1 >= d2)
        ),
        select_a2=(
            (d2 >= POSTHOC_POSITIVE_REFERENCE) & (d2 > d1)
        ),
    )

    oracle_df = pd.DataFrame(oracle_rows)
    oracle_df.to_csv(UTILITY_ORACLE, sep="\t", index=False)

    # ------------------------------------------------------------------
    # 2. Marginal + within-scenario comparator decomposition.
    # ------------------------------------------------------------------
    marginal_rows: List[Dict[str, Any]] = []
    within_rows: List[Dict[str, Any]] = []
    quartile_rows: List[Dict[str, Any]] = []
    abs_auc_rows: List[Dict[str, Any]] = []

    for events in EVENT_BUDGETS:
        ep = reps[reps["target_events"] == events].copy()

        for engine in ENGINES:
            compatibility = ep[
                "compatibility_source_train_c"
            ].to_numpy(dtype=float)
            c_transfer = ep[
                f"uno_c_{engine}"
            ].to_numpy(dtype=float)
            c_b0 = ep["uno_c_B0"].to_numpy(dtype=float)
            delta = ep[f"delta_c_{engine}"].to_numpy(dtype=float)

            marginal_rows.append({
                "target_events": events,
                "engine": engine,
                "n_replicates": len(ep),
                "rho_compatibility_vs_C_transfer": spearman(
                    compatibility, c_transfer
                ),
                "rho_compatibility_vs_C_B0": spearman(
                    compatibility, c_b0
                ),
                "rho_compatibility_vs_deltaC": spearman(
                    compatibility, delta
                ),
            })

            # Absolute-C sensitivity: comparator-free and post-hoc only.
            abs_label = np.full(len(ep), -1, dtype=int)
            abs_label[c_transfer >= ABSOLUTE_C_HIGH] = 1
            abs_label[c_transfer <= ABSOLUTE_C_LOW] = 0
            keep = np.isin(abs_label, [0, 1])

            abs_auc_rows.append({
                "target_events": events,
                "engine": engine,
                "label_definition": (
                    f"positive C_transfer>={ABSOLUTE_C_HIGH:.2f}; "
                    f"negative C_transfer<={ABSOLUTE_C_LOW:.2f}; "
                    "middle excluded"
                ),
                "n_labeled": int(np.sum(keep)),
                "n_positive": int(np.sum(abs_label[keep] == 1)),
                "n_negative": int(np.sum(abs_label[keep] == 0)),
                "n_middle_excluded": int(np.sum(abs_label == -1)),
                "compatibility_AUROC_absolute_C_sensitivity": safe_auc(
                    abs_label[keep],
                    compatibility[keep],
                ),
                "can_change_05d0c_STOP_A6": False,
            })

            # Within-scenario correlations across 100/200 replicate realizations.
            for scenario_id, g in ep.groupby("scenario_id"):
                comp_g = g[
                    "compatibility_source_train_c"
                ].to_numpy(dtype=float)
                ct_g = g[
                    f"uno_c_{engine}"
                ].to_numpy(dtype=float)
                b0_g = g["uno_c_B0"].to_numpy(dtype=float)
                d_g = g[f"delta_c_{engine}"].to_numpy(dtype=float)

                within_rows.append({
                    "target_events": events,
                    "engine": engine,
                    "scenario_id": str(scenario_id),
                    "transfer_regime": str(
                        g["transfer_regime"].iloc[0]
                    ),
                    "mechanistic_group": str(
                        g["mechanistic_group"].iloc[0]
                    ),
                    "n_replicates": len(g),
                    "rho_compatibility_vs_C_transfer": spearman(
                        comp_g, ct_g
                    ),
                    "rho_compatibility_vs_C_B0": spearman(
                        comp_g, b0_g
                    ),
                    "rho_compatibility_vs_deltaC": spearman(
                        comp_g, d_g
                    ),
                })

            # Quartiles within event budget.
            temp = ep[
                [
                    "compatibility_source_train_c",
                    f"uno_c_{engine}",
                    "uno_c_B0",
                    f"delta_c_{engine}",
                ]
            ].copy()
            temp = temp[
                np.isfinite(
                    temp["compatibility_source_train_c"]
                )
            ].copy()

            ranks = temp[
                "compatibility_source_train_c"
            ].rank(method="first")

            temp["_quartile"] = pd.qcut(
                ranks,
                q=4,
                labels=["Q1_LOW", "Q2", "Q3", "Q4_HIGH"],
            )

            for q, g in temp.groupby("_quartile", observed=True):
                gd = g[
                    f"delta_c_{engine}"
                ].to_numpy(dtype=float)

                quartile_rows.append({
                    "target_events": events,
                    "engine": engine,
                    "compatibility_quartile": str(q),
                    "n_replicates": len(g),
                    "mean_compatibility": finite_mean(
                        g["compatibility_source_train_c"]
                    ),
                    "mean_C_transfer": finite_mean(
                        g[f"uno_c_{engine}"]
                    ),
                    "mean_C_B0": finite_mean(
                        g["uno_c_B0"]
                    ),
                    "mean_deltaC": finite_mean(
                        g[f"delta_c_{engine}"]
                    ),
                    "fraction_delta_ge_plus_0_02": float(
                        np.mean(
                            gd >= POSTHOC_POSITIVE_REFERENCE
                        )
                    ),
                    "fraction_delta_le_minus_0_02": float(
                        np.mean(
                            gd <= NEGATIVE_TRANSFER_DELTA
                        )
                    ),
                })

    marginal_df = pd.DataFrame(marginal_rows)
    within_df = pd.DataFrame(within_rows)
    quartile_df = pd.DataFrame(quartile_rows)
    abs_auc_df = pd.DataFrame(abs_auc_rows)

    marginal_df.to_csv(MARGINAL_CORR, sep="\t", index=False)
    within_df.to_csv(WITHIN_SCENARIO, sep="\t", index=False)
    quartile_df.to_csv(QUARTILES, sep="\t", index=False)
    abs_auc_df.to_csv(
        ABSOLUTE_C_SENSITIVITY,
        sep="\t",
        index=False,
    )

    # Summarize within-scenario rho distributions.
    within_summary_rows: List[Dict[str, Any]] = []

    for keys, g in within_df.groupby(
        ["target_events", "engine", "mechanistic_group"]
    ):
        events, engine, mech = keys

        for metric in [
            "rho_compatibility_vs_C_transfer",
            "rho_compatibility_vs_C_B0",
            "rho_compatibility_vs_deltaC",
        ]:
            vals = g[metric].to_numpy(dtype=float)
            finite = vals[np.isfinite(vals)]

            within_summary_rows.append({
                "target_events": int(events),
                "engine": str(engine),
                "mechanistic_group": str(mech),
                "metric": metric,
                "n_scenarios": len(g),
                "n_finite_scenarios": len(finite),
                "median_within_scenario_rho": finite_median(finite),
                "q25_within_scenario_rho": finite_quantile(
                    finite, 0.25
                ),
                "q75_within_scenario_rho": finite_quantile(
                    finite, 0.75
                ),
                "fraction_scenarios_rho_positive": (
                    float(np.mean(finite > 0))
                    if len(finite)
                    else float("nan")
                ),
                "fraction_scenarios_rho_negative": (
                    float(np.mean(finite < 0))
                    if len(finite)
                    else float("nan")
                ),
            })

    within_summary_df = pd.DataFrame(within_summary_rows)
    within_summary_df.to_csv(
        WITHIN_SCENARIO_SUMMARY,
        sep="\t",
        index=False,
    )

    # ------------------------------------------------------------------
    # 3. Exact R0-R5 semantics.
    # ------------------------------------------------------------------
    regime_rows: List[Dict[str, Any]] = []

    for events in EVENT_BUDGETS:
        ep = reps[reps["target_events"] == events].copy()

        for engine in ENGINES:
            for regime, g in ep.groupby("transfer_regime"):
                gd = g[
                    f"delta_c_{engine}"
                ].to_numpy(dtype=float)

                regime_rows.append({
                    "target_events": events,
                    "engine": engine,
                    "transfer_regime": str(regime),
                    "mechanistic_group": mechanistic_group(
                        str(regime)
                    ),
                    "n_replicates": len(g),
                    "mean_compatibility": finite_mean(
                        g["compatibility_source_train_c"]
                    ),
                    "mean_C_transfer": finite_mean(
                        g[f"uno_c_{engine}"]
                    ),
                    "mean_C_B0": finite_mean(
                        g["uno_c_B0"]
                    ),
                    "mean_deltaC": finite_mean(gd),
                    "fraction_delta_ge_plus_0_02": float(
                        np.mean(
                            gd >= POSTHOC_POSITIVE_REFERENCE
                        )
                    ),
                    "fraction_delta_le_minus_0_02": float(
                        np.mean(
                            gd <= NEGATIVE_TRANSFER_DELTA
                        )
                    ),
                })

    regime_df = pd.DataFrame(regime_rows)
    regime_df.to_csv(EXACT_REGIMES, sep="\t", index=False)

    # ------------------------------------------------------------------
    # 4. Complete event-count-only ladder with regime-specific NT.
    # ------------------------------------------------------------------
    ladder_rows: List[Dict[str, Any]] = []

    for engine in ENGINES:
        for threshold in EVENT_POLICY_THRESHOLDS:
            temp = reps[
                [
                    "scenario_id",
                    "target_events",
                    "mechanistic_group",
                    f"delta_c_{engine}",
                ]
            ].copy()

            use = (
                temp["target_events"].to_numpy(dtype=int)
                >= threshold
            )
            d = temp[
                f"delta_c_{engine}"
            ].to_numpy(dtype=float)

            temp["_policy_delta"] = np.where(
                use,
                d,
                0.0,
            )
            temp["_negative"] = (
                temp["_policy_delta"]
                <= NEGATIVE_TRANSFER_DELTA
            ).astype(float)
            temp["_active"] = use.astype(float)

            compatible = temp[
                temp["mechanistic_group"]
                == "MECHANISTICALLY_COMPATIBLE_R0_R3"
            ]
            incompatible = temp[
                temp["mechanistic_group"]
                == "MECHANISTICALLY_INCOMPATIBLE_R4_R5"
            ]

            ladder_rows.append({
                "engine": engine,
                "minimum_events_to_transfer": threshold,
                "always_abstain": bool(threshold == 41),
                "equal_scenario_activation_fraction": scenario_equal_mean(
                    temp, "_active"
                ),
                "equal_scenario_mean_delta_c_vs_B0": scenario_equal_mean(
                    temp, "_policy_delta"
                ),
                "aggregate_negative_transfer_rate": scenario_equal_rate(
                    temp, "_negative"
                ),
                "R0_R3_mechanistically_compatible_mean_delta_c": (
                    scenario_equal_mean(
                        compatible,
                        "_policy_delta",
                    )
                ),
                "R0_R3_mechanistically_compatible_NT_rate": (
                    scenario_equal_rate(
                        compatible,
                        "_negative",
                    )
                ),
                "R4_R5_mechanistically_incompatible_mean_delta_c": (
                    scenario_equal_mean(
                        incompatible,
                        "_policy_delta",
                    )
                ),
                "R4_R5_mechanistically_incompatible_NT_rate": (
                    scenario_equal_rate(
                        incompatible,
                        "_negative",
                    )
                ),
            })

    ladder_df = pd.DataFrame(ladder_rows)

    # Exact replay of existing full 05d0c aggregate ladder.
    prior = prior_policies[
        [
            "engine",
            "minimum_events_to_transfer",
            "equal_scenario_mean_delta_c_vs_B0",
            "aggregate_negative_transfer_rate",
        ]
    ].copy()

    replay = ladder_df.merge(
        prior,
        on=[
            "engine",
            "minimum_events_to_transfer",
        ],
        how="inner",
        suffixes=("_replay", "_05d0c"),
        validate="one_to_one",
    )

    replay["delta_abs_diff"] = np.abs(
        replay[
            "equal_scenario_mean_delta_c_vs_B0_replay"
        ]
        - replay[
            "equal_scenario_mean_delta_c_vs_B0_05d0c"
        ]
    )
    replay["NT_abs_diff"] = np.abs(
        replay[
            "aggregate_negative_transfer_rate_replay"
        ]
        - replay[
            "aggregate_negative_transfer_rate_05d0c"
        ]
    )

    ladder_replay_pass = bool(
        len(replay) == len(ladder_df)
        and (replay["delta_abs_diff"] <= 1e-10).all()
        and (replay["NT_abs_diff"] <= 1e-10).all()
    )

    ladder_df.to_csv(FULL_LADDER, sep="\t", index=False)

    # ------------------------------------------------------------------
    # Summary.
    # ------------------------------------------------------------------
    marginal29 = marginal_df[
        marginal_df["target_events"] == PRIMARY_EVENT_BUDGET
    ].set_index("engine")

    abs29 = abs_auc_df[
        abs_auc_df["target_events"] == PRIMARY_EVENT_BUDGET
    ].set_index("engine")

    best_oracle = oracle_df[
        oracle_df["policy"]
        == "BEST_OF_B0_A1_A2_PER_REPLICATE_ORACLE"
    ].iloc[0]

    summary = {
        "script_version": SCRIPT_VERSION,
        "status": "PASS" if ladder_replay_pass else "HOLD",
        "scientific_status": (
            "PASS_POSTHOC_UTILITY_COMPARATOR_WITHINSCENARIO_PROVENANCE_AUDIT_COMPLETE"
            if ladder_replay_pass
            else "HOLD_EVENT_LADDER_REPLAY_DISCREPANCY"
        ),
        "run_started_utc": started,
        "run_finished_utc": now_utc(),
        "old_05d0e_preserved": True,
        "old_05d0e_summary_sha256": sha256_file(
            OLD_D0E_SUMMARY
        ),
        "prior_05d0f_v1_preserved": True,
        "prior_05d0f_v1_summary_sha256": sha256_file(
            PREV_D0F_SUMMARY
        ),
        "A6_status_unchanged": EXPECTED_D0C_STATUS,
        "best_B0_A1_A2_outcome_selected_upper_bound": {
            "mean_delta_c_vs_B0": float(
                best_oracle[
                    "equal_scenario_mean_delta_c_vs_B0"
                ]
            ),
            "negative_transfer_rate": float(
                best_oracle[
                    "aggregate_negative_transfer_rate"
                ]
            ),
            "activation_fraction": float(
                best_oracle[
                    "equal_scenario_activation_fraction"
                ]
            ),
            "reaches_posthoc_plus_0_02_reference": bool(
                best_oracle[
                    "reaches_posthoc_mean_plus_0_02_reference"
                ]
            ),
            "scope": (
                "evaluated B0/A1/A2 family; frozen 180-scenario grid; "
                "equal-scenario weighting; selected on same test outcome; "
                "optimistically biased upper bound"
            ),
        },
        "marginal_comparator_decomposition_at_29": {
            engine: {
                "rho_comp_C_transfer": float(
                    marginal29.loc[
                        engine,
                        "rho_compatibility_vs_C_transfer",
                    ]
                ),
                "rho_comp_C_B0": float(
                    marginal29.loc[
                        engine,
                        "rho_compatibility_vs_C_B0",
                    ]
                ),
                "rho_comp_deltaC": float(
                    marginal29.loc[
                        engine,
                        "rho_compatibility_vs_deltaC",
                    ]
                ),
            }
            for engine in ENGINES
        },
        "absolute_C_sensitivity_AUROC_at_29": {
            engine: float(
                abs29.loc[
                    engine,
                    "compatibility_AUROC_absolute_C_sensitivity",
                ]
            )
            for engine in ENGINES
        },
        "complete_event_ladder_exact_replay": (
            ladder_replay_pass
        ),
        "guardrails": {
            "A6_STOP_cannot_be_reopened_by_posthoc_sensitivity": True,
            "absolute_C_sensitivity_is_descriptive_only": True,
            "oracle_is_winners_curse_upper_bound": True,
            "plus_0_02_is_not_frozen_05a_positive_utility_rule": True,
            "R0_R3_not_called_beneficial": True,
            "human_outcomes_read": False,
        },
        "final_artifact_hashes": {
            "diagnostic_nomenclature_erratum.json": sha256_file(
                ERRATUM
            ),
            "threshold_0p02_provenance.tsv": sha256_file(
                THRESHOLD_PROVENANCE
            ),
            "posthoc_plus002_utility_oracle_upper_bounds.tsv": sha256_file(
                UTILITY_ORACLE
            ),
            "marginal_comparator_decomposition.tsv": sha256_file(
                MARGINAL_CORR
            ),
            "within_scenario_comparator_decomposition.tsv": sha256_file(
                WITHIN_SCENARIO
            ),
            "within_scenario_correlation_summary.tsv": sha256_file(
                WITHIN_SCENARIO_SUMMARY
            ),
            "absolute_C_operational_sensitivity.tsv": sha256_file(
                ABSOLUTE_C_SENSITIVITY
            ),
            "compatibility_quartile_comparator_decomposition.tsv": sha256_file(
                QUARTILES
            ),
            "exact_regime_A1_A2_outcomes.tsv": sha256_file(
                EXACT_REGIMES
            ),
            "complete_A1_A2_event_abstention_ladder_regime_specific.tsv": sha256_file(
                FULL_LADDER
            ),
        },
    }
    write_json(SUMMARY_JSON, summary)

    # ------------------------------------------------------------------
    # Console.
    # ------------------------------------------------------------------
    print("=" * 120)
    print("05d0g ERRATUM / 0.02 PROVENANCE")
    print("=" * 120)
    print(
        "+0.02 positive utility: POST-HOC 05d0c operational reference; "
        "not a frozen 05a mean-gain selection threshold."
    )
    print(
        "R0-R3: MECHANISTICALLY COMPATIBLE; not operationally beneficial by definition."
    )
    print()
    print(provenance_df.to_string(index=False))

    print()
    print("=" * 120)
    print("05d0g POST-HOC +0.02 UTILITY-REFERENCE ORACLE UPPER BOUNDS")
    print("=" * 120)
    print(oracle_df.to_string(index=False))

    print()
    print("=" * 120)
    print("05d0g MARGINAL COMPARATOR DECOMPOSITION")
    print("=" * 120)
    print(marginal_df.to_string(index=False))

    print()
    print("=" * 120)
    print("05d0g WITHIN-SCENARIO CORRELATION SUMMARY @ 29 EVENTS")
    print("=" * 120)
    print(
        within_summary_df[
            within_summary_df["target_events"]
            == PRIMARY_EVENT_BUDGET
        ].to_string(index=False)
    )

    print()
    print("=" * 120)
    print("05d0g ABSOLUTE-C SENSITIVITY")
    print("=" * 120)
    print(abs_auc_df.to_string(index=False))
    print()
    print(
        "Sensitivity definition frozen in this script before calculation: "
        f"positive C>={ABSOLUTE_C_HIGH:.2f}, negative C<={ABSOLUTE_C_LOW:.2f}."
    )
    print("This sensitivity CANNOT reopen A6.")

    print()
    print("=" * 120)
    print("05d0g EXACT R0-R5 OUTCOMES @ 29 EVENTS")
    print("=" * 120)
    print(
        regime_df[
            regime_df["target_events"]
            == PRIMARY_EVENT_BUDGET
        ].to_string(index=False)
    )

    print()
    print("=" * 120)
    print("05d0g COMPLETE A1/A2 EVENT LADDER + REGIME-SPECIFIC NT")
    print("=" * 120)
    print(ladder_df.to_string(index=False))
    print()
    print(
        "Exact replay of existing 05d0c aggregate ladder: "
        + ("PASS" if ladder_replay_pass else "FAIL")
    )

    print()
    print("=" * 120)
    print("05d0g SUMMARY")
    print("=" * 120)

    upper = summary["best_B0_A1_A2_outcome_selected_upper_bound"]
    print(
        "Best B0/A1/A2 outcome-selected oracle upper bound "
        "[optimistically biased]:"
    )
    print(
        f"  mean deltaC={upper['mean_delta_c_vs_B0']:.4f}, "
        f"NT={upper['negative_transfer_rate']:.4f}, "
        f"activation={upper['activation_fraction']:.4f}, "
        f"reaches post-hoc +0.02={upper['reaches_posthoc_plus_0_02_reference']}"
    )
    print()

    for engine in ENGINES:
        x = summary[
            "marginal_comparator_decomposition_at_29"
        ][engine]
        print(
            f"{engine} @29 marginal: "
            f"rho(comp,C_transfer)={x['rho_comp_C_transfer']:.4f}, "
            f"rho(comp,C_B0)={x['rho_comp_C_B0']:.4f}, "
            f"rho(comp,deltaC)={x['rho_comp_deltaC']:.4f}"
        )

    print()
    print(
        "Absolute-C sensitivity AUROC @29 [post-hoc only]: "
        + ", ".join(
            f"{engine}="
            f"{summary['absolute_C_sensitivity_AUROC_at_29'][engine]:.4f}"
            for engine in ENGINES
        )
    )
    print()
    print("A6 remains STOP for Paper 6: YES")
    print("Old 05d0e preserved: YES")
    print("Previous 05d0f v1 preserved: YES")
    print("Human outcomes read: NO")
    print("=" * 120)

    if not ladder_replay_pass:
        raise RuntimeError(
            "Complete event-count ladder failed exact replay of 05d0c."
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 120, file=sys.stderr)
        print("05d0g audit: FAIL", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print("=" * 120, file=sys.stderr)
        raise
