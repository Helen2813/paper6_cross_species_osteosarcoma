#!/usr/bin/env python3
"""
Paper 6 - 05d0f audit of post-hoc utility-reference feasibility,
comparator decomposition, and regime-specific safety.

Chronology
----------
This stage is AFTER the already-completed old 05d0e regime-semantics audit.
That audit is preserved and required as an input. Nothing in its output is
overwritten.

Known state:
- 05d: HOLD_NO_SAFE_SELECTABLE_ARCHITECTURE
- 05d0c: STOP_A6_FOR_PAPER6
- old 05d0e: regime-semantics audit complete; raw operational inversion is
  largely between-regime for A1 and weak after regime residualization for A2.

IMPORTANT TERMINOLOGY CORRECTION
--------------------------------
The original frozen 05a simulation-selection contract did NOT impose a
positive mean utility threshold of +0.02.

The +0.02 value was introduced later in 05d0c as an OPERATIONAL CLASS LABEL:
    beneficial if independent-test delta C(engine-B0) >= +0.02
and 05d0d subsequently described it too strongly as a "frozen practical gain."

Therefore this script treats +0.02 ONLY as a POST-HOC OPERATIONAL/UTILITY
REFERENCE. It is not used to alter any frozen 05a/05d decision.

Questions
---------
1. Within already-existing A1/A2 predictions, is a mean +0.02 gain attainable
   under perfect independent-test-outcome oracle selection?
   This is an upper-bound diagnostic only, never an implementable policy.

2. Is the negative raw association
       compatibility vs deltaC = C(transfer) - C(B0)
   caused by:
       a) C(transfer) decreasing with compatibility,
       b) C(B0) increasing faster with compatibility,
       c) both?

3. Quantify the exact R0-R5 semantics without calling R0-R3 "beneficial."
   R0-R3 are MECHANISTICALLY COMPATIBLE generator regimes only.

4. Reconstruct the COMPLETE A1/A2 event-count abstention ladder and report
   negative-transfer rate separately in R0-R3 and R4-R5.

No model fitting.
No new trust signal.
No threshold tuning.
No human outcomes.
No GPU.
A6 remains STOP.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


SCRIPT_VERSION = (
    "05d0f-utility-reference-comparator-decomposition-regime-safety-v1-no-cli"
)

ROOT = Path(__file__).resolve().parents[1]

A_DIR = ROOT / "results" / "simulation_contract" / "05a"
A_CONTRACT = A_DIR / "negative_transfer_simulation_contract.json"

D0C_DIR = ROOT / "results" / "simulation_phase_diagram_diagnostics" / "05d0c"
D0C_REPS = D0C_DIR / "global_trust_replicate_statistics.tsv"
D0C_POLICY = D0C_DIR / "event_count_only_abstention_policies.tsv"
D0C_SUMMARY = D0C_DIR / "summary.json"

D0D_DIR = ROOT / "results" / "simulation_phase_diagram_diagnostics" / "05d0d"
D0D_SUMMARY = D0D_DIR / "summary.json"

# Preserve and require the already-run OLD 05d0e.
OLD_D0E_DIR = ROOT / "results" / "simulation_phase_diagram_diagnostics" / "05d0e"
OLD_D0E_SUMMARY = OLD_D0E_DIR / "summary.json"

OUT_DIR = ROOT / "results" / "simulation_phase_diagram_diagnostics" / "05d0f"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEMANTICS_CORRECTION = OUT_DIR / "posthoc_threshold_and_regime_semantics_correction.json"
UTILITY_ORACLE = OUT_DIR / "posthoc_plus002_utility_oracle_feasibility.tsv"
CORR_DECOMP = OUT_DIR / "compatibility_comparator_decomposition.tsv"
QUARTILES = OUT_DIR / "compatibility_quartile_comparator_decomposition.tsv"
EXACT_REGIMES = OUT_DIR / "exact_regime_A1_A2_outcomes.tsv"
FULL_LADDER = OUT_DIR / "complete_A1_A2_event_abstention_ladder_regime_specific.tsv"
SUMMARY_JSON = OUT_DIR / "summary.json"

EXPECTED_D0C_STATUS = "STOP_A6_FOR_PAPER6"
EXPECTED_D0D_STATUS = "PASS_POSTHOC_MECHANISTIC_OPERATIONAL_CROSSWALK_COMPLETE"
EXPECTED_OLD_D0E_STATUS = (
    "PASS_POSTHOC_REGIME_SEMANTICS_AUDIT_COMPLETE_A6_REMAINS_STOP"
)

ENGINES = ["A1", "A2"]
EVENT_BUDGETS = [5, 10, 15, 20, 29, 40]
EVENT_POLICY_THRESHOLDS = [5, 10, 15, 20, 29, 40, 41]

PRIMARY_EVENT_BUDGET = 29

# Frozen negative-transfer definition from 05a.
NEGATIVE_TRANSFER_DELTA = -0.02

# POST-HOC operational positive reference from 05d0c.
# This is NOT a frozen 05a positive-utility selection threshold.
POSTHOC_POSITIVE_REFERENCE = +0.02


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


def mechanistic_group(regime: str) -> str:
    r = str(regime)
    if r.startswith(("R0_", "R1_", "R2_", "R3_")):
        return "MECHANISTICALLY_COMPATIBLE_R0_R3"
    if r.startswith(("R4_", "R5_")):
        return "MECHANISTICALLY_INCOMPATIBLE_R4_R5"
    raise RuntimeError(f"Unrecognized transfer regime: {regime}")


def scenario_equal_mean(frame: pd.DataFrame, value_col: str) -> float:
    x = (
        frame.groupby("scenario_id", as_index=False)[value_col]
        .mean()[value_col]
        .to_numpy(dtype=float)
    )
    return finite_mean(x)


def scenario_equal_rate(frame: pd.DataFrame, indicator_col: str) -> float:
    x = (
        frame.groupby("scenario_id", as_index=False)[indicator_col]
        .mean()[indicator_col]
        .to_numpy(dtype=float)
    )
    return finite_mean(x)


def main() -> None:
    started = now_utc()

    print("=" * 120)
    print("Paper 6 - 05d0f utility-reference / comparator / regime-safety audit")
    print("=" * 120)
    print(f"Script version: {SCRIPT_VERSION}")
    print()
    print("Scope:")
    print("  Old 05d0e preserved: YES")
    print("  New model fitting: NO")
    print("  +0.02 treated as frozen 05a utility criterion: NO")
    print("  +0.02 treated as post-hoc operational reference: YES")
    print("  A6 reopened: NO")
    print("  Human outcomes: NO")
    print("  GPU: NO")
    print()

    for p in [
        A_CONTRACT,
        D0C_REPS,
        D0C_POLICY,
        D0C_SUMMARY,
        D0D_SUMMARY,
        OLD_D0E_SUMMARY,
    ]:
        require_file(p)

    d0c = read_json(D0C_SUMMARY)
    d0d = read_json(D0D_SUMMARY)
    old_d0e = read_json(OLD_D0E_SUMMARY)
    contract = read_json(A_CONTRACT)

    if str(d0c.get("scientific_status")) != EXPECTED_D0C_STATUS:
        raise RuntimeError("05d0c is not STOP_A6_FOR_PAPER6.")
    if str(d0d.get("scientific_status")) != EXPECTED_D0D_STATUS:
        raise RuntimeError("05d0d crosswalk diagnostic is not complete.")
    if str(old_d0e.get("scientific_status")) != EXPECTED_OLD_D0E_STATUS:
        raise RuntimeError(
            "The already-run old 05d0e regime-semantics audit is not in expected PASS state."
        )

    reps = pd.read_csv(D0C_REPS, sep="\t")
    prior_policies = pd.read_csv(D0C_POLICY, sep="\t")

    if len(reps) != 21600:
        raise RuntimeError(
            f"Expected 21,600 05d0c replicate rows; observed {len(reps)}."
        )

    reps["mechanistic_group"] = [
        mechanistic_group(x)
        for x in reps["transfer_regime"].astype(str)
    ]

    # ------------------------------------------------------------------
    # Explicit provenance correction of +0.02 terminology.
    # Do not assume internal threshold-key naming beyond preserving the full
    # actual frozen threshold registry.
    # ------------------------------------------------------------------
    selection_thresholds = contract.get("selection_thresholds", {})

    semantics = {
        "script_version": SCRIPT_VERSION,
        "created_utc": now_utc(),
        "05a_contract_sha256": sha256_file(A_CONTRACT),
        "actual_05a_selection_threshold_registry": selection_thresholds,
        "posthoc_positive_reference": POSTHOC_POSITIVE_REFERENCE,
        "correction": {
            "plus_0_02": (
                "The +0.02 positive value used in 05d0c/05d0d is a post-result "
                "operational benefit label/reference. It is NOT asserted here "
                "to be a frozen 05a mean-utility selection threshold."
            ),
            "R0_R3": (
                "R0-R3 are MECHANISTICALLY COMPATIBLE generator regimes, "
                "not 'beneficial regimes' for a particular transfer architecture."
            ),
            "numerical_results_changed": False,
        },
        "old_05d0e_preserved": True,
        "old_05d0e_summary_sha256": sha256_file(OLD_D0E_SUMMARY),
    }
    write_json(SEMANTICS_CORRECTION, semantics)

    # ------------------------------------------------------------------
    # 1. POST-HOC +0.02 utility-reference feasibility.
    #
    # These are upper-bound ORACLES because they inspect independent-test delta.
    # They can describe attainable utility inside existing A1/A2 predictions,
    # but are not implementable policies.
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

        oracle_rows.append(
            {
                "policy": policy,
                "engine_scope": engine_scope,
                "uses_independent_test_outcome_for_selection": True,
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
            }
        )

    d1 = reps["delta_c_A1"].to_numpy(dtype=float)
    d2 = reps["delta_c_A2"].to_numpy(dtype=float)

    for engine, d in [("A1", d1), ("A2", d2)]:
        append_oracle(
            f"{engine}_ALWAYS_TRANSFER_REPLAY",
            engine,
            d,
            np.ones(len(d), dtype=float),
        )

        # Perfect sign oracle: B0 whenever transfer would not improve test C.
        append_oracle(
            f"{engine}_PERFECT_POSITIVE_DELTA_ORACLE",
            engine,
            np.where(d > 0.0, d, 0.0),
            (d > 0.0).astype(float),
        )

        # Same concept, requiring realized test delta >= post-hoc +0.02 reference.
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

    # Strongest attainable upper envelope among existing B0/A1/A2 predictions.
    best_delta = np.maximum.reduce(
        [
            np.zeros(len(reps), dtype=float),
            d1,
            d2,
        ]
    )
    choose_a1 = (d1 > 0.0) & (d1 >= d2)
    choose_a2 = (d2 > 0.0) & (d2 > d1)

    append_oracle(
        "BEST_OF_B0_A1_A2_PER_REPLICATE_ORACLE",
        "B0/A1/A2",
        best_delta,
        (best_delta > 0.0).astype(float),
        select_a1=choose_a1,
        select_a2=choose_a2,
    )

    best_delta02 = np.maximum.reduce(
        [
            np.zeros(len(reps), dtype=float),
            np.where(
                d1 >= POSTHOC_POSITIVE_REFERENCE,
                d1,
                0.0,
            ),
            np.where(
                d2 >= POSTHOC_POSITIVE_REFERENCE,
                d2,
                0.0,
            ),
        ]
    )
    choose_a1_02 = (
        (d1 >= POSTHOC_POSITIVE_REFERENCE)
        & (d1 >= d2)
    )
    choose_a2_02 = (
        (d2 >= POSTHOC_POSITIVE_REFERENCE)
        & (d2 > d1)
    )

    append_oracle(
        "BEST_OF_B0_A1_A2_DELTA_GE_0_02_ORACLE",
        "B0/A1/A2",
        best_delta02,
        (best_delta02 > 0.0).astype(float),
        select_a1=choose_a1_02,
        select_a2=choose_a2_02,
    )

    oracle_df = pd.DataFrame(oracle_rows)
    oracle_df.to_csv(UTILITY_ORACLE, sep="\t", index=False)

    # ------------------------------------------------------------------
    # 2. Comparator decomposition.
    # ------------------------------------------------------------------
    corr_rows: List[Dict[str, Any]] = []
    quartile_rows: List[Dict[str, Any]] = []

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

            rho_transfer = spearman(
                compatibility,
                c_transfer,
            )
            rho_b0 = spearman(
                compatibility,
                c_b0,
            )
            rho_delta = spearman(
                compatibility,
                delta,
            )

            if (
                np.isfinite(rho_transfer)
                and np.isfinite(rho_b0)
                and np.isfinite(rho_delta)
                and rho_transfer > 0
                and rho_b0 > rho_transfer
                and rho_delta < 0
            ):
                interpretation = "COMPARATOR_COMPRESSION_COMPATIBLE"
            elif (
                np.isfinite(rho_transfer)
                and np.isfinite(rho_delta)
                and rho_transfer < 0
                and rho_delta < 0
            ):
                interpretation = "TRANSFER_C_ITSELF_INVERSE"
            elif (
                np.isfinite(rho_b0)
                and np.isfinite(rho_delta)
                and rho_b0 > 0
                and rho_delta < 0
            ):
                interpretation = "MIXED_WITH_POSITIVE_B0_ASSOCIATION"
            else:
                interpretation = "NO_SINGLE_SIMPLE_PATTERN"

            corr_rows.append(
                {
                    "target_events": events,
                    "engine": engine,
                    "n_replicates": len(ep),
                    "rho_compatibility_vs_C_transfer": rho_transfer,
                    "rho_compatibility_vs_C_B0": rho_b0,
                    "rho_compatibility_vs_deltaC": rho_delta,
                    "diagnostic_interpretation": interpretation,
                }
            )

            # Fixed rank quartiles within event budget.
            temp = ep[
                [
                    "compatibility_source_train_c",
                    f"uno_c_{engine}",
                    "uno_c_B0",
                    f"delta_c_{engine}",
                ]
            ].copy()

            finite = np.isfinite(
                temp["compatibility_source_train_c"].to_numpy(dtype=float)
            )
            temp = temp.loc[finite].copy()

            ranks = temp[
                "compatibility_source_train_c"
            ].rank(method="first")

            temp["_quartile"] = pd.qcut(
                ranks,
                q=4,
                labels=["Q1_LOW", "Q2", "Q3", "Q4_HIGH"],
            )

            for quartile, g in temp.groupby(
                "_quartile",
                observed=True,
            ):
                gd = g[f"delta_c_{engine}"].to_numpy(dtype=float)

                quartile_rows.append(
                    {
                        "target_events": events,
                        "engine": engine,
                        "compatibility_quartile": str(quartile),
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
                        "fraction_operational_beneficial_ge_plus_0_02": float(
                            np.mean(
                                gd
                                >= POSTHOC_POSITIVE_REFERENCE
                            )
                        ),
                        "fraction_operational_harmful_le_minus_0_02": float(
                            np.mean(
                                gd
                                <= NEGATIVE_TRANSFER_DELTA
                            )
                        ),
                    }
                )

    corr_df = pd.DataFrame(corr_rows)
    quartile_df = pd.DataFrame(quartile_rows)

    corr_df.to_csv(CORR_DECOMP, sep="\t", index=False)
    quartile_df.to_csv(QUARTILES, sep="\t", index=False)

    # ------------------------------------------------------------------
    # 3. Exact regime semantics: never call R0-R3 "beneficial."
    # ------------------------------------------------------------------
    regime_rows: List[Dict[str, Any]] = []

    for events in EVENT_BUDGETS:
        ep = reps[reps["target_events"] == events].copy()

        for engine in ENGINES:
            for regime, g in ep.groupby("transfer_regime"):
                gd = g[f"delta_c_{engine}"].to_numpy(dtype=float)

                regime_rows.append(
                    {
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
                        "median_deltaC": finite_median(gd),
                        "fraction_operational_beneficial_ge_plus_0_02": float(
                            np.mean(
                                gd >= POSTHOC_POSITIVE_REFERENCE
                            )
                        ),
                        "fraction_operational_harmful_le_minus_0_02": float(
                            np.mean(
                                gd <= NEGATIVE_TRANSFER_DELTA
                            )
                        ),
                    }
                )

    regime_df = pd.DataFrame(regime_rows)
    regime_df.to_csv(EXACT_REGIMES, sep="\t", index=False)

    # ------------------------------------------------------------------
    # 4. Complete event-count abstention ladder with regime-specific safety.
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

            use_transfer = (
                temp["target_events"].to_numpy(dtype=int)
                >= threshold
            )
            engine_delta = temp[
                f"delta_c_{engine}"
            ].to_numpy(dtype=float)

            temp["_policy_delta"] = np.where(
                use_transfer,
                engine_delta,
                0.0,
            )
            temp["_negative"] = (
                temp["_policy_delta"]
                <= NEGATIVE_TRANSFER_DELTA
            ).astype(float)
            temp["_active"] = use_transfer.astype(float)

            compatible = temp[
                temp["mechanistic_group"]
                == "MECHANISTICALLY_COMPATIBLE_R0_R3"
            ].copy()
            incompatible = temp[
                temp["mechanistic_group"]
                == "MECHANISTICALLY_INCOMPATIBLE_R4_R5"
            ].copy()

            ladder_rows.append(
                {
                    "engine": engine,
                    "minimum_events_to_transfer": threshold,
                    "always_abstain": bool(threshold == 41),
                    "equal_scenario_activation_fraction": scenario_equal_mean(
                        temp,
                        "_active",
                    ),
                    "equal_scenario_mean_delta_c_vs_B0": scenario_equal_mean(
                        temp,
                        "_policy_delta",
                    ),
                    "aggregate_negative_transfer_rate": scenario_equal_rate(
                        temp,
                        "_negative",
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
                }
            )

    ladder_df = pd.DataFrame(ladder_rows)

    # Verify against the already-existing complete 05d0c aggregate ladder.
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

    replay["delta_abs_difference"] = np.abs(
        replay[
            "equal_scenario_mean_delta_c_vs_B0_replay"
        ]
        - replay[
            "equal_scenario_mean_delta_c_vs_B0_05d0c"
        ]
    )
    replay["NT_abs_difference"] = np.abs(
        replay[
            "aggregate_negative_transfer_rate_replay"
        ]
        - replay[
            "aggregate_negative_transfer_rate_05d0c"
        ]
    )

    policy_replay_pass = bool(
        len(replay) == len(ladder_df)
        and (replay["delta_abs_difference"] <= 1e-10).all()
        and (replay["NT_abs_difference"] <= 1e-10).all()
    )

    ladder_df.to_csv(FULL_LADDER, sep="\t", index=False)

    # ------------------------------------------------------------------
    # Summary.
    # ------------------------------------------------------------------
    corr29 = corr_df[
        corr_df["target_events"] == PRIMARY_EVENT_BUDGET
    ].set_index("engine")

    best_oracle = oracle_df[
        oracle_df["policy"]
        == "BEST_OF_B0_A1_A2_PER_REPLICATE_ORACLE"
    ].iloc[0]

    best02_oracle = oracle_df[
        oracle_df["policy"]
        == "BEST_OF_B0_A1_A2_DELTA_GE_0_02_ORACLE"
    ].iloc[0]

    summary = {
        "script_version": SCRIPT_VERSION,
        "status": "PASS" if policy_replay_pass else "HOLD",
        "scientific_status": (
            "PASS_POSTHOC_UTILITY_COMPARATOR_REGIME_SAFETY_AUDIT_COMPLETE"
            if policy_replay_pass
            else "HOLD_EVENT_POLICY_REPLAY_DISCREPANCY"
        ),
        "run_started_utc": started,
        "run_finished_utc": now_utc(),
        "old_05d0e_preserved_and_required": True,
        "old_05d0e_summary_sha256": sha256_file(
            OLD_D0E_SUMMARY
        ),
        "A6_status_unchanged": EXPECTED_D0C_STATUS,
        "plus_0_02_semantics": (
            "POSTHOC_OPERATIONAL_REFERENCE_NOT_FROZEN_05a_MEAN_UTILITY_THRESHOLD"
        ),
        "best_existing_B0_A1_A2_positive_delta_oracle": {
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
        },
        "best_existing_B0_A1_A2_delta_ge_0_02_oracle": {
            "mean_delta_c_vs_B0": float(
                best02_oracle[
                    "equal_scenario_mean_delta_c_vs_B0"
                ]
            ),
            "activation_fraction": float(
                best02_oracle[
                    "equal_scenario_activation_fraction"
                ]
            ),
            "reaches_posthoc_plus_0_02_reference": bool(
                best02_oracle[
                    "reaches_posthoc_mean_plus_0_02_reference"
                ]
            ),
        },
        "correlation_decomposition_at_29": {
            engine: {
                "rho_compatibility_vs_C_transfer": float(
                    corr29.loc[
                        engine,
                        "rho_compatibility_vs_C_transfer",
                    ]
                ),
                "rho_compatibility_vs_C_B0": float(
                    corr29.loc[
                        engine,
                        "rho_compatibility_vs_C_B0",
                    ]
                ),
                "rho_compatibility_vs_deltaC": float(
                    corr29.loc[
                        engine,
                        "rho_compatibility_vs_deltaC",
                    ]
                ),
                "diagnostic_interpretation": str(
                    corr29.loc[
                        engine,
                        "diagnostic_interpretation",
                    ]
                ),
            }
            for engine in ENGINES
        },
        "complete_event_ladder_exactly_replays_05d0c": (
            policy_replay_pass
        ),
        "interpretation_guardrails": {
            "oracle_uses_test_outcomes_and_is_not_implementable": True,
            "plus_0_02_not_frozen_05a_positive_utility_threshold": True,
            "R0_R3_not_operationally_beneficial_by_definition": True,
            "correlations_descriptive_not_causal": True,
            "old_05d0e_not_overwritten": True,
            "A6_not_reopened": True,
            "human_outcomes_read": False,
        },
        "final_artifact_hashes": {
            "posthoc_threshold_and_regime_semantics_correction.json": sha256_file(
                SEMANTICS_CORRECTION
            ),
            "posthoc_plus002_utility_oracle_feasibility.tsv": sha256_file(
                UTILITY_ORACLE
            ),
            "compatibility_comparator_decomposition.tsv": sha256_file(
                CORR_DECOMP
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
    print("05d0f TERMINOLOGY / CONTRACT CORRECTION")
    print("=" * 120)
    print(
        "+0.02: POST-HOC operational positive reference from 05d0c; "
        "NOT a frozen 05a positive mean-utility selection threshold."
    )
    print(
        "R0-R3: MECHANISTICALLY COMPATIBLE generator regimes; "
        "NOT operationally beneficial regimes."
    )
    print("Old 05d0e outputs preserved: YES")

    print()
    print("=" * 120)
    print("05d0f POST-HOC +0.02 UTILITY-REFERENCE ORACLE FEASIBILITY")
    print("=" * 120)
    print(oracle_df.to_string(index=False))

    print()
    print("=" * 120)
    print("05d0f CORRELATION DECOMPOSITION")
    print("=" * 120)
    print(corr_df.to_string(index=False))

    print()
    print("=" * 120)
    print("05d0f COMPATIBILITY QUARTILES @ 29 EVENTS")
    print("=" * 120)
    print(
        quartile_df[
            quartile_df["target_events"]
            == PRIMARY_EVENT_BUDGET
        ].to_string(index=False)
    )

    print()
    print("=" * 120)
    print("05d0f EXACT R0-R5 OUTCOMES @ 29 EVENTS")
    print("=" * 120)
    print(
        regime_df[
            regime_df["target_events"]
            == PRIMARY_EVENT_BUDGET
        ].to_string(index=False)
    )

    print()
    print("=" * 120)
    print("05d0f COMPLETE A1/A2 EVENT-COUNT LADDER + REGIME-SPECIFIC NT")
    print("=" * 120)
    print(ladder_df.to_string(index=False))
    print()
    print(
        "Exact replay of prior 05d0c aggregate ladder: "
        + ("PASS" if policy_replay_pass else "FAIL")
    )

    print()
    print("=" * 120)
    print("05d0f SUMMARY")
    print("=" * 120)
    print(
        "Best perfect B0/A1/A2 positive-delta oracle: "
        f"mean deltaC="
        f"{summary['best_existing_B0_A1_A2_positive_delta_oracle']['mean_delta_c_vs_B0']:.4f}, "
        f"NT="
        f"{summary['best_existing_B0_A1_A2_positive_delta_oracle']['negative_transfer_rate']:.4f}, "
        f"activation="
        f"{summary['best_existing_B0_A1_A2_positive_delta_oracle']['activation_fraction']:.4f}, "
        f"reaches descriptive +0.02="
        f"{summary['best_existing_B0_A1_A2_positive_delta_oracle']['reaches_posthoc_plus_0_02_reference']}"
    )

    print()
    for engine in ENGINES:
        x = summary["correlation_decomposition_at_29"][engine]
        print(
            f"{engine} @29: "
            f"rho(comp,C_transfer)={x['rho_compatibility_vs_C_transfer']:.4f}, "
            f"rho(comp,C_B0)={x['rho_compatibility_vs_C_B0']:.4f}, "
            f"rho(comp,deltaC)={x['rho_compatibility_vs_deltaC']:.4f} "
            f"[{x['diagnostic_interpretation']}]"
        )

    print()
    print("A6 remains STOP for Paper 6: YES")
    print("Old 05d0e preserved: YES")
    print("Human outcomes read: NO")
    print("=" * 120)

    if not policy_replay_pass:
        raise RuntimeError(
            "Full A1/A2 event-count ladder failed exact replay of 05d0c."
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 120, file=sys.stderr)
        print("05d0f utility/comparator audit: FAIL", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print("=" * 120, file=sys.stderr)
        raise
