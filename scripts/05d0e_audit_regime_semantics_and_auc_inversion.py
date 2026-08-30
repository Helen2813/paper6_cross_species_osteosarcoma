#!/usr/bin/env python3
"""
Paper 6 - post-05d0d audit of regime semantics and operational-AUROC inversion.

Why this exists
---------------
05d0c prospectively STOPPED the proposed A6 branch because the prespecified
compatibility score, with its prespecified direction "higher compatibility =>
greater transfer benefit", had operational AUROC < 0.70 at 29 events.

05d0d then exposed a striking descriptive inversion:
- the same score almost perfectly distinguishes generator regimes R0-R3 vs R4-R5;
- yet at 29 events, A1/A2 benefit is often MORE frequent in R4-R5;
- operational AUROC is therefore < 0.5, especially for A1.

This script asks whether that inversion is:
1. primarily BETWEEN-regime (generator labels do not map to architecture-level
   utility for A1/A2), or
2. also present WITHIN each regime (the training compatibility statistic itself
   is inversely related to independent-test gain).

It also reports 1-AUROC as a descriptive "reversed-direction information"
quantity. This is NOT a rescue criterion and CANNOT reopen A6, because the
direction of the 05d0c trust rule was frozen before results were read.

No model fitting.
No new trust signal.
No threshold changes.
No human outcomes.
No GPU.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd


SCRIPT_VERSION = "05d0e-regime-semantics-operational-auc-inversion-v1-no-cli"

ROOT = Path(__file__).resolve().parents[1]

A_DIR = ROOT / "results" / "simulation_contract" / "05a"
A_SCENARIOS = A_DIR / "simulation_scenario_registry.tsv"

D0C_DIR = ROOT / "results" / "simulation_phase_diagram_diagnostics" / "05d0c"
D0C_REPS = D0C_DIR / "global_trust_replicate_statistics.tsv"
D0C_AUC = D0C_DIR / "global_trust_AUROC_by_event_budget.tsv"
D0C_SUMMARY = D0C_DIR / "summary.json"

D0D_DIR = ROOT / "results" / "simulation_phase_diagram_diagnostics" / "05d0d"
D0D_SUMMARY = D0D_DIR / "summary.json"

OUT_DIR = ROOT / "results" / "simulation_phase_diagram_diagnostics" / "05d0e"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REGIME_TABLE = OUT_DIR / "exact_regime_operational_outcomes.tsv"
WITHIN_REGIME_ASSOC = OUT_DIR / "within_regime_compatibility_vs_deltaC.tsv"
RESIDUALIZED_ASSOC = OUT_DIR / "regime_residualized_compatibility_vs_deltaC.tsv"
AUC_DIRECTION = OUT_DIR / "operational_auc_direction_audit.tsv"
MATCHED_REGIME = OUT_DIR / "matched_regime_contrasts.tsv"
SUMMARY_JSON = OUT_DIR / "summary.json"

EXPECTED_D0C_STATUS = "STOP_A6_FOR_PAPER6"
EXPECTED_D0D_STATUS = "PASS_POSTHOC_MECHANISTIC_OPERATIONAL_CROSSWALK_COMPLETE"

ENGINES = ["A1", "A2"]
EVENT_BUDGETS = [5, 10, 15, 20, 29, 40]
PRIMARY_BUDGET = 29

BENEFIT_DELTA = +0.02
HARM_DELTA = -0.02

MATCH_AXES_CANDIDATES = [
    "target_events",
    "covariance_shift",
    "mapping_error",
    "censoring",
    "source_prior_state",
]


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


def finite_mean(x: Iterable[float]) -> float:
    a = np.asarray(list(x), dtype=float)
    a = a[np.isfinite(a)]
    return float(np.mean(a)) if len(a) else float("nan")


def finite_median(x: Iterable[float]) -> float:
    a = np.asarray(list(x), dtype=float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if len(a) else float("nan")


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


def operational_class(delta: np.ndarray) -> np.ndarray:
    d = np.asarray(delta, dtype=float)
    out = np.full(len(d), "NEUTRAL", dtype=object)
    out[d >= BENEFIT_DELTA] = "BENEFICIAL"
    out[d <= HARM_DELTA] = "HARMFUL"
    out[~np.isfinite(d)] = "UNDEFINED"
    return out


def main() -> None:
    started = datetime.now(timezone.utc).isoformat()

    print("=" * 120)
    print("Paper 6 - 05d0e regime semantics / operational-AUROC inversion audit")
    print("=" * 120)
    print(f"Script version: {SCRIPT_VERSION}")
    print()
    print("Scope:")
    print("  New model fitting: NO")
    print("  New trust signal: NO")
    print("  Reversed AUROC used for selection: NO")
    print("  05d0c STOP_A6 changed: NO")
    print("  Human outcomes: NO")
    print("  GPU: NO")
    print()

    for p in [A_SCENARIOS, D0C_REPS, D0C_AUC, D0C_SUMMARY, D0D_SUMMARY]:
        require_file(p)

    d0c_summary = read_json(D0C_SUMMARY)
    d0d_summary = read_json(D0D_SUMMARY)

    if str(d0c_summary.get("scientific_status")) != EXPECTED_D0C_STATUS:
        raise RuntimeError("05d0c is not in frozen STOP_A6_FOR_PAPER6 state.")
    if str(d0d_summary.get("scientific_status")) != EXPECTED_D0D_STATUS:
        raise RuntimeError("05d0d crosswalk diagnostic is not complete.")

    scenarios = pd.read_csv(A_SCENARIOS, sep="\t")
    reps = pd.read_csv(D0C_REPS, sep="\t")
    auc = pd.read_csv(D0C_AUC, sep="\t")

    if len(reps) != 21600:
        raise RuntimeError(f"Expected 21,600 replicate rows, observed {len(reps)}.")

    # Add any scenario metadata not already carried in 05d0c.
    meta_cols = ["scenario_id"] + [
        c for c in MATCH_AXES_CANDIDATES
        if c in scenarios.columns and c not in reps.columns
    ]
    if len(meta_cols) > 1:
        reps = reps.merge(
            scenarios[meta_cols].drop_duplicates("scenario_id"),
            on="scenario_id",
            how="left",
            validate="many_to_one",
        )

    regimes = list(
        scenarios["transfer_regime"].astype(str).drop_duplicates()
    )

    print("Frozen transfer regimes:")
    for regime in regimes:
        print(f"  {regime}")
    print()

    # ------------------------------------------------------------------
    # Exact regime outcome table.
    # ------------------------------------------------------------------
    regime_rows: List[Dict[str, Any]] = []
    within_rows: List[Dict[str, Any]] = []
    residual_rows: List[Dict[str, Any]] = []

    for events in EVENT_BUDGETS:
        ep = reps[reps["target_events"] == events].copy()

        for engine in ENGINES:
            delta_col = f"delta_c_{engine}"
            c_col = f"uno_c_{engine}"
            classes = operational_class(
                ep[delta_col].to_numpy(dtype=float)
            )
            ep_engine = ep.copy()
            ep_engine["_op"] = classes

            for regime in regimes:
                g = ep_engine[
                    ep_engine["transfer_regime"].astype(str) == regime
                ].copy()

                if len(g) == 0:
                    continue

                regime_rows.append(
                    {
                        "target_events": events,
                        "engine": engine,
                        "transfer_regime": regime,
                        "n_replicates": len(g),
                        "mean_source_train_compatibility": finite_mean(
                            g["compatibility_source_train_c"]
                        ),
                        "mean_engine_uno_c": finite_mean(g[c_col]),
                        "mean_B0_uno_c": finite_mean(g["uno_c_B0"]),
                        "mean_delta_c_vs_B0": finite_mean(g[delta_col]),
                        "median_delta_c_vs_B0": finite_median(g[delta_col]),
                        "fraction_beneficial": float(
                            np.mean(g["_op"] == "BENEFICIAL")
                        ),
                        "fraction_neutral": float(
                            np.mean(g["_op"] == "NEUTRAL")
                        ),
                        "fraction_harmful": float(
                            np.mean(g["_op"] == "HARMFUL")
                        ),
                    }
                )

                within_rows.append(
                    {
                        "target_events": events,
                        "engine": engine,
                        "transfer_regime": regime,
                        "n_replicates": len(g),
                        "spearman_compatibility_vs_delta_c": spearman(
                            g["compatibility_source_train_c"].to_numpy(dtype=float),
                            g[delta_col].to_numpy(dtype=float),
                        ),
                    }
                )

            # Regime-residualized association:
            # remove each regime's mean compatibility and mean delta, then correlate.
            residualized = ep_engine[
                [
                    "transfer_regime",
                    "compatibility_source_train_c",
                    delta_col,
                ]
            ].copy()

            residualized["_compat_resid"] = (
                residualized["compatibility_source_train_c"]
                - residualized.groupby("transfer_regime")[
                    "compatibility_source_train_c"
                ].transform("mean")
            )
            residualized["_delta_resid"] = (
                residualized[delta_col]
                - residualized.groupby("transfer_regime")[
                    delta_col
                ].transform("mean")
            )

            residual_rows.append(
                {
                    "target_events": events,
                    "engine": engine,
                    "n_replicates": len(residualized),
                    "raw_spearman_compatibility_vs_delta_c": spearman(
                        residualized["compatibility_source_train_c"].to_numpy(dtype=float),
                        residualized[delta_col].to_numpy(dtype=float),
                    ),
                    "regime_residualized_spearman": spearman(
                        residualized["_compat_resid"].to_numpy(dtype=float),
                        residualized["_delta_resid"].to_numpy(dtype=float),
                    ),
                }
            )

    regime_df = pd.DataFrame(regime_rows)
    within_df = pd.DataFrame(within_rows)
    residual_df = pd.DataFrame(residual_rows)

    regime_df.to_csv(REGIME_TABLE, sep="\t", index=False)
    within_df.to_csv(WITHIN_REGIME_ASSOC, sep="\t", index=False)
    residual_df.to_csv(RESIDUALIZED_ASSOC, sep="\t", index=False)

    # ------------------------------------------------------------------
    # AUC direction audit. This is descriptive only.
    # ------------------------------------------------------------------
    op_auc = auc[
        auc["endpoint"] == "OPERATIONAL_BENEFIT_vs_HARM"
    ].copy()

    op_auc["reversed_direction_AUROC"] = (
        1.0 - op_auc["source_train_compatibility_AUROC"]
    )
    op_auc["descriptive_direction"] = np.where(
        op_auc["source_train_compatibility_AUROC"] < 0.5,
        "LOWER_COMPATIBILITY_ASSOCIATED_WITH_BENEFIT",
        "HIGHER_COMPATIBILITY_ASSOCIATED_WITH_BENEFIT",
    )
    op_auc["eligible_to_reopen_05d0c_A6"] = False
    op_auc["reason_not_reopened"] = (
        "05d0c signal direction and GO/STOP rule were frozen before results; "
        "1-AUROC is post-result descriptive only."
    )

    op_auc.to_csv(AUC_DIRECTION, sep="\t", index=False)

    # ------------------------------------------------------------------
    # Matched regime contrasts across the scenario registry.
    # This asks whether exact-regime ordering persists after matching available
    # nuisance axes in the frozen scenario grid.
    # ------------------------------------------------------------------
    match_axes = [
        c for c in MATCH_AXES_CANDIDATES
        if c in scenarios.columns
    ]

    scenario_level = (
        reps.groupby(
            ["scenario_id", "transfer_regime"] + [
                c for c in match_axes if c not in {"target_events"}
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            target_events=("target_events", "first"),
            mean_compatibility=("compatibility_source_train_c", "mean"),
            mean_delta_A1=("delta_c_A1", "mean"),
            mean_delta_A2=("delta_c_A2", "mean"),
        )
    )

    # Build a single match key from scientific nuisance axes, excluding regime.
    key_cols = [
        c for c in match_axes
        if c in scenario_level.columns
    ]
    if "target_events" not in key_cols:
        key_cols = ["target_events"] + key_cols

    matched_rows: List[Dict[str, Any]] = []

    for regime_a, regime_b in combinations(regimes, 2):
        a = scenario_level[
            scenario_level["transfer_regime"].astype(str) == regime_a
        ].copy()
        b = scenario_level[
            scenario_level["transfer_regime"].astype(str) == regime_b
        ].copy()

        if len(a) == 0 or len(b) == 0:
            continue

        merged = a.merge(
            b,
            on=key_cols,
            suffixes=("_A", "_B"),
            how="inner",
        )

        if len(merged) == 0:
            continue

        for engine in ENGINES:
            da = merged[f"mean_delta_{engine}_A"].to_numpy(dtype=float)
            db = merged[f"mean_delta_{engine}_B"].to_numpy(dtype=float)
            diff = db - da

            matched_rows.append(
                {
                    "engine": engine,
                    "regime_A": regime_a,
                    "regime_B": regime_b,
                    "contrast": "B_minus_A",
                    "match_axes": "|".join(key_cols),
                    "n_matched_scenario_pairs": len(merged),
                    "mean_delta_regime_A": finite_mean(da),
                    "mean_delta_regime_B": finite_mean(db),
                    "mean_B_minus_A": finite_mean(diff),
                    "median_B_minus_A": finite_median(diff),
                    "fraction_pairs_B_gt_A": float(np.mean(diff > 0)),
                }
            )

    matched_df = pd.DataFrame(matched_rows)
    matched_df.to_csv(MATCHED_REGIME, sep="\t", index=False)

    # ------------------------------------------------------------------
    # Primary-budget summary.
    # ------------------------------------------------------------------
    regime29 = regime_df[regime_df["target_events"] == PRIMARY_BUDGET].copy()
    within29 = within_df[within_df["target_events"] == PRIMARY_BUDGET].copy()
    resid29 = residual_df[
        residual_df["target_events"] == PRIMARY_BUDGET
    ].set_index("engine")
    auc29 = op_auc[
        op_auc["target_events"] == PRIMARY_BUDGET
    ].set_index("engine")

    summary = {
        "script_version": SCRIPT_VERSION,
        "status": "PASS",
        "scientific_status": (
            "PASS_POSTHOC_REGIME_SEMANTICS_AUDIT_COMPLETE_A6_REMAINS_STOP"
        ),
        "run_started_utc": started,
        "run_finished_utc": datetime.now(timezone.utc).isoformat(),
        "A6_status_unchanged": EXPECTED_D0C_STATUS,
        "primary_event_budget": PRIMARY_BUDGET,
        "operational_AUROC_at_29": {
            engine: float(
                auc29.loc[
                    engine,
                    "source_train_compatibility_AUROC",
                ]
            )
            for engine in ENGINES
        },
        "reversed_direction_AUROC_at_29_descriptive_only": {
            engine: float(
                auc29.loc[
                    engine,
                    "reversed_direction_AUROC",
                ]
            )
            for engine in ENGINES
        },
        "raw_spearman_at_29": {
            engine: float(
                resid29.loc[
                    engine,
                    "raw_spearman_compatibility_vs_delta_c",
                ]
            )
            for engine in ENGINES
        },
        "regime_residualized_spearman_at_29": {
            engine: float(
                resid29.loc[
                    engine,
                    "regime_residualized_spearman",
                ]
            )
            for engine in ENGINES
        },
        "interpretation_guardrails": {
            "reversed_AUROC_cannot_reopen_A6": True,
            "generator_regime_is_not_architecture_level_utility_label": True,
            "posthoc_only": True,
            "no_new_method_selected": True,
            "human_outcomes_read": False,
        },
        "final_artifact_hashes": {
            "exact_regime_operational_outcomes.tsv": sha256_file(REGIME_TABLE),
            "within_regime_compatibility_vs_deltaC.tsv": sha256_file(
                WITHIN_REGIME_ASSOC
            ),
            "regime_residualized_compatibility_vs_deltaC.tsv": sha256_file(
                RESIDUALIZED_ASSOC
            ),
            "operational_auc_direction_audit.tsv": sha256_file(AUC_DIRECTION),
            "matched_regime_contrasts.tsv": sha256_file(MATCHED_REGIME),
        },
    }
    write_json(SUMMARY_JSON, summary)

    print("=" * 120)
    print("05d0e EXACT REGIME OUTCOMES @ 29 EVENTS")
    print("=" * 120)
    print(
        regime29[
            [
                "engine",
                "transfer_regime",
                "n_replicates",
                "mean_source_train_compatibility",
                "mean_delta_c_vs_B0",
                "fraction_beneficial",
                "fraction_neutral",
                "fraction_harmful",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 120)
    print("05d0e WITHIN-REGIME ASSOCIATIONS @ 29 EVENTS")
    print("=" * 120)
    print(
        within29[
            [
                "engine",
                "transfer_regime",
                "n_replicates",
                "spearman_compatibility_vs_delta_c",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 120)
    print("05d0e RAW VS REGIME-RESIDUALIZED ASSOCIATION @ 29 EVENTS")
    print("=" * 120)
    print(
        residual_df[
            residual_df["target_events"] == PRIMARY_BUDGET
        ].to_string(index=False)
    )

    print()
    print("=" * 120)
    print("05d0e OPERATIONAL AUC DIRECTION AUDIT")
    print("=" * 120)
    print(
        op_auc[
            [
                "target_events",
                "engine",
                "source_train_compatibility_AUROC",
                "reversed_direction_AUROC",
                "descriptive_direction",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 120)
    print("05d0e MATCHED REGIME CONTRASTS")
    print("=" * 120)
    if len(matched_df):
        print(matched_df.to_string(index=False))
    else:
        print("No exact matched regime pairs available under frozen nuisance-axis matching.")

    print()
    print("=" * 120)
    print("05d0e SUMMARY")
    print("=" * 120)
    print(
        "Operational AUROC @29: "
        + ", ".join(
            f"{e}={summary['operational_AUROC_at_29'][e]:.4f}"
            for e in ENGINES
        )
    )
    print(
        "Descriptive reversed AUROC @29: "
        + ", ".join(
            f"{e}={summary['reversed_direction_AUROC_at_29_descriptive_only'][e]:.4f}"
            for e in ENGINES
        )
    )
    print(
        "Raw Spearman @29: "
        + ", ".join(
            f"{e}={summary['raw_spearman_at_29'][e]:.4f}"
            for e in ENGINES
        )
    )
    print(
        "Regime-residualized Spearman @29: "
        + ", ".join(
            f"{e}={summary['regime_residualized_spearman_at_29'][e]:.4f}"
            for e in ENGINES
        )
    )
    print()
    print("A6 remains STOP for Paper 6: YES")
    print("Reversed AUROC is descriptive only and cannot change 05d0c: YES")
    print("Human outcomes read: NO")
    print("=" * 120)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 120, file=sys.stderr)
        print("05d0e regime-semantics audit: FAIL", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print("=" * 120, file=sys.stderr)
        raise
