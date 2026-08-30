#!/usr/bin/env python3
"""
Paper 6 - post-05d0c mechanistic-vs-operational crosswalk and safety-utility frontier.

Purpose
-------
05d0c prospectively decided STOP_A6_FOR_PAPER6 for the simple global-trust
signal. This stage does NOT reopen A6 and does NOT create a new method.

It quantifies two post-result explanatory questions using ONLY already-written
05d/05d0a/05d0c artifacts:

1. Why can source-target TRAINING compatibility discriminate the simulation's
   mechanistic regime (R0-R3 vs R4-R5) while failing to predict actual
   independent-test transfer benefit/harm?

2. What utility is attainable by the already-evaluated frozen/trivial
   abstention policies at the already-frozen safety target NT <= 0.10?

No model fitting.
No threshold tuning.
No new compatibility statistic.
No human outcomes.
No GPU.

Important interpretation
------------------------
"Mechanistic compatible" refers only to the simulation generator label R0-R3.
It does NOT imply that A1/A2 must improve prediction in a finite target sample.

"Operational beneficial/harmful" is defined exactly as in 05d0c:
    beneficial deltaC(engine-B0) >= +0.02
    harmful    deltaC(engine-B0) <= -0.02
    neutral    otherwise

This script is descriptive/explanatory. It cannot change:
    05d HOLD
    A5 CLOSE
    05d0c STOP_A6_FOR_PAPER6
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd


SCRIPT_VERSION = "05d0d-mechanistic-operational-crosswalk-safety-utility-v1-no-cli"

ROOT = Path(__file__).resolve().parents[1]

D_DIR = ROOT / "results" / "simulation_phase_diagram" / "05d"
D_SUMMARY = D_DIR / "summary.json"

D0A_DIR = ROOT / "results" / "simulation_phase_diagram_diagnostics" / "05d0a"
D0A_ORACLE = D0A_DIR / "threshold_feasibility_oracle_summary.tsv"
D0A_COMPARATORS = D0A_DIR / "frozen_and_diagnostic_comparator_summary.tsv"
D0A_SUMMARY = D0A_DIR / "summary.json"

D0C_DIR = ROOT / "results" / "simulation_phase_diagram_diagnostics" / "05d0c"
D0C_REPLICATES = D0C_DIR / "global_trust_replicate_statistics.tsv"
D0C_AUROC = D0C_DIR / "global_trust_AUROC_by_event_budget.tsv"
D0C_POLICY = D0C_DIR / "event_count_only_abstention_policies.tsv"
D0C_GATE_MASS = D0C_DIR / "R5_continuous_harmful_gate_mass_stratification.tsv"
D0C_SUMMARY = D0C_DIR / "summary.json"

OUT_DIR = ROOT / "results" / "simulation_phase_diagram_diagnostics" / "05d0d"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CROSSWALK = OUT_DIR / "mechanistic_operational_crosswalk.tsv"
COMPAT_QUARTILES = OUT_DIR / "compatibility_quartile_operational_outcomes.tsv"
COMPAT_ASSOCIATION = OUT_DIR / "compatibility_vs_independent_deltaC.tsv"
SAFETY_UTILITY = OUT_DIR / "safety_utility_policy_table.tsv"
PARETO_FRONTIER = OUT_DIR / "safety_utility_pareto_frontier.tsv"
SUMMARY_JSON = OUT_DIR / "summary.json"

EXPECTED_05D_STATUS = "HOLD_NO_SAFE_SELECTABLE_ARCHITECTURE"
EXPECTED_05D0C_STATUS = "STOP_A6_FOR_PAPER6"

ENGINES = ["A1", "A2"]
EVENT_BUDGETS = [5, 10, 15, 20, 29, 40]

BENEFIT_DELTA = +0.02
HARM_DELTA = -0.02
FROZEN_NT_LIMIT = 0.10
FROZEN_PRACTICAL_GAIN = 0.02


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
    return float(np.corrcoef(rx, ry)[0, 1])


def operational_class(delta: float) -> str:
    if not np.isfinite(delta):
        return "UNDEFINED"
    if delta >= BENEFIT_DELTA:
        return "BENEFICIAL"
    if delta <= HARM_DELTA:
        return "HARMFUL"
    return "NEUTRAL"


def mechanistic_class(label: int) -> str:
    return "COMPATIBLE_R0_R3" if int(label) == 1 else "INCOMPATIBLE_R4_R5"


def pareto_nondominated(frame: pd.DataFrame) -> pd.Series:
    """Maximize mean_delta, minimize NT."""
    gain = frame["mean_delta_c_vs_B0"].to_numpy(dtype=float)
    nt = frame["negative_transfer_rate"].to_numpy(dtype=float)

    out = np.ones(len(frame), dtype=bool)

    for i in range(len(frame)):
        for j in range(len(frame)):
            if i == j:
                continue
            weak_better = (
                gain[j] >= gain[i] - 1e-12
                and nt[j] <= nt[i] + 1e-12
            )
            strict = (
                gain[j] > gain[i] + 1e-12
                or nt[j] < nt[i] - 1e-12
            )
            if weak_better and strict:
                out[i] = False
                break

    return pd.Series(out, index=frame.index)


def main() -> None:
    started = datetime.now(timezone.utc).isoformat()

    print("=" * 120)
    print("Paper 6 - mechanistic/operational crosswalk + safety-utility frontier")
    print("=" * 120)
    print(f"Script version: {SCRIPT_VERSION}")
    print()
    print("Scope:")
    print("  New model fitting: NO")
    print("  New trust signal: NO")
    print("  Threshold changes: NO")
    print("  Frozen 05d HOLD changed: NO")
    print("  05d0c A6 STOP changed: NO")
    print("  Human outcomes: NO")
    print("  GPU: NO")
    print()

    for p in [
        D_SUMMARY,
        D0A_ORACLE,
        D0A_COMPARATORS,
        D0A_SUMMARY,
        D0C_REPLICATES,
        D0C_AUROC,
        D0C_POLICY,
        D0C_GATE_MASS,
        D0C_SUMMARY,
    ]:
        require_file(p)

    d_summary = read_json(D_SUMMARY)
    d0c_summary = read_json(D0C_SUMMARY)

    if str(d_summary.get("scientific_status")) != EXPECTED_05D_STATUS:
        raise RuntimeError("05d is not in expected frozen HOLD state.")
    if str(d0c_summary.get("scientific_status")) != EXPECTED_05D0C_STATUS:
        raise RuntimeError("05d0c did not prospectively STOP A6 for Paper 6.")

    reps = pd.read_csv(D0C_REPLICATES, sep="\t")
    policies = pd.read_csv(D0C_POLICY, sep="\t")
    oracle = pd.read_csv(D0A_ORACLE, sep="\t")
    comparators = pd.read_csv(D0A_COMPARATORS, sep="\t")

    if len(reps) != 21600:
        raise RuntimeError(
            f"05d0c replicate rows={len(reps)}, expected=21600."
        )

    crosswalk_rows: List[Dict[str, Any]] = []
    association_rows: List[Dict[str, Any]] = []
    quartile_rows: List[Dict[str, Any]] = []

    for events in EVENT_BUDGETS:
        event_part = reps[reps["target_events"] == events].copy()

        for engine in ENGINES:
            delta_col = f"delta_c_{engine}"
            temp = event_part[
                [
                    "scenario_id",
                    "replicate",
                    "compatibility_source_train_c",
                    "mechanistic_label",
                    delta_col,
                ]
            ].copy()
            temp["operational_class"] = [
                operational_class(x) for x in temp[delta_col]
            ]
            temp["mechanistic_class"] = [
                mechanistic_class(x) for x in temp["mechanistic_label"]
            ]

            for mech_label in ["COMPATIBLE_R0_R3", "INCOMPATIBLE_R4_R5"]:
                m = temp[temp["mechanistic_class"] == mech_label]
                denom = len(m)
                for op_label in ["BENEFICIAL", "NEUTRAL", "HARMFUL"]:
                    g = m[m["operational_class"] == op_label]
                    crosswalk_rows.append(
                        {
                            "target_events": events,
                            "engine": engine,
                            "mechanistic_class": mech_label,
                            "operational_class": op_label,
                            "n_replicates": len(g),
                            "fraction_within_mechanistic_class": (
                                len(g) / denom if denom else float("nan")
                            ),
                            "mean_compatibility_source_train_c": finite_mean(
                                g["compatibility_source_train_c"]
                            ),
                            "mean_independent_test_delta_c": finite_mean(
                                g[delta_col]
                            ),
                        }
                    )

            association_rows.append(
                {
                    "target_events": events,
                    "engine": engine,
                    "n_replicates": len(temp),
                    "spearman_compatibility_vs_test_delta_c": spearman(
                        temp["compatibility_source_train_c"].to_numpy(dtype=float),
                        temp[delta_col].to_numpy(dtype=float),
                    ),
                    "mean_test_delta_c": finite_mean(temp[delta_col]),
                    "fraction_beneficial": float(
                        np.mean(temp["operational_class"] == "BENEFICIAL")
                    ),
                    "fraction_harmful": float(
                        np.mean(temp["operational_class"] == "HARMFUL")
                    ),
                    "fraction_neutral": float(
                        np.mean(temp["operational_class"] == "NEUTRAL")
                    ),
                }
            )

            finite = np.isfinite(
                temp["compatibility_source_train_c"].to_numpy(dtype=float)
            )
            tf = temp.loc[finite].copy()
            if len(tf) >= 4:
                ranks = tf["compatibility_source_train_c"].rank(method="first")
                tf["compatibility_quartile"] = pd.qcut(
                    ranks,
                    q=4,
                    labels=["Q1_LOW", "Q2", "Q3", "Q4_HIGH"],
                )
                for qlabel, g in tf.groupby(
                    "compatibility_quartile", observed=True
                ):
                    quartile_rows.append(
                        {
                            "target_events": events,
                            "engine": engine,
                            "compatibility_quartile": str(qlabel),
                            "n_replicates": len(g),
                            "mean_compatibility_source_train_c": finite_mean(
                                g["compatibility_source_train_c"]
                            ),
                            "mean_independent_test_delta_c": finite_mean(
                                g[delta_col]
                            ),
                            "fraction_beneficial": float(
                                np.mean(g["operational_class"] == "BENEFICIAL")
                            ),
                            "fraction_harmful": float(
                                np.mean(g["operational_class"] == "HARMFUL")
                            ),
                        }
                    )

    crosswalk_df = pd.DataFrame(crosswalk_rows)
    association_df = pd.DataFrame(association_rows)
    quartile_df = pd.DataFrame(quartile_rows)

    crosswalk_df.to_csv(CROSSWALK, sep="\t", index=False)
    association_df.to_csv(COMPAT_ASSOCIATION, sep="\t", index=False)
    quartile_df.to_csv(COMPAT_QUARTILES, sep="\t", index=False)

    policy_rows: List[Dict[str, Any]] = []

    b0_row = comparators[comparators["model"] == "B0"]
    if len(b0_row) != 1:
        raise RuntimeError("B0 comparator not unique.")
    b0_c = float(b0_row.iloc[0]["equal_scenario_mean_uno_c"])

    policy_rows.append(
        {
            "policy": "B0_TARGET_ONLY",
            "policy_type": "FROZEN_BASELINE",
            "engine": "B0",
            "activation_fraction": 0.0,
            "mean_uno_c": b0_c,
            "mean_delta_c_vs_B0": 0.0,
            "negative_transfer_rate": 0.0,
            "meets_frozen_NT_0_10": True,
            "meets_practical_gain_0_02": False,
        }
    )

    for engine in ENGINES:
        row = comparators[comparators["model"] == engine]
        if len(row) != 1:
            raise RuntimeError(f"Comparator {engine} not unique.")
        r = row.iloc[0]
        gain = float(r["equal_scenario_mean_uno_c"] - b0_c)
        nt = float(r["aggregate_negative_transfer_rate"])
        policy_rows.append(
            {
                "policy": f"{engine}_ALWAYS_TRANSFER",
                "policy_type": "FROZEN_MODEL",
                "engine": engine,
                "activation_fraction": 1.0,
                "mean_uno_c": float(r["equal_scenario_mean_uno_c"]),
                "mean_delta_c_vs_B0": gain,
                "negative_transfer_rate": nt,
                "meets_frozen_NT_0_10": bool(nt <= FROZEN_NT_LIMIT),
                "meets_practical_gain_0_02": bool(
                    gain >= FROZEN_PRACTICAL_GAIN
                ),
            }
        )

    for r in policies.itertuples(index=False):
        policy_rows.append(
            {
                "policy": (
                    f"{r.engine}_TRANSFER_IF_EVENTS_GE_"
                    f"{int(r.minimum_events_to_transfer)}"
                ),
                "policy_type": "TRIVIAL_EVENT_COUNT_ABSTENTION",
                "engine": str(r.engine),
                "activation_fraction": float(r.scenario_activation_fraction),
                "mean_uno_c": float(r.equal_scenario_mean_uno_c),
                "mean_delta_c_vs_B0": float(
                    r.equal_scenario_mean_delta_c_vs_B0
                ),
                "negative_transfer_rate": float(
                    r.aggregate_negative_transfer_rate
                ),
                "meets_frozen_NT_0_10": bool(r.meets_frozen_NT_0_10),
                "meets_practical_gain_0_02": bool(
                    float(r.equal_scenario_mean_delta_c_vs_B0)
                    >= FROZEN_PRACTICAL_GAIN
                ),
            }
        )

    for model_name in ["REGIME_ABSTENTION_A2", "REGIME_ABSTENTION_A3"]:
        row = oracle[oracle["diagnostic_model"] == model_name]
        if len(row) != 1:
            raise RuntimeError(f"Oracle summary {model_name} not unique.")
        r = row.iloc[0]
        gain = float(r["equal_scenario_mean_uno_c"] - b0_c)
        nt = float(r["aggregate_negative_transfer_rate"])
        policy_rows.append(
            {
                "policy": model_name,
                "policy_type": "DIAGNOSTIC_ORACLE_UPPER_BOUND",
                "engine": model_name.split("_")[-1],
                "activation_fraction": float("nan"),
                "mean_uno_c": float(r["equal_scenario_mean_uno_c"]),
                "mean_delta_c_vs_B0": gain,
                "negative_transfer_rate": nt,
                "meets_frozen_NT_0_10": bool(nt <= FROZEN_NT_LIMIT),
                "meets_practical_gain_0_02": bool(
                    gain >= FROZEN_PRACTICAL_GAIN
                ),
            }
        )

    safety_df = pd.DataFrame(policy_rows)

    implementable = safety_df[
        safety_df["policy_type"] != "DIAGNOSTIC_ORACLE_UPPER_BOUND"
    ].copy()
    implementable["pareto_nondominated_gain_vs_NT"] = pareto_nondominated(
        implementable
    )
    safety_df = safety_df.merge(
        implementable[["policy", "pareto_nondominated_gain_vs_NT"]],
        on="policy",
        how="left",
    )
    safety_df.to_csv(SAFETY_UTILITY, sep="\t", index=False)

    frontier = implementable[
        implementable["pareto_nondominated_gain_vs_NT"]
    ].sort_values(["negative_transfer_rate", "mean_delta_c_vs_B0"])
    frontier.to_csv(PARETO_FRONTIER, sep="\t", index=False)

    safe = implementable[
        implementable["negative_transfer_rate"] <= FROZEN_NT_LIMIT
    ].copy()
    if len(safe):
        best_safe = safe.loc[safe["mean_delta_c_vs_B0"].idxmax()]
        best_safe_policy = str(best_safe["policy"])
        best_safe_gain = float(best_safe["mean_delta_c_vs_B0"])
        best_safe_nt = float(best_safe["negative_transfer_rate"])
        best_safe_activation = float(best_safe["activation_fraction"])
    else:
        best_safe_policy = None
        best_safe_gain = float("nan")
        best_safe_nt = float("nan")
        best_safe_activation = float("nan")

    assoc29 = association_df[
        association_df["target_events"] == 29
    ].set_index("engine")

    cross29 = crosswalk_df[
        (crosswalk_df["target_events"] == 29)
        & (crosswalk_df["mechanistic_class"] == "COMPATIBLE_R0_R3")
    ]

    compatible_harm_fraction: Dict[str, float] = {}
    for engine in ENGINES:
        row = cross29[
            (cross29["engine"] == engine)
            & (cross29["operational_class"] == "HARMFUL")
        ]
        compatible_harm_fraction[engine] = (
            float(row.iloc[0]["fraction_within_mechanistic_class"])
            if len(row) == 1
            else float("nan")
        )

    summary = {
        "script_version": SCRIPT_VERSION,
        "status": "PASS",
        "scientific_status": (
            "PASS_POSTHOC_MECHANISTIC_OPERATIONAL_CROSSWALK_COMPLETE"
        ),
        "run_started_utc": started,
        "run_finished_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_05d_status_unchanged": EXPECTED_05D_STATUS,
        "A6_status_unchanged": EXPECTED_05D0C_STATUS,
        "primary_event_budget": 29,
        "spearman_compatibility_vs_test_deltaC_at_29": {
            engine: float(
                assoc29.loc[
                    engine,
                    "spearman_compatibility_vs_test_delta_c",
                ]
            )
            for engine in ENGINES
        },
        "fraction_harmful_within_mechanistically_compatible_R0_R3_at_29": (
            compatible_harm_fraction
        ),
        "best_implementable_policy_under_frozen_NT_0_10": {
            "policy": best_safe_policy,
            "mean_delta_c_vs_B0": best_safe_gain,
            "negative_transfer_rate": best_safe_nt,
            "activation_fraction": best_safe_activation,
            "meets_practical_gain_0_02": bool(
                np.isfinite(best_safe_gain)
                and best_safe_gain >= FROZEN_PRACTICAL_GAIN
            ),
        },
        "interpretation_guardrails": {
            "mechanistic_compatibility_is_not_operational_utility": True,
            "no_causal_claim": True,
            "no_new_model_selected": True,
            "no_A6_reopening": True,
            "oracle_points_are_not_implementable_policies": True,
        },
        "final_artifact_hashes": {
            "mechanistic_operational_crosswalk.tsv": sha256_file(CROSSWALK),
            "compatibility_quartile_operational_outcomes.tsv": sha256_file(
                COMPAT_QUARTILES
            ),
            "compatibility_vs_independent_deltaC.tsv": sha256_file(
                COMPAT_ASSOCIATION
            ),
            "safety_utility_policy_table.tsv": sha256_file(SAFETY_UTILITY),
            "safety_utility_pareto_frontier.tsv": sha256_file(
                PARETO_FRONTIER
            ),
        },
    }
    write_json(SUMMARY_JSON, summary)

    print("-" * 120)
    print("Mechanistic compatibility -> operational outcome at 29 events")
    print("-" * 120)
    display = crosswalk_df[
        crosswalk_df["target_events"] == 29
    ][
        [
            "engine",
            "mechanistic_class",
            "operational_class",
            "n_replicates",
            "fraction_within_mechanistic_class",
            "mean_compatibility_source_train_c",
            "mean_independent_test_delta_c",
        ]
    ]
    print(display.to_string(index=False))

    print()
    print("-" * 120)
    print("Compatibility statistic vs independent transfer gain")
    print("-" * 120)
    print(association_df.to_string(index=False))

    print()
    print("-" * 120)
    print("Implementable safety-utility frontier")
    print("-" * 120)
    print(
        frontier[
            [
                "policy",
                "engine",
                "activation_fraction",
                "mean_uno_c",
                "mean_delta_c_vs_B0",
                "negative_transfer_rate",
                "meets_frozen_NT_0_10",
                "meets_practical_gain_0_02",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 120)
    print("05d0d SUMMARY")
    print("=" * 120)
    print(
        "Spearman compatibility vs independent ΔC @29: "
        + ", ".join(
            f"{engine}={summary['spearman_compatibility_vs_test_deltaC_at_29'][engine]:.4f}"
            for engine in ENGINES
        )
    )
    print(
        "Harmful fraction inside mechanistically compatible R0-R3 @29: "
        + ", ".join(
            f"{engine}={compatible_harm_fraction[engine]:.4f}"
            for engine in ENGINES
        )
    )
    print(
        f"Best implementable policy with frozen NT<=0.10: "
        f"{best_safe_policy}"
    )
    print(
        f"  mean ΔC={best_safe_gain:.4f}, "
        f"NT={best_safe_nt:.4f}, "
        f"activation={best_safe_activation:.4f}"
    )
    print(
        f"  reaches frozen practical +0.02 gain: "
        f"{bool(np.isfinite(best_safe_gain) and best_safe_gain >= FROZEN_PRACTICAL_GAIN)}"
    )
    print()
    print("05d HOLD unchanged: YES")
    print("A6 remains STOP for Paper 6: YES")
    print("Human outcomes read: NO")
    print("=" * 120)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 120, file=sys.stderr)
        print(
            "05d0d mechanistic/operational crosswalk diagnostic: FAIL",
            file=sys.stderr,
        )
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print("=" * 120, file=sys.stderr)
        raise
