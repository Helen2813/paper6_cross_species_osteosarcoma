#!/usr/bin/env python
"""
Paper 6 - export frozen all-model safety summary for CBM strengthening.

Implementation correction after the read-only 05h2a schema probe.

Key correction:
  Bind explicitly to the authoritative frozen 05d scenario/model summary and
  use its existing scenario-level fields:
    - mean_uno_c
    - mean_delta_c_vs_B0
    - negative_transfer_rate
    - catastrophic_negative_transfer_rate

No model fitting, no survival-metric recomputation, no outcome access, and no
reconstruction of catastrophic-transfer rates from flags.

Expected placement:
    scripts/05h2b_export_all_model_safety_summary.py

Run:
    python scripts/05h2b_export_all_model_safety_summary.py
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_VERSION = "05h2b-export-all-model-safety-summary-v1-explicit-frozen-schema-no-cli"

MAIN_MODELS = ["B0", "B4", "A0", "A1", "A2", "A3", "A4"]
REFERENCE_MODEL = "B0"
ELIGIBLE_MODELS = {"A2", "A3"}

NEGATIVE_THRESHOLD = -0.02
CATASTROPHIC_THRESHOLD = -0.05

EXPECTED_SCENARIOS = 180
EXPECTED_MAIN_ROWS = EXPECTED_SCENARIOS * len(MAIN_MODELS)
EXPECTED_REGIME_COUNTS = {
    "R0": 6,
    "R1": 6,
    "R2": 78,
    "R3": 6,
    "R4": 6,
    "R5": 78,
}

# Independent accepted replay anchors. These are acceptance checks only; they
# are never used to create the exported values.
EXPECTED_EQUAL_SCENARIO_UNO_C = {
    "B0": 0.555554,
    "B4": 0.544989,
    "A0": 0.555351,
    "A1": 0.572198,
    "A2": 0.569809,
    "A3": 0.530067,
    "A4": 0.519107,
}
EXPECTED_NEGATIVE_TRANSFER_RATE = {
    "B0": 0.000000,
    "B4": 0.305528,
    "A0": 0.009000,
    "A1": 0.286778,
    "A2": 0.215417,
    "A3": 0.486333,
    "A4": 0.511722,
}
EXPECTED_R5_CATASTROPHIC_RATE = {
    "B0": 0.000000,
    "B4": 0.219936,
    "A0": 0.000000,
    "A1": 0.020513,
    "A2": 0.059551,
    "A3": 0.712179,
    "A4": 0.879744,
}
REPLAY_TOL = 7.5e-6
FROZEN_AGGREGATE_TOL = 5e-10

FROZEN_ROLES = {
    "B0": "frozen target-only reference",
    "B4": "frozen non-selectable comparator",
    "A0": "frozen non-selectable comparator",
    "A1": "frozen non-selectable comparator",
    "A2": "frozen selectable architecture",
    "A3": "frozen selectable architecture",
    "A4": "frozen non-selectable comparator",
}


def project_root() -> Path:
    p = Path(__file__).resolve()
    if p.parent.name.lower() != "scripts":
        raise RuntimeError(
            "Place 05h2b_export_all_model_safety_summary.py in the repository scripts/ directory."
        )
    return p.parent.parent


ROOT = project_root()

H0_JSON = (
    ROOT
    / "method_contract"
    / "05h0_cbm_strengthening_contract"
    / "cbm_strengthening_contract.json"
)

SCENARIO_SUMMARY = (
    ROOT
    / "results"
    / "simulation_phase_diagram"
    / "05d"
    / "scenario_model_metric_summary.tsv"
)

MODEL_AGGREGATE = (
    ROOT
    / "results"
    / "simulation_phase_diagram"
    / "05d"
    / "model_aggregate_summary.tsv"
)

METRIC_CONTRACT = (
    ROOT
    / "results"
    / "simulation_phase_diagram"
    / "05d"
    / "metric_implementation_contract.json"
)

FROZEN_SELECTION = (
    ROOT
    / "results"
    / "simulation_phase_diagram"
    / "05d"
    / "frozen_architecture_selection.json"
)

OUT_DIR = ROOT / "method_contract" / "05h2b_all_model_safety_summary"
WORK_DIR = ROOT / "method_contract" / ".05h2b_all_model_safety_summary_work"


REQUIRED_SCENARIO_COLUMNS = [
    "scenario_id",
    "transfer_regime",
    "replicates",
    "model",
    "mean_uno_c",
    "mean_delta_c_vs_B0",
    "negative_transfer_rate",
    "catastrophic_negative_transfer_rate",
]

REQUIRED_AGGREGATE_COLUMNS = [
    "model",
    "n_scenarios",
    "equal_scenario_mean_uno_c",
    "equal_scenario_mean_delta_c_vs_B0",
    "equal_scenario_negative_transfer_rate",
    "equal_scenario_catastrophic_rate",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.write_text(
        json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def finite_float(x: Any, label: str) -> float:
    v = float(x)
    if not math.isfinite(v):
        raise RuntimeError(f"Non-finite value for {label}: {x!r}")
    return v


def recursive_find_key(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            hit = recursive_find_key(v, key)
            if hit is not None:
                return hit
    elif isinstance(obj, list):
        for v in obj:
            hit = recursive_find_key(v, key)
            if hit is not None:
                return hit
    return None


def verify_h0_contract() -> dict[str, Any]:
    if not H0_JSON.exists():
        raise FileNotFoundError(f"Missing frozen 05h0 contract: {H0_JSON}")

    obj = json.loads(H0_JSON.read_text(encoding="utf-8"))
    block = recursive_find_key(obj, "all_model_safety_summary")
    if not isinstance(block, dict):
        raise RuntimeError("Frozen 05h0 contract lacks all_model_safety_summary.")

    if block.get("analysis_role") != "POST-HOLD DESCRIPTIVE CONTEXT":
        raise RuntimeError("Unexpected 05h0 all-model analysis_role.")
    if block.get("models") != MAIN_MODELS:
        raise RuntimeError(
            f"Frozen 05h0 model list changed: {block.get('models')!r}"
        )
    if block.get("reference_model") != REFERENCE_MODEL:
        raise RuntimeError("Frozen 05h0 reference model changed from B0.")

    expected_fields = {
        "model",
        "frozen_role",
        "eligible_for_primary_selection",
        "mean_delta_uno_c_vs_B0_if_defined",
        "negative_transfer_rate_at_deltaC_le_-0.02",
        "catastrophic_transfer_rate_at_deltaC_le_-0.05",
        "regime_specific_rates_or_effects_if_already_estimable",
    }
    actual = set(block.get("minimum_reported_fields", []))
    if not expected_fields.issubset(actual):
        raise RuntimeError(
            "Frozen 05h0 minimum all-model fields no longer match the 05h2b export contract."
        )
    return obj


def load_frozen_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    for p in [SCENARIO_SUMMARY, MODEL_AGGREGATE, METRIC_CONTRACT, FROZEN_SELECTION]:
        if not p.exists():
            raise FileNotFoundError(f"Required frozen 05d artifact missing: {p}")

    scenario = pd.read_csv(SCENARIO_SUMMARY, sep="\t", low_memory=False)
    aggregate = pd.read_csv(MODEL_AGGREGATE, sep="\t", low_memory=False)

    missing = [c for c in REQUIRED_SCENARIO_COLUMNS if c not in scenario.columns]
    if missing:
        raise RuntimeError(
            "Authoritative scenario_model_metric_summary.tsv schema changed; "
            f"missing columns: {missing}"
        )

    missing = [c for c in REQUIRED_AGGREGATE_COLUMNS if c not in aggregate.columns]
    if missing:
        raise RuntimeError(
            "Frozen model_aggregate_summary.tsv schema changed; "
            f"missing columns: {missing}"
        )

    return scenario, aggregate


def normalize_regime(x: Any) -> str:
    s = str(x).strip().upper()
    m = re.match(r"^(R[0-5])(?:_|$)", s)
    if not m:
        raise RuntimeError(f"Cannot normalize frozen transfer regime label: {x!r}")
    return m.group(1)


def prepare_main_scenario_table(scenario: pd.DataFrame) -> pd.DataFrame:
    x = scenario[scenario["model"].isin(MAIN_MODELS)].copy()

    if len(x) != EXPECTED_MAIN_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_MAIN_ROWS} main-model scenario rows; found {len(x)}."
        )

    if x["scenario_id"].nunique() != EXPECTED_SCENARIOS:
        raise RuntimeError(
            f"Expected {EXPECTED_SCENARIOS} frozen scenarios; "
            f"found {x['scenario_id'].nunique()}."
        )

    if x.duplicated(["scenario_id", "model"]).any():
        dup = x.loc[
            x.duplicated(["scenario_id", "model"], keep=False),
            ["scenario_id", "model"],
        ].head(10)
        raise RuntimeError(
            "Authoritative 05d summary is not unique at scenario x model.\n"
            + dup.to_string(index=False)
        )

    for model in MAIN_MODELS:
        n = int((x["model"] == model).sum())
        if n != EXPECTED_SCENARIOS:
            raise RuntimeError(f"{model}: expected 180 scenario rows, found {n}.")

    for c in [
        "mean_uno_c",
        "mean_delta_c_vs_B0",
        "negative_transfer_rate",
        "catastrophic_negative_transfer_rate",
    ]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
        vals = x[c].to_numpy(dtype=float)
        if not np.isfinite(vals).all():
            raise RuntimeError(f"Non-finite values in authoritative 05d field {c}.")

    x["replicates"] = pd.to_numeric(x["replicates"], errors="raise").astype(int)
    if (x["replicates"] <= 0).any():
        raise RuntimeError("Non-positive replicate count in frozen scenario summary.")

    x["regime"] = x["transfer_regime"].map(normalize_regime)

    scenario_regime = x[["scenario_id", "regime"]].drop_duplicates()
    if scenario_regime["scenario_id"].duplicated().any():
        raise RuntimeError("A frozen scenario maps to more than one transfer regime.")

    counts = (
        scenario_regime["regime"]
        .value_counts()
        .reindex(["R0", "R1", "R2", "R3", "R4", "R5"], fill_value=0)
        .astype(int)
        .to_dict()
    )
    if counts != EXPECTED_REGIME_COUNTS:
        raise RuntimeError(
            f"Frozen regime counts changed. Expected {EXPECTED_REGIME_COUNTS}; found {counts}."
        )

    core = x["scenario_id"].astype(str).str.startswith("CORE_")
    stress = x["scenario_id"].astype(str).str.startswith("STRESS_")
    scenario_class = (
        x.loc[:, ["scenario_id"]]
        .drop_duplicates()
        .assign(
            is_core=lambda d: d["scenario_id"].astype(str).str.startswith("CORE_"),
            is_stress=lambda d: d["scenario_id"].astype(str).str.startswith("STRESS_"),
        )
    )
    if int(scenario_class["is_core"].sum()) != 36:
        raise RuntimeError("Expected exactly 36 CORE scenarios.")
    if int(scenario_class["is_stress"].sum()) != 144:
        raise RuntimeError("Expected exactly 144 STRESS scenarios.")
    if not (core | stress).all():
        raise RuntimeError("Unexpected frozen scenario identifier outside CORE_/STRESS_.")

    # Structural B0 checks: the frozen reference must remain exactly zero-delta
    # and zero-safety-failure at every scenario.
    b0 = x[x["model"] == REFERENCE_MODEL]
    if not np.array_equal(
        b0["mean_delta_c_vs_B0"].to_numpy(dtype=float),
        np.zeros(len(b0), dtype=float),
    ):
        raise RuntimeError("B0 mean_delta_c_vs_B0 is not exactly zero in every scenario.")
    if not np.array_equal(
        b0["negative_transfer_rate"].to_numpy(dtype=float),
        np.zeros(len(b0), dtype=float),
    ):
        raise RuntimeError("B0 negative-transfer rate is not exactly zero in every scenario.")
    if not np.array_equal(
        b0["catastrophic_negative_transfer_rate"].to_numpy(dtype=float),
        np.zeros(len(b0), dtype=float),
    ):
        raise RuntimeError(
            "B0 catastrophic-negative-transfer rate is not exactly zero in every scenario."
        )

    return x.sort_values(["scenario_id", "model"], kind="mergesort").reset_index(drop=True)


def aggregate_overall(x: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in MAIN_MODELS:
        z = x[x["model"] == model]
        rows.append(
            {
                "model": model,
                "frozen_role": FROZEN_ROLES[model],
                "eligible_for_primary_selection": model in ELIGIBLE_MODELS,
                "n_frozen_scenarios": int(len(z)),
                "equal_scenario_mean_uno_c": float(z["mean_uno_c"].mean()),
                "mean_delta_uno_c_vs_B0_if_defined": float(
                    z["mean_delta_c_vs_B0"].mean()
                ),
                "negative_transfer_rate_at_deltaC_le_-0.02": float(
                    z["negative_transfer_rate"].mean()
                ),
                "catastrophic_transfer_rate_at_deltaC_le_-0.05": float(
                    z["catastrophic_negative_transfer_rate"].mean()
                ),
                "aggregation": (
                    "existing within-scenario metrics; equal weight across all 180 frozen scenarios"
                ),
                "analysis_role": "POST-HOLD DESCRIPTIVE CONTEXT",
            }
        )
    return pd.DataFrame(rows)


def aggregate_by_regime(x: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for regime in ["R0", "R1", "R2", "R3", "R4", "R5"]:
        for model in MAIN_MODELS:
            z = x[(x["regime"] == regime) & (x["model"] == model)]
            rows.append(
                {
                    "regime": regime,
                    "model": model,
                    "frozen_role": FROZEN_ROLES[model],
                    "eligible_for_primary_selection": model in ELIGIBLE_MODELS,
                    "n_frozen_scenarios": int(len(z)),
                    "mean_uno_c": float(z["mean_uno_c"].mean()),
                    "mean_delta_uno_c_vs_B0": float(
                        z["mean_delta_c_vs_B0"].mean()
                    ),
                    "negative_transfer_rate_at_deltaC_le_-0.02": float(
                        z["negative_transfer_rate"].mean()
                    ),
                    "catastrophic_transfer_rate_at_deltaC_le_-0.05": float(
                        z["catastrophic_negative_transfer_rate"].mean()
                    ),
                    "aggregation": (
                        "existing within-scenario metrics; equal weight across frozen scenarios within regime"
                    ),
                    "analysis_role": "POST-HOLD DESCRIPTIVE CONTEXT",
                }
            )
    return pd.DataFrame(rows)


def check_against_frozen_model_aggregate(
    overall: pd.DataFrame,
    frozen_aggregate: pd.DataFrame,
) -> list[dict[str, Any]]:
    frozen = frozen_aggregate[
        frozen_aggregate["model"].isin(MAIN_MODELS)
    ].copy()

    if set(frozen["model"]) != set(MAIN_MODELS):
        raise RuntimeError(
            "Frozen model_aggregate_summary.tsv does not contain exactly the seven main models."
        )
    if frozen["model"].duplicated().any():
        raise RuntimeError("Duplicate main-model rows in frozen model_aggregate_summary.tsv.")

    a = overall.set_index("model")
    f = frozen.set_index("model")

    pairs = [
        (
            "equal_scenario_mean_uno_c",
            "equal_scenario_mean_uno_c",
        ),
        (
            "mean_delta_uno_c_vs_B0_if_defined",
            "equal_scenario_mean_delta_c_vs_B0",
        ),
        (
            "negative_transfer_rate_at_deltaC_le_-0.02",
            "equal_scenario_negative_transfer_rate",
        ),
        (
            "catastrophic_transfer_rate_at_deltaC_le_-0.05",
            "equal_scenario_catastrophic_rate",
        ),
    ]

    checks: list[dict[str, Any]] = []
    for model in MAIN_MODELS:
        if int(f.loc[model, "n_scenarios"]) != EXPECTED_SCENARIOS:
            raise RuntimeError(
                f"{model}: frozen aggregate n_scenarios is not {EXPECTED_SCENARIOS}."
            )
        for export_col, frozen_col in pairs:
            obs = finite_float(a.loc[model, export_col], f"{model}/{export_col}")
            exp = finite_float(f.loc[model, frozen_col], f"{model}/{frozen_col}")
            diff = abs(obs - exp)
            passed = diff <= FROZEN_AGGREGATE_TOL
            checks.append(
                {
                    "check_family": "canonical_05d_vs_frozen_model_aggregate",
                    "model": model,
                    "quantity": export_col,
                    "observed": obs,
                    "expected": exp,
                    "absolute_difference": diff,
                    "tolerance": FROZEN_AGGREGATE_TOL,
                    "pass": passed,
                }
            )

    failed = [c for c in checks if not c["pass"]]
    if failed:
        msg = "\n".join(
            f"  {c['model']} {c['quantity']}: "
            f"observed={c['observed']:.12g}, expected={c['expected']:.12g}, "
            f"diff={c['absolute_difference']:.3g}"
            for c in failed
        )
        raise RuntimeError(
            "05h2b aggregation does not reproduce frozen model_aggregate_summary.tsv:\n"
            + msg
        )

    return checks


def check_independent_replay_anchors(
    overall: pd.DataFrame,
    by_regime: pd.DataFrame,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    o = overall.set_index("model")
    r5 = by_regime[by_regime["regime"] == "R5"].set_index("model")

    def add(
        model: str,
        quantity: str,
        observed: float,
        expected: float,
    ) -> None:
        diff = abs(observed - expected)
        checks.append(
            {
                "check_family": "accepted_independent_posthold_replay_anchor",
                "model": model,
                "quantity": quantity,
                "observed": observed,
                "expected": expected,
                "absolute_difference": diff,
                "tolerance": REPLAY_TOL,
                "pass": diff <= REPLAY_TOL,
            }
        )

    for model in MAIN_MODELS:
        add(
            model,
            "equal_scenario_mean_uno_c",
            float(o.loc[model, "equal_scenario_mean_uno_c"]),
            EXPECTED_EQUAL_SCENARIO_UNO_C[model],
        )
        add(
            model,
            "negative_transfer_rate_at_deltaC_le_-0.02",
            float(o.loc[model, "negative_transfer_rate_at_deltaC_le_-0.02"]),
            EXPECTED_NEGATIVE_TRANSFER_RATE[model],
        )
        add(
            model,
            "R5_catastrophic_transfer_rate_at_deltaC_le_-0.05",
            float(
                r5.loc[
                    model,
                    "catastrophic_transfer_rate_at_deltaC_le_-0.05",
                ]
            ),
            EXPECTED_R5_CATASTROPHIC_RATE[model],
        )

    failed = [c for c in checks if not c["pass"]]
    if failed:
        msg = "\n".join(
            f"  {c['model']} {c['quantity']}: "
            f"observed={c['observed']:.9f}, expected={c['expected']:.9f}"
            for c in failed
        )
        raise RuntimeError(
            "Accepted independent replay anchors were not reproduced:\n" + msg
        )
    return checks


def build_source_inventory() -> pd.DataFrame:
    rows = []
    for role, p in [
        ("05h0_frozen_strengthening_contract", H0_JSON),
        ("authoritative_05d_scenario_model_metric_summary", SCENARIO_SUMMARY),
        ("frozen_05d_model_aggregate_summary_crosscheck", MODEL_AGGREGATE),
        ("frozen_05d_metric_implementation_contract", METRIC_CONTRACT),
        ("frozen_05d_architecture_selection", FROZEN_SELECTION),
    ]:
        rows.append(
            {
                "source_role": role,
                "relative_path": str(p.relative_to(ROOT)),
                "sha256": sha256_file(p),
                "bytes": p.stat().st_size,
            }
        )
    return pd.DataFrame(rows)


def build_readme(overall: pd.DataFrame) -> str:
    o = overall.set_index("model")
    return f"""Paper 6 - 05h2b frozen all-model safety summary

Analysis role
-------------
POST-HOLD DESCRIPTIVE CONTEXT.

Scientific source
-----------------
The exporter is explicitly bound to the authoritative frozen 05d table:
results/simulation_phase_diagram/05d/scenario_model_metric_summary.tsv

No diagnostic table is used as the scientific source.

Existing scenario-level fields used directly
--------------------------------------------
mean_uno_c
mean_delta_c_vs_B0
negative_transfer_rate
catastrophic_negative_transfer_rate

The catastrophic quantity is therefore NOT reconstructed from a scenario flag.

Frozen model family
-------------------
B0, B4, A0, A1, A2, A3, A4.
B0 is the reference.
Only A2 and A3 remain eligible for the frozen primary architecture-selection rule.
This descriptive export cannot make another model selectable and cannot reopen
HOLD_NO_SAFE_SELECTABLE_ARCHITECTURE.

Frozen safety definitions
-------------------------
Negative transfer: delta Uno-C <= -0.02.
Catastrophic negative transfer: delta Uno-C <= -0.05.

Aggregation
-----------
The source table already contains within-scenario means/rates. 05h2b assigns equal
weight to each of the 180 frozen scenarios. Regime summaries assign equal weight
to each frozen scenario within its R0-R5 regime.

Structural checks
-----------------
180 total scenarios.
36 CORE scenarios.
144 STRESS scenarios.
R0/R1/R2/R3/R4/R5 counts = 6/6/78/6/6/78.
1260 rows for the seven frozen main models.
B0 delta-C, negative-transfer rate, and catastrophic-transfer rate are exactly zero
for every frozen scenario.

Frozen aggregate cross-check
----------------------------
Every exported all-model overall quantity is independently compared against the
pre-existing 05d model_aggregate_summary.tsv.

Accepted replay cross-checks
----------------------------
Previously accepted post-HOLD all-model Uno-C, negative-transfer rates, and R5
catastrophic rates must also be reproduced within the frozen rounding tolerance.

Key quantities after a PASS
---------------------------
A2 mean Uno-C: {o.loc['A2', 'equal_scenario_mean_uno_c']:.6f}
A2 negative-transfer rate: {o.loc['A2', 'negative_transfer_rate_at_deltaC_le_-0.02']:.6f}
A3 mean Uno-C: {o.loc['A3', 'equal_scenario_mean_uno_c']:.6f}
A3 negative-transfer rate: {o.loc['A3', 'negative_transfer_rate_at_deltaC_le_-0.02']:.6f}

Prohibited operations
---------------------
No model fitting.
No retuning.
No reading of TARGET, GSE21257, GSE39055, or DOG2 outcomes.
No recomputation of survival metrics from predictions.
No threshold change.
No architecture-selection change.
"""


def main() -> None:
    print("=" * 118)
    print("Paper 6 - export frozen all-model safety summary for CBM strengthening")
    print("=" * 118)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {ROOT}")
    print(f"Output directory: {OUT_DIR}")
    print()
    print("Safety / execution contract:")
    print("  TARGET survival outcomes read: NO")
    print("  GSE21257 outcomes read: NO")
    print("  GSE39055 outcomes read: NO")
    print("  DOG2 outcomes read: NO")
    print("  model fitting / retuning: NO")
    print("  survival-metric re-estimation from prediction arrays: NO")
    print("  catastrophic rates reconstructed from scenario flags: NO")
    print("  frozen thresholds changed: NO")
    print("  frozen A2/A3 selection decision reopened: NO")
    print("  authoritative frozen 05d scenario/model summary read: YES")
    print("  frozen 05d aggregate summary used as independent cross-check: YES")
    print()

    if OUT_DIR.exists():
        raise RuntimeError(
            f"Final 05h2b output already exists; refusing overwrite: {OUT_DIR}"
        )
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir(parents=True, exist_ok=False)

    try:
        verify_h0_contract()
        print("05h0 all-model strengthening contract: PASS")
        print(f"  SHA256: {sha256_file(H0_JSON)}")
        print()

        scenario_raw, aggregate_raw = load_frozen_tables()

        print("Authoritative source binding: PASS")
        print(f"  scenario source: {SCENARIO_SUMMARY.relative_to(ROOT)}")
        print(f"  scenario source SHA256: {sha256_file(SCENARIO_SUMMARY)}")
        print(f"  source rows before 7-model filter: {len(scenario_raw)}")
        print(
            "  direct catastrophic field: catastrophic_negative_transfer_rate"
        )
        print("  diagnostic 05d0/05d0a tables used as scientific source: NO")
        print()

        scenario = prepare_main_scenario_table(scenario_raw)

        scenario_regime = scenario[
            ["scenario_id", "regime"]
        ].drop_duplicates()
        print("Frozen scenario structure: PASS")
        print(f"  scenarios: {scenario['scenario_id'].nunique()}/180")
        print(f"  main models: {scenario['model'].nunique()}/7")
        print(f"  scenario x main-model rows: {len(scenario)}/1260")
        print(
            "  CORE/STRESS: "
            f"{scenario_regime['scenario_id'].str.startswith('CORE_').sum()}/"
            f"{scenario_regime['scenario_id'].str.startswith('STRESS_').sum()}"
        )
        for r in ["R0", "R1", "R2", "R3", "R4", "R5"]:
            print(
                f"  {r}: "
                f"{int((scenario_regime['regime'] == r).sum())} scenarios"
            )
        print()

        overall = aggregate_overall(scenario)
        by_regime = aggregate_by_regime(scenario)

        checks_frozen = check_against_frozen_model_aggregate(
            overall, aggregate_raw
        )
        checks_replay = check_independent_replay_anchors(
            overall, by_regime
        )
        checks = pd.DataFrame(checks_frozen + checks_replay)

        print("-" * 118)
        print("05h2b ALL-MODEL DESCRIPTIVE SAFETY SUMMARY")
        print("-" * 118)
        display_cols = [
            "model",
            "eligible_for_primary_selection",
            "equal_scenario_mean_uno_c",
            "mean_delta_uno_c_vs_B0_if_defined",
            "negative_transfer_rate_at_deltaC_le_-0.02",
            "catastrophic_transfer_rate_at_deltaC_le_-0.05",
        ]
        print(
            overall[display_cols].to_string(
                index=False,
                float_format=lambda v: f"{v:.6f}",
            )
        )
        print()

        print("Frozen 05d aggregate reproduction: PASS")
        print(f"  checks: {len(checks_frozen)}/{len(checks_frozen)}")
        print("Accepted independent post-HOLD replay anchors: PASS")
        print(f"  checks: {len(checks_replay)}/{len(checks_replay)}")
        print("Frozen primary architecture-selection decision changed: NO")
        print("Non-eligible models made selectable: NO")
        print()

        overall_path = (
            WORK_DIR / "Supplementary_Table_all_model_safety_summary.csv"
        )
        regime_path = (
            WORK_DIR / "Supplementary_Data_all_model_regime_safety.csv"
        )
        scenario_path = (
            WORK_DIR / "Supplementary_Data_all_model_scenario_safety.csv"
        )
        checks_path = WORK_DIR / "replay_and_frozen_crosschecks.csv"
        sources_path = WORK_DIR / "source_artifact_inventory.csv"
        json_path = WORK_DIR / "all_model_safety_summary.json"
        readme_path = WORK_DIR / "README_for_supplement.txt"

        overall.to_csv(
            overall_path, index=False, float_format="%.10g"
        )
        by_regime.to_csv(
            regime_path, index=False, float_format="%.10g"
        )

        scenario_export = scenario[
            [
                "scenario_id",
                "regime",
                "transfer_regime",
                "replicates",
                "model",
                "mean_uno_c",
                "mean_delta_c_vs_B0",
                "negative_transfer_rate",
                "catastrophic_negative_transfer_rate",
            ]
        ].copy()
        scenario_export["frozen_role"] = scenario_export["model"].map(
            FROZEN_ROLES
        )
        scenario_export[
            "eligible_for_primary_selection"
        ] = scenario_export["model"].isin(ELIGIBLE_MODELS)
        scenario_export.to_csv(
            scenario_path, index=False, float_format="%.10g"
        )

        checks.to_csv(
            checks_path, index=False, float_format="%.12g"
        )

        source_inventory = build_source_inventory()
        source_inventory.to_csv(sources_path, index=False)

        summary = {
            "script_version": SCRIPT_VERSION,
            "analysis_role": "POST-HOLD DESCRIPTIVE CONTEXT",
            "status": "PASS_FROZEN_ALL_MODEL_SAFETY_SUMMARY_EXPORTED",
            "frozen_primary_decision": "HOLD_NO_SAFE_SELECTABLE_ARCHITECTURE",
            "primary_decision_changed": False,
            "reference_model": REFERENCE_MODEL,
            "models": MAIN_MODELS,
            "eligible_for_primary_selection": sorted(ELIGIBLE_MODELS),
            "thresholds": {
                "negative_transfer_delta_uno_c": NEGATIVE_THRESHOLD,
                "catastrophic_transfer_delta_uno_c": CATASTROPHIC_THRESHOLD,
            },
            "authoritative_scientific_source": str(
                SCENARIO_SUMMARY.relative_to(ROOT)
            ),
            "authoritative_scientific_source_sha256": sha256_file(
                SCENARIO_SUMMARY
            ),
            "direct_source_fields": [
                "mean_uno_c",
                "mean_delta_c_vs_B0",
                "negative_transfer_rate",
                "catastrophic_negative_transfer_rate",
            ],
            "aggregation": (
                "existing within-scenario metrics; equal weight per frozen scenario"
            ),
            "n_scenarios": EXPECTED_SCENARIOS,
            "regime_counts": EXPECTED_REGIME_COUNTS,
            "model_summary": overall.to_dict(orient="records"),
            "all_crosschecks_passed": True,
            "human_outcomes_read": False,
            "model_fitting": False,
            "survival_metric_reestimation": False,
            "interpretation_rule": (
                "Non-eligible models remain non-eligible. Descriptive 05h2b results "
                "cannot reopen the frozen primary architecture-selection decision."
            ),
        }
        write_json(json_path, summary)

        readme_path.write_text(
            build_readme(overall),
            encoding="utf-8",
            newline="\n",
        )

        output_files = sorted(
            p for p in WORK_DIR.iterdir()
            if p.is_file() and p.name != "freeze_manifest.json"
        )

        manifest = {
            "script_version": SCRIPT_VERSION,
            "script_relative_path": str(
                Path(__file__).resolve().relative_to(ROOT)
            ),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "05h0_contract_sha256": sha256_file(H0_JSON),
            "authoritative_05d_scenario_summary_sha256": sha256_file(
                SCENARIO_SUMMARY
            ),
            "frozen_05d_model_aggregate_sha256": sha256_file(
                MODEL_AGGREGATE
            ),
            "status": "PASS_FROZEN_ALL_MODEL_SAFETY_SUMMARY_EXPORTED",
            "outputs": [
                {
                    "file": p.name,
                    "sha256": sha256_file(p),
                    "bytes": p.stat().st_size,
                }
                for p in output_files
            ],
        }
        write_json(WORK_DIR / "freeze_manifest.json", manifest)

        # Verify output hashes before atomic finalization.
        for item in manifest["outputs"]:
            p = WORK_DIR / item["file"]
            if sha256_file(p) != item["sha256"]:
                raise RuntimeError(
                    f"Output changed before finalization: {p.name}"
                )

        os.replace(WORK_DIR, OUT_DIR)

        print("=" * 118)
        print("05h2b frozen all-model safety summary: PASS")
        print("=" * 118)
        for p in sorted(OUT_DIR.iterdir()):
            if p.is_file():
                print(f"Created: {p}")

    except Exception:
        if WORK_DIR.exists():
            shutil.rmtree(WORK_DIR, ignore_errors=True)
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 118)
        print("05h2b frozen all-model safety summary: FAIL")
        print(f"{type(exc).__name__}: {exc}")
        print("=" * 118)
        raise
