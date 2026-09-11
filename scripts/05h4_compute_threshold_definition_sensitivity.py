#!/usr/bin/env python
"""
Paper 6 - post-HOLD threshold-definition sensitivity for the frozen safety profile.

This stage implements ONLY the prespecified neighboring definitions frozen in 05h0:

Negative-transfer severity grid:
    delta Uno-C <= -0.01, -0.02, -0.03

Catastrophic-transfer severity grid:
    delta Uno-C <= -0.04, -0.05, -0.06

The original definitions (-0.02 and -0.05) remain the only frozen primary
definitions. Reference rate lines are descriptive only and cannot reopen or
replace the original architecture-selection decision.

Technical identities are inherited from the PASS 05h3c input contract; this
script performs no schema discovery, model fitting, retuning, or human-outcome
access.

Aggregation:
    within each of the 180 fixed frozen scenarios, calculate the replicate-level
    event rate at the specified severity definition; then give each scenario
    equal weight. This reproduces the frozen-original estimand.

Expected placement:
    scripts/05h4_compute_threshold_definition_sensitivity.py

Run:
    python scripts\05h4_compute_threshold_definition_sensitivity.py
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_VERSION = "05h4-compute-threshold-definition-sensitivity-v1-no-discovery-no-cli"

PRIMARY_MODELS = ["A2", "A3"]

NEGATIVE_GRID = [-0.01, -0.02, -0.03]
CATASTROPHIC_GRID = [-0.04, -0.05, -0.06]

PRIMARY_NEGATIVE_THRESHOLD = -0.02
PRIMARY_CATASTROPHIC_THRESHOLD = -0.05

NEGATIVE_REFERENCE_RATE_LINES = [0.05, 0.10, 0.15]
CATASTROPHIC_REFERENCE_RATE_LINES = [0.025, 0.05, 0.075]

REPLAY_TOL = 5e-10


def project_root() -> Path:
    p = Path(__file__).resolve()
    if p.parent.name.lower() != "scripts":
        raise RuntimeError(
            "Place 05h4_compute_threshold_definition_sensitivity.py "
            "in the repository scripts/ directory."
        )
    return p.parent.parent


ROOT = project_root()

H0_JSON = (
    ROOT
    / "method_contract"
    / "05h0_cbm_strengthening_contract"
    / "cbm_strengthening_contract.json"
)

H3C_DIR = ROOT / "method_contract" / "05h3c_aggregation_input_contract"
H3C_CONTRACT = H3C_DIR / "aggregation_input_contract.json"
H3C_MANIFEST = H3C_DIR / "freeze_manifest.json"
H3C_DESIGN = H3C_DIR / "resolved_frozen_scenario_design.tsv"
H3C_NPZ_INVENTORY = H3C_DIR / "retained_npz_source_inventory.csv"

H3D_DIR = ROOT / "method_contract" / "05h3d_aggregation_sensitivity"
H3D_JSON = H3D_DIR / "aggregation_sensitivity.json"
H3D_MANIFEST = H3D_DIR / "freeze_manifest.json"

OUT_DIR = ROOT / "method_contract" / "05h4_threshold_definition_sensitivity"
WORK_DIR = ROOT / "method_contract" / ".05h4_threshold_definition_sensitivity_work"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.write_text(
        json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def recursive_find_key(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            hit = recursive_find_key(value, key)
            if hit is not None:
                return hit
    elif isinstance(obj, list):
        for value in obj:
            hit = recursive_find_key(value, key)
            if hit is not None:
                return hit
    return None


def verify_05h0_threshold_contract() -> dict[str, Any]:
    if not H0_JSON.exists():
        raise FileNotFoundError(f"Missing frozen 05h0 contract: {H0_JSON}")

    h0 = load_json(H0_JSON)
    block = recursive_find_key(h0, "threshold_definition_sensitivity")
    if not isinstance(block, dict):
        raise RuntimeError("05h0 lacks threshold_definition_sensitivity.")

    expected_role = (
        "POST-HOLD SENSITIVITY OF THE ESTIMATED SAFETY PROFILE; "
        "NOT A REOPENED PASS/FAIL DECISION"
    )
    if block.get("analysis_role") != expected_role:
        raise RuntimeError(
            f"05h0 threshold-sensitivity analysis role changed: "
            f"{block.get('analysis_role')!r}"
        )

    if block.get("primary_models") != PRIMARY_MODELS:
        raise RuntimeError(
            f"05h0 threshold-sensitivity primary models changed: "
            f"{block.get('primary_models')!r}"
        )

    if [float(x) for x in block.get("negative_transfer_severity_grid_delta_uno_c", [])] != NEGATIVE_GRID:
        raise RuntimeError("05h0 negative-transfer severity grid changed.")

    if [float(x) for x in block.get("catastrophic_transfer_severity_grid_delta_uno_c", [])] != CATASTROPHIC_GRID:
        raise RuntimeError("05h0 catastrophic-transfer severity grid changed.")

    reference = block.get("reference_rate_lines", {})
    if [float(x) for x in reference.get("negative_transfer", [])] != NEGATIVE_REFERENCE_RATE_LINES:
        raise RuntimeError("05h0 negative-transfer reference rate lines changed.")
    if [float(x) for x in reference.get("catastrophic_transfer", [])] != CATASTROPHIC_REFERENCE_RATE_LINES:
        raise RuntimeError("05h0 catastrophic-transfer reference rate lines changed.")

    presentation = str(block.get("presentation_rule", ""))
    if "-0.02" not in presentation or "-0.05" not in presentation:
        raise RuntimeError(
            "05h0 presentation rule no longer explicitly preserves the frozen definitions."
        )
    if "never replaced" not in presentation:
        raise RuntimeError(
            "05h0 presentation rule no longer contains the no-replacement guardrail."
        )

    return h0


def verify_frozen_chain() -> tuple[dict[str, Any], dict[str, Any]]:
    required = [
        H3C_CONTRACT,
        H3C_MANIFEST,
        H3C_DESIGN,
        H3C_NPZ_INVENTORY,
        H3D_JSON,
        H3D_MANIFEST,
    ]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"Required prior PASS artifact missing: {p}")

    h3c = load_json(H3C_CONTRACT)
    h3c_manifest = load_json(H3C_MANIFEST)
    h3d = load_json(H3D_JSON)
    h3d_manifest = load_json(H3D_MANIFEST)

    if h3c.get("status") != "PASS_READY_FOR_05H3D_AGGREGATION_SENSITIVITY":
        raise RuntimeError("05h3c contract is not PASS.")
    if h3c_manifest.get("status") != "PASS_READY_FOR_05H3D_AGGREGATION_SENSITIVITY":
        raise RuntimeError("05h3c manifest is not PASS.")
    if h3d.get("status") != "PASS_POSTHOLD_AGGREGATION_SENSITIVITY":
        raise RuntimeError("05h3d scientific result is not PASS.")
    if h3d_manifest.get("status") != "PASS_POSTHOLD_AGGREGATION_SENSITIVITY":
        raise RuntimeError("05h3d manifest is not PASS.")

    if bool(h3d.get("primary_decision_changed")):
        raise RuntimeError("05h3d says the frozen primary decision changed.")
    if h3d.get("primary_models") != PRIMARY_MODELS:
        raise RuntimeError("05h3d primary-model registry differs from A2/A3.")

    # Verify 05h3c output hashes using its own freeze manifest.
    output_hashes = {
        str(row["file"]): str(row["sha256"])
        for row in h3c_manifest.get("outputs", [])
        if isinstance(row, dict) and "file" in row and "sha256" in row
    }
    for p in [H3C_CONTRACT, H3C_DESIGN, H3C_NPZ_INVENTORY]:
        expected = output_hashes.get(p.name)
        if expected is None:
            raise RuntimeError(f"05h3c manifest does not lock {p.name}.")
        if sha256_file(p) != expected:
            raise RuntimeError(f"05h3c output changed after freeze: {p.name}")

    return h3c, h3d


def verify_h3c_semantics(h3c: dict[str, Any]) -> None:
    threshold = h3c.get("frozen_thresholds", {})
    if float(threshold.get("negative_transfer_delta_uno_c")) != PRIMARY_NEGATIVE_THRESHOLD:
        raise RuntimeError("05h3c negative-transfer threshold is not -0.02.")
    if float(threshold.get("catastrophic_transfer_delta_uno_c")) != PRIMARY_CATASTROPHIC_THRESHOLD:
        raise RuntimeError("05h3c catastrophic-transfer threshold is not -0.05.")
    if threshold.get("retained_flags_match_thresholds_exactly") is not True:
        raise RuntimeError("05h3c did not verify exact primary threshold flags.")

    schema = h3c.get("retained_metric_schema", {})
    order = schema.get("model_axis_order")
    if not isinstance(order, list):
        raise RuntimeError("05h3c lacks model_axis_order.")
    if "A2" not in order or "A3" not in order:
        raise RuntimeError("05h3c model axis lacks A2/A3.")
    if int(schema.get("A2_index")) != order.index("A2"):
        raise RuntimeError("05h3c A2_index is inconsistent.")
    if int(schema.get("A3_index")) != order.index("A3"):
        raise RuntimeError("05h3c A3_index is inconsistent.")

    reps = h3c.get("replicates", {})
    if int(reps.get("n_scenarios")) != 180:
        raise RuntimeError("05h3c scenario count changed.")
    if int(reps.get("retained_npz_count")) != 180:
        raise RuntimeError("05h3c retained NPZ count changed.")
    if int(reps.get("total_replicates")) != 21600:
        raise RuntimeError("05h3c retained replicate total changed.")


def load_and_verify_design(h3c: dict[str, Any]) -> pd.DataFrame:
    design = pd.read_csv(H3C_DESIGN, sep="\t", low_memory=False)

    required = ["scenario_id", "scenario_class", "family", "transfer_regime", "regime", "replicates"]
    missing = [c for c in required if c not in design.columns]
    if missing:
        raise RuntimeError(f"05h3c frozen design lost columns: {missing}")

    if len(design) != 180 or design["scenario_id"].nunique() != 180:
        raise RuntimeError("05h3c frozen design is not 180 unique scenarios.")

    core_locked = set(h3c["scenario_class_rule"]["core_scenario_ids"])
    stress_locked = set(h3c["scenario_class_rule"]["stress_scenario_ids"])
    core_obs = set(design.loc[design["scenario_class"] == "CORE", "scenario_id"].astype(str))
    stress_obs = set(design.loc[design["scenario_class"] == "STRESS", "scenario_id"].astype(str))
    if core_obs != core_locked:
        raise RuntimeError("05h3c CORE membership changed.")
    if stress_obs != stress_locked:
        raise RuntimeError("05h3c STRESS membership changed.")

    design["replicates"] = pd.to_numeric(design["replicates"], errors="raise").astype(int)
    if int(design["replicates"].sum()) != 21600:
        raise RuntimeError("05h3c design replicate total changed.")

    return design.sort_values("scenario_id").reset_index(drop=True)


def load_and_verify_inventory(
    h3c: dict[str, Any],
    design: pd.DataFrame,
) -> pd.DataFrame:
    inv = pd.read_csv(H3C_NPZ_INVENTORY, low_memory=False)

    required = [
        "scenario_id",
        "scenario_class",
        "family",
        "regime",
        "replicates",
        "npz_relative_path",
        "npz_sha256",
        "npz_bytes",
    ]
    missing = [c for c in required if c not in inv.columns]
    if missing:
        raise RuntimeError(f"05h3c NPZ inventory lost columns: {missing}")

    if len(inv) != 180 or inv["scenario_id"].nunique() != 180:
        raise RuntimeError("05h3c NPZ inventory is not 180 unique scenarios.")

    expected_sids = set(design["scenario_id"].astype(str))
    if set(inv["scenario_id"].astype(str)) != expected_sids:
        raise RuntimeError("05h3c NPZ inventory scenario set changed.")

    # Re-verify every NPZ hash/size before using delta arrays.
    for row in inv.itertuples(index=False):
        path = ROOT / str(row.npz_relative_path)
        if not path.exists():
            raise FileNotFoundError(f"Retained NPZ missing: {path}")
        if sha256_file(path) != str(row.npz_sha256):
            raise RuntimeError(f"Retained NPZ hash changed: {row.scenario_id}")
        if path.stat().st_size != int(row.npz_bytes):
            raise RuntimeError(f"Retained NPZ byte size changed: {row.scenario_id}")

    return inv.sort_values("scenario_id").reset_index(drop=True)


def primary_rates_from_h3d(h3d: dict[str, Any]) -> dict[tuple[str, str], float]:
    points = h3d.get("point_estimates")
    if not isinstance(points, list):
        raise RuntimeError("05h3d lacks point_estimates.")

    lookup: dict[tuple[str, str], float] = {}
    for row in points:
        if not isinstance(row, dict):
            continue
        if row.get("estimand") != "frozen_original":
            continue
        model = str(row.get("model"))
        if model not in PRIMARY_MODELS:
            continue
        lookup[(model, "negative")] = float(
            row["negative_transfer_rate_at_deltaC_le_-0.02"]
        )
        lookup[(model, "catastrophic")] = float(
            row["catastrophic_transfer_rate_at_deltaC_le_-0.05"]
        )

    if set(lookup) != {
        ("A2", "negative"),
        ("A2", "catastrophic"),
        ("A3", "negative"),
        ("A3", "catastrophic"),
    }:
        raise RuntimeError("Could not resolve primary frozen-original rates from 05h3d.")

    return lookup


def compute_sensitivity(
    h3c: dict[str, Any],
    design: pd.DataFrame,
    inventory: pd.DataFrame,
    h3d: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    schema = h3c["retained_metric_schema"]
    model_order = list(schema["model_axis_order"])
    model_indices = {
        "A2": int(schema["A2_index"]),
        "A3": int(schema["A3_index"]),
    }
    for model in PRIMARY_MODELS:
        idx = model_indices[model]
        if model_order[idx] != model:
            raise RuntimeError(f"Frozen model-axis mismatch for {model}.")

    inv_lookup = inventory.set_index("scenario_id")
    design_sids = list(design["scenario_id"].astype(str))

    all_thresholds = sorted(
        set(NEGATIVE_GRID + CATASTROPHIC_GRID),
        reverse=True,
    )

    # Store scenario-level rates so the final aggregate is demonstrably
    # equal-scenario weighted rather than replicate-pooled.
    scenario_rows: list[dict[str, Any]] = []

    for sid in design_sids:
        row = inv_lookup.loc[sid]
        path = ROOT / str(row["npz_relative_path"])

        with np.load(path, allow_pickle=False) as npz:
            if "delta_c_vs_B0" not in npz.files:
                raise RuntimeError(f"{sid}: retained NPZ lacks delta_c_vs_B0.")
            delta_all = np.asarray(npz["delta_c_vs_B0"], dtype=float)

        expected_n = int(row["replicates"])
        expected_shape = (expected_n, len(model_order))
        if delta_all.shape != expected_shape:
            raise RuntimeError(
                f"{sid}: delta_c_vs_B0 shape={delta_all.shape}, expected={expected_shape}"
            )

        for model in PRIMARY_MODELS:
            delta = delta_all[:, model_indices[model]]
            if not np.isfinite(delta).all():
                raise RuntimeError(f"{sid}/{model}: non-finite delta values.")

            for threshold in all_thresholds:
                scenario_rows.append(
                    {
                        "scenario_id": sid,
                        "model": model,
                        "threshold_delta_uno_c": float(threshold),
                        "scenario_rate": float(np.mean(delta <= threshold)),
                        "replicates": expected_n,
                    }
                )

    scenario_rates = pd.DataFrame(scenario_rows)

    rows: list[dict[str, Any]] = []

    for model in PRIMARY_MODELS:
        for threshold in NEGATIVE_GRID:
            z = scenario_rates[
                (scenario_rates["model"] == model)
                & (scenario_rates["threshold_delta_uno_c"] == threshold)
            ]
            if len(z) != 180:
                raise RuntimeError(
                    f"{model}/negative/{threshold}: expected 180 scenario rates, found {len(z)}."
                )
            rate = float(z["scenario_rate"].mean())
            rows.append(
                {
                    "definition_family": "negative_transfer",
                    "model": model,
                    "severity_threshold_delta_uno_c": threshold,
                    "is_frozen_primary_definition": threshold == PRIMARY_NEGATIVE_THRESHOLD,
                    "equal_scenario_rate": rate,
                    "difference_vs_frozen_primary_definition": np.nan,
                    "reference_rate_line_1": NEGATIVE_REFERENCE_RATE_LINES[0],
                    "reference_rate_line_2": NEGATIVE_REFERENCE_RATE_LINES[1],
                    "reference_rate_line_3": NEGATIVE_REFERENCE_RATE_LINES[2],
                    "analysis_role": (
                        "POST-HOLD SENSITIVITY OF ESTIMATED SAFETY PROFILE; "
                        "NOT A REOPENED PASS/FAIL DECISION"
                    ),
                }
            )

        for threshold in CATASTROPHIC_GRID:
            z = scenario_rates[
                (scenario_rates["model"] == model)
                & (scenario_rates["threshold_delta_uno_c"] == threshold)
            ]
            if len(z) != 180:
                raise RuntimeError(
                    f"{model}/catastrophic/{threshold}: expected 180 scenario rates, found {len(z)}."
                )
            rate = float(z["scenario_rate"].mean())
            rows.append(
                {
                    "definition_family": "catastrophic_transfer",
                    "model": model,
                    "severity_threshold_delta_uno_c": threshold,
                    "is_frozen_primary_definition": threshold == PRIMARY_CATASTROPHIC_THRESHOLD,
                    "equal_scenario_rate": rate,
                    "difference_vs_frozen_primary_definition": np.nan,
                    "reference_rate_line_1": CATASTROPHIC_REFERENCE_RATE_LINES[0],
                    "reference_rate_line_2": CATASTROPHIC_REFERENCE_RATE_LINES[1],
                    "reference_rate_line_3": CATASTROPHIC_REFERENCE_RATE_LINES[2],
                    "analysis_role": (
                        "POST-HOLD SENSITIVITY OF ESTIMATED SAFETY PROFILE; "
                        "NOT A REOPENED PASS/FAIL DECISION"
                    ),
                }
            )

    summary = pd.DataFrame(rows)

    # Fill changes versus the original frozen definition within family/model.
    for family, primary_threshold in [
        ("negative_transfer", PRIMARY_NEGATIVE_THRESHOLD),
        ("catastrophic_transfer", PRIMARY_CATASTROPHIC_THRESHOLD),
    ]:
        for model in PRIMARY_MODELS:
            mask = (
                (summary["definition_family"] == family)
                & (summary["model"] == model)
            )
            primary_row = summary[
                mask
                & (summary["severity_threshold_delta_uno_c"] == primary_threshold)
            ]
            if len(primary_row) != 1:
                raise RuntimeError(f"Cannot resolve unique primary row for {family}/{model}.")
            primary_rate = float(primary_row.iloc[0]["equal_scenario_rate"])
            summary.loc[
                mask,
                "difference_vs_frozen_primary_definition",
            ] = (
                summary.loc[mask, "equal_scenario_rate"] - primary_rate
            )

    # Exact chain check at the original definitions.
    expected_primary = primary_rates_from_h3d(h3d)
    check_rows = []
    for model in PRIMARY_MODELS:
        observed_nt = float(
            summary[
                (summary["definition_family"] == "negative_transfer")
                & (summary["model"] == model)
                & (summary["severity_threshold_delta_uno_c"] == PRIMARY_NEGATIVE_THRESHOLD)
            ].iloc[0]["equal_scenario_rate"]
        )
        expected_nt = expected_primary[(model, "negative")]
        nt_diff = abs(observed_nt - expected_nt)
        check_rows.append(
            {
                "model": model,
                "definition_family": "negative_transfer",
                "threshold": PRIMARY_NEGATIVE_THRESHOLD,
                "observed": observed_nt,
                "expected_05h3d": expected_nt,
                "absolute_difference": nt_diff,
                "tolerance": REPLAY_TOL,
                "pass": nt_diff <= REPLAY_TOL,
            }
        )

        observed_cat = float(
            summary[
                (summary["definition_family"] == "catastrophic_transfer")
                & (summary["model"] == model)
                & (summary["severity_threshold_delta_uno_c"] == PRIMARY_CATASTROPHIC_THRESHOLD)
            ].iloc[0]["equal_scenario_rate"]
        )
        expected_cat = expected_primary[(model, "catastrophic")]
        cat_diff = abs(observed_cat - expected_cat)
        check_rows.append(
            {
                "model": model,
                "definition_family": "catastrophic_transfer",
                "threshold": PRIMARY_CATASTROPHIC_THRESHOLD,
                "observed": observed_cat,
                "expected_05h3d": expected_cat,
                "absolute_difference": cat_diff,
                "tolerance": REPLAY_TOL,
                "pass": cat_diff <= REPLAY_TOL,
            }
        )

    checks = pd.DataFrame(check_rows)
    if not checks["pass"].all():
        raise RuntimeError(
            "05h4 primary definitions do not reproduce 05h3d frozen-original rates:\n"
            + checks.loc[~checks["pass"]].to_string(index=False)
        )

    return summary, scenario_rates


def add_descriptive_reference_line_indicators(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out["at_or_below_reference_rate_line_1"] = (
        out["equal_scenario_rate"] <= out["reference_rate_line_1"]
    )
    out["at_or_below_reference_rate_line_2"] = (
        out["equal_scenario_rate"] <= out["reference_rate_line_2"]
    )
    out["at_or_below_reference_rate_line_3"] = (
        out["equal_scenario_rate"] <= out["reference_rate_line_3"]
    )
    return out


def build_readme(summary: pd.DataFrame) -> str:
    lines = [
        "Paper 6 - 05h4 threshold-definition sensitivity",
        "",
        "Role",
        "----",
        "POST-HOLD SENSITIVITY OF THE ESTIMATED SAFETY PROFILE.",
        "This is NOT a reopened pass/fail decision.",
        "",
        "Primary definitions remain unchanged",
        "------------------------------------",
        "Negative transfer: delta Uno-C <= -0.02.",
        "Catastrophic transfer: delta Uno-C <= -0.05.",
        "The original frozen architecture-selection decision is never replaced.",
        "",
        "Prespecified neighboring severity definitions from 05h0",
        "-------------------------------------------------------",
        "Negative transfer: -0.01, -0.02, -0.03.",
        "Catastrophic transfer: -0.04, -0.05, -0.06.",
        "",
        "Aggregation",
        "-----------",
        "For each fixed frozen scenario, the replicate-level rate is calculated at",
        "the specified severity threshold. The 180 scenario rates are then averaged",
        "with equal scenario weight, matching the frozen-original estimand.",
        "",
        "Reference rate lines",
        "--------------------",
        "Negative-transfer descriptive reference lines: 0.05, 0.10, 0.15.",
        "Catastrophic-transfer descriptive reference lines: 0.025, 0.05, 0.075.",
        "These are presentation aids only; neighboring definitions/reference lines",
        "cannot create a new PASS or replace the original frozen decision.",
        "",
        "Results",
        "-------",
    ]

    for _, row in summary.sort_values(
        ["definition_family", "model", "severity_threshold_delta_uno_c"],
        ascending=[True, True, False],
    ).iterrows():
        marker = " [FROZEN PRIMARY DEFINITION]" if bool(row["is_frozen_primary_definition"]) else ""
        lines.append(
            f"{row['definition_family']} / {row['model']} / "
            f"threshold {float(row['severity_threshold_delta_uno_c']):.2f}: "
            f"rate={float(row['equal_scenario_rate']):.6f}; "
            f"change_vs_primary={float(row['difference_vs_frozen_primary_definition']):+.6f}"
            f"{marker}"
        )

    lines += [
        "",
        "Technical provenance",
        "--------------------",
        "No model was refit.",
        "No human outcome was read.",
        "No schema discovery was performed.",
        "All retained NPZ identities/model indices come from the PASS 05h3c contract.",
        "The -0.02/-0.05 rows must exactly reproduce the frozen-original rates in 05h3d.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    print("=" * 118)
    print("Paper 6 - post-HOLD threshold-definition sensitivity")
    print("=" * 118)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {ROOT}")
    print(f"Output directory: {OUT_DIR}")
    print()
    print("Execution contract:")
    print("  05h0 neighboring-definition grid changed: NO")
    print("  frozen primary -0.02/-0.05 definitions changed: NO")
    print("  reopened pass/fail decision: NO")
    print("  schema discovery: NO")
    print("  model fitting / retuning: NO")
    print("  human outcomes read: NO")
    print("  retained frozen delta arrays read: YES")
    print("  aggregation: equal scenario weight across frozen 180-scenario benchmark")
    print()

    if OUT_DIR.exists():
        raise RuntimeError(
            f"05h4 final output already exists; refusing overwrite: {OUT_DIR}"
        )
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir(parents=True, exist_ok=False)

    try:
        verify_05h0_threshold_contract()
        h3c, h3d = verify_frozen_chain()
        verify_h3c_semantics(h3c)

        print("Frozen contract chain: PASS")
        print(f"  05h0 SHA256: {sha256_file(H0_JSON)}")
        print(f"  05h3c contract SHA256: {sha256_file(H3C_CONTRACT)}")
        print(f"  05h3d result SHA256: {sha256_file(H3D_JSON)}")
        print()

        design = load_and_verify_design(h3c)
        inventory = load_and_verify_inventory(h3c, design)

        print("Frozen retained-array provenance: PASS")
        print("  scenarios: 180/180")
        print("  retained NPZ hashes reverified: 180/180")
        print("  model-axis A2/A3 indices inherited from 05h3c: YES")
        print()

        summary, scenario_rates = compute_sensitivity(
            h3c,
            design,
            inventory,
            h3d,
        )
        summary = add_descriptive_reference_line_indicators(summary)

        print("Primary-definition replay against 05h3d: PASS")
        print("  A2/A3 negative-transfer rates at -0.02: reproduced")
        print("  A2/A3 catastrophic-transfer rates at -0.05: reproduced")
        print()

        print("-" * 118)
        print("05h4 THRESHOLD-DEFINITION SENSITIVITY")
        print("-" * 118)
        display = summary[
            [
                "definition_family",
                "model",
                "severity_threshold_delta_uno_c",
                "is_frozen_primary_definition",
                "equal_scenario_rate",
                "difference_vs_frozen_primary_definition",
            ]
        ].sort_values(
            ["definition_family", "model", "severity_threshold_delta_uno_c"],
            ascending=[True, True, False],
        )
        print(
            display.to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}",
            )
        )
        print()

        print("Interpretation guardrails:")
        print("  -0.02 / -0.05 remain the only frozen primary definitions: YES")
        print("  neighboring definitions are descriptive sensitivities only: YES")
        print("  reference rate lines create a new PASS/FAIL rule: NO")
        print("  HOLD_NO_SAFE_SELECTABLE_ARCHITECTURE changed: NO")
        print()

        table_path = WORK_DIR / "Supplementary_Table_threshold_definition_sensitivity.csv"
        scenario_path = WORK_DIR / "Supplementary_Data_threshold_scenario_rates.csv"
        checks_path = WORK_DIR / "primary_definition_replay_checks.csv"
        json_path = WORK_DIR / "threshold_definition_sensitivity.json"
        readme_path = WORK_DIR / "README_for_supplement.txt"

        summary.to_csv(
            table_path,
            index=False,
            float_format="%.10g",
        )
        scenario_rates.to_csv(
            scenario_path,
            index=False,
            float_format="%.10g",
        )

        # Re-create compact replay-check table for export.
        expected_primary = primary_rates_from_h3d(h3d)
        check_rows = []
        for model in PRIMARY_MODELS:
            for family, threshold, expected_key in [
                ("negative_transfer", PRIMARY_NEGATIVE_THRESHOLD, "negative"),
                ("catastrophic_transfer", PRIMARY_CATASTROPHIC_THRESHOLD, "catastrophic"),
            ]:
                observed = float(
                    summary[
                        (summary["definition_family"] == family)
                        & (summary["model"] == model)
                        & (summary["severity_threshold_delta_uno_c"] == threshold)
                    ].iloc[0]["equal_scenario_rate"]
                )
                expected = expected_primary[(model, expected_key)]
                diff = abs(observed - expected)
                check_rows.append(
                    {
                        "model": model,
                        "definition_family": family,
                        "threshold": threshold,
                        "observed": observed,
                        "expected_05h3d": expected,
                        "absolute_difference": diff,
                        "tolerance": REPLAY_TOL,
                        "pass": diff <= REPLAY_TOL,
                    }
                )
        pd.DataFrame(check_rows).to_csv(
            checks_path,
            index=False,
            float_format="%.12g",
        )

        result = {
            "script_version": SCRIPT_VERSION,
            "status": "PASS_POSTHOLD_THRESHOLD_DEFINITION_SENSITIVITY",
            "analysis_role": (
                "POST-HOLD SENSITIVITY OF THE ESTIMATED SAFETY PROFILE; "
                "NOT A REOPENED PASS/FAIL DECISION"
            ),
            "frozen_primary_decision": "HOLD_NO_SAFE_SELECTABLE_ARCHITECTURE",
            "primary_decision_changed": False,
            "primary_models": PRIMARY_MODELS,
            "frozen_primary_definitions": {
                "negative_transfer_delta_uno_c": PRIMARY_NEGATIVE_THRESHOLD,
                "catastrophic_transfer_delta_uno_c": PRIMARY_CATASTROPHIC_THRESHOLD,
            },
            "severity_grids": {
                "negative_transfer": NEGATIVE_GRID,
                "catastrophic_transfer": CATASTROPHIC_GRID,
            },
            "reference_rate_lines": {
                "negative_transfer": NEGATIVE_REFERENCE_RATE_LINES,
                "catastrophic_transfer": CATASTROPHIC_REFERENCE_RATE_LINES,
            },
            "aggregation": "equal scenario weight across all 180 frozen scenarios",
            "05h0_sha256": sha256_file(H0_JSON),
            "05h3c_contract_sha256": sha256_file(H3C_CONTRACT),
            "05h3d_result_sha256": sha256_file(H3D_JSON),
            "results": summary.to_dict(orient="records"),
            "model_fitting": False,
            "human_outcomes_read": False,
            "thresholds_changed": False,
        }
        write_json(json_path, result)

        readme_path.write_text(
            build_readme(summary),
            encoding="utf-8",
            newline="\n",
        )

        outputs = sorted(
            p for p in WORK_DIR.iterdir()
            if p.is_file() and p.name != "freeze_manifest.json"
        )
        manifest = {
            "script_version": SCRIPT_VERSION,
            "script_relative_path": str(
                Path(__file__).resolve().relative_to(ROOT)
            ),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "status": "PASS_POSTHOLD_THRESHOLD_DEFINITION_SENSITIVITY",
            "inputs": {
                "05h0_contract": {
                    "path": str(H0_JSON.relative_to(ROOT)),
                    "sha256": sha256_file(H0_JSON),
                },
                "05h3c_contract": {
                    "path": str(H3C_CONTRACT.relative_to(ROOT)),
                    "sha256": sha256_file(H3C_CONTRACT),
                },
                "05h3c_npz_inventory": {
                    "path": str(H3C_NPZ_INVENTORY.relative_to(ROOT)),
                    "sha256": sha256_file(H3C_NPZ_INVENTORY),
                },
                "05h3d_result": {
                    "path": str(H3D_JSON.relative_to(ROOT)),
                    "sha256": sha256_file(H3D_JSON),
                },
            },
            "outputs": [
                {
                    "file": p.name,
                    "sha256": sha256_file(p),
                    "bytes": p.stat().st_size,
                }
                for p in outputs
            ],
        }
        write_json(WORK_DIR / "freeze_manifest.json", manifest)

        for item in manifest["outputs"]:
            p = WORK_DIR / item["file"]
            if sha256_file(p) != item["sha256"]:
                raise RuntimeError(f"05h4 output changed before finalization: {p.name}")

        os.replace(WORK_DIR, OUT_DIR)

        print("=" * 118)
        print("05h4 post-HOLD threshold-definition sensitivity: PASS")
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
        print("05h4 post-HOLD threshold-definition sensitivity: FAIL")
        print(f"{type(exc).__name__}: {exc}")
        print("=" * 118)
        raise
