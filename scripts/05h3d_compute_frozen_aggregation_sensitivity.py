#!/usr/bin/env python
"""
Paper 6 - compute frozen aggregation-sensitivity estimands after 05h3c preflight.

This script performs NO schema discovery and NO label interpretation.

All technical identities are read from the already-PASS frozen 05h3c contract:
  - exact source hashes
  - exact column bindings
  - exact 36 CORE scenario IDs
  - exact 144 STRESS scenario IDs
  - exact scenario -> R0...R5 mapping
  - exact retained NPZ paths/hashes
  - validated model-axis order and A2/A3 indices
  - frozen -0.02 / -0.05 flag identities

Scientific estimands are exactly those frozen in 05h0:
  1) frozen_original: all 180 scenarios, equal scenario weight
  2) core_only: exact 36 CORE scenarios, equal scenario weight
  3) regime_balanced: all 180 scenarios, equal within-regime scenario weight,
     then equal 1/6 weight for each R0...R5 regime

Reported metrics:
  - mean delta Uno-C vs B0
  - negative-transfer rate at delta C <= -0.02
  - catastrophic-transfer rate at delta C <= -0.05

Uncertainty:
  4,000 deterministic bootstrap draws resampling retained replicates WITHIN
  each fixed scenario. Scenarios themselves are never resampled.

Expected placement:
    scripts/05h3d_compute_frozen_aggregation_sensitivity.py

Run:
    python scripts\05h3d_compute_frozen_aggregation_sensitivity.py
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_VERSION = "05h3d-compute-frozen-aggregation-sensitivity-v1-no-discovery-no-cli"

PRIMARY_MODELS = ["A2", "A3"]
ESTIMANDS = ["frozen_original", "core_only", "regime_balanced"]

NEGATIVE_THRESHOLD = -0.02
CATASTROPHIC_THRESHOLD = -0.05

# This numerical choice was already fixed in the earlier failed 05h3 implementation
# before any 05h3 sensitivity estimate was observed.
MC_BOOTSTRAP_DRAWS = 4000
MC_QLOW = 0.025
MC_QHIGH = 0.975

POINT_REPLAY_TOL = 5e-10
NPZ_REPLAY_TOL = 1e-7


def project_root() -> Path:
    p = Path(__file__).resolve()
    if p.parent.name.lower() != "scripts":
        raise RuntimeError(
            "Place 05h3d_compute_frozen_aggregation_sensitivity.py "
            "in the repository scripts/ directory."
        )
    return p.parent.parent


ROOT = project_root()

H3C_DIR = ROOT / "method_contract" / "05h3c_aggregation_input_contract"
H3C_CONTRACT = H3C_DIR / "aggregation_input_contract.json"
H3C_MANIFEST = H3C_DIR / "freeze_manifest.json"
H3C_DESIGN = H3C_DIR / "resolved_frozen_scenario_design.tsv"
H3C_NPZ_INVENTORY = H3C_DIR / "retained_npz_source_inventory.csv"

OUT_DIR = ROOT / "method_contract" / "05h3d_aggregation_sensitivity"
WORK_DIR = ROOT / "method_contract" / ".05h3d_aggregation_sensitivity_work"


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


def load_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return obj


def verify_05h3c_outputs() -> tuple[dict[str, Any], dict[str, Any]]:
    for p in [H3C_CONTRACT, H3C_MANIFEST, H3C_DESIGN, H3C_NPZ_INVENTORY]:
        if not p.exists():
            raise FileNotFoundError(f"Required 05h3c artifact missing: {p}")

    manifest = load_json(H3C_MANIFEST)
    if manifest.get("status") != "PASS_READY_FOR_05H3D_AGGREGATION_SENSITIVITY":
        raise RuntimeError(
            f"05h3c manifest status is not PASS: {manifest.get('status')!r}"
        )

    output_hashes = {
        str(row["file"]): str(row["sha256"])
        for row in manifest.get("outputs", [])
        if isinstance(row, dict) and "file" in row and "sha256" in row
    }
    for p in [H3C_CONTRACT, H3C_DESIGN, H3C_NPZ_INVENTORY]:
        expected = output_hashes.get(p.name)
        if expected is None:
            raise RuntimeError(
                f"05h3c freeze manifest does not lock required output {p.name}."
            )
        observed = sha256_file(p)
        if observed != expected:
            raise RuntimeError(
                f"05h3c output hash changed for {p.name}: "
                f"observed={observed}, expected={expected}"
            )

    contract = load_json(H3C_CONTRACT)
    if contract.get("status") != "PASS_READY_FOR_05H3D_AGGREGATION_SENSITIVITY":
        raise RuntimeError(
            f"05h3c contract status is not PASS: {contract.get('status')!r}"
        )
    if bool(contract.get("scientific_estimate_computed")):
        raise RuntimeError("05h3c unexpectedly claims a scientific estimate was computed.")
    if bool(contract.get("frozen_primary_decision_changed")):
        raise RuntimeError("05h3c unexpectedly changed the frozen primary decision.")

    return contract, manifest


def verify_locked_sources(contract: dict[str, Any]) -> dict[str, Path]:
    sources = contract.get("sources")
    if not isinstance(sources, dict):
        raise RuntimeError("05h3c contract lacks sources.")

    resolved: dict[str, Path] = {}
    for role, meta in sources.items():
        if not isinstance(meta, dict):
            raise RuntimeError(f"Invalid source metadata for {role}.")
        rel = meta.get("path")
        expected_hash = meta.get("sha256")
        if not isinstance(rel, str) or not isinstance(expected_hash, str):
            raise RuntimeError(f"Incomplete source lock for {role}.")
        path = ROOT / rel
        if not path.exists():
            raise FileNotFoundError(f"Locked source missing for {role}: {path}")
        observed_hash = sha256_file(path)
        if observed_hash != expected_hash:
            raise RuntimeError(
                f"Locked source changed for {role}: "
                f"observed={observed_hash}, expected={expected_hash}"
            )
        resolved[role] = path

    required_roles = {
        "05h0_contract",
        "05h2b_summary",
        "05h2b_manifest",
        "05d_scenario_summary",
        "05d_metric_contract",
        "05d_metric_manifest",
    }
    if not required_roles.issubset(resolved):
        raise RuntimeError(
            f"05h3c source lock lacks roles: {sorted(required_roles - set(resolved))}"
        )

    return resolved


def verify_contract_semantics(contract: dict[str, Any]) -> None:
    bindings = contract.get("exact_column_bindings", {})
    expected_bindings = {
        "scenario_id": "scenario_id",
        "descriptive_family": "family",
        "transfer_regime": "transfer_regime",
        "replicate_count": "replicates",
        "model": "model",
        "mean_delta_uno_c": "mean_delta_c_vs_B0",
        "negative_transfer_rate": "negative_transfer_rate",
        "catastrophic_transfer_rate": "catastrophic_negative_transfer_rate",
    }
    if bindings != expected_bindings:
        raise RuntimeError(
            "05h3c exact column bindings differ from the validated expected bindings."
        )

    thresholds = contract.get("frozen_thresholds", {})
    if float(thresholds.get("negative_transfer_delta_uno_c")) != NEGATIVE_THRESHOLD:
        raise RuntimeError("05h3c negative-transfer threshold changed.")
    if float(thresholds.get("catastrophic_transfer_delta_uno_c")) != CATASTROPHIC_THRESHOLD:
        raise RuntimeError("05h3c catastrophic-transfer threshold changed.")
    if thresholds.get("retained_flags_match_thresholds_exactly") is not True:
        raise RuntimeError("05h3c did not verify exact retained threshold flags.")

    schema = contract.get("retained_metric_schema", {})
    order = schema.get("model_axis_order")
    if not isinstance(order, list) or PRIMARY_MODELS[0] not in order or PRIMARY_MODELS[1] not in order:
        raise RuntimeError("05h3c retained model order does not contain A2/A3.")
    if int(schema.get("A2_index")) != order.index("A2"):
        raise RuntimeError("05h3c A2 model index is internally inconsistent.")
    if int(schema.get("A3_index")) != order.index("A3"):
        raise RuntimeError("05h3c A3 model index is internally inconsistent.")

    scenario_rule = contract.get("scenario_class_rule", {})
    core_ids = scenario_rule.get("core_scenario_ids")
    stress_ids = scenario_rule.get("stress_scenario_ids")
    if not isinstance(core_ids, list) or len(core_ids) != 36 or len(set(core_ids)) != 36:
        raise RuntimeError("05h3c core scenario lock is not exactly 36 unique IDs.")
    if not isinstance(stress_ids, list) or len(stress_ids) != 144 or len(set(stress_ids)) != 144:
        raise RuntimeError("05h3c stress scenario lock is not exactly 144 unique IDs.")
    if set(core_ids) & set(stress_ids):
        raise RuntimeError("05h3c CORE and STRESS scenario locks overlap.")

    regime_rule = contract.get("regime_rule", {})
    counts = regime_rule.get("counts")
    expected_counts = {
        "R0": 6, "R1": 6, "R2": 78, "R3": 6, "R4": 6, "R5": 78
    }
    if counts != expected_counts:
        raise RuntimeError(
            f"05h3c regime counts changed: observed={counts}, expected={expected_counts}"
        )

    reps = contract.get("replicates", {})
    if int(reps.get("n_scenarios")) != 180:
        raise RuntimeError("05h3c scenario count changed from 180.")
    if int(reps.get("total_replicates")) != 21600:
        raise RuntimeError("05h3c replicate total changed from 21,600.")
    if int(reps.get("retained_npz_count")) != 180:
        raise RuntimeError("05h3c retained NPZ count changed from 180.")


def load_locked_design(contract: dict[str, Any]) -> pd.DataFrame:
    design = pd.read_csv(H3C_DESIGN, sep="\t", low_memory=False)

    required = [
        "scenario_id",
        "scenario_class",
        "family",
        "transfer_regime",
        "regime",
        "replicates",
    ]
    missing = [c for c in required if c not in design.columns]
    if missing:
        raise RuntimeError(f"Frozen 05h3c design table missing columns: {missing}")

    if len(design) != 180 or design["scenario_id"].nunique() != 180:
        raise RuntimeError("Frozen 05h3c design is not exactly 180 unique scenarios.")

    core_ids = set(contract["scenario_class_rule"]["core_scenario_ids"])
    stress_ids = set(contract["scenario_class_rule"]["stress_scenario_ids"])

    observed_core = set(
        design.loc[design["scenario_class"] == "CORE", "scenario_id"].astype(str)
    )
    observed_stress = set(
        design.loc[design["scenario_class"] == "STRESS", "scenario_id"].astype(str)
    )
    if observed_core != core_ids:
        raise RuntimeError("05h3c design CORE membership differs from frozen contract.")
    if observed_stress != stress_ids:
        raise RuntimeError("05h3c design STRESS membership differs from frozen contract.")

    expected_regime_map = {
        str(row["scenario_id"]): str(row["regime"])
        for row in contract["regime_rule"]["scenario_to_regime"]
    }
    observed_regime_map = dict(
        zip(design["scenario_id"].astype(str), design["regime"].astype(str))
    )
    if observed_regime_map != expected_regime_map:
        raise RuntimeError("05h3c scenario->regime mapping differs from frozen contract.")

    design["replicates"] = pd.to_numeric(design["replicates"], errors="raise").astype(int)
    if int(design["replicates"].sum()) != 21600:
        raise RuntimeError("05h3c design replicate total no longer equals 21,600.")

    return design.sort_values("scenario_id").reset_index(drop=True)


def load_primary_scenario_metrics(
    contract: dict[str, Any],
    sources: dict[str, Path],
    design: pd.DataFrame,
) -> pd.DataFrame:
    bindings = contract["exact_column_bindings"]
    df = pd.read_csv(
        sources["05d_scenario_summary"],
        sep="\t",
        low_memory=False,
    )

    required_source_columns = sorted(set(bindings.values()))
    missing = [c for c in required_source_columns if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Locked 05d scenario summary lost required columns: {missing}"
        )

    out = df[
        [
            bindings["scenario_id"],
            bindings["model"],
            bindings["mean_delta_uno_c"],
            bindings["negative_transfer_rate"],
            bindings["catastrophic_transfer_rate"],
        ]
    ].copy()

    out.columns = [
        "scenario_id",
        "model",
        "mean_delta_uno_c",
        "negative_transfer_rate_at_deltaC_le_-0.02",
        "catastrophic_transfer_rate_at_deltaC_le_-0.05",
    ]

    out["scenario_id"] = out["scenario_id"].astype(str)
    out["model"] = out["model"].astype(str)
    out = out[out["model"].isin(PRIMARY_MODELS)].copy()

    if len(out) != 360:
        raise RuntimeError(f"Expected 360 A2/A3 scenario rows; observed {len(out)}.")
    if out.duplicated(["scenario_id", "model"]).any():
        raise RuntimeError("A2/A3 scenario metrics are not unique.")

    for model in PRIMARY_MODELS:
        if int((out["model"] == model).sum()) != 180:
            raise RuntimeError(f"{model} does not have exactly 180 scenario rows.")

    expected_scenarios = set(design["scenario_id"].astype(str))
    if set(out["scenario_id"]) != expected_scenarios:
        raise RuntimeError(
            "A2/A3 metric scenario set differs from frozen 05h3c design."
        )

    for col in [
        "mean_delta_uno_c",
        "negative_transfer_rate_at_deltaC_le_-0.02",
        "catastrophic_transfer_rate_at_deltaC_le_-0.05",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        if not np.isfinite(out[col].to_numpy(dtype=float)).all():
            raise RuntimeError(f"Non-finite frozen A2/A3 metric: {col}")

    out = out.merge(
        design[["scenario_id", "scenario_class", "regime"]],
        on="scenario_id",
        how="left",
        validate="many_to_one",
    )
    if out[["scenario_class", "regime"]].isna().any().any():
        raise RuntimeError("Failed to attach frozen 05h3c design identities.")

    return out.sort_values(["scenario_id", "model"]).reset_index(drop=True)


def build_estimand_weights(
    contract: dict[str, Any],
    design: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    scenario_ids = list(design["scenario_id"].astype(str))
    core_ids = set(contract["scenario_class_rule"]["core_scenario_ids"])

    weights: dict[str, dict[str, float]] = {}

    weights["frozen_original"] = {
        sid: 1.0 / 180.0 for sid in scenario_ids
    }

    weights["core_only"] = {
        sid: (1.0 / 36.0 if sid in core_ids else 0.0)
        for sid in scenario_ids
    }

    rb: dict[str, float] = {}
    for regime in ["R0", "R1", "R2", "R3", "R4", "R5"]:
        ids = list(
            design.loc[design["regime"] == regime, "scenario_id"].astype(str)
        )
        if not ids:
            raise RuntimeError(f"No frozen scenarios for {regime}.")
        per_scenario_weight = (1.0 / 6.0) / len(ids)
        for sid in ids:
            rb[sid] = per_scenario_weight
    weights["regime_balanced"] = rb

    for name, mapping in weights.items():
        if set(mapping) != set(scenario_ids):
            raise RuntimeError(f"{name} weight map does not cover all frozen scenarios.")
        total = float(sum(mapping.values()))
        if abs(total - 1.0) > 1e-12:
            raise RuntimeError(f"{name} weights sum to {total}, not 1.")

    positive_counts = {
        name: sum(1 for w in mapping.values() if w > 0.0)
        for name, mapping in weights.items()
    }
    if positive_counts != {
        "frozen_original": 180,
        "core_only": 36,
        "regime_balanced": 180,
    }:
        raise RuntimeError(
            f"Unexpected positive-weight scenario counts: {positive_counts}"
        )

    return weights


def compute_point_estimates(
    metrics: pd.DataFrame,
    weights: dict[str, dict[str, float]],
) -> pd.DataFrame:
    metric_cols = [
        "mean_delta_uno_c",
        "negative_transfer_rate_at_deltaC_le_-0.02",
        "catastrophic_transfer_rate_at_deltaC_le_-0.05",
    ]

    rows: list[dict[str, Any]] = []
    for estimand in ESTIMANDS:
        wmap = weights[estimand]
        n_positive = sum(1 for w in wmap.values() if w > 0.0)
        for model in PRIMARY_MODELS:
            z = metrics[metrics["model"] == model].copy()
            z["_weight"] = z["scenario_id"].map(wmap).astype(float)

            row: dict[str, Any] = {
                "estimand": estimand,
                "model": model,
                "n_scenarios_with_positive_weight": n_positive,
            }
            for metric in metric_cols:
                row[metric] = float(np.sum(z["_weight"] * z[metric]))
            rows.append(row)

    return pd.DataFrame(rows)


def compute_regime_components(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for regime in ["R0", "R1", "R2", "R3", "R4", "R5"]:
        for model in PRIMARY_MODELS:
            z = metrics[
                (metrics["regime"] == regime)
                & (metrics["model"] == model)
            ]
            rows.append(
                {
                    "regime": regime,
                    "model": model,
                    "n_scenarios": int(len(z)),
                    "mean_delta_uno_c": float(z["mean_delta_uno_c"].mean()),
                    "negative_transfer_rate_at_deltaC_le_-0.02": float(
                        z["negative_transfer_rate_at_deltaC_le_-0.02"].mean()
                    ),
                    "catastrophic_transfer_rate_at_deltaC_le_-0.05": float(
                        z["catastrophic_transfer_rate_at_deltaC_le_-0.05"].mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def verify_frozen_original_against_05h2b(
    points: pd.DataFrame,
    sources: dict[str, Path],
) -> list[dict[str, Any]]:
    h2b = load_json(sources["05h2b_summary"])
    model_summary = h2b.get("model_summary")
    if not isinstance(model_summary, list):
        raise RuntimeError("05h2b summary lacks model_summary.")

    lookup = {
        str(row["model"]): row
        for row in model_summary
        if isinstance(row, dict) and "model" in row
    }

    original = points[points["estimand"] == "frozen_original"].set_index("model")

    pairs = [
        ("mean_delta_uno_c", "mean_delta_uno_c_vs_B0_if_defined"),
        (
            "negative_transfer_rate_at_deltaC_le_-0.02",
            "negative_transfer_rate_at_deltaC_le_-0.02",
        ),
        (
            "catastrophic_transfer_rate_at_deltaC_le_-0.05",
            "catastrophic_transfer_rate_at_deltaC_le_-0.05",
        ),
    ]

    checks: list[dict[str, Any]] = []
    for model in PRIMARY_MODELS:
        if model not in lookup:
            raise RuntimeError(f"05h2b model summary lacks {model}.")
        for point_col, h2b_col in pairs:
            observed = float(original.loc[model, point_col])
            expected = float(lookup[model][h2b_col])
            diff = abs(observed - expected)
            passed = diff <= POINT_REPLAY_TOL
            checks.append(
                {
                    "check": "frozen_original_reproduces_05h2b",
                    "model": model,
                    "metric": point_col,
                    "observed": observed,
                    "expected": expected,
                    "absolute_difference": diff,
                    "tolerance": POINT_REPLAY_TOL,
                    "pass": passed,
                }
            )

    failed = [row for row in checks if not row["pass"]]
    if failed:
        detail = "\n".join(
            f"  {r['model']} {r['metric']}: "
            f"{r['observed']:.12g} vs {r['expected']:.12g}"
            for r in failed
        )
        raise RuntimeError(
            "05h3d frozen_original does not reproduce accepted 05h2b:\n" + detail
        )

    return checks


def load_locked_npz_inventory(
    contract: dict[str, Any],
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
        raise RuntimeError(f"05h3c NPZ inventory missing columns: {missing}")

    if len(inv) != 180 or inv["scenario_id"].nunique() != 180:
        raise RuntimeError("05h3c NPZ inventory is not exactly 180 unique scenarios.")

    expected_scenarios = set(design["scenario_id"].astype(str))
    if set(inv["scenario_id"].astype(str)) != expected_scenarios:
        raise RuntimeError("05h3c NPZ inventory scenario set changed.")

    inv = inv.merge(
        design[["scenario_id", "scenario_class", "regime", "replicates"]],
        on="scenario_id",
        how="left",
        suffixes=("_inventory", "_design"),
        validate="one_to_one",
    )

    for col in ["scenario_class", "regime"]:
        if not (
            inv[f"{col}_inventory"].astype(str)
            == inv[f"{col}_design"].astype(str)
        ).all():
            raise RuntimeError(f"05h3c NPZ inventory {col} differs from frozen design.")

    if not np.array_equal(
        pd.to_numeric(inv["replicates_inventory"], errors="raise").astype(int).to_numpy(),
        pd.to_numeric(inv["replicates_design"], errors="raise").astype(int).to_numpy(),
    ):
        raise RuntimeError("05h3c NPZ replicate counts differ from frozen design.")

    # Re-verify every retained file hash BEFORE computation.
    for row in inv.itertuples(index=False):
        path = ROOT / str(row.npz_relative_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Frozen retained NPZ missing for {row.scenario_id}: {path}"
            )
        observed_hash = sha256_file(path)
        if observed_hash != str(row.npz_sha256):
            raise RuntimeError(
                f"Frozen retained NPZ changed for {row.scenario_id}: "
                f"observed={observed_hash}, expected={row.npz_sha256}"
            )
        if path.stat().st_size != int(row.npz_bytes):
            raise RuntimeError(
                f"Frozen retained NPZ byte size changed for {row.scenario_id}."
            )

    return inv


def replay_npz_before_bootstrap(
    contract: dict[str, Any],
    metrics: pd.DataFrame,
    inventory: pd.DataFrame,
) -> dict[str, dict[str, np.ndarray]]:
    schema = contract["retained_metric_schema"]
    model_order = list(schema["model_axis_order"])
    a2_idx = int(schema["A2_index"])
    a3_idx = int(schema["A3_index"])

    if model_order[a2_idx] != "A2" or model_order[a3_idx] != "A3":
        raise RuntimeError("05h3c frozen A2/A3 model indices are inconsistent.")

    lookup = metrics.set_index(["scenario_id", "model"])
    data_by_scenario: dict[str, dict[str, np.ndarray]] = {}

    max_delta_diff = 0.0
    max_nt_diff = 0.0
    max_cat_diff = 0.0

    for row in inventory.itertuples(index=False):
        sid = str(row.scenario_id)
        n = int(row.replicates_design)
        path = ROOT / str(row.npz_relative_path)

        with np.load(path, allow_pickle=False) as npz:
            required = [
                "replicate_seed",
                "delta_c_vs_B0",
                "negative_transfer",
                "catastrophic_negative_transfer",
            ]
            missing = [key for key in required if key not in npz.files]
            if missing:
                raise RuntimeError(f"{sid}: retained NPZ missing keys {missing}")

            seeds = np.asarray(npz["replicate_seed"])
            delta_all = np.asarray(npz["delta_c_vs_B0"])
            neg_all = np.asarray(npz["negative_transfer"])
            cat_all = np.asarray(npz["catastrophic_negative_transfer"])

        expected_shape = (n, len(model_order))
        for name, arr in [
            ("delta_c_vs_B0", delta_all),
            ("negative_transfer", neg_all),
            ("catastrophic_negative_transfer", cat_all),
        ]:
            if arr.shape != expected_shape:
                raise RuntimeError(
                    f"{sid}: {name} shape={arr.shape}, expected={expected_shape}"
                )
        if seeds.shape != (n,):
            raise RuntimeError(
                f"{sid}: replicate_seed shape={seeds.shape}, expected={(n,)}"
            )

        idx = [a2_idx, a3_idx]
        delta = delta_all[:, idx].astype(np.float64, copy=False)
        neg = neg_all[:, idx].astype(np.float64, copy=False)
        cat = cat_all[:, idx].astype(np.float64, copy=False)

        # Exact threshold identity re-check.
        if not np.array_equal(
            neg_all.astype(np.uint8),
            (delta_all <= NEGATIVE_THRESHOLD).astype(np.uint8),
        ):
            raise RuntimeError(f"{sid}: negative-transfer flags no longer match -0.02.")
        if not np.array_equal(
            cat_all.astype(np.uint8),
            (delta_all <= CATASTROPHIC_THRESHOLD).astype(np.uint8),
        ):
            raise RuntimeError(f"{sid}: catastrophic flags no longer match -0.05.")

        for j, model in enumerate(PRIMARY_MODELS):
            frozen = lookup.loc[(sid, model)]

            observed_delta = float(np.nanmean(delta[:, j]))
            expected_delta = float(frozen["mean_delta_uno_c"])
            delta_diff = abs(observed_delta - expected_delta)
            max_delta_diff = max(max_delta_diff, delta_diff)
            delta_ok = (
                delta_diff <= NPZ_REPLAY_TOL
                or np.float32(observed_delta).tobytes()
                == np.float32(expected_delta).tobytes()
            )
            if not delta_ok:
                raise RuntimeError(
                    f"{sid}/{model}: delta replay mismatch before bootstrap: "
                    f"{observed_delta:.12g} vs {expected_delta:.12g}"
                )

            observed_nt = float(np.mean(neg[:, j]))
            expected_nt = float(
                frozen["negative_transfer_rate_at_deltaC_le_-0.02"]
            )
            nt_diff = abs(observed_nt - expected_nt)
            max_nt_diff = max(max_nt_diff, nt_diff)
            if nt_diff > 1e-12:
                raise RuntimeError(
                    f"{sid}/{model}: NT-rate replay mismatch before bootstrap."
                )

            observed_cat = float(np.mean(cat[:, j]))
            expected_cat = float(
                frozen["catastrophic_transfer_rate_at_deltaC_le_-0.05"]
            )
            cat_diff = abs(observed_cat - expected_cat)
            max_cat_diff = max(max_cat_diff, cat_diff)
            if cat_diff > 1e-12:
                raise RuntimeError(
                    f"{sid}/{model}: catastrophic-rate replay mismatch before bootstrap."
                )

        data_by_scenario[sid] = {
            "delta": delta,
            "negative": neg,
            "catastrophic": cat,
        }

    print("Pre-bootstrap retained-array replay: PASS")
    print(f"  180/180 retained NPZ hashes reverified: YES")
    print(f"  max A2/A3 mean-delta replay difference: {max_delta_diff:.3e}")
    print(f"  max A2/A3 NT-rate replay difference: {max_nt_diff:.3e}")
    print(f"  max A2/A3 catastrophic-rate replay difference: {max_cat_diff:.3e}")
    print("  threshold flags exact at -0.02/-0.05: YES")
    print()

    return data_by_scenario


def deterministic_mc_seed(contract: dict[str, Any]) -> int:
    # The seed comes only from the frozen 05h3c contract bytes, not from any
    # sensitivity result. Keep it inside the signed 32-bit range for portability.
    digest = sha256_file(H3C_CONTRACT)
    return int(digest[:8], 16) % (2**31 - 1)


def bootstrap_fixed_scenarios(
    design: pd.DataFrame,
    data_by_scenario: dict[str, dict[str, np.ndarray]],
    weights: dict[str, dict[str, float]],
    seed: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)

    # [draw, model] accumulator for each estimand x metric.
    metric_keys = [
        "mean_delta_uno_c",
        "negative_transfer_rate_at_deltaC_le_-0.02",
        "catastrophic_transfer_rate_at_deltaC_le_-0.05",
    ]
    accumulator = {
        estimand: {
            metric: np.zeros(
                (MC_BOOTSTRAP_DRAWS, len(PRIMARY_MODELS)),
                dtype=np.float64,
            )
            for metric in metric_keys
        }
        for estimand in ESTIMANDS
    }

    for row in design.sort_values("scenario_id").itertuples(index=False):
        sid = str(row.scenario_id)
        arrays = data_by_scenario[sid]
        n = arrays["delta"].shape[0]

        # Shared replicate resample for A2/A3 and all metrics preserves pairing.
        draw_idx = rng.integers(
            0,
            n,
            size=(MC_BOOTSTRAP_DRAWS, n),
            dtype=np.int32,
        )

        with np.errstate(invalid="ignore"):
            delta_boot = np.nanmean(arrays["delta"][draw_idx, :], axis=1)
        if not np.isfinite(delta_boot).all():
            raise RuntimeError(
                f"{sid}: non-finite bootstrap scenario mean delta encountered."
            )

        nt_boot = arrays["negative"][draw_idx, :].mean(axis=1)
        cat_boot = arrays["catastrophic"][draw_idx, :].mean(axis=1)

        scenario_boot = {
            "mean_delta_uno_c": delta_boot,
            "negative_transfer_rate_at_deltaC_le_-0.02": nt_boot,
            "catastrophic_transfer_rate_at_deltaC_le_-0.05": cat_boot,
        }

        for estimand in ESTIMANDS:
            weight = float(weights[estimand][sid])
            if weight == 0.0:
                continue
            for metric, values in scenario_boot.items():
                accumulator[estimand][metric] += weight * values

    rows: list[dict[str, Any]] = []
    saved_arrays: dict[str, np.ndarray] = {}

    for estimand in ESTIMANDS:
        for metric in metric_keys:
            arr = accumulator[estimand][metric]
            saved_arrays[f"{estimand}__{metric}"] = arr
            for j, model in enumerate(PRIMARY_MODELS):
                values = arr[:, j]
                rows.append(
                    {
                        "estimand": estimand,
                        "model": model,
                        "metric": metric,
                        "mc_bootstrap_draws": MC_BOOTSTRAP_DRAWS,
                        "mc_seed": seed,
                        "mc_mean": float(np.mean(values)),
                        "mc_sd": float(np.std(values, ddof=1)),
                        "mc_p025": float(np.quantile(values, MC_QLOW)),
                        "mc_median": float(np.quantile(values, 0.5)),
                        "mc_p975": float(np.quantile(values, MC_QHIGH)),
                        "resampling_unit": (
                            "retained replicate within each fixed frozen scenario"
                        ),
                        "scenario_resampling": False,
                    }
                )

    return pd.DataFrame(rows), saved_arrays


def combine_points_with_mc(
    points: pd.DataFrame,
    mc: pd.DataFrame,
) -> pd.DataFrame:
    metric_cols = [
        "mean_delta_uno_c",
        "negative_transfer_rate_at_deltaC_le_-0.02",
        "catastrophic_transfer_rate_at_deltaC_le_-0.05",
    ]

    out = points.copy()
    for metric in metric_cols:
        part = mc[mc["metric"] == metric][
            [
                "estimand",
                "model",
                "mc_sd",
                "mc_p025",
                "mc_p975",
            ]
        ].copy()
        part = part.rename(
            columns={
                "mc_sd": f"{metric}__mc_sd",
                "mc_p025": f"{metric}__mc_p025",
                "mc_p975": f"{metric}__mc_p975",
            }
        )
        out = out.merge(
            part,
            on=["estimand", "model"],
            how="left",
            validate="one_to_one",
        )

    # Add deterministic changes relative to the immutable frozen-original estimand.
    original = (
        out[out["estimand"] == "frozen_original"]
        .set_index("model")
    )
    for metric in metric_cols:
        out[f"{metric}__difference_vs_frozen_original"] = [
            float(row[metric]) - float(original.loc[row["model"], metric])
            for _, row in out.iterrows()
        ]

    return out


def write_mc_draws_npz(
    path: Path,
    arrays: dict[str, np.ndarray],
    seed: int,
) -> None:
    payload: dict[str, Any] = {
        "mc_bootstrap_draws": np.asarray([MC_BOOTSTRAP_DRAWS], dtype=np.int64),
        "mc_seed": np.asarray([seed], dtype=np.int64),
        "model_order": np.asarray(PRIMARY_MODELS, dtype="U2"),
    }
    payload.update(arrays)
    np.savez_compressed(path, **payload)


def build_readme(
    final_table: pd.DataFrame,
    seed: int,
) -> str:
    point_cols = [
        "mean_delta_uno_c",
        "negative_transfer_rate_at_deltaC_le_-0.02",
        "catastrophic_transfer_rate_at_deltaC_le_-0.05",
    ]

    lines = [
        "Paper 6 - 05h3d frozen aggregation sensitivity",
        "",
        "Analysis role",
        "-------------",
        "POST-HOLD ROBUSTNESS ANALYSIS.",
        "The frozen-original 180-scenario equal-weight estimand remains PRIMARY and unchanged.",
        "core_only and regime_balanced are sensitivity estimands only.",
        "",
        "Technical binding",
        "-----------------",
        "05h3d performs no schema discovery and no label interpretation.",
        "All source hashes, columns, scenario memberships, regime mappings, NPZ paths,",
        "model-axis indices, and threshold identities come from the PASS 05h3c contract.",
        "",
        "Estimands",
        "---------",
        "frozen_original: all 180 frozen scenarios, equal weight per scenario.",
        "core_only: exact 36 CORE scenarios frozen by 05h3c, equal weight per scenario.",
        "regime_balanced: all 180 scenarios; equal scenario weight within each R0-R5,",
        "then equal 1/6 weight for each regime.",
        "",
        "Metrics",
        "-------",
        "mean delta Uno-C vs B0.",
        "negative-transfer rate at delta C <= -0.02.",
        "catastrophic-transfer rate at delta C <= -0.05.",
        "",
        "Monte Carlo uncertainty",
        "-----------------------",
        f"{MC_BOOTSTRAP_DRAWS:,} deterministic within-scenario replicate bootstrap draws.",
        f"Seed: {seed}, derived from the frozen 05h3c contract hash.",
        "Scenarios are fixed and are never resampled.",
        "The reported 2.5% and 97.5% quantiles quantify Monte Carlo uncertainty from",
        "the retained finite replicate runs under the fixed scenario design. They are not",
        "superpopulation confidence intervals over possible scenario designs.",
        "",
        "Point estimates",
        "---------------",
    ]

    for _, row in final_table.iterrows():
        vals = "; ".join(
            f"{metric}={float(row[metric]):.6f}"
            for metric in point_cols
        )
        lines.append(f"{row['estimand']} / {row['model']}: {vals}")

    lines += [
        "",
        "Interpretation guardrail",
        "------------------------",
        "Neither sensitivity estimand can replace the frozen original estimand, alter",
        "the -0.02/-0.05 definitions, reopen A2/A3 architecture selection, or alter",
        "HOLD_NO_SAFE_SELECTABLE_ARCHITECTURE.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    print("=" * 118)
    print("Paper 6 - compute frozen aggregation sensitivity after 05h3c preflight")
    print("=" * 118)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {ROOT}")
    print(f"Output directory: {OUT_DIR}")
    print()
    print("Execution contract:")
    print("  technical schema discovery: NO")
    print("  label guessing/parsing beyond frozen 05h3c mappings: NO")
    print("  frozen original estimand remains primary: YES")
    print("  core-only sensitivity: YES")
    print("  regime-balanced sensitivity: YES")
    print("  model fitting / retuning: NO")
    print("  human outcomes read: NO")
    print("  thresholds changed: NO")
    print("  scenarios resampled: NO")
    print("  retained replicates resampled within fixed scenarios: YES")
    print()

    if OUT_DIR.exists():
        raise RuntimeError(
            f"05h3d final output already exists; refusing overwrite: {OUT_DIR}"
        )
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir(parents=True, exist_ok=False)

    try:
        contract, h3c_manifest = verify_05h3c_outputs()
        verify_contract_semantics(contract)
        sources = verify_locked_sources(contract)

        print("05h3c frozen technical contract: PASS")
        print(f"  contract SHA256: {sha256_file(H3C_CONTRACT)}")
        print(f"  manifest SHA256: {sha256_file(H3C_MANIFEST)}")
        print("  all locked upstream source hashes: PASS")
        print()

        design = load_locked_design(contract)
        metrics = load_primary_scenario_metrics(contract, sources, design)
        weights = build_estimand_weights(contract, design)

        print("Frozen estimand construction: PASS")
        print("  frozen_original: 180 positive-weight scenarios")
        print("  core_only: 36 positive-weight scenarios")
        print("  regime_balanced: 180 positive-weight scenarios; R0-R5 each weight 1/6")
        print()

        points = compute_point_estimates(metrics, weights)
        regime_components = compute_regime_components(metrics)
        h2b_checks = verify_frozen_original_against_05h2b(points, sources)

        print("Deterministic point-estimate calculation: PASS")
        print("  frozen_original reproduces 05h2b: 6/6")
        print()

        inventory = load_locked_npz_inventory(contract, design)
        data_by_scenario = replay_npz_before_bootstrap(
            contract,
            metrics,
            inventory,
        )

        seed = deterministic_mc_seed(contract)
        print(
            f"Running {MC_BOOTSTRAP_DRAWS:,} within-scenario replicate bootstrap draws..."
        )
        mc_summary, mc_arrays = bootstrap_fixed_scenarios(
            design,
            data_by_scenario,
            weights,
            seed,
        )
        final_table = combine_points_with_mc(points, mc_summary)

        print("Monte Carlo uncertainty calculation: PASS")
        print(f"  seed: {seed}")
        print("  fixed scenarios resampled: NO")
        print("  retained replicates resampled within scenario: YES")
        print()

        print("-" * 118)
        print("05h3d AGGREGATION-SENSITIVITY POINT ESTIMATES")
        print("-" * 118)
        print(
            points[
                [
                    "estimand",
                    "model",
                    "n_scenarios_with_positive_weight",
                    "mean_delta_uno_c",
                    "negative_transfer_rate_at_deltaC_le_-0.02",
                    "catastrophic_transfer_rate_at_deltaC_le_-0.05",
                ]
            ].to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}",
            )
        )
        print()

        print("-" * 118)
        print("05h3d MONTE CARLO 95% REPLICATE-RESAMPLING INTERVALS")
        print("-" * 118)
        mc_display = mc_summary[
            [
                "estimand",
                "model",
                "metric",
                "mc_p025",
                "mc_p975",
            ]
        ].copy()
        print(
            mc_display.to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}",
            )
        )
        print()

        print("Interpretation guardrails:")
        print("  frozen_original remains PRIMARY: YES")
        print("  core_only is sensitivity only: YES")
        print("  regime_balanced is sensitivity only: YES")
        print("  frozen A2/A3 selection reopened: NO")
        print("  HOLD_NO_SAFE_SELECTABLE_ARCHITECTURE changed: NO")
        print()

        table_path = WORK_DIR / "Supplementary_Table_aggregation_sensitivity.csv"
        regime_path = WORK_DIR / "Supplementary_Data_regime_components.csv"
        scenario_path = WORK_DIR / "Supplementary_Data_A2_A3_scenario_metrics.csv"
        mc_summary_path = WORK_DIR / "Supplementary_Data_aggregation_sensitivity_mc_summary.csv"
        mc_draws_path = WORK_DIR / "Supplementary_Data_aggregation_sensitivity_mc_draws.npz"
        checks_path = WORK_DIR / "chain_checks.csv"
        result_json_path = WORK_DIR / "aggregation_sensitivity.json"
        readme_path = WORK_DIR / "README_for_supplement.txt"

        final_table.to_csv(
            table_path,
            index=False,
            float_format="%.10g",
        )
        regime_components.to_csv(
            regime_path,
            index=False,
            float_format="%.10g",
        )
        metrics.to_csv(
            scenario_path,
            index=False,
            float_format="%.10g",
        )
        mc_summary.to_csv(
            mc_summary_path,
            index=False,
            float_format="%.10g",
        )
        write_mc_draws_npz(mc_draws_path, mc_arrays, seed)
        pd.DataFrame(h2b_checks).to_csv(
            checks_path,
            index=False,
            float_format="%.12g",
        )

        result_json = {
            "script_version": SCRIPT_VERSION,
            "status": "PASS_POSTHOLD_AGGREGATION_SENSITIVITY",
            "analysis_role": "POST-HOLD ROBUSTNESS ANALYSIS",
            "frozen_primary_decision": "HOLD_NO_SAFE_SELECTABLE_ARCHITECTURE",
            "primary_decision_changed": False,
            "primary_models": PRIMARY_MODELS,
            "05h3c_contract_sha256": sha256_file(H3C_CONTRACT),
            "estimands": {
                "frozen_original": {
                    "scenario_count": 180,
                    "weighting": "equal weight per frozen scenario",
                    "role": "PRIMARY; unchanged",
                },
                "core_only": {
                    "scenario_count": 36,
                    "weighting": "equal weight per exact frozen CORE scenario",
                    "role": "POST-HOLD robustness sensitivity",
                },
                "regime_balanced": {
                    "scenario_count": 180,
                    "weighting": (
                        "equal within-regime scenario weight, then equal 1/6 weight per R0-R5"
                    ),
                    "role": "POST-HOLD robustness sensitivity",
                },
            },
            "thresholds": {
                "negative_transfer_delta_uno_c": NEGATIVE_THRESHOLD,
                "catastrophic_transfer_delta_uno_c": CATASTROPHIC_THRESHOLD,
            },
            "mc_uncertainty": {
                "draws": MC_BOOTSTRAP_DRAWS,
                "seed": seed,
                "seed_source": "05h3c contract SHA256",
                "resampling_unit": "retained replicate within fixed scenario",
                "scenario_resampling": False,
                "interval_quantiles": [MC_QLOW, MC_QHIGH],
            },
            "point_estimates": points.to_dict(orient="records"),
            "human_outcomes_read": False,
            "model_fitting": False,
            "thresholds_changed": False,
        }
        write_json(result_json_path, result_json)

        readme_path.write_text(
            build_readme(final_table, seed),
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
            "status": "PASS_POSTHOLD_AGGREGATION_SENSITIVITY",
            "inputs": {
                "05h3c_contract": {
                    "path": str(H3C_CONTRACT.relative_to(ROOT)),
                    "sha256": sha256_file(H3C_CONTRACT),
                },
                "05h3c_manifest": {
                    "path": str(H3C_MANIFEST.relative_to(ROOT)),
                    "sha256": sha256_file(H3C_MANIFEST),
                },
                "05h3c_design": {
                    "path": str(H3C_DESIGN.relative_to(ROOT)),
                    "sha256": sha256_file(H3C_DESIGN),
                },
                "05h3c_npz_inventory": {
                    "path": str(H3C_NPZ_INVENTORY.relative_to(ROOT)),
                    "sha256": sha256_file(H3C_NPZ_INVENTORY),
                },
            },
            "mc_bootstrap_draws": MC_BOOTSTRAP_DRAWS,
            "mc_seed": seed,
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

        for item in manifest["outputs"]:
            p = WORK_DIR / item["file"]
            if sha256_file(p) != item["sha256"]:
                raise RuntimeError(
                    f"05h3d output hash changed before finalization: {p.name}"
                )

        os.replace(WORK_DIR, OUT_DIR)

        print("=" * 118)
        print("05h3d post-HOLD aggregation sensitivity: PASS")
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
        print("05h3d post-HOLD aggregation sensitivity: FAIL")
        print(f"{type(exc).__name__}: {exc}")
        print("=" * 118)
        raise
