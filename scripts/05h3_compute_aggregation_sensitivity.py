#!/usr/bin/env python
"""
Paper 6 - frozen aggregation-sensitivity analysis for CBM strengthening.

Implements the 05h0 post-HOLD robustness contract for the frozen selectable
architectures A2 and A3:

  1) frozen_original:
       all 180 frozen scenarios, equal weight per scenario

  2) core_only:
       exactly the 36 frozen CORE scenarios, equal weight per scenario

  3) regime_balanced:
       all 180 frozen scenarios; equal weight within each frozen R0-R5 regime,
       followed by equal 1/6 weight for each of the six regimes

Reported metrics:
  - mean delta Uno-C vs B0
  - negative-transfer rate at delta C <= -0.02
  - catastrophic-transfer rate at delta C <= -0.05

Because retained per-replicate 05d metric arrays exist, Monte Carlo uncertainty
is estimated exactly as allowed/preferred by 05h0: resample retained replicates
within each fixed scenario, recompute the scenario summaries, and then reapply
the fixed aggregation weights. Fixed scenarios themselves are NOT resampled.

No model fitting, retuning, threshold changes, or outcome access occur.

Expected placement:
    scripts/05h3_compute_aggregation_sensitivity.py

Run:
    python scripts/05h3_compute_aggregation_sensitivity.py
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SCRIPT_VERSION = "05h3-compute-aggregation-sensitivity-v1-no-cli"

PRIMARY_MODELS = ["A2", "A3"]
REFERENCE_MODEL = "B0"

NEGATIVE_THRESHOLD = -0.02
CATASTROPHIC_THRESHOLD = -0.05

EXPECTED_SCENARIOS = 180
EXPECTED_CORE_SCENARIOS = 36
EXPECTED_STRESS_SCENARIOS = 144
EXPECTED_REPLICATES_TOTAL = 21600

EXPECTED_REGIME_COUNTS = {
    "R0": 6,
    "R1": 6,
    "R2": 78,
    "R3": 6,
    "R4": 6,
    "R5": 78,
}

# Computational—not scientific—Monte Carlo controls. The seed is derived from
# the frozen 05h0 contract hash at runtime, so it is outcome-independent and
# deterministic. 4,000 within-scenario bootstrap draws give stable descriptive
# 95% Monte Carlo intervals without treating scenarios as random draws.
MC_BOOTSTRAP_DRAWS = 4000
MC_INTERVAL_LOWER = 0.025
MC_INTERVAL_UPPER = 0.975

SOURCE_REPLAY_TOL = 2e-9
ORIGINAL_05H2B_TOL = 2e-9

SCENARIO_ID_RE = re.compile(r"\b(?:CORE|STRESS)_\d{4}\b", re.IGNORECASE)


def project_root() -> Path:
    p = Path(__file__).resolve()
    if p.parent.name.lower() != "scripts":
        raise RuntimeError(
            "Place 05h3_compute_aggregation_sensitivity.py in the repository scripts/ directory."
        )
    return p.parent.parent


ROOT = project_root()

H0_JSON = (
    ROOT
    / "method_contract"
    / "05h0_cbm_strengthening_contract"
    / "cbm_strengthening_contract.json"
)

H2B_JSON = (
    ROOT
    / "method_contract"
    / "05h2b_all_model_safety_summary"
    / "all_model_safety_summary.json"
)

H2B_MANIFEST = (
    ROOT
    / "method_contract"
    / "05h2b_all_model_safety_summary"
    / "freeze_manifest.json"
)

D_DIR = ROOT / "results" / "simulation_phase_diagram" / "05d"

SCENARIO_SUMMARY = D_DIR / "scenario_model_metric_summary.tsv"
MODEL_AGGREGATE = D_DIR / "model_aggregate_summary.tsv"
METRIC_CONTRACT = D_DIR / "metric_implementation_contract.json"
METRIC_MANIFEST = D_DIR / "scenario_metric_manifest.tsv"
D_SCRIPT = ROOT / "scripts" / "05d_aggregate_negative_transfer_phase_diagram.py"

OUT_DIR = ROOT / "method_contract" / "05h3_aggregation_sensitivity"
WORK_DIR = ROOT / "method_contract" / ".05h3_aggregation_sensitivity_work"

REQUIRED_SCENARIO_COLUMNS = [
    "scenario_id",
    "family",
    "transfer_regime",
    "replicates",
    "model",
    "mean_delta_c_vs_B0",
    "negative_transfer_rate",
    "catastrophic_negative_transfer_rate",
]

ESTIMAND_ORDER = ["frozen_original", "core_only", "regime_balanced"]

METRIC_SPECS = {
    "mean_delta_uno_c": {
        "scenario_column": "mean_delta_c_vs_B0",
        "npz_key": "delta_c_vs_B0",
    },
    "negative_transfer_rate_at_deltaC_le_-0.02": {
        "scenario_column": "negative_transfer_rate",
        "npz_key": "negative_transfer",
    },
    "catastrophic_transfer_rate_at_deltaC_le_-0.05": {
        "scenario_column": "catastrophic_negative_transfer_rate",
        "npz_key": "catastrophic_negative_transfer",
    },
}


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


def verify_h0_contract() -> dict[str, Any]:
    if not H0_JSON.exists():
        raise FileNotFoundError(f"Missing frozen 05h0 contract: {H0_JSON}")

    obj = json.loads(H0_JSON.read_text(encoding="utf-8"))
    block = recursive_find_key(obj, "aggregation_sensitivity")
    if not isinstance(block, dict):
        raise RuntimeError("05h0 lacks aggregation_sensitivity.")

    if block.get("analysis_role") != "POST-HOLD ROBUSTNESS ANALYSIS":
        raise RuntimeError("Unexpected 05h0 aggregation-sensitivity analysis role.")

    if block.get("primary_models") != PRIMARY_MODELS:
        raise RuntimeError(
            f"05h0 primary_models changed: {block.get('primary_models')!r}"
        )

    estimands = block.get("estimands")
    if not isinstance(estimands, dict):
        raise RuntimeError("05h0 aggregation_sensitivity.estimands is missing.")

    if list(estimands.keys()) != ESTIMAND_ORDER:
        # Dict insertion order is part of the frozen JSON representation in this
        # project; fail rather than silently reinterpreting the contract.
        if set(estimands.keys()) != set(ESTIMAND_ORDER):
            raise RuntimeError(
                f"05h0 estimand registry changed: {list(estimands.keys())!r}"
            )

    core = estimands.get("core_only", {})
    if "36" not in str(core.get("scenario_set", "")):
        raise RuntimeError("05h0 no longer specifies exactly 36 core scenarios.")

    rb = estimands.get("regime_balanced", {})
    regimes = rb.get("regimes")
    if regimes != ["R0", "R1", "R2", "R3", "R4", "R5"]:
        raise RuntimeError(f"05h0 frozen regime list changed: {regimes!r}")

    metrics = block.get("minimum_reported_metrics")
    expected_metrics = [
        "mean_delta_uno_c",
        "negative_transfer_rate_at_deltaC_le_-0.02",
        "catastrophic_transfer_rate_at_deltaC_le_-0.05",
    ]
    if metrics != expected_metrics:
        raise RuntimeError(
            f"05h0 minimum aggregation metrics changed: {metrics!r}"
        )

    policy = block.get("uncertainty_policy", {})
    preferred = str(
        policy.get("preferred_if_retained_replicate_outputs_exist", "")
    ).lower()
    if "resampling retained replicates" not in preferred:
        raise RuntimeError(
            "05h0 retained-replicate uncertainty policy is not the expected frozen policy."
        )

    return obj


def verify_05h2b_chain() -> dict[str, Any]:
    if not H2B_JSON.exists() or not H2B_MANIFEST.exists():
        raise FileNotFoundError(
            "05h2b PASS artifacts are required before 05h3. "
            f"Missing: {H2B_JSON if not H2B_JSON.exists() else H2B_MANIFEST}"
        )

    obj = json.loads(H2B_JSON.read_text(encoding="utf-8"))
    if obj.get("status") != "PASS_FROZEN_ALL_MODEL_SAFETY_SUMMARY_EXPORTED":
        raise RuntimeError(f"05h2b status is not PASS: {obj.get('status')!r}")
    if bool(obj.get("primary_decision_changed")):
        raise RuntimeError("05h2b indicates that the frozen primary decision changed.")
    if obj.get("eligible_for_primary_selection") != PRIMARY_MODELS:
        raise RuntimeError("05h2b selectable-model registry changed from A2/A3.")
    return obj


def normalize_regime(x: Any) -> str:
    s = str(x).strip().upper()
    m = re.match(r"^(R[0-5])(?:_|$)", s)
    if not m:
        raise RuntimeError(f"Cannot normalize transfer regime: {x!r}")
    return m.group(1)


def normalize_family(x: Any) -> str:
    s = str(x).strip().upper()
    if s in {"CORE", "PRE_STRESS", "PRE-STRESS"}:
        return "CORE"
    if s in {"STRESS", "STRESS_TEST", "STRESS-TEST"}:
        return "STRESS"
    # The scenario IDs are only a cross-check, not the primary resolver.
    raise RuntimeError(f"Unexpected frozen scenario family label: {x!r}")


def load_scenario_design_and_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    for p in [
        SCENARIO_SUMMARY,
        MODEL_AGGREGATE,
        METRIC_CONTRACT,
        METRIC_MANIFEST,
    ]:
        if not p.exists():
            raise FileNotFoundError(f"Required frozen 05d artifact missing: {p}")

    raw = pd.read_csv(SCENARIO_SUMMARY, sep="\t", low_memory=False)
    missing = [c for c in REQUIRED_SCENARIO_COLUMNS if c not in raw.columns]
    if missing:
        raise RuntimeError(
            "Authoritative 05d scenario summary schema changed; "
            f"missing columns: {missing}"
        )

    # The scenario-design metadata are carried in the authoritative frozen 05d
    # table: family + transfer_regime + replicate count. Resolve each scenario
    # once and require exact consistency across all model rows.
    design_cols = ["scenario_id", "family", "transfer_regime", "replicates"]
    design = raw[design_cols].drop_duplicates().copy()

    if design["scenario_id"].duplicated().any():
        bad = design[
            design["scenario_id"].duplicated(keep=False)
        ].sort_values("scenario_id")
        raise RuntimeError(
            "Frozen scenario-design metadata disagree across model rows:\n"
            + bad.head(20).to_string(index=False)
        )

    if len(design) != EXPECTED_SCENARIOS:
        raise RuntimeError(
            f"Expected 180 frozen scenarios in design metadata; found {len(design)}."
        )

    design["family_resolved"] = design["family"].map(normalize_family)
    design["regime"] = design["transfer_regime"].map(normalize_regime)
    design["replicates"] = pd.to_numeric(
        design["replicates"], errors="raise"
    ).astype(int)

    n_core = int((design["family_resolved"] == "CORE").sum())
    n_stress = int((design["family_resolved"] == "STRESS").sum())
    if n_core != EXPECTED_CORE_SCENARIOS:
        raise RuntimeError(
            f"05h0 requires exactly 36 core scenarios; frozen design has {n_core}."
        )
    if n_stress != EXPECTED_STRESS_SCENARIOS:
        raise RuntimeError(
            f"Expected 144 stress scenarios; frozen design has {n_stress}."
        )

    # Cross-check the family metadata against the historical scenario IDs, while
    # retaining family as the actual resolver.
    core_id = design["scenario_id"].astype(str).str.startswith("CORE_")
    stress_id = design["scenario_id"].astype(str).str.startswith("STRESS_")
    if not (
        ((design["family_resolved"] == "CORE") == core_id)
        & ((design["family_resolved"] == "STRESS") == stress_id)
    ).all():
        raise RuntimeError(
            "Frozen family metadata disagree with CORE_/STRESS_ scenario identifiers."
        )

    regime_counts = (
        design["regime"]
        .value_counts()
        .reindex(["R0", "R1", "R2", "R3", "R4", "R5"], fill_value=0)
        .astype(int)
        .to_dict()
    )
    if regime_counts != EXPECTED_REGIME_COUNTS:
        raise RuntimeError(
            f"Frozen regime counts changed: expected {EXPECTED_REGIME_COUNTS}, "
            f"found {regime_counts}."
        )

    if int(design["replicates"].sum()) != EXPECTED_REPLICATES_TOTAL:
        raise RuntimeError(
            "Frozen scenario-design replicate total changed: "
            f"expected {EXPECTED_REPLICATES_TOTAL}, "
            f"found {int(design['replicates'].sum())}."
        )

    metrics = raw[raw["model"].isin(PRIMARY_MODELS)].copy()
    if len(metrics) != EXPECTED_SCENARIOS * len(PRIMARY_MODELS):
        raise RuntimeError(
            f"Expected 360 A2/A3 scenario rows; found {len(metrics)}."
        )
    if metrics.duplicated(["scenario_id", "model"]).any():
        raise RuntimeError("A2/A3 scenario metrics are not unique.")

    for c in [
        "mean_delta_c_vs_B0",
        "negative_transfer_rate",
        "catastrophic_negative_transfer_rate",
    ]:
        metrics[c] = pd.to_numeric(metrics[c], errors="coerce")
        if not np.isfinite(metrics[c].to_numpy(dtype=float)).all():
            raise RuntimeError(f"Non-finite A2/A3 frozen scenario metric: {c}")

    metrics = metrics.merge(
        design[
            [
                "scenario_id",
                "family_resolved",
                "regime",
                "replicates",
            ]
        ],
        on="scenario_id",
        how="left",
        suffixes=("", "_design"),
        validate="many_to_one",
    )
    if metrics[["family_resolved", "regime", "replicates_design"]].isna().any().any():
        raise RuntimeError("Failed to attach frozen design metadata to A2/A3 metrics.")

    if not np.array_equal(
        metrics["replicates"].astype(int).to_numpy(),
        metrics["replicates_design"].astype(int).to_numpy(),
    ):
        raise RuntimeError("Replicate counts disagree between frozen metric and design views.")

    metrics.drop(columns=["replicates_design"], inplace=True)

    return (
        design.sort_values("scenario_id").reset_index(drop=True),
        metrics.sort_values(["scenario_id", "model"], kind="mergesort").reset_index(drop=True),
    )


def estimand_weights(design: pd.DataFrame) -> dict[str, dict[str, float]]:
    weights: dict[str, dict[str, float]] = {}

    # Frozen original: equal scenario weight across all 180.
    weights["frozen_original"] = {
        sid: 1.0 / EXPECTED_SCENARIOS
        for sid in design["scenario_id"]
    }

    # Core-only: exactly the 36 pre-stress scenarios, resolved from frozen family.
    core_ids = list(
        design.loc[
            design["family_resolved"] == "CORE",
            "scenario_id",
        ]
    )
    if len(core_ids) != EXPECTED_CORE_SCENARIOS:
        raise RuntimeError("Core-only resolver did not return exactly 36 scenarios.")
    weights["core_only"] = {
        sid: (1.0 / EXPECTED_CORE_SCENARIOS if sid in set(core_ids) else 0.0)
        for sid in design["scenario_id"]
    }

    # Regime-balanced: equal scenario weight within each regime, then 1/6 per regime.
    rb: dict[str, float] = {}
    for regime in ["R0", "R1", "R2", "R3", "R4", "R5"]:
        ids = list(design.loc[design["regime"] == regime, "scenario_id"])
        if len(ids) != EXPECTED_REGIME_COUNTS[regime]:
            raise RuntimeError(f"Unexpected scenario count in {regime}.")
        w = (1.0 / 6.0) / len(ids)
        for sid in ids:
            rb[sid] = w
    weights["regime_balanced"] = rb

    for name, mapping in weights.items():
        total = float(sum(mapping.values()))
        if abs(total - 1.0) > 1e-12:
            raise RuntimeError(f"{name} weights do not sum to 1: {total}")

    return weights


def point_estimates(
    metrics: pd.DataFrame,
    weights: dict[str, dict[str, float]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    metric_map = {
        "mean_delta_uno_c": "mean_delta_c_vs_B0",
        "negative_transfer_rate_at_deltaC_le_-0.02": "negative_transfer_rate",
        "catastrophic_transfer_rate_at_deltaC_le_-0.05": (
            "catastrophic_negative_transfer_rate"
        ),
    }

    for estimand in ESTIMAND_ORDER:
        wmap = weights[estimand]
        positive_ids = {sid for sid, w in wmap.items() if w > 0}
        for model in PRIMARY_MODELS:
            z = metrics[metrics["model"] == model].copy()
            z["_w"] = z["scenario_id"].map(wmap).astype(float)
            if z["_w"].isna().any():
                raise RuntimeError(f"Missing {estimand} weight for {model}.")
            row: dict[str, Any] = {
                "estimand": estimand,
                "model": model,
                "n_scenarios_with_positive_weight": len(positive_ids),
            }
            for out_name, col in metric_map.items():
                row[out_name] = float(np.sum(z["_w"] * z[col]))
            rows.append(row)

    out = pd.DataFrame(rows)

    descriptions = {
        "frozen_original": (
            "all 180 frozen scenarios; equal weight per scenario; PRIMARY unchanged"
        ),
        "core_only": (
            "exactly 36 frozen CORE scenarios; equal weight per core scenario"
        ),
        "regime_balanced": (
            "all 180 frozen scenarios; equal within-regime scenario weights; "
            "R0-R5 each receive weight 1/6"
        ),
    }
    out["weighting"] = out["estimand"].map(descriptions)
    out["analysis_role"] = np.where(
        out["estimand"] == "frozen_original",
        "PRIMARY FROZEN ESTIMAND; UNCHANGED",
        "POST-HOLD ROBUSTNESS ANALYSIS",
    )

    return out[
        [
            "estimand",
            "model",
            "n_scenarios_with_positive_weight",
            "mean_delta_uno_c",
            "negative_transfer_rate_at_deltaC_le_-0.02",
            "catastrophic_transfer_rate_at_deltaC_le_-0.05",
            "weighting",
            "analysis_role",
        ]
    ]


def regime_components(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for regime in ["R0", "R1", "R2", "R3", "R4", "R5"]:
        for model in PRIMARY_MODELS:
            z = metrics[
                (metrics["regime"] == regime) & (metrics["model"] == model)
            ]
            rows.append(
                {
                    "regime": regime,
                    "model": model,
                    "n_scenarios": int(len(z)),
                    "mean_delta_uno_c": float(z["mean_delta_c_vs_B0"].mean()),
                    "negative_transfer_rate_at_deltaC_le_-0.02": float(
                        z["negative_transfer_rate"].mean()
                    ),
                    "catastrophic_transfer_rate_at_deltaC_le_-0.05": float(
                        z["catastrophic_negative_transfer_rate"].mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def verify_original_against_05h2b(
    points: pd.DataFrame,
    h2b: dict[str, Any],
) -> list[dict[str, Any]]:
    summaries = h2b.get("model_summary")
    if not isinstance(summaries, list):
        raise RuntimeError("05h2b JSON lacks model_summary.")

    h = {
        str(row["model"]): row
        for row in summaries
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
        if model not in h:
            raise RuntimeError(f"05h2b JSON lacks {model}.")
        for pcol, hcol in pairs:
            observed = float(original.loc[model, pcol])
            expected = float(h[model][hcol])
            diff = abs(observed - expected)
            checks.append(
                {
                    "check": "05h3_frozen_original_vs_05h2b",
                    "model": model,
                    "metric": pcol,
                    "observed": observed,
                    "expected": expected,
                    "absolute_difference": diff,
                    "tolerance": ORIGINAL_05H2B_TOL,
                    "pass": diff <= ORIGINAL_05H2B_TOL,
                }
            )

    failed = [c for c in checks if not c["pass"]]
    if failed:
        details = "\n".join(
            f"  {c['model']} {c['metric']}: "
            f"{c['observed']:.12g} vs {c['expected']:.12g}"
            for c in failed
        )
        raise RuntimeError(
            "Frozen-original 05h3 values do not reproduce accepted 05h2b:\n"
            + details
        )
    return checks


def scenario_id_from_path(path: Path) -> str | None:
    m = SCENARIO_ID_RE.search(str(path))
    return m.group(0).upper() if m else None


def resolve_npz_files(design: pd.DataFrame) -> dict[str, Path]:
    npz_files = sorted(D_DIR.rglob("*.npz"))
    if len(npz_files) != EXPECTED_SCENARIOS:
        raise RuntimeError(
            f"Expected exactly 180 retained 05d NPZ metric artifacts; found {len(npz_files)}."
        )

    mapping: dict[str, Path] = {}
    for p in npz_files:
        sid = scenario_id_from_path(p)
        if sid is None:
            raise RuntimeError(f"Cannot resolve scenario ID from retained NPZ path: {p}")
        if sid in mapping:
            raise RuntimeError(
                f"Multiple retained NPZ artifacts resolve to scenario {sid}: "
                f"{mapping[sid]} and {p}"
            )
        mapping[sid] = p

    expected = set(design["scenario_id"].astype(str))
    if set(mapping) != expected:
        missing = sorted(expected - set(mapping))
        extra = sorted(set(mapping) - expected)
        raise RuntimeError(
            "Retained NPZ scenario set does not equal the frozen design. "
            f"Missing={missing[:10]}, extra={extra[:10]}"
        )
    return mapping


def collect_string_list_candidates(
    obj: Any,
    n_models: int,
) -> list[tuple[str, tuple[str, ...]]]:
    out: list[tuple[str, tuple[str, ...]]] = []

    def walk(x: Any, label: str) -> None:
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, f"{label}.{k}" if label else str(k))
        elif isinstance(x, list):
            if (
                len(x) == n_models
                and all(isinstance(v, str) for v in x)
                and set(PRIMARY_MODELS + [REFERENCE_MODEL]).issubset(set(x))
            ):
                out.append((label, tuple(str(v) for v in x)))
            for i, v in enumerate(x):
                walk(v, f"{label}[{i}]")
    walk(obj, "")
    return out


def ast_model_order_candidates(
    path: Path,
    n_models: int,
) -> list[tuple[str, tuple[str, ...]]]:
    if not path.exists():
        return []

    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    out: list[tuple[str, tuple[str, ...]]] = []

    def names_for_target(target: ast.AST) -> list[str]:
        if isinstance(target, ast.Name):
            return [target.id]
        if isinstance(target, (ast.Tuple, ast.List)):
            names = []
            for elt in target.elts:
                names.extend(names_for_target(elt))
            return names
        return []

    for node in ast.walk(tree):
        name_candidates: list[str] = []
        value: ast.AST | None = None
        if isinstance(node, ast.Assign):
            for target in node.targets:
                name_candidates.extend(names_for_target(target))
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            name_candidates.extend(names_for_target(node.target))
            value = node.value

        if value is None:
            continue

        if not any(
            ("MODEL" in name.upper() or "ARCH" in name.upper())
            for name in name_candidates
        ):
            continue

        try:
            literal = ast.literal_eval(value)
        except Exception:
            continue

        if isinstance(literal, (list, tuple)):
            seq = tuple(str(v) for v in literal)
            if (
                len(seq) == n_models
                and set(PRIMARY_MODELS + [REFERENCE_MODEL]).issubset(set(seq))
            ):
                out.append(
                    (
                        f"{path.name}:{','.join(name_candidates)}",
                        seq,
                    )
                )
        elif isinstance(literal, dict):
            try:
                if (
                    len(literal) == n_models
                    and all(isinstance(k, str) for k in literal)
                    and all(isinstance(v, int) for v in literal.values())
                ):
                    seq_list = [None] * n_models
                    for k, idx in literal.items():
                        if 0 <= idx < n_models:
                            seq_list[idx] = str(k)
                    if (
                        all(v is not None for v in seq_list)
                        and set(PRIMARY_MODELS + [REFERENCE_MODEL]).issubset(
                            set(seq_list)
                        )
                    ):
                        out.append(
                            (
                                f"{path.name}:{','.join(name_candidates)}",
                                tuple(seq_list),
                            )
                        )
            except Exception:
                pass

    return out


def validate_model_order_candidate(
    candidate: tuple[str, ...],
    raw_summary: pd.DataFrame,
    npz_map: dict[str, Path],
) -> bool:
    summary_models = set(raw_summary["model"].astype(str).unique())
    if len(candidate) != len(summary_models) or set(candidate) != summary_models:
        return False

    # Validate the candidate against every retained scenario by reproducing the
    # frozen scenario mean delta-C for every model. This is identity validation,
    # not model selection.
    for sid in sorted(npz_map):
        p = npz_map[sid]
        with np.load(p, allow_pickle=False) as data:
            if "delta_c_vs_B0" not in data.files:
                return False
            arr = np.asarray(data["delta_c_vs_B0"], dtype=float)

        if arr.ndim != 2 or arr.shape[1] != len(candidate):
            return False

        part = raw_summary[raw_summary["scenario_id"] == sid]
        frozen = {
            str(row.model): float(row.mean_delta_c_vs_B0)
            for row in part.itertuples()
        }
        if set(frozen) != set(candidate):
            return False

        for idx, model in enumerate(candidate):
            observed = float(np.nanmean(arr[:, idx]))
            expected = frozen[model]
            if not (
                math.isfinite(observed)
                and math.isfinite(expected)
                and abs(observed - expected) <= SOURCE_REPLAY_TOL
            ):
                return False

    return True


def resolve_model_order(
    raw_summary: pd.DataFrame,
    npz_map: dict[str, Path],
) -> tuple[tuple[str, ...], list[str]]:
    first_path = npz_map[sorted(npz_map)[0]]
    with np.load(first_path, allow_pickle=False) as data:
        if "delta_c_vs_B0" not in data.files:
            raise RuntimeError(
                f"Retained metric NPZ lacks delta_c_vs_B0: {first_path}"
            )
        delta = np.asarray(data["delta_c_vs_B0"], dtype=float)
        if delta.ndim != 2:
            raise RuntimeError(
                f"Unexpected delta_c_vs_B0 shape in {first_path}: {delta.shape}"
            )
        n_models = int(delta.shape[1])

        candidates: list[tuple[str, tuple[str, ...]]] = []

        # Prefer an explicit string model order stored in the NPZ if present.
        for key in data.files:
            try:
                arr = np.asarray(data[key])
            except Exception:
                continue
            if (
                arr.ndim == 1
                and len(arr) == n_models
                and arr.dtype.kind in {"U", "S"}
            ):
                seq = tuple(
                    v.decode("utf-8") if isinstance(v, bytes) else str(v)
                    for v in arr.tolist()
                )
                if set(PRIMARY_MODELS + [REFERENCE_MODEL]).issubset(set(seq)):
                    candidates.append((f"NPZ:{key}", seq))

    # Frozen metric implementation contract.
    try:
        contract_obj = json.loads(METRIC_CONTRACT.read_text(encoding="utf-8"))
        for label, seq in collect_string_list_candidates(contract_obj, n_models):
            candidates.append((f"metric_contract:{label}", seq))
    except Exception:
        pass

    # Frozen 05d implementation source, parsed without executing it.
    candidates.extend(ast_model_order_candidates(D_SCRIPT, n_models))

    # Final identity candidate: stable row order in the authoritative frozen 05d
    # scenario table. It is only accepted if it exactly reproduces every
    # scenario/model mean delta-C from all 180 retained NPZ arrays.
    orders = []
    for sid, part in raw_summary.groupby("scenario_id", sort=False):
        seq = tuple(part["model"].astype(str).tolist())
        orders.append(seq)
    if orders and all(seq == orders[0] for seq in orders):
        candidates.append(("authoritative_05d_summary_stable_row_order", orders[0]))

    # Deduplicate identical candidate sequences but preserve provenance labels.
    by_seq: dict[tuple[str, ...], list[str]] = {}
    for label, seq in candidates:
        by_seq.setdefault(seq, []).append(label)

    validated: list[tuple[tuple[str, ...], list[str]]] = []
    for seq, labels in by_seq.items():
        if validate_model_order_candidate(seq, raw_summary, npz_map):
            validated.append((seq, labels))

    if not validated:
        labels = [label for label, _ in candidates]
        raise RuntimeError(
            "Could not validate the retained NPZ model-axis order against all 180 "
            "frozen scenario summaries. Candidate sources tried: "
            + (", ".join(labels) if labels else "none")
        )

    unique_sequences = {seq for seq, _ in validated}
    if len(unique_sequences) != 1:
        details = "\n".join(
            f"  {labels}: {seq}"
            for seq, labels in validated
        )
        raise RuntimeError(
            "More than one non-equivalent model-axis order validated; refusing ambiguity:\n"
            + details
        )

    seq, labels = validated[0]
    return seq, labels


def verify_replicate_sources_and_collect(
    design: pd.DataFrame,
    metrics: pd.DataFrame,
    raw_summary: pd.DataFrame,
    npz_map: dict[str, Path],
    model_order: tuple[str, ...],
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    list[dict[str, Any]],
]:
    model_index = {m: i for i, m in enumerate(model_order)}
    for m in PRIMARY_MODELS:
        if m not in model_index:
            raise RuntimeError(f"Retained NPZ model order lacks {m}.")

    replicate_data: dict[str, dict[str, np.ndarray]] = {}
    source_rows: list[dict[str, Any]] = []

    metric_lookup = metrics.set_index(["scenario_id", "model"])

    for row in design.itertuples(index=False):
        sid = str(row.scenario_id)
        path = npz_map[sid]

        with np.load(path, allow_pickle=False) as data:
            required_keys = [
                "replicate_seed",
                "delta_c_vs_B0",
                "negative_transfer",
                "catastrophic_negative_transfer",
            ]
            missing = [k for k in required_keys if k not in data.files]
            if missing:
                raise RuntimeError(
                    f"{sid}: retained NPZ lacks required frozen metric arrays {missing}."
                )

            seeds = np.asarray(data["replicate_seed"], dtype=np.int64)
            delta = np.asarray(data["delta_c_vs_B0"], dtype=float)
            neg = np.asarray(data["negative_transfer"])
            cat = np.asarray(data["catastrophic_negative_transfer"])

        n = int(row.replicates)
        expected_shape = (n, len(model_order))
        if delta.shape != expected_shape:
            raise RuntimeError(
                f"{sid}: delta shape {delta.shape}, expected {expected_shape}."
            )
        if neg.shape != expected_shape:
            raise RuntimeError(
                f"{sid}: negative-transfer shape {neg.shape}, expected {expected_shape}."
            )
        if cat.shape != expected_shape:
            raise RuntimeError(
                f"{sid}: catastrophic-transfer shape {cat.shape}, expected {expected_shape}."
            )
        if seeds.shape != (n,):
            raise RuntimeError(
                f"{sid}: replicate_seed shape {seeds.shape}, expected {(n,)}."
            )
        if len(np.unique(seeds)) != n:
            raise RuntimeError(f"{sid}: retained replicate seeds are not unique.")

        # Extract only the two frozen selectable architectures required by 05h0.
        idx = [model_index[m] for m in PRIMARY_MODELS]
        d = delta[:, idx].astype(np.float64, copy=False)
        nflag = neg[:, idx].astype(np.float64, copy=False)
        cflag = cat[:, idx].astype(np.float64, copy=False)

        if not np.isin(nflag, [0.0, 1.0]).all():
            raise RuntimeError(f"{sid}: negative-transfer array is not binary.")
        if not np.isin(cflag, [0.0, 1.0]).all():
            raise RuntimeError(f"{sid}: catastrophic-transfer array is not binary.")

        # Replay every A2/A3 scenario-level metric from retained replicate arrays
        # before any sensitivity computation.
        for j, model in enumerate(PRIMARY_MODELS):
            frozen = metric_lookup.loc[(sid, model)]

            observed_delta = float(np.nanmean(d[:, j]))
            expected_delta = float(frozen["mean_delta_c_vs_B0"])

            observed_nt = float(np.mean(nflag[:, j]))
            expected_nt = float(frozen["negative_transfer_rate"])

            observed_cat = float(np.mean(cflag[:, j]))
            expected_cat = float(
                frozen["catastrophic_negative_transfer_rate"]
            )

            checks = [
                ("mean_delta_c_vs_B0", observed_delta, expected_delta),
                ("negative_transfer_rate", observed_nt, expected_nt),
                (
                    "catastrophic_negative_transfer_rate",
                    observed_cat,
                    expected_cat,
                ),
            ]
            for label, observed, expected in checks:
                if not (
                    math.isfinite(observed)
                    and math.isfinite(expected)
                    and abs(observed - expected) <= SOURCE_REPLAY_TOL
                ):
                    raise RuntimeError(
                        f"{sid}/{model}: retained NPZ replay mismatch for {label}: "
                        f"{observed:.12g} vs frozen {expected:.12g}."
                    )

        replicate_data[sid] = {
            "delta": d,
            "negative": nflag,
            "catastrophic": cflag,
        }
        source_rows.append(
            {
                "scenario_id": sid,
                "family": str(row.family_resolved),
                "regime": str(row.regime),
                "replicates": n,
                "relative_path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )

    return replicate_data, source_rows


def derive_mc_seed() -> int:
    digest = sha256_file(H0_JSON)
    # Deterministic outcome-independent 32-bit seed derived from the frozen
    # strengthening contract itself.
    return int(digest[:8], 16)


def bootstrap_fixed_scenarios(
    design: pd.DataFrame,
    replicate_data: dict[str, dict[str, np.ndarray]],
    weights: dict[str, dict[str, float]],
) -> tuple[pd.DataFrame, dict[str, np.ndarray], int]:
    seed = derive_mc_seed()
    rng = np.random.default_rng(seed)

    # Arrays are accumulated as [draw, model].
    accumulator: dict[str, dict[str, np.ndarray]] = {}
    for estimand in ESTIMAND_ORDER:
        accumulator[estimand] = {
            metric: np.zeros(
                (MC_BOOTSTRAP_DRAWS, len(PRIMARY_MODELS)),
                dtype=np.float64,
            )
            for metric in METRIC_SPECS
        }

    for row in design.sort_values("scenario_id").itertuples(index=False):
        sid = str(row.scenario_id)
        n = int(row.replicates)
        arrays = replicate_data[sid]

        # The same resampled replicate indices are used for A2/A3 and all three
        # metrics, preserving the frozen within-replicate pairing.
        draw_idx = rng.integers(
            0,
            n,
            size=(MC_BOOTSTRAP_DRAWS, n),
            dtype=np.int32,
        )

        # delta: retained floating point metric, use nanmean to match the frozen
        # finite-mean behavior if any invalid Uno-C replicate were ever present.
        with np.errstate(invalid="ignore"):
            dboot = np.nanmean(arrays["delta"][draw_idx, :], axis=1)
        if not np.isfinite(dboot).all():
            raise RuntimeError(
                f"{sid}: non-finite bootstrap mean delta-C encountered."
            )

        nboot = arrays["negative"][draw_idx, :].mean(axis=1)
        cboot = arrays["catastrophic"][draw_idx, :].mean(axis=1)

        scenario_boot = {
            "mean_delta_uno_c": dboot,
            "negative_transfer_rate_at_deltaC_le_-0.02": nboot,
            "catastrophic_transfer_rate_at_deltaC_le_-0.05": cboot,
        }

        for estimand in ESTIMAND_ORDER:
            w = float(weights[estimand][sid])
            if w == 0.0:
                continue
            for metric, values in scenario_boot.items():
                accumulator[estimand][metric] += w * values

    rows: list[dict[str, Any]] = []
    saved_arrays: dict[str, np.ndarray] = {}

    for estimand in ESTIMAND_ORDER:
        for metric in METRIC_SPECS:
            arr = accumulator[estimand][metric]
            saved_arrays[f"{estimand}__{metric}"] = arr
            for j, model in enumerate(PRIMARY_MODELS):
                vals = arr[:, j]
                rows.append(
                    {
                        "estimand": estimand,
                        "model": model,
                        "metric": metric,
                        "mc_bootstrap_draws": MC_BOOTSTRAP_DRAWS,
                        "mc_seed": seed,
                        "mc_bootstrap_mean": float(np.mean(vals)),
                        "mc_bootstrap_sd": float(
                            np.std(vals, ddof=1)
                        ),
                        "mc_bootstrap_p025": float(
                            np.quantile(vals, MC_INTERVAL_LOWER)
                        ),
                        "mc_bootstrap_p500": float(
                            np.quantile(vals, 0.5)
                        ),
                        "mc_bootstrap_p975": float(
                            np.quantile(vals, MC_INTERVAL_UPPER)
                        ),
                        "resampling_unit": (
                            "retained replicate within each fixed frozen scenario"
                        ),
                        "scenario_resampling": False,
                    }
                )

    return pd.DataFrame(rows), saved_arrays, seed


def combine_points_and_mc(
    points: pd.DataFrame,
    mc: pd.DataFrame,
) -> pd.DataFrame:
    p = points.copy()

    for metric in METRIC_SPECS:
        m = mc[mc["metric"] == metric][
            [
                "estimand",
                "model",
                "mc_bootstrap_sd",
                "mc_bootstrap_p025",
                "mc_bootstrap_p975",
            ]
        ].copy()
        rename = {
            "mc_bootstrap_sd": f"{metric}__mc_sd",
            "mc_bootstrap_p025": f"{metric}__mc_p025",
            "mc_bootstrap_p975": f"{metric}__mc_p975",
        }
        m.rename(columns=rename, inplace=True)
        p = p.merge(m, on=["estimand", "model"], how="left", validate="one_to_one")

    ordered = [
        "estimand",
        "model",
        "n_scenarios_with_positive_weight",
        "mean_delta_uno_c",
        "mean_delta_uno_c__mc_sd",
        "mean_delta_uno_c__mc_p025",
        "mean_delta_uno_c__mc_p975",
        "negative_transfer_rate_at_deltaC_le_-0.02",
        "negative_transfer_rate_at_deltaC_le_-0.02__mc_sd",
        "negative_transfer_rate_at_deltaC_le_-0.02__mc_p025",
        "negative_transfer_rate_at_deltaC_le_-0.02__mc_p975",
        "catastrophic_transfer_rate_at_deltaC_le_-0.05",
        "catastrophic_transfer_rate_at_deltaC_le_-0.05__mc_sd",
        "catastrophic_transfer_rate_at_deltaC_le_-0.05__mc_p025",
        "catastrophic_transfer_rate_at_deltaC_le_-0.05__mc_p975",
        "weighting",
        "analysis_role",
    ]
    return p[ordered]


def write_mc_npz(
    path: Path,
    arrays: dict[str, np.ndarray],
    seed: int,
) -> None:
    payload: dict[str, Any] = {
        "mc_bootstrap_draws": np.asarray([MC_BOOTSTRAP_DRAWS], dtype=np.int64),
        "mc_seed": np.asarray([seed], dtype=np.uint64),
        "model_order": np.asarray(PRIMARY_MODELS, dtype="U2"),
    }
    payload.update(arrays)
    np.savez_compressed(path, **payload)


def build_readme(
    final_table: pd.DataFrame,
    seed: int,
    model_order: tuple[str, ...],
    model_order_sources: list[str],
) -> str:
    def get(estimand: str, model: str, metric: str) -> float:
        row = final_table[
            (final_table["estimand"] == estimand)
            & (final_table["model"] == model)
        ].iloc[0]
        return float(row[metric])

    lines = [
        "Paper 6 - 05h3 aggregation-sensitivity analysis",
        "",
        "Analysis role",
        "-------------",
        "POST-HOLD ROBUSTNESS ANALYSIS. The frozen original estimand remains primary",
        "and is not replaced by either sensitivity estimand.",
        "",
        "Frozen selectable architectures",
        "-------------------------------",
        "A2 and A3 only. No architecture-selection decision is reopened.",
        "",
        "Estimands",
        "---------",
        "frozen_original:",
        "  all 180 frozen scenarios; equal weight per scenario; PRIMARY unchanged.",
        "",
        "core_only:",
        "  exactly 36 frozen CORE/pre-stress scenarios; equal weight per scenario.",
        "",
        "regime_balanced:",
        "  all 180 frozen scenarios; equal scenario weight within each R0-R5 regime,",
        "  followed by equal 1/6 weight per regime.",
        "",
        "Frozen design checks",
        "--------------------",
        "180 total scenarios; 36 CORE; 144 STRESS.",
        "R0/R1/R2/R3/R4/R5 scenario counts = 6/6/78/6/6/78.",
        "21,600 retained synthetic replicates total.",
        "",
        "Metrics",
        "-------",
        "mean delta Uno-C vs B0.",
        "negative-transfer rate at delta C <= -0.02.",
        "catastrophic-transfer rate at delta C <= -0.05.",
        "No threshold is changed.",
        "",
        "Monte Carlo uncertainty",
        "-----------------------",
        f"{MC_BOOTSTRAP_DRAWS:,} deterministic bootstrap draws.",
        f"Seed = {seed}, derived from the frozen 05h0 SHA256 rather than outcome values.",
        "Within every fixed scenario, retained replicate indices are resampled with",
        "replacement. The same replicate draw is used jointly for A2/A3 and all metrics.",
        "Fixed scenarios are never resampled, because the scenario design is not treated",
        "as a random sample from a scenario superpopulation.",
        "The reported 2.5% and 97.5% quantiles are Monte Carlo uncertainty intervals for",
        "the fixed-design estimands, not population-level scenario confidence intervals.",
        "",
        "Retained metric provenance",
        "--------------------------",
        "The 180 frozen 05d NPZ metric arrays are used only for within-scenario Monte Carlo",
        "resampling. Before analysis, A2/A3 scenario-level delta-C, negative-transfer",
        "rate, and catastrophic-transfer rate are replayed from those arrays and must",
        "match scenario_model_metric_summary.tsv.",
        f"Validated retained-NPZ model order: {', '.join(model_order)}",
        "Model-order provenance/validation:",
    ]
    lines.extend([f"  - {x}" for x in model_order_sources])

    lines += [
        "",
        "Key deterministic point estimates",
        "---------------------------------",
    ]
    for estimand in ESTIMAND_ORDER:
        for model in PRIMARY_MODELS:
            delta = get(estimand, model, "mean_delta_uno_c")
            nt = get(
                estimand,
                model,
                "negative_transfer_rate_at_deltaC_le_-0.02",
            )
            cat = get(
                estimand,
                model,
                "catastrophic_transfer_rate_at_deltaC_le_-0.05",
            )
            lines.append(
                f"{estimand} / {model}: deltaC={delta:.6f}; NT={nt:.6f}; catastrophic={cat:.6f}"
            )

    lines += [
        "",
        "Interpretation guardrail",
        "------------------------",
        "Neither core_only nor regime_balanced can replace the frozen original estimand,",
        "change the frozen -0.02/-0.05 safety definitions, or reverse/reopen the original",
        "HOLD_NO_SAFE_SELECTABLE_ARCHITECTURE decision.",
        "",
        "Prohibited operations",
        "---------------------",
        "No model fitting or retuning.",
        "No TARGET/GSE21257/GSE39055/DOG2 outcomes.",
        "No survival-metric recomputation from patient predictions.",
        "No scenario resampling.",
        "No retrospective threshold change.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    print("=" * 118)
    print("Paper 6 - frozen aggregation-sensitivity analysis for CBM strengthening")
    print("=" * 118)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {ROOT}")
    print(f"Output directory: {OUT_DIR}")
    print()
    print("Safety / execution contract:")
    print("  analysis role: POST-HOLD ROBUSTNESS ANALYSIS")
    print("  frozen original estimand remains primary: YES")
    print("  primary models: A2, A3")
    print("  TARGET survival outcomes read: NO")
    print("  GSE21257 outcomes read: NO")
    print("  GSE39055 outcomes read: NO")
    print("  DOG2 outcomes read: NO")
    print("  model fitting / retuning: NO")
    print("  thresholds changed: NO")
    print("  scenario design resampled: NO")
    print("  retained synthetic replicates resampled within fixed scenarios: YES")
    print("  retained 05d model metrics re-estimated from patient predictions: NO")
    print()

    if OUT_DIR.exists():
        raise RuntimeError(
            f"Final 05h3 output already exists; refusing overwrite: {OUT_DIR}"
        )
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir(parents=True, exist_ok=False)

    try:
        verify_h0_contract()
        h2b = verify_05h2b_chain()
        print("Frozen contract chain: PASS")
        print(f"  05h0 SHA256: {sha256_file(H0_JSON)}")
        print(f"  05h2b summary SHA256: {sha256_file(H2B_JSON)}")
        print()

        raw_summary = pd.read_csv(
            SCENARIO_SUMMARY,
            sep="\t",
            low_memory=False,
        )
        design, metrics = load_scenario_design_and_metrics()
        weights = estimand_weights(design)

        print("Frozen scenario-design resolution: PASS")
        print("  resolver: authoritative 05d family + transfer_regime metadata")
        print(f"  scenarios: {len(design)}/180")
        print(
            "  CORE/STRESS: "
            f"{int((design['family_resolved'] == 'CORE').sum())}/"
            f"{int((design['family_resolved'] == 'STRESS').sum())}"
        )
        print(f"  retained replicate total: {int(design['replicates'].sum())}/21600")
        for regime in ["R0", "R1", "R2", "R3", "R4", "R5"]:
            print(
                f"  {regime}: "
                f"{int((design['regime'] == regime).sum())} scenarios"
            )
        print()

        points = point_estimates(metrics, weights)
        components = regime_components(metrics)
        chain_checks = verify_original_against_05h2b(points, h2b)

        print("Deterministic aggregation: PASS")
        print("  frozen_original reproduces accepted 05h2b: YES")
        print("  core_only scenario count: 36")
        print("  regime_balanced regime weights: 1/6 each")
        print()

        npz_map = resolve_npz_files(design)
        model_order, model_order_sources = resolve_model_order(
            raw_summary,
            npz_map,
        )

        print("Retained per-replicate metric source resolution: PASS")
        print(f"  retained 05d NPZ artifacts: {len(npz_map)}/180")
        print(f"  validated model axis: {', '.join(model_order)}")
        for src in model_order_sources:
            print(f"  model-order evidence: {src}")
        print()

        replicate_data, replicate_source_rows = (
            verify_replicate_sources_and_collect(
                design,
                metrics,
                raw_summary,
                npz_map,
                model_order,
            )
        )
        print("Retained-replicate replay against frozen scenario summaries: PASS")
        print("  A2/A3 delta-C replayed in all 180 scenarios: YES")
        print("  A2/A3 negative-transfer rates replayed in all 180 scenarios: YES")
        print("  A2/A3 catastrophic-transfer rates replayed in all 180 scenarios: YES")
        print()

        print(
            f"Running fixed-scenario Monte Carlo resampling "
            f"({MC_BOOTSTRAP_DRAWS:,} within-scenario bootstrap draws)..."
        )
        mc, mc_arrays, seed = bootstrap_fixed_scenarios(
            design,
            replicate_data,
            weights,
        )
        final_table = combine_points_and_mc(points, mc)
        print("Monte Carlo resampling: PASS")
        print(f"  seed derived from frozen 05h0 hash: {seed}")
        print("  scenarios resampled: NO")
        print("  retained replicates resampled within each scenario: YES")
        print()

        print("-" * 118)
        print("05h3 AGGREGATION-SENSITIVITY POINT ESTIMATES")
        print("-" * 118)
        show = points[
            [
                "estimand",
                "model",
                "n_scenarios_with_positive_weight",
                "mean_delta_uno_c",
                "negative_transfer_rate_at_deltaC_le_-0.02",
                "catastrophic_transfer_rate_at_deltaC_le_-0.05",
            ]
        ]
        print(
            show.to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}",
            )
        )
        print()

        print("Interpretation guardrails:")
        print("  frozen_original remains PRIMARY: YES")
        print("  core_only is sensitivity only: YES")
        print("  regime_balanced is sensitivity only: YES")
        print("  frozen primary architecture-selection decision reopened: NO")
        print()

        table_path = (
            WORK_DIR
            / "Supplementary_Table_aggregation_sensitivity.csv"
        )
        component_path = (
            WORK_DIR
            / "Supplementary_Data_regime_component_summaries.csv"
        )
        scenario_path = (
            WORK_DIR
            / "Supplementary_Data_A2_A3_scenario_metrics.csv"
        )
        mc_summary_path = (
            WORK_DIR
            / "Supplementary_Data_aggregation_sensitivity_mc_summary.csv"
        )
        mc_draws_path = (
            WORK_DIR
            / "Supplementary_Data_aggregation_sensitivity_mc_draws.npz"
        )
        source_path = WORK_DIR / "retained_metric_source_inventory.csv"
        checks_path = WORK_DIR / "chain_and_replay_checks.csv"
        json_path = WORK_DIR / "aggregation_sensitivity.json"
        readme_path = WORK_DIR / "README_for_supplement.txt"

        final_table.to_csv(
            table_path,
            index=False,
            float_format="%.10g",
        )
        components.to_csv(
            component_path,
            index=False,
            float_format="%.10g",
        )
        metrics[
            [
                "scenario_id",
                "family_resolved",
                "regime",
                "replicates",
                "model",
                "mean_delta_c_vs_B0",
                "negative_transfer_rate",
                "catastrophic_negative_transfer_rate",
            ]
        ].to_csv(
            scenario_path,
            index=False,
            float_format="%.10g",
        )
        mc.to_csv(
            mc_summary_path,
            index=False,
            float_format="%.10g",
        )
        write_mc_npz(mc_draws_path, mc_arrays, seed)

        pd.DataFrame(replicate_source_rows).to_csv(
            source_path,
            index=False,
        )
        pd.DataFrame(chain_checks).to_csv(
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
            "thresholds": {
                "negative_transfer_delta_uno_c": NEGATIVE_THRESHOLD,
                "catastrophic_transfer_delta_uno_c": CATASTROPHIC_THRESHOLD,
            },
            "estimands": {
                "frozen_original": {
                    "n_scenarios": 180,
                    "weighting": "equal weight per frozen scenario",
                    "role": "PRIMARY; unchanged",
                },
                "core_only": {
                    "n_scenarios": 36,
                    "weighting": "equal weight per frozen CORE scenario",
                    "role": "POST-HOLD robustness sensitivity",
                },
                "regime_balanced": {
                    "n_scenarios": 180,
                    "regimes": ["R0", "R1", "R2", "R3", "R4", "R5"],
                    "weighting": (
                        "equal scenario weight within regime, then equal 1/6 regime weight"
                    ),
                    "role": "POST-HOLD robustness sensitivity",
                },
            },
            "frozen_regime_counts": EXPECTED_REGIME_COUNTS,
            "mc_uncertainty": {
                "method": (
                    "resample retained replicates with replacement within each fixed "
                    "scenario; recompute weighted fixed-design aggregate"
                ),
                "draws": MC_BOOTSTRAP_DRAWS,
                "seed": seed,
                "seed_source": "first 8 hexadecimal digits of frozen 05h0 SHA256",
                "scenario_resampling": False,
                "interval_quantiles": [
                    MC_INTERVAL_LOWER,
                    MC_INTERVAL_UPPER,
                ],
                "interpretation": (
                    "Monte Carlo uncertainty for the fixed frozen scenario design; "
                    "not a superpopulation scenario confidence interval"
                ),
            },
            "validated_npz_model_order": list(model_order),
            "model_order_validation_sources": model_order_sources,
            "point_estimates": points.to_dict(orient="records"),
            "human_outcomes_read": False,
            "model_fitting": False,
            "threshold_change": False,
        }
        write_json(json_path, result_json)

        readme_path.write_text(
            build_readme(
                final_table,
                seed,
                model_order,
                model_order_sources,
            ),
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
                "05h0_contract": {
                    "path": str(H0_JSON.relative_to(ROOT)),
                    "sha256": sha256_file(H0_JSON),
                },
                "05h2b_summary": {
                    "path": str(H2B_JSON.relative_to(ROOT)),
                    "sha256": sha256_file(H2B_JSON),
                },
                "05h2b_manifest": {
                    "path": str(H2B_MANIFEST.relative_to(ROOT)),
                    "sha256": sha256_file(H2B_MANIFEST),
                },
                "05d_scenario_summary": {
                    "path": str(SCENARIO_SUMMARY.relative_to(ROOT)),
                    "sha256": sha256_file(SCENARIO_SUMMARY),
                },
                "05d_metric_contract": {
                    "path": str(METRIC_CONTRACT.relative_to(ROOT)),
                    "sha256": sha256_file(METRIC_CONTRACT),
                },
                "retained_05d_npz_count": len(npz_map),
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
                    f"Output hash changed before finalization: {p.name}"
                )

        os.replace(WORK_DIR, OUT_DIR)

        print("=" * 118)
        print("05h3 post-HOLD aggregation sensitivity: PASS")
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
        print("05h3 post-HOLD aggregation sensitivity: FAIL")
        print(f"{type(exc).__name__}: {exc}")
        print("=" * 118)
        raise
