#!/usr/bin/env python
"""
Paper 6 - freeze technical input/schema contract for 05h3 aggregation sensitivity.

This is a PRE-FLIGHT / TECHNICAL CONTRACT stage only.

Why this stage exists
---------------------
Earlier 05h3 attempts failed because implementation code guessed the meaning or
format of existing labels. This script makes no such guesses. It inspects the
authoritative frozen artifacts first, resolves the exact technical identities
needed by 05h3, validates them against retained per-replicate arrays, and freezes
the resolved schema/mappings before any core-only or regime-balanced scientific
estimate is computed.

It performs NO model fitting, NO retuning, NO human-outcome access, NO bootstrap,
and NO new sensitivity estimate.

Expected placement:
    scripts/05h3c_freeze_aggregation_input_contract.py

Run:
    python scripts\05h3c_freeze_aggregation_input_contract.py
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


SCRIPT_VERSION = "05h3c-freeze-aggregation-input-contract-v1-no-cli"

PRIMARY_MODELS = ["A2", "A3"]
REFERENCE_MODEL = "B0"

NEGATIVE_THRESHOLD = -0.02
CATASTROPHIC_THRESHOLD = -0.05

EXPECTED_SCENARIOS = 180
EXPECTED_CORE_COUNT = 36
EXPECTED_STRESS_COUNT = 144
EXPECTED_REPLICATES_TOTAL = 21600

EXPECTED_REGIME_COUNTS = {
    "R0": 6,
    "R1": 6,
    "R2": 78,
    "R3": 6,
    "R4": 6,
    "R5": 78,
}

# These are exact column names already verified by the earlier 05h2a schema probe.
# We intentionally do not invent aliases here.
REQUIRED_COLUMNS = [
    "scenario_id",
    "family",
    "transfer_regime",
    "replicates",
    "model",
    "mean_delta_c_vs_B0",
    "negative_transfer_rate",
    "catastrophic_negative_transfer_rate",
]

REQUIRED_NPZ_KEYS = [
    "replicate_seed",
    "delta_c_vs_B0",
    "negative_transfer",
    "catastrophic_negative_transfer",
]


def project_root() -> Path:
    p = Path(__file__).resolve()
    if p.parent.name.lower() != "scripts":
        raise RuntimeError(
            "Place 05h3c_freeze_aggregation_input_contract.py in the repository scripts/ directory."
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
METRIC_CONTRACT = D_DIR / "metric_implementation_contract.json"
METRIC_MANIFEST = D_DIR / "scenario_metric_manifest.tsv"

OUT_DIR = ROOT / "method_contract" / "05h3c_aggregation_input_contract"
WORK_DIR = ROOT / "method_contract" / ".05h3c_aggregation_input_contract_work"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def verify_chain() -> tuple[dict[str, Any], dict[str, Any]]:
    for p in [H0_JSON, H2B_JSON, H2B_MANIFEST, SCENARIO_SUMMARY, METRIC_CONTRACT, METRIC_MANIFEST]:
        if not p.exists():
            raise FileNotFoundError(f"Required frozen artifact missing: {p}")

    h0 = json.loads(H0_JSON.read_text(encoding="utf-8"))
    agg = recursive_find_key(h0, "aggregation_sensitivity")
    if not isinstance(agg, dict):
        raise RuntimeError("05h0 lacks aggregation_sensitivity.")

    if agg.get("analysis_role") != "POST-HOLD ROBUSTNESS ANALYSIS":
        raise RuntimeError("Unexpected 05h0 aggregation-sensitivity analysis role.")
    if agg.get("primary_models") != PRIMARY_MODELS:
        raise RuntimeError(
            f"05h0 primary models changed: {agg.get('primary_models')!r}"
        )

    estimands = agg.get("estimands", {})
    if set(estimands) != {"frozen_original", "core_only", "regime_balanced"}:
        raise RuntimeError(
            f"05h0 estimand registry changed: {sorted(estimands)!r}"
        )

    core_text = str(estimands["core_only"].get("scenario_set", ""))
    if "36" not in core_text:
        raise RuntimeError("05h0 no longer requires exactly 36 core scenarios.")

    rb_regimes = estimands["regime_balanced"].get("regimes")
    if rb_regimes != ["R0", "R1", "R2", "R3", "R4", "R5"]:
        raise RuntimeError(
            f"05h0 regime registry changed: {rb_regimes!r}"
        )

    metrics = agg.get("minimum_reported_metrics")
    if metrics != [
        "mean_delta_uno_c",
        "negative_transfer_rate_at_deltaC_le_-0.02",
        "catastrophic_transfer_rate_at_deltaC_le_-0.05",
    ]:
        raise RuntimeError(
            f"05h0 minimum metrics changed: {metrics!r}"
        )

    h2b = json.loads(H2B_JSON.read_text(encoding="utf-8"))
    if h2b.get("status") != "PASS_FROZEN_ALL_MODEL_SAFETY_SUMMARY_EXPORTED":
        raise RuntimeError(f"05h2b is not PASS: {h2b.get('status')!r}")
    if bool(h2b.get("primary_decision_changed")):
        raise RuntimeError("05h2b says the frozen primary decision changed.")
    if h2b.get("eligible_for_primary_selection") != PRIMARY_MODELS:
        raise RuntimeError(
            "05h2b selectable-model registry is not exactly A2/A3."
        )

    return h0, h2b


def read_and_validate_schema() -> pd.DataFrame:
    df = pd.read_csv(SCENARIO_SUMMARY, sep="\t", low_memory=False)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(
            "Authoritative scenario summary schema does not match the already-probed 05d schema. "
            f"Missing: {missing}"
        )

    print("Authoritative 05d schema binding: PASS")
    for semantic_name in REQUIRED_COLUMNS:
        print(f"  {semantic_name}: column exists exactly")

    print(f"  rows: {len(df)}")
    print(f"  columns: {len(df.columns)}")
    print()

    return df


def resolve_scenario_design(df: pd.DataFrame) -> pd.DataFrame:
    # Require all model rows for a scenario to agree on the design metadata.
    design_cols = ["scenario_id", "family", "transfer_regime", "replicates"]
    design = df[design_cols].drop_duplicates().copy()

    if design["scenario_id"].duplicated().any():
        bad = design[
            design["scenario_id"].duplicated(keep=False)
        ].sort_values("scenario_id")
        raise RuntimeError(
            "Frozen design metadata are inconsistent across model rows:\n"
            + bad.head(30).to_string(index=False)
        )

    if len(design) != EXPECTED_SCENARIOS:
        raise RuntimeError(
            f"Expected 180 unique frozen scenarios; observed {len(design)}."
        )

    sid = design["scenario_id"].astype(str).str.strip()

    # No regex. The existing scenario IDs themselves define the binary class.
    is_core = sid.str.startswith("CORE_")
    is_stress = sid.str.startswith("STRESS_")

    if ((is_core.astype(int) + is_stress.astype(int)) != 1).any():
        bad = design.loc[
            (is_core.astype(int) + is_stress.astype(int)) != 1,
            "scenario_id",
        ].tolist()
        raise RuntimeError(
            "Some frozen scenario IDs are neither uniquely CORE_* nor STRESS_*: "
            f"{bad[:20]}"
        )

    design["scenario_class"] = np.where(is_core, "CORE", "STRESS")

    n_core = int((design["scenario_class"] == "CORE").sum())
    n_stress = int((design["scenario_class"] == "STRESS").sum())
    if n_core != EXPECTED_CORE_COUNT or n_stress != EXPECTED_STRESS_COUNT:
        raise RuntimeError(
            f"CORE/STRESS count mismatch: observed {n_core}/{n_stress}, "
            f"expected {EXPECTED_CORE_COUNT}/{EXPECTED_STRESS_COUNT}."
        )

    # Regime is resolved directly from the existing transfer_regime label.
    # Example: R0_FULLY_TRANSPORTABLE -> R0.
    transfer = design["transfer_regime"].astype(str).str.strip()
    regime = transfer.str.split("_", n=1).str[0]

    allowed_regimes = set(EXPECTED_REGIME_COUNTS)
    observed_regimes = set(regime)
    if observed_regimes != allowed_regimes:
        raise RuntimeError(
            f"Frozen regime prefixes changed: observed={sorted(observed_regimes)}, "
            f"expected={sorted(allowed_regimes)}."
        )
    design["regime"] = regime

    regime_counts = (
        design["regime"]
        .value_counts()
        .reindex(["R0", "R1", "R2", "R3", "R4", "R5"], fill_value=0)
        .astype(int)
        .to_dict()
    )
    if regime_counts != EXPECTED_REGIME_COUNTS:
        raise RuntimeError(
            f"Frozen regime counts changed: {regime_counts}"
        )

    design["replicates"] = pd.to_numeric(
        design["replicates"], errors="raise"
    ).astype(int)
    if (design["replicates"] <= 0).any():
        raise RuntimeError("Non-positive replicate count in frozen design.")

    total_reps = int(design["replicates"].sum())
    if total_reps != EXPECTED_REPLICATES_TOTAL:
        raise RuntimeError(
            f"Frozen replicate total changed: observed={total_reps}, "
            f"expected={EXPECTED_REPLICATES_TOTAL}."
        )

    # family is descriptive provenance only. We deliberately do not parse it.
    family_counts = (
        design["family"]
        .astype(str)
        .value_counts(dropna=False)
        .sort_index()
    )

    print("Frozen scenario-design preflight: PASS")
    print(f"  scenarios: {len(design)}/180")
    print(f"  CORE via scenario_id prefix: {n_core}/36")
    print(f"  STRESS via scenario_id prefix: {n_stress}/144")
    print(f"  retained replicate total from design: {total_reps}/21600")
    print("  descriptive family labels observed (NOT coerced):")
    for label, count in family_counts.items():
        print(f"    {label}: {int(count)}")
    print("  normalized regime counts:")
    for r in ["R0", "R1", "R2", "R3", "R4", "R5"]:
        print(f"    {r}: {regime_counts[r]}")
    print()

    return design.sort_values("scenario_id").reset_index(drop=True)


def validate_primary_metric_rows(
    df: pd.DataFrame,
    design: pd.DataFrame,
) -> pd.DataFrame:
    models = list(dict.fromkeys(df["model"].astype(str).tolist()))
    unique_models = set(models)

    for m in PRIMARY_MODELS + [REFERENCE_MODEL]:
        if m not in unique_models:
            raise RuntimeError(f"Frozen scenario summary lacks required model {m}.")

    primary = df[df["model"].isin(PRIMARY_MODELS)].copy()
    if len(primary) != EXPECTED_SCENARIOS * len(PRIMARY_MODELS):
        raise RuntimeError(
            f"Expected 360 A2/A3 scenario rows; found {len(primary)}."
        )
    if primary.duplicated(["scenario_id", "model"]).any():
        raise RuntimeError("A2/A3 scenario rows are not unique.")

    for col in [
        "mean_delta_c_vs_B0",
        "negative_transfer_rate",
        "catastrophic_negative_transfer_rate",
    ]:
        primary[col] = pd.to_numeric(primary[col], errors="coerce")
        if not np.isfinite(primary[col].to_numpy(dtype=float)).all():
            raise RuntimeError(f"Non-finite primary frozen metric in {col}.")

    # Attach exact preflight-resolved design identities.
    primary = primary.merge(
        design[
            [
                "scenario_id",
                "scenario_class",
                "regime",
            ]
        ],
        on="scenario_id",
        how="left",
        validate="many_to_one",
    )
    if primary[["scenario_class", "regime"]].isna().any().any():
        raise RuntimeError("Failed to attach preflight design identities to A2/A3.")

    print("Primary A2/A3 metric-row preflight: PASS")
    print("  A2 rows: 180")
    print("  A3 rows: 180")
    print("  required existing metrics:")
    print("    mean_delta_c_vs_B0")
    print("    negative_transfer_rate")
    print("    catastrophic_negative_transfer_rate")
    print()

    return primary


def map_npz_files_to_scenarios(design: pd.DataFrame) -> dict[str, Path]:
    scenario_ids = set(design["scenario_id"].astype(str))
    npz_files = sorted(D_DIR.rglob("*.npz"))

    if len(npz_files) != EXPECTED_SCENARIOS:
        raise RuntimeError(
            f"Expected exactly 180 retained 05d NPZ files; observed {len(npz_files)}."
        )

    mapping: dict[str, Path] = {}
    for path in npz_files:
        upper = str(path).upper()
        matches = [sid for sid in scenario_ids if sid.upper() in upper]
        if len(matches) != 1:
            raise RuntimeError(
                f"Cannot map retained NPZ uniquely to one frozen scenario: {path}; "
                f"matches={matches[:10]}"
            )
        sid = matches[0]
        if sid in mapping:
            raise RuntimeError(
                f"Two NPZ files map to the same scenario {sid}: "
                f"{mapping[sid]} and {path}"
            )
        mapping[sid] = path

    if set(mapping) != scenario_ids:
        missing = sorted(scenario_ids - set(mapping))
        extra = sorted(set(mapping) - scenario_ids)
        raise RuntimeError(
            f"NPZ scenario-set mismatch. Missing={missing[:10]}, extra={extra[:10]}"
        )

    print("Retained NPZ file mapping preflight: PASS")
    print(f"  NPZ files: {len(mapping)}/180")
    print(f"  example first: {mapping[sorted(mapping)[0]].relative_to(ROOT)}")
    print(f"  example last:  {mapping[sorted(mapping)[-1]].relative_to(ROOT)}")
    print()

    return mapping


def string_model_order_candidates(
    obj: Any,
    exact_model_set: set[str],
) -> list[tuple[str, tuple[str, ...]]]:
    out: list[tuple[str, tuple[str, ...]]] = []

    def walk(x: Any, label: str) -> None:
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, f"{label}.{k}" if label else str(k))
        elif isinstance(x, list):
            if (
                len(x) == len(exact_model_set)
                and all(isinstance(v, str) for v in x)
                and set(x) == exact_model_set
            ):
                out.append((label, tuple(x)))
            for i, v in enumerate(x):
                walk(v, f"{label}[{i}]")

    walk(obj, "")
    return out


def candidate_model_orders(
    df: pd.DataFrame,
    npz_map: dict[str, Path],
) -> list[tuple[str, tuple[str, ...]]]:
    exact_model_set = set(df["model"].astype(str).unique())
    candidates: list[tuple[str, tuple[str, ...]]] = []

    # Candidate 1: explicit string model order inside NPZ, if stored.
    first_path = npz_map[sorted(npz_map)[0]]
    with np.load(first_path, allow_pickle=False) as data:
        for key in data.files:
            arr = np.asarray(data[key])
            if (
                arr.ndim == 1
                and len(arr) == len(exact_model_set)
                and arr.dtype.kind in {"U", "S"}
            ):
                seq = tuple(
                    v.decode("utf-8") if isinstance(v, bytes) else str(v)
                    for v in arr.tolist()
                )
                if set(seq) == exact_model_set:
                    candidates.append((f"NPZ:{key}", seq))

    # Candidate 2: exact list inside frozen metric contract.
    contract = json.loads(METRIC_CONTRACT.read_text(encoding="utf-8"))
    for label, seq in string_model_order_candidates(contract, exact_model_set):
        candidates.append((f"metric_contract:{label}", seq))

    # Candidate 3: stable row order already present in authoritative frozen summary.
    orders = []
    for _, part in df.groupby("scenario_id", sort=False):
        seq = tuple(part["model"].astype(str).tolist())
        if len(seq) != len(exact_model_set) or set(seq) != exact_model_set:
            raise RuntimeError(
                "A scenario does not contain exactly one row for every frozen model."
            )
        orders.append(seq)

    if orders and all(seq == orders[0] for seq in orders):
        candidates.append(
            ("authoritative_scenario_summary_stable_row_order", orders[0])
        )

    # Deduplicate identical sequence while retaining evidence labels.
    seen: set[tuple[str, ...]] = set()
    deduped = []
    for label, seq in candidates:
        if seq not in seen:
            deduped.append((label, seq))
            seen.add(seq)

    return deduped


def validate_model_order(
    seq: tuple[str, ...],
    df: pd.DataFrame,
    npz_map: dict[str, Path],
) -> tuple[bool, float]:
    if not seq:
        return False, float("inf")

    model_index = {m: i for i, m in enumerate(seq)}
    if any(m not in model_index for m in PRIMARY_MODELS):
        return False, float("inf")

    lookup = df.set_index(["scenario_id", "model"])
    max_abs_delta_diff = 0.0

    for sid in sorted(npz_map):
        path = npz_map[sid]
        with np.load(path, allow_pickle=False) as data:
            if "delta_c_vs_B0" not in data.files:
                return False, float("inf")
            delta_raw = np.asarray(data["delta_c_vs_B0"])

        if delta_raw.ndim != 2 or delta_raw.shape[1] != len(seq):
            return False, float("inf")

        for model in PRIMARY_MODELS:
            idx = model_index[model]
            arr = delta_raw[:, idx]
            observed_native = float(np.nanmean(arr))
            expected = float(lookup.loc[(sid, model), "mean_delta_c_vs_B0"])
            diff = abs(observed_native - expected)
            max_abs_delta_diff = max(max_abs_delta_diff, diff)

            # Accept either near-exact numeric equality or exact float32
            # serialization equivalence. This is only axis/provenance validation.
            numeric_ok = diff <= 1e-7
            float32_ok = (
                np.float32(observed_native).tobytes()
                == np.float32(expected).tobytes()
            )
            if not (numeric_ok or float32_ok):
                return False, max_abs_delta_diff

    return True, max_abs_delta_diff


def resolve_model_order(
    df: pd.DataFrame,
    npz_map: dict[str, Path],
) -> tuple[tuple[str, ...], list[str], float]:
    candidates = candidate_model_orders(df, npz_map)

    if not candidates:
        raise RuntimeError(
            "No model-axis order candidate could be resolved from existing frozen artifacts."
        )

    validated: dict[tuple[str, ...], dict[str, Any]] = {}
    for label, seq in candidates:
        ok, max_diff = validate_model_order(seq, df, npz_map)
        if ok:
            entry = validated.setdefault(
                seq,
                {"labels": [], "max_diff": max_diff},
            )
            entry["labels"].append(label)
            entry["max_diff"] = min(entry["max_diff"], max_diff)

    if not validated:
        attempted = [label for label, _ in candidates]
        raise RuntimeError(
            "No candidate model-axis order reproduced A2/A3 frozen scenario means "
            "across all 180 retained NPZ files. Attempted: "
            + ", ".join(attempted)
        )

    if len(validated) != 1:
        detail = "\n".join(
            f"  {seq}: {meta['labels']}"
            for seq, meta in validated.items()
        )
        raise RuntimeError(
            "More than one non-equivalent model-axis order validated:\n" + detail
        )

    seq, meta = next(iter(validated.items()))

    print("Retained NPZ model-axis preflight: PASS")
    print("  resolved model order:")
    print("    " + ", ".join(seq))
    print("  evidence source(s):")
    for label in meta["labels"]:
        print(f"    {label}")
    print(
        "  max |retained NPZ A2/A3 scenario mean delta - frozen table|: "
        f"{meta['max_diff']:.3e}"
    )
    print()

    return seq, list(meta["labels"]), float(meta["max_diff"])


def validate_all_npz_metrics(
    design: pd.DataFrame,
    primary: pd.DataFrame,
    npz_map: dict[str, Path],
    model_order: tuple[str, ...],
) -> dict[str, Any]:
    model_index = {m: i for i, m in enumerate(model_order)}
    lookup = primary.set_index(["scenario_id", "model"])

    max_delta_diff = 0.0
    max_nt_diff = 0.0
    max_cat_diff = 0.0
    flag_identity_failures = 0

    source_rows = []

    for row in design.itertuples(index=False):
        sid = str(row.scenario_id)
        path = npz_map[sid]
        expected_n = int(row.replicates)

        with np.load(path, allow_pickle=False) as data:
            missing = [k for k in REQUIRED_NPZ_KEYS if k not in data.files]
            if missing:
                raise RuntimeError(
                    f"{sid}: retained NPZ is missing required keys {missing}."
                )

            seeds = np.asarray(data["replicate_seed"])
            delta = np.asarray(data["delta_c_vs_B0"])
            neg = np.asarray(data["negative_transfer"])
            cat = np.asarray(data["catastrophic_negative_transfer"])

        expected_shape = (expected_n, len(model_order))
        for name, arr in [
            ("delta_c_vs_B0", delta),
            ("negative_transfer", neg),
            ("catastrophic_negative_transfer", cat),
        ]:
            if arr.shape != expected_shape:
                raise RuntimeError(
                    f"{sid}: {name} shape={arr.shape}, expected={expected_shape}."
                )

        if seeds.shape != (expected_n,):
            raise RuntimeError(
                f"{sid}: replicate_seed shape={seeds.shape}, expected={(expected_n,)}."
            )
        if len(np.unique(seeds)) != expected_n:
            raise RuntimeError(f"{sid}: replicate seeds are not unique.")

        # Threshold flags must be exactly the frozen definitions already in 05d.
        neg_from_delta = (delta <= NEGATIVE_THRESHOLD).astype(np.uint8)
        cat_from_delta = (delta <= CATASTROPHIC_THRESHOLD).astype(np.uint8)

        if not np.array_equal(neg.astype(np.uint8), neg_from_delta):
            flag_identity_failures += 1
        if not np.array_equal(cat.astype(np.uint8), cat_from_delta):
            flag_identity_failures += 1

        for model in PRIMARY_MODELS:
            idx = model_index[model]
            frozen = lookup.loc[(sid, model)]

            observed_delta = float(np.nanmean(delta[:, idx]))
            observed_nt = float(np.mean(neg[:, idx]))
            observed_cat = float(np.mean(cat[:, idx]))

            expected_delta = float(frozen["mean_delta_c_vs_B0"])
            expected_nt = float(frozen["negative_transfer_rate"])
            expected_cat = float(
                frozen["catastrophic_negative_transfer_rate"]
            )

            max_delta_diff = max(
                max_delta_diff,
                abs(observed_delta - expected_delta),
            )
            max_nt_diff = max(
                max_nt_diff,
                abs(observed_nt - expected_nt),
            )
            max_cat_diff = max(
                max_cat_diff,
                abs(observed_cat - expected_cat),
            )

            delta_ok = (
                abs(observed_delta - expected_delta) <= 1e-7
                or np.float32(observed_delta).tobytes()
                == np.float32(expected_delta).tobytes()
            )
            if not delta_ok:
                raise RuntimeError(
                    f"{sid}/{model}: retained delta replay mismatch: "
                    f"{observed_delta:.12g} vs {expected_delta:.12g}"
                )
            if abs(observed_nt - expected_nt) > 1e-12:
                raise RuntimeError(
                    f"{sid}/{model}: retained NT-rate replay mismatch: "
                    f"{observed_nt:.12g} vs {expected_nt:.12g}"
                )
            if abs(observed_cat - expected_cat) > 1e-12:
                raise RuntimeError(
                    f"{sid}/{model}: retained catastrophic-rate replay mismatch: "
                    f"{observed_cat:.12g} vs {expected_cat:.12g}"
                )

        source_rows.append(
            {
                "scenario_id": sid,
                "scenario_class": str(row.scenario_class),
                "family": str(row.family),
                "regime": str(row.regime),
                "replicates": expected_n,
                "npz_relative_path": str(path.relative_to(ROOT)),
                "npz_sha256": sha256_file(path),
                "npz_bytes": path.stat().st_size,
            }
        )

    if flag_identity_failures != 0:
        raise RuntimeError(
            f"Frozen threshold-flag identity failed in {flag_identity_failures} NPZ checks."
        )

    print("Full retained-metric preflight replay: PASS")
    print("  180/180 NPZ shapes: PASS")
    print("  replicate-seed uniqueness: PASS")
    print("  negative_transfer == (delta <= -0.02): EXACT")
    print("  catastrophic_negative_transfer == (delta <= -0.05): EXACT")
    print(
        f"  max A2/A3 scenario mean-delta replay difference: {max_delta_diff:.3e}"
    )
    print(
        f"  max A2/A3 scenario NT-rate replay difference: {max_nt_diff:.3e}"
    )
    print(
        f"  max A2/A3 scenario catastrophic-rate replay difference: {max_cat_diff:.3e}"
    )
    print()

    return {
        "max_delta_replay_difference": max_delta_diff,
        "max_nt_rate_replay_difference": max_nt_diff,
        "max_catastrophic_rate_replay_difference": max_cat_diff,
        "threshold_flag_identity_exact": True,
        "source_rows": source_rows,
    }


def frozen_mapping_hashes(design: pd.DataFrame) -> dict[str, str]:
    core_ids = sorted(
        design.loc[
            design["scenario_class"] == "CORE",
            "scenario_id",
        ].astype(str)
    )
    stress_ids = sorted(
        design.loc[
            design["scenario_class"] == "STRESS",
            "scenario_id",
        ].astype(str)
    )

    regime_rows = (
        design[["scenario_id", "regime"]]
        .sort_values("scenario_id")
        .astype(str)
    )
    regime_text = regime_rows.to_csv(
        index=False,
        lineterminator="\n",
    )

    return {
        "core_scenario_set_sha256": sha256_text("\n".join(core_ids) + "\n"),
        "stress_scenario_set_sha256": sha256_text("\n".join(stress_ids) + "\n"),
        "scenario_to_regime_mapping_sha256": sha256_text(regime_text),
    }


def main() -> None:
    print("=" * 118)
    print("Paper 6 - 05h3c technical preflight / aggregation-input contract")
    print("=" * 118)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {ROOT}")
    print(f"Output directory: {OUT_DIR}")
    print()
    print("Execution scope:")
    print("  core-only scientific estimate computed: NO")
    print("  regime-balanced scientific estimate computed: NO")
    print("  Monte Carlo/bootstrap computed: NO")
    print("  model fitting / retuning: NO")
    print("  human outcomes read: NO")
    print("  existing frozen 05d schema inspected: YES")
    print("  existing retained NPZ metric arrays inspected: YES")
    print("  exact technical bindings frozen before 05h3 calculation: YES")
    print()

    if OUT_DIR.exists():
        raise RuntimeError(
            f"05h3c contract output already exists; refusing overwrite: {OUT_DIR}"
        )
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir(parents=True, exist_ok=False)

    try:
        _, h2b = verify_chain()
        print("Frozen contract chain preflight: PASS")
        print(f"  05h0 SHA256: {sha256_file(H0_JSON)}")
        print(f"  05h2b summary SHA256: {sha256_file(H2B_JSON)}")
        print(f"  05h2b manifest SHA256: {sha256_file(H2B_MANIFEST)}")
        print()

        df = read_and_validate_schema()
        design = resolve_scenario_design(df)
        primary = validate_primary_metric_rows(df, design)

        npz_map = map_npz_files_to_scenarios(design)
        model_order, model_order_evidence, order_max_diff = resolve_model_order(
            df,
            npz_map,
        )

        replay = validate_all_npz_metrics(
            design,
            primary,
            npz_map,
            model_order,
        )

        mapping_hashes = frozen_mapping_hashes(design)

        # Save exact technical contract only after all preflight checks pass.
        core_ids = sorted(
            design.loc[
                design["scenario_class"] == "CORE",
                "scenario_id",
            ].astype(str)
        )
        stress_ids = sorted(
            design.loc[
                design["scenario_class"] == "STRESS",
                "scenario_id",
            ].astype(str)
        )
        regime_map = (
            design[["scenario_id", "regime"]]
            .sort_values("scenario_id")
            .to_dict(orient="records")
        )
        family_counts = (
            design["family"]
            .astype(str)
            .value_counts()
            .sort_index()
            .astype(int)
            .to_dict()
        )

        contract = {
            "script_version": SCRIPT_VERSION,
            "status": "PASS_READY_FOR_05H3D_AGGREGATION_SENSITIVITY",
            "analysis_role": "TECHNICAL PREFLIGHT / INPUT CONTRACT ONLY",
            "scientific_estimate_computed": False,
            "frozen_primary_decision_changed": False,
            "sources": {
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
                "05d_metric_manifest": {
                    "path": str(METRIC_MANIFEST.relative_to(ROOT)),
                    "sha256": sha256_file(METRIC_MANIFEST),
                },
            },
            "exact_column_bindings": {
                "scenario_id": "scenario_id",
                "descriptive_family": "family",
                "transfer_regime": "transfer_regime",
                "replicate_count": "replicates",
                "model": "model",
                "mean_delta_uno_c": "mean_delta_c_vs_B0",
                "negative_transfer_rate": "negative_transfer_rate",
                "catastrophic_transfer_rate": "catastrophic_negative_transfer_rate",
            },
            "scenario_class_rule": {
                "resolver": "existing scenario_id string prefix only; no regex",
                "core_prefix": "CORE_",
                "stress_prefix": "STRESS_",
                "n_core": EXPECTED_CORE_COUNT,
                "n_stress": EXPECTED_STRESS_COUNT,
                "core_scenario_ids": core_ids,
                "stress_scenario_ids": stress_ids,
                "core_scenario_set_sha256": mapping_hashes[
                    "core_scenario_set_sha256"
                ],
                "stress_scenario_set_sha256": mapping_hashes[
                    "stress_scenario_set_sha256"
                ],
            },
            "descriptive_family_labels": family_counts,
            "regime_rule": {
                "resolver": (
                    "prefix before first underscore in existing transfer_regime label"
                ),
                "counts": EXPECTED_REGIME_COUNTS,
                "scenario_to_regime": regime_map,
                "scenario_to_regime_mapping_sha256": mapping_hashes[
                    "scenario_to_regime_mapping_sha256"
                ],
            },
            "replicates": {
                "n_scenarios": EXPECTED_SCENARIOS,
                "total_replicates": EXPECTED_REPLICATES_TOTAL,
                "retained_npz_count": len(npz_map),
            },
            "retained_metric_schema": {
                "required_npz_keys": REQUIRED_NPZ_KEYS,
                "model_axis_order": list(model_order),
                "model_axis_evidence": model_order_evidence,
                "model_axis_validation_max_abs_delta_difference": order_max_diff,
                "A2_index": int(model_order.index("A2")),
                "A3_index": int(model_order.index("A3")),
            },
            "frozen_thresholds": {
                "negative_transfer_delta_uno_c": NEGATIVE_THRESHOLD,
                "catastrophic_transfer_delta_uno_c": CATASTROPHIC_THRESHOLD,
                "retained_flags_match_thresholds_exactly": True,
            },
            "replay": {
                "A2_A3_all_180_scenarios": "PASS",
                "max_delta_replay_difference": replay[
                    "max_delta_replay_difference"
                ],
                "max_nt_rate_replay_difference": replay[
                    "max_nt_rate_replay_difference"
                ],
                "max_catastrophic_rate_replay_difference": replay[
                    "max_catastrophic_rate_replay_difference"
                ],
            },
            "05h2b_chain": {
                "status": h2b.get("status"),
                "primary_models": PRIMARY_MODELS,
                "primary_decision_changed": False,
            },
            "next_stage": (
                "05h3d may compute frozen_original, core_only, and regime_balanced "
                "only after re-verifying every source hash and this exact contract."
            ),
        }

        contract_path = WORK_DIR / "aggregation_input_contract.json"
        write_json(contract_path, contract)

        design_export = design[
            [
                "scenario_id",
                "scenario_class",
                "family",
                "transfer_regime",
                "regime",
                "replicates",
            ]
        ].copy()
        design_export.to_csv(
            WORK_DIR / "resolved_frozen_scenario_design.tsv",
            sep="\t",
            index=False,
        )

        pd.DataFrame(replay["source_rows"]).to_csv(
            WORK_DIR / "retained_npz_source_inventory.csv",
            index=False,
        )

        readme = f"""Paper 6 - 05h3c aggregation-input technical preflight

Status
------
PASS_READY_FOR_05H3D_AGGREGATION_SENSITIVITY

This stage computes no core-only or regime-balanced scientific estimate.

Resolved identities
-------------------
Scenario table:
{SCENARIO_SUMMARY.relative_to(ROOT)}

CORE/STRESS:
Resolved only from literal existing scenario_id prefixes CORE_ and STRESS_.
No regex and no interpretation of the descriptive `family` column.
Counts: 36 CORE, 144 STRESS.

Descriptive family labels:
{json.dumps(family_counts, indent=2, sort_keys=True)}

Regimes:
Resolved from the prefix before the first underscore in transfer_regime.
Counts: {EXPECTED_REGIME_COUNTS}

Retained replicate arrays:
180 NPZ files, 21,600 frozen replicates total.
Required keys: {", ".join(REQUIRED_NPZ_KEYS)}

Validated model axis:
{", ".join(model_order)}
A2 index: {model_order.index("A2")}
A3 index: {model_order.index("A3")}

Threshold identity:
negative_transfer == (delta_c_vs_B0 <= -0.02): exact
catastrophic_negative_transfer == (delta_c_vs_B0 <= -0.05): exact

A2/A3 retained-array replay:
all 180 scenarios PASS.
max mean-delta difference = {replay["max_delta_replay_difference"]:.3e}
max NT-rate difference = {replay["max_nt_rate_replay_difference"]:.3e}
max catastrophic-rate difference = {replay["max_catastrophic_rate_replay_difference"]:.3e}

Next
----
A new 05h3d computation script should read this contract and refuse to run if
any source hash, column binding, scenario membership, regime mapping, NPZ count,
model-axis identity, or frozen threshold identity changes.
"""
        (WORK_DIR / "README.txt").write_text(
            readme,
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
            "status": "PASS_READY_FOR_05H3D_AGGREGATION_SENSITIVITY",
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
                    f"Preflight output changed before commit: {p.name}"
                )

        os.replace(WORK_DIR, OUT_DIR)

        print("=" * 118)
        print("05h3c aggregation-input technical preflight: PASS")
        print("=" * 118)
        print("Scientific sensitivity estimates computed: NO")
        print("Bootstrap performed: NO")
        print("Ready for 05h3d computation: YES")
        print()
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
        print("05h3c aggregation-input technical preflight: FAIL")
        print(f"{type(exc).__name__}: {exc}")
        print("=" * 118)
        raise
