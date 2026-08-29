#!/usr/bin/env python3
"""
Paper 6 - bounded post-gate diagnostics for DOG2 Source Prognostic Gate.

Purpose
-------
03b produced the first central Paper-6 result:
SOURCE_AMBER_WEAK_AND_CONTEXT_SENSITIVE.

This script is NOT a new source-model search and MUST NOT change any frozen gate.
It performs only bounded descriptive diagnostics needed to interpret the already
observed COTC021<->COTC022 asymmetry and to document provenance.

It answers:
1. How many OS/DFI events and censored observations are in each randomized arm?
2. How do follow-up/censoring distributions and the frozen directional Uno-C tau
   values differ by arm?
3. Were any of the four arm->arm ridge-Cox models numerically degenerate?
4. Does the directional pattern persist when BOTH directions are evaluated at one
   deterministic common tau = min(direction-specific frozen tau), without refit?
5. Was reuse of v1 OS artifacts by 03b v2 scientifically safe?
6. What can be said, from already-frozen 02g files only, about the 7,041 unresolved
   DOG2 feature names?
7. What is the pre-human action if the later premise test is INCONCLUSIVE_NEUTRAL?

Guardrails
----------
- DOG2 outcome/expression values: YES, already opened by 03b.
- Human outcome/clinical values: NO.
- New feature selection: NO.
- New hyperparameter tuning: NO.
- New source model family: NO.
- Source Gate reclassification: NO.
- Permutation count increase: NO.
- Network access: NO.
- Common-tau analysis: NON-GATING descriptive sensitivity only.
- Arm models are refit ONCE using the exact alpha already selected/saved by 03b;
  no tuning or model choice is repeated.

No CLI arguments.
"""

from __future__ import annotations

import ast
import difflib
import hashlib
import importlib.util
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from sksurv.metrics import concordance_index_ipcw
    from sksurv.nonparametric import kaplan_meier_estimator
except ImportError as exc:
    raise ImportError(
        "03c requires scikit-survival in the active Paper-6 .venv. "
        "03b already required the same dependency."
    ) from exc


SCRIPT_VERSION = "03c-bounded-source-gate-diagnostics-v1-no-cli"

ROOT = Path(__file__).resolve().parents[1]

# Frozen design/protocol.
I_DIR = ROOT / "results" / "revised_design" / "02i"
I_CONTRACT = I_DIR / "revised_model_benchmark_source_gate_contract.json"
I_SUMMARY = I_DIR / "summary.json"

A_DIR = ROOT / "results" / "source_gate_protocol" / "03a"
A_PROTOCOL = A_DIR / "source_prognostic_gate_protocol.json"
A_SUMMARY = A_DIR / "summary.json"

# Completed source gate.
B_DIR = ROOT / "results" / "source_gate" / "03b"
B_SUMMARY = B_DIR / "summary.json"
B_RESULTS = B_DIR / "source_gate_results.json"
B_OS_ARM = B_DIR / "os_arm_to_arm_metrics.tsv"
B_DFI_ARM = B_DIR / "dfi_arm_to_arm_metrics.tsv"

B_CHECKPOINT = B_DIR / "checkpoints"
B_OS_PRIMARY_STAGE = B_CHECKPOINT / "os_primary_random_cv_stage.json"
B_OS_SUPPORT_STAGE = B_CHECKPOINT / "os_supporting_gene_stage.json"
B_OS_ARM_STAGE = B_CHECKPOINT / "os_arm_stage.json"

# Script implementations used for provenance.
B_V1_SCRIPT = ROOT / "scripts" / "03b_run_dog2_source_prognostic_gate.py"
B_V2_SCRIPT = ROOT / "scripts" / "03b_run_dog2_source_prognostic_gate_v2.py"

# 02g outcome-blind mapping files.
G_DIR = ROOT / "results" / "ortholog_bridge" / "02g_v3"
G_RESOLUTION = G_DIR / "dog2_feature_symbol_resolution.tsv"
G_RAW = G_DIR / "ensembl_current_dog_human_biomart_raw.tsv"
G_SUMMARY = G_DIR / "summary.json"

UPSTREAM_LOCK = ROOT / "contracts" / "00_upstream_input_lock.json"

OUT_DIR = ROOT / "results" / "source_gate_diagnostics" / "03c"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ARM_FOLLOWUP = OUT_DIR / "arm_endpoint_followup_summary.tsv"
CENSORING_KM = OUT_DIR / "reverse_km_censoring_curves.tsv"
DIRECTION_DIAG = OUT_DIR / "directional_arm_model_diagnostics.tsv"
COMMON_TAU = OUT_DIR / "common_tau_non_gating_sensitivity.tsv"
REUSE_JSON = OUT_DIR / "03b_v1_v2_reuse_provenance.json"
REUSE_DIFF = OUT_DIR / "03b_v1_v2_unified_diff.txt"
TRIAL_JSON = OUT_DIR / "trial_design_provenance.json"
UNRESOLVED_TSV = OUT_DIR / "unresolved_feature_annotation_audit.tsv"
UNRESOLVED_SUMMARY = OUT_DIR / "unresolved_feature_annotation_summary.tsv"
PREMISE_POLICY = OUT_DIR / "premise_test_action_policy_addendum.json"
DIAGNOSTIC_JSON = OUT_DIR / "bounded_diagnostic_record.json"
SUMMARY_JSON = OUT_DIR / "summary.json"

EXPECTED_03B_STATUS = "SOURCE_AMBER_WEAK_AND_CONTEXT_SENSITIVE"
EXPECTED_N = 186
EXPECTED_ARMS = {"COTC021": 93, "COTC022": 93}
EXPECTED_OS_EVENTS = 124
EXPECTED_DFI_EVENTS = 143
EXPECTED_UNRESOLVED = 7041

DIRECTIONS = [
    ("COTC021", "COTC022"),
    ("COTC022", "COTC021"),
]

# Numerical diagnostic only; this DOES NOT define a scientific gate.
DEGENERACY_SD_EPS = 1e-8
DEGENERACY_COEF_NORM_EPS = 1e-8

CORE_AST_NAMES = [
    "HallmarkTransformer",
    "GeneTransformer",
    "survival_array",
    "tau_from_training",
    "safe_uno_c",
    "stratified_splits",
    "fit_ridge",
    "tune_ridge",
    "primary_cv",
    "aggregate_cv_predictions",
    "repeat_metrics",
    "patient_bootstrap_uno",
    "bootstrap_summary",
    "fit_coxnet_path",
    "tune_elastic_net",
    "supporting_gene_cv",
    "arm_direction",
    "run_primary_stage",
    "run_support_stage",
    "run_arm_stage",
    "run_permutations",
]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "na", "n/a"}:
        return ""
    return text


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def require_file(path: Path) -> Path:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Required artifact missing: {path}")
    return path


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def safe_uno_at_fixed_tau(
    y_train: np.ndarray,
    y_test: np.ndarray,
    risk: np.ndarray,
    tau: float,
) -> float:
    risk = np.asarray(risk, dtype=float)

    try:
        return float(
            concordance_index_ipcw(
                y_train,
                y_test,
                risk,
                tau=float(tau),
            )[0]
        )
    except ValueError:
        # Same conservative truncation strategy as 03b safe_uno_c.
        mask = np.asarray(y_test["time"], dtype=float) <= float(tau)
        if int(mask.sum()) < 10:
            return float("nan")
        return float(
            concordance_index_ipcw(
                y_train,
                y_test[mask],
                risk[mask],
                tau=float(tau),
            )[0]
        )


def reverse_km(
    time_values: np.ndarray,
    event_values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    # Original event=True means outcome event.
    # For reverse-KM censoring distribution, censoring is treated as the event.
    censoring_event = ~np.asarray(event_values, dtype=bool)

    times, survival = kaplan_meier_estimator(
        censoring_event,
        np.asarray(time_values, dtype=float),
    )
    return np.asarray(times, dtype=float), np.asarray(survival, dtype=float)


def reverse_km_median(times: np.ndarray, survival: np.ndarray) -> float:
    idx = np.flatnonzero(survival <= 0.5)
    if len(idx) == 0:
        return float("nan")
    return float(times[idx[0]])


def endpoint_followup_tables(
    data: Dict[str, Any],
    protocol: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    arms = data["arms"]
    summary_rows: List[Dict[str, Any]] = []
    km_rows: List[Dict[str, Any]] = []

    for endpoint in ["OS", "DFI"]:
        ep = data["endpoint_data"][endpoint]
        times = np.asarray(ep["time"], dtype=float)
        events = np.asarray(ep["event"], dtype=bool)

        for arm in ["COTC021", "COTC022"]:
            idx = np.flatnonzero(arms == arm)
            t = times[idx]
            e = events[idx]

            tau = float(
                np.quantile(
                    t,
                    float(protocol["uno_c"]["tau_quantile"]),
                )
            )

            km_t, km_g = reverse_km(t, e)
            median_censoring_time = reverse_km_median(km_t, km_g)

            summary_rows.append(
                {
                    "endpoint": endpoint,
                    "arm": arm,
                    "n": int(len(idx)),
                    "events": int(e.sum()),
                    "censored": int((~e).sum()),
                    "event_fraction": float(e.mean()),
                    "censor_fraction": float((~e).mean()),
                    "followup_min": float(np.min(t)),
                    "followup_q25": float(np.quantile(t, 0.25)),
                    "followup_median": float(np.median(t)),
                    "followup_q75": float(np.quantile(t, 0.75)),
                    "followup_q90_tau": tau,
                    "followup_max": float(np.max(t)),
                    "reverse_km_median_censoring_time": median_censoring_time,
                }
            )

            for time_point, g_hat in zip(km_t, km_g):
                km_rows.append(
                    {
                        "endpoint": endpoint,
                        "arm": arm,
                        "time": float(time_point),
                        "reverse_km_G_hat_probability_not_yet_censored": float(g_hat),
                    }
                )

    return pd.DataFrame(summary_rows), pd.DataFrame(km_rows)


def refit_direction_without_tuning(
    m03b,
    data: Dict[str, Any],
    endpoint: str,
    source_arm: str,
    target_arm: str,
    chosen_alpha: float,
    protocol: Dict[str, Any],
) -> Dict[str, Any]:
    arms = data["arms"]
    ep = data["endpoint_data"][endpoint]
    y = ep["y"]
    event = ep["event"]

    train_idx = np.flatnonzero(arms == source_arm)
    test_idx = np.flatnonzero(arms == target_arm)

    transformer = m03b.HallmarkTransformer(
        data["module_to_indices"],
        data["module_order"],
        10,
        1e-12,
    ).fit(data["X_hallmark_gene"][train_idx])

    X_train = transformer.transform(data["X_hallmark_gene"][train_idx])
    X_test = transformer.transform(data["X_hallmark_gene"][test_idx])

    model = m03b.fit_ridge(
        X_train,
        y[train_idx],
        float(chosen_alpha),
    )

    train_risk_raw = np.asarray(model.predict(X_train), dtype=float)
    test_risk_raw = np.asarray(model.predict(X_test), dtype=float)

    train_mean = float(np.mean(train_risk_raw))
    train_sd = float(np.std(train_risk_raw, ddof=0))
    scale_sd = train_sd if np.isfinite(train_sd) and train_sd > 1e-12 else 1.0
    test_risk_std = (test_risk_raw - train_mean) / scale_sd

    tau = float(
        m03b.tau_from_training(
            y[train_idx],
            float(protocol["uno_c"]["tau_quantile"]),
        )
    )

    reproduced_c = float(
        m03b.safe_uno_c(
            y[train_idx],
            y[test_idx],
            test_risk_std,
            float(protocol["uno_c"]["tau_quantile"]),
        )
    )

    coef = np.asarray(model.coef_, dtype=float).reshape(-1)
    coef_norm = float(np.linalg.norm(coef))

    return {
        "endpoint": endpoint,
        "source_arm": source_arm,
        "target_arm": target_arm,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "y_train": y[train_idx],
        "y_test": y[test_idx],
        "train_event": event[train_idx],
        "test_event": event[test_idx],
        "train_time": np.asarray(ep["time"], dtype=float)[train_idx],
        "test_time": np.asarray(ep["time"], dtype=float)[test_idx],
        "chosen_alpha": float(chosen_alpha),
        "coef_l2_norm": coef_norm,
        "coef_max_abs": float(np.max(np.abs(coef))),
        "train_risk_raw_mean": train_mean,
        "train_risk_raw_sd": train_sd,
        "train_risk_raw_min": float(np.min(train_risk_raw)),
        "train_risk_raw_max": float(np.max(train_risk_raw)),
        "test_risk_raw_mean": float(np.mean(test_risk_raw)),
        "test_risk_raw_sd": float(np.std(test_risk_raw, ddof=0)),
        "test_risk_raw_min": float(np.min(test_risk_raw)),
        "test_risk_raw_max": float(np.max(test_risk_raw)),
        "test_risk_standardized_sd": float(np.std(test_risk_std, ddof=0)),
        "test_risk_standardized_min": float(np.min(test_risk_std)),
        "test_risk_standardized_max": float(np.max(test_risk_std)),
        "tau_frozen_direction_specific": tau,
        "reproduced_uno_c": reproduced_c,
        "test_risk_standardized": test_risk_std,
        "model_degenerate_numeric_flag": bool(
            coef_norm <= DEGENERACY_COEF_NORM_EPS
            or train_sd <= DEGENERACY_SD_EPS
            or float(np.std(test_risk_raw, ddof=0)) <= DEGENERACY_SD_EPS
        ),
    }


def directional_diagnostics(
    m03b,
    data: Dict[str, Any],
    protocol: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    saved_by_endpoint = {
        "OS": pd.read_csv(B_OS_ARM, sep="\t"),
        "DFI": pd.read_csv(B_DFI_ARM, sep="\t"),
    }

    fitted: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []

    for endpoint in ["OS", "DFI"]:
        saved = saved_by_endpoint[endpoint]

        for source_arm, target_arm in DIRECTIONS:
            hit = saved[
                (saved["source_arm"].astype(str) == source_arm)
                & (saved["target_arm"].astype(str) == target_arm)
            ]
            if len(hit) != 1:
                raise RuntimeError(
                    f"{endpoint} {source_arm}->{target_arm}: expected one saved metric row."
                )
            saved_row = hit.iloc[0]

            diag = refit_direction_without_tuning(
                m03b,
                data,
                endpoint,
                source_arm,
                target_arm,
                float(saved_row["chosen_alpha"]),
                protocol,
            )
            fitted[(endpoint, source_arm, target_arm)] = diag

            saved_c = float(saved_row["uno_c"])
            delta = float(diag["reproduced_uno_c"] - saved_c)

            if not np.isclose(
                diag["reproduced_uno_c"],
                saved_c,
                atol=1e-10,
                rtol=1e-10,
            ):
                raise RuntimeError(
                    f"{endpoint} {source_arm}->{target_arm}: diagnostic refit does "
                    f"not reproduce frozen Uno C. saved={saved_c}, "
                    f"refit={diag['reproduced_uno_c']}"
                )

            rows.append(
                {
                    "endpoint": endpoint,
                    "source_arm": source_arm,
                    "target_arm": target_arm,
                    "n_train": int(len(diag["train_idx"])),
                    "events_train": int(diag["train_event"].sum()),
                    "censored_train": int((~diag["train_event"]).sum()),
                    "n_test": int(len(diag["test_idx"])),
                    "events_test": int(diag["test_event"].sum()),
                    "censored_test": int((~diag["test_event"]).sum()),
                    "censor_fraction_test": float((~diag["test_event"]).mean()),
                    "tau_frozen_direction_specific": diag["tau_frozen_direction_specific"],
                    "fraction_test_followup_above_tau": float(
                        np.mean(diag["test_time"] > diag["tau_frozen_direction_specific"])
                    ),
                    "chosen_alpha": diag["chosen_alpha"],
                    "coef_l2_norm": diag["coef_l2_norm"],
                    "coef_max_abs": diag["coef_max_abs"],
                    "train_risk_raw_sd": diag["train_risk_raw_sd"],
                    "test_risk_raw_sd": diag["test_risk_raw_sd"],
                    "test_risk_standardized_sd": diag["test_risk_standardized_sd"],
                    "test_risk_raw_min": diag["test_risk_raw_min"],
                    "test_risk_raw_max": diag["test_risk_raw_max"],
                    "saved_uno_c": saved_c,
                    "saved_bootstrap_lower_95": float(saved_row["bootstrap_lower_95"]),
                    "saved_bootstrap_upper_95": float(saved_row["bootstrap_upper_95"]),
                    "refit_uno_c": diag["reproduced_uno_c"],
                    "refit_minus_saved_uno_c": delta,
                    "numeric_degeneracy_flag": diag["model_degenerate_numeric_flag"],
                }
            )

    common_rows: List[Dict[str, Any]] = []

    for endpoint in ["OS", "DFI"]:
        d1 = fitted[(endpoint, "COTC021", "COTC022")]
        d2 = fitted[(endpoint, "COTC022", "COTC021")]

        common_tau = float(
            min(
                d1["tau_frozen_direction_specific"],
                d2["tau_frozen_direction_specific"],
            )
        )

        for diag in [d1, d2]:
            common_c = safe_uno_at_fixed_tau(
                diag["y_train"],
                diag["y_test"],
                diag["test_risk_standardized"],
                common_tau,
            )

            common_rows.append(
                {
                    "endpoint": endpoint,
                    "source_arm": diag["source_arm"],
                    "target_arm": diag["target_arm"],
                    "original_direction_specific_tau": diag[
                        "tau_frozen_direction_specific"
                    ],
                    "common_tau_min_of_directional_taus": common_tau,
                    "original_frozen_uno_c": diag["reproduced_uno_c"],
                    "common_tau_uno_c_non_gating": common_c,
                    "common_minus_original_uno_c": (
                        float(common_c - diag["reproduced_uno_c"])
                        if np.isfinite(common_c)
                        else float("nan")
                    ),
                    "test_fraction_followup_above_common_tau": float(
                        np.mean(diag["test_time"] > common_tau)
                    ),
                    "interpretation_role": "NON_GATING_COMMON_HORIZON_SENSITIVITY",
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(common_rows)


def ast_definition_hashes(path: Path, names: Sequence[str]) -> Dict[str, str]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    found: Dict[str, str] = {}

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in names:
                payload = ast.dump(node, annotate_fields=True, include_attributes=False)
                found[node.name] = sha256_text(payload)

    return found


def validate_stage_artifact_hashes(
    stage_json: Path,
    protocol_hash: str,
    artifact_dir: Path,
) -> Dict[str, Any]:
    payload = read_json(stage_json)

    if clean(payload.get("status")) != "PASS":
        raise RuntimeError(f"Stage checkpoint is not PASS: {stage_json}")
    if clean(payload.get("protocol_sha256")) != protocol_hash:
        raise RuntimeError(f"Stage protocol hash mismatch: {stage_json}")

    hashes = payload.get("artifact_hashes") or {}
    checked = []

    for filename, expected in hashes.items():
        path = artifact_dir / filename
        require_file(path)
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(
                f"Stage artifact hash mismatch: {path.name}"
            )
        checked.append(
            {
                "artifact": filename,
                "sha256": observed,
            }
        )

    return {
        "stage": stage_json.name,
        "protocol_sha256": protocol_hash,
        "n_artifacts_verified": len(checked),
        "artifacts": checked,
    }


def audit_v1_v2_reuse(protocol_hash: str) -> Dict[str, Any]:
    require_file(B_V1_SCRIPT)
    require_file(B_V2_SCRIPT)

    v1_text = B_V1_SCRIPT.read_text(encoding="utf-8")
    v2_text = B_V2_SCRIPT.read_text(encoding="utf-8")

    diff_lines = list(
        difflib.unified_diff(
            v1_text.splitlines(),
            v2_text.splitlines(),
            fromfile=B_V1_SCRIPT.name,
            tofile=B_V2_SCRIPT.name,
            lineterm="",
        )
    )
    REUSE_DIFF.write_text("\n".join(diff_lines) + "\n", encoding="utf-8")

    h1 = ast_definition_hashes(B_V1_SCRIPT, CORE_AST_NAMES)
    h2 = ast_definition_hashes(B_V2_SCRIPT, CORE_AST_NAMES)

    missing_v1 = sorted(set(CORE_AST_NAMES) - set(h1))
    missing_v2 = sorted(set(CORE_AST_NAMES) - set(h2))
    if missing_v1 or missing_v2:
        raise RuntimeError(
            f"Core AST audit incomplete. missing_v1={missing_v1}, missing_v2={missing_v2}"
        )

    identical = {
        name: bool(h1[name] == h2[name])
        for name in CORE_AST_NAMES
    }

    if not all(identical.values()):
        changed = [name for name, same in identical.items() if not same]
        raise RuntimeError(
            "v1->v2 changed scientific fit/evaluation definitions: "
            + ", ".join(changed)
        )

    stages = [
        validate_stage_artifact_hashes(
            B_OS_PRIMARY_STAGE,
            protocol_hash,
            B_DIR,
        ),
        validate_stage_artifact_hashes(
            B_OS_SUPPORT_STAGE,
            protocol_hash,
            B_DIR,
        ),
        validate_stage_artifact_hashes(
            B_OS_ARM_STAGE,
            protocol_hash,
            B_DIR,
        ),
    ]

    payload = {
        "status": "PASS_REUSE_SCIENTIFICALLY_SAFE",
        "created_utc": now_utc(),
        "v1_script": {
            "path": str(B_V1_SCRIPT.relative_to(ROOT)),
            "sha256": sha256_file(B_V1_SCRIPT),
        },
        "v2_script": {
            "path": str(B_V2_SCRIPT.relative_to(ROOT)),
            "sha256": sha256_file(B_V2_SCRIPT),
        },
        "03a_protocol_sha256": protocol_hash,
        "core_result_producing_AST_definitions_identical": True,
        "core_definition_hashes_v1": h1,
        "core_definition_hashes_v2": h2,
        "per_definition_identity": identical,
        "unified_diff_path": str(REUSE_DIFF.relative_to(ROOT)),
        "unified_diff_lines": len(diff_lines),
        "validated_reused_OS_stages": stages,
        "scientific_interpretation": (
            "03b v2 changed interpretation-layer routing of frozen scientific "
            "thresholds to the exact hash-locked 02i contract. The ASTs of all "
            "listed result-producing model/preprocessing/evaluation functions and "
            "classes are identical between v1 and v2, and reused OS stage artifacts "
            "validate against the unchanged 03a protocol SHA256."
        ),
        "publication_action": (
            "Retain reuse as valid. Perform one clean from-scratch reproducibility "
            "rerun before final manuscript freeze."
        ),
    }
    write_json(REUSE_JSON, payload)
    return payload


def trial_design_provenance(data: Dict[str, Any]) -> Dict[str, Any]:
    # Search only already-existing project artifacts. No network access.
    candidate_files: List[Dict[str, Any]] = []

    for path in ROOT.rglob("*02e5*"):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        if "COTC021" in text and "COTC022" in text:
            candidate_files.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": sha256_file(path),
                    "contains_random_token": bool(
                        re.search(r"random", text, flags=re.IGNORECASE)
                    ),
                    "contains_SOC_PLUS_RAPAMYCIN": "SOC_PLUS_RAPAMYCIN" in text,
                    "contains_SOC_CONTROL": "SOC_CONTROL" in text,
                }
            )

    roster = data["roster"].copy()
    arm_map = (
        roster[["study", "gate_zero_arm"]]
        .drop_duplicates()
        .sort_values(["study", "gate_zero_arm"])
        .to_dict(orient="records")
    )

    expected_map = {
        "COTC021": "SOC_PLUS_RAPAMYCIN",
        "COTC022": "SOC_CONTROL",
    }
    observed_map = {
        clean(row["study"]): clean(row["gate_zero_arm"])
        for row in arm_map
    }

    if observed_map != expected_map:
        raise RuntimeError(
            f"Frozen study->arm mapping changed: {observed_map}"
        )

    payload = {
        "status": "PASS_FROZEN_TRIAL_ARM_IDENTITY_RETAINED",
        "created_utc": now_utc(),
        "frozen_sample_counts": {
            arm: int(np.sum(data["arms"] == arm))
            for arm in ["COTC021", "COTC022"]
        },
        "frozen_study_to_treatment_arm_mapping": observed_map,
        "local_02e5_provenance_files": candidate_files,
        "interpretation": (
            "For Paper-6 computation, ICDC study labels COTC021 and COTC022 are "
            "the frozen operational identifiers for the two treatment allocations "
            "SOC_PLUS_RAPAMYCIN and SOC_CONTROL. The manuscript should describe "
            "them as the randomized treatment arms of the unified COTC021/022 "
            "trial design, while noting that ICDC exposes them as separate study records."
        ),
        "pretreatment_expression_note": (
            "03c does not infer tissue timing from outcomes. The manuscript-level "
            "statement that tumor RNA was pretreatment and randomization occurred "
            "at the beginning of carboplatin therapy should be cited directly to "
            "the trial/DOG2 primary publications."
        ),
        "network_access_in_03c": False,
    }
    write_json(TRIAL_JSON, payload)
    return payload


def unresolved_annotation_audit() -> Tuple[pd.DataFrame, pd.DataFrame]:
    require_file(G_RESOLUTION)
    require_file(G_RAW)

    resolution = pd.read_csv(
        G_RESOLUTION,
        sep="\t",
        dtype=str,
        low_memory=False,
    ).fillna("")

    unresolved = resolution[
        resolution["resolution_status"].astype(str)
        == "UNRESOLVED_CURRENT_ENSEMBL_SYMBOL"
    ].copy()

    if len(unresolved) != EXPECTED_UNRESOLVED:
        raise RuntimeError(
            f"02g unresolved feature count={len(unresolved)}, "
            f"expected={EXPECTED_UNRESOLVED}."
        )

    raw_map = pd.read_csv(
        G_RAW,
        sep="\t",
        dtype=str,
        low_memory=False,
    ).fillna("")

    required = {"ensembl_gene_id", "external_gene_name"}
    if not required.issubset(raw_map.columns):
        raise RuntimeError(
            f"02g raw Ensembl snapshot lacks {sorted(required - set(raw_map.columns))}"
        )

    current_symbols = set(
        raw_map.loc[
            raw_map["external_gene_name"].astype(str).str.strip().ne(""),
            "external_gene_name",
        ].astype(str).str.strip()
    )
    current_gene_ids = set(
        raw_map.loc[
            raw_map["ensembl_gene_id"].astype(str).str.strip().ne(""),
            "ensembl_gene_id",
        ].astype(str).str.strip()
    )

    gene_id_to_symbols = (
        raw_map.assign(
            ensembl_gene_id=raw_map["ensembl_gene_id"].astype(str).str.strip(),
            external_gene_name=raw_map["external_gene_name"].astype(str).str.strip(),
        )
        .groupby("ensembl_gene_id")["external_gene_name"]
        .apply(lambda s: sorted({x for x in s if x}))
        .to_dict()
    )

    any_suffix_re = re.compile(r"^(.+)_([0-9]+)$")

    rows: List[Dict[str, Any]] = []

    for raw in unresolved["dog2_raw_feature"].astype(str):
        match = any_suffix_re.match(raw)
        base_any_suffix = match.group(1) if match else ""
        suffix_n = match.group(2) if match else ""

        exact_symbol = raw in current_symbols
        base_symbol = bool(base_any_suffix and base_any_suffix in current_symbols)

        exact_gene_id = raw in current_gene_ids
        base_gene_id = bool(base_any_suffix and base_any_suffix in current_gene_ids)

        matched_gene_id = raw if exact_gene_id else (
            base_any_suffix if base_gene_id else ""
        )
        symbols_for_gene_id = gene_id_to_symbols.get(matched_gene_id, [])

        if raw.startswith("ENSCAFG"):
            lexical_class = "ENSCAFG_GENE_STABLE_ID_LIKE"
        elif raw.startswith("ENSCAFT"):
            lexical_class = "ENSCAFT_TRANSCRIPT_STABLE_ID_LIKE"
        elif raw.upper().startswith("LOC"):
            lexical_class = "LOC_LIKE"
        elif re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.\-]*", raw):
            lexical_class = "SYMBOL_LIKE"
        else:
            lexical_class = "OTHER"

        rows.append(
            {
                "dog2_raw_feature": raw,
                "lexical_class": lexical_class,
                "terminal_numeric_suffix_any": suffix_n,
                "base_after_any_terminal_numeric_suffix": base_any_suffix,
                "exact_match_current_Ensembl_gene_symbol": exact_symbol,
                "base_after_any_suffix_matches_current_Ensembl_gene_symbol": base_symbol,
                "exact_match_current_Ensembl_gene_id": exact_gene_id,
                "base_after_any_suffix_matches_current_Ensembl_gene_id": base_gene_id,
                "matched_current_Ensembl_gene_id": matched_gene_id,
                "matched_gene_id_has_nonempty_current_symbol": bool(symbols_for_gene_id),
                "current_symbols_for_matched_gene_id": ";".join(symbols_for_gene_id),
                "note": (
                    "DIAGNOSTIC_ONLY_NO_CHANGE_TO_FROZEN_02G_RESOLUTION"
                ),
            }
        )

    audit = pd.DataFrame(rows)

    summary_rows = [
        {
            "category": "unresolved_total",
            "count": int(len(audit)),
            "fraction_of_unresolved": 1.0,
        },
        {
            "category": "exact_current_Ensembl_symbol_match_among_unresolved",
            "count": int(
                audit["exact_match_current_Ensembl_gene_symbol"].sum()
            ),
            "fraction_of_unresolved": float(
                audit["exact_match_current_Ensembl_gene_symbol"].mean()
            ),
        },
        {
            "category": "any_suffix_base_current_Ensembl_symbol_match_among_unresolved",
            "count": int(
                audit[
                    "base_after_any_suffix_matches_current_Ensembl_gene_symbol"
                ].sum()
            ),
            "fraction_of_unresolved": float(
                audit[
                    "base_after_any_suffix_matches_current_Ensembl_gene_symbol"
                ].mean()
            ),
        },
        {
            "category": "exact_current_Ensembl_gene_id_match_among_unresolved",
            "count": int(
                audit["exact_match_current_Ensembl_gene_id"].sum()
            ),
            "fraction_of_unresolved": float(
                audit["exact_match_current_Ensembl_gene_id"].mean()
            ),
        },
        {
            "category": "gene_id_match_with_nonempty_current_symbol_among_unresolved",
            "count": int(
                audit["matched_gene_id_has_nonempty_current_symbol"].sum()
            ),
            "fraction_of_unresolved": float(
                audit["matched_gene_id_has_nonempty_current_symbol"].mean()
            ),
        },
    ]

    lexical_counts = audit["lexical_class"].value_counts()
    for category, count in lexical_counts.items():
        summary_rows.append(
            {
                "category": f"lexical_class:{category}",
                "count": int(count),
                "fraction_of_unresolved": float(count / len(audit)),
            }
        )

    summary = pd.DataFrame(summary_rows)
    audit.to_csv(UNRESOLVED_TSV, sep="\t", index=False)
    summary.to_csv(UNRESOLVED_SUMMARY, sep="\t", index=False)

    return audit, summary


def freeze_premise_action_policy(i_contract_hash: str) -> Dict[str, Any]:
    payload = {
        "status": "PASS_PREMISE_ACTION_POLICY_FROZEN_BEFORE_HUMAN_PREMISE_OUTCOMES",
        "created_utc": now_utc(),
        "02i_contract_sha256": i_contract_hash,
        "premise_materiality_thresholds_changed": False,
        "human_outcomes_read_in_03c": False,
        "actions": {
            "SUPPORTS_CANINE_ADDED_VALUE": {
                "continue_closed_AI_simulation_development": True,
                "continue_frozen_human_confirmatory_program": True,
                "empirical_premise_label": "SUPPORTED_ON_SACRIFICIAL_TARGET",
                "claim_rule": (
                    "Support on the premise target motivates, but does not substitute "
                    "for, frozen confirmatory evaluation on reserved human cohorts."
                ),
            },
            "INCONCLUSIVE_NEUTRAL": {
                "continue_closed_AI_simulation_development": True,
                "continue_frozen_human_confirmatory_program": True,
                "empirical_premise_label": "UNRESOLVED",
                "claim_rule": (
                    "Do not reinterpret a small positive/negative delta as support. "
                    "Premise-target outcomes may not tune the proposed AI method. "
                    "Architecture selection remains simulation-first and must be frozen "
                    "before TARGET/GSE21257/GSE39055 outcome evaluation."
                ),
                "reason_for_continuing": (
                    "An inconclusive sacrificial cohort is not evidence of no canine "
                    "value; it leaves the empirical premise unresolved while the method "
                    "question of selective borrowing/negative-transfer protection remains valid."
                ),
            },
            "ARGUES_AGAINST_CANINE_ADDED_VALUE": {
                "continue_closed_AI_simulation_development": True,
                "continue_frozen_human_confirmatory_program": True,
                "empirical_premise_label": "NEGATIVE_ON_SACRIFICIAL_TARGET",
                "claim_rule": (
                    "Canine-benefit/superiority is not the expected empirical story. "
                    "Reserved human evaluations, if run after method freeze, are treated "
                    "as confirmatory falsification/negative-transfer-safety tests rather "
                    "than opportunities for outcome-guided rescue."
                ),
            },
        },
        "central_framing_after_SOURCE_AMBER": (
            "When does cross-species borrowing remain useful, when does it become "
            "context-sensitive, and can selective transfer protect against negative "
            "transfer under human event scarcity?"
        ),
    }
    write_json(PREMISE_POLICY, payload)
    return payload


def main() -> None:
    started = now_utc()

    print("=" * 120)
    print("Paper 6 - bounded post-gate diagnostics after DOG2 Source Prognostic Gate")
    print("=" * 120)
    print(f"Script version: {SCRIPT_VERSION}")
    print()
    print("Scope:")
    print("  DOG2 outcome/expression values read: YES [already opened in 03b]")
    print("  Human outcome/clinical values read: NO")
    print("  New model tuning: NO")
    print("  New model family: NO")
    print("  Source Gate status changed: NO")
    print("  Permutations increased: NO")
    print("  Network access: NO")
    print("  Common-tau calculation: NON-GATING diagnostic only")
    print()

    for path in [
        I_CONTRACT,
        I_SUMMARY,
        A_PROTOCOL,
        A_SUMMARY,
        B_SUMMARY,
        B_RESULTS,
        B_OS_ARM,
        B_DFI_ARM,
        B_V1_SCRIPT,
        B_V2_SCRIPT,
        G_RESOLUTION,
        G_RAW,
        G_SUMMARY,
        UPSTREAM_LOCK,
    ]:
        require_file(path)

    i_contract = read_json(I_CONTRACT)
    a_protocol = read_json(A_PROTOCOL)
    a_summary = read_json(A_SUMMARY)
    b_summary = read_json(B_SUMMARY)
    b_results = read_json(B_RESULTS)
    upstream = read_json(UPSTREAM_LOCK)

    if clean(i_contract.get("status")) != "PASS":
        raise RuntimeError("02i contract is not PASS.")
    if clean(a_protocol.get("status")) != "PASS":
        raise RuntimeError("03a protocol is not PASS.")
    if clean(a_summary.get("scientific_status")) != (
        "PASS_SOURCE_PROGNOSTIC_GATE_PROTOCOL_FROZEN"
    ):
        raise RuntimeError("03a protocol is not in frozen PASS state.")
    if clean(b_summary.get("status")) != "PASS":
        raise RuntimeError("03b summary is not PASS.")
    if clean(b_summary.get("scientific_status")) != EXPECTED_03B_STATUS:
        raise RuntimeError(
            f"03b status changed: {b_summary.get('scientific_status')}"
        )

    protocol_hash = sha256_file(A_PROTOCOL)
    expected_protocol_hash = clean(
        (a_summary.get("final_artifact_hashes") or {}).get(
            "source_prognostic_gate_protocol_json"
        )
    )
    if protocol_hash != expected_protocol_hash:
        raise RuntimeError("03a protocol hash verification failed.")

    i_hash = sha256_file(I_CONTRACT)
    if clean((a_protocol.get("upstream") or {}).get("02i_contract_sha256")) != i_hash:
        raise RuntimeError("03a does not point to the current exact 02i contract.")

    print("Frozen provenance:")
    print(f"  02i SHA256: {i_hash}")
    print(f"  03a SHA256: {protocol_hash}")
    print(f"  03b scientific status: {b_summary['scientific_status']}")
    print()

    # Import the exact v2 implementation and use its frozen input loader and
    # model/preprocessing functions. No main() is executed on import.
    m03b = load_module(B_V2_SCRIPT, "paper6_03b_v2_for_03c")

    data = m03b.load_inputs(a_protocol, upstream)

    if len(data["sample_ids"]) != EXPECTED_N:
        raise RuntimeError("03c source n is not 186.")
    observed_arm_counts = {
        arm: int(np.sum(data["arms"] == arm))
        for arm in ["COTC021", "COTC022"]
    }
    if observed_arm_counts != EXPECTED_ARMS:
        raise RuntimeError(
            f"03c arm counts changed: {observed_arm_counts}"
        )

    if int(data["endpoint_data"]["OS"]["event"].sum()) != EXPECTED_OS_EVENTS:
        raise RuntimeError("03c OS event count changed.")
    if int(data["endpoint_data"]["DFI"]["event"].sum()) != EXPECTED_DFI_EVENTS:
        raise RuntimeError("03c DFI event count changed.")

    # ------------------------------------------------------------------
    # 1-2. Event/censoring/follow-up/tau diagnostics.
    # ------------------------------------------------------------------
    followup, km = endpoint_followup_tables(data, a_protocol)
    followup.to_csv(ARM_FOLLOWUP, sep="\t", index=False)
    km.to_csv(CENSORING_KM, sep="\t", index=False)

    print("-" * 120)
    print("Arm-specific events, censoring, follow-up and frozen tau")
    print("-" * 120)
    print(
        followup[
            [
                "endpoint",
                "arm",
                "n",
                "events",
                "censored",
                "censor_fraction",
                "followup_median",
                "followup_q90_tau",
                "followup_max",
                "reverse_km_median_censoring_time",
            ]
        ].to_string(index=False)
    )

    # ------------------------------------------------------------------
    # 3-4. Exact-alpha no-retuning refits + common tau.
    # ------------------------------------------------------------------
    direction_diag, common_tau = directional_diagnostics(
        m03b,
        data,
        a_protocol,
    )
    direction_diag.to_csv(DIRECTION_DIAG, sep="\t", index=False)
    common_tau.to_csv(COMMON_TAU, sep="\t", index=False)

    print()
    print("-" * 120)
    print("Directional arm-model diagnostic")
    print("-" * 120)
    print(
        direction_diag[
            [
                "endpoint",
                "source_arm",
                "target_arm",
                "events_train",
                "events_test",
                "tau_frozen_direction_specific",
                "chosen_alpha",
                "coef_l2_norm",
                "train_risk_raw_sd",
                "test_risk_raw_sd",
                "numeric_degeneracy_flag",
                "saved_uno_c",
                "saved_bootstrap_lower_95",
                "saved_bootstrap_upper_95",
            ]
        ].to_string(index=False)
    )

    print()
    print("Common-tau NON-GATING sensitivity:")
    print(
        common_tau[
            [
                "endpoint",
                "source_arm",
                "target_arm",
                "original_direction_specific_tau",
                "common_tau_min_of_directional_taus",
                "original_frozen_uno_c",
                "common_tau_uno_c_non_gating",
            ]
        ].to_string(index=False)
    )

    # ------------------------------------------------------------------
    # 5. v1->v2 reuse provenance.
    # ------------------------------------------------------------------
    reuse = audit_v1_v2_reuse(protocol_hash)

    # ------------------------------------------------------------------
    # 5b. Trial arm provenance from already-frozen local artifacts.
    # ------------------------------------------------------------------
    trial = trial_design_provenance(data)

    # ------------------------------------------------------------------
    # 6. Unresolved 02g feature-name limitation.
    # ------------------------------------------------------------------
    unresolved, unresolved_summary = unresolved_annotation_audit()

    print()
    print("-" * 120)
    print("02g unresolved feature-name diagnostic [NO bridge change]")
    print("-" * 120)
    print(unresolved_summary.to_string(index=False))

    # ------------------------------------------------------------------
    # 7. Freeze future INCONCLUSIVE premise action before human outcomes.
    # ------------------------------------------------------------------
    premise_policy = freeze_premise_action_policy(i_hash)

    # ------------------------------------------------------------------
    # Final bounded diagnostic record.
    # ------------------------------------------------------------------
    os_rows = direction_diag[direction_diag["endpoint"] == "OS"]
    dfi_rows = direction_diag[direction_diag["endpoint"] == "DFI"]

    diagnostic = {
        "script_version": SCRIPT_VERSION,
        "status": "PASS",
        "created_utc": now_utc(),
        "scientific_status": (
            "PASS_BOUNDED_DIAGNOSTIC_COMPLETE_SOURCE_GATE_UNCHANGED"
        ),
        "source_gate_status_before_03c": EXPECTED_03B_STATUS,
        "source_gate_status_after_03c": EXPECTED_03B_STATUS,
        "gate_reclassification_performed": False,
        "new_tuning_performed": False,
        "human_outcomes_read": False,
        "permutation_count_changed": False,
        "arm_followup": followup.to_dict(orient="records"),
        "directional_model_diagnostics": direction_diag.to_dict(orient="records"),
        "common_tau_sensitivity": common_tau.to_dict(orient="records"),
        "v1_v2_reuse_provenance_status": reuse["status"],
        "trial_design_provenance_status": trial["status"],
        "unresolved_feature_audit": {
            "n_unresolved": int(len(unresolved)),
            "exact_symbol_match_among_unresolved": int(
                unresolved["exact_match_current_Ensembl_gene_symbol"].sum()
            ),
            "any_suffix_base_symbol_match_among_unresolved": int(
                unresolved[
                    "base_after_any_suffix_matches_current_Ensembl_gene_symbol"
                ].sum()
            ),
            "exact_gene_id_match_among_unresolved": int(
                unresolved["exact_match_current_Ensembl_gene_id"].sum()
            ),
            "gene_id_match_with_nonempty_symbol_among_unresolved": int(
                unresolved["matched_gene_id_has_nonempty_current_symbol"].sum()
            ),
        },
        "premise_INCONCLUSIVE_action": premise_policy["actions"][
            "INCONCLUSIVE_NEUTRAL"
        ],
        "interpretation_guardrail": (
            "Directional arm asymmetry may be discussed only after considering "
            "arm-specific event information, censoring/follow-up support, tau, and "
            "risk-score degeneracy. 03c does not establish a causal treatment "
            "interaction or change the frozen ARM_WARN classification."
        ),
        "next": "04a sacrificial human cohort metadata/endpoint/lineage freeze",
    }
    write_json(DIAGNOSTIC_JSON, diagnostic)

    artifacts = [
        ARM_FOLLOWUP,
        CENSORING_KM,
        DIRECTION_DIAG,
        COMMON_TAU,
        REUSE_JSON,
        REUSE_DIFF,
        TRIAL_JSON,
        UNRESOLVED_TSV,
        UNRESOLVED_SUMMARY,
        PREMISE_POLICY,
        DIAGNOSTIC_JSON,
    ]

    summary = {
        "script_version": SCRIPT_VERSION,
        "status": "PASS",
        "scientific_status": (
            "PASS_BOUNDED_DIAGNOSTIC_COMPLETE_SOURCE_GATE_UNCHANGED"
        ),
        "run_started_utc": started,
        "run_finished_utc": now_utc(),
        "source_gate_status": EXPECTED_03B_STATUS,
        "OS_events_COTC021": int(
            followup[
                (followup["endpoint"] == "OS")
                & (followup["arm"] == "COTC021")
            ].iloc[0]["events"]
        ),
        "OS_events_COTC022": int(
            followup[
                (followup["endpoint"] == "OS")
                & (followup["arm"] == "COTC022")
            ].iloc[0]["events"]
        ),
        "DFI_events_COTC021": int(
            followup[
                (followup["endpoint"] == "DFI")
                & (followup["arm"] == "COTC021")
            ].iloc[0]["events"]
        ),
        "DFI_events_COTC022": int(
            followup[
                (followup["endpoint"] == "DFI")
                & (followup["arm"] == "COTC022")
            ].iloc[0]["events"]
        ),
        "n_directional_numeric_degeneracy_flags": int(
            direction_diag["numeric_degeneracy_flag"].sum()
        ),
        "03b_v1_v2_reuse_safe": bool(
            reuse["status"] == "PASS_REUSE_SCIENTIFICALLY_SAFE"
        ),
        "premise_INCONCLUSIVE_policy": (
            "CONTINUE_AI_SIMULATION_AND_FROZEN_HUMAN_CONFIRMATORY_PROGRAM;"
            " EMPIRICAL_CANINE_ADDED_VALUE_REMAINS_UNRESOLVED"
        ),
        "final_artifact_hashes": {
            path.name: sha256_file(path)
            for path in artifacts
        },
        "DOG2_outcome_values_read": True,
        "DOG2_expression_values_read": True,
        "human_outcome_values_read": False,
        "new_model_tuning": False,
        "source_gate_changed": False,
        "next": "04a",
    }
    write_json(SUMMARY_JSON, summary)

    print()
    print("=" * 120)
    print("03c BOUNDED POST-GATE DIAGNOSTIC SUMMARY")
    print("=" * 120)
    print(
        "Source Gate remains: "
        f"{EXPECTED_03B_STATUS} [UNCHANGED]"
    )
    print(
        "Directional numerical degeneracy flags: "
        f"{int(direction_diag['numeric_degeneracy_flag'].sum())}/4"
    )
    print(
        "03b v1->v2 reused OS stages scientifically safe: "
        f"{reuse['status']}"
    )
    print(
        "Premise INCONCLUSIVE action: CONTINUE simulation/closed-AI + "
        "frozen human confirmatory program; canine added value remains UNRESOLVED."
    )
    print()
    print("No new source-model search is authorized.")
    print("Next: 04a sacrificial-human cohort metadata/endpoint/lineage freeze.")
    print()
    print("Artifacts:")
    for path in artifacts + [SUMMARY_JSON]:
        print(f"  {path.relative_to(ROOT)}")
    print("=" * 120)
    print("03c bounded source-gate diagnostics: PASS")
    print("=" * 120)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 120, file=sys.stderr)
        print("03c bounded source-gate diagnostics: FAIL", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print("=" * 120, file=sys.stderr)
        raise
