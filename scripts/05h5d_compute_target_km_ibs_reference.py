#!/usr/bin/env python
"""
Paper 6 - compute the 05h0-authorized TARGET no-covariate Kaplan-Meier IBS reference.

This is the final post-opening descriptive null-reference calculation.

Scientific role
---------------
POST-OPENING DESCRIPTIVE NULL REFERENCE.

For each already-frozen TARGET outer-training partition:
  1) estimate a no-covariate Kaplan-Meier survival curve using ONLY that
     outer-training partition's (time, event) outcomes;
  2) apply the same survival curve to every patient in the corresponding
     held-out outer-test partition;
  3) construct the IBS evaluation grid with the exact frozen 04c
     choose_ibs_grid() implementation;
  4) calculate IBS with the exact frozen 04c integrated_brier_custom()
     implementation, including its censoring/IPCW mechanics;
  5) aggregate across the exact frozen 20 x 5 partition structure using the
     aggregation rule first replayed against the existing TARGET model summary.

This reference is descriptive only. It cannot select a model, change T-A/B/C/D,
reopen 05d, or alter any frozen human result.

Expected placement:
    scripts/05h5d_compute_target_km_ibs_reference.py

Run:
    python scripts\\05h5d_compute_target_km_ibs_reference.py
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.util import Surv


SCRIPT_VERSION = "05h5d-compute-target-km-ibs-reference-v1-safe-module-import-no-cli"

EXPECTED_N = 86
EXPECTED_EVENTS = 29
EXPECTED_CENSORED = 57
EXPECTED_REPEATS = 20
EXPECTED_FOLDS = 5
EXPECTED_MODELS = 9

AGG_REPLAY_TOL = 1e-10
KM_CROSSCHECK_TOL = 1e-12


def project_root() -> Path:
    p = Path(__file__).resolve()
    if p.parent.name.lower() != "scripts":
        raise RuntimeError(
            "Place 05h5d_compute_target_km_ibs_reference.py "
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

H5B_DIR = ROOT / "method_contract" / "05h5b_target_ibs_mechanics"
H5B_CONTRACT = H5B_DIR / "target_ibs_mechanics_contract.json"
H5B_MANIFEST = H5B_DIR / "freeze_manifest.json"
FOLD_STRUCTURE = H5B_DIR / "frozen_outer_fold_structure.tsv"

OUT_DIR = ROOT / "method_contract" / "05h5d_target_km_ibs_reference"
WORK_DIR = ROOT / "method_contract" / ".05h5d_target_km_ibs_reference_work"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def verify_05h0() -> dict[str, Any]:
    if not H0_JSON.exists():
        raise FileNotFoundError(f"Missing frozen 05h0 contract: {H0_JSON}")

    h0 = load_json(H0_JSON)
    block = recursive_find_key(h0, "target_km_ibs_reference")
    if not isinstance(block, dict):
        raise RuntimeError("05h0 lacks target_km_ibs_reference.")

    if block.get("analysis_role") != "POST-OPENING DESCRIPTIVE NULL REFERENCE":
        raise RuntimeError(
            f"Unexpected 05h0 KM-reference role: {block.get('analysis_role')!r}"
        )
    if block.get("allowed_new_estimation") is not True:
        raise RuntimeError("05h0 does not authorize KM-reference estimation.")

    method = str(block.get("method", "")).lower()
    required = [
        "outer-training",
        "kaplan-meier",
        "outer-test",
        "same frozen time horizon",
        "censoring mechanics",
        "fold structure",
        "aggregation",
    ]
    missing = [x for x in required if x not in method]
    if missing:
        raise RuntimeError(f"05h0 KM-reference method text missing: {missing}")

    prohibited = " ".join(map(str, block.get("prohibited_uses", []))).lower()
    for phrase in ["tuning", "selection"]:
        if phrase not in prohibited:
            raise RuntimeError(
                f"05h0 prohibited-use text no longer contains {phrase!r}."
            )

    return block


def verify_05h5b() -> dict[str, Any]:
    for p in [H5B_CONTRACT, H5B_MANIFEST, FOLD_STRUCTURE]:
        if not p.exists():
            raise FileNotFoundError(f"Required 05h5b artifact missing: {p}")

    manifest = load_json(H5B_MANIFEST)
    if manifest.get("status") != "PASS_READY_FOR_05H5C_KM_IBS_ESTIMATION":
        raise RuntimeError(
            f"05h5b manifest is not PASS: {manifest.get('status')!r}"
        )

    output_hashes = {
        str(row["file"]): str(row["sha256"])
        for row in manifest.get("outputs", [])
        if isinstance(row, dict) and "file" in row and "sha256" in row
    }

    for p in [H5B_CONTRACT, FOLD_STRUCTURE]:
        expected = output_hashes.get(p.name)
        if expected is None:
            raise RuntimeError(f"05h5b manifest does not lock {p.name}.")
        observed = sha256_file(p)
        if observed != expected:
            raise RuntimeError(
                f"05h5b output changed after freeze: {p.name}; "
                f"observed={observed}, expected={expected}"
            )

    contract = load_json(H5B_CONTRACT)
    if contract.get("status") != "PASS_READY_FOR_05H5C_KM_IBS_ESTIMATION":
        raise RuntimeError(
            f"05h5b contract is not PASS: {contract.get('status')!r}"
        )
    if contract.get("scientific_km_reference_computed") is not False:
        raise RuntimeError("05h5b unexpectedly computed a KM reference.")
    if contract.get("scientific_ibs_reference_computed") is not False:
        raise RuntimeError("05h5b unexpectedly computed an IBS reference.")

    return contract


def resolve_and_verify_locked_sources(
    contract: dict[str, Any],
) -> dict[str, Path]:
    source_meta = contract.get("source_files")
    if not isinstance(source_meta, dict):
        raise RuntimeError("05h5b contract lacks source_files.")

    required_roles = {
        "05h0_contract",
        "05f3f_summary",
        "05f3f_input_audit",
        "TARGET86_endpoint",
        "TARGET_split_registry",
        "TARGET_fold_metrics",
        "TARGET_model_summary",
        "TARGET_oof_predictions",
        "TARGET_runner_v3_bytesafe",
        "frozen_04c_metric_implementation",
    }
    missing_roles = required_roles - set(source_meta)
    if missing_roles:
        raise RuntimeError(
            f"05h5b source lock lacks required roles: {sorted(missing_roles)}"
        )

    out: dict[str, Path] = {}
    for role in required_roles:
        meta = source_meta[role]
        if not isinstance(meta, dict):
            raise RuntimeError(f"Invalid source metadata for role {role}.")
        rel = meta.get("path")
        expected_hash = meta.get("sha256")
        expected_bytes = meta.get("bytes")
        if not isinstance(rel, str) or not isinstance(expected_hash, str):
            raise RuntimeError(f"Incomplete source lock for role {role}.")

        path = ROOT / rel
        if not path.exists():
            raise FileNotFoundError(f"Locked source missing for {role}: {path}")
        if sha256_file(path) != expected_hash:
            raise RuntimeError(f"Locked source hash changed for {role}: {path}")
        if expected_bytes is not None and path.stat().st_size != int(expected_bytes):
            raise RuntimeError(f"Locked source byte size changed for {role}: {path}")
        out[role] = path

    return out


def verify_contract_semantics(contract: dict[str, Any]) -> None:
    target = contract.get("TARGET86", {})
    checks = [
        ("n", target.get("n"), EXPECTED_N),
        ("events", target.get("events"), EXPECTED_EVENTS),
        ("censored", target.get("censored"), EXPECTED_CENSORED),
        ("zero_time_n", target.get("zero_time_n"), 1),
    ]
    bad = [(k, obs, exp) for k, obs, exp in checks if int(obs) != exp]
    if bad:
        raise RuntimeError(f"05h5b TARGET86 contract changed: {bad}")
    if target.get("zero_time_is_censored") is not True:
        raise RuntimeError("05h5b no longer records the time-zero TARGET case as censored.")

    outer = contract.get("outer_cv", {})
    if len(outer.get("repeat_labels", [])) != EXPECTED_REPEATS:
        raise RuntimeError("05h5b outer-CV repeat registry is not 20 repeats.")
    if len(outer.get("outer_fold_labels", [])) != EXPECTED_FOLDS:
        raise RuntimeError("05h5b outer-CV fold registry is not 5 folds.")
    if int(outer.get("n_partitions")) != EXPECTED_REPEATS * EXPECTED_FOLDS:
        raise RuntimeError("05h5b outer-CV partition count is not 100.")
    if outer.get("heldout_role") != "TEST":
        raise RuntimeError("05h5b held-out role changed from TEST.")
    if outer.get("training_role") != "TRAIN":
        raise RuntimeError("05h5b training role changed from TRAIN.")

    existing = contract.get("existing_result_structure", {})
    if len(existing.get("model_ids", [])) != EXPECTED_MODELS:
        raise RuntimeError("05h5b existing TARGET model roster is not 9 models.")
    if int(existing.get("fold_metric_rows")) != 900:
        raise RuntimeError("05h5b fold-metric row count changed.")
    if int(existing.get("oof_prediction_rows")) != 15480:
        raise RuntimeError("05h5b OOF row count changed.")
    if int(existing.get("model_summary_rows")) != 9:
        raise RuntimeError("05h5b model-summary row count changed.")


def extract_function_source(path: Path, name: str) -> str:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    tree = ast.parse(text)

    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one function {name!r} in {path}; found {len(matches)}."
        )
    node = matches[0]
    start = int(node.lineno)
    end = int(getattr(node, "end_lineno", start))
    return "\n".join(lines[start - 1:end]) + "\n"


def verify_frozen_function_hashes(
    contract: dict[str, Any],
    sources: dict[str, Path],
) -> None:
    records = contract.get("frozen_functions")
    if not isinstance(records, list):
        raise RuntimeError("05h5b contract lacks frozen_functions.")

    role_for_path = {
        str(sources["frozen_04c_metric_implementation"].relative_to(ROOT)):
            sources["frozen_04c_metric_implementation"],
        str(sources["TARGET_runner_v3_bytesafe"].relative_to(ROOT)):
            sources["TARGET_runner_v3_bytesafe"],
    }

    for row in records:
        if not isinstance(row, dict):
            raise RuntimeError("Malformed frozen-function record.")
        rel = str(row.get("relative_path"))
        name = str(row.get("function"))
        expected = str(row.get("source_sha256"))

        path = role_for_path.get(rel)
        if path is None:
            path = ROOT / rel
        if not path.exists():
            raise FileNotFoundError(f"Frozen function source path missing: {path}")

        observed = sha256_text(extract_function_source(path, name))
        if observed != expected:
            raise RuntimeError(
                f"Frozen function source changed: {rel}::{name}"
            )


def import_module_from_path(path: Path, module_name: str) -> Any:
    """
    Import a frozen Python source file with normal import semantics.

    The failed 05h5c implementation created a module object but did not register
    it in sys.modules before exec_module(). That is not equivalent to a normal
    Python import: decorators such as @dataclass inspect sys.modules using
    cls.__module__ while the module body is being executed.

    This is an implementation-only correction. The frozen 04c source file is
    not modified, and its SHA256/function hashes are verified before import.
    """
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not create import spec for {path}")

    if module_name in sys.modules:
        raise RuntimeError(
            f"Dynamic import module name already exists in sys.modules: {module_name}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        # Do not leave a partially initialized frozen module registered.
        sys.modules.pop(module_name, None)
        raise

    if sys.modules.get(module_name) is not module:
        raise RuntimeError(
            "Frozen 04c module registration changed unexpectedly during import."
        )

    if Path(getattr(module, "__file__", "")).resolve() != path.resolve():
        raise RuntimeError(
            "Dynamically imported module __file__ does not equal the locked source path."
        )

    return module


def load_endpoint(path: Path) -> pd.DataFrame:
    ep = pd.read_csv(path, sep="\t", low_memory=False)
    required = [
        "execution_index_86",
        "sample_id",
        "case_key",
        "os_time_days",
        "os_event",
    ]
    missing = [c for c in required if c not in ep.columns]
    if missing:
        raise RuntimeError(f"TARGET86 endpoint lost columns: {missing}")

    if len(ep) != EXPECTED_N:
        raise RuntimeError(f"TARGET86 endpoint has {len(ep)} rows, expected 86.")

    ep["execution_index_86"] = pd.to_numeric(
        ep["execution_index_86"], errors="raise"
    ).astype(int)
    ep["os_time_days"] = pd.to_numeric(
        ep["os_time_days"], errors="raise"
    ).astype(float)
    ep["os_event"] = pd.to_numeric(
        ep["os_event"], errors="raise"
    ).astype(int)

    if ep["execution_index_86"].nunique() != EXPECTED_N:
        raise RuntimeError("TARGET86 execution indices are not unique.")
    if int(ep["os_event"].sum()) != EXPECTED_EVENTS:
        raise RuntimeError("TARGET86 event count changed from 29.")
    if int((ep["os_event"] == 0).sum()) != EXPECTED_CENSORED:
        raise RuntimeError("TARGET86 censored count changed from 57.")
    if (ep["os_time_days"] < 0).any():
        raise RuntimeError("TARGET86 contains negative time.")

    zero = ep[ep["os_time_days"] == 0]
    if len(zero) != 1 or int(zero.iloc[0]["os_event"]) != 0:
        raise RuntimeError("TARGET86 time-zero semantics changed.")

    return ep.set_index("execution_index_86", drop=False).sort_index()


def load_fold_structure(contract: dict[str, Any]) -> pd.DataFrame:
    outer = contract["outer_cv"]
    expected_hash = str(outer["fold_structure_sha256"])
    if sha256_file(FOLD_STRUCTURE) != expected_hash:
        raise RuntimeError("05h5b frozen fold-structure file hash changed.")

    fs = pd.read_csv(FOLD_STRUCTURE, sep="\t", low_memory=False)
    required = [
        "repeat", "outer_fold",
        "n_train", "events_train",
        "n_test", "events_test",
        "train_indices_csv", "test_indices_csv",
    ]
    missing = [c for c in required if c not in fs.columns]
    if missing:
        raise RuntimeError(f"Frozen fold structure lost columns: {missing}")

    if len(fs) != 100:
        raise RuntimeError(f"Frozen fold structure has {len(fs)} rows, expected 100.")
    if fs.duplicated(["repeat", "outer_fold"]).any():
        raise RuntimeError("Frozen fold structure has duplicate repeat/fold rows.")

    return fs.sort_values(["repeat", "outer_fold"]).reset_index(drop=True)


def parse_index_csv(value: Any) -> list[int]:
    text = str(value).strip()
    if not text:
        return []
    return [int(x) for x in text.split(",") if str(x).strip()]


def make_surv(time: np.ndarray, event: np.ndarray) -> np.ndarray:
    return Surv.from_arrays(
        event=np.asarray(event, dtype=bool),
        time=np.asarray(time, dtype=float),
    )


def evaluate_step_function(
    step_times: np.ndarray,
    step_surv: np.ndarray,
    eval_times: np.ndarray,
) -> np.ndarray:
    step_times = np.asarray(step_times, dtype=float)
    step_surv = np.asarray(step_surv, dtype=float)
    eval_times = np.asarray(eval_times, dtype=float)

    idx = np.searchsorted(step_times, eval_times, side="right") - 1
    out = np.ones(len(eval_times), dtype=float)
    mask = idx >= 0
    out[mask] = step_surv[idx[mask]]
    return out


def manual_km_at_times(
    time: np.ndarray,
    event: np.ndarray,
    eval_times: np.ndarray,
) -> np.ndarray:
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=bool)
    eval_times = np.asarray(eval_times, dtype=float)

    unique_times = np.unique(time)
    surv = 1.0
    km_times: list[float] = []
    km_surv: list[float] = []

    for t in unique_times:
        at_risk = int(np.sum(time >= t))
        d = int(np.sum((time == t) & event))
        if at_risk <= 0:
            raise RuntimeError("Invalid KM risk set.")
        if d > 0:
            surv *= (1.0 - d / at_risk)
        km_times.append(float(t))
        km_surv.append(float(surv))

    return evaluate_step_function(
        np.asarray(km_times, dtype=float),
        np.asarray(km_surv, dtype=float),
        eval_times,
    )


def sksurv_km_at_times(
    time: np.ndarray,
    event: np.ndarray,
    eval_times: np.ndarray,
) -> np.ndarray:
    km_time, km_prob = kaplan_meier_estimator(
        np.asarray(event, dtype=bool),
        np.asarray(time, dtype=float),
    )
    return evaluate_step_function(km_time, km_prob, eval_times)


def replay_existing_ibs_aggregation(
    m04c: Any,
    fold_metrics_path: Path,
    model_summary_path: Path,
) -> tuple[str, pd.DataFrame]:
    fm = pd.read_csv(fold_metrics_path, sep="\t", low_memory=False)
    ms = pd.read_csv(model_summary_path, sep="\t", low_memory=False)

    fm["ibs"] = pd.to_numeric(fm["ibs"], errors="coerce")
    fm["n_test"] = pd.to_numeric(fm["n_test"], errors="raise").astype(int)

    if not np.isfinite(fm["ibs"]).all():
        raise RuntimeError("Existing TARGET fold IBS contains non-finite values.")
    if len(ms) != EXPECTED_MODELS:
        raise RuntimeError("Existing TARGET model summary is not 9 rows.")

    # Provide both common model-column names so the exact historical helper can
    # be exercised without guessing which alias it used internally.
    helper_df = fm.copy()
    helper_df["model"] = helper_df["model_id"].astype(str)
    helper_df["model_id"] = helper_df["model_id"].astype(str)

    rows: list[dict[str, Any]] = []

    helper_matches_all = True
    helper_values: dict[str, float] = {}
    helper_error = ""

    try:
        for model in ms["model_id"].astype(str):
            value = float(m04c.weighted_mean_ibs(helper_df, model))
            helper_values[model] = value
            expected = float(
                ms.loc[ms["model_id"].astype(str) == model, "ibs"].iloc[0]
            )
            if abs(value - expected) > AGG_REPLAY_TOL:
                helper_matches_all = False
    except Exception as exc:
        helper_matches_all = False
        helper_error = f"{type(exc).__name__}: {exc}"

    candidate_names = [
        "simple_mean_all_folds",
        "n_test_weighted_all_folds",
        "mean_of_repeat_simple_means",
        "mean_of_repeat_n_test_weighted_means",
    ]

    candidate_match_counts = {name: 0 for name in candidate_names}
    candidate_values: dict[str, dict[str, float]] = {
        name: {} for name in candidate_names
    }

    for model in ms["model_id"].astype(str):
        z = fm[fm["model_id"].astype(str) == model].copy()
        expected = float(
            ms.loc[ms["model_id"].astype(str) == model, "ibs"].iloc[0]
        )

        simple = float(z["ibs"].mean())
        weighted = float(np.average(z["ibs"], weights=z["n_test"]))

        repeat_simple = float(
            z.groupby("repeat", sort=True)["ibs"].mean().mean()
        )

        repeat_weighted_values = []
        for _, part in z.groupby("repeat", sort=True):
            repeat_weighted_values.append(
                float(np.average(part["ibs"], weights=part["n_test"]))
            )
        repeat_weighted = float(np.mean(repeat_weighted_values))

        vals = {
            "simple_mean_all_folds": simple,
            "n_test_weighted_all_folds": weighted,
            "mean_of_repeat_simple_means": repeat_simple,
            "mean_of_repeat_n_test_weighted_means": repeat_weighted,
        }

        for name, value in vals.items():
            candidate_values[name][model] = value
            if abs(value - expected) <= AGG_REPLAY_TOL:
                candidate_match_counts[name] += 1

        rows.append(
            {
                "model_id": model,
                "published_existing_ibs": expected,
                "historical_helper_ibs": helper_values.get(model, np.nan),
                "historical_helper_abs_diff": (
                    abs(helper_values[model] - expected)
                    if model in helper_values else np.nan
                ),
                **{
                    f"{name}_ibs": vals[name]
                    for name in candidate_names
                },
                **{
                    f"{name}_abs_diff": abs(vals[name] - expected)
                    for name in candidate_names
                },
            }
        )

    replay = pd.DataFrame(rows)

    if helper_matches_all:
        return "frozen_04c_weighted_mean_ibs_helper", replay

    matching_candidates = [
        name for name, count in candidate_match_counts.items()
        if count == EXPECTED_MODELS
    ]
    if not matching_candidates:
        raise RuntimeError(
            "Could not reproduce existing TARGET model-summary IBS aggregation "
            f"with the frozen helper or prespecified candidate formulas. "
            f"helper_error={helper_error!r}; "
            f"candidate_match_counts={candidate_match_counts}"
        )

    # If several formulas are numerically equivalent on the exact 20x5 design,
    # use the n_test-weighted form when available because that preserves each
    # patient's contribution under unequal 17/18-person folds.
    preferred_order = [
        "n_test_weighted_all_folds",
        "mean_of_repeat_n_test_weighted_means",
        "simple_mean_all_folds",
        "mean_of_repeat_simple_means",
    ]
    for name in preferred_order:
        if name in matching_candidates:
            return name, replay

    raise RuntimeError("Aggregation resolution fell through unexpectedly.")


def aggregate_new_km_ibs(
    method: str,
    m04c: Any,
    fold_df: pd.DataFrame,
) -> float:
    if method == "frozen_04c_weighted_mean_ibs_helper":
        helper_df = fold_df.copy()
        helper_df["model"] = "KM_NULL"
        helper_df["model_id"] = "KM_NULL"
        return float(m04c.weighted_mean_ibs(helper_df, "KM_NULL"))

    if method == "n_test_weighted_all_folds":
        return float(np.average(fold_df["ibs"], weights=fold_df["n_test"]))

    if method == "mean_of_repeat_n_test_weighted_means":
        vals = []
        for _, part in fold_df.groupby("repeat", sort=True):
            vals.append(float(np.average(part["ibs"], weights=part["n_test"])))
        return float(np.mean(vals))

    if method == "simple_mean_all_folds":
        return float(fold_df["ibs"].mean())

    if method == "mean_of_repeat_simple_means":
        return float(
            fold_df.groupby("repeat", sort=True)["ibs"].mean().mean()
        )

    raise RuntimeError(f"Unknown resolved aggregation method: {method}")


def compute_km_reference(
    ep: pd.DataFrame,
    fs: pd.DataFrame,
    m04c: Any,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rows: list[dict[str, Any]] = []

    max_manual_vs_sksurv = 0.0
    min_grid_n = math.inf
    max_grid_n = 0

    for row in fs.itertuples(index=False):
        repeat = int(row.repeat)
        fold = int(row.outer_fold)
        train_idx = parse_index_csv(row.train_indices_csv)
        test_idx = parse_index_csv(row.test_indices_csv)

        if len(train_idx) != int(row.n_train):
            raise RuntimeError(
                f"repeat={repeat}, fold={fold}: train-index count changed."
            )
        if len(test_idx) != int(row.n_test):
            raise RuntimeError(
                f"repeat={repeat}, fold={fold}: test-index count changed."
            )
        if set(train_idx) & set(test_idx):
            raise RuntimeError(
                f"repeat={repeat}, fold={fold}: train/test overlap."
            )
        if set(train_idx) | set(test_idx) != set(ep.index.astype(int)):
            raise RuntimeError(
                f"repeat={repeat}, fold={fold}: partition does not cover TARGET86."
            )

        train_time = ep.loc[train_idx, "os_time_days"].to_numpy(dtype=float)
        train_event = ep.loc[train_idx, "os_event"].to_numpy(dtype=int)
        test_time = ep.loc[test_idx, "os_time_days"].to_numpy(dtype=float)
        test_event = ep.loc[test_idx, "os_event"].to_numpy(dtype=int)

        if int(train_event.sum()) != int(row.events_train):
            raise RuntimeError(
                f"repeat={repeat}, fold={fold}: training-event count changed."
            )
        if int(test_event.sum()) != int(row.events_test):
            raise RuntimeError(
                f"repeat={repeat}, fold={fold}: test-event count changed."
            )

        y_train = make_surv(train_time, train_event)
        y_test = make_surv(test_time, test_event)

        grid = np.asarray(
            m04c.choose_ibs_grid(y_train, y_test),
            dtype=float,
        )
        if grid.ndim != 1 or len(grid) < 5:
            raise RuntimeError(
                f"repeat={repeat}, fold={fold}: frozen IBS grid invalid, n={len(grid)}."
            )
        if not np.all(np.diff(grid) > 0):
            raise RuntimeError(
                f"repeat={repeat}, fold={fold}: frozen IBS grid is not strictly increasing."
            )

        manual_surv = manual_km_at_times(
            train_time,
            train_event,
            grid,
        )
        sksurv_surv = sksurv_km_at_times(
            train_time,
            train_event,
            grid,
        )

        diff = float(np.max(np.abs(manual_surv - sksurv_surv)))
        max_manual_vs_sksurv = max(max_manual_vs_sksurv, diff)
        if diff > KM_CROSSCHECK_TOL:
            raise RuntimeError(
                f"repeat={repeat}, fold={fold}: manual and sksurv KM disagree "
                f"by {diff:.3e}."
            )

        if np.any(sksurv_surv < -1e-15) or np.any(sksurv_surv > 1.0 + 1e-15):
            raise RuntimeError(
                f"repeat={repeat}, fold={fold}: KM survival leaves [0,1]."
            )
        if np.any(np.diff(sksurv_surv) > 1e-12):
            raise RuntimeError(
                f"repeat={repeat}, fold={fold}: KM survival is not non-increasing."
            )

        # No-covariate null: every held-out patient receives the SAME
        # outer-training-only KM survival curve.
        survival_prob = np.repeat(
            sksurv_surv[np.newaxis, :],
            repeats=len(test_idx),
            axis=0,
        )
        if survival_prob.shape != (len(test_idx), len(grid)):
            raise RuntimeError("Unexpected KM null survival-probability matrix shape.")
        if not np.allclose(
            survival_prob,
            survival_prob[0:1, :],
            rtol=0.0,
            atol=0.0,
        ):
            raise RuntimeError("KM null does not assign identical curves to all test patients.")

        ibs = float(
            m04c.integrated_brier_custom(
                y_train,
                y_test,
                survival_prob,
                grid,
            )
        )
        if not math.isfinite(ibs):
            raise RuntimeError(
                f"repeat={repeat}, fold={fold}: non-finite KM-null IBS."
            )

        min_grid_n = min(min_grid_n, len(grid))
        max_grid_n = max(max_grid_n, len(grid))

        rows.append(
            {
                "repeat": repeat,
                "outer_fold": fold,
                "model_id": "KM_NULL",
                "n_train": len(train_idx),
                "events_train": int(train_event.sum()),
                "n_test": len(test_idx),
                "events_test": int(test_event.sum()),
                "ibs": ibs,
                "ibs_status": "PASS",
                "ibs_grid_n": len(grid),
                "ibs_grid_min": float(grid[0]),
                "ibs_grid_max": float(grid[-1]),
                "km_survival_at_grid_start": float(sksurv_surv[0]),
                "km_survival_at_grid_end": float(sksurv_surv[-1]),
                "zero_time_case_in_train": bool(
                    19 in train_idx
                ),
                "zero_time_case_in_test": bool(
                    19 in test_idx
                ),
            }
        )

    out = pd.DataFrame(rows)
    if len(out) != 100:
        raise RuntimeError(f"Expected 100 KM-null fold rows; got {len(out)}.")
    if not np.isfinite(out["ibs"]).all():
        raise RuntimeError("KM-null fold IBS contains non-finite values.")

    audit = {
        "max_manual_vs_sksurv_km_abs_difference": max_manual_vs_sksurv,
        "min_ibs_grid_points": int(min_grid_n),
        "max_ibs_grid_points": int(max_grid_n),
        "folds_with_finite_ibs": int(np.isfinite(out["ibs"]).sum()),
    }
    return out, audit


def existing_model_comparison(
    model_summary_path: Path,
    km_ibs: float,
) -> pd.DataFrame:
    ms = pd.read_csv(model_summary_path, sep="\t", low_memory=False)
    ms["ibs"] = pd.to_numeric(ms["ibs"], errors="coerce")
    if not np.isfinite(ms["ibs"]).all():
        raise RuntimeError("Existing TARGET model summary contains non-finite IBS.")

    out = ms[
        [
            "model_id",
            "ibs",
            "ibs_valid_repeats",
            "risk_sd",
            "risk_q99_q01",
        ]
    ].copy()
    out = out.rename(columns={"ibs": "existing_model_ibs"})
    out["km_null_ibs"] = float(km_ibs)
    out["existing_model_minus_km_null_ibs"] = (
        out["existing_model_ibs"] - float(km_ibs)
    )
    out["lower_ibs_than_km_null"] = (
        out["existing_model_minus_km_null_ibs"] < 0.0
    )
    out["comparison_role"] = (
        "DESCRIPTIVE CONTEXT ONLY; NOT MODEL SELECTION"
    )
    return out


def build_readme(
    km_ibs: float,
    fold_df: pd.DataFrame,
    aggregation_method: str,
    audit: dict[str, float],
    comparison: pd.DataFrame,
) -> str:
    lines = [
        "Paper 6 - 05h5c TARGET no-covariate Kaplan-Meier IBS reference",
        "",
        "Role",
        "----",
        "POST-OPENING DESCRIPTIVE NULL REFERENCE.",
        "This reference cannot be used for tuning, model selection, branch reassignment,",
        "or reopening any frozen Paper-6 decision.",
        "",
        "Construction",
        "------------",
        "For each of the exact 100 frozen outer partitions (20 repeats x 5 folds),",
        "a no-covariate Kaplan-Meier survival curve was estimated using only that",
        "outer-training fold's TARGET86 time/event outcomes.",
        "",
        "Every patient in the corresponding outer-test fold received exactly the same",
        "training-derived survival curve. No expression, risk score, or model prediction",
        "entered the null reference.",
        "",
        "The IBS grid was generated by the exact frozen 04c choose_ibs_grid() helper.",
        "IBS was calculated by the exact frozen 04c integrated_brier_custom() helper,",
        "thereby retaining the original censoring/IPCW mechanics and time support.",
        "",
        f"Resolved frozen aggregation method: {aggregation_method}",
        "",
        "Result",
        "------",
        f"TARGET86 no-covariate KM IBS reference: {km_ibs:.10f}",
        f"Finite fold IBS values: {int(np.isfinite(fold_df['ibs']).sum())}/100",
        f"Fold IBS median: {float(fold_df['ibs'].median()):.10f}",
        f"Fold IBS range: {float(fold_df['ibs'].min()):.10f} to {float(fold_df['ibs'].max()):.10f}",
        f"IBS-grid points per fold: {int(audit['min_ibs_grid_points'])} to {int(audit['max_ibs_grid_points'])}",
        f"Max manual-vs-sksurv KM discrepancy: {audit['max_manual_vs_sksurv_km_abs_difference']:.3e}",
        "",
        "Existing frozen TARGET model IBS relative to KM null",
        "----------------------------------------------------",
    ]

    for row in comparison.itertuples(index=False):
        lines.append(
            f"{row.model_id}: existing IBS={float(row.existing_model_ibs):.10f}; "
            f"model-minus-KM={float(row.existing_model_minus_km_null_ibs):+.10f}"
        )

    lines += [
        "",
        "Interpretation",
        "--------------",
        "A negative model-minus-KM difference means that the existing frozen model had",
        "lower prediction error than the fold-local no-covariate KM reference; a positive",
        "difference means higher prediction error. These differences are descriptive and",
        "do not alter the original model/branch decisions.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    print("=" * 118)
    print("Paper 6 - TARGET no-covariate Kaplan-Meier IBS reference")
    print("=" * 118)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {ROOT}")
    print(f"Output directory: {OUT_DIR}")
    print()
    print("Execution contract:")
    print("  analysis role: POST-OPENING DESCRIPTIVE NULL REFERENCE")
    print("  new model fitting: NO")
    print("  expression / molecular features read: NO")
    print("  TARGET outcome use: ONLY exact already-opened outer-training folds")
    print("  GSE21257/GSE39055 outcomes read: NO")
    print("  fold structure changed: NO")
    print("  IBS grid/censoring implementation changed: NO")
    print("  model/branch selection reopened: NO")
    print()

    if OUT_DIR.exists():
        raise RuntimeError(
            f"05h5d final output already exists; refusing overwrite: {OUT_DIR}"
        )
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir(parents=True, exist_ok=False)

    try:
        h0_block = verify_05h0()
        h5b = verify_05h5b()
        verify_contract_semantics(h5b)
        sources = resolve_and_verify_locked_sources(h5b)
        verify_frozen_function_hashes(h5b, sources)

        print("Frozen contract / mechanics chain: PASS")
        print(f"  05h0 SHA256: {sha256_file(H0_JSON)}")
        print(f"  05h5b contract SHA256: {sha256_file(H5B_CONTRACT)}")
        print("  all 05h5b-locked source hashes: PASS")
        print("  frozen helper-function source hashes: PASS")
        print()

        m04c = import_module_from_path(
            sources["frozen_04c_metric_implementation"],
            "_paper6_frozen_04c_for_05h5d",
        )

        for fn in [
            "choose_ibs_grid",
            "integrated_brier_custom",
            "weighted_mean_ibs",
        ]:
            if not callable(getattr(m04c, fn, None)):
                raise RuntimeError(
                    f"Frozen 04c module does not expose callable {fn}."
                )

        print("Frozen 04c dynamic-import smoke test: PASS")
        print(f"  imported file: {Path(m04c.__file__).resolve()}")
        print(f"  module registered in sys.modules: {sys.modules.get(m04c.__name__) is m04c}")
        print("  required IBS helpers callable: YES")
        print()

        aggregation_method, aggregation_replay = replay_existing_ibs_aggregation(
            m04c,
            sources["TARGET_fold_metrics"],
            sources["TARGET_model_summary"],
        )
        aggregation_replay.to_csv(
            WORK_DIR / "existing_target_ibs_aggregation_replay.tsv",
            sep="\t",
            index=False,
            float_format="%.12g",
        )

        print("Existing TARGET IBS aggregation replay: PASS")
        print(f"  resolved method: {aggregation_method}")
        print("  existing model-summary IBS reproduced for all 9 frozen models: YES")
        print()

        ep = load_endpoint(sources["TARGET86_endpoint"])
        fs = load_fold_structure(h5b)

        print("TARGET86 / frozen folds: PASS")
        print("  n/events/censored: 86/29/57")
        print("  frozen partitions: 100")
        print("  repeats x folds: 20 x 5")
        print()

        fold_df, km_audit = compute_km_reference(ep, fs, m04c)
        km_ibs = aggregate_new_km_ibs(
            aggregation_method,
            m04c,
            fold_df,
        )
        if not math.isfinite(km_ibs):
            raise RuntimeError("Aggregate KM-null IBS is non-finite.")

        comparison = existing_model_comparison(
            sources["TARGET_model_summary"],
            km_ibs,
        )

        print("Fold-local KM reference calculation: PASS")
        print("  outer-training-only KM fits: 100/100")
        print("  identical curve assigned to every patient within each test fold: YES")
        print(
            f"  manual-vs-sksurv KM max abs diff: "
            f"{km_audit['max_manual_vs_sksurv_km_abs_difference']:.3e}"
        )
        print(
            f"  IBS grid points per fold: "
            f"{km_audit['min_ibs_grid_points']}-"
            f"{km_audit['max_ibs_grid_points']}"
        )
        print("  finite fold IBS: 100/100")
        print()

        print("-" * 118)
        print("05h5d TARGET NO-COVARIATE KM IBS REFERENCE")
        print("-" * 118)
        print(f"KM-null IBS: {km_ibs:.10f}")
        print(f"Fold IBS median: {float(fold_df['ibs'].median()):.10f}")
        print(
            f"Fold IBS range: {float(fold_df['ibs'].min()):.10f} "
            f"to {float(fold_df['ibs'].max()):.10f}"
        )
        print()

        print("Existing frozen TARGET model IBS relative to KM null:")
        display = comparison[
            [
                "model_id",
                "existing_model_ibs",
                "km_null_ibs",
                "existing_model_minus_km_null_ibs",
                "lower_ibs_than_km_null",
            ]
        ]
        print(
            display.to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}",
            )
        )
        print()

        print("Interpretation guardrails:")
        print("  KM null used for tuning/selection: NO")
        print("  T-A/T-B/T-C/T-D changed: NO")
        print("  05d HOLD changed: NO")
        print("  existing TARGET model estimates changed: NO")
        print()

        fold_df.to_csv(
            WORK_DIR / "TARGET_KM_null_fold_metrics.tsv",
            sep="\t",
            index=False,
            float_format="%.12g",
        )
        comparison.to_csv(
            WORK_DIR / "TARGET_existing_models_vs_KM_null.tsv",
            sep="\t",
            index=False,
            float_format="%.12g",
        )

        summary_df = pd.DataFrame(
            [
                {
                    "reference_id": "KM_NULL_NO_COVARIATE",
                    "analysis_role": "POST-OPENING DESCRIPTIVE NULL REFERENCE",
                    "TARGET_n": EXPECTED_N,
                    "TARGET_events": EXPECTED_EVENTS,
                    "TARGET_censored": EXPECTED_CENSORED,
                    "outer_repeats": EXPECTED_REPEATS,
                    "outer_folds": EXPECTED_FOLDS,
                    "valid_fold_ibs": int(np.isfinite(fold_df["ibs"]).sum()),
                    "km_null_ibs": km_ibs,
                    "fold_ibs_median": float(fold_df["ibs"].median()),
                    "fold_ibs_min": float(fold_df["ibs"].min()),
                    "fold_ibs_max": float(fold_df["ibs"].max()),
                    "aggregation_method": aggregation_method,
                    "min_ibs_grid_points": int(km_audit["min_ibs_grid_points"]),
                    "max_ibs_grid_points": int(km_audit["max_ibs_grid_points"]),
                    "max_manual_vs_sksurv_km_abs_difference": float(
                        km_audit["max_manual_vs_sksurv_km_abs_difference"]
                    ),
                    "model_selection_performed": False,
                    "branch_decision_changed": False,
                }
            ]
        )
        summary_df.to_csv(
            WORK_DIR / "Supplementary_Table_TARGET_KM_null_IBS_reference.csv",
            index=False,
            float_format="%.12g",
        )

        result = {
            "script_version": SCRIPT_VERSION,
            "status": "PASS_POSTOPENING_TARGET_KM_IBS_NULL_REFERENCE",
            "analysis_role": "POST-OPENING DESCRIPTIVE NULL REFERENCE",
            "05h0_sha256": sha256_file(H0_JSON),
            "05h5b_contract_sha256": sha256_file(H5B_CONTRACT),
            "TARGET86": {
                "n": EXPECTED_N,
                "events": EXPECTED_EVENTS,
                "censored": EXPECTED_CENSORED,
            },
            "method": {
                "fold_structure": "exact frozen 20x5 TARGET outer partitions",
                "km_fit": "no-covariate Kaplan-Meier on outer-training outcomes only",
                "test_prediction": "same training-derived survival curve for every outer-test patient",
                "ibs_grid": "exact frozen 04c choose_ibs_grid",
                "ibs": "exact frozen 04c integrated_brier_custom",
                "aggregation": aggregation_method,
            },
            "km_null_ibs": km_ibs,
            "fold_ibs": {
                "valid": int(np.isfinite(fold_df["ibs"]).sum()),
                "median": float(fold_df["ibs"].median()),
                "min": float(fold_df["ibs"].min()),
                "max": float(fold_df["ibs"].max()),
            },
            "km_crosscheck": km_audit,
            "existing_models_vs_null": comparison.to_dict(orient="records"),
            "prohibited_use_respected": True,
            "model_selection_performed": False,
            "branch_decision_changed": False,
            "05d_hold_changed": False,
        }
        write_json(
            WORK_DIR / "target_km_ibs_reference.json",
            result,
        )

        (WORK_DIR / "README_for_supplement.txt").write_text(
            build_readme(
                km_ibs,
                fold_df,
                aggregation_method,
                km_audit,
                comparison,
            ),
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
            "status": "PASS_POSTOPENING_TARGET_KM_IBS_NULL_REFERENCE",
            "inputs": {
                "05h0_contract": {
                    "path": str(H0_JSON.relative_to(ROOT)),
                    "sha256": sha256_file(H0_JSON),
                },
                "05h5b_contract": {
                    "path": str(H5B_CONTRACT.relative_to(ROOT)),
                    "sha256": sha256_file(H5B_CONTRACT),
                },
                **{
                    role: {
                        "path": str(path.relative_to(ROOT)),
                        "sha256": sha256_file(path),
                    }
                    for role, path in sources.items()
                    if role in {
                        "TARGET86_endpoint",
                        "TARGET_fold_metrics",
                        "TARGET_model_summary",
                        "frozen_04c_metric_implementation",
                    }
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
                raise RuntimeError(
                    f"05h5d output changed before finalization: {p.name}"
                )

        os.replace(WORK_DIR, OUT_DIR)

        print("=" * 118)
        print("05h5d TARGET KM-IBS null reference: PASS")
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
        print("05h5d TARGET KM-IBS null reference: FAIL")
        print(f"{type(exc).__name__}: {exc}")
        print("=" * 118)
        raise
