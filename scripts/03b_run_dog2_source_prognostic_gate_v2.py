#!/usr/bin/env python3
"""
Paper 6 - run frozen DOG2 Source Prognostic Gate.

This is the FIRST Paper-6 stage that intentionally reads the locked DOG2
outcome values.

It executes the 03a protocol under the 02i scientific thresholds:
- Primary endpoint: DOG2 OS.
- Secondary endpoint: DOG2 DFI.
- Primary model: Hallmark-50 ridge Cox.
- Supporting OS model: 11,815-gene elastic-net Cox.
- Primary random CV: 5-fold x 20 repeats, inner 4-fold.
- Supporting OS random CV: 5-fold x 5 repeats, inner 3-fold.
- Arm->arm: COTC021->COTC022 and COTC022->COTC021.
- Patient bootstrap uncertainty.
- Supporting OS outcome-pair permutation inference within randomized arm.

Human outcomes are never read.

Restart behavior
----------------
Validated completed stage artifacts are reused. Long OS permutation inference is
checkpointed after every permutation. Existing checkpoints are accepted only when
their protocol SHA256 and stage identity match the current frozen 03a protocol.

No CLI arguments.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.model_selection import StratifiedKFold
    from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
    from sksurv.metrics import concordance_index_ipcw
    from sksurv.util import Surv
except ImportError as exc:
    raise ImportError(
        "03b requires scikit-survival in the active Paper-6 .venv. "
        "Install with: python -m pip install scikit-survival"
    ) from exc


SCRIPT_VERSION = "03b-run-dog2-source-prognostic-gate-v2-implementation-fix-no-cli"

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "_config"
PAPER4_BASENAME = "paper4_sarcoma_dog"

I_DIR = ROOT / "results" / "revised_design" / "02i"
I_CONTRACT = I_DIR / "revised_model_benchmark_source_gate_contract.json"

P_DIR = ROOT / "results" / "source_gate_protocol" / "03a"
P_PROTOCOL = P_DIR / "source_prognostic_gate_protocol.json"
P_SUMMARY = P_DIR / "summary.json"
P_ENDPOINTS = P_DIR / "endpoint_header_contract.tsv"
P_ROSTER = P_DIR / "source_sample_arm_roster.tsv"
P_HALLMARK = P_DIR / "hallmark_dog2_feature_map.tsv"
P_RIDGE_GRID = P_DIR / "ridge_alpha_grid.tsv"
P_ELASTIC = P_DIR / "elastic_net_path_contract.tsv"

UPSTREAM_LOCK = ROOT / "contracts" / "00_upstream_input_lock.json"

OUT_DIR = ROOT / "results" / "source_gate" / "03b"
CHECKPOINT_DIR = OUT_DIR / "checkpoints"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_AUDIT = OUT_DIR / "input_identity_audit.tsv"

OS_PRIMARY_PRED = OUT_DIR / "os_primary_random_cv_predictions.tsv"
OS_PRIMARY_FOLDS = OUT_DIR / "os_primary_random_cv_folds.tsv"
OS_PRIMARY_TUNING = OUT_DIR / "os_primary_random_cv_tuning.tsv"
OS_PRIMARY_REPEATS = OUT_DIR / "os_primary_repeat_metrics.tsv"
OS_PRIMARY_BOOT = OUT_DIR / "os_primary_patient_bootstrap.tsv"
OS_PRIMARY_STAGE = CHECKPOINT_DIR / "os_primary_random_cv_stage.json"

OS_SUPPORT_PRED = OUT_DIR / "os_supporting_gene_random_cv_predictions.tsv"
OS_SUPPORT_FOLDS = OUT_DIR / "os_supporting_gene_random_cv_folds.tsv"
OS_SUPPORT_TUNING = OUT_DIR / "os_supporting_gene_random_cv_tuning.tsv"
OS_SUPPORT_REPEATS = OUT_DIR / "os_supporting_gene_repeat_metrics.tsv"
OS_SUPPORT_BOOT = OUT_DIR / "os_supporting_gene_patient_bootstrap.tsv"
OS_SUPPORT_STAGE = CHECKPOINT_DIR / "os_supporting_gene_stage.json"

OS_ARM_METRICS = OUT_DIR / "os_arm_to_arm_metrics.tsv"
OS_ARM_BOOT = OUT_DIR / "os_arm_to_arm_bootstrap.tsv"
OS_ARM_TUNING = OUT_DIR / "os_arm_to_arm_tuning.tsv"
OS_ARM_STAGE = CHECKPOINT_DIR / "os_arm_stage.json"

DFI_PRIMARY_PRED = OUT_DIR / "dfi_primary_random_cv_predictions.tsv"
DFI_PRIMARY_FOLDS = OUT_DIR / "dfi_primary_random_cv_folds.tsv"
DFI_PRIMARY_TUNING = OUT_DIR / "dfi_primary_random_cv_tuning.tsv"
DFI_PRIMARY_REPEATS = OUT_DIR / "dfi_primary_repeat_metrics.tsv"
DFI_PRIMARY_BOOT = OUT_DIR / "dfi_primary_patient_bootstrap.tsv"
DFI_PRIMARY_STAGE = CHECKPOINT_DIR / "dfi_primary_random_cv_stage.json"

DFI_ARM_METRICS = OUT_DIR / "dfi_arm_to_arm_metrics.tsv"
DFI_ARM_BOOT = OUT_DIR / "dfi_arm_to_arm_bootstrap.tsv"
DFI_ARM_TUNING = OUT_DIR / "dfi_arm_to_arm_tuning.tsv"
DFI_ARM_STAGE = CHECKPOINT_DIR / "dfi_arm_stage.json"

PERM_CHECKPOINT = CHECKPOINT_DIR / "os_primary_permutation_checkpoint.json"
PERM_RESULTS = OUT_DIR / "os_primary_permutation_results.tsv"

GATE_RESULTS = OUT_DIR / "source_gate_results.json"
SUMMARY_JSON = OUT_DIR / "summary.json"


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


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def require_file(path: Path) -> Path:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Required frozen artifact missing: {path}")
    return path


def local_path_config() -> Dict[str, Any]:
    path = CONFIG_DIR / "paths.local.json"
    if not path.exists():
        return {}
    payload = read_json(path)
    return payload if isinstance(payload, dict) else {}


def resolve_paper4_root() -> Tuple[Path, str]:
    candidates: List[Tuple[Path, str]] = []

    env_value = os.environ.get("PAPER4_ROOT", "").strip()
    if env_value:
        candidates.append((Path(env_value).expanduser(), "environment:PAPER4_ROOT"))

    config_value = clean(local_path_config().get("paper4_root"))
    if config_value:
        candidates.append((Path(config_value).expanduser(), "_config/paths.local.json"))

    candidates.extend(
        [
            (ROOT.parent / PAPER4_BASENAME, "sibling_repository"),
            (Path.home() / "Desktop" / PAPER4_BASENAME, "home_desktop_fallback"),
        ]
    )

    checked: List[str] = []
    for candidate, source in candidates:
        resolved = candidate.resolve()
        checked.append(f"{source}: {resolved}")
        if resolved.is_dir():
            return resolved, source

    raise FileNotFoundError(
        "Could not resolve Paper 4 root. Checked:\n  - " + "\n  - ".join(checked)
    )


def get_asset(
    lock: Dict[str, Any],
    role: str,
    paper4_root: Path,
) -> Tuple[Path, Dict[str, Any]]:
    item = (lock.get("assets") or {}).get(role)
    if not isinstance(item, dict):
        raise RuntimeError(f"Upstream lock missing asset {role!r}.")

    relative = clean(item.get("relative_path"))
    expected = clean(item.get("sha256")).lower()
    path = paper4_root / relative
    require_file(path)

    actual = sha256_file(path).lower()
    if actual != expected:
        raise RuntimeError(
            f"Hash mismatch for {role}: expected {expected}, observed {actual}"
        )
    return path, item


def canonical_patient_id(value: Any) -> str:
    text = clean(value)
    if not text:
        return ""

    if text.endswith(".0"):
        maybe = text[:-2]
        if maybe.isdigit():
            text = maybe

    if text.isdigit():
        return str(int(text))

    return text


def survival_array(time_values: Sequence[float], event_values: Sequence[bool]) -> np.ndarray:
    return Surv.from_arrays(
        event=np.asarray(event_values, dtype=bool),
        time=np.asarray(time_values, dtype=float),
    )


def tau_from_training(y_train: np.ndarray, quantile: float) -> float:
    times = np.asarray(y_train["time"], dtype=float)
    tau = float(np.quantile(times, quantile))
    max_time = float(np.max(times))
    if not np.isfinite(tau) or tau <= 0:
        raise RuntimeError("Invalid Uno-C tau.")
    return min(tau, np.nextafter(max_time, 0.0))


def safe_uno_c(
    y_train: np.ndarray,
    y_test: np.ndarray,
    risk: np.ndarray,
    quantile: float,
) -> float:
    risk = np.asarray(risk, dtype=float)
    if not np.all(np.isfinite(risk)):
        return float("nan")

    tau = tau_from_training(y_train, quantile)

    try:
        result = concordance_index_ipcw(
            y_train,
            y_test,
            risk,
            tau=tau,
        )
        return float(result[0])
    except ValueError:
        # Rare support mismatch: restrict the evaluation population to follow-up
        # <= tau. This is still the frozen truncated-Uno estimand.
        mask = np.asarray(y_test["time"], dtype=float) <= tau
        if int(mask.sum()) < 10:
            return float("nan")
        try:
            result = concordance_index_ipcw(
                y_train,
                y_test[mask],
                risk[mask],
                tau=tau,
            )
            return float(result[0])
        except Exception:
            return float("nan")
    except Exception:
        return float("nan")


def stratified_splits(
    event: np.ndarray,
    arm: Optional[np.ndarray],
    n_splits: int,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    event_int = np.asarray(event, dtype=int)

    if arm is None:
        labels = event_int.astype(str)
    else:
        labels = np.asarray(
            [f"{a}__{e}" for a, e in zip(arm, event_int)],
            dtype=object,
        )

    counts = pd.Series(labels).value_counts()
    if counts.min() < n_splits:
        raise RuntimeError(
            f"Cannot create {n_splits}-fold stratification; minimum stratum count={counts.min()}."
        )

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    dummy = np.zeros(len(labels))
    return list(splitter.split(dummy, labels))


class HallmarkTransformer:
    def __init__(
        self,
        module_to_feature_indices: Dict[str, np.ndarray],
        module_order: List[str],
        min_genes: int,
        sd_eps: float,
    ):
        self.module_to_feature_indices = module_to_feature_indices
        self.module_order = module_order
        self.min_genes = min_genes
        self.sd_eps = sd_eps

        self.gene_mean_: Optional[np.ndarray] = None
        self.gene_sd_: Optional[np.ndarray] = None
        self.module_valid_indices_: Dict[str, np.ndarray] = {}
        self.module_mean_: Optional[np.ndarray] = None
        self.module_sd_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "HallmarkTransformer":
        X = np.asarray(X, dtype=float)
        self.gene_mean_ = np.mean(X, axis=0)
        self.gene_sd_ = np.std(X, axis=0, ddof=0)

        valid_gene = np.isfinite(self.gene_sd_) & (self.gene_sd_ > self.sd_eps)

        module_raw = []
        self.module_valid_indices_ = {}

        Z = np.zeros_like(X, dtype=float)
        Z[:, valid_gene] = (
            X[:, valid_gene] - self.gene_mean_[valid_gene]
        ) / self.gene_sd_[valid_gene]

        for module in self.module_order:
            indices = np.asarray(self.module_to_feature_indices[module], dtype=int)
            indices = indices[valid_gene[indices]]

            if len(indices) < self.min_genes:
                raise RuntimeError(
                    f"{module}: only {len(indices)} nonzero-variance genes in training fold; "
                    f"minimum={self.min_genes}."
                )

            self.module_valid_indices_[module] = indices
            module_raw.append(np.mean(Z[:, indices], axis=1))

        M = np.column_stack(module_raw)
        self.module_mean_ = np.mean(M, axis=0)
        self.module_sd_ = np.std(M, axis=0, ddof=0)

        if np.any(~np.isfinite(self.module_sd_)):
            raise RuntimeError("Non-finite Hallmark module SD in training fold.")

        # A constant module is not scientifically dropped; its scaled values are zero.
        self.module_sd_[self.module_sd_ <= self.sd_eps] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.gene_mean_ is None or self.gene_sd_ is None:
            raise RuntimeError("HallmarkTransformer not fitted.")

        X = np.asarray(X, dtype=float)
        Z = np.zeros_like(X, dtype=float)

        valid = self.gene_sd_ > self.sd_eps
        Z[:, valid] = (
            X[:, valid] - self.gene_mean_[valid]
        ) / self.gene_sd_[valid]

        module_raw = []
        for module in self.module_order:
            indices = self.module_valid_indices_[module]
            module_raw.append(np.mean(Z[:, indices], axis=1))

        M = np.column_stack(module_raw)
        return (M - self.module_mean_) / self.module_sd_


class GeneTransformer:
    def __init__(self, sd_eps: float):
        self.sd_eps = sd_eps
        self.mean_: Optional[np.ndarray] = None
        self.sd_: Optional[np.ndarray] = None
        self.valid_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "GeneTransformer":
        X = np.asarray(X, dtype=float)
        self.mean_ = np.mean(X, axis=0)
        self.sd_ = np.std(X, axis=0, ddof=0)
        self.valid_ = np.isfinite(self.sd_) & (self.sd_ > self.sd_eps)

        if int(self.valid_.sum()) < 2:
            raise RuntimeError("Too few nonzero-variance genes in training fold.")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.sd_ is None or self.valid_ is None:
            raise RuntimeError("GeneTransformer not fitted.")

        X = np.asarray(X, dtype=float)
        return (
            X[:, self.valid_] - self.mean_[self.valid_]
        ) / self.sd_[self.valid_]


def fit_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float,
) -> CoxPHSurvivalAnalysis:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = CoxPHSurvivalAnalysis(
            alpha=float(alpha),
            ties="breslow",
            n_iter=200,
            tol=1e-9,
        )
        model.fit(X_train, y_train)
    return model


def tune_ridge(
    X: np.ndarray,
    y: np.ndarray,
    event: np.ndarray,
    arm: Optional[np.ndarray],
    module_to_indices: Dict[str, np.ndarray],
    module_order: List[str],
    alphas: Sequence[float],
    inner_splits: int,
    seed: int,
    tau_quantile: float,
    min_module_genes: int,
    sd_eps: float,
    tie_tol: float,
) -> Tuple[float, pd.DataFrame]:
    splits = stratified_splits(event, arm, inner_splits, seed)

    rows: List[Dict[str, Any]] = []

    for inner_fold, (train_idx, val_idx) in enumerate(splits):
        transformer = HallmarkTransformer(
            module_to_indices,
            module_order,
            min_module_genes,
            sd_eps,
        ).fit(X[train_idx])

        X_train_m = transformer.transform(X[train_idx])
        X_val_m = transformer.transform(X[val_idx])

        for alpha in alphas:
            score = float("nan")
            status = "PASS"
            try:
                model = fit_ridge(X_train_m, y[train_idx], float(alpha))
                risk = model.predict(X_val_m)
                score = safe_uno_c(
                    y[train_idx],
                    y[val_idx],
                    risk,
                    tau_quantile,
                )
                if not np.isfinite(score):
                    status = "NONFINITE_UNO"
            except Exception as exc:
                status = f"FIT_FAIL:{type(exc).__name__}"

            rows.append(
                {
                    "inner_fold": inner_fold,
                    "alpha": float(alpha),
                    "uno_c": score,
                    "status": status,
                }
            )

    audit = pd.DataFrame(rows)

    grouped = (
        audit.groupby("alpha", as_index=False)
        .agg(
            mean_uno_c=("uno_c", "mean"),
            n_finite=("uno_c", lambda s: int(np.isfinite(s).sum())),
        )
    )

    grouped = grouped[grouped["n_finite"] == inner_splits].copy()
    if grouped.empty:
        raise RuntimeError("No ridge alpha had finite Uno C in all inner folds.")

    best = float(grouped["mean_uno_c"].max())
    candidates = grouped[grouped["mean_uno_c"] >= best - tie_tol].copy()

    # Stronger ridge = larger alpha.
    chosen = float(candidates["alpha"].max())

    audit = audit.merge(
        grouped.rename(columns={"mean_uno_c": "alpha_mean_uno_c"}),
        on="alpha",
        how="left",
    )
    audit["chosen_alpha"] = chosen

    return chosen, audit


def primary_cv(
    X: np.ndarray,
    y: np.ndarray,
    event: np.ndarray,
    arm: np.ndarray,
    sample_ids: np.ndarray,
    module_to_indices: Dict[str, np.ndarray],
    module_order: List[str],
    ridge_alphas: Sequence[float],
    outer_splits: int,
    outer_repeats: int,
    inner_splits: int,
    base_seed: int,
    tau_quantile: float,
    min_module_genes: int,
    sd_eps: float,
    tie_tol: float,
    *,
    tuning_label: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(y)
    prediction_rows: List[Dict[str, Any]] = []
    fold_rows: List[Dict[str, Any]] = []
    tuning_rows: List[pd.DataFrame] = []

    for repeat in range(outer_repeats):
        split_seed = base_seed + repeat
        splits = stratified_splits(event, arm, outer_splits, split_seed)

        for outer_fold, (train_idx, test_idx) in enumerate(splits):
            inner_seed = base_seed + 1000 + repeat * 100 + outer_fold

            chosen_alpha, audit = tune_ridge(
                X[train_idx],
                y[train_idx],
                event[train_idx],
                arm[train_idx],
                module_to_indices,
                module_order,
                ridge_alphas,
                inner_splits,
                inner_seed,
                tau_quantile,
                min_module_genes,
                sd_eps,
                tie_tol,
            )

            audit = audit.copy()
            audit["analysis"] = tuning_label
            audit["repeat"] = repeat
            audit["outer_fold"] = outer_fold
            tuning_rows.append(audit)

            transformer = HallmarkTransformer(
                module_to_indices,
                module_order,
                min_module_genes,
                sd_eps,
            ).fit(X[train_idx])

            X_train_m = transformer.transform(X[train_idx])
            X_test_m = transformer.transform(X[test_idx])

            model = fit_ridge(X_train_m, y[train_idx], chosen_alpha)

            train_risk = np.asarray(model.predict(X_train_m), dtype=float)
            test_risk = np.asarray(model.predict(X_test_m), dtype=float)

            mu = float(np.mean(train_risk))
            sd = float(np.std(train_risk, ddof=0))
            if not np.isfinite(sd) or sd <= 1e-12:
                sd = 1.0

            test_std = (test_risk - mu) / sd
            fold_c = safe_uno_c(
                y[train_idx],
                y[test_idx],
                test_std,
                tau_quantile,
            )

            fold_rows.append(
                {
                    "analysis": tuning_label,
                    "repeat": repeat,
                    "outer_fold": outer_fold,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "chosen_alpha": chosen_alpha,
                    "fold_uno_c": fold_c,
                    "train_risk_mean": mu,
                    "train_risk_sd": sd,
                }
            )

            for local_idx, global_idx in enumerate(test_idx):
                prediction_rows.append(
                    {
                        "analysis": tuning_label,
                        "repeat": repeat,
                        "outer_fold": outer_fold,
                        "sample_id": sample_ids[global_idx],
                        "sample_index": int(global_idx),
                        "risk_standardized": float(test_std[local_idx]),
                    }
                )

    predictions = pd.DataFrame(prediction_rows)
    folds = pd.DataFrame(fold_rows)
    tuning = pd.concat(tuning_rows, ignore_index=True)

    expected_rows = n * outer_repeats
    if len(predictions) != expected_rows:
        raise RuntimeError(
            f"{tuning_label}: prediction rows={len(predictions)}, expected={expected_rows}."
        )

    per_repeat_counts = predictions.groupby("repeat")["sample_id"].nunique()
    if not (per_repeat_counts == n).all():
        raise RuntimeError(f"{tuning_label}: incomplete OOF coverage in a repeat.")

    return predictions, folds, tuning


def aggregate_cv_predictions(
    predictions: pd.DataFrame,
    sample_ids: np.ndarray,
) -> np.ndarray:
    mean_risk = predictions.groupby("sample_id")["risk_standardized"].mean()
    missing = [x for x in sample_ids if x not in mean_risk.index]
    if missing:
        raise RuntimeError(f"Missing aggregated OOF predictions: {missing[:10]}")
    return np.asarray([mean_risk[x] for x in sample_ids], dtype=float)


def repeat_metrics(
    predictions: pd.DataFrame,
    y_full: np.ndarray,
    sample_ids: np.ndarray,
    tau_quantile: float,
) -> pd.DataFrame:
    rows = []

    for repeat, part in predictions.groupby("repeat"):
        risk_map = part.set_index("sample_id")["risk_standardized"].to_dict()
        risk = np.asarray([risk_map[x] for x in sample_ids], dtype=float)
        c = safe_uno_c(y_full, y_full, risk, tau_quantile)

        rows.append(
            {
                "repeat": int(repeat),
                "uno_c": c,
            }
        )

    return pd.DataFrame(rows)


def patient_bootstrap_uno(
    y_reference: np.ndarray,
    y_test: np.ndarray,
    risk: np.ndarray,
    n_boot: int,
    seed: int,
    tau_quantile: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(y_test)
    values: List[float] = []

    attempts = 0
    max_attempts = n_boot * 5

    while len(values) < n_boot and attempts < max_attempts:
        attempts += 1
        idx = rng.integers(0, n, size=n)
        c = safe_uno_c(
            y_reference,
            y_test[idx],
            risk[idx],
            tau_quantile,
        )
        if np.isfinite(c):
            values.append(float(c))

    if len(values) != n_boot:
        raise RuntimeError(
            f"Only {len(values)}/{n_boot} finite bootstrap Uno-C replicates obtained."
        )

    return np.asarray(values, dtype=float)


def bootstrap_summary(values: np.ndarray) -> Dict[str, float]:
    return {
        "lower_95": float(np.quantile(values, 0.025)),
        "median": float(np.quantile(values, 0.50)),
        "upper_95": float(np.quantile(values, 0.975)),
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)),
    }


def load_stage(
    stage_json: Path,
    expected_protocol_hash: str,
    required_files: Sequence[Path],
) -> Optional[Dict[str, Any]]:
    if not stage_json.exists():
        return None

    payload = read_json(stage_json)

    if clean(payload.get("status")) != "PASS":
        return None
    if clean(payload.get("protocol_sha256")) != expected_protocol_hash:
        return None

    hashes = payload.get("artifact_hashes") or {}

    for path in required_files:
        if not path.exists():
            return None
        key = path.name
        expected = clean(hashes.get(key))
        if len(expected) != 64 or sha256_file(path) != expected:
            return None

    return payload


def save_stage(
    stage_json: Path,
    protocol_hash: str,
    artifacts: Sequence[Path],
    metrics: Dict[str, Any],
) -> None:
    payload = {
        "status": "PASS",
        "created_utc": now_utc(),
        "protocol_sha256": protocol_hash,
        "artifact_hashes": {
            path.name: sha256_file(path)
            for path in artifacts
        },
        "metrics": metrics,
    }
    write_json(stage_json, payload)


def fit_coxnet_path(
    X_train: np.ndarray,
    y_train: np.ndarray,
    l1_ratio: float,
    n_alphas: int,
    alpha_min_ratio: float,
) -> CoxnetSurvivalAnalysis:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = CoxnetSurvivalAnalysis(
            l1_ratio=float(l1_ratio),
            n_alphas=int(n_alphas),
            alpha_min_ratio=float(alpha_min_ratio),
            normalize=False,
            fit_baseline_model=False,
            max_iter=100000,
            tol=1e-7,
        )
        model.fit(X_train, y_train)

    if len(model.alphas_) < n_alphas:
        raise RuntimeError(
            f"Coxnet returned only {len(model.alphas_)} alphas; expected >= {n_alphas}."
        )
    return model


def tune_elastic_net(
    X: np.ndarray,
    y: np.ndarray,
    event: np.ndarray,
    arm: np.ndarray,
    l1_ratios: Sequence[float],
    n_alphas: int,
    alpha_min_ratio: float,
    inner_splits: int,
    seed: int,
    tau_quantile: float,
    sd_eps: float,
    tie_tol: float,
) -> Tuple[float, int, pd.DataFrame]:
    splits = stratified_splits(event, arm, inner_splits, seed)

    rows: List[Dict[str, Any]] = []

    for inner_fold, (train_idx, val_idx) in enumerate(splits):
        transformer = GeneTransformer(sd_eps).fit(X[train_idx])
        X_train_g = transformer.transform(X[train_idx])
        X_val_g = transformer.transform(X[val_idx])

        for l1_ratio in l1_ratios:
            model = fit_coxnet_path(
                X_train_g,
                y[train_idx],
                l1_ratio,
                n_alphas,
                alpha_min_ratio,
            )

            for path_index in range(n_alphas):
                alpha = float(model.alphas_[path_index])
                score = float("nan")
                status = "PASS"

                try:
                    risk = model.predict(X_val_g, alpha=alpha)
                    score = safe_uno_c(
                        y[train_idx],
                        y[val_idx],
                        risk,
                        tau_quantile,
                    )
                    if not np.isfinite(score):
                        status = "NONFINITE_UNO"
                except Exception as exc:
                    status = f"PREDICT_FAIL:{type(exc).__name__}"

                rows.append(
                    {
                        "inner_fold": inner_fold,
                        "l1_ratio": float(l1_ratio),
                        "path_index": int(path_index),
                        "alpha": alpha,
                        "uno_c": score,
                        "status": status,
                    }
                )

    audit = pd.DataFrame(rows)

    grouped = (
        audit.groupby(["l1_ratio", "path_index"], as_index=False)
        .agg(
            mean_uno_c=("uno_c", "mean"),
            n_finite=("uno_c", lambda s: int(np.isfinite(s).sum())),
        )
    )

    grouped = grouped[grouped["n_finite"] == inner_splits].copy()
    if grouped.empty:
        raise RuntimeError("No elastic-net candidate had finite Uno C in all inner folds.")

    best = float(grouped["mean_uno_c"].max())
    candidates = grouped[grouped["mean_uno_c"] >= best - tie_tol].copy()

    # Stronger regularization = lower relative path index.
    min_path = int(candidates["path_index"].min())
    candidates = candidates[candidates["path_index"] == min_path].copy()

    # Then larger l1_ratio, per 03a.
    chosen_l1 = float(candidates["l1_ratio"].max())
    chosen_path = int(min_path)

    audit = audit.merge(
        grouped.rename(columns={"mean_uno_c": "candidate_mean_uno_c"}),
        on=["l1_ratio", "path_index"],
        how="left",
    )
    audit["chosen_l1_ratio"] = chosen_l1
    audit["chosen_path_index"] = chosen_path

    return chosen_l1, chosen_path, audit


def supporting_gene_cv(
    X: np.ndarray,
    y: np.ndarray,
    event: np.ndarray,
    arm: np.ndarray,
    sample_ids: np.ndarray,
    l1_ratios: Sequence[float],
    n_alphas: int,
    alpha_min_ratio: float,
    outer_splits: int,
    outer_repeats: int,
    inner_splits: int,
    base_seed: int,
    tau_quantile: float,
    sd_eps: float,
    tie_tol: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(y)
    prediction_rows: List[Dict[str, Any]] = []
    fold_rows: List[Dict[str, Any]] = []
    tuning_rows: List[pd.DataFrame] = []

    for repeat in range(outer_repeats):
        split_seed = base_seed + repeat
        splits = stratified_splits(event, arm, outer_splits, split_seed)

        for outer_fold, (train_idx, test_idx) in enumerate(splits):
            inner_seed = base_seed + 1000 + repeat * 100 + outer_fold

            l1_ratio, path_index, audit = tune_elastic_net(
                X[train_idx],
                y[train_idx],
                event[train_idx],
                arm[train_idx],
                l1_ratios,
                n_alphas,
                alpha_min_ratio,
                inner_splits,
                inner_seed,
                tau_quantile,
                sd_eps,
                tie_tol,
            )

            audit = audit.copy()
            audit["analysis"] = "OS_SUPPORTING_GENE_ELASTIC_NET"
            audit["repeat"] = repeat
            audit["outer_fold"] = outer_fold
            tuning_rows.append(audit)

            transformer = GeneTransformer(sd_eps).fit(X[train_idx])
            X_train_g = transformer.transform(X[train_idx])
            X_test_g = transformer.transform(X[test_idx])

            model = fit_coxnet_path(
                X_train_g,
                y[train_idx],
                l1_ratio,
                n_alphas,
                alpha_min_ratio,
            )
            alpha = float(model.alphas_[path_index])

            train_risk = np.asarray(
                model.predict(X_train_g, alpha=alpha),
                dtype=float,
            )
            test_risk = np.asarray(
                model.predict(X_test_g, alpha=alpha),
                dtype=float,
            )

            mu = float(np.mean(train_risk))
            sd = float(np.std(train_risk, ddof=0))
            if not np.isfinite(sd) or sd <= 1e-12:
                sd = 1.0

            test_std = (test_risk - mu) / sd
            fold_c = safe_uno_c(
                y[train_idx],
                y[test_idx],
                test_std,
                tau_quantile,
            )

            fold_rows.append(
                {
                    "repeat": repeat,
                    "outer_fold": outer_fold,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "chosen_l1_ratio": l1_ratio,
                    "chosen_path_index": path_index,
                    "chosen_alpha": alpha,
                    "n_training_genes": int(transformer.valid_.sum()),
                    "fold_uno_c": fold_c,
                }
            )

            for local_idx, global_idx in enumerate(test_idx):
                prediction_rows.append(
                    {
                        "repeat": repeat,
                        "outer_fold": outer_fold,
                        "sample_id": sample_ids[global_idx],
                        "sample_index": int(global_idx),
                        "risk_standardized": float(test_std[local_idx]),
                    }
                )

    predictions = pd.DataFrame(prediction_rows)
    folds = pd.DataFrame(fold_rows)
    tuning = pd.concat(tuning_rows, ignore_index=True)

    if len(predictions) != n * outer_repeats:
        raise RuntimeError("Supporting gene OOF coverage incomplete.")

    return predictions, folds, tuning


def arm_direction(
    X: np.ndarray,
    y: np.ndarray,
    event: np.ndarray,
    arm: np.ndarray,
    sample_ids: np.ndarray,
    source_arm: str,
    target_arm: str,
    module_to_indices: Dict[str, np.ndarray],
    module_order: List[str],
    ridge_alphas: Sequence[float],
    inner_splits: int,
    seed: int,
    tau_quantile: float,
    min_module_genes: int,
    sd_eps: float,
    tie_tol: float,
    n_boot: int,
    boot_seed: int,
) -> Tuple[Dict[str, Any], np.ndarray, pd.DataFrame]:
    train_idx = np.flatnonzero(arm == source_arm)
    test_idx = np.flatnonzero(arm == target_arm)

    chosen_alpha, tuning = tune_ridge(
        X[train_idx],
        y[train_idx],
        event[train_idx],
        None,
        module_to_indices,
        module_order,
        ridge_alphas,
        inner_splits,
        seed,
        tau_quantile,
        min_module_genes,
        sd_eps,
        tie_tol,
    )

    transformer = HallmarkTransformer(
        module_to_indices,
        module_order,
        min_module_genes,
        sd_eps,
    ).fit(X[train_idx])

    X_train_m = transformer.transform(X[train_idx])
    X_test_m = transformer.transform(X[test_idx])

    model = fit_ridge(X_train_m, y[train_idx], chosen_alpha)

    train_risk = np.asarray(model.predict(X_train_m), dtype=float)
    test_risk = np.asarray(model.predict(X_test_m), dtype=float)

    mu = float(np.mean(train_risk))
    sd = float(np.std(train_risk, ddof=0))
    if not np.isfinite(sd) or sd <= 1e-12:
        sd = 1.0

    test_std = (test_risk - mu) / sd

    c = safe_uno_c(
        y[train_idx],
        y[test_idx],
        test_std,
        tau_quantile,
    )

    boot = patient_bootstrap_uno(
        y[train_idx],
        y[test_idx],
        test_std,
        n_boot,
        boot_seed,
        tau_quantile,
    )
    bs = bootstrap_summary(boot)

    result = {
        "source_arm": source_arm,
        "target_arm": target_arm,
        "n_source": len(train_idx),
        "n_target": len(test_idx),
        "source_events": int(event[train_idx].sum()),
        "target_events": int(event[test_idx].sum()),
        "chosen_alpha": chosen_alpha,
        "uno_c": c,
        "bootstrap_lower_95": bs["lower_95"],
        "bootstrap_upper_95": bs["upper_95"],
        "bootstrap_median": bs["median"],
    }

    tuning = tuning.copy()
    tuning["source_arm"] = source_arm
    tuning["target_arm"] = target_arm

    return result, boot, tuning


def source_status(
    primary_c: float,
    primary_lower: float,
    supporting_c: Optional[float],
    supporting_lower: Optional[float],
    pass_c: float,
    warn_floor: float,
) -> Tuple[str, bool]:
    primary_pass = (
        primary_c >= pass_c
        and primary_lower > 0.50
    )

    supporting_pass = False
    if supporting_c is not None and supporting_lower is not None:
        supporting_pass = (
            supporting_c >= pass_c
            and supporting_lower > 0.50
        )

    if primary_pass:
        return "SOURCE_PASS", supporting_pass

    if primary_c >= warn_floor or supporting_pass:
        return "SOURCE_WARN", supporting_pass

    return "SOURCE_FAIL", supporting_pass


def dfi_status(
    primary_c: float,
    primary_lower: float,
    pass_c: float,
    warn_floor: float,
) -> str:
    if primary_c >= pass_c and primary_lower > 0.50:
        return "DFI_PASS"
    if primary_c >= warn_floor:
        return "DFI_WARN"
    return "DFI_FAIL"


def arm_status(
    directional: pd.DataFrame,
    random_cv_c: float,
    pass_mean: float,
    warn_mean: float,
    max_drop: float,
) -> Tuple[str, Dict[str, float]]:
    mean_c = float(directional["uno_c"].mean())
    min_point = float(directional["uno_c"].min())
    min_upper = float(directional["bootstrap_upper_95"].min())
    drop = float(random_cv_c - mean_c)

    if (
        mean_c >= pass_mean
        and min_point >= 0.50
        and drop <= max_drop
    ):
        status = "ARM_PASS"
    elif mean_c >= warn_mean and min_upper >= 0.50:
        status = "ARM_WARN"
    else:
        status = "ARM_FAIL"

    return status, {
        "mean_directional_uno_c": mean_c,
        "minimum_directional_point_estimate": min_point,
        "minimum_directional_upper_95": min_upper,
        "drop_from_random_cv": drop,
    }


def endpoint_discordance_action(
    os_source_status: str,
    dfi_source_status: str,
) -> str:
    if os_source_status == "SOURCE_PASS":
        return "OS_PASS_CONTINUE_PRIMARY_OS_TO_OS"

    if os_source_status == "SOURCE_WARN" and dfi_source_status == "DFI_PASS":
        return "OS_WARN_DFI_PASS_OS_REMAINS_PRIMARY_WEAK_SOURCE_FLAG"

    if os_source_status == "SOURCE_FAIL" and dfi_source_status == "DFI_PASS":
        return "OS_FAIL_DFI_PASS_NO_POSTHOC_SWITCH_NEW_AMENDMENT_REQUIRED"

    if os_source_status == "SOURCE_FAIL" and dfi_source_status == "DFI_FAIL":
        return "OS_FAIL_DFI_FAIL_EMPIRICAL_TRANSFER_BRANCH_STOPS"

    return f"{os_source_status}__{dfi_source_status}_NO_ENDPOINT_SWITCH"


def load_inputs(
    protocol: Dict[str, Any],
    upstream: Dict[str, Any],
) -> Dict[str, Any]:
    paper4_root, source = resolve_paper4_root()

    clinical_path, clinical_asset = get_asset(
        upstream,
        "dog2_clinical",
        paper4_root,
    )
    expression_path, expression_asset = get_asset(
        upstream,
        "dog2_expression",
        paper4_root,
    )

    roster = pd.read_csv(P_ROSTER, sep="\t", dtype=str).fillna("")
    hallmark_map = pd.read_csv(P_HALLMARK, sep="\t", dtype=str).fillna("")
    ridge_grid = pd.read_csv(P_RIDGE_GRID, sep="\t")
    elastic_grid = pd.read_csv(P_ELASTIC, sep="\t")

    expected_roster_hash = clean(
        (read_json(P_SUMMARY).get("final_artifact_hashes") or {}).get(
            "source_sample_arm_roster_tsv"
        )
    )
    if sha256_file(P_ROSTER) != expected_roster_hash:
        raise RuntimeError("03a source roster hash mismatch.")

    expected_hallmark_hash = clean(
        (read_json(P_SUMMARY).get("final_artifact_hashes") or {}).get(
            "hallmark_dog2_feature_map_tsv"
        )
    )
    if sha256_file(P_HALLMARK) != expected_hallmark_hash:
        raise RuntimeError("03a Hallmark feature map hash mismatch.")

    if len(roster) != 186:
        raise RuntimeError("Frozen source roster no longer has 186 rows.")

    supporting_features = pd.read_csv(
        ROOT / "results" / "ortholog_bridge" / "02g_v3" / "primary_outcome_blind_ortholog_bridge.tsv",
        sep="\t",
        dtype=str,
        low_memory=False,
    ).fillna("")

    mask = (
        supporting_features["primary_dog2_to_target_os"]
        .astype(str)
        .str.lower()
        .isin({"true", "1", "yes"})
    )
    supporting_features = supporting_features[mask].copy()

    gene_features = supporting_features["dog2_raw_feature"].astype(str).tolist()
    if len(gene_features) != 11815 or len(set(gene_features)) != 11815:
        raise RuntimeError("Supporting gene feature universe is not exact 11,815.")

    hallmark_features = hallmark_map["dog2_raw_feature"].astype(str).unique().tolist()
    all_features = sorted(set(gene_features) | set(hallmark_features))

    header = pd.read_csv(expression_path, nrows=0)
    index_col = str(header.columns[0])
    header_features = set(str(x) for x in header.columns[1:])

    missing_features = sorted(set(all_features) - header_features)
    if missing_features:
        raise RuntimeError(
            f"Locked expression file lacks frozen feature(s): {missing_features[:20]}"
        )

    expr = pd.read_csv(
        expression_path,
        usecols=[index_col] + all_features,
        dtype={index_col: str},
        low_memory=False,
    )
    expr = expr.set_index(index_col)
    expr.index = expr.index.astype(str)

    if expr.index.duplicated().any():
        raise RuntimeError("Expression sample index is not unique.")

    expr = expr.apply(pd.to_numeric, errors="raise")
    if not np.isfinite(expr.to_numpy(dtype=float)).all():
        raise RuntimeError("Non-finite DOG2 expression values.")

    clinical = pd.read_csv(
        clinical_path,
        usecols=["Patient ID", "os_time", "os_event", "dfi_time", "dfi_event"],
        low_memory=False,
    )
    clinical["_canonical_patient_id"] = clinical["Patient ID"].map(canonical_patient_id)

    if clinical["_canonical_patient_id"].eq("").any():
        raise RuntimeError("Empty canonical clinical Patient ID.")
    if clinical["_canonical_patient_id"].duplicated().any():
        raise RuntimeError("Clinical Patient ID is not unique after canonicalization.")

    roster["_canonical_patient_id"] = roster["paper4_patient_id"].map(canonical_patient_id)

    if roster["_canonical_patient_id"].duplicated().any():
        raise RuntimeError("Frozen roster patient ID is not unique.")

    clinical = clinical.set_index("_canonical_patient_id")

    roster_patients = set(roster["_canonical_patient_id"])
    clinical_patients = set(clinical.index)

    if roster_patients != clinical_patients:
        raise RuntimeError(
            f"Clinical/roster Patient-ID set mismatch: "
            f"roster_only={len(roster_patients-clinical_patients)}, "
            f"clinical_only={len(clinical_patients-roster_patients)}"
        )

    roster_samples = set(roster["paper4_sample_id"].astype(str))
    expr_samples = set(expr.index.astype(str))

    if roster_samples != expr_samples:
        raise RuntimeError(
            f"Expression/roster sample-ID set mismatch: "
            f"roster_only={len(roster_samples-expr_samples)}, "
            f"expression_only={len(expr_samples-roster_samples)}"
        )

    # Frozen sample order.
    roster = roster.reset_index(drop=True)
    sample_ids = roster["paper4_sample_id"].astype(str).to_numpy()
    patient_ids = roster["_canonical_patient_id"].astype(str).to_numpy()
    arms = roster["study"].astype(str).to_numpy()

    X_gene = expr.loc[sample_ids, gene_features].to_numpy(dtype=float)

    hallmark_feature_order = sorted(set(hallmark_features))
    X_hallmark_gene = expr.loc[sample_ids, hallmark_feature_order].to_numpy(dtype=float)

    hallmark_feature_index = {
        feature: i
        for i, feature in enumerate(hallmark_feature_order)
    }

    module_order = sorted(hallmark_map["hallmark_module"].unique().tolist())
    module_to_indices: Dict[str, np.ndarray] = {}

    for module in module_order:
        features = (
            hallmark_map.loc[
                hallmark_map["hallmark_module"] == module,
                "dog2_raw_feature",
            ]
            .astype(str)
            .tolist()
        )
        module_to_indices[module] = np.asarray(
            [hallmark_feature_index[x] for x in features],
            dtype=int,
        )

    endpoint_data = {}

    input_audit_rows = []

    for endpoint in ["OS", "DFI"]:
        spec = protocol["endpoints"][endpoint]
        time_col = spec["time_col"]
        event_col = spec["event_col"]

        time_values = pd.to_numeric(
            clinical.loc[patient_ids, time_col],
            errors="raise",
        ).to_numpy(dtype=float)

        event_numeric = pd.to_numeric(
            clinical.loc[patient_ids, event_col],
            errors="raise",
        ).to_numpy(dtype=float)

        if not np.all(np.isfinite(time_values)):
            raise RuntimeError(f"{endpoint}: non-finite survival times.")
        if np.any(time_values <= 0):
            raise RuntimeError(f"{endpoint}: survival times must be strictly positive.")
        if not np.all(np.isfinite(event_numeric)):
            raise RuntimeError(f"{endpoint}: non-finite event values.")
        if not set(np.unique(event_numeric)).issubset({0.0, 1.0}):
            raise RuntimeError(f"{endpoint}: event indicator is not binary.")

        events = event_numeric.astype(bool)
        y = survival_array(time_values, events)

        expected_n = int(spec["expected_complete_n_for_identity_check"])
        expected_events = int(spec["expected_events_for_identity_check"])

        if len(y) != expected_n:
            raise RuntimeError(
                f"{endpoint}: n={len(y)}, expected={expected_n}."
            )
        if int(events.sum()) != expected_events:
            raise RuntimeError(
                f"{endpoint}: events={int(events.sum())}, expected={expected_events}."
            )

        endpoint_data[endpoint] = {
            "time": time_values,
            "event": events,
            "y": y,
        }

        input_audit_rows.append(
            {
                "endpoint": endpoint,
                "n": len(y),
                "events": int(events.sum()),
                "time_min": float(np.min(time_values)),
                "time_median": float(np.median(time_values)),
                "time_max": float(np.max(time_values)),
                "identity_check": "PASS",
            }
        )

    pd.DataFrame(input_audit_rows).to_csv(
        INPUT_AUDIT,
        sep="\t",
        index=False,
    )

    return {
        "paper4_root": paper4_root,
        "paper4_source": source,
        "clinical_path": clinical_path,
        "expression_path": expression_path,
        "clinical_asset": clinical_asset,
        "expression_asset": expression_asset,
        "roster": roster,
        "sample_ids": sample_ids,
        "patient_ids": patient_ids,
        "arms": arms,
        "gene_features": gene_features,
        "X_gene": X_gene,
        "hallmark_feature_order": hallmark_feature_order,
        "X_hallmark_gene": X_hallmark_gene,
        "module_order": module_order,
        "module_to_indices": module_to_indices,
        "ridge_alphas": ridge_grid["alpha"].to_numpy(dtype=float),
        "elastic_l1_ratios": elastic_grid["l1_ratio"].to_numpy(dtype=float),
        "endpoint_data": endpoint_data,
    }


def run_primary_stage(
    endpoint: str,
    X: np.ndarray,
    y: np.ndarray,
    event: np.ndarray,
    arm: np.ndarray,
    sample_ids: np.ndarray,
    module_to_indices: Dict[str, np.ndarray],
    module_order: List[str],
    protocol: Dict[str, Any],
    protocol_hash: str,
    pred_path: Path,
    folds_path: Path,
    tuning_path: Path,
    repeats_path: Path,
    boot_path: Path,
    stage_path: Path,
) -> Dict[str, Any]:
    required = [
        pred_path,
        folds_path,
        tuning_path,
        repeats_path,
        boot_path,
    ]
    existing = load_stage(stage_path, protocol_hash, required)

    if existing is not None:
        print(f"  {endpoint} primary random CV: REUSED validated stage")
        return existing["metrics"]

    print(f"  {endpoint} primary random CV: RUNNING")

    random_cv = protocol["random_cv"]["primary"]
    primary_model = protocol["primary_model"]
    uno = protocol["uno_c"]
    bootstrap = protocol["patient_bootstrap"]

    endpoint_offset = 0 if endpoint == "OS" else 10000

    pred, folds, tuning = primary_cv(
        X,
        y,
        event,
        arm,
        sample_ids,
        module_to_indices,
        module_order,
        primary_model["ridge_alpha_grid"],
        int(random_cv["outer_splits"]),
        int(random_cv["outer_repeats"]),
        int(random_cv["inner_splits"]),
        int(random_cv["base_seed"]) + endpoint_offset,
        float(uno["tau_quantile"]),
        10,
        1e-12,
        float(primary_model["inner_tie_tolerance"]),
        tuning_label=f"{endpoint}_PRIMARY_HALLMARK_RIDGE",
    )

    agg_risk = aggregate_cv_predictions(pred, sample_ids)
    final_c = safe_uno_c(
        y,
        y,
        agg_risk,
        float(uno["tau_quantile"]),
    )

    repeats = repeat_metrics(
        pred,
        y,
        sample_ids,
        float(uno["tau_quantile"]),
    )

    boot_seed = int(bootstrap["seed"]) + endpoint_offset
    boot = patient_bootstrap_uno(
        y,
        y,
        agg_risk,
        int(bootstrap["replicates"]),
        boot_seed,
        float(uno["tau_quantile"]),
    )
    bs = bootstrap_summary(boot)

    pred = pred.copy()
    pred["aggregated_mean_risk_for_sample"] = pred["sample_id"].map(
        dict(zip(sample_ids, agg_risk))
    )

    pred.to_csv(pred_path, sep="\t", index=False)
    folds.to_csv(folds_path, sep="\t", index=False)
    tuning.to_csv(tuning_path, sep="\t", index=False)
    repeats.to_csv(repeats_path, sep="\t", index=False)
    pd.DataFrame(
        {
            "bootstrap_index": np.arange(len(boot)),
            "uno_c": boot,
        }
    ).to_csv(boot_path, sep="\t", index=False)

    metrics = {
        "endpoint": endpoint,
        "uno_c": final_c,
        "bootstrap_lower_95": bs["lower_95"],
        "bootstrap_upper_95": bs["upper_95"],
        "bootstrap_median": bs["median"],
        "repeat_uno_c_mean": float(repeats["uno_c"].mean()),
        "repeat_uno_c_sd": float(repeats["uno_c"].std(ddof=1)),
        "repeat_uno_c_min": float(repeats["uno_c"].min()),
        "repeat_uno_c_max": float(repeats["uno_c"].max()),
    }

    save_stage(
        stage_path,
        protocol_hash,
        required,
        metrics,
    )

    return metrics


def run_support_stage(
    X: np.ndarray,
    y: np.ndarray,
    event: np.ndarray,
    arm: np.ndarray,
    sample_ids: np.ndarray,
    protocol: Dict[str, Any],
    protocol_hash: str,
) -> Dict[str, Any]:
    required = [
        OS_SUPPORT_PRED,
        OS_SUPPORT_FOLDS,
        OS_SUPPORT_TUNING,
        OS_SUPPORT_REPEATS,
        OS_SUPPORT_BOOT,
    ]

    existing = load_stage(OS_SUPPORT_STAGE, protocol_hash, required)
    if existing is not None:
        print("  OS supporting gene elastic-net: REUSED validated stage")
        return existing["metrics"]

    print("  OS supporting gene elastic-net: RUNNING")

    cfg = protocol["supporting_model"]
    cv = protocol["random_cv"]["supporting"]
    uno = protocol["uno_c"]
    bootstrap = protocol["patient_bootstrap"]

    pred, folds, tuning = supporting_gene_cv(
        X,
        y,
        event,
        arm,
        sample_ids,
        cfg["l1_ratio_grid"],
        int(cfg["n_alphas_per_path"]),
        float(cfg["alpha_min_ratio"]),
        int(cv["outer_splits"]),
        int(cv["outer_repeats"]),
        int(cv["inner_splits"]),
        int(cv["base_seed"]),
        float(uno["tau_quantile"]),
        1e-12,
        float(cfg["inner_tie_tolerance"]),
    )

    agg_risk = aggregate_cv_predictions(pred, sample_ids)
    final_c = safe_uno_c(
        y,
        y,
        agg_risk,
        float(uno["tau_quantile"]),
    )

    repeats = repeat_metrics(
        pred,
        y,
        sample_ids,
        float(uno["tau_quantile"]),
    )

    boot = patient_bootstrap_uno(
        y,
        y,
        agg_risk,
        int(bootstrap["replicates"]),
        int(bootstrap["seed"]) + 50000,
        float(uno["tau_quantile"]),
    )
    bs = bootstrap_summary(boot)

    pred = pred.copy()
    pred["aggregated_mean_risk_for_sample"] = pred["sample_id"].map(
        dict(zip(sample_ids, agg_risk))
    )

    pred.to_csv(OS_SUPPORT_PRED, sep="\t", index=False)
    folds.to_csv(OS_SUPPORT_FOLDS, sep="\t", index=False)
    tuning.to_csv(OS_SUPPORT_TUNING, sep="\t", index=False)
    repeats.to_csv(OS_SUPPORT_REPEATS, sep="\t", index=False)
    pd.DataFrame(
        {
            "bootstrap_index": np.arange(len(boot)),
            "uno_c": boot,
        }
    ).to_csv(OS_SUPPORT_BOOT, sep="\t", index=False)

    metrics = {
        "endpoint": "OS",
        "uno_c": final_c,
        "bootstrap_lower_95": bs["lower_95"],
        "bootstrap_upper_95": bs["upper_95"],
        "bootstrap_median": bs["median"],
        "repeat_uno_c_mean": float(repeats["uno_c"].mean()),
        "repeat_uno_c_sd": float(repeats["uno_c"].std(ddof=1)),
        "repeat_uno_c_min": float(repeats["uno_c"].min()),
        "repeat_uno_c_max": float(repeats["uno_c"].max()),
    }

    save_stage(
        OS_SUPPORT_STAGE,
        protocol_hash,
        required,
        metrics,
    )
    return metrics


def run_arm_stage(
    endpoint: str,
    X: np.ndarray,
    y: np.ndarray,
    event: np.ndarray,
    arm: np.ndarray,
    sample_ids: np.ndarray,
    module_to_indices: Dict[str, np.ndarray],
    module_order: List[str],
    protocol: Dict[str, Any],
    protocol_hash: str,
    metrics_path: Path,
    boot_path: Path,
    tuning_path: Path,
    stage_path: Path,
) -> Dict[str, Any]:
    required = [
        metrics_path,
        boot_path,
        tuning_path,
    ]

    existing = load_stage(stage_path, protocol_hash, required)
    if existing is not None:
        print(f"  {endpoint} arm->arm: REUSED validated stage")
        return existing["metrics"]

    print(f"  {endpoint} arm->arm: RUNNING")

    cfg = protocol["arm_to_arm"]
    primary = protocol["primary_model"]
    uno = protocol["uno_c"]

    directions = [
        ("COTC021", "COTC022"),
        ("COTC022", "COTC021"),
    ]

    endpoint_offset = 0 if endpoint == "OS" else 10000

    metric_rows = []
    boot_rows = []
    tuning_parts = []

    for direction_index, (source_arm, target_arm) in enumerate(directions):
        result, boot, tuning = arm_direction(
            X,
            y,
            event,
            arm,
            sample_ids,
            source_arm,
            target_arm,
            module_to_indices,
            module_order,
            primary["ridge_alpha_grid"],
            int(cfg["inner_splits"]),
            int(cfg["base_seed"]) + endpoint_offset + direction_index,
            float(uno["tau_quantile"]),
            10,
            1e-12,
            float(primary["inner_tie_tolerance"]),
            int(cfg["test_bootstrap_replicates"]),
            int(cfg["test_bootstrap_seed"]) + endpoint_offset + direction_index,
        )

        metric_rows.append(result)

        for i, value in enumerate(boot):
            boot_rows.append(
                {
                    "source_arm": source_arm,
                    "target_arm": target_arm,
                    "bootstrap_index": i,
                    "uno_c": float(value),
                }
            )

        tuning = tuning.copy()
        tuning["endpoint"] = endpoint
        tuning_parts.append(tuning)

        print(
            f"    {source_arm}->{target_arm}: Uno C={result['uno_c']:.4f}, "
            f"95% CI [{result['bootstrap_lower_95']:.4f}, "
            f"{result['bootstrap_upper_95']:.4f}]"
        )

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(metrics_path, sep="\t", index=False)
    pd.DataFrame(boot_rows).to_csv(boot_path, sep="\t", index=False)
    pd.concat(tuning_parts, ignore_index=True).to_csv(
        tuning_path,
        sep="\t",
        index=False,
    )

    metrics = {
        "endpoint": endpoint,
        "directional": metric_rows,
        "mean_directional_uno_c": float(metrics_df["uno_c"].mean()),
    }

    save_stage(
        stage_path,
        protocol_hash,
        required,
        metrics,
    )
    return metrics


def run_permutations(
    X: np.ndarray,
    y: np.ndarray,
    time_values: np.ndarray,
    event: np.ndarray,
    arm: np.ndarray,
    sample_ids: np.ndarray,
    module_to_indices: Dict[str, np.ndarray],
    module_order: List[str],
    protocol: Dict[str, Any],
    protocol_hash: str,
) -> Dict[str, Any]:
    cfg = protocol["permutation_supporting_inference"]
    primary = protocol["primary_model"]
    uno = protocol["uno_c"]

    n_perm = int(cfg["permutations"])
    seed = int(cfg["seed"])

    observed_seed = seed + 999999

    checkpoint = None
    if PERM_CHECKPOINT.exists():
        checkpoint = read_json(PERM_CHECKPOINT)
        if clean(checkpoint.get("protocol_sha256")) != protocol_hash:
            checkpoint = None

    if checkpoint is None:
        print("  OS permutation inference: computing observed same-design statistic")

        observed_pred, _, _ = primary_cv(
            X,
            y,
            event,
            arm,
            sample_ids,
            module_to_indices,
            module_order,
            primary["ridge_alpha_grid"],
            int(cfg["cv_splits"]),
            int(cfg["cv_repeats"]),
            int(protocol["random_cv"]["primary"]["inner_splits"]),
            observed_seed,
            float(uno["tau_quantile"]),
            10,
            1e-12,
            float(primary["inner_tie_tolerance"]),
            tuning_label="OS_PERMUTATION_OBSERVED",
        )

        observed_risk = aggregate_cv_predictions(
            observed_pred,
            sample_ids,
        )
        observed_c = safe_uno_c(
            y,
            y,
            observed_risk,
            float(uno["tau_quantile"]),
        )

        checkpoint = {
            "status": "IN_PROGRESS",
            "protocol_sha256": protocol_hash,
            "observed_uno_c": observed_c,
            "completed": 0,
            "null_uno_c": [],
            "created_utc": now_utc(),
        }
        write_json(PERM_CHECKPOINT, checkpoint)

        print(f"    observed same-design Uno C={observed_c:.4f}")
    else:
        print(
            f"  OS permutation inference: resuming {checkpoint['completed']}/{n_perm}"
        )

    observed_c = float(checkpoint["observed_uno_c"])
    null_values = [float(x) for x in checkpoint.get("null_uno_c", [])]
    start = len(null_values)

    for perm_index in range(start, n_perm):
        rng = np.random.default_rng(seed + perm_index)

        perm_time = np.array(time_values, copy=True)
        perm_event = np.array(event, copy=True)

        for study in ["COTC021", "COTC022"]:
            idx = np.flatnonzero(arm == study)
            shuffled = rng.permutation(idx)

            # Shuffle paired (time,event) outcomes within randomized study arm.
            perm_time[idx] = time_values[shuffled]
            perm_event[idx] = event[shuffled]

        y_perm = survival_array(perm_time, perm_event)

        pred, _, _ = primary_cv(
            X,
            y_perm,
            perm_event,
            arm,
            sample_ids,
            module_to_indices,
            module_order,
            primary["ridge_alpha_grid"],
            int(cfg["cv_splits"]),
            int(cfg["cv_repeats"]),
            int(protocol["random_cv"]["primary"]["inner_splits"]),
            observed_seed,
            float(uno["tau_quantile"]),
            10,
            1e-12,
            float(primary["inner_tie_tolerance"]),
            tuning_label=f"OS_PERMUTATION_{perm_index+1}",
        )

        risk = aggregate_cv_predictions(pred, sample_ids)
        c = safe_uno_c(
            y_perm,
            y_perm,
            risk,
            float(uno["tau_quantile"]),
        )

        null_values.append(float(c))

        checkpoint = {
            "status": "IN_PROGRESS",
            "protocol_sha256": protocol_hash,
            "observed_uno_c": observed_c,
            "completed": len(null_values),
            "null_uno_c": null_values,
            "updated_utc": now_utc(),
        }
        write_json(PERM_CHECKPOINT, checkpoint)

        if (perm_index + 1) % 10 == 0 or perm_index + 1 == n_perm:
            print(
                f"    permutations: {perm_index+1}/{n_perm}, "
                f"current null mean={np.mean(null_values):.4f}"
            )

    null = np.asarray(null_values, dtype=float)
    p_value = float(
        (1 + np.sum(null >= observed_c))
        / (1 + len(null))
    )

    pd.DataFrame(
        {
            "permutation_index": np.arange(1, len(null) + 1),
            "null_uno_c": null,
        }
    ).to_csv(
        PERM_RESULTS,
        sep="\t",
        index=False,
    )

    final_checkpoint = {
        "status": "PASS",
        "protocol_sha256": protocol_hash,
        "observed_uno_c": observed_c,
        "completed": len(null),
        "null_uno_c": null.tolist(),
        "null_mean": float(np.mean(null)),
        "null_sd": float(np.std(null, ddof=1)),
        "p_value_one_sided": p_value,
        "permutation_results_sha256": sha256_file(PERM_RESULTS),
        "updated_utc": now_utc(),
    }
    write_json(PERM_CHECKPOINT, final_checkpoint)

    return {
        "observed_same_design_uno_c": observed_c,
        "null_mean": float(np.mean(null)),
        "null_sd": float(np.std(null, ddof=1)),
        "p_value_one_sided": p_value,
        "n_permutations": len(null),
    }


def main() -> None:
    started = now_utc()

    print("=" * 120)
    print("Paper 6 - run frozen DOG2 Source Prognostic Gate")
    print("=" * 120)
    print(f"Script version: {SCRIPT_VERSION}")
    print()
    print("Safety / execution contract:")
    print("  DOG2 clinical/outcome values read in 03b: YES")
    print("  DOG2 expression values read in 03b: YES")
    print("  Human clinical/outcome values read: NO")
    print("  Treatment-administration values read: NO")
    print("  Post-baseline sample annotations read: NO")
    print("  Samples dropped based on outcome/model results: NO")
    print("  Scientific thresholds changed after outcome access: NO")
    print("  GPU execution: NO")
    print()

    for path in [
        I_CONTRACT,
        P_PROTOCOL,
        P_SUMMARY,
        P_ENDPOINTS,
        P_ROSTER,
        P_HALLMARK,
        P_RIDGE_GRID,
        P_ELASTIC,
        UPSTREAM_LOCK,
    ]:
        require_file(path)

    i_contract = read_json(I_CONTRACT)
    protocol = read_json(P_PROTOCOL)
    protocol_summary = read_json(P_SUMMARY)
    upstream = read_json(UPSTREAM_LOCK)

    if clean(i_contract.get("status")) != "PASS":
        raise RuntimeError("02i contract is not PASS.")
    if clean(protocol.get("status")) != "PASS":
        raise RuntimeError("03a protocol is not PASS.")
    if clean(protocol_summary.get("status")) != "PASS":
        raise RuntimeError("03a summary is not PASS.")
    if clean(protocol_summary.get("scientific_status")) != (
        "PASS_SOURCE_PROGNOSTIC_GATE_PROTOCOL_FROZEN"
    ):
        raise RuntimeError("03a is not in frozen PASS state.")

    expected_protocol_hash = clean(
        (protocol_summary.get("final_artifact_hashes") or {}).get(
            "source_prognostic_gate_protocol_json"
        )
    )
    protocol_hash = sha256_file(P_PROTOCOL)

    if protocol_hash != expected_protocol_hash:
        raise RuntimeError("03a protocol hash verification failed.")

    expected_02i_hash = clean((protocol.get("upstream") or {}).get("02i_contract_sha256"))
    observed_02i_hash = sha256_file(I_CONTRACT)
    if len(expected_02i_hash) != 64 or observed_02i_hash != expected_02i_hash:
        raise RuntimeError(
            "02i contract hash does not match the exact 03a-frozen upstream contract."
        )

    # Scientific decision rules come from immutable 02i; 03a contains the
    # computational implementation only.
    frozen_source_gate = i_contract.get("source_gate") or {}
    frozen_arm_gate = i_contract.get("arm_to_arm_generalization") or {}
    frozen_combined_matrix = i_contract.get("combined_source_gate_matrix") or {}

    if not frozen_source_gate or not frozen_arm_gate or not frozen_combined_matrix:
        raise RuntimeError("02i frozen gate definitions are incomplete.")

    print("03a protocol verification: PASS")
    print("02i scientific-threshold hash verification: PASS")
    print(f"Protocol SHA256: {protocol_hash}")

    data = load_inputs(protocol, upstream)

    print()
    print("Locked DOG2 input identity:")
    print(f"  physical Paper4 root: {data['paper4_root']}")
    print(f"  expression SHA256: {data['expression_asset']['sha256']}")
    print(f"  clinical SHA256: {data['clinical_asset']['sha256']}")
    print("  sample alignment: 186/186 PASS")
    print("  patient alignment: 186/186 PASS")
    print("  COTC021: 93")
    print("  COTC022: 93")

    os_data = data["endpoint_data"]["OS"]
    dfi_data = data["endpoint_data"]["DFI"]

    print()
    print("Outcome identity checks [FIRST Paper-6 outcome access]:")
    print(f"  OS: n=186, events={int(os_data['event'].sum())} [expected 124]")
    print(f"  DFI: n=186, events={int(dfi_data['event'].sum())} [expected 143]")

    # ------------------------------------------------------------------
    # Primary OS.
    # ------------------------------------------------------------------
    print()
    print("-" * 120)
    print("OS primary source signal")
    print("-" * 120)

    os_primary = run_primary_stage(
        "OS",
        data["X_hallmark_gene"],
        os_data["y"],
        os_data["event"],
        data["arms"],
        data["sample_ids"],
        data["module_to_indices"],
        data["module_order"],
        protocol,
        protocol_hash,
        pred_path=OS_PRIMARY_PRED,
        folds_path=OS_PRIMARY_FOLDS,
        tuning_path=OS_PRIMARY_TUNING,
        repeats_path=OS_PRIMARY_REPEATS,
        boot_path=OS_PRIMARY_BOOT,
        stage_path=OS_PRIMARY_STAGE,
    )

    print(
        f"  OS primary OOS Uno C={os_primary['uno_c']:.4f}, "
        f"95% bootstrap CI [{os_primary['bootstrap_lower_95']:.4f}, "
        f"{os_primary['bootstrap_upper_95']:.4f}]"
    )

    # ------------------------------------------------------------------
    # Supporting OS gene elastic-net.
    # ------------------------------------------------------------------
    print()
    print("-" * 120)
    print("OS supporting gene-level signal")
    print("-" * 120)

    os_support = run_support_stage(
        data["X_gene"],
        os_data["y"],
        os_data["event"],
        data["arms"],
        data["sample_ids"],
        protocol,
        protocol_hash,
    )

    print(
        f"  OS supporting OOS Uno C={os_support['uno_c']:.4f}, "
        f"95% bootstrap CI [{os_support['bootstrap_lower_95']:.4f}, "
        f"{os_support['bootstrap_upper_95']:.4f}]"
    )

    # ------------------------------------------------------------------
    # OS arm-to-arm.
    # ------------------------------------------------------------------
    print()
    print("-" * 120)
    print("OS randomized treatment-context generalization")
    print("-" * 120)

    os_arm = run_arm_stage(
        "OS",
        data["X_hallmark_gene"],
        os_data["y"],
        os_data["event"],
        data["arms"],
        data["sample_ids"],
        data["module_to_indices"],
        data["module_order"],
        protocol,
        protocol_hash,
        OS_ARM_METRICS,
        OS_ARM_BOOT,
        OS_ARM_TUNING,
        OS_ARM_STAGE,
    )

    os_arm_df = pd.read_csv(OS_ARM_METRICS, sep="\t")
    os_arm_status, os_arm_diag = arm_status(
        os_arm_df,
        float(os_primary["uno_c"]),
        float(frozen_arm_gate["arm_pass"]["mean_directional_uno_c_min"]),
        float(frozen_arm_gate["arm_warn"]["mean_directional_uno_c_min"]),
        float(frozen_arm_gate["arm_pass"]["max_drop_from_random_cv_mean_uno_c"]),
    )

    print(f"  OS arm status: {os_arm_status}")
    print(f"  mean directional Uno C={os_arm_diag['mean_directional_uno_c']:.4f}")
    print(f"  drop vs random CV={os_arm_diag['drop_from_random_cv']:.4f}")

    # ------------------------------------------------------------------
    # DFI primary + arm-to-arm.
    # ------------------------------------------------------------------
    print()
    print("-" * 120)
    print("DFI secondary source signal")
    print("-" * 120)

    dfi_primary = run_primary_stage(
        "DFI",
        data["X_hallmark_gene"],
        dfi_data["y"],
        dfi_data["event"],
        data["arms"],
        data["sample_ids"],
        data["module_to_indices"],
        data["module_order"],
        protocol,
        protocol_hash,
        pred_path=DFI_PRIMARY_PRED,
        folds_path=DFI_PRIMARY_FOLDS,
        tuning_path=DFI_PRIMARY_TUNING,
        repeats_path=DFI_PRIMARY_REPEATS,
        boot_path=DFI_PRIMARY_BOOT,
        stage_path=DFI_PRIMARY_STAGE,
    )

    print(
        f"  DFI primary OOS Uno C={dfi_primary['uno_c']:.4f}, "
        f"95% bootstrap CI [{dfi_primary['bootstrap_lower_95']:.4f}, "
        f"{dfi_primary['bootstrap_upper_95']:.4f}]"
    )

    dfi_arm = run_arm_stage(
        "DFI",
        data["X_hallmark_gene"],
        dfi_data["y"],
        dfi_data["event"],
        data["arms"],
        data["sample_ids"],
        data["module_to_indices"],
        data["module_order"],
        protocol,
        protocol_hash,
        DFI_ARM_METRICS,
        DFI_ARM_BOOT,
        DFI_ARM_TUNING,
        DFI_ARM_STAGE,
    )

    dfi_arm_df = pd.read_csv(DFI_ARM_METRICS, sep="\t")
    dfi_arm_status_value, dfi_arm_diag = arm_status(
        dfi_arm_df,
        float(dfi_primary["uno_c"]),
        float(frozen_arm_gate["arm_pass"]["mean_directional_uno_c_min"]),
        float(frozen_arm_gate["arm_warn"]["mean_directional_uno_c_min"]),
        float(frozen_arm_gate["arm_pass"]["max_drop_from_random_cv_mean_uno_c"]),
    )

    # ------------------------------------------------------------------
    # Frozen gate interpretation.
    # ------------------------------------------------------------------
    os_source_status, supporting_pass = source_status(
        float(os_primary["uno_c"]),
        float(os_primary["bootstrap_lower_95"]),
        float(os_support["uno_c"]),
        float(os_support["bootstrap_lower_95"]),
        pass_c=float(frozen_source_gate["source_pass"]["uno_c_min"]),
        warn_floor=float(frozen_source_gate["source_warn_material_floor"]),
    )

    dfi_source_status = dfi_status(
        float(dfi_primary["uno_c"]),
        float(dfi_primary["bootstrap_lower_95"]),
        pass_c=float(frozen_source_gate["source_pass"]["uno_c_min"]),
        warn_floor=float(frozen_source_gate["source_warn_material_floor"]),
    )

    combined_key = f"{os_source_status}__{os_arm_status}"

    protocol_matrix = protocol.get("combined_gate_matrix") or {}
    if protocol_matrix != frozen_combined_matrix:
        raise RuntimeError(
            "03a combined gate matrix does not exactly match the frozen 02i matrix."
        )

    combined_status = clean(frozen_combined_matrix.get(combined_key))

    if not combined_status:
        raise RuntimeError(
            f"Frozen 02i combined gate matrix lacks key {combined_key!r}."
        )

    discordance_action = endpoint_discordance_action(
        os_source_status,
        dfi_source_status,
    )

    # ------------------------------------------------------------------
    # Supporting permutation inference.
    # ------------------------------------------------------------------
    print()
    print("-" * 120)
    print("OS primary supporting permutation inference")
    print("-" * 120)

    perm = run_permutations(
        data["X_hallmark_gene"],
        os_data["y"],
        os_data["time"],
        os_data["event"],
        data["arms"],
        data["sample_ids"],
        data["module_to_indices"],
        data["module_order"],
        protocol,
        protocol_hash,
    )

    print(
        f"  observed same-design Uno C={perm['observed_same_design_uno_c']:.4f}"
    )
    print(
        f"  null mean={perm['null_mean']:.4f}, "
        f"one-sided p={perm['p_value_one_sided']:.6f}"
    )

    # ------------------------------------------------------------------
    # Final frozen gate record.
    # ------------------------------------------------------------------
    result = {
        "script_version": SCRIPT_VERSION,
        "status": "PASS",
        "created_utc": now_utc(),
        "protocol_sha256": protocol_hash,
        "02i_contract_sha256": observed_02i_hash,
        "implementation_note": (
            "v2 fixes threshold routing only: numerical scientific thresholds are read "
            "from the exact hash-locked 02i contract; no scientific protocol changed."
        ),
        "input_identity": {
            "n": 186,
            "COTC021": 93,
            "COTC022": 93,
            "OS_events": int(os_data["event"].sum()),
            "DFI_events": int(dfi_data["event"].sum()),
            "expression_sha256": clean(data["expression_asset"]["sha256"]),
            "clinical_sha256": clean(data["clinical_asset"]["sha256"]),
        },
        "OS": {
            "primary_Hallmark_random_cv": os_primary,
            "supporting_gene_elastic_net_random_cv": os_support,
            "source_status": os_source_status,
            "supporting_gene_independently_passes": supporting_pass,
            "arm_to_arm": {
                "status": os_arm_status,
                "diagnostics": os_arm_diag,
                "directional": os_arm_df.to_dict(orient="records"),
            },
            "permutation_supporting_inference": perm,
        },
        "DFI": {
            "primary_Hallmark_random_cv": dfi_primary,
            "source_status": dfi_source_status,
            "arm_to_arm": {
                "status": dfi_arm_status_value,
                "diagnostics": dfi_arm_diag,
                "directional": dfi_arm_df.to_dict(orient="records"),
            },
        },
        "combined_primary_source_gate": combined_status,
        "endpoint_discordance_action": discordance_action,
        "thresholds_source": "02i/03a frozen before outcome access",
        "human_outcome_values_read": False,
        "human_clinical_values_read": False,
        "samples_dropped_based_on_model_or_outcome_results": False,
    }
    write_json(GATE_RESULTS, result)

    artifacts = [
        INPUT_AUDIT,
        OS_PRIMARY_PRED,
        OS_PRIMARY_FOLDS,
        OS_PRIMARY_TUNING,
        OS_PRIMARY_REPEATS,
        OS_PRIMARY_BOOT,
        OS_SUPPORT_PRED,
        OS_SUPPORT_FOLDS,
        OS_SUPPORT_TUNING,
        OS_SUPPORT_REPEATS,
        OS_SUPPORT_BOOT,
        OS_ARM_METRICS,
        OS_ARM_BOOT,
        OS_ARM_TUNING,
        DFI_PRIMARY_PRED,
        DFI_PRIMARY_FOLDS,
        DFI_PRIMARY_TUNING,
        DFI_PRIMARY_REPEATS,
        DFI_PRIMARY_BOOT,
        DFI_ARM_METRICS,
        DFI_ARM_BOOT,
        DFI_ARM_TUNING,
        PERM_RESULTS,
        PERM_CHECKPOINT,
        GATE_RESULTS,
    ]

    final_hashes = {
        path.name: sha256_file(path)
        for path in artifacts
        if path.exists()
    }

    summary = {
        "script_version": SCRIPT_VERSION,
        "status": "PASS",
        "scientific_status": combined_status,
        "implementation_note": (
            "v2 threshold-routing bugfix only; completed v1 stages reused only when "
            "their 03a protocol SHA256 and artifact hashes validate."
        ),
        "run_started_utc": started,
        "run_finished_utc": now_utc(),
        "OS_primary_uno_c": float(os_primary["uno_c"]),
        "OS_primary_bootstrap_lower_95": float(os_primary["bootstrap_lower_95"]),
        "OS_primary_bootstrap_upper_95": float(os_primary["bootstrap_upper_95"]),
        "OS_supporting_uno_c": float(os_support["uno_c"]),
        "OS_supporting_bootstrap_lower_95": float(os_support["bootstrap_lower_95"]),
        "OS_source_status": os_source_status,
        "OS_arm_status": os_arm_status,
        "OS_arm_mean_uno_c": float(os_arm_diag["mean_directional_uno_c"]),
        "DFI_primary_uno_c": float(dfi_primary["uno_c"]),
        "DFI_primary_bootstrap_lower_95": float(dfi_primary["bootstrap_lower_95"]),
        "DFI_primary_bootstrap_upper_95": float(dfi_primary["bootstrap_upper_95"]),
        "DFI_source_status": dfi_source_status,
        "DFI_arm_status": dfi_arm_status_value,
        "DFI_arm_mean_uno_c": float(dfi_arm_diag["mean_directional_uno_c"]),
        "endpoint_discordance_action": discordance_action,
        "OS_permutation_p_one_sided": float(perm["p_value_one_sided"]),
        "OS_permutation_null_mean": float(perm["null_mean"]),
        "final_artifact_hashes": final_hashes,
        "DOG2_outcome_values_read": True,
        "DOG2_expression_values_read": True,
        "human_outcome_values_read": False,
        "human_clinical_values_read": False,
        "samples_dropped_based_on_model_or_outcome_results": False,
    }
    write_json(SUMMARY_JSON, summary)

    print()
    print("=" * 120)
    print("03b DOG2 SOURCE PROGNOSTIC GATE SUMMARY")
    print("=" * 120)
    print(
        f"OS primary Hallmark: Uno C={os_primary['uno_c']:.4f}, "
        f"95% CI [{os_primary['bootstrap_lower_95']:.4f}, "
        f"{os_primary['bootstrap_upper_95']:.4f}] -> {os_source_status}"
    )
    print(
        f"OS supporting gene elastic-net: Uno C={os_support['uno_c']:.4f}, "
        f"95% CI [{os_support['bootstrap_lower_95']:.4f}, "
        f"{os_support['bootstrap_upper_95']:.4f}]"
    )
    print(
        f"OS arm->arm mean Uno C={os_arm_diag['mean_directional_uno_c']:.4f}, "
        f"drop vs random CV={os_arm_diag['drop_from_random_cv']:.4f} "
        f"-> {os_arm_status}"
    )
    print(
        f"OS permutation: p={perm['p_value_one_sided']:.6f}, "
        f"null mean={perm['null_mean']:.4f} [SUPPORTING ONLY]"
    )
    print()
    print(
        f"DFI primary Hallmark: Uno C={dfi_primary['uno_c']:.4f}, "
        f"95% CI [{dfi_primary['bootstrap_lower_95']:.4f}, "
        f"{dfi_primary['bootstrap_upper_95']:.4f}] -> {dfi_source_status}"
    )
    print(
        f"DFI arm->arm mean Uno C={dfi_arm_diag['mean_directional_uno_c']:.4f} "
        f"-> {dfi_arm_status_value}"
    )
    print()
    print(f"Combined primary Source Gate: {combined_status}")
    print(f"Endpoint discordance action: {discordance_action}")
    print()
    print("Human outcome/clinical values read: NO")
    print("Samples dropped based on model/outcome results: NO")
    print()
    print("Next scientific stage:")
    if combined_status.startswith("SOURCE_GREEN"):
        print(
            "  Proceed to the frozen classical survival-transfer benchmark stage "
            "before serious AI architecture development."
        )
    elif combined_status.startswith("SOURCE_AMBER"):
        print(
            "  Proceed cautiously to the classical benchmark/premise stage with the "
            "frozen weak/context-sensitive source flag; do not upgrade claims."
        )
    else:
        print(
            "  Do not proceed as if the empirical source premise passed. "
            "Apply the frozen methods/negative-transfer fallback logic."
        )
    print()
    print("Artifacts: results\\source_gate\\03b")
    print("=" * 120)
    print("03b frozen DOG2 Source Prognostic Gate execution: PASS")
    print("=" * 120)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 120, file=sys.stderr)
        print("03b frozen DOG2 Source Prognostic Gate execution: FAIL", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print("=" * 120, file=sys.stderr)
        raise
