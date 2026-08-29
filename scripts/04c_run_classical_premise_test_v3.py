#!/usr/bin/env python3
"""
Paper 6 - run frozen GSE16091 classical canine-added-value premise test.

This is the FIRST Paper-6 stage that intentionally opens GSE16091 expression
and OS outcome values.

Frozen upstream scientific contract
-----------------------------------
Target:
    GSE16091, n=34, GPL96, overall survival.

Human pool:
    the GSE16091 outer-training fold in repeated outer CV.

Primary paired premise comparison:
    B4 DOG2 + HUMAN residual ridge Cox
        versus
    B0 HUMAN-only Hallmark ridge Cox

Frozen decision:
SUPPORTS_CANINE_ADDED_VALUE if all:
    delta Uno C (B4-B0) >= +0.02
    paired bootstrap P(delta C > 0) >= 0.90
    delta IBS (B4-B0) <= +0.01

ARGUES_AGAINST_CANINE_ADDED_VALUE if all:
    delta Uno C (B4-B0) <= -0.02
    paired bootstrap P(delta C < 0) >= 0.90

otherwise:
    INCONCLUSIVE_NEUTRAL

Secondary classical benchmarks:
    B1 HUMAN-only gene elastic-net Cox
    B2 DOG zero-shot Hallmark ridge score
    B3 DOG CORAL zero-shot Hallmark score
    event-matched DOG source information-efficiency analysis

Reserved human cohorts remain outcome-closed:
    TARGET-OS
    GSE21257
    GSE39055

Restart behavior
----------------
The primary B0/B2/B3/B4 analysis is checkpointed per repeat.
The B1 gene-level analysis is checkpointed per repeat.
The event-matched DOG secondary analysis is checkpointed per repeat.
The central premise decision is materialized immediately after the primary
stage and paired bootstrap, before secondary long-running analyses.

No CLI arguments.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests

from scipy.optimize import minimize
from scipy.special import logsumexp

try:
    from sklearn.model_selection import StratifiedKFold
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    from sksurv.metrics import concordance_index_ipcw
    from sksurv.util import Surv
except ImportError as exc:
    raise ImportError(
        "04c requires scikit-survival in the active Paper-6 .venv."
    ) from exc


SCRIPT_VERSION = "04c-run-classical-premise-test-v3-secondary-output-path-fix-no-cli"

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Frozen 04b protocol.
# ---------------------------------------------------------------------------

B4_DIR = ROOT / "results" / "human_premise_protocol" / "04b"
B4_PROTOCOL = B4_DIR / "classical_premise_test_protocol.json"
B4_SUMMARY = B4_DIR / "summary.json"
B4_PROBE_MAP = B4_DIR / "gse16091_gpl96_probe_gene_map.tsv"
B4_GENE_UNIVERSE = B4_DIR / "gse16091_outcome_blind_gene_universe.tsv"
B4_HALLMARK_MAP = B4_DIR / "gse16091_hallmark_feature_map.tsv"
B4_DOG_ALPHA = B4_DIR / "dog2_full_source_alpha_contract.tsv"

# Explicit post-outcome technical amendment. This amendment cannot make the
# premise conclusion more favorable; it only handles an observed frozen-IBS
# support failure conservatively.
C40_DIR = ROOT / "results" / "human_premise_protocol" / "04c0_ibs_amendment"
C40_AMENDMENT = C40_DIR / "ibs_feasibility_amendment.json"
C40_SUMMARY = C40_DIR / "summary.json"

# 04a target roster / contract.
A4_DIR = ROOT / "results" / "human_premise_target" / "04a"
A4_CONTRACT = A4_DIR / "sacrificial_human_premise_target_contract.json"
A4_ROSTER = A4_DIR / "sacrificial_target_sample_roster.tsv"

# DOG2 source data/protocol.
A3_DIR = ROOT / "results" / "source_gate_protocol" / "03a"
A3_PROTOCOL = A3_DIR / "source_prognostic_gate_protocol.json"
A3_SUMMARY = A3_DIR / "summary.json"

B3_SCRIPT = ROOT / "scripts" / "03b_run_dog2_source_prognostic_gate_v2.py"
UPSTREAM_LOCK = ROOT / "contracts" / "00_upstream_input_lock.json"

# Upstream source-gate status.
B3_DIR = ROOT / "results" / "source_gate" / "03b"
B3_SUMMARY = B3_DIR / "summary.json"

# 03c premise action policy.
C3_DIR = ROOT / "results" / "source_gate_diagnostics" / "03c"
C3_POLICY = C3_DIR / "premise_test_action_policy_addendum.json"
C3_SUMMARY = C3_DIR / "summary.json"

# ---------------------------------------------------------------------------
# Outputs.
# ---------------------------------------------------------------------------

OUT_DIR = ROOT / "results" / "human_premise" / "04c"
RAW_DIR = OUT_DIR / "raw"
CHECKPOINT_DIR = OUT_DIR / "checkpoints"
PRIMARY_CKPT = CHECKPOINT_DIR / "primary"
B1_CKPT = CHECKPOINT_DIR / "B1"
EVENT_MATCHED_CKPT = CHECKPOINT_DIR / "event_matched"

for d in [OUT_DIR, RAW_DIR, CHECKPOINT_DIR, PRIMARY_CKPT, B1_CKPT, EVENT_MATCHED_CKPT]:
    d.mkdir(parents=True, exist_ok=True)

SERIES_MATRIX_GZ = RAW_DIR / "GSE16091_series_matrix.txt.gz"
SERIES_MATRIX_LOCK = RAW_DIR / "GSE16091_series_matrix_lock.json"

INPUT_AUDIT = OUT_DIR / "gse16091_input_identity_audit.tsv"
OUTCOME_AUDIT = OUT_DIR / "gse16091_os_identity_audit.tsv"
DOG_SOURCE_COEFS = OUT_DIR / "dog2_full_source_hallmark_coefficients.tsv"

PRIMARY_PRED = OUT_DIR / "primary_classical_oof_predictions.tsv"
PRIMARY_FOLDS = OUT_DIR / "primary_classical_fold_metrics.tsv"
PRIMARY_TUNING = OUT_DIR / "primary_classical_tuning.tsv"
PRIMARY_IBS = OUT_DIR / "primary_classical_fold_ibs.tsv"
PRIMARY_BOOTSTRAP = OUT_DIR / "primary_paired_bootstrap.tsv"
PRIMARY_RESULT = OUT_DIR / "primary_premise_result.json"

B1_PRED = OUT_DIR / "B1_gene_elastic_net_oof_predictions.tsv"
B1_FOLDS = OUT_DIR / "B1_gene_elastic_net_fold_metrics.tsv"
B1_TUNING = OUT_DIR / "B1_gene_elastic_net_tuning.tsv"
B1_RESULT = OUT_DIR / "B1_gene_elastic_net_result.json"

EVENT_MATCHED_PRED = OUT_DIR / "event_matched_dog_predictions.tsv"
EVENT_MATCHED_DRAWS_TSV = OUT_DIR / "event_matched_dog_draw_metrics.tsv"
EVENT_MATCHED_RESULT = OUT_DIR / "event_matched_dog_result.json"

ALL_MODEL_SUMMARY = OUT_DIR / "classical_model_summary.tsv"
FINAL_RESULT = OUT_DIR / "classical_premise_test_results.json"
SUMMARY_JSON = OUT_DIR / "summary.json"

# ---------------------------------------------------------------------------
# Frozen constants; verified against 04b JSON before outcome access.
# ---------------------------------------------------------------------------

EXPECTED_TARGET = "GSE16091"
EXPECTED_TARGET_N = 34
EXPECTED_SOURCE_GATE = "SOURCE_AMBER_WEAK_AND_CONTEXT_SENSITIVE"

OUTER_SPLITS = 5
OUTER_REPEATS = 20
INNER_SPLITS = 4
PATIENT_BOOTSTRAPS = 5000

RIDGE_ALPHAS = np.logspace(-4, 4, 17)
ELASTIC_L1_RATIOS = [0.10, 0.50, 0.90]
ELASTIC_N_ALPHAS = 30
ELASTIC_ALPHA_MIN_RATIO = 0.01

INNER_TIE_TOLERANCE = 0.005
UNO_TAU_QUANTILE = 0.90

MIN_HALLMARK_GENES = 10
SD_EPS = 1e-12
CORAL_EPS = 1e-5

PREMISE_DELTA_C_SUPPORT = 0.02
PREMISE_PROB_SUPPORT = 0.90
PREMISE_MAX_DELTA_IBS = 0.01
PREMISE_DELTA_C_AGAINST = -0.02
PREMISE_PROB_AGAINST = 0.90

IBS_GRID_POINTS = 20
IBS_LOW_Q = 0.20
IBS_HIGH_Q = 0.80
IBS_MIN_POINTS = 5

EVENT_MATCHED_DRAWS_PER_OUTER_FOLD = 50

# Download path is already frozen inside 04b. Do not replace with another source.
DEFAULT_SERIES_MATRIX_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE16nnn/GSE16091/"
    "matrix/GSE16091_series_matrix.txt.gz"
)


# ===========================================================================
# Utility / provenance.
# ===========================================================================

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


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_surv(time_values: Sequence[float], event_values: Sequence[bool]) -> np.ndarray:
    return Surv.from_arrays(
        event=np.asarray(event_values, dtype=bool),
        time=np.asarray(time_values, dtype=float),
    )


def validate_upstream_hash(
    current_path: Path,
    expected_hash: str,
    label: str,
) -> None:
    observed = sha256_file(current_path)
    if clean(expected_hash) != observed:
        raise RuntimeError(
            f"{label} SHA256 mismatch: expected={expected_hash}, observed={observed}"
        )


# ===========================================================================
# GSE16091 retrieval / parsing.
# ===========================================================================

def download_locked_series_matrix(
    url: str,
    destination: Path,
    lock_path: Path,
) -> None:
    if destination.exists() and lock_path.exists():
        lock = read_json(lock_path)
        expected = clean(lock.get("sha256"))
        if expected != sha256_file(destination):
            raise RuntimeError("Existing GSE16091 series-matrix hash mismatch.")
        return

    if destination.exists() != lock_path.exists():
        raise RuntimeError(
            "Partial GSE16091 raw lock state exists; do not silently overwrite."
        )

    session = requests.Session()
    tmp = destination.with_suffix(destination.suffix + ".part")

    last_exc: Optional[Exception] = None
    for attempt in range(1, 4):
        try:
            response = session.get(
                url,
                timeout=120,
                headers={"User-Agent": "Paper6-04c-frozen-premise/1.0"},
            )
            response.raise_for_status()
            payload = response.content
            if len(payload) < 1000:
                raise RuntimeError("Downloaded GSE16091 series matrix is implausibly small.")

            # Verify gzip can be opened before freezing.
            with gzip.GzipFile(fileobj=io.BytesIO(payload), mode="rb") as gz:
                prefix = gz.read(4096).decode("utf-8", errors="replace")
            if "!Series_geo_accession" not in prefix:
                raise RuntimeError("Downloaded file does not look like GEO series matrix.")

            tmp.write_bytes(payload)
            tmp.replace(destination)

            lock = {
                "status": "PASS_FIRST_HUMAN_PREMISE_OUTCOME_SNAPSHOT",
                "created_utc": now_utc(),
                "accession": EXPECTED_TARGET,
                "url": url,
                "path": str(destination.relative_to(ROOT)),
                "sha256": sha256_file(destination),
                "size_bytes": destination.stat().st_size,
                "human_outcomes_present_in_snapshot": True,
                "frozen_on_first_04c_access": True,
            }
            write_json(lock_path, lock)
            return
        except Exception as exc:
            last_exc = exc
            tmp.unlink(missing_ok=True)
            if attempt < 3:
                time.sleep(2 * attempt)

    raise RuntimeError(f"Could not download frozen GSE16091 series matrix: {last_exc}")


def _parse_geo_metadata_values(line: str) -> List[str]:
    # GEO series-matrix metadata are tab-delimited and quoted.
    reader = csv.reader([line], delimiter="\t", quotechar='"')
    fields = next(reader)
    if len(fields) < 2:
        return []
    return [field.strip() for field in fields[1:]]


def parse_series_matrix(
    path: Path,
) -> Tuple[List[str], Dict[str, Dict[str, str]], pd.DataFrame]:
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()

    sample_accessions: Optional[List[str]] = None
    characteristic_rows: List[List[str]] = []
    table_begin = None
    table_end = None

    for i, raw in enumerate(lines):
        line = raw.rstrip("\r\n")

        if line.startswith("!Sample_geo_accession"):
            values = _parse_geo_metadata_values(line)
            sample_accessions = [clean(x) for x in values]

        if line.startswith("!Sample_characteristics_ch1"):
            characteristic_rows.append(_parse_geo_metadata_values(line))

        if line.strip() == "!series_matrix_table_begin":
            table_begin = i
        elif line.strip() == "!series_matrix_table_end":
            table_end = i
            break

    if sample_accessions is None:
        raise RuntimeError("GSE16091 series matrix lacks !Sample_geo_accession.")
    if table_begin is None or table_end is None or table_end <= table_begin + 1:
        raise RuntimeError("Could not find GSE16091 expression matrix boundaries.")

    n = len(sample_accessions)
    if n != EXPECTED_TARGET_N:
        raise RuntimeError(f"GSE16091 sample count={n}, expected={EXPECTED_TARGET_N}.")

    characteristics: Dict[str, Dict[str, str]] = {
        acc: {} for acc in sample_accessions
    }

    for row in characteristic_rows:
        if len(row) != n:
            raise RuntimeError(
                "GSE16091 sample-characteristics row length does not match sample count."
            )

        for acc, value in zip(sample_accessions, row):
            text = clean(value)
            if ":" not in text:
                continue
            key, val = text.split(":", 1)
            key = key.strip().lower()
            val = val.strip()

            if key in characteristics[acc] and characteristics[acc][key] != val:
                raise RuntimeError(
                    f"{acc}: conflicting characteristic values for {key!r}."
                )
            characteristics[acc][key] = val

    table_text = "".join(lines[table_begin + 1 : table_end])
    expression = pd.read_csv(
        io.StringIO(table_text),
        sep="\t",
        dtype=str,
        low_memory=False,
    )

    if expression.empty:
        raise RuntimeError("GSE16091 expression table parsed as empty.")

    expression.columns = [str(x).strip().strip('"') for x in expression.columns]
    first_col = expression.columns[0]

    if first_col not in {"ID_REF", "ID"}:
        raise RuntimeError(f"Unexpected first GSE16091 expression column: {first_col!r}")

    expression[first_col] = (
        expression[first_col].astype(str).str.strip().str.strip('"')
    )

    expected_sample_cols = sample_accessions
    missing = sorted(set(expected_sample_cols) - set(expression.columns))
    extra = sorted(
        set(expression.columns[1:]) - set(expected_sample_cols)
    )
    if missing or extra:
        raise RuntimeError(
            f"GSE16091 expression/sample metadata mismatch: "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )

    expression = expression.set_index(first_col)
    if expression.index.duplicated().any():
        raise RuntimeError("GSE16091 expression probe IDs are duplicated.")

    expression = expression[expected_sample_cols].apply(
        pd.to_numeric,
        errors="raise",
    )

    if not np.isfinite(expression.to_numpy(dtype=float)).all():
        raise RuntimeError("Non-finite values in GSE16091 expression matrix.")

    return sample_accessions, characteristics, expression


def build_target_outcome(
    sample_accessions: List[str],
    characteristics: Dict[str, Dict[str, str]],
    time_field: str,
    status_field: str,
    status_mapping: Dict[str, int],
) -> pd.DataFrame:
    rows = []

    time_key = time_field.lower()
    status_key = status_field.lower()

    for acc in sample_accessions:
        char = characteristics[acc]

        if time_key not in char:
            raise RuntimeError(f"{acc}: missing frozen time field {time_field!r}.")
        if status_key not in char:
            raise RuntimeError(f"{acc}: missing frozen status field {status_field!r}.")

        time_value = float(char[time_key])
        status_raw = clean(char[status_key])

        if status_raw not in status_mapping:
            raise RuntimeError(
                f"{acc}: unexpected status value {status_raw!r}; "
                "04a froze FAIL_CLOSED_NO_RECODING."
            )

        event = int(status_mapping[status_raw])

        if not np.isfinite(time_value) or time_value <= 0:
            raise RuntimeError(f"{acc}: invalid survival time {time_value}.")
        if event not in {0, 1}:
            raise RuntimeError(f"{acc}: non-binary event mapping.")

        rows.append(
            {
                "geo_accession": acc,
                "os_time": time_value,
                "os_event": event,
                "status_raw": status_raw,
            }
        )

    return pd.DataFrame(rows)


# ===========================================================================
# Human expression representation.
# ===========================================================================

def collapse_probes_to_genes(
    expression_probe_by_sample: pd.DataFrame,
    probe_map: pd.DataFrame,
    gene_universe: pd.DataFrame,
    sample_order: Sequence[str],
) -> Tuple[np.ndarray, List[str]]:
    desired_genes = gene_universe["human_gene_symbol"].astype(str).tolist()
    desired_set = set(desired_genes)

    usable = probe_map[
        probe_map["symbol_resolution_status"].astype(str) == "UNIQUE_SYMBOL"
    ].copy()
    usable = usable[
        usable["human_gene_symbol"].astype(str).isin(desired_set)
    ].copy()

    probe_to_gene = {
        str(row.probe_id): str(row.human_gene_symbol)
        for row in usable.itertuples(index=False)
    }

    probes = [
        probe for probe in probe_to_gene
        if probe in expression_probe_by_sample.index
    ]

    if not probes:
        raise RuntimeError("No frozen GPL96 probes found in GSE16091 matrix.")

    block = expression_probe_by_sample.loc[probes, list(sample_order)].copy()
    block["__gene__"] = [probe_to_gene[p] for p in probes]

    gene_by_sample = block.groupby("__gene__", sort=False).median(numeric_only=True)

    missing_genes = sorted(desired_set - set(gene_by_sample.index))
    if missing_genes:
        raise RuntimeError(
            f"GSE16091 matrix lacks frozen gene(s) after probe collapse: "
            f"{missing_genes[:20]}"
        )

    gene_by_sample = gene_by_sample.loc[desired_genes, list(sample_order)]

    X = gene_by_sample.T.to_numpy(dtype=float)

    if not np.isfinite(X).all():
        raise RuntimeError("Non-finite gene-collapsed GSE16091 expression.")

    return X, desired_genes


class HumanHallmarkTransformer:
    def __init__(
        self,
        module_to_gene_indices: Dict[str, np.ndarray],
        module_order: List[str],
        min_genes: int = MIN_HALLMARK_GENES,
        sd_eps: float = SD_EPS,
    ):
        self.module_to_gene_indices = module_to_gene_indices
        self.module_order = module_order
        self.min_genes = min_genes
        self.sd_eps = sd_eps

        self.gene_mean_: Optional[np.ndarray] = None
        self.gene_sd_: Optional[np.ndarray] = None
        self.module_indices_: Dict[str, np.ndarray] = {}
        self.module_mean_: Optional[np.ndarray] = None
        self.module_sd_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "HumanHallmarkTransformer":
        X = np.asarray(X, dtype=float)

        self.gene_mean_ = np.mean(X, axis=0)
        self.gene_sd_ = np.std(X, axis=0, ddof=0)

        valid = np.isfinite(self.gene_sd_) & (self.gene_sd_ > self.sd_eps)

        Z = np.zeros_like(X, dtype=float)
        Z[:, valid] = (
            X[:, valid] - self.gene_mean_[valid]
        ) / self.gene_sd_[valid]

        raw_modules = []
        self.module_indices_ = {}

        for module in self.module_order:
            indices = np.asarray(self.module_to_gene_indices[module], dtype=int)
            indices = indices[valid[indices]]

            if len(indices) < self.min_genes:
                raise RuntimeError(
                    f"{module}: only {len(indices)} nonzero-variance human genes "
                    f"in training fold; minimum={self.min_genes}."
                )

            self.module_indices_[module] = indices
            raw_modules.append(np.mean(Z[:, indices], axis=1))

        M = np.column_stack(raw_modules)

        self.module_mean_ = np.mean(M, axis=0)
        self.module_sd_ = np.std(M, axis=0, ddof=0)

        bad = ~np.isfinite(self.module_sd_)
        if bad.any():
            raise RuntimeError("Non-finite human Hallmark module SD.")

        self.module_sd_[self.module_sd_ <= self.sd_eps] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if (
            self.gene_mean_ is None
            or self.gene_sd_ is None
            or self.module_mean_ is None
            or self.module_sd_ is None
        ):
            raise RuntimeError("HumanHallmarkTransformer not fitted.")

        X = np.asarray(X, dtype=float)
        valid = self.gene_sd_ > self.sd_eps

        Z = np.zeros_like(X, dtype=float)
        Z[:, valid] = (
            X[:, valid] - self.gene_mean_[valid]
        ) / self.gene_sd_[valid]

        raw_modules = []
        for module in self.module_order:
            idx = self.module_indices_[module]
            raw_modules.append(np.mean(Z[:, idx], axis=1))

        M = np.column_stack(raw_modules)
        return (M - self.module_mean_) / self.module_sd_


class GeneTransformer:
    def __init__(self, sd_eps: float = SD_EPS):
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
            raise RuntimeError("Too few nonzero-variance genes.")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.sd_ is None or self.valid_ is None:
            raise RuntimeError("GeneTransformer not fitted.")

        X = np.asarray(X, dtype=float)
        return (
            X[:, self.valid_] - self.mean_[self.valid_]
        ) / self.sd_[self.valid_]


# ===========================================================================
# Cox / metrics.
# ===========================================================================

def tau_from_training(y_train: np.ndarray, quantile: float) -> float:
    times = np.asarray(y_train["time"], dtype=float)
    tau = float(np.quantile(times, quantile))
    return min(tau, np.nextafter(float(np.max(times)), 0.0))


def safe_uno_c(
    y_train: np.ndarray,
    y_test: np.ndarray,
    risk: np.ndarray,
    quantile: float = UNO_TAU_QUANTILE,
) -> float:
    risk = np.asarray(risk, dtype=float)
    if not np.isfinite(risk).all():
        return float("nan")

    tau = tau_from_training(y_train, quantile)

    try:
        return float(
            concordance_index_ipcw(
                y_train,
                y_test,
                risk,
                tau=tau,
            )[0]
        )
    except ValueError:
        mask = np.asarray(y_test["time"], dtype=float) <= tau
        if int(mask.sum()) < 3:
            return float("nan")
        try:
            return float(
                concordance_index_ipcw(
                    y_train,
                    y_test[mask],
                    risk[mask],
                    tau=tau,
                )[0]
            )
        except Exception:
            return float("nan")
    except Exception:
        return float("nan")


def stratified_splits(
    event: np.ndarray,
    n_splits: int,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    labels = np.asarray(event, dtype=int)
    counts = pd.Series(labels).value_counts()

    if len(counts) != 2 or int(counts.min()) < n_splits:
        raise RuntimeError(
            f"Cannot make frozen {n_splits}-fold event-stratified split; "
            f"class counts={counts.to_dict()}."
        )

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    dummy = np.zeros(len(labels))
    return list(splitter.split(dummy, labels))


@dataclass
class CenteredRidgeCox:
    alpha: float
    center: np.ndarray
    beta_: Optional[np.ndarray] = None
    event_times_: Optional[np.ndarray] = None
    base_cum_hazard_: Optional[np.ndarray] = None
    optimization_success_: bool = False
    optimization_message_: str = ""
    gradient_norm_: float = float("nan")

    def _objective_gradient(
        self,
        beta: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        times = np.asarray(y["time"], dtype=float)
        events = np.asarray(y["event"], dtype=bool)

        eta = X @ beta
        unique_event_times = np.unique(times[events])

        nll = 0.0
        grad = np.zeros(X.shape[1], dtype=float)

        for t in unique_event_times:
            event_mask = events & (times == t)
            risk_mask = times >= t
            d = int(event_mask.sum())

            eta_risk = eta[risk_mask]
            X_risk = X[risk_mask]

            lse = logsumexp(eta_risk)
            weights = np.exp(eta_risk - lse)

            nll -= float(np.sum(eta[event_mask]) - d * lse)
            grad -= (
                np.sum(X[event_mask], axis=0)
                - d * np.sum(X_risk * weights[:, None], axis=0)
            )

        delta = beta - self.center
        nll += 0.5 * float(self.alpha) * float(delta @ delta)
        grad += float(self.alpha) * delta

        return float(nll), grad

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CenteredRidgeCox":
        X = np.asarray(X, dtype=float)
        center = np.asarray(self.center, dtype=float)

        if center.shape != (X.shape[1],):
            raise RuntimeError("Centered Cox prior-center dimension mismatch.")

        def fun(beta):
            value, gradient = self._objective_gradient(beta, X, y)
            return value, gradient

        result = minimize(
            fun,
            x0=center.copy(),
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": 2000,
                "ftol": 1e-12,
                "gtol": 1e-8,
                "maxls": 50,
            },
        )

        if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
            raise RuntimeError("Centered ridge Cox optimization produced non-finite result.")

        _, final_grad = self._objective_gradient(result.x, X, y)
        grad_norm = float(np.linalg.norm(final_grad))

        if not result.success and grad_norm > 1e-4:
            raise RuntimeError(
                f"Centered ridge Cox failed to converge: {result.message}; "
                f"gradient norm={grad_norm:.6g}"
            )

        self.beta_ = np.asarray(result.x, dtype=float)
        self.optimization_success_ = bool(result.success)
        self.optimization_message_ = str(result.message)
        self.gradient_norm_ = grad_norm

        self._fit_breslow_baseline(X, y)
        return self

    def _fit_breslow_baseline(self, X: np.ndarray, y: np.ndarray) -> None:
        if self.beta_ is None:
            raise RuntimeError("Model beta missing.")

        times = np.asarray(y["time"], dtype=float)
        events = np.asarray(y["event"], dtype=bool)
        eta = np.clip(X @ self.beta_, -50, 50)
        exp_eta = np.exp(eta)

        event_times = np.unique(times[events])
        increments = []

        for t in event_times:
            d = int(np.sum(events & (times == t)))
            risk_sum = float(np.sum(exp_eta[times >= t]))
            if risk_sum <= 0 or not np.isfinite(risk_sum):
                raise RuntimeError("Invalid Breslow risk-set sum.")
            increments.append(d / risk_sum)

        self.event_times_ = np.asarray(event_times, dtype=float)
        self.base_cum_hazard_ = np.cumsum(np.asarray(increments, dtype=float))

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.beta_ is None:
            raise RuntimeError("Centered ridge Cox not fitted.")
        return np.asarray(X, dtype=float) @ self.beta_

    def predict_survival_matrix(
        self,
        X: np.ndarray,
        times: np.ndarray,
    ) -> np.ndarray:
        if self.beta_ is None or self.event_times_ is None or self.base_cum_hazard_ is None:
            raise RuntimeError("Centered ridge Cox baseline not fitted.")

        times = np.asarray(times, dtype=float)
        idx = np.searchsorted(self.event_times_, times, side="right") - 1

        H0 = np.zeros(len(times), dtype=float)
        valid = idx >= 0
        H0[valid] = self.base_cum_hazard_[idx[valid]]

        eta = np.clip(self.predict(X), -50, 50)
        multiplier = np.exp(eta)

        return np.exp(-multiplier[:, None] * H0[None, :])


def tune_centered_ridge(
    X_gene: np.ndarray,
    y: np.ndarray,
    event: np.ndarray,
    module_to_indices: Dict[str, np.ndarray],
    module_order: List[str],
    center_beta: np.ndarray,
    alphas: Sequence[float],
    inner_splits: int,
    seed: int,
) -> Tuple[float, pd.DataFrame]:
    splits = stratified_splits(event, inner_splits, seed)
    rows = []

    for inner_fold, (train_idx, val_idx) in enumerate(splits):
        transformer = HumanHallmarkTransformer(
            module_to_indices,
            module_order,
        ).fit(X_gene[train_idx])

        X_train = transformer.transform(X_gene[train_idx])
        X_val = transformer.transform(X_gene[val_idx])

        for alpha in alphas:
            score = float("nan")
            status = "PASS"
            grad_norm = float("nan")

            try:
                model = CenteredRidgeCox(
                    alpha=float(alpha),
                    center=np.asarray(center_beta, dtype=float),
                ).fit(X_train, y[train_idx])

                score = safe_uno_c(
                    y[train_idx],
                    y[val_idx],
                    model.predict(X_val),
                )
                grad_norm = model.gradient_norm_
                if not np.isfinite(score):
                    status = "NONFINITE_UNO"
            except Exception as exc:
                status = f"FIT_FAIL:{type(exc).__name__}"

            rows.append(
                {
                    "inner_fold": inner_fold,
                    "alpha": float(alpha),
                    "uno_c": score,
                    "gradient_norm": grad_norm,
                    "status": status,
                }
            )

    audit = pd.DataFrame(rows)

    grouped = (
        audit.groupby("alpha", as_index=False)
        .agg(
            mean_uno_c=("uno_c", "mean"),
            n_finite=("uno_c", lambda x: int(np.isfinite(x).sum())),
        )
    )
    grouped = grouped[grouped["n_finite"] == inner_splits].copy()

    if grouped.empty:
        raise RuntimeError("No centered-ridge alpha has finite Uno C in all inner folds.")

    best = float(grouped["mean_uno_c"].max())
    candidates = grouped[
        grouped["mean_uno_c"] >= best - INNER_TIE_TOLERANCE
    ]
    chosen = float(candidates["alpha"].max())

    audit = audit.merge(
        grouped.rename(columns={"mean_uno_c": "alpha_mean_uno_c"}),
        on="alpha",
        how="left",
    )
    audit["chosen_alpha"] = chosen

    return chosen, audit


def fit_coxnet_path(
    X_train: np.ndarray,
    y_train: np.ndarray,
    l1_ratio: float,
) -> CoxnetSurvivalAnalysis:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = CoxnetSurvivalAnalysis(
            l1_ratio=float(l1_ratio),
            n_alphas=ELASTIC_N_ALPHAS,
            alpha_min_ratio=ELASTIC_ALPHA_MIN_RATIO,
            normalize=False,
            fit_baseline_model=False,
            max_iter=100000,
            tol=1e-7,
        )
        model.fit(X_train, y_train)

    if len(model.alphas_) < ELASTIC_N_ALPHAS:
        raise RuntimeError(
            f"Coxnet returned {len(model.alphas_)} alphas, "
            f"expected {ELASTIC_N_ALPHAS}."
        )

    return model


def tune_gene_elastic_net(
    X_gene: np.ndarray,
    y: np.ndarray,
    event: np.ndarray,
    inner_splits: int,
    seed: int,
) -> Tuple[float, int, pd.DataFrame]:
    splits = stratified_splits(event, inner_splits, seed)
    rows = []

    for inner_fold, (train_idx, val_idx) in enumerate(splits):
        transformer = GeneTransformer().fit(X_gene[train_idx])
        X_train = transformer.transform(X_gene[train_idx])
        X_val = transformer.transform(X_gene[val_idx])

        for l1 in ELASTIC_L1_RATIOS:
            try:
                model = fit_coxnet_path(X_train, y[train_idx], l1)
            except Exception as exc:
                for path_index in range(ELASTIC_N_ALPHAS):
                    rows.append(
                        {
                            "inner_fold": inner_fold,
                            "l1_ratio": l1,
                            "path_index": path_index,
                            "alpha": float("nan"),
                            "uno_c": float("nan"),
                            "status": f"FIT_FAIL:{type(exc).__name__}",
                        }
                    )
                continue

            for path_index in range(ELASTIC_N_ALPHAS):
                alpha = float(model.alphas_[path_index])
                score = float("nan")
                status = "PASS"

                try:
                    risk = model.predict(X_val, alpha=alpha)
                    score = safe_uno_c(
                        y[train_idx],
                        y[val_idx],
                        risk,
                    )
                    if not np.isfinite(score):
                        status = "NONFINITE_UNO"
                except Exception as exc:
                    status = f"PREDICT_FAIL:{type(exc).__name__}"

                rows.append(
                    {
                        "inner_fold": inner_fold,
                        "l1_ratio": float(l1),
                        "path_index": path_index,
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
            n_finite=("uno_c", lambda x: int(np.isfinite(x).sum())),
        )
    )
    grouped = grouped[grouped["n_finite"] == inner_splits].copy()

    if grouped.empty:
        raise RuntimeError("No B1 elastic-net candidate is finite in all inner folds.")

    best = float(grouped["mean_uno_c"].max())
    candidates = grouped[
        grouped["mean_uno_c"] >= best - INNER_TIE_TOLERANCE
    ]

    min_path = int(candidates["path_index"].min())
    candidates = candidates[candidates["path_index"] == min_path]
    chosen_l1 = float(candidates["l1_ratio"].max())

    audit = audit.merge(
        grouped.rename(columns={"mean_uno_c": "candidate_mean_uno_c"}),
        on=["l1_ratio", "path_index"],
        how="left",
    )
    audit["chosen_l1_ratio"] = chosen_l1
    audit["chosen_path_index"] = min_path

    return chosen_l1, min_path, audit


# ===========================================================================
# Brier / IBS.
# ===========================================================================

def censoring_km_training(
    y_train: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    times = np.asarray(y_train["time"], dtype=float)
    original_event = np.asarray(y_train["event"], dtype=bool)
    censor_event = ~original_event

    unique_times = np.sort(np.unique(times))
    surv = 1.0
    out_times = []
    out_surv = []

    for t in unique_times:
        at_risk = int(np.sum(times >= t))
        d_censor = int(np.sum((times == t) & censor_event))

        if at_risk > 0 and d_censor > 0:
            surv *= 1.0 - d_censor / at_risk

        out_times.append(float(t))
        out_surv.append(float(surv))

    return np.asarray(out_times), np.asarray(out_surv)


def km_step_value(
    km_times: np.ndarray,
    km_surv: np.ndarray,
    t: float,
    *,
    left_limit: bool,
) -> float:
    side = "left" if left_limit else "right"
    idx = np.searchsorted(km_times, float(t), side=side) - 1
    if idx < 0:
        return 1.0
    return float(km_surv[idx])


def ipcw_brier_score(
    y_train: np.ndarray,
    y_test: np.ndarray,
    survival_prob: np.ndarray,
    eval_times: np.ndarray,
) -> np.ndarray:
    km_times, km_surv = censoring_km_training(y_train)

    test_time = np.asarray(y_test["time"], dtype=float)
    test_event = np.asarray(y_test["event"], dtype=bool)

    survival_prob = np.asarray(survival_prob, dtype=float)
    eval_times = np.asarray(eval_times, dtype=float)

    if survival_prob.shape != (len(y_test), len(eval_times)):
        raise RuntimeError("Brier survival-probability shape mismatch.")

    scores = []

    for j, t in enumerate(eval_times):
        G_t = km_step_value(km_times, km_surv, t, left_limit=False)
        if G_t <= 1e-8:
            raise RuntimeError("Training censoring survival too small at IBS time.")

        weighted_error = np.zeros(len(y_test), dtype=float)

        for i in range(len(y_test)):
            Ti = float(test_time[i])
            di = bool(test_event[i])
            S = float(survival_prob[i, j])

            if Ti <= t and di:
                G_Tminus = km_step_value(
                    km_times,
                    km_surv,
                    Ti,
                    left_limit=True,
                )
                if G_Tminus <= 1e-8:
                    raise RuntimeError(
                        "Training censoring survival too small at test event time."
                    )
                weighted_error[i] = (S ** 2) / G_Tminus

            elif Ti > t:
                weighted_error[i] = ((1.0 - S) ** 2) / G_t

            else:
                # Censored at/before t: zero IPCW contribution.
                weighted_error[i] = 0.0

        scores.append(float(np.mean(weighted_error)))

    return np.asarray(scores, dtype=float)


def choose_ibs_grid(
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> np.ndarray:
    train_time = np.asarray(y_train["time"], dtype=float)
    train_event = np.asarray(y_train["event"], dtype=bool)
    test_time = np.asarray(y_test["time"], dtype=float)

    event_times = train_time[train_event]
    if len(event_times) < 2:
        raise RuntimeError("Too few training events for frozen IBS grid.")

    lo = float(np.quantile(event_times, IBS_LOW_Q))
    hi = float(np.quantile(event_times, IBS_HIGH_Q))

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise RuntimeError("Invalid frozen IBS event-time quantiles.")

    grid = np.linspace(lo, hi, IBS_GRID_POINTS)

    lower_support = max(
        float(np.min(train_time)),
        float(np.min(test_time)),
    )
    upper_support = min(
        float(np.max(train_time)),
        float(np.max(test_time)),
    )

    grid = grid[
        (grid > lower_support)
        & (grid < np.nextafter(upper_support, 0.0))
    ]

    if len(grid) < IBS_MIN_POINTS:
        raise RuntimeError(
            f"Only {len(grid)} supported IBS points; "
            f"minimum={IBS_MIN_POINTS}."
        )

    return grid


def integrated_brier_custom(
    y_train: np.ndarray,
    y_test: np.ndarray,
    survival_prob: np.ndarray,
    eval_times: np.ndarray,
) -> float:
    bs = ipcw_brier_score(
        y_train,
        y_test,
        survival_prob,
        eval_times,
    )
    span = float(eval_times[-1] - eval_times[0])
    if span <= 0:
        raise RuntimeError("Invalid IBS integration span.")
    return float(np.trapz(bs, eval_times) / span)


# ===========================================================================
# CORAL.
# ===========================================================================

def symmetric_matrix_power(
    matrix: np.ndarray,
    power: float,
    eps: float,
) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    matrix = (matrix + matrix.T) / 2.0
    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, eps)
    powered = np.diag(values ** power)
    return vectors @ powered @ vectors.T


def coral_beta_to_target(
    X_source: np.ndarray,
    X_target_train: np.ndarray,
    beta_source: np.ndarray,
    eps: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    Cs = np.cov(X_source, rowvar=False, ddof=1)
    Ct = np.cov(X_target_train, rowvar=False, ddof=1)

    I = np.eye(Cs.shape[0])
    Cs_reg = Cs + eps * I
    Ct_reg = Ct + eps * I

    A = (
        symmetric_matrix_power(Cs_reg, -0.5, eps)
        @ symmetric_matrix_power(Ct_reg, 0.5, eps)
    )

    beta_target = np.linalg.pinv(A, rcond=1e-10) @ beta_source

    return beta_target, {
        "source_cov_condition": float(np.linalg.cond(Cs_reg)),
        "target_cov_condition": float(np.linalg.cond(Ct_reg)),
        "alignment_condition": float(np.linalg.cond(A)),
    }


# ===========================================================================
# Checkpointing.
# ===========================================================================

def checkpoint_manifest_path(directory: Path, repeat: int) -> Path:
    return directory / f"repeat_{repeat:02d}.json"


def checkpoint_data_path(
    directory: Path,
    repeat: int,
    kind: str,
) -> Path:
    return directory / f"repeat_{repeat:02d}_{kind}.tsv"


def load_repeat_checkpoint(
    directory: Path,
    repeat: int,
    protocol_hash: str,
    kinds: Sequence[str],
) -> Optional[Dict[str, pd.DataFrame]]:
    manifest_path = checkpoint_manifest_path(directory, repeat)
    if not manifest_path.exists():
        return None

    manifest = read_json(manifest_path)

    if clean(manifest.get("status")) != "PASS":
        return None
    if clean(manifest.get("protocol_sha256")) != protocol_hash:
        return None

    frames = {}

    for kind in kinds:
        path = checkpoint_data_path(directory, repeat, kind)
        expected = clean((manifest.get("artifact_hashes") or {}).get(path.name))
        if not path.exists() or len(expected) != 64:
            return None
        if sha256_file(path) != expected:
            return None
        frames[kind] = pd.read_csv(path, sep="\t")

    return frames


def save_repeat_checkpoint(
    directory: Path,
    repeat: int,
    protocol_hash: str,
    frames: Dict[str, pd.DataFrame],
) -> None:
    paths = []

    for kind, frame in frames.items():
        path = checkpoint_data_path(directory, repeat, kind)
        frame.to_csv(path, sep="\t", index=False)
        paths.append(path)

    manifest = {
        "status": "PASS",
        "created_utc": now_utc(),
        "protocol_sha256": protocol_hash,
        "repeat": repeat,
        "artifact_hashes": {
            path.name: sha256_file(path)
            for path in paths
        },
    }
    write_json(
        checkpoint_manifest_path(directory, repeat),
        manifest,
    )


# ===========================================================================
# Primary B0/B2/B3/B4 repeated CV.
# ===========================================================================

def run_primary_repeat(
    repeat: int,
    X_gene: np.ndarray,
    y: np.ndarray,
    event: np.ndarray,
    sample_ids: np.ndarray,
    module_to_indices: Dict[str, np.ndarray],
    module_order: List[str],
    beta_dog: np.ndarray,
    dog_modules_full: np.ndarray,
    protocol: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    base_seed = int(protocol["paired_human_evaluation"]["base_seed"])
    outer = stratified_splits(
        event,
        OUTER_SPLITS,
        base_seed + repeat,
    )

    pred_rows = []
    fold_rows = []
    tuning_parts = []
    ibs_rows = []

    for outer_fold, (train_idx, test_idx) in enumerate(outer):
        inner_seed = base_seed + 1000 + repeat * 100 + outer_fold

        # ------------------------------------------------------------------
        # B0 human-only centered at zero.
        # ------------------------------------------------------------------
        b0_alpha, b0_audit = tune_centered_ridge(
            X_gene[train_idx],
            y[train_idx],
            event[train_idx],
            module_to_indices,
            module_order,
            np.zeros(len(module_order), dtype=float),
            RIDGE_ALPHAS,
            INNER_SPLITS,
            inner_seed,
        )

        b0_audit = b0_audit.copy()
        b0_audit["model"] = "B0"
        b0_audit["repeat"] = repeat
        b0_audit["outer_fold"] = outer_fold
        tuning_parts.append(b0_audit)

        # ------------------------------------------------------------------
        # B4 residual Cox centered at frozen beta_DOG2.
        # ------------------------------------------------------------------
        b4_alpha, b4_audit = tune_centered_ridge(
            X_gene[train_idx],
            y[train_idx],
            event[train_idx],
            module_to_indices,
            module_order,
            beta_dog,
            RIDGE_ALPHAS,
            INNER_SPLITS,
            inner_seed + 50000,
        )

        b4_audit = b4_audit.copy()
        b4_audit["model"] = "B4"
        b4_audit["repeat"] = repeat
        b4_audit["outer_fold"] = outer_fold
        tuning_parts.append(b4_audit)

        transformer = HumanHallmarkTransformer(
            module_to_indices,
            module_order,
        ).fit(X_gene[train_idx])

        X_train_m = transformer.transform(X_gene[train_idx])
        X_test_m = transformer.transform(X_gene[test_idx])

        b0 = CenteredRidgeCox(
            alpha=b0_alpha,
            center=np.zeros(len(module_order), dtype=float),
        ).fit(X_train_m, y[train_idx])

        b4 = CenteredRidgeCox(
            alpha=b4_alpha,
            center=beta_dog.copy(),
        ).fit(X_train_m, y[train_idx])

        # B2 zero-shot.
        b2_beta = beta_dog.copy()

        # B3 CORAL zero-shot.
        b3_beta, coral_diag = coral_beta_to_target(
            dog_modules_full,
            X_train_m,
            beta_dog,
            CORAL_EPS,
        )

        model_risks: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
            "B0": (
                b0.predict(X_train_m),
                b0.predict(X_test_m),
            ),
            "B2": (
                X_train_m @ b2_beta,
                X_test_m @ b2_beta,
            ),
            "B3": (
                X_train_m @ b3_beta,
                X_test_m @ b3_beta,
            ),
            "B4": (
                b4.predict(X_train_m),
                b4.predict(X_test_m),
            ),
        }

        for model_name, (train_risk, test_risk) in model_risks.items():
            mu = float(np.mean(train_risk))
            sd = float(np.std(train_risk, ddof=0))
            if not np.isfinite(sd) or sd <= SD_EPS:
                sd = 1.0

            test_std = (test_risk - mu) / sd
            fold_c = safe_uno_c(
                y[train_idx],
                y[test_idx],
                test_std,
            )

            fold_rows.append(
                {
                    "repeat": repeat,
                    "outer_fold": outer_fold,
                    "model": model_name,
                    "n_train": len(train_idx),
                    "events_train": int(event[train_idx].sum()),
                    "n_test": len(test_idx),
                    "events_test": int(event[test_idx].sum()),
                    "chosen_alpha": (
                        b0_alpha if model_name == "B0"
                        else b4_alpha if model_name == "B4"
                        else float("nan")
                    ),
                    "fold_uno_c": fold_c,
                    "train_risk_mean": mu,
                    "train_risk_sd": sd,
                    "coral_alignment_condition": (
                        coral_diag["alignment_condition"]
                        if model_name == "B3"
                        else float("nan")
                    ),
                }
            )

            for local_idx, global_idx in enumerate(test_idx):
                pred_rows.append(
                    {
                        "repeat": repeat,
                        "outer_fold": outer_fold,
                        "model": model_name,
                        "sample_id": sample_ids[global_idx],
                        "sample_index": int(global_idx),
                        "risk_standardized": float(test_std[local_idx]),
                    }
                )

        # Shared B0/B4 IBS grid. The original 04b construction is unchanged.
        # 04c0 only defines conservative handling when that frozen grid is
        # infeasible in a small held-out fold. No alternative grid is created.
        ibs_status = "PASS"
        ibs_error = ""
        grid = None
        ibs_b0 = float("nan")
        ibs_b4 = float("nan")

        try:
            grid = choose_ibs_grid(
                y[train_idx],
                y[test_idx],
            )

            b0_surv = b0.predict_survival_matrix(
                X_test_m,
                grid,
            )
            b4_surv = b4.predict_survival_matrix(
                X_test_m,
                grid,
            )

            ibs_b0 = integrated_brier_custom(
                y[train_idx],
                y[test_idx],
                b0_surv,
                grid,
            )
            ibs_b4 = integrated_brier_custom(
                y[train_idx],
                y[test_idx],
                b4_surv,
                grid,
            )

        except RuntimeError as exc:
            message = str(exc)
            if "supported IBS points; minimum=" not in message:
                raise
            ibs_status = "IBS_UNAVAILABLE_INSUFFICIENT_COMMON_SUPPORT"
            ibs_error = message

        grid_n = int(len(grid)) if grid is not None else 0
        grid_min = float(grid[0]) if grid is not None and len(grid) else float("nan")
        grid_max = float(grid[-1]) if grid is not None and len(grid) else float("nan")

        ibs_rows.extend(
            [
                {
                    "repeat": repeat,
                    "outer_fold": outer_fold,
                    "model": "B0",
                    "n_test": len(test_idx),
                    "time_grid_n": grid_n,
                    "time_grid_min": grid_min,
                    "time_grid_max": grid_max,
                    "ibs": ibs_b0,
                    "ibs_status": ibs_status,
                    "ibs_error": ibs_error,
                },
                {
                    "repeat": repeat,
                    "outer_fold": outer_fold,
                    "model": "B4",
                    "n_test": len(test_idx),
                    "time_grid_n": grid_n,
                    "time_grid_min": grid_min,
                    "time_grid_max": grid_max,
                    "ibs": ibs_b4,
                    "ibs_status": ibs_status,
                    "ibs_error": ibs_error,
                },
            ]
        )

        print(
            f"    repeat {repeat+1:02d}/{OUTER_REPEATS}, fold {outer_fold+1}/5: "
            f"B0 C={model_risks and fold_rows[-4]['fold_uno_c']:.3f} "
            f"B4 alpha={b4_alpha:g}"
        )

    return {
        "predictions": pd.DataFrame(pred_rows),
        "folds": pd.DataFrame(fold_rows),
        "tuning": pd.concat(tuning_parts, ignore_index=True),
        "ibs": pd.DataFrame(ibs_rows),
    }


def aggregate_predictions(
    predictions: pd.DataFrame,
    sample_ids: np.ndarray,
    model: str,
) -> np.ndarray:
    part = predictions[predictions["model"] == model]
    means = part.groupby("sample_id")["risk_standardized"].mean()

    if set(means.index.astype(str)) != set(sample_ids.astype(str)):
        raise RuntimeError(f"{model}: aggregated OOF sample coverage mismatch.")

    return np.asarray(
        [means.loc[x] for x in sample_ids],
        dtype=float,
    )


def weighted_mean_ibs(
    ibs: pd.DataFrame,
    model: str,
    *,
    require_complete: bool,
) -> float:
    part = ibs[ibs["model"] == model].copy()
    values = pd.to_numeric(part["ibs"], errors="coerce").to_numpy(dtype=float)
    weights = pd.to_numeric(part["n_test"], errors="raise").to_numpy(dtype=float)

    finite = np.isfinite(values)

    if require_complete and not finite.all():
        raise RuntimeError(f"{model}: primary IBS is incomplete under frozen support rule.")

    if int(finite.sum()) == 0:
        return float("nan")

    return float(np.average(values[finite], weights=weights[finite]))


def paired_bootstrap_delta_c(
    y: np.ndarray,
    risk_b0: np.ndarray,
    risk_b4: np.ndarray,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(y)

    rows = []
    attempts = 0
    max_attempts = n_boot * 10

    while len(rows) < n_boot and attempts < max_attempts:
        attempts += 1
        idx = rng.integers(0, n, size=n)

        c0 = safe_uno_c(y, y[idx], risk_b0[idx])
        c4 = safe_uno_c(y, y[idx], risk_b4[idx])

        if not np.isfinite(c0) or not np.isfinite(c4):
            continue

        rows.append(
            {
                "bootstrap_index": len(rows),
                "uno_c_B0": float(c0),
                "uno_c_B4": float(c4),
                "delta_c_B4_minus_B0": float(c4 - c0),
            }
        )

    if len(rows) != n_boot:
        raise RuntimeError(
            f"Only {len(rows)}/{n_boot} finite paired bootstrap replicates."
        )

    return pd.DataFrame(rows)


# ===========================================================================
# B1 repeated gene elastic-net.
# ===========================================================================

def run_B1_repeat(
    repeat: int,
    X_gene: np.ndarray,
    y: np.ndarray,
    event: np.ndarray,
    sample_ids: np.ndarray,
    protocol: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    base_seed = int(protocol["paired_human_evaluation"]["base_seed"]) + 200000
    outer = stratified_splits(
        event,
        OUTER_SPLITS,
        base_seed + repeat,
    )

    pred_rows = []
    fold_rows = []
    tuning_parts = []

    for outer_fold, (train_idx, test_idx) in enumerate(outer):
        inner_seed = base_seed + 1000 + repeat * 100 + outer_fold

        l1, path_index, audit = tune_gene_elastic_net(
            X_gene[train_idx],
            y[train_idx],
            event[train_idx],
            INNER_SPLITS,
            inner_seed,
        )

        audit = audit.copy()
        audit["repeat"] = repeat
        audit["outer_fold"] = outer_fold
        tuning_parts.append(audit)

        transformer = GeneTransformer().fit(X_gene[train_idx])
        X_train = transformer.transform(X_gene[train_idx])
        X_test = transformer.transform(X_gene[test_idx])

        model = fit_coxnet_path(X_train, y[train_idx], l1)
        alpha = float(model.alphas_[path_index])

        train_risk = np.asarray(
            model.predict(X_train, alpha=alpha),
            dtype=float,
        )
        test_risk = np.asarray(
            model.predict(X_test, alpha=alpha),
            dtype=float,
        )

        mu = float(np.mean(train_risk))
        sd = float(np.std(train_risk, ddof=0))
        if not np.isfinite(sd) or sd <= SD_EPS:
            sd = 1.0

        test_std = (test_risk - mu) / sd
        c = safe_uno_c(
            y[train_idx],
            y[test_idx],
            test_std,
        )

        fold_rows.append(
            {
                "repeat": repeat,
                "outer_fold": outer_fold,
                "n_train": len(train_idx),
                "events_train": int(event[train_idx].sum()),
                "n_test": len(test_idx),
                "events_test": int(event[test_idx].sum()),
                "chosen_l1_ratio": l1,
                "chosen_path_index": path_index,
                "chosen_alpha": alpha,
                "n_training_genes": int(transformer.valid_.sum()),
                "fold_uno_c": c,
            }
        )

        for local_idx, global_idx in enumerate(test_idx):
            pred_rows.append(
                {
                    "repeat": repeat,
                    "outer_fold": outer_fold,
                    "model": "B1",
                    "sample_id": sample_ids[global_idx],
                    "sample_index": int(global_idx),
                    "risk_standardized": float(test_std[local_idx]),
                }
            )

    return {
        "predictions": pd.DataFrame(pred_rows),
        "folds": pd.DataFrame(fold_rows),
        "tuning": pd.concat(tuning_parts, ignore_index=True),
    }


# ===========================================================================
# Event-matched DOG secondary.
# ===========================================================================

def balanced_dog_subset(
    dog_event: np.ndarray,
    dog_arm: np.ndarray,
    n_total: int,
    n_events: int,
    seed: int,
    draw_index: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    counts_needed = {
        True: int(n_events),
        False: int(n_total - n_events),
    }

    selected: List[int] = []

    for stratum_event, k in counts_needed.items():
        # Alternate which arm receives the odd sample.
        if k % 2 == 0:
            k21 = k // 2
            k22 = k // 2
        else:
            if draw_index % 2 == 0:
                k21 = k // 2 + 1
                k22 = k // 2
            else:
                k21 = k // 2
                k22 = k // 2 + 1

        idx21 = np.flatnonzero(
            (dog_event == stratum_event)
            & (dog_arm == "COTC021")
        )
        idx22 = np.flatnonzero(
            (dog_event == stratum_event)
            & (dog_arm == "COTC022")
        )

        if len(idx21) < k21 or len(idx22) < k22:
            raise RuntimeError(
                "DOG2 cannot support frozen balanced event-matched draw."
            )

        selected.extend(
            rng.choice(idx21, size=k21, replace=False).tolist()
        )
        selected.extend(
            rng.choice(idx22, size=k22, replace=False).tolist()
        )

    selected = np.asarray(selected, dtype=int)

    if len(selected) != n_total:
        raise RuntimeError("Event-matched DOG draw has wrong sample count.")
    if int(dog_event[selected].sum()) != n_events:
        raise RuntimeError("Event-matched DOG draw has wrong event count.")
    if len(np.unique(selected)) != len(selected):
        raise RuntimeError("Event-matched DOG draw contains duplicates.")

    return selected


def run_event_matched_repeat(
    repeat: int,
    m03b,
    X_human_gene: np.ndarray,
    y_human: np.ndarray,
    event_human: np.ndarray,
    sample_ids: np.ndarray,
    human_module_to_indices: Dict[str, np.ndarray],
    human_module_order: List[str],
    dog_data: Dict[str, Any],
    source_alpha: float,
    protocol: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    base_seed = int(
        protocol["secondary_event_matched_information_efficiency"]["base_seed"]
    )
    outer_seed = int(protocol["paired_human_evaluation"]["base_seed"]) + repeat

    outer = stratified_splits(
        event_human,
        OUTER_SPLITS,
        outer_seed,
    )

    pred_rows = []
    draw_rows = []

    for outer_fold, (train_idx, test_idx) in enumerate(outer):
        n_h = len(train_idx)
        e_h = int(event_human[train_idx].sum())

        human_transformer = HumanHallmarkTransformer(
            human_module_to_indices,
            human_module_order,
        ).fit(X_human_gene[train_idx])

        H_train = human_transformer.transform(X_human_gene[train_idx])
        H_test = human_transformer.transform(X_human_gene[test_idx])

        for draw in range(EVENT_MATCHED_DRAWS_PER_OUTER_FOLD):
            draw_seed = (
                base_seed
                + repeat * 10000
                + outer_fold * 100
                + draw
            )

            dog_idx = balanced_dog_subset(
                dog_data["endpoint_data"]["OS"]["event"],
                dog_data["arms"],
                n_h,
                e_h,
                draw_seed,
                draw,
            )

            dog_transformer = m03b.HallmarkTransformer(
                dog_data["module_to_indices"],
                dog_data["module_order"],
                10,
                1e-12,
            ).fit(dog_data["X_hallmark_gene"][dog_idx])

            D_train = dog_transformer.transform(
                dog_data["X_hallmark_gene"][dog_idx]
            )

            model = m03b.fit_ridge(
                D_train,
                dog_data["endpoint_data"]["OS"]["y"][dog_idx],
                source_alpha,
            )

            beta = np.asarray(model.coef_, dtype=float)

            # DOG and human module orders are both lexicographically frozen;
            # verify exact identity before applying coefficients.
            if dog_data["module_order"] != human_module_order:
                raise RuntimeError(
                    "DOG/HUMAN Hallmark module order mismatch in event-matched analysis."
                )

            train_risk = H_train @ beta
            test_risk = H_test @ beta

            mu = float(np.mean(train_risk))
            sd = float(np.std(train_risk, ddof=0))
            if not np.isfinite(sd) or sd <= SD_EPS:
                sd = 1.0

            test_std = (test_risk - mu) / sd

            fold_c = safe_uno_c(
                y_human[train_idx],
                y_human[test_idx],
                test_std,
            )

            draw_rows.append(
                {
                    "repeat": repeat,
                    "outer_fold": outer_fold,
                    "draw": draw,
                    "human_train_n": n_h,
                    "human_train_events": e_h,
                    "dog_draw_n": len(dog_idx),
                    "dog_draw_events": int(
                        dog_data["endpoint_data"]["OS"]["event"][dog_idx].sum()
                    ),
                    "dog_COTC021_n": int(
                        np.sum(dog_data["arms"][dog_idx] == "COTC021")
                    ),
                    "dog_COTC022_n": int(
                        np.sum(dog_data["arms"][dog_idx] == "COTC022")
                    ),
                    "fold_uno_c": fold_c,
                }
            )

            for local_idx, global_idx in enumerate(test_idx):
                pred_rows.append(
                    {
                        "repeat": repeat,
                        "outer_fold": outer_fold,
                        "draw": draw,
                        "sample_id": sample_ids[global_idx],
                        "sample_index": int(global_idx),
                        "risk_standardized": float(test_std[local_idx]),
                    }
                )

    return {
        "predictions": pd.DataFrame(pred_rows),
        "draws": pd.DataFrame(draw_rows),
    }


# ===========================================================================
# Main.
# ===========================================================================

def main() -> None:
    started = now_utc()

    print("=" * 120)
    print("Paper 6 - run frozen GSE16091 classical canine-added-value premise test")
    print("=" * 120)
    print(f"Script version: {SCRIPT_VERSION}")
    print()
    print("Safety / execution contract:")
    print("  GSE16091 expression values read: YES")
    print("  GSE16091 OS outcome values read: YES")
    print("  TARGET-OS outcomes read: NO")
    print("  GSE21257 outcomes read: NO")
    print("  GSE39055 outcomes read: NO")
    print("  Premise thresholds changed after outcome access: NO")
    print("  AI architecture tuning: NO")
    print("  DOG2 new hyperparameter tuning: NO")
    print("  GPU execution: NO")
    print("  Post-outcome technical amendment: YES [04c0 conservative-only]")
    print("  Amendment can create SUPPORTS result: NO")
    print("  v3 scientific change vs v2: NO [secondary TSV path/count namespace fix only]")
    print()

    required = [
        B4_PROTOCOL,
        B4_SUMMARY,
        B4_PROBE_MAP,
        B4_GENE_UNIVERSE,
        B4_HALLMARK_MAP,
        B4_DOG_ALPHA,
        C40_AMENDMENT,
        C40_SUMMARY,
        A4_CONTRACT,
        A4_ROSTER,
        A3_PROTOCOL,
        A3_SUMMARY,
        B3_SCRIPT,
        UPSTREAM_LOCK,
        B3_SUMMARY,
        C3_POLICY,
        C3_SUMMARY,
    ]
    for path in required:
        require_file(path)

    protocol = read_json(B4_PROTOCOL)
    protocol_summary = read_json(B4_SUMMARY)
    ibs_amendment = read_json(C40_AMENDMENT)
    ibs_amendment_summary = read_json(C40_SUMMARY)
    a4_contract = read_json(A4_CONTRACT)
    a3_protocol = read_json(A3_PROTOCOL)
    b3_summary = read_json(B3_SUMMARY)
    c3_policy = read_json(C3_POLICY)
    upstream_lock = read_json(UPSTREAM_LOCK)

    if clean(protocol.get("status")) != "PASS":
        raise RuntimeError("04b protocol is not PASS.")
    if clean(protocol_summary.get("scientific_status")) != (
        "PASS_CLASSICAL_PREMISE_TEST_IMPLEMENTATION_FROZEN"
    ):
        raise RuntimeError("04b protocol is not in expected frozen PASS state.")
    if clean(b3_summary.get("scientific_status")) != EXPECTED_SOURCE_GATE:
        raise RuntimeError("03b source gate status changed.")

    protocol_hash = sha256_file(B4_PROTOCOL)
    expected_protocol_hash = clean(
        (protocol_summary.get("final_artifact_hashes") or {}).get(
            "classical_premise_test_protocol_json"
        )
    )
    if protocol_hash != expected_protocol_hash:
        raise RuntimeError("04b protocol hash verification failed.")

    if clean(ibs_amendment.get("status")) != "PASS":
        raise RuntimeError("04c0 IBS-feasibility amendment is not PASS.")
    if clean(ibs_amendment.get("scientific_status")) != (
        "PASS_POSTOUTCOME_TECHNICAL_AMENDMENT_CONSERVATIVE_ONLY"
    ):
        raise RuntimeError("04c0 amendment is not the expected conservative-only amendment.")
    if clean(ibs_amendment_summary.get("status")) != "PASS":
        raise RuntimeError("04c0 amendment summary is not PASS.")
    if clean(ibs_amendment.get("unchanged", {}).get("04b_protocol_sha256")) != protocol_hash:
        raise RuntimeError("04c0 amendment does not lock the exact current 04b protocol.")
    if not bool(
        ibs_amendment.get("frozen_execution_handling", {}).get(
            "SUPPORTS_CANINE_ADDED_VALUE"
        )
    ):
        raise RuntimeError("04c0 amendment handling is incomplete.")

    # ------------------------------------------------------------------
    # Verify frozen constants BEFORE human outcome access.
    # ------------------------------------------------------------------
    paired = protocol["paired_human_evaluation"]
    premise = protocol["primary_premise_contrast"]

    if int(paired["outer_splits"]) != OUTER_SPLITS:
        raise RuntimeError("04b outer split count changed.")
    if int(paired["outer_repeats"]) != OUTER_REPEATS:
        raise RuntimeError("04b outer repeat count changed.")
    if int(paired["inner_splits"]) != INNER_SPLITS:
        raise RuntimeError("04b inner split count changed.")
    if int(paired["patient_bootstrap_replicates"]) != PATIENT_BOOTSTRAPS:
        raise RuntimeError("04b patient bootstrap count changed.")

    if float(
        premise["SUPPORTS_CANINE_ADDED_VALUE"]["delta_uno_c_min"]
    ) != PREMISE_DELTA_C_SUPPORT:
        raise RuntimeError("04b premise delta-C support threshold changed.")
    if float(
        premise["SUPPORTS_CANINE_ADDED_VALUE"][
            "bootstrap_P_delta_c_gt_zero_min"
        ]
    ) != PREMISE_PROB_SUPPORT:
        raise RuntimeError("04b premise positive-probability threshold changed.")
    if float(
        premise["SUPPORTS_CANINE_ADDED_VALUE"]["delta_IBS_max"]
    ) != PREMISE_MAX_DELTA_IBS:
        raise RuntimeError("04b premise IBS threshold changed.")
    if float(
        premise["ARGUES_AGAINST_CANINE_ADDED_VALUE"]["delta_uno_c_max"]
    ) != PREMISE_DELTA_C_AGAINST:
        raise RuntimeError("04b premise negative delta-C threshold changed.")
    if float(
        premise["ARGUES_AGAINST_CANINE_ADDED_VALUE"][
            "bootstrap_P_delta_c_lt_zero_min"
        ]
    ) != PREMISE_PROB_AGAINST:
        raise RuntimeError("04b premise negative-probability threshold changed.")

    selected = protocol["target"]
    if clean(selected["accession"]) != EXPECTED_TARGET:
        raise RuntimeError("04b target changed.")
    if int(selected["n"]) != EXPECTED_TARGET_N:
        raise RuntimeError("04b target n changed.")

    print("04b protocol verification: PASS")
    print(f"Protocol SHA256: {protocol_hash}")
    print()

    # ------------------------------------------------------------------
    # Fit frozen full-DOG2 source model BEFORE using GSE16091 outcomes.
    # ------------------------------------------------------------------
    print("-" * 120)
    print("Frozen full-DOG2 source model")
    print("-" * 120)

    m03b = load_module(B3_SCRIPT, "paper6_03b_v2_for_04c")
    dog_data = m03b.load_inputs(a3_protocol, upstream_lock)

    alpha_df = pd.read_csv(B4_DOG_ALPHA, sep="\t")
    if len(alpha_df) != 1:
        raise RuntimeError("04b DOG source-alpha contract is malformed.")

    source_alpha = float(
        alpha_df.iloc[0]["selected_full_DOG2_source_alpha"]
    )
    if source_alpha != float(protocol["DOG2_full_source_model"]["source_alpha"]):
        raise RuntimeError("DOG source alpha differs from 04b frozen value.")

    dog_transformer = m03b.HallmarkTransformer(
        dog_data["module_to_indices"],
        dog_data["module_order"],
        10,
        1e-12,
    ).fit(dog_data["X_hallmark_gene"])

    dog_modules_full = dog_transformer.transform(
        dog_data["X_hallmark_gene"]
    )

    dog_model = m03b.fit_ridge(
        dog_modules_full,
        dog_data["endpoint_data"]["OS"]["y"],
        source_alpha,
    )
    beta_dog = np.asarray(dog_model.coef_, dtype=float)

    if len(beta_dog) != 50 or not np.isfinite(beta_dog).all():
        raise RuntimeError("Invalid frozen full-DOG2 coefficient vector.")

    pd.DataFrame(
        {
            "hallmark_module": dog_data["module_order"],
            "beta_DOG2": beta_dog,
        }
    ).to_csv(DOG_SOURCE_COEFS, sep="\t", index=False)

    print(f"DOG2 n: {len(dog_data['sample_ids'])}")
    print(f"DOG2 OS events: {int(dog_data['endpoint_data']['OS']['event'].sum())}")
    print(f"Frozen source alpha: {source_alpha}")
    print(f"Source beta L2 norm: {np.linalg.norm(beta_dog):.6f}")
    print("New DOG2 hyperparameter search: NO")

    # ------------------------------------------------------------------
    # FIRST GSE16091 outcome/expression access.
    # ------------------------------------------------------------------
    print()
    print("-" * 120)
    print("GSE16091 frozen snapshot [FIRST Paper-6 premise-target outcome access]")
    print("-" * 120)

    matrix_url = clean(
        protocol["human_expression_retrieval_after_04b"]["series_matrix_url"]
    )
    if matrix_url != DEFAULT_SERIES_MATRIX_URL:
        raise RuntimeError("04b frozen GSE16091 matrix URL differs from expected.")

    download_locked_series_matrix(
        matrix_url,
        SERIES_MATRIX_GZ,
        SERIES_MATRIX_LOCK,
    )

    sample_accessions, characteristics, probe_expression = parse_series_matrix(
        SERIES_MATRIX_GZ
    )

    roster = pd.read_csv(A4_ROSTER, sep="\t", dtype=str).fillna("")
    frozen_samples = roster["geo_accession"].astype(str).tolist()

    if set(sample_accessions) != set(frozen_samples):
        raise RuntimeError(
            "GSE16091 series-matrix samples do not exactly match 04a frozen roster."
        )

    # Reorder all target data to exact 04a frozen roster.
    sample_accessions = frozen_samples
    probe_expression = probe_expression[sample_accessions]

    status_mapping = {
        str(k): int(v)
        for k, v in selected["status_mapping"].items()
    }

    outcome = build_target_outcome(
        sample_accessions,
        characteristics,
        selected["time_field"],
        selected["status_field"],
        status_mapping,
    )

    outcome = outcome.set_index("geo_accession").loc[sample_accessions].reset_index()

    if len(outcome) != EXPECTED_TARGET_N:
        raise RuntimeError("GSE16091 outcome rows are not 34.")

    event = outcome["os_event"].to_numpy(dtype=int).astype(bool)
    time_values = outcome["os_time"].to_numpy(dtype=float)
    y = make_surv(time_values, event)
    sample_ids = np.asarray(sample_accessions, dtype=str)

    class_counts = pd.Series(event.astype(int)).value_counts().to_dict()
    if len(class_counts) != 2 or min(class_counts.values()) < OUTER_SPLITS:
        raise RuntimeError(
            f"GSE16091 cannot support frozen 5-fold stratification: {class_counts}"
        )

    # Probe -> outcome-blind gene collapse.
    probe_map = pd.read_csv(B4_PROBE_MAP, sep="\t", dtype=str).fillna("")
    gene_universe = pd.read_csv(
        B4_GENE_UNIVERSE,
        sep="\t",
        dtype=str,
    ).fillna("")
    hallmark_map = pd.read_csv(
        B4_HALLMARK_MAP,
        sep="\t",
        dtype=str,
    ).fillna("")

    X_gene, gene_order = collapse_probes_to_genes(
        probe_expression,
        probe_map,
        gene_universe,
        sample_accessions,
    )

    gene_index = {
        gene: i for i, gene in enumerate(gene_order)
    }

    human_module_order = sorted(
        hallmark_map["hallmark_module"].astype(str).unique().tolist()
    )

    human_module_to_indices = {}

    for module in human_module_order:
        genes = (
            hallmark_map.loc[
                hallmark_map["hallmark_module"].astype(str) == module,
                "human_gene_symbol",
            ]
            .astype(str)
            .tolist()
        )
        human_module_to_indices[module] = np.asarray(
            [gene_index[g] for g in genes],
            dtype=int,
        )

    if human_module_order != dog_data["module_order"]:
        raise RuntimeError(
            "Frozen DOG2 and GSE16091 Hallmark module orders are not identical."
        )

    # Input identity artifacts.
    input_audit = pd.DataFrame(
        [
            {
                "target": EXPECTED_TARGET,
                "n_samples": len(sample_ids),
                "n_probes": int(probe_expression.shape[0]),
                "n_common_genes": int(X_gene.shape[1]),
                "n_hallmark_modules": len(human_module_order),
                "series_matrix_sha256": sha256_file(SERIES_MATRIX_GZ),
                "sample_roster_sha256": sha256_file(A4_ROSTER),
                "sample_set_match": True,
                "reserved_human_cohorts_read": False,
            }
        ]
    )
    input_audit.to_csv(INPUT_AUDIT, sep="\t", index=False)

    outcome_audit = pd.DataFrame(
        [
            {
                "endpoint": "OS",
                "n": len(y),
                "events": int(event.sum()),
                "censored": int((~event).sum()),
                "event_fraction": float(event.mean()),
                "time_min": float(np.min(time_values)),
                "time_median": float(np.median(time_values)),
                "time_max": float(np.max(time_values)),
                "status_values": ";".join(
                    sorted(outcome["status_raw"].astype(str).unique())
                ),
                "frozen_status_mapping": json.dumps(
                    status_mapping,
                    sort_keys=True,
                ),
            }
        ]
    )
    outcome_audit.to_csv(OUTCOME_AUDIT, sep="\t", index=False)

    print(f"GSE16091 sample identity: {len(sample_ids)}/{EXPECTED_TARGET_N} PASS")
    print(f"OS events: {int(event.sum())}")
    print(f"OS censored: {int((~event).sum())}")
    print(f"Outcome-blind common genes: {X_gene.shape[1]:,}")
    print(f"Hallmark modules: {len(human_module_order)}/50")

    # ------------------------------------------------------------------
    # Primary B0/B2/B3/B4 stage, checkpointed per repeat.
    # ------------------------------------------------------------------
    print()
    print("-" * 120)
    print("Primary paired classical premise stage: B4 versus B0")
    print("-" * 120)

    primary_parts = {
        "predictions": [],
        "folds": [],
        "tuning": [],
        "ibs": [],
    }

    for repeat in range(OUTER_REPEATS):
        cached = load_repeat_checkpoint(
            PRIMARY_CKPT,
            repeat,
            protocol_hash,
            ["predictions", "folds", "tuning", "ibs"],
        )

        if cached is not None:
            print(f"  primary repeat {repeat+1:02d}/{OUTER_REPEATS}: REUSED")
            result = cached
        else:
            print(f"  primary repeat {repeat+1:02d}/{OUTER_REPEATS}: RUNNING")
            result = run_primary_repeat(
                repeat,
                X_gene,
                y,
                event,
                sample_ids,
                human_module_to_indices,
                human_module_order,
                beta_dog,
                dog_modules_full,
                protocol,
            )
            save_repeat_checkpoint(
                PRIMARY_CKPT,
                repeat,
                protocol_hash,
                result,
            )

        for kind in primary_parts:
            primary_parts[kind].append(result[kind])

    primary_pred = pd.concat(
        primary_parts["predictions"],
        ignore_index=True,
    )
    primary_folds = pd.concat(
        primary_parts["folds"],
        ignore_index=True,
    )
    primary_tuning = pd.concat(
        primary_parts["tuning"],
        ignore_index=True,
    )
    primary_ibs = pd.concat(
        primary_parts["ibs"],
        ignore_index=True,
    )

    # v1 validated repeats predate the explicit status columns; successful
    # cached rows are therefore PASS by construction.
    if "ibs_status" not in primary_ibs.columns:
        primary_ibs["ibs_status"] = "PASS"
    else:
        primary_ibs["ibs_status"] = primary_ibs["ibs_status"].fillna("PASS")
    if "ibs_error" not in primary_ibs.columns:
        primary_ibs["ibs_error"] = ""
    else:
        primary_ibs["ibs_error"] = primary_ibs["ibs_error"].fillna("")

    primary_pred.to_csv(PRIMARY_PRED, sep="\t", index=False)
    primary_folds.to_csv(PRIMARY_FOLDS, sep="\t", index=False)
    primary_tuning.to_csv(PRIMARY_TUNING, sep="\t", index=False)
    primary_ibs.to_csv(PRIMARY_IBS, sep="\t", index=False)

    expected_pred_rows = EXPECTED_TARGET_N * OUTER_REPEATS * 4
    if len(primary_pred) != expected_pred_rows:
        raise RuntimeError(
            f"Primary prediction rows={len(primary_pred)}, expected={expected_pred_rows}."
        )

    risk = {}
    c_index = {}

    for model_name in ["B0", "B2", "B3", "B4"]:
        risk[model_name] = aggregate_predictions(
            primary_pred,
            sample_ids,
            model_name,
        )
        c_index[model_name] = safe_uno_c(
            y,
            y,
            risk[model_name],
        )

    if not all(np.isfinite(list(c_index.values()))):
        raise RuntimeError(f"Non-finite primary model Uno C: {c_index}")

    unavailable_ibs_fold_keys = (
        primary_ibs.loc[
            primary_ibs["ibs_status"].astype(str) != "PASS",
            ["repeat", "outer_fold"],
        ]
        .drop_duplicates()
        .sort_values(["repeat", "outer_fold"])
    )
    n_unavailable_ibs_folds = int(len(unavailable_ibs_fold_keys))
    ibs_complete_for_primary = n_unavailable_ibs_folds == 0

    # Feasible-fold summaries are retained for transparency only when the
    # original IBS estimand is incomplete. They cannot establish SUPPORTS.
    ibs_b0_descriptive = weighted_mean_ibs(
        primary_ibs,
        "B0",
        require_complete=False,
    )
    ibs_b4_descriptive = weighted_mean_ibs(
        primary_ibs,
        "B4",
        require_complete=False,
    )
    delta_ibs_descriptive = (
        float(ibs_b4_descriptive - ibs_b0_descriptive)
        if np.isfinite(ibs_b0_descriptive) and np.isfinite(ibs_b4_descriptive)
        else float("nan")
    )

    if ibs_complete_for_primary:
        ibs_b0 = weighted_mean_ibs(
            primary_ibs,
            "B0",
            require_complete=True,
        )
        ibs_b4 = weighted_mean_ibs(
            primary_ibs,
            "B4",
            require_complete=True,
        )
        delta_ibs = float(ibs_b4 - ibs_b0)
    else:
        ibs_b0 = None
        ibs_b4 = None
        delta_ibs = None

    bootstrap = paired_bootstrap_delta_c(
        y,
        risk["B0"],
        risk["B4"],
        PATIENT_BOOTSTRAPS,
        int(protocol["paired_human_evaluation"]["patient_bootstrap_seed"]),
    )
    bootstrap.to_csv(PRIMARY_BOOTSTRAP, sep="\t", index=False)

    delta_boot = bootstrap["delta_c_B4_minus_B0"].to_numpy(dtype=float)

    delta_c = float(c_index["B4"] - c_index["B0"])
    prob_gt_zero = float(np.mean(delta_boot > 0))
    prob_lt_zero = float(np.mean(delta_boot < 0))
    delta_ci = [
        float(np.quantile(delta_boot, 0.025)),
        float(np.quantile(delta_boot, 0.975)),
    ]

    negative_rule_met = (
        delta_c <= PREMISE_DELTA_C_AGAINST
        and prob_lt_zero >= PREMISE_PROB_AGAINST
    )

    positive_discrimination_rule_met = (
        delta_c >= PREMISE_DELTA_C_SUPPORT
        and prob_gt_zero >= PREMISE_PROB_SUPPORT
    )

    if negative_rule_met:
        # Exact pre-frozen negative rule; IBS was never part of this criterion.
        premise_state = "ARGUES_AGAINST_CANINE_ADDED_VALUE"
    elif (
        ibs_complete_for_primary
        and positive_discrimination_rule_met
        and delta_ibs is not None
        and delta_ibs <= PREMISE_MAX_DELTA_IBS
    ):
        premise_state = "SUPPORTS_CANINE_ADDED_VALUE"
    else:
        # Conservative 04c0 handling: incomplete IBS can never create SUPPORTS.
        premise_state = "INCONCLUSIVE_NEUTRAL"

    premise_support_evaluable = bool(ibs_complete_for_primary)
    ibs_feasibility_limitation = not ibs_complete_for_primary

    primary_result = {
        "status": "PASS",
        "created_utc": now_utc(),
        "protocol_sha256": protocol_hash,
        "target": EXPECTED_TARGET,
        "n": EXPECTED_TARGET_N,
        "events": int(event.sum()),
        "models": {
            model: {
                "aggregated_oof_uno_c": float(c_index[model]),
            }
            for model in ["B0", "B2", "B3", "B4"]
        },
        "primary_comparison": {
            "reference": "B0",
            "canine_added_value_model": "B4",
            "uno_c_B0": float(c_index["B0"]),
            "uno_c_B4": float(c_index["B4"]),
            "delta_uno_c_B4_minus_B0": delta_c,
            "delta_uno_c_bootstrap_95": delta_ci,
            "bootstrap_P_delta_c_gt_zero": prob_gt_zero,
            "bootstrap_P_delta_c_lt_zero": prob_lt_zero,
            "weighted_mean_IBS_B0": ibs_b0,
            "weighted_mean_IBS_B4": ibs_b4,
            "delta_IBS_B4_minus_B0": delta_ibs,
            "IBS_complete_for_primary_decision": ibs_complete_for_primary,
            "n_unavailable_primary_IBS_folds": n_unavailable_ibs_folds,
            "unavailable_IBS_fold_keys": unavailable_ibs_fold_keys.to_dict(orient="records"),
            "feasible_fold_weighted_mean_IBS_B0_descriptive_only": (
                float(ibs_b0_descriptive) if np.isfinite(ibs_b0_descriptive) else None
            ),
            "feasible_fold_weighted_mean_IBS_B4_descriptive_only": (
                float(ibs_b4_descriptive) if np.isfinite(ibs_b4_descriptive) else None
            ),
            "feasible_fold_delta_IBS_descriptive_only": (
                float(delta_ibs_descriptive) if np.isfinite(delta_ibs_descriptive) else None
            ),
            "premise_support_evaluable": premise_support_evaluable,
            "IBS_feasibility_limitation": ibs_feasibility_limitation,
            "premise_state": premise_state,
        },
        "04c0_amendment_sha256": sha256_file(C40_AMENDMENT),
        "thresholds_changed_after_outcome_access": False,
        "reserved_human_outcomes_read": False,
    }
    write_json(PRIMARY_RESULT, primary_result)

    print()
    print("=" * 120)
    print("04c PRIMARY PREMISE RESULT [MATERIALIZED BEFORE SECONDARY ANALYSES]")
    print("=" * 120)
    print(f"B0 human-only Uno C: {c_index['B0']:.4f}")
    print(f"B4 DOG2+human residual Uno C: {c_index['B4']:.4f}")
    print(f"Delta Uno C (B4-B0): {delta_c:+.4f}")
    print(
        f"Paired bootstrap 95% delta-C interval: "
        f"[{delta_ci[0]:+.4f}, {delta_ci[1]:+.4f}]"
    )
    print(f"P_boot(delta C > 0): {prob_gt_zero:.4f}")
    print(f"P_boot(delta C < 0): {prob_lt_zero:.4f}")
    if ibs_complete_for_primary:
        print(f"B0 weighted IBS: {ibs_b0:.5f}")
        print(f"B4 weighted IBS: {ibs_b4:.5f}")
        print(f"Delta IBS (B4-B0): {delta_ibs:+.5f}")
    else:
        print(
            "Primary IBS: NOT FULLY EVALUABLE under frozen support rule "
            f"({n_unavailable_ibs_folds} outer fold(s) unavailable)"
        )
        if np.isfinite(delta_ibs_descriptive):
            print(
                "Feasible-fold delta IBS: "
                f"{delta_ibs_descriptive:+.5f} [DESCRIPTIVE ONLY; NON-GATING]"
            )
        print("SUPPORTS_CANINE_ADDED_VALUE disabled by conservative 04c0 amendment.")
    print(f"FROZEN PREMISE STATE: {premise_state}")
    print("=" * 120)

    # ------------------------------------------------------------------
    # Secondary B1 gene elastic-net. Fail-soft because B1 is not a premise
    # decision model, but preserve exact failure if numerical execution fails.
    # ------------------------------------------------------------------
    print()
    print("-" * 120)
    print("Secondary B1 human-only gene elastic-net")
    print("-" * 120)

    b1_status = "PASS"
    b1_c = float("nan")
    b1_error = ""

    try:
        b1_parts = {
            "predictions": [],
            "folds": [],
            "tuning": [],
        }

        for repeat in range(OUTER_REPEATS):
            cached = load_repeat_checkpoint(
                B1_CKPT,
                repeat,
                protocol_hash,
                ["predictions", "folds", "tuning"],
            )

            if cached is not None:
                print(f"  B1 repeat {repeat+1:02d}/{OUTER_REPEATS}: REUSED")
                result = cached
            else:
                print(f"  B1 repeat {repeat+1:02d}/{OUTER_REPEATS}: RUNNING")
                result = run_B1_repeat(
                    repeat,
                    X_gene,
                    y,
                    event,
                    sample_ids,
                    protocol,
                )
                save_repeat_checkpoint(
                    B1_CKPT,
                    repeat,
                    protocol_hash,
                    result,
                )

            for kind in b1_parts:
                b1_parts[kind].append(result[kind])

        b1_pred = pd.concat(
            b1_parts["predictions"],
            ignore_index=True,
        )
        b1_folds = pd.concat(
            b1_parts["folds"],
            ignore_index=True,
        )
        b1_tuning = pd.concat(
            b1_parts["tuning"],
            ignore_index=True,
        )

        b1_pred.to_csv(B1_PRED, sep="\t", index=False)
        b1_folds.to_csv(B1_FOLDS, sep="\t", index=False)
        b1_tuning.to_csv(B1_TUNING, sep="\t", index=False)

        b1_risk = aggregate_predictions(
            b1_pred,
            sample_ids,
            "B1",
        )
        b1_c = safe_uno_c(y, y, b1_risk)

        if not np.isfinite(b1_c):
            raise RuntimeError("B1 aggregated Uno C is non-finite.")

    except Exception as exc:
        b1_status = "SECONDARY_MODEL_UNAVAILABLE"
        b1_error = f"{type(exc).__name__}: {exc}"
        print(f"  B1 secondary model unavailable: {b1_error}")

    write_json(
        B1_RESULT,
        {
            "status": b1_status,
            "uno_c": (
                float(b1_c) if np.isfinite(b1_c) else None
            ),
            "error": b1_error,
            "primary_premise_decision_affected": False,
        },
    )

    if np.isfinite(b1_c):
        print(f"  B1 aggregated OOF Uno C={b1_c:.4f}")

    # ------------------------------------------------------------------
    # Secondary event-matched DOG information-efficiency analysis.
    # ------------------------------------------------------------------
    print()
    print("-" * 120)
    print("Secondary event-matched DOG information-efficiency analysis")
    print("-" * 120)

    em_parts = {
        "predictions": [],
        "draws": [],
    }

    for repeat in range(OUTER_REPEATS):
        cached = load_repeat_checkpoint(
            EVENT_MATCHED_CKPT,
            repeat,
            protocol_hash,
            ["predictions", "draws"],
        )

        if cached is not None:
            print(
                f"  event-matched repeat {repeat+1:02d}/{OUTER_REPEATS}: REUSED"
            )
            result = cached
        else:
            print(
                f"  event-matched repeat {repeat+1:02d}/{OUTER_REPEATS}: RUNNING"
            )
            result = run_event_matched_repeat(
                repeat,
                m03b,
                X_gene,
                y,
                event,
                sample_ids,
                human_module_to_indices,
                human_module_order,
                dog_data,
                source_alpha,
                protocol,
            )
            save_repeat_checkpoint(
                EVENT_MATCHED_CKPT,
                repeat,
                protocol_hash,
                result,
            )

        for kind in em_parts:
            em_parts[kind].append(result[kind])

    em_pred = pd.concat(
        em_parts["predictions"],
        ignore_index=True,
    )
    em_draws = pd.concat(
        em_parts["draws"],
        ignore_index=True,
    )

    em_pred.to_csv(EVENT_MATCHED_PRED, sep="\t", index=False)
    em_draws.to_csv(EVENT_MATCHED_DRAWS_TSV, sep="\t", index=False)

    # Consensus prediction averages over 50 DOG draws and 20 repeats.
    em_mean = em_pred.groupby("sample_id")["risk_standardized"].mean()

    if set(em_mean.index.astype(str)) != set(sample_ids.astype(str)):
        raise RuntimeError("Event-matched DOG prediction coverage mismatch.")

    em_risk = np.asarray(
        [em_mean.loc[x] for x in sample_ids],
        dtype=float,
    )
    em_c = safe_uno_c(y, y, em_risk)

    write_json(
        EVENT_MATCHED_RESULT,
        {
            "status": "PASS",
            "aggregated_consensus_uno_c": float(em_c),
            "human_B0_uno_c": float(c_index["B0"]),
            "delta_vs_B0": float(em_c - c_index["B0"]),
            "role": "SECONDARY_ONLY",
            "premise_decision_affected": False,
            "draws_per_outer_fold": EVENT_MATCHED_DRAWS_PER_OUTER_FOLD,
        },
    )

    print(f"  Event-matched DOG consensus Uno C={em_c:.4f}")
    print(f"  Delta versus B0={em_c - c_index['B0']:+.4f} [SECONDARY ONLY]")

    # ------------------------------------------------------------------
    # Final model table / result.
    # ------------------------------------------------------------------
    model_rows = [
        {
            "model": "B0",
            "description": "human-only Hallmark ridge Cox",
            "aggregated_oof_uno_c": c_index["B0"],
            "primary_premise_role": "REFERENCE",
        },
        {
            "model": "B1",
            "description": "human-only gene elastic-net Cox",
            "aggregated_oof_uno_c": (
                b1_c if np.isfinite(b1_c) else np.nan
            ),
            "primary_premise_role": "SECONDARY",
        },
        {
            "model": "B2",
            "description": "DOG2 zero-shot Hallmark ridge score",
            "aggregated_oof_uno_c": c_index["B2"],
            "primary_premise_role": "SECONDARY",
        },
        {
            "model": "B3",
            "description": "DOG2 CORAL zero-shot Hallmark score",
            "aggregated_oof_uno_c": c_index["B3"],
            "primary_premise_role": "SECONDARY",
        },
        {
            "model": "B4",
            "description": "DOG2 + human residual ridge Cox",
            "aggregated_oof_uno_c": c_index["B4"],
            "primary_premise_role": "CANINE_ADDED_VALUE_MODEL",
        },
        {
            "model": "EVENT_MATCHED_DOG",
            "description": "event-matched DOG2 zero-shot consensus",
            "aggregated_oof_uno_c": em_c,
            "primary_premise_role": "SECONDARY_INFORMATION_EFFICIENCY",
        },
    ]
    pd.DataFrame(model_rows).to_csv(
        ALL_MODEL_SUMMARY,
        sep="\t",
        index=False,
    )

    action_policy = (
        c3_policy.get("actions") or {}
    ).get(premise_state)

    if not isinstance(action_policy, dict):
        raise RuntimeError(
            f"03c premise action policy lacks state {premise_state!r}."
        )

    result = {
        "script_version": SCRIPT_VERSION,
        "status": "PASS",
        "created_utc": now_utc(),
        "scientific_status": premise_state,
        "protocol_sha256": protocol_hash,
        "04c0_amendment_sha256": sha256_file(C40_AMENDMENT),
        "postoutcome_technical_amendment": True,
        "postoutcome_amendment_can_make_result_more_favorable": False,
        "v3_implementation_note": (
            "Implementation-only fix after all 20 event-matched repeat checkpoints "
            "completed: separated the TSV output-path variable from the frozen numeric "
            "draw-count variable (50). No scientific calculation or decision rule changed."
        ),
        "target": {
            "accession": EXPECTED_TARGET,
            "n": EXPECTED_TARGET_N,
            "events": int(event.sum()),
            "censored": int((~event).sum()),
            "series_matrix_sha256": sha256_file(SERIES_MATRIX_GZ),
        },
        "source_gate_context": EXPECTED_SOURCE_GATE,
        "models": model_rows,
        "primary_premise": primary_result["primary_comparison"],
        "B1_secondary_status": b1_status,
        "event_matched_secondary": read_json(EVENT_MATCHED_RESULT),
        "frozen_action_policy": action_policy,
        "reserved_human_outcome_firewall": {
            "TARGET_OS": "CLOSED",
            "GSE21257": "CLOSED",
            "GSE39055": "CLOSED",
        },
        "AI_tuning_from_GSE16091": False,
        "samples_dropped_based_on_model_performance": False,
        "premise_thresholds_changed_after_outcome_access": False,
        "next": (
            "Apply frozen premise-state action policy; continue simulation/closed-AI "
            "method development without using GSE16091 outcomes for architecture tuning."
        ),
    }
    write_json(FINAL_RESULT, result)

    artifacts = [
        SERIES_MATRIX_GZ,
        SERIES_MATRIX_LOCK,
        INPUT_AUDIT,
        OUTCOME_AUDIT,
        DOG_SOURCE_COEFS,
        PRIMARY_PRED,
        PRIMARY_FOLDS,
        PRIMARY_TUNING,
        PRIMARY_IBS,
        PRIMARY_BOOTSTRAP,
        PRIMARY_RESULT,
        B1_RESULT,
        EVENT_MATCHED_PRED,
        EVENT_MATCHED_DRAWS_TSV,
        EVENT_MATCHED_RESULT,
        ALL_MODEL_SUMMARY,
        FINAL_RESULT,
    ]

    final_hashes = {
        path.name: sha256_file(path)
        for path in artifacts
        if path.exists()
    }

    summary = {
        "script_version": SCRIPT_VERSION,
        "status": "PASS",
        "scientific_status": premise_state,
        "v3_implementation_note": (
            "Secondary output-path namespace fix only; primary/B1/event-matched "
            "repeat checkpoints are reused only with the unchanged 04b protocol SHA256."
        ),
        "run_started_utc": started,
        "run_finished_utc": now_utc(),
        "target": EXPECTED_TARGET,
        "target_n": EXPECTED_TARGET_N,
        "target_events": int(event.sum()),
        "target_censored": int((~event).sum()),
        "B0_uno_c": float(c_index["B0"]),
        "B1_uno_c": (
            float(b1_c) if np.isfinite(b1_c) else None
        ),
        "B2_uno_c": float(c_index["B2"]),
        "B3_uno_c": float(c_index["B3"]),
        "B4_uno_c": float(c_index["B4"]),
        "delta_uno_c_B4_minus_B0": delta_c,
        "bootstrap_P_delta_c_gt_zero": prob_gt_zero,
        "bootstrap_P_delta_c_lt_zero": prob_lt_zero,
        "delta_uno_c_bootstrap_lower_95": delta_ci[0],
        "delta_uno_c_bootstrap_upper_95": delta_ci[1],
        "B0_weighted_IBS": ibs_b0,
        "B4_weighted_IBS": ibs_b4,
        "delta_IBS_B4_minus_B0": delta_ibs,
        "IBS_complete_for_primary_decision": ibs_complete_for_primary,
        "n_unavailable_primary_IBS_folds": n_unavailable_ibs_folds,
        "IBS_feasibility_limitation": ibs_feasibility_limitation,
        "feasible_fold_delta_IBS_descriptive_only": (
            float(delta_ibs_descriptive) if np.isfinite(delta_ibs_descriptive) else None
        ),
        "04c0_amendment_sha256": sha256_file(C40_AMENDMENT),
        "event_matched_DOG_uno_c": float(em_c),
        "premise_state": premise_state,
        "reserved_human_outcomes_read": False,
        "AI_architecture_tuned_from_premise_target": False,
        "final_artifact_hashes": final_hashes,
        "next": (
            "simulation/negative-transfer phase diagram and closed AI candidate "
            "development under frozen premise-state action policy"
        ),
    }
    write_json(SUMMARY_JSON, summary)

    print()
    print("=" * 120)
    print("04c CLASSICAL CANINE-ADDED-VALUE PREMISE TEST SUMMARY")
    print("=" * 120)
    print(f"GSE16091 n={EXPECTED_TARGET_N}, OS events={int(event.sum())}")
    print()
    print("Classical models:")
    print(f"  B0 human-only Hallmark ridge:       Uno C={c_index['B0']:.4f}")
    if np.isfinite(b1_c):
        print(f"  B1 human-only gene elastic-net:     Uno C={b1_c:.4f}")
    else:
        print(f"  B1 human-only gene elastic-net:     {b1_status}")
    print(f"  B2 DOG2 zero-shot:                  Uno C={c_index['B2']:.4f}")
    print(f"  B3 DOG2 CORAL zero-shot:            Uno C={c_index['B3']:.4f}")
    print(f"  B4 DOG2+human residual:             Uno C={c_index['B4']:.4f}")
    print()
    print("PRIMARY PAIRED PREMISE CONTRAST [B4-B0]:")
    print(f"  delta Uno C:                        {delta_c:+.4f}")
    print(
        f"  bootstrap 95% interval:             "
        f"[{delta_ci[0]:+.4f}, {delta_ci[1]:+.4f}]"
    )
    print(f"  P_boot(delta C > 0):                {prob_gt_zero:.4f}")
    print(f"  P_boot(delta C < 0):                {prob_lt_zero:.4f}")
    if ibs_complete_for_primary:
        print(f"  delta IBS:                          {delta_ibs:+.5f}")
    else:
        print(
            "  delta IBS:                          NOT FULLY EVALUABLE "
            f"[{n_unavailable_ibs_folds} fold(s) insufficient support]"
        )
        if np.isfinite(delta_ibs_descriptive):
            print(
                f"  feasible-fold delta IBS:            {delta_ibs_descriptive:+.5f} "
                "[DESCRIPTIVE ONLY]"
            )
    print()
    print(f"FROZEN PREMISE STATE: {premise_state}")
    if ibs_feasibility_limitation:
        print("Premise limitation: IBS_FEASIBILITY_LIMITATION")
    print()
    print("Secondary event-matched DOG:")
    print(f"  consensus Uno C:                    {em_c:.4f}")
    print(f"  delta vs human B0:                  {em_c - c_index['B0']:+.4f}")
    print()
    print("Reserved human outcomes read:")
    print("  TARGET-OS: NO")
    print("  GSE21257: NO")
    print("  GSE39055: NO")
    print("AI architecture tuned using GSE16091 outcomes: NO")
    print()
    print("Frozen next action:")
    print(json.dumps(action_policy, indent=2, ensure_ascii=False))
    print()
    print("Artifacts: results\\human_premise\\04c")
    print("=" * 120)
    print("04c frozen classical premise-test execution: PASS")
    print("=" * 120)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 120, file=sys.stderr)
        print("04c frozen classical premise-test execution: FAIL", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print("=" * 120, file=sys.stderr)
        raise
