#!/usr/bin/env python3
"""
Paper 6 - run frozen expression-level Gate Zero arm/confounding analysis.

This is the FIRST Paper-6 step that reads DOG² expression VALUES.

It executes the 02e8 pre-expression contract exactly:
  * DOG² cohort: 186 samples, COTC021=93, COTC022=93;
  * locked expression asset:
      data/processed/GSE238110_DOG2_expression_log2cpm_matched_allgenes.csv
      186 x 21,016;
  * authoritative sample->COTC mapping from 02a;
  * primary: top 5,000 label-blind variable genes;
  * sensitivities: top 2,000 and top 10,000;
  * global PCA: PC1-PC20;
  * PERMANOVA: 5,000 arm-label permutations;
  * arm prediction: repeated 5-fold x 20 L2 logistic CV;
  * classifier null: 1,000 global label permutations preserving 93/93;
  * gene-level Welch/SMD/BH diagnostics are supporting only;
  * baseline-adjusted sensitivity uses all and only the 02e7 DOG2_ARM_BALANCE
    warning covariates frozen by 02e8.

Outcomes remain CLOSED:
  * no outcome/response/follow-up values;
  * no treatment-administration values;
  * no post-baseline sample annotations;
  * no Paper-4 clinical file.

The gate asks whether randomized study identity is materially encoded in the
DOG² transcriptome. It is NOT a treatment-effect analysis.

Restart safety
--------------
Classifier-permutation null distributions are checkpointed by feature scale.
A checkpoint is reused only when the 02e8 contract hash, expression hash,
roster/mapping hashes, feature scale and requested case IDs all match.

No command-line arguments are used.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler


SCRIPT_VERSION = "02e9-run-expression-gate-zero-v1-no-cli"

ROOT = Path(__file__).resolve().parents[1]

# Frozen 02e8 contract.
E8_DIR = ROOT / "results" / "expression_gate_zero_contract" / "02e8"
E8_SUMMARY = E8_DIR / "summary.json"
E8_CONTRACT = E8_DIR / "expression_gate_zero_contract.json"
E8_ADJUSTMENT = E8_DIR / "baseline_adjustment_covariates.tsv"

# Authoritative sample/case bridge.
MAP_LOCK = ROOT / "contracts" / "02a_dog2_authoritative_id_mapping_lock.json"
MAP_FILE = ROOT / "manifests" / "02a_dog2_selected186_authoritative_id_mapping.csv"

# Frozen DOG² roster and baseline materialization.
E7_ROSTER = ROOT / "results" / "dog2_selection_audit" / "02e7" / "dog2_rna186_roster.tsv"
E7_AUDIT = ROOT / "results" / "dog2_selection_audit" / "02e7" / "selection_audit.json"
E6_JSONL = ROOT / "results" / "icdc_baseline_batch" / "02e6" / "icdc_baseline_309.jsonl"

OUT_DIR = ROOT / "results" / "expression_gate_zero" / "02e9"
CHECKPOINT_DIR = OUT_DIR / "checkpoints"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_SCALE_TSV = OUT_DIR / "feature_scale_results.tsv"
PC_TSV = OUT_DIR / "pc_diagnostics.tsv"
GENE_TSV = OUT_DIR / "gene_level_supporting_diagnostics.tsv"
CV_FOLD_TSV = OUT_DIR / "classifier_observed_fold_results.tsv"
ADJUSTED_JSON = OUT_DIR / "baseline_adjusted_sensitivity.json"
RESULTS_JSON = OUT_DIR / "gate_zero_results.json"
SUMMARY_JSON = OUT_DIR / "summary.json"

EXPECTED_N = 186
EXPECTED_ARMS = {"COTC021": 93, "COTC022": 93}

# Small dataset / low-dimensional model: CPU is scientifically appropriate.
# GPU overhead would dominate this workload.
LOGISTIC_SOLVER = "liblinear"
LOGISTIC_MAX_ITER = 5000

# Checkpoint every N completed classifier permutations.
PERMUTATION_CHECKPOINT_EVERY = 50


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def clean(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if s.lower() in {"", "nan", "none", "null", "na", "n/a"}:
        return ""
    return s


def lookup_hash(mapping: Dict[str, Any], path: Path) -> str:
    key = rel(path)
    value = mapping.get(key)
    if value is None:
        value = mapping.get(key.replace("/", "\\"))
    if isinstance(value, dict):
        value = value.get("sha256")
    return str(value or "")


@dataclass(frozen=True)
class FeatureScale:
    name: str
    top_n: int
    role: str


@dataclass
class FoldCache:
    fold_id: int
    repeat_id: int
    split_id: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    X_train_pc: np.ndarray
    X_test_pc: np.ndarray


def verify_e8() -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    for path in (E8_SUMMARY, E8_CONTRACT, E8_ADJUSTMENT):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required 02e8 artifact: {path}\n"
                "Run scripts\\02e8_freeze_expression_gate_zero_contract.py first."
            )

    summary = read_json(E8_SUMMARY)
    if summary.get("status") != "PASS":
        raise RuntimeError("02e8 summary status is not PASS.")

    for key in (
        "network_access",
        "dog2_expression_values_read",
        "omics_values_read",
        "outcome_response_followup_values_read",
        "treatment_administration_values_read",
        "postbaseline_sample_annotations_read",
    ):
        if summary.get(key) is not False:
            raise RuntimeError(f"02e8 provenance mismatch: {key}")

    expected_contract_sha = clean(summary.get("contract_sha256"))
    actual_contract_sha = sha256_file(E8_CONTRACT)
    if (
        not expected_contract_sha
        or expected_contract_sha.lower() != actual_contract_sha.lower()
    ):
        raise RuntimeError(
            "02e8 contract hash mismatch: "
            f"expected={expected_contract_sha}, actual={actual_contract_sha}"
        )

    expected_adjust_sha = clean(summary.get("adjustment_sha256"))
    actual_adjust_sha = sha256_file(E8_ADJUSTMENT)
    if (
        not expected_adjust_sha
        or expected_adjust_sha.lower() != actual_adjust_sha.lower()
    ):
        raise RuntimeError("02e8 adjustment-covariate hash mismatch.")

    contract = read_json(E8_CONTRACT)
    if contract.get("status") != "PASS":
        raise RuntimeError("02e8 contract status is not PASS.")

    hard = contract.get("hard_guardrails") or {}
    required_true = (
        "no_outcome_access",
        "no_response_access",
        "no_follow_up_access",
        "no_treatment_administration_access",
        "no_postbaseline_sample_annotation_access",
        "no_paper4_clinical_file_read",
        "no_arm_supervised_feature_selection",
        "no_row_order_matching",
        "do_not_drop_samples_based_on_gate_results",
        "do_not_change_thresholds_after_expression_opened",
    )
    bad = [key for key in required_true if hard.get(key) is not True]
    if bad:
        raise RuntimeError(
            "02e8 hard guardrails missing/not TRUE: " + ", ".join(bad)
        )

    cohort = contract.get("cohort") or {}
    if cohort.get("n") != EXPECTED_N:
        raise RuntimeError("02e8 cohort n changed.")
    if cohort.get("arm_counts") != EXPECTED_ARMS:
        raise RuntimeError("02e8 arm counts changed.")

    return summary, contract, actual_contract_sha


def load_mapping() -> pd.DataFrame:
    for path in (MAP_LOCK, MAP_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Missing 02a mapping artifact: {path}")

    lock = read_json(MAP_LOCK)
    if lock.get("status") != "PASS":
        raise RuntimeError("02a mapping lock is not PASS.")

    expected = clean(lock.get("output_mapping_sha256"))
    actual = sha256_file(MAP_FILE)
    if not expected or expected.lower() != actual.lower():
        raise RuntimeError("02a mapping hash mismatch.")

    df = pd.read_csv(MAP_FILE, dtype=str)
    required = {"paper4_sample_id", "paper4_patient_id", "cotc_subject_id"}
    if not required.issubset(df.columns):
        raise RuntimeError("02a mapping lacks required columns.")

    df = df[list(required)].copy()
    for col in required:
        df[col] = df[col].fillna("").astype(str).str.strip()

    if len(df) != EXPECTED_N:
        raise RuntimeError(f"02a mapping rows={len(df)}, expected 186.")
    if df["paper4_sample_id"].eq("").any():
        raise RuntimeError("02a has empty paper4_sample_id.")
    if df["cotc_subject_id"].eq("").any():
        raise RuntimeError("02a has empty cotc_subject_id.")
    if df["paper4_sample_id"].nunique() != EXPECTED_N:
        raise RuntimeError("02a paper4_sample_id is not 186 unique.")
    if df["cotc_subject_id"].nunique() != EXPECTED_N:
        raise RuntimeError("02a cotc_subject_id is not 186 unique.")

    return df


def load_roster() -> pd.DataFrame:
    if not E7_ROSTER.exists():
        raise FileNotFoundError(f"Missing 02e7 roster: {E7_ROSTER}")

    df = pd.read_csv(E7_ROSTER, sep="\t", dtype=str).fillna("")
    required = {"case_id", "study", "gate_zero_arm", "patient_id"}
    if not required.issubset(df.columns):
        raise RuntimeError("02e7 roster lacks required columns.")

    if len(df) != EXPECTED_N:
        raise RuntimeError("02e7 roster rows != 186.")
    if df["case_id"].nunique() != EXPECTED_N:
        raise RuntimeError("02e7 case_id is not 186 unique.")

    counts = df["study"].value_counts().to_dict()
    observed = {
        "COTC021": int(counts.get("COTC021", 0)),
        "COTC022": int(counts.get("COTC022", 0)),
    }
    if observed != EXPECTED_ARMS:
        raise RuntimeError(f"02e7 roster arm counts changed: {observed}")

    return df


def expression_path_from_contract(contract: Dict[str, Any]) -> Path:
    asset = contract.get("expression_asset") or {}
    path_text = clean(asset.get("expression_relative_path"))
    if not path_text:
        raise RuntimeError("02e8 contract lacks expression path.")

    path = ROOT / Path(path_text)
    if not path.exists():
        raise FileNotFoundError(
            f"Locked DOG² expression file does not exist: {path}"
        )

    expected_sha = clean(asset.get("expression_sha256"))
    actual_sha = sha256_file(path)
    if not expected_sha or expected_sha.lower() != actual_sha.lower():
        raise RuntimeError(
            "DOG² expression SHA256 mismatch: "
            f"expected={expected_sha}, actual={actual_sha}"
        )
    return path


def load_expression(
    path: Path,
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    print("Opening locked DOG² expression values...")

    df = pd.read_csv(path, index_col=0)

    if df.shape[0] != EXPECTED_N:
        raise RuntimeError(
            f"Expression row count={df.shape[0]}, expected 186."
        )

    expected_features = int(
        (read_json(E8_CONTRACT).get("expression_asset") or {}).get(
            "expression_features", 0
        )
    )
    if expected_features and df.shape[1] != expected_features:
        raise RuntimeError(
            f"Expression feature count={df.shape[1]}, expected {expected_features}."
        )

    df.index = df.index.astype(str).str.strip()
    if df.index.has_duplicates:
        dupes = df.index[df.index.duplicated()].tolist()[:10]
        raise RuntimeError(f"Expression sample IDs are duplicated: {dupes}")

    expected_sample_ids = set(mapping["paper4_sample_id"])
    observed_sample_ids = set(df.index)
    if observed_sample_ids != expected_sample_ids:
        missing = sorted(expected_sample_ids - observed_sample_ids)[:20]
        extra = sorted(observed_sample_ids - expected_sample_ids)[:20]
        raise RuntimeError(
            "Expression index != authoritative 02a sample-ID set. "
            f"missing={missing}, extra={extra}"
        )

    # Numeric fail-closed coercion.
    numeric = df.apply(pd.to_numeric, errors="coerce")
    arr = numeric.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(arr).all():
        bad = np.argwhere(~np.isfinite(arr))
        preview = [
            {
                "sample_id": str(numeric.index[i]),
                "gene": str(numeric.columns[j]),
                "value": df.iloc[i, j],
            }
            for i, j in bad[:10]
        ]
        raise RuntimeError(
            "Expression contains non-finite/non-numeric values after coercion. "
            f"Examples: {preview}"
        )

    if numeric.columns.duplicated().any():
        duplicated = numeric.columns[numeric.columns.duplicated()].tolist()[:20]
        raise RuntimeError(f"Expression gene columns duplicated: {duplicated}")

    return numeric


def align_analysis_table(
    expression: pd.DataFrame,
    mapping: pd.DataFrame,
    roster: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray]:
    map_by_sample = mapping.set_index("paper4_sample_id")
    roster_by_case = roster.set_index("case_id")

    rows = []
    labels = []

    for sample_id in expression.index:
        if sample_id not in map_by_sample.index:
            raise RuntimeError(f"Expression sample missing from 02a: {sample_id}")
        case_id = clean(map_by_sample.loc[sample_id, "cotc_subject_id"])
        if case_id not in roster_by_case.index:
            raise RuntimeError(f"Mapped case missing from 02e7 roster: {case_id}")

        study = clean(roster_by_case.loc[case_id, "study"])
        expected_prefix = case_id.split("-", 1)[0]
        if study != expected_prefix:
            raise RuntimeError(
                f"{case_id}: study {study!r} != case prefix {expected_prefix!r}"
            )

        arm = clean(roster_by_case.loc[case_id, "gate_zero_arm"])
        expected_arm = (
            "SOC_PLUS_RAPAMYCIN" if study == "COTC021" else "SOC_CONTROL"
        )
        if arm != expected_arm:
            raise RuntimeError(f"{case_id}: frozen arm mapping mismatch.")

        rows.append(
            {
                "paper4_sample_id": sample_id,
                "case_id": case_id,
                "study": study,
                "gate_zero_arm": arm,
            }
        )
        labels.append(1 if study == "COTC021" else 0)

    meta = pd.DataFrame(rows).set_index("paper4_sample_id")
    y = np.asarray(labels, dtype=np.int8)

    if int(y.sum()) != 93 or int((1 - y).sum()) != 93:
        raise RuntimeError(
            f"Aligned labels are not 93/93: COTC021={int(y.sum())}, "
            f"COTC022={int((1-y).sum())}"
        )

    return meta, y


def read_adjustment_covariate_names() -> List[Dict[str, str]]:
    df = pd.read_csv(E8_ADJUSTMENT, sep="\t", dtype=str).fillna("")
    rows = df.to_dict("records")
    return rows


def verify_e6_hash_from_e7() -> str:
    if not E7_AUDIT.exists():
        raise FileNotFoundError(f"Missing 02e7 selection audit: {E7_AUDIT}")
    if not E6_JSONL.exists():
        raise FileNotFoundError(f"Missing 02e6 JSONL: {E6_JSONL}")

    audit = read_json(E7_AUDIT)
    hashes = audit.get("upstream_02e6_hashes") or {}
    expected = lookup_hash(hashes, E6_JSONL)
    actual = sha256_file(E6_JSONL)
    if not expected or expected.lower() != actual.lower():
        raise RuntimeError("02e6 baseline JSONL hash not verified by 02e7.")
    return actual


def load_baseline_covariates(
    meta: pd.DataFrame,
    covariate_specs: List[Dict[str, str]],
) -> pd.DataFrame:
    if not covariate_specs:
        return pd.DataFrame(index=meta.index)

    verify_e6_hash_from_e7()

    needed = {clean(row.get("variable")) for row in covariate_specs}
    supported = {"registering_institution", "veterinary_medical_center"}
    unsupported = sorted(needed - supported)
    if unsupported:
        raise RuntimeError(
            "02e9 currently lacks a frozen-baseline extractor for covariate(s): "
            + ", ".join(unsupported)
        )

    cases_needed = set(meta["case_id"])
    case_to_values: Dict[str, Dict[str, str]] = {}

    with E6_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            case_id = clean(obj.get("case_id"))
            if case_id not in cases_needed:
                continue

            case = obj.get("case")
            if not isinstance(case, dict):
                raise RuntimeError(f"{case_id}: malformed 02e6 case object.")
            enrollment = case.get("enrollment")
            if not isinstance(enrollment, dict):
                raise RuntimeError(f"{case_id}: missing frozen enrollment.")

            case_to_values[case_id] = {
                "registering_institution": clean(
                    enrollment.get("registering_institution")
                ),
                "veterinary_medical_center": clean(
                    enrollment.get("veterinary_medical_center")
                ),
            }

    if set(case_to_values) != cases_needed:
        missing = sorted(cases_needed - set(case_to_values))
        raise RuntimeError(
            f"Missing frozen baseline values for DOG² cases: {missing[:20]}"
        )

    rows = []
    for sample_id, row in meta.iterrows():
        case_id = row["case_id"]
        values = case_to_values[case_id]
        out = {"paper4_sample_id": sample_id}
        for spec in covariate_specs:
            variable = clean(spec["variable"])
            value = values[variable]
            out[variable] = value if value else "<MISSING>"
        rows.append(out)

    return pd.DataFrame(rows).set_index("paper4_sample_id")


def one_hot_design(
    covariates: pd.DataFrame,
    frozen_levels: Optional[Dict[str, List[str]]] = None,
) -> Tuple[np.ndarray, Dict[str, List[str]]]:
    """
    Intercept + reference-coded one-hot categorical design.
    All 02e8-selected covariates in this run are categorical.
    """
    n = len(covariates)
    columns = [np.ones((n, 1), dtype=np.float64)]
    levels_out: Dict[str, List[str]] = {}

    for variable in covariates.columns:
        values = covariates[variable].fillna("<MISSING>").astype(str).to_numpy()

        if frozen_levels is None:
            levels = sorted(set(values.tolist()))
        else:
            levels = list(frozen_levels[variable])

        if not levels:
            raise RuntimeError(f"No levels for baseline covariate {variable}")

        levels_out[variable] = levels

        # Reference coding: first sorted level is reference.
        for level in levels[1:]:
            col = (values == level).astype(np.float64).reshape(-1, 1)
            columns.append(col)

    design = np.hstack(columns)
    return design, levels_out


def sample_variance(X: np.ndarray) -> np.ndarray:
    return np.var(X, axis=0, ddof=1)


def rank_variable_genes(
    X: np.ndarray,
    gene_names: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    variances = sample_variance(X)
    finite = np.isfinite(variances)
    nonzero = finite & (variances > 0)

    if not np.any(nonzero):
        raise RuntimeError("No nonzero-variance genes remain.")

    indices = np.flatnonzero(nonzero)
    # Deterministic tie-breaking by gene name after descending variance.
    order = sorted(
        indices.tolist(),
        key=lambda j: (-float(variances[j]), str(gene_names[j])),
    )
    ranked = np.asarray(order, dtype=np.int64)
    return ranked, variances, nonzero


def select_top(ranked: np.ndarray, top_n: int) -> np.ndarray:
    if len(ranked) < top_n:
        raise RuntimeError(
            f"Only {len(ranked)} nonzero genes available, requested top {top_n}."
        )
    return ranked[:top_n]


def global_pca(
    X: np.ndarray,
    selected_idx: np.ndarray,
    n_pcs: int,
) -> Tuple[np.ndarray, np.ndarray]:
    Xs = X[:, selected_idx]
    scaler = StandardScaler(with_mean=True, with_std=True)
    Z = scaler.fit_transform(Xs)

    if not np.isfinite(Z).all():
        raise RuntimeError("Non-finite values after global gene z-scoring.")

    pca = PCA(n_components=n_pcs, svd_solver="full")
    scores = pca.fit_transform(Z)
    return scores, pca.explained_variance_ratio_


def residualized_global_pca(
    X: np.ndarray,
    selected_idx: np.ndarray,
    covariate_design: np.ndarray,
    n_pcs: int,
) -> Tuple[np.ndarray, np.ndarray]:
    Xs = X[:, selected_idx]
    scaler = StandardScaler(with_mean=True, with_std=True)
    Z = scaler.fit_transform(Xs)

    beta = np.linalg.pinv(covariate_design) @ Z
    residual = Z - covariate_design @ beta

    pca = PCA(n_components=n_pcs, svd_solver="full")
    scores = pca.fit_transform(residual)
    return scores, pca.explained_variance_ratio_


def permanova_stat(scores: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Euclidean one-factor PERMANOVA in coordinate space.
    For Euclidean distances, between/total sums of squares in centered
    coordinate space are equivalent to the standard distance-based statistic.
    """
    if scores.ndim != 2 or len(scores) != len(y):
        raise RuntimeError("PERMANOVA input shape mismatch.")

    n = len(y)
    groups = np.unique(y)
    k = len(groups)
    if k != 2:
        raise RuntimeError("Gate Zero expects exactly two groups.")

    grand = scores.mean(axis=0)
    total_ss = float(np.sum((scores - grand) ** 2))
    if total_ss <= 0:
        raise RuntimeError("PERMANOVA total sum of squares is zero.")

    between_ss = 0.0
    for group in groups:
        block = scores[y == group]
        center = block.mean(axis=0)
        between_ss += len(block) * float(np.sum((center - grand) ** 2))

    within_ss = total_ss - between_ss
    r2 = between_ss / total_ss

    df_between = k - 1
    df_within = n - k
    if within_ss <= 0:
        pseudo_f = float("inf")
    else:
        pseudo_f = (between_ss / df_between) / (within_ss / df_within)

    return float(pseudo_f), float(r2)


def permanova_permutation(
    scores: np.ndarray,
    y: np.ndarray,
    n_perm: int,
    seed: int,
) -> Dict[str, Any]:
    observed_f, observed_r2 = permanova_stat(scores, y)
    rng = np.random.default_rng(seed)

    null_r2 = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        yp = rng.permutation(y)
        _, null_r2[i] = permanova_stat(scores, yp)

    p = (1.0 + float(np.sum(null_r2 >= observed_r2))) / (n_perm + 1.0)

    return {
        "pseudo_F": observed_f,
        "R2_arm": observed_r2,
        "permutation_p": float(p),
        "null_R2_mean": float(np.mean(null_r2)),
        "null_R2_q95": float(np.quantile(null_r2, 0.95)),
        "permutations": int(n_perm),
        "seed": int(seed),
    }


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=np.float64)
    out = np.full_like(p, np.nan)

    finite = np.isfinite(p)
    if not finite.any():
        return out

    vals = p[finite]
    m = len(vals)
    order = np.argsort(vals)
    ranked = vals[order]

    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    restored = np.empty(m, dtype=np.float64)
    restored[order] = adjusted
    out[finite] = restored
    return out


def smd(a: np.ndarray, b: np.ndarray) -> float:
    ma = float(np.mean(a))
    mb = float(np.mean(b))
    va = float(np.var(a, ddof=1))
    vb = float(np.var(b, ddof=1))
    pooled = math.sqrt(max(0.0, (va + vb) / 2.0))
    if pooled == 0:
        return 0.0 if ma == mb else float("nan")
    return (ma - mb) / pooled


def pc_diagnostics(
    scores: np.ndarray,
    y: np.ndarray,
    scale_name: str,
) -> List[Dict[str, Any]]:
    rows = []
    pvals = []

    g1 = y == 1
    g0 = y == 0

    for pc in range(scores.shape[1]):
        a = scores[g1, pc]
        b = scores[g0, pc]

        test = stats.ttest_ind(
            a,
            b,
            equal_var=False,
            nan_policy="raise",
        )

        effect = smd(a, b)
        p = float(test.pvalue)
        pvals.append(p)

        rows.append(
            {
                "feature_scale": scale_name,
                "PC": pc + 1,
                "mean_COTC021": float(np.mean(a)),
                "mean_COTC022": float(np.mean(b)),
                "mean_difference_COTC021_minus_COTC022": (
                    float(np.mean(a) - np.mean(b))
                ),
                "SMD_COTC021_minus_COTC022": effect,
                "abs_SMD": abs(effect) if math.isfinite(effect) else None,
                "welch_t": float(test.statistic),
                "welch_p": p,
                "BH_q": None,
            }
        )

    q = bh_adjust(np.asarray(pvals))
    for row, qv in zip(rows, q):
        row["BH_q"] = float(qv) if np.isfinite(qv) else None

    return rows


def prepare_fold_caches(
    X: np.ndarray,
    y: np.ndarray,
    top_n: int,
    n_pcs: int,
    n_splits: int,
    n_repeats: int,
    seed: int,
) -> List[FoldCache]:
    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=seed,
    )

    caches: List[FoldCache] = []

    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        Xtr = X[train_idx]
        Xte = X[test_idx]

        variances = np.var(Xtr, axis=0, ddof=1)
        finite_nonzero = np.isfinite(variances) & (variances > 0)
        candidate = np.flatnonzero(finite_nonzero)
        if len(candidate) < top_n:
            raise RuntimeError(
                f"CV fold {fold_id}: only {len(candidate)} variable genes "
                f"for requested top {top_n}."
            )

        # Deterministic variance-only ranking.
        order = candidate[np.argsort(-variances[candidate], kind="stable")]
        selected = order[:top_n]

        scaler = StandardScaler(with_mean=True, with_std=True)
        Ztr = scaler.fit_transform(Xtr[:, selected])
        Zte = scaler.transform(Xte[:, selected])

        pca = PCA(n_components=n_pcs, svd_solver="full")
        Ptr = pca.fit_transform(Ztr)
        Pte = pca.transform(Zte)

        repeat_id = fold_id // n_splits
        split_id = fold_id % n_splits

        caches.append(
            FoldCache(
                fold_id=fold_id,
                repeat_id=repeat_id,
                split_id=split_id,
                train_idx=train_idx.astype(np.int64),
                test_idx=test_idx.astype(np.int64),
                X_train_pc=Ptr.astype(np.float64),
                X_test_pc=Pte.astype(np.float64),
            )
        )

    return caches


def fit_fold_metric(
    cache: FoldCache,
    y: np.ndarray,
) -> Tuple[float, float]:
    ytr = y[cache.train_idx]
    yte = y[cache.test_idx]

    if len(np.unique(ytr)) != 2 or len(np.unique(yte)) != 2:
        raise RuntimeError(
            f"Fold {cache.fold_id}: arm label lacks both classes."
        )

    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver=LOGISTIC_SOLVER,
        max_iter=LOGISTIC_MAX_ITER,
        random_state=0,
    )
    model.fit(cache.X_train_pc, ytr)
    prob = model.predict_proba(cache.X_test_pc)[:, 1]
    pred = model.predict(cache.X_test_pc)

    auc = roc_auc_score(yte, prob)
    bacc = balanced_accuracy_score(yte, pred)
    return float(auc), float(bacc)


def observed_cv(
    caches: Sequence[FoldCache],
    y: np.ndarray,
    scale_name: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    rows = []
    aucs = []
    baccs = []

    for cache in caches:
        auc, bacc = fit_fold_metric(cache, y)
        aucs.append(auc)
        baccs.append(bacc)
        rows.append(
            {
                "feature_scale": scale_name,
                "fold_id": cache.fold_id,
                "repeat_id": cache.repeat_id,
                "split_id": cache.split_id,
                "n_train": len(cache.train_idx),
                "n_test": len(cache.test_idx),
                "ROC_AUC": auc,
                "balanced_accuracy": bacc,
            }
        )

    summary = {
        "mean_ROC_AUC": float(np.mean(aucs)),
        "sd_ROC_AUC": float(np.std(aucs, ddof=1)),
        "mean_balanced_accuracy": float(np.mean(baccs)),
        "sd_balanced_accuracy": float(np.std(baccs, ddof=1)),
        "n_fold_evaluations": len(aucs),
    }
    return summary, rows


def classifier_checkpoint_path(scale_name: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in scale_name)
    return CHECKPOINT_DIR / f"classifier_permutations_{safe}.json"


def load_classifier_checkpoint(
    path: Path,
    contract_sha: str,
    expression_sha: str,
    scale_name: str,
    top_n: int,
    n_perm: int,
    y: np.ndarray,
) -> List[Dict[str, float]]:
    if not path.exists():
        return []

    try:
        obj = read_json(path)
    except Exception:
        return []

    expected = {
        "script_version": SCRIPT_VERSION,
        "02e8_contract_sha256": contract_sha,
        "expression_sha256": expression_sha,
        "feature_scale": scale_name,
        "top_variable_genes": top_n,
        "requested_permutations": n_perm,
        "label_sum": int(np.sum(y)),
        "n_samples": len(y),
    }
    for key, value in expected.items():
        if obj.get(key) != value:
            return []

    completed = obj.get("null_metrics")
    if not isinstance(completed, list):
        return []

    cleaned = []
    for row in completed:
        if not isinstance(row, dict):
            return []
        auc = row.get("mean_ROC_AUC")
        bacc = row.get("mean_balanced_accuracy")
        if not isinstance(auc, (int, float)) or not isinstance(bacc, (int, float)):
            return []
        cleaned.append(
            {
                "mean_ROC_AUC": float(auc),
                "mean_balanced_accuracy": float(bacc),
            }
        )

    if len(cleaned) > n_perm:
        return []

    return cleaned


def save_classifier_checkpoint(
    path: Path,
    contract_sha: str,
    expression_sha: str,
    scale_name: str,
    top_n: int,
    n_perm: int,
    y: np.ndarray,
    null_metrics: List[Dict[str, float]],
) -> None:
    write_json(
        path,
        {
            "script_version": SCRIPT_VERSION,
            "updated_utc": now_utc(),
            "02e8_contract_sha256": contract_sha,
            "expression_sha256": expression_sha,
            "feature_scale": scale_name,
            "top_variable_genes": top_n,
            "requested_permutations": n_perm,
            "label_sum": int(np.sum(y)),
            "n_samples": len(y),
            "completed_permutations": len(null_metrics),
            "null_metrics": null_metrics,
        },
    )


def classifier_permutation_test(
    caches: Sequence[FoldCache],
    y: np.ndarray,
    observed: Dict[str, Any],
    n_perm: int,
    seed: int,
    scale_name: str,
    top_n: int,
    contract_sha: str,
    expression_sha: str,
) -> Dict[str, Any]:
    """
    Fixed repeated-stratified folds and cached label-blind train-fold
    preprocessing are reused under globally permuted arm labels.
    """
    checkpoint = classifier_checkpoint_path(scale_name)
    null_metrics = load_classifier_checkpoint(
        checkpoint,
        contract_sha,
        expression_sha,
        scale_name,
        top_n,
        n_perm,
        y,
    )

    start = len(null_metrics)
    if start:
        print(
            f"    classifier null: reusing {start}/{n_perm} "
            f"permutations from checkpoint"
        )

    # Deterministic sequence regardless of restart: generate each permutation
    # from SeedSequence([seed, permutation_index]).
    for perm_index in range(start, n_perm):
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, perm_index])
        )
        yp = rng.permutation(y)

        aucs = []
        baccs = []

        for cache in caches:
            # A globally shuffled balanced label vector will almost certainly
            # retain both classes in every ~37-sample test fold. Fail closed if
            # an exceptional single-class fold occurs.
            ytr = yp[cache.train_idx]
            yte = yp[cache.test_idx]
            if len(np.unique(ytr)) != 2 or len(np.unique(yte)) != 2:
                raise RuntimeError(
                    f"{scale_name} permutation {perm_index}: single-class fold."
                )

            model = LogisticRegression(
                penalty="l2",
                C=1.0,
                solver=LOGISTIC_SOLVER,
                max_iter=LOGISTIC_MAX_ITER,
                random_state=0,
            )
            model.fit(cache.X_train_pc, ytr)
            prob = model.predict_proba(cache.X_test_pc)[:, 1]
            pred = model.predict(cache.X_test_pc)

            aucs.append(float(roc_auc_score(yte, prob)))
            baccs.append(float(balanced_accuracy_score(yte, pred)))

        null_metrics.append(
            {
                "mean_ROC_AUC": float(np.mean(aucs)),
                "mean_balanced_accuracy": float(np.mean(baccs)),
            }
        )

        completed = perm_index + 1
        if (
            completed % PERMUTATION_CHECKPOINT_EVERY == 0
            or completed == n_perm
        ):
            save_classifier_checkpoint(
                checkpoint,
                contract_sha,
                expression_sha,
                scale_name,
                top_n,
                n_perm,
                y,
                null_metrics,
            )
            print(
                f"    classifier null permutations: "
                f"{completed}/{n_perm}"
            )

    null_auc = np.asarray(
        [row["mean_ROC_AUC"] for row in null_metrics], dtype=np.float64
    )
    null_bacc = np.asarray(
        [row["mean_balanced_accuracy"] for row in null_metrics],
        dtype=np.float64,
    )

    observed_auc = float(observed["mean_ROC_AUC"])
    observed_bacc = float(observed["mean_balanced_accuracy"])

    auc_p = (1.0 + float(np.sum(null_auc >= observed_auc))) / (n_perm + 1.0)
    bacc_p = (1.0 + float(np.sum(null_bacc >= observed_bacc))) / (n_perm + 1.0)

    return {
        "mean_ROC_AUC": observed_auc,
        "sd_ROC_AUC": float(observed["sd_ROC_AUC"]),
        "ROC_AUC_permutation_p": float(auc_p),
        "ROC_AUC_null_mean": float(np.mean(null_auc)),
        "ROC_AUC_null_q95": float(np.quantile(null_auc, 0.95)),
        "mean_balanced_accuracy": observed_bacc,
        "sd_balanced_accuracy": float(observed["sd_balanced_accuracy"]),
        "balanced_accuracy_permutation_p": float(bacc_p),
        "balanced_accuracy_null_mean": float(np.mean(null_bacc)),
        "balanced_accuracy_null_q95": float(np.quantile(null_bacc, 0.95)),
        "classifier_permutations": n_perm,
        "checkpoint_path": rel(checkpoint),
    }


def build_adjusted_fold_caches(
    X: np.ndarray,
    y: np.ndarray,
    covariates: pd.DataFrame,
    top_n: int,
    n_pcs: int,
    n_splits: int,
    n_repeats: int,
    seed: int,
) -> List[FoldCache]:
    """
    Baseline-adjusted sensitivity, primary top-N only.

    Per fold:
      1. rank genes by training-fold variance only;
      2. z-score selected genes on training fold;
      3. regress each selected standardized gene on frozen baseline covariates
         using training-fold coefficients;
      4. residualize train and test using those coefficients;
      5. fit PCA20 on residualized train;
      6. transform residualized test.

    Arm labels are not used in feature selection/residualization/PCA.
    """
    _, frozen_levels = one_hot_design(covariates)

    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=seed,
    )

    caches = []

    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        Xtr = X[train_idx]
        Xte = X[test_idx]

        variances = np.var(Xtr, axis=0, ddof=1)
        candidate = np.flatnonzero(np.isfinite(variances) & (variances > 0))
        if len(candidate) < top_n:
            raise RuntimeError(
                f"Adjusted CV fold {fold_id}: insufficient variable genes."
            )
        order = candidate[np.argsort(-variances[candidate], kind="stable")]
        selected = order[:top_n]

        scaler = StandardScaler()
        Ztr = scaler.fit_transform(Xtr[:, selected])
        Zte = scaler.transform(Xte[:, selected])

        Ctr, _ = one_hot_design(
            covariates.iloc[train_idx],
            frozen_levels=frozen_levels,
        )
        Cte, _ = one_hot_design(
            covariates.iloc[test_idx],
            frozen_levels=frozen_levels,
        )

        beta = np.linalg.pinv(Ctr) @ Ztr
        Rtr = Ztr - Ctr @ beta
        Rte = Zte - Cte @ beta

        pca = PCA(n_components=n_pcs, svd_solver="full")
        Ptr = pca.fit_transform(Rtr)
        Pte = pca.transform(Rte)

        caches.append(
            FoldCache(
                fold_id=fold_id,
                repeat_id=fold_id // n_splits,
                split_id=fold_id % n_splits,
                train_idx=train_idx.astype(np.int64),
                test_idx=test_idx.astype(np.int64),
                X_train_pc=Ptr,
                X_test_pc=Pte,
            )
        )

    return caches


def gene_level_diagnostics(
    X: np.ndarray,
    y: np.ndarray,
    gene_names: np.ndarray,
    nonzero_mask: np.ndarray,
    q_threshold: float,
    smd_threshold: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    idx = np.flatnonzero(nonzero_mask)
    Xv = X[:, idx]
    names = gene_names[idx]

    a = Xv[y == 1]
    b = Xv[y == 0]

    t_stat, p_val = stats.ttest_ind(
        a,
        b,
        axis=0,
        equal_var=False,
        nan_policy="raise",
    )

    mean_a = np.mean(a, axis=0)
    mean_b = np.mean(b, axis=0)
    var_a = np.var(a, axis=0, ddof=1)
    var_b = np.var(b, axis=0, ddof=1)
    pooled = np.sqrt((var_a + var_b) / 2.0)

    effect = np.zeros_like(mean_a, dtype=np.float64)
    nonzero_pool = pooled > 0
    effect[nonzero_pool] = (
        mean_a[nonzero_pool] - mean_b[nonzero_pool]
    ) / pooled[nonzero_pool]
    effect[~nonzero_pool] = 0.0

    q = bh_adjust(np.asarray(p_val, dtype=np.float64))
    strong = (
        (q < q_threshold)
        & (np.abs(effect) >= smd_threshold)
    )

    df = pd.DataFrame(
        {
            "gene": names.astype(str),
            "mean_COTC021": mean_a,
            "mean_COTC022": mean_b,
            "mean_difference_COTC021_minus_COTC022": mean_a - mean_b,
            "SMD_COTC021_minus_COTC022": effect,
            "abs_SMD": np.abs(effect),
            "welch_t": t_stat,
            "welch_p": p_val,
            "BH_q": q,
            "strong_signal": strong,
        }
    )

    df = df.sort_values(
        ["BH_q", "abs_SMD", "gene"],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    summary = {
        "genes_tested": int(len(df)),
        "strong_genes": int(np.sum(strong)),
        "strong_gene_fraction": float(np.mean(strong)),
        "q_threshold": q_threshold,
        "abs_SMD_threshold": smd_threshold,
    }
    return df, summary


def primary_gate(
    r2: float,
    auc: float,
    thresholds: Dict[str, Any],
) -> str:
    green = thresholds["GREEN"]
    red = thresholds["RED"]

    red_r2 = float(red["primary_R2_gte"])
    red_auc = float(red["OR_primary_CV_AUC_gte"])
    green_r2 = float(green["primary_R2_lt"])
    green_auc = float(green["AND_primary_CV_AUC_lt"])

    if r2 >= red_r2 or auc >= red_auc:
        return "RED"
    if r2 < green_r2 and auc < green_auc:
        return "GREEN"
    return "AMBER"


def write_tsv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty TSV: {path}")

    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    print("=" * 112)
    print("Paper 6 - run frozen expression-level Gate Zero arm/confounding analysis")
    print("=" * 112)
    print(f"Script version: {SCRIPT_VERSION}")
    print()
    print("Safety / execution contract:")
    print("  DOG² expression values read in 02e9: YES")
    print("  Outcome/response/follow-up values read: NO")
    print("  Treatment-administration values read: NO")
    print("  Post-baseline sample annotations read: NO")
    print("  Paper-4 clinical file read: NO")
    print("  Samples dropped based on expression results: NO")
    print("  GPU execution: NO [186 samples; low-dimensional inference is CPU-appropriate]")
    print()

    started = now_utc()

    e8_summary, contract, contract_sha = verify_e8()
    mapping = load_mapping()
    roster = load_roster()

    if set(mapping["cotc_subject_id"]) != set(roster["case_id"]):
        raise RuntimeError("02a mapped COTC set != 02e7 DOG² roster.")

    expr_path = expression_path_from_contract(contract)
    expression_sha = sha256_file(expr_path)

    print("Frozen upstream verification:")
    print("  02e8 design contract: PASS")
    print("  02e8 contract hash: PASS")
    print("  locked DOG² expression hash: PASS")
    print("  02a authoritative mapping: PASS")
    print("  02e7 DOG² roster: PASS")
    print()

    expression = load_expression(expr_path, mapping)
    meta, y = align_analysis_table(expression, mapping, roster)

    X = expression.to_numpy(dtype=np.float64, copy=False)
    gene_names = expression.columns.astype(str).to_numpy()

    print("Expression validation:")
    print(f"  matrix shape: {X.shape[0]} x {X.shape[1]}")
    print("  expression index == 02a sample-ID set: PASS")
    print("  mapped COTC set == 02e7 roster: PASS")
    print("  COTC021: 93")
    print("  COTC022: 93")
    print("  finite numeric values: PASS")
    print()

    ranked, variances, nonzero_mask = rank_variable_genes(X, gene_names)
    n_zero = int(np.sum(~nonzero_mask))
    n_nonzero = int(np.sum(nonzero_mask))

    print("Label-blind variance inventory:")
    print(f"  nonzero-variance genes: {n_nonzero}")
    print(f"  zero/nonfinite-variance genes removed: {n_zero}")
    print()

    scales = [
        FeatureScale(
            name=str(row["name"]),
            top_n=int(row["top_variable_genes"]),
            role=str(row["role"]),
        )
        for row in contract["feature_scales"]
    ]

    pca_components = int(contract["global_structure"]["pca_components"])
    perm_n = int(
        contract["global_structure"]["permanova"]["permutations"]
    )
    perm_seed = int(
        contract["global_structure"]["permanova"]["random_seed"]
    )
    cv_splits = int(contract["arm_predictability"]["cv_splits"])
    cv_repeats = int(contract["arm_predictability"]["cv_repeats"])
    classifier_perm_n = int(
        contract["arm_predictability"]["label_permutations"]
    )
    classifier_seed = int(
        contract["arm_predictability"]["random_seed"]
    )

    scale_results: List[Dict[str, Any]] = []
    pc_rows: List[Dict[str, Any]] = []
    cv_rows: List[Dict[str, Any]] = []

    scale_detail: Dict[str, Dict[str, Any]] = {}

    for scale_index, scale in enumerate(scales):
        print("-" * 112)
        print(
            f"Feature scale: {scale.name} "
            f"[{scale.role}; top {scale.top_n:,} variable genes]"
        )
        print("-" * 112)

        selected = select_top(ranked, scale.top_n)
        selected_gene_names = gene_names[selected]

        scores, explained = global_pca(
            X,
            selected,
            pca_components,
        )

        permanova = permanova_permutation(
            scores,
            y,
            perm_n,
            seed=perm_seed + scale_index,
        )

        pcs = pc_diagnostics(scores, y, scale.name)
        for row, evr in zip(pcs, explained):
            row["explained_variance_ratio"] = float(evr)
        pc_rows.extend(pcs)

        print(
            f"  global PCA/PERMANOVA: "
            f"R2={permanova['R2_arm']:.5f}, "
            f"pseudo-F={permanova['pseudo_F']:.4f}, "
            f"p={permanova['permutation_p']:.6g}"
        )

        print(
            f"  preparing repeated {cv_splits}-fold x {cv_repeats} "
            f"training-fold-only classifier preprocessing..."
        )
        caches = prepare_fold_caches(
            X,
            y,
            top_n=scale.top_n,
            n_pcs=pca_components,
            n_splits=cv_splits,
            n_repeats=cv_repeats,
            seed=classifier_seed,
        )

        observed, fold_rows = observed_cv(caches, y, scale.name)
        cv_rows.extend(fold_rows)

        print(
            f"  observed arm predictability: "
            f"mean AUC={observed['mean_ROC_AUC']:.4f}, "
            f"mean balanced accuracy={observed['mean_balanced_accuracy']:.4f}"
        )

        classifier = classifier_permutation_test(
            caches=caches,
            y=y,
            observed=observed,
            n_perm=classifier_perm_n,
            seed=classifier_seed + 10000 * scale_index,
            scale_name=scale.name,
            top_n=scale.top_n,
            contract_sha=contract_sha,
            expression_sha=expression_sha,
        )

        print(
            f"  classifier null test: "
            f"AUC p={classifier['ROC_AUC_permutation_p']:.6g}, "
            f"null mean={classifier['ROC_AUC_null_mean']:.4f}"
        )

        detail = {
            "feature_scale": scale.name,
            "role": scale.role,
            "top_variable_genes": scale.top_n,
            "selected_gene_set_sha256": hashlib.sha256(
                "\n".join(map(str, selected_gene_names)).encode("utf-8")
            ).hexdigest(),
            "PCA_explained_variance_ratio_first20": [
                float(x) for x in explained
            ],
            "PCA_cumulative_variance_first20": float(np.sum(explained)),
            "PERMANOVA": permanova,
            "classifier": classifier,
        }
        scale_detail[scale.name] = detail

        scale_results.append(
            {
                "feature_scale": scale.name,
                "role": scale.role,
                "top_variable_genes": scale.top_n,
                "PCA20_cumulative_variance": float(np.sum(explained)),
                "PERMANOVA_R2_arm": permanova["R2_arm"],
                "PERMANOVA_pseudo_F": permanova["pseudo_F"],
                "PERMANOVA_permutation_p": permanova["permutation_p"],
                "classifier_mean_ROC_AUC": classifier["mean_ROC_AUC"],
                "classifier_sd_ROC_AUC": classifier["sd_ROC_AUC"],
                "classifier_ROC_AUC_permutation_p": (
                    classifier["ROC_AUC_permutation_p"]
                ),
                "classifier_mean_balanced_accuracy": (
                    classifier["mean_balanced_accuracy"]
                ),
                "classifier_balanced_accuracy_permutation_p": (
                    classifier["balanced_accuracy_permutation_p"]
                ),
            }
        )
        print()

    # ------------------------------------------------------------------
    # Supporting gene-level arm diagnostics across all nonzero-variance genes.
    # ------------------------------------------------------------------
    print("-" * 112)
    print("Gene-level supporting diagnostics [all nonzero-variance genes]")
    print("-" * 112)

    gene_contract = contract["gene_level_supporting"]
    strong_def = gene_contract["strong_gene_definition"]

    gene_df, gene_summary = gene_level_diagnostics(
        X,
        y,
        gene_names,
        nonzero_mask,
        q_threshold=float(strong_def["q_lt"]),
        smd_threshold=float(strong_def["abs_SMD_gte"]),
    )
    gene_df.to_csv(GENE_TSV, sep="\t", index=False)

    strong_fraction_warning_threshold = float(
        gene_contract["strong_gene_fraction_warning"]
    )
    gene_summary["strong_gene_fraction_warning_threshold"] = (
        strong_fraction_warning_threshold
    )
    gene_summary["strong_gene_fraction_warning"] = (
        gene_summary["strong_gene_fraction"]
        >= strong_fraction_warning_threshold
    )

    print(f"  genes tested: {gene_summary['genes_tested']}")
    print(f"  strong genes: {gene_summary['strong_genes']}")
    print(
        f"  strong-gene fraction: "
        f"{gene_summary['strong_gene_fraction']:.5f}"
    )
    print(
        "  supporting strong-gene warning: "
        f"{'YES' if gene_summary['strong_gene_fraction_warning'] else 'NO'}"
    )
    print()

    # ------------------------------------------------------------------
    # Baseline-adjusted sensitivity: primary top-5000 only.
    # ------------------------------------------------------------------
    adjustment_specs = read_adjustment_covariate_names()
    adjusted_result: Dict[str, Any]

    if adjustment_specs:
        print("-" * 112)
        print("Baseline-adjusted sensitivity [primary top-5,000 only]")
        print("-" * 112)

        covariates = load_baseline_covariates(meta, adjustment_specs)
        covariates = covariates.loc[expression.index]

        design, levels = one_hot_design(covariates)
        primary_scale = next(s for s in scales if s.role == "PRIMARY")
        primary_selected = select_top(ranked, primary_scale.top_n)

        adjusted_scores, adjusted_explained = residualized_global_pca(
            X,
            primary_selected,
            design,
            pca_components,
        )
        adjusted_permanova = permanova_permutation(
            adjusted_scores,
            y,
            perm_n,
            seed=perm_seed + 50000,
        )

        print(
            f"  residualized global PERMANOVA: "
            f"R2={adjusted_permanova['R2_arm']:.5f}, "
            f"p={adjusted_permanova['permutation_p']:.6g}"
        )

        adjusted_caches = build_adjusted_fold_caches(
            X,
            y,
            covariates,
            top_n=primary_scale.top_n,
            n_pcs=pca_components,
            n_splits=cv_splits,
            n_repeats=cv_repeats,
            seed=classifier_seed,
        )
        adjusted_cv, _ = observed_cv(
            adjusted_caches,
            y,
            "primary_top5000_baseline_adjusted",
        )

        print(
            f"  residualized observed classifier: "
            f"mean AUC={adjusted_cv['mean_ROC_AUC']:.4f}, "
            f"balanced accuracy={adjusted_cv['mean_balanced_accuracy']:.4f}"
        )
        print("  adjusted classifier permutation inference: NOT GATING / NOT RUN")
        print()

        adjusted_result = {
            "enabled": True,
            "gate_role": "INTERPRETIVE_SENSITIVITY_ONLY",
            "covariates": adjustment_specs,
            "one_hot_reference_levels": {
                key: levels_list[0]
                for key, levels_list in levels.items()
                if levels_list
            },
            "one_hot_all_levels": levels,
            "method": {
                "global": (
                    "zscore primary top5000 genes; regress standardized genes "
                    "on frozen baseline design; PCA20 on residuals; PERMANOVA arm"
                ),
                "predictive_CV": (
                    "within each training fold: variance rank, zscore, regress "
                    "genes on frozen baseline design using training coefficients, "
                    "PCA20, L2 logistic; apply all transformations to test fold"
                ),
                "classifier_permutation_inference": False,
            },
            "PERMANOVA": adjusted_permanova,
            "PCA20_cumulative_variance": float(
                np.sum(adjusted_explained)
            ),
            "classifier_observed": adjusted_cv,
        }
    else:
        adjusted_result = {
            "enabled": False,
            "reason": "02e8 froze zero baseline adjustment covariates",
        }

    write_json(ADJUSTED_JSON, adjusted_result)

    # ------------------------------------------------------------------
    # Frozen gate decision.
    # ------------------------------------------------------------------
    thresholds = contract["gate_thresholds"]
    primary = next(row for row in scale_results if row["role"] == "PRIMARY")

    primary_decision = primary_gate(
        r2=float(primary["PERMANOVA_R2_arm"]),
        auc=float(primary["classifier_mean_ROC_AUC"]),
        thresholds=thresholds,
    )

    sensitivity_decisions = {}
    sensitivity_red = False
    for row in scale_results:
        if row["role"] != "SENSITIVITY":
            continue
        decision = primary_gate(
            r2=float(row["PERMANOVA_R2_arm"]),
            auc=float(row["classifier_mean_ROC_AUC"]),
            thresholds=thresholds,
        )
        sensitivity_decisions[row["feature_scale"]] = decision
        if decision == "RED":
            sensitivity_red = True

    if primary_decision == "RED":
        final_gate = "RED_HOLD_TRANSFER_ANALYSIS"
    elif sensitivity_red:
        final_gate = "HOLD_SENSITIVITY_RED_DISCORDANCE"
    elif primary_decision == "GREEN":
        final_gate = "GREEN_NO_MATERIAL_ARM_EXPRESSION_SEPARATION"
    else:
        final_gate = "AMBER_EXPLICIT_ARM_HANDLING_REQUIRED"

    # Supporting PC warning count.
    pc_warning_threshold = float(
        contract["global_structure"]["pc_supporting_diagnostics"][
            "absolute_SMD_warning"
        ]
    )
    pc_warning_rows = [
        row
        for row in pc_rows
        if row["feature_scale"] == "primary_top5000"
        and row["abs_SMD"] is not None
        and float(row["abs_SMD"]) >= pc_warning_threshold
    ]

    # Persist tabular outputs.
    write_tsv(FEATURE_SCALE_TSV, scale_results)
    write_tsv(PC_TSV, pc_rows)
    write_tsv(CV_FOLD_TSV, cv_rows)

    results = {
        "script_version": SCRIPT_VERSION,
        "status": "PASS",
        "run_started_utc": started,
        "run_finished_utc": now_utc(),
        "scientific_stage": "EXPRESSION_GATE_ZERO",
        "question": contract.get("question"),
        "cohort": {
            "n": EXPECTED_N,
            "arm_counts": EXPECTED_ARMS,
            "expression_sha256": expression_sha,
            "02e8_contract_sha256": contract_sha,
        },
        "expression_inventory": {
            "shape": [int(X.shape[0]), int(X.shape[1])],
            "nonzero_variance_genes": n_nonzero,
            "removed_zero_or_nonfinite_variance_genes": n_zero,
        },
        "feature_scales": scale_detail,
        "gene_level_supporting": gene_summary,
        "primary_pc_supporting": {
            "absolute_SMD_warning_threshold": pc_warning_threshold,
            "PC_warning_count": len(pc_warning_rows),
            "warning_PCs": [
                {
                    "PC": row["PC"],
                    "abs_SMD": row["abs_SMD"],
                    "BH_q": row["BH_q"],
                }
                for row in pc_warning_rows
            ],
        },
        "baseline_adjusted_sensitivity": adjusted_result,
        "gate": {
            "primary_decision": primary_decision,
            "sensitivity_decisions": sensitivity_decisions,
            "sensitivity_RED_present": sensitivity_red,
            "final_gate": final_gate,
            "thresholds": thresholds,
            "p_values_alone_change_gate": False,
            "gene_level_supporting_changes_gate": False,
            "baseline_adjusted_sensitivity_changes_primary_gate": False,
        },
        "safety": {
            "outcome_response_followup_values_read": False,
            "treatment_administration_values_read": False,
            "postbaseline_sample_annotations_read": False,
            "paper4_clinical_file_read": False,
            "samples_dropped_based_on_expression_results": False,
        },
    }
    write_json(RESULTS_JSON, results)

    final_hashes = {
        rel(FEATURE_SCALE_TSV): sha256_file(FEATURE_SCALE_TSV),
        rel(PC_TSV): sha256_file(PC_TSV),
        rel(GENE_TSV): sha256_file(GENE_TSV),
        rel(CV_FOLD_TSV): sha256_file(CV_FOLD_TSV),
        rel(ADJUSTED_JSON): sha256_file(ADJUSTED_JSON),
        rel(RESULTS_JSON): sha256_file(RESULTS_JSON),
    }

    summary = {
        "script_version": SCRIPT_VERSION,
        "status": "PASS",
        "run_started_utc": started,
        "run_finished_utc": now_utc(),
        "dog2_expression_values_read": True,
        "outcome_response_followup_values_read": False,
        "treatment_administration_values_read": False,
        "postbaseline_sample_annotations_read": False,
        "paper4_clinical_file_read": False,
        "samples_analyzed": EXPECTED_N,
        "arm_counts": EXPECTED_ARMS,
        "expression_features": int(X.shape[1]),
        "nonzero_variance_genes": n_nonzero,
        "primary_PERMANOVA_R2": float(primary["PERMANOVA_R2_arm"]),
        "primary_PERMANOVA_p": float(primary["PERMANOVA_permutation_p"]),
        "primary_classifier_mean_AUC": float(
            primary["classifier_mean_ROC_AUC"]
        ),
        "primary_classifier_AUC_permutation_p": float(
            primary["classifier_ROC_AUC_permutation_p"]
        ),
        "primary_classifier_mean_balanced_accuracy": float(
            primary["classifier_mean_balanced_accuracy"]
        ),
        "primary_decision": primary_decision,
        "sensitivity_decisions": sensitivity_decisions,
        "final_gate": final_gate,
        "primary_PC_warning_count": len(pc_warning_rows),
        "strong_gene_count": gene_summary["strong_genes"],
        "strong_gene_fraction": gene_summary["strong_gene_fraction"],
        "strong_gene_fraction_warning": (
            gene_summary["strong_gene_fraction_warning"]
        ),
        "baseline_adjusted_sensitivity_enabled": bool(
            adjusted_result.get("enabled")
        ),
        "final_artifact_hashes": final_hashes,
    }
    write_json(SUMMARY_JSON, summary)

    print("=" * 112)
    print("EXPRESSION GATE ZERO SUMMARY")
    print("=" * 112)

    for row in scale_results:
        decision = primary_gate(
            float(row["PERMANOVA_R2_arm"]),
            float(row["classifier_mean_ROC_AUC"]),
            thresholds,
        )
        print(
            f"{row['feature_scale']}: "
            f"R2={float(row['PERMANOVA_R2_arm']):.5f}, "
            f"PERMANOVA p={float(row['PERMANOVA_permutation_p']):.6g}, "
            f"CV AUC={float(row['classifier_mean_ROC_AUC']):.4f}, "
            f"AUC perm p={float(row['classifier_ROC_AUC_permutation_p']):.6g}, "
            f"decision={decision}"
        )

    print()
    print("Supporting diagnostics:")
    print(f"  primary PC |SMD| warning count: {len(pc_warning_rows)}")
    print(
        f"  strong genes: {gene_summary['strong_genes']}/"
        f"{gene_summary['genes_tested']} "
        f"({gene_summary['strong_gene_fraction']:.5f})"
    )
    print(
        "  strong-gene fraction warning: "
        f"{'YES' if gene_summary['strong_gene_fraction_warning'] else 'NO'}"
    )

    if adjusted_result.get("enabled"):
        print()
        print("Baseline-adjusted sensitivity [not gating]:")
        print(
            f"  PERMANOVA R2="
            f"{adjusted_result['PERMANOVA']['R2_arm']:.5f}, "
            f"p={adjusted_result['PERMANOVA']['permutation_p']:.6g}"
        )
        print(
            f"  observed CV AUC="
            f"{adjusted_result['classifier_observed']['mean_ROC_AUC']:.4f}"
        )

    print()
    print(f"Primary frozen decision: {primary_decision}")
    print(f"FINAL GATE ZERO: {final_gate}")
    print()
    if final_gate == "GREEN_NO_MATERIAL_ARM_EXPRESSION_SEPARATION":
        print(
            "Interpretation: no material arm-level expression separation under "
            "the frozen primary thresholds. Proceed only to the next "
            "pre-outcome transport-design step; outcomes remain closed."
        )
    elif final_gate == "AMBER_EXPLICIT_ARM_HANDLING_REQUIRED":
        print(
            "Interpretation: moderate arm/domain structure is present. "
            "Downstream transfer analysis must explicitly preserve/adjust for "
            "arm identity before outcomes are opened."
        )
    else:
        print(
            "Interpretation: Gate Zero is on HOLD. Do not proceed to outcome "
            "analysis or unqualified cross-species transfer claims."
        )

    print()
    print(f"Artifacts: {OUT_DIR.relative_to(ROOT)}")
    print(f"  {FEATURE_SCALE_TSV.name}")
    print(f"  {PC_TSV.name}")
    print(f"  {GENE_TSV.name}")
    print(f"  {CV_FOLD_TSV.name}")
    print(f"  {ADJUSTED_JSON.name}")
    print(f"  {RESULTS_JSON.name}")
    print(f"  {SUMMARY_JSON.name}")
    print(f"  checkpoints\\classifier_permutations_*.json")
    print()
    print("DOG² expression values read: YES")
    print("Outcome/response/follow-up values read: NO")
    print("Treatment-administration values read: NO")
    print("Post-baseline sample annotations read: NO")
    print("Paper-4 clinical file read: NO")
    print("Samples dropped based on expression results: NO")
    print()
    print("02e9 frozen expression Gate Zero execution: PASS")
    print("=" * 112)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 112, file=sys.stderr)
        print("02e9 frozen expression Gate Zero execution: FAIL", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print("=" * 112, file=sys.stderr)
        raise
