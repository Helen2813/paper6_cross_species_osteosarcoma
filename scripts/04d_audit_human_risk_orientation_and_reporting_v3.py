#!/usr/bin/env python3
"""
Paper 6 - audit GSE16091 survival-risk orientation and premise reporting.

Purpose
-------
This is a bounded post-premise implementation audit prompted by the unexpectedly
low target-fitted OOF concordance values in 04c.

It DOES NOT retune, reselect, or reclassify the primary premise result.

Questions
---------
1. Does the custom B0 CenteredRidgeCox use the same risk orientation as an
   independent scikit-survival CoxPHSurvivalAnalysis reference?
2. Do B0/B4 models rank TRAINING patients in the hazard-consistent direction
   while their held-out predictions become anti-concordant?
3. Does B1 Coxnet predict() have the expected +X beta risk orientation?
4. Does score negation merely produce the expected mirror C-index?
5. Did B1 actually use the same outer folds as B0/B4 as required by 04b?
6. How should the 04c premise result be reported when the positive IBS criterion
   was not fully assessable?
7. Freeze, before looking at module-context rankings, the exploratory mTOR
   treatment-context hypothesis.

No network access.
No reserved human outcomes.
No new feature selection.
No hyperparameter tuning.
No premise reclassification.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sksurv.linear_model import CoxPHSurvivalAnalysis


SCRIPT_VERSION = "04d-audit-human-risk-orientation-and-reporting-v3-dynamic-import-fix-no-cli"

ROOT = Path(__file__).resolve().parents[1]

# Frozen implementation/data assets.
B4_DIR = ROOT / "results" / "human_premise_protocol" / "04b"
B4_PROTOCOL = B4_DIR / "classical_premise_test_protocol.json"
B4_SUMMARY = B4_DIR / "summary.json"
B4_PROBE_MAP = B4_DIR / "gse16091_gpl96_probe_gene_map.tsv"
B4_GENE_UNIVERSE = B4_DIR / "gse16091_outcome_blind_gene_universe.tsv"
B4_HALLMARK_MAP = B4_DIR / "gse16091_hallmark_feature_map.tsv"

A4_DIR = ROOT / "results" / "human_premise_target" / "04a"
A4_ROSTER = A4_DIR / "sacrificial_target_sample_roster.tsv"

C4_DIR = ROOT / "results" / "human_premise" / "04c"
C4_RAW = C4_DIR / "raw" / "GSE16091_series_matrix.txt.gz"
C4_RAW_LOCK = C4_DIR / "raw" / "GSE16091_series_matrix_lock.json"
C4_PRIMARY_PRED = C4_DIR / "primary_classical_oof_predictions.tsv"
C4_PRIMARY_FOLDS = C4_DIR / "primary_classical_fold_metrics.tsv"
C4_PRIMARY_RESULT = C4_DIR / "primary_premise_result.json"
C4_FINAL_RESULT = C4_DIR / "classical_premise_test_results.json"
C4_SUMMARY = C4_DIR / "summary.json"

C4_B1_PRED = C4_DIR / "B1_gene_elastic_net_oof_predictions.tsv"
C4_B1_FOLDS = C4_DIR / "B1_gene_elastic_net_fold_metrics.tsv"

C4_SCRIPT = ROOT / "scripts" / "04c_run_classical_premise_test_v3.py"

OUT_DIR = ROOT / "results" / "human_premise_diagnostics" / "04d"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCORE_MIRROR = OUT_DIR / "score_orientation_mirror_audit.tsv"
B0_REFERENCE = OUT_DIR / "B0_custom_vs_sksurv_reference.tsv"
TRAIN_TEST = OUT_DIR / "target_fitted_train_test_orientation.tsv"
B1_ORIENTATION = OUT_DIR / "B1_coxnet_orientation_audit.tsv"
B1_SPLITS = OUT_DIR / "B1_outer_split_contract_audit.tsv"
TRAIN_FIT_SUMMARY = OUT_DIR / "target_fitted_training_generalization_summary.tsv"
REPORTING = OUT_DIR / "premise_reporting_addendum.json"
MTOR_FREEZE = OUT_DIR / "mtor_context_hypothesis_contract.json"
SUMMARY_JSON = OUT_DIR / "summary.json"

EXPECTED_N = 34
EXPECTED_EVENTS = 15
EXPECTED_OUTER_SPLITS = 5
EXPECTED_OUTER_REPEATS = 20
EXPECTED_INNER_SPLITS = 4

REFERENCE_RISK_CORR_FLOOR = 0.90
REFERENCE_COEF_COSINE_FLOOR = 0.80
TRAIN_C_MEDIAN_FLOOR = 0.50

# Interpretive-only, frozen before 04d results are viewed.
# These thresholds DO NOT alter the orientation PASS/HOLD rule or any premise result.
TRAIN_C_STRONG_FIT = 0.75
TRAIN_C_SEVERE_OVERFIT = 0.90
TRAIN_C_NEAR_MEMORIZATION = 0.95

MTOR_MODULES = [
    "HALLMARK_MTORC1_SIGNALING",
    "HALLMARK_PI3K_AKT_MTOR_SIGNALING",
]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


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
        raise FileNotFoundError(f"Required artifact missing: {path}")
    return path


def load_module(path: Path, name: str):
    """
    Import a Python file under an explicit module name.

    IMPORTANT:
    Register the module in sys.modules BEFORE exec_module().  Python's
    dataclasses implementation resolves cls.__module__ through sys.modules while
    decorating classes; omitting this registration produces:
        AttributeError: 'NoneType' object has no attribute '__dict__'
    when importing 04c_run_classical_premise_test_v3.py.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")

    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(name)
    sys.modules[name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        # Restore the pre-existing module state on failed import.
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
        raise

    return module


def corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) != len(y):
        return float("nan")
    if np.std(x) <= 1e-14 or np.std(y) <= 1e-14:
        return float("nan")

    return float(np.corrcoef(x, y)[0, 1])


def rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    rx = pd.Series(np.asarray(x, dtype=float)).rank(method="average").to_numpy()
    ry = pd.Series(np.asarray(y, dtype=float)).rank(method="average").to_numpy()
    return corr(rx, ry)


def cosine(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)

    nx = float(np.linalg.norm(x))
    ny = float(np.linalg.norm(y))

    if nx <= 1e-14 or ny <= 1e-14:
        return float("nan")

    return float(np.dot(x, y) / (nx * ny))


def aggregate_saved_risk(
    frame: pd.DataFrame,
    sample_ids: np.ndarray,
    model: str,
) -> np.ndarray:
    part = frame[frame["model"].astype(str) == model]
    risk = part.groupby("sample_id")["risk_standardized"].mean()

    if set(risk.index.astype(str)) != set(sample_ids.astype(str)):
        raise RuntimeError(f"{model}: saved OOF sample coverage mismatch.")

    return np.asarray([risk.loc[x] for x in sample_ids], dtype=float)


def exact_set_signature(values: Sequence[str]) -> str:
    payload = "\n".join(sorted(str(x) for x in values))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def training_fit_label(median_train_c: float) -> str:
    if not np.isfinite(median_train_c):
        return "NONFINITE"
    if median_train_c >= TRAIN_C_NEAR_MEMORIZATION:
        return "NEAR_MEMORIZATION_WARNING"
    if median_train_c >= TRAIN_C_SEVERE_OVERFIT:
        return "SEVERE_IN_SAMPLE_OVERFIT"
    if median_train_c >= TRAIN_C_STRONG_FIT:
        return "STRONG_IN_SAMPLE_FIT_COMPATIBLE_WITH_OVERFITTING"
    if median_train_c > TRAIN_C_MEDIAN_FLOOR:
        return "MODEST_IN_SAMPLE_FIT"
    return "ORIENTATION_OR_IDENTIFIABILITY_CONCERN"


def reconstruct_target(m04c, protocol: Dict[str, Any]) -> Dict[str, Any]:
    sample_accessions, characteristics, probe_expression = (
        m04c.parse_series_matrix(C4_RAW)
    )

    roster = pd.read_csv(A4_ROSTER, sep="\t", dtype=str).fillna("")
    frozen_samples = roster["geo_accession"].astype(str).tolist()

    if set(sample_accessions) != set(frozen_samples):
        raise RuntimeError("04d sample set differs from frozen 04a roster.")

    sample_accessions = frozen_samples
    probe_expression = probe_expression[sample_accessions]

    target = protocol["target"]
    status_mapping = {
        str(k): int(v)
        for k, v in target["status_mapping"].items()
    }

    outcome = m04c.build_target_outcome(
        sample_accessions,
        characteristics,
        target["time_field"],
        target["status_field"],
        status_mapping,
    )
    outcome = (
        outcome.set_index("geo_accession")
        .loc[sample_accessions]
        .reset_index()
    )

    event = outcome["os_event"].to_numpy(dtype=int).astype(bool)
    time_values = outcome["os_time"].to_numpy(dtype=float)
    y = m04c.make_surv(time_values, event)
    sample_ids = np.asarray(sample_accessions, dtype=str)

    if len(y) != EXPECTED_N or int(event.sum()) != EXPECTED_EVENTS:
        raise RuntimeError(
            f"GSE16091 identity changed: n={len(y)}, events={int(event.sum())}."
        )

    probe_map = pd.read_csv(
        B4_PROBE_MAP,
        sep="\t",
        dtype=str,
    ).fillna("")
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

    X_gene, gene_order = m04c.collapse_probes_to_genes(
        probe_expression,
        probe_map,
        gene_universe,
        sample_accessions,
    )

    gene_index = {
        gene: i
        for i, gene in enumerate(gene_order)
    }

    module_order = sorted(
        hallmark_map["hallmark_module"]
        .astype(str)
        .unique()
        .tolist()
    )

    module_to_indices = {}

    for module in module_order:
        genes = (
            hallmark_map.loc[
                hallmark_map["hallmark_module"].astype(str) == module,
                "human_gene_symbol",
            ]
            .astype(str)
            .tolist()
        )
        module_to_indices[module] = np.asarray(
            [gene_index[g] for g in genes],
            dtype=int,
        )

    return {
        "sample_ids": sample_ids,
        "outcome": outcome,
        "event": event,
        "y": y,
        "X_gene": X_gene,
        "module_order": module_order,
        "module_to_indices": module_to_indices,
    }


def main() -> None:
    started = now_utc()

    print("=" * 120)
    print("Paper 6 - audit GSE16091 survival-risk orientation and premise reporting")
    print("=" * 120)
    print(f"Script version: {SCRIPT_VERSION}")
    print()
    print("Scope:")
    print("  GSE16091 outcome/expression values read: YES [already opened in 04c]")
    print("  Reserved human outcomes read: NO")
    print("  New hyperparameter tuning: NO")
    print("  Primary premise reclassification: NO")
    print("  Network access: NO")
    print("  Dynamic import registration before dataclass execution: YES [v3 fix]")
    print()

    for path in [
        B4_PROTOCOL,
        B4_SUMMARY,
        B4_PROBE_MAP,
        B4_GENE_UNIVERSE,
        B4_HALLMARK_MAP,
        A4_ROSTER,
        C4_RAW,
        C4_RAW_LOCK,
        C4_PRIMARY_PRED,
        C4_PRIMARY_FOLDS,
        C4_PRIMARY_RESULT,
        C4_FINAL_RESULT,
        C4_SUMMARY,
        C4_B1_PRED,
        C4_B1_FOLDS,
        C4_SCRIPT,
    ]:
        require_file(path)

    protocol = read_json(B4_PROTOCOL)
    protocol_summary = read_json(B4_SUMMARY)
    primary_result = read_json(C4_PRIMARY_RESULT)
    final_result = read_json(C4_FINAL_RESULT)
    summary_04c = read_json(C4_SUMMARY)

    if clean(protocol.get("status")) != "PASS":
        raise RuntimeError("04b protocol is not PASS.")
    if clean(summary_04c.get("status")) != "PASS":
        raise RuntimeError("04c final summary is not PASS.")

    protocol_hash = sha256_file(B4_PROTOCOL)
    expected_protocol_hash = clean(
        (protocol_summary.get("final_artifact_hashes") or {}).get(
            "classical_premise_test_protocol_json"
        )
    )
    if protocol_hash != expected_protocol_hash:
        raise RuntimeError("04b protocol hash verification failed.")

    m04c = load_module(
        C4_SCRIPT,
        "paper6_04c_v3_for_04d",
    )
    target = reconstruct_target(m04c, protocol)

    sample_ids = target["sample_ids"]
    event = target["event"]
    y = target["y"]
    X_gene = target["X_gene"]
    module_order = target["module_order"]
    module_to_indices = target["module_to_indices"]

    primary_pred = pd.read_csv(C4_PRIMARY_PRED, sep="\t")
    primary_folds = pd.read_csv(C4_PRIMARY_FOLDS, sep="\t")
    b1_pred = pd.read_csv(C4_B1_PRED, sep="\t")
    b1_folds = pd.read_csv(C4_B1_FOLDS, sep="\t")

    # ------------------------------------------------------------------
    # 1. Saved-score mirror audit.
    # ------------------------------------------------------------------
    mirror_rows: List[Dict[str, Any]] = []

    for model in ["B0", "B2", "B3", "B4"]:
        risk = aggregate_saved_risk(
            primary_pred,
            sample_ids,
            model,
        )
        c_pos = m04c.safe_uno_c(y, y, risk)
        c_neg = m04c.safe_uno_c(y, y, -risk)

        mirror_rows.append(
            {
                "model": model,
                "fit_uses_human_outcomes": model in {"B0", "B4"},
                "uno_c_saved_risk": c_pos,
                "uno_c_negated_risk": c_neg,
                "c_plus_negated_c_minus_1": c_pos + c_neg - 1.0,
                "interpretation": (
                    "Mirror relation is algebraic/ranking sanity only; "
                    "it does NOT diagnose which sign is scientifically correct."
                ),
            }
        )

    risk_b1 = aggregate_saved_risk(
        b1_pred,
        sample_ids,
        "B1",
    )
    c1 = m04c.safe_uno_c(y, y, risk_b1)
    c1_neg = m04c.safe_uno_c(y, y, -risk_b1)

    mirror_rows.append(
        {
            "model": "B1",
            "fit_uses_human_outcomes": True,
            "uno_c_saved_risk": c1,
            "uno_c_negated_risk": c1_neg,
            "c_plus_negated_c_minus_1": c1 + c1_neg - 1.0,
            "interpretation": (
                "Mirror relation is algebraic/ranking sanity only; "
                "it does NOT diagnose which sign is scientifically correct."
            ),
        }
    )

    mirror = pd.DataFrame(mirror_rows)
    mirror.to_csv(SCORE_MIRROR, sep="\t", index=False)

    # ------------------------------------------------------------------
    # 2. Refit B0/B4 exact outer folds; compare B0 with independent
    #    scikit-survival CoxPH reference implementation.
    # ------------------------------------------------------------------
    base_seed = int(
        protocol["paired_human_evaluation"]["base_seed"]
    )
    outer_splits = int(
        protocol["paired_human_evaluation"]["outer_splits"]
    )
    outer_repeats = int(
        protocol["paired_human_evaluation"]["outer_repeats"]
    )

    if outer_splits != EXPECTED_OUTER_SPLITS or outer_repeats != EXPECTED_OUTER_REPEATS:
        raise RuntimeError("04b outer-CV design changed.")

    reference_rows: List[Dict[str, Any]] = []
    train_test_rows: List[Dict[str, Any]] = []

    for repeat in range(outer_repeats):
        splits = m04c.stratified_splits(
            event,
            outer_splits,
            base_seed + repeat,
        )

        for outer_fold, (train_idx, test_idx) in enumerate(splits):
            transformer = m04c.HumanHallmarkTransformer(
                module_to_indices,
                module_order,
            ).fit(X_gene[train_idx])

            X_train = transformer.transform(X_gene[train_idx])
            X_test = transformer.transform(X_gene[test_idx])

            for model_name in ["B0", "B4"]:
                saved = primary_folds[
                    (primary_folds["repeat"] == repeat)
                    & (primary_folds["outer_fold"] == outer_fold)
                    & (primary_folds["model"].astype(str) == model_name)
                ]
                if len(saved) != 1:
                    raise RuntimeError(
                        f"{model_name} repeat={repeat} fold={outer_fold}: "
                        "saved fold row not unique."
                    )

                saved_row = saved.iloc[0]
                alpha = float(saved_row["chosen_alpha"])

                if model_name == "B0":
                    center = np.zeros(len(module_order), dtype=float)
                else:
                    # Reconstruct the exact frozen full-DOG2 beta vector from 04c.
                    dog_coef_path = C4_DIR / "dog2_full_source_hallmark_coefficients.tsv"
                    require_file(dog_coef_path)
                    dog_coef = pd.read_csv(dog_coef_path, sep="\t")
                    dog_coef = (
                        dog_coef.set_index("hallmark_module")
                        .loc[module_order, "beta_DOG2"]
                        .to_numpy(dtype=float)
                    )
                    center = dog_coef

                custom = m04c.CenteredRidgeCox(
                    alpha=alpha,
                    center=center,
                ).fit(X_train, y[train_idx])

                train_risk = custom.predict(X_train)
                test_risk = custom.predict(X_test)

                c_train = m04c.safe_uno_c(
                    y[train_idx],
                    y[train_idx],
                    train_risk,
                )
                c_test = m04c.safe_uno_c(
                    y[train_idx],
                    y[test_idx],
                    test_risk,
                )

                train_test_rows.append(
                    {
                        "model": model_name,
                        "repeat": repeat,
                        "outer_fold": outer_fold,
                        "alpha": alpha,
                        "train_uno_c": c_train,
                        "test_uno_c_refit": c_test,
                        "saved_test_uno_c": float(saved_row["fold_uno_c"]),
                        "refit_minus_saved_test_c": (
                            c_test - float(saved_row["fold_uno_c"])
                        ),
                        "train_risk_sd": float(np.std(train_risk, ddof=0)),
                        "test_risk_sd": float(np.std(test_risk, ddof=0)),
                    }
                )

                if model_name == "B0":
                    reference = CoxPHSurvivalAnalysis(
                        alpha=alpha,
                        ties="breslow",
                        n_iter=500,
                        tol=1e-9,
                    ).fit(X_train, y[train_idx])

                    ref_train = np.asarray(
                        reference.predict(X_train),
                        dtype=float,
                    )
                    ref_test = np.asarray(
                        reference.predict(X_test),
                        dtype=float,
                    )

                    reference_rows.append(
                        {
                            "repeat": repeat,
                            "outer_fold": outer_fold,
                            "alpha": alpha,
                            "custom_reference_coef_cosine": cosine(
                                custom.beta_,
                                reference.coef_,
                            ),
                            "custom_reference_train_risk_pearson": corr(
                                train_risk,
                                ref_train,
                            ),
                            "custom_reference_test_risk_pearson": corr(
                                test_risk,
                                ref_test,
                            ),
                            "custom_reference_test_risk_spearman": rank_corr(
                                test_risk,
                                ref_test,
                            ),
                            "custom_train_uno_c": c_train,
                            "reference_train_uno_c": m04c.safe_uno_c(
                                y[train_idx],
                                y[train_idx],
                                ref_train,
                            ),
                            "custom_test_uno_c": c_test,
                            "reference_test_uno_c": m04c.safe_uno_c(
                                y[train_idx],
                                y[test_idx],
                                ref_test,
                            ),
                        }
                    )

    reference_df = pd.DataFrame(reference_rows)
    train_test_df = pd.DataFrame(train_test_rows)

    reference_df.to_csv(B0_REFERENCE, sep="\t", index=False)
    train_test_df.to_csv(TRAIN_TEST, sep="\t", index=False)

    # ------------------------------------------------------------------
    # 3. B1 Coxnet orientation and split-contract audit.
    # ------------------------------------------------------------------
    b1_orientation_rows: List[Dict[str, Any]] = []
    split_rows: List[Dict[str, Any]] = []

    b1_seed_base = base_seed + 200000

    for repeat in range(outer_repeats):
        actual_splits = m04c.stratified_splits(
            event,
            outer_splits,
            b1_seed_base + repeat,
        )
        intended_splits = m04c.stratified_splits(
            event,
            outer_splits,
            base_seed + repeat,
        )

        for outer_fold, ((train_idx, test_idx), (_, intended_test_idx)) in enumerate(
            zip(actual_splits, intended_splits)
        ):
            saved = b1_folds[
                (b1_folds["repeat"] == repeat)
                & (b1_folds["outer_fold"] == outer_fold)
            ]
            if len(saved) != 1:
                raise RuntimeError(
                    f"B1 repeat={repeat} fold={outer_fold}: saved fold row not unique."
                )

            saved_row = saved.iloc[0]

            actual_test_ids = sample_ids[test_idx]
            intended_test_ids = sample_ids[intended_test_idx]

            split_rows.append(
                {
                    "repeat": repeat,
                    "outer_fold": outer_fold,
                    "actual_B1_test_set_sha256": exact_set_signature(actual_test_ids),
                    "intended_primary_test_set_sha256": exact_set_signature(intended_test_ids),
                    "exact_same_test_set": bool(
                        set(actual_test_ids) == set(intended_test_ids)
                    ),
                    "actual_test_n": len(test_idx),
                    "intended_test_n": len(intended_test_idx),
                    "contract_rule": (
                        "04b required exact same outer patient splits for B0-B4/B1."
                    ),
                }
            )

            transformer = m04c.GeneTransformer().fit(
                X_gene[train_idx]
            )
            X_train = transformer.transform(X_gene[train_idx])
            X_test = transformer.transform(X_gene[test_idx])

            l1 = float(saved_row["chosen_l1_ratio"])
            path_index = int(saved_row["chosen_path_index"])

            model = m04c.fit_coxnet_path(
                X_train,
                y[train_idx],
                l1,
            )

            alpha = float(model.alphas_[path_index])
            saved_alpha = float(saved_row["chosen_alpha"])

            if not np.isclose(alpha, saved_alpha, rtol=1e-8, atol=1e-12):
                raise RuntimeError(
                    f"B1 repeat={repeat} fold={outer_fold}: "
                    f"alpha replay mismatch {alpha} vs {saved_alpha}."
                )

            train_risk = np.asarray(
                model.predict(X_train, alpha=alpha),
                dtype=float,
            )
            test_risk = np.asarray(
                model.predict(X_test, alpha=alpha),
                dtype=float,
            )

            # Coxnet coef_ columns correspond to alpha path; an additive offset
            # does not change ranking, so correlation with X beta must be +1.
            beta = np.asarray(
                model.coef_[:, path_index],
                dtype=float,
            )
            manual_train = X_train @ beta
            manual_test = X_test @ beta

            c_train = m04c.safe_uno_c(
                y[train_idx],
                y[train_idx],
                train_risk,
            )
            c_test = m04c.safe_uno_c(
                y[train_idx],
                y[test_idx],
                test_risk,
            )

            b1_orientation_rows.append(
                {
                    "repeat": repeat,
                    "outer_fold": outer_fold,
                    "l1_ratio": l1,
                    "path_index": path_index,
                    "alpha": alpha,
                    "predict_vs_Xbeta_train_pearson": corr(
                        train_risk,
                        manual_train,
                    ),
                    "predict_vs_Xbeta_test_pearson": corr(
                        test_risk,
                        manual_test,
                    ),
                    "train_uno_c": c_train,
                    "test_uno_c_refit": c_test,
                    "saved_test_uno_c": float(saved_row["fold_uno_c"]),
                    "refit_minus_saved_test_c": (
                        c_test - float(saved_row["fold_uno_c"])
                    ),
                    "train_risk_sd": float(np.std(train_risk, ddof=0)),
                    "test_risk_sd": float(np.std(test_risk, ddof=0)),
                }
            )

    b1_orientation_df = pd.DataFrame(b1_orientation_rows)
    b1_split_df = pd.DataFrame(split_rows)

    b1_orientation_df.to_csv(
        B1_ORIENTATION,
        sep="\t",
        index=False,
    )
    b1_split_df.to_csv(
        B1_SPLITS,
        sep="\t",
        index=False,
    )

    # ------------------------------------------------------------------
    # 4. Summarize sign/orientation evidence.
    # ------------------------------------------------------------------
    ref_risk_corr = pd.to_numeric(
        reference_df["custom_reference_test_risk_pearson"],
        errors="coerce",
    )
    ref_coef_cos = pd.to_numeric(
        reference_df["custom_reference_coef_cosine"],
        errors="coerce",
    )

    b0_train = train_test_df.loc[
        train_test_df["model"] == "B0",
        "train_uno_c",
    ].to_numpy(dtype=float)
    b4_train = train_test_df.loc[
        train_test_df["model"] == "B4",
        "train_uno_c",
    ].to_numpy(dtype=float)
    b1_train = b1_orientation_df[
        "train_uno_c"
    ].to_numpy(dtype=float)

    b1_predict_corr = b1_orientation_df[
        "predict_vs_Xbeta_test_pearson"
    ].to_numpy(dtype=float)

    sign_checks = {
        "B0_reference_test_risk_correlation_median": float(
            np.nanmedian(ref_risk_corr)
        ),
        "B0_reference_coef_cosine_median": float(
            np.nanmedian(ref_coef_cos)
        ),
        "B0_training_UnoC_median": float(np.nanmedian(b0_train)),
        "B4_training_UnoC_median": float(np.nanmedian(b4_train)),
        "B1_training_UnoC_median": float(np.nanmedian(b1_train)),
        "B1_predict_vs_Xbeta_correlation_median": float(
            np.nanmedian(b1_predict_corr)
        ),
        "B1_exact_same_outer_test_sets_fraction": float(
            b1_split_df["exact_same_test_set"].mean()
        ),
    }

    # Per-fold distribution summary is diagnostic: it distinguishes a
    # systematically anti-concordant held-out pattern from a mixture of very
    # good and catastrophic folds. Training-C labels are interpretive only.
    train_fit_rows: List[Dict[str, Any]] = []

    model_fold_sources = {
        "B0": train_test_df[train_test_df["model"] == "B0"].copy(),
        "B4": train_test_df[train_test_df["model"] == "B4"].copy(),
        "B1": b1_orientation_df.copy(),
    }

    for model_name, frame in model_fold_sources.items():
        train_values = pd.to_numeric(
            frame["train_uno_c"], errors="coerce"
        ).to_numpy(dtype=float)
        test_values = pd.to_numeric(
            frame["test_uno_c_refit"], errors="coerce"
        ).to_numpy(dtype=float)

        finite = np.isfinite(train_values) & np.isfinite(test_values)
        train_values = train_values[finite]
        test_values = test_values[finite]

        if len(train_values) == 0:
            raise RuntimeError(f"{model_name}: no finite train/test fold C values.")

        median_train = float(np.median(train_values))
        median_test = float(np.median(test_values))

        train_fit_rows.append(
            {
                "model": model_name,
                "n_finite_folds": int(len(train_values)),
                "train_c_median": median_train,
                "train_c_q10": float(np.quantile(train_values, 0.10)),
                "train_c_q25": float(np.quantile(train_values, 0.25)),
                "train_c_q75": float(np.quantile(train_values, 0.75)),
                "train_c_q90": float(np.quantile(train_values, 0.90)),
                "train_c_min": float(np.min(train_values)),
                "train_c_max": float(np.max(train_values)),
                "test_c_median": median_test,
                "test_c_q10": float(np.quantile(test_values, 0.10)),
                "test_c_q25": float(np.quantile(test_values, 0.25)),
                "test_c_q75": float(np.quantile(test_values, 0.75)),
                "test_c_q90": float(np.quantile(test_values, 0.90)),
                "test_c_min": float(np.min(test_values)),
                "test_c_max": float(np.max(test_values)),
                "fraction_train_c_ge_0_95": float(
                    np.mean(train_values >= TRAIN_C_NEAR_MEMORIZATION)
                ),
                "fraction_test_c_lt_0_50": float(
                    np.mean(test_values < 0.50)
                ),
                "fraction_train_gt_0_50_and_test_lt_0_50": float(
                    np.mean(
                        (train_values > 0.50)
                        & (test_values < 0.50)
                    )
                ),
                "training_fit_interpretation": training_fit_label(median_train),
                "interpretive_only": True,
            }
        )

    train_fit_summary = pd.DataFrame(train_fit_rows)
    train_fit_summary.to_csv(
        TRAIN_FIT_SUMMARY,
        sep="\t",
        index=False,
    )

    orientation_pass = (
        sign_checks["B0_reference_test_risk_correlation_median"]
        >= REFERENCE_RISK_CORR_FLOOR
        and sign_checks["B0_reference_coef_cosine_median"]
        >= REFERENCE_COEF_COSINE_FLOOR
        and sign_checks["B0_training_UnoC_median"]
        > TRAIN_C_MEDIAN_FLOOR
        and sign_checks["B4_training_UnoC_median"]
        > TRAIN_C_MEDIAN_FLOOR
        and sign_checks["B1_training_UnoC_median"]
        > TRAIN_C_MEDIAN_FLOOR
        and sign_checks["B1_predict_vs_Xbeta_correlation_median"]
        >= REFERENCE_RISK_CORR_FLOOR
    )

    orientation_verdict = (
        "PASS_NO_IMPLEMENTATION_SIGN_INVERSION_SUPPORTED"
        if orientation_pass
        else "HOLD_SIGN_ORIENTATION_REQUIRES_MANUAL_REVIEW"
    )

    b1_split_contract_pass = bool(
        b1_split_df["exact_same_test_set"].all()
    )

    # ------------------------------------------------------------------
    # 5. Reporting addendum: preserve frozen branch, clarify assessability.
    # ------------------------------------------------------------------
    primary_cmp = primary_result.get("primary_comparison") or {}

    ibs_complete = bool(
        primary_cmp.get("IBS_complete_for_primary_decision")
    )
    n_unavailable = int(
        primary_cmp.get("n_unavailable_primary_IBS_folds", 0)
    )

    if ibs_complete:
        assessability = "PRIMARY_PREMISE_CRITERION_FULLY_ASSESSABLE"
    else:
        assessability = (
            "POSITIVE_PREMISE_CRITERION_NOT_FULLY_ASSESSABLE_DUE_IBS_SUPPORT"
        )

    reporting = {
        "status": "PASS",
        "created_utc": now_utc(),
        "frozen_machine_decision_state": clean(
            primary_cmp.get("premise_state")
        ),
        "reporting_assessability_status": assessability,
        "n_unavailable_primary_IBS_folds": n_unavailable,
        "wording_rule": (
            "Do not describe INCONCLUSIVE_NEUTRAL as evidence of no difference. "
            "If IBS is incomplete, report that the prespecified positive premise "
            "criterion was not fully assessable; separately report the observed "
            "discrimination contrast and its uncertainty."
        ),
        "human_pool_reporting_rule": (
            "04b operationalized the human pool as the GSE16091 outer-training "
            "pool. This is available-human-target-training data, NOT an independent "
            "external pooled-human source cohort. The originally envisioned "
            "external pooled-human-source premise remains untested."
        ),
        "B1_reporting_rule": (
            "If 04d confirms split mismatch, current B1=secondary implementation "
            "result is not an exactly paired comparator and must be rerun under "
            "the frozen B0/B4 outer splits before publication."
        ),
        "training_fit_reporting_guardrail": {
            "median_train_C_lt_0_75": "MODEST_IN_SAMPLE_FIT",
            "median_train_C_0_75_to_lt_0_90": (
                "STRONG_IN_SAMPLE_FIT_COMPATIBLE_WITH_OVERFITTING"
            ),
            "median_train_C_0_90_to_lt_0_95": "SEVERE_IN_SAMPLE_OVERFIT",
            "median_train_C_ge_0_95": (
                "NEAR_MEMORIZATION_WARNING; do not describe as meaningful "
                "biological structure learned in the target cohort"
            ),
            "scientific_gate_role": "INTERPRETIVE_ONLY",
        },
        "primary_B4_vs_B0_decision_affected_by_B1": False,
        "orientation_verdict": orientation_verdict,
    }
    write_json(REPORTING, reporting)

    # ------------------------------------------------------------------
    # 6. Freeze mTOR contextual-sensitivity hypothesis BEFORE ranking modules.
    # ------------------------------------------------------------------
    hallmark_modules = set(module_order)
    missing_mtor = sorted(set(MTOR_MODULES) - hallmark_modules)
    if missing_mtor:
        raise RuntimeError(
            f"Prespecified mTOR Hallmark modules missing: {missing_mtor}"
        )

    mtor_contract = {
        "status": "PASS_FROZEN_BEFORE_MODULE_CONTEXT_RANKING",
        "created_utc": now_utc(),
        "modules": MTOR_MODULES,
        "endpoint": "DOG2 OS",
        "hypothesis": (
            "The two prespecified mTOR-related Hallmark modules are enriched "
            "toward the most treatment-context-sensitive modules."
        ),
        "context_sensitivity_metric": (
            "Absolute difference in fitted standardized-module Cox coefficients "
            "between the COTC021-trained and COTC022-trained directional Hallmark "
            "models, using the already-frozen 03b directional model family and "
            "already-selected directional ridge penalties; no retuning."
        ),
        "ranking": (
            "Descending context_sensitivity_metric across all 50 Hallmark modules; "
            "rank 1 = most context-sensitive."
        ),
        "number_of_prespecified_tests": 1,
        "test_unit": "JOINT_TWO_MODULE_PAIR",
        "primary_test_statistic": (
            "Mean rank of the prespecified pair "
            "{HALLMARK_MTORC1_SIGNALING, HALLMARK_PI3K_AKT_MTOR_SIGNALING}; "
            "the two modules are NOT tested separately."
        ),
        "null_distribution": (
            "Exact distribution of the mean rank over all unordered pairs of "
            "2 distinct modules among the 50 Hallmark modules (1225 pairs)."
        ),
        "one_sided_p_value": (
            "Fraction of null module-pair mean ranks <= the observed mTOR-pair mean rank."
        ),
        "multiple_testing": (
            "No post-hoc pathway search is part of this prespecified positive-control test."
        ),
        "causal_claim": False,
        "allowed_wording_if_supported": (
            "consistent with treatment-context modulation of mTOR-related biology"
        ),
    }
    write_json(MTOR_FREEZE, mtor_contract)

    summary = {
        "script_version": SCRIPT_VERSION,
        "implementation_note": (
            "v3 fixes dynamic-module registration only: imported 04c v3 is "
            "inserted into sys.modules before exec_module so @dataclass can "
            "resolve cls.__module__. No scientific calculation or threshold changed."
        ),
        "status": "PASS" if orientation_pass else "HOLD",
        "scientific_status": orientation_verdict,
        "run_started_utc": started,
        "run_finished_utc": now_utc(),
        **sign_checks,
        "B1_outer_split_contract_pass": b1_split_contract_pass,
        "B1_outer_split_contract_defect": not b1_split_contract_pass,
        "premise_reporting_assessability_status": assessability,
        "primary_premise_state_changed": False,
        "mTOR_context_test_frozen_before_ranking": True,
        "reserved_human_outcomes_read": False,
        "final_artifact_hashes": {
            "score_orientation_mirror_audit_tsv": sha256_file(SCORE_MIRROR),
            "B0_custom_vs_sksurv_reference_tsv": sha256_file(B0_REFERENCE),
            "target_fitted_train_test_orientation_tsv": sha256_file(TRAIN_TEST),
            "B1_coxnet_orientation_audit_tsv": sha256_file(B1_ORIENTATION),
            "B1_outer_split_contract_audit_tsv": sha256_file(B1_SPLITS),
            "target_fitted_training_generalization_summary_tsv": sha256_file(
                TRAIN_FIT_SUMMARY
            ),
            "premise_reporting_addendum_json": sha256_file(REPORTING),
            "mtor_context_hypothesis_contract_json": sha256_file(MTOR_FREEZE),
        },
        "next_if_orientation_PASS": (
            "Correct the secondary B1 outer-split implementation to the frozen "
            "shared splits, then freeze the real evolutionary conservation prior "
            "(dN/dS + protein identity; technical annotation metrics separate) "
            "before reserved human outcomes."
        ),
        "next_if_orientation_HOLD": (
            "Do not run 05c. Resolve survival-risk orientation implementation "
            "before any further interpretation or AI development."
        ),
    }
    write_json(SUMMARY_JSON, summary)

    print("-" * 120)
    print("Saved-score mirror audit")
    print("-" * 120)
    print(
        mirror[
            [
                "model",
                "uno_c_saved_risk",
                "uno_c_negated_risk",
                "c_plus_negated_c_minus_1",
            ]
        ].to_string(index=False)
    )

    print()
    print("-" * 120)
    print("Independent B0 orientation reference")
    print("-" * 120)
    print(
        f"Median custom-vs-sksurv TEST risk correlation: "
        f"{sign_checks['B0_reference_test_risk_correlation_median']:.6f}"
    )
    print(
        f"Median custom-vs-sksurv coefficient cosine: "
        f"{sign_checks['B0_reference_coef_cosine_median']:.6f}"
    )
    print(
        f"Median B0 training Uno C: "
        f"{sign_checks['B0_training_UnoC_median']:.4f}"
    )
    print(
        f"Median B4 training Uno C: "
        f"{sign_checks['B4_training_UnoC_median']:.4f}"
    )
    print(
        f"Median B1 training Uno C: "
        f"{sign_checks['B1_training_UnoC_median']:.4f}"
    )
    print(
        f"Median B1 predict-vs-Xbeta correlation: "
        f"{sign_checks['B1_predict_vs_Xbeta_correlation_median']:.6f}"
    )
    print(
        "Decisive metric-orientation check: the SAME positive-risk score path "
        "must yield median TRAIN Uno C > 0.50 for B0/B4/B1 while the independent "
        "B0 implementation agrees in coefficient/risk direction. If so, a "
        "held-out C < 0.50 is not explained by passing the score to Uno C with "
        "the wrong sign."
    )

    print()
    print("-" * 120)
    print("Training-vs-held-out fold distribution [interpretive guardrail]")
    print("-" * 120)
    print(
        train_fit_summary[
            [
                "model",
                "train_c_median",
                "test_c_median",
                "fraction_train_c_ge_0_95",
                "fraction_test_c_lt_0_50",
                "fraction_train_gt_0_50_and_test_lt_0_50",
                "training_fit_interpretation",
            ]
        ].to_string(index=False)
    )

    print()
    print("B1 exact shared outer-split contract:")
    print(
        f"  fraction exact same B1 vs primary test folds: "
        f"{sign_checks['B1_exact_same_outer_test_sets_fraction']:.3f}"
    )
    print(
        "  status: "
        + ("PASS" if b1_split_contract_pass else "DEFECT_SECONDARY_ONLY")
    )

    print()
    print("Premise reporting:")
    print(f"  frozen machine state: {primary_cmp.get('premise_state')}")
    print(f"  assessability: {assessability}")
    print("  external pooled-human source actually used: NO")
    print("  GSE16091 outer-training pool used as human reference: YES")

    print()
    print("mTOR context hypothesis frozen before module ranking: PASS")

    print()
    print("=" * 120)
    print("04d HUMAN RISK-ORIENTATION AUDIT SUMMARY")
    print("=" * 120)
    print(f"Risk-orientation verdict: {orientation_verdict}")
    print(
        "B1 shared-split implementation: "
        + ("PASS" if b1_split_contract_pass else "DEFECT_SECONDARY_ONLY")
    )
    print("Primary B4-vs-B0 premise state changed: NO")
    print()
    if orientation_pass:
        print(
            "Interpretation if PASS: target-fitted models have correct hazard-score "
            "orientation in training/reference implementations; below-0.5 OOF "
            "performance reflects held-out generalization instability/anti-concordance, "
            "not a simple sign-convention inversion."
        )
    else:
        print(
            "HOLD: do not proceed to 05c until the orientation discrepancy is resolved."
        )
    print("=" * 120)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 120, file=sys.stderr)
        print("04d human risk-orientation audit: FAIL", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print("=" * 120, file=sys.stderr)
        raise
