#!/usr/bin/env python3
"""
Paper 6 - audit frozen B0/B4 ridge regularization selection.

Read-only post-premise diagnostic.

This script DOES NOT refit any model, change any alpha, re-evaluate the premise,
or access reserved human outcomes. It reads the already-materialized 04c
outer-fold and inner-CV audit tables and the 04d train/test orientation table.

Questions:
1. What alpha values were selected for B0 and B4 across all 100 outer folds?
2. How often was the lower/upper boundary of the frozen 1e-4..1e4 grid selected?
3. Did the stored inner-CV audit actually support each selected alpha by
   validation Uno C in all four inner folds?
4. Was the frozen tie rule respected: among candidates within 0.005 of the best
   mean inner validation Uno C, select the LARGEST alpha?
5. How do selected alphas relate descriptively to training and held-out C?

No model fitting.
No network.
No GPU.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


SCRIPT_VERSION = "04e-audit-ridge-regularization-selection-v1-no-cli"

ROOT = Path(__file__).resolve().parents[1]

C4_DIR = ROOT / "results" / "human_premise" / "04c"
PRIMARY_FOLDS = C4_DIR / "primary_classical_fold_metrics.tsv"
PRIMARY_TUNING = C4_DIR / "primary_classical_tuning.tsv"

D4_DIR = ROOT / "results" / "human_premise_diagnostics" / "04d"
TRAIN_TEST = D4_DIR / "target_fitted_train_test_orientation.tsv"

OUT_DIR = ROOT / "results" / "human_premise_diagnostics" / "04e"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA_FREQUENCY = OUT_DIR / "B0_B4_chosen_alpha_frequency.tsv"
FOLD_DIAGNOSTICS = OUT_DIR / "B0_B4_fold_regularization_diagnostics.tsv"
MODEL_SUMMARY = OUT_DIR / "B0_B4_regularization_summary.tsv"
SUMMARY_JSON = OUT_DIR / "summary.json"

EXPECTED_MODELS = ["B0", "B4"]
EXPECTED_OUTER_FOLDS_PER_MODEL = 100
EXPECTED_INNER_SPLITS = 4
EXPECTED_ALPHA_GRID = np.logspace(-4, 4, 17)
INNER_TIE_TOLERANCE = 0.005


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path) -> Path:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Required artifact missing: {path}")
    return path


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def close_to_any(value: float, grid: np.ndarray) -> bool:
    return bool(np.any(np.isclose(float(value), grid, rtol=1e-10, atol=1e-12)))


def main() -> None:
    started = now_utc()

    print("=" * 120)
    print("Paper 6 - audit frozen B0/B4 ridge regularization selection")
    print("=" * 120)
    print(f"Script version: {SCRIPT_VERSION}")
    print()
    print("Scope:")
    print("  Model fitting: NO")
    print("  Hyperparameter changes: NO")
    print("  Reserved human outcomes read: NO")
    print("  Existing 04c inner-validation audit read: YES")
    print("  GPU execution: NO")
    print()

    for path in [PRIMARY_FOLDS, PRIMARY_TUNING, TRAIN_TEST]:
        require_file(path)

    folds = pd.read_csv(PRIMARY_FOLDS, sep="\t")
    tuning = pd.read_csv(PRIMARY_TUNING, sep="\t")
    train_test = pd.read_csv(TRAIN_TEST, sep="\t")

    folds = folds[folds["model"].astype(str).isin(EXPECTED_MODELS)].copy()
    tuning = tuning[tuning["model"].astype(str).isin(EXPECTED_MODELS)].copy()
    train_test = train_test[
        train_test["model"].astype(str).isin(EXPECTED_MODELS)
    ].copy()

    observed_grid = np.sort(
        pd.to_numeric(tuning["alpha"], errors="raise").unique()
    )

    if len(observed_grid) != len(EXPECTED_ALPHA_GRID) or not np.allclose(
        observed_grid,
        EXPECTED_ALPHA_GRID,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise RuntimeError(
            "Observed 04c alpha grid differs from frozen 17-point 1e-4..1e4 grid."
        )

    grid_min = float(observed_grid[0])
    grid_max = float(observed_grid[-1])

    diagnostic_rows: List[Dict[str, Any]] = []
    frequency_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for model in EXPECTED_MODELS:
        model_folds = folds[folds["model"].astype(str) == model].copy()

        if len(model_folds) != EXPECTED_OUTER_FOLDS_PER_MODEL:
            raise RuntimeError(
                f"{model}: outer-fold rows={len(model_folds)}, "
                f"expected={EXPECTED_OUTER_FOLDS_PER_MODEL}."
            )

        # Frequency table from authoritative outer-fold selections.
        selected = pd.to_numeric(
            model_folds["chosen_alpha"],
            errors="raise",
        ).to_numpy(dtype=float)

        if not all(close_to_any(x, observed_grid) for x in selected):
            raise RuntimeError(f"{model}: chosen alpha outside frozen grid.")

        counts = pd.Series(selected).value_counts().sort_index()
        for alpha in observed_grid:
            n = int(counts.get(alpha, 0))
            frequency_rows.append(
                {
                    "model": model,
                    "alpha": float(alpha),
                    "log10_alpha": float(np.log10(alpha)),
                    "n_outer_folds_selected": n,
                    "fraction_outer_folds_selected": n / EXPECTED_OUTER_FOLDS_PER_MODEL,
                    "is_lower_grid_boundary": bool(np.isclose(alpha, grid_min)),
                    "is_upper_grid_boundary": bool(np.isclose(alpha, grid_max)),
                }
            )

        for fold_row in model_folds.itertuples(index=False):
            repeat = int(fold_row.repeat)
            outer_fold = int(fold_row.outer_fold)
            chosen_alpha = float(fold_row.chosen_alpha)

            part = tuning[
                (tuning["model"].astype(str) == model)
                & (tuning["repeat"].astype(int) == repeat)
                & (tuning["outer_fold"].astype(int) == outer_fold)
            ].copy()

            expected_rows = len(observed_grid) * EXPECTED_INNER_SPLITS
            if len(part) != expected_rows:
                raise RuntimeError(
                    f"{model} repeat={repeat} fold={outer_fold}: "
                    f"tuning rows={len(part)}, expected={expected_rows}."
                )

            # Independently reconstruct mean validation Uno C from raw inner rows.
            grouped = (
                part.groupby("alpha", as_index=False)
                .agg(
                    recomputed_mean_inner_val_uno_c=("uno_c", "mean"),
                    n_finite_inner=("uno_c", lambda x: int(np.isfinite(
                        pd.to_numeric(x, errors="coerce")
                    ).sum())),
                )
            )
            grouped = grouped[
                grouped["n_finite_inner"] == EXPECTED_INNER_SPLITS
            ].copy()

            if len(grouped) != len(observed_grid):
                raise RuntimeError(
                    f"{model} repeat={repeat} fold={outer_fold}: not all alpha "
                    "candidates have four finite inner validation scores."
                )

            best_score = float(
                grouped["recomputed_mean_inner_val_uno_c"].max()
            )
            candidates = grouped[
                grouped["recomputed_mean_inner_val_uno_c"]
                >= best_score - INNER_TIE_TOLERANCE
            ]
            reconstructed_chosen = float(candidates["alpha"].max())

            tie_rule_pass = bool(
                np.isclose(
                    reconstructed_chosen,
                    chosen_alpha,
                    rtol=1e-10,
                    atol=1e-12,
                )
            )

            chosen_row = grouped[
                np.isclose(
                    grouped["alpha"].to_numpy(dtype=float),
                    chosen_alpha,
                    rtol=1e-10,
                    atol=1e-12,
                )
            ]
            if len(chosen_row) != 1:
                raise RuntimeError("Chosen-alpha tuning row not unique.")

            chosen_inner_score = float(
                chosen_row.iloc[0]["recomputed_mean_inner_val_uno_c"]
            )

            tt = train_test[
                (train_test["model"].astype(str) == model)
                & (train_test["repeat"].astype(int) == repeat)
                & (train_test["outer_fold"].astype(int) == outer_fold)
            ]
            if len(tt) != 1:
                raise RuntimeError(
                    f"{model} repeat={repeat} fold={outer_fold}: "
                    "04d train/test row not unique."
                )

            ttrow = tt.iloc[0]

            diagnostic_rows.append(
                {
                    "model": model,
                    "repeat": repeat,
                    "outer_fold": outer_fold,
                    "chosen_alpha": chosen_alpha,
                    "log10_chosen_alpha": float(np.log10(chosen_alpha)),
                    "lower_grid_boundary_selected": bool(
                        np.isclose(chosen_alpha, grid_min)
                    ),
                    "upper_grid_boundary_selected": bool(
                        np.isclose(chosen_alpha, grid_max)
                    ),
                    "best_mean_inner_validation_uno_c": best_score,
                    "chosen_mean_inner_validation_uno_c": chosen_inner_score,
                    "chosen_gap_from_best": best_score - chosen_inner_score,
                    "n_alpha_candidates_within_tie_tolerance": int(len(candidates)),
                    "reconstructed_tie_rule_alpha": reconstructed_chosen,
                    "frozen_tie_rule_pass": tie_rule_pass,
                    "train_uno_c": float(ttrow["train_uno_c"]),
                    "heldout_uno_c": float(ttrow["test_uno_c_refit"]),
                }
            )

        diag = pd.DataFrame(
            [x for x in diagnostic_rows if x["model"] == model]
        )

        if not diag["frozen_tie_rule_pass"].all():
            raise RuntimeError(f"{model}: frozen alpha tie rule replay failed.")

        frac_min = float(diag["lower_grid_boundary_selected"].mean())
        frac_max = float(diag["upper_grid_boundary_selected"].mean())

        summary_rows.append(
            {
                "model": model,
                "n_outer_folds": len(diag),
                "alpha_min_selected": float(diag["chosen_alpha"].min()),
                "alpha_q25": float(diag["chosen_alpha"].quantile(0.25)),
                "alpha_median": float(diag["chosen_alpha"].median()),
                "alpha_q75": float(diag["chosen_alpha"].quantile(0.75)),
                "alpha_max_selected": float(diag["chosen_alpha"].max()),
                "fraction_lower_grid_boundary": frac_min,
                "fraction_upper_grid_boundary": frac_max,
                "n_unique_selected_alphas": int(diag["chosen_alpha"].nunique()),
                "median_inner_validation_uno_c_at_chosen": float(
                    diag["chosen_mean_inner_validation_uno_c"].median()
                ),
                "median_train_uno_c": float(diag["train_uno_c"].median()),
                "median_heldout_uno_c": float(diag["heldout_uno_c"].median()),
                "all_four_inner_validation_folds_finite": True,
                "frozen_tie_rule_replay_pass": True,
                "selection_criterion": (
                    "mean INNER-VALIDATION Uno C; not training likelihood"
                ),
                "tie_rule": (
                    "within 0.005 of best mean inner-validation Uno C, "
                    "select largest alpha"
                ),
            }
        )

    frequency = pd.DataFrame(frequency_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    summary = pd.DataFrame(summary_rows)

    frequency.to_csv(ALPHA_FREQUENCY, sep="\t", index=False)
    diagnostics.to_csv(FOLD_DIAGNOSTICS, sep="\t", index=False)
    summary.to_csv(MODEL_SUMMARY, sep="\t", index=False)

    payload = {
        "script_version": SCRIPT_VERSION,
        "status": "PASS",
        "scientific_status": "PASS_RIDGE_REGULARIZATION_SELECTION_AUDIT",
        "run_started_utc": started,
        "run_finished_utc": now_utc(),
        "alpha_grid": observed_grid.tolist(),
        "alpha_grid_min": grid_min,
        "alpha_grid_max": grid_max,
        "models": summary.to_dict(orient="records"),
        "model_fitting": False,
        "hyperparameters_changed": False,
        "primary_premise_state_changed": False,
        "reserved_human_outcomes_read": False,
        "final_artifact_hashes": {
            "B0_B4_chosen_alpha_frequency.tsv": sha256_file(ALPHA_FREQUENCY),
            "B0_B4_fold_regularization_diagnostics.tsv": sha256_file(
                FOLD_DIAGNOSTICS
            ),
            "B0_B4_regularization_summary.tsv": sha256_file(MODEL_SUMMARY),
        },
    }
    write_json(SUMMARY_JSON, payload)

    print("-" * 120)
    print("B0/B4 regularization summary")
    print("-" * 120)
    print(
        summary[
            [
                "model",
                "alpha_min_selected",
                "alpha_median",
                "alpha_max_selected",
                "fraction_lower_grid_boundary",
                "fraction_upper_grid_boundary",
                "n_unique_selected_alphas",
                "median_inner_validation_uno_c_at_chosen",
                "median_train_uno_c",
                "median_heldout_uno_c",
            ]
        ].to_string(index=False)
    )

    print()
    print("Selected-alpha frequencies:")
    display = frequency[frequency["n_outer_folds_selected"] > 0].copy()
    print(
        display[
            [
                "model",
                "alpha",
                "n_outer_folds_selected",
                "fraction_outer_folds_selected",
                "is_lower_grid_boundary",
                "is_upper_grid_boundary",
            ]
        ].to_string(index=False)
    )

    print()
    print("Frozen selection implementation:")
    print("  criterion: mean INNER-VALIDATION Uno C")
    print("  training likelihood used to select alpha: NO")
    print("  tie rule: select LARGEST alpha within 0.005 of best validation C")
    print("  tie-rule replay across all B0/B4 outer folds: PASS")
    print()
    print("=" * 120)
    print("04e ridge regularization selection audit: PASS")
    print("=" * 120)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 120, file=sys.stderr)
        print("04e ridge regularization selection audit: FAIL", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print("=" * 120, file=sys.stderr)
        raise
