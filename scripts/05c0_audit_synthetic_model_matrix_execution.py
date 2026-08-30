#!/usr/bin/env python3
"""
Paper 6 - audit 05c synthetic model-matrix execution completeness.

Read-only structural audit of already-written 05c scenario shards.

Important:
05c did NOT calculate final Uno C / IBS; it stored predictions and losses for
05d. Therefore this audit does not invent a new scientific metric. Instead it
checks the exact preconditions needed for later C-index evaluation:
- all expected model branches exist in every shard;
- all risk predictions are finite;
- test-risk vectors are nonconstant replicate by replicate;
- all branch-specific training losses are finite;
- A3 gate arrays exist, are finite, and remain in [0,1];
- all scenario output hashes match the 05c manifest;
- scenario-level elapsed-time distribution is summarized.

No model fitting.
No GPU.
No architecture selection.
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


SCRIPT_VERSION = "05c0-audit-synthetic-model-matrix-execution-v1-no-cli"

ROOT = Path(__file__).resolve().parents[1]

C5_DIR = ROOT / "results" / "simulation_model_matrix" / "05c"
C5_SUMMARY = C5_DIR / "summary.json"
C5_MANIFEST = C5_DIR / "scenario_output_manifest.tsv"
C5_IMPLEMENTATION = C5_DIR / "model_implementation_contract.json"

OUT_DIR = ROOT / "results" / "simulation_model_matrix_diagnostics" / "05c0"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_AUDIT = OUT_DIR / "model_branch_execution_audit.tsv"
SCENARIO_AUDIT = OUT_DIR / "scenario_execution_audit.tsv"
TIMING_SUMMARY = OUT_DIR / "scenario_timing_summary.tsv"
SUMMARY_JSON = OUT_DIR / "summary.json"

MAIN_MODELS = ["B0", "B4", "A0", "A1", "A2", "A3", "A4"]
ABLATIONS = [
    "A3_NO_EVOLUTION_PRIOR",
    "A3_NO_TARGET_OVERRIDE",
    "A3_PERMUTED_PRIOR",
    "A3_MAPPING_PERMUTATION",
    "A3_SOURCE_OUTCOME_PERMUTATION",
]
ALL_MODELS = MAIN_MODELS + ABLATIONS

EXPECTED_SCENARIOS = 180
EXPECTED_REPLICATES = 21600
EXPECTED_TEST_N = 500
RISK_SD_EPS = 1e-10

LOSS_KEYS = {
    "B0": "train_loss_B0",
    "B4": "train_loss_B4",
    "A0": "train_loss_A0",
    "A1": "train_loss_A1",
    "A2": "train_loss_A2",
    "A3": "train_loss_A3",
    "A4": "train_loss_A4",
    "A3_NO_EVOLUTION_PRIOR": "train_loss_A3_NO_EVOLUTION_PRIOR",
    "A3_NO_TARGET_OVERRIDE": "train_loss_A3_NO_TARGET_OVERRIDE",
    "A3_PERMUTED_PRIOR": "train_loss_A3_PERMUTED_PRIOR",
    "A3_MAPPING_PERMUTATION": "train_loss_A3_MAPPING_PERMUTATION",
    "A3_SOURCE_OUTCOME_PERMUTATION": (
        "train_loss_A3_SOURCE_OUTCOME_PERMUTATION"
    ),
}

GATE_MODELS = {
    "A3": "gate_A3",
    "A3_NO_EVOLUTION_PRIOR": "gate_A3_NO_EVOLUTION_PRIOR",
    "A3_NO_TARGET_OVERRIDE": "gate_A3_NO_TARGET_OVERRIDE",
    "A3_PERMUTED_PRIOR": "gate_A3_PERMUTED_PRIOR",
    "A3_MAPPING_PERMUTATION": "gate_A3_MAPPING_PERMUTATION",
    "A3_SOURCE_OUTCOME_PERMUTATION": "gate_A3_SOURCE_OUTCOME_PERMUTATION",
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_file(path: Path) -> Path:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Required artifact missing: {path}")
    return path


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


def main() -> None:
    started = now_utc()

    print("=" * 120)
    print("Paper 6 - audit 05c synthetic model-matrix execution completeness")
    print("=" * 120)
    print(f"Script version: {SCRIPT_VERSION}")
    print()
    print("Scope:")
    print("  Model fitting: NO")
    print("  GPU execution: NO")
    print("  Scientific metric/architecture selection: NO")
    print("  Scenario shard hashes verified: YES")
    print("  Every saved model branch inspected: YES")
    print()

    for path in [C5_SUMMARY, C5_MANIFEST, C5_IMPLEMENTATION]:
        require_file(path)

    summary_05c = read_json(C5_SUMMARY)
    manifest = pd.read_csv(C5_MANIFEST, sep="\t")

    if str(summary_05c.get("scientific_status")) != (
        "PASS_CLOSED_SYNTHETIC_TRANSFER_MODEL_MATRIX_COMPLETE"
    ):
        raise RuntimeError("05c summary is not in expected PASS state.")

    if len(manifest) != EXPECTED_SCENARIOS:
        raise RuntimeError(
            f"05c manifest scenarios={len(manifest)}, expected={EXPECTED_SCENARIOS}."
        )

    if int(manifest["replicates"].sum()) != EXPECTED_REPLICATES:
        raise RuntimeError("05c manifest total replicate count mismatch.")

    model_counts: Dict[str, Dict[str, float]] = {
        model: {
            "replicates": 0,
            "finite_test_risk_replicates": 0,
            "nonconstant_test_risk_replicates": 0,
            "finite_train_risk_replicates": 0,
            "nonconstant_train_risk_replicates": 0,
            "finite_loss_replicates": 0,
            "gate_replicates": 0,
            "gate_in_unit_interval_replicates": 0,
            "gate_nonconstant_modules_replicates": 0,
        }
        for model in ALL_MODELS
    }

    scenario_rows: List[Dict[str, Any]] = []

    for i, row in enumerate(manifest.itertuples(index=False), start=1):
        scenario_id = str(row.scenario_id)
        output_path = ROOT / str(row.output_path)
        require_file(output_path)

        observed_hash = sha256_file(output_path)
        if observed_hash != str(row.output_sha256):
            raise RuntimeError(f"{scenario_id}: output SHA256 mismatch.")

        with np.load(output_path, allow_pickle=False) as data:
            files = set(data.files)
            n_rep = int(len(data["replicate_seed"]))

            if n_rep != int(row.replicates):
                raise RuntimeError(
                    f"{scenario_id}: shard replicate count={n_rep}, "
                    f"manifest={int(row.replicates)}."
                )

            target_test_time = np.asarray(data["target_test_time"])
            target_test_event = np.asarray(data["target_test_event"])

            if target_test_time.shape != (n_rep, EXPECTED_TEST_N):
                raise RuntimeError(
                    f"{scenario_id}: target_test_time shape={target_test_time.shape}."
                )
            if target_test_event.shape != (n_rep, EXPECTED_TEST_N):
                raise RuntimeError(
                    f"{scenario_id}: target_test_event shape={target_test_event.shape}."
                )
            if not np.isfinite(target_test_time).all() or np.any(target_test_time <= 0):
                raise RuntimeError(f"{scenario_id}: invalid target-test times.")
            if not np.isin(target_test_event, [0, 1]).all():
                raise RuntimeError(f"{scenario_id}: invalid target-test event coding.")

            # Each replicate must contain at least one event and one censored sample
            # for downstream survival discrimination/calibration evaluation.
            event_counts = target_test_event.sum(axis=1)
            metric_outcome_precondition = (
                (event_counts > 0)
                & (event_counts < EXPECTED_TEST_N)
            )

            scenario_all_models_valid = True

            for model in ALL_MODELS:
                train_key = f"risk_train_{model}"
                test_key = f"risk_test_{model}"
                loss_key = LOSS_KEYS[model]

                for key in [train_key, test_key, loss_key]:
                    if key not in files:
                        raise RuntimeError(
                            f"{scenario_id}: expected branch artifact missing: {key}"
                        )

                risk_train = np.asarray(data[train_key], dtype=float)
                risk_test = np.asarray(data[test_key], dtype=float)
                loss = np.asarray(data[loss_key], dtype=float).reshape(-1)

                if risk_test.shape != (n_rep, EXPECTED_TEST_N):
                    raise RuntimeError(
                        f"{scenario_id}:{model} test-risk shape={risk_test.shape}."
                    )
                if risk_train.shape[0] != n_rep:
                    raise RuntimeError(
                        f"{scenario_id}:{model} train-risk replicate dimension mismatch."
                    )
                if loss.shape != (n_rep,):
                    raise RuntimeError(
                        f"{scenario_id}:{model} loss shape={loss.shape}, expected={(n_rep,)}."
                    )

                finite_test = np.isfinite(risk_test).all(axis=1)
                finite_train = np.isfinite(risk_train).all(axis=1)
                nonconstant_test = (
                    np.std(risk_test, axis=1, ddof=0) > RISK_SD_EPS
                ) & finite_test
                nonconstant_train = (
                    np.std(risk_train, axis=1, ddof=0) > RISK_SD_EPS
                ) & finite_train
                finite_loss = np.isfinite(loss)

                c_precondition = (
                    finite_test
                    & nonconstant_test
                    & metric_outcome_precondition
                )

                counts = model_counts[model]
                counts["replicates"] += n_rep
                counts["finite_test_risk_replicates"] += int(finite_test.sum())
                counts["nonconstant_test_risk_replicates"] += int(
                    nonconstant_test.sum()
                )
                counts["finite_train_risk_replicates"] += int(finite_train.sum())
                counts["nonconstant_train_risk_replicates"] += int(
                    nonconstant_train.sum()
                )
                counts["finite_loss_replicates"] += int(finite_loss.sum())

                if model in GATE_MODELS:
                    gate_key = GATE_MODELS[model]
                    if gate_key not in files:
                        raise RuntimeError(
                            f"{scenario_id}: missing A3 gate array {gate_key}"
                        )

                    gate = np.asarray(data[gate_key], dtype=float)
                    if gate.shape != (n_rep, 50):
                        raise RuntimeError(
                            f"{scenario_id}:{model} gate shape={gate.shape}."
                        )

                    gate_finite = np.isfinite(gate).all(axis=1)
                    gate_unit = (
                        gate_finite
                        & (np.min(gate, axis=1) >= -1e-7)
                        & (np.max(gate, axis=1) <= 1.0 + 1e-7)
                    )
                    gate_var = (
                        np.std(gate, axis=1, ddof=0) > RISK_SD_EPS
                    ) & gate_finite

                    counts["gate_replicates"] += n_rep
                    counts["gate_in_unit_interval_replicates"] += int(
                        gate_unit.sum()
                    )
                    counts["gate_nonconstant_modules_replicates"] += int(
                        gate_var.sum()
                    )

                if not c_precondition.all() or not finite_loss.all():
                    scenario_all_models_valid = False

            # Exact-duplicate branch sanity. Identical entire tensors would be
            # suspicious for independently implemented main models.
            main_duplicate_pairs = 0
            for a_idx, model_a in enumerate(MAIN_MODELS):
                arr_a = np.asarray(data[f"risk_test_{model_a}"])
                for model_b in MAIN_MODELS[a_idx + 1:]:
                    arr_b = np.asarray(data[f"risk_test_{model_b}"])
                    if np.array_equal(arr_a, arr_b):
                        main_duplicate_pairs += 1

            scenario_rows.append(
                {
                    "scenario_id": scenario_id,
                    "replicates": n_rep,
                    "elapsed_seconds": float(row.elapsed_seconds),
                    "output_sha256_verified": True,
                    "all_model_C_preconditions_and_losses_valid": (
                        scenario_all_models_valid
                    ),
                    "exact_duplicate_main_model_tensor_pairs": (
                        main_duplicate_pairs
                    ),
                    "target_test_event_count_min": int(event_counts.min()),
                    "target_test_event_count_max": int(event_counts.max()),
                }
            )

        if i % 20 == 0 or i == len(manifest):
            print(f"  audited scenarios: {i}/{len(manifest)}")

    model_rows = []
    for model in ALL_MODELS:
        c = model_counts[model]
        n = int(c["replicates"])

        row = {
            "model": model,
            "replicates": n,
            "fraction_finite_test_risk": (
                c["finite_test_risk_replicates"] / n
            ),
            "fraction_nonconstant_test_risk": (
                c["nonconstant_test_risk_replicates"] / n
            ),
            "fraction_finite_train_risk": (
                c["finite_train_risk_replicates"] / n
            ),
            "fraction_nonconstant_train_risk": (
                c["nonconstant_train_risk_replicates"] / n
            ),
            "fraction_finite_training_loss": (
                c["finite_loss_replicates"] / n
            ),
            "C_index_input_precondition_fraction": (
                c["nonconstant_test_risk_replicates"] / n
            ),
        }

        if model in GATE_MODELS:
            gn = int(c["gate_replicates"])
            row["fraction_gate_in_unit_interval"] = (
                c["gate_in_unit_interval_replicates"] / gn
            )
            row["fraction_gate_nonconstant_across_modules"] = (
                c["gate_nonconstant_modules_replicates"] / gn
            )
        else:
            row["fraction_gate_in_unit_interval"] = np.nan
            row["fraction_gate_nonconstant_across_modules"] = np.nan

        model_rows.append(row)

    model_audit = pd.DataFrame(model_rows)
    scenario_audit = pd.DataFrame(scenario_rows)

    timing_summary = pd.DataFrame(
        [
            {
                "n_scenarios": len(scenario_audit),
                "elapsed_seconds_min": float(
                    scenario_audit["elapsed_seconds"].min()
                ),
                "elapsed_seconds_q25": float(
                    scenario_audit["elapsed_seconds"].quantile(0.25)
                ),
                "elapsed_seconds_median": float(
                    scenario_audit["elapsed_seconds"].median()
                ),
                "elapsed_seconds_q75": float(
                    scenario_audit["elapsed_seconds"].quantile(0.75)
                ),
                "elapsed_seconds_q95": float(
                    scenario_audit["elapsed_seconds"].quantile(0.95)
                ),
                "elapsed_seconds_max": float(
                    scenario_audit["elapsed_seconds"].max()
                ),
                "elapsed_seconds_sum": float(
                    scenario_audit["elapsed_seconds"].sum()
                ),
                "model_specific_timing_recoverable_posthoc": False,
                "reason": (
                    "05c recorded elapsed_seconds per scenario, not per model. "
                    "Scientific branch execution is instead verified by distinct "
                    "saved predictions/losses/gates for each branch."
                ),
            }
        ]
    )

    model_audit.to_csv(MODEL_AUDIT, sep="\t", index=False)
    scenario_audit.to_csv(SCENARIO_AUDIT, sep="\t", index=False)
    timing_summary.to_csv(TIMING_SUMMARY, sep="\t", index=False)

    all_model_preconditions = bool(
        np.allclose(
            model_audit["fraction_finite_test_risk"].to_numpy(dtype=float),
            1.0,
        )
        and np.allclose(
            model_audit["fraction_nonconstant_test_risk"].to_numpy(dtype=float),
            1.0,
        )
        and np.allclose(
            model_audit["fraction_finite_training_loss"].to_numpy(dtype=float),
            1.0,
        )
    )

    all_hashes = bool(scenario_audit["output_sha256_verified"].all())
    no_main_tensor_duplicates = bool(
        (scenario_audit["exact_duplicate_main_model_tensor_pairs"] == 0).all()
    )

    status = (
        "PASS_SYNTHETIC_MODEL_MATRIX_EXECUTION_COMPLETE"
        if all_model_preconditions and all_hashes and no_main_tensor_duplicates
        else "HOLD_SYNTHETIC_MODEL_MATRIX_EXECUTION_AUDIT"
    )

    payload = {
        "script_version": SCRIPT_VERSION,
        "status": "PASS" if status.startswith("PASS") else "HOLD",
        "scientific_status": status,
        "run_started_utc": started,
        "run_finished_utc": now_utc(),
        "n_scenarios": len(scenario_audit),
        "total_replicates": int(model_audit.iloc[0]["replicates"]),
        "all_model_C_index_input_preconditions_valid": all_model_preconditions,
        "all_scenario_hashes_verified": all_hashes,
        "no_exact_duplicate_main_model_prediction_tensors": (
            no_main_tensor_duplicates
        ),
        "actual_C_index_computed_here": False,
        "reason_actual_C_not_computed": (
            "05d is the frozen scientific metric stage; 05c0 is structural only."
        ),
        "model_specific_timing_recoverable": False,
        "scenario_timing_available": True,
        "model_fitting": False,
        "GPU_execution": False,
        "architecture_selection": False,
        "final_artifact_hashes": {
            "model_branch_execution_audit.tsv": sha256_file(MODEL_AUDIT),
            "scenario_execution_audit.tsv": sha256_file(SCENARIO_AUDIT),
            "scenario_timing_summary.tsv": sha256_file(TIMING_SUMMARY),
        },
    }
    write_json(SUMMARY_JSON, payload)

    print()
    print("-" * 120)
    print("Model-branch execution audit")
    print("-" * 120)
    print(
        model_audit[
            [
                "model",
                "replicates",
                "fraction_finite_test_risk",
                "fraction_nonconstant_test_risk",
                "fraction_finite_training_loss",
                "fraction_gate_in_unit_interval",
            ]
        ].to_string(index=False)
    )

    print()
    print("-" * 120)
    print("Scenario timing [recoverable timing granularity]")
    print("-" * 120)
    print(timing_summary.to_string(index=False))

    print()
    print("Important:")
    print(
        "  05c used scenario-batched GPU fitting; 100-200 replicates were "
        "optimized together, not as 100-200 serial training jobs."
    )
    print(
        "  Per-model elapsed time was not recorded in 05c and cannot be "
        "honestly reconstructed after the fact."
    )
    print(
        "  05c0 therefore verifies actual branch artifacts "
        "(predictions/losses/gates) instead of inventing timing."
    )

    print()
    print("=" * 120)
    print(f"05c0 execution audit: {status}")
    print("=" * 120)

    if not status.startswith("PASS"):
        raise RuntimeError(status)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 120, file=sys.stderr)
        print("05c0 synthetic model-matrix execution audit: FAIL", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print("=" * 120, file=sys.stderr)
        raise
