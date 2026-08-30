#!/usr/bin/env python3
"""
Paper 6 - materialize post-HOLD controlled-mechanism new-seed recipes.

Stage 05e1 follows the successfully frozen 05e0 contract.

Scientific role
---------------
Materialize the 12 focused controlled-mechanism cells and their 6,000 entirely
new replicate seeds. This stage DOES NOT generate expression/outcome matrices
and DOES NOT fit any model.

The actual synthetic matrices remain generated on demand in 05e2 from the
already-authoritative 05b generator using:
    (frozen cell recipe, frozen replicate seed, target_test_n=500)

This keeps 05e1 lightweight while preserving exact replayability.

Safety / provenance
-------------------
- Requires the exact already-run 05e0 contract SHA256:
    05cd5bc9180909c4c76f4e9a086a6fa1a2c37faee2f2675008e6e1dd91c581fb
- Verifies the 05e0 scenario registry hash through its summary.
- Verifies all 6,000 seeds are unique.
- Verifies zero overlap with every original 05b replicate seed.
- Verifies no duplicate scientific cell exists.
- Writes one immutable NPZ recipe per cell plus manifest/hashes.
- Performs one deterministic 05b generator smoke replay per cell WITHOUT saving
  generated matrices, to prove each recipe is executable under the authoritative
  generator before 05e2.

No real DOG2/human values.
No reserved human outcomes.
No model fitting.
No GPU.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


SCRIPT_VERSION = "05e1-materialize-posthold-controlled-recipes-v1-no-cli"

ROOT = Path(__file__).resolve().parents[1]

# 05e0 frozen controlled-mechanism contract.
E0_DIR = ROOT / "results" / "posthold_controlled_mechanism" / "05e0"
E0_CONTRACT = E0_DIR / "controlled_mechanism_contract.json"
E0_SCENARIOS = E0_DIR / "controlled_mechanism_scenario_registry.tsv"
E0_MODELS = E0_DIR / "controlled_mechanism_model_registry.tsv"
E0_CONTRASTS = E0_DIR / "controlled_mechanism_contrast_registry.tsv"
E0_SUMMARY = E0_DIR / "summary.json"

EXPECTED_E0_CONTRACT_SHA256 = (
    "05cd5bc9180909c4c76f4e9a086a6fa1a2c37faee2f2675008e6e1dd91c581fb"
)
EXPECTED_E0_STATUS = (
    "PASS_POSTHOLD_CONTROLLED_MECHANISM_EXPERIMENT_FROZEN"
)

# Authoritative original synthetic generator / recipes.
B_DIR = ROOT / "simulations" / "05b"
B_GENERATOR_CONTRACT = B_DIR / "generator_contract.json"
B_ORIGINAL_MANIFEST = B_DIR / "scenario_recipe_manifest.tsv"

# New 05e1 materialization.
OUT_DIR = ROOT / "simulations" / "05e_controlled_mechanism"
RECIPE_DIR = OUT_DIR / "recipes"
RESULT_DIR = ROOT / "results" / "posthold_controlled_mechanism" / "05e1"

for d in [OUT_DIR, RECIPE_DIR, RESULT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

MANIFEST = RESULT_DIR / "controlled_recipe_manifest.tsv"
CELL_AUDIT = RESULT_DIR / "controlled_recipe_cell_audit.tsv"
GENERATOR_SMOKE = RESULT_DIR / "generator_smoke_replay.tsv"
SUMMARY_JSON = RESULT_DIR / "summary.json"

EXPECTED_CELLS = 12
EXPECTED_REPLICATES_PER_CELL = 500
EXPECTED_TOTAL_REPLICATES = 6000
EXPECTED_TARGET_TEST_N = 500


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_file(path: Path) -> Path:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Required artifact missing: {path}")
    return path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def atomic_savez_compressed(path: Path, **arrays: np.ndarray) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    tmp.replace(path)


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import Python module from {path}")

    module = importlib.util.module_from_spec(spec)
    prior = sys.modules.get(module_name)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        if prior is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = prior
        raise

    return module


def scalar_text(value: Any) -> np.ndarray:
    return np.asarray(str(value), dtype="U128")


def scalar_float(value: Any) -> np.ndarray:
    return np.asarray(float(value), dtype=np.float64)


def scalar_int(value: Any) -> np.ndarray:
    return np.asarray(int(value), dtype=np.int64)


def scenario_payload(row: pd.Series) -> Dict[str, Any]:
    """
    Generator-facing row.

    Keep both the symbolic labels and the numerical fields present in 05e0.
    Extra administrative 05e0 columns are intentionally excluded.
    """
    return {
        "target_events": int(row["target_events"]),
        "transfer_regime": str(row["transfer_regime"]),
        "transferable_fraction": float(row["transferable_fraction"]),
        "effect_concordance": float(row["effect_concordance"]),
        "nontransferable_sign_flip_fraction": float(
            row["nontransferable_sign_flip_fraction"]
        ),
        "covariance_shift": str(row["covariance_shift"]),
        "covariance_shift_value": float(row["covariance_shift_value"]),
        "censoring": str(row["censoring"]),
        "censoring_fraction": float(row["censoring_fraction"]),
        "mapping_error": str(row["mapping_error"]),
        "mapping_error_fraction": float(row["mapping_error_fraction"]),
        "source_prior_state": str(row["source_prior_state"]),
        "replicates": int(row["replicates"]),
        "target_test_n": int(row["target_test_n"]),
    }


def extract_seed_column(recipe_path: Path) -> np.ndarray:
    """
    Read the original authoritative 05b seed vector.

    Original 05b recipe files are known to contain replicate_seed. Fail rather
    than silently guessing another field name.
    """
    with np.load(recipe_path, allow_pickle=False) as data:
        if "replicate_seed" not in data.files:
            raise RuntimeError(
                f"Original 05b recipe lacks replicate_seed: {recipe_path}"
            )
        seeds = np.asarray(data["replicate_seed"], dtype=np.int64)

    if seeds.ndim != 1:
        raise RuntimeError(
            f"Original seed vector is not 1-D: {recipe_path}"
        )

    return seeds


def smoke_validate_generated(
    generated: Dict[str, Any],
    *,
    expected_events: int,
    expected_test_n: int,
) -> Dict[str, Any]:
    """
    Minimal structural validation only.

    We deliberately do not inspect model performance or perform any scientific
    selection. This merely proves the frozen recipe executes under the
    authoritative generator.
    """
    required_keys = [
        "X_target_train",
        "X_target_test",
        "target_train_time",
        "target_train_event",
        "target_test_time",
        "target_test_event",
        "beta_target",
        "mapping_index",
    ]

    missing = [k for k in required_keys if k not in generated]
    if missing:
        raise RuntimeError(
            "Authoritative 05b generator smoke replay missing keys: "
            + ", ".join(missing)
        )

    x_train = np.asarray(generated["X_target_train"])
    x_test = np.asarray(generated["X_target_test"])
    train_event = np.asarray(generated["target_train_event"])
    test_event = np.asarray(generated["target_test_event"])

    if x_train.ndim != 2 or x_test.ndim != 2:
        raise RuntimeError("Generated target matrices are not 2-D.")

    if x_test.shape[0] != expected_test_n:
        raise RuntimeError(
            f"Generated target test n={x_test.shape[0]}, "
            f"expected={expected_test_n}."
        )

    if x_train.shape[1] != x_test.shape[1]:
        raise RuntimeError("Train/test generated feature counts differ.")

    observed_train_events = int(np.asarray(train_event, dtype=np.uint8).sum())

    if observed_train_events != expected_events:
        raise RuntimeError(
            f"Generated training events={observed_train_events}, "
            f"expected exact target_events={expected_events}."
        )

    if len(test_event) != expected_test_n:
        raise RuntimeError("Generated target test-event length mismatch.")

    if not np.isfinite(x_train).all() or not np.isfinite(x_test).all():
        raise RuntimeError("Generated expression contains non-finite values.")

    return {
        "target_train_n": int(x_train.shape[0]),
        "target_train_events": observed_train_events,
        "target_test_n": int(x_test.shape[0]),
        "n_modules": int(x_train.shape[1]),
        "target_test_events": int(
            np.asarray(test_event, dtype=np.uint8).sum()
        ),
    }


def main() -> None:
    started = now_utc()

    print("=" * 120)
    print("Paper 6 - materialize 05e controlled-mechanism new-seed recipes")
    print("=" * 120)
    print(f"Script version: {SCRIPT_VERSION}")
    print()
    print("Scope:")
    print("  05e0 frozen contract read: YES")
    print("  New controlled recipe seeds materialized: YES")
    print("  Full synthetic expression/outcome matrices saved: NO")
    print("  One generator smoke replay per cell: YES")
    print("  Model fitting: NO")
    print("  Human outcomes read: NO")
    print("  GPU execution: NO")
    print()

    for path in [
        E0_CONTRACT,
        E0_SCENARIOS,
        E0_MODELS,
        E0_CONTRASTS,
        E0_SUMMARY,
        B_GENERATOR_CONTRACT,
        B_ORIGINAL_MANIFEST,
    ]:
        require_file(path)

    observed_contract_hash = sha256_file(E0_CONTRACT)
    if observed_contract_hash != EXPECTED_E0_CONTRACT_SHA256:
        raise RuntimeError(
            "05e0 contract SHA256 differs from the already-run frozen contract. "
            f"Observed={observed_contract_hash}"
        )

    e0_summary = read_json(E0_SUMMARY)
    if str(e0_summary.get("scientific_status")) != EXPECTED_E0_STATUS:
        raise RuntimeError("05e0 is not in the expected frozen PASS state.")

    frozen_hashes = e0_summary.get("artifact_hashes") or {}

    expected_scenario_hash = str(
        frozen_hashes.get("controlled_mechanism_scenario_registry.tsv", "")
    )
    if not expected_scenario_hash:
        raise RuntimeError(
            "05e0 summary lacks frozen scenario-registry hash."
        )

    if sha256_file(E0_SCENARIOS) != expected_scenario_hash:
        raise RuntimeError(
            "05e0 scenario registry changed after freeze."
        )

    scenarios = pd.read_csv(E0_SCENARIOS, sep="\t")

    if len(scenarios) != EXPECTED_CELLS:
        raise RuntimeError(
            f"05e0 cells={len(scenarios)}, expected={EXPECTED_CELLS}."
        )

    if not np.all(
        scenarios["replicates"].astype(int).to_numpy()
        == EXPECTED_REPLICATES_PER_CELL
    ):
        raise RuntimeError("05e0 replicates/cell changed.")

    if int(scenarios["replicates"].astype(int).sum()) != EXPECTED_TOTAL_REPLICATES:
        raise RuntimeError("05e0 total replicate count changed.")

    if not np.all(
        scenarios["target_test_n"].astype(int).to_numpy()
        == EXPECTED_TARGET_TEST_N
    ):
        raise RuntimeError("05e0 target_test_n changed.")

    # No duplicate scientific cells.
    scientific_cell_cols = [
        "target_events",
        "transfer_regime",
        "covariance_shift",
        "censoring",
        "mapping_error",
        "source_prior_state",
    ]
    if scenarios.duplicated(scientific_cell_cols).any():
        dup = scenarios.loc[
            scenarios.duplicated(scientific_cell_cols, keep=False),
            scientific_cell_cols,
        ]
        raise RuntimeError(
            "Duplicate scientific cells in 05e0 registry:\n"
            + dup.to_string(index=False)
        )

    # ------------------------------------------------------------------
    # Authoritative generator provenance.
    # ------------------------------------------------------------------
    generator_contract = read_json(B_GENERATOR_CONTRACT)
    generator_path = ROOT / str(
        generator_contract["authoritative_generator_script"]
    )
    require_file(generator_path)

    expected_generator_hash = str(
        generator_contract["authoritative_generator_script_sha256"]
    )
    observed_generator_hash = sha256_file(generator_path)

    if observed_generator_hash != expected_generator_hash:
        raise RuntimeError(
            "Authoritative 05b generator script SHA256 mismatch."
        )

    generator = load_module(
        generator_path,
        "paper6_05b_authoritative_generator_for_05e1",
    )

    if not hasattr(generator, "generate_full_replicate"):
        raise RuntimeError(
            "Authoritative 05b generator lacks generate_full_replicate()."
        )

    # ------------------------------------------------------------------
    # Reconstruct original seed universe and prove zero overlap.
    # ------------------------------------------------------------------
    original_manifest = pd.read_csv(
        B_ORIGINAL_MANIFEST,
        sep="\t",
    )

    if "recipe_path" not in original_manifest.columns:
        raise RuntimeError(
            "Original 05b manifest lacks recipe_path column."
        )

    old_seed_set = set()

    for row in original_manifest.itertuples(index=False):
        recipe_path = ROOT / str(row.recipe_path)
        require_file(recipe_path)

        seeds = extract_seed_column(recipe_path)
        old_seed_set.update(seeds.tolist())

    if not old_seed_set:
        raise RuntimeError("Original 05b seed universe is empty.")

    # ------------------------------------------------------------------
    # Materialize exact 05e1 recipe files.
    # ------------------------------------------------------------------
    manifest_rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []
    smoke_rows: List[Dict[str, Any]] = []

    all_new_seeds: List[int] = []

    for i, row in scenarios.reset_index(drop=True).iterrows():
        cell_id = str(row["cell_id"])
        replicates = int(row["replicates"])
        seed_start = int(row["seed_start"])
        seed_end = int(row["seed_end"])

        expected_end = seed_start + replicates - 1
        if seed_end != expected_end:
            raise RuntimeError(
                f"{cell_id}: seed_end={seed_end}, expected={expected_end}."
            )

        seeds = np.arange(
            seed_start,
            seed_end + 1,
            dtype=np.int64,
        )

        if len(seeds) != EXPECTED_REPLICATES_PER_CELL:
            raise RuntimeError(
                f"{cell_id}: seed count={len(seeds)}, expected=500."
            )

        payload = scenario_payload(row)

        recipe_path = RECIPE_DIR / f"{cell_id}.npz"

        # Do not silently overwrite a different preexisting recipe.
        candidate_arrays = {
            "replicate_seed": seeds,
            "cell_id": scalar_text(cell_id),
            "target_events": scalar_int(payload["target_events"]),
            "transfer_regime": scalar_text(payload["transfer_regime"]),
            "transferable_fraction": scalar_float(
                payload["transferable_fraction"]
            ),
            "effect_concordance": scalar_float(
                payload["effect_concordance"]
            ),
            "nontransferable_sign_flip_fraction": scalar_float(
                payload["nontransferable_sign_flip_fraction"]
            ),
            "covariance_shift": scalar_text(
                payload["covariance_shift"]
            ),
            "covariance_shift_value": scalar_float(
                payload["covariance_shift_value"]
            ),
            "censoring": scalar_text(payload["censoring"]),
            "censoring_fraction": scalar_float(
                payload["censoring_fraction"]
            ),
            "mapping_error": scalar_text(
                payload["mapping_error"]
            ),
            "mapping_error_fraction": scalar_float(
                payload["mapping_error_fraction"]
            ),
            "source_prior_state": scalar_text(
                payload["source_prior_state"]
            ),
            "replicates": scalar_int(replicates),
            "target_test_n": scalar_int(
                payload["target_test_n"]
            ),
            "05e0_contract_sha256": scalar_text(
                observed_contract_hash
            ),
            "05b_generator_sha256": scalar_text(
                observed_generator_hash
            ),
        }

        if recipe_path.exists():
            # Validate full deterministic contents before reusing.
            with np.load(recipe_path, allow_pickle=False) as existing:
                existing_names = set(existing.files)
                candidate_names = set(candidate_arrays)

                if existing_names != candidate_names:
                    raise RuntimeError(
                        f"{cell_id}: preexisting recipe schema differs."
                    )

                for key, expected in candidate_arrays.items():
                    observed = np.asarray(existing[key])
                    if not np.array_equal(observed, expected):
                        raise RuntimeError(
                            f"{cell_id}: preexisting recipe differs at {key}."
                        )
        else:
            atomic_savez_compressed(
                recipe_path,
                **candidate_arrays,
            )

        recipe_hash = sha256_file(recipe_path)

        overlap_count = int(
            sum(int(seed) in old_seed_set for seed in seeds)
        )
        if overlap_count != 0:
            raise RuntimeError(
                f"{cell_id}: {overlap_count} seeds overlap original 05b."
            )

        all_new_seeds.extend(seeds.tolist())

        # Generator smoke replay: first frozen seed only.
        smoke_seed = int(seeds[0])
        generated = generator.generate_full_replicate(
            smoke_seed,
            payload,
            target_test_n=int(payload["target_test_n"]),
        )

        smoke = smoke_validate_generated(
            generated,
            expected_events=int(payload["target_events"]),
            expected_test_n=int(payload["target_test_n"]),
        )

        smoke_rows.append(
            {
                "cell_id": cell_id,
                "smoke_seed": smoke_seed,
                "generator_sha256": observed_generator_hash,
                "target_train_n": smoke["target_train_n"],
                "target_train_events": smoke["target_train_events"],
                "target_test_n": smoke["target_test_n"],
                "target_test_events": smoke["target_test_events"],
                "n_modules": smoke["n_modules"],
                "status": "PASS",
            }
        )

        manifest_rows.append(
            {
                "cell_id": cell_id,
                "recipe_path": str(recipe_path.relative_to(ROOT)),
                "recipe_sha256": recipe_hash,
                "replicates": replicates,
                "seed_start": seed_start,
                "seed_end": seed_end,
                "target_events": int(payload["target_events"]),
                "transfer_regime": str(payload["transfer_regime"]),
                "target_test_n": int(payload["target_test_n"]),
                "05e0_contract_sha256": observed_contract_hash,
                "05b_generator_sha256": observed_generator_hash,
            }
        )

        audit_rows.append(
            {
                "cell_id": cell_id,
                "cell_index": int(row["cell_index"]),
                "target_events": int(payload["target_events"]),
                "transfer_regime": str(payload["transfer_regime"]),
                "transferable_fraction": float(
                    payload["transferable_fraction"]
                ),
                "effect_concordance": float(
                    payload["effect_concordance"]
                ),
                "nontransferable_sign_flip_fraction": float(
                    payload["nontransferable_sign_flip_fraction"]
                ),
                "covariance_shift": str(
                    payload["covariance_shift"]
                ),
                "censoring": str(payload["censoring"]),
                "mapping_error": str(payload["mapping_error"]),
                "source_prior_state": str(
                    payload["source_prior_state"]
                ),
                "seed_count": len(seeds),
                "old_seed_overlap": overlap_count,
                "recipe_sha256": recipe_hash,
                "generator_smoke_status": "PASS",
            }
        )

        print(
            f"  {cell_id}: PASS recipe "
            f"[events={payload['target_events']}, "
            f"{payload['transfer_regime']}, "
            f"seeds={seed_start}-{seed_end}]"
        )

    # ------------------------------------------------------------------
    # Global new-seed uniqueness and cross-checks.
    # ------------------------------------------------------------------
    new_seed_arr = np.asarray(all_new_seeds, dtype=np.int64)

    if len(new_seed_arr) != EXPECTED_TOTAL_REPLICATES:
        raise RuntimeError(
            f"Materialized new seeds={len(new_seed_arr)}, expected=6000."
        )

    unique_new = np.unique(new_seed_arr)
    if len(unique_new) != EXPECTED_TOTAL_REPLICATES:
        raise RuntimeError(
            "New 05e seed namespace contains duplicates."
        )

    global_overlap = set(unique_new.tolist()).intersection(
        old_seed_set
    )

    if global_overlap:
        sample = sorted(global_overlap)[:10]
        raise RuntimeError(
            f"Global 05e/05b seed overlap found: {sample}"
        )

    manifest_df = pd.DataFrame(manifest_rows)
    audit_df = pd.DataFrame(audit_rows)
    smoke_df = pd.DataFrame(smoke_rows)

    if len(manifest_df) != EXPECTED_CELLS:
        raise RuntimeError("05e1 manifest row count mismatch.")

    if not (audit_df["old_seed_overlap"] == 0).all():
        raise RuntimeError("At least one 05e1 cell has old seed overlap.")

    if not (smoke_df["status"] == "PASS").all():
        raise RuntimeError("At least one generator smoke replay failed.")

    manifest_df.to_csv(
        MANIFEST,
        sep="\t",
        index=False,
    )
    audit_df.to_csv(
        CELL_AUDIT,
        sep="\t",
        index=False,
    )
    smoke_df.to_csv(
        GENERATOR_SMOKE,
        sep="\t",
        index=False,
    )

    summary = {
        "script_version": SCRIPT_VERSION,
        "status": "PASS",
        "scientific_status": (
            "PASS_CONTROLLED_MECHANISM_NEW_SEED_RECIPES_MATERIALIZED"
        ),
        "run_started_utc": started,
        "run_finished_utc": now_utc(),
        "05e0_contract_sha256": observed_contract_hash,
        "05e0_scenario_registry_sha256": sha256_file(
            E0_SCENARIOS
        ),
        "05b_generator_contract_sha256": sha256_file(
            B_GENERATOR_CONTRACT
        ),
        "05b_authoritative_generator_sha256": observed_generator_hash,
        "n_cells": EXPECTED_CELLS,
        "replicates_per_cell": EXPECTED_REPLICATES_PER_CELL,
        "total_new_replicates": EXPECTED_TOTAL_REPLICATES,
        "unique_new_seeds": len(unique_new),
        "old_seed_universe_size": len(old_seed_set),
        "old_new_seed_overlap": 0,
        "generator_smoke_replays": EXPECTED_CELLS,
        "full_synthetic_matrices_saved": False,
        "real_data_values_read": False,
        "human_outcomes_read": False,
        "model_fitting": False,
        "GPU_execution": False,
        "next": (
            "05e2 run the frozen M0-M6 controlled model matrix using only these "
            "05e1 recipes and the authoritative 05b generator."
        ),
        "artifact_hashes": {
            "controlled_recipe_manifest.tsv": sha256_file(
                MANIFEST
            ),
            "controlled_recipe_cell_audit.tsv": sha256_file(
                CELL_AUDIT
            ),
            "generator_smoke_replay.tsv": sha256_file(
                GENERATOR_SMOKE
            ),
        },
    }

    write_json(
        SUMMARY_JSON,
        summary,
    )

    print()
    print("=" * 120)
    print("05e1 CONTROLLED NEW-SEED RECIPE MATERIALIZATION")
    print("=" * 120)
    print(f"Cells materialized: {len(manifest_df)}/{EXPECTED_CELLS}")
    print(
        f"New replicate seeds: {len(unique_new):,}/"
        f"{EXPECTED_TOTAL_REPLICATES:,}"
    )
    print(f"Old 05b seed universe: {len(old_seed_set):,}")
    print("Old/new seed overlap: 0")
    print(
        f"Generator smoke replays: "
        f"{int((smoke_df['status'] == 'PASS').sum())}/"
        f"{EXPECTED_CELLS} PASS"
    )
    print("Full synthetic matrices saved: NO")
    print("Model fitting: NO")
    print("Human outcomes read: NO")
    print("GPU execution: NO")
    print()
    print(f"Recipe manifest SHA256: {sha256_file(MANIFEST)}")
    print("=" * 120)
    print("05e1: PASS_CONTROLLED_MECHANISM_NEW_SEED_RECIPES_MATERIALIZED")
    print("=" * 120)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 120, file=sys.stderr)
        print(
            "05e1 controlled recipe materialization: FAIL",
            file=sys.stderr,
        )
        print(
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        print("=" * 120, file=sys.stderr)
        raise
