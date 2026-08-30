#!/usr/bin/env python3
"""
Paper 6 - materialize deterministic known-truth simulation recipes/shards.

This stage implements the already-passed 05a simulation contract WITHOUT model
fitting and WITHOUT reading any real DOG2/human expression or outcome values.

Why recipe shards instead of full matrices?
--------------------------------------------
05a freezes 180 scenarios and 21,600 total simulation replicates. Persisting the
full source + target matrices for every replicate would consume substantial disk
space while providing no scientific benefit, because every replicate can be
generated exactly from a frozen deterministic seed.

05b therefore materializes:
1. one compressed KNOWN-TRUTH RECIPE shard per scenario containing the exact
   replicate seeds, source/target coefficient truth, transportability truth,
   evolutionary-prior scores, and mapping-corruption maps;
2. a replicate-level truth summary table;
3. representative FULL-DATA fixtures spanning easy/partial/misleading/stress
   regimes, with exact replay verification;
4. one authoritative generator contract. 05c must import/reuse this generator
   rather than reimplement it.

No real outcomes/expression values are read.
No model fitting.
No network access.
CPU only.
No CLI arguments.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


SCRIPT_VERSION = "05b-materialize-known-truth-simulation-recipes-v1-no-cli"
GENERATOR_VERSION = "paper6-known-truth-module-survival-generator-v1"

ROOT = Path(__file__).resolve().parents[1]

A_DIR = ROOT / "results" / "simulation_contract" / "05a"
A_CONTRACT = A_DIR / "negative_transfer_simulation_contract.json"
A_SUMMARY = A_DIR / "summary.json"
A_SCENARIOS = A_DIR / "simulation_scenario_registry.tsv"
A_MODELS = A_DIR / "closed_model_registry.tsv"
A_SELECTION = A_DIR / "architecture_selection_rules.tsv"

OUT_DIR = ROOT / "simulations" / "05b"
RECIPE_DIR = OUT_DIR / "recipes"
FIXTURE_DIR = OUT_DIR / "fixtures"

for directory in [OUT_DIR, RECIPE_DIR, FIXTURE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

SCENARIO_MANIFEST = OUT_DIR / "scenario_recipe_manifest.tsv"
TRUTH_SUMMARY = OUT_DIR / "replicate_truth_summary.tsv"
FIXTURE_MANIFEST = OUT_DIR / "fixture_manifest.tsv"
GENERATOR_CONTRACT = OUT_DIR / "generator_contract.json"
SUMMARY_JSON = OUT_DIR / "summary.json"

EXPECTED_N_MODULES = 50
EXPECTED_N_CAUSAL = 10
EXPECTED_SOURCE_N = 186
EXPECTED_SOURCE_EVENTS = 124
EXPECTED_EVENT_GRID = [5, 10, 15, 20, 29, 40]
EXPECTED_N_SCENARIOS = 180

# Evaluation population is independent of target training data. Large enough to
# make simulation evaluation stable, but not materialized to disk except fixtures.
TARGET_TEST_N = 500

# Synthetic PH time scale. This has no relation to observed real-cohort times.
SOURCE_BASELINE_HAZARD = 0.010
TARGET_BASELINE_HAZARD = 0.010

# Source module covariance: five 10-module blocks.
N_BLOCKS = 5
BLOCK_SIZE = 10
SOURCE_WITHIN_BLOCK_CORR = 0.30
SOURCE_GLOBAL_CORR = 0.03
SPD_EIGEN_FLOOR = 1e-5

# Source causal effect distribution; fixed independently of empirical 04c values.
SOURCE_EFFECT_MIN = 0.20
SOURCE_EFFECT_MAX = 0.45

# Transferable target effect multiplicative jitter.
TRANSFER_SCALE_SD = 0.10
TRANSFER_SCALE_MIN = 0.05

# Mapping corruption: within selected corrupted modules, half are dropped and
# the remainder are cyclically mis-mapped.
MAPPING_DROPOUT_FRACTION_OF_CORRUPTED = 0.50

# Full-data fixtures: one replicate from each requested stress pattern.
N_EXPECTED_FIXTURES = 6


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


def sha256_array(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode("utf-8"))
    digest.update(str(arr.shape).encode("utf-8"))
    digest.update(arr.tobytes(order="C"))
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


def atomic_savez_compressed(path: Path, **arrays: np.ndarray) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    tmp.replace(path)


def stable_seed(base_seed: int, scenario_index: int, replicate_index: int) -> int:
    # Large spacing prevents collisions between scenarios while remaining well
    # within PCG64's accepted integer seed domain.
    return int(base_seed) + int(scenario_index) * 1_000_003 + int(replicate_index)


def nearest_target_n(target_events: int, target_censor_fraction: float) -> int:
    if not (0 <= target_censor_fraction < 1):
        raise ValueError("Target censor fraction must be in [0,1).")
    if target_events <= 0:
        raise ValueError("Target event count must be positive.")

    event_fraction = 1.0 - target_censor_fraction
    n = int(round(target_events / event_fraction))
    n = max(n, target_events + 1)
    return n


def source_correlation_matrix() -> np.ndarray:
    p = EXPECTED_N_MODULES

    corr = np.full((p, p), SOURCE_GLOBAL_CORR, dtype=float)
    np.fill_diagonal(corr, 1.0)

    for block in range(N_BLOCKS):
        start = block * BLOCK_SIZE
        stop = start + BLOCK_SIZE
        corr[start:stop, start:stop] = SOURCE_WITHIN_BLOCK_CORR
        np.fill_diagonal(corr[start:stop, start:stop], 1.0)

    return make_correlation_spd(corr)


def make_correlation_spd(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    matrix = (matrix + matrix.T) / 2.0

    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, SPD_EIGEN_FLOOR)
    spd = vectors @ np.diag(values) @ vectors.T

    sd = np.sqrt(np.diag(spd))
    corr = spd / np.outer(sd, sd)
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)
    return corr


def random_target_correlation(
    rng: np.random.Generator,
    source_corr: np.ndarray,
    shift_strength: float,
) -> np.ndarray:
    if shift_strength <= 0:
        return source_corr.copy()

    # Random low-rank covariance structure converted to a correlation matrix.
    latent_dim = 8
    loadings = rng.normal(0, 1, size=(EXPECTED_N_MODULES, latent_dim))
    random_cov = loadings @ loadings.T / latent_dim + np.eye(EXPECTED_N_MODULES)
    random_corr = make_correlation_spd(random_cov)

    mixed = (
        (1.0 - float(shift_strength)) * source_corr
        + float(shift_strength) * random_corr
    )
    return make_correlation_spd(mixed)


def draw_causal_mask(
    rng: np.random.Generator,
) -> np.ndarray:
    mask = np.zeros(EXPECTED_N_MODULES, dtype=bool)
    indices = rng.choice(
        EXPECTED_N_MODULES,
        size=EXPECTED_N_CAUSAL,
        replace=False,
    )
    mask[indices] = True
    return mask


def draw_transportable_mask(
    rng: np.random.Generator,
    transferable_fraction: float,
) -> np.ndarray:
    n = int(round(float(transferable_fraction) * EXPECTED_N_MODULES))
    n = min(max(n, 0), EXPECTED_N_MODULES)

    mask = np.zeros(EXPECTED_N_MODULES, dtype=bool)
    if n > 0:
        idx = rng.choice(EXPECTED_N_MODULES, size=n, replace=False)
        mask[idx] = True
    return mask


def draw_source_beta(
    rng: np.random.Generator,
    causal_mask: np.ndarray,
) -> np.ndarray:
    beta = np.zeros(EXPECTED_N_MODULES, dtype=float)
    idx = np.flatnonzero(causal_mask)

    magnitudes = rng.uniform(
        SOURCE_EFFECT_MIN,
        SOURCE_EFFECT_MAX,
        size=len(idx),
    )
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=len(idx))

    beta[idx] = magnitudes * signs
    return beta


def draw_target_beta(
    rng: np.random.Generator,
    beta_source: np.ndarray,
    causal_mask: np.ndarray,
    transportable_mask: np.ndarray,
    effect_concordance: float,
    sign_flip_fraction: float,
) -> np.ndarray:
    beta_target = np.zeros(EXPECTED_N_MODULES, dtype=float)

    transferable_causal = causal_mask & transportable_mask
    nontransferable_causal = causal_mask & (~transportable_mask)

    idx_t = np.flatnonzero(transferable_causal)
    if len(idx_t):
        # "effect_concordance" is used as the prespecified effect-retention
        # parameter. Transferable effects retain source direction and have a
        # positive jittered magnitude scale.
        center = max(float(effect_concordance), TRANSFER_SCALE_MIN)
        scales = rng.normal(
            loc=center,
            scale=TRANSFER_SCALE_SD,
            size=len(idx_t),
        )
        scales = np.maximum(scales, TRANSFER_SCALE_MIN)
        beta_target[idx_t] = beta_source[idx_t] * scales

    idx_nt = np.flatnonzero(nontransferable_causal)
    if len(idx_nt):
        n_flip = int(round(float(sign_flip_fraction) * len(idx_nt)))
        n_flip = min(max(n_flip, 0), len(idx_nt))

        if n_flip > 0:
            flip_idx = rng.choice(idx_nt, size=n_flip, replace=False)

            # Misleading/nontransportable effects reverse source direction.
            # Magnitude jitter is outcome-independent and prespecified.
            flip_scale = np.maximum(
                rng.normal(1.0, 0.10, size=n_flip),
                TRANSFER_SCALE_MIN,
            )
            beta_target[flip_idx] = -beta_source[flip_idx] * flip_scale

        # Remaining nontransferable causal modules are attenuated to zero.

    return beta_target


def draw_prior_score(
    rng: np.random.Generator,
    transportable_mask: np.ndarray,
    prior_state: str,
) -> np.ndarray:
    prior_state = str(prior_state)

    if prior_state == "P0_CORRECT":
        high_a, high_b = 8.0, 2.0
        low_a, low_b = 2.0, 8.0

        scores = np.empty(EXPECTED_N_MODULES, dtype=float)
        n_hi = int(transportable_mask.sum())
        n_lo = EXPECTED_N_MODULES - n_hi

        if n_hi:
            scores[transportable_mask] = rng.beta(high_a, high_b, size=n_hi)
        if n_lo:
            scores[~transportable_mask] = rng.beta(low_a, low_b, size=n_lo)
        return scores

    if prior_state == "P1_UNINFORMATIVE":
        return rng.beta(5.0, 5.0, size=EXPECTED_N_MODULES)

    if prior_state == "P2_MISLEADING":
        scores = np.empty(EXPECTED_N_MODULES, dtype=float)
        n_hi = int(transportable_mask.sum())
        n_lo = EXPECTED_N_MODULES - n_hi

        if n_hi:
            scores[transportable_mask] = rng.beta(2.0, 8.0, size=n_hi)
        if n_lo:
            scores[~transportable_mask] = rng.beta(8.0, 2.0, size=n_lo)
        return scores

    raise ValueError(f"Unknown source prior state: {prior_state}")


def draw_mapping_index(
    rng: np.random.Generator,
    mapping_error_fraction: float,
) -> np.ndarray:
    p = EXPECTED_N_MODULES
    mapping = np.arange(p, dtype=np.int16)

    k = int(round(float(mapping_error_fraction) * p))
    k = min(max(k, 0), p)

    if k == 0:
        return mapping

    corrupted = np.sort(rng.choice(p, size=k, replace=False))

    n_drop = int(round(MAPPING_DROPOUT_FRACTION_OF_CORRUPTED * k))
    n_drop = min(max(n_drop, 0), k)

    if n_drop:
        drop_idx = corrupted[:n_drop]
        mapping[drop_idx] = -1

    perm_idx = corrupted[n_drop:]
    if len(perm_idx) >= 2:
        shifted = np.roll(perm_idx, 1)
        mapping[perm_idx] = shifted.astype(np.int16)
    elif len(perm_idx) == 1:
        # A single non-dropped corruption cannot be meaningfully permuted.
        mapping[perm_idx[0]] = -1

    return mapping


def generate_truth(
    replicate_seed: int,
    scenario: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(int(replicate_seed))

    causal_mask = draw_causal_mask(rng)
    transportable_mask = draw_transportable_mask(
        rng,
        float(scenario["transferable_fraction"]),
    )

    beta_source = draw_source_beta(rng, causal_mask)
    beta_target = draw_target_beta(
        rng,
        beta_source,
        causal_mask,
        transportable_mask,
        float(scenario["effect_concordance"]),
        float(scenario["sign_flip_fraction"]),
    )

    prior_score = draw_prior_score(
        rng,
        transportable_mask,
        str(scenario["source_prior_state"]),
    )

    mapping_index = draw_mapping_index(
        rng,
        float(scenario["mapping_error_fraction"]),
    )

    return {
        "causal_mask": causal_mask.astype(np.uint8),
        "transportable_mask": transportable_mask.astype(np.uint8),
        "beta_source": beta_source.astype(np.float32),
        "beta_target": beta_target.astype(np.float32),
        "prior_score": prior_score.astype(np.float32),
        "mapping_index": mapping_index.astype(np.int16),
    }


def exact_event_survival(
    rng: np.random.Generator,
    X: np.ndarray,
    beta: np.ndarray,
    n_events: int,
    baseline_hazard: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    X = np.asarray(X, dtype=float)
    beta = np.asarray(beta, dtype=float)

    n = X.shape[0]
    if not (0 < n_events < n):
        raise ValueError(
            f"Exact-event survival requires 0 < n_events < n; got {n_events}/{n}."
        )

    eta = np.clip(X @ beta, -8.0, 8.0)
    u = np.clip(rng.uniform(size=n), 1e-12, 1.0 - 1e-12)

    event_time = -np.log(u) / (
        float(baseline_hazard) * np.exp(eta)
    )

    censor_latent = np.clip(
        rng.exponential(scale=1.0, size=n),
        1e-12,
        None,
    )

    ratio = event_time / censor_latent
    ordered = np.sort(ratio)

    lower = float(ordered[n_events - 1])
    upper = float(ordered[n_events])

    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        # Probability-zero tie safety.
        scale = np.nextafter(lower, np.inf)
    else:
        scale = (lower + upper) / 2.0

    censor_time = scale * censor_latent
    event = event_time <= censor_time
    observed_time = np.minimum(event_time, censor_time)

    if int(event.sum()) != int(n_events):
        raise RuntimeError(
            f"Exact-event calibration failed: observed={int(event.sum())}, "
            f"requested={n_events}."
        )

    return (
        observed_time.astype(np.float32),
        event.astype(np.uint8),
        event_time.astype(np.float32),
        float(scale),
    )


def generate_full_replicate(
    replicate_seed: int,
    scenario: Dict[str, Any],
    *,
    target_test_n: int = TARGET_TEST_N,
) -> Dict[str, np.ndarray]:
    # Independent streams make truth invariant to changes in array-generation
    # implementation details.
    truth = generate_truth(replicate_seed, scenario)

    seed_seq = np.random.SeedSequence(int(replicate_seed))
    truth_seq, cov_seq, source_seq, target_train_seq, target_test_seq = (
        seed_seq.spawn(5)
    )

    # generate_truth() used a direct RNG from replicate_seed. The child streams
    # below are only for data generation and therefore do not alter truth.
    rng_cov = np.random.default_rng(cov_seq)
    rng_source = np.random.default_rng(source_seq)
    rng_target_train = np.random.default_rng(target_train_seq)
    rng_target_test = np.random.default_rng(target_test_seq)

    source_corr = source_correlation_matrix()
    target_corr = random_target_correlation(
        rng_cov,
        source_corr,
        float(scenario["covariance_shift_strength"]),
    )

    source_chol = np.linalg.cholesky(source_corr)
    target_chol = np.linalg.cholesky(target_corr)

    X_source = (
        rng_source.normal(
            size=(EXPECTED_SOURCE_N, EXPECTED_N_MODULES)
        )
        @ source_chol.T
    ).astype(np.float32)

    source_time, source_event, source_event_time, source_censor_scale = (
        exact_event_survival(
            rng_source,
            X_source,
            truth["beta_source"],
            EXPECTED_SOURCE_EVENTS,
            SOURCE_BASELINE_HAZARD,
        )
    )

    target_events = int(scenario["target_events"])
    target_censor_fraction = float(scenario["target_censor_fraction"])
    target_train_n = nearest_target_n(
        target_events,
        target_censor_fraction,
    )

    X_target_train = (
        rng_target_train.normal(
            size=(target_train_n, EXPECTED_N_MODULES)
        )
        @ target_chol.T
    ).astype(np.float32)

    (
        target_train_time,
        target_train_event,
        target_train_event_time,
        target_train_censor_scale,
    ) = exact_event_survival(
        rng_target_train,
        X_target_train,
        truth["beta_target"],
        target_events,
        TARGET_BASELINE_HAZARD,
    )

    target_test_events = int(
        round(target_test_n * (1.0 - target_censor_fraction))
    )
    target_test_events = min(
        max(target_test_events, 1),
        target_test_n - 1,
    )

    X_target_test = (
        rng_target_test.normal(
            size=(target_test_n, EXPECTED_N_MODULES)
        )
        @ target_chol.T
    ).astype(np.float32)

    (
        target_test_time,
        target_test_event,
        target_test_event_time,
        target_test_censor_scale,
    ) = exact_event_survival(
        rng_target_test,
        X_target_test,
        truth["beta_target"],
        target_test_events,
        TARGET_BASELINE_HAZARD,
    )

    return {
        **truth,
        "source_corr": source_corr.astype(np.float32),
        "target_corr": target_corr.astype(np.float32),
        "X_source": X_source,
        "source_time": source_time,
        "source_event": source_event,
        "source_event_time_latent": source_event_time,
        "source_censor_scale": np.asarray(
            [source_censor_scale], dtype=np.float64
        ),
        "X_target_train": X_target_train,
        "target_train_time": target_train_time,
        "target_train_event": target_train_event,
        "target_train_event_time_latent": target_train_event_time,
        "target_train_censor_scale": np.asarray(
            [target_train_censor_scale], dtype=np.float64
        ),
        "X_target_test": X_target_test,
        "target_test_time": target_test_time,
        "target_test_event": target_test_event,
        "target_test_event_time_latent": target_test_event_time,
        "target_test_censor_scale": np.asarray(
            [target_test_censor_scale], dtype=np.float64
        ),
    }


def causal_effect_correlation(
    beta_source: np.ndarray,
    beta_target: np.ndarray,
    causal_mask: np.ndarray,
) -> float:
    idx = np.flatnonzero(np.asarray(causal_mask, dtype=bool))

    if len(idx) < 2:
        return float("nan")

    x = np.asarray(beta_source, dtype=float)[idx]
    y = np.asarray(beta_target, dtype=float)[idx]

    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")

    return float(np.corrcoef(x, y)[0, 1])


def select_fixture_scenarios(scenarios: pd.DataFrame) -> List[str]:
    selected: List[str] = []

    def pick(mask: pd.Series, label: str) -> str:
        subset = scenarios[mask]
        if subset.empty:
            raise RuntimeError(f"Could not select fixture scenario: {label}")
        return str(subset.iloc[0]["scenario_id"])

    selected.append(
        pick(
            (scenarios["family"] == "CORE_PHASE_DIAGRAM")
            & (scenarios["target_events"] == 5)
            & (scenarios["transfer_regime"] == "R0_FULLY_TRANSPORTABLE"),
            "easy-low-event",
        )
    )

    selected.append(
        pick(
            (scenarios["family"] == "CORE_PHASE_DIAGRAM")
            & (scenarios["target_events"] == 15)
            & (scenarios["transfer_regime"] == "R2_PARTIALLY_TRANSPORTABLE"),
            "partial-15-event",
        )
    )

    selected.append(
        pick(
            (scenarios["family"] == "CORE_PHASE_DIAGRAM")
            & (scenarios["target_events"] == 29)
            & (scenarios["transfer_regime"] == "R5_MISLEADING_SOURCE"),
            "misleading-29-event",
        )
    )

    selected.append(
        pick(
            (scenarios["family"] == "SHIFT_CENSOR_STRESS")
            & (scenarios["target_events"] == 15)
            & (scenarios["transfer_regime"] == "R2_PARTIALLY_TRANSPORTABLE")
            & (scenarios["covariance_shift"] == "S2_SEVERE")
            & (scenarios["censoring"] == "C2_HIGH"),
            "severe-shift-high-censor",
        )
    )

    selected.append(
        pick(
            (scenarios["family"] == "MAPPING_PRIOR_STRESS")
            & (scenarios["target_events"] == 29)
            & (scenarios["transfer_regime"] == "R2_PARTIALLY_TRANSPORTABLE")
            & (scenarios["mapping_error"] == "M2_SEVERE")
            & (scenarios["source_prior_state"] == "P2_MISLEADING"),
            "mapping-misleading-prior",
        )
    )

    selected.append(
        pick(
            (scenarios["family"] == "MAPPING_PRIOR_STRESS")
            & (scenarios["target_events"] == 40)
            & (scenarios["transfer_regime"] == "R5_MISLEADING_SOURCE")
            & (scenarios["mapping_error"] == "M2_SEVERE")
            & (scenarios["source_prior_state"] == "P2_MISLEADING"),
            "worst-case",
        )
    )

    if len(set(selected)) != N_EXPECTED_FIXTURES:
        raise RuntimeError("Fixture scenario selection is not unique.")

    return selected


def main() -> None:
    started = now_utc()

    print("=" * 120)
    print("Paper 6 - materialize deterministic known-truth simulation recipes/shards")
    print("=" * 120)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Generator version: {GENERATOR_VERSION}")
    print()
    print("Safety / execution contract:")
    print("  Real DOG2 outcome/expression values read: NO")
    print("  Real human outcome/expression values read: NO")
    print("  TARGET/GSE21257/GSE39055 outcomes read: NO")
    print("  Model fitting: NO")
    print("  Network access: NO")
    print("  GPU execution: NO [deterministic data generation is CPU-appropriate]")
    print("  Full matrices materialized for all 21,600 replicates: NO")
    print("  Known-truth recipe shards materialized for all replicates: YES")
    print("  Full-data deterministic fixtures materialized: YES")
    print()

    for path in [
        A_CONTRACT,
        A_SUMMARY,
        A_SCENARIOS,
        A_MODELS,
        A_SELECTION,
    ]:
        require_file(path)

    contract_05a = read_json(A_CONTRACT)
    summary_05a = read_json(A_SUMMARY)

    if clean(contract_05a.get("status")) != "PASS":
        raise RuntimeError("05a contract is not PASS.")
    if clean(summary_05a.get("scientific_status")) != (
        "PASS_NEGATIVE_TRANSFER_SIMULATION_AND_AI_SELECTION_CONTRACT_FROZEN"
    ):
        raise RuntimeError("05a is not in expected frozen PASS state.")

    scenarios = pd.read_csv(A_SCENARIOS, sep="\t")

    if len(scenarios) != EXPECTED_N_SCENARIOS:
        raise RuntimeError(
            f"Scenario count={len(scenarios)}, expected={EXPECTED_N_SCENARIOS}."
        )

    observed_event_grid = sorted(
        int(x) for x in scenarios["target_events"].unique()
    )
    if observed_event_grid != EXPECTED_EVENT_GRID:
        raise RuntimeError(
            f"Target event grid changed: {observed_event_grid}"
        )

    gen = contract_05a.get("generative_structure") or {}
    if int(gen.get("n_modules")) != EXPECTED_N_MODULES:
        raise RuntimeError("05a n_modules changed.")
    if int(gen.get("n_causal_modules")) != EXPECTED_N_CAUSAL:
        raise RuntimeError("05a n_causal_modules changed.")
    if int(gen.get("source_n")) != EXPECTED_SOURCE_N:
        raise RuntimeError("05a source_n changed.")
    if int(gen.get("source_event_target")) != EXPECTED_SOURCE_EVENTS:
        raise RuntimeError("05a source_event_target changed.")

    base_seed = int(contract_05a["replicates"]["base_seed"])

    total_replicates_expected = int(
        scenarios["replicates"].astype(int).sum()
    )

    print(f"Frozen scenarios: {len(scenarios)}")
    print(f"Total frozen replicates: {total_replicates_expected:,}")
    print(f"Base seed: {base_seed}")
    print()

    manifest_rows: List[Dict[str, Any]] = []
    truth_rows: List[Dict[str, Any]] = []

    for scenario_index, row in scenarios.reset_index(drop=True).iterrows():
        scenario = row.to_dict()
        scenario_id = str(scenario["scenario_id"])
        n_reps = int(scenario["replicates"])
        recipe_path = RECIPE_DIR / f"{scenario_id}.npz"

        replicate_seeds = np.empty(n_reps, dtype=np.int64)
        causal = np.empty(
            (n_reps, EXPECTED_N_MODULES),
            dtype=np.uint8,
        )
        transportable = np.empty_like(causal)
        beta_source = np.empty(
            (n_reps, EXPECTED_N_MODULES),
            dtype=np.float32,
        )
        beta_target = np.empty_like(beta_source)
        prior_score = np.empty_like(beta_source)
        mapping_index = np.empty(
            (n_reps, EXPECTED_N_MODULES),
            dtype=np.int16,
        )

        target_train_n = nearest_target_n(
            int(scenario["target_events"]),
            float(scenario["target_censor_fraction"]),
        )
        realized_train_censor_fraction = (
            1.0
            - int(scenario["target_events"]) / target_train_n
        )

        for rep in range(n_reps):
            seed = stable_seed(base_seed, scenario_index, rep)
            truth = generate_truth(seed, scenario)

            replicate_seeds[rep] = seed
            causal[rep] = truth["causal_mask"]
            transportable[rep] = truth["transportable_mask"]
            beta_source[rep] = truth["beta_source"]
            beta_target[rep] = truth["beta_target"]
            prior_score[rep] = truth["prior_score"]
            mapping_index[rep] = truth["mapping_index"]

            causal_mask = truth["causal_mask"].astype(bool)
            trans_mask = truth["transportable_mask"].astype(bool)

            truth_rows.append(
                {
                    "scenario_id": scenario_id,
                    "replicate": rep,
                    "replicate_seed": seed,
                    "target_events": int(scenario["target_events"]),
                    "target_train_n": target_train_n,
                    "target_train_realized_censor_fraction": (
                        realized_train_censor_fraction
                    ),
                    "n_transportable_modules": int(trans_mask.sum()),
                    "n_causal_modules": int(causal_mask.sum()),
                    "n_causal_transportable_modules": int(
                        np.sum(causal_mask & trans_mask)
                    ),
                    "beta_source_l2": float(
                        np.linalg.norm(truth["beta_source"])
                    ),
                    "beta_target_l2": float(
                        np.linalg.norm(truth["beta_target"])
                    ),
                    "causal_source_target_effect_correlation": (
                        causal_effect_correlation(
                            truth["beta_source"],
                            truth["beta_target"],
                            causal_mask,
                        )
                    ),
                    "mean_prior_transportable": (
                        float(np.mean(truth["prior_score"][trans_mask]))
                        if trans_mask.any()
                        else float("nan")
                    ),
                    "mean_prior_nontransportable": (
                        float(np.mean(truth["prior_score"][~trans_mask]))
                        if (~trans_mask).any()
                        else float("nan")
                    ),
                    "n_mapping_dropped": int(
                        np.sum(truth["mapping_index"] < 0)
                    ),
                    "n_mapping_mismapped": int(
                        np.sum(
                            (truth["mapping_index"] >= 0)
                            & (
                                truth["mapping_index"]
                                != np.arange(EXPECTED_N_MODULES)
                            )
                        )
                    ),
                }
            )

        atomic_savez_compressed(
            recipe_path,
            replicate_seed=replicate_seeds,
            causal_mask=causal,
            transportable_mask=transportable,
            beta_source=beta_source,
            beta_target=beta_target,
            prior_score=prior_score,
            mapping_index=mapping_index,
            scenario_index=np.asarray(
                [scenario_index],
                dtype=np.int32,
            ),
            target_train_n=np.asarray(
                [target_train_n],
                dtype=np.int32,
            ),
            target_test_n=np.asarray(
                [TARGET_TEST_N],
                dtype=np.int32,
            ),
        )

        manifest_rows.append(
            {
                "scenario_id": scenario_id,
                "scenario_index": scenario_index,
                "family": str(scenario["family"]),
                "target_events": int(scenario["target_events"]),
                "replicates": n_reps,
                "target_train_n": target_train_n,
                "target_test_n": TARGET_TEST_N,
                "recipe_path": str(recipe_path.relative_to(ROOT)),
                "recipe_sha256": sha256_file(recipe_path),
                "recipe_size_bytes": recipe_path.stat().st_size,
            }
        )

        if (
            (scenario_index + 1) % 20 == 0
            or scenario_index + 1 == len(scenarios)
        ):
            print(
                f"  recipes: {scenario_index+1}/{len(scenarios)} scenarios "
                f"materialized"
            )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(SCENARIO_MANIFEST, sep="\t", index=False)

    truth_summary = pd.DataFrame(truth_rows)
    truth_summary.to_csv(TRUTH_SUMMARY, sep="\t", index=False)

    if len(truth_summary) != total_replicates_expected:
        raise RuntimeError(
            f"Truth-summary rows={len(truth_summary)}, "
            f"expected={total_replicates_expected}."
        )

    # ------------------------------------------------------------------
    # Representative full-data fixtures + exact replay checks.
    # ------------------------------------------------------------------

    fixture_scenario_ids = select_fixture_scenarios(scenarios)
    fixture_rows: List[Dict[str, Any]] = []

    scenario_index_map = {
        str(row.scenario_id): int(i)
        for i, row in scenarios.reset_index(drop=True).iterrows()
    }

    for fixture_index, scenario_id in enumerate(fixture_scenario_ids):
        scenario_row = scenarios[
            scenarios["scenario_id"].astype(str) == scenario_id
        ]
        if len(scenario_row) != 1:
            raise RuntimeError(
                f"Fixture scenario {scenario_id} is not unique."
            )

        scenario = scenario_row.iloc[0].to_dict()
        scenario_index = scenario_index_map[scenario_id]
        replicate_index = 0
        seed = stable_seed(base_seed, scenario_index, replicate_index)

        generated = generate_full_replicate(
            seed,
            scenario,
            target_test_n=TARGET_TEST_N,
        )

        fixture_path = FIXTURE_DIR / f"fixture_{fixture_index:02d}_{scenario_id}.npz"

        atomic_savez_compressed(
            fixture_path,
            replicate_seed=np.asarray([seed], dtype=np.int64),
            **generated,
        )

        # Immediate replay from frozen generator functions.
        replay = generate_full_replicate(
            seed,
            scenario,
            target_test_n=TARGET_TEST_N,
        )

        array_hashes = {}
        for key, array in generated.items():
            if key not in replay:
                raise RuntimeError(
                    f"Fixture replay missing array {key!r}."
                )
            if not np.array_equal(array, replay[key]):
                raise RuntimeError(
                    f"Fixture replay mismatch for {scenario_id}:{key}"
                )
            array_hashes[key] = sha256_array(array)

        fixture_rows.append(
            {
                "fixture_index": fixture_index,
                "scenario_id": scenario_id,
                "replicate": replicate_index,
                "replicate_seed": seed,
                "fixture_path": str(fixture_path.relative_to(ROOT)),
                "fixture_sha256": sha256_file(fixture_path),
                "fixture_size_bytes": fixture_path.stat().st_size,
                "array_hashes_json": json.dumps(
                    array_hashes,
                    sort_keys=True,
                ),
                "exact_replay_pass": True,
                "source_n": int(generated["X_source"].shape[0]),
                "source_events": int(
                    generated["source_event"].sum()
                ),
                "target_train_n": int(
                    generated["X_target_train"].shape[0]
                ),
                "target_train_events": int(
                    generated["target_train_event"].sum()
                ),
                "target_test_n": int(
                    generated["X_target_test"].shape[0]
                ),
                "target_test_events": int(
                    generated["target_test_event"].sum()
                ),
            }
        )

    fixture_manifest = pd.DataFrame(fixture_rows)
    fixture_manifest.to_csv(FIXTURE_MANIFEST, sep="\t", index=False)

    if len(fixture_manifest) != N_EXPECTED_FIXTURES:
        raise RuntimeError(
            f"Fixture count={len(fixture_manifest)}, "
            f"expected={N_EXPECTED_FIXTURES}."
        )
    if not fixture_manifest["exact_replay_pass"].all():
        raise RuntimeError("At least one full-data fixture failed replay.")

    # ------------------------------------------------------------------
    # Generator contract.
    # ------------------------------------------------------------------

    script_path = Path(__file__).resolve()

    generator_contract = {
        "script_version": SCRIPT_VERSION,
        "generator_version": GENERATOR_VERSION,
        "status": "PASS",
        "scientific_status": (
            "PASS_KNOWN_TRUTH_SIMULATION_RECIPES_AND_GENERATOR_FROZEN"
        ),
        "created_utc": now_utc(),
        "authoritative_generator_script": str(
            script_path.relative_to(ROOT)
        ),
        "authoritative_generator_script_sha256": sha256_file(script_path),
        "05a_contract_sha256": sha256_file(A_CONTRACT),
        "05a_scenario_registry_sha256": sha256_file(A_SCENARIOS),
        "scenario_recipe_manifest_sha256": sha256_file(SCENARIO_MANIFEST),
        "replicate_truth_summary_sha256": sha256_file(TRUTH_SUMMARY),
        "fixture_manifest_sha256": sha256_file(FIXTURE_MANIFEST),
        "simulation_scale": {
            "n_scenarios": int(len(scenarios)),
            "total_replicates": total_replicates_expected,
            "n_modules": EXPECTED_N_MODULES,
            "n_causal_modules": EXPECTED_N_CAUSAL,
            "source_n": EXPECTED_SOURCE_N,
            "source_events": EXPECTED_SOURCE_EVENTS,
            "target_event_grid": EXPECTED_EVENT_GRID,
            "target_test_n": TARGET_TEST_N,
        },
        "data_generation": {
            "module_distribution": "multivariate_normal",
            "source_covariance": {
                "blocks": N_BLOCKS,
                "block_size": BLOCK_SIZE,
                "within_block_correlation": SOURCE_WITHIN_BLOCK_CORR,
                "global_correlation": SOURCE_GLOBAL_CORR,
            },
            "target_covariance_shift": (
                "convex blend of source correlation with a random low-rank "
                "correlation matrix at the frozen scenario shift strength"
            ),
            "source_effect_magnitude": [
                SOURCE_EFFECT_MIN,
                SOURCE_EFFECT_MAX,
            ],
            "source_baseline_hazard": SOURCE_BASELINE_HAZARD,
            "target_baseline_hazard": TARGET_BASELINE_HAZARD,
            "exact_event_count_calibration": (
                "Independent exponential censoring latent variables are scaled "
                "between ordered event-time/censor-latent ratios so the requested "
                "observed event count is exact. No real-data times are used."
            ),
            "target_train_n_rule": (
                "round(target_events / (1-target_censor_fraction)), "
                "at least target_events+1"
            ),
            "mapping_corruption_rule": (
                "corrupt frozen fraction; approximately half dropped (-1), "
                "remaining corrupted module indices cyclically mis-mapped"
            ),
            "evolutionary_prior_rule": {
                "P0_CORRECT": (
                    "Beta(8,2) transportable; Beta(2,8) nontransportable"
                ),
                "P1_UNINFORMATIVE": "Beta(5,5) all modules",
                "P2_MISLEADING": (
                    "Beta(2,8) transportable; Beta(8,2) nontransportable"
                ),
            },
        },
        "storage_policy": {
            "full_matrices_for_all_replicates": False,
            "reason": (
                "Disk-efficient exact regeneration from frozen recipes/seeds; "
                "05c must generate arrays in memory from this authoritative generator."
            ),
            "recipe_files": int(len(manifest)),
            "full_data_fixtures": N_EXPECTED_FIXTURES,
        },
        "05c_requirements": {
            "must_import_authoritative_05b_generator": True,
            "must_verify_generator_script_sha256": True,
            "must_verify_recipe_manifest_sha256": True,
            "must_replay_fixture_before_model_matrix": True,
            "may_change_scientific_generator_parameters": False,
        },
        "safety": {
            "real_DOG2_values_read": False,
            "real_human_values_read": False,
            "model_fitting": False,
            "network_access": False,
            "GPU_execution": False,
        },
    }
    write_json(GENERATOR_CONTRACT, generator_contract)

    total_recipe_bytes = int(
        manifest["recipe_size_bytes"].sum()
    )
    total_fixture_bytes = int(
        fixture_manifest["fixture_size_bytes"].sum()
    )

    summary = {
        "script_version": SCRIPT_VERSION,
        "generator_version": GENERATOR_VERSION,
        "status": "PASS",
        "scientific_status": generator_contract["scientific_status"],
        "run_started_utc": started,
        "run_finished_utc": now_utc(),
        "n_scenarios": int(len(scenarios)),
        "total_replicates": total_replicates_expected,
        "recipe_files": int(len(manifest)),
        "recipe_storage_bytes": total_recipe_bytes,
        "full_data_fixtures": int(len(fixture_manifest)),
        "fixture_storage_bytes": total_fixture_bytes,
        "fixture_replay_pass": True,
        "real_data_values_read": False,
        "model_fitting": False,
        "GPU_execution": False,
        "final_artifact_hashes": {
            "scenario_recipe_manifest_tsv": sha256_file(
                SCENARIO_MANIFEST
            ),
            "replicate_truth_summary_tsv": sha256_file(
                TRUTH_SUMMARY
            ),
            "fixture_manifest_tsv": sha256_file(
                FIXTURE_MANIFEST
            ),
            "generator_contract_json": sha256_file(
                GENERATOR_CONTRACT
            ),
        },
        "next": (
            "05c run the closed B0/B4/A0-A4 model matrix using on-the-fly "
            "regeneration from frozen 05b recipes."
        ),
    }
    write_json(SUMMARY_JSON, summary)

    print()
    print("-" * 120)
    print("Known-truth recipe materialization")
    print("-" * 120)
    print(f"Scenarios: {len(scenarios)}")
    print(f"Total replicates: {total_replicates_expected:,}")
    print(f"Recipe files: {len(manifest)}")
    print(
        f"Recipe storage: {total_recipe_bytes / (1024**2):.2f} MiB"
    )
    print(f"Full-data fixtures: {len(fixture_manifest)}")
    print(
        f"Fixture storage: {total_fixture_bytes / (1024**2):.2f} MiB"
    )
    print("Fixture exact replay: PASS")
    print()

    print("Synthetic evaluation design:")
    print(f"  source: n={EXPECTED_SOURCE_N}, exact events={EXPECTED_SOURCE_EVENTS}")
    print(
        "  target train: event-count-driven n with exact frozen scenario events"
    )
    print(
        f"  independent target test: n={TARGET_TEST_N} per replicate"
    )
    print("  full matrices generated on-the-fly in 05c: YES")
    print()

    print("=" * 120)
    print("05b KNOWN-TRUTH SIMULATION MATERIALIZATION SUMMARY")
    print("=" * 120)
    print("Real DOG2/human values read: NO")
    print("Model fitting: NO")
    print("GPU execution: NO")
    print("All frozen replicate recipes materialized: YES")
    print("Representative full-data fixture replay: PASS")
    print()
    print("Next: 05c closed classical/AI model matrix.")
    print("=" * 120)
    print("05b known-truth simulation recipes: PASS")
    print("=" * 120)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 120, file=sys.stderr)
        print("05b known-truth simulation recipes: FAIL", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print("=" * 120, file=sys.stderr)
        raise
