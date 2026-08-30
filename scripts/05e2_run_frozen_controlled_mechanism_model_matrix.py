#!/usr/bin/env python3
"""
Paper 6 - run frozen post-HOLD controlled transfer-mechanism model matrix.

Stage 05e2 follows:
- 05e0 frozen post-HOLD mechanism contract;
- 05e1 v3 PASS new-seed recipe materialization.

Scientific role
---------------
Fit ONLY the seven frozen controlled branches M0-M6 on the 12 focused,
entirely new-seed cells (6,000 total replicates).

05e2 does NOT evaluate/select a winning model. It stores paired held-out
predictions and fitted mechanism parameters. 05e3 will compute the frozen
paired contrasts.

Critical implementation rule
----------------------------
Do not reimplement the original 05c machinery from memory.

This script imports the exact frozen:
    scripts/05c_run_closed_synthetic_transfer_model_matrix.py
and verifies its SHA256 against:
    results/simulation_model_matrix/05c/model_implementation_contract.json

Then it reuses the frozen 05c:
- target/source standardization,
- batched Cox loss,
- target B0 fitting,
- source MLP fitting,
- A1 source-centered target-head fitting,
- A3 soft-gate fitting,
- A3 prediction function,
- mapping/alignment implementation,
- numerical constants / optimizer semantics.

New controlled branches are minimal interventions around that frozen code:
M0_B0
    exact 05c B0 replay.

M1_SOURCE_NETWORK_ZERO_SHOT
    exact frozen source MLP encoder + exact frozen source head on target;
    no target update.

M2_A1_SOURCE_CENTERED_HEAD
    exact 05c A1 replay.

M3_A1_FREE_HEAD
    same source encoder, same source-head initialization, same A1 epochs/lr;
    ONLY the A1 L2-to-source-head penalty is set to zero.

M4_A3_SOFT_LEARNED
    exact 05c A3 replay.

M5_A3_HARDENED_PREDICTION
    no new fit. Reuse M4's learned gate/residual; at prediction only:
        gate_hard = I(gate_M4 >= 0.5)
    and recompute both source and complementary residual paths.

M6_A3_ORACLE_HARD_GATE
    true synthetic transportable_mask is the fixed binary gate;
    use frozen 05c A3 allow_target_override=False implementation to fit only
    the complementary residual_beta. This is nonimplementable oracle evidence.

No model selection.
No A6 reopening.
No human outcomes.
GPU-first, restart-safe by cell.
No CLI arguments.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
except ImportError as exc:
    raise ImportError(
        "05e2 requires the same CUDA-enabled PyTorch environment used by 05c."
    ) from exc


SCRIPT_VERSION = (
    "05e2-run-frozen-posthold-controlled-mechanism-model-matrix-v1-no-cli"
)
IMPLEMENTATION_VERSION = (
    "paper6-posthold-controlled-mechanism-model-matrix-v1"
)

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Frozen 05e0.
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# PASS 05e1 v3.
# ---------------------------------------------------------------------------
E1_DIR = ROOT / "results" / "posthold_controlled_mechanism" / "05e1_v3"
E1_MANIFEST = E1_DIR / "controlled_recipe_manifest.tsv"
E1_AUDIT = E1_DIR / "controlled_recipe_cell_audit.tsv"
E1_SMOKE = E1_DIR / "generator_smoke_replay.tsv"
E1_SUMMARY = E1_DIR / "summary.json"

EXPECTED_E1_STATUS = (
    "PASS_CONTROLLED_MECHANISM_NEW_SEED_RECIPES_MATERIALIZED"
)
EXPECTED_E1_MANIFEST_SHA256 = (
    "8e2b4e5ade37d2cffdcc9450b21a60bba8598bd76f7eadb31c149ae1c6a3431e"
)

# ---------------------------------------------------------------------------
# Authoritative 05b generator.
# ---------------------------------------------------------------------------
B_DIR = ROOT / "simulations" / "05b"
B_GENERATOR_CONTRACT = B_DIR / "generator_contract.json"

# ---------------------------------------------------------------------------
# Exact frozen 05c implementation.
# ---------------------------------------------------------------------------
C_SCRIPT = (
    ROOT / "scripts" / "05c_run_closed_synthetic_transfer_model_matrix.py"
)
C_DIR = ROOT / "results" / "simulation_model_matrix" / "05c"
C_IMPLEMENTATION = C_DIR / "model_implementation_contract.json"
C_SUMMARY = C_DIR / "summary.json"

EXPECTED_05C_STATUS = (
    "PASS_CLOSED_SYNTHETIC_TRANSFER_MODEL_MATRIX_COMPLETE"
)

# ---------------------------------------------------------------------------
# 05e2 outputs.
# ---------------------------------------------------------------------------
OUT_DIR = ROOT / "results" / "posthold_controlled_mechanism" / "05e2"
CELL_DIR = OUT_DIR / "cells"
CELL_MANIFEST_DIR = OUT_DIR / "cell_manifests"

for d in [OUT_DIR, CELL_DIR, CELL_MANIFEST_DIR]:
    d.mkdir(parents=True, exist_ok=True)

IMPLEMENTATION_CONTRACT = OUT_DIR / "controlled_model_implementation_contract.json"
OUTPUT_MANIFEST = OUT_DIR / "controlled_model_output_manifest.tsv"
SUMMARY_JSON = OUT_DIR / "summary.json"

EXPECTED_CELLS = 12
EXPECTED_REPLICATES_PER_CELL = 500
EXPECTED_TOTAL_REPLICATES = 6000
EXPECTED_SOURCE_N = 186
EXPECTED_SOURCE_EVENTS = 124
EXPECTED_N_MODULES = 50
EXPECTED_TARGET_TEST_N = 500

MODEL_NAMES = [
    "M0_B0",
    "M1_SOURCE_NETWORK_ZERO_SHOT",
    "M2_A1_SOURCE_CENTERED_HEAD",
    "M3_A1_FREE_HEAD",
    "M4_A3_SOFT_LEARNED",
    "M5_A3_HARDENED_PREDICTION",
    "M6_A3_ORACLE_HARD_GATE",
]

# Frozen by 05e0.
A3_HARD_GATE_THRESHOLD = 0.50

# Reuse exact 05c deterministic initialization namespace.
MODEL_BASE_SEED = 20260901
CELL_SEED_STRIDE = 100_003

# GPU safety mirrors 05c.
MIN_RECOMMENDED_FREE_GPU_BYTES = 3 * 1024**3


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def require_file(path: Path) -> Path:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Required frozen artifact missing: {path}")
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


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")

    module = importlib.util.module_from_spec(spec)
    prior = sys.modules.get(name)
    sys.modules[name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        if prior is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = prior
        raise

    return module


def choose_device() -> torch.device:
    request = os.environ.get("PAPER6_SIM_DEVICE", "auto").strip().lower()

    if request not in {"auto", "cuda", "cpu"}:
        raise RuntimeError(
            "PAPER6_SIM_DEVICE must be one of: auto, cuda, cpu."
        )

    if request == "cpu":
        return torch.device("cpu")

    if not torch.cuda.is_available():
        if request == "cuda":
            raise RuntimeError(
                "PAPER6_SIM_DEVICE=cuda requested but CUDA is unavailable."
            )
        raise RuntimeError(
            "CUDA unavailable. 05e2 is GPU-first. "
            "Set PAPER6_SIM_DEVICE=cpu only if CPU execution is intentional."
        )

    device = torch.device("cuda")
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    allow_busy = os.environ.get("PAPER6_ALLOW_BUSY_GPU", "0").strip() == "1"

    if free_bytes < MIN_RECOMMENDED_FREE_GPU_BYTES and not allow_busy:
        raise RuntimeError(
            "CUDA is available but <3 GiB is free. Another GPU job may be active. "
            "Wait for it to finish or set PAPER6_ALLOW_BUSY_GPU=1 only if "
            "concurrent GPU use is intentional."
        )

    return device


def configure_torch(c05, device: torch.device) -> None:
    # Exact original 05c deterministic policy.
    c05.configure_torch(device)


def scalar_from_recipe(recipe: Dict[str, np.ndarray], key: str) -> Any:
    if key not in recipe:
        raise RuntimeError(f"Controlled recipe missing scalar field {key!r}.")
    value = np.asarray(recipe[key])
    if value.size != 1:
        raise RuntimeError(f"Recipe field {key!r} is not scalar.")
    return value.reshape(-1)[0].item()


def load_recipe(path: Path) -> Dict[str, np.ndarray]:
    require_file(path)
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def generator_payload(recipe: Dict[str, np.ndarray]) -> Dict[str, Any]:
    return {
        "target_events": int(scalar_from_recipe(recipe, "target_events")),
        "transferable_fraction": float(
            scalar_from_recipe(recipe, "transferable_fraction")
        ),
        "effect_concordance": float(
            scalar_from_recipe(recipe, "effect_concordance")
        ),
        "sign_flip_fraction": float(
            scalar_from_recipe(recipe, "sign_flip_fraction")
        ),
        "source_prior_state": str(
            scalar_from_recipe(recipe, "source_prior_state")
        ),
        "mapping_error_fraction": float(
            scalar_from_recipe(recipe, "mapping_error_fraction")
        ),
        "covariance_shift_strength": float(
            scalar_from_recipe(recipe, "covariance_shift_strength")
        ),
        "target_censor_fraction": float(
            scalar_from_recipe(recipe, "target_censor_fraction")
        ),
    }


def generate_cell_arrays(
    generator,
    recipe: Dict[str, np.ndarray],
    *,
    cell_id: str,
) -> Dict[str, np.ndarray]:
    seeds = np.asarray(recipe["replicate_seed"], dtype=np.int64)

    if seeds.ndim != 1 or len(seeds) != EXPECTED_REPLICATES_PER_CELL:
        raise RuntimeError(
            f"{cell_id}: expected {EXPECTED_REPLICATES_PER_CELL} recipe seeds."
        )

    scenario = generator_payload(recipe)
    target_test_n = int(scalar_from_recipe(recipe, "target_test_n"))

    if target_test_n != EXPECTED_TARGET_TEST_N:
        raise RuntimeError(
            f"{cell_id}: target_test_n={target_test_n}, "
            f"expected={EXPECTED_TARGET_TEST_N}."
        )

    first = generator.generate_full_replicate(
        int(seeds[0]),
        scenario,
        target_test_n=target_test_n,
    )

    required = [
        "causal_mask",
        "transportable_mask",
        "beta_source",
        "beta_target",
        "prior_score",
        "mapping_index",
        "X_source",
        "source_time",
        "source_event",
        "X_target_train",
        "target_train_time",
        "target_train_event",
        "X_target_test",
        "target_test_time",
        "target_test_event",
    ]
    missing = [k for k in required if k not in first]
    if missing:
        raise RuntimeError(
            f"{cell_id}: generator missing required arrays: {missing}"
        )

    source_n, p = np.asarray(first["X_source"]).shape
    target_train_n, p_target = np.asarray(first["X_target_train"]).shape
    test_n, p_test = np.asarray(first["X_target_test"]).shape

    if source_n != EXPECTED_SOURCE_N:
        raise RuntimeError(
            f"{cell_id}: source n={source_n}, expected={EXPECTED_SOURCE_N}."
        )
    if p != EXPECTED_N_MODULES or p_target != p or p_test != p:
        raise RuntimeError(
            f"{cell_id}: module-count identity failed."
        )
    if test_n != EXPECTED_TARGET_TEST_N:
        raise RuntimeError(
            f"{cell_id}: generated test n={test_n}, expected=500."
        )

    r = len(seeds)

    X_source = np.empty((r, source_n, p), dtype=np.float32)
    source_time = np.empty((r, source_n), dtype=np.float32)
    source_event = np.empty((r, source_n), dtype=np.uint8)

    X_target_train = np.empty(
        (r, target_train_n, p),
        dtype=np.float32,
    )
    target_train_time = np.empty(
        (r, target_train_n),
        dtype=np.float32,
    )
    target_train_event = np.empty(
        (r, target_train_n),
        dtype=np.uint8,
    )

    X_target_test = np.empty(
        (r, test_n, p),
        dtype=np.float32,
    )
    target_test_time = np.empty(
        (r, test_n),
        dtype=np.float32,
    )
    target_test_event = np.empty(
        (r, test_n),
        dtype=np.uint8,
    )

    causal_mask = np.empty((r, p), dtype=np.uint8)
    transportable_mask = np.empty((r, p), dtype=np.uint8)
    beta_source = np.empty((r, p), dtype=np.float32)
    beta_target = np.empty((r, p), dtype=np.float32)
    prior_score = np.empty((r, p), dtype=np.float32)
    mapping_index = np.empty((r, p), dtype=np.int16)

    def put(rep: int, generated: Dict[str, Any]) -> None:
        xs = np.asarray(generated["X_source"])
        xt = np.asarray(generated["X_target_train"])
        xv = np.asarray(generated["X_target_test"])

        if xs.shape != (source_n, p):
            raise RuntimeError(
                f"{cell_id} rep {rep}: source matrix shape changed: {xs.shape}."
            )
        if xt.shape != (target_train_n, p):
            raise RuntimeError(
                f"{cell_id} rep {rep}: target-train shape changed: {xt.shape}."
            )
        if xv.shape != (test_n, p):
            raise RuntimeError(
                f"{cell_id} rep {rep}: target-test shape changed: {xv.shape}."
            )

        X_source[rep] = xs
        source_time[rep] = generated["source_time"]
        source_event[rep] = generated["source_event"]

        X_target_train[rep] = xt
        target_train_time[rep] = generated["target_train_time"]
        target_train_event[rep] = generated["target_train_event"]

        X_target_test[rep] = xv
        target_test_time[rep] = generated["target_test_time"]
        target_test_event[rep] = generated["target_test_event"]

        causal_mask[rep] = np.asarray(
            generated["causal_mask"], dtype=np.uint8
        )
        transportable_mask[rep] = np.asarray(
            generated["transportable_mask"], dtype=np.uint8
        )
        beta_source[rep] = np.asarray(
            generated["beta_source"], dtype=np.float32
        )
        beta_target[rep] = np.asarray(
            generated["beta_target"], dtype=np.float32
        )
        prior_score[rep] = np.asarray(
            generated["prior_score"], dtype=np.float32
        )
        mapping_index[rep] = np.asarray(
            generated["mapping_index"], dtype=np.int16
        )

    put(0, first)

    for rep in range(1, r):
        generated = generator.generate_full_replicate(
            int(seeds[rep]),
            scenario,
            target_test_n=target_test_n,
        )
        put(rep, generated)

    expected_target_events = int(scenario["target_events"])

    if not np.all(
        source_event.sum(axis=1) == EXPECTED_SOURCE_EVENTS
    ):
        raise RuntimeError(
            f"{cell_id}: source event-count identity failed."
        )

    if not np.all(
        target_train_event.sum(axis=1) == expected_target_events
    ):
        raise RuntimeError(
            f"{cell_id}: target-training event-count identity failed."
        )

    float_arrays = [
        X_source,
        source_time,
        X_target_train,
        target_train_time,
        X_target_test,
        target_test_time,
        beta_source,
        beta_target,
        prior_score,
    ]
    if not all(np.isfinite(a).all() for a in float_arrays):
        raise RuntimeError(
            f"{cell_id}: generated arrays contain non-finite values."
        )

    return {
        "replicate_seed": seeds,
        "causal_mask": causal_mask,
        "transportable_mask": transportable_mask,
        "beta_source_truth": beta_source,
        "beta_target_truth": beta_target,
        "prior_score": prior_score,
        "mapping_index": mapping_index,
        "X_source": X_source,
        "source_time": source_time,
        "source_event": source_event,
        "X_target_train": X_target_train,
        "target_train_time": target_train_time,
        "target_train_event": target_train_event,
        "X_target_test": X_target_test,
        "target_test_time": target_test_time,
        "target_test_event": target_test_event,
    }


def train_A1_free_head(
    c05,
    source_model,
    X_train_aligned: torch.Tensor,
    time_tensor: torch.Tensor,
    event_tensor: torch.Tensor,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Controlled intervention around exact 05c A1:
    same frozen encoder, source-head initialization, epochs and LR;
    only L2-to-source-head is removed.
    """
    with torch.no_grad():
        z_train = source_model.encode(X_train_aligned).detach()

    source_head = source_model.head.detach().clone()
    head = nn.Parameter(source_head.clone())

    optimizer = torch.optim.Adam(
        [head],
        lr=float(c05.A1_LR),
    )

    for _ in range(int(c05.A1_EPOCHS)):
        optimizer.zero_grad(set_to_none=True)

        risk = torch.einsum(
            "rnl,rl->rn",
            z_train,
            head,
        )
        loss = c05.batched_cox_nll(
            risk,
            time_tensor,
            event_tensor,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Non-finite M3 A1-free-head loss."
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [head],
            c05.GRAD_CLIP_NORM,
        )
        optimizer.step()

    with torch.no_grad():
        risk = torch.einsum(
            "rnl,rl->rn",
            z_train,
            head,
        )
        final = c05.per_replicate_cox_nll(
            risk,
            time_tensor,
            event_tensor,
        )

    return head.detach(), final.astype(np.float32)


def cpu32(tensor: torch.Tensor) -> np.ndarray:
    return (
        tensor.detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )


def validate_risk_matrix(
    risk: np.ndarray,
    *,
    name: str,
    cell_id: str,
) -> None:
    if risk.ndim != 2:
        raise RuntimeError(
            f"{cell_id}: {name} risk is not 2-D."
        )
    if not np.isfinite(risk).all():
        raise RuntimeError(
            f"{cell_id}: {name} risk contains non-finite values."
        )

    sd = np.std(risk, axis=1, ddof=0)
    if not np.all(sd > 1e-8):
        bad = int(np.sum(sd <= 1e-8))
        raise RuntimeError(
            f"{cell_id}: {name} has {bad} replicate(s) "
            "with effectively constant risk."
        )


def run_controlled_cell(
    c05,
    arrays: Dict[str, np.ndarray],
    *,
    cell_index: int,
    cell_id: str,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    r = len(arrays["replicate_seed"])
    p = EXPECTED_N_MODULES

    # ------------------------------------------------------------------
    # Exact 05c device arrays and train-only standardization.
    # ------------------------------------------------------------------
    Xs = torch.as_tensor(
        arrays["X_source"],
        dtype=torch.float32,
        device=device,
    )
    ts = torch.as_tensor(
        arrays["source_time"],
        dtype=torch.float32,
        device=device,
    )
    es = torch.as_tensor(
        arrays["source_event"],
        dtype=torch.float32,
        device=device,
    )

    Xt = torch.as_tensor(
        arrays["X_target_train"],
        dtype=torch.float32,
        device=device,
    )
    tt = torch.as_tensor(
        arrays["target_train_time"],
        dtype=torch.float32,
        device=device,
    )
    et = torch.as_tensor(
        arrays["target_train_event"],
        dtype=torch.float32,
        device=device,
    )

    Xv = torch.as_tensor(
        arrays["X_target_test"],
        dtype=torch.float32,
        device=device,
    )

    prior = torch.as_tensor(
        arrays["prior_score"],
        dtype=torch.float32,
        device=device,
    )
    mapping = torch.as_tensor(
        arrays["mapping_index"],
        dtype=torch.long,
        device=device,
    )
    truth_gate = torch.as_tensor(
        arrays["transportable_mask"],
        dtype=torch.float32,
        device=device,
    )

    Xs_z, _, _, _ = c05.standardize_train_test(Xs)
    Xt_z, Xv_z, _, _ = c05.standardize_train_test(Xt, Xv)

    if Xv_z is None:
        raise RuntimeError(
            f"{cell_id}: target-test standardization failed."
        )

    mapping_matrix = c05.build_mapping_matrix(
        mapping,
        p,
    )

    Xt_aligned = c05.align_target_to_source(
        Xt_z,
        mapping_matrix,
    )
    Xv_aligned = c05.align_target_to_source(
        Xv_z,
        mapping_matrix,
    )

    model_seed_base = (
        MODEL_BASE_SEED
        + int(cell_index) * CELL_SEED_STRIDE
    )

    # ------------------------------------------------------------------
    # M0: exact 05c B0 target-only ridge Cox.
    # ------------------------------------------------------------------
    beta_M0, loss_M0 = c05.train_linear_cox(
        Xt_z,
        tt,
        et,
        center=None,
        alpha=c05.TARGET_B0_ALPHA,
        epochs=c05.LINEAR_EPOCHS_TARGET,
        lr=c05.LINEAR_LR_TARGET,
        init_seed=model_seed_base + 20,
    )

    with torch.no_grad():
        risk_train_M0 = torch.einsum(
            "rnp,rp->rn",
            Xt_z,
            beta_M0,
        )
        risk_test_M0 = torch.einsum(
            "rnp,rp->rn",
            Xv_z,
            beta_M0,
        )

    # ------------------------------------------------------------------
    # Shared exact 05c source MLP.
    # ------------------------------------------------------------------
    source_model, loss_source_mlp = c05.train_source_mlp(
        Xs_z,
        ts,
        es,
        epochs=c05.SOURCE_MLP_EPOCHS,
        lr=c05.SOURCE_MLP_LR,
        weight_decay=c05.SOURCE_MLP_WEIGHT_DECAY,
        init_seed=model_seed_base + 100,
    )

    c05.freeze_module(source_model)

    with torch.no_grad():
        source_head = source_model.head.detach().clone()

        # M1: source network zero-shot on mapped target inputs.
        risk_train_M1 = source_model(Xt_aligned)
        risk_test_M1 = source_model(Xv_aligned)

        loss_M1 = c05.per_replicate_cox_nll(
            risk_train_M1,
            tt,
            et,
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # M2: exact A1.
    # ------------------------------------------------------------------
    head_M2, loss_M2 = c05.train_A1_head(
        source_model,
        Xt_aligned,
        tt,
        et,
    )

    with torch.no_grad():
        risk_train_M2 = c05.predict_A1(
            source_model,
            head_M2,
            Xt_aligned,
        )
        risk_test_M2 = c05.predict_A1(
            source_model,
            head_M2,
            Xv_aligned,
        )

    # ------------------------------------------------------------------
    # M3: A1 with only L2-to-source-head removed.
    # ------------------------------------------------------------------
    head_M3, loss_M3 = train_A1_free_head(
        c05,
        source_model,
        Xt_aligned,
        tt,
        et,
    )

    with torch.no_grad():
        risk_train_M3 = c05.predict_A1(
            source_model,
            head_M3,
            Xt_aligned,
        )
        risk_test_M3 = c05.predict_A1(
            source_model,
            head_M3,
            Xv_aligned,
        )

    # ------------------------------------------------------------------
    # M4: exact 05c A3 soft learned gate.
    # ------------------------------------------------------------------
    gate_M4, residual_M4, loss_M4 = c05.train_A3(
        source_model,
        Xt_z,
        mapping_matrix,
        prior,
        tt,
        et,
        allow_target_override=True,
    )

    with torch.no_grad():
        risk_train_M4 = c05.predict_A3(
            source_model,
            gate_M4,
            residual_M4,
            Xt_z,
            mapping_matrix,
        )
        risk_test_M4 = c05.predict_A3(
            source_model,
            gate_M4,
            residual_M4,
            Xv_z,
            mapping_matrix,
        )

    # ------------------------------------------------------------------
    # M5: prediction-only hardening. NO refit.
    # ------------------------------------------------------------------
    gate_M5 = (
        gate_M4 >= float(A3_HARD_GATE_THRESHOLD)
    ).to(dtype=torch.float32)

    with torch.no_grad():
        risk_train_M5 = c05.predict_A3(
            source_model,
            gate_M5,
            residual_M4,
            Xt_z,
            mapping_matrix,
        )
        risk_test_M5 = c05.predict_A3(
            source_model,
            gate_M5,
            residual_M4,
            Xv_z,
            mapping_matrix,
        )
        loss_M5 = c05.per_replicate_cox_nll(
            risk_train_M5,
            tt,
            et,
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # M6: truth-fixed 0/1 transportability gate, residual only learned.
    #
    # Exact 05c train_A3(... allow_target_override=False):
    # - gate fixed to provided "prior" tensor;
    # - prior penalty is identically zero because gate == source_prior;
    # - no override parameter exists;
    # - only residual_beta is optimized with exact A3 epochs/lr/residual L2.
    # ------------------------------------------------------------------
    gate_M6, residual_M6, loss_M6 = c05.train_A3(
        source_model,
        Xt_z,
        mapping_matrix,
        truth_gate,
        tt,
        et,
        allow_target_override=False,
    )

    # Exact identity: train_A3 no-override must return the truth gate.
    if not torch.equal(
        gate_M6,
        truth_gate,
    ):
        raise RuntimeError(
            f"{cell_id}: M6 truth-gate identity failed."
        )

    with torch.no_grad():
        risk_train_M6 = c05.predict_A3(
            source_model,
            gate_M6,
            residual_M6,
            Xt_z,
            mapping_matrix,
        )
        risk_test_M6 = c05.predict_A3(
            source_model,
            gate_M6,
            residual_M6,
            Xv_z,
            mapping_matrix,
        )

    risk_train_tensors = {
        "M0_B0": risk_train_M0,
        "M1_SOURCE_NETWORK_ZERO_SHOT": risk_train_M1,
        "M2_A1_SOURCE_CENTERED_HEAD": risk_train_M2,
        "M3_A1_FREE_HEAD": risk_train_M3,
        "M4_A3_SOFT_LEARNED": risk_train_M4,
        "M5_A3_HARDENED_PREDICTION": risk_train_M5,
        "M6_A3_ORACLE_HARD_GATE": risk_train_M6,
    }
    risk_test_tensors = {
        "M0_B0": risk_test_M0,
        "M1_SOURCE_NETWORK_ZERO_SHOT": risk_test_M1,
        "M2_A1_SOURCE_CENTERED_HEAD": risk_test_M2,
        "M3_A1_FREE_HEAD": risk_test_M3,
        "M4_A3_SOFT_LEARNED": risk_test_M4,
        "M5_A3_HARDENED_PREDICTION": risk_test_M5,
        "M6_A3_ORACLE_HARD_GATE": risk_test_M6,
    }

    output: Dict[str, np.ndarray] = {
        "replicate_seed": arrays[
            "replicate_seed"
        ].astype(np.int64),

        "target_train_time": arrays[
            "target_train_time"
        ].astype(np.float32),
        "target_train_event": arrays[
            "target_train_event"
        ].astype(np.uint8),
        "target_test_time": arrays[
            "target_test_time"
        ].astype(np.float32),
        "target_test_event": arrays[
            "target_test_event"
        ].astype(np.uint8),

        "causal_mask": arrays[
            "causal_mask"
        ].astype(np.uint8),
        "transportable_mask": arrays[
            "transportable_mask"
        ].astype(np.uint8),
        "beta_source_truth": arrays[
            "beta_source_truth"
        ].astype(np.float32),
        "beta_target_truth": arrays[
            "beta_target_truth"
        ].astype(np.float32),
        "prior_score": arrays[
            "prior_score"
        ].astype(np.float32),
        "mapping_index": arrays[
            "mapping_index"
        ].astype(np.int16),

        "head_source": cpu32(source_head),
        "head_M2_A1_SOURCE_CENTERED": cpu32(head_M2),
        "head_M3_A1_FREE": cpu32(head_M3),

        "gate_M4_A3_SOFT": cpu32(gate_M4),
        "gate_M5_A3_HARDENED": cpu32(gate_M5),
        "gate_M6_A3_ORACLE": cpu32(gate_M6),

        "residual_M4_A3_SOFT": cpu32(residual_M4),
        # M5 deliberately reuses residual_M4, so do not duplicate the array.
        "residual_M6_A3_ORACLE": cpu32(residual_M6),

        "beta_M0_B0": cpu32(beta_M0),

        "train_loss_source_mlp": loss_source_mlp.astype(np.float32),
        "train_loss_M0_B0": loss_M0.astype(np.float32),
        "train_loss_M1_SOURCE_NETWORK_ZERO_SHOT": loss_M1,
        "train_loss_M2_A1_SOURCE_CENTERED_HEAD": loss_M2.astype(np.float32),
        "train_loss_M3_A1_FREE_HEAD": loss_M3.astype(np.float32),
        "train_loss_M4_A3_SOFT_LEARNED": loss_M4.astype(np.float32),
        "train_loss_M5_A3_HARDENED_PREDICTION": loss_M5,
        "train_loss_M6_A3_ORACLE_HARD_GATE": loss_M6.astype(np.float32),
    }

    for name in MODEL_NAMES:
        train_arr = cpu32(risk_train_tensors[name])
        test_arr = cpu32(risk_test_tensors[name])

        validate_risk_matrix(
            train_arr,
            name=f"risk_train_{name}",
            cell_id=cell_id,
        )
        validate_risk_matrix(
            test_arr,
            name=f"risk_test_{name}",
            cell_id=cell_id,
        )

        output[f"risk_train_{name}"] = train_arr
        output[f"risk_test_{name}"] = test_arr

    # Additional execution invariants.
    if not np.array_equal(
        output["gate_M6_A3_ORACLE"],
        arrays["transportable_mask"].astype(np.float32),
    ):
        raise RuntimeError(
            f"{cell_id}: saved M6 oracle gate differs from truth mask."
        )

    if not np.array_equal(
        output["gate_M5_A3_HARDENED"],
        (
            output["gate_M4_A3_SOFT"]
            >= A3_HARD_GATE_THRESHOLD
        ).astype(np.float32),
    ):
        raise RuntimeError(
            f"{cell_id}: saved M5 hard gate identity failed."
        )

    # M5 is definitionally no-refit and must reuse M4 residual.
    output["M5_reuses_M4_residual_flag"] = np.ones(
        (r,),
        dtype=np.uint8,
    )

    for key, arr in output.items():
        if np.issubdtype(arr.dtype, np.floating):
            if not np.isfinite(arr).all():
                raise RuntimeError(
                    f"{cell_id}: non-finite output array {key}."
                )

    return output


def cell_output_paths(cell_id: str) -> Tuple[Path, Path]:
    return (
        CELL_DIR / f"{cell_id}.npz",
        CELL_MANIFEST_DIR / f"{cell_id}.json",
    )


def load_completed_cell(
    *,
    cell_id: str,
    implementation_hash: str,
    recipe_hash: str,
    generator_script_hash: str,
) -> Optional[Dict[str, Any]]:
    output_path, manifest_path = cell_output_paths(cell_id)

    if not output_path.exists() or not manifest_path.exists():
        return None

    manifest = read_json(manifest_path)

    if clean(manifest.get("status")) != "PASS":
        return None
    if clean(
        manifest.get("implementation_contract_sha256")
    ) != implementation_hash:
        return None
    if clean(
        manifest.get("recipe_sha256")
    ) != recipe_hash:
        return None
    if clean(
        manifest.get("05b_generator_script_sha256")
    ) != generator_script_hash:
        return None
    if clean(
        manifest.get("output_sha256")
    ) != sha256_file(output_path):
        return None

    return manifest


def implementation_payload(
    *,
    script_path: Path,
    device: torch.device,
    c05,
    c05_impl: Dict[str, Any],
    generator_script_hash: str,
    e0_models: pd.DataFrame,
) -> Dict[str, Any]:
    return {
        "script_version": SCRIPT_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "status": "PASS_FROZEN_BEFORE_FIRST_05E2_MODEL_FIT",
        "created_utc": now_utc(),
        "script_sha256": sha256_file(script_path),

        "05e0_contract_sha256": sha256_file(E0_CONTRACT),
        "05e1_manifest_sha256": sha256_file(E1_MANIFEST),
        "05b_generator_script_sha256": generator_script_hash,
        "05c_script_sha256": sha256_file(C_SCRIPT),
        "05c_implementation_contract_sha256": sha256_file(
            C_IMPLEMENTATION
        ),

        "models": MODEL_NAMES,
        "05e0_model_registry_records": e0_models.to_dict(
            orient="records"
        ),

        "frozen_05c_reuse": {
            "source_standardization": "exact 05c standardize_train_test",
            "target_standardization": "exact 05c standardize_train_test",
            "cox_loss": "exact 05c batched_cox_nll",
            "B0": "exact 05c train_linear_cox target B0 configuration",
            "source_MLP": "exact 05c train_source_mlp",
            "A1": "exact 05c train_A1_head",
            "A3": "exact 05c train_A3",
            "A3_prediction": "exact 05c predict_A3",
            "mapping": "exact 05c build_mapping_matrix/align_target_to_source",
            "model_base_seed": MODEL_BASE_SEED,
            "cell_seed_stride": CELL_SEED_STRIDE,
        },

        "controlled_interventions": {
            "M1_SOURCE_NETWORK_ZERO_SHOT": (
                "No target fit; frozen 05c source MLP encoder+source head."
            ),
            "M3_A1_FREE_HEAD": {
                "difference_from_exact_A1": (
                    "A1 L2-to-source-head penalty set from "
                    f"{c05.A1_HEAD_L2_TO_SOURCE} to 0 only"
                ),
                "same_source_encoder": True,
                "same_head_initialization": True,
                "same_epochs": int(c05.A1_EPOCHS),
                "same_lr": float(c05.A1_LR),
            },
            "M5_A3_HARDENED_PREDICTION": {
                "M4_refit": False,
                "threshold": A3_HARD_GATE_THRESHOLD,
                "hardening": "I(g_M4 >= 0.5)",
                "residual_refit": False,
                "reuses_M4_residual": True,
                "source_and_residual_paths_recomputed": True,
            },
            "M6_A3_ORACLE_HARD_GATE": {
                "gate": "true transportable_mask",
                "gate_learning": False,
                "implementation": (
                    "05c train_A3 with truth mask supplied as fixed prior "
                    "and allow_target_override=False"
                ),
                "only_residual_trainable": True,
                "oracle_nonimplementable": True,
            },
        },

        "scientific_hyperparameters": {
            "TARGET_B0_ALPHA": float(c05.TARGET_B0_ALPHA),
            "LINEAR_EPOCHS_TARGET": int(c05.LINEAR_EPOCHS_TARGET),
            "LINEAR_LR_TARGET": float(c05.LINEAR_LR_TARGET),

            "HIDDEN_DIM": int(c05.HIDDEN_DIM),
            "LATENT_DIM": int(c05.LATENT_DIM),
            "SOURCE_MLP_EPOCHS": int(c05.SOURCE_MLP_EPOCHS),
            "SOURCE_MLP_LR": float(c05.SOURCE_MLP_LR),
            "SOURCE_MLP_WEIGHT_DECAY": float(
                c05.SOURCE_MLP_WEIGHT_DECAY
            ),

            "A1_EPOCHS": int(c05.A1_EPOCHS),
            "A1_LR": float(c05.A1_LR),
            "A1_HEAD_L2_TO_SOURCE": float(
                c05.A1_HEAD_L2_TO_SOURCE
            ),

            "A3_EPOCHS": int(c05.A3_EPOCHS),
            "A3_LR": float(c05.A3_LR),
            "A3_PRIOR_PENALTY": float(c05.A3_PRIOR_PENALTY),
            "A3_GATE_OVERRIDE_L2": float(c05.A3_GATE_OVERRIDE_L2),
            "A3_RESIDUAL_L2": float(c05.A3_RESIDUAL_L2),
            "A3_LOGIT_EPS": float(c05.A3_LOGIT_EPS),

            "GRAD_CLIP_NORM": float(c05.GRAD_CLIP_NORM),
            "RISK_CLIP": float(c05.RISK_CLIP),
        },

        "runtime_first_run": {
            "device": str(device),
            "torch_version": str(torch.__version__),
            "numpy_version": str(np.__version__),
        },

        "guardrails": {
            "model_selection": False,
            "A6_reopened": False,
            "scenario_specific_tuning": False,
            "postresult_grid_expansion": False,
            "human_outcomes_read": False,
            "real_data_values_read": False,
        },
    }


def freeze_or_verify_implementation(
    payload: Dict[str, Any],
) -> None:
    if not IMPLEMENTATION_CONTRACT.exists():
        write_json(
            IMPLEMENTATION_CONTRACT,
            payload,
        )
        return

    existing = read_json(IMPLEMENTATION_CONTRACT)

    # Runtime/device can differ on restart. Scientific implementation cannot.
    keys = [
        "script_version",
        "implementation_version",
        "script_sha256",
        "05e0_contract_sha256",
        "05e1_manifest_sha256",
        "05b_generator_script_sha256",
        "05c_script_sha256",
        "05c_implementation_contract_sha256",
        "models",
        "05e0_model_registry_records",
        "frozen_05c_reuse",
        "controlled_interventions",
        "scientific_hyperparameters",
        "guardrails",
    ]

    for key in keys:
        if existing.get(key) != payload.get(key):
            raise RuntimeError(
                f"Existing 05e2 implementation contract differs at {key!r}. "
                "Do not continue with modified scientific implementation."
            )


def main() -> None:
    started = now_utc()

    print("=" * 120)
    print("Paper 6 - run frozen post-HOLD controlled mechanism model matrix")
    print("=" * 120)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Implementation version: {IMPLEMENTATION_VERSION}")
    print()
    print("Scope:")
    print("  New-seed controlled cells: YES [12 cells; 6,000 replicates]")
    print("  Exact frozen 05c implementation imported: YES")
    print("  Controlled branches M0-M6 only: YES")
    print("  Model selection / architecture rescue: NO")
    print("  A6 reopened: NO")
    print("  Human outcomes read: NO")
    print("  Real DOG2/human values read: NO")
    print("  GPU-first execution: YES")
    print()

    for path in [
        E0_CONTRACT,
        E0_SCENARIOS,
        E0_MODELS,
        E0_CONTRASTS,
        E0_SUMMARY,
        E1_MANIFEST,
        E1_AUDIT,
        E1_SMOKE,
        E1_SUMMARY,
        B_GENERATOR_CONTRACT,
        C_SCRIPT,
        C_IMPLEMENTATION,
        C_SUMMARY,
    ]:
        require_file(path)

    # ------------------------------------------------------------------
    # Frozen 05e0 identity.
    # ------------------------------------------------------------------
    if sha256_file(E0_CONTRACT) != EXPECTED_E0_CONTRACT_SHA256:
        raise RuntimeError(
            "05e0 contract SHA256 changed after frozen PASS."
        )

    e0_summary = read_json(E0_SUMMARY)
    if clean(e0_summary.get("scientific_status")) != EXPECTED_E0_STATUS:
        raise RuntimeError(
            "05e0 is not in expected frozen PASS state."
        )

    e0_hashes = e0_summary.get("artifact_hashes") or {}
    e0_artifacts = {
        "contract": E0_CONTRACT,
        "scenario_registry": E0_SCENARIOS,
        "model_registry": E0_MODELS,
        "contrast_registry": E0_CONTRASTS,
    }
    for key, path in e0_artifacts.items():
        expected = clean(e0_hashes.get(key))
        if not expected:
            raise RuntimeError(
                f"05e0 summary missing frozen artifact hash {key}."
            )
        if sha256_file(path) != expected:
            raise RuntimeError(
                f"05e0 frozen artifact changed: {key}."
            )

    # ------------------------------------------------------------------
    # PASS 05e1 v3 identity.
    # ------------------------------------------------------------------
    e1_summary = read_json(E1_SUMMARY)
    if clean(e1_summary.get("scientific_status")) != EXPECTED_E1_STATUS:
        raise RuntimeError(
            "05e1 v3 is not in expected PASS state."
        )

    observed_e1_manifest_hash = sha256_file(E1_MANIFEST)
    if observed_e1_manifest_hash != EXPECTED_E1_MANIFEST_SHA256:
        raise RuntimeError(
            "05e1 recipe manifest differs from the completed PASS run. "
            f"Observed={observed_e1_manifest_hash}"
        )

    e1_hashes = e1_summary.get("artifact_hashes") or {}
    if clean(
        e1_hashes.get("controlled_recipe_manifest.tsv")
    ) != observed_e1_manifest_hash:
        raise RuntimeError(
            "05e1 summary/manifest hash mismatch."
        )

    recipe_manifest = pd.read_csv(
        E1_MANIFEST,
        sep="\t",
    )
    scenarios = pd.read_csv(
        E0_SCENARIOS,
        sep="\t",
    )
    e0_models = pd.read_csv(
        E0_MODELS,
        sep="\t",
    )

    if len(recipe_manifest) != EXPECTED_CELLS:
        raise RuntimeError(
            "05e1 manifest does not contain all 12 cells."
        )
    if len(scenarios) != EXPECTED_CELLS:
        raise RuntimeError(
            "05e0 scenario registry does not contain all 12 cells."
        )
    if int(recipe_manifest["replicates"].sum()) != EXPECTED_TOTAL_REPLICATES:
        raise RuntimeError(
            "05e1 replicate total changed."
        )

    observed_model_names = e0_models["model"].astype(str).tolist()
    if observed_model_names != MODEL_NAMES:
        raise RuntimeError(
            "05e0 model registry order/content changed. "
            f"Observed={observed_model_names}"
        )

    # ------------------------------------------------------------------
    # Authoritative 05b generator.
    # ------------------------------------------------------------------
    generator_contract = read_json(
        B_GENERATOR_CONTRACT
    )
    generator_path = ROOT / clean(
        generator_contract.get("authoritative_generator_script")
    )
    generator_script_hash = clean(
        generator_contract.get(
            "authoritative_generator_script_sha256"
        )
    )

    require_file(generator_path)

    if sha256_file(generator_path) != generator_script_hash:
        raise RuntimeError(
            "Authoritative 05b generator SHA256 mismatch."
        )

    if clean(
        e1_summary.get("05b_authoritative_generator_sha256")
    ) != generator_script_hash:
        raise RuntimeError(
            "05e1 generator hash differs from current authoritative 05b generator."
        )

    generator = load_module(
        generator_path,
        "paper6_05b_generator_for_05e2",
    )

    # ------------------------------------------------------------------
    # Exact frozen 05c import and verification.
    # ------------------------------------------------------------------
    c_impl = read_json(C_IMPLEMENTATION)
    c_summary = read_json(C_SUMMARY)

    if clean(c_summary.get("scientific_status")) != EXPECTED_05C_STATUS:
        raise RuntimeError(
            "05c is not in expected frozen PASS state."
        )

    expected_c_script_hash = clean(
        c_impl.get("script_sha256")
    )
    observed_c_script_hash = sha256_file(C_SCRIPT)

    if not expected_c_script_hash:
        raise RuntimeError(
            "05c implementation contract lacks script_sha256."
        )

    if observed_c_script_hash != expected_c_script_hash:
        raise RuntimeError(
            "Frozen 05c script SHA256 differs from its implementation contract."
        )

    c05 = load_module(
        C_SCRIPT,
        "paper6_frozen_05c_for_05e2",
    )

    # Verify critical frozen constants before first fit.
    constant_checks = {
        "MODEL_BASE_SEED": MODEL_BASE_SEED,
        "EXPECTED_N_MODULES": EXPECTED_N_MODULES,
        "EXPECTED_SOURCE_N": EXPECTED_SOURCE_N,
        "EXPECTED_SOURCE_EVENTS": EXPECTED_SOURCE_EVENTS,
        "TARGET_B0_ALPHA": 2.0,
        "A1_EPOCHS": 80,
        "A1_LR": 0.020,
        "A1_HEAD_L2_TO_SOURCE": 1e-2,
        "A3_EPOCHS": 120,
        "A3_LR": 0.020,
        "A3_RESIDUAL_L2": 0.01,
    }

    for key, expected in constant_checks.items():
        if not hasattr(c05, key):
            raise RuntimeError(
                f"Frozen 05c module lacks constant {key}."
            )
        observed = getattr(c05, key)

        if isinstance(expected, float):
            if not np.isclose(
                float(observed),
                expected,
                atol=1e-12,
                rtol=0,
            ):
                raise RuntimeError(
                    f"Frozen 05c constant changed: {key}={observed}."
                )
        else:
            if int(observed) != int(expected):
                raise RuntimeError(
                    f"Frozen 05c constant changed: {key}={observed}."
                )

    print("05e0 frozen artifact verification: PASS")
    print("05e1 v3 recipe manifest verification: PASS")
    print("05b authoritative generator verification: PASS")
    print("05c script/implementation verification: PASS")
    print()

    # ------------------------------------------------------------------
    # Device + 05e2 implementation freeze BEFORE first new model fit.
    # ------------------------------------------------------------------
    device = choose_device()
    configure_torch(c05, device)

    if device.type == "cuda":
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        print(
            f"Execution device: CUDA "
            f"[{torch.cuda.get_device_name(device)}]"
        )
        print(
            f"CUDA memory free/total: "
            f"{free_bytes / 1024**3:.2f}/"
            f"{total_bytes / 1024**3:.2f} GiB"
        )
    else:
        print("Execution device: CPU [explicitly requested]")

    impl_payload = implementation_payload(
        script_path=Path(__file__).resolve(),
        device=device,
        c05=c05,
        c05_impl=c_impl,
        generator_script_hash=generator_script_hash,
        e0_models=e0_models,
    )
    freeze_or_verify_implementation(
        impl_payload
    )
    implementation_hash = sha256_file(
        IMPLEMENTATION_CONTRACT
    )

    print(
        f"05e2 implementation contract SHA256: "
        f"{implementation_hash}"
    )
    print("Implementation frozen before first new-seed model fit: PASS")
    print()

    # ------------------------------------------------------------------
    # Cell loop.
    # ------------------------------------------------------------------
    recipe_by_cell = {
        str(row.cell_id): row
        for row in recipe_manifest.itertuples(index=False)
    }

    output_manifest_rows: List[Dict[str, Any]] = []

    reused = 0
    fitted = 0
    done = 0
    total_replicates = 0

    wall_start = time.time()

    for _, scenario_row in scenarios.sort_values(
        "cell_index"
    ).iterrows():
        cell_id = str(scenario_row["cell_id"])
        cell_index = int(scenario_row["cell_index"])

        recipe_row = recipe_by_cell.get(cell_id)
        if recipe_row is None:
            raise RuntimeError(
                f"No 05e1 recipe-manifest row for {cell_id}."
            )

        recipe_path = ROOT / str(
            recipe_row.recipe_path
        )
        require_file(recipe_path)

        recipe_hash = sha256_file(recipe_path)

        if recipe_hash != str(recipe_row.recipe_sha256):
            raise RuntimeError(
                f"{cell_id}: recipe SHA256 mismatch."
            )

        existing = load_completed_cell(
            cell_id=cell_id,
            implementation_hash=implementation_hash,
            recipe_hash=recipe_hash,
            generator_script_hash=generator_script_hash,
        )

        if existing is not None:
            reused += 1
            done += 1
            total_replicates += int(
                existing["replicates"]
            )

            output_manifest_rows.append({
                "cell_id": cell_id,
                "cell_index": cell_index,
                "target_events": int(
                    scenario_row["target_events"]
                ),
                "transfer_regime": str(
                    scenario_row["transfer_regime"]
                ),
                "replicates": int(
                    existing["replicates"]
                ),
                "status": "REUSED",
                "output_path": existing["output_path"],
                "output_sha256": existing["output_sha256"],
                "output_size_bytes": int(
                    existing["output_size_bytes"]
                ),
                "elapsed_seconds": float(
                    existing.get(
                        "elapsed_seconds",
                        0.0,
                    )
                ),
            })

            print(
                f"  {cell_id}: REUSED "
                f"overall {done}/{EXPECTED_CELLS}"
            )
            continue

        cell_start = time.time()

        recipe = load_recipe(
            recipe_path
        )

        # Verify recipe is bound to the same frozen 05e0/generator.
        if str(
            scalar_from_recipe(
                recipe,
                "05e0_contract_sha256",
            )
        ) != EXPECTED_E0_CONTRACT_SHA256:
            raise RuntimeError(
                f"{cell_id}: recipe 05e0 contract identity mismatch."
            )

        if str(
            scalar_from_recipe(
                recipe,
                "05b_generator_sha256",
            )
        ) != generator_script_hash:
            raise RuntimeError(
                f"{cell_id}: recipe generator identity mismatch."
            )

        arrays = generate_cell_arrays(
            generator,
            recipe,
            cell_id=cell_id,
        )

        result = run_controlled_cell(
            c05,
            arrays,
            cell_index=cell_index,
            cell_id=cell_id,
            device=device,
        )

        output_path, manifest_path = (
            cell_output_paths(cell_id)
        )

        atomic_savez_compressed(
            output_path,
            **result,
        )

        elapsed = time.time() - cell_start

        cell_manifest = {
            "status": "PASS",
            "created_utc": now_utc(),
            "cell_id": cell_id,
            "cell_index": cell_index,
            "target_events": int(
                scenario_row["target_events"]
            ),
            "transfer_regime": str(
                scenario_row["transfer_regime"]
            ),
            "replicates": int(
                scenario_row["replicates"]
            ),
            "models": MODEL_NAMES,
            "implementation_contract_sha256": implementation_hash,
            "05e0_contract_sha256": EXPECTED_E0_CONTRACT_SHA256,
            "05e1_manifest_sha256": observed_e1_manifest_hash,
            "recipe_sha256": recipe_hash,
            "05b_generator_script_sha256": generator_script_hash,
            "05c_script_sha256": observed_c_script_hash,
            "output_path": str(
                output_path.relative_to(ROOT)
            ),
            "output_sha256": sha256_file(
                output_path
            ),
            "output_size_bytes": int(
                output_path.stat().st_size
            ),
            "elapsed_seconds": elapsed,
            "device": str(device),
        }

        write_json(
            manifest_path,
            cell_manifest,
        )

        fitted += 1
        done += 1
        total_replicates += int(
            scenario_row["replicates"]
        )

        output_manifest_rows.append({
            "cell_id": cell_id,
            "cell_index": cell_index,
            "target_events": int(
                scenario_row["target_events"]
            ),
            "transfer_regime": str(
                scenario_row["transfer_regime"]
            ),
            "replicates": int(
                scenario_row["replicates"]
            ),
            "status": "FITTED",
            "output_path": cell_manifest[
                "output_path"
            ],
            "output_sha256": cell_manifest[
                "output_sha256"
            ],
            "output_size_bytes": cell_manifest[
                "output_size_bytes"
            ],
            "elapsed_seconds": elapsed,
        })

        if device.type == "cuda":
            torch.cuda.empty_cache()

        elapsed_total = time.time() - wall_start
        rate = done / max(
            elapsed_total,
            1e-9,
        )
        remaining = (
            EXPECTED_CELLS - done
        )
        eta_min = (
            remaining
            / max(rate, 1e-9)
            / 60.0
        )

        print(
            f"  {cell_id}: PASS "
            f"[{int(scenario_row['replicates'])} reps, "
            f"{elapsed:.1f}s] "
            f"overall {done}/{EXPECTED_CELLS}"
        )
        print(
            f"    progress: reused={reused}, "
            f"fitted={fitted}, "
            f"approx remaining={eta_min:.1f} min"
        )

    manifest_df = pd.DataFrame(
        output_manifest_rows
    ).sort_values("cell_index")

    manifest_df.to_csv(
        OUTPUT_MANIFEST,
        sep="\t",
        index=False,
    )

    if len(manifest_df) != EXPECTED_CELLS:
        raise RuntimeError(
            "05e2 output manifest does not contain all 12 cells."
        )

    if int(
        manifest_df["replicates"].sum()
    ) != EXPECTED_TOTAL_REPLICATES:
        raise RuntimeError(
            "05e2 output manifest replicate total mismatch."
        )

    if total_replicates != EXPECTED_TOTAL_REPLICATES:
        raise RuntimeError(
            "05e2 processed replicate total mismatch."
        )

    total_output_bytes = int(
        manifest_df[
            "output_size_bytes"
        ].sum()
    )

    summary = {
        "script_version": SCRIPT_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "status": "PASS",
        "scientific_status": (
            "PASS_FROZEN_CONTROLLED_MECHANISM_MODEL_MATRIX_COMPLETE"
        ),
        "run_started_utc": started,
        "run_finished_utc": now_utc(),

        "device": str(device),
        "n_cells": EXPECTED_CELLS,
        "total_replicates": EXPECTED_TOTAL_REPLICATES,
        "models": MODEL_NAMES,

        "cell_outputs_reused_this_run": reused,
        "cell_outputs_fitted_this_run": fitted,

        "implementation_contract_sha256": implementation_hash,
        "05e0_contract_sha256": EXPECTED_E0_CONTRACT_SHA256,
        "05e1_manifest_sha256": observed_e1_manifest_hash,
        "05b_generator_script_sha256": generator_script_hash,
        "05c_script_sha256": observed_c_script_hash,

        "controlled_model_output_manifest_sha256": sha256_file(
            OUTPUT_MANIFEST
        ),
        "output_storage_bytes": total_output_bytes,

        "scientific_results_evaluated": False,
        "model_selection": False,
        "A6_reopened": False,
        "human_outcomes_read": False,
        "real_data_values_read": False,
        "scenario_specific_tuning": False,

        "next": (
            "05e3 compute frozen Uno-C/IBS/risk-scale/negative-transfer "
            "metrics and evaluate ONLY the four prespecified paired mechanism "
            "contrasts from 05e0."
        ),
    }

    write_json(
        SUMMARY_JSON,
        summary,
    )

    print()
    print("=" * 120)
    print("05e2 CONTROLLED MECHANISM MODEL MATRIX SUMMARY")
    print("=" * 120)
    print(
        f"Cells complete: "
        f"{EXPECTED_CELLS}/{EXPECTED_CELLS}"
    )
    print(
        f"Replicates complete: "
        f"{EXPECTED_TOTAL_REPLICATES:,}"
    )
    print(
        "Models: "
        + ", ".join(MODEL_NAMES)
    )
    print(
        f"Stored output: "
        f"{total_output_bytes / 1024**3:.2f} GiB"
    )
    print(f"Execution device: {device}")
    print()
    print("05e0 frozen contract changed: NO")
    print("05e1 recipes changed: NO")
    print("Exact frozen 05c machinery reused: YES")
    print("Model selection performed: NO")
    print("A6 reopened: NO")
    print("Human outcomes read: NO")
    print("Scientific contrast evaluation performed: NO")
    print()
    print(
        f"Output manifest SHA256: "
        f"{sha256_file(OUTPUT_MANIFEST)}"
    )
    print("=" * 120)
    print(
        "05e2: PASS_FROZEN_CONTROLLED_MECHANISM_MODEL_MATRIX_COMPLETE"
    )
    print("=" * 120)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 120, file=sys.stderr)
        print(
            "05e2 controlled mechanism model matrix: FAIL",
            file=sys.stderr,
        )
        print(
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        print("=" * 120, file=sys.stderr)
        raise
