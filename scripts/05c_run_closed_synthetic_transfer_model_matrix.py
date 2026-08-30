#!/usr/bin/env python3
"""
Paper 6 - run closed synthetic classical/AI transfer model matrix.

This is the heavy simulation-fitting stage following the frozen 05a/05b
known-truth contracts.

Scientific role
---------------
05c does NOT select the final architecture. It fits the already-closed model
matrix and stores held-out synthetic predictions, A3 borrowing weights, and
known-truth labels. 05d will compute Uno C / IBS / calibration / negative
transfer and apply the frozen architecture-selection rule.

Models
------
B0  target-only ridge Cox
B4  classical residual transfer Cox

A0  target-only small survival network
A1  frozen source encoder + target head only
A2  frozen source encoder + low-rank latent residual adapter
A3  evolution-conditioned module-resolved selective borrowing
A4  unrestricted/full neural fine-tuning

Required A3 ablations
---------------------
A3_NO_EVOLUTION_PRIOR
A3_NO_TARGET_OVERRIDE
A3_PERMUTED_PRIOR
A3_MAPPING_PERMUTATION
A3_SOURCE_OUTCOME_PERMUTATION

Implementation principles
-------------------------
- Synthetic source/target data are regenerated exactly from 05b recipes.
- No real DOG2/GSE16091/TARGET/GSE21257/GSE39055 values are read.
- All neural models use fixed hyperparameters frozen by THIS script before the
  first model fit. No scenario-specific tuning is allowed.
- Source pretraining is shared within a replicate across A1/A2/A3/A4.
- A3 exposes one borrowing weight per module, enabling known-truth recovery
  analysis in 05d.
- One compressed output shard is written per 05a scenario.
- Scenario shards are restart-safe and validated by SHA256.
- GPU is required by default because the full matrix contains 21,600
  replicates. Set PAPER6_SIM_DEVICE=cpu explicitly only if CPU execution is
  intentionally desired.

No CLI arguments.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
except ImportError as exc:
    raise ImportError(
        "05c requires PyTorch in the active Paper-6 .venv. "
        "Install a CUDA-enabled PyTorch build appropriate for this machine."
    ) from exc


SCRIPT_VERSION = "05c-run-closed-synthetic-transfer-model-matrix-v1-no-cli"
IMPLEMENTATION_VERSION = "paper6-closed-transfer-model-matrix-v1"

ROOT = Path(__file__).resolve().parents[1]

A_DIR = ROOT / "results" / "simulation_contract" / "05a"
A_CONTRACT = A_DIR / "negative_transfer_simulation_contract.json"
A_SUMMARY = A_DIR / "summary.json"
A_SCENARIOS = A_DIR / "simulation_scenario_registry.tsv"
A_MODELS = A_DIR / "closed_model_registry.tsv"
A_SELECTION = A_DIR / "architecture_selection_rules.tsv"

B_DIR = ROOT / "simulations" / "05b"
B_CONTRACT = B_DIR / "generator_contract.json"
B_SUMMARY = B_DIR / "summary.json"
B_MANIFEST = B_DIR / "scenario_recipe_manifest.tsv"
B_FIXTURES = B_DIR / "fixture_manifest.tsv"

OUT_DIR = ROOT / "results" / "simulation_model_matrix" / "05c"
SCENARIO_DIR = OUT_DIR / "scenarios"
MANIFEST_DIR = OUT_DIR / "scenario_manifests"

for d in [OUT_DIR, SCENARIO_DIR, MANIFEST_DIR]:
    d.mkdir(parents=True, exist_ok=True)

IMPLEMENTATION_CONTRACT = OUT_DIR / "model_implementation_contract.json"
SCENARIO_OUTPUT_MANIFEST = OUT_DIR / "scenario_output_manifest.tsv"
SUMMARY_JSON = OUT_DIR / "summary.json"

EXPECTED_05A_STATUS = (
    "PASS_NEGATIVE_TRANSFER_SIMULATION_AND_AI_SELECTION_CONTRACT_FROZEN"
)
EXPECTED_05B_STATUS = (
    "PASS_KNOWN_TRUTH_SIMULATION_RECIPES_AND_GENERATOR_FROZEN"
)

EXPECTED_N_SCENARIOS = 180
EXPECTED_TOTAL_REPLICATES = 21600
EXPECTED_N_MODULES = 50
EXPECTED_SOURCE_N = 186
EXPECTED_SOURCE_EVENTS = 124
EXPECTED_TARGET_TEST_N = 500

MAIN_MODELS = ["B0", "B4", "A0", "A1", "A2", "A3", "A4"]
ABLATIONS = [
    "A3_NO_EVOLUTION_PRIOR",
    "A3_NO_TARGET_OVERRIDE",
    "A3_PERMUTED_PRIOR",
    "A3_MAPPING_PERMUTATION",
    "A3_SOURCE_OUTCOME_PERMUTATION",
]
ALL_PREDICTION_MODELS = MAIN_MODELS + ABLATIONS

# ---------------------------------------------------------------------------
# Fixed model hyperparameters.
# These are NOT scenario tuned.
# ---------------------------------------------------------------------------

MODEL_BASE_SEED = 20260901

# Classical linear Cox anchors.
SOURCE_LINEAR_ALPHA = 1.0
TARGET_B0_ALPHA = 2.0
TARGET_B4_ALPHA = 2.0
LINEAR_EPOCHS_SOURCE = 120
LINEAR_EPOCHS_TARGET = 120
LINEAR_LR_SOURCE = 0.05
LINEAR_LR_TARGET = 0.05

# Shared source neural encoder.
HIDDEN_DIM = 32
LATENT_DIM = 16
SOURCE_MLP_EPOCHS = 120
SOURCE_MLP_LR = 0.010
SOURCE_MLP_WEIGHT_DECAY = 1e-4

# Target-only neural model A0.
A0_EPOCHS = 140
A0_LR = 0.010
A0_WEIGHT_DECAY = 1e-3

# Frozen encoder + head A1.
A1_EPOCHS = 80
A1_LR = 0.020
A1_HEAD_L2_TO_SOURCE = 1e-2

# Low-rank adapter A2.
A2_ADAPTER_RANK = 4
A2_EPOCHS = 100
A2_LR = 0.020
A2_ADAPTER_L2 = 1e-2

# Selective borrowing A3.
A3_EPOCHS = 120
A3_LR = 0.020
A3_PRIOR_PENALTY = 0.10
A3_GATE_OVERRIDE_L2 = 0.02
A3_RESIDUAL_L2 = 0.01
A3_LOGIT_EPS = 1e-4

# Full fine-tuning A4.
A4_EPOCHS = 80
A4_LR = 0.003
A4_WEIGHT_DECAY = 5e-4

GRAD_CLIP_NORM = 5.0
SD_EPS = 1e-6

# For numerical stability in Cox losses.
RISK_CLIP = 30.0

# GPU safety.
MIN_RECOMMENDED_FREE_GPU_BYTES = 3 * 1024**3


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
        raise FileNotFoundError(f"Required frozen artifact missing: {path}")
    return path


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
    spec.loader.exec_module(module)
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
                "PAPER6_SIM_DEVICE=cuda was requested but CUDA is unavailable."
            )
        raise RuntimeError(
            "CUDA is unavailable in this Paper-6 .venv. 05c is intentionally "
            "GPU-first because the frozen run contains 21,600 replicates. "
            "Install a CUDA-enabled PyTorch build, or explicitly set "
            "PAPER6_SIM_DEVICE=cpu to accept a much slower CPU run."
        )

    device = torch.device("cuda")

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    allow_busy = os.environ.get("PAPER6_ALLOW_BUSY_GPU", "0").strip() == "1"

    if free_bytes < MIN_RECOMMENDED_FREE_GPU_BYTES and not allow_busy:
        raise RuntimeError(
            "CUDA is available, but less than 3 GiB GPU memory is currently free. "
            "This may indicate another long-running GPU job. Wait for it to finish "
            "or set PAPER6_ALLOW_BUSY_GPU=1 if concurrent GPU use is intentional."
        )

    return device


def configure_torch(device: torch.device) -> None:
    torch.manual_seed(MODEL_BASE_SEED)
    np.random.seed(MODEL_BASE_SEED)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(MODEL_BASE_SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def implementation_payload(
    script_path: Path,
    device: torch.device,
    generator_contract: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "script_version": SCRIPT_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "status": "PASS_FROZEN_BEFORE_FIRST_05C_MODEL_FIT",
        "created_utc": now_utc(),
        "script_sha256": sha256_file(script_path),
        "05b_generator_script_sha256": clean(
            generator_contract.get("authoritative_generator_script_sha256")
        ),
        "models": MAIN_MODELS,
        "ablations": ABLATIONS,
        "architecture": {
            "input_modules": EXPECTED_N_MODULES,
            "hidden_dim": HIDDEN_DIM,
            "latent_dim": LATENT_DIM,
            "activation": "tanh",
            "A2_adapter_rank": A2_ADAPTER_RANK,
        },
        "classical": {
            "source_linear_alpha": SOURCE_LINEAR_ALPHA,
            "target_B0_alpha": TARGET_B0_ALPHA,
            "target_B4_alpha": TARGET_B4_ALPHA,
            "source_epochs": LINEAR_EPOCHS_SOURCE,
            "target_epochs": LINEAR_EPOCHS_TARGET,
            "source_lr": LINEAR_LR_SOURCE,
            "target_lr": LINEAR_LR_TARGET,
        },
        "neural": {
            "source_epochs": SOURCE_MLP_EPOCHS,
            "source_lr": SOURCE_MLP_LR,
            "source_weight_decay": SOURCE_MLP_WEIGHT_DECAY,
            "A0_epochs": A0_EPOCHS,
            "A0_lr": A0_LR,
            "A0_weight_decay": A0_WEIGHT_DECAY,
            "A1_epochs": A1_EPOCHS,
            "A1_lr": A1_LR,
            "A1_head_l2_to_source": A1_HEAD_L2_TO_SOURCE,
            "A2_epochs": A2_EPOCHS,
            "A2_lr": A2_LR,
            "A2_adapter_l2": A2_ADAPTER_L2,
            "A3_epochs": A3_EPOCHS,
            "A3_lr": A3_LR,
            "A3_prior_penalty": A3_PRIOR_PENALTY,
            "A3_gate_override_l2": A3_GATE_OVERRIDE_L2,
            "A3_residual_l2": A3_RESIDUAL_L2,
            "A4_epochs": A4_EPOCHS,
            "A4_lr": A4_LR,
            "A4_weight_decay": A4_WEIGHT_DECAY,
        },
        "optimization": {
            "optimizer": "Adam",
            "gradient_clip_norm": GRAD_CLIP_NORM,
            "early_stopping": False,
            "scenario_specific_hyperparameter_tuning": False,
            "fixed_epochs": True,
        },
        "transfer_implementation": {
            "module_mapping": (
                "mapping_index maps target module j to source-network input "
                "position mapping_index[j]; -1 drops the source correspondence"
            ),
            "B4": (
                "target linear Cox coefficients penalized toward the mapped "
                "source linear-Cox coefficient vector"
            ),
            "A1": (
                "source encoder frozen; target head initialized from source head "
                "and updated on target"
            ),
            "A2": (
                "source network frozen; rank-4 latent residual adapter only"
            ),
            "A3": (
                "source network frozen; module borrowing gates initialized from "
                "the scenario evolutionary prior and target-overridable; "
                "nonborrowed target signal enters a linear residual path"
            ),
            "A4": (
                "source network initialized from source-pretrained parameters and "
                "fully fine-tuned on mapped target modules"
            ),
        },
        "ablation_implementation": {
            "A3_NO_EVOLUTION_PRIOR": "replace prior_score by 0.5 for all modules",
            "A3_NO_TARGET_OVERRIDE": "gate fixed to prior_score; only target residual learned",
            "A3_PERMUTED_PRIOR": "deterministically permute prior_score within replicate",
            "A3_MAPPING_PERMUTATION": "deterministically replace source-target mapping by a full random permutation",
            "A3_SOURCE_OUTCOME_PERMUTATION": "retrain source MLP after deterministic within-replicate permutation of paired source (time,event) outcomes",
        },
        "device_first_run": str(device),
        "GSE16091_numeric_results_used": False,
        "real_data_values_read": False,
    }


def freeze_or_verify_implementation(
    payload: Dict[str, Any],
) -> None:
    if IMPLEMENTATION_CONTRACT.exists():
        existing = read_json(IMPLEMENTATION_CONTRACT)

        # Device is allowed to differ on restart, but scientific implementation
        # and script hash are not.
        for key in [
            "script_version",
            "implementation_version",
            "script_sha256",
            "05b_generator_script_sha256",
            "models",
            "ablations",
            "architecture",
            "classical",
            "neural",
            "optimization",
            "transfer_implementation",
            "ablation_implementation",
        ]:
            if existing.get(key) != payload.get(key):
                raise RuntimeError(
                    f"Existing 05c implementation contract differs at {key!r}. "
                    "Do not continue with a modified implementation."
                )
        return

    write_json(IMPLEMENTATION_CONTRACT, payload)


# ===========================================================================
# Batched synthetic survival utilities.
# ===========================================================================

def standardize_train_test(
    X_train: torch.Tensor,
    X_test: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    mean = X_train.mean(dim=1, keepdim=True)
    sd = X_train.std(dim=1, keepdim=True, unbiased=False)
    sd = torch.where(sd > SD_EPS, sd, torch.ones_like(sd))

    train_z = (X_train - mean) / sd
    test_z = None if X_test is None else (X_test - mean) / sd

    return train_z, test_z, mean, sd


def batched_cox_nll(
    risk: torch.Tensor,
    time_tensor: torch.Tensor,
    event_tensor: torch.Tensor,
) -> torch.Tensor:
    """
    Cox partial negative log-likelihood for continuous event times.

    risk/time/event shapes: [R, N]
    returns scalar mean across replicate-specific normalized losses.
    """
    risk = torch.clamp(risk, -RISK_CLIP, RISK_CLIP)

    order = torch.argsort(time_tensor, dim=1, descending=True)
    risk_sorted = torch.gather(risk, 1, order)
    event_sorted = torch.gather(event_tensor, 1, order)

    log_risk_sum = torch.logcumsumexp(risk_sorted, dim=1)
    contrib = (risk_sorted - log_risk_sum) * event_sorted

    event_count = event_sorted.sum(dim=1).clamp_min(1.0)
    per_rep = -contrib.sum(dim=1) / event_count
    return per_rep.mean()


def build_mapping_matrix(
    mapping_index: torch.Tensor,
    p: int,
) -> torch.Tensor:
    """
    mapping_index[r, target_j] = source_input_index or -1.
    Returns M[r, target_j, source_k].
    """
    r = mapping_index.shape[0]
    M = torch.zeros(
        (r, p, p),
        dtype=torch.float32,
        device=mapping_index.device,
    )

    valid = mapping_index >= 0
    rr, jj = torch.where(valid)
    kk = mapping_index[rr, jj].long()

    M[rr, jj, kk] = 1.0
    return M


def align_target_to_source(
    X_target: torch.Tensor,
    mapping_matrix: torch.Tensor,
) -> torch.Tensor:
    return torch.einsum("rnp,rpq->rnq", X_target, mapping_matrix)


def map_source_beta_to_target(
    beta_source: torch.Tensor,
    mapping_index: torch.Tensor,
) -> torch.Tensor:
    r, p = mapping_index.shape
    out = torch.zeros_like(beta_source)

    valid = mapping_index >= 0
    rr, jj = torch.where(valid)
    ss = mapping_index[rr, jj].long()
    out[rr, jj] = beta_source[rr, ss]
    return out


def deterministic_permutation_mapping(
    replicate_seeds: np.ndarray,
    p: int,
) -> np.ndarray:
    mappings = np.empty((len(replicate_seeds), p), dtype=np.int16)

    for i, seed in enumerate(replicate_seeds):
        rng = np.random.default_rng(int(seed) + 8_000_003)
        mappings[i] = rng.permutation(p).astype(np.int16)

    return mappings


def deterministic_permuted_prior(
    prior: np.ndarray,
    replicate_seeds: np.ndarray,
) -> np.ndarray:
    result = np.empty_like(prior)

    for i, seed in enumerate(replicate_seeds):
        rng = np.random.default_rng(int(seed) + 7_000_003)
        result[i] = prior[i, rng.permutation(prior.shape[1])]

    return result


def deterministic_outcome_pair_permutation(
    time_array: np.ndarray,
    event_array: np.ndarray,
    replicate_seeds: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    t = np.empty_like(time_array)
    e = np.empty_like(event_array)

    for i, seed in enumerate(replicate_seeds):
        rng = np.random.default_rng(int(seed) + 9_000_001)
        order = rng.permutation(time_array.shape[1])
        t[i] = time_array[i, order]
        e[i] = event_array[i, order]

    return t, e


# ===========================================================================
# Batched models.
# ===========================================================================

class BatchedMLP(nn.Module):
    def __init__(
        self,
        n_replicates: int,
        p: int,
        hidden_dim: int,
        latent_dim: int,
        *,
        device: torch.device,
        init_seed: int,
    ):
        super().__init__()

        gen = torch.Generator(device=device)
        gen.manual_seed(int(init_seed))

        self.W1 = nn.Parameter(
            torch.randn(
                (n_replicates, p, hidden_dim),
                generator=gen,
                device=device,
            )
            / math.sqrt(p)
        )
        self.b1 = nn.Parameter(
            torch.zeros(
                (n_replicates, hidden_dim),
                device=device,
            )
        )

        self.W2 = nn.Parameter(
            torch.randn(
                (n_replicates, hidden_dim, latent_dim),
                generator=gen,
                device=device,
            )
            / math.sqrt(hidden_dim)
        )
        self.b2 = nn.Parameter(
            torch.zeros(
                (n_replicates, latent_dim),
                device=device,
            )
        )

        self.head = nn.Parameter(
            torch.randn(
                (n_replicates, latent_dim),
                generator=gen,
                device=device,
            )
            / math.sqrt(latent_dim)
        )

    def encode(self, X: torch.Tensor) -> torch.Tensor:
        h1 = torch.tanh(
            torch.einsum("rnp,rph->rnh", X, self.W1)
            + self.b1[:, None, :]
        )
        z = torch.tanh(
            torch.einsum("rnh,rhl->rnl", h1, self.W2)
            + self.b2[:, None, :]
        )
        return z

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        z = self.encode(X)
        return torch.einsum("rnl,rl->rn", z, self.head)


def clone_batched_mlp(model: BatchedMLP) -> BatchedMLP:
    clone = BatchedMLP(
        model.W1.shape[0],
        model.W1.shape[1],
        model.W1.shape[2],
        model.W2.shape[2],
        device=model.W1.device,
        init_seed=1,
    )
    clone.load_state_dict(
        {k: v.detach().clone() for k, v in model.state_dict().items()}
    )
    return clone


def freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def train_linear_cox(
    X: torch.Tensor,
    time_tensor: torch.Tensor,
    event_tensor: torch.Tensor,
    *,
    center: Optional[torch.Tensor],
    alpha: float,
    epochs: int,
    lr: float,
    init_seed: int,
) -> Tuple[torch.Tensor, np.ndarray]:
    r, _, p = X.shape

    gen = torch.Generator(device=X.device)
    gen.manual_seed(int(init_seed))

    if center is None:
        center_tensor = torch.zeros(
            (r, p),
            dtype=X.dtype,
            device=X.device,
        )
        beta_init = (
            torch.randn(
                (r, p),
                generator=gen,
                device=X.device,
            )
            * 0.01
        )
    else:
        center_tensor = center.detach().clone()
        beta_init = center_tensor.clone()

    beta = nn.Parameter(beta_init)
    optimizer = torch.optim.Adam([beta], lr=float(lr))

    final_per_rep = None

    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)

        risk = torch.einsum("rnp,rp->rn", X, beta)
        cox = batched_cox_nll(risk, time_tensor, event_tensor)

        delta = beta - center_tensor
        penalty = 0.5 * float(alpha) * torch.mean(delta * delta)
        loss = cox + penalty

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite batched linear Cox loss.")

        loss.backward()
        torch.nn.utils.clip_grad_norm_([beta], GRAD_CLIP_NORM)
        optimizer.step()

    with torch.no_grad():
        risk = torch.einsum("rnp,rp->rn", X, beta)
        order = torch.argsort(time_tensor, dim=1, descending=True)
        rs = torch.gather(torch.clamp(risk, -RISK_CLIP, RISK_CLIP), 1, order)
        es = torch.gather(event_tensor, 1, order)
        lse = torch.logcumsumexp(rs, dim=1)
        counts = es.sum(dim=1).clamp_min(1.0)
        final_per_rep = (
            -((rs - lse) * es).sum(dim=1) / counts
        ).detach().cpu().numpy()

    return beta.detach(), final_per_rep.astype(np.float32)


def train_source_mlp(
    X: torch.Tensor,
    time_tensor: torch.Tensor,
    event_tensor: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    init_seed: int,
) -> Tuple[BatchedMLP, np.ndarray]:
    r, _, p = X.shape

    model = BatchedMLP(
        r,
        p,
        HIDDEN_DIM,
        LATENT_DIM,
        device=X.device,
        init_seed=init_seed,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(lr),
        weight_decay=float(weight_decay),
    )

    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        risk = model(X)
        loss = batched_cox_nll(
            risk,
            time_tensor,
            event_tensor,
        )

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite source MLP Cox loss.")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP_NORM,
        )
        optimizer.step()

    with torch.no_grad():
        risk = model(X)
        order = torch.argsort(time_tensor, dim=1, descending=True)
        rs = torch.gather(torch.clamp(risk, -RISK_CLIP, RISK_CLIP), 1, order)
        es = torch.gather(event_tensor, 1, order)
        lse = torch.logcumsumexp(rs, dim=1)
        counts = es.sum(dim=1).clamp_min(1.0)
        per_rep = (
            -((rs - lse) * es).sum(dim=1) / counts
        ).detach().cpu().numpy()

    return model, per_rep.astype(np.float32)


def train_A0(
    X_train: torch.Tensor,
    time_tensor: torch.Tensor,
    event_tensor: torch.Tensor,
    *,
    init_seed: int,
) -> Tuple[BatchedMLP, np.ndarray]:
    return train_source_mlp(
        X_train,
        time_tensor,
        event_tensor,
        epochs=A0_EPOCHS,
        lr=A0_LR,
        weight_decay=A0_WEIGHT_DECAY,
        init_seed=init_seed,
    )


def train_A1_head(
    source_model: BatchedMLP,
    X_train_aligned: torch.Tensor,
    time_tensor: torch.Tensor,
    event_tensor: torch.Tensor,
) -> Tuple[torch.Tensor, np.ndarray]:
    with torch.no_grad():
        z_train = source_model.encode(X_train_aligned).detach()

    source_head = source_model.head.detach().clone()
    head = nn.Parameter(source_head.clone())

    optimizer = torch.optim.Adam([head], lr=A1_LR)

    for _ in range(A1_EPOCHS):
        optimizer.zero_grad(set_to_none=True)

        risk = torch.einsum("rnl,rl->rn", z_train, head)
        cox = batched_cox_nll(risk, time_tensor, event_tensor)
        penalty = (
            0.5
            * A1_HEAD_L2_TO_SOURCE
            * torch.mean((head - source_head) ** 2)
        )
        loss = cox + penalty

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite A1 loss.")

        loss.backward()
        torch.nn.utils.clip_grad_norm_([head], GRAD_CLIP_NORM)
        optimizer.step()

    with torch.no_grad():
        risk = torch.einsum("rnl,rl->rn", z_train, head)
        final = per_replicate_cox_nll(
            risk,
            time_tensor,
            event_tensor,
        )

    return head.detach(), final.astype(np.float32)


def per_replicate_cox_nll(
    risk: torch.Tensor,
    time_tensor: torch.Tensor,
    event_tensor: torch.Tensor,
) -> np.ndarray:
    with torch.no_grad():
        order = torch.argsort(time_tensor, dim=1, descending=True)
        rs = torch.gather(
            torch.clamp(risk, -RISK_CLIP, RISK_CLIP),
            1,
            order,
        )
        es = torch.gather(event_tensor, 1, order)
        lse = torch.logcumsumexp(rs, dim=1)
        counts = es.sum(dim=1).clamp_min(1.0)
        values = -((rs - lse) * es).sum(dim=1) / counts
        return values.detach().cpu().numpy()


def train_A2_adapter(
    source_model: BatchedMLP,
    X_train_aligned: torch.Tensor,
    time_tensor: torch.Tensor,
    event_tensor: torch.Tensor,
    *,
    init_seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    with torch.no_grad():
        z = source_model.encode(X_train_aligned).detach()
        fixed_head = source_model.head.detach().clone()

    r = z.shape[0]
    device = z.device

    gen = torch.Generator(device=device)
    gen.manual_seed(int(init_seed))

    V = nn.Parameter(
        torch.randn(
            (r, LATENT_DIM, A2_ADAPTER_RANK),
            generator=gen,
            device=device,
        )
        * 0.01
    )
    U = nn.Parameter(
        torch.zeros(
            (r, A2_ADAPTER_RANK, LATENT_DIM),
            device=device,
        )
    )

    optimizer = torch.optim.Adam([V, U], lr=A2_LR)

    for _ in range(A2_EPOCHS):
        optimizer.zero_grad(set_to_none=True)

        low = torch.einsum("rnl,rlk->rnk", z, V)
        delta = torch.einsum("rnk,rkl->rnl", low, U)
        z_adapt = z + delta

        risk = torch.einsum(
            "rnl,rl->rn",
            z_adapt,
            fixed_head,
        )

        cox = batched_cox_nll(risk, time_tensor, event_tensor)
        penalty = 0.5 * A2_ADAPTER_L2 * (
            torch.mean(V * V) + torch.mean(U * U)
        )
        loss = cox + penalty

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite A2 loss.")

        loss.backward()
        torch.nn.utils.clip_grad_norm_([V, U], GRAD_CLIP_NORM)
        optimizer.step()

    with torch.no_grad():
        low = torch.einsum("rnl,rlk->rnk", z, V)
        delta = torch.einsum("rnk,rkl->rnl", low, U)
        risk = torch.einsum(
            "rnl,rl->rn",
            z + delta,
            fixed_head,
        )
        final = per_replicate_cox_nll(
            risk,
            time_tensor,
            event_tensor,
        )

    return V.detach(), U.detach(), final.astype(np.float32)


def prior_to_logits(prior: torch.Tensor) -> torch.Tensor:
    p = torch.clamp(
        prior,
        A3_LOGIT_EPS,
        1.0 - A3_LOGIT_EPS,
    )
    return torch.log(p) - torch.log1p(-p)


def train_A3(
    source_model: BatchedMLP,
    X_target_train: torch.Tensor,
    mapping_matrix: torch.Tensor,
    prior_score: torch.Tensor,
    time_tensor: torch.Tensor,
    event_tensor: torch.Tensor,
    *,
    allow_target_override: bool,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    source_prior = prior_score.detach().clone()

    if allow_target_override:
        delta_logit = nn.Parameter(
            torch.zeros_like(source_prior)
        )
        trainable: List[torch.Tensor] = [delta_logit]
    else:
        delta_logit = None
        trainable = []

    residual_beta = nn.Parameter(
        torch.zeros_like(source_prior)
    )
    trainable.append(residual_beta)

    optimizer = torch.optim.Adam(trainable, lr=A3_LR)

    prior_logits = prior_to_logits(source_prior)

    for _ in range(A3_EPOCHS):
        optimizer.zero_grad(set_to_none=True)

        if delta_logit is None:
            gate = source_prior
        else:
            gate = torch.sigmoid(
                prior_logits + delta_logit
            )

        gated_target = X_target_train * gate[:, None, :]
        aligned = align_target_to_source(
            gated_target,
            mapping_matrix,
        )

        source_risk = source_model(aligned)
        residual_risk = torch.einsum(
            "rnp,rp->rn",
            X_target_train * (1.0 - gate[:, None, :]),
            residual_beta,
        )
        risk = source_risk + residual_risk

        cox = batched_cox_nll(
            risk,
            time_tensor,
            event_tensor,
        )

        prior_penalty = (
            0.5
            * A3_PRIOR_PENALTY
            * torch.mean((gate - source_prior) ** 2)
        )
        residual_penalty = (
            0.5
            * A3_RESIDUAL_L2
            * torch.mean(residual_beta * residual_beta)
        )

        if delta_logit is None:
            override_penalty = torch.tensor(
                0.0,
                device=X_target_train.device,
            )
        else:
            override_penalty = (
                0.5
                * A3_GATE_OVERRIDE_L2
                * torch.mean(delta_logit * delta_logit)
            )

        loss = (
            cox
            + prior_penalty
            + residual_penalty
            + override_penalty
        )

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite A3 loss.")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            trainable,
            GRAD_CLIP_NORM,
        )
        optimizer.step()

    with torch.no_grad():
        if delta_logit is None:
            gate = source_prior
        else:
            gate = torch.sigmoid(
                prior_logits + delta_logit
            )

        aligned = align_target_to_source(
            X_target_train * gate[:, None, :],
            mapping_matrix,
        )
        source_risk = source_model(aligned)
        residual_risk = torch.einsum(
            "rnp,rp->rn",
            X_target_train * (1.0 - gate[:, None, :]),
            residual_beta,
        )
        risk = source_risk + residual_risk
        final = per_replicate_cox_nll(
            risk,
            time_tensor,
            event_tensor,
        )

    return (
        gate.detach(),
        residual_beta.detach(),
        final.astype(np.float32),
    )


def train_A4_full_finetune(
    source_model: BatchedMLP,
    X_train_aligned: torch.Tensor,
    time_tensor: torch.Tensor,
    event_tensor: torch.Tensor,
) -> Tuple[BatchedMLP, np.ndarray]:
    model = clone_batched_mlp(source_model)

    for p in model.parameters():
        p.requires_grad_(True)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=A4_LR,
        weight_decay=A4_WEIGHT_DECAY,
    )

    for _ in range(A4_EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        risk = model(X_train_aligned)
        loss = batched_cox_nll(
            risk,
            time_tensor,
            event_tensor,
        )

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite A4 loss.")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP_NORM,
        )
        optimizer.step()

    with torch.no_grad():
        final = per_replicate_cox_nll(
            model(X_train_aligned),
            time_tensor,
            event_tensor,
        )

    return model, final.astype(np.float32)


def predict_A1(
    source_model: BatchedMLP,
    head: torch.Tensor,
    X: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        z = source_model.encode(X)
        return torch.einsum("rnl,rl->rn", z, head)


def predict_A2(
    source_model: BatchedMLP,
    V: torch.Tensor,
    U: torch.Tensor,
    X: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        z = source_model.encode(X)
        low = torch.einsum("rnl,rlk->rnk", z, V)
        delta = torch.einsum("rnk,rkl->rnl", low, U)
        return torch.einsum(
            "rnl,rl->rn",
            z + delta,
            source_model.head,
        )


def predict_A3(
    source_model: BatchedMLP,
    gate: torch.Tensor,
    residual_beta: torch.Tensor,
    X_target: torch.Tensor,
    mapping_matrix: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        aligned = align_target_to_source(
            X_target * gate[:, None, :],
            mapping_matrix,
        )
        source_risk = source_model(aligned)
        residual_risk = torch.einsum(
            "rnp,rp->rn",
            X_target * (1.0 - gate[:, None, :]),
            residual_beta,
        )
        return source_risk + residual_risk


# ===========================================================================
# Scenario loading / generation.
# ===========================================================================

def load_scenario_recipe(
    path: Path,
) -> Dict[str, np.ndarray]:
    require_file(path)
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def generate_scenario_arrays(
    generator_module,
    scenario: Dict[str, Any],
    scenario_index: int,
    recipe: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    seeds = recipe["replicate_seed"].astype(np.int64)
    r = len(seeds)

    target_train_n = int(recipe["target_train_n"][0])
    target_test_n = int(recipe["target_test_n"][0])

    X_source = np.empty(
        (r, EXPECTED_SOURCE_N, EXPECTED_N_MODULES),
        dtype=np.float32,
    )
    source_time = np.empty(
        (r, EXPECTED_SOURCE_N),
        dtype=np.float32,
    )
    source_event = np.empty(
        (r, EXPECTED_SOURCE_N),
        dtype=np.uint8,
    )

    X_target_train = np.empty(
        (r, target_train_n, EXPECTED_N_MODULES),
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
        (r, target_test_n, EXPECTED_N_MODULES),
        dtype=np.float32,
    )
    target_test_time = np.empty(
        (r, target_test_n),
        dtype=np.float32,
    )
    target_test_event = np.empty(
        (r, target_test_n),
        dtype=np.uint8,
    )

    for rep, seed in enumerate(seeds):
        generated = generator_module.generate_full_replicate(
            int(seed),
            scenario,
            target_test_n=target_test_n,
        )

        # Known-truth recipe replay must match before any model fit.
        for key in [
            "causal_mask",
            "transportable_mask",
            "beta_source",
            "beta_target",
            "prior_score",
            "mapping_index",
        ]:
            if not np.array_equal(
                generated[key],
                recipe[key][rep],
            ):
                raise RuntimeError(
                    f"{scenario['scenario_id']} replicate {rep}: "
                    f"05b recipe replay mismatch for {key}."
                )

        X_source[rep] = generated["X_source"]
        source_time[rep] = generated["source_time"]
        source_event[rep] = generated["source_event"]

        X_target_train[rep] = generated["X_target_train"]
        target_train_time[rep] = generated["target_train_time"]
        target_train_event[rep] = generated["target_train_event"]

        X_target_test[rep] = generated["X_target_test"]
        target_test_time[rep] = generated["target_test_time"]
        target_test_event[rep] = generated["target_test_event"]

    if not np.all(source_event.sum(axis=1) == EXPECTED_SOURCE_EVENTS):
        raise RuntimeError(
            f"{scenario['scenario_id']}: source event-count identity failed."
        )

    expected_target_events = int(scenario["target_events"])
    if not np.all(
        target_train_event.sum(axis=1) == expected_target_events
    ):
        raise RuntimeError(
            f"{scenario['scenario_id']}: target-train event-count identity failed."
        )

    return {
        "replicate_seed": seeds,
        "causal_mask": recipe["causal_mask"],
        "transportable_mask": recipe["transportable_mask"],
        "beta_source_truth": recipe["beta_source"],
        "beta_target_truth": recipe["beta_target"],
        "prior_score": recipe["prior_score"],
        "mapping_index": recipe["mapping_index"],
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


def fixture_replay_preflight(
    generator_module,
    scenarios: pd.DataFrame,
    fixture_manifest: pd.DataFrame,
) -> None:
    for row in fixture_manifest.itertuples(index=False):
        fixture_path = ROOT / str(row.fixture_path)
        require_file(fixture_path)

        if sha256_file(fixture_path) != str(row.fixture_sha256):
            raise RuntimeError(
                f"05b fixture hash mismatch: {fixture_path}"
            )

        scenario_row = scenarios[
            scenarios["scenario_id"].astype(str)
            == str(row.scenario_id)
        ]
        if len(scenario_row) != 1:
            raise RuntimeError(
                f"Could not resolve fixture scenario {row.scenario_id}."
            )

        scenario = scenario_row.iloc[0].to_dict()
        replay = generator_module.generate_full_replicate(
            int(row.replicate_seed),
            scenario,
            target_test_n=int(row.target_test_n),
        )

        with np.load(fixture_path, allow_pickle=False) as stored:
            for key, value in replay.items():
                if key not in stored.files:
                    raise RuntimeError(
                        f"Fixture missing array {key}: {fixture_path}"
                    )
                if not np.array_equal(stored[key], value):
                    raise RuntimeError(
                        f"Fixture replay mismatch {row.scenario_id}:{key}"
                    )


# ===========================================================================
# Scenario model matrix.
# ===========================================================================

def scenario_output_paths(
    scenario_id: str,
) -> Tuple[Path, Path]:
    return (
        SCENARIO_DIR / f"{scenario_id}.npz",
        MANIFEST_DIR / f"{scenario_id}.json",
    )


def load_completed_scenario(
    scenario_id: str,
    implementation_hash: str,
    generator_hash: str,
) -> Optional[Dict[str, Any]]:
    output_path, manifest_path = scenario_output_paths(scenario_id)

    if not output_path.exists() or not manifest_path.exists():
        return None

    manifest = read_json(manifest_path)

    if clean(manifest.get("status")) != "PASS":
        return None
    if clean(manifest.get("implementation_contract_sha256")) != implementation_hash:
        return None
    if clean(manifest.get("generator_contract_sha256")) != generator_hash:
        return None
    if clean(manifest.get("output_sha256")) != sha256_file(output_path):
        return None

    return manifest


def run_scenario_model_matrix(
    arrays: Dict[str, np.ndarray],
    scenario: Dict[str, Any],
    scenario_index: int,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    seeds = arrays["replicate_seed"]
    r = len(seeds)
    p = EXPECTED_N_MODULES

    # ------------------------------------------------------------------
    # Move arrays to device and standardize source/target independently.
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

    Xs_z, _, _, _ = standardize_train_test(Xs)
    Xt_z, Xv_z, _, _ = standardize_train_test(Xt, Xv)

    if Xv_z is None:
        raise RuntimeError("Target-test standardization failed.")

    mapping_matrix = build_mapping_matrix(mapping, p)

    Xt_aligned = align_target_to_source(
        Xt_z,
        mapping_matrix,
    )
    Xv_aligned = align_target_to_source(
        Xv_z,
        mapping_matrix,
    )

    model_seed_base = (
        MODEL_BASE_SEED
        + scenario_index * 100_003
    )

    # ------------------------------------------------------------------
    # Classical source linear model and B0/B4.
    # ------------------------------------------------------------------
    beta_source_linear, loss_source_linear = train_linear_cox(
        Xs_z,
        ts,
        es,
        center=None,
        alpha=SOURCE_LINEAR_ALPHA,
        epochs=LINEAR_EPOCHS_SOURCE,
        lr=LINEAR_LR_SOURCE,
        init_seed=model_seed_base + 10,
    )

    beta_prior_target = map_source_beta_to_target(
        beta_source_linear,
        mapping,
    )

    beta_B0, loss_B0 = train_linear_cox(
        Xt_z,
        tt,
        et,
        center=None,
        alpha=TARGET_B0_ALPHA,
        epochs=LINEAR_EPOCHS_TARGET,
        lr=LINEAR_LR_TARGET,
        init_seed=model_seed_base + 20,
    )

    beta_B4, loss_B4 = train_linear_cox(
        Xt_z,
        tt,
        et,
        center=beta_prior_target,
        alpha=TARGET_B4_ALPHA,
        epochs=LINEAR_EPOCHS_TARGET,
        lr=LINEAR_LR_TARGET,
        init_seed=model_seed_base + 30,
    )

    with torch.no_grad():
        risk_train_B0 = torch.einsum(
            "rnp,rp->rn", Xt_z, beta_B0
        )
        risk_test_B0 = torch.einsum(
            "rnp,rp->rn", Xv_z, beta_B0
        )

        risk_train_B4 = torch.einsum(
            "rnp,rp->rn", Xt_z, beta_B4
        )
        risk_test_B4 = torch.einsum(
            "rnp,rp->rn", Xv_z, beta_B4
        )

    # ------------------------------------------------------------------
    # Shared source MLP pretraining.
    # ------------------------------------------------------------------
    source_model, loss_source_mlp = train_source_mlp(
        Xs_z,
        ts,
        es,
        epochs=SOURCE_MLP_EPOCHS,
        lr=SOURCE_MLP_LR,
        weight_decay=SOURCE_MLP_WEIGHT_DECAY,
        init_seed=model_seed_base + 100,
    )

    # Clone for A4 BEFORE freezing shared source network.
    A4_model, loss_A4 = train_A4_full_finetune(
        source_model,
        Xt_aligned,
        tt,
        et,
    )

    freeze_module(source_model)

    # ------------------------------------------------------------------
    # A0 target-only small neural survival network.
    # ------------------------------------------------------------------
    A0_model, loss_A0 = train_A0(
        Xt_z,
        tt,
        et,
        init_seed=model_seed_base + 200,
    )

    # ------------------------------------------------------------------
    # A1 target head only.
    # ------------------------------------------------------------------
    A1_head, loss_A1 = train_A1_head(
        source_model,
        Xt_aligned,
        tt,
        et,
    )

    # ------------------------------------------------------------------
    # A2 low-rank adapter only.
    # ------------------------------------------------------------------
    A2_V, A2_U, loss_A2 = train_A2_adapter(
        source_model,
        Xt_aligned,
        tt,
        et,
        init_seed=model_seed_base + 300,
    )

    # ------------------------------------------------------------------
    # A3 main.
    # ------------------------------------------------------------------
    gate_A3, residual_A3, loss_A3 = train_A3(
        source_model,
        Xt_z,
        mapping_matrix,
        prior,
        tt,
        et,
        allow_target_override=True,
    )

    # A3 no evolutionary prior.
    neutral_prior = torch.full_like(prior, 0.5)
    gate_no_prior, residual_no_prior, loss_no_prior = train_A3(
        source_model,
        Xt_z,
        mapping_matrix,
        neutral_prior,
        tt,
        et,
        allow_target_override=True,
    )

    # A3 no target override.
    gate_no_override, residual_no_override, loss_no_override = train_A3(
        source_model,
        Xt_z,
        mapping_matrix,
        prior,
        tt,
        et,
        allow_target_override=False,
    )

    # A3 permuted prior.
    perm_prior_np = deterministic_permuted_prior(
        arrays["prior_score"],
        seeds,
    )
    perm_prior = torch.as_tensor(
        perm_prior_np,
        dtype=torch.float32,
        device=device,
    )
    gate_perm_prior, residual_perm_prior, loss_perm_prior = train_A3(
        source_model,
        Xt_z,
        mapping_matrix,
        perm_prior,
        tt,
        et,
        allow_target_override=True,
    )

    # A3 mapping permutation.
    perm_mapping_np = deterministic_permutation_mapping(
        seeds,
        p,
    )
    perm_mapping = torch.as_tensor(
        perm_mapping_np,
        dtype=torch.long,
        device=device,
    )
    perm_mapping_matrix = build_mapping_matrix(
        perm_mapping,
        p,
    )
    gate_map_perm, residual_map_perm, loss_map_perm = train_A3(
        source_model,
        Xt_z,
        perm_mapping_matrix,
        prior,
        tt,
        et,
        allow_target_override=True,
    )

    # A3 source-outcome permutation.
    perm_source_time_np, perm_source_event_np = (
        deterministic_outcome_pair_permutation(
            arrays["source_time"],
            arrays["source_event"],
            seeds,
        )
    )
    perm_source_time = torch.as_tensor(
        perm_source_time_np,
        dtype=torch.float32,
        device=device,
    )
    perm_source_event = torch.as_tensor(
        perm_source_event_np,
        dtype=torch.float32,
        device=device,
    )

    source_perm_model, loss_source_perm = train_source_mlp(
        Xs_z,
        perm_source_time,
        perm_source_event,
        epochs=SOURCE_MLP_EPOCHS,
        lr=SOURCE_MLP_LR,
        weight_decay=SOURCE_MLP_WEIGHT_DECAY,
        init_seed=model_seed_base + 900,
    )
    freeze_module(source_perm_model)

    gate_source_perm, residual_source_perm, loss_source_perm_A3 = train_A3(
        source_perm_model,
        Xt_z,
        mapping_matrix,
        prior,
        tt,
        et,
        allow_target_override=True,
    )

    # ------------------------------------------------------------------
    # Predictions.
    # ------------------------------------------------------------------
    with torch.no_grad():
        risk_train_A0 = A0_model(Xt_z)
        risk_test_A0 = A0_model(Xv_z)

        risk_train_A1 = predict_A1(
            source_model,
            A1_head,
            Xt_aligned,
        )
        risk_test_A1 = predict_A1(
            source_model,
            A1_head,
            Xv_aligned,
        )

        risk_train_A2 = predict_A2(
            source_model,
            A2_V,
            A2_U,
            Xt_aligned,
        )
        risk_test_A2 = predict_A2(
            source_model,
            A2_V,
            A2_U,
            Xv_aligned,
        )

        risk_train_A3 = predict_A3(
            source_model,
            gate_A3,
            residual_A3,
            Xt_z,
            mapping_matrix,
        )
        risk_test_A3 = predict_A3(
            source_model,
            gate_A3,
            residual_A3,
            Xv_z,
            mapping_matrix,
        )

        risk_train_A4 = A4_model(Xt_aligned)
        risk_test_A4 = A4_model(Xv_aligned)

        risk_train_A3_no_prior = predict_A3(
            source_model,
            gate_no_prior,
            residual_no_prior,
            Xt_z,
            mapping_matrix,
        )
        risk_test_A3_no_prior = predict_A3(
            source_model,
            gate_no_prior,
            residual_no_prior,
            Xv_z,
            mapping_matrix,
        )

        risk_train_A3_no_override = predict_A3(
            source_model,
            gate_no_override,
            residual_no_override,
            Xt_z,
            mapping_matrix,
        )
        risk_test_A3_no_override = predict_A3(
            source_model,
            gate_no_override,
            residual_no_override,
            Xv_z,
            mapping_matrix,
        )

        risk_train_A3_perm_prior = predict_A3(
            source_model,
            gate_perm_prior,
            residual_perm_prior,
            Xt_z,
            mapping_matrix,
        )
        risk_test_A3_perm_prior = predict_A3(
            source_model,
            gate_perm_prior,
            residual_perm_prior,
            Xv_z,
            mapping_matrix,
        )

        risk_train_A3_map_perm = predict_A3(
            source_model,
            gate_map_perm,
            residual_map_perm,
            Xt_z,
            perm_mapping_matrix,
        )
        risk_test_A3_map_perm = predict_A3(
            source_model,
            gate_map_perm,
            residual_map_perm,
            Xv_z,
            perm_mapping_matrix,
        )

        risk_train_A3_source_perm = predict_A3(
            source_perm_model,
            gate_source_perm,
            residual_source_perm,
            Xt_z,
            mapping_matrix,
        )
        risk_test_A3_source_perm = predict_A3(
            source_perm_model,
            gate_source_perm,
            residual_source_perm,
            Xv_z,
            mapping_matrix,
        )

    # ------------------------------------------------------------------
    # Convert to compact NumPy output.
    # ------------------------------------------------------------------
    def cpu32(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy().astype(np.float32)

    output: Dict[str, np.ndarray] = {
        "replicate_seed": seeds.astype(np.int64),
        "target_train_time": arrays["target_train_time"].astype(np.float32),
        "target_train_event": arrays["target_train_event"].astype(np.uint8),
        "target_test_time": arrays["target_test_time"].astype(np.float32),
        "target_test_event": arrays["target_test_event"].astype(np.uint8),
        "transportable_mask": arrays["transportable_mask"].astype(np.uint8),
        "causal_mask": arrays["causal_mask"].astype(np.uint8),
        "prior_score": arrays["prior_score"].astype(np.float32),
        "mapping_index": arrays["mapping_index"].astype(np.int16),
        "beta_source_truth": arrays["beta_source_truth"].astype(np.float32),
        "beta_target_truth": arrays["beta_target_truth"].astype(np.float32),

        "risk_train_B0": cpu32(risk_train_B0),
        "risk_test_B0": cpu32(risk_test_B0),
        "risk_train_B4": cpu32(risk_train_B4),
        "risk_test_B4": cpu32(risk_test_B4),

        "risk_train_A0": cpu32(risk_train_A0),
        "risk_test_A0": cpu32(risk_test_A0),
        "risk_train_A1": cpu32(risk_train_A1),
        "risk_test_A1": cpu32(risk_test_A1),
        "risk_train_A2": cpu32(risk_train_A2),
        "risk_test_A2": cpu32(risk_test_A2),
        "risk_train_A3": cpu32(risk_train_A3),
        "risk_test_A3": cpu32(risk_test_A3),
        "risk_train_A4": cpu32(risk_train_A4),
        "risk_test_A4": cpu32(risk_test_A4),

        "risk_train_A3_NO_EVOLUTION_PRIOR": cpu32(
            risk_train_A3_no_prior
        ),
        "risk_test_A3_NO_EVOLUTION_PRIOR": cpu32(
            risk_test_A3_no_prior
        ),
        "risk_train_A3_NO_TARGET_OVERRIDE": cpu32(
            risk_train_A3_no_override
        ),
        "risk_test_A3_NO_TARGET_OVERRIDE": cpu32(
            risk_test_A3_no_override
        ),
        "risk_train_A3_PERMUTED_PRIOR": cpu32(
            risk_train_A3_perm_prior
        ),
        "risk_test_A3_PERMUTED_PRIOR": cpu32(
            risk_test_A3_perm_prior
        ),
        "risk_train_A3_MAPPING_PERMUTATION": cpu32(
            risk_train_A3_map_perm
        ),
        "risk_test_A3_MAPPING_PERMUTATION": cpu32(
            risk_test_A3_map_perm
        ),
        "risk_train_A3_SOURCE_OUTCOME_PERMUTATION": cpu32(
            risk_train_A3_source_perm
        ),
        "risk_test_A3_SOURCE_OUTCOME_PERMUTATION": cpu32(
            risk_test_A3_source_perm
        ),

        "gate_A3": cpu32(gate_A3),
        "gate_A3_NO_EVOLUTION_PRIOR": cpu32(gate_no_prior),
        "gate_A3_NO_TARGET_OVERRIDE": cpu32(gate_no_override),
        "gate_A3_PERMUTED_PRIOR": cpu32(gate_perm_prior),
        "gate_A3_MAPPING_PERMUTATION": cpu32(gate_map_perm),
        "gate_A3_SOURCE_OUTCOME_PERMUTATION": cpu32(
            gate_source_perm
        ),

        "beta_source_linear_fit": cpu32(beta_source_linear),
        "beta_B0_fit": cpu32(beta_B0),
        "beta_B4_fit": cpu32(beta_B4),

        "train_loss_source_linear": loss_source_linear,
        "train_loss_source_mlp": loss_source_mlp,
        "train_loss_B0": loss_B0,
        "train_loss_B4": loss_B4,
        "train_loss_A0": loss_A0,
        "train_loss_A1": loss_A1,
        "train_loss_A2": loss_A2,
        "train_loss_A3": loss_A3,
        "train_loss_A4": loss_A4,
        "train_loss_A3_NO_EVOLUTION_PRIOR": loss_no_prior,
        "train_loss_A3_NO_TARGET_OVERRIDE": loss_no_override,
        "train_loss_A3_PERMUTED_PRIOR": loss_perm_prior,
        "train_loss_A3_MAPPING_PERMUTATION": loss_map_perm,
        "train_loss_source_permuted_mlp": loss_source_perm,
        "train_loss_A3_SOURCE_OUTCOME_PERMUTATION": loss_source_perm_A3,

        "permuted_prior_score": perm_prior_np.astype(np.float32),
        "permuted_mapping_index": perm_mapping_np.astype(np.int16),
    }

    for name, arr in output.items():
        if np.issubdtype(arr.dtype, np.floating):
            if not np.isfinite(arr).all():
                raise RuntimeError(
                    f"{scenario['scenario_id']}: non-finite output array {name}."
                )

    return output


# ===========================================================================
# Main.
# ===========================================================================

def main() -> None:
    started = now_utc()

    print("=" * 120)
    print("Paper 6 - run closed synthetic classical/AI transfer model matrix")
    print("=" * 120)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Implementation version: {IMPLEMENTATION_VERSION}")
    print()

    print("Safety / scope:")
    print("  Real DOG2 outcome/expression values read: NO")
    print("  Real human outcome/expression values read: NO")
    print("  TARGET/GSE21257/GSE39055 outcomes read: NO")
    print("  GSE16091 used for architecture tuning: NO")
    print("  Closed models only: YES")
    print("  Scenario-specific hyperparameter tuning: NO")
    print("  Network access: NO")
    print()

    for path in [
        A_CONTRACT,
        A_SUMMARY,
        A_SCENARIOS,
        A_MODELS,
        A_SELECTION,
        B_CONTRACT,
        B_SUMMARY,
        B_MANIFEST,
        B_FIXTURES,
    ]:
        require_file(path)

    contract_05a = read_json(A_CONTRACT)
    summary_05a = read_json(A_SUMMARY)
    contract_05b = read_json(B_CONTRACT)
    summary_05b = read_json(B_SUMMARY)

    if clean(summary_05a.get("scientific_status")) != EXPECTED_05A_STATUS:
        raise RuntimeError("05a is not in expected frozen PASS state.")
    if clean(summary_05b.get("scientific_status")) != EXPECTED_05B_STATUS:
        raise RuntimeError("05b is not in expected frozen PASS state.")

    scenarios = pd.read_csv(A_SCENARIOS, sep="\t")
    recipe_manifest = pd.read_csv(B_MANIFEST, sep="\t")
    fixture_manifest = pd.read_csv(B_FIXTURES, sep="\t")

    if len(scenarios) != EXPECTED_N_SCENARIOS:
        raise RuntimeError(
            f"05a scenarios={len(scenarios)}, expected={EXPECTED_N_SCENARIOS}."
        )
    if int(scenarios["replicates"].astype(int).sum()) != EXPECTED_TOTAL_REPLICATES:
        raise RuntimeError("05a total replicate count changed.")
    if len(recipe_manifest) != EXPECTED_N_SCENARIOS:
        raise RuntimeError("05b recipe manifest scenario count changed.")

    # ------------------------------------------------------------------
    # Verify authoritative 05b generator.
    # ------------------------------------------------------------------
    generator_rel = clean(
        contract_05b.get("authoritative_generator_script")
    )
    generator_hash = clean(
        contract_05b.get("authoritative_generator_script_sha256")
    )
    generator_path = ROOT / generator_rel

    require_file(generator_path)

    if sha256_file(generator_path) != generator_hash:
        raise RuntimeError(
            "Authoritative 05b generator script SHA256 mismatch."
        )

    if sha256_file(B_MANIFEST) != clean(
        contract_05b.get("scenario_recipe_manifest_sha256")
    ):
        raise RuntimeError("05b recipe manifest SHA256 mismatch.")

    generator_module = load_module(
        generator_path,
        "paper6_05b_generator_for_05c",
    )

    fixture_replay_preflight(
        generator_module,
        scenarios,
        fixture_manifest,
    )

    print("05b generator SHA256 verification: PASS")
    print("05b representative fixture replay: PASS")

    # ------------------------------------------------------------------
    # Device and implementation freeze.
    # ------------------------------------------------------------------
    device = choose_device()
    configure_torch(device)

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        print(f"Execution device: CUDA [{gpu_name}]")
        print(
            f"CUDA memory free/total: "
            f"{free_bytes / 1024**3:.2f}/{total_bytes / 1024**3:.2f} GiB"
        )
    else:
        print("Execution device: CPU [explicitly requested]")

    script_path = Path(__file__).resolve()
    impl_payload = implementation_payload(
        script_path,
        device,
        contract_05b,
    )
    freeze_or_verify_implementation(impl_payload)
    impl_hash = sha256_file(IMPLEMENTATION_CONTRACT)
    generator_contract_hash = sha256_file(B_CONTRACT)

    print(f"05c implementation contract SHA256: {impl_hash}")
    print()

    # ------------------------------------------------------------------
    # Scenario loop.
    # ------------------------------------------------------------------
    manifest_rows: List[Dict[str, Any]] = []

    recipe_by_scenario = {
        str(row.scenario_id): row
        for row in recipe_manifest.itertuples(index=False)
    }

    total_done = 0
    total_reused = 0
    total_fitted = 0
    total_replicates_processed = 0

    wall_start = time.time()

    for scenario_index, row in scenarios.reset_index(drop=True).iterrows():
        scenario = row.to_dict()
        scenario_id = str(scenario["scenario_id"])

        existing = load_completed_scenario(
            scenario_id,
            impl_hash,
            generator_contract_hash,
        )

        if existing is not None:
            total_reused += 1
            total_done += 1
            total_replicates_processed += int(existing["replicates"])

            manifest_rows.append(
                {
                    "scenario_id": scenario_id,
                    "scenario_index": scenario_index,
                    "replicates": int(existing["replicates"]),
                    "status": "REUSED",
                    "output_path": existing["output_path"],
                    "output_sha256": existing["output_sha256"],
                    "output_size_bytes": int(existing["output_size_bytes"]),
                    "elapsed_seconds": float(
                        existing.get("elapsed_seconds", 0.0)
                    ),
                }
            )

            if total_done % 10 == 0 or total_done == len(scenarios):
                print(
                    f"  scenarios: {total_done}/{len(scenarios)} "
                    f"[reused={total_reused}, fitted={total_fitted}]"
                )
            continue

        scenario_start = time.time()

        recipe_row = recipe_by_scenario.get(scenario_id)
        if recipe_row is None:
            raise RuntimeError(
                f"No 05b recipe manifest row for {scenario_id}."
            )

        recipe_path = ROOT / str(recipe_row.recipe_path)
        require_file(recipe_path)

        if sha256_file(recipe_path) != str(recipe_row.recipe_sha256):
            raise RuntimeError(
                f"{scenario_id}: recipe shard SHA256 mismatch."
            )

        recipe = load_scenario_recipe(recipe_path)

        arrays = generate_scenario_arrays(
            generator_module,
            scenario,
            scenario_index,
            recipe,
        )

        result = run_scenario_model_matrix(
            arrays,
            scenario,
            scenario_index,
            device,
        )

        output_path, scenario_manifest_path = (
            scenario_output_paths(scenario_id)
        )
        atomic_savez_compressed(
            output_path,
            **result,
        )

        elapsed = time.time() - scenario_start

        scenario_manifest = {
            "status": "PASS",
            "created_utc": now_utc(),
            "scenario_id": scenario_id,
            "scenario_index": scenario_index,
            "replicates": int(scenario["replicates"]),
            "target_events": int(scenario["target_events"]),
            "transfer_regime": str(scenario["transfer_regime"]),
            "implementation_contract_sha256": impl_hash,
            "generator_contract_sha256": generator_contract_hash,
            "recipe_sha256": sha256_file(recipe_path),
            "output_path": str(output_path.relative_to(ROOT)),
            "output_sha256": sha256_file(output_path),
            "output_size_bytes": output_path.stat().st_size,
            "elapsed_seconds": elapsed,
            "device": str(device),
            "models": ALL_PREDICTION_MODELS,
        }
        write_json(
            scenario_manifest_path,
            scenario_manifest,
        )

        total_fitted += 1
        total_done += 1
        total_replicates_processed += int(scenario["replicates"])

        manifest_rows.append(
            {
                "scenario_id": scenario_id,
                "scenario_index": scenario_index,
                "replicates": int(scenario["replicates"]),
                "status": "FITTED",
                "output_path": scenario_manifest["output_path"],
                "output_sha256": scenario_manifest["output_sha256"],
                "output_size_bytes": int(
                    scenario_manifest["output_size_bytes"]
                ),
                "elapsed_seconds": elapsed,
            }
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()

        elapsed_total = time.time() - wall_start
        print(
            f"  {scenario_id}: PASS "
            f"[{int(scenario['replicates'])} reps, {elapsed:.1f}s] "
            f"overall {total_done}/{len(scenarios)}"
        )

        if total_done % 10 == 0:
            rate = total_done / max(elapsed_total, 1e-9)
            remaining = len(scenarios) - total_done
            eta_min = remaining / max(rate, 1e-9) / 60.0
            print(
                f"    progress: reused={total_reused}, fitted={total_fitted}, "
                f"approx remaining={eta_min:.1f} min"
            )

    manifest = pd.DataFrame(manifest_rows).sort_values(
        "scenario_index"
    )
    manifest.to_csv(
        SCENARIO_OUTPUT_MANIFEST,
        sep="\t",
        index=False,
    )

    if len(manifest) != EXPECTED_N_SCENARIOS:
        raise RuntimeError(
            "05c final output manifest does not contain all 180 scenarios."
        )
    if int(manifest["replicates"].sum()) != EXPECTED_TOTAL_REPLICATES:
        raise RuntimeError(
            "05c final output manifest replicate count mismatch."
        )

    total_output_bytes = int(
        manifest["output_size_bytes"].sum()
    )

    summary = {
        "script_version": SCRIPT_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "status": "PASS",
        "scientific_status": (
            "PASS_CLOSED_SYNTHETIC_TRANSFER_MODEL_MATRIX_COMPLETE"
        ),
        "run_started_utc": started,
        "run_finished_utc": now_utc(),
        "device": str(device),
        "n_scenarios": EXPECTED_N_SCENARIOS,
        "total_replicates": EXPECTED_TOTAL_REPLICATES,
        "models": MAIN_MODELS,
        "ablations": ABLATIONS,
        "scenario_outputs_reused_this_run": total_reused,
        "scenario_outputs_fitted_this_run": total_fitted,
        "output_storage_bytes": total_output_bytes,
        "implementation_contract_sha256": impl_hash,
        "generator_contract_sha256": generator_contract_hash,
        "scenario_output_manifest_sha256": sha256_file(
            SCENARIO_OUTPUT_MANIFEST
        ),
        "real_data_values_read": False,
        "GSE16091_used_for_architecture_selection": False,
        "scenario_specific_hyperparameter_tuning": False,
        "next": (
            "05d compute frozen Uno-C/IBS/calibration/negative-transfer/module-"
            "recovery metrics and apply the 05a A2-vs-A3 architecture-selection rule."
        ),
    }
    write_json(SUMMARY_JSON, summary)

    print()
    print("=" * 120)
    print("05c CLOSED SYNTHETIC TRANSFER MODEL MATRIX SUMMARY")
    print("=" * 120)
    print(f"Scenarios complete: {EXPECTED_N_SCENARIOS}/{EXPECTED_N_SCENARIOS}")
    print(f"Replicates complete: {EXPECTED_TOTAL_REPLICATES:,}")
    print(f"Main models: {', '.join(MAIN_MODELS)}")
    print(f"A3 ablations: {len(ABLATIONS)}")
    print(
        f"Stored model-matrix outputs: "
        f"{total_output_bytes / 1024**3:.2f} GiB"
    )
    print(f"Execution device: {device}")
    print()
    print("Real DOG2/human values read: NO")
    print("GSE16091 used for model selection: NO")
    print("Scenario-specific tuning: NO")
    print()
    print("Next: 05d aggregate phase diagram and apply frozen A2-vs-A3 selection rule.")
    print("=" * 120)
    print("05c closed synthetic transfer model matrix: PASS")
    print("=" * 120)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 120, file=sys.stderr)
        print(
            "05c closed synthetic transfer model matrix: FAIL",
            file=sys.stderr,
        )
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print("=" * 120, file=sys.stderr)
        raise
