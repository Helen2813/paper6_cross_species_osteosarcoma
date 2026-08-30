#!/usr/bin/env python3
"""
Paper 6 - evaluate frozen post-HOLD controlled transfer-mechanism experiment.

Stage 05e3 follows the completed 05e2 controlled model matrix.

Scientific role
---------------
Compute held-out survival metrics for M0-M6 and evaluate ONLY the four
prespecified 05e0 paired mechanism contrasts:

H1_HEAD_RETARGET_R5
H2_HEAD_CONSTRAINT_R5
H3_SOFT_LEAKAGE_R5
H4_ORACLE_GATE_R5

This stage performs no model fitting and no model selection.

Implementation discipline
-------------------------
- Reuses the exact frozen 05d IBS implementation (ibs_time_grid / safe_ibs)
  after verifying the original 05d script SHA against its metric contract.
- Replays Uno C directly with the exact 05d/05d0b time-support definition:
    tau = 80th percentile of training event times, truncated to common support,
    then nextafter toward -infinity.
- Freezes the 05e3 evaluation implementation and the interpretation rules
  BEFORE opening any 05e2 prediction array.
- Stores per-replicate metrics by cell and is restart-safe.

Contrast interpretation rules
-----------------------------
These rules are frozen here before the first 05e2 output is opened.

H1 HEAD RETARGETING:
  SUPPORTS if:
    mean paired delta UnoC(M2-M1) >= +0.02 at 29-event R5
    AND mean paired delta UnoC(M2-M1) >= 0 at 10-event R5
    AND mean paired delta UnoC(M2-M1) >= 0 at 40-event R5.

H2 TARGET-HEAD CONSTRAINT:
  SUPPORTS if:
    mean paired delta UnoC(M2-M3) >= +0.01 at 29-event R5
    AND mean paired delta IBS(M2-M3) <= 0
    AND mean paired delta test-risk-SD(M2-M3) <= 0.
  If the Uno threshold passes but the secondary directions are mixed:
    MIXED_SECONDARIES.
  No claim is made from Uno C alone.

H3 SOFT-GATE LEAKAGE:
  SUPPORTS if:
    mean paired delta UnoC(M5-M4) >= +0.02 at 29-event R5
    AND catastrophic-rate reduction (M4 - M5) >= 0.10.

H4 ORACLE-GATE LIMITATION:
  SUPPORTS if:
    mean paired delta UnoC(M6-M4) >= +0.02 at 29-event R5
    AND catastrophic-rate reduction (M4 - M6) >= 0.10.

Bootstrap:
- 5,000 paired replicate bootstraps within the fixed 29-event R5 cell.
- Percentile 95% CI and P_boot(delta>0) are uncertainty summaries only.
- Bootstrap does NOT change the frozen SUPPORTS / DOES_NOT_SUPPORT rule.

No A6 reopening.
No human outcomes.
CPU-only.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from sksurv.metrics import concordance_index_ipcw
    from sksurv.util import Surv
except ImportError as exc:
    raise ImportError("05e3 requires scikit-survival.") from exc


SCRIPT_VERSION = (
    "05e3-evaluate-frozen-controlled-mechanism-contrasts-v1-no-cli"
)
METRIC_IMPLEMENTATION_VERSION = (
    "paper6-posthold-controlled-mechanism-evaluation-v1"
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
# Completed 05e2.
# ---------------------------------------------------------------------------
E2_DIR = ROOT / "results" / "posthold_controlled_mechanism" / "05e2"
E2_IMPLEMENTATION = E2_DIR / "controlled_model_implementation_contract.json"
E2_MANIFEST = E2_DIR / "controlled_model_output_manifest.tsv"
E2_SUMMARY = E2_DIR / "summary.json"

EXPECTED_E2_STATUS = (
    "PASS_FROZEN_CONTROLLED_MECHANISM_MODEL_MATRIX_COMPLETE"
)
EXPECTED_E2_IMPLEMENTATION_SHA256 = (
    "a336ea56fc200452153beeab997f83f3ad03dc95cba3c02813f07066e1249e99"
)
EXPECTED_E2_MANIFEST_SHA256 = (
    "f49a5870ae7d14fabedab1bb969360b46b9119d35d1a5630af648a9b8a3fa538"
)

# ---------------------------------------------------------------------------
# Exact frozen 05d metric helpers.
# ---------------------------------------------------------------------------
D_SCRIPT = ROOT / "scripts" / "05d_aggregate_negative_transfer_phase_diagram.py"
D_DIR = ROOT / "results" / "simulation_phase_diagram" / "05d"
D_METRIC_CONTRACT = D_DIR / "metric_implementation_contract.json"

# ---------------------------------------------------------------------------
# 05e3 outputs.
# ---------------------------------------------------------------------------
OUT_DIR = ROOT / "results" / "posthold_controlled_mechanism" / "05e3"
CELL_METRIC_DIR = OUT_DIR / "cell_metrics"
CELL_MANIFEST_DIR = OUT_DIR / "cell_manifests"

for d in [OUT_DIR, CELL_METRIC_DIR, CELL_MANIFEST_DIR]:
    d.mkdir(parents=True, exist_ok=True)

METRIC_CONTRACT = OUT_DIR / "evaluation_implementation_contract.json"
METRIC_MANIFEST = OUT_DIR / "controlled_metric_manifest.tsv"
CELL_MODEL_SUMMARY = OUT_DIR / "cell_model_summary.tsv"
R5_LEARNING_TABLE = OUT_DIR / "R5_event_scaled_mechanism_table.tsv"
CONTRAST_SUMMARY = OUT_DIR / "prespecified_mechanism_contrast_summary.tsv"
SUMMARY_JSON = OUT_DIR / "summary.json"

EXPECTED_CELLS = 12
EXPECTED_REPLICATES = 6000

MODEL_NAMES = [
    "M0_B0",
    "M1_SOURCE_NETWORK_ZERO_SHOT",
    "M2_A1_SOURCE_CENTERED_HEAD",
    "M3_A1_FREE_HEAD",
    "M4_A3_SOFT_LEARNED",
    "M5_A3_HARDENED_PREDICTION",
    "M6_A3_ORACLE_HARD_GATE",
]
MODEL_INDEX = {m: i for i, m in enumerate(MODEL_NAMES)}

NEGATIVE_TRANSFER_DELTA = -0.02
CATASTROPHIC_DELTA = -0.05

PRIMARY_EVENTS = 29
PRIMARY_REGIME = "R5_MISLEADING_SOURCE"

H1_DELTA_C = 0.02
H2_DELTA_C = 0.01
H3_DELTA_C = 0.02
H4_DELTA_C = 0.02
H3_CATASTROPHIC_REDUCTION = 0.10
H4_CATASTROPHIC_REDUCTION = 0.10

BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 930_777_001


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


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


def make_surv(time_values: np.ndarray, event_values: np.ndarray) -> np.ndarray:
    return Surv.from_arrays(
        event=np.asarray(event_values, dtype=bool),
        time=np.asarray(time_values, dtype=float),
    )


def uno_tau(
    train_time: np.ndarray,
    train_event: np.ndarray,
    test_time: np.ndarray,
) -> Optional[float]:
    train_time = np.asarray(train_time, dtype=float)
    train_event = np.asarray(train_event, dtype=bool)
    test_time = np.asarray(test_time, dtype=float)

    event_times = train_time[train_event]
    if len(event_times) < 2:
        return None

    preferred = float(np.quantile(event_times, 0.80))
    upper = min(
        preferred,
        float(np.max(train_time)),
        float(np.max(test_time)),
    )
    upper = float(np.nextafter(upper, -np.inf))
    lower = max(
        float(np.min(train_time)),
        float(np.min(test_time)),
    )

    if not np.isfinite(upper) or upper <= lower:
        return None

    return upper


def safe_uno(
    train_time: np.ndarray,
    train_event: np.ndarray,
    test_time: np.ndarray,
    test_event: np.ndarray,
    risk_test: np.ndarray,
) -> float:
    tau = uno_tau(
        train_time,
        train_event,
        test_time,
    )
    if tau is None:
        return float("nan")

    try:
        return float(
            concordance_index_ipcw(
                make_surv(train_time, train_event),
                make_surv(test_time, test_event),
                np.asarray(risk_test, dtype=float),
                tau=tau,
            )[0]
        )
    except Exception:
        return float("nan")


def finite_mean(values: Iterable[float]) -> float:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if len(x) else float("nan")


def finite_median(values: Iterable[float]) -> float:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else float("nan")


def bootstrap_mean_diff(
    diff: np.ndarray,
    *,
    seed: int,
) -> Dict[str, float]:
    x = np.asarray(diff, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return {
            "mean": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_positive": float("nan"),
            "p_negative": float("nan"),
            "n": 0,
        }

    rng = np.random.default_rng(seed)
    n = len(x)
    boot = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)

    for b in range(BOOTSTRAP_DRAWS):
        idx = rng.integers(
            0,
            n,
            size=n,
            endpoint=False,
        )
        boot[b] = float(np.mean(x[idx]))

    return {
        "mean": float(np.mean(x)),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "p_positive": float(np.mean(boot > 0)),
        "p_negative": float(np.mean(boot < 0)),
        "n": int(n),
    }


def metric_output_paths(cell_id: str) -> Tuple[Path, Path]:
    return (
        CELL_METRIC_DIR / f"{cell_id}.npz",
        CELL_MANIFEST_DIR / f"{cell_id}.json",
    )


def metric_checkpoint_valid(
    *,
    cell_id: str,
    metric_contract_hash: str,
    source_output_hash: str,
) -> bool:
    metric_path, manifest_path = metric_output_paths(cell_id)

    if not metric_path.exists() or not manifest_path.exists():
        return False

    meta = read_json(manifest_path)

    return (
        clean(meta.get("status")) == "PASS"
        and clean(meta.get("evaluation_contract_sha256"))
        == metric_contract_hash
        and clean(meta.get("source_output_sha256"))
        == source_output_hash
        and clean(meta.get("metric_output_sha256"))
        == sha256_file(metric_path)
    )


def compute_cell_metrics(
    *,
    m05d,
    source_path: Path,
    cell_id: str,
) -> Dict[str, np.ndarray]:
    with np.load(source_path, allow_pickle=False) as data:
        replicate_seed = np.asarray(
            data["replicate_seed"],
            dtype=np.int64,
        )
        train_time = np.asarray(
            data["target_train_time"],
            dtype=float,
        )
        train_event = np.asarray(
            data["target_train_event"],
            dtype=np.uint8,
        )
        test_time = np.asarray(
            data["target_test_time"],
            dtype=float,
        )
        test_event = np.asarray(
            data["target_test_event"],
            dtype=np.uint8,
        )

        n_rep = len(replicate_seed)
        n_models = len(MODEL_NAMES)

        uno = np.full(
            (n_rep, n_models),
            np.nan,
            dtype=np.float64,
        )
        ibs = np.full(
            (n_rep, n_models),
            np.nan,
            dtype=np.float64,
        )
        risk_sd = np.full(
            (n_rep, n_models),
            np.nan,
            dtype=np.float64,
        )
        risk_q99_q01 = np.full(
            (n_rep, n_models),
            np.nan,
            dtype=np.float64,
        )

        risk_train = {}
        risk_test = {}

        for model in MODEL_NAMES:
            risk_train[model] = np.asarray(
                data[f"risk_train_{model}"],
                dtype=float,
            )
            risk_test[model] = np.asarray(
                data[f"risk_test_{model}"],
                dtype=float,
            )

            if risk_train[model].shape[0] != n_rep:
                raise RuntimeError(
                    f"{cell_id}: train-risk replicate count mismatch for {model}."
                )
            if risk_test[model].shape[0] != n_rep:
                raise RuntimeError(
                    f"{cell_id}: test-risk replicate count mismatch for {model}."
                )

        for r in range(n_rep):
            grid = m05d.ibs_time_grid(
                train_time[r],
                train_event[r],
                test_time[r],
            )

            for model in MODEL_NAMES:
                mi = MODEL_INDEX[model]

                trisk = risk_train[model][r]
                vrisk = risk_test[model][r]

                uno[r, mi] = safe_uno(
                    train_time[r],
                    train_event[r],
                    test_time[r],
                    test_event[r],
                    vrisk,
                )

                ibs[r, mi] = m05d.safe_ibs(
                    train_time[r],
                    train_event[r],
                    trisk,
                    test_time[r],
                    test_event[r],
                    vrisk,
                    grid,
                )

                risk_sd[r, mi] = float(
                    np.std(
                        vrisk,
                        ddof=0,
                    )
                )

                risk_q99_q01[r, mi] = float(
                    np.quantile(vrisk, 0.99)
                    - np.quantile(vrisk, 0.01)
                )

        b0 = uno[:, MODEL_INDEX["M0_B0"]][:, None]
        delta_c_vs_B0 = uno - b0

        negative_transfer = (
            delta_c_vs_B0
            <= NEGATIVE_TRANSFER_DELTA
        ).astype(np.uint8)

        catastrophic = (
            delta_c_vs_B0
            <= CATASTROPHIC_DELTA
        ).astype(np.uint8)

        # B0 is reference by definition.
        negative_transfer[:, MODEL_INDEX["M0_B0"]] = 0
        catastrophic[:, MODEL_INDEX["M0_B0"]] = 0

        # Basic metric invariants.
        if not np.allclose(
            delta_c_vs_B0[:, MODEL_INDEX["M0_B0"]],
            0.0,
            atol=1e-12,
            rtol=0,
            equal_nan=False,
        ):
            raise RuntimeError(
                f"{cell_id}: B0 delta-C identity failed."
            )

        return {
            "replicate_seed": replicate_seed,
            "uno_c": uno,
            "ibs": ibs,
            "test_risk_sd": risk_sd,
            "test_risk_q99_q01": risk_q99_q01,
            "delta_c_vs_B0": delta_c_vs_B0,
            "negative_transfer": negative_transfer,
            "catastrophic_negative_transfer": catastrophic,
        }


def freeze_or_verify_metric_contract(payload: Dict[str, Any]) -> None:
    if not METRIC_CONTRACT.exists():
        write_json(
            METRIC_CONTRACT,
            payload,
        )
        return

    existing = read_json(METRIC_CONTRACT)

    keys = [
        "script_version",
        "metric_implementation_version",
        "script_sha256",
        "05e0_contract_sha256",
        "05e2_implementation_sha256",
        "05e2_manifest_sha256",
        "05d_script_sha256",
        "05d_metric_contract_sha256",
        "models",
        "metrics",
        "contrast_rules",
        "bootstrap",
        "guardrails",
    ]

    for key in keys:
        if existing.get(key) != payload.get(key):
            raise RuntimeError(
                f"Existing 05e3 evaluation contract differs at {key!r}."
            )


def main() -> None:
    started = now_utc()

    print("=" * 120)
    print("Paper 6 - evaluate frozen controlled transfer-mechanism experiment")
    print("=" * 120)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Metric implementation: {METRIC_IMPLEMENTATION_VERSION}")
    print()
    print("Scope:")
    print("  05e2 predictions read: YES")
    print("  New model fitting: NO")
    print("  Prespecified H1-H4 contrasts only: YES")
    print("  Model selection: NO")
    print("  A6 reopened: NO")
    print("  Human outcomes read: NO")
    print("  GPU execution: NO")
    print()

    for path in [
        E0_CONTRACT,
        E0_SCENARIOS,
        E0_MODELS,
        E0_CONTRASTS,
        E0_SUMMARY,
        E2_IMPLEMENTATION,
        E2_MANIFEST,
        E2_SUMMARY,
        D_SCRIPT,
        D_METRIC_CONTRACT,
    ]:
        require_file(path)

    # ------------------------------------------------------------------
    # Verify frozen upstream state WITHOUT opening prediction arrays.
    # ------------------------------------------------------------------
    if sha256_file(E0_CONTRACT) != EXPECTED_E0_CONTRACT_SHA256:
        raise RuntimeError(
            "05e0 contract SHA256 changed."
        )

    e0_summary = read_json(E0_SUMMARY)
    if clean(e0_summary.get("scientific_status")) != EXPECTED_E0_STATUS:
        raise RuntimeError(
            "05e0 is not in expected frozen PASS state."
        )

    if sha256_file(E2_IMPLEMENTATION) != EXPECTED_E2_IMPLEMENTATION_SHA256:
        raise RuntimeError(
            "05e2 implementation contract SHA256 differs from completed run."
        )

    if sha256_file(E2_MANIFEST) != EXPECTED_E2_MANIFEST_SHA256:
        raise RuntimeError(
            "05e2 output manifest SHA256 differs from completed run."
        )

    e2_summary = read_json(E2_SUMMARY)
    if clean(e2_summary.get("scientific_status")) != EXPECTED_E2_STATUS:
        raise RuntimeError(
            "05e2 is not in expected PASS state."
        )

    if clean(
        e2_summary.get("implementation_contract_sha256")
    ) != EXPECTED_E2_IMPLEMENTATION_SHA256:
        raise RuntimeError(
            "05e2 summary implementation hash mismatch."
        )

    if clean(
        e2_summary.get("controlled_model_output_manifest_sha256")
    ) != EXPECTED_E2_MANIFEST_SHA256:
        raise RuntimeError(
            "05e2 summary output-manifest hash mismatch."
        )

    scenarios = pd.read_csv(
        E0_SCENARIOS,
        sep="\t",
    )
    models = pd.read_csv(
        E0_MODELS,
        sep="\t",
    )
    contrasts = pd.read_csv(
        E0_CONTRASTS,
        sep="\t",
    )
    e2_manifest = pd.read_csv(
        E2_MANIFEST,
        sep="\t",
    )

    if len(scenarios) != EXPECTED_CELLS:
        raise RuntimeError(
            "05e0 scenario cell count mismatch."
        )
    if len(e2_manifest) != EXPECTED_CELLS:
        raise RuntimeError(
            "05e2 output manifest cell count mismatch."
        )
    if int(e2_manifest["replicates"].sum()) != EXPECTED_REPLICATES:
        raise RuntimeError(
            "05e2 output manifest replicate count mismatch."
        )

    observed_model_names = models["model"].astype(str).tolist()
    if observed_model_names != MODEL_NAMES:
        raise RuntimeError(
            "05e0 controlled model registry changed."
        )

    expected_contrast_ids = {
        "H1_HEAD_RETARGET_R5",
        "H2_HEAD_CONSTRAINT_R5",
        "H3_SOFT_LEAKAGE_R5",
        "H4_ORACLE_GATE_R5",
    }
    observed_contrast_ids = set(
        contrasts["contrast_id"].astype(str)
    )
    if observed_contrast_ids != expected_contrast_ids:
        raise RuntimeError(
            "05e0 contrast registry differs from expected H1-H4 set."
        )

    # ------------------------------------------------------------------
    # Verify/import exact frozen 05d metric helper code.
    # ------------------------------------------------------------------
    d_metric_contract = read_json(
        D_METRIC_CONTRACT
    )
    expected_d_script_hash = clean(
        d_metric_contract.get("script_sha256")
    )
    observed_d_script_hash = sha256_file(
        D_SCRIPT
    )

    if not expected_d_script_hash:
        raise RuntimeError(
            "05d metric contract lacks script_sha256."
        )

    if observed_d_script_hash != expected_d_script_hash:
        raise RuntimeError(
            "Frozen 05d script hash mismatch."
        )

    m05d = load_module(
        D_SCRIPT,
        "paper6_frozen_05d_for_05e3",
    )

    for helper in [
        "ibs_time_grid",
        "safe_ibs",
    ]:
        if not hasattr(m05d, helper):
            raise RuntimeError(
                f"Frozen 05d implementation lacks helper {helper}."
            )

    # ------------------------------------------------------------------
    # Freeze 05e3 evaluation implementation BEFORE opening predictions.
    # ------------------------------------------------------------------
    metric_contract_payload = {
        "script_version": SCRIPT_VERSION,
        "metric_implementation_version": METRIC_IMPLEMENTATION_VERSION,
        "status": "PASS_FROZEN_BEFORE_FIRST_05E2_PREDICTION_READ",
        "created_utc": now_utc(),
        "script_sha256": sha256_file(
            Path(__file__).resolve()
        ),

        "05e0_contract_sha256": EXPECTED_E0_CONTRACT_SHA256,
        "05e2_implementation_sha256": EXPECTED_E2_IMPLEMENTATION_SHA256,
        "05e2_manifest_sha256": EXPECTED_E2_MANIFEST_SHA256,
        "05d_script_sha256": observed_d_script_hash,
        "05d_metric_contract_sha256": sha256_file(
            D_METRIC_CONTRACT
        ),

        "models": MODEL_NAMES,

        "metrics": {
            "Uno_C": (
                "IPCW concordance_index_ipcw; tau = 80th percentile of "
                "training event times truncated to common support and nextafter(-inf)"
            ),
            "IBS": (
                "exact frozen 05d ibs_time_grid + safe_ibs using train/test risks"
            ),
            "test_risk_SD": "population SD of held-out risk score",
            "test_risk_q99_q01": "held-out risk 99th minus 1st percentile",
            "delta_C_vs_B0": "UnoC(model) - UnoC(M0_B0)",
            "negative_transfer": "delta_C_vs_B0 <= -0.02",
            "catastrophic_negative_transfer": "delta_C_vs_B0 <= -0.05",
        },

        "contrast_rules": {
            "H1_HEAD_RETARGET_R5": {
                "primary_cell": "29-event R5",
                "primary": "mean paired UnoC(M2-M1) >= +0.02",
                "additional": (
                    "mean paired UnoC(M2-M1) >=0 at both 10-event and 40-event R5"
                ),
                "support_status": "SUPPORTS only if all conditions hold",
            },
            "H2_HEAD_CONSTRAINT_R5": {
                "primary_cell": "29-event R5",
                "primary": "mean paired UnoC(M2-M3) >= +0.01",
                "secondary": [
                    "mean paired IBS(M2-M3) <= 0",
                    "mean paired test-risk-SD(M2-M3) <= 0",
                ],
                "support_status": (
                    "SUPPORTS if primary and both secondary directions hold; "
                    "MIXED_SECONDARIES if primary holds but either secondary fails; "
                    "otherwise DOES_NOT_SUPPORT"
                ),
            },
            "H3_SOFT_LEAKAGE_R5": {
                "primary_cell": "29-event R5",
                "primary": "mean paired UnoC(M5-M4) >= +0.02",
                "additional": "catastrophic rate reduction M4-M5 >=0.10",
                "support_status": "SUPPORTS only if both conditions hold",
            },
            "H4_ORACLE_GATE_R5": {
                "primary_cell": "29-event R5",
                "primary": "mean paired UnoC(M6-M4) >= +0.02",
                "additional": "catastrophic rate reduction M4-M6 >=0.10",
                "support_status": "SUPPORTS only if both conditions hold",
                "oracle_nonimplementable": True,
            },
        },

        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "unit": "paired replicate within fixed cell",
            "interval": "percentile 95%",
            "p_positive": "fraction bootstrap mean differences >0",
            "affects_support_status": False,
        },

        "guardrails": {
            "no_model_fitting": True,
            "no_model_selection": True,
            "no_A6_reopening": True,
            "no_posthoc_grid_expansion": True,
            "oracle_M6_nonimplementable": True,
            "human_outcomes_read": False,
            "real_data_values_read": False,
        },
    }

    freeze_or_verify_metric_contract(
        metric_contract_payload
    )

    metric_contract_hash = sha256_file(
        METRIC_CONTRACT
    )

    print(
        f"05e3 evaluation contract SHA256: "
        f"{metric_contract_hash}"
    )
    print(
        "Evaluation/interpretation frozen before first 05e2 prediction read: PASS"
    )
    print()

    # ------------------------------------------------------------------
    # Per-cell metric computation / checkpoint reuse.
    # ------------------------------------------------------------------
    scenario_by_cell = {
        str(row.cell_id): row
        for row in scenarios.itertuples(index=False)
    }

    metric_manifest_rows: List[Dict[str, Any]] = []

    for i, source_row in enumerate(
        e2_manifest.sort_values("cell_index").itertuples(index=False),
        start=1,
    ):
        cell_id = str(source_row.cell_id)
        source_path = ROOT / str(
            source_row.output_path
        )
        require_file(source_path)

        source_hash = sha256_file(
            source_path
        )
        if source_hash != str(
            source_row.output_sha256
        ):
            raise RuntimeError(
                f"{cell_id}: 05e2 source-output hash mismatch."
            )

        metric_path, cell_manifest_path = metric_output_paths(
            cell_id
        )

        if metric_checkpoint_valid(
            cell_id=cell_id,
            metric_contract_hash=metric_contract_hash,
            source_output_hash=source_hash,
        ):
            status = "REUSED"
        else:
            t0 = time.time()

            metrics = compute_cell_metrics(
                m05d=m05d,
                source_path=source_path,
                cell_id=cell_id,
            )

            atomic_savez_compressed(
                metric_path,
                **metrics,
            )

            elapsed = time.time() - t0

            meta = {
                "status": "PASS",
                "created_utc": now_utc(),
                "cell_id": cell_id,
                "replicates": int(
                    len(metrics["replicate_seed"])
                ),
                "evaluation_contract_sha256": metric_contract_hash,
                "source_output_sha256": source_hash,
                "metric_output_path": str(
                    metric_path.relative_to(ROOT)
                ),
                "metric_output_sha256": sha256_file(
                    metric_path
                ),
                "elapsed_seconds": elapsed,
            }
            write_json(
                cell_manifest_path,
                meta,
            )
            status = "COMPUTED"

        meta = read_json(
            cell_manifest_path
        )
        scenario_row = scenario_by_cell[cell_id]

        metric_manifest_rows.append({
            "cell_id": cell_id,
            "cell_index": int(
                source_row.cell_index
            ),
            "target_events": int(
                scenario_row.target_events
            ),
            "transfer_regime": str(
                scenario_row.transfer_regime
            ),
            "replicates": int(
                meta["replicates"]
            ),
            "status": status,
            "metric_output_path": meta[
                "metric_output_path"
            ],
            "metric_output_sha256": meta[
                "metric_output_sha256"
            ],
            "source_output_sha256": source_hash,
            "elapsed_seconds": float(
                meta.get(
                    "elapsed_seconds",
                    0.0,
                )
            ),
        })

        print(
            f"  {cell_id}: {status} metrics "
            f"overall {i}/{EXPECTED_CELLS}"
        )

    metric_manifest_df = pd.DataFrame(
        metric_manifest_rows
    ).sort_values("cell_index")

    metric_manifest_df.to_csv(
        METRIC_MANIFEST,
        sep="\t",
        index=False,
    )

    # ------------------------------------------------------------------
    # Aggregate cell/model summaries.
    # ------------------------------------------------------------------
    cell_model_rows: List[Dict[str, Any]] = []

    for row in metric_manifest_df.itertuples(index=False):
        metric_path = ROOT / str(
            row.metric_output_path
        )

        with np.load(
            metric_path,
            allow_pickle=False,
        ) as data:
            uno = np.asarray(
                data["uno_c"],
                dtype=float,
            )
            ibs = np.asarray(
                data["ibs"],
                dtype=float,
            )
            risk_sd = np.asarray(
                data["test_risk_sd"],
                dtype=float,
            )
            risk_qrange = np.asarray(
                data["test_risk_q99_q01"],
                dtype=float,
            )
            delta = np.asarray(
                data["delta_c_vs_B0"],
                dtype=float,
            )
            nt = np.asarray(
                data["negative_transfer"],
                dtype=np.uint8,
            )
            catastrophic = np.asarray(
                data["catastrophic_negative_transfer"],
                dtype=np.uint8,
            )

            for model in MODEL_NAMES:
                mi = MODEL_INDEX[model]

                cell_model_rows.append({
                    "cell_id": str(row.cell_id),
                    "target_events": int(
                        row.target_events
                    ),
                    "transfer_regime": str(
                        row.transfer_regime
                    ),
                    "model": model,
                    "n_replicates": int(
                        uno.shape[0]
                    ),
                    "mean_uno_c": finite_mean(
                        uno[:, mi]
                    ),
                    "median_uno_c": finite_median(
                        uno[:, mi]
                    ),
                    "mean_delta_c_vs_B0": finite_mean(
                        delta[:, mi]
                    ),
                    "mean_ibs": finite_mean(
                        ibs[:, mi]
                    ),
                    "median_ibs": finite_median(
                        ibs[:, mi]
                    ),
                    "mean_test_risk_sd": finite_mean(
                        risk_sd[:, mi]
                    ),
                    "median_test_risk_sd": finite_median(
                        risk_sd[:, mi]
                    ),
                    "mean_test_risk_q99_q01": finite_mean(
                        risk_qrange[:, mi]
                    ),
                    "negative_transfer_rate": finite_mean(
                        nt[:, mi]
                    ),
                    "catastrophic_negative_transfer_rate": finite_mean(
                        catastrophic[:, mi]
                    ),
                })

    cell_model_df = pd.DataFrame(
        cell_model_rows
    ).sort_values(
        [
            "target_events",
            "transfer_regime",
            "model",
        ]
    )

    cell_model_df.to_csv(
        CELL_MODEL_SUMMARY,
        sep="\t",
        index=False,
    )

    r5_table = cell_model_df[
        cell_model_df["transfer_regime"]
        == PRIMARY_REGIME
    ].copy()

    r5_table.to_csv(
        R5_LEARNING_TABLE,
        sep="\t",
        index=False,
    )

    # ------------------------------------------------------------------
    # Paired prespecified contrast evaluation.
    # ------------------------------------------------------------------
    metric_by_cell = {
        str(row.cell_id): ROOT / str(
            row.metric_output_path
        )
        for row in metric_manifest_df.itertuples(index=False)
    }

    cell_lookup = {
        (
            int(row.target_events),
            str(row.transfer_regime),
        ): str(row.cell_id)
        for row in metric_manifest_df.itertuples(index=False)
    }

    def load_metric_cell(
        events: int,
        regime: str,
    ) -> Dict[str, np.ndarray]:
        cell_id = cell_lookup.get(
            (
                int(events),
                str(regime),
            )
        )
        if cell_id is None:
            raise RuntimeError(
                f"No controlled cell for events={events}, regime={regime}."
            )

        with np.load(
            metric_by_cell[cell_id],
            allow_pickle=False,
        ) as data:
            return {
                key: data[key].copy()
                for key in data.files
            }

    r5_10 = load_metric_cell(
        10,
        PRIMARY_REGIME,
    )
    r5_29 = load_metric_cell(
        29,
        PRIMARY_REGIME,
    )
    r5_40 = load_metric_cell(
        40,
        PRIMARY_REGIME,
    )

    contrast_rows: List[Dict[str, Any]] = []

    # H1 ----------------------------------------------------------------
    m2 = MODEL_INDEX[
        "M2_A1_SOURCE_CENTERED_HEAD"
    ]
    m1 = MODEL_INDEX[
        "M1_SOURCE_NETWORK_ZERO_SHOT"
    ]

    h1_d29 = (
        r5_29["uno_c"][:, m2]
        - r5_29["uno_c"][:, m1]
    )
    h1_d10 = (
        r5_10["uno_c"][:, m2]
        - r5_10["uno_c"][:, m1]
    )
    h1_d40 = (
        r5_40["uno_c"][:, m2]
        - r5_40["uno_c"][:, m1]
    )

    h1_boot = bootstrap_mean_diff(
        h1_d29,
        seed=BOOTSTRAP_SEED + 1,
    )
    h1_mean10 = finite_mean(
        h1_d10
    )
    h1_mean40 = finite_mean(
        h1_d40
    )
    h1_support = bool(
        h1_boot["mean"] >= H1_DELTA_C
        and h1_mean10 >= 0
        and h1_mean40 >= 0
    )

    contrast_rows.append({
        "contrast_id": "H1_HEAD_RETARGET_R5",
        "treatment": "M2_A1_SOURCE_CENTERED_HEAD",
        "control": "M1_SOURCE_NETWORK_ZERO_SHOT",
        "primary_events": 29,
        "transfer_regime": PRIMARY_REGIME,
        "mean_delta_uno_c": h1_boot["mean"],
        "bootstrap_ci_low": h1_boot["ci_low"],
        "bootstrap_ci_high": h1_boot["ci_high"],
        "bootstrap_p_positive": h1_boot[
            "p_positive"
        ],
        "mean_delta_uno_c_at_10_events": h1_mean10,
        "mean_delta_uno_c_at_40_events": h1_mean40,
        "mean_delta_ibs": finite_mean(
            r5_29["ibs"][:, m2]
            - r5_29["ibs"][:, m1]
        ),
        "mean_delta_test_risk_sd": finite_mean(
            r5_29["test_risk_sd"][:, m2]
            - r5_29["test_risk_sd"][:, m1]
        ),
        "catastrophic_rate_treatment": finite_mean(
            r5_29[
                "catastrophic_negative_transfer"
            ][:, m2]
        ),
        "catastrophic_rate_control": finite_mean(
            r5_29[
                "catastrophic_negative_transfer"
            ][:, m1]
        ),
        "catastrophic_rate_reduction_control_minus_treatment": (
            finite_mean(
                r5_29[
                    "catastrophic_negative_transfer"
                ][:, m1]
            )
            - finite_mean(
                r5_29[
                    "catastrophic_negative_transfer"
                ][:, m2]
            )
        ),
        "frozen_support_status": (
            "SUPPORTS"
            if h1_support
            else "DOES_NOT_SUPPORT"
        ),
    })

    # H2 ----------------------------------------------------------------
    m3 = MODEL_INDEX[
        "M3_A1_FREE_HEAD"
    ]

    h2_d29 = (
        r5_29["uno_c"][:, m2]
        - r5_29["uno_c"][:, m3]
    )
    h2_ibs = (
        r5_29["ibs"][:, m2]
        - r5_29["ibs"][:, m3]
    )
    h2_risk_sd = (
        r5_29["test_risk_sd"][:, m2]
        - r5_29["test_risk_sd"][:, m3]
    )

    h2_boot = bootstrap_mean_diff(
        h2_d29,
        seed=BOOTSTRAP_SEED + 2,
    )

    h2_primary = bool(
        h2_boot["mean"] >= H2_DELTA_C
    )
    h2_ibs_ok = bool(
        finite_mean(h2_ibs) <= 0
    )
    h2_scale_ok = bool(
        finite_mean(h2_risk_sd) <= 0
    )

    if (
        h2_primary
        and h2_ibs_ok
        and h2_scale_ok
    ):
        h2_status = "SUPPORTS"
    elif h2_primary:
        h2_status = "MIXED_SECONDARIES"
    else:
        h2_status = "DOES_NOT_SUPPORT"

    contrast_rows.append({
        "contrast_id": "H2_HEAD_CONSTRAINT_R5",
        "treatment": "M2_A1_SOURCE_CENTERED_HEAD",
        "control": "M3_A1_FREE_HEAD",
        "primary_events": 29,
        "transfer_regime": PRIMARY_REGIME,
        "mean_delta_uno_c": h2_boot["mean"],
        "bootstrap_ci_low": h2_boot["ci_low"],
        "bootstrap_ci_high": h2_boot["ci_high"],
        "bootstrap_p_positive": h2_boot[
            "p_positive"
        ],
        "mean_delta_uno_c_at_10_events": float("nan"),
        "mean_delta_uno_c_at_40_events": float("nan"),
        "mean_delta_ibs": finite_mean(
            h2_ibs
        ),
        "mean_delta_test_risk_sd": finite_mean(
            h2_risk_sd
        ),
        "catastrophic_rate_treatment": finite_mean(
            r5_29[
                "catastrophic_negative_transfer"
            ][:, m2]
        ),
        "catastrophic_rate_control": finite_mean(
            r5_29[
                "catastrophic_negative_transfer"
            ][:, m3]
        ),
        "catastrophic_rate_reduction_control_minus_treatment": (
            finite_mean(
                r5_29[
                    "catastrophic_negative_transfer"
                ][:, m3]
            )
            - finite_mean(
                r5_29[
                    "catastrophic_negative_transfer"
                ][:, m2]
            )
        ),
        "frozen_support_status": h2_status,
    })

    # H3 ----------------------------------------------------------------
    m5 = MODEL_INDEX[
        "M5_A3_HARDENED_PREDICTION"
    ]
    m4 = MODEL_INDEX[
        "M4_A3_SOFT_LEARNED"
    ]

    h3_d29 = (
        r5_29["uno_c"][:, m5]
        - r5_29["uno_c"][:, m4]
    )
    h3_boot = bootstrap_mean_diff(
        h3_d29,
        seed=BOOTSTRAP_SEED + 3,
    )
    h3_cat_reduction = (
        finite_mean(
            r5_29[
                "catastrophic_negative_transfer"
            ][:, m4]
        )
        - finite_mean(
            r5_29[
                "catastrophic_negative_transfer"
            ][:, m5]
        )
    )
    h3_support = bool(
        h3_boot["mean"] >= H3_DELTA_C
        and h3_cat_reduction
        >= H3_CATASTROPHIC_REDUCTION
    )

    contrast_rows.append({
        "contrast_id": "H3_SOFT_LEAKAGE_R5",
        "treatment": "M5_A3_HARDENED_PREDICTION",
        "control": "M4_A3_SOFT_LEARNED",
        "primary_events": 29,
        "transfer_regime": PRIMARY_REGIME,
        "mean_delta_uno_c": h3_boot["mean"],
        "bootstrap_ci_low": h3_boot["ci_low"],
        "bootstrap_ci_high": h3_boot["ci_high"],
        "bootstrap_p_positive": h3_boot[
            "p_positive"
        ],
        "mean_delta_uno_c_at_10_events": finite_mean(
            r5_10["uno_c"][:, m5]
            - r5_10["uno_c"][:, m4]
        ),
        "mean_delta_uno_c_at_40_events": finite_mean(
            r5_40["uno_c"][:, m5]
            - r5_40["uno_c"][:, m4]
        ),
        "mean_delta_ibs": finite_mean(
            r5_29["ibs"][:, m5]
            - r5_29["ibs"][:, m4]
        ),
        "mean_delta_test_risk_sd": finite_mean(
            r5_29["test_risk_sd"][:, m5]
            - r5_29["test_risk_sd"][:, m4]
        ),
        "catastrophic_rate_treatment": finite_mean(
            r5_29[
                "catastrophic_negative_transfer"
            ][:, m5]
        ),
        "catastrophic_rate_control": finite_mean(
            r5_29[
                "catastrophic_negative_transfer"
            ][:, m4]
        ),
        "catastrophic_rate_reduction_control_minus_treatment": (
            h3_cat_reduction
        ),
        "frozen_support_status": (
            "SUPPORTS"
            if h3_support
            else "DOES_NOT_SUPPORT"
        ),
    })

    # H4 ----------------------------------------------------------------
    m6 = MODEL_INDEX[
        "M6_A3_ORACLE_HARD_GATE"
    ]

    h4_d29 = (
        r5_29["uno_c"][:, m6]
        - r5_29["uno_c"][:, m4]
    )
    h4_boot = bootstrap_mean_diff(
        h4_d29,
        seed=BOOTSTRAP_SEED + 4,
    )
    h4_cat_reduction = (
        finite_mean(
            r5_29[
                "catastrophic_negative_transfer"
            ][:, m4]
        )
        - finite_mean(
            r5_29[
                "catastrophic_negative_transfer"
            ][:, m6]
        )
    )
    h4_support = bool(
        h4_boot["mean"] >= H4_DELTA_C
        and h4_cat_reduction
        >= H4_CATASTROPHIC_REDUCTION
    )

    contrast_rows.append({
        "contrast_id": "H4_ORACLE_GATE_R5",
        "treatment": "M6_A3_ORACLE_HARD_GATE",
        "control": "M4_A3_SOFT_LEARNED",
        "primary_events": 29,
        "transfer_regime": PRIMARY_REGIME,
        "mean_delta_uno_c": h4_boot["mean"],
        "bootstrap_ci_low": h4_boot["ci_low"],
        "bootstrap_ci_high": h4_boot["ci_high"],
        "bootstrap_p_positive": h4_boot[
            "p_positive"
        ],
        "mean_delta_uno_c_at_10_events": finite_mean(
            r5_10["uno_c"][:, m6]
            - r5_10["uno_c"][:, m4]
        ),
        "mean_delta_uno_c_at_40_events": finite_mean(
            r5_40["uno_c"][:, m6]
            - r5_40["uno_c"][:, m4]
        ),
        "mean_delta_ibs": finite_mean(
            r5_29["ibs"][:, m6]
            - r5_29["ibs"][:, m4]
        ),
        "mean_delta_test_risk_sd": finite_mean(
            r5_29["test_risk_sd"][:, m6]
            - r5_29["test_risk_sd"][:, m4]
        ),
        "catastrophic_rate_treatment": finite_mean(
            r5_29[
                "catastrophic_negative_transfer"
            ][:, m6]
        ),
        "catastrophic_rate_control": finite_mean(
            r5_29[
                "catastrophic_negative_transfer"
            ][:, m4]
        ),
        "catastrophic_rate_reduction_control_minus_treatment": (
            h4_cat_reduction
        ),
        "frozen_support_status": (
            "SUPPORTS"
            if h4_support
            else "DOES_NOT_SUPPORT"
        ),
    })

    contrast_df = pd.DataFrame(
        contrast_rows
    )

    contrast_df.to_csv(
        CONTRAST_SUMMARY,
        sep="\t",
        index=False,
    )

    # ------------------------------------------------------------------
    # Final status.
    # ------------------------------------------------------------------
    support_map = {
        str(row.contrast_id): str(
            row.frozen_support_status
        )
        for row in contrast_df.itertuples(index=False)
    }

    summary = {
        "script_version": SCRIPT_VERSION,
        "metric_implementation_version": METRIC_IMPLEMENTATION_VERSION,
        "status": "PASS",
        "scientific_status": (
            "PASS_CONTROLLED_MECHANISM_CONTRAST_EVALUATION_COMPLETE"
        ),
        "run_started_utc": started,
        "run_finished_utc": now_utc(),

        "evaluation_contract_sha256": metric_contract_hash,
        "metric_manifest_sha256": sha256_file(
            METRIC_MANIFEST
        ),
        "cell_model_summary_sha256": sha256_file(
            CELL_MODEL_SUMMARY
        ),
        "contrast_summary_sha256": sha256_file(
            CONTRAST_SUMMARY
        ),

        "contrast_support": support_map,

        "interpretation": {
            "H1_HEAD_RETARGET_R5": (
                "Tests whether target-head retargeting materially improves "
                "R5 performance over frozen source-head reuse."
            ),
            "H2_HEAD_CONSTRAINT_R5": (
                "Tests whether source-centered target-head regularization "
                "improves discrimination AND does not worsen IBS/risk scale."
            ),
            "H3_SOFT_LEAKAGE_R5": (
                "Tests whether prediction-only hardening of the same learned "
                "A3 gate materially reduces R5 negative transfer."
            ),
            "H4_ORACLE_GATE_R5": (
                "Tests whether perfect true binary transportability gating "
                "can materially rescue the A3 source+residual construction. "
                "M6 is nonimplementable oracle evidence."
            ),
        },

        "guardrails": {
            "no_model_fitting": True,
            "no_model_selection": True,
            "no_A6_reopening": True,
            "no_human_outcomes": True,
            "oracle_M6_nonimplementable": True,
            "same_DGP_new_seed_confirmation_only": True,
        },

        "next": (
            "Use H1-H4 results to decide manuscript mechanism claims and whether "
            "a descriptive TARGET protocol is scientifically justified. "
            "Do not create another rescue architecture from these same simulations."
        ),
    }

    write_json(
        SUMMARY_JSON,
        summary,
    )

    # ------------------------------------------------------------------
    # User-facing console.
    # ------------------------------------------------------------------
    print()
    print("=" * 120)
    print("05e3 R5 CONTROLLED MODEL TABLE")
    print("=" * 120)
    print(
        r5_table[
            [
                "target_events",
                "model",
                "mean_uno_c",
                "mean_delta_c_vs_B0",
                "mean_ibs",
                "mean_test_risk_sd",
                "negative_transfer_rate",
                "catastrophic_negative_transfer_rate",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 120)
    print("05e3 PRESPECIFIED MECHANISM CONTRASTS")
    print("=" * 120)
    print(
        contrast_df[
            [
                "contrast_id",
                "mean_delta_uno_c",
                "bootstrap_ci_low",
                "bootstrap_ci_high",
                "bootstrap_p_positive",
                "mean_delta_ibs",
                "mean_delta_test_risk_sd",
                "catastrophic_rate_reduction_control_minus_treatment",
                "frozen_support_status",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 120)
    print("05e3 FINAL MECHANISM SUMMARY")
    print("=" * 120)
    for contrast_id in [
        "H1_HEAD_RETARGET_R5",
        "H2_HEAD_CONSTRAINT_R5",
        "H3_SOFT_LEAKAGE_R5",
        "H4_ORACLE_GATE_R5",
    ]:
        print(
            f"{contrast_id}: "
            f"{support_map[contrast_id]}"
        )
    print()
    print("Model selection performed: NO")
    print("A6 reopened: NO")
    print("M6 oracle implementable: NO")
    print("Human outcomes read: NO")
    print("GPU execution: NO")
    print("=" * 120)
    print(
        "05e3: PASS_CONTROLLED_MECHANISM_CONTRAST_EVALUATION_COMPLETE"
    )
    print("=" * 120)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 120, file=sys.stderr)
        print(
            "05e3 controlled mechanism evaluation: FAIL",
            file=sys.stderr,
        )
        print(
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        print("=" * 120, file=sys.stderr)
        raise
