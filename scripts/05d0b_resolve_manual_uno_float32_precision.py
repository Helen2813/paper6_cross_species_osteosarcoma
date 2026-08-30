#!/usr/bin/env python3
"""
Paper 6 - resolve 05d0a manual Uno replay precision discrepancy.

Technical follow-up only.

05d0a recomputed Uno C in float64 and compared it to the frozen 05d checkpoint
using an absolute tolerance of 1e-8. The frozen 05d scenario metrics, however,
were intentionally serialized as float32:

    "uno_c": uno.astype(np.float32)

For C values near 0.5-0.7, one float32 ULP is approximately 6e-8, so correct
round-to-float32 serialization can differ from the float64 recomputation by
~3e-8 while representing exactly the same stored scientific metric.

This script does NOT change the frozen Uno metric or threshold. It verifies
whether every manual float64 replay, when rounded to the exact frozen storage
dtype, equals the original checkpoint float32 value exactly.

No model fitting.
No metric recomputation beyond the already-fixed manual examples.
No real data.
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

try:
    from sksurv.metrics import concordance_index_ipcw
    from sksurv.util import Surv
except ImportError as exc:
    raise ImportError("05d0b requires scikit-survival.") from exc


SCRIPT_VERSION = "05d0b-resolve-manual-uno-float32-storage-precision-v1-no-cli"

ROOT = Path(__file__).resolve().parents[1]

D0A_DIR = ROOT / "results" / "simulation_phase_diagram_diagnostics" / "05d0a"
D0A_MANUAL = D0A_DIR / "manual_uno_metric_replay.tsv"
D0A_SUMMARY = D0A_DIR / "summary.json"

C_DIR = ROOT / "results" / "simulation_model_matrix" / "05c"
C_MANIFEST = C_DIR / "scenario_output_manifest.tsv"

D_DIR = ROOT / "results" / "simulation_phase_diagram" / "05d"
D_METRIC_MANIFEST = D_DIR / "scenario_metric_manifest.tsv"
D_METRIC_CONTRACT = D_DIR / "metric_implementation_contract.json"

OUT_DIR = ROOT / "results" / "simulation_phase_diagram_diagnostics" / "05d0b"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRECISION_AUDIT = OUT_DIR / "manual_uno_float32_precision_audit.tsv"
SUMMARY_JSON = OUT_DIR / "summary.json"

EXPECTED_05D0A_STATUS = "HOLD_TECHNICAL_METRIC_OR_INVARIANT_DISCREPANCY"

ALL_MODELS = [
    "B0", "B4", "A0", "A1", "A2", "A3", "A4",
    "A3_NO_EVOLUTION_PRIOR",
    "A3_NO_TARGET_OVERRIDE",
    "A3_PERMUTED_PRIOR",
    "A3_MAPPING_PERMUTATION",
    "A3_SOURCE_OUTCOME_PERMUTATION",
]
MODEL_INDEX = {m: i for i, m in enumerate(ALL_MODELS)}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_file(path: Path) -> Path:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Required artifact missing: {path}")
    return path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")


def make_surv(t: np.ndarray, e: np.ndarray) -> np.ndarray:
    return Surv.from_arrays(
        event=np.asarray(e, dtype=bool),
        time=np.asarray(t, dtype=float),
    )


def frozen_tau(
    train_time: np.ndarray,
    train_event: np.ndarray,
    test_time: np.ndarray,
) -> float:
    event_times = np.asarray(train_time, dtype=float)[
        np.asarray(train_event, dtype=bool)
    ]
    preferred = float(np.quantile(event_times, 0.80))
    upper = min(
        preferred,
        float(np.max(train_time)),
        float(np.max(test_time)),
    )
    upper = float(np.nextafter(upper, -np.inf))
    lower = max(float(np.min(train_time)), float(np.min(test_time)))
    if not np.isfinite(upper) or upper <= lower:
        raise RuntimeError("Frozen Uno tau is not assessable in manual replay.")
    return upper


def direct_uno(
    train_time: np.ndarray,
    train_event: np.ndarray,
    test_time: np.ndarray,
    test_event: np.ndarray,
    risk: np.ndarray,
) -> float:
    tau = frozen_tau(train_time, train_event, test_time)
    value = float(
        concordance_index_ipcw(
            make_surv(train_time, train_event),
            make_surv(test_time, test_event),
            np.asarray(risk, dtype=float),
            tau=tau,
        )[0]
    )
    if not np.isfinite(value):
        raise RuntimeError("Non-finite manual Uno C.")
    return value


def main() -> None:
    print("=" * 120)
    print("Paper 6 - resolve 05d0a manual Uno float32 storage precision")
    print("=" * 120)
    print(f"Script version: {SCRIPT_VERSION}")
    print()
    print("Scope:")
    print("  Scientific metric changed: NO")
    print("  Tolerance relaxed: NO")
    print("  Exact float32 serialization equivalence tested: YES")
    print("  Model fitting: NO")
    print("  Human outcomes: NO")
    print("  GPU: NO")
    print()

    for p in [
        D0A_MANUAL,
        D0A_SUMMARY,
        C_MANIFEST,
        D_METRIC_MANIFEST,
        D_METRIC_CONTRACT,
    ]:
        require_file(p)

    d0a_summary = read_json(D0A_SUMMARY)
    if str(d0a_summary.get("scientific_status")) != EXPECTED_05D0A_STATUS:
        raise RuntimeError(
            "05d0a is not in the expected manual-replay technical HOLD state."
        )

    manual_prior = pd.read_csv(D0A_MANUAL, sep="\t")
    c_manifest = pd.read_csv(C_MANIFEST, sep="\t")
    d_manifest = pd.read_csv(D_METRIC_MANIFEST, sep="\t")

    c_by_id = {
        str(r.scenario_id): r
        for r in c_manifest.itertuples(index=False)
    }
    d_by_id = {
        str(r.scenario_id): r
        for r in d_manifest.itertuples(index=False)
    }

    rows: List[Dict[str, Any]] = []

    for prior in manual_prior.itertuples(index=False):
        sid = str(prior.scenario_id)
        rep = int(prior.replicate)
        model = str(prior.model)

        if model not in MODEL_INDEX:
            raise RuntimeError(f"Unknown model in manual replay: {model}")

        c_row = c_by_id[sid]
        d_row = d_by_id[sid]

        c_path = ROOT / str(c_row.output_path)
        d_path = ROOT / str(d_row.metric_output_path)

        require_file(c_path)
        require_file(d_path)

        if sha256_file(c_path) != str(c_row.output_sha256):
            raise RuntimeError(f"{sid}: 05c hash mismatch.")
        if sha256_file(d_path) != str(d_row.metric_output_sha256):
            raise RuntimeError(f"{sid}: 05d metric hash mismatch.")

        with np.load(c_path, allow_pickle=False) as cdata, np.load(
            d_path, allow_pickle=False
        ) as ddata:
            frozen_array = ddata["uno_c"]
            if frozen_array.dtype != np.float32:
                raise RuntimeError(
                    f"{sid}: frozen Uno dtype={frozen_array.dtype}; expected float32."
                )

            frozen32 = np.float32(
                frozen_array[rep, MODEL_INDEX[model]]
            )

            train_time = np.asarray(
                cdata["target_train_time"][rep], dtype=float
            )
            train_event = np.asarray(
                cdata["target_train_event"][rep], dtype=np.uint8
            )
            test_time = np.asarray(
                cdata["target_test_time"][rep], dtype=float
            )
            test_event = np.asarray(
                cdata["target_test_event"][rep], dtype=np.uint8
            )
            risk = np.asarray(
                cdata[f"risk_test_{model}"][rep], dtype=float
            )

            manual64 = direct_uno(
                train_time,
                train_event,
                test_time,
                test_event,
                risk,
            )
            rounded32 = np.float32(manual64)

            exact_after_storage_rounding = bool(
                rounded32.view(np.uint32) == frozen32.view(np.uint32)
            )

            ulp = float(np.spacing(frozen32))
            abs_diff = abs(manual64 - float(frozen32))
            half_ulp = abs(ulp) / 2.0

            rows.append(
                {
                    "scenario_id": sid,
                    "replicate": rep,
                    "model": model,
                    "manual_uno_float64": manual64,
                    "manual_rounded_float32": float(rounded32),
                    "frozen_checkpoint_float32": float(frozen32),
                    "absolute_float64_vs_frozen_difference": abs_diff,
                    "float32_ulp_at_frozen_value": abs(ulp),
                    "half_float32_ulp": half_ulp,
                    "within_half_ulp_plus_roundoff": bool(
                        abs_diff <= half_ulp + 1e-15
                    ),
                    "exact_float32_bitpattern_match": (
                        exact_after_storage_rounding
                    ),
                    "prior_05d0a_absolute_difference": float(
                        prior.absolute_difference
                    ),
                }
            )

    audit = pd.DataFrame(rows)
    audit.to_csv(PRECISION_AUDIT, sep="\t", index=False)

    exact_all = bool(audit["exact_float32_bitpattern_match"].all())
    half_ulp_all = bool(audit["within_half_ulp_plus_roundoff"].all())

    if exact_all:
        scientific_status = (
            "PASS_MANUAL_UNO_REPLAY_EXACT_AFTER_FROZEN_FLOAT32_SERIALIZATION"
        )
        status = "PASS"
    else:
        scientific_status = (
            "HOLD_MANUAL_UNO_DISCREPANCY_NOT_EXPLAINED_BY_FLOAT32_STORAGE"
        )
        status = "HOLD"

    summary = {
        "script_version": SCRIPT_VERSION,
        "status": status,
        "scientific_status": scientific_status,
        "created_utc": now_utc(),
        "n_manual_replays": len(audit),
        "all_exact_float32_bitpattern_matches": exact_all,
        "all_float64_differences_within_half_float32_ulp": half_ulp_all,
        "max_absolute_float64_vs_frozen_difference": float(
            audit["absolute_float64_vs_frozen_difference"].max()
        ),
        "interpretation": (
            "05d0a manual replay failures arose from comparing a float64 "
            "recomputation against a deliberately float32-serialized frozen "
            "checkpoint with an absolute 1e-8 tolerance."
            if exact_all
            else
            "At least one discrepancy exceeds what frozen float32 serialization explains."
        ),
        "scientific_05d_hold_changed": False,
        "A5_close_changed": False,
        "model_fitting": False,
        "human_outcomes_read": False,
        "final_artifact_hashes": {
            "manual_uno_float32_precision_audit.tsv": sha256_file(
                PRECISION_AUDIT
            ),
        },
    }
    write_json(SUMMARY_JSON, summary)

    print("-" * 120)
    print("Manual Uno serialization audit")
    print("-" * 120)
    print(
        audit[
            [
                "scenario_id",
                "replicate",
                "model",
                "manual_uno_float64",
                "frozen_checkpoint_float32",
                "absolute_float64_vs_frozen_difference",
                "half_float32_ulp",
                "exact_float32_bitpattern_match",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 120)
    print("05d0b MANUAL UNO PRECISION SUMMARY")
    print("=" * 120)
    print(f"Status: {scientific_status}")
    print(f"Exact float32 bit-pattern matches: {int(audit['exact_float32_bitpattern_match'].sum())}/{len(audit)}")
    print(
        f"Max |float64 replay - frozen float32|: "
        f"{summary['max_absolute_float64_vs_frozen_difference']:.3e}"
    )
    print("Frozen 05d scientific HOLD changed: NO")
    print("A5 CLOSE changed: NO")
    print("=" * 120)

    if not exact_all:
        raise RuntimeError(scientific_status)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 120, file=sys.stderr)
        print("05d0b manual Uno precision audit: FAIL", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print("=" * 120, file=sys.stderr)
        raise
