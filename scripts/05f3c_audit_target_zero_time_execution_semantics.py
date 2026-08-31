#!/usr/bin/env python3
"""
Paper 6 - post-opening TARGET zero-time execution-semantics audit.

Scientific role
---------------
TARGET primary OS is already open. 05f3b v2 proved that the current endpoint
EXACTLY reproduces the pre-existing Paper-4 complete-case population:
    n=86, events=29, censored=57,
including exactly one censored observation at time 0:
    TARGET-40-PAKUZU, event=0, time=0.

Before making ANY post-opening runner amendment or fitting a TARGET model,
05f3c determines whether the exact installed survival stack and frozen Paper-6
metric/loss functions support that pre-existing zero-time censored observation.

This is a bounded TECHNICAL execution audit.

It DOES:
- verify the exact 05f3b-v2 reconciliation artifact;
- verify the exact frozen 05f2c runner SHA;
- read only the already-materialized 86-case primary endpoint roster;
- inspect the installed scikit-survival version;
- run synthetic zero-time tests through:
    sksurv Surv.from_arrays,
    frozen 04c safe_uno_c,
    frozen 04c IBS grid/custom IBS,
    frozen 04c centered-ridge Cox,
    frozen 05c batched Cox loss;
- generate all frozen 5 x 20 TARGET event-stratified splits on the already-open
  86-case endpoint and test metric SUPPORT only using a deterministic,
  outcome-independent dummy risk based solely on sample index;
- inspect the frozen runner and freeze the exact minimal proposed technical
  patch:
      n=88 -> n=86
      X shape (88,11815) -> (86,11815)
      reject time <0 instead of time <=0
  with NO other runner change authorized.

It DOES NOT:
- read any new clinical column;
- fit any real TARGET model;
- use TARGET expression values in any model;
- evaluate any real source/target model risk;
- select/tune a model;
- assign T-A/T-B/T-C/T-D;
- change CV, metrics, bootstrap or branch rules;
- read GSE21257/GSE39055 outcomes.

No CLI arguments.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import torch
except ImportError as exc:
    raise ImportError("05f3c requires PyTorch.") from exc

try:
    import sksurv
    from sksurv.util import Surv
except ImportError as exc:
    raise ImportError("05f3c requires scikit-survival.") from exc

from sklearn.model_selection import StratifiedKFold


SCRIPT_VERSION = (
    "05f3c-audit-target-zero-time-execution-semantics-v1-no-cli"
)
AUDIT_VERSION = (
    "paper6-postopening-target-zero-time-execution-semantics-v1"
)

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Reconciled already-opened TARGET endpoint.
# ---------------------------------------------------------------------------
F3B_DIR = (
    ROOT
    / "results"
    / "human_posthold_descriptive"
    / "05f3b_v2"
)
F3B_RECONCILIATION = (
    F3B_DIR
    / "postopening_complete_case_reconciliation.json"
)
F3B_COMPLETE_ROSTER = (
    F3B_DIR
    / "TARGET_OS_complete_primary_endpoint_roster.tsv"
)
F3B_INCOMPLETE_ROSTER = (
    F3B_DIR
    / "TARGET_OS_incomplete_primary_endpoint_roster.tsv"
)
F3B_SUMMARY = F3B_DIR / "summary.json"

EXPECTED_F3B_RECONCILIATION_SHA256 = (
    "0baf87b4cb63fe09418f89231330842efafdeec40cb996d36a0470c7ddc4285f"
)
EXPECTED_F3B_STATUS = (
    "PASS_POSTOPENING_TARGET_OS_PREEXISTING_COMPLETE_CASE_POLICY_RECONCILED"
)

EXPECTED_N = 86
EXPECTED_EVENTS = 29
EXPECTED_CENSORED = 57
EXPECTED_ZERO_TIME_N = 1
EXPECTED_ZERO_TIME_SAMPLE = "TARGET-40-PAKUZU"

# ---------------------------------------------------------------------------
# Frozen Paper-6 implementation.
# ---------------------------------------------------------------------------
F2C_RUNNER = (
    ROOT
    / "results"
    / "human_posthold_descriptive"
    / "05f2c_v2"
    / "runner"
    / "frozen_TARGET_evaluation_runner_v2.py"
)
EXPECTED_F2C_RUNNER_SHA256 = (
    "8c7d5e03912463f51220057a35aaa14811c9f423c6fe0e24061aff29251f8075"
)

C4_SCRIPT = (
    ROOT
    / "scripts"
    / "04c_run_classical_premise_test_v3.py"
)
C05_SCRIPT = (
    ROOT
    / "scripts"
    / "05c_run_closed_synthetic_transfer_model_matrix.py"
)

F2A_CONTRACT = (
    ROOT
    / "results"
    / "human_posthold_descriptive"
    / "05f2a_v2"
    / "implementation_source_inventory_contract.json"
)
EXPECTED_F2A_CONTRACT_SHA256 = (
    "6cacaef003b2fe7c4830b144e274a9b751d93dbaf679785dfee8115191db03e9"
)

# Frozen CV from 05f1c.
OUTER_SPLITS = 5
OUTER_REPEATS = 20
BASE_SEED = 20260830

# ---------------------------------------------------------------------------
# Outputs.
# ---------------------------------------------------------------------------
OUT_DIR = (
    ROOT
    / "results"
    / "human_posthold_descriptive"
    / "05f3c"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYNTHETIC_AUDIT = (
    OUT_DIR
    / "zero_time_synthetic_survival_stack_audit.tsv"
)
REAL_SUPPORT_AUDIT = (
    OUT_DIR
    / "TARGET86_zero_time_metric_support_audit.tsv"
)
PATCH_PLAN = (
    OUT_DIR
    / "minimal_runner_patch_plan.json"
)
AUDIT_JSON = (
    OUT_DIR
    / "zero_time_execution_semantics_audit.json"
)
SUMMARY_JSON = OUT_DIR / "summary.json"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def require_file(path: Path) -> Path:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(
            f"Required artifact missing: {path}"
        )
    return path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def write_json(
    path: Path,
    payload: Any,
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        handle.write("\n")


def load_module(
    path: Path,
    name: str,
):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not import {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

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


def require_finite(
    value: float,
    label: str,
) -> None:
    if not np.isfinite(float(value)):
        raise RuntimeError(
            f"{label} is non-finite."
        )


def synthetic_zero_time_tests(
    m04c,
    c05,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    # --------------------------------------------------------------
    # Deterministic synthetic survival data.
    # Index 0 is censored at time zero.
    # --------------------------------------------------------------
    train_time = np.asarray(
        [
            0.0,
            10.0,
            20.0,
            30.0,
            40.0,
            50.0,
            60.0,
            70.0,
            80.0,
            90.0,
            100.0,
            110.0,
            120.0,
            130.0,
            140.0,
            150.0,
            160.0,
            170.0,
            180.0,
            190.0,
        ],
        dtype=float,
    )

    train_event = np.asarray(
        [
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
        ],
        dtype=bool,
    )

    test_time = np.asarray(
        [
            0.0,
            15.0,
            35.0,
            55.0,
            75.0,
            95.0,
            115.0,
            135.0,
            155.0,
            175.0,
        ],
        dtype=float,
    )

    test_event = np.asarray(
        [
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
        ],
        dtype=bool,
    )

    train_risk = np.linspace(
        -1.0,
        1.0,
        len(train_time),
    )

    test_risk = np.linspace(
        -0.9,
        0.9,
        len(test_time),
    )

    # 1. sksurv Surv constructor.
    try:
        y_train = Surv.from_arrays(
            event=train_event,
            time=train_time,
        )
        y_test = Surv.from_arrays(
            event=test_event,
            time=test_time,
        )
        rows.append(
            {
                "check": "sksurv_Surv_from_arrays_zero_censored",
                "status": "PASS",
                "detail": (
                    f"sksurv={sksurv.__version__}"
                ),
            }
        )
    except Exception as exc:
        rows.append(
            {
                "check": "sksurv_Surv_from_arrays_zero_censored",
                "status": "FAIL",
                "detail": (
                    f"{type(exc).__name__}: {exc}"
                ),
            }
        )
        return pd.DataFrame(rows)

    # 2. Frozen 04c Uno-C.
    try:
        uno = m04c.safe_uno_c(
            y_train,
            y_test,
            test_risk,
            quantile=0.90,
        )
        require_finite(
            uno,
            "synthetic zero-time Uno-C",
        )
        rows.append(
            {
                "check": "frozen_04c_safe_uno_c_zero_censored",
                "status": "PASS",
                "detail": "finite",
            }
        )
    except Exception as exc:
        rows.append(
            {
                "check": "frozen_04c_safe_uno_c_zero_censored",
                "status": "FAIL",
                "detail": (
                    f"{type(exc).__name__}: {exc}"
                ),
            }
        )

    # 3. Frozen IBS grid and custom IBS.
    try:
        grid = m04c.choose_ibs_grid(
            y_train,
            y_test,
        )

        # Outcome-independent smooth survival probabilities for API test.
        scale = np.exp(
            np.clip(
                test_risk,
                -2,
                2,
            )
        )

        survival = np.exp(
            -scale[:, None]
            * (
                grid[None, :]
                / 500.0
            )
        )

        ibs = m04c.integrated_brier_custom(
            y_train,
            y_test,
            survival,
            grid,
        )

        require_finite(
            ibs,
            "synthetic zero-time IBS",
        )

        rows.append(
            {
                "check": "frozen_04c_IBS_zero_censored",
                "status": "PASS",
                "detail": (
                    f"grid_n={len(grid)}"
                ),
            }
        )

    except Exception as exc:
        rows.append(
            {
                "check": "frozen_04c_IBS_zero_censored",
                "status": "FAIL",
                "detail": (
                    f"{type(exc).__name__}: {exc}"
                ),
            }
        )

    # 4. Frozen 04c custom centered ridge Cox on synthetic data.
    try:
        rng = np.random.default_rng(
            20260830
        )

        X = rng.normal(
            0.0,
            1.0,
            size=(
                len(train_time),
                3,
            ),
        )

        model = m04c.CenteredRidgeCox(
            alpha=1.0,
            center=np.zeros(
                3,
                dtype=float,
            ),
        ).fit(
            X,
            y_train,
        )

        risk = model.predict(X)

        if not np.isfinite(
            risk
        ).all():
            raise RuntimeError(
                "Synthetic centered-ridge risk non-finite."
            )

        rows.append(
            {
                "check": "frozen_04c_centered_ridge_zero_censored",
                "status": "PASS",
                "detail": (
                    f"gradient_norm={model.gradient_norm_:.3e}"
                ),
            }
        )

    except Exception as exc:
        rows.append(
            {
                "check": "frozen_04c_centered_ridge_zero_censored",
                "status": "FAIL",
                "detail": (
                    f"{type(exc).__name__}: {exc}"
                ),
            }
        )

    # 5. Frozen 05c Cox loss on synthetic time=0 censored row.
    try:
        risk_t = torch.as_tensor(
            train_risk[None, :],
            dtype=torch.float32,
        )
        time_t = torch.as_tensor(
            train_time[None, :],
            dtype=torch.float32,
        )
        event_t = torch.as_tensor(
            train_event
            .astype(np.float32)[None, :],
            dtype=torch.float32,
        )

        loss = c05.batched_cox_nll(
            risk_t,
            time_t,
            event_t,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Frozen 05c batched Cox loss non-finite."
            )

        per_rep = c05.per_replicate_cox_nll(
            risk_t,
            time_t,
            event_t,
        )

        if not np.isfinite(
            np.asarray(per_rep)
        ).all():
            raise RuntimeError(
                "Frozen 05c per-replicate Cox loss non-finite."
            )

        rows.append(
            {
                "check": "frozen_05c_cox_loss_zero_censored",
                "status": "PASS",
                "detail": "finite",
            }
        )

    except Exception as exc:
        rows.append(
            {
                "check": "frozen_05c_cox_loss_zero_censored",
                "status": "FAIL",
                "detail": (
                    f"{type(exc).__name__}: {exc}"
                ),
            }
        )

    return pd.DataFrame(rows)


def actual_endpoint_metric_support(
    m04c,
    complete: pd.DataFrame,
) -> pd.DataFrame:
    """
    Use real already-opened endpoint ONLY to test frozen metric support.

    Risk is deterministic sample-index rank and contains NO expression/model
    information. Numerical Uno-C values are intentionally not written; only
    finite/nonfinite status is recorded.
    """
    time_values = pd.to_numeric(
        complete["os_time_days_parsed"],
        errors="raise",
    ).to_numpy(
        dtype=float
    )

    event_values = pd.to_numeric(
        complete["os_event_parsed"],
        errors="raise",
    ).astype(int).to_numpy(
        dtype=int
    ).astype(bool)

    sample_index = pd.to_numeric(
        complete["sample_index_88"],
        errors="raise",
    ).to_numpy(
        dtype=int
    )

    # Pure identifier/order-based dummy score. Not a model prediction.
    dummy_risk = (
        sample_index.astype(float)
        - float(
            np.mean(
                sample_index
            )
        )
    )

    rows = []

    for repeat in range(
        OUTER_REPEATS
    ):
        splitter = StratifiedKFold(
            n_splits=OUTER_SPLITS,
            shuffle=True,
            random_state=(
                BASE_SEED
                + repeat
            ),
        )

        for fold, (
            train_idx,
            test_idx,
        ) in enumerate(
            splitter.split(
                np.zeros(
                    len(event_values)
                ),
                event_values.astype(int),
            )
        ):
            row: Dict[str, Any] = {
                "repeat": repeat,
                "outer_fold": fold,
                "n_train": int(
                    len(train_idx)
                ),
                "n_test": int(
                    len(test_idx)
                ),
                "events_train": int(
                    event_values[
                        train_idx
                    ].sum()
                ),
                "events_test": int(
                    event_values[
                        test_idx
                    ].sum()
                ),
                "zero_time_in_train": int(
                    np.sum(
                        time_values[
                            train_idx
                        ]
                        == 0.0
                    )
                ),
                "zero_time_in_test": int(
                    np.sum(
                        time_values[
                            test_idx
                        ]
                        == 0.0
                    )
                ),
            }

            try:
                y_train = Surv.from_arrays(
                    event=event_values[
                        train_idx
                    ],
                    time=time_values[
                        train_idx
                    ],
                )

                y_test = Surv.from_arrays(
                    event=event_values[
                        test_idx
                    ],
                    time=time_values[
                        test_idx
                    ],
                )

                row[
                    "surv_constructor_status"
                ] = "PASS"

            except Exception as exc:
                row[
                    "surv_constructor_status"
                ] = (
                    "FAIL:"
                    f"{type(exc).__name__}"
                )
                row[
                    "uno_support_status"
                ] = "NOT_RUN"
                row[
                    "ibs_grid_status"
                ] = "NOT_RUN"
                rows.append(row)
                continue

            try:
                uno = m04c.safe_uno_c(
                    y_train,
                    y_test,
                    dummy_risk[
                        test_idx
                    ],
                    quantile=0.90,
                )

                row[
                    "uno_support_status"
                ] = (
                    "PASS_FINITE"
                    if np.isfinite(uno)
                    else "NONFINITE"
                )

            except Exception as exc:
                row[
                    "uno_support_status"
                ] = (
                    "FAIL:"
                    f"{type(exc).__name__}"
                )

            try:
                grid = m04c.choose_ibs_grid(
                    y_train,
                    y_test,
                )

                row[
                    "ibs_grid_status"
                ] = "PASS"
                row[
                    "ibs_grid_n"
                ] = int(
                    len(grid)
                )

            except Exception as exc:
                # IBS grid non-assessability is already an allowed frozen
                # secondary-metric outcome. It is recorded, not patched.
                row[
                    "ibs_grid_status"
                ] = (
                    "NOT_ASSESSABLE:"
                    f"{type(exc).__name__}"
                )
                row[
                    "ibs_grid_n"
                ] = 0

            rows.append(row)

    return pd.DataFrame(rows)


def inspect_minimal_runner_patch(
    runner_path: Path,
) -> Dict[str, Any]:
    text = runner_path.read_text(
        encoding="utf-8"
    )

    old_n = (
        "n != 88"
    )
    old_shape = (
        "X_raw.shape != ("
        "\n            88,"
        "\n            11815,"
        "\n        )"
    )
    old_time = (
        "np.any("
        "\n            time_values <= 0"
        "\n        )"
    )

    counts = {
        "n_88_assertion_occurrences": text.count(
            old_n
        ),
        "shape_88_assertion_occurrences": text.count(
            old_shape
        ),
        "time_le_zero_assertion_occurrences": text.count(
            old_time
        ),
    }

    if counts != {
        "n_88_assertion_occurrences": 1,
        "shape_88_assertion_occurrences": 1,
        "time_le_zero_assertion_occurrences": 1,
    }:
        raise RuntimeError(
            "Frozen runner does not contain the exact expected minimal-patch "
            f"sites: {counts}"
        )

    proposed = text.replace(
        old_n,
        "n != 86",
        1,
    )

    proposed = proposed.replace(
        old_shape,
        (
            "X_raw.shape != ("
            "\n            86,"
            "\n            11815,"
            "\n        )"
        ),
        1,
    )

    proposed = proposed.replace(
        old_time,
        (
            "np.any("
            "\n            time_values < 0"
            "\n        )"
        ),
        1,
    )

    ast.parse(
        proposed,
        filename=(
            str(runner_path)
            + ".proposed"
        ),
    )

    original_sha = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

    proposed_sha = hashlib.sha256(
        proposed.encode("utf-8")
    ).hexdigest()

    return {
        "original_runner_sha256": original_sha,
        "proposed_runner_text_sha256": proposed_sha,
        "exact_patch_sites": counts,
        "authorized_substitutions": [
            {
                "old": "n != 88",
                "new": "n != 86",
                "reason": (
                    "pre-existing Paper4 TARGET OS complete-case n=86"
                ),
            },
            {
                "old": "X_raw.shape != (88, 11815)",
                "new": "X_raw.shape != (86, 11815)",
                "reason": (
                    "exact 86-case reconciled primary OS roster"
                ),
            },
            {
                "old": "time_values <= 0",
                "new": "time_values < 0",
                "reason": (
                    "retain pre-existing censored time=0 observation; "
                    "negative times remain forbidden"
                ),
            },
        ],
        "all_other_runner_bytes_must_remain_identical": True,
    }


def main() -> None:
    started = now_utc()

    print("=" * 120)
    print(
        "Paper 6 - post-opening TARGET zero-time execution-semantics audit"
    )
    print("=" * 120)
    print(
        f"Script version: {SCRIPT_VERSION}"
    )
    print()
    print("Scientific status:")
    print(
        "  TARGET outcomes already open: YES"
    )
    print(
        "  reconciled upstream OS population: n=86, events=29, censored=57"
    )
    print(
        "  zero-time observation: 1, censored"
    )
    print(
        "  real TARGET model fitting in 05f3c: NO"
    )
    print(
        "  model predictions read: NO"
    )
    print(
        "  branch assignment: NO"
    )
    print(
        "  new clinical columns read: NO"
    )
    print(
        "  GSE21257/GSE39055 outcomes read: NO"
    )
    print()

    for path in [
        F3B_RECONCILIATION,
        F3B_COMPLETE_ROSTER,
        F3B_INCOMPLETE_ROSTER,
        F3B_SUMMARY,
        F2C_RUNNER,
        C4_SCRIPT,
        C05_SCRIPT,
        F2A_CONTRACT,
    ]:
        require_file(path)

    if (
        sha256_file(
            F3B_RECONCILIATION
        )
        != EXPECTED_F3B_RECONCILIATION_SHA256
    ):
        raise RuntimeError(
            "05f3b-v2 reconciliation hash differs from supplied PASS run."
        )

    if (
        sha256_file(
            F2C_RUNNER
        )
        != EXPECTED_F2C_RUNNER_SHA256
    ):
        raise RuntimeError(
            "Frozen 05f2c runner hash changed."
        )

    if (
        sha256_file(
            F2A_CONTRACT
        )
        != EXPECTED_F2A_CONTRACT_SHA256
    ):
        raise RuntimeError(
            "05f2a implementation inventory hash changed."
        )

    f3b_summary = read_json(
        F3B_SUMMARY
    )

    if (
        clean(
            f3b_summary.get(
                "scientific_status"
            )
        )
        != EXPECTED_F3B_STATUS
    ):
        raise RuntimeError(
            "05f3b-v2 is not in expected PASS state."
        )

    if (
        f3b_summary.get(
            "TARGET_models_fit"
        )
        is not False
    ):
        raise RuntimeError(
            "05f3b-v2 provenance unexpectedly indicates TARGET model fitting."
        )

    complete = pd.read_csv(
        F3B_COMPLETE_ROSTER,
        sep="\t",
        low_memory=False,
    )

    incomplete = pd.read_csv(
        F3B_INCOMPLETE_ROSTER,
        sep="\t",
        low_memory=False,
    )

    if len(complete) != EXPECTED_N:
        raise RuntimeError(
            f"Complete roster n={len(complete)}, expected 86."
        )

    if len(incomplete) != 2:
        raise RuntimeError(
            f"Incomplete roster n={len(incomplete)}, expected 2."
        )

    event = pd.to_numeric(
        complete[
            "os_event_parsed"
        ],
        errors="raise",
    ).astype(int)

    time_values = pd.to_numeric(
        complete[
            "os_time_days_parsed"
        ],
        errors="raise",
    ).astype(float)

    if int(event.sum()) != EXPECTED_EVENTS:
        raise RuntimeError(
            "Complete-roster events changed."
        )

    if int((event == 0).sum()) != EXPECTED_CENSORED:
        raise RuntimeError(
            "Complete-roster censored count changed."
        )

    zero = complete[
        time_values == 0.0
    ].copy()

    if len(zero) != EXPECTED_ZERO_TIME_N:
        raise RuntimeError(
            f"Zero-time rows={len(zero)}, expected 1."
        )

    zero_sample = str(
        zero.iloc[0][
            "sample_id"
        ]
    )

    zero_event = int(
        zero.iloc[0][
            "os_event_parsed"
        ]
    )

    if (
        zero_sample
        != EXPECTED_ZERO_TIME_SAMPLE
    ):
        raise RuntimeError(
            "Zero-time sample identity changed."
        )

    if zero_event != 0:
        raise RuntimeError(
            "Zero-time observation is not censored."
        )

    m04c = load_module(
        C4_SCRIPT,
        "paper6_04c_05f3c_zero_time_audit",
    )

    c05 = load_module(
        C05_SCRIPT,
        "paper6_05c_05f3c_zero_time_audit",
    )

    print("Installed survival stack:")
    print(
        f"  scikit-survival: {sksurv.__version__}"
    )
    print(
        f"  Python: {sys.version.split()[0]}"
    )
    print()

    synthetic = synthetic_zero_time_tests(
        m04c,
        c05,
    )

    synthetic.to_csv(
        SYNTHETIC_AUDIT,
        sep="\t",
        index=False,
    )

    print(
        "Synthetic zero-time survival-stack audit:"
    )

    for row in synthetic.itertuples(
        index=False
    ):
        print(
            f"  {row.check}: {row.status}"
        )

    synthetic_failed = synthetic[
        synthetic["status"]
        != "PASS"
    ]

    if not synthetic_failed.empty:
        raise RuntimeError(
            "At least one exact synthetic zero-time survival-stack check failed: "
            + "; ".join(
                synthetic_failed[
                    "check"
                ].astype(str)
            )
        )

    print()

    support = actual_endpoint_metric_support(
        m04c,
        complete,
    )

    support.to_csv(
        REAL_SUPPORT_AUDIT,
        sep="\t",
        index=False,
    )

    expected_rows = (
        OUTER_REPEATS
        * OUTER_SPLITS
    )

    if len(support) != expected_rows:
        raise RuntimeError(
            f"Metric-support rows={len(support)}, expected 100."
        )

    surv_pass = int(
        (
            support[
                "surv_constructor_status"
            ]
            == "PASS"
        ).sum()
    )

    uno_pass = int(
        (
            support[
                "uno_support_status"
            ]
            == "PASS_FINITE"
        ).sum()
    )

    ibs_pass = int(
        (
            support[
                "ibs_grid_status"
            ]
            == "PASS"
        ).sum()
    )

    folds_zero_train = int(
        (
            support[
                "zero_time_in_train"
            ]
            > 0
        ).sum()
    )

    folds_zero_test = int(
        (
            support[
                "zero_time_in_test"
            ]
            > 0
        ).sum()
    )

    print(
        "Actual 86-case endpoint metric-support audit "
        "[dummy index-only risk; NO MODEL]:"
    )
    print(
        f"  frozen split configurations: {len(support)}/100"
    )
    print(
        f"  Surv constructor PASS: {surv_pass}/100"
    )
    print(
        f"  finite Uno-C support: {uno_pass}/100"
    )
    print(
        f"  IBS grid support: {ibs_pass}/100"
    )
    print(
        f"  folds with zero-time patient in train: {folds_zero_train}"
    )
    print(
        f"  folds with zero-time patient in test: {folds_zero_test}"
    )
    print()

    if surv_pass != expected_rows:
        raise RuntimeError(
            "Installed survival stack does not support the reconciled zero-time "
            "endpoint in every frozen fold."
        )

    if uno_pass != expected_rows:
        raise RuntimeError(
            "Frozen Uno-C implementation is not finite in every frozen fold "
            "under deterministic no-model metric-support audit."
        )

    # IBS grid is secondary and frozen as potentially NOT_ASSESSABLE.
    # We do NOT require 100/100; we only record the pre-model support count.

    patch_plan = inspect_minimal_runner_patch(
        F2C_RUNNER
    )

    write_json(
        PATCH_PLAN,
        patch_plan,
    )

    print(
        "Minimal runner patch-site audit: PASS"
    )
    print(
        "  authorized changes:"
    )
    print(
        "    n assertion: 88 -> 86"
    )
    print(
        "    X rows assertion: 88 -> 86"
    )
    print(
        "    time validity: reject <0, retain censored time=0"
    )
    print(
        "  all other runner bytes: MUST remain unchanged"
    )
    print()

    audit = {
        "script_version": SCRIPT_VERSION,
        "audit_version": AUDIT_VERSION,
        "status": "PASS",
        "scientific_status": (
            "PASS_TARGET_ZERO_TIME_EXECUTION_SEMANTICS_SUPPORTED"
        ),
        "created_utc": now_utc(),

        "TARGET_outcomes_already_open": True,
        "TARGET_model_fitting": False,
        "TARGET_model_predictions_read": False,

        "reconciled_endpoint": {
            "n": EXPECTED_N,
            "events": EXPECTED_EVENTS,
            "censored": EXPECTED_CENSORED,
            "zero_time_n": EXPECTED_ZERO_TIME_N,
            "zero_time_sample": EXPECTED_ZERO_TIME_SAMPLE,
            "zero_time_event": 0,
        },

        "survival_stack": {
            "scikit_survival_version": str(
                sksurv.__version__
            ),
            "synthetic_all_checks_pass": True,
            "actual_100_fold_Surv_support_pass": (
                surv_pass
                == expected_rows
            ),
            "actual_100_fold_Uno_support_pass": (
                uno_pass
                == expected_rows
            ),
            "actual_100_fold_IBS_grid_support_n": ibs_pass,
            "IBS_grid_nonassessability_policy_changed": False,
        },

        "postopening_technical_patch_authorization": {
            "authorized": True,
            "only_changes": [
                "runner expected TARGET n: 88 -> 86",
                "runner expected TARGET X rows: 88 -> 86",
                "runner invalid-time guard: <=0 -> <0",
            ],
            "zero_time_imputation": False,
            "zero_time_epsilon_shift": False,
            "zero_time_sample_exclusion": False,
            "models_changed": False,
            "hyperparameters_changed": False,
            "metrics_changed": False,
            "CV_changed": False,
            "bootstrap_changed": False,
            "branch_rules_changed": False,
            "all_other_runner_bytes_must_remain_identical": True,
        },

        "artifacts": {
            SYNTHETIC_AUDIT.name: sha256_file(
                SYNTHETIC_AUDIT
            ),
            REAL_SUPPORT_AUDIT.name: sha256_file(
                REAL_SUPPORT_AUDIT
            ),
            PATCH_PLAN.name: sha256_file(
                PATCH_PLAN
            ),
        },

        "safety": {
            "new_clinical_columns_read": False,
            "GSE21257_outcomes_read": False,
            "GSE39055_outcomes_read": False,
            "real_TARGET_model_fit": False,
            "real_model_predictions_read": False,
            "dummy_risk_uses_expression": False,
            "dummy_risk_uses_outcome": False,
            "branch_assignment_performed": False,
        },

        "next": (
            "05f3d create runner v3 by applying ONLY the three SHA-audited "
            "technical substitutions, subset the immutable 88-row TARGET matrix "
            "to the exact 86-case 05f3b-v2 complete-primary-OS roster by sample ID, "
            "then run the otherwise unchanged frozen T0-T2/N0-N5 evaluation."
        ),
    }

    write_json(
        AUDIT_JSON,
        audit,
    )

    summary = {
        "script_version": SCRIPT_VERSION,
        "status": "PASS",
        "scientific_status": audit[
            "scientific_status"
        ],
        "run_started_utc": started,
        "run_finished_utc": now_utc(),
        "TARGET_n": EXPECTED_N,
        "TARGET_events": EXPECTED_EVENTS,
        "TARGET_censored": EXPECTED_CENSORED,
        "zero_time_n": EXPECTED_ZERO_TIME_N,
        "zero_time_censored": True,
        "synthetic_survival_stack_checks": "PASS",
        "actual_frozen_fold_Surv_support": f"{surv_pass}/100",
        "actual_frozen_fold_Uno_support": f"{uno_pass}/100",
        "actual_frozen_fold_IBS_grid_support": f"{ibs_pass}/100",
        "real_TARGET_models_fit": False,
        "minimal_runner_patch_authorized": True,
        "audit_sha256": sha256_file(
            AUDIT_JSON
        ),
        "next": audit["next"],
    }

    write_json(
        SUMMARY_JSON,
        summary,
    )

    print("=" * 120)
    print(
        "05f3c ZERO-TIME EXECUTION-SEMANTICS SUMMARY"
    )
    print("=" * 120)
    print(
        "Reconciled endpoint: n=86, events=29, censored=57"
    )
    print(
        "Zero-time case: TARGET-40-PAKUZU, censored"
    )
    print()
    print(
        "Exact installed/frozen survival stack supports censored time=0: PASS"
    )
    print(
        f"All 100 frozen folds Surv support: {surv_pass}/100"
    )
    print(
        f"All 100 frozen folds finite Uno support: {uno_pass}/100"
    )
    print(
        f"IBS grid support before model fitting: {ibs_pass}/100 "
        "[secondary; frozen NOT_ASSESSABLE policy unchanged]"
    )
    print()
    print(
        "Real TARGET models fit in 05f3c: NO"
    )
    print(
        "Model/metric/CV/bootstrap/branch changes: NO"
    )
    print(
        "Zero-time imputation/epsilon/exclusion: NO"
    )
    print()
    print(
        "Minimal runner amendment authorized: YES"
    )
    print(
        "  88 -> 86 row assertions"
    )
    print(
        "  <=0 -> <0 time guard"
    )
    print(
        "  no other runner change"
    )
    print()
    print(
        f"Audit SHA256: {sha256_file(AUDIT_JSON)}"
    )
    print()
    print(
        "Next: 05f3d apply only audited runner patch and execute exact 86-case TARGET evaluation."
    )
    print("=" * 120)
    print(
        "05f3c: PASS_TARGET_ZERO_TIME_EXECUTION_SEMANTICS_SUPPORTED"
    )
    print("=" * 120)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 120, file=sys.stderr)
        print(
            "05f3c zero-time execution-semantics audit: FAIL",
            file=sys.stderr,
        )
        print(
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        print("=" * 120, file=sys.stderr)
        raise
