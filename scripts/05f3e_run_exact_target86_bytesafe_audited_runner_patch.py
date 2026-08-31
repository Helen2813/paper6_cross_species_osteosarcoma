#!/usr/bin/env python3
"""
Paper 6 - execute exact 86-case reconciled TARGET-OS evaluation.

This is the first REAL TARGET model execution after outcome opening.

Scientific status
-----------------
TARGET outcomes are already open.

Post-opening technical amendments are permitted ONLY because:
- 05f3b v2 proved the exact current primary endpoint reproduces the
  PRE-EXISTING Paper-4 complete-case TARGET OS population:
      n=86, events=29, censored=57;
- 05f3c proved the installed/frozen survival stack supports the single
  censored zero-time case in all 100 frozen outer-fold configurations;
- 05f3c SHA-audited exactly three runner substitutions and authorized no others.

AUTHORIZED runner substitutions
-------------------------------
1. expected n: 88 -> 86
2. expected X shape: (88, 11815) -> (86, 11815)
3. invalid-time guard: time <= 0 -> time < 0

NO OTHER runner change is permitted.

This script:
- verifies every relevant frozen/reconciled artifact SHA;
- creates runner v3 from authoritative 05f2c runner v2 using ONLY the three
  authorized exact substitutions;
- proves that reconstructing those substitutions yields byte-identical v3;
- subsets the immutable 88-row TARGET matrix to the exact 86-case complete
  primary-OS roster by frozen sample_index_88 / sample_id identity;
- verifies n=86/events=29/censored=57/one censored time-zero case;
- executes the otherwise unchanged frozen T0-T2/N0-N5 evaluation;
- writes all outputs under a NEW 05f3d namespace.

It DOES NOT:
- read any new clinical column;
- impute or shift the zero-time value;
- exclude any additional sample;
- change model hyperparameters;
- change CV/bootstrap/metric/branch logic;
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
    raise ImportError("05f3d requires PyTorch.") from exc


SCRIPT_VERSION = (
    "05f3e-run-exact-target86-bytesafe-audited-runner-patch-v1-no-cli"
)
EXECUTION_VERSION = (
    "paper6-target86-postopening-bytesafe-audited-execution-v1"
)

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Authoritative pre-opening runner v2.
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

# ---------------------------------------------------------------------------
# Post-opening endpoint reconciliation.
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

# ---------------------------------------------------------------------------
# Zero-time execution-semantics audit.
# ---------------------------------------------------------------------------
F3C_DIR = (
    ROOT
    / "results"
    / "human_posthold_descriptive"
    / "05f3c"
)
F3C_AUDIT = (
    F3C_DIR
    / "zero_time_execution_semantics_audit.json"
)
F3C_PATCH_PLAN = (
    F3C_DIR
    / "minimal_runner_patch_plan.json"
)
F3C_SUMMARY = F3C_DIR / "summary.json"

EXPECTED_F3C_AUDIT_SHA256 = (
    "f5241599f88c63e39ef0e44bc350c92e886e2e28ab9816575cf96e20d64d1143"
)
EXPECTED_F3C_STATUS = (
    "PASS_TARGET_ZERO_TIME_EXECUTION_SEMANTICS_SUPPORTED"
)

# ---------------------------------------------------------------------------
# Immutable 88-row TARGET expression matrix.
# ---------------------------------------------------------------------------
F1B_DIR = (
    ROOT
    / "results"
    / "human_posthold_descriptive"
    / "05f1b"
)
TARGET_MATRIX_88 = (
    F1B_DIR
    / "matrices"
    / "TARGET_OS_raw_aligned_11815genes.npz"
)
EXPECTED_TARGET_MATRIX_88_SHA256 = (
    "7aca20d98731a24d90882f2042b1aa7c7aefc0095a250d4ed42c3502cc3c7b19"
)

EXPECTED_N_88 = 88
EXPECTED_N_86 = 86
EXPECTED_EVENTS = 29
EXPECTED_CENSORED = 57
EXPECTED_FEATURES = 11815
EXPECTED_ZERO_TIME_N = 1
EXPECTED_ZERO_TIME_SAMPLE = "TARGET-40-PAKUZU"

# Execution device is not a scientific degree of freedom.
# CPU is retained because pre-opening smoke/05f3c stack validation used CPU.
EXECUTION_DEVICE = "cpu"

# ---------------------------------------------------------------------------
# Outputs.
# ---------------------------------------------------------------------------
OUT_DIR = (
    ROOT
    / "results"
    / "human_posthold_descriptive"
    / "05f3e"
)
RUNNER_DIR = OUT_DIR / "runner"
INPUT_DIR = OUT_DIR / "execution_input"
EXEC_DIR = OUT_DIR / "frozen_execution"

for directory in [
    OUT_DIR,
    RUNNER_DIR,
    INPUT_DIR,
    EXEC_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

RUNNER_V3 = (
    RUNNER_DIR
    / "frozen_TARGET86_evaluation_runner_v3_bytesafe.py"
)

TARGET86_MATRIX = (
    INPUT_DIR
    / "TARGET_OS_complete86_raw_aligned_11815genes.npz"
)

TARGET86_ENDPOINT = (
    INPUT_DIR
    / "TARGET_OS_complete86_primary_endpoint.tsv"
)

PATCH_AUDIT = (
    OUT_DIR
    / "runner_v2_to_v3_exact_patch_audit.json"
)

INPUT_AUDIT = (
    OUT_DIR
    / "TARGET86_execution_input_audit.json"
)

SUMMARY_JSON = OUT_DIR / "summary.json"


def now_utc() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def require_file(
    path: Path,
) -> Path:
    if (
        not path.exists()
        or not path.is_file()
    ):
        raise FileNotFoundError(
            f"Required artifact missing: {path}"
        )
    return path


def sha256_file(
    path: Path,
) -> str:
    h = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        for block in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(
                block
            )

    return h.hexdigest()


def sha256_bytes(
    payload: bytes,
) -> str:
    return hashlib.sha256(
        payload
    ).hexdigest()


def read_json(
    path: Path,
) -> Any:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(
            handle
        )


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
        handle.write(
            "\n"
        )


def atomic_savez(
    path: Path,
    **arrays: np.ndarray,
) -> None:
    tmp = path.with_suffix(
        path.suffix + ".part"
    )

    with tmp.open(
        "wb"
    ) as handle:
        np.savez_compressed(
            handle,
            **arrays,
        )

    tmp.replace(
        path
    )


def load_module(
    path: Path,
    name: str,
):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            f"Could not import {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    prior = sys.modules.get(
        name
    )
    sys.modules[name] = module

    try:
        spec.loader.exec_module(
            module
        )
    except Exception:
        if prior is None:
            sys.modules.pop(
                name,
                None,
            )
        else:
            sys.modules[
                name
            ] = prior

        raise

    return module


def verify_authorization() -> Dict[str, Any]:
    for path in [
        F2C_RUNNER,
        F3B_RECONCILIATION,
        F3B_COMPLETE_ROSTER,
        F3B_INCOMPLETE_ROSTER,
        F3B_SUMMARY,
        F3C_AUDIT,
        F3C_PATCH_PLAN,
        F3C_SUMMARY,
        TARGET_MATRIX_88,
    ]:
        require_file(
            path
        )

    if (
        sha256_file(
            F2C_RUNNER
        )
        != EXPECTED_F2C_RUNNER_SHA256
    ):
        raise RuntimeError(
            "Authoritative runner v2 hash changed."
        )

    if (
        sha256_file(
            F3B_RECONCILIATION
        )
        != EXPECTED_F3B_RECONCILIATION_SHA256
    ):
        raise RuntimeError(
            "05f3b-v2 reconciliation hash changed."
        )

    if (
        sha256_file(
            F3C_AUDIT
        )
        != EXPECTED_F3C_AUDIT_SHA256
    ):
        raise RuntimeError(
            "05f3c zero-time audit hash changed."
        )

    if (
        sha256_file(
            TARGET_MATRIX_88
        )
        != EXPECTED_TARGET_MATRIX_88_SHA256
    ):
        raise RuntimeError(
            "Immutable TARGET 88-row matrix hash changed."
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
            "05f3b-v2 unexpectedly indicates real TARGET model fitting."
        )

    f3c_summary = read_json(
        F3C_SUMMARY
    )

    if (
        clean(
            f3c_summary.get(
                "scientific_status"
            )
        )
        != EXPECTED_F3C_STATUS
    ):
        raise RuntimeError(
            "05f3c is not in expected PASS state."
        )

    if (
        f3c_summary.get(
            "real_TARGET_models_fit"
        )
        is not False
    ):
        raise RuntimeError(
            "05f3c unexpectedly indicates real TARGET model fitting."
        )

    if (
        f3c_summary.get(
            "minimal_runner_patch_authorized"
        )
        is not True
    ):
        raise RuntimeError(
            "05f3c did not authorize the minimal runner amendment."
        )

    f3b_reconciliation = read_json(
        F3B_RECONCILIATION
    )

    f3b_artifacts = (
        f3b_reconciliation.get(
            "artifact_hashes"
        )
        or {}
    )

    expected_complete_roster_hash = clean(
        f3b_artifacts.get(
            F3B_COMPLETE_ROSTER.name
        )
    )
    expected_incomplete_roster_hash = clean(
        f3b_artifacts.get(
            F3B_INCOMPLETE_ROSTER.name
        )
    )

    if (
        len(expected_complete_roster_hash) != 64
        or sha256_file(F3B_COMPLETE_ROSTER)
        != expected_complete_roster_hash
    ):
        raise RuntimeError(
            "05f3b-v2 complete roster differs from the SHA recorded "
            "in the authoritative reconciliation artifact."
        )

    if (
        len(expected_incomplete_roster_hash) != 64
        or sha256_file(F3B_INCOMPLETE_ROSTER)
        != expected_incomplete_roster_hash
    ):
        raise RuntimeError(
            "05f3b-v2 incomplete roster differs from the SHA recorded "
            "in the authoritative reconciliation artifact."
        )

    f3c_audit = read_json(
        F3C_AUDIT
    )

    f3c_artifacts = (
        f3c_audit.get(
            "artifacts"
        )
        or {}
    )

    expected_patch_plan_hash = clean(
        f3c_artifacts.get(
            F3C_PATCH_PLAN.name
        )
    )

    if (
        len(expected_patch_plan_hash) != 64
        or sha256_file(F3C_PATCH_PLAN)
        != expected_patch_plan_hash
    ):
        raise RuntimeError(
            "05f3c patch plan differs from the SHA recorded in the "
            "authoritative zero-time audit."
        )

    patch_plan = read_json(
        F3C_PATCH_PLAN
    )

    authorization = (
        f3c_audit.get(
            "postopening_technical_patch_authorization"
        )
        or {}
    )

    if (
        authorization.get(
            "authorized"
        )
        is not True
    ):
        raise RuntimeError(
            "05f3c audit does not authorize runner patch."
        )

    if (
        authorization.get(
            "all_other_runner_bytes_must_remain_identical"
        )
        is not True
    ):
        raise RuntimeError(
            "05f3c did not freeze all non-patched runner bytes."
        )

    expected_patch_old_new = [
        (
            "n != 88",
            "n != 86",
        ),
        (
            "X_raw.shape != (88, 11815)",
            "X_raw.shape != (86, 11815)",
        ),
        (
            "time_values <= 0",
            "time_values < 0",
        ),
    ]

    plan_substitutions = (
        patch_plan.get(
            "authorized_substitutions"
        )
        or []
    )

    observed_patch_old_new = [
        (
            clean(item.get("old")),
            clean(item.get("new")),
        )
        for item in plan_substitutions
    ]

    if (
        observed_patch_old_new
        != expected_patch_old_new
    ):
        raise RuntimeError(
            "05f3c patch plan substitutions differ from the exact "
            f"authorized three changes: {observed_patch_old_new}"
        )

    if (
        patch_plan.get(
            "all_other_runner_bytes_must_remain_identical"
        )
        is not True
    ):
        raise RuntimeError(
            "05f3c patch plan does not require all other runner bytes "
            "to remain identical."
        )

    return {
        "f3b_summary": f3b_summary,
        "f3c_summary": f3c_summary,
        "f3c_audit": f3c_audit,
        "f3b_complete_roster_sha256": sha256_file(
            F3B_COMPLETE_ROSTER
        ),
        "f3b_incomplete_roster_sha256": sha256_file(
            F3B_INCOMPLETE_ROSTER
        ),
        "f3c_patch_plan_sha256": sha256_file(
            F3C_PATCH_PLAN
        ),
    }


def create_exact_runner_v3() -> Dict[str, Any]:
    """
    Apply the three 05f3c-authorized substitutions directly to RAW BYTES.

    This deliberately avoids Python text-mode newline translation on Windows.
    The source runner's newline convention is preserved exactly, and reverse
    byte patching MUST reproduce the original runner v2 bytes byte-for-byte.
    """
    original_bytes = F2C_RUNNER.read_bytes()

    if (
        sha256_bytes(
            original_bytes
        )
        != EXPECTED_F2C_RUNNER_SHA256
    ):
        raise RuntimeError(
            "Authoritative runner v2 raw-byte SHA changed."
        )

    crlf_count = original_bytes.count(
        b"\r\n"
    )

    lf_count_total = original_bytes.count(
        b"\n"
    )

    bare_lf_count = (
        lf_count_total
        - crlf_count
    )

    if (
        crlf_count > 0
        and bare_lf_count > 0
    ):
        raise RuntimeError(
            "Authoritative runner v2 contains mixed CRLF/LF newlines; "
            "raw-byte patch is intentionally fail-closed."
        )

    if crlf_count > 0:
        nl = b"\r\n"
        newline_style = "CRLF"
    else:
        nl = b"\n"
        newline_style = "LF"

    old_n = b"n != 88"
    new_n = b"n != 86"

    old_shape = (
        b"X_raw.shape != ("
        + nl
        + b"            88,"
        + nl
        + b"            11815,"
        + nl
        + b"        )"
    )

    new_shape = (
        b"X_raw.shape != ("
        + nl
        + b"            86,"
        + nl
        + b"            11815,"
        + nl
        + b"        )"
    )

    old_time = (
        b"np.any("
        + nl
        + b"            time_values <= 0"
        + nl
        + b"        )"
    )

    new_time = (
        b"np.any("
        + nl
        + b"            time_values < 0"
        + nl
        + b"        )"
    )

    expected_counts = {
        "n_88_assertion_occurrences": 1,
        "shape_88_assertion_occurrences": 1,
        "time_le_zero_assertion_occurrences": 1,
    }

    observed_counts = {
        "n_88_assertion_occurrences": original_bytes.count(
            old_n
        ),
        "shape_88_assertion_occurrences": original_bytes.count(
            old_shape
        ),
        "time_le_zero_assertion_occurrences": original_bytes.count(
            old_time
        ),
    }

    if (
        observed_counts
        != expected_counts
    ):
        raise RuntimeError(
            "Runner v2 raw-byte patch-site counts differ from the "
            f"frozen 05f3c audit: {observed_counts}"
        )

    patched_bytes = original_bytes.replace(
        old_n,
        new_n,
        1,
    )

    patched_bytes = patched_bytes.replace(
        old_shape,
        new_shape,
        1,
    )

    patched_bytes = patched_bytes.replace(
        old_time,
        new_time,
        1,
    )

    # Source remains valid UTF-8/Python after the raw-byte amendment.
    try:
        patched_text = patched_bytes.decode(
            "utf-8"
        )
    except UnicodeDecodeError as exc:
        raise RuntimeError(
            "Patched runner v3 is not valid UTF-8."
        ) from exc

    ast.parse(
        patched_text,
        filename=str(
            RUNNER_V3
        ),
    )

    # Write bytes, never text, so Windows cannot translate newline bytes.
    RUNNER_V3.write_bytes(
        patched_bytes
    )

    disk_bytes = RUNNER_V3.read_bytes()

    if (
        disk_bytes
        != patched_bytes
    ):
        raise RuntimeError(
            "Runner v3 raw-byte disk replay differs from constructed bytes."
        )

    # Require expected new patterns exactly once.
    new_counts = {
        "n_86_assertion_occurrences": disk_bytes.count(
            new_n
        ),
        "shape_86_assertion_occurrences": disk_bytes.count(
            new_shape
        ),
        "time_lt_zero_assertion_occurrences": disk_bytes.count(
            new_time
        ),
    }

    if (
        new_counts
        != {
            "n_86_assertion_occurrences": 1,
            "shape_86_assertion_occurrences": 1,
            "time_lt_zero_assertion_occurrences": 1,
        }
    ):
        raise RuntimeError(
            "Runner v3 does not contain the exact three patched raw-byte "
            f"sites once each: {new_counts}"
        )

    # Reverse RAW-BYTE patch and demand byte-for-byte identity with runner v2.
    reverse_bytes = disk_bytes.replace(
        new_n,
        old_n,
        1,
    )

    reverse_bytes = reverse_bytes.replace(
        new_shape,
        old_shape,
        1,
    )

    reverse_bytes = reverse_bytes.replace(
        new_time,
        old_time,
        1,
    )

    if (
        reverse_bytes
        != original_bytes
    ):
        raise RuntimeError(
            "Reverse raw-byte patch does not reproduce runner v2 "
            "byte-for-byte; unauthorized runner change detected."
        )

    if (
        sha256_bytes(
            reverse_bytes
        )
        != EXPECTED_F2C_RUNNER_SHA256
    ):
        raise RuntimeError(
            "Reverse-patched runner SHA does not reproduce the authoritative "
            "runner v2 SHA."
        )

    patch_audit = {
        "status": "PASS",
        "created_utc": now_utc(),
        "patch_mode": "RAW_BYTES_PRESERVE_SOURCE_NEWLINES",
        "source_newline_style": newline_style,
        "source_runner_path": str(
            F2C_RUNNER.relative_to(
                ROOT
            )
        ),
        "source_runner_sha256": sha256_file(
            F2C_RUNNER
        ),
        "runner_v3_path": str(
            RUNNER_V3.relative_to(
                ROOT
            )
        ),
        "runner_v3_sha256": sha256_file(
            RUNNER_V3
        ),
        "authorized_substitutions_applied": [
            {
                "old": "n != 88",
                "new": "n != 86",
            },
            {
                "old": "X_raw.shape != (88, 11815)",
                "new": "X_raw.shape != (86, 11815)",
            },
            {
                "old": "time_values <= 0",
                "new": "time_values < 0",
            },
        ],
        "source_patch_site_counts": observed_counts,
        "patched_site_counts": new_counts,
        "disk_bytes_equal_constructed_bytes": True,
        "reverse_raw_byte_patch_reproduces_runner_v2": True,
        "reverse_raw_byte_sha256": sha256_bytes(
            reverse_bytes
        ),
        "unauthorized_runner_changes": False,
    }

    write_json(
        PATCH_AUDIT,
        patch_audit,
    )

    return patch_audit


def materialize_target86() -> Dict[str, Any]:
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

    if (
        len(complete)
        != EXPECTED_N_86
    ):
        raise RuntimeError(
            f"Complete TARGET roster n={len(complete)}, expected 86."
        )

    if (
        len(incomplete)
        != (
            EXPECTED_N_88
            - EXPECTED_N_86
        )
    ):
        raise RuntimeError(
            f"Incomplete TARGET roster n={len(incomplete)}, expected 2."
        )

    complete[
        "sample_index_88"
    ] = pd.to_numeric(
        complete[
            "sample_index_88"
        ],
        errors="raise",
    ).astype(int)

    if (
        complete[
            "sample_index_88"
        ].duplicated().any()
    ):
        raise RuntimeError(
            "Complete roster has duplicate sample_index_88."
        )

    if (
        complete[
            "sample_id"
        ].astype(
            str
        ).duplicated().any()
    ):
        raise RuntimeError(
            "Complete roster has duplicate sample_id."
        )

    # 05f3b-v2 roster was created in immutable NPZ order. Enforce that.
    complete = complete.sort_values(
        "sample_index_88",
        kind="mergesort",
    ).reset_index(
        drop=True
    )

    selected_idx = (
        complete[
            "sample_index_88"
        ]
        .to_numpy(
            dtype=int
        )
    )

    if (
        np.any(
            selected_idx < 0
        )
        or np.any(
            selected_idx
            >= EXPECTED_N_88
        )
    ):
        raise RuntimeError(
            "Complete roster sample_index_88 outside 0..87."
        )

    with np.load(
        TARGET_MATRIX_88,
        allow_pickle=False,
    ) as data:
        X88 = np.asarray(
            data[
                "X"
            ],
            dtype=np.float64,
        )
        sample88 = (
            data[
                "sample_id"
            ]
            .astype(str)
        )
        genes = (
            data[
                "human_gene_symbol"
            ]
            .astype(str)
        )
        source_features = (
            data[
                "source_expression_feature"
            ]
            .astype(str)
        )
        aligned_index = (
            data[
                "aligned_feature_index"
            ]
            .copy()
        )

    if (
        X88.shape
        != (
            EXPECTED_N_88,
            EXPECTED_FEATURES,
        )
    ):
        raise RuntimeError(
            f"Immutable TARGET matrix shape changed: {X88.shape}."
        )

    roster_ids = (
        complete[
            "sample_id"
        ]
        .astype(str)
        .to_numpy()
    )

    if not np.array_equal(
        sample88[
            selected_idx
        ],
        roster_ids,
    ):
        raise RuntimeError(
            "Complete roster sample_id does not match immutable NPZ at sample_index_88."
        )

    X86 = X88[
        selected_idx
    ].copy()

    if (
        X86.shape
        != (
            EXPECTED_N_86,
            EXPECTED_FEATURES,
        )
    ):
        raise RuntimeError(
            f"TARGET86 matrix shape={X86.shape}, expected (86, 11815)."
        )

    if not np.isfinite(
        X86
    ).all():
        raise RuntimeError(
            "TARGET86 expression matrix contains non-finite values."
        )

    time_values = pd.to_numeric(
        complete[
            "os_time_days_parsed"
        ],
        errors="raise",
    ).to_numpy(
        dtype=float
    )

    event_values = pd.to_numeric(
        complete[
            "os_event_parsed"
        ],
        errors="raise",
    ).astype(
        int
    ).to_numpy(
        dtype=int
    )

    if not set(
        np.unique(
            event_values
        ).tolist()
    ).issubset(
        {0, 1}
    ):
        raise RuntimeError(
            "TARGET86 event values are not binary 0/1."
        )

    event_bool = event_values.astype(
        bool
    )

    if (
        int(
            event_bool.sum()
        )
        != EXPECTED_EVENTS
    ):
        raise RuntimeError(
            "TARGET86 event count is not 29."
        )

    if (
        int(
            (
                ~event_bool
            ).sum()
        )
        != EXPECTED_CENSORED
    ):
        raise RuntimeError(
            "TARGET86 censored count is not 57."
        )

    if not np.isfinite(
        time_values
    ).all():
        raise RuntimeError(
            "TARGET86 contains non-finite time."
        )

    if np.any(
        time_values < 0
    ):
        raise RuntimeError(
            "TARGET86 contains negative time."
        )

    zero_mask = (
        time_values
        == 0.0
    )

    if (
        int(
            zero_mask.sum()
        )
        != EXPECTED_ZERO_TIME_N
    ):
        raise RuntimeError(
            f"TARGET86 zero-time count={int(zero_mask.sum())}, expected 1."
        )

    zero_ids = (
        roster_ids[
            zero_mask
        ]
        .tolist()
    )

    if (
        zero_ids
        != [
            EXPECTED_ZERO_TIME_SAMPLE
        ]
    ):
        raise RuntimeError(
            f"TARGET86 zero-time identity changed: {zero_ids}"
        )

    if bool(
        event_bool[
            zero_mask
        ][0]
    ):
        raise RuntimeError(
            "TARGET86 zero-time case is not censored."
        )

    case_keys = (
        complete[
            "case_key"
        ]
        .astype(str)
        .to_numpy()
    )

    if (
        len(
            set(
                case_keys.tolist()
            )
        )
        != EXPECTED_N_86
    ):
        raise RuntimeError(
            "TARGET86 case keys are not unique."
        )

    max_sample_len = max(
        len(
            str(x)
        )
        for x
        in roster_ids
    )

    max_gene_len = max(
        len(
            str(x)
        )
        for x
        in genes
    )

    max_feature_len = max(
        len(
            str(x)
        )
        for x
        in source_features
    )

    atomic_savez(
        TARGET86_MATRIX,
        X=X86,
        sample_id=np.asarray(
            roster_ids,
            dtype=(
                f"<U{max_sample_len}"
            ),
        ),
        human_gene_symbol=np.asarray(
            genes,
            dtype=(
                f"<U{max_gene_len}"
            ),
        ),
        source_expression_feature=np.asarray(
            source_features,
            dtype=(
                f"<U{max_feature_len}"
            ),
        ),
        aligned_feature_index=np.asarray(
            aligned_index
        ),
        source_sample_index_88=np.asarray(
            selected_idx,
            dtype=np.int32,
        ),
    )

    endpoint = pd.DataFrame(
        {
            "execution_index_86": np.arange(
                EXPECTED_N_86,
                dtype=int,
            ),
            "source_sample_index_88": selected_idx,
            "sample_id": roster_ids,
            "case_key": case_keys,
            "os_time_days": time_values,
            "os_event": event_values,
        }
    )

    endpoint.to_csv(
        TARGET86_ENDPOINT,
        sep="\t",
        index=False,
    )

    input_audit = {
        "status": "PASS",
        "created_utc": now_utc(),
        "source_TARGET88_matrix_sha256": sha256_file(
            TARGET_MATRIX_88
        ),
        "source_complete_roster_sha256": sha256_file(
            F3B_COMPLETE_ROSTER
        ),
        "TARGET86_matrix_path": str(
            TARGET86_MATRIX.relative_to(
                ROOT
            )
        ),
        "TARGET86_matrix_sha256": sha256_file(
            TARGET86_MATRIX
        ),
        "TARGET86_endpoint_path": str(
            TARGET86_ENDPOINT.relative_to(
                ROOT
            )
        ),
        "TARGET86_endpoint_sha256": sha256_file(
            TARGET86_ENDPOINT
        ),
        "n": EXPECTED_N_86,
        "features": EXPECTED_FEATURES,
        "events": EXPECTED_EVENTS,
        "censored": EXPECTED_CENSORED,
        "zero_time_n": EXPECTED_ZERO_TIME_N,
        "zero_time_sample": EXPECTED_ZERO_TIME_SAMPLE,
        "zero_time_event": 0,
        "additional_case_exclusions": 0,
        "outcome_imputation": False,
        "zero_time_shift": False,
        "subset_rule": (
            "exact 05f3b-v2 complete-primary-OS sample_index_88/sample_id roster"
        ),
    }

    write_json(
        INPUT_AUDIT,
        input_audit,
    )

    return {
        "X": X86,
        "sample_ids": roster_ids,
        "case_keys": case_keys,
        "time": time_values,
        "event": event_bool,
        "audit": input_audit,
    }


def main() -> None:
    started = now_utc()

    print("=" * 120)
    print(
        "Paper 6 - execute exact 86-case reconciled TARGET-OS evaluation"
    )
    print("=" * 120)
    print(
        f"Script version: {SCRIPT_VERSION}"
    )
    print()
    print("Post-opening execution contract:")
    print(
        "  TARGET outcomes already open: YES"
    )
    print(
        "  recovery from failed 05f3d Windows newline-byte check: YES"
    )
    print(
        "  failed 05f3d real TARGET model fitting occurred: NO"
    )
    print(
        "  runner patch mode: RAW BYTES / preserve source newline style"
    )
    print(
        "  authoritative endpoint population: 86/29/57"
    )
    print(
        "  zero-time case retained unchanged: YES"
    )
    print(
        "  additional clinical columns read: NO"
    )
    print(
        "  additional sample exclusions: NO"
    )
    print(
        "  outcome imputation/epsilon shift: NO"
    )
    print(
        "  authorized runner changes: EXACTLY 3"
    )
    print(
        "  model/hyperparameter/metric/CV/bootstrap/branch changes: NO"
    )
    print(
        "  GSE21257/GSE39055 outcomes read: NO"
    )
    print(
        f"  execution device: {EXECUTION_DEVICE.upper()}"
    )
    print()

    state = verify_authorization()

    patch_audit = create_exact_runner_v3()

    print("Runner v2 -> v3 exact RAW-BYTE amendment: PASS")
    print(
        f"  source runner SHA256: {EXPECTED_F2C_RUNNER_SHA256}"
    )
    print(
        f"  runner v3 SHA256: {sha256_file(RUNNER_V3)}"
    )
    print(
        "  reverse RAW-BYTE patch reproduces v2 byte-exact: PASS"
    )
    print(
        "  unauthorized runner changes: NO"
    )
    print(
        "  source newline style preserved: "
        f"{patch_audit['source_newline_style']}"
    )
    print()

    data = materialize_target86()

    print("TARGET86 execution input: PASS")
    print(
        f"  matrix shape: {data['X'].shape}"
    )
    print(
        "  events/censored: "
        f"{int(data['event'].sum())}/"
        f"{int((~data['event']).sum())}"
    )
    print(
        "  zero-time: 1 censored "
        f"({EXPECTED_ZERO_TIME_SAMPLE})"
    )
    print(
        "  additional exclusions: 0"
    )
    print()

    runner = load_module(
        RUNNER_V3,
        "paper6_TARGET86_runner_v3_05f3d",
    )

    preflight = runner.preflight()

    if (
        clean(
            preflight.get(
                "status"
            )
        )
        != "PASS"
    ):
        raise RuntimeError(
            "Runner v3 preflight failed."
        )

    if (
        EXECUTION_DEVICE
        != "cpu"
    ):
        raise RuntimeError(
            "Execution device changed from audited CPU path."
        )

    device = torch.device(
        "cpu"
    )

    print("=" * 120)
    print(
        "RUNNING REAL TARGET86 FROZEN EVALUATION"
    )
    print("=" * 120)
    print(
        f"Runner v3 SHA256: {sha256_file(RUNNER_V3)}"
    )
    print(
        "Models: T0 T1 T2 N0 N1 N2 N3 N4 N5"
    )
    print(
        "A3 / hardened A3: EXCLUDED"
    )
    print(
        "Outer CV: 5 folds x 20 repeats"
    )
    print(
        "Bootstrap: 5000 paired patient-clustered draws"
    )
    print(
        "Model selection: NO"
    )
    print(
        "Post-opening scientific tuning: NO"
    )
    print()

    branch_result = runner.run_full_evaluation(
        data["X"],
        data["sample_ids"],
        data["case_keys"],
        data["time"],
        data["event"],
        device=device,
        output_dir=EXEC_DIR,
    )

    model_summary_path = (
        EXEC_DIR
        / "TARGET_model_summary.tsv"
    )

    fold_metrics_path = (
        EXEC_DIR
        / "TARGET_fold_metrics.tsv"
    )

    branch_path = (
        EXEC_DIR
        / "TARGET_branch_result.json"
    )

    for path in [
        model_summary_path,
        fold_metrics_path,
        branch_path,
    ]:
        require_file(
            path
        )

    model_summary = pd.read_csv(
        model_summary_path,
        sep="\t",
    )

    final_branch = clean(
        branch_result.get(
            "final_branch_after_frozen_uncertainty_rules"
        )
    )

    candidate_branch = clean(
        branch_result.get(
            "candidate_branch_point_estimate"
        )
    )

    summary = {
        "script_version": SCRIPT_VERSION,
        "execution_version": EXECUTION_VERSION,
        "status": "PASS",
        "scientific_status": (
            "PASS_TARGET86_POSTOPENING_AUDITED_FROZEN_DESCRIPTIVE_EVALUATION_COMPLETE"
        ),
        "run_started_utc": started,
        "run_finished_utc": now_utc(),

        "scientific_role": (
            "post-HOLD descriptive/non-confirmatory human mechanistic stress test"
        ),

        "postopening_amendment_status": {
            "05f3b_preexisting_complete_case_reconciliation": "PASS",
            "05f3c_zero_time_execution_semantics": "PASS",
            "runner_v2_to_v3_exact_raw_byte_patch": "PASS",
            "05f3d_failed_before_real_model_fit": True,
            "05f3d_failure_reason": (
                "Windows text-mode LF-to-CRLF translation caused a false "
                "raw-byte replay mismatch"
            ),
            "scientific_contract_changed": False,
        },

        "TARGET_n": EXPECTED_N_86,
        "TARGET_events": EXPECTED_EVENTS,
        "TARGET_censored": EXPECTED_CENSORED,
        "TARGET_zero_time_n": EXPECTED_ZERO_TIME_N,
        "TARGET_zero_time_sample": EXPECTED_ZERO_TIME_SAMPLE,

        "candidate_branch_point_estimate": candidate_branch,
        "final_branch": final_branch,
        "bootstrap_status": branch_result.get(
            "bootstrap_status"
        ),
        "reporting_flags": branch_result.get(
            "reporting_flags"
        ),

        "model_selection_performed": False,
        "05d_HOLD_may_change": False,
        "A6_may_reopen": False,

        "GSE21257_outcomes_read": False,
        "GSE39055_outcomes_read": False,

        "runner_v3_sha256": sha256_file(
            RUNNER_V3
        ),
        "runner_patch_audit_sha256": sha256_file(
            PATCH_AUDIT
        ),
        "05f3b_complete_roster_sha256": state[
            "f3b_complete_roster_sha256"
        ],
        "05f3b_incomplete_roster_sha256": state[
            "f3b_incomplete_roster_sha256"
        ],
        "05f3c_patch_plan_sha256": state[
            "f3c_patch_plan_sha256"
        ],
        "TARGET86_matrix_sha256": sha256_file(
            TARGET86_MATRIX
        ),
        "TARGET86_endpoint_sha256": sha256_file(
            TARGET86_ENDPOINT
        ),
        "TARGET86_input_audit_sha256": sha256_file(
            INPUT_AUDIT
        ),

        "model_summary_sha256": sha256_file(
            model_summary_path
        ),
        "fold_metrics_sha256": sha256_file(
            fold_metrics_path
        ),
        "branch_result_sha256": sha256_file(
            branch_path
        ),

        "execution_output_hashes": {
            path.name: sha256_file(
                path
            )
            for path in sorted(
                EXEC_DIR.iterdir()
            )
            if path.is_file()
        },

        "post_result_rule": (
            "No runner/model/hyperparameter/metric/CV/bootstrap/branch "
            "modification is permitted in response to TARGET performance. "
            "Interpret only through the frozen descriptive branch framework; "
            "GSE21257 and GSE39055 remain sealed."
        ),
    }

    write_json(
        SUMMARY_JSON,
        summary,
    )

    print()
    print("=" * 120)
    print(
        "05f3e REAL TARGET86 RESULT"
    )
    print("=" * 120)
    print(
        f"n={EXPECTED_N_86}, "
        f"OS events={EXPECTED_EVENTS}, "
        f"censored={EXPECTED_CENSORED}"
    )
    print(
        f"zero-time censored case retained: {EXPECTED_ZERO_TIME_SAMPLE}"
    )
    print()

    display_cols = [
        column
        for column in [
            "model_id",
            "uno_c",
            "ibs",
            "risk_sd",
            "risk_q99_q01",
            "uno_valid_repeats",
            "ibs_valid_repeats",
        ]
        if column
        in model_summary.columns
    ]

    print(
        model_summary[
            display_cols
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"Point-estimate candidate branch: {candidate_branch}"
    )
    print(
        f"Final frozen branch: {final_branch}"
    )
    print(
        "Bootstrap status: "
        f"{branch_result.get('bootstrap_status', 'UNKNOWN')}"
    )

    print()
    print(
        "Reporting flags:"
    )

    for key, value in sorted(
        (
            branch_result.get(
                "reporting_flags"
            )
            or {}
        ).items()
    ):
        print(
            f"  {key}: {value}"
        )

    print()
    print(
        "Frozen prior states remain:"
    )
    print(
        "  05d architecture selection: HOLD_NO_SAFE_SELECTABLE_ARCHITECTURE"
    )
    print(
        "  A6: STOP_A6_FOR_PAPER6"
    )
    print(
        "  TARGET role: DESCRIPTIVE / NON-CONFIRMATORY"
    )
    print(
        "  GSE21257 outcomes: SEALED"
    )
    print(
        "  GSE39055 outcomes: SEALED"
    )

    print()
    print(
        f"05f3d summary SHA256: {sha256_file(SUMMARY_JSON)}"
    )

    print("=" * 120)
    print(
        "05f3e: PASS_TARGET86_POSTOPENING_AUDITED_FROZEN_DESCRIPTIVE_EVALUATION_COMPLETE"
    )
    print("=" * 120)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print(
            "=" * 120,
            file=sys.stderr,
        )
        print(
            "05f3e exact TARGET86 bytesafe frozen execution: FAIL",
            file=sys.stderr,
        )
        print(
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        print(
            "=" * 120,
            file=sys.stderr,
        )
        raise
