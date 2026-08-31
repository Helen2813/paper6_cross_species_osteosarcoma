#!/usr/bin/env python3
"""
Paper 6 - FIRST reserved TARGET-OS outcome opening and frozen execution.

IMPORTANT
---------
This is the first Paper-6 access to reserved TARGET-OS endpoint VALUES.

It is authorized only because:
- 05f1a froze the outcome-free representation;
- 05f1b materialized raw aligned DOG2/TARGET matrices;
- 05f1c v2 froze clinical schema, ID mapping, endpoint parser,
  5x20 CV, bootstrap and branch rules;
- 05f2a v2 froze exact implementation sources;
- 05f2b froze DOG2 source artifacts and the complete TARGET runner;
- 05f2c v2 technically validated/amended that runner pre-outcome and
  froze the authoritative runner v2 SHA.

This script MUST NOT modify any frozen model/source/runner artifact.

Allowed TARGET clinical columns read:
    case_submitter_id
    os_event
    os_time_days

No other TARGET clinical column is read.
No secondary endpoint is read.
GSE21257 and GSE39055 remain sealed.

After endpoint opening, the exact SHA-locked runner v2 is executed without
scientific modification.

No CLI arguments.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import torch
except ImportError as exc:
    raise ImportError("05f3 requires PyTorch in the active Paper-6 .venv.") from exc


SCRIPT_VERSION = (
    "05f3a-open-reserved-target-os-id-order-fix-and-run-frozen-evaluation-v1-no-cli"
)
EXECUTION_VERSION = (
    "paper6-first-reserved-target-os-frozen-execution-id-order-fix-v1"
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "_config"
PAPER4_BASENAME = "paper4_sarcoma_dog"

# ---------------------------------------------------------------------------
# Final pre-outcome authoritative runner.
# ---------------------------------------------------------------------------
F2C_DIR = ROOT / "results" / "human_posthold_descriptive" / "05f2c_v2"
F2C_CONTRACT = F2C_DIR / "technical_runner_amendment_contract.json"
F2C_SUMMARY = F2C_DIR / "summary.json"
F2C_RUNNER = (
    F2C_DIR / "runner" / "frozen_TARGET_evaluation_runner_v2.py"
)
F2C_SMOKE = F2C_DIR / "synthetic_runner_smoke_test.json"

EXPECTED_F2C_STATUS = (
    "PASS_TARGET_RUNNER_V2_TECHNICALLY_VALIDATED_PREOUTCOME"
)
EXPECTED_F2C_RUNNER_SHA256 = (
    "8c7d5e03912463f51220057a35aaa14811c9f423c6fe0e24061aff29251f8075"
)

# ---------------------------------------------------------------------------
# Frozen clinical schema/mapping.
# ---------------------------------------------------------------------------
F1C_DIR = ROOT / "results" / "human_posthold_descriptive" / "05f1c_v2"
F1C_ENDPOINT_SCHEMA = F1C_DIR / "TARGET_OS_endpoint_schema.json"
F1C_ID_MAPPING = F1C_DIR / "TARGET_expression_to_clinical_id_mapping.tsv"
F1C_RESAMPLING = F1C_DIR / "TARGET_resampling_and_CI_contract.json"
F1C_SUMMARY = F1C_DIR / "summary.json"

EXPECTED_F1C_ENDPOINT_SCHEMA_SHA256 = (
    "6c827bb13d3a44a651d012b5a51a0b96ffb45682cd73ab545da2cb9edda79ce2"
)
EXPECTED_F1C_RESAMPLING_SHA256 = (
    "0d6d004b2ea7007c0fde853ff10d97e0df734ea4571f1d55bc2efcfef9367e68"
)

# ---------------------------------------------------------------------------
# Frozen TARGET matrix.
# ---------------------------------------------------------------------------
F1B_DIR = ROOT / "results" / "human_posthold_descriptive" / "05f1b"
TARGET_MATRIX = (
    F1B_DIR / "matrices" / "TARGET_OS_raw_aligned_11815genes.npz"
)
EXPECTED_TARGET_MATRIX_SHA256 = (
    "7aca20d98731a24d90882f2042b1aa7c7aefc0095a250d4ed42c3502cc3c7b19"
)

# ---------------------------------------------------------------------------
# Clinical asset lock.
# ---------------------------------------------------------------------------
UPSTREAM_LOCK = ROOT / "contracts" / "00_upstream_input_lock.json"

EXPECTED_CLINICAL_ROLE = "target_clinical"
EXPECTED_ID_COLUMN = "case_submitter_id"
EXPECTED_EVENT_COLUMN = "os_event"
EXPECTED_TIME_COLUMN = "os_time_days"
EXPECTED_TARGET_N = 88

# ---------------------------------------------------------------------------
# 05f3 outputs.
# ---------------------------------------------------------------------------
OUT_DIR = ROOT / "results" / "human_posthold_descriptive" / "05f3a"
EXEC_DIR = OUT_DIR / "frozen_execution"
OUT_DIR.mkdir(parents=True, exist_ok=True)
EXEC_DIR.mkdir(parents=True, exist_ok=True)

PREOPEN_LOCK = OUT_DIR / "PREOPENING_LOCK.json"
OPENING_AUDIT = OUT_DIR / "FIRST_TARGET_OS_OPENING_AUDIT.json"
PARSED_ENDPOINT = OUT_DIR / "TARGET_OS_parsed_primary_endpoint.tsv"
SUMMARY_JSON = OUT_DIR / "summary.json"

# Device is fixed BEFORE endpoint values are read.
# The pre-outcome runner smoke was validated on CPU; target folds are tiny.
EXECUTION_DEVICE = "cpu"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "na", "n/a"}:
        return ""
    return text


def require_file(path: Path) -> Path:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Required frozen artifact missing: {path}")
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
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        handle.write("\n")


def local_path_config() -> Dict[str, Any]:
    path = CONFIG_DIR / "paths.local.json"
    if not path.exists():
        return {}
    payload = read_json(path)
    return payload if isinstance(payload, dict) else {}


def resolve_paper4_root() -> Tuple[Path, str]:
    candidates: List[Tuple[Path, str]] = []

    env_value = os.environ.get("PAPER4_ROOT", "").strip()
    if env_value:
        candidates.append(
            (Path(env_value).expanduser(), "environment:PAPER4_ROOT")
        )

    config_value = clean(local_path_config().get("paper4_root"))
    if config_value:
        candidates.append(
            (Path(config_value).expanduser(), "_config/paths.local.json")
        )

    candidates.extend(
        [
            (ROOT.parent / PAPER4_BASENAME, "sibling_repository"),
            (Path.home() / "Desktop" / PAPER4_BASENAME, "home_desktop_fallback"),
        ]
    )

    checked = []

    for candidate, source in candidates:
        resolved = candidate.resolve()
        checked.append(f"{source}: {resolved}")
        if resolved.is_dir():
            return resolved, source

    raise FileNotFoundError(
        "Could not resolve Paper 4 root. Checked:\n  - "
        + "\n  - ".join(checked)
    )


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import frozen runner: {path}")

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


def get_locked_clinical_asset(
    lock: Dict[str, Any],
    paper4_root: Path,
) -> Tuple[Path, Dict[str, Any]]:
    item = (lock.get("assets") or {}).get(EXPECTED_CLINICAL_ROLE)

    if not isinstance(item, dict):
        raise RuntimeError(
            f"Upstream lock lacks exact role {EXPECTED_CLINICAL_ROLE!r}."
        )

    rel = clean(item.get("relative_path"))
    expected_hash = clean(item.get("sha256")).lower()

    if not rel or len(expected_hash) != 64:
        raise RuntimeError("Malformed locked TARGET clinical asset.")

    path = paper4_root / rel
    require_file(path)

    observed_hash = sha256_file(path).lower()

    if observed_hash != expected_hash:
        raise RuntimeError(
            "Locked TARGET clinical file changed before first outcome opening: "
            f"expected={expected_hash}, observed={observed_hash}"
        )

    return path, item


def verify_final_preoutcome_freeze() -> Dict[str, Any]:
    for path in [
        F2C_CONTRACT,
        F2C_SUMMARY,
        F2C_RUNNER,
        F2C_SMOKE,
        F1C_ENDPOINT_SCHEMA,
        F1C_ID_MAPPING,
        F1C_RESAMPLING,
        F1C_SUMMARY,
        TARGET_MATRIX,
        UPSTREAM_LOCK,
    ]:
        require_file(path)

    if sha256_file(F2C_RUNNER) != EXPECTED_F2C_RUNNER_SHA256:
        raise RuntimeError(
            "Authoritative 05f2c runner SHA changed before first outcome opening."
        )

    if sha256_file(F1C_ENDPOINT_SCHEMA) != EXPECTED_F1C_ENDPOINT_SCHEMA_SHA256:
        raise RuntimeError("05f1c endpoint schema hash changed.")

    if sha256_file(F1C_RESAMPLING) != EXPECTED_F1C_RESAMPLING_SHA256:
        raise RuntimeError("05f1c resampling contract hash changed.")

    if sha256_file(TARGET_MATRIX) != EXPECTED_TARGET_MATRIX_SHA256:
        raise RuntimeError("05f1b TARGET raw matrix hash changed.")

    f2c = read_json(F2C_SUMMARY)
    f2c_contract = read_json(F2C_CONTRACT)

    if clean(f2c.get("scientific_status")) != EXPECTED_F2C_STATUS:
        raise RuntimeError("05f2c v2 is not in expected final PASS state.")

    if f2c.get("TARGET_OS_event_values_read") is not False:
        raise RuntimeError(
            "05f2c provenance unexpectedly indicates prior TARGET event access."
        )

    if f2c.get("TARGET_OS_time_values_read") is not False:
        raise RuntimeError(
            "05f2c provenance unexpectedly indicates prior TARGET time access."
        )

    if f2c.get(
        "TARGET_endpoint_values_may_be_read_after_this_PASS"
    ) is not True:
        raise RuntimeError("05f2c did not authorize endpoint opening.")

    if f2c.get("runner_v2_sha256") != EXPECTED_F2C_RUNNER_SHA256:
        raise RuntimeError("05f2c summary runner SHA mismatch.")

    if f2c.get("source_artifacts_unchanged") is not True:
        raise RuntimeError("05f2c source-artifact immutability check not PASS.")

    authorization = (
        f2c_contract.get("outcome_opening_authorization") or {}
    )

    if authorization.get(
        "TARGET_endpoint_values_may_be_read_after_05f2c_PASS"
    ) is not True:
        raise RuntimeError("05f2c contract does not authorize opening.")

    if clean(
        authorization.get("authoritative_runner_sha256")
    ) != EXPECTED_F2C_RUNNER_SHA256:
        raise RuntimeError("05f2c contract authoritative runner SHA mismatch.")

    endpoint_schema = read_json(F1C_ENDPOINT_SCHEMA)

    if endpoint_schema.get("endpoint_value_access_in_05f1c") is not False:
        raise RuntimeError("05f1c endpoint provenance changed.")

    os_schema = endpoint_schema.get("OS_schema") or {}

    if clean(endpoint_schema.get("identifier_column")) != EXPECTED_ID_COLUMN:
        raise RuntimeError("Frozen identifier column changed.")

    if clean(os_schema.get("event_column")) != EXPECTED_EVENT_COLUMN:
        raise RuntimeError("Frozen OS event column changed.")

    if clean(os_schema.get("time_column")) != EXPECTED_TIME_COLUMN:
        raise RuntimeError("Frozen OS time column changed.")

    if clean(os_schema.get("schema_type")) != "PRECOMPUTED_OS_EVENT_PLUS_TIME":
        raise RuntimeError("Frozen TARGET OS schema type changed.")

    return {
        "05f2c_contract_sha256": sha256_file(F2C_CONTRACT),
        "05f2c_summary_sha256": sha256_file(F2C_SUMMARY),
        "05f2c_runner_sha256": sha256_file(F2C_RUNNER),
        "05f2c_smoke_sha256": sha256_file(F2C_SMOKE),
        "05f1c_endpoint_schema_sha256": sha256_file(F1C_ENDPOINT_SCHEMA),
        "05f1c_id_mapping_sha256": sha256_file(F1C_ID_MAPPING),
        "05f1c_resampling_sha256": sha256_file(F1C_RESAMPLING),
        "TARGET_matrix_sha256": sha256_file(TARGET_MATRIX),
        "endpoint_schema": endpoint_schema,
    }


def parse_event_value(
    raw_value: Any,
    parser_contract: Dict[str, Any],
) -> bool:
    text = clean(raw_value)

    if not text:
        raise RuntimeError("Missing TARGET os_event value.")

    lower = text.lower()

    event_strings = {
        clean(x).lower()
        for x in parser_contract.get(
            "accepted_event_strings_case_insensitive",
            []
        )
    }
    censored_strings = {
        clean(x).lower()
        for x in parser_contract.get(
            "accepted_censored_strings_case_insensitive",
            []
        )
    }

    if lower in event_strings:
        return True

    if lower in censored_strings:
        return False

    # Frozen contract also explicitly allows numeric 0/1 encoding.
    try:
        numeric = float(text)
    except ValueError:
        numeric = float("nan")

    if np.isfinite(numeric):
        if numeric == 1.0:
            return True
        if numeric == 0.0:
            return False

    raise RuntimeError(
        f"Unrecognized TARGET os_event value under frozen parser: {text!r}"
    )


def parse_time_value(
    raw_value: Any,
) -> float:
    text = clean(raw_value)

    if not text:
        raise RuntimeError("Missing TARGET os_time_days value.")

    try:
        value = float(text)
    except ValueError as exc:
        raise RuntimeError(
            f"Non-numeric TARGET os_time_days value: {text!r}"
        ) from exc

    if not np.isfinite(value):
        raise RuntimeError(
            f"Non-finite TARGET os_time_days value: {text!r}"
        )

    if value <= 0:
        raise RuntimeError(
            f"Non-positive TARGET os_time_days value: {value}"
        )

    return float(value)


def read_exact_primary_endpoint(
    clinical_path: Path,
) -> pd.DataFrame:
    # FIRST reserved outcome read occurs here.
    # No other clinical columns are loaded.
    frame = pd.read_csv(
        clinical_path,
        usecols=[
            EXPECTED_ID_COLUMN,
            EXPECTED_EVENT_COLUMN,
            EXPECTED_TIME_COLUMN,
        ],
        dtype=str,
        low_memory=False,
    ).fillna("")

    if list(frame.columns) != [
        EXPECTED_ID_COLUMN,
        EXPECTED_EVENT_COLUMN,
        EXPECTED_TIME_COLUMN,
    ]:
        # pandas normally retains source-file order for usecols.
        # Reorder explicitly but fail if any unexpected column was read.
        if set(frame.columns) != {
            EXPECTED_ID_COLUMN,
            EXPECTED_EVENT_COLUMN,
            EXPECTED_TIME_COLUMN,
        }:
            raise RuntimeError(
                f"Unexpected TARGET clinical columns read: {list(frame.columns)}"
            )

        frame = frame[
            [
                EXPECTED_ID_COLUMN,
                EXPECTED_EVENT_COLUMN,
                EXPECTED_TIME_COLUMN,
            ]
        ]

    return frame


def main() -> None:
    started = now_utc()

    print("=" * 120)
    print("Paper 6 - FIRST reserved TARGET-OS opening: pre-outcome ID-order fix + frozen execution")
    print("=" * 120)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Execution version: {EXECUTION_VERSION}")
    print()
    print("PRE-OPENING SAFETY:")
    print("  technical recovery from failed 05f3 v1 ID-order check: YES")
    print("  failed 05f3 v1 TARGET outcome values read: NO")
    print("  05f1c mapping row-order trusted: NO")
    print("  mapping identity reconciled by exact expression_sample_id: YES")
    print("  scientific contract changes: NO")
    print("  final 05f2c runner v2 SHA verification: REQUIRED")
    print("  TARGET raw matrix SHA verification: REQUIRED")
    print("  clinical file SHA verification: REQUIRED")
    print("  allowed clinical columns: case_submitter_id, os_event, os_time_days")
    print("  secondary endpoint columns: FORBIDDEN")
    print("  GSE21257/GSE39055 outcomes: FORBIDDEN")
    print(f"  execution device fixed pre-outcome: {EXECUTION_DEVICE.upper()}")
    print()

    pre_state = verify_final_preoutcome_freeze()

    upstream_lock = read_json(UPSTREAM_LOCK)
    paper4_root, paper4_resolution = resolve_paper4_root()

    clinical_path, clinical_asset = get_locked_clinical_asset(
        upstream_lock,
        paper4_root,
    )

    endpoint_schema = pre_state["endpoint_schema"]

    # Load outcome-free artifacts before first endpoint read.
    mapping = pd.read_csv(
        F1C_ID_MAPPING,
        sep="\t",
        dtype=str,
        low_memory=False,
    ).fillna("")

    # ------------------------------------------------------------------
    # PRE-OUTCOME TECHNICAL RECOVERY:
    # 05f1c read sample_index as string before sort_values(), so its mapping
    # row order can be lexicographic (0,1,10,...,2,...) even though the
    # expression_sample_id -> clinical-row identity is correct.
    #
    # Scientific identity is therefore reconciled by exact sample ID, not by
    # trusting 05f1c row order. This occurs BEFORE any clinical outcome value
    # is read.
    # ------------------------------------------------------------------
    mapping["expression_sample_index"] = pd.to_numeric(
        mapping["expression_sample_index"],
        errors="raise",
    ).astype(int)

    mapping["clinical_row_index"] = pd.to_numeric(
        mapping["clinical_row_index"],
        errors="raise",
    ).astype(int)

    if len(mapping) != EXPECTED_TARGET_N:
        raise RuntimeError(
            f"Frozen ID mapping has {len(mapping)} rows; expected 88."
        )

    if mapping["expression_sample_id"].astype(str).duplicated().any():
        duplicated = (
            mapping.loc[
                mapping["expression_sample_id"].astype(str).duplicated(),
                "expression_sample_id",
            ]
            .astype(str)
            .unique()
            .tolist()
        )
        raise RuntimeError(
            f"Frozen 05f1c mapping has duplicate expression_sample_id values: "
            f"{duplicated[:10]}"
        )

    if mapping["clinical_row_index"].duplicated().any():
        raise RuntimeError(
            "Frozen 05f1c mapping clinical_row_index is not one-to-one."
        )

    with np.load(
        TARGET_MATRIX,
        allow_pickle=False,
    ) as target_npz:
        X_target = np.asarray(
            target_npz["X"],
            dtype=np.float64,
        )
        sample_ids = target_npz["sample_id"].astype(str)

    if X_target.shape != (EXPECTED_TARGET_N, 11815):
        raise RuntimeError(
            f"Frozen TARGET matrix shape={X_target.shape}; expected (88, 11815)."
        )

    if len(set(sample_ids.tolist())) != EXPECTED_TARGET_N:
        raise RuntimeError(
            "Frozen TARGET NPZ sample IDs are not unique 88/88."
        )

    mapping_ids = (
        mapping["expression_sample_id"]
        .astype(str)
        .to_numpy()
    )

    if set(mapping_ids.tolist()) != set(sample_ids.tolist()):
        missing_in_mapping = sorted(
            set(sample_ids.tolist()) - set(mapping_ids.tolist())
        )
        extra_in_mapping = sorted(
            set(mapping_ids.tolist()) - set(sample_ids.tolist())
        )
        raise RuntimeError(
            "Frozen TARGET NPZ and 05f1c mapping do not contain the same "
            "expression sample IDs. "
            f"missing_in_mapping={missing_in_mapping[:10]}, "
            f"extra_in_mapping={extra_in_mapping[:10]}"
        )

    mapping_order_before = mapping_ids.copy()

    # Exact deterministic reconciliation to the immutable 05f1b NPZ order.
    mapping = (
        mapping
        .set_index("expression_sample_id", drop=False)
        .loc[sample_ids.tolist()]
        .reset_index(drop=True)
    )

    reconciled_ids = (
        mapping["expression_sample_id"]
        .astype(str)
        .to_numpy()
    )

    if not np.array_equal(sample_ids, reconciled_ids):
        raise RuntimeError(
            "Exact sample-ID reconciliation to TARGET NPZ order failed."
        )

    mapping_order_was_different = bool(
        not np.array_equal(
            mapping_order_before,
            sample_ids,
        )
    )

    if not np.isfinite(X_target).all():
        raise RuntimeError("Frozen TARGET matrix contains non-finite values.")

    print("PRE-OUTCOME ID-ORDER RECOVERY: PASS")
    print("  NPZ/mapping sample-ID sets identical: 88/88")
    print(f"  original 05f1c row order differed from NPZ: {mapping_order_was_different}")
    print("  mapping reordered by exact expression_sample_id: YES")
    print("  clinical outcome values read during recovery: NO")
    print()

    runner = load_module(
        F2C_RUNNER,
        "paper6_authoritative_target_runner_05f3",
    )

    runner_preflight = runner.preflight()

    if clean(runner_preflight.get("status")) != "PASS":
        raise RuntimeError("Authoritative TARGET runner preflight failed.")

    # ------------------------------------------------------------------
    # Immutable evidence written BEFORE first endpoint value read.
    # ------------------------------------------------------------------
    preopen_payload = {
        "script_version": SCRIPT_VERSION,
        "execution_version": EXECUTION_VERSION,
        "created_utc": now_utc(),
        "status": "LOCKED_BEFORE_FIRST_TARGET_ENDPOINT_VALUE_READ",
        "Paper6_TARGET_outcomes_read_at_lock_time": False,
        "execution_device": EXECUTION_DEVICE,
        "paper4_root_resolution_source": paper4_resolution,
        "clinical_asset_role": EXPECTED_CLINICAL_ROLE,
        "clinical_asset_relative_path": clean(
            clinical_asset.get("relative_path")
        ),
        "clinical_asset_sha256": clean(
            clinical_asset.get("sha256")
        ),
        "allowed_columns": [
            EXPECTED_ID_COLUMN,
            EXPECTED_EVENT_COLUMN,
            EXPECTED_TIME_COLUMN,
        ],
        "secondary_columns_allowed": False,
        "technical_recovery": {
            "prior_failed_script": (
                "05f3_open_reserved_target_os_and_run_frozen_evaluation.py"
            ),
            "prior_failure_stage": "pre-outcome NPZ-vs-05f1c mapping order check",
            "prior_runs_count": 2,
            "prior_runs_reached_endpoint_read": False,
            "root_cause": (
                "05f1c mapping order was generated after lexicographic sorting "
                "of string-valued sample_index; sample identities themselves "
                "were correct and unique"
            ),
            "NPZ_mapping_ID_set_identity": "88/88 exact",
            "mapping_order_was_different": mapping_order_was_different,
            "reconciliation_rule": (
                "reorder frozen 05f1c mapping by exact expression_sample_id "
                "to immutable 05f1b NPZ sample order"
            ),
            "scientific_contract_changed": False,
        },
        "preoutcome_hashes": {
            key: value
            for key, value in pre_state.items()
            if key != "endpoint_schema"
        },
        "runner_preflight": runner_preflight,
        "runner_or_source_modification_after_opening": "FORBIDDEN",
    }

    write_json(
        PREOPEN_LOCK,
        preopen_payload,
    )

    print("PRE-OPENING LOCK: PASS")
    print(f"  clinical file SHA256: {sha256_file(clinical_path)}")
    print(f"  runner v2 SHA256: {sha256_file(F2C_RUNNER)}")
    print(f"  TARGET matrix SHA256: {sha256_file(TARGET_MATRIX)}")
    print(f"  pre-opening lock SHA256: {sha256_file(PREOPEN_LOCK)}")
    print()
    print("=" * 120)
    print("OPENING RESERVED TARGET-OS NOW")
    print("=" * 120)

    # ==================================================================
    # FIRST PAPER-6 TARGET OUTCOME VALUE ACCESS.
    # ==================================================================
    clinical = read_exact_primary_endpoint(
        clinical_path
    )
    first_opened_utc = now_utc()

    event_parser = endpoint_schema.get("event_parser") or {}

    clinical_ids = clinical[EXPECTED_ID_COLUMN].astype(str).map(clean)

    if clinical_ids.eq("").any():
        raise RuntimeError("Blank case_submitter_id in TARGET clinical file.")

    if clinical_ids.duplicated().any():
        duplicated = (
            clinical_ids[
                clinical_ids.duplicated()
            ]
            .unique()
            .tolist()
        )
        raise RuntimeError(
            f"Duplicate TARGET case_submitter_id values: {duplicated[:10]}"
        )

    event_by_row = np.asarray(
        [
            parse_event_value(value, event_parser)
            for value in clinical[EXPECTED_EVENT_COLUMN]
        ],
        dtype=bool,
    )

    time_by_row = np.asarray(
        [
            parse_time_value(value)
            for value in clinical[EXPECTED_TIME_COLUMN]
        ],
        dtype=np.float64,
    )

    # Frozen mapping uses exact clinical row indices discovered pre-outcome.
    clinical_row_index = (
        mapping["clinical_row_index"]
        .astype(int)
        .to_numpy()
    )

    if np.any(clinical_row_index < 0) or np.any(
        clinical_row_index >= len(clinical)
    ):
        raise RuntimeError(
            "Frozen clinical row index outside current clinical table."
        )

    mapped_clinical_ids = (
        clinical.iloc[clinical_row_index][EXPECTED_ID_COLUMN]
        .astype(str)
        .map(clean)
        .to_numpy()
    )

    frozen_clinical_ids = (
        mapping["clinical_identifier"]
        .astype(str)
        .to_numpy()
    )

    if not np.array_equal(
        np.char.upper(mapped_clinical_ids.astype(str)),
        np.char.upper(frozen_clinical_ids.astype(str)),
    ):
        raise RuntimeError(
            "TARGET clinical row identity differs from frozen 05f1c mapping."
        )

    event = event_by_row[clinical_row_index]
    time_days = time_by_row[clinical_row_index]

    case_keys = (
        mapping["frozen_case_key"]
        .astype(str)
        .to_numpy()
    )

    if len(set(case_keys.tolist())) != EXPECTED_TARGET_N:
        raise RuntimeError("Frozen TARGET case keys are not unique.")

    if int(event.sum()) < 5 or int((~event).sum()) < 5:
        raise RuntimeError(
            "TARGET event/censor support cannot satisfy frozen 5-fold design."
        )

    parsed_endpoint = pd.DataFrame(
        {
            "sample_index": np.arange(
                EXPECTED_TARGET_N,
                dtype=int,
            ),
            "sample_id": sample_ids,
            "case_key": case_keys,
            "os_time_days": time_days,
            "os_event": event.astype(int),
        }
    )

    parsed_endpoint.to_csv(
        PARSED_ENDPOINT,
        sep="\t",
        index=False,
    )

    opening_audit = {
        "script_version": SCRIPT_VERSION,
        "execution_version": EXECUTION_VERSION,
        "status": "FIRST_RESERVED_TARGET_OS_OPENED_AND_PARSED",
        "first_TARGET_endpoint_value_access_utc": first_opened_utc,
        "Paper6_TARGET_outcomes_read": True,
        "primary_endpoint": "OS",
        "columns_read": [
            EXPECTED_ID_COLUMN,
            EXPECTED_EVENT_COLUMN,
            EXPECTED_TIME_COLUMN,
        ],
        "secondary_endpoint_values_read": False,
        "clinical_rows": int(len(clinical)),
        "expression_samples_mapped": int(len(mapping)),
        "mapped_one_to_one": True,
        "preoutcome_ID_order_recovery": {
            "performed": True,
            "NPZ_mapping_ID_set_identity": "88/88 exact",
            "mapping_order_was_different": mapping_order_was_different,
            "reordered_by_expression_sample_id": True,
            "scientific_contract_changed": False,
        },
        "n_samples": int(len(event)),
        "n_events": int(event.sum()),
        "n_censored": int((~event).sum()),
        "event_fraction": float(event.mean()),
        "time_days_min": float(np.min(time_days)),
        "time_days_median": float(np.median(time_days)),
        "time_days_max": float(np.max(time_days)),
        "all_times_finite_positive": True,
        "event_parser_status": "PASS_FROZEN_GRAMMAR",
        "id_mapping_status": "PASS_EXACT_FROZEN_88_OF_88",
        "parsed_endpoint_sha256": sha256_file(PARSED_ENDPOINT),
        "preopening_lock_sha256": sha256_file(PREOPEN_LOCK),
        "runner_sha256_at_opening": sha256_file(F2C_RUNNER),
        "TARGET_matrix_sha256_at_opening": sha256_file(TARGET_MATRIX),
        "GSE21257_outcomes_read": False,
        "GSE39055_outcomes_read": False,
    }

    write_json(
        OPENING_AUDIT,
        opening_audit,
    )

    print("FIRST TARGET-OS OPENING: PASS")
    print(f"  samples: {len(event)}")
    print(f"  OS events: {int(event.sum())}")
    print(f"  censored: {int((~event).sum())}")
    print(
        f"  event fraction: {float(event.mean()):.4f}"
    )
    print(
        "  OS time days min/median/max: "
        f"{np.min(time_days):.1f}/"
        f"{np.median(time_days):.1f}/"
        f"{np.max(time_days):.1f}"
    )
    print("  ID mapping 88/88: PASS")
    print("  frozen parser: PASS")
    print("  secondary endpoints read: NO")
    print(f"  opening audit SHA256: {sha256_file(OPENING_AUDIT)}")
    print()

    # ------------------------------------------------------------------
    # Frozen execution begins immediately after successful opening audit.
    # ------------------------------------------------------------------
    if EXECUTION_DEVICE != "cpu":
        raise RuntimeError(
            "05f3 execution device was not frozen to CPU before outcome opening."
        )

    device = torch.device("cpu")

    print("=" * 120)
    print("RUNNING SHA-LOCKED TARGET EVALUATION")
    print("=" * 120)
    print(f"Runner SHA256: {sha256_file(F2C_RUNNER)}")
    print("Models: T0-T2 / N0-N5")
    print("A3/hardened A3: EXCLUDED")
    print("Outer CV: 5 folds x 20 repeats")
    print("Bootstrap: 5,000 paired patient-clustered draws")
    print("Model selection: NO")
    print("Scientific changes after opening: NO")
    print()

    branch_result = runner.run_full_evaluation(
        X_target,
        sample_ids,
        case_keys,
        time_days,
        event,
        device=device,
        output_dir=EXEC_DIR,
    )

    model_summary_path = EXEC_DIR / "TARGET_model_summary.tsv"
    branch_result_path = EXEC_DIR / "TARGET_branch_result.json"

    require_file(model_summary_path)
    require_file(branch_result_path)

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
            "PASS_FIRST_RESERVED_TARGET_OS_FROZEN_DESCRIPTIVE_EVALUATION_COMPLETE"
        ),
        "run_started_utc": started,
        "first_TARGET_endpoint_value_access_utc": first_opened_utc,
        "run_finished_utc": now_utc(),
        "TARGET_outcomes_now_open_for_Paper6": True,
        "preoutcome_ID_order_recovery": {
            "performed": True,
            "mapping_order_was_different": mapping_order_was_different,
            "NPZ_mapping_ID_set_identity": "88/88 exact",
            "scientific_contract_changed": False,
        },
        "scientific_role": (
            "post-HOLD descriptive/non-confirmatory human mechanistic stress test"
        ),
        "n_samples": int(len(event)),
        "OS_events": int(event.sum()),
        "OS_censored": int((~event).sum()),
        "candidate_branch_point_estimate": candidate_branch,
        "final_branch": final_branch,
        "bootstrap_status": branch_result.get("bootstrap_status"),
        "reporting_flags": branch_result.get("reporting_flags"),
        "model_selection_performed": False,
        "05d_HOLD_may_change": False,
        "A6_may_reopen": False,
        "GSE21257_outcomes_read": False,
        "GSE39055_outcomes_read": False,
        "secondary_ENDPOINT_values_read": False,
        "authoritative_runner_sha256": sha256_file(F2C_RUNNER),
        "preopening_lock_sha256": sha256_file(PREOPEN_LOCK),
        "opening_audit_sha256": sha256_file(OPENING_AUDIT),
        "parsed_endpoint_sha256": sha256_file(PARSED_ENDPOINT),
        "model_summary_sha256": sha256_file(model_summary_path),
        "branch_result_sha256": sha256_file(branch_result_path),
        "execution_output_hashes": {
            path.name: sha256_file(path)
            for path in sorted(EXEC_DIR.iterdir())
            if path.is_file()
        },
        "post_opening_rule": (
            "No runner/model/source/branch/resampling changes are permitted. "
            "Unexpected results are interpreted through frozen T-A/T-B/T-C/T-D "
            "or T-D/NOT_ASSESSABLE; GSE21257/GSE39055 remain sealed."
        ),
    }

    write_json(
        SUMMARY_JSON,
        summary,
    )

    print()
    print("=" * 120)
    print("05f3 FIRST RESERVED TARGET-OS RESULT")
    print("=" * 120)
    print(f"n={len(event)}, OS events={int(event.sum())}")
    print()
    print("Frozen model summary:")
    display_cols = [
        col
        for col in [
            "model_id",
            "uno_c",
            "ibs",
            "risk_sd",
            "risk_q99_q01",
        ]
        if col in model_summary.columns
    ]
    print(
        model_summary[
            display_cols
        ].to_string(index=False)
    )
    print()
    print(f"Point-estimate candidate branch: {candidate_branch}")
    print(f"Final frozen branch: {final_branch}")
    print(
        f"Bootstrap status: "
        f"{branch_result.get('bootstrap_status', 'UNKNOWN')}"
    )
    print()
    print("Reporting flags:")
    for key, value in sorted(
        (branch_result.get("reporting_flags") or {}).items()
    ):
        print(f"  {key}: {value}")
    print()
    print("Frozen prior states remain:")
    print("  05d architecture selection: HOLD_NO_SAFE_SELECTABLE_ARCHITECTURE")
    print("  A6: STOP_A6_FOR_PAPER6")
    print("  TARGET role: DESCRIPTIVE / NON-CONFIRMATORY")
    print("  GSE21257 outcomes: SEALED")
    print("  GSE39055 outcomes: SEALED")
    print()
    print(f"05f3 summary SHA256: {sha256_file(SUMMARY_JSON)}")
    print("=" * 120)
    print(
        "05f3a: PASS_FIRST_RESERVED_TARGET_OS_FROZEN_DESCRIPTIVE_EVALUATION_COMPLETE"
    )
    print("=" * 120)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 120, file=sys.stderr)
        print(
            "05f3 reserved TARGET-OS frozen execution: FAIL",
            file=sys.stderr,
        )
        print(
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        print("=" * 120, file=sys.stderr)
        raise
