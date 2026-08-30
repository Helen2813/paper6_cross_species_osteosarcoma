#!/usr/bin/env python3
"""
Paper 6 - freeze TARGET clinical schema and evaluation mechanics without
opening TARGET outcome values.

Stage 05f1c follows:
- 05f1a frozen outcome-free representation contract;
- 05f1b exact raw aligned DOG2/TARGET matrix materialization.

Scientific role
---------------
Freeze, before any TARGET survival value is read:

1. exact locked TARGET clinical-file identity;
2. clinical column names (HEADER ONLY);
3. exact non-outcome clinical identifier column and expression<->clinical
   identifier mapping;
4. exact OS endpoint COLUMN mapping and deterministic future value parser;
5. exact repeated 5x20 outer-CV split algorithm/seeds;
6. exact 4-fold inner-CV seed namespace for classical target fitting;
7. exact fold/repeat aggregation rule;
8. exact paired patient-level bootstrap/CI mechanics;
9. exact mechanical T-A/T-B/T-C/T-D uncertainty resolution rules;
10. the next-stage outcome-opening firewall.

This script MAY read:
- TARGET clinical HEADER;
- the selected TARGET clinical identifier column VALUES only.

This script MUST NOT read:
- OS time values;
- OS event/status values;
- any secondary endpoint values;
- any other TARGET clinical values;
- GSE21257/GSE39055 outcomes.

The endpoint parser is frozen here before values are seen. When outcomes are
eventually opened, any value outside the frozen parser grammar causes
FAIL_CLOSED; parser rules may not be broadened after inspection.

No model fitting.
No survival split generation (labels are still closed).
No GPU.
No CLI arguments.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


SCRIPT_VERSION = (
    "05f1c-freeze-target-clinical-schema-and-evaluation-mechanics-v1-no-cli"
)
CONTRACT_VERSION = (
    "paper6-posthold-target-clinical-schema-evaluation-v1"
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "_config"
PAPER4_BASENAME = "paper4_sarcoma_dog"

# ---------------------------------------------------------------------------
# Frozen 05f1a / completed 05f1b.
# ---------------------------------------------------------------------------
F1A_DIR = ROOT / "results" / "human_posthold_descriptive" / "05f1a"
F1A_CONTRACT = F1A_DIR / "outcome_free_TARGET_representation_contract.json"
F1A_SUMMARY = F1A_DIR / "summary.json"
F1A_TARGET_ROSTER = F1A_DIR / "TARGET_expression_sample_roster.tsv"
F1A_MODELS = F1A_DIR / "TARGET_frozen_model_registry.tsv"
F1A_BRANCHES = F1A_DIR / "TARGET_frozen_interpretation_branch_registry.tsv"
F1A_FLAGS = F1A_DIR / "TARGET_frozen_reporting_flags.tsv"

EXPECTED_F1A_CONTRACT_SHA256 = (
    "f0d5296d4df5e8dcb3b677127d2c3b5474791fb2cbaeb207e094b54fb44d1434"
)
EXPECTED_F1A_STATUS = (
    "PASS_OUTCOME_FREE_TARGET_REPRESENTATION_CONTRACT_FROZEN"
)

F1B_DIR = ROOT / "results" / "human_posthold_descriptive" / "05f1b"
F1B_MANIFEST = F1B_DIR / "raw_aligned_matrix_manifest.tsv"
F1B_AUDIT = F1B_DIR / "raw_aligned_value_integrity_audit.tsv"
F1B_SUMMARY = F1B_DIR / "summary.json"
F1B_TARGET_MATRIX = (
    F1B_DIR / "matrices" / "TARGET_OS_raw_aligned_11815genes.npz"
)
F1B_DOG_MATRIX = (
    F1B_DIR / "matrices" / "DOG2_raw_aligned_11815genes.npz"
)

EXPECTED_F1B_STATUS = (
    "PASS_RAW_ALIGNED_DOG2_TARGET_EXPRESSION_MATERIALIZED_OUTCOME_FREE"
)
EXPECTED_F1B_MANIFEST_SHA256 = (
    "d5e267da52955332f845655325de7a66fcf4fbc63fcde9bfac8669c3a5f46b63"
)
EXPECTED_F1B_DOG_MATRIX_SHA256 = (
    "af7b8b65f9485df6a7ba3d0061bc12c0023719e4114d7f10fb3d731e918e3875"
)
EXPECTED_F1B_TARGET_MATRIX_SHA256 = (
    "7aca20d98731a24d90882f2042b1aa7c7aefc0095a250d4ed42c3502cc3c7b19"
)

# ---------------------------------------------------------------------------
# Locked clinical asset.
# ---------------------------------------------------------------------------
UPSTREAM_LOCK = ROOT / "contracts" / "00_upstream_input_lock.json"

# ---------------------------------------------------------------------------
# Outputs.
# ---------------------------------------------------------------------------
OUT_DIR = ROOT / "results" / "human_posthold_descriptive" / "05f1c"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLINICAL_HEADER = OUT_DIR / "TARGET_clinical_header.tsv"
ID_MAPPING = OUT_DIR / "TARGET_expression_to_clinical_id_mapping.tsv"
ENDPOINT_SCHEMA = OUT_DIR / "TARGET_OS_endpoint_schema.json"
RESAMPLING_CONTRACT = OUT_DIR / "TARGET_resampling_and_CI_contract.json"
OUTCOME_OPENING_CONTRACT = OUT_DIR / "TARGET_outcome_opening_contract.json"
SUMMARY_JSON = OUT_DIR / "summary.json"

EXPECTED_TARGET_EXPRESSION_SAMPLES = 88

# ---------------------------------------------------------------------------
# Frozen evaluation constants inherited from textual 05f0 / 05f1a.
# ---------------------------------------------------------------------------
PRIMARY_ENDPOINT = "OS"

OUTER_SPLITS = 5
OUTER_REPEATS = 20
INNER_SPLITS_CLASSICAL = 4
BASE_SEED = 20260830

OUTER_SEED_OFFSET = 0
INNER_SEED_OFFSET = 100000

BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = BASE_SEED + 900000
BOOTSTRAP_MAX_ATTEMPTS = 25000

DESCRIPTIVE_DELTA_C_REFERENCE = 0.02
ORIENTATION_REFERENCE = 0.50

MIN_VALID_REPEATS = 16  # 80% of the frozen 20 repeats.

# ---------------------------------------------------------------------------
# Header-only schema recognition.
#
# These aliases are frozen in source code BEFORE reading the clinical header.
# Matching is case/punctuation insensitive via normalize_header().
# ---------------------------------------------------------------------------
ID_ALIAS_PRIORITY = [
    "targetusi",
    "sampleid",
    "caseid",
    "patientid",
    "participantid",
    "submitterid",
    "bcrpatientbarcode",
    "patient",
    "sample",
]

# Event aliases are ordered from explicit OS event/status fields to the more
# generic vital-status fallback.
OS_EVENT_ALIAS_PRIORITY = [
    "osevent",
    "osstatus",
    "overall survivalevent",
    "overall survivalstatus",
    "overallsurvivalevent",
    "overallsurvivalstatus",
    "os",
    "vitalstatus",
]

OS_TIME_ALIAS_PRIORITY = [
    "ostime",
    "osdays",
    "osmonths",
    "overall survivaltime",
    "overall survivaldays",
    "overall survivalmonths",
    "overallsurvivaltime",
    "overallsurvivaldays",
    "overallsurvivalmonths",
    "survivaltime",
]

RAW_DEATH_TIME_ALIASES = [
    "daystodeath",
    "deathdays",
]

RAW_FOLLOWUP_TIME_ALIASES = [
    "daystolastfollowup",
    "daystolastfollow up",
    "daystolastknownalive",
    "lastfollowupdays",
]

# Predeclared deterministic identifier transforms. Selection uses identifier
# values only, never endpoints, and chooses the FIRST transform yielding a
# unique complete 88-sample mapping.
ID_TRANSFORM_PRIORITY = [
    "UPPER_STRIPPED_EXACT",
    "TARGET_FIRST_THREE_HYPHEN_TOKENS",
]


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
            (
                Path(config_value).expanduser(),
                "_config/paths.local.json",
            )
        )

    candidates.extend(
        [
            (
                ROOT.parent / PAPER4_BASENAME,
                "sibling_repository",
            ),
            (
                Path.home() / "Desktop" / PAPER4_BASENAME,
                "home_desktop_fallback",
            ),
        ]
    )

    checked: List[str] = []

    for candidate, source in candidates:
        resolved = candidate.resolve()
        checked.append(f"{source}: {resolved}")
        if resolved.is_dir():
            return resolved, source

    raise FileNotFoundError(
        "Could not resolve Paper 4 root. Checked:\n  - "
        + "\n  - ".join(checked)
    )


def normalize_header(value: str) -> str:
    # Preserve only alphanumeric content so OS.time, OS_time and OS time match.
    return re.sub(
        r"[^a-z0-9]+",
        "",
        clean(value).lower(),
    )


def normalized_alias(value: str) -> str:
    return normalize_header(value)


def header_alias_map(columns: Sequence[str]) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}

    for column in columns:
        key = normalize_header(column)
        mapping.setdefault(key, []).append(str(column))

    return mapping


def choose_header_column(
    columns: Sequence[str],
    aliases: Sequence[str],
    *,
    role: str,
    required: bool,
) -> Optional[str]:
    mapping = header_alias_map(columns)

    for alias in aliases:
        key = normalized_alias(alias)
        matched = mapping.get(key, [])

        if len(matched) > 1:
            raise RuntimeError(
                f"Clinical header has multiple columns matching {role} "
                f"alias {alias!r}: {matched}"
            )

        if len(matched) == 1:
            return matched[0]

    if required:
        raise RuntimeError(
            f"Could not identify {role} from frozen alias list. "
            f"Clinical columns={list(columns)}"
        )

    return None


def preferred_clinical_asset_role(lock: Dict[str, Any]) -> str:
    assets = lock.get("assets") or {}

    preferred = [
        "target_clinical",
        "target_os_clinical",
        "target_clinical_matched",
        "target_os_clinical_matched",
        "target_survival",
    ]

    for role in preferred:
        if role in assets:
            return role

    candidates = [
        str(role)
        for role in assets
        if (
            "target" in str(role).lower()
            and (
                "clinical" in str(role).lower()
                or "survival" in str(role).lower()
            )
            and "expression" not in str(role).lower()
        )
    ]

    if len(candidates) == 1:
        return candidates[0]

    raise RuntimeError(
        "Could not uniquely identify locked TARGET clinical asset. "
        f"Candidate roles={candidates}; available roles={sorted(assets)}"
    )


def get_locked_asset(
    lock: Dict[str, Any],
    role: str,
    paper4_root: Path,
) -> Tuple[Path, Dict[str, Any]]:
    item = (lock.get("assets") or {}).get(role)

    if not isinstance(item, dict):
        raise RuntimeError(
            f"Upstream lock missing asset role {role!r}."
        )

    rel = clean(item.get("relative_path"))
    expected_hash = clean(item.get("sha256")).lower()

    if not rel or len(expected_hash) != 64:
        raise RuntimeError(
            f"Malformed upstream lock entry for {role!r}."
        )

    path = paper4_root / rel
    require_file(path)

    observed_hash = sha256_file(path).lower()
    if observed_hash != expected_hash:
        raise RuntimeError(
            f"Locked clinical asset changed: {role}; "
            f"expected={expected_hash}, observed={observed_hash}"
        )

    return path, item


def read_clinical_header(path: Path) -> List[str]:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        header = pd.read_csv(
            path,
            nrows=0,
        )
    elif suffix in {".tsv", ".txt"}:
        header = pd.read_csv(
            path,
            sep="\t",
            nrows=0,
        )
    elif suffix in {".xlsx", ".xls"}:
        header = pd.read_excel(
            path,
            nrows=0,
        )
    else:
        raise RuntimeError(
            f"Unsupported clinical file type for header-only read: {suffix}"
        )

    columns = [str(x) for x in header.columns]

    if not columns:
        raise RuntimeError("TARGET clinical header is empty.")

    if len(columns) != len(set(columns)):
        raise RuntimeError(
            "TARGET clinical header contains duplicate raw column names."
        )

    normalized = [normalize_header(x) for x in columns]
    duplicated_norm = (
        pd.Series(normalized)
        .loc[pd.Series(normalized).duplicated()]
        .unique()
        .tolist()
    )
    if duplicated_norm:
        raise RuntimeError(
            "TARGET clinical header has collisions after normalization: "
            f"{duplicated_norm}"
        )

    return columns


def read_single_clinical_id_column(
    path: Path,
    column: str,
) -> List[str]:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        frame = pd.read_csv(
            path,
            usecols=[column],
            dtype=str,
            low_memory=False,
        )
    elif suffix in {".tsv", ".txt"}:
        frame = pd.read_csv(
            path,
            sep="\t",
            usecols=[column],
            dtype=str,
            low_memory=False,
        )
    elif suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(
            path,
            usecols=[column],
            dtype=str,
        )
    else:
        raise RuntimeError(
            f"Unsupported clinical file type for ID-only read: {suffix}"
        )

    values = [
        clean(x)
        for x in frame[column].fillna("")
    ]

    if any(not x for x in values):
        raise RuntimeError(
            f"TARGET clinical identifier column {column!r} contains blank IDs."
        )

    return values


def id_transform(
    value: str,
    transform_name: str,
) -> str:
    text = clean(value).upper()

    if transform_name == "UPPER_STRIPPED_EXACT":
        return text

    if transform_name == "TARGET_FIRST_THREE_HYPHEN_TOKENS":
        parts = text.split("-")

        if len(parts) >= 3 and parts[0] == "TARGET":
            return "-".join(parts[:3])

        return text

    raise RuntimeError(
        f"Unknown frozen ID transform {transform_name!r}."
    )


def freeze_id_mapping(
    expression_ids: Sequence[str],
    clinical_ids: Sequence[str],
) -> Tuple[str, pd.DataFrame]:
    expr_ids = [clean(x) for x in expression_ids]
    clin_ids = [clean(x) for x in clinical_ids]

    successful: List[Tuple[str, List[str], Dict[str, int]]] = []

    for transform in ID_TRANSFORM_PRIORITY:
        expr_keys = [
            id_transform(x, transform)
            for x in expr_ids
        ]
        clin_keys = [
            id_transform(x, transform)
            for x in clin_ids
        ]

        # TARGET expression samples must remain one-to-one under the transform.
        if len(expr_keys) != len(set(expr_keys)):
            continue

        clinical_lookup: Dict[str, int] = {}
        duplicate_clinical_key = False

        for row_index, key in enumerate(clin_keys):
            if key in clinical_lookup:
                duplicate_clinical_key = True
                break
            clinical_lookup[key] = row_index

        if duplicate_clinical_key:
            continue

        if not all(
            key in clinical_lookup
            for key in expr_keys
        ):
            continue

        successful.append(
            (
                transform,
                expr_keys,
                clinical_lookup,
            )
        )

    if not successful:
        raise RuntimeError(
            "None of the frozen non-outcome identifier transforms produced "
            "a complete unique mapping of all TARGET expression samples into "
            "the locked clinical table."
        )

    # Choose the FIRST successful predeclared transform.
    transform, expr_keys, clinical_lookup = successful[0]

    rows = []

    for expression_index, (
        expression_id,
        case_key,
    ) in enumerate(
        zip(expr_ids, expr_keys)
    ):
        clinical_row_index = int(
            clinical_lookup[case_key]
        )
        clinical_id = clin_ids[
            clinical_row_index
        ]

        rows.append({
            "expression_sample_index": expression_index,
            "expression_sample_id": expression_id,
            "frozen_case_key": case_key,
            "clinical_row_index": clinical_row_index,
            "clinical_identifier": clinical_id,
            "identifier_transform": transform,
            "outcome_values_read": False,
        })

    mapping = pd.DataFrame(rows)

    if len(mapping) != EXPECTED_TARGET_EXPRESSION_SAMPLES:
        raise RuntimeError(
            f"Identifier mapping has {len(mapping)} rows; expected "
            f"{EXPECTED_TARGET_EXPRESSION_SAMPLES}."
        )

    if mapping["clinical_row_index"].duplicated().any():
        raise RuntimeError(
            "Expression-to-clinical mapping is not one-to-one."
        )

    return transform, mapping


def detect_os_schema(
    columns: Sequence[str],
) -> Dict[str, Any]:
    event_col = choose_header_column(
        columns,
        OS_EVENT_ALIAS_PRIORITY,
        role="OS event/status column",
        required=False,
    )
    time_col = choose_header_column(
        columns,
        OS_TIME_ALIAS_PRIORITY,
        role="OS time column",
        required=False,
    )

    # Preferred schema: explicit OS event/status + explicit OS time.
    if event_col is not None and time_col is not None:
        return {
            "schema_type": "PRECOMPUTED_OS_EVENT_PLUS_TIME",
            "event_column": event_col,
            "time_column": time_col,
            "death_time_column": None,
            "followup_time_column": None,
            "time_unit_policy": "NATIVE_NUMERIC_UNIT_NO_CONVERSION",
        }

    # Fallback schema: vital status plus event-dependent death/follow-up time.
    if event_col is None:
        event_col = choose_header_column(
            columns,
            ["vitalstatus"],
            role="vital-status column",
            required=False,
        )

    death_col = choose_header_column(
        columns,
        RAW_DEATH_TIME_ALIASES,
        role="days-to-death column",
        required=False,
    )
    followup_col = choose_header_column(
        columns,
        RAW_FOLLOWUP_TIME_ALIASES,
        role="days-to-last-followup column",
        required=False,
    )

    if (
        event_col is not None
        and death_col is not None
        and followup_col is not None
    ):
        return {
            "schema_type": "VITAL_STATUS_PLUS_EVENT_DEPENDENT_TIME",
            "event_column": event_col,
            "time_column": None,
            "death_time_column": death_col,
            "followup_time_column": followup_col,
            "time_unit_policy": "NATIVE_NUMERIC_UNIT_NO_CONVERSION",
        }

    raise RuntimeError(
        "Could not freeze an unambiguous OS endpoint schema from header only. "
        f"event_candidate={event_col!r}, time_candidate={time_col!r}, "
        f"death_candidate={death_col!r}, followup_candidate={followup_col!r}. "
        f"Clinical columns={list(columns)}"
    )


def frozen_event_parser_contract() -> Dict[str, Any]:
    return {
        "numeric_encoding": {
            "0": "CENSORED",
            "1": "EVENT",
        },
        "accepted_event_strings_case_insensitive": [
            "dead",
            "deceased",
            "died",
            "death",
            "event",
            "yes",
            "true",
            "1",
            "1:deceased",
            "1:dead",
        ],
        "accepted_censored_strings_case_insensitive": [
            "alive",
            "living",
            "censored",
            "no",
            "false",
            "0",
            "0:living",
            "0:alive",
        ],
        "whitespace_trimmed": True,
        "unrecognized_value_action": "FAIL_CLOSED",
        "missing_event_action": "FAIL_CLOSED",
    }


def frozen_time_parser_contract(
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "schema_type": schema["schema_type"],
        "numeric_parse": "FLOAT64_STRICT",
        "required": "FINITE_AND_STRICTLY_POSITIVE",
        "unit_conversion": "NONE_USE_NATIVE_NUMERIC_UNIT",
        "precomputed_schema": (
            "use time_column unchanged"
            if schema["schema_type"]
            == "PRECOMPUTED_OS_EVENT_PLUS_TIME"
            else None
        ),
        "event_dependent_schema": (
            "if event=1 use death_time_column; if event=0 use "
            "followup_time_column"
            if schema["schema_type"]
            == "VITAL_STATUS_PLUS_EVENT_DEPENDENT_TIME"
            else None
        ),
        "missing_required_time_action": "FAIL_CLOSED",
        "nonfinite_time_action": "FAIL_CLOSED",
        "nonpositive_time_action": "FAIL_CLOSED",
    }


def verify_upstream() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    for path in [
        F1A_CONTRACT,
        F1A_SUMMARY,
        F1A_TARGET_ROSTER,
        F1A_MODELS,
        F1A_BRANCHES,
        F1A_FLAGS,
        F1B_MANIFEST,
        F1B_AUDIT,
        F1B_SUMMARY,
        F1B_TARGET_MATRIX,
        F1B_DOG_MATRIX,
    ]:
        require_file(path)

    if sha256_file(F1A_CONTRACT) != EXPECTED_F1A_CONTRACT_SHA256:
        raise RuntimeError(
            "05f1a contract differs from frozen PASS hash."
        )

    f1a = read_json(F1A_SUMMARY)
    if clean(f1a.get("scientific_status")) != EXPECTED_F1A_STATUS:
        raise RuntimeError(
            "05f1a is not in expected frozen PASS state."
        )

    if sha256_file(F1B_MANIFEST) != EXPECTED_F1B_MANIFEST_SHA256:
        raise RuntimeError(
            "05f1b raw matrix manifest differs from completed PASS run."
        )

    if sha256_file(F1B_DOG_MATRIX) != EXPECTED_F1B_DOG_MATRIX_SHA256:
        raise RuntimeError(
            "05f1b DOG2 matrix differs from completed PASS run."
        )

    if sha256_file(F1B_TARGET_MATRIX) != EXPECTED_F1B_TARGET_MATRIX_SHA256:
        raise RuntimeError(
            "05f1b TARGET matrix differs from completed PASS run."
        )

    f1b = read_json(F1B_SUMMARY)

    if clean(f1b.get("scientific_status")) != EXPECTED_F1B_STATUS:
        raise RuntimeError(
            "05f1b is not in expected PASS state."
        )

    if f1b.get("TARGET_outcomes_read") is not False:
        raise RuntimeError(
            "05f1b provenance unexpectedly indicates TARGET outcome access."
        )

    if f1b.get("TARGET_clinical_file_read") is not False:
        raise RuntimeError(
            "05f1b provenance unexpectedly indicates TARGET clinical access."
        )

    return read_json(F1A_CONTRACT), f1b


def main() -> None:
    started = now_utc()

    print("=" * 120)
    print("Paper 6 - freeze TARGET clinical schema and evaluation mechanics")
    print("=" * 120)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Contract version: {CONTRACT_VERSION}")
    print()
    print("Safety / scope:")
    print("  05f1a frozen representation verified: YES")
    print("  05f1b raw matrices verified: YES")
    print("  TARGET clinical HEADER read: YES")
    print("  TARGET clinical identifier column values read: YES [non-outcome only]")
    print("  TARGET OS/event values read: NO")
    print("  TARGET secondary endpoint values read: NO")
    print("  Other TARGET clinical values read: NO")
    print("  GSE21257/GSE39055 outcomes read: NO")
    print("  Survival split generation: NO")
    print("  Model fitting: NO")
    print("  GPU execution: NO")
    print()

    f1a_contract, f1b_summary = verify_upstream()

    require_file(UPSTREAM_LOCK)
    upstream_lock = read_json(UPSTREAM_LOCK)

    paper4_root, paper4_resolution = resolve_paper4_root()

    clinical_role = preferred_clinical_asset_role(
        upstream_lock
    )
    clinical_path, clinical_asset = get_locked_asset(
        upstream_lock,
        clinical_role,
        paper4_root,
    )

    columns = read_clinical_header(
        clinical_path
    )

    # Write ordered header immediately; header names are not outcome values.
    header_df = pd.DataFrame(
        {
            "column_index": np.arange(
                len(columns),
                dtype=int,
            ),
            "column_name": columns,
            "normalized_name": [
                normalize_header(x)
                for x in columns
            ],
            "values_read_in_05f1c": False,
        }
    )

    # Exact identifier column is chosen from the frozen alias priority.
    id_column = choose_header_column(
        columns,
        ID_ALIAS_PRIORITY,
        role="TARGET clinical identifier column",
        required=True,
    )

    # OS schema is inferred from names only.
    os_schema = detect_os_schema(
        columns
    )

    # Mark only the chosen identifier column as value-read.
    header_df.loc[
        header_df["column_name"] == id_column,
        "values_read_in_05f1c",
    ] = True

    header_df.to_csv(
        CLINICAL_HEADER,
        sep="\t",
        index=False,
    )

    clinical_ids = read_single_clinical_id_column(
        clinical_path,
        id_column,
    )

    expression_roster = pd.read_csv(
        F1A_TARGET_ROSTER,
        sep="\t",
        dtype=str,
        low_memory=False,
    ).fillna("")

    expression_ids = (
        expression_roster.sort_values(
            "sample_index"
        )["sample_id"]
        .astype(str)
        .map(clean)
        .tolist()
    )

    if len(expression_ids) != EXPECTED_TARGET_EXPRESSION_SAMPLES:
        raise RuntimeError(
            "Frozen TARGET expression roster is not 88 samples."
        )

    id_transform_name, id_mapping = freeze_id_mapping(
        expression_ids,
        clinical_ids,
    )

    id_mapping.to_csv(
        ID_MAPPING,
        sep="\t",
        index=False,
    )

    endpoint_schema_payload = {
        "script_version": SCRIPT_VERSION,
        "contract_version": CONTRACT_VERSION,
        "status": "FROZEN_BEFORE_ENDPOINT_VALUE_READ",
        "created_utc": now_utc(),

        "primary_endpoint": PRIMARY_ENDPOINT,
        "clinical_asset_role": clinical_role,
        "clinical_asset_relative_path": clean(
            clinical_asset.get("relative_path")
        ),
        "clinical_asset_sha256": clean(
            clinical_asset.get("sha256")
        ),
        "identifier_column": id_column,
        "identifier_transform": id_transform_name,
        "expression_samples_mapped": int(
            len(id_mapping)
        ),
        "expression_samples_unmapped": 0,

        "OS_schema": os_schema,
        "event_parser": frozen_event_parser_contract(),
        "time_parser": frozen_time_parser_contract(
            os_schema
        ),

        "endpoint_value_access_in_05f1c": False,
        "secondary_endpoint_value_access_in_05f1c": False,

        "post_opening_parser_change_allowed": False,
        "unrecognized_event_value_action": "FAIL_CLOSED",
        "invalid_time_action": "FAIL_CLOSED",
        "post_outcome_ID_mapping_change_allowed": False,
    }

    write_json(
        ENDPOINT_SCHEMA,
        endpoint_schema_payload,
    )

    # ------------------------------------------------------------------
    # Exact repeated-CV + bootstrap mechanics frozen BEFORE labels are read.
    # ------------------------------------------------------------------
    resampling_payload = {
        "script_version": SCRIPT_VERSION,
        "contract_version": CONTRACT_VERSION,
        "status": "FROZEN_BEFORE_TARGET_ENDPOINT_VALUE_READ",
        "created_utc": now_utc(),

        "primary_endpoint": PRIMARY_ENDPOINT,

        "outer_CV": {
            "algorithm": "sklearn.model_selection.StratifiedKFold",
            "n_splits": OUTER_SPLITS,
            "n_repeats": OUTER_REPEATS,
            "shuffle": True,
            "stratification_variable": "OS_event",
            "repeat_seed_formula": (
                f"{BASE_SEED} + repeat_index"
            ),
            "repeat_index_range": "0..19",
            "same_splits_all_models": True,
            "patient_level_split": True,
            "split_generation_after_outcome_opening": True,
            "split_generation_rule_change_after_outcomes": False,
        },

        "inner_CV_classical": {
            "algorithm": "sklearn.model_selection.StratifiedKFold",
            "n_splits": INNER_SPLITS_CLASSICAL,
            "shuffle": True,
            "stratification_variable": "OS_event within outer training only",
            "seed_formula": (
                f"{BASE_SEED + INNER_SEED_OFFSET} "
                "+ outer_repeat_index*100 + outer_fold_index"
            ),
            "target_outer_test_used": False,
            "exact_candidate_hyperparameter_grids": (
                "MUST_BE_FROZEN_IN_MODEL_IMPLEMENTATION_STAGE_BEFORE "
                "FIRST_TARGET_ENDPOINT_VALUE_READ"
            ),
        },

        "fold_metric_definition": {
            "Uno_C": (
                "Compute on each outer-test fold using the corresponding "
                "outer-training survival distribution for IPCW censoring weights."
            ),
            "IBS": (
                "Compute on each outer-test fold using corresponding "
                "outer-training survival distribution and training-fitted "
                "risk calibration transform exactly as frozen in the "
                "future model/evaluation implementation."
            ),
            "risk_SD": (
                "Population SD (ddof=0) of held-out risk score within "
                "each outer-test fold."
            ),
            "risk_q99_q01": (
                "99th minus 1st percentile of held-out risk within each "
                "outer-test fold."
            ),
        },

        "point_estimate_aggregation": {
            "fold_to_repeat": (
                "Sample-count-weighted mean of the 5 outer-fold metrics "
                "within each repeat."
            ),
            "repeat_to_final": (
                "Arithmetic mean across valid repeat-level estimates."
            ),
            "minimum_valid_repeats": MIN_VALID_REPEATS,
            "if_fewer_than_min_valid_repeats": (
                "METRIC_NOT_ASSESSABLE; any branch requiring that metric -> T-D"
            ),
            "paired_model_contrast": (
                "Within identical outer fold/repeat, treatment metric minus "
                "control metric; aggregate with the same fold/repeat rule."
            ),
        },

        "patient_clustered_bootstrap": {
            "draws_required": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "maximum_draw_attempts": BOOTSTRAP_MAX_ATTEMPTS,
            "unit": "unique TARGET patient/case key",
            "stratification": "OS_event",
            "resample_with_replacement_within_event_stratum": True,
            "same_patient_multiplicity_all_models": True,
            "same_patient_multiplicity_all_20_outer_test_appearances": True,
            "model_refitting_inside_bootstrap": False,
            "outer_split_regeneration_inside_bootstrap": False,
            "training_survival_distribution_inside_bootstrap": (
                "Keep the original frozen outer-training survival distribution "
                "for each fold; bootstrap only the already-held-out patient "
                "observations by patient multiplicity."
            ),
            "test_duplicate_semantics": (
                "Patient multiplicity is represented by literal repeated copies "
                "of that held-out patient's outcome/risk row when evaluating "
                "the fixed fold prediction."
            ),
            "fold_metric_recomputed_each_draw": True,
            "fold_to_repeat_aggregation_same_as_point_estimate": True,
            "repeat_to_final_aggregation_same_as_point_estimate": True,
            "paired_delta_C": (
                "Compute treatment-control difference from the same bootstrap "
                "draw and same patient multiplicities."
            ),
            "CI": "percentile 2.5th and 97.5th percentiles of valid bootstrap draws",
            "invalid_draw": (
                "A draw is invalid if a required fold metric cannot be evaluated "
                "under frozen metric-support rules."
            ),
            "valid_draw_collection": (
                "Continue deterministic RNG stream until 5000 valid draws or "
                "25000 total attempts."
            ),
            "if_5000_valid_draws_not_obtained": (
                "CI_NOT_ASSESSABLE; branch-defining comparison -> T-D"
            ),
            "interpretation": (
                "Conditional uncertainty of fixed repeated-CV predictions; "
                "not full model-training-process uncertainty."
            ),
        },

        "mechanical_branch_uncertainty_rules": {
            "point_estimate_candidate_branch_first": True,
            "orientation_unresolved": (
                "95% bootstrap CI for C(N0) includes 0.50."
            ),
            "delta_unresolved": (
                "95% bootstrap CI for a branch-required deltaC contains BOTH "
                "-0.02 and +0.02."
            ),
            "T_D_override": (
                "If any condition required by the point-estimate candidate "
                "T-A/T-B/T-C branch is unresolved, final branch=T-D."
            ),
            "no_unique_point_estimate_branch": "T-D",
            "subjective_override": False,
            "T_D_is_valid_expected_result": True,
        },

        "reporting_flags": {
            "branch_assignment_arm": "NEURAL_N0_N1_WITH_T0_REFERENCE",
            "classical_neural_zero_shot_direction_discordance_reported": True,
            "flags_change_branch": False,
        },

        "endpoint_firewall": {
            "OS_primary": True,
            "secondary_endpoint_upgrade_after_opening": False,
            "resampling_change_after_opening": False,
            "CI_method_change_after_opening": False,
            "fold_exclusion_after_opening": False,
            "patient_QC_exclusion_after_opening": False,
        },
    }

    write_json(
        RESAMPLING_CONTRACT,
        resampling_payload,
    )

    # ------------------------------------------------------------------
    # Outcome-opening contract. This does NOT itself authorize immediate
    # endpoint reading: exact model implementation still must be frozen.
    # ------------------------------------------------------------------
    opening_payload = {
        "script_version": SCRIPT_VERSION,
        "contract_version": CONTRACT_VERSION,
        "status": "PRE_OUTCOME_FIREWALL_INCOMPLETE_MODEL_IMPLEMENTATION_PENDING",
        "created_utc": now_utc(),

        "completed_before_outcome_opening": [
            "05f1a outcome-free representation freeze",
            "05f1b raw aligned expression materialization",
            "05f1c clinical header/identifier/endpoint-schema freeze",
            "05f1c repeated-CV/bootstrap/branch-uncertainty freeze",
        ],

        "still_required_before_first_TARGET_endpoint_value_read": [
            "freeze exact T0-T2/N0-N5 implementation",
            "freeze exact classical hyperparameter grids",
            "freeze exact neural real-data architecture/training implementation",
            "freeze source DOG2 model artifacts or deterministic source-fitting contract",
            "freeze exact Uno-C tau/time-support implementation",
            "freeze exact IBS prediction-survival conversion/calibration implementation",
            "freeze exact fold-safe Hallmark implementation code",
            "freeze output/reporting schema",
        ],

        "TARGET_endpoint_values_may_be_read_now": False,

        "next_stage": (
            "05f2a freeze and validate exact TARGET model/evaluation "
            "implementation and source-model artifacts while TARGET endpoint "
            "values remain closed."
        ),

        "forbidden_before_model_freeze": [
            "read TARGET OS event values",
            "read TARGET OS time values",
            "generate OS-stratified TARGET splits",
            "fit any TARGET-outcome model",
        ],

        "permanently_forbidden_after_outcome_opening": [
            "change clinical ID mapping",
            "change OS endpoint columns/parser",
            "change resampling/bootstrap/CI mechanics",
            "change OS primary endpoint",
            "change 05f0 branches",
            "add sign-orientation selector",
            "add real-data A3 prior",
            "add new gate hardening branch",
            "drop samples/folds/repeats based on observed outcomes/model behavior",
            "open GSE21257/GSE39055 to explain TARGET",
        ],
    }

    write_json(
        OUTCOME_OPENING_CONTRACT,
        opening_payload,
    )

    summary = {
        "script_version": SCRIPT_VERSION,
        "contract_version": CONTRACT_VERSION,
        "status": "PASS",
        "scientific_status": (
            "PASS_TARGET_CLINICAL_SCHEMA_AND_EVALUATION_MECHANICS_FROZEN_OUTCOME_CLOSED"
        ),
        "run_started_utc": started,
        "run_finished_utc": now_utc(),

        "05f1a_contract_sha256": EXPECTED_F1A_CONTRACT_SHA256,
        "05f1b_manifest_sha256": EXPECTED_F1B_MANIFEST_SHA256,
        "05f1b_TARGET_matrix_sha256": EXPECTED_F1B_TARGET_MATRIX_SHA256,

        "clinical_asset_role": clinical_role,
        "clinical_asset_relative_path": clean(
            clinical_asset.get("relative_path")
        ),
        "clinical_asset_sha256": clean(
            clinical_asset.get("sha256")
        ),
        "paper4_root_resolution_source": paper4_resolution,
        "absolute_paper4_path_recorded": False,

        "clinical_column_count": len(columns),
        "identifier_column": id_column,
        "identifier_transform": id_transform_name,
        "TARGET_expression_samples_mapped_to_unique_clinical_rows": (
            len(id_mapping)
        ),

        "OS_schema_type": os_schema["schema_type"],
        "OS_event_column": os_schema["event_column"],
        "OS_time_column": os_schema["time_column"],
        "OS_death_time_column": os_schema[
            "death_time_column"
        ],
        "OS_followup_time_column": os_schema[
            "followup_time_column"
        ],

        "outer_CV": f"{OUTER_SPLITS} folds x {OUTER_REPEATS} repeats",
        "inner_classical_CV": f"{INNER_SPLITS_CLASSICAL} folds",
        "patient_bootstraps": BOOTSTRAP_DRAWS,
        "minimum_valid_repeats": MIN_VALID_REPEATS,

        "TARGET_clinical_header_read": True,
        "TARGET_clinical_identifier_values_read": True,
        "TARGET_OS_event_values_read": False,
        "TARGET_OS_time_values_read": False,
        "TARGET_secondary_endpoint_values_read": False,
        "GSE21257_outcomes_read": False,
        "GSE39055_outcomes_read": False,
        "survival_splits_generated": False,
        "model_fitting": False,
        "GPU_execution": False,

        "TARGET_endpoint_values_may_be_read_now": False,

        "artifact_hashes": {
            "TARGET_clinical_header.tsv": sha256_file(
                CLINICAL_HEADER
            ),
            "TARGET_expression_to_clinical_id_mapping.tsv": sha256_file(
                ID_MAPPING
            ),
            "TARGET_OS_endpoint_schema.json": sha256_file(
                ENDPOINT_SCHEMA
            ),
            "TARGET_resampling_and_CI_contract.json": sha256_file(
                RESAMPLING_CONTRACT
            ),
            "TARGET_outcome_opening_contract.json": sha256_file(
                OUTCOME_OPENING_CONTRACT
            ),
        },

        "next": opening_payload["next_stage"],
    }

    write_json(
        SUMMARY_JSON,
        summary,
    )

    print("=" * 120)
    print("05f1c TARGET CLINICAL-SCHEMA / EVALUATION FREEZE SUMMARY")
    print("=" * 120)
    print(
        f"Locked clinical asset role: {clinical_role}"
    )
    print(
        f"Clinical columns observed: {len(columns)} [HEADER ONLY]"
    )
    print(
        f"Clinical identifier column: {id_column}"
    )
    print(
        f"Identifier transform: {id_transform_name}"
    )
    print(
        f"TARGET expression samples mapped: "
        f"{len(id_mapping)}/{EXPECTED_TARGET_EXPRESSION_SAMPLES}"
    )
    print()
    print("Frozen primary OS schema:")
    print(
        f"  schema type: {os_schema['schema_type']}"
    )
    print(
        f"  event/status column: {os_schema['event_column']}"
    )
    if os_schema["time_column"] is not None:
        print(
            f"  OS time column: {os_schema['time_column']}"
        )
    else:
        print(
            f"  death time column: {os_schema['death_time_column']}"
        )
        print(
            f"  follow-up time column: {os_schema['followup_time_column']}"
        )
    print("  endpoint VALUES read: NO")
    print()
    print("Frozen evaluation mechanics:")
    print(
        f"  outer CV: {OUTER_SPLITS} folds x {OUTER_REPEATS} repeats"
    )
    print(
        f"  inner classical CV: {INNER_SPLITS_CLASSICAL} folds"
    )
    print(
        f"  patient-level paired bootstrap: {BOOTSTRAP_DRAWS:,}"
    )
    print(
        "  same patient multiplicity across all models/repeats: YES"
    )
    print(
        "  bootstrap model refit: NO"
    )
    print(
        "  T-D uncertainty override mechanical: YES"
    )
    print()
    print("Outcome firewall:")
    print("  TARGET OS event values read: NO")
    print("  TARGET OS time values read: NO")
    print("  secondary endpoint values read: NO")
    print("  survival splits generated: NO")
    print("  model fitting: NO")
    print("  GSE21257/GSE39055 outcomes read: NO")
    print()
    print("TARGET endpoint values may be read now: NO")
    print(
        "Reason: exact T0-T2/N0-N5 model/evaluation implementation "
        "must still be frozen."
    )
    print()
    print(
        f"Endpoint schema SHA256: {sha256_file(ENDPOINT_SCHEMA)}"
    )
    print(
        f"Resampling/CI contract SHA256: "
        f"{sha256_file(RESAMPLING_CONTRACT)}"
    )
    print()
    print("Next:")
    print(
        "  05f2a freeze exact human model/evaluation implementation and "
        "source-model artifacts while TARGET endpoints remain CLOSED."
    )
    print("=" * 120)
    print(
        "05f1c: PASS_TARGET_CLINICAL_SCHEMA_AND_EVALUATION_MECHANICS_"
        "FROZEN_OUTCOME_CLOSED"
    )
    print("=" * 120)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 120, file=sys.stderr)
        print(
            "05f1c TARGET clinical schema/evaluation freeze: FAIL",
            file=sys.stderr,
        )
        print(
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        print("=" * 120, file=sys.stderr)
        raise
