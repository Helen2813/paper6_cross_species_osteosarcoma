#!/usr/bin/env python3
"""
Paper 6 - post-opening TARGET-OS complete-case reconciliation audit.

CONTEXT
-------
TARGET-OS endpoint values were first opened by failed 05f3a. That run stopped
immediately because at least one os_event value was missing; no TARGET model
was fitted.

After the failure, the pre-existing Paper-4 endpoint-preparation provenance was
reviewed. It shows that BEFORE Paper-6:
- script 22 intentionally encoded os_event=NaN when vital status was neither
  recognized Dead nor Alive;
- the preparation summary recorded 88 expression/clinical cases but only
  86 OS-complete cases;
- script 23 analyzed TARGET survival using complete primary OS rows;
- the existing Paper-4 TARGET primary result used n=86 with 29 events.

Therefore this stage does NOT invent an event value, impute an outcome, or
choose cases based on predictions. It performs a bounded post-opening
technical reconciliation of Paper-6 endpoint eligibility to the already-
existing upstream complete-case policy.

READS
-----
TARGET:
- case_submitter_id
- os_event
- os_time_days
ONLY. No vital_status, death/follow-up, secondary endpoints, covariates, etc.

Paper-4 pre-existing provenance:
- human_validation_cohort_preparation_summary.csv
- TARGET_OS_primary_frozen_program_validation.csv
- scripts 22 and 23 source code

DOES NOT
--------
- fit a TARGET model;
- generate CV splits;
- read model predictions;
- assign T-A/T-B/T-C/T-D;
- read GSE21257/GSE39055 outcomes;
- fill missing outcomes;
- drop a case based on model/QC behavior.

No CLI arguments.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


SCRIPT_VERSION = (
    "05f3b-audit-target-os-preexisting-complete-case-policy-v1-no-cli"
)
AUDIT_VERSION = (
    "paper6-postopening-target-os-complete-case-reconciliation-v1"
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "_config"
PAPER4_BASENAME = "paper4_sarcoma_dog"

# ---------------------------------------------------------------------------
# Paper-6 frozen assets.
# ---------------------------------------------------------------------------
F1B_DIR = ROOT / "results" / "human_posthold_descriptive" / "05f1b"
TARGET_MATRIX = (
    F1B_DIR / "matrices" / "TARGET_OS_raw_aligned_11815genes.npz"
)
EXPECTED_TARGET_MATRIX_SHA256 = (
    "7aca20d98731a24d90882f2042b1aa7c7aefc0095a250d4ed42c3502cc3c7b19"
)

F1C_DIR = ROOT / "results" / "human_posthold_descriptive" / "05f1c_v2"
F1C_ENDPOINT_SCHEMA = F1C_DIR / "TARGET_OS_endpoint_schema.json"
F1C_ID_MAPPING = F1C_DIR / "TARGET_expression_to_clinical_id_mapping.tsv"

EXPECTED_F1C_ENDPOINT_SCHEMA_SHA256 = (
    "6c827bb13d3a44a651d012b5a51a0b96ffb45682cd73ab545da2cb9edda79ce2"
)

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

# Failed 05f3a wrote this BEFORE opening outcomes, then failed during parser.
F3A_DIR = ROOT / "results" / "human_posthold_descriptive" / "05f3a"
F3A_PREOPEN = F3A_DIR / "PREOPENING_LOCK.json"
F3A_OPENING_AUDIT = F3A_DIR / "FIRST_TARGET_OS_OPENING_AUDIT.json"

EXPECTED_F3A_PREOPEN_SHA256 = (
    "b13effe799ea6b357e447a47a40ac985c3446eba0845a325dd6051e1dedd7615"
)

UPSTREAM_LOCK = ROOT / "contracts" / "00_upstream_input_lock.json"

EXPECTED_CLINICAL_ROLE = "target_clinical"
EXPECTED_CLINICAL_SHA256 = (
    "c89d6718bac4b9d2b3e880302cb5f48501814ee5a57fb609a174bdd505476793"
)

EXPECTED_ID_COLUMN = "case_submitter_id"
EXPECTED_EVENT_COLUMN = "os_event"
EXPECTED_TIME_COLUMN = "os_time_days"

# ---------------------------------------------------------------------------
# Pre-existing Paper-4 endpoint provenance.
# These values were produced before the current Paper-6 TARGET opening.
# ---------------------------------------------------------------------------
EXPECTED_TOTAL_EXPRESSION_N = 88
EXPECTED_TOTAL_CLINICAL_N = 88
EXPECTED_OS_COMPLETE_N = 86
EXPECTED_OS_EVENTS = 29
EXPECTED_OS_CENSORED = 57
EXPECTED_OS_INCOMPLETE_N = 2

P4_PREP_SCRIPT_REL = Path("scripts") / "22_prepare_human_osteosarcoma_cohorts.py"
P4_VALID_SCRIPT_REL = Path("scripts") / "23_external_human_validation.py"
P4_PREP_SUMMARY_REL = (
    Path("results") / "tables" / "human_validation_cohort_preparation_summary.csv"
)
P4_TARGET_PRIMARY_REL = (
    Path("results") / "tables" / "TARGET_OS_primary_frozen_program_validation.csv"
)

EXPECTED_P4_PREP_SCRIPT_VERSION = "22-human-cohort-preparation-v1"
EXPECTED_P4_VALID_SCRIPT_VERSION = "23-human-external-validation-v1"

# ---------------------------------------------------------------------------
# Outputs.
# ---------------------------------------------------------------------------
OUT_DIR = ROOT / "results" / "human_posthold_descriptive" / "05f3b"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MISSINGNESS_AUDIT = OUT_DIR / "TARGET_OS_endpoint_completeness_audit.tsv"
COMPLETE_ROSTER = OUT_DIR / "TARGET_OS_complete_primary_endpoint_roster.tsv"
INCOMPLETE_ROSTER = OUT_DIR / "TARGET_OS_incomplete_primary_endpoint_roster.tsv"
RECONCILIATION_JSON = OUT_DIR / "postopening_complete_case_reconciliation.json"
SUMMARY_JSON = OUT_DIR / "summary.json"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def require_file(path: Path) -> Path:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Required file missing: {path}")
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


def script_version_from_ast(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "SCRIPT_VERSION":
                value = ast.literal_eval(node.value)
                if isinstance(value, str):
                    return value

    raise RuntimeError(f"{path.name}: SCRIPT_VERSION not found.")


def function_source(path: Path, function_name: str) -> str:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    tree = ast.parse(text, filename=str(path))

    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"{path.name}: expected one function {function_name!r}, "
            f"observed {len(matches)}."
        )

    node = matches[0]
    return "".join(lines[node.lineno - 1 : node.end_lineno])


def get_locked_clinical(
    paper4_root: Path,
) -> Path:
    require_file(UPSTREAM_LOCK)
    lock = read_json(UPSTREAM_LOCK)

    item = (lock.get("assets") or {}).get(EXPECTED_CLINICAL_ROLE)
    if not isinstance(item, dict):
        raise RuntimeError("Locked target_clinical role missing.")

    rel = clean(item.get("relative_path"))
    expected = clean(item.get("sha256")).lower()

    if expected != EXPECTED_CLINICAL_SHA256:
        raise RuntimeError(
            "Upstream lock TARGET clinical SHA differs from frozen value."
        )

    path = paper4_root / rel
    require_file(path)

    if sha256_file(path) != EXPECTED_CLINICAL_SHA256:
        raise RuntimeError(
            "TARGET clinical file differs from locked SHA."
        )

    return path


def parse_nonmissing_event(
    raw: Any,
    parser: Dict[str, Any],
) -> Tuple[bool, int | None, str]:
    text = str(raw).strip()

    if text == "":
        return False, None, "MISSING"

    lower = text.lower()

    event_tokens = {
        clean(x).lower()
        for x in parser.get(
            "accepted_event_strings_case_insensitive",
            []
        )
    }
    censored_tokens = {
        clean(x).lower()
        for x in parser.get(
            "accepted_censored_strings_case_insensitive",
            []
        )
    }

    if lower in event_tokens:
        return True, 1, "PARSED_EVENT"

    if lower in censored_tokens:
        return True, 0, "PARSED_CENSORED"

    try:
        numeric = float(text)
    except Exception:
        numeric = float("nan")

    if np.isfinite(numeric) and numeric == 1.0:
        return True, 1, "PARSED_NUMERIC_EVENT"

    if np.isfinite(numeric) and numeric == 0.0:
        return True, 0, "PARSED_NUMERIC_CENSORED"

    # A NON-BLANK unexpected value is not treated as missing.
    return False, None, f"INVALID_NONBLANK:{text}"


def parse_nonmissing_time(
    raw: Any,
) -> Tuple[bool, float | None, str]:
    text = str(raw).strip()

    if text == "":
        return False, None, "MISSING"

    try:
        value = float(text)
    except Exception:
        return False, None, f"INVALID_NONNUMERIC:{text}"

    if not np.isfinite(value):
        return False, None, f"INVALID_NONFINITE:{text}"

    if value <= 0:
        return False, None, f"INVALID_NONPOSITIVE:{value}"

    return True, float(value), "PARSED"


def main() -> None:
    started = now_utc()

    print("=" * 120)
    print("Paper 6 - post-opening TARGET-OS complete-case reconciliation audit")
    print("=" * 120)
    print(f"Script version: {SCRIPT_VERSION}")
    print()
    print("Status:")
    print("  TARGET outcomes already opened by failed 05f3a: YES")
    print("  failed 05f3a TARGET models fit: NO")
    print("  this stage is post-opening technical reconciliation: YES")
    print()
    print("Read scope:")
    print("  case_submitter_id: YES")
    print("  os_event: YES [already-opened primary endpoint]")
    print("  os_time_days: YES [already-opened primary endpoint]")
    print("  vital_status: NO")
    print("  days_to_death/follow_up: NO")
    print("  secondary endpoint/covariates: NO")
    print("  model predictions: NO")
    print("  GSE21257/GSE39055 outcomes: NO")
    print()
    print("Model activity:")
    print("  TARGET model fitting: NO")
    print("  split generation: NO")
    print("  branch assignment: NO")
    print("  imputation of outcome: NO")
    print()

    # ------------------------------------------------------------------
    # Verify immutable Paper-6 state and failed-opening provenance.
    # ------------------------------------------------------------------
    for path in [
        TARGET_MATRIX,
        F1C_ENDPOINT_SCHEMA,
        F1C_ID_MAPPING,
        F2C_RUNNER,
        F3A_PREOPEN,
    ]:
        require_file(path)

    if sha256_file(TARGET_MATRIX) != EXPECTED_TARGET_MATRIX_SHA256:
        raise RuntimeError("TARGET matrix hash changed.")

    if sha256_file(F1C_ENDPOINT_SCHEMA) != EXPECTED_F1C_ENDPOINT_SCHEMA_SHA256:
        raise RuntimeError("05f1c endpoint schema hash changed.")

    if sha256_file(F2C_RUNNER) != EXPECTED_F2C_RUNNER_SHA256:
        raise RuntimeError("05f2c runner hash changed.")

    if sha256_file(F3A_PREOPEN) != EXPECTED_F3A_PREOPEN_SHA256:
        raise RuntimeError(
            "Failed 05f3a PREOPENING_LOCK differs from logged immutable hash."
        )

    if F3A_OPENING_AUDIT.exists():
        raise RuntimeError(
            "05f3a opening audit exists unexpectedly. "
            "The supplied failure occurred before that artifact should have been written."
        )

    # ------------------------------------------------------------------
    # Verify PRE-EXISTING Paper-4 complete-case provenance.
    # ------------------------------------------------------------------
    paper4_root, paper4_resolution = resolve_paper4_root()

    p4_prep_script = paper4_root / P4_PREP_SCRIPT_REL
    p4_valid_script = paper4_root / P4_VALID_SCRIPT_REL
    p4_prep_summary = paper4_root / P4_PREP_SUMMARY_REL
    p4_target_primary = paper4_root / P4_TARGET_PRIMARY_REL

    for path in [
        p4_prep_script,
        p4_valid_script,
        p4_prep_summary,
        p4_target_primary,
    ]:
        require_file(path)

    prep_version = script_version_from_ast(p4_prep_script)
    valid_version = script_version_from_ast(p4_valid_script)

    if prep_version != EXPECTED_P4_PREP_SCRIPT_VERSION:
        raise RuntimeError(
            f"Paper4 script22 version changed: {prep_version!r}"
        )

    if valid_version != EXPECTED_P4_VALID_SCRIPT_VERSION:
        raise RuntimeError(
            f"Paper4 script23 version changed: {valid_version!r}"
        )

    prep_fn = function_source(
        p4_prep_script,
        "standardize_target_clinical",
    )
    valid_fn = function_source(
        p4_valid_script,
        "fit_cox_score",
    )

    # Code-level proof of the pre-existing missing-event and complete-case policy.
    required_prep_fragments = [
        'vital_status.lower() == "dead"',
        'vital_status.lower() == "alive"',
        "else np.nan",
        '"os_event": os_event',
        '"os_time_days": os_time',
    ]
    for fragment in required_prep_fragments:
        if fragment not in prep_fn:
            raise RuntimeError(
                "Paper4 script22 no longer contains expected endpoint "
                f"preparation fragment: {fragment}"
            )

    if ".dropna()" not in valid_fn:
        raise RuntimeError(
            "Paper4 script23 fit_cox_score no longer contains complete-case dropna()."
        )

    prep_summary = pd.read_csv(
        p4_prep_summary,
        low_memory=False,
    )

    target_prep = prep_summary[
        prep_summary["cohort"].astype(str) == "TARGET_OS"
    ].copy()

    if len(target_prep) != 1:
        raise RuntimeError(
            "Paper4 preparation summary lacks unique TARGET_OS row."
        )

    target_prep_row = target_prep.iloc[0]

    observed_preexisting = {
        "n_expression_samples": int(target_prep_row["n_expression_samples"]),
        "n_clinical_rows": int(target_prep_row["n_clinical_rows"]),
        "n_os_complete": int(target_prep_row["n_os_complete"]),
    }

    expected_preexisting = {
        "n_expression_samples": EXPECTED_TOTAL_EXPRESSION_N,
        "n_clinical_rows": EXPECTED_TOTAL_CLINICAL_N,
        "n_os_complete": EXPECTED_OS_COMPLETE_N,
    }

    if observed_preexisting != expected_preexisting:
        raise RuntimeError(
            "Pre-existing Paper4 preparation counts differ from expected "
            f"88/88/86: {observed_preexisting}"
        )

    old_primary = pd.read_csv(
        p4_target_primary,
        low_memory=False,
    )

    if "n" not in old_primary.columns or "events" not in old_primary.columns:
        raise RuntimeError(
            "Paper4 TARGET primary validation lacks n/events columns."
        )

    old_n_values = sorted(
        set(pd.to_numeric(old_primary["n"], errors="coerce").dropna().astype(int))
    )
    old_event_values = sorted(
        set(pd.to_numeric(old_primary["events"], errors="coerce").dropna().astype(int))
    )

    if old_n_values != [EXPECTED_OS_COMPLETE_N]:
        raise RuntimeError(
            f"Paper4 TARGET survival result n differs from 86: {old_n_values}"
        )

    if old_event_values != [EXPECTED_OS_EVENTS]:
        raise RuntimeError(
            f"Paper4 TARGET survival result events differ from 29: {old_event_values}"
        )

    print("Pre-existing Paper-4 endpoint provenance: PASS")
    print("  TARGET expression/clinical: 88/88")
    print("  OS-complete recorded before Paper-6 TARGET opening: 86")
    print("  old TARGET survival analysis n/events: 86/29")
    print("  old complete-case survival policy in fit_cox_score: PASS")
    print()

    # ------------------------------------------------------------------
    # Load frozen Paper-6 sample identity.
    # ------------------------------------------------------------------
    mapping = pd.read_csv(
        F1C_ID_MAPPING,
        sep="\t",
        dtype=str,
        low_memory=False,
    ).fillna("")

    mapping["clinical_row_index"] = pd.to_numeric(
        mapping["clinical_row_index"],
        errors="raise",
    ).astype(int)

    if len(mapping) != EXPECTED_TOTAL_EXPRESSION_N:
        raise RuntimeError("05f1c mapping is not 88 rows.")

    if mapping["expression_sample_id"].astype(str).duplicated().any():
        raise RuntimeError("05f1c mapping sample IDs are not unique.")

    with np.load(
        TARGET_MATRIX,
        allow_pickle=False,
    ) as data:
        sample_ids = data["sample_id"].astype(str)
        X_shape = tuple(data["X"].shape)

    if X_shape != (EXPECTED_TOTAL_EXPRESSION_N, 11815):
        raise RuntimeError(
            f"TARGET matrix shape changed: {X_shape}"
        )

    if len(set(sample_ids.tolist())) != EXPECTED_TOTAL_EXPRESSION_N:
        raise RuntimeError("TARGET NPZ IDs not unique 88/88.")

    mapping_ids = mapping["expression_sample_id"].astype(str).tolist()

    if set(mapping_ids) != set(sample_ids.tolist()):
        raise RuntimeError(
            "05f1c mapping and TARGET NPZ do not contain identical 88 IDs."
        )

    # Deterministic reconciliation to immutable NPZ order.
    mapping = (
        mapping
        .set_index("expression_sample_id", drop=False)
        .loc[sample_ids.tolist()]
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------------
    # Read ONLY already-authorized/opened primary endpoint columns.
    # ------------------------------------------------------------------
    clinical_path = get_locked_clinical(paper4_root)

    clinical = pd.read_csv(
        clinical_path,
        usecols=[
            EXPECTED_ID_COLUMN,
            EXPECTED_EVENT_COLUMN,
            EXPECTED_TIME_COLUMN,
        ],
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )

    if set(clinical.columns) != {
        EXPECTED_ID_COLUMN,
        EXPECTED_EVENT_COLUMN,
        EXPECTED_TIME_COLUMN,
    }:
        raise RuntimeError(
            f"Unexpected clinical columns read: {list(clinical.columns)}"
        )

    if len(clinical) != EXPECTED_TOTAL_CLINICAL_N:
        raise RuntimeError(
            f"Clinical table rows={len(clinical)}, expected 88."
        )

    clinical_ids = clinical[EXPECTED_ID_COLUMN].astype(str).map(str.strip)

    if clinical_ids.eq("").any() or clinical_ids.duplicated().any():
        raise RuntimeError(
            "TARGET clinical case_submitter_id is blank or duplicated."
        )

    endpoint_schema = read_json(F1C_ENDPOINT_SCHEMA)
    parser = endpoint_schema.get("event_parser") or {}

    audit_rows = []

    for npz_index, row in mapping.iterrows():
        clinical_row_index = int(row["clinical_row_index"])

        if clinical_row_index < 0 or clinical_row_index >= len(clinical):
            raise RuntimeError(
                f"Clinical row index out of bounds: {clinical_row_index}"
            )

        clinical_row = clinical.iloc[clinical_row_index]

        current_id = str(clinical_row[EXPECTED_ID_COLUMN]).strip()
        frozen_id = str(row["clinical_identifier"]).strip()

        if current_id.upper() != frozen_id.upper():
            raise RuntimeError(
                "Frozen clinical-row identity mismatch for "
                f"expression sample {row['expression_sample_id']!r}."
            )

        event_ok, event_value, event_status = parse_nonmissing_event(
            clinical_row[EXPECTED_EVENT_COLUMN],
            parser,
        )

        time_ok, time_value, time_status = parse_nonmissing_time(
            clinical_row[EXPECTED_TIME_COLUMN]
        )

        # Only BLANK missingness is eligible for the pre-existing complete-case
        # rule. Any nonblank malformed value remains FAIL_CLOSED.
        if event_status.startswith("INVALID_"):
            raise RuntimeError(
                f"Nonblank invalid os_event for {current_id}: {event_status}"
            )

        if time_status.startswith("INVALID_"):
            raise RuntimeError(
                f"Nonblank invalid os_time_days for {current_id}: {time_status}"
            )

        complete = bool(event_ok and time_ok)

        audit_rows.append(
            {
                "sample_index_88": int(npz_index),
                "sample_id": str(sample_ids[npz_index]),
                "case_key": str(row["frozen_case_key"]),
                "clinical_row_index": clinical_row_index,
                "clinical_identifier": current_id,
                "os_event_raw": str(clinical_row[EXPECTED_EVENT_COLUMN]),
                "os_time_days_raw": str(clinical_row[EXPECTED_TIME_COLUMN]),
                "event_parse_status": event_status,
                "time_parse_status": time_status,
                "endpoint_complete": complete,
                "os_event_parsed": (
                    int(event_value)
                    if event_value is not None
                    else np.nan
                ),
                "os_time_days_parsed": (
                    float(time_value)
                    if time_value is not None
                    else np.nan
                ),
            }
        )

    audit = pd.DataFrame(audit_rows)

    n_complete = int(audit["endpoint_complete"].sum())
    n_incomplete = int((~audit["endpoint_complete"]).sum())

    complete = audit[audit["endpoint_complete"]].copy()
    incomplete = audit[~audit["endpoint_complete"]].copy()

    n_events = int(
        pd.to_numeric(
            complete["os_event_parsed"],
            errors="raise",
        ).sum()
    )
    n_censored = int(
        (
            pd.to_numeric(
                complete["os_event_parsed"],
                errors="raise",
            )
            == 0
        ).sum()
    )

    # This is the critical anti-post-hoc constraint:
    # current already-opened primary endpoint must EXACTLY reproduce the
    # pre-existing Paper-4 aggregate endpoint eligibility and event counts.
    if n_complete != EXPECTED_OS_COMPLETE_N:
        raise RuntimeError(
            f"Current TARGET OS complete n={n_complete}, "
            f"but pre-existing Paper4 recorded {EXPECTED_OS_COMPLETE_N}."
        )

    if n_incomplete != EXPECTED_OS_INCOMPLETE_N:
        raise RuntimeError(
            f"Current TARGET incomplete n={n_incomplete}, expected 2."
        )

    if n_events != EXPECTED_OS_EVENTS:
        raise RuntimeError(
            f"Current TARGET complete-case events={n_events}, "
            f"but pre-existing Paper4 recorded {EXPECTED_OS_EVENTS}."
        )

    if n_censored != EXPECTED_OS_CENSORED:
        raise RuntimeError(
            f"Current TARGET complete-case censored={n_censored}, "
            f"expected {EXPECTED_OS_CENSORED}."
        )

    audit.to_csv(
        MISSINGNESS_AUDIT,
        sep="\t",
        index=False,
    )

    complete.to_csv(
        COMPLETE_ROSTER,
        sep="\t",
        index=False,
    )

    incomplete.to_csv(
        INCOMPLETE_ROSTER,
        sep="\t",
        index=False,
    )

    print("=" * 120)
    print("CURRENT ALREADY-OPENED PRIMARY ENDPOINT RECONCILIATION")
    print("=" * 120)
    print(f"TARGET expression cases: {EXPECTED_TOTAL_EXPRESSION_N}")
    print(f"TARGET primary OS complete: {n_complete}")
    print(f"TARGET primary OS incomplete: {n_incomplete}")
    print(f"Complete-case OS events: {n_events}")
    print(f"Complete-case OS censored: {n_censored}")
    print()
    print("Incomplete primary-endpoint cases:")
    for row in incomplete.itertuples(index=False):
        print(
            f"  {row.sample_id}: "
            f"event={row.event_parse_status}, "
            f"time={row.time_parse_status}"
        )
    print()
    print("Agreement with PRE-EXISTING Paper-4 counts:")
    print("  complete n 86: PASS")
    print("  events 29: PASS")
    print("  censored 57: PASS")
    print("  no outcome imputation: PASS")
    print()

    reconciliation = {
        "script_version": SCRIPT_VERSION,
        "audit_version": AUDIT_VERSION,
        "status": "PASS",
        "scientific_status": (
            "PASS_POSTOPENING_TARGET_OS_PREEXISTING_COMPLETE_CASE_POLICY_RECONCILED"
        ),
        "created_utc": now_utc(),

        "Paper6_TARGET_outcomes_already_open": True,
        "opening_stage": "failed 05f3a",
        "05f3a_model_fitting_occurred": False,

        "postopening_amendment": {
            "type": "TECHNICAL_ENDPOINT_ELIGIBILITY_RECONCILIATION",
            "scientific_models_changed": False,
            "hyperparameters_changed": False,
            "metrics_changed": False,
            "branch_rules_changed": False,
            "resampling_rules_changed": False,
            "outcome_values_imputed": False,
            "new_clinical_columns_read": False,
            "reason": (
                "Paper6 05f1c mistakenly required endpoint completeness for all "
                "88 expression cases. Pre-existing Paper4 preparation/validation "
                "already defined TARGET OS analysis on 86 complete time+event rows."
            ),
            "eligibility_rule": (
                "include a TARGET expression case iff BOTH frozen primary "
                "os_event and os_time_days are nonmissing and valid; blank "
                "primary endpoint field -> endpoint-incomplete; any nonblank "
                "malformed value -> FAIL_CLOSED"
            ),
            "eligibility_is_prediction_independent": True,
        },

        "preexisting_Paper4_provenance": {
            "paper4_root_resolution": paper4_resolution,
            "script22_path": str(P4_PREP_SCRIPT_REL),
            "script22_sha256": sha256_file(p4_prep_script),
            "script22_version": prep_version,
            "standardize_target_clinical_function_sha256": hashlib.sha256(
                prep_fn.encode("utf-8")
            ).hexdigest(),
            "script23_path": str(P4_VALID_SCRIPT_REL),
            "script23_sha256": sha256_file(p4_valid_script),
            "script23_version": valid_version,
            "fit_cox_score_function_sha256": hashlib.sha256(
                valid_fn.encode("utf-8")
            ).hexdigest(),
            "preparation_summary_path": str(P4_PREP_SUMMARY_REL),
            "preparation_summary_sha256": sha256_file(p4_prep_summary),
            "preparation_counts": observed_preexisting,
            "old_TARGET_primary_result_path": str(P4_TARGET_PRIMARY_REL),
            "old_TARGET_primary_result_sha256": sha256_file(p4_target_primary),
            "old_TARGET_primary_unique_n": old_n_values,
            "old_TARGET_primary_unique_events": old_event_values,
        },

        "current_TARGET_primary_endpoint": {
            "total_expression_cases": EXPECTED_TOTAL_EXPRESSION_N,
            "complete_cases": n_complete,
            "incomplete_cases": n_incomplete,
            "events": n_events,
            "censored": n_censored,
            "matches_preexisting_counts_exactly": True,
        },

        "allowed_columns_read": [
            EXPECTED_ID_COLUMN,
            EXPECTED_EVENT_COLUMN,
            EXPECTED_TIME_COLUMN,
        ],
        "forbidden_columns_read": False,
        "GSE21257_outcomes_read": False,
        "GSE39055_outcomes_read": False,
        "model_predictions_read": False,
        "TARGET_models_fit": False,

        "artifact_hashes": {
            MISSINGNESS_AUDIT.name: sha256_file(MISSINGNESS_AUDIT),
            COMPLETE_ROSTER.name: sha256_file(COMPLETE_ROSTER),
            INCOMPLETE_ROSTER.name: sha256_file(INCOMPLETE_ROSTER),
        },

        "next": (
            "05f3c technical runner amendment: change only the frozen TARGET "
            "runner sample-count assertion from 88 to the pre-existing 86 "
            "OS-complete cases; execute T0-T2/N0-N5 on this exact 86-case "
            "roster with all model/metric/branch/resampling rules unchanged."
        ),
    }

    write_json(
        RECONCILIATION_JSON,
        reconciliation,
    )

    summary = {
        "script_version": SCRIPT_VERSION,
        "status": "PASS",
        "scientific_status": reconciliation["scientific_status"],
        "run_started_utc": started,
        "run_finished_utc": now_utc(),
        "TARGET_outcomes_already_open": True,
        "TARGET_models_fit": False,
        "complete_n": n_complete,
        "incomplete_n": n_incomplete,
        "events": n_events,
        "censored": n_censored,
        "matches_preexisting_Paper4_endpoint_counts": True,
        "reconciliation_sha256": sha256_file(RECONCILIATION_JSON),
        "next": reconciliation["next"],
    }

    write_json(
        SUMMARY_JSON,
        summary,
    )

    print("=" * 120)
    print("05f3b RECONCILIATION SUMMARY")
    print("=" * 120)
    print("Pre-existing Paper4 complete-case policy recovered: PASS")
    print("Current TARGET endpoint reproduces old n/events exactly: PASS")
    print("Model/metric/branch/resampling changes: NO")
    print("Outcome imputation: NO")
    print("TARGET model fitting in 05f3b: NO")
    print("GSE21257/GSE39055 outcomes read: NO")
    print()
    print(f"Reconciliation SHA256: {sha256_file(RECONCILIATION_JSON)}")
    print()
    print("Next:")
    print(
        "  05f3c patch ONLY runner n=88 assertion -> n=86 complete OS, "
        "then run the otherwise unchanged frozen evaluation."
    )
    print("=" * 120)
    print(
        "05f3b: PASS_POSTOPENING_TARGET_OS_PREEXISTING_COMPLETE_CASE_POLICY_RECONCILED"
    )
    print("=" * 120)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 120, file=sys.stderr)
        print(
            "05f3b TARGET-OS complete-case reconciliation: FAIL",
            file=sys.stderr,
        )
        print(
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        print("=" * 120, file=sys.stderr)
        raise
