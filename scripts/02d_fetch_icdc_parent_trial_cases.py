from __future__ import annotations

from pathlib import Path
import json
import re
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests


SCRIPT_VERSION = "02d-fetch-icdc-parent-trial-cases-v1-no-cli"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_DIR = PROJECT_ROOT / "contracts"
MANIFEST_DIR = PROJECT_ROOT / "manifests"

ICDC_GRAPHQL_URL = "https://caninecommons.cancer.gov/v1/graphql/"

PROBE_STATUS = CONTRACT_DIR / "02c_icdc_parent_trial_api_status.json"
SELECTED_MAPPING = MANIFEST_DIR / "02a_dog2_selected186_authoritative_id_mapping.csv"
SELECTED_LOCK = CONTRACT_DIR / "02a_dog2_authoritative_id_mapping_lock.json"

OUT_ATTEMPTS = MANIFEST_DIR / "02d_icdc_case_query_attempts.csv"
OUT_PARENT_FLAT = MANIFEST_DIR / "02d_icdc_parent_trial_cases_flat.csv"
OUT_PARENT_RAW = MANIFEST_DIR / "02d_icdc_parent_trial_cases_raw.json"
OUT_FIELD_AUDIT = MANIFEST_DIR / "02d_icdc_parent_trial_identifier_field_audit.csv"
OUT_STATUS = CONTRACT_DIR / "02d_icdc_parent_trial_case_fetch_status.json"
OUT_README = MANIFEST_DIR / "02d_icdc_parent_trial_case_fetch_README.txt"

EXPECTED_COUNTS = {
    "COTC021": 152,
    "COTC022": 157,
}
EXPECTED_TOTAL = 309
EXPECTED_SELECTED_N = 186

REQUEST_TIMEOUT_SECONDS = 90
MAX_FIRST = 1000

COTC_PATTERN = re.compile(
    r"\b(COTC0?\d{2,3})[-_ ]+0*(\d{2,6})(?:[-_ ]+[A-Z])?\b",
    flags=re.IGNORECASE,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "na", "n/a"}:
        return ""
    return text


def normalize_cotc(value: Any) -> str:
    text = clean_text(value).upper()
    if not text:
        return ""

    match = COTC_PATTERN.search(text)
    if match is None:
        return ""

    study_digits = re.sub(r"\D", "", match.group(1)[4:])
    subject_digits = re.sub(r"\D", "", match.group(2))

    if len(study_digits) == 2:
        study_digits = "0" + study_digits

    if not study_digits or not subject_digits:
        return ""

    return f"COTC{study_digits}-{subject_digits.zfill(4)}"


def graphql_post(query: str) -> dict[str, Any]:
    response = requests.post(
        ICDC_GRAPHQL_URL,
        json={"query": query},
        timeout=REQUEST_TIMEOUT_SECONDS,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "paper6-cross-species-osteosarcoma/02d",
        },
    )
    response.raise_for_status()

    payload = response.json()

    if payload.get("errors"):
        raise RuntimeError(
            "GraphQL errors:\n"
            + json.dumps(payload["errors"], indent=2)
        )

    if "data" not in payload:
        raise RuntimeError(
            "ICDC GraphQL response has no data object."
        )

    return payload


def type_to_string(node: dict[str, Any] | None) -> str:
    if not node:
        return ""

    kind = node.get("kind")
    name = node.get("name")
    of_type = node.get("ofType")

    if kind == "NON_NULL":
        return type_to_string(of_type) + "!"
    if kind == "LIST":
        return "[" + type_to_string(of_type) + "]"

    return str(name or kind or "")


def named_type(node: dict[str, Any] | None) -> str:
    current = node or {}

    for _ in range(10):
        name = current.get("name")
        if name:
            return str(name)
        current = current.get("ofType") or {}

    return ""


def base_kind(node: dict[str, Any] | None) -> str:
    current = node or {}

    for _ in range(10):
        kind = current.get("kind")
        if kind not in {"NON_NULL", "LIST"}:
            return str(kind or "")
        current = current.get("ofType") or {}

    return ""


def is_required(node: dict[str, Any] | None) -> bool:
    return bool(node and node.get("kind") == "NON_NULL")


def verify_prerequisites() -> tuple[set[str], dict[str, Any]]:
    if not PROBE_STATUS.exists():
        raise FileNotFoundError(
            "Run scripts/02c_probe_icdc_parent_trial_api.py first."
        )

    probe = json.loads(
        PROBE_STATUS.read_text(encoding="utf-8")
    )

    if not str(probe.get("status", "")).startswith("PASS_"):
        raise RuntimeError(
            "02c ICDC probe status is not PASS."
        )

    observed = {
        str(key): int(value)
        for key, value in probe.get("observed_counts", {}).items()
    }

    if observed != EXPECTED_COUNTS:
        raise RuntimeError(
            f"02c observed counts changed: {observed}"
        )

    if not SELECTED_MAPPING.exists() or not SELECTED_LOCK.exists():
        raise FileNotFoundError(
            "Run the PASSING Script 02a identifier bridge first."
        )

    selected_lock = json.loads(
        SELECTED_LOCK.read_text(encoding="utf-8")
    )

    if selected_lock.get("status") != "PASS":
        raise RuntimeError(
            "02a selected-subject lock is not PASS."
        )

    selected = pd.read_csv(
        SELECTED_MAPPING,
        dtype=str,
        low_memory=False,
    ).fillna("")

    selected_ids = set(
        selected["cotc_subject_id"].map(normalize_cotc)
    )
    selected_ids.discard("")

    if len(selected_ids) != EXPECTED_SELECTED_N:
        raise RuntimeError(
            f"02a selected COTC subject count={len(selected_ids)}, "
            f"expected {EXPECTED_SELECTED_N}."
        )

    return selected_ids, probe


def introspect_query_fields() -> list[dict[str, Any]]:
    query = r"""
    query Paper6QueryFields02d {
      __schema {
        queryType {
          fields {
            name
            description
            args {
              name
              description
              type {
                kind
                name
                ofType {
                  kind
                  name
                  ofType {
                    kind
                    name
                    ofType {
                      kind
                      name
                    }
                  }
                }
              }
            }
            type {
              kind
              name
              ofType {
                kind
                name
                ofType {
                  kind
                  name
                  ofType {
                    kind
                    name
                  }
                }
              }
            }
          }
        }
      }
    }
    """

    payload = graphql_post(query)

    return (
        payload
        .get("data", {})
        .get("__schema", {})
        .get("queryType", {})
        .get("fields", [])
    )


def introspect_type(type_name: str) -> list[dict[str, Any]]:
    query = f"""
    query Paper6Type02d {{
      __type(name: "{type_name}") {{
        kind
        name
        fields {{
          name
          description
          type {{
            kind
            name
            ofType {{
              kind
              name
              ofType {{
                kind
                name
                ofType {{
                  kind
                  name
                }}
              }}
            }}
          }}
        }}
      }}
    }}
    """

    payload = graphql_post(query)

    type_info = (
        payload
        .get("data", {})
        .get("__type")
    )

    if not type_info:
        return []

    return type_info.get("fields") or []


def candidate_query_fields(
    fields: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates = []

    for field in fields:
        name = str(field.get("name", ""))
        description = str(field.get("description") or "")
        args = field.get("args") or []

        arg_names = {
            str(arg.get("name", ""))
            for arg in args
        }

        has_study_arg = (
            "study_code" in arg_names
            or "studyCode" in arg_names
        )

        if not has_study_arg:
            continue

        searchable = (
            name
            + " "
            + description
            + " "
            + named_type(field.get("type"))
        ).lower()

        if not any(
            token in searchable
            for token in (
                "case",
                "subject",
                "patient",
                "clinical",
                "demographic",
                "diagnosis",
                "treatment",
                "outcome",
            )
        ):
            continue

        allowed_required = {
            "study_code",
            "studyCode",
            "first",
            "offset",
        }

        unsupported_required = [
            str(arg.get("name", ""))
            for arg in args
            if is_required(arg.get("type"))
            and str(arg.get("name", ""))
            not in allowed_required
        ]

        if unsupported_required:
            continue

        candidates.append(field)

    return candidates


def scalar_fields_for_type(
    type_name: str,
) -> list[str]:
    fields = introspect_type(type_name)

    output = []

    for field in fields:
        field_name = str(field.get("name", ""))
        kind = base_kind(field.get("type"))

        if kind in {
            "SCALAR",
            "ENUM",
        }:
            output.append(field_name)

    return output


def build_args(
    field: dict[str, Any],
    study_code: str,
) -> str:
    pieces = []

    for arg in field.get("args") or []:
        name = str(arg.get("name", ""))

        if name in {"study_code", "studyCode"}:
            pieces.append(
                f'{name}: "{study_code}"'
            )
        elif name == "first":
            pieces.append(
                f"first: {MAX_FIRST}"
            )
        elif name == "offset":
            pieces.append(
                "offset: 0"
            )

    return ", ".join(pieces)


def fetch_candidate(
    field: dict[str, Any],
    study_code: str,
    scalar_fields: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    field_name = str(field["name"])
    args = build_args(
        field,
        study_code,
    )

    selection = "\n        ".join(
        scalar_fields
    )

    query = f"""
    query Paper6Fetch02d {{
      {field_name}({args}) {{
        {selection}
      }}
    }}
    """

    payload = graphql_post(query)

    value = (
        payload
        .get("data", {})
        .get(field_name)
    )

    if value is None:
        return [], payload

    if isinstance(value, list):
        records = [
            item
            for item in value
            if isinstance(item, dict)
        ]
        return records, payload

    if isinstance(value, dict):
        return [value], payload

    return [], payload


def best_identifier_field(
    frame: pd.DataFrame,
    selected_ids: set[str],
) -> tuple[pd.DataFrame, str | None, str | None]:
    rows = []

    best_unique_field = None
    best_cotc_field = None
    best_unique_score = (-1, -1)
    best_cotc_score = (-1, -1)

    for column in frame.columns:
        if column in {
            "__study_code_requested__",
            "__query_field__",
        }:
            continue

        values = frame[column].map(clean_text)
        nonempty = values[values.ne("")]
        unique_n = int(nonempty.nunique())

        lower = column.lower()
        identifier_bonus = int(
            any(
                token in lower
                for token in (
                    "case",
                    "subject",
                    "patient",
                    "submitter",
                    "id",
                )
            )
        )

        unique_score = (
            unique_n,
            identifier_bonus,
        )

        if unique_score > best_unique_score:
            best_unique_score = unique_score
            best_unique_field = column

        cotc = values.map(normalize_cotc)
        cotc_nonempty = cotc[cotc.ne("")]
        cotc_unique = int(
            cotc_nonempty.nunique()
        )
        cotc_set = set(cotc_nonempty)
        selected_overlap = len(
            cotc_set & selected_ids
        )

        rows.append(
            {
                "field": column,
                "nonempty_n": int(nonempty.shape[0]),
                "unique_n": unique_n,
                "identifier_name_hint": bool(identifier_bonus),
                "parseable_cotc_n": int(cotc.ne("").sum()),
                "unique_cotc_n": cotc_unique,
                "selected186_cotc_overlap": selected_overlap,
            }
        )

        cotc_score = (
            selected_overlap,
            cotc_unique,
        )

        if cotc_score > best_cotc_score:
            best_cotc_score = cotc_score
            best_cotc_field = column

    return (
        pd.DataFrame(rows),
        best_unique_field,
        best_cotc_field,
    )


def main() -> None:
    print("=" * 96)
    print("Paper 6 - fetch ICDC COTC021/COTC022 parent-trial cases")
    print("=" * 96)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print("")
    print("Purpose:")
    print("  Discover a live case-level ICDC query that returns exactly")
    print("  152 COTC021 + 157 COTC022 records.")
    print("  Save only scalar clinical/case metadata; download no omics files.")
    print("  Audit whether any returned identifier directly resolves to the")
    print("  frozen 186 DOG2 COTC subjects.")
    print("")

    MANIFEST_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    CONTRACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected_ids, probe = verify_prerequisites()

    print("02a selected-subject lock: PASS")
    print("02c ICDC study-count/schema probe: PASS")
    print(f"Frozen selected COTC subjects: {len(selected_ids)}")
    print("")

    query_fields = introspect_query_fields()
    candidates = candidate_query_fields(
        query_fields
    )

    print(
        "Direct study-linked case/clinical query candidates: "
        f"{len(candidates)}"
    )

    attempts = []
    successful_payloads: dict[str, Any] = {}
    candidate_frames: dict[str, pd.DataFrame] = {}

    for field in candidates:
        field_name = str(field["name"])
        return_type = named_type(
            field.get("type")
        )

        try:
            scalar_fields = scalar_fields_for_type(
                return_type
            )
        except Exception as exc:
            attempts.append(
                {
                    "query_field": field_name,
                    "return_named_type": return_type,
                    "scalar_field_count": 0,
                    "cotc021_records": None,
                    "cotc022_records": None,
                    "combined_records": None,
                    "exact_expected_counts": False,
                    "status": (
                        "TYPE_INTROSPECTION_ERROR: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                }
            )
            continue

        if not scalar_fields:
            attempts.append(
                {
                    "query_field": field_name,
                    "return_named_type": return_type,
                    "scalar_field_count": 0,
                    "cotc021_records": 0,
                    "cotc022_records": 0,
                    "combined_records": 0,
                    "exact_expected_counts": False,
                    "status": "NO_SCALAR_FIELDS",
                }
            )
            continue

        records_by_study = {}
        payloads_by_study = {}
        error_text = None

        for study_code in EXPECTED_COUNTS:
            try:
                records, payload = fetch_candidate(
                    field=field,
                    study_code=study_code,
                    scalar_fields=scalar_fields,
                )
                records_by_study[study_code] = records
                payloads_by_study[study_code] = payload
            except Exception as exc:
                error_text = (
                    f"{type(exc).__name__}: {exc}"
                )
                break

        if error_text is not None:
            attempts.append(
                {
                    "query_field": field_name,
                    "return_named_type": return_type,
                    "scalar_field_count": len(scalar_fields),
                    "cotc021_records": None,
                    "cotc022_records": None,
                    "combined_records": None,
                    "exact_expected_counts": False,
                    "status": (
                        "QUERY_ERROR: "
                        + error_text
                    ),
                }
            )
            continue

        observed = {
            study_code: len(
                records_by_study.get(
                    study_code,
                    [],
                )
            )
            for study_code in EXPECTED_COUNTS
        }

        exact = observed == EXPECTED_COUNTS

        attempts.append(
            {
                "query_field": field_name,
                "return_named_type": return_type,
                "scalar_field_count": len(scalar_fields),
                "cotc021_records": observed["COTC021"],
                "cotc022_records": observed["COTC022"],
                "combined_records": sum(observed.values()),
                "exact_expected_counts": exact,
                "status": "SUCCESS",
            }
        )

        if exact:
            rows = []

            for study_code in EXPECTED_COUNTS:
                for record in records_by_study[
                    study_code
                ]:
                    row = dict(record)
                    row[
                        "__study_code_requested__"
                    ] = study_code
                    row[
                        "__query_field__"
                    ] = field_name
                    rows.append(row)

            frame = pd.DataFrame(rows)
            candidate_frames[field_name] = frame
            successful_payloads[field_name] = (
                payloads_by_study
            )

    attempts_df = pd.DataFrame(attempts)

    if not attempts_df.empty:
        attempts_df = attempts_df.sort_values(
            [
                "exact_expected_counts",
                "combined_records",
                "query_field",
            ],
            ascending=[False, False, True],
        ).reset_index(drop=True)

    attempts_df.to_csv(
        OUT_ATTEMPTS,
        index=False,
    )

    if not candidate_frames:
        status = "BLOCKED_NO_EXACT_309_CASE_QUERY"
        reason = (
            "No introspected direct study-linked case/clinical GraphQL "
            "field returned exactly 152 COTC021 and 157 COTC022 records."
        )

        pd.DataFrame().to_csv(
            OUT_PARENT_FLAT,
            index=False,
        )
        pd.DataFrame().to_csv(
            OUT_FIELD_AUDIT,
            index=False,
        )
        OUT_PARENT_RAW.write_text(
            json.dumps(
                successful_payloads,
                indent=2,
            ),
            encoding="utf-8",
        )

        selected_field = None
        selected_identifier_field = None
        selected_cotc_field = None
        field_audit = pd.DataFrame()
        parent_frame = pd.DataFrame()

    else:
        scored = []

        for field_name, frame in candidate_frames.items():
            (
                field_audit_candidate,
                unique_field,
                cotc_field,
            ) = best_identifier_field(
                frame=frame,
                selected_ids=selected_ids,
            )

            best_overlap = (
                int(
                    field_audit_candidate[
                        "selected186_cotc_overlap"
                    ].max()
                )
                if not field_audit_candidate.empty
                else 0
            )

            max_unique = (
                int(
                    field_audit_candidate[
                        "unique_n"
                    ].max()
                )
                if not field_audit_candidate.empty
                else 0
            )

            scored.append(
                (
                    best_overlap,
                    max_unique,
                    field_name,
                    unique_field,
                    cotc_field,
                    field_audit_candidate,
                )
            )

        scored.sort(
            key=lambda item: (
                item[0],
                item[1],
                item[2],
            ),
            reverse=True,
        )

        (
            best_overlap,
            max_unique,
            selected_field,
            selected_identifier_field,
            selected_cotc_field,
            field_audit,
        ) = scored[0]

        parent_frame = candidate_frames[
            selected_field
        ].copy()

        field_audit[
            "selected_query_field"
        ] = selected_field

        parent_frame.to_csv(
            OUT_PARENT_FLAT,
            index=False,
        )
        field_audit.to_csv(
            OUT_FIELD_AUDIT,
            index=False,
        )

        OUT_PARENT_RAW.write_text(
            json.dumps(
                successful_payloads[
                    selected_field
                ],
                indent=2,
            ),
            encoding="utf-8",
        )

        if best_overlap == EXPECTED_SELECTED_N:
            status = "PASS_CASES_FETCHED_SELECTED186_LINKED"
            reason = (
                "A live ICDC query returned exactly 309 parent-trial cases "
                "and at least one scalar identifier field directly recovered "
                "all 186 frozen DOG2 COTC subjects."
            )
        else:
            status = "PASS_CASES_FETCHED_ID_LINKAGE_NOT_YET_RESOLVED"
            reason = (
                "A live ICDC query returned exactly 309 parent-trial cases, "
                "but no returned scalar field directly recovered all 186 "
                "frozen DOG2 COTC subject IDs. A second metadata layer or "
                "case-ID crosswalk is required before selection auditing."
            )

    status_payload = {
        "script_version": SCRIPT_VERSION,
        "created_utc": utc_now_iso(),
        "graphql_endpoint": ICDC_GRAPHQL_URL,
        "status": status,
        "status_reason": reason,
        "expected_parent_counts": EXPECTED_COUNTS,
        "expected_parent_total": EXPECTED_TOTAL,
        "selected_subject_n": EXPECTED_SELECTED_N,
        "query_candidate_count": len(
            candidates
        ),
        "exact_309_query_count": len(
            candidate_frames
        ),
        "selected_query_field": selected_field,
        "selected_identifier_field": selected_identifier_field,
        "selected_cotc_field": selected_cotc_field,
        "downloaded_omics": False,
        "model_fitting": False,
        "outcome_association_testing": False,
        "outputs": {
            "query_attempts": OUT_ATTEMPTS.name,
            "parent_cases_flat": OUT_PARENT_FLAT.name,
            "parent_cases_raw": OUT_PARENT_RAW.name,
            "identifier_field_audit": OUT_FIELD_AUDIT.name,
        },
    }

    OUT_STATUS.write_text(
        json.dumps(
            status_payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    OUT_README.write_text(
        f"""Paper 6 ICDC parent-trial case fetch
Script version: {SCRIPT_VERSION}

Status
------
{status}

Reason
------
{reason}

Method
------
The script uses live GraphQL introspection rather than hard-coding an assumed
ICDC case query. It searches only direct query fields with a study-code
argument and case/subject/clinical semantics, introspects each return object,
requests only scalar fields, and selects a query only if it returns exactly
152 COTC021 and 157 COTC022 records.

No omics files are downloaded. The saved JSON/CSV files are small clinical/case
metadata snapshots used only to establish the parent population and identifier
linkage before the RNA186 selection audit.
""",
        encoding="utf-8",
    )

    print("")
    print("=" * 96)
    print(f"ICDC parent-trial case fetch: {status}")
    print("=" * 96)
    print(reason)

    if selected_field is not None:
        print("")
        print(f"Selected live query field: {selected_field}")
        print(
            "Returned parent records: "
            f"{parent_frame.shape[0]}"
        )
        print(
            "Best generic identifier field: "
            f"{selected_identifier_field}"
        )
        print(
            "Best COTC-like field: "
            f"{selected_cotc_field}"
        )

        if not field_audit.empty:
            best_overlap_value = int(
                field_audit[
                    "selected186_cotc_overlap"
                ].max()
            )
            print(
                "Maximum direct frozen-selected overlap: "
                f"{best_overlap_value}/{EXPECTED_SELECTED_N}"
            )

    print("")
    print("Saved:")
    for path in (
        OUT_ATTEMPTS,
        OUT_PARENT_FLAT,
        OUT_PARENT_RAW,
        OUT_FIELD_AUDIT,
        OUT_STATUS,
        OUT_README,
    ):
        print(f"  {path}")

    print("")
    print("Next:")
    if status == "PASS_CASES_FETCHED_SELECTED186_LINKED":
        print(
            "  Freeze the 309-case ICDC snapshot and run the parent -> RNA186 "
            "selection-balance audit."
        )
    elif status.startswith("PASS_CASES_FETCHED"):
        print(
            "  Resolve the ICDC case-ID -> COTC subject crosswalk from the "
            "returned case metadata schema before any selection analysis."
        )
    else:
        print(
            "  Review 02d_icdc_case_query_attempts.csv; the next script should "
            "target the best live ICDC connection/filter field explicitly."
        )

    print("Done.")


if __name__ == "__main__":
    main()
