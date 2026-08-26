from __future__ import annotations

from pathlib import Path
import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests


SCRIPT_VERSION = "02e0-probe-icdc-case-detail-v1-no-cli"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_DIR = PROJECT_ROOT / "contracts"
MANIFEST_DIR = PROJECT_ROOT / "manifests"

ICDC_GRAPHQL_URL = "https://caninecommons.cancer.gov/v1/graphql/"
TEST_CASE_ID = "COTC021-0101"

OUT_PROBES = MANIFEST_DIR / "02e0_icdc_case_detail_probe.csv"
OUT_RAW = MANIFEST_DIR / "02e0_icdc_case_detail_probe_raw.json"
OUT_STATUS = CONTRACT_DIR / "02e0_icdc_case_detail_probe_status.json"

REQUEST_TIMEOUT_SECONDS = 60


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def post(query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
    response = requests.post(
        ICDC_GRAPHQL_URL,
        json={
            "query": query,
            "variables": variables or {},
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "paper6-cross-species-osteosarcoma/02e0",
        },
    )
    result = {
        "http_status": response.status_code,
        "text": response.text,
    }
    try:
        result["json"] = response.json()
    except Exception:
        result["json"] = None
    return result


def run_probe(
    name: str,
    selection: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    query = f"""
    query Paper6CaseProbe($case_id: String!) {{
      case(case_id: $case_id) {{
        {selection}
      }}
    }}
    """

    raw = post(
        query,
        {"case_id": TEST_CASE_ID},
    )

    payload = raw.get("json")
    errors = []
    case_value = None

    if isinstance(payload, dict):
        errors = payload.get("errors") or []
        case_value = (
            payload
            .get("data", {})
            .get("case")
        )

    status = "PASS"
    if raw["http_status"] != 200:
        status = "HTTP_FAIL"
    elif errors:
        status = "GRAPHQL_FAIL"
    elif not isinstance(case_value, dict):
        status = "NO_CASE_OBJECT"

    error_text = ""
    if errors:
        error_text = " || ".join(
            str(item.get("message", item))
            if isinstance(item, dict)
            else str(item)
            for item in errors
        )

    row = {
        "probe": name,
        "status": status,
        "http_status": raw["http_status"],
        "error": error_text,
        "returned_case_keys": (
            " | ".join(sorted(case_value.keys()))
            if isinstance(case_value, dict)
            else ""
        ),
    }

    return row, raw


def main() -> None:
    print("=" * 96)
    print("Paper 6 - probe ICDC case-detail GraphQL before batch fetching")
    print("=" * 96)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Endpoint: {ICDC_GRAPHQL_URL}")
    print(f"Test case: {TEST_CASE_ID}")
    print("")
    print("This diagnostic performs only a handful of tiny API requests.")
    print("It does not download omics data and does not access outcome fields.")
    print("")

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    CONTRACT_DIR.mkdir(parents=True, exist_ok=True)

    probes = [
        (
            "minimal_case",
            """
            case_id
            patient_id
            """,
        ),
        (
            "study",
            """
            case_id
            study {
              clinical_study_name
              clinical_study_designation
              accession_id
            }
            """,
        ),
        (
            "demographic",
            """
            case_id
            demographic {
              breed
              sex
              patient_age_at_enrollment
              neutered_indicator
              weight
              additional_breed_detail
            }
            """,
        ),
        (
            "cohort_arm",
            """
            case_id
            cohort {
              cohort_description
              study_arm {
                arm
                ctep_treatment_assignment_code
              }
            }
            """,
        ),
        (
            "enrollment",
            """
            case_id
            enrollment {
              site_short_name
              patient_subgroup
            }
            """,
        ),
        (
            "baseline_diagnosis",
            """
            case_id
            diagnoses {
              disease_term
              stage_of_disease
              primary_disease_site
              histological_grade
              histology_cytopathology
            }
            """,
        ),
    ]

    rows = []
    raw_outputs: dict[str, Any] = {}

    for name, selection in probes:
        print(f"Probe: {name} ...", end=" ")
        row, raw = run_probe(
            name=name,
            selection=selection,
        )
        rows.append(row)
        raw_outputs[name] = raw
        print(row["status"])

        if row["error"]:
            print(f"  {row['error']}")

    frame = pd.DataFrame(rows)
    frame.to_csv(
        OUT_PROBES,
        index=False,
    )

    OUT_RAW.write_text(
        json.dumps(raw_outputs, indent=2),
        encoding="utf-8",
    )

    minimal_status = frame.loc[
        frame["probe"].eq("minimal_case"),
        "status",
    ].iloc[0]

    if minimal_status != "PASS":
        status = "BLOCKED_CASE_QUERY_NOT_PUBLIC_OR_SCHEMA_MISMATCH"
        reason = (
            "The minimal public case(case_id) probe failed. The 309-case batch "
            "must not be retried until the API route/schema is corrected."
        )
    else:
        failed_blocks = frame.loc[
            frame["status"].ne("PASS"),
            "probe",
        ].tolist()

        if not failed_blocks:
            status = "PASS_ALL_BASELINE_BLOCKS"
            reason = (
                "Minimal case query and all requested baseline metadata blocks "
                "work on the public ICDC endpoint. A corrected batch fetch can "
                "safely request these fields."
            )
        else:
            status = "PASS_MINIMAL_CASE_PARTIAL_BASELINE_SCHEMA"
            reason = (
                "The case query is public, but these baseline blocks fail on "
                "the current production schema: "
                + ", ".join(failed_blocks)
                + ". The corrected batch fetch should request only working "
                "blocks or introspect replacements."
            )

    payload = {
        "script_version": SCRIPT_VERSION,
        "created_utc": utc_now_iso(),
        "endpoint": ICDC_GRAPHQL_URL,
        "test_case_id": TEST_CASE_ID,
        "status": status,
        "status_reason": reason,
        "probe_results": rows,
        "outcome_fields_requested": False,
        "omics_downloaded": False,
    }

    OUT_STATUS.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    print("")
    print("=" * 96)
    print(f"ICDC case-detail probe: {status}")
    print("=" * 96)
    print(reason)
    print("")
    print("Saved:")
    print(f"  {OUT_PROBES}")
    print(f"  {OUT_RAW}")
    print(f"  {OUT_STATUS}")
    print("")
    print("Do not re-run the 309-case batch until this probe result is reviewed.")
    print("Done.")


if __name__ == "__main__":
    main()
