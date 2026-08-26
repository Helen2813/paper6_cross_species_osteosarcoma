from __future__ import annotations

from pathlib import Path
import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests


SCRIPT_VERSION = "02e1-probe-icdc-case-routes-v1-no-cli"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_DIR = PROJECT_ROOT / "contracts"
MANIFEST_DIR = PROJECT_ROOT / "manifests"

ENDPOINTS = {
    "production": "https://caninecommons.cancer.gov/v1/graphql/",
    "development": "https://caninecommons-dev.cancer.gov/v1/graphql/",
}

TEST_CASE_ID = "COTC021-0101"
REQUEST_TIMEOUT_SECONDS = 60

OUT_RESULTS = MANIFEST_DIR / "02e1_icdc_case_route_probe.csv"
OUT_RAW = MANIFEST_DIR / "02e1_icdc_case_route_probe_raw.json"
OUT_STATUS = CONTRACT_DIR / "02e1_icdc_case_route_probe_status.json"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def post(
    endpoint: str,
    query: str,
    variables: dict[str, Any] | None = None,
) -> dict[str, Any]:
    response = requests.post(
        endpoint,
        json={
            "query": query,
            "variables": variables or {},
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "paper6-cross-species-osteosarcoma/02e1",
        },
    )

    result: dict[str, Any] = {
        "http_status": response.status_code,
        "text": response.text,
    }

    try:
        result["json"] = response.json()
    except Exception:
        result["json"] = None

    return result


def error_text(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""

    errors = payload.get("errors") or []

    return " || ".join(
        str(item.get("message", item))
        if isinstance(item, dict)
        else str(item)
        for item in errors
    )


def classify(
    raw: dict[str, Any],
    path: tuple[str, ...],
) -> tuple[str, str, Any]:
    if raw.get("http_status") != 200:
        return "HTTP_FAIL", "", None

    payload = raw.get("json")

    if not isinstance(payload, dict):
        return "NON_JSON", "", None

    errors = payload.get("errors") or []

    if errors:
        return "GRAPHQL_FAIL", error_text(payload), None

    current: Any = payload.get("data")

    for key in path:
        if not isinstance(current, dict):
            current = None
            break
        current = current.get(key)

    if current is None:
        return "NULL_DATA", "", None

    return "PASS", "", current


def run_probe(
    endpoint_name: str,
    endpoint: str,
    probe_name: str,
    query: str,
    variables: dict[str, Any],
    path: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = post(
        endpoint=endpoint,
        query=query,
        variables=variables,
    )

    status, err, value = classify(
        raw=raw,
        path=path,
    )

    if isinstance(value, list):
        value_summary = f"list_len={len(value)}"
    elif isinstance(value, dict):
        value_summary = (
            "dict_keys="
            + "|".join(sorted(value.keys()))
        )
    else:
        value_summary = str(value)[:500] if value is not None else ""

    row = {
        "endpoint_name": endpoint_name,
        "endpoint": endpoint,
        "probe": probe_name,
        "status": status,
        "http_status": raw.get("http_status"),
        "error": err,
        "value_summary": value_summary,
    }

    return row, raw


def main() -> None:
    print("=" * 100)
    print("Paper 6 - probe ICDC case-detail access routes")
    print("=" * 100)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Test case: {TEST_CASE_ID}")
    print("")
    print("Purpose:")
    print("  Determine whether the production and development ICDC GraphQL endpoints")
    print("  expose case details through case(), searchCases(), or caseOverview().")
    print("  Only one known case is queried. No outcomes and no omics are requested.")
    print("")

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    CONTRACT_DIR.mkdir(parents=True, exist_ok=True)

    probes = [
        (
            "globalSearch_case",
            r"""
            query Probe($input: String, $first: Int, $offset: Int) {
              globalSearch(input: $input, first: $first, offset: $offset) {
                cases {
                  case_id
                  clinical_study_designation
                }
              }
            }
            """,
            {
                "input": TEST_CASE_ID,
                "first": 20,
                "offset": 0,
            },
            ("globalSearch", "cases"),
        ),
        (
            "case_minimal",
            r"""
            query Probe($case_id: String!) {
              case(case_id: $case_id) {
                case_id
                patient_id
              }
            }
            """,
            {"case_id": TEST_CASE_ID},
            ("case",),
        ),
        (
            "searchCases_case_ids",
            r"""
            query Probe($case_ids: [String] = []) {
              searchCases(case_ids: $case_ids) {
                numberOfCases
                caseIds
              }
            }
            """,
            {"case_ids": [TEST_CASE_ID]},
            ("searchCases",),
        ),
        (
            "caseOverview_case_ids",
            r"""
            query Probe($case_ids: [String] = [], $first: Int, $offset: Int) {
              caseOverview(
                case_ids: $case_ids
                first: $first
                offset: $offset
              ) {
                case_id
                study_code
              }
            }
            """,
            {
                "case_ids": [TEST_CASE_ID],
                "first": 10,
                "offset": 0,
            },
            ("caseOverview",),
        ),
        (
            "query_field_introspection",
            r"""
            query Probe {
              __schema {
                queryType {
                  fields {
                    name
                    args {
                      name
                    }
                  }
                }
              }
            }
            """,
            {},
            ("__schema", "queryType", "fields"),
        ),
    ]

    rows = []
    raw_outputs: dict[str, Any] = {}

    for endpoint_name, endpoint in ENDPOINTS.items():
        print(f"Endpoint: {endpoint_name}")
        print(f"  {endpoint}")

        for probe_name, query, variables, path in probes:
            row, raw = run_probe(
                endpoint_name=endpoint_name,
                endpoint=endpoint,
                probe_name=probe_name,
                query=query,
                variables=variables,
                path=path,
            )

            rows.append(row)
            raw_outputs[
                f"{endpoint_name}::{probe_name}"
            ] = raw

            print(
                f"  {probe_name}: {row['status']}"
                + (
                    f" — {row['error']}"
                    if row["error"]
                    else ""
                )
            )

        print("")

    results = pd.DataFrame(rows)
    results.to_csv(
        OUT_RESULTS,
        index=False,
    )

    OUT_RAW.write_text(
        json.dumps(raw_outputs, indent=2),
        encoding="utf-8",
    )

    working_routes = results[
        results["status"].eq("PASS")
        & results["probe"].isin(
            {
                "case_minimal",
                "searchCases_case_ids",
                "caseOverview_case_ids",
            }
        )
    ][
        ["endpoint_name", "probe"]
    ].to_dict("records")

    prod_case_status = results.loc[
        (
            results["endpoint_name"].eq("production")
            & results["probe"].eq("case_minimal")
        ),
        "status",
    ].iloc[0]

    dev_case_status = results.loc[
        (
            results["endpoint_name"].eq("development")
            & results["probe"].eq("case_minimal")
        ),
        "status",
    ].iloc[0]

    if prod_case_status == "PASS":
        status = "PASS_PRODUCTION_CASE_DETAIL_AVAILABLE"
        reason = (
            "The production ICDC endpoint returns a case object for the known "
            "COTC021 case. Batch clinical retrieval can remain on production."
        )
    elif dev_case_status == "PASS":
        status = "PASS_DEV_CASE_DETAIL_ONLY"
        reason = (
            "The production endpoint does not return the case detail object, "
            "but the current ICDC development endpoint does. Do not yet freeze "
            "research metadata from dev; first compare production-accessible "
            "alternatives and verify dev/production roster equivalence."
        )
    elif working_routes:
        status = "PASS_ALTERNATE_PUBLIC_ROUTE_AVAILABLE"
        reason = (
            "case(case_id) is unavailable, but another public route can resolve "
            "the test case. Use that route for the next schema-specific probe."
        )
    else:
        status = "BLOCKED_NO_PUBLIC_CASE_DETAIL_ROUTE_FOUND"
        reason = (
            "Neither tested endpoint exposed the test case through case(), "
            "searchCases(), or caseOverview() using the known public signatures."
        )

    payload = {
        "script_version": SCRIPT_VERSION,
        "created_utc": utc_now_iso(),
        "test_case_id": TEST_CASE_ID,
        "status": status,
        "status_reason": reason,
        "working_routes": working_routes,
        "outcome_fields_requested": False,
        "omics_downloaded": False,
        "outputs": {
            "results": OUT_RESULTS.name,
            "raw": OUT_RAW.name,
        },
    }

    OUT_STATUS.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    print("=" * 100)
    print(f"ICDC case-route probe: {status}")
    print("=" * 100)
    print(reason)
    print("")
    print("Working case-level routes:")
    if working_routes:
        for item in working_routes:
            print(
                f"  {item['endpoint_name']}: {item['probe']}"
            )
    else:
        print("  none")

    print("")
    print("Saved:")
    print(f"  {OUT_RESULTS}")
    print(f"  {OUT_RAW}")
    print(f"  {OUT_STATUS}")
    print("")
    print("Do not re-run the 309-case batch until this route probe is reviewed.")
    print("Done.")


if __name__ == "__main__":
    main()
