from __future__ import annotations

from pathlib import Path
import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests


SCRIPT_VERSION = "02c-probe-icdc-parent-trial-api-v1-no-cli"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_DIR = PROJECT_ROOT / "contracts"
MANIFEST_DIR = PROJECT_ROOT / "manifests"

ICDC_GRAPHQL_URL = "https://caninecommons.cancer.gov/v1/graphql/"

OUT_STUDIES = MANIFEST_DIR / "02c_icdc_study_inventory.csv"
OUT_QUERY_FIELDS = MANIFEST_DIR / "02c_icdc_graphql_query_fields.csv"
OUT_SCHEMA_JSON = MANIFEST_DIR / "02c_icdc_graphql_schema_probe.json"
OUT_STATUS = CONTRACT_DIR / "02c_icdc_parent_trial_api_status.json"
OUT_README = MANIFEST_DIR / "02c_icdc_parent_trial_api_README.txt"

EXPECTED_PARENT_STUDIES = {
    "COTC021": 152,
    "COTC022": 157,
}
EXPECTED_COMBINED_CASES = 309

REQUEST_TIMEOUT_SECONDS = 60


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def graphql_post(query: str) -> dict[str, Any]:
    response = requests.post(
        ICDC_GRAPHQL_URL,
        json={"query": query},
        timeout=REQUEST_TIMEOUT_SECONDS,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "paper6-cross-species-osteosarcoma/02c",
        },
    )
    response.raise_for_status()

    payload = response.json()

    if payload.get("errors"):
        raise RuntimeError(
            "ICDC GraphQL returned errors:\n"
            + json.dumps(payload["errors"], indent=2)
        )

    if "data" not in payload:
        raise RuntimeError(
            "ICDC GraphQL response did not contain a data object."
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


def unwrap_named_type(node: dict[str, Any] | None) -> str:
    current = node or {}
    seen = 0
    while current and seen < 10:
        name = current.get("name")
        if name:
            return str(name)
        current = current.get("ofType") or {}
        seen += 1
    return ""


def main() -> None:
    print("=" * 96)
    print("Paper 6 - probe ICDC parent-trial clinical metadata API")
    print("=" * 96)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print("")
    print("Purpose:")
    print("  Verify public ICDC study-level counts for COTC021 and COTC022.")
    print("  Discover the current GraphQL query surface for patient/case metadata.")
    print("  Download NO omics files and perform NO outcome/model analysis.")
    print("")

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    CONTRACT_DIR.mkdir(parents=True, exist_ok=True)

    studies_query = r"""
    query Paper6StudyInventory {
      studiesByProgram {
        program_id
        clinical_study_designation
        clinical_study_name
        clinical_study_type
        numberOfCases
      }
    }
    """

    print("Querying ICDC studiesByProgram...")
    study_payload = graphql_post(studies_query)
    studies = study_payload["data"].get("studiesByProgram", [])

    if not isinstance(studies, list):
        raise RuntimeError(
            "ICDC studiesByProgram did not return a list."
        )

    studies_df = pd.DataFrame(studies)

    required_columns = {
        "clinical_study_designation",
        "numberOfCases",
    }
    missing_columns = required_columns - set(studies_df.columns)
    if missing_columns:
        raise RuntimeError(
            "ICDC study inventory is missing expected columns: "
            + ", ".join(sorted(missing_columns))
        )

    studies_df.to_csv(OUT_STUDIES, index=False)

    parent_rows = studies_df[
        studies_df["clinical_study_designation"].isin(
            EXPECTED_PARENT_STUDIES
        )
    ].copy()

    observed_counts: dict[str, int] = {}
    for study_code in EXPECTED_PARENT_STUDIES:
        match = parent_rows[
            parent_rows["clinical_study_designation"].eq(study_code)
        ]

        if match.shape[0] != 1:
            raise RuntimeError(
                f"Expected exactly one ICDC study row for {study_code}; "
                f"observed {match.shape[0]}."
            )

        observed_counts[study_code] = int(
            pd.to_numeric(
                match.iloc[0]["numberOfCases"],
                errors="raise",
            )
        )

    observed_total = sum(observed_counts.values())

    print("")
    print("ICDC parent-trial study counts:")
    for study_code in sorted(observed_counts):
        expected = EXPECTED_PARENT_STUDIES[study_code]
        observed = observed_counts[study_code]
        print(
            f"  {study_code}: observed={observed}, expected_reference={expected}"
        )
    print(f"  combined: observed={observed_total}")

    count_match = (
        observed_counts == EXPECTED_PARENT_STUDIES
        and observed_total == EXPECTED_COMBINED_CASES
    )

    print("")
    print("Probing GraphQL schema for case/subject/clinical query fields...")

    introspection_query = r"""
    query Paper6QuerySurface {
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

    introspection_available = False
    candidate_query_fields: list[dict[str, Any]] = []
    schema_payload: dict[str, Any] = {}

    try:
        schema_payload = graphql_post(introspection_query)
        fields = (
            schema_payload
            .get("data", {})
            .get("__schema", {})
            .get("queryType", {})
            .get("fields", [])
        )

        if isinstance(fields, list):
            introspection_available = True

            for field in fields:
                name = str(field.get("name", ""))
                description = str(field.get("description") or "")
                searchable = (name + " " + description).lower()

                if not any(
                    token in searchable
                    for token in (
                        "case",
                        "subject",
                        "patient",
                        "clinical",
                        "study",
                        "diagnosis",
                        "demographic",
                        "treatment",
                        "outcome",
                    )
                ):
                    continue

                args = field.get("args") or []
                args_text = "; ".join(
                    f"{arg.get('name')}:{type_to_string(arg.get('type'))}"
                    for arg in args
                )

                candidate_query_fields.append(
                    {
                        "query_field": name,
                        "description": description,
                        "arguments": args_text,
                        "return_type": type_to_string(field.get("type")),
                        "return_named_type": unwrap_named_type(
                            field.get("type")
                        ),
                    }
                )

    except Exception as exc:
        schema_payload = {
            "introspection_error": (
                f"{type(exc).__name__}: {exc}"
            )
        }

    query_fields_df = pd.DataFrame(candidate_query_fields)
    query_fields_df.to_csv(
        OUT_QUERY_FIELDS,
        index=False,
    )

    OUT_SCHEMA_JSON.write_text(
        json.dumps(schema_payload, indent=2),
        encoding="utf-8",
    )

    if count_match and introspection_available:
        status = "PASS_READY_FOR_ICDC_CASE_QUERY_DESIGN"
        reason = (
            "ICDC exposes COTC021 and COTC022 with the expected public "
            "case counts (152 + 157 = 309), and GraphQL schema "
            "introspection succeeded. A case-level metadata query can now "
            "be frozen in the next script."
        )
    elif count_match:
        status = "PASS_STUDY_COUNTS_SCHEMA_INTROSPECTION_BLOCKED"
        reason = (
            "ICDC study-level parent counts match 152 + 157 = 309, but "
            "GraphQL schema introspection was unavailable. Case-level "
            "metadata should be obtained via the portal export or a "
            "documented query endpoint."
        )
    else:
        status = "WARN_ICDC_COUNTS_CHANGED"
        reason = (
            "ICDC is reachable, but current COTC021/COTC022 case counts "
            "do not equal the 152/157 reference counts. Inspect the study "
            "inventory before defining the parent audit population."
        )

    status_payload = {
        "script_version": SCRIPT_VERSION,
        "created_utc": utc_now_iso(),
        "graphql_endpoint": ICDC_GRAPHQL_URL,
        "status": status,
        "status_reason": reason,
        "expected_reference_counts": EXPECTED_PARENT_STUDIES,
        "observed_counts": observed_counts,
        "observed_combined_cases": observed_total,
        "expected_combined_cases": EXPECTED_COMBINED_CASES,
        "study_count_match": count_match,
        "graphql_introspection_available": introspection_available,
        "candidate_query_field_count": len(candidate_query_fields),
        "scientific_data_copied": False,
        "omics_downloaded": False,
        "model_fitting": False,
        "outcome_association_testing": False,
        "outputs": {
            "study_inventory": OUT_STUDIES.name,
            "candidate_query_fields": OUT_QUERY_FIELDS.name,
            "schema_probe": OUT_SCHEMA_JSON.name,
        },
    }

    OUT_STATUS.write_text(
        json.dumps(status_payload, indent=2),
        encoding="utf-8",
    )

    OUT_README.write_text(
        f"""Paper 6 ICDC parent-trial API probe
Script version: {SCRIPT_VERSION}

Status
------
{status}

Reason
------
{reason}

Why this stage exists
---------------------
The local Paper 4 repository does not contain a patient-level parent-trial
table large enough to audit selection into the 186-dog RNA cohort.

The ICDC publicly exposes the related COTC021 and COTC022 studies and clinical
metadata through its data commons/API. This probe verifies the live study
counts and discovers the current GraphQL query surface before any case-level
download logic is frozen.

No omics files are downloaded. No survival, treatment-effect, or molecular
model is fit.
""",
        encoding="utf-8",
    )

    print("")
    print("=" * 96)
    print(f"ICDC parent-trial API probe: {status}")
    print("=" * 96)
    print(reason)
    print("")
    print(f"GraphQL introspection available: {introspection_available}")
    print(
        "Candidate case/subject/clinical query fields discovered: "
        f"{len(candidate_query_fields)}"
    )

    if not query_fields_df.empty:
        print("")
        print("Top candidate query fields:")
        for _, row in query_fields_df.head(20).iterrows():
            print(
                f"  {row['query_field']}({row['arguments']}) "
                f"-> {row['return_type']}"
            )

    print("")
    print("Saved:")
    for path in (
        OUT_STUDIES,
        OUT_QUERY_FIELDS,
        OUT_SCHEMA_JSON,
        OUT_STATUS,
        OUT_README,
    ):
        print(f"  {path}")

    print("")
    print("Next:")
    if introspection_available:
        print(
            "  Use the discovered live query field names to write a tiny "
            "case-level COTC021/COTC022 clinical metadata fetcher."
        )
    else:
        print(
            "  Export COTC021/COTC022 clinical metadata JSON from ICDC "
            "or use its documented REST/API route; no omics download is needed."
        )
    print("Done.")


if __name__ == "__main__":
    main()
