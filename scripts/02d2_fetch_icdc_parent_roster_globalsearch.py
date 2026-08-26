from __future__ import annotations

from pathlib import Path
import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests


SCRIPT_VERSION = "02d2-fetch-icdc-parent-roster-globalsearch-v1-no-cli"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_DIR = PROJECT_ROOT / "contracts"
MANIFEST_DIR = PROJECT_ROOT / "manifests"

ICDC_GRAPHQL_URL = "https://caninecommons.cancer.gov/v1/graphql/"

PROBE_STATUS = CONTRACT_DIR / "02c_icdc_parent_trial_api_status.json"
SELECTED_MAPPING = MANIFEST_DIR / "02a_dog2_selected186_authoritative_id_mapping.csv"
SELECTED_LOCK = CONTRACT_DIR / "02a_dog2_authoritative_id_mapping_lock.json"

OUT_PARENT = MANIFEST_DIR / "02d2_icdc_parent_trial_roster.csv"
OUT_OVERLAP = MANIFEST_DIR / "02d2_icdc_parent_vs_rna186_overlap.csv"
OUT_RAW = MANIFEST_DIR / "02d2_icdc_parent_trial_globalsearch_raw.json"
OUT_STATUS = CONTRACT_DIR / "02d2_icdc_parent_trial_roster_status.json"
OUT_README = MANIFEST_DIR / "02d2_icdc_parent_trial_roster_README.txt"

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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def graphql_post(query: str, variables: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(
        ICDC_GRAPHQL_URL,
        json={
            "query": query,
            "variables": variables,
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "paper6-cross-species-osteosarcoma/02d2",
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


def verify_prerequisites() -> set[str]:
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

    observed_counts = {
        str(key): int(value)
        for key, value in probe.get("observed_counts", {}).items()
    }

    if observed_counts != EXPECTED_COUNTS:
        raise RuntimeError(
            f"02c ICDC study counts changed: {observed_counts}"
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

    observed_mapping_hash = sha256_file(SELECTED_MAPPING)
    expected_mapping_hash = selected_lock.get("output_mapping_sha256")

    if observed_mapping_hash != expected_mapping_hash:
        raise RuntimeError(
            "02a selected-subject mapping hash mismatch."
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
            f"Frozen selected COTC subjects={len(selected_ids)}; "
            f"expected {EXPECTED_SELECTED_N}."
        )

    return selected_ids


def fetch_global_search_cases(
    study_code: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # This is the same public query pattern used by the ICDC frontend search
    # page. We search by study code, then strictly retain only case rows whose
    # returned clinical_study_designation equals the requested study.
    query = r"""
    query Paper6GlobalSearchCases(
      $input: String
      $first: Int
      $offset: Int
    ) {
      globalSearch(
        input: $input
        first: $first
        offset: $offset
      ) {
        cases {
          type
          case_id
          program_name
          clinical_study_designation
          disease_term
          breed
        }
      }
    }
    """

    variables = {
        "input": study_code,
        "first": MAX_FIRST,
        "offset": 0,
    }

    payload = graphql_post(
        query=query,
        variables=variables,
    )

    cases = (
        payload
        .get("data", {})
        .get("globalSearch", {})
        .get("cases", [])
    )

    if not isinstance(cases, list):
        raise RuntimeError(
            f"globalSearch cases for {study_code} did not return a list."
        )

    filtered = [
        row
        for row in cases
        if isinstance(row, dict)
        and clean_text(
            row.get("clinical_study_designation")
        ).upper() == study_code
    ]

    return filtered, payload


def main() -> None:
    print("=" * 96)
    print("Paper 6 - fetch ICDC parent-trial roster via public globalSearch")
    print("=" * 96)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print("")
    print("Purpose:")
    print("  Fetch the official ICDC case roster for COTC021 and COTC022.")
    print("  Use the public globalSearch case query used by the ICDC frontend.")
    print("  Require exact study designation, exact expected counts, unique case IDs,")
    print("  and exact overlap auditing against the frozen RNA186 COTC subjects.")
    print("")
    print("No omics download, no survival analysis, and no treatment-effect model.")
    print("")

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    CONTRACT_DIR.mkdir(parents=True, exist_ok=True)

    selected_ids = verify_prerequisites()

    print("02a selected-subject lock: PASS")
    print("02c ICDC study-count probe: PASS")
    print(f"Frozen RNA186 COTC subjects: {len(selected_ids)}")
    print("")

    all_rows = []
    raw_payloads: dict[str, Any] = {}
    observed_counts: dict[str, int] = {}

    for study_code, expected_n in EXPECTED_COUNTS.items():
        print(f"Querying globalSearch cases for {study_code}...")

        cases, payload = fetch_global_search_cases(
            study_code=study_code,
        )
        raw_payloads[study_code] = payload

        frame = pd.DataFrame(cases)

        if frame.empty:
            observed_counts[study_code] = 0
            continue

        required_columns = {
            "case_id",
            "clinical_study_designation",
        }

        missing = required_columns - set(frame.columns)

        if missing:
            raise RuntimeError(
                f"{study_code} globalSearch cases missing columns: "
                + ", ".join(sorted(missing))
            )

        frame["case_id"] = frame["case_id"].map(clean_text)
        frame["cotc_subject_id"] = frame["case_id"].map(
            normalize_cotc
        )
        frame["__requested_study__"] = study_code

        observed_counts[study_code] = int(frame.shape[0])

        print(
            f"  returned and study-filtered: {frame.shape[0]} "
            f"(expected {expected_n})"
        )
        print(
            f"  unique case_id: {frame['case_id'].nunique()}"
        )
        print(
            f"  parseable unique COTC IDs: "
            f"{frame['cotc_subject_id'].replace('', pd.NA).nunique(dropna=True)}"
        )

        all_rows.append(frame)

    if all_rows:
        parent = pd.concat(
            all_rows,
            ignore_index=True,
            sort=False,
        )
    else:
        parent = pd.DataFrame()

    OUT_RAW.write_text(
        json.dumps(
            raw_payloads,
            indent=2,
        ),
        encoding="utf-8",
    )

    count_match = observed_counts == EXPECTED_COUNTS

    if parent.empty:
        unique_case_ids = 0
        unique_cotc = 0
        duplicate_case_rows = 0
        duplicate_cotc_rows = 0
        parent_ids: set[str] = set()
    else:
        unique_case_ids = int(
            parent["case_id"].nunique()
        )
        unique_cotc = int(
            parent["cotc_subject_id"]
            .replace("", pd.NA)
            .nunique(dropna=True)
        )
        duplicate_case_rows = int(
            parent["case_id"].duplicated(
                keep=False
            ).sum()
        )
        duplicate_cotc_rows = int(
            parent["cotc_subject_id"]
            .replace("", pd.NA)
            .dropna()
            .duplicated(keep=False)
            .sum()
        )
        parent_ids = set(
            parent["cotc_subject_id"]
        )
        parent_ids.discard("")

    overlap = selected_ids & parent_ids
    selected_missing = selected_ids - parent_ids
    parent_not_selected = parent_ids - selected_ids

    parent["selected_rna186"] = (
        parent["cotc_subject_id"].isin(selected_ids)
        if not parent.empty
        else pd.Series(dtype=bool)
    )

    parent.to_csv(
        OUT_PARENT,
        index=False,
    )

    overlap_rows = []

    for cotc_id in sorted(selected_ids):
        overlap_rows.append(
            {
                "cotc_subject_id": cotc_id,
                "in_frozen_rna186": True,
                "in_icdc_parent309": cotc_id in parent_ids,
            }
        )

    for cotc_id in sorted(parent_not_selected):
        overlap_rows.append(
            {
                "cotc_subject_id": cotc_id,
                "in_frozen_rna186": False,
                "in_icdc_parent309": True,
            }
        )

    overlap_df = pd.DataFrame(overlap_rows)

    overlap_df.to_csv(
        OUT_OVERLAP,
        index=False,
    )

    checks = {
        "study_counts_match_152_157": count_match,
        "combined_rows_309": int(parent.shape[0]) == EXPECTED_TOTAL,
        "unique_case_ids_309": unique_case_ids == EXPECTED_TOTAL,
        "parseable_unique_cotc_309": unique_cotc == EXPECTED_TOTAL,
        "duplicate_case_rows_zero": duplicate_case_rows == 0,
        "duplicate_cotc_rows_zero": duplicate_cotc_rows == 0,
        "all_rna186_found_in_parent": len(overlap) == EXPECTED_SELECTED_N,
        "rna186_missing_zero": len(selected_missing) == 0,
        "parent_not_selected_123": len(parent_not_selected) == (
            EXPECTED_TOTAL - EXPECTED_SELECTED_N
        ),
    }

    passed = all(checks.values())

    if passed:
        status = "PASS_PARENT309_ROSTER_LOCK_READY"
        reason = (
            "ICDC globalSearch returned exactly 152 COTC021 and 157 COTC022 "
            "cases; all 309 case IDs map uniquely to COTC subjects, all 186 "
            "frozen RNA-profiled DOG2 subjects are contained in the parent "
            "roster, and exactly 123 parent cases are not in RNA186."
        )
    else:
        status = "FAIL_PARENT_ROSTER_INTEGRITY"
        failed = [
            key
            for key, value in checks.items()
            if not value
        ]
        reason = (
            "One or more parent-roster integrity checks failed: "
            + ", ".join(failed)
        )

    status_payload = {
        "script_version": SCRIPT_VERSION,
        "created_utc": utc_now_iso(),
        "graphql_endpoint": ICDC_GRAPHQL_URL,
        "status": status,
        "status_reason": reason,
        "expected_counts": EXPECTED_COUNTS,
        "observed_counts": observed_counts,
        "expected_total": EXPECTED_TOTAL,
        "observed_total_rows": int(parent.shape[0]),
        "unique_case_ids": unique_case_ids,
        "unique_cotc_subject_ids": unique_cotc,
        "selected_rna186_overlap": len(overlap),
        "selected_rna186_missing": len(selected_missing),
        "parent_not_selected": len(parent_not_selected),
        "checks": checks,
        "scientific_data_copied": False,
        "omics_downloaded": False,
        "model_fitting": False,
        "outcome_association_testing": False,
        "parent_roster_sha256": sha256_file(OUT_PARENT),
        "outputs": {
            "parent_roster": OUT_PARENT.name,
            "overlap_audit": OUT_OVERLAP.name,
            "raw_graphql": OUT_RAW.name,
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
        f"""Paper 6 ICDC COTC021/COTC022 parent roster
Script version: {SCRIPT_VERSION}

Status
------
{status}

Reason
------
{reason}

Method
------
The ICDC public frontend uses globalSearch(input, first, offset) to retrieve
case search results including case_id and clinical_study_designation. This
script queries each trial study code, strictly filters returned case rows to
the requested study designation, and requires the published/public ICDC counts
of 152 COTC021 and 157 COTC022 cases.

The parent roster is accepted only if:
- 309 total rows are returned;
- all 309 case IDs are unique;
- all 309 case IDs parse to unique COTC subject identifiers;
- all frozen 186 RNA-profiled DOG2 subjects occur in that parent roster; and
- exactly 123 parent subjects remain outside RNA186.

No omics data are downloaded and no outcome model is fit.
""",
        encoding="utf-8",
    )

    print("")
    print("=" * 96)
    print(f"ICDC parent roster: {status}")
    print("=" * 96)
    print(reason)
    print("")
    print("Roster summary:")
    print(f"  total rows: {parent.shape[0]}")
    print(f"  unique case IDs: {unique_case_ids}")
    print(f"  unique COTC subjects: {unique_cotc}")
    print(
        f"  RNA186 found in parent: "
        f"{len(overlap)}/{EXPECTED_SELECTED_N}"
    )
    print(
        f"  RNA186 missing from parent: "
        f"{len(selected_missing)}"
    )
    print(
        f"  parent cases not in RNA186: "
        f"{len(parent_not_selected)}"
    )
    print("")
    print("Saved:")
    for path in (
        OUT_PARENT,
        OUT_OVERLAP,
        OUT_RAW,
        OUT_STATUS,
        OUT_README,
    ):
        print(f"  {path}")

    if not passed:
        raise RuntimeError(
            "ICDC parent roster integrity checks failed. "
            "Do not proceed to selection-balance analysis."
        )

    print("")
    print("Next:")
    print(
        "  Enrich these 309 case IDs with the ICDC clinical/demographic "
        "fields required for treatment-arm and baseline-selection auditing."
    )
    print("Done.")


if __name__ == "__main__":
    main()
