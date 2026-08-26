from __future__ import annotations

from pathlib import Path
import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests


SCRIPT_VERSION = "02e-fetch-icdc-parent309-baseline-clinical-v1-no-cli"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_DIR = PROJECT_ROOT / "contracts"
MANIFEST_DIR = PROJECT_ROOT / "manifests"

ICDC_GRAPHQL_URL = "https://caninecommons.cancer.gov/v1/graphql/"

ROSTER = MANIFEST_DIR / "02d2_icdc_parent_trial_roster.csv"
ROSTER_STATUS = CONTRACT_DIR / "02d2_icdc_parent_trial_roster_status.json"

OUT_CLINICAL = MANIFEST_DIR / "02e_icdc_parent309_baseline_clinical.csv"
OUT_COMPLETENESS = MANIFEST_DIR / "02e_icdc_parent309_field_completeness.csv"
OUT_FETCH_AUDIT = MANIFEST_DIR / "02e_icdc_parent309_case_fetch_audit.csv"
OUT_RAW = MANIFEST_DIR / "02e_icdc_parent309_baseline_raw.json"
OUT_STATUS = CONTRACT_DIR / "02e_icdc_parent309_baseline_clinical_status.json"
OUT_README = MANIFEST_DIR / "02e_icdc_parent309_baseline_clinical_README.txt"

EXPECTED_TOTAL = 309
EXPECTED_COUNTS = {
    "COTC021": 152,
    "COTC022": 157,
}
EXPECTED_SELECTED_N = 186

REQUEST_TIMEOUT_SECONDS = 60
MAX_ATTEMPTS_PER_CASE = 4
RETRY_SLEEP_SECONDS = 1.0

# This stage intentionally excludes outcome/response fields. It is for
# selection auditing only.
FORBIDDEN_OUTCOME_TOKENS = (
    "best_response",
    "survival",
    "death",
    "deceased",
    "event",
    "relapse",
    "recurrence",
    "progression",
    "disease_free",
    "overall_survival",
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


def verify_prerequisites() -> pd.DataFrame:
    if not ROSTER.exists() or not ROSTER_STATUS.exists():
        raise FileNotFoundError(
            "Run scripts/02d2_fetch_icdc_parent_roster_globalsearch.py first."
        )

    status = json.loads(
        ROSTER_STATUS.read_text(encoding="utf-8")
    )

    if status.get("status") != "PASS_PARENT309_ROSTER_LOCK_READY":
        raise RuntimeError(
            "02d2 parent-roster status is not PASS_PARENT309_ROSTER_LOCK_READY."
        )

    expected_hash = status.get("parent_roster_sha256")
    observed_hash = sha256_file(ROSTER)

    if expected_hash != observed_hash:
        raise RuntimeError(
            "02d2 parent-roster hash mismatch."
        )

    roster = pd.read_csv(
        ROSTER,
        dtype=str,
        low_memory=False,
    ).fillna("")

    if roster.shape[0] != EXPECTED_TOTAL:
        raise RuntimeError(
            f"02d2 roster rows={roster.shape[0]}; expected {EXPECTED_TOTAL}."
        )

    if roster["case_id"].nunique() != EXPECTED_TOTAL:
        raise RuntimeError(
            "02d2 roster case_id is not unique for all 309 cases."
        )

    selected = roster["selected_rna186"].astype(str).str.lower().isin(
        {"true", "1", "yes"}
    )

    if int(selected.sum()) != EXPECTED_SELECTED_N:
        raise RuntimeError(
            f"02d2 selected_rna186 count={int(selected.sum())}; "
            f"expected {EXPECTED_SELECTED_N}."
        )

    observed_counts = (
        roster["clinical_study_designation"]
        .value_counts()
        .to_dict()
    )

    observed_counts = {
        str(key): int(value)
        for key, value in observed_counts.items()
    }

    if observed_counts != EXPECTED_COUNTS:
        raise RuntimeError(
            f"02d2 study counts changed: {observed_counts}"
        )

    return roster


def graphql_case_query() -> str:
    # This field set mirrors the public ICDC case-details query, but
    # intentionally excludes response/outcome fields.
    return r"""
    query Paper6CaseBaseline02e($case_id: String!) {
      case(case_id: $case_id) {
        case_id
        patient_id
        study {
          clinical_study_name
          clinical_study_designation
          accession_id
          program {
            program_acronym
          }
        }
        demographic {
          breed
          sex
          patient_age_at_enrollment
          neutered_indicator
          weight
          additional_breed_detail
        }
        cohort {
          cohort_description
          study_arm {
            arm
            ctep_treatment_assignment_code
          }
        }
        enrollment {
          site_short_name
          patient_subgroup
        }
        diagnoses {
          disease_term
          stage_of_disease
          primary_disease_site
          histological_grade
          histology_cytopathology
        }
      }
    }
    """


def post_case_query(
    session: requests.Session,
    case_id: str,
) -> dict[str, Any]:
    query = graphql_case_query()

    last_error = None

    for attempt in range(1, MAX_ATTEMPTS_PER_CASE + 1):
        try:
            response = session.post(
                ICDC_GRAPHQL_URL,
                json={
                    "query": query,
                    "variables": {
                        "case_id": case_id,
                    },
                },
                timeout=REQUEST_TIMEOUT_SECONDS,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "User-Agent": "paper6-cross-species-osteosarcoma/02e",
                },
            )
            response.raise_for_status()

            payload = response.json()

            if payload.get("errors"):
                raise RuntimeError(
                    json.dumps(payload["errors"], indent=2)
                )

            case = (
                payload
                .get("data", {})
                .get("case")
            )

            if not isinstance(case, dict):
                raise RuntimeError(
                    f"ICDC case query returned no case object for {case_id}."
                )

            return {
                "payload": payload,
                "attempts": attempt,
            }

        except Exception as exc:
            last_error = exc

            if attempt < MAX_ATTEMPTS_PER_CASE:
                time.sleep(
                    RETRY_SLEEP_SECONDS * attempt
                )

    raise RuntimeError(
        f"Failed to fetch {case_id} after "
        f"{MAX_ATTEMPTS_PER_CASE} attempts: "
        f"{type(last_error).__name__}: {last_error}"
    )


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def scalar_values_from_nested(
    value: Any,
    key_path: tuple[str, ...],
) -> list[str]:
    current = as_list(value)

    for key in key_path:
        next_values = []

        for item in current:
            if isinstance(item, dict):
                candidate = item.get(key)

                if isinstance(candidate, list):
                    next_values.extend(candidate)
                elif candidate is not None:
                    next_values.append(candidate)

        current = next_values

    cleaned = []

    for item in current:
        if isinstance(item, dict):
            continue

        text = clean_text(item)

        if text:
            cleaned.append(text)

    return cleaned


def join_unique(values: list[str]) -> str:
    seen = []
    seen_set = set()

    for value in values:
        text = clean_text(value)

        if text and text not in seen_set:
            seen.append(text)
            seen_set.add(text)

    return " || ".join(seen)


def first_scalar(
    value: Any,
    key: str,
) -> str:
    if isinstance(value, dict):
        return clean_text(
            value.get(key)
        )

    if isinstance(value, list):
        values = []

        for item in value:
            if isinstance(item, dict):
                text = clean_text(
                    item.get(key)
                )
                if text:
                    values.append(text)

        return join_unique(values)

    return ""


def flatten_case(case: dict[str, Any]) -> dict[str, Any]:
    study = case.get("study")
    demographic = case.get("demographic")
    cohort = case.get("cohort")
    enrollment = case.get("enrollment")
    diagnoses = case.get("diagnoses")

    row = {
        "case_id": clean_text(case.get("case_id")),
        "patient_id": clean_text(case.get("patient_id")),
        "study_clinical_study_name": first_scalar(
            study,
            "clinical_study_name",
        ),
        "study_clinical_study_designation": first_scalar(
            study,
            "clinical_study_designation",
        ),
        "study_accession_id": first_scalar(
            study,
            "accession_id",
        ),
        "study_program_acronym": join_unique(
            scalar_values_from_nested(
                study,
                ("program", "program_acronym"),
            )
        ),
        "demographic_breed": first_scalar(
            demographic,
            "breed",
        ),
        "demographic_sex": first_scalar(
            demographic,
            "sex",
        ),
        "demographic_patient_age_at_enrollment": first_scalar(
            demographic,
            "patient_age_at_enrollment",
        ),
        "demographic_neutered_indicator": first_scalar(
            demographic,
            "neutered_indicator",
        ),
        "demographic_weight": first_scalar(
            demographic,
            "weight",
        ),
        "demographic_additional_breed_detail": first_scalar(
            demographic,
            "additional_breed_detail",
        ),
        "cohort_description": first_scalar(
            cohort,
            "cohort_description",
        ),
        "cohort_study_arm": join_unique(
            scalar_values_from_nested(
                cohort,
                ("study_arm", "arm"),
            )
        ),
        "cohort_ctep_treatment_assignment_code": join_unique(
            scalar_values_from_nested(
                cohort,
                (
                    "study_arm",
                    "ctep_treatment_assignment_code",
                ),
            )
        ),
        "enrollment_site_short_name": first_scalar(
            enrollment,
            "site_short_name",
        ),
        "enrollment_patient_subgroup": first_scalar(
            enrollment,
            "patient_subgroup",
        ),
        "diagnosis_disease_term": join_unique(
            scalar_values_from_nested(
                diagnoses,
                ("disease_term",),
            )
        ),
        "diagnosis_stage_of_disease": join_unique(
            scalar_values_from_nested(
                diagnoses,
                ("stage_of_disease",),
            )
        ),
        "diagnosis_primary_disease_site": join_unique(
            scalar_values_from_nested(
                diagnoses,
                ("primary_disease_site",),
            )
        ),
        "diagnosis_histological_grade": join_unique(
            scalar_values_from_nested(
                diagnoses,
                ("histological_grade",),
            )
        ),
        "diagnosis_histology_cytopathology": join_unique(
            scalar_values_from_nested(
                diagnoses,
                ("histology_cytopathology",),
            )
        ),
    }

    return row


def audit_forbidden_columns(frame: pd.DataFrame) -> list[str]:
    violations = []

    for column in frame.columns:
        lower = column.lower()

        if any(
            token in lower
            for token in FORBIDDEN_OUTCOME_TOKENS
        ):
            violations.append(column)

    return violations


def build_completeness(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for column in frame.columns:
        values = frame[column].map(clean_text)
        nonempty = values.ne("")

        rows.append(
            {
                "field": column,
                "n_rows": int(frame.shape[0]),
                "nonmissing_n": int(nonempty.sum()),
                "missing_n": int((~nonempty).sum()),
                "nonmissing_fraction": float(nonempty.mean()),
                "unique_nonmissing_n": int(
                    values[nonempty].nunique()
                ),
                "example_values": " | ".join(
                    values[nonempty]
                    .drop_duplicates()
                    .head(8)
                    .tolist()
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    print("=" * 100)
    print("Paper 6 - fetch ICDC parent309 baseline clinical metadata")
    print("=" * 100)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print("")
    print("Purpose:")
    print("  Fetch baseline/design metadata for the locked 309-case parent roster.")
    print("  Use the public ICDC case-details GraphQL object one case at a time.")
    print("  Preserve treatment arm, enrollment, demographic, and baseline diagnosis")
    print("  variables needed for the later 309 -> RNA186 selection audit.")
    print("")
    print("Outcome/response fields are intentionally excluded.")
    print("No omics data are downloaded and no model is fit.")
    print("")

    MANIFEST_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    CONTRACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    roster = verify_prerequisites()

    print("02d2 parent309 roster lock: PASS")
    print(f"Cases to fetch: {roster.shape[0]}")
    print("")

    session = requests.Session()

    raw_payloads = {}
    flat_rows = []
    fetch_rows = []

    case_ids = roster["case_id"].tolist()

    for index, case_id in enumerate(
        case_ids,
        start=1,
    ):
        if (
            index == 1
            or index % 25 == 0
            or index == len(case_ids)
        ):
            print(
                f"Fetching case {index}/{len(case_ids)}: {case_id}"
            )

        try:
            result = post_case_query(
                session=session,
                case_id=case_id,
            )

            payload = result["payload"]
            case = (
                payload
                .get("data", {})
                .get("case")
            )

            raw_payloads[case_id] = payload
            flat = flatten_case(case)
            flat_rows.append(flat)

            fetch_rows.append(
                {
                    "case_id": case_id,
                    "status": "PASS",
                    "attempts": int(result["attempts"]),
                    "error": "",
                }
            )

        except Exception as exc:
            fetch_rows.append(
                {
                    "case_id": case_id,
                    "status": "FAIL",
                    "attempts": MAX_ATTEMPTS_PER_CASE,
                    "error": (
                        f"{type(exc).__name__}: {exc}"
                    ),
                }
            )

    fetch_audit = pd.DataFrame(
        fetch_rows
    )

    failed = fetch_audit[
        fetch_audit["status"].ne("PASS")
    ]

    fetch_audit.to_csv(
        OUT_FETCH_AUDIT,
        index=False,
    )

    OUT_RAW.write_text(
        json.dumps(
            raw_payloads,
            indent=2,
        ),
        encoding="utf-8",
    )

    if not failed.empty:
        status = "FAIL_CASE_FETCH_INCOMPLETE"
        reason = (
            f"{failed.shape[0]} of {EXPECTED_TOTAL} ICDC case-detail "
            "requests failed after retries."
        )

        pd.DataFrame(
            flat_rows
        ).to_csv(
            OUT_CLINICAL,
            index=False,
        )

        pd.DataFrame().to_csv(
            OUT_COMPLETENESS,
            index=False,
        )

    else:
        clinical = pd.DataFrame(
            flat_rows
        )

        if clinical.shape[0] != EXPECTED_TOTAL:
            raise RuntimeError(
                f"Flattened clinical rows={clinical.shape[0]}; "
                f"expected {EXPECTED_TOTAL}."
            )

        if clinical["case_id"].nunique() != EXPECTED_TOTAL:
            raise RuntimeError(
                "Flattened clinical case_id is not unique for all 309 cases."
            )

        roster_small = roster[
            [
                "case_id",
                "clinical_study_designation",
                "cotc_subject_id",
                "selected_rna186",
            ]
        ].copy()

        clinical = roster_small.merge(
            clinical,
            on="case_id",
            how="left",
            validate="one_to_one",
        )

        study_match = (
            clinical[
                "clinical_study_designation"
            ].map(clean_text)
            == clinical[
                "study_clinical_study_designation"
            ].map(clean_text)
        )

        case_match = (
            clinical["case_id"].map(clean_text)
            == clinical["case_id"].map(clean_text)
        )

        forbidden_columns = audit_forbidden_columns(
            clinical
        )

        checks = {
            "rows_309": int(clinical.shape[0]) == EXPECTED_TOTAL,
            "unique_case_ids_309": (
                clinical["case_id"].nunique()
                == EXPECTED_TOTAL
            ),
            "study_designation_matches_roster_all_309": (
                bool(study_match.all())
            ),
            "forbidden_outcome_columns_zero": (
                len(forbidden_columns) == 0
            ),
        }

        clinical.to_csv(
            OUT_CLINICAL,
            index=False,
        )

        completeness = build_completeness(
            clinical
        )
        completeness.to_csv(
            OUT_COMPLETENESS,
            index=False,
        )

        passed = all(
            checks.values()
        )

        if passed:
            status = "PASS_PARENT309_BASELINE_CLINICAL_LOCK_READY"
            reason = (
                "All 309 locked ICDC parent cases were fetched successfully, "
                "their study designations match the frozen parent roster, and "
                "the baseline clinical snapshot contains no intentionally "
                "excluded outcome/response fields."
            )
        else:
            failed_checks = [
                key
                for key, value in checks.items()
                if not value
            ]
            status = "FAIL_BASELINE_CLINICAL_INTEGRITY"
            reason = (
                "One or more baseline-clinical integrity checks failed: "
                + ", ".join(failed_checks)
            )

    if status == "FAIL_CASE_FETCH_INCOMPLETE":
        checks = {
            "rows_309": False,
            "unique_case_ids_309": False,
            "study_designation_matches_roster_all_309": False,
            "forbidden_outcome_columns_zero": True,
        }
        forbidden_columns = []

    status_payload = {
        "script_version": SCRIPT_VERSION,
        "created_utc": utc_now_iso(),
        "graphql_endpoint": ICDC_GRAPHQL_URL,
        "status": status,
        "status_reason": reason,
        "parent_roster_sha256": sha256_file(
            ROSTER
        ),
        "expected_parent_n": EXPECTED_TOTAL,
        "fetched_pass_n": int(
            fetch_audit["status"].eq("PASS").sum()
        ),
        "fetched_fail_n": int(
            fetch_audit["status"].ne("PASS").sum()
        ),
        "checks": checks,
        "forbidden_outcome_columns_detected": forbidden_columns,
        "outcome_fields_requested": False,
        "omics_downloaded": False,
        "model_fitting": False,
        "outcome_association_testing": False,
        "baseline_clinical_sha256": (
            sha256_file(OUT_CLINICAL)
            if OUT_CLINICAL.exists()
            else None
        ),
        "outputs": {
            "baseline_clinical": OUT_CLINICAL.name,
            "field_completeness": OUT_COMPLETENESS.name,
            "case_fetch_audit": OUT_FETCH_AUDIT.name,
            "raw_case_details": OUT_RAW.name,
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
        f"""Paper 6 ICDC parent309 baseline clinical snapshot
Script version: {SCRIPT_VERSION}

Status
------
{status}

Reason
------
{reason}

Scope
-----
This stage fetches only baseline/design metadata needed to audit selection from
the 309-case public ICDC parent population into the 186-dog RNA cohort:
study identity, demographic variables, study arm/cohort, enrollment site and
subgroup, and baseline diagnosis/site/pathology fields.

Outcome/response variables are deliberately not requested. In particular, the
public ICDC case-details object also exposes response-related information, but
this Paper 6 stage does not retrieve it.

No omics files are downloaded and no predictive/causal model is fit.
""",
        encoding="utf-8",
    )

    print("")
    print("=" * 100)
    print(f"ICDC parent309 baseline clinical fetch: {status}")
    print("=" * 100)
    print(reason)
    print("")
    print("Fetch summary:")
    print(
        f"  PASS: "
        f"{fetch_audit['status'].eq('PASS').sum()}/{EXPECTED_TOTAL}"
    )
    print(
        f"  FAIL: "
        f"{fetch_audit['status'].ne('PASS').sum()}/{EXPECTED_TOTAL}"
    )

    if OUT_CLINICAL.exists():
        clinical_preview = pd.read_csv(
            OUT_CLINICAL,
            dtype=str,
            low_memory=False,
        ).fillna("")

        print(
            f"  saved clinical rows: {clinical_preview.shape[0]}"
        )
        print(
            f"  saved clinical columns: {clinical_preview.shape[1]}"
        )

        for field in (
            "cohort_study_arm",
            "demographic_patient_age_at_enrollment",
            "demographic_weight",
            "demographic_sex",
            "demographic_breed",
            "enrollment_site_short_name",
            "enrollment_patient_subgroup",
            "diagnosis_primary_disease_site",
        ):
            if field in clinical_preview.columns:
                nonmissing = int(
                    clinical_preview[field]
                    .map(clean_text)
                    .ne("")
                    .sum()
                )
                print(
                    f"  {field}: nonmissing={nonmissing}/{EXPECTED_TOTAL}"
                )

    print("")
    print("Saved:")
    for path in (
        OUT_CLINICAL,
        OUT_COMPLETENESS,
        OUT_FETCH_AUDIT,
        OUT_RAW,
        OUT_STATUS,
        OUT_README,
    ):
        print(f"  {path}")

    if not status.startswith("PASS_"):
        raise RuntimeError(
            "ICDC parent309 baseline clinical snapshot did not pass. "
            "Do not proceed to 309->186 selection-balance inference."
        )

    print("")
    print("Next:")
    print(
        "  Run a dedicated 309 -> RNA186 selection-balance audit using only "
        "this frozen baseline snapshot; keep outcome fields isolated."
    )
    print("Done.")


if __name__ == "__main__":
    main()
