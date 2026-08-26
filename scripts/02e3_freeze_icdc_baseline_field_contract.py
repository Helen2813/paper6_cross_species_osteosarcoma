#!/usr/bin/env python3
"""
Paper 6 - freeze ICDC baseline/linkage field contract from the 02e2 schema inventory.

IMPORTANT
---------
This script is OFFLINE with respect to ICDC.

It:
  * reads only the schema/provenance artifacts produced by 02e2;
  * verifies the frozen 02e2 artifact hashes;
  * freezes an explicit field-level allow/deny contract;
  * writes a GraphQL selection fragment for a later baseline-only probe.

It does NOT:
  * call the ICDC API;
  * read any case/patient values;
  * read outcome/response/follow-up values;
  * read treatment administration values;
  * read omics values;
  * download any data files.

Scientific rationale
--------------------
Automatic keyword labels from 02e2 are useful diagnostics but are not sufficient
to decide admissibility. For example, the ICDC `off_study` object contains the
word "study" but is post-baseline. Conversely, some baseline diagnosis fields
are scientifically admissible even if their names contain temporal language.

Therefore this script freezes a deliberately conservative, exact allow-list
before any production case values are queried.

The contract is designed for the current pre-outcome Paper 6 stage:
  * stable subject/study linkage;
  * baseline demographic covariates;
  * baseline diagnosis characterization;
  * enrollment-site characterization;
  * arm/cohort identity only;
  * sample ID linkage only.

Outcome, response, follow-up, longitudinal treatment, adverse-event,
metastatic-origin/sample-chronology, necropsy, and free-text clinical fields
remain blocked.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


SCRIPT_VERSION = "02e3-freeze-icdc-baseline-field-contract-v1-no-cli"
EXPECTED_02E2_PREFIX = "02e2-inventory-icdc-case-schema-"
EXPECTED_ENDPOINT = "https://caninecommons.cancer.gov/v1/graphql/"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_ROOT / "results" / "icdc_case_schema" / "02e2"
INPUT_SUMMARY = INPUT_DIR / "summary.json"
INPUT_QUERY_FIELD = INPUT_DIR / "query_field_case.json"
INPUT_TYPE_GRAPH = INPUT_DIR / "case_type_graph.json"
INPUT_FIELDS_TSV = INPUT_DIR / "case_schema_fields.tsv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "icdc_baseline_contract" / "02e3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONTRACT_JSON = OUTPUT_DIR / "baseline_field_contract.json"
CONTRACT_TSV = OUTPUT_DIR / "baseline_field_contract.tsv"
SELECTION_GRAPHQL = OUTPUT_DIR / "baseline_selection.graphql"
SUMMARY_JSON = OUTPUT_DIR / "summary.json"


# ---------------------------------------------------------------------------
# Exact pre-value policy.
#
# Required = contract must find this exact field in the 02e2 production schema.
# Optional = include if present; absence is recorded but does not fail.
#
# No free-text description fields are allowed at this stage.
# No dates are allowed at this stage.
# ---------------------------------------------------------------------------

POLICY: Dict[str, Dict[str, Any]] = {
    "case": {
        "relation_from_case": None,
        "required": {
            "case_id": "LINKAGE_ID",
            "patient_id": "LINKAGE_ID",
        },
        "optional": {},
    },
    "canine_individual": {
        "relation_from_case": "canine_individual",
        "required": {},
        "optional": {
            "canine_individual_id": "LINKAGE_ID",
        },
    },
    "demographic": {
        "relation_from_case": "demographic",
        "required": {
            "breed": "BASELINE_DEMOGRAPHIC",
            "patient_age_at_enrollment": "BASELINE_DEMOGRAPHIC",
            "sex": "BASELINE_DEMOGRAPHIC",
        },
        "optional": {
            "demographic_record_id": "LINKAGE_ID",
            "additional_breed_detail": "BASELINE_DEMOGRAPHIC",
            "weight": "BASELINE_DEMOGRAPHIC",
            "neutered_indicator": "BASELINE_DEMOGRAPHIC",
        },
    },
    "diagnosis": {
        "relation_from_case": "diagnoses",
        "required": {
            "disease_term": "BASELINE_DIAGNOSIS",
            "primary_disease_site": "BASELINE_DIAGNOSIS",
        },
        "optional": {
            "diagnosis_record_id": "LINKAGE_ID",
            "stage_of_disease": "BASELINE_DIAGNOSIS",
            "histology_cytopathology": "BASELINE_DIAGNOSIS",
            "histological_grade": "BASELINE_DIAGNOSIS",
        },
    },
    "enrollment": {
        "relation_from_case": "enrollment",
        "required": {},
        "optional": {
            "enrollment_record_id": "LINKAGE_ID",
            "site_short_name": "BASELINE_ENROLLMENT",
            "veterinary_medical_center": "BASELINE_ENROLLMENT",
            "registering_institution": "BASELINE_ENROLLMENT",
            "patient_subgroup": "BASELINE_ENROLLMENT",
        },
    },
    "study": {
        "relation_from_case": "study",
        "required": {
            "clinical_study_designation": "STUDY_IDENTITY",
        },
        "optional": {
            "clinical_study_id": "STUDY_IDENTITY",
            "clinical_study_name": "STUDY_IDENTITY",
            "clinical_study_type": "STUDY_IDENTITY",
            "accession_id": "STUDY_IDENTITY",
            "study_accession": "STUDY_IDENTITY",
        },
    },
    "study_arm": {
        "relation_from_case": "study_arm",
        "required": {},
        "optional": {
            "arm_id": "ARM_IDENTITY",
            "arm": "ARM_IDENTITY",
        },
    },
    "cohort": {
        "relation_from_case": "cohort",
        "required": {},
        "optional": {
            "cohort_record_id": "COHORT_IDENTITY",
        },
    },
    "registration": {
        "relation_from_case": "registrations",
        "required": {},
        "optional": {
            "registration_record_id": "LINKAGE_ID",
            "registration_origin": "LINKAGE_ID",
            "registration_id": "LINKAGE_ID",
        },
    },
    "sample": {
        "relation_from_case": "samples",
        "required": {},
        "optional": {
            # Deliberately ONLY sample_id. Tumor origin, chronology, collection
            # date, necropsy status, site and other biological annotations remain
            # blocked until the biospecimen/outcome design is frozen.
            "sample_id": "SAMPLE_LINKAGE_ID",
        },
    },
}


# Entire case-linked branches forbidden at this stage.
BLOCKED_CASE_RELATIONS = {
    "adverse_event": "POSTBASELINE_ADVERSE_EVENT",
    "adverse_events": "POSTBASELINE_ADVERSE_EVENT",
    "cycles": "POSTBASELINE_TREATMENT_CYCLE",
    "follow_ups": "OUTCOME_FOLLOWUP",
    "off_study": "POSTBASELINE_OFF_STUDY",
    "off_treatment": "POSTBASELINE_OFF_TREATMENT",
    "visits": "POSTBASELINE_VISIT",
    "files": "DEFER_FILE_MANIFEST_ROUTE",
    # patient_first_name is scalar, handled below.
}

# Exact known high-risk fields. Presence is expected/diagnostic; they must never
# appear in the allowed leaf set.
EXPLICIT_BLOCKED_FIELDS: Dict[str, Dict[str, str]] = {
    "case": {
        "patient_first_name": "DIRECT_IDENTIFIER_NOT_NEEDED",
    },
    "demographic": {
        "date_of_birth": "DIRECT_DATE_NOT_NEEDED",
        "crf_id": "FORM_IDENTIFIER_NOT_NEEDED",
    },
    "diagnosis": {
        "date_of_diagnosis": "TEMPORAL_FIELD_DEFERRED",
        "date_of_histology_confirmation": "TEMPORAL_FIELD_DEFERRED",
        "best_response": "OUTCOME_RESPONSE",
        "pathology_report": "FREE_TEXT_OR_REPORT_DEFERRED",
        "treatment_data": "TREATMENT_INFORMATION",
        "follow_up_data": "OUTCOME_FOLLOWUP",
        "concurrent_disease": "AMBIGUOUS_CLINICAL_FIELD_DEFERRED",
        "concurrent_disease_type": "AMBIGUOUS_CLINICAL_FIELD_DEFERRED",
    },
    "enrollment": {
        "initials": "DIRECT_IDENTIFIER_NOT_NEEDED",
        "date_of_registration": "TEMPORAL_FIELD_DEFERRED",
        "date_of_informed_consent": "TEMPORAL_FIELD_DEFERRED",
    },
    "study": {
        "clinical_study_description": "FREE_TEXT_STUDY_DESCRIPTION_DEFERRED",
        "date_of_iacuc_approval": "TEMPORAL_FIELD_NOT_NEEDED",
        "dates_of_conduct": "TEMPORAL_FIELD_NOT_NEEDED",
        "study_disposition": "STUDY_RESULT_OR_DISPOSITION_DEFERRED",
    },
    "study_arm": {
        "arm_description": "TREATMENT_DESCRIPTION_DEFERRED",
    },
    "cohort": {
        "cohort_description": "MAY_ENCODE_AGENT_OR_DOSE",
        "cohort_dose": "TREATMENT_DOSE",
    },
    "sample": {
        "sample_site": "MAY_ENCODE_METASTATIC_SITE",
        "physical_sample_type": "BIOSPECIMEN_DETAIL_DEFERRED",
        "general_sample_pathology": "PATHOLOGY_DEFERRED",
        "tumor_sample_origin": "MAY_REVEAL_PRIMARY_RECURRENT_METASTATIC_STATUS",
        "summarized_sample_type": "MAY_DERIVE_FROM_TUMOR_ORIGIN",
        "molecular_subtype": "BIOLOGY_DEFERRED",
        "specific_sample_pathology": "PATHOLOGY_DEFERRED",
        "date_of_sample_collection": "TEMPORAL_FIELD_DEFERRED",
        "sample_chronology": "POSTBASELINE_CHRONOLOGY",
        "necropsy_sample": "OUTCOME_ADJACENT_NECROPSY",
        "tumor_grade": "BIOSPECIMEN_PATHOLOGY_DEFERRED",
        "length_of_tumor": "BIOSPECIMEN_MEASUREMENT_DEFERRED",
        "width_of_tumor": "BIOSPECIMEN_MEASUREMENT_DEFERRED",
        "volume_of_tumor": "BIOSPECIMEN_MEASUREMENT_DEFERRED",
        "percentage_tumor": "BIOSPECIMEN_PATHOLOGY_DEFERRED",
        "sample_preservation": "BIOSPECIMEN_TECHNICAL_DEFERRED",
        "comment": "FREE_TEXT_DEFERRED",
    },
}

# Additional object types that are globally inaccessible to the later
# baseline-only fetch even if reachable in the schema.
BLOCKED_TYPES = {
    "adverse_event": "POSTBASELINE",
    "agent": "TREATMENT",
    "agent_administration": "TREATMENT",
    "cycle": "TREATMENT",
    "disease_extent": "TEMPORAL_OR_PROGRESSION_AMBIGUOUS",
    "follow_up": "OUTCOME",
    "lab_exam": "LONGITUDINAL_CLINICAL",
    "off_study": "POSTBASELINE",
    "off_treatment": "POSTBASELINE",
    "physical_exam": "LONGITUDINAL_CLINICAL",
    "prior_surgery": "MIXED_BASELINE_TREATMENT_NODE_DEFERRED",
    "prior_therapy": "MIXED_BASELINE_TREATMENT_RESPONSE_NODE_DEFERRED",
    "visit": "LONGITUDINAL_CLINICAL",
    "vital_signs": "LONGITUDINAL_CLINICAL",
}

# Backstop: these terms are never acceptable in an ALLOWED leaf path at this
# stage. Exact allow-listing comes first; this is only a safety assertion.
FORBIDDEN_ALLOWED_PATH_TERMS = (
    "response",
    "follow_up",
    "followup",
    "survival",
    "death",
    "deceased",
    "mortality",
    "recurrence",
    "relapse",
    "progression",
    "off_study",
    "off_treatment",
    "adverse",
    "necropsy",
    "treatment_data",
    "therapy",
    "agent",
    "dose",
    "cycle",
    "tumor_sample_origin",
    "sample_chronology",
    "date_of_",
    "patient_first_name",
    "initials",
)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def unwrap_type(type_ref: Optional[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str], str]:
    if not type_ref:
        return None, None, "UNKNOWN"

    def render(ref: Optional[Dict[str, Any]]) -> str:
        if not ref:
            return "UNKNOWN"
        kind = ref.get("kind")
        name = ref.get("name")
        child = ref.get("ofType")
        if kind == "NON_NULL":
            return f"{render(child)}!"
        if kind == "LIST":
            return f"[{render(child)}]"
        return name or kind or "UNKNOWN"

    ref = type_ref
    while ref and ref.get("kind") in {"NON_NULL", "LIST"}:
        ref = ref.get("ofType")

    if not ref:
        return None, None, render(type_ref)

    return ref.get("kind"), ref.get("name"), render(type_ref)


def verify_02e2_inputs(summary: Dict[str, Any]) -> Dict[str, str]:
    if summary.get("status") != "PASS":
        raise RuntimeError("02e2 summary status is not PASS.")

    version = str(summary.get("script_version", ""))
    if not version.startswith(EXPECTED_02E2_PREFIX):
        raise RuntimeError(
            f"Unexpected 02e2 script version: {version!r}; "
            f"expected prefix {EXPECTED_02E2_PREFIX!r}."
        )

    endpoint = summary.get("endpoint")
    if endpoint != EXPECTED_ENDPOINT:
        raise RuntimeError(
            f"02e2 endpoint mismatch: {endpoint!r} != {EXPECTED_ENDPOINT!r}"
        )

    safety = summary.get("safety_contract") or {}
    required_safety = {
        "schema_introspection_only": True,
        "case_records_requested": False,
        "outcome_values_requested": False,
        "treatment_values_requested": False,
        "omics_requested_or_downloaded": False,
    }
    for key, expected in required_safety.items():
        actual = safety.get(key)
        if actual is not expected:
            raise RuntimeError(
                f"02e2 safety contract mismatch for {key}: "
                f"expected {expected}, found {actual}"
            )

    expected_paths = {
        INPUT_QUERY_FIELD,
        INPUT_TYPE_GRAPH,
        INPUT_FIELDS_TSV,
    }
    for path in expected_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing required 02e2 artifact: {path}")

    artifact_meta = summary.get("artifacts") or {}
    verified: Dict[str, str] = {}

    for path in sorted(expected_paths, key=lambda p: str(p)):
        rel = str(path.relative_to(PROJECT_ROOT))
        meta = artifact_meta.get(rel)
        if not meta:
            # Handle possible Windows separator serialization.
            meta = artifact_meta.get(rel.replace("/", "\\"))
        if not meta:
            raise RuntimeError(
                f"02e2 summary does not contain frozen hash metadata for {rel}"
            )

        expected_sha = str(meta.get("sha256", "")).lower()
        actual_sha = sha256_file(path).lower()
        if not expected_sha or actual_sha != expected_sha:
            raise RuntimeError(
                f"02e2 artifact hash mismatch for {rel}: "
                f"expected={expected_sha}, actual={actual_sha}"
            )
        verified[rel] = actual_sha

    return verified


def type_fields(type_graph: Dict[str, Any], type_name: str) -> Dict[str, Dict[str, Any]]:
    types = type_graph.get("types") or {}
    info = types.get(type_name)
    if not info:
        return {}
    return {
        field.get("name"): field
        for field in (info.get("fields") or [])
        if field.get("name")
    }


def build_contract_rows(type_graph: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    rows: List[Dict[str, Any]] = []
    missing_required: List[str] = []
    missing_optional: List[str] = []

    types = type_graph.get("types") or {}
    if "case" not in types:
        raise RuntimeError("02e2 type graph does not contain the `case` type.")

    for type_name, policy in POLICY.items():
        fields = type_fields(type_graph, type_name)
        if not fields:
            if policy.get("required"):
                for field_name in policy["required"]:
                    missing_required.append(f"{type_name}.{field_name} (type missing)")
            for field_name in policy.get("optional", {}):
                missing_optional.append(f"{type_name}.{field_name} (type missing)")
            continue

        desired: List[Tuple[str, str, bool]] = []
        for field_name, role in policy.get("required", {}).items():
            desired.append((field_name, role, True))
        for field_name, role in policy.get("optional", {}).items():
            desired.append((field_name, role, False))

        for field_name, role, required in desired:
            field = fields.get(field_name)
            if not field:
                target = f"{type_name}.{field_name}"
                if required:
                    missing_required.append(target)
                else:
                    missing_optional.append(target)
                continue

            base_kind, base_name, display_type = unwrap_type(field.get("type"))
            if base_kind in {"OBJECT", "INTERFACE", "UNION"}:
                raise RuntimeError(
                    f"Policy expected leaf field {type_name}.{field_name}, "
                    f"but schema reports object-like type {display_type}."
                )

            rows.append(
                {
                    "decision": "ALLOW",
                    "role": role,
                    "parent_type": type_name,
                    "field_name": field_name,
                    "field_path": f"{type_name}.{field_name}",
                    "graphql_type": display_type,
                    "base_kind": base_kind or "",
                    "base_type": base_name or "",
                    "required_by_contract": required,
                    "reason": "EXACT_PREVALUE_ALLOWLIST",
                }
            )

    # Explicit blocked fields that currently exist.
    for type_name, blocked_fields in EXPLICIT_BLOCKED_FIELDS.items():
        fields = type_fields(type_graph, type_name)
        for field_name, reason in blocked_fields.items():
            field = fields.get(field_name)
            if not field:
                continue
            base_kind, base_name, display_type = unwrap_type(field.get("type"))
            rows.append(
                {
                    "decision": "BLOCK",
                    "role": "FORBIDDEN_OR_DEFERRED",
                    "parent_type": type_name,
                    "field_name": field_name,
                    "field_path": f"{type_name}.{field_name}",
                    "graphql_type": display_type,
                    "base_kind": base_kind or "",
                    "base_type": base_name or "",
                    "required_by_contract": False,
                    "reason": reason,
                }
            )

    # Direct case relations: freeze each as TRAVERSE or BLOCK.
    case_fields = type_fields(type_graph, "case")
    allowed_relation_names = {
        policy["relation_from_case"]: type_name
        for type_name, policy in POLICY.items()
        if policy.get("relation_from_case")
    }

    for relation_name, target_type in sorted(allowed_relation_names.items()):
        field = case_fields.get(relation_name)
        if not field:
            # Relations are optional in the contract except the case leaf fields.
            missing_optional.append(f"case.{relation_name} (relation to {target_type})")
            continue

        base_kind, base_name, display_type = unwrap_type(field.get("type"))
        if base_kind not in {"OBJECT", "INTERFACE", "UNION"}:
            raise RuntimeError(
                f"Expected case.{relation_name} to be object-like, got {display_type}."
            )
        if base_name != target_type:
            raise RuntimeError(
                f"case.{relation_name} target changed: expected {target_type}, "
                f"found {base_name} ({display_type})."
            )

        rows.append(
            {
                "decision": "TRAVERSE",
                "role": "SAFE_RELATION",
                "parent_type": "case",
                "field_name": relation_name,
                "field_path": f"case.{relation_name}",
                "graphql_type": display_type,
                "base_kind": base_kind or "",
                "base_type": base_name or "",
                "required_by_contract": False,
                "reason": f"TRAVERSE_ONLY_TO_EXACT_ALLOWLISTED_{target_type.upper()}_LEAVES",
            }
        )

    for relation_name, reason in sorted(BLOCKED_CASE_RELATIONS.items()):
        field = case_fields.get(relation_name)
        if not field:
            continue
        base_kind, base_name, display_type = unwrap_type(field.get("type"))
        rows.append(
            {
                "decision": "BLOCK",
                "role": "FORBIDDEN_OR_DEFERRED_RELATION",
                "parent_type": "case",
                "field_name": relation_name,
                "field_path": f"case.{relation_name}",
                "graphql_type": display_type,
                "base_kind": base_kind or "",
                "base_type": base_name or "",
                "required_by_contract": False,
                "reason": reason,
            }
        )

    # Record reachable high-risk object types as blocked, even when not directly
    # attached to case. This is a hard downstream contract.
    for type_name, reason in sorted(BLOCKED_TYPES.items()):
        if type_name not in types:
            continue
        rows.append(
            {
                "decision": "BLOCK_TYPE",
                "role": "FORBIDDEN_OR_DEFERRED_TYPE",
                "parent_type": type_name,
                "field_name": "*",
                "field_path": f"{type_name}.*",
                "graphql_type": "OBJECT",
                "base_kind": "OBJECT",
                "base_type": type_name,
                "required_by_contract": False,
                "reason": reason,
            }
        )

    rows.sort(
        key=lambda row: (
            {"ALLOW": 0, "TRAVERSE": 1, "BLOCK": 2, "BLOCK_TYPE": 3}.get(
                row["decision"], 9
            ),
            row["field_path"],
        )
    )
    return rows, sorted(set(missing_required)), sorted(set(missing_optional))


def assert_no_forbidden_allowed_paths(rows: Iterable[Dict[str, Any]]) -> None:
    bad: List[Tuple[str, str]] = []
    for row in rows:
        if row.get("decision") != "ALLOW":
            continue
        path = str(row.get("field_path", "")).lower()
        for term in FORBIDDEN_ALLOWED_PATH_TERMS:
            if term.lower() in path:
                bad.append((path, term))
    if bad:
        details = ", ".join(f"{path} <- {term}" for path, term in bad)
        raise RuntimeError(
            "Safety backstop failed: forbidden term found in allowed path(s): "
            + details
        )


def build_selection(type_graph: Dict[str, Any], rows: List[Dict[str, Any]]) -> str:
    """
    Build the selection set only. 02e4 can embed it into a case(...) query.
    """
    allowed_by_type: Dict[str, List[str]] = {}
    for row in rows:
        if row["decision"] == "ALLOW":
            allowed_by_type.setdefault(row["parent_type"], []).append(row["field_name"])

    for key in allowed_by_type:
        allowed_by_type[key] = sorted(set(allowed_by_type[key]))

    lines: List[str] = []

    # Direct scalar fields from case.
    for field_name in allowed_by_type.get("case", []):
        lines.append(field_name)

    # Nested branches in deterministic order.
    nested_order = [
        "canine_individual",
        "demographic",
        "diagnosis",
        "enrollment",
        "study",
        "study_arm",
        "cohort",
        "registration",
        "sample",
    ]

    for type_name in nested_order:
        fields = allowed_by_type.get(type_name, [])
        if not fields:
            continue
        relation_name = POLICY[type_name]["relation_from_case"]

        # Only include relation if it exists and points to the expected type.
        case_field = type_fields(type_graph, "case").get(relation_name)
        if not case_field:
            continue
        _, base_name, _ = unwrap_type(case_field.get("type"))
        if base_name != type_name:
            continue

        lines.append(f"{relation_name} {{")
        for field_name in fields:
            lines.append(f"  {field_name}")
        lines.append("}")

    if not lines:
        raise RuntimeError("Generated GraphQL selection set is empty.")

    body = "\n".join(f"  {line}" if not line.startswith("  ") else f"  {line}" for line in lines)
    return (
        "# Frozen by 02e3. Selection set only; no query is executed here.\n"
        "# Do not add fields ad hoc in downstream scripts.\n"
        "{\n"
        f"{body}\n"
        "}\n"
    )


def main() -> None:
    print("=" * 104)
    print("Paper 6 - freeze ICDC baseline/linkage field contract")
    print("=" * 104)
    print(f"Script version: {SCRIPT_VERSION}")
    print()
    print("Execution contract:")
    print("  Network/API access in 02e3: NO")
    print("  Case/patient values read: NO")
    print("  Outcome/response/follow-up values read: NO")
    print("  Treatment administration values read: NO")
    print("  Omics values read/downloaded: NO")
    print()

    for path in (INPUT_SUMMARY, INPUT_QUERY_FIELD, INPUT_TYPE_GRAPH, INPUT_FIELDS_TSV):
        if not path.exists():
            raise FileNotFoundError(
                f"Required 02e2 artifact is missing: {path}\n"
                "Run scripts\\02e2_inventory_icdc_case_schema.py first."
            )

    run_started_utc = datetime.now(timezone.utc).isoformat()

    summary_02e2 = read_json(INPUT_SUMMARY)
    verified_hashes = verify_02e2_inputs(summary_02e2)

    type_graph = read_json(INPUT_TYPE_GRAPH)
    query_field = read_json(INPUT_QUERY_FIELD)

    case_return = type_graph.get("case_return_type") or {}
    if case_return.get("name") != "case":
        raise RuntimeError(
            "02e2 case return type changed or is missing: "
            f"{case_return!r}"
        )

    qf_return = query_field.get("case_return_name")
    if qf_return != "case":
        raise RuntimeError(
            f"02e2 query-field artifact reports case return type {qf_return!r}, "
            "expected 'case'."
        )

    print("02e2 provenance verification:")
    print("  02e2 status: PASS")
    print("  production endpoint lock: PASS")
    print("  frozen artifact hashes: PASS")
    print(f"  verified artifacts: {len(verified_hashes)}")
    print()

    rows, missing_required, missing_optional = build_contract_rows(type_graph)

    if missing_required:
        print("Required field mismatches:")
        for item in missing_required:
            print(f"  - {item}")
        raise RuntimeError(
            "Production schema no longer satisfies the frozen baseline contract. "
            "No downstream value query should be run."
        )

    assert_no_forbidden_allowed_paths(rows)

    allowed_rows = [row for row in rows if row["decision"] == "ALLOW"]
    traverse_rows = [row for row in rows if row["decision"] == "TRAVERSE"]
    blocked_rows = [
        row for row in rows if row["decision"] in {"BLOCK", "BLOCK_TYPE"}
    ]

    # Extra invariant: no allowed field may come from a globally blocked type.
    blocked_type_names = {
        row["parent_type"]
        for row in rows
        if row["decision"] == "BLOCK_TYPE"
    }
    overlap = sorted(
        {
            row["parent_type"]
            for row in allowed_rows
            if row["parent_type"] in blocked_type_names
        }
    )
    if overlap:
        raise RuntimeError(
            "Allowed fields unexpectedly originate from blocked types: "
            + ", ".join(overlap)
        )

    selection = build_selection(type_graph, rows)
    with SELECTION_GRAPHQL.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(selection)

    tsv_columns = [
        "decision",
        "role",
        "parent_type",
        "field_name",
        "field_path",
        "graphql_type",
        "base_kind",
        "base_type",
        "required_by_contract",
        "reason",
    ]
    with CONTRACT_TSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tsv_columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    role_counts = Counter(row["role"] for row in allowed_rows)
    decision_counts = Counter(row["decision"] for row in rows)

    contract = {
        "script_version": SCRIPT_VERSION,
        "run_started_utc": run_started_utc,
        "source_02e2": {
            "script_version": summary_02e2.get("script_version"),
            "endpoint": summary_02e2.get("endpoint"),
            "verified_artifact_hashes": verified_hashes,
        },
        "scientific_stage": "PRE_OUTCOME_PRE_RESPONSE_PRE_TREATMENT_EFFECT",
        "policy": {
            "allowed_leaf_fields": allowed_rows,
            "safe_case_relations": traverse_rows,
            "blocked_or_deferred": blocked_rows,
            "missing_optional_fields_or_relations": missing_optional,
            "forbidden_allowed_path_terms": list(FORBIDDEN_ALLOWED_PATH_TERMS),
        },
        "hard_constraints": {
            "no_outcome_fields": True,
            "no_response_fields": True,
            "no_follow_up_fields": True,
            "no_longitudinal_treatment_fields": True,
            "no_adverse_event_fields": True,
            "no_metastatic_origin_field": True,
            "no_sample_chronology_field": True,
            "no_necropsy_field": True,
            "no_free_text_clinical_fields": True,
            "no_patient_name_or_initials": True,
            "no_dates_in_allowed_leaf_fields": True,
            "sample_branch_limited_to_sample_id": True,
        },
        "selection_artifact": str(SELECTION_GRAPHQL.relative_to(PROJECT_ROOT)),
    }
    write_json(CONTRACT_JSON, contract)

    output_hashes = {
        str(CONTRACT_JSON.relative_to(PROJECT_ROOT)): sha256_file(CONTRACT_JSON),
        str(CONTRACT_TSV.relative_to(PROJECT_ROOT)): sha256_file(CONTRACT_TSV),
        str(SELECTION_GRAPHQL.relative_to(PROJECT_ROOT)): sha256_file(SELECTION_GRAPHQL),
    }

    summary = {
        "script_version": SCRIPT_VERSION,
        "status": "PASS",
        "run_started_utc": run_started_utc,
        "run_finished_utc": datetime.now(timezone.utc).isoformat(),
        "network_access": False,
        "case_values_read": False,
        "outcome_values_read": False,
        "response_values_read": False,
        "follow_up_values_read": False,
        "treatment_administration_values_read": False,
        "omics_values_read_or_downloaded": False,
        "input_02e2_hashes_verified": verified_hashes,
        "contract_counts": {
            "allowed_leaf_fields": len(allowed_rows),
            "safe_case_relations": len(traverse_rows),
            "blocked_or_deferred_entries": len(blocked_rows),
            "decision_counts": dict(sorted(decision_counts.items())),
            "allowed_role_counts": dict(sorted(role_counts.items())),
            "missing_optional_fields_or_relations": len(missing_optional),
        },
        "missing_optional_fields_or_relations": missing_optional,
        "output_hashes": output_hashes,
    }
    write_json(SUMMARY_JSON, summary)

    print("-" * 104)
    print("Frozen contract summary")
    print("-" * 104)
    print(f"Allowed leaf fields: {len(allowed_rows)}")
    print(f"Safe case relations: {len(traverse_rows)}")
    print(f"Blocked/deferred entries: {len(blocked_rows)}")
    print(f"Missing required fields: {len(missing_required)}")
    print(f"Missing optional fields/relations: {len(missing_optional)}")
    print()

    print("Allowed leaf fields:")
    for row in allowed_rows:
        print(
            f"  {row['field_path']}: {row['graphql_type']} "
            f"[{row['role']}]"
        )

    if missing_optional:
        print()
        print("Optional contract items absent from current production schema:")
        for item in missing_optional:
            print(f"  - {item}")

    print()
    print("High-risk fields/branches explicitly blocked:")
    important_block_reasons = {
        "OUTCOME_RESPONSE",
        "OUTCOME_FOLLOWUP",
        "TREATMENT_INFORMATION",
        "MAY_REVEAL_PRIMARY_RECURRENT_METASTATIC_STATUS",
        "POSTBASELINE_CHRONOLOGY",
        "OUTCOME_ADJACENT_NECROPSY",
        "POSTBASELINE_ADVERSE_EVENT",
        "POSTBASELINE_TREATMENT_CYCLE",
        "POSTBASELINE_OFF_STUDY",
        "POSTBASELINE_OFF_TREATMENT",
        "POSTBASELINE_VISIT",
    }
    shown = 0
    for row in blocked_rows:
        if row["reason"] in important_block_reasons:
            print(f"  {row['field_path']} [{row['reason']}]")
            shown += 1
    if shown == 0:
        print("  <no named high-risk field was present; global blocked-type contract still applies>")

    print()
    print("Frozen downstream selection set:")
    print(selection.rstrip())

    print()
    print(f"Artifacts: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")
    print(f"  {CONTRACT_JSON.name}")
    print(f"  {CONTRACT_TSV.name}")
    print(f"  {SELECTION_GRAPHQL.name}")
    print(f"  {SUMMARY_JSON.name}")
    print()
    print("Network/API access in 02e3: NO")
    print("Case/patient values read: NO")
    print("Outcome/response/follow-up values read: NO")
    print("Treatment administration values read: NO")
    print("Omics values read/downloaded: NO")
    print()
    print("02e3 baseline/linkage contract freeze: PASS")
    print("=" * 104)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 104, file=sys.stderr)
        print("02e3 baseline/linkage contract freeze: FAIL", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print("=" * 104, file=sys.stderr)
        raise
