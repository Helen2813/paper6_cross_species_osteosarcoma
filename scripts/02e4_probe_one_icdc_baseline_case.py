#!/usr/bin/env python3
"""
Paper 6 - controlled one-case ICDC baseline/linkage value probe.

FIRST real-value read in the 02e sequence.

Reads exactly one known public case (COTC021-0101) using the hash-locked
selection produced by 02e3. The response is rejected if it contains any
field outside the frozen 02e3 allow-list.

No outcome/response/follow-up, longitudinal treatment/adverse-event,
metastatic-origin/sample-chronology/necropsy, patient-name/initial, or
omics fields are requested.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import requests


SCRIPT_VERSION = "02e4-probe-one-icdc-baseline-case-v1-no-cli"
ENDPOINT = "https://caninecommons.cancer.gov/v1/graphql/"
TEST_CASE_ID = "COTC021-0101"
TIMEOUT_SECONDS = 45

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "results" / "icdc_baseline_contract" / "02e3"
INPUT_SUMMARY = INPUT_DIR / "summary.json"
INPUT_CONTRACT = INPUT_DIR / "baseline_field_contract.json"
INPUT_CONTRACT_TSV = INPUT_DIR / "baseline_field_contract.tsv"
INPUT_SELECTION = INPUT_DIR / "baseline_selection.graphql"

OUTPUT_DIR = PROJECT_ROOT / "results" / "icdc_baseline_probe" / "02e4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_JSON = OUTPUT_DIR / "cotc021_0101_allowed_response.json"
FLAT_TSV = OUTPUT_DIR / "cotc021_0101_allowed_flat.tsv"
SUMMARY_JSON = OUTPUT_DIR / "summary.json"

EXPECTED_02E3_PREFIX = "02e3-freeze-icdc-baseline-field-contract-"

FORBIDDEN_SELECTION_TOKENS = (
    "best_response",
    "follow_up",
    "follow_ups",
    "treatment_data",
    "off_study",
    "off_treatment",
    "adverse_event",
    "adverse_events",
    "cycles",
    "visits",
    "tumor_sample_origin",
    "sample_chronology",
    "necropsy_sample",
    "patient_first_name",
    "date_of_birth",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")


def lookup_hash(mapping: Dict[str, Any], path: Path) -> str:
    rel = str(path.relative_to(PROJECT_ROOT))
    value = mapping.get(rel)
    if value is None:
        value = mapping.get(rel.replace("/", "\\"))
    return str(value or "")


def verify_02e3() -> Tuple[Dict[str, Any], Dict[str, str]]:
    for path in (INPUT_SUMMARY, INPUT_CONTRACT, INPUT_CONTRACT_TSV, INPUT_SELECTION):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required 02e3 artifact: {path}\n"
                "Run scripts\\02e3_freeze_icdc_baseline_field_contract.py first."
            )

    summary = read_json(INPUT_SUMMARY)
    if summary.get("status") != "PASS":
        raise RuntimeError("02e3 summary status is not PASS.")

    version = str(summary.get("script_version", ""))
    if not version.startswith(EXPECTED_02E3_PREFIX):
        raise RuntimeError(f"Unexpected 02e3 script version: {version!r}")

    safety_false = (
        "network_access",
        "case_values_read",
        "outcome_values_read",
        "response_values_read",
        "follow_up_values_read",
        "treatment_administration_values_read",
        "omics_values_read_or_downloaded",
    )
    for key in safety_false:
        if summary.get(key) is not False:
            raise RuntimeError(f"02e3 provenance mismatch: {key} is not False.")

    output_hashes = summary.get("output_hashes") or {}
    verified: Dict[str, str] = {}
    for path in (INPUT_CONTRACT, INPUT_CONTRACT_TSV, INPUT_SELECTION):
        expected = lookup_hash(output_hashes, path)
        actual = sha256_file(path)
        if not expected or actual.lower() != expected.lower():
            raise RuntimeError(
                f"02e3 artifact hash mismatch for {path.name}: "
                f"expected={expected}, actual={actual}"
            )
        verified[str(path.relative_to(PROJECT_ROOT))] = actual

    return summary, verified


def runtime_policy(contract: Dict[str, Any]) -> Tuple[
    Set[str], Dict[str, Set[str]], Dict[str, str]
]:
    policy = contract.get("policy") or {}
    allowed_rows = policy.get("allowed_leaf_fields") or []
    relation_rows = policy.get("safe_case_relations") or []

    direct: Set[str] = set()
    leaves_by_type: Dict[str, Set[str]] = {}
    relation_to_type: Dict[str, str] = {}

    for row in allowed_rows:
        if row.get("decision") != "ALLOW":
            raise RuntimeError("Malformed 02e3 allowed-leaf contract.")
        parent = str(row.get("parent_type", ""))
        field = str(row.get("field_name", ""))
        if not parent or not field:
            raise RuntimeError("Malformed allowed field row in 02e3 contract.")
        if parent == "case":
            direct.add(field)
        else:
            leaves_by_type.setdefault(parent, set()).add(field)

    for row in relation_rows:
        if row.get("decision") != "TRAVERSE" or row.get("parent_type") != "case":
            raise RuntimeError("Malformed 02e3 safe-relation contract.")
        relation = str(row.get("field_name", ""))
        target = str(row.get("base_type", ""))
        if not relation or not target:
            raise RuntimeError("Malformed safe relation row in 02e3 contract.")
        relation_to_type[relation] = target

    if "case_id" not in direct or "patient_id" not in direct:
        raise RuntimeError("02e3 no longer allows required case linkage IDs.")

    return direct, leaves_by_type, relation_to_type


def frozen_selection() -> str:
    text = INPUT_SELECTION.read_text(encoding="utf-8")
    first = text.find("{")
    if first < 0:
        raise RuntimeError("02e3 selection contains no opening brace.")

    selection = text[first:].strip()
    if not selection.startswith("{") or not selection.endswith("}"):
        raise RuntimeError("02e3 frozen GraphQL selection is malformed.")

    lower = selection.lower()
    hits = sorted(
        token for token in FORBIDDEN_SELECTION_TOKENS
        if token.lower() in lower
    )
    if hits:
        raise RuntimeError(
            "Forbidden token found in frozen selection: " + ", ".join(hits)
        )
    return selection


def validate_nested(
    relation: str,
    target_type: str,
    value: Any,
    allowed: Set[str],
) -> None:
    if value is None:
        return

    items = value if isinstance(value, list) else [value]
    for i, item in enumerate(items):
        if item is None:
            continue
        if not isinstance(item, dict):
            raise RuntimeError(
                f"case.{relation}[{i}] is not an object: {type(item).__name__}"
            )
        unexpected = sorted(set(item) - allowed)
        if unexpected:
            raise RuntimeError(
                f"case.{relation} ({target_type}) returned fields outside "
                f"02e3 allow-list: {unexpected}"
            )
        for key, leaf in item.items():
            if isinstance(leaf, (dict, list)):
                raise RuntimeError(
                    f"Allowed leaf {target_type}.{key} unexpectedly returned "
                    f"{type(leaf).__name__}."
                )


def validate_case(
    case_obj: Dict[str, Any],
    direct: Set[str],
    leaves_by_type: Dict[str, Set[str]],
    relation_to_type: Dict[str, str],
) -> None:
    allowed_case_keys = direct | set(relation_to_type)
    unexpected = sorted(set(case_obj) - allowed_case_keys)
    if unexpected:
        raise RuntimeError(
            "Top-level case fields outside 02e3 contract: " + ", ".join(unexpected)
        )

    for key, value in case_obj.items():
        if key in direct:
            if isinstance(value, (dict, list)):
                raise RuntimeError(f"case.{key} unexpectedly returned nested data.")
            continue

        target = relation_to_type[key]
        allowed = leaves_by_type.get(target, set())
        if not allowed:
            raise RuntimeError(
                f"Returned relation case.{key} has no allowed leaves in 02e3."
            )
        validate_nested(key, target, value, allowed)


def post_graphql(query: str, variables: Dict[str, Any]) -> Dict[str, Any]:
    try:
        r = requests.post(
            ENDPOINT,
            json={"query": query, "variables": variables},
            timeout=TIMEOUT_SECONDS,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": f"paper6-cross-species-osteosarcoma/{SCRIPT_VERSION}",
            },
        )
        r.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"ICDC production request failed: {exc}") from exc

    try:
        payload = r.json()
    except ValueError as exc:
        raise RuntimeError(
            f"ICDC returned non-JSON content; HTTP {r.status_code}; "
            f"preview={r.text[:500]!r}"
        ) from exc

    if payload.get("errors"):
        raise RuntimeError(
            "ICDC GraphQL returned errors:\n"
            + json.dumps(payload["errors"], indent=2, ensure_ascii=False)
        )
    if "data" not in payload:
        raise RuntimeError("ICDC GraphQL response contains no data object.")
    return payload


def relation_count(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, list):
        return len(value)
    return 1


def cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def write_flat(
    case_obj: Dict[str, Any],
    direct: Set[str],
    leaves_by_type: Dict[str, Set[str]],
    relation_to_type: Dict[str, str],
) -> None:
    columns: List[str] = []
    row: Dict[str, str] = {}

    for field in sorted(direct):
        col = f"case.{field}"
        columns.append(col)
        row[col] = cell(case_obj.get(field))

    for relation in sorted(relation_to_type):
        target = relation_to_type[relation]
        fields = sorted(leaves_by_type.get(target, set()))
        value = case_obj.get(relation)

        for field in fields:
            col = f"case.{relation}.{field}"
            columns.append(col)

            if isinstance(value, list):
                values = [
                    item.get(field) if isinstance(item, dict) else None
                    for item in value
                ]
                row[col] = cell(values)
            elif isinstance(value, dict):
                row[col] = cell(value.get(field))
            elif value is None:
                row[col] = ""
            else:
                raise RuntimeError(
                    f"Unexpected value type for case.{relation}: "
                    f"{type(value).__name__}"
                )

    with FLAT_TSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerow(row)


def print_relation(name: str, value: Any, max_items: int = 12) -> None:
    print(f"{name}:")
    if value is None:
        print("  <null>")
        return
    if isinstance(value, dict):
        for key in sorted(value):
            print(f"  {key}: {value[key]!r}")
        return
    if isinstance(value, list):
        print(f"  count: {len(value)}")
        for i, item in enumerate(value[:max_items], start=1):
            if isinstance(item, dict):
                compact = ", ".join(f"{k}={item[k]!r}" for k in sorted(item))
                print(f"  [{i}] {compact}")
            else:
                print(f"  [{i}] {item!r}")
        if len(value) > max_items:
            print(f"  ... {len(value) - max_items} additional item(s) not printed")
        return
    print(f"  {value!r}")


def main() -> None:
    print("=" * 104)
    print("Paper 6 - controlled one-case ICDC baseline/linkage value probe")
    print("=" * 104)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Endpoint: {ENDPOINT}")
    print(f"Test case: {TEST_CASE_ID}")
    print()
    print("Safety contract:")
    print("  Cases requested: 1")
    print("  Selection source: hash-locked 02e3 artifact")
    print("  Outcome/response/follow-up fields requested: NO")
    print("  Longitudinal treatment/adverse-event fields requested: NO")
    print("  Metastatic-origin/sample-chronology/necropsy fields requested: NO")
    print("  Patient name/initial fields requested: NO")
    print("  Omics requested/downloaded: NO")
    print()

    run_started = datetime.now(timezone.utc).isoformat()

    _, verified_hashes = verify_02e3()
    contract = read_json(INPUT_CONTRACT)

    source_endpoint = (contract.get("source_02e2") or {}).get("endpoint")
    if source_endpoint != ENDPOINT:
        raise RuntimeError(
            f"Frozen source endpoint changed: {source_endpoint!r} != {ENDPOINT!r}"
        )

    hard = contract.get("hard_constraints") or {}
    hard_required = (
        "no_outcome_fields",
        "no_response_fields",
        "no_follow_up_fields",
        "no_longitudinal_treatment_fields",
        "no_adverse_event_fields",
        "no_metastatic_origin_field",
        "no_sample_chronology_field",
        "no_necropsy_field",
        "no_free_text_clinical_fields",
        "no_patient_name_or_initials",
        "no_dates_in_allowed_leaf_fields",
        "sample_branch_limited_to_sample_id",
    )
    bad_hard = [key for key in hard_required if hard.get(key) is not True]
    if bad_hard:
        raise RuntimeError(
            "02e3 hard constraints missing/not TRUE: " + ", ".join(bad_hard)
        )

    direct, leaves_by_type, relation_to_type = runtime_policy(contract)
    selection = frozen_selection()

    print("02e3 contract verification:")
    print("  02e3 status: PASS")
    print("  frozen artifact hashes: PASS")
    print(f"  verified artifacts: {len(verified_hashes)}")
    print(f"  direct case leaves allowed: {len(direct)}")
    print(f"  safe case relations: {len(relation_to_type)}")
    print(f"  nested allowed leaves: {sum(len(x) for x in leaves_by_type.values())}")
    print()

    query = f"""
query Paper6FrozenBaselineCaseProbe($caseId: String!) {{
  case(case_id: $caseId) {selection}
}}
"""
    payload = post_graphql(query, {"caseId": TEST_CASE_ID})
    cases = (payload.get("data") or {}).get("case")

    if not isinstance(cases, list):
        raise RuntimeError(
            f"Expected data.case list, found {type(cases).__name__}."
        )
    if len(cases) != 1:
        raise RuntimeError(
            f"Expected exactly one case for {TEST_CASE_ID}, found {len(cases)}."
        )

    case_obj = cases[0]
    if not isinstance(case_obj, dict):
        raise RuntimeError("Returned case is not an object.")
    if case_obj.get("case_id") != TEST_CASE_ID:
        raise RuntimeError(
            f"case_id mismatch: expected {TEST_CASE_ID!r}, "
            f"found {case_obj.get('case_id')!r}."
        )

    # Fail closed if anything outside the frozen contract appears.
    validate_case(case_obj, direct, leaves_by_type, relation_to_type)

    write_json(
        RAW_JSON,
        {
            "script_version": SCRIPT_VERSION,
            "endpoint": ENDPOINT,
            "case_id_requested": TEST_CASE_ID,
            "frozen_selection_sha256": sha256_file(INPUT_SELECTION),
            "case": case_obj,
        },
    )
    write_flat(case_obj, direct, leaves_by_type, relation_to_type)

    relation_counts = {
        relation: relation_count(case_obj.get(relation))
        for relation in sorted(relation_to_type)
    }

    summary = {
        "script_version": SCRIPT_VERSION,
        "status": "PASS",
        "run_started_utc": run_started,
        "run_finished_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": ENDPOINT,
        "case_id_requested": TEST_CASE_ID,
        "cases_requested": 1,
        "cases_returned": 1,
        "response_contract_validation": "PASS",
        "outcome_response_followup_fields_requested": False,
        "longitudinal_treatment_or_adverse_event_fields_requested": False,
        "metastatic_origin_sample_chronology_necropsy_fields_requested": False,
        "patient_name_or_initial_fields_requested": False,
        "omics_requested_or_downloaded": False,
        "verified_02e3_hashes": verified_hashes,
        "relation_cardinality": relation_counts,
        "artifacts": {
            str(RAW_JSON.relative_to(PROJECT_ROOT)): {
                "sha256": sha256_file(RAW_JSON)
            },
            str(FLAT_TSV.relative_to(PROJECT_ROOT)): {
                "sha256": sha256_file(FLAT_TSV)
            },
        },
    }
    write_json(SUMMARY_JSON, summary)

    print("-" * 104)
    print("Controlled value-probe result")
    print("-" * 104)
    print("GraphQL request: PASS")
    print("Cases returned: 1")
    print("case_id identity check: PASS")
    print("Frozen response allow-list validation: PASS")
    print()
    print(f"case_id: {case_obj.get('case_id')!r}")
    print(f"patient_id: {case_obj.get('patient_id')!r}")
    print()

    for relation in (
        "canine_individual",
        "demographic",
        "diagnoses",
        "enrollment",
        "study",
        "study_arm",
        "registrations",
        "samples",
    ):
        if relation in relation_to_type:
            print_relation(relation, case_obj.get(relation))
            print()

    print("Relation cardinalities:")
    for relation, count in relation_counts.items():
        print(f"  {relation}: {count}")

    print()
    print(f"Artifacts: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")
    print(f"  {RAW_JSON.name}")
    print(f"  {FLAT_TSV.name}")
    print(f"  {SUMMARY_JSON.name}")
    print()
    print("Outcome/response/follow-up fields requested/read: NO")
    print("Longitudinal treatment/adverse-event fields requested/read: NO")
    print("Metastatic-origin/sample-chronology/necropsy fields requested/read: NO")
    print("Patient name/initial fields requested/read: NO")
    print("Omics requested/downloaded: NO")
    print()
    print("02e4 controlled one-case baseline probe: PASS")
    print("=" * 104)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 104, file=sys.stderr)
        print("02e4 controlled one-case baseline probe: FAIL", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print("=" * 104, file=sys.stderr)
        raise
