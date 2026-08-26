#!/usr/bin/env python3
"""
Paper 6 - inventory ICDC production Case schema before clinical batch fetching.

This script performs GraphQL schema introspection only.
It does NOT fetch case records, outcome values, treatment values, or omics data.

Purpose
-------
1. Confirm the production `case` query field and its argument/return types.
2. Inventory the Case return type and reachable child object/interface types.
3. Label field names that may represent outcome or post-baseline information so
   they can be excluded explicitly before any clinical batch fetch is written.
4. Freeze the schema inventory as JSON/TSV artifacts for audit/provenance.

No command-line arguments are used.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests


SCRIPT_VERSION = "02e2-inventory-icdc-case-schema-v1-no-cli"
ENDPOINT = "https://caninecommons.cancer.gov/v1/graphql/"
REQUEST_TIMEOUT_SECONDS = 45

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "results" / "icdc_case_schema" / "02e2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

QUERY_FIELD_JSON = OUTPUT_DIR / "query_field_case.json"
TYPE_GRAPH_JSON = OUTPUT_DIR / "case_type_graph.json"
FIELDS_TSV = OUTPUT_DIR / "case_schema_fields.tsv"
SUMMARY_JSON = OUTPUT_DIR / "summary.json"

# We only need a bounded local neighborhood around the Case type.
MAX_OBJECT_DEPTH = 3
MAX_TYPES = 80

# Conservative labels only. These labels DO NOT decide what is scientifically
# admissible; they force explicit review before later data extraction.
OUTCOME_TERMS = (
    "survival",
    "death",
    "deceased",
    "mortality",
    "outcome",
    "recurrence",
    "relapse",
    "progression",
    "response",
    "disease_free",
    "event_free",
    "vital_status",
    "cause_of_death",
    "euthanasia",
    "follow_up",
    "followup",
    "last_known_alive",
    "last_contact",
    "time_to",
    "days_to",
    "date_of_death",
)

POSTBASELINE_TERMS = (
    "treatment",
    "therapy",
    "therapeutic",
    "drug",
    "medication",
    "dose",
    "dosing",
    "regimen",
    "adverse",
    "toxicity",
    "intervention",
    "response",
)

TEMPORAL_REVIEW_TERMS = (
    "metastasis",
    "metastatic",
    "disease_status",
    "stage",
    "diagnosis",
    "collection",
    "visit",
    "specimen",
    "sample",
    "date",
    "time",
)

LIKELY_BASELINE_TERMS = (
    "case_id",
    "study",
    "cohort",
    "arm",
    "breed",
    "sex",
    "gender",
    "age",
    "weight",
    "site",
    "anatomic",
    "primary_tumor",
    "histology",
)


TYPE_REF_SELECTION = """
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
"""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def post_graphql(query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        response = requests.post(
            ENDPOINT,
            json={"query": query, "variables": variables or {}},
            timeout=REQUEST_TIMEOUT_SECONDS,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": f"paper6-cross-species-osteosarcoma/{SCRIPT_VERSION}",
            },
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"ICDC production endpoint request failed: {exc}") from exc

    try:
        payload = response.json()
    except ValueError as exc:
        preview = response.text[:500]
        raise RuntimeError(
            "ICDC production endpoint returned non-JSON content. "
            f"HTTP {response.status_code}; preview={preview!r}"
        ) from exc

    if payload.get("errors"):
        raise RuntimeError(
            "ICDC GraphQL returned errors:\n"
            + json.dumps(payload["errors"], indent=2, ensure_ascii=False)
        )

    if "data" not in payload:
        raise RuntimeError(
            "ICDC GraphQL response did not contain a data object:\n"
            + json.dumps(payload, indent=2, ensure_ascii=False)[:4000]
        )

    return payload


def unwrap_type(type_ref: Optional[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str], str]:
    """
    Return (base_kind, base_name, display_type) from a nested GraphQL type ref.
    Example: NON_NULL -> LIST -> NON_NULL -> OBJECT Case becomes [Case!]!
    """
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


def normalize_text(value: Optional[str]) -> str:
    if not value:
        return ""
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_")


def keyword_hits(text: str, terms: Iterable[str]) -> List[str]:
    hits: List[str] = []
    for term in terms:
        normalized_term = normalize_text(term)
        if normalized_term and normalized_term in text:
            hits.append(term)
    return sorted(set(hits))


def classify_field(path: str, field_name: str, description: Optional[str]) -> Tuple[str, List[str]]:
    # Field name/path carry most of the signal. Description is included only
    # after normalization and is deliberately secondary.
    primary = normalize_text(f"{path} {field_name}")
    secondary = normalize_text(description or "")
    searchable = f"{primary} {secondary}"

    outcome_hits = keyword_hits(searchable, OUTCOME_TERMS)
    if outcome_hits:
        return "BLOCK_OUTCOME", outcome_hits

    postbaseline_hits = keyword_hits(searchable, POSTBASELINE_TERMS)
    if postbaseline_hits:
        return "REVIEW_POSTBASELINE", postbaseline_hits

    temporal_hits = keyword_hits(searchable, TEMPORAL_REVIEW_TERMS)
    if temporal_hits:
        return "REVIEW_TEMPORAL", temporal_hits

    baseline_hits = keyword_hits(primary, LIKELY_BASELINE_TERMS)
    if baseline_hits:
        return "LIKELY_BASELINE", baseline_hits

    return "UNCLASSIFIED", []


def query_root_fields() -> Dict[str, Any]:
    query = f"""
    query Paper6QueryRootIntrospection {{
      __schema {{
        queryType {{
          name
          fields(includeDeprecated: true) {{
            name
            description
            isDeprecated
            deprecationReason
            args {{
              name
              description
              defaultValue
              type {{
                {TYPE_REF_SELECTION}
              }}
            }}
            type {{
              {TYPE_REF_SELECTION}
            }}
          }}
        }}
      }}
    }}
    """
    return post_graphql(query)


def query_named_type(type_name: str) -> Dict[str, Any]:
    query = f"""
    query Paper6NamedTypeIntrospection($typeName: String!) {{
      __type(name: $typeName) {{
        kind
        name
        description
        fields(includeDeprecated: true) {{
          name
          description
          isDeprecated
          deprecationReason
          args {{
            name
            description
            defaultValue
            type {{
              {TYPE_REF_SELECTION}
            }}
          }}
          type {{
            {TYPE_REF_SELECTION}
          }}
        }}
        possibleTypes {{
          kind
          name
        }}
        enumValues(includeDeprecated: true) {{
          name
          description
          isDeprecated
          deprecationReason
        }}
      }}
    }}
    """
    return post_graphql(query, {"typeName": type_name})


def main() -> None:
    print("=" * 100)
    print("Paper 6 - inventory ICDC production Case schema before clinical batch fetching")
    print("=" * 100)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Endpoint: {ENDPOINT}")
    print()
    print("Safety contract:")
    print("  GraphQL schema introspection only: YES")
    print("  Case records requested: NO")
    print("  Outcome values requested: NO")
    print("  Treatment values requested: NO")
    print("  Omics requested/downloaded: NO")
    print()

    run_started_utc = datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # 1) Locate the production query field named `case`.
    # ------------------------------------------------------------------
    root_payload = query_root_fields()
    query_type = (
        root_payload.get("data", {})
        .get("__schema", {})
        .get("queryType")
    )
    if not query_type:
        raise RuntimeError("Could not introspect the GraphQL query root.")

    root_fields = query_type.get("fields") or []
    case_fields = [field for field in root_fields if field.get("name") == "case"]

    if len(case_fields) != 1:
        available = sorted(field.get("name") for field in root_fields if field.get("name"))
        raise RuntimeError(
            f"Expected exactly one query field named 'case'; found {len(case_fields)}. "
            f"Available query fields: {available}"
        )

    case_field = case_fields[0]
    case_return_kind, case_return_name, case_return_display = unwrap_type(case_field.get("type"))
    if not case_return_name:
        raise RuntimeError(
            "The `case` query field was found, but its concrete return type could not be resolved."
        )

    case_field_record = {
        "script_version": SCRIPT_VERSION,
        "endpoint": ENDPOINT,
        "run_started_utc": run_started_utc,
        "query_type_name": query_type.get("name"),
        "case_query_field": case_field,
        "case_return_kind": case_return_kind,
        "case_return_name": case_return_name,
        "case_return_display": case_return_display,
    }
    write_json(QUERY_FIELD_JSON, case_field_record)

    print("Production query route:")
    print("  case query field: PASS")
    print(f"  return type: {case_return_display}")
    print("  arguments:")
    for arg in case_field.get("args") or []:
        _, _, display = unwrap_type(arg.get("type"))
        default = arg.get("defaultValue")
        suffix = f", default={default}" if default is not None else ""
        print(f"    - {arg.get('name')}: {display}{suffix}")
    if not (case_field.get("args") or []):
        print("    - <none>")
    print()

    # ------------------------------------------------------------------
    # 2) Breadth-first inventory of reachable object/interface types.
    #    No actual case query is executed.
    # ------------------------------------------------------------------
    queue = deque([(case_return_name, 0, case_return_name)])
    best_depth: Dict[str, int] = {}
    type_records: Dict[str, Dict[str, Any]] = {}
    field_rows: List[Dict[str, Any]] = []

    while queue:
        type_name, depth, path_prefix = queue.popleft()

        if type_name in best_depth and best_depth[type_name] <= depth:
            continue
        best_depth[type_name] = depth

        if len(type_records) >= MAX_TYPES:
            raise RuntimeError(
                f"Schema traversal exceeded MAX_TYPES={MAX_TYPES}. "
                "Increase the bound only after reviewing the current schema."
            )

        payload = query_named_type(type_name)
        type_info = payload.get("data", {}).get("__type")
        if not type_info:
            raise RuntimeError(f"GraphQL __type lookup returned no type for {type_name!r}.")

        type_records[type_name] = type_info

        fields = type_info.get("fields") or []
        for field in fields:
            base_kind, base_name, display_type = unwrap_type(field.get("type"))
            field_name = field.get("name") or ""
            field_path = f"{path_prefix}.{field_name}"
            risk_class, matched_terms = classify_field(
                field_path,
                field_name,
                field.get("description"),
            )

            field_rows.append(
                {
                    "depth": depth,
                    "parent_type": type_name,
                    "field_path": field_path,
                    "field_name": field_name,
                    "return_type": display_type,
                    "base_kind": base_kind or "",
                    "base_type": base_name or "",
                    "risk_class": risk_class,
                    "matched_terms": ",".join(matched_terms),
                    "deprecated": str(bool(field.get("isDeprecated"))).upper(),
                    "deprecation_reason": field.get("deprecationReason") or "",
                    "description": (field.get("description") or "").replace("\r", " ").replace("\n", " "),
                }
            )

            if (
                depth < MAX_OBJECT_DEPTH
                and base_name
                and base_kind in {"OBJECT", "INTERFACE", "UNION"}
                and not base_name.startswith("__")
            ):
                # Use the first observed path for the bounded traversal.
                queue.append((base_name, depth + 1, field_path))

        # For an interface/union, inspect possible concrete object types as well.
        if depth < MAX_OBJECT_DEPTH:
            for possible in type_info.get("possibleTypes") or []:
                possible_name = possible.get("name")
                possible_kind = possible.get("kind")
                if possible_name and possible_kind == "OBJECT" and not possible_name.startswith("__"):
                    queue.append((possible_name, depth + 1, f"{path_prefix}::<{possible_name}>"))

    graph_record = {
        "script_version": SCRIPT_VERSION,
        "endpoint": ENDPOINT,
        "run_started_utc": run_started_utc,
        "case_return_type": {
            "kind": case_return_kind,
            "name": case_return_name,
            "display": case_return_display,
        },
        "max_object_depth": MAX_OBJECT_DEPTH,
        "max_types": MAX_TYPES,
        "types": type_records,
    }
    write_json(TYPE_GRAPH_JSON, graph_record)

    field_rows.sort(
        key=lambda row: (
            int(row["depth"]),
            str(row["parent_type"]),
            str(row["field_name"]),
        )
    )

    tsv_columns = [
        "depth",
        "parent_type",
        "field_path",
        "field_name",
        "return_type",
        "base_kind",
        "base_type",
        "risk_class",
        "matched_terms",
        "deprecated",
        "deprecation_reason",
        "description",
    ]
    with FIELDS_TSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tsv_columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(field_rows)

    class_counts = Counter(row["risk_class"] for row in field_rows)
    kind_counts = Counter(row["base_kind"] for row in field_rows)

    # Direct Case fields are the most important immediate output for the next script.
    direct_case_rows = [
        row
        for row in field_rows
        if row["parent_type"] == case_return_name and int(row["depth"]) == 0
    ]

    summary = {
        "script_version": SCRIPT_VERSION,
        "endpoint": ENDPOINT,
        "run_started_utc": run_started_utc,
        "run_finished_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "safety_contract": {
            "schema_introspection_only": True,
            "case_records_requested": False,
            "outcome_values_requested": False,
            "treatment_values_requested": False,
            "omics_requested_or_downloaded": False,
        },
        "case_query_field": {
            "name": "case",
            "return_kind": case_return_kind,
            "return_name": case_return_name,
            "return_display": case_return_display,
            "args": [
                {
                    "name": arg.get("name"),
                    "type": unwrap_type(arg.get("type"))[2],
                    "defaultValue": arg.get("defaultValue"),
                }
                for arg in (case_field.get("args") or [])
            ],
        },
        "inventory": {
            "reachable_types": len(type_records),
            "fields": len(field_rows),
            "direct_case_fields": len(direct_case_rows),
            "risk_class_counts": dict(sorted(class_counts.items())),
            "base_kind_counts": dict(sorted(kind_counts.items())),
        },
        "artifacts": {},
    }

    # Freeze hashes after all artifacts except summary itself are complete.
    summary["artifacts"] = {
        str(QUERY_FIELD_JSON.relative_to(PROJECT_ROOT)): {
            "sha256": sha256_file(QUERY_FIELD_JSON)
        },
        str(TYPE_GRAPH_JSON.relative_to(PROJECT_ROOT)): {
            "sha256": sha256_file(TYPE_GRAPH_JSON)
        },
        str(FIELDS_TSV.relative_to(PROJECT_ROOT)): {
            "sha256": sha256_file(FIELDS_TSV)
        },
    }
    write_json(SUMMARY_JSON, summary)

    print("-" * 100)
    print("Schema inventory summary")
    print("-" * 100)
    print(f"Reachable schema types inventoried: {len(type_records)}")
    print(f"Fields inventoried: {len(field_rows)}")
    print(f"Direct {case_return_name} fields: {len(direct_case_rows)}")
    print()
    print("Direct Case-return fields:")
    for row in direct_case_rows:
        print(
            f"  {row['field_name']}: {row['return_type']} "
            f"[{row['risk_class']}]"
        )
    if not direct_case_rows:
        print("  <none>")

    print()
    print("Reachable object/interface/union types:")
    for type_name in sorted(type_records):
        type_kind = type_records[type_name].get("kind") or "UNKNOWN"
        print(f"  {type_name}: {type_kind}")

    print()
    print("Risk labels:")
    for label in (
        "LIKELY_BASELINE",
        "REVIEW_TEMPORAL",
        "REVIEW_POSTBASELINE",
        "BLOCK_OUTCOME",
        "UNCLASSIFIED",
    ):
        print(f"  {label}: {class_counts.get(label, 0)}")

    print()
    print(f"Artifacts: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")
    print(f"  {QUERY_FIELD_JSON.name}")
    print(f"  {TYPE_GRAPH_JSON.name}")
    print(f"  {FIELDS_TSV.name}")
    print(f"  {SUMMARY_JSON.name}")
    print()
    print("Case records read: NO")
    print("Outcome values read: NO")
    print("Treatment values read: NO")
    print("Omics data read/downloaded: NO")
    print()
    print("02e2 schema inventory: PASS")
    print("=" * 100)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 100, file=sys.stderr)
        print("02e2 schema inventory: FAIL", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print("=" * 100, file=sys.stderr)
        raise
