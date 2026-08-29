#!/usr/bin/env python3
"""
Paper 6 - materialize frozen baseline/linkage metadata for the public
COTC021/COTC022 ICDC roster.

This is a controlled real-value batch read.

Scientific / leakage contract
-----------------------------
* The treatment-group definition is already frozen by 02e5:
    COTC021 -> SOC_PLUS_RAPAMYCIN
    COTC022 -> SOC_CONTROL
* The GraphQL selection is read verbatim from the hash-locked 02e3 artifact.
* No outcome, response, follow-up, longitudinal treatment, adverse-event,
  metastatic-origin, sample-chronology, necropsy, name/initial, or omics fields
  are requested.
* Every returned object is validated against the exact 02e3 allow-list.
* Every case must have study.clinical_study_designation consistent with its
  frozen roster study.
* The public roster must contain exactly:
    COTC021 = 152
    COTC022 = 157
    total   = 309
* Final materialization is written only after all 309 cases validate.

Roster provenance
-----------------
The exact 309 case IDs were previously frozen by 02d2. Older local runs may
have used different result-directory/file names, so v2 searches non-source
artifacts across the project rather than relying on a filename convention.

It accepts only evidence that reconstructs exactly the already-frozen
152 COTC021 + 157 COTC022 = 309 unique case-ID set. Preferred evidence is one
exact file; a consistent exact study-specific pair or exact multi-file union is
accepted only under the same frozen count contract. Ambiguous evidence fails
closed.

Restart safety
--------------
GraphQL requests are grouped into small alias batches. Each validated batch is
checkpointed. Re-running the script reuses only checkpoints whose roster hash,
selection hash, 02e5 contract hash, and exact requested case IDs still match.

No command-line arguments are used.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import requests


SCRIPT_VERSION = "02e6-materialize-icdc-baseline-309-v2-no-cli"
ENDPOINT = "https://caninecommons.cancer.gov/v1/graphql/"

ROOT = Path(__file__).resolve().parents[1]

# Frozen upstream artifacts.
E3_DIR = ROOT / "results" / "icdc_baseline_contract" / "02e3"
E3_SUMMARY = E3_DIR / "summary.json"
E3_CONTRACT = E3_DIR / "baseline_field_contract.json"
E3_CONTRACT_TSV = E3_DIR / "baseline_field_contract.tsv"
E3_SELECTION = E3_DIR / "baseline_selection.graphql"

E5_DIR = ROOT / "results" / "cotc_trial_arm_contract" / "02e5"
E5_SUMMARY = E5_DIR / "summary.json"
E5_CONTRACT = E5_DIR / "cotc_study_as_arm_contract.json"

OUT_DIR = ROOT / "results" / "icdc_baseline_batch" / "02e6"
CHECKPOINT_DIR = OUT_DIR / "checkpoints"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

ROSTER_TSV = OUT_DIR / "frozen_public_roster_309.tsv"
JSONL_OUT = OUT_DIR / "icdc_baseline_309.jsonl"
FLAT_TSV_OUT = OUT_DIR / "icdc_baseline_309_flat.tsv"
MANIFEST_JSON = OUT_DIR / "materialization_manifest.json"
SUMMARY_JSON = OUT_DIR / "summary.json"

EXPECTED_COUNTS = {"COTC021": 152, "COTC022": 157}
EXPECTED_TOTAL = 309

# Modest request size; restart checkpoints make conservative batching cheap.
BATCH_SIZE = 20
REQUEST_TIMEOUT_SECONDS = 60
MAX_REQUEST_ATTEMPTS = 4
BACKOFF_SECONDS = (1.0, 2.0, 4.0)

ROSTER_PATH_HINTS = ("02d2", "02d", "parent_roster", "globalsearch", "icdc", "cotc", "case", "roster")
ROSTER_EXTENSIONS = {".json", ".jsonl", ".csv", ".tsv", ".txt", ".yaml", ".yml", ".md", ".log"}
ROSTER_EXCLUDED_DIRS = {".venv", "venv", ".git", "__pycache__", "node_modules", ".idea", ".vscode"}
MAX_ROSTER_FILE_BYTES = 50 * 1024 * 1024

CASE_ID_RE = re.compile(r"\b(COTC021|COTC022)-([A-Za-z0-9][A-Za-z0-9._-]*)\b")

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

EXPECTED_ARM_MAP = {
    "COTC021": "SOC_PLUS_RAPAMYCIN",
    "COTC022": "SOC_CONTROL",
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def lookup_hash(mapping: Dict[str, Any], path: Path) -> str:
    rel = relative(path)
    value = mapping.get(rel)
    if value is None:
        value = mapping.get(rel.replace("/", "\\"))
    if isinstance(value, dict):
        value = value.get("sha256")
    return str(value or "")


def verify_e3() -> Tuple[Dict[str, Any], Dict[str, str]]:
    for path in (E3_SUMMARY, E3_CONTRACT, E3_CONTRACT_TSV, E3_SELECTION):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required 02e3 artifact: {path}\n"
                "Run scripts\\02e3_freeze_icdc_baseline_field_contract.py first."
            )

    summary = read_json(E3_SUMMARY)
    if summary.get("status") != "PASS":
        raise RuntimeError("02e3 summary status is not PASS.")

    for key, expected in {
        "network_access": False,
        "case_values_read": False,
        "outcome_values_read": False,
        "response_values_read": False,
        "follow_up_values_read": False,
        "treatment_administration_values_read": False,
        "omics_values_read_or_downloaded": False,
    }.items():
        if summary.get(key) is not expected:
            raise RuntimeError(f"02e3 provenance mismatch for {key}.")

    hashes = summary.get("output_hashes") or {}
    verified: Dict[str, str] = {}

    for path in (E3_CONTRACT, E3_CONTRACT_TSV, E3_SELECTION):
        expected = lookup_hash(hashes, path)
        actual = sha256_file(path)
        if not expected or expected.lower() != actual.lower():
            raise RuntimeError(
                f"02e3 hash mismatch for {path.name}: "
                f"expected={expected}, actual={actual}"
            )
        verified[relative(path)] = actual

    contract = read_json(E3_CONTRACT)
    if (contract.get("source_02e2") or {}).get("endpoint") != ENDPOINT:
        raise RuntimeError("02e3 production endpoint lock mismatch.")

    return contract, verified


def verify_e5() -> Tuple[Dict[str, Any], str]:
    for path in (E5_SUMMARY, E5_CONTRACT):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required 02e5 artifact: {path}\n"
                "Run scripts\\02e5_freeze_cotc_study_as_arm_contract.py first."
            )

    summary = read_json(E5_SUMMARY)
    if summary.get("status") != "PASS":
        raise RuntimeError("02e5 summary status is not PASS.")

    for key in (
        "network_access",
        "new_icdc_case_values_read",
        "outcome_response_followup_values_read",
        "treatment_administration_values_read",
        "postbaseline_sample_annotations_read",
        "omics_values_read_or_downloaded",
    ):
        if summary.get(key) is not False:
            raise RuntimeError(f"02e5 provenance mismatch for {key}.")

    expected_hash = str(summary.get("contract_sha256") or "")
    actual_hash = sha256_file(E5_CONTRACT)
    if not expected_hash or expected_hash.lower() != actual_hash.lower():
        raise RuntimeError(
            f"02e5 contract hash mismatch: "
            f"expected={expected_hash}, actual={actual_hash}"
        )

    contract = read_json(E5_CONTRACT)
    assignment = contract.get("assignment_variable") or {}
    if assignment.get("source_field") != "study.clinical_study_designation":
        raise RuntimeError("02e5 assignment field changed.")
    if assignment.get("use_case_study_arm_relation") is not False:
        raise RuntimeError("02e5 unexpectedly requires case.study_arm.")
    if assignment.get("null_case_study_arm_is_exclusion") is not False:
        raise RuntimeError("02e5 unexpectedly excludes null case.study_arm.")

    public = contract.get("public_icdc_analysis_roster_expectation") or {}
    counts = public.get("counts_by_study") or {}
    if counts != EXPECTED_COUNTS or public.get("total") != EXPECTED_TOTAL:
        raise RuntimeError(
            f"02e5 public-roster expectation changed: counts={counts}, "
            f"total={public.get('total')}"
        )

    arm_map = contract.get("arm_map") or {}
    for study, arm in EXPECTED_ARM_MAP.items():
        if (arm_map.get(study) or {}).get("gate_zero_arm") != arm:
            raise RuntimeError(f"02e5 arm mapping changed for {study}.")

    return contract, actual_hash


def load_frozen_selection() -> str:
    text = E3_SELECTION.read_text(encoding="utf-8")
    first = text.find("{")
    if first < 0:
        raise RuntimeError("02e3 frozen selection has no opening brace.")

    selection = text[first:].strip()
    if not selection.startswith("{") or not selection.endswith("}"):
        raise RuntimeError("02e3 frozen selection is malformed.")

    lower = selection.lower()
    bad = sorted(
        token for token in FORBIDDEN_SELECTION_TOKENS
        if token.lower() in lower
    )
    if bad:
        raise RuntimeError(
            "Forbidden token found in 02e3 selection: " + ", ".join(bad)
        )
    return selection


def build_runtime_policy(
    e3_contract: Dict[str, Any],
) -> Tuple[Set[str], Dict[str, Set[str]], Dict[str, str]]:
    policy = e3_contract.get("policy") or {}
    allowed_rows = policy.get("allowed_leaf_fields") or []
    relation_rows = policy.get("safe_case_relations") or []

    direct: Set[str] = set()
    leaves_by_type: Dict[str, Set[str]] = {}
    relation_to_type: Dict[str, str] = {}

    for row in allowed_rows:
        if row.get("decision") != "ALLOW":
            raise RuntimeError("Malformed 02e3 allowed-field contract.")
        parent = str(row.get("parent_type") or "")
        field = str(row.get("field_name") or "")
        if not parent or not field:
            raise RuntimeError("Malformed 02e3 allowed-field row.")
        if parent == "case":
            direct.add(field)
        else:
            leaves_by_type.setdefault(parent, set()).add(field)

    for row in relation_rows:
        if row.get("decision") != "TRAVERSE":
            raise RuntimeError("Malformed 02e3 safe-relation decision.")
        if row.get("parent_type") != "case":
            raise RuntimeError("02e3 safe relation is not rooted at case.")
        relation = str(row.get("field_name") or "")
        target = str(row.get("base_type") or "")
        if not relation or not target:
            raise RuntimeError("Malformed 02e3 safe-relation row.")
        relation_to_type[relation] = target

    required = {"case_id", "patient_id"}
    if not required.issubset(direct):
        raise RuntimeError("02e3 no longer allows required case linkage IDs.")
    if "study" not in relation_to_type:
        raise RuntimeError("02e3 no longer permits the study relation.")
    if "clinical_study_designation" not in leaves_by_type.get("study", set()):
        raise RuntimeError(
            "02e3 no longer permits study.clinical_study_designation."
        )

    return direct, leaves_by_type, relation_to_type


def safe_read_roster_candidate(path: Path) -> Set[str]:
    try:
        if not path.is_file():
            return set()
        if path.suffix.lower() not in ROSTER_EXTENSIONS:
            return set()
        if path.stat().st_size > MAX_ROSTER_FILE_BYTES:
            return set()
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return set()

    ids = {
        f"{study}-{suffix}"
        for study, suffix in CASE_ID_RE.findall(text)
    }
    return ids


def counts_by_study(case_ids: Iterable[str]) -> Dict[str, int]:
    c = Counter(case_id.split("-", 1)[0] for case_id in case_ids)
    return {study: int(c.get(study, 0)) for study in EXPECTED_COUNTS}


def exact_roster(case_ids: Set[str]) -> bool:
    return (
        len(case_ids) == EXPECTED_TOTAL
        and counts_by_study(case_ids) == EXPECTED_COUNTS
    )


def _path_is_excluded(path: Path) -> bool:
    try:
        rel_parts = path.relative_to(ROOT).parts
    except ValueError:
        return True

    if any(part.lower() in ROSTER_EXCLUDED_DIRS for part in rel_parts):
        return True

    # Never use the current 02e6 outputs/checkpoints as upstream roster evidence.
    try:
        path.relative_to(OUT_DIR)
        return True
    except ValueError:
        pass

    # Source code may contain example/test IDs and must not serve as roster evidence.
    if rel_parts and rel_parts[0].lower() == "scripts":
        return True

    return False


def _candidate_priority(path: Path) -> Tuple[int, int, str]:
    """
    Smaller tuple = better provenance candidate.

    Prefer paths with strong historical-roster hints, then ordinary data/results
    artifacts. File size is used only as a deterministic secondary preference.
    """
    rel = relative(path).lower()
    strong = any(
        hint in rel
        for hint in ("02d2", "parent_roster", "globalsearch")
    )
    medium = any(
        hint in rel
        for hint in ("02d", "roster", "icdc", "cotc")
    )
    rank = 0 if strong else (1 if medium else 2)
    try:
        size = path.stat().st_size
    except OSError:
        size = MAX_ROSTER_FILE_BYTES + 1
    return rank, size, rel


def _roster_source_record(path: Path, ids: Set[str], mode: str) -> Dict[str, Any]:
    return {
        "path": relative(path),
        "sha256": sha256_file(path),
        "unique_case_ids": len(ids),
        "counts_by_study": counts_by_study(ids),
        "discovery_mode": mode,
    }


def discover_frozen_roster() -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Recover the previously frozen public roster without assuming its filename.

    Accepted evidence, in descending preference:
      1. one local text artifact containing exactly 152/157/309;
      2. an exact study-specific pair (152 COTC021 + 157 COTC022);
      3. a consistent union of local non-source artifacts whose total is exactly
         the same 152/157/309 set.

    The function never queries ICDC and never accepts a roster with different
    counts. If candidate evidence is ambiguous, it fails closed.
    """
    candidates: List[Tuple[Path, Set[str]]] = []
    seen_paths: Set[Path] = set()

    # First pass: whole project tree, excluding environments/source/current outputs.
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if _path_is_excluded(path):
            continue
        if path.suffix.lower() not in ROSTER_EXTENSIONS:
            continue

        try:
            if path.stat().st_size > MAX_ROSTER_FILE_BYTES:
                continue
        except OSError:
            continue

        rp = path.resolve()
        if rp in seen_paths:
            continue
        seen_paths.add(rp)

        ids = safe_read_roster_candidate(path)
        if ids:
            candidates.append((path, ids))

    candidates.sort(key=lambda item: _candidate_priority(item[0]))

    # ------------------------------------------------------------------
    # Mode 1: exact 309-ID artifact.
    # ------------------------------------------------------------------
    exact = [(path, ids) for path, ids in candidates if exact_roster(ids)]
    if exact:
        canonical = exact[0][1]
        disagreements = [
            relative(path)
            for path, ids in exact
            if ids != canonical
        ]
        if disagreements:
            raise RuntimeError(
                "Multiple exact 309-case roster artifacts disagree on case IDs: "
                + ", ".join(disagreements)
            )

        provenance = [
            _roster_source_record(path, ids, "EXACT_SINGLE_FILE")
            for path, ids in exact
            if ids == canonical
        ]
        ordered = sorted(canonical, key=lambda x: (x.split("-", 1)[0], x))
        return ordered, provenance

    # ------------------------------------------------------------------
    # Mode 2: exact study-specific files. Multiple files for a study are
    # acceptable only when their complete ID sets agree.
    # ------------------------------------------------------------------
    exact_by_study: Dict[str, List[Tuple[Path, Set[str]]]] = {
        "COTC021": [],
        "COTC022": [],
    }

    for path, ids in candidates:
        counts = counts_by_study(ids)
        if len(ids) == EXPECTED_COUNTS["COTC021"] and counts == {
            "COTC021": EXPECTED_COUNTS["COTC021"],
            "COTC022": 0,
        }:
            exact_by_study["COTC021"].append((path, ids))
        if len(ids) == EXPECTED_COUNTS["COTC022"] and counts == {
            "COTC021": 0,
            "COTC022": EXPECTED_COUNTS["COTC022"],
        }:
            exact_by_study["COTC022"].append((path, ids))

    if exact_by_study["COTC021"] and exact_by_study["COTC022"]:
        canonical_study_sets: Dict[str, Set[str]] = {}
        provenance: List[Dict[str, Any]] = []

        for study in ("COTC021", "COTC022"):
            canonical = exact_by_study[study][0][1]
            disagree = [
                relative(path)
                for path, ids in exact_by_study[study]
                if ids != canonical
            ]
            if disagree:
                raise RuntimeError(
                    f"Multiple exact {study} roster artifacts disagree on case IDs: "
                    + ", ".join(disagree)
                )
            canonical_study_sets[study] = canonical
            for path, ids in exact_by_study[study]:
                provenance.append(
                    _roster_source_record(
                        path,
                        ids,
                        f"EXACT_STUDY_FILE_{study}",
                    )
                )

        combined = (
            canonical_study_sets["COTC021"]
            | canonical_study_sets["COTC022"]
        )
        if not exact_roster(combined):
            raise RuntimeError(
                "Exact study-specific roster files did not combine to "
                "152/157/309 as required."
            )

        ordered = sorted(combined, key=lambda x: (x.split("-", 1)[0], x))
        return ordered, provenance

    # ------------------------------------------------------------------
    # Mode 3: consistent artifact union.
    #
    # This is useful when 02d2 saved a roster table plus separate audit/linkage
    # artifacts rather than one monolithic 309-ID file. We only accept the union
    # if it lands EXACTLY on the already-frozen public counts.
    # ------------------------------------------------------------------
    union_ids: Set[str] = set()
    union_sources: List[Tuple[Path, Set[str]]] = []

    # Prefer files whose paths look like data/results artifacts. Because source
    # code is already excluded, all surviving IDs are observed artifact content.
    for path, ids in candidates:
        proposed = union_ids | ids
        proposed_counts = counts_by_study(proposed)

        # Never let the union exceed the frozen contract.
        if len(proposed) > EXPECTED_TOTAL:
            continue
        if any(
            proposed_counts[study] > EXPECTED_COUNTS[study]
            for study in EXPECTED_COUNTS
        ):
            continue

        if proposed != union_ids:
            union_ids = proposed
            union_sources.append((path, ids))

        if exact_roster(union_ids):
            break

    if exact_roster(union_ids):
        # Independent safety check: adding ANY remaining candidate must not reveal
        # a different COTC021/COTC022 ID outside the recovered frozen set. If it
        # does, local evidence is ambiguous and we refuse to choose silently.
        outsiders: List[Tuple[str, List[str]]] = []
        for path, ids in candidates:
            extra = sorted(ids - union_ids)
            if extra:
                outsiders.append((relative(path), extra[:10]))

        if outsiders:
            preview = "; ".join(
                f"{path}: {extra}"
                for path, extra in outsiders[:10]
            )
            raise RuntimeError(
                "Recovered a 152/157/309 roster union, but other local artifacts "
                "contain additional COTC021/COTC022-like IDs outside that set. "
                "Roster evidence is ambiguous; refusing to proceed. "
                f"Examples: {preview}"
            )

        provenance = [
            _roster_source_record(path, ids, "CONSISTENT_MULTI_FILE_UNION")
            for path, ids in union_sources
        ]
        ordered = sorted(union_ids, key=lambda x: (x.split("-", 1)[0], x))
        return ordered, provenance

    # ------------------------------------------------------------------
    # Fail with useful diagnostics rather than another opaque "0 candidates".
    # ------------------------------------------------------------------
    diagnostics = []
    for path, ids in candidates[:40]:
        diagnostics.append(
            {
                "path": relative(path),
                "unique_ids": len(ids),
                "counts": counts_by_study(ids),
            }
        )

    aggregate = set()
    for _, ids in candidates:
        aggregate |= ids

    raise RuntimeError(
        "Could not reconstruct the frozen 309-case public roster from local "
        "artifacts. No ICDC batch request was made. "
        f"Files containing COTC021/COTC022 IDs: {len(candidates)}; "
        f"aggregate unique IDs: {len(aggregate)}; "
        f"aggregate counts: {counts_by_study(aggregate)}. "
        f"Candidate diagnostics: {diagnostics}"
    )


def write_roster_tsv(case_ids: Sequence[str]) -> None:
    with ROSTER_TSV.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "case_id",
            "expected_study_designation",
            "gate_zero_arm",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for case_id in case_ids:
            study = case_id.split("-", 1)[0]
            writer.writerow(
                {
                    "case_id": case_id,
                    "expected_study_designation": study,
                    "gate_zero_arm": EXPECTED_ARM_MAP[study],
                }
            )


def validate_nested(
    relation: str,
    target_type: str,
    value: Any,
    allowed: Set[str],
) -> None:
    if value is None:
        return

    items = value if isinstance(value, list) else [value]
    for index, item in enumerate(items):
        if item is None:
            continue
        if not isinstance(item, dict):
            raise RuntimeError(
                f"case.{relation}[{index}] returned non-object "
                f"{type(item).__name__}."
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
    expected_case_id: str,
    direct: Set[str],
    leaves_by_type: Dict[str, Set[str]],
    relation_to_type: Dict[str, str],
) -> None:
    allowed_case_keys = direct | set(relation_to_type)
    unexpected = sorted(set(case_obj) - allowed_case_keys)
    if unexpected:
        raise RuntimeError(
            f"{expected_case_id}: top-level fields outside 02e3 contract: "
            + ", ".join(unexpected)
        )

    if case_obj.get("case_id") != expected_case_id:
        raise RuntimeError(
            f"Case identity mismatch: requested={expected_case_id!r}, "
            f"returned={case_obj.get('case_id')!r}"
        )

    for key, value in case_obj.items():
        if key in direct:
            if isinstance(value, (dict, list)):
                raise RuntimeError(
                    f"{expected_case_id}: direct leaf case.{key} returned nested data."
                )
            continue

        target = relation_to_type[key]
        allowed = leaves_by_type.get(target, set())
        if not allowed:
            raise RuntimeError(
                f"{expected_case_id}: relation case.{key} has no allowed leaves."
            )
        validate_nested(key, target, value, allowed)

    study = case_obj.get("study")
    if not isinstance(study, dict):
        raise RuntimeError(f"{expected_case_id}: missing study object.")

    designation = study.get("clinical_study_designation")
    expected_study = expected_case_id.split("-", 1)[0]
    if designation != expected_study:
        raise RuntimeError(
            f"{expected_case_id}: study designation mismatch: "
            f"expected={expected_study!r}, returned={designation!r}"
        )

    if designation not in EXPECTED_ARM_MAP:
        raise RuntimeError(
            f"{expected_case_id}: unexpected study designation {designation!r}."
        )


def graphql_string(value: str) -> str:
    # Case IDs are already regex-constrained, but JSON quoting is also valid
    # GraphQL string literal syntax and safely escapes special characters.
    return json.dumps(value)


def build_batch_query(case_ids: Sequence[str], selection: str) -> str:
    lines = ["query Paper6FrozenBaselineBatch {"]
    for index, case_id in enumerate(case_ids):
        alias = f"c{index:03d}"
        lines.append(
            f"  {alias}: case(case_id: {graphql_string(case_id)}) {selection}"
        )
    lines.append("}")
    return "\n".join(lines) + "\n"


def post_graphql(query: str) -> Dict[str, Any]:
    last_exc: Optional[Exception] = None

    for attempt in range(1, MAX_REQUEST_ATTEMPTS + 1):
        try:
            response = requests.post(
                ENDPOINT,
                json={"query": query},
                timeout=REQUEST_TIMEOUT_SECONDS,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "User-Agent": (
                        f"paper6-cross-species-osteosarcoma/{SCRIPT_VERSION}"
                    ),
                },
            )
            response.raise_for_status()
            payload = response.json()

            if payload.get("errors"):
                raise RuntimeError(
                    "ICDC GraphQL returned errors:\n"
                    + json.dumps(payload["errors"], indent=2, ensure_ascii=False)
                )
            if "data" not in payload:
                raise RuntimeError("ICDC GraphQL response has no data object.")
            return payload

        except (requests.RequestException, ValueError, RuntimeError) as exc:
            last_exc = exc
            if attempt >= MAX_REQUEST_ATTEMPTS:
                break
            delay = BACKOFF_SECONDS[min(attempt - 1, len(BACKOFF_SECONDS) - 1)]
            time.sleep(delay)

    raise RuntimeError(
        f"ICDC request failed after {MAX_REQUEST_ATTEMPTS} attempts: {last_exc}"
    ) from last_exc


def checkpoint_path(batch_index: int) -> Path:
    return CHECKPOINT_DIR / f"batch_{batch_index:03d}.json"


def load_checkpoint(
    path: Path,
    expected_case_ids: Sequence[str],
    roster_sha: str,
    selection_sha: str,
    e5_sha: str,
    direct: Set[str],
    leaves_by_type: Dict[str, Set[str]],
    relation_to_type: Dict[str, str],
) -> Optional[List[Dict[str, Any]]]:
    if not path.exists():
        return None

    try:
        obj = read_json(path)
    except Exception:
        return None

    if obj.get("script_version") != SCRIPT_VERSION:
        return None
    if obj.get("roster_sha256") != roster_sha:
        return None
    if obj.get("selection_sha256") != selection_sha:
        return None
    if obj.get("e5_contract_sha256") != e5_sha:
        return None
    if obj.get("case_ids") != list(expected_case_ids):
        return None

    cases = obj.get("cases")
    if not isinstance(cases, list) or len(cases) != len(expected_case_ids):
        return None

    # Revalidate cached values against the current frozen contract.
    for expected_case_id, case_obj in zip(expected_case_ids, cases):
        if not isinstance(case_obj, dict):
            return None
        validate_case(
            case_obj,
            expected_case_id,
            direct,
            leaves_by_type,
            relation_to_type,
        )

    return cases


def fetch_batch(
    batch_index: int,
    case_ids: Sequence[str],
    selection: str,
    roster_sha: str,
    selection_sha: str,
    e5_sha: str,
    direct: Set[str],
    leaves_by_type: Dict[str, Set[str]],
    relation_to_type: Dict[str, str],
) -> Tuple[List[Dict[str, Any]], bool]:
    path = checkpoint_path(batch_index)

    cached = load_checkpoint(
        path,
        case_ids,
        roster_sha,
        selection_sha,
        e5_sha,
        direct,
        leaves_by_type,
        relation_to_type,
    )
    if cached is not None:
        return cached, True

    query = build_batch_query(case_ids, selection)
    payload = post_graphql(query)
    data = payload.get("data")
    if not isinstance(data, dict):
        raise RuntimeError(f"Batch {batch_index}: response data is not an object.")

    cases: List[Dict[str, Any]] = []

    for index, expected_case_id in enumerate(case_ids):
        alias = f"c{index:03d}"
        value = data.get(alias)
        if not isinstance(value, list):
            raise RuntimeError(
                f"Batch {batch_index} {expected_case_id}: "
                f"expected list, found {type(value).__name__}."
            )
        if len(value) != 1:
            raise RuntimeError(
                f"Batch {batch_index} {expected_case_id}: "
                f"expected exactly one result, found {len(value)}."
            )
        case_obj = value[0]
        if not isinstance(case_obj, dict):
            raise RuntimeError(
                f"Batch {batch_index} {expected_case_id}: result is not object."
            )

        validate_case(
            case_obj,
            expected_case_id,
            direct,
            leaves_by_type,
            relation_to_type,
        )
        cases.append(case_obj)

    checkpoint = {
        "script_version": SCRIPT_VERSION,
        "created_utc": now_utc(),
        "batch_index": batch_index,
        "case_ids": list(case_ids),
        "roster_sha256": roster_sha,
        "selection_sha256": selection_sha,
        "e5_contract_sha256": e5_sha,
        "cases": cases,
    }
    write_json(path, checkpoint)
    return cases, False


def scalar_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def flatten_case(
    case_obj: Dict[str, Any],
    direct: Set[str],
    leaves_by_type: Dict[str, Set[str]],
    relation_to_type: Dict[str, str],
) -> Dict[str, str]:
    row: Dict[str, str] = {}

    for field in sorted(direct):
        row[f"case.{field}"] = scalar_cell(case_obj.get(field))

    for relation in sorted(relation_to_type):
        target = relation_to_type[relation]
        fields = sorted(leaves_by_type.get(target, set()))
        value = case_obj.get(relation)

        for field in fields:
            col = f"case.{relation}.{field}"
            if isinstance(value, list):
                vals = [
                    item.get(field) if isinstance(item, dict) else None
                    for item in value
                ]
                row[col] = scalar_cell(vals)
            elif isinstance(value, dict):
                row[col] = scalar_cell(value.get(field))
            elif value is None:
                row[col] = ""
            else:
                raise RuntimeError(
                    f"Unexpected relation value type for {relation}: "
                    f"{type(value).__name__}"
                )

    study = case_obj["study"]["clinical_study_designation"]
    row["frozen_gate_zero_arm"] = EXPECTED_ARM_MAP[study]
    return row


def write_final_outputs(
    case_ids: Sequence[str],
    cases_by_id: Dict[str, Dict[str, Any]],
    direct: Set[str],
    leaves_by_type: Dict[str, Set[str]],
    relation_to_type: Dict[str, str],
) -> None:
    # JSONL: one validated allowed case object per line.
    with JSONL_OUT.open("w", encoding="utf-8", newline="\n") as handle:
        for case_id in case_ids:
            record = {
                "case_id": case_id,
                "gate_zero_arm": EXPECTED_ARM_MAP[case_id.split("-", 1)[0]],
                "case": cases_by_id[case_id],
            }
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False))
            handle.write("\n")

    flat_rows = [
        flatten_case(
            cases_by_id[case_id],
            direct,
            leaves_by_type,
            relation_to_type,
        )
        for case_id in case_ids
    ]

    fieldnames: List[str] = []
    seen: Set[str] = set()
    for row in flat_rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    # Stable ordering: case linkage first, frozen arm last after GraphQL columns.
    preferred = [
        "case.case_id",
        "case.patient_id",
    ]
    ordered: List[str] = []
    for key in preferred:
        if key in seen:
            ordered.append(key)

    for key in sorted(seen):
        if key not in ordered and key != "frozen_gate_zero_arm":
            ordered.append(key)
    if "frozen_gate_zero_arm" in seen:
        ordered.append("frozen_gate_zero_arm")

    with FLAT_TSV_OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered, delimiter="\t")
        writer.writeheader()
        writer.writerows(flat_rows)


def audit_materialization(
    case_ids: Sequence[str],
    cases_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    if len(case_ids) != EXPECTED_TOTAL:
        raise RuntimeError(f"Roster size changed: {len(case_ids)}")
    if len(cases_by_id) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"Materialized unique case count is {len(cases_by_id)}, "
            f"expected {EXPECTED_TOTAL}."
        )
    if set(case_ids) != set(cases_by_id):
        missing = sorted(set(case_ids) - set(cases_by_id))
        extra = sorted(set(cases_by_id) - set(case_ids))
        raise RuntimeError(
            f"Materialized roster mismatch. missing={missing}, extra={extra}"
        )

    designation_counts = Counter()
    arm_counts = Counter()
    null_counts = Counter()
    relation_cardinality: Dict[str, List[int]] = defaultdict(list)

    patient_ids: List[str] = []
    sample_ids: List[str] = []

    for case_id in case_ids:
        case = cases_by_id[case_id]
        study = case["study"]["clinical_study_designation"]
        designation_counts[study] += 1
        arm_counts[EXPECTED_ARM_MAP[study]] += 1

        patient_id = case.get("patient_id")
        if patient_id not in (None, ""):
            patient_ids.append(str(patient_id))

        for key, value in case.items():
            if value is None:
                null_counts[f"case.{key}"] += 1
            if isinstance(value, list):
                relation_cardinality[key].append(len(value))
            elif isinstance(value, dict):
                relation_cardinality[key].append(1)
            elif key not in ("case_id", "patient_id"):
                relation_cardinality[key].append(0)

        samples = case.get("samples")
        if isinstance(samples, list):
            for sample in samples:
                if isinstance(sample, dict):
                    sid = sample.get("sample_id")
                    if sid not in (None, ""):
                        sample_ids.append(str(sid))

    observed_counts = {
        study: int(designation_counts.get(study, 0))
        for study in EXPECTED_COUNTS
    }
    if observed_counts != EXPECTED_COUNTS:
        raise RuntimeError(
            f"Returned study-designation counts mismatch: "
            f"observed={observed_counts}, expected={EXPECTED_COUNTS}"
        )

    patient_duplicate_count = len(patient_ids) - len(set(patient_ids))
    sample_duplicate_count = len(sample_ids) - len(set(sample_ids))

    relation_summary = {}
    for relation, values in sorted(relation_cardinality.items()):
        if not values:
            continue
        relation_summary[relation] = {
            "min": min(values),
            "max": max(values),
            "zero_count": sum(v == 0 for v in values),
            "one_count": sum(v == 1 for v in values),
            "gt1_count": sum(v > 1 for v in values),
        }

    return {
        "cases": EXPECTED_TOTAL,
        "study_designation_counts": observed_counts,
        "gate_zero_arm_counts": {
            arm: int(count)
            for arm, count in sorted(arm_counts.items())
        },
        "patient_id_nonempty_count": len(patient_ids),
        "patient_id_duplicate_count": patient_duplicate_count,
        "sample_id_count": len(sample_ids),
        "sample_id_duplicate_count": sample_duplicate_count,
        "relation_cardinality": relation_summary,
    }


def main() -> None:
    print("=" * 112)
    print("Paper 6 - materialize frozen ICDC baseline/linkage metadata for COTC021/COTC022")
    print("=" * 112)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Endpoint: {ENDPOINT}")
    print()
    print("Safety contract:")
    print("  Public cases expected: 309")
    print("  COTC021 expected: 152")
    print("  COTC022 expected: 157")
    print("  Treatment assignment source: frozen 02e5 study designation")
    print("  Query selection source: hash-locked 02e3 artifact")
    print("  Outcome/response/follow-up fields requested: NO")
    print("  Longitudinal treatment/adverse-event fields requested: NO")
    print("  Metastatic-origin/sample-chronology/necropsy fields requested: NO")
    print("  Patient name/initial fields requested: NO")
    print("  Omics requested/downloaded: NO")
    print()

    run_started = now_utc()

    e3_contract, e3_hashes = verify_e3()
    _, e5_sha = verify_e5()
    selection = load_frozen_selection()
    selection_sha = sha256_file(E3_SELECTION)

    direct, leaves_by_type, relation_to_type = build_runtime_policy(e3_contract)

    print("Frozen upstream verification:")
    print("  02e3 baseline field contract: PASS")
    print("  02e3 artifact hashes: PASS")
    print("  02e5 study-as-arm contract: PASS")
    print("  02e5 contract hash: PASS")
    print()

    print("Discovering exact frozen 02d2 public roster...")
    case_ids, roster_sources = discover_frozen_roster()

    roster_counts = counts_by_study(case_ids)
    if roster_counts != EXPECTED_COUNTS or len(case_ids) != EXPECTED_TOTAL:
        raise RuntimeError("Discovered roster failed frozen count contract.")

    write_roster_tsv(case_ids)
    roster_sha = sha256_file(ROSTER_TSV)

    print("Frozen roster discovery: PASS")
    print(f"  exact roster source file(s): {len(roster_sources)}")
    print(f"  COTC021: {roster_counts['COTC021']}")
    print(f"  COTC022: {roster_counts['COTC022']}")
    print(f"  total:   {len(case_ids)}")
    print(f"  normalized roster SHA256: {roster_sha}")
    print()

    total_batches = (len(case_ids) + BATCH_SIZE - 1) // BATCH_SIZE
    cases_by_id: Dict[str, Dict[str, Any]] = {}
    reused_batches = 0
    fetched_batches = 0

    print(f"Materializing {len(case_ids)} cases in {total_batches} validated batches...")

    for batch_index in range(total_batches):
        start = batch_index * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(case_ids))
        batch_case_ids = case_ids[start:end]

        cases, reused = fetch_batch(
            batch_index=batch_index,
            case_ids=batch_case_ids,
            selection=selection,
            roster_sha=roster_sha,
            selection_sha=selection_sha,
            e5_sha=e5_sha,
            direct=direct,
            leaves_by_type=leaves_by_type,
            relation_to_type=relation_to_type,
        )

        for case_id, case_obj in zip(batch_case_ids, cases):
            if case_id in cases_by_id:
                raise RuntimeError(f"Duplicate materialized case_id: {case_id}")
            cases_by_id[case_id] = case_obj

        if reused:
            reused_batches += 1
            status = "CHECKPOINT"
        else:
            fetched_batches += 1
            status = "FETCHED"

        print(
            f"  batch {batch_index + 1:02d}/{total_batches:02d}: "
            f"{len(batch_case_ids):2d} cases [{status}]"
        )

    print()
    print("All batches returned. Running final fail-closed audit...")
    audit = audit_materialization(case_ids, cases_by_id)

    write_final_outputs(
        case_ids,
        cases_by_id,
        direct,
        leaves_by_type,
        relation_to_type,
    )

    manifest = {
        "script_version": SCRIPT_VERSION,
        "status": "PASS",
        "run_started_utc": run_started,
        "run_finished_utc": now_utc(),
        "endpoint": ENDPOINT,
        "frozen_provenance": {
            "02e3_hashes": e3_hashes,
            "02e3_selection_sha256": selection_sha,
            "02e5_contract_sha256": e5_sha,
            "normalized_roster_sha256": roster_sha,
            "roster_sources": roster_sources,
        },
        "materialization": {
            "cases": EXPECTED_TOTAL,
            "counts_by_study": EXPECTED_COUNTS,
            "batch_size": BATCH_SIZE,
            "total_batches": total_batches,
            "fetched_batches_this_run": fetched_batches,
            "reused_checkpoint_batches_this_run": reused_batches,
        },
        "audit": audit,
        "safety": {
            "outcome_response_followup_fields_requested": False,
            "longitudinal_treatment_adverse_event_fields_requested": False,
            "metastatic_origin_sample_chronology_necropsy_fields_requested": False,
            "patient_name_initial_fields_requested": False,
            "omics_requested_or_downloaded": False,
        },
    }
    write_json(MANIFEST_JSON, manifest)

    final_hashes = {
        relative(ROSTER_TSV): sha256_file(ROSTER_TSV),
        relative(JSONL_OUT): sha256_file(JSONL_OUT),
        relative(FLAT_TSV_OUT): sha256_file(FLAT_TSV_OUT),
        relative(MANIFEST_JSON): sha256_file(MANIFEST_JSON),
    }

    summary = {
        "script_version": SCRIPT_VERSION,
        "status": "PASS",
        "run_started_utc": run_started,
        "run_finished_utc": now_utc(),
        "endpoint": ENDPOINT,
        "cases_materialized": EXPECTED_TOTAL,
        "study_designation_counts": audit["study_designation_counts"],
        "gate_zero_arm_counts": audit["gate_zero_arm_counts"],
        "response_contract_validation": "PASS_ALL_309",
        "fetched_batches_this_run": fetched_batches,
        "reused_checkpoint_batches_this_run": reused_batches,
        "outcome_response_followup_fields_requested": False,
        "longitudinal_treatment_adverse_event_fields_requested": False,
        "metastatic_origin_sample_chronology_necropsy_fields_requested": False,
        "patient_name_initial_fields_requested": False,
        "omics_requested_or_downloaded": False,
        "final_artifact_hashes": final_hashes,
    }
    write_json(SUMMARY_JSON, summary)

    print("-" * 112)
    print("Final materialization audit")
    print("-" * 112)
    print("Unique cases materialized: 309")
    print(
        f"Study designation counts: "
        f"COTC021={audit['study_designation_counts']['COTC021']}, "
        f"COTC022={audit['study_designation_counts']['COTC022']}"
    )
    print(
        "Gate Zero arm counts: "
        f"SOC_PLUS_RAPAMYCIN={audit['gate_zero_arm_counts'].get('SOC_PLUS_RAPAMYCIN', 0)}, "
        f"SOC_CONTROL={audit['gate_zero_arm_counts'].get('SOC_CONTROL', 0)}"
    )
    print("Frozen response allow-list validation: PASS_ALL_309")
    print(f"Batches fetched this run: {fetched_batches}")
    print(f"Batches reused from checkpoints: {reused_batches}")
    print()
    print("Linkage diagnostics [descriptive only; not exclusion rules]:")
    print(f"  non-empty patient_id values: {audit['patient_id_nonempty_count']}")
    print(f"  duplicate patient_id values: {audit['patient_id_duplicate_count']}")
    print(f"  sample_id values: {audit['sample_id_count']}")
    print(f"  duplicate sample_id values: {audit['sample_id_duplicate_count']}")
    print()

    print("Selected relation cardinalities:")
    for relation in (
        "canine_individual",
        "demographic",
        "diagnoses",
        "enrollment",
        "registrations",
        "samples",
        "study",
        "study_arm",
    ):
        info = audit["relation_cardinality"].get(relation)
        if info:
            print(
                f"  {relation}: min={info['min']}, max={info['max']}, "
                f"zero={info['zero_count']}, one={info['one_count']}, "
                f">1={info['gt1_count']}"
            )

    print()
    print(f"Artifacts: {OUT_DIR.relative_to(ROOT)}")
    print(f"  {ROSTER_TSV.name}")
    print(f"  {JSONL_OUT.name}")
    print(f"  {FLAT_TSV_OUT.name}")
    print(f"  {MANIFEST_JSON.name}")
    print(f"  {SUMMARY_JSON.name}")
    print(f"  checkpoints\\batch_*.json")
    print()
    print("Outcome/response/follow-up fields requested/read: NO")
    print("Longitudinal treatment/adverse-event fields requested/read: NO")
    print("Metastatic-origin/sample-chronology/necropsy fields requested/read: NO")
    print("Patient name/initial fields requested/read: NO")
    print("Omics requested/downloaded: NO")
    print()
    print("02e6 public COTC021/COTC022 baseline materialization: PASS")
    print("=" * 112)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 112, file=sys.stderr)
        print("02e6 public COTC021/COTC022 baseline materialization: FAIL", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print("=" * 112, file=sys.stderr)
        raise
