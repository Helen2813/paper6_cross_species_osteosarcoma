#!/usr/bin/env python3
"""
Paper 6 - audit 309 -> DOG² RNA-186 selection before any outcome access.

Purpose
-------
The public COTC021/COTC022 parent roster (309 cases) has already been
materialized under the frozen 02e3 baseline contract. This script determines
which exact 186 cases comprise the DOG² transcriptomic subset and quantifies
whether transcriptomic availability is associated with randomized study/arm
or with permitted baseline covariates.

This script is OFFLINE:
  * no ICDC/API access;
  * no outcome/response/follow-up access;
  * no treatment-administration access;
  * no post-baseline sample-annotation access;
  * no omics values are read.

Only case IDs and the already-frozen 02e6 baseline metadata are used.

Scientific interpretation
-------------------------
This is a PRE-GATE-ZERO selection/linkage audit. It does not estimate treatment
effects and it does not decide a biological claim.

Hard PASS criteria are structural only:
  1. exactly 186 unique DOG² case IDs are recovered from an authoritative local
     mapping artifact;
  2. all 186 are a subset of the frozen public 309-case roster;
  3. both randomized study groups are represented among the 186.

Balance thresholds below are frozen BEFORE this aggregate audit and are
diagnostic, not exclusion criteria:
  * absolute arm-specific selection-rate difference >= 0.10 -> warning;
  * numeric |SMD| >= 0.20 -> warning;
  * categorical max absolute category-proportion difference >= 0.20 -> warning;
  * absolute missingness difference >= 0.10 -> warning.

A warning means the later transport/representation analysis must explicitly
account for the selection structure; it does NOT fail the cohort.

No command-line arguments are used.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


SCRIPT_VERSION = "02e7-audit-dog2-186-selection-v1-no-cli"

ROOT = Path(__file__).resolve().parents[1]

E6_DIR = ROOT / "results" / "icdc_baseline_batch" / "02e6"
E6_SUMMARY = E6_DIR / "summary.json"
E6_JSONL = E6_DIR / "icdc_baseline_309.jsonl"
E6_ROSTER = E6_DIR / "frozen_public_roster_309.tsv"
E6_MANIFEST = E6_DIR / "materialization_manifest.json"

OUT_DIR = ROOT / "results" / "dog2_selection_audit" / "02e7"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DOG2_ROSTER_TSV = OUT_DIR / "dog2_rna186_roster.tsv"
ARM_AUDIT_TSV = OUT_DIR / "arm_selection_audit.tsv"
BALANCE_TSV = OUT_DIR / "baseline_selection_balance.tsv"
AUDIT_JSON = OUT_DIR / "selection_audit.json"
SUMMARY_JSON = OUT_DIR / "summary.json"

EXPECTED_PARENT_COUNTS = {"COTC021": 152, "COTC022": 157}
EXPECTED_PARENT_TOTAL = 309
EXPECTED_DOG2_TOTAL = 186

CASE_ID_RE = re.compile(r"\b(COTC021|COTC022)-([A-Za-z0-9][A-Za-z0-9._-]*)\b")

CANDIDATE_EXTENSIONS = {
    ".json", ".jsonl", ".csv", ".tsv", ".txt", ".yaml", ".yml", ".md", ".log"
}
EXCLUDED_DIRS = {
    ".venv", "venv", ".git", "__pycache__", "node_modules", ".idea", ".vscode"
}
MAX_CANDIDATE_BYTES = 100 * 1024 * 1024

# Prefer artifacts known to belong to the authoritative DOG² ID-resolution step.
STRONG_PATH_HINTS = (
    "02a",
    "resolve_dog2",
    "authoritative",
    "subject_mapping",
    "resolved",
)
MEDIUM_PATH_HINTS = (
    "dog2",
    "dog",
    "rna186",
    "rna_186",
    "mapping",
    "selected",
    "paper5",
)

# Diagnostic thresholds, frozen pre-audit.
ARM_SELECTION_RATE_DIFF_WARN = 0.10
NUMERIC_ABS_SMD_WARN = 0.20
CATEGORICAL_MAX_PROP_DIFF_WARN = 0.20
MISSINGNESS_DIFF_WARN = 0.10

NUMERIC_VARIABLES = {
    "age_at_enrollment": ("demographic", "patient_age_at_enrollment"),
    "weight": ("demographic", "weight"),
}

CATEGORICAL_VARIABLES = {
    "breed": ("demographic", "breed"),
    "sex": ("demographic", "sex"),
    "neutered_indicator": ("demographic", "neutered_indicator"),
    "disease_term": ("diagnosis", "disease_term"),
    "primary_disease_site": ("diagnosis", "primary_disease_site"),
    "stage_of_disease": ("diagnosis", "stage_of_disease"),
    "histology_cytopathology": ("diagnosis", "histology_cytopathology"),
    "histological_grade": ("diagnosis", "histological_grade"),
    "registering_institution": ("enrollment", "registering_institution"),
    "veterinary_medical_center": ("enrollment", "veterinary_medical_center"),
}

COMPARISONS = (
    "FULL_ARM_BALANCE",
    "DOG2_ARM_BALANCE",
    "SELECTION_OVERALL",
    "SELECTION_WITHIN_COTC021",
    "SELECTION_WITHIN_COTC022",
)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def counts_by_study(case_ids: Iterable[str]) -> Dict[str, int]:
    counter = Counter(case_id.split("-", 1)[0] for case_id in case_ids)
    return {
        study: int(counter.get(study, 0))
        for study in ("COTC021", "COTC022")
    }


def verify_e6() -> Dict[str, str]:
    for path in (E6_SUMMARY, E6_JSONL, E6_ROSTER, E6_MANIFEST):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required 02e6 artifact: {path}\n"
                "Run scripts\\02e6_materialize_icdc_baseline_309_v2.py first."
            )

    summary = read_json(E6_SUMMARY)
    if summary.get("status") != "PASS":
        raise RuntimeError("02e6 summary status is not PASS.")
    if summary.get("cases_materialized") != EXPECTED_PARENT_TOTAL:
        raise RuntimeError("02e6 did not materialize exactly 309 cases.")
    if summary.get("study_designation_counts") != EXPECTED_PARENT_COUNTS:
        raise RuntimeError(
            "02e6 study designation counts no longer match 152/157."
        )
    if summary.get("response_contract_validation") != "PASS_ALL_309":
        raise RuntimeError("02e6 did not validate all 309 responses.")

    for key in (
        "outcome_response_followup_fields_requested",
        "longitudinal_treatment_adverse_event_fields_requested",
        "metastatic_origin_sample_chronology_necropsy_fields_requested",
        "patient_name_initial_fields_requested",
        "omics_requested_or_downloaded",
    ):
        if summary.get(key) is not False:
            raise RuntimeError(f"02e6 safety provenance mismatch: {key}")

    hashes = summary.get("final_artifact_hashes") or {}
    verified: Dict[str, str] = {}
    for path in (E6_JSONL, E6_ROSTER, E6_MANIFEST):
        expected = lookup_hash(hashes, path)
        actual = sha256_file(path)
        if not expected or expected.lower() != actual.lower():
            raise RuntimeError(
                f"02e6 hash mismatch for {path.name}: "
                f"expected={expected}, actual={actual}"
            )
        verified[relative(path)] = actual

    return verified


def load_parent_cases() -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    ordered_ids: List[str] = []
    cases: Dict[str, Dict[str, Any]] = {}

    with E6_JSONL.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            case_id = str(obj.get("case_id") or "")
            case_obj = obj.get("case")

            if not CASE_ID_RE.fullmatch(case_id):
                raise RuntimeError(
                    f"02e6 JSONL line {line_number}: invalid case_id {case_id!r}"
                )
            if not isinstance(case_obj, dict):
                raise RuntimeError(
                    f"02e6 JSONL line {line_number}: case is not an object."
                )
            if case_obj.get("case_id") != case_id:
                raise RuntimeError(
                    f"02e6 JSONL line {line_number}: case_id mismatch."
                )
            if case_id in cases:
                raise RuntimeError(f"Duplicate parent case_id: {case_id}")

            study = case_obj.get("study")
            if not isinstance(study, dict):
                raise RuntimeError(f"{case_id}: missing study object.")
            designation = study.get("clinical_study_designation")
            if designation != case_id.split("-", 1)[0]:
                raise RuntimeError(
                    f"{case_id}: study designation mismatch {designation!r}"
                )

            ordered_ids.append(case_id)
            cases[case_id] = case_obj

    if len(cases) != EXPECTED_PARENT_TOTAL:
        raise RuntimeError(
            f"02e6 JSONL contains {len(cases)} unique cases, expected 309."
        )
    if counts_by_study(cases) != EXPECTED_PARENT_COUNTS:
        raise RuntimeError("02e6 JSONL parent study counts changed.")

    return ordered_ids, cases


def path_excluded(path: Path) -> bool:
    try:
        parts = path.relative_to(ROOT).parts
    except ValueError:
        return True

    if any(part.lower() in EXCLUDED_DIRS for part in parts):
        return True
    if parts and parts[0].lower() == "scripts":
        return True

    # Do not discover the selected roster from our own output.
    try:
        path.relative_to(OUT_DIR)
        return True
    except ValueError:
        pass

    # 02e6 contains the complete 309 set, not the DOG² selected set.
    try:
        path.relative_to(E6_DIR)
        return True
    except ValueError:
        pass

    return False


def candidate_priority(path: Path) -> Tuple[int, int, str]:
    rel = relative(path).lower()
    if any(hint in rel for hint in STRONG_PATH_HINTS):
        rank = 0
    elif any(hint in rel for hint in MEDIUM_PATH_HINTS):
        rank = 1
    else:
        rank = 2
    try:
        size = path.stat().st_size
    except OSError:
        size = MAX_CANDIDATE_BYTES + 1
    return rank, size, rel


def extract_case_ids(path: Path) -> Set[str]:
    try:
        if path.stat().st_size > MAX_CANDIDATE_BYTES:
            return set()
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return set()

    return {
        f"{study}-{suffix}"
        for study, suffix in CASE_ID_RE.findall(text)
    }


def discover_dog2_186(parent_ids: Set[str]) -> Tuple[List[str], List[Dict[str, Any]]]:
    candidates: List[Tuple[Path, Set[str]]] = []

    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path_excluded(path):
            continue
        if path.suffix.lower() not in CANDIDATE_EXTENSIONS:
            continue

        ids = extract_case_ids(path)
        if not ids:
            continue

        # Any ID outside the frozen public roster means this file cannot be the
        # exact authoritative DOG²-186 mapping artifact.
        if not ids.issubset(parent_ids):
            continue

        candidates.append((path, ids))

    candidates.sort(key=lambda item: candidate_priority(item[0]))

    exact = [
        (path, ids)
        for path, ids in candidates
        if len(ids) == EXPECTED_DOG2_TOTAL
    ]

    if not exact:
        diagnostics = [
            {
                "path": relative(path),
                "unique_parent_case_ids": len(ids),
                "counts_by_study": counts_by_study(ids),
                "priority": candidate_priority(path)[0],
            }
            for path, ids in candidates[:50]
        ]
        raise RuntimeError(
            "Could not locate an authoritative local artifact containing exactly "
            "186 unique COTC021/COTC022 case IDs that are a subset of the frozen "
            "309-case public roster. No outcomes or omics were accessed. "
            f"Candidate diagnostics: {diagnostics}"
        )

    # Prefer strongest provenance rank, but all exact candidates must agree.
    best_rank = min(candidate_priority(path)[0] for path, _ in exact)
    best = [
        (path, ids)
        for path, ids in exact
        if candidate_priority(path)[0] == best_rank
    ]

    canonical = best[0][1]
    disagreements = [
        relative(path)
        for path, ids in exact
        if ids != canonical
    ]
    if disagreements:
        raise RuntimeError(
            "Multiple exact 186-ID local mapping artifacts disagree. "
            "Refusing to choose a DOG² cohort silently. Conflicting files: "
            + ", ".join(disagreements)
        )

    provenance = [
        {
            "path": relative(path),
            "sha256": sha256_file(path),
            "unique_case_ids": len(ids),
            "counts_by_study": counts_by_study(ids),
            "priority_rank": candidate_priority(path)[0],
        }
        for path, ids in exact
        if ids == canonical
    ]

    ordered = sorted(canonical, key=lambda x: (x.split("-", 1)[0], x))
    return ordered, provenance


def first_diagnosis(case: Dict[str, Any]) -> Dict[str, Any]:
    diagnoses = case.get("diagnoses")
    if not isinstance(diagnoses, list) or len(diagnoses) != 1:
        raise RuntimeError(
            f"{case.get('case_id')}: expected exactly one frozen baseline diagnosis."
        )
    diagnosis = diagnoses[0]
    if not isinstance(diagnosis, dict):
        raise RuntimeError(f"{case.get('case_id')}: diagnosis is not object.")
    return diagnosis


def baseline_record(case_id: str, case: Dict[str, Any], selected: bool) -> Dict[str, Any]:
    demographic = case.get("demographic")
    enrollment = case.get("enrollment")

    if not isinstance(demographic, dict):
        raise RuntimeError(f"{case_id}: demographic missing.")
    if not isinstance(enrollment, dict):
        raise RuntimeError(f"{case_id}: enrollment missing.")

    diagnosis = first_diagnosis(case)
    study = case["study"]["clinical_study_designation"]

    rec: Dict[str, Any] = {
        "case_id": case_id,
        "study": study,
        "selected_dog2": selected,
    }

    for name, (_, field) in NUMERIC_VARIABLES.items():
        rec[name] = demographic.get(field)

    for name, (source, field) in CATEGORICAL_VARIABLES.items():
        if source == "demographic":
            rec[name] = demographic.get(field)
        elif source == "diagnosis":
            rec[name] = diagnosis.get(field)
        elif source == "enrollment":
            rec[name] = enrollment.get(field)
        else:
            raise RuntimeError(f"Unknown variable source: {source}")

    return rec


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def numeric_values(records: Sequence[Dict[str, Any]], variable: str) -> List[float]:
    values: List[float] = []
    for record in records:
        value = record.get(variable)
        if is_missing(value):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise RuntimeError(
                f"Non-numeric value for {variable}: {value!r}"
            )
        if math.isfinite(number):
            values.append(number)
    return values


def mean_or_none(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(statistics.fmean(values))


def sample_variance(values: Sequence[float]) -> Optional[float]:
    if len(values) < 2:
        return None
    return float(statistics.variance(values))


def numeric_smd(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    if not a or not b:
        return None

    mean_a = statistics.fmean(a)
    mean_b = statistics.fmean(b)

    var_a = statistics.variance(a) if len(a) >= 2 else 0.0
    var_b = statistics.variance(b) if len(b) >= 2 else 0.0
    pooled = math.sqrt((var_a + var_b) / 2.0)

    if pooled == 0:
        return 0.0 if mean_a == mean_b else None
    return float((mean_a - mean_b) / pooled)


def missing_fraction(records: Sequence[Dict[str, Any]], variable: str) -> float:
    if not records:
        return float("nan")
    missing = sum(is_missing(record.get(variable)) for record in records)
    return missing / len(records)


def normalized_category(value: Any) -> str:
    if is_missing(value):
        return "<MISSING>"
    return str(value).strip()


def category_distribution(
    records: Sequence[Dict[str, Any]],
    variable: str,
) -> Dict[str, float]:
    if not records:
        return {}
    counts = Counter(normalized_category(record.get(variable)) for record in records)
    n = len(records)
    return {
        category: count / n
        for category, count in sorted(counts.items())
    }


def categorical_distance(
    a_records: Sequence[Dict[str, Any]],
    b_records: Sequence[Dict[str, Any]],
    variable: str,
) -> Tuple[Optional[float], Optional[float], Dict[str, float], Dict[str, float]]:
    if not a_records or not b_records:
        return None, None, {}, {}

    da = category_distribution(a_records, variable)
    db = category_distribution(b_records, variable)
    categories = sorted(set(da) | set(db))

    diffs = [
        abs(da.get(category, 0.0) - db.get(category, 0.0))
        for category in categories
    ]
    max_abs = max(diffs) if diffs else 0.0
    tv = 0.5 * sum(diffs)
    return float(max_abs), float(tv), da, db


def comparison_groups(
    records: Sequence[Dict[str, Any]],
    comparison: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str, str]:
    if comparison == "FULL_ARM_BALANCE":
        a = [r for r in records if r["study"] == "COTC021"]
        b = [r for r in records if r["study"] == "COTC022"]
        return a, b, "COTC021", "COTC022"

    if comparison == "DOG2_ARM_BALANCE":
        selected = [r for r in records if r["selected_dog2"]]
        a = [r for r in selected if r["study"] == "COTC021"]
        b = [r for r in selected if r["study"] == "COTC022"]
        return a, b, "COTC021_selected", "COTC022_selected"

    if comparison == "SELECTION_OVERALL":
        a = [r for r in records if r["selected_dog2"]]
        b = [r for r in records if not r["selected_dog2"]]
        return a, b, "DOG2_selected", "not_selected"

    if comparison == "SELECTION_WITHIN_COTC021":
        subset = [r for r in records if r["study"] == "COTC021"]
        a = [r for r in subset if r["selected_dog2"]]
        b = [r for r in subset if not r["selected_dog2"]]
        return a, b, "COTC021_selected", "COTC021_not_selected"

    if comparison == "SELECTION_WITHIN_COTC022":
        subset = [r for r in records if r["study"] == "COTC022"]
        a = [r for r in subset if r["selected_dog2"]]
        b = [r for r in subset if not r["selected_dog2"]]
        return a, b, "COTC022_selected", "COTC022_not_selected"

    raise RuntimeError(f"Unknown comparison {comparison}")


def fisher_two_sided_2x2(a: int, b: int, c: int, d: int) -> float:
    """
    Two-sided Fisher exact p-value using the probability <= observed-probability
    definition. Standard-library only; counts here are small.
    """
    row1 = a + b
    row2 = c + d
    col1 = a + c
    total = row1 + row2

    if total == 0:
        return 1.0

    low = max(0, col1 - row2)
    high = min(row1, col1)

    denom = math.comb(total, col1)

    def prob(x: int) -> float:
        return (
            math.comb(row1, x)
            * math.comb(row2, col1 - x)
            / denom
        )

    observed = prob(a)
    p = 0.0
    tolerance = 1e-15
    for x in range(low, high + 1):
        px = prob(x)
        if px <= observed + tolerance:
            p += px
    return min(1.0, float(p))


def arm_selection_audit(
    records: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    arm_rows: List[Dict[str, Any]] = []

    counts: Dict[str, Dict[str, int]] = {}
    rates: Dict[str, float] = {}

    for study in ("COTC021", "COTC022"):
        arm_records = [r for r in records if r["study"] == study]
        selected = sum(bool(r["selected_dog2"]) for r in arm_records)
        total = len(arm_records)
        not_selected = total - selected

        counts[study] = {
            "selected": selected,
            "not_selected": not_selected,
            "total": total,
        }
        rates[study] = selected / total

        arm_rows.append(
            {
                "study": study,
                "gate_zero_arm": (
                    "SOC_PLUS_RAPAMYCIN" if study == "COTC021"
                    else "SOC_CONTROL"
                ),
                "parent_n": total,
                "dog2_selected_n": selected,
                "not_selected_n": not_selected,
                "selection_rate": rates[study],
            }
        )

    diff = rates["COTC021"] - rates["COTC022"]
    abs_diff = abs(diff)

    a = counts["COTC021"]["selected"]
    b = counts["COTC021"]["not_selected"]
    c = counts["COTC022"]["selected"]
    d = counts["COTC022"]["not_selected"]

    fisher_p = fisher_two_sided_2x2(a, b, c, d)

    risk_ratio = (
        rates["COTC021"] / rates["COTC022"]
        if rates["COTC022"] > 0 else None
    )

    odds_ratio = None
    if b > 0 and c > 0 and d >= 0:
        if c * b != 0:
            odds_ratio = (a * d) / (b * c)

    summary = {
        "selection_rates": rates,
        "selection_rate_difference_COTC021_minus_COTC022": diff,
        "absolute_selection_rate_difference": abs_diff,
        "risk_ratio_COTC021_vs_COTC022": risk_ratio,
        "odds_ratio_COTC021_vs_COTC022": odds_ratio,
        "fisher_exact_two_sided_p": fisher_p,
        "warning_threshold": ARM_SELECTION_RATE_DIFF_WARN,
        "warning": abs_diff >= ARM_SELECTION_RATE_DIFF_WARN,
    }
    return arm_rows, summary


def baseline_balance_rows(
    records: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    for comparison in COMPARISONS:
        a, b, label_a, label_b = comparison_groups(records, comparison)
        if not a or not b:
            raise RuntimeError(
                f"{comparison}: one comparison group is empty "
                f"({label_a} n={len(a)}, {label_b} n={len(b)})."
            )

        for variable in NUMERIC_VARIABLES:
            va = numeric_values(a, variable)
            vb = numeric_values(b, variable)
            smd = numeric_smd(va, vb)
            miss_a = missing_fraction(a, variable)
            miss_b = missing_fraction(b, variable)
            miss_diff = miss_a - miss_b

            warning_reasons = []
            if smd is not None and abs(smd) >= NUMERIC_ABS_SMD_WARN:
                warning_reasons.append("ABS_SMD")
            if abs(miss_diff) >= MISSINGNESS_DIFF_WARN:
                warning_reasons.append("MISSINGNESS")

            row = {
                "comparison": comparison,
                "variable": variable,
                "variable_type": "numeric",
                "group_a": label_a,
                "group_b": label_b,
                "n_a": len(a),
                "n_b": len(b),
                "nonmissing_a": len(va),
                "nonmissing_b": len(vb),
                "mean_a": mean_or_none(va),
                "mean_b": mean_or_none(vb),
                "smd_a_minus_b": smd,
                "abs_smd": abs(smd) if smd is not None else None,
                "max_abs_category_prop_diff": None,
                "total_variation_distance": None,
                "missing_fraction_a": miss_a,
                "missing_fraction_b": miss_b,
                "missingness_difference_a_minus_b": miss_diff,
                "distribution_a_json": "",
                "distribution_b_json": "",
                "warning": bool(warning_reasons),
                "warning_reasons": ",".join(warning_reasons),
            }
            rows.append(row)

            if warning_reasons:
                warnings.append(
                    {
                        "comparison": comparison,
                        "variable": variable,
                        "reasons": warning_reasons,
                    }
                )

        for variable in CATEGORICAL_VARIABLES:
            max_diff, tv, da, db = categorical_distance(a, b, variable)
            miss_a = missing_fraction(a, variable)
            miss_b = missing_fraction(b, variable)
            miss_diff = miss_a - miss_b

            warning_reasons = []
            if (
                max_diff is not None
                and max_diff >= CATEGORICAL_MAX_PROP_DIFF_WARN
            ):
                warning_reasons.append("MAX_CATEGORY_PROP_DIFF")
            if abs(miss_diff) >= MISSINGNESS_DIFF_WARN:
                warning_reasons.append("MISSINGNESS")

            row = {
                "comparison": comparison,
                "variable": variable,
                "variable_type": "categorical",
                "group_a": label_a,
                "group_b": label_b,
                "n_a": len(a),
                "n_b": len(b),
                "nonmissing_a": len(a) - round(miss_a * len(a)),
                "nonmissing_b": len(b) - round(miss_b * len(b)),
                "mean_a": None,
                "mean_b": None,
                "smd_a_minus_b": None,
                "abs_smd": None,
                "max_abs_category_prop_diff": max_diff,
                "total_variation_distance": tv,
                "missing_fraction_a": miss_a,
                "missing_fraction_b": miss_b,
                "missingness_difference_a_minus_b": miss_diff,
                "distribution_a_json": json.dumps(
                    da, sort_keys=True, ensure_ascii=False
                ),
                "distribution_b_json": json.dumps(
                    db, sort_keys=True, ensure_ascii=False
                ),
                "warning": bool(warning_reasons),
                "warning_reasons": ",".join(warning_reasons),
            }
            rows.append(row)

            if warning_reasons:
                warnings.append(
                    {
                        "comparison": comparison,
                        "variable": variable,
                        "reasons": warning_reasons,
                    }
                )

    return rows, warnings


def write_dog2_roster(
    selected_ids: Sequence[str],
    parent_cases: Dict[str, Dict[str, Any]],
) -> None:
    with DOG2_ROSTER_TSV.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "case_id",
            "study",
            "gate_zero_arm",
            "patient_id",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()

        for case_id in selected_ids:
            case = parent_cases[case_id]
            study = case["study"]["clinical_study_designation"]
            writer.writerow(
                {
                    "case_id": case_id,
                    "study": study,
                    "gate_zero_arm": (
                        "SOC_PLUS_RAPAMYCIN"
                        if study == "COTC021"
                        else "SOC_CONTROL"
                    ),
                    "patient_id": case.get("patient_id") or "",
                }
            )


def write_tsv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty TSV: {path}")

    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    print("=" * 112)
    print("Paper 6 - audit public 309 -> DOG² RNA-186 selection before outcomes")
    print("=" * 112)
    print(f"Script version: {SCRIPT_VERSION}")
    print()
    print("Execution contract:")
    print("  Network/API access in 02e7: NO")
    print("  Outcome/response/follow-up values read: NO")
    print("  Treatment administration values read: NO")
    print("  Post-baseline sample annotations read: NO")
    print("  Omics values read: NO")
    print("  Omics file contents read: NO")
    print("  Inputs: case IDs + frozen 02e6 baseline/linkage metadata only")
    print()

    started = now_utc()

    e6_hashes = verify_e6()
    parent_order, parent_cases = load_parent_cases()
    parent_set = set(parent_cases)

    print("02e6 verification:")
    print("  materialization status: PASS")
    print("  final artifact hashes: PASS")
    print("  unique parent cases: 309")
    print("  COTC021: 152")
    print("  COTC022: 157")
    print()

    print("Discovering authoritative DOG² RNA-186 case mapping...")
    selected_ids, mapping_sources = discover_dog2_186(parent_set)
    selected_set = set(selected_ids)

    if len(selected_set) != EXPECTED_DOG2_TOTAL:
        raise RuntimeError("DOG² selected set is not exactly 186 cases.")
    if not selected_set.issubset(parent_set):
        raise RuntimeError("DOG² selected set is not a subset of public 309.")

    selected_counts = counts_by_study(selected_ids)
    if selected_counts["COTC021"] == 0 or selected_counts["COTC022"] == 0:
        raise RuntimeError(
            "Structural Gate Zero failure: one randomized study group is absent "
            "from the DOG² RNA-186 subset."
        )

    print("DOG² authoritative mapping: PASS")
    print(f"  exact mapping source file(s): {len(mapping_sources)}")
    print(f"  unique DOG² cases: {len(selected_ids)}")
    print(f"  COTC021 selected: {selected_counts['COTC021']}")
    print(f"  COTC022 selected: {selected_counts['COTC022']}")
    print()

    write_dog2_roster(selected_ids, parent_cases)

    records = [
        baseline_record(
            case_id,
            parent_cases[case_id],
            case_id in selected_set,
        )
        for case_id in parent_order
    ]

    arm_rows, arm_summary = arm_selection_audit(records)
    balance_rows, balance_warnings = baseline_balance_rows(records)

    write_tsv(ARM_AUDIT_TSV, arm_rows)
    write_tsv(BALANCE_TSV, balance_rows)

    selected_warning_rows = [
        warning
        for warning in balance_warnings
        if warning["comparison"] in {
            "SELECTION_OVERALL",
            "SELECTION_WITHIN_COTC021",
            "SELECTION_WITHIN_COTC022",
        }
    ]

    arm_balance_warning_rows = [
        warning
        for warning in balance_warnings
        if warning["comparison"] == "DOG2_ARM_BALANCE"
    ]

    adjustment_signal = (
        bool(arm_summary["warning"])
        or bool(selected_warning_rows)
    )

    gate_status = (
        "PASS_SELECTION_STRUCTURE_REQUIRES_EXPLICIT_ADJUSTMENT"
        if adjustment_signal
        else "PASS_NO_MAJOR_BASELINE_SELECTION_SIGNAL"
    )

    audit = {
        "script_version": SCRIPT_VERSION,
        "status": "PASS",
        "run_started_utc": started,
        "run_finished_utc": now_utc(),
        "scientific_stage": "PRE_OUTCOME_PRE_GATE_ZERO_SELECTION_AUDIT",
        "parent_roster": {
            "n": EXPECTED_PARENT_TOTAL,
            "counts_by_study": EXPECTED_PARENT_COUNTS,
        },
        "dog2_selected_roster": {
            "n": EXPECTED_DOG2_TOTAL,
            "counts_by_study": selected_counts,
            "mapping_sources": mapping_sources,
            "roster_sha256": sha256_file(DOG2_ROSTER_TSV),
        },
        "arm_selection": arm_summary,
        "diagnostic_thresholds_frozen_pre_audit": {
            "arm_selection_rate_difference_warning": ARM_SELECTION_RATE_DIFF_WARN,
            "numeric_abs_smd_warning": NUMERIC_ABS_SMD_WARN,
            "categorical_max_proportion_difference_warning": (
                CATEGORICAL_MAX_PROP_DIFF_WARN
            ),
            "missingness_difference_warning": MISSINGNESS_DIFF_WARN,
            "thresholds_are_exclusion_criteria": False,
        },
        "diagnostic_warning_counts": {
            "selection_related_balance_warnings": len(selected_warning_rows),
            "dog2_arm_balance_warnings": len(arm_balance_warning_rows),
            "all_balance_warnings": len(balance_warnings),
        },
        "gate_status": gate_status,
        "hard_structural_checks": {
            "exact_186_unique_ids": True,
            "all_186_subset_of_309": True,
            "both_randomized_studies_represented": True,
        },
        "scientific_interpretation": {
            "diagnostic_warnings_fail_cohort": False,
            "outcomes_may_be_opened_after_this_script": False,
            "next_stage": (
                "freeze and run expression-level Gate Zero arm/confounding audit "
                "before any outcome access"
            ),
        },
        "upstream_02e6_hashes": e6_hashes,
    }
    write_json(AUDIT_JSON, audit)

    final_hashes = {
        relative(DOG2_ROSTER_TSV): sha256_file(DOG2_ROSTER_TSV),
        relative(ARM_AUDIT_TSV): sha256_file(ARM_AUDIT_TSV),
        relative(BALANCE_TSV): sha256_file(BALANCE_TSV),
        relative(AUDIT_JSON): sha256_file(AUDIT_JSON),
    }

    summary = {
        "script_version": SCRIPT_VERSION,
        "status": "PASS",
        "run_started_utc": started,
        "run_finished_utc": now_utc(),
        "network_access": False,
        "outcome_response_followup_values_read": False,
        "treatment_administration_values_read": False,
        "postbaseline_sample_annotations_read": False,
        "omics_values_read": False,
        "omics_file_contents_read": False,
        "parent_n": EXPECTED_PARENT_TOTAL,
        "dog2_selected_n": EXPECTED_DOG2_TOTAL,
        "dog2_counts_by_study": selected_counts,
        "arm_selection_absolute_rate_difference": (
            arm_summary["absolute_selection_rate_difference"]
        ),
        "arm_selection_fisher_p": arm_summary["fisher_exact_two_sided_p"],
        "selection_related_balance_warning_count": len(selected_warning_rows),
        "dog2_arm_balance_warning_count": len(arm_balance_warning_rows),
        "gate_status": gate_status,
        "final_artifact_hashes": final_hashes,
    }
    write_json(SUMMARY_JSON, summary)

    print("-" * 112)
    print("309 -> DOG² RNA-186 selection audit")
    print("-" * 112)
    print(f"Parent public roster: {EXPECTED_PARENT_TOTAL}")
    print(f"DOG² selected roster: {EXPECTED_DOG2_TOTAL}")
    print()
    print("Selected cases by randomized study:")
    print(f"  COTC021 / SOC_PLUS_RAPAMYCIN: {selected_counts['COTC021']}")
    print(f"  COTC022 / SOC_CONTROL:        {selected_counts['COTC022']}")
    print()
    print("Transcriptomic selection rates:")
    print(
        f"  COTC021: "
        f"{arm_summary['selection_rates']['COTC021']:.4f}"
    )
    print(
        f"  COTC022: "
        f"{arm_summary['selection_rates']['COTC022']:.4f}"
    )
    print(
        "  absolute rate difference: "
        f"{arm_summary['absolute_selection_rate_difference']:.4f}"
    )
    print(
        "  Fisher exact two-sided p: "
        f"{arm_summary['fisher_exact_two_sided_p']:.6g}"
    )
    print(
        "  arm-selection warning: "
        f"{'YES' if arm_summary['warning'] else 'NO'}"
    )
    print()
    print("Baseline diagnostic warnings:")
    print(
        "  selection-related warnings: "
        f"{len(selected_warning_rows)}"
    )
    print(
        "  DOG² arm-balance warnings: "
        f"{len(arm_balance_warning_rows)}"
    )
    print(
        "  all recorded balance warnings: "
        f"{len(balance_warnings)}"
    )
    print()
    print(f"Pre-Gate-Zero selection status: {gate_status}")
    print()
    print("Important:")
    print("  Diagnostic imbalance does NOT fail the cohort.")
    print("  Outcomes remain CLOSED.")
    print("  Next step is expression-level arm/confounding Gate Zero.")
    print()
    print(f"Artifacts: {OUT_DIR.relative_to(ROOT)}")
    print(f"  {DOG2_ROSTER_TSV.name}")
    print(f"  {ARM_AUDIT_TSV.name}")
    print(f"  {BALANCE_TSV.name}")
    print(f"  {AUDIT_JSON.name}")
    print(f"  {SUMMARY_JSON.name}")
    print()
    print("Network/API access in 02e7: NO")
    print("Outcome/response/follow-up values read: NO")
    print("Treatment administration values read: NO")
    print("Post-baseline sample annotations read: NO")
    print("Omics values read: NO")
    print("Omics file contents read: NO")
    print()
    print("02e7 DOG² selection/linkage audit: PASS")
    print("=" * 112)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 112, file=sys.stderr)
        print("02e7 DOG² selection/linkage audit: FAIL", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print("=" * 112, file=sys.stderr)
        raise
