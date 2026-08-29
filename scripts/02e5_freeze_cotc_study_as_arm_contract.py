#!/usr/bin/env python3
"""
Paper 6 - freeze COTC021/COTC022 randomized study-as-arm contract.

OFFLINE only. No ICDC/API access and no new patient values.

Frozen Gate Zero interpretation:
  COTC021 -> standard of care + adjuvant rapamycin/sirolimus
  COTC022 -> contemporaneous standard-of-care control

Treatment assignment is represented by study.clinical_study_designation,
not by case.study_arm. A null case.study_arm is NOT an exclusion criterion.

Expected current public ICDC roster:
  COTC021 = 152
  COTC022 = 157
  total   = 309

The 324 dogs reported in the original randomized trial literature are context
only and are not the expected public ICDC batch size.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

SCRIPT_VERSION = "02e5-freeze-cotc-study-as-arm-contract-v1-no-cli"
ENDPOINT = "https://caninecommons.cancer.gov/v1/graphql/"
TEST_CASE = "COTC021-0101"

ROOT = Path(__file__).resolve().parents[1]

E3 = ROOT / "results" / "icdc_baseline_contract" / "02e3"
E3_SUMMARY = E3 / "summary.json"
E3_CONTRACT = E3 / "baseline_field_contract.json"
E3_SELECTION = E3 / "baseline_selection.graphql"

E4 = ROOT / "results" / "icdc_baseline_probe" / "02e4"
E4_SUMMARY = E4 / "summary.json"
E4_RESPONSE = E4 / "cotc021_0101_allowed_response.json"

OUT = ROOT / "results" / "cotc_trial_arm_contract" / "02e5"
OUT.mkdir(parents=True, exist_ok=True)
CONTRACT_JSON = OUT / "cotc_study_as_arm_contract.json"
SUMMARY_JSON = OUT / "summary.json"

PUBLIC_COUNTS = {"COTC021": 152, "COTC022": 157}
PUBLIC_TOTAL = 309
TRIAL_ENROLLED_CONTEXT_ONLY = 324

ARM_MAP = {
    "COTC021": {
        "gate_zero_arm": "SOC_PLUS_RAPAMYCIN",
        "treatment_strategy": "standard_of_care_plus_adjuvant_rapamycin",
        "rapamycin_indicator": 1,
        "control_indicator": 0,
    },
    "COTC022": {
        "gate_zero_arm": "SOC_CONTROL",
        "treatment_strategy": "contemporaneous_standard_of_care_control",
        "rapamycin_indicator": 0,
        "control_indicator": 1,
    },
}

PROVENANCE = [
    {
        "source": "CRDC Components: Updates June 2026",
        "url": "https://datacommons.cancer.gov/news/crdc-components-updates-june-2026",
        "supports": (
            "COTC021 is Rapamycin + standard-of-care (152 public subjects); "
            "COTC022 is the contemporaneous standard-of-care control "
            "(157 public subjects)."
        ),
    },
    {
        "source": (
            "Large Scale Comparative Deconvolution Analysis of the Canine and "
            "Human Osteosarcoma Tumor Microenvironment"
        ),
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10557692/",
        "supports": (
            "COTC021/022 dogs were randomized to standard of care or standard "
            "of care plus adjuvant sirolimus/rapamycin; original trial "
            "enrollment was 324 dogs."
        ),
    },
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def find_hash(mapping: Dict[str, Any], path: Path) -> str:
    rel = str(path.relative_to(ROOT))
    v = mapping.get(rel)
    if v is None:
        v = mapping.get(rel.replace("/", "\\"))
    if isinstance(v, dict):
        v = v.get("sha256")
    return str(v or "")


def verify_e3() -> Dict[str, str]:
    for p in (E3_SUMMARY, E3_CONTRACT, E3_SELECTION):
        if not p.exists():
            raise FileNotFoundError(f"Missing 02e3 artifact: {p}")

    s = load_json(E3_SUMMARY)
    if s.get("status") != "PASS":
        raise RuntimeError("02e3 status is not PASS.")
    if s.get("network_access") is not False:
        raise RuntimeError("02e3 network_access is not False.")

    for key in (
        "case_values_read",
        "outcome_values_read",
        "response_values_read",
        "follow_up_values_read",
        "treatment_administration_values_read",
        "omics_values_read_or_downloaded",
    ):
        if s.get(key) is not False:
            raise RuntimeError(f"02e3 provenance mismatch: {key}")

    hashes = s.get("output_hashes") or {}
    verified = {}
    for p in (E3_CONTRACT, E3_SELECTION):
        exp = find_hash(hashes, p)
        act = sha256(p)
        if not exp or exp.lower() != act.lower():
            raise RuntimeError(
                f"02e3 hash mismatch for {p.name}: expected={exp}, actual={act}"
            )
        verified[str(p.relative_to(ROOT))] = act

    c = load_json(E3_CONTRACT)
    if (c.get("source_02e2") or {}).get("endpoint") != ENDPOINT:
        raise RuntimeError("02e3 production endpoint lock mismatch.")

    allowed = c.get("policy", {}).get("allowed_leaf_fields", [])
    paths = {
        str(r.get("field_path"))
        for r in allowed
        if r.get("decision") == "ALLOW"
    }
    if "study.clinical_study_designation" not in paths:
        raise RuntimeError(
            "02e3 does not allow study.clinical_study_designation."
        )
    return verified


def verify_e4() -> Dict[str, Any]:
    for p in (E4_SUMMARY, E4_RESPONSE):
        if not p.exists():
            raise FileNotFoundError(f"Missing 02e4 artifact: {p}")

    s = load_json(E4_SUMMARY)
    if s.get("status") != "PASS":
        raise RuntimeError("02e4 status is not PASS.")
    if s.get("response_contract_validation") != "PASS":
        raise RuntimeError("02e4 response-contract validation did not PASS.")
    if s.get("endpoint") != ENDPOINT:
        raise RuntimeError("02e4 endpoint mismatch.")
    if s.get("case_id_requested") != TEST_CASE:
        raise RuntimeError("02e4 test-case mismatch.")
    if s.get("cases_requested") != 1 or s.get("cases_returned") != 1:
        raise RuntimeError("02e4 was not an exact one-case probe.")

    for key in (
        "outcome_response_followup_fields_requested",
        "longitudinal_treatment_or_adverse_event_fields_requested",
        "metastatic_origin_sample_chronology_necropsy_fields_requested",
        "patient_name_or_initial_fields_requested",
        "omics_requested_or_downloaded",
    ):
        if s.get(key) is not False:
            raise RuntimeError(f"02e4 safety mismatch: {key}")

    exp = find_hash(s.get("artifacts") or {}, E4_RESPONSE)
    act = sha256(E4_RESPONSE)
    if not exp or exp.lower() != act.lower():
        raise RuntimeError(
            f"02e4 response hash mismatch: expected={exp}, actual={act}"
        )

    obj = load_json(E4_RESPONSE)
    case = obj.get("case") or {}
    if case.get("case_id") != TEST_CASE:
        raise RuntimeError("02e4 response case_id mismatch.")
    study = case.get("study")
    if not isinstance(study, dict):
        raise RuntimeError("02e4 response lacks study object.")
    designation = study.get("clinical_study_designation")
    if designation != "COTC021":
        raise RuntimeError(
            f"02e4 study designation mismatch: {designation!r}"
        )

    return {
        "response_sha256": act,
        "observed_study_designation": designation,
        "observed_study_arm_is_null": case.get("study_arm") is None,
    }


def main() -> None:
    print("=" * 108)
    print("Paper 6 - freeze COTC021/COTC022 randomized study-as-arm contract")
    print("=" * 108)
    print(f"Script version: {SCRIPT_VERSION}")
    print()
    print("Execution contract:")
    print("  Network/API access in 02e5: NO")
    print("  New ICDC case/patient values read: NO")
    print("  Outcome/response/follow-up values read: NO")
    print("  Treatment administration values read: NO")
    print("  Post-baseline sample annotations read: NO")
    print("  Omics values read/downloaded: NO")
    print()

    started = datetime.now(timezone.utc).isoformat()
    e3 = verify_e3()
    e4 = verify_e4()

    print("Upstream verification:")
    print("  02e3 baseline/linkage contract: PASS")
    print("  02e3 frozen hashes: PASS")
    print("  02e4 controlled one-case probe: PASS")
    print("  02e4 frozen response hash: PASS")
    print()

    if sum(PUBLIC_COUNTS.values()) != PUBLIC_TOTAL:
        raise RuntimeError("Public-count arithmetic mismatch.")
    if set(ARM_MAP) != set(PUBLIC_COUNTS):
        raise RuntimeError("Arm map and public study set differ.")

    for study, rec in ARM_MAP.items():
        if rec["rapamycin_indicator"] + rec["control_indicator"] != 1:
            raise RuntimeError(f"Non-exclusive treatment indicators: {study}")

    contract = {
        "script_version": SCRIPT_VERSION,
        "status": "PASS",
        "run_started_utc": started,
        "scientific_stage": "PRE_OUTCOME_GATE_ZERO_DESIGN_FREEZE",
        "assignment_variable": {
            "source_field": "study.clinical_study_designation",
            "interpretation": "randomized_trial_treatment_group",
            "use_case_study_arm_relation": False,
            "null_case_study_arm_is_exclusion": False,
        },
        "arm_map": ARM_MAP,
        "public_icdc_analysis_roster_expectation": {
            "counts_by_study": PUBLIC_COUNTS,
            "total": PUBLIC_TOTAL,
            "must_be_reverified_before_batch_materialization": True,
        },
        "original_trial_context": {
            "reported_enrolled_dogs": TRIAL_ENROLLED_CONTEXT_ONLY,
            "analysis_roster_target": False,
            "note": (
                "The original trial enrollment count is contextual only. "
                "Do not require 324 public ICDC cases."
            ),
        },
        "gate_zero": {
            "contrast": "SOC_PLUS_RAPAMYCIN_vs_SOC_CONTROL",
            "treatment_group_field": "study.clinical_study_designation",
            "study_arm_graphql_field_required": False,
        },
        "hard_constraints": {
            "do_not_infer_arm_from_outcomes": True,
            "do_not_infer_arm_from_follow_up": True,
            "do_not_infer_arm_from_treatment_administration": True,
            "do_not_infer_arm_from_sample_origin_or_chronology": True,
            "do_not_exclude_for_null_case_study_arm": True,
            "require_exact_cotc021_or_cotc022_designation": True,
            "require_public_batch_count_reverification": True,
            "no_outcome_access_before_gate_zero_design_lock": True,
        },
        "scientific_provenance": PROVENANCE,
        "upstream": {
            "02e3_verified_hashes": e3,
            "02e4_verification": e4,
        },
    }
    dump_json(CONTRACT_JSON, contract)
    csha = sha256(CONTRACT_JSON)

    summary = {
        "script_version": SCRIPT_VERSION,
        "status": "PASS",
        "run_started_utc": started,
        "run_finished_utc": datetime.now(timezone.utc).isoformat(),
        "network_access": False,
        "new_icdc_case_values_read": False,
        "outcome_response_followup_values_read": False,
        "treatment_administration_values_read": False,
        "postbaseline_sample_annotations_read": False,
        "omics_values_read_or_downloaded": False,
        "assignment_field": "study.clinical_study_designation",
        "study_arm_relation_required": False,
        "null_study_arm_exclusion": False,
        "expected_public_counts": PUBLIC_COUNTS,
        "expected_public_total": PUBLIC_TOTAL,
        "original_trial_enrolled_context_only": TRIAL_ENROLLED_CONTEXT_ONLY,
        "contract_path": str(CONTRACT_JSON.relative_to(ROOT)),
        "contract_sha256": csha,
    }
    dump_json(SUMMARY_JSON, summary)

    print("-" * 108)
    print("Frozen randomized study-as-arm contract")
    print("-" * 108)
    print("Treatment assignment source:")
    print("  study.clinical_study_designation")
    print()
    print("Frozen mapping:")
    print("  COTC021 -> SOC_PLUS_RAPAMYCIN")
    print("  COTC022 -> SOC_CONTROL")
    print()
    print("GraphQL case.study_arm:")
    print("  required for treatment assignment: NO")
    print("  null value is exclusion criterion: NO")
    print()
    print("Public ICDC roster target for next batch:")
    print(f"  COTC021: {PUBLIC_COUNTS['COTC021']}")
    print(f"  COTC022: {PUBLIC_COUNTS['COTC022']}")
    print(f"  total:   {PUBLIC_TOTAL}")
    print()
    print("Original trial enrollment:")
    print(f"  {TRIAL_ENROLLED_CONTEXT_ONLY} dogs [CONTEXT ONLY; NOT public batch target]")
    print()
    print("Gate Zero contrast:")
    print("  SOC_PLUS_RAPAMYCIN vs SOC_CONTROL")
    print()
    print(f"Artifacts: {OUT.relative_to(ROOT)}")
    print(f"  {CONTRACT_JSON.name}")
    print(f"  {SUMMARY_JSON.name}")
    print(f"Contract SHA256: {csha}")
    print()
    print("Network/API access in 02e5: NO")
    print("Outcome/response/follow-up values read: NO")
    print("Treatment administration values read: NO")
    print("Post-baseline sample annotations read: NO")
    print("Omics values read/downloaded: NO")
    print()
    print("02e5 randomized study-as-arm contract freeze: PASS")
    print("=" * 108)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 108, file=sys.stderr)
        print("02e5 randomized study-as-arm contract freeze: FAIL", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print("=" * 108, file=sys.stderr)
        raise
