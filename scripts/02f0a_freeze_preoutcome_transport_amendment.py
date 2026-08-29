#!/usr/bin/env python3
"""
Paper 6 - freeze pre-outcome amendment superseding the affected portions of 02f0.

This amendment is created AFTER the 02f0_v3 provenance preflight confirmed that
the Paper-6 02f0 strict/primary feature universe inherited the row universe of the
Paper-4 outcome-aware RNA candidate/evidence pipeline.

The amendment does NOT delete, overwrite, rewrite, or retroactively invalidate 02f0.
Instead it records, before Paper-6 outcome-model fitting:

1. what remains historically valid from 02f0;
2. what is scientifically superseded;
3. the measured reason for supersession;
4. the required outcome-blind rebuild path;
5. the outcome-access firewall;
6. the Paper-4 frozen-comparator rule;
7. the source-gate endpoint hierarchy and no-post-hoc-switch rule;
8. the sacrificial-human-cohort premise-test policy;
9. the architecture-selection anti-forking-path policy;
10. downstream numbering and execution order.

No command-line arguments are used.

SAFETY:
- no network access;
- no clinical table values read;
- no outcome/response/follow-up values read;
- no treatment-administration values read;
- no expression matrix values read;
- no model fitting;
- no target splits;
- no existing artifact modified.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


SCRIPT_VERSION = "02f0a-freeze-preoutcome-transport-amendment-v1-no-cli"

ROOT = Path(__file__).resolve().parents[1]

# Confirmed provenance audit.
PREFLIGHT_DIR = ROOT / "results" / "transport_contract" / "02f0_preflight_audit_v3"
PREFLIGHT_AUDIT = PREFLIGHT_DIR / "universe_lineage_audit.json"
PREFLIGHT_SUMMARY = PREFLIGHT_DIR / "summary.json"
PREFLIGHT_LINEAGE = PREFLIGHT_DIR / "universe_lineage_summary.tsv"
PREFLIGHT_SET_AUDIT = PREFLIGHT_DIR / "gene_set_lineage_audit.tsv"
PREFLIGHT_SOURCE_LINEAGE = PREFLIGHT_DIR / "source_lineage_evidence.tsv"

# Existing 02f0 execution that is preserved unchanged.
OLD_DIR = ROOT / "results" / "transport_contract" / "02f0"
OLD_CONTRACT = OLD_DIR / "cross_species_transport_contract.json"
OLD_SUMMARY = OLD_DIR / "summary.json"
OLD_STRICT_UNIVERSE = OLD_DIR / "strict_ortholog_universe.tsv"
OLD_FEATURE_SETS = OLD_DIR / "transport_feature_sets.tsv"
OLD_FEATURE_SUMMARY = OLD_DIR / "feature_set_summary.tsv"
OLD_SCRIPT = ROOT / "scripts" / "02f0_freeze_cross_species_transport_contract.py"

# Gate Zero context, still valid.
GATE_DIR = ROOT / "results" / "expression_gate_zero" / "02e9"
GATE_SUMMARY = GATE_DIR / "summary.json"
GATE_RESULTS = GATE_DIR / "gate_zero_results.json"

# Output: separate immutable amendment record.
OUT_DIR = ROOT / "results" / "transport_contract" / "02f0a_amendment"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AMENDMENT_JSON = OUT_DIR / "transport_contract_amendment.json"
SUPERSESSION_TSV = OUT_DIR / "supersession_scope.tsv"
POLICY_TSV = OUT_DIR / "downstream_policy.tsv"
README_TXT = OUT_DIR / "AMENDMENT_README.txt"
SUMMARY_JSON = OUT_DIR / "summary.json"


EXPECTED_PREFLIGHT_STATUS = "PASS"
EXPECTED_PREFLIGHT_VERDICT = "CONFIRMED_OUTCOME_AWARE_ROW_UNIVERSE_INHERITANCE"
EXPECTED_PREFLIGHT_SCIENTIFIC_STATUS = (
    "02F0_PRIMARY_FEATURE_UNIVERSE_NOT_ELIGIBLE_FOR_PAPER6_PRIMARY_ANALYSIS"
)
EXPECTED_GATE = "GREEN_NO_MATERIAL_ARM_EXPRESSION_SEPARATION"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_lines(values: Iterable[str]) -> str:
    text = "\n".join(str(x) for x in values) + "\n"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def write_tsv(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            values = []
            for col in columns:
                value = row.get(col, "")
                if isinstance(value, bool):
                    text = "TRUE" if value else "FALSE"
                elif value is None:
                    text = ""
                else:
                    text = str(value)
                text = text.replace("\t", " ").replace("\r", " ").replace("\n", " ")
                values.append(text)
            handle.write("\t".join(values) + "\n")


def require_file(path: Path) -> Path:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Required frozen artifact missing: {path}")
    return path


def clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def require_bool(mapping: Dict[str, Any], key: str, expected: bool) -> None:
    observed = mapping.get(key)
    if observed is not expected:
        raise RuntimeError(
            f"Safety/provenance mismatch for {key}: expected {expected}, observed {observed!r}"
        )


def require_int(mapping: Dict[str, Any], key: str, expected: int) -> None:
    try:
        observed = int(mapping.get(key))
    except Exception as exc:
        raise RuntimeError(f"Missing/non-integer measured count {key}") from exc
    if observed != expected:
        raise RuntimeError(
            f"Measured-count mismatch for {key}: expected {expected}, observed {observed}"
        )


def main() -> None:
    started = now_utc()

    print("=" * 120)
    print("Paper 6 - freeze pre-outcome amendment to the 02f0 transport contract")
    print("=" * 120)
    print(f"Script version: {SCRIPT_VERSION}")
    print()
    print("Safety / execution contract:")
    print("  Network/API access: NO")
    print("  Clinical table values read: NO")
    print("  Outcome/response/follow-up values read: NO")
    print("  Treatment-administration values read: NO")
    print("  Expression matrix values read: NO")
    print("  Existing 02f0 artifacts modified/deleted: NO")
    print("  Model fitting: NO")
    print("  Human target splits generated: NO")
    print()

    required = [
        PREFLIGHT_AUDIT,
        PREFLIGHT_SUMMARY,
        PREFLIGHT_LINEAGE,
        PREFLIGHT_SET_AUDIT,
        PREFLIGHT_SOURCE_LINEAGE,
        OLD_CONTRACT,
        OLD_SUMMARY,
        OLD_STRICT_UNIVERSE,
        OLD_FEATURE_SETS,
        OLD_FEATURE_SUMMARY,
        OLD_SCRIPT,
        GATE_SUMMARY,
        GATE_RESULTS,
    ]
    for path in required:
        require_file(path)

    audit = read_json(PREFLIGHT_AUDIT)
    preflight_summary = read_json(PREFLIGHT_SUMMARY)
    old_contract = read_json(OLD_CONTRACT)
    old_summary = read_json(OLD_SUMMARY)
    gate_summary = read_json(GATE_SUMMARY)
    gate_results = read_json(GATE_RESULTS)

    # ----------------------------------------------------------------------------------
    # Confirm the provenance audit before writing an immutable amendment.
    # ----------------------------------------------------------------------------------
    if clean(audit.get("status")) != EXPECTED_PREFLIGHT_STATUS:
        raise RuntimeError(
            f"02f0 v3 preflight status is not PASS: {audit.get('status')!r}"
        )
    if clean(audit.get("verdict")) != EXPECTED_PREFLIGHT_VERDICT:
        raise RuntimeError(
            "02f0 v3 preflight verdict is not the required confirmed inheritance verdict."
        )
    if clean(audit.get("scientific_status")) != EXPECTED_PREFLIGHT_SCIENTIFIC_STATUS:
        raise RuntimeError(
            "02f0 v3 preflight scientific status does not match the expected supersession trigger."
        )
    if clean(preflight_summary.get("status")) != "PASS":
        raise RuntimeError("02f0 v3 preflight summary status is not PASS.")

    preflight_hashes = {
        "audit_json": sha256_file(PREFLIGHT_AUDIT),
        "summary_json": sha256_file(PREFLIGHT_SUMMARY),
        "universe_lineage_summary_tsv": sha256_file(PREFLIGHT_LINEAGE),
        "gene_set_lineage_audit_tsv": sha256_file(PREFLIGHT_SET_AUDIT),
        "source_lineage_evidence_tsv": sha256_file(PREFLIGHT_SOURCE_LINEAGE),
    }

    measured = audit.get("measured_counts") or {}

    # These are the measured local facts from the confirmed v3 audit.
    require_int(measured, "dog2_expression_features", 21016)
    require_int(measured, "paper4_master_unique_genes", 5013)
    require_int(measured, "paper4_with_orthologs_unique_genes", 5013)
    require_int(measured, "paper4_ortholog_qc_unique_genes", 5013)
    require_int(measured, "paper4_qc_strict_unique_canine_genes", 3391)
    require_int(measured, "paper6_02f0_strict_unique_canine_genes", 3391)
    require_int(measured, "paper6_02f0_primary_unique_canine_genes", 3386)
    require_int(measured, "biomart_rows", 34159)
    require_int(measured, "biomart_unique_dog_symbols", 16955)
    require_int(measured, "biomart_dog_symbols_with_any_human_homolog", 16063)
    require_int(measured, "biomart_dog_symbols_with_one2one_human_ortholog", 15318)

    dfi_header_count = len(audit.get("dfi_derived_master_header_columns") or [])
    os_header_count = len(audit.get("os_derived_master_header_columns") or [])
    model_header_count = len(audit.get("model_selection_master_header_columns") or [])
    all_outcome_header_count = len(audit.get("outcome_aware_master_header_columns") or [])

    if (dfi_header_count, os_header_count, model_header_count, all_outcome_header_count) != (
        38,
        38,
        67,
        80,
    ):
        raise RuntimeError(
            "Confirmed preflight header-evidence counts differ from the expected local audit: "
            f"DFI={dfi_header_count}, OS={os_header_count}, "
            f"model/selection={model_header_count}, all={all_outcome_header_count}"
        )

    checks = audit.get("checks") or {}
    required_true_checks = [
        "paper4_master_to_with_orthologs_exact_gene_set_equality",
        "paper4_master_to_ortholog_qc_exact_gene_set_equality",
        "paper6_02f0_strict_gene_set_is_subset_of_paper4_master_evidence_universe",
        "paper6_02f0_primary_gene_set_is_subset_of_paper4_master_evidence_universe",
        "paper6_02f0_strict_gene_set_equals_qc_strict_gene_set",
        "paper6_upstream_lock_ortholog_qc_path_matches_paper4_qc",
        "paper6_upstream_lock_ortholog_qc_hash_matches_paper4_qc",
        "paper6_02f0_contract_ortholog_qc_hash_matches_paper4_qc",
        "paper4_master_source_contains_all_prespecified_outcome_model_input_tokens",
        "master_has_dfi_derived_header_fields",
        "master_has_os_derived_header_fields",
        "master_has_model_selection_or_association_header_fields",
        "master_is_restricted_relative_to_full_dog2_transcriptome",
    ]
    false_checks = [key for key in required_true_checks if checks.get(key) is not True]
    if false_checks:
        raise RuntimeError(
            "Confirmed preflight no longer satisfies required lineage checks: "
            + ", ".join(false_checks)
        )

    # ----------------------------------------------------------------------------------
    # Confirm old 02f0 remains a historically valid execution.
    # ----------------------------------------------------------------------------------
    if clean(old_contract.get("status")) != "PASS":
        raise RuntimeError("Existing 02f0 contract status is not PASS.")
    if clean(old_summary.get("status")) != "PASS":
        raise RuntimeError("Existing 02f0 summary status is not PASS.")

    expected_old = audit.get("existing_02f0") or {}
    if sha256_file(OLD_CONTRACT) != clean(expected_old.get("contract_sha256")):
        raise RuntimeError("Existing 02f0 contract hash differs from the v3 preflight lock.")
    if sha256_file(OLD_SUMMARY) != clean(expected_old.get("summary_sha256")):
        raise RuntimeError("Existing 02f0 summary hash differs from the v3 preflight lock.")
    if sha256_file(OLD_STRICT_UNIVERSE) != clean(expected_old.get("strict_universe_sha256")):
        raise RuntimeError("Existing 02f0 strict universe hash differs from the v3 preflight lock.")

    old_design = old_contract.get("transport_design") or {}
    old_primary_model = clean(old_design.get("primary_model_family"))
    if "residual" not in old_primary_model.lower() or "cox" not in old_primary_model.lower():
        raise RuntimeError(
            "Existing 02f0 primary model family is not the residual-Cox family expected by this amendment."
        )

    old_hashes = {
        "02f0_script": sha256_file(OLD_SCRIPT),
        "02f0_contract": sha256_file(OLD_CONTRACT),
        "02f0_summary": sha256_file(OLD_SUMMARY),
        "02f0_strict_ortholog_universe": sha256_file(OLD_STRICT_UNIVERSE),
        "02f0_transport_feature_sets": sha256_file(OLD_FEATURE_SETS),
        "02f0_feature_set_summary": sha256_file(OLD_FEATURE_SUMMARY),
    }

    # ----------------------------------------------------------------------------------
    # Confirm Expression Arm/Confounding Gate remains valid and distinct.
    # ----------------------------------------------------------------------------------
    if clean(gate_summary.get("status")) != "PASS":
        raise RuntimeError("02e9 summary status is not PASS.")
    if clean(gate_results.get("status")) != "PASS":
        raise RuntimeError("02e9 gate results status is not PASS.")
    if clean(gate_summary.get("final_gate")) != EXPECTED_GATE:
        raise RuntimeError(
            f"02e9 final gate is not GREEN: {gate_summary.get('final_gate')!r}"
        )

    gate_safety = gate_results.get("safety") or {}
    for key in (
        "outcome_response_followup_values_read",
        "treatment_administration_values_read",
        "postbaseline_sample_annotations_read",
        "paper4_clinical_file_read",
    ):
        require_bool(gate_safety, key, False)

    gate_hashes = {
        "02e9_summary": sha256_file(GATE_SUMMARY),
        "02e9_gate_zero_results": sha256_file(GATE_RESULTS),
    }

    # ----------------------------------------------------------------------------------
    # Freeze the scientific amendment.
    # ----------------------------------------------------------------------------------
    supersession_rows = [
        {
            "component": "02f0_execution_record",
            "status_after_amendment": "RETAINED_HISTORICAL_PASS",
            "reason": (
                "02f0 executed its written safety contract correctly and read no clinical/outcome "
                "values; its artifacts/hashes are preserved unchanged."
            ),
        },
        {
            "component": "02f0_gate_zero_precondition",
            "status_after_amendment": "RETAINED",
            "reason": (
                "02e9 remains the valid Expression Arm/Confounding Gate and is GREEN."
            ),
        },
        {
            "component": "02f0_upstream_dataset_identities_and_hashes",
            "status_after_amendment": "RETAINED",
            "reason": (
                "The defect concerns feature-universe provenance/method role, not identity "
                "of the locked expression assets."
            ),
        },
        {
            "component": "02f0_strict_ortholog_universe_3391",
            "status_after_amendment": "SUPERSEDED_PRE_OUTCOME",
            "reason": (
                "Confirmed exact inheritance from the 5,013-gene Paper-4 outcome-aware "
                "candidate/evidence row universe."
            ),
        },
        {
            "component": "02f0_primary_DOG2_TARGET_feature_set_3386",
            "status_after_amendment": "SUPERSEDED_PRE_OUTCOME",
            "reason": (
                "Derived from the superseded candidate-restricted 3,391-gene strict universe."
            ),
        },
        {
            "component": "02f0_secondary_and_stress_feature_sets",
            "status_after_amendment": "SUPERSEDED_PRE_OUTCOME",
            "reason": (
                "All were derived from the same inherited candidate-restricted strict universe."
            ),
        },
        {
            "component": "02f0_primary_model_family_regularized_residual_cox",
            "status_after_amendment": "SUPERSEDED_AS_PRIMARY_RETAINED_AS_COMPARATOR",
            "reason": (
                "Residual/Trans-Cox-style adaptation is retained as an important classical "
                "transfer comparator but is not the Paper-6 methodological novelty."
            ),
        },
        {
            "component": "Paper4_frozen_zero_shot_comparator",
            "status_after_amendment": "RETAINED_EXACTLY_AS_FROZEN_IN_PAPER4",
            "reason": (
                "Its historical feature/program definitions, weights, and scoring constitute "
                "the comparator itself and must not be remapped, refit, or reweighted into "
                "the new Paper-6 universe."
            ),
        },
    ]

    policy_rows = [
        {
            "policy_id": "P01",
            "topic": "gate_terminology",
            "rule": (
                "02e9 is named Expression Arm/Confounding Gate; the pending canine-outcome "
                "gate is Source Prognostic Gate; the future confirmatory human evaluation is Human Transfer Gate."
            ),
        },
        {
            "policy_id": "P02",
            "topic": "human_outcome_firewall",
            "rule": (
                "Paper-6 method/architecture development code must not access TARGET, GSE21257, "
                "GSE39055, or sacrificial-target outcomes before each role-specific protocol is frozen. "
                "This is outcome-isolated Paper-6 development, not a claim of author-level prospective blinding, "
                "because human outcomes were previously encountered in Paper 4."
            ),
        },
        {
            "policy_id": "P03",
            "topic": "canine_outcome_firewall",
            "rule": (
                "DOG2 outcome values remain closed at 02f0a and may be opened only after 03a "
                "freezes the Source Prognostic Gate endpoint, cohort, metrics, preprocessing, "
                "resampling, models, uncertainty, and PASS/WARN/FAIL rules."
            ),
        },
        {
            "policy_id": "P04",
            "topic": "source_endpoint_hierarchy",
            "rule": (
                "DOG2 OS is the primary Source Prognostic Gate endpoint because the primary "
                "cross-species transfer setting is OS->OS. DOG2 DFI is prespecified secondary."
            ),
        },
        {
            "policy_id": "P05",
            "topic": "source_endpoint_discordance",
            "rule": (
                "OS PASS -> continue primary OS->OS programme. OS WARN with DFI PASS -> OS remains "
                "primary and a weak-source flag is carried forward. OS FAIL with DFI PASS -> no post-hoc "
                "switch of Paper-6 primary endpoint; OS->OS empirical premise fails and any DFI-led "
                "transfer requires a separate pre-human amendment. OS FAIL with DFI FAIL -> empirical "
                "source-transfer branch stops and methods/simulation fallback remains."
            ),
        },
        {
            "policy_id": "P06",
            "topic": "new_primary_feature_universe",
            "rule": (
                "All Paper-6 primary models must use a new transcriptome-wide outcome-blind bridge "
                "starting from the full 21,016 DOG2 expression feature header. Paper-4 candidate/evidence "
                "tables are forbidden for Paper-6 primary row selection."
            ),
        },
        {
            "policy_id": "P07",
            "topic": "ortholog_mapping_provenance",
            "rule": (
                "Only outcome-blind orthology/annotation information may define the new bridge. "
                "A versioned fresh Ensembl/Compara or BioMart snapshot with retrieval date/release/assembly "
                "and SHA256 is required for the final primary mapping. The historical Paper-4 BioMart "
                "snapshot is retained as reference/sensitivity, not as an undocumented sole primary source."
            ),
        },
        {
            "policy_id": "P08",
            "topic": "mapping_ambiguity",
            "rule": (
                "Canine feature-symbol collisions, many-to-one or many-to-many orthology, missing symbols, "
                "and duplicate human mappings must be handled by deterministic outcome-blind rules frozen "
                "before outcome modelling; no gene version/duplicate may be chosen by prognostic performance."
            ),
        },
        {
            "policy_id": "P09",
            "topic": "Paper4_comparator_universe",
            "rule": (
                "The frozen Paper-4 comparator is evaluated exactly under its original Paper-4 "
                "feature/program/weight/scoring definitions. It is compared at the prediction/performance "
                "level and is never refit/remapped onto the new Paper-6 universe."
            ),
        },
        {
            "policy_id": "P10",
            "topic": "residual_cox_role",
            "rule": (
                "Regularized residual Cox / Trans-Cox-style transfer is a required classical comparator, "
                "not the proposed primary methodological contribution."
            ),
        },
        {
            "policy_id": "P11",
            "topic": "architecture_selection",
            "rule": (
                "Before comparing candidate Paper-6 architectures, 02i must freeze a closed candidate list, "
                "selection metrics, tie rule favoring lower complexity, and a rule for disagreement between "
                "simulation and canine empirical evidence. Open-ended winner-picking on DOG2 or human outcomes "
                "is forbidden."
            ),
        },
        {
            "policy_id": "P12",
            "topic": "simulation_role",
            "rule": (
                "Simulation/semi-synthetic experiments are the primary environment for selecting the "
                "negative-transfer protection mechanism and assessing known-truth borrowing recovery; "
                "DOG2 is primarily a source-viability and empirical-stability gate rather than an unlimited "
                "architecture leaderboard."
            ),
        },
        {
            "policy_id": "P13",
            "topic": "sacrificial_human_premise_test",
            "rule": (
                "Before finalizing the flagship architecture, a separate public human osteosarcoma cohort "
                "not reserved as TARGET primary, GSE21257 external validation, or GSE39055 stress evaluation "
                "must be selected by metadata/endpoint/lineage criteria only as a sacrificial premise-test target. "
                "Its exact cohort and decision rule must be frozen before its outcomes are read in Paper 6."
            ),
        },
        {
            "policy_id": "P14",
            "topic": "premise_test_comparison",
            "rule": (
                "The premise test compares DOG2 source, pooled-human source excluding the target, and "
                "DOG2+pooled-human source using a prespecified low-capacity classical transfer benchmark. "
                "It tests the scientific value of canine source information and is not used to tune the proposed architecture."
            ),
        },
        {
            "policy_id": "P15",
            "topic": "module_coverage_gate",
            "rule": (
                "No module-resolved selective-borrowing architecture is finalized until 02h reports "
                "coverage of prespecified biological module libraries/compartments on the new 02g universe."
            ),
        },
        {
            "policy_id": "P16",
            "topic": "downstream_numbering",
            "rule": (
                "02g = transcriptome-wide outcome-blind ortholog bridge; 02h = module/compartment coverage audit; "
                "02i = revised transport/method-selection contract; 03a = Source Prognostic Gate protocol; "
                "03b = Source Prognostic Gate execution."
            ),
        },
    ]

    amendment = {
        "script_version": SCRIPT_VERSION,
        "status": "PASS",
        "created_utc": now_utc(),
        "scientific_stage": "PRE_OUTCOME_TRANSPORT_CONTRACT_AMENDMENT",
        "amendment_id": "02f0a",
        "amendment_class": "PRE_OUTCOME_PROVENANCE_AND_METHOD_ROLE_CORRECTION",
        "old_02f0_execution_status": "HISTORICALLY_VALID_PASS",
        "old_02f0_effective_status": "PARTIALLY_SUPERSEDED_PRE_OUTCOME",
        "reason_for_amendment": {
            "measured_provenance_defect": (
                "The Paper-6 02f0 feature universe inherited the exact row universe of a "
                "Paper-4 outcome-aware RNA candidate/evidence pipeline. 02f0 itself read only "
                "gene/mapping/QC columns and did not load outcome values; the defect is upstream "
                "row-universe selection inheritance."
            ),
            "method_role_correction": (
                "The 02f0 regularized residual-Cox family is retained as a classical comparator "
                "but removed from the role of proposed primary methodological contribution."
            ),
            "timing": (
                "Amendment frozen before Paper-6 source-outcome model fitting and before Paper-6 "
                "human-outcome model development."
            ),
        },
        "confirmed_measured_facts": {
            "DOG2_expression_features": int(measured["dog2_expression_features"]),
            "Paper4_master_candidate_evidence_genes": int(measured["paper4_master_unique_genes"]),
            "Paper4_master_with_orthologs_genes": int(measured["paper4_with_orthologs_unique_genes"]),
            "Paper4_ortholog_qc_genes": int(measured["paper4_ortholog_qc_unique_genes"]),
            "Paper4_BioMart_raw_rows": int(measured["biomart_rows"]),
            "Paper4_BioMart_unique_dog_symbols": int(measured["biomart_unique_dog_symbols"]),
            "Paper4_BioMart_dog_symbols_with_any_human_homolog": int(
                measured["biomart_dog_symbols_with_any_human_homolog"]
            ),
            "Paper4_BioMart_dog_symbols_with_one2one_human_ortholog": int(
                measured["biomart_dog_symbols_with_one2one_human_ortholog"]
            ),
            "Paper4_QC_strict_unique_canine_genes": int(
                measured["paper4_qc_strict_unique_canine_genes"]
            ),
            "Paper6_02f0_strict_unique_canine_genes": int(
                measured["paper6_02f0_strict_unique_canine_genes"]
            ),
            "Paper6_02f0_primary_DOG2_TARGET_genes": int(
                measured["paper6_02f0_primary_unique_canine_genes"]
            ),
            "master_DFI_derived_header_fields": dfi_header_count,
            "master_OS_derived_header_fields": os_header_count,
            "master_model_or_selection_header_fields": model_header_count,
            "master_all_outcome_aware_header_markers": all_outcome_header_count,
            "master_to_with_orthologs_exact_gene_set_equality": True,
            "master_to_ortholog_qc_exact_gene_set_equality": True,
            "Paper4_QC_strict_to_Paper6_02f0_strict_exact_gene_set_equality": True,
            "locked_ortholog_qc_path_identity": True,
            "locked_ortholog_qc_SHA256_identity": True,
        },
        "preflight_lock": {
            "verdict": EXPECTED_PREFLIGHT_VERDICT,
            "scientific_status": EXPECTED_PREFLIGHT_SCIENTIFIC_STATUS,
            "artifact_hashes": preflight_hashes,
        },
        "old_02f0_lock": {
            "old_primary_model_family": old_primary_model,
            "artifact_hashes": old_hashes,
        },
        "expression_arm_confounding_gate": {
            "terminology": "Expression Arm/Confounding Gate",
            "status": EXPECTED_GATE,
            "retained_as_valid": True,
            "artifact_hashes": gate_hashes,
            "note": (
                "This amendment does not reinterpret or rerun 02e9. "
                "The later Source Prognostic Gate is a distinct outcome-aware gate."
            ),
        },
        "supersession_scope": supersession_rows,
        "frozen_downstream_policies": policy_rows,
        "outcome_access_at_amendment": {
            "DOG2_source_outcomes_read_by_02f0a": False,
            "human_target_outcomes_read_by_02f0a": False,
            "clinical_values_read_by_02f0a": False,
            "expression_values_read_by_02f0a": False,
            "author_level_human_outcome_blinding_claim_allowed": False,
            "Paper6_outcome_isolated_method_development_claim_allowed": True,
        },
        "required_next_execution_order": [
            "02g_build_transcriptomewide_outcome_blind_ortholog_bridge.py",
            "02h_audit_module_coverage.py",
            "audit_and_freeze_sacrificial_human_premise_target_before_outcome_access",
            "02i_freeze_revised_transport_contract.py",
            "03a_freeze_dog2_source_prognostic_gate.py",
            "03b_run_dog2_source_prognostic_gate.py",
        ],
        "forbidden_until_revised_contract": [
            "Use 02f0 3,391-gene strict universe as the Paper-6 primary feature universe",
            "Use 02f0 3,386-gene DOG2->TARGET set as the Paper-6 primary modelling set",
            "Treat regularized residual Cox as the proposed Paper-6 methodological novelty",
            "Open TARGET/GSE21257/GSE39055 outcomes for architecture selection",
            "Switch primary source endpoint from OS to DFI because of observed source-gate performance",
            "Refit/remap/reweight the frozen Paper-4 comparator onto the new Paper-6 universe",
        ],
    }

    # Write human-readable supersession/policy tables first.
    write_tsv(
        SUPERSESSION_TSV,
        supersession_rows,
        ["component", "status_after_amendment", "reason"],
    )
    write_tsv(
        POLICY_TSV,
        policy_rows,
        ["policy_id", "topic", "rule"],
    )

    readme = f"""Paper 6 02f0a PRE-OUTCOME TRANSPORT AMENDMENT
==================================================

Script version:
{SCRIPT_VERSION}

Status:
PASS

Why this amendment exists
-------------------------
The confirmed 02f0_v3 provenance preflight established that the feature universe
used by 02f0 was inherited from the Paper-4 RNA master candidate/evidence row
universe.

Measured local lineage:
  DOG2 expression header                          21,016 features
  Paper4 master candidate/evidence                5,013 genes
  Paper4 master + ortholog annotation             5,013 genes
  Paper4 ortholog QC                              5,013 genes
  broader outcome-blind BioMart snapshot          34,159 rows
  BioMart unique dog symbols                      16,955
  BioMart dog symbols with any human homolog      16,063
  BioMart dog symbols with one2one human ortholog 15,318
  Paper4 candidate-QC strict genes                3,391
  Paper6 02f0 strict genes                        3,391
  Paper6 02f0 primary DOG2->TARGET genes          3,386

The preflight also confirmed:
  - exact master -> with-orthologs gene-set equality;
  - exact master -> ortholog-QC gene-set equality;
  - exact Paper4 strict-QC -> Paper6 02f0 strict-set equality;
  - path and SHA256 identity for the locked ortholog_qc asset;
  - 38 DFI-derived master header fields;
  - 38 OS-derived master header fields;
  - 67 model/selection header fields;
  - 80 total outcome-aware header markers.

Scientific interpretation
-------------------------
02f0 itself did NOT read clinical/outcome values. Its technical safety contract
was respected. The defect is upstream row-universe selection inheritance.

Therefore:
  - 02f0 remains a preserved historical PASS;
  - its 3,391 strict universe and all derived feature sets are SUPERSEDED_PRE_OUTCOME
    for Paper-6 primary modelling;
  - its residual-Cox primary-method role is superseded;
  - residual/Trans-Cox-style transfer remains a required comparator;
  - the frozen Paper-4 comparator remains exactly as frozen in Paper 4 and must not
    be remapped/refit/reweighted into the new Paper-6 universe.

Next sequence
-------------
02g  transcriptome-wide outcome-blind ortholog bridge
02h  biological-module / compartment coverage audit
      + metadata-only selection/freezing of a sacrificial human premise-test target
02i  revised Paper-6 transport/method-selection contract
03a  DOG2 Source Prognostic Gate protocol
03b  DOG2 Source Prognostic Gate execution

Outcome policy
--------------
DOG2 outcomes remain CLOSED until 03a passes.
Human outcomes remain CLOSED for Paper-6 architecture development.
TARGET, GSE21257 and GSE39055 are not sacrificial architecture-development targets.

The project may describe this as outcome-isolated Paper-6 method development.
It must NOT claim author-level prospective human-outcome blinding because the
authors had prior outcome exposure through Paper 4.

Existing 02f0 files were not edited or deleted.
"""
    README_TXT.write_text(readme, encoding="utf-8", newline="\n")

    # Hash the human-readable amendment components and then write the main JSON.
    amendment_component_hashes = {
        "supersession_scope_tsv": sha256_file(SUPERSESSION_TSV),
        "downstream_policy_tsv": sha256_file(POLICY_TSV),
        "amendment_README_txt": sha256_file(README_TXT),
    }
    amendment["amendment_component_hashes"] = amendment_component_hashes
    write_json(AMENDMENT_JSON, amendment)

    final_hashes = {
        "transport_contract_amendment_json": sha256_file(AMENDMENT_JSON),
        "supersession_scope_tsv": sha256_file(SUPERSESSION_TSV),
        "downstream_policy_tsv": sha256_file(POLICY_TSV),
        "amendment_README_txt": sha256_file(README_TXT),
    }

    summary = {
        "script_version": SCRIPT_VERSION,
        "status": "PASS",
        "created_utc": now_utc(),
        "run_started_utc": started,
        "run_finished_utc": now_utc(),
        "amendment_id": "02f0a",
        "old_02f0_execution_status": "HISTORICALLY_VALID_PASS",
        "old_02f0_effective_status": "PARTIALLY_SUPERSEDED_PRE_OUTCOME",
        "confirmed_preflight_verdict": EXPECTED_PREFLIGHT_VERDICT,
        "confirmed_measured_counts": amendment["confirmed_measured_facts"],
        "old_02f0_artifact_hashes": old_hashes,
        "preflight_artifact_hashes": preflight_hashes,
        "gate_zero_artifact_hashes": gate_hashes,
        "final_artifact_hashes": final_hashes,
        "outcome_response_followup_values_read": False,
        "clinical_values_read": False,
        "treatment_administration_values_read": False,
        "expression_values_read": False,
        "model_fitting": False,
        "existing_02f0_artifacts_modified": False,
        "next": "02g transcriptome-wide outcome-blind ortholog bridge",
    }
    write_json(SUMMARY_JSON, summary)

    print("Confirmed v3 provenance defect:")
    print(f"  DOG2 full expression features: {int(measured['dog2_expression_features']):,}")
    print(f"  Paper4 outcome-aware master universe: {int(measured['paper4_master_unique_genes']):,}")
    print(f"  Paper4 QC strict genes: {int(measured['paper4_qc_strict_unique_canine_genes']):,}")
    print(f"  Paper6 02f0 strict genes: {int(measured['paper6_02f0_strict_unique_canine_genes']):,}")
    print(f"  Paper6 02f0 primary DOG2->TARGET genes: {int(measured['paper6_02f0_primary_unique_canine_genes']):,}")
    print(f"  broader BioMart one2one dog symbols: {int(measured['biomart_dog_symbols_with_one2one_human_ortholog']):,}")
    print()
    print("Amendment effect:")
    print("  02f0 execution record: RETAINED_HISTORICAL_PASS")
    print("  02e9 Expression Arm/Confounding Gate: RETAINED_GREEN")
    print("  02f0 3,391-gene strict universe: SUPERSEDED_PRE_OUTCOME")
    print("  02f0 3,386-gene primary feature set: SUPERSEDED_PRE_OUTCOME")
    print("  residual Cox primary-method role: SUPERSEDED_AS_PRIMARY / RETAINED_AS_COMPARATOR")
    print("  Paper4 frozen comparator: RETAINED_EXACTLY_AS_FROZEN")
    print()
    print("Outcome firewall after amendment:")
    print("  DOG2 outcomes: CLOSED until 03a Source Prognostic Gate contract")
    print("  TARGET outcomes: CLOSED for Paper6 method development")
    print("  GSE21257 outcomes: CLOSED / external-validation reserve")
    print("  GSE39055 outcomes: CLOSED / stress-evaluation reserve")
    print("  author-level prospective human-outcome blinding claim: NOT ALLOWED")
    print("  outcome-isolated Paper6 method-development claim: ALLOWED")
    print()
    print("Next frozen sequence:")
    print("  02g -> transcriptome-wide outcome-blind ortholog bridge")
    print("  02h -> module/compartment coverage audit")
    print("  metadata-only sacrificial human premise-target audit/freeze")
    print("  02i -> revised transport/method-selection contract")
    print("  03a -> Source Prognostic Gate protocol")
    print("  03b -> Source Prognostic Gate execution")
    print()
    print("Artifacts:")
    for path in [AMENDMENT_JSON, SUPERSESSION_TSV, POLICY_TSV, README_TXT, SUMMARY_JSON]:
        print(f"  {path.relative_to(ROOT)}")
    print()
    print("Outcome/response/follow-up values read: NO")
    print("Clinical values read: NO")
    print("Treatment-administration values read: NO")
    print("Expression values read: NO")
    print("Model fitting: NO")
    print("Existing 02f0 artifacts modified/deleted: NO")
    print()
    print("=" * 120)
    print("02f0a pre-outcome transport amendment freeze: PASS")
    print("=" * 120)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 120, file=sys.stderr)
        print("02f0a pre-outcome transport amendment freeze: FAIL", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print("=" * 120, file=sys.stderr)
        raise
