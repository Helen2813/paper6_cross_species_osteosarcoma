#!/usr/bin/env python3
"""
Paper 6 - freeze revised pre-outcome model/benchmark/source-gate contract.

This is the FINAL broad pre-outcome design freeze before the DOG2 Source
Prognostic Gate.

It does not read clinical/outcome/expression values and does not fit a model.

Frozen here
-----------
1. Gate terminology.
2. DOG2 OS primary / DFI secondary endpoint hierarchy.
3. Source predictive PASS/WARN/FAIL materiality rules.
4. Random-CV vs treatment-arm->arm generalization rules.
5. OS/DFI discordance rule (no post-hoc endpoint switch).
6. Required classical survival-transfer benchmark ladder.
7. Closed AI candidate/ablation registry.
8. Architecture-selection anti-forking rule.
9. Sacrificial-human premise-test tri-state criterion.
10. Execution order after this freeze.

03a is allowed to translate this frozen design into exact computational details
(folds, regularization grids, seeds, bootstrap/permutation counts), but it may not
change the scientific thresholds, endpoint hierarchy, model families, or decision
logic frozen here.

No CLI arguments.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd


SCRIPT_VERSION = "02i-freeze-revised-model-benchmark-source-gate-contract-v1-no-cli"

ROOT = Path(__file__).resolve().parents[1]

AMEND_DIR = ROOT / "results" / "transport_contract" / "02f0a_amendment"
AMENDMENT = AMEND_DIR / "transport_contract_amendment.json"
AMENDMENT_SUMMARY = AMEND_DIR / "summary.json"

BRIDGE_DIR = ROOT / "results" / "ortholog_bridge" / "02g_v3"
BRIDGE_CONTRACT = BRIDGE_DIR / "ortholog_bridge_contract.json"
BRIDGE_SUMMARY = BRIDGE_DIR / "summary.json"

MODULE_DIR = ROOT / "results" / "module_coverage" / "02h"
MODULE_CONTRACT = MODULE_DIR / "module_coverage_contract.json"
MODULE_SUMMARY = MODULE_DIR / "summary.json"

OUT_DIR = ROOT / "results" / "revised_design" / "02i"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_GATE_RULES = OUT_DIR / "source_gate_rules.tsv"
ARM_GATE_RULES = OUT_DIR / "arm_generalization_rules.tsv"
BENCHMARK_REGISTRY = OUT_DIR / "classical_benchmark_registry.tsv"
AI_REGISTRY = OUT_DIR / "ai_candidate_registry.tsv"
PREMISE_RULES = OUT_DIR / "premise_test_rules.tsv"
CONTRACT_JSON = OUT_DIR / "revised_model_benchmark_source_gate_contract.json"
SUMMARY_JSON = OUT_DIR / "summary.json"


EXPECTED_BRIDGE_STATUS = (
    "PASS_TRANSCRIPTOMEWIDE_OUTCOME_BLIND_BRIDGE_READY_FOR_MODULE_COVERAGE"
)
EXPECTED_MODULE_STATUS = "PASS_MODULE_REPRESENTATION_COVERAGE_READY_FOR_02I"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def require_file(path: Path) -> Path:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Required frozen artifact missing: {path}")
    return path


def write_tsv(path: Path, rows: List[Dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def main() -> None:
    started = now_utc()

    print("=" * 120)
    print("Paper 6 - freeze revised model/benchmark/source-gate contract")
    print("=" * 120)
    print(f"Script version: {SCRIPT_VERSION}")
    print()
    print("Execution contract:")
    print("  Network access: NO")
    print("  DOG2 clinical/outcome values read: NO")
    print("  Human clinical/outcome values read: NO")
    print("  Expression values read: NO")
    print("  Model fitting/tuning: NO")
    print("  Target/source splits generated: NO")
    print("  This is the final broad pre-outcome design freeze: YES")
    print()

    for path in [
        AMENDMENT,
        AMENDMENT_SUMMARY,
        BRIDGE_CONTRACT,
        BRIDGE_SUMMARY,
        MODULE_CONTRACT,
        MODULE_SUMMARY,
    ]:
        require_file(path)

    amendment = read_json(AMENDMENT)
    amendment_summary = read_json(AMENDMENT_SUMMARY)
    bridge_contract = read_json(BRIDGE_CONTRACT)
    bridge_summary = read_json(BRIDGE_SUMMARY)
    module_contract = read_json(MODULE_CONTRACT)
    module_summary = read_json(MODULE_SUMMARY)

    if clean(amendment.get("status")) != "PASS":
        raise RuntimeError("02f0a amendment is not PASS.")
    if clean(amendment_summary.get("status")) != "PASS":
        raise RuntimeError("02f0a amendment summary is not PASS.")
    if clean(bridge_contract.get("status")) != "PASS":
        raise RuntimeError("02g contract is not PASS.")
    if clean(bridge_summary.get("status")) != "PASS":
        raise RuntimeError("02g summary is not PASS.")
    if clean(bridge_summary.get("scientific_status")) != EXPECTED_BRIDGE_STATUS:
        raise RuntimeError("02g scientific status differs from expected PASS state.")
    if clean(module_contract.get("status")) != "PASS":
        raise RuntimeError("02h contract is not PASS.")
    if clean(module_summary.get("status")) != "PASS":
        raise RuntimeError("02h summary is not PASS.")
    if clean(module_summary.get("scientific_status")) != EXPECTED_MODULE_STATUS:
        raise RuntimeError("02h scientific status differs from expected PASS state.")

    # Hash-chain verification.
    expected_bridge_contract_hash = clean(
        (bridge_summary.get("final_artifact_hashes") or {}).get(
            "ortholog_bridge_contract_json"
        )
    )
    if expected_bridge_contract_hash != sha256_file(BRIDGE_CONTRACT):
        raise RuntimeError("02g contract hash verification failed.")

    expected_module_contract_hash = clean(
        (module_summary.get("final_artifact_hashes") or {}).get(
            "module_coverage_contract_json"
        )
    )
    if expected_module_contract_hash != sha256_file(MODULE_CONTRACT):
        raise RuntimeError("02h contract hash verification failed.")

    # Freeze observed feature/module capacities, but NOT outcomes.
    bridge_counts = bridge_summary.get("measured_counts") or {}
    module_counts = {
        "Hallmark_primary_eligible": int(
            module_summary["Hallmark_primary_eligible"]
        ),
        "Hallmark_common_all_four_eligible": int(
            module_summary["Hallmark_common_all_four_eligible"]
        ),
        "Reactome_primary_eligible": int(
            module_summary["Reactome_primary_eligible"]
        ),
        "Reactome_common_all_four_eligible": int(
            module_summary["Reactome_common_all_four_eligible"]
        ),
    }

    if int(bridge_counts["primary_dog2_to_target_os"]) != 11815:
        raise RuntimeError("Unexpected 02g primary feature count.")
    if int(bridge_counts["common_all_four"]) != 8652:
        raise RuntimeError("Unexpected 02g common-all-four feature count.")
    if module_counts["Hallmark_primary_eligible"] != 50:
        raise RuntimeError("Unexpected Hallmark primary coverage.")
    if module_counts["Hallmark_common_all_four_eligible"] != 49:
        raise RuntimeError("Unexpected Hallmark common-all-four coverage.")

    # ------------------------------------------------------------------
    # 1. Source Prognostic Gate.
    # ------------------------------------------------------------------
    source_gate_rows = [
        {
            "axis": "PRIMARY_ENDPOINT",
            "rule": "DOG2 overall survival (OS)",
            "threshold_or_action": "Primary source endpoint; no post-hoc replacement by DFI.",
        },
        {
            "axis": "SECONDARY_ENDPOINT",
            "rule": "DOG2 disease-free interval (DFI)",
            "threshold_or_action": "Secondary/sensitivity endpoint only unless a new pre-human amendment is frozen.",
        },
        {
            "axis": "PRIMARY_GATE_MODEL",
            "rule": "Hallmark-50 module representation + ridge-penalized Cox",
            "threshold_or_action": (
                "External outcome-blind Hallmark modules; fold-safe gene standardization/module scoring; "
                "regularization chosen only inside training data."
            ),
        },
        {
            "axis": "SUPPORTING_GATE_MODEL",
            "rule": "Gene-level elastic-net Cox on the 11,815-gene primary 02g bridge",
            "threshold_or_action": (
                "Prespecified sensitivity/supporting model; may create representation-discordance WARN "
                "but may not silently replace the primary gate model."
            ),
        },
        {
            "axis": "PRIMARY_METRIC",
            "rule": "Uno concordance (Uno C)",
            "threshold_or_action": "Chance reference = 0.50.",
        },
        {
            "axis": "SOURCE_PASS",
            "rule": "Primary Hallmark ridge-Cox OOS Uno C >= 0.60 AND lower 95% patient-bootstrap CI > 0.50",
            "threshold_or_action": "Source predictive signal passes.",
        },
        {
            "axis": "SOURCE_WARN",
            "rule": (
                "Primary Hallmark ridge-Cox OOS Uno C >= 0.55 but PASS not met; "
                "OR primary < 0.55 while supporting gene-level model meets the PASS criterion."
            ),
            "threshold_or_action": (
                "Continue with weak-source or representation-discordance flag; "
                "claims shift toward mechanisms/limits rather than superiority."
            ),
        },
        {
            "axis": "SOURCE_FAIL",
            "rule": (
                "Primary Hallmark ridge-Cox OOS Uno C < 0.55 AND supporting gene-level model "
                "does not meet the PASS criterion."
            ),
            "threshold_or_action": (
                "No strong empirical source premise; unrestricted transfer programme contracts "
                "to methods/negative-transfer/phase-diagram fallback."
            ),
        },
        {
            "axis": "PERMUTATION_ROLE",
            "rule": "Outcome-label permutation inference is required supporting inference",
            "threshold_or_action": (
                "Permutation significance alone cannot upgrade a SOURCE_WARN/FAIL based on material Uno-C thresholds."
            ),
        },
    ]
    write_tsv(SOURCE_GATE_RULES, source_gate_rows)

    # ------------------------------------------------------------------
    # 2. Treatment-context arm->arm generalization gate.
    # ------------------------------------------------------------------
    arm_gate_rows = [
        {
            "status": "ARM_PASS",
            "rule": (
                "Mean of COTC021->COTC022 and COTC022->COTC021 Uno C >= 0.55; "
                "neither directional point estimate < 0.50; "
                "mean arm->arm Uno C is no more than 0.05 below random-CV primary Uno C."
            ),
            "interpretation": "Source signal is reasonably stable to randomized treatment-context change.",
        },
        {
            "status": "ARM_WARN",
            "rule": (
                "ARM_PASS not met, but mean directional Uno C >= 0.52 and no directional "
                "upper 95% bootstrap CI is < 0.50."
            ),
            "interpretation": (
                "Source signal exists but treatment-context generalization is weak/context-sensitive; "
                "carry explicit caution into cross-species transport."
            ),
        },
        {
            "status": "ARM_FAIL",
            "rule": (
                "Mean directional Uno C < 0.52 OR at least one directional upper 95% bootstrap CI < 0.50."
            ),
            "interpretation": (
                "Cross-treatment-context prognostic structure is not sufficiently transport-stable; "
                "unrestricted dog->human prognostic transfer is RED even if random CV is good."
            ),
        },
        {
            "status": "ARM_PRIMARY_MODEL",
            "rule": "Same Hallmark-50 ridge-Cox source model family used for the primary Source Prognostic Gate",
            "interpretation": "No model-family switching between random CV and arm->arm evaluation.",
        },
    ]
    write_tsv(ARM_GATE_RULES, arm_gate_rows)

    # ------------------------------------------------------------------
    # 3. Endpoint discordance: frozen BEFORE DOG2 outcomes.
    # ------------------------------------------------------------------
    endpoint_discordance = {
        "OS_PASS": "Continue primary OS->OS transfer programme.",
        "OS_WARN_DFI_PASS": (
            "OS remains primary. Carry weak-source flag. DFI may support secondary/mechanistic analyses only."
        ),
        "OS_FAIL_DFI_PASS": (
            "Do NOT switch Paper-6 primary endpoint post hoc. Primary OS->OS empirical premise fails. "
            "A DFI-led transfer programme requires a separate amendment frozen before human target outcomes."
        ),
        "OS_FAIL_DFI_FAIL": (
            "Empirical source-transfer branch stops; methods/simulation/negative empirical fallback remains."
        ),
    }

    # ------------------------------------------------------------------
    # 4. Classical survival-transfer benchmark ladder.
    # ------------------------------------------------------------------
    benchmark_rows = [
        {
            "benchmark_id": "B0",
            "family": "TARGET_ONLY",
            "model": "Ridge Cox",
            "role": "Low-capacity no-transfer baseline",
            "required": True,
        },
        {
            "benchmark_id": "B1",
            "family": "TARGET_ONLY",
            "model": "Elastic-net Cox",
            "role": "Gene-level no-transfer baseline",
            "required": True,
        },
        {
            "benchmark_id": "B2",
            "family": "ZERO_SHOT",
            "model": "Frozen DOG2 penalized-Cox relative-risk score",
            "role": "Direct source transport baseline",
            "required": True,
        },
        {
            "benchmark_id": "B3",
            "family": "SIMPLE_ALIGNMENT",
            "model": "Fold-safe within-domain standardization / CORAL + Cox",
            "role": "Classical representation-alignment baseline",
            "required": True,
        },
        {
            "benchmark_id": "B4",
            "family": "CLASSICAL_TRANSFER",
            "model": "Regularized residual / Trans-Cox-style adaptation",
            "role": "Primary classical transfer comparator; NOT methodological novelty",
            "required": True,
        },
        {
            "benchmark_id": "B5",
            "family": "OPTIONAL_CLASSICAL_SENSITIVITY",
            "model": "Hierarchical / multi-task Cox",
            "role": "Secondary classical sensitivity if implementation is stable",
            "required": False,
        },
    ]
    write_tsv(BENCHMARK_REGISTRY, benchmark_rows)

    # ------------------------------------------------------------------
    # 5. Closed AI candidate list.
    # ------------------------------------------------------------------
    ai_rows = [
        {
            "candidate_id": "A0",
            "model": "Small Hallmark-module survival network from scratch",
            "role": "Neural no-transfer baseline",
            "eligible_for_primary_selection": False,
        },
        {
            "candidate_id": "A1",
            "model": "DOG2-pretrained module encoder + target head only",
            "role": "Standard frozen-representation transfer baseline",
            "eligible_for_primary_selection": False,
        },
        {
            "candidate_id": "A2",
            "model": "DOG2-pretrained module encoder + low-rank residual target adapter",
            "role": "Parameter-efficient transfer candidate",
            "eligible_for_primary_selection": True,
        },
        {
            "candidate_id": "A3",
            "model": (
                "Evolution-conditioned module-resolved selective borrowing with target-overridable gate"
            ),
            "role": "Full proposed-method candidate",
            "eligible_for_primary_selection": True,
        },
        {
            "candidate_id": "A4",
            "model": "Unrestricted/full neural fine-tuning",
            "role": "Negative-transfer / instability baseline",
            "eligible_for_primary_selection": False,
        },
    ]
    write_tsv(AI_REGISTRY, ai_rows)

    architecture_selection_rule = {
        "candidate_list_is_closed": True,
        "new_candidate_after_02i": (
            "Forbidden unless a new observed blocking defect requires a formal pre-human amendment."
        ),
        "primary_development_environment": (
            "Known-truth simulation/semi-synthetic experiments; DOG2 is not an open-ended architecture leaderboard."
        ),
        "primary_selection_priorities_in_order": [
            "negative-transfer protection / failure avoidance",
            "recovery of known true module borrowing under simulation",
            "calibration/uncertainty behavior",
            "predictive performance",
            "lower complexity when practically tied",
        ],
        "tie_rule": (
            "If candidate performance differences are within Monte-Carlo/uncertainty tolerance "
            "defined in the later simulation protocol, select the lower-complexity model."
        ),
        "simulation_vs_DOG2_disagreement": (
            "Do not choose whichever looks better. Retain simulation-selected architecture; "
            "report DOG2 disagreement as empirical limitation unless a new pre-human amendment "
            "is scientifically required."
        ),
    }

    # ------------------------------------------------------------------
    # 6. Sacrificial-human premise test.
    # ------------------------------------------------------------------
    premise_rows = [
        {
            "contrast": "PRIMARY",
            "comparison": "DOG2 + pooled-human source versus pooled-human source alone",
            "criterion": (
                "SUPPORTS_CANINE_ADDED_VALUE if paired delta Uno C >= +0.02, "
                "paired-bootstrap P(delta Uno C > 0) >= 0.90, "
                "and delta IBS <= +0.01 (positive delta IBS = worse)."
            ),
        },
        {
            "contrast": "PRIMARY",
            "comparison": "DOG2 + pooled-human source versus pooled-human source alone",
            "criterion": (
                "ARGUES_AGAINST_CANINE_ADDED_VALUE if paired delta Uno C <= -0.02 "
                "and paired-bootstrap P(delta Uno C < 0) >= 0.90."
            ),
        },
        {
            "contrast": "PRIMARY",
            "comparison": "DOG2 + pooled-human source versus pooled-human source alone",
            "criterion": "Otherwise INCONCLUSIVE_NEUTRAL.",
        },
        {
            "contrast": "SECONDARY",
            "comparison": "Event-matched DOG2 source versus event-matched human source",
            "criterion": "Information-efficiency analysis only; not the primary premise decision.",
        },
        {
            "contrast": "TARGET_SELECTION",
            "comparison": "Sacrificial human premise-test cohort",
            "criterion": (
                "Must be selected and frozen from metadata/endpoint/sample-lineage criteria only; "
                "not TARGET, GSE21257, or GSE39055; no Paper-6 outcome access before target role is frozen."
            ),
        },
        {
            "contrast": "MODEL",
            "comparison": "Premise test model family",
            "criterion": (
                "Low-capacity prespecified classical survival-transfer models only; "
                "premise-test outcomes cannot tune the proposed neural architecture."
            ),
        },
    ]
    write_tsv(PREMISE_RULES, premise_rows)

    gate_matrix = {
        "SOURCE_PASS__ARM_PASS": "SOURCE_GREEN",
        "SOURCE_PASS__ARM_WARN": "SOURCE_AMBER_CONTEXT_SENSITIVE",
        "SOURCE_PASS__ARM_FAIL": "SOURCE_RED_FOR_UNRESTRICTED_CROSS_CONTEXT_TRANSFER",
        "SOURCE_WARN__ARM_PASS": "SOURCE_AMBER_WEAK_BUT_CONTEXT_STABLE",
        "SOURCE_WARN__ARM_WARN": "SOURCE_AMBER_WEAK_AND_CONTEXT_SENSITIVE",
        "SOURCE_WARN__ARM_FAIL": "SOURCE_RED_FOR_UNRESTRICTED_CROSS_CONTEXT_TRANSFER",
        "SOURCE_FAIL__ARM_PASS": "SOURCE_RED_NO_MATERIAL_SOURCE_SIGNAL",
        "SOURCE_FAIL__ARM_WARN": "SOURCE_RED_NO_MATERIAL_SOURCE_SIGNAL",
        "SOURCE_FAIL__ARM_FAIL": "SOURCE_RED_NO_MATERIAL_SOURCE_SIGNAL",
    }

    contract = {
        "script_version": SCRIPT_VERSION,
        "status": "PASS",
        "created_utc": now_utc(),
        "scientific_stage": "FINAL_BROAD_PREOUTCOME_DESIGN_FREEZE",
        "gate_terminology": {
            "02e9": "Expression Arm/Confounding Gate",
            "03b": "Source Prognostic Gate",
            "future_human": "Human Transfer Gate",
        },
        "frozen_data_capacity": {
            "primary_DOG2_TARGET_genes": int(
                bridge_counts["primary_dog2_to_target_os"]
            ),
            "common_all_four_genes": int(bridge_counts["common_all_four"]),
            **module_counts,
        },
        "source_endpoint_hierarchy": {
            "primary": "DOG2 OS",
            "secondary": "DOG2 DFI",
            "discordance_rules": endpoint_discordance,
            "posthoc_primary_endpoint_switch": "FORBIDDEN",
        },
        "source_gate": {
            "primary_representation": "MSigDB Hallmark-50 module representation",
            "primary_model_family": "ridge-penalized Cox",
            "supporting_model_family": (
                "gene-level elastic-net Cox on the 11,815-gene 02g primary bridge"
            ),
            "primary_metric": "Uno C",
            "source_pass": {
                "uno_c_min": 0.60,
                "lower_95pct_patient_bootstrap_ci_strictly_above": 0.50,
            },
            "source_warn_material_floor": 0.55,
            "source_fail_rule": (
                "Primary Uno C <0.55 AND supporting gene-level model does not meet source PASS."
            ),
            "permutation_inference_role": "supporting_not_gate_upgrading",
        },
        "arm_to_arm_generalization": {
            "directions": ["COTC021->COTC022", "COTC022->COTC021"],
            "arm_pass": {
                "mean_directional_uno_c_min": 0.55,
                "minimum_directional_point_estimate": 0.50,
                "max_drop_from_random_cv_mean_uno_c": 0.05,
            },
            "arm_warn": {
                "mean_directional_uno_c_min": 0.52,
                "no_directional_upper_95pct_bootstrap_ci_below": 0.50,
            },
            "arm_fail": (
                "Mean directional Uno C <0.52 OR any directional upper 95% bootstrap CI <0.50."
            ),
            "model_switch_between_random_cv_and_arm_to_arm": "FORBIDDEN",
        },
        "combined_source_gate_matrix": gate_matrix,
        "classical_benchmark_registry": benchmark_rows,
        "ai_candidate_registry": ai_rows,
        "architecture_selection_rule": architecture_selection_rule,
        "premise_test": {
            "primary_contrast": "DOG2_PLUS_HUMAN_POOL_MINUS_HUMAN_POOL_ONLY",
            "supports_canine_added_value": {
                "paired_delta_uno_c_min": 0.02,
                "paired_bootstrap_probability_delta_c_gt_zero_min": 0.90,
                "maximum_allowed_delta_IBS": 0.01,
            },
            "argues_against_canine_added_value": {
                "paired_delta_uno_c_max": -0.02,
                "paired_bootstrap_probability_delta_c_lt_zero_min": 0.90,
            },
            "otherwise": "INCONCLUSIVE_NEUTRAL",
            "target_role_freeze_required_before_outcome_access": True,
            "premise_outcomes_may_tune_proposed_AI": False,
        },
        "implementation_scope_rule_after_02i": {
            "new_scientific_pre_result_stage": (
                "Allowed only for an already observed blocking defect; hypothetical additional "
                "controls do not justify another broad pre-result stage."
            ),
            "implementation_bugfix_v2_v3": (
                "Allowed when scientific contract, inputs, thresholds, and intended output do not change."
            ),
            "scope_expansion_during_bugfix": "FORBIDDEN",
            "time_budget_guideline": (
                "A new blocking pre-result scientific issue should not expand into an open-ended branch; "
                "document limitation rather than broadening scope."
            ),
        },
        "03a_allowed_scope": {
            "allowed": [
                "resolve exact OS/DFI column names without changing endpoint meaning",
                "freeze seeds",
                "freeze exact outer/inner resampling counts",
                "freeze regularization grids for already-approved model families",
                "freeze bootstrap/permutation counts",
                "implement fold-safe preprocessing details",
            ],
            "forbidden": [
                "change PASS/WARN/FAIL thresholds",
                "change OS/DFI hierarchy",
                "add model families",
                "change combined source-gate matrix",
                "change premise-test materiality criteria",
                "open human outcomes",
            ],
        },
        "required_execution_order": [
            "03a_freeze_dog2_source_prognostic_gate.py",
            "03b_run_dog2_source_prognostic_gate.py",
            "classical_survival_transfer_benchmark_stage",
            "sacrificial_human_premise_test",
            "simulation_phase_diagram_and_closed_AI_candidate_selection",
            "freeze_selected_method_before_primary_human_outcome_evaluation",
            "TARGET_primary_human_transfer_gate",
            "GSE21257_external_replication",
            "GSE39055_cross_platform_stress_evaluation",
        ],
        "upstream_hashes": {
            "02f0a_amendment": sha256_file(AMENDMENT),
            "02g_contract": sha256_file(BRIDGE_CONTRACT),
            "02g_summary": sha256_file(BRIDGE_SUMMARY),
            "02h_contract": sha256_file(MODULE_CONTRACT),
            "02h_summary": sha256_file(MODULE_SUMMARY),
        },
        "safety": {
            "DOG2_outcome_values_read": False,
            "human_outcome_values_read": False,
            "clinical_values_read": False,
            "expression_values_read": False,
            "model_fitting": False,
        },
    }
    write_json(CONTRACT_JSON, contract)

    final_hashes = {
        "source_gate_rules_tsv": sha256_file(SOURCE_GATE_RULES),
        "arm_generalization_rules_tsv": sha256_file(ARM_GATE_RULES),
        "classical_benchmark_registry_tsv": sha256_file(BENCHMARK_REGISTRY),
        "ai_candidate_registry_tsv": sha256_file(AI_REGISTRY),
        "premise_test_rules_tsv": sha256_file(PREMISE_RULES),
        "revised_contract_json": sha256_file(CONTRACT_JSON),
    }

    summary = {
        "script_version": SCRIPT_VERSION,
        "status": "PASS",
        "scientific_status": "PASS_FINAL_BROAD_PREOUTCOME_DESIGN_FREEZE",
        "run_started_utc": started,
        "run_finished_utc": now_utc(),
        "primary_source_endpoint": "DOG2 OS",
        "secondary_source_endpoint": "DOG2 DFI",
        "source_pass_uno_c": 0.60,
        "source_warn_floor_uno_c": 0.55,
        "arm_pass_mean_uno_c": 0.55,
        "arm_warn_mean_floor_uno_c": 0.52,
        "premise_material_delta_uno_c": 0.02,
        "required_classical_benchmarks": int(
            sum(bool(row["required"]) for row in benchmark_rows)
        ),
        "closed_AI_candidates": len(ai_rows),
        "final_artifact_hashes": final_hashes,
        "DOG2_outcome_values_read": False,
        "human_outcome_values_read": False,
        "clinical_values_read": False,
        "expression_values_read": False,
        "model_fitting": False,
        "next": "03a Source Prognostic Gate computational protocol",
    }
    write_json(SUMMARY_JSON, summary)

    print("-" * 120)
    print("Frozen Source Prognostic Gate")
    print("-" * 120)
    print("Primary endpoint: DOG2 OS")
    print("Secondary endpoint: DOG2 DFI")
    print("Primary gate model: Hallmark-50 ridge Cox")
    print("Supporting model: 11,815-gene elastic-net Cox")
    print("Primary metric: Uno C")
    print("SOURCE PASS: Uno C >= 0.60 AND lower 95% bootstrap CI > 0.50")
    print("SOURCE WARN floor: Uno C >= 0.55 without PASS")
    print("SOURCE FAIL: primary < 0.55 and supporting model does not PASS")
    print()

    print("Frozen treatment-context generalization:")
    print("ARM PASS: mean directional Uno C >= 0.55, neither direction <0.50,")
    print("          and drop from random-CV mean <=0.05")
    print("ARM WARN: mean directional Uno C >=0.52 without clear directional reversal")
    print("ARM FAIL: mean directional Uno C <0.52 OR directional upper 95% CI <0.50")
    print()

    print("Frozen endpoint-discordance rule:")
    print("OS FAIL + DFI PASS does NOT switch Paper6 primary endpoint.")
    print("A DFI-led programme would require a separate pre-human amendment.")
    print()

    print("Required classical benchmark ladder:")
    for row in benchmark_rows:
        if row["required"]:
            print(f"  {row['benchmark_id']}: {row['model']}")
    print()

    print("Closed AI registry:")
    for row in ai_rows:
        print(f"  {row['candidate_id']}: {row['model']}")
    print()

    print("Frozen premise-test materiality:")
    print("SUPPORTS canine added value:")
    print("  paired delta Uno C >= +0.02")
    print("  bootstrap P(delta C > 0) >= 0.90")
    print("  delta IBS <= +0.01")
    print("ARGUES AGAINST canine added value:")
    print("  paired delta Uno C <= -0.02")
    print("  bootstrap P(delta C < 0) >= 0.90")
    print("otherwise: INCONCLUSIVE_NEUTRAL")
    print()

    print("=" * 120)
    print("02i REVISED PRE-OUTCOME DESIGN SUMMARY")
    print("=" * 120)
    print("02g primary DOG2->TARGET genes: 11,815")
    print("02g common-all-four genes: 8,652")
    print("02h Hallmark primary eligible: 50/50")
    print("02h Hallmark common-all-four eligible: 49/50")
    print("02h Reactome primary eligible: 1,121")
    print("02h Reactome common-all-four eligible: 971")
    print()
    print("DOG2 outcome values read: NO")
    print("Human outcome values read: NO")
    print("Clinical values read: NO")
    print("Expression values read: NO")
    print("Model fitting: NO")
    print()
    print("Next:")
    print("  03a: freeze exact computational Source Prognostic Gate protocol.")
    print("  03b: run Source Prognostic Gate and obtain the first central Paper-6 result.")
    print()
    print("Artifacts:")
    for path in [
        SOURCE_GATE_RULES,
        ARM_GATE_RULES,
        BENCHMARK_REGISTRY,
        AI_REGISTRY,
        PREMISE_RULES,
        CONTRACT_JSON,
        SUMMARY_JSON,
    ]:
        print(f"  {path.relative_to(ROOT)}")

    print("=" * 120)
    print("02i final broad pre-outcome design freeze: PASS")
    print("=" * 120)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 120, file=sys.stderr)
        print("02i final broad pre-outcome design freeze: FAIL", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print("=" * 120, file=sys.stderr)
        raise
