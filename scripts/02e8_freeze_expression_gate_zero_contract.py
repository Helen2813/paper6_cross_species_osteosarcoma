#!/usr/bin/env python3
"""
Paper 6 - freeze expression-level Gate Zero arm/confounding contract.

OFFLINE ONLY. This script DOES NOT read the DOG² expression matrix.

It freezes, before expression values are opened:
  - exact DOG² 186-sample cohort and 93/93 arm labels;
  - exact script-00 locked expression asset identity;
  - exact 02a authoritative sample -> COTC bridge;
  - baseline covariates to use for adjustment sensitivity, selected
    deterministically as all 02e7 DOG2_ARM_BALANCE warning variables;
  - expression preprocessing, feature scales, PCA/PERMANOVA,
    arm-prediction CV, permutations, and decision thresholds.

No outcomes, response, follow-up, treatment-administration values,
post-baseline sample annotations, or omics values are read.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


SCRIPT_VERSION = "02e8-freeze-expression-gate-zero-contract-v1-no-cli"

ROOT = Path(__file__).resolve().parents[1]

UPSTREAM_LOCK = ROOT / "contracts" / "00_upstream_input_lock.json"
UPSTREAM_MANIFEST = ROOT / "manifests" / "00_upstream_input_manifest.csv"

MAP_LOCK = ROOT / "contracts" / "02a_dog2_authoritative_id_mapping_lock.json"
MAP_FILE = ROOT / "manifests" / "02a_dog2_selected186_authoritative_id_mapping.csv"

E7_DIR = ROOT / "results" / "dog2_selection_audit" / "02e7"
E7_SUMMARY = E7_DIR / "summary.json"
E7_ROSTER = E7_DIR / "dog2_rna186_roster.tsv"
E7_BALANCE = E7_DIR / "baseline_selection_balance.tsv"
E7_AUDIT = E7_DIR / "selection_audit.json"

OUT_DIR = ROOT / "results" / "expression_gate_zero_contract" / "02e8"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONTRACT_JSON = OUT_DIR / "expression_gate_zero_contract.json"
ADJUSTMENT_TSV = OUT_DIR / "baseline_adjustment_covariates.tsv"
SUMMARY_JSON = OUT_DIR / "summary.json"

EXPECTED_N = 186
EXPECTED_ARMS = {"COTC021": 93, "COTC022": 93}
EXPR_REL = "data/processed/GSE238110_DOG2_expression_log2cpm_matched_allgenes.csv"
EXPECTED_FEATURES = 21016

PRIMARY_TOP_N = 5000
SENSITIVITY_TOP_N = [2000, 10000]
N_PCS = 20

CV_SPLITS = 5
CV_REPEATS = 20
PERMANOVA_PERMUTATIONS = 5000
CLASSIFIER_PERMUTATIONS = 1000
RANDOM_SEED = 20260828

GREEN_R2_MAX = 0.05
GREEN_AUC_MAX = 0.65
RED_R2_MIN = 0.10
RED_AUC_MIN = 0.75

PC_ABS_SMD_WARNING = 0.50
GENE_Q = 0.05
GENE_ABS_SMD = 0.50
GENE_STRONG_FRACTION_WARNING = 0.005


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def get_hash(mapping: Dict[str, Any], path: Path) -> str:
    key = rel(path)
    value = mapping.get(key)
    if value is None:
        value = mapping.get(key.replace("/", "\\"))
    if isinstance(value, dict):
        value = value.get("sha256")
    return str(value or "")


def text(value: Any) -> str:
    if value is None:
        return ""
    out = str(value).strip()
    if out.lower() in {"nan", "none", "null", "na", "n/a"}:
        return ""
    return out


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    v = text(value).lower()
    if v in {"true", "1", "yes", "y"}:
        return True
    if v in {"false", "0", "no", "n", ""}:
        return False
    raise RuntimeError(f"Cannot parse boolean value {value!r}")


def verify_script00() -> Dict[str, Any]:
    for path in (UPSTREAM_LOCK, UPSTREAM_MANIFEST):
        if not path.exists():
            raise FileNotFoundError(f"Missing script-00 artifact: {path}")

    lock = read_json(UPSTREAM_LOCK)
    expected_manifest = text(lock.get("manifest_sha256"))
    observed_manifest = sha256_file(UPSTREAM_MANIFEST)
    if not expected_manifest or expected_manifest.lower() != observed_manifest.lower():
        raise RuntimeError("Script-00 upstream manifest hash mismatch.")

    asset = (lock.get("assets") or {}).get("dog2_expression")
    if not isinstance(asset, dict):
        raise RuntimeError("Script-00 lock lacks dog2_expression.")

    if asset.get("relative_path") != EXPR_REL:
        raise RuntimeError(
            f"Locked expression path changed: {asset.get('relative_path')!r}"
        )
    if int(asset.get("n_rows") or 0) != EXPECTED_N:
        raise RuntimeError(
            f"Locked expression rows changed: {asset.get('n_rows')!r}"
        )
    if int(asset.get("n_columns") or 0) != EXPECTED_FEATURES:
        raise RuntimeError(
            f"Locked expression features changed: {asset.get('n_columns')!r}"
        )

    asset_sha = text(asset.get("sha256"))
    if len(asset_sha) != 64:
        raise RuntimeError("Locked dog2_expression SHA256 missing/malformed.")

    return {
        "manifest_sha256": observed_manifest,
        "expression_sha256": asset_sha,
        "expression_relative_path": EXPR_REL,
        "expression_rows": EXPECTED_N,
        "expression_features": EXPECTED_FEATURES,
    }


def verify_mapping() -> Dict[str, Any]:
    for path in (MAP_LOCK, MAP_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Missing 02a mapping artifact: {path}")

    lock = read_json(MAP_LOCK)
    if lock.get("status") != "PASS":
        raise RuntimeError("02a mapping lock status is not PASS.")

    required_checks = (
        "locked_rows_186",
        "authoritative_cotc_186",
        "authoritative_numeric_suffixes_unique_186",
        "locked_patient_ids_unique_186",
        "locked_patient_id_set_equals_authoritative_suffix_set",
        "mapped_cotc_unique_186",
        "mapped_cotc_set_equals_authoritative_set",
    )
    checks = lock.get("checks") or {}
    failed = [key for key in required_checks if checks.get(key) is not True]
    if failed:
        raise RuntimeError("02a mapping checks failed: " + ", ".join(failed))

    expected_sha = text(lock.get("output_mapping_sha256"))
    actual_sha = sha256_file(MAP_FILE)
    if not expected_sha or expected_sha.lower() != actual_sha.lower():
        raise RuntimeError("02a mapping file hash mismatch.")

    rows: List[Dict[str, str]] = []
    with MAP_FILE.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"paper4_sample_id", "paper4_patient_id", "cotc_subject_id"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise RuntimeError("02a mapping lacks required columns.")
        rows.extend(reader)

    if len(rows) != EXPECTED_N:
        raise RuntimeError(f"02a mapping rows={len(rows)}, expected 186.")

    sample_ids = [text(row["paper4_sample_id"]) for row in rows]
    cotc_ids = [text(row["cotc_subject_id"]) for row in rows]

    if "" in sample_ids or len(set(sample_ids)) != EXPECTED_N:
        raise RuntimeError("02a paper4_sample_id is not exact 186 unique.")
    if "" in cotc_ids or len(set(cotc_ids)) != EXPECTED_N:
        raise RuntimeError("02a cotc_subject_id is not exact 186 unique.")

    return {
        "sha256": actual_sha,
        "sample_ids": sorted(sample_ids),
        "cotc_ids": sorted(cotc_ids),
    }


def verify_e7() -> Dict[str, Any]:
    for path in (E7_SUMMARY, E7_ROSTER, E7_BALANCE, E7_AUDIT):
        if not path.exists():
            raise FileNotFoundError(f"Missing 02e7 artifact: {path}")

    summary = read_json(E7_SUMMARY)
    if summary.get("status") != "PASS":
        raise RuntimeError("02e7 summary status is not PASS.")
    if int(summary.get("dog2_selected_n") or 0) != EXPECTED_N:
        raise RuntimeError("02e7 selected n is not 186.")
    if summary.get("dog2_counts_by_study") != EXPECTED_ARMS:
        raise RuntimeError("02e7 arm counts are not exact 93/93.")

    for key in (
        "network_access",
        "outcome_response_followup_values_read",
        "treatment_administration_values_read",
        "postbaseline_sample_annotations_read",
        "omics_values_read",
        "omics_file_contents_read",
    ):
        if summary.get(key) is not False:
            raise RuntimeError(f"02e7 provenance mismatch: {key}")

    hashes = summary.get("final_artifact_hashes") or {}
    verified = {}
    for path in (E7_ROSTER, E7_BALANCE, E7_AUDIT):
        expected = get_hash(hashes, path)
        actual = sha256_file(path)
        if not expected or expected.lower() != actual.lower():
            raise RuntimeError(f"02e7 hash mismatch for {path.name}.")
        verified[rel(path)] = actual

    roster_rows: List[Dict[str, str]] = []
    with E7_ROSTER.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"case_id", "study", "gate_zero_arm"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise RuntimeError("02e7 roster lacks required columns.")
        roster_rows.extend(reader)

    if len(roster_rows) != EXPECTED_N:
        raise RuntimeError("02e7 roster does not contain 186 rows.")

    counts = {"COTC021": 0, "COTC022": 0}
    case_ids = []
    for row in roster_rows:
        case_id = text(row["case_id"])
        study = text(row["study"])
        arm = text(row["gate_zero_arm"])
        if study not in counts:
            raise RuntimeError(f"Unexpected 02e7 study: {study!r}")
        expected_arm = "SOC_PLUS_RAPAMYCIN" if study == "COTC021" else "SOC_CONTROL"
        if arm != expected_arm:
            raise RuntimeError(f"02e7 arm mapping mismatch for {case_id}.")
        counts[study] += 1
        case_ids.append(case_id)

    if counts != EXPECTED_ARMS:
        raise RuntimeError(f"02e7 roster counts changed: {counts}")
    if "" in case_ids or len(set(case_ids)) != EXPECTED_N:
        raise RuntimeError("02e7 roster case IDs not exact 186 unique.")

    audit = read_json(E7_AUDIT)
    if audit.get("status") != "PASS":
        raise RuntimeError("02e7 selection audit status is not PASS.")

    return {
        "case_ids": sorted(case_ids),
        "verified_hashes": verified,
        "arm_warning_count": int(summary.get("dog2_arm_balance_warning_count") or 0),
        "selection_warning_count": int(
            summary.get("selection_related_balance_warning_count") or 0
        ),
        "selection_gate_status": summary.get("gate_status"),
    }


def adjustment_covariates(expected_count: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []

    with E7_BALANCE.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {
            "comparison",
            "variable",
            "variable_type",
            "warning",
            "warning_reasons",
        }
        if not required.issubset(set(reader.fieldnames or [])):
            raise RuntimeError("02e7 balance TSV lacks required columns.")

        for row in reader:
            if text(row.get("comparison")) != "DOG2_ARM_BALANCE":
                continue
            if not bool_value(row.get("warning")):
                continue

            variable = text(row.get("variable"))
            variable_type = text(row.get("variable_type"))
            if not variable or variable_type not in {"numeric", "categorical"}:
                raise RuntimeError(f"Malformed 02e7 balance warning row: {row}")

            rows.append(
                {
                    "variable": variable,
                    "variable_type": variable_type,
                    "warning_reasons": text(row.get("warning_reasons")),
                    "selection_rule": "02e7_DOG2_ARM_BALANCE_warning_TRUE",
                }
            )

    rows.sort(key=lambda x: x["variable"])

    if len(rows) != expected_count:
        raise RuntimeError(
            "02e7 arm-warning count mismatch: "
            f"summary={expected_count}, TSV={len(rows)}"
        )
    return rows


def write_adjustment(rows: List[Dict[str, str]]) -> None:
    fields = ["variable", "variable_type", "warning_reasons", "selection_rule"]
    with ADJUSTMENT_TSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    print("=" * 112)
    print("Paper 6 - freeze expression-level Gate Zero arm/confounding contract")
    print("=" * 112)
    print(f"Script version: {SCRIPT_VERSION}")
    print()
    print("Execution contract:")
    print("  Network/API access in 02e8: NO")
    print("  DOG² expression values read: NO")
    print("  Any omics values read: NO")
    print("  Outcome/response/follow-up values read: NO")
    print("  Treatment-administration values read: NO")
    print("  Post-baseline sample annotations read: NO")
    print()

    started = now_utc()

    lock00 = verify_script00()
    mapping = verify_mapping()
    e7 = verify_e7()

    if set(mapping["cotc_ids"]) != set(e7["case_ids"]):
        raise RuntimeError("02a COTC set != 02e7 DOG² roster set.")

    adjust = adjustment_covariates(e7["arm_warning_count"])
    write_adjustment(adjust)

    contract = {
        "script_version": SCRIPT_VERSION,
        "status": "PASS",
        "run_started_utc": started,
        "scientific_stage": "PRE_EXPRESSION_GATE_ZERO_DESIGN_FREEZE",
        "question": (
            "Is randomized study identity materially encoded in the DOG2 "
            "transcriptome before cross-species transfer or outcome access?"
        ),
        "cohort": {
            "n": EXPECTED_N,
            "arm_field": "study.clinical_study_designation",
            "arm_counts": EXPECTED_ARMS,
            "arm_map": {
                "COTC021": "SOC_PLUS_RAPAMYCIN",
                "COTC022": "SOC_CONTROL",
            },
            "roster_path": rel(E7_ROSTER),
            "roster_sha256": sha256_file(E7_ROSTER),
        },
        "expression_asset": lock00,
        "authoritative_mapping": {
            "path": rel(MAP_FILE),
            "sha256": mapping["sha256"],
            "expression_key": "paper4_sample_id",
            "case_key": "cotc_subject_id",
            "row_order_matching_allowed": False,
        },
        "preprocessing": {
            "require_exact_186_expression_rows": True,
            "require_expression_index_equals_02a_sample_id_set": True,
            "require_02a_cotc_set_equals_02e7_roster": True,
            "nonfinite_policy": "FAIL_IF_ANY_AFTER_NUMERIC_COERCION",
            "zero_variance_genes": "REMOVE",
            "variance_ranking": "label_blind_sample_variance_across_all_186",
            "unsupervised_global_scaling": "gene_wise_zscore",
            "predictive_cv_preprocessing": (
                "feature_variance_ranking_scaling_and_PCA_fit_within_training_fold"
            ),
        },
        "feature_scales": [
            {
                "name": "primary_top5000",
                "top_variable_genes": PRIMARY_TOP_N,
                "role": "PRIMARY",
            },
            {
                "name": "sensitivity_top2000",
                "top_variable_genes": 2000,
                "role": "SENSITIVITY",
            },
            {
                "name": "sensitivity_top10000",
                "top_variable_genes": 10000,
                "role": "SENSITIVITY",
            },
        ],
        "global_structure": {
            "pca_components": N_PCS,
            "permanova": {
                "space": "PC1-PC20",
                "distance": "euclidean",
                "factor": "COTC021_vs_COTC022",
                "permutations": PERMANOVA_PERMUTATIONS,
                "random_seed": RANDOM_SEED,
                "report": ["pseudo_F", "R2_arm", "permutation_p"],
            },
            "pc_supporting_diagnostics": {
                "welch_t_each_PC": True,
                "BH_across_20_PCs": True,
                "absolute_SMD_warning": PC_ABS_SMD_WARNING,
                "gate_role": "SUPPORTING_ONLY",
            },
        },
        "arm_predictability": {
            "model": "L2_logistic_regression",
            "input": "training_fold_top_variable_genes_to_PCA20",
            "cv_splits": CV_SPLITS,
            "cv_repeats": CV_REPEATS,
            "random_seed": RANDOM_SEED,
            "metrics": ["ROC_AUC", "balanced_accuracy"],
            "label_permutations": CLASSIFIER_PERMUTATIONS,
            "permutation_preserves_93_93_counts": True,
        },
        "gene_level_supporting": {
            "test": "Welch_two_sample_t",
            "effect_size": "SMD_COTC021_minus_COTC022",
            "multiple_testing": "Benjamini_Hochberg",
            "strong_gene_definition": {
                "q_lt": GENE_Q,
                "abs_SMD_gte": GENE_ABS_SMD,
            },
            "strong_gene_fraction_warning": GENE_STRONG_FRACTION_WARNING,
            "gate_role": "SUPPORTING_ONLY",
        },
        "baseline_adjustment_sensitivity": {
            "enabled": len(adjust) > 0,
            "selection_rule": "all_and_only_02e7_DOG2_ARM_BALANCE_warning_TRUE",
            "covariates": adjust,
            "numeric_encoding": "zscore",
            "categorical_encoding": "one_hot_with_explicit_missing_level",
            "purpose": (
                "check whether arm-expression structure persists after accounting "
                "for baseline variables already flagged before expression access"
            ),
            "may_change_primary_gate": False,
        },
        "gate_thresholds": {
            "GREEN": {
                "primary_R2_lt": GREEN_R2_MAX,
                "AND_primary_CV_AUC_lt": GREEN_AUC_MAX,
            },
            "AMBER": {
                "condition": (
                    "not_RED_and_(primary_R2_gte_0.05_or_primary_CV_AUC_gte_0.65)"
                )
            },
            "RED": {
                "primary_R2_gte": RED_R2_MIN,
                "OR_primary_CV_AUC_gte": RED_AUC_MIN,
            },
            "p_values_alone_change_gate": False,
            "sensitivity_scale_crossing_RED": "HOLD_FOR_DISCORDANCE_REVIEW",
        },
        "hard_guardrails": {
            "no_outcome_access": True,
            "no_response_access": True,
            "no_follow_up_access": True,
            "no_treatment_administration_access": True,
            "no_postbaseline_sample_annotation_access": True,
            "no_paper4_clinical_file_read": True,
            "no_arm_supervised_feature_selection": True,
            "no_row_order_matching": True,
            "do_not_drop_samples_based_on_gate_results": True,
            "do_not_change_thresholds_after_expression_opened": True,
        },
        "upstream_hashes": {
            "script00_manifest_sha256": lock00["manifest_sha256"],
            "02a_mapping_sha256": mapping["sha256"],
            "02e7": e7["verified_hashes"],
        },
    }

    write_json(CONTRACT_JSON, contract)
    contract_sha = sha256_file(CONTRACT_JSON)
    adjustment_sha = sha256_file(ADJUSTMENT_TSV)

    summary = {
        "script_version": SCRIPT_VERSION,
        "status": "PASS",
        "run_started_utc": started,
        "run_finished_utc": now_utc(),
        "network_access": False,
        "dog2_expression_values_read": False,
        "omics_values_read": False,
        "outcome_response_followup_values_read": False,
        "treatment_administration_values_read": False,
        "postbaseline_sample_annotations_read": False,
        "cohort_n": EXPECTED_N,
        "arm_counts": EXPECTED_ARMS,
        "expression_relative_path": EXPR_REL,
        "expression_sha256": lock00["expression_sha256"],
        "expression_shape": [EXPECTED_N, EXPECTED_FEATURES],
        "primary_top_variable_genes": PRIMARY_TOP_N,
        "sensitivity_top_variable_genes": SENSITIVITY_TOP_N,
        "pca_components": N_PCS,
        "permanova_permutations": PERMANOVA_PERMUTATIONS,
        "classifier_permutations": CLASSIFIER_PERMUTATIONS,
        "baseline_adjustment_covariate_count": len(adjust),
        "baseline_adjustment_covariates": [row["variable"] for row in adjust],
        "gate_thresholds": {
            "green_R2_max": GREEN_R2_MAX,
            "green_AUC_max": GREEN_AUC_MAX,
            "red_R2_min": RED_R2_MIN,
            "red_AUC_min": RED_AUC_MIN,
        },
        "contract_path": rel(CONTRACT_JSON),
        "contract_sha256": contract_sha,
        "adjustment_path": rel(ADJUSTMENT_TSV),
        "adjustment_sha256": adjustment_sha,
    }
    write_json(SUMMARY_JSON, summary)

    print("Upstream verification:")
    print("  script 00 upstream lock: PASS")
    print("  locked DOG² expression identity: PASS")
    print("  02a authoritative sample/COTC bridge: PASS")
    print("  02e7 exact DOG² roster: PASS")
    print("  02a COTC set == 02e7 COTC set: PASS")
    print()
    print("Frozen cohort:")
    print("  total: 186")
    print("  COTC021: 93")
    print("  COTC022: 93")
    print()
    print("Frozen expression input [IDENTITY ONLY; VALUES NOT READ]:")
    print(f"  {EXPR_REL}")
    print("  shape: 186 x 21,016")
    print(f"  SHA256: {lock00['expression_sha256']}")
    print()
    print("Baseline adjustment covariates selected from 02e7:")
    if adjust:
        for row in adjust:
            print(
                f"  {row['variable']} [{row['variable_type']}] "
                f"({row['warning_reasons']})"
            )
    else:
        print("  <none>")
    print()
    print("Primary Gate Zero specification:")
    print("  primary: top 5,000 label-blind variable genes")
    print("  sensitivities: top 2,000 and top 10,000")
    print("  global space: PC1-PC20")
    print(f"  PERMANOVA permutations: {PERMANOVA_PERMUTATIONS}")
    print(f"  arm classifier CV: {CV_SPLITS}-fold x {CV_REPEATS} repeats")
    print(f"  classifier label permutations: {CLASSIFIER_PERMUTATIONS}")
    print()
    print("Frozen decision thresholds:")
    print(f"  GREEN: R2 < {GREEN_R2_MAX:.2f} AND CV AUC < {GREEN_AUC_MAX:.2f}")
    print(f"  RED:   R2 >= {RED_R2_MIN:.2f} OR CV AUC >= {RED_AUC_MIN:.2f}")
    print("  otherwise: AMBER")
    print("  p-values alone do not change the gate")
    print()
    print(f"Artifacts: {OUT_DIR.relative_to(ROOT)}")
    print(f"  {CONTRACT_JSON.name}")
    print(f"  {ADJUSTMENT_TSV.name}")
    print(f"  {SUMMARY_JSON.name}")
    print(f"Contract SHA256: {contract_sha}")
    print()
    print("DOG² expression values read: NO")
    print("Outcome/response/follow-up values read: NO")
    print("Treatment-administration values read: NO")
    print("Post-baseline sample annotations read: NO")
    print("Any omics values read: NO")
    print()
    print("02e8 expression Gate Zero design freeze: PASS")
    print("=" * 112)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 112, file=sys.stderr)
        print("02e8 expression Gate Zero design freeze: FAIL", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print("=" * 112, file=sys.stderr)
        raise
