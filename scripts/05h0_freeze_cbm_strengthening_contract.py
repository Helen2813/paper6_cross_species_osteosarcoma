from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "05h0-freeze-cbm-strengthening-contract-v1-no-cli"

# This script freezes only the post-HOLD strengthening plan.
# It MUST NOT read scientific result tables, target outcomes, or sealed cohorts.
CONTRACT_DIRNAME = "05h0_cbm_strengthening_contract"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_bytes(obj: Any) -> bytes:
    return (
        json.dumps(
            obj,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def project_root() -> Path:
    # .../scripts/05h0_*.py -> project root
    return Path(__file__).resolve().parents[1]


def build_contract() -> dict[str, Any]:
    return {
        "contract_name": "Paper 6 CBM pre-submission strengthening contract",
        "contract_version": "cbm-strengthening-v1",
        "script_version": SCRIPT_VERSION,
        "purpose": (
            "Freeze a limited, post-HOLD strengthening program for submission to "
            "Computers in Biology and Medicine without reopening the primary benchmark, "
            "model selection, TARGET interpretation, or sealed external cohorts."
        ),
        "status_of_existing_science": {
            "primary_HOLD_decision": "LOCKED_AND_UNCHANGED",
            "selectable_architectures": ["A2", "A3"],
            "selectable_architecture_eligibility": "LOCKED_AND_UNCHANGED",
            "model_architectures": "LOCKED_AND_UNCHANGED",
            "hyperparameters": "LOCKED_AND_UNCHANGED",
            "benchmark_predictions": "LOCKED; reuse retained outputs only",
            "TARGET_primary_interpretation": "LOCKED; T-D remains unchanged",
            "frozen_primary_negative_transfer_definition": {
                "delta_uno_c_threshold": -0.02,
                "maximum_allowed_rate": 0.10,
            },
            "frozen_primary_catastrophic_transfer_definition": {
                "delta_uno_c_threshold": -0.05,
                "maximum_allowed_rate": 0.05,
            },
        },
        "prohibited_actions": [
            "No refitting of the original 21,600-replicate benchmark models.",
            "No new architecture, hyperparameter, gate, or transfer recipe.",
            "No retrospective change to A2/A3 eligibility.",
            "No retrospective change to the frozen HOLD safety rules.",
            "No reopening or replacement of the frozen TARGET T-D interpretation.",
            "No outcome access for GSE21257.",
            "No outcome access for GSE39055.",
            "No CORAL or other new domain-adaptation method.",
            "No selection of sensitivity definitions after seeing sensitivity results.",
            "No suppression of an unfavorable strengthening result.",
        ],
        "required_strengthening_analyses": {
            "anchor_provenance": {
                "goal": (
                    "Make M34, M40, M11, and M24 anchors independently auditable "
                    "without requiring acceptance of the separate molecular-preservation paper."
                ),
                "programs": ["M34", "M40", "M11", "M24"],
                "required_fields": [
                    "program",
                    "frozen_source_gene_count",
                    "frozen_membership",
                    "frozen_weights_or_signs",
                    "prior_preservation_class",
                    "prior_edge_statistic_if_available",
                    "prior_loading_statistic_if_available",
                    "prior_specificity_result_if_available",
                    "freeze_date_or_timestamp_if_available",
                    "artifact_sha256",
                    "source_dataset_identifiers",
                ],
                "machine_readable_membership_weights_required": True,
                "rule": (
                    "Use the authoritative pre-existing frozen artifacts. "
                    "Do not recreate memberships, weights, or labels from Paper 6 outcomes."
                ),
            },
            "all_model_safety_summary": {
                "analysis_role": "POST-HOLD DESCRIPTIVE CONTEXT",
                "models": ["B0", "B4", "A0", "A1", "A2", "A3", "A4"],
                "reference_model": "B0",
                "minimum_reported_fields": [
                    "model",
                    "frozen_role",
                    "eligible_for_primary_selection",
                    "mean_delta_uno_c_vs_B0_if_defined",
                    "negative_transfer_rate_at_deltaC_le_-0.02",
                    "catastrophic_transfer_rate_at_deltaC_le_-0.05",
                    "regime_specific_rates_or_effects_if_already_estimable",
                ],
                "interpretation_rule": (
                    "Non-eligible models remain non-eligible. Their results are descriptive "
                    "and cannot reopen the frozen primary architecture-selection decision."
                ),
            },
            "aggregation_sensitivity": {
                "analysis_role": "POST-HOLD ROBUSTNESS ANALYSIS",
                "primary_models": ["A2", "A3"],
                "estimands": {
                    "frozen_original": {
                        "scenario_set": "all 180 frozen scenarios",
                        "weighting": "equal weight per frozen scenario",
                        "role": "PRIMARY; unchanged",
                    },
                    "core_only": {
                        "scenario_set": (
                            "exactly the 36 pre-stress core scenarios from the authoritative "
                            "frozen scenario design"
                        ),
                        "weighting": "equal weight per core scenario",
                        "guardrail": (
                            "Resolve the core set only from the frozen scenario-design metadata; "
                            "fail if it is not exactly 36 scenarios."
                        ),
                    },
                    "regime_balanced": {
                        "scenario_set": "all 180 frozen scenarios",
                        "regimes": ["R0", "R1", "R2", "R3", "R4", "R5"],
                        "weighting": (
                            "first aggregate equally across scenarios within each regime, "
                            "then assign each of the six regimes weight 1/6"
                        ),
                        "guardrail": (
                            "Regime labels must come from the authoritative frozen scenario design."
                        ),
                    },
                },
                "minimum_reported_metrics": [
                    "mean_delta_uno_c",
                    "negative_transfer_rate_at_deltaC_le_-0.02",
                    "catastrophic_transfer_rate_at_deltaC_le_-0.05",
                ],
                "uncertainty_policy": {
                    "preferred_if_retained_replicate_outputs_exist": (
                        "Estimate Monte Carlo uncertainty by resampling retained replicates "
                        "within each fixed scenario and recomputing the aggregate."
                    ),
                    "if_replicate_outputs_are_not_retained": (
                        "Report deterministic sensitivity summaries and state that no additional "
                        "Monte Carlo interval was reconstructed. Do not refit models solely "
                        "to manufacture uncertainty."
                    ),
                    "scenario_design_interpretation": (
                        "Do not treat the fixed 36 or 180 scenarios as a random sample from "
                        "a superpopulation of scenarios."
                    ),
                },
            },
            "threshold_definition_sensitivity": {
                "analysis_role": (
                    "POST-HOLD SENSITIVITY OF THE ESTIMATED SAFETY PROFILE; "
                    "NOT A REOPENED PASS/FAIL DECISION"
                ),
                "primary_models": ["A2", "A3"],
                "negative_transfer_severity_grid_delta_uno_c": [
                    -0.01,
                    -0.02,
                    -0.03,
                ],
                "catastrophic_transfer_severity_grid_delta_uno_c": [
                    -0.04,
                    -0.05,
                    -0.06,
                ],
                "reference_rate_lines": {
                    "negative_transfer": [0.05, 0.10, 0.15],
                    "catastrophic_transfer": [0.025, 0.05, 0.075],
                },
                "presentation_rule": (
                    "Report how estimated rates vary with prespecified neighboring definitions. "
                    "The frozen thresholds (-0.02, -0.05; 0.10, 0.05 rate limits) remain the "
                    "only primary decision rules and are never replaced by a more favorable pair."
                ),
            },
            "target_km_ibs_reference": {
                "analysis_role": "POST-OPENING DESCRIPTIVE NULL REFERENCE",
                "purpose": (
                    "Contextualize the low IBS of a near-constant-risk transfer model without "
                    "using the reference for model selection."
                ),
                "allowed_new_estimation": True,
                "method": (
                    "Within each already-frozen TARGET outer-training split, estimate a "
                    "no-covariate Kaplan-Meier survival curve using only outer-training outcomes; "
                    "assign that same curve to every corresponding outer-test patient; evaluate "
                    "IBS using the same frozen time horizon, censoring mechanics, fold structure, "
                    "and aggregation used for the existing TARGET IBS results."
                ),
                "prohibited_uses": [
                    "No tuning.",
                    "No model selection.",
                    "No change to T-A/T-B/T-C/T-D.",
                    "No use of full-cohort outcomes to fit a test-fold predictor.",
                ],
            },
        },
        "required_interpretive_fixes": {
            "compatibility_terminology": {
                "outcome_using_measure": "training-set prognostic compatibility",
                "outcome_blind_measure": "outcome-blind structural concordance",
                "rule": (
                    "Do not describe a target-training-outcome Uno-C quantity as outcome-blind."
                ),
            },
            "zero_shot_terminology": {
                "preferred_term": "no-target-outcome transfer",
                "rule": (
                    "Where target-training expression means/SDs are used, avoid implying that "
                    "the target distribution was completely unseen."
                ),
            },
            "operational_auroc_below_half": {
                "required_fact": (
                    "Explicitly acknowledge that AUROC 0.254 has complementary orientation 0.746."
                ),
                "interpretation_rule": (
                    "Do not claim that score inversion was forbidden unless an authoritative "
                    "pre-result contract explicitly says so. The substantive interpretation "
                    "must instead use the residualized and within-scenario evidence to assess "
                    "whether the marginal inversion is stable."
                ),
            },
            "target_N4_N5": {
                "required_fact": (
                    "Explicitly acknowledge that N4/N5 have the highest TARGET discrimination "
                    "point estimates if that remains true in the frozen table."
                ),
                "interpretation_rule": (
                    "Do not infer a ranking when intervals overlap and no frozen N4-vs-N5 "
                    "contrast was specified."
                ),
            },
            "T1_IBS": {
                "rule": (
                    "Do not present the lowest T1 IBS as evidence of superior individualized "
                    "prediction without context from its compressed risk distribution and the "
                    "descriptive no-covariate reference."
                ),
            },
        },
        "presentation_and_submission_fixes": {
            "figure_4": (
                "Rebuild as a compact 2x2 grid with consistent panel dimensions, readable fonts, "
                "and minimal unused whitespace."
            ),
            "figure_5b": (
                "Remove internal wording 'Paper-4 anchor'; use publication-facing wording such as "
                "'Frozen external-preservation anchors'."
            ),
            "literature": (
                "Add only publisher/PubMed/DOI-verified recent work relevant to transcriptomic "
                "survival transfer, negative transfer, cross-domain/cross-species transfer, "
                "event-limited survival modeling, and omics domain adaptation."
            ),
            "format": (
                "Convert the final manuscript from the IEEE microwave example shell to the "
                "Computers in Biology and Medicine / Elsevier submission structure."
            ),
            "supplement": (
                "Prepare Supplementary Material as a separate submission file rather than keeping "
                "the full supplement inside the main manuscript."
            ),
            "code_availability": (
                "Remove copied Molecular Transport Audit language and state only the actual "
                "Paper 6 repository/material availability at submission."
            ),
            "ai_declaration": (
                "Use a separate Elsevier-compatible declaration immediately before References."
            ),
        },
        "sealed_future_assets": {
            "GSE21257_outcomes": "SEALED",
            "GSE39055_outcomes": "SEALED",
            "reason": (
                "Do not spend untouched external cohorts to rescue or alter the current TARGET "
                "answer; reserve them for prospectively specified future work."
            ),
        },
        "reporting_commitment": (
            "Every analysis authorized by this contract must be reported regardless of whether "
            "it strengthens, weakens, or leaves unchanged the current manuscript conclusions."
        ),
    }


def build_markdown(contract: dict[str, Any]) -> str:
    return f"""# Paper 6 CBM strengthening contract

**Contract version:** `{contract['contract_version']}`  
**Script version:** `{contract['script_version']}`

## Purpose

{contract['purpose']}

## Existing science remains locked

- Primary HOLD decision: **LOCKED AND UNCHANGED**
- Selectable architectures: **A2 and A3 only**
- Original architectures and hyperparameters: **LOCKED**
- Original benchmark predictions: **reuse only; no refitting**
- TARGET interpretation: **T-D remains unchanged**
- Primary negative-transfer rule: **Delta Uno-C <= -0.02; maximum allowed rate 0.10**
- Primary catastrophic-transfer rule: **Delta Uno-C <= -0.05; maximum allowed rate 0.05**
- GSE21257 outcomes: **SEALED**
- GSE39055 outcomes: **SEALED**

## Authorized strengthening work

1. Make the M34/M40/M11/M24 anchor provenance self-contained and machine-readable.
2. Report an all-model descriptive safety summary for B0, B4, A0, A1, A2, A3, and A4.
3. Recompute A2/A3 aggregate summaries under:
   - the unchanged original 180-scenario equal-scenario weighting;
   - the exact 36 pre-stress core scenarios;
   - a six-regime balanced estimand giving R0--R5 equal weight.
4. Report prespecified neighboring negative/catastrophic-transfer severity definitions as
   **sensitivity of the estimated safety profile**, never as a reopened pass/fail decision.
5. Add a post-opening no-covariate Kaplan-Meier IBS reference using only frozen outer-training
   outcomes and the existing TARGET evaluation mechanics.
6. Fix terminology and interpretation for prognostic compatibility, no-target-outcome transfer,
   AUROC 0.254, N4/N5, and T1 IBS.
7. Rebuild Figure 4, clean Figure 5B, verify recent literature, and convert the package to
   Computers in Biology and Medicine / Elsevier submission structure.

## Explicit prohibitions

- No new transfer architecture or domain-adaptation method.
- No original benchmark refitting.
- No retrospective A2/A3 eligibility changes.
- No retrospective threshold changes.
- No reopening TARGET T-D.
- No GSE21257/GSE39055 outcome access.
- No hiding an unfavorable sensitivity result.

## Reporting commitment

{contract['reporting_commitment']}
"""


def main() -> None:
    root = project_root()
    scripts_dir = root / "scripts"
    output_dir = root / "method_contract" / CONTRACT_DIRNAME

    print("=" * 112)
    print("Paper 6 - freeze CBM pre-submission strengthening contract")
    print("=" * 112)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {root}")
    print(f"Output directory: {output_dir}")
    print()
    print("Safety / execution contract:")
    print("  scientific result tables read: NO")
    print("  TARGET outcome values read: NO")
    print("  GSE21257 outcomes read: NO")
    print("  GSE39055 outcomes read: NO")
    print("  model fitting: NO")
    print("  scientific metric calculation: NO")
    print("  existing HOLD/TARGET decisions modified: NO")
    print()

    if not scripts_dir.is_dir():
        raise FileNotFoundError(f"Expected scripts directory does not exist: {scripts_dir}")

    if output_dir.exists():
        raise FileExistsError(
            f"Immutable freeze directory already exists: {output_dir}\n"
            "Do not overwrite a scientific freeze. Inspect the existing files instead."
        )

    # Snapshot code state only. We hash Python scripts as bytes; we do not import or execute them.
    script_inventory: list[dict[str, Any]] = []
    for path in sorted(scripts_dir.glob("*.py"), key=lambda p: p.name.lower()):
        script_inventory.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    contract = build_contract()
    contract_bytes = canonical_json_bytes(contract)
    markdown_bytes = build_markdown(contract).encode("utf-8")
    inventory_bytes = canonical_json_bytes(
        {
            "inventory_type": "Python script byte-hash snapshot at CBM strengthening freeze",
            "scripts": script_inventory,
        }
    )

    output_dir.mkdir(parents=True, exist_ok=False)

    contract_path = output_dir / "cbm_strengthening_contract.json"
    markdown_path = output_dir / "cbm_strengthening_contract.md"
    inventory_path = output_dir / "script_inventory_sha256.json"

    contract_path.write_bytes(contract_bytes)
    markdown_path.write_bytes(markdown_bytes)
    inventory_path.write_bytes(inventory_bytes)

    frozen_files = {
        contract_path.name: sha256_file(contract_path),
        markdown_path.name: sha256_file(markdown_path),
        inventory_path.name: sha256_file(inventory_path),
    }

    manifest = {
        "freeze_name": CONTRACT_DIRNAME,
        "freeze_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script_version": SCRIPT_VERSION,
        "project_root_at_freeze": str(root),
        "python_script_count_snapshotted": len(script_inventory),
        "frozen_files_sha256": frozen_files,
        "contract_sha256": frozen_files[contract_path.name],
        "outcomes_accessed": False,
        "scientific_results_accessed": False,
        "models_fitted": False,
        "metrics_calculated": False,
    }
    manifest_path = output_dir / "freeze_manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest))

    # Write a short immutable checksum file last.
    checksum_text = (
        f"contract_sha256  {manifest['contract_sha256']}\n"
        f"freeze_manifest_sha256  {sha256_file(manifest_path)}\n"
    )
    checksum_path = output_dir / "FREEZE_SHA256.txt"
    checksum_path.write_text(checksum_text, encoding="utf-8", newline="\n")

    print("-" * 112)
    print("Frozen strengthening scope")
    print("-" * 112)
    print("  anchor provenance: YES")
    print("  all-model descriptive safety summary: YES")
    print("  core-only aggregation sensitivity: YES")
    print("  six-regime-balanced aggregation sensitivity: YES")
    print("  prespecified threshold-definition sensitivity: YES")
    print("  descriptive TARGET KM IBS null reference: YES")
    print("  GSE21257/GSE39055 outcome opening: NO")
    print("  original benchmark refitting: NO")
    print("  primary decision reopening: NO")
    print()
    print(f"Python scripts snapshotted: {len(script_inventory)}")
    print(f"Contract SHA-256: {manifest['contract_sha256']}")
    print()
    print("=" * 112)
    print("05h0 CBM strengthening contract freeze: PASS")
    print("=" * 112)
    print(f"Created: {contract_path}")
    print(f"Created: {markdown_path}")
    print(f"Created: {inventory_path}")
    print(f"Created: {manifest_path}")
    print(f"Created: {checksum_path}")


if __name__ == "__main__":
    main()
