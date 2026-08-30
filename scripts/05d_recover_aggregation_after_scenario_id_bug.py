#!/usr/bin/env python3
"""
Paper 6 - recover 05d aggregation after scenario_id index-drop implementation bug.

TECHNICAL RECOVERY ONLY.

Background
----------
The original frozen 05d v1 successfully computed and checkpointed synthetic
metrics for all 180 scenarios / 21,600 replicates, then failed before any
aggregate architecture-selection result was produced:

    scenario_by_id = scenarios.set_index("scenario_id")
    ...
    aggregate_scenario(scenario_by_id.loc[scenario_id], ...)

pandas removes scenario_id from the selected Series by default, while the
already-frozen aggregate_scenario() function expects scenario["scenario_id"].

The scientific fix is therefore only:
    scenarios.set_index("scenario_id", drop=False)

Critical provenance rule
------------------------
DO NOT modify or replace the original 05d metric contract. Its metrics were
already computed under the original frozen v1 script/contract.

This recovery script:
1. verifies the exact existing frozen 05d metric contract;
2. verifies that the original 05d script still matches the script SHA stored in
   that metric contract;
3. verifies all 180 metric checkpoints and their manifests/hashes;
4. verifies that no final architecture/A5 decision existed before recovery;
5. imports and reuses the original 05d v1 aggregation/selection functions;
6. applies ONLY the drop=False identity repair;
7. writes the originally intended aggregate outputs and a technical recovery
   provenance record.

No synthetic metric is recomputed.
No model is fit.
No threshold changes.
No real data are read.
No GPU.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


SCRIPT_VERSION = "05d-recover-aggregation-scenario-id-keyfix-v1-no-cli"

ROOT = Path(__file__).resolve().parents[1]

ORIGINAL_05D_SCRIPT = (
    ROOT / "scripts" / "05d_aggregate_negative_transfer_phase_diagram.py"
)

A_DIR = ROOT / "results" / "simulation_contract" / "05a"
A_CONTRACT = A_DIR / "negative_transfer_simulation_contract.json"
A_SCENARIOS = A_DIR / "simulation_scenario_registry.tsv"

C_DIR = ROOT / "results" / "simulation_model_matrix" / "05c"
C_MANIFEST = C_DIR / "scenario_output_manifest.tsv"

C1_DIR = ROOT / "results" / "simulation_extension_contract" / "05c1"
C1_CONTRACT = C1_DIR / "A5_ECHRR_branch_contract.json"

C1A_DIR = ROOT / "results" / "simulation_extension_contract" / "05c1a"
C1A_CONTRACT = C1A_DIR / "A5_operational_branch_definitions.json"
C1A_CONTRASTS = C1A_DIR / "A5_shift_contrast_registry.tsv"

OUT_DIR = ROOT / "results" / "simulation_phase_diagram" / "05d"
METRIC_CONTRACT = OUT_DIR / "metric_implementation_contract.json"
SCENARIO_METRIC_MANIFEST = OUT_DIR / "scenario_metric_manifest.tsv"
CHECKPOINT_DIR = OUT_DIR / "scenario_metrics"
CHECKPOINT_MANIFEST_DIR = OUT_DIR / "scenario_manifests"

RECOVERY_PROVENANCE = OUT_DIR / "technical_aggregation_recovery.json"

# The exact metric-contract SHA printed by the already-executed frozen 05d run
# before any scenario metric was read.
EXPECTED_METRIC_CONTRACT_SHA256 = (
    "9e575f076a92e7ce3002df7b1b0efc4a356371ceaa0b64cfea9de8cfb804a113"
)

EXPECTED_ORIGINAL_SCRIPT_VERSION = (
    "05d-aggregate-negative-transfer-phase-diagram-v1-no-cli"
)
EXPECTED_METRIC_IMPLEMENTATION_VERSION = (
    "paper6-05d-synthetic-evaluation-v1"
)

EXPECTED_SCENARIOS = 180
EXPECTED_REPLICATES = 21600


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_file(path: Path) -> Path:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Required artifact missing: {path}")
    return path


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


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")

    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(name)
    sys.modules[name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
        raise

    return module


def finite_median(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else float("nan")


def main() -> None:
    started = now_utc()

    print("=" * 120)
    print("Paper 6 - recover 05d aggregation after scenario_id identity bug")
    print("=" * 120)
    print(f"Recovery script version: {SCRIPT_VERSION}")
    print()
    print("Recovery contract:")
    print("  05d synthetic metrics recomputed: NO")
    print("  05d metric contract changed: NO")
    print("  Model fitting / retuning: NO")
    print("  Selection thresholds changed: NO")
    print("  Real DOG2/human values read: NO")
    print("  GPU execution: NO")
    print("  Technical change: preserve scenario_id column during indexed aggregation ONLY")
    print()

    for path in [
        ORIGINAL_05D_SCRIPT,
        A_CONTRACT,
        A_SCENARIOS,
        C_MANIFEST,
        C1_CONTRACT,
        C1A_CONTRACT,
        C1A_CONTRASTS,
        METRIC_CONTRACT,
        SCENARIO_METRIC_MANIFEST,
    ]:
        require_file(path)

    # ------------------------------------------------------------------
    # Frozen metric-contract verification.
    # ------------------------------------------------------------------
    observed_metric_contract_hash = sha256_file(METRIC_CONTRACT)

    if observed_metric_contract_hash != EXPECTED_METRIC_CONTRACT_SHA256:
        raise RuntimeError(
            "Existing 05d metric-contract SHA256 differs from the exact hash "
            "printed before the original 05d scenario read. "
            f"Observed={observed_metric_contract_hash}"
        )

    metric_contract = read_json(METRIC_CONTRACT)

    if (
        str(metric_contract.get("script_version"))
        != EXPECTED_ORIGINAL_SCRIPT_VERSION
    ):
        raise RuntimeError("Unexpected original 05d script version in metric contract.")

    if (
        str(metric_contract.get("metric_implementation_version"))
        != EXPECTED_METRIC_IMPLEMENTATION_VERSION
    ):
        raise RuntimeError("Unexpected 05d metric implementation version.")

    expected_original_script_hash = str(
        metric_contract.get("script_sha256")
    )
    observed_original_script_hash = sha256_file(
        ORIGINAL_05D_SCRIPT
    )

    if observed_original_script_hash != expected_original_script_hash:
        raise RuntimeError(
            "Original 05d script was modified after the frozen metric contract "
            "was created. Restore the exact original v1 before recovery."
        )

    # Import the exact original frozen implementation.
    m05d = load_module(
        ORIGINAL_05D_SCRIPT,
        "paper6_05d_frozen_v1_for_aggregation_recovery",
    )

    if str(m05d.SCRIPT_VERSION) != EXPECTED_ORIGINAL_SCRIPT_VERSION:
        raise RuntimeError("Imported 05d implementation identity mismatch.")

    # ------------------------------------------------------------------
    # Prove that final scientific decisions did NOT exist before recovery.
    # ------------------------------------------------------------------
    forbidden_preexisting = [
        m05d.ARCHITECTURE_DECISION,
        m05d.A5_DECISION,
        m05d.SUMMARY_JSON,
    ]

    existing_final = [
        p for p in forbidden_preexisting
        if p.exists() and p.is_file()
    ]

    if existing_final:
        raise RuntimeError(
            "Final 05d scientific decision artifact already exists; do not "
            "overwrite it with this recovery script. Existing: "
            + ", ".join(str(p) for p in existing_final)
        )

    # ------------------------------------------------------------------
    # Verify all scenario/checkpoint identities.
    # ------------------------------------------------------------------
    scenarios = pd.read_csv(A_SCENARIOS, sep="\t")
    source_manifest = pd.read_csv(C_MANIFEST, sep="\t")
    metric_manifest = pd.read_csv(
        SCENARIO_METRIC_MANIFEST,
        sep="\t",
    )
    contrast_registry = pd.read_csv(
        C1A_CONTRASTS,
        sep="\t",
    )

    if len(scenarios) != EXPECTED_SCENARIOS:
        raise RuntimeError(
            f"Scenario registry rows={len(scenarios)}, expected={EXPECTED_SCENARIOS}."
        )
    if len(source_manifest) != EXPECTED_SCENARIOS:
        raise RuntimeError("05c source manifest does not contain 180 scenarios.")
    if len(metric_manifest) != EXPECTED_SCENARIOS:
        raise RuntimeError(
            "05d metric manifest does not contain all 180 completed scenarios."
        )

    if int(scenarios["replicates"].astype(int).sum()) != EXPECTED_REPLICATES:
        raise RuntimeError("Frozen scenario replicate total changed.")
    if int(metric_manifest["replicates"].astype(int).sum()) != EXPECTED_REPLICATES:
        raise RuntimeError("05d metric-manifest replicate total mismatch.")

    expected_ids = set(
        scenarios["scenario_id"].astype(str)
    )
    metric_ids = set(
        metric_manifest["scenario_id"].astype(str)
    )

    if metric_ids != expected_ids:
        raise RuntimeError("05d completed metric scenario IDs do not match 05a.")

    source_by_id = {
        str(row.scenario_id): row
        for row in source_manifest.itertuples(index=False)
    }

    verified_rows: List[Dict[str, Any]] = []

    for i, row in enumerate(
        metric_manifest.itertuples(index=False),
        start=1,
    ):
        scenario_id = str(row.scenario_id)
        metric_path = ROOT / str(row.metric_output_path)
        require_file(metric_path)

        metric_hash = sha256_file(metric_path)
        if metric_hash != str(row.metric_output_sha256):
            raise RuntimeError(
                f"{scenario_id}: metric checkpoint hash differs from metric manifest."
            )

        checkpoint_manifest_path = (
            CHECKPOINT_MANIFEST_DIR / f"{scenario_id}.json"
        )
        require_file(checkpoint_manifest_path)
        checkpoint_manifest = read_json(
            checkpoint_manifest_path
        )

        if str(checkpoint_manifest.get("status")) != "PASS":
            raise RuntimeError(
                f"{scenario_id}: checkpoint manifest is not PASS."
            )

        if (
            str(checkpoint_manifest.get("metric_contract_sha256"))
            != observed_metric_contract_hash
        ):
            raise RuntimeError(
                f"{scenario_id}: checkpoint was not computed under the frozen "
                "05d metric contract."
            )

        if (
            str(checkpoint_manifest.get("metric_output_sha256"))
            != metric_hash
        ):
            raise RuntimeError(
                f"{scenario_id}: checkpoint manifest metric hash mismatch."
            )

        source_row = source_by_id.get(scenario_id)
        if source_row is None:
            raise RuntimeError(
                f"{scenario_id}: no 05c source manifest row."
            )

        source_path = ROOT / str(source_row.output_path)
        require_file(source_path)
        source_hash = sha256_file(source_path)

        if source_hash != str(source_row.output_sha256):
            raise RuntimeError(
                f"{scenario_id}: 05c source output hash mismatch."
            )

        if (
            str(checkpoint_manifest.get("source_output_sha256"))
            != source_hash
        ):
            raise RuntimeError(
                f"{scenario_id}: checkpoint/source provenance mismatch."
            )

        verified_rows.append(
            {
                "scenario_id": scenario_id,
                "metric_output_sha256": metric_hash,
                "source_output_sha256": source_hash,
            }
        )

        if i % 30 == 0 or i == EXPECTED_SCENARIOS:
            print(
                f"  verified completed 05d checkpoints: "
                f"{i}/{EXPECTED_SCENARIOS}"
            )

    # ------------------------------------------------------------------
    # Write recovery provenance BEFORE reading aggregate metric values.
    # ------------------------------------------------------------------
    recovery_preaggregate = {
        "recovery_script_version": SCRIPT_VERSION,
        "status": "RECOVERY_FROZEN_BEFORE_AGGREGATE_METRIC_READ",
        "created_utc": now_utc(),
        "failure": {
            "type": "KeyError",
            "key": "scenario_id",
            "stage": "post-checkpoint aggregation",
            "scientific_metric_computation_completed_before_failure": True,
            "completed_scenarios": EXPECTED_SCENARIOS,
            "completed_replicates": EXPECTED_REPLICATES,
        },
        "root_cause": (
            "pandas set_index('scenario_id') dropped scenario_id from selected "
            "Series while frozen aggregate_scenario() expected that field"
        ),
        "technical_fix": (
            "scenarios.set_index('scenario_id', drop=False) for aggregation lookup"
        ),
        "scientific_changes": False,
        "metric_recomputation": False,
        "threshold_changes": False,
        "model_changes": False,
        "original_metric_contract_sha256": observed_metric_contract_hash,
        "original_05d_script_sha256": observed_original_script_hash,
        "verified_scenario_checkpoints": EXPECTED_SCENARIOS,
        "verified_replicates": EXPECTED_REPLICATES,
        "final_selection_artifacts_present_before_recovery": False,
    }
    write_json(
        RECOVERY_PROVENANCE,
        recovery_preaggregate,
    )

    recovery_hash_before_results = sha256_file(
        RECOVERY_PROVENANCE
    )

    print()
    print("Checkpoint/provenance verification: PASS")
    print(
        f"Recovery provenance SHA256 before aggregate read: "
        f"{recovery_hash_before_results}"
    )
    print()

    # ------------------------------------------------------------------
    # Corrected aggregation identity lookup.
    # THIS IS THE ONLY IMPLEMENTATION REPAIR.
    # ------------------------------------------------------------------
    scenario_rows: List[Dict[str, Any]] = []
    common_ibs_rows: List[Dict[str, Any]] = []

    # BUGFIX: preserve scenario_id as a Series field.
    scenario_by_id = scenarios.set_index(
        "scenario_id",
        drop=False,
    )

    for row in metric_manifest.itertuples(index=False):
        scenario_id = str(row.scenario_id)
        metric_path = ROOT / str(row.metric_output_path)

        rows, common = m05d.aggregate_scenario(
            scenario_by_id.loc[scenario_id],
            metric_path,
        )
        scenario_rows.extend(rows)
        common_ibs_rows.append(common)

    scenario_summary = pd.DataFrame(scenario_rows)
    common_ibs = pd.DataFrame(common_ibs_rows)

    if len(scenario_summary) != EXPECTED_SCENARIOS * len(
        m05d.ALL_MODELS
    ):
        raise RuntimeError(
            "Recovered scenario/model summary has unexpected row count."
        )

    scenario_summary.to_csv(
        m05d.SCENARIO_SUMMARY,
        sep="\t",
        index=False,
    )

    # ------------------------------------------------------------------
    # Exact original v1 downstream aggregation logic.
    # ------------------------------------------------------------------
    model_rows = []

    for model in m05d.ALL_MODELS:
        part = scenario_summary[
            scenario_summary["model"] == model
        ]

        model_rows.append(
            {
                "model": model,
                "n_scenarios": len(part),
                "equal_scenario_mean_uno_c": (
                    m05d.equal_scenario_mean(
                        part,
                        "mean_uno_c",
                    )
                ),
                "equal_scenario_mean_delta_c_vs_B0": (
                    m05d.equal_scenario_mean(
                        part,
                        "mean_delta_c_vs_B0",
                    )
                ),
                "equal_scenario_negative_transfer_rate": (
                    m05d.equal_scenario_mean(
                        part,
                        "negative_transfer_rate",
                    )
                ),
                "equal_scenario_catastrophic_rate": (
                    m05d.equal_scenario_mean(
                        part,
                        "catastrophic_negative_transfer_rate",
                    )
                ),
                "equal_scenario_mean_IBS": (
                    m05d.equal_scenario_mean(
                        part,
                        "mean_ibs",
                    )
                ),
                "median_scenario_calibration_slope": (
                    m05d.finite_median(
                        pd.to_numeric(
                            part["median_calibration_slope"],
                            errors="coerce",
                        ).to_numpy(dtype=float)
                    )
                ),
                "median_scenario_abs_calibration_intercept": (
                    m05d.finite_median(
                        pd.to_numeric(
                            part["median_abs_calibration_intercept"],
                            errors="coerce",
                        ).to_numpy(dtype=float)
                    )
                ),
                "equal_scenario_module_recovery_AUROC": (
                    m05d.equal_scenario_mean(
                        part,
                        "mean_module_recovery_auc",
                    )
                ),
                "equal_scenario_false_borrow_rate": (
                    m05d.equal_scenario_mean(
                        part,
                        "mean_false_borrow_rate",
                    )
                ),
            }
        )

    model_summary = pd.DataFrame(model_rows)
    model_summary.to_csv(
        m05d.MODEL_SUMMARY,
        sep="\t",
        index=False,
    )

    # Event-scaled learning curves from CORE only.
    core = scenario_summary[
        scenario_summary["family"] == "CORE_PHASE_DIAGRAM"
    ].copy()

    learning = (
        core.groupby(
            ["target_events", "model"],
            as_index=False,
        )
        .agg(
            mean_uno_c=("mean_uno_c", "mean"),
            mean_delta_c_vs_B0=(
                "mean_delta_c_vs_B0",
                "mean",
            ),
            negative_transfer_rate=(
                "negative_transfer_rate",
                "mean",
            ),
            catastrophic_negative_transfer_rate=(
                "catastrophic_negative_transfer_rate",
                "mean",
            ),
            mean_ibs=("mean_ibs", "mean"),
        )
        .sort_values(["target_events", "model"])
    )
    learning.to_csv(
        m05d.LEARNING_CURVES,
        sep="\t",
        index=False,
    )

    # A3 ablations.
    abl = model_summary[
        model_summary["model"].isin(
            ["A3"] + m05d.ABLATIONS
        )
    ].copy()
    abl.to_csv(
        m05d.ABLATION_SUMMARY,
        sep="\t",
        index=False,
    )

    # A3/gate module recovery.
    gate_rows = []

    for (model, regime), part in scenario_summary[
        scenario_summary["model"].isin(
            list(m05d.GATE_KEYS)
        )
    ].groupby(["model", "transfer_regime"]):
        gate_rows.append(
            {
                "model": model,
                "transfer_regime": regime,
                "n_scenarios": len(part),
                "mean_module_recovery_AUROC": (
                    m05d.equal_scenario_mean(
                        part,
                        "mean_module_recovery_auc",
                    )
                ),
                "mean_false_borrow_rate": (
                    m05d.equal_scenario_mean(
                        part,
                        "mean_false_borrow_rate",
                    )
                ),
                "mean_gate_true": (
                    m05d.equal_scenario_mean(
                        part,
                        "mean_gate_true",
                    )
                ),
                "mean_gate_false": (
                    m05d.equal_scenario_mean(
                        part,
                        "mean_gate_false",
                    )
                ),
            }
        )

    module_summary = pd.DataFrame(gate_rows)
    module_summary.to_csv(
        m05d.MODULE_SUMMARY,
        sep="\t",
        index=False,
    )

    # Frozen A2/A3 selection.
    candidate_summary = m05d.build_candidate_summary(
        scenario_summary,
        common_ibs,
    )
    candidate_summary.to_csv(
        m05d.CANDIDATE_SUMMARY,
        sep="\t",
        index=False,
    )

    architecture_decision = m05d.select_architecture(
        candidate_summary
    )
    architecture_decision.update(
        {
            "created_utc": now_utc(),
            "05a_contract_sha256": sha256_file(
                A_CONTRACT
            ),
            "05d_metric_contract_sha256": (
                observed_metric_contract_hash
            ),
            "candidate_summary_sha256": sha256_file(
                m05d.CANDIDATE_SUMMARY
            ),
            "GSE16091_used_for_selection": False,
            "reserved_human_outcomes_read": False,
            "technical_aggregation_recovery": True,
            "technical_recovery_provenance_sha256": (
                recovery_hash_before_results
            ),
        }
    )
    write_json(
        m05d.ARCHITECTURE_DECISION,
        architecture_decision,
    )

    # Frozen A5 branch decision.
    a5_shift = m05d.compute_a5_shift_diagnostic(
        scenario_summary,
        contrast_registry,
    )
    a5_shift.to_csv(
        m05d.A5_SHIFT_DIAGNOSTIC,
        sep="\t",
        index=False,
    )

    a5_decision = m05d.decide_a5_branch(
        a5_shift,
        candidate_summary,
        architecture_decision,
    )
    a5_decision.update(
        {
            "created_utc": now_utc(),
            "05c1_contract_sha256": sha256_file(
                C1_CONTRACT
            ),
            "05c1a_contract_sha256": sha256_file(
                C1A_CONTRACT
            ),
            "A5_shift_diagnostic_sha256": sha256_file(
                m05d.A5_SHIFT_DIAGNOSTIC
            ),
            "GSE16091_allowed_for_A5": False,
            "technical_aggregation_recovery": True,
        }
    )
    write_json(
        m05d.A5_DECISION,
        a5_decision,
    )

    # ------------------------------------------------------------------
    # Final scientific status. Scientific HOLD is NOT a technical FAIL.
    # ------------------------------------------------------------------
    selected = architecture_decision.get(
        "selected_architecture"
    )

    if architecture_decision["status"] != "PASS":
        scientific_status = architecture_decision[
            "decision_rule"
        ]
        overall_status = "HOLD"
    elif a5_decision["status"] != "PASS":
        scientific_status = (
            "HOLD_A5_BRANCH_TECHNICAL_METRIC_UNRESOLVED"
        )
        overall_status = "HOLD"
    else:
        scientific_status = (
            f"PASS_FROZEN_AI_SELECTED_{selected}_"
            f"{a5_decision['branch_decision']}"
        )
        overall_status = "PASS"

    summary = {
        "script_version": m05d.SCRIPT_VERSION,
        "metric_implementation_version": (
            m05d.METRIC_IMPLEMENTATION_VERSION
        ),
        "aggregation_recovery_script_version": (
            SCRIPT_VERSION
        ),
        "status": overall_status,
        "scientific_status": scientific_status,
        "run_started_utc": started,
        "run_finished_utc": now_utc(),
        "n_scenarios": EXPECTED_SCENARIOS,
        "total_replicates": EXPECTED_REPLICATES,
        "selected_architecture": selected,
        "architecture_decision_rule": (
            architecture_decision["decision_rule"]
        ),
        "A5_branch_decision": a5_decision[
            "branch_decision"
        ],
        "real_data_values_read": False,
        "reserved_human_outcomes_read": False,
        "GSE16091_used_for_selection": False,
        "model_fitting": False,
        "GPU_execution": False,
        "scenario_metric_checkpoints_reused_this_run": (
            EXPECTED_SCENARIOS
        ),
        "scenario_metric_checkpoints_computed_this_run": 0,
        "technical_aggregation_recovery": {
            "applied": True,
            "scientific_metric_changes": False,
            "original_metric_contract_sha256": (
                observed_metric_contract_hash
            ),
            "recovery_provenance_sha256_before_results": (
                recovery_hash_before_results
            ),
        },
        "final_artifact_hashes": {
            "metric_implementation_contract_json": sha256_file(
                METRIC_CONTRACT
            ),
            "scenario_metric_manifest_tsv": sha256_file(
                SCENARIO_METRIC_MANIFEST
            ),
            "scenario_model_metric_summary_tsv": sha256_file(
                m05d.SCENARIO_SUMMARY
            ),
            "model_aggregate_summary_tsv": sha256_file(
                m05d.MODEL_SUMMARY
            ),
            "A2_A3_frozen_selection_summary_tsv": sha256_file(
                m05d.CANDIDATE_SUMMARY
            ),
            "event_scaled_learning_curves_tsv": sha256_file(
                m05d.LEARNING_CURVES
            ),
            "A3_ablation_summary_tsv": sha256_file(
                m05d.ABLATION_SUMMARY
            ),
            "A3_module_recovery_summary_tsv": sha256_file(
                m05d.MODULE_SUMMARY
            ),
            "A5_shift_trigger_diagnostics_tsv": sha256_file(
                m05d.A5_SHIFT_DIAGNOSTIC
            ),
            "frozen_architecture_selection_json": sha256_file(
                m05d.ARCHITECTURE_DECISION
            ),
            "A5_branch_decision_json": sha256_file(
                m05d.A5_DECISION
            ),
            "technical_aggregation_recovery_json": sha256_file(
                RECOVERY_PROVENANCE
            ),
        },
        "next_if_PASS": (
            "Proceed according to the already-frozen selected-architecture/A5 "
            "branch logic; do not reinterpret the technical aggregation bug as "
            "a scientific amendment."
        ),
    }
    write_json(
        m05d.SUMMARY_JSON,
        summary,
    )

    # Update recovery record AFTER scientific result only to append final artifact
    # identities. The pre-result SHA is preserved separately in every decision.
    recovery_final = read_json(
        RECOVERY_PROVENANCE
    )
    recovery_final.update(
        {
            "status": "PASS_TECHNICAL_AGGREGATION_RECOVERY_COMPLETE",
            "run_finished_utc": now_utc(),
            "scientific_status": scientific_status,
            "selected_architecture": selected,
            "A5_branch_decision": a5_decision[
                "branch_decision"
            ],
            "preaggregate_recovery_sha256": (
                recovery_hash_before_results
            ),
        }
    )
    write_json(
        RECOVERY_PROVENANCE,
        recovery_final,
    )

    print()
    print("=" * 120)
    print("05d FROZEN A2-vs-A3 SELECTION [RECOVERED AGGREGATION]")
    print("=" * 120)
    print(
        candidate_summary[
            [
                "model",
                "aggregate_negative_transfer_rate",
                "catastrophic_rate_misleading",
                "catastrophic_rate_severe_shift",
                "module_recovery_AUROC_partial",
                "false_borrow_rate_misleading",
                "equal_scenario_mean_uno_c",
                "common_A2_A3_equal_scenario_mean_IBS",
                "overall_protection_pass",
            ]
        ].to_string(index=False)
    )
    print()
    print(
        f"Selected architecture: "
        f"{architecture_decision.get('selected_architecture')}"
    )
    print(
        f"Selection rule: "
        f"{architecture_decision.get('decision_rule')}"
    )

    print()
    print("=" * 120)
    print("05d PREREGISTERED A5 BRANCH DECISION")
    print("=" * 120)
    print(a5_shift.to_string(index=False))
    print()
    print(
        f"A5 branch: {a5_decision['branch_decision']}"
    )
    print(f"Reason: {a5_decision['reason']}")

    print()
    print("=" * 120)
    print("05d NEGATIVE-TRANSFER PHASE DIAGRAM SUMMARY")
    print("=" * 120)
    print(f"Status: {overall_status}")
    print(f"Scientific status: {scientific_status}")
    print(f"Selected AI: {selected}")
    print(f"A5: {a5_decision['branch_decision']}")
    print("Metric checkpoints recomputed during recovery: NO")
    print("Real human outcomes read: NO")
    print("Model fitting during recovery: NO")
    print("GPU execution: NO")
    print("=" * 120)

    # Deliberately do NOT raise for a scientific HOLD. A HOLD is a valid
    # scientific outcome, not a technical execution failure.


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 120, file=sys.stderr)
        print(
            "05d technical aggregation recovery: FAIL",
            file=sys.stderr,
        )
        print(
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        print("=" * 120, file=sys.stderr)
        raise
