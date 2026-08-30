#!/usr/bin/env python3
"""
Paper 6 - 05e3a audit of prespecified mechanism-support rule feasibility.

Why this audit is necessary
---------------------------
The already-completed frozen 05e3 evaluation returned:

H1_HEAD_RETARGET_R5      SUPPORTS
H2_HEAD_CONSTRAINT_R5    DOES_NOT_SUPPORT
H3_SOFT_LEAKAGE_R5       DOES_NOT_SUPPORT
H4_ORACLE_GATE_R5        DOES_NOT_SUPPORT

After seeing those results, a deterministic arithmetic issue is apparent for
H3/H4:

The frozen H3/H4 secondary rule requires an ABSOLUTE catastrophic-negative-
transfer rate reduction >= 0.10 in the primary 29-event R5 cell.

But the control M4_A3_SOFT_LEARNED catastrophic rate in that cell is 0.082.
Because a treatment catastrophic rate cannot be below 0, the maximum attainable
absolute reduction is exactly 0.082. Therefore the >=0.10 secondary component
was structurally infeasible in that realized primary cell.

This does NOT retroactively change the frozen 05e3 machine status.
It only corrects the scientific interpretation of the composite rule.

Important distinctions
----------------------
H3:
- primary Uno-C component PASSED strongly;
- secondary catastrophic-reduction component was STRUCTURALLY INFEASIBLE;
- therefore report the frozen machine status as DOES_NOT_SUPPORT, but state that
  the prespecified composite criterion was not fully assessable because one
  component could not be attained given the realized control event rate.
- Do NOT relabel H3 as SUPPORTS.

H4:
- primary Uno-C component itself FAILED (+0.011... < +0.02);
- secondary catastrophic-reduction component was also structurally infeasible.
- therefore the negative H4 conclusion remains supported by its primary
  component independently of the secondary-rule floor problem.

No model fitting.
No metric recomputation.
No threshold changes.
No A6 reopening.
No human outcomes.
CPU-only.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


SCRIPT_VERSION = (
    "05e3a-audit-mechanism-support-rule-feasibility-v1-no-cli"
)

ROOT = Path(__file__).resolve().parents[1]

E0_DIR = ROOT / "results" / "posthold_controlled_mechanism" / "05e0"
E0_CONTRACT = E0_DIR / "controlled_mechanism_contract.json"
E0_CONTRASTS = E0_DIR / "controlled_mechanism_contrast_registry.tsv"

E3_DIR = ROOT / "results" / "posthold_controlled_mechanism" / "05e3"
E3_EVAL_CONTRACT = E3_DIR / "evaluation_implementation_contract.json"
E3_R5_TABLE = E3_DIR / "R5_event_scaled_mechanism_table.tsv"
E3_CONTRASTS = E3_DIR / "prespecified_mechanism_contrast_summary.tsv"
E3_SUMMARY = E3_DIR / "summary.json"

EXPECTED_E0_CONTRACT_SHA256 = (
    "05cd5bc9180909c4c76f4e9a086a6fa1a2c37faee2f2675008e6e1dd91c581fb"
)
EXPECTED_E3_EVAL_CONTRACT_SHA256 = (
    "ca547a9c6c539d7258e7023c4b6fd12064a604bdb442112fd8e15db126c59c63"
)
EXPECTED_E3_STATUS = (
    "PASS_CONTROLLED_MECHANISM_CONTRAST_EVALUATION_COMPLETE"
)

OUT_DIR = ROOT / "results" / "posthold_controlled_mechanism" / "05e3a"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COMPONENT_AUDIT = OUT_DIR / "mechanism_contrast_component_feasibility.tsv"
ERRATUM_JSON = OUT_DIR / "mechanism_support_interpretation_erratum.json"
SUMMARY_JSON = OUT_DIR / "summary.json"

PRIMARY_EVENTS = 29
PRIMARY_REGIME = "R5_MISLEADING_SOURCE"

H1_DELTA_C = 0.02
H2_DELTA_C = 0.01
H3_DELTA_C = 0.02
H4_DELTA_C = 0.02

H3_REQUIRED_CATASTROPHIC_REDUCTION = 0.10
H4_REQUIRED_CATASTROPHIC_REDUCTION = 0.10


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def require_file(path: Path) -> Path:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Required artifact missing: {path}")
    return path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")


def one_row(
    frame: pd.DataFrame,
    *,
    column: str,
    value: str,
) -> pd.Series:
    part = frame[frame[column].astype(str) == str(value)]
    if len(part) != 1:
        raise RuntimeError(
            f"Expected exactly one row where {column}={value!r}; observed {len(part)}."
        )
    return part.iloc[0]


def main() -> None:
    started = now_utc()

    print("=" * 120)
    print("Paper 6 - 05e3a mechanism support-rule feasibility audit")
    print("=" * 120)
    print(f"Script version: {SCRIPT_VERSION}")
    print()
    print("Scope:")
    print("  05e3 frozen machine statuses changed: NO")
    print("  Model fitting: NO")
    print("  Metric recomputation: NO")
    print("  Threshold changes: NO")
    print("  A6 reopened: NO")
    print("  Human outcomes read: NO")
    print("  GPU execution: NO")
    print()

    for path in [
        E0_CONTRACT,
        E0_CONTRASTS,
        E3_EVAL_CONTRACT,
        E3_R5_TABLE,
        E3_CONTRASTS,
        E3_SUMMARY,
    ]:
        require_file(path)

    if sha256_file(E0_CONTRACT) != EXPECTED_E0_CONTRACT_SHA256:
        raise RuntimeError("05e0 frozen contract SHA256 changed.")

    if sha256_file(E3_EVAL_CONTRACT) != EXPECTED_E3_EVAL_CONTRACT_SHA256:
        raise RuntimeError("05e3 evaluation-contract SHA256 changed.")

    e3_summary = read_json(E3_SUMMARY)
    if clean(e3_summary.get("scientific_status")) != EXPECTED_E3_STATUS:
        raise RuntimeError("05e3 is not in the expected completed PASS state.")

    e0_contrasts = pd.read_csv(E0_CONTRASTS, sep="\t")
    e3_contrasts = pd.read_csv(E3_CONTRASTS, sep="\t")
    r5 = pd.read_csv(E3_R5_TABLE, sep="\t")

    required_ids = {
        "H1_HEAD_RETARGET_R5",
        "H2_HEAD_CONSTRAINT_R5",
        "H3_SOFT_LEAKAGE_R5",
        "H4_ORACLE_GATE_R5",
    }

    if set(e3_contrasts["contrast_id"].astype(str)) != required_ids:
        raise RuntimeError("05e3 H1-H4 contrast set changed.")

    primary_r5 = r5[
        (r5["target_events"].astype(int) == PRIMARY_EVENTS)
        & (r5["transfer_regime"].astype(str) == PRIMARY_REGIME)
    ].copy()

    expected_models = {
        "M1_SOURCE_NETWORK_ZERO_SHOT",
        "M2_A1_SOURCE_CENTERED_HEAD",
        "M3_A1_FREE_HEAD",
        "M4_A3_SOFT_LEARNED",
        "M5_A3_HARDENED_PREDICTION",
        "M6_A3_ORACLE_HARD_GATE",
    }

    if not expected_models.issubset(
        set(primary_r5["model"].astype(str))
    ):
        raise RuntimeError("Primary 29-event R5 model table is incomplete.")

    rows: List[Dict[str, Any]] = []

    # H1 ----------------------------------------------------------------
    h1 = one_row(
        e3_contrasts,
        column="contrast_id",
        value="H1_HEAD_RETARGET_R5",
    )
    h1_primary_pass = bool(
        float(h1["mean_delta_uno_c"]) >= H1_DELTA_C
    )
    h1_10_pass = bool(
        float(h1["mean_delta_uno_c_at_10_events"]) >= 0
    )
    h1_40_pass = bool(
        float(h1["mean_delta_uno_c_at_40_events"]) >= 0
    )

    rows.append({
        "contrast_id": "H1_HEAD_RETARGET_R5",
        "frozen_machine_status": str(h1["frozen_support_status"]),
        "primary_component": "mean_delta_uno_c >= +0.02",
        "primary_observed": float(h1["mean_delta_uno_c"]),
        "primary_threshold": H1_DELTA_C,
        "primary_component_pass": h1_primary_pass,
        "secondary_component": "direction >=0 at 10 and 40 events",
        "secondary_observed": (
            f"10={float(h1['mean_delta_uno_c_at_10_events']):.6f};"
            f"40={float(h1['mean_delta_uno_c_at_40_events']):.6f}"
        ),
        "secondary_threshold": ">=0 at both event budgets",
        "secondary_requirement_structurally_feasible": True,
        "secondary_component_pass": bool(h1_10_pass and h1_40_pass),
        "maximum_attainable_secondary_value": float("nan"),
        "posthoc_interpretation_status": "FORMALLY_SUPPORTS_AS_FROZEN",
    })

    # H2 ----------------------------------------------------------------
    h2 = one_row(
        e3_contrasts,
        column="contrast_id",
        value="H2_HEAD_CONSTRAINT_R5",
    )
    h2_primary_pass = bool(
        float(h2["mean_delta_uno_c"]) >= H2_DELTA_C
    )
    h2_ibs_pass = bool(float(h2["mean_delta_ibs"]) <= 0)
    h2_scale_pass = bool(
        float(h2["mean_delta_test_risk_sd"]) <= 0
    )

    rows.append({
        "contrast_id": "H2_HEAD_CONSTRAINT_R5",
        "frozen_machine_status": str(h2["frozen_support_status"]),
        "primary_component": "mean_delta_uno_c >= +0.01",
        "primary_observed": float(h2["mean_delta_uno_c"]),
        "primary_threshold": H2_DELTA_C,
        "primary_component_pass": h2_primary_pass,
        "secondary_component": "delta_IBS<=0 AND delta_test_risk_SD<=0",
        "secondary_observed": (
            f"dIBS={float(h2['mean_delta_ibs']):.6f};"
            f"dRiskSD={float(h2['mean_delta_test_risk_sd']):.6f}"
        ),
        "secondary_threshold": "<=0 for both",
        "secondary_requirement_structurally_feasible": True,
        "secondary_component_pass": bool(
            h2_ibs_pass and h2_scale_pass
        ),
        "maximum_attainable_secondary_value": float("nan"),
        "posthoc_interpretation_status": (
            "DOES_NOT_SUPPORT_PRIMARY_EFFECT"
            if not h2_primary_pass
            else "MIXED_OR_SUPPORTIVE_SECONDARIES"
        ),
    })

    # Shared M4 catastrophic control floor for H3/H4 --------------------
    m4 = one_row(
        primary_r5,
        column="model",
        value="M4_A3_SOFT_LEARNED",
    )
    m4_cat = float(
        m4["catastrophic_negative_transfer_rate"]
    )

    if not (0 <= m4_cat <= 1):
        raise RuntimeError("M4 catastrophic rate is outside [0,1].")

    # Absolute rate reduction = control - treatment.
    # Treatment rate cannot be <0, so mathematical maximum = control rate.
    max_cat_reduction = m4_cat

    # H3 ----------------------------------------------------------------
    h3 = one_row(
        e3_contrasts,
        column="contrast_id",
        value="H3_SOFT_LEAKAGE_R5",
    )
    h3_primary_pass = bool(
        float(h3["mean_delta_uno_c"]) >= H3_DELTA_C
    )
    h3_secondary_feasible = bool(
        max_cat_reduction
        >= H3_REQUIRED_CATASTROPHIC_REDUCTION
    )
    h3_secondary_pass = bool(
        float(
            h3[
                "catastrophic_rate_reduction_control_minus_treatment"
            ]
        )
        >= H3_REQUIRED_CATASTROPHIC_REDUCTION
    )

    if h3_primary_pass and not h3_secondary_feasible:
        h3_posthoc = (
            "PRIMARY_SUPPORTIVE_COMPOSITE_SECONDARY_STRUCTURALLY_INFEASIBLE"
        )
    elif not h3_primary_pass:
        h3_posthoc = "DOES_NOT_SUPPORT_PRIMARY_EFFECT"
    else:
        h3_posthoc = "COMPOSITE_ASSESSABLE"

    rows.append({
        "contrast_id": "H3_SOFT_LEAKAGE_R5",
        "frozen_machine_status": str(h3["frozen_support_status"]),
        "primary_component": "mean_delta_uno_c >= +0.02",
        "primary_observed": float(h3["mean_delta_uno_c"]),
        "primary_threshold": H3_DELTA_C,
        "primary_component_pass": h3_primary_pass,
        "secondary_component": "absolute catastrophic-rate reduction M4-M5 >=0.10",
        "secondary_observed": float(
            h3[
                "catastrophic_rate_reduction_control_minus_treatment"
            ]
        ),
        "secondary_threshold": H3_REQUIRED_CATASTROPHIC_REDUCTION,
        "secondary_requirement_structurally_feasible": h3_secondary_feasible,
        "secondary_component_pass": h3_secondary_pass,
        "maximum_attainable_secondary_value": max_cat_reduction,
        "posthoc_interpretation_status": h3_posthoc,
    })

    # H4 ----------------------------------------------------------------
    h4 = one_row(
        e3_contrasts,
        column="contrast_id",
        value="H4_ORACLE_GATE_R5",
    )
    h4_primary_pass = bool(
        float(h4["mean_delta_uno_c"]) >= H4_DELTA_C
    )
    h4_secondary_feasible = bool(
        max_cat_reduction
        >= H4_REQUIRED_CATASTROPHIC_REDUCTION
    )
    h4_secondary_pass = bool(
        float(
            h4[
                "catastrophic_rate_reduction_control_minus_treatment"
            ]
        )
        >= H4_REQUIRED_CATASTROPHIC_REDUCTION
    )

    if not h4_primary_pass:
        h4_posthoc = (
            "DOES_NOT_SUPPORT_PRIMARY_EFFECT_SECONDARY_ALSO_STRUCTURALLY_INFEASIBLE"
            if not h4_secondary_feasible
            else "DOES_NOT_SUPPORT_PRIMARY_EFFECT"
        )
    elif not h4_secondary_feasible:
        h4_posthoc = (
            "PRIMARY_SUPPORTIVE_COMPOSITE_SECONDARY_STRUCTURALLY_INFEASIBLE"
        )
    else:
        h4_posthoc = "COMPOSITE_ASSESSABLE"

    rows.append({
        "contrast_id": "H4_ORACLE_GATE_R5",
        "frozen_machine_status": str(h4["frozen_support_status"]),
        "primary_component": "mean_delta_uno_c >= +0.02",
        "primary_observed": float(h4["mean_delta_uno_c"]),
        "primary_threshold": H4_DELTA_C,
        "primary_component_pass": h4_primary_pass,
        "secondary_component": "absolute catastrophic-rate reduction M4-M6 >=0.10",
        "secondary_observed": float(
            h4[
                "catastrophic_rate_reduction_control_minus_treatment"
            ]
        ),
        "secondary_threshold": H4_REQUIRED_CATASTROPHIC_REDUCTION,
        "secondary_requirement_structurally_feasible": h4_secondary_feasible,
        "secondary_component_pass": h4_secondary_pass,
        "maximum_attainable_secondary_value": max_cat_reduction,
        "posthoc_interpretation_status": h4_posthoc,
    })

    audit = pd.DataFrame(rows)
    audit.to_csv(COMPONENT_AUDIT, sep="\t", index=False)

    erratum = {
        "script_version": SCRIPT_VERSION,
        "created_utc": now_utc(),
        "status": "POSTRESULT_SUPPORT_RULE_FEASIBILITY_ERRATUM",
        "05e0_contract_sha256": sha256_file(E0_CONTRACT),
        "05e3_evaluation_contract_sha256": sha256_file(
            E3_EVAL_CONTRACT
        ),
        "05e3_summary_sha256": sha256_file(E3_SUMMARY),
        "frozen_machine_statuses_changed": False,
        "finding": {
            "primary_cell": "29-event R5_MISLEADING_SOURCE",
            "M4_control_catastrophic_rate": m4_cat,
            "maximum_possible_absolute_catastrophic_reduction": (
                max_cat_reduction
            ),
            "H3_required_absolute_reduction": (
                H3_REQUIRED_CATASTROPHIC_REDUCTION
            ),
            "H4_required_absolute_reduction": (
                H4_REQUIRED_CATASTROPHIC_REDUCTION
            ),
            "H3_secondary_requirement_structurally_feasible": (
                h3_secondary_feasible
            ),
            "H4_secondary_requirement_structurally_feasible": (
                h4_secondary_feasible
            ),
        },
        "reporting_correction": {
            "H3": (
                "Keep frozen machine status DOES_NOT_SUPPORT. Do not state that "
                "hardening failed. State that the primary discrimination "
                "component was strongly supportive, whereas the prespecified "
                "composite rule could not be fully assessed because its absolute "
                "catastrophic-reduction requirement (>=0.10) exceeded the "
                "realized control catastrophic rate (0.082), making that "
                "component unattainable."
            ),
            "H4": (
                "Keep DOES_NOT_SUPPORT. The primary discrimination component "
                "itself was below the prespecified +0.02 threshold, so the H4 "
                "negative result does not depend on the infeasible catastrophic "
                "reduction component."
            ),
        },
        "guardrails": {
            "no_retroactive_threshold_change": True,
            "no_H3_relabel_to_SUPPORTS": True,
            "no_A6_reopening": True,
            "no_model_selection": True,
            "no_human_outcomes": True,
        },
    }
    write_json(ERRATUM_JSON, erratum)

    summary = {
        "script_version": SCRIPT_VERSION,
        "status": "PASS",
        "scientific_status": (
            "PASS_POSTRESULT_MECHANISM_SUPPORT_RULE_FEASIBILITY_AUDIT_COMPLETE"
        ),
        "run_started_utc": started,
        "run_finished_utc": now_utc(),
        "frozen_05e3_machine_statuses_changed": False,
        "H1_interpretation": "FORMALLY_SUPPORTS_AS_FROZEN",
        "H2_interpretation": (
            "DOES_NOT_SUPPORT_SOURCE_CENTERING_PENALTY_EFFECT"
        ),
        "H3_interpretation": h3_posthoc,
        "H4_interpretation": h4_posthoc,
        "M4_primary_R5_catastrophic_rate": m4_cat,
        "maximum_attainable_H3_H4_catastrophic_reduction": (
            max_cat_reduction
        ),
        "H3_H4_required_catastrophic_reduction": 0.10,
        "artifact_hashes": {
            "mechanism_contrast_component_feasibility.tsv": sha256_file(
                COMPONENT_AUDIT
            ),
            "mechanism_support_interpretation_erratum.json": sha256_file(
                ERRATUM_JSON
            ),
        },
        "guardrails": {
            "no_threshold_changes": True,
            "no_model_fitting": True,
            "no_model_selection": True,
            "no_A6_reopening": True,
            "no_human_outcomes": True,
        },
    }
    write_json(SUMMARY_JSON, summary)

    print("=" * 120)
    print("05e3a COMPONENT FEASIBILITY AUDIT")
    print("=" * 120)
    print(
        audit[
            [
                "contrast_id",
                "frozen_machine_status",
                "primary_observed",
                "primary_threshold",
                "primary_component_pass",
                "secondary_observed",
                "secondary_threshold",
                "secondary_requirement_structurally_feasible",
                "secondary_component_pass",
                "maximum_attainable_secondary_value",
                "posthoc_interpretation_status",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 120)
    print("05e3a KEY ARITHMETIC")
    print("=" * 120)
    print(
        f"M4 catastrophic rate in primary 29-event R5 cell: {m4_cat:.3f}"
    )
    print(
        f"Maximum mathematically attainable absolute catastrophic-rate reduction: "
        f"{max_cat_reduction:.3f}"
    )
    print(
        f"Frozen H3/H4 required absolute reduction: "
        f"{H3_REQUIRED_CATASTROPHIC_REDUCTION:.3f}"
    )
    print(
        "H3/H4 secondary requirement structurally feasible: "
        f"{h3_secondary_feasible}"
    )

    print()
    print("=" * 120)
    print("05e3a INTERPRETATION SUMMARY")
    print("=" * 120)
    print("H1: FORMALLY SUPPORTS AS FROZEN")
    print(
        "H2: DOES NOT SUPPORT a material benefit from source-centering "
        "the target head; M2 and M3 are effectively equivalent."
    )
    print(
        "H3: frozen status remains DOES_NOT_SUPPORT, but the primary Uno-C "
        "component is strongly supportive and the composite secondary "
        "catastrophic-reduction rule was structurally infeasible."
    )
    print(
        "H4: DOES_NOT_SUPPORT remains interpretable because its primary Uno-C "
        "component failed independently of the secondary-rule infeasibility."
    )
    print()
    print("Frozen 05e3 statuses changed: NO")
    print("A6 reopened: NO")
    print("Human outcomes read: NO")
    print("=" * 120)
    print(
        "05e3a: PASS_POSTRESULT_MECHANISM_SUPPORT_RULE_FEASIBILITY_AUDIT_COMPLETE"
    )
    print("=" * 120)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 120, file=sys.stderr)
        print("05e3a support-rule feasibility audit: FAIL", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print("=" * 120, file=sys.stderr)
        raise
