#!/usr/bin/env python3
"""
Paper 6 - freeze conservative handling of observed 04c IBS support failure.

Observed blocking defect
------------------------
The frozen 04c v1 execution opened GSE16091 and then failed during the
prespecified B0/B4 IBS calculation because one outer test fold retained only
1 supported evaluation time after the frozen train/test support restriction,
whereas 04b required at least 5.

This amendment is deliberately conservative and is frozen AFTER GSE16091
outcome access. It therefore MUST NOT create a new favorable route.

It changes NO:
- target cohort;
- samples;
- features/modules;
- source model;
- human model family;
- CV splits/repeats/seeds;
- tuning grids;
- Uno-C calculation;
- bootstrap count;
- premise delta-C thresholds;
- negative premise rule;
- reserved-human firewall.

Frozen failure handling
-----------------------
1. Keep the original 04b IBS grid construction unchanged.
2. Never substitute an alternative IBS grid/horizon.
3. If an outer fold has <5 supported IBS grid points:
   - record IBS for that fold as unavailable;
   - continue the already-frozen discrimination analysis;
   - do NOT use a feasible-fold-only IBS estimate to establish SUPPORTS.
4. If ANY primary B0/B4 fold has unavailable IBS:
   - SUPPORTS_CANINE_ADDED_VALUE is disabled.
5. ARGUES_AGAINST_CANINE_ADDED_VALUE remains evaluable exactly as frozen,
   because its 02i/04b rule depends only on paired delta Uno C and its
   bootstrap probability, not on IBS.
6. Otherwise the premise state is INCONCLUSIVE_NEUTRAL with an explicit
   IBS_FEASIBILITY_LIMITATION.
7. Feasible-fold IBS summaries may be reported DESCRIPTIVELY ONLY.
8. Existing v1 repeat checkpoints may be reused only when their original
   04b protocol SHA256 and artifact hashes validate.

This is an implementation-feasibility amendment, not a scientific threshold
amendment.

No CLI arguments.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


SCRIPT_VERSION = "04c0-freeze-conservative-ibs-feasibility-amendment-v1-no-cli"

ROOT = Path(__file__).resolve().parents[1]

B4_DIR = ROOT / "results" / "human_premise_protocol" / "04b"
B4_PROTOCOL = B4_DIR / "classical_premise_test_protocol.json"
B4_SUMMARY = B4_DIR / "summary.json"

C4_DIR = ROOT / "results" / "human_premise" / "04c"
C4_RAW_LOCK = C4_DIR / "raw" / "GSE16091_series_matrix_lock.json"
C4_OUTCOME_AUDIT = C4_DIR / "gse16091_os_identity_audit.tsv"
C4_PRIMARY_RESULT = C4_DIR / "primary_premise_result.json"
C4_PRIMARY_CKPT = C4_DIR / "checkpoints" / "primary"

V1_SCRIPT = ROOT / "scripts" / "04c_run_classical_premise_test.py"

OUT_DIR = ROOT / "results" / "human_premise_protocol" / "04c0_ibs_amendment"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AMENDMENT_JSON = OUT_DIR / "ibs_feasibility_amendment.json"
CHECKPOINT_AUDIT = OUT_DIR / "existing_primary_checkpoint_audit.tsv"
SUMMARY_JSON = OUT_DIR / "summary.json"

EXPECTED_TARGET = "GSE16091"
EXPECTED_MIN_IBS_POINTS = 5


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
        raise FileNotFoundError(f"Required artifact missing: {path}")
    return path


def main() -> None:
    started = now_utc()

    print("=" * 120)
    print("Paper 6 - freeze conservative handling of observed 04c IBS support failure")
    print("=" * 120)
    print(f"Script version: {SCRIPT_VERSION}")
    print()
    print("Contract:")
    print("  Human outcomes already opened before this amendment: YES")
    print("  New model/tuning/feature selection: NO")
    print("  Uno-C protocol changed: NO")
    print("  IBS grid construction changed: NO")
    print("  SUPPORTS route made easier: NO")
    print("  SUPPORTS disabled if any primary IBS fold is infeasible: YES")
    print("  Negative premise rule changed: NO")
    print()

    for path in [
        B4_PROTOCOL,
        B4_SUMMARY,
        C4_RAW_LOCK,
        C4_OUTCOME_AUDIT,
        V1_SCRIPT,
    ]:
        require_file(path)

    protocol = read_json(B4_PROTOCOL)
    protocol_summary = read_json(B4_SUMMARY)
    raw_lock = read_json(C4_RAW_LOCK)

    if clean(protocol.get("status")) != "PASS":
        raise RuntimeError("04b protocol is not PASS.")
    if clean(protocol_summary.get("scientific_status")) != (
        "PASS_CLASSICAL_PREMISE_TEST_IMPLEMENTATION_FROZEN"
    ):
        raise RuntimeError("04b protocol is not in frozen PASS state.")

    protocol_hash = sha256_file(B4_PROTOCOL)
    expected_protocol_hash = clean(
        (protocol_summary.get("final_artifact_hashes") or {}).get(
            "classical_premise_test_protocol_json"
        )
    )
    if protocol_hash != expected_protocol_hash:
        raise RuntimeError("04b protocol hash verification failed.")

    if clean(raw_lock.get("accession")) != EXPECTED_TARGET:
        raise RuntimeError("04c raw lock is not GSE16091.")
    if clean(raw_lock.get("status")) != "PASS_FIRST_HUMAN_PREMISE_OUTCOME_SNAPSHOT":
        raise RuntimeError("04c GSE16091 raw snapshot is not in expected locked state.")

    ibs_contract = protocol.get("IBS") or {}
    if int(ibs_contract.get("minimum_supported_grid_points")) != EXPECTED_MIN_IBS_POINTS:
        raise RuntimeError("04b frozen minimum IBS grid-point requirement changed.")
    if not bool(ibs_contract.get("required_for_primary_premise_decision")):
        raise RuntimeError("04b no longer marks IBS as required for primary premise decision.")

    # This amendment is only valid because v1 did NOT materialize a primary result.
    if C4_PRIMARY_RESULT.exists():
        raise RuntimeError(
            "04c primary_premise_result.json already exists; "
            "this amendment must not overwrite a materialized primary premise result."
        )

    # Audit reusable completed repeat checkpoints without reading their performance values.
    checkpoint_rows: List[Dict[str, Any]] = []

    for repeat in range(20):
        manifest_path = C4_PRIMARY_CKPT / f"repeat_{repeat:02d}.json"

        if not manifest_path.exists():
            checkpoint_rows.append(
                {
                    "repeat": repeat,
                    "manifest_exists": False,
                    "protocol_sha256_match": False,
                    "all_artifact_hashes_match": False,
                    "reusable": False,
                }
            )
            continue

        manifest = read_json(manifest_path)
        protocol_match = clean(manifest.get("protocol_sha256")) == protocol_hash
        artifact_hashes = manifest.get("artifact_hashes") or {}

        all_match = True
        for filename, expected_hash in artifact_hashes.items():
            path = C4_PRIMARY_CKPT / filename
            if (
                not path.exists()
                or len(clean(expected_hash)) != 64
                or sha256_file(path) != clean(expected_hash)
            ):
                all_match = False
                break

        reusable = (
            clean(manifest.get("status")) == "PASS"
            and protocol_match
            and all_match
        )

        checkpoint_rows.append(
            {
                "repeat": repeat,
                "manifest_exists": True,
                "protocol_sha256_match": protocol_match,
                "all_artifact_hashes_match": all_match,
                "reusable": reusable,
            }
        )

    checkpoint_audit = pd.DataFrame(checkpoint_rows)
    checkpoint_audit.to_csv(CHECKPOINT_AUDIT, sep="\t", index=False)

    n_reusable = int(checkpoint_audit["reusable"].sum())

    amendment = {
        "script_version": SCRIPT_VERSION,
        "status": "PASS",
        "scientific_status": (
            "PASS_POSTOUTCOME_TECHNICAL_AMENDMENT_CONSERVATIVE_ONLY"
        ),
        "created_utc": now_utc(),
        "observed_blocking_defect": {
            "stage": "04c primary B0/B4 repeated CV",
            "component": "IBS",
            "error": "Only 1 supported IBS points; minimum=5.",
            "interpretation": (
                "The prespecified fold-specific IBS horizon was not estimable "
                "with the required minimum time-grid support in at least one "
                "small held-out GSE16091 fold."
            ),
        },
        "postoutcome_amendment": True,
        "bias_direction_guardrail": (
            "The amendment can only prevent SUPPORTS_CANINE_ADDED_VALUE; "
            "it cannot create or ease a positive premise classification."
        ),
        "unchanged": {
            "04b_protocol_sha256": protocol_hash,
            "target": "GSE16091",
            "features_modules": True,
            "source_model": True,
            "model_families": True,
            "outer_cv": True,
            "inner_cv": True,
            "seeds": True,
            "regularization_grids": True,
            "Uno_C": True,
            "paired_bootstrap": True,
            "premise_delta_C_thresholds": True,
            "negative_premise_rule": True,
            "reserved_human_firewall": True,
            "IBS_grid_construction": True,
            "IBS_minimum_supported_points": EXPECTED_MIN_IBS_POINTS,
        },
        "frozen_execution_handling": {
            "per_fold_IBS": (
                "Apply the original 04b grid exactly. If fewer than 5 supported "
                "points remain, mark that fold IBS_UNAVAILABLE_INSUFFICIENT_COMMON_SUPPORT. "
                "Do not create an alternative grid."
            ),
            "discrimination_analysis": (
                "Continue the exact frozen B0/B2/B3/B4 Uno-C analysis."
            ),
            "SUPPORTS_CANINE_ADDED_VALUE": (
                "May be declared only if ALL primary B0/B4 IBS folds are available "
                "and the original delta-C/bootstrap/delta-IBS criteria all pass."
            ),
            "ARGUES_AGAINST_CANINE_ADDED_VALUE": (
                "Evaluate exactly using the original delta-C <= -0.02 and "
                "bootstrap P(delta C < 0) >= 0.90 rule; IBS is not part of that "
                "pre-frozen negative rule."
            ),
            "otherwise": (
                "INCONCLUSIVE_NEUTRAL. If IBS is incomplete, attach "
                "IBS_FEASIBILITY_LIMITATION and do not interpret feasible-fold-only "
                "IBS as a premise criterion."
            ),
            "feasible_fold_IBS": "DESCRIPTIVE_ONLY",
        },
        "checkpoint_reuse": {
            "allowed": True,
            "require_original_04b_protocol_sha256": True,
            "require_artifact_hash_match": True,
            "reusable_repeat_checkpoints_observed_now": n_reusable,
            "checkpoint_audit_sha256": sha256_file(CHECKPOINT_AUDIT),
        },
        "primary_result_existed_before_amendment": False,
        "reserved_human_outcomes": {
            "TARGET_OS": "CLOSED",
            "GSE21257": "CLOSED",
            "GSE39055": "CLOSED",
        },
        "required_next": (
            "04c v2 executes the unchanged discrimination protocol with this "
            "conservative IBS-feasibility handling."
        ),
    }
    write_json(AMENDMENT_JSON, amendment)

    summary = {
        "script_version": SCRIPT_VERSION,
        "status": "PASS",
        "scientific_status": amendment["scientific_status"],
        "run_started_utc": started,
        "run_finished_utc": now_utc(),
        "04b_protocol_sha256": protocol_hash,
        "GSE16091_outcomes_already_opened": True,
        "primary_result_preexisting": False,
        "reusable_primary_repeat_checkpoints": n_reusable,
        "SUPPORTS_disabled_if_any_IBS_fold_unavailable": True,
        "negative_rule_changed": False,
        "final_artifact_hashes": {
            "ibs_feasibility_amendment_json": sha256_file(AMENDMENT_JSON),
            "existing_primary_checkpoint_audit_tsv": sha256_file(CHECKPOINT_AUDIT),
        },
        "next": "04c v2",
    }
    write_json(SUMMARY_JSON, summary)

    print(f"04b protocol SHA256: {protocol_hash}")
    print(f"Validated reusable v1 primary repeat checkpoints: {n_reusable}/20")
    print()
    print("Frozen conservative rule:")
    print("  original IBS grid stays unchanged")
    print("  infeasible IBS fold -> IBS unavailable, no alternate horizon")
    print("  any unavailable primary IBS fold -> SUPPORTS cannot be declared")
    print("  original ARGUES_AGAINST rule remains evaluable")
    print("  otherwise -> INCONCLUSIVE_NEUTRAL + IBS_FEASIBILITY_LIMITATION")
    print()
    print("=" * 120)
    print("04c0 conservative post-outcome IBS-feasibility amendment: PASS")
    print("=" * 120)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 120, file=sys.stderr)
        print("04c0 conservative IBS-feasibility amendment: FAIL", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print("=" * 120, file=sys.stderr)
        raise
