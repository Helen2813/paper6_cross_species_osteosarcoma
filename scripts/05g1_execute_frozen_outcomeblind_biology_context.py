#!/usr/bin/env python3
"""
Paper 6 - execute frozen outcome-blind DOG2->TARGET biological-context analysis.

Stage 05g1
----------
05g1 executes ONLY the biological-context rules frozen in 05g0.

Evidence status inherited from 05g0:
    SECONDARY / POST-TARGET-OPENING /
    PRE-05F3F-RESULT-INSPECTION /
    OUTCOME-BLIND / MECHANISTIC-CONTEXT

Critical firewall
-----------------
This script MAY read:
  - the immutable 05g0 contract and its non-outcome artifacts;
  - all 186 DOG2 raw aligned expression samples from 05f1b;
  - all 88 TARGET raw aligned expression samples from 05f1b;
  - the exact 05f1a Hallmark memberships;
  - the exact Paper-4 frozen program memberships and outcome-blind
    external-representation labels already bound by 05g0.

This script MUST NOT read:
  - TARGET clinical/outcome values;
  - scientific contents of the completed 05f3f TARGET result;
  - GSE21257 outcomes;
  - GSE39055 outcomes;
  - DOG2 outcomes;
  - any Paper-5 result chosen after seeing Paper-6 results.

The completed 05f3f summary is checked for EXISTENCE ONLY. Its contents are
never opened.

Frozen 05g0 analysis executed here
----------------------------------
For each of:
  A. all 50 frozen MSigDB Hallmark sets; and
  B. M34/M40/M11/M24 as separate Paper-4 structural anchors,

05g1:
  1. uses all 186 DOG2 and all 88 TARGET expression samples;
  2. computes within-cohort Pearson gene-gene correlation matrices;
  3. extracts identical strict upper-triangle edge vectors in sorted
     human-gene-symbol order;
  4. reports primary Spearman DOG2-vs-TARGET edge concordance;
  5. reports secondary Pearson edge concordance and nonzero sign agreement;
  6. generates 1,000 exact-size matched random panels per tested set;
  7. matches the full joint 4-D tertile-count vector of:
       DOG2 median-expression rank,
       DOG2 variance rank,
       TARGET median-expression rank,
       TARGET variance rank;
  8. excludes the tested set itself from its random-control pool;
  9. has no unmatched/nearest-neighbor fallback;
 10. reports a matched-control percentile, NEVER a p-value;
 11. does not treat the 50 overlapping Hallmarks as independent observations;
 12. does not construct or fit a survival model.

Implementation notes
--------------------
The 05g0 contract fixes the scientific algorithm but not a computational
schedule. To make the 54 x 1,000 panel calculation practical on Windows:
  - two full 11,815 x 11,815 Pearson correlation matrices are computed once
    in float64 and stored as temporary .npy files;
  - random-panel memberships are generated serially from the one frozen RNG
    stream before any parallel work;
  - panel statistics are then evaluated in independent CPU worker processes
    using read-only memory maps;
  - worker count changes only wall-clock scheduling, never sampled panels,
    statistics, or output order.

The temporary correlation matrices are deleted before the final 05g1 directory
is committed. Exact random-panel membership is retained in a compact NPZ so
the null calculations can be replayed without relying on RNG implementation
details alone.

No CLI arguments.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import rankdata
except ImportError as exc:
    raise ImportError(
        "05g1 requires scipy (rankdata) in the active Paper-6 environment."
    ) from exc

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None


SCRIPT_VERSION = (
    "05g1-execute-frozen-outcomeblind-biological-context-v1-no-cli"
)
EXECUTION_VERSION = (
    "paper6-secondary-postopening-preresult-outcomeblind-biological-context-execution-v1"
)
SCIENTIFIC_STATUS = (
    "PASS_FROZEN_SECONDARY_POSTOPENING_PRE_05F3F_RESULT_OUTCOMEBLIND_BIOLOGY_EXECUTION"
)

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Immutable 05g0 freeze.
# ---------------------------------------------------------------------------
G0_SCRIPT = (
    ROOT
    / "scripts"
    / "05g0_freeze_postopening_preresult_outcomeblind_biology_contract.py"
)
G0_DIR = ROOT / "results" / "human_posthold_descriptive" / "05g0"
G0_CONTRACT = G0_DIR / "biological_context_analysis_contract.json"
G0_SUMMARY = G0_DIR / "summary.json"
G0_SET_REGISTRY = G0_DIR / "biological_gene_set_registry.tsv"
G0_OVERLAP_AUDIT = G0_DIR / "hallmark50_overlap_audit.tsv"
G0_EVIDENCE_STATUS = G0_DIR / "section_3_6_evidence_status.tsv"
G0_F3F_SNAPSHOT = G0_DIR / "05f3f_filesystem_snapshot_at_05g0.tsv"
G0_README = G0_DIR / "README.txt"

EXPECTED_G0_SCRIPT_SHA256 = (
    "b7b3229033da1dc1063c82738d864e9baa688379137c9f521c34e88ffeb82ead"
)
EXPECTED_G0_CONTRACT_SHA256 = (
    "405d6d6e6c8f2a7dfeb85d32c7ac045543e9e1ea8b61458e74d166367b4370ea"
)
EXPECTED_G0_STATUS = (
    "FROZEN_SECONDARY_POSTOPENING_PRE_05F3F_RESULT_OUTCOMEBLIND_BIOLOGY_CONTRACT"
)

# ---------------------------------------------------------------------------
# 05f3f completion gate. EXISTENCE ONLY; contents are never read.
# ---------------------------------------------------------------------------
F3F_SUMMARY = (
    ROOT
    / "results"
    / "human_posthold_descriptive"
    / "05f3f"
    / "summary.json"
)

# ---------------------------------------------------------------------------
# Exact expression artifacts already bound by 05g0.
# ---------------------------------------------------------------------------
F1B_DIR = ROOT / "results" / "human_posthold_descriptive" / "05f1b"
F1B_DOG2 = F1B_DIR / "matrices" / "DOG2_raw_aligned_11815genes.npz"
F1B_TARGET = F1B_DIR / "matrices" / "TARGET_OS_raw_aligned_11815genes.npz"

EXPECTED_ALIGNED_GENES = 11815
EXPECTED_DOG2_SAMPLES = 186
EXPECTED_TARGET_SAMPLES = 88
EXPECTED_HALLMARKS = 50
EXPECTED_ANCHORS = 4
EXPECTED_TOTAL_SETS = 54

# Frozen by 05g0.
BIOLOGY_RANDOM_SEED = 20860830
N_MATCHED_RANDOM_PANELS = 1000
MATCHING_TERTILES = 3
MIN_GENESET_SIZE = 3
PRIMARY_STATISTIC_NAME = (
    "spearman_between_DOG2_and_TARGET_upper_triangle_Pearson_gene_correlation_edges"
)

# Computational schedule only. Does not alter scientific output.
CPU_COUNT = max(1, os.cpu_count() or 1)
BLAS_THREADS = min(24, CPU_COUNT)
PANEL_WORKERS = min(16, max(1, CPU_COUNT - 2))

# ---------------------------------------------------------------------------
# Restart-safe staging. Final directory is immutable once present.
# ---------------------------------------------------------------------------
PARENT_DIR = ROOT / "results" / "human_posthold_descriptive"
OUT_DIR = PARENT_DIR / "05g1"
WORK_DIR = PARENT_DIR / "_05g1_work"

TMP_DOG_CORR = WORK_DIR / "_DOG2_full_gene_correlation.npy"
TMP_TARGET_CORR = WORK_DIR / "_TARGET_full_gene_correlation.npy"

OBSERVED_TSV = WORK_DIR / "structural_concordance_observed.tsv"
CONTROL_SUMMARY_TSV = WORK_DIR / "matched_random_control_summary.tsv"
CONTROL_DRAWS_TSV = WORK_DIR / "matched_random_control_draws.tsv"
PANEL_MEMBERSHIP_NPZ = WORK_DIR / "matched_random_panel_membership.npz"
MATCHING_GENE_TSV = WORK_DIR / "gene_level_matching_strata.tsv"
MATCHING_AUDIT_TSV = WORK_DIR / "matched_panel_integrity_audit.tsv"
HALLMARK_LANDSCAPE_TSV = WORK_DIR / "hallmark50_structural_landscape.tsv"
ANCHOR_TSV = WORK_DIR / "paper4_anchor_structural_context.tsv"
FIGURE_DATA_TSV = WORK_DIR / "figure5_biological_context_data.tsv"
README_TXT = WORK_DIR / "README.txt"
SUMMARY_JSON = WORK_DIR / "summary.json"


# ---------------------------------------------------------------------------
# Worker globals.
# ---------------------------------------------------------------------------
_WORKER_DOG_CORR: np.ndarray | None = None
_WORKER_TARGET_CORR: np.ndarray | None = None
_WORKER_THREADPOOL_CONTEXT: Any = None


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "na", "n/a"}:
        return ""
    return text


def normalize_symbol(value: Any) -> str:
    return clean(value).upper()


def require_file(path: Path) -> Path:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Required frozen artifact missing: {path}")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_lines(values: Iterable[str]) -> str:
    text = "\n".join(str(x) for x in values) + "\n"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
    )


def verify_exact_sha(path: Path, expected: str, label: str) -> str:
    require_file(path)
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(
            f"{label} SHA256 changed: expected={expected}, observed={observed}"
        )
    return observed


def load_05g0_module():
    require_file(G0_SCRIPT)
    spec = importlib.util.spec_from_file_location(
        "paper6_frozen_05g0",
        G0_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not import frozen 05g0 implementation.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_05g0_contract() -> Tuple[Dict[str, Any], Dict[str, Any], Any]:
    """
    Verify the exact pre-05f3f-result contract and its bound artifacts.

    Reading 05g0 is outcome-blind. We intentionally do NOT open 05f3f summary.
    """
    for path in [
        G0_SCRIPT,
        G0_CONTRACT,
        G0_SUMMARY,
        G0_SET_REGISTRY,
        G0_OVERLAP_AUDIT,
        G0_EVIDENCE_STATUS,
        G0_F3F_SNAPSHOT,
        G0_README,
    ]:
        require_file(path)

    verify_exact_sha(
        G0_SCRIPT,
        EXPECTED_G0_SCRIPT_SHA256,
        "05g0 script",
    )
    verify_exact_sha(
        G0_CONTRACT,
        EXPECTED_G0_CONTRACT_SHA256,
        "05g0 biological contract",
    )

    contract = read_json(G0_CONTRACT)
    summary = read_json(G0_SUMMARY)

    if clean(contract.get("scientific_status")) != EXPECTED_G0_STATUS:
        raise RuntimeError("05g0 contract scientific status changed.")
    if clean(summary.get("scientific_status")) != EXPECTED_G0_STATUS:
        raise RuntimeError("05g0 summary scientific status changed.")

    if clean(summary.get("script_sha256")) != EXPECTED_G0_SCRIPT_SHA256:
        raise RuntimeError("05g0 summary does not bind the expected script SHA.")
    if clean(summary.get("contract_sha256")) != EXPECTED_G0_CONTRACT_SHA256:
        raise RuntimeError("05g0 summary does not bind the expected contract SHA.")

    chronology = contract.get("chronology_proof") or {}
    evidentiary = contract.get("evidentiary_status") or {}
    if evidentiary.get(
        "05f3f_completed_summary_present_at_freeze_boundary"
    ) is not False:
        raise RuntimeError(
            "05g0 no longer documents a pre-05f3f-result freeze boundary."
        )

    if clean(evidentiary.get("claimable_status")) != (
        "SECONDARY_POST_TARGET_OPENING_PRE_05F3F_RESULT_INSPECTION_OUTCOME_BLIND"
    ):
        raise RuntimeError("05g0 claimable evidence status changed.")

    next_stage = contract.get("next_authorized_stage") or {}
    required_next = {
        "stage": "05g1",
        "may_read_DOG2_expression": True,
        "may_read_TARGET_expression": True,
        "may_read_TARGET_outcomes": False,
        "may_read_GSE21257_outcomes": False,
        "may_read_GSE39055_outcomes": False,
        "may_fit_survival_model": False,
        "may_change_05f3f": False,
    }
    for key, expected in required_next.items():
        if next_stage.get(key) != expected:
            raise RuntimeError(
                f"05g0 next-stage authorization changed for {key!r}: "
                f"observed={next_stage.get(key)!r}, expected={expected!r}"
            )

    primary = contract.get("primary_structural_statistic") or {}
    if clean(primary.get("name")) != PRIMARY_STATISTIC_NAME:
        raise RuntimeError("05g0 primary structural statistic changed.")
    if int(primary.get("gene_set_minimum_size", -1)) != MIN_GENESET_SIZE:
        raise RuntimeError("05g0 minimum gene-set size changed.")
    if clean(primary.get("within_cohort_gene_edge_measure")) != "Pearson correlation":
        raise RuntimeError("05g0 within-cohort edge measure changed.")
    if clean(primary.get("cross_cohort_edge_concordance")) != "Spearman correlation":
        raise RuntimeError("05g0 cross-cohort edge statistic changed.")
    if primary.get("TARGET_outcome_complete_case_filter") is not False:
        raise RuntimeError("05g0 unexpectedly authorizes TARGET survival intersection.")

    matched = contract.get("matched_random_control_contract") or {}
    checks = {
        "n_panels_per_gene_set": N_MATCHED_RANDOM_PANELS,
        "random_seed": BIOLOGY_RANDOM_SEED,
        "exact_gene_count_match": True,
        "sampling_without_replacement_within_panel": True,
        "sampling_independent_across_panels": True,
        "tested_gene_set_excluded_from_its_own_control_pool": True,
        "matching_bins_per_covariate": MATCHING_TERTILES,
        "nearest_neighbor_or_unmatched_fallback_allowed": False,
        "matched_control_percentile_is_p_value": False,
        "multiple_testing_claim_from_50_percentiles": False,
    }
    for key, expected in checks.items():
        if matched.get(key) != expected:
            raise RuntimeError(
                f"05g0 matched-control rule changed for {key!r}: "
                f"observed={matched.get(key)!r}, expected={expected!r}"
            )
    if clean(
        matched.get("fallback_if_any_stratum_has_insufficient_candidates")
    ) != "FAIL_CLOSED":
        raise RuntimeError("05g0 matched-control fallback is no longer FAIL_CLOSED.")

    dependence = contract.get("dependence_and_inference_guardrails") or {}
    if dependence.get("treat_50_Hallmarks_as_independent_n50") is not False:
        raise RuntimeError("05g0 Hallmark dependence guardrail changed.")
    if clean(dependence.get("module_level_regression_p_value")) != "FORBIDDEN":
        raise RuntimeError("05g0 regression p-value guardrail changed.")
    if clean(dependence.get("module_level_correlation_p_value")) != "FORBIDDEN":
        raise RuntimeError("05g0 correlation p-value guardrail changed.")

    # Independently verify the 05g0 auxiliary artifacts against hashes frozen
    # inside the contract itself.
    artifact_expectations = [
        (
            G0_SET_REGISTRY,
            clean(
                (contract.get("analysis_populations") or {}).get(
                    "gene_set_registry_sha256"
                )
            ),
            "05g0 gene-set registry",
        ),
        (
            G0_OVERLAP_AUDIT,
            clean(dependence.get("hallmark_overlap_audit_sha256")),
            "05g0 Hallmark overlap audit",
        ),
        (
            G0_EVIDENCE_STATUS,
            clean(
                (contract.get("manuscript_status_table") or {}).get("sha256")
            ),
            "05g0 evidence-status table",
        ),
        (
            G0_F3F_SNAPSHOT,
            clean(chronology.get("05f3f_output_filesystem_snapshot_sha256")),
            "05g0 05f3f filesystem snapshot",
        ),
    ]

    for path, expected, label in artifact_expectations:
        if len(expected) != 64:
            raise RuntimeError(f"{label}: missing/malformed frozen SHA.")
        verify_exact_sha(path, expected, label)

    # 05f3f is now required to be complete, but content is not read.
    if not F3F_SUMMARY.exists() or not F3F_SUMMARY.is_file():
        raise RuntimeError(
            "05g1 is authorized after 05f3f completion, but 05f3f/summary.json "
            "does not yet exist. Do not run biology early."
        )

    g0 = load_05g0_module()
    return contract, summary, g0


def load_exact_memberships(
    g0: Any,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    List[Dict[str, Any]],
    Dict[str, Any],
]:
    """
    Reconstruct memberships from the same frozen sources used by 05g0 and
    prove exact n/membership hashes against biological_gene_set_registry.tsv.
    """
    state = g0.verify_paper6_representation()
    anchors, paper4_provenance = g0.verify_paper4_anchors(
        state["alignment"]
    )

    registry = pd.read_csv(
        G0_SET_REGISTRY,
        sep="\t",
        dtype=str,
        low_memory=False,
    ).fillna("")

    required_registry = {
        "set_family",
        "set_id",
        "n_genes",
        "membership_sha256",
        "prior_anchor_class",
        "analysis_role",
        "is_one_of_50_hallmarks",
        "is_paper4_anchor",
    }
    missing = sorted(required_registry - set(registry.columns))
    if missing:
        raise RuntimeError(
            f"05g0 gene-set registry lacks required columns: {missing}"
        )

    hallmark = state["hallmark"].copy()
    hallmark["hallmark_module"] = hallmark["hallmark_module"].map(clean)
    hallmark["human_gene_symbol"] = hallmark["human_gene_symbol"].map(
        normalize_symbol
    )

    anchors = anchors.copy()
    anchors["set_id"] = anchors["set_id"].map(clean)
    anchors["human_gene_symbol"] = anchors["human_gene_symbol"].map(
        normalize_symbol
    )

    set_defs: List[Dict[str, Any]] = []

    for row in registry.itertuples(index=False):
        family = clean(row.set_family)
        set_id = clean(row.set_id)
        expected_n = int(clean(row.n_genes))
        expected_hash = clean(row.membership_sha256)

        if family == "MSIGDB_HALLMARK50_LANDSCAPE":
            genes = sorted(
                set(
                    hallmark.loc[
                        hallmark["hallmark_module"].eq(set_id),
                        "human_gene_symbol",
                    ].astype(str)
                )
            )
        elif family == "PAPER4_FROZEN_PROGRAM_ANCHOR":
            genes = sorted(
                set(
                    anchors.loc[
                        anchors["set_id"].eq(set_id),
                        "human_gene_symbol",
                    ].astype(str)
                )
            )
        else:
            raise RuntimeError(
                f"Unexpected frozen gene-set family in 05g0: {family!r}"
            )

        if len(genes) != expected_n:
            raise RuntimeError(
                f"{family}/{set_id}: membership size changed; "
                f"observed={len(genes)}, expected={expected_n}"
            )
        if len(genes) < MIN_GENESET_SIZE:
            raise RuntimeError(
                f"{family}/{set_id}: n={len(genes)} < frozen minimum "
                f"{MIN_GENESET_SIZE}."
            )

        observed_hash = sha256_lines(genes)
        if observed_hash != expected_hash:
            raise RuntimeError(
                f"{family}/{set_id}: membership SHA changed; "
                f"expected={expected_hash}, observed={observed_hash}"
            )

        set_defs.append(
            {
                "set_index": len(set_defs),
                "set_family": family,
                "set_id": set_id,
                "genes": genes,
                "n_genes": len(genes),
                "membership_sha256": observed_hash,
                "prior_anchor_class": clean(row.prior_anchor_class),
                "analysis_role": clean(row.analysis_role),
                "is_hallmark": family == "MSIGDB_HALLMARK50_LANDSCAPE",
                "is_anchor": family == "PAPER4_FROZEN_PROGRAM_ANCHOR",
            }
        )

    if len(set_defs) != EXPECTED_TOTAL_SETS:
        raise RuntimeError(
            f"05g1 reconstructed {len(set_defs)} sets, expected "
            f"{EXPECTED_TOTAL_SETS}."
        )
    if sum(x["is_hallmark"] for x in set_defs) != EXPECTED_HALLMARKS:
        raise RuntimeError("05g1 did not reconstruct exactly 50 Hallmarks.")
    if sum(x["is_anchor"] for x in set_defs) != EXPECTED_ANCHORS:
        raise RuntimeError("05g1 did not reconstruct exactly four anchors.")

    return state["alignment"], registry, set_defs, paper4_provenance


def load_raw_expression(
    alignment: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load only raw aligned expression matrices already outcome-free in 05f1b.
    """
    require_file(F1B_DOG2)
    require_file(F1B_TARGET)

    with np.load(F1B_DOG2, allow_pickle=False) as dog_npz:
        X_dog = np.asarray(dog_npz["X"], dtype=np.float64)
        dog_genes = np.asarray(
            [normalize_symbol(x) for x in dog_npz["human_gene_symbol"].astype(str)]
        )

    with np.load(F1B_TARGET, allow_pickle=False) as target_npz:
        X_target = np.asarray(target_npz["X"], dtype=np.float64)
        target_genes = np.asarray(
            [
                normalize_symbol(x)
                for x in target_npz["human_gene_symbol"].astype(str)
            ]
        )

    expected_genes = alignment["human_gene_symbol"].map(
        normalize_symbol
    ).to_numpy(dtype=str)

    if X_dog.shape != (EXPECTED_DOG2_SAMPLES, EXPECTED_ALIGNED_GENES):
        raise RuntimeError(
            f"DOG2 matrix shape={X_dog.shape}, expected "
            f"({EXPECTED_DOG2_SAMPLES}, {EXPECTED_ALIGNED_GENES})."
        )
    if X_target.shape != (EXPECTED_TARGET_SAMPLES, EXPECTED_ALIGNED_GENES):
        raise RuntimeError(
            f"TARGET matrix shape={X_target.shape}, expected "
            f"({EXPECTED_TARGET_SAMPLES}, {EXPECTED_ALIGNED_GENES})."
        )

    if not np.array_equal(dog_genes, expected_genes):
        raise RuntimeError("DOG2 serialized gene order differs from 05g0 alignment.")
    if not np.array_equal(target_genes, expected_genes):
        raise RuntimeError("TARGET serialized gene order differs from 05g0 alignment.")
    if not np.array_equal(dog_genes, target_genes):
        raise RuntimeError("DOG2/TARGET gene order differs.")

    if not np.isfinite(X_dog).all():
        raise RuntimeError("DOG2 raw aligned matrix contains nonfinite values.")
    if not np.isfinite(X_target).all():
        raise RuntimeError("TARGET raw aligned matrix contains nonfinite values.")

    return X_dog, X_target, expected_genes


def percentile_ranks(values: np.ndarray) -> np.ndarray:
    ranks = rankdata(values, method="average")
    n = len(values)
    if n < 2:
        raise RuntimeError("Cannot compute frozen percentile ranks with n<2.")
    q = (ranks - 1.0) / float(n - 1)
    return np.asarray(q, dtype=np.float64)


def tertile_labels(q: np.ndarray) -> np.ndarray:
    labels = np.where(
        q < (1.0 / 3.0),
        0,
        np.where(q < (2.0 / 3.0), 1, 2),
    )
    return labels.astype(np.int8)


def build_matching_strata(
    X_dog: np.ndarray,
    X_target: np.ndarray,
    genes: np.ndarray,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Frozen 4-D rank matching.

    Variance uses ddof=0. Because every gene has the same cohort sample count
    and no missing values, ddof=0 vs ddof=1 differs only by a positive constant
    and therefore cannot change the frozen variance ranks or tertile labels.
    """
    dog_median = np.median(X_dog, axis=0)
    target_median = np.median(X_target, axis=0)
    dog_var = np.var(X_dog, axis=0, ddof=0)
    target_var = np.var(X_target, axis=0, ddof=0)

    if not all(
        np.isfinite(x).all()
        for x in [dog_median, target_median, dog_var, target_var]
    ):
        raise RuntimeError("Nonfinite matching covariate detected.")

    # Pearson correlations are undefined for zero-variance genes. 05g0 did not
    # authorize filtering the 11,815-gene random-control universe, so fail
    # closed rather than silently dropping such genes.
    zero_dog = np.flatnonzero(dog_var <= 0.0)
    zero_target = np.flatnonzero(target_var <= 0.0)
    if len(zero_dog) or len(zero_target):
        examples = sorted(
            set(genes[zero_dog].tolist() + genes[zero_target].tolist())
        )[:20]
        raise RuntimeError(
            "Frozen 11,815-gene universe contains zero-variance gene(s), "
            "for which Pearson edge correlation is undefined. 05g0 authorizes "
            "no variance-filter fallback. FAIL_CLOSED. Examples="
            f"{examples}"
        )

    dog_med_q = percentile_ranks(dog_median)
    dog_var_q = percentile_ranks(dog_var)
    target_med_q = percentile_ranks(target_median)
    target_var_q = percentile_ranks(target_var)

    dmb = tertile_labels(dog_med_q)
    dvb = tertile_labels(dog_var_q)
    tmb = tertile_labels(target_med_q)
    tvb = tertile_labels(target_var_q)

    # Stable 0..80 code for 3^4 Cartesian-product strata.
    stratum = (
        dmb.astype(np.int16) * 27
        + dvb.astype(np.int16) * 9
        + tmb.astype(np.int16) * 3
        + tvb.astype(np.int16)
    ).astype(np.int16)

    frame = pd.DataFrame(
        {
            "aligned_feature_index": np.arange(
                EXPECTED_ALIGNED_GENES, dtype=np.int32
            ),
            "human_gene_symbol": genes,
            "DOG2_median_expression": dog_median,
            "DOG2_variance_ddof0": dog_var,
            "TARGET_median_expression": target_median,
            "TARGET_variance_ddof0": target_var,
            "DOG2_median_rank_q": dog_med_q,
            "DOG2_variance_rank_q": dog_var_q,
            "TARGET_median_rank_q": target_med_q,
            "TARGET_variance_rank_q": target_var_q,
            "DOG2_median_tertile": dmb,
            "DOG2_variance_tertile": dvb,
            "TARGET_median_tertile": tmb,
            "TARGET_variance_tertile": tvb,
            "joint_stratum_0_80": stratum,
        }
    )

    return frame, stratum


def build_symbol_rank(genes: np.ndarray) -> np.ndarray:
    order = np.argsort(genes, kind="mergesort")
    rank = np.empty(len(genes), dtype=np.int32)
    rank[order] = np.arange(len(genes), dtype=np.int32)
    return rank


def generate_matched_panels(
    set_defs: List[Dict[str, Any]],
    genes: np.ndarray,
    stratum: np.ndarray,
) -> Tuple[
    Dict[int, np.ndarray],
    Dict[int, np.ndarray],
    pd.DataFrame,
]:
    """
    Generate exact random-panel memberships serially from the one frozen RNG
    stream. Parallel execution begins only after membership is fixed.
    """
    gene_to_index = {gene: i for i, gene in enumerate(genes)}
    if len(gene_to_index) != EXPECTED_ALIGNED_GENES:
        raise RuntimeError("Aligned human gene symbols are not unique.")

    symbol_rank = build_symbol_rank(genes)
    by_stratum: Dict[int, np.ndarray] = {
        s: np.flatnonzero(stratum == s).astype(np.int32)
        for s in range(81)
    }

    rng = np.random.default_rng(BIOLOGY_RANDOM_SEED)

    observed_indices: Dict[int, np.ndarray] = {}
    panel_indices: Dict[int, np.ndarray] = {}
    audit_rows: List[Dict[str, Any]] = []

    for i, definition in enumerate(set_defs, start=1):
        set_index = int(definition["set_index"])
        set_id = definition["set_id"]
        family = definition["set_family"]
        k = int(definition["n_genes"])

        try:
            obs = np.asarray(
                [gene_to_index[g] for g in definition["genes"]],
                dtype=np.int32,
            )
        except KeyError as exc:
            raise RuntimeError(
                f"{family}/{set_id}: gene outside aligned universe: {exc}"
            ) from exc

        # Frozen edge order is sorted human-gene-symbol order.
        obs = obs[np.argsort(symbol_rank[obs], kind="mergesort")]
        observed_indices[set_index] = obs

        required_counts = np.bincount(
            stratum[obs],
            minlength=81,
        ).astype(np.int32)

        tested_mask = np.zeros(EXPECTED_ALIGNED_GENES, dtype=bool)
        tested_mask[obs] = True

        candidates_by_stratum: Dict[int, np.ndarray] = {}
        slacks: List[int] = []

        for s in np.flatnonzero(required_counts > 0):
            candidates = by_stratum[int(s)]
            candidates = candidates[~tested_mask[candidates]]
            need = int(required_counts[int(s)])

            if len(candidates) < need:
                raise RuntimeError(
                    f"{family}/{set_id}: stratum {int(s)} requires {need} "
                    f"control genes but only {len(candidates)} remain after "
                    "excluding the tested set. 05g0 requires FAIL_CLOSED."
                )

            candidates_by_stratum[int(s)] = candidates
            slacks.append(int(len(candidates) - need))

        panels = np.empty(
            (N_MATCHED_RANDOM_PANELS, k),
            dtype=np.int32,
        )

        for draw in range(N_MATCHED_RANDOM_PANELS):
            selected_parts: List[np.ndarray] = []
            for s in np.flatnonzero(required_counts > 0):
                s_int = int(s)
                need = int(required_counts[s_int])
                candidates = candidates_by_stratum[s_int]
                selected = rng.choice(
                    candidates,
                    size=need,
                    replace=False,
                ).astype(np.int32, copy=False)
                selected_parts.append(selected)

            selected_all = np.concatenate(selected_parts)
            if len(selected_all) != k:
                raise RuntimeError(
                    f"{family}/{set_id}: random panel size construction error."
                )

            # Exact frozen gene-symbol edge order.
            selected_all = selected_all[
                np.argsort(
                    symbol_rank[selected_all],
                    kind="mergesort",
                )
            ]
            panels[draw, :] = selected_all

        # Independent replay of the exact-matching invariants.
        if np.isin(panels, obs).any():
            raise RuntimeError(
                f"{family}/{set_id}: random control contains tested-set gene."
            )

        sorted_numeric = np.sort(panels, axis=1)
        if (np.diff(sorted_numeric, axis=1) == 0).any():
            raise RuntimeError(
                f"{family}/{set_id}: duplicate gene within a random panel."
            )

        all_exact = True
        for s in np.flatnonzero(required_counts > 0):
            s_int = int(s)
            observed_count = int(required_counts[s_int])
            per_draw = np.sum(
                stratum[panels] == s_int,
                axis=1,
            )
            if not np.all(per_draw == observed_count):
                all_exact = False
                break

        if not all_exact:
            raise RuntimeError(
                f"{family}/{set_id}: exact 81-stratum count replay failed."
            )

        panel_indices[set_index] = panels

        audit_rows.append(
            {
                "set_index": set_index,
                "set_family": family,
                "set_id": set_id,
                "n_genes": k,
                "n_panels": N_MATCHED_RANDOM_PANELS,
                "used_joint_strata": int(np.sum(required_counts > 0)),
                "minimum_candidate_slack_after_set_exclusion": int(min(slacks)),
                "tested_set_excluded_from_control_pool": True,
                "all_panels_exact_size": True,
                "all_panels_unique_within_panel": True,
                "all_panels_exact_81_stratum_count_match": True,
                "fallback_used": False,
            }
        )

        print(
            f"  matched panels {i:02d}/{len(set_defs)}: "
            f"{family}/{set_id} [n={k}] PASS"
        )

    return observed_indices, panel_indices, pd.DataFrame(audit_rows)


def save_panel_membership(
    set_defs: List[Dict[str, Any]],
    observed_indices: Dict[int, np.ndarray],
    panel_indices: Dict[int, np.ndarray],
    genes: np.ndarray,
) -> None:
    payload: Dict[str, Any] = {
        "random_seed": np.asarray([BIOLOGY_RANDOM_SEED], dtype=np.int64),
        "n_panels_per_set": np.asarray(
            [N_MATCHED_RANDOM_PANELS], dtype=np.int32
        ),
        "human_gene_symbol": np.asarray(genes, dtype=str),
        "set_id": np.asarray([x["set_id"] for x in set_defs], dtype=str),
        "set_family": np.asarray(
            [x["set_family"] for x in set_defs], dtype=str
        ),
        "membership_sha256": np.asarray(
            [x["membership_sha256"] for x in set_defs], dtype=str
        ),
    }

    for definition in set_defs:
        idx = int(definition["set_index"])
        payload[f"observed_{idx:03d}"] = observed_indices[idx].astype(
            np.int32, copy=False
        )
        payload[f"panels_{idx:03d}"] = panel_indices[idx].astype(
            np.int32, copy=False
        )

    tmp = PANEL_MEMBERSHIP_NPZ.with_suffix(
        PANEL_MEMBERSHIP_NPZ.suffix + ".tmp"
    )
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(tmp, PANEL_MEMBERSHIP_NPZ)


def compute_full_pearson_correlation(
    X: np.ndarray,
    path: Path,
    label: str,
) -> None:
    """
    Pearson matrix from centered columns normalized to unit Euclidean norm:
        R = Z.T @ Z
    which is algebraically identical to Pearson correlation.
    """
    print(
        f"Computing full {label} {EXPECTED_ALIGNED_GENES:,} x "
        f"{EXPECTED_ALIGNED_GENES:,} Pearson matrix "
        f"[CPU BLAS threads <= {BLAS_THREADS}]..."
    )

    centered = X - np.mean(X, axis=0, keepdims=True)
    ss = np.einsum("ij,ij->j", centered, centered, optimize=True)

    if not np.isfinite(ss).all() or np.any(ss <= 0.0):
        bad = np.flatnonzero((~np.isfinite(ss)) | (ss <= 0.0))[:20]
        raise RuntimeError(
            f"{label}: zero/nonfinite column variance prevents Pearson "
            f"correlation. Feature indices={bad.tolist()}"
        )

    Z = centered / np.sqrt(ss)[None, :]

    if threadpool_limits is not None:
        with threadpool_limits(limits=BLAS_THREADS):
            corr = np.matmul(Z.T, Z)
    else:
        corr = np.matmul(Z.T, Z)

    corr = np.asarray(corr, dtype=np.float64)
    np.clip(corr, -1.0, 1.0, out=corr)
    np.fill_diagonal(corr, 1.0)

    if corr.shape != (EXPECTED_ALIGNED_GENES, EXPECTED_ALIGNED_GENES):
        raise RuntimeError(f"{label}: correlation matrix shape changed.")
    if not np.isfinite(corr).all():
        raise RuntimeError(f"{label}: nonfinite Pearson correlation detected.")

    # Numerical symmetry check.
    max_asym = float(np.max(np.abs(corr - corr.T)))
    if max_asym > 1e-10:
        raise RuntimeError(
            f"{label}: correlation matrix asymmetry {max_asym:.3e} > 1e-10."
        )

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, corr, allow_pickle=False)
    os.replace(tmp, path)

    # Read-only mmap replay.
    replay = np.load(path, mmap_mode="r", allow_pickle=False)
    if replay.shape != corr.shape or replay.dtype != np.float64:
        raise RuntimeError(f"{label}: saved correlation matrix replay failed.")

    print(
        f"  {label} Pearson matrix: PASS "
        f"[max asymmetry={max_asym:.3e}; "
        f"file={path.stat().st_size / (1024**3):.2f} GiB]"
    )


def pearson_1d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.shape != y.shape or x.ndim != 1:
        raise RuntimeError("pearson_1d received incompatible vectors.")

    xc = x - np.mean(x)
    yc = y - np.mean(y)

    denom = math.sqrt(
        float(np.dot(xc, xc)) * float(np.dot(yc, yc))
    )
    if not math.isfinite(denom) or denom <= 0.0:
        return float("nan")

    value = float(np.dot(xc, yc) / denom)
    return float(np.clip(value, -1.0, 1.0))


def structural_statistics(
    corr_dog: np.ndarray,
    corr_target: np.ndarray,
    indices: np.ndarray,
) -> Tuple[float, float, float, int, int]:
    indices = np.asarray(indices, dtype=np.int32)
    k = len(indices)
    if k < MIN_GENESET_SIZE:
        raise RuntimeError(
            f"Structural statistic requires >= {MIN_GENESET_SIZE} genes."
        )

    tri = np.triu_indices(k, k=1)
    left = indices[tri[0]]
    right = indices[tri[1]]

    edge_dog = np.asarray(corr_dog[left, right], dtype=np.float64)
    edge_target = np.asarray(corr_target[left, right], dtype=np.float64)

    if not np.isfinite(edge_dog).all() or not np.isfinite(edge_target).all():
        return float("nan"), float("nan"), float("nan"), len(edge_dog), 0

    rank_dog = rankdata(edge_dog, method="average")
    rank_target = rankdata(edge_target, method="average")

    spearman = pearson_1d(rank_dog, rank_target)
    pearson = pearson_1d(edge_dog, edge_target)

    nonzero = (edge_dog != 0.0) & (edge_target != 0.0)
    n_nonzero = int(np.sum(nonzero))
    if n_nonzero == 0:
        sign_agreement = float("nan")
    else:
        sign_agreement = float(
            np.mean(
                np.sign(edge_dog[nonzero])
                == np.sign(edge_target[nonzero])
            )
        )

    return (
        spearman,
        pearson,
        sign_agreement,
        int(len(edge_dog)),
        n_nonzero,
    )


def worker_init(
    dog_corr_path: str,
    target_corr_path: str,
) -> None:
    global _WORKER_DOG_CORR
    global _WORKER_TARGET_CORR
    global _WORKER_THREADPOOL_CONTEXT

    _WORKER_DOG_CORR = np.load(
        dog_corr_path,
        mmap_mode="r",
        allow_pickle=False,
    )
    _WORKER_TARGET_CORR = np.load(
        target_corr_path,
        mmap_mode="r",
        allow_pickle=False,
    )

    # Prevent accidental nested BLAS over-subscription inside worker processes.
    if threadpool_limits is not None:
        _WORKER_THREADPOOL_CONTEXT = threadpool_limits(limits=1)
        _WORKER_THREADPOOL_CONTEXT.__enter__()


def evaluate_set_task(
    task: Tuple[
        int,
        str,
        str,
        np.ndarray,
        np.ndarray,
    ],
) -> Dict[str, Any]:
    if _WORKER_DOG_CORR is None or _WORKER_TARGET_CORR is None:
        raise RuntimeError("05g1 worker correlation matrices are not initialized.")

    set_index, family, set_id, observed, panels = task

    obs = structural_statistics(
        _WORKER_DOG_CORR,
        _WORKER_TARGET_CORR,
        observed,
    )

    if not all(math.isfinite(x) for x in obs[:3]):
        raise RuntimeError(
            f"{family}/{set_id}: nonfinite observed structural statistic."
        )

    controls = np.empty(
        (panels.shape[0], 3),
        dtype=np.float64,
    )

    for draw in range(panels.shape[0]):
        stats = structural_statistics(
            _WORKER_DOG_CORR,
            _WORKER_TARGET_CORR,
            panels[draw],
        )
        if not all(math.isfinite(x) for x in stats[:3]):
            raise RuntimeError(
                f"{family}/{set_id}: nonfinite random-control statistic "
                f"at draw={draw}."
            )
        controls[draw, :] = stats[:3]

    return {
        "set_index": int(set_index),
        "set_family": family,
        "set_id": set_id,
        "observed_spearman": float(obs[0]),
        "observed_pearson": float(obs[1]),
        "observed_sign_agreement": float(obs[2]),
        "n_edges": int(obs[3]),
        "n_nonzero_sign_edges": int(obs[4]),
        "control_stats": controls,
    }


def evaluate_all_sets(
    set_defs: List[Dict[str, Any]],
    observed_indices: Dict[int, np.ndarray],
    panel_indices: Dict[int, np.ndarray],
) -> List[Dict[str, Any]]:
    tasks = [
        (
            int(d["set_index"]),
            str(d["set_family"]),
            str(d["set_id"]),
            observed_indices[int(d["set_index"])],
            panel_indices[int(d["set_index"])],
        )
        for d in set_defs
    ]

    print(
        f"Evaluating {len(tasks)} observed sets + "
        f"{len(tasks) * N_MATCHED_RANDOM_PANELS:,} matched panels "
        f"with {PANEL_WORKERS} CPU worker process(es)..."
    )

    if PANEL_WORKERS <= 1:
        worker_init(str(TMP_DOG_CORR), str(TMP_TARGET_CORR))
        results = []
        for i, task in enumerate(tasks, start=1):
            result = evaluate_set_task(task)
            results.append(result)
            print(
                f"  structural statistics {i:02d}/{len(tasks)}: "
                f"{result['set_family']}/{result['set_id']} PASS"
            )
        return results

    results: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=PANEL_WORKERS,
        initializer=worker_init,
        initargs=(str(TMP_DOG_CORR), str(TMP_TARGET_CORR)),
    ) as executor:
        for i, result in enumerate(
            executor.map(evaluate_set_task, tasks, chunksize=1),
            start=1,
        ):
            results.append(result)
            print(
                f"  structural statistics {i:02d}/{len(tasks)}: "
                f"{result['set_family']}/{result['set_id']} PASS"
            )

    return results


def quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q))


def build_result_tables(
    set_defs: List[Dict[str, Any]],
    results: List[Dict[str, Any]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_index = {int(x["set_index"]): x for x in results}

    observed_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    draw_rows: List[Dict[str, Any]] = []

    for definition in set_defs:
        idx = int(definition["set_index"])
        result = by_index[idx]
        controls = result["control_stats"]

        control_s = controls[:, 0]
        control_p = controls[:, 1]
        control_sign = controls[:, 2]

        observed_s = float(result["observed_spearman"])
        percentile = float(
            (1 + np.sum(control_s <= observed_s))
            / (N_MATCHED_RANDOM_PANELS + 1)
        )

        common = {
            "set_index": idx,
            "set_family": definition["set_family"],
            "set_id": definition["set_id"],
            "n_genes": int(definition["n_genes"]),
            "membership_sha256": definition["membership_sha256"],
            "prior_anchor_class": definition["prior_anchor_class"],
            "analysis_role": definition["analysis_role"],
        }

        observed_rows.append(
            {
                **common,
                "n_edges": int(result["n_edges"]),
                "n_nonzero_sign_edges": int(
                    result["n_nonzero_sign_edges"]
                ),
                "spearman_edge_concordance": observed_s,
                "pearson_edge_concordance": float(
                    result["observed_pearson"]
                ),
                "nonzero_edge_sign_agreement": float(
                    result["observed_sign_agreement"]
                ),
            }
        )

        summary_rows.append(
            {
                **common,
                "observed_spearman_edge_concordance": observed_s,
                "random_spearman_median": quantile(control_s, 0.50),
                "random_spearman_q05": quantile(control_s, 0.05),
                "random_spearman_q95": quantile(control_s, 0.95),
                "matched_control_percentile": percentile,
                "matched_control_percentile_is_p_value": False,
                "observed_pearson_edge_concordance": float(
                    result["observed_pearson"]
                ),
                "random_pearson_median": quantile(control_p, 0.50),
                "random_pearson_q05": quantile(control_p, 0.05),
                "random_pearson_q95": quantile(control_p, 0.95),
                "observed_nonzero_edge_sign_agreement": float(
                    result["observed_sign_agreement"]
                ),
                "random_sign_agreement_median": quantile(
                    control_sign, 0.50
                ),
                "random_sign_agreement_q05": quantile(control_sign, 0.05),
                "random_sign_agreement_q95": quantile(control_sign, 0.95),
                "n_random_panels": N_MATCHED_RANDOM_PANELS,
            }
        )

        for draw in range(N_MATCHED_RANDOM_PANELS):
            draw_rows.append(
                {
                    "set_index": idx,
                    "set_family": definition["set_family"],
                    "set_id": definition["set_id"],
                    "draw_index_0based": draw,
                    "spearman_edge_concordance": float(
                        controls[draw, 0]
                    ),
                    "pearson_edge_concordance": float(
                        controls[draw, 1]
                    ),
                    "nonzero_edge_sign_agreement": float(
                        controls[draw, 2]
                    ),
                }
            )

    return (
        pd.DataFrame(observed_rows),
        pd.DataFrame(summary_rows),
        pd.DataFrame(draw_rows),
    )


def write_derived_views(
    control_summary: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hallmarks = control_summary[
        control_summary["set_family"].eq(
            "MSIGDB_HALLMARK50_LANDSCAPE"
        )
    ].copy()

    if len(hallmarks) != EXPECTED_HALLMARKS:
        raise RuntimeError("Hallmark result population is not exactly 50.")

    hallmarks = hallmarks.sort_values(
        [
            "observed_spearman_edge_concordance",
            "set_id",
        ],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    hallmarks.insert(
        0,
        "structural_concordance_rank_desc",
        np.arange(1, len(hallmarks) + 1, dtype=int),
    )

    anchors = control_summary[
        control_summary["set_family"].eq(
            "PAPER4_FROZEN_PROGRAM_ANCHOR"
        )
    ].copy()

    frozen_anchor_order = {
        "M34": 0,
        "M40": 1,
        "M11": 2,
        "M24": 3,
    }
    anchors["_order"] = anchors["set_id"].map(frozen_anchor_order)
    anchors = anchors.sort_values(
        "_order",
        kind="mergesort",
    ).drop(columns="_order").reset_index(drop=True)

    if len(anchors) != EXPECTED_ANCHORS:
        raise RuntimeError("Anchor result population is not exactly four.")

    anchors["coherence_inference_status"] = (
        "DESCRIPTIVE_ONLY_NO_FROZEN_PASS_FAIL_THRESHOLD"
    )
    anchors["prior_label_retained_regardless_of_result"] = True

    figure_data = pd.concat(
        [
            hallmarks.assign(
                plot_group="HALLMARK50",
                plot_order=hallmarks[
                    "structural_concordance_rank_desc"
                ],
            ),
            anchors.assign(
                plot_group="PAPER4_ANCHOR",
                plot_order=np.arange(1, len(anchors) + 1, dtype=int),
            ),
        ],
        ignore_index=True,
        sort=False,
    )

    return hallmarks, anchors, figure_data


def write_readme() -> None:
    text = f"""Paper 6 05g1 frozen outcome-blind biological-context execution
====================================================================

Script version
--------------
{SCRIPT_VERSION}

Evidence status
---------------
SECONDARY / POST-TARGET-OPENING / PRE-05F3F-RESULT-INSPECTION /
OUTCOME-BLIND / MECHANISTIC-CONTEXT.

05g1 executes the exact scientific rules frozen in 05g0. The 05f3f summary
must exist before 05g1 starts, but 05g1 checks only its filesystem existence
and never opens its scientific contents.

Populations
-----------
- 50 frozen MSigDB Hallmark sets: primary biological/domain landscape.
- 4 separate frozen Paper-4 anchors: M34, M40, M11, M24.
- DOG2 expression: all 186 samples.
- TARGET expression: all 88 samples; no survival-complete intersection.
- Aligned gene universe: 11,815.

Primary statistic
-----------------
Within each cohort:
  Pearson gene-gene correlation.

Across cohorts:
  Spearman correlation between identical strict upper-triangle edge vectors
  in sorted human-gene-symbol order.

Secondary descriptive statistics:
  Pearson edge concordance;
  nonzero edge-sign agreement.

Matched random controls
-----------------------
- {N_MATCHED_RANDOM_PANELS} panels per tested gene set.
- Frozen RNG seed: {BIOLOGY_RANDOM_SEED}.
- Exact gene-set size.
- Tested set excluded from its own control pool.
- Exact full 81-stratum count match across the 4-D tertile Cartesian product:
    DOG2 median-expression rank;
    DOG2 variance rank;
    TARGET median-expression rank;
    TARGET variance rank.
- No nearest-neighbor/unmatched fallback; insufficiency fails closed.
- Reported matched-control percentile is descriptive and is NOT a p-value.

Dependence guardrail
--------------------
The 50 Hallmarks overlap. They are not treated as 50 independent observations.
No module-level regression/correlation p-value or multiple-testing procedure
treating them as independent is produced.

Optional layers deliberately omitted
------------------------------------
1. 05f3f module-level decomposition:
   omitted under the frozen default rule. 05g1 remains expression-only and does
   not inspect any outcome-derived 05f3f scientific output.

2. Paper-5 shared/private overlay:
   omitted because 05g0 did not bind one unique eligible Paper-5 artifact.
   No post-result artifact selection is performed.

Computational schedule
----------------------
Full gene-correlation matrices are computed once in float64.
Random-panel memberships are generated serially before parallel evaluation.
Panel statistics use up to {PANEL_WORKERS} CPU worker processes and read-only
memory-mapped correlation matrices. This affects wall-clock time only.
"""
    atomic_write_text(README_TXT, text)


def json_float(value: Any) -> float | None:
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def main() -> None:
    started = now_utc()

    print("=" * 120)
    print("Paper 6 - execute frozen outcome-blind DOG2->TARGET biological-context analysis")
    print("=" * 120)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Execution version: {EXECUTION_VERSION}")
    print()
    print("Evidence status:")
    print("  pre-TARGET prespecified: NO")
    print("  post-TARGET-opening: YES")
    print("  05g0 frozen before completed 05f3f result: YES [must verify]")
    print("  outcome-blind: YES")
    print("  confirmatory: NO")
    print()
    print("Read scope:")
    print("  DOG2 expression values: YES [186 x 11,815]")
    print("  TARGET expression values: YES [88 x 11,815]")
    print("  TARGET clinical/outcome values: NO")
    print("  05f3f scientific result contents: NO")
    print("  05f3f summary existence check only: YES")
    print("  GSE21257/GSE39055 outcomes: NO")
    print("  DOG2 outcomes: NO")
    print()
    print("Scientific activity:")
    print("  Hallmark structural landscape: YES [50]")
    print("  Paper-4 structural anchors: YES [4]")
    print(f"  matched random panels/set: {N_MATCHED_RANDOM_PANELS}")
    print("  survival/model fitting: NO")
    print("  feature selection: NO")
    print("  A3/new gate construction: NO")
    print("  05f3f modification: NO")
    print()
    print("Computation:")
    print("  device: CPU")
    print(f"  visible logical CPUs: {CPU_COUNT}")
    print(f"  correlation BLAS threads cap: {BLAS_THREADS}")
    print(f"  random-panel worker processes: {PANEL_WORKERS}")
    print()

    if OUT_DIR.exists():
        raise RuntimeError(
            "Final 05g1 output already exists. Do not overwrite/rerun an "
            f"accepted execution: {OUT_DIR}"
        )

    # Incomplete staging from an interrupted run is not a frozen artifact.
    if WORK_DIR.exists():
        print(
            "Removing incomplete prior 05g1 staging directory "
            "(no final 05g1 output exists)..."
        )
        shutil.rmtree(WORK_DIR)

    WORK_DIR.mkdir(parents=True, exist_ok=False)

    try:
        contract, g0_summary, g0 = verify_05g0_contract()
        print("05g0 exact contract + chronology: PASS")
        print(
            "05f3f completion gate: PASS "
            "[summary exists; scientific contents NOT read]"
        )

        alignment, registry, set_defs, paper4_provenance = (
            load_exact_memberships(g0)
        )
        print("Frozen Hallmark/anchor membership replay: PASS")
        print(
            f"  sets: {EXPECTED_HALLMARKS} Hallmarks + "
            f"{EXPECTED_ANCHORS} Paper-4 anchors"
        )

        X_dog, X_target, genes = load_raw_expression(alignment)
        print("Raw aligned expression replay: PASS")
        print(
            f"  DOG2={X_dog.shape}; TARGET={X_target.shape}; "
            "all values finite"
        )

        print()
        print("Building frozen 4-D matching strata...")
        matching_frame, stratum = build_matching_strata(
            X_dog,
            X_target,
            genes,
        )
        matching_frame.to_csv(
            MATCHING_GENE_TSV,
            sep="\t",
            index=False,
        )
        observed_strata = int(len(np.unique(stratum)))
        print(
            f"  observed joint strata: {observed_strata}/81; "
            "zero-variance genes: 0"
        )

        print()
        print("Generating exact matched random-panel memberships...")
        observed_indices, panel_indices, matching_audit = (
            generate_matched_panels(
                set_defs,
                genes,
                stratum,
            )
        )
        matching_audit.to_csv(
            MATCHING_AUDIT_TSV,
            sep="\t",
            index=False,
        )
        save_panel_membership(
            set_defs,
            observed_indices,
            panel_indices,
            genes,
        )
        print("Matched random-panel generation: PASS")

        print()
        compute_full_pearson_correlation(
            X_dog,
            TMP_DOG_CORR,
            "DOG2",
        )
        compute_full_pearson_correlation(
            X_target,
            TMP_TARGET_CORR,
            "TARGET",
        )

        # Expression values no longer needed after the frozen full Pearson
        # matrices and matching strata have been constructed.
        del X_dog
        del X_target

        print()
        results = evaluate_all_sets(
            set_defs,
            observed_indices,
            panel_indices,
        )

        observed_df, control_summary_df, control_draws_df = (
            build_result_tables(
                set_defs,
                results,
            )
        )

        observed_df.to_csv(
            OBSERVED_TSV,
            sep="\t",
            index=False,
        )
        control_summary_df.to_csv(
            CONTROL_SUMMARY_TSV,
            sep="\t",
            index=False,
        )
        control_draws_df.to_csv(
            CONTROL_DRAWS_TSV,
            sep="\t",
            index=False,
        )

        hallmarks, anchors, figure_data = write_derived_views(
            control_summary_df
        )
        hallmarks.to_csv(
            HALLMARK_LANDSCAPE_TSV,
            sep="\t",
            index=False,
        )
        anchors.to_csv(
            ANCHOR_TSV,
            sep="\t",
            index=False,
        )
        figure_data.to_csv(
            FIGURE_DATA_TSV,
            sep="\t",
            index=False,
        )

        write_readme()

        # Temporary full correlation matrices are implementation scratch only.
        # Random membership + raw locked expression matrices are sufficient
        # for exact scientific replay.
        for tmp_corr in [TMP_DOG_CORR, TMP_TARGET_CORR]:
            if tmp_corr.exists():
                tmp_corr.unlink()

        # Compact manuscript-facing descriptive summary. No p-values and no
        # post-result "informative/null" threshold are introduced.
        hs = hallmarks["observed_spearman_edge_concordance"].to_numpy(
            dtype=float
        )
        hp = hallmarks["matched_control_percentile"].to_numpy(dtype=float)

        top5 = hallmarks.head(5)[
            [
                "set_id",
                "observed_spearman_edge_concordance",
                "matched_control_percentile",
            ]
        ].to_dict(orient="records")

        bottom5 = hallmarks.tail(5)[
            [
                "set_id",
                "observed_spearman_edge_concordance",
                "matched_control_percentile",
            ]
        ].to_dict(orient="records")

        anchor_records = anchors[
            [
                "set_id",
                "n_genes",
                "prior_anchor_class",
                "observed_spearman_edge_concordance",
                "matched_control_percentile",
                "observed_pearson_edge_concordance",
                "observed_nonzero_edge_sign_agreement",
            ]
        ].to_dict(orient="records")

        artifact_paths = [
            OBSERVED_TSV,
            CONTROL_SUMMARY_TSV,
            CONTROL_DRAWS_TSV,
            PANEL_MEMBERSHIP_NPZ,
            MATCHING_GENE_TSV,
            MATCHING_AUDIT_TSV,
            HALLMARK_LANDSCAPE_TSV,
            ANCHOR_TSV,
            FIGURE_DATA_TSV,
            README_TXT,
        ]

        summary = {
            "script_version": SCRIPT_VERSION,
            "execution_version": EXECUTION_VERSION,
            "scientific_status": SCIENTIFIC_STATUS,
            "run_started_utc": started,
            "run_finished_utc": now_utc(),
            "script_sha256": sha256_file(Path(__file__).resolve()),

            "05g0_script_sha256": EXPECTED_G0_SCRIPT_SHA256,
            "05g0_contract_sha256": EXPECTED_G0_CONTRACT_SHA256,
            "05g0_evidence_status": clean(
                (contract.get("evidentiary_status") or {}).get(
                    "claimable_status"
                )
            ),
            "05f3f_summary_exists_at_05g1_start": True,
            "05f3f_scientific_contents_read": False,

            "DOG2_expression_samples": EXPECTED_DOG2_SAMPLES,
            "TARGET_expression_samples": EXPECTED_TARGET_SAMPLES,
            "TARGET_survival_complete_case_intersection_used": False,
            "aligned_genes": EXPECTED_ALIGNED_GENES,
            "Hallmark_sets": EXPECTED_HALLMARKS,
            "Paper4_anchor_sets": EXPECTED_ANCHORS,
            "total_tested_sets": EXPECTED_TOTAL_SETS,

            "primary_statistic": PRIMARY_STATISTIC_NAME,
            "secondary_statistics": [
                "pearson_edge_concordance",
                "nonzero_edge_sign_agreement",
            ],

            "matched_random_panels_per_set": N_MATCHED_RANDOM_PANELS,
            "random_seed": BIOLOGY_RANDOM_SEED,
            "rng": "numpy.random.default_rng / PCG64",
            "matching_bins_per_covariate": MATCHING_TERTILES,
            "possible_joint_strata": 81,
            "observed_joint_strata": observed_strata,
            "all_panels_exact_4D_stratum_match": bool(
                matching_audit[
                    "all_panels_exact_81_stratum_count_match"
                ].all()
            ),
            "fallback_used": False,
            "matched_control_percentile_is_p_value": False,

            "Hallmarks_treated_as_independent_n50": False,
            "module_level_inferential_p_values_computed": False,
            "multiple_testing_over_50_percentiles_computed": False,

            "hallmark_landscape_descriptive": {
                "spearman_median": json_float(np.median(hs)),
                "spearman_q25": json_float(np.quantile(hs, 0.25)),
                "spearman_q75": json_float(np.quantile(hs, 0.75)),
                "spearman_min": json_float(np.min(hs)),
                "spearman_max": json_float(np.max(hs)),
                "matched_control_percentile_median": json_float(
                    np.median(hp)
                ),
                "matched_control_percentile_min": json_float(np.min(hp)),
                "matched_control_percentile_max": json_float(np.max(hp)),
                "top5_by_observed_spearman": top5,
                "bottom5_by_observed_spearman": bottom5,
            },

            "paper4_anchor_context": anchor_records,
            "anchor_pass_fail_threshold_defined": False,
            "prior_anchor_labels_retained": True,

            "05f3f_module_level_diagnostic": (
                "OMITTED_UNDER_FROZEN_DEFAULT_GATE"
            ),
            "05f3f_module_level_diagnostic_reason": (
                "05g1 remains expression-only and does not inspect outcome-derived "
                "05f3f scientific outputs; 05g0 did not bind an already-available "
                "exact Hallmark decomposition artifact."
            ),

            "paper5_overlay": "OMITTED_NO_UNIQUE_05G0_BOUND_ARTIFACT",

            "circularity_guardrail": (
                "Structural concordance characterizes the expression/domain setting "
                "and is not independent proof that concordance predicts transfer utility."
            ),
            "null_or_mixed_result_must_remain_reported": True,
            "figure5_allocation_threshold_defined": False,

            "TARGET_clinical_or_outcome_values_read": False,
            "TARGET_05f3f_scientific_results_read": False,
            "DOG2_outcome_values_read": False,
            "GSE21257_outcomes_read": False,
            "GSE39055_outcomes_read": False,
            "survival_model_fitting": False,
            "prediction_model_fitting": False,
            "feature_selection": False,
            "A3_or_new_gate_construction": False,
            "05f3f_modified": False,

            "execution_device": "CPU",
            "logical_cpu_count": CPU_COUNT,
            "correlation_blas_threads_cap": BLAS_THREADS,
            "panel_worker_processes": PANEL_WORKERS,

            "paper4_anchor_provenance_replayed": {
                "strict_weights_sha256": clean(
                    paper4_provenance.get("strict_weights_sha256")
                ),
                "external_outcome_blind_representation_sha256": clean(
                    paper4_provenance.get(
                        "external_outcome_blind_representation_sha256"
                    )
                ),
                "external_outcome_blind_manifest_sha256": clean(
                    paper4_provenance.get(
                        "external_outcome_blind_manifest_sha256"
                    )
                ),
            },

            "final_artifact_hashes": {
                path.name: sha256_file(path)
                for path in artifact_paths
            },
        }

        atomic_write_json(SUMMARY_JSON, summary)

        # Final commit: no final output is visible until all scientific and
        # provenance artifacts have been written successfully.
        os.replace(WORK_DIR, OUT_DIR)

        print()
        print("=" * 120)
        print("05g1 FROZEN OUTCOME-BLIND BIOLOGICAL-CONTEXT RESULT")
        print("=" * 120)
        print("Evidence status:")
        print("  SECONDARY / POST-TARGET-OPENING /")
        print("  PRE-05F3F-RESULT-INSPECTION / OUTCOME-BLIND")
        print()
        print("Hallmark-50 structural landscape:")
        print(
            "  Spearman median [Q25,Q75]: "
            f"{np.median(hs):.4f} "
            f"[{np.quantile(hs, 0.25):.4f}, {np.quantile(hs, 0.75):.4f}]"
        )
        print(
            "  Spearman range: "
            f"{np.min(hs):.4f} to {np.max(hs):.4f}"
        )
        print(
            "  matched-control percentile median/range: "
            f"{np.median(hp):.4f} / "
            f"{np.min(hp):.4f}-{np.max(hp):.4f}"
        )
        print()
        print("Top 5 Hallmarks by observed structural concordance:")
        for row in top5:
            print(
                f"  {row['set_id']}: "
                f"rho={row['observed_spearman_edge_concordance']:.4f}, "
                f"matched percentile={row['matched_control_percentile']:.4f}"
            )
        print()
        print("Paper-4 frozen anchors [descriptive coherence only]:")
        for row in anchor_records:
            print(
                f"  {row['set_id']}: "
                f"prior={row['prior_anchor_class']}; "
                f"rho={row['observed_spearman_edge_concordance']:.4f}; "
                f"matched percentile={row['matched_control_percentile']:.4f}"
            )
        print()
        print("Guardrails:")
        print("  matched-control percentile called p-value: NO")
        print("  Hallmarks treated as independent n=50: NO")
        print("  module-level inferential p-values: NO")
        print("  05f3f scientific results read: NO")
        print("  TARGET/GSE21257/GSE39055 outcomes read: NO")
        print("  Paper-5 overlay imported: NO")
        print("  new model/gate fitted: NO")
        print()
        print(
            f"05g1 script SHA256: "
            f"{summary['script_sha256']}"
        )
        print(
            f"05g0 contract SHA256: "
            f"{summary['05g0_contract_sha256']}"
        )
        print("=" * 120)
        print(f"05g1: {SCIENTIFIC_STATUS}")
        print("=" * 120)

    except Exception:
        # Keep staging for debugging if execution fails. It is NOT a frozen
        # scientific output and will be deleted automatically on the next run.
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 120, file=sys.stderr)
        print(
            "05g1 frozen outcome-blind biological-context execution: FAIL",
            file=sys.stderr,
        )
        print(
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        print(
            "No final 05g1 directory was committed. "
            "Any _05g1_work directory is incomplete staging only.",
            file=sys.stderr,
        )
        print("=" * 120, file=sys.stderr)
        raise
