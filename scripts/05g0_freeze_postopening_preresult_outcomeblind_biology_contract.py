#!/usr/bin/env python3
"""
Paper 6 - freeze post-opening, pre-05f3f-result, outcome-blind biological-context contract.

Stage 05g0
----------
This stage exists because TARGET outcomes have already been opened for the frozen
05f3 execution, while a separate biological-context analysis was not frozen
before that opening.

The correct evidentiary status is therefore:

    SECONDARY / POST-TARGET-OPENING /
    PRE-05F3F-RESULT-INSPECTION /
    OUTCOME-BLIND / MECHANISTIC-CONTEXT

This script DOES NOT perform the biological analysis. It freezes its exact future
rules while 05f3f is still running and before a completed 05f3f summary exists.

Scientific purpose
------------------
Characterize the DOG2 -> TARGET expression-domain setting using:
  A. all 50 already-frozen MSigDB Hallmark gene sets from 05f1a; and
  B. four independently frozen Paper-4 programs (M34/M40/M11/M24) as separate
     biological anchors evaluated with the SAME structural statistic.

Important correction:
M34/M40/M11/M24 are NOT members of the MSigDB Hallmark-50 collection. They are
separate frozen Paper-4 programs and must never be represented as four of the
50 Hallmark observations.

The future outcome-blind structural statistic is frozen here:
  1. Use all 186 DOG2 expression samples and all 88 TARGET expression samples.
     Do NOT intersect TARGET expression with survival completeness.
  2. For each gene set, compute the within-cohort Pearson gene-gene correlation
     matrix in DOG2 and TARGET.
  3. Extract identical upper-triangle edge vectors.
  4. Primary structural-concordance statistic =
       Spearman(edge_DOG2, edge_TARGET).
     This deliberately mirrors the direct edge-preservation idea used in Paper 4.
  5. Secondary descriptive statistics =
       Pearson(edge_DOG2, edge_TARGET) and edge-sign concordance.
  6. Compare each observed gene set with 1,000 exact-size random panels matched
     jointly on outcome-blind expression level and variance in BOTH cohorts.
  7. Matching uses four within-cohort rank variables:
       DOG2 median-expression rank tertile,
       DOG2 variance rank tertile,
       TARGET median-expression rank tertile,
       TARGET variance rank tertile.
     A random panel must reproduce the exact 4-D stratum-count vector of the
     observed gene set. No unmatched/fallback panel is permitted.
  8. Random panels are sampled without replacement within a panel, from the
     aligned 11,815-gene universe excluding the tested set itself.
  9. Report a matched-control percentile and random-control distribution.
     DO NOT call the percentile a p-value.
 10. Hallmark modules overlap and are not independent observations. No test,
     regression p-value, or effective n=50 inference treating them as independent
     is authorized.
 11. Paper-4 anchors are a coherence check against prior outcome-blind structural
     evidence. Their labels are retained regardless of whether the new statistic
     agrees with the prior.
 12. This analysis characterizes the biological/domain-shift setting. It is NOT
     independent proof that structural concordance predicts transfer success.
 13. A module-level diagnostic extracted from 05f3f is authorized ONLY if a
     frozen output can be decomposed exactly by Hallmark dimension without model
     modification, refitting, new priors, new gates, or reinterpretation. If not,
     that diagnostic is omitted.
 14. GSE21257/GSE39055 outcomes remain sealed and may not be opened to explain
     either this biology analysis or TARGET.
 15. A null/mixed biological result remains reportable. It is never optimized
     away. Figure-5 allocation may then return to the prespecified gating result
     for manuscript page economy without changing scientific status.

Chronology / fail-closed rule
-----------------------------
05g0 may PASS only if results/human_posthold_descriptive/05f3f/summary.json does
NOT exist at the contract-freeze boundary. The script never reads any 05f3f
scientific output. It records only filesystem metadata for files that may already
exist in the running 05f3f directory.

No CLI arguments.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


SCRIPT_VERSION = (
    "05g0-freeze-postopening-preresult-outcomeblind-biology-contract-v1-no-cli"
)
CONTRACT_VERSION = (
    "paper6-secondary-postopening-preresult-outcomeblind-biological-context-v1"
)
SCIENTIFIC_STATUS = (
    "FROZEN_SECONDARY_POSTOPENING_PRE_05F3F_RESULT_OUTCOMEBLIND_BIOLOGY_CONTRACT"
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "_config"
PAPER4_BASENAME = "paper4_sarcoma_dog"

# ---------------------------------------------------------------------------
# Exact upstream Paper-6 representation artifacts.
# ---------------------------------------------------------------------------
F1A_DIR = ROOT / "results" / "human_posthold_descriptive" / "05f1a"
F1A_CONTRACT = F1A_DIR / "outcome_free_TARGET_representation_contract.json"
F1A_SUMMARY = F1A_DIR / "summary.json"
F1A_ALIGNMENT = F1A_DIR / "TARGET_primary_gene_alignment.tsv"
F1A_HALLMARK = F1A_DIR / "TARGET_hallmark50_gene_map.tsv"

EXPECTED_F1A_CONTRACT_SHA256 = (
    "f0d5296d4df5e8dcb3b677127d2c3b5474791fb2cbaeb207e094b54fb44d1434"
)
EXPECTED_F1A_STATUS = (
    "PASS_OUTCOME_FREE_TARGET_REPRESENTATION_CONTRACT_FROZEN"
)

F1B_DIR = ROOT / "results" / "human_posthold_descriptive" / "05f1b"
F1B_DOG2 = F1B_DIR / "matrices" / "DOG2_raw_aligned_11815genes.npz"
F1B_TARGET = F1B_DIR / "matrices" / "TARGET_OS_raw_aligned_11815genes.npz"
F1B_MANIFEST = F1B_DIR / "raw_aligned_matrix_manifest.tsv"
F1B_SUMMARY = F1B_DIR / "summary.json"

EXPECTED_DOG2_MATRIX_SHA256 = (
    "af7b8b65f9485df6a7ba3d0061bc12c0023719e4114d7f10fb3d731e918e3875"
)
EXPECTED_TARGET_MATRIX_SHA256 = (
    "7aca20d98731a24d90882f2042b1aa7c7aefc0095a250d4ed42c3502cc3c7b19"
)
EXPECTED_F1B_MANIFEST_SHA256 = (
    "d5e267da52955332f845655325de7a66fcf4fbc63fcde9bfac8669c3a5f46b63"
)

EXPECTED_ALIGNED_GENES = 11815
EXPECTED_HALLMARK_MODULES = 50
EXPECTED_DOG2_SAMPLES = 186
EXPECTED_TARGET_EXPRESSION_SAMPLES = 88

# ---------------------------------------------------------------------------
# Running 05f3f chronology anchor.
# ---------------------------------------------------------------------------
F3F_SCRIPT = (
    ROOT
    / "scripts"
    / "05f3f_run_exact_target86_semantic_f3c_lock.py"
)
F3F_DIR = (
    ROOT
    / "results"
    / "human_posthold_descriptive"
    / "05f3f"
)
F3F_SUMMARY = F3F_DIR / "summary.json"

# ---------------------------------------------------------------------------
# Paper-4 frozen anchors.
# ---------------------------------------------------------------------------
UPSTREAM_LOCK = ROOT / "contracts" / "00_upstream_input_lock.json"

P4_STRICT_WEIGHTS_REL = (
    Path("results")
    / "tables"
    / "GSE238110_frozen_transfer_gene_weights_strict.csv"
)
P4_EXTERNAL_REP_REL = (
    Path("results")
    / "tables"
    / "paper4_locked_independent_canine_representation.csv"
)
P4_EXTERNAL_REP_MANIFEST_REL = (
    Path("results")
    / "tables"
    / "paper4_external_canine_evidence_manifest.json"
)

ANCHOR_ORDER = ["M34", "M40", "M11", "M24"]
EXPECTED_ANCHOR_GENE_COUNTS = {
    "M34": 162,
    "M40": 111,
    "M11": 7,
    "M24": 7,
}
EXPECTED_ANCHOR_PRIOR_CLASSES = {
    "M34": "strong_external_canine_representation_preservation",
    "M40": "strong_external_canine_representation_preservation",
    "M11": "no_clear_external_canine_representation_preservation",
    "M24": "no_clear_external_canine_representation_preservation",
}

# ---------------------------------------------------------------------------
# Future biological analysis constants frozen here.
# ---------------------------------------------------------------------------
BIOLOGY_RANDOM_SEED = 20860830
N_MATCHED_RANDOM_PANELS = 1000
MATCHING_TERTILES = 3
MIN_GENESET_SIZE = 3

PRIMARY_STATISTIC = (
    "spearman_between_DOG2_and_TARGET_upper_triangle_Pearson_gene_correlation_edges"
)
SECONDARY_STATISTICS = [
    "pearson_between_DOG2_and_TARGET_upper_triangle_Pearson_gene_correlation_edges",
    "nonzero_edge_sign_concordance",
]

# ---------------------------------------------------------------------------
# Outputs. 05g0 is immutable-once-written.
# ---------------------------------------------------------------------------
OUT_DIR = ROOT / "results" / "human_posthold_descriptive" / "05g0"
CONTRACT_JSON = OUT_DIR / "biological_context_analysis_contract.json"
SUMMARY_JSON = OUT_DIR / "summary.json"
SET_REGISTRY = OUT_DIR / "biological_gene_set_registry.tsv"
OVERLAP_AUDIT = OUT_DIR / "hallmark50_overlap_audit.tsv"
EVIDENCE_STATUS = OUT_DIR / "section_3_6_evidence_status.tsv"
F3F_FILESYSTEM_SNAPSHOT = OUT_DIR / "05f3f_filesystem_snapshot_at_05g0.tsv"
README = OUT_DIR / "README.txt"


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


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    text = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    ) + "\n"
    atomic_write_text(path, text)


def sha256_lines(values: Iterable[str]) -> str:
    text = "\n".join(str(x) for x in values) + "\n"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def local_path_config() -> Dict[str, Any]:
    path = CONFIG_DIR / "paths.local.json"
    if not path.exists():
        return {}
    payload = read_json(path)
    return payload if isinstance(payload, dict) else {}


def resolve_paper4_root() -> Tuple[Path, str]:
    candidates: List[Tuple[Path, str]] = []

    env_value = os.environ.get("PAPER4_ROOT", "").strip()
    if env_value:
        candidates.append(
            (Path(env_value).expanduser(), "environment:PAPER4_ROOT")
        )

    config_value = clean(local_path_config().get("paper4_root"))
    if config_value:
        candidates.append(
            (
                Path(config_value).expanduser(),
                "_config/paths.local.json",
            )
        )

    candidates.extend(
        [
            (ROOT.parent / PAPER4_BASENAME, "sibling_repository"),
            (
                Path.home() / "Desktop" / PAPER4_BASENAME,
                "home_desktop_fallback",
            ),
        ]
    )

    checked: List[str] = []
    for candidate, source in candidates:
        resolved = candidate.resolve()
        checked.append(f"{source}: {resolved}")
        if resolved.is_dir():
            return resolved, source

    raise FileNotFoundError(
        "Could not resolve Paper 4 root. Checked:\n  - "
        + "\n  - ".join(checked)
    )


def find_locked_asset_by_filename(
    lock: Dict[str, Any],
    filename: str,
    paper4_root: Path,
) -> Tuple[Path, Dict[str, Any], str]:
    assets = lock.get("assets") or {}
    matches: List[Tuple[str, Dict[str, Any]]] = []

    for role, item in assets.items():
        if not isinstance(item, dict):
            continue
        rel = clean(item.get("relative_path"))
        if rel and Path(rel).name == filename:
            matches.append((str(role), item))

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one upstream-lock asset named {filename!r}; "
            f"found {len(matches)}."
        )

    role, item = matches[0]
    rel = clean(item.get("relative_path"))
    expected = clean(item.get("sha256")).lower()
    if len(expected) != 64:
        raise RuntimeError(
            f"Malformed SHA in upstream lock for {filename}."
        )

    path = paper4_root / rel
    require_file(path)
    observed = sha256_file(path).lower()
    if observed != expected:
        raise RuntimeError(
            f"Paper-4 frozen asset hash changed: {filename}; "
            f"expected={expected}, observed={observed}"
        )

    return path, item, role


def verify_exact_sha(path: Path, expected: str, label: str) -> str:
    require_file(path)
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(
            f"{label} hash changed: expected={expected}, observed={observed}"
        )
    return observed


def refuse_if_already_frozen() -> None:
    for path in [
        CONTRACT_JSON,
        SUMMARY_JSON,
        SET_REGISTRY,
        EVIDENCE_STATUS,
    ]:
        if path.exists():
            raise RuntimeError(
                "05g0 has already produced frozen artifacts. "
                "Do NOT overwrite or rerun this freeze. Existing artifact: "
                f"{path}"
            )


def assert_05f3f_result_not_available(stage: str) -> None:
    if F3F_SUMMARY.exists():
        stat = F3F_SUMMARY.stat()
        raise RuntimeError(
            "05g0 chronology gate failed: completed 05f3f summary already "
            f"exists at {stage}. mtime_ns={stat.st_mtime_ns}. "
            "Do not claim PRE-05F3F-RESULT specification."
        )


def filesystem_snapshot(directory: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    if not directory.exists():
        return pd.DataFrame(
            columns=[
                "relative_path",
                "is_file",
                "is_dir",
                "size_bytes",
                "mtime_ns",
                "mtime_utc",
            ]
        )

    for path in sorted(directory.rglob("*"), key=lambda x: str(x)):
        try:
            stat = path.stat()
        except FileNotFoundError:
            # A running process may atomically rename/remove a temporary file.
            continue

        rows.append(
            {
                "relative_path": path.relative_to(directory).as_posix(),
                "is_file": path.is_file(),
                "is_dir": path.is_dir(),
                "size_bytes": int(stat.st_size) if path.is_file() else 0,
                "mtime_ns": int(stat.st_mtime_ns),
                "mtime_utc": datetime.fromtimestamp(
                    stat.st_mtime,
                    tz=timezone.utc,
                ).isoformat(),
            }
        )

    return pd.DataFrame(rows)


def verify_paper6_representation() -> Dict[str, Any]:
    for path in [
        F1A_CONTRACT,
        F1A_SUMMARY,
        F1A_ALIGNMENT,
        F1A_HALLMARK,
        F1B_DOG2,
        F1B_TARGET,
        F1B_MANIFEST,
        F1B_SUMMARY,
        F3F_SCRIPT,
        UPSTREAM_LOCK,
    ]:
        require_file(path)

    f1a_hash = verify_exact_sha(
        F1A_CONTRACT,
        EXPECTED_F1A_CONTRACT_SHA256,
        "05f1a outcome-free representation contract",
    )

    f1a_summary = read_json(F1A_SUMMARY)
    if clean(f1a_summary.get("scientific_status")) != EXPECTED_F1A_STATUS:
        raise RuntimeError(
            "05f1a scientific status is not the expected frozen PASS state."
        )
    if f1a_summary.get("TARGET_outcomes_read") is not False:
        raise RuntimeError(
            "05f1a provenance unexpectedly reports TARGET outcome access."
        )

    verify_exact_sha(
        F1B_DOG2,
        EXPECTED_DOG2_MATRIX_SHA256,
        "05f1b DOG2 aligned matrix",
    )
    verify_exact_sha(
        F1B_TARGET,
        EXPECTED_TARGET_MATRIX_SHA256,
        "05f1b TARGET aligned matrix",
    )
    verify_exact_sha(
        F1B_MANIFEST,
        EXPECTED_F1B_MANIFEST_SHA256,
        "05f1b aligned matrix manifest",
    )

    alignment = pd.read_csv(
        F1A_ALIGNMENT,
        sep="\t",
        dtype=str,
        low_memory=False,
    ).fillna("")

    required_alignment = {
        "aligned_feature_index",
        "human_gene_symbol",
        "target_expression_feature",
        "dog_gene_symbol",
        "dog2_raw_feature",
    }
    missing = sorted(required_alignment - set(alignment.columns))
    if missing:
        raise RuntimeError(
            f"05f1a alignment missing columns: {missing}"
        )

    alignment["human_gene_symbol"] = (
        alignment["human_gene_symbol"].map(normalize_symbol)
    )

    if len(alignment) != EXPECTED_ALIGNED_GENES:
        raise RuntimeError(
            f"Aligned gene rows={len(alignment)}, expected "
            f"{EXPECTED_ALIGNED_GENES}."
        )
    if alignment["human_gene_symbol"].duplicated().any():
        raise RuntimeError(
            "Aligned human gene symbols are not unique."
        )

    hallmark = pd.read_csv(
        F1A_HALLMARK,
        sep="\t",
        dtype=str,
        low_memory=False,
    ).fillna("")

    required_h = {
        "hallmark_module",
        "human_gene_symbol",
        "aligned_feature_index",
    }
    missing_h = sorted(required_h - set(hallmark.columns))
    if missing_h:
        raise RuntimeError(
            f"05f1a Hallmark map missing columns: {missing_h}"
        )

    hallmark["hallmark_module"] = (
        hallmark["hallmark_module"].astype(str).map(clean)
    )
    hallmark["human_gene_symbol"] = (
        hallmark["human_gene_symbol"].map(normalize_symbol)
    )

    if hallmark["hallmark_module"].nunique() != EXPECTED_HALLMARK_MODULES:
        raise RuntimeError(
            "05f1a Hallmark population is not exactly 50 modules."
        )

    if hallmark.duplicated(
        ["hallmark_module", "human_gene_symbol"]
    ).any():
        raise RuntimeError(
            "05f1a Hallmark map has duplicate module/gene membership."
        )

    if not set(hallmark["human_gene_symbol"]).issubset(
        set(alignment["human_gene_symbol"])
    ):
        raise RuntimeError(
            "05f1a Hallmark map contains genes outside the 11,815 aligned universe."
        )

    f1b_summary = read_json(F1B_SUMMARY)
    dog_n = int(
        f1b_summary.get(
            "DOG2_samples",
            f1b_summary.get("dog2_samples", -1),
        )
    )
    target_n = int(
        f1b_summary.get(
            "TARGET_samples",
            f1b_summary.get(
                "TARGET_expression_samples",
                f1b_summary.get("target_samples", -1),
            ),
        )
    )

    # Do not require exact summary field spelling; verify matrix array shapes by
    # NPZ metadata without loading expression arrays into memory.
    with np.load(F1B_DOG2, allow_pickle=False) as dog_npz:
        if "X" not in dog_npz.files:
            raise RuntimeError("DOG2 NPZ lacks X array.")
        dog_shape = dog_npz["X"].shape

    with np.load(F1B_TARGET, allow_pickle=False) as target_npz:
        if "X" not in target_npz.files:
            raise RuntimeError("TARGET NPZ lacks X array.")
        target_shape = target_npz["X"].shape

    if tuple(dog_shape) != (
        EXPECTED_DOG2_SAMPLES,
        EXPECTED_ALIGNED_GENES,
    ):
        raise RuntimeError(
            f"DOG2 aligned matrix shape={dog_shape}, expected "
            f"({EXPECTED_DOG2_SAMPLES}, {EXPECTED_ALIGNED_GENES})."
        )

    if tuple(target_shape) != (
        EXPECTED_TARGET_EXPRESSION_SAMPLES,
        EXPECTED_ALIGNED_GENES,
    ):
        raise RuntimeError(
            f"TARGET aligned matrix shape={target_shape}, expected "
            f"({EXPECTED_TARGET_EXPRESSION_SAMPLES}, {EXPECTED_ALIGNED_GENES})."
        )

    return {
        "f1a_contract_sha256": f1a_hash,
        "f1a_summary_sha256": sha256_file(F1A_SUMMARY),
        "f1a_alignment_sha256": sha256_file(F1A_ALIGNMENT),
        "f1a_hallmark_sha256": sha256_file(F1A_HALLMARK),
        "f1b_dog2_matrix_sha256": EXPECTED_DOG2_MATRIX_SHA256,
        "f1b_target_matrix_sha256": EXPECTED_TARGET_MATRIX_SHA256,
        "f1b_manifest_sha256": EXPECTED_F1B_MANIFEST_SHA256,
        "f1b_summary_sha256": sha256_file(F1B_SUMMARY),
        "dog2_matrix_shape": list(dog_shape),
        "target_matrix_shape": list(target_shape),
        "aligned_genes": len(alignment),
        "hallmark_modules": int(hallmark["hallmark_module"].nunique()),
        "f3f_script_sha256": sha256_file(F3F_SCRIPT),
        "alignment": alignment,
        "hallmark": hallmark,
        "f1b_summary_reported_dog_n": dog_n,
        "f1b_summary_reported_target_n": target_n,
    }


def infer_human_gene_column(frame: pd.DataFrame) -> str:
    preferred = [
        "human_gene_symbol",
        "gene_symbol",
        "human_symbol",
        "ortholog_human_symbol",
    ]
    for column in preferred:
        if column in frame.columns:
            return column

    candidates = [
        column
        for column in frame.columns
        if "gene" in column.lower()
        and "symbol" in column.lower()
        and "canine" not in column.lower()
        and "dog" not in column.lower()
    ]
    if len(candidates) == 1:
        return candidates[0]

    raise RuntimeError(
        "Could not uniquely identify human gene-symbol column in frozen "
        f"Paper-4 weights. Columns={list(frame.columns)}"
    )


def verify_paper4_anchors(
    alignment: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    paper4_root, resolution = resolve_paper4_root()
    upstream_lock = read_json(UPSTREAM_LOCK)

    strict_weights_path, strict_item, strict_role = (
        find_locked_asset_by_filename(
            upstream_lock,
            P4_STRICT_WEIGHTS_REL.name,
            paper4_root,
        )
    )

    external_rep = paper4_root / P4_EXTERNAL_REP_REL
    external_manifest = paper4_root / P4_EXTERNAL_REP_MANIFEST_REL
    require_file(external_rep)
    require_file(external_manifest)

    manifest = read_json(external_manifest)
    if bool(manifest.get("outcome_loaded", False)):
        raise RuntimeError(
            "Paper-4 external representation manifest unexpectedly reports "
            "outcome_loaded=true."
        )

    weights = pd.read_csv(
        strict_weights_path,
        low_memory=False,
    )
    if "module_label" not in weights.columns:
        raise RuntimeError(
            "Frozen strict weights lack module_label."
        )

    gene_col = infer_human_gene_column(weights)
    weights["module_label"] = (
        weights["module_label"].astype(str).map(clean)
    )
    weights["_human_gene_symbol"] = (
        weights[gene_col].map(normalize_symbol)
    )

    anchors = weights[
        weights["module_label"].isin(ANCHOR_ORDER)
    ].copy()

    if anchors["_human_gene_symbol"].eq("").any():
        raise RuntimeError(
            "Blank human gene symbol in Paper-4 anchor weights."
        )

    if anchors.duplicated(
        ["module_label", "_human_gene_symbol"]
    ).any():
        raise RuntimeError(
            "Duplicate Paper-4 anchor module/gene membership."
        )

    observed_counts = (
        anchors.groupby("module_label")["_human_gene_symbol"]
        .nunique()
        .to_dict()
    )
    if observed_counts != EXPECTED_ANCHOR_GENE_COUNTS:
        raise RuntimeError(
            "Paper-4 anchor gene counts changed: "
            f"observed={observed_counts}, expected={EXPECTED_ANCHOR_GENE_COUNTS}"
        )

    external = pd.read_csv(
        external_rep,
        low_memory=False,
    )
    required_external = {
        "module_label",
        "external_canine_representation_class",
    }
    missing = sorted(required_external - set(external.columns))
    if missing:
        raise RuntimeError(
            f"Paper-4 outcome-blind external representation table lacks: {missing}"
        )

    external = external[
        external["module_label"].astype(str).isin(ANCHOR_ORDER)
    ].copy()
    if set(external["module_label"].astype(str)) != set(ANCHOR_ORDER):
        raise RuntimeError(
            "Paper-4 external representation table does not contain exact four anchors."
        )

    observed_classes = {
        clean(row.module_label): clean(
            row.external_canine_representation_class
        )
        for row in external.itertuples(index=False)
    }
    if observed_classes != EXPECTED_ANCHOR_PRIOR_CLASSES:
        raise RuntimeError(
            "Paper-4 prior anchor classes changed: "
            f"observed={observed_classes}, "
            f"expected={EXPECTED_ANCHOR_PRIOR_CLASSES}"
        )

    aligned_set = set(alignment["human_gene_symbol"].astype(str))
    rows: List[Dict[str, Any]] = []

    for module in ANCHOR_ORDER:
        genes = sorted(
            set(
                anchors.loc[
                    anchors["module_label"].eq(module),
                    "_human_gene_symbol",
                ].astype(str)
            )
        )
        aligned_genes = sorted(set(genes) & aligned_set)

        if len(aligned_genes) != len(genes):
            missing_genes = sorted(set(genes) - aligned_set)
            raise RuntimeError(
                f"Paper-4 anchor {module} has frozen gene(s) outside the "
                f"11,815 aligned universe: {missing_genes[:20]}"
            )

        for gene in aligned_genes:
            rows.append(
                {
                    "set_family": "PAPER4_FROZEN_PROGRAM_ANCHOR",
                    "set_id": module,
                    "human_gene_symbol": gene,
                    "prior_anchor_class": EXPECTED_ANCHOR_PRIOR_CLASSES[module],
                    "analysis_role": (
                        "INDEPENDENT_OUTCOME_BLIND_STRUCTURAL_COHERENCE_ANCHOR"
                    ),
                }
            )

    anchor_membership = pd.DataFrame(rows)

    provenance = {
        "paper4_root_resolution_source": resolution,
        "absolute_paper4_path_written_to_contract": False,
        "strict_weights_relative_path": str(
            strict_weights_path.relative_to(paper4_root).as_posix()
        ),
        "strict_weights_sha256": sha256_file(strict_weights_path),
        "strict_weights_upstream_lock_role": strict_role,
        "strict_weights_upstream_lock_sha256": clean(
            strict_item.get("sha256")
        ),
        "external_outcome_blind_representation_relative_path": (
            P4_EXTERNAL_REP_REL.as_posix()
        ),
        "external_outcome_blind_representation_sha256": sha256_file(
            external_rep
        ),
        "external_outcome_blind_manifest_relative_path": (
            P4_EXTERNAL_REP_MANIFEST_REL.as_posix()
        ),
        "external_outcome_blind_manifest_sha256": sha256_file(
            external_manifest
        ),
        "external_manifest_outcome_loaded": bool(
            manifest.get("outcome_loaded", False)
        ),
        "anchor_gene_counts": EXPECTED_ANCHOR_GENE_COUNTS,
        "anchor_prior_classes": EXPECTED_ANCHOR_PRIOR_CLASSES,
    }

    return anchor_membership, provenance


def build_gene_set_registry(
    hallmark: pd.DataFrame,
    anchors: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for module, part in hallmark.groupby(
        "hallmark_module",
        sort=True,
    ):
        genes = sorted(set(part["human_gene_symbol"].astype(str)))
        rows.append(
            {
                "set_family": "MSIGDB_HALLMARK50_LANDSCAPE",
                "set_id": module,
                "n_genes": len(genes),
                "membership_sha256": sha256_lines(genes),
                "prior_anchor_class": "",
                "analysis_role": (
                    "PRIMARY_OUTCOME_BLIND_BIOLOGICAL_DOMAIN_STRUCTURE_LANDSCAPE"
                ),
                "is_one_of_50_hallmarks": True,
                "is_paper4_anchor": False,
            }
        )

    for module, part in anchors.groupby(
        "set_id",
        sort=False,
    ):
        genes = sorted(set(part["human_gene_symbol"].astype(str)))
        prior = clean(part["prior_anchor_class"].iloc[0])
        rows.append(
            {
                "set_family": "PAPER4_FROZEN_PROGRAM_ANCHOR",
                "set_id": module,
                "n_genes": len(genes),
                "membership_sha256": sha256_lines(genes),
                "prior_anchor_class": prior,
                "analysis_role": (
                    "INDEPENDENT_OUTCOME_BLIND_STRUCTURAL_COHERENCE_ANCHOR"
                ),
                "is_one_of_50_hallmarks": False,
                "is_paper4_anchor": True,
            }
        )

    registry = pd.DataFrame(rows)

    if int(
        registry["is_one_of_50_hallmarks"].sum()
    ) != EXPECTED_HALLMARK_MODULES:
        raise RuntimeError(
            "Gene-set registry does not contain exactly 50 Hallmarks."
        )

    if int(
        registry["is_paper4_anchor"].sum()
    ) != 4:
        raise RuntimeError(
            "Gene-set registry does not contain exactly four Paper-4 anchors."
        )

    return registry


def build_hallmark_overlap_audit(
    hallmark: pd.DataFrame,
) -> pd.DataFrame:
    sets = {
        module: set(part["human_gene_symbol"].astype(str))
        for module, part in hallmark.groupby(
            "hallmark_module",
            sort=True,
        )
    }
    modules = sorted(sets)
    rows: List[Dict[str, Any]] = []

    for i, left in enumerate(modules):
        for right in modules[i + 1 :]:
            a = sets[left]
            b = sets[right]
            overlap = len(a & b)
            union = len(a | b)
            rows.append(
                {
                    "hallmark_a": left,
                    "hallmark_b": right,
                    "n_a": len(a),
                    "n_b": len(b),
                    "n_overlap": overlap,
                    "jaccard": (
                        float(overlap / union)
                        if union > 0
                        else np.nan
                    ),
                    "overlap_fraction_smaller": (
                        float(overlap / min(len(a), len(b)))
                        if min(len(a), len(b)) > 0
                        else np.nan
                    ),
                }
            )

    audit = pd.DataFrame(rows)
    expected_pairs = EXPECTED_HALLMARK_MODULES * (
        EXPECTED_HALLMARK_MODULES - 1
    ) // 2

    if len(audit) != expected_pairs:
        raise RuntimeError(
            f"Hallmark overlap pair rows={len(audit)}, expected {expected_pairs}."
        )

    return audit


def write_evidence_status() -> pd.DataFrame:
    rows = [
        {
            "manuscript_section": "3.1-3.5",
            "analysis_layer": "FROZEN_SIMULATION_HOLD_AND_CONTROLLED_MECHANISM",
            "evidence_status": (
                "CONFIRMATORY_OR_PRESPECIFIED_ACCORDING_TO_EXISTING_FROZEN_LEDGER"
            ),
            "TARGET_outcomes_used": False,
            "biological_outcomes_used": False,
            "claim_scope": (
                "primary methodological evidence and controlled mechanism localization"
            ),
        },
        {
            "manuscript_section": "3.6",
            "analysis_layer": "OUTCOME_BLIND_BIOLOGICAL_CONTEXT",
            "evidence_status": (
                "SECONDARY_POST_TARGET_OPENING_PRE_05F3F_RESULT_INSPECTION"
            ),
            "TARGET_outcomes_used": False,
            "biological_outcomes_used": False,
            "claim_scope": (
                "characterizes biological/domain structure; does not independently "
                "validate TARGET transfer performance"
            ),
        },
        {
            "manuscript_section": "3.7",
            "analysis_layer": "TARGET86_HUMAN_EVALUATION",
            "evidence_status": "DESCRIPTIVE_NON_CONFIRMATORY",
            "TARGET_outcomes_used": True,
            "biological_outcomes_used": False,
            "claim_scope": (
                "descriptive human stress test under frozen TARGET protocol"
            ),
        },
    ]
    return pd.DataFrame(rows)


def build_contract(
    state: Dict[str, Any],
    paper4_provenance: Dict[str, Any],
    registry: pd.DataFrame,
    overlap: pd.DataFrame,
    freeze_utc: str,
    f3f_snapshot: pd.DataFrame,
) -> Dict[str, Any]:
    hallmark_registry = registry[
        registry["set_family"].eq(
            "MSIGDB_HALLMARK50_LANDSCAPE"
        )
    ].copy()

    anchor_registry = registry[
        registry["set_family"].eq(
            "PAPER4_FROZEN_PROGRAM_ANCHOR"
        )
    ].copy()

    return {
        "script_version": SCRIPT_VERSION,
        "contract_version": CONTRACT_VERSION,
        "scientific_status": SCIENTIFIC_STATUS,
        "contract_freeze_utc": freeze_utc,

        "evidentiary_status": {
            "TARGET_outcomes_already_open": True,
            "05f3f_currently_or_recently_executing": True,
            "05f3f_completed_summary_present_at_freeze_boundary": False,
            "05f3f_scientific_result_read_by_05g0": False,
            "claimable_status": (
                "SECONDARY_POST_TARGET_OPENING_PRE_05F3F_RESULT_INSPECTION_OUTCOME_BLIND"
            ),
            "NOT_claimable_status": [
                "pre-TARGET prespecified",
                "confirmatory TARGET biology",
                "independent validation of TARGET performance",
            ],
        },

        "chronology_proof": {
            "05f3f_script_relative_path": str(
                F3F_SCRIPT.relative_to(ROOT).as_posix()
            ),
            "05f3f_script_sha256": state["f3f_script_sha256"],
            "05f3f_summary_relative_path": str(
                F3F_SUMMARY.relative_to(ROOT).as_posix()
            ),
            "05f3f_summary_absent_at_initial_gate": True,
            "05f3f_summary_absent_immediately_before_contract_write": True,
            "05f3f_output_filesystem_snapshot_relative_path": str(
                F3F_FILESYSTEM_SNAPSHOT.relative_to(ROOT).as_posix()
            ),
            "05f3f_output_filesystem_snapshot_sha256": sha256_file(
                F3F_FILESYSTEM_SNAPSHOT
            ),
            "snapshot_reads_scientific_contents": False,
            "snapshot_records_only": [
                "path",
                "file/directory flag",
                "size",
                "mtime",
            ],
            "snapshot_entry_count": int(len(f3f_snapshot)),
        },

        "immutable_representation_inputs": {
            "05f1a_contract_sha256": state["f1a_contract_sha256"],
            "05f1a_alignment_sha256": state["f1a_alignment_sha256"],
            "05f1a_hallmark_map_sha256": state["f1a_hallmark_sha256"],
            "05f1b_DOG2_matrix_sha256": state["f1b_dog2_matrix_sha256"],
            "05f1b_TARGET_matrix_sha256": state["f1b_target_matrix_sha256"],
            "05f1b_manifest_sha256": state["f1b_manifest_sha256"],
            "DOG2_expression_samples": EXPECTED_DOG2_SAMPLES,
            "TARGET_expression_samples": EXPECTED_TARGET_EXPRESSION_SAMPLES,
            "TARGET_survival_complete_case_intersection_used_for_3_6": False,
            "aligned_gene_universe_n": EXPECTED_ALIGNED_GENES,
            "Hallmark_modules_n": EXPECTED_HALLMARK_MODULES,
        },

        "analysis_populations": {
            "hallmark_landscape": {
                "family": "MSIGDB_HALLMARK50_LANDSCAPE",
                "n_gene_sets": EXPECTED_HALLMARK_MODULES,
                "source": (
                    "exact 05f1a TARGET_hallmark50_gene_map.tsv membership"
                ),
                "role": (
                    "primary descriptive landscape of outcome-blind cross-species "
                    "structural concordance"
                ),
                "not_independent_observations": True,
            },
            "paper4_anchors": {
                "family": "PAPER4_FROZEN_PROGRAM_ANCHOR",
                "n_gene_sets": 4,
                "sets": ANCHOR_ORDER,
                "role": (
                    "independently frozen outcome-blind coherence anchors; "
                    "not members of Hallmark-50"
                ),
                "same_statistic_as_hallmark_landscape": True,
                "same_random_control_scheme": True,
                "prior_classes": EXPECTED_ANCHOR_PRIOR_CLASSES,
                "anchor_disagreement_must_be_reported": True,
                "anchor_disagreement_may_trigger_redefinition": False,
            },
            "gene_set_registry_relative_path": str(
                SET_REGISTRY.relative_to(ROOT).as_posix()
            ),
            "gene_set_registry_sha256": sha256_file(SET_REGISTRY),
        },

        "primary_structural_statistic": {
            "name": PRIMARY_STATISTIC,
            "cohorts": ["DOG2", "TARGET_OS"],
            "samples": {
                "DOG2": EXPECTED_DOG2_SAMPLES,
                "TARGET_OS": EXPECTED_TARGET_EXPRESSION_SAMPLES,
            },
            "TARGET_outcome_complete_case_filter": False,
            "gene_set_minimum_size": MIN_GENESET_SIZE,
            "within_cohort_gene_edge_measure": "Pearson correlation",
            "edge_vector": (
                "strict upper triangle in identical sorted human-gene-symbol order"
            ),
            "cross_cohort_edge_concordance": "Spearman correlation",
            "secondary_descriptive_statistics": SECONDARY_STATISTICS,
            "gene_orientation_or_score_direction_used": False,
            "clinical_or_survival_value_used": False,
        },

        "matched_random_control_contract": {
            "n_panels_per_gene_set": N_MATCHED_RANDOM_PANELS,
            "random_seed": BIOLOGY_RANDOM_SEED,
            "sampling_universe": (
                "the exact 11,815 aligned DOG2-TARGET genes from 05f1a/05f1b"
            ),
            "tested_gene_set_excluded_from_its_own_control_pool": True,
            "sampling_without_replacement_within_panel": True,
            "sampling_independent_across_panels": True,
            "exact_gene_count_match": True,

            "gene_level_matching_covariates": [
                "DOG2 median expression within-cohort percentile rank",
                "DOG2 sample variance within-cohort percentile rank",
                "TARGET median expression within-cohort percentile rank",
                "TARGET sample variance within-cohort percentile rank",
            ],
            "why_rank_covariates": (
                "within-cohort ranks reduce dependence on incomparable absolute "
                "platform scales while still matching expression level and variability"
            ),
            "rank_tie_method": "average",
            "rank_scale": "(average_rank - 1) / (n_genes - 1)",
            "matching_bins_per_covariate": MATCHING_TERTILES,
            "bin_definition": (
                "[0,1/3), [1/3,2/3), [2/3,1] for each of four rank covariates"
            ),
            "joint_matching_strata": (
                "Cartesian product of four tertile labels: 3^4 = 81 possible strata"
            ),
            "panel_requirement": (
                "exactly reproduce the tested set's full 81-stratum gene-count vector"
            ),
            "fallback_if_any_stratum_has_insufficient_candidates": "FAIL_CLOSED",
            "nearest_neighbor_or_unmatched_fallback_allowed": False,
            "platform_difference_guard": (
                "DOG2 and TARGET expression/variance ranks enter matching separately"
            ),

            "reported_null_summary": [
                "random_control_median",
                "random_control_q05",
                "random_control_q95",
                "matched_control_percentile",
            ],
            "matched_control_percentile_formula": (
                "(1 + count(random_statistic <= observed_statistic)) / "
                "(n_random_panels + 1)"
            ),
            "matched_control_percentile_is_p_value": False,
            "multiple_testing_claim_from_50_percentiles": False,
        },

        "dependence_and_inference_guardrails": {
            "Hallmark_modules_overlap": True,
            "hallmark_overlap_audit_relative_path": str(
                OVERLAP_AUDIT.relative_to(ROOT).as_posix()
            ),
            "hallmark_overlap_audit_sha256": sha256_file(OVERLAP_AUDIT),
            "hallmark_pair_count": int(len(overlap)),
            "treat_50_Hallmarks_as_independent_n50": False,
            "module_level_regression_p_value": "FORBIDDEN",
            "module_level_correlation_p_value": "FORBIDDEN",
            "BH_or_Bonferroni_across_50_as_if_independent": "FORBIDDEN",
            "allowed_summary": [
                "rank ordering",
                "effect-size continuum",
                "matched-control percentiles",
                "overlap-aware descriptive visualization",
                "explicit anchor positions",
            ],
        },

        "anchor_coherence_rule": {
            "M34_M40_prior": (
                "strong external canine representation preservation"
            ),
            "M11_M24_prior": (
                "no clear external canine representation preservation"
            ),
            "if_new_structural_statistic_agrees": (
                "report descriptive coherence with independently frozen prior"
            ),
            "if_new_structural_statistic_disagrees_or_is_mixed": (
                "report the disagreement/mixed pattern unchanged; do not remove "
                "anchors, redefine sets, alter statistic, or search for replacement anchors"
            ),
            "coherence_check_is_formal_independent_validation": False,
        },

        "circularity_guardrail": {
            "statement": (
                "The structural concordance analysis is close to the expression "
                "structure a frozen cross-species encoder may exploit. Section 3.6 "
                "therefore characterizes the biological/domain-shift setting and "
                "must not be presented as independent proof that this statistic "
                "predicts TARGET transfer utility."
            ),
            "wording_may_be_upgraded_to_independent_predictor_claim": False,
        },

        "05f3f_module_level_diagnostic_gate": {
            "default_status": "OMIT_UNLESS_EXACT_DECOMPOSITION_IS_ALREADY_AVAILABLE",
            "permitted_only_if_all_conditions_true": [
                "the frozen 05f3f output already exposes an exact mapping to the 50 Hallmark dimensions",
                "the risk function is exactly decomposable by Hallmark without approximation",
                "no model refit is needed",
                "no parameter re-estimation is needed",
                "no new A3-style prior is constructed",
                "no new gate is constructed or hardened",
                "no post-result model choice is introduced",
            ],
            "if_any_condition_false_or_ambiguous": "DO_NOT_COMPUTE_DIAGNOSTIC",
            "N1_N2_hidden_representation_forced_decomposition": "FORBIDDEN",
            "N3_low_rank_adapter_posthoc_Hallmark_attribution": "FORBIDDEN",
            "new_real_data_A3": "FORBIDDEN",
            "omission_changes_primary_3_6_status": False,
        },

        "paper5_layer": {
            "status_at_05g0": "DEFERRED_NOT_BOUND_TO_A_SPECIFIC_ARTIFACT",
            "reason": (
                "05g0 does not guess a Paper-5 file path or import an unfrozen/ambiguous "
                "shared-private result. Paper-5 biology may be added only through one "
                "final outcome-blind frozen artifact with an explicit hash and without "
                "selection based on Paper-6 outcomes/results."
            ),
            "future_import_rule": [
                "one final Paper-5 outcome-blind frozen artifact only",
                "artifact path and SHA must be recorded before use",
                "categories/values imported verbatim",
                "no choice among multiple Paper-5 outputs based on Paper-6 results",
                "Paper-5 layer is mechanistic context, not a TARGET validation test",
            ],
            "absence_of_eligible_paper5_artifact": (
                "omit Paper-5 overlay; do not replace it with a posthoc proxy"
            ),
        },

        "null_and_figure_policy": {
            "null_or_mixed_biological_result_must_remain_reported": True,
            "metric_or_anchor_reoptimization_after_null": False,
            "gene_set_redefinition_after_null": False,
            "matched_control_redefinition_after_null": False,
            "section_3_6_if_null": (
                "retain a short main-text negative/mixed biological-context result"
            ),
            "figure_5_if_biology_not_manuscript_efficient": (
                "may return to the frozen gating/mechanism result for page economy"
            ),
            "figure_allocation_is_not_a_change_in_scientific_evidence_status": True,
        },

        "outcome_firewall": {
            "TARGET_clinical_or_outcome_values_read_by_05g0": False,
            "TARGET_05f3f_results_read_by_05g0": False,
            "GSE21257_outcomes_read": False,
            "GSE39055_outcomes_read": False,
            "DOG2_outcome_values_read": False,
            "Paper4_anchor_source_uses_outcome_blind_external_representation": True,
            "new_survival_fit": False,
            "new_model_fit": False,
            "new_feature_selection": False,
        },

        "manuscript_status_table": {
            "relative_path": str(
                EVIDENCE_STATUS.relative_to(ROOT).as_posix()
            ),
            "sha256": sha256_file(EVIDENCE_STATUS),
            "require_distinct_status_for_section_3_6": True,
        },

        "paper4_anchor_provenance": paper4_provenance,

        "next_authorized_stage": {
            "stage": "05g1",
            "purpose": (
                "execute only the frozen outcome-blind structural-context analysis "
                "defined by this 05g0 contract"
            ),
            "may_read_DOG2_expression": True,
            "may_read_TARGET_expression": True,
            "may_read_TARGET_outcomes": False,
            "may_read_GSE21257_outcomes": False,
            "may_read_GSE39055_outcomes": False,
            "may_fit_survival_model": False,
            "may_change_05f3f": False,
        },
    }


def write_readme() -> None:
    text = f"""Paper 6 05g0 biological-context freeze
================================================

Script version
--------------
{SCRIPT_VERSION}

Evidence status
---------------
SECONDARY / POST-TARGET-OPENING / PRE-05F3F-RESULT-INSPECTION /
OUTCOME-BLIND / MECHANISTIC-CONTEXT.

This is deliberately weaker than the frozen simulation/HOLD evidence in
Results 3.1-3.5 and different from the descriptive TARGET evaluation in 3.7.

What is frozen
--------------
1. All 50 05f1a MSigDB Hallmark sets form the main biological landscape.
2. M34/M40/M11/M24 are separate Paper-4 frozen program anchors, not Hallmarks.
3. Primary statistic: Spearman concordance of DOG2 versus TARGET upper-triangle
   within-gene-set Pearson correlation edges.
4. Random controls: 1,000 exact-size panels per set, matched exactly on the
   joint 4-D tertile distribution of DOG2/TARGET expression rank and variance rank.
5. The 50 Hallmarks are overlapping and are NOT treated as 50 independent samples.
6. No module-level inferential p-value treating them as independent is authorized.
7. Anchor disagreement is retained and reported.
8. Structural concordance characterizes the transfer setting; it is not
   independent proof of transfer success.
9. A 05f3f module diagnostic is omitted unless exact Hallmark decomposition is
   already available without modifying the frozen model.
10. A null/mixed biological result remains reportable.
11. GSE21257/GSE39055 outcomes remain sealed.

Chronology
----------
05g0 is permitted only while 05f3f/summary.json is absent.
No 05f3f scientific output is read by this script.
"""
    atomic_write_text(README, text)


def main() -> None:
    started = now_utc()

    print("=" * 120)
    print("Paper 6 - freeze post-opening, pre-05f3f-result outcome-blind biological-context contract")
    print("=" * 120)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Contract version: {CONTRACT_VERSION}")
    print()
    print("Critical chronology:")
    print("  TARGET outcomes already open: YES")
    print("  biological analysis was pre-TARGET frozen: NO")
    print("  intended status: POST-OPENING / PRE-05F3F-RESULT / OUTCOME-BLIND")
    print("  05f3f scientific outputs read here: NO")
    print("  completed 05f3f summary must be absent: YES")
    print()
    print("Read scope:")
    print("  TARGET clinical/outcomes: NO")
    print("  GSE21257/GSE39055 outcomes: NO")
    print("  DOG2/TARGET expression VALUES for analysis: NO")
    print("  frozen expression matrix bytes for SHA/shape verification: YES")
    print("  05f1a Hallmark memberships: YES")
    print("  Paper-4 frozen anchor memberships: YES")
    print("  Paper-4 outcome-blind external representation classes: YES")
    print()
    print("Model activity:")
    print("  model fitting: NO")
    print("  survival fitting: NO")
    print("  feature selection: NO")
    print("  A3 construction: NO")
    print("  05f3f modification: NO")
    print()

    refuse_if_already_frozen()

    # First objective chronology gate.
    assert_05f3f_result_not_available("initial 05g0 gate")

    # Capture running-05f3f filesystem metadata only. No scientific content read.
    f3f_snapshot = filesystem_snapshot(F3F_DIR)

    # Verify immutable representation and anchor definitions.
    state = verify_paper6_representation()
    anchors, paper4_provenance = verify_paper4_anchors(
        state["alignment"]
    )

    registry = build_gene_set_registry(
        state["hallmark"],
        anchors,
    )
    overlap = build_hallmark_overlap_audit(
        state["hallmark"]
    )
    evidence = write_evidence_status()

    # Create output directory only after all upstream checks have passed.
    OUT_DIR.mkdir(parents=True, exist_ok=False)

    # Write auxiliary artifacts before the final contract. These contain no
    # result-dependent scientific statistic.
    registry.to_csv(
        SET_REGISTRY,
        sep="\t",
        index=False,
    )
    overlap.to_csv(
        OVERLAP_AUDIT,
        sep="\t",
        index=False,
    )
    evidence.to_csv(
        EVIDENCE_STATUS,
        sep="\t",
        index=False,
    )
    f3f_snapshot.to_csv(
        F3F_FILESYSTEM_SNAPSHOT,
        sep="\t",
        index=False,
    )
    write_readme()

    # Second chronology gate immediately before the immutable contract write.
    # If 05f3f completed while 05g0 was checking upstream files, fail rather
    # than pretending the order was preserved.
    assert_05f3f_result_not_available(
        "immediately before 05g0 contract freeze"
    )

    freeze_utc = now_utc()
    contract = build_contract(
        state=state,
        paper4_provenance=paper4_provenance,
        registry=registry,
        overlap=overlap,
        freeze_utc=freeze_utc,
        f3f_snapshot=f3f_snapshot,
    )

    # Atomic contract write is the actual freeze boundary.
    atomic_write_json(
        CONTRACT_JSON,
        contract,
    )
    contract_stat = CONTRACT_JSON.stat()

    # It is okay if 05f3f completes AFTER the contract was atomically frozen.
    # We still do not read that summary. Record only whether it appeared after.
    f3f_summary_after_contract = F3F_SUMMARY.exists()
    f3f_summary_after_mtime_ns = None
    chronology_relation = "05F3F_SUMMARY_STILL_ABSENT_AFTER_05G0_FREEZE"

    if f3f_summary_after_contract:
        summary_stat = F3F_SUMMARY.stat()
        f3f_summary_after_mtime_ns = int(summary_stat.st_mtime_ns)

        if summary_stat.st_mtime_ns < contract_stat.st_mtime_ns:
            raise RuntimeError(
                "Chronology race detected: 05f3f summary mtime predates the "
                "05g0 contract mtime after the contract write. Do not use this "
                "05g0 freeze as PRE-RESULT evidence."
            )

        chronology_relation = (
            "05G0_CONTRACT_FROZEN_BEFORE_05F3F_SUMMARY_MTIME"
        )

    summary = {
        "script_version": SCRIPT_VERSION,
        "contract_version": CONTRACT_VERSION,
        "scientific_status": SCIENTIFIC_STATUS,
        "run_started_utc": started,
        "contract_freeze_utc": freeze_utc,
        "run_finished_utc": now_utc(),

        "script_sha256": sha256_file(Path(__file__).resolve()),
        "contract_sha256": sha256_file(CONTRACT_JSON),

        "05f3f_script_sha256": state["f3f_script_sha256"],
        "05f3f_summary_absent_at_initial_gate": True,
        "05f3f_summary_absent_immediately_before_contract_write": True,
        "05f3f_summary_present_after_contract_write": (
            f3f_summary_after_contract
        ),
        "05f3f_summary_after_contract_mtime_ns": (
            f3f_summary_after_mtime_ns
        ),
        "chronology_relation": chronology_relation,

        "TARGET_outcomes_already_open": True,
        "TARGET_outcomes_read_by_05g0": False,
        "TARGET_05f3f_results_read_by_05g0": False,
        "GSE21257_outcomes_read": False,
        "GSE39055_outcomes_read": False,
        "DOG2_outcomes_read": False,

        "Hallmark_modules": EXPECTED_HALLMARK_MODULES,
        "Paper4_anchor_programs": 4,
        "Paper4_anchors_are_Hallmark_modules": False,
        "matched_random_panels_per_set": N_MATCHED_RANDOM_PANELS,
        "Hallmarks_treated_as_independent_n50": False,
        "module_level_inferential_p_values_authorized": False,

        "model_fitting": False,
        "survival_fitting": False,
        "feature_selection": False,
        "A3_construction": False,
        "05f3f_modified": False,

        "final_artifact_hashes": {
            SET_REGISTRY.name: sha256_file(SET_REGISTRY),
            OVERLAP_AUDIT.name: sha256_file(OVERLAP_AUDIT),
            EVIDENCE_STATUS.name: sha256_file(EVIDENCE_STATUS),
            F3F_FILESYSTEM_SNAPSHOT.name: sha256_file(
                F3F_FILESYSTEM_SNAPSHOT
            ),
            README.name: sha256_file(README),
            CONTRACT_JSON.name: sha256_file(CONTRACT_JSON),
        },
        "next": (
            "05g1 may execute the exact outcome-blind biology contract after "
            "05f3f completes; 05g1 must not read TARGET/GSE21257/GSE39055 outcomes."
        ),
    }

    atomic_write_json(
        SUMMARY_JSON,
        summary,
    )

    print("=" * 120)
    print("05g0 BIOLOGICAL-CONTEXT FREEZE SUMMARY")
    print("=" * 120)
    print("Evidence status:")
    print("  pre-TARGET prespecified: NO")
    print("  post-TARGET-opening: YES")
    print("  pre-05f3f-result freeze boundary: PASS")
    print("  outcome-blind: YES")
    print("  confirmatory: NO")
    print()
    print("Gene-set populations:")
    print("  Hallmark landscape: 50")
    print("  Paper-4 anchors: 4")
    print("  anchors are four of Hallmark-50: NO")
    print("  M34/M40 prior: strong external canine preservation")
    print("  M11/M24 prior: no clear external canine preservation")
    print()
    print("Frozen structural statistic:")
    print("  within-cohort edge measure: Pearson gene-gene correlation")
    print("  cross-cohort concordance: Spearman across identical upper triangles")
    print("  TARGET survival-complete intersection: NO [all 88 expression samples]")
    print()
    print("Matched controls:")
    print(f"  panels/set: {N_MATCHED_RANDOM_PANELS}")
    print("  exact size: YES")
    print("  4-D DOG2/TARGET expression+variance tertile match: YES")
    print("  unmatched fallback: NO / FAIL_CLOSED")
    print()
    print("Inference:")
    print("  Hallmark modules treated as independent n=50: NO")
    print("  module-level inferential p-values: NO")
    print("  matched-control percentile called p-value: NO")
    print()
    print("Chronology:")
    print("  05f3f summary absent at initial gate: PASS")
    print("  05f3f summary absent immediately before contract write: PASS")
    print(f"  after contract write: {chronology_relation}")
    print()
    print("Safety:")
    print("  TARGET outcomes read: NO")
    print("  05f3f scientific results read: NO")
    print("  GSE21257/GSE39055 outcomes read: NO")
    print("  model/survival fitting: NO")
    print("  05f3f modified: NO")
    print()
    print(f"05g0 script SHA256:   {summary['script_sha256']}")
    print(f"05g0 contract SHA256: {summary['contract_sha256']}")
    print("=" * 120)
    print(f"05g0: {SCIENTIFIC_STATUS}")
    print("=" * 120)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 120, file=sys.stderr)
        print(
            "05g0 outcome-blind biological-context freeze: FAIL",
            file=sys.stderr,
        )
        print(
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        print("=" * 120, file=sys.stderr)
        raise
