#!/usr/bin/env python3
"""
Paper 6 - freeze outcome-free DOG2/TARGET representation contract.

Stage 05f1a is the first computational stage after the final textual 05f0
post-HOLD TARGET protocol freeze.

Scientific role
---------------
Freeze the EXACT outcome-free representation that will later be used for the
post-HOLD descriptive/non-confirmatory TARGET-OS evaluation.

This script:
- verifies the final 05e3b simulation stop state;
- verifies the immutable upstream DOG2/TARGET expression assets;
- verifies the exact 02g v3 transcriptome-wide outcome-blind bridge;
- verifies the exact 02h Hallmark coverage/reference;
- verifies that the 03a Hallmark source map is exactly consistent with 02h;
- reads expression HEADERS and SAMPLE-ID columns only;
- freezes the aligned 11,815-gene DOG2<->TARGET feature order;
- freezes the exact Hallmark-50 gene membership;
- freezes fold-local preprocessing / QC rules;
- machine-records the already-agreed textual 05f0 v3 TARGET model/branch guardrails.

This script DOES NOT:
- read DOG2 expression values;
- read TARGET expression values;
- read TARGET clinical values;
- read TARGET outcomes or event counts;
- read GSE21257/GSE39055 outcomes;
- generate survival splits;
- fit any source or target model;
- perform variance filtering;
- compute TARGET Hallmark scores;
- create any whole-cohort TARGET scaler.

Critical representation principle
---------------------------------
05f1b may materialize only RAW, aligned, outcome-free gene matrices.

Final TARGET Hallmark matrices are necessarily fold-dependent:
1. within each inner/outer training partition, fit gene means and population SDs;
2. remove training-zero-variance genes for that partition only;
3. apply training transformation unchanged to validation/test;
4. module score = unweighted mean of surviving mapped gene z-scores;
5. require >=10 surviving genes per Hallmark;
6. fit module means/SDs on the same training partition;
7. apply those module transforms unchanged to validation/test.

Therefore no whole-cohort TARGET Hallmark matrix is authorized before outcomes.

No CLI arguments.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


SCRIPT_VERSION = (
    "05f1a-freeze-outcome-free-target-representation-contract-v1-no-cli"
)
CONTRACT_VERSION = (
    "paper6-posthold-target-outcome-free-representation-v1"
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "_config"
PAPER4_BASENAME = "paper4_sarcoma_dog"

# ---------------------------------------------------------------------------
# Final simulation stop state.
# ---------------------------------------------------------------------------
E3B_DIR = ROOT / "results" / "posthold_controlled_mechanism" / "05e3b"
E3B_AUDIT_CONTRACT = E3B_DIR / "final_simulation_audit_contract.json"
E3B_SUMMARY = E3B_DIR / "summary.json"

EXPECTED_E3B_STATUS = (
    "PASS_FINAL_POSTHOLD_SIMULATION_MECHANISM_AUDIT_COMPLETE"
)
EXPECTED_E3B_BRANCH = "BRANCH_A_HETEROGENEITY_VALLEY"
EXPECTED_E3B_AUDIT_CONTRACT_SHA256 = (
    "58ee6398d0291bf47eb290505c82fb5e774d5a6ff06d28ad163b267c17f0c00c"
)
EXPECTED_E3B_STOP_RULE = (
    "NO_MORE_SIMULATION_MODEL_BRANCHES_OR_DIAGNOSTICS_BEFORE_METHODS_DRAFT"
)

# ---------------------------------------------------------------------------
# Immutable upstream expression assets.
# ---------------------------------------------------------------------------
UPSTREAM_LOCK = ROOT / "contracts" / "00_upstream_input_lock.json"

# ---------------------------------------------------------------------------
# 02g v3 / 02h outcome-blind representation assets.
# ---------------------------------------------------------------------------
G_DIR = ROOT / "results" / "ortholog_bridge" / "02g_v3"
G_BRIDGE = G_DIR / "primary_outcome_blind_ortholog_bridge.tsv"
G_CONTRACT = G_DIR / "ortholog_bridge_contract.json"
G_SUMMARY = G_DIR / "summary.json"

H_DIR = ROOT / "results" / "module_coverage" / "02h"
H_COVERAGE = H_DIR / "module_coverage.tsv"
H_HALLMARK = H_DIR / "reference" / "h.all.v2026.1.Hs.symbols.gmt"
H_REF_LOCK = H_DIR / "reference" / "msigdb_reference_lock.json"
H_CONTRACT = H_DIR / "module_coverage_contract.json"
H_SUMMARY = H_DIR / "summary.json"

EXPECTED_02G_STATUS = (
    "PASS_TRANSCRIPTOMEWIDE_OUTCOME_BLIND_BRIDGE_READY_FOR_MODULE_COVERAGE"
)
EXPECTED_02H_STATUS = "PASS_MODULE_REPRESENTATION_COVERAGE_READY_FOR_02I"
EXPECTED_PRIMARY_BRIDGE_N = 11815
EXPECTED_HALLMARK_N = 50
EXPECTED_HALLMARK_COMMON4_N = 49

# ---------------------------------------------------------------------------
# Exact 03a source Hallmark map.
# ---------------------------------------------------------------------------
A3_DIR = ROOT / "results" / "source_gate_protocol" / "03a"
A3_HALLMARK = A3_DIR / "hallmark_dog2_feature_map.tsv"
A3_SUMMARY = A3_DIR / "summary.json"

EXPECTED_03A_STATUS = "PASS_SOURCE_PROGNOSTIC_GATE_PROTOCOL_FROZEN"

# Existing source-gate status is recorded as provenance only.
B3_DIR = ROOT / "results" / "source_gate" / "03b"
B3_SUMMARY = B3_DIR / "summary.json"
EXPECTED_03B_STATUS = "SOURCE_AMBER_WEAK_AND_CONTEXT_SENSITIVE"

# ---------------------------------------------------------------------------
# 05f1a outputs.
# ---------------------------------------------------------------------------
OUT_DIR = ROOT / "results" / "human_posthold_descriptive" / "05f1a"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GENE_ALIGNMENT = OUT_DIR / "TARGET_primary_gene_alignment.tsv"
HALLMARK_MAP = OUT_DIR / "TARGET_hallmark50_gene_map.tsv"
DOG2_ROSTER = OUT_DIR / "DOG2_expression_sample_roster.tsv"
TARGET_ROSTER = OUT_DIR / "TARGET_expression_sample_roster.tsv"
PREPROCESSING_RULES = OUT_DIR / "TARGET_fold_safe_preprocessing_rules.tsv"
MODEL_REGISTRY = OUT_DIR / "TARGET_frozen_model_registry.tsv"
BRANCH_REGISTRY = OUT_DIR / "TARGET_frozen_interpretation_branch_registry.tsv"
REPORTING_FLAGS = OUT_DIR / "TARGET_frozen_reporting_flags.tsv"
CONTRACT_JSON = OUT_DIR / "outcome_free_TARGET_representation_contract.json"
SUMMARY_JSON = OUT_DIR / "summary.json"

# ---------------------------------------------------------------------------
# Frozen representation / future evaluation constants.
# ---------------------------------------------------------------------------
MIN_HALLMARK_GENES_PER_TRAINING_PARTITION = 10
ZERO_VARIANCE_SD_EPS = 1e-12
POPULATION_SD_DDOF = 0

# Human CV design is frozen here before TARGET outcome access. These reuse the
# already-established human premise design namespace rather than choosing seeds
# after seeing TARGET event counts.
TARGET_BASE_SEED = 20260830
OUTER_SPLITS = 5
OUTER_REPEATS = 20
INNER_SPLITS = 4

PATIENT_BOOTSTRAPS = 5000
BOOTSTRAP_SEED = TARGET_BASE_SEED + 900000

DESCRIPTIVE_MATERIALITY_REFERENCE = 0.02

# 05f1b materialization format.
MATRIX_DTYPE = "float64"
MATRIX_FORMAT = "compressed_npz_samples_by_aligned_genes"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "na", "n/a"}:
        return ""
    return text


def require_file(path: Path) -> Path:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Required frozen artifact missing: {path}")
    return path


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_lines(values: Iterable[str]) -> str:
    text = "\n".join(str(x) for x in values) + "\n"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def bool_mask(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False).astype(bool)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y", "t"})
    )


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
            (
                ROOT.parent / PAPER4_BASENAME,
                "sibling_repository",
            ),
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


def get_locked_asset(
    lock: Dict[str, Any],
    role: str,
    paper4_root: Path,
) -> Tuple[Path, Dict[str, Any]]:
    item = (lock.get("assets") or {}).get(role)

    if not isinstance(item, dict):
        raise RuntimeError(
            f"Upstream lock missing asset role {role!r}."
        )

    rel = clean(item.get("relative_path"))
    expected = clean(item.get("sha256")).lower()

    if not rel or len(expected) != 64:
        raise RuntimeError(
            f"Malformed upstream lock entry for {role!r}."
        )

    path = paper4_root / rel
    require_file(path)

    observed = sha256_file(path).lower()
    if observed != expected:
        raise RuntimeError(
            f"Locked upstream asset changed: {role}; "
            f"expected={expected}, observed={observed}"
        )

    return path, item


def read_expression_header(path: Path) -> Tuple[str, List[str]]:
    header = pd.read_csv(
        path,
        nrows=0,
    )

    if len(header.columns) < 2:
        raise RuntimeError(
            f"{path.name}: expression table has <2 columns."
        )

    sample_col = str(header.columns[0])
    features = [clean(x) for x in header.columns[1:]]

    if any(not x for x in features):
        raise RuntimeError(
            f"{path.name}: blank expression feature name."
        )

    if len(features) != len(set(features)):
        duplicated = (
            pd.Series(features)
            .loc[pd.Series(features).duplicated()]
            .unique()
            .tolist()
        )
        raise RuntimeError(
            f"{path.name}: duplicate feature names: {duplicated[:20]}"
        )

    return sample_col, features


def read_sample_ids_only(
    path: Path,
    sample_col: str,
) -> List[str]:
    # Reads only the first/sample-ID column. No expression values are loaded.
    frame = pd.read_csv(
        path,
        usecols=[sample_col],
        dtype=str,
        low_memory=False,
    ).fillna("")

    ids = [clean(x) for x in frame[sample_col]]

    if any(not x for x in ids):
        raise RuntimeError(
            f"{path.name}: blank sample ID in expression index."
        )
    if len(ids) != len(set(ids)):
        duplicated = (
            pd.Series(ids)
            .loc[pd.Series(ids).duplicated()]
            .unique()
            .tolist()
        )
        raise RuntimeError(
            f"{path.name}: duplicate sample IDs: {duplicated[:20]}"
        )

    return ids


def normalize_symbol(value: Any) -> str:
    return clean(value).upper()


def parse_mapped_genes(value: Any) -> List[str]:
    text = clean(value)
    if not text:
        return []
    return sorted(
        {
            normalize_symbol(x)
            for x in text.split(";")
            if normalize_symbol(x)
        }
    )


def verify_05e3b_stop_state() -> Dict[str, Any]:
    for path in [
        E3B_AUDIT_CONTRACT,
        E3B_SUMMARY,
    ]:
        require_file(path)

    observed_contract_hash = sha256_file(
        E3B_AUDIT_CONTRACT
    )
    if observed_contract_hash != EXPECTED_E3B_AUDIT_CONTRACT_SHA256:
        raise RuntimeError(
            "05e3b audit contract differs from the final completed freeze."
        )

    summary = read_json(E3B_SUMMARY)

    if clean(summary.get("scientific_status")) != EXPECTED_E3B_STATUS:
        raise RuntimeError(
            "05e3b is not in the expected final PASS state."
        )

    if clean(
        summary.get("audit_contract_sha256")
    ) != EXPECTED_E3B_AUDIT_CONTRACT_SHA256:
        raise RuntimeError(
            "05e3b summary/audit-contract hash mismatch."
        )

    if clean(
        summary.get("predeclared_interpretation_branch")
    ) != EXPECTED_E3B_BRANCH:
        raise RuntimeError(
            "05e3b final predeclared interpretation branch changed."
        )

    if clean(
        summary.get("stop_rule")
    ) != EXPECTED_E3B_STOP_RULE:
        raise RuntimeError(
            "05e3b final simulation stop rule changed."
        )

    if summary.get("human_outcomes_read") is not False:
        raise RuntimeError(
            "05e3b provenance unexpectedly indicates human outcome access."
        )

    return {
        "audit_contract_sha256": observed_contract_hash,
        "summary_sha256": sha256_file(E3B_SUMMARY),
        "branch": EXPECTED_E3B_BRANCH,
        "stop_rule": EXPECTED_E3B_STOP_RULE,
    }


def verify_02g_02h_03a() -> Dict[str, Any]:
    for path in [
        G_BRIDGE,
        G_CONTRACT,
        G_SUMMARY,
        H_COVERAGE,
        H_HALLMARK,
        H_REF_LOCK,
        H_CONTRACT,
        H_SUMMARY,
        A3_HALLMARK,
        A3_SUMMARY,
        B3_SUMMARY,
    ]:
        require_file(path)

    g_summary = read_json(G_SUMMARY)
    h_summary = read_json(H_SUMMARY)
    a3_summary = read_json(A3_SUMMARY)
    b3_summary = read_json(B3_SUMMARY)

    if clean(g_summary.get("status")) != "PASS":
        raise RuntimeError("02g v3 summary is not PASS.")
    if clean(
        g_summary.get("scientific_status")
    ) != EXPECTED_02G_STATUS:
        raise RuntimeError(
            "02g v3 scientific status changed."
        )

    expected_g_bridge_hash = clean(
        (g_summary.get("final_artifact_hashes") or {}).get(
            "primary_outcome_blind_ortholog_bridge_tsv"
        )
    )
    if sha256_file(G_BRIDGE) != expected_g_bridge_hash:
        raise RuntimeError("02g primary bridge hash mismatch.")

    measured = g_summary.get("measured_counts") or {}
    if int(
        measured.get("primary_dog2_to_target_os", -1)
    ) != EXPECTED_PRIMARY_BRIDGE_N:
        raise RuntimeError(
            "02g primary DOG2->TARGET feature count changed."
        )

    if clean(h_summary.get("status")) != "PASS":
        raise RuntimeError("02h summary is not PASS.")
    if clean(
        h_summary.get("scientific_status")
    ) != EXPECTED_02H_STATUS:
        raise RuntimeError(
            "02h scientific status changed."
        )

    if int(
        h_summary.get("Hallmark_primary_eligible", -1)
    ) != EXPECTED_HALLMARK_N:
        raise RuntimeError(
            "02h no longer has 50/50 primary Hallmark modules."
        )

    if int(
        h_summary.get("Hallmark_common_all_four_eligible", -1)
    ) != EXPECTED_HALLMARK_COMMON4_N:
        raise RuntimeError(
            "02h common-all-four Hallmark count changed."
        )

    h_hashes = h_summary.get("final_artifact_hashes") or {}
    expected_cov_hash = clean(
        h_hashes.get("module_coverage_tsv")
    )
    if sha256_file(H_COVERAGE) != expected_cov_hash:
        raise RuntimeError(
            "02h module coverage hash mismatch."
        )

    if clean(
        a3_summary.get("scientific_status")
    ) != EXPECTED_03A_STATUS:
        raise RuntimeError(
            "03a source representation protocol is not in expected PASS state."
        )

    expected_a3_map_hash = clean(
        (a3_summary.get("final_artifact_hashes") or {}).get(
            "hallmark_dog2_feature_map_tsv"
        )
    )
    if sha256_file(A3_HALLMARK) != expected_a3_map_hash:
        raise RuntimeError(
            "03a Hallmark source map hash mismatch."
        )

    if clean(
        b3_summary.get("scientific_status")
    ) != EXPECTED_03B_STATUS:
        raise RuntimeError(
            "03b source-gate status differs from the frozen AMBER state."
        )

    return {
        "02g_bridge_sha256": sha256_file(G_BRIDGE),
        "02g_contract_sha256": sha256_file(G_CONTRACT),
        "02g_summary_sha256": sha256_file(G_SUMMARY),
        "02h_coverage_sha256": sha256_file(H_COVERAGE),
        "02h_hallmark_gmt_sha256": sha256_file(H_HALLMARK),
        "02h_reference_lock_sha256": sha256_file(H_REF_LOCK),
        "02h_contract_sha256": sha256_file(H_CONTRACT),
        "02h_summary_sha256": sha256_file(H_SUMMARY),
        "03a_hallmark_map_sha256": sha256_file(A3_HALLMARK),
        "03a_summary_sha256": sha256_file(A3_SUMMARY),
        "03b_summary_sha256": sha256_file(B3_SUMMARY),
        "03b_source_status": EXPECTED_03B_STATUS,
    }


def build_gene_alignment(
    dog_features: Sequence[str],
    target_features: Sequence[str],
) -> pd.DataFrame:
    bridge = pd.read_csv(
        G_BRIDGE,
        sep="\t",
        dtype=str,
        low_memory=False,
    ).fillna("")

    required = {
        "dog2_raw_feature",
        "dog_gene_symbol",
        "dog_ensembl_gene_id",
        "human_gene_symbol",
        "human_ensembl_gene_id",
        "primary_dog2_to_target_os",
    }
    missing = sorted(required - set(bridge.columns))
    if missing:
        raise RuntimeError(
            f"02g bridge lacks required columns: {missing}"
        )

    primary = bridge[
        bool_mask(bridge["primary_dog2_to_target_os"])
    ].copy()

    if len(primary) != EXPECTED_PRIMARY_BRIDGE_N:
        raise RuntimeError(
            f"Primary bridge rows={len(primary)}, expected "
            f"{EXPECTED_PRIMARY_BRIDGE_N}."
        )

    primary["dog2_raw_feature"] = (
        primary["dog2_raw_feature"].astype(str).map(clean)
    )
    primary["dog_gene_symbol"] = (
        primary["dog_gene_symbol"].astype(str).map(clean)
    )
    primary["human_gene_symbol"] = (
        primary["human_gene_symbol"].map(normalize_symbol)
    )
    primary["dog_ensembl_gene_id"] = (
        primary["dog_ensembl_gene_id"].astype(str).map(clean)
    )
    primary["human_ensembl_gene_id"] = (
        primary["human_ensembl_gene_id"].astype(str).map(clean)
    )

    if primary["dog2_raw_feature"].duplicated().any():
        raise RuntimeError(
            "Primary bridge has duplicate DOG2 raw features."
        )
    if primary["human_gene_symbol"].duplicated().any():
        raise RuntimeError(
            "Primary bridge has duplicate human gene symbols."
        )

    dog_set = set(str(x) for x in dog_features)
    target_upper_to_raw: Dict[str, str] = {}

    for raw in target_features:
        upper = normalize_symbol(raw)
        if upper in target_upper_to_raw:
            raise RuntimeError(
                "TARGET expression header has a symbol collision after uppercase: "
                f"{upper}"
            )
        target_upper_to_raw[upper] = str(raw)

    missing_dog = sorted(
        set(primary["dog2_raw_feature"]) - dog_set
    )
    missing_target = sorted(
        set(primary["human_gene_symbol"])
        - set(target_upper_to_raw)
    )

    if missing_dog:
        raise RuntimeError(
            f"DOG2 expression header lacks primary feature(s): "
            f"{missing_dog[:20]}"
        )

    if missing_target:
        raise RuntimeError(
            f"TARGET expression header lacks primary human gene(s): "
            f"{missing_target[:20]}"
        )

    primary["target_expression_feature"] = (
        primary["human_gene_symbol"].map(target_upper_to_raw)
    )

    primary = primary.sort_values(
        [
            "human_gene_symbol",
            "dog_gene_symbol",
            "dog2_raw_feature",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    primary.insert(
        0,
        "aligned_feature_index",
        np.arange(len(primary), dtype=int),
    )

    primary["materialization_role"] = (
        "RAW_ALIGNED_GENE_MATRIX_ONLY"
    )
    primary["outcome_selected"] = False

    return primary[
        [
            "aligned_feature_index",
            "human_gene_symbol",
            "target_expression_feature",
            "dog_gene_symbol",
            "dog2_raw_feature",
            "human_ensembl_gene_id",
            "dog_ensembl_gene_id",
            "materialization_role",
            "outcome_selected",
        ]
    ].copy()


def build_and_verify_hallmark_map(
    alignment: pd.DataFrame,
) -> pd.DataFrame:
    a3_map = pd.read_csv(
        A3_HALLMARK,
        sep="\t",
        dtype=str,
        low_memory=False,
    ).fillna("")

    required_a3 = {
        "hallmark_module",
        "human_gene_symbol",
        "dog_gene_symbol",
        "dog2_raw_feature",
    }
    missing = sorted(required_a3 - set(a3_map.columns))
    if missing:
        raise RuntimeError(
            f"03a Hallmark map lacks required columns: {missing}"
        )

    a3_map["hallmark_module"] = (
        a3_map["hallmark_module"].astype(str).map(clean)
    )
    a3_map["human_gene_symbol"] = (
        a3_map["human_gene_symbol"].map(normalize_symbol)
    )
    a3_map["dog_gene_symbol"] = (
        a3_map["dog_gene_symbol"].astype(str).map(clean)
    )
    a3_map["dog2_raw_feature"] = (
        a3_map["dog2_raw_feature"].astype(str).map(clean)
    )

    if a3_map["hallmark_module"].nunique() != EXPECTED_HALLMARK_N:
        raise RuntimeError(
            "03a Hallmark map no longer contains exactly 50 modules."
        )

    if a3_map.duplicated(
        ["hallmark_module", "human_gene_symbol"]
    ).any():
        raise RuntimeError(
            "03a Hallmark map has duplicate module/gene membership."
        )

    # Verify that every A3 gene is in the exact 05f1a aligned universe and maps
    # to the same DOG2 raw feature.
    align_lookup = alignment.set_index(
        "human_gene_symbol"
    )

    missing_genes = sorted(
        set(a3_map["human_gene_symbol"])
        - set(align_lookup.index)
    )
    if missing_genes:
        raise RuntimeError(
            f"03a Hallmark map contains gene(s) outside primary TARGET bridge: "
            f"{missing_genes[:20]}"
        )

    for row in a3_map.itertuples(index=False):
        expected_raw = clean(
            align_lookup.loc[
                row.human_gene_symbol,
                "dog2_raw_feature",
            ]
        )
        if clean(row.dog2_raw_feature) != expected_raw:
            raise RuntimeError(
                f"03a/02g mapping disagreement for "
                f"{row.human_gene_symbol}: "
                f"03a={row.dog2_raw_feature}, 02g={expected_raw}"
            )

    # Exact independent set identity against 02h primary-Hallmark coverage.
    coverage = pd.read_csv(
        H_COVERAGE,
        sep="\t",
        low_memory=False,
    )

    required_cov = {
        "library",
        "module",
        "feature_set",
        "mapped_gene_count",
        "technical_eligibility_pass",
        "mapped_genes",
    }
    missing_cov = sorted(required_cov - set(coverage.columns))
    if missing_cov:
        raise RuntimeError(
            f"02h coverage lacks required columns: {missing_cov}"
        )

    primary_h = coverage[
        (coverage["library"].astype(str) == "HALLMARK")
        & (
            coverage["feature_set"].astype(str)
            == "primary_dog2_to_target_os"
        )
    ].copy()

    if len(primary_h) != EXPECTED_HALLMARK_N:
        raise RuntimeError(
            f"02h primary Hallmark rows={len(primary_h)}, expected 50."
        )

    if not bool_mask(
        primary_h["technical_eligibility_pass"]
    ).all():
        raise RuntimeError(
            "At least one primary Hallmark is no longer technically eligible."
        )

    a3_sets = {
        module: set(
            part["human_gene_symbol"].astype(str)
        )
        for module, part in a3_map.groupby("hallmark_module")
    }

    for row in primary_h.itertuples(index=False):
        module = clean(row.module)
        expected_genes = set(
            parse_mapped_genes(row.mapped_genes)
        )
        observed_genes = a3_sets.get(module)

        if observed_genes is None:
            raise RuntimeError(
                f"03a Hallmark map lacks 02h module {module}."
            )

        if observed_genes != expected_genes:
            raise RuntimeError(
                f"03a/02h mapped-gene set mismatch for {module}: "
                f"03a={len(observed_genes)}, 02h={len(expected_genes)}"
            )

        if len(expected_genes) != int(row.mapped_gene_count):
            raise RuntimeError(
                f"02h mapped_gene_count disagrees with mapped_genes for {module}."
            )

    # Add aligned index / TARGET raw feature and freeze order.
    aligned_meta = alignment[
        [
            "human_gene_symbol",
            "aligned_feature_index",
            "target_expression_feature",
        ]
    ]

    out = a3_map.merge(
        aligned_meta,
        on="human_gene_symbol",
        how="left",
        validate="many_to_one",
    )

    out = out.sort_values(
        [
            "hallmark_module",
            "human_gene_symbol",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    module_order = sorted(
        out["hallmark_module"].unique().tolist()
    )
    module_rank = {
        module: rank
        for rank, module in enumerate(module_order)
    }
    out.insert(
        0,
        "hallmark_module_index",
        out["hallmark_module"].map(module_rank).astype(int),
    )
    out["module_score_rule"] = (
        "training-partition gene z-scores -> unweighted mean of surviving mapped genes"
    )
    out["minimum_training_partition_genes"] = (
        MIN_HALLMARK_GENES_PER_TRAINING_PARTITION
    )

    return out[
        [
            "hallmark_module_index",
            "hallmark_module",
            "human_gene_symbol",
            "aligned_feature_index",
            "target_expression_feature",
            "dog_gene_symbol",
            "dog2_raw_feature",
            "module_score_rule",
            "minimum_training_partition_genes",
        ]
    ].copy()


def write_preprocessing_rules() -> pd.DataFrame:
    rows = [
        {
            "order": 1,
            "stage": "RAW_MATERIALIZATION_05F1B",
            "fit_scope": "NONE",
            "rule": (
                "Materialize exact aligned 11,815-gene DOG2 and TARGET raw "
                "expression matrices in identical human-symbol feature order."
            ),
            "failure_action": (
                "Any missing aligned feature, duplicate sample ID, nonnumeric "
                "value, or nonfinite value -> FAIL_CLOSED; do not drop samples/genes."
            ),
        },
        {
            "order": 2,
            "stage": "INNER_TRAIN_GENE_SCALER",
            "fit_scope": "INNER_TRAIN_ONLY",
            "rule": (
                "Compute gene mean and population SD (ddof=0) on inner-training "
                "samples only."
            ),
            "failure_action": (
                "Genes with training SD<=1e-12 are excluded for that inner split only."
            ),
        },
        {
            "order": 3,
            "stage": "INNER_VALIDATION_TRANSFORM",
            "fit_scope": "INNER_TRAIN_ONLY",
            "rule": (
                "Apply inner-training gene means/SDs unchanged to inner validation."
            ),
            "failure_action": "No validation-informed rescaling.",
        },
        {
            "order": 4,
            "stage": "INNER_HALLMARK_SCORE",
            "fit_scope": "INNER_TRAIN_ONLY",
            "rule": (
                "Each Hallmark score is the unweighted mean of surviving mapped "
                "gene z-scores."
            ),
            "failure_action": (
                "If any Hallmark has <10 surviving genes, that model/fold "
                "configuration FAILS_CLOSED; no module substitution."
            ),
        },
        {
            "order": 5,
            "stage": "INNER_MODULE_SCALER",
            "fit_scope": "INNER_TRAIN_ONLY",
            "rule": (
                "Compute 50 Hallmark means and population SDs on inner training; "
                "apply unchanged to inner validation."
            ),
            "failure_action": (
                "A nonfinite/constant Hallmark training score -> FAIL_CLOSED; "
                "do not silently remove a required module."
            ),
        },
        {
            "order": 6,
            "stage": "OUTER_TRAIN_REFIT",
            "fit_scope": "FULL_OUTER_TRAIN_ONLY",
            "rule": (
                "After hyperparameter choice, refit gene scaler, Hallmark scores, "
                "module scaler, and target-fitted model using all outer-training samples."
            ),
            "failure_action": "No outer-test information may enter refit.",
        },
        {
            "order": 7,
            "stage": "OUTER_TEST_TRANSFORM",
            "fit_scope": "FULL_OUTER_TRAIN_ONLY",
            "rule": (
                "Apply full outer-training gene/module transformations unchanged "
                "to held-out TARGET samples."
            ),
            "failure_action": (
                "No whole-cohort/transductive TARGET standardization."
            ),
        },
        {
            "order": 8,
            "stage": "SOURCE_FULL_DOG2_REPRESENTATION",
            "fit_scope": "FULL_DOG2_SOURCE_ONLY",
            "rule": (
                "For frozen full-DOG2 source models, source gene and module "
                "standardization may use all 186 source dogs; TARGET data are never "
                "used to normalize source parameters."
            ),
            "failure_action": "No new DOG2 outcome hyperparameter tuning.",
        },
        {
            "order": 9,
            "stage": "TARGET_SAMPLE_QC",
            "fit_scope": "OUTCOME_FREE_ONLY",
            "rule": (
                "05f1b sample roster is exact locked TARGET expression roster. "
                "No sample may be removed because of later outcome/model behavior."
            ),
            "failure_action": (
                "Any outcome-free data-integrity defect requiring sample deletion "
                "must be resolved before outcome access via an explicit pre-outcome amendment."
            ),
        },
    ]

    frame = pd.DataFrame(rows)
    frame.to_csv(
        PREPROCESSING_RULES,
        sep="\t",
        index=False,
    )
    return frame


def write_model_registry() -> pd.DataFrame:
    rows = [
        {
            "model_id": "T0",
            "model": "TARGET_ONLY_HALLMARK_RIDGE_COX",
            "uses_DOG2": False,
            "target_outcome_fit": True,
            "role": "PRIMARY_CLASSICAL_TARGET_ONLY_REFERENCE",
            "include": True,
        },
        {
            "model_id": "T1",
            "model": "FROZEN_DOG_HALLMARK_RIDGE_ZERO_SHOT",
            "uses_DOG2": True,
            "target_outcome_fit": False,
            "role": "CLASSICAL_DIRECT_SOURCE_COMPARATOR",
            "include": True,
        },
        {
            "model_id": "T2",
            "model": "DOG_RISK_PLUS_TARGET_RESIDUAL_COX",
            "uses_DOG2": True,
            "target_outcome_fit": True,
            "role": "CLASSICAL_RESIDUAL_TRANSFER_COMPARATOR",
            "include": True,
        },
        {
            "model_id": "N0",
            "model": "FROZEN_DOG_ENCODER_PLUS_FROZEN_DOG_HEAD",
            "uses_DOG2": True,
            "target_outcome_fit": False,
            "role": "NEURAL_ZERO_SHOT_COMPARATOR",
            "include": True,
        },
        {
            "model_id": "N1",
            "model": "FROZEN_DOG_ENCODER_PLUS_A1_TARGET_HEAD",
            "uses_DOG2": True,
            "target_outcome_fit": True,
            "role": "PRIMARY_HEAD_ADAPTATION_BRANCH",
            "include": True,
        },
        {
            "model_id": "N2",
            "model": "FROZEN_DOG_ENCODER_PLUS_FREE_TARGET_HEAD",
            "uses_DOG2": True,
            "target_outcome_fit": True,
            "role": "SOURCE_CENTERING_CONTROL",
            "include": True,
        },
        {
            "model_id": "N3",
            "model": "ORIGINAL_FROZEN_A2_LOW_RANK_ADAPTER",
            "uses_DOG2": True,
            "target_outcome_fit": True,
            "role": "ORIGINAL_A2_DESCRIPTIVE_COMPARATOR",
            "include": True,
        },
        {
            "model_id": "N4",
            "model": "TARGET_ONLY_NEURAL_SCRATCH",
            "uses_DOG2": False,
            "target_outcome_fit": True,
            "role": "NEURAL_TARGET_ONLY_FLEXIBILITY_CONTROL",
            "include": True,
        },
        {
            "model_id": "N5",
            "model": "ORIGINAL_FROZEN_A4_FULL_FINE_TUNE",
            "uses_DOG2": True,
            "target_outcome_fit": True,
            "role": "HIGH_FLEXIBILITY_ADAPTATION_CONTROL",
            "include": True,
        },
        {
            "model_id": "A3_REAL_DATA",
            "model": "MODULE_SELECTIVE_SOFT_GATE",
            "uses_DOG2": True,
            "target_outcome_fit": True,
            "role": (
                "EXCLUDED_NO_PRERESULT_REAL_EVOLUTIONARY_MODULE_PRIOR"
            ),
            "include": False,
        },
        {
            "model_id": "A3_HARD_REAL_DATA",
            "model": "HARDENED_MODULE_SELECTIVE_GATE",
            "uses_DOG2": True,
            "target_outcome_fit": True,
            "role": "EXCLUDED_INHERITS_UNAVAILABLE_REAL_A3_PRIOR",
            "include": False,
        },
    ]

    frame = pd.DataFrame(rows)
    frame.to_csv(MODEL_REGISTRY, sep="\t", index=False)
    return frame


def write_branch_registry() -> pd.DataFrame:
    rows = [
        {
            "branch": "T-A",
            "name": "CONCORDANT_SOURCE_PATTERN",
            "point_estimate_rule": (
                "C(N0)>0.50 AND deltaC(N1-N0)<+0.02 AND "
                "deltaC(N0-T0)>-0.02"
            ),
            "interpretation": (
                "Neural canine zero-shot is concordant; head adaptation adds "
                "<0.02; direct neural source reuse is not materially worse than T0."
            ),
            "synthetic_regime_label_allowed": False,
        },
        {
            "branch": "T-B",
            "name": "REVERSAL_LIKE_ADAPTATION_PATTERN",
            "point_estimate_rule": (
                "C(N0)<0.50 AND deltaC(N1-N0)>=+0.02"
            ),
            "interpretation": (
                "Direct neural canine ranking is anticoncordant and target-head "
                "adaptation materially improves it."
            ),
            "synthetic_regime_label_allowed": False,
        },
        {
            "branch": "T-C",
            "name": "WEAK_NONBENEFICIAL_SOURCE_TRANSFER_PATTERN",
            "point_estimate_rule": (
                "only if T-A/T-B not satisfied: "
                "deltaC(N0-T0)<+0.02 AND deltaC(N1-T0)<+0.02"
            ),
            "interpretation": (
                "Neither direct neural source reuse nor head adaptation "
                "materially exceeds T0."
            ),
            "synthetic_regime_label_allowed": False,
        },
        {
            "branch": "T-D",
            "name": "INCONCLUSIVE_UNRESOLVED",
            "point_estimate_rule": (
                "candidate branch becomes unresolved if any required condition "
                "is unresolved by frozen CI rules; also used if point-estimate "
                "rules do not yield a unique T-A/T-B/T-C interpretation"
            ),
            "interpretation": (
                "TARGET does not localize the canine-to-human prognostic "
                "relationship sufficiently to favor a predeclared descriptive pattern."
            ),
            "synthetic_regime_label_allowed": False,
        },
    ]

    frame = pd.DataFrame(rows)
    frame.to_csv(
        BRANCH_REGISTRY,
        sep="\t",
        index=False,
    )
    return frame


def write_reporting_flags() -> pd.DataFrame:
    rows = [
        {
            "flag": "NEURAL_DIRECT_SOURCE_MATERIAL_BENEFIT",
            "rule": "deltaC(N0-T0)>=+0.02",
            "changes_branch": False,
        },
        {
            "flag": "CLASSICAL_DIRECT_SOURCE_MATERIAL_BENEFIT",
            "rule": "deltaC(T1-T0)>=+0.02",
            "changes_branch": False,
        },
        {
            "flag": "ANY_DIRECT_SOURCE_MATERIAL_BENEFIT",
            "rule": (
                "NEURAL_DIRECT_SOURCE_MATERIAL_BENEFIT OR "
                "CLASSICAL_DIRECT_SOURCE_MATERIAL_BENEFIT"
            ),
            "changes_branch": False,
        },
        {
            "flag": "CLASSICAL_NEURAL_ZERO_SHOT_DIRECTION_DISCORDANCE",
            "rule": (
                "sign(C(T1)-0.50) != sign(C(N0)-0.50)"
            ),
            "changes_branch": False,
        },
        {
            "flag": "RETARGETED_HEAD_RESTORES_POSITIVE_CONCORDANCE",
            "rule": "C(N1)>0.50",
            "changes_branch": False,
        },
    ]

    frame = pd.DataFrame(rows)
    frame.to_csv(
        REPORTING_FLAGS,
        sep="\t",
        index=False,
    )
    return frame


def main() -> None:
    started = now_utc()

    print("=" * 120)
    print("Paper 6 - freeze outcome-free DOG2/TARGET representation contract")
    print("=" * 120)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Contract version: {CONTRACT_VERSION}")
    print()
    print("Safety / scope:")
    print("  05e3b final simulation stop state verified: YES")
    print("  DOG2 expression values read: NO")
    print("  TARGET expression values read: NO")
    print("  Expression headers read: YES")
    print("  Expression sample-ID columns read: YES")
    print("  TARGET clinical file read: NO")
    print("  TARGET outcome/event values read: NO")
    print("  GSE21257/GSE39055 outcomes read: NO")
    print("  TARGET Hallmark matrix computed: NO")
    print("  Whole-cohort TARGET scaler fit: NO")
    print("  Model fitting: NO")
    print("  Survival split generation: NO")
    print("  GPU execution: NO")
    print()

    stop_state = verify_05e3b_stop_state()
    upstream_state = verify_02g_02h_03a()

    require_file(UPSTREAM_LOCK)
    upstream_lock = read_json(UPSTREAM_LOCK)

    paper4_root, paper4_resolution = resolve_paper4_root()

    dog_path, dog_asset = get_locked_asset(
        upstream_lock,
        "dog2_expression",
        paper4_root,
    )
    target_path, target_asset = get_locked_asset(
        upstream_lock,
        "target_expression",
        paper4_root,
    )

    dog_sample_col, dog_features = read_expression_header(
        dog_path
    )
    target_sample_col, target_features = read_expression_header(
        target_path
    )

    dog_samples = read_sample_ids_only(
        dog_path,
        dog_sample_col,
    )
    target_samples = read_sample_ids_only(
        target_path,
        target_sample_col,
    )

    # Compare DOG2 expression roster against the exact frozen 03a source roster.
    source_roster_03a = pd.read_csv(
        A3_DIR / "source_sample_arm_roster.tsv",
        sep="\t",
        dtype=str,
        low_memory=False,
    ).fillna("")

    if "paper4_sample_id" not in source_roster_03a.columns:
        raise RuntimeError(
            "03a source roster lacks paper4_sample_id."
        )

    expected_dog_samples = set(
        source_roster_03a["paper4_sample_id"]
        .astype(str)
        .map(clean)
    )
    if set(dog_samples) != expected_dog_samples:
        raise RuntimeError(
            "DOG2 expression sample roster differs from frozen 03a source roster."
        )

    # Respect lock row/column identity when metadata are present.
    for cohort, samples, features, asset in [
        ("DOG2", dog_samples, dog_features, dog_asset),
        ("TARGET", target_samples, target_features, target_asset),
    ]:
        locked_rows = asset.get("n_rows")
        locked_columns = asset.get("n_columns")

        if locked_rows is not None and int(locked_rows) != len(samples):
            raise RuntimeError(
                f"{cohort}: sample count {len(samples)} differs from lock "
                f"{locked_rows}."
            )

        # n_columns in the lock may describe the full CSV including index or
        # feature columns only depending on the original lock version. We only
        # enforce it if it equals one of the two unambiguous schema counts.
        if locked_columns is not None:
            locked_columns = int(locked_columns)
            if locked_columns not in {
                len(features),
                len(features) + 1,
            }:
                raise RuntimeError(
                    f"{cohort}: locked column count {locked_columns} is "
                    f"incompatible with observed schema "
                    f"{len(features)} features + 1 sample-ID column."
                )

    alignment = build_gene_alignment(
        dog_features,
        target_features,
    )
    hallmark = build_and_verify_hallmark_map(
        alignment
    )

    if len(alignment) != EXPECTED_PRIMARY_BRIDGE_N:
        raise RuntimeError(
            "05f1a aligned gene universe is not exact 11,815."
        )

    module_counts = (
        hallmark.groupby("hallmark_module")["human_gene_symbol"]
        .nunique()
        .sort_index()
    )

    if len(module_counts) != EXPECTED_HALLMARK_N:
        raise RuntimeError(
            "05f1a Hallmark map is not exact 50 modules."
        )

    if int(module_counts.min()) < MIN_HALLMARK_GENES_PER_TRAINING_PARTITION:
        raise RuntimeError(
            "At least one structural Hallmark map has <10 genes before fold-local QC."
        )

    # Freeze rosters in their exact source-file order.
    dog_roster = pd.DataFrame(
        {
            "sample_index": np.arange(
                len(dog_samples),
                dtype=int,
            ),
            "sample_id": dog_samples,
            "cohort": "DOG2",
            "source_expression_index_column": dog_sample_col,
            "outcome_selected": False,
        }
    )
    dog_roster.to_csv(
        DOG2_ROSTER,
        sep="\t",
        index=False,
    )

    target_roster = pd.DataFrame(
        {
            "sample_index": np.arange(
                len(target_samples),
                dtype=int,
            ),
            "sample_id": target_samples,
            "cohort": "TARGET_OS",
            "source_expression_index_column": target_sample_col,
            "outcome_selected": False,
        }
    )
    target_roster.to_csv(
        TARGET_ROSTER,
        sep="\t",
        index=False,
    )

    alignment.to_csv(
        GENE_ALIGNMENT,
        sep="\t",
        index=False,
    )
    hallmark.to_csv(
        HALLMARK_MAP,
        sep="\t",
        index=False,
    )

    preprocessing = write_preprocessing_rules()
    models = write_model_registry()
    branches = write_branch_registry()
    flags = write_reporting_flags()

    # ------------------------------------------------------------------
    # Full 05f1a machine-readable contract.
    # ------------------------------------------------------------------
    contract = {
        "script_version": SCRIPT_VERSION,
        "contract_version": CONTRACT_VERSION,
        "status": "PASS",
        "scientific_status": (
            "PASS_OUTCOME_FREE_TARGET_REPRESENTATION_CONTRACT_FROZEN"
        ),
        "created_utc": now_utc(),

        "05e3b_final_state": stop_state,

        "05f0_textual_freeze": {
            "version": "05f0-text-freeze-v3",
            "scientific_role": (
                "TARGET-OS is a post-HOLD descriptive/non-confirmatory "
                "human mechanistic stress test."
            ),
            "original_05d_HOLD_may_change": False,
            "A6_may_reopen": False,
            "model_selection_on_TARGET": False,
            "GSE21257_outcomes_remain_sealed": True,
            "GSE39055_outcomes_remain_sealed": True,
            "primary_endpoint": "OS",
            "secondary_endpoints_may_replace_OS": False,
            "descriptive_materiality_reference_delta_C": (
                DESCRIPTIVE_MATERIALITY_REFERENCE
            ),
            "materiality_reference_is_05a_positive_gate": False,
        },

        "upstream_representation_identity": upstream_state,

        "locked_expression_assets": {
            "paper4_root_resolution_source": paper4_resolution,
            "absolute_paper4_path_recorded": False,
            "DOG2": {
                "relative_path": clean(dog_asset.get("relative_path")),
                "sha256": clean(dog_asset.get("sha256")),
                "sample_id_column": dog_sample_col,
                "n_samples": len(dog_samples),
                "n_expression_features": len(dog_features),
                "sample_roster_sha256": sha256_file(DOG2_ROSTER),
            },
            "TARGET_OS": {
                "relative_path": clean(target_asset.get("relative_path")),
                "sha256": clean(target_asset.get("sha256")),
                "sample_id_column": target_sample_col,
                "n_expression_samples_before_outcome_intersection": (
                    len(target_samples)
                ),
                "n_expression_features": len(target_features),
                "sample_roster_sha256": sha256_file(TARGET_ROSTER),
            },
        },

        "aligned_gene_representation": {
            "n_pairs": len(alignment),
            "ordering": (
                "lexicographic human_gene_symbol, then dog_gene_symbol, "
                "then dog2_raw_feature"
            ),
            "alignment_artifact": str(
                GENE_ALIGNMENT.relative_to(ROOT)
            ),
            "alignment_sha256": sha256_file(GENE_ALIGNMENT),
            "human_gene_order_sha256": sha256_lines(
                alignment["human_gene_symbol"].tolist()
            ),
            "dog2_raw_feature_order_sha256": sha256_lines(
                alignment["dog2_raw_feature"].tolist()
            ),
            "one_to_one": True,
            "post_outcome_feature_change_allowed": False,
        },

        "hallmark_representation": {
            "library": "MSigDB Hallmark 2026.1.Hs",
            "n_modules": EXPECTED_HALLMARK_N,
            "module_order": sorted(
                hallmark["hallmark_module"].unique().tolist()
            ),
            "gene_map_artifact": str(
                HALLMARK_MAP.relative_to(ROOT)
            ),
            "gene_map_sha256": sha256_file(HALLMARK_MAP),
            "mapped_gene_rows": int(len(hallmark)),
            "min_structural_genes_per_module": int(
                module_counts.min()
            ),
            "median_structural_genes_per_module": float(
                module_counts.median()
            ),
            "max_structural_genes_per_module": int(
                module_counts.max()
            ),
            "score_rule": (
                "unweighted mean of training-transform gene z-scores"
            ),
            "minimum_surviving_genes_per_training_partition": (
                MIN_HALLMARK_GENES_PER_TRAINING_PARTITION
            ),
            "whole_TARGET_module_matrix_before_splits": "FORBIDDEN",
        },

        "05f1b_materialization_contract": {
            "read_expression_values": True,
            "read_TARGET_outcomes": False,
            "read_TARGET_clinical_values": False,
            "cohorts_to_materialize": [
                "DOG2",
                "TARGET_OS",
            ],
            "GSE21257_expression_materialized": False,
            "GSE39055_expression_materialized": False,
            "matrix_format": MATRIX_FORMAT,
            "matrix_dtype": MATRIX_DTYPE,
            "matrix_orientation": "samples_x_aligned_genes",
            "exact_gene_order": (
                "TARGET_primary_gene_alignment.tsv aligned_feature_index"
            ),
            "allowed_transformations": [
                "numeric parse to float64",
                "exact column reordering to frozen alignment",
            ],
            "forbidden_transformations": [
                "imputation",
                "whole-cohort scaling",
                "variance filtering",
                "module scoring",
                "PCA",
                "CORAL",
                "outcome-based sample restriction",
                "post-result feature removal",
            ],
            "nonfinite_value_action": "FAIL_CLOSED",
            "duplicate_sample_action": "FAIL_CLOSED",
            "missing_gene_action": "FAIL_CLOSED",
            "sample_deletion_action": "FORBIDDEN",
        },

        "fold_safe_preprocessing": {
            "artifact": str(
                PREPROCESSING_RULES.relative_to(ROOT)
            ),
            "artifact_sha256": sha256_file(
                PREPROCESSING_RULES
            ),
            "gene_sd_ddof": POPULATION_SD_DDOF,
            "zero_variance_eps": ZERO_VARIANCE_SD_EPS,
            "nested_inner_preprocessing": True,
            "outer_train_refit_after_tuning": True,
            "whole_cohort_target_scaling": False,
            "target_test_used_for_scaling": False,
            "post_outcome_QC_exclusion": False,
        },

        "future_TARGET_evaluation_design": {
            "outer_splits": OUTER_SPLITS,
            "outer_repeats": OUTER_REPEATS,
            "inner_splits_classical": INNER_SPLITS,
            "base_seed": TARGET_BASE_SEED,
            "outer_stratification": "OS_event",
            "same_outer_splits_all_models": True,
            "split_generation_in_05f1a": False,
            "patient_bootstraps": PATIENT_BOOTSTRAPS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_unit": "patient_id",
            "bootstrap_pairing": "same patient multiplicity across all models",
            "bootstrap_stratification": "primary OS event status",
            "bootstrap_model_refit": False,
            "repeated_CV_pseudoreplication_forbidden": True,
            "implementation_detail_before_outcome_open": (
                "05f1c must freeze exact repeated-OOF aggregation and bootstrap "
                "propagation before TARGET outcome values are read."
            ),
        },

        "TARGET_model_registry": {
            "artifact": str(
                MODEL_REGISTRY.relative_to(ROOT)
            ),
            "artifact_sha256": sha256_file(
                MODEL_REGISTRY
            ),
            "included_model_ids": (
                models.loc[models["include"], "model_id"]
                .astype(str)
                .tolist()
            ),
            "A3_real_data_included": False,
            "post_outcome_model_addition_allowed": False,
        },

        "TARGET_interpretation_branches": {
            "artifact": str(
                BRANCH_REGISTRY.relative_to(ROOT)
            ),
            "artifact_sha256": sha256_file(
                BRANCH_REGISTRY
            ),
            "assignment_arm": "NEURAL_N0_N1_WITH_T0_REFERENCE",
            "T_D_is_valid_expected_outcome": True,
            "CI_orientation_unresolved_rule": (
                "If 95% CI for C(N0) includes 0.50 and orientation is needed "
                "for candidate branch T-A/T-B, final branch=T-D."
            ),
            "CI_delta_unresolved_rule": (
                "If 95% CI for a required branch-defining deltaC contains "
                "both -0.02 and +0.02, that condition is unresolved; "
                "candidate branch becomes T-D."
            ),
            "subjective_uncertainty_override_allowed": False,
            "synthetic_R_label_assignment_to_TARGET_allowed": False,
        },

        "TARGET_reporting_flags": {
            "artifact": str(
                REPORTING_FLAGS.relative_to(ROOT)
            ),
            "artifact_sha256": sha256_file(
                REPORTING_FLAGS
            ),
            "flags_change_branch": False,
        },

        "classical_neural_zero_shot_policy": {
            "branch_assignment_uses": "N0/N1 neural mechanism arm",
            "T1_classical_zero_shot_is_reported": True,
            "direction_disagreement_flagged": True,
            "direction_disagreement_resolved_posthoc": False,
        },

        "endpoint_firewall": {
            "TARGET_clinical_file_read_in_05f1a": False,
            "TARGET_outcome_values_read_in_05f1a": False,
            "primary_endpoint": "OS",
            "secondary_endpoint_upgrade_after_opening": "FORBIDDEN",
            "exact_TARGET_endpoint_columns": (
                "DEFERRED_TO_05F1C_CLINICAL_HEADER_ONLY_FREEZE"
            ),
        },

        "forbidden_after_first_TARGET_outcome_access": [
            "change orthology mapping",
            "change Hallmark membership or coverage thresholds",
            "change raw sample QC/exclusion rules",
            "change scaling policy",
            "change model registry",
            "add sign-orientation selector",
            "construct a real-data A3 evolutionary prior",
            "add a new gate-hardening rule",
            "change resampling/bootstrap/CI aggregation scheme",
            "change OS primary endpoint",
            "exclude samples/folds/repeats for QC discovered after outcomes",
            "open GSE21257 outcomes to explain TARGET",
            "open GSE39055 outcomes to explain TARGET",
            "hide an included model because it performs poorly",
        ],

        "safety": {
            "DOG2_expression_values_read": False,
            "TARGET_expression_values_read": False,
            "expression_headers_read": True,
            "expression_sample_ids_read": True,
            "TARGET_clinical_values_read": False,
            "TARGET_outcomes_read": False,
            "GSE21257_outcomes_read": False,
            "GSE39055_outcomes_read": False,
            "TARGET_Hallmark_scores_computed": False,
            "whole_cohort_TARGET_scaler_fit": False,
            "survival_splits_generated": False,
            "model_fitting": False,
            "GPU_execution": False,
        },

        "next_required_stage": (
            "05f1b materialize exact raw aligned DOG2/TARGET gene matrices "
            "under this contract; still do not read TARGET clinical/outcome values."
        ),
    }

    write_json(
        CONTRACT_JSON,
        contract,
    )

    final_hashes = {
        "TARGET_primary_gene_alignment.tsv": sha256_file(
            GENE_ALIGNMENT
        ),
        "TARGET_hallmark50_gene_map.tsv": sha256_file(
            HALLMARK_MAP
        ),
        "DOG2_expression_sample_roster.tsv": sha256_file(
            DOG2_ROSTER
        ),
        "TARGET_expression_sample_roster.tsv": sha256_file(
            TARGET_ROSTER
        ),
        "TARGET_fold_safe_preprocessing_rules.tsv": sha256_file(
            PREPROCESSING_RULES
        ),
        "TARGET_frozen_model_registry.tsv": sha256_file(
            MODEL_REGISTRY
        ),
        "TARGET_frozen_interpretation_branch_registry.tsv": sha256_file(
            BRANCH_REGISTRY
        ),
        "TARGET_frozen_reporting_flags.tsv": sha256_file(
            REPORTING_FLAGS
        ),
        "outcome_free_TARGET_representation_contract.json": sha256_file(
            CONTRACT_JSON
        ),
    }

    summary = {
        "script_version": SCRIPT_VERSION,
        "contract_version": CONTRACT_VERSION,
        "status": "PASS",
        "scientific_status": (
            "PASS_OUTCOME_FREE_TARGET_REPRESENTATION_CONTRACT_FROZEN"
        ),
        "run_started_utc": started,
        "run_finished_utc": now_utc(),
        "05e3b_branch": EXPECTED_E3B_BRANCH,
        "05e3b_stop_rule": EXPECTED_E3B_STOP_RULE,
        "DOG2_samples": len(dog_samples),
        "TARGET_expression_samples_before_outcome_intersection": (
            len(target_samples)
        ),
        "aligned_gene_pairs": len(alignment),
        "Hallmark_modules": len(module_counts),
        "Hallmark_mapped_gene_rows": int(len(hallmark)),
        "Hallmark_min_structural_genes": int(
            module_counts.min()
        ),
        "Hallmark_median_structural_genes": float(
            module_counts.median()
        ),
        "Hallmark_max_structural_genes": int(
            module_counts.max()
        ),
        "TARGET_outcomes_read": False,
        "TARGET_clinical_values_read": False,
        "DOG2_expression_values_read": False,
        "TARGET_expression_values_read": False,
        "model_fitting": False,
        "survival_splits_generated": False,
        "final_artifact_hashes": final_hashes,
        "contract_sha256": sha256_file(
            CONTRACT_JSON
        ),
        "next": contract["next_required_stage"],
    }

    write_json(
        SUMMARY_JSON,
        summary,
    )

    print("=" * 120)
    print("05f1a OUTCOME-FREE TARGET REPRESENTATION SUMMARY")
    print("=" * 120)
    print(f"05e3b branch: {EXPECTED_E3B_BRANCH}")
    print("05e3b stop rule verification: PASS")
    print(f"Paper4 root resolution: {paper4_resolution}")
    print()
    print("Locked expression metadata:")
    print(
        f"  DOG2: {len(dog_samples):,} samples, "
        f"{len(dog_features):,} expression features"
    )
    print(
        f"  TARGET: {len(target_samples):,} expression samples, "
        f"{len(target_features):,} expression features"
    )
    print()
    print("Frozen cross-species representation:")
    print(
        f"  aligned DOG2<->TARGET genes: {len(alignment):,}"
    )
    print(
        f"  Hallmark modules: {len(module_counts)}/{EXPECTED_HALLMARK_N}"
    )
    print(
        "  mapped genes/module "
        f"min/median/max: "
        f"{int(module_counts.min())}/"
        f"{float(module_counts.median()):.1f}/"
        f"{int(module_counts.max())}"
    )
    print("  03a <-> 02h exact per-module mapped-gene identity: PASS")
    print()
    print("Future TARGET preprocessing:")
    print("  raw aligned matrix materialization first: YES")
    print("  whole-cohort TARGET Hallmark matrix before splits: NO")
    print("  inner-training-only gene/module scaling: YES")
    print("  outer-training refit and held-out transform: YES")
    print("  post-outcome sample QC deletion: NO")
    print()
    print("TARGET textual 05f0 guardrails machine-recorded: YES")
    print("  allowed models: T0-T2, N0-N5")
    print("  real-data A3/hardened A3: EXCLUDED")
    print("  branches: T-A / T-B / T-C / T-D")
    print("  T-D valid outcome: YES")
    print()
    print("TARGET clinical file read: NO")
    print("TARGET outcomes read: NO")
    print("DOG2 expression values read: NO")
    print("TARGET expression values read: NO")
    print("Model fitting: NO")
    print("Survival split generation: NO")
    print("GPU execution: NO")
    print()
    print(f"05f1a contract SHA256: {sha256_file(CONTRACT_JSON)}")
    print()
    print("Next:")
    print(
        "  05f1b materialize exact raw aligned DOG2/TARGET gene matrices."
    )
    print("  TARGET clinical/outcome values remain CLOSED.")
    print("=" * 120)
    print(
        "05f1a: PASS_OUTCOME_FREE_TARGET_REPRESENTATION_CONTRACT_FROZEN"
    )
    print("=" * 120)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 120, file=sys.stderr)
        print(
            "05f1a outcome-free TARGET representation freeze: FAIL",
            file=sys.stderr,
        )
        print(
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        print("=" * 120, file=sys.stderr)
        raise
