#!/usr/bin/env python3
"""
Paper 6 - preflight audit of the 02f0 transport feature-universe provenance.

Purpose
-------
This is a PRE-AMENDMENT audit. It does not change or supersede 02f0 itself.

It measures, from the local frozen repositories, whether the feature universe
used by Paper-6 script 02f0 inherits the row universe of the Paper-4
outcome-aware RNA candidate/evidence table.

The audit establishes the lineage:

Paper-4 outcome-aware RNA evidence inputs
    -> GSE238110_RNA_master_candidate_evidence_table.csv
    -> ..._with_orthologs.csv
    -> ..._with_ortholog_qc.csv
    -> Paper-6 02f0 strict ortholog universe / primary feature set

It also compares this candidate-restricted lineage with the broader, outcome-blind
Ensembl BioMart dog-human ortholog cache that was downloaded upstream.

Safety
------
This script:
- reads NO clinical table;
- reads NO outcome/response/follow-up values;
- reads NO treatment-administration values;
- reads NO expression matrix values (header only);
- fits NO model;
- changes NO existing contract;
- performs NO network access.

Only gene identifiers, table headers, source-code text, hashes, and pre-existing
02f0 metadata/universe artifacts are read.

No command-line arguments are used.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd


SCRIPT_VERSION = "02f0-preflight-audit-transport-universe-v3-no-cli"

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "_config"

PAPER4_BASENAME = "paper4_sarcoma_dog"

OUT_DIR = ROOT / "results" / "transport_contract" / "02f0_preflight_audit_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LINEAGE_SUMMARY_TSV = OUT_DIR / "universe_lineage_summary.tsv"
SOURCE_LINEAGE_TSV = OUT_DIR / "source_lineage_evidence.tsv"
SET_AUDIT_TSV = OUT_DIR / "gene_set_lineage_audit.tsv"
AUDIT_JSON = OUT_DIR / "universe_lineage_audit.json"
SUMMARY_JSON = OUT_DIR / "summary.json"

# Paper-6 02f0 artifacts.
P6_02F0_DIR = ROOT / "results" / "transport_contract" / "02f0"
P6_02F0_UNIVERSE = P6_02F0_DIR / "strict_ortholog_universe.tsv"
P6_02F0_FEATURE_SUMMARY = P6_02F0_DIR / "feature_set_summary.tsv"
P6_02F0_CONTRACT = P6_02F0_DIR / "cross_species_transport_contract.json"
P6_02F0_SUMMARY = P6_02F0_DIR / "summary.json"
P6_02F0_SCRIPT = ROOT / "scripts" / "02f0_freeze_cross_species_transport_contract.py"
P6_UPSTREAM_LOCK = ROOT / "contracts" / "00_upstream_input_lock.json"

# Paper-4 artifacts.
P4_DOG2_EXPRESSION = (
    Path("data") / "processed" / "GSE238110_DOG2_expression_log2cpm_matched_allgenes.csv"
)
P4_MASTER = (
    Path("results") / "tables" / "GSE238110_RNA_master_candidate_evidence_table.csv"
)
P4_MASTER_ORTHOLOGS = (
    Path("results") / "tables" / "GSE238110_RNA_master_candidate_evidence_table_with_orthologs.csv"
)
P4_ORTHOLOG_QC = (
    Path("results") / "tables" / "GSE238110_RNA_master_candidate_evidence_table_with_ortholog_qc.csv"
)
P4_BIOMART_CACHE = (
    Path("data") / "external" / "ensembl_dog_human_orthologs_biomart.tsv"
)

P4_SCRIPT12 = Path("scripts") / "12_build_rna_master_candidate_evidence_table.py"
P4_SCRIPT15 = Path("scripts") / "15_ortholog_mapping_dog_to_human.py"
P4_SCRIPT17 = Path("scripts") / "17_ortholog_mapping_qc_transfer_sets.py"

STRICT_STATUS = "strict_symbol_concordant_one_to_one"

OUTCOME_AWARE_INPUT_TOKENS = [
    "GSE238110_dfi_univariate_cox_top5000var.csv",
    "GSE238110_os_univariate_cox_top5000var.csv",
    "GSE238110_mb_candidate_genes_from_univariate_cox.csv",
    "GSE238110_dfi_conditional_cox_mb_selected.csv",
    "GSE238110_os_conditional_cox_mb_selected.csv",
    "GSE238110_true_iamb_gsmb_ablation_selected_genes.csv",
    "GSE238110_nested_cv_selected_genes.csv",
    "GSE238110_nested_cv_selected_gene_stability.csv",
    "GSE238110_nested_cv_method_benchmark_summary.csv",
    "GSE238110_nested_cv_method_vs_random_percentile_summary.csv",
]

EXPECTED_LINEAGE_TOKENS = {
    "script12_output_master": "GSE238110_RNA_master_candidate_evidence_table.csv",
    "script15_input_master": "GSE238110_RNA_master_candidate_evidence_table.csv",
    "script15_output_orthologs": "GSE238110_RNA_master_candidate_evidence_table_with_orthologs.csv",
    "script17_input_orthologs": "GSE238110_RNA_master_candidate_evidence_table_with_orthologs.csv",
    "script17_output_qc": "GSE238110_RNA_master_candidate_evidence_table_with_ortholog_qc.csv",
    "script02f0_input_qc": "GSE238110_RNA_master_candidate_evidence_table_with_ortholog_qc.csv",
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_lines(values: Iterable[str]) -> str:
    text = "\n".join(sorted(str(x) for x in values)) + "\n"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "na", "n/a"}:
        return ""
    return text


def exact_gene_set_from_csv(path: Path, gene_col: str = "gene") -> Tuple[int, set[str]]:
    header = pd.read_csv(path, nrows=0)
    if gene_col not in header.columns:
        raise RuntimeError(
            f"{path.name}: required gene column {gene_col!r} not found. "
            f"Available columns: {list(header.columns)[:30]}"
        )
    values = pd.read_csv(
        path,
        usecols=[gene_col],
        dtype=str,
        low_memory=False,
    )[gene_col].map(clean)
    genes = {x for x in values if x}
    return int(len(values)), genes


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y", "t"})
    )


def read_text_required(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Required source file missing: {path}")
    return path.read_text(encoding="utf-8", errors="replace")


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
        candidates.append((Path(env_value).expanduser(), "environment:PAPER4_ROOT"))

    config_value = clean(local_path_config().get("paper4_root"))
    if config_value:
        candidates.append((Path(config_value).expanduser(), "_config/paths.local.json"))

    candidates.extend(
        [
            (ROOT.parent / PAPER4_BASENAME, "sibling_repository"),
            (Path.home() / "Desktop" / PAPER4_BASENAME, "home_desktop_fallback"),
        ]
    )

    checked: List[str] = []
    for candidate, source in candidates:
        resolved = candidate.resolve()
        checked.append(f"{source}: {resolved}")
        sentinel = resolved / P4_DOG2_EXPRESSION
        if resolved.is_dir() and sentinel.exists():
            return resolved, source

    raise FileNotFoundError(
        "Could not resolve Paper 4 root. Checked:\n  - " + "\n  - ".join(checked)
    )


def require_file(path: Path) -> Path:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Required file missing: {path}")
    return path


def source_token_row(
    source_name: str,
    source_path: Path,
    source_text: str,
    token: str,
    evidence_role: str,
) -> Dict[str, Any]:
    return {
        "source": source_name,
        "source_file": source_path.name,
        "source_sha256": sha256_file(source_path),
        "evidence_role": evidence_role,
        "token": token,
        "token_present": token in source_text,
    }


def detect_biomart_columns(frame: pd.DataFrame) -> Tuple[str, str, str]:
    symbol_candidates = [
        "external_gene_name",
        "dog_gene_symbol",
        "canine_gene_symbol",
    ]
    human_candidates = [
        "hsapiens_homolog_associated_gene_name",
        "human_gene_symbol",
    ]
    type_candidates = [
        "hsapiens_homolog_orthology_type",
        "dog_human_orthology_type",
        "orthology_type",
    ]

    symbol_col = next((c for c in symbol_candidates if c in frame.columns), None)
    human_col = next((c for c in human_candidates if c in frame.columns), None)
    type_col = next((c for c in type_candidates if c in frame.columns), None)

    if symbol_col is None or human_col is None or type_col is None:
        raise RuntimeError(
            "Could not identify expected BioMart columns. "
            f"Observed columns: {list(frame.columns)}"
        )
    return symbol_col, human_col, type_col


def main() -> None:
    started = now_utc()

    print("=" * 118)
    print("Paper 6 - preflight audit of 02f0 transport feature-universe provenance")
    print("=" * 118)
    print(f"Script version: {SCRIPT_VERSION}")
    print()
    print("Safety / execution contract:")
    print("  Network access: NO")
    print("  Clinical table values read: NO")
    print("  Outcome/response/follow-up values read: NO")
    print("  Treatment-administration values read: NO")
    print("  Expression matrix values read: NO [header only]")
    print("  Gene identifiers / table headers read: YES")
    print("  Source-code text read: YES")
    print("  Model fitting: NO")
    print("  Existing 02f0 contract modified: NO")
    print()

    paper4_root, paper4_source = resolve_paper4_root()
    print(f"Paper 4 root: {paper4_root}")
    print(f"Resolution source: {paper4_source}")
    print()

    p4_expr = require_file(paper4_root / P4_DOG2_EXPRESSION)
    p4_master = require_file(paper4_root / P4_MASTER)
    p4_master_orthologs = require_file(paper4_root / P4_MASTER_ORTHOLOGS)
    p4_qc = require_file(paper4_root / P4_ORTHOLOG_QC)
    p4_biomart = require_file(paper4_root / P4_BIOMART_CACHE)

    p4_s12 = require_file(paper4_root / P4_SCRIPT12)
    p4_s15 = require_file(paper4_root / P4_SCRIPT15)
    p4_s17 = require_file(paper4_root / P4_SCRIPT17)

    for path in [
        P6_02F0_UNIVERSE,
        P6_02F0_FEATURE_SUMMARY,
        P6_02F0_CONTRACT,
        P6_02F0_SUMMARY,
        P6_02F0_SCRIPT,
        P6_UPSTREAM_LOCK,
    ]:
        require_file(path)

    p6_summary = read_json(P6_02F0_SUMMARY)
    p6_contract = read_json(P6_02F0_CONTRACT)
    p6_upstream_lock = read_json(P6_UPSTREAM_LOCK)
    if p6_summary.get("status") != "PASS" or p6_contract.get("status") != "PASS":
        raise RuntimeError("Existing 02f0 artifacts are not PASS; provenance audit aborted.")

    upstream_assets = p6_upstream_lock.get("assets") or {}
    locked_ortholog_asset = upstream_assets.get("ortholog_qc")
    if not isinstance(locked_ortholog_asset, dict):
        raise RuntimeError("00 upstream lock lacks ortholog_qc asset metadata.")

    locked_ortholog_relative_path = clean(locked_ortholog_asset.get("relative_path"))
    locked_ortholog_sha256 = clean(locked_ortholog_asset.get("sha256")).lower()
    actual_p4_qc_sha256 = sha256_file(p4_qc).lower()

    p6_required_hashes = ((p6_contract.get("upstream") or {}).get("required_asset_hashes") or {})
    p6_contract_ortholog_sha256 = clean(p6_required_hashes.get("ortholog_qc")).lower()

    ortholog_lock_path_identity = (
        Path(locked_ortholog_relative_path).as_posix()
        == P4_ORTHOLOG_QC.as_posix()
    )
    ortholog_lock_hash_identity = (
        len(locked_ortholog_sha256) == 64
        and locked_ortholog_sha256 == actual_p4_qc_sha256
    )
    ortholog_02f0_contract_hash_identity = (
        len(p6_contract_ortholog_sha256) == 64
        and p6_contract_ortholog_sha256 == actual_p4_qc_sha256
    )

    # DOG2 transcriptome header only.
    dog2_header = pd.read_csv(p4_expr, nrows=0, index_col=0)
    dog2_features = [str(x).strip() for x in dog2_header.columns]
    dog2_feature_set = {x for x in dog2_features if x}
    if len(dog2_features) != len(dog2_feature_set):
        raise RuntimeError("DOG2 expression header contains duplicate feature identifiers.")

    # Exact candidate/evidence -> ortholog-QC row-universe equality.
    master_rows, master_genes = exact_gene_set_from_csv(p4_master, "gene")
    ortholog_rows, ortholog_genes = exact_gene_set_from_csv(p4_master_orthologs, "gene")
    qc_rows, qc_genes = exact_gene_set_from_csv(p4_qc, "gene")

    master_equals_ortholog = master_genes == ortholog_genes
    master_equals_qc = master_genes == qc_genes
    master_subset_expression = master_genes.issubset(dog2_feature_set)

    qc_header = pd.read_csv(p4_qc, nrows=0)
    for required in ["gene", "human_gene_symbol", "ortholog_qc_status"]:
        if required not in qc_header.columns:
            raise RuntimeError(f"{p4_qc.name}: missing required column {required}")

    qc_map = pd.read_csv(
        p4_qc,
        usecols=["gene", "human_gene_symbol", "ortholog_qc_status"],
        dtype=str,
        low_memory=False,
    ).fillna("")
    strict_qc = qc_map[
        qc_map["ortholog_qc_status"].astype(str).eq(STRICT_STATUS)
    ].copy()
    strict_qc_genes = {
        clean(x) for x in strict_qc["gene"].tolist() if clean(x)
    }

    # Broader outcome-blind BioMart mapping snapshot.
    biomart = pd.read_csv(p4_biomart, sep="\t", dtype=str, low_memory=False).fillna("")
    symbol_col, human_col, type_col = detect_biomart_columns(biomart)

    dog_symbol = biomart[symbol_col].astype(str).str.strip()
    human_symbol = biomart[human_col].astype(str).str.strip()
    orthology_type = biomart[type_col].astype(str).str.strip().str.lower()

    valid_dog = dog_symbol.ne("")
    any_human = valid_dog & human_symbol.ne("")
    one2one = any_human & orthology_type.str.contains("one2one", regex=False)

    biomart_unique_dog_symbols = set(dog_symbol[valid_dog])
    biomart_any_human_symbols = set(dog_symbol[any_human])
    biomart_one2one_dog_symbols = set(dog_symbol[one2one])

    # Existing Paper-6 02f0 universe inheritance.
    p6_universe = pd.read_csv(P6_02F0_UNIVERSE, sep="\t", dtype=str, low_memory=False)
    if "canine_gene" not in p6_universe.columns:
        raise RuntimeError("02f0 strict_ortholog_universe.tsv lacks canine_gene.")

    p6_strict_genes = {
        clean(x) for x in p6_universe["canine_gene"].tolist() if clean(x)
    }

    if "primary_dog2_to_target_os" not in p6_universe.columns:
        raise RuntimeError(
            "02f0 strict_ortholog_universe.tsv lacks primary_dog2_to_target_os."
        )
    primary_mask = bool_series(p6_universe["primary_dog2_to_target_os"])
    p6_primary_genes = {
        clean(x)
        for x in p6_universe.loc[primary_mask, "canine_gene"].tolist()
        if clean(x)
    }

    p6_strict_subset_master = p6_strict_genes.issubset(master_genes)
    p6_primary_subset_master = p6_primary_genes.issubset(master_genes)
    p6_strict_equals_qc_strict = p6_strict_genes == strict_qc_genes

    # Source-code lineage: no inference from filenames alone.
    s12_text = read_text_required(p4_s12)
    s15_text = read_text_required(p4_s15)
    s17_text = read_text_required(p4_s17)
    p602f0_text = read_text_required(P6_02F0_SCRIPT)

    source_rows: List[Dict[str, Any]] = []

    for token in OUTCOME_AWARE_INPUT_TOKENS:
        source_rows.append(
            source_token_row(
                "paper4_script12",
                p4_s12,
                s12_text,
                token,
                "outcome_or_model_derived_input_to_master_evidence_universe",
            )
        )

    source_rows.extend(
        [
            source_token_row(
                "paper4_script12",
                p4_s12,
                s12_text,
                EXPECTED_LINEAGE_TOKENS["script12_output_master"],
                "master_evidence_table_output",
            ),
            source_token_row(
                "paper4_script15",
                p4_s15,
                s15_text,
                EXPECTED_LINEAGE_TOKENS["script15_input_master"],
                "master_evidence_table_input_to_ortholog_mapping",
            ),
            source_token_row(
                "paper4_script15",
                p4_s15,
                s15_text,
                EXPECTED_LINEAGE_TOKENS["script15_output_orthologs"],
                "candidate_restricted_ortholog_table_output",
            ),
            source_token_row(
                "paper4_script17",
                p4_s17,
                s17_text,
                EXPECTED_LINEAGE_TOKENS["script17_input_orthologs"],
                "candidate_restricted_ortholog_table_input_to_qc",
            ),
            source_token_row(
                "paper4_script17",
                p4_s17,
                s17_text,
                EXPECTED_LINEAGE_TOKENS["script17_output_qc"],
                "candidate_restricted_ortholog_qc_output",
            ),
            source_token_row(
                "paper6_02f0",
                P6_02F0_SCRIPT,
                p602f0_text,
                EXPECTED_LINEAGE_TOKENS["script02f0_input_qc"],
                "Paper6_02f0_mapping_reference",
            ),
        ]
    )

    source_lineage = pd.DataFrame(source_rows)

    # Historical literal filename tokens are diagnostics only. In particular,
    # Paper-6 02f0 resolves the ortholog table indirectly through the immutable
    # upstream asset role "ortholog_qc", so a literal filename need not occur in
    # the 02f0 source code. Scientific lineage is therefore established below
    # through locked path/hash identity plus exact gene-set inheritance.
    outcome_input_mask = source_lineage["evidence_role"].eq(
        "outcome_or_model_derived_input_to_master_evidence_universe"
    )

    outcome_input_tokens_present_n = int(
        source_lineage.loc[outcome_input_mask, "token_present"].sum()
    )
    outcome_input_tokens_total_n = int(outcome_input_mask.sum())

    all_literal_lineage_tokens_present = bool(
        source_lineage.loc[~outcome_input_mask, "token_present"].all()
    )

    missing_literal_lineage_tokens = (
        source_lineage.loc[
            (~outcome_input_mask) & ~source_lineage["token_present"],
            ["token", "source", "evidence_role"],
        ]
        .copy()
        .reset_index(drop=True)
    )

    missing_diagnostic_tokens = (
        source_lineage.loc[
            outcome_input_mask & ~source_lineage["token_present"],
            ["token", "source", "evidence_role"],
        ]
        .copy()
        .reset_index(drop=True)
    )

    # Header-level evidence only; outcome values are never read.
    master_header = list(pd.read_csv(p4_master, nrows=0).columns)
    outcome_aware_header_patterns = [
        r"(^|_)dfi(_|$)",
        r"(^|_)os(_|$)",
        r"univ",
        r"conditional",
        r"iamb",
        r"gsmb",
        r"nested",
        r"selection",
        r"rna_evidence",
    ]
    outcome_aware_header_cols = sorted(
        {
            col
            for col in master_header
            if any(
                re.search(pattern, str(col), flags=re.IGNORECASE)
                for pattern in outcome_aware_header_patterns
            )
        }
    )

    dfi_header_cols = sorted(
        [col for col in master_header if re.search(r"(^|_)dfi(_|$)", str(col), re.IGNORECASE)]
    )
    os_header_cols = sorted(
        [col for col in master_header if re.search(r"(^|_)os(_|$)", str(col), re.IGNORECASE)]
    )
    model_selection_header_cols = sorted(
        [
            col for col in master_header
            if re.search(
                r"(conditional|iamb|gsmb|nested|selection|stability|univ)",
                str(col),
                re.IGNORECASE,
            )
        ]
    )

    checks = {
        "paper4_master_to_with_orthologs_exact_gene_set_equality": master_equals_ortholog,
        "paper4_master_to_ortholog_qc_exact_gene_set_equality": master_equals_qc,
        "paper4_master_gene_set_is_subset_of_dog2_transcriptome": master_subset_expression,
        "paper6_02f0_strict_gene_set_is_subset_of_paper4_master_evidence_universe": p6_strict_subset_master,
        "paper6_02f0_primary_gene_set_is_subset_of_paper4_master_evidence_universe": p6_primary_subset_master,
        "paper6_02f0_strict_gene_set_equals_qc_strict_gene_set": p6_strict_equals_qc_strict,
        "paper6_upstream_lock_ortholog_qc_path_matches_paper4_qc": ortholog_lock_path_identity,
        "paper6_upstream_lock_ortholog_qc_hash_matches_paper4_qc": ortholog_lock_hash_identity,
        "paper6_02f0_contract_ortholog_qc_hash_matches_paper4_qc": ortholog_02f0_contract_hash_identity,
        "paper4_master_source_contains_all_prespecified_outcome_model_input_tokens": (
            outcome_input_tokens_present_n == outcome_input_tokens_total_n
        ),
        "literal_historical_lineage_tokens_all_present_diagnostic_only": all_literal_lineage_tokens_present,
        "master_has_dfi_derived_header_fields": len(dfi_header_cols) > 0,
        "master_has_os_derived_header_fields": len(os_header_cols) > 0,
        "master_has_model_selection_or_association_header_fields": len(model_selection_header_cols) > 0,
        "master_is_restricted_relative_to_full_dog2_transcriptome": len(master_genes) < len(dog2_feature_set),
        "outcome_aware_master_header_markers_present": len(outcome_aware_header_cols) > 0,
        "broader_biomart_one_to_one_space_exceeds_candidate_strict_space": (
            len(biomart_one2one_dog_symbols) > len(strict_qc_genes)
        ),
    }

    critical_checks = [
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

    confirmed = all(checks[key] for key in critical_checks)

    if confirmed:
        verdict = "CONFIRMED_OUTCOME_AWARE_ROW_UNIVERSE_INHERITANCE"
        scientific_status = (
            "02F0_PRIMARY_FEATURE_UNIVERSE_NOT_ELIGIBLE_FOR_PAPER6_PRIMARY_ANALYSIS"
        )
    else:
        verdict = "LINEAGE_NOT_FULLY_CONFIRMED"
        scientific_status = "HOLD_FOR_MANUAL_REVIEW_BEFORE_AMENDMENT"

    summary_rows = [
        {
            "layer": "DOG2_expression_header",
            "n_rows_or_records": "",
            "n_unique_genes_or_symbols": len(dog2_feature_set),
            "description": "Full DOG2 expression feature universe; header only.",
            "sha256": sha256_file(p4_expr),
        },
        {
            "layer": "Paper4_master_candidate_evidence",
            "n_rows_or_records": master_rows,
            "n_unique_genes_or_symbols": len(master_genes),
            "description": "Outcome-aware Paper4 RNA master evidence row universe.",
            "sha256": sha256_file(p4_master),
        },
        {
            "layer": "Paper4_master_with_orthologs",
            "n_rows_or_records": ortholog_rows,
            "n_unique_genes_or_symbols": len(ortholog_genes),
            "description": "Paper4 master evidence rows after ortholog annotation.",
            "sha256": sha256_file(p4_master_orthologs),
        },
        {
            "layer": "Paper4_ortholog_qc",
            "n_rows_or_records": qc_rows,
            "n_unique_genes_or_symbols": len(qc_genes),
            "description": "Paper4 master evidence rows after ortholog QC.",
            "sha256": sha256_file(p4_qc),
        },
        {
            "layer": "Paper4_ortholog_qc_strict_status",
            "n_rows_or_records": int(strict_qc.shape[0]),
            "n_unique_genes_or_symbols": len(strict_qc_genes),
            "description": f"Rows with ortholog_qc_status={STRICT_STATUS}.",
            "sha256": sha256_file(p4_qc),
        },
        {
            "layer": "Paper4_BioMart_cache_raw",
            "n_rows_or_records": int(biomart.shape[0]),
            "n_unique_genes_or_symbols": len(biomart_unique_dog_symbols),
            "description": "Broader outcome-blind Ensembl BioMart mapping snapshot.",
            "sha256": sha256_file(p4_biomart),
        },
        {
            "layer": "Paper4_BioMart_any_human_homolog",
            "n_rows_or_records": int(any_human.sum()),
            "n_unique_genes_or_symbols": len(biomart_any_human_symbols),
            "description": "Dog symbols with at least one non-empty human homolog in BioMart cache.",
            "sha256": sha256_file(p4_biomart),
        },
        {
            "layer": "Paper4_BioMart_one2one",
            "n_rows_or_records": int(one2one.sum()),
            "n_unique_genes_or_symbols": len(biomart_one2one_dog_symbols),
            "description": "Dog symbols with one2one human orthology in BioMart cache.",
            "sha256": sha256_file(p4_biomart),
        },
        {
            "layer": "Paper6_02f0_strict_universe",
            "n_rows_or_records": int(p6_universe.shape[0]),
            "n_unique_genes_or_symbols": len(p6_strict_genes),
            "description": "Strict canine genes frozen by Paper6 02f0.",
            "sha256": sha256_file(P6_02F0_UNIVERSE),
        },
        {
            "layer": "Paper6_02f0_primary_DOG2_TARGET",
            "n_rows_or_records": int(primary_mask.sum()),
            "n_unique_genes_or_symbols": len(p6_primary_genes),
            "description": "Primary DOG2->TARGET feature set frozen by Paper6 02f0.",
            "sha256": sha256_file(P6_02F0_UNIVERSE),
        },
    ]
    lineage_summary = pd.DataFrame(summary_rows)
    lineage_summary.to_csv(LINEAGE_SUMMARY_TSV, sep="\t", index=False)

    source_lineage.to_csv(SOURCE_LINEAGE_TSV, sep="\t", index=False)

    pd.DataFrame(
        [{"check": name, "passed": bool(passed)} for name, passed in checks.items()]
    ).to_csv(SET_AUDIT_TSV, sep="\t", index=False)

    p6_primary_count_from_summary = p6_summary.get("primary_feature_count")
    p6_strict_count_from_summary = p6_summary.get("strict_one_to_one_pairs")

    audit = {
        "script_version": SCRIPT_VERSION,
        "status": "PASS" if confirmed else "HOLD",
        "created_utc": now_utc(),
        "verdict": verdict,
        "scientific_status": scientific_status,
        "safety": {
            "network_access": False,
            "clinical_values_read": False,
            "outcome_response_followup_values_read": False,
            "treatment_administration_values_read": False,
            "expression_values_read": False,
            "expression_header_read": True,
            "gene_identifiers_read": True,
            "source_code_text_read": True,
            "model_fitting": False,
            "existing_02f0_contract_modified": False,
        },
        "paper4_absolute_path_recorded": False,
        "measured_counts": {
            "dog2_expression_features": len(dog2_feature_set),
            "paper4_master_rows": master_rows,
            "paper4_master_unique_genes": len(master_genes),
            "paper4_with_orthologs_rows": ortholog_rows,
            "paper4_with_orthologs_unique_genes": len(ortholog_genes),
            "paper4_ortholog_qc_rows": qc_rows,
            "paper4_ortholog_qc_unique_genes": len(qc_genes),
            "paper4_qc_strict_rows": int(strict_qc.shape[0]),
            "paper4_qc_strict_unique_canine_genes": len(strict_qc_genes),
            "biomart_rows": int(biomart.shape[0]),
            "biomart_unique_dog_symbols": len(biomart_unique_dog_symbols),
            "biomart_dog_symbols_with_any_human_homolog": len(biomart_any_human_symbols),
            "biomart_dog_symbols_with_one2one_human_ortholog": len(biomart_one2one_dog_symbols),
            "paper6_02f0_strict_unique_canine_genes": len(p6_strict_genes),
            "paper6_02f0_primary_unique_canine_genes": len(p6_primary_genes),
            "paper6_02f0_summary_strict_one_to_one_pairs": p6_strict_count_from_summary,
            "paper6_02f0_summary_primary_feature_count": p6_primary_count_from_summary,
        },
        "gene_set_hashes": {
            "dog2_expression_feature_set_sha256_sorted": sha256_lines(dog2_feature_set),
            "paper4_master_gene_set_sha256_sorted": sha256_lines(master_genes),
            "paper4_with_orthologs_gene_set_sha256_sorted": sha256_lines(ortholog_genes),
            "paper4_ortholog_qc_gene_set_sha256_sorted": sha256_lines(qc_genes),
            "paper4_qc_strict_gene_set_sha256_sorted": sha256_lines(strict_qc_genes),
            "paper6_02f0_strict_gene_set_sha256_sorted": sha256_lines(p6_strict_genes),
            "paper6_02f0_primary_gene_set_sha256_sorted": sha256_lines(p6_primary_genes),
        },
        "checks": checks,
        "outcome_aware_master_header_columns": outcome_aware_header_cols,
        "dfi_derived_master_header_columns": dfi_header_cols,
        "os_derived_master_header_columns": os_header_cols,
        "model_selection_master_header_columns": model_selection_header_cols,
        "literal_historical_lineage_tokens_all_present_diagnostic_only": all_literal_lineage_tokens_present,
        "historical_missing_literal_lineage_tokens_diagnostic_only": missing_literal_lineage_tokens["token"].tolist(),
        "historical_outcome_input_filename_tokens_present": outcome_input_tokens_present_n,
        "historical_outcome_input_filename_tokens_total": outcome_input_tokens_total_n,
        "historical_missing_input_tokens_are_diagnostic_only": missing_diagnostic_tokens["token"].tolist(),
        "locked_ortholog_qc_identity": {
            "relative_path": locked_ortholog_relative_path,
            "upstream_lock_sha256": locked_ortholog_sha256,
            "paper4_actual_sha256": actual_p4_qc_sha256,
            "02f0_contract_sha256": p6_contract_ortholog_sha256,
            "path_match": ortholog_lock_path_identity,
            "upstream_hash_match": ortholog_lock_hash_identity,
            "02f0_contract_hash_match": ortholog_02f0_contract_hash_identity,
        },
        "source_files": {
            "paper4_script12_sha256": sha256_file(p4_s12),
            "paper4_script15_sha256": sha256_file(p4_s15),
            "paper4_script17_sha256": sha256_file(p4_s17),
            "paper6_02f0_script_sha256": sha256_file(P6_02F0_SCRIPT),
        },
        "existing_02f0": {
            "contract_sha256": sha256_file(P6_02F0_CONTRACT),
            "summary_sha256": sha256_file(P6_02F0_SUMMARY),
            "strict_universe_sha256": sha256_file(P6_02F0_UNIVERSE),
            "primary_model_family": (
                (p6_contract.get("transport_design") or {}).get("primary_model_family")
            ),
        },
        "interpretation": {
            "measured_fact": (
                "The Paper-6 upstream lock and 02f0 contract both identify the exact Paper-4 "
                "ortholog-QC file by path/hash; its gene universe is identical to the Paper-4 "
                "RNA master evidence universe, whose construction source explicitly consumes "
                "DFI/OS and model-selection evidence."
            ),
            "consequence": (
                "Although 02f0 loaded only gene/mapping/QC columns and no outcome values, "
                "its primary feature universe inherits prior Paper-4 outcome-aware row selection "
                "and therefore must not serve as the primary outcome-blind Paper-6 transcriptome-wide universe."
            ),
            "broader_mapping_context": (
                "The local BioMart cache contains a substantially broader dog-human mapping "
                "space and is the appropriate starting snapshot for a new outcome-blind bridge, "
                "subject to explicit mapping/QC rules and a fresh-mapping sensitivity audit."
            ),
            "paper4_comparator_rule": (
                "The frozen Paper-4 comparator is not invalidated. It must remain evaluated exactly "
                "under its own frozen Paper-4 feature/program definitions; Paper-6 models use the "
                "new transcriptome-wide universe. Do not refit or remap the Paper-4 comparator to the new universe."
            ),
        },
        "required_next_action": (
            "Create a versioned pre-outcome amendment that supersedes 02f0 for Paper-6 primary "
            "feature-universe/method-role purposes, preserving 02f0 artifacts and hashes unchanged."
        ),
    }
    write_json(AUDIT_JSON, audit)

    output_hashes = {
        str(LINEAGE_SUMMARY_TSV.relative_to(ROOT)): sha256_file(LINEAGE_SUMMARY_TSV),
        str(SOURCE_LINEAGE_TSV.relative_to(ROOT)): sha256_file(SOURCE_LINEAGE_TSV),
        str(SET_AUDIT_TSV.relative_to(ROOT)): sha256_file(SET_AUDIT_TSV),
        str(AUDIT_JSON.relative_to(ROOT)): sha256_file(AUDIT_JSON),
    }

    summary = {
        "script_version": SCRIPT_VERSION,
        "status": audit["status"],
        "run_started_utc": started,
        "run_finished_utc": now_utc(),
        "verdict": verdict,
        "scientific_status": scientific_status,
        "existing_02f0_contract_sha256": sha256_file(P6_02F0_CONTRACT),
        "measured_counts": audit["measured_counts"],
        "checks": checks,
        "final_artifact_hashes": output_hashes,
        "outcomes_read": False,
        "clinical_values_read": False,
        "expression_values_read": False,
        "model_fitting": False,
    }
    write_json(SUMMARY_JSON, summary)

    print("-" * 118)
    print("Measured universe lineage")
    print("-" * 118)
    print(f"DOG2 expression features [header only]: {len(dog2_feature_set):,}")
    print(
        "Paper4 master candidate/evidence: "
        f"{master_rows:,} rows / {len(master_genes):,} unique genes"
    )
    print(
        "Paper4 master + orthologs: "
        f"{ortholog_rows:,} rows / {len(ortholog_genes):,} unique genes"
    )
    print(
        "Paper4 ortholog QC: "
        f"{qc_rows:,} rows / {len(qc_genes):,} unique genes"
    )
    print(f"Master gene set == with-orthologs gene set: {master_equals_ortholog}")
    print(f"Master gene set == ortholog-QC gene set: {master_equals_qc}")
    print(f"Master gene set subset of DOG2 transcriptome: {master_subset_expression}")
    print()
    print("Outcome-blind BioMart snapshot:")
    print(f"  raw mapping rows: {int(biomart.shape[0]):,}")
    print(f"  unique dog symbols: {len(biomart_unique_dog_symbols):,}")
    print(f"  dog symbols with any human homolog: {len(biomart_any_human_symbols):,}")
    print(f"  dog symbols with one2one human ortholog: {len(biomart_one2one_dog_symbols):,}")
    print()
    print("Current Paper6 02f0 inheritance:")
    print(f"  Paper4 QC strict unique canine genes: {len(strict_qc_genes):,}")
    print(f"  02f0 strict unique canine genes: {len(p6_strict_genes):,}")
    print(f"  02f0 primary DOG2->TARGET genes: {len(p6_primary_genes):,}")
    print(f"  02f0 strict subset of Paper4 master evidence universe: {p6_strict_subset_master}")
    print(f"  02f0 primary subset of Paper4 master evidence universe: {p6_primary_subset_master}")
    print(f"  02f0 strict set == Paper4 QC strict set: {p6_strict_equals_qc_strict}")
    print()
    print("Locked asset lineage:")
    print(f"  upstream ortholog_qc relative path: {locked_ortholog_relative_path}")
    print(f"  upstream lock path matches Paper4 QC: {ortholog_lock_path_identity}")
    print(f"  upstream lock SHA256 matches Paper4 QC: {ortholog_lock_hash_identity}")
    print(f"  02f0 contract ortholog_qc SHA256 matches Paper4 QC: {ortholog_02f0_contract_hash_identity}")
    print(f"  02f0 strict set == Paper4 QC strict set: {p6_strict_equals_qc_strict}")
    print()
    print("Source-code/header evidence:")
    print(
        "  Paper4 master source outcome/model input tokens: "
        f"{outcome_input_tokens_present_n}/{outcome_input_tokens_total_n}"
    )
    print(
        "  all literal historical lineage filename tokens present "
        f"[DIAGNOSTIC ONLY]: {all_literal_lineage_tokens_present}"
    )
    if not missing_literal_lineage_tokens.empty:
        print("  missing literal lineage tokens [NON-GATING; asset indirection allowed]:")
        for token in missing_literal_lineage_tokens["token"].tolist():
            print(f"    - {token}")
    if not missing_diagnostic_tokens.empty:
        print("  missing historical outcome/model input tokens [NON-GATING diagnostic]:")
        for token in missing_diagnostic_tokens["token"].tolist():
            print(f"    - {token}")
    print(f"  DFI-derived master header fields: {len(dfi_header_cols)}")
    print(f"  OS-derived master header fields: {len(os_header_cols)}")
    print(f"  model/selection master header fields: {len(model_selection_header_cols)}")
    print(f"  all outcome-aware master header markers: {len(outcome_aware_header_cols)}")
    if outcome_aware_header_cols:
        preview = ", ".join(outcome_aware_header_cols[:20])
        if len(outcome_aware_header_cols) > 20:
            preview += ", ..."
        print(f"  header marker preview: {preview}")
    print()

    print("=" * 118)
    print("02f0 TRANSPORT-UNIVERSE PREFLIGHT AUDIT")
    print("=" * 118)
    print(f"Verdict: {verdict}")
    print(f"Scientific status: {scientific_status}")
    print()
    if confirmed:
        print(
            "Interpretation: the 02f0 primary feature universe is confirmed to inherit "
            "the Paper-4 outcome-aware candidate/evidence row universe."
        )
        print(
            "The defect is provenance/selection inheritance, not accidental loading of "
            "outcome values inside 02f0."
        )
        print(
            "Paper-4 frozen comparator remains valid as a separate frozen comparator and "
            "must not be remapped/refit into the new Paper-6 universe."
        )
    else:
        print(
            "Interpretation: one or more lineage checks did not confirm. "
            "Do not write the amendment until the failed checks are reviewed."
        )
    print()
    print("Artifacts:")
    for path in [
        LINEAGE_SUMMARY_TSV,
        SOURCE_LINEAGE_TSV,
        SET_AUDIT_TSV,
        AUDIT_JSON,
        SUMMARY_JSON,
    ]:
        print(f"  {path.relative_to(ROOT)}")
    print()
    print("Outcome/response/follow-up values read: NO")
    print("Clinical values read: NO")
    print("Expression values read: NO")
    print("Model fitting: NO")
    print("Existing 02f0 artifacts modified: NO")
    print()
    print("Next:")
    print("  Write 02f0a pre-outcome amendment using these measured counts/hashes.")
    print(
        "  Then build a transcriptome-wide outcome-blind ortholog bridge from "
        "the broader mapping snapshot."
    )
    print("=" * 118)

    if not confirmed:
        raise RuntimeError(
            "Preflight lineage was not fully confirmed. Review the generated audit before amendment."
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 118, file=sys.stderr)
        print("02f0 transport-universe preflight audit: FAIL", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print("=" * 118, file=sys.stderr)
        raise
