#!/usr/bin/env python3
"""
Paper 6 - audit biological-module coverage of the new 02g outcome-blind bridge.

Purpose
-------
This is a deliberately bounded pre-outcome step.

It asks only:
Does the transcriptome-wide outcome-blind DOG2->human bridge produced by 02g
retain enough genes to support a low-capacity biological-module representation?

External outcome-blind libraries:
1. MSigDB Hallmark v2026.1.Hs - primary low-capacity module library candidate.
2. MSigDB Reactome v2026.1.Hs - secondary expanded pathway library candidate.

This script DOES NOT:
- read DOG2 or human outcome/clinical values;
- read expression matrix values;
- fit or tune any predictive model;
- choose modules using outcome performance;
- define TME cell compartments;
- use Paper-4 outcome-derived module prioritization for Paper-6 feature selection.

The goal is technical/biological coverage only.

No CLI arguments.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests


SCRIPT_VERSION = "02h-audit-module-coverage-v1-no-cli"

ROOT = Path(__file__).resolve().parents[1]

BRIDGE_DIR = ROOT / "results" / "ortholog_bridge" / "02g_v3"
BRIDGE_SUMMARY = BRIDGE_DIR / "summary.json"
BRIDGE_CONTRACT = BRIDGE_DIR / "ortholog_bridge_contract.json"
BRIDGE_TSV = BRIDGE_DIR / "primary_outcome_blind_ortholog_bridge.tsv"

OUT_DIR = ROOT / "results" / "module_coverage" / "02h"
REF_DIR = OUT_DIR / "reference"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REF_DIR.mkdir(parents=True, exist_ok=True)

HALLMARK_GMT = REF_DIR / "h.all.v2026.1.Hs.symbols.gmt"
REACTOME_GMT = REF_DIR / "c2.cp.reactome.v2026.1.Hs.symbols.gmt"
REFERENCE_LOCK = REF_DIR / "msigdb_reference_lock.json"

COVERAGE_TSV = OUT_DIR / "module_coverage.tsv"
LIBRARY_SUMMARY_TSV = OUT_DIR / "library_coverage_summary.tsv"
HALLMARK_FAMILY_TSV = OUT_DIR / "hallmark_mechanistic_family_coverage.tsv"
ELIGIBLE_MODULES_TSV = OUT_DIR / "technically_eligible_modules.tsv"
CONTRACT_JSON = OUT_DIR / "module_coverage_contract.json"
SUMMARY_JSON = OUT_DIR / "summary.json"

MSIGDB_RELEASE = "2026.1.Hs"
HALLMARK_URL = (
    "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/"
    "2026.1.Hs/h.all.v2026.1.Hs.symbols.gmt"
)
REACTOME_URL = (
    "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/"
    "2026.1.Hs/c2.cp.reactome.v2026.1.Hs.symbols.gmt"
)

EXPECTED_HALLMARK_SETS = 50

# Technical module eligibility only. These thresholds are NOT outcome-optimized.
MIN_MAPPED_GENES = 10
MIN_COVERAGE_FRACTION = 0.50
REACTOME_MIN_SOURCE_GENES = 10
REACTOME_MAX_SOURCE_GENES = 500

# Deliberately permissive readiness thresholds: this gate asks only whether a
# module-based representation is feasible, not which modules are predictive.
MIN_HALLMARK_PRIMARY_ELIGIBLE = 40
MIN_HALLMARK_COMMON4_ELIGIBLE = 35
MIN_REACTOME_PRIMARY_ELIGIBLE = 300
MIN_REACTOME_COMMON4_ELIGIBLE = 200

EXPECTED_02G_STATUS = (
    "PASS_TRANSCRIPTOMEWIDE_OUTCOME_BLIND_BRIDGE_READY_FOR_MODULE_COVERAGE"
)

FEATURE_SET_COLUMNS = {
    "primary_dog2_to_target_os": "primary_dog2_to_target_os",
    "secondary_dog2_to_gse21257": "secondary_dog2_to_gse21257",
    "stress_dog2_to_gse39055": "stress_dog2_to_gse39055",
    "common_dog2_target_gse21257": "common_dog2_target_gse21257",
    "common_all_four": "common_all_four",
}

# These are mechanistic families, not cell-type/TME compartment labels.
# They are used only to ensure that several major biological axes retain coverage.
HALLMARK_FAMILIES = {
    "PROLIFERATION_DNA": [
        "HALLMARK_E2F_TARGETS",
        "HALLMARK_G2M_CHECKPOINT",
        "HALLMARK_MYC_TARGETS_V1",
        "HALLMARK_MYC_TARGETS_V2",
        "HALLMARK_DNA_REPAIR",
        "HALLMARK_MITOTIC_SPINDLE",
    ],
    "IMMUNE_INFLAMMATORY": [
        "HALLMARK_INTERFERON_ALPHA_RESPONSE",
        "HALLMARK_INTERFERON_GAMMA_RESPONSE",
        "HALLMARK_INFLAMMATORY_RESPONSE",
        "HALLMARK_IL6_JAK_STAT3_SIGNALING",
        "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
        "HALLMARK_COMPLEMENT",
        "HALLMARK_ALLOGRAFT_REJECTION",
    ],
    "STROMAL_ECM_SIGNALING": [
        "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
        "HALLMARK_TGF_BETA_SIGNALING",
        "HALLMARK_APICAL_JUNCTION",
        "HALLMARK_ANGIOGENESIS",
        "HALLMARK_COAGULATION",
    ],
    "METABOLIC_STRESS": [
        "HALLMARK_HYPOXIA",
        "HALLMARK_MTORC1_SIGNALING",
        "HALLMARK_GLYCOLYSIS",
        "HALLMARK_OXIDATIVE_PHOSPHORYLATION",
        "HALLMARK_REACTIVE_OXYGEN_SPECIES_PATHWAY",
        "HALLMARK_FATTY_ACID_METABOLISM",
    ],
    "DEVELOPMENT_GROWTH_SIGNALING": [
        "HALLMARK_WNT_BETA_CATENIN_SIGNALING",
        "HALLMARK_TGF_BETA_SIGNALING",
        "HALLMARK_NOTCH_SIGNALING",
        "HALLMARK_HEDGEHOG_SIGNALING",
    ],
}


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
        raise FileNotFoundError(f"Required file missing: {path}")
    return path


def bool_mask(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False).astype(bool)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y", "t"})
    )


def download_text(
    session: requests.Session,
    url: str,
    destination: Path,
    *,
    timeout: int = 120,
    retries: int = 3,
) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        return

    last_exc: Optional[Exception] = None
    temporary = destination.with_suffix(destination.suffix + ".part")

    for attempt in range(1, retries + 1):
        try:
            response = session.get(
                url,
                timeout=timeout,
                headers={"User-Agent": "Paper6-module-coverage/1.0"},
            )
            response.raise_for_status()
            text = response.text
            if not text.strip():
                raise RuntimeError("Downloaded GMT was empty.")
            temporary.write_text(text, encoding="utf-8", newline="\n")
            temporary.replace(destination)
            return
        except Exception as exc:
            last_exc = exc
            temporary.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(2 * attempt)

    raise RuntimeError(f"Could not download {url}: {last_exc}")


def freeze_reference_files() -> Dict[str, Any]:
    session = requests.Session()

    if REFERENCE_LOCK.exists():
        lock = read_json(REFERENCE_LOCK)
        for name, path in [
            ("hallmark", HALLMARK_GMT),
            ("reactome", REACTOME_GMT),
        ]:
            require_file(path)
            expected = clean((lock.get("files") or {}).get(name, {}).get("sha256"))
            if len(expected) != 64:
                raise RuntimeError(
                    f"Existing MSigDB reference lock lacks valid hash for {name}."
                )
            observed = sha256_file(path)
            if observed != expected:
                raise RuntimeError(
                    f"MSigDB reference hash mismatch for {name}: "
                    f"expected {expected}, observed {observed}"
                )
        return lock

    download_text(session, HALLMARK_URL, HALLMARK_GMT)
    download_text(session, REACTOME_URL, REACTOME_GMT)

    lock = {
        "status": "PASS",
        "created_utc": now_utc(),
        "release": MSIGDB_RELEASE,
        "source": "Human MSigDB public symbol GMT release files",
        "files": {
            "hallmark": {
                "url": HALLMARK_URL,
                "path": str(HALLMARK_GMT.relative_to(ROOT)),
                "sha256": sha256_file(HALLMARK_GMT),
                "size_bytes": HALLMARK_GMT.stat().st_size,
            },
            "reactome": {
                "url": REACTOME_URL,
                "path": str(REACTOME_GMT.relative_to(ROOT)),
                "sha256": sha256_file(REACTOME_GMT),
                "size_bytes": REACTOME_GMT.stat().st_size,
            },
        },
    }
    write_json(REFERENCE_LOCK, lock)
    return lock


def parse_gmt(path: Path, library: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue

            module = parts[0].strip()
            description = parts[1].strip()
            genes = []
            seen = set()

            for value in parts[2:]:
                gene = value.strip().upper()
                if not gene or gene in seen:
                    continue
                seen.add(gene)
                genes.append(gene)

            rows.append(
                {
                    "library": library,
                    "module": module,
                    "description": description,
                    "source_gene_count": len(genes),
                    "genes": genes,
                    "line_number": line_number,
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"No gene sets parsed from {path}")
    return frame


def coverage_rows(
    modules: pd.DataFrame,
    feature_sets: Dict[str, set[str]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for module_row in modules.itertuples(index=False):
        module_genes = set(module_row.genes)
        source_n = len(module_genes)

        for feature_set_name, available in feature_sets.items():
            mapped = sorted(module_genes & available)
            mapped_n = len(mapped)
            fraction = mapped_n / source_n if source_n else 0.0

            if module_row.library == "HALLMARK":
                source_size_pass = True
            else:
                source_size_pass = (
                    REACTOME_MIN_SOURCE_GENES
                    <= source_n
                    <= REACTOME_MAX_SOURCE_GENES
                )

            eligible = (
                source_size_pass
                and mapped_n >= MIN_MAPPED_GENES
                and fraction >= MIN_COVERAGE_FRACTION
            )

            rows.append(
                {
                    "library": module_row.library,
                    "module": module_row.module,
                    "feature_set": feature_set_name,
                    "source_gene_count": source_n,
                    "mapped_gene_count": mapped_n,
                    "coverage_fraction": fraction,
                    "source_size_pass": source_size_pass,
                    "technical_eligibility_pass": eligible,
                    "mapped_genes": ";".join(mapped),
                }
            )

    return rows


def main() -> None:
    started = now_utc()

    print("=" * 116)
    print("Paper 6 - audit module coverage on the 02g outcome-blind ortholog bridge")
    print("=" * 116)
    print(f"Script version: {SCRIPT_VERSION}")
    print()
    print("Safety / scope:")
    print("  DOG2/human outcome or clinical values read: NO")
    print("  Expression values read: NO")
    print("  Model fitting/tuning: NO")
    print("  Paper4 outcome-derived module prioritization used: NO")
    print("  External module definitions: MSigDB Hallmark + Reactome only")
    print("  TME cell-compartment definitions created here: NO")
    print()

    for path in [BRIDGE_SUMMARY, BRIDGE_CONTRACT, BRIDGE_TSV]:
        require_file(path)

    bridge_summary = read_json(BRIDGE_SUMMARY)
    bridge_contract = read_json(BRIDGE_CONTRACT)

    if clean(bridge_summary.get("status")) != "PASS":
        raise RuntimeError("02g summary is not PASS.")
    if clean(bridge_contract.get("status")) != "PASS":
        raise RuntimeError("02g contract is not PASS.")
    if clean(bridge_summary.get("scientific_status")) != EXPECTED_02G_STATUS:
        raise RuntimeError(
            "02g scientific status is not the expected module-coverage-ready state."
        )

    expected_bridge_hash = clean(
        (bridge_summary.get("final_artifact_hashes") or {}).get(
            "primary_outcome_blind_ortholog_bridge_tsv"
        )
    )
    observed_bridge_hash = sha256_file(BRIDGE_TSV)

    if len(expected_bridge_hash) != 64 or observed_bridge_hash != expected_bridge_hash:
        raise RuntimeError("02g primary bridge hash verification failed.")

    bridge = pd.read_csv(BRIDGE_TSV, sep="\t", dtype=str, low_memory=False).fillna("")

    if "human_gene_symbol" not in bridge.columns:
        raise RuntimeError("02g bridge lacks human_gene_symbol.")

    for col in FEATURE_SET_COLUMNS.values():
        if col not in bridge.columns:
            raise RuntimeError(f"02g bridge lacks feature-set column {col!r}.")

    bridge["human_gene_symbol"] = (
        bridge["human_gene_symbol"].astype(str).str.strip().str.upper()
    )

    feature_sets: Dict[str, set[str]] = {}

    for name, column in FEATURE_SET_COLUMNS.items():
        mask = bool_mask(bridge[column])
        genes = set(
            bridge.loc[mask, "human_gene_symbol"]
            .astype(str)
            .str.strip()
            .str.upper()
        )
        genes.discard("")
        feature_sets[name] = genes

    print("02g feature-set sizes:")
    for name, genes in feature_sets.items():
        print(f"  {name}: {len(genes):,}")

    measured_02g = bridge_summary.get("measured_counts") or {}
    expected_counts = {
        "primary_dog2_to_target_os": int(
            measured_02g["primary_dog2_to_target_os"]
        ),
        "secondary_dog2_to_gse21257": int(
            measured_02g["secondary_dog2_to_gse21257"]
        ),
        "stress_dog2_to_gse39055": int(
            measured_02g["stress_dog2_to_gse39055"]
        ),
        "common_dog2_target_gse21257": int(
            measured_02g["common_dog2_target_gse21257"]
        ),
        "common_all_four": int(measured_02g["common_all_four"]),
    }

    for name, expected in expected_counts.items():
        observed = len(feature_sets[name])
        if observed != expected:
            raise RuntimeError(
                f"02g feature-set count mismatch for {name}: "
                f"expected {expected}, observed {observed}"
            )

    ref_lock = freeze_reference_files()
    print()
    print("Frozen external module references:")
    print(f"  MSigDB release: {ref_lock['release']}")
    print(
        f"  Hallmark SHA256: "
        f"{ref_lock['files']['hallmark']['sha256']}"
    )
    print(
        f"  Reactome SHA256: "
        f"{ref_lock['files']['reactome']['sha256']}"
    )

    hallmark = parse_gmt(HALLMARK_GMT, "HALLMARK")
    reactome = parse_gmt(REACTOME_GMT, "REACTOME")

    if hallmark.shape[0] != EXPECTED_HALLMARK_SETS:
        raise RuntimeError(
            f"Expected {EXPECTED_HALLMARK_SETS} Hallmark sets in "
            f"{MSIGDB_RELEASE}, observed {hallmark.shape[0]}"
        )

    modules = pd.concat([hallmark, reactome], axis=0, ignore_index=True)

    coverage = pd.DataFrame(coverage_rows(modules, feature_sets))
    coverage.to_csv(COVERAGE_TSV, sep="\t", index=False)

    eligible = coverage[coverage["technical_eligibility_pass"]].copy()
    eligible.to_csv(ELIGIBLE_MODULES_TSV, sep="\t", index=False)

    summary_rows: List[Dict[str, Any]] = []

    for (library, feature_set), part in coverage.groupby(
        ["library", "feature_set"],
        sort=True,
    ):
        eligible_part = part[part["technical_eligibility_pass"]]

        summary_rows.append(
            {
                "library": library,
                "feature_set": feature_set,
                "n_modules_total": int(part.shape[0]),
                "n_modules_technically_eligible": int(eligible_part.shape[0]),
                "fraction_modules_technically_eligible": (
                    float(eligible_part.shape[0] / part.shape[0])
                    if part.shape[0]
                    else 0.0
                ),
                "median_mapped_genes": float(part["mapped_gene_count"].median()),
                "median_coverage_fraction": float(
                    part["coverage_fraction"].median()
                ),
                "min_mapped_genes": int(part["mapped_gene_count"].min()),
                "max_mapped_genes": int(part["mapped_gene_count"].max()),
            }
        )

    library_summary = pd.DataFrame(summary_rows)
    library_summary.to_csv(LIBRARY_SUMMARY_TSV, sep="\t", index=False)

    # Hallmark mechanistic-family coverage.
    hallmark_primary = coverage[
        (coverage["library"] == "HALLMARK")
        & (coverage["feature_set"] == "primary_dog2_to_target_os")
    ].copy()

    family_rows: List[Dict[str, Any]] = []
    hallmark_index = hallmark_primary.set_index("module")

    for family, modules_in_family in HALLMARK_FAMILIES.items():
        for module in modules_in_family:
            if module not in hallmark_index.index:
                family_rows.append(
                    {
                        "family": family,
                        "module": module,
                        "found_in_hallmark": False,
                        "mapped_gene_count": 0,
                        "coverage_fraction": 0.0,
                        "technical_eligibility_pass": False,
                    }
                )
                continue

            row = hallmark_index.loc[module]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]

            family_rows.append(
                {
                    "family": family,
                    "module": module,
                    "found_in_hallmark": True,
                    "mapped_gene_count": int(row["mapped_gene_count"]),
                    "coverage_fraction": float(row["coverage_fraction"]),
                    "technical_eligibility_pass": bool(
                        row["technical_eligibility_pass"]
                    ),
                }
            )

    family_frame = pd.DataFrame(family_rows)
    family_frame.to_csv(HALLMARK_FAMILY_TSV, sep="\t", index=False)

    def eligible_count(library: str, feature_set: str) -> int:
        row = library_summary[
            (library_summary["library"] == library)
            & (library_summary["feature_set"] == feature_set)
        ]
        if row.empty:
            return 0
        return int(row.iloc[0]["n_modules_technically_eligible"])

    hallmark_primary_n = eligible_count(
        "HALLMARK", "primary_dog2_to_target_os"
    )
    hallmark_common4_n = eligible_count("HALLMARK", "common_all_four")
    reactome_primary_n = eligible_count(
        "REACTOME", "primary_dog2_to_target_os"
    )
    reactome_common4_n = eligible_count("REACTOME", "common_all_four")

    checks = {
        "hallmark_primary_readiness": (
            hallmark_primary_n >= MIN_HALLMARK_PRIMARY_ELIGIBLE
        ),
        "hallmark_common_all_four_readiness": (
            hallmark_common4_n >= MIN_HALLMARK_COMMON4_ELIGIBLE
        ),
        "reactome_primary_readiness": (
            reactome_primary_n >= MIN_REACTOME_PRIMARY_ELIGIBLE
        ),
        "reactome_common_all_four_readiness": (
            reactome_common4_n >= MIN_REACTOME_COMMON4_ELIGIBLE
        ),
    }

    if all(checks.values()):
        scientific_status = (
            "PASS_MODULE_REPRESENTATION_COVERAGE_READY_FOR_02I"
        )
        status = "PASS"
    else:
        scientific_status = (
            "HOLD_MODULE_REPRESENTATION_COVERAGE_REQUIRES_REVIEW"
        )
        status = "HOLD"

    contract = {
        "script_version": SCRIPT_VERSION,
        "status": status,
        "scientific_status": scientific_status,
        "created_utc": now_utc(),
        "scope": (
            "Outcome-blind technical coverage audit of external biological "
            "module definitions on the 02g cross-species feature universe."
        ),
        "02g": {
            "summary_sha256": sha256_file(BRIDGE_SUMMARY),
            "contract_sha256": sha256_file(BRIDGE_CONTRACT),
            "bridge_sha256": observed_bridge_hash,
            "feature_set_counts": expected_counts,
        },
        "external_reference": ref_lock,
        "technical_eligibility_rule": {
            "minimum_mapped_genes": MIN_MAPPED_GENES,
            "minimum_coverage_fraction": MIN_COVERAGE_FRACTION,
            "Hallmark_source_size_filter": "NONE",
            "Reactome_minimum_source_genes": REACTOME_MIN_SOURCE_GENES,
            "Reactome_maximum_source_genes": REACTOME_MAX_SOURCE_GENES,
            "rule_is_outcome_optimized": False,
        },
        "readiness_thresholds": {
            "minimum_Hallmark_primary_eligible": MIN_HALLMARK_PRIMARY_ELIGIBLE,
            "minimum_Hallmark_common_all_four_eligible": MIN_HALLMARK_COMMON4_ELIGIBLE,
            "minimum_Reactome_primary_eligible": MIN_REACTOME_PRIMARY_ELIGIBLE,
            "minimum_Reactome_common_all_four_eligible": MIN_REACTOME_COMMON4_ELIGIBLE,
        },
        "observed_readiness": {
            "Hallmark_primary_eligible": hallmark_primary_n,
            "Hallmark_common_all_four_eligible": hallmark_common4_n,
            "Reactome_primary_eligible": reactome_primary_n,
            "Reactome_common_all_four_eligible": reactome_common4_n,
        },
        "checks": checks,
        "guardrails": {
            "outcomes_read": False,
            "clinical_values_read": False,
            "expression_values_read": False,
            "model_fitting": False,
            "module_choice_by_outcome_performance": False,
            "Paper4_outcome_derived_module_prioritization_used": False,
            "TME_cell_compartment_claim_made": False,
        },
        "interpretation": (
            "PASS means only that an externally defined module representation "
            "is technically supportable on the new outcome-blind bridge. "
            "It does not select a predictive architecture or predictive modules."
        ),
        "required_next": (
            "02i: freeze the bounded model/benchmark/source-gate contract; "
            "then proceed directly to 03a/03b Source Prognostic Gate."
        ),
    }
    write_json(CONTRACT_JSON, contract)

    final_hashes = {
        "module_coverage_tsv": sha256_file(COVERAGE_TSV),
        "library_coverage_summary_tsv": sha256_file(LIBRARY_SUMMARY_TSV),
        "hallmark_mechanistic_family_coverage_tsv": sha256_file(
            HALLMARK_FAMILY_TSV
        ),
        "technically_eligible_modules_tsv": sha256_file(
            ELIGIBLE_MODULES_TSV
        ),
        "module_coverage_contract_json": sha256_file(CONTRACT_JSON),
        "msigdb_reference_lock_json": sha256_file(REFERENCE_LOCK),
    }

    summary = {
        "script_version": SCRIPT_VERSION,
        "status": status,
        "scientific_status": scientific_status,
        "run_started_utc": started,
        "run_finished_utc": now_utc(),
        "02g_feature_set_counts": expected_counts,
        "Hallmark_modules_total": int(hallmark.shape[0]),
        "Reactome_modules_total": int(reactome.shape[0]),
        "Hallmark_primary_eligible": hallmark_primary_n,
        "Hallmark_common_all_four_eligible": hallmark_common4_n,
        "Reactome_primary_eligible": reactome_primary_n,
        "Reactome_common_all_four_eligible": reactome_common4_n,
        "checks": checks,
        "final_artifact_hashes": final_hashes,
        "outcomes_read": False,
        "clinical_values_read": False,
        "expression_values_read": False,
        "model_fitting": False,
    }
    write_json(SUMMARY_JSON, summary)

    print()
    print("-" * 116)
    print("Module coverage summary")
    print("-" * 116)
    display_cols = [
        "library",
        "feature_set",
        "n_modules_total",
        "n_modules_technically_eligible",
        "fraction_modules_technically_eligible",
        "median_mapped_genes",
        "median_coverage_fraction",
    ]
    print(library_summary[display_cols].to_string(index=False))

    print()
    print("Mechanistic Hallmark families [primary TARGET bridge]:")
    family_summary = (
        family_frame.groupby("family")
        .agg(
            n_modules=("module", "size"),
            n_eligible=("technical_eligibility_pass", "sum"),
            median_mapped_genes=("mapped_gene_count", "median"),
            median_coverage=("coverage_fraction", "median"),
        )
        .reset_index()
    )
    print(family_summary.to_string(index=False))

    print()
    print("=" * 116)
    print("02h MODULE COVERAGE SUMMARY")
    print("=" * 116)
    print(f"MSigDB release: {MSIGDB_RELEASE}")
    print(f"Hallmark modules: {hallmark.shape[0]:,}")
    print(f"Reactome modules: {reactome.shape[0]:,}")
    print(f"Hallmark primary eligible: {hallmark_primary_n:,}")
    print(f"Hallmark common-all-four eligible: {hallmark_common4_n:,}")
    print(f"Reactome primary eligible: {reactome_primary_n:,}")
    print(f"Reactome common-all-four eligible: {reactome_common4_n:,}")
    print(f"Scientific status: {scientific_status}")
    print()
    print("Outcomes/clinical/expression values read: NO")
    print("Model fitting: NO")
    print("Paper4 outcome-derived module prioritization used: NO")
    print()
    print("Artifacts:")
    for path in [
        HALLMARK_GMT,
        REACTOME_GMT,
        REFERENCE_LOCK,
        COVERAGE_TSV,
        LIBRARY_SUMMARY_TSV,
        HALLMARK_FAMILY_TSV,
        ELIGIBLE_MODULES_TSV,
        CONTRACT_JSON,
        SUMMARY_JSON,
    ]:
        print(f"  {path.relative_to(ROOT)}")

    if status == "PASS":
        print()
        print("Next: 02i short revised model/benchmark/source-gate contract, then 03a/03b.")
        print("=" * 116)
        print("02h module coverage audit: PASS")
        print("=" * 116)
    else:
        print()
        print("HOLD: module representation coverage did not meet the prespecified technical floor.")
        print("=" * 116)
        print("02h module coverage audit: HOLD")
        print("=" * 116)
        raise RuntimeError(scientific_status)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 116, file=sys.stderr)
        print("02h module coverage audit: FAIL", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print("=" * 116, file=sys.stderr)
        raise
