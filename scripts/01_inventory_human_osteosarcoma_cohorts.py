from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_VERSION = "01-inventory-human-osteosarcoma-cohorts-v1-no-cli"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "_config"
CONTRACT_DIR = PROJECT_ROOT / "contracts"
MANIFEST_DIR = PROJECT_ROOT / "manifests"

LOCAL_PATH_CONFIG = CONFIG_DIR / "paths.local.json"

UPSTREAM_LOCK = CONTRACT_DIR / "00_upstream_input_lock.json"
UPSTREAM_MANIFEST = MANIFEST_DIR / "00_upstream_input_manifest.csv"

OUT_PUBLIC_REGISTRY = MANIFEST_DIR / "01_human_public_cohort_candidate_registry.csv"
OUT_LOCAL_SUMMARY = MANIFEST_DIR / "01_human_local_cohort_audit.csv"
OUT_CLINICAL_COLUMNS = MANIFEST_DIR / "01_human_clinical_column_audit.csv"
OUT_ENDPOINT_CANDIDATES = MANIFEST_DIR / "01_human_endpoint_candidate_audit.csv"
OUT_OVERLAP = MANIFEST_DIR / "01_human_exact_sample_id_overlap.csv"
OUT_CONTRACT = CONTRACT_DIR / "01_human_cohort_inventory_status.json"
OUT_README = MANIFEST_DIR / "01_human_cohort_inventory_README.txt"

EXPECTED_PAPER4_BASENAME = "paper4_sarcoma_dog"


# This is a candidate registry, not a declaration that every cohort is eligible
# for Paper 6 survival modelling. Eligibility requires later endpoint,
# sample-lineage, platform, and independence audits.
PUBLIC_COHORT_CANDIDATES: list[dict[str, Any]] = [
    {
        "cohort_id": "TARGET-OS",
        "accession_or_resource": "TARGET-OS",
        "species": "human",
        "data_type": "bulk RNA-seq",
        "reported_n": 87,
        "candidate_outcome": "OS/PFS and clinical covariates reported in public resource/literature",
        "paper6_status": "LOCKED_LOCALLY_PRIMARY_HUMAN",
        "independence_note": "Primary Paper 6 human cohort; do not pool into its own source comparator.",
        "provenance_note": "Included in 2024 Clinical Cancer Research comparative osteosarcoma analysis.",
    },
    {
        "cohort_id": "GSE21257",
        "accession_or_resource": "GSE21257",
        "species": "human",
        "data_type": "bulk microarray",
        "reported_n": 53,
        "candidate_outcome": "5-year metastasis; OS/EFS used in later reanalyses",
        "paper6_status": "LOCKED_LOCALLY_EXTERNAL_CANDIDATE",
        "independence_note": "Do not assume independence from GSE42352/GSE33382; published overlap has been reported.",
        "provenance_note": "Pre-chemotherapy biopsies; 34 metastasis within 5y, 19 no metastasis within 5y in GEO.",
    },
    {
        "cohort_id": "GSE39055",
        "accession_or_resource": "GSE39055",
        "species": "human",
        "data_type": "bulk microarray",
        "reported_n": 37,
        "candidate_outcome": "clinical survival/outcome and chemoresponse metadata",
        "paper6_status": "LOCKED_LOCALLY_EXTERNAL_CANDIDATE",
        "independence_note": "Audit patient relationship to companion miRNA/paired-resection accessions before any pooled analysis.",
        "provenance_note": "37 unique diagnostic biopsy specimens in GEO.",
    },
    {
        "cohort_id": "GSE16091",
        "accession_or_resource": "GSE16091",
        "species": "human",
        "data_type": "bulk microarray",
        "reported_n": 34,
        "candidate_outcome": "survival follow-up reported in later osteosarcoma prognostic studies",
        "paper6_status": "PUBLIC_CANDIDATE_NEEDS_LOCAL_IMPORT_AUDIT",
        "independence_note": "Needs explicit sample-lineage audit before treating as independent.",
        "provenance_note": "Included among public human osteosarcoma datasets in 2024 CCR comparative analysis.",
    },
    {
        "cohort_id": "GSE32981",
        "accession_or_resource": "GSE32981",
        "species": "human",
        "data_type": "bulk microarray",
        "reported_n": 23,
        "candidate_outcome": "clinical/PFS contribution reported in comparative osteosarcoma literature",
        "paper6_status": "PUBLIC_CANDIDATE_NEEDS_ENDPOINT_AUDIT",
        "independence_note": "Needs endpoint and sample-lineage audit.",
        "provenance_note": "Included among public human osteosarcoma datasets in 2024 CCR comparative analysis.",
    },
    {
        "cohort_id": "GSE33383",
        "accession_or_resource": "GSE33383",
        "species": "human",
        "data_type": "bulk microarray",
        "reported_n": 84,
        "candidate_outcome": "53/84 reported with binary metastasis-within-5-years outcome",
        "paper6_status": "PUBLIC_CANDIDATE_NEEDS_LINEAGE_AUDIT",
        "independence_note": "Do not treat as independent until relationship to GSE42352/GSE33382 and GSE21257 is resolved.",
        "provenance_note": "53 patients with binary 5-year progression outcome used in 2024 CCR comparative analysis.",
    },
    {
        "cohort_id": "GSE30699",
        "accession_or_resource": "GSE30699",
        "species": "human",
        "data_type": "bulk microarray / model-system-rich",
        "reported_n": 76,
        "candidate_outcome": "not assumed; must audit patient specimens and usable clinical outcomes",
        "paper6_status": "PUBLIC_CANDIDATE_NEEDS_ENDPOINT_AUDIT",
        "independence_note": "Contains model-system material; patient-level survival eligibility must be established.",
        "provenance_note": "Included among public datasets in 2024 CCR comparative analysis; not automatically survival-eligible.",
    },
    {
        "cohort_id": "GSE152048",
        "accession_or_resource": "GSE152048",
        "species": "human",
        "data_type": "single-cell RNA-seq",
        "reported_n": 11,
        "candidate_outcome": "not assumed",
        "paper6_status": "BIOLOGY_ONLY_CANDIDATE",
        "independence_note": "Not a default survival-transfer source; may support cell-state/compartment interpretation.",
        "provenance_note": "11 human osteosarcoma scRNA-seq samples in 2024 CCR comparative analysis.",
    },
    {
        "cohort_id": "GSE42352_GSE33382",
        "accession_or_resource": "GSE42352 / GSE33382",
        "species": "human",
        "data_type": "bulk microarray",
        "reported_n": 84,
        "candidate_outcome": "metastasis/prognostic use reported in later reanalyses",
        "paper6_status": "OVERLAP_RISK_NOT_INDEPENDENT_UNTIL_PROVEN",
        "independence_note": "Published work reports 27 osteosarcoma samples overlapping GSE21257; never count as an independent cohort without patient-level lineage resolution.",
        "provenance_note": "Added specifically as an overlap-risk cohort relevant to pooled-human source construction.",
    },
]


LOCAL_HUMAN_COHORTS: dict[str, dict[str, str]] = {
    "TARGET-OS": {
        "expression_asset": "target_expression",
        "clinical_asset": "target_clinical",
    },
    "GSE21257": {
        "expression_asset": "gse21257_expression",
        "clinical_asset": "gse21257_clinical",
    },
    "GSE39055": {
        "expression_asset": "gse39055_expression",
        "clinical_asset": "gse39055_clinical",
    },
}


ENDPOINT_KEYWORDS = [
    "overall survival",
    "overall_survival",
    "survival",
    "death",
    "dead",
    "event",
    "relapse",
    "recurrence",
    "metast",
    "progress",
    "disease free",
    "disease_free",
    "event free",
    "event_free",
    "progression free",
    "progression_free",
    "recurrence free",
    "recurrence_free",
    "time",
    "days",
    "months",
    "years",
    "necrosis",
    "response",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_local_path_config() -> dict[str, Any]:
    if not LOCAL_PATH_CONFIG.exists():
        return {}

    payload = json.loads(LOCAL_PATH_CONFIG.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Local path config must be a JSON object: {LOCAL_PATH_CONFIG}"
        )
    return payload


def resolve_paper4_root() -> tuple[Path, str]:
    candidates: list[tuple[Path, str]] = []

    env_value = os.environ.get("PAPER4_ROOT", "").strip()
    if env_value:
        candidates.append((Path(env_value).expanduser(), "environment:PAPER4_ROOT"))

    local_config = load_local_path_config()
    config_value = str(local_config.get("paper4_root", "")).strip()
    if config_value:
        candidates.append(
            (Path(config_value).expanduser(), "_config/paths.local.json")
        )

    candidates.extend(
        [
            (PROJECT_ROOT.parent / EXPECTED_PAPER4_BASENAME, "sibling_repository"),
            (
                Path.home() / "Desktop" / EXPECTED_PAPER4_BASENAME,
                "home_desktop_fallback",
            ),
        ]
    )

    for candidate, source in candidates:
        resolved = candidate.resolve()
        sentinel = (
            resolved
            / "data"
            / "processed"
            / "GSE238110_DOG2_expression_log2cpm_matched_allgenes.csv"
        )
        if resolved.is_dir() and sentinel.exists():
            return resolved, source

    raise FileNotFoundError(
        "Could not resolve Paper 4 root. Set PAPER4_ROOT or "
        "create _config/paths.local.json."
    )


def verify_upstream_lock(paper4_root: Path) -> dict[str, Any]:
    if not UPSTREAM_LOCK.exists() or not UPSTREAM_MANIFEST.exists():
        raise FileNotFoundError(
            "Run scripts/00_lock_upstream_inputs.py first and commit its outputs."
        )

    lock = json.loads(UPSTREAM_LOCK.read_text(encoding="utf-8"))
    recorded_manifest_hash = lock.get("manifest_sha256")
    observed_manifest_hash = sha256_file(UPSTREAM_MANIFEST)

    if recorded_manifest_hash != observed_manifest_hash:
        raise RuntimeError(
            "00 upstream manifest hash mismatch. Re-run Script 00 and investigate "
            "before continuing."
        )

    assets = lock.get("assets", {})
    required_asset_ids = {
        item
        for cohort in LOCAL_HUMAN_COHORTS.values()
        for item in cohort.values()
    }

    for asset_id in sorted(required_asset_ids):
        if asset_id not in assets:
            raise RuntimeError(f"Required asset absent from 00 lock: {asset_id}")

        metadata = assets[asset_id]
        path = paper4_root / metadata["relative_path"]
        if not path.exists():
            raise FileNotFoundError(f"Locked upstream asset is missing: {path}")

        observed_hash = sha256_file(path)
        if observed_hash != metadata["sha256"]:
            raise RuntimeError(
                f"Locked upstream asset changed since Script 00: {asset_id}\n"
                f"Expected: {metadata['sha256']}\n"
                f"Observed: {observed_hash}"
            )

    return lock


def normalize_sample_id(value: Any) -> str:
    text = str(value).strip()
    return text.upper()


def load_index(path: Path) -> list[str]:
    df = pd.read_csv(path, index_col=0, usecols=[0])
    return [normalize_sample_id(item) for item in df.index]


def clinical_dataframe(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, low_memory=False)


def is_endpoint_candidate(column: str) -> bool:
    lower = re.sub(r"[_\-]+", " ", str(column).lower()).strip()

    if re.search(r"(^|\s)os($|\s)", lower):
        return True
    if re.search(r"(^|\s)(dfs|dss|efs|pfs|rfs)($|\s)", lower):
        return True

    return any(keyword in lower for keyword in ENDPOINT_KEYWORDS)


def summarize_column(series: pd.Series) -> dict[str, Any]:
    nonmissing = series.dropna()
    unique_count = int(nonmissing.nunique(dropna=True))

    examples = []
    for value in nonmissing.astype(str).drop_duplicates().head(5):
        examples.append(value)

    numeric = pd.to_numeric(nonmissing, errors="coerce")
    numeric_fraction = (
        float(numeric.notna().mean()) if len(nonmissing) else 0.0
    )

    result: dict[str, Any] = {
        "nonmissing_n": int(nonmissing.shape[0]),
        "missing_n": int(series.shape[0] - nonmissing.shape[0]),
        "unique_n": unique_count,
        "dtype": str(series.dtype),
        "example_values": " | ".join(examples),
        "numeric_fraction": numeric_fraction,
        "numeric_min": None,
        "numeric_median": None,
        "numeric_max": None,
        "binary_or_low_cardinality_counts": None,
    }

    if numeric_fraction >= 0.90 and numeric.notna().any():
        result["numeric_min"] = float(numeric.min())
        result["numeric_median"] = float(numeric.median())
        result["numeric_max"] = float(numeric.max())

    if 1 <= unique_count <= 10:
        counts = nonmissing.astype(str).value_counts(dropna=False).to_dict()
        result["binary_or_low_cardinality_counts"] = json.dumps(
            counts, sort_keys=True
        )

    return result


def main() -> None:
    print("=" * 92)
    print("Paper 6 - inventory human osteosarcoma cohorts and audit locked local inputs")
    print("=" * 92)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print("")
    print("Scientific scope:")
    print("  Verify Script-00 input lock.")
    print("  Audit currently available human expression/clinical cohorts.")
    print("  Inventory candidate public human osteosarcoma cohorts.")
    print("  Flag known/possible sample-lineage risks.")
    print("  Perform NO model fitting and NO outcome association testing.")
    print("")

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    CONTRACT_DIR.mkdir(parents=True, exist_ok=True)

    paper4_root, resolution_source = resolve_paper4_root()
    lock = verify_upstream_lock(paper4_root)
    assets = lock["assets"]

    print(f"Paper 4 root: {paper4_root}")
    print(f"Resolution source: {resolution_source}")
    print("00 upstream lock verification: PASS")
    print("")

    local_rows: list[dict[str, Any]] = []
    clinical_column_rows: list[dict[str, Any]] = []
    endpoint_rows: list[dict[str, Any]] = []
    expression_id_sets: dict[str, set[str]] = {}
    clinical_id_sets: dict[str, set[str]] = {}

    for cohort_id, mapping in LOCAL_HUMAN_COHORTS.items():
        expression_meta = assets[mapping["expression_asset"]]
        clinical_meta = assets[mapping["clinical_asset"]]

        expression_path = paper4_root / expression_meta["relative_path"]
        clinical_path = paper4_root / clinical_meta["relative_path"]

        expression_ids = load_index(expression_path)
        clinical = clinical_dataframe(clinical_path)
        clinical_ids = [normalize_sample_id(item) for item in clinical.index]

        expr_set = set(expression_ids)
        clin_set = set(clinical_ids)
        expression_id_sets[cohort_id] = expr_set
        clinical_id_sets[cohort_id] = clin_set

        intersection = expr_set & clin_set
        expr_only = expr_set - clin_set
        clinical_only = clin_set - expr_set

        local_rows.append(
            {
                "cohort_id": cohort_id,
                "expression_rows_locked": expression_meta.get("n_rows"),
                "expression_features_locked": expression_meta.get("n_columns"),
                "clinical_rows_locked": clinical_meta.get("n_rows"),
                "clinical_columns_locked": clinical_meta.get("n_columns"),
                "expression_unique_ids": len(expr_set),
                "clinical_unique_ids": len(clin_set),
                "exact_expr_clinical_intersection": len(intersection),
                "expression_ids_without_clinical": len(expr_only),
                "clinical_ids_without_expression": len(clinical_only),
                "expression_duplicate_id_count": len(expression_ids) - len(expr_set),
                "clinical_duplicate_id_count": len(clinical_ids) - len(clin_set),
                "expression_sha256": expression_meta["sha256"],
                "clinical_sha256": clinical_meta["sha256"],
            }
        )

        for column in clinical.columns:
            summary = summarize_column(clinical[column])
            row = {
                "cohort_id": cohort_id,
                "column": str(column),
                **summary,
                "endpoint_candidate": is_endpoint_candidate(str(column)),
            }
            clinical_column_rows.append(row)

            if row["endpoint_candidate"]:
                endpoint_rows.append(row.copy())

        print(
            f"{cohort_id}: expr={len(expr_set)}, clinical={len(clin_set)}, "
            f"matched={len(intersection)}, endpoint-like columns="
            f"{sum(is_endpoint_candidate(str(c)) for c in clinical.columns)}"
        )

    # Exact ID overlap across different locally available cohorts.
    overlap_rows: list[dict[str, Any]] = []
    cohort_ids = sorted(LOCAL_HUMAN_COHORTS)
    for i, left in enumerate(cohort_ids):
        for right in cohort_ids[i + 1 :]:
            left_ids = expression_id_sets[left] | clinical_id_sets[left]
            right_ids = expression_id_sets[right] | clinical_id_sets[right]
            overlap = sorted(left_ids & right_ids)
            overlap_rows.append(
                {
                    "cohort_a": left,
                    "cohort_b": right,
                    "exact_normalized_id_overlap_n": len(overlap),
                    "example_overlapping_ids": " | ".join(overlap[:10]),
                    "interpretation": (
                        "Exact ID overlap found; investigate before pooling."
                        if overlap
                        else "No exact ID overlap. This does NOT prove patient-level independence."
                    ),
                }
            )

    registry = pd.DataFrame(PUBLIC_COHORT_CANDIDATES)
    registry["currently_locked_locally"] = registry["cohort_id"].isin(
        LOCAL_HUMAN_COHORTS
    )

    local_df = pd.DataFrame(local_rows)
    clinical_df = pd.DataFrame(clinical_column_rows)
    endpoint_df = pd.DataFrame(endpoint_rows)
    overlap_df = pd.DataFrame(overlap_rows)

    registry.to_csv(OUT_PUBLIC_REGISTRY, index=False)
    local_df.to_csv(OUT_LOCAL_SUMMARY, index=False)
    clinical_df.to_csv(OUT_CLINICAL_COLUMNS, index=False)
    endpoint_df.to_csv(OUT_ENDPOINT_CANDIDATES, index=False)
    overlap_df.to_csv(OUT_OVERLAP, index=False)

    # Premise gate is deliberately NOT passed here. Public cohorts have not yet
    # been fully acquired/audited for compatible time-to-event endpoints and
    # patient-level independence.
    status_payload = {
        "script_version": SCRIPT_VERSION,
        "created_utc": utc_now_iso(),
        "upstream_lock_verified": True,
        "scientific_data_copied": False,
        "model_fitting": False,
        "outcome_association_testing": False,
        "local_human_cohorts_audited": sorted(LOCAL_HUMAN_COHORTS),
        "public_candidate_count": int(registry.shape[0]),
        "known_overlap_risks": [
            {
                "cohort": "GSE42352_GSE33382",
                "risk": "published patient/sample overlap with GSE21257; independent-cohort status forbidden until lineage audit",
            }
        ],
        "pooled_human_source_gate": "NOT_READY",
        "pooled_human_source_gate_reason": (
            "Compatible survival endpoints, patient-level independence, and "
            "cross-platform harmonization eligibility have not yet been frozen "
            "for all public candidate cohorts."
        ),
        "next_required_stage": (
            "Acquire/audit metadata only for candidate human cohorts and build "
            "a patient-lineage/endpoint compatibility table before any "
            "dog-vs-pooled-human source comparison."
        ),
        "outputs": {
            "public_registry": OUT_PUBLIC_REGISTRY.name,
            "local_summary": OUT_LOCAL_SUMMARY.name,
            "clinical_columns": OUT_CLINICAL_COLUMNS.name,
            "endpoint_candidates": OUT_ENDPOINT_CANDIDATES.name,
            "exact_id_overlap": OUT_OVERLAP.name,
        },
    }

    OUT_CONTRACT.write_text(
        json.dumps(status_payload, indent=2),
        encoding="utf-8",
    )

    readme = f"""Paper 6 human cohort inventory
Script version: {SCRIPT_VERSION}

This stage is a premise audit, not a modelling stage.

What this script does
---------------------
1. Re-verifies the Script-00 SHA-256 lock for locally available human inputs.
2. Audits expression/clinical sample-ID matching for TARGET-OS, GSE21257,
   and GSE39055.
3. Enumerates clinical columns and flags endpoint-like fields without fitting
   any outcome model.
4. Computes exact normalized sample-ID overlap across the locally available
   cohorts.
5. Writes a candidate registry of public human osteosarcoma datasets relevant
   to later endpoint and sample-lineage auditing.

Critical interpretation rule
----------------------------
Zero exact sample-ID overlap does NOT establish patient-level independence.
Different GEO accessions can rename the same patient/specimen. Independence
must be established from source publications, sample annotations, and, where
possible, patient/sample lineage identifiers.

The pooled-human-source comparator remains NOT READY after this script.
No scientific dataset is copied into the Paper 6 repository.
"""
    OUT_README.write_text(readme, encoding="utf-8")

    print("")
    print("=" * 92)
    print("Paper 6 human cohort inventory: PASS")
    print("=" * 92)
    print(f"Local human cohorts audited: {len(LOCAL_HUMAN_COHORTS)}")
    print(f"Public candidate records: {registry.shape[0]}")
    print("Pooled-human source gate: NOT READY (expected at this stage)")
    print("Scientific data copied: NO")
    print("Model fitting: NO")
    print("Outcome association testing: NO")
    print("")
    print("Saved:")
    for output in [
        OUT_PUBLIC_REGISTRY,
        OUT_LOCAL_SUMMARY,
        OUT_CLINICAL_COLUMNS,
        OUT_ENDPOINT_CANDIDATES,
        OUT_OVERLAP,
        OUT_CONTRACT,
        OUT_README,
    ]:
        print(f"  {output}")
    print("")
    print("Next:")
    print("  Review the endpoint-candidate audit and candidate registry.")
    print("  Then perform the external metadata + patient-lineage audit before")
    print("  constructing any pooled-human source comparator.")
    print("Done.")


if __name__ == "__main__":
    main()
