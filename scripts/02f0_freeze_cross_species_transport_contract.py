#!/usr/bin/env python3
"""
Paper 6 - freeze the pre-outcome cross-species transport contract.

This stage follows a GREEN expression-level Gate Zero and still keeps every
clinical endpoint CLOSED.

Purpose
-------
1. Verify the immutable Paper-4 upstream lock and the GREEN 02e9 Gate Zero.
2. Read expression HEADERS ONLY for DOG², TARGET-OS, GSE21257, and GSE39055.
3. Read ONLY the canine-gene / human-gene / ortholog-QC columns from the
   ortholog table; no DFI/OS/outcome-statistic columns are loaded.
4. Freeze strict symbol-concordant one-to-one ortholog feature universes for:
      * primary:   DOG² -> TARGET-OS
      * secondary: DOG² -> GSE21257
      * stress:    DOG² -> GSE39055
      * three-cohort and four-cohort common sensitivities
5. Freeze high-level transfer-model roles and leakage guardrails before any
   Paper-6 endpoint values are opened.

This script DOES NOT:
  * read DOG² clinical values;
  * read human clinical values;
  * read outcome/response/follow-up values;
  * read treatment-administration values;
  * read expression matrix VALUES;
  * fit any model;
  * select genes using outcomes, arm labels, or expression variance;
  * create target train/test splits.

No command-line arguments are used.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd


SCRIPT_VERSION = "02f0-freeze-cross-species-transport-contract-v1-no-cli"

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "_config"
CONTRACT_DIR = ROOT / "contracts"
MANIFEST_DIR = ROOT / "manifests"

LOCAL_PATH_CONFIG = CONFIG_DIR / "paths.local.json"
UPSTREAM_LOCK = CONTRACT_DIR / "00_upstream_input_lock.json"
UPSTREAM_MANIFEST = MANIFEST_DIR / "00_upstream_input_manifest.csv"

GATE_DIR = ROOT / "results" / "expression_gate_zero" / "02e9"
GATE_SUMMARY = GATE_DIR / "summary.json"
GATE_RESULTS = GATE_DIR / "gate_zero_results.json"

OUT_DIR = ROOT / "results" / "transport_contract" / "02f0"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STRICT_UNIVERSE_TSV = OUT_DIR / "strict_ortholog_universe.tsv"
FEATURE_SETS_TSV = OUT_DIR / "transport_feature_sets.tsv"
FEATURE_SUMMARY_TSV = OUT_DIR / "feature_set_summary.tsv"
CONTRACT_JSON = OUT_DIR / "cross_species_transport_contract.json"
SUMMARY_JSON = OUT_DIR / "summary.json"

EXPECTED_PAPER4_BASENAME = "paper4_sarcoma_dog"
STRICT_STATUS = "strict_symbol_concordant_one_to_one"

REQUIRED_ASSETS = {
    "dog2_expression",
    "target_expression",
    "gse21257_expression",
    "gse39055_expression",
    "ortholog_qc",
}

EXPRESSION_ASSETS = {
    "DOG2": "dog2_expression",
    "TARGET_OS": "target_expression",
    "GSE21257": "gse21257_expression",
    "GSE39055": "gse39055_expression",
}

FEATURE_SET_SPECS: Dict[str, Dict[str, Any]] = {
    "primary_dog2_to_target_os": {
        "role": "PRIMARY",
        "cohorts": ["DOG2", "TARGET_OS"],
        "endpoint_role": "DOG2 OS -> TARGET-OS OS adaptive survival transfer",
    },
    "secondary_dog2_to_gse21257": {
        "role": "SECONDARY",
        "cohorts": ["DOG2", "GSE21257"],
        "endpoint_role": (
            "DOG2 DFI/source representation -> GSE21257 metastasis-within-5-years"
        ),
    },
    "stress_dog2_to_gse39055": {
        "role": "STRESS_TEST",
        "cohorts": ["DOG2", "GSE39055"],
        "endpoint_role": "DOG2 source -> GSE39055 recurrence-free-survival stress test",
    },
    "common_dog2_target_gse21257": {
        "role": "SENSITIVITY",
        "cohorts": ["DOG2", "TARGET_OS", "GSE21257"],
        "endpoint_role": "three-cohort common-feature sensitivity only",
    },
    "common_all_four": {
        "role": "SENSITIVITY",
        "cohorts": ["DOG2", "TARGET_OS", "GSE21257", "GSE39055"],
        "endpoint_role": "four-cohort common-feature sensitivity only",
    },
}


# --------------------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------------------

def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_lines(values: Iterable[str]) -> str:
    text = "\n".join(str(value) for value in values) + "\n"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "na", "n/a"}:
        return ""
    return text


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


# --------------------------------------------------------------------------------------
# Resolve and verify the immutable upstream Paper-4 assets.
# --------------------------------------------------------------------------------------

def load_local_path_config() -> Dict[str, Any]:
    if not LOCAL_PATH_CONFIG.exists():
        return {}
    payload = read_json(LOCAL_PATH_CONFIG)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Local path config must be a JSON object: {LOCAL_PATH_CONFIG}")
    return payload


def resolve_paper4_root() -> Tuple[Path, str]:
    candidates: List[Tuple[Path, str]] = []

    env_value = os.environ.get("PAPER4_ROOT", "").strip()
    if env_value:
        candidates.append((Path(env_value).expanduser(), "environment:PAPER4_ROOT"))

    config_value = clean(load_local_path_config().get("paper4_root"))
    if config_value:
        candidates.append((Path(config_value).expanduser(), "_config/paths.local.json"))

    candidates.extend(
        [
            (ROOT.parent / EXPECTED_PAPER4_BASENAME, "sibling_repository"),
            (Path.home() / "Desktop" / EXPECTED_PAPER4_BASENAME, "home_desktop_fallback"),
        ]
    )

    checked: List[str] = []
    for candidate, source in candidates:
        resolved = candidate.resolve()
        checked.append(f"{source}: {resolved}")
        sentinel = (
            resolved
            / "data"
            / "processed"
            / "GSE238110_DOG2_expression_log2cpm_matched_allgenes.csv"
        )
        if resolved.is_dir() and sentinel.exists():
            return resolved, source

    raise FileNotFoundError(
        "Could not resolve Paper 4 root. Checked:\n  - " + "\n  - ".join(checked)
    )


def verify_upstream_lock(paper4_root: Path) -> Dict[str, Any]:
    for path in (UPSTREAM_LOCK, UPSTREAM_MANIFEST):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing upstream lock artifact: {path}\n"
                "Run scripts\\00_lock_upstream_inputs.py first."
            )

    lock = read_json(UPSTREAM_LOCK)
    expected_manifest_sha = clean(lock.get("manifest_sha256"))
    actual_manifest_sha = sha256_file(UPSTREAM_MANIFEST)
    if expected_manifest_sha != actual_manifest_sha:
        raise RuntimeError("00 upstream manifest SHA256 mismatch.")

    assets = lock.get("assets")
    if not isinstance(assets, dict):
        raise RuntimeError("00 upstream lock lacks an assets object.")

    missing = sorted(REQUIRED_ASSETS - set(assets))
    if missing:
        raise RuntimeError("00 upstream lock is missing required assets: " + ", ".join(missing))

    verified: Dict[str, Any] = {}
    for asset_id in sorted(REQUIRED_ASSETS):
        metadata = assets[asset_id]
        if not isinstance(metadata, dict):
            raise RuntimeError(f"Malformed upstream metadata for {asset_id}.")

        relative_path = clean(metadata.get("relative_path"))
        expected_sha = clean(metadata.get("sha256"))
        if not relative_path or len(expected_sha) != 64:
            raise RuntimeError(f"Incomplete upstream metadata for {asset_id}.")

        path = paper4_root / relative_path
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Locked upstream asset is missing: {path}")

        actual_sha = sha256_file(path)
        if actual_sha.lower() != expected_sha.lower():
            raise RuntimeError(
                f"Locked upstream asset changed: {asset_id}\n"
                f"Expected SHA256: {expected_sha}\nObserved SHA256: {actual_sha}"
            )

        verified[asset_id] = {
            "path": path,
            "relative_path": relative_path,
            "sha256": actual_sha,
            "n_rows_locked": metadata.get("n_rows"),
            "n_columns_locked": metadata.get("n_columns"),
        }

    return {
        "lock": lock,
        "manifest_sha256": actual_manifest_sha,
        "assets": verified,
    }


# --------------------------------------------------------------------------------------
# Gate Zero verification.
# --------------------------------------------------------------------------------------

def verify_gate_zero() -> Dict[str, Any]:
    for path in (GATE_SUMMARY, GATE_RESULTS):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing 02e9 Gate Zero artifact: {path}\n"
                "Run scripts\\02e9_run_expression_gate_zero_v2.py first."
            )

    summary = read_json(GATE_SUMMARY)
    results = read_json(GATE_RESULTS)

    if summary.get("status") != "PASS" or results.get("status") != "PASS":
        raise RuntimeError("02e9 Gate Zero status is not PASS.")

    final_gate = clean(summary.get("final_gate"))
    if final_gate != "GREEN_NO_MATERIAL_ARM_EXPRESSION_SEPARATION":
        raise RuntimeError(
            "02f0 requires GREEN Gate Zero; observed final_gate=" + repr(final_gate)
        )

    for key in (
        "outcome_response_followup_values_read",
        "treatment_administration_values_read",
        "postbaseline_sample_annotations_read",
        "paper4_clinical_file_read",
    ):
        if summary.get(key) is not False:
            raise RuntimeError(f"02e9 provenance is not closed for {key}.")

    result_safety = results.get("safety") or {}
    for key in (
        "outcome_response_followup_values_read",
        "treatment_administration_values_read",
        "postbaseline_sample_annotations_read",
        "paper4_clinical_file_read",
    ):
        if result_safety.get(key) is not False:
            raise RuntimeError(f"02e9 result safety mismatch for {key}.")

    return {
        "summary_sha256": sha256_file(GATE_SUMMARY),
        "results_sha256": sha256_file(GATE_RESULTS),
        "final_gate": final_gate,
        "primary_PERMANOVA_R2": summary.get("primary_PERMANOVA_R2"),
        "primary_classifier_mean_AUC": summary.get("primary_classifier_mean_AUC"),
    }


# --------------------------------------------------------------------------------------
# Outcome-blind feature inventory.
# --------------------------------------------------------------------------------------

def read_expression_header(path: Path, cohort: str) -> List[str]:
    # nrows=0 reads CSV schema/header only; no expression matrix values are loaded.
    header = pd.read_csv(path, nrows=0, index_col=0)
    columns = [str(column).strip() for column in header.columns]

    if not columns:
        raise RuntimeError(f"{cohort}: expression header has zero features.")
    if any(not value for value in columns):
        raise RuntimeError(f"{cohort}: expression header contains blank feature names.")
    if len(columns) != len(set(columns)):
        duplicated = pd.Index(columns)[pd.Index(columns).duplicated()].tolist()[:20]
        raise RuntimeError(f"{cohort}: duplicated expression feature names: {duplicated}")

    # Human matrices are gene-symbol harmonized upstream.  Case normalization
    # must not silently merge two distinct columns at the Paper-6 boundary.
    if cohort != "DOG2":
        normalized = [value.upper() for value in columns]
        if len(normalized) != len(set(normalized)):
            duplicated = (
                pd.Index(normalized)[pd.Index(normalized).duplicated()]
                .unique()
                .tolist()[:20]
            )
            raise RuntimeError(
                f"{cohort}: duplicated gene symbols after uppercase normalization: "
                f"{duplicated}"
            )

    return columns


def detect_ortholog_columns(path: Path) -> Tuple[str, str, str]:
    header = pd.read_csv(path, nrows=0)
    columns = list(header.columns)

    dog_candidates = [
        "gene",
        "gene_symbol_clean",
        "canine_gene_symbol",
        "canine_gene",
        "dog_gene_symbol",
        "dog_gene",
    ]
    human_candidates = [
        "human_gene_symbol",
        "human_symbol",
        "human_gene",
    ]
    status_candidates = [
        "ortholog_qc_status",
        "mapping_status",
        "qc_status",
    ]

    dog_col = next((item for item in dog_candidates if item in columns), None)
    human_col = next((item for item in human_candidates if item in columns), None)
    status_col = next((item for item in status_candidates if item in columns), None)

    if dog_col is None or human_col is None or status_col is None:
        raise RuntimeError(
            "Could not detect required ortholog columns. Available columns: "
            + ", ".join(map(str, columns))
        )

    return dog_col, human_col, status_col


def load_strict_orthologs(path: Path) -> Tuple[pd.DataFrame, Dict[str, int], Tuple[str, str, str]]:
    dog_col, human_col, status_col = detect_ortholog_columns(path)

    # Critical guardrail: usecols prevents outcome-derived columns present in the
    # wider Paper-4 QC table from being loaded into this Paper-6 stage.
    table = pd.read_csv(
        path,
        usecols=[dog_col, human_col, status_col],
        dtype=str,
        low_memory=False,
    ).fillna("")

    table[dog_col] = table[dog_col].map(clean)
    table[human_col] = table[human_col].map(clean).str.upper()
    table[status_col] = table[status_col].map(clean)

    raw_rows = int(len(table))
    strict = table.loc[table[status_col].eq(STRICT_STATUS), [dog_col, human_col, status_col]].copy()
    strict = strict.loc[strict[dog_col].ne("") & strict[human_col].ne("")].copy()

    strict_before_exact_dedup = int(len(strict))
    strict = strict.drop_duplicates([dog_col, human_col], keep="first").copy()

    dog_duplicate_mask = strict[dog_col].duplicated(keep=False)
    human_duplicate_mask = strict[human_col].duplicated(keep=False)
    ambiguous_mask = dog_duplicate_mask | human_duplicate_mask

    ambiguous_rows = int(ambiguous_mask.sum())
    ambiguous_dog_genes = int(strict.loc[dog_duplicate_mask, dog_col].nunique())
    ambiguous_human_genes = int(strict.loc[human_duplicate_mask, human_col].nunique())

    strict = strict.loc[~ambiguous_mask].copy()
    strict = strict.rename(
        columns={
            dog_col: "canine_gene",
            human_col: "human_gene_symbol",
            status_col: "ortholog_qc_status",
        }
    )
    strict = strict.sort_values(
        ["human_gene_symbol", "canine_gene"], kind="mergesort"
    ).reset_index(drop=True)

    if strict.empty:
        raise RuntimeError("Strict ortholog universe is empty after one-to-one enforcement.")
    if strict["canine_gene"].duplicated().any():
        raise RuntimeError("Internal error: canine strict mapping is not one-to-one.")
    if strict["human_gene_symbol"].duplicated().any():
        raise RuntimeError("Internal error: human strict mapping is not one-to-one.")

    diagnostics = {
        "ortholog_table_rows_read": raw_rows,
        "strict_rows_before_exact_dedup": strict_before_exact_dedup,
        "strict_exact_duplicate_rows_removed": strict_before_exact_dedup - int(len(strict)) - ambiguous_rows,
        "strict_ambiguous_rows_excluded": ambiguous_rows,
        "strict_ambiguous_canine_gene_count": ambiguous_dog_genes,
        "strict_ambiguous_human_gene_count": ambiguous_human_genes,
        "strict_one_to_one_pairs_retained": int(len(strict)),
    }
    return strict, diagnostics, (dog_col, human_col, status_col)


def add_availability(
    strict: pd.DataFrame,
    feature_headers: Dict[str, List[str]],
) -> pd.DataFrame:
    out = strict.copy()

    dog_features = set(feature_headers["DOG2"])
    out["available_DOG2"] = out["canine_gene"].isin(dog_features)

    for cohort in ("TARGET_OS", "GSE21257", "GSE39055"):
        human_features = {value.upper() for value in feature_headers[cohort]}
        out[f"available_{cohort}"] = out["human_gene_symbol"].isin(human_features)

    return out


def build_feature_sets(universe: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    membership = universe.copy()
    summary_rows: List[Dict[str, Any]] = []

    for set_name, specification in FEATURE_SET_SPECS.items():
        cohorts = list(specification["cohorts"])
        required_columns = [f"available_{cohort}" for cohort in cohorts]
        mask = membership[required_columns].all(axis=1)
        membership[set_name] = mask

        selected = membership.loc[mask, ["canine_gene", "human_gene_symbol"]]
        genes = selected["human_gene_symbol"].tolist()
        canine = selected["canine_gene"].tolist()

        summary_rows.append(
            {
                "feature_set": set_name,
                "role": specification["role"],
                "cohorts": ";".join(cohorts),
                "n_features": int(mask.sum()),
                "human_gene_list_sha256": sha256_lines(genes),
                "canine_gene_list_sha256": sha256_lines(canine),
                "endpoint_role": specification["endpoint_role"],
            }
        )

        # A transfer setting with fewer than 50 strict genes is not credible as
        # the planned high-dimensional transfer-learning universe.
        if int(mask.sum()) < 50:
            raise RuntimeError(
                f"Feature set {set_name} retained only {int(mask.sum())} strict orthologs."
            )

    summary = pd.DataFrame(summary_rows)
    return membership, summary


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main() -> None:
    started = now_utc()

    print("=" * 112)
    print("Paper 6 - freeze pre-outcome cross-species transport contract")
    print("=" * 112)
    print(f"Script version: {SCRIPT_VERSION}")
    print()
    print("Safety / execution contract:")
    print("  Gate Zero must be GREEN: YES")
    print("  Expression matrix VALUES read: NO [headers only]")
    print("  DOG² clinical values read: NO")
    print("  Human clinical values read: NO")
    print("  Outcome/response/follow-up values read: NO")
    print("  Treatment-administration values read: NO")
    print("  Paper-4 prognostic weights used for feature selection: NO")
    print("  Model fitting: NO")
    print("  Target train/test splits generated: NO")
    print("  GPU execution: NO [metadata/header freeze only]")
    print()

    gate = verify_gate_zero()
    print("02e9 Gate Zero verification:")
    print(f"  final gate: {gate['final_gate']}")
    print("  02e9 summary hash: PASS")
    print("  02e9 results hash: PASS")
    print()

    paper4_root, paper4_resolution = resolve_paper4_root()
    upstream = verify_upstream_lock(paper4_root)
    assets = upstream["assets"]

    print("Upstream Paper-4 lock verification:")
    print(f"  physical root: {paper4_root}")
    print(f"  resolution source: {paper4_resolution}")
    print("  00 manifest hash: PASS")
    for asset_id in sorted(REQUIRED_ASSETS):
        print(f"  {asset_id}: PASS")
    print()

    feature_headers: Dict[str, List[str]] = {}
    print("Expression header inventory [NO matrix values read]:")
    for cohort, asset_id in EXPRESSION_ASSETS.items():
        columns = read_expression_header(assets[asset_id]["path"], cohort)
        feature_headers[cohort] = columns
        locked_n = assets[asset_id].get("n_columns_locked")
        if locked_n is not None and int(locked_n) != len(columns):
            raise RuntimeError(
                f"{cohort}: header feature count {len(columns)} != locked {locked_n}."
            )
        print(f"  {cohort}: {len(columns):,} feature columns")
    print()

    strict, ortholog_diag, ortholog_columns = load_strict_orthologs(
        assets["ortholog_qc"]["path"]
    )
    print("Strict ortholog inventory:")
    print(f"  source dog column: {ortholog_columns[0]}")
    print(f"  source human column: {ortholog_columns[1]}")
    print(f"  source QC column: {ortholog_columns[2]}")
    print(f"  required QC status: {STRICT_STATUS}")
    print(
        "  strict one-to-one pairs retained: "
        f"{ortholog_diag['strict_one_to_one_pairs_retained']:,}"
    )
    print(
        "  ambiguous strict rows excluded: "
        f"{ortholog_diag['strict_ambiguous_rows_excluded']:,}"
    )
    print("  outcome-related ortholog-table columns loaded: NO")
    print()

    universe = add_availability(strict, feature_headers)
    membership, feature_summary = build_feature_sets(universe)

    print("Frozen transport feature universes:")
    for row in feature_summary.to_dict("records"):
        print(
            f"  {row['feature_set']}: n={int(row['n_features']):,} "
            f"[{row['role']}]"
        )
    print()

    membership.to_csv(STRICT_UNIVERSE_TSV, sep="\t", index=False)

    long_rows: List[Dict[str, Any]] = []
    for set_name, specification in FEATURE_SET_SPECS.items():
        selected = membership.loc[
            membership[set_name], ["canine_gene", "human_gene_symbol"]
        ]
        for rank, row in enumerate(selected.itertuples(index=False), start=1):
            long_rows.append(
                {
                    "feature_set": set_name,
                    "role": specification["role"],
                    "feature_rank_lexicographic": rank,
                    "canine_gene": row.canine_gene,
                    "human_gene_symbol": row.human_gene_symbol,
                }
            )
    pd.DataFrame(long_rows).to_csv(FEATURE_SETS_TSV, sep="\t", index=False)
    feature_summary.to_csv(FEATURE_SUMMARY_TSV, sep="\t", index=False)

    primary_n = int(
        feature_summary.loc[
            feature_summary["feature_set"].eq("primary_dog2_to_target_os"),
            "n_features",
        ].iloc[0]
    )

    contract = {
        "script_version": SCRIPT_VERSION,
        "status": "PASS",
        "created_utc": now_utc(),
        "scientific_stage": "PRE_OUTCOME_CROSS_SPECIES_TRANSPORT_FREEZE",
        "gate_zero": gate,
        "upstream": {
            "paper4_project": EXPECTED_PAPER4_BASENAME,
            "paper4_absolute_path_recorded": False,
            "upstream_manifest_sha256": upstream["manifest_sha256"],
            "required_asset_hashes": {
                asset_id: assets[asset_id]["sha256"]
                for asset_id in sorted(REQUIRED_ASSETS)
            },
        },
        "data_access": {
            "expression_headers_read": True,
            "expression_values_read": False,
            "dog2_clinical_values_read": False,
            "human_clinical_values_read": False,
            "outcome_response_followup_values_read": False,
            "treatment_administration_values_read": False,
            "postbaseline_annotations_read": False,
            "paper4_prognostic_weight_values_read": False,
            "ortholog_mapping_columns_read": list(ortholog_columns),
            "other_ortholog_qc_columns_read": False,
        },
        "ortholog_rule": {
            "required_status": STRICT_STATUS,
            "human_symbol_normalization": "strip + uppercase",
            "canine_feature_matching": "exact expression-column identifier",
            "exact_duplicate_pair_policy": "collapse exact duplicate pairs",
            "ambiguous_one_to_many_or_many_to_one_policy": (
                "exclude every strict row participating in a duplicated canine or human key"
            ),
            "diagnostics": ortholog_diag,
        },
        "feature_sets": {
            row["feature_set"]: {
                "role": row["role"],
                "cohorts": row["cohorts"].split(";"),
                "n_features": int(row["n_features"]),
                "human_gene_list_sha256": row["human_gene_list_sha256"],
                "canine_gene_list_sha256": row["canine_gene_list_sha256"],
                "endpoint_role": row["endpoint_role"],
            }
            for row in feature_summary.to_dict("records")
        },
        "transport_design": {
            "primary_question": (
                "Does controlled canine-to-human adaptation improve TARGET-OS "
                "overall-survival prediction over target-only learning without "
                "forcing zero-shot equality of prognostic effects?"
            ),
            "primary_setting": "DOG2 OS -> TARGET-OS OS",
            "primary_model_family": (
                "regularized residual Cox transfer: beta_human = beta_dog + delta"
            ),
            "primary_required_comparators": [
                "target-only regularized Cox",
                "direct/zero-shot canine source model",
                "adapted residual-Cox transfer",
            ],
            "secondary_setting": "DOG2 DFI/source representation -> GSE21257 metastasis within 5 years",
            "secondary_model_family": (
                "source-pretrained low-dimensional encoder with target-specific binary head"
            ),
            "stress_setting": "DOG2 source -> GSE39055 recurrence-free survival",
            "stress_role": (
                "locked evaluation/stress-test cohort; no model or hyperparameter selection on GSE39055"
            ),
            "negative_controls_to_freeze_before_first_fit": [
                "source-outcome permutation",
                "random initialization / no-source-pretraining comparator for encoder branch",
                "mismatched-endpoint transfer where endpoint semantics permit",
            ],
            "exact_endpoint_columns": "DEFERRED_TO_NEXT_PRE_MODEL_ENDPOINT_CONTRACT",
            "exact_event_counts": "DEFERRED_UNTIL_ENDPOINT_VALUES_ARE_EXPLICITLY_OPENED",
            "exact_outer_splits": "DEFERRED_UNTIL_ENDPOINT_CONTRACT_IS_FROZEN",
            "hyperparameter_grid": "DEFERRED_TO_PRE_MODEL_CONTRACT; must be frozen before first outcome fit",
        },
        "leakage_guardrails": {
            "no_outcome_based_feature_selection": True,
            "no_arm_supervised_feature_selection": True,
            "no_human_outcome_use_in_ortholog_mapping": True,
            "no_paper4_frozen_prognostic_weight_use_for_primary_feature_selection": True,
            "no_transductive_target_preprocessing": True,
            "target_scaler_fit_inside_outer_training_fold": True,
            "target_test_fold_never_used_for_preprocessing_or_tuning": True,
            "all_target_adaptation_and_tuning_nested_inside_outer_training_fold": True,
            "gse39055_never_used_for_tuning": True,
            "do_not_replace_primary_feature_set_with_four_cohort_intersection": True,
            "do_not_pool_human_cohorts_without_a_separate_lineage_endpoint_contract": True,
            "do_not_open_outcomes_in_02f0": True,
        },
        "preprocessing_principles_for_later_modeling": {
            "source": (
                "fit source-only feature scaling on source training data; do not use target data "
                "to normalize source coefficients"
            ),
            "target": (
                "within each outer target fold, fit target scaling on target-training samples only "
                "and apply unchanged to target-test samples"
            ),
            "feature_filtering": (
                "only deterministic frozen-universe restriction plus training-fold-only removal of "
                "non-finite/zero-variance features; any additional dimensionality reduction must be "
                "fit inside the relevant training fold"
            ),
        },
        "next_required_stage": (
            "Freeze exact endpoint definitions and resampling/tuning contract before the first "
            "Paper-6 outcome value is used for model fitting."
        ),
        "primary_feature_count": primary_n,
    }
    write_json(CONTRACT_JSON, contract)

    output_hashes = {
        rel(STRICT_UNIVERSE_TSV): sha256_file(STRICT_UNIVERSE_TSV),
        rel(FEATURE_SETS_TSV): sha256_file(FEATURE_SETS_TSV),
        rel(FEATURE_SUMMARY_TSV): sha256_file(FEATURE_SUMMARY_TSV),
        rel(CONTRACT_JSON): sha256_file(CONTRACT_JSON),
    }

    summary = {
        "script_version": SCRIPT_VERSION,
        "status": "PASS",
        "run_started_utc": started,
        "run_finished_utc": now_utc(),
        "gate_zero_final": gate["final_gate"],
        "expression_headers_read": True,
        "expression_values_read": False,
        "clinical_values_read": False,
        "outcome_response_followup_values_read": False,
        "treatment_administration_values_read": False,
        "model_fitting": False,
        "target_splits_generated": False,
        "strict_one_to_one_pairs": int(len(strict)),
        "primary_feature_count": primary_n,
        "feature_set_counts": {
            row["feature_set"]: int(row["n_features"])
            for row in feature_summary.to_dict("records")
        },
        "contract_sha256": sha256_file(CONTRACT_JSON),
        "final_artifact_hashes": output_hashes,
        "next_required_stage": contract["next_required_stage"],
    }
    write_json(SUMMARY_JSON, summary)

    print("=" * 112)
    print("02f0 PRE-OUTCOME TRANSPORT CONTRACT SUMMARY")
    print("=" * 112)
    print(f"Gate Zero: {gate['final_gate']}")
    print(f"Strict one-to-one ortholog pairs: {len(strict):,}")
    print(f"Primary DOG2 -> TARGET-OS features: {primary_n:,}")
    print()
    print("Frozen scientific roles:")
    print("  Primary: DOG2 OS -> TARGET-OS OS | regularized residual Cox transfer")
    print("  Secondary: DOG2 DFI/source representation -> GSE21257 5y metastasis")
    print("  Stress: DOG2 source -> GSE39055 RFS | evaluation only, never tuning")
    print()
    print("Outcome/response/follow-up values read: NO")
    print("Clinical values read: NO")
    print("Expression values read: NO")
    print("Model fitting: NO")
    print("Target splits generated: NO")
    print()
    print("Artifacts:")
    print(f"  {STRICT_UNIVERSE_TSV.relative_to(ROOT)}")
    print(f"  {FEATURE_SETS_TSV.relative_to(ROOT)}")
    print(f"  {FEATURE_SUMMARY_TSV.relative_to(ROOT)}")
    print(f"  {CONTRACT_JSON.relative_to(ROOT)}")
    print(f"  {SUMMARY_JSON.relative_to(ROOT)}")
    print()
    print("Next:")
    print("  Freeze exact endpoint definitions + resampling/tuning contract.")
    print("  Do NOT fit any outcome model before that contract passes.")
    print()
    print("02f0 frozen pre-outcome cross-species transport contract: PASS")
    print("=" * 112)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 112, file=sys.stderr)
        print("02f0 frozen pre-outcome cross-species transport contract: FAIL", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print("=" * 112, file=sys.stderr)
        raise
