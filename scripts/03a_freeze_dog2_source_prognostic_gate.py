#!/usr/bin/env python3
"""
Paper 6 - freeze exact computational DOG2 Source Prognostic Gate protocol.

This is a MECHANICAL implementation freeze under the already-passed 02i
scientific contract.

03a DOES NOT read DOG2 outcome values. It may read:
- the clinical CSV HEADER only, to lock exact endpoint column names;
- sample/case IDs and randomized study labels from already-frozen Paper-6
  mapping/roster artifacts;
- the 02g bridge table and 02h Hallmark GMT definitions, which are outcome-blind.

03a MAY freeze:
- exact resampling counts and seeds;
- exact regularization grids/path rules for model families already approved by 02i;
- exact fold-safe preprocessing;
- bootstrap/permutation counts;
- exact Uno-C evaluation support rule.

03a MUST NOT:
- change PASS/WARN/FAIL thresholds;
- change OS/DFI hierarchy;
- add model families;
- change premise-test criteria;
- open human outcomes;
- read DOG2 outcome values.

No CLI arguments.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


SCRIPT_VERSION = "03a-freeze-dog2-source-prognostic-gate-protocol-v1-no-cli"

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "_config"
PAPER4_BASENAME = "paper4_sarcoma_dog"

# Final broad scientific design freeze.
I_DIR = ROOT / "results" / "revised_design" / "02i"
I_CONTRACT = I_DIR / "revised_model_benchmark_source_gate_contract.json"
I_SUMMARY = I_DIR / "summary.json"

# Outcome-blind bridge/module assets.
G_DIR = ROOT / "results" / "ortholog_bridge" / "02g_v3"
G_BRIDGE = G_DIR / "primary_outcome_blind_ortholog_bridge.tsv"
G_SUMMARY = G_DIR / "summary.json"
G_CONTRACT = G_DIR / "ortholog_bridge_contract.json"

H_DIR = ROOT / "results" / "module_coverage" / "02h"
H_HALLMARK = H_DIR / "reference" / "h.all.v2026.1.Hs.symbols.gmt"
H_REF_LOCK = H_DIR / "reference" / "msigdb_reference_lock.json"
H_SUMMARY = H_DIR / "summary.json"
H_CONTRACT = H_DIR / "module_coverage_contract.json"

# Frozen source mapping/roster.
MAP_LOCK = ROOT / "contracts" / "02a_dog2_authoritative_id_mapping_lock.json"
MAP_FILE = ROOT / "manifests" / "02a_dog2_selected186_authoritative_id_mapping.csv"

E7_DIR = ROOT / "results" / "dog2_selection_audit" / "02e7"
E7_ROSTER = E7_DIR / "dog2_rna186_roster.tsv"
E7_SUMMARY = E7_DIR / "summary.json"

# Script-00 upstream lock for clinical/expression file identity.
UPSTREAM_LOCK = ROOT / "contracts" / "00_upstream_input_lock.json"

# Outputs.
OUT_DIR = ROOT / "results" / "source_gate_protocol" / "03a"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ENDPOINT_HEADER_TSV = OUT_DIR / "endpoint_header_contract.tsv"
SOURCE_ROSTER_TSV = OUT_DIR / "source_sample_arm_roster.tsv"
HALLMARK_MAP_TSV = OUT_DIR / "hallmark_dog2_feature_map.tsv"
RIDGE_GRID_TSV = OUT_DIR / "ridge_alpha_grid.tsv"
ELASTIC_NET_TSV = OUT_DIR / "elastic_net_path_contract.tsv"
PROTOCOL_JSON = OUT_DIR / "source_prognostic_gate_protocol.json"
SUMMARY_JSON = OUT_DIR / "summary.json"

EXPECTED_N = 186
EXPECTED_ARM_COUNTS = {"COTC021": 93, "COTC022": 93}
EXPECTED_PRIMARY_BRIDGE = 11815
EXPECTED_HALLMARK_N = 50

# These are previously known upstream cohort facts, frozen here as 03b identity
# checks. 03a DOES NOT read the values needed to verify them.
EXPECTED_ENDPOINT_COUNTS = {
    "OS": {
        "n_complete": 186,
        "events": 124,
    },
    "DFI": {
        "n_complete": 186,
        "events": 143,
    },
}

ENDPOINTS = {
    "OS": {
        "role": "PRIMARY",
        "time_col": "os_time",
        "event_col": "os_event",
    },
    "DFI": {
        "role": "SECONDARY",
        "time_col": "dfi_time",
        "event_col": "dfi_event",
    },
}
PATIENT_ID_COL = "Patient ID"

# ---------------------------------------------------------------------------
# Exact computation freeze.
# ---------------------------------------------------------------------------

BASE_SEED = 20260829

PRIMARY_OUTER_SPLITS = 5
PRIMARY_OUTER_REPEATS = 20
PRIMARY_INNER_SPLITS = 4

SUPPORTING_OUTER_SPLITS = 5
SUPPORTING_OUTER_REPEATS = 5
SUPPORTING_INNER_SPLITS = 3

ARM_INNER_SPLITS = 4

PATIENT_BOOTSTRAPS = 2000
ARM_TEST_BOOTSTRAPS = 2000

PERMUTATIONS = 100
PERMUTATION_CV_SPLITS = 5
PERMUTATION_CV_REPEATS = 1

UNO_TAU_TRAIN_QUANTILE = 0.90

# Hallmark ridge Cox.
RIDGE_ALPHAS = np.logspace(-4, 4, 17)

# Gene-level elastic-net supporting model.
ELASTIC_L1_RATIOS = [0.10, 0.50, 0.90]
ELASTIC_N_ALPHAS = 30
ELASTIC_ALPHA_MIN_RATIO = 0.01

# Deterministic tuning tie tolerance in Uno C units.
INNER_TIE_TOLERANCE = 0.005

# Fold-safety.
MIN_HALLMARK_GENES_PER_MODULE_IN_TRAIN = 10
ZERO_VARIANCE_SD_EPS = 1e-12

# Primary Source Gate thresholds already frozen by 02i.
SOURCE_PASS_C = 0.60
SOURCE_WARN_FLOOR_C = 0.55
CHANCE_C = 0.50

ARM_PASS_MEAN_C = 0.55
ARM_WARN_MEAN_C = 0.52
ARM_MAX_DROP_FROM_RANDOM_CV = 0.05

# Supporting gene-model PASS means it independently satisfies the same material
# Source PASS discrimination rule frozen by 02i.
SUPPORTING_PASS_C = 0.60
SUPPORTING_PASS_LOWER_CI = 0.50


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "na", "n/a"}:
        return ""
    return text


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
        raise FileNotFoundError(f"Required frozen artifact missing: {path}")
    return path


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
        sentinel = (
            resolved
            / "data"
            / "processed"
            / "GSE238110_DOG2_clinical_matched_indexed.csv"
        )
        if resolved.is_dir() and sentinel.exists():
            return resolved, source

    raise FileNotFoundError(
        "Could not resolve Paper 4 root. Checked:\n  - " + "\n  - ".join(checked)
    )


def get_asset(
    lock: Dict[str, Any],
    role: str,
    paper4_root: Path,
) -> Tuple[Path, Dict[str, Any]]:
    assets = lock.get("assets") or {}
    item = assets.get(role)
    if not isinstance(item, dict):
        raise RuntimeError(f"Upstream lock missing asset role {role!r}.")

    relative = clean(item.get("relative_path"))
    expected_hash = clean(item.get("sha256")).lower()
    if not relative or len(expected_hash) != 64:
        raise RuntimeError(f"Malformed upstream asset metadata for {role!r}.")

    path = paper4_root / relative
    require_file(path)

    actual_hash = sha256_file(path).lower()
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"Upstream hash mismatch for {role}: "
            f"expected={expected_hash}, actual={actual_hash}"
        )
    return path, item


def bool_mask(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False).astype(bool)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y", "t"})
    )


def parse_gmt(path: Path) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue

            module = parts[0].strip()
            genes: List[str] = []
            seen = set()

            for value in parts[2:]:
                gene = value.strip().upper()
                if not gene or gene in seen:
                    continue
                seen.add(gene)
                genes.append(gene)

            result[module] = genes

    return result


def load_source_roster() -> pd.DataFrame:
    require_file(MAP_FILE)
    require_file(MAP_LOCK)
    require_file(E7_ROSTER)
    require_file(E7_SUMMARY)

    map_lock = read_json(MAP_LOCK)
    if clean(map_lock.get("status")) != "PASS":
        raise RuntimeError("02a mapping lock is not PASS.")

    expected_map_hash = clean(map_lock.get("output_mapping_sha256"))
    if expected_map_hash != sha256_file(MAP_FILE):
        raise RuntimeError("02a mapping hash verification failed.")

    mapping = pd.read_csv(MAP_FILE, dtype=str).fillna("")
    required_mapping = {
        "paper4_sample_id",
        "paper4_patient_id",
        "cotc_subject_id",
    }
    if not required_mapping.issubset(mapping.columns):
        raise RuntimeError(
            f"02a mapping lacks columns: {sorted(required_mapping - set(mapping.columns))}"
        )

    mapping = mapping[
        ["paper4_sample_id", "paper4_patient_id", "cotc_subject_id"]
    ].copy()

    roster = pd.read_csv(E7_ROSTER, sep="\t", dtype=str).fillna("")
    required_roster = {"case_id", "study", "gate_zero_arm", "patient_id"}
    if not required_roster.issubset(roster.columns):
        raise RuntimeError(
            f"02e7 roster lacks columns: {sorted(required_roster - set(roster.columns))}"
        )

    out = mapping.merge(
        roster[
            ["case_id", "study", "gate_zero_arm", "patient_id"]
        ],
        left_on="cotc_subject_id",
        right_on="case_id",
        how="inner",
        validate="one_to_one",
    )

    if len(out) != EXPECTED_N:
        raise RuntimeError(f"Source roster rows={len(out)}, expected 186.")
    if out["paper4_sample_id"].nunique() != EXPECTED_N:
        raise RuntimeError("Source roster paper4_sample_id is not unique.")
    if out["cotc_subject_id"].nunique() != EXPECTED_N:
        raise RuntimeError("Source roster COTC subject ID is not unique.")

    counts = out["study"].value_counts().to_dict()
    observed = {
        "COTC021": int(counts.get("COTC021", 0)),
        "COTC022": int(counts.get("COTC022", 0)),
    }
    if observed != EXPECTED_ARM_COUNTS:
        raise RuntimeError(f"Source arm counts changed: {observed}")

    expected_arm = {
        "COTC021": "SOC_PLUS_RAPAMYCIN",
        "COTC022": "SOC_CONTROL",
    }
    bad = out[
        out.apply(
            lambda r: clean(r["gate_zero_arm"]) != expected_arm[clean(r["study"])],
            axis=1,
        )
    ]
    if not bad.empty:
        raise RuntimeError("Source roster contains study/arm mapping inconsistency.")

    return out.sort_values("paper4_sample_id").reset_index(drop=True)


def build_hallmark_feature_map() -> pd.DataFrame:
    require_file(G_BRIDGE)
    require_file(H_HALLMARK)
    require_file(H_REF_LOCK)

    bridge = pd.read_csv(G_BRIDGE, sep="\t", dtype=str, low_memory=False).fillna("")
    required = {
        "dog2_raw_feature",
        "dog_gene_symbol",
        "human_gene_symbol",
        "primary_dog2_to_target_os",
    }
    if not required.issubset(bridge.columns):
        raise RuntimeError(
            f"02g bridge lacks columns: {sorted(required - set(bridge.columns))}"
        )

    primary = bridge[bool_mask(bridge["primary_dog2_to_target_os"])].copy()
    if len(primary) != EXPECTED_PRIMARY_BRIDGE:
        raise RuntimeError(
            f"02g primary bridge rows={len(primary)}, expected {EXPECTED_PRIMARY_BRIDGE}."
        )

    primary["human_gene_symbol_upper"] = (
        primary["human_gene_symbol"].astype(str).str.strip().str.upper()
    )

    if primary["human_gene_symbol_upper"].duplicated().any():
        raise RuntimeError("02g primary bridge has duplicate human symbols.")
    if primary["dog2_raw_feature"].duplicated().any():
        raise RuntimeError("02g primary bridge has duplicate DOG2 raw features.")

    human_to_row = {
        row["human_gene_symbol_upper"]: row
        for _, row in primary.iterrows()
    }

    hallmark = parse_gmt(H_HALLMARK)
    if len(hallmark) != EXPECTED_HALLMARK_N:
        raise RuntimeError(
            f"Hallmark GMT contains {len(hallmark)} modules, expected 50."
        )

    rows: List[Dict[str, Any]] = []

    for module in sorted(hallmark):
        mapped = 0
        for human_gene in hallmark[module]:
            if human_gene not in human_to_row:
                continue
            source = human_to_row[human_gene]
            rows.append(
                {
                    "hallmark_module": module,
                    "human_gene_symbol": human_gene,
                    "dog_gene_symbol": clean(source["dog_gene_symbol"]),
                    "dog2_raw_feature": clean(source["dog2_raw_feature"]),
                }
            )
            mapped += 1

        if mapped < MIN_HALLMARK_GENES_PER_MODULE_IN_TRAIN:
            raise RuntimeError(
                f"{module}: only {mapped} mapped genes, expected at least "
                f"{MIN_HALLMARK_GENES_PER_MODULE_IN_TRAIN}."
            )

    frame = pd.DataFrame(rows)
    if frame["hallmark_module"].nunique() != EXPECTED_HALLMARK_N:
        raise RuntimeError("Not all 50 Hallmark modules are represented.")

    return frame.sort_values(
        ["hallmark_module", "human_gene_symbol", "dog2_raw_feature"]
    ).reset_index(drop=True)


def main() -> None:
    started = now_utc()

    print("=" * 120)
    print("Paper 6 - freeze exact DOG2 Source Prognostic Gate computational protocol")
    print("=" * 120)
    print(f"Script version: {SCRIPT_VERSION}")
    print()
    print("Safety / execution contract:")
    print("  DOG2 clinical CSV header read: YES")
    print("  DOG2 clinical/outcome VALUES read: NO")
    print("  DOG2 expression VALUES read: NO")
    print("  Human outcome/clinical values read: NO")
    print("  Model fitting: NO")
    print("  Split generation: NO")
    print("  Scientific thresholds changed from 02i: NO")
    print()

    for path in [
        I_CONTRACT,
        I_SUMMARY,
        G_BRIDGE,
        G_SUMMARY,
        G_CONTRACT,
        H_HALLMARK,
        H_REF_LOCK,
        H_SUMMARY,
        H_CONTRACT,
        UPSTREAM_LOCK,
    ]:
        require_file(path)

    i_contract = read_json(I_CONTRACT)
    i_summary = read_json(I_SUMMARY)
    g_summary = read_json(G_SUMMARY)
    h_summary = read_json(H_SUMMARY)
    upstream = read_json(UPSTREAM_LOCK)

    if clean(i_summary.get("status")) != "PASS":
        raise RuntimeError("02i summary is not PASS.")
    if clean(i_summary.get("scientific_status")) != (
        "PASS_FINAL_BROAD_PREOUTCOME_DESIGN_FREEZE"
    ):
        raise RuntimeError("02i is not in final broad pre-outcome PASS state.")
    if clean(i_contract.get("status")) != "PASS":
        raise RuntimeError("02i contract is not PASS.")

    # Verify 02i threshold identity rather than re-defining it silently.
    source_gate = i_contract.get("source_gate") or {}
    arm_gate = i_contract.get("arm_to_arm_generalization") or {}

    if float(source_gate["source_pass"]["uno_c_min"]) != SOURCE_PASS_C:
        raise RuntimeError("02i source PASS Uno-C threshold changed.")
    if float(source_gate["source_warn_material_floor"]) != SOURCE_WARN_FLOOR_C:
        raise RuntimeError("02i source WARN floor changed.")
    if float(arm_gate["arm_pass"]["mean_directional_uno_c_min"]) != ARM_PASS_MEAN_C:
        raise RuntimeError("02i ARM PASS threshold changed.")
    if float(arm_gate["arm_warn"]["mean_directional_uno_c_min"]) != ARM_WARN_MEAN_C:
        raise RuntimeError("02i ARM WARN threshold changed.")

    if int(g_summary["measured_counts"]["primary_dog2_to_target_os"]) != EXPECTED_PRIMARY_BRIDGE:
        raise RuntimeError("02g primary bridge count changed.")
    if int(h_summary["Hallmark_primary_eligible"]) != EXPECTED_HALLMARK_N:
        raise RuntimeError("02h Hallmark primary coverage changed.")

    paper4_root, paper4_source = resolve_paper4_root()
    clinical_path, clinical_asset = get_asset(
        upstream,
        "dog2_clinical",
        paper4_root,
    )
    expression_path, expression_asset = get_asset(
        upstream,
        "dog2_expression",
        paper4_root,
    )

    print(f"Paper 4 root: {paper4_root}")
    print(f"Resolution source: {paper4_source}")
    print("Locked DOG2 assets:")
    print(f"  clinical SHA256: {clinical_asset['sha256']}")
    print(f"  expression SHA256: {expression_asset['sha256']}")
    print()

    # Header only. NO clinical row/value is read here.
    clinical_header = pd.read_csv(clinical_path, nrows=0)
    clinical_columns = list(clinical_header.columns)

    required_endpoint_columns = {
        PATIENT_ID_COL,
        "os_time",
        "os_event",
        "dfi_time",
        "dfi_event",
    }
    missing = sorted(required_endpoint_columns - set(clinical_columns))
    if missing:
        raise RuntimeError(
            f"Locked DOG2 clinical header lacks required column(s): {missing}"
        )

    endpoint_rows = []
    for endpoint, spec in ENDPOINTS.items():
        endpoint_rows.append(
            {
                "endpoint": endpoint,
                "role": spec["role"],
                "time_col": spec["time_col"],
                "event_col": spec["event_col"],
                "patient_id_col": PATIENT_ID_COL,
                "expected_complete_n_for_03b_identity_check": (
                    EXPECTED_ENDPOINT_COUNTS[endpoint]["n_complete"]
                ),
                "expected_events_for_03b_identity_check": (
                    EXPECTED_ENDPOINT_COUNTS[endpoint]["events"]
                ),
                "values_read_in_03a": False,
            }
        )
    pd.DataFrame(endpoint_rows).to_csv(
        ENDPOINT_HEADER_TSV,
        sep="\t",
        index=False,
    )

    source_roster = load_source_roster()
    source_roster.to_csv(SOURCE_ROSTER_TSV, sep="\t", index=False)

    hallmark_map = build_hallmark_feature_map()
    hallmark_map.to_csv(HALLMARK_MAP_TSV, sep="\t", index=False)

    module_counts = (
        hallmark_map.groupby("hallmark_module")["dog2_raw_feature"]
        .nunique()
        .sort_index()
    )

    pd.DataFrame(
        {
            "alpha": [float(x) for x in RIDGE_ALPHAS],
            "log10_alpha": [float(np.log10(x)) for x in RIDGE_ALPHAS],
            "tie_preference_rank": list(range(len(RIDGE_ALPHAS), 0, -1)),
        }
    ).to_csv(RIDGE_GRID_TSV, sep="\t", index=False)

    elastic_rows = []
    for l1_ratio in ELASTIC_L1_RATIOS:
        elastic_rows.append(
            {
                "l1_ratio": l1_ratio,
                "n_alphas": ELASTIC_N_ALPHAS,
                "alpha_min_ratio": ELASTIC_ALPHA_MIN_RATIO,
                "path_selection_unit": "relative_alpha_path_index",
                "tie_rule": (
                    "within 0.005 inner Uno C of best -> stronger regularization "
                    "(lower path index), then larger l1_ratio"
                ),
            }
        )
    pd.DataFrame(elastic_rows).to_csv(
        ELASTIC_NET_TSV,
        sep="\t",
        index=False,
    )

    sksurv_available = importlib.util.find_spec("sksurv") is not None

    protocol = {
        "script_version": SCRIPT_VERSION,
        "status": "PASS",
        "created_utc": now_utc(),
        "scientific_stage": "SOURCE_PROGNOSTIC_GATE_COMPUTATIONAL_FREEZE",
        "upstream": {
            "02i_contract_sha256": sha256_file(I_CONTRACT),
            "02i_summary_sha256": sha256_file(I_SUMMARY),
            "02g_bridge_sha256": sha256_file(G_BRIDGE),
            "02h_hallmark_sha256": sha256_file(H_HALLMARK),
            "02a_mapping_sha256": sha256_file(MAP_FILE),
            "02e7_roster_sha256": sha256_file(E7_ROSTER),
            "dog2_clinical_sha256": clean(clinical_asset.get("sha256")),
            "dog2_expression_sha256": clean(expression_asset.get("sha256")),
        },
        "cohort": {
            "n": EXPECTED_N,
            "arm_counts": EXPECTED_ARM_COUNTS,
            "sample_arm_roster_artifact": str(
                SOURCE_ROSTER_TSV.relative_to(ROOT)
            ),
            "clinical_alignment": (
                "paper4_patient_id from frozen 02a mapping -> locked clinical 'Patient ID'; "
                "paper4_sample_id -> locked DOG2 expression index"
            ),
        },
        "endpoints": {
            endpoint: {
                **spec,
                "expected_complete_n_for_identity_check": (
                    EXPECTED_ENDPOINT_COUNTS[endpoint]["n_complete"]
                ),
                "expected_events_for_identity_check": (
                    EXPECTED_ENDPOINT_COUNTS[endpoint]["events"]
                ),
            }
            for endpoint, spec in ENDPOINTS.items()
        },
        "endpoint_hierarchy": {
            "primary": "OS",
            "secondary": "DFI",
            "posthoc_switch": "FORBIDDEN",
            "discordance_rule_source": "02i",
        },
        "primary_model": {
            "name": "Hallmark-50 ridge Cox",
            "library": "MSigDB Hallmark 2026.1.Hs",
            "n_modules": EXPECTED_HALLMARK_N,
            "module_feature_map": str(HALLMARK_MAP_TSV.relative_to(ROOT)),
            "mapped_genes_per_module_min": int(module_counts.min()),
            "mapped_genes_per_module_median": float(module_counts.median()),
            "mapped_genes_per_module_max": int(module_counts.max()),
            "fold_safe_preprocessing": [
                "select fixed outcome-blind DOG2 feature list from frozen module map",
                "within training fold: compute gene mean and population SD",
                "training-SD <= 1e-12 gene is excluded for that fold only",
                "apply training gene mean/SD to validation/test",
                "module score = unweighted mean of available gene z-scores",
                "require >=10 training-nonzero genes per Hallmark module",
                "within training fold: standardize 50 module scores",
                "apply training module mean/SD to validation/test",
                "fit ridge Cox on 50 standardized module scores",
            ],
            "ridge_alpha_grid": [float(x) for x in RIDGE_ALPHAS],
            "inner_selection_metric": "Uno C",
            "inner_tie_tolerance": INNER_TIE_TOLERANCE,
            "inner_tie_rule": (
                "choose largest alpha among settings within 0.005 Uno C of the best"
            ),
            "arm_as_predictor": False,
            "clinical_covariates_as_predictors": False,
        },
        "supporting_model": {
            "name": "11,815-gene elastic-net Cox",
            "role": "SUPPORTING_ONLY",
            "feature_universe": "02g primary_dog2_to_target_os",
            "fold_safe_preprocessing": [
                "fixed outcome-blind 11,815-gene list",
                "remove training-fold zero-variance genes only",
                "training-fold standardization",
                "apply training transformation to validation/test",
            ],
            "l1_ratio_grid": ELASTIC_L1_RATIOS,
            "n_alphas_per_path": ELASTIC_N_ALPHAS,
            "alpha_min_ratio": ELASTIC_ALPHA_MIN_RATIO,
            "relative_path_index_selection": True,
            "inner_tie_tolerance": INNER_TIE_TOLERANCE,
            "inner_tie_rule": (
                "within 0.005 Uno C of best: choose stronger regularization "
                "(lower relative path index), then larger l1_ratio"
            ),
            "cannot_replace_primary_model": True,
        },
        "random_cv": {
            "primary": {
                "outer_splits": PRIMARY_OUTER_SPLITS,
                "outer_repeats": PRIMARY_OUTER_REPEATS,
                "inner_splits": PRIMARY_INNER_SPLITS,
                "base_seed": BASE_SEED,
                "outer_stratification": "study_arm x endpoint_event",
                "inner_stratification": "study_arm x endpoint_event",
                "single_stratum_fallback": "NONE_FAIL_CLOSED",
                "prediction_scale": (
                    "within each outer fold standardize test linear predictor using "
                    "training-fold linear-predictor mean/SD"
                ),
                "repeat_aggregation": (
                    "one OOF prediction per patient per repeat; mean standardized OOF "
                    "risk across 20 repeats"
                ),
            },
            "supporting": {
                "outer_splits": SUPPORTING_OUTER_SPLITS,
                "outer_repeats": SUPPORTING_OUTER_REPEATS,
                "inner_splits": SUPPORTING_INNER_SPLITS,
                "base_seed": BASE_SEED + 50000,
                "outer_stratification": "study_arm x endpoint_event",
                "inner_stratification": "study_arm x endpoint_event",
            },
        },
        "uno_c": {
            "primary_metric": True,
            "orientation": "higher linear predictor = higher event risk",
            "chance_reference": CHANCE_C,
            "tau_rule": (
                "for each train->validation/test evaluation, tau = 90th percentile "
                "of training follow-up time; for final aggregated OOF metric, tau = "
                "90th percentile of full source follow-up"
            ),
            "tau_quantile": UNO_TAU_TRAIN_QUANTILE,
            "censoring_distribution": (
                "IPCW estimated from the corresponding training survival data; "
                "final OOF bootstrap uses full source endpoint as IPCW reference"
            ),
        },
        "patient_bootstrap": {
            "replicates": PATIENT_BOOTSTRAPS,
            "seed": BASE_SEED + 1000,
            "unit": "patient",
            "statistic": "Uno C of final aggregated cross-fitted risk",
            "percentile_ci": [0.025, 0.975],
            "model_refit_in_bootstrap": False,
            "interpretation": (
                "uncertainty of cross-fitted patient-level discrimination; "
                "model-fitting variation is represented by repeated CV, not by bootstrap refitting"
            ),
        },
        "arm_to_arm": {
            "directions": [
                "COTC021->COTC022",
                "COTC022->COTC021",
            ],
            "primary_model_family": "same Hallmark-50 ridge Cox as random CV",
            "inner_splits": ARM_INNER_SPLITS,
            "inner_stratification": "endpoint_event only",
            "base_seed": BASE_SEED + 100000,
            "test_bootstrap_replicates": ARM_TEST_BOOTSTRAPS,
            "test_bootstrap_seed": BASE_SEED + 101000,
            "bootstrap_unit": "target-arm patient",
            "training_model_refit_in_test_bootstrap": False,
            "decision_thresholds_source": "02i",
        },
        "permutation_supporting_inference": {
            "enabled": True,
            "endpoint": "OS",
            "model": "primary Hallmark-50 ridge Cox only",
            "permutations": PERMUTATIONS,
            "seed": BASE_SEED + 200000,
            "permutation_unit": "(time,event) outcome pair",
            "permutation_scope": "shuffle outcome pairs within randomized study arm",
            "cv_splits": PERMUTATION_CV_SPLITS,
            "cv_repeats": PERMUTATION_CV_REPEATS,
            "observed_statistic_recomputed_on_same_permutation_CV_design": True,
            "full_training_pipeline_repeated": True,
            "role": "SUPPORTING_ONLY_DOES_NOT_UPGRADE_MATERIAL_GATE",
        },
        "endpoint_execution": {
            "OS": {
                "primary_Hallmark_random_cv": True,
                "primary_Hallmark_arm_to_arm": True,
                "supporting_gene_elastic_net_random_cv": True,
                "permutation_supporting_inference": True,
            },
            "DFI": {
                "primary_Hallmark_random_cv": True,
                "primary_Hallmark_arm_to_arm": True,
                "supporting_gene_elastic_net_random_cv": False,
                "permutation_supporting_inference": False,
                "DFI_PASS_definition": (
                    "same Hallmark primary random-CV Source PASS discrimination rule "
                    "used for OS; DFI cannot replace OS without a new amendment"
                ),
            },
        },
        "source_gate_thresholds_locked_from_02i": {
            "source_pass": (
                "primary Hallmark OOS Uno C >=0.60 AND lower 95% patient-bootstrap CI >0.50"
            ),
            "source_warn_floor": 0.55,
            "source_fail": (
                "primary Hallmark OOS Uno C <0.55 AND OS supporting gene-level model "
                "does not independently meet source PASS"
            ),
            "supporting_gene_pass": (
                "OOS Uno C >=0.60 AND lower 95% patient-bootstrap CI >0.50"
            ),
            "arm_pass": (
                "mean directional Uno C >=0.55; neither directional point estimate <0.50; "
                "mean arm->arm drop vs primary random CV <=0.05"
            ),
            "arm_warn": (
                "ARM_PASS not met but mean directional Uno C >=0.52 and no "
                "directional upper 95% bootstrap CI <0.50"
            ),
            "arm_fail": (
                "mean directional Uno C <0.52 OR any directional upper 95% bootstrap CI <0.50"
            ),
        },
        "combined_gate_matrix": i_contract.get("combined_source_gate_matrix"),
        "endpoint_discordance_rule": (
            i_contract.get("source_endpoint_hierarchy") or {}
        ).get("discordance_rules"),
        "runtime": {
            "CPU_only": True,
            "recommended_max_parallel_workers": 4,
            "GPU_required": False,
            "required_python_package_for_03b": "scikit-survival",
            "sksurv_currently_importable_in_03a_environment": sksurv_available,
            "do_not_modify_CUDA_driver_or_Paper5_GPU_environment": True,
        },
        "03b_fail_closed_checks_before_model_fit": [
            "locked clinical and expression hashes still match script-00",
            "clinical Patient ID maps 186/186 one-to-one to frozen 02a paper4_patient_id",
            "expression index maps 186/186 one-to-one to frozen 02a paper4_sample_id",
            "OS complete n == 186 and OS events == 124",
            "DFI complete n == 186 and DFI events == 143",
            "event indicators are binary",
            "survival times are finite and strictly positive",
            "study counts remain 93/93",
            "all 50 Hallmark modules retain >=10 nonzero training genes in every fold",
        ],
        "safety": {
            "DOG2_clinical_header_read": True,
            "DOG2_clinical_values_read": False,
            "DOG2_outcome_values_read": False,
            "DOG2_expression_values_read": False,
            "human_outcome_values_read": False,
            "model_fitting": False,
            "splits_generated": False,
        },
    }
    write_json(PROTOCOL_JSON, protocol)

    final_hashes = {
        "endpoint_header_contract_tsv": sha256_file(ENDPOINT_HEADER_TSV),
        "source_sample_arm_roster_tsv": sha256_file(SOURCE_ROSTER_TSV),
        "hallmark_dog2_feature_map_tsv": sha256_file(HALLMARK_MAP_TSV),
        "ridge_alpha_grid_tsv": sha256_file(RIDGE_GRID_TSV),
        "elastic_net_path_contract_tsv": sha256_file(ELASTIC_NET_TSV),
        "source_prognostic_gate_protocol_json": sha256_file(PROTOCOL_JSON),
    }

    summary = {
        "script_version": SCRIPT_VERSION,
        "status": "PASS",
        "scientific_status": "PASS_SOURCE_PROGNOSTIC_GATE_PROTOCOL_FROZEN",
        "run_started_utc": started,
        "run_finished_utc": now_utc(),
        "primary_endpoint": "OS",
        "secondary_endpoint": "DFI",
        "clinical_header_columns_verified": sorted(required_endpoint_columns),
        "source_n": EXPECTED_N,
        "arm_counts": EXPECTED_ARM_COUNTS,
        "Hallmark_modules": int(hallmark_map["hallmark_module"].nunique()),
        "Hallmark_mapped_gene_rows": int(len(hallmark_map)),
        "Hallmark_min_mapped_genes": int(module_counts.min()),
        "Hallmark_median_mapped_genes": float(module_counts.median()),
        "Hallmark_max_mapped_genes": int(module_counts.max()),
        "primary_outer_splits": PRIMARY_OUTER_SPLITS,
        "primary_outer_repeats": PRIMARY_OUTER_REPEATS,
        "primary_inner_splits": PRIMARY_INNER_SPLITS,
        "supporting_outer_repeats": SUPPORTING_OUTER_REPEATS,
        "patient_bootstraps": PATIENT_BOOTSTRAPS,
        "arm_test_bootstraps": ARM_TEST_BOOTSTRAPS,
        "permutations": PERMUTATIONS,
        "base_seed": BASE_SEED,
        "sksurv_importable": sksurv_available,
        "final_artifact_hashes": final_hashes,
        "DOG2_clinical_values_read": False,
        "DOG2_outcome_values_read": False,
        "DOG2_expression_values_read": False,
        "human_outcome_values_read": False,
        "model_fitting": False,
        "splits_generated": False,
        "next": "03b run frozen DOG2 Source Prognostic Gate",
    }
    write_json(SUMMARY_JSON, summary)

    print("-" * 120)
    print("Frozen endpoint header contract")
    print("-" * 120)
    print("Primary: OS -> os_time / os_event")
    print("Secondary: DFI -> dfi_time / dfi_event")
    print("Patient alignment column: Patient ID")
    print("Clinical values read in 03a: NO")
    print()

    print("Frozen source cohort:")
    print("  n=186")
    print("  COTC021=93")
    print("  COTC022=93")
    print()

    print("Frozen Hallmark primary representation:")
    print(f"  modules: {hallmark_map['hallmark_module'].nunique()}")
    print(f"  mapped gene rows: {len(hallmark_map):,}")
    print(f"  min genes/module: {int(module_counts.min())}")
    print(f"  median genes/module: {float(module_counts.median()):.1f}")
    print(f"  max genes/module: {int(module_counts.max())}")
    print()

    print("Frozen primary random CV:")
    print(f"  outer: {PRIMARY_OUTER_SPLITS}-fold x {PRIMARY_OUTER_REPEATS} repeats")
    print(f"  inner: {PRIMARY_INNER_SPLITS}-fold")
    print("  stratification: study arm x event")
    print(f"  ridge alpha grid: {len(RIDGE_ALPHAS)} values, 1e-4 ... 1e4")
    print(f"  tie tolerance: {INNER_TIE_TOLERANCE:.3f} Uno C")
    print()

    print("Frozen supporting gene elastic-net:")
    print(
        f"  outer: {SUPPORTING_OUTER_SPLITS}-fold x "
        f"{SUPPORTING_OUTER_REPEATS} repeats"
    )
    print(f"  inner: {SUPPORTING_INNER_SPLITS}-fold")
    print(f"  l1 ratios: {ELASTIC_L1_RATIOS}")
    print(
        f"  alpha path: {ELASTIC_N_ALPHAS} values, "
        f"alpha_min_ratio={ELASTIC_ALPHA_MIN_RATIO}"
    )
    print()

    print("Frozen uncertainty / null inference:")
    print(f"  patient bootstraps: {PATIENT_BOOTSTRAPS}")
    print(f"  arm test bootstraps: {ARM_TEST_BOOTSTRAPS}")
    print(
        f"  OS outcome-pair permutations within arm: {PERMUTATIONS} "
        f"({PERMUTATION_CV_SPLITS}-fold x {PERMUTATION_CV_REPEATS})"
    )
    print()

    print("Runtime dependency:")
    print(
        "  scikit-survival importable now: "
        f"{'YES' if sksurv_available else 'NO'}"
    )
    if not sksurv_available:
        print(
            "  Before 03b install in THIS Paper-6 .venv only: "
            "python -m pip install scikit-survival"
        )
    print("  GPU: NOT USED")
    print("  recommended parallel workers: <=4")
    print()

    print("=" * 120)
    print("03a SOURCE PROGNOSTIC GATE PROTOCOL SUMMARY")
    print("=" * 120)
    print("Scientific PASS/WARN/FAIL thresholds: unchanged from 02i")
    print("OS/DFI hierarchy: unchanged from 02i")
    print("Human outcomes read: NO")
    print("DOG2 outcomes read: NO")
    print("DOG2 clinical values read: NO")
    print("DOG2 expression values read: NO")
    print("Model fitting: NO")
    print()
    print("Next: 03b opens locked DOG2 OS/DFI + expression and runs the gate.")
    print("No additional broad pre-result audit is scheduled.")
    print()
    print("Artifacts:")
    for path in [
        ENDPOINT_HEADER_TSV,
        SOURCE_ROSTER_TSV,
        HALLMARK_MAP_TSV,
        RIDGE_GRID_TSV,
        ELASTIC_NET_TSV,
        PROTOCOL_JSON,
        SUMMARY_JSON,
    ]:
        print(f"  {path.relative_to(ROOT)}")

    print("=" * 120)
    print("03a Source Prognostic Gate computational protocol freeze: PASS")
    print("=" * 120)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 120, file=sys.stderr)
        print(
            "03a Source Prognostic Gate computational protocol freeze: FAIL",
            file=sys.stderr,
        )
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print("=" * 120, file=sys.stderr)
        raise
