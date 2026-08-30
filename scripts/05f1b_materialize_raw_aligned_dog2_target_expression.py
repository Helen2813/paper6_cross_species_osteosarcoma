#!/usr/bin/env python3
"""
Paper 6 - materialize exact raw aligned DOG2/TARGET expression matrices.

Stage 05f1b follows the PASS 05f1a outcome-free representation freeze.

Scientific role
---------------
Read ONLY the two already-locked expression matrices and materialize the exact
11,815-gene DOG2<->TARGET raw aligned representation frozen by 05f1a.

Allowed operations:
- verify 05f1a contract/artifact hashes;
- verify locked upstream expression-file SHA256 values;
- read expression values for the exact frozen aligned columns only;
- parse numeric values as float64;
- reorder rows to the frozen 05f1a sample rosters;
- reorder columns to frozen aligned_feature_index;
- require all values finite;
- serialize immutable compressed NPZ matrices.

Forbidden operations:
- TARGET clinical file read;
- TARGET outcome/event read;
- GSE21257/GSE39055 outcome read;
- outcome-based sample intersection;
- imputation;
- variance filtering;
- gene scaling;
- Hallmark/module scoring;
- PCA/CORAL;
- sample deletion;
- model fitting;
- survival split generation.

Output matrices are RAW ALIGNED GENE MATRICES ONLY.
They are NOT model-ready Hallmark matrices.

No CLI arguments.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


SCRIPT_VERSION = (
    "05f1b-materialize-raw-aligned-dog2-target-expression-v1-no-cli"
)
MATERIALIZATION_VERSION = (
    "paper6-posthold-target-raw-aligned-expression-v1"
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "_config"
PAPER4_BASENAME = "paper4_sarcoma_dog"

# ---------------------------------------------------------------------------
# Frozen 05f1a.
# ---------------------------------------------------------------------------
F1A_DIR = ROOT / "results" / "human_posthold_descriptive" / "05f1a"
F1A_CONTRACT = F1A_DIR / "outcome_free_TARGET_representation_contract.json"
F1A_SUMMARY = F1A_DIR / "summary.json"
F1A_ALIGNMENT = F1A_DIR / "TARGET_primary_gene_alignment.tsv"
F1A_HALLMARK = F1A_DIR / "TARGET_hallmark50_gene_map.tsv"
F1A_DOG_ROSTER = F1A_DIR / "DOG2_expression_sample_roster.tsv"
F1A_TARGET_ROSTER = F1A_DIR / "TARGET_expression_sample_roster.tsv"
F1A_PREPROCESS = F1A_DIR / "TARGET_fold_safe_preprocessing_rules.tsv"
F1A_MODELS = F1A_DIR / "TARGET_frozen_model_registry.tsv"
F1A_BRANCHES = F1A_DIR / "TARGET_frozen_interpretation_branch_registry.tsv"
F1A_FLAGS = F1A_DIR / "TARGET_frozen_reporting_flags.tsv"

EXPECTED_F1A_CONTRACT_SHA256 = (
    "f0d5296d4df5e8dcb3b677127d2c3b5474791fb2cbaeb207e094b54fb44d1434"
)
EXPECTED_F1A_STATUS = (
    "PASS_OUTCOME_FREE_TARGET_REPRESENTATION_CONTRACT_FROZEN"
)

EXPECTED_DOG2_SAMPLES = 186
EXPECTED_TARGET_SAMPLES = 88
EXPECTED_ALIGNED_GENES = 11815
EXPECTED_HALLMARK_MODULES = 50

# ---------------------------------------------------------------------------
# Upstream expression asset lock.
# ---------------------------------------------------------------------------
UPSTREAM_LOCK = ROOT / "contracts" / "00_upstream_input_lock.json"

# ---------------------------------------------------------------------------
# Outputs.
# ---------------------------------------------------------------------------
OUT_DIR = ROOT / "results" / "human_posthold_descriptive" / "05f1b"
MATRIX_DIR = OUT_DIR / "matrices"

for directory in [OUT_DIR, MATRIX_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

DOG2_MATRIX = MATRIX_DIR / "DOG2_raw_aligned_11815genes.npz"
TARGET_MATRIX = MATRIX_DIR / "TARGET_OS_raw_aligned_11815genes.npz"
MATRIX_MANIFEST = OUT_DIR / "raw_aligned_matrix_manifest.tsv"
VALUE_AUDIT = OUT_DIR / "raw_aligned_value_integrity_audit.tsv"
SUMMARY_JSON = OUT_DIR / "summary.json"

MATRIX_DTYPE = np.float64


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text_lines(values: Sequence[str]) -> str:
    payload = "\n".join(str(x) for x in values) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_ndarray(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode("ascii"))
    digest.update(b"\n")
    digest.update(",".join(map(str, arr.shape)).encode("ascii"))
    digest.update(b"\n")
    digest.update(memoryview(arr).cast("B"))
    return digest.hexdigest()


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
    expected_hash = clean(item.get("sha256")).lower()

    if not rel or len(expected_hash) != 64:
        raise RuntimeError(
            f"Malformed upstream lock entry for {role!r}."
        )

    path = paper4_root / rel
    require_file(path)

    observed_hash = sha256_file(path).lower()
    if observed_hash != expected_hash:
        raise RuntimeError(
            f"Locked upstream asset changed: {role}; "
            f"expected={expected_hash}, observed={observed_hash}"
        )

    return path, item


def verify_f1a() -> Dict[str, Any]:
    required = [
        F1A_CONTRACT,
        F1A_SUMMARY,
        F1A_ALIGNMENT,
        F1A_HALLMARK,
        F1A_DOG_ROSTER,
        F1A_TARGET_ROSTER,
        F1A_PREPROCESS,
        F1A_MODELS,
        F1A_BRANCHES,
        F1A_FLAGS,
    ]

    for path in required:
        require_file(path)

    observed_contract_hash = sha256_file(F1A_CONTRACT)

    if observed_contract_hash != EXPECTED_F1A_CONTRACT_SHA256:
        raise RuntimeError(
            "05f1a contract SHA256 differs from completed PASS run. "
            f"Observed={observed_contract_hash}"
        )

    summary = read_json(F1A_SUMMARY)
    contract = read_json(F1A_CONTRACT)

    if clean(summary.get("scientific_status")) != EXPECTED_F1A_STATUS:
        raise RuntimeError(
            "05f1a summary is not in expected frozen PASS state."
        )

    if clean(contract.get("scientific_status")) != EXPECTED_F1A_STATUS:
        raise RuntimeError(
            "05f1a contract is not in expected frozen PASS state."
        )

    if clean(summary.get("contract_sha256")) != observed_contract_hash:
        raise RuntimeError(
            "05f1a summary/contract SHA256 mismatch."
        )

    frozen_hashes = summary.get("final_artifact_hashes") or {}

    artifact_map = {
        "TARGET_primary_gene_alignment.tsv": F1A_ALIGNMENT,
        "TARGET_hallmark50_gene_map.tsv": F1A_HALLMARK,
        "DOG2_expression_sample_roster.tsv": F1A_DOG_ROSTER,
        "TARGET_expression_sample_roster.tsv": F1A_TARGET_ROSTER,
        "TARGET_fold_safe_preprocessing_rules.tsv": F1A_PREPROCESS,
        "TARGET_frozen_model_registry.tsv": F1A_MODELS,
        "TARGET_frozen_interpretation_branch_registry.tsv": F1A_BRANCHES,
        "TARGET_frozen_reporting_flags.tsv": F1A_FLAGS,
        "outcome_free_TARGET_representation_contract.json": F1A_CONTRACT,
    }

    for name, path in artifact_map.items():
        expected = clean(frozen_hashes.get(name))
        if not expected:
            raise RuntimeError(
                f"05f1a summary lacks artifact hash for {name}."
            )
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(
                f"05f1a artifact changed: {name}; "
                f"expected={expected}, observed={observed}"
            )

    safety = contract.get("safety") or {}

    required_false = [
        "DOG2_expression_values_read",
        "TARGET_expression_values_read",
        "TARGET_clinical_values_read",
        "TARGET_outcomes_read",
        "GSE21257_outcomes_read",
        "GSE39055_outcomes_read",
        "TARGET_Hallmark_scores_computed",
        "whole_cohort_TARGET_scaler_fit",
        "survival_splits_generated",
        "model_fitting",
        "GPU_execution",
    ]

    for key in required_false:
        if safety.get(key) is not False:
            raise RuntimeError(
                f"05f1a safety field {key!r} is not False."
            )

    materialization = (
        contract.get("05f1b_materialization_contract") or {}
    )

    if materialization.get("read_expression_values") is not True:
        raise RuntimeError(
            "05f1a does not authorize 05f1b expression-value reading."
        )

    if materialization.get("read_TARGET_outcomes") is not False:
        raise RuntimeError(
            "05f1a unexpectedly authorizes TARGET outcome access."
        )

    if clean(
        materialization.get("matrix_dtype")
    ) != "float64":
        raise RuntimeError(
            "05f1a materialization dtype is not float64."
        )

    if clean(
        materialization.get("matrix_orientation")
    ) != "samples_x_aligned_genes":
        raise RuntimeError(
            "05f1a matrix orientation changed."
        )

    return {
        "contract": contract,
        "contract_sha256": observed_contract_hash,
        "summary_sha256": sha256_file(F1A_SUMMARY),
    }


def load_frozen_alignment() -> pd.DataFrame:
    frame = pd.read_csv(
        F1A_ALIGNMENT,
        sep="\t",
        dtype=str,
        low_memory=False,
    ).fillna("")

    required = [
        "aligned_feature_index",
        "human_gene_symbol",
        "target_expression_feature",
        "dog_gene_symbol",
        "dog2_raw_feature",
    ]

    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise RuntimeError(
            f"Frozen alignment lacks required column(s): {missing}"
        )

    frame["aligned_feature_index"] = pd.to_numeric(
        frame["aligned_feature_index"],
        errors="raise",
    ).astype(int)

    frame = frame.sort_values(
        "aligned_feature_index",
        kind="mergesort",
    ).reset_index(drop=True)

    expected_index = np.arange(
        EXPECTED_ALIGNED_GENES,
        dtype=int,
    )

    if len(frame) != EXPECTED_ALIGNED_GENES:
        raise RuntimeError(
            f"Frozen alignment has {len(frame)} rows; "
            f"expected {EXPECTED_ALIGNED_GENES}."
        )

    if not np.array_equal(
        frame["aligned_feature_index"].to_numpy(),
        expected_index,
    ):
        raise RuntimeError(
            "Frozen alignment feature indices are not exact 0..11814."
        )

    for column in [
        "human_gene_symbol",
        "target_expression_feature",
        "dog2_raw_feature",
    ]:
        values = [clean(x) for x in frame[column]]
        if any(not x for x in values):
            raise RuntimeError(
                f"Frozen alignment contains blank {column}."
            )
        if len(values) != len(set(values)):
            raise RuntimeError(
                f"Frozen alignment contains duplicate {column}."
            )

    return frame


def load_frozen_roster(
    path: Path,
    *,
    expected_n: int,
    expected_cohort: str,
) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        low_memory=False,
    ).fillna("")

    required = {
        "sample_index",
        "sample_id",
        "cohort",
        "source_expression_index_column",
    }
    missing = sorted(required - set(frame.columns))

    if missing:
        raise RuntimeError(
            f"{path.name} lacks required column(s): {missing}"
        )

    frame["sample_index"] = pd.to_numeric(
        frame["sample_index"],
        errors="raise",
    ).astype(int)

    frame = frame.sort_values(
        "sample_index",
        kind="mergesort",
    ).reset_index(drop=True)

    if len(frame) != expected_n:
        raise RuntimeError(
            f"{path.name} has {len(frame)} rows; expected {expected_n}."
        )

    if not np.array_equal(
        frame["sample_index"].to_numpy(),
        np.arange(expected_n, dtype=int),
    ):
        raise RuntimeError(
            f"{path.name}: sample_index is not exact 0..n-1."
        )

    ids = [clean(x) for x in frame["sample_id"]]

    if any(not x for x in ids):
        raise RuntimeError(
            f"{path.name}: blank sample ID."
        )

    if len(ids) != len(set(ids)):
        raise RuntimeError(
            f"{path.name}: duplicate sample ID."
        )

    if set(frame["cohort"].astype(str)) != {expected_cohort}:
        raise RuntimeError(
            f"{path.name}: cohort identity changed."
        )

    sample_columns = {
        clean(x)
        for x in frame["source_expression_index_column"]
    }
    if len(sample_columns) != 1 or "" in sample_columns:
        raise RuntimeError(
            f"{path.name}: source sample-ID column is not unique."
        )

    return frame


def materialize_one(
    *,
    cohort: str,
    expression_path: Path,
    roster: pd.DataFrame,
    sample_id_column: str,
    expression_feature_order: Sequence[str],
    human_gene_order: Sequence[str],
    output_path: Path,
) -> Dict[str, Any]:
    """
    Read only exact frozen sample-ID + aligned expression columns.
    No scientific transformation is performed.
    """
    requested_columns = [
        sample_id_column,
        *expression_feature_order,
    ]

    if len(requested_columns) != EXPECTED_ALIGNED_GENES + 1:
        raise RuntimeError(
            f"{cohort}: requested column count is not 11,816."
        )

    if len(requested_columns) != len(set(requested_columns)):
        raise RuntimeError(
            f"{cohort}: duplicate requested expression column name."
        )

    # Read exact frozen columns only. Keep string initially so that malformed
    # values fail explicitly under pd.to_numeric below.
    frame = pd.read_csv(
        expression_path,
        usecols=requested_columns,
        dtype=str,
        low_memory=False,
    )

    if sample_id_column not in frame.columns:
        raise RuntimeError(
            f"{cohort}: expression sample-ID column missing after read."
        )

    observed_ids = [
        clean(x)
        for x in frame[sample_id_column].fillna("")
    ]

    if any(not x for x in observed_ids):
        raise RuntimeError(
            f"{cohort}: blank sample ID in expression matrix."
        )

    if len(observed_ids) != len(set(observed_ids)):
        raise RuntimeError(
            f"{cohort}: duplicate expression sample IDs."
        )

    frozen_ids = roster["sample_id"].astype(str).tolist()

    if set(observed_ids) != set(frozen_ids):
        missing = sorted(set(frozen_ids) - set(observed_ids))
        unexpected = sorted(set(observed_ids) - set(frozen_ids))
        raise RuntimeError(
            f"{cohort}: expression roster differs from frozen 05f1a roster. "
            f"Missing={missing[:10]}, unexpected={unexpected[:10]}"
        )

    # Exact frozen sample order.
    frame = frame.set_index(sample_id_column).loc[frozen_ids]

    # `usecols` does not guarantee requested order; enforce frozen order.
    missing_features = [
        col
        for col in expression_feature_order
        if col not in frame.columns
    ]
    if missing_features:
        raise RuntimeError(
            f"{cohort}: missing aligned expression features after read: "
            f"{missing_features[:20]}"
        )

    frame = frame.loc[:, list(expression_feature_order)]

    # Strict numeric parse. No coercion/imputation.
    numeric_columns: List[np.ndarray] = []
    malformed_examples: List[str] = []

    for j, column in enumerate(expression_feature_order):
        parsed = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

        invalid = parsed.isna() & frame[column].notna()

        if invalid.any():
            rows = np.flatnonzero(invalid.to_numpy())[:5]
            for row_index in rows:
                malformed_examples.append(
                    f"{frozen_ids[row_index]}::{column}="
                    f"{frame.iloc[row_index, j]!r}"
                )
            break

        if parsed.isna().any():
            rows = np.flatnonzero(parsed.isna().to_numpy())[:5]
            for row_index in rows:
                malformed_examples.append(
                    f"{frozen_ids[row_index]}::{column}=<missing>"
                )
            break

        numeric_columns.append(
            parsed.to_numpy(dtype=MATRIX_DTYPE, copy=False)
        )

    if malformed_examples:
        raise RuntimeError(
            f"{cohort}: nonnumeric/missing aligned expression value(s): "
            + "; ".join(malformed_examples)
        )

    X = np.column_stack(
        numeric_columns
    ).astype(MATRIX_DTYPE, copy=False)

    expected_shape = (
        len(frozen_ids),
        EXPECTED_ALIGNED_GENES,
    )

    if X.shape != expected_shape:
        raise RuntimeError(
            f"{cohort}: matrix shape={X.shape}, expected={expected_shape}."
        )

    finite_mask = np.isfinite(X)

    if not finite_mask.all():
        first = np.argwhere(~finite_mask)[0]
        row, col = int(first[0]), int(first[1])
        raise RuntimeError(
            f"{cohort}: nonfinite value at "
            f"sample={frozen_ids[row]!r}, "
            f"gene={human_gene_order[col]!r}, "
            f"value={X[row, col]!r}. FAIL_CLOSED."
        )

    # Serialization uses only non-object arrays so allow_pickle=False works.
    sample_width = max(
        1,
        max(len(x) for x in frozen_ids),
    )
    gene_width = max(
        1,
        max(len(x) for x in human_gene_order),
    )
    expr_width = max(
        1,
        max(len(x) for x in expression_feature_order),
    )

    sample_array = np.asarray(
        frozen_ids,
        dtype=f"<U{sample_width}",
    )
    gene_array = np.asarray(
        human_gene_order,
        dtype=f"<U{gene_width}",
    )
    expr_feature_array = np.asarray(
        expression_feature_order,
        dtype=f"<U{expr_width}",
    )
    aligned_index = np.arange(
        EXPECTED_ALIGNED_GENES,
        dtype=np.int32,
    )

    tmp = output_path.with_suffix(
        output_path.suffix + ".part"
    )

    with tmp.open("wb") as handle:
        np.savez_compressed(
            handle,
            X=X,
            sample_id=sample_array,
            human_gene_symbol=gene_array,
            source_expression_feature=expr_feature_array,
            aligned_feature_index=aligned_index,
        )

    tmp.replace(output_path)

    # Immediate exact replay from disk.
    with np.load(
        output_path,
        allow_pickle=False,
    ) as saved:
        saved_X = saved["X"]
        saved_samples = saved["sample_id"].astype(str)
        saved_genes = saved["human_gene_symbol"].astype(str)
        saved_features = saved[
            "source_expression_feature"
        ].astype(str)
        saved_index = saved["aligned_feature_index"]

        if saved_X.dtype != np.float64:
            raise RuntimeError(
                f"{cohort}: saved matrix dtype is {saved_X.dtype}, not float64."
            )

        if saved_X.shape != expected_shape:
            raise RuntimeError(
                f"{cohort}: saved matrix shape changed."
            )

        if not np.array_equal(
            saved_samples,
            sample_array.astype(str),
        ):
            raise RuntimeError(
                f"{cohort}: saved sample order differs."
            )

        if not np.array_equal(
            saved_genes,
            gene_array.astype(str),
        ):
            raise RuntimeError(
                f"{cohort}: saved gene order differs."
            )

        if not np.array_equal(
            saved_features,
            expr_feature_array.astype(str),
        ):
            raise RuntimeError(
                f"{cohort}: saved source feature order differs."
            )

        if not np.array_equal(
            saved_index,
            aligned_index,
        ):
            raise RuntimeError(
                f"{cohort}: saved aligned feature indices differ."
            )

        if not np.array_equal(
            saved_X,
            X,
        ):
            raise RuntimeError(
                f"{cohort}: saved numeric matrix does not replay exactly."
            )

    # Value audit records integrity only; it does NOT compute/select variance,
    # model relevance, outcome associations, or filters.
    return {
        "cohort": cohort,
        "n_samples": int(X.shape[0]),
        "n_aligned_genes": int(X.shape[1]),
        "dtype": str(X.dtype),
        "all_finite": True,
        "missing_values": 0,
        "imputed_values": 0,
        "samples_dropped": 0,
        "genes_dropped": 0,
        "matrix_memory_bytes": int(X.nbytes),
        "raw_numeric_array_sha256": sha256_ndarray(X),
        "sample_order_sha256": sha256_text_lines(frozen_ids),
        "human_gene_order_sha256": sha256_text_lines(
            list(human_gene_order)
        ),
        "source_feature_order_sha256": sha256_text_lines(
            list(expression_feature_order)
        ),
        "output_path": str(
            output_path.relative_to(ROOT)
        ),
        "output_file_sha256": sha256_file(output_path),
        "output_size_bytes": int(
            output_path.stat().st_size
        ),
    }


def main() -> None:
    started = now_utc()

    print("=" * 120)
    print("Paper 6 - materialize exact raw aligned DOG2/TARGET expression matrices")
    print("=" * 120)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Materialization version: {MATERIALIZATION_VERSION}")
    print()
    print("Safety / execution contract:")
    print("  05f1a frozen contract verified: YES")
    print("  DOG2 expression values read: YES [exact aligned genes only]")
    print("  TARGET expression values read: YES [exact aligned genes only]")
    print("  TARGET clinical file read: NO")
    print("  TARGET outcomes/events read: NO")
    print("  GSE21257/GSE39055 outcomes read: NO")
    print("  Outcome-based sample intersection: NO")
    print("  Imputation: NO")
    print("  Variance filtering: NO")
    print("  Gene scaling: NO")
    print("  Hallmark/module scoring: NO")
    print("  Whole-cohort TARGET scaling: NO")
    print("  PCA/CORAL: NO")
    print("  Sample deletion: NO")
    print("  Model fitting: NO")
    print("  Survival split generation: NO")
    print("  GPU execution: NO")
    print()

    f1a_state = verify_f1a()
    contract = f1a_state["contract"]

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

    locked_assets = contract.get(
        "locked_expression_assets"
    ) or {}

    for cohort_name, asset, frozen_key in [
        ("DOG2", dog_asset, "DOG2"),
        ("TARGET_OS", target_asset, "TARGET_OS"),
    ]:
        frozen = locked_assets.get(frozen_key) or {}

        if clean(asset.get("relative_path")) != clean(
            frozen.get("relative_path")
        ):
            raise RuntimeError(
                f"{cohort_name}: 05f1a relative expression path differs "
                "from upstream lock."
            )

        if clean(asset.get("sha256")).lower() != clean(
            frozen.get("sha256")
        ).lower():
            raise RuntimeError(
                f"{cohort_name}: 05f1a/upstream expression SHA mismatch."
            )

    alignment = load_frozen_alignment()

    dog_roster = load_frozen_roster(
        F1A_DOG_ROSTER,
        expected_n=EXPECTED_DOG2_SAMPLES,
        expected_cohort="DOG2",
    )
    target_roster = load_frozen_roster(
        F1A_TARGET_ROSTER,
        expected_n=EXPECTED_TARGET_SAMPLES,
        expected_cohort="TARGET_OS",
    )

    dog_sample_col = clean(
        dog_roster["source_expression_index_column"].iloc[0]
    )
    target_sample_col = clean(
        target_roster["source_expression_index_column"].iloc[0]
    )

    human_gene_order = (
        alignment["human_gene_symbol"]
        .astype(str)
        .tolist()
    )
    dog_feature_order = (
        alignment["dog2_raw_feature"]
        .astype(str)
        .tolist()
    )
    target_feature_order = (
        alignment["target_expression_feature"]
        .astype(str)
        .tolist()
    )

    # Independent order-hash check against 05f1a contract.
    aligned_contract = (
        contract.get("aligned_gene_representation") or {}
    )

    if sha256_text_lines(
        human_gene_order
    ) != clean(
        aligned_contract.get("human_gene_order_sha256")
    ):
        raise RuntimeError(
            "05f1a frozen human gene-order SHA mismatch."
        )

    if sha256_text_lines(
        dog_feature_order
    ) != clean(
        aligned_contract.get("dog2_raw_feature_order_sha256")
    ):
        raise RuntimeError(
            "05f1a frozen DOG2 feature-order SHA mismatch."
        )

    print("05f1a contract/artifact verification: PASS")
    print("Locked upstream expression SHA256 verification: PASS")
    print(f"Paper4 root resolution: {paper4_resolution}")
    print()
    print(
        f"Frozen aligned representation: "
        f"{EXPECTED_ALIGNED_GENES:,} genes"
    )
    print(
        f"DOG2 roster: {len(dog_roster):,} samples"
    )
    print(
        f"TARGET roster: {len(target_roster):,} samples"
    )
    print()

    print("Materializing DOG2 raw aligned matrix...")
    dog_result = materialize_one(
        cohort="DOG2",
        expression_path=dog_path,
        roster=dog_roster,
        sample_id_column=dog_sample_col,
        expression_feature_order=dog_feature_order,
        human_gene_order=human_gene_order,
        output_path=DOG2_MATRIX,
    )
    print(
        f"  PASS: {dog_result['n_samples']} x "
        f"{dog_result['n_aligned_genes']:,}, "
        f"float64, all finite"
    )

    print("Materializing TARGET-OS raw aligned matrix...")
    target_result = materialize_one(
        cohort="TARGET_OS",
        expression_path=target_path,
        roster=target_roster,
        sample_id_column=target_sample_col,
        expression_feature_order=target_feature_order,
        human_gene_order=human_gene_order,
        output_path=TARGET_MATRIX,
    )
    print(
        f"  PASS: {target_result['n_samples']} x "
        f"{target_result['n_aligned_genes']:,}, "
        f"float64, all finite"
    )

    # Cross-cohort feature identity is exact by construction and is replayed
    # from the serialized artifacts here.
    with np.load(
        DOG2_MATRIX,
        allow_pickle=False,
    ) as dog_saved, np.load(
        TARGET_MATRIX,
        allow_pickle=False,
    ) as target_saved:
        dog_genes = dog_saved[
            "human_gene_symbol"
        ].astype(str)
        target_genes = target_saved[
            "human_gene_symbol"
        ].astype(str)

        if not np.array_equal(
            dog_genes,
            target_genes,
        ):
            raise RuntimeError(
                "Serialized DOG2/TARGET human-gene order differs."
            )

        if not np.array_equal(
            dog_saved["aligned_feature_index"],
            target_saved["aligned_feature_index"],
        ):
            raise RuntimeError(
                "Serialized DOG2/TARGET aligned indices differ."
            )

    integrity_df = pd.DataFrame(
        [
            {
                key: value
                for key, value in result.items()
                if key not in {
                    "output_path",
                    "output_file_sha256",
                    "output_size_bytes",
                }
            }
            for result in [
                dog_result,
                target_result,
            ]
        ]
    )

    integrity_df.to_csv(
        VALUE_AUDIT,
        sep="\t",
        index=False,
    )

    manifest_df = pd.DataFrame(
        [
            {
                "cohort": result["cohort"],
                "matrix_path": result["output_path"],
                "matrix_file_sha256": result[
                    "output_file_sha256"
                ],
                "raw_numeric_array_sha256": result[
                    "raw_numeric_array_sha256"
                ],
                "n_samples": result["n_samples"],
                "n_aligned_genes": result[
                    "n_aligned_genes"
                ],
                "dtype": result["dtype"],
                "output_size_bytes": result[
                    "output_size_bytes"
                ],
                "sample_order_sha256": result[
                    "sample_order_sha256"
                ],
                "human_gene_order_sha256": result[
                    "human_gene_order_sha256"
                ],
                "source_feature_order_sha256": result[
                    "source_feature_order_sha256"
                ],
            }
            for result in [
                dog_result,
                target_result,
            ]
        ]
    )

    manifest_df.to_csv(
        MATRIX_MANIFEST,
        sep="\t",
        index=False,
    )

    summary = {
        "script_version": SCRIPT_VERSION,
        "materialization_version": MATERIALIZATION_VERSION,
        "status": "PASS",
        "scientific_status": (
            "PASS_RAW_ALIGNED_DOG2_TARGET_EXPRESSION_MATERIALIZED_OUTCOME_FREE"
        ),
        "run_started_utc": started,
        "run_finished_utc": now_utc(),

        "05f1a_contract_sha256": (
            f1a_state["contract_sha256"]
        ),
        "05f1a_summary_sha256": (
            f1a_state["summary_sha256"]
        ),

        "paper4_root_resolution_source": paper4_resolution,
        "absolute_paper4_path_recorded": False,

        "aligned_genes": EXPECTED_ALIGNED_GENES,
        "DOG2_samples": EXPECTED_DOG2_SAMPLES,
        "TARGET_expression_samples": EXPECTED_TARGET_SAMPLES,

        "matrix_manifest_sha256": sha256_file(
            MATRIX_MANIFEST
        ),
        "value_integrity_audit_sha256": sha256_file(
            VALUE_AUDIT
        ),
        "DOG2_matrix_file_sha256": sha256_file(
            DOG2_MATRIX
        ),
        "TARGET_matrix_file_sha256": sha256_file(
            TARGET_MATRIX
        ),
        "DOG2_raw_numeric_array_sha256": dog_result[
            "raw_numeric_array_sha256"
        ],
        "TARGET_raw_numeric_array_sha256": target_result[
            "raw_numeric_array_sha256"
        ],

        "cross_cohort_human_gene_order_identical": True,
        "all_values_float64": True,
        "all_values_finite": True,
        "imputation_performed": False,
        "variance_filtering_performed": False,
        "gene_scaling_performed": False,
        "Hallmark_scoring_performed": False,
        "whole_cohort_TARGET_scaling_performed": False,
        "samples_dropped": 0,
        "genes_dropped": 0,

        "TARGET_clinical_file_read": False,
        "TARGET_outcomes_read": False,
        "GSE21257_outcomes_read": False,
        "GSE39055_outcomes_read": False,
        "model_fitting": False,
        "survival_splits_generated": False,
        "GPU_execution": False,

        "next": (
            "05f1c freeze the exact TARGET clinical-schema/header-only endpoint "
            "mapping, repeated outer-CV split/aggregation/bootstrap implementation, "
            "and outcome-opening execution contract. Do not read TARGET outcome "
            "values in 05f1c."
        ),
    }

    write_json(
        SUMMARY_JSON,
        summary,
    )

    print()
    print("=" * 120)
    print("05f1b RAW ALIGNED MATRIX MATERIALIZATION SUMMARY")
    print("=" * 120)
    print(
        f"DOG2 matrix: "
        f"{EXPECTED_DOG2_SAMPLES} x {EXPECTED_ALIGNED_GENES:,}"
    )
    print(
        f"TARGET matrix: "
        f"{EXPECTED_TARGET_SAMPLES} x {EXPECTED_ALIGNED_GENES:,}"
    )
    print("Matrix dtype: float64")
    print("Cross-cohort human-gene order identical: PASS")
    print("All aligned values finite: PASS")
    print("Samples dropped: 0")
    print("Genes dropped: 0")
    print()
    print("Scientific transformations:")
    print("  imputation: NO")
    print("  variance filtering: NO")
    print("  gene scaling: NO")
    print("  Hallmark scoring: NO")
    print("  whole-cohort TARGET scaling: NO")
    print("  PCA/CORAL: NO")
    print()
    print("Outcome firewall:")
    print("  TARGET clinical file read: NO")
    print("  TARGET outcomes/events read: NO")
    print("  GSE21257/GSE39055 outcomes read: NO")
    print("  survival splits generated: NO")
    print("  model fitting: NO")
    print("  GPU execution: NO")
    print()
    print(
        f"Matrix manifest SHA256: "
        f"{sha256_file(MATRIX_MANIFEST)}"
    )
    print(
        f"DOG2 matrix SHA256: "
        f"{sha256_file(DOG2_MATRIX)}"
    )
    print(
        f"TARGET matrix SHA256: "
        f"{sha256_file(TARGET_MATRIX)}"
    )
    print()
    print("Next:")
    print(
        "  05f1c freeze TARGET clinical schema/header mapping and exact "
        "evaluation/CI mechanics BEFORE outcome values are opened."
    )
    print("=" * 120)
    print(
        "05f1b: PASS_RAW_ALIGNED_DOG2_TARGET_EXPRESSION_MATERIALIZED_OUTCOME_FREE"
    )
    print("=" * 120)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 120, file=sys.stderr)
        print(
            "05f1b raw aligned expression materialization: FAIL",
            file=sys.stderr,
        )
        print(
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        print("=" * 120, file=sys.stderr)
        raise
