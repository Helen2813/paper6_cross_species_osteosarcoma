from __future__ import annotations

from pathlib import Path
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any

import pandas as pd


SCRIPT_VERSION = "00-lock-upstream-inputs-v1-no-cli"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "_config"
MANIFEST_DIR = PROJECT_ROOT / "manifests"
CONTRACT_DIR = PROJECT_ROOT / "contracts"

LOCAL_PATH_CONFIG = CONFIG_DIR / "paths.local.json"

OUTPUT_MANIFEST = MANIFEST_DIR / "00_upstream_input_manifest.csv"
OUTPUT_LOCK = CONTRACT_DIR / "00_upstream_input_lock.json"
OUTPUT_README = MANIFEST_DIR / "00_upstream_input_lock_README.txt"

EXPECTED_PAPER4_BASENAME = "paper4_sarcoma_dog"


# Paper 6 does not copy these files.  It records their immutable identity
# (relative path, SHA-256, size and lightweight table dimensions) and then
# downstream scripts read them in-place from Paper 4.
ASSETS: list[dict[str, Any]] = [
    {
        "asset_id": "dog2_expression",
        "role": "required_source_expression",
        "relative_path": "data/processed/GSE238110_DOG2_expression_log2cpm_matched_allgenes.csv",
        "kind": "matrix_csv",
        "expected_rows": 186,
        "expected_features": 21016,
    },
    {
        "asset_id": "dog2_clinical",
        "role": "required_source_clinical",
        "relative_path": "data/processed/GSE238110_DOG2_clinical_matched_indexed.csv",
        "kind": "indexed_csv",
        "expected_rows": 186,
    },
    {
        "asset_id": "target_expression",
        "role": "required_primary_human_expression",
        "relative_path": "data/processed/human_validation/TARGET_OS_expression_log2_gene_symbol.csv",
        "kind": "matrix_csv",
        "expected_rows": 88,
        "expected_features": 56621,
    },
    {
        "asset_id": "target_clinical",
        "role": "required_primary_human_clinical",
        "relative_path": "data/processed/human_validation/TARGET_OS_clinical_standardized.csv",
        "kind": "indexed_csv",
        "expected_rows": 88,
    },
    {
        "asset_id": "gse21257_expression",
        "role": "required_human_external_expression",
        "relative_path": "data/processed/human_validation/GSE21257_expression_gene_symbol.csv",
        "kind": "matrix_csv",
        "expected_rows": 53,
        "expected_features": 24996,
    },
    {
        "asset_id": "gse21257_clinical",
        "role": "required_human_external_clinical",
        "relative_path": "data/processed/human_validation/GSE21257_clinical_standardized.csv",
        "kind": "indexed_csv",
        "expected_rows": 53,
    },
    {
        "asset_id": "gse39055_expression",
        "role": "required_human_stress_expression",
        "relative_path": "data/processed/human_validation/GSE39055_expression_gene_symbol.csv",
        "kind": "matrix_csv",
        "expected_rows": 37,
        "expected_features": 20793,
    },
    {
        "asset_id": "gse39055_clinical",
        "role": "required_human_stress_clinical",
        "relative_path": "data/processed/human_validation/GSE39055_clinical_standardized.csv",
        "kind": "indexed_csv",
        "expected_rows": 37,
    },
    {
        "asset_id": "gse239948_expression",
        "role": "required_external_canine_expression",
        "relative_path": "data/processed/canine_validation_GSE239948_expression_log2_symbol.csv",
        "kind": "matrix_csv",
        "expected_rows": 43,
        "expected_features": 15307,
    },
    {
        "asset_id": "ortholog_qc",
        "role": "required_cross_species_mapping_reference",
        "relative_path": "results/tables/GSE238110_RNA_master_candidate_evidence_table_with_ortholog_qc.csv",
        "kind": "csv",
    },
    {
        "asset_id": "frozen_strict_weights",
        "role": "paper4_frozen_reference_only",
        "relative_path": "results/tables/GSE238110_frozen_transfer_gene_weights_strict.csv",
        "kind": "csv",
    },
    {
        "asset_id": "frozen_program_manifest",
        "role": "paper4_frozen_reference_only",
        "relative_path": "results/tables/GSE238110_frozen_canine_transfer_program_manifest.csv",
        "kind": "csv",
    },
    {
        "asset_id": "frozen_scoring_spec",
        "role": "paper4_frozen_reference_only",
        "relative_path": "results/tables/GSE238110_frozen_transfer_scoring_specification.csv",
        "kind": "csv",
    },
    {
        "asset_id": "paper4_freeze_json",
        "role": "required_upstream_provenance",
        "relative_path": "results/tables/GSE238110_frozen_transfer_program_freeze.json",
        "kind": "json",
    },
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def canonical_relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def load_local_path_config() -> dict[str, Any]:
    if not LOCAL_PATH_CONFIG.exists():
        return {}

    try:
        payload = json.loads(LOCAL_PATH_CONFIG.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"Could not parse local path config: {LOCAL_PATH_CONFIG}"
        ) from exc

    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Local path config must contain a JSON object: {LOCAL_PATH_CONFIG}"
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
            (
                PROJECT_ROOT.parent / EXPECTED_PAPER4_BASENAME,
                "sibling_repository",
            ),
            (
                Path.home() / "Desktop" / EXPECTED_PAPER4_BASENAME,
                "home_desktop_fallback",
            ),
        ]
    )

    checked: list[str] = []
    for candidate, source in candidates:
        resolved = candidate.resolve()
        checked.append(f"{source}: {resolved}")
        if not resolved.exists() or not resolved.is_dir():
            continue

        sentinel = (
            resolved
            / "data"
            / "processed"
            / "GSE238110_DOG2_expression_log2cpm_matched_allgenes.csv"
        )
        if sentinel.exists():
            return resolved, source

    message = "\n".join(f"  - {item}" for item in checked)
    raise FileNotFoundError(
        "Could not locate the Paper 4 repository.\n"
        "Checked:\n"
        f"{message}\n\n"
        "Either place paper4_sarcoma_dog next to this repository, set the "
        "PAPER4_ROOT environment variable, or create "
        "_config/paths.local.json with:\n"
        '{\n  "paper4_root": "C:\\\\path\\\\to\\\\paper4_sarcoma_dog"\n}\n'
    )


def inspect_csv(path: Path, indexed: bool) -> tuple[int, int]:
    # Chunked reading keeps memory bounded even for very wide expression files.
    rows = 0
    columns: int | None = None

    try:
        iterator = pd.read_csv(
            path,
            index_col=0 if indexed else None,
            chunksize=16,
            low_memory=False,
        )
        for chunk in iterator:
            rows += int(chunk.shape[0])
            if columns is None:
                columns = int(chunk.shape[1])
            elif int(chunk.shape[1]) != columns:
                raise RuntimeError(
                    f"Inconsistent column count while reading {path}"
                )
    except pd.errors.EmptyDataError as exc:
        raise RuntimeError(f"CSV is empty: {path}") from exc

    return rows, int(columns or 0)


def inspect_asset(path: Path, kind: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
        "n_rows": None,
        "n_columns": None,
    }

    if kind == "matrix_csv":
        rows, cols = inspect_csv(path, indexed=True)
        result["n_rows"] = rows
        result["n_columns"] = cols
    elif kind == "indexed_csv":
        rows, cols = inspect_csv(path, indexed=True)
        result["n_rows"] = rows
        result["n_columns"] = cols
    elif kind == "csv":
        rows, cols = inspect_csv(path, indexed=False)
        result["n_rows"] = rows
        result["n_columns"] = cols
    elif kind == "json":
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Invalid JSON input: {path}") from exc
    else:
        raise ValueError(f"Unsupported asset kind: {kind}")

    return result


def verify_expected_shape(
    asset: dict[str, Any],
    observed: dict[str, Any],
) -> list[str]:
    problems: list[str] = []

    expected_rows = asset.get("expected_rows")
    expected_features = asset.get("expected_features")

    if expected_rows is not None and observed.get("n_rows") != expected_rows:
        problems.append(
            f"rows expected={expected_rows} observed={observed.get('n_rows')}"
        )

    if (
        expected_features is not None
        and observed.get("n_columns") != expected_features
    ):
        problems.append(
            "features expected="
            f"{expected_features} observed={observed.get('n_columns')}"
        )

    return problems


def verify_paper4_internal_freeze(
    paper4_root: Path,
    manifest_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    freeze_path = (
        paper4_root
        / "results"
        / "tables"
        / "GSE238110_frozen_transfer_program_freeze.json"
    )
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    recorded = freeze.get("files", {})

    observed_by_name = {
        Path(str(row["relative_path"])).name: row
        for row in manifest_rows
    }

    checks: list[dict[str, Any]] = []
    for filename, metadata in recorded.items():
        expected_hash = metadata.get("sha256")
        row = observed_by_name.get(filename)

        if row is None:
            checks.append(
                {
                    "filename": filename,
                    "status": "not_in_paper6_lock_scope",
                    "expected_sha256": expected_hash,
                    "observed_sha256": None,
                }
            )
            continue

        observed_hash = row["sha256"]
        status = (
            "verified"
            if expected_hash and observed_hash == expected_hash
            else "no_recorded_hash"
            if not expected_hash
            else "HASH_MISMATCH"
        )
        checks.append(
            {
                "filename": filename,
                "status": status,
                "expected_sha256": expected_hash,
                "observed_sha256": observed_hash,
            }
        )

        if status == "HASH_MISMATCH":
            raise RuntimeError(
                "Paper 4 frozen-file integrity failure: "
                f"{filename} does not match the SHA-256 recorded in "
                f"{freeze_path.name}. Do not continue."
            )

    return {
        "freeze_file_relative_path": canonical_relative(
            freeze_path, paper4_root
        ),
        "freeze_file_sha256": sha256_file(freeze_path),
        "frozen_after_script": freeze.get("frozen_after_script"),
        "primary_canine_endpoint": freeze.get("primary_canine_endpoint"),
        "checks": checks,
    }


def write_readme(
    paper4_resolution_source: str,
    n_assets: int,
) -> None:
    text = f"""Paper 6 upstream input lock
Script version: {SCRIPT_VERSION}

Purpose
-------
Paper 6 reads immutable upstream data directly from Paper 4 rather than copying
large scientific files into the Paper 6 repository.

This lock records:
- Paper-4-relative file path
- SHA-256
- file size
- lightweight row/column dimensions for tabular inputs
- verification against the Paper 4 frozen-program manifest where applicable

The absolute Paper 4 path is intentionally NOT written into the committed
manifest or lock JSON. This keeps machine-specific paths out of Git.

Paper 4 root resolution
-----------------------
Resolved via: {paper4_resolution_source}

Locked assets
-------------
{n_assets}

Scientific guardrails
---------------------
1. This script performs no model fitting.
2. This script performs no feature selection.
3. This script performs no outcome association testing.
4. This script copies no scientific datasets.
5. Downstream Paper 6 scripts should verify these hashes before analysis.
6. Paper 4 frozen assets are reference/provenance inputs; Paper 6 must not
   silently modify their scientific definitions.
"""
    OUTPUT_README.write_text(text, encoding="utf-8")


def main() -> None:
    print("=" * 88)
    print("Paper 6 - lock upstream Paper 4 inputs without copying data")
    print("=" * 88)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Paper 6 project root: {PROJECT_ROOT}")
    print("")
    print("Design:")
    print("  Resolve the Paper 4 repository locally.")
    print("  Hash required upstream files in place.")
    print("  Record table dimensions and Paper-4-relative paths.")
    print("  Verify frozen Paper 4 artifacts where recorded hashes exist.")
    print("  Copy no scientific data.")
    print("  Fit no model and test no outcome association.")
    print("")

    for directory in [CONFIG_DIR, MANIFEST_DIR, CONTRACT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    paper4_root, resolution_source = resolve_paper4_root()

    print(f"Resolved Paper 4 root: {paper4_root}")
    print(f"Resolution source: {resolution_source}")
    print("")

    rows: list[dict[str, Any]] = []
    failures: list[str] = []

    for index, asset in enumerate(ASSETS, start=1):
        path = paper4_root / str(asset["relative_path"])
        print(
            f"[{index:02d}/{len(ASSETS):02d}] "
            f"{asset['asset_id']}: {asset['relative_path']}"
        )

        if not path.exists():
            failures.append(
                f"{asset['asset_id']}: missing {asset['relative_path']}"
            )
            print("  MISSING")
            continue

        observed = inspect_asset(path, str(asset["kind"]))
        shape_problems = verify_expected_shape(asset, observed)

        if shape_problems:
            failures.append(
                f"{asset['asset_id']}: " + "; ".join(shape_problems)
            )
            shape_status = "FAIL"
        else:
            shape_status = "PASS"

        row = {
            "asset_id": asset["asset_id"],
            "role": asset["role"],
            "relative_path": canonical_relative(path, paper4_root),
            "kind": asset["kind"],
            "size_bytes": observed["size_bytes"],
            "sha256": observed["sha256"],
            "n_rows": observed["n_rows"],
            "n_columns": observed["n_columns"],
            "expected_rows": asset.get("expected_rows"),
            "expected_features": asset.get("expected_features"),
            "shape_check": shape_status,
        }
        rows.append(row)

        dimension_text = ""
        if observed["n_rows"] is not None:
            dimension_text = (
                f", rows={observed['n_rows']}, "
                f"columns={observed['n_columns']}"
            )
        print(
            f"  sha256={observed['sha256'][:16]}..., "
            f"bytes={observed['size_bytes']}{dimension_text}, "
            f"shape_check={shape_status}"
        )

    if failures:
        print("")
        print("=" * 88)
        print("UPSTREAM LOCK FAILED")
        print("=" * 88)
        for failure in failures:
            print(f"  - {failure}")
        raise RuntimeError(
            "Required Paper 4 inputs are missing or differ from the expected "
            "dimensions. No Paper 6 input lock was written."
        )

    internal_freeze = verify_paper4_internal_freeze(
        paper4_root=paper4_root,
        manifest_rows=rows,
    )

    manifest = pd.DataFrame(rows).sort_values("asset_id").reset_index(drop=True)
    manifest.to_csv(OUTPUT_MANIFEST, index=False)

    lock_payload = {
        "script_version": SCRIPT_VERSION,
        "created_utc": utc_now_iso(),
        "paper6_project": PROJECT_ROOT.name,
        "upstream_project": EXPECTED_PAPER4_BASENAME,
        "upstream_root_resolution_source": resolution_source,
        "absolute_upstream_path_recorded": False,
        "copy_policy": "read_in_place_no_scientific_data_copy",
        "n_locked_assets": len(rows),
        "manifest_file": OUTPUT_MANIFEST.name,
        "manifest_sha256": sha256_file(OUTPUT_MANIFEST),
        "paper4_internal_freeze_verification": internal_freeze,
        "assets": {
            row["asset_id"]: {
                "role": row["role"],
                "relative_path": row["relative_path"],
                "sha256": row["sha256"],
                "size_bytes": int(row["size_bytes"]),
                "n_rows": (
                    None
                    if pd.isna(row["n_rows"])
                    else int(row["n_rows"])
                ),
                "n_columns": (
                    None
                    if pd.isna(row["n_columns"])
                    else int(row["n_columns"])
                ),
            }
            for row in rows
        },
    }

    OUTPUT_LOCK.write_text(
        json.dumps(lock_payload, indent=2),
        encoding="utf-8",
    )

    write_readme(
        paper4_resolution_source=resolution_source,
        n_assets=len(rows),
    )

    print("")
    print("=" * 88)
    print("Paper 6 upstream lock: PASS")
    print("=" * 88)
    print(f"Locked assets: {len(rows)}")
    print("Scientific data copied into Paper 6: NO")
    print("Model fitting: NO")
    print("Outcome association testing: NO")
    print("")
    print("Saved:")
    print(f"  {OUTPUT_MANIFEST}")
    print(f"  {OUTPUT_LOCK}")
    print(f"  {OUTPUT_README}")
    print("")
    print("Next:")
    print("  Run the external-cohort inventory / premise-audit stage only after")
    print("  this upstream lock is committed.")
    print("Done.")


if __name__ == "__main__":
    main()
