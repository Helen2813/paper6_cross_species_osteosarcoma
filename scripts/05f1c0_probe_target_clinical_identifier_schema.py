#!/usr/bin/env python3
"""
Paper 6 - pre-outcome probe of TARGET clinical identifier schema.

PURPOSE
-------
Diagnose the 05f1c identifier-mapping failure without reading ANY survival or
other clinical outcome values.

Reads:
- frozen TARGET expression sample roster from 05f1a;
- locked TARGET clinical HEADER;
- values only from columns whose NAMES look identifier-like and do NOT look
  outcome/time/status-like.

Does NOT read:
- OS/event/status values;
- survival/death/follow-up/time values;
- any model result;
- any secondary endpoint value.

No scientific artifact is modified or frozen.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd


SCRIPT_VERSION = "05f1c0-probe-target-clinical-identifier-schema-v1-no-cli"

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "_config"
PAPER4_BASENAME = "paper4_sarcoma_dog"

UPSTREAM_LOCK = ROOT / "contracts" / "00_upstream_input_lock.json"

TARGET_ROSTER = (
    ROOT
    / "results"
    / "human_posthold_descriptive"
    / "05f1a"
    / "TARGET_expression_sample_roster.tsv"
)

OUTCOMEISH_TOKENS = {
    "os",
    "survival",
    "death",
    "dead",
    "event",
    "status",
    "followup",
    "follow",
    "time",
    "days",
    "months",
    "efs",
    "dfi",
    "dfs",
    "pfs",
    "vital",
}

IDISH_TOKENS = {
    "id",
    "identifier",
    "barcode",
    "sample",
    "patient",
    "participant",
    "case",
    "submitter",
    "target",
    "usi",
    "subject",
}


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "na", "n/a"}:
        return ""
    return text


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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

    for candidate, source in candidates:
        resolved = candidate.resolve()
        if resolved.is_dir():
            return resolved, source

    raise FileNotFoundError("Could not resolve Paper 4 root.")


def preferred_clinical_asset_role(lock: Dict[str, Any]) -> str:
    assets = lock.get("assets") or {}

    preferred = [
        "target_clinical",
        "target_os_clinical",
        "target_clinical_matched",
        "target_os_clinical_matched",
        "target_survival",
    ]

    for role in preferred:
        if role in assets:
            return role

    candidates = [
        str(role)
        for role in assets
        if (
            "target" in str(role).lower()
            and (
                "clinical" in str(role).lower()
                or "survival" in str(role).lower()
            )
            and "expression" not in str(role).lower()
        )
    ]

    if len(candidates) == 1:
        return candidates[0]

    raise RuntimeError(
        "Could not uniquely identify TARGET clinical asset. "
        f"Candidates={candidates}; available={sorted(assets)}"
    )


def read_header(path: Path) -> List[str]:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path, nrows=0)
    elif suffix in {".tsv", ".txt"}:
        df = pd.read_csv(path, sep="\t", nrows=0)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path, nrows=0)
    else:
        raise RuntimeError(f"Unsupported clinical file type: {suffix}")

    return [str(c) for c in df.columns]


def read_columns(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(
            path,
            usecols=list(columns),
            dtype=str,
            low_memory=False,
        ).fillna("")

    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(
            path,
            sep="\t",
            usecols=list(columns),
            dtype=str,
            low_memory=False,
        ).fillna("")

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(
            path,
            usecols=list(columns),
            dtype=str,
        ).fillna("")

    raise RuntimeError(f"Unsupported clinical file type: {suffix}")


def tokenize_header(name: str) -> List[str]:
    # Split camel-ish / punctuation names into alphanumeric tokens as well as
    # keeping compact normalized form.
    lower = str(name).lower()
    tokens = re.findall(r"[a-z0-9]+", lower)
    return tokens


def is_safe_identifier_candidate(name: str) -> bool:
    lower = str(name).lower()
    tokens = set(tokenize_header(name))
    compact = re.sub(r"[^a-z0-9]+", "", lower)

    # Fail closed for anything outcome/time/status-like.
    if any(tok in tokens for tok in OUTCOMEISH_TOKENS):
        return False

    if any(tok in compact for tok in [
        "survival",
        "death",
        "event",
        "status",
        "followup",
        "vital",
        "ostime",
        "osdays",
        "osmonths",
    ]):
        return False

    # Positive identifier-like signal.
    if any(tok in tokens for tok in IDISH_TOKENS):
        return True

    if any(part in compact for part in [
        "sampleid",
        "patientid",
        "caseid",
        "participantid",
        "submitterid",
        "targetusi",
        "barcode",
        "identifier",
    ]):
        return True

    return False


def normalize_exact(value: str) -> str:
    return clean(value).upper()


def first_n_hyphen_tokens(value: str, n: int) -> str:
    text = normalize_exact(value)
    parts = text.split("-")
    if len(parts) >= n:
        return "-".join(parts[:n])
    return text


def strip_common_sample_suffix(value: str) -> str:
    """
    Conservative TARGET/TCGA-style suffix stripping for diagnostics only.
    This probe does NOT freeze or authorize the transform.
    """
    text = normalize_exact(value)

    # Remove common aliquot/sample suffix after an already case-like barcode:
    # TARGET-XX-YYYY-...
    parts = text.split("-")
    if len(parts) >= 3 and parts[0] == "TARGET":
        return "-".join(parts[:3])

    return text


def overlap_stats(
    expr_ids: Sequence[str],
    clin_ids: Sequence[str],
    transform_name: str,
) -> Dict[str, Any]:
    if transform_name == "EXACT_UPPER":
        ex = [normalize_exact(x) for x in expr_ids]
        cl = [normalize_exact(x) for x in clin_ids]
    elif transform_name == "FIRST_3_HYPHEN":
        ex = [first_n_hyphen_tokens(x, 3) for x in expr_ids]
        cl = [first_n_hyphen_tokens(x, 3) for x in clin_ids]
    elif transform_name == "FIRST_4_HYPHEN":
        ex = [first_n_hyphen_tokens(x, 4) for x in expr_ids]
        cl = [first_n_hyphen_tokens(x, 4) for x in clin_ids]
    elif transform_name == "TARGET_CASE_SUFFIX_STRIP":
        ex = [strip_common_sample_suffix(x) for x in expr_ids]
        cl = [strip_common_sample_suffix(x) for x in clin_ids]
    else:
        raise ValueError(transform_name)

    cl_set = set(cl)
    matched = [x for x in ex if x in cl_set]

    return {
        "transform": transform_name,
        "expression_unique": len(set(ex)),
        "clinical_unique": len(set(cl)),
        "matched_expression": len(matched),
        "expression_total": len(ex),
        "complete_expression_match": len(matched) == len(ex),
        "clinical_duplicate_keys": len(cl) - len(set(cl)),
        "expression_duplicate_keys": len(ex) - len(set(ex)),
    }


def main() -> None:
    print("=" * 120)
    print("Paper 6 - probe TARGET clinical identifier schema (OUTCOME-FREE)")
    print("=" * 120)
    print(f"Script version: {SCRIPT_VERSION}")
    print()
    print("Safety:")
    print("  clinical HEADER read: YES")
    print("  identifier-like clinical columns read: YES")
    print("  OS/event/status/survival/time columns read: NO")
    print("  scientific artifacts modified: NO")
    print()

    if not UPSTREAM_LOCK.exists():
        raise FileNotFoundError(UPSTREAM_LOCK)
    if not TARGET_ROSTER.exists():
        raise FileNotFoundError(TARGET_ROSTER)

    paper4_root, root_source = resolve_paper4_root()
    lock = read_json(UPSTREAM_LOCK)
    role = preferred_clinical_asset_role(lock)

    asset = (lock.get("assets") or {}).get(role)
    if not isinstance(asset, dict):
        raise RuntimeError(f"Missing asset role {role}")

    rel = clean(asset.get("relative_path"))
    clinical_path = paper4_root / rel

    if not clinical_path.exists():
        raise FileNotFoundError(clinical_path)

    header = read_header(clinical_path)

    print(f"Paper4 root resolution: {root_source}")
    print(f"Locked clinical asset role: {role}")
    print(f"Clinical relative path: {rel}")
    print()
    print("Clinical header columns:")
    for i, name in enumerate(header):
        print(f"  [{i:02d}] {name}")

    candidates = [
        name
        for name in header
        if is_safe_identifier_candidate(name)
    ]

    print()
    print("Safe identifier-like candidate columns:")
    if not candidates:
        print("  NONE")
        raise RuntimeError(
            "No safe identifier-like column candidates found from header."
        )

    for name in candidates:
        print(f"  - {name}")

    # Read ONLY safe identifier-like columns.
    clinical_ids_df = read_columns(
        clinical_path,
        candidates,
    )

    roster = pd.read_csv(
        TARGET_ROSTER,
        sep="\t",
        dtype=str,
        low_memory=False,
    ).fillna("")

    expr_ids = (
        roster.sort_values("sample_index")["sample_id"]
        .astype(str)
        .map(clean)
        .tolist()
    )

    print()
    print("Frozen TARGET expression IDs (first 12):")
    for value in expr_ids[:12]:
        print(f"  {value}")

    transforms = [
        "EXACT_UPPER",
        "FIRST_3_HYPHEN",
        "FIRST_4_HYPHEN",
        "TARGET_CASE_SUFFIX_STRIP",
    ]

    print()
    print("=" * 120)
    print("IDENTIFIER OVERLAP AUDIT")
    print("=" * 120)

    best = None

    for column in candidates:
        values = [
            clean(x)
            for x in clinical_ids_df[column]
            if clean(x)
        ]

        print()
        print(f"Candidate column: {column}")
        print(
            f"  rows={len(clinical_ids_df)}, "
            f"nonblank={len(values)}, unique_nonblank={len(set(values))}"
        )
        print("  first 12 nonblank values:")
        for value in values[:12]:
            print(f"    {value}")

        for transform in transforms:
            stats = overlap_stats(
                expr_ids,
                values,
                transform,
            )
            print(
                f"  {transform:24s} "
                f"match={stats['matched_expression']:2d}/{stats['expression_total']} "
                f"clinical_dup_keys={stats['clinical_duplicate_keys']} "
                f"expr_dup_keys={stats['expression_duplicate_keys']}"
            )

            score = (
                stats["matched_expression"],
                -stats["clinical_duplicate_keys"],
                -stats["expression_duplicate_keys"],
            )

            if best is None or score > best[0]:
                best = (
                    score,
                    column,
                    stats,
                )

    print()
    print("=" * 120)
    print("BEST NON-OUTCOME ID MATCH")
    print("=" * 120)

    if best is None:
        print("No candidate/transform combination evaluated.")
    else:
        _, column, stats = best
        print(f"Column: {column}")
        print(f"Transform: {stats['transform']}")
        print(
            f"Matched expression samples: "
            f"{stats['matched_expression']}/{stats['expression_total']}"
        )
        print(
            f"Clinical duplicate transformed keys: "
            f"{stats['clinical_duplicate_keys']}"
        )
        print(
            f"Expression duplicate transformed keys: "
            f"{stats['expression_duplicate_keys']}"
        )

    print()
    print("Outcome values read: NO")
    print("This probe freezes nothing; use result only to make a pre-outcome technical schema fix.")
    print("=" * 120)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 120, file=sys.stderr)
        print("05f1c0 identifier probe: FAIL", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print("=" * 120, file=sys.stderr)
        raise
