#!/usr/bin/env python
"""
Paper 6 - read-only probe for 05h2 safety-source schema.

Purpose:
  Resolve exactly where the already-computed per-scenario / per-replicate
  catastrophic-transfer rate lives before writing the corrected 05h2 exporter.

This script:
  - does NOT fit or retune models;
  - does NOT read TARGET/GSE21257/GSE39055/DOG2 outcomes;
  - does NOT modify any existing result;
  - creates no scientific output artifact;
  - only prints schemas, file inventory, hashes, and relevant source-code lines.

Expected placement:
    scripts/05h2a_probe_safety_source_schema.py

Run:
    python scripts\05h2a_probe_safety_source_schema.py
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path

import pandas as pd


SCRIPT_VERSION = "05h2a-probe-safety-source-schema-v1-no-cli"


def project_root() -> Path:
    p = Path(__file__).resolve()
    if p.parent.name.lower() != "scripts":
        raise RuntimeError("Place this file in the repository scripts/ directory.")
    return p.parent.parent


ROOT = project_root()

CANONICAL = (
    ROOT
    / "results"
    / "simulation_phase_diagram"
    / "05d"
    / "scenario_model_metric_summary.tsv"
)
DIAGNOSTIC = (
    ROOT
    / "results"
    / "simulation_phase_diagram_diagnostics"
    / "05d0a"
    / "scenario_failure_diagnostics.tsv"
)

INVENTORY_DIRS = [
    ROOT / "results" / "simulation_phase_diagram" / "05d",
    ROOT / "results" / "simulation_phase_diagram_diagnostics" / "05d0",
    ROOT / "results" / "simulation_phase_diagram_diagnostics" / "05d0a",
    ROOT / "results" / "simulation_phase_diagram_diagnostics" / "05d0f",
]

SOURCE_SCRIPT_HINTS = [
    "05d0_diagnose_frozen_ai_failure_modes.py",
    "05d0a_followup_units_oracle_riskscale_diagnostic.py",
    "05d0f_audit_utility_comparator_regime_safety.py",
]

MATCH_RE = re.compile(
    r"catastroph|negative_transfer|delta[_ ]?c|delta[_ ]?uno|"
    r"scenario_model_metric_summary|scenario_failure_diagnostics|"
    r"read_csv|read_parquet|read_pickle|np\.load|parquet|\.npz|\.pkl|"
    r"replicate",
    re.IGNORECASE,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t", low_memory=False)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    raise ValueError(path)


def show_table(path: Path, label: str) -> None:
    print("-" * 118)
    print(label)
    print("-" * 118)
    print(f"path: {path}")
    print(f"exists: {path.exists()}")
    if not path.exists():
        print()
        return

    print(f"bytes: {path.stat().st_size}")
    print(f"sha256: {sha256_file(path)}")
    df = read_table(path)
    print(f"shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("columns:")
    for i, c in enumerate(df.columns, start=1):
        print(f"  {i:02d}. {c}")

    interesting = [
        c for c in df.columns
        if re.search(
            r"scenario|regime|model|rep|uno|delta|negative|catastroph|rate|flag",
            str(c),
            re.IGNORECASE,
        )
    ]
    if interesting:
        print()
        print("first 3 rows — safety-relevant columns:")
        with pd.option_context(
            "display.max_columns", None,
            "display.width", 240,
            "display.max_colwidth", 80,
        ):
            print(df[interesting].head(3).to_string(index=False))
    print()


def show_inventory(base: Path) -> None:
    print("-" * 118)
    print(f"FILE INVENTORY: {base.relative_to(ROOT) if base.exists() else base}")
    print("-" * 118)
    if not base.exists():
        print("directory does not exist")
        print()
        return

    files = sorted(p for p in base.rglob("*") if p.is_file())
    print(f"file count: {len(files)}")
    suffixes = Counter(p.suffix.lower() or "<none>" for p in files)
    print("suffix counts:", dict(sorted(suffixes.items())))

    # Print everything for small diagnostic directories; cap very large trees.
    limit = 160
    for p in files[:limit]:
        rel = p.relative_to(ROOT)
        print(f"  {rel} [{p.stat().st_size} bytes]")
    if len(files) > limit:
        print(f"  ... {len(files) - limit} additional files omitted from display")
    print()


def show_other_tabular_schemas(base: Path) -> None:
    if not base.exists():
        return
    files = sorted(
        p for p in base.rglob("*")
        if p.is_file() and p.suffix.lower() in {".csv", ".tsv"}
    )
    if not files:
        return

    print("-" * 118)
    print(f"TABULAR SCHEMAS: {base.relative_to(ROOT)}")
    print("-" * 118)
    for p in files:
        if p in {CANONICAL, DIAGNOSTIC}:
            continue
        try:
            sep = "\t" if p.suffix.lower() == ".tsv" else ","
            df = pd.read_csv(p, sep=sep, nrows=2, low_memory=False)
        except Exception as exc:
            print(f"{p.relative_to(ROOT)} :: READ_FAIL {type(exc).__name__}: {exc}")
            continue
        cols = [str(c) for c in df.columns]
        safety = [
            c for c in cols
            if re.search(
                r"scenario|regime|model|rep|uno|delta|negative|catastroph|rate|flag",
                c,
                re.IGNORECASE,
            )
        ]
        if safety:
            print(f"{p.relative_to(ROOT)}")
            print("  " + " | ".join(safety))
    print()


def show_source_matches() -> None:
    print("-" * 118)
    print("RELEVANT SOURCE-CODE LINES")
    print("-" * 118)

    scripts = []
    for name in SOURCE_SCRIPT_HINTS:
        p = ROOT / "scripts" / name
        if p.exists():
            scripts.append(p)

    # Also include closely named 05d/05d0 scripts in case the exact historical
    # filename differs.
    for p in sorted((ROOT / "scripts").glob("05d*.py")):
        if p not in scripts:
            scripts.append(p)

    total_matches = 0
    max_total = 320
    for p in scripts:
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        matches = [
            (i, line.strip())
            for i, line in enumerate(lines, start=1)
            if MATCH_RE.search(line)
        ]
        if not matches:
            continue

        print()
        print(f"[{p.name}]")
        for i, line in matches:
            print(f"{i}: {line}")
            total_matches += 1
            if total_matches >= max_total:
                print()
                print(f"Source-match display capped at {max_total} lines.")
                return
    print()


def main() -> None:
    print("=" * 118)
    print("Paper 6 - read-only probe for 05h2 safety-source schema")
    print("=" * 118)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {ROOT}")
    print()
    print("Safety / execution contract:")
    print("  TARGET survival outcomes read: NO")
    print("  GSE21257 outcomes read: NO")
    print("  GSE39055 outcomes read: NO")
    print("  DOG2 outcomes read: NO")
    print("  model fitting / retuning: NO")
    print("  survival-metric re-estimation: NO")
    print("  scientific output files created: NO")
    print("  existing result-table schemas inspected: YES")
    print("  historical 05d/05d0 source code inspected as text: YES")
    print()

    show_table(CANONICAL, "CANONICAL FROZEN 05d SCENARIO-MODEL SUMMARY")
    show_table(DIAGNOSTIC, "POST-HOLD 05d0a SCENARIO FAILURE DIAGNOSTICS")

    for base in INVENTORY_DIRS:
        show_inventory(base)

    for base in INVENTORY_DIRS:
        show_other_tabular_schemas(base)

    show_source_matches()

    print("=" * 118)
    print("05h2a read-only safety-source probe: PASS")
    print("=" * 118)
    print(
        "No scientific result was created or changed. "
        "Use this output only to bind the corrected 05h2 exporter to an existing metric source."
    )


if __name__ == "__main__":
    main()
