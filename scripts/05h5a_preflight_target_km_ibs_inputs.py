#!/usr/bin/env python
"""
Paper 6 - read-only preflight for the 05h5 TARGET Kaplan-Meier IBS reference.

This stage computes NO Kaplan-Meier curve and NO new IBS value.

Purpose
-------
Before implementing 05h5, inspect and bind the actual frozen TARGET evaluation
artifacts and source code that already exist in this repository. The goal is to
identify, rather than guess:

  - authoritative 05f3f completion/result artifacts
  - parsed TARGET86 endpoint file and exact column names
  - 20 x 5 outer-fold identity / patient membership representation
  - exact existing IBS implementation and its time-support construction
  - censoring/IPCW implementation
  - aggregation level used for the existing TARGET IBS values
  - whether per-fold predictions or only aggregate results were retained

The output is a technical inventory only. It must be reviewed before a separate
05h5b scientific computation script is written.

Expected placement:
    scripts/05h5a_preflight_target_km_ibs_inputs.py

Run:
    python scripts\05h5a_preflight_target_km_ibs_inputs.py
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


SCRIPT_VERSION = "05h5a-preflight-target-km-ibs-inputs-v1-readonly-no-cli"

KEYWORDS = [
    "integrated_brier",
    "integrated brier",
    "brier",
    "breslow",
    "kaplan",
    "censor",
    "ipcw",
    "surv",
    "time_grid",
    "times_grid",
    "eval_times",
    "evaluation_times",
    "tau",
    "fold",
    "repeat",
    "outer",
    "train_idx",
    "test_idx",
    "train_ids",
    "test_ids",
    "os_event",
    "os_time_days",
]

TEXT_SUFFIXES = {
    ".py", ".json", ".txt", ".md", ".csv", ".tsv", ".yaml", ".yml",
}

TABULAR_SUFFIXES = {".csv", ".tsv"}


def project_root() -> Path:
    p = Path(__file__).resolve()
    if p.parent.name.lower() != "scripts":
        raise RuntimeError(
            "Place 05h5a_preflight_target_km_ibs_inputs.py "
            "in the repository scripts/ directory."
        )
    return p.parent.parent


ROOT = project_root()

H0_JSON = (
    ROOT
    / "method_contract"
    / "05h0_cbm_strengthening_contract"
    / "cbm_strengthening_contract.json"
)

F3F_DIR = ROOT / "results" / "human_posthold_descriptive" / "05f3f"

# Historical frozen/pre-outcome namespaces that may contain exact fold mechanics
# or the frozen runner. We inspect only what actually exists.
CANDIDATE_RESULT_DIRS = [
    ROOT / "results" / "human_posthold_descriptive" / "05f3f",
    ROOT / "results" / "human_posthold_descriptive" / "05f3e",
    ROOT / "results" / "human_posthold_descriptive" / "05f3d",
    ROOT / "results" / "human_posthold_descriptive" / "05f3c",
    ROOT / "results" / "human_posthold_descriptive" / "05f3b_v2",
    ROOT / "results" / "human_posthold_descriptive" / "05f2c_v2",
    ROOT / "results" / "human_posthold_descriptive" / "05f2b",
    ROOT / "results" / "human_posthold_descriptive" / "05f1c",
]

CANDIDATE_SCRIPT_GLOBS = [
    "05f3f*.py",
    "05f3e*.py",
    "05f3d*.py",
    "05f3c*.py",
    "05f3b*.py",
    "05f2c*.py",
    "05f2b*.py",
    "05f1c*.py",
    "04c*.py",
]

OUT_DIR = ROOT / "method_contract" / "05h5a_target_km_ibs_preflight"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8", errors="strict"))


def recursive_find_key(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            hit = recursive_find_key(value, key)
            if hit is not None:
                return hit
    elif isinstance(obj, list):
        for value in obj:
            hit = recursive_find_key(value, key)
            if hit is not None:
                return hit
    return None


def verify_h0_contract() -> dict[str, Any]:
    if not H0_JSON.exists():
        raise FileNotFoundError(f"Missing 05h0 contract: {H0_JSON}")

    h0 = load_json(H0_JSON)
    block = recursive_find_key(h0, "target_km_ibs_reference")
    if not isinstance(block, dict):
        raise RuntimeError("05h0 lacks target_km_ibs_reference.")

    if block.get("analysis_role") != "POST-OPENING DESCRIPTIVE NULL REFERENCE":
        raise RuntimeError(
            "05h0 target_km_ibs_reference analysis role is not the frozen expected role."
        )
    if block.get("allowed_new_estimation") is not True:
        raise RuntimeError("05h0 does not authorize the KM IBS reference estimation.")

    method = str(block.get("method", ""))
    required_phrases = [
        "outer-training",
        "Kaplan-Meier",
        "outer-test",
        "same frozen time horizon",
        "censoring mechanics",
        "fold structure",
        "aggregation",
    ]
    missing = [p for p in required_phrases if p.lower() not in method.lower()]
    if missing:
        raise RuntimeError(
            f"05h0 KM-reference method text lost expected elements: {missing}"
        )

    prohibited = block.get("prohibited_uses", [])
    if not isinstance(prohibited, list) or len(prohibited) < 4:
        raise RuntimeError("05h0 KM-reference prohibited-use list is incomplete.")

    return block


def inventory_files() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for base in CANDIDATE_RESULT_DIRS:
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if not p.is_file():
                continue
            rows.append(
                {
                    "kind": "result_artifact",
                    "namespace": str(base.relative_to(ROOT)),
                    "relative_path": str(p.relative_to(ROOT)),
                    "suffix": p.suffix.lower(),
                    "bytes": p.stat().st_size,
                    "sha256": sha256_file(p),
                }
            )

    seen = set()
    for pattern in CANDIDATE_SCRIPT_GLOBS:
        for p in sorted((ROOT / "scripts").glob(pattern)):
            if not p.is_file():
                continue
            key = str(p.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "kind": "script",
                    "namespace": "scripts",
                    "relative_path": str(p.relative_to(ROOT)),
                    "suffix": p.suffix.lower(),
                    "bytes": p.stat().st_size,
                    "sha256": sha256_file(p),
                }
            )

    return rows


def read_table_header_and_shape(path: Path) -> tuple[int | None, int | None, list[str], str | None]:
    try:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        df = pd.read_csv(path, sep=sep, low_memory=False)
        return len(df), len(df.columns), [str(c) for c in df.columns], None
    except Exception as exc:
        return None, None, [], f"{type(exc).__name__}: {exc}"


def score_tabular_candidate(columns: Iterable[str]) -> dict[str, bool]:
    cols = [str(c).lower() for c in columns]

    def has_any(parts: Iterable[str]) -> bool:
        return any(any(part in c for part in parts) for c in cols)

    return {
        "has_id": has_any(["case", "sample", "patient", "subject", "id"]),
        "has_event": has_any(["event", "status"]),
        "has_time": has_any(["time", "days", "survival", "os_"]),
        "has_repeat": has_any(["repeat"]),
        "has_fold": has_any(["fold"]),
        "has_train_test": has_any(["train", "test"]),
        "has_model": has_any(["model"]),
        "has_ibs": has_any(["ibs", "brier"]),
        "has_risk": has_any(["risk", "score", "prediction"]),
    }


def inspect_tabular_files(inventory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in inventory:
        if item["suffix"] not in TABULAR_SUFFIXES:
            continue
        p = ROOT / item["relative_path"]
        nrows, ncols, columns, error = read_table_header_and_shape(p)
        flags = score_tabular_candidate(columns)
        rows.append(
            {
                "relative_path": item["relative_path"],
                "n_rows": nrows,
                "n_columns": ncols,
                "columns": " | ".join(columns),
                **flags,
                "read_error": error or "",
            }
        )
    return rows


def safe_read_text(path: Path) -> str | None:
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return None
    # Avoid accidentally reading an enormous tabular file as plain text.
    if path.stat().st_size > 25 * 1024 * 1024:
        return None
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


def keyword_matches(inventory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in inventory:
        p = ROOT / item["relative_path"]
        text = safe_read_text(p)
        if text is None:
            continue

        for lineno, line in enumerate(text.splitlines(), start=1):
            low = line.lower()
            hits = [kw for kw in KEYWORDS if kw in low]
            if hits:
                rows.append(
                    {
                        "relative_path": item["relative_path"],
                        "line_number": lineno,
                        "keywords": ";".join(sorted(set(hits))),
                        "line": line.strip()[:1000],
                    }
                )
    return rows


def python_function_inventory(inventory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    interesting_tokens = {
        "brier", "ibs", "fold", "surv", "censor", "breslow", "kaplan",
        "bootstrap", "metric", "evaluate", "split",
    }
    rows: list[dict[str, Any]] = []

    for item in inventory:
        if item["kind"] != "script" or item["suffix"] != ".py":
            continue
        p = ROOT / item["relative_path"]
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(text)
        except Exception:
            continue

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = node.name
                low = name.lower()
                if any(token in low for token in interesting_tokens):
                    rows.append(
                        {
                            "relative_path": item["relative_path"],
                            "function": name,
                            "line_number": int(node.lineno),
                            "end_line_number": int(
                                getattr(node, "end_lineno", node.lineno)
                            ),
                        }
                    )
    return rows


def inspect_05f3f_jsons() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not F3F_DIR.exists():
        return rows

    for p in sorted(F3F_DIR.rglob("*.json")):
        try:
            obj = load_json(p)
        except Exception as exc:
            rows.append(
                {
                    "relative_path": str(p.relative_to(ROOT)),
                    "status": "",
                    "top_level_keys": "",
                    "read_error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        if isinstance(obj, dict):
            keys = sorted(str(k) for k in obj.keys())
            status = (
                obj.get("status")
                or obj.get("scientific_status")
                or obj.get("decision")
                or ""
            )
            rows.append(
                {
                    "relative_path": str(p.relative_to(ROOT)),
                    "status": str(status),
                    "top_level_keys": " | ".join(keys),
                    "read_error": "",
                }
            )
        else:
            rows.append(
                {
                    "relative_path": str(p.relative_to(ROOT)),
                    "status": "",
                    "top_level_keys": f"<{type(obj).__name__}>",
                    "read_error": "",
                }
            )
    return rows


def print_candidate_tables(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No readable CSV/TSV files found in inspected namespaces.")
        return

    # Only print tables likely to matter for endpoint/fold/metric binding.
    interesting = [
        r for r in rows
        if (
            r["has_repeat"]
            or r["has_fold"]
            or (r["has_id"] and r["has_event"] and r["has_time"])
            or r["has_ibs"]
            or r["has_risk"]
        )
    ]

    print("Candidate endpoint/fold/metric tables:")
    if not interesting:
        print("  none detected by column-name inspection")
        return

    for r in interesting:
        print(f"  {r['relative_path']}")
        print(f"    shape: {r['n_rows']} x {r['n_columns']}")
        print(f"    columns: {r['columns']}")
        if r["read_error"]:
            print(f"    read_error: {r['read_error']}")


def print_code_matches(matches: list[dict[str, Any]]) -> None:
    selected = [
        row for row in matches
        if any(
            k in row["keywords"].split(";")
            for k in [
                "integrated_brier",
                "integrated brier",
                "brier",
                "breslow",
                "kaplan",
                "ipcw",
                "censor",
                "time_grid",
                "times_grid",
                "eval_times",
                "evaluation_times",
            ]
        )
    ]

    print("IBS / censoring / time-support source-code matches:")
    if not selected:
        print("  none found in inspected text files")
        return

    # Cap console volume but save the complete list to CSV.
    for row in selected[:180]:
        print(
            f"  {row['relative_path']}:{row['line_number']}: "
            f"{row['line']}"
        )
    if len(selected) > 180:
        print(f"  ... {len(selected) - 180} additional matches saved to CSV")


def main() -> None:
    print("=" * 118)
    print("Paper 6 - 05h5a read-only TARGET KM-IBS preflight")
    print("=" * 118)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {ROOT}")
    print(f"Output directory: {OUT_DIR}")
    print()
    print("Execution scope:")
    print("  Kaplan-Meier curve fitted: NO")
    print("  new IBS value calculated: NO")
    print("  model fitting / retuning: NO")
    print("  GSE21257/GSE39055 outcomes read: NO")
    print("  TARGET scientific result modified: NO")
    print("  existing TARGET artifacts/schemas inspected: YES")
    print("  existing source code inspected as text: YES")
    print()

    if OUT_DIR.exists():
        raise RuntimeError(
            f"05h5a output already exists; refusing overwrite: {OUT_DIR}"
        )
    OUT_DIR.mkdir(parents=True, exist_ok=False)

    try:
        block = verify_h0_contract()
        print("05h0 KM-reference contract: PASS")
        print(f"  05h0 SHA256: {sha256_file(H0_JSON)}")
        print(f"  analysis role: {block['analysis_role']}")
        print()

        if not F3F_DIR.exists():
            raise FileNotFoundError(
                f"Expected completed 05f3f namespace does not exist: {F3F_DIR}"
            )

        inventory = inventory_files()
        inv_df = pd.DataFrame(inventory)
        inv_df.to_csv(OUT_DIR / "file_inventory.csv", index=False)

        print("Existing artifact/script inventory:")
        print(f"  total inspected files: {len(inv_df)}")
        f3f_count = int(
            (
                inv_df["relative_path"]
                .astype(str)
                .str.startswith("results\\human_posthold_descriptive\\05f3f")
                |
                inv_df["relative_path"]
                .astype(str)
                .str.startswith("results/human_posthold_descriptive/05f3f")
            ).sum()
        )
        print(f"  files under 05f3f: {f3f_count}")
        print()

        json_rows = inspect_05f3f_jsons()
        pd.DataFrame(json_rows).to_csv(
            OUT_DIR / "05f3f_json_inventory.csv",
            index=False,
        )

        print("05f3f JSON artifacts:")
        for row in json_rows:
            print(f"  {row['relative_path']}")
            if row["status"]:
                print(f"    status/decision: {row['status']}")
            print(f"    keys: {row['top_level_keys']}")
            if row["read_error"]:
                print(f"    read_error: {row['read_error']}")
        print()

        table_rows = inspect_tabular_files(inventory)
        pd.DataFrame(table_rows).to_csv(
            OUT_DIR / "tabular_schema_inventory.csv",
            index=False,
        )
        print_candidate_tables(table_rows)
        print()

        matches = keyword_matches(inventory)
        pd.DataFrame(matches).to_csv(
            OUT_DIR / "keyword_source_matches.csv",
            index=False,
        )
        print_code_matches(matches)
        print()

        funcs = python_function_inventory(inventory)
        pd.DataFrame(funcs).to_csv(
            OUT_DIR / "python_function_inventory.csv",
            index=False,
        )

        print("Potentially relevant Python functions:")
        for row in funcs[:160]:
            print(
                f"  {row['relative_path']}:{row['line_number']}-"
                f"{row['end_line_number']}  {row['function']}"
            )
        if len(funcs) > 160:
            print(f"  ... {len(funcs) - 160} additional functions saved to CSV")
        print()

        # Save the exact 05h0 block so 05h5b can be written against this preflight
        # without another semantic guess.
        (OUT_DIR / "05h0_target_km_ibs_reference.json").write_text(
            json.dumps(block, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )

        # A compact hash manifest for the preflight itself.
        outputs = sorted(p for p in OUT_DIR.iterdir() if p.is_file())
        manifest = {
            "script_version": SCRIPT_VERSION,
            "status": "PASS_TECHNICAL_INVENTORY_READY_FOR_05H5B_DESIGN",
            "scientific_km_or_ibs_estimate_computed": False,
            "05h0_sha256": sha256_file(H0_JSON),
            "05f3f_namespace": str(F3F_DIR.relative_to(ROOT)),
            "outputs": [
                {
                    "file": p.name,
                    "sha256": sha256_file(p),
                    "bytes": p.stat().st_size,
                }
                for p in outputs
            ],
        }
        (OUT_DIR / "preflight_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )

        print("=" * 118)
        print("05h5a TARGET KM-IBS technical preflight: PASS")
        print("=" * 118)
        print("Scientific KM reference calculated: NO")
        print("Scientific IBS reference calculated: NO")
        print("Ready to bind 05h5b to actual frozen TARGET mechanics: YES")
        print()
        print("Saved:")
        for p in sorted(OUT_DIR.iterdir()):
            if p.is_file():
                print(f"  {p}")

    except Exception:
        # Keep the read-only diagnostic directory only if it contains useful
        # inventory output; no scientific artifact can be mistaken for a PASS.
        raise


if __name__ == "__main__":
    main()
