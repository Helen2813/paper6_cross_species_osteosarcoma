from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


SCRIPT_VERSION = "05h1-export-anchor-provenance-v1-no-cli"
ANCHOR_ORDER = ["M34", "M40", "M11", "M24"]
EXPECTED_COUNTS = {"M34": 162, "M40": 111, "M11": 7, "M24": 7}
EXPECTED_CLASSES = {
    "M34": "strong_external_canine_representation_preservation",
    "M40": "strong_external_canine_representation_preservation",
    "M11": "no_clear_external_canine_representation_preservation",
    "M24": "no_clear_external_canine_representation_preservation",
}

STRICT_WEIGHTS_BASENAME = "GSE238110_frozen_transfer_gene_weights_strict.csv"
EXTERNAL_EVIDENCE_BASENAME = "paper4_locked_independent_canine_representation.csv"
EXTERNAL_MANIFEST_BASENAME = "paper4_external_canine_evidence_manifest.json"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_bytes(obj: Any) -> bytes:
    return (
        json.dumps(
            obj,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def recursively_collect_strings(obj: Any) -> list[str]:
    out: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.append(str(k))
            out.extend(recursively_collect_strings(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(recursively_collect_strings(v))
    elif isinstance(obj, str):
        out.append(obj)
    return out


def find_dict_containing_keys(obj: Any, required: set[str]) -> dict[str, Any] | None:
    if isinstance(obj, dict):
        if required.issubset(set(obj.keys())):
            return obj
        for value in obj.values():
            found = find_dict_containing_keys(value, required)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = find_dict_containing_keys(value, required)
            if found is not None:
                return found
    return None


def load_g0_module(root: Path):
    path = root / "scripts" / "05g0_freeze_postopening_preresult_outcomeblind_biology_contract.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("paper6_05g0_provenance_replay", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import 05g0 module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def path_strings_from_known_configs(root: Path) -> list[Path]:
    candidates: list[Path] = []
    json_files = [
        root / "contracts" / "00_upstream_input_lock.json",
        root / "_config" / "paths.local.json",
        root / "config" / "paths.local.json",
    ]
    for jf in json_files:
        if not jf.is_file():
            continue
        try:
            payload = read_json(jf)
        except Exception:
            continue
        for text in recursively_collect_strings(payload):
            t = clean(text).strip('"').strip("'")
            if not t:
                continue
            # Avoid treating ordinary labels as filesystem paths.
            if (":\\" not in t) and (":/" not in t) and not t.startswith("\\\\"):
                continue
            try:
                p = Path(os.path.expandvars(t)).expanduser()
            except Exception:
                continue
            candidates.append(p if p.is_dir() else p.parent)
    return candidates


def dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for p in paths:
        try:
            key = str(p.resolve()).lower()
        except Exception:
            key = str(p).lower()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def find_by_relative_paths(
    roots: list[Path],
    relpaths: list[Path],
    expected_sha256: str | None = None,
) -> list[Path]:
    found: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        bases = [root]
        # Sometimes a config points one or two levels inside the upstream project.
        bases.extend(list(root.parents)[:4])
        for base in bases:
            for rel in relpaths:
                p = base / rel
                if p.is_file():
                    if expected_sha256 and sha256_file(p) != expected_sha256:
                        continue
                    found.append(p)
    return dedupe_paths(found)


def bounded_filename_search(
    base: Path,
    filename: str,
    expected_sha256: str | None = None,
    max_depth: int = 6,
) -> list[Path]:
    found: list[Path] = []
    if not base.is_dir():
        return found

    base_depth = len(base.parts)
    prune_names = {
        ".git", ".venv", "venv", "env", "node_modules", "__pycache__",
        ".idea", ".vscode", "dist", "build"
    }

    for dirpath, dirnames, filenames in os.walk(base):
        current = Path(dirpath)
        depth = len(current.parts) - base_depth
        if depth >= max_depth:
            dirnames[:] = []
        else:
            dirnames[:] = [d for d in dirnames if d not in prune_names]

        if filename in filenames:
            p = current / filename
            if expected_sha256 and sha256_file(p) != expected_sha256:
                continue
            found.append(p)

    return dedupe_paths(found)


def resolve_source_file(
    *,
    root: Path,
    basename: str,
    relpaths: list[Path],
    expected_sha256: str | None,
    extra_roots: list[Path],
) -> Path:
    candidate_roots = dedupe_paths(
        extra_roots
        + path_strings_from_known_configs(root)
        + [root, root.parent]
    )

    matches = find_by_relative_paths(
        candidate_roots,
        relpaths,
        expected_sha256=expected_sha256,
    )

    # Also allow an exact file path already embedded in configs.
    for r in candidate_roots:
        direct = r / basename
        if direct.is_file():
            if expected_sha256 is None or sha256_file(direct) == expected_sha256:
                matches.append(direct)
    matches = dedupe_paths(matches)

    if not matches:
        # Last-resort bounded search on the Desktop/project parent.
        matches = bounded_filename_search(
            root.parent,
            basename,
            expected_sha256=expected_sha256,
            max_depth=6,
        )

    if len(matches) == 0:
        raise FileNotFoundError(
            f"Could not locate authoritative upstream file {basename!r}. "
            f"Expected SHA-256={expected_sha256 or 'not frozen here'}."
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"More than one authoritative match found for {basename!r}:\n"
            + "\n".join(str(x) for x in matches)
            + "\nRefusing to choose between duplicate matching artifacts."
        )
    return matches[0]


def find_registry_file(g0_dir: Path) -> Path:
    candidates = sorted(
        [
            p for p in g0_dir.iterdir()
            if p.is_file()
            and "registry" in p.name.lower()
            and p.suffix.lower() in {".tsv", ".csv"}
        ],
        key=lambda p: p.name.lower(),
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one 05g0 registry artifact; found {len(candidates)}: "
            f"{[p.name for p in candidates]}"
        )
    return candidates[0]


def read_table(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    return pd.read_csv(path, sep=sep)


def normalize_anchor_source(
    weights: pd.DataFrame,
    g0: Any,
) -> tuple[pd.DataFrame, str]:
    if "module_label" not in weights.columns:
        raise RuntimeError(
            f"Strict weights do not contain module_label. Columns={list(weights.columns)}"
        )

    if hasattr(g0, "infer_human_gene_column"):
        gene_col = str(g0.infer_human_gene_column(weights))
    else:
        guesses = [
            "human_gene_symbol",
            "canine_gene_symbol",
            "gene_symbol",
            "symbol",
            "gene",
        ]
        gene_col = next((x for x in guesses if x in weights.columns), "")
        if not gene_col:
            raise RuntimeError(
                f"Could not infer gene-symbol column. Columns={list(weights.columns)}"
            )

    out = weights.loc[
        weights["module_label"].astype(str).str.strip().isin(ANCHOR_ORDER)
    ].copy()

    if hasattr(g0, "normalize_symbol"):
        out["canonical_human_gene_symbol"] = out[gene_col].map(g0.normalize_symbol)
    else:
        out["canonical_human_gene_symbol"] = (
            out[gene_col].astype(str).str.strip().str.upper()
        )

    out["module_label"] = out["module_label"].astype(str).str.strip()
    if out["canonical_human_gene_symbol"].eq("").any():
        raise RuntimeError("Blank canonical gene symbol in frozen anchor source.")
    if out.duplicated(["module_label", "canonical_human_gene_symbol"]).any():
        raise RuntimeError("Duplicate anchor module/gene membership in strict weights.")

    observed_counts = (
        out.groupby("module_label")["canonical_human_gene_symbol"]
        .nunique()
        .to_dict()
    )
    observed_counts = {str(k): int(v) for k, v in observed_counts.items()}
    if observed_counts != EXPECTED_COUNTS:
        raise RuntimeError(
            f"Frozen anchor gene counts changed: observed={observed_counts}, "
            f"expected={EXPECTED_COUNTS}"
        )

    order_map = {m: i for i, m in enumerate(ANCHOR_ORDER)}
    out["_module_order"] = out["module_label"].map(order_map)
    out = out.sort_values(
        ["_module_order", "canonical_human_gene_symbol"],
        kind="mergesort",
    ).drop(columns="_module_order")
    return out, gene_col


def membership_hash(genes: list[str], g0: Any) -> str:
    genes = [clean(x) for x in genes if clean(x)]
    if hasattr(g0, "sha256_lines"):
        return clean(g0.sha256_lines(genes))
    payload = ("\n".join(sorted(genes)) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    root = project_root()
    h0_dir = root / "method_contract" / "05h0_cbm_strengthening_contract"
    g0_dir = root / "results" / "human_posthold_descriptive" / "05g0"
    g1_dir = root / "results" / "human_posthold_descriptive" / "05g1"
    out_dir = root / "method_contract" / "05h1_anchor_provenance"

    print("=" * 118)
    print("Paper 6 - export self-contained frozen anchor provenance for CBM strengthening")
    print("=" * 118)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {root}")
    print(f"Output directory: {out_dir}")
    print()
    print("Safety / execution contract:")
    print("  TARGET survival outcomes read: NO")
    print("  GSE21257 outcomes read: NO")
    print("  GSE39055 outcomes read: NO")
    print("  model fitting: NO")
    print("  scientific re-estimation: NO")
    print("  Paper-4 outcome-blind frozen anchor artifacts read: YES")
    print("  pre-05f3f outcome-blind 05g0/05g1 provenance read: YES")
    print()

    if out_dir.exists():
        raise FileExistsError(
            f"Immutable 05h1 output already exists: {out_dir}\n"
            "Do not overwrite a provenance freeze."
        )

    required = [
        h0_dir / "cbm_strengthening_contract.json",
        h0_dir / "freeze_manifest.json",
        g0_dir / "biological_context_analysis_contract.json",
        g0_dir / "summary.json",
        g1_dir / "summary.json",
        g1_dir / "paper4_anchor_structural_context.tsv",
    ]
    for p in required:
        if not p.is_file():
            raise FileNotFoundError(f"Required pre-existing artifact missing: {p}")

    # Verify 05h0 contract against its own freeze manifest.
    h0_manifest = read_json(h0_dir / "freeze_manifest.json")
    h0_contract_path = h0_dir / "cbm_strengthening_contract.json"
    h0_contract_sha = sha256_file(h0_contract_path)
    expected_h0_sha = clean(h0_manifest.get("contract_sha256"))
    if h0_contract_sha != expected_h0_sha:
        raise RuntimeError(
            "05h0 strengthening contract SHA mismatch; refusing to continue."
        )

    g0_contract_path = g0_dir / "biological_context_analysis_contract.json"
    g0_contract = read_json(g0_contract_path)
    g0_summary_path = g0_dir / "summary.json"
    g0_summary = read_json(g0_summary_path)
    g1_summary_path = g1_dir / "summary.json"
    g1_summary = read_json(g1_summary_path)

    # Locate the provenance block frozen by 05g0.
    provenance = find_dict_containing_keys(
        g0_contract,
        {"strict_weights_sha256", "anchor_gene_counts", "anchor_prior_classes"},
    )
    if provenance is None:
        provenance = find_dict_containing_keys(
            g0_summary,
            {"strict_weights_sha256", "anchor_gene_counts", "anchor_prior_classes"},
        )
    if provenance is None:
        raise RuntimeError(
            "Could not locate the frozen Paper-4 anchor provenance block in 05g0."
        )

    frozen_counts = {
        str(k): int(v) for k, v in dict(provenance["anchor_gene_counts"]).items()
    }
    frozen_classes = {
        str(k): clean(v) for k, v in dict(provenance["anchor_prior_classes"]).items()
    }
    if frozen_counts != EXPECTED_COUNTS:
        raise RuntimeError(
            f"05g0 frozen anchor counts differ from contract: {frozen_counts}"
        )
    if frozen_classes != EXPECTED_CLASSES:
        raise RuntimeError(
            f"05g0 frozen anchor classes differ from contract: {frozen_classes}"
        )

    strict_expected_sha = clean(provenance.get("strict_weights_sha256"))
    manifest_expected_sha = clean(
        provenance.get("external_outcome_blind_manifest_sha256")
    )
    if not strict_expected_sha or not manifest_expected_sha:
        raise RuntimeError("05g0 provenance is missing authoritative source hashes.")

    # Import 05g0 only for its frozen path constants / normalization functions.
    # Its main routine is not executed.
    g0 = load_g0_module(root)

    def attr_path(name: str) -> Path | None:
        value = getattr(g0, name, None)
        return Path(value) if value is not None else None

    strict_rel = attr_path("P4_STRICT_WEIGHTS_REL")
    external_manifest_rel = attr_path("P4_EXTERNAL_REP_MANIFEST_REL")

    # The exact external evidence-table constant name is intentionally discovered,
    # because historical script revisions used slightly different names.
    external_table_rel: Path | None = None
    for name, value in vars(g0).items():
        if not name.startswith("P4_"):
            continue
        if not isinstance(value, Path):
            continue
        if value.name == EXTERNAL_EVIDENCE_BASENAME:
            external_table_rel = value
            break

    strict_relpaths = [x for x in [strict_rel, Path(STRICT_WEIGHTS_BASENAME)] if x]
    manifest_relpaths = [
        x for x in [external_manifest_rel, Path(EXTERNAL_MANIFEST_BASENAME)] if x
    ]
    external_relpaths = [
        x for x in [external_table_rel, Path(EXTERNAL_EVIDENCE_BASENAME)] if x
    ]

    extra_roots: list[Path] = []
    # Existing absolute source paths may be represented in g0 attributes.
    for value in vars(g0).values():
        if isinstance(value, Path) and value.is_absolute():
            extra_roots.append(value if value.is_dir() else value.parent)

    strict_path = resolve_source_file(
        root=root,
        basename=STRICT_WEIGHTS_BASENAME,
        relpaths=strict_relpaths,
        expected_sha256=strict_expected_sha,
        extra_roots=extra_roots,
    )
    external_manifest_path = resolve_source_file(
        root=root,
        basename=EXTERNAL_MANIFEST_BASENAME,
        relpaths=manifest_relpaths,
        expected_sha256=manifest_expected_sha,
        extra_roots=extra_roots,
    )
    external_table_path = resolve_source_file(
        root=root,
        basename=EXTERNAL_EVIDENCE_BASENAME,
        relpaths=external_relpaths,
        expected_sha256=None,
        extra_roots=extra_roots + [strict_path.parent, external_manifest_path.parent],
    )

    if sha256_file(strict_path) != strict_expected_sha:
        raise RuntimeError("Strict frozen weights no longer match 05g0 SHA-256.")
    if sha256_file(external_manifest_path) != manifest_expected_sha:
        raise RuntimeError("External outcome-blind manifest no longer matches 05g0 SHA-256.")

    external_manifest = read_json(external_manifest_path)
    if bool(external_manifest.get("outcome_loaded", False)):
        raise RuntimeError(
            "Authoritative external anchor manifest reports outcome_loaded=true."
        )

    # Registry was frozen before 05f3f result inspection.
    registry_path = find_registry_file(g0_dir)
    registry = read_table(registry_path)

    id_col = next(
        (c for c in ["set_id", "module_label", "program"] if c in registry.columns),
        None,
    )
    if id_col is None:
        raise RuntimeError(
            f"Could not identify set ID column in 05g0 registry. "
            f"Columns={list(registry.columns)}"
        )

    if "is_paper4_anchor" in registry.columns:
        raw = registry["is_paper4_anchor"]
        if raw.dtype == bool:
            anchor_registry = registry.loc[raw].copy()
        else:
            anchor_registry = registry.loc[
                raw.astype(str).str.strip().str.lower().isin(
                    {"true", "1", "yes", "y"}
                )
            ].copy()
    elif "set_family" in registry.columns:
        anchor_registry = registry.loc[
            registry["set_family"].astype(str).str.strip()
            == "PAPER4_FROZEN_PROGRAM_ANCHOR"
        ].copy()
    else:
        anchor_registry = registry.loc[
            registry[id_col].astype(str).str.strip().isin(ANCHOR_ORDER)
        ].copy()

    anchor_registry[id_col] = anchor_registry[id_col].astype(str).str.strip()
    if set(anchor_registry[id_col]) != set(ANCHOR_ORDER) or len(anchor_registry) != 4:
        raise RuntimeError(
            "05g0 registry does not contain exactly the four frozen anchors."
        )

    weights = pd.read_csv(strict_path)
    anchor_weights, original_gene_col = normalize_anchor_source(weights, g0)

    # Verify exact memberships against the pre-05f3f registry hash if available.
    registry_hash_col = (
        "membership_sha256" if "membership_sha256" in anchor_registry.columns else None
    )
    summary_rows: list[dict[str, Any]] = []
    for module in ANCHOR_ORDER:
        part = anchor_weights.loc[anchor_weights["module_label"].eq(module)].copy()
        genes = part["canonical_human_gene_symbol"].tolist()
        observed_membership_sha = membership_hash(genes, g0)

        reg_row = anchor_registry.loc[
            anchor_registry[id_col].eq(module)
        ].iloc[0]

        if registry_hash_col:
            expected_membership_sha = clean(reg_row[registry_hash_col])
            if observed_membership_sha != expected_membership_sha:
                raise RuntimeError(
                    f"{module}: frozen membership SHA differs from 05g0 registry.\n"
                    f"observed={observed_membership_sha}\n"
                    f"expected={expected_membership_sha}"
                )
        else:
            expected_membership_sha = observed_membership_sha

        if "prior_anchor_class" in anchor_registry.columns:
            prior = clean(reg_row["prior_anchor_class"])
            if prior != EXPECTED_CLASSES[module]:
                raise RuntimeError(
                    f"{module}: registry prior class changed: {prior!r}"
                )

        summary_rows.append(
            {
                "program": module,
                "frozen_source_gene_count": EXPECTED_COUNTS[module],
                "membership_sha256": observed_membership_sha,
                "prior_preservation_class": EXPECTED_CLASSES[module],
                "strict_weights_source_sha256": strict_expected_sha,
                "external_evidence_source_sha256": sha256_file(external_table_path),
                "external_outcome_blind_manifest_sha256": manifest_expected_sha,
                "external_manifest_outcome_loaded": False,
                "chronology_anchor": (
                    "05g0 pre-05f3f-result-inspection outcome-blind freeze"
                ),
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    external = pd.read_csv(external_table_path)
    external_id_col = next(
        (c for c in ["module_label", "set_id", "program"] if c in external.columns),
        None,
    )
    if external_id_col is None:
        raise RuntimeError(
            f"Could not identify program column in external evidence table. "
            f"Columns={list(external.columns)}"
        )
    external_subset = external.loc[
        external[external_id_col].astype(str).str.strip().isin(ANCHOR_ORDER)
    ].copy()
    external_subset[external_id_col] = (
        external_subset[external_id_col].astype(str).str.strip()
    )
    if set(external_subset[external_id_col]) != set(ANCHOR_ORDER):
        raise RuntimeError(
            "External evidence table does not contain all four frozen anchors."
        )
    external_subset["_order"] = external_subset[external_id_col].map(
        {m: i for i, m in enumerate(ANCHOR_ORDER)}
    )
    external_subset = external_subset.sort_values(
        "_order", kind="mergesort"
    ).drop(columns="_order")

    g1_anchor_path = g1_dir / "paper4_anchor_structural_context.tsv"
    g1_anchor = pd.read_csv(g1_anchor_path, sep="\t")
    g1_id_col = next(
        (c for c in ["set_id", "module_label", "program"] if c in g1_anchor.columns),
        None,
    )
    if g1_id_col is None:
        raise RuntimeError(
            f"Could not identify anchor ID in 05g1 structural context. "
            f"Columns={list(g1_anchor.columns)}"
        )
    if set(g1_anchor[g1_id_col].astype(str).str.strip()) != set(ANCHOR_ORDER):
        raise RuntimeError(
            "05g1 structural-context artifact does not contain exact four anchors."
        )

    # Create immutable output only after all checks pass.
    out_dir.mkdir(parents=True, exist_ok=False)

    weights_out = out_dir / "Supplementary_Data_anchor_membership_weights.csv"
    evidence_out = out_dir / "Supplementary_Data_anchor_prior_evidence.csv"
    context_out = out_dir / "Supplementary_Data_anchor_structural_context.tsv"
    summary_out = out_dir / "Supplementary_Table_anchor_provenance.csv"
    provenance_out = out_dir / "anchor_provenance.json"
    readme_out = out_dir / "README_for_supplement.txt"

    anchor_weights.to_csv(weights_out, index=False)
    external_subset.to_csv(evidence_out, index=False)
    g1_anchor.to_csv(context_out, sep="\t", index=False)
    summary_df.to_csv(summary_out, index=False)

    provenance_payload = {
        "artifact_name": "Paper 6 self-contained frozen molecular-program anchor provenance",
        "script_version": SCRIPT_VERSION,
        "strengthening_contract_sha256": h0_contract_sha,
        "anchor_order": ANCHOR_ORDER,
        "anchor_gene_counts": EXPECTED_COUNTS,
        "anchor_prior_classes": EXPECTED_CLASSES,
        "source_gene_column_in_strict_weights": original_gene_col,
        "source_artifacts": {
            "strict_weights": {
                "basename": strict_path.name,
                "frozen_relative_path": clean(
                    provenance.get("strict_weights_relative_path")
                ),
                "sha256": sha256_file(strict_path),
                "sha256_frozen_by_05g0": strict_expected_sha,
            },
            "external_prior_evidence": {
                "basename": external_table_path.name,
                "sha256": sha256_file(external_table_path),
            },
            "external_outcome_blind_manifest": {
                "basename": external_manifest_path.name,
                "frozen_relative_path": clean(
                    provenance.get("external_outcome_blind_manifest_relative_path")
                ),
                "sha256": sha256_file(external_manifest_path),
                "sha256_frozen_by_05g0": manifest_expected_sha,
                "outcome_loaded": False,
            },
            "paper6_05g0_contract": {
                "relative_path": g0_contract_path.relative_to(root).as_posix(),
                "sha256": sha256_file(g0_contract_path),
            },
            "paper6_05g0_registry": {
                "relative_path": registry_path.relative_to(root).as_posix(),
                "sha256": sha256_file(registry_path),
            },
            "paper6_05g1_anchor_context": {
                "relative_path": g1_anchor_path.relative_to(root).as_posix(),
                "sha256": sha256_file(g1_anchor_path),
            },
            "paper6_05g1_summary": {
                "relative_path": g1_summary_path.relative_to(root).as_posix(),
                "sha256": sha256_file(g1_summary_path),
            },
        },
        "scientific_guardrails": {
            "source_is_preexisting": True,
            "paper4_external_manifest_outcome_loaded": False,
            "paper6_05g0_frozen_before_05f3f_result_inspection": True,
            "anchor_membership_reestimated": False,
            "anchor_weights_reestimated": False,
            "anchor_prior_classes_reestimated": False,
            "TARGET_survival_outcomes_read_by_05h1": False,
            "GSE21257_outcomes_read_by_05h1": False,
            "GSE39055_outcomes_read_by_05h1": False,
        },
        "programs": summary_rows,
    }
    provenance_out.write_bytes(canonical_json_bytes(provenance_payload))

    readme = """Paper 6 supplementary anchor provenance
================================================

Purpose
-------
This bundle makes the four frozen molecular-program anchors used in Paper 6
independently auditable without requiring publication of the separate upstream
molecular-preservation manuscript.

Files
-----
1. Supplementary_Data_anchor_membership_weights.csv
   Exact frozen anchor membership and all retained source weight/sign columns
   for M34, M40, M11, and M24, plus a canonical normalized gene symbol.

2. Supplementary_Data_anchor_prior_evidence.csv
   The four rows from the pre-existing independent external canine
   representation analysis used to define the prior anchor classes.

3. Supplementary_Data_anchor_structural_context.tsv
   The pre-05f3f-bound, outcome-blind Paper 6 structural-context results
   generated in 05g1.

4. Supplementary_Table_anchor_provenance.csv
   Compact program-level provenance table suitable for the Supplement.

5. anchor_provenance.json
   Machine-readable source hashes, chronology, memberships, and guardrails.

Suggested manuscript wording
----------------------------
"Four molecular programs (M34, M40, M11, and M24) were retained as
independently frozen structural anchors from a pre-existing outcome-blind
external canine preservation analysis. Their exact memberships, source
weights/signs, prior preservation labels, and cryptographic provenance are
provided in Supplementary Data, and were not re-estimated from Paper 6
outcomes."

Important
---------
These artifacts document previously frozen definitions. They do not reopen
or modify the primary Paper 6 HOLD decision, TARGET interpretation, or any
sealed external cohort.
"""
    readme_out.write_text(readme, encoding="utf-8", newline="\n")

    output_files = [
        weights_out,
        evidence_out,
        context_out,
        summary_out,
        provenance_out,
        readme_out,
    ]
    freeze_manifest = {
        "freeze_name": "05h1_anchor_provenance",
        "freeze_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script_version": SCRIPT_VERSION,
        "strengthening_contract_sha256": h0_contract_sha,
        "output_files_sha256": {
            p.name: sha256_file(p) for p in output_files
        },
        "source_files_sha256": {
            strict_path.name: sha256_file(strict_path),
            external_table_path.name: sha256_file(external_table_path),
            external_manifest_path.name: sha256_file(external_manifest_path),
            registry_path.name: sha256_file(registry_path),
            g1_anchor_path.name: sha256_file(g1_anchor_path),
        },
        "TARGET_outcomes_accessed": False,
        "sealed_cohort_outcomes_accessed": False,
        "models_fitted": False,
        "metrics_reestimated": False,
    }
    freeze_manifest_path = out_dir / "freeze_manifest.json"
    freeze_manifest_path.write_bytes(canonical_json_bytes(freeze_manifest))

    print("-" * 118)
    print("Authoritative anchor provenance resolved")
    print("-" * 118)
    print(f"  strict weights: {strict_path}")
    print(f"  external prior evidence: {external_table_path}")
    print(f"  external outcome-blind manifest: {external_manifest_path}")
    print(f"  05g0 registry: {registry_path}")
    print(f"  05g1 structural context: {g1_anchor_path}")
    print()
    for row in summary_rows:
        print(
            f"  {row['program']}: genes={row['frozen_source_gene_count']}; "
            f"prior={row['prior_preservation_class']}; "
            f"membership_sha256={row['membership_sha256'][:16]}..."
        )
    print()
    print("=" * 118)
    print("05h1 self-contained anchor provenance: PASS")
    print("=" * 118)
    for p in output_files + [freeze_manifest_path]:
        print(f"Created: {p}")


if __name__ == "__main__":
    main()
