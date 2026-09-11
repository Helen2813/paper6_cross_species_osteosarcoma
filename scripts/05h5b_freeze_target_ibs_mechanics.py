#!/usr/bin/env python
"""
Paper 6 - freeze exact TARGET IBS/fold mechanics before KM-null estimation.

This is a TECHNICAL CONTRACT stage only. It computes NO new Kaplan-Meier
reference and NO new IBS value.

Why this stage exists
---------------------
05h5a successfully identified the authoritative TARGET86 artifacts and the
historical IBS implementation. Before any new null-reference estimation, this
script freezes the exact mechanics that 05h5c must reuse:

  - TARGET86 endpoint identity and 86/29/57 counts
  - exact 20 x 5 outer split registry and held-out memberships
  - exact existing fold-metric and OOF-prediction structure
  - exact 05f3f runner and 04c implementation hashes
  - exact source text/signatures of the frozen IBS helper functions
  - exact call site used by the TARGET runner for IBS
  - exact model-summary aggregation structure observed in existing outputs

No model fitting, no KM fitting, no new IBS, and no human cohort other than the
already-opened TARGET86 endpoint is analyzed.

Expected placement:
    scripts/05h5b_freeze_target_ibs_mechanics.py

Run:
    python scripts\\05h5b_freeze_target_ibs_mechanics.py
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_VERSION = "05h5b-freeze-target-ibs-mechanics-v1-readonly-no-cli"


def project_root() -> Path:
    p = Path(__file__).resolve()
    if p.parent.name.lower() != "scripts":
        raise RuntimeError(
            "Place 05h5b_freeze_target_ibs_mechanics.py in the repository scripts/ directory."
        )
    return p.parent.parent


ROOT = project_root()

H0_JSON = (
    ROOT
    / "method_contract"
    / "05h0_cbm_strengthening_contract"
    / "cbm_strengthening_contract.json"
)

H5A_DIR = ROOT / "method_contract" / "05h5a_target_km_ibs_preflight"
H5A_MANIFEST = H5A_DIR / "preflight_manifest.json"

F3F_DIR = ROOT / "results" / "human_posthold_descriptive" / "05f3f"
F3F_SUMMARY = F3F_DIR / "summary.json"
F3F_INPUT_AUDIT = F3F_DIR / "TARGET86_execution_input_audit.json"
ENDPOINT = F3F_DIR / "execution_input" / "TARGET_OS_complete86_primary_endpoint.tsv"

FROZEN_EXEC = F3F_DIR / "frozen_execution"
SPLIT_REGISTRY = FROZEN_EXEC / "TARGET_split_registry.tsv"
FOLD_METRICS = FROZEN_EXEC / "TARGET_fold_metrics.tsv"
MODEL_SUMMARY = FROZEN_EXEC / "TARGET_model_summary.tsv"
OOF_PRED = FROZEN_EXEC / "TARGET_oof_predictions.tsv"

RUNNER = (
    F3F_DIR
    / "runner"
    / "frozen_TARGET86_evaluation_runner_v3_bytesafe.py"
)
M04C = ROOT / "scripts" / "04c_run_classical_premise_test.py"

OUT_DIR = ROOT / "method_contract" / "05h5b_target_ibs_mechanics"

EXPECTED_N = 86
EXPECTED_EVENTS = 29
EXPECTED_CENSORED = 57
EXPECTED_REPEATS = 20
EXPECTED_FOLDS = 5
EXPECTED_MODELS = 9
EXPECTED_FOLD_METRIC_ROWS = EXPECTED_REPEATS * EXPECTED_FOLDS * EXPECTED_MODELS
EXPECTED_SPLIT_ROWS = EXPECTED_N * EXPECTED_REPEATS * EXPECTED_FOLDS
EXPECTED_OOF_ROWS = EXPECTED_N * EXPECTED_REPEATS * EXPECTED_MODELS

M04C_FUNCTIONS = [
    "censoring_km_training",
    "ipcw_brier_score",
    "choose_ibs_grid",
    "integrated_brier_custom",
    "weighted_mean_ibs",
]

RUNNER_FUNCTIONS = [
    "_breslow_survival_from_fixed_risk",
]

RUNNER_KEY_PATTERNS = [
    "choose_ibs_grid",
    "integrated_brier_custom",
    "weighted_mean_ibs",
    "ibs_valid_repeats",
    "TARGET_fold_metrics",
    "TARGET_model_summary",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.write_text(
        json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


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


def verify_required_files() -> None:
    required = [
        H0_JSON,
        H5A_MANIFEST,
        F3F_SUMMARY,
        F3F_INPUT_AUDIT,
        ENDPOINT,
        SPLIT_REGISTRY,
        FOLD_METRICS,
        MODEL_SUMMARY,
        OOF_PRED,
        RUNNER,
        M04C,
    ]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"Required artifact missing: {p}")


def verify_h0() -> dict[str, Any]:
    h0 = load_json(H0_JSON)
    block = recursive_find_key(h0, "target_km_ibs_reference")
    if not isinstance(block, dict):
        raise RuntimeError("05h0 lacks target_km_ibs_reference.")
    if block.get("analysis_role") != "POST-OPENING DESCRIPTIVE NULL REFERENCE":
        raise RuntimeError("Unexpected 05h0 target_km_ibs_reference role.")
    if block.get("allowed_new_estimation") is not True:
        raise RuntimeError("05h0 does not authorize the new KM reference.")

    text = str(block.get("method", "")).lower()
    required_phrases = [
        "outer-training",
        "kaplan-meier",
        "outer-test",
        "same frozen time horizon",
        "censoring mechanics",
        "fold structure",
        "aggregation",
    ]
    missing = [x for x in required_phrases if x not in text]
    if missing:
        raise RuntimeError(f"05h0 method text missing required concepts: {missing}")
    return block


def verify_05h5a() -> dict[str, Any]:
    manifest = load_json(H5A_MANIFEST)
    if manifest.get("status") != "PASS_TECHNICAL_INVENTORY_READY_FOR_05H5B_DESIGN":
        raise RuntimeError(
            f"05h5a preflight is not PASS: {manifest.get('status')!r}"
        )
    if manifest.get("scientific_km_or_ibs_estimate_computed") is not False:
        raise RuntimeError("05h5a unexpectedly reports a scientific estimate.")
    if manifest.get("05h0_sha256") != sha256_file(H0_JSON):
        raise RuntimeError("05h5a 05h0 hash no longer matches current 05h0.")
    return manifest


def verify_f3f_summary() -> tuple[dict[str, Any], dict[str, Any]]:
    summary = load_json(F3F_SUMMARY)
    audit = load_json(F3F_INPUT_AUDIT)

    if summary.get("status") != "PASS":
        raise RuntimeError(f"05f3f summary status is not PASS: {summary.get('status')!r}")
    if audit.get("status") != "PASS":
        raise RuntimeError(
            f"05f3f TARGET86 input audit is not PASS: {audit.get('status')!r}"
        )

    checks = [
        ("TARGET_n", summary.get("TARGET_n"), EXPECTED_N),
        ("TARGET_events", summary.get("TARGET_events"), EXPECTED_EVENTS),
        ("TARGET_censored", summary.get("TARGET_censored"), EXPECTED_CENSORED),
        ("audit.n", audit.get("n"), EXPECTED_N),
        ("audit.events", audit.get("events"), EXPECTED_EVENTS),
        ("audit.censored", audit.get("censored"), EXPECTED_CENSORED),
    ]
    bad = [(name, obs, exp) for name, obs, exp in checks if int(obs) != exp]
    if bad:
        raise RuntimeError(f"TARGET86 counts changed: {bad}")

    endpoint_hash = sha256_file(ENDPOINT)
    if summary.get("TARGET86_endpoint_sha256") != endpoint_hash:
        raise RuntimeError(
            "05f3f summary endpoint hash differs from current TARGET86 endpoint."
        )
    if audit.get("TARGET86_endpoint_sha256") != endpoint_hash:
        raise RuntimeError(
            "05f3f input-audit endpoint hash differs from current TARGET86 endpoint."
        )

    return summary, audit


def load_endpoint() -> pd.DataFrame:
    ep = pd.read_csv(ENDPOINT, sep="\t", low_memory=False)
    required = [
        "execution_index_86",
        "source_sample_index_88",
        "sample_id",
        "case_key",
        "os_time_days",
        "os_event",
    ]
    missing = [c for c in required if c not in ep.columns]
    if missing:
        raise RuntimeError(f"TARGET86 endpoint lost columns: {missing}")
    if len(ep) != EXPECTED_N:
        raise RuntimeError(f"TARGET86 endpoint rows={len(ep)}, expected=86.")

    ep["execution_index_86"] = pd.to_numeric(
        ep["execution_index_86"], errors="raise"
    ).astype(int)
    ep["os_time_days"] = pd.to_numeric(ep["os_time_days"], errors="raise").astype(float)
    ep["os_event"] = pd.to_numeric(ep["os_event"], errors="raise").astype(int)

    if ep["execution_index_86"].nunique() != EXPECTED_N:
        raise RuntimeError("TARGET86 execution_index_86 is not unique.")
    if ep["sample_id"].astype(str).nunique() != EXPECTED_N:
        raise RuntimeError("TARGET86 sample_id is not unique.")
    if ep["case_key"].astype(str).nunique() != EXPECTED_N:
        raise RuntimeError("TARGET86 case_key is not unique.")
    if not set(ep["os_event"].unique()).issubset({0, 1}):
        raise RuntimeError("TARGET86 os_event is not binary.")
    if int(ep["os_event"].sum()) != EXPECTED_EVENTS:
        raise RuntimeError("TARGET86 event count is not 29.")
    if int((ep["os_event"] == 0).sum()) != EXPECTED_CENSORED:
        raise RuntimeError("TARGET86 censored count is not 57.")
    if (ep["os_time_days"] < 0).any():
        raise RuntimeError("TARGET86 contains a negative survival time.")

    zero = ep[ep["os_time_days"] == 0]
    if len(zero) != 1 or int(zero.iloc[0]["os_event"]) != 0:
        raise RuntimeError(
            "Expected exactly one retained censored time-zero TARGET observation."
        )

    return ep.sort_values("execution_index_86").reset_index(drop=True)


def load_and_validate_splits(ep: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    splits = pd.read_csv(SPLIT_REGISTRY, sep="\t", low_memory=False)
    required = ["repeat", "outer_fold", "sample_index", "sample_id", "case_key", "role"]
    missing = [c for c in required if c not in splits.columns]
    if missing:
        raise RuntimeError(f"TARGET split registry lost columns: {missing}")
    if len(splits) != EXPECTED_SPLIT_ROWS:
        raise RuntimeError(
            f"TARGET split registry rows={len(splits)}, expected={EXPECTED_SPLIT_ROWS}."
        )

    splits["repeat"] = pd.to_numeric(splits["repeat"], errors="raise").astype(int)
    splits["outer_fold"] = pd.to_numeric(
        splits["outer_fold"], errors="raise"
    ).astype(int)
    splits["sample_index"] = pd.to_numeric(
        splits["sample_index"], errors="raise"
    ).astype(int)
    splits["role_norm"] = splits["role"].astype(str).str.strip().str.upper()

    repeats = sorted(splits["repeat"].unique().tolist())
    folds = sorted(splits["outer_fold"].unique().tolist())
    roles = sorted(splits["role_norm"].unique().tolist())

    if len(repeats) != EXPECTED_REPEATS:
        raise RuntimeError(f"Observed repeat labels {repeats}; expected 20 unique repeats.")
    if len(folds) != EXPECTED_FOLDS:
        raise RuntimeError(f"Observed outer_fold labels {folds}; expected 5 unique folds.")
    if len(roles) != 2:
        raise RuntimeError(f"Expected exactly two split roles; observed {roles}.")

    # Detect which role is held out from its exact combinatorial property:
    # within each repeat, each patient must occur exactly once in that role.
    role_stats: dict[str, dict[str, Any]] = {}
    for role in roles:
        r = splits[splits["role_norm"] == role]
        per_repeat_sample = (
            r.groupby(["repeat", "sample_index"]).size()
        )
        role_stats[role] = {
            "rows": int(len(r)),
            "min_rows_per_repeat_sample": int(per_repeat_sample.min()),
            "max_rows_per_repeat_sample": int(per_repeat_sample.max()),
            "all_repeat_sample_pairs_present": (
                len(per_repeat_sample) == EXPECTED_REPEATS * EXPECTED_N
            ),
        }

    test_candidates = [
        role
        for role, stat in role_stats.items()
        if stat["all_repeat_sample_pairs_present"]
        and stat["min_rows_per_repeat_sample"] == 1
        and stat["max_rows_per_repeat_sample"] == 1
    ]
    if len(test_candidates) != 1:
        raise RuntimeError(
            f"Could not uniquely identify held-out split role from registry: {role_stats}"
        )
    test_role = test_candidates[0]
    train_role = [r for r in roles if r != test_role][0]

    train_counts = (
        splits[splits["role_norm"] == train_role]
        .groupby(["repeat", "sample_index"])
        .size()
    )
    if len(train_counts) != EXPECTED_REPEATS * EXPECTED_N:
        raise RuntimeError("Not every TARGET patient has train-role records in every repeat.")
    if train_counts.min() != EXPECTED_FOLDS - 1 or train_counts.max() != EXPECTED_FOLDS - 1:
        raise RuntimeError(
            "Train-role multiplicity is not exactly 4 folds per patient per repeat."
        )

    # Exact endpoint identity cross-check.
    ep_lookup = ep.set_index("execution_index_86")
    if set(splits["sample_index"]) != set(ep_lookup.index):
        raise RuntimeError("Split registry sample_index set differs from TARGET86 endpoint.")

    for row in splits[["sample_index", "sample_id", "case_key"]].drop_duplicates().itertuples(index=False):
        ref = ep_lookup.loc[int(row.sample_index)]
        if str(row.sample_id) != str(ref["sample_id"]):
            raise RuntimeError(f"sample_id mismatch for execution index {row.sample_index}.")
        if str(row.case_key) != str(ref["case_key"]):
            raise RuntimeError(f"case_key mismatch for execution index {row.sample_index}.")

    # Within every repeat/fold, train and test must partition all 86 exactly.
    fold_structure_rows = []
    for repeat in repeats:
        for fold in folds:
            part = splits[
                (splits["repeat"] == repeat)
                & (splits["outer_fold"] == fold)
            ]
            if len(part) != EXPECTED_N:
                raise RuntimeError(
                    f"repeat={repeat}, fold={fold}: split rows={len(part)}, expected=86."
                )
            if part["sample_index"].nunique() != EXPECTED_N:
                raise RuntimeError(
                    f"repeat={repeat}, fold={fold}: patient membership is not unique."
                )

            test_idx = sorted(
                part.loc[part["role_norm"] == test_role, "sample_index"]
                .astype(int)
                .tolist()
            )
            train_idx = sorted(
                part.loc[part["role_norm"] == train_role, "sample_index"]
                .astype(int)
                .tolist()
            )
            if set(test_idx) & set(train_idx):
                raise RuntimeError(f"repeat={repeat}, fold={fold}: train/test overlap.")
            if set(test_idx) | set(train_idx) != set(ep_lookup.index):
                raise RuntimeError(f"repeat={repeat}, fold={fold}: train/test do not cover all 86.")

            train_events = int(ep_lookup.loc[train_idx, "os_event"].sum())
            test_events = int(ep_lookup.loc[test_idx, "os_event"].sum())
            fold_structure_rows.append(
                {
                    "repeat": repeat,
                    "outer_fold": fold,
                    "n_train": len(train_idx),
                    "events_train": train_events,
                    "n_test": len(test_idx),
                    "events_test": test_events,
                    "train_indices_csv": ",".join(map(str, train_idx)),
                    "test_indices_csv": ",".join(map(str, test_idx)),
                }
            )

    fold_structure = pd.DataFrame(fold_structure_rows)
    if len(fold_structure) != EXPECTED_REPEATS * EXPECTED_FOLDS:
        raise RuntimeError("Expected exactly 100 repeat/fold partitions.")

    metadata = {
        "repeat_labels": repeats,
        "outer_fold_labels": folds,
        "role_labels_observed": roles,
        "heldout_role_resolved": test_role,
        "training_role_resolved": train_role,
        "role_stats": role_stats,
    }
    return fold_structure, metadata


def load_and_validate_fold_metrics(
    ep: pd.DataFrame,
    fold_structure: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    fm = pd.read_csv(FOLD_METRICS, sep="\t", low_memory=False)
    required = [
        "repeat", "outer_fold", "model_id",
        "n_train", "events_train", "n_test", "events_test",
        "uno_c", "ibs", "ibs_status", "risk_sd", "risk_q99_q01",
    ]
    missing = [c for c in required if c not in fm.columns]
    if missing:
        raise RuntimeError(f"TARGET fold metrics lost columns: {missing}")
    if len(fm) != EXPECTED_FOLD_METRIC_ROWS:
        raise RuntimeError(
            f"TARGET fold metrics rows={len(fm)}, expected={EXPECTED_FOLD_METRIC_ROWS}."
        )

    models = list(dict.fromkeys(fm["model_id"].astype(str).tolist()))
    if len(set(models)) != EXPECTED_MODELS:
        raise RuntimeError(f"Expected 9 unique TARGET models; observed {models}.")

    if fm.duplicated(["repeat", "outer_fold", "model_id"]).any():
        raise RuntimeError("TARGET fold metrics are not unique at repeat/fold/model.")

    fs = fold_structure.set_index(["repeat", "outer_fold"])
    for row in fm.itertuples(index=False):
        ref = fs.loc[(int(row.repeat), int(row.outer_fold))]
        for col in ["n_train", "events_train", "n_test", "events_test"]:
            if int(getattr(row, col)) != int(ref[col]):
                raise RuntimeError(
                    f"Fold-metric {col} mismatch at repeat={row.repeat}, "
                    f"fold={row.outer_fold}, model={row.model_id}."
                )

    return fm, models


def load_and_validate_oof(
    ep: pd.DataFrame,
    fold_structure: pd.DataFrame,
    models: list[str],
    split_meta: dict[str, Any],
) -> pd.DataFrame:
    oof = pd.read_csv(OOF_PRED, sep="\t", low_memory=False)
    required = [
        "repeat", "outer_fold", "model_id",
        "sample_index", "sample_id", "case_key", "risk",
    ]
    missing = [c for c in required if c not in oof.columns]
    if missing:
        raise RuntimeError(f"TARGET OOF predictions lost columns: {missing}")
    if len(oof) != EXPECTED_OOF_ROWS:
        raise RuntimeError(
            f"TARGET OOF rows={len(oof)}, expected={EXPECTED_OOF_ROWS}."
        )

    if set(oof["model_id"].astype(str)) != set(models):
        raise RuntimeError("TARGET OOF model set differs from fold-metric model set.")

    if oof.duplicated(["repeat", "model_id", "sample_index"]).any():
        raise RuntimeError(
            "TARGET OOF predictions are not unique at repeat/model/sample."
        )

    # Every patient must have one held-out prediction for every repeat/model.
    counts = oof.groupby(["repeat", "model_id"]).size()
    if counts.min() != EXPECTED_N or counts.max() != EXPECTED_N:
        raise RuntimeError("Every repeat/model does not have exactly 86 OOF predictions.")

    # outer_fold in OOF must equal the held-out fold in the split registry.
    test_role = split_meta["heldout_role_resolved"]
    splits = pd.read_csv(SPLIT_REGISTRY, sep="\t", low_memory=False)
    splits["role_norm"] = splits["role"].astype(str).str.strip().str.upper()
    test_map = (
        splits[splits["role_norm"] == test_role]
        [["repeat", "outer_fold", "sample_index"]]
        .drop_duplicates()
    )
    if test_map.duplicated(["repeat", "sample_index"]).any():
        raise RuntimeError("Held-out split map is not unique at repeat/sample.")

    chk = oof.merge(
        test_map,
        on=["repeat", "sample_index"],
        how="left",
        suffixes=("_oof", "_split"),
        validate="many_to_one",
    )
    if chk["outer_fold_split"].isna().any():
        raise RuntimeError("Some OOF rows have no held-out split membership.")
    if not (
        pd.to_numeric(chk["outer_fold_oof"], errors="raise").astype(int)
        == pd.to_numeric(chk["outer_fold_split"], errors="raise").astype(int)
    ).all():
        raise RuntimeError("OOF outer_fold disagrees with held-out split registry.")

    # Endpoint IDs must match.
    ep_lookup = ep.set_index("execution_index_86")
    unique_ids = oof[["sample_index", "sample_id", "case_key"]].drop_duplicates()
    for row in unique_ids.itertuples(index=False):
        ref = ep_lookup.loc[int(row.sample_index)]
        if str(row.sample_id) != str(ref["sample_id"]):
            raise RuntimeError(f"OOF sample_id mismatch at sample_index={row.sample_index}.")
        if str(row.case_key) != str(ref["case_key"]):
            raise RuntimeError(f"OOF case_key mismatch at sample_index={row.sample_index}.")

    return oof


def load_and_validate_model_summary(
    fm: pd.DataFrame,
    models: list[str],
) -> pd.DataFrame:
    ms = pd.read_csv(MODEL_SUMMARY, sep="\t", low_memory=False)
    required = [
        "model_id", "uno_c", "uno_valid_repeats", "ibs", "ibs_valid_repeats",
        "risk_sd", "risk_sd_valid_repeats",
        "risk_q99_q01", "risk_q99_q01_valid_repeats",
    ]
    missing = [c for c in required if c not in ms.columns]
    if missing:
        raise RuntimeError(f"TARGET model summary lost columns: {missing}")
    if len(ms) != EXPECTED_MODELS:
        raise RuntimeError(f"TARGET model summary rows={len(ms)}, expected=9.")
    if set(ms["model_id"].astype(str)) != set(models):
        raise RuntimeError("TARGET model-summary model set differs from fold metrics.")
    if ms["model_id"].duplicated().any():
        raise RuntimeError("TARGET model summary has duplicate model IDs.")
    return ms


def parse_functions(path: Path) -> tuple[str, ast.AST, dict[str, ast.AST]]:
    text = path.read_text(encoding="utf-8", errors="strict")
    tree = ast.parse(text)
    funcs: dict[str, ast.AST] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs[node.name] = node
    return text, tree, funcs


def source_segment(text: str, node: ast.AST) -> str:
    lines = text.splitlines()
    start = int(getattr(node, "lineno"))
    end = int(getattr(node, "end_lineno", start))
    return "\n".join(lines[start - 1:end]) + "\n"


def function_signature_from_ast(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    # Preserve exact syntactic signature by taking the def header through the colon.
    header = source_segment("", node) if False else None  # keeps mypy quiet
    args = []
    posonly = getattr(node.args, "posonlyargs", [])
    regular = node.args.args
    all_pos = list(posonly) + list(regular)
    n_defaults = len(node.args.defaults)
    default_start = len(all_pos) - n_defaults

    for i, arg in enumerate(all_pos):
        piece = arg.arg
        if arg.annotation is not None:
            piece += ": " + ast.unparse(arg.annotation)
        if i >= default_start:
            default_node = node.args.defaults[i - default_start]
            piece += "=" + ast.unparse(default_node)
        args.append(piece)
        if posonly and i + 1 == len(posonly):
            args.append("/")

    if node.args.vararg is not None:
        piece = "*" + node.args.vararg.arg
        if node.args.vararg.annotation is not None:
            piece += ": " + ast.unparse(node.args.vararg.annotation)
        args.append(piece)
    elif node.args.kwonlyargs:
        args.append("*")

    for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
        piece = arg.arg
        if arg.annotation is not None:
            piece += ": " + ast.unparse(arg.annotation)
        if default is not None:
            piece += "=" + ast.unparse(default)
        args.append(piece)

    if node.args.kwarg is not None:
        piece = "**" + node.args.kwarg.arg
        if node.args.kwarg.annotation is not None:
            piece += ": " + ast.unparse(node.args.kwarg.annotation)
        args.append(piece)

    ret = ""
    if node.returns is not None:
        ret = " -> " + ast.unparse(node.returns)

    return f"{node.name}({', '.join(args)}){ret}"


def extract_required_functions(
    path: Path,
    names: list[str],
) -> tuple[list[dict[str, Any]], str]:
    text, _, funcs = parse_functions(path)
    records = []
    chunks = []

    for name in names:
        node = funcs.get(name)
        if node is None:
            raise RuntimeError(f"Required function {name!r} not found in {path}.")
        assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        segment = source_segment(text, node)
        signature = function_signature_from_ast(node)
        records.append(
            {
                "relative_path": str(path.relative_to(ROOT)),
                "function": name,
                "line_start": int(node.lineno),
                "line_end": int(getattr(node, "end_lineno", node.lineno)),
                "signature": signature,
                "source_sha256": sha256_text(segment),
            }
        )
        chunks.append(
            f"### {path.relative_to(ROOT)} :: {name}\n"
            f"### signature: {signature}\n"
            f"### source_sha256: {sha256_text(segment)}\n\n"
            f"{segment}\n"
        )

    return records, "\n".join(chunks)


def extract_runner_call_context(path: Path) -> tuple[list[dict[str, Any]], str]:
    text = path.read_text(encoding="utf-8", errors="strict")
    lines = text.splitlines()
    records = []
    chunks = []
    seen_ranges = set()

    for lineno, line in enumerate(lines, start=1):
        if not any(pattern in line for pattern in RUNNER_KEY_PATTERNS):
            continue

        start = max(1, lineno - 18)
        end = min(len(lines), lineno + 22)
        key = (start, end)
        if key in seen_ranges:
            continue
        seen_ranges.add(key)

        segment = "\n".join(
            f"{i}: {lines[i-1]}"
            for i in range(start, end + 1)
        ) + "\n"
        records.append(
            {
                "relative_path": str(path.relative_to(ROOT)),
                "match_line": lineno,
                "matched_text": line.strip(),
                "context_start": start,
                "context_end": end,
                "context_sha256": sha256_text(segment),
            }
        )
        chunks.append(
            f"### runner context lines {start}-{end}, match at {lineno}\n"
            f"{segment}\n"
        )

    if not records:
        raise RuntimeError("No IBS/aggregation call contexts found in frozen TARGET runner.")

    return records, "\n".join(chunks)


def characterize_fold_ibs(fm: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    rows = []
    for model in models:
        part = fm[fm["model_id"].astype(str) == model].copy()
        ibs_numeric = pd.to_numeric(part["ibs"], errors="coerce")
        statuses = (
            part["ibs_status"].astype(str).value_counts(dropna=False).to_dict()
        )
        rows.append(
            {
                "model_id": model,
                "fold_rows": int(len(part)),
                "finite_fold_ibs": int(np.isfinite(ibs_numeric).sum()),
                "nonfinite_fold_ibs": int((~np.isfinite(ibs_numeric)).sum()),
                "ibs_status_counts_json": json.dumps(
                    {str(k): int(v) for k, v in statuses.items()},
                    sort_keys=True,
                ),
            }
        )
    return pd.DataFrame(rows)


def write_source_extracts(
    function_chunks: list[str],
    runner_context: str,
) -> None:
    text = [
        "Paper 6 05h5b - exact frozen IBS mechanics source extracts",
        "",
        "These excerpts are copied verbatim from the current frozen source files",
        "and are hash-locked by this technical contract.",
        "",
    ]
    text.extend(function_chunks)
    text.append("\n### TARGET runner call/aggregation contexts\n")
    text.append(runner_context)

    (OUT_DIR / "frozen_ibs_mechanics_source_extract.txt").write_text(
        "\n".join(text),
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    print("=" * 118)
    print("Paper 6 - 05h5b freeze exact TARGET IBS/fold mechanics")
    print("=" * 118)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {ROOT}")
    print(f"Output directory: {OUT_DIR}")
    print()
    print("Execution scope:")
    print("  Kaplan-Meier reference fitted: NO")
    print("  new IBS calculated: NO")
    print("  model fitting / retuning: NO")
    print("  TARGET86 endpoint read for structural verification: YES")
    print("  GSE21257/GSE39055 outcomes read: NO")
    print("  exact existing fold structure frozen: YES")
    print("  exact existing IBS helper source frozen: YES")
    print("  exact TARGET runner IBS call context frozen: YES")
    print()

    if OUT_DIR.exists():
        raise RuntimeError(
            f"05h5b output already exists; refusing overwrite: {OUT_DIR}"
        )

    verify_required_files()
    h0_block = verify_h0()
    h5a_manifest = verify_05h5a()
    f3f_summary, f3f_audit = verify_f3f_summary()

    OUT_DIR.mkdir(parents=True, exist_ok=False)

    ep = load_endpoint()
    fold_structure, split_meta = load_and_validate_splits(ep)
    fm, models = load_and_validate_fold_metrics(ep, fold_structure)
    oof = load_and_validate_oof(ep, fold_structure, models, split_meta)
    ms = load_and_validate_model_summary(fm, models)

    fold_structure.to_csv(
        OUT_DIR / "frozen_outer_fold_structure.tsv",
        sep="\t",
        index=False,
    )

    ibs_profile = characterize_fold_ibs(fm, models)
    ibs_profile.to_csv(
        OUT_DIR / "existing_fold_ibs_availability.tsv",
        sep="\t",
        index=False,
    )

    m04c_records, m04c_extract = extract_required_functions(M04C, M04C_FUNCTIONS)
    runner_records, runner_func_extract = extract_required_functions(
        RUNNER, RUNNER_FUNCTIONS
    )
    call_records, runner_context = extract_runner_call_context(RUNNER)

    pd.DataFrame(m04c_records + runner_records).to_csv(
        OUT_DIR / "frozen_function_signatures_and_hashes.csv",
        index=False,
    )
    pd.DataFrame(call_records).to_csv(
        OUT_DIR / "target_runner_call_context_index.csv",
        index=False,
    )
    write_source_extracts(
        [m04c_extract, runner_func_extract],
        runner_context,
    )

    # Freeze exact current source/input identities.
    source_files = {
        "05h0_contract": H0_JSON,
        "05h5a_manifest": H5A_MANIFEST,
        "05f3f_summary": F3F_SUMMARY,
        "05f3f_input_audit": F3F_INPUT_AUDIT,
        "TARGET86_endpoint": ENDPOINT,
        "TARGET_split_registry": SPLIT_REGISTRY,
        "TARGET_fold_metrics": FOLD_METRICS,
        "TARGET_model_summary": MODEL_SUMMARY,
        "TARGET_oof_predictions": OOF_PRED,
        "TARGET_runner_v3_bytesafe": RUNNER,
        "frozen_04c_metric_implementation": M04C,
    }

    print("TARGET86 identity / outcome counts: PASS")
    print(f"  n: {len(ep)}/86")
    print(f"  events: {int(ep['os_event'].sum())}/29")
    print(f"  censored: {int((ep['os_event'] == 0).sum())}/57")
    zero = ep[ep["os_time_days"] == 0].iloc[0]
    print(
        f"  retained censored time-zero case: "
        f"{zero['sample_id']} [execution_index={int(zero['execution_index_86'])}]"
    )
    print()

    print("Frozen outer-fold structure: PASS")
    print(f"  repeats: {len(split_meta['repeat_labels'])}/20")
    print(f"  outer folds: {len(split_meta['outer_fold_labels'])}/5")
    print(
        f"  held-out role resolved from exact combinatorics: "
        f"{split_meta['heldout_role_resolved']}"
    )
    print(
        f"  training role resolved: {split_meta['training_role_resolved']}"
    )
    print("  100/100 train/test partitions cover 86 patients without overlap: YES")
    print()

    print("Existing TARGET result structure: PASS")
    print(f"  model IDs ({len(models)}): {', '.join(models)}")
    print(f"  fold-metric rows: {len(fm)}/900")
    print(f"  OOF prediction rows: {len(oof)}/15480")
    print(f"  model-summary rows: {len(ms)}/9")
    print("  OOF held-out fold identity matches split registry: YES")
    print()

    print("Existing fold IBS availability:")
    print(ibs_profile.to_string(index=False))
    print()

    print("Frozen IBS helper signatures:")
    for row in m04c_records + runner_records:
        print(
            f"  {row['function']}: {row['signature']} "
            f"[lines {row['line_start']}-{row['line_end']}]"
        )
    print()

    print("TARGET runner IBS/aggregation call sites:")
    for row in call_records:
        print(
            f"  line {row['match_line']}: {row['matched_text']}"
        )
    print()

    contract = {
        "script_version": SCRIPT_VERSION,
        "status": "PASS_READY_FOR_05H5C_KM_IBS_ESTIMATION",
        "analysis_role": "TECHNICAL MECHANICS FREEZE ONLY",
        "scientific_km_reference_computed": False,
        "scientific_ibs_reference_computed": False,
        "05h0_target_km_ibs_reference": h0_block,
        "TARGET86": {
            "n": EXPECTED_N,
            "events": EXPECTED_EVENTS,
            "censored": EXPECTED_CENSORED,
            "endpoint_columns": list(ep.columns),
            "zero_time_n": int((ep["os_time_days"] == 0).sum()),
            "zero_time_is_censored": bool(
                int(ep.loc[ep["os_time_days"] == 0, "os_event"].iloc[0]) == 0
            ),
        },
        "outer_cv": {
            "repeat_labels": split_meta["repeat_labels"],
            "outer_fold_labels": split_meta["outer_fold_labels"],
            "heldout_role": split_meta["heldout_role_resolved"],
            "training_role": split_meta["training_role_resolved"],
            "n_partitions": len(fold_structure),
            "fold_structure_file": "frozen_outer_fold_structure.tsv",
            "fold_structure_sha256": sha256_file(
                OUT_DIR / "frozen_outer_fold_structure.tsv"
            ),
        },
        "existing_result_structure": {
            "model_ids": models,
            "fold_metric_rows": len(fm),
            "oof_prediction_rows": len(oof),
            "model_summary_rows": len(ms),
            "ibs_profile_file": "existing_fold_ibs_availability.tsv",
        },
        "source_files": {
            role: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for role, path in source_files.items()
        },
        "frozen_functions": m04c_records + runner_records,
        "runner_call_context_index": call_records,
        "source_extract_file": "frozen_ibs_mechanics_source_extract.txt",
        "source_extract_sha256": sha256_file(
            OUT_DIR / "frozen_ibs_mechanics_source_extract.txt"
        ),
        "next_stage_rule": (
            "05h5c must use these exact TARGET86 memberships and the exact frozen "
            "04c IBS mechanics. It may fit only the 05h0-authorized no-covariate "
            "Kaplan-Meier survival curve within each frozen outer-training partition."
        ),
    }
    write_json(OUT_DIR / "target_ibs_mechanics_contract.json", contract)

    outputs_before_manifest = sorted(
        p for p in OUT_DIR.iterdir()
        if p.is_file() and p.name != "freeze_manifest.json"
    )
    manifest = {
        "script_version": SCRIPT_VERSION,
        "script_relative_path": str(Path(__file__).resolve().relative_to(ROOT)),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "status": "PASS_READY_FOR_05H5C_KM_IBS_ESTIMATION",
        "outputs": [
            {
                "file": p.name,
                "sha256": sha256_file(p),
                "bytes": p.stat().st_size,
            }
            for p in outputs_before_manifest
        ],
    }
    write_json(OUT_DIR / "freeze_manifest.json", manifest)

    print("=" * 118)
    print("05h5b TARGET IBS mechanics freeze: PASS")
    print("=" * 118)
    print("Scientific KM reference calculated: NO")
    print("Scientific IBS reference calculated: NO")
    print("Ready for 05h5c exact KM-null estimation: YES")
    print()
    print("Created:")
    for p in sorted(OUT_DIR.iterdir()):
        if p.is_file():
            print(f"  {p}")


if __name__ == "__main__":
    main()
