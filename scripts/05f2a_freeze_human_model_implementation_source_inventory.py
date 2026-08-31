#!/usr/bin/env python3
"""
Paper 6 - freeze exact implementation-source inventory for the post-HOLD
descriptive TARGET model matrix.

Stage 05f2a follows the PASS 05f1c v2 clinical-schema/evaluation freeze.

WHY THIS STAGE EXISTS
---------------------
Before any TARGET endpoint value can be opened, Paper 6 must bind every allowed
human model to concrete, already-existing implementation code rather than
reconstructing algorithms from memory.

This stage therefore freezes:
- exact 05c neural implementation source + scientific constants/functions;
- exact existing 03b source-gate implementation candidate(s);
- exact existing 04c/04e premise/classical implementation candidate(s);
- exact model-to-implementation SOURCE bindings for T0-T2/N0-N5;
- exact source-artifact materialization plan for the next stage.

This is an IMPLEMENTATION-SOURCE freeze, not yet the final executable human
model runner. 05f2b must use only these frozen implementation sources, resolve
the exact callable adapters, fit/materialize DOG2 source-only artifacts, and
freeze the final runner before TARGET endpoints may be opened.

This script does NOT:
- read TARGET OS/event/time values;
- read secondary TARGET endpoints;
- generate TARGET survival splits;
- fit TARGET models;
- fit DOG2 source models;
- read GSE21257/GSE39055 outcomes;
- use GPU.

No CLI arguments.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


SCRIPT_VERSION = (
    "05f2a-freeze-human-model-implementation-source-inventory-v1-no-cli"
)
CONTRACT_VERSION = (
    "paper6-posthold-human-model-implementation-source-inventory-v1"
)

ROOT = Path(__file__).resolve().parents[1]

F1A_DIR = ROOT / "results" / "human_posthold_descriptive" / "05f1a"
F1A_CONTRACT = F1A_DIR / "outcome_free_TARGET_representation_contract.json"
F1A_HALLMARK = F1A_DIR / "TARGET_hallmark50_gene_map.tsv"
F1A_MODELS = F1A_DIR / "TARGET_frozen_model_registry.tsv"

EXPECTED_F1A_CONTRACT_SHA256 = (
    "f0d5296d4df5e8dcb3b677127d2c3b5474791fb2cbaeb207e094b54fb44d1434"
)

F1B_DIR = ROOT / "results" / "human_posthold_descriptive" / "05f1b"
F1B_MANIFEST = F1B_DIR / "raw_aligned_matrix_manifest.tsv"
F1B_DOG_MATRIX = (
    F1B_DIR / "matrices" / "DOG2_raw_aligned_11815genes.npz"
)
F1B_TARGET_MATRIX = (
    F1B_DIR / "matrices" / "TARGET_OS_raw_aligned_11815genes.npz"
)

EXPECTED_F1B_MANIFEST_SHA256 = (
    "d5e267da52955332f845655325de7a66fcf4fbc63fcde9bfac8669c3a5f46b63"
)
EXPECTED_F1B_DOG_MATRIX_SHA256 = (
    "af7b8b65f9485df6a7ba3d0061bc12c0023719e4114d7f10fb3d731e918e3875"
)
EXPECTED_F1B_TARGET_MATRIX_SHA256 = (
    "7aca20d98731a24d90882f2042b1aa7c7aefc0095a250d4ed42c3502cc3c7b19"
)

F1C_DIR = ROOT / "results" / "human_posthold_descriptive" / "05f1c_v2"
F1C_ENDPOINT_SCHEMA = F1C_DIR / "TARGET_OS_endpoint_schema.json"
F1C_RESAMPLING = F1C_DIR / "TARGET_resampling_and_CI_contract.json"
F1C_OPENING = F1C_DIR / "TARGET_outcome_opening_contract.json"
F1C_SUMMARY = F1C_DIR / "summary.json"

EXPECTED_F1C_STATUS = (
    "PASS_TARGET_CLINICAL_SCHEMA_AND_EVALUATION_MECHANICS_FROZEN_OUTCOME_CLOSED"
)
EXPECTED_F1C_ENDPOINT_SCHEMA_SHA256 = (
    "6c827bb13d3a44a651d012b5a51a0b96ffb45682cd73ab545da2cb9edda79ce2"
)
EXPECTED_F1C_RESAMPLING_SHA256 = (
    "0d6d004b2ea7007c0fde853ff10d97e0df734ea4571f1d55bc2efcfef9367e68"
)

C05_SCRIPT = (
    ROOT / "scripts" / "05c_run_closed_synthetic_transfer_model_matrix.py"
)
C05_DIR = ROOT / "results" / "simulation_model_matrix" / "05c"
C05_IMPL_CONTRACT = C05_DIR / "model_implementation_contract.json"
C05_SUMMARY = C05_DIR / "summary.json"

EXPECTED_C05_IMPL_CONTRACT_SHA256 = (
    "b3c4d14a0db2b999d211d2f173d89f437772b7948c1fc6eee53908b78afa782f"
)
EXPECTED_C05_STATUS = (
    "PASS_CLOSED_SYNTHETIC_TRANSFER_MODEL_MATRIX_COMPLETE"
)

A3_DIR = ROOT / "results" / "source_gate_protocol" / "03a"
A3_SUMMARY = A3_DIR / "summary.json"

B3_DIR = ROOT / "results" / "source_gate" / "03b"
B3_SUMMARY = B3_DIR / "summary.json"

CLASSICAL_STAGE_PREFIXES = ["03b", "04c", "04d", "04e"]

OUT_DIR = ROOT / "results" / "human_posthold_descriptive" / "05f2a"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_INVENTORY = OUT_DIR / "implementation_source_inventory.tsv"
FUNCTION_INVENTORY = OUT_DIR / "scientific_function_inventory.tsv"
CONSTANT_INVENTORY = OUT_DIR / "scientific_constant_inventory.tsv"
MODEL_BINDINGS = OUT_DIR / "TARGET_model_implementation_source_bindings.tsv"
SOURCE_ARTIFACT_PLAN = OUT_DIR / "DOG2_source_artifact_materialization_plan.tsv"
CONTRACT_JSON = OUT_DIR / "implementation_source_inventory_contract.json"
SUMMARY_JSON = OUT_DIR / "summary.json"

REQUIRED_05C_FUNCTIONS = [
    "standardize_train_test",
    "batched_cox_nll",
    "per_replicate_cox_nll",
    "train_linear_cox",
    "train_source_mlp",
    "freeze_module",
    "train_A1_head",
    "predict_A1",
    "train_A3",
    "predict_A3",
]

A2_FUNCTION_PATTERNS = [r"^train_A2$", r"^train_A2_", r"^fit_A2"]
A4_FUNCTION_PATTERNS = [r"^train_A4$", r"^train_A4_", r"^fit_A4"]
A0_FUNCTION_PATTERNS = [r"^train_A0$", r"^train_A0_", r"^fit_A0"]

REQUIRED_05C_CONSTANTS = [
    "MODEL_BASE_SEED",
    "TARGET_B0_ALPHA",
    "LINEAR_EPOCHS_TARGET",
    "LINEAR_LR_TARGET",
    "HIDDEN_DIM",
    "LATENT_DIM",
    "SOURCE_MLP_EPOCHS",
    "SOURCE_MLP_LR",
    "SOURCE_MLP_WEIGHT_DECAY",
    "A1_EPOCHS",
    "A1_LR",
    "A1_HEAD_L2_TO_SOURCE",
    "GRAD_CLIP_NORM",
    "RISK_CLIP",
]

CLASSICAL_SCRIPT_TOKENS = {
    "03b": ["source", "prognostic", "gate"],
    "04c": ["premise", "transfer", "evaluate", "run"],
    "04d": ["premise", "diagnostic", "audit"],
    "04e": ["alpha", "audit", "premise"],
}


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
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import implementation source: {path}")

    module = importlib.util.module_from_spec(spec)
    prior = sys.modules.get(name)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        if prior is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = prior
        raise
    return module


def source_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def parse_ast(path: Path) -> ast.Module:
    return ast.parse(source_text(path), filename=str(path))


def function_source_hash(path: Path, function_name: str) -> Tuple[str, int, int]:
    text = source_text(path)
    lines = text.splitlines(keepends=True)
    tree = ast.parse(text, filename=str(path))

    matches = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"{path.name}: expected one top-level function {function_name!r}, "
            f"observed {len(matches)}."
        )

    node = matches[0]
    start = int(node.lineno)
    end = int(node.end_lineno)
    snippet = "".join(lines[start - 1:end])
    return (
        hashlib.sha256(snippet.encode("utf-8")).hexdigest(),
        start,
        end,
    )


def literal_upper_constants(path: Path) -> Dict[str, Any]:
    tree = parse_ast(path)
    out: Dict[str, Any] = {}

    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
            value_node = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value_node = node.value
        else:
            continue

        if value_node is None:
            continue

        for target in targets:
            if not isinstance(target, ast.Name):
                continue
            name = target.id
            if not name.isupper():
                continue
            try:
                value = ast.literal_eval(value_node)
            except Exception:
                continue

            if isinstance(value, (str, int, float, bool, type(None))):
                out[name] = value
            elif isinstance(value, (list, tuple)) and all(
                isinstance(x, (str, int, float, bool, type(None)))
                for x in value
            ):
                out[name] = list(value)

    return out


def top_level_function_names(path: Path) -> List[str]:
    return [
        node.name
        for node in parse_ast(path).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def choose_pattern_function(
    function_names: Sequence[str],
    patterns: Sequence[str],
    *,
    role: str,
) -> str:
    for pattern in patterns:
        matches = sorted(
            name for name in function_names
            if re.search(pattern, name)
        )
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise RuntimeError(
                f"05c has ambiguous {role} functions for pattern "
                f"{pattern!r}: {matches}"
            )

    raise RuntimeError(
        f"05c lacks a unique callable for {role}; "
        f"available functions={function_names}"
    )


def verify_human_firewall() -> Dict[str, str]:
    for path in [
        F1A_CONTRACT,
        F1A_HALLMARK,
        F1A_MODELS,
        F1B_MANIFEST,
        F1B_DOG_MATRIX,
        F1B_TARGET_MATRIX,
        F1C_ENDPOINT_SCHEMA,
        F1C_RESAMPLING,
        F1C_OPENING,
        F1C_SUMMARY,
    ]:
        require_file(path)

    if sha256_file(F1A_CONTRACT) != EXPECTED_F1A_CONTRACT_SHA256:
        raise RuntimeError("05f1a contract hash changed.")
    if sha256_file(F1B_MANIFEST) != EXPECTED_F1B_MANIFEST_SHA256:
        raise RuntimeError("05f1b manifest hash changed.")
    if sha256_file(F1B_DOG_MATRIX) != EXPECTED_F1B_DOG_MATRIX_SHA256:
        raise RuntimeError("05f1b DOG2 matrix hash changed.")
    if sha256_file(F1B_TARGET_MATRIX) != EXPECTED_F1B_TARGET_MATRIX_SHA256:
        raise RuntimeError("05f1b TARGET matrix hash changed.")
    if sha256_file(F1C_ENDPOINT_SCHEMA) != EXPECTED_F1C_ENDPOINT_SCHEMA_SHA256:
        raise RuntimeError("05f1c endpoint schema hash changed.")
    if sha256_file(F1C_RESAMPLING) != EXPECTED_F1C_RESAMPLING_SHA256:
        raise RuntimeError("05f1c resampling contract hash changed.")

    f1c = read_json(F1C_SUMMARY)

    if clean(f1c.get("scientific_status")) != EXPECTED_F1C_STATUS:
        raise RuntimeError("05f1c is not in expected frozen PASS state.")
    if f1c.get("TARGET_OS_event_values_read") is not False:
        raise RuntimeError("05f1c indicates TARGET event access.")
    if f1c.get("TARGET_OS_time_values_read") is not False:
        raise RuntimeError("05f1c indicates TARGET time access.")
    if f1c.get("TARGET_endpoint_values_may_be_read_now") is not False:
        raise RuntimeError("05f1c unexpectedly authorizes endpoint opening.")

    return {
        "05f1a_contract_sha256": sha256_file(F1A_CONTRACT),
        "05f1b_manifest_sha256": sha256_file(F1B_MANIFEST),
        "05f1b_DOG2_matrix_sha256": sha256_file(F1B_DOG_MATRIX),
        "05f1b_TARGET_matrix_sha256": sha256_file(F1B_TARGET_MATRIX),
        "05f1c_endpoint_schema_sha256": sha256_file(F1C_ENDPOINT_SCHEMA),
        "05f1c_resampling_contract_sha256": sha256_file(F1C_RESAMPLING),
        "05f1c_outcome_opening_contract_sha256": sha256_file(F1C_OPENING),
        "05f1c_summary_sha256": sha256_file(F1C_SUMMARY),
    }


def verify_05c() -> Dict[str, Any]:
    for path in [C05_SCRIPT, C05_IMPL_CONTRACT, C05_SUMMARY]:
        require_file(path)

    if sha256_file(C05_IMPL_CONTRACT) != EXPECTED_C05_IMPL_CONTRACT_SHA256:
        raise RuntimeError("05c implementation contract differs from frozen hash.")

    impl = read_json(C05_IMPL_CONTRACT)
    summary = read_json(C05_SUMMARY)

    if clean(summary.get("scientific_status")) != EXPECTED_C05_STATUS:
        raise RuntimeError("05c is not in expected frozen PASS state.")

    expected_script_hash = clean(impl.get("script_sha256"))
    observed_script_hash = sha256_file(C05_SCRIPT)

    if not expected_script_hash:
        raise RuntimeError("05c implementation contract lacks script_sha256.")
    if observed_script_hash != expected_script_hash:
        raise RuntimeError("05c script differs from frozen implementation contract.")

    module = load_module(C05_SCRIPT, "paper6_frozen_05c_for_05f2a")
    function_names = top_level_function_names(C05_SCRIPT)

    for name in REQUIRED_05C_FUNCTIONS:
        if not hasattr(module, name) or name not in function_names:
            raise RuntimeError(
                f"Frozen 05c lacks required top-level callable {name!r}."
            )

    a0 = choose_pattern_function(
        function_names, A0_FUNCTION_PATTERNS, role="A0 target-scratch branch"
    )
    a2 = choose_pattern_function(
        function_names, A2_FUNCTION_PATTERNS, role="A2 low-rank adapter branch"
    )
    a4 = choose_pattern_function(
        function_names, A4_FUNCTION_PATTERNS, role="A4 full-fine-tune branch"
    )

    constants = literal_upper_constants(C05_SCRIPT)
    missing = [name for name in REQUIRED_05C_CONSTANTS if name not in constants]

    for name in list(missing):
        if hasattr(module, name):
            value = getattr(module, name)
            if isinstance(value, (str, int, float, bool)):
                constants[name] = value
                missing.remove(name)

    if missing:
        raise RuntimeError(
            "Frozen 05c lacks resolvable required constants: "
            f"{missing}"
        )

    return {
        "function_names": function_names,
        "constants": constants,
        "A0_function": a0,
        "A2_function": a2,
        "A4_function": a4,
        "script_sha256": observed_script_hash,
        "implementation_contract_sha256": sha256_file(C05_IMPL_CONTRACT),
        "summary_sha256": sha256_file(C05_SUMMARY),
    }


def stage_script_candidates(prefix: str) -> List[Path]:
    return sorted(
        path for path in (ROOT / "scripts").glob(f"{prefix}*.py")
        if path.is_file()
    )


def rank_classical_candidate(path: Path, prefix: str) -> Tuple[int, int, str]:
    filename = path.name.lower()
    text = source_text(path).lower()
    score = 0

    for token in CLASSICAL_SCRIPT_TOKENS[prefix]:
        if token in filename:
            score += 10
        if token in text[:10000]:
            score += 1

    if any(token in filename for token in ["run", "evaluate", "gate", "audit"]):
        score += 3
    if any(token in filename for token in ["probe", "freeze", "protocol"]):
        score -= 2

    return score, -len(filename), filename


def choose_classical_script(prefix: str) -> Path:
    candidates = stage_script_candidates(prefix)
    if not candidates:
        raise RuntimeError(
            f"No local script found for frozen classical stage {prefix}."
        )

    ranked = sorted(
        [(rank_classical_candidate(path, prefix), path) for path in candidates],
        reverse=True,
    )
    best_score = ranked[0][0][0]
    best = [path for rank, path in ranked if rank[0] == best_score]

    if len(best) != 1:
        raise RuntimeError(
            f"Ambiguous implementation-source discovery for stage {prefix}. "
            f"Best candidates={[p.name for p in best]}. "
            "Resolve pre-outcome; do not choose after TARGET outcomes."
        )

    return best[0]


def function_inventory_rows(path: Path, *, source_role: str) -> List[Dict[str, Any]]:
    rows = []
    for name in top_level_function_names(path):
        digest, start, end = function_source_hash(path, name)
        lower = name.lower()
        relevant = any(
            token in lower
            for token in [
                "cox", "surv", "risk", "ridge", "elastic", "alpha",
                "train", "fit", "predict", "standard", "scale",
                "coral", "residual", "uno", "ibs", "split",
            ]
        )
        rows.append({
            "source_role": source_role,
            "script_path": str(path.relative_to(ROOT)),
            "script_sha256": sha256_file(path),
            "function_name": name,
            "function_sha256": digest,
            "start_line": start,
            "end_line": end,
            "scientifically_relevant_by_name": relevant,
        })
    return rows


def constant_inventory_rows(path: Path, *, source_role: str) -> List[Dict[str, Any]]:
    rows = []
    for name, value in sorted(literal_upper_constants(path).items()):
        lower = name.lower()
        relevant = any(
            token in lower
            for token in [
                "alpha", "lambda", "penalty", "epoch", "lr", "seed",
                "fold", "split", "hidden", "latent", "dropout",
                "clip", "threshold", "grid",
            ]
        )
        rows.append({
            "source_role": source_role,
            "script_path": str(path.relative_to(ROOT)),
            "script_sha256": sha256_file(path),
            "constant_name": name,
            "constant_value_json": json.dumps(value, sort_keys=True),
            "scientifically_relevant_by_name": relevant,
        })
    return rows


def main() -> None:
    started = now_utc()

    print("=" * 120)
    print("Paper 6 - freeze human model implementation-source inventory")
    print("=" * 120)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Contract version: {CONTRACT_VERSION}")
    print()
    print("Safety / scope:")
    print("  05f1a/05f1b/05f1c frozen human firewall verified: YES")
    print("  Existing implementation SOURCE files read: YES")
    print("  TARGET OS/event/time values read: NO")
    print("  TARGET secondary endpoint values read: NO")
    print("  TARGET survival splits generated: NO")
    print("  TARGET models fit: NO")
    print("  DOG2 source models fit: NO")
    print("  GSE21257/GSE39055 outcomes read: NO")
    print("  GPU execution: NO")
    print()

    firewall = verify_human_firewall()
    c05 = verify_05c()

    for path in [A3_SUMMARY, B3_SUMMARY]:
        require_file(path)

    classical_scripts = {
        prefix: choose_classical_script(prefix)
        for prefix in CLASSICAL_STAGE_PREFIXES
    }

    inventory_rows = [{
        "source_role": "NEURAL_05C",
        "stage": "05c",
        "script_path": str(C05_SCRIPT.relative_to(ROOT)),
        "script_sha256": c05["script_sha256"],
        "selection_method": "exact frozen 05c implementation contract",
    }]

    for prefix, path in classical_scripts.items():
        inventory_rows.append({
            "source_role": f"CLASSICAL_{prefix.upper()}",
            "stage": prefix,
            "script_path": str(path.relative_to(ROOT)),
            "script_sha256": sha256_file(path),
            "selection_method": "pre-outcome deterministic stage-prefix/token ranking",
        })

    pd.DataFrame(inventory_rows).to_csv(
        SOURCE_INVENTORY, sep="\t", index=False
    )

    function_rows = function_inventory_rows(C05_SCRIPT, source_role="NEURAL_05C")
    constant_rows = constant_inventory_rows(C05_SCRIPT, source_role="NEURAL_05C")

    for prefix, path in classical_scripts.items():
        role = f"CLASSICAL_{prefix.upper()}"
        function_rows.extend(function_inventory_rows(path, source_role=role))
        constant_rows.extend(constant_inventory_rows(path, source_role=role))

    function_df = pd.DataFrame(function_rows)
    constant_df = pd.DataFrame(constant_rows)

    function_df.to_csv(FUNCTION_INVENTORY, sep="\t", index=False)
    constant_df.to_csv(CONSTANT_INVENTORY, sep="\t", index=False)

    bindings = [
        {
            "model_id": "T0",
            "implementation_family": "CLASSICAL_TARGET_RIDGE",
            "primary_source_stage": "04c/04e",
            "frozen_source_scripts": (
                f"{classical_scripts['04c'].relative_to(ROOT)};"
                f"{classical_scripts['04e'].relative_to(ROOT)}"
            ),
            "exact_callable_status": "TO_BE_BOUND_IN_05F2B_FROM_FROZEN_INVENTORY",
            "post_outcome_implementation_change_allowed": False,
        },
        {
            "model_id": "T1",
            "implementation_family": "CLASSICAL_DOG_HALLMARK_RIDGE_ZERO_SHOT",
            "primary_source_stage": "03b",
            "frozen_source_scripts": str(classical_scripts["03b"].relative_to(ROOT)),
            "exact_callable_status": "TO_BE_BOUND_IN_05F2B_FROM_FROZEN_INVENTORY",
            "post_outcome_implementation_change_allowed": False,
        },
        {
            "model_id": "T2",
            "implementation_family": "CLASSICAL_DOG_PLUS_TARGET_RESIDUAL",
            "primary_source_stage": "04c/04d",
            "frozen_source_scripts": (
                f"{classical_scripts['04c'].relative_to(ROOT)};"
                f"{classical_scripts['04d'].relative_to(ROOT)}"
            ),
            "exact_callable_status": "TO_BE_BOUND_IN_05F2B_FROM_FROZEN_INVENTORY",
            "post_outcome_implementation_change_allowed": False,
        },
        {
            "model_id": "N0",
            "implementation_family": "NEURAL_SOURCE_ZERO_SHOT",
            "primary_source_stage": "05c",
            "frozen_source_scripts": str(C05_SCRIPT.relative_to(ROOT)),
            "exact_callable_status": "train_source_mlp",
            "post_outcome_implementation_change_allowed": False,
        },
        {
            "model_id": "N1",
            "implementation_family": "NEURAL_A1_SOURCE_CENTERED_HEAD",
            "primary_source_stage": "05c",
            "frozen_source_scripts": str(C05_SCRIPT.relative_to(ROOT)),
            "exact_callable_status": "train_A1_head",
            "post_outcome_implementation_change_allowed": False,
        },
        {
            "model_id": "N2",
            "implementation_family": "NEURAL_A1_FREE_HEAD",
            "primary_source_stage": "05c+controlled_05e",
            "frozen_source_scripts": str(C05_SCRIPT.relative_to(ROOT)),
            "exact_callable_status": (
                "exact train_A1_head copy; ONLY A1_HEAD_L2_TO_SOURCE=0"
            ),
            "post_outcome_implementation_change_allowed": False,
        },
        {
            "model_id": "N3",
            "implementation_family": "NEURAL_A2_LOW_RANK_ADAPTER",
            "primary_source_stage": "05c",
            "frozen_source_scripts": str(C05_SCRIPT.relative_to(ROOT)),
            "exact_callable_status": c05["A2_function"],
            "post_outcome_implementation_change_allowed": False,
        },
        {
            "model_id": "N4",
            "implementation_family": "NEURAL_A0_TARGET_SCRATCH",
            "primary_source_stage": "05c",
            "frozen_source_scripts": str(C05_SCRIPT.relative_to(ROOT)),
            "exact_callable_status": c05["A0_function"],
            "post_outcome_implementation_change_allowed": False,
        },
        {
            "model_id": "N5",
            "implementation_family": "NEURAL_A4_FULL_FINE_TUNE",
            "primary_source_stage": "05c",
            "frozen_source_scripts": str(C05_SCRIPT.relative_to(ROOT)),
            "exact_callable_status": c05["A4_function"],
            "post_outcome_implementation_change_allowed": False,
        },
    ]

    pd.DataFrame(bindings).to_csv(
        MODEL_BINDINGS, sep="\t", index=False
    )

    source_plan = pd.DataFrame([
        {
            "artifact_id": "DOG2_FULL_SOURCE_HALLMARK50_REPRESENTATION",
            "input": str(F1B_DOG_MATRIX.relative_to(ROOT)),
            "outcome_access": "NO",
            "implementation_source": str(F1A_HALLMARK.relative_to(ROOT)),
            "materialization_stage": "05f2b",
            "rule": (
                "full-DOG2 gene scaler -> Hallmark unweighted mean -> "
                "full-DOG2 module scaler, all using frozen 05f1a rules"
            ),
        },
        {
            "artifact_id": "DOG2_FULL_SOURCE_CLASSICAL_RIDGE",
            "input": "DOG2 Hallmark50 + frozen DOG2 OS",
            "outcome_access": "DOG2_OS_ONLY_ALLOWED",
            "implementation_source": str(
                classical_scripts["03b"].relative_to(ROOT)
            ),
            "materialization_stage": "05f2b",
            "rule": (
                "exact source Hallmark ridge implementation/hyperparameter "
                "policy bound from frozen 03b source"
            ),
        },
        {
            "artifact_id": "DOG2_FULL_SOURCE_NEURAL_MLP",
            "input": "DOG2 Hallmark50 + frozen DOG2 OS",
            "outcome_access": "DOG2_OS_ONLY_ALLOWED",
            "implementation_source": str(C05_SCRIPT.relative_to(ROOT)),
            "materialization_stage": "05f2b",
            "rule": (
                "exact 05c train_source_mlp with frozen constants and "
                "deterministic source seed"
            ),
        },
    ])
    source_plan.to_csv(SOURCE_ARTIFACT_PLAN, sep="\t", index=False)

    exact_05c_functions = sorted(set(
        REQUIRED_05C_FUNCTIONS
        + [c05["A0_function"], c05["A2_function"], c05["A4_function"]]
    ))

    c05_function_hashes = {}
    for name in exact_05c_functions:
        digest, start, end = function_source_hash(C05_SCRIPT, name)
        c05_function_hashes[name] = {
            "sha256": digest,
            "start_line": start,
            "end_line": end,
        }

    contract = {
        "script_version": SCRIPT_VERSION,
        "contract_version": CONTRACT_VERSION,
        "status": "PASS",
        "scientific_status": (
            "PASS_HUMAN_MODEL_IMPLEMENTATION_SOURCE_INVENTORY_FROZEN_OUTCOME_CLOSED"
        ),
        "created_utc": now_utc(),
        "human_firewall_hashes": firewall,
        "neural_05c": {
            "script_path": str(C05_SCRIPT.relative_to(ROOT)),
            "script_sha256": c05["script_sha256"],
            "implementation_contract_sha256": c05["implementation_contract_sha256"],
            "summary_sha256": c05["summary_sha256"],
            "exact_function_hashes": c05_function_hashes,
            "scientific_constants": {
                name: c05["constants"][name]
                for name in REQUIRED_05C_CONSTANTS
            },
            "A0_callable": c05["A0_function"],
            "A2_callable": c05["A2_function"],
            "A4_callable": c05["A4_function"],
            "A1_free_head_rule": (
                "same train_A1_head implementation/init/epochs/lr/source encoder; "
                "ONLY L2-to-source-head coefficient=0"
            ),
        },
        "classical_sources": {
            prefix: {
                "script_path": str(path.relative_to(ROOT)),
                "script_sha256": sha256_file(path),
                "selection_rule": "pre-outcome deterministic stage-prefix/token ranking",
            }
            for prefix, path in classical_scripts.items()
        },
        "artifacts": {
            "implementation_source_inventory.tsv": sha256_file(SOURCE_INVENTORY),
            "scientific_function_inventory.tsv": sha256_file(FUNCTION_INVENTORY),
            "scientific_constant_inventory.tsv": sha256_file(CONSTANT_INVENTORY),
            "TARGET_model_implementation_source_bindings.tsv": sha256_file(MODEL_BINDINGS),
            "DOG2_source_artifact_materialization_plan.tsv": sha256_file(SOURCE_ARTIFACT_PLAN),
        },
        "source_model_policy": {
            "DOG2_OS_may_be_read_in_05f2b": True,
            "TARGET_OS_may_be_read_in_05f2b": False,
            "source_hyperparameter_retuning_after_TARGET_open": False,
            "source_model_refit_after_TARGET_open": False,
        },
        "remaining_pre_outcome_work": [
            "bind exact classical callables and grids from frozen inventories",
            "freeze exact fold-safe Hallmark implementation code",
            "materialize DOG2 Hallmark50 source representation",
            "materialize/freeze full-DOG2 classical source model",
            "materialize/freeze full-DOG2 neural source model",
            "freeze exact T0-T2/N0-N5 final runner",
            "freeze exact Uno-C and IBS implementation adapters",
            "freeze final output/reporting schema",
        ],
        "TARGET_endpoint_values_may_be_read_now": False,
        "safety": {
            "TARGET_OS_event_values_read": False,
            "TARGET_OS_time_values_read": False,
            "TARGET_secondary_endpoint_values_read": False,
            "TARGET_survival_splits_generated": False,
            "TARGET_model_fitting": False,
            "DOG2_source_model_fitting": False,
            "GSE21257_outcomes_read": False,
            "GSE39055_outcomes_read": False,
            "GPU_execution": False,
        },
        "next": (
            "05f2b bind exact classical adapters from this frozen inventory, "
            "materialize DOG2 source Hallmark/classical/neural artifacts, and "
            "freeze the final human runner while TARGET endpoints remain closed."
        ),
    }

    write_json(CONTRACT_JSON, contract)

    summary = {
        "script_version": SCRIPT_VERSION,
        "contract_version": CONTRACT_VERSION,
        "status": "PASS",
        "scientific_status": contract["scientific_status"],
        "run_started_utc": started,
        "run_finished_utc": now_utc(),
        "TARGET_endpoint_values_may_be_read_now": False,
        "TARGET_OS_event_values_read": False,
        "TARGET_OS_time_values_read": False,
        "DOG2_source_models_fit": False,
        "05c_A0_callable": c05["A0_function"],
        "05c_A2_callable": c05["A2_function"],
        "05c_A4_callable": c05["A4_function"],
        "classical_script_03b": str(classical_scripts["03b"].relative_to(ROOT)),
        "classical_script_04c": str(classical_scripts["04c"].relative_to(ROOT)),
        "classical_script_04d": str(classical_scripts["04d"].relative_to(ROOT)),
        "classical_script_04e": str(classical_scripts["04e"].relative_to(ROOT)),
        "contract_sha256": sha256_file(CONTRACT_JSON),
        "artifact_hashes": contract["artifacts"],
        "next": contract["next"],
    }

    write_json(SUMMARY_JSON, summary)

    print("=" * 120)
    print("05f2a IMPLEMENTATION-SOURCE FREEZE SUMMARY")
    print("=" * 120)
    print("Frozen neural implementation:")
    print(f"  05c: {C05_SCRIPT.relative_to(ROOT)}")
    print(f"  A0 callable: {c05['A0_function']}")
    print("  A1 callable: train_A1_head")
    print(f"  A2 callable: {c05['A2_function']}")
    print(f"  A4 callable: {c05['A4_function']}")
    print()
    print("Frozen classical implementation sources:")
    for prefix in CLASSICAL_STAGE_PREFIXES:
        print(f"  {prefix}: {classical_scripts[prefix].relative_to(ROOT)}")
    print()
    print(f"Functions inventoried: {len(function_df):,}")
    print(f"Literal scientific constants inventoried: {len(constant_df):,}")
    print()
    print("Outcome firewall:")
    print("  TARGET OS event values read: NO")
    print("  TARGET OS time values read: NO")
    print("  TARGET survival splits generated: NO")
    print("  TARGET model fitting: NO")
    print("  DOG2 source model fitting: NO")
    print("  GSE21257/GSE39055 outcomes read: NO")
    print()
    print("TARGET endpoint values may be read now: NO")
    print(
        "Reason: final callable adapters/source artifacts/human runner "
        "must still be frozen in 05f2b."
    )
    print()
    print(f"05f2a contract SHA256: {sha256_file(CONTRACT_JSON)}")
    print()
    print("Next:")
    print(
        "  05f2b materialize/freeze DOG2 source artifacts and final "
        "T0-T2/N0-N5 runner while TARGET endpoints remain CLOSED."
    )
    print("=" * 120)
    print(
        "05f2a: PASS_HUMAN_MODEL_IMPLEMENTATION_SOURCE_INVENTORY_"
        "FROZEN_OUTCOME_CLOSED"
    )
    print("=" * 120)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 120, file=sys.stderr)
        print("05f2a implementation-source freeze: FAIL", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print("=" * 120, file=sys.stderr)
        raise
