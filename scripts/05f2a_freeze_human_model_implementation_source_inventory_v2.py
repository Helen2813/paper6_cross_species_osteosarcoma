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
    "05f2a-freeze-human-model-implementation-source-inventory-v2-explicit-provenance-bindings-no-cli"
)
CONTRACT_VERSION = (
    "paper6-posthold-human-model-implementation-source-inventory-v2-explicit-provenance-bindings"
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

# Exact authoritative classical implementation sources resolved PRE-OUTCOME
# from already-existing project provenance, not from filename ranking:
# - 03c explicitly imports 03b v2 and proves v1/v2 result-producing AST identity.
# - 04d v3 explicitly imports 04c v3.
# - 04c v3 states its v3 change is secondary output-path namespace only.
# - 04d v3 states its v3 change is dynamic-import registration only.
# - 04e is the frozen read-only regularization audit used downstream.
AUTHORITATIVE_CLASSICAL_SCRIPTS = {
    "03b": ROOT / "scripts" / "03b_run_dog2_source_prognostic_gate_v2.py",
    "04c": ROOT / "scripts" / "04c_run_classical_premise_test_v3.py",
    "04d": ROOT / "scripts" / "04d_audit_human_risk_orientation_and_reporting_v3.py",
    "04e": ROOT / "scripts" / "04e_audit_ridge_regularization_selection.py",
}
CLASSICAL_STAGE_PREFIXES = ["03b", "04c", "04d", "04e"]

# Provenance artifacts establishing those choices.
C3_DIR = ROOT / "results" / "source_gate_diagnostics" / "03c"
C3_REUSE = C3_DIR / "03b_v1_v2_reuse_provenance.json"
C3_SUMMARY = C3_DIR / "summary.json"

C4_DIR = ROOT / "results" / "human_premise" / "04c"
C4_SUMMARY = C4_DIR / "summary.json"

D4_DIR = ROOT / "results" / "human_premise_diagnostics" / "04d"
D4_SUMMARY = D4_DIR / "summary.json"

E4_DIR = ROOT / "results" / "human_premise_diagnostics" / "04e"
E4_SUMMARY = E4_DIR / "summary.json"

OUT_DIR = ROOT / "results" / "human_posthold_descriptive" / "05f2a_v2"
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


def extract_script_version(path: Path) -> str:
    constants = literal_upper_constants(path)
    value = constants.get("SCRIPT_VERSION")
    if not isinstance(value, str) or not value:
        raise RuntimeError(
            f"{path.name}: top-level SCRIPT_VERSION literal not found."
        )
    return value


def verify_authoritative_classical_sources() -> Dict[str, Any]:
    """
    Resolve classical implementation sources only from pre-existing
    pre-TARGET provenance. No filename ranking and no outcome-guided choice.
    """
    for path in [
        *AUTHORITATIVE_CLASSICAL_SCRIPTS.values(),
        C3_REUSE,
        C3_SUMMARY,
        C4_SUMMARY,
        D4_SUMMARY,
        E4_SUMMARY,
    ]:
        require_file(path)

    reuse = read_json(C3_REUSE)
    c3_summary = read_json(C3_SUMMARY)
    c4_summary = read_json(C4_SUMMARY)
    d4_summary = read_json(D4_SUMMARY)
    e4_summary = read_json(E4_SUMMARY)

    if clean(reuse.get("status")) != "PASS_REUSE_SCIENTIFICALLY_SAFE":
        raise RuntimeError(
            "03c does not certify 03b v1->v2 reuse as scientifically safe."
        )

    if reuse.get(
        "core_result_producing_AST_definitions_identical"
    ) is not True:
        raise RuntimeError(
            "03c provenance does not certify identical 03b v1/v2 "
            "result-producing AST definitions."
        )

    v2_prov = reuse.get("v2_script") or {}
    expected_v2_path = str(
        AUTHORITATIVE_CLASSICAL_SCRIPTS["03b"].relative_to(ROOT)
    )
    if clean(v2_prov.get("path")) != expected_v2_path:
        raise RuntimeError(
            "03c v2 provenance path differs from authoritative 03b v2 path."
        )

    expected_v2_hash = clean(v2_prov.get("sha256"))
    observed_v2_hash = sha256_file(
        AUTHORITATIVE_CLASSICAL_SCRIPTS["03b"]
    )
    if observed_v2_hash != expected_v2_hash:
        raise RuntimeError(
            "Authoritative 03b v2 script differs from 03c provenance hash."
        )

    if c3_summary.get("03b_v1_v2_reuse_safe") is not True:
        raise RuntimeError(
            "03c summary does not retain 03b v1/v2 reuse-safe status."
        )

    # Verify the expected implementation-fix identities from the source files
    # themselves. These checks read code only.
    expected_versions = {
        "03b": "03b-run-dog2-source-prognostic-gate-v2-implementation-fix-no-cli",
        "04c": "04c-run-classical-premise-test-v3-secondary-output-path-fix-no-cli",
        "04d": "04d-audit-human-risk-orientation-and-reporting-v3-dynamic-import-fix-no-cli",
        "04e": "04e-audit-ridge-regularization-selection-v1-no-cli",
    }

    observed_versions = {}
    for stage, path in AUTHORITATIVE_CLASSICAL_SCRIPTS.items():
        version = extract_script_version(path)
        observed_versions[stage] = version
        if version != expected_versions[stage]:
            raise RuntimeError(
                f"{stage}: authoritative script version changed: "
                f"{version!r} != {expected_versions[stage]!r}"
            )

    # 04c/04d/04e summaries are already-produced sacrificial-human diagnostics;
    # reading their metadata does not open reserved TARGET outcomes.
    if clean(c4_summary.get("status")) != "PASS":
        raise RuntimeError("04c summary is not PASS.")

    if clean(d4_summary.get("status")) != "PASS":
        raise RuntimeError("04d summary is not PASS.")

    if clean(
        d4_summary.get("scientific_status")
    ) != "PASS_NO_IMPLEMENTATION_SIGN_INVERSION_SUPPORTED":
        raise RuntimeError(
            "04d is not in the expected risk-orientation PASS state."
        )

    if clean(e4_summary.get("status")) != "PASS":
        raise RuntimeError("04e summary is not PASS.")

    if clean(
        e4_summary.get("scientific_status")
    ) != "PASS_RIDGE_REGULARIZATION_SELECTION_AUDIT":
        raise RuntimeError(
            "04e is not in the expected ridge-selection audit PASS state."
        )

    return {
        "scripts": dict(AUTHORITATIVE_CLASSICAL_SCRIPTS),
        "script_versions": observed_versions,
        "03c_reuse_provenance_sha256": sha256_file(C3_REUSE),
        "03c_summary_sha256": sha256_file(C3_SUMMARY),
        "04c_summary_sha256": sha256_file(C4_SUMMARY),
        "04d_summary_sha256": sha256_file(D4_SUMMARY),
        "04e_summary_sha256": sha256_file(E4_SUMMARY),
        "resolution_rule": (
            "EXPLICIT_PREOUTCOME_PROVENANCE_BINDING_NO_FILENAME_RANKING"
        ),
        "scientific_contract_changed": False,
    }


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
    print("  Technical recovery from 05f2a v1 ambiguous 03b filename ranking: YES")
    print("  Filename ranking used in v2: NO")
    print("  Explicit pre-outcome provenance binding used: YES")
    print("  Scientific contract changes: NO")
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

    classical_state = verify_authoritative_classical_sources()
    classical_scripts = classical_state["scripts"]

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
            "selection_method": "explicit pre-outcome provenance binding",
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
        "technical_recovery": {
            "prior_failed_script_version": (
                "05f2a-freeze-human-model-implementation-source-inventory-v1-no-cli"
            ),
            "failure_stage": "pre-outcome classical implementation-source discovery",
            "failure_reason": (
                "ambiguous filename ranking between 03b v1 and 03b v2"
            ),
            "TARGET_outcomes_read_before_failure": False,
            "resolution": (
                "bind exact scripts from existing 03c/04c/04d/04e provenance"
            ),
            "filename_ranking_removed": True,
            "scientific_contract_changed": False,
        },

        "classical_provenance_resolution": {
            key: value
            for key, value in classical_state.items()
            if key != "scripts"
        },

        "classical_sources": {
            prefix: {
                "script_path": str(path.relative_to(ROOT)),
                "script_sha256": sha256_file(path),
                "selection_rule": "explicit pre-outcome provenance binding",
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
        "technical_recovery_from_05f2a_v1": True,
        "filename_ranking_used": False,
        "classical_resolution_rule": classical_state["resolution_rule"],
        "03c_reuse_provenance_sha256": classical_state[
            "03c_reuse_provenance_sha256"
        ],
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
    print("Technical recovery:")
    print("  05f2a v1 ambiguous 03b source selection: CLOSED")
    print("  03b v2 selected from 03c reuse provenance: PASS")
    print("  filename ranking used: NO")
    print("  scientific contract changed: NO")
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
