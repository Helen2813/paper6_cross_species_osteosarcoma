#!/usr/bin/env python3
"""
Paper 6 - 05e3b final post-HOLD simulation mechanism audit.

This is the LAST simulation diagnostic permitted before the Paper-6 methods
section is drafted. No additional simulation model branch or diagnostic may be
created from these same results before that write-up.

Chronology
----------
Known before 05e3b:
- 05d original A2/A3 selection = HOLD.
- 05d0c A6 feasibility = STOP_A6_FOR_PAPER6.
- 05e0 controlled new-seed experiment frozen.
- 05e1 v3 materialized 6,000 untouched seeds.
- 05e2 fitted M0-M6.
- 05e3 evaluated frozen H1-H4.
- 05e3a corrected the structurally infeasible H3/H4 absolute catastrophic
  reduction component without changing frozen machine statuses.

05e3b addresses ONLY four predeclared questions:

Q1. H1 SIGN-INVERSION AUDIT — PRIMARY PREDICTION TEST
    Does retargeted A1 (M2) materially outperform simply negating the frozen
    zero-shot source risk (−M1)?

    Primary 29-event R5 comparison:
        delta C = C(M2) - C(-M1)

    BEYOND_SIGN_INVERSION if:
        mean delta C >= +0.02
        AND paired bootstrap 95% CI lower bound > 0.

    SIGN_INVERSION_EQUIVALENT if:
        the full paired bootstrap 95% CI lies inside [-0.02, +0.02].

    Otherwise:
        INCONCLUSIVE_RELATIVE_TO_SIGN_INVERSION.

    The +/-0.02 margin is a NEW, explicitly post-HOLD interpretive margin for
    this audit, informed by the already-used 05d0c operational reference. It is
    NOT retroactively assigned to 05a.

    Risk-score correlation and head geometry are SECONDARY explanatory
    quantities only. They can never upgrade a prediction-level result.

    R0 is a directional positive control:
        median corr(M2 risk, M1 risk) > 0
        AND median cosine(head_M2, source_head) > 0.
    Failure of that directional control blocks geometric interpretation but
    does not rewrite prediction metrics.

Q2. DISCRETE REGIME-SHAPE / HETEROGENEITY-VALLEY AUDIT
    On the untouched 05e seeds, at 29 events, is M2's transfer gain relative to
    B0 lowest in R2_PARTIALLY_TRANSPORTABLE compared with each of:
        R0_FULLY_TRANSPORTABLE,
        R4_NONTRANSPORTABLE,
        R5_MISLEADING_SOURCE?

    Primary quantity:
        delta C(M2-B0), not absolute C.

    DISCRETE_HETEROGENEITY_VALLEY_SUPPORTS if independent two-sample bootstrap
    95% CIs for:
        mean[R0 delta] - mean[R2 delta]
        mean[R4 delta] - mean[R2 delta]   <-- key anti-"low magnitude" contrast
        mean[R5 delta] - mean[R2 delta]
    are all strictly > 0.

    This is a FOUR-CATEGORY generator contrast, not a claim about a continuous
    U-shaped biological function.

    R4 is the key logic cell: it has no transportable source effect, so if R4
    still exceeds R2, a simple "less transferable signal magnitude causes the
    valley" explanation is insufficient.

Q3. M6-PRIME FIXED-RESIDUAL GATE SUBSTITUTION
    In the primary 29-event R5 cell only, replay the exact source network and
    substitute the TRUE transportability mask at prediction while keeping the
    already-fitted M4 residual COMPLETELY UNCHANGED.

    This is:
        M6_PRIME_TRUE_GATE_FIXED_M4_RESIDUAL.

    It is a prediction-stage gate-substitution diagnostic.

    IMPORTANT ASYMMETRY:
    M5's hardened learned gate is consistent with the gate under which the M4
    residual was trained. M6-prime's truth gate is NOT. Therefore this
    comparison structurally favors the learned-gate construction and MUST NOT
    be reported as "learned gate better than true gate."

    M6-prime cannot change H4's frozen status.

Q4. INTERPRETATION BRANCHES FROZEN BEFORE READING 05e3b VALUES

    BRANCH_A_HETEROGENEITY_VALLEY:
      selected only if Q1 = SIGN_INVERSION_EQUIVALENT
      AND Q2 = DISCRETE_HETEROGENEITY_VALLEY_SUPPORTS.

      Prewritten framing:
      "Under event scarcity, homogeneous cross-species prognostic relationships
       at either extreme are comparatively tractable, whereas heterogeneous
       partial transportability is the difficult regime; global sign reversal
       can be corrected largely by a one-dimensional head inversion."

    BRANCH_B_REPRESENTATION_RETARGETING:
      selected if Q1 = BEYOND_SIGN_INVERSION.

      Prewritten framing:
      "A frozen cross-species representation retains target-relevant prognostic
       structure beyond simple sign inversion, while heterogeneous partial
       transportability remains a distinct transfer challenge."

    If neither rule is met:
      NO_PREDECLARED_TITLE_BRANCH_CONFIRMED.

Future threshold policy
-----------------------
This audit also records a prospective project-wide policy:
Future rate-reduction decision criteria must be expressed as a prespecified
RELATIVE reduction and may be assessed only above a prespecified control
base-rate floor; otherwise the component is NOT_ASSESSABLE. Exact Y/floor must
be frozen before the future result. This rule is NOT retroactively applied to
05e3.

No human outcomes.
No model selection.
No A6 reopening.
No new architecture.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import torch
except ImportError as exc:
    raise ImportError("05e3b M6-prime replay requires PyTorch.") from exc


SCRIPT_VERSION = (
    "05e3b-final-signinversion-regimeshape-fixedresidual-audit-v1-no-cli"
)
AUDIT_VERSION = "paper6-05e3b-final-simulation-mechanism-audit-v1"

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Frozen/completed upstream artifacts.
# ---------------------------------------------------------------------------
E0_DIR = ROOT / "results" / "posthold_controlled_mechanism" / "05e0"
E0_CONTRACT = E0_DIR / "controlled_mechanism_contract.json"
E0_SCENARIOS = E0_DIR / "controlled_mechanism_scenario_registry.tsv"

E1_DIR = ROOT / "results" / "posthold_controlled_mechanism" / "05e1_v3"
E1_MANIFEST = E1_DIR / "controlled_recipe_manifest.tsv"
E1_SUMMARY = E1_DIR / "summary.json"

E2_DIR = ROOT / "results" / "posthold_controlled_mechanism" / "05e2"
E2_IMPL = E2_DIR / "controlled_model_implementation_contract.json"
E2_MANIFEST = E2_DIR / "controlled_model_output_manifest.tsv"
E2_SUMMARY = E2_DIR / "summary.json"

E3_DIR = ROOT / "results" / "posthold_controlled_mechanism" / "05e3"
E3_SCRIPT = ROOT / "scripts" / "05e3_evaluate_frozen_controlled_mechanism_contrasts.py"
E3_EVAL_CONTRACT = E3_DIR / "evaluation_implementation_contract.json"
E3_METRIC_MANIFEST = E3_DIR / "controlled_metric_manifest.tsv"
E3_CELL_SUMMARY = E3_DIR / "cell_model_summary.tsv"
E3_SUMMARY = E3_DIR / "summary.json"

E3A_DIR = ROOT / "results" / "posthold_controlled_mechanism" / "05e3a"
E3A_SUMMARY = E3A_DIR / "summary.json"

E2_SCRIPT = ROOT / "scripts" / "05e2_run_frozen_controlled_mechanism_model_matrix.py"
C05_SCRIPT = ROOT / "scripts" / "05c_run_closed_synthetic_transfer_model_matrix.py"

B_DIR = ROOT / "simulations" / "05b"
B_GENERATOR_CONTRACT = B_DIR / "generator_contract.json"

D_SCRIPT = ROOT / "scripts" / "05d_aggregate_negative_transfer_phase_diagram.py"
D_METRIC_CONTRACT = ROOT / "results" / "simulation_phase_diagram" / "05d" / "metric_implementation_contract.json"

EXPECTED_E0_CONTRACT_SHA256 = (
    "05cd5bc9180909c4c76f4e9a086a6fa1a2c37faee2f2675008e6e1dd91c581fb"
)
EXPECTED_E2_STATUS = "PASS_FROZEN_CONTROLLED_MECHANISM_MODEL_MATRIX_COMPLETE"
EXPECTED_E3_STATUS = "PASS_CONTROLLED_MECHANISM_CONTRAST_EVALUATION_COMPLETE"
EXPECTED_E3A_STATUS = (
    "PASS_POSTRESULT_MECHANISM_SUPPORT_RULE_FEASIBILITY_AUDIT_COMPLETE"
)
EXPECTED_E3_EVAL_SHA256 = (
    "ca547a9c6c539d7258e7023c4b6fd12064a604bdb442112fd8e15db126c59c63"
)

# ---------------------------------------------------------------------------
# Outputs.
# ---------------------------------------------------------------------------
OUT_DIR = ROOT / "results" / "posthold_controlled_mechanism" / "05e3b"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AUDIT_CONTRACT = OUT_DIR / "final_simulation_audit_contract.json"
H1_REPLICATE = OUT_DIR / "H1_sign_inversion_replicate_metrics.tsv"
H1_SUMMARY = OUT_DIR / "H1_sign_inversion_cell_summary.tsv"
REGIME_TABLE = OUT_DIR / "newseed_regime_shape_table.tsv"
VALLEY_CONTRASTS = OUT_DIR / "newseed_heterogeneity_valley_contrasts.tsv"
M6P_DIAGNOSTIC = OUT_DIR / "M6prime_fixed_residual_R5_diagnostic.tsv"
FUTURE_THRESHOLD_POLICY = OUT_DIR / "future_rate_reduction_feasibility_policy.json"
SUMMARY_JSON = OUT_DIR / "summary.json"

MODEL_M0 = "M0_B0"
MODEL_M1 = "M1_SOURCE_NETWORK_ZERO_SHOT"
MODEL_M2 = "M2_A1_SOURCE_CENTERED_HEAD"
MODEL_M4 = "M4_A3_SOFT_LEARNED"
MODEL_M5 = "M5_A3_HARDENED_PREDICTION"

PRIMARY_EVENTS = 29
REGIME_R0 = "R0_FULLY_TRANSPORTABLE"
REGIME_R2 = "R2_PARTIALLY_TRANSPORTABLE"
REGIME_R4 = "R4_NONTRANSPORTABLE"
REGIME_R5 = "R5_MISLEADING_SOURCE"
REGIMES = [REGIME_R0, REGIME_R2, REGIME_R4, REGIME_R5]
EVENTS = [10, 29, 40]

SIGN_EQUIVALENCE_MARGIN = 0.02
BEYOND_SIGN_MARGIN = 0.02

NEGATIVE_TRANSFER_DELTA = -0.02
CATASTROPHIC_DELTA = -0.05

BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 931_300_001

MIN_GPU_FREE_BYTES = 3 * 1024**3


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def require_file(path: Path) -> Path:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Required artifact missing: {path}")
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
        raise RuntimeError(f"Could not import {path}")

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


def finite_mean(values: Iterable[float]) -> float:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if len(x) else float("nan")


def finite_median(values: Iterable[float]) -> float:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else float("nan")


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]

    if len(x) < 3 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")

    return float(np.corrcoef(x, y)[0, 1])


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)


def head_projection_metrics(
    source_head: np.ndarray,
    target_head: np.ndarray,
) -> Tuple[float, float, float]:
    source = np.asarray(source_head, dtype=float)
    target = np.asarray(target_head, dtype=float)

    source_sq = float(np.dot(source, source))
    target_norm = float(np.linalg.norm(target))

    if source_sq <= 1e-12 or target_norm <= 1e-12:
        return float("nan"), float("nan"), float("nan")

    alpha = float(np.dot(target, source) / source_sq)
    perp = target - alpha * source
    orthogonal_fraction = float(
        np.linalg.norm(perp) / target_norm
    )
    return cosine(source, target), alpha, orthogonal_fraction


def paired_bootstrap(
    diff: np.ndarray,
    *,
    seed: int,
) -> Dict[str, float]:
    x = np.asarray(diff, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        raise RuntimeError("Paired bootstrap received zero finite observations.")

    rng = np.random.default_rng(seed)
    n = len(x)
    boot = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)

    for b in range(BOOTSTRAP_DRAWS):
        idx = rng.integers(0, n, size=n)
        boot[b] = float(np.mean(x[idx]))

    return {
        "n": int(n),
        "mean": float(np.mean(x)),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "p_positive": float(np.mean(boot > 0)),
        "p_negative": float(np.mean(boot < 0)),
    }


def independent_mean_difference_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    *,
    seed: int,
) -> Dict[str, float]:
    """
    Difference mean(a) - mean(b), independent cell seeds.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if len(a) == 0 or len(b) == 0:
        raise RuntimeError("Independent bootstrap received empty vector.")

    rng = np.random.default_rng(seed)
    boot = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)

    for k in range(BOOTSTRAP_DRAWS):
        ai = rng.integers(0, len(a), size=len(a))
        bi = rng.integers(0, len(b), size=len(b))
        boot[k] = float(np.mean(a[ai]) - np.mean(b[bi]))

    obs = float(np.mean(a) - np.mean(b))
    return {
        "mean_difference": obs,
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "p_positive": float(np.mean(boot > 0)),
    }


def choose_replay_device() -> torch.device:
    request = os.environ.get(
        "PAPER6_MECH_REPLAY_DEVICE",
        "auto",
    ).strip().lower()

    if request not in {"auto", "cuda", "cpu"}:
        raise RuntimeError(
            "PAPER6_MECH_REPLAY_DEVICE must be auto, cuda, or cpu."
        )

    if request == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        free_bytes, _ = torch.cuda.mem_get_info()

        if free_bytes >= MIN_GPU_FREE_BYTES:
            return torch.device("cuda")

        if request == "cuda":
            raise RuntimeError(
                "CUDA explicitly requested but <3 GiB is free."
            )

    # Only one 500-replicate source replay is required, so CPU fallback is
    # allowed rather than competing with a long-running unrelated GPU job.
    return torch.device("cpu")


def model_metric_vectors(
    metric_path: Path,
    model_index: int,
) -> Dict[str, np.ndarray]:
    with np.load(metric_path, allow_pickle=False) as data:
        return {
            "uno_c": np.asarray(data["uno_c"][:, model_index], dtype=float),
            "ibs": np.asarray(data["ibs"][:, model_index], dtype=float),
            "risk_sd": np.asarray(
                data["test_risk_sd"][:, model_index],
                dtype=float,
            ),
            "delta_c_vs_B0": np.asarray(
                data["delta_c_vs_B0"][:, model_index],
                dtype=float,
            ),
            "negative_transfer": np.asarray(
                data["negative_transfer"][:, model_index],
                dtype=np.uint8,
            ),
            "catastrophic": np.asarray(
                data["catastrophic_negative_transfer"][:, model_index],
                dtype=np.uint8,
            ),
        }


def main() -> None:
    started = now_utc()

    print("=" * 120)
    print("Paper 6 - 05e3b final post-HOLD simulation mechanism audit")
    print("=" * 120)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Audit version: {AUDIT_VERSION}")
    print()
    print("Scope:")
    print("  New architecture/model branch: NO")
    print("  Existing 05e2 prediction arrays read: YES")
    print("  One deterministic source-network replay for M6-prime: YES")
    print("  Model selection: NO")
    print("  A6 reopened: NO")
    print("  Human outcomes read: NO")
    print("  Last simulation diagnostic before methods draft: YES")
    print()

    for path in [
        E0_CONTRACT,
        E0_SCENARIOS,
        E1_MANIFEST,
        E1_SUMMARY,
        E2_IMPL,
        E2_MANIFEST,
        E2_SUMMARY,
        E3_SCRIPT,
        E3_EVAL_CONTRACT,
        E3_METRIC_MANIFEST,
        E3_CELL_SUMMARY,
        E3_SUMMARY,
        E3A_SUMMARY,
        E2_SCRIPT,
        C05_SCRIPT,
        B_GENERATOR_CONTRACT,
        D_SCRIPT,
        D_METRIC_CONTRACT,
    ]:
        require_file(path)

    if sha256_file(E0_CONTRACT) != EXPECTED_E0_CONTRACT_SHA256:
        raise RuntimeError("05e0 contract SHA256 changed.")

    e2_summary = read_json(E2_SUMMARY)
    e3_summary = read_json(E3_SUMMARY)
    e3a_summary = read_json(E3A_SUMMARY)
    e2_impl = read_json(E2_IMPL)
    e3_eval = read_json(E3_EVAL_CONTRACT)

    if clean(e2_summary.get("scientific_status")) != EXPECTED_E2_STATUS:
        raise RuntimeError("05e2 is not in expected PASS state.")
    if clean(e3_summary.get("scientific_status")) != EXPECTED_E3_STATUS:
        raise RuntimeError("05e3 is not in expected PASS state.")
    if clean(e3a_summary.get("scientific_status")) != EXPECTED_E3A_STATUS:
        raise RuntimeError("05e3a is not in expected PASS state.")
    if sha256_file(E3_EVAL_CONTRACT) != EXPECTED_E3_EVAL_SHA256:
        raise RuntimeError("05e3 evaluation contract SHA256 changed.")

    if sha256_file(E2_SCRIPT) != clean(e2_impl.get("script_sha256")):
        raise RuntimeError("05e2 script differs from frozen implementation contract.")
    if sha256_file(C05_SCRIPT) != clean(e2_impl.get("05c_script_sha256")):
        raise RuntimeError("Frozen 05c script differs from 05e2 contract.")
    if sha256_file(E3_SCRIPT) != clean(e3_eval.get("script_sha256")):
        raise RuntimeError("05e3 script differs from frozen evaluation contract.")
    if sha256_file(D_SCRIPT) != clean(e3_eval.get("05d_script_sha256")):
        raise RuntimeError("Frozen 05d script differs from 05e3 contract.")

    # ------------------------------------------------------------------
    # Freeze interpretation BEFORE reading any new 05e3b values.
    # ------------------------------------------------------------------
    audit_contract = {
        "script_version": SCRIPT_VERSION,
        "audit_version": AUDIT_VERSION,
        "status": "FROZEN_BEFORE_05E3B_NEW_VALUES_READ",
        "created_utc": now_utc(),
        "known_prior_results": {
            "05e3_H1_support": True,
            "05e3_H2_support": False,
            "05e3_H3_primary_supportive_composite_infeasible": True,
            "05e3_H4_primary_failed": True,
        },
        "Q1_H1_sign_inversion": {
            "primary_cell": "29-event R5_MISLEADING_SOURCE",
            "primary_estimand": "paired C(M2) - C(-M1)",
            "beyond_sign_inversion": (
                "mean >= +0.02 AND paired bootstrap 95% CI lower > 0"
            ),
            "sign_inversion_equivalent": (
                "entire paired bootstrap 95% CI inside [-0.02,+0.02]"
            ),
            "otherwise": "INCONCLUSIVE_RELATIVE_TO_SIGN_INVERSION",
            "geometry_secondary_only": True,
            "R0_directional_positive_control": (
                "median risk corr(M2,M1)>0 AND median head cosine(source,M2)>0"
            ),
        },
        "Q2_discrete_regime_shape": {
            "primary_event_budget": 29,
            "primary_quantity": "deltaC(M2-B0)",
            "R2_valley_support": (
                "bootstrap CI lower >0 for R0-R2, R4-R2, and R5-R2"
            ),
            "R4_role": (
                "key anti-low-transferable-signal-magnitude contrast"
            ),
            "continuous_U_shape_claim_allowed": False,
        },
        "Q3_M6prime": {
            "definition": (
                "truth hard gate substituted at prediction with residual_M4 unchanged"
            ),
            "primary_cell": "29-event R5",
            "new_fit": False,
            "asymmetry": (
                "M4 residual was optimized under learned soft gate; M5 hardening "
                "is gate-consistent, M6-prime truth substitution is not. "
                "Do not compare them as optimal learned-vs-true gates."
            ),
            "can_change_H4_status": False,
        },
        "predeclared_interpretation_branches": {
            "BRANCH_A_HETEROGENEITY_VALLEY": (
                "Require SIGN_INVERSION_EQUIVALENT and discrete R2 valley support. "
                "Frame homogeneous agreement/reversal as comparatively tractable "
                "and heterogeneous partial transportability as the difficult regime; "
                "H1 is supporting one-dimensional sign-correction mechanism."
            ),
            "BRANCH_B_REPRESENTATION_RETARGETING": (
                "Require BEYOND_SIGN_INVERSION. Frame frozen representation "
                "retargeting beyond simple sign inversion as primary; discrete "
                "regime shape is secondary."
            ),
            "otherwise": "NO_PREDECLARED_TITLE_BRANCH_CONFIRMED",
        },
        "future_rate_reduction_policy": (
            "Future decision criteria for rate reduction must prespecify a "
            "RELATIVE reduction Y and a control base-rate floor before results; "
            "if control rate is below that floor the component is NOT_ASSESSABLE. "
            "No retroactive application to 05e3."
        ),
        "stop_rule": (
            "After 05e3b, no additional simulation model branch or simulation "
            "diagnostic may be initiated from these results before the Paper-6 "
            "methods/results framework is drafted."
        ),
        "guardrails": {
            "no_A6_reopening": True,
            "no_model_selection": True,
            "no_human_outcomes": True,
            "no_geometry_only_claim": True,
        },
    }
    write_json(AUDIT_CONTRACT, audit_contract)
    audit_contract_hash = sha256_file(AUDIT_CONTRACT)

    future_policy = {
        "created_utc": now_utc(),
        "status": "PROSPECTIVE_PROJECT_POLICY_FOR_FUTURE_CONTRACTS",
        "rule": (
            "Rate-reduction support criteria should use a prespecified relative "
            "reduction and a prespecified control base-rate floor. If the realized "
            "control rate is below the frozen floor, that criterion is NOT_ASSESSABLE."
        ),
        "exact_relative_reduction_Y": "MUST_BE_FROZEN_IN_FUTURE_CONTRACT",
        "exact_control_base_rate_floor": "MUST_BE_FROZEN_IN_FUTURE_CONTRACT",
        "retroactively_changes_05e3": False,
    }
    write_json(FUTURE_THRESHOLD_POLICY, future_policy)

    print(f"05e3b audit contract SHA256: {audit_contract_hash}")
    print("Interpretation branches frozen before new audit values: PASS")
    print()

    # ------------------------------------------------------------------
    # Load metadata / locate cells.
    # ------------------------------------------------------------------
    scenarios = pd.read_csv(E0_SCENARIOS, sep="\t")
    e1_manifest = pd.read_csv(E1_MANIFEST, sep="\t")
    e2_manifest = pd.read_csv(E2_MANIFEST, sep="\t")
    e3_metric_manifest = pd.read_csv(E3_METRIC_MANIFEST, sep="\t")

    cell_lookup = {
        (int(r.target_events), str(r.transfer_regime)): str(r.cell_id)
        for r in scenarios.itertuples(index=False)
    }
    scenario_by_cell = {
        str(r.cell_id): r
        for r in scenarios.itertuples(index=False)
    }
    e1_by_cell = {
        str(r.cell_id): r
        for r in e1_manifest.itertuples(index=False)
    }
    e2_by_cell = {
        str(r.cell_id): r
        for r in e2_manifest.itertuples(index=False)
    }
    e3_by_cell = {
        str(r.cell_id): r
        for r in e3_metric_manifest.itertuples(index=False)
    }

    required_cells = {
        (e, r) for e in EVENTS for r in REGIMES
    }
    if set(cell_lookup) != required_cells:
        raise RuntimeError(
            "05e0 controlled cell grid differs from required 3x4 design."
        )

    # Import exact evaluation function after contract freeze.
    e3mod = load_module(E3_SCRIPT, "paper6_05e3_for_05e3b")
    d05 = load_module(D_SCRIPT, "paper6_05d_for_05e3b")

    # ------------------------------------------------------------------
    # Q1 H1 sign-inversion audit across all 12 cells.
    # ------------------------------------------------------------------
    h1_rep_rows: List[Dict[str, Any]] = []
    h1_summary_rows: List[Dict[str, Any]] = []

    for cell_counter, ((events, regime), cell_id) in enumerate(
        sorted(cell_lookup.items(), key=lambda x: (x[0][0], x[0][1]))
    ):
        e2row = e2_by_cell[cell_id]
        e3row = e3_by_cell[cell_id]

        source_path = ROOT / str(e2row.output_path)
        metric_path = ROOT / str(e3row.metric_output_path)

        require_file(source_path)
        require_file(metric_path)

        if sha256_file(source_path) != str(e2row.output_sha256):
            raise RuntimeError(f"{cell_id}: 05e2 output hash mismatch.")
        if sha256_file(metric_path) != str(e3row.metric_output_sha256):
            raise RuntimeError(f"{cell_id}: 05e3 metric hash mismatch.")

        with np.load(source_path, allow_pickle=False) as source, np.load(
            metric_path, allow_pickle=False
        ) as metric:
            seeds = np.asarray(source["replicate_seed"], dtype=np.int64)
            train_time = np.asarray(source["target_train_time"], dtype=float)
            train_event = np.asarray(source["target_train_event"], dtype=np.uint8)
            test_time = np.asarray(source["target_test_time"], dtype=float)
            test_event = np.asarray(source["target_test_event"], dtype=np.uint8)

            risk_m1 = np.asarray(
                source[f"risk_test_{MODEL_M1}"],
                dtype=float,
            )
            risk_m2 = np.asarray(
                source[f"risk_test_{MODEL_M2}"],
                dtype=float,
            )
            head_source = np.asarray(source["head_source"], dtype=float)
            head_m2 = np.asarray(
                source["head_M2_A1_SOURCE_CENTERED"],
                dtype=float,
            )

            model_names_e3 = [
                "M0_B0",
                "M1_SOURCE_NETWORK_ZERO_SHOT",
                "M2_A1_SOURCE_CENTERED_HEAD",
                "M3_A1_FREE_HEAD",
                "M4_A3_SOFT_LEARNED",
                "M5_A3_HARDENED_PREDICTION",
                "M6_A3_ORACLE_HARD_GATE",
            ]
            m1i = model_names_e3.index(MODEL_M1)
            m2i = model_names_e3.index(MODEL_M2)

            c_m1_saved = np.asarray(metric["uno_c"][:, m1i], dtype=float)
            c_m2_saved = np.asarray(metric["uno_c"][:, m2i], dtype=float)

            c_neg_m1 = np.full(len(seeds), np.nan, dtype=float)
            risk_corr = np.full(len(seeds), np.nan, dtype=float)
            cos_head = np.full(len(seeds), np.nan, dtype=float)
            proj_alpha = np.full(len(seeds), np.nan, dtype=float)
            orth_frac = np.full(len(seeds), np.nan, dtype=float)

            for r in range(len(seeds)):
                c_neg_m1[r] = e3mod.safe_uno(
                    train_time[r],
                    train_event[r],
                    test_time[r],
                    test_event[r],
                    -risk_m1[r],
                )
                risk_corr[r] = pearson(risk_m2[r], risk_m1[r])
                cos_head[r], proj_alpha[r], orth_frac[r] = (
                    head_projection_metrics(
                        head_source[r],
                        head_m2[r],
                    )
                )

                h1_rep_rows.append({
                    "cell_id": cell_id,
                    "target_events": events,
                    "transfer_regime": regime,
                    "replicate": r,
                    "replicate_seed": int(seeds[r]),
                    "C_M1": float(c_m1_saved[r]),
                    "C_neg_M1": float(c_neg_m1[r]),
                    "C_M2": float(c_m2_saved[r]),
                    "delta_C_M2_minus_negM1": float(
                        c_m2_saved[r] - c_neg_m1[r]
                    ),
                    "pearson_risk_M2_vs_M1": float(risk_corr[r]),
                    "pearson_risk_M2_vs_negM1": float(-risk_corr[r]),
                    "head_cosine_M2_vs_source": float(cos_head[r]),
                    "head_projection_alpha_on_source": float(proj_alpha[r]),
                    "head_orthogonal_fraction": float(orth_frac[r]),
                })

            diff = c_m2_saved - c_neg_m1
            boot = paired_bootstrap(
                diff,
                seed=BOOTSTRAP_SEED + cell_counter,
            )

            if (
                boot["mean"] >= BEYOND_SIGN_MARGIN
                and boot["ci_low"] > 0
            ):
                prediction_class = "BEYOND_SIGN_INVERSION"
            elif (
                boot["ci_low"] > -SIGN_EQUIVALENCE_MARGIN
                and boot["ci_high"] < SIGN_EQUIVALENCE_MARGIN
            ):
                prediction_class = "SIGN_INVERSION_EQUIVALENT"
            else:
                prediction_class = "INCONCLUSIVE_RELATIVE_TO_SIGN_INVERSION"

            h1_summary_rows.append({
                "cell_id": cell_id,
                "target_events": events,
                "transfer_regime": regime,
                "n_replicates": len(seeds),
                "mean_C_M1": finite_mean(c_m1_saved),
                "mean_C_neg_M1": finite_mean(c_neg_m1),
                "mean_C_M2": finite_mean(c_m2_saved),
                "mean_delta_C_M2_minus_negM1": boot["mean"],
                "bootstrap_ci_low": boot["ci_low"],
                "bootstrap_ci_high": boot["ci_high"],
                "bootstrap_p_positive": boot["p_positive"],
                "prediction_level_classification": prediction_class,
                "median_pearson_risk_M2_vs_M1": finite_median(risk_corr),
                "median_pearson_risk_M2_vs_negM1": finite_median(-risk_corr),
                "median_head_cosine_M2_vs_source": finite_median(cos_head),
                "median_head_projection_alpha": finite_median(proj_alpha),
                "median_head_orthogonal_fraction": finite_median(orth_frac),
            })

    h1_rep_df = pd.DataFrame(h1_rep_rows)
    h1_summary_df = pd.DataFrame(h1_summary_rows)

    h1_rep_df.to_csv(H1_REPLICATE, sep="\t", index=False)
    h1_summary_df.to_csv(H1_SUMMARY, sep="\t", index=False)

    primary_h1 = h1_summary_df[
        (h1_summary_df["target_events"] == PRIMARY_EVENTS)
        & (h1_summary_df["transfer_regime"] == REGIME_R5)
    ]
    if len(primary_h1) != 1:
        raise RuntimeError("Primary 29-event R5 H1 row not unique.")
    primary_h1 = primary_h1.iloc[0]
    h1_primary_class = str(
        primary_h1["prediction_level_classification"]
    )

    r0_primary = h1_summary_df[
        (h1_summary_df["target_events"] == PRIMARY_EVENTS)
        & (h1_summary_df["transfer_regime"] == REGIME_R0)
    ]
    if len(r0_primary) != 1:
        raise RuntimeError("Primary 29-event R0 H1 row not unique.")
    r0_primary = r0_primary.iloc[0]

    r0_directional_control_pass = bool(
        float(r0_primary["median_pearson_risk_M2_vs_M1"]) > 0
        and float(r0_primary["median_head_cosine_M2_vs_source"]) > 0
    )

    # ------------------------------------------------------------------
    # Q2 untouched new-seed regime-shape table and 29-event valley contrasts.
    # ------------------------------------------------------------------
    regime_rows: List[Dict[str, Any]] = []
    cell_metric_vectors: Dict[Tuple[int, str], Dict[str, np.ndarray]] = {}

    for events in EVENTS:
        for regime in REGIMES:
            cell_id = cell_lookup[(events, regime)]
            metric_path = ROOT / str(e3_by_cell[cell_id].metric_output_path)

            with np.load(metric_path, allow_pickle=False) as metric:
                uno = np.asarray(metric["uno_c"], dtype=float)
                delta = np.asarray(metric["delta_c_vs_B0"], dtype=float)

                m0i = 0
                m1i = 1
                m2i = 2

                cell_metric_vectors[(events, regime)] = {
                    "C_M0": uno[:, m0i].copy(),
                    "C_M1": uno[:, m1i].copy(),
                    "C_M2": uno[:, m2i].copy(),
                    "delta_M2_B0": delta[:, m2i].copy(),
                    "delta_M2_M1": (
                        uno[:, m2i] - uno[:, m1i]
                    ).copy(),
                }

                scenario = scenario_by_cell[cell_id]
                regime_rows.append({
                    "cell_id": cell_id,
                    "target_events": events,
                    "transfer_regime": regime,
                    "transferable_fraction": float(
                        scenario.transferable_fraction
                    ),
                    "effect_concordance": float(
                        scenario.effect_concordance
                    ),
                    "nontransferable_sign_flip_fraction": float(
                        scenario.nontransferable_sign_flip_fraction
                    ),
                    "mean_C_B0": finite_mean(uno[:, m0i]),
                    "mean_C_M1_zero_shot": finite_mean(uno[:, m1i]),
                    "mean_C_M2_retargeted": finite_mean(uno[:, m2i]),
                    "mean_delta_C_M2_minus_B0": finite_mean(delta[:, m2i]),
                    "mean_delta_C_M2_minus_M1": finite_mean(
                        uno[:, m2i] - uno[:, m1i]
                    ),
                })

    regime_df = pd.DataFrame(regime_rows)
    regime_df.to_csv(REGIME_TABLE, sep="\t", index=False)

    valley_rows: List[Dict[str, Any]] = []
    r2_delta = cell_metric_vectors[
        (PRIMARY_EVENTS, REGIME_R2)
    ]["delta_M2_B0"]

    for j, regime in enumerate([REGIME_R0, REGIME_R4, REGIME_R5]):
        other_delta = cell_metric_vectors[
            (PRIMARY_EVENTS, regime)
        ]["delta_M2_B0"]

        boot = independent_mean_difference_bootstrap(
            other_delta,
            r2_delta,
            seed=BOOTSTRAP_SEED + 100 + j,
        )

        valley_rows.append({
            "target_events": PRIMARY_EVENTS,
            "contrast": f"{regime}_minus_{REGIME_R2}",
            "regime_other": regime,
            "regime_reference": REGIME_R2,
            "mean_difference_deltaC_M2_B0": boot["mean_difference"],
            "bootstrap_ci_low": boot["ci_low"],
            "bootstrap_ci_high": boot["ci_high"],
            "bootstrap_p_positive": boot["p_positive"],
            "strictly_positive_CI": bool(boot["ci_low"] > 0),
            "logical_role": (
                "KEY_ANTI_LOW_MAGNITUDE_CONTRAST"
                if regime == REGIME_R4
                else "EDGE_VS_PARTIAL_CONTRAST"
            ),
        })

    valley_df = pd.DataFrame(valley_rows)
    valley_df.to_csv(VALLEY_CONTRASTS, sep="\t", index=False)

    valley_support = bool(
        valley_df["strictly_positive_CI"].all()
    )

    # Directional replication at 10 and 40 events, descriptive only.
    valley_direction_by_budget = {}
    for events in [10, 40]:
        r2 = finite_mean(
            cell_metric_vectors[(events, REGIME_R2)]["delta_M2_B0"]
        )
        valley_direction_by_budget[str(events)] = {
            regime: bool(
                finite_mean(
                    cell_metric_vectors[(events, regime)]["delta_M2_B0"]
                )
                > r2
            )
            for regime in [REGIME_R0, REGIME_R4, REGIME_R5]
        }

    # ------------------------------------------------------------------
    # Q3 M6-prime: truth gate substituted at prediction; residual_M4 unchanged.
    # ------------------------------------------------------------------
    r5_cell_id = cell_lookup[(PRIMARY_EVENTS, REGIME_R5)]
    e1row = e1_by_cell[r5_cell_id]
    e2row = e2_by_cell[r5_cell_id]
    e3row = e3_by_cell[r5_cell_id]

    recipe_path = ROOT / str(e1row.recipe_path)
    source_output_path = ROOT / str(e2row.output_path)
    metric_path = ROOT / str(e3row.metric_output_path)

    require_file(recipe_path)
    require_file(source_output_path)
    require_file(metric_path)

    # Import exact frozen implementations.
    e2mod = load_module(E2_SCRIPT, "paper6_05e2_for_05e3b")
    c05 = load_module(C05_SCRIPT, "paper6_05c_for_05e3b")

    generator_contract = read_json(B_GENERATOR_CONTRACT)
    generator_path = ROOT / clean(
        generator_contract.get("authoritative_generator_script")
    )
    generator_hash = clean(
        generator_contract.get("authoritative_generator_script_sha256")
    )
    require_file(generator_path)
    if sha256_file(generator_path) != generator_hash:
        raise RuntimeError("Authoritative 05b generator hash mismatch.")
    generator = load_module(generator_path, "paper6_05b_for_05e3b")

    replay_device = choose_replay_device()
    c05.configure_torch(replay_device)

    recipe = e2mod.load_recipe(recipe_path)
    arrays = e2mod.generate_cell_arrays(
        generator,
        recipe,
        cell_id=r5_cell_id,
    )

    with np.load(source_output_path, allow_pickle=False) as saved:
        saved_head_source = np.asarray(saved["head_source"], dtype=float)
        saved_m1_train = np.asarray(
            saved[f"risk_train_{MODEL_M1}"], dtype=float
        )
        saved_m1_test = np.asarray(
            saved[f"risk_test_{MODEL_M1}"], dtype=float
        )
        residual_m4 = np.asarray(
            saved["residual_M4_A3_SOFT"], dtype=np.float32
        )
        truth_gate_np = np.asarray(
            saved["transportable_mask"], dtype=np.float32
        )

    # Reconstruct exact source network / target standardization for primary cell.
    Xs = torch.as_tensor(
        arrays["X_source"],
        dtype=torch.float32,
        device=replay_device,
    )
    ts = torch.as_tensor(
        arrays["source_time"],
        dtype=torch.float32,
        device=replay_device,
    )
    es = torch.as_tensor(
        arrays["source_event"],
        dtype=torch.float32,
        device=replay_device,
    )
    Xt = torch.as_tensor(
        arrays["X_target_train"],
        dtype=torch.float32,
        device=replay_device,
    )
    Xv = torch.as_tensor(
        arrays["X_target_test"],
        dtype=torch.float32,
        device=replay_device,
    )
    mapping = torch.as_tensor(
        arrays["mapping_index"],
        dtype=torch.long,
        device=replay_device,
    )

    Xs_z, _, _, _ = c05.standardize_train_test(Xs)
    Xt_z, Xv_z, _, _ = c05.standardize_train_test(Xt, Xv)
    if Xv_z is None:
        raise RuntimeError("M6-prime target test standardization failed.")

    mapping_matrix = c05.build_mapping_matrix(
        mapping,
        int(Xt_z.shape[-1]),
    )
    Xt_aligned = c05.align_target_to_source(
        Xt_z,
        mapping_matrix,
    )
    Xv_aligned = c05.align_target_to_source(
        Xv_z,
        mapping_matrix,
    )

    cell_index = int(scenario_by_cell[r5_cell_id].cell_index)
    model_seed_base = (
        int(e2mod.MODEL_BASE_SEED)
        + cell_index * int(e2mod.CELL_SEED_STRIDE)
    )

    source_model, _ = c05.train_source_mlp(
        Xs_z,
        ts,
        es,
        epochs=c05.SOURCE_MLP_EPOCHS,
        lr=c05.SOURCE_MLP_LR,
        weight_decay=c05.SOURCE_MLP_WEIGHT_DECAY,
        init_seed=model_seed_base + 100,
    )
    c05.freeze_module(source_model)

    with torch.no_grad():
        replay_head = (
            source_model.head.detach().cpu().numpy().astype(np.float32)
        )
        replay_m1_train = (
            source_model(Xt_aligned).detach().cpu().numpy().astype(np.float32)
        )
        replay_m1_test = (
            source_model(Xv_aligned).detach().cpu().numpy().astype(np.float32)
        )

    # Numerical replay criteria: device may be CPU fallback, so use strong
    # numerical equivalence rather than bitwise equality.
    head_max_abs = float(
        np.max(np.abs(replay_head.astype(float) - saved_head_source))
    )
    m1_train_corr = pearson(
        replay_m1_train.reshape(-1),
        saved_m1_train.reshape(-1),
    )
    m1_test_corr = pearson(
        replay_m1_test.reshape(-1),
        saved_m1_test.reshape(-1),
    )
    m1_test_max_abs = float(
        np.max(np.abs(replay_m1_test.astype(float) - saved_m1_test))
    )

    replay_equivalence_pass = bool(
        head_max_abs <= 1e-5
        and m1_train_corr >= 0.999999
        and m1_test_corr >= 0.999999
        and m1_test_max_abs <= 1e-4
    )

    if not replay_equivalence_pass:
        raise RuntimeError(
            "M6-prime source-network replay failed numerical equivalence. "
            f"head_max_abs={head_max_abs:.3e}, "
            f"train_corr={m1_train_corr:.9f}, "
            f"test_corr={m1_test_corr:.9f}, "
            f"test_max_abs={m1_test_max_abs:.3e}"
        )

    residual_m4_t = torch.as_tensor(
        residual_m4,
        dtype=torch.float32,
        device=replay_device,
    )
    truth_gate_t = torch.as_tensor(
        truth_gate_np,
        dtype=torch.float32,
        device=replay_device,
    )

    with torch.no_grad():
        risk_train_m6p = (
            c05.predict_A3(
                source_model,
                truth_gate_t,
                residual_m4_t,
                Xt_z,
                mapping_matrix,
            )
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        risk_test_m6p = (
            c05.predict_A3(
                source_model,
                truth_gate_t,
                residual_m4_t,
                Xv_z,
                mapping_matrix,
            )
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    # Evaluate M6-prime using exact 05e3/05d metric functions.
    n_rep = len(arrays["replicate_seed"])
    c_m6p = np.full(n_rep, np.nan, dtype=float)
    ibs_m6p = np.full(n_rep, np.nan, dtype=float)
    risk_sd_m6p = np.std(risk_test_m6p, axis=1, ddof=0).astype(float)

    for r in range(n_rep):
        c_m6p[r] = e3mod.safe_uno(
            arrays["target_train_time"][r],
            arrays["target_train_event"][r],
            arrays["target_test_time"][r],
            arrays["target_test_event"][r],
            risk_test_m6p[r],
        )
        grid = d05.ibs_time_grid(
            arrays["target_train_time"][r],
            arrays["target_train_event"][r],
            arrays["target_test_time"][r],
        )
        ibs_m6p[r] = d05.safe_ibs(
            arrays["target_train_time"][r],
            arrays["target_train_event"][r],
            risk_train_m6p[r],
            arrays["target_test_time"][r],
            arrays["target_test_event"][r],
            risk_test_m6p[r],
            grid,
        )

    with np.load(metric_path, allow_pickle=False) as metric:
        uno_all = np.asarray(metric["uno_c"], dtype=float)
        ibs_all = np.asarray(metric["ibs"], dtype=float)
        risk_sd_all = np.asarray(metric["test_risk_sd"], dtype=float)

    m0i, m4i, m5i = 0, 4, 5
    c_b0 = uno_all[:, m0i]
    delta_m6p_b0 = c_m6p - c_b0
    nt_m6p = delta_m6p_b0 <= NEGATIVE_TRANSFER_DELTA
    cat_m6p = delta_m6p_b0 <= CATASTROPHIC_DELTA

    d_m6p_m4 = c_m6p - uno_all[:, m4i]
    d_m6p_m5 = c_m6p - uno_all[:, m5i]

    boot_m4 = paired_bootstrap(
        d_m6p_m4,
        seed=BOOTSTRAP_SEED + 500,
    )
    boot_m5 = paired_bootstrap(
        d_m6p_m5,
        seed=BOOTSTRAP_SEED + 501,
    )

    m6p_rows = [
        {
            "model": "M4_A3_SOFT_LEARNED",
            "diagnostic_role": "existing reference",
            "mean_uno_c": finite_mean(uno_all[:, m4i]),
            "mean_delta_c_vs_B0": finite_mean(uno_all[:, m4i] - c_b0),
            "mean_ibs": finite_mean(ibs_all[:, m4i]),
            "mean_test_risk_sd": finite_mean(risk_sd_all[:, m4i]),
            "negative_transfer_rate": float(
                np.mean((uno_all[:, m4i] - c_b0) <= NEGATIVE_TRANSFER_DELTA)
            ),
            "catastrophic_rate": float(
                np.mean((uno_all[:, m4i] - c_b0) <= CATASTROPHIC_DELTA)
            ),
        },
        {
            "model": "M5_A3_HARDENED_PREDICTION",
            "diagnostic_role": (
                "existing learned-gate hardening; residual is gate-consistent"
            ),
            "mean_uno_c": finite_mean(uno_all[:, m5i]),
            "mean_delta_c_vs_B0": finite_mean(uno_all[:, m5i] - c_b0),
            "mean_ibs": finite_mean(ibs_all[:, m5i]),
            "mean_test_risk_sd": finite_mean(risk_sd_all[:, m5i]),
            "negative_transfer_rate": float(
                np.mean((uno_all[:, m5i] - c_b0) <= NEGATIVE_TRANSFER_DELTA)
            ),
            "catastrophic_rate": float(
                np.mean((uno_all[:, m5i] - c_b0) <= CATASTROPHIC_DELTA)
            ),
        },
        {
            "model": "M6_PRIME_TRUE_GATE_FIXED_M4_RESIDUAL",
            "diagnostic_role": (
                "posthoc prediction-stage truth-gate substitution; "
                "M4 residual unchanged and not truth-gate-consistent"
            ),
            "mean_uno_c": finite_mean(c_m6p),
            "mean_delta_c_vs_B0": finite_mean(delta_m6p_b0),
            "mean_ibs": finite_mean(ibs_m6p),
            "mean_test_risk_sd": finite_mean(risk_sd_m6p),
            "negative_transfer_rate": float(np.mean(nt_m6p)),
            "catastrophic_rate": float(np.mean(cat_m6p)),
        },
    ]
    m6p_df = pd.DataFrame(m6p_rows)
    m6p_df["replay_device"] = str(replay_device)
    m6p_df["source_replay_head_max_abs_diff"] = head_max_abs
    m6p_df["source_replay_M1_test_corr"] = m1_test_corr
    m6p_df["source_replay_M1_test_max_abs_diff"] = m1_test_max_abs
    m6p_df["M6prime_minus_M4_mean_delta_uno"] = boot_m4["mean"]
    m6p_df["M6prime_minus_M4_ci_low"] = boot_m4["ci_low"]
    m6p_df["M6prime_minus_M4_ci_high"] = boot_m4["ci_high"]
    m6p_df["M6prime_minus_M5_mean_delta_uno"] = boot_m5["mean"]
    m6p_df["M6prime_minus_M5_ci_low"] = boot_m5["ci_low"]
    m6p_df["M6prime_minus_M5_ci_high"] = boot_m5["ci_high"]
    m6p_df["can_change_H4_status"] = False
    m6p_df["learned_vs_true_gate_optimality_claim_allowed"] = False

    m6p_df.to_csv(M6P_DIAGNOSTIC, sep="\t", index=False)

    if replay_device.type == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Q4 predeclared branch selection.
    # ------------------------------------------------------------------
    if h1_primary_class == "BEYOND_SIGN_INVERSION":
        title_branch = "BRANCH_B_REPRESENTATION_RETARGETING"
        prewritten_frame = (
            "A frozen cross-species representation retains target-relevant "
            "prognostic structure beyond simple sign inversion, while "
            "heterogeneous partial transportability remains a distinct "
            "transfer challenge."
        )
    elif (
        h1_primary_class == "SIGN_INVERSION_EQUIVALENT"
        and valley_support
    ):
        title_branch = "BRANCH_A_HETEROGENEITY_VALLEY"
        prewritten_frame = (
            "Under event scarcity, homogeneous cross-species prognostic "
            "relationships at either extreme are comparatively tractable, "
            "whereas heterogeneous partial transportability is the difficult "
            "regime; global sign reversal can be corrected largely by a "
            "one-dimensional head inversion."
        )
    else:
        title_branch = "NO_PREDECLARED_TITLE_BRANCH_CONFIRMED"
        prewritten_frame = (
            "No predeclared 05e3b headline branch satisfied its frozen rule."
        )

    summary = {
        "script_version": SCRIPT_VERSION,
        "audit_version": AUDIT_VERSION,
        "status": "PASS",
        "scientific_status": (
            "PASS_FINAL_POSTHOLD_SIMULATION_MECHANISM_AUDIT_COMPLETE"
        ),
        "run_started_utc": started,
        "run_finished_utc": now_utc(),
        "audit_contract_sha256": audit_contract_hash,

        "H1_sign_inversion": {
            "primary_cell": "29-event R5",
            "classification": h1_primary_class,
            "mean_delta_C_M2_minus_negM1": float(
                primary_h1["mean_delta_C_M2_minus_negM1"]
            ),
            "bootstrap_ci_low": float(
                primary_h1["bootstrap_ci_low"]
            ),
            "bootstrap_ci_high": float(
                primary_h1["bootstrap_ci_high"]
            ),
            "median_risk_corr_M2_vs_M1": float(
                primary_h1["median_pearson_risk_M2_vs_M1"]
            ),
            "median_head_cosine_M2_vs_source": float(
                primary_h1["median_head_cosine_M2_vs_source"]
            ),
            "median_head_orthogonal_fraction": float(
                primary_h1["median_head_orthogonal_fraction"]
            ),
            "R0_directional_positive_control_pass": (
                r0_directional_control_pass
            ),
        },

        "discrete_regime_shape": {
            "primary_event_budget": 29,
            "classification": (
                "DISCRETE_HETEROGENEITY_VALLEY_SUPPORTS"
                if valley_support
                else "DOES_NOT_SUPPORT_DISCRETE_HETEROGENEITY_VALLEY"
            ),
            "R4_key_contrast_positive_CI": bool(
                valley_df.loc[
                    valley_df["regime_other"] == REGIME_R4,
                    "strictly_positive_CI",
                ].iloc[0]
            ),
            "directional_replication_at_10_40": valley_direction_by_budget,
            "continuous_U_shape_claim_allowed": False,
        },

        "M6prime": {
            "status": "POSTHOC_FIXED_RESIDUAL_GATE_SUBSTITUTION_COMPLETE",
            "primary_cell": "29-event R5",
            "replay_device": str(replay_device),
            "source_replay_numerical_equivalence_pass": replay_equivalence_pass,
            "M6prime_minus_M4_mean_delta_uno": boot_m4["mean"],
            "M6prime_minus_M4_ci": [
                boot_m4["ci_low"],
                boot_m4["ci_high"],
            ],
            "M6prime_minus_M5_mean_delta_uno": boot_m5["mean"],
            "M6prime_minus_M5_ci": [
                boot_m5["ci_low"],
                boot_m5["ci_high"],
            ],
            "can_change_H4_status": False,
            "learned_vs_true_gate_optimality_claim_allowed": False,
        },

        "predeclared_interpretation_branch": title_branch,
        "prewritten_frame": prewritten_frame,

        "future_rate_reduction_policy_sha256": sha256_file(
            FUTURE_THRESHOLD_POLICY
        ),

        "stop_rule": (
            "NO_MORE_SIMULATION_MODEL_BRANCHES_OR_DIAGNOSTICS_BEFORE_METHODS_DRAFT"
        ),

        "human_outcomes_read": False,
        "A6_reopened": False,
        "model_selection": False,

        "artifact_hashes": {
            "final_simulation_audit_contract.json": sha256_file(
                AUDIT_CONTRACT
            ),
            "H1_sign_inversion_replicate_metrics.tsv": sha256_file(
                H1_REPLICATE
            ),
            "H1_sign_inversion_cell_summary.tsv": sha256_file(
                H1_SUMMARY
            ),
            "newseed_regime_shape_table.tsv": sha256_file(
                REGIME_TABLE
            ),
            "newseed_heterogeneity_valley_contrasts.tsv": sha256_file(
                VALLEY_CONTRASTS
            ),
            "M6prime_fixed_residual_R5_diagnostic.tsv": sha256_file(
                M6P_DIAGNOSTIC
            ),
            "future_rate_reduction_feasibility_policy.json": sha256_file(
                FUTURE_THRESHOLD_POLICY
            ),
        },
    }

    write_json(SUMMARY_JSON, summary)

    # ------------------------------------------------------------------
    # Console.
    # ------------------------------------------------------------------
    print("=" * 120)
    print("05e3b H1 SIGN-INVERSION AUDIT")
    print("=" * 120)
    print(
        h1_summary_df[
            [
                "target_events",
                "transfer_regime",
                "mean_C_M1",
                "mean_C_neg_M1",
                "mean_C_M2",
                "mean_delta_C_M2_minus_negM1",
                "bootstrap_ci_low",
                "bootstrap_ci_high",
                "prediction_level_classification",
                "median_pearson_risk_M2_vs_M1",
                "median_head_cosine_M2_vs_source",
                "median_head_orthogonal_fraction",
            ]
        ].to_string(index=False)
    )
    print()
    print(
        f"R0 directional geometry/risk positive control @29: "
        f"{'PASS' if r0_directional_control_pass else 'FAIL'}"
    )

    print()
    print("=" * 120)
    print("05e3b NEW-SEED DISCRETE REGIME-SHAPE TABLE")
    print("=" * 120)
    print(regime_df.to_string(index=False))

    print()
    print("29-event R2 valley contrasts [other - R2, using deltaC(M2-B0)]:")
    print(valley_df.to_string(index=False))
    print()
    print(
        "Discrete heterogeneity-valley status: "
        + (
            "DISCRETE_HETEROGENEITY_VALLEY_SUPPORTS"
            if valley_support
            else "DOES_NOT_SUPPORT_DISCRETE_HETEROGENEITY_VALLEY"
        )
    )

    print()
    print("=" * 120)
    print("05e3b M6-PRIME FIXED-RESIDUAL GATE-SUBSTITUTION DIAGNOSTIC")
    print("=" * 120)
    print(m6p_df.to_string(index=False))
    print()
    print(
        "Interpretation guardrail: M6-prime truth gate is substituted into a "
        "residual trained under M4's learned soft gate. "
        "Do NOT claim learned-vs-true gate optimality."
    )

    print()
    print("=" * 120)
    print("05e3b FINAL INTERPRETATION BRANCH")
    print("=" * 120)
    print(f"H1 classification: {h1_primary_class}")
    print(
        "Regime-shape classification: "
        + (
            "DISCRETE_HETEROGENEITY_VALLEY_SUPPORTS"
            if valley_support
            else "DOES_NOT_SUPPORT_DISCRETE_HETEROGENEITY_VALLEY"
        )
    )
    print(f"Predeclared branch: {title_branch}")
    print(f"Prewritten frame: {prewritten_frame}")
    print()
    print("A6 reopened: NO")
    print("Human outcomes read: NO")
    print("Model selection performed: NO")
    print(
        "Further simulation diagnostics before methods draft: FORBIDDEN BY 05e3b STOP RULE"
    )
    print("=" * 120)
    print(
        "05e3b: PASS_FINAL_POSTHOLD_SIMULATION_MECHANISM_AUDIT_COMPLETE"
    )
    print("=" * 120)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 120, file=sys.stderr)
        print("05e3b final mechanism audit: FAIL", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print("=" * 120, file=sys.stderr)
        raise
