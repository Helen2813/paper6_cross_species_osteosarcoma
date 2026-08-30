#!/usr/bin/env python3
"""
Paper 6 - global-trust / abstention feasibility diagnostic after frozen A2/A3 HOLD.

This is NOT an A6 model-development stage.
It asks whether an abstention-aware transfer branch is worth opening at all.

Known state entering 05d0c
---------------------------
- frozen 05d: HOLD_NO_SAFE_SELECTABLE_ARCHITECTURE
- A5: CLOSE
- 05d0 v1: structural invariants PASS
- 05d0a: threshold feasible; A3 risk-scale/composition pathology documented
- 05d0b: manual Uno replay exact after frozen float32 serialization

Two questions
-------------
A. Continuous-leakage question
   In R5, among replicates with ZERO thresholded harmful sign-flip borrows
   (all harmful gates < 0.5), does catastrophic failure decrease as the
   CONTINUOUS total gate mass on harmful sign-flipped modules approaches zero?

B. Global-trust feasibility
   Can one simple outcome-sparse compatibility statistic, available from the
   TARGET-TRAINING data only, discriminate when source transfer is likely to
   help versus harm?

Primary compatibility statistic
-------------------------------
A frozen source LINEAR Cox score is applied to the target-training expression
after target-training-only standardization. No target outcome is used to fit
that source score. The statistic is Uno C of that frozen source score evaluated
on the target-training patients themselves:

    source_target_train_C

This is intentionally simple and noisy. The point of the experiment is to learn
whether 10-40 target events contain enough information for a future global
trust/abstention layer.

Labels
------
Mechanistic (secondary):
    compatible  = R0-R3
    incompatible = R4-R5

Operational (PRIMARY):
    for each frozen transfer engine A1 and A2 separately,
    beneficial = independent synthetic TEST deltaC(engine-B0) >= +0.02
    harmful    = independent synthetic TEST deltaC(engine-B0) <= -0.02
    neutral    = excluded from operational AUROC

Decision rule, frozen BEFORE feasibility results are read
---------------------------------------------------------
Primary information budget = 29 target events.

GO_CURRENT_PAPER_A6:
    operational AUROC >= 0.70 at 29 events for A1 OR A2.

LIMITED_FUTURE_METHOD_ONLY:
    no engine reaches 0.70 at 29 events, but an engine has
    0.65 <= AUROC < 0.70 at 29 AND AUROC >= 0.70 at 40 events.
    This does NOT permit A6 human evaluation in current Paper 6.

STOP_A6_FOR_PAPER6:
    otherwise.

If both A1 and A2 reach 0.70 at 29, both remain eligible bases for a future
prospective A6 contract; this diagnostic does not select between them.

Trivial comparator
------------------
At a fixed event budget, an event-count-only rule contains no within-budget
information, so its AUROC is 0.50 by construction. In addition, this script
evaluates fixed event-count-only abstention policies:
    use engine only when target events >= N, else B0
for N in {5,10,15,20,29,40,41}; N=41 means always abstain.

This determines whether a future trust signal would add anything beyond the
trivial strategy "do not transfer when n_events is small."

No model fitting.
No classifier training.
No hyperparameter tuning.
No human data.
No GPU.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import roc_auc_score
except ImportError as exc:
    raise ImportError("05d0c requires scikit-learn.") from exc

try:
    from sksurv.metrics import concordance_index_ipcw
    from sksurv.util import Surv
except ImportError as exc:
    raise ImportError("05d0c requires scikit-survival.") from exc


SCRIPT_VERSION = "05d0c-global-trust-feasibility-v1-no-cli"

ROOT = Path(__file__).resolve().parents[1]

# Frozen scenario/generator.
A_DIR = ROOT / "results" / "simulation_contract" / "05a"
A_SCENARIOS = A_DIR / "simulation_scenario_registry.tsv"

B_DIR = ROOT / "simulations" / "05b"
B_GENERATOR_CONTRACT = B_DIR / "generator_contract.json"

# Frozen model outputs and metrics.
C_DIR = ROOT / "results" / "simulation_model_matrix" / "05c"
C_MANIFEST = C_DIR / "scenario_output_manifest.tsv"

D_DIR = ROOT / "results" / "simulation_phase_diagram" / "05d"
D_METRIC_MANIFEST = D_DIR / "scenario_metric_manifest.tsv"
D_SUMMARY = D_DIR / "summary.json"

# Prior diagnostics.
D0_V1_DIR = ROOT / "results" / "simulation_phase_diagram_diagnostics" / "05d0"
D0_V1_SUMMARY = D0_V1_DIR / "summary.json"

D0A_DIR = ROOT / "results" / "simulation_phase_diagram_diagnostics" / "05d0a"
D0A_SUMMARY = D0A_DIR / "summary.json"

D0B_DIR = ROOT / "results" / "simulation_phase_diagram_diagnostics" / "05d0b"
D0B_SUMMARY = D0B_DIR / "summary.json"

OUT_DIR = ROOT / "results" / "simulation_phase_diagram_diagnostics" / "05d0c"
CHECKPOINT_DIR = OUT_DIR / "scenario_checkpoints"
CHECKPOINT_MANIFEST_DIR = OUT_DIR / "scenario_manifests"

for d in [OUT_DIR, CHECKPOINT_DIR, CHECKPOINT_MANIFEST_DIR]:
    d.mkdir(parents=True, exist_ok=True)

FEASIBILITY_CONTRACT = OUT_DIR / "global_trust_feasibility_contract.json"
TRUST_REPLICATES = OUT_DIR / "global_trust_replicate_statistics.tsv"
AUROC_BY_BUDGET = OUT_DIR / "global_trust_AUROC_by_event_budget.tsv"
EVENT_POLICY = OUT_DIR / "event_count_only_abstention_policies.tsv"
GATE_MASS_STRATA = OUT_DIR / "R5_continuous_harmful_gate_mass_stratification.tsv"
SCENARIO_MANIFEST = OUT_DIR / "scenario_checkpoint_manifest.tsv"
SUMMARY_JSON = OUT_DIR / "summary.json"

EXPECTED_05D_STATUS = "HOLD_NO_SAFE_SELECTABLE_ARCHITECTURE"
EXPECTED_D0_V1_STATUS = (
    "PASS_NO_STRUCTURAL_IMPLEMENTATION_BUG_DETECTED_IN_BOUNDED_AUDIT"
)
EXPECTED_D0B_STATUS = (
    "PASS_MANUAL_UNO_REPLAY_EXACT_AFTER_FROZEN_FLOAT32_SERIALIZATION"
)

EXPECTED_SCENARIOS = 180
EXPECTED_REPLICATES = 21600
EXPECTED_MODULES = 50
EXPECTED_TEST_N = 500

EVENT_BUDGETS = [5, 10, 15, 20, 29, 40]
PRIMARY_EVENT_BUDGET = 29
LIMITED_EVENT_BUDGET = 40

OPERATIONAL_GO_AUROC = 0.70
OPERATIONAL_LIMITED_LOW = 0.65
OPERATIONAL_LIMITED_HIGH = 0.70

BENEFIT_DELTA = +0.02
HARM_DELTA = -0.02

BASE_ENGINES = ["A1", "A2"]
ALL_MODELS = [
    "B0", "B4", "A0", "A1", "A2", "A3", "A4",
    "A3_NO_EVOLUTION_PRIOR",
    "A3_NO_TARGET_OVERRIDE",
    "A3_PERMUTED_PRIOR",
    "A3_MAPPING_PERMUTATION",
    "A3_SOURCE_OUTCOME_PERMUTATION",
]
MODEL_INDEX = {m: i for i, m in enumerate(ALL_MODELS)}

GATE_THRESHOLD = 0.50

# Fixed harmful continuous-mass bins. These are absolute source-gate mass over
# causal nontransportable sign-flipped modules.
HARM_MASS_BINS = [
    (-1e-12, 0.10, "[0,0.10]"),
    (0.10, 0.25, "(0.10,0.25]"),
    (0.25, 0.50, "(0.25,0.50]"),
    (0.50, 1.00, "(0.50,1.00]"),
    (1.00, 2.00, "(1.00,2.00]"),
    (2.00, 3.00, "(2.00,3.00]"),
    (3.00, 5.00 + 1e-12, "(3.00,5.00]"),
]

EVENT_ONLY_THRESHOLDS = [5, 10, 15, 20, 29, 40, 41]

CHECKPOINT_SCHEMA_VERSION = 1


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def atomic_savez_compressed(path: Path, **arrays: np.ndarray) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")

    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(name)
    sys.modules[name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
        raise

    return module


def make_surv(t: np.ndarray, e: np.ndarray) -> np.ndarray:
    return Surv.from_arrays(
        event=np.asarray(e, dtype=bool),
        time=np.asarray(t, dtype=float),
    )


def training_uno(
    time_values: np.ndarray,
    event_values: np.ndarray,
    risk: np.ndarray,
) -> float:
    """
    Uno C for a frozen source score evaluated on target-training outcomes.

    The score itself is NOT fitted on target outcomes.
    """
    t = np.asarray(time_values, dtype=float)
    e = np.asarray(event_values, dtype=bool)
    r = np.asarray(risk, dtype=float)

    event_times = t[e]
    if len(event_times) < 2:
        return float("nan")

    tau = float(np.quantile(event_times, 0.80))
    tau = min(tau, float(np.max(t)))
    tau = float(np.nextafter(tau, -np.inf))

    if not np.isfinite(tau) or tau <= float(np.min(t)):
        return float("nan")

    try:
        c = float(
            concordance_index_ipcw(
                make_surv(t, e),
                make_surv(t, e),
                r,
                tau=tau,
            )[0]
        )
        return c if np.isfinite(c) else float("nan")
    except Exception:
        return float("nan")


def standardize_train(x: np.ndarray) -> np.ndarray:
    mean = np.mean(x, axis=0, keepdims=True)
    sd = np.std(x, axis=0, ddof=0, keepdims=True)
    sd = np.where(sd > 1e-6, sd, 1.0)
    return (x - mean) / sd


def mapped_source_beta(
    beta_source_fit: np.ndarray,
    mapping_index: np.ndarray,
) -> np.ndarray:
    out = np.zeros(EXPECTED_MODULES, dtype=float)

    for target_j in range(EXPECTED_MODULES):
        source_j = int(mapping_index[target_j])
        if source_j >= 0:
            out[target_j] = float(beta_source_fit[source_j])

    return out


def safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)

    keep = np.isfinite(score) & np.isin(y, [0, 1])
    y = y[keep]
    score = score[keep]

    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")

    try:
        return float(roc_auc_score(y, score))
    except Exception:
        return float("nan")


def finite_mean(x: Iterable[float]) -> float:
    arr = np.asarray(list(x), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) else float("nan")


def finite_median(x: Iterable[float]) -> float:
    arr = np.asarray(list(x), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if len(arr) else float("nan")


def mass_bin(value: float) -> str:
    x = float(value)
    for lo, hi, label in HARM_MASS_BINS:
        if lo < x <= hi or (label == "[0,0.10]" and 0.0 <= x <= hi):
            return label
    return "OUTSIDE"


def checkpoint_paths(scenario_id: str) -> Tuple[Path, Path]:
    return (
        CHECKPOINT_DIR / f"{scenario_id}.npz",
        CHECKPOINT_MANIFEST_DIR / f"{scenario_id}.json",
    )


def valid_checkpoint(
    scenario_id: str,
    c_hash: str,
    d_hash: str,
    contract_hash: str,
) -> bool:
    cp, mp = checkpoint_paths(scenario_id)

    if not cp.exists() or not mp.exists():
        return False

    meta = read_json(mp)

    return (
        str(meta.get("status")) == "PASS"
        and str(meta.get("05c_output_sha256")) == c_hash
        and str(meta.get("05d_metric_sha256")) == d_hash
        and str(meta.get("feasibility_contract_sha256")) == contract_hash
        and str(meta.get("checkpoint_sha256")) == sha256_file(cp)
    )


def main() -> None:
    started = now_utc()

    print("=" * 120)
    print("Paper 6 - 05d0c global-trust / abstention feasibility diagnostic")
    print("=" * 120)
    print(f"Script version: {SCRIPT_VERSION}")
    print()
    print("Scope:")
    print("  A6 model fitting: NO")
    print("  Compatibility classifier training: NO")
    print("  Existing 21,600 simulations used for diagnosis: YES")
    print("  Human outcomes read: NO")
    print("  GPU execution: NO")
    print()

    for p in [
        A_SCENARIOS,
        B_GENERATOR_CONTRACT,
        C_MANIFEST,
        D_METRIC_MANIFEST,
        D_SUMMARY,
        D0_V1_SUMMARY,
        D0A_SUMMARY,
        D0B_SUMMARY,
    ]:
        require_file(p)

    d_summary = read_json(D_SUMMARY)
    d0_v1 = read_json(D0_V1_SUMMARY)
    d0a = read_json(D0A_SUMMARY)
    d0b = read_json(D0B_SUMMARY)

    if str(d_summary.get("scientific_status")) != EXPECTED_05D_STATUS:
        raise RuntimeError("05d is not in expected frozen HOLD state.")

    if str(d0_v1.get("scientific_status")) != EXPECTED_D0_V1_STATUS:
        raise RuntimeError("05d0 v1 structural diagnostic is not PASS.")

    if str(d0b.get("scientific_status")) != EXPECTED_D0B_STATUS:
        raise RuntimeError("05d0b manual Uno precision audit is not PASS.")

    # 05d0a is expected to be HOLD only because of the now-resolved 1e-8 vs
    # float32 storage precision criterion.
    if not bool(d0a.get("implementation_invariants_pass")):
        raise RuntimeError("05d0a implementation invariants did not pass.")
    if not bool(d0a.get("frozen_selection_metric_replay_pass")):
        raise RuntimeError("05d0a frozen selection replay did not pass.")

    scenarios = pd.read_csv(A_SCENARIOS, sep="\t")
    c_manifest = pd.read_csv(C_MANIFEST, sep="\t")
    d_manifest = pd.read_csv(D_METRIC_MANIFEST, sep="\t")

    if len(scenarios) != EXPECTED_SCENARIOS:
        raise RuntimeError("Scenario registry is not 180 rows.")

    generator_contract = read_json(B_GENERATOR_CONTRACT)
    generator_path = ROOT / str(
        generator_contract["authoritative_generator_script"]
    )
    require_file(generator_path)

    if sha256_file(generator_path) != str(
        generator_contract["authoritative_generator_script_sha256"]
    ):
        raise RuntimeError("05b generator SHA256 mismatch.")

    gen05b = load_module(
        generator_path,
        "paper6_05b_generator_for_05d0c",
    )

    # ------------------------------------------------------------------
    # Freeze GO / LIMITED / STOP before reading any new compatibility values.
    # ------------------------------------------------------------------
    feasibility_contract = {
        "script_version": SCRIPT_VERSION,
        "status": "FROZEN_BEFORE_GLOBAL_TRUST_FEASIBILITY_VALUES_READ",
        "created_utc": now_utc(),
        "chronology": {
            "05d_results_known": True,
            "05d0_v1_results_known": True,
            "05d0a_results_known": True,
            "05d0b_precision_resolution_known": True,
            "global_trust_AUROC_not_yet_computed": True,
            "continuous_harmful_gate_mass_stratification_not_yet_computed": True,
        },
        "compatibility_signal": {
            "name": "source_target_train_UnoC",
            "definition": (
                "Uno C of the frozen 05c source-linear Cox score on target-training "
                "patients after target-training-only expression standardization. "
                "The source score is not fit to target outcomes."
            ),
            "only_target_training_outcomes_used": True,
            "test_outcomes_used_in_signal": False,
        },
        "mechanistic_label": {
            "positive": "R0,R1,R2,R3",
            "negative": "R4,R5",
            "role": "SECONDARY_FEASIBILITY",
        },
        "operational_label": {
            "engines": BASE_ENGINES,
            "positive": "independent test deltaC(engine-B0) >= +0.02",
            "negative": "independent test deltaC(engine-B0) <= -0.02",
            "neutral": "excluded from AUROC",
            "role": "PRIMARY_FEASIBILITY",
        },
        "event_budgets": EVENT_BUDGETS,
        "primary_event_budget": PRIMARY_EVENT_BUDGET,
        "decision": {
            "GO_CURRENT_PAPER_A6": (
                "At least one of A1/A2 has operational AUROC >=0.70 at 29 events."
            ),
            "LIMITED_FUTURE_METHOD_ONLY": (
                "No A1/A2 operational AUROC reaches 0.70 at 29, but at least one "
                "engine has 0.65<=AUROC<0.70 at 29 AND AUROC>=0.70 at 40. "
                "This does not permit A6 human evaluation in current Paper 6."
            ),
            "STOP_A6_FOR_PAPER6": "otherwise",
            "if_both_engines_pass": (
                "both remain eligible; 05d0c does not choose an A6 base engine"
            ),
        },
        "trivial_comparator": {
            "within_budget_event_count_only_AUROC": 0.50,
            "policy_thresholds": EVENT_ONLY_THRESHOLDS,
            "policy": (
                "use frozen A1/A2 when target_events>=N, else exact frozen B0"
            ),
        },
        "continuous_gate_mass_test": {
            "population": "R5_MISLEADING_SOURCE",
            "primary_subset": (
                "harmful_signflip_borrow_count_at_gate_0.5 == 0"
            ),
            "mass": (
                "sum of continuous A3 gate weights over causal, nontransportable, "
                "sign-flipped modules"
            ),
            "fixed_mass_bins": [x[2] for x in HARM_MASS_BINS],
            "interpretation": (
                "If catastrophic rate remains high in the near-zero-mass bin, "
                "continuous harmful leakage cannot by itself explain baseline R5 failure."
            ),
        },
        "A6_guardrail": (
            "05d0c GO only permits writing a NEW prospective A6 contract. It does "
            "not permit model fitting on human outcomes. Existing 21,600 simulations "
            "cannot be the final unbiased A6 benchmark."
        ),
    }
    write_json(FEASIBILITY_CONTRACT, feasibility_contract)
    contract_hash = sha256_file(FEASIBILITY_CONTRACT)

    print("05d0c GO/LIMITED/STOP rule frozen before feasibility read: PASS")
    print(f"Feasibility contract SHA256: {contract_hash}")
    print()

    scenario_by_id = scenarios.set_index("scenario_id", drop=False)
    c_by_id = {
        str(r.scenario_id): r
        for r in c_manifest.itertuples(index=False)
    }
    d_by_id = {
        str(r.scenario_id): r
        for r in d_manifest.itertuples(index=False)
    }

    # ------------------------------------------------------------------
    # Scenario checkpoint calculation.
    # ------------------------------------------------------------------
    manifest_rows: List[Dict[str, Any]] = []

    for i, scenario in scenarios.reset_index(drop=True).iterrows():
        sid = str(scenario["scenario_id"])
        crow = c_by_id[sid]
        drow = d_by_id[sid]

        cpath = ROOT / str(crow.output_path)
        dpath = ROOT / str(drow.metric_output_path)

        require_file(cpath)
        require_file(dpath)

        c_hash = sha256_file(cpath)
        d_hash = sha256_file(dpath)

        if c_hash != str(crow.output_sha256):
            raise RuntimeError(f"{sid}: 05c output hash mismatch.")
        if d_hash != str(drow.metric_output_sha256):
            raise RuntimeError(f"{sid}: 05d metric hash mismatch.")

        cp_path, cp_manifest_path = checkpoint_paths(sid)

        if not valid_checkpoint(
            sid, c_hash, d_hash, contract_hash
        ):
            with np.load(cpath, allow_pickle=False) as cdata, np.load(
                dpath, allow_pickle=False
            ) as ddata:
                seeds = np.asarray(
                    cdata["replicate_seed"], dtype=np.int64
                )
                n_rep = len(seeds)

                uno = np.asarray(ddata["uno_c"], dtype=float)
                delta = np.asarray(
                    ddata["delta_c_vs_B0"], dtype=float
                )

                gate = np.asarray(cdata["gate_A3"], dtype=float)
                causal = np.asarray(
                    cdata["causal_mask"], dtype=bool
                )
                transportable = np.asarray(
                    cdata["transportable_mask"], dtype=bool
                )
                beta_source_truth = np.asarray(
                    cdata["beta_source_truth"], dtype=float
                )
                beta_target_truth = np.asarray(
                    cdata["beta_target_truth"], dtype=float
                )
                mapping = np.asarray(
                    cdata["mapping_index"], dtype=int
                )
                beta_source_linear = np.asarray(
                    cdata["beta_source_linear_fit"], dtype=float
                )

                train_time = np.asarray(
                    cdata["target_train_time"], dtype=float
                )
                train_event = np.asarray(
                    cdata["target_train_event"], dtype=np.uint8
                )

                compatibility = np.full(
                    n_rep, np.nan, dtype=np.float32
                )
                harmful_count = np.zeros(
                    n_rep, dtype=np.int16
                )
                harmful_mass = np.zeros(
                    n_rep, dtype=np.float32
                )

                for r, seed in enumerate(seeds):
                    generated = gen05b.generate_full_replicate(
                        int(seed),
                        scenario.to_dict(),
                        target_test_n=EXPECTED_TEST_N,
                    )

                    # Exact identity checks before use.
                    if not np.array_equal(
                        generated["mapping_index"].astype(np.int16),
                        cdata["mapping_index"][r],
                    ):
                        raise RuntimeError(
                            f"{sid} rep {r}: mapping replay mismatch."
                        )

                    xtr = np.asarray(
                        generated["X_target_train"], dtype=float
                    )
                    xtr_z = standardize_train(xtr)

                    beta_prior = mapped_source_beta(
                        beta_source_linear[r],
                        mapping[r],
                    )
                    source_train_risk = xtr_z @ beta_prior

                    compatibility[r] = np.float32(
                        training_uno(
                            train_time[r],
                            train_event[r],
                            source_train_risk,
                        )
                    )

                    harmful = (
                        causal[r]
                        & (~transportable[r])
                        & (
                            beta_source_truth[r]
                            * beta_target_truth[r]
                            < 0
                        )
                        & (np.abs(beta_source_truth[r]) > 0)
                        & (np.abs(beta_target_truth[r]) > 0)
                    )

                    harmful_count[r] = int(
                        np.sum(
                            harmful
                            & (gate[r] >= GATE_THRESHOLD)
                        )
                    )
                    harmful_mass[r] = np.float32(
                        np.sum(gate[r, harmful])
                    )

                regime = str(scenario["transfer_regime"])
                mechanistic_label = np.full(
                    n_rep,
                    1 if regime in {
                        "R0_FULLY_TRANSPORTABLE",
                        "R1_MOSTLY_TRANSPORTABLE",
                        "R2_PARTIALLY_TRANSPORTABLE",
                        "R3_WEAKLY_TRANSPORTABLE",
                    } else 0,
                    dtype=np.int8,
                )

                op_label_a1 = np.full(n_rep, -1, dtype=np.int8)
                op_label_a2 = np.full(n_rep, -1, dtype=np.int8)

                for engine, label_array in [
                    ("A1", op_label_a1),
                    ("A2", op_label_a2),
                ]:
                    d = delta[:, MODEL_INDEX[engine]]
                    label_array[d >= BENEFIT_DELTA] = 1
                    label_array[d <= HARM_DELTA] = 0

                atomic_savez_compressed(
                    cp_path,
                    replicate_seed=seeds,
                    compatibility=compatibility,
                    mechanistic_label=mechanistic_label,
                    operational_label_A1=op_label_a1,
                    operational_label_A2=op_label_a2,
                    harmful_signflip_threshold_count=harmful_count,
                    harmful_signflip_gate_mass=harmful_mass,
                    delta_c_A1=delta[:, MODEL_INDEX["A1"]].astype(
                        np.float32
                    ),
                    delta_c_A2=delta[:, MODEL_INDEX["A2"]].astype(
                        np.float32
                    ),
                    delta_c_A3=delta[:, MODEL_INDEX["A3"]].astype(
                        np.float32
                    ),
                    uno_c_B0=uno[:, MODEL_INDEX["B0"]].astype(
                        np.float32
                    ),
                    uno_c_A1=uno[:, MODEL_INDEX["A1"]].astype(
                        np.float32
                    ),
                    uno_c_A2=uno[:, MODEL_INDEX["A2"]].astype(
                        np.float32
                    ),
                    uno_c_A3=uno[:, MODEL_INDEX["A3"]].astype(
                        np.float32
                    ),
                )

                cp_meta = {
                    "status": "PASS",
                    "schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "created_utc": now_utc(),
                    "scenario_id": sid,
                    "05c_output_sha256": c_hash,
                    "05d_metric_sha256": d_hash,
                    "feasibility_contract_sha256": contract_hash,
                    "checkpoint_sha256": sha256_file(cp_path),
                    "replicates": n_rep,
                }
                write_json(cp_manifest_path, cp_meta)

        manifest_rows.append(
            {
                "scenario_id": sid,
                "checkpoint_path": str(cp_path.relative_to(ROOT)),
                "checkpoint_sha256": sha256_file(cp_path),
                "replicates": int(scenario["replicates"]),
            }
        )

        if (i + 1) % 20 == 0 or i + 1 == EXPECTED_SCENARIOS:
            print(
                f"  global-trust feasibility scenarios: "
                f"{i+1}/{EXPECTED_SCENARIOS}"
            )

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(
        SCENARIO_MANIFEST,
        sep="\t",
        index=False,
    )

    # ------------------------------------------------------------------
    # Combine replicate diagnostics.
    # ------------------------------------------------------------------
    replicate_rows: List[Dict[str, Any]] = []

    for row in manifest_df.itertuples(index=False):
        sid = str(row.scenario_id)
        scenario = scenario_by_id.loc[sid]
        cp_path = ROOT / str(row.checkpoint_path)

        with np.load(cp_path, allow_pickle=False) as data:
            n_rep = len(data["replicate_seed"])

            for r in range(n_rep):
                replicate_rows.append(
                    {
                        "scenario_id": sid,
                        "replicate": r,
                        "target_events": int(
                            scenario["target_events"]
                        ),
                        "transfer_regime": str(
                            scenario["transfer_regime"]
                        ),
                        "family": str(scenario["family"]),
                        "source_prior_state": str(
                            scenario["source_prior_state"]
                        ),
                        "compatibility_source_train_c": float(
                            data["compatibility"][r]
                        ),
                        "mechanistic_label": int(
                            data["mechanistic_label"][r]
                        ),
                        "operational_label_A1": int(
                            data["operational_label_A1"][r]
                        ),
                        "operational_label_A2": int(
                            data["operational_label_A2"][r]
                        ),
                        "harmful_signflip_threshold_count": int(
                            data[
                                "harmful_signflip_threshold_count"
                            ][r]
                        ),
                        "harmful_signflip_gate_mass": float(
                            data["harmful_signflip_gate_mass"][r]
                        ),
                        "delta_c_A1": float(data["delta_c_A1"][r]),
                        "delta_c_A2": float(data["delta_c_A2"][r]),
                        "delta_c_A3": float(data["delta_c_A3"][r]),
                        "uno_c_B0": float(data["uno_c_B0"][r]),
                        "uno_c_A1": float(data["uno_c_A1"][r]),
                        "uno_c_A2": float(data["uno_c_A2"][r]),
                        "uno_c_A3": float(data["uno_c_A3"][r]),
                    }
                )

    reps = pd.DataFrame(replicate_rows)
    reps.to_csv(TRUST_REPLICATES, sep="\t", index=False)

    # ------------------------------------------------------------------
    # AUROC by event budget.
    # ------------------------------------------------------------------
    auc_rows: List[Dict[str, Any]] = []

    for events in EVENT_BUDGETS:
        part = reps[reps["target_events"] == events].copy()

        mech_auc = safe_auc(
            part["mechanistic_label"].to_numpy(dtype=int),
            part["compatibility_source_train_c"].to_numpy(dtype=float),
        )

        auc_rows.append(
            {
                "target_events": events,
                "endpoint": "MECHANISTIC_R0_R3_vs_R4_R5",
                "engine": "NONE",
                "n_total": len(part),
                "n_positive": int(
                    np.sum(part["mechanistic_label"] == 1)
                ),
                "n_negative": int(
                    np.sum(part["mechanistic_label"] == 0)
                ),
                "n_neutral_excluded": 0,
                "source_train_compatibility_AUROC": mech_auc,
                "event_count_only_within_budget_AUROC": 0.50,
            }
        )

        for engine in BASE_ENGINES:
            label_col = f"operational_label_{engine}"
            labels = part[label_col].to_numpy(dtype=int)
            keep = np.isin(labels, [0, 1])

            op_auc = safe_auc(
                labels[keep],
                part.loc[
                    keep, "compatibility_source_train_c"
                ].to_numpy(dtype=float),
            )

            auc_rows.append(
                {
                    "target_events": events,
                    "endpoint": "OPERATIONAL_BENEFIT_vs_HARM",
                    "engine": engine,
                    "n_total": int(np.sum(keep)),
                    "n_positive": int(
                        np.sum(labels[keep] == 1)
                    ),
                    "n_negative": int(
                        np.sum(labels[keep] == 0)
                    ),
                    "n_neutral_excluded": int(
                        np.sum(labels == -1)
                    ),
                    "source_train_compatibility_AUROC": op_auc,
                    "event_count_only_within_budget_AUROC": 0.50,
                }
            )

    auc_df = pd.DataFrame(auc_rows)
    auc_df.to_csv(
        AUROC_BY_BUDGET,
        sep="\t",
        index=False,
    )

    # ------------------------------------------------------------------
    # Event-count-only abstention comparator policies.
    # Equal-scenario weighting, matching 05d aggregation philosophy.
    # ------------------------------------------------------------------
    scenario_policy_base = (
        reps.groupby(
            [
                "scenario_id",
                "target_events",
                "transfer_regime",
            ],
            as_index=False,
        )
        .agg(
            mean_B0=("uno_c_B0", "mean"),
            mean_A1=("uno_c_A1", "mean"),
            mean_A2=("uno_c_A2", "mean"),
            nt_A1=(
                "delta_c_A1",
                lambda x: float(
                    np.mean(np.asarray(x, dtype=float) <= HARM_DELTA)
                ),
            ),
            nt_A2=(
                "delta_c_A2",
                lambda x: float(
                    np.mean(np.asarray(x, dtype=float) <= HARM_DELTA)
                ),
            ),
            mean_delta_A1=("delta_c_A1", "mean"),
            mean_delta_A2=("delta_c_A2", "mean"),
        )
    )

    policy_rows: List[Dict[str, Any]] = []

    for engine in BASE_ENGINES:
        for threshold in EVENT_ONLY_THRESHOLDS:
            use_transfer = (
                scenario_policy_base["target_events"] >= threshold
            )

            engine_c = scenario_policy_base[
                f"mean_{engine}"
            ].to_numpy(dtype=float)
            b0_c = scenario_policy_base[
                "mean_B0"
            ].to_numpy(dtype=float)
            engine_delta = scenario_policy_base[
                f"mean_delta_{engine}"
            ].to_numpy(dtype=float)
            engine_nt = scenario_policy_base[
                f"nt_{engine}"
            ].to_numpy(dtype=float)

            policy_c = np.where(
                use_transfer,
                engine_c,
                b0_c,
            )
            policy_delta = np.where(
                use_transfer,
                engine_delta,
                0.0,
            )
            policy_nt = np.where(
                use_transfer,
                engine_nt,
                0.0,
            )

            beneficial_regime = scenario_policy_base[
                "transfer_regime"
            ].isin(
                [
                    "R0_FULLY_TRANSPORTABLE",
                    "R1_MOSTLY_TRANSPORTABLE",
                    "R2_PARTIALLY_TRANSPORTABLE",
                    "R3_WEAKLY_TRANSPORTABLE",
                ]
            ).to_numpy()

            policy_rows.append(
                {
                    "engine": engine,
                    "minimum_events_to_transfer": threshold,
                    "always_abstain": bool(threshold == 41),
                    "scenario_activation_fraction": float(
                        np.mean(use_transfer)
                    ),
                    "beneficial_regime_activation_fraction": float(
                        np.mean(use_transfer[beneficial_regime])
                    ),
                    "equal_scenario_mean_uno_c": float(
                        np.mean(policy_c)
                    ),
                    "equal_scenario_mean_delta_c_vs_B0": float(
                        np.mean(policy_delta)
                    ),
                    "aggregate_negative_transfer_rate": float(
                        np.mean(policy_nt)
                    ),
                    "mean_delta_in_beneficial_regimes": float(
                        np.mean(policy_delta[beneficial_regime])
                    ),
                    "meets_frozen_NT_0_10": bool(
                        np.mean(policy_nt) <= 0.10
                    ),
                }
            )

    policy_df = pd.DataFrame(policy_rows)
    policy_df.to_csv(EVENT_POLICY, sep="\t", index=False)

    # ------------------------------------------------------------------
    # Continuous harmful gate mass among ZERO thresholded harmful borrows.
    # ------------------------------------------------------------------
    r5_zero = reps[
        (reps["transfer_regime"] == "R5_MISLEADING_SOURCE")
        & (reps["harmful_signflip_threshold_count"] == 0)
    ].copy()

    r5_zero["harmful_gate_mass_bin"] = [
        mass_bin(x)
        for x in r5_zero[
            "harmful_signflip_gate_mass"
        ].to_numpy(dtype=float)
    ]

    gate_rows: List[Dict[str, Any]] = []

    ordered_labels = [x[2] for x in HARM_MASS_BINS]

    for label in ordered_labels:
        g = r5_zero[
            r5_zero["harmful_gate_mass_bin"] == label
        ]
        if len(g) == 0:
            gate_rows.append(
                {
                    "harmful_gate_mass_bin": label,
                    "n_replicates": 0,
                    "mean_harmful_gate_mass": float("nan"),
                    "catastrophic_rate_A3": float("nan"),
                    "mean_delta_c_A3": float("nan"),
                    "median_delta_c_A3": float("nan"),
                }
            )
            continue

        gate_rows.append(
            {
                "harmful_gate_mass_bin": label,
                "n_replicates": len(g),
                "mean_harmful_gate_mass": finite_mean(
                    g["harmful_signflip_gate_mass"]
                ),
                "catastrophic_rate_A3": float(
                    np.mean(g["delta_c_A3"] <= -0.05)
                ),
                "mean_delta_c_A3": finite_mean(g["delta_c_A3"]),
                "median_delta_c_A3": finite_median(
                    g["delta_c_A3"]
                ),
            }
        )

    gate_df = pd.DataFrame(gate_rows)
    gate_df.to_csv(
        GATE_MASS_STRATA,
        sep="\t",
        index=False,
    )

    # ------------------------------------------------------------------
    # Mechanical GO / LIMITED / STOP.
    # ------------------------------------------------------------------
    primary = auc_df[
        (auc_df["target_events"] == PRIMARY_EVENT_BUDGET)
        & (auc_df["endpoint"] == "OPERATIONAL_BENEFIT_vs_HARM")
    ].set_index("engine")

    limited = auc_df[
        (auc_df["target_events"] == LIMITED_EVENT_BUDGET)
        & (auc_df["endpoint"] == "OPERATIONAL_BENEFIT_vs_HARM")
    ].set_index("engine")

    eligible_current = []
    eligible_limited = []

    for engine in BASE_ENGINES:
        a29 = float(
            primary.loc[
                engine,
                "source_train_compatibility_AUROC",
            ]
        )
        a40 = float(
            limited.loc[
                engine,
                "source_train_compatibility_AUROC",
            ]
        )

        if np.isfinite(a29) and a29 >= OPERATIONAL_GO_AUROC:
            eligible_current.append(engine)
        elif (
            np.isfinite(a29)
            and np.isfinite(a40)
            and OPERATIONAL_LIMITED_LOW <= a29 < OPERATIONAL_LIMITED_HIGH
            and a40 >= OPERATIONAL_GO_AUROC
        ):
            eligible_limited.append(engine)

    if eligible_current:
        decision = "GO_CURRENT_PAPER_A6"
        eligible_engines = eligible_current
    elif eligible_limited:
        decision = "LIMITED_FUTURE_METHOD_ONLY"
        eligible_engines = eligible_limited
    else:
        decision = "STOP_A6_FOR_PAPER6"
        eligible_engines = []

    # Near-zero harmful gate mass diagnostic.
    near_zero = r5_zero[
        r5_zero["harmful_signflip_gate_mass"] <= 0.10
    ]
    near_zero_cat = (
        float(np.mean(near_zero["delta_c_A3"] <= -0.05))
        if len(near_zero)
        else float("nan")
    )

    mechanistic_29 = auc_df[
        (auc_df["target_events"] == 29)
        & (auc_df["endpoint"] == "MECHANISTIC_R0_R3_vs_R4_R5")
    ]
    mechanistic_29_auc = (
        float(
            mechanistic_29.iloc[0][
                "source_train_compatibility_AUROC"
            ]
        )
        if len(mechanistic_29) == 1
        else float("nan")
    )

    summary = {
        "script_version": SCRIPT_VERSION,
        "status": "PASS",
        "scientific_status": decision,
        "run_started_utc": started,
        "run_finished_utc": now_utc(),
        "feasibility_contract_sha256": contract_hash,
        "primary_event_budget": PRIMARY_EVENT_BUDGET,
        "eligible_engines_for_future_A6_contract": eligible_engines,
        "operational_AUROC_at_29": {
            engine: float(
                primary.loc[
                    engine,
                    "source_train_compatibility_AUROC",
                ]
            )
            for engine in BASE_ENGINES
        },
        "operational_AUROC_at_40": {
            engine: float(
                limited.loc[
                    engine,
                    "source_train_compatibility_AUROC",
                ]
            )
            for engine in BASE_ENGINES
        },
        "mechanistic_AUROC_at_29": mechanistic_29_auc,
        "continuous_leakage": {
            "R5_zero_thresholded_harmful_borrow_replicates": len(
                r5_zero
            ),
            "R5_near_zero_harmful_gate_mass_replicates": len(
                near_zero
            ),
            "R5_catastrophic_rate_when_harmful_gate_mass_le_0_10": (
                near_zero_cat
            ),
        },
        "decision_interpretation": {
            "GO_CURRENT_PAPER_A6": (
                "Simple target-training compatibility contains enough information "
                "at 29 events to justify writing a NEW prospective A6 contract. "
                "No human outcomes may be opened yet."
            ),
            "LIMITED_FUTURE_METHOD_ONLY": (
                "Compatibility becomes potentially usable only above the current "
                "human event budget; do not pursue A6 empirical Paper-6 branch."
            ),
            "STOP_A6_FOR_PAPER6": (
                "This simple global-trust signal is not sufficiently informative "
                "at the current event budget. Do not start A6 for Paper 6."
            ),
        }[decision],
        "model_fitting": False,
        "classifier_training": False,
        "human_outcomes_read": False,
        "GPU_execution": False,
        "final_artifact_hashes": {
            "global_trust_feasibility_contract.json": sha256_file(
                FEASIBILITY_CONTRACT
            ),
            "global_trust_replicate_statistics.tsv": sha256_file(
                TRUST_REPLICATES
            ),
            "global_trust_AUROC_by_event_budget.tsv": sha256_file(
                AUROC_BY_BUDGET
            ),
            "event_count_only_abstention_policies.tsv": sha256_file(
                EVENT_POLICY
            ),
            "R5_continuous_harmful_gate_mass_stratification.tsv": sha256_file(
                GATE_MASS_STRATA
            ),
            "scenario_checkpoint_manifest.tsv": sha256_file(
                SCENARIO_MANIFEST
            ),
        },
    }
    write_json(SUMMARY_JSON, summary)

    # ------------------------------------------------------------------
    # Console output.
    # ------------------------------------------------------------------
    print()
    print("=" * 120)
    print("05d0c GLOBAL TRUST FEASIBILITY BY EVENT BUDGET")
    print("=" * 120)
    print(auc_df.to_string(index=False))

    print()
    print("=" * 120)
    print("05d0c TRIVIAL EVENT-COUNT-ONLY ABSTENTION POLICIES")
    print("=" * 120)
    print(policy_df.to_string(index=False))

    print()
    print("=" * 120)
    print("05d0c CONTINUOUS HARMFUL-GATE MASS")
    print("=" * 120)
    print(
        "Population: R5 replicates with ZERO thresholded harmful sign-flip borrows."
    )
    print(gate_df.to_string(index=False))
    print()
    print(
        f"Near-zero harmful gate mass <=0.10: "
        f"n={len(near_zero)}, catastrophic={near_zero_cat}"
    )

    print()
    print("=" * 120)
    print("05d0c FINAL GO / LIMITED / STOP DECISION")
    print("=" * 120)
    print(f"Decision: {decision}")
    print(f"Eligible engine(s): {eligible_engines}")
    print(
        "Operational AUROC @29: "
        + ", ".join(
            f"{engine}={summary['operational_AUROC_at_29'][engine]:.4f}"
            for engine in BASE_ENGINES
        )
    )
    print(
        "Operational AUROC @40: "
        + ", ".join(
            f"{engine}={summary['operational_AUROC_at_40'][engine]:.4f}"
            for engine in BASE_ENGINES
        )
    )
    print(f"Mechanistic AUROC @29: {mechanistic_29_auc:.4f}")
    print()
    print(summary["decision_interpretation"])
    print()
    print("Frozen 05d HOLD changed: NO")
    print("A5 CLOSE changed: NO")
    print("TARGET/GSE21257/GSE39055 outcomes read: NO")
    print("=" * 120)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 120, file=sys.stderr)
        print("05d0c global-trust feasibility diagnostic: FAIL", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print("=" * 120, file=sys.stderr)
        raise
