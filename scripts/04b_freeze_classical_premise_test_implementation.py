#!/usr/bin/env python3
"""
Paper 6 - freeze classical sacrificial-human premise-test implementation.

This is the FINAL implementation freeze before opening GSE16091 expression and OS
outcome values for Paper 6.

Scientific situation
--------------------
04a selected GSE16091 (n=34) as the sacrificial human OS premise-test target.
TARGET-OS, GSE21257, and GSE39055 remain reserved and outcome-closed.

Because no additional eligible non-reserved human OS source cohort is available
under the frozen reservation policy, the 02i phrase "pooled-human source" is
operationalized BEFORE GSE16091 outcome access as:

    the GSE16091 HUMAN OUTER-TRAINING POOL in each repeated outer CV split.

Thus the primary paired premise contrast is:

    DOG2 + HUMAN-TRAINING-POOL residual Cox
        versus
    HUMAN-TRAINING-POOL-only ridge Cox

Both are evaluated on the exact same held-out GSE16091 patients.

This is not a change to the 02i materiality criterion. It is the pre-outcome
operational definition of the previously unspecified human pool.

Primary classical models
------------------------
B0 HUMAN_ONLY_RIDGE
    Hallmark-module ridge Cox fitted only to the human outer-training fold.

B1 HUMAN_ONLY_GENE_ELASTIC_NET
    Gene-level elastic-net Cox, secondary no-transfer benchmark.

B2 DOG_ZERO_SHOT
    Full-DOG2 Hallmark ridge-Cox source score, transferred without human outcome
    adaptation after fold-safe within-domain standardization.

B3 DOG_CORAL_ZERO_SHOT
    Same DOG2 source information with unsupervised module-level CORAL alignment
    to the human outer-training expression distribution. No human outcomes used
    in alignment.

B4 DOG_PLUS_HUMAN_RESIDUAL
    beta_human = beta_dog + delta
    delta is ridge-penalized and fitted only in the human outer-training fold.
    This is the PRIMARY canine-added-value premise model and is a classical
    comparator, not Paper-6 methodological novelty.

Primary premise decision
------------------------
B4 versus B0, using the exact 02i thresholds:
SUPPORTS:
    paired delta Uno C >= +0.02
    bootstrap P(delta C > 0) >= 0.90
    delta IBS <= +0.01
ARGUES AGAINST:
    paired delta Uno C <= -0.02
    bootstrap P(delta C < 0) >= 0.90
otherwise:
    INCONCLUSIVE_NEUTRAL

No human outcome, event count, survival distribution, or expression value is
read in 04b.

Network access
--------------
Allowed ONLY to freeze the GPL96 platform annotation needed for outcome-blind
probe->gene mapping. GSE16091 series-matrix/sample outcome data are NOT fetched.

No CLI arguments.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests


SCRIPT_VERSION = "04b-freeze-classical-premise-test-implementation-v1-no-cli"

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Frozen upstream artifacts.
# ---------------------------------------------------------------------------

I_DIR = ROOT / "results" / "revised_design" / "02i"
I_CONTRACT = I_DIR / "revised_model_benchmark_source_gate_contract.json"
I_SUMMARY = I_DIR / "summary.json"

A3_DIR = ROOT / "results" / "source_gate_protocol" / "03a"
A3_PROTOCOL = A3_DIR / "source_prognostic_gate_protocol.json"
A3_SUMMARY = A3_DIR / "summary.json"
A3_HALLMARK_DOG = A3_DIR / "hallmark_dog2_feature_map.tsv"

B3_DIR = ROOT / "results" / "source_gate" / "03b"
B3_SUMMARY = B3_DIR / "summary.json"
B3_OS_FOLDS = B3_DIR / "os_primary_random_cv_folds.tsv"

C3_DIR = ROOT / "results" / "source_gate_diagnostics" / "03c"
C3_SUMMARY = C3_DIR / "summary.json"
C3_POLICY = C3_DIR / "premise_test_action_policy_addendum.json"

A4_DIR = ROOT / "results" / "human_premise_target" / "04a"
A4_CONTRACT = A4_DIR / "sacrificial_human_premise_target_contract.json"
A4_SUMMARY = A4_DIR / "summary.json"
A4_ROSTER = A4_DIR / "sacrificial_target_sample_roster.tsv"
A4_CANDIDATES = A4_DIR / "candidate_metadata_audit.tsv"

G2_DIR = ROOT / "results" / "ortholog_bridge" / "02g_v3"
G2_BRIDGE = G2_DIR / "primary_outcome_blind_ortholog_bridge.tsv"
G2_SUMMARY = G2_DIR / "summary.json"

H2_DIR = ROOT / "results" / "module_coverage" / "02h"
H2_HALLMARK = H2_DIR / "reference" / "h.all.v2026.1.Hs.symbols.gmt"
H2_SUMMARY = H2_DIR / "summary.json"

# ---------------------------------------------------------------------------
# Outputs.
# ---------------------------------------------------------------------------

OUT_DIR = ROOT / "results" / "human_premise_protocol" / "04b"
REF_DIR = OUT_DIR / "reference"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REF_DIR.mkdir(parents=True, exist_ok=True)

GPL96_RAW = REF_DIR / "GPL96_full_platform.txt"
GPL96_LOCK = REF_DIR / "GPL96_platform_lock.json"
GPL96_PROBE_MAP = OUT_DIR / "gse16091_gpl96_probe_gene_map.tsv"
GSE16091_GENE_UNIVERSE = OUT_DIR / "gse16091_outcome_blind_gene_universe.tsv"
GSE16091_HALLMARK_MAP = OUT_DIR / "gse16091_hallmark_feature_map.tsv"
DOG_SOURCE_ALPHA = OUT_DIR / "dog2_full_source_alpha_contract.tsv"
BENCHMARK_REGISTRY = OUT_DIR / "classical_premise_benchmark_registry.tsv"
CV_CONTRACT = OUT_DIR / "gse16091_cv_contract.tsv"
PREMISE_RULES = OUT_DIR / "premise_decision_rules.tsv"
PROTOCOL_JSON = OUT_DIR / "classical_premise_test_protocol.json"
SUMMARY_JSON = OUT_DIR / "summary.json"

# ---------------------------------------------------------------------------
# Selected target and retrieval lock.
# ---------------------------------------------------------------------------

SELECTED_TARGET = "GSE16091"
SELECTED_PLATFORM = "GPL96"

GPL96_TEXT_URL = (
    "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
    "?acc=GPL96&targ=self&form=text&view=full"
)

# This is FROZEN but NOT downloaded/read in 04b.
GSE16091_SERIES_MATRIX_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE16nnn/GSE16091/"
    "matrix/GSE16091_series_matrix.txt.gz"
)

EXPECTED_TARGET_N = 34
EXPECTED_DOG_PRIMARY_BRIDGE_N = 11815
EXPECTED_HALLMARK_N = 50

# ---------------------------------------------------------------------------
# Frozen human CV and inference settings.
# ---------------------------------------------------------------------------

BASE_SEED = 20260830

OUTER_SPLITS = 5
OUTER_REPEATS = 20
INNER_SPLITS = 4

PATIENT_BOOTSTRAPS = 5000
BOOTSTRAP_SEED = BASE_SEED + 900000

UNO_TAU_QUANTILE = 0.90
INNER_TIE_TOLERANCE = 0.005

RIDGE_ALPHAS = np.logspace(-4, 4, 17)
ELASTIC_L1_RATIOS = [0.10, 0.50, 0.90]
ELASTIC_N_ALPHAS = 30
ELASTIC_ALPHA_MIN_RATIO = 0.01

MIN_HALLMARK_GENES = 10
ZERO_VARIANCE_SD_EPS = 1e-12

# CORAL numerical stabilization.
CORAL_COV_EPS = 1e-5

# Brier/IBS evaluation.
IBS_GRID_POINTS = 20
IBS_TRAIN_EVENT_TIME_LOWER_QUANTILE = 0.20
IBS_TRAIN_EVENT_TIME_UPPER_QUANTILE = 0.80
IBS_MIN_GRID_POINTS_AFTER_SUPPORT = 5

# Source consensus-alpha extraction from already-completed 03b.
DOG_SOURCE_ALPHA_RULE = (
    "Take all 100 chosen_alpha values from frozen OS primary outer folds; "
    "compute median log10(alpha); snap to nearest member of the frozen 17-value "
    "03a ridge grid; ties choose stronger regularization (larger alpha). "
    "No new DOG2 outcome tuning."
)

# Event-matched secondary information-efficiency analysis.
EVENT_MATCHED_DOG_DRAWS_PER_OUTER_FOLD = 50
EVENT_MATCHED_BASE_SEED = BASE_SEED + 500000

# ---------------------------------------------------------------------------
# Frozen premise thresholds copied from 02i; verified at runtime.
# ---------------------------------------------------------------------------

PREMISE_DELTA_C_SUPPORT = 0.02
PREMISE_PROB_SUPPORT = 0.90
PREMISE_MAX_DELTA_IBS = 0.01
PREMISE_DELTA_C_AGAINST = -0.02
PREMISE_PROB_AGAINST = 0.90


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "na", "n/a"}:
        return ""
    return text


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def require_file(path: Path) -> Path:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Required frozen artifact missing: {path}")
    return path


def bool_mask(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False).astype(bool)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y", "t"})
    )


def download_platform_text(
    session: requests.Session,
    url: str,
    destination: Path,
    retries: int = 3,
) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        return

    last_exc: Optional[Exception] = None
    tmp = destination.with_suffix(destination.suffix + ".part")

    for attempt in range(1, retries + 1):
        try:
            response = session.get(
                url,
                timeout=120,
                headers={"User-Agent": "Paper6-04b-platform-only/1.0"},
            )
            response.raise_for_status()
            text = response.text

            if "!platform_table_begin" not in text.lower():
                raise RuntimeError(
                    "GPL96 response does not contain a GEO platform table."
                )

            # Platform pages must not contain GSE16091 sample outcome fields.
            forbidden = [
                "days-followup",
                "alive-or-dead",
                "!sample_characteristics_ch1",
                "!sample_table_begin",
            ]
            lower = text.lower()
            if any(token in lower for token in forbidden):
                raise RuntimeError(
                    "Unexpected sample/outcome-like content in platform response."
                )

            tmp.write_text(text, encoding="utf-8", newline="\n")
            tmp.replace(destination)
            return
        except Exception as exc:
            last_exc = exc
            tmp.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(2 * attempt)

    raise RuntimeError(f"Could not freeze GPL96 platform annotation: {last_exc}")


def parse_geo_platform_table(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    begin = None
    end = None

    for i, line in enumerate(lines):
        if line.strip().lower() == "!platform_table_begin":
            begin = i
        elif line.strip().lower() == "!platform_table_end":
            end = i
            break

    if begin is None or end is None or end <= begin + 1:
        raise RuntimeError("Could not locate GPL96 platform table bounds.")

    table_text = "\n".join(lines[begin + 1 : end]) + "\n"
    frame = pd.read_csv(
        io.StringIO(table_text),
        sep="\t",
        dtype=str,
        low_memory=False,
    ).fillna("")

    if frame.empty:
        raise RuntimeError("GPL96 platform table parsed as empty.")

    return frame


def identify_column(
    columns: Sequence[str],
    candidates: Sequence[str],
) -> str:
    normalized = {
        re.sub(r"[^a-z0-9]+", "", str(col).lower()): str(col)
        for col in columns
    }

    for candidate in candidates:
        key = re.sub(r"[^a-z0-9]+", "", candidate.lower())
        if key in normalized:
            return normalized[key]

    raise RuntimeError(
        f"Could not identify required GPL96 column. "
        f"Candidates={list(candidates)}; columns={list(columns)}"
    )


def parse_single_gene_symbol(value: Any) -> Tuple[str, str]:
    text = clean(value)

    if not text:
        return "", "EMPTY"

    # GPL annotation commonly separates alternatives with ///.
    pieces = [
        x.strip()
        for x in re.split(r"\s*///\s*|[;,]\s*", text)
        if x.strip()
    ]

    # Remove placeholder-like entries.
    bad = {"---", "NA", "N/A", "NULL", "NONE"}
    pieces = [x for x in pieces if x.upper() not in bad]

    unique = []
    seen = set()

    for piece in pieces:
        symbol = piece.upper()
        if symbol not in seen:
            unique.append(symbol)
            seen.add(symbol)

    if len(unique) == 0:
        return "", "EMPTY"
    if len(unique) > 1:
        return "", "MULTI_SYMBOL"

    symbol = unique[0]

    if not re.fullmatch(r"[A-Z0-9][A-Z0-9.\-]*", symbol):
        return "", "NONSTANDARD_SYMBOL"

    return symbol, "UNIQUE_SYMBOL"


def parse_gmt(path: Path) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\r\n")
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < 3:
                continue

            module = parts[0].strip()
            genes = []
            seen = set()

            for value in parts[2:]:
                gene = value.strip().upper()
                if not gene or gene in seen:
                    continue
                seen.add(gene)
                genes.append(gene)

            result[module] = genes

    return result


def nearest_grid_alpha(
    selected_alphas: np.ndarray,
    grid: np.ndarray,
) -> Tuple[float, float]:
    selected_alphas = np.asarray(selected_alphas, dtype=float)
    grid = np.asarray(grid, dtype=float)

    if np.any(~np.isfinite(selected_alphas)) or np.any(selected_alphas <= 0):
        raise RuntimeError("Invalid chosen DOG2 alpha values.")
    if np.any(~np.isfinite(grid)) or np.any(grid <= 0):
        raise RuntimeError("Invalid frozen ridge grid.")

    median_log = float(np.median(np.log10(selected_alphas)))
    distances = np.abs(np.log10(grid) - median_log)
    best = np.min(distances)

    candidates = grid[np.isclose(distances, best, atol=1e-12, rtol=0)]

    # Frozen tie rule: stronger ridge.
    chosen = float(np.max(candidates))
    return chosen, median_log


def main() -> None:
    started = now_utc()

    print("=" * 120)
    print("Paper 6 - freeze classical sacrificial-human premise-test implementation")
    print("=" * 120)
    print(f"Script version: {SCRIPT_VERSION}")
    print()
    print("Safety / scope:")
    print("  GSE16091 outcome VALUES read: NO")
    print("  GSE16091 event count read: NO")
    print("  GSE16091 survival distribution read: NO")
    print("  GSE16091 expression VALUES read: NO")
    print("  GSE16091 series matrix fetched: NO")
    print("  GPL96 platform annotation fetched/read: YES [outcome-blind]")
    print("  TARGET/GSE21257/GSE39055 outcomes read: NO")
    print("  DOG2 model refit/tuning in 04b: NO")
    print("  Human model fitting in 04b: NO")
    print("  Premise materiality thresholds changed: NO")
    print()

    for path in [
        I_CONTRACT,
        I_SUMMARY,
        A3_PROTOCOL,
        A3_SUMMARY,
        A3_HALLMARK_DOG,
        B3_SUMMARY,
        B3_OS_FOLDS,
        C3_SUMMARY,
        C3_POLICY,
        A4_CONTRACT,
        A4_SUMMARY,
        A4_ROSTER,
        A4_CANDIDATES,
        G2_BRIDGE,
        G2_SUMMARY,
        H2_HALLMARK,
        H2_SUMMARY,
    ]:
        require_file(path)

    i_contract = read_json(I_CONTRACT)
    a3_protocol = read_json(A3_PROTOCOL)
    a3_summary = read_json(A3_SUMMARY)
    b3_summary = read_json(B3_SUMMARY)
    c3_summary = read_json(C3_SUMMARY)
    c3_policy = read_json(C3_POLICY)
    a4_contract = read_json(A4_CONTRACT)
    a4_summary = read_json(A4_SUMMARY)
    g2_summary = read_json(G2_SUMMARY)
    h2_summary = read_json(H2_SUMMARY)

    if clean(i_contract.get("status")) != "PASS":
        raise RuntimeError("02i contract is not PASS.")
    if clean(a3_summary.get("scientific_status")) != (
        "PASS_SOURCE_PROGNOSTIC_GATE_PROTOCOL_FROZEN"
    ):
        raise RuntimeError("03a is not in expected frozen PASS state.")
    if clean(b3_summary.get("scientific_status")) != (
        "SOURCE_AMBER_WEAK_AND_CONTEXT_SENSITIVE"
    ):
        raise RuntimeError("03b source status is not the expected AMBER state.")
    if clean(c3_summary.get("scientific_status")) != (
        "PASS_BOUNDED_DIAGNOSTIC_COMPLETE_SOURCE_GATE_UNCHANGED"
    ):
        raise RuntimeError("03c bounded diagnostic is not PASS.")
    if clean(a4_summary.get("scientific_status")) != (
        "PASS_SACRIFICIAL_HUMAN_PREMISE_TARGET_FROZEN"
    ):
        raise RuntimeError("04a target freeze is not PASS.")
    if clean(a4_summary.get("selected_target")) != SELECTED_TARGET:
        raise RuntimeError("04a selected target is not GSE16091.")

    selected = a4_contract.get("selected_target") or {}
    if clean(selected.get("accession")) != SELECTED_TARGET:
        raise RuntimeError("04a contract selected target mismatch.")
    if int(selected.get("n")) != EXPECTED_TARGET_N:
        raise RuntimeError("04a target n changed.")

    # ------------------------------------------------------------------
    # Verify exact premise thresholds frozen in 02i.
    # ------------------------------------------------------------------
    premise = i_contract.get("premise_test") or {}

    support = premise.get("supports_canine_added_value") or {}
    against = premise.get("argues_against_canine_added_value") or {}

    if float(support["paired_delta_uno_c_min"]) != PREMISE_DELTA_C_SUPPORT:
        raise RuntimeError("02i support delta-C threshold changed.")
    if (
        float(
            support[
                "paired_bootstrap_probability_delta_c_gt_zero_min"
            ]
        )
        != PREMISE_PROB_SUPPORT
    ):
        raise RuntimeError("02i positive-bootstrap threshold changed.")
    if float(support["maximum_allowed_delta_IBS"]) != PREMISE_MAX_DELTA_IBS:
        raise RuntimeError("02i delta-IBS threshold changed.")
    if float(against["paired_delta_uno_c_max"]) != PREMISE_DELTA_C_AGAINST:
        raise RuntimeError("02i against delta-C threshold changed.")
    if (
        float(
            against[
                "paired_bootstrap_probability_delta_c_lt_zero_min"
            ]
        )
        != PREMISE_PROB_AGAINST
    ):
        raise RuntimeError("02i negative-bootstrap threshold changed.")
    if clean(premise.get("otherwise")) != "INCONCLUSIVE_NEUTRAL":
        raise RuntimeError("02i neutral premise state changed.")

    # ------------------------------------------------------------------
    # Verify reservation and candidate feasibility basis.
    # ------------------------------------------------------------------
    reserved = set(a4_contract.get("reserved_confirmatory_cohorts") or [])
    if reserved != {"TARGET-OS", "GSE21257", "GSE39055"}:
        raise RuntimeError(f"Reserved cohort set changed: {reserved}")

    candidate_audit = pd.read_csv(A4_CANDIDATES, sep="\t")
    eligible = candidate_audit[
        candidate_audit["eligible"].astype(str).str.lower().isin({"true", "1"})
    ]

    if list(eligible["candidate"].astype(str)) != ["GSE16091"]:
        raise RuntimeError(
            "04a metadata audit no longer has GSE16091 as the sole eligible "
            "sacrificial target under its frozen candidate rules."
        )

    # This operationalizes "human pool" BEFORE target outcome access.
    human_pool_definition = (
        "Within each repeated outer CV split, the pooled-human source is the "
        "GSE16091 outer-training fold only. Reserved human cohorts are not used "
        "for model fitting, hyperparameter tuning, source construction, or "
        "premise-test outcome access."
    )

    # ------------------------------------------------------------------
    # Freeze GPL96 outcome-blind platform annotation.
    # ------------------------------------------------------------------
    session = requests.Session()
    download_platform_text(session, GPL96_TEXT_URL, GPL96_RAW)

    if GPL96_LOCK.exists():
        lock = read_json(GPL96_LOCK)
        expected = clean(lock.get("sha256"))
        observed = sha256_file(GPL96_RAW)
        if expected != observed:
            raise RuntimeError("GPL96 frozen platform hash mismatch.")
    else:
        lock = {
            "status": "PASS",
            "created_utc": now_utc(),
            "platform": SELECTED_PLATFORM,
            "url": GPL96_TEXT_URL,
            "path": str(GPL96_RAW.relative_to(ROOT)),
            "sha256": sha256_file(GPL96_RAW),
            "network_scope": "GEO_PLATFORM_ONLY_NO_SAMPLE_OR_OUTCOME_ACCESS",
        }
        write_json(GPL96_LOCK, lock)

    platform = parse_geo_platform_table(GPL96_RAW)

    probe_col = identify_column(
        platform.columns,
        ["ID", "ID_REF", "Probe Set ID", "Probe Set Id"],
    )
    symbol_col = identify_column(
        platform.columns,
        ["Gene Symbol", "Gene symbol", "GENE_SYMBOL", "Symbol"],
    )

    probe_rows: List[Dict[str, Any]] = []

    for _, row in platform.iterrows():
        probe = clean(row[probe_col])
        symbol, status = parse_single_gene_symbol(row[symbol_col])

        if not probe:
            continue

        probe_rows.append(
            {
                "probe_id": probe,
                "raw_gene_symbol_field": clean(row[symbol_col]),
                "human_gene_symbol": symbol,
                "symbol_resolution_status": status,
            }
        )

    probe_map = pd.DataFrame(probe_rows)

    if probe_map["probe_id"].duplicated().any():
        raise RuntimeError("GPL96 platform contains duplicate probe IDs.")

    probe_map.to_csv(GPL96_PROBE_MAP, sep="\t", index=False)

    # ------------------------------------------------------------------
    # Freeze target gene universe using ONLY outcome-blind bridge+platform.
    # ------------------------------------------------------------------
    bridge = pd.read_csv(
        G2_BRIDGE,
        sep="\t",
        dtype=str,
        low_memory=False,
    ).fillna("")

    if "primary_dog2_to_target_os" not in bridge.columns:
        raise RuntimeError("02g bridge lacks primary feature flag.")

    primary_bridge = bridge[
        bool_mask(bridge["primary_dog2_to_target_os"])
    ].copy()

    if len(primary_bridge) != EXPECTED_DOG_PRIMARY_BRIDGE_N:
        raise RuntimeError(
            f"02g primary bridge rows={len(primary_bridge)}, "
            f"expected={EXPECTED_DOG_PRIMARY_BRIDGE_N}."
        )

    bridge_symbols = set(
        primary_bridge["human_gene_symbol"]
        .astype(str)
        .str.strip()
        .str.upper()
    )
    bridge_symbols.discard("")

    unique_probe_map = probe_map[
        probe_map["symbol_resolution_status"] == "UNIQUE_SYMBOL"
    ].copy()

    platform_symbols = set(unique_probe_map["human_gene_symbol"])
    common_symbols = sorted(bridge_symbols & platform_symbols)

    gene_rows: List[Dict[str, Any]] = []

    for symbol in common_symbols:
        probes = sorted(
            unique_probe_map.loc[
                unique_probe_map["human_gene_symbol"] == symbol,
                "probe_id",
            ].astype(str)
        )

        gene_rows.append(
            {
                "human_gene_symbol": symbol,
                "n_GPL96_probes": len(probes),
                "GPL96_probe_ids": ";".join(probes),
                "probe_collapse_rule": (
                    "median across all uniquely-symbol-mapped GPL96 probes "
                    "for this gene within each sample"
                ),
                "in_02g_primary_DOG2_TARGET_bridge": True,
                "outcome_selected": False,
            }
        )

    gene_universe = pd.DataFrame(gene_rows)
    if gene_universe.empty:
        raise RuntimeError("GSE16091 outcome-blind common gene universe is empty.")

    GSE16091_GENE_UNIVERSE.parent.mkdir(parents=True, exist_ok=True)
    gene_universe.to_csv(
        GSE16091_GENE_UNIVERSE,
        sep="\t",
        index=False,
    )

    # ------------------------------------------------------------------
    # Freeze Hallmark module map on common dog-human-GPL96 genes.
    # ------------------------------------------------------------------
    hallmark = parse_gmt(H2_HALLMARK)
    if len(hallmark) != EXPECTED_HALLMARK_N:
        raise RuntimeError(
            f"Hallmark module count={len(hallmark)}, expected 50."
        )

    common_symbol_set = set(common_symbols)
    hallmark_rows: List[Dict[str, Any]] = []

    module_counts: Dict[str, int] = {}

    for module in sorted(hallmark):
        genes = sorted(set(hallmark[module]) & common_symbol_set)
        module_counts[module] = len(genes)

        for gene in genes:
            probes = gene_universe.loc[
                gene_universe["human_gene_symbol"] == gene,
                "GPL96_probe_ids",
            ].iloc[0]

            hallmark_rows.append(
                {
                    "hallmark_module": module,
                    "human_gene_symbol": gene,
                    "GPL96_probe_ids": probes,
                    "n_GPL96_probes": int(
                        gene_universe.loc[
                            gene_universe["human_gene_symbol"] == gene,
                            "n_GPL96_probes",
                        ].iloc[0]
                    ),
                    "module_score_rule": (
                        "within human training fold: gene-level z-score after "
                        "median probe collapse; module score = unweighted mean "
                        "of available mapped gene z-scores"
                    ),
                }
            )

    hallmark_map = pd.DataFrame(hallmark_rows)
    hallmark_map.to_csv(
        GSE16091_HALLMARK_MAP,
        sep="\t",
        index=False,
    )

    low_modules = {
        module: n
        for module, n in module_counts.items()
        if n < MIN_HALLMARK_GENES
    }

    if low_modules:
        raise RuntimeError(
            "GSE16091 GPL96 bridge does not support the frozen Hallmark-50 "
            f"representation at >= {MIN_HALLMARK_GENES} genes/module: "
            f"{low_modules}"
        )

    # ------------------------------------------------------------------
    # Freeze full-DOG2 source-alpha rule WITHOUT new DOG2 tuning.
    # ------------------------------------------------------------------
    os_folds = pd.read_csv(B3_OS_FOLDS, sep="\t")

    if "chosen_alpha" not in os_folds.columns:
        raise RuntimeError("03b OS fold file lacks chosen_alpha.")

    chosen_alphas = pd.to_numeric(
        os_folds["chosen_alpha"],
        errors="raise",
    ).to_numpy(dtype=float)

    expected_fold_n = (
        int(a3_protocol["random_cv"]["primary"]["outer_splits"])
        * int(a3_protocol["random_cv"]["primary"]["outer_repeats"])
    )

    if len(chosen_alphas) != expected_fold_n:
        raise RuntimeError(
            f"03b OS chosen-alpha rows={len(chosen_alphas)}, "
            f"expected={expected_fold_n}."
        )

    frozen_grid = np.asarray(
        a3_protocol["primary_model"]["ridge_alpha_grid"],
        dtype=float,
    )

    dog_source_alpha, median_log10_alpha = nearest_grid_alpha(
        chosen_alphas,
        frozen_grid,
    )

    alpha_summary = pd.DataFrame(
        [
            {
                "rule": DOG_SOURCE_ALPHA_RULE,
                "n_frozen_outer_fold_alphas": len(chosen_alphas),
                "median_log10_chosen_alpha": median_log10_alpha,
                "selected_full_DOG2_source_alpha": dog_source_alpha,
                "new_DOG2_tuning_performed_in_04b": False,
            }
        ]
    )
    alpha_summary.to_csv(DOG_SOURCE_ALPHA, sep="\t", index=False)

    # ------------------------------------------------------------------
    # Freeze benchmark ladder.
    # ------------------------------------------------------------------
    benchmark_rows = [
        {
            "benchmark_id": "B0",
            "model": "HUMAN_ONLY_HALLMARK_RIDGE_COX",
            "uses_DOG2_outcomes": False,
            "uses_human_training_outcomes": True,
            "uses_human_test_outcomes_for_fit": False,
            "representation": "Hallmark modules",
            "role": "PRIMARY HUMAN-ONLY COMPARATOR",
            "premise_decision_model": True,
        },
        {
            "benchmark_id": "B1",
            "model": "HUMAN_ONLY_GENE_ELASTIC_NET_COX",
            "uses_DOG2_outcomes": False,
            "uses_human_training_outcomes": True,
            "uses_human_test_outcomes_for_fit": False,
            "representation": "outcome-blind common gene universe",
            "role": "SECONDARY NO-TRANSFER BENCHMARK",
            "premise_decision_model": False,
        },
        {
            "benchmark_id": "B2",
            "model": "DOG_ZERO_SHOT_HALLMARK_RIDGE_SCORE",
            "uses_DOG2_outcomes": True,
            "uses_human_training_outcomes": False,
            "uses_human_test_outcomes_for_fit": False,
            "representation": "Hallmark modules",
            "role": "DIRECT SOURCE-TRANSPORT BENCHMARK",
            "premise_decision_model": False,
        },
        {
            "benchmark_id": "B3",
            "model": "DOG_CORAL_ZERO_SHOT_HALLMARK_COX",
            "uses_DOG2_outcomes": True,
            "uses_human_training_outcomes": False,
            "uses_human_test_outcomes_for_fit": False,
            "representation": "Hallmark modules + unsupervised CORAL",
            "role": "SIMPLE ALIGNMENT BENCHMARK",
            "premise_decision_model": False,
        },
        {
            "benchmark_id": "B4",
            "model": "DOG_PLUS_HUMAN_RESIDUAL_RIDGE_COX",
            "uses_DOG2_outcomes": True,
            "uses_human_training_outcomes": True,
            "uses_human_test_outcomes_for_fit": False,
            "representation": "Hallmark modules",
            "role": "PRIMARY CANINE-ADDED-VALUE CLASSICAL TRANSFER MODEL",
            "premise_decision_model": True,
        },
    ]
    pd.DataFrame(benchmark_rows).to_csv(
        BENCHMARK_REGISTRY,
        sep="\t",
        index=False,
    )

    # ------------------------------------------------------------------
    # Freeze CV protocol.
    # ------------------------------------------------------------------
    cv_rows = [
        {
            "component": "outer_cv",
            "value": f"{OUTER_SPLITS}-fold x {OUTER_REPEATS} repeats",
            "rule": (
                "Stratified by OS event. Exact same patient splits for B0-B4. "
                "If either event stratum has fewer than 5 patients, FAIL_CLOSED; "
                "do not change fold count after outcome opening."
            ),
        },
        {
            "component": "inner_cv",
            "value": f"{INNER_SPLITS}-fold",
            "rule": (
                "Within outer-training fold only; stratified by OS event. "
                "If either stratum has fewer than 4 observations in an "
                "outer-training fold, FAIL_CLOSED."
            ),
        },
        {
            "component": "primary_metric",
            "value": "Uno C",
            "rule": (
                "tau = 90th percentile of outer-training follow-up for fold "
                "evaluation; final aggregated OOF Uno C uses 90th percentile "
                "of full GSE16091 follow-up."
            ),
        },
        {
            "component": "patient_bootstrap",
            "value": str(PATIENT_BOOTSTRAPS),
            "rule": (
                "Paired patient bootstrap of final aggregated B0 and B4 "
                "cross-fitted risk vectors; no model refit in bootstrap."
            ),
        },
        {
            "component": "IBS",
            "value": f"{IBS_GRID_POINTS} time points",
            "rule": (
                "Within each outer fold use one shared B0/B4 evaluation grid "
                "from 20th to 80th percentile of HUMAN outer-training EVENT times, "
                "trimmed to common train/test support; IPCW from human training fold. "
                "Delta IBS = B4 minus B0; smaller is better."
            ),
        },
        {
            "component": "risk_pooling",
            "value": "training-risk standardized",
            "rule": (
                "Before pooling OOF predictions across folds/repeats, standardize "
                "each model's held-out linear predictor using that model's "
                "corresponding human outer-training risk mean/SD."
            ),
        },
    ]
    pd.DataFrame(cv_rows).to_csv(CV_CONTRACT, sep="\t", index=False)

    # ------------------------------------------------------------------
    # Freeze premise decision rules verbatim from 02i.
    # ------------------------------------------------------------------
    premise_rows = [
        {
            "state": "SUPPORTS_CANINE_ADDED_VALUE",
            "rule": (
                "paired delta Uno C (B4-B0) >= +0.02 AND "
                "paired-bootstrap P(delta C > 0) >= 0.90 AND "
                "delta IBS (B4-B0) <= +0.01"
            ),
        },
        {
            "state": "ARGUES_AGAINST_CANINE_ADDED_VALUE",
            "rule": (
                "paired delta Uno C (B4-B0) <= -0.02 AND "
                "paired-bootstrap P(delta C < 0) >= 0.90"
            ),
        },
        {
            "state": "INCONCLUSIVE_NEUTRAL",
            "rule": "All other outcomes.",
        },
    ]
    pd.DataFrame(premise_rows).to_csv(
        PREMISE_RULES,
        sep="\t",
        index=False,
    )

    # ------------------------------------------------------------------
    # Full frozen implementation contract.
    # ------------------------------------------------------------------
    protocol = {
        "script_version": SCRIPT_VERSION,
        "status": "PASS",
        "scientific_status": (
            "PASS_CLASSICAL_PREMISE_TEST_IMPLEMENTATION_FROZEN"
        ),
        "created_utc": now_utc(),
        "target": {
            "accession": SELECTED_TARGET,
            "n": EXPECTED_TARGET_N,
            "platform": SELECTED_PLATFORM,
            "endpoint": "overall survival",
            "time_field": selected[
                "time_field_frozen_for_future_outcome_opening"
            ],
            "status_field": selected[
                "status_field_frozen_for_future_outcome_opening"
            ],
            "status_mapping": selected[
                "status_mapping_frozen_for_future_outcome_opening"
            ],
            "sample_roster_sha256": sha256_file(A4_ROSTER),
            "historical_cross_species_use_flag": bool(
                selected["historical_cross_species_use_flag"]
            ),
        },
        "reserved_confirmatory_cohorts": sorted(reserved),
        "human_pool_operationalization": {
            "definition": human_pool_definition,
            "external_nonreserved_human_OS_source_used": False,
            "reason": (
                "Under the frozen 04a candidate/reservation policy there is no "
                "additional eligible survival-labelled non-reserved human source "
                "cohort available for the premise analysis."
            ),
            "materiality_criterion_changed": False,
            "human_pool_definition_frozen_before_GSE16091_outcome_access": True,
        },
        "human_expression_retrieval_after_04b": {
            "series_matrix_url": GSE16091_SERIES_MATRIX_URL,
            "download_allowed_only_after_04b_PASS": True,
            "04b_series_matrix_fetched": False,
            "sample_accessions_must_exactly_match_04a_roster": True,
            "unexpected_extra_or_missing_sample_action": "FAIL_CLOSED",
        },
        "GPL96_mapping": {
            "platform_snapshot_sha256": sha256_file(GPL96_RAW),
            "platform_lock_sha256": sha256_file(GPL96_LOCK),
            "probe_map_sha256": sha256_file(GPL96_PROBE_MAP),
            "probe_to_gene_rule": (
                "Keep probes resolving to exactly one nonempty gene symbol; "
                "exclude multi-symbol/empty/nonstandard entries."
            ),
            "multiple_probes_per_gene_rule": "median expression across probes",
            "outcome_dependent_probe_selection": False,
        },
        "outcome_blind_feature_universe": {
            "02g_primary_bridge_genes": EXPECTED_DOG_PRIMARY_BRIDGE_N,
            "GPL96_common_bridge_genes": int(len(gene_universe)),
            "gene_universe_sha256": sha256_file(GSE16091_GENE_UNIVERSE),
            "Hallmark_modules": EXPECTED_HALLMARK_N,
            "Hallmark_min_common_genes": int(min(module_counts.values())),
            "Hallmark_median_common_genes": float(
                np.median(list(module_counts.values()))
            ),
            "Hallmark_max_common_genes": int(max(module_counts.values())),
            "Hallmark_map_sha256": sha256_file(GSE16091_HALLMARK_MAP),
            "minimum_genes_per_module": MIN_HALLMARK_GENES,
        },
        "fold_safe_human_preprocessing": [
            "collapse GPL96 probes to frozen human gene symbols by median",
            "within human outer-training fold compute gene mean and population SD",
            "exclude training-zero-variance genes for that fold only",
            "apply training gene mean/SD to validation/test",
            "Hallmark score = unweighted mean of available gene z-scores",
            "require >=10 nonzero-variance genes per Hallmark module",
            "within human outer-training fold standardize 50 Hallmark scores",
            "apply training module mean/SD to validation/test",
        ],
        "DOG2_full_source_model": {
            "representation": "same 50 Hallmark modules as 03b",
            "endpoint": "OS",
            "fit_population": "all frozen 186 DOG2 dogs",
            "source_alpha": dog_source_alpha,
            "source_alpha_rule": DOG_SOURCE_ALPHA_RULE,
            "source_alpha_contract_sha256": sha256_file(DOG_SOURCE_ALPHA),
            "new_DOG2_hyperparameter_search_after_03b": False,
            "fit_timing": (
                "04c may fit the full DOG2 coefficient vector once at this "
                "already-frozen alpha before opening/using GSE16091 outcomes "
                "for any human model decision."
            ),
        },
        "models": {
            "B0": {
                "name": "HUMAN_ONLY_HALLMARK_RIDGE_COX",
                "ridge_alpha_grid": [float(x) for x in RIDGE_ALPHAS],
                "inner_metric": "Uno C",
                "tie_tolerance": INNER_TIE_TOLERANCE,
                "tie_rule": (
                    "largest alpha within 0.005 Uno C of inner-CV best"
                ),
            },
            "B1": {
                "name": "HUMAN_ONLY_GENE_ELASTIC_NET_COX",
                "l1_ratio_grid": ELASTIC_L1_RATIOS,
                "n_alphas": ELASTIC_N_ALPHAS,
                "alpha_min_ratio": ELASTIC_ALPHA_MIN_RATIO,
                "inner_metric": "Uno C",
                "tie_tolerance": INNER_TIE_TOLERANCE,
                "tie_rule": (
                    "stronger regularization path index within tolerance, "
                    "then larger l1_ratio"
                ),
                "role": "SECONDARY_ONLY",
            },
            "B2": {
                "name": "DOG_ZERO_SHOT_HALLMARK_RIDGE_SCORE",
                "human_outcomes_used_for_fit": False,
                "alignment": (
                    "within-domain fold-safe gene/module standardization only"
                ),
                "role": "SECONDARY_ZERO_SHOT_CONTEXT",
            },
            "B3": {
                "name": "DOG_CORAL_ZERO_SHOT_HALLMARK_COX",
                "human_outcomes_used_for_alignment": False,
                "CORAL_space": "50 Hallmark modules",
                "covariance_epsilon": CORAL_COV_EPS,
                "role": "SECONDARY_SIMPLE_ALIGNMENT",
            },
            "B4": {
                "name": "DOG_PLUS_HUMAN_RESIDUAL_RIDGE_COX",
                "equation": "beta_human = beta_DOG2 + delta",
                "delta_ridge_alpha_grid": [float(x) for x in RIDGE_ALPHAS],
                "inner_metric": "Uno C",
                "tie_tolerance": INNER_TIE_TOLERANCE,
                "tie_rule": (
                    "largest delta-penalty alpha within 0.005 Uno C of inner-CV best"
                ),
                "implementation": (
                    "penalized Cox partial likelihood with L2 penalty centered "
                    "at frozen beta_DOG2; optimize delta only"
                ),
                "role": "PRIMARY_CANINE_ADDED_VALUE_MODEL",
            },
        },
        "paired_human_evaluation": {
            "outer_splits": OUTER_SPLITS,
            "outer_repeats": OUTER_REPEATS,
            "inner_splits": INNER_SPLITS,
            "base_seed": BASE_SEED,
            "stratification": "OS_event",
            "same_outer_splits_for_all_models": True,
            "split_failure_rule": (
                "If either full-cohort event stratum has fewer than 5 samples, "
                "or an outer-training fold cannot support 4-fold event-stratified "
                "inner CV, FAIL_CLOSED. Do not reduce folds after outcome access."
            ),
            "primary_metric": "Uno C",
            "Uno_tau_quantile": UNO_TAU_QUANTILE,
            "patient_bootstrap_replicates": PATIENT_BOOTSTRAPS,
            "patient_bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_model_refit": False,
        },
        "IBS": {
            "required_for_primary_premise_decision": True,
            "grid_points": IBS_GRID_POINTS,
            "training_event_time_lower_quantile": (
                IBS_TRAIN_EVENT_TIME_LOWER_QUANTILE
            ),
            "training_event_time_upper_quantile": (
                IBS_TRAIN_EVENT_TIME_UPPER_QUANTILE
            ),
            "minimum_supported_grid_points": (
                IBS_MIN_GRID_POINTS_AFTER_SUPPORT
            ),
            "same_time_grid_for_B0_and_B4_within_fold": True,
            "IPCW_reference": "human outer-training survival",
            "delta_definition": "IBS_B4_minus_IBS_B0",
            "positive_delta_is_worse": True,
        },
        "primary_premise_contrast": {
            "numerator_model": "B4",
            "reference_model": "B0",
            "delta_uno_c": "UnoC_B4_minus_UnoC_B0",
            "delta_IBS": "IBS_B4_minus_IBS_B0",
            "SUPPORTS_CANINE_ADDED_VALUE": {
                "delta_uno_c_min": PREMISE_DELTA_C_SUPPORT,
                "bootstrap_P_delta_c_gt_zero_min": PREMISE_PROB_SUPPORT,
                "delta_IBS_max": PREMISE_MAX_DELTA_IBS,
            },
            "ARGUES_AGAINST_CANINE_ADDED_VALUE": {
                "delta_uno_c_max": PREMISE_DELTA_C_AGAINST,
                "bootstrap_P_delta_c_lt_zero_min": PREMISE_PROB_AGAINST,
            },
            "otherwise": "INCONCLUSIVE_NEUTRAL",
            "thresholds_source": "02i",
        },
        "secondary_event_matched_information_efficiency": {
            "enabled": True,
            "decision_role": "SECONDARY_ONLY",
            "within_each_outer_fold": (
                "Let n_H and e_H be human outer-training sample/event counts. "
                "Draw DOG2 subsets with exactly n_H samples and e_H OS events."
            ),
            "dog_draws_per_outer_fold": EVENT_MATCHED_DOG_DRAWS_PER_OUTER_FOLD,
            "base_seed": EVENT_MATCHED_BASE_SEED,
            "dog_arm_sampling_rule": (
                "Within event/censor strata, sample DOG2 patients without replacement "
                "with COTC021/COTC022 composition as balanced as feasible; any "
                "one-sample imbalance alternates deterministically by draw seed."
            ),
            "comparison": (
                "event-matched DOG2 zero-shot information efficiency versus "
                "human-training-pool-only B0; cannot determine primary premise state."
            ),
        },
        "CORAL": {
            "outcome_blind": True,
            "source": "DOG2 standardized Hallmark modules",
            "target": "human outer-training standardized Hallmark modules",
            "transform_direction": "source_to_target",
            "covariance_regularization_epsilon": CORAL_COV_EPS,
            "human_test_expression_used_to_align": False,
            "human_outcomes_used_to_align": False,
        },
        "premise_result_action_policy": c3_policy["actions"],
        "reserved_outcome_firewall": {
            "TARGET_OS": "CLOSED",
            "GSE21257": "CLOSED",
            "GSE39055": "CLOSED",
            "GSE16091": (
                "May open only in 04c under this exact 04b protocol."
            ),
        },
        "human_outcome_access_before_04c": False,
        "human_expression_access_before_04c": False,
        "AI_architecture_tuning_from_premise_target": "FORBIDDEN",
        "new_model_family_after_04b": (
            "FORBIDDEN for the premise test unless an observed implementation "
            "defect prevents execution without changing the scientific question."
        ),
        "upstream_hashes": {
            "02i_contract": sha256_file(I_CONTRACT),
            "03a_protocol": sha256_file(A3_PROTOCOL),
            "03b_summary": sha256_file(B3_SUMMARY),
            "03c_summary": sha256_file(C3_SUMMARY),
            "03c_premise_policy": sha256_file(C3_POLICY),
            "04a_contract": sha256_file(A4_CONTRACT),
            "04a_summary": sha256_file(A4_SUMMARY),
            "04a_roster": sha256_file(A4_ROSTER),
            "02g_bridge": sha256_file(G2_BRIDGE),
            "02h_hallmark": sha256_file(H2_HALLMARK),
        },
        "required_next": (
            "04c opens ONLY GSE16091 expression/OS under this protocol and "
            "runs the frozen paired classical premise test."
        ),
    }
    write_json(PROTOCOL_JSON, protocol)

    final_hashes = {
        "GPL96_platform_lock_json": sha256_file(GPL96_LOCK),
        "gse16091_gpl96_probe_gene_map_tsv": sha256_file(GPL96_PROBE_MAP),
        "gse16091_outcome_blind_gene_universe_tsv": sha256_file(
            GSE16091_GENE_UNIVERSE
        ),
        "gse16091_hallmark_feature_map_tsv": sha256_file(
            GSE16091_HALLMARK_MAP
        ),
        "dog2_full_source_alpha_contract_tsv": sha256_file(
            DOG_SOURCE_ALPHA
        ),
        "classical_premise_benchmark_registry_tsv": sha256_file(
            BENCHMARK_REGISTRY
        ),
        "gse16091_cv_contract_tsv": sha256_file(CV_CONTRACT),
        "premise_decision_rules_tsv": sha256_file(PREMISE_RULES),
        "classical_premise_test_protocol_json": sha256_file(PROTOCOL_JSON),
    }

    summary = {
        "script_version": SCRIPT_VERSION,
        "status": "PASS",
        "scientific_status": (
            "PASS_CLASSICAL_PREMISE_TEST_IMPLEMENTATION_FROZEN"
        ),
        "run_started_utc": started,
        "run_finished_utc": now_utc(),
        "target": SELECTED_TARGET,
        "target_n": EXPECTED_TARGET_N,
        "human_pool": "GSE16091_OUTER_TRAINING_POOL",
        "reserved_human_outcomes_used": False,
        "GPL96_platform_rows": int(len(platform)),
        "GPL96_unique_symbol_probes": int(
            (probe_map["symbol_resolution_status"] == "UNIQUE_SYMBOL").sum()
        ),
        "common_DOG2_GPL96_bridge_genes": int(len(gene_universe)),
        "Hallmark_modules": EXPECTED_HALLMARK_N,
        "Hallmark_min_common_genes": int(min(module_counts.values())),
        "Hallmark_median_common_genes": float(
            np.median(list(module_counts.values()))
        ),
        "Hallmark_max_common_genes": int(max(module_counts.values())),
        "DOG2_full_source_alpha": dog_source_alpha,
        "outer_cv": f"{OUTER_SPLITS}-fold x {OUTER_REPEATS}",
        "inner_cv": f"{INNER_SPLITS}-fold",
        "patient_bootstraps": PATIENT_BOOTSTRAPS,
        "primary_premise_models": ["B4", "B0"],
        "human_outcome_values_read": False,
        "human_event_count_read": False,
        "human_expression_values_read": False,
        "human_model_fitting": False,
        "DOG2_new_tuning": False,
        "final_artifact_hashes": final_hashes,
        "next": "04c run frozen GSE16091 classical premise test",
    }
    write_json(SUMMARY_JSON, summary)

    print("-" * 120)
    print("Outcome-blind GSE16091 / GPL96 representation")
    print("-" * 120)
    print(f"GPL96 platform rows: {len(platform):,}")
    print(
        "GPL96 uniquely-symbol-mapped probes: "
        f"{int((probe_map['symbol_resolution_status'] == 'UNIQUE_SYMBOL').sum()):,}"
    )
    print(f"02g primary DOG2 bridge genes: {EXPECTED_DOG_PRIMARY_BRIDGE_N:,}")
    print(f"Common DOG2-GPL96 bridge genes: {len(gene_universe):,}")
    print(f"Hallmark modules supported: {len(module_counts)}/{EXPECTED_HALLMARK_N}")
    print(f"Hallmark min common genes/module: {min(module_counts.values())}")
    print(
        "Hallmark median common genes/module: "
        f"{np.median(list(module_counts.values())):.1f}"
    )
    print(f"Hallmark max common genes/module: {max(module_counts.values())}")
    print()

    print("Frozen full-DOG2 source model:")
    print(f"  alpha rule: {DOG_SOURCE_ALPHA_RULE}")
    print(f"  selected alpha: {dog_source_alpha}")
    print("  new DOG2 tuning after 03b: NO")
    print()

    print("Frozen human pool:")
    print("  GSE16091 outer-training fold")
    print("  TARGET/GSE21257/GSE39055 used as source: NO")
    print()

    print("Frozen primary premise comparison:")
    print("  B4 DOG2 + human residual ridge Cox")
    print("  versus")
    print("  B0 human-only Hallmark ridge Cox")
    print()
    print("Decision thresholds [UNCHANGED FROM 02i]:")
    print("  SUPPORTS:")
    print("    delta Uno C >= +0.02")
    print("    bootstrap P(delta C > 0) >= 0.90")
    print("    delta IBS <= +0.01")
    print("  ARGUES AGAINST:")
    print("    delta Uno C <= -0.02")
    print("    bootstrap P(delta C < 0) >= 0.90")
    print("  otherwise: INCONCLUSIVE_NEUTRAL")
    print()

    print("Frozen human CV:")
    print(f"  outer: {OUTER_SPLITS}-fold x {OUTER_REPEATS}")
    print(f"  inner: {INNER_SPLITS}-fold")
    print(f"  paired patient bootstraps: {PATIENT_BOOTSTRAPS}")
    print()

    print("=" * 120)
    print("04b CLASSICAL PREMISE-TEST IMPLEMENTATION SUMMARY")
    print("=" * 120)
    print("GSE16091 outcomes read: NO")
    print("GSE16091 event count read: NO")
    print("GSE16091 expression values read: NO")
    print("GSE16091 series matrix fetched: NO")
    print("Reserved human outcomes read: NO")
    print("Human model fitting: NO")
    print("Premise thresholds changed: NO")
    print()
    print("Next: 04c opens GSE16091 expression + OS and runs the frozen paired premise test.")
    print("No TARGET/GSE21257/GSE39055 outcome access is authorized in 04c.")
    print()
    print("Artifacts:")
    for path in [
        GPL96_LOCK,
        GPL96_PROBE_MAP,
        GSE16091_GENE_UNIVERSE,
        GSE16091_HALLMARK_MAP,
        DOG_SOURCE_ALPHA,
        BENCHMARK_REGISTRY,
        CV_CONTRACT,
        PREMISE_RULES,
        PROTOCOL_JSON,
        SUMMARY_JSON,
    ]:
        print(f"  {path.relative_to(ROOT)}")
    print("=" * 120)
    print("04b classical premise-test implementation freeze: PASS")
    print("=" * 120)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 120, file=sys.stderr)
        print(
            "04b classical premise-test implementation freeze: FAIL",
            file=sys.stderr,
        )
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print("=" * 120, file=sys.stderr)
        raise
