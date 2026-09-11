#!/usr/bin/env python
"""
Paper 6 - export frozen all-model safety summary for CBM strengthening.

05h2 is a POST-HOLD DESCRIPTIVE CONTEXT export. It does not fit models,
re-estimate survival metrics from predictions, change thresholds, or reopen
the frozen A2/A3 architecture-selection decision.

It consumes already-computed synthetic metric artifacts and the authoritative
frozen scenario design, reproduces equal-scenario summaries for the seven
frozen main models, and writes reviewer-facing supplementary artifacts.

Expected repository placement:
    scripts/05h2_export_all_model_safety_summary.py

Run:
    python scripts/05h2_export_all_model_safety_summary.py
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SCRIPT_VERSION = "05h2-export-all-model-safety-summary-v2-canonical-source-no-cli"

MAIN_MODELS = ["B0", "B4", "A0", "A1", "A2", "A3", "A4"]
REFERENCE_MODEL = "B0"
ELIGIBLE_MODELS = {"A2", "A3"}

NEGATIVE_THRESHOLD = -0.02
CATASTROPHIC_THRESHOLD = -0.05

# Independent frozen/post-HOLD replay anchors. These are NOT used to create the
# results; they are acceptance checks after the existing metric artifacts have
# been summarized.
EXPECTED_EQUAL_SCENARIO_UNO_C = {
    "B0": 0.555554,
    "B4": 0.544989,
    "A0": 0.555351,
    "A1": 0.572198,
    "A2": 0.569809,
    "A3": 0.530067,
    "A4": 0.519107,
}
EXPECTED_NEGATIVE_TRANSFER_RATE = {
    "B0": 0.000000,
    "B4": 0.305528,
    "A0": 0.009000,
    "A1": 0.286778,
    "A2": 0.215417,
    "A3": 0.486333,
    "A4": 0.511722,
}
# R5 is the frozen misleading-source regime. This cross-check comes from the
# accepted post-HOLD replay and is used only as a provenance/sanity guard.
EXPECTED_R5_CATASTROPHIC_RATE = {
    "B0": 0.000000,
    "B4": 0.219936,
    "A0": 0.000000,
    "A1": 0.020513,
    "A2": 0.059551,
    "A3": 0.712179,
    "A4": 0.879744,
}
SANITY_TOL = 7.5e-6

FROZEN_ROLES = {
    "B0": "target-only classical reference",
    "B4": "classical residual-transfer comparator",
    "A0": "target-only neural flexibility control",
    "A1": "source-initialized target-head transfer branch",
    "A2": "rank-4 latent residual adapter; frozen selectable candidate",
    "A3": "module-selective borrowing; frozen selectable candidate",
    "A4": "full neural fine-tuning control",
}

EXPECTED_REGIME_COUNTS = {
    "R0": 6,
    "R1": 6,
    "R2": 78,
    "R3": 6,
    "R4": 6,
    "R5": 78,
}

TABLE_SUFFIXES = {".csv", ".tsv", ".txt", ".parquet"}
MAX_TABULAR_BYTES = 250 * 1024 * 1024

SCENARIO_ALIASES = [
    "scenario_id", "scenario", "scenario_name", "scenarioid",
]
MODEL_ALIASES = [
    "model", "model_id", "modelid", "architecture", "method",
]
REPLICATE_ALIASES = [
    "replicate", "replicate_id", "replicate_index", "rep", "rep_id",
]
REGIME_ALIASES = [
    "transfer_regime", "regime", "regime_id", "transport_regime",
]
UNO_ALIASES = [
    "uno_c", "unoc", "uno", "test_uno_c",
    "mean_uno_c", "scenario_mean_uno_c", "mean_test_uno_c",
    "equal_scenario_mean_uno_c",
]
DELTA_ALIASES = [
    "delta_c", "deltac", "delta_uno_c", "delta_unoc",
    "mean_delta_c", "mean_delta_uno_c", "mean_delta_unoc",
    "mean_delta_uno_c_vs_b0", "delta_uno_c_vs_b0",
]
NEG_RATE_ALIASES = [
    "negative_transfer_rate", "aggregate_negative_transfer_rate",
    "negative_rate", "fraction_negative_transfer",
    "negative_transfer_fraction", "nt_rate",
]
CAT_RATE_ALIASES = [
    "catastrophic_transfer_rate", "aggregate_catastrophic_transfer_rate",
    "catastrophic_rate", "fraction_catastrophic_transfer",
    "catastrophic_transfer_fraction", "cat_rate",
]
NEG_FLAG_ALIASES = [
    "negative_transfer", "negative_transfer_flag", "is_negative_transfer",
    "negative_flag",
]
CAT_FLAG_ALIASES = [
    "catastrophic_transfer", "catastrophic_transfer_flag",
    "is_catastrophic_transfer", "catastrophic_flag",
]

SCENARIO_RE = re.compile(r"\b(CORE|STRESS)[_-]?(\d{4})\b", re.IGNORECASE)
REGIME_RE = re.compile(r"\b(R[0-5])(?:\b|_)", re.IGNORECASE)


@dataclass(frozen=True)
class MetricSource:
    paths: tuple[Path, ...]
    raw: pd.DataFrame
    source_mode: str


def project_root() -> Path:
    p = Path(__file__).resolve()
    if p.parent.name.lower() == "scripts":
        return p.parent.parent
    raise RuntimeError(
        "05h2 must be placed in the repository scripts/ directory before execution."
    )


ROOT = project_root()
H0_DIR = ROOT / "method_contract" / "05h0_cbm_strengthening_contract"
H0_JSON = H0_DIR / "cbm_strengthening_contract.json"
OUT_DIR = ROOT / "method_contract" / "05h2_all_model_safety_summary"
WORK_DIR = ROOT / "method_contract" / ".05h2_all_model_safety_summary_work"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_json_bytes(obj: Any) -> bytes:
    return (
        json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def write_json(path: Path, obj: Any) -> None:
    path.write_bytes(stable_json_bytes(obj))


def normalize_col_name(x: Any) -> str:
    s = str(x).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def normalized_columns(df: pd.DataFrame) -> dict[str, str]:
    out: dict[str, str] = {}
    for c in df.columns:
        n = normalize_col_name(c)
        if n and n not in out:
            out[n] = c
    return out


def find_col(df: pd.DataFrame, aliases: Iterable[str]) -> str | None:
    cols = normalized_columns(df)
    for a in aliases:
        aa = normalize_col_name(a)
        if aa in cols:
            return cols[aa]
    return None


def scenario_from_text(x: Any) -> str | None:
    m = SCENARIO_RE.search(str(x))
    if not m:
        return None
    return f"{m.group(1).upper()}_{int(m.group(2)):04d}"


def normalize_regime(x: Any) -> str | None:
    m = REGIME_RE.search(str(x).strip().upper())
    if not m:
        return None
    return m.group(1).upper()


def numeric_series(s: pd.Series, label: str) -> pd.Series:
    z = pd.to_numeric(s, errors="coerce")
    if z.isna().all():
        raise RuntimeError(f"Column {label!r} cannot be interpreted numerically.")
    return z.astype(float)


def bool_to_float(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.astype(float)
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() > 0.95:
        return num.astype(float)
    txt = s.astype(str).str.strip().str.lower()
    mapping = {
        "true": 1.0, "false": 0.0,
        "yes": 1.0, "no": 0.0,
        "1": 1.0, "0": 0.0,
    }
    return txt.map(mapping).astype(float)


def read_table(path: Path, nrows: int | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        # Parquet header-only reading is backend-dependent; tables used here are
        # small metric artifacts, so a full read is acceptable.
        df = pd.read_parquet(path)
        return df if nrows is None else df.head(nrows).copy()

    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", nrows=nrows, low_memory=False)

    if suffix in {".csv", ".txt"}:
        # Prefer delimiter from extension; for .txt, infer from a small sample.
        if suffix == ".csv":
            return pd.read_csv(path, nrows=nrows, low_memory=False)
        try:
            return pd.read_csv(path, sep="\t", nrows=nrows, low_memory=False)
        except Exception:
            return pd.read_csv(path, nrows=nrows, low_memory=False)

    raise ValueError(f"Unsupported table type: {path}")


def iter_candidate_table_paths() -> list[Path]:
    results = ROOT / "results"
    if not results.exists():
        raise FileNotFoundError(f"Missing results directory: {results}")

    paths: list[Path] = []
    for p in results.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in TABLE_SUFFIXES:
            continue
        try:
            size = p.stat().st_size
        except OSError:
            continue
        if size <= 0 or size > MAX_TABULAR_BYTES:
            continue

        # Exclude final 05h outputs if the user is recovering from a partial copy.
        low = str(p).lower()
        if "05h2_all_model_safety_summary" in low:
            continue
        paths.append(p)
    return sorted(set(paths))


def header_looks_metric_like(df: pd.DataFrame, path: Path) -> bool:
    model_col = find_col(df, MODEL_ALIASES)
    if model_col is None:
        return False

    scenario_col = find_col(df, SCENARIO_ALIASES)
    path_sid = scenario_from_text(path.name)
    if scenario_col is None and path_sid is None:
        return False

    metric_cols = [
        find_col(df, UNO_ALIASES),
        find_col(df, DELTA_ALIASES),
        find_col(df, NEG_RATE_ALIASES),
        find_col(df, CAT_RATE_ALIASES),
        find_col(df, NEG_FLAG_ALIASES),
        find_col(df, CAT_FLAG_ALIASES),
    ]
    return any(x is not None for x in metric_cols)


def canonicalize_raw(df: pd.DataFrame, source_path: Path) -> pd.DataFrame:
    if df.empty:
        raise RuntimeError(f"Empty metric table: {source_path}")

    model_col = find_col(df, MODEL_ALIASES)
    if model_col is None:
        raise RuntimeError(f"No model column in candidate metric table: {source_path}")

    scenario_col = find_col(df, SCENARIO_ALIASES)
    replicate_col = find_col(df, REPLICATE_ALIASES)
    regime_col = find_col(df, REGIME_ALIASES)
    uno_col = find_col(df, UNO_ALIASES)
    delta_col = find_col(df, DELTA_ALIASES)
    neg_rate_col = find_col(df, NEG_RATE_ALIASES)
    cat_rate_col = find_col(df, CAT_RATE_ALIASES)
    neg_flag_col = find_col(df, NEG_FLAG_ALIASES)
    cat_flag_col = find_col(df, CAT_FLAG_ALIASES)

    out = pd.DataFrame(index=df.index)
    out["model"] = df[model_col].astype(str).str.strip().str.upper()

    if scenario_col is not None:
        out["scenario_id"] = df[scenario_col].map(scenario_from_text)
        missing = out["scenario_id"].isna()
        if missing.any():
            # Some source tables may already use exact non-prefixed scenario
            # identifiers. Preserve them only if they parse after string cleanup.
            alt = df.loc[missing, scenario_col].astype(str).str.strip()
            alt_parsed = alt.map(scenario_from_text)
            out.loc[missing, "scenario_id"] = alt_parsed
    else:
        sid = scenario_from_text(source_path.name)
        if sid is None:
            sid = scenario_from_text(str(source_path))
        if sid is None:
            raise RuntimeError(f"Cannot infer scenario ID from {source_path}")
        out["scenario_id"] = sid

    if replicate_col is not None:
        out["replicate"] = df[replicate_col]
    if regime_col is not None:
        out["regime"] = df[regime_col].map(normalize_regime)

    if uno_col is not None:
        out["uno_c"] = numeric_series(df[uno_col], uno_col)
    if delta_col is not None:
        out["delta_uno_c"] = numeric_series(df[delta_col], delta_col)
    if neg_rate_col is not None:
        out["negative_rate"] = numeric_series(df[neg_rate_col], neg_rate_col)
    if cat_rate_col is not None:
        out["catastrophic_rate"] = numeric_series(df[cat_rate_col], cat_rate_col)
    if neg_flag_col is not None:
        out["negative_flag"] = bool_to_float(df[neg_flag_col])
    if cat_flag_col is not None:
        out["catastrophic_flag"] = bool_to_float(df[cat_flag_col])

    out = out[out["model"].isin(MAIN_MODELS)].copy()
    out = out[out["scenario_id"].notna()].copy()
    if out.empty:
        raise RuntimeError(f"No main-model scenario rows remain in {source_path}")
    return out.reset_index(drop=True)


def scientific_signature(df: pd.DataFrame) -> str:
    cols = [
        c for c in [
            "scenario_id", "replicate", "model", "uno_c", "delta_uno_c",
            "negative_rate", "catastrophic_rate", "negative_flag",
            "catastrophic_flag", "regime",
        ]
        if c in df.columns
    ]
    z = df[cols].copy()
    for c in z.columns:
        if c not in {"scenario_id", "model", "regime"}:
            z[c] = z[c].astype(str)
    sort_cols = [c for c in ["scenario_id", "replicate", "model"] if c in z.columns]
    if sort_cols:
        z = z.sort_values(sort_cols, kind="mergesort")
    data = z.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def resolve_metric_source() -> MetricSource:
    """Resolve the frozen primary 05d metric summary by scientific role, not by values.

    05h2 v1 used a deliberately broad discovery pass and therefore treated the
    05d0a *diagnostic* table as a competing scientific metric artifact. That is a
    source-classification bug: 05d0a belongs to the diagnostics branch, whereas
    the frozen 180-scenario phase-diagram summary is the primary 05d artifact
    below. v2 binds the source path explicitly and FAILs if it is unavailable; it
    never substitutes a diagnostic or post-hoc proxy.
    """
    canonical_path = (
        ROOT
        / "results"
        / "simulation_phase_diagram"
        / "05d"
        / "scenario_model_metric_summary.tsv"
    )
    diagnostic_path = (
        ROOT
        / "results"
        / "simulation_phase_diagram_diagnostics"
        / "05d0a"
        / "scenario_failure_diagnostics.tsv"
    )

    if not canonical_path.exists():
        raise FileNotFoundError(
            "Authoritative frozen 05d scenario-model metric summary is missing: "
            f"{canonical_path}\n"
            "05h2 will not substitute a diagnostic or post-hoc proxy."
        )

    try:
        raw = canonicalize_raw(read_table(canonical_path), canonical_path)
    except Exception as exc:
        raise RuntimeError(
            "The authoritative frozen 05d metric summary exists but could not be "
            f"parsed under the frozen 05h2 schema: {canonical_path}"
        ) from exc

    models = set(raw["model"])
    n_scen = raw["scenario_id"].nunique()
    if not set(MAIN_MODELS).issubset(models):
        raise RuntimeError(
            "Authoritative 05d metric summary does not contain all seven frozen "
            f"main models. Found: {sorted(models)}"
        )
    if n_scen != 180:
        raise RuntimeError(
            f"Authoritative 05d metric summary must contain 180 scenarios; found {n_scen}."
        )
    if len(raw) != 180 * len(MAIN_MODELS):
        raise RuntimeError(
            "Authoritative 05d metric summary is expected to be one row per "
            f"scenario-model (1260 rows); found {len(raw)}."
        )
    if raw.duplicated(["scenario_id", "model"]).any():
        raise RuntimeError(
            "Authoritative 05d metric summary is not unique by scenario_id/model."
        )

    # Explicitly document the table that triggered the v1 false ambiguity. Its
    # existence is harmless, but it is never read as a scientific result source.
    if diagnostic_path.exists():
        print(
            "  diagnostic 05d0a table present and intentionally excluded from "
            "scientific source resolution"
        )
        print(f"  excluded diagnostic artifact: {diagnostic_path.relative_to(ROOT)}")

    return MetricSource(
        paths=(canonical_path,),
        raw=raw,
        source_mode="canonical_frozen_05d_scenario_model_metric_summary",
    )


def ensure_delta_from_uno(raw: pd.DataFrame) -> pd.DataFrame:
    x = raw.copy()

    if "delta_uno_c" in x.columns and x["delta_uno_c"].notna().all():
        return x

    if "uno_c" not in x.columns:
        return x

    # Replicate-level pairing if a stable replicate identifier exists.
    if "replicate" in x.columns and x.duplicated(["scenario_id", "model"]).any():
        key = ["scenario_id", "replicate"]
        b0 = (
            x[x["model"] == REFERENCE_MODEL][key + ["uno_c"]]
            .rename(columns={"uno_c": "_b0_uno"})
        )
        if b0.duplicated(key).any():
            return x
        x = x.merge(b0, on=key, how="left", validate="many_to_one")
        if x["_b0_uno"].notna().all():
            x["delta_uno_c"] = x["uno_c"] - x["_b0_uno"]
            x.drop(columns=["_b0_uno"], inplace=True)
            return x

    # Scenario-level pairing.
    if not x.duplicated(["scenario_id", "model"]).any():
        b0 = (
            x[x["model"] == REFERENCE_MODEL][["scenario_id", "uno_c"]]
            .rename(columns={"uno_c": "_b0_uno"})
        )
        if not b0.duplicated(["scenario_id"]).any():
            x = x.merge(b0, on="scenario_id", how="left", validate="many_to_one")
            if x["_b0_uno"].notna().all():
                x["delta_uno_c"] = x["uno_c"] - x["_b0_uno"]
                x.drop(columns=["_b0_uno"], inplace=True)
                return x

    return x


def to_scenario_model(raw: pd.DataFrame) -> pd.DataFrame:
    x = ensure_delta_from_uno(raw)

    if "delta_uno_c" in x.columns:
        if "negative_flag" not in x.columns:
            x["negative_flag"] = (x["delta_uno_c"] <= NEGATIVE_THRESHOLD).astype(float)
        if "catastrophic_flag" not in x.columns:
            x["catastrophic_flag"] = (
                x["delta_uno_c"] <= CATASTROPHIC_THRESHOLD
            ).astype(float)

    replicate_level = x.duplicated(["scenario_id", "model"]).any()

    if replicate_level:
        required = ["uno_c", "delta_uno_c"]
        missing = [c for c in required if c not in x.columns]
        if missing:
            raise RuntimeError(
                "Resolved metric artifact is replicate-level but lacks required "
                f"already-computed discrimination quantities: {missing}"
            )

        if "negative_flag" not in x.columns or "catastrophic_flag" not in x.columns:
            raise RuntimeError(
                "Cannot derive safety rates because replicate-level delta/flag data "
                "are incomplete."
            )

        agg = (
            x.groupby(["scenario_id", "model"], as_index=False)
            .agg(
                mean_uno_c=("uno_c", "mean"),
                mean_delta_uno_c=("delta_uno_c", "mean"),
                negative_transfer_rate=("negative_flag", "mean"),
                catastrophic_transfer_rate=("catastrophic_flag", "mean"),
                n_metric_rows=("model", "size"),
            )
        )
    else:
        agg = x[["scenario_id", "model"]].copy()

        if "uno_c" in x.columns:
            agg["mean_uno_c"] = x["uno_c"].to_numpy()
        if "delta_uno_c" in x.columns:
            agg["mean_delta_uno_c"] = x["delta_uno_c"].to_numpy()

        if "negative_rate" in x.columns:
            agg["negative_transfer_rate"] = x["negative_rate"].to_numpy()
        elif "negative_flag" in x.columns:
            # Only valid if each row itself is a replicate; but uniqueness tells us
            # this is scenario-level. A 0/1 scenario flag is not a replicate rate.
            raise RuntimeError(
                "Scenario-level artifact contains only a negative flag, not a "
                "scenario negative-transfer rate."
            )

        if "catastrophic_rate" in x.columns:
            agg["catastrophic_transfer_rate"] = x["catastrophic_rate"].to_numpy()
        elif "catastrophic_flag" in x.columns:
            raise RuntimeError(
                "Scenario-level artifact contains only a catastrophic flag, not a "
                "scenario catastrophic-transfer rate."
            )
        agg["n_metric_rows"] = 1

    # Derive scenario-level delta from scenario mean Uno-C only when necessary.
    if "mean_delta_uno_c" not in agg.columns:
        if "mean_uno_c" not in agg.columns:
            raise RuntimeError("No existing Uno-C or delta-C quantity is available.")
        b0 = (
            agg[agg["model"] == REFERENCE_MODEL][["scenario_id", "mean_uno_c"]]
            .rename(columns={"mean_uno_c": "_b0"})
        )
        agg = agg.merge(b0, on="scenario_id", how="left", validate="many_to_one")
        agg["mean_delta_uno_c"] = agg["mean_uno_c"] - agg["_b0"]
        agg.drop(columns=["_b0"], inplace=True)

    missing_safety = [
        c for c in ["negative_transfer_rate", "catastrophic_transfer_rate"]
        if c not in agg.columns
    ]
    if missing_safety:
        raise RuntimeError(
            "The existing scenario-level artifact does not contain replicate-level "
            f"safety information needed for {missing_safety}. 05h2 will not "
            "re-estimate survival metrics from prediction arrays."
        )

    # Reference-model safety is definitionally zero; enforce exactness.
    b0mask = agg["model"] == REFERENCE_MODEL
    agg.loc[b0mask, "mean_delta_uno_c"] = 0.0
    agg.loc[b0mask, "negative_transfer_rate"] = 0.0
    agg.loc[b0mask, "catastrophic_transfer_rate"] = 0.0

    required_cols = [
        "mean_delta_uno_c",
        "negative_transfer_rate",
        "catastrophic_transfer_rate",
    ]
    if "mean_uno_c" not in agg.columns:
        # Mean Uno-C is a useful reviewer-facing context field and is expected in
        # the accepted metric artifacts for this project.
        raise RuntimeError("Existing metric artifact lacks scenario-level mean Uno-C.")

    for c in ["mean_uno_c"] + required_cols:
        agg[c] = pd.to_numeric(agg[c], errors="coerce")
        if not np.isfinite(agg[c].to_numpy(dtype=float)).all():
            raise RuntimeError(f"Non-finite values in scenario-model field: {c}")

    if agg.duplicated(["scenario_id", "model"]).any():
        raise RuntimeError("Scenario-model table is not unique after aggregation.")

    scenarios = sorted(agg["scenario_id"].unique())
    if len(scenarios) != 180:
        raise RuntimeError(f"Expected 180 scenarios; found {len(scenarios)}.")

    for model in MAIN_MODELS:
        n = int((agg["model"] == model).sum())
        if n != 180:
            raise RuntimeError(f"{model}: expected 180 scenario rows; found {n}.")

    if len(agg) != 180 * len(MAIN_MODELS):
        raise RuntimeError(
            f"Expected {180 * len(MAIN_MODELS)} main-model scenario rows; found {len(agg)}."
        )

    return agg.sort_values(["scenario_id", "model"], kind="mergesort").reset_index(drop=True)


def recursive_records(obj: Any) -> Iterable[list[dict[str, Any]]]:
    if isinstance(obj, list) and obj and all(isinstance(v, dict) for v in obj):
        yield obj
        for item in obj:
            yield from recursive_records(item)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from recursive_records(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from recursive_records(v)


def mapping_from_df(df: pd.DataFrame) -> pd.DataFrame | None:
    scenario_col = find_col(df, SCENARIO_ALIASES)
    regime_col = find_col(df, REGIME_ALIASES)
    if scenario_col is None or regime_col is None:
        return None

    z = pd.DataFrame(
        {
            "scenario_id": df[scenario_col].map(scenario_from_text),
            "regime": df[regime_col].map(normalize_regime),
        }
    ).dropna()
    if z.empty:
        return None

    z = z.drop_duplicates()
    if z["scenario_id"].duplicated().any():
        return None
    if z["scenario_id"].nunique() != 180:
        return None
    if set(z["regime"]) != set(EXPECTED_REGIME_COUNTS):
        return None
    return z.sort_values("scenario_id").reset_index(drop=True)


def resolve_authoritative_regime_mapping(
    metric_source: MetricSource,
) -> tuple[pd.DataFrame, tuple[Path, ...], str]:
    preferred_roots = [
        ROOT / "results" / "simulation_contract" / "05a",
        ROOT / "results" / "simulation_contract",
        ROOT / "contracts",
        ROOT / "manifests",
    ]
    candidates: list[tuple[Path, pd.DataFrame]] = []

    # First inspect dedicated frozen design/contract locations.
    for base in preferred_roots:
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() in TABLE_SUFFIXES:
                try:
                    m = mapping_from_df(read_table(p))
                except Exception:
                    m = None
                if m is not None:
                    candidates.append((p, m))
            elif p.suffix.lower() == ".json" and p.stat().st_size <= 50 * 1024 * 1024:
                try:
                    obj = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    continue
                for records in recursive_records(obj):
                    try:
                        m = mapping_from_df(pd.DataFrame(records))
                    except Exception:
                        m = None
                    if m is not None:
                        candidates.append((p, m))
                        break

    # If dedicated locations did not expose a 180-row design table, allow the
    # already-resolved metric artifact to supply the labels only if it contains a
    # complete one-to-one mapping. This is logged as a fallback, not silently used.
    metric_map = None
    if "regime" in metric_source.raw.columns:
        mm = (
            metric_source.raw[["scenario_id", "regime"]]
            .dropna()
            .drop_duplicates()
        )
        if not mm["scenario_id"].duplicated().any() and mm["scenario_id"].nunique() == 180:
            if set(mm["regime"]) == set(EXPECTED_REGIME_COUNTS):
                metric_map = mm.sort_values("scenario_id").reset_index(drop=True)

    if not candidates and metric_map is None:
        raise RuntimeError(
            "Could not resolve the frozen R0-R5 scenario mapping from the authoritative "
            "simulation-design/contract artifacts, and the accepted metric artifact "
            "does not carry a complete regime mapping."
        )

    if candidates:
        signatures: dict[str, list[tuple[Path, pd.DataFrame]]] = {}
        for p, m in candidates:
            sig = hashlib.sha256(
                m.to_csv(index=False, lineterminator="\n").encode("utf-8")
            ).hexdigest()
            signatures.setdefault(sig, []).append((p, m))
        if len(signatures) != 1:
            details = "\n".join(str(p) for p, _ in candidates)
            raise RuntimeError(
                "Non-equivalent frozen scenario-to-regime mappings were found; "
                "refusing post-hoc source choice.\n" + details
            )
        equiv = sorted(next(iter(signatures.values())), key=lambda x: str(x[0]).lower())
        mapping = equiv[0][1]
        paths = tuple(p for p, _ in equiv)
        mode = "authoritative_frozen_scenario_design"

        if metric_map is not None:
            check = mapping.merge(
                metric_map, on="scenario_id", suffixes=("_design", "_metric"),
                validate="one_to_one",
            )
            if not (check["regime_design"] == check["regime_metric"]).all():
                raise RuntimeError(
                    "Metric-artifact regime labels disagree with the authoritative "
                    "frozen scenario design."
                )
    else:
        mapping = metric_map
        assert mapping is not None
        paths = metric_source.paths
        mode = "metric_artifact_regime_fallback"

    counts = mapping["regime"].value_counts().to_dict()
    if counts != EXPECTED_REGIME_COUNTS:
        raise RuntimeError(
            f"Frozen regime counts mismatch. Expected {EXPECTED_REGIME_COUNTS}; found {counts}."
        )

    return mapping, paths, mode


def load_and_verify_h0() -> dict[str, Any]:
    if not H0_JSON.exists():
        raise FileNotFoundError(f"Missing frozen 05h0 contract: {H0_JSON}")

    contract = json.loads(H0_JSON.read_text(encoding="utf-8"))

    def find_named_block(obj: Any, key: str) -> Any:
        if isinstance(obj, dict):
            if key in obj:
                return obj[key]
            for v in obj.values():
                hit = find_named_block(v, key)
                if hit is not None:
                    return hit
        elif isinstance(obj, list):
            for v in obj:
                hit = find_named_block(v, key)
                if hit is not None:
                    return hit
        return None

    block = find_named_block(contract, "all_model_safety_summary")
    if not isinstance(block, dict):
        raise RuntimeError("05h0 does not contain all_model_safety_summary.")

    if block.get("analysis_role") != "POST-HOLD DESCRIPTIVE CONTEXT":
        raise RuntimeError("05h0 all-model analysis role is not the frozen expected role.")
    if block.get("models") != MAIN_MODELS:
        raise RuntimeError(
            f"05h0 model registry mismatch: expected {MAIN_MODELS}, got {block.get('models')}"
        )
    if block.get("reference_model") != REFERENCE_MODEL:
        raise RuntimeError("05h0 reference model is not B0.")

    required_fields = {
        "model",
        "frozen_role",
        "eligible_for_primary_selection",
        "mean_delta_uno_c_vs_B0_if_defined",
        "negative_transfer_rate_at_deltaC_le_-0.02",
        "catastrophic_transfer_rate_at_deltaC_le_-0.05",
        "regime_specific_rates_or_effects_if_already_estimable",
    }
    actual_fields = set(block.get("minimum_reported_fields", []))
    if not required_fields.issubset(actual_fields):
        raise RuntimeError(
            "05h0 minimum all-model report fields do not match the frozen 05h2 contract."
        )
    return contract


def aggregate_overall(scenario_model: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MAIN_MODELS:
        z = scenario_model[scenario_model["model"] == model]
        rows.append(
            {
                "model": model,
                "frozen_role": FROZEN_ROLES[model],
                "eligible_for_primary_selection": model in ELIGIBLE_MODELS,
                "equal_scenario_mean_uno_c": float(z["mean_uno_c"].mean()),
                "mean_delta_uno_c_vs_B0_if_defined": float(z["mean_delta_uno_c"].mean()),
                "negative_transfer_rate_at_deltaC_le_-0.02": float(
                    z["negative_transfer_rate"].mean()
                ),
                "catastrophic_transfer_rate_at_deltaC_le_-0.05": float(
                    z["catastrophic_transfer_rate"].mean()
                ),
                "n_frozen_scenarios": int(len(z)),
                "aggregation": "mean within frozen scenario, then equal weight across 180 scenarios",
                "analysis_role": "POST-HOLD DESCRIPTIVE CONTEXT",
            }
        )
    return pd.DataFrame(rows)


def aggregate_by_regime(
    scenario_model: pd.DataFrame, regime_map: pd.DataFrame
) -> pd.DataFrame:
    x = scenario_model.merge(
        regime_map, on="scenario_id", how="left", validate="many_to_one"
    )
    if x["regime"].isna().any():
        raise RuntimeError("Missing frozen regime labels after scenario mapping.")

    rows = []
    for regime in ["R0", "R1", "R2", "R3", "R4", "R5"]:
        for model in MAIN_MODELS:
            z = x[(x["regime"] == regime) & (x["model"] == model)]
            rows.append(
                {
                    "regime": regime,
                    "model": model,
                    "frozen_role": FROZEN_ROLES[model],
                    "eligible_for_primary_selection": model in ELIGIBLE_MODELS,
                    "n_frozen_scenarios": int(len(z)),
                    "mean_uno_c": float(z["mean_uno_c"].mean()),
                    "mean_delta_uno_c_vs_B0": float(z["mean_delta_uno_c"].mean()),
                    "negative_transfer_rate_at_deltaC_le_-0.02": float(
                        z["negative_transfer_rate"].mean()
                    ),
                    "catastrophic_transfer_rate_at_deltaC_le_-0.05": float(
                        z["catastrophic_transfer_rate"].mean()
                    ),
                    "aggregation": "mean within frozen scenario, then equal weight within regime",
                    "analysis_role": "POST-HOLD DESCRIPTIVE CONTEXT",
                }
            )
    return pd.DataFrame(rows)


def sanity_check(overall: pd.DataFrame, regime: pd.DataFrame) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def record(name: str, observed: float, expected: float, tol: float = SANITY_TOL) -> None:
        ok = math.isfinite(observed) and abs(observed - expected) <= tol
        checks.append(
            {
                "check": name,
                "observed": observed,
                "expected": expected,
                "absolute_difference": abs(observed - expected),
                "tolerance": tol,
                "pass": bool(ok),
            }
        )

    o = overall.set_index("model")
    for model in MAIN_MODELS:
        record(
            f"{model}_equal_scenario_mean_uno_c",
            float(o.loc[model, "equal_scenario_mean_uno_c"]),
            EXPECTED_EQUAL_SCENARIO_UNO_C[model],
        )
        record(
            f"{model}_negative_transfer_rate",
            float(o.loc[model, "negative_transfer_rate_at_deltaC_le_-0.02"]),
            EXPECTED_NEGATIVE_TRANSFER_RATE[model],
        )

    r5 = regime[regime["regime"] == "R5"].set_index("model")
    for model in MAIN_MODELS:
        record(
            f"{model}_R5_catastrophic_rate",
            float(r5.loc[model, "catastrophic_transfer_rate_at_deltaC_le_-0.05"]),
            EXPECTED_R5_CATASTROPHIC_RATE[model],
        )

    # Exact structural checks.
    b0 = o.loc["B0"]
    exact_checks = {
        "B0_delta_exact_zero": float(b0["mean_delta_uno_c_vs_B0_if_defined"]) == 0.0,
        "B0_NT_exact_zero": float(
            b0["negative_transfer_rate_at_deltaC_le_-0.02"]
        ) == 0.0,
        "B0_catastrophic_exact_zero": float(
            b0["catastrophic_transfer_rate_at_deltaC_le_-0.05"]
        ) == 0.0,
        "only_A2_A3_eligible": set(
            overall.loc[overall["eligible_for_primary_selection"], "model"]
        ) == ELIGIBLE_MODELS,
    }
    for name, ok in exact_checks.items():
        checks.append(
            {
                "check": name,
                "observed": bool(ok),
                "expected": True,
                "absolute_difference": None,
                "tolerance": None,
                "pass": bool(ok),
            }
        )

    failed = [c for c in checks if not c["pass"]]
    if failed:
        msg = "\n".join(
            f"  {c['check']}: observed={c['observed']} expected={c['expected']}"
            for c in failed
        )
        raise RuntimeError(
            "05h2 independent replay guard failed. Existing artifacts do not "
            "reproduce the accepted frozen/post-HOLD safety quantities:\n" + msg
        )
    return checks


def build_readme(
    metric_source: MetricSource,
    regime_mode: str,
    overall: pd.DataFrame,
) -> str:
    a2 = overall.set_index("model").loc["A2"]
    a3 = overall.set_index("model").loc["A3"]
    return f"""Paper 6 - Supplementary data for 05h2 all-model safety summary

Analysis role
-------------
POST-HOLD DESCRIPTIVE CONTEXT. This export does not alter the frozen primary
architecture-selection result and does not make any non-eligible model eligible.

Frozen model family
-------------------
B0, B4, A0, A1, A2, A3, A4. Only A2 and A3 were eligible under the original
architecture-selection rule. B0 is the target-only reference.

Safety definitions
------------------
Negative transfer: delta Uno-C <= -0.02.
Catastrophic negative transfer: delta Uno-C <= -0.05.
The thresholds are unchanged.

Aggregation
-----------
Metrics are summarized within each already-frozen scenario, then all 180
scenarios receive equal weight. Regime-specific summaries use the frozen R0-R5
scenario labels and equal weight per scenario within each regime.

Source policy
-------------
Existing already-computed synthetic metric artifacts only.
Metric-source mode: {metric_source.source_mode}
Regime-label source mode: {regime_mode}
No model fitting, no survival-metric recomputation from saved prediction arrays,
no TARGET/GSE21257/GSE39055 outcomes, and no retrospective threshold changes.

Key replay anchors
------------------
A2 negative-transfer rate: {a2['negative_transfer_rate_at_deltaC_le_-0.02']:.6f}
A3 negative-transfer rate: {a3['negative_transfer_rate_at_deltaC_le_-0.02']:.6f}

Interpretation
--------------
The complete model-family table is reviewer-facing context. It cannot reopen the
frozen HOLD_NO_SAFE_SELECTABLE_ARCHITECTURE result. Non-eligible models remain
descriptive comparators/controls.
"""


def main() -> None:
    print("=" * 118)
    print("Paper 6 - export frozen all-model safety summary for CBM strengthening")
    print("=" * 118)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {ROOT}")
    print(f"Output directory: {OUT_DIR}")
    print()
    print("Safety / execution contract:")
    print("  TARGET survival outcomes read: NO")
    print("  GSE21257 outcomes read: NO")
    print("  GSE39055 outcomes read: NO")
    print("  DOG2 outcomes read: NO")
    print("  model fitting / retuning: NO")
    print("  survival-metric re-estimation from prediction arrays: NO")
    print("  frozen thresholds changed: NO")
    print("  frozen A2/A3 selection decision reopened: NO")
    print("  existing synthetic metric artifacts read: YES")
    print("  authoritative frozen scenario design read: YES")
    print()

    if OUT_DIR.exists():
        raise RuntimeError(
            f"Final 05h2 output already exists; refusing overwrite/rerun: {OUT_DIR}"
        )
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir(parents=True, exist_ok=False)

    try:
        contract = load_and_verify_h0()
        print("05h0 all-model contract verification: PASS")
        print(f"  05h0 SHA256: {sha256_file(H0_JSON)}")
        print(f"  models: {', '.join(MAIN_MODELS)}")
        print(f"  reference: {REFERENCE_MODEL}")
        print(f"  selectable models remain: {', '.join(sorted(ELIGIBLE_MODELS))}")
        print()

        print("Resolving authoritative already-computed synthetic metric artifacts...")
        metric_source = resolve_metric_source()
        print(f"Metric source resolution: PASS [{metric_source.source_mode}]")
        print(f"  physical source artifact(s): {len(metric_source.paths)}")
        for p in metric_source.paths[:8]:
            print(f"  - {p.relative_to(ROOT)}")
        if len(metric_source.paths) > 8:
            print(f"  ... plus {len(metric_source.paths) - 8} additional equivalent/scenario artifacts")
        print()

        scenario_model = to_scenario_model(metric_source.raw)
        print("Existing metric-table normalization: PASS")
        print(f"  frozen scenarios: {scenario_model['scenario_id'].nunique()}/180")
        print(f"  main models: {scenario_model['model'].nunique()}/7")
        print(f"  scenario-model rows: {len(scenario_model)}")
        print()

        regime_map, regime_sources, regime_mode = resolve_authoritative_regime_mapping(
            metric_source
        )
        print(f"Frozen R0-R5 regime mapping: PASS [{regime_mode}]")
        for r in ["R0", "R1", "R2", "R3", "R4", "R5"]:
            print(f"  {r}: {(regime_map['regime'] == r).sum()} scenarios")
        print()

        overall = aggregate_overall(scenario_model)
        regime = aggregate_by_regime(scenario_model, regime_map)
        checks = sanity_check(overall, regime)

        print("-" * 118)
        print("05h2 ALL-MODEL DESCRIPTIVE SAFETY SUMMARY")
        print("-" * 118)
        show_cols = [
            "model",
            "eligible_for_primary_selection",
            "equal_scenario_mean_uno_c",
            "mean_delta_uno_c_vs_B0_if_defined",
            "negative_transfer_rate_at_deltaC_le_-0.02",
            "catastrophic_transfer_rate_at_deltaC_le_-0.05",
        ]
        print(overall[show_cols].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
        print()
        print("Independent frozen/post-HOLD replay checks: PASS")
        print(f"  checks passed: {sum(bool(c['pass']) for c in checks)}/{len(checks)}")
        print("  frozen primary decision changed: NO")
        print("  non-eligible models made selectable: NO")
        print()

        # Reviewer-facing outputs.
        overall_path = WORK_DIR / "Supplementary_Table_all_model_safety_summary.csv"
        regime_path = WORK_DIR / "Supplementary_Data_all_model_regime_safety.csv"
        scenario_path = WORK_DIR / "Supplementary_Data_all_model_scenario_safety.csv"
        inventory_path = WORK_DIR / "source_artifact_inventory.csv"
        checks_path = WORK_DIR / "replay_sanity_checks.csv"
        summary_path = WORK_DIR / "all_model_safety_summary.json"
        readme_path = WORK_DIR / "README_for_supplement.txt"

        overall.to_csv(overall_path, index=False, float_format="%.10g")
        regime.to_csv(regime_path, index=False, float_format="%.10g")

        scenario_export = scenario_model.merge(
            regime_map, on="scenario_id", how="left", validate="many_to_one"
        )[
            [
                "scenario_id", "regime", "model", "mean_uno_c", "mean_delta_uno_c",
                "negative_transfer_rate", "catastrophic_transfer_rate",
                "n_metric_rows",
            ]
        ].copy()
        scenario_export.to_csv(scenario_path, index=False, float_format="%.10g")
        pd.DataFrame(checks).to_csv(checks_path, index=False)

        source_rows = []
        seen = set()
        for role, paths in [
            ("05h0_frozen_contract", (H0_JSON,)),
            ("existing_synthetic_metric_artifact", metric_source.paths),
            ("authoritative_scenario_regime_source", regime_sources),
        ]:
            for p in paths:
                p = Path(p).resolve()
                key = (role, str(p).lower())
                if key in seen:
                    continue
                seen.add(key)
                source_rows.append(
                    {
                        "source_role": role,
                        "relative_path": str(p.relative_to(ROOT)),
                        "sha256": sha256_file(p),
                        "bytes": p.stat().st_size,
                    }
                )
        source_inventory = pd.DataFrame(source_rows)
        source_inventory.to_csv(inventory_path, index=False)

        summary_payload = {
            "script_version": SCRIPT_VERSION,
            "analysis_role": "POST-HOLD DESCRIPTIVE CONTEXT",
            "status": "PASS_FROZEN_ALL_MODEL_SAFETY_SUMMARY_EXPORTED",
            "frozen_primary_decision": "HOLD_NO_SAFE_SELECTABLE_ARCHITECTURE",
            "primary_decision_changed": False,
            "models": MAIN_MODELS,
            "reference_model": REFERENCE_MODEL,
            "eligible_for_primary_selection": sorted(ELIGIBLE_MODELS),
            "negative_transfer_threshold_delta_uno_c": NEGATIVE_THRESHOLD,
            "catastrophic_transfer_threshold_delta_uno_c": CATASTROPHIC_THRESHOLD,
            "aggregation": {
                "original_frozen_estimand": "equal weight per frozen scenario",
                "n_scenarios": 180,
                "regime_counts": EXPECTED_REGIME_COUNTS,
            },
            "metric_source_mode": metric_source.source_mode,
            "regime_source_mode": regime_mode,
            "all_model_summary": overall.to_dict(orient="records"),
            "replay_checks_passed": True,
            "human_outcomes_read": False,
            "model_fitting": False,
            "survival_metric_reestimation_from_predictions": False,
            "interpretation_rule": (
                "Non-eligible models remain non-eligible. These results are descriptive "
                "and cannot reopen the frozen primary architecture-selection decision."
            ),
        }
        write_json(summary_path, summary_payload)
        readme_path.write_text(
            build_readme(metric_source, regime_mode, overall),
            encoding="utf-8",
            newline="\n",
        )

        # Freeze manifest is computed after every scientific/export artifact exists.
        output_files = sorted(
            p for p in WORK_DIR.iterdir()
            if p.is_file() and p.name != "freeze_manifest.json"
        )
        manifest = {
            "script_version": SCRIPT_VERSION,
            "script_relative_path": str(Path(__file__).resolve().relative_to(ROOT)),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "05h0_contract_sha256": sha256_file(H0_JSON),
            "status": "PASS_FROZEN_ALL_MODEL_SAFETY_SUMMARY_EXPORTED",
            "outputs": [
                {
                    "file": p.name,
                    "sha256": sha256_file(p),
                    "bytes": p.stat().st_size,
                }
                for p in output_files
            ],
        }
        write_json(WORK_DIR / "freeze_manifest.json", manifest)

        # Final consistency check before atomic commit.
        for item in manifest["outputs"]:
            p = WORK_DIR / item["file"]
            if sha256_file(p) != item["sha256"]:
                raise RuntimeError(f"Output hash changed before commit: {p}")

        os.replace(WORK_DIR, OUT_DIR)

        print("=" * 118)
        print("05h2 frozen all-model safety summary: PASS")
        print("=" * 118)
        for p in sorted(OUT_DIR.iterdir()):
            if p.is_file():
                print(f"Created: {p}")
    except Exception:
        if WORK_DIR.exists():
            shutil.rmtree(WORK_DIR, ignore_errors=True)
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 118)
        print("05h2 frozen all-model safety summary: FAIL")
        print(f"{type(exc).__name__}: {exc}")
        print("=" * 118)
        raise
