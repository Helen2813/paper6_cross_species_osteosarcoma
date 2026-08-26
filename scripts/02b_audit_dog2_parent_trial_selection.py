from __future__ import annotations

from pathlib import Path
import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception:
    stats = None


SCRIPT_VERSION = "02b-audit-dog2-parent-trial-selection-v2-no-cli"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "_config"
CONTRACT_DIR = PROJECT_ROOT / "contracts"
MANIFEST_DIR = PROJECT_ROOT / "manifests"

LOCAL_PATH_CONFIG = CONFIG_DIR / "paths.local.json"

UPSTREAM_LOCK = CONTRACT_DIR / "00_upstream_input_lock.json"
UPSTREAM_MANIFEST = MANIFEST_DIR / "00_upstream_input_manifest.csv"

ID_LOCK = CONTRACT_DIR / "02a_dog2_authoritative_id_mapping_lock.json"
ID_MAPPING = MANIFEST_DIR / "02a_dog2_selected186_authoritative_id_mapping.csv"

OUT_CANDIDATES = MANIFEST_DIR / "02b_dog2_parent_trial_candidate_tables.csv"
OUT_PARENT_COLUMNS = MANIFEST_DIR / "02b_dog2_parent_trial_column_audit.csv"
OUT_ID_COVERAGE = MANIFEST_DIR / "02b_dog2_parent_selected_id_coverage.csv"
OUT_BALANCE = MANIFEST_DIR / "02b_dog2_selected_vs_not_selected_balance.csv"
OUT_ARM = MANIFEST_DIR / "02b_dog2_treatment_arm_selection_audit.csv"
OUT_STRATA = MANIFEST_DIR / "02b_dog2_randomization_strata_selection_audit.csv"
OUT_STATUS = CONTRACT_DIR / "02b_dog2_parent_trial_selection_status.json"
OUT_README = MANIFEST_DIR / "02b_dog2_parent_trial_selection_README.txt"

EXPECTED_PAPER4_BASENAME = "paper4_sarcoma_dog"
EXPECTED_SELECTED_N = 186

# Published design numbers are recorded as references only. The script never
# forces a local candidate table to have exactly one of these row counts.
PUBLISHED_RANDOMIZED_N = 324
PUBLISHED_ITT_N = 309

MAX_CANDIDATE_FILE_BYTES = 60 * 1024 * 1024
MIN_PARENT_UNIQUE_COTC = 250
MIN_SELECTED_LINKAGE_FRACTION = 0.95

COTC_PATTERN = re.compile(
    r"\b(COTC0?\d{2,3})[-_ ]+0*(\d{2,6})(?:[-_ ]+[A-Z])?\b",
    flags=re.IGNORECASE,
)

# Conservative semantic aliases. Automatic selection is only a convenience;
# every chosen column is written to an audit file for human review.
SEMANTIC_ALIASES: dict[str, tuple[str, ...]] = {
    "treatment": (
        "treatment",
        "treatment arm",
        "treatment_arm",
        "arm",
        "therapy",
        "sirolimus",
        "rapamycin",
        "carboplatin",
    ),
    "alp": (
        "alp",
        "alkaline phosphatase",
        "alkaline_phosphatase",
    ),
    "tumor_location": (
        "tumor location",
        "tumor_location",
        "tumor site",
        "tumor_site",
        "location",
        "proximal humerus",
        "proximal_humerus",
    ),
    "age": ("age",),
    "weight": (
        "weight",
        "body weight",
        "body_weight",
    ),
    "sex": ("sex", "gender"),
    "breed": ("breed",),
    "center": (
        "center",
        "centre",
        "institution",
        "hospital",
        "university",
        "enrolling site",
        "enrolling_site",
        "study site",
        "study_site",
    ),
    "study": (
        "study",
        "trial",
        "protocol",
        "cotc",
    ),
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "na", "n/a"}:
        return ""
    return text


def normalize_cotc(value: Any) -> str:
    text = clean_text(value).upper()
    if not text:
        return ""

    match = COTC_PATTERN.search(text)
    if match is None:
        return ""

    study_digits = re.sub(r"\D", "", match.group(1)[4:])
    subject_digits = re.sub(r"\D", "", match.group(2))

    if not study_digits or not subject_digits:
        return ""

    if len(study_digits) == 2:
        study_digits = "0" + study_digits

    return f"COTC{study_digits}-{subject_digits.zfill(4)}"


def load_local_path_config() -> dict[str, Any]:
    if not LOCAL_PATH_CONFIG.exists():
        return {}

    payload = json.loads(
        LOCAL_PATH_CONFIG.read_text(encoding="utf-8")
    )
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Local path config must be a JSON object: {LOCAL_PATH_CONFIG}"
        )
    return payload


def resolve_paper4_root() -> tuple[Path, str]:
    candidates: list[tuple[Path, str]] = []

    env_value = os.environ.get("PAPER4_ROOT", "").strip()
    if env_value:
        candidates.append(
            (Path(env_value).expanduser(), "environment:PAPER4_ROOT")
        )

    local_config = load_local_path_config()
    config_value = str(
        local_config.get("paper4_root", "")
    ).strip()
    if config_value:
        candidates.append(
            (
                Path(config_value).expanduser(),
                "_config/paths.local.json",
            )
        )

    candidates.extend(
        [
            (
                PROJECT_ROOT.parent / EXPECTED_PAPER4_BASENAME,
                "sibling_repository",
            ),
            (
                Path.home() / "Desktop" / EXPECTED_PAPER4_BASENAME,
                "home_desktop_fallback",
            ),
        ]
    )

    for candidate, source in candidates:
        resolved = candidate.resolve()
        sentinel = (
            resolved
            / "data"
            / "processed"
            / "GSE238110_DOG2_clinical_matched_indexed.csv"
        )
        if resolved.is_dir() and sentinel.exists():
            return resolved, source

    raise FileNotFoundError(
        "Could not resolve Paper 4 root. Set PAPER4_ROOT or create "
        "_config/paths.local.json."
    )


def verify_script00_lock(paper4_root: Path) -> dict[str, Any]:
    if not UPSTREAM_LOCK.exists() or not UPSTREAM_MANIFEST.exists():
        raise FileNotFoundError(
            "Run scripts/00_lock_upstream_inputs.py first."
        )

    lock = json.loads(
        UPSTREAM_LOCK.read_text(encoding="utf-8")
    )

    if sha256_file(UPSTREAM_MANIFEST) != lock.get("manifest_sha256"):
        raise RuntimeError(
            "Script-00 upstream manifest hash mismatch."
        )

    return lock


def verify_script02a_lock() -> tuple[pd.DataFrame, dict[str, Any]]:
    if not ID_LOCK.exists() or not ID_MAPPING.exists():
        raise FileNotFoundError(
            "Run the PASSING v3 version of "
            "scripts/02a_resolve_dog2_authoritative_ids.py first."
        )

    lock = json.loads(
        ID_LOCK.read_text(encoding="utf-8")
    )

    if lock.get("status") != "PASS":
        raise RuntimeError(
            "02a identifier lock is not PASS."
        )

    observed_mapping_hash = sha256_file(ID_MAPPING)
    if observed_mapping_hash != lock.get("output_mapping_sha256"):
        raise RuntimeError(
            "02a identifier mapping hash mismatch."
        )

    mapping = pd.read_csv(
        ID_MAPPING,
        dtype=str,
        low_memory=False,
    ).fillna("")

    if mapping.shape[0] != EXPECTED_SELECTED_N:
        raise RuntimeError(
            f"02a mapping rows={mapping.shape[0]}; "
            f"expected {EXPECTED_SELECTED_N}."
        )

    if mapping["cotc_subject_id"].nunique() != EXPECTED_SELECTED_N:
        raise RuntimeError(
            "02a mapping does not contain 186 unique COTC subjects."
        )

    return mapping, lock


def candidate_files(paper4_root: Path) -> list[Path]:
    allowed_suffixes = {
        ".csv",
        ".tsv",
        ".txt",
        ".xlsx",
        ".xls",
    }

    output: list[Path] = []

    for path in paper4_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in allowed_suffixes:
            continue

        try:
            size = path.stat().st_size
        except OSError:
            continue

        if size <= 0 or size > MAX_CANDIDATE_FILE_BYTES:
            continue

        text = path.as_posix().lower()

        metadata_hint = any(
            token in text
            for token in (
                "clinical",
                "metadata",
                "supp",
                "table",
                "cotc",
                "patient",
                "trial",
                "demograph",
                "outcome",
            )
        )

        obvious_omics = any(
            token in path.name.lower()
            for token in (
                "expression",
                "counts",
                "log2cpm",
                "tpm",
                "gene_weights",
                "gene_membership",
                "methyl",
                "cnv",
            )
        )

        if metadata_hint and not obvious_omics:
            output.append(path)

    return sorted(set(output))


def safe_read_csv(path: Path) -> list[tuple[str, pd.DataFrame]]:
    attempts = [
        {"sep": None, "engine": "python"},
        {"sep": ",", "engine": "c"},
        {"sep": "\t", "engine": "c"},
    ]

    for kwargs in attempts:
        try:
            frame = pd.read_csv(
                path,
                dtype=str,
                low_memory=False,
                **kwargs,
            ).fillna("")
        except Exception:
            continue

        if 0 < frame.shape[0] <= 5000 and 0 < frame.shape[1] <= 500:
            return [("__table__", frame)]

    return []


def safe_read_excel(path: Path) -> list[tuple[str, pd.DataFrame]]:
    try:
        workbook = pd.ExcelFile(path)
    except Exception:
        return []

    output: list[tuple[str, pd.DataFrame]] = []

    for sheet in workbook.sheet_names:
        try:
            frame = pd.read_excel(
                path,
                sheet_name=sheet,
                dtype=str,
            ).fillna("")
        except Exception:
            continue

        if 0 < frame.shape[0] <= 5000 and 0 < frame.shape[1] <= 500:
            output.append(
                (str(sheet), frame)
            )

    return output


def all_series(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Return every usable column plus a safe string representation of the index.

    Some metadata workbooks/loosely parsed delimited files can produce a
    pandas MultiIndex. ``MultiIndex.astype(str)`` is not supported in the
    pandas version used on the analysis machine, so index values are converted
    element-by-element instead. Tuple-valued MultiIndex rows are serialized
    conservatively without changing the scientific contents.
    """

    if isinstance(frame.index, pd.MultiIndex):
        index_values = [
            " | ".join(
                clean_text(part)
                for part in item
                if clean_text(part)
            )
            for item in frame.index.tolist()
        ]
    else:
        index_values = [
            clean_text(item)
            for item in frame.index.tolist()
        ]

    output: dict[str, pd.Series] = {
        "__index__": pd.Series(
            index_values,
            index=frame.index,
            dtype="object",
        )
    }

    for position, column in enumerate(frame.columns):
        values = frame.iloc[:, position]

        # Duplicate or unusual column labels should never make the discovery
        # scan crash. Give them deterministic names in the audit layer.
        label = str(column)
        if label in output:
            label = f"{label}__column_{position}"

        output[label] = values

    return output


def best_cotc_source(
    frame: pd.DataFrame,
) -> tuple[str | None, pd.Series, int, int]:
    best_source = None
    best_values = pd.Series(
        "",
        index=frame.index,
        dtype="object",
    )
    best_unique = 0
    best_nonempty = 0
    best_score = (-1, -1, -1)

    for source, values in all_series(frame).items():
        mapped = values.map(normalize_cotc)
        nonempty = int(mapped.ne("").sum())
        unique = int(
            mapped[mapped.ne("")].nunique()
        )

        source_lower = source.lower()
        semantic_bonus = int(
            any(
                token in source_lower
                for token in (
                    "cotc",
                    "subject",
                    "sample",
                    "patient",
                )
            )
        )

        score = (unique, nonempty, semantic_bonus)

        if score > best_score:
            best_score = score
            best_source = source
            best_values = mapped
            best_unique = unique
            best_nonempty = nonempty

    return (
        best_source,
        best_values,
        best_unique,
        best_nonempty,
    )


def scan_parent_candidates(
    paper4_root: Path,
    selected_subjects: set[str],
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    rows = []
    best = None

    for path in candidate_files(paper4_root):
        if path.suffix.lower() in {".xlsx", ".xls"}:
            tables = safe_read_excel(path)
        else:
            tables = safe_read_csv(path)

        for sheet_name, frame in tables:
            (
                cotc_source,
                cotc_values,
                n_unique_cotc,
                n_nonempty_cotc,
            ) = best_cotc_source(frame)

            cotc_set = set(
                cotc_values[cotc_values.ne("")]
            )

            selected_overlap = len(
                cotc_set & selected_subjects
            )
            selected_missing = len(
                selected_subjects - cotc_set
            )

            row = {
                "relative_path": path.resolve().relative_to(
                    paper4_root.resolve()
                ).as_posix(),
                "sheet": sheet_name,
                "rows": int(frame.shape[0]),
                "columns": int(frame.shape[1]),
                "file_size_bytes": int(path.stat().st_size),
                "best_cotc_source": cotc_source,
                "cotc_nonempty": n_nonempty_cotc,
                "unique_cotc": n_unique_cotc,
                "selected186_overlap": selected_overlap,
                "selected186_missing": selected_missing,
                "selected186_overlap_fraction": (
                    selected_overlap / EXPECTED_SELECTED_N
                ),
                "parent_candidate": (
                    n_unique_cotc >= MIN_PARENT_UNIQUE_COTC
                ),
            }
            rows.append(row)

            if n_unique_cotc < MIN_PARENT_UNIQUE_COTC:
                continue

            score = (
                selected_overlap,
                n_unique_cotc,
                int(frame.shape[1]),
            )

            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "path": path,
                    "sheet": sheet_name,
                    "frame": frame.copy(),
                    "cotc_source": cotc_source,
                    "cotc_values": cotc_values.copy(),
                    "summary": row,
                }

    candidates = pd.DataFrame(rows)

    if not candidates.empty:
        candidates = candidates.sort_values(
            [
                "parent_candidate",
                "selected186_overlap",
                "unique_cotc",
                "rows",
            ],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)

    return candidates, best


def normalize_column_name(name: Any) -> str:
    text = str(name).lower().strip()
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def semantic_score(
    column: Any,
    semantic: str,
) -> int:
    name = normalize_column_name(column)
    aliases = SEMANTIC_ALIASES[semantic]

    score = 0

    for alias in aliases:
        alias_norm = normalize_column_name(alias)

        if name == alias_norm:
            score = max(score, 100)

        elif alias_norm in name:
            score = max(
                score,
                50 + len(alias_norm),
            )

    return score


def choose_semantic_column(
    frame: pd.DataFrame,
    semantic: str,
) -> str | None:
    candidates = []

    for column in frame.columns:
        score = semantic_score(
            column,
            semantic,
        )
        if score <= 0:
            continue

        nonmissing = frame[column].map(clean_text).ne("")
        candidates.append(
            (
                score,
                int(nonmissing.sum()),
                str(column),
            )
        )

    if not candidates:
        return None

    candidates.sort(reverse=True)

    return candidates[0][2]


def column_audit(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for column in frame.columns:
        series = frame[column]
        cleaned = series.map(clean_text)
        nonempty = cleaned[cleaned.ne("")]

        semantic_hits = []

        for semantic in SEMANTIC_ALIASES:
            if semantic_score(column, semantic) > 0:
                semantic_hits.append(semantic)

        rows.append(
            {
                "column": str(column),
                "nonmissing_n": int(nonempty.shape[0]),
                "unique_n": int(nonempty.nunique()),
                "semantic_candidates": ";".join(
                    semantic_hits
                ),
                "example_values": " | ".join(
                    nonempty.drop_duplicates().head(8).tolist()
                ),
            }
        )

    return pd.DataFrame(rows)


def numeric_series(
    series: pd.Series,
) -> pd.Series:
    return pd.to_numeric(
        series.map(clean_text),
        errors="coerce",
    )


def standardized_mean_difference(
    included: pd.Series,
    not_selected: pd.Series,
) -> float:
    a = numeric_series(included).dropna()
    b = numeric_series(not_selected).dropna()

    if len(a) < 2 or len(b) < 2:
        return np.nan

    pooled_var = (
        a.var(ddof=1) + b.var(ddof=1)
    ) / 2.0

    if not np.isfinite(pooled_var) or pooled_var <= 0:
        if np.isclose(a.mean(), b.mean()):
            return 0.0
        return np.nan

    return float(
        (a.mean() - b.mean())
        / math.sqrt(pooled_var)
    )


def cramers_v(
    table: pd.DataFrame,
) -> tuple[float, float]:
    if table.shape[0] < 2 or table.shape[1] < 2:
        return np.nan, np.nan

    if stats is None:
        return np.nan, np.nan

    try:
        chi2, p_value, _, _ = stats.chi2_contingency(
            table.to_numpy(),
            correction=False,
        )
    except Exception:
        return np.nan, np.nan

    n = float(
        table.to_numpy().sum()
    )

    if n <= 0:
        return np.nan, np.nan

    phi2 = chi2 / n
    r, k = table.shape
    denom = min(r - 1, k - 1)

    if denom <= 0:
        return np.nan, float(p_value)

    value = math.sqrt(
        max(phi2, 0.0) / denom
    )

    return float(value), float(p_value)


def welch_p_value(
    included: pd.Series,
    not_selected: pd.Series,
) -> float:
    if stats is None:
        return np.nan

    a = numeric_series(included).dropna()
    b = numeric_series(not_selected).dropna()

    if len(a) < 2 or len(b) < 2:
        return np.nan

    try:
        _, p_value = stats.ttest_ind(
            a,
            b,
            equal_var=False,
            nan_policy="omit",
        )
    except Exception:
        return np.nan

    return float(p_value)


def selection_balance(
    parent: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for semantic in (
        "alp",
        "tumor_location",
        "age",
        "weight",
        "sex",
        "breed",
        "center",
    ):
        column = choose_semantic_column(
            parent,
            semantic,
        )

        if column is None:
            continue

        included = parent.loc[
            parent["__selected186__"],
            column,
        ]
        not_selected = parent.loc[
            ~parent["__selected186__"],
            column,
        ]

        combined_numeric = numeric_series(
            parent[column]
        )
        numeric_fraction = float(
            combined_numeric.notna().mean()
        )

        if (
            semantic in {"age", "weight"}
            and numeric_fraction >= 0.70
        ):
            effect_size = standardized_mean_difference(
                included,
                not_selected,
            )

            rows.append(
                {
                    "semantic": semantic,
                    "column": column,
                    "variable_type": "continuous",
                    "selected_nonmissing_n": int(
                        numeric_series(included).notna().sum()
                    ),
                    "not_selected_nonmissing_n": int(
                        numeric_series(not_selected).notna().sum()
                    ),
                    "selected_summary": (
                        "mean="
                        f"{numeric_series(included).mean():.6g}; "
                        "median="
                        f"{numeric_series(included).median():.6g}"
                    ),
                    "not_selected_summary": (
                        "mean="
                        f"{numeric_series(not_selected).mean():.6g}; "
                        "median="
                        f"{numeric_series(not_selected).median():.6g}"
                    ),
                    "effect_size_name": "standardized_mean_difference",
                    "effect_size": effect_size,
                    "nominal_p": welch_p_value(
                        included,
                        not_selected,
                    ),
                    "large_descriptive_imbalance_flag": (
                        bool(
                            np.isfinite(effect_size)
                            and abs(effect_size) >= 0.25
                        )
                    ),
                }
            )

        else:
            selected_clean = included.map(clean_text)
            selected_clean = selected_clean[
                selected_clean.ne("")
            ]

            not_selected_clean = not_selected.map(
                clean_text
            )
            not_selected_clean = not_selected_clean[
                not_selected_clean.ne("")
            ]

            labels = pd.DataFrame(
                {
                    "value": pd.concat(
                        [
                            selected_clean,
                            not_selected_clean,
                        ],
                        axis=0,
                    ),
                    "selection": (
                        ["selected186"]
                        * len(selected_clean)
                        + ["not_selected"]
                        * len(not_selected_clean)
                    ),
                }
            )

            ctab = pd.crosstab(
                labels["selection"],
                labels["value"],
            )

            effect_size, p_value = cramers_v(
                ctab
            )

            rows.append(
                {
                    "semantic": semantic,
                    "column": column,
                    "variable_type": "categorical",
                    "selected_nonmissing_n": int(
                        len(selected_clean)
                    ),
                    "not_selected_nonmissing_n": int(
                        len(not_selected_clean)
                    ),
                    "selected_summary": json.dumps(
                        selected_clean.value_counts().to_dict(),
                        sort_keys=True,
                    ),
                    "not_selected_summary": json.dumps(
                        not_selected_clean.value_counts().to_dict(),
                        sort_keys=True,
                    ),
                    "effect_size_name": "cramers_v",
                    "effect_size": effect_size,
                    "nominal_p": p_value,
                    "large_descriptive_imbalance_flag": (
                        bool(
                            np.isfinite(effect_size)
                            and effect_size >= 0.25
                        )
                    ),
                }
            )

    return pd.DataFrame(rows)


def treatment_arm_audit(
    parent: pd.DataFrame,
) -> pd.DataFrame:
    column = choose_semantic_column(
        parent,
        "treatment",
    )

    if column is None:
        return pd.DataFrame(
            columns=[
                "treatment_column",
                "arm",
                "parent_n",
                "selected186_n",
                "not_selected_n",
                "selection_fraction",
                "global_nominal_p",
            ]
        )

    treatment = parent[column].map(clean_text)
    working = parent.copy()
    working["__treatment_clean__"] = treatment
    working = working[
        working["__treatment_clean__"].ne("")
    ]

    rows = []

    for arm, group in working.groupby(
        "__treatment_clean__",
        sort=True,
    ):
        parent_n = int(group.shape[0])
        selected_n = int(
            group["__selected186__"].sum()
        )

        rows.append(
            {
                "treatment_column": column,
                "arm": arm,
                "parent_n": parent_n,
                "selected186_n": selected_n,
                "not_selected_n": parent_n - selected_n,
                "selection_fraction": (
                    selected_n / parent_n
                    if parent_n
                    else np.nan
                ),
                "global_nominal_p": np.nan,
            }
        )

    output = pd.DataFrame(rows)

    if output.shape[0] < 2:
        return output

    ctab = pd.crosstab(
        working["__treatment_clean__"],
        working["__selected186__"],
    )

    p_value = np.nan

    if stats is not None:
        try:
            if ctab.shape == (2, 2):
                _, p_value = stats.fisher_exact(
                    ctab.to_numpy()
                )
            else:
                _, p_value, _, _ = stats.chi2_contingency(
                    ctab.to_numpy(),
                    correction=False,
                )
            p_value = float(p_value)
        except Exception:
            p_value = np.nan

    output["global_nominal_p"] = p_value

    return output


def randomization_strata_audit(
    parent: pd.DataFrame,
) -> pd.DataFrame:
    alp_column = choose_semantic_column(
        parent,
        "alp",
    )
    location_column = choose_semantic_column(
        parent,
        "tumor_location",
    )

    if alp_column is None or location_column is None:
        return pd.DataFrame(
            columns=[
                "alp_column",
                "location_column",
                "alp_level",
                "location_level",
                "parent_n",
                "selected186_n",
                "selection_fraction",
            ]
        )

    working = parent[
        [
            alp_column,
            location_column,
            "__selected186__",
        ]
    ].copy()

    working["__alp__"] = working[
        alp_column
    ].map(clean_text)
    working["__location__"] = working[
        location_column
    ].map(clean_text)

    working = working[
        working["__alp__"].ne("")
        & working["__location__"].ne("")
    ]

    rows = []

    for (
        alp_level,
        location_level,
    ), group in working.groupby(
        ["__alp__", "__location__"],
        sort=True,
    ):
        parent_n = int(group.shape[0])
        selected_n = int(
            group["__selected186__"].sum()
        )

        rows.append(
            {
                "alp_column": alp_column,
                "location_column": location_column,
                "alp_level": alp_level,
                "location_level": location_level,
                "parent_n": parent_n,
                "selected186_n": selected_n,
                "selection_fraction": (
                    selected_n / parent_n
                    if parent_n
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    print("=" * 96)
    print("Paper 6 - DOG2 parent-trial -> RNA186 selection audit")
    print("=" * 96)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print("")
    print("Purpose:")
    print("  Use the frozen Script-02a 186-subject COTC mapping.")
    print("  Search Paper 4 for a patient-level COTC021/022 parent-trial table.")
    print("  Quantify RNA-subset selection by treatment arm and measured baseline")
    print("  covariates if such a parent table is available.")
    print("")
    print("This script performs no molecular model fitting and no treatment-effect")
    print("or survival association testing.")
    print("")

    MANIFEST_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    CONTRACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    paper4_root, resolution_source = resolve_paper4_root()
    verify_script00_lock(
        paper4_root
    )
    id_mapping, id_lock = verify_script02a_lock()

    selected_subjects = set(
        id_mapping["cotc_subject_id"].map(
            normalize_cotc
        )
    )

    if "" in selected_subjects:
        selected_subjects.remove("")

    if len(selected_subjects) != EXPECTED_SELECTED_N:
        raise RuntimeError(
            f"Frozen selected-subject set has {len(selected_subjects)} "
            f"COTC IDs; expected {EXPECTED_SELECTED_N}."
        )

    print(f"Paper 4 root: {paper4_root}")
    print("00 upstream lock: PASS")
    print("02a authoritative selected-subject lock: PASS")
    print(
        f"Frozen selected COTC subjects: "
        f"{len(selected_subjects)}"
    )
    print("")
    print(
        "Scanning Paper 4 for lightweight patient-level parent-trial tables..."
    )

    candidates, best = scan_parent_candidates(
        paper4_root=paper4_root,
        selected_subjects=selected_subjects,
    )

    candidates.to_csv(
        OUT_CANDIDATES,
        index=False,
    )

    parent_columns = pd.DataFrame()
    id_coverage = pd.DataFrame()
    balance = pd.DataFrame()
    arm = pd.DataFrame()
    strata = pd.DataFrame()

    if best is None:
        gate = "BLOCKED_NO_PARENT_TRIAL_PATIENT_TABLE"
        reason = (
            "No local lightweight table with at least "
            f"{MIN_PARENT_UNIQUE_COTC} unique COTC subjects was found. "
            "The 324/309 parent-trial -> RNA186 selection mechanism cannot "
            "yet be audited at patient level."
        )
        best_summary = None

    else:
        parent = best["frame"].copy()
        parent["__cotc_subject_id__"] = best[
            "cotc_values"
        ].values

        parent = parent[
            parent["__cotc_subject_id__"].ne("")
        ].copy()

        # One row per subject is required for a patient-level selection audit.
        duplicate_subjects = parent[
            "__cotc_subject_id__"
        ].duplicated(keep=False)

        if duplicate_subjects.any():
            gate = "BLOCKED_PARENT_TABLE_NOT_PATIENT_UNIQUE"
            reason = (
                "The strongest parent-level candidate contains duplicated COTC "
                "subjects. Subject-level deduplication rules are not prespecified, "
                "so the audit is stopped."
            )
            best_summary = best["summary"]

        else:
            parent_subjects = set(
                parent["__cotc_subject_id__"]
            )

            overlap = (
                selected_subjects
                & parent_subjects
            )
            missing_selected = (
                selected_subjects
                - parent_subjects
            )
            parent_not_selected = (
                parent_subjects
                - selected_subjects
            )

            parent["__selected186__"] = parent[
                "__cotc_subject_id__"
            ].isin(selected_subjects)

            linkage_fraction = (
                len(overlap)
                / EXPECTED_SELECTED_N
            )

            id_coverage = pd.DataFrame(
                [
                    {
                        "metric": "selected186_subjects",
                        "value": EXPECTED_SELECTED_N,
                    },
                    {
                        "metric": "parent_unique_subjects",
                        "value": len(parent_subjects),
                    },
                    {
                        "metric": "selected186_found_in_parent",
                        "value": len(overlap),
                    },
                    {
                        "metric": "selected186_missing_from_parent",
                        "value": len(missing_selected),
                    },
                    {
                        "metric": "parent_subjects_not_in_selected186",
                        "value": len(parent_not_selected),
                    },
                ]
            )

            if linkage_fraction < MIN_SELECTED_LINKAGE_FRACTION:
                gate = "BLOCKED_SELECTED_LINKAGE_BELOW_95_PERCENT"
                reason = (
                    f"Only {len(overlap)}/{EXPECTED_SELECTED_N} frozen RNA186 "
                    "subjects were found in the strongest parent candidate."
                )
                best_summary = best["summary"]

            else:
                parent_columns = column_audit(
                    parent.drop(
                        columns=[
                            "__cotc_subject_id__",
                            "__selected186__",
                        ],
                        errors="ignore",
                    )
                )

                balance = selection_balance(
                    parent
                )

                arm = treatment_arm_audit(
                    parent
                )

                strata = randomization_strata_audit(
                    parent
                )

                large_balance_flags = (
                    int(
                        balance[
                            "large_descriptive_imbalance_flag"
                        ].sum()
                    )
                    if (
                        not balance.empty
                        and "large_descriptive_imbalance_flag"
                        in balance.columns
                    )
                    else 0
                )

                arm_fraction_range = np.nan

                if (
                    not arm.empty
                    and "selection_fraction"
                    in arm.columns
                ):
                    fractions = pd.to_numeric(
                        arm["selection_fraction"],
                        errors="coerce",
                    ).dropna()

                    if len(fractions) >= 2:
                        arm_fraction_range = float(
                            fractions.max()
                            - fractions.min()
                        )

                if (
                    large_balance_flags > 0
                    or (
                        np.isfinite(
                            arm_fraction_range
                        )
                        and arm_fraction_range >= 0.10
                    )
                ):
                    gate = "AUDITABLE_MEASURED_SELECTION_IMBALANCE_FLAGGED"
                    reason = (
                        "A suitable parent-level table was found and linkage is "
                        "adequate, but one or more prespecified descriptive "
                        "selection-imbalance flags were triggered. Randomized-arm "
                        "causal interpretation for RNA186 requires explicit "
                        "selection-bias handling/sensitivity analysis."
                    )
                else:
                    gate = "AUDITABLE_NO_MAJOR_MEASURED_SELECTION_IMBALANCE"
                    reason = (
                        "A suitable patient-level parent table was found, at "
                        "least 95% of frozen RNA186 subjects linked to it, and "
                        "no large measured selection imbalance was detected by "
                        "the prespecified descriptive thresholds. This supports "
                        "further randomization-arm auditing but does not prove "
                        "exchangeability after RNA-subset selection."
                    )

                best_summary = best["summary"]

    parent_columns.to_csv(
        OUT_PARENT_COLUMNS,
        index=False,
    )
    id_coverage.to_csv(
        OUT_ID_COVERAGE,
        index=False,
    )
    balance.to_csv(
        OUT_BALANCE,
        index=False,
    )
    arm.to_csv(
        OUT_ARM,
        index=False,
    )
    strata.to_csv(
        OUT_STRATA,
        index=False,
    )

    status_payload = {
        "script_version": SCRIPT_VERSION,
        "created_utc": utc_now_iso(),
        "status": gate,
        "status_reason": reason,
        "paper4_root_resolution_source": resolution_source,
        "script00_lock_verified": True,
        "script02a_lock_verified": True,
        "selected_subject_mapping_sha256": sha256_file(
            ID_MAPPING
        ),
        "selected_subject_n": len(
            selected_subjects
        ),
        "published_reference_counts": {
            "randomized_n": PUBLISHED_RANDOMIZED_N,
            "itt_n": PUBLISHED_ITT_N,
            "rna_subset_n": EXPECTED_SELECTED_N,
        },
        "best_parent_candidate": best_summary,
        "causal_randomization_anchor_permission": (
            "NOT_GRANTED"
            if gate
            != "AUDITABLE_NO_MAJOR_MEASURED_SELECTION_IMBALANCE"
            else (
                "SELECTION_AUDIT_SUPPORTS_NEXT_RANDOMIZATION_PROTOCOL_AUDIT_"
                "BUT_DOES_NOT_BY_ITSELF_AUTHORIZE_CAUSAL_CLAIMS"
            )
        ),
        "guardrails": [
            (
                "The original randomized trial does not automatically confer "
                "exchangeability on a molecular subset."
            ),
            (
                "A balanced treatment count in RNA186 is insufficient evidence "
                "that RNA availability/QC selection was treatment-independent."
            ),
            (
                "Measured baseline balance cannot rule out selection on "
                "unmeasured variables."
            ),
            (
                "Exact blocked-randomization inference must wait until the "
                "original randomization strata and their availability in RNA186 "
                "are explicitly verified."
            ),
            (
                "No treatment-effect heterogeneity or outcome association is "
                "tested in this script."
            ),
        ],
        "outputs": {
            "candidate_tables": OUT_CANDIDATES.name,
            "parent_columns": OUT_PARENT_COLUMNS.name,
            "id_coverage": OUT_ID_COVERAGE.name,
            "selection_balance": OUT_BALANCE.name,
            "treatment_arm_selection": OUT_ARM.name,
            "randomization_strata_selection": OUT_STRATA.name,
        },
    }

    OUT_STATUS.write_text(
        json.dumps(
            status_payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    OUT_README.write_text(
        f"""Paper 6 DOG2 parent-trial -> RNA186 selection audit
Script version: {SCRIPT_VERSION}

Status
------
{gate}

Reason
------
{reason}

Question
--------
Can the 186-dog pretreatment RNA subset be treated as preserving enough of the
original randomized parent-trial structure to justify later treatment-arm
invariance analyses?

Interpretation
--------------
This script is deliberately conservative. It requires a patient-level parent
table with explicit COTC subject identifiers and strong linkage to the frozen
186-subject mapping from Script 02a.

Even a PASS-like descriptive result does not prove that RNA availability,
pathology review, tissue quality, or record completeness were independent of
all potential outcomes or modifiers. It only supports proceeding to a separate
randomization-protocol audit.

Published parent counts ({PUBLISHED_RANDOMIZED_N} randomized; {PUBLISHED_ITT_N}
ITT) are reference values only. The script reports the locally observed
patient-level table rather than forcing these counts.

No molecular model, survival association, treatment effect, or treatment-effect
heterogeneity analysis is performed here.
""",
        encoding="utf-8",
    )

    print("")
    print("=" * 96)
    print("DOG2 parent-trial selection audit complete")
    print("=" * 96)
    print(f"Gate: {gate}")
    print(f"Reason: {reason}")

    if best is not None:
        print("")
        print("Best parent candidate:")
        print(
            "  file: "
            f"{best['summary']['relative_path']}"
        )
        print(
            "  sheet: "
            f"{best['summary']['sheet']}"
        )
        print(
            "  unique COTC subjects: "
            f"{best['summary']['unique_cotc']}"
        )
        print(
            "  selected186 overlap: "
            f"{best['summary']['selected186_overlap']}/"
            f"{EXPECTED_SELECTED_N}"
        )

    print("")
    print("Saved:")
    for path in (
        OUT_CANDIDATES,
        OUT_PARENT_COLUMNS,
        OUT_ID_COVERAGE,
        OUT_BALANCE,
        OUT_ARM,
        OUT_STRATA,
        OUT_STATUS,
        OUT_README,
    ):
        print(f"  {path}")

    print("")
    if gate == "BLOCKED_NO_PARENT_TRIAL_PATIENT_TABLE":
        print("Next:")
        print(
            "  Obtain only a small patient-level parent-trial clinical/metadata "
            "table with COTC021/022 subject IDs. No omics data are required."
        )
        print(
            "  Then place the file anywhere under Paper 4 and re-run Script 02b."
        )
    elif gate == "AUDITABLE_NO_MAJOR_MEASURED_SELECTION_IMBALANCE":
        print("Next:")
        print(
            "  Freeze and verify the original randomization protocol/strata, "
            "then design arm-to-arm Gate Zero."
        )
    else:
        print("Next:")
        print(
            "  Review the generated selection-balance and treatment-arm files "
            "before treating randomized arms as a causal anchor."
        )

    print("Done.")


if __name__ == "__main__":
    main()
