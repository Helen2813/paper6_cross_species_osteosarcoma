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
from scipy import stats


SCRIPT_VERSION = "02-audit-dog2-transcriptomic-selection-v1-no-cli"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "_config"
CONTRACT_DIR = PROJECT_ROOT / "contracts"
MANIFEST_DIR = PROJECT_ROOT / "manifests"

LOCAL_PATH_CONFIG = CONFIG_DIR / "paths.local.json"

UPSTREAM_LOCK = CONTRACT_DIR / "00_upstream_input_lock.json"
UPSTREAM_MANIFEST = MANIFEST_DIR / "00_upstream_input_manifest.csv"

OUT_CANDIDATES = MANIFEST_DIR / "02_dog2_parent_cohort_candidate_tables.csv"
OUT_SELECTED_COLUMNS = MANIFEST_DIR / "02_dog2_selected186_column_audit.csv"
OUT_PARENT_COLUMNS = MANIFEST_DIR / "02_dog2_parent_cohort_column_audit.csv"
OUT_BALANCE = MANIFEST_DIR / "02_dog2_selection_balance_audit.csv"
OUT_ARM_AUDIT = MANIFEST_DIR / "02_dog2_treatment_arm_selection_audit.csv"
OUT_ID_AUDIT = MANIFEST_DIR / "02_dog2_parent_selected_id_audit.csv"
OUT_CONTRACT = CONTRACT_DIR / "02_dog2_transcriptomic_selection_status.json"
OUT_README = MANIFEST_DIR / "02_dog2_transcriptomic_selection_README.txt"

EXPECTED_PAPER4_BASENAME = "paper4_sarcoma_dog"

EXPECTED_SELECTED_N = 186
PARENT_RANDOMIZED_N = 324
PARENT_ITT_N = 309

MAX_CANDIDATE_FILE_BYTES = 50 * 1024 * 1024
MIN_PARENT_COTC_IDS = 250

COTC_PATTERN = re.compile(
    r"\b(COTC0?(?:21|22))[-_ ]+0*([0-9]{2,6})(?:[-_ ]+[A-Z])?\b",
    flags=re.IGNORECASE,
)

COLUMN_PATTERNS: dict[str, tuple[str, ...]] = {
    "treatment": (
        "treatment", "arm", "therapy", "sirolimus", "rapamycin",
        "soc", "carboplatin",
    ),
    "alp": (
        "alp", "alkaline phosphatase", "alkaline_phosphatase",
    ),
    "tumor_location": (
        "location", "tumor site", "tumor_site", "proximal humerus",
        "proximal_humerus", "ph", "nph",
    ),
    "age": ("age",),
    "weight": ("weight", "body weight", "body_weight"),
    "sex": ("sex", "gender"),
    "breed": ("breed",),
    "center": (
        "center", "centre", "institution", "site", "hospital",
        "university", "enrolling",
    ),
    "dfi": (
        "dfi", "disease free", "disease_free", "metastasis free",
        "metastasis_free",
    ),
    "os": (
        "overall survival", "overall_survival", "os_time", "os event",
        "os_event",
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


def load_local_path_config() -> dict[str, Any]:
    if not LOCAL_PATH_CONFIG.exists():
        return {}
    payload = json.loads(LOCAL_PATH_CONFIG.read_text(encoding="utf-8"))
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
    config_value = str(local_config.get("paper4_root", "")).strip()
    if config_value:
        candidates.append(
            (Path(config_value).expanduser(), "_config/paths.local.json")
        )

    candidates.extend(
        [
            (PROJECT_ROOT.parent / EXPECTED_PAPER4_BASENAME, "sibling_repository"),
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


def verify_upstream_lock(paper4_root: Path) -> dict[str, Any]:
    if not UPSTREAM_LOCK.exists() or not UPSTREAM_MANIFEST.exists():
        raise FileNotFoundError(
            "Run scripts/00_lock_upstream_inputs.py first."
        )

    lock = json.loads(UPSTREAM_LOCK.read_text(encoding="utf-8"))
    if sha256_file(UPSTREAM_MANIFEST) != lock.get("manifest_sha256"):
        raise RuntimeError("Script-00 upstream manifest hash mismatch.")

    metadata = lock["assets"]["dog2_clinical"]
    clinical_path = paper4_root / metadata["relative_path"]
    if not clinical_path.exists():
        raise FileNotFoundError(f"Missing locked DOG2 clinical file: {clinical_path}")
    if sha256_file(clinical_path) != metadata["sha256"]:
        raise RuntimeError(
            "DOG2 clinical input changed since Script 00. Stop and investigate."
        )
    return lock


def normalize_cotc_id(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "na"}:
        return ""

    match = COTC_PATTERN.search(text)
    if match is None:
        return ""

    study = match.group(1).upper()
    if study == "COTC21":
        study = "COTC021"
    elif study == "COTC22":
        study = "COTC022"

    subject_digits = match.group(2).lstrip("0")
    if not subject_digits:
        subject_digits = "0"

    return f"{study}-{subject_digits.zfill(4)}"


def extract_row_cotc_ids(table: pd.DataFrame) -> pd.Series:
    index_ids = pd.Series(
        [normalize_cotc_id(x) for x in table.index],
        index=table.index,
        dtype="object",
    )
    if int(index_ids.ne("").sum()) >= max(10, int(0.5 * table.shape[0])):
        return index_ids

    ordered_columns = sorted(
        table.columns,
        key=lambda c: (
            0 if any(
                token in str(c).lower()
                for token in ("cotc", "patient", "subject", "case", "sample", "dog")
            ) else 1,
            str(c),
        ),
    )

    best = pd.Series("", index=table.index, dtype="object")
    best_count = 0

    for column in ordered_columns:
        mapped = table[column].map(normalize_cotc_id)
        count = int(mapped.ne("").sum())
        if count > best_count:
            best = mapped
            best_count = count

    if best_count < max(10, int(0.5 * table.shape[0])):
        row_ids: list[str] = []
        for _, row in table.iterrows():
            found = ""
            for value in row.tolist():
                found = normalize_cotc_id(value)
                if found:
                    break
            row_ids.append(found)
        candidate = pd.Series(row_ids, index=table.index, dtype="object")
        if int(candidate.ne("").sum()) > best_count:
            best = candidate

    return best


def safe_read_csv(path: Path) -> list[tuple[str, pd.DataFrame]]:
    attempts = [
        {"sep": None, "engine": "python"},
        {"sep": ",", "engine": "c"},
        {"sep": "\t", "engine": "c"},
    ]
    for kwargs in attempts:
        try:
            table = pd.read_csv(path, low_memory=False, **kwargs)
            if table.shape[0] > 0 and table.shape[1] > 0:
                return [("__table__", table)]
        except Exception:
            pass
    return []


def safe_read_excel(path: Path) -> list[tuple[str, pd.DataFrame]]:
    try:
        workbook = pd.ExcelFile(path)
    except Exception:
        return []

    outputs: list[tuple[str, pd.DataFrame]] = []
    for sheet in workbook.sheet_names:
        try:
            table = pd.read_excel(path, sheet_name=sheet)
        except Exception:
            continue
        if table.shape[0] > 0 and table.shape[1] > 0:
            outputs.append((str(sheet), table))
    return outputs


def candidate_files(paper4_root: Path) -> list[Path]:
    extensions = {".csv", ".tsv", ".txt", ".xlsx", ".xls"}
    candidates: list[Path] = []

    for path in paper4_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in extensions:
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size <= 0 or size > MAX_CANDIDATE_FILE_BYTES:
            continue

        lower = path.name.lower()
        full_lower = path.as_posix().lower()

        metadata_hint = any(
            token in full_lower
            for token in (
                "clinical",
                "metadata",
                "supp",
                "table",
                "cotc",
                "dog2",
                "patient",
                "demograph",
                "outcome",
                "manifest",
            )
        )
        expression_exclusion = any(
            token in lower
            for token in (
                "expression",
                "counts",
                "tpm",
                "log2cpm",
                "gene_weights",
                "gene_membership",
            )
        )
        if metadata_hint and not expression_exclusion:
            candidates.append(path)

    return sorted(set(candidates))


def scan_parent_candidates(
    paper4_root: Path,
    selected_ids: set[str],
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    selected_clinical_path = (
        paper4_root
        / "data"
        / "processed"
        / "GSE238110_DOG2_clinical_matched_indexed.csv"
    ).resolve()

    for path in candidate_files(paper4_root):
        if path.resolve() == selected_clinical_path:
            continue

        if path.suffix.lower() in {".xlsx", ".xls"}:
            tables = safe_read_excel(path)
        else:
            tables = safe_read_csv(path)

        for sheet_name, table in tables:
            if table.shape[0] > 5000 or table.shape[1] > 500:
                continue

            ids = extract_row_cotc_ids(table)
            unique_ids = {x for x in ids.tolist() if x}
            cotc021_022_ids = {
                x for x in unique_ids
                if x.startswith("COTC021-") or x.startswith("COTC022-")
            }
            overlap = selected_ids & cotc021_022_ids

            record = {
                "relative_path": path.resolve().relative_to(
                    paper4_root.resolve()
                ).as_posix(),
                "sheet": sheet_name,
                "file_size_bytes": int(path.stat().st_size),
                "rows": int(table.shape[0]),
                "columns": int(table.shape[1]),
                "unique_cotc021_022_ids": len(cotc021_022_ids),
                "selected186_id_overlap": len(overlap),
                "selected186_overlap_fraction": (
                    len(overlap) / len(selected_ids) if selected_ids else np.nan
                ),
                "candidate_parent_status": (
                    "POTENTIAL_PARENT_LEVEL"
                    if len(cotc021_022_ids) >= MIN_PARENT_COTC_IDS
                    else "SUBSET_OR_INSUFFICIENT"
                ),
            }
            rows.append(record)

            score = (
                len(cotc021_022_ids),
                len(overlap),
                table.shape[1],
            )
            if (
                len(cotc021_022_ids) >= MIN_PARENT_COTC_IDS
                and (best is None or score > best["score"])
            ):
                best = {
                    "score": score,
                    "path": path,
                    "sheet": sheet_name,
                    "table": table.copy(),
                    "ids": ids.copy(),
                    "record": record,
                }

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(
            ["unique_cotc021_022_ids", "selected186_id_overlap", "rows"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    return frame, best


def column_audit(table: pd.DataFrame, source_label: str) -> pd.DataFrame:
    rows = []
    for column in table.columns:
        series = table[column]
        nonmissing = series.dropna()
        unique_n = int(nonmissing.nunique(dropna=True))
        numeric = pd.to_numeric(nonmissing, errors="coerce")
        numeric_fraction = float(numeric.notna().mean()) if len(nonmissing) else 0.0

        semantic_hits = []
        normalized_name = re.sub(
            r"[_\-]+", " ", str(column).lower()
        )
        for semantic, patterns in COLUMN_PATTERNS.items():
            if any(pattern in normalized_name for pattern in patterns):
                semantic_hits.append(semantic)

        rows.append(
            {
                "source": source_label,
                "column": str(column),
                "nonmissing_n": int(nonmissing.shape[0]),
                "missing_n": int(table.shape[0] - nonmissing.shape[0]),
                "unique_n": unique_n,
                "dtype": str(series.dtype),
                "numeric_fraction": numeric_fraction,
                "semantic_candidates": ";".join(semantic_hits),
                "examples": " | ".join(
                    nonmissing.astype(str).drop_duplicates().head(5).tolist()
                ),
            }
        )
    return pd.DataFrame(rows)


def choose_semantic_column(
    table: pd.DataFrame,
    semantic: str,
) -> str | None:
    patterns = COLUMN_PATTERNS[semantic]
    scores: list[tuple[int, float, str]] = []

    for column in table.columns:
        name = re.sub(r"[_\-]+", " ", str(column).lower())
        hit_count = sum(pattern in name for pattern in patterns)
        if hit_count == 0:
            continue
        nonmissing_fraction = float(table[column].notna().mean())
        scores.append((hit_count, nonmissing_fraction, str(column)))

    if not scores:
        return None
    scores.sort(reverse=True)
    return scores[0][2]


def clean_category(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "na", "n/a"}:
        return ""
    return text


def standardized_mean_difference(
    included: pd.Series,
    excluded: pd.Series,
) -> float:
    a = pd.to_numeric(included, errors="coerce").dropna().astype(float)
    b = pd.to_numeric(excluded, errors="coerce").dropna().astype(float)

    if len(a) < 2 or len(b) < 2:
        return np.nan

    pooled_var = (a.var(ddof=1) + b.var(ddof=1)) / 2.0
    if not np.isfinite(pooled_var) or pooled_var <= 0:
        return 0.0 if np.isclose(a.mean(), b.mean()) else np.nan
    return float((a.mean() - b.mean()) / math.sqrt(pooled_var))


def cramers_v(table: pd.DataFrame) -> tuple[float, float]:
    if table.shape[0] < 2 or table.shape[1] < 2:
        return np.nan, np.nan
    try:
        chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
    except Exception:
        return np.nan, np.nan
    n = float(table.to_numpy().sum())
    if n <= 0:
        return np.nan, np.nan
    phi2 = chi2 / n
    r, k = table.shape
    denom = min(k - 1, r - 1)
    if denom <= 0:
        return np.nan, float(p)
    return float(math.sqrt(max(phi2, 0.0) / denom)), float(p)


def compare_selection(
    parent: pd.DataFrame,
    parent_ids: pd.Series,
    selected_ids: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    working = parent.copy()
    working["__cotc_id__"] = parent_ids.values
    working = working[working["__cotc_id__"].ne("")].copy()
    working = working.drop_duplicates("__cotc_id__", keep="first")
    working["__included_rna186__"] = working["__cotc_id__"].isin(selected_ids)

    id_rows = [
        {
            "metric": "parent_unique_cotc021_022",
            "value": int(
                working["__cotc_id__"].str.startswith(
                    ("COTC021-", "COTC022-")
                ).sum()
            ),
        },
        {
            "metric": "selected186_ids",
            "value": len(selected_ids),
        },
        {
            "metric": "selected_ids_found_in_parent",
            "value": int(working["__included_rna186__"].sum()),
        },
        {
            "metric": "selected_ids_missing_from_parent",
            "value": int(len(selected_ids) - working["__included_rna186__"].sum()),
        },
        {
            "metric": "parent_ids_not_in_selected186",
            "value": int((~working["__included_rna186__"]).sum()),
        },
    ]
    id_audit = pd.DataFrame(id_rows)

    balance_rows: list[dict[str, Any]] = []

    for semantic in [
        "alp",
        "tumor_location",
        "age",
        "weight",
        "sex",
        "breed",
        "center",
    ]:
        column = choose_semantic_column(working, semantic)
        if column is None:
            continue

        included = working.loc[working["__included_rna186__"], column]
        excluded = working.loc[~working["__included_rna186__"], column]

        numeric = pd.to_numeric(working[column], errors="coerce")
        numeric_fraction = float(numeric.notna().mean())

        if semantic in {"age", "weight"} and numeric_fraction >= 0.70:
            smd = standardized_mean_difference(included, excluded)
            try:
                _, p_value = stats.ttest_ind(
                    pd.to_numeric(included, errors="coerce").dropna(),
                    pd.to_numeric(excluded, errors="coerce").dropna(),
                    equal_var=False,
                    nan_policy="omit",
                )
                p_value = float(p_value)
            except Exception:
                p_value = np.nan

            balance_rows.append(
                {
                    "semantic": semantic,
                    "column": column,
                    "variable_type": "continuous",
                    "included_nonmissing_n": int(included.notna().sum()),
                    "excluded_nonmissing_n": int(excluded.notna().sum()),
                    "included_summary": (
                        f"mean={pd.to_numeric(included, errors='coerce').mean():.6g};"
                        f"median={pd.to_numeric(included, errors='coerce').median():.6g}"
                    ),
                    "excluded_summary": (
                        f"mean={pd.to_numeric(excluded, errors='coerce').mean():.6g};"
                        f"median={pd.to_numeric(excluded, errors='coerce').median():.6g}"
                    ),
                    "effect_size": smd,
                    "effect_size_name": "standardized_mean_difference",
                    "nominal_p": p_value,
                }
            )
        else:
            inc_cat = included.map(clean_category)
            exc_cat = excluded.map(clean_category)
            temp = pd.DataFrame(
                {
                    "value": pd.concat([inc_cat, exc_cat], axis=0),
                    "included": (
                        ["included"] * len(inc_cat)
                        + ["excluded"] * len(exc_cat)
                    ),
                }
            )
            temp = temp[temp["value"].ne("")]
            ctab = pd.crosstab(temp["included"], temp["value"])
            cv, p_value = cramers_v(ctab)

            balance_rows.append(
                {
                    "semantic": semantic,
                    "column": column,
                    "variable_type": "categorical",
                    "included_nonmissing_n": int(inc_cat.ne("").sum()),
                    "excluded_nonmissing_n": int(exc_cat.ne("").sum()),
                    "included_summary": json.dumps(
                        inc_cat[inc_cat.ne("")].value_counts().to_dict(),
                        sort_keys=True,
                    ),
                    "excluded_summary": json.dumps(
                        exc_cat[exc_cat.ne("")].value_counts().to_dict(),
                        sort_keys=True,
                    ),
                    "effect_size": cv,
                    "effect_size_name": "cramers_v",
                    "nominal_p": p_value,
                }
            )

    balance = pd.DataFrame(balance_rows)

    treatment_col = choose_semantic_column(working, "treatment")
    arm_rows: list[dict[str, Any]] = []

    if treatment_col is not None:
        treatment = working[treatment_col].map(clean_category)
        arm_table = pd.crosstab(
            treatment,
            working["__included_rna186__"],
        )
        for arm, group in working.groupby(treatment, dropna=False):
            arm_text = clean_category(arm)
            if not arm_text:
                continue
            n_total = int(group.shape[0])
            n_included = int(group["__included_rna186__"].sum())
            arm_rows.append(
                {
                    "treatment_column": treatment_col,
                    "arm": arm_text,
                    "parent_n": n_total,
                    "selected186_n": n_included,
                    "not_selected_n": n_total - n_included,
                    "selection_fraction": (
                        n_included / n_total if n_total else np.nan
                    ),
                    "fisher_or_chi2_p_for_arm_by_selection": np.nan,
                }
            )

        if arm_table.shape == (2, 2):
            try:
                _, p = stats.fisher_exact(arm_table.to_numpy())
                p = float(p)
            except Exception:
                p = np.nan
        elif arm_table.shape[0] >= 2 and arm_table.shape[1] == 2:
            try:
                _, p, _, _ = stats.chi2_contingency(
                    arm_table.to_numpy(), correction=False
                )
                p = float(p)
            except Exception:
                p = np.nan
        else:
            p = np.nan

        for row in arm_rows:
            row["fisher_or_chi2_p_for_arm_by_selection"] = p

    return id_audit, balance, pd.DataFrame(arm_rows)


def selected_186_summary(clinical: pd.DataFrame) -> dict[str, Any]:
    treatment_col = choose_semantic_column(clinical, "treatment")
    result = {
        "n_selected": int(clinical.shape[0]),
        "treatment_column": treatment_col,
        "treatment_counts": None,
    }
    if treatment_col is not None:
        counts = (
            clinical[treatment_col]
            .map(clean_category)
            .loc[lambda s: s.ne("")]
            .value_counts()
            .to_dict()
        )
        result["treatment_counts"] = counts
    return result


def main() -> None:
    print("=" * 96)
    print("Paper 6 - audit DOG2 324 -> 186 transcriptomic selection")
    print("=" * 96)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print("")
    print("Purpose:")
    print("  Test whether patient-level parent-cohort metadata are available locally.")
    print("  Compare the 186 RNA-profiled dogs with non-profiled parent-trial dogs.")
    print("  Audit treatment-arm selection and baseline covariate balance if possible.")
    print("  Make NO causal claim if parent-level patient metadata are unavailable.")
    print("")

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    CONTRACT_DIR.mkdir(parents=True, exist_ok=True)

    paper4_root, resolution_source = resolve_paper4_root()
    lock = verify_upstream_lock(paper4_root)

    clinical_meta = lock["assets"]["dog2_clinical"]
    clinical_path = paper4_root / clinical_meta["relative_path"]
    clinical = pd.read_csv(clinical_path, index_col=0, low_memory=False)

    selected_ids_series = extract_row_cotc_ids(clinical)
    selected_ids = {
        x for x in selected_ids_series.tolist()
        if x.startswith("COTC021-") or x.startswith("COTC022-")
    }

    print(f"Paper 4 root: {paper4_root}")
    print("00 upstream lock: PASS")
    print(f"Selected transcriptomic clinical rows: {clinical.shape[0]}")
    print(f"Selected transcriptomic COTC021/022 IDs detected: {len(selected_ids)}")

    selected_column_frame = column_audit(
        clinical.reset_index(drop=False),
        source_label="DOG2_selected186",
    )
    selected_column_frame.to_csv(OUT_SELECTED_COLUMNS, index=False)

    candidate_frame = pd.DataFrame()
    parent_column_frame = pd.DataFrame()
    balance = pd.DataFrame()
    arm_audit = pd.DataFrame()
    id_audit = pd.DataFrame()
    best = None

    if len(selected_ids) < 170:
        gate = "BLOCKED_SELECTED_ID_MAPPING_INCOMPLETE"
        reason = (
            "Fewer than 170 of 186 transcriptomic dogs could be resolved to "
            "COTC021/022 subject IDs from the locked clinical table."
        )
    else:
        print("")
        print("Scanning Paper 4 for lightweight patient-level parent-cohort tables...")
        candidate_frame, best = scan_parent_candidates(
            paper4_root=paper4_root,
            selected_ids=selected_ids,
        )
        candidate_frame.to_csv(OUT_CANDIDATES, index=False)

        if best is None:
            gate = "BLOCKED_NO_PATIENT_LEVEL_PARENT_COHORT"
            reason = (
                "No local metadata table with at least "
                f"{MIN_PARENT_COTC_IDS} unique COTC021/022 subject IDs was found. "
                "The 324->186 selection mechanism therefore cannot yet be "
                "audited at patient level."
            )
        else:
            parent = best["table"]
            parent_ids = best["ids"]
            parent_column_frame = column_audit(
                parent,
                source_label=(
                    best["record"]["relative_path"]
                    + "::"
                    + best["sheet"]
                ),
            )
            id_audit, balance, arm_audit = compare_selection(
                parent=parent,
                parent_ids=parent_ids,
                selected_ids=selected_ids,
            )

            selected_found = int(
                id_audit.loc[
                    id_audit["metric"].eq("selected_ids_found_in_parent"),
                    "value",
                ].iloc[0]
            )
            parent_unique = int(
                id_audit.loc[
                    id_audit["metric"].eq("parent_unique_cotc021_022"),
                    "value",
                ].iloc[0]
            )

            strong_effects = []
            if not balance.empty:
                effect = pd.to_numeric(balance["effect_size"], errors="coerce").abs()
                strong_effects = balance.loc[
                    effect >= 0.25,
                    "semantic",
                ].astype(str).tolist()

            arm_problem = False
            if not arm_audit.empty and arm_audit.shape[0] == 2:
                fractions = pd.to_numeric(
                    arm_audit["selection_fraction"], errors="coerce"
                )
                if fractions.notna().all():
                    arm_problem = (
                        float(fractions.max() - fractions.min()) >= 0.10
                    )

            if selected_found < 0.95 * len(selected_ids):
                gate = "AUDITABLE_ID_COVERAGE_INCOMPLETE"
                reason = (
                    "A parent-level candidate table was found, but fewer than "
                    "95% of selected transcriptomic dogs were linked to it."
                )
            elif strong_effects or arm_problem:
                gate = "AUDITABLE_SELECTION_IMBALANCE_DETECTED"
                reason = (
                    "Parent-level metadata are available, but baseline or "
                    "treatment-arm selection differences require explicit "
                    "selection-bias handling before randomized-arm causal claims."
                )
            else:
                gate = "AUDITABLE_NO_MAJOR_IMBALANCE_DETECTED"
                reason = (
                    "Parent-level metadata were found and no large measured "
                    "selection imbalance was detected by prespecified descriptive "
                    "thresholds. This supports, but does not prove, preservation "
                    "of randomized-arm comparability in the RNA subset."
                )

            print("")
            print("Best parent-level candidate:")
            print(f"  file: {best['record']['relative_path']}")
            print(f"  sheet: {best['sheet']}")
            print(f"  unique COTC021/022 IDs: {parent_unique}")
            print(f"  selected IDs found: {selected_found}/{len(selected_ids)}")

    if not OUT_CANDIDATES.exists():
        pd.DataFrame(
            columns=[
                "relative_path",
                "sheet",
                "file_size_bytes",
                "rows",
                "columns",
                "unique_cotc021_022_ids",
                "selected186_id_overlap",
                "selected186_overlap_fraction",
                "candidate_parent_status",
            ]
        ).to_csv(OUT_CANDIDATES, index=False)

    parent_column_frame.to_csv(OUT_PARENT_COLUMNS, index=False)
    balance.to_csv(OUT_BALANCE, index=False)
    arm_audit.to_csv(OUT_ARM_AUDIT, index=False)
    id_audit.to_csv(OUT_ID_AUDIT, index=False)

    selected_summary = selected_186_summary(clinical)

    status_payload = {
        "script_version": SCRIPT_VERSION,
        "created_utc": utc_now_iso(),
        "paper4_root_resolution_source": resolution_source,
        "upstream_lock_verified": True,
        "selected_transcriptomic_n": int(clinical.shape[0]),
        "selected_cotc021_022_ids_detected": len(selected_ids),
        "published_parent_randomized_n": PARENT_RANDOMIZED_N,
        "published_parent_itt_n": PARENT_ITT_N,
        "selected186_summary": selected_summary,
        "selection_audit_gate": gate,
        "selection_audit_reason": reason,
        "causal_arm_invariance_permission": (
            "NOT_YET_GRANTED"
            if gate != "AUDITABLE_NO_MAJOR_IMBALANCE_DETECTED"
            else "MEASURED_SELECTION_BALANCE_SUPPORTS_FURTHER_RANDOMIZATION_AUDIT"
        ),
        "interpretation_guardrails": [
            (
                "The original 324-dog randomization does not automatically "
                "guarantee exchangeability in the 186-dog RNA subset."
            ),
            (
                "Absence of large measured imbalance cannot rule out selection "
                "on unmeasured variables."
            ),
            (
                "Exact blocked-randomization inference must not be claimed until "
                "the original randomization strata and RNA-subset selection "
                "mechanism are verified."
            ),
            (
                "A 93/93 treatment split inside the RNA subset alone is not "
                "evidence that selection was independent of treatment."
            ),
        ],
        "best_parent_candidate": (
            None
            if best is None
            else {
                "relative_path": best["record"]["relative_path"],
                "sheet": best["sheet"],
                "unique_cotc021_022_ids": int(
                    best["record"]["unique_cotc021_022_ids"]
                ),
                "selected186_id_overlap": int(
                    best["record"]["selected186_id_overlap"]
                ),
                "sha256": sha256_file(best["path"]),
            }
        ),
        "outputs": {
            "candidate_tables": OUT_CANDIDATES.name,
            "selected186_columns": OUT_SELECTED_COLUMNS.name,
            "parent_columns": OUT_PARENT_COLUMNS.name,
            "selection_balance": OUT_BALANCE.name,
            "treatment_arm_selection": OUT_ARM_AUDIT.name,
            "id_audit": OUT_ID_AUDIT.name,
        },
    }

    OUT_CONTRACT.write_text(
        json.dumps(status_payload, indent=2),
        encoding="utf-8",
    )

    readme = f"""Paper 6 DOG2 transcriptomic-selection audit
Script version: {SCRIPT_VERSION}

Question
--------
Does the 186-dog bulk-RNA subset preserve enough of the original COTC021/022
randomized parent cohort structure to justify later treatment-arm invariance
analyses?

Published design facts used only as reference
---------------------------------------------
- Randomized parent trial enrollment: {PARENT_RANDOMIZED_N} dogs.
- Intent-to-treat population reported in the trial publication: {PARENT_ITT_N}.
- Bulk-RNA subset: {EXPECTED_SELECTED_N} dogs.

Current gate
------------
{gate}

Reason
------
{reason}

Important
---------
The original trial randomization applies to the randomized parent population.
Subsetting by tissue availability, RNA quality, pathology review, or medical
record completeness can induce selection bias. A balanced 93/93 RNA subset is
therefore not sufficient by itself to justify exact randomization inference.

This script does not fit any survival or molecular model and does not perform
treatment-effect discovery.
"""
    OUT_README.write_text(readme, encoding="utf-8")

    print("")
    print("=" * 96)
    print("DOG2 transcriptomic-selection audit complete")
    print("=" * 96)
    print(f"Selection audit gate: {gate}")
    print(f"Reason: {reason}")
    print("")
    print("Saved:")
    for path in [
        OUT_CANDIDATES,
        OUT_SELECTED_COLUMNS,
        OUT_PARENT_COLUMNS,
        OUT_BALANCE,
        OUT_ARM_AUDIT,
        OUT_ID_AUDIT,
        OUT_CONTRACT,
        OUT_README,
    ]:
        print(f"  {path}")
    print("")
    if gate == "BLOCKED_NO_PATIENT_LEVEL_PARENT_COHORT":
        print("Next action:")
        print(
            "  Obtain a small patient-level COTC021/022 parent metadata table "
            "(no omics required), place it anywhere under Paper 4, then re-run "
            "this script."
        )
        print(
            "  A current candidate is the 2026 DOG2 multiomic Supplementary "
            "Table S2, but its COTC021/022 coverage must be checked rather than "
            "assumed."
        )
    elif gate == "AUDITABLE_NO_MAJOR_IMBALANCE_DETECTED":
        print("Next action:")
        print(
            "  Proceed to a dedicated randomization/arm-invariance protocol "
            "audit before any Gate-Zero model fitting."
        )
    else:
        print("Next action:")
        print(
            "  Review the selection-balance outputs before using treatment "
            "arms as a causal anchor."
        )
    print("Done.")


if __name__ == "__main__":
    main()
