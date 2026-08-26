from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from typing import Any

import pandas as pd


SCRIPT_VERSION = "02a-resolve-dog2-authoritative-ids-v3-no-cli"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "_config"
CONTRACT_DIR = PROJECT_ROOT / "contracts"
MANIFEST_DIR = PROJECT_ROOT / "manifests"

LOCAL_PATH_CONFIG = CONFIG_DIR / "paths.local.json"
UPSTREAM_LOCK = CONTRACT_DIR / "00_upstream_input_lock.json"
UPSTREAM_MANIFEST = MANIFEST_DIR / "00_upstream_input_manifest.csv"

OUTPUT_MAPPING = MANIFEST_DIR / "02a_dog2_selected186_authoritative_id_mapping.csv"
OUTPUT_AUDIT = MANIFEST_DIR / "02a_dog2_authoritative_id_mapping_audit.csv"
OUTPUT_LOCK = CONTRACT_DIR / "02a_dog2_authoritative_id_mapping_lock.json"
OUTPUT_README = MANIFEST_DIR / "02a_dog2_authoritative_id_mapping_README.txt"

EXPECTED_PAPER4_BASENAME = "paper4_sarcoma_dog"
EXPECTED_N = 186

AUTHORITATIVE_SUBJECT_PATH = (
    "paper5/data_manifest/DOG2_subject_ids_authoritative.csv"
)

SAMPLE_MAP_PATH = "data/processed/GSE238110_sample_id_map.csv"
MATCHED_CLINICAL_PATH = "data/processed/DOG2_clinical_matched_to_GSE238110.csv"

COTC_PATTERN = re.compile(
    r"\b(COTC0?\d{2,3})[-_ ]+0*(\d{2,6})(?:[-_ ]+[A-Z])?\b",
    flags=re.IGNORECASE,
)

EXPRESSION_PATIENT_PATTERN = re.compile(
    r"^\d+_0*(\d+)",
    flags=re.IGNORECASE,
)


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


def normalize_integer_id(value: Any) -> str:
    """Normalize an explicitly patient-like scalar to an integer string.

    Examples:
        514 -> "514"
        "0514" -> "514"
        "514.0" -> "514"

    This function intentionally does NOT search arbitrary strings for digits.
    """
    text = clean_text(value)
    if not text:
        return ""

    try:
        numeric = float(text)
    except ValueError:
        return ""

    if not numeric.is_integer() or numeric < 0:
        return ""

    return str(int(numeric))


def extract_expression_patient_id(value: Any) -> str:
    """Recover the patient identifier used in the original Paper 4 matching.

    The original GSE238110 sample columns used a pattern such as:
        100_0514_tumor
    which was matched to clinical Patient ID 514.
    """
    text = clean_text(value)
    if not text:
        return ""

    match = EXPRESSION_PATIENT_PATTERN.search(text)
    if match is None:
        return ""

    digits = match.group(1).lstrip("0") or "0"
    return str(int(digits))


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


def cotc_patient_suffix(cotc_id: str) -> str:
    normalized = normalize_cotc(cotc_id)
    if not normalized:
        return ""
    suffix = normalized.split("-", 1)[1]
    return str(int(suffix))


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
            (
                Path(env_value).expanduser(),
                "environment:PAPER4_ROOT",
            )
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


def verify_upstream_lock(paper4_root: Path) -> dict[str, Any]:
    if not UPSTREAM_LOCK.exists() or not UPSTREAM_MANIFEST.exists():
        raise FileNotFoundError(
            "Run scripts/00_lock_upstream_inputs.py first."
        )

    lock = json.loads(
        UPSTREAM_LOCK.read_text(encoding="utf-8")
    )

    observed_manifest_hash = sha256_file(UPSTREAM_MANIFEST)
    if observed_manifest_hash != lock.get("manifest_sha256"):
        raise RuntimeError(
            "Script-00 upstream manifest hash mismatch."
        )

    metadata = lock["assets"]["dog2_clinical"]
    locked_path = paper4_root / metadata["relative_path"]

    if not locked_path.exists():
        raise FileNotFoundError(
            f"Missing locked DOG2 clinical table: {locked_path}"
        )

    if sha256_file(locked_path) != metadata["sha256"]:
        raise RuntimeError(
            "Locked DOG2 clinical table changed since Script 00."
        )

    return lock


def all_series(table: pd.DataFrame) -> dict[str, pd.Series]:
    output = {
        "__index__": pd.Series(
            table.index.astype(str),
            index=table.index,
            dtype="object",
        )
    }
    for column in table.columns:
        output[str(column)] = table[column]
    return output


def load_authoritative_subjects(
    paper4_root: Path,
) -> tuple[pd.DataFrame, Path, str]:
    path = paper4_root / AUTHORITATIVE_SUBJECT_PATH
    if not path.is_file():
        raise FileNotFoundError(
            "Paper 5 authoritative DOG2 subject file not found:\n"
            f"  {path}"
        )

    table = pd.read_csv(
        path,
        dtype=str,
        low_memory=False,
    ).fillna("")

    candidate_rows = []
    best_source = None
    best_values = None

    for source, values in all_series(table).items():
        normalized = values.map(normalize_cotc)
        n_parseable = int(normalized.ne("").sum())
        n_unique = int(
            normalized[normalized.ne("")].nunique()
        )

        candidate_rows.append(
            {
                "section": "authoritative_cotc_source",
                "table": AUTHORITATIVE_SUBJECT_PATH,
                "source": source,
                "n_rows": int(table.shape[0]),
                "n_parseable": n_parseable,
                "n_unique": n_unique,
                "example_values": " | ".join(
                    values.astype(str).head(5).tolist()
                ),
            }
        )

        if (
            n_parseable == EXPECTED_N
            and n_unique == EXPECTED_N
        ):
            if (
                best_source is None
                or "cotc" in source.lower()
                or "subject" in source.lower()
            ):
                best_source = source
                best_values = normalized

    audit = pd.DataFrame(candidate_rows)

    if best_source is None or best_values is None:
        raise RuntimeError(
            "No authoritative column contains exactly 186 unique COTC IDs.\n"
            + audit.sort_values(
                ["n_parseable", "n_unique"],
                ascending=False,
            ).head(10).to_string(index=False)
        )

    authoritative = pd.DataFrame(
        {
            "cotc_subject_id": best_values,
        }
    )
    authoritative = authoritative[
        authoritative["cotc_subject_id"].ne("")
    ].drop_duplicates()

    authoritative["paper4_patient_id"] = (
        authoritative["cotc_subject_id"]
        .map(cotc_patient_suffix)
    )

    if authoritative.shape[0] != EXPECTED_N:
        raise RuntimeError(
            f"Authoritative subject rows={authoritative.shape[0]}, "
            f"expected {EXPECTED_N}."
        )

    return authoritative, path, best_source


def find_exact_patient_id_source(
    locked: pd.DataFrame,
    authoritative_patient_ids: set[str],
) -> tuple[str, pd.Series, pd.DataFrame]:
    rows = []
    exact_sources = []

    for source, values in all_series(locked).items():
        normalized = values.map(normalize_integer_id)
        nonempty = normalized.ne("")
        value_set = set(normalized[nonempty])

        rows.append(
            {
                "section": "locked_clinical_patient_id_source",
                "source": source,
                "n_rows": int(locked.shape[0]),
                "n_parseable_integer_ids": int(nonempty.sum()),
                "n_unique_integer_ids": len(value_set),
                "overlap_with_authoritative_suffixes": len(
                    value_set & authoritative_patient_ids
                ),
                "missing_from_authoritative_suffixes": len(
                    value_set - authoritative_patient_ids
                ),
                "authoritative_suffixes_not_present": len(
                    authoritative_patient_ids - value_set
                ),
                "exact_set_match": (
                    len(value_set) == EXPECTED_N
                    and value_set == authoritative_patient_ids
                ),
                "example_values": " | ".join(
                    values.astype(str).head(5).tolist()
                ),
            }
        )

        if (
            int(nonempty.sum()) == EXPECTED_N
            and len(value_set) == EXPECTED_N
            and value_set == authoritative_patient_ids
        ):
            exact_sources.append(
                (
                    1 if str(source).lower() == "patient id" else 0,
                    1 if "patient" in str(source).lower() else 0,
                    str(source),
                    normalized,
                )
            )

    audit = pd.DataFrame(rows)

    if not exact_sources:
        raise RuntimeError(
            "No locked clinical identifier column matches the 186 "
            "authoritative COTC patient suffixes exactly.\n"
            "Best candidates:\n"
            + audit.sort_values(
                [
                    "overlap_with_authoritative_suffixes",
                    "n_unique_integer_ids",
                ],
                ascending=False,
            ).head(10).to_string(index=False)
        )

    exact_sources.sort(
        key=lambda x: (x[0], x[1], x[2]),
        reverse=True,
    )
    _, _, source, normalized = exact_sources[0]

    return source, normalized, audit


def optional_sample_map_crosscheck(
    paper4_root: Path,
    authoritative_patient_ids: set[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = paper4_root / SAMPLE_MAP_PATH

    if not path.is_file():
        return pd.DataFrame(), {
            "available": False,
            "relative_path": SAMPLE_MAP_PATH,
        }

    table = pd.read_csv(
        path,
        dtype=str,
        low_memory=False,
    ).fillna("")

    rows = []

    for source, values in all_series(table).items():
        parsed = values.map(extract_expression_patient_id)
        parsed_set = set(parsed[parsed.ne("")])

        rows.append(
            {
                "section": "sample_map_expression_patient_id_source",
                "source": source,
                "n_rows": int(table.shape[0]),
                "n_parseable_expression_patient_ids": int(
                    parsed.ne("").sum()
                ),
                "n_unique_expression_patient_ids": len(parsed_set),
                "overlap_with_authoritative_suffixes": len(
                    parsed_set & authoritative_patient_ids
                ),
                "authoritative_suffixes_not_present": len(
                    authoritative_patient_ids - parsed_set
                ),
                "example_values": " | ".join(
                    values.astype(str).head(5).tolist()
                ),
            }
        )

    audit = pd.DataFrame(rows)

    best = audit.sort_values(
        [
            "overlap_with_authoritative_suffixes",
            "n_unique_expression_patient_ids",
        ],
        ascending=False,
    ).iloc[0]

    return audit, {
        "available": True,
        "relative_path": SAMPLE_MAP_PATH,
        "sha256": sha256_file(path),
        "rows": int(table.shape[0]),
        "best_source": str(best["source"]),
        "best_unique_expression_patient_ids": int(
            best["n_unique_expression_patient_ids"]
        ),
        "best_authoritative_overlap": int(
            best["overlap_with_authoritative_suffixes"]
        ),
        "authoritative_suffixes_not_present": int(
            best["authoritative_suffixes_not_present"]
        ),
    }


def optional_original_match_crosscheck(
    paper4_root: Path,
    authoritative_patient_ids: set[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = paper4_root / MATCHED_CLINICAL_PATH

    if not path.is_file():
        return pd.DataFrame(), {
            "available": False,
            "relative_path": MATCHED_CLINICAL_PATH,
        }

    table = pd.read_csv(
        path,
        dtype=str,
        low_memory=False,
    ).fillna("")

    rows = []

    for source, values in all_series(table).items():
        numeric = values.map(normalize_integer_id)
        numeric_set = set(numeric[numeric.ne("")])

        expression = values.map(
            extract_expression_patient_id
        )
        expression_set = set(
            expression[expression.ne("")]
        )

        rows.append(
            {
                "section": "original_match_identifier_source",
                "source": source,
                "n_rows": int(table.shape[0]),
                "numeric_unique": len(numeric_set),
                "numeric_authoritative_overlap": len(
                    numeric_set & authoritative_patient_ids
                ),
                "expression_pattern_unique": len(
                    expression_set
                ),
                "expression_authoritative_overlap": len(
                    expression_set & authoritative_patient_ids
                ),
                "example_values": " | ".join(
                    values.astype(str).head(5).tolist()
                ),
            }
        )

    audit = pd.DataFrame(rows)

    best_numeric = audit.sort_values(
        [
            "numeric_authoritative_overlap",
            "numeric_unique",
        ],
        ascending=False,
    ).iloc[0]

    best_expression = audit.sort_values(
        [
            "expression_authoritative_overlap",
            "expression_pattern_unique",
        ],
        ascending=False,
    ).iloc[0]

    return audit, {
        "available": True,
        "relative_path": MATCHED_CLINICAL_PATH,
        "sha256": sha256_file(path),
        "rows": int(table.shape[0]),
        "best_numeric_source": str(
            best_numeric["source"]
        ),
        "best_numeric_authoritative_overlap": int(
            best_numeric["numeric_authoritative_overlap"]
        ),
        "best_expression_source": str(
            best_expression["source"]
        ),
        "best_expression_authoritative_overlap": int(
            best_expression[
                "expression_authoritative_overlap"
            ]
        ),
    }


def main() -> None:
    print("=" * 96)
    print("Paper 6 - authoritative DOG2 identifier bridge v3")
    print("=" * 96)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print("")
    print("Repair strategy:")
    print("  Use the already-frozen Paper 5 list of 186 authoritative COTC subjects.")
    print("  Recover each COTC subject's numeric patient suffix.")
    print("  Require those 186 suffixes to be globally unique.")
    print("  Require an exact 186/186 set match to one identifier field in the")
    print("  locked Paper 4 DOG2 clinical table.")
    print("  Only then create the Paper 4 row -> COTC subject bridge.")
    print("")
    print("No outcome modeling, no feature selection, no row-order matching,")
    print("and no many-to-many merge are allowed.")
    print("")

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    CONTRACT_DIR.mkdir(parents=True, exist_ok=True)

    paper4_root, resolution_source = resolve_paper4_root()
    lock = verify_upstream_lock(paper4_root)

    clinical_meta = lock["assets"]["dog2_clinical"]
    locked_path = paper4_root / clinical_meta["relative_path"]

    locked = pd.read_csv(
        locked_path,
        index_col=0,
        dtype=str,
        low_memory=False,
    ).fillna("")

    if locked.shape[0] != EXPECTED_N:
        raise RuntimeError(
            f"Locked DOG2 clinical rows={locked.shape[0]}; "
            f"expected {EXPECTED_N}."
        )

    authoritative, authoritative_path, authoritative_source = (
        load_authoritative_subjects(paper4_root)
    )

    authoritative_patient_ids = set(
        authoritative["paper4_patient_id"]
    )

    suffix_collision_rows = (
        authoritative[
            authoritative["paper4_patient_id"].duplicated(
                keep=False
            )
        ]
        .sort_values("paper4_patient_id")
        .copy()
    )

    if len(authoritative_patient_ids) != EXPECTED_N:
        print("")
        print("COTC suffix collision audit:")
        if suffix_collision_rows.empty:
            print("  No duplicate rows found, but suffix count is unexpected.")
        else:
            print(
                suffix_collision_rows.to_string(
                    index=False
                )
            )
        raise RuntimeError(
            "The 186 authoritative COTC subjects do not have 186 globally "
            "unique numeric patient suffixes. Numeric bridging is therefore "
            "not scientifically safe and has been stopped."
        )

    patient_source, patient_ids, locked_audit = (
        find_exact_patient_id_source(
            locked=locked,
            authoritative_patient_ids=authoritative_patient_ids,
        )
    )

    suffix_to_cotc = dict(
        zip(
            authoritative["paper4_patient_id"],
            authoritative["cotc_subject_id"],
            strict=True,
        )
    )

    mapping = pd.DataFrame(
        {
            "paper4_sample_id": locked.index.astype(str),
            "paper4_patient_id": patient_ids.values,
        }
    )
    mapping["cotc_subject_id"] = (
        mapping["paper4_patient_id"].map(
            suffix_to_cotc
        )
    )

    if mapping["paper4_patient_id"].eq("").any():
        raise RuntimeError(
            "At least one locked Paper 4 row has an empty patient ID."
        )

    if mapping["paper4_patient_id"].duplicated().any():
        raise RuntimeError(
            "Locked Paper 4 patient IDs are not one-to-one."
        )

    if mapping["cotc_subject_id"].isna().any():
        raise RuntimeError(
            "At least one locked Paper 4 patient ID failed to map to COTC."
        )

    selected_cotc = set(
        mapping["cotc_subject_id"].astype(str)
    )
    authoritative_cotc = set(
        authoritative["cotc_subject_id"].astype(str)
    )

    sample_map_audit, sample_map_summary = (
        optional_sample_map_crosscheck(
            paper4_root=paper4_root,
            authoritative_patient_ids=authoritative_patient_ids,
        )
    )

    original_match_audit, original_match_summary = (
        optional_original_match_crosscheck(
            paper4_root=paper4_root,
            authoritative_patient_ids=authoritative_patient_ids,
        )
    )

    checks = {
        "locked_rows_186": int(locked.shape[0]) == EXPECTED_N,
        "authoritative_cotc_186": (
            len(authoritative_cotc) == EXPECTED_N
        ),
        "authoritative_numeric_suffixes_unique_186": (
            len(authoritative_patient_ids) == EXPECTED_N
        ),
        "locked_patient_ids_unique_186": (
            mapping["paper4_patient_id"].nunique()
            == EXPECTED_N
        ),
        "locked_patient_id_set_equals_authoritative_suffix_set": (
            set(mapping["paper4_patient_id"])
            == authoritative_patient_ids
        ),
        "mapped_cotc_unique_186": (
            mapping["cotc_subject_id"].nunique()
            == EXPECTED_N
        ),
        "mapped_cotc_set_equals_authoritative_set": (
            selected_cotc == authoritative_cotc
        ),
    }

    passed = all(checks.values())
    status = "PASS" if passed else "FAIL"

    reason = (
        "The locked 186-dog Paper 4 clinical cohort matches the 186 "
        "authoritative GSE238110/Paper 5 COTC subjects exactly through "
        "a globally unique numeric patient identifier bridge."
        if passed
        else "One or more exact identifier-bridge checks failed."
    )

    mapping.to_csv(
        OUTPUT_MAPPING,
        index=False,
    )

    audit_frames = [
        locked_audit,
    ]

    if not sample_map_audit.empty:
        audit_frames.append(sample_map_audit)

    if not original_match_audit.empty:
        audit_frames.append(original_match_audit)

    audit_output = pd.concat(
        audit_frames,
        ignore_index=True,
        sort=False,
    )
    audit_output.to_csv(
        OUTPUT_AUDIT,
        index=False,
    )

    payload = {
        "script_version": SCRIPT_VERSION,
        "created_utc": utc_now_iso(),
        "status": status,
        "status_reason": reason,
        "paper4_root_resolution_source": resolution_source,
        "upstream_lock_verified": True,
        "mapping_strategy": (
            "exact_set_equality_between_locked_clinical_patient_ids_"
            "and_unique_numeric_suffixes_of_authoritative_COTC_subjects"
        ),
        "selected_locked_patient_id_source": patient_source,
        "sources": {
            "locked_dog2_clinical": {
                "relative_path": clinical_meta["relative_path"],
                "sha256": clinical_meta["sha256"],
            },
            "paper5_authoritative_subjects": {
                "relative_path": AUTHORITATIVE_SUBJECT_PATH,
                "sha256": sha256_file(
                    authoritative_path
                ),
                "selected_cotc_source": authoritative_source,
            },
        },
        "checks": checks,
        "counts": {
            "locked_rows": int(locked.shape[0]),
            "authoritative_cotc_subjects": len(
                authoritative_cotc
            ),
            "authoritative_unique_numeric_suffixes": len(
                authoritative_patient_ids
            ),
            "mapped_rows": int(mapping.shape[0]),
            "mapped_unique_patient_ids": int(
                mapping["paper4_patient_id"].nunique()
            ),
            "mapped_unique_cotc_subjects": int(
                mapping["cotc_subject_id"].nunique()
            ),
        },
        "optional_sample_map_crosscheck": sample_map_summary,
        "optional_original_match_crosscheck": original_match_summary,
        "scientific_data_copied": False,
        "model_fitting": False,
        "outcome_association_testing": False,
        "row_order_matching_used": False,
        "many_to_many_merge_used": False,
        "output_mapping": OUTPUT_MAPPING.name,
        "output_mapping_sha256": sha256_file(
            OUTPUT_MAPPING
        ),
    }

    OUTPUT_LOCK.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    OUTPUT_README.write_text(
        f"""Paper 6 authoritative DOG2 identifier bridge
Script version: {SCRIPT_VERSION}

Status
------
{status}

Reason
------
{reason}

Why earlier versions failed
---------------------------
The processed Paper 4 clinical table does not store COTC subject strings or a
unique GSM identifier. Earlier scripts therefore attempted joins in identifier
spaces that were not actually present.

Version 3 uses the matching relationship that was used in Paper 4 itself:
the DOG2 clinical data contain a numeric Patient ID, while official GSE238110
COTC titles contain the same numeric patient identifier as the subject suffix.

This bridge is accepted only if:
1. all 186 authoritative COTC subjects have globally unique numeric suffixes;
2. one locked clinical identifier field contains exactly 186 unique IDs;
3. the two 186-ID sets are exactly equal; and
4. the resulting COTC set exactly equals the authoritative Paper 5 subject set.

If any condition fails, the script stops. It never falls back to row order.

No scientific dataset is copied. No outcome model or feature selection is run.
""",
        encoding="utf-8",
    )

    print(f"Paper 4 root: {paper4_root}")
    print("00 upstream lock: PASS")
    print("")
    print("Authoritative subject source:")
    print(
        "  "
        + authoritative_path.resolve().relative_to(
            paper4_root.resolve()
        ).as_posix()
    )
    print(f"  COTC column/source: {authoritative_source}")
    print("")
    print(f"Locked clinical patient-ID source: {patient_source}")
    print("")
    print("Exact bridge counts:")
    print(f"  locked rows: {locked.shape[0]}")
    print(
        "  authoritative COTC subjects: "
        f"{len(authoritative_cotc)}"
    )
    print(
        "  unique authoritative numeric suffixes: "
        f"{len(authoritative_patient_ids)}"
    )
    print(
        "  mapped unique patient IDs: "
        f"{mapping['paper4_patient_id'].nunique()}"
    )
    print(
        "  mapped unique COTC subjects: "
        f"{mapping['cotc_subject_id'].nunique()}"
    )
    print(
        "  patient-ID set equality: "
        f"{checks['locked_patient_id_set_equals_authoritative_suffix_set']}"
    )
    print(
        "  COTC-set equality: "
        f"{checks['mapped_cotc_set_equals_authoritative_set']}"
    )

    if sample_map_summary.get("available"):
        print("")
        print("Optional GSE238110 sample-map cross-check:")
        print(
            "  best source: "
            f"{sample_map_summary['best_source']}"
        )
        print(
            "  authoritative patient IDs recovered: "
            f"{sample_map_summary['best_authoritative_overlap']}/{EXPECTED_N}"
        )

    if original_match_summary.get("available"):
        print("")
        print("Optional original Paper 4 clinical-match cross-check:")
        print(
            "  best numeric source: "
            f"{original_match_summary['best_numeric_source']}"
        )
        print(
            "  numeric authoritative overlap: "
            f"{original_match_summary['best_numeric_authoritative_overlap']}/{EXPECTED_N}"
        )
        print(
            "  best expression-name source: "
            f"{original_match_summary['best_expression_source']}"
        )
        print(
            "  expression-name authoritative overlap: "
            f"{original_match_summary['best_expression_authoritative_overlap']}/{EXPECTED_N}"
        )

    print("")
    print("=" * 96)
    print(f"Authoritative DOG2 identifier bridge v3: {status}")
    print("=" * 96)
    print(reason)
    print("")
    print("Saved:")
    print(f"  {OUTPUT_MAPPING}")
    print(f"  {OUTPUT_AUDIT}")
    print(f"  {OUTPUT_LOCK}")
    print(f"  {OUTPUT_README}")

    if not passed:
        raise RuntimeError(
            "Authoritative DOG2 identifier bridge v3 failed. "
            "Do not continue to the 324->186 selection audit."
        )

    print("")
    print("Next:")
    print(
        "  Script 02b can now audit 324->186 selection using the frozen "
        "COTC mapping generated here."
    )
    print("Done.")


if __name__ == "__main__":
    main()
