#!/usr/bin/env python3
'''
Paper 6 - build transcriptome-wide outcome-blind dog->human ortholog bridge.

This stage replaces the superseded 02f0 candidate-restricted feature universe.

PRIMARY PRINCIPLE
-----------------
Start from ALL 21,016 locked DOG2 expression feature names and an outcome-blind,
versioned Ensembl dog-human orthology snapshot.

The Paper-4 RNA candidate/evidence tables are FORBIDDEN as row selectors here.
The historical Paper-4 BioMart cache is allowed only as an outcome-blind mapping
sensitivity/reference snapshot.

FRESH PRIMARY MAPPING
---------------------
- current Ensembl release is queried at runtime and recorded;
- current dog and human assemblies are recorded;
- BioMart dataset/attributes are discovered and frozen;
- raw mapping response is cached and SHA256-locked;
- reruns reuse the first validated snapshot rather than silently moving releases.

PRIMARY BRIDGE RULE
-------------------
A DOG2 feature is primary-eligible only if:
1. its canine symbol is resolved deterministically without outcome information;
2. the resolved symbol is represented by exactly one DOG2 feature (no feature collision);
3. current Ensembl reports a dog-human one-to-one orthology row;
4. that canine symbol maps uniquely to one dog Ensembl gene, one human Ensembl gene,
   and one non-empty human symbol within the one-to-one mapping;
5. the human symbol is not shared by another primary-eligible canine symbol;
6. if current BioMart exposes orthology confidence, confidence must equal 1;
7. the mapped human symbol is present in the relevant locked human expression header.

Dog feature-symbol resolution:
- exact current-Ensembl symbol match has priority;
- if exact match is absent, a terminal duplicate suffix _N (N>=2) may be stripped only
  when the base symbol exists in the current Ensembl dog-symbol inventory;
- if multiple DOG2 raw features resolve to the same canine symbol, the whole symbol
  group is excluded from the PRIMARY bridge rather than choosing a feature by outcome.
  A later sensitivity may prespecify deterministic aggregation.

No clinical/outcome values are read. No model is fit.

No command-line arguments are used.
'''

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import requests


SCRIPT_VERSION = "02g-build-transcriptomewide-outcome-blind-ortholog-bridge-v1-no-cli"

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "_config"
PAPER4_BASENAME = "paper4_sarcoma_dog"

AMEND_DIR = ROOT / "results" / "transport_contract" / "02f0a_amendment"
AMENDMENT = AMEND_DIR / "transport_contract_amendment.json"
AMENDMENT_SUMMARY = AMEND_DIR / "summary.json"
UPSTREAM_LOCK = ROOT / "contracts" / "00_upstream_input_lock.json"

OUT_DIR = ROOT / "results" / "ortholog_bridge" / "02g"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SNAPSHOT_RAW = OUT_DIR / "ensembl_current_dog_human_biomart_raw.tsv"
SNAPSHOT_META = OUT_DIR / "ensembl_current_snapshot_metadata.json"
ATTR_AUDIT = OUT_DIR / "ensembl_biomart_attribute_audit.tsv"
FEATURE_RESOLUTION = OUT_DIR / "dog2_feature_symbol_resolution.tsv"
CURRENT_ONE2ONE = OUT_DIR / "current_one2one_mapping_audit.tsv"
PRIMARY_BRIDGE = OUT_DIR / "primary_outcome_blind_ortholog_bridge.tsv"
TRANSPORT_FEATURES = OUT_DIR / "transport_feature_sets.tsv"
FUNNEL_TSV = OUT_DIR / "mapping_funnel.tsv"
FEATURE_SUMMARY = OUT_DIR / "feature_set_summary.tsv"
HIST_SENSITIVITY = OUT_DIR / "historical_biomart_mapping_sensitivity.tsv"
CONTRACT_JSON = OUT_DIR / "ortholog_bridge_contract.json"
SUMMARY_JSON = OUT_DIR / "summary.json"

ENSEMBL_REST = "https://rest.ensembl.org"
ENSEMBL_MART = "https://www.ensembl.org/biomart/martservice"
DOG_DATASET = "clfamiliaris_gene_ensembl"

EXPECTED_DOG2_FEATURES = 21016
EXPECTED_HIST_BIOMART_ROWS = 34159
EXPECTED_HIST_UNIQUE_DOG_SYMBOLS = 16955
EXPECTED_HIST_ANY_HUMAN = 16063
EXPECTED_HIST_ONE2ONE = 15318

REQUIRED_CURRENT_ATTRIBUTES = [
    "ensembl_gene_id",
    "external_gene_name",
    "hsapiens_homolog_ensembl_gene",
    "hsapiens_homolog_associated_gene_name",
    "hsapiens_homolog_orthology_type",
]

OPTIONAL_CURRENT_ATTRIBUTES = [
    "hsapiens_homolog_orthology_confidence",
    "hsapiens_homolog_perc_id",
    "hsapiens_homolog_perc_id_r1",
    "hsapiens_homolog_goc_score",
    "hsapiens_homolog_wga_coverage",
]

HIST_BIOMART_REL = Path("data") / "external" / "ensembl_dog_human_orthologs_biomart.tsv"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_lines(values: Iterable[str]) -> str:
    text = "\n".join(sorted(str(x) for x in values)) + "\n"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "na", "n/a"}:
        return ""
    return text


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def require_file(path: Path) -> Path:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Required file missing: {path}")
    return path


def local_path_config() -> Dict[str, Any]:
    path = CONFIG_DIR / "paths.local.json"
    if not path.exists():
        return {}
    payload = read_json(path)
    return payload if isinstance(payload, dict) else {}


def resolve_paper4_root() -> Tuple[Path, str]:
    candidates: List[Tuple[Path, str]] = []
    env_value = os.environ.get("PAPER4_ROOT", "").strip()
    if env_value:
        candidates.append((Path(env_value).expanduser(), "environment:PAPER4_ROOT"))

    config_value = clean(local_path_config().get("paper4_root"))
    if config_value:
        candidates.append((Path(config_value).expanduser(), "_config/paths.local.json"))

    candidates.extend(
        [
            (ROOT.parent / PAPER4_BASENAME, "sibling_repository"),
            (Path.home() / "Desktop" / PAPER4_BASENAME, "home_desktop_fallback"),
        ]
    )

    checked: List[str] = []
    for candidate, source in candidates:
        resolved = candidate.resolve()
        checked.append(f"{source}: {resolved}")
        if resolved.is_dir():
            return resolved, source

    raise FileNotFoundError(
        "Could not resolve Paper 4 root. Checked:\n  - " + "\n  - ".join(checked)
    )


def http_get_json(
    session: requests.Session,
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = 60,
    retries: int = 3,
) -> Any:
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(
                url,
                params=params,
                headers={"Accept": "application/json", "User-Agent": "Paper6-ortholog-bridge/1.0"},
                timeout=timeout,
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(2 * attempt)
    raise RuntimeError(f"GET JSON failed after {retries} attempts: {url}: {last_exc}")


def http_get_text(
    session: requests.Session,
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = 120,
    retries: int = 3,
) -> str:
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(
                url,
                params=params,
                headers={"User-Agent": "Paper6-ortholog-bridge/1.0"},
                timeout=timeout,
            )
            response.raise_for_status()
            return response.text
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(2 * attempt)
    raise RuntimeError(f"GET text failed after {retries} attempts: {url}: {last_exc}")


def http_post_text(
    session: requests.Session,
    url: str,
    *,
    data: Dict[str, Any],
    timeout: int = 240,
    retries: int = 3,
) -> str:
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            response = session.post(
                url,
                data=data,
                headers={"User-Agent": "Paper6-ortholog-bridge/1.0"},
                timeout=timeout,
            )
            response.raise_for_status()
            return response.text
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(3 * attempt)
    raise RuntimeError(f"POST failed after {retries} attempts: {url}: {last_exc}")


def discover_current_release(session: requests.Session) -> int:
    payload = http_get_json(
        session,
        f"{ENSEMBL_REST}/info/data/",
        params={"content-type": "application/json"},
    )
    releases = payload.get("releases") if isinstance(payload, dict) else None
    if not isinstance(releases, list) or not releases:
        raise RuntimeError(f"Could not determine current Ensembl release: {payload!r}")
    return max(int(x) for x in releases)


def discover_assembly(session: requests.Session, aliases: Sequence[str]) -> Dict[str, Any]:
    errors: List[str] = []
    for alias in aliases:
        try:
            payload = http_get_json(
                session,
                f"{ENSEMBL_REST}/info/assembly/{alias}",
                params={"content-type": "application/json"},
                retries=2,
            )
            if isinstance(payload, dict) and clean(payload.get("assembly_name")):
                return {
                    "query_alias": alias,
                    "assembly_name": clean(payload.get("assembly_name")),
                    "assembly_date": clean(payload.get("assembly_date")),
                    "assembly_accession": clean(payload.get("assembly_accession")),
                    "genebuild_last_geneset_update": clean(
                        payload.get("genebuild_last_geneset_update")
                    ),
                    "raw": payload,
                }
        except Exception as exc:
            errors.append(f"{alias}: {exc}")

    raise RuntimeError(
        "Could not determine Ensembl assembly for aliases "
        f"{list(aliases)}. Errors: {' | '.join(errors)}"
    )


def list_biomart_attributes(session: requests.Session) -> pd.DataFrame:
    text = http_get_text(
        session,
        ENSEMBL_MART,
        params={"type": "attributes", "dataset": DOG_DATASET},
        timeout=120,
    )

    rows: List[Dict[str, str]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.rstrip("\n").split("\t")
        rows.append(
            {
                "name": parts[0] if len(parts) > 0 else "",
                "description": parts[1] if len(parts) > 1 else "",
                "page": parts[3] if len(parts) > 3 else "",
                "raw": line,
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty or "name" not in frame.columns:
        raise RuntimeError("BioMart attribute inventory was empty/unparseable.")
    return frame


def build_biomart_query(attributes: Sequence[str]) -> str:
    attribute_xml = "\n".join(
        f'      <Attribute name="{attr}" />' for attr in attributes
    )
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query virtualSchemaName="default"
       formatter="TSV"
       header="1"
       uniqueRows="0"
       count=""
       datasetConfigVersion="0.6">
  <Dataset name="{DOG_DATASET}" interface="default">
{attribute_xml}
  </Dataset>
</Query>'''


def fetch_or_reuse_current_snapshot(
    session: requests.Session,
) -> Tuple[pd.DataFrame, Dict[str, Any], bool]:
    if SNAPSHOT_RAW.exists() and SNAPSHOT_META.exists():
        meta = read_json(SNAPSHOT_META)
        expected_hash = clean(meta.get("raw_sha256"))
        if len(expected_hash) != 64:
            raise RuntimeError(
                "Existing current-Ensembl snapshot metadata lacks a valid raw SHA256."
            )
        actual_hash = sha256_file(SNAPSHOT_RAW)
        if actual_hash != expected_hash:
            raise RuntimeError(
                "Existing current-Ensembl raw snapshot hash does not match its metadata. "
                "Refusing silent replacement."
            )

        attributes = meta.get("selected_attributes") or []
        if not all(attr in attributes for attr in REQUIRED_CURRENT_ATTRIBUTES):
            raise RuntimeError(
                "Existing current-Ensembl snapshot lacks required bridge attributes."
            )

        frame = pd.read_csv(SNAPSHOT_RAW, sep="\t", dtype=str, low_memory=False).fillna("")
        return frame, meta, True

    release = discover_current_release(session)
    dog_assembly = discover_assembly(
        session,
        ["canis_lupus_familiaris", "dog", "canis_familiaris"],
    )
    human_assembly = discover_assembly(session, ["homo_sapiens", "human"])

    attributes = list_biomart_attributes(session)
    available = set(attributes["name"].astype(str))

    missing = [x for x in REQUIRED_CURRENT_ATTRIBUTES if x not in available]
    if missing:
        raise RuntimeError(
            f"Current {DOG_DATASET} BioMart lacks required attributes: {missing}"
        )

    selected = list(REQUIRED_CURRENT_ATTRIBUTES)
    selected.extend([x for x in OPTIONAL_CURRENT_ATTRIBUTES if x in available])

    attr_audit = attributes.copy()
    attr_audit["required_for_02g"] = attr_audit["name"].isin(REQUIRED_CURRENT_ATTRIBUTES)
    attr_audit["selected_for_02g"] = attr_audit["name"].isin(selected)
    attr_audit.to_csv(ATTR_AUDIT, sep="\t", index=False)

    query = build_biomart_query(selected)
    text = http_post_text(
        session,
        ENSEMBL_MART,
        data={"query": query},
        timeout=300,
    )

    if not text.strip():
        raise RuntimeError("Current Ensembl BioMart returned an empty mapping.")
    if text.lstrip().lower().startswith("query error"):
        raise RuntimeError(f"BioMart query error: {text[:1000]}")

    frame = pd.read_csv(io.StringIO(text), sep="\t", dtype=str, low_memory=False).fillna("")
    if frame.empty:
        raise RuntimeError("Current Ensembl BioMart mapping parsed to zero rows.")
    if frame.shape[1] != len(selected):
        raise RuntimeError(
            f"BioMart returned {frame.shape[1]} columns for {len(selected)} selected attributes."
        )
    frame.columns = selected

    frame.to_csv(SNAPSHOT_RAW, sep="\t", index=False)
    raw_hash = sha256_file(SNAPSHOT_RAW)

    meta = {
        "status": "PASS_FROZEN_CURRENT_ENSEMBL_SNAPSHOT",
        "created_utc": now_utc(),
        "ensembl_release": release,
        "rest_endpoint": ENSEMBL_REST,
        "biomart_endpoint": ENSEMBL_MART,
        "biomart_dataset": DOG_DATASET,
        "selected_attributes": selected,
        "required_attributes": REQUIRED_CURRENT_ATTRIBUTES,
        "optional_attributes_available_and_selected": [
            x for x in OPTIONAL_CURRENT_ATTRIBUTES if x in available
        ],
        "dog_assembly": {
            key: dog_assembly.get(key)
            for key in [
                "query_alias",
                "assembly_name",
                "assembly_date",
                "assembly_accession",
                "genebuild_last_geneset_update",
            ]
        },
        "human_assembly": {
            key: human_assembly.get(key)
            for key in [
                "query_alias",
                "assembly_name",
                "assembly_date",
                "assembly_accession",
                "genebuild_last_geneset_update",
            ]
        },
        "biomart_query_xml": query,
        "raw_rows": int(frame.shape[0]),
        "raw_columns": int(frame.shape[1]),
        "raw_sha256": raw_hash,
    }
    write_json(SNAPSHOT_META, meta)
    return frame, meta, False


def get_asset(
    lock: Dict[str, Any],
    role: str,
    paper4_root: Path,
) -> Tuple[Path, Dict[str, Any]]:
    assets = lock.get("assets") or {}
    item = assets.get(role)
    if not isinstance(item, dict):
        raise RuntimeError(f"Upstream lock missing asset role {role!r}.")

    rel = clean(item.get("relative_path"))
    expected_hash = clean(item.get("sha256")).lower()
    path = paper4_root / rel
    require_file(path)

    if len(expected_hash) != 64:
        raise RuntimeError(f"Upstream asset {role} lacks a valid SHA256.")
    actual = sha256_file(path).lower()
    if actual != expected_hash:
        raise RuntimeError(
            f"Upstream asset hash mismatch for {role}: expected {expected_hash}, observed {actual}"
        )
    return path, item


def expression_header(path: Path) -> List[str]:
    header = pd.read_csv(path, nrows=0, index_col=0)
    cols = [clean(x) for x in header.columns]

    if any(not x for x in cols):
        raise RuntimeError(f"{path.name}: empty expression feature name in header.")
    if len(cols) != len(set(cols)):
        duplicates = pd.Series(cols)[pd.Series(cols).duplicated()].unique().tolist()
        raise RuntimeError(
            f"{path.name}: duplicate expression columns are not allowed: {duplicates[:10]}"
        )
    return cols


def standardize_mapping(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy().fillna("")
    for col in REQUIRED_CURRENT_ATTRIBUTES:
        if col not in out.columns:
            raise RuntimeError(f"Current mapping missing required column {col}")

    rename = {
        "ensembl_gene_id": "dog_ensembl_gene_id",
        "external_gene_name": "dog_gene_symbol",
        "hsapiens_homolog_ensembl_gene": "human_ensembl_gene_id",
        "hsapiens_homolog_associated_gene_name": "human_gene_symbol",
        "hsapiens_homolog_orthology_type": "orthology_type",
        "hsapiens_homolog_orthology_confidence": "orthology_confidence",
        "hsapiens_homolog_perc_id": "human_homolog_perc_id",
        "hsapiens_homolog_perc_id_r1": "dog_homolog_perc_id",
        "hsapiens_homolog_goc_score": "goc_score",
        "hsapiens_homolog_wga_coverage": "wga_coverage",
    }
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})

    for col in [
        "dog_ensembl_gene_id",
        "dog_gene_symbol",
        "human_ensembl_gene_id",
        "human_gene_symbol",
        "orthology_type",
    ]:
        out[col] = out[col].astype(str).str.strip()

    if "orthology_confidence" not in out.columns:
        out["orthology_confidence"] = ""

    for col in [
        "human_homolog_perc_id",
        "dog_homolog_perc_id",
        "goc_score",
        "wga_coverage",
    ]:
        if col not in out.columns:
            out[col] = ""

    return out.drop_duplicates().reset_index(drop=True)


def resolve_dog2_features(
    raw_features: Sequence[str],
    current_dog_symbols: set[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    suffix_re = re.compile(r"^(.+)_([2-9][0-9]*)$")

    for raw in raw_features:
        if raw in current_dog_symbols:
            resolved = raw
            status = "EXACT_CURRENT_ENSEMBL_SYMBOL"
            suffix_n = ""
        else:
            match = suffix_re.match(raw)
            if match and match.group(1) in current_dog_symbols:
                resolved = match.group(1)
                status = "DUPLICATE_SUFFIX_TO_CURRENT_ENSEMBL_BASE"
                suffix_n = match.group(2)
            else:
                resolved = ""
                status = "UNRESOLVED_CURRENT_ENSEMBL_SYMBOL"
                suffix_n = ""

        rows.append(
            {
                "dog2_raw_feature": raw,
                "resolved_dog_symbol": resolved,
                "resolution_status": status,
                "duplicate_suffix_n": suffix_n,
            }
        )

    frame = pd.DataFrame(rows)
    counts = (
        frame.loc[frame["resolved_dog_symbol"].ne(""), "resolved_dog_symbol"]
        .value_counts()
        .to_dict()
    )
    frame["dog2_raw_features_per_resolved_symbol"] = frame["resolved_dog_symbol"].map(
        lambda x: int(counts.get(x, 0)) if x else 0
    )
    frame["feature_symbol_collision"] = (
        frame["dog2_raw_features_per_resolved_symbol"] > 1
    )
    frame["primary_symbol_resolution_eligible"] = (
        frame["resolved_dog_symbol"].ne("") & ~frame["feature_symbol_collision"]
    )
    return frame


def current_one2one_audit(
    current: pd.DataFrame,
    feature_resolution: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    mapped = current.copy()
    mapped["orthology_type_norm"] = mapped["orthology_type"].str.lower()

    one = mapped[
        mapped["dog_gene_symbol"].ne("")
        & mapped["human_gene_symbol"].ne("")
        & mapped["dog_ensembl_gene_id"].ne("")
        & mapped["human_ensembl_gene_id"].ne("")
        & mapped["orthology_type_norm"].str.contains("one2one", regex=False)
    ].copy()

    confidence_available = one["orthology_confidence"].astype(str).str.strip().ne("").any()
    if confidence_available:
        conf_num = pd.to_numeric(one["orthology_confidence"], errors="coerce")
        one["confidence_pass"] = conf_num.eq(1.0)
    else:
        one["confidence_pass"] = True

    dog_group = one.groupby("dog_gene_symbol", dropna=False)
    one["dog_symbol_n_dog_ensembl"] = one["dog_gene_symbol"].map(
        dog_group["dog_ensembl_gene_id"].nunique()
    )
    one["dog_symbol_n_human_ensembl"] = one["dog_gene_symbol"].map(
        dog_group["human_ensembl_gene_id"].nunique()
    )
    one["dog_symbol_n_human_symbols"] = one["dog_gene_symbol"].map(
        dog_group["human_gene_symbol"].nunique()
    )

    human_group = one.groupby("human_gene_symbol", dropna=False)
    one["human_symbol_n_dog_symbols"] = one["human_gene_symbol"].map(
        human_group["dog_gene_symbol"].nunique()
    )
    one["human_symbol_n_human_ensembl"] = one["human_gene_symbol"].map(
        human_group["human_ensembl_gene_id"].nunique()
    )

    one["mapping_unique"] = (
        one["dog_symbol_n_dog_ensembl"].eq(1)
        & one["dog_symbol_n_human_ensembl"].eq(1)
        & one["dog_symbol_n_human_symbols"].eq(1)
        & one["human_symbol_n_dog_symbols"].eq(1)
        & one["human_symbol_n_human_ensembl"].eq(1)
    )

    one = one.sort_values(
        [
            "dog_gene_symbol",
            "human_gene_symbol",
            "dog_ensembl_gene_id",
            "human_ensembl_gene_id",
        ]
    ).reset_index(drop=True)

    feature_ok = feature_resolution[
        feature_resolution["primary_symbol_resolution_eligible"]
    ][["dog2_raw_feature", "resolved_dog_symbol", "resolution_status"]].copy()
    feature_ok = feature_ok.rename(columns={"resolved_dog_symbol": "dog_gene_symbol"})

    bridge = feature_ok.merge(
        one,
        on="dog_gene_symbol",
        how="left",
        validate="one_to_many",
    )

    bridge["current_one2one_present"] = bridge["human_gene_symbol"].fillna("").ne("")
    bridge["primary_mapping_eligible"] = (
        bridge["current_one2one_present"]
        & bridge["mapping_unique"].fillna(False)
        & bridge["confidence_pass"].fillna(False)
    )

    def reason(row: pd.Series) -> str:
        if not bool(row["current_one2one_present"]):
            return "NO_CURRENT_ONE2ONE_MAPPING"
        if not bool(row.get("mapping_unique", False)):
            return "AMBIGUOUS_CURRENT_ONE2ONE_SYMBOL_OR_STABLE_ID_MAPPING"
        if not bool(row.get("confidence_pass", False)):
            return "ORTHOLOGY_CONFIDENCE_NOT_ONE"
        return "PRIMARY_MAPPING_ELIGIBLE"

    bridge["mapping_decision"] = bridge.apply(reason, axis=1)

    primary = bridge[bridge["primary_mapping_eligible"]].copy()

    if primary["dog_gene_symbol"].duplicated().any():
        raise RuntimeError("Primary bridge unexpectedly contains duplicate dog symbols.")
    if primary["human_gene_symbol"].duplicated().any():
        raise RuntimeError("Primary bridge unexpectedly contains duplicate human symbols.")
    if primary["dog2_raw_feature"].duplicated().any():
        raise RuntimeError("Primary bridge unexpectedly contains duplicate DOG2 raw features.")

    return one, primary


def detect_historical_columns(frame: pd.DataFrame) -> Tuple[str, str, str]:
    dog_candidates = ["external_gene_name", "dog_gene_symbol", "canine_gene_symbol"]
    human_candidates = [
        "hsapiens_homolog_associated_gene_name",
        "human_gene_symbol",
    ]
    type_candidates = [
        "hsapiens_homolog_orthology_type",
        "dog_human_orthology_type",
        "orthology_type",
    ]

    dog_col = next((x for x in dog_candidates if x in frame.columns), None)
    human_col = next((x for x in human_candidates if x in frame.columns), None)
    type_col = next((x for x in type_candidates if x in frame.columns), None)

    if not all([dog_col, human_col, type_col]):
        raise RuntimeError(
            "Historical BioMart snapshot lacks required mapping columns. "
            f"Observed: {list(frame.columns)}"
        )
    return str(dog_col), str(human_col), str(type_col)


def historical_sensitivity(
    hist: pd.DataFrame,
    current_primary: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    dog_col, human_col, type_col = detect_historical_columns(hist)

    work = hist[[dog_col, human_col, type_col]].copy().fillna("")
    work.columns = ["dog_gene_symbol", "human_gene_symbol", "orthology_type"]

    for col in work.columns:
        work[col] = work[col].astype(str).str.strip()

    hist_one = work[
        work["dog_gene_symbol"].ne("")
        & work["human_gene_symbol"].ne("")
        & work["orthology_type"].str.lower().str.contains("one2one", regex=False)
    ].copy()

    dog_counts = hist_one.groupby("dog_gene_symbol")["human_gene_symbol"].nunique()
    human_counts = hist_one.groupby("human_gene_symbol")["dog_gene_symbol"].nunique()

    hist_one["dog_n_human_symbols"] = hist_one["dog_gene_symbol"].map(dog_counts)
    hist_one["human_n_dog_symbols"] = hist_one["human_gene_symbol"].map(human_counts)

    hist_unique = hist_one[
        hist_one["dog_n_human_symbols"].eq(1)
        & hist_one["human_n_dog_symbols"].eq(1)
    ][["dog_gene_symbol", "human_gene_symbol"]].drop_duplicates()

    current_pairs = current_primary[
        ["dog_gene_symbol", "human_gene_symbol"]
    ].drop_duplicates()

    current_map = dict(
        zip(current_pairs["dog_gene_symbol"], current_pairs["human_gene_symbol"])
    )
    hist_map = dict(
        zip(hist_unique["dog_gene_symbol"], hist_unique["human_gene_symbol"])
    )

    all_dogs = sorted(set(current_map) | set(hist_map))
    rows = []

    for dog in all_dogs:
        cur = current_map.get(dog, "")
        old = hist_map.get(dog, "")

        if cur and old and cur == old:
            status = "CURRENT_AND_HISTORICAL_SAME_HUMAN_SYMBOL"
        elif cur and old and cur != old:
            status = "CURRENT_AND_HISTORICAL_DIFFERENT_HUMAN_SYMBOL"
        elif cur:
            status = "CURRENT_ONLY"
        else:
            status = "HISTORICAL_ONLY"

        rows.append(
            {
                "dog_gene_symbol": dog,
                "current_human_gene_symbol": cur,
                "historical_human_gene_symbol": old,
                "mapping_drift_status": status,
            }
        )

    audit = pd.DataFrame(rows)
    current_dogs = set(current_map)
    hist_dogs = set(hist_map)
    intersection = current_dogs & hist_dogs
    union = current_dogs | hist_dogs
    same = sum(current_map[x] == hist_map[x] for x in intersection)
    changed = len(intersection) - same

    metrics = {
        "historical_one2one_rows": int(hist_one.shape[0]),
        "historical_unique_symbol_pairs": int(hist_unique.shape[0]),
        "current_primary_unique_symbol_pairs": int(current_pairs.shape[0]),
        "dog_symbol_intersection": len(intersection),
        "dog_symbol_union": len(union),
        "dog_symbol_jaccard": (len(intersection) / len(union)) if union else None,
        "same_human_partner_within_intersection": same,
        "changed_human_partner_within_intersection": changed,
    }

    return audit, metrics


def main() -> None:
    started = now_utc()

    print("=" * 120)
    print("Paper 6 - build transcriptome-wide outcome-blind dog->human ortholog bridge")
    print("=" * 120)
    print(f"Script version: {SCRIPT_VERSION}")
    print()
    print("Safety / execution contract:")
    print("  Paper4 candidate/evidence tables used for row selection: NO")
    print("  DOG2 expression values parsed: NO [header only; file hash verified]")
    print("  Human expression values parsed: NO [headers only; file hashes verified]")
    print("  Clinical values read: NO")
    print("  Outcome/response/follow-up values read: NO")
    print("  Treatment-administration values read: NO")
    print("  Model fitting: NO")
    print("  Network access: YES only for fresh Ensembl release/assembly/BioMart mapping if no frozen 02g snapshot exists")
    print("  GPU execution: NO")
    print()

    for path in [AMENDMENT, AMENDMENT_SUMMARY, UPSTREAM_LOCK]:
        require_file(path)

    amendment = read_json(AMENDMENT)
    amendment_summary = read_json(AMENDMENT_SUMMARY)
    upstream = read_json(UPSTREAM_LOCK)

    if clean(amendment.get("status")) != "PASS":
        raise RuntimeError("02f0a amendment is not PASS.")
    if clean(amendment_summary.get("status")) != "PASS":
        raise RuntimeError("02f0a amendment summary is not PASS.")
    if sha256_file(AMENDMENT) != clean(
        (amendment_summary.get("final_artifact_hashes") or {}).get(
            "transport_contract_amendment_json"
        )
    ):
        raise RuntimeError("02f0a amendment hash verification failed.")

    effective = clean(amendment.get("old_02f0_effective_status"))
    if effective != "PARTIALLY_SUPERSEDED_PRE_OUTCOME":
        raise RuntimeError(f"Unexpected 02f0a effective status: {effective!r}")

    forbidden = amendment.get("forbidden_until_revised_contract") or []
    if not any("3,391" in str(x) for x in forbidden):
        raise RuntimeError("02f0a does not explicitly supersede the old strict universe.")

    paper4_root, paper4_source = resolve_paper4_root()
    print(f"Paper 4 root: {paper4_root}")
    print(f"Resolution source: {paper4_source}")

    dog2_path, dog2_asset = get_asset(upstream, "dog2_expression", paper4_root)
    target_path, target_asset = get_asset(upstream, "target_expression", paper4_root)
    gse21257_path, gse21257_asset = get_asset(
        upstream, "gse21257_expression", paper4_root
    )
    gse39055_path, gse39055_asset = get_asset(
        upstream, "gse39055_expression", paper4_root
    )

    hist_path = require_file(paper4_root / HIST_BIOMART_REL)

    print("Locked expression-header verification:")
    dog2_features = expression_header(dog2_path)
    target_features = expression_header(target_path)
    gse21257_features = expression_header(gse21257_path)
    gse39055_features = expression_header(gse39055_path)

    if len(dog2_features) != EXPECTED_DOG2_FEATURES:
        raise RuntimeError(
            f"DOG2 feature-count mismatch: expected {EXPECTED_DOG2_FEATURES}, observed {len(dog2_features)}"
        )

    print(f"  DOG2: {len(dog2_features):,}")
    print(f"  TARGET_OS: {len(target_features):,}")
    print(f"  GSE21257: {len(gse21257_features):,}")
    print(f"  GSE39055: {len(gse39055_features):,}")

    target_set = set(target_features)
    gse21257_set = set(gse21257_features)
    gse39055_set = set(gse39055_features)

    hist = pd.read_csv(hist_path, sep="\t", dtype=str, low_memory=False).fillna("")
    if int(hist.shape[0]) != EXPECTED_HIST_BIOMART_ROWS:
        raise RuntimeError(
            f"Historical BioMart row-count mismatch: expected {EXPECTED_HIST_BIOMART_ROWS}, observed {hist.shape[0]}"
        )

    hist_dog_col, hist_human_col, hist_type_col = detect_historical_columns(hist)
    hist_dog = hist[hist_dog_col].astype(str).str.strip()
    hist_human = hist[hist_human_col].astype(str).str.strip()
    hist_type = hist[hist_type_col].astype(str).str.strip().str.lower()

    hist_unique_dog = set(hist_dog[hist_dog.ne("")])
    hist_any_human = set(hist_dog[hist_dog.ne("") & hist_human.ne("")])
    hist_one2one = set(
        hist_dog[
            hist_dog.ne("")
            & hist_human.ne("")
            & hist_type.str.contains("one2one", regex=False)
        ]
    )

    if len(hist_unique_dog) != EXPECTED_HIST_UNIQUE_DOG_SYMBOLS:
        raise RuntimeError("Historical BioMart unique dog-symbol count changed.")
    if len(hist_any_human) != EXPECTED_HIST_ANY_HUMAN:
        raise RuntimeError("Historical BioMart any-human-homolog count changed.")
    if len(hist_one2one) != EXPECTED_HIST_ONE2ONE:
        raise RuntimeError("Historical BioMart one2one dog-symbol count changed.")

    print("Historical outcome-blind BioMart reference [SENSITIVITY ONLY]:")
    print(f"  raw rows: {hist.shape[0]:,}")
    print(f"  unique dog symbols: {len(hist_unique_dog):,}")
    print(f"  any human homolog: {len(hist_any_human):,}")
    print(f"  one2one dog symbols: {len(hist_one2one):,}")
    print(f"  SHA256: {sha256_file(hist_path)}")

    session = requests.Session()
    current_raw, current_meta, reused_snapshot = fetch_or_reuse_current_snapshot(
        session
    )

    current_release = current_meta.get("ensembl_release")
    dog_assembly = (current_meta.get("dog_assembly") or {}).get("assembly_name")
    human_assembly = (current_meta.get("human_assembly") or {}).get("assembly_name")

    print("Fresh Ensembl mapping snapshot:")
    print(
        "  snapshot source: "
        + (
            "REUSED_FROZEN_02G_SNAPSHOT"
            if reused_snapshot
            else "FETCHED_AND_FROZEN_THIS_RUN"
        )
    )
    print(f"  Ensembl release: {current_release}")
    print(f"  dog assembly: {dog_assembly}")
    print(f"  human assembly: {human_assembly}")
    print(f"  raw mapping rows: {current_raw.shape[0]:,}")
    print(f"  raw mapping SHA256: {sha256_file(SNAPSHOT_RAW)}")

    current = standardize_mapping(current_raw)
    current_symbols = set(
        current.loc[current["dog_gene_symbol"].ne(""), "dog_gene_symbol"]
    )

    feature_resolution = resolve_dog2_features(dog2_features, current_symbols)
    feature_resolution.to_csv(FEATURE_RESOLUTION, sep="\t", index=False)

    exact_n = int(
        feature_resolution["resolution_status"].eq(
            "EXACT_CURRENT_ENSEMBL_SYMBOL"
        ).sum()
    )
    suffix_n = int(
        feature_resolution["resolution_status"].eq(
            "DUPLICATE_SUFFIX_TO_CURRENT_ENSEMBL_BASE"
        ).sum()
    )
    unresolved_n = int(
        feature_resolution["resolution_status"].eq(
            "UNRESOLVED_CURRENT_ENSEMBL_SYMBOL"
        ).sum()
    )
    collision_feature_n = int(feature_resolution["feature_symbol_collision"].sum())
    collision_symbol_n = int(
        feature_resolution.loc[
            feature_resolution["feature_symbol_collision"],
            "resolved_dog_symbol",
        ].nunique()
    )
    symbol_eligible_n = int(
        feature_resolution["primary_symbol_resolution_eligible"].sum()
    )

    current_one, primary = current_one2one_audit(current, feature_resolution)
    current_one.to_csv(CURRENT_ONE2ONE, sep="\t", index=False)

    primary["present_TARGET_OS"] = primary["human_gene_symbol"].isin(target_set)
    primary["present_GSE21257"] = primary["human_gene_symbol"].isin(gse21257_set)
    primary["present_GSE39055"] = primary["human_gene_symbol"].isin(gse39055_set)
    primary["symbol_concordant_dog_human"] = (
        primary["dog_gene_symbol"] == primary["human_gene_symbol"]
    )

    primary["primary_dog2_to_target_os"] = primary["present_TARGET_OS"]
    primary["secondary_dog2_to_gse21257"] = primary["present_GSE21257"]
    primary["stress_dog2_to_gse39055"] = primary["present_GSE39055"]
    primary["common_dog2_target_gse21257"] = (
        primary["present_TARGET_OS"] & primary["present_GSE21257"]
    )
    primary["common_all_four"] = (
        primary["present_TARGET_OS"]
        & primary["present_GSE21257"]
        & primary["present_GSE39055"]
    )

    primary = primary.sort_values(
        ["dog_gene_symbol", "human_gene_symbol"]
    ).reset_index(drop=True)
    primary.to_csv(PRIMARY_BRIDGE, sep="\t", index=False)

    transport_cols = [
        "dog2_raw_feature",
        "dog_gene_symbol",
        "dog_ensembl_gene_id",
        "human_gene_symbol",
        "human_ensembl_gene_id",
        "orthology_type",
        "orthology_confidence",
        "human_homolog_perc_id",
        "dog_homolog_perc_id",
        "goc_score",
        "wga_coverage",
        "resolution_status",
        "symbol_concordant_dog_human",
        "present_TARGET_OS",
        "present_GSE21257",
        "present_GSE39055",
        "primary_dog2_to_target_os",
        "secondary_dog2_to_gse21257",
        "stress_dog2_to_gse39055",
        "common_dog2_target_gse21257",
        "common_all_four",
    ]
    primary[transport_cols].to_csv(TRANSPORT_FEATURES, sep="\t", index=False)

    current_one2one_rows = int(current_one.shape[0])
    current_one2one_dog_symbols = int(current_one["dog_gene_symbol"].nunique())
    current_one2one_human_symbols = int(current_one["human_gene_symbol"].nunique())

    current_unique_mapping_dog_symbols = int(
        current_one.loc[
            current_one["mapping_unique"] & current_one["confidence_pass"],
            "dog_gene_symbol",
        ].nunique()
    )

    primary_bridge_n = int(primary.shape[0])
    target_n = int(primary["primary_dog2_to_target_os"].sum())
    g21257_n = int(primary["secondary_dog2_to_gse21257"].sum())
    g39055_n = int(primary["stress_dog2_to_gse39055"].sum())
    common3_n = int(primary["common_dog2_target_gse21257"].sum())
    common4_n = int(primary["common_all_four"].sum())
    symbol_concordant_n = int(primary["symbol_concordant_dog_human"].sum())
    target_symbol_concordant_n = int(
        (
            primary["primary_dog2_to_target_os"]
            & primary["symbol_concordant_dog_human"]
        ).sum()
    )

    funnel_rows = [
        {
            "stage_order": 1,
            "stage": "DOG2_raw_expression_features",
            "count": len(dog2_features),
            "scope": "all locked DOG2 expression columns",
        },
        {
            "stage_order": 2,
            "stage": "exact_current_Ensembl_symbol_resolution",
            "count": exact_n,
            "scope": "exact dog symbol match",
        },
        {
            "stage_order": 3,
            "stage": "duplicate_suffix_resolution",
            "count": suffix_n,
            "scope": "raw feature _N -> current-Ensembl base, only when exact raw symbol absent",
        },
        {
            "stage_order": 4,
            "stage": "unresolved_current_Ensembl_symbols",
            "count": unresolved_n,
            "scope": "excluded from primary",
        },
        {
            "stage_order": 5,
            "stage": "features_in_symbol_collision_groups",
            "count": collision_feature_n,
            "scope": f"{collision_symbol_n} resolved symbols; all excluded from primary",
        },
        {
            "stage_order": 6,
            "stage": "primary_symbol_resolution_eligible_features",
            "count": symbol_eligible_n,
            "scope": "one raw DOG2 feature per resolved canine symbol",
        },
        {
            "stage_order": 7,
            "stage": "current_Ensembl_one2one_dog_symbols",
            "count": current_one2one_dog_symbols,
            "scope": "all current one-to-one mapping symbols before expression-feature join",
        },
        {
            "stage_order": 8,
            "stage": "current_unique_confidence_pass_one2one_dog_symbols",
            "count": current_unique_mapping_dog_symbols,
            "scope": "unique stable-ID/symbol mapping; confidence=1 when available",
        },
        {
            "stage_order": 9,
            "stage": "DOG2_primary_outcome_blind_bridge_pairs",
            "count": primary_bridge_n,
            "scope": "DOG2-resolved + unique current 1:1 mapping",
        },
        {
            "stage_order": 10,
            "stage": "primary_DOG2_to_TARGET_OS",
            "count": target_n,
            "scope": "human symbol present in TARGET expression header",
        },
        {
            "stage_order": 11,
            "stage": "secondary_DOG2_to_GSE21257",
            "count": g21257_n,
            "scope": "human symbol present in GSE21257 expression header",
        },
        {
            "stage_order": 12,
            "stage": "stress_DOG2_to_GSE39055",
            "count": g39055_n,
            "scope": "human symbol present in GSE39055 expression header",
        },
        {
            "stage_order": 13,
            "stage": "common_DOG2_TARGET_GSE21257",
            "count": common3_n,
            "scope": "DOG2 bridge + both primary/secondary human expression headers",
        },
        {
            "stage_order": 14,
            "stage": "common_all_four",
            "count": common4_n,
            "scope": "DOG2 bridge + TARGET + GSE21257 + GSE39055",
        },
    ]
    pd.DataFrame(funnel_rows).to_csv(FUNNEL_TSV, sep="\t", index=False)

    feature_summary_rows = [
        {
            "feature_set": "primary_outcome_blind_bridge_all_human_mapped",
            "role": "BRIDGE",
            "n_features": primary_bridge_n,
        },
        {
            "feature_set": "primary_dog2_to_target_os",
            "role": "PRIMARY",
            "n_features": target_n,
        },
        {
            "feature_set": "secondary_dog2_to_gse21257",
            "role": "SECONDARY",
            "n_features": g21257_n,
        },
        {
            "feature_set": "stress_dog2_to_gse39055",
            "role": "STRESS_TEST",
            "n_features": g39055_n,
        },
        {
            "feature_set": "common_dog2_target_gse21257",
            "role": "SENSITIVITY",
            "n_features": common3_n,
        },
        {
            "feature_set": "common_all_four",
            "role": "SENSITIVITY",
            "n_features": common4_n,
        },
        {
            "feature_set": "symbol_concordant_current_one2one_bridge",
            "role": "MAPPING_SENSITIVITY",
            "n_features": symbol_concordant_n,
        },
        {
            "feature_set": "symbol_concordant_primary_dog2_to_target_os",
            "role": "MAPPING_SENSITIVITY",
            "n_features": target_symbol_concordant_n,
        },
    ]
    pd.DataFrame(feature_summary_rows).to_csv(
        FEATURE_SUMMARY, sep="\t", index=False
    )

    hist_audit, hist_metrics = historical_sensitivity(hist, primary)
    hist_audit.to_csv(HIST_SENSITIVITY, sep="\t", index=False)

    if primary_bridge_n <= 3391:
        scientific_status = (
            "HOLD_NEW_BRIDGE_NOT_LARGER_THAN_SUPERSEDED_STRICT_UNIVERSE"
        )
    elif target_n <= 3386:
        scientific_status = (
            "HOLD_NEW_TARGET_BRIDGE_NOT_LARGER_THAN_SUPERSEDED_PRIMARY_SET"
        )
    else:
        scientific_status = (
            "PASS_TRANSCRIPTOMEWIDE_OUTCOME_BLIND_BRIDGE_READY_FOR_MODULE_COVERAGE"
        )

    confidence_present = (
        current_one["orthology_confidence"].astype(str).str.strip().ne("").any()
    )

    contract = {
        "script_version": SCRIPT_VERSION,
        "status": "PASS" if scientific_status.startswith("PASS_") else "HOLD",
        "created_utc": now_utc(),
        "scientific_status": scientific_status,
        "upstream_amendment": {
            "amendment_id": amendment.get("amendment_id"),
            "amendment_sha256": sha256_file(AMENDMENT),
            "old_02f0_effective_status": amendment.get(
                "old_02f0_effective_status"
            ),
        },
        "safety": {
            "paper4_candidate_evidence_tables_used_for_row_selection": False,
            "clinical_values_read": False,
            "outcome_response_followup_values_read": False,
            "treatment_administration_values_read": False,
            "expression_values_parsed": False,
            "expression_headers_parsed": True,
            "model_fitting": False,
            "network_access_for_fresh_ensembl_snapshot": not reused_snapshot,
        },
        "fresh_primary_mapping_snapshot": {
            "ensembl_release": current_release,
            "dog_assembly": current_meta.get("dog_assembly"),
            "human_assembly": current_meta.get("human_assembly"),
            "biomart_dataset": DOG_DATASET,
            "selected_attributes": current_meta.get("selected_attributes"),
            "raw_rows": int(current_raw.shape[0]),
            "raw_sha256": sha256_file(SNAPSHOT_RAW),
            "snapshot_metadata_sha256": sha256_file(SNAPSHOT_META),
            "snapshot_reused": reused_snapshot,
        },
        "historical_mapping_sensitivity": {
            "role": "SENSITIVITY_ONLY_NOT_PRIMARY_ROW_SELECTION",
            "historical_cache_relative_path": str(HIST_BIOMART_REL).replace(
                "\\", "/"
            ),
            "historical_cache_sha256": sha256_file(hist_path),
            "confirmed_preflight_counts": {
                "raw_rows": int(hist.shape[0]),
                "unique_dog_symbols": len(hist_unique_dog),
                "dog_symbols_with_any_human_homolog": len(hist_any_human),
                "dog_symbols_with_one2one_human_ortholog": len(hist_one2one),
            },
            "mapping_drift_metrics": hist_metrics,
        },
        "dog2_symbol_resolution_rule": {
            "exact_match_priority": True,
            "terminal_duplicate_suffix_pattern": r"^(.+)_([2-9][0-9]*)$",
            "suffix_stripping_allowed_only_if_exact_raw_symbol_absent": True,
            "suffix_stripping_allowed_only_if_base_is_current_ensembl_symbol": True,
            "resolved_symbol_collision_primary_rule": "EXCLUDE_ALL_COLLIDING_RAW_FEATURES",
            "outcome_information_used": False,
        },
        "primary_mapping_rule": {
            "orthology": "CURRENT_ENSEMBL_ONE_TO_ONE",
            "require_nonempty_dog_symbol": True,
            "require_nonempty_human_symbol": True,
            "require_unique_dog_ensembl_gene_per_dog_symbol": True,
            "require_unique_human_ensembl_gene_per_dog_symbol": True,
            "require_unique_human_symbol_per_dog_symbol": True,
            "require_unique_dog_symbol_per_human_symbol": True,
            "require_unique_human_ensembl_gene_per_human_symbol": True,
            "orthology_confidence": (
                "REQUIRE_1"
                if confidence_present
                else "ATTRIBUTE_NOT_AVAILABLE_NO_CONFIDENCE_CLAIM"
            ),
            "protein_identity_used_as_filter": False,
            "dog_human_symbol_equality_required": False,
            "symbol_concordant_subset_role": "SENSITIVITY_ONLY",
            "outcome_information_used": False,
        },
        "measured_counts": {
            "DOG2_raw_features": len(dog2_features),
            "exact_symbol_resolution": exact_n,
            "duplicate_suffix_resolution": suffix_n,
            "unresolved_symbols": unresolved_n,
            "features_in_symbol_collision_groups": collision_feature_n,
            "symbol_collision_groups": collision_symbol_n,
            "primary_symbol_resolution_eligible_features": symbol_eligible_n,
            "current_one2one_rows": current_one2one_rows,
            "current_one2one_dog_symbols": current_one2one_dog_symbols,
            "current_one2one_human_symbols": current_one2one_human_symbols,
            "current_unique_confidence_pass_one2one_dog_symbols": current_unique_mapping_dog_symbols,
            "primary_outcome_blind_bridge_pairs": primary_bridge_n,
            "primary_dog2_to_target_os": target_n,
            "secondary_dog2_to_gse21257": g21257_n,
            "stress_dog2_to_gse39055": g39055_n,
            "common_dog2_target_gse21257": common3_n,
            "common_all_four": common4_n,
            "symbol_concordant_bridge": symbol_concordant_n,
            "symbol_concordant_primary_dog2_to_target_os": target_symbol_concordant_n,
        },
        "locked_expression_assets": {
            "dog2_expression_sha256": clean(dog2_asset.get("sha256")),
            "target_expression_sha256": clean(target_asset.get("sha256")),
            "gse21257_expression_sha256": clean(
                gse21257_asset.get("sha256")
            ),
            "gse39055_expression_sha256": clean(
                gse39055_asset.get("sha256")
            ),
        },
        "feature_set_gene_hashes": {
            "primary_bridge_dog_raw_features_sha256_sorted": sha256_lines(
                primary["dog2_raw_feature"].tolist()
            ),
            "primary_bridge_dog_symbols_sha256_sorted": sha256_lines(
                primary["dog_gene_symbol"].tolist()
            ),
            "primary_bridge_human_symbols_sha256_sorted": sha256_lines(
                primary["human_gene_symbol"].tolist()
            ),
            "target_dog_symbols_sha256_sorted": sha256_lines(
                primary.loc[
                    primary["primary_dog2_to_target_os"], "dog_gene_symbol"
                ].tolist()
            ),
            "target_human_symbols_sha256_sorted": sha256_lines(
                primary.loc[
                    primary["primary_dog2_to_target_os"], "human_gene_symbol"
                ].tolist()
            ),
        },
        "required_next": (
            "02h module/compartment coverage audit on this new bridge; "
            "do not finalize selective-borrowing architecture before 02h."
        ),
    }
    write_json(CONTRACT_JSON, contract)

    final_hashes = {
        "ensembl_current_dog_human_biomart_raw_tsv": sha256_file(SNAPSHOT_RAW),
        "ensembl_current_snapshot_metadata_json": sha256_file(SNAPSHOT_META),
        "dog2_feature_symbol_resolution_tsv": sha256_file(FEATURE_RESOLUTION),
        "current_one2one_mapping_audit_tsv": sha256_file(CURRENT_ONE2ONE),
        "primary_outcome_blind_ortholog_bridge_tsv": sha256_file(PRIMARY_BRIDGE),
        "transport_feature_sets_tsv": sha256_file(TRANSPORT_FEATURES),
        "mapping_funnel_tsv": sha256_file(FUNNEL_TSV),
        "feature_set_summary_tsv": sha256_file(FEATURE_SUMMARY),
        "historical_biomart_mapping_sensitivity_tsv": sha256_file(
            HIST_SENSITIVITY
        ),
        "ortholog_bridge_contract_json": sha256_file(CONTRACT_JSON),
    }

    summary = {
        "script_version": SCRIPT_VERSION,
        "status": contract["status"],
        "scientific_status": scientific_status,
        "run_started_utc": started,
        "run_finished_utc": now_utc(),
        "ensembl_release": current_release,
        "dog_assembly": dog_assembly,
        "human_assembly": human_assembly,
        "snapshot_reused": reused_snapshot,
        "measured_counts": contract["measured_counts"],
        "historical_mapping_drift_metrics": hist_metrics,
        "final_artifact_hashes": final_hashes,
        "paper4_candidate_evidence_tables_used_for_row_selection": False,
        "outcome_response_followup_values_read": False,
        "clinical_values_read": False,
        "treatment_administration_values_read": False,
        "expression_values_parsed": False,
        "model_fitting": False,
    }
    write_json(SUMMARY_JSON, summary)

    print("-" * 120)
    print("DOG2 feature-symbol resolution")
    print("-" * 120)
    print(f"raw DOG2 features: {len(dog2_features):,}")
    print(f"exact current-Ensembl symbol matches: {exact_n:,}")
    print(f"duplicate-suffix -> current-Ensembl base: {suffix_n:,}")
    print(f"unresolved current-Ensembl symbols: {unresolved_n:,}")
    print(f"features in resolved-symbol collision groups: {collision_feature_n:,}")
    print(f"resolved-symbol collision groups: {collision_symbol_n:,}")
    print(f"primary symbol-resolution eligible features: {symbol_eligible_n:,}")
    print()

    print("Current Ensembl one-to-one mapping:")
    print(f"one-to-one mapping rows: {current_one2one_rows:,}")
    print(f"one-to-one dog symbols: {current_one2one_dog_symbols:,}")
    print(f"one-to-one human symbols: {current_one2one_human_symbols:,}")
    print(f"unique/confidence-pass dog symbols: {current_unique_mapping_dog_symbols:,}")
    print(f"DOG2 primary outcome-blind bridge pairs: {primary_bridge_n:,}")
    print()

    print("Frozen transport feature universes:")
    print(f"primary_dog2_to_target_os: {target_n:,}")
    print(f"secondary_dog2_to_gse21257: {g21257_n:,}")
    print(f"stress_dog2_to_gse39055: {g39055_n:,}")
    print(f"common_dog2_target_gse21257: {common3_n:,}")
    print(f"common_all_four: {common4_n:,}")
    print(f"symbol-concordant primary sensitivity: {target_symbol_concordant_n:,}")
    print()

    print("Historical BioMart drift sensitivity:")
    print(
        f"current primary unique pairs: "
        f"{hist_metrics['current_primary_unique_symbol_pairs']:,}"
    )
    print(
        f"historical unique one2one symbol pairs: "
        f"{hist_metrics['historical_unique_symbol_pairs']:,}"
    )
    print(f"dog-symbol intersection: {hist_metrics['dog_symbol_intersection']:,}")
    print(
        f"same human partner: "
        f"{hist_metrics['same_human_partner_within_intersection']:,}"
    )
    print(
        f"changed human partner: "
        f"{hist_metrics['changed_human_partner_within_intersection']:,}"
    )
    print(f"dog-symbol Jaccard: {hist_metrics['dog_symbol_jaccard']:.4f}")
    print()

    print("=" * 120)
    print("02g TRANSCRIPTOME-WIDE OUTCOME-BLIND ORTHOLOG BRIDGE SUMMARY")
    print("=" * 120)
    print(f"Ensembl release: {current_release}")
    print(f"dog assembly: {dog_assembly}")
    print(f"human assembly: {human_assembly}")
    print(f"scientific status: {scientific_status}")
    print()
    print("Paper4 candidate/evidence tables used for row selection: NO")
    print("Clinical values read: NO")
    print("Outcome/response/follow-up values read: NO")
    print("Treatment-administration values read: NO")
    print("Expression values parsed: NO")
    print("Model fitting: NO")
    print()
    print("Artifacts:")

    paths = [
        SNAPSHOT_RAW,
        SNAPSHOT_META,
        ATTR_AUDIT if ATTR_AUDIT.exists() else None,
        FEATURE_RESOLUTION,
        CURRENT_ONE2ONE,
        PRIMARY_BRIDGE,
        TRANSPORT_FEATURES,
        FUNNEL_TSV,
        FEATURE_SUMMARY,
        HIST_SENSITIVITY,
        CONTRACT_JSON,
        SUMMARY_JSON,
    ]
    for path in paths:
        if path is not None and path.exists():
            print(f"  {path.relative_to(ROOT)}")

    print()
    if contract["status"] == "PASS":
        print("Next: run 02h module/compartment coverage audit on the new bridge.")
        print("=" * 120)
        print("02g transcriptome-wide outcome-blind ortholog bridge: PASS")
        print("=" * 120)
    else:
        print(
            "Next: HOLD. Review the new bridge before module coverage or any outcome model."
        )
        print("=" * 120)
        print("02g transcriptome-wide outcome-blind ortholog bridge: HOLD")
        print("=" * 120)
        raise RuntimeError(scientific_status)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 120, file=sys.stderr)
        print(
            "02g transcriptome-wide outcome-blind ortholog bridge: FAIL",
            file=sys.stderr,
        )
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print("=" * 120, file=sys.stderr)
        raise
