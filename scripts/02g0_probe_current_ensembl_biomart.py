#!/usr/bin/env python3
"""
Paper 6 - probe current Ensembl BioMart route/dataset/attribute surface for 02g.

Purpose
-------
The full 02g v1/v2 failed before scientific mapping because the BioMart metadata
discovery route did not expose the expected dog dataset/attributes as parsed.

This diagnostic:
1. probes multiple official Ensembl BioMart entry points;
2. reads the BioMart registry;
3. follows the registry-advertised Ensembl Genes mart host/path;
4. inventories datasets;
5. finds dog/Canis lupus familiaris candidates robustly;
6. inventories attributes for each dog candidate;
7. reports exact or alias-like dog-human homology attributes.

It retrieves metadata only. It does NOT retrieve orthology mapping rows.

Safety
------
- Network access: YES, Ensembl metadata only
- BioMart mapping rows read: NO
- Expression values read: NO
- Clinical values read: NO
- Outcome/response/follow-up values read: NO
- Treatment values read: NO
- Model fitting: NO

No CLI arguments.
"""

from __future__ import annotations

import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import pandas as pd
import requests


SCRIPT_VERSION = "02g0-probe-current-ensembl-biomart-v1-no-cli"

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "ortholog_bridge" / "02g0_biomart_probe"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_JSON = OUT_DIR / "summary.json"
ROUTE_AUDIT = OUT_DIR / "route_audit.tsv"
MART_AUDIT = OUT_DIR / "mart_registry_audit.tsv"
DATASET_AUDIT = OUT_DIR / "dataset_audit.tsv"
ATTRIBUTE_AUDIT = OUT_DIR / "attribute_audit.tsv"

ENTRY_POINTS = [
    "https://mart.ensembl.org/biomart/martservice",
    "https://www.ensembl.org/biomart/martservice",
]

EXPECTED_MART = "ENSEMBL_MART_ENSEMBL"

REQUIRED_CONCEPTS = {
    "dog_ensembl_gene_id": [
        "ensembl_gene_id",
    ],
    "dog_gene_symbol": [
        "external_gene_name",
        "gene_name",
    ],
    "human_ensembl_gene_id": [
        "hsapiens_homolog_ensembl_gene",
    ],
    "human_gene_symbol": [
        "hsapiens_homolog_associated_gene_name",
        "hsapiens_homolog_gene_name",
    ],
    "orthology_type": [
        "hsapiens_homolog_orthology_type",
    ],
}

OPTIONAL_CONCEPTS = {
    "orthology_confidence": [
        "hsapiens_homolog_orthology_confidence",
    ],
    "human_perc_id": [
        "hsapiens_homolog_perc_id",
    ],
    "dog_perc_id": [
        "hsapiens_homolog_perc_id_r1",
    ],
    "goc_score": [
        "hsapiens_homolog_goc_score",
    ],
    "wga_coverage": [
        "hsapiens_homolog_wga_coverage",
    ],
}

DOG_PATTERNS = [
    re.compile(r"\bdog\b", re.I),
    re.compile(r"canis\s+lupus\s+familiaris", re.I),
    re.compile(r"familiaris", re.I),
    re.compile(r"ros_cfam", re.I),
]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def save_raw(name: str, text: str) -> Path:
    path = OUT_DIR / name
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def request_text(
    session: requests.Session,
    url: str,
    params: Dict[str, Any],
    *,
    timeout: int = 60,
    retries: int = 2,
) -> Tuple[str, str, int]:
    last_exc: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        try:
            response = session.get(
                url,
                params=params,
                headers={"User-Agent": "Paper6-Ensembl-BioMart-Probe/1.0"},
                timeout=timeout,
                allow_redirects=True,
            )
            response.raise_for_status()
            return response.text, response.url, response.status_code
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(attempt * 2)

    raise RuntimeError(
        f"Request failed after {retries} attempt(s): {url} params={params}: {last_exc}"
    )


def parse_registry(text: str) -> List[Dict[str, str]]:
    try:
        root = ET.fromstring(text)
    except Exception as exc:
        raise RuntimeError(f"Registry response is not valid XML: {exc}") from exc

    rows: List[Dict[str, str]] = []
    for elem in root.iter():
        if elem.tag.endswith("MartURLLocation"):
            rows.append(
                {
                    "name": clean(elem.attrib.get("name")),
                    "displayName": clean(elem.attrib.get("displayName")),
                    "host": clean(elem.attrib.get("host")),
                    "path": clean(elem.attrib.get("path")),
                    "port": clean(elem.attrib.get("port")),
                    "serverVirtualSchema": clean(
                        elem.attrib.get("serverVirtualSchema")
                    ),
                    "database": clean(elem.attrib.get("database")),
                    "visible": clean(elem.attrib.get("visible")),
                }
            )
    return rows


def registry_location_to_url(row: Dict[str, str]) -> str:
    host = clean(row.get("host"))
    path = clean(row.get("path")) or "/biomart/martservice"
    port = clean(row.get("port"))

    if not host:
        raise RuntimeError("Registry mart location has no host.")

    scheme = "https"
    if host.startswith("http://") or host.startswith("https://"):
        parsed = urlparse(host)
        scheme = parsed.scheme
        host = parsed.netloc

    if port and port not in {"80", "443"} and ":" not in host:
        host = f"{host}:{port}"

    return f"{scheme}://{host}{path}"


def parse_dataset_lines(text: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for i, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue

        parts = line.rstrip("\n").split("\t")
        gene_dataset_fields = [
            p.strip()
            for p in parts
            if re.search(r"_gene_ensembl$", p.strip(), re.I)
        ]

        full = " | ".join(parts)
        dog_like = any(pattern.search(full) for pattern in DOG_PATTERNS)

        rows.append(
            {
                "line_number": i,
                "n_fields": len(parts),
                "field_0": parts[0] if len(parts) > 0 else "",
                "field_1": parts[1] if len(parts) > 1 else "",
                "field_2": parts[2] if len(parts) > 2 else "",
                "field_3": parts[3] if len(parts) > 3 else "",
                "field_4": parts[4] if len(parts) > 4 else "",
                "gene_dataset_tokens": ";".join(gene_dataset_fields),
                "dog_like": dog_like,
                "raw": line,
            }
        )

    return pd.DataFrame(rows)


def dataset_candidates(frame: pd.DataFrame) -> List[str]:
    candidates: List[str] = []

    if frame.empty:
        return candidates

    dog_rows = frame[frame["dog_like"]].copy()

    # Prefer actual *_gene_ensembl tokens from dog-like lines.
    for value in dog_rows["gene_dataset_tokens"].astype(str):
        for token in value.split(";"):
            token = token.strip()
            if token and token not in candidates:
                candidates.append(token)

    # If no dog-like description survived parsing, search all dataset tokens for
    # familiaris/canis-like prefixes as a fallback.
    if not candidates:
        for value in frame["gene_dataset_tokens"].astype(str):
            for token in value.split(";"):
                token = token.strip()
                if not token:
                    continue
                if re.search(r"(familiaris|canis|dog)", token, re.I):
                    if token not in candidates:
                        candidates.append(token)

    return candidates


def parse_attribute_lines(text: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for i, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue

        parts = line.rstrip("\n").split("\t")
        name = parts[0].strip() if parts else ""
        description = parts[1].strip() if len(parts) > 1 else ""
        full = " | ".join(parts)

        rows.append(
            {
                "line_number": i,
                "name": name,
                "description": description,
                "n_fields": len(parts),
                "human_homology_like": bool(
                    re.search(
                        r"(hsapiens|human).*(homolog|ortholog)|(homolog|ortholog).*(hsapiens|human)",
                        full,
                        re.I,
                    )
                ),
                "core_gene_like": bool(
                    re.search(r"ensembl_gene|external_gene|gene_name", full, re.I)
                ),
                "raw": line,
            }
        )

    return pd.DataFrame(rows)


def concept_resolution(
    available: set[str],
    concepts: Dict[str, List[str]],
) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}

    for concept, aliases in concepts.items():
        exact = next((alias for alias in aliases if alias in available), "")
        result[concept] = {
            "aliases": aliases,
            "resolved": exact,
            "available": bool(exact),
        }

    return result


def main() -> None:
    print("=" * 118)
    print("Paper 6 - probe current Ensembl BioMart route/dataset/attribute surface")
    print("=" * 118)
    print(f"Script version: {SCRIPT_VERSION}")
    print()
    print("Safety contract:")
    print("  Network access: YES [Ensembl metadata only]")
    print("  Orthology mapping rows retrieved: NO")
    print("  Expression values read: NO")
    print("  Clinical/outcome/treatment values read: NO")
    print("  Model fitting: NO")
    print()

    session = requests.Session()

    route_rows: List[Dict[str, Any]] = []
    mart_rows: List[Dict[str, Any]] = []
    dataset_rows: List[Dict[str, Any]] = []
    attribute_rows: List[Dict[str, Any]] = []

    successful_mart_routes: List[Tuple[str, str, str]] = []

    # ------------------------------------------------------------------
    # 1. Registry probes.
    # ------------------------------------------------------------------
    print("Registry probes:")

    for idx, entry in enumerate(ENTRY_POINTS, start=1):
        row: Dict[str, Any] = {
            "entry_point": entry,
            "registry_status": "FAIL",
            "final_url": "",
            "ensembl_genes_mart_found": False,
            "registry_martservice_url": "",
            "error": "",
        }

        try:
            text, final_url, status_code = request_text(
                session,
                entry,
                {"type": "registry"},
                timeout=60,
            )
            raw_path = save_raw(f"registry_{idx}.xml", text)

            marts = parse_registry(text)
            genes_marts = [m for m in marts if m.get("name") == EXPECTED_MART]

            row["registry_status"] = "PASS"
            row["final_url"] = final_url
            row["http_status"] = status_code
            row["ensembl_genes_mart_found"] = bool(genes_marts)
            row["raw_artifact"] = str(raw_path.relative_to(ROOT))

            for mart in marts:
                mart_rows.append(
                    {
                        "entry_point": entry,
                        "registry_final_url": final_url,
                        **mart,
                    }
                )

            if genes_marts:
                mart = genes_marts[0]
                mart_url = registry_location_to_url(mart)
                schema = clean(mart.get("serverVirtualSchema")) or "default"

                row["registry_martservice_url"] = mart_url
                row["registry_virtual_schema"] = schema

                successful_mart_routes.append(
                    (entry, mart_url, schema)
                )

            print(
                f"  {entry}: PASS, final={final_url}, "
                f"Ensembl Genes mart={'YES' if genes_marts else 'NO'}"
            )

        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"  {entry}: FAIL - {exc}")

        route_rows.append(row)

    if not successful_mart_routes:
        pd.DataFrame(route_rows).to_csv(ROUTE_AUDIT, sep="\t", index=False)
        pd.DataFrame(mart_rows).to_csv(MART_AUDIT, sep="\t", index=False)

        status = "BLOCKED_NO_ENSEMBL_GENES_MART_FROM_REGISTRY"
        write_json(
            SUMMARY_JSON,
            {
                "script_version": SCRIPT_VERSION,
                "status": status,
                "created_utc": now_utc(),
                "route_rows": route_rows,
            },
        )
        raise RuntimeError(status)

    # Deduplicate actual mart routes.
    unique_routes: List[Tuple[str, str]] = []
    for _, mart_url, schema in successful_mart_routes:
        pair = (mart_url, schema)
        if pair not in unique_routes:
            unique_routes.append(pair)

    # Also explicitly probe current mart.ensembl.org route even if registry points
    # elsewhere, because it is the current public Ensembl host.
    explicit_pair = (
        "https://mart.ensembl.org/biomart/martservice",
        "default",
    )
    if explicit_pair not in unique_routes:
        unique_routes.insert(0, explicit_pair)

    # ------------------------------------------------------------------
    # 2. Dataset probes.
    # ------------------------------------------------------------------
    print()
    print("Dataset probes:")

    dog_dataset_hits: List[Dict[str, str]] = []

    for idx, (mart_url, schema) in enumerate(unique_routes, start=1):
        try:
            text, final_url, _ = request_text(
                session,
                mart_url,
                {
                    "type": "datasets",
                    "mart": EXPECTED_MART,
                    "virtualSchema": schema,
                },
                timeout=90,
            )
            raw_path = save_raw(f"datasets_{idx}.txt", text)
            frame = parse_dataset_lines(text)

            if not frame.empty:
                frame = frame.copy()
                frame["queried_mart_url"] = mart_url
                frame["final_url"] = final_url
                frame["virtual_schema"] = schema
                frame["raw_artifact"] = str(raw_path.relative_to(ROOT))
                dataset_rows.extend(frame.to_dict(orient="records"))

            candidates = dataset_candidates(frame)

            print(
                f"  {mart_url} -> {final_url}: "
                f"{len(frame)} non-empty line(s), dog candidate(s)={candidates or '<none>'}"
            )

            for dataset in candidates:
                dog_dataset_hits.append(
                    {
                        "mart_url": mart_url,
                        "final_url": final_url,
                        "virtual_schema": schema,
                        "dataset": dataset,
                    }
                )

        except Exception as exc:
            dataset_rows.append(
                {
                    "queried_mart_url": mart_url,
                    "virtual_schema": schema,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"  {mart_url}: FAIL - {exc}")

    pd.DataFrame(route_rows).to_csv(ROUTE_AUDIT, sep="\t", index=False)
    pd.DataFrame(mart_rows).to_csv(MART_AUDIT, sep="\t", index=False)
    pd.DataFrame(dataset_rows).to_csv(DATASET_AUDIT, sep="\t", index=False)

    # Deduplicate hits.
    seen = set()
    unique_hits: List[Dict[str, str]] = []

    for hit in dog_dataset_hits:
        key = (
            hit["final_url"],
            hit["virtual_schema"],
            hit["dataset"],
        )
        if key not in seen:
            seen.add(key)
            unique_hits.append(hit)

    if not unique_hits:
        status = "BLOCKED_NO_DOG_DATASET_FOUND"
        write_json(
            SUMMARY_JSON,
            {
                "script_version": SCRIPT_VERSION,
                "status": status,
                "created_utc": now_utc(),
                "route_rows": route_rows,
                "dog_dataset_hits": [],
            },
        )
        raise RuntimeError(
            f"{status}. Inspect {DATASET_AUDIT.relative_to(ROOT)} and datasets_*.txt."
        )

    # ------------------------------------------------------------------
    # 3. Attribute probes.
    # ------------------------------------------------------------------
    print()
    print("Attribute probes:")

    ready_hits: List[Dict[str, Any]] = []
    alias_patch_hits: List[Dict[str, Any]] = []

    for idx, hit in enumerate(unique_hits, start=1):
        final_url = hit["final_url"]
        schema = hit["virtual_schema"]
        dataset = hit["dataset"]

        try:
            text, attr_final_url, _ = request_text(
                session,
                final_url,
                {
                    "type": "attributes",
                    "dataset": dataset,
                    "mart": EXPECTED_MART,
                    "virtualSchema": schema,
                },
                timeout=90,
            )
            raw_path = save_raw(f"attributes_{idx}_{dataset}.txt", text)

            frame = parse_attribute_lines(text)
            if frame.empty:
                raise RuntimeError("attribute response parsed to zero lines")

            frame = frame.copy()
            frame["dataset"] = dataset
            frame["queried_url"] = final_url
            frame["final_url"] = attr_final_url
            frame["virtual_schema"] = schema
            frame["raw_artifact"] = str(raw_path.relative_to(ROOT))

            attribute_rows.extend(frame.to_dict(orient="records"))

            available = set(frame["name"].astype(str).str.strip())

            required_resolution = concept_resolution(
                available,
                REQUIRED_CONCEPTS,
            )
            optional_resolution = concept_resolution(
                available,
                OPTIONAL_CONCEPTS,
            )

            missing_required = [
                concept
                for concept, info in required_resolution.items()
                if not info["available"]
            ]

            homology_candidates = sorted(
                frame.loc[
                    frame["human_homology_like"], "name"
                ]
                .astype(str)
                .unique()
                .tolist()
            )
            core_candidates = sorted(
                frame.loc[
                    frame["core_gene_like"], "name"
                ]
                .astype(str)
                .unique()
                .tolist()
            )

            payload = {
                **hit,
                "attribute_final_url": attr_final_url,
                "required_resolution": required_resolution,
                "optional_resolution": optional_resolution,
                "missing_required_concepts": missing_required,
                "homology_candidate_names": homology_candidates,
                "core_gene_candidate_names": core_candidates,
            }

            if not missing_required:
                ready_hits.append(payload)
                label = "READY"
            else:
                alias_patch_hits.append(payload)
                label = "NEEDS_ALIAS_PATCH"

            print(
                f"  {dataset} @ {attr_final_url}: {label}, "
                f"attributes={len(frame)}, missing required concepts={missing_required or '<none>'}"
            )

            if missing_required:
                print(
                    "    core candidates: "
                    + ", ".join(core_candidates[:20])
                )
                print(
                    "    human/homology candidates: "
                    + ", ".join(homology_candidates[:40])
                )

        except Exception as exc:
            attribute_rows.append(
                {
                    "dataset": dataset,
                    "queried_url": final_url,
                    "virtual_schema": schema,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"  {dataset}: FAIL - {exc}")

    pd.DataFrame(attribute_rows).to_csv(
        ATTRIBUTE_AUDIT,
        sep="\t",
        index=False,
    )

    if ready_hits:
        status = "PASS_CURRENT_BIOMART_ROUTE_DATASET_ATTRIBUTES_RESOLVED"
        preferred = ready_hits[0]
    elif alias_patch_hits:
        status = "PASS_DOG_DATASET_FOUND_ATTRIBUTE_ALIASES_NEED_PATCH"
        preferred = alias_patch_hits[0]
    else:
        status = "BLOCKED_DOG_DATASET_ATTRIBUTE_QUERY_FAILED"
        preferred = None

    summary = {
        "script_version": SCRIPT_VERSION,
        "status": status,
        "created_utc": now_utc(),
        "entry_points": ENTRY_POINTS,
        "preferred_route": preferred,
        "ready_hits": ready_hits,
        "alias_patch_hits": alias_patch_hits,
        "artifacts": {
            "route_audit": str(ROUTE_AUDIT.relative_to(ROOT)),
            "mart_registry_audit": str(MART_AUDIT.relative_to(ROOT)),
            "dataset_audit": str(DATASET_AUDIT.relative_to(ROOT)),
            "attribute_audit": str(ATTRIBUTE_AUDIT.relative_to(ROOT)),
        },
        "safety": {
            "orthology_mapping_rows_retrieved": False,
            "expression_values_read": False,
            "clinical_values_read": False,
            "outcome_response_followup_values_read": False,
            "treatment_values_read": False,
            "model_fitting": False,
        },
    }
    write_json(SUMMARY_JSON, summary)

    print()
    print("=" * 118)
    print("02g0 ENSEMBL BIOMART PROBE SUMMARY")
    print("=" * 118)
    print(f"Status: {status}")

    if preferred is not None:
        print(f"Preferred final mart URL: {preferred['final_url']}")
        print(f"Virtual schema: {preferred['virtual_schema']}")
        print(f"Dog dataset: {preferred['dataset']}")

        required_resolution = preferred["required_resolution"]
        print("Required concept resolution:")

        for concept, info in required_resolution.items():
            print(
                f"  {concept}: "
                f"{info['resolved'] if info['available'] else '<MISSING>'}"
            )

        print("Optional concept resolution:")
        for concept, info in preferred["optional_resolution"].items():
            print(
                f"  {concept}: "
                f"{info['resolved'] if info['available'] else '<ABSENT>'}"
            )

    print()
    print("Orthology mapping rows retrieved: NO")
    print("Expression values read: NO")
    print("Clinical/outcome/treatment values read: NO")
    print("Model fitting: NO")
    print()
    print("Artifacts:")
    print(f"  {ROUTE_AUDIT.relative_to(ROOT)}")
    print(f"  {MART_AUDIT.relative_to(ROOT)}")
    print(f"  {DATASET_AUDIT.relative_to(ROOT)}")
    print(f"  {ATTRIBUTE_AUDIT.relative_to(ROOT)}")
    print(f"  {SUMMARY_JSON.relative_to(ROOT)}")
    print("=" * 118)

    if status.startswith("BLOCKED"):
        raise RuntimeError(status)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 118, file=sys.stderr)
        print("02g0 Ensembl BioMart probe: FAIL", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print("=" * 118, file=sys.stderr)
        raise
