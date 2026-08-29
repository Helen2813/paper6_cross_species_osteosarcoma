#!/usr/bin/env python3
"""
Paper 6 - freeze sacrificial human premise-test target from metadata only.

Purpose
-------
Select and freeze ONE human osteosarcoma cohort for the sacrificial
canine-added-value premise test BEFORE any Paper-6 access to its outcome values.

Selection is based only on:
- cohort identity and sample size;
- human primary osteosarcoma lineage;
- pretreatment/primary-tumor provenance;
- documented presence of an overall-survival time/status schema;
- public expression availability;
- overlap risk with reserved confirmatory cohorts.

NO human outcome value, event count, survival-time distribution, expression
value, prognostic performance, or model result is read.

Frozen candidate set
--------------------
GSE16091
GSE30699
GSE32981
GSE33382

Reserved confirmatory cohorts
-----------------------------
TARGET-OS
GSE21257
GSE39055

Selection rule
--------------
A candidate is eligible only if all are true:
1. Human osteosarcoma patient tumors, not cell lines/xenografts.
2. At least 30 patient tumor samples.
3. Primary/pretreatment tumor lineage is documented.
4. A continuous OS time + event/censoring status schema is publicly documented.
5. Expression data are public.
6. No known material patient/sample overlap with reserved cohorts.
7. Candidate is not selected by outcome distribution/performance.

Among eligible candidates, select the candidate with the simplest directly
reproducible outcome schema; ties are resolved by larger sample count.

Important
---------
GSE16091 has historical comparative-oncology use. This is frozen as a
transparency flag, NOT used as a positive selection criterion.

Network access is limited to GEO SERIES-LEVEL quick metadata pages. Sample-level
pages are NOT fetched, so outcome values remain unopened by Paper 6.

No CLI arguments.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import requests


SCRIPT_VERSION = "04a-freeze-sacrificial-human-premise-target-v1-no-cli"

ROOT = Path(__file__).resolve().parents[1]

# Upstream frozen design and completed DOG2 gate.
I_DIR = ROOT / "results" / "revised_design" / "02i"
I_CONTRACT = I_DIR / "revised_model_benchmark_source_gate_contract.json"
I_SUMMARY = I_DIR / "summary.json"

C_DIR = ROOT / "results" / "source_gate_diagnostics" / "03c"
C_SUMMARY = C_DIR / "summary.json"
C_PREMISE_POLICY = C_DIR / "premise_test_action_policy_addendum.json"

B_DIR = ROOT / "results" / "source_gate" / "03b"
B_SUMMARY = B_DIR / "summary.json"

OUT_DIR = ROOT / "results" / "human_premise_target" / "04a"
SNAP_DIR = OUT_DIR / "geo_series_quick_snapshots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SNAP_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATE_AUDIT = OUT_DIR / "candidate_metadata_audit.tsv"
GEO_OVERLAP_AUDIT = OUT_DIR / "geo_sample_accession_overlap_audit.tsv"
TARGET_SAMPLE_ROSTER = OUT_DIR / "sacrificial_target_sample_roster.tsv"
CONTRACT_JSON = OUT_DIR / "sacrificial_human_premise_target_contract.json"
SUMMARY_JSON = OUT_DIR / "summary.json"

EXPECTED_SOURCE_GATE = "SOURCE_AMBER_WEAK_AND_CONTEXT_SENSITIVE"

CANDIDATES = ["GSE16091", "GSE30699", "GSE32981", "GSE33382"]
RESERVED_GEO = ["GSE21257", "GSE39055"]
RESERVED_ALL = ["TARGET-OS", "GSE21257", "GSE39055"]

GEO_QUICK_URL = (
    "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
    "?acc={acc}&targ=self&form=text&view=quick"
)

# Metadata facts frozen from public cohort records/publications BEFORE Paper-6
# human outcome access. These are schema/provenance facts, not outcome values.
#
# GSE16091:
# - GEO: 34 human osteosarcoma samples, GPL96.
# - Paoloni et al. 2009 / PMID 20028558:
#   distinct population of 34 pre-treatment primary human osteosarcoma tumors
#   linked to overall survival.
# - GEO sample schema contains alive-or-dead and days-followup.
#
# GSE30699:
# - 76 pre-treatment diagnostic high-grade osteosarcoma biopsies, but series also
#   contains cell lines/xenografts; no directly documented continuous OS schema
#   in the GEO series-level metadata used for this freeze.
#
# GSE32981:
# - 23 osteosarcoma samples of primary/metastatic origin; below n>=30 and no
#   directly documented continuous OS schema used here.
#
# GSE33382:
# - 84 pre-treatment diagnostic biopsies.
# - GEO directly exposes metastasis-within-5yrs metadata rather than a simple
#   continuous OS time/status schema.
# - independent literature reports 27 samples overlap GSE21257, which is reserved.
CANDIDATE_FACTS: Dict[str, Dict[str, Any]] = {
    "GSE16091": {
        "documented_patient_tumor_n": 34,
        "organism": "Homo sapiens",
        "lineage": "primary human osteosarcoma tumors",
        "pretreatment_primary_documented": True,
        "continuous_os_schema_documented": True,
        "os_time_field": "days-followup",
        "os_status_field": "alive-or-dead",
        "os_status_mapping_frozen_for_future_opening": {
            "D": 1,
            "A": 0,
        },
        "public_expression": True,
        "platform": "GPL96 / Affymetrix Human Genome U133A",
        "known_material_overlap_with_reserved": False,
        "known_overlap_note": (
            "No exact GEO GSM overlap with reserved GEO cohorts is permitted; "
            "no material patient overlap is known from the frozen metadata evidence. "
            "Cross-repository identity with TARGET cannot be proven from GEO metadata alone."
        ),
        "historical_cross_species_use": True,
        "historical_cross_species_note": (
            "This 34-patient outcome cohort was previously used in Paoloni et al. "
            "(PMID 20028558) to evaluate dog-like genes defined from a DIFFERENT "
            "canine osteosarcoma dataset. Paper 6 must disclose this history."
        ),
        "primary_publication_pmids": "20028558",
        "selection_evidence_note": (
            "Distinct 34-patient pre-treatment primary-tumor population with "
            "overall-survival times documented in primary publication."
        ),
    },
    "GSE30699": {
        "documented_patient_tumor_n": 76,
        "organism": "Homo sapiens",
        "lineage": (
            "pre-treatment diagnostic high-grade osteosarcoma biopsies plus "
            "cell-line/xenograft records in the wider series"
        ),
        "pretreatment_primary_documented": True,
        "continuous_os_schema_documented": False,
        "os_time_field": "",
        "os_status_field": "",
        "os_status_mapping_frozen_for_future_opening": {},
        "public_expression": True,
        "platform": "GPL10295 / Illumina human-6 v2.0",
        "known_material_overlap_with_reserved": False,
        "known_overlap_note": (
            "Potential EuroBoNet lineage overlap requires patient-level resolution; "
            "not eligible here because direct continuous OS schema criterion already fails."
        ),
        "historical_cross_species_use": False,
        "historical_cross_species_note": "",
        "primary_publication_pmids": "",
        "selection_evidence_note": (
            "Good pretreatment biopsy lineage, but no simple directly reproducible "
            "continuous OS time/status schema frozen from GEO/publication metadata."
        ),
    },
    "GSE32981": {
        "documented_patient_tumor_n": 23,
        "organism": "Homo sapiens",
        "lineage": "osteosarcoma samples of primary and metastatic origin",
        "pretreatment_primary_documented": False,
        "continuous_os_schema_documented": False,
        "os_time_field": "",
        "os_status_field": "",
        "os_status_mapping_frozen_for_future_opening": {},
        "public_expression": True,
        "platform": "GPL3307 / Applied Biosystems Human Genome Survey Microarray",
        "known_material_overlap_with_reserved": False,
        "known_overlap_note": "",
        "historical_cross_species_use": False,
        "historical_cross_species_note": "",
        "primary_publication_pmids": "",
        "selection_evidence_note": (
            "Below n>=30 and mixed primary/metastatic lineage."
        ),
    },
    "GSE33382": {
        "documented_patient_tumor_n": 84,
        "organism": "Homo sapiens",
        "lineage": "pre-treatment high-grade osteosarcoma diagnostic biopsies",
        "pretreatment_primary_documented": True,
        "continuous_os_schema_documented": False,
        "os_time_field": "",
        "os_status_field": "",
        "os_status_mapping_frozen_for_future_opening": {},
        "public_expression": True,
        "platform": "GPL10295 / Illumina human-6 v2.0",
        "known_material_overlap_with_reserved": True,
        "known_overlap_note": (
            "Published overlap audit reports 27 GSE42352/GSE33382 samples overlap "
            "the reserved GSE21257 cohort."
        ),
        "historical_cross_species_use": False,
        "historical_cross_species_note": "",
        "primary_publication_pmids": "22454324;23688189;24447333",
        "selection_evidence_note": (
            "Large pretreatment cohort, but known overlap with reserved GSE21257 "
            "and no directly frozen simple continuous OS schema in GEO."
        ),
    },
}

MIN_PATIENT_TUMOR_N = 30


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


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


def fetch_series_quick(
    session: requests.Session,
    accession: str,
    retries: int = 3,
) -> Path:
    path = SNAP_DIR / f"{accession}_series_quick.txt"

    if path.exists() and path.stat().st_size > 0:
        return path

    url = GEO_QUICK_URL.format(acc=accession)
    last_exc: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        try:
            response = session.get(
                url,
                timeout=60,
                headers={"User-Agent": "Paper6-04a-metadata-only/1.0"},
            )
            response.raise_for_status()
            text = response.text

            if not text.strip():
                raise RuntimeError("Empty GEO quick response.")

            # Fail closed if a SERIES-level request unexpectedly returns a sample
            # expression table or sample-level outcome-characteristic payload.
            forbidden_markers = [
                "!sample_table_begin",
                "!sample_characteristics_ch1",
                "!sample_description",
                "id_ref\tvalue",
            ]
            lower = text.lower()
            if any(marker in lower for marker in forbidden_markers):
                raise RuntimeError(
                    f"{accession}: GEO quick response unexpectedly contains "
                    "sample-level/value-like content; refusing pre-outcome read."
                )

            path.write_text(text, encoding="utf-8", newline="\n")
            return path
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(2 * attempt)

    raise RuntimeError(
        f"Could not fetch SERIES-level GEO quick metadata for {accession}: {last_exc}"
    )


def parse_quick_series(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")

    def values(field: str) -> List[str]:
        prefix = f"!Series_{field} = "
        out = []
        for line in text.splitlines():
            if line.startswith(prefix):
                out.append(line[len(prefix):].strip())
        return out

    sample_ids = values("sample_id")
    platform_ids = values("platform_id")

    title_values = values("title")
    organism_values = values("organism_ch1")
    summary_values = values("summary")
    design_values = values("overall_design")
    pubmed_values = values("pubmed_id")

    return {
        "title": title_values[0] if title_values else "",
        "organism": organism_values[0] if organism_values else "",
        "platform_ids": sorted(set(platform_ids)),
        "sample_ids": sample_ids,
        "sample_n_from_series_quick": len(sample_ids),
        "summary_text_present": bool(summary_values),
        "overall_design_text_present": bool(design_values),
        "pubmed_ids": sorted(set(pubmed_values)),
        "snapshot_sha256": sha256_file(path),
    }


def normalized_accession_set(values: Sequence[str]) -> set[str]:
    return {clean(x).upper() for x in values if clean(x)}


def main() -> None:
    started = now_utc()

    print("=" * 120)
    print("Paper 6 - freeze sacrificial human premise-test target from metadata only")
    print("=" * 120)
    print(f"Script version: {SCRIPT_VERSION}")
    print()
    print("Safety / scope:")
    print("  Human outcome VALUES read: NO")
    print("  Human event counts read: NO")
    print("  Human survival-time distributions read: NO")
    print("  Human expression VALUES read: NO")
    print("  Human sample-level GEO pages fetched: NO")
    print("  GEO SERIES-level metadata fetched: YES")
    print("  Candidate selection by performance: NO")
    print("  TARGET/GSE21257/GSE39055 outcome access: NO")
    print("  Model fitting: NO")
    print()

    for path in [
        I_CONTRACT,
        I_SUMMARY,
        C_SUMMARY,
        C_PREMISE_POLICY,
        B_SUMMARY,
    ]:
        require_file(path)

    i_contract = read_json(I_CONTRACT)
    c_summary = read_json(C_SUMMARY)
    premise_policy = read_json(C_PREMISE_POLICY)
    b_summary = read_json(B_SUMMARY)

    if clean(i_contract.get("status")) != "PASS":
        raise RuntimeError("02i contract is not PASS.")
    if clean(c_summary.get("status")) != "PASS":
        raise RuntimeError("03c summary is not PASS.")
    if clean(c_summary.get("scientific_status")) != (
        "PASS_BOUNDED_DIAGNOSTIC_COMPLETE_SOURCE_GATE_UNCHANGED"
    ):
        raise RuntimeError("03c bounded diagnostic is not in expected PASS state.")
    if clean(b_summary.get("scientific_status")) != EXPECTED_SOURCE_GATE:
        raise RuntimeError("03b Source Gate status changed.")
    if not clean(premise_policy.get("status")).startswith("PASS_"):
        raise RuntimeError("03c premise action policy is not PASS.")

    frozen_premise = i_contract.get("premise_test") or {}
    if clean(frozen_premise.get("primary_contrast")) != (
        "DOG2_PLUS_HUMAN_POOL_MINUS_HUMAN_POOL_ONLY"
    ):
        raise RuntimeError("02i premise-test primary contrast changed.")

    session = requests.Session()

    series_meta: Dict[str, Dict[str, Any]] = {}
    for acc in CANDIDATES + RESERVED_GEO:
        path = fetch_series_quick(session, acc)
        series_meta[acc] = parse_quick_series(path)

    # Exact GSM overlap audit only. This does NOT prove patient independence
    # across repositories/renamed samples; limitations are retained explicitly.
    overlap_rows: List[Dict[str, Any]] = []

    for candidate in CANDIDATES:
        cset = normalized_accession_set(series_meta[candidate]["sample_ids"])

        for reserved in RESERVED_GEO:
            rset = normalized_accession_set(series_meta[reserved]["sample_ids"])
            overlap = sorted(cset & rset)

            overlap_rows.append(
                {
                    "candidate": candidate,
                    "reserved_cohort": reserved,
                    "candidate_geo_sample_n": len(cset),
                    "reserved_geo_sample_n": len(rset),
                    "exact_GSM_overlap_n": len(overlap),
                    "exact_GSM_overlap_accessions": ";".join(overlap),
                    "interpretation": (
                        "Exact GEO accession overlap only; absence does not prove "
                        "patient independence if the same patient was deposited "
                        "under different accessions."
                    ),
                }
            )

    overlap_df = pd.DataFrame(overlap_rows)
    overlap_df.to_csv(GEO_OVERLAP_AUDIT, sep="\t", index=False)

    audit_rows: List[Dict[str, Any]] = []

    for candidate in CANDIDATES:
        facts = CANDIDATE_FACTS[candidate]
        geo = series_meta[candidate]

        exact_reserved_gsm_overlap_n = int(
            overlap_df.loc[
                overlap_df["candidate"] == candidate,
                "exact_GSM_overlap_n",
            ].sum()
        )

        criterion_human = clean(facts["organism"]) == "Homo sapiens"
        criterion_n = int(facts["documented_patient_tumor_n"]) >= MIN_PATIENT_TUMOR_N
        criterion_lineage = bool(facts["pretreatment_primary_documented"])
        criterion_os = bool(facts["continuous_os_schema_documented"])
        criterion_expression = bool(facts["public_expression"])
        criterion_overlap = (
            not bool(facts["known_material_overlap_with_reserved"])
            and exact_reserved_gsm_overlap_n == 0
        )

        eligible = all(
            [
                criterion_human,
                criterion_n,
                criterion_lineage,
                criterion_os,
                criterion_expression,
                criterion_overlap,
            ]
        )

        # Simplicity score is metadata-only. Direct time/status schema = 2;
        # external/non-direct schema would be lower but is not eligible here.
        schema_simplicity_score = 2 if criterion_os else 0

        audit_rows.append(
            {
                "candidate": candidate,
                "documented_patient_tumor_n": int(
                    facts["documented_patient_tumor_n"]
                ),
                "geo_series_sample_n": int(geo["sample_n_from_series_quick"]),
                "platform": facts["platform"],
                "human_osteosarcoma": criterion_human,
                "pretreatment_primary_documented": criterion_lineage,
                "continuous_OS_time_status_schema_documented": criterion_os,
                "public_expression": criterion_expression,
                "known_material_overlap_with_reserved": bool(
                    facts["known_material_overlap_with_reserved"]
                ),
                "exact_GSM_overlap_with_reserved_n": exact_reserved_gsm_overlap_n,
                "historical_cross_species_use": bool(
                    facts["historical_cross_species_use"]
                ),
                "schema_simplicity_score": schema_simplicity_score,
                "eligible": eligible,
                "selection_evidence_note": facts["selection_evidence_note"],
                "known_overlap_note": facts["known_overlap_note"],
            }
        )

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(CANDIDATE_AUDIT, sep="\t", index=False)

    eligible = audit[audit["eligible"]].copy()
    if eligible.empty:
        raise RuntimeError(
            "No sacrificial premise-test target satisfies the frozen metadata criteria."
        )

    eligible = eligible.sort_values(
        ["schema_simplicity_score", "documented_patient_tumor_n", "candidate"],
        ascending=[False, False, True],
    )

    selected = str(eligible.iloc[0]["candidate"])

    if selected != "GSE16091":
        raise RuntimeError(
            f"Metadata-only rule selected {selected}, but this v1 contract expected "
            "GSE16091 from the pre-frozen candidate evidence. Review before outcome access."
        )

    selected_facts = CANDIDATE_FACTS[selected]
    selected_geo = series_meta[selected]

    # Freeze exact GEO sample accession roster WITHOUT fetching sample pages.
    target_roster = pd.DataFrame(
        {
            "geo_accession": selected_geo["sample_ids"],
            "target_role": "SACRIFICIAL_PREMISE_TEST_TARGET",
            "outcome_values_read_in_04a": False,
            "expression_values_read_in_04a": False,
        }
    )

    if len(target_roster) != int(selected_facts["documented_patient_tumor_n"]):
        raise RuntimeError(
            f"{selected}: GEO series sample roster has {len(target_roster)} rows, "
            f"but documented patient tumor n is "
            f"{selected_facts['documented_patient_tumor_n']}."
        )

    if target_roster["geo_accession"].duplicated().any():
        raise RuntimeError("Selected GEO sample accession roster is not unique.")

    target_roster.to_csv(TARGET_SAMPLE_ROSTER, sep="\t", index=False)

    # Reserved-cohort exact GSM overlap must be zero for selected target.
    selected_overlap = overlap_df[overlap_df["candidate"] == selected]
    if int(selected_overlap["exact_GSM_overlap_n"].sum()) != 0:
        raise RuntimeError(
            "Selected sacrificial target has exact GSM overlap with a reserved cohort."
        )

    contract = {
        "script_version": SCRIPT_VERSION,
        "status": "PASS",
        "scientific_status": "PASS_SACRIFICIAL_HUMAN_PREMISE_TARGET_FROZEN",
        "created_utc": now_utc(),
        "selection_basis": (
            "Metadata/endpoint-schema/sample-lineage criteria only; no outcome "
            "values, event counts, survival distributions, expression values, or "
            "model performance were used."
        ),
        "candidate_set_frozen": CANDIDATES,
        "reserved_confirmatory_cohorts": RESERVED_ALL,
        "selection_rules": {
            "human_osteosarcoma_patient_tumors": True,
            "minimum_patient_tumor_n": MIN_PATIENT_TUMOR_N,
            "pretreatment_primary_tumor_documented": True,
            "continuous_OS_time_and_status_schema_documented": True,
            "public_expression_required": True,
            "no_known_material_overlap_with_reserved": True,
            "no_exact_GSM_overlap_with_reserved_GEO_cohorts": True,
            "primary_tie_break_1": "simplest directly reproducible OS schema",
            "primary_tie_break_2": "larger documented patient tumor n",
            "outcome_distribution_or_performance_used": False,
        },
        "selected_target": {
            "accession": selected,
            "role": "SACRIFICIAL_PREMISE_TEST_TARGET",
            "n": int(selected_facts["documented_patient_tumor_n"]),
            "platform": selected_facts["platform"],
            "lineage": selected_facts["lineage"],
            "pretreatment_primary_documented": bool(
                selected_facts["pretreatment_primary_documented"]
            ),
            "endpoint": "overall survival",
            "time_field_frozen_for_future_outcome_opening": selected_facts[
                "os_time_field"
            ],
            "status_field_frozen_for_future_outcome_opening": selected_facts[
                "os_status_field"
            ],
            "status_mapping_frozen_for_future_outcome_opening": selected_facts[
                "os_status_mapping_frozen_for_future_opening"
            ],
            "unexpected_status_value_action": "FAIL_CLOSED_NO_RECODING_AFTER_OPENING",
            "historical_cross_species_use_flag": bool(
                selected_facts["historical_cross_species_use"]
            ),
            "historical_cross_species_use_note": selected_facts[
                "historical_cross_species_note"
            ],
            "primary_publication_pmids": selected_facts[
                "primary_publication_pmids"
            ],
            "sample_roster_artifact": str(TARGET_SAMPLE_ROSTER.relative_to(ROOT)),
            "sample_roster_sha256": sha256_file(TARGET_SAMPLE_ROSTER),
        },
        "overlap_guardrail": {
            "selected_target_exact_GSM_overlap_GSE21257": int(
                selected_overlap.loc[
                    selected_overlap["reserved_cohort"] == "GSE21257",
                    "exact_GSM_overlap_n",
                ].iloc[0]
            ),
            "selected_target_exact_GSM_overlap_GSE39055": int(
                selected_overlap.loc[
                    selected_overlap["reserved_cohort"] == "GSE39055",
                    "exact_GSM_overlap_n",
                ].iloc[0]
            ),
            "TARGET_cross_repository_patient_identity": (
                "Cannot be proven from GEO accession metadata alone; no material "
                "overlap is known, and this uncertainty must remain a limitation."
            ),
        },
        "premise_test_materiality_unchanged_from_02i": {
            "primary_contrast": frozen_premise["primary_contrast"],
            "supports_canine_added_value": frozen_premise[
                "supports_canine_added_value"
            ],
            "argues_against_canine_added_value": frozen_premise[
                "argues_against_canine_added_value"
            ],
            "otherwise": frozen_premise["otherwise"],
        },
        "premise_result_action_policy": premise_policy["actions"],
        "premise_outcomes_may_tune_proposed_AI": False,
        "GSE16091_historical_use_guardrail": (
            "Because GSE16091 was previously used to test dog-like prognostic "
            "genes in an earlier cross-species study using a different canine "
            "dataset, Paper 6 must describe the premise test as sacrificial and "
            "historically comparative-oncology-exposed, not as a pristine neutral "
            "external validation cohort."
        ),
        "outcome_access_after_04a": {
            "GSE16091": (
                "May open only for the frozen classical premise test after the "
                "04b model/source-pool implementation contract is frozen."
            ),
            "TARGET_OS": "CLOSED",
            "GSE21257": "CLOSED",
            "GSE39055": "CLOSED",
        },
        "upstream_hashes": {
            "02i_contract": sha256_file(I_CONTRACT),
            "03b_summary": sha256_file(B_SUMMARY),
            "03c_summary": sha256_file(C_SUMMARY),
            "03c_premise_policy": sha256_file(C_PREMISE_POLICY),
        },
        "geo_series_snapshot_hashes": {
            acc: series_meta[acc]["snapshot_sha256"]
            for acc in CANDIDATES + RESERVED_GEO
        },
        "safety": {
            "human_outcome_values_read": False,
            "human_event_counts_read": False,
            "human_survival_distribution_read": False,
            "human_expression_values_read": False,
            "human_sample_level_GEO_pages_fetched": False,
            "human_model_fitting": False,
        },
        "required_next": (
            "04b freeze the classical premise-test/source-pool implementation "
            "before opening GSE16091 outcome values."
        ),
    }
    write_json(CONTRACT_JSON, contract)

    final_hashes = {
        "candidate_metadata_audit_tsv": sha256_file(CANDIDATE_AUDIT),
        "geo_sample_accession_overlap_audit_tsv": sha256_file(GEO_OVERLAP_AUDIT),
        "sacrificial_target_sample_roster_tsv": sha256_file(TARGET_SAMPLE_ROSTER),
        "sacrificial_human_premise_target_contract_json": sha256_file(
            CONTRACT_JSON
        ),
    }

    summary = {
        "script_version": SCRIPT_VERSION,
        "status": "PASS",
        "scientific_status": "PASS_SACRIFICIAL_HUMAN_PREMISE_TARGET_FROZEN",
        "run_started_utc": started,
        "run_finished_utc": now_utc(),
        "selected_target": selected,
        "selected_target_n": int(selected_facts["documented_patient_tumor_n"]),
        "selected_target_platform": selected_facts["platform"],
        "selected_endpoint": "overall survival",
        "selected_time_field": selected_facts["os_time_field"],
        "selected_status_field": selected_facts["os_status_field"],
        "historical_cross_species_use_flag": bool(
            selected_facts["historical_cross_species_use"]
        ),
        "exact_GSM_overlap_with_GSE21257": int(
            selected_overlap.loc[
                selected_overlap["reserved_cohort"] == "GSE21257",
                "exact_GSM_overlap_n",
            ].iloc[0]
        ),
        "exact_GSM_overlap_with_GSE39055": int(
            selected_overlap.loc[
                selected_overlap["reserved_cohort"] == "GSE39055",
                "exact_GSM_overlap_n",
            ].iloc[0]
        ),
        "human_outcome_values_read": False,
        "human_expression_values_read": False,
        "human_model_fitting": False,
        "final_artifact_hashes": final_hashes,
        "next": "04b classical premise-test/source-pool implementation freeze",
    }
    write_json(SUMMARY_JSON, summary)

    print("-" * 120)
    print("Metadata-only candidate audit")
    print("-" * 120)
    print(
        audit[
            [
                "candidate",
                "documented_patient_tumor_n",
                "pretreatment_primary_documented",
                "continuous_OS_time_status_schema_documented",
                "known_material_overlap_with_reserved",
                "historical_cross_species_use",
                "eligible",
            ]
        ].to_string(index=False)
    )

    print()
    print("Exact GEO sample-accession overlap with reserved GEO cohorts:")
    print(
        overlap_df[
            [
                "candidate",
                "reserved_cohort",
                "exact_GSM_overlap_n",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 120)
    print("04a SACRIFICIAL HUMAN PREMISE-TARGET SUMMARY")
    print("=" * 120)
    print(f"Selected target: {selected}")
    print(f"Role: SACRIFICIAL_PREMISE_TEST_TARGET")
    print(f"Documented patient tumors: {selected_facts['documented_patient_tumor_n']}")
    print(f"Platform: {selected_facts['platform']}")
    print("Endpoint: overall survival")
    print(
        f"Future time/status fields: "
        f"{selected_facts['os_time_field']} / {selected_facts['os_status_field']}"
    )
    print("Outcome values read in 04a: NO")
    print("Expression values read in 04a: NO")
    print(
        "Historical comparative-oncology use flag: "
        f"{'YES' if selected_facts['historical_cross_species_use'] else 'NO'}"
    )
    print()
    print("Reserved outcomes remain CLOSED:")
    print("  TARGET-OS")
    print("  GSE21257")
    print("  GSE39055")
    print()
    print("Next: 04b freeze classical premise-test/source-pool implementation.")
    print("Do NOT open GSE16091 outcome values before 04b.")
    print()
    print("Artifacts:")
    for path in [
        CANDIDATE_AUDIT,
        GEO_OVERLAP_AUDIT,
        TARGET_SAMPLE_ROSTER,
        CONTRACT_JSON,
        SUMMARY_JSON,
    ]:
        print(f"  {path.relative_to(ROOT)}")
    print("=" * 120)
    print("04a sacrificial human premise-test target freeze: PASS")
    print("=" * 120)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 120, file=sys.stderr)
        print(
            "04a sacrificial human premise-test target freeze: FAIL",
            file=sys.stderr,
        )
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print("=" * 120, file=sys.stderr)
        raise
