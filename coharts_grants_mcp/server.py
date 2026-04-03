#!/usr/bin/env python3
"""
Coharts Government Grant Search MCP Server
==========================================
Searches SBIR.gov and SAM.gov for grants relevant to Coharts across:
  - AI x Bio  (AI/ML applied to biology, genomics, drug discovery)
  - Clinical Development  (trials, therapeutics, medical devices, FDA)
  - Precision Medicine  (genomics, biomarkers, personalised therapy)

Tools exposed:
  search_grants          — Semantic keyword search across all grant sources
  get_grant_detail       — Fetch full detail for a specific SBIR award
  list_open_solicitations— Open funding opportunities from SBIR.gov
  rank_by_relevance      — Score + rank a set of grants for Coharts fit
"""

from __future__ import annotations

import json
import math
from typing import Any

import httpx
import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

# ── Constants ────────────────────────────────────────────────────────────────

SBIR_API   = "https://api.sbir.gov/public/api"
SAM_API    = "https://api.sam.gov/opportunities/v2/search"

# All three Coharts focus areas with weighted keyword sets
FOCUS_AREAS: dict[str, list[str]] = {
    "AI x Bio": [
        "artificial intelligence", "machine learning", "deep learning",
        "neural network", "large language model", "foundation model",
        "generative AI", "AI-enabled", "NLP biology",
        "bioinformatics", "computational biology", "systems biology",
        "genomics AI", "proteomics AI", "drug discovery AI",
        "predictive modeling", "biological data", "multiomics",
        "single cell", "spatial transcriptomics", "digital pathology",
        "AI diagnostics", "image analysis biology",
    ],
    "Clinical Development": [
        "clinical development", "clinical trial", "clinical study",
        "phase I", "phase II", "phase III",
        "drug development", "IND application", "FDA approval",
        "therapeutics", "pharmaceutical", "clinical stage",
        "patient outcomes", "medical device", "in vivo",
        "preclinical", "translational medicine", "first-in-human",
        "GLP", "GMP", "regulatory pathway", "NDA", "BLA",
        "cell therapy", "gene therapy", "biologics",
    ],
    "Precision Medicine": [
        "precision medicine", "personalised medicine",
        "personalized medicine", "biomarker", "companion diagnostic",
        "genomics", "whole exome", "whole genome sequencing",
        "liquid biopsy", "circulating tumor DNA", "ctDNA",
        "pharmacogenomics", "molecular profiling", "targeted therapy",
        "patient stratification", "polygenic risk score",
        "rare disease", "orphan drug", "next generation sequencing",
        "proteomics", "metabolomics", "multi-omics",
    ],
}

# All unique keywords (flattened) for broad search queries
ALL_KEYWORDS = sorted({kw for kws in FOCUS_AREAS.values() for kw in kws})

# Pre-built short query strings per focus area (first 3 keywords joined)
AREA_QUERIES = {area: " ".join(kws[:3]) for area, kws in FOCUS_AREAS.items()}


# ── Relevance scoring ────────────────────────────────────────────────────────

def _score_grant(title: str, abstract: str, agency: str = "") -> dict[str, Any]:
    """
    Return per-area keyword hit counts, overall score (0–10),
    ranked focus area list, and a recommendation label.
    """
    text = f"{title} {abstract}".lower()

    area_scores: dict[str, int] = {
        area: sum(1 for kw in kws if kw.lower() in text)
        for area, kws in FOCUS_AREAS.items()
    }
    total_hits = sum(area_scores.values())

    # Normalise to 0–10 (asymptotic so max ≠ achievable instantly)
    overall = round(min(10.0, total_hits * 0.8), 1)

    # Rank focus areas by score
    ranked_areas = sorted(area_scores.items(), key=lambda x: x[1], reverse=True)

    if overall >= 6:
        recommendation = "HIGH — strong Coharts fit; prioritise outreach"
    elif overall >= 3:
        recommendation = "MEDIUM — relevant; monitor and assess"
    elif overall >= 1:
        recommendation = "LOW — tangential relevance; file for reference"
    else:
        recommendation = "NONE — not aligned with Coharts focus areas"

    return {
        "overall_score": overall,
        "area_scores": area_scores,
        "ranked_focus_areas": [a for a, _ in ranked_areas if area_scores[a] > 0],
        "recommendation": recommendation,
    }


# ── SBIR.gov helpers ─────────────────────────────────────────────────────────

def _normalise_award(item: dict) -> dict:
    return {
        "source": "SBIR Award",
        "id": item.get("contract", ""),
        "title": item.get("award_title", ""),
        "abstract": (item.get("abstract") or "")[:800],
        "agency": item.get("agency", ""),
        "program": item.get("program", ""),
        "phase": item.get("phase", ""),
        "award_amount_usd": item.get("award_amount", 0),
        "award_year": item.get("award_year", ""),
        "company": item.get("firm", ""),
        "pi_name": item.get("pi_name", ""),
        "topic_code": item.get("solicitation_topic_code", ""),
    }


def _normalise_solicitation(item: dict) -> dict:
    return {
        "source": "SBIR Solicitation",
        "solicitation_number": item.get("solicitation_number", ""),
        "title": item.get("solicitation_title", ""),
        "abstract": (item.get("description") or "")[:600],
        "agency": item.get("agency", ""),
        "program": item.get("program", ""),
        "phase": item.get("phase", ""),
        "open_date": item.get("open_date", ""),
        "close_date": item.get("close_date", ""),
        "url": item.get("solicitation_url", ""),
    }


async def _sbir_awards(
    keyword: str,
    agency: str = "",
    rows: int = 20,
) -> list[dict]:
    params: dict[str, Any] = {"keyword": keyword, "rows": min(rows, 25), "start": 0}
    if agency:
        params["agency"] = agency

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(f"{SBIR_API}/award", params=params)
            r.raise_for_status()
            data = r.json()
        return [_normalise_award(item) for item in data.get("data", [])]
    except httpx.ProxyError as exc:
        return [{"_api_error": "proxy_blocked", "detail": str(exc), "source": "sbir_awards"}]
    except httpx.ConnectError as exc:
        return [{"_api_error": "connect_failed", "detail": str(exc), "source": "sbir_awards"}]
    except Exception as exc:
        return [{"_api_error": str(exc), "source": "sbir_awards"}]


async def _sbir_solicitations(
    keyword: str,
    agency: str = "",
    rows: int = 15,
) -> list[dict]:
    params: dict[str, Any] = {"keyword": keyword, "rows": min(rows, 25), "start": 0}
    if agency:
        params["agency"] = agency

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(f"{SBIR_API}/solicitation", params=params)
            r.raise_for_status()
            data = r.json()
        return [_normalise_solicitation(item) for item in data.get("data", [])]
    except httpx.ProxyError as exc:
        return [{"_api_error": "proxy_blocked", "detail": str(exc), "source": "sbir_solicitations"}]
    except httpx.ConnectError as exc:
        return [{"_api_error": "connect_failed", "detail": str(exc), "source": "sbir_solicitations"}]
    except Exception as exc:
        return [{"_api_error": str(exc), "source": "sbir_solicitations"}]


# ── MCP Server ────────────────────────────────────────────────────────────────

app = Server("coharts-grants")


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="search_grants",
            description=(
                "Search SBIR.gov awarded grants for topics relevant to Coharts. "
                "Covers AI x Bio, Clinical Development, and Precision Medicine. "
                "Each result includes a relevance score and ranked focus-area tags."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Free-text query (e.g. 'machine learning drug discovery', "
                            "'clinical trial genomics'). Leave empty to run a broad "
                            "Coharts-relevant sweep."
                        ),
                    },
                    "focus_area": {
                        "type": "string",
                        "enum": ["AI x Bio", "Clinical Development", "Precision Medicine", "All"],
                        "description": "Restrict results to one focus area or search All.",
                        "default": "All",
                    },
                    "agency": {
                        "type": "string",
                        "description": (
                            "Optional agency filter (e.g. 'NIH', 'DoD', 'DARPA', 'NSF', "
                            "'HHS'). Leave empty for all agencies."
                        ),
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum grants to return (1–50). Default: 20.",
                        "default": 20,
                    },
                    "min_relevance_score": {
                        "type": "number",
                        "description": "Only return grants with score ≥ this value (0–10). Default: 1.",
                        "default": 1,
                    },
                },
                "required": [],
            },
        ),
        types.Tool(
            name="list_open_solicitations",
            description=(
                "Fetch currently open SBIR/STTR solicitations (grant competitions) "
                "relevant to Coharts. Returns deadlines, agency, phase, and a direct URL."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "focus_area": {
                        "type": "string",
                        "enum": ["AI x Bio", "Clinical Development", "Precision Medicine", "All"],
                        "description": "Filter solicitations by focus area.",
                        "default": "All",
                    },
                    "agency": {
                        "type": "string",
                        "description": "Optional agency filter (e.g. 'NIH', 'DoD').",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum solicitations to return. Default: 15.",
                        "default": 15,
                    },
                },
                "required": [],
            },
        ),
        types.Tool(
            name="get_grant_detail",
            description=(
                "Fetch full detail for a specific SBIR award by contract number. "
                "Returns complete abstract, company info, PI, and Coharts relevance scoring."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "contract_number": {
                        "type": "string",
                        "description": "SBIR contract/award number (e.g. 'W81XWH-20-1-0001').",
                    },
                },
                "required": ["contract_number"],
            },
        ),
        types.Tool(
            name="rank_by_relevance",
            description=(
                "Score and rank a list of grant objects by Coharts relevance. "
                "Pass in raw grant dicts (with 'title' and 'abstract' fields) "
                "and receive them back sorted from most to least relevant, with "
                "per-area scores and GTM recommendations."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "grants": {
                        "type": "array",
                        "description": "Array of grant objects, each with at least 'title' and 'abstract'.",
                        "items": {"type": "object"},
                    },
                    "min_score": {
                        "type": "number",
                        "description": "Drop grants below this score. Default: 0 (return all).",
                        "default": 0,
                    },
                },
                "required": ["grants"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Dispatch MCP tool calls."""

    # ── search_grants ─────────────────────────────────────────────────────
    if name == "search_grants":
        query       = arguments.get("query", "").strip()
        focus_area  = arguments.get("focus_area", "All")
        agency      = arguments.get("agency", "")
        max_results = int(arguments.get("max_results", 20))
        min_score   = float(arguments.get("min_relevance_score", 1))

        # Decide which queries to run
        if query:
            queries = [query]
        elif focus_area != "All":
            queries = [AREA_QUERIES[focus_area]]
        else:
            # Broad sweep: one query per focus area
            queries = list(AREA_QUERIES.values())

        # Collect unique grants (deduplicate by contract number / title)
        seen: set[str] = set()
        raw_grants: list[dict] = []
        api_errors: list[str] = []
        for q in queries:
            batch = await _sbir_awards(q, agency=agency, rows=max_results)
            for g in batch:
                if "_api_error" in g:
                    api_errors.append(g["_api_error"])
                    continue
                key = g.get("id") or g.get("title", "")
                if key and key not in seen:
                    seen.add(key)
                    raw_grants.append(g)

        # Score and filter
        scored = []
        for g in raw_grants:
            s = _score_grant(g.get("title", ""), g.get("abstract", ""), g.get("agency", ""))
            if focus_area != "All":
                # Require the requested focus area to have at least 1 hit
                if s["area_scores"].get(focus_area, 0) == 0:
                    continue
            if s["overall_score"] < min_score:
                continue
            scored.append({**g, **s})

        # Sort by overall_score desc, then award_year desc
        scored.sort(
            key=lambda x: (x["overall_score"], x.get("award_year", 0)),
            reverse=True,
        )
        scored = scored[:max_results]

        # Build response
        if not scored and api_errors:
            result = {
                "status": "api_error",
                "message": (
                    "Could not reach SBIR.gov API. "
                    "Ensure the server has internet access. "
                    f"Errors: {list(set(api_errors))}"
                ),
            }
        elif not scored:
            result = {
                "status": "no_results",
                "message": (
                    f"No grants found matching query='{query}', "
                    f"focus_area='{focus_area}', agency='{agency}'. "
                    "Try broadening the query or removing the agency filter."
                ),
            }
        else:
            result = {
                "status": "ok",
                "total_returned": len(scored),
                "focus_area_filter": focus_area,
                "agency_filter": agency or "All",
                **({"api_warnings": list(set(api_errors))} if api_errors else {}),
                "grants": [
                    {
                        "rank": i + 1,
                        "title": g["title"],
                        "source": g["source"],
                        "agency": g["agency"],
                        "phase": g["phase"],
                        "award_amount_usd": g.get("award_amount_usd", 0),
                        "award_year": g.get("award_year", ""),
                        "company": g.get("company", ""),
                        "contract_number": g.get("id", ""),
                        "overall_score": g["overall_score"],
                        "ranked_focus_areas": g["ranked_focus_areas"],
                        "recommendation": g["recommendation"],
                        "abstract_preview": g.get("abstract", "")[:300],
                    }
                    for i, g in enumerate(scored)
                ],
            }

        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

    # ── list_open_solicitations ───────────────────────────────────────────
    elif name == "list_open_solicitations":
        focus_area  = arguments.get("focus_area", "All")
        agency      = arguments.get("agency", "")
        max_results = int(arguments.get("max_results", 15))

        if focus_area != "All":
            queries = [AREA_QUERIES[focus_area]]
        else:
            queries = list(AREA_QUERIES.values())

        seen: set[str] = set()
        raw: list[dict] = []
        sol_errors: list[str] = []
        for q in queries:
            batch = await _sbir_solicitations(q, agency=agency, rows=max_results)
            for s in batch:
                if "_api_error" in s:
                    sol_errors.append(s["_api_error"])
                    continue
                key = s.get("solicitation_number") or s.get("title", "")
                if key and key not in seen:
                    seen.add(key)
                    raw.append(s)

        # Score and sort
        scored = []
        for s in raw:
            sc = _score_grant(s.get("title", ""), s.get("abstract", ""))
            if focus_area != "All" and sc["area_scores"].get(focus_area, 0) == 0:
                continue
            scored.append({**s, **sc})

        scored.sort(key=lambda x: x["overall_score"], reverse=True)
        scored = scored[:max_results]

        result = {
            "status": "ok" if scored else ("api_error" if sol_errors else "no_results"),
            "total_returned": len(scored),
            **({"api_errors": list(set(sol_errors))} if sol_errors else {}),
            "solicitations": [
                {
                    "rank": i + 1,
                    "title": s["title"],
                    "agency": s["agency"],
                    "program": s["program"],
                    "phase": s["phase"],
                    "open_date": s.get("open_date", ""),
                    "close_date": s.get("close_date", ""),
                    "url": s.get("url", ""),
                    "overall_score": s["overall_score"],
                    "ranked_focus_areas": s["ranked_focus_areas"],
                    "recommendation": s["recommendation"],
                }
                for i, s in enumerate(scored)
            ],
        }
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

    # ── get_grant_detail ──────────────────────────────────────────────────
    elif name == "get_grant_detail":
        contract_number = arguments["contract_number"].strip()

        async with httpx.AsyncClient(timeout=30) as client:
            try:
                r = await client.get(
                    f"{SBIR_API}/award",
                    params={"keyword": contract_number, "rows": 5},
                )
                r.raise_for_status()
                data = r.json()
            except Exception as exc:
                result = {"error": str(exc)}
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

        # Find exact match by contract number
        match = None
        for item in data.get("data", []):
            if item.get("contract", "") == contract_number:
                match = item
                break
        if not match and data.get("data"):
            match = data["data"][0]  # closest result

        if not match:
            result = {"error": f"Contract '{contract_number}' not found."}
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

        title    = match.get("award_title", "")
        abstract = match.get("abstract") or ""
        scoring  = _score_grant(title, abstract, match.get("agency", ""))

        result = {
            "contract_number": match.get("contract", ""),
            "title": title,
            "abstract": abstract,
            "agency": match.get("agency", ""),
            "program": match.get("program", ""),
            "phase": match.get("phase", ""),
            "award_amount_usd": match.get("award_amount", 0),
            "award_year": match.get("award_year", ""),
            "company": match.get("firm", ""),
            "company_location": (
                f"{match.get('city', '')}, {match.get('state', '')}".strip(", ")
            ),
            "pi_name": match.get("pi_name", ""),
            "topic_code": match.get("solicitation_topic_code", ""),
            "coharts_relevance": scoring,
        }
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

    # ── rank_by_relevance ────────────────────────────────────────────────
    elif name == "rank_by_relevance":
        grants   = arguments["grants"]
        min_score = float(arguments.get("min_score", 0))

        scored = []
        for g in grants:
            s = _score_grant(
                g.get("title", ""),
                g.get("abstract", ""),
                g.get("agency", ""),
            )
            if s["overall_score"] >= min_score:
                scored.append({**g, **s})

        scored.sort(key=lambda x: x["overall_score"], reverse=True)

        result = {
            "status": "ok",
            "total_scored": len(scored),
            "ranked_grants": [
                {
                    "rank": i + 1,
                    **g,
                }
                for i, g in enumerate(scored)
            ],
        }
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

    else:
        return [types.TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]


# ── Entry point ───────────────────────────────────────────────────────────────

async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
