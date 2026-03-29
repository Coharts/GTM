#!/usr/bin/env python3
"""
SBIR Grant Reviewer Agent using OTA (Other Transaction Authority)
Reviews SBIR grants for relevance to:
  - Clinical Development
  - AI-Enabled Biological Advancements
Outputs a structured OTA summary report.
"""

import json
import os
import sys
import requests
import anthropic
from datetime import datetime
from typing import Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SBIR_API_BASE = "https://api.sbir.gov/public/api"

# Keywords that signal Clinical Development relevance
CLINICAL_KEYWORDS = [
    "clinical development", "clinical trial", "clinical study", "phase i trial",
    "phase ii trial", "phase iii trial", "drug development", "therapeutics",
    "pharmaceutical", "fda approval", "ind application", "clinical stage",
    "patient outcomes", "medical device", "diagnostic", "biomarker",
    "treatment", "therapy", "in vivo", "preclinical", "translational",
]

# Keywords that signal AI-Enabled Biological Advancement relevance
AI_BIO_KEYWORDS = [
    "artificial intelligence", "machine learning", "deep learning",
    "neural network", "natural language processing", "computer vision",
    "bioinformatics", "genomics", "proteomics", "transcriptomics",
    "drug discovery", "precision medicine", "digital health",
    "ai-enabled", "computational biology", "systems biology",
    "ai diagnostics", "predictive modeling", "biological data",
    "ai drug", "large language model", "foundation model",
    "multiomics", "single cell", "spatial transcriptomics",
]

# DoD components eligible for OTA (Other Transaction Authority) mechanisms
DOD_OTA_AGENCIES = {
    "DARPA", "Army", "Navy", "Air Force", "DoD", "SOCOM",
    "DHA", "TATRC", "USAMRDC", "AFWERX", "NavalX",
}

# ---------------------------------------------------------------------------
# Tool definitions passed to Claude
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "search_sbir_awards",
        "description": (
            "Search SBIR.gov for awarded grants using a keyword. "
            "Returns grant titles, abstracts, award amounts, agencies, and phases."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "Search keyword (e.g. 'clinical trial AI', 'machine learning genomics')",
                },
                "agency": {
                    "type": "string",
                    "description": (
                        "Optional agency filter (e.g. 'DoD', 'NIH', 'DARPA', 'NSF'). "
                        "Leave empty to search all agencies."
                    ),
                },
                "rows": {
                    "type": "integer",
                    "description": "Number of results to return (1-25). Default: 15.",
                },
            },
            "required": ["keyword"],
        },
    },
    {
        "name": "search_sbir_solicitations",
        "description": (
            "Search SBIR.gov for open solicitations (funding opportunities). "
            "Returns solicitation titles, agencies, deadlines, and descriptions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "Search keyword for solicitations.",
                },
                "agency": {
                    "type": "string",
                    "description": "Optional agency filter. Leave empty to search all agencies.",
                },
                "rows": {
                    "type": "integer",
                    "description": "Number of results to return (1-25). Default: 10.",
                },
            },
            "required": ["keyword"],
        },
    },
    {
        "name": "assess_ota_relevance",
        "description": (
            "Score a single grant for OTA (Other Transaction Authority) relevance "
            "across Clinical Development and AI-Enabled Biological Advancement dimensions. "
            "Returns detailed relevance scores and an OTA pathway recommendation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "grant_title": {
                    "type": "string",
                    "description": "Title of the grant.",
                },
                "grant_abstract": {
                    "type": "string",
                    "description": "Abstract or description of the grant.",
                },
                "agency": {
                    "type": "string",
                    "description": "Funding agency name (used to determine DoD OTA eligibility).",
                },
                "award_amount": {
                    "type": "number",
                    "description": "Award amount in USD.",
                },
                "phase": {
                    "type": "string",
                    "description": "SBIR phase (Phase I, Phase II, Phase IIB, etc.).",
                },
            },
            "required": ["grant_title", "grant_abstract"],
        },
    },
    {
        "name": "generate_ota_summary_report",
        "description": (
            "Compile all assessed grants into a final OTA summary report. "
            "Ranks grants by relevance, groups by category, and produces "
            "actionable GTM recommendations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "assessed_grants": {
                    "type": "array",
                    "description": "List of grants with their OTA relevance assessments.",
                    "items": {"type": "object"},
                },
                "open_solicitations": {
                    "type": "array",
                    "description": "List of open solicitations found during the review.",
                    "items": {"type": "object"},
                },
            },
            "required": ["assessed_grants"],
        },
    },
]

# ---------------------------------------------------------------------------
# Tool implementation functions
# ---------------------------------------------------------------------------


def search_sbir_awards(keyword: str, agency: str = "", rows: int = 15) -> dict:
    """Query SBIR.gov awards endpoint."""
    params: dict[str, Any] = {
        "keyword": keyword,
        "rows": max(1, min(rows, 25)),
        "start": 0,
    }
    if agency:
        params["agency"] = agency

    try:
        resp = requests.get(f"{SBIR_API_BASE}/award", params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        grants = []
        for item in data.get("data", []):
            grants.append({
                "title": item.get("award_title", ""),
                "abstract": (item.get("abstract") or "")[:600],
                "agency": item.get("agency", ""),
                "program": item.get("program", ""),
                "phase": item.get("phase", ""),
                "award_amount": item.get("award_amount", 0),
                "award_year": item.get("award_year", ""),
                "company": item.get("firm", ""),
                "pi_name": item.get("pi_name", ""),
                "contract_number": item.get("contract", ""),
                "solicitation_topic_code": item.get("solicitation_topic_code", ""),
            })

        return {
            "total_found": data.get("total", 0),
            "grants": grants,
            "keyword_searched": keyword,
        }
    except requests.RequestException as exc:
        return {"error": str(exc), "grants": [], "total_found": 0}
    except (ValueError, KeyError) as exc:
        return {"error": f"Parsing error: {exc}", "grants": [], "total_found": 0}


def search_sbir_solicitations(keyword: str, agency: str = "", rows: int = 10) -> dict:
    """Query SBIR.gov solicitations endpoint."""
    params: dict[str, Any] = {
        "keyword": keyword,
        "rows": max(1, min(rows, 25)),
        "start": 0,
    }
    if agency:
        params["agency"] = agency

    try:
        resp = requests.get(f"{SBIR_API_BASE}/solicitation", params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        solicitations = []
        for item in data.get("data", []):
            solicitations.append({
                "title": item.get("solicitation_title", ""),
                "agency": item.get("agency", ""),
                "program": item.get("program", ""),
                "phase": item.get("phase", ""),
                "open_date": item.get("open_date", ""),
                "close_date": item.get("close_date", ""),
                "description": (item.get("description") or "")[:500],
                "solicitation_number": item.get("solicitation_number", ""),
                "url": item.get("solicitation_url", ""),
            })

        return {
            "total_found": data.get("total", 0),
            "solicitations": solicitations,
            "keyword_searched": keyword,
        }
    except requests.RequestException as exc:
        return {"error": str(exc), "solicitations": [], "total_found": 0}
    except (ValueError, KeyError) as exc:
        return {"error": f"Parsing error: {exc}", "solicitations": [], "total_found": 0}


def assess_ota_relevance(
    grant_title: str,
    grant_abstract: str,
    agency: str = "",
    award_amount: float = 0,
    phase: str = "",
) -> dict:
    """Score a grant's relevance to OTA Clinical Dev and AI Biology tracks."""
    combined = f"{grant_title} {grant_abstract}".lower()

    # Keyword hit counts
    clinical_hits = [kw for kw in CLINICAL_KEYWORDS if kw in combined]
    ai_bio_hits = [kw for kw in AI_BIO_KEYWORDS if kw in combined]

    clinical_score = len(clinical_hits)
    ai_bio_score = len(ai_bio_hits)

    # Primary category
    primary_category = (
        "Clinical Development"
        if clinical_score >= ai_bio_score
        else "AI-Enabled Biological Advancement"
    )

    # Overall relevance 0–10
    overall = round(min(10.0, (clinical_score + ai_bio_score) * 1.2), 1)

    # DoD OTA eligibility
    agency_upper = agency.upper()
    is_dod = any(a in agency_upper for a in DOD_OTA_AGENCIES)

    # Phase weight (Phase II / IIB preferred for OTA transition)
    phase_lower = phase.lower()
    is_phase_ii = "ii" in phase_lower

    # OTA recommendation
    if is_dod and overall >= 5 and is_phase_ii:
        ota_rec = "HIGH — Strong DoD OTA candidate; Phase II ready for transition"
    elif is_dod and overall >= 3:
        ota_rec = "MEDIUM — DoD OTA-eligible agency; build case for OTA transition"
    elif overall >= 5:
        ota_rec = "MEDIUM — Strong relevance; explore DARPA/DHA OTA partnership"
    elif overall >= 2:
        ota_rec = "LOW — Monitor; limited OTA alignment currently"
    else:
        ota_rec = "NOT RECOMMENDED — Insufficient alignment with OTA criteria"

    return {
        "title": grant_title,
        "agency": agency,
        "phase": phase,
        "award_amount_usd": award_amount,
        "clinical_score": clinical_score,
        "ai_bio_score": ai_bio_score,
        "overall_relevance_score": overall,
        "primary_category": primary_category,
        "is_dod_ota_eligible": is_dod,
        "is_phase_ii": is_phase_ii,
        "ota_recommendation": ota_rec,
        "top_clinical_keywords": clinical_hits[:6],
        "top_ai_bio_keywords": ai_bio_hits[:6],
    }


def generate_ota_summary_report(
    assessed_grants: list,
    open_solicitations: list | None = None,
) -> dict:
    """Compile a ranked OTA summary report from assessed grants."""
    open_solicitations = open_solicitations or []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

    # Sort by overall relevance (descending)
    ranked = sorted(
        assessed_grants,
        key=lambda g: g.get("overall_relevance_score", 0),
        reverse=True,
    )

    high = [g for g in ranked if g.get("overall_relevance_score", 0) >= 6]
    medium = [g for g in ranked if 3 <= g.get("overall_relevance_score", 0) < 6]
    low = [g for g in ranked if g.get("overall_relevance_score", 0) < 3]

    clinical_grants = [
        g for g in ranked if g.get("primary_category") == "Clinical Development"
    ]
    ai_bio_grants = [
        g for g in ranked
        if g.get("primary_category") == "AI-Enabled Biological Advancement"
    ]
    dod_ota_candidates = [g for g in ranked if g.get("is_dod_ota_eligible")]

    return {
        "report_title": "SBIR OTA Grant Review Summary",
        "review_focus": "Clinical Development | AI-Enabled Biological Advancements",
        "generated_at": now,
        "statistics": {
            "total_grants_reviewed": len(assessed_grants),
            "high_relevance": len(high),
            "medium_relevance": len(medium),
            "low_relevance": len(low),
            "clinical_development_grants": len(clinical_grants),
            "ai_bio_grants": len(ai_bio_grants),
            "dod_ota_candidates": len(dod_ota_candidates),
            "open_solicitations_found": len(open_solicitations),
        },
        "executive_summary": (
            f"Reviewed {len(assessed_grants)} SBIR grants for OTA alignment. "
            f"Identified {len(high)} HIGH, {len(medium)} MEDIUM, and {len(low)} LOW "
            f"relevance grants. {len(dod_ota_candidates)} are DoD OTA-eligible. "
            f"{len(clinical_grants)} target Clinical Development and "
            f"{len(ai_bio_grants)} target AI-Enabled Biological Advancements."
        ),
        "top_10_grants": ranked[:10],
        "dod_ota_candidates": dod_ota_candidates[:5],
        "clinical_development_highlights": clinical_grants[:5],
        "ai_bio_highlights": ai_bio_grants[:5],
        "open_solicitations": open_solicitations[:10],
        "gtm_recommendations": [
            f"Immediately engage the {len(high)} HIGH-relevance grantees for partnership/licensing discussions.",
            f"Track {len(medium)} MEDIUM-relevance grants for Phase II → OTA transition opportunities.",
            f"Prioritize DoD OTA outreach to {len(dod_ota_candidates)} eligible companies.",
            "Respond to open solicitations with aligned capabilities before deadlines.",
            "Monitor AI-Biology grants closely — sector is growing rapidly.",
        ],
    }


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------


def execute_tool(tool_name: str, tool_input: dict) -> Any:
    """Route a tool call to the appropriate Python function."""
    if tool_name == "search_sbir_awards":
        return search_sbir_awards(
            keyword=tool_input["keyword"],
            agency=tool_input.get("agency", ""),
            rows=tool_input.get("rows", 15),
        )
    if tool_name == "search_sbir_solicitations":
        return search_sbir_solicitations(
            keyword=tool_input["keyword"],
            agency=tool_input.get("agency", ""),
            rows=tool_input.get("rows", 10),
        )
    if tool_name == "assess_ota_relevance":
        return assess_ota_relevance(
            grant_title=tool_input["grant_title"],
            grant_abstract=tool_input["grant_abstract"],
            agency=tool_input.get("agency", ""),
            award_amount=tool_input.get("award_amount", 0),
            phase=tool_input.get("phase", ""),
        )
    if tool_name == "generate_ota_summary_report":
        return generate_ota_summary_report(
            assessed_grants=tool_input["assessed_grants"],
            open_solicitations=tool_input.get("open_solicitations", []),
        )
    return {"error": f"Unknown tool: {tool_name}"}


# ---------------------------------------------------------------------------
# Agent entry point
# ---------------------------------------------------------------------------


def run_sbir_reviewer_agent() -> str:
    """
    Run the SBIR OTA Grant Reviewer Agent.

    Conducts a multi-step agentic loop:
      1. Searches SBIR.gov for Clinical Development + AI Biology grants
      2. Assesses each grant's OTA pathway potential
      3. Generates a ranked, structured OTA summary report

    Returns the final summary as a formatted string.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("Error: ANTHROPIC_API_KEY environment variable is not set.")

    claude = anthropic.Anthropic(api_key=api_key)

    system_prompt = """You are a senior SBIR Grant Analyst specializing in Other Transaction \
Authority (OTA) mechanisms for the defense and health sectors.

Your mission is to identify SBIR grants with strong alignment to:
  1. Clinical Development — drug development, clinical trials, medical devices, \
diagnostics, FDA pathways, therapeutics
  2. AI-Enabled Biological Advancements — AI/ML applied to genomics, proteomics, \
drug discovery, bioinformatics, precision medicine, digital health

Workflow:
  Step 1 — Search SBIR awards using diverse keywords (minimum 6 searches covering \
both clinical and AI-bio domains). Include DoD-specific searches.
  Step 2 — Search open solicitations for upcoming opportunities.
  Step 3 — Assess OTA relevance for every unique grant discovered.
  Step 4 — Call generate_ota_summary_report with ALL assessed grants to produce \
the final report.
  Step 5 — Write a concise, executive-level OTA Summary (plain text) that a GTM \
executive can act on immediately. Include:
      • Executive Summary (3-5 sentences)
      • Top 5 HIGH-relevance grants (title, agency, phase, amount, why it matters)
      • Top 3 Open Solicitations (if any)
      • DoD OTA Opportunities (list companies + OTA recommendation)
      • GTM Action Items (numbered, specific)

Be thorough: search broadly before summarizing. Do NOT skip the assessment step."""

    user_message = """Conduct a comprehensive SBIR grant review for our GTM strategy.

Focus:
• Clinical Development (Phase II preferred, but include Phase I if highly relevant)
• AI-Enabled Biological Advancements (any phase)

Deliver a structured OTA Summary Report I can present to our executive team today."""

    messages = [{"role": "user", "content": user_message}]

    header = (
        "\n" + "=" * 70 + "\n"
        "  SBIR GRANT REVIEWER AGENT — OTA ANALYSIS\n"
        "=" * 70 + "\n"
        f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        "  Focus   : Clinical Development | AI-Enabled Biological Advancements\n"
        "  Mode    : OTA (Other Transaction Authority) Grant Review\n"
        + "=" * 70 + "\n"
    )
    print(header)

    # -----------------------------------------------------------------------
    # Agentic loop
    # -----------------------------------------------------------------------
    iteration = 0
    while True:
        iteration += 1
        response = claude.messages.create(
            model="claude-opus-4-6",
            max_tokens=16000,
            thinking={"type": "adaptive"},
            system=system_prompt,
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            final_text = "\n".join(
                block.text for block in response.content if block.type == "text"
            )
            divider = "\n" + "=" * 70 + "\n"
            print(divider + "  OTA GRANT REVIEW — FINAL REPORT" + divider)
            print(final_text)
            print(divider)
            return final_text

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    preview = json.dumps(block.input)[:120].rstrip()
                    print(f"  [Tool #{iteration}] {block.name}  →  {preview}…")
                    result = execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    })

            messages.append({"role": "user", "content": tool_results})
            continue

        # pause_turn — server-side loop needs to continue
        if response.stop_reason == "pause_turn":
            messages.append({"role": "assistant", "content": response.content})
            continue

        # Unexpected stop
        print(f"  [Warning] Unexpected stop_reason: {response.stop_reason}")
        break

    return "Agent stopped without producing a final summary."


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    report = run_sbir_reviewer_agent()

    # Persist report to disk
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"sbir_ota_review_{ts}.txt"
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(report)
    print(f"\n  Report saved → {output_path}\n")
