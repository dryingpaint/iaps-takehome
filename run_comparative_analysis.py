#!/usr/bin/env python3
"""
Smart AI Policy Proposal Analysis Pipeline

1. Critically assess organizations for actual influence
2. Filter proposals to high-impact organizations only
3. Rank proposals comparatively to avoid score inflation
"""

import asyncio
import os
from pathlib import Path

from comparative_proposal_analyzer import run_comparative_analysis
from org_evaluator import evaluate_organizations_critically, identify_potentially_influential_orgs


def load_iaps_context() -> str:
    """Load IAPS context from research files."""
    context_parts = []

    iaps_files = [
        "iaps_task/iaps_ai_policy_strategy/research_report.md",
        "iaps_task/iaps_ai_policy_strategy/report_2.md",
    ]

    for file_path in iaps_files:
        if Path(file_path).exists():
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
                context_parts.append(f"=== {file_path} ===\n{content}\n")

    if not context_parts:
        print("❌ No IAPS research files found!")
        return ""

    return "\n".join(context_parts)


def extract_high_impact_org_names() -> list:
    """Extract list of high-impact organization names for filtering."""

    # Get the candidate organizations
    candidates = identify_potentially_influential_orgs("IFP AI Action Plan Database file.csv")

    # Flatten all organization names
    high_impact_orgs = []
    for category, org_list in candidates.items():
        if org_list:
            high_impact_orgs.extend(org_list)

    # Remove duplicates while preserving order
    seen = set()
    unique_orgs = []
    for org in high_impact_orgs:
        if org not in seen:
            seen.add(org)
            unique_orgs.append(org)

    return unique_orgs


async def run_complete_analysis():
    """Run the complete smart analysis pipeline."""

    # Check API keys
    openai_api_key = os.getenv("OPENAI_API_KEY")
    exa_api_key = os.getenv("EXA_API_KEY")

    if not openai_api_key:
        print("❌ OPENAI_API_KEY not set")
        return
    if not exa_api_key:
        print("❌ EXA_API_KEY not set")
        return

    # Load IAPS context
    print("📚 Loading IAPS context...")
    iaps_context = load_iaps_context()
    if not iaps_context:
        return
    print(f"✅ Loaded {len(iaps_context):,} characters of IAPS context")

    # Stage 1: Critical Organization Assessment (optional - can skip if already done)
    org_assessment_dir = "critical_org_assessment"
    if not Path(org_assessment_dir).exists():
        print("\n🔬 STAGE 1: Critical Organization Assessment")
        print("=" * 60)
        print("This will research organizations to verify their actual influence...")

        proceed = input("Run organization assessment? (y/N): ").strip().lower()
        if proceed in ["y", "yes"]:
            await evaluate_organizations_critically("IFP AI Action Plan Database file.csv")
        else:
            print("⏭️  Skipping organization assessment")
    else:
        print(f"✅ Organization assessment already exists in {org_assessment_dir}")

    # Stage 2: Get High-Impact Organizations
    print("\n📊 STAGE 2: Identifying High-Impact Organizations")
    print("=" * 60)

    high_impact_orgs = extract_high_impact_org_names()
    print(f"🎯 Selected {len(high_impact_orgs)} high-impact organizations:")

    # Group by type for display
    candidates = identify_potentially_influential_orgs("IFP AI Action Plan Database file.csv")
    for category, orgs in candidates.items():
        if orgs:
            print(f"   • {category}: {len(orgs)} orgs")

    # Stage 3: Comparative Proposal Analysis
    print("\n🔬 STAGE 3: Comparative Proposal Ranking")
    print("=" * 60)
    print("This will rank proposals using comparative analysis to avoid score inflation...")
    print(f"Analyzing proposals from {len(high_impact_orgs)} high-impact organizations")

    # Run the comparative analysis
    rankings = await run_comparative_analysis(
        csv_path="IFP AI Action Plan Database file.csv",
        iaps_context=iaps_context,
        openai_api_key=openai_api_key,
        high_impact_orgs=high_impact_orgs,
        top_n=50,
    )

    # Final Summary
    if rankings:
        print("\n" + "=" * 80)
        print("🎯 SMART ANALYSIS COMPLETE")
        print("=" * 80)

        scores = [r.composite_score for r in rankings]
        print("📊 Analysis Results:")
        print(f"   • Total proposals ranked: {len(rankings)}")
        print(f"   • Score range: {min(scores):.2f} - {max(scores):.2f}")
        print(f"   • Average score: {sum(scores)/len(scores):.2f}")

        print("\n🏆 TOP 10 HIGHEST-IMPACT PROPOSALS:")
        print("-" * 80)

        for i, ranking in enumerate(rankings[:10], 1):
            print(f"{i:2d}. {ranking.title}")
            print(f"    Organization: {ranking.organization}")
            print(f"    Composite Score: {ranking.composite_score:.2f}/10")
            print(
                f"    (IAPS: {ranking.iaps_alignment:.1f}, Policy Specificity: {ranking.policy_specificity:.1f}, Political Viability: {ranking.political_viability:.1f})"
            )
            print()

        print("📁 Detailed results saved in 'comparative_analysis/' directory")

    else:
        print("❌ No rankings produced")


if __name__ == "__main__":
    print("🚀 Smart AI Policy Proposal Analysis")
    print("This pipeline uses critical organization assessment + comparative ranking")
    print("to identify the highest-impact, most feasible proposals aligned with IAPS.\n")

    asyncio.run(run_complete_analysis())
