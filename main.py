#!/usr/bin/env python3
"""
Comprehensive example demonstrating two rounds of batch search + research report generation.
Round 1: Initial research on IAPS
Round 2: Deeper analysis based on Round 1 findings
Make sure to set your OPENAI_API_KEY and EXA_API_KEY environment variables.
"""

import asyncio
import os
from typing import Tuple

from batch_search import BatchSearcher
from research_report import ResearchReportGenerator


async def perform_search_and_report_round(
    openai_api_key: str,
    exa_api_key: str,
    query: str,
    instruction: str,
    research_query: str,
    round_name: str,
    base_output_dir: str = "iaps_task",
    num_queries: int = 6,
    max_results: int = 12,
) -> Tuple[str, str]:
    """
    Perform a single round of batch search followed by report generation.

    Args:
        openai_api_key: OpenAI API key
        exa_api_key: Exa API key
        query: Search query for batch search
        instruction: Search instruction for batch search
        research_query: Question for the research report
        round_name: Name of the round (e.g., "round1", "round2")
        base_output_dir: Base directory for outputs
        num_queries: Number of search queries to generate
        max_results: Maximum results to process

    Returns:
        Tuple of (report_content, round_directory_path)
    """
    round_dir = os.path.join(base_output_dir, round_name)

    print(f"\n{'='*80}")
    print(f"🔍 {round_name.upper()}: Starting batch search + report generation")
    print(f"📁 Round directory: {round_dir}")
    print(f"🔎 Search Query: {query}")
    print(f"📋 Search Instruction: {instruction}")
    print(f"❓ Research Question: {research_query}")

    # Step 1: Perform batch search
    print(f"\n🚀 {round_name.upper()} - Step 1: Performing batch search...")
    async with BatchSearcher(openai_api_key, exa_api_key, output_dir=round_dir) as searcher:
        batch_result = await searcher.batch_search(
            query=query,
            instruction=instruction,
            num_queries=num_queries,
            max_results=max_results,
        )

        print(batch_result.results_summary)

    # Step 2: Generate research report
    print(f"\n📝 {round_name.upper()} - Step 2: Generating research report...")
    async with ResearchReportGenerator(openai_api_key) as report_generator:
        report_path = await report_generator.generate_and_save_report(
            query=research_query,
            directory=batch_result.output_directory,
            url_to_file_mapping=batch_result.url_to_file_mapping,
        )

        print(f"📄 Report saved to: {report_path}")

        # Read the generated report
        with open(report_path, encoding="utf-8") as f:
            report_content = f.read()

    # Display round summary
    print(f"\n📊 {round_name.upper()} SUMMARY:")
    print(f"   • Directory: {batch_result.output_directory}")
    print(f"   • Files Created: {batch_result.saved_files_count}")
    print(f"   • URLs with Citations: {len(batch_result.url_to_file_mapping)}")
    print(f"   • Report: {os.path.basename(report_path)}")

    # Show report preview
    preview_length = 400
    if len(report_content) > preview_length:
        print(f"\n📖 Report Preview (first {preview_length} characters):")
        print("─" * 60)
        print(report_content[:preview_length] + "...")
        print("─" * 60)
    else:
        print("\n📖 Full Report:")
        print("─" * 60)
        print(report_content)
        print("─" * 60)

    return report_content, batch_result.output_directory


def create_second_round_query(first_report: str) -> Tuple[str, str, str]:
    """
    Create search query, instruction, and research question for the second round
    based on the first round report.

    Args:
        first_report: Content of the first round report

    Returns:
        Tuple of (search_query, search_instruction, research_question)
    """
    # Create a more focused search based on initial findings
    search_query = "IAPS Institute AI Policy Strategy detailed research publications policy analysis governance"

    search_instruction = f"""Based on this initial research about IAPS:

{first_report[:1000]}...

Find more detailed and specific information about:
1. Specific policy recommendations and white papers published by IAPS
2. Detailed analysis of their research methodologies and frameworks
3. Key partnerships and collaborations with other institutions
4. Specific case studies or policy implementations they've influenced
5. Technical details about their AI governance frameworks
6. Recent developments and current projects (2023-2024)
7. Funding sources and organizational structure
8. International relationships and policy influence"""

    research_question = """Based on the comprehensive initial research about IAPS, provide a deeper analytical report that answers:

1. What are the specific policy frameworks and methodologies that IAPS has developed for AI governance?
2. How does IAPS's approach to AI policy differ from other think tanks and policy institutions?
3. What concrete policy recommendations has IAPS made, and what has been their real-world impact?
4. What are the key research areas where IAPS is making unique contributions to AI policy?
5. How does IAPS engage with government, industry, and international bodies?
6. What are the current priority projects and future directions for IAPS?

Please provide specific examples, case studies, and concrete evidence wherever possible."""

    return search_query, search_instruction, research_question


async def main():
    """Run a comprehensive two-round research process on IAPS."""
    # Check for API keys
    openai_api_key = os.getenv("OPENAI_API_KEY")
    exa_api_key = os.getenv("EXA_API_KEY")

    if not openai_api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        return

    if not exa_api_key:
        print("❌ Error: EXA_API_KEY environment variable not set")
        return

    print("🔍 Starting comprehensive two-round IAPS research process...")
    print("📁 Results will be organized in ./iaps_task/round1/ and ./iaps_task/round2/")

    # ROUND 1: Initial comprehensive research on IAPS
    round1_query = "Institute for AI Policy and Strategy IAPS"
    round1_instruction = """Find comprehensive information about the Institute for AI Policy and Strategy (IAPS), including:
- Organization overview, mission, and founding
- Research areas and focus topics
- Team members, leadership, and key personnel
- Publications, papers, and research output
- Policy recommendations and positions
- Focus on AI security, compute policy, international strategy
- Role in AI governance and national security
- Recent activities and current projects"""

    round1_research_question = """What is the Institute for AI Policy and Strategy (IAPS)? Provide a comprehensive overview including:
- Mission, goals, and organizational structure
- Key research areas and methodologies
- Leadership team and key personnel
- Major publications and research contributions
- Policy positions and recommendations
- Role in AI governance and national security policy
- Current focus areas and recent developments"""

    # Perform Round 1
    first_report, round1_dir = await perform_search_and_report_round(
        openai_api_key=openai_api_key,
        exa_api_key=exa_api_key,
        query=round1_query,
        instruction=round1_instruction,
        research_query=round1_research_question,
        round_name="round1",
        num_queries=6,
        max_results=12,
    )

    # ROUND 2: Deeper analysis based on Round 1 findings
    round2_query, round2_instruction, round2_research_question = create_second_round_query(first_report)

    # Perform Round 2
    second_report, round2_dir = await perform_search_and_report_round(
        openai_api_key=openai_api_key,
        exa_api_key=exa_api_key,
        query=round2_query,
        instruction=round2_instruction,
        research_query=round2_research_question,
        round_name="round2",
        num_queries=8,  # More queries for deeper research
        max_results=15,  # More results for comprehensive analysis
    )

    # Final summary
    print(f"\n{'='*100}")
    print("🎯 FINAL RESEARCH SUMMARY")
    print(f"{'='*100}")
    print("✅ Two-round research process completed successfully!")
    print("\n📊 ROUND 1 (Initial Research):")
    print(f"   • Directory: {round1_dir}")
    print("   • Focus: Comprehensive IAPS overview")
    print(f"   • Report length: {len(first_report):,} characters")

    print("\n📊 ROUND 2 (Deep Analysis):")
    print(f"   • Directory: {round2_dir}")
    print("   • Focus: Detailed policy frameworks and impact analysis")
    print(f"   • Report length: {len(second_report):,} characters")

    print("\n🗂️ Complete research package available in:")
    print("   📁 ./iaps_task/round1/ - Initial comprehensive research")
    print("   📁 ./iaps_task/round2/ - Deep analytical follow-up")

    print("\n💡 Research Evolution:")
    print("   • Round 1 established foundational understanding of IAPS")
    print("   • Round 2 used Round 1 findings to pursue deeper policy analysis")
    print("   • Both rounds include source files + comprehensive reports with citations")


if __name__ == "__main__":
    asyncio.run(main())
