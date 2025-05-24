#!/usr/bin/env python3
"""
Simple example showing independent usage of batch search and research report modules.
"""

import asyncio
import os

from batch_search import BatchSearcher
from research_report import ResearchReportGenerator


async def example_batch_search_only():
    """Example of using just the batch search functionality."""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    exa_api_key = os.getenv("EXA_API_KEY")

    if not openai_api_key or not exa_api_key:
        print("❌ Missing API keys")
        return None

    async with BatchSearcher(openai_api_key, exa_api_key) as searcher:
        result = await searcher.batch_search(
            query="benefits of meditation",
            instruction="Find scientific evidence about meditation benefits",
            num_queries=3,
            max_results=5,
        )

        print("📋 BATCH SEARCH RESULTS:")
        print(result.results_summary)
        print(f"\n📁 Saved to: {result.output_directory}")
        print(f"🔗 URL mappings: {result.url_to_file_mapping}")

        return result


async def example_research_report_only():
    """Example of generating a research report from existing files."""
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        print("❌ Missing OPENAI_API_KEY")
        return

    # This assumes you have an existing directory with .md files
    directory = "search_results/meditation_benefits"  # Adjust as needed

    async with ResearchReportGenerator(openai_api_key) as generator:
        try:
            report = await generator.generate_report(
                query="What does research show about meditation's impact on mental health?", directory=directory
            )

            print("📝 RESEARCH REPORT GENERATED:")
            print(report[:300] + "..." if len(report) > 300 else report)

        except Exception as e:
            print(f"❌ Could not generate report: {e}")
            print("💡 Make sure the directory exists and contains .md files")


async def example_combined_workflow():
    """Example of the complete workflow: search + report."""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    exa_api_key = os.getenv("EXA_API_KEY")

    if not openai_api_key or not exa_api_key:
        print("❌ Missing API keys")
        return

    # Step 1: Batch search
    async with BatchSearcher(openai_api_key, exa_api_key) as searcher:
        batch_result = await searcher.batch_search(
            query="benefits of meditation", instruction="Find scientific evidence about meditation benefits"
        )

    # Step 2: Generate report
    async with ResearchReportGenerator(openai_api_key) as generator:
        report_path = await generator.generate_and_save_report(
            query="What are the proven benefits of meditation according to scientific research?",
            directory=batch_result.output_directory,
        )

        print("✅ Complete workflow finished!")
        print(f"📁 Files: {batch_result.output_directory}")
        print(f"📄 Report: {report_path}")


async def main():
    """Run examples based on what's available."""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    exa_api_key = os.getenv("EXA_API_KEY")

    if not openai_api_key:
        print("❌ OPENAI_API_KEY not set")
        return

    print("🔍 Simple Examples - Choose your workflow:")
    print("1. Batch search only")
    print("2. Research report only (requires existing files)")
    print("3. Combined workflow")

    choice = input("\nEnter choice (1-3): ").strip()

    if choice == "1":
        if not exa_api_key:
            print("❌ EXA_API_KEY needed for batch search")
            return
        await example_batch_search_only()
    elif choice == "2":
        await example_research_report_only()
    elif choice == "3":
        if not exa_api_key:
            print("❌ EXA_API_KEY needed for batch search")
            return
        await example_combined_workflow()
    else:
        print("❌ Invalid choice")


if __name__ == "__main__":
    asyncio.run(main())
