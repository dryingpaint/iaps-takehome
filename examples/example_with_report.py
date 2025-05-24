#!/usr/bin/env python3
"""
Comprehensive example demonstrating batch search + research report generation.
Make sure to set your OPENAI_API_KEY and EXA_API_KEY environment variables.
"""

import asyncio
import os

from batch_search import BatchSearcher
from research_report import ResearchReportGenerator


async def main():
    """Run a comprehensive example with both batch search and report generation."""
    # Check for API keys
    openai_api_key = os.getenv("OPENAI_API_KEY")
    exa_api_key = os.getenv("EXA_API_KEY")

    if not openai_api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        return

    if not exa_api_key:
        print("❌ Error: EXA_API_KEY environment variable not set")
        return

    print("🔍 Starting comprehensive batch search + research report example...")
    print("📁 Results will be saved to ./research_example/")

    # Create searcher with custom output directory
    async with BatchSearcher(openai_api_key, exa_api_key, output_dir="research_example") as searcher:
        # Define our research topic
        query = "benefits and risks of intermittent fasting"
        instruction = "Find scientific research about intermittent fasting, including health benefits, potential risks, different methods, and clinical study results"

        print(f"\n🔎 Research Query: {query}")
        print(f"📋 Research Task: {instruction}")
        print("\n" + "=" * 80)

        # Step 1: Perform batch search
        print("🚀 Step 1: Performing batch search...")
        batch_result = await searcher.batch_search(
            query=query,
            instruction=instruction,
            num_queries=4,  # Generate 4 diverse search queries
            max_results=8,  # Process top 8 results
        )

        # Display search results summary
        print(batch_result.results_summary)

        # Step 2: Generate comprehensive research report using separate module
        print(f"\n{'='*80}")
        print("📝 Step 2: Generating comprehensive research report...")
        print("🔗 This will include detailed analysis with URL citations")

        # Create research report generator
        async with ResearchReportGenerator(openai_api_key) as report_generator:
            # Define the research question for the report
            research_query = "What are the scientifically proven benefits and risks of intermittent fasting, and what do clinical studies show about its effectiveness?"

            # Generate the research report
            report_path = await report_generator.generate_and_save_report(
                query=research_query,
                directory=batch_result.output_directory,
                url_to_file_mapping=batch_result.url_to_file_mapping,
            )

            print(f"📄 Research report saved to: {report_path}")
            print(f"🎯 Report answers: {research_query}")

            # Read and display report preview
            with open(report_path, encoding="utf-8") as f:
                report_content = f.read()

        # Show summary of what was generated
        print(f"\n{'='*80}")
        print("📊 SUMMARY:")
        print(f"   • Search Directory: {batch_result.output_directory}")
        print(f"   • Files Created: {batch_result.saved_files_count}")
        print(f"   • URLs with Citations: {len(batch_result.url_to_file_mapping)}")
        print(f"   • Research Report: {os.path.basename(report_path)}")

        # Show the beginning of the report
        if len(report_content) > 500:
            print("\n📖 Report Preview (first 500 characters):")
            print("─" * 60)
            print(report_content[:500] + "...")
            print("─" * 60)
            print(f"💡 View the full report in: {report_path}")
        else:
            print("\n📖 Full Report:")
            print("─" * 60)
            print(report_content)
            print("─" * 60)

        # Show file organization
        if batch_result.url_to_file_mapping:
            print("\n🗂️ File Organization:")
            for url, file_path in batch_result.url_to_file_mapping.items():
                print(f"   • {os.path.basename(file_path)} → {url}")

        print(f"\n✅ Complete! Your research is organized in: {batch_result.output_directory}")
        print("   📝 Individual source files + comprehensive research report with citations")


if __name__ == "__main__":
    asyncio.run(main())
