#!/usr/bin/env python3
"""
Report-only example demonstrating how to generate research reports from existing data.

This example shows how to use the ResearchReportGenerator to create comprehensive
research reports from a directory of markdown files that contain saved research data.

Usage:
    python report_only.py

Requirements:
    - OPENAI_API_KEY environment variable set
    - Directory with markdown files containing research data
"""

import asyncio
import os
from pathlib import Path
from typing import Optional

from research_report import ResearchReportGenerator


async def generate_report_from_directory(directory: str, query: str, output_path: Optional[str] = None):
    """
    Generate a research report from markdown files in a directory.

    Args:
        directory: Path to directory containing markdown files with research data
        query: The research question to answer
        output_path: Optional custom output path for the report
    """
    # Check for API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        return None

    print(f"📁 Analyzing files in: {directory}")
    print(f"❓ Research Query: {query}")
    print("─" * 60)

    # Check if directory exists and has markdown files
    dir_path = Path(directory)
    if not dir_path.exists():
        print(f"❌ Error: Directory '{directory}' does not exist")
        return None

    md_files = list(dir_path.glob("*.md"))
    if not md_files:
        print(f"❌ Error: No markdown files found in '{directory}'")
        return None

    print(f"📄 Found {len(md_files)} markdown files to analyze")

    # Generate the research report
    async with ResearchReportGenerator(openai_api_key) as report_generator:
        try:
            print("🔄 Generating comprehensive research report...")

            # Generate and save the report
            report_path = await report_generator.generate_and_save_report(
                query=query, directory=directory, output_path=output_path
            )

            print("✅ Research report generated successfully!")
            print(f"📄 Report saved to: {report_path}")

            # Read and display a preview of the report
            with open(report_path, encoding="utf-8") as f:
                report_content = f.read()

            # Show preview (first 800 characters)
            if len(report_content) > 800:
                print("\n📖 Report Preview:")
                print("─" * 60)
                print(report_content[:800] + "...")
                print("─" * 60)
                print(f"💡 View the complete report in: {report_path}")
            else:
                print("\n📖 Complete Report:")
                print("─" * 60)
                print(report_content)
                print("─" * 60)

            return report_path

        except Exception as e:
            print(f"❌ Error generating report: {e}")
            return None


async def main():
    """Run the report-only example with sample data."""
    print("🔬 Research Report Generator - Report Only Example")
    print("=" * 60)

    # Example 1: Use IAPS AI Policy Strategy research data
    print("\n📋 Example 1: IAPS AI Policy Strategy Analysis")

    # Path to IAPS research data
    data_directory = "iaps_task/iaps_ai_policy_strategy"
    research_query = (
        "What is the Institute for AI Policy and Strategy (IAPS), what are their key research areas, "
        "programs, and policy initiatives in AI governance and safety?"
    )

    # Generate report with custom output path
    custom_output = f"{data_directory}/iaps_analysis_report.md"

    report_path = await generate_report_from_directory(
        directory=data_directory, query=research_query, output_path=custom_output
    )

    if report_path:
        print(f"\n🎯 Report successfully created: {report_path}")

        # Show file structure
        print(f"\n🗂️ File Structure in {data_directory}:")
        for file_path in Path(data_directory).glob("*.md"):
            file_size = file_path.stat().st_size
            print(f"   • {file_path.name} ({file_size:,} bytes)")

    print("\n" + "=" * 60)
    print("🔧 How to use with your own data:")
    print("   1. Create a directory with markdown files containing research data")
    print("   2. Each markdown file should end with 'Source URL: <url>'")
    print("   3. Call generate_report_from_directory() with your directory and query")
    print("   4. The report will be generated with proper citations")


async def custom_example():
    """
    Example showing how to use with custom directory and query.
    Uncomment and modify this function to use with your own data.
    """
    # Your custom parameters
    # your_directory = "path/to/your/research/files"
    # your_query = "Your research question here"
    # your_output = "path/to/output/report.md"  # Optional

    # report_path = await generate_report_from_directory(
    #     directory=your_directory,
    #     query=your_query,
    #     output_path=your_output  # Optional - will default to directory/research_report.md
    # )

    pass


if __name__ == "__main__":
    # Run the main example
    asyncio.run(main())

    # Uncomment to run your custom example:
    # asyncio.run(custom_example())
