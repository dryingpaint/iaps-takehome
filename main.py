#!/usr/bin/env python3
"""
Simple example demonstrating the batch search functionality.
Make sure to set your OPENAI_API_KEY and EXA_API_KEY environment variables.
"""

import asyncio
import os

from batch_search import BatchSearcher


async def main():
    """Run a simple batch search example."""
    # Check for API keys
    openai_api_key = os.getenv("OPENAI_API_KEY")
    exa_api_key = os.getenv("EXA_API_KEY")

    if not openai_api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        return

    if not exa_api_key:
        print("❌ Error: EXA_API_KEY environment variable not set")
        return

    print("🔍 Starting batch search example...")
    print("📁 Results will be saved to ./example_results/ with query-specific folders")

    # Create searcher with custom output directory
    async with BatchSearcher(openai_api_key, exa_api_key, output_dir="example_results") as searcher:
        # Simple search example
        query = "benefits of meditation for mental health"
        instruction = "Find scientific research and evidence about meditation's positive effects on mental health, stress reduction, and well-being"

        print(f"\n🔎 Query: {query}")
        print(f"📋 Task: {instruction}")
        print("\n" + "=" * 60)

        result = await searcher.batch_search(
            query=query,
            instruction=instruction,
            num_queries=3,  # Generate 3 diverse search queries
            max_results=5,  # Process top 5 results (for quick demo)
        )

        print(result)

        # Show saved files in the query-specific folder
        if searcher.output_dir and searcher.output_dir.exists():
            saved_files = list(searcher.output_dir.glob("*.md"))
            if saved_files:
                print(f"\n📄 {len(saved_files)} files saved in '{searcher.output_dir}':")
                for file_path in saved_files:
                    print(f"   • {file_path.name}")

                print("\n💡 Tip: Each query gets its own folder to keep results organized!")
                print(f"    Check '{searcher.output_dir}' to see the extracted content!")
            else:
                print("\n❌ No files were saved (no relevant content found)")

        # Run another search to show multiple folders
        print(f"\n{'='*60}")
        print("🔍 Running second search to demonstrate folder organization...")

        query2 = "sustainable energy solutions"
        instruction2 = "Research renewable energy technologies and their environmental impact"

        result2 = await searcher.batch_search(
            query=query2,
            instruction=instruction2,
            num_queries=2,
            max_results=3,
        )

        print(f"\n📁 Second search saved to: {searcher.output_dir}")

        # Show all folders created
        if searcher.base_output_dir.exists():
            folders = [d for d in searcher.base_output_dir.iterdir() if d.is_dir()]
            if folders:
                print("\n📂 All query folders created:")
                for folder in folders:
                    file_count = len(list(folder.glob("*.md")))
                    print(f"   • {folder.name}/ ({file_count} files)")


if __name__ == "__main__":
    asyncio.run(main())
