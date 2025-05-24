#!/usr/bin/env python3
"""
Research Report Generator

This module provides functionality to generate comprehensive research reports
from saved markdown files with proper URL citations.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

from openai import AsyncOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResearchReportGenerator:
    """Generates comprehensive research reports from saved files with proper citations."""

    def __init__(self, openai_api_key: str):
        self.openai_client = AsyncOpenAI(api_key=openai_api_key)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    def _read_source_files(self, directory: str) -> List[Dict[str, str]]:
        """Read and organize content from all markdown files in a directory."""
        directory_path = Path(directory)

        if not directory_path.exists():
            raise FileNotFoundError(f"Directory '{directory}' does not exist.")

        # Read all markdown files in the directory
        md_files = list(directory_path.glob("*.md"))

        if not md_files:
            raise FileNotFoundError(f"No markdown files found in directory '{directory}'.")

        logger.info(f"Found {len(md_files)} files to analyze for research report")

        sources_content = []

        for file_path in md_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                # Extract URL from the content (should be at the end)
                url_match = re.search(r"Source URL: (https?://[^\s]+)", content)
                source_url = url_match.group(1) if url_match else "Unknown URL"

                # Clean content (remove the source URL line for processing)
                clean_content = re.sub(r"\nSource URL: https?://[^\s]+\s*$", "", content)

                sources_content.append(
                    {
                        "url": source_url,
                        "content": clean_content,
                        "file_path": str(file_path),
                        "file_name": file_path.name,
                    }
                )

            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")
                continue

        if not sources_content:
            raise ValueError(f"Could not read any content from files in directory '{directory}'.")

        return sources_content

    def _format_sources_for_prompt(self, sources: List[Dict[str, str]]) -> str:
        """Format sources for the LLM prompt."""
        sources_text = ""
        for i, source in enumerate(sources, 1):
            sources_text += f"\n--- SOURCE {i} ---\n"
            sources_text += f"URL: {source['url']}\n"
            sources_text += f"File: {source['file_name']}\n"
            sources_text += f"Content:\n{source['content']}\n"
        return sources_text

    def _create_research_prompt(self, query: str, sources_text: str) -> str:
        """Create the comprehensive research prompt for the LLM."""
        return f"""You are an expert research analyst. Based on the provided sources, write a comprehensive research report that answers the following query:

QUERY: "{query}"

INSTRUCTIONS:
1. Analyze all provided sources thoroughly
2. Write a well-structured, comprehensive response that directly addresses the query
3. Include specific facts, data, insights, and evidence from the sources
4. Use clear section headings to organize information
5. CRITICAL: Include precise URL citations for EVERY claim, fact, or piece of information
6. Use this citation format: [URL](URL) or reference "Source X" with the URL
7. Synthesize information across sources when possible
8. If sources contradict each other, acknowledge this explicitly
9. Include a summary/conclusion section
10. Make the report professional and well-researched

CITATION REQUIREMENTS:
- Every factual claim must be followed by a citation
- Use format: "According to [research study](https://example.com), meditation reduces stress by 30%"
- When referencing multiple sources: "Multiple studies [Source 1](URL1), [Source 2](URL2) confirm that..."
- Always include the full URL in citations

SOURCES:
{sources_text}

Write a comprehensive research report that thoroughly answers the query with proper citations:"""

    def _add_report_metadata(self, report: str, query: str, directory: str, sources_count: int) -> str:
        """Add header and footer metadata to the research report."""
        # Add header with metadata
        header = f"""# Research Report: {query}

**Generated from:** {Path(directory).name}
**Sources analyzed:** {sources_count} files
**Directory:** {directory}

---

"""

        return header + report

    def _add_sources_index(self, report: str, sources: List[Dict[str, str]]) -> str:
        """Add sources index at the end of the report."""
        sources_index = "\n\n---\n\n## Sources Index\n\n"
        for i, source in enumerate(sources, 1):
            sources_index += f"{i}. [{source['url']}]({source['url']}) - {source['file_name']}\n"

        return report + sources_index

    async def generate_report(
        self, query: str, directory: str, url_to_file_mapping: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Generate a comprehensive research report based on saved files in a directory.

        Args:
            query: The research question/query to answer
            directory: Path to directory containing saved markdown files
            url_to_file_mapping: Optional mapping of URLs to file paths for better citation

        Returns:
            A comprehensive research report with proper URL citations
        """
        try:
            # Read and organize source files
            sources_content = self._read_source_files(directory)

            # Format sources for the LLM prompt
            sources_text = self._format_sources_for_prompt(sources_content)

            # Create the research prompt
            prompt = self._create_research_prompt(query, sources_text)

            # Generate the report using LLM
            response = await self.openai_client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=10000,
            )

            report = response.choices[0].message.content
            if not report:
                raise ValueError("Could not generate research report from LLM.")

            # Add metadata and sources index
            report_with_metadata = self._add_report_metadata(report, query, directory, len(sources_content))
            final_report = self._add_sources_index(report_with_metadata, sources_content)

            logger.info(f"Successfully generated research report with {len(sources_content)} sources")
            return final_report

        except Exception as e:
            logger.error(f"Error generating research report: {e}")
            return f"Error generating research report: {str(e)}"

    async def save_report(self, report: str, output_path: str) -> str:
        """
        Save the research report to a file.

        Args:
            report: The generated research report
            output_path: Path where to save the report

        Returns:
            The path where the report was saved
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(report)

            logger.info(f"Research report saved to: {output_file}")
            return str(output_file)

        except Exception as e:
            logger.error(f"Error saving research report: {e}")
            raise

    async def generate_and_save_report(
        self,
        query: str,
        directory: str,
        output_path: Optional[str] = None,
        url_to_file_mapping: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Generate and save a research report in one step.

        Args:
            query: The research question/query to answer
            directory: Path to directory containing saved markdown files
            output_path: Where to save the report (defaults to directory/research_report.md)
            url_to_file_mapping: Optional mapping of URLs to file paths

        Returns:
            The path where the report was saved
        """
        # Generate the report
        report = await self.generate_report(query, directory, url_to_file_mapping)

        # Determine output path
        if output_path is None:
            output_path = f"{directory}/research_report.md"

        # Save the report
        return await self.save_report(report, output_path)
