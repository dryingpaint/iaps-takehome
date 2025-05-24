import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from openai import AsyncOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProposalScore:
    """Represents a proposal with its alignment score and analysis."""

    row_index: int
    proposal_id: Optional[str]
    title: str
    content: str
    alignment_score: float  # 0-10 scale
    reasoning: str
    key_alignments: List[str]
    potential_gaps: List[str]
    raw_data: Dict  # Original row data


@dataclass
class BatchAnalysisResult:
    """Results from analyzing a batch of proposals."""

    total_processed: int
    successful_analyses: int
    failed_analyses: int
    top_proposals: List[ProposalScore]
    all_scores: List[ProposalScore]
    analysis_summary: str


class CSVProposalAnalyzer:
    """Analyzes CSV proposals for alignment with IAPS AI policy framework."""

    def __init__(
        self, openai_api_key: str, iaps_context: str, batch_size: int = 10, output_dir: str = "proposal_analysis"
    ):
        self.openai_client = AsyncOpenAI(api_key=openai_api_key)
        self.iaps_context = iaps_context
        self.batch_size = batch_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Create the analysis prompt template
        self.analysis_prompt = self._create_analysis_prompt_template()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    def _create_analysis_prompt_template(self) -> str:
        """Create the prompt template for analyzing individual proposals."""
        base_template = """You are an expert AI policy analyst specializing in evaluating proposals for alignment with the Institute for AI Policy and Strategy (IAPS) framework.

IAPS CONTEXT AND FRAMEWORK:
{iaps_context}

ANALYSIS TASK:
Analyze the following proposal for alignment with IAPS's AI policy framework and priorities.

PROPOSAL TO ANALYZE:
Title: {{title}}
Content: {{content}}

ANALYSIS REQUIREMENTS:
1. Score the proposal's alignment with IAPS framework on a scale of 0-10 (where 10 = perfect alignment)
2. Provide clear reasoning for the score
3. Identify specific areas where the proposal aligns with IAPS priorities
4. Identify potential gaps or areas where the proposal could better align with IAPS framework
5. Consider IAPS's focus on: AI governance, compute policy, international strategy, national security, export controls, AI safety institutes

RESPONSE FORMAT - Return ONLY valid JSON in this exact format:
```json
{{{{
  "alignment_score": 7.5,
  "reasoning": "This proposal shows strong alignment with IAPS priorities because...",
  "key_alignments": ["Specific alignment point 1", "Specific alignment point 2"],
  "potential_gaps": ["Gap or improvement area 1", "Gap or improvement area 2"],
  "iaps_priority_areas": {{{{
    "ai_governance": 8,
    "compute_policy": 6, 
    "international_strategy": 5,
    "national_security": 7,
    "export_controls": 4,
    "ai_safety": 6
  }}}}
}}}}
```

IMPORTANT: Return only the JSON object above, no additional text, explanations, or formatting."""

        return base_template.format(iaps_context=self.iaps_context)

    async def analyze_single_proposal(
        self, proposal_data: Dict, title_column: str, content_column: str, row_index: int
    ) -> Optional[ProposalScore]:
        """Analyze a single proposal for IAPS alignment."""

        try:
            title = str(proposal_data.get(title_column, "")).strip()
            content = str(proposal_data.get(content_column, "")).strip()

            if not title and not content:
                logger.warning(f"Row {row_index}: Empty title and content")
                return None

            logger.debug(f"Row {row_index}: Analyzing proposal - Title: {title[:50]}...")

            # Create the analysis prompt
            prompt = self.analysis_prompt.format(title=title, content=content)

            logger.debug(f"Row {row_index}: Calling OpenAI API...")

            # Call OpenAI API
            response = await self.openai_client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000,
            )

            response_content = response.choices[0].message.content
            if not response_content:
                logger.error(f"Row {row_index}: Empty response from OpenAI")
                return None

            logger.debug(f"Row {row_index}: Received response, length: {len(response_content)}")

            # Parse JSON response
            try:
                # Clean the response content to ensure it's valid JSON
                cleaned_response = response_content.strip()

                # Remove any markdown code blocks if present
                if cleaned_response.startswith("```"):
                    lines = cleaned_response.split("\n")
                    # Skip the first line if it's ```json or ```
                    start_idx = 1
                    # Remove the last line if it's ```
                    end_idx = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
                    cleaned_response = "\n".join(lines[start_idx:end_idx]).strip()

                # Try to find JSON in the response if it's still mixed with other text
                if not cleaned_response.startswith("{"):
                    start = cleaned_response.find("{")
                    end = cleaned_response.rfind("}") + 1
                    if start != -1 and end > start:
                        cleaned_response = cleaned_response[start:end]

                logger.debug(f"Row {row_index}: Attempting to parse JSON...")
                analysis = json.loads(cleaned_response)

                # Validate required fields
                if "alignment_score" not in analysis:
                    logger.error(f"Row {row_index}: Missing alignment_score in response")
                    return None

            except json.JSONDecodeError as e:
                logger.error(f"Row {row_index}: Failed to parse JSON response: {e}")
                logger.error(f"Row {row_index}: Original response: {response_content[:300]}...")
                logger.error(f"Row {row_index}: Cleaned response: {cleaned_response[:200]}...")

                # Try a fallback: extract score from text if possible
                try:
                    import re

                    score_match = re.search(r'"alignment_score":\s*(\d+(?:\.\d+)?)', response_content)
                    if score_match:
                        score = float(score_match.group(1))
                        # Create minimal valid analysis
                        analysis = {
                            "alignment_score": score,
                            "reasoning": "Analysis failed to parse properly",
                            "key_alignments": [],
                            "potential_gaps": [],
                        }
                        logger.warning(f"Row {row_index}: Used fallback parsing, score: {score}")
                    else:
                        logger.error(f"Row {row_index}: No alignment score found in fallback parsing")
                        return None
                except Exception as fallback_error:
                    logger.error(f"Row {row_index}: Fallback parsing also failed: {fallback_error}")
                    return None

            # Extract proposal ID if available
            proposal_id = proposal_data.get("id") or proposal_data.get("proposal_id") or str(row_index)

            logger.debug(f"Row {row_index}: Successfully analyzed proposal, score: {analysis.get('alignment_score')}")

            return ProposalScore(
                row_index=row_index,
                proposal_id=str(proposal_id),
                title=title,
                content=content,
                alignment_score=float(analysis.get("alignment_score", 0)),
                reasoning=analysis.get("reasoning", ""),
                key_alignments=analysis.get("key_alignments", []),
                potential_gaps=analysis.get("potential_gaps", []),
                raw_data=proposal_data,
            )

        except Exception as e:
            logger.error(f"Row {row_index}: Error analyzing proposal: {e}")
            logger.error(f"Row {row_index}: Exception type: {type(e)}")
            logger.error(f"Row {row_index}: Exception details: {str(e)}")
            return None

    async def analyze_batch(
        self, batch_data: List[Tuple[int, Dict]], title_column: str, content_column: str
    ) -> List[ProposalScore]:
        """Analyze a batch of proposals in parallel."""

        tasks = [
            self.analyze_single_proposal(data, title_column, content_column, row_idx) for row_idx, data in batch_data
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out None results and exceptions
        successful_results = []
        for result in results:
            if isinstance(result, ProposalScore):
                successful_results.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Exception in batch analysis: {result}")

        return successful_results

    def load_csv(self, csv_path: str) -> pd.DataFrame:
        """Load CSV file into a pandas DataFrame."""
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded CSV with {len(df)} rows and columns: {list(df.columns)}")
            return df
        except Exception as e:
            logger.error(f"Error loading CSV file {csv_path}: {e}")
            raise

    async def analyze_csv(
        self,
        csv_path: str,
        title_column: str,
        content_column: str,
        top_n: int = 10,
        start_row: int = 0,
        max_rows: Optional[int] = None,
    ) -> BatchAnalysisResult:
        """
        Analyze proposals in a CSV file for IAPS alignment.

        Args:
            csv_path: Path to the CSV file
            title_column: Name of the column containing proposal titles
            content_column: Name of the column containing proposal content
            top_n: Number of top proposals to return
            start_row: Row index to start analysis from
            max_rows: Maximum number of rows to process (None for all)
        """

        # Load CSV
        df = self.load_csv(csv_path)

        # Determine the range of rows to process
        end_row = len(df) if max_rows is None else min(start_row + max_rows, len(df))
        df_subset = df.iloc[start_row:end_row]

        logger.info(f"Analyzing rows {start_row} to {end_row-1} ({len(df_subset)} total rows)")

        # Validate columns exist
        if title_column not in df.columns:
            raise ValueError(f"Title column '{title_column}' not found in CSV. Available columns: {list(df.columns)}")
        if content_column not in df.columns:
            raise ValueError(
                f"Content column '{content_column}' not found in CSV. Available columns: {list(df.columns)}"
            )

        all_scores = []
        total_processed = 0
        successful_analyses = 0
        failed_analyses = 0

        # Process in batches
        for i in range(0, len(df_subset), self.batch_size):
            batch_end = min(i + self.batch_size, len(df_subset))
            batch_df = df_subset.iloc[i:batch_end]

            logger.info(
                f"Processing batch {i//self.batch_size + 1}: rows {start_row + i} to {start_row + batch_end - 1}"
            )

            # Prepare batch data
            batch_data = [(start_row + idx, row.to_dict()) for idx, (_, row) in enumerate(batch_df.iterrows(), i)]

            # Analyze batch
            batch_results = await self.analyze_batch(batch_data, title_column, content_column)

            all_scores.extend(batch_results)
            total_processed += len(batch_data)
            successful_analyses += len(batch_results)
            failed_analyses += len(batch_data) - len(batch_results)

            # Log progress
            logger.info(f"Batch complete: {len(batch_results)}/{len(batch_data)} successful analyses")

            # Small delay to avoid rate limiting
            await asyncio.sleep(1)

        # Sort by alignment score and get top N
        all_scores.sort(key=lambda x: x.alignment_score, reverse=True)
        top_proposals = all_scores[:top_n]

        # Generate summary
        analysis_summary = await self._generate_analysis_summary(
            total_processed, successful_analyses, failed_analyses, top_proposals
        )

        return BatchAnalysisResult(
            total_processed=total_processed,
            successful_analyses=successful_analyses,
            failed_analyses=failed_analyses,
            top_proposals=top_proposals,
            all_scores=all_scores,
            analysis_summary=analysis_summary,
        )

    async def _generate_analysis_summary(
        self, total_processed: int, successful: int, failed: int, top_proposals: List[ProposalScore]
    ) -> str:
        """Generate a summary of the analysis results."""

        if not top_proposals:
            return f"Analysis complete: {successful}/{total_processed} proposals analyzed successfully. No high-scoring proposals found."

        avg_score = sum(p.alignment_score for p in top_proposals) / len(top_proposals)

        summary = f"""IAPS Proposal Alignment Analysis Summary
===============================================

Processing Results:
- Total proposals processed: {total_processed}
- Successful analyses: {successful}
- Failed analyses: {failed}
- Success rate: {(successful/total_processed)*100:.1f}%

Top {len(top_proposals)} Proposals:
- Average alignment score: {avg_score:.2f}/10
- Highest score: {max(p.alignment_score for p in top_proposals):.2f}/10
- Lowest in top {len(top_proposals)}: {min(p.alignment_score for p in top_proposals):.2f}/10

Top 3 Proposals:"""

        for i, proposal in enumerate(top_proposals[:3], 1):
            summary += f"""

{i}. {proposal.title[:100]}{'...' if len(proposal.title) > 100 else ''}
   Score: {proposal.alignment_score:.2f}/10
   Key alignments: {', '.join(proposal.key_alignments[:3])}"""

        return summary

    async def save_results(
        self, results: BatchAnalysisResult, output_filename: str = "iaps_alignment_analysis"
    ) -> Dict[str, str]:
        """Save analysis results to files."""

        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{output_filename}_{timestamp}"

        saved_files = {}

        # Save summary report
        summary_path = self.output_dir / f"{base_filename}_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(results.analysis_summary)
        saved_files["summary"] = str(summary_path)

        # Save detailed results as JSON
        json_path = self.output_dir / f"{base_filename}_detailed.json"
        detailed_results = {
            "metadata": {
                "total_processed": results.total_processed,
                "successful_analyses": results.successful_analyses,
                "failed_analyses": results.failed_analyses,
                "timestamp": timestamp,
            },
            "top_proposals": [
                {
                    "row_index": p.row_index,
                    "proposal_id": p.proposal_id,
                    "title": p.title,
                    "alignment_score": p.alignment_score,
                    "reasoning": p.reasoning,
                    "key_alignments": p.key_alignments,
                    "potential_gaps": p.potential_gaps,
                }
                for p in results.top_proposals
            ],
            "all_scores": [
                {
                    "row_index": p.row_index,
                    "proposal_id": p.proposal_id,
                    "title": p.title,
                    "alignment_score": p.alignment_score,
                }
                for p in results.all_scores
            ],
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        saved_files["detailed_json"] = str(json_path)

        # Save top proposals as CSV
        csv_path = self.output_dir / f"{base_filename}_top_proposals.csv"
        top_proposals_data = []
        for p in results.top_proposals:
            row_data = {
                "row_index": p.row_index,
                "proposal_id": p.proposal_id,
                "title": p.title,
                "alignment_score": p.alignment_score,
                "reasoning": p.reasoning,
                "key_alignments": "; ".join(p.key_alignments),
                "potential_gaps": "; ".join(p.potential_gaps),
            }
            # Add original data columns
            for key, value in p.raw_data.items():
                if key not in row_data:
                    row_data[f"original_{key}"] = value
            top_proposals_data.append(row_data)

        if top_proposals_data:
            pd.DataFrame(top_proposals_data).to_csv(csv_path, index=False)
            saved_files["top_proposals_csv"] = str(csv_path)

        logger.info(f"Results saved to {len(saved_files)} files in {self.output_dir}")
        return saved_files


async def analyze_proposals_for_iaps_alignment(
    csv_path: str,
    iaps_context: str,
    openai_api_key: str,
    title_column: str = "title",
    content_column: str = "description",
    top_n: int = 10,
    batch_size: int = 10,
    start_row: int = 0,
    max_rows: Optional[int] = None,
    output_dir: str = "proposal_analysis",
) -> BatchAnalysisResult:
    """
    Convenience function to analyze proposals for IAPS alignment.

    Args:
        csv_path: Path to CSV file containing proposals
        iaps_context: Context about IAPS framework from research
        openai_api_key: OpenAI API key
        title_column: Column name containing proposal titles
        content_column: Column name containing proposal content/descriptions
        top_n: Number of top-scoring proposals to return
        batch_size: Number of proposals to process in parallel
        start_row: Row index to start processing from
        max_rows: Maximum number of rows to process
        output_dir: Directory to save results
    """

    async with CSVProposalAnalyzer(
        openai_api_key=openai_api_key, iaps_context=iaps_context, batch_size=batch_size, output_dir=output_dir
    ) as analyzer:
        results = await analyzer.analyze_csv(
            csv_path=csv_path,
            title_column=title_column,
            content_column=content_column,
            top_n=top_n,
            start_row=start_row,
            max_rows=max_rows,
        )

        # Save results
        saved_files = await analyzer.save_results(results)

        # Print summary
        print("\n" + "=" * 80)
        print("🎯 IAPS PROPOSAL ALIGNMENT ANALYSIS COMPLETE")
        print("=" * 80)
        print(results.analysis_summary)
        print("\n📁 Files saved:")
        for file_type, path in saved_files.items():
            print(f"   • {file_type}: {path}")

        return results
