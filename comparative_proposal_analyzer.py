#!/usr/bin/env python3
"""
Critical comparative proposal analyzer using evidence-based criteria.
Ranks proposals against each other in batches using rigorous criteria to avoid score inflation
and identify truly actionable, specific policy recommendations.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from openai import AsyncOpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProposalRanking:
    """Represents a proposal with critical ranking scores."""

    row_index: int
    proposal_id: str
    title: str
    content: str
    organization: str
    org_type: str

    # Critical analysis scores (0-10 scale)
    iaps_alignment: float  # IAPS Strategic Alignment (30%)
    policy_specificity: float  # Policy Specificity & Actionability (25%)
    evidence_base: float  # Evidence-Based Justification (25%)
    political_viability: float  # Political Viability & Implementation (20%)

    # Derived metrics
    composite_score: float
    batch_rank: int  # Rank within batch (1 = best)
    reasoning: str

    raw_data: Dict


class ComparativeProposalAnalyzer:
    """Critical proposal analyzer that ranks proposals using evidence-based criteria and comparative ranking to avoid score inflation."""

    def __init__(self, openai_api_key: str, iaps_context: str, batch_size: int = 75):
        self.openai_client = AsyncOpenAI(api_key=openai_api_key)
        self.iaps_context = iaps_context
        self.batch_size = batch_size

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    def _create_comparative_prompt(self, proposals: List[Dict]) -> str:
        """Create prompt for critical comparative ranking of proposals."""

        # Format proposals for comparison
        proposals_text = ""
        for i, proposal in enumerate(proposals, 1):
            proposals_text += f"""
PROPOSAL {i}:
Title: {proposal['title']}
Organization: {proposal['organization']} ({proposal['org_type']})
Content: {proposal['content'][:800]}...
{"="*60}"""

        return f"""You are a CRITICAL AI policy analyst. Rank these {len(proposals)} proposals using evidence-based criteria. BE SKEPTICAL - most proposals are vague, self-serving, or impractical.

IAPS CONTEXT:
{self.iaps_context}

CRITICAL RANKING CRITERIA:

1. IAPS STRATEGIC ALIGNMENT (30%): Does this directly advance IAPS priorities?
   ✅ ACCEPT: Specific recommendations for AI governance frameworks, compute policy, international coordination, national security AI applications, export controls, AI safety institutes
   ❌ REJECT: Generic "AI ethics" talk, vague "responsible AI" without specifics, broad corporate wish lists

2. POLICY SPECIFICITY & ACTIONABILITY (25%): Are these concrete, implementable recommendations?
   ✅ ACCEPT: Specific regulatory text, agency authorities, budget allocations, enforcement mechanisms, measurable objectives
   ❌ REJECT: Vague aspirations ("increase collaboration"), undefined terms ("ethical AI"), hand-waving about "frameworks"

3. EVIDENCE-BASED JUSTIFICATION (25%): Is this grounded in real analysis and evidence?
   ✅ ACCEPT: Cites specific risks/gaps, references existing policy precedents, provides quantitative analysis, addresses real-world constraints
   ❌ REJECT: Unsupported claims, marketing speak, theoretical wishful thinking, ignores implementation realities

4. POLITICAL VIABILITY & IMPLEMENTATION FEASIBILITY (20%): Can this actually be implemented?
   ✅ ACCEPT: Works within existing agency authorities, considers Congressional dynamics, addresses stakeholder concerns, has realistic timelines
   ❌ REJECT: Requires impossible political consensus, ignores bureaucratic realities, unrealistic scope/timeline, politically naive

CRITICAL FILTERS - AUTOMATICALLY SCORE LOW (1-3):
• Corporate lobbying disguised as policy expertise (pushing company interests)
• Vague recommendations that mean nothing ("enhance coordination")
• Proposals that ignore existing government capabilities/constraints
• Academic theorizing without practical implementation path
• Recommendations that duplicate existing efforts without adding value
• Industry associations pushing deregulation as "innovation policy"

SCORE HIGH (7-10) ONLY FOR:
• Specific, actionable policy recommendations with clear implementation path
• Evidence-based analysis that identifies real gaps and proposes concrete solutions
• Proposals that demonstrate deep understanding of government capabilities and constraints
• Recommendations that directly advance U.S. competitive position in AI
• Clear articulation of how success would be measured

PROPOSALS TO RANK:
{proposals_text}

RANKING INSTRUCTIONS:
- Rank all {len(proposals)} proposals from 1 (best) to {len(proposals)} (worst)
- USE FULL SCORE RANGE: Top 20% = 7-10 points, Middle 60% = 4-7 points, Bottom 20% = 1-3 points
- BE CRITICAL: Most proposals are mediocre corporate lobbying or academic theorizing
- Favor CONCRETE RECOMMENDATIONS over vague aspirations
- Consider organization credibility but focus on proposal substance

REQUIRED JSON RESPONSE:
{{
  "rankings": [
    {{
      "proposal_number": 1,
      "rank": 1,
      "iaps_alignment": 9.0,
      "policy_specificity": 8.5,
      "evidence_base": 7.5,
      "political_viability": 8.0,
      "reasoning": "Specific recommendation for [X] with clear implementation path because [evidence]. Directly advances IAPS priority of [Y] through [concrete mechanism]."
    }},
    // ... continue for all {len(proposals)} proposals
  ]
}}

Be brutally honest. Most proposals deserve low scores. Return ONLY the JSON object."""

    async def rank_proposal_batch(self, proposals: List[Dict]) -> List[ProposalRanking]:
        """Rank a batch of proposals comparatively."""

        # Create comparative prompt
        prompt = self._create_comparative_prompt(proposals)

        # Get ranking from OpenAI
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )

            if not response or not response.choices or len(response.choices) == 0:
                logger.error("Invalid response from OpenAI")
                return []

            message = response.choices[0].message
            if not message or not message.content:
                logger.error("Empty response from OpenAI")
                return []

            response_content = message.content

            # Clean and parse JSON
            cleaned_response = response_content.strip()
            if cleaned_response.startswith("```"):
                lines = cleaned_response.split("\n")
                start_idx = 1
                end_idx = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
                cleaned_response = "\n".join(lines[start_idx:end_idx]).strip()

            if not cleaned_response.startswith("{"):
                start = cleaned_response.find("{")
                end = cleaned_response.rfind("}") + 1
                if start != -1 and end > start:
                    cleaned_response = cleaned_response[start:end]

            rankings_data = json.loads(cleaned_response)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse ranking JSON: {e}")
            logger.error(f"Response: {response_content[:500]}...")
            return []
        except Exception as e:
            logger.error(f"Error getting rankings: {e}")
            return []

        # Convert to ProposalRanking objects
        ranked_proposals = []

        for ranking in rankings_data.get("rankings", []):
            try:
                proposal_idx = ranking["proposal_number"] - 1  # Convert to 0-based
                if proposal_idx >= len(proposals):
                    continue

                proposal = proposals[proposal_idx]

                # Calculate composite score with new criteria weights
                composite = (
                    0.30 * ranking["iaps_alignment"]
                    + 0.25 * ranking["policy_specificity"]
                    + 0.25 * ranking["evidence_base"]
                    + 0.20 * ranking["political_viability"]
                )

                ranked_proposal = ProposalRanking(
                    row_index=proposal["row_index"],
                    proposal_id=proposal["proposal_id"],
                    title=proposal["title"],
                    content=proposal["content"],
                    organization=proposal["organization"],
                    org_type=proposal["org_type"],
                    iaps_alignment=ranking["iaps_alignment"],
                    policy_specificity=ranking["policy_specificity"],
                    evidence_base=ranking["evidence_base"],
                    political_viability=ranking["political_viability"],
                    composite_score=composite,
                    batch_rank=ranking["rank"],
                    reasoning=ranking["reasoning"],
                    raw_data=proposal["raw_data"],
                )

                ranked_proposals.append(ranked_proposal)

            except Exception as e:
                logger.error(f"Error processing ranking: {e}")
                continue

        # Sort by rank to maintain order
        ranked_proposals.sort(key=lambda x: x.batch_rank)

        return ranked_proposals

    def load_and_prepare_proposals(self, csv_path: str, high_impact_orgs: List[str]) -> List[Dict]:
        """Load CSV and prepare proposals from high-impact organizations."""

        df = pd.read_csv(csv_path)

        # Filter to high-impact organizations only
        df_filtered = df[df["Organization"].isin(high_impact_orgs)].copy()

        logger.info(f"Filtered from {len(df)} to {len(df_filtered)} proposals from high-impact orgs")

        # Prepare proposal data
        proposals = []
        for idx, row in df_filtered.iterrows():
            proposal = {
                "row_index": idx,
                "proposal_id": str(row.get("id", idx)),
                "title": str(row.get("Recommendation", "")).strip(),
                "content": str(row.get("FullText", "")).strip(),
                "organization": str(row.get("Organization", "")),
                "org_type": str(row.get("OrgType", "")),
                "raw_data": row.to_dict(),
            }

            # Skip empty proposals
            if proposal["title"] and proposal["content"]:
                proposals.append(proposal)

        return proposals

    async def analyze_all_proposals(
        self, proposals: List[Dict], output_dir: str = "comparative_analysis"
    ) -> List[ProposalRanking]:
        """Analyze all proposals using comparative ranking, saving results after each batch."""

        Path(output_dir).mkdir(exist_ok=True)

        # Check for existing batch results and resume from where we left off
        existing_results, start_batch = await self.load_existing_results(output_dir, len(proposals))
        all_rankings = existing_results

        if start_batch > 1:
            logger.info(f"Found existing results for {start_batch-1} batches ({len(existing_results)} proposals)")
            logger.info(f"Resuming from batch {start_batch}")

        # Process remaining batches
        total_batches = (len(proposals) + self.batch_size - 1) // self.batch_size

        for batch_num in range(start_batch, total_batches + 1):
            i = (batch_num - 1) * self.batch_size
            batch_end = min(i + self.batch_size, len(proposals))
            batch = proposals[i:batch_end]

            logger.info(f"Processing batch {batch_num}/{total_batches}: proposals {i+1}-{batch_end}")

            try:
                batch_rankings = await self.rank_proposal_batch(batch)
                all_rankings.extend(batch_rankings)

                logger.info(f"Batch {batch_num} complete: {len(batch_rankings)}/{len(batch)} ranked")

                # Save batch results immediately
                await self.save_batch_results(batch_rankings, batch_num, output_dir)

                # Save cumulative results so far
                await self.save_cumulative_results(all_rankings, batch_num, output_dir)

            except Exception as e:
                logger.error(f"Error processing batch {batch_num}: {e}")
                # Continue with next batch rather than stopping completely
                continue

            # Rate limiting
            await asyncio.sleep(2)

        # Sort all proposals by composite score for final results
        all_rankings.sort(key=lambda x: x.composite_score, reverse=True)

        # Save final comprehensive results
        await self.save_final_results(all_rankings, output_dir)

        return all_rankings

    async def save_batch_results(self, batch_rankings: List[ProposalRanking], batch_num: int, output_dir: str):
        """Save results for a single batch."""
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

        # Save batch CSV
        batch_csv_path = Path(output_dir) / f"batch_{batch_num}_results_{timestamp}.csv"
        batch_data = []
        for r in batch_rankings:
            batch_data.append(
                {
                    "batch_rank": r.batch_rank,
                    "title": r.title,
                    "organization": r.organization,
                    "org_type": r.org_type,
                    "composite_score": r.composite_score,
                    "iaps_alignment": r.iaps_alignment,
                    "policy_specificity": r.policy_specificity,
                    "evidence_base": r.evidence_base,
                    "political_viability": r.political_viability,
                    "reasoning": r.reasoning,
                    "row_index": r.row_index,
                }
            )

        pd.DataFrame(batch_data).to_csv(batch_csv_path, index=False)
        logger.info(f"Batch {batch_num} results saved to: {batch_csv_path}")

    async def save_cumulative_results(self, all_rankings: List[ProposalRanking], batch_num: int, output_dir: str):
        """Save cumulative results after each batch."""
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

        # Sort by composite score for cumulative ranking
        sorted_rankings = sorted(all_rankings, key=lambda x: x.composite_score, reverse=True)

        # Save cumulative CSV
        cumulative_csv_path = Path(output_dir) / f"cumulative_results_through_batch_{batch_num}_{timestamp}.csv"
        cumulative_data = []
        for i, r in enumerate(sorted_rankings, 1):
            cumulative_data.append(
                {
                    "overall_rank": i,
                    "batch_rank": r.batch_rank,
                    "title": r.title,
                    "organization": r.organization,
                    "org_type": r.org_type,
                    "composite_score": r.composite_score,
                    "iaps_alignment": r.iaps_alignment,
                    "policy_specificity": r.policy_specificity,
                    "evidence_base": r.evidence_base,
                    "political_viability": r.political_viability,
                    "reasoning": r.reasoning,
                    "row_index": r.row_index,
                }
            )

        pd.DataFrame(cumulative_data).to_csv(cumulative_csv_path, index=False)
        logger.info(f"Cumulative results through batch {batch_num} saved to: {cumulative_csv_path}")

    async def save_final_results(self, rankings: List[ProposalRanking], output_dir: str):
        """Save final comprehensive results."""

        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

        # Save summary
        summary_path = Path(output_dir) / f"FINAL_comparative_ranking_summary_{timestamp}.md"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("# FINAL Comparative Proposal Ranking Results\n\n")
            f.write(f"**Total Proposals Ranked**: {len(rankings)}\n")
            f.write("**Analysis Method**: Critical comparative ranking in batches to avoid score inflation\n\n")

            if rankings:
                f.write("**Score Distribution**:\n")
                scores = [r.composite_score for r in rankings]
                f.write(f"- Highest: {max(scores):.2f}/10\n")
                f.write(f"- Average: {sum(scores)/len(scores):.2f}/10\n")
                f.write(f"- Lowest: {min(scores):.2f}/10\n\n")

                f.write("## Top 20 Proposals\n\n")
                for i, ranking in enumerate(rankings[:20], 1):
                    f.write(f"### {i}. {ranking.title}\n")
                    f.write(f"**Organization**: {ranking.organization}\n")
                    f.write(
                        f"**Scores**: IAPS: {ranking.iaps_alignment:.1f}, Policy Specificity: {ranking.policy_specificity:.1f}, Evidence Base: {ranking.evidence_base:.1f}, Political Viability: {ranking.political_viability:.1f}\n"
                    )
                    f.write(f"**Composite**: {ranking.composite_score:.2f}/10\n")
                    f.write(f"**Reasoning**: {ranking.reasoning}\n\n")

        # Save detailed CSV
        csv_path = Path(output_dir) / f"FINAL_comparative_ranking_detailed_{timestamp}.csv"
        ranking_data = []
        for r in rankings:
            ranking_data.append(
                {
                    "rank": rankings.index(r) + 1,
                    "title": r.title,
                    "organization": r.organization,
                    "org_type": r.org_type,
                    "composite_score": r.composite_score,
                    "iaps_alignment": r.iaps_alignment,
                    "policy_specificity": r.policy_specificity,
                    "evidence_base": r.evidence_base,
                    "political_viability": r.political_viability,
                    "reasoning": r.reasoning,
                    "row_index": r.row_index,
                }
            )

        pd.DataFrame(ranking_data).to_csv(csv_path, index=False)

        logger.info(f"FINAL results saved to {summary_path} and {csv_path}")

    async def load_existing_results(self, output_dir: str, total_proposals: int) -> Tuple[List[ProposalRanking], int]:
        """Load existing batch results and determine where to resume analysis."""

        output_path = Path(output_dir)
        if not output_path.exists():
            return [], 1

        # Find existing batch files
        batch_files = list(output_path.glob("batch_*_results_*.csv"))
        if not batch_files:
            logger.info("No existing batch results found. Starting from batch 1.")
            return [], 1

        # Extract batch numbers from filenames
        batch_numbers = []
        for batch_file in batch_files:
            try:
                # Extract batch number from filename like "batch_3_results_20241201_123456.csv"
                filename = batch_file.name
                if filename.startswith("batch_") and "_results_" in filename:
                    batch_num_str = filename.split("_")[1]
                    batch_numbers.append(int(batch_num_str))
            except (ValueError, IndexError):
                continue

        if not batch_numbers:
            logger.info("No valid batch result files found. Starting from batch 1.")
            return [], 1

        # Find the highest completed batch number
        max_completed_batch = max(batch_numbers)
        logger.info(f"Found completed batches: {sorted(batch_numbers)}")
        logger.info(f"Highest completed batch: {max_completed_batch}")

        # Load all existing batch results
        all_existing_rankings = []

        for batch_num in range(1, max_completed_batch + 1):
            if batch_num not in batch_numbers:
                logger.warning(f"Missing batch {batch_num} results. Will restart from batch {batch_num}")
                return all_existing_rankings, batch_num

            # Find the most recent file for this batch number
            batch_pattern = f"batch_{batch_num}_results_*.csv"
            batch_files_for_num = list(output_path.glob(batch_pattern))

            if not batch_files_for_num:
                logger.warning(f"No files found for batch {batch_num}. Will restart from batch {batch_num}")
                return all_existing_rankings, batch_num

            # Use the most recent file (sort by timestamp in filename)
            latest_batch_file = sorted(batch_files_for_num)[-1]

            try:
                # Load the batch results
                batch_df = pd.read_csv(latest_batch_file)

                # Convert back to ProposalRanking objects
                for _, row in batch_df.iterrows():
                    ranking = ProposalRanking(
                        row_index=row["row_index"],
                        proposal_id=str(row.get("proposal_id", row["row_index"])),
                        title=row["title"],
                        content="",  # Content not saved in batch files to save space
                        organization=row["organization"],
                        org_type=row["org_type"],
                        iaps_alignment=row["iaps_alignment"],
                        policy_specificity=row["policy_specificity"],
                        evidence_base=row["evidence_base"],
                        political_viability=row["political_viability"],
                        composite_score=row["composite_score"],
                        batch_rank=row["batch_rank"],
                        reasoning=row["reasoning"],
                        raw_data={},  # Raw data not saved in batch files
                    )
                    all_existing_rankings.append(ranking)

                logger.info(f"Loaded {len(batch_df)} results from batch {batch_num}")

            except Exception as e:
                logger.error(f"Error loading batch {batch_num} from {latest_batch_file}: {e}")
                logger.warning(f"Will restart from batch {batch_num}")
                return all_existing_rankings, batch_num

        # All batches loaded successfully, resume from next batch
        next_batch = max_completed_batch + 1
        total_batches = (total_proposals + self.batch_size - 1) // self.batch_size

        if next_batch > total_batches:
            logger.info(f"Analysis already complete! All {max_completed_batch} batches found.")
            return all_existing_rankings, next_batch

        return all_existing_rankings, next_batch


async def run_comparative_analysis(
    csv_path: str, iaps_context: str, openai_api_key: str, high_impact_orgs: List[str], top_n: int = 50
):
    """Run the complete comparative analysis."""

    print("🚀 Starting Comparative Proposal Analysis")
    print("=" * 60)

    async with ComparativeProposalAnalyzer(openai_api_key, iaps_context) as analyzer:
        # Load and prepare proposals
        print("📋 Loading proposals from high-impact organizations...")
        proposals = analyzer.load_and_prepare_proposals(csv_path, high_impact_orgs)
        print(f"📊 Found {len(proposals)} proposals to analyze")

        # Run comparative analysis
        print("\n🔬 Running comparative ranking analysis...")
        rankings = await analyzer.analyze_all_proposals(proposals)

        # Display results
        print(f"\n✅ Analysis complete! Ranked {len(rankings)} proposals")

        if rankings:
            print(f"\n🏆 TOP {min(top_n, len(rankings))} PROPOSALS:")
            print("-" * 80)

            for i, ranking in enumerate(rankings[:top_n], 1):
                print(f"{i:2d}. {ranking.title[:60]}...")
                print(f"    Org: {ranking.organization}")
                print(
                    f"    Score: {ranking.composite_score:.2f} (IAPS: {ranking.iaps_alignment:.1f}, Policy Specificity: {ranking.policy_specificity:.1f}, Evidence Base: {ranking.evidence_base:.1f}, Political Viability: {ranking.political_viability:.1f})"
                )
                print()

        return rankings


if __name__ == "__main__":
    # Example usage - would need to be configured with actual parameters
    print("Run this via a configuration script - see run_comparative_analysis function")
