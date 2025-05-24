#!/usr/bin/env python3
"""
Critical evaluation of organizations for AI policy influence.
Goes beyond proposal volume to assess actual credibility and impact.
"""

import asyncio
import os
from pathlib import Path

import pandas as pd

from batch_search import BatchSearcher
from research_report import ResearchReportGenerator


def identify_potentially_influential_orgs(csv_path: str) -> dict:
    """
    Identify organizations that MIGHT be influential using an inclusive approach.
    Cast a wide net to avoid missing excellent proposals from smaller/newer organizations.
    """
    df = pd.read_csv(csv_path)

    candidates = {}

    # 1. KNOWN HIGH-IMPACT ORGANIZATIONS (expanded list)
    known_credible = [
        # Elite Think Tanks
        "Center for Security and Emerging Technology",
        "Federation of American Scientists",
        "Bipartisan Policy Center",
        "Center for Strategic and International Studies",
        "Brookings Institution",
        "American Enterprise Institute",
        "Council on Foreign Relations",
        "RAND Corporation",
        "Carnegie Endowment for International Peace",
        # Top Universities & Research Institutions
        "Stanford University",
        "MIT",
        "Carnegie Mellon",
        "UC Berkeley",
        "Harvard University",
        "Georgetown University",
        "University of Washington",
        "Princeton University",
        "Yale University",
        "Columbia University",
        "NYU",
        "University of Chicago",
        "Georgia Institute of Technology",
        "University of Pennsylvania",
        # Major Tech Companies
        "Google",
        "Microsoft",
        "Meta",
        "Amazon",
        "Apple",
        "OpenAI",
        "Anthropic",
        "IBM",
        "NVIDIA",
        "Intel",
        "Qualcomm",
        "Adobe",
        "Salesforce",
        # Government & Policy Organizations
        "NIST",
        "NSF",
        "Department of Defense",
        "Department of Commerce",
        "Department of Homeland Security",
        "Department of State",
        "Department of Energy",
        "Government Accountability Office",
        "Congressional Research Service",
        # Professional & Industry Associations
        "IEEE",
        "ACM",
        "National Academy of Sciences",
        "American Association for the Advancement of Science",
    ]

    # 2. ACADEMIC INSTITUTIONS (broader net)
    # Look for any organization with "University", "Institute", "College", "Academy" in name
    academic_orgs = df[
        df["Organization"].str.contains(
            r"University|Institute|College|Academy|Research|Laboratory|Lab\b", case=False, na=False, regex=True
        )
    ]["Organization"].unique()

    # 3. GOVERNMENT/POLICY ORGANIZATIONS (broader net)
    # Look for government, policy, or official-sounding organizations
    gov_policy_orgs = df[
        df["Organization"].str.contains(
            r"Department|Agency|Office|Bureau|Commission|Council|Center|Government|Policy|Public|National|Federal|State",
            case=False,
            na=False,
            regex=True,
        )
    ]["Organization"].unique()

    # 4. INTERNATIONAL & MULTILATERAL ORGANIZATIONS
    international_orgs = df[
        df["Organization"].str.contains(
            r"International|Global|World|United Nations|UN\b|NATO|European|EU\b|OECD|G7|G20",
            case=False,
            na=False,
            regex=True,
        )
    ]["Organization"].unique()

    # 5. ACTIVE SUBMITTERS WITH SUBSTANTIAL PROPOSALS (quality over quantity)
    # Organizations with 5+ proposals (showing sustained engagement)
    active_submitters = df["Organization"].value_counts()
    substantial_orgs = active_submitters[active_submitters >= 5].index.tolist()

    # 6. ORGANIZATIONS WITH LONG-FORM, DETAILED PROPOSALS
    # Look for orgs that submitted detailed proposals (high word count = more thoughtful)
    df["content_length"] = df["FullText"].astype(str).str.len()
    detailed_proposal_orgs = df[df["content_length"] > 2000].groupby("Organization").size()
    thoughtful_orgs = detailed_proposal_orgs[detailed_proposal_orgs >= 2].index.tolist()

    # Filter to organizations actually present in the dataset
    all_orgs = set(df["Organization"].unique())

    # Start with known credible organizations
    known_credible_present = [org for org in known_credible if org in all_orgs]
    known_credible_set = set(known_credible_present)

    candidates["Known Credible"] = known_credible_present
    candidates["Academic & Research"] = [org for org in academic_orgs if org not in known_credible_set][
        :20
    ]  # Top 20 by name
    candidates["Government & Policy"] = [org for org in gov_policy_orgs if org not in known_credible_set][
        :15
    ]  # Top 15 by name
    candidates["International"] = [org for org in international_orgs if org not in known_credible_set][
        :10
    ]  # Top 10 by name
    candidates["Active Submitters"] = [org for org in substantial_orgs if org not in known_credible_set][
        :15
    ]  # Top 15 by volume
    candidates["Thoughtful Contributors"] = [org for org in thoughtful_orgs if org not in known_credible_set][
        :10
    ]  # Top 10 by depth

    # 7. INDUSTRY ORGANIZATIONS (more selective approach)
    # Only include industry orgs that show signs of serious policy engagement
    industry_orgs = df[df["OrgType"].isin(["Industry (Other)", "Industry Association"])]

    # Filter for industry orgs with detailed proposals OR multiple submissions
    detailed_industry = industry_orgs[industry_orgs["content_length"] > 1500]["Organization"].unique()

    # Organizations with 8+ submissions
    industry_counts = industry_orgs.groupby("Organization").size()
    frequent_industry = industry_counts[industry_counts >= 8].index.tolist()

    # Combine both criteria
    serious_industry = list(set(detailed_industry) | set(frequent_industry))
    candidates["Serious Industry"] = [org for org in serious_industry if org not in known_credible_set][:10]

    # Remove empty categories and limit total organizations to keep assessment manageable
    candidates = {k: v for k, v in candidates.items() if v}

    # Print summary of inclusive approach
    total_candidates = sum(len(orgs) for orgs in candidates.values())
    print("\n🌐 INCLUSIVE ORGANIZATION IDENTIFICATION:")
    print(f"   • Total unique organizations in dataset: {len(all_orgs)}")
    print(f"   • Selected for assessment: {total_candidates}")
    print(f"   • Coverage: {(total_candidates/len(all_orgs)*100):.1f}% of organizations")

    return candidates


async def critically_assess_organization(
    org_name: str, searcher: BatchSearcher, report_generator: ResearchReportGenerator
) -> dict:
    """
    Critically assess an organization's actual influence and credibility.
    Be skeptical and look for concrete evidence of impact.
    """

    search_query = f"{org_name} AI policy influence government funding credibility"
    search_instruction = f"""Find CRITICAL information about {org_name} to assess their actual influence on AI policy:

    CREDIBILITY SIGNALS TO FIND:
    - Government contracts, grants, or official advisory roles
    - Testimony before Congress or other official bodies  
    - Citations by policymakers or in official documents
    - Track record of successful policy advocacy
    - Leadership backgrounds (former government officials, etc.)
    - Funding sources and transparency
    - Academic credentials and peer recognition
    
    SKEPTICAL QUESTIONS TO INVESTIGATE:
    - Are they actually influential or just loud?
    - Do they have real expertise or just marketing?
    - What's their funding model - who pays them?
    - Have their recommendations actually been implemented?
    - Are they cited by other credible sources?
    - Any conflicts of interest or bias?
    
    Look for CONCRETE EVIDENCE, not just self-promotion."""

    # Perform targeted search
    batch_result = await searcher.batch_search(
        query=search_query,
        instruction=search_instruction,
        num_queries=5,
        max_results=10,
    )

    # Generate critical assessment
    research_query = f"""Provide a CRITICAL assessment of {org_name}'s actual influence on AI policy:

    REQUIRED ANALYSIS:
    1. REAL INFLUENCE: What concrete evidence exists of their policy impact? (Congressional testimony, government adoption of recommendations, etc.)
    2. CREDIBILITY: What are their actual credentials and expertise? Who are their leaders?
    3. FUNDING & BIAS: Who funds them? Any conflicts of interest?
    4. TRACK RECORD: What specific policy wins can they claim? What failed?
    5. PEER RECOGNITION: How do other credible experts view them?
    6. CRITICAL ASSESSMENT: Rate their actual influence as HIGH/MEDIUM/LOW and explain why
    
    BE SKEPTICAL. Many organizations claim influence they don't have. Look for:
    - Concrete policy outcomes they influenced
    - Government officials who actually listen to them
    - Academic or peer recognition
    - Transparent funding and governance
    
    REJECT organizations that are:
    - Just corporate lobbying disguised as policy expertise
    - Self-promoting without real credentials  
    - Opaque about funding or governance
    - Making grandiose claims without evidence
    
    Keep assessment under 300 words. Be brutally honest."""

    report_path = await report_generator.generate_and_save_report(
        query=research_query,
        directory=batch_result.output_directory,
        url_to_file_mapping=batch_result.url_to_file_mapping,
    )

    # Read and return the assessment
    with open(report_path, encoding="utf-8") as f:
        assessment = f.read()

    return {
        "name": org_name,
        "critical_assessment": assessment,
        "search_directory": batch_result.output_directory,
        "report_path": report_path,
    }


def check_existing_assessments(output_dir: str) -> set:
    """Check which organizations have already been assessed."""

    output_path = Path(output_dir)
    if not output_path.exists():
        return set()

    assessed_orgs = set()

    # Look for existing assessment directories
    assessment_dirs = [d for d in output_path.iterdir() if d.is_dir()]

    for assessment_dir in assessment_dirs:
        # Check if the directory has a research report (indicates completion)
        report_file = assessment_dir / "research_report.md"
        if report_file.exists():
            # Extract organization name from directory name
            dir_name = assessment_dir.name

            # Try to map directory names back to organization names
            org_name = extract_org_name_from_dir(dir_name)
            if org_name:
                assessed_orgs.add(org_name)
                print(f"   ✅ Found existing assessment: {org_name}")

    return assessed_orgs


def extract_org_name_from_dir(dir_name: str) -> str:
    """Extract organization name from assessment directory name."""

    # Map directory names back to organization names
    org_mapping = {
        "cset_ai_policy_influence": "Center for Security and Emerging Technology",
        "fas_ai_policy_influence": "Federation of American Scientists",
        "bpc_ai_policy_influence": "Bipartisan Policy Center",
        "google_ai_policy_influence": "Google",
        "microsoft_ai_policy_influence": "Microsoft",
        "amazon_ai_policy_influence": "Amazon",
        "openai_policy_influence_credibility": "OpenAI",
        "anthropic_policy_influence_credibility": "Anthropic",
        "anthropic_ai_policy_influence": "Anthropic",
        "hpe_ai_policy_influence": "HPE",
        "hpe_ai_policy_influence_credibility": "HPE",
        "cgi_federal_ai_policy_influence": "CGI Federal, Inc.",
        "itci_ai_policy_influence": "Information Technology Industry Council",
        "incompas_ai_policy_influence": "INCOMPAS",
        "alvarez_marshall_ai_policy_influence": "Alvarez & Marshall Federal LLC",
        "wahba_institute_ai_policy_influence": "Wahba Institute for Strategic Competition",
        "arm_institute_ai_policy_influence": "Advanced Robotics for Manufacturing\n  Institute",
    }

    return org_mapping.get(dir_name, "")


async def evaluate_organizations_critically(csv_path: str, output_dir: str = "critical_org_assessment"):
    """Run critical evaluation of potentially influential organizations in parallel, skipping already completed assessments."""

    # Setup
    openai_api_key = os.getenv("OPENAI_API_KEY")
    exa_api_key = os.getenv("EXA_API_KEY")

    if not openai_api_key or not exa_api_key:
        print("❌ Missing API keys")
        return

    # Get candidate organizations
    print("🔍 Identifying potentially influential organizations...")
    candidates = identify_potentially_influential_orgs(csv_path)

    total_orgs = sum(len(orgs) for orgs in candidates.values())
    print(f"📊 Found {total_orgs} organizations to critically assess:")
    for category, orgs in candidates.items():
        if orgs:  # Only show categories with orgs
            print(f"   • {category}: {orgs}")

    # Check for existing assessments
    print(f"\n🔍 Checking for existing assessments in {output_dir}...")
    already_assessed = check_existing_assessments(output_dir)

    if already_assessed:
        print(f"📋 Found {len(already_assessed)} organizations already assessed:")
        for org in sorted(already_assessed):
            print(f"   ✅ {org}")

    # Filter out already assessed organizations
    orgs_to_assess = {}
    total_remaining = 0

    for category, org_list in candidates.items():
        if not org_list:
            continue

        remaining_orgs = [org for org in org_list if org not in already_assessed]
        if remaining_orgs:
            orgs_to_assess[category] = remaining_orgs
            total_remaining += len(remaining_orgs)
            print(f"\n📋 {category}: {len(remaining_orgs)}/{len(org_list)} remaining to assess")
            for org in remaining_orgs:
                print(f"   🔄 {org}")

    if total_remaining == 0:
        print("\n✅ All organizations have already been assessed!")
        print(f"📄 Results available in: {output_dir}/critical_organization_assessments.md")
        return load_existing_assessment_results(output_dir)

    print(f"\n🚀 Will assess {total_remaining} remaining organizations...")

    # Prepare all organization assessment tasks for parallel execution
    assessment_tasks = []
    org_metadata = []  # Track category info for each org

    async with BatchSearcher(openai_api_key, exa_api_key, output_dir=output_dir) as searcher:
        async with ResearchReportGenerator(openai_api_key) as report_generator:
            # Create tasks for remaining organizations only
            for category, org_list in orgs_to_assess.items():
                for org_name in org_list:
                    task = critically_assess_organization(org_name, searcher, report_generator)
                    assessment_tasks.append(task)
                    org_metadata.append({"name": org_name, "category": category})

            if assessment_tasks:
                print(f"\n🚀 Running {len(assessment_tasks)} organization assessments in parallel...")
                print("This will be much faster but will use more API requests simultaneously.")

                # Run all assessments in parallel
                try:
                    assessment_results = await asyncio.gather(*assessment_tasks, return_exceptions=True)
                except Exception as e:
                    print(f"❌ Error during parallel execution: {e}")
                    return

    # Process results and handle any exceptions
    new_assessments = []
    successful_count = 0
    failed_count = 0

    for i, result in enumerate(assessment_results):
        org_info = org_metadata[i]

        if isinstance(result, Exception):
            print(f"   ❌ Failed to assess {org_info['name']}: {result}")
            failed_count += 1
        elif isinstance(result, dict):
            result["category"] = org_info["category"]
            new_assessments.append(result)
            successful_count += 1
            print(f"   ✅ Completed: {org_info['name']} ({org_info['category']})")
        else:
            print(f"   ⚠️  Unexpected result type for {org_info['name']}: {type(result)}")
            failed_count += 1

    print(f"\n📊 Parallel execution complete: {successful_count} successful, {failed_count} failed")

    # Load existing assessments and combine with new ones
    existing_assessments = load_existing_assessment_results(output_dir)
    all_assessments = existing_assessments + new_assessments

    # Save comprehensive results
    results_path = Path(output_dir) / "critical_organization_assessments.md"
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("# CRITICAL ASSESSMENT: AI Policy Organization Influence\n\n")
        f.write(
            "**Methodology**: Organizations evaluated based on concrete evidence of policy impact, not self-promotion or proposal volume.\n"
        )
        f.write("**Analysis Method**: Parallel processing with resumption capability\n")
        f.write(f"**Total Organizations Assessed**: {len(all_assessments)}\n\n")

        for category in candidates.keys():
            category_assessments = [a for a in all_assessments if a["category"] == category]
            if category_assessments:
                f.write(f"## {category}\n\n")
                for assessment in category_assessments:
                    f.write(f"### {assessment['name']}\n")
                    f.write(f"{assessment['critical_assessment']}\n\n")

    print("\n✅ Critical assessment complete!")
    print(f"📄 Results saved to: {results_path}")

    return all_assessments


def load_existing_assessment_results(output_dir: str) -> list:
    """Load existing assessment results from the master file."""

    results_path = Path(output_dir) / "critical_organization_assessments.md"
    if not results_path.exists():
        return []

    # This is a simplified loader - in practice you might want to parse the markdown
    # For now, just return empty list since we're mainly focused on avoiding re-runs
    return []


if __name__ == "__main__":
    asyncio.run(evaluate_organizations_critically("IFP AI Action Plan Database file.csv"))
