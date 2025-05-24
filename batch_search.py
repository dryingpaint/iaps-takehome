import asyncio
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from exa_py import Exa  # type: ignore
from openai import AsyncOpenAI

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    url: str
    title: str
    content: str
    summary: str = ""
    success: bool = True
    error_message: str = ""
    saved_file: Optional[str] = None  # Path to saved file with relevant information


class BatchSearcher:
    def __init__(self, openai_api_key: str, exa_api_key: str, output_dir: str = "search_results"):
        self.openai_client = AsyncOpenAI(api_key=openai_api_key)
        self.exa = Exa(api_key=exa_api_key)
        self.base_output_dir = Path(output_dir)
        self.base_output_dir.mkdir(exist_ok=True)
        self.output_dir: Optional[Path] = None  # Will be set per query

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def _generate_folder_name(self, query: str, instruction: str) -> str:
        """Generate a descriptive folder name for the query using LLM."""
        prompt = f"""Generate a short, descriptive folder name (2-4 words) for organizing search results about this query.

Query: "{query}"
Task: "{instruction}"

Requirements:
- Use only letters, numbers, underscores, and hyphens
- Keep it under 50 characters
- Make it descriptive but concise
- Use snake_case format
- Focus on the main topic/subject

Examples:
- "AI healthcare applications" → "ai_healthcare_apps"
- "climate change renewable energy" → "climate_renewable_energy"
- "electric vehicle market trends" → "ev_market_trends"

Folder name:"""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=50,
            )

            content = response.choices[0].message.content
            if content:
                # Clean the folder name to ensure it's filesystem-safe
                folder_name = content.strip().lower()
                folder_name = re.sub(r"[^\w\s-]", "", folder_name)
                folder_name = re.sub(r"[-\s]+", "_", folder_name)
                return folder_name[:50]  # Limit length

            # Fallback to a simple version of the query
            return re.sub(r"[^\w\s-]", "", query.lower())[:30].replace(" ", "_")
        except Exception as e:
            logger.error(f"Error generating folder name: {e}")
            # Fallback to a simple version of the query
            return re.sub(r"[^\w\s-]", "", query.lower())[:30].replace(" ", "_")

    def _sanitize_filename(self, url: str, title: str) -> str:
        """Create a safe filename from URL and title."""
        # Extract domain from URL
        domain = re.sub(r"https?://", "", url).split("/")[0]

        # Clean title for filename
        if title:
            # Take first 50 chars of title and clean it
            clean_title = re.sub(r"[^\w\s-]", "", title[:50]).strip()
            clean_title = re.sub(r"[-\s]+", "_", clean_title)
        else:
            clean_title = "untitled"

        # Combine domain and title
        filename = f"{domain}_{clean_title}.md"

        # Ensure it's a valid filename
        filename = re.sub(r'[<>:"/\\|?*]', "_", filename)
        return filename

    async def _generate_saved_artifact_from_web_scrape(self, task: str, url: str, content: str) -> str | None:
        """
        Generate a saved artifact for the web page that only contains information that is relevant to the task.
        If no information is relevant, return None.

        Args:
            task: The research task to evaluate content against
            url: The source URL of the webpage
            content: The webpage content to analyze

        Returns:
            str | None: A markdown artifact containing relevant information, or None if no relevant information found
        """
        query = f"""You are an expert research assistant analyzing webpage content for relevance to a specific task.

TASK: "{task}"
URL: {url}

INSTRUCTIONS:
1. First, determine if ANY content is relevant to the task. If not, respond with ONLY "NO_RELEVANT_INFORMATION"
2. If relevant content exists:
   - Extract and organize ALL information that is directly relevant to the task
   - Include a brief summary of how the content relates to the task
   - Include key facts, data points, insights, and important quotes
   - Include relevant images, tables, and other media
   - Preserve any context needed to understand the information
   - If content is only tangentially related, explain why it might be useful

RULES:
- Use clear, hierarchical markdown formatting
- Include ALL information that helps address the task
- Exclude duplicate information and external links
- Maintain a logical flow of information
- Add the source URL at the end of your response with a new line: `Source URL: {url}`

CONTENT TO ANALYZE:
{content}"""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": query}],
                temperature=0.3,
                max_tokens=2000,
            )

            content_response = response.choices[0].message.content
            if not content_response:
                return None

            # Check if the response indicates no relevant information
            if "no_relevant_information" in content_response.strip().lower():
                return None

            if "source url" not in content_response.strip().lower():
                content_response += f"\nSource URL: {url}"

            return content_response
        except Exception as e:
            logger.error(f"Error generating artifact for {url}: {e}")
            return None

    async def _save_webpage_content(self, result: dict, task: str) -> Optional[str]:
        """Save relevant webpage content to a file."""
        try:
            if self.output_dir is None:
                logger.error("Output directory not set")
                return None

            url = result.get("url", "")
            title = result.get("title", "")
            content = result.get("text", "")

            if not content:
                return None

            # Generate relevant artifact
            artifact = await self._generate_saved_artifact_from_web_scrape(task, url, content)

            if not artifact:
                logger.info(f"No relevant information found for {url}")
                return None

            # Create filename
            filename = self._sanitize_filename(url, title)
            filepath = self.output_dir / filename

            # Save to file
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(artifact)

            logger.info(f"Saved relevant content to {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Error saving content for {result.get('url', 'unknown')}: {e}")
            return None

    async def generate_search_queries(self, base_query: str, n_queries: int = 5) -> List[str]:
        """Generate {n_queries} diverse search queries based on the base query using OpenAI."""
        prompt = f"""Given the search query: "{base_query}", generate {n_queries} focused, keyword-rich search queries optimized for web search.
        
        Guidelines:
        - Each query should be short (3-5 words) and focused on a specific entity or aspect
        - Use important keywords and avoid unnecessary words
        - If the query contains multiple entities (like names), create separate queries for each
        - Avoid complex phrases or questions
        - Each query should be distinct and target different information
        
        Return only the queries, one per line, without numbering or additional text.
        
        Example input: "LinkedIn profiles of Sandra Rivera and Marc Graff at Altera Intel"
        Example output:
        sandra rivera altera intel career
        marc graff altera intel linkedin
        sandra rivera intel executive
        marc graff altera leadership
        altera intel management team"""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=300,
            )

            content = response.choices[0].message.content
            if content:
                queries = [q.strip() for q in content.split("\n") if q.strip()]
                return queries[:n_queries]
            return [base_query]  # Fallback if no content
        except Exception as e:
            logger.error(f"Error generating search queries: {e}")
            return [base_query]  # Fallback to original query

    async def extract_relevant_urls(self, search_results: List[dict], instruction: str) -> List[str]:
        """Use OpenAI to extract relevant URLs from Exa search results based on the instruction."""

        # Format search results for the prompt
        results_text = ""
        for i, result in enumerate(search_results[:10], 1):  # Limit to top 10 results
            results_text += f"{i}. Title: {result.get('title', 'N/A')}\n"
            results_text += f"   URL: {result.get('url', 'N/A')}\n"
            results_text += f"   Summary: {result.get('text', 'N/A')[:200]}...\n\n"

        prompt = f"""Given the following search results and instruction, extract URLs that are relevant to answering the instruction.
        If there are no relevant URLs, return `NONE`.
        Return only the URLs, one per line, without any additional text or explanation.

        Instruction: {instruction}

        Search Results:
        {results_text}

        Relevant URLs:"""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500,
            )

            content = response.choices[0].message.content
            if content:
                urls = [url.strip() for url in content.split("\n") if url.strip()]
                # Filter to only valid URLs
                urls = [url for url in urls if url.startswith("http")]
                if not urls or "none" in content.lower():
                    return []
                return urls
            return []
        except Exception as e:
            logger.error(f"Error extracting URLs: {e}")
            # Fallback: return all URLs from search results
            valid_urls: List[str] = []
            for result in search_results:
                url = result.get("url")
                if url is not None and isinstance(url, str):
                    valid_urls.append(url)
            return valid_urls

    async def summarize_url_content(self, content: str, url: str) -> str:
        """Generate a one-sentence summary of the URL content using OpenAI."""
        if not content or "error" in content.lower():
            return f"Failed to access content from {url}"

        prompt = f"""Summarize the following webpage content in one clear, informative sentence.
        Focus on the main topic and key information.
        Keep the summary concise but specific.

        URL: {url}
        Content:
        {content[:2000]}  

        One-sentence summary:"""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150,
            )

            content_response = response.choices[0].message.content
            if content_response:
                return content_response.strip()
            return f"Content available from {url} (summary generation failed)"
        except Exception as e:
            logger.warning(f"Failed to summarize {url}: {e}")
            return f"Content available from {url} (summary generation failed)"

    async def search_and_get_content_with_exa(self, query: str, num_results: int = 10) -> List[dict]:
        """Perform search using Exa API with content included."""
        try:
            result = self.exa.search_and_contents(
                query=query,
                num_results=num_results,
                text=True,  # Include full text content
            )
            # Convert Exa result objects to dictionaries for easier processing
            results = []
            for item in result.results:
                results.append(
                    {
                        "url": item.url,
                        "title": item.title or "",
                        "text": item.text or "",
                        "published_date": item.published_date,
                        "author": item.author,
                        "score": getattr(item, "score", None),
                    }
                )
            return results
        except Exception as e:
            logger.error(f"Error searching with Exa for query '{query}': {e}")
            return []

    async def process_search_results(self, search_queries: List[str], instruction: str) -> List[dict]:
        """Process search results and return results with content."""
        all_results = []

        # Perform searches in parallel
        search_tasks = [self.search_and_get_content_with_exa(query) for query in search_queries]
        search_results_list = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Collect all results
        for results in search_results_list:
            if isinstance(results, Exception):
                logger.warning(f"Error in search result: {results}")
                continue
            if isinstance(results, list):
                all_results.extend(results)

        # Filter results by relevance using OpenAI
        if all_results:
            # Instead of just extracting URLs, we'll filter the entire results
            relevant_results = await self.filter_relevant_results(all_results, instruction)
            # Remove duplicates while preserving order
            unique_results = []
            seen_urls = set()
            for result in relevant_results:
                if result["url"] not in seen_urls:
                    unique_results.append(result)
                    seen_urls.add(result["url"])
            return unique_results

        return []

    async def filter_relevant_results(self, search_results: List[dict], instruction: str) -> List[dict]:
        """Use OpenAI to filter relevant search results based on the instruction."""

        # Format search results for the prompt
        results_text = ""
        for i, result in enumerate(search_results[:15], 1):  # Limit to top 15 results
            results_text += f"{i}. Title: {result.get('title', 'N/A')}\n"
            results_text += f"   URL: {result.get('url', 'N/A')}\n"
            results_text += f"   Content Preview: {result.get('text', 'N/A')[:300]}...\n\n"

        prompt = f"""Given the following search results and instruction, identify which results are relevant to answering the instruction.
        Return the numbers (1, 2, 3, etc.) of the relevant results, one per line.
        If no results are relevant, return `NONE`.

        Instruction: {instruction}

        Search Results:
        {results_text}

        Relevant result numbers:"""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200,
            )

            content = response.choices[0].message.content
            if content and "none" not in content.lower():
                # Extract numbers from the response
                numbers = []
                for line in content.split("\n"):
                    line = line.strip()
                    if line.isdigit():
                        numbers.append(int(line) - 1)  # Convert to 0-based index

                # Return the relevant results
                relevant_results = []
                for num in numbers:
                    if 0 <= num < len(search_results):
                        relevant_results.append(search_results[num])
                return relevant_results

            return []
        except Exception as e:
            logger.error(f"Error filtering results: {e}")
            # Fallback: return first 10 results
            return search_results[:10]

    async def generate_summaries(self, results: List[dict], task: str) -> List[SearchResult]:
        """Generate summaries for search results in parallel and save relevant content to files."""
        logger.info(f"Generating summaries for {len(results)} results")

        # Generate summaries and save files in parallel
        summary_tasks = [
            self.summarize_url_content(result.get("text", ""), result.get("url", "")) for result in results
        ]
        file_save_tasks = [self._save_webpage_content(result, task) for result in results]

        # Execute both sets of tasks in parallel
        summaries, saved_files = await asyncio.gather(
            asyncio.gather(*summary_tasks, return_exceptions=True),
            asyncio.gather(*file_save_tasks, return_exceptions=True),
        )

        # Create SearchResult objects
        search_results = []
        for result, summary, saved_file in zip(results, summaries, saved_files, strict=False):
            # Handle summary
            if isinstance(summary, str):
                summary_text = summary
                success = True
                error_msg = ""
            else:
                summary_text = f"Error generating summary: {str(summary)}"
                success = False
                error_msg = str(summary)

            # Handle saved file
            file_path = None
            if isinstance(saved_file, str):
                file_path = saved_file
            elif isinstance(saved_file, Exception):
                logger.warning(f"Error saving file for {result.get('url', 'unknown')}: {saved_file}")

            search_results.append(
                SearchResult(
                    url=result.get("url", ""),
                    title=result.get("title", ""),
                    content=result.get("text", ""),
                    summary=summary_text,
                    success=success,
                    error_message=error_msg,
                    saved_file=file_path,
                )
            )

        return search_results

    async def batch_search(
        self,
        query: str,
        instruction: str = "Find relevant information about the query",
        num_queries: int = 5,
        max_results: int = 20,
    ) -> str:
        """Perform comprehensive web research by executing multiple parallel searches."""
        logger.info(f"Starting batch search for: {query}")

        try:
            # Generate folder name for this query and set up directory
            folder_name = await self._generate_folder_name(query, instruction)
            self.output_dir = self.base_output_dir / folder_name
            self.output_dir.mkdir(exist_ok=True)
            logger.info(f"Created output directory: {self.output_dir}")

            # Generate diverse search queries
            search_queries = await self.generate_search_queries(query, n_queries=num_queries)
            logger.info(f"Generated {len(search_queries)} search queries: {search_queries}")

            # Process search results and get relevant results with content
            relevant_results = await self.process_search_results(search_queries, instruction)
            logger.info(f"Found {len(relevant_results)} relevant results")

            if not relevant_results:
                return "No relevant results found for the given query."

            # Limit results and generate summaries
            limited_results = relevant_results[:max_results]
            search_results = await self.generate_summaries(limited_results, instruction)

            # Count saved files
            saved_files_count = sum(1 for result in search_results if result.saved_file)

            # Format results
            output_lines = [f"Batch search results for: {query}\n"]
            output_lines.append(f"Task/Instruction: {instruction}\n")
            output_lines.append(f"Generated {len(search_queries)} search queries:")
            for i, sq in enumerate(search_queries, 1):
                output_lines.append(f"  {i}. {sq}")
            output_lines.append(f"\nProcessed {len(search_results)} results")
            output_lines.append(f"Saved {saved_files_count} files with relevant content to: {self.output_dir}\n")

            for i, result in enumerate(search_results, 1):
                if result.success:
                    output_lines.append(f"{i}. {result.url}")
                    output_lines.append(f"   Title: {result.title}")
                    output_lines.append(f"   Summary: {result.summary}")
                    if result.saved_file:
                        output_lines.append(f"   📄 Saved File: {result.saved_file}")
                    else:
                        output_lines.append("   ❌ No relevant content saved")
                    output_lines.append("")
                else:
                    output_lines.append(f"{i}. {result.url}")
                    output_lines.append(f"   Error: {result.error_message}")
                    if result.saved_file:
                        output_lines.append(f"   📄 Saved File: {result.saved_file}")
                    output_lines.append("")

            return "\n".join(output_lines)

        except Exception as e:
            logger.error(f"Error in batch search: {e}")
            return f"Error during batch search: {str(e)}"
