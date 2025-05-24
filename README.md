# Batch Search Script

A comprehensive web research tool that performs parallel searches using OpenAI and Exa APIs to gather diverse information from multiple sources.

## Features

- **Intelligent Query Generation**: Uses OpenAI to generate diverse, focused search queries from a base query
- **Parallel Web Search**: Executes multiple searches simultaneously using the Exa search API with full content retrieval
- **Smart Content Filtering**: Uses AI to extract only relevant content based on your specific instruction
- **Automatic File Saving**: Saves relevant information from each webpage to individual markdown files
- **AI-Powered Content Extraction**: Analyzes webpage content and extracts only information relevant to your research task
- **Comprehensive Summaries**: Generates concise summaries of each webpage's content
- **Async Processing**: All operations are performed asynchronously for maximum efficiency

## Prerequisites

- Python 3.8+
- OpenAI API key
- Exa API key

## Installation

### Using uv (Recommended)

1. Install uv if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Sync the project dependencies:
```bash
uv sync
```

This will create a virtual environment and install all dependencies defined in `pyproject.toml`.

3. Activate the virtual environment:
```bash
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows
```

### Alternative: Using pip

Install dependencies using regular pip:
```bash
pip install -r requirements.txt
```

Or install the project in development mode:
```bash
pip install -e .
```

## Setup

Set your API keys as environment variables:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export EXA_API_KEY="your-exa-api-key"
```

Or create a `.env` file:
```
OPENAI_API_KEY=your-openai-api-key
EXA_API_KEY=your-exa-api-key
```

## Usage

### Basic Usage

```python
import asyncio
import os
from batch_search import BatchSearcher

async def main():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    exa_api_key = os.getenv("EXA_API_KEY")
    
    # You can specify a custom output directory
    async with BatchSearcher(openai_api_key, exa_api_key, output_dir="my_research") as searcher:
        result = await searcher.batch_search(
            query="impact of artificial intelligence on healthcare",
            instruction="Find detailed information about AI applications in healthcare, including use cases, benefits, and challenges",
            num_queries=5,  # Generate 5 diverse search queries
            max_results=15  # Process top 15 results
        )
        print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

### Running the Example

You can run the script in several ways:

**With uv (recommended):**
```bash
uv run python batch_search.py
```

**With activated virtual environment:**
```bash
python batch_search.py
```

**As a module:**
```bash
python -m batch_search
```

This will execute searches for several example topics and display the results.

## API Reference

### BatchSearcher Class

#### `__init__(openai_api_key: str, exa_api_key: str)`
Initialize the batch searcher with API keys.

#### `batch_search(query: str, instruction: str, num_queries: int, max_results: int) -> str`
Perform comprehensive web research.

**Parameters:**
- `query`: Base search query (should be broad enough for diverse sub-queries)
- `instruction`: Specific instruction for filtering relevant content
- `num_queries`: Number of diverse queries to generate (default: 5)
- `max_results`: Maximum number of results to process (default: 20)

**Returns:**
Formatted string containing search results, visited URLs, and summaries.

## How It Works

1. **Query Generation**: Takes your base query and uses OpenAI to generate diverse, focused search queries
2. **Parallel Search with Content**: Executes all generated queries simultaneously using Exa's `search_and_contents` API to get both search results and full page content
3. **Content Filtering**: Uses AI to analyze search results and filter only those relevant to your specific instruction
4. **Content Extraction & File Saving**: For each relevant webpage:
   - Analyzes the full content using AI to extract only information relevant to your research task
   - Saves the relevant information to a markdown file (one file per webpage)
   - Generates a concise summary of the content
5. **Results Compilation**: Formats all findings into a comprehensive report with links to saved files

## File Organization

The script automatically organizes results with intelligent folder structure:

### Directory Structure
```
search_results/                    # Base output directory (configurable)
├── ai_healthcare_apps/           # LLM-generated folder for first query
│   ├── example_com_ai_medical.md
│   ├── pubmed_gov_healthcare_ai.md
│   └── ...
├── climate_renewable_energy/     # LLM-generated folder for second query  
│   ├── energy_gov_renewables.md
│   ├── nature_com_climate_study.md
│   └── ...
└── ev_market_trends/            # LLM-generated folder for third query
    ├── tesla_com_market_data.md
    └── ...
```

### Folder Naming
- **Automatic Generation**: Each query gets its own folder with an AI-generated descriptive name
- **Smart Naming**: Uses the query and instruction to create meaningful folder names (e.g., "ai_healthcare_apps", "climate_renewable_energy")
- **Filesystem Safe**: All folder names are sanitized for cross-platform compatibility
- **Organized Results**: Keeps different research topics completely separate

### File Details
- **Individual webpage files**: Each relevant webpage gets its own markdown file with extracted information
- **Filename format**: `{domain}_{sanitized_title}.md`
- **Content format**: Well-structured markdown with relevant information only, including source URL

## Example Queries

### Good Queries (broad topics that allow diverse exploration):
- "impact of artificial intelligence on healthcare"
- "climate change and renewable energy solutions" 
- "future of electric vehicles market trends"
- "blockchain technology applications"

### Less Ideal Queries (too specific):
- "Tesla Model 3 price in California on March 15, 2024"
- "exact population of Tokyo in 2023"

For very specific queries, consider using individual web search tools instead.

## Error Handling

The script includes comprehensive error handling:
- Failed API calls fallback to alternative strategies
- Network timeouts are handled gracefully
- Individual URL failures don't stop the entire process
- All errors are logged with detailed information

## Rate Limiting

The script respects API rate limits by:
- Using reasonable timeouts for web requests
- Processing URLs in controlled batches
- Implementing retry logic for failed requests

## Configuration

You can customize the behavior by modifying these parameters:
- `num_queries`: Number of search queries to generate
- `max_results`: Maximum results to process
- `output_dir`: Directory where extracted content files are saved
- OpenAI model and parameters in generation functions

## Dependencies

- `openai`: OpenAI API client
- `exa-py`: Exa search API client
- `python-dotenv`: Environment variable management

## License

This project is open source and available under the MIT License.
