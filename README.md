# Batch Search Script

A comprehensive web research tool that performs parallel searches using OpenAI and Exa APIs to gather diverse information from multiple sources.

## Features

- **Intelligent Query Generation**: Uses OpenAI to generate diverse, focused search queries from a base query
- **Parallel Web Search**: Executes multiple searches simultaneously using the Exa search API
- **Smart URL Filtering**: Uses AI to extract only relevant URLs based on your specific instruction
- **Content Extraction**: Fetches and processes content from web pages
- **AI-Powered Summaries**: Generates concise summaries of each visited page
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
    
    async with BatchSearcher(openai_api_key, exa_api_key) as searcher:
        result = await searcher.batch_search(
            query="impact of artificial intelligence on healthcare",
            instruction="Find detailed information about AI applications in healthcare",
            num_queries=5,  # Generate 5 diverse search queries
            max_urls=15     # Visit top 15 URLs
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

#### `batch_search(query: str, instruction: str, num_queries: int, max_urls: int) -> str`
Perform comprehensive web research.

**Parameters:**
- `query`: Base search query (should be broad enough for diverse sub-queries)
- `instruction`: Specific instruction for filtering relevant URLs
- `num_queries`: Number of diverse queries to generate (default: 5)
- `max_urls`: Maximum number of URLs to visit (default: 20)

**Returns:**
Formatted string containing search results, visited URLs, and summaries.

## How It Works

1. **Query Generation**: Takes your base query and uses OpenAI to generate diverse, focused search queries
2. **Parallel Search**: Executes all generated queries simultaneously using Exa's search API
3. **URL Filtering**: Uses AI to analyze search results and extract only URLs relevant to your instruction
4. **Content Fetching**: Visits selected URLs in parallel to extract content
5. **Summarization**: Generates concise summaries of each page's content using OpenAI
6. **Results Compilation**: Formats all findings into a comprehensive report

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
- `max_urls`: Maximum URLs to visit
- Timeout values in `fetch_url_content()`
- OpenAI model and parameters in generation functions

## Dependencies

- `openai`: OpenAI API client
- `exa-py`: Exa search API client

## License

This project is open source and available under the MIT License.
