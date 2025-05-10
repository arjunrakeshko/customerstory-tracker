# Customer Story Tracker

A tool for scraping, processing, and analyzing customer success stories and case studies. The system can use either a local LLM (Mistral via Ollama) or OpenAI's GPT-4o-mini API for processing stories.

## Features

- Scrapes customer stories from configured sources
- Processes stories to extract key information:
  - Title and publication date
  - Customer name, location, and industry
  - Use case categorization
  - Key benefits and metrics
  - Technologies and partners mentioned
  - Insight score (1-5)
  - Thematic tags
- Stores data in SQLite database
- Generates insights and analytics
- Supports both local LLM and OpenAI API processing

## Prerequisites

- Python 3.8+
- For local LLM: [Ollama](https://ollama.ai/) installed with Mistral model
- For OpenAI: Valid OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd customerstory-tracker
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up the local LLM (optional, if using OpenAI API):
```bash
# Install Ollama
curl https://ollama.ai/install.sh | sh

# Pull and run Mistral model
ollama pull mistral
ollama run mistral
```

## Configuration

1. Create a `targets.yaml` file with your target websites:
```yaml
websites:
  - name: "Example Corp"
    url: "https://example.com/case-studies"
    selectors:
      story_container: ".case-study"
      title: "h1"
      content: ".content"
```

2. (Optional) Set OpenAI API key as environment variable:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### Interactive Mode (Default)

Run the script without arguments to use interactive mode:
```bash
python main.py
```

This will:
1. Prompt you to choose between Local LLM and OpenAI API
2. If using OpenAI:
   - Ask for your API key (or use environment variable)
   - Let you set a token limit (default: 100,000)
3. Process stories and show progress
4. Generate insights and save them to `data/insights/`

### Non-Interactive Mode

Run with command-line arguments:
```bash
# Use local LLM
python main.py --non-interactive

# Use OpenAI API with custom token limit
python main.py --use-openai --token-limit 50000 --non-interactive

# Specify custom database path
python main.py --db-path custom/path/stories.db
```

### Command Line Options

- `--use-openai`: Use OpenAI API instead of local LLM
- `--non-interactive`: Run without prompts
- `--token-limit`: Set token limit for OpenAI API (default: 100,000)
- `--db-path`: Specify custom database path (default: data/customer_stories.db)

## Output

The system generates:
1. SQLite database with processed stories
2. JSON insights file in `data/insights/` with:
   - Story leaderboard
   - Publishing cadence
   - Geographic distribution
   - Industry distribution
   - Tag distribution
   - Average insight scores

## Cost Management (OpenAI API)

When using OpenAI API:
- Token usage is tracked and limited
- Cost estimates are provided
- Progress shows remaining tokens
- Final summary includes:
  - Total tokens used
  - Input/output token breakdown
  - Estimated cost
  - Remaining tokens

## Database Management

The system uses SQLite for data storage. The database file is located at `data/customer_stories.db` by default.

### Database Files
- `data/customer_stories.db`: Main database file
- If a `customer_stories_new.db` file exists and the main database is empty, the system will automatically use the new database file.

### Database Configuration
You can specify a custom database path using the `--db-path` argument:
```bash
python main.py --db-path /path/to/your/database.db
```

### Data Directory Structure
```
data/
├── customer_stories.db    # Main database file
├── html/                  # Cached HTML content
└── insights/             # Generated insights reports
```

## Development and Debugging Tools

The `tools/` directory contains utility scripts for development, testing, and debugging:

### Available Tools
- `test_openai.py`: Test OpenAI API connection and configuration
- `setup_local_llm.py`: Set up and configure local LLM service
- `query_stories.py`: Query and display customer stories from database

For more details about these tools, see [tools/README.md](tools/README.md).

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Your chosen license]

## Support

For issues and feature requests, please use the GitHub issue tracker. 