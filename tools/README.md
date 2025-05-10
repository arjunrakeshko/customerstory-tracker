# Development and Debugging Tools

This directory contains utility scripts for development, testing, and debugging purposes.

## Available Tools

### `test_openai.py`
Tests the OpenAI API connection and configuration. Useful for:
- Verifying API key setup
- Testing API connectivity
- Checking token usage

### `setup_local_llm.py`
Sets up and configures a local LLM service using Ollama. Useful for:
- Installing Ollama
- Pulling required models
- Testing local LLM connectivity
- Setting up environment variables

### `query_stories.py`
Utility for querying and displaying customer stories from the database. Useful for:
- Testing database connectivity
- Viewing stored stories
- Debugging story processing
- Verifying data quality

### `logger_config.py`
Configuration for the application's logging system. Defines:
- Log formats
- Log levels
- Output handlers
- Log rotation settings

## Testing

The `tests/` directory contains test files and configuration:

### Test Files
- `conftest.py`: Test configuration and fixtures
- `test_basic.py`: Basic functionality tests

### Running Tests
```bash
# Run all tests
pytest tools/tests/

# Run specific test file
pytest tools/tests/test_basic.py

# Run with verbose output
pytest -v tools/tests/
```

## Usage

Each tool can be run independently:

```bash
# Test OpenAI API
python tools/test_openai.py

# Setup local LLM
python tools/setup_local_llm.py

# Query stories
python tools/query_stories.py
```

## Note
These tools are for development and debugging purposes only. They are not part of the main application workflow but are maintained for troubleshooting and testing. 