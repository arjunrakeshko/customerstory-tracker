# Customer Story Processing System

A system for processing and analyzing customer success stories from HTML files, extracting key information, and storing it in a structured database.

## Main Components

- `main.py` - Main script to process HTML files
- `processor.py` - Core processing logic for stories
- `database.py` - Database operations and schema management
- `schema.sql` - Database schema definition
- `scraper.py` - HTML content extraction and processing

## Prerequisites

- Python 3.8+
- Required packages:
  ```bash
  pip install beautifulsoup4 schedule
  ```

## Directory Structure

```
.
├── data/
│   ├── html/           # Place HTML files here
│   └── customer_stories.db  # SQLite database
├── main.py
├── processor.py
├── database.py
├── scraper.py
└── schema.sql
```

## How to Run

1. Place your HTML customer story files in the `data/html/` directory

2. Run the processing script:
   ```bash
   python main.py
   ```

3. The script will:
   - Process each HTML file
   - Extract key information using LLM-based extraction
   - Store data in SQLite database
   - Log processing results

## What Gets Processed

For each customer story, the system extracts:
- Basic information (title, date)
- Customer details (name, country, industry)
- Benefits achieved
- Technologies used
- Implementation partners
- Use case and persona information

## Database Output

The processed data is stored in a SQLite database (`data/customer_stories.db`) with the following main tables:
- `case_studies` - Main story information
- `benefits` - Extracted benefits
- `technologies` - Used technologies
- `partners` - Implementation partners
- `use_cases` - Categorized use cases
- `personas` - Target user personas

## Logging

The system logs processing information to the console, including:
- Number of files processed
- Success/failure status
- Any errors encountered

## Error Handling

- Invalid HTML files are logged but don't stop processing
- Database errors are caught and logged
- Processing continues even if individual files fail 