import logging
from dotenv import load_dotenv
from database import CustomerStoryDB
from processor import StoryProcessor
from scraper import Scraper
from tools.logger_config import setup_logging, get_logger

# Load environment variables
load_dotenv()

def main():
    """Default entry point: runs the full workflow (scrape, process, reprocess failed)."""
    # Set up logging
    setup_logging()
    logger = get_logger('main')

    # Interactive LLM selection
    print("Select LLM backend:")
    print("  1. OpenAI (default)")
    print("  2. Local LLM")
    choice = input("Enter 1 for OpenAI or 2 for Local LLM [1]: ").strip()
    if choice == '2' or choice.lower() == 'local':
        from local_llm_interface import LocalLLMInterface
        llm_interface = LocalLLMInterface()
        logger.info("Using Local LLM Interface")
    else:
        from openai_llm_interface import OpenAILLMInterface
        llm_interface = OpenAILLMInterface()
        logger.info("Using OpenAI LLM Interface")

    # Initialize components
    db = CustomerStoryDB()
    processor = StoryProcessor(llm_interface, db)
    scraper = Scraper(processor=processor)

    logger.info("--- Customer Story Tracker: Starting full workflow ---")

    # 1. Scrape new content
    logger.info("Starting scraping process...")
    stories = scraper.scrape()
    logger.info(f"Scraped {len(stories)} stories")

    # 2. Process any pending files
    logger.info("Processing any pending files...")
    processor.resume_processing("data/html")

    # 3. Reprocess failed files
    logger.info("Reprocessing failed files...")
    processor.reprocess_failed_files("data/html")

    # 4. Final status check
    logger.info("Final status check...")
    processor.get_pending_files()
    logger.info("--- Workflow complete ---")

    # 5. Show OpenAI token/cost usage if applicable
    if llm_interface.__class__.__name__ == "OpenAILLMInterface":
        stats = llm_interface.get_usage_stats()
        print("\n--- OpenAI Token Usage Summary ---")
        print(f"Total input tokens: {stats['total_input_tokens']}")
        print(f"Total output tokens: {stats['total_output_tokens']}")
        print(f"Total embedding tokens: {stats['total_embedding_tokens']}")
        print(f"Total estimated cost: ${stats['total_cost']:.4f}")
        logger.info(f"OpenAI Token Usage: {stats}")

if __name__ == "__main__":
    main() 