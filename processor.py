import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime
import schedule
import time
from local_llm_interface import LocalLLMInterface
from openai_llm_interface import OpenAILLMInterface
from database import CustomerStoryDB
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

class StoryProcessor:
    def __init__(self, llm_interface: OpenAILLMInterface, db: CustomerStoryDB):
        self.db = db
        self.llm_interface = llm_interface
        self.logger = logging.getLogger(__name__)
        self.local_llm = LocalLLMInterface()  # Create instance for local processing

    def get_pending_files(self) -> List[Dict[str, Any]]:
        """Get list of files that need processing."""
        with self.db._get_connection() as conn:
            cursor = conn.execute("""
                SELECT file_path, url, status, error_message, last_attempt
                FROM file_processing_status
                WHERE status IN ('pending', 'failed')
                ORDER BY last_attempt ASC NULLS FIRST
            """)
            files = [dict(row) for row in cursor.fetchall()]
            
            # Log pending files
            pending_count = sum(1 for f in files if f['status'] == 'pending')
            failed_count = sum(1 for f in files if f['status'] == 'failed')
            
            if pending_count > 0:
                self.logger.info(f"Found {pending_count} pending files:")
                for f in files:
                    if f['status'] == 'pending':
                        self.logger.info(f"  - {f['file_path']} (URL: {f['url']})")
            
            if failed_count > 0:
                self.logger.info(f"Found {failed_count} failed files:")
                for f in files:
                    if f['status'] == 'failed':
                        self.logger.info(f"  - {f['file_path']} (URL: {f['url']})")
                        self.logger.info(f"    Error: {f['error_message']}")
            
            return files

    def get_failed_files(self) -> List[Dict[str, Any]]:
        """Get list of files that failed processing."""
        with self.db._get_connection() as conn:
            cursor = conn.execute("""
                SELECT file_path, url, error_message, last_attempt
                FROM file_processing_status
                WHERE status = 'failed'
                ORDER BY last_attempt ASC
            """)
            files = [dict(row) for row in cursor.fetchall()]
            
            if files:
                self.logger.info(f"Found {len(files)} failed files:")
                for f in files:
                    self.logger.info(f"  - {f['file_path']} (URL: {f['url']})")
                    self.logger.info(f"    Error: {f['error_message']}")
                    self.logger.info(f"    Last attempt: {f['last_attempt']}")
            else:
                self.logger.info("No failed files found")
            
            return files

    def update_file_status(self, file_path: str, status: str, error_message: Optional[str] = None, 
                          case_study_id: Optional[int] = None, url: Optional[str] = None) -> None:
        """Update the processing status of a file.
        
        Args:
            file_path: Path to the file
            status: One of 'pending', 'processing', 'completed', 'failed'
            error_message: Optional error message for failed status
            case_study_id: Optional ID of the processed case study
            url: Optional URL associated with the file
        """
        if status not in ['pending', 'processing', 'completed', 'failed']:
            raise ValueError(f"Invalid status: {status}. Must be one of: pending, processing, completed, failed")
            
        with self.db._get_connection() as conn:
            # If URL is not provided, try to get it from existing record
            if not url:
                cursor = conn.execute("""
                    SELECT url FROM file_processing_status WHERE file_path = ?
                """, (file_path,))
                row = cursor.fetchone()
                if row:
                    url = row['url']
            
            conn.execute("""
                INSERT OR REPLACE INTO file_processing_status
                (file_path, url, status, error_message, last_attempt, case_study_id)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (file_path, url, status, error_message, datetime.now().isoformat(), case_study_id))
            conn.commit()
            
            # Log the status update
            self.logger.info(f"Updated file status: {file_path} -> {status}")
            if error_message:
                self.logger.info(f"Error message: {error_message}")
            if url:
                self.logger.info(f"URL: {url}")

    def process_file(self, file_path: str, url: str) -> None:
        """Process a single HTML file with improved content extraction and fallback logic."""
        try:
            self.db.update_file_status(file_path, 'processing')
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()

            soup = BeautifulSoup(html_content, 'html.parser')
            # Remove unwanted elements
            for element in soup.find_all(['script', 'style', 'nav', 'footer', 'header', 'noscript', 'aside', 'sidebar', 'menu', 'picture', 'source', 'track', 'textarea', 'canvas', 'svg', 'figure', 'figcaption']):
                element.decompose()
            for div in soup.find_all('div', class_=lambda x: x and any(term in str(x).lower() for term in ['cookie', 'popup', 'ad', 'social', 'share', 'related', 'sidebar', 'menu', 'footer', 'header'])):
                div.decompose()

            # Try to find main content container
            main_content = None
            for selector in ['article', 'main', 'div[class*="content"]', 'div[class*="story"]', 'div[class*="case-study"]']:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            content = []
            if main_content:
                for tag in main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']):
                    text = tag.get_text(strip=True)
                    if text and not any(phrase in text.lower() for phrase in ['cookie', 'privacy', 'terms', 'subscribe', 'newsletter']):
                        content.append(text)
            clean_text = '\n'.join(content)

            # Fallback 1: meta description if main content is empty
            if not clean_text.strip():
                meta_desc = soup.find('meta', attrs={'name': 'description'})
                if meta_desc and meta_desc.get('content'):
                    clean_text = meta_desc['content'].strip()
                    self.logger.info(f"Used <meta name='description'> as fallback for {file_path}")

            # Fallback 2: title if still empty
            if not clean_text.strip():
                title_tag = soup.find('title')
                if title_tag and title_tag.text.strip():
                    clean_text = title_tag.text.strip()
                    self.logger.info(f"Used <title> as fallback for {file_path}")

            # Fallback 3: concatenate all <p> tags if still empty
            if not clean_text.strip():
                all_p = [p.get_text(strip=True) for p in soup.find_all('p') if p.get_text(strip=True)]
                if all_p:
                    clean_text = '\n'.join(all_p)
                    self.logger.info(f"Used all <p> tags as fallback for {file_path}")

            # Log cleaned text length and a sample
            self.logger.info(f"Cleaned text length for {file_path}: {len(clean_text)}")
            sample_text = clean_text[:200].replace('\n', ' ')
            self.logger.info(f"Sample cleaned text: {sample_text}...")

            # If still too short, skip LLM extraction
            if len(clean_text.strip()) < 100:
                msg = f"Cleaned text too short after all fallbacks for {file_path}. Skipping LLM extraction."
                self.logger.warning(msg)
                self.db.update_file_status(file_path, 'failed', msg)
                return

            # Extract story data using LLM
            story_data = self.llm_interface.extract_story_from_html(clean_text, url)
            # Add case study to database
            case_study_id = self.db.add_case_study(
                url=url,  # Use url parameter directly
                customer_name=story_data.get('customer_name'),
                customer_industry=story_data.get('customer_industry'),
                use_case=story_data.get('use_case'),
                benefits=story_data.get('benefits'),
                benefit_tags=story_data.get('benefit_tags'),
                technologies=story_data.get('technologies'),
                partners=story_data.get('partners')
            )
            self.db.update_file_status(file_path, 'completed')
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            self.db.update_file_status(file_path, 'failed', str(e))

    def reprocess_failed_files(self, html_dir: str = "data/html") -> None:
        """Reprocess files that previously failed."""
        html_dir = Path(html_dir)
        if not html_dir.exists():
            self.logger.error(f"HTML directory not found: {html_dir}")
            return

        # Get list of failed files
        failed_files = self.get_failed_files()
        if not failed_files:
            self.logger.info("No failed files to reprocess")
            return

        self.logger.info(f"Found {len(failed_files)} failed files to reprocess")
        
        for file_info in failed_files:
            file_path = file_info['file_path']
            url = file_info['url']
            try:
                # Process the file
                self.process_file(file_path, url)

            except Exception as e:
                self.logger.error(f"Error reprocessing {file_path}: {str(e)}")
                continue

    def resume_processing(self, html_dir: str = "data/html") -> None:
        """Resume processing of pending files."""
        html_dir = Path(html_dir)
        if not html_dir.exists():
            self.logger.error(f"HTML directory not found: {html_dir}")
            return

        # Get list of pending files
        pending_files = self.get_pending_files()
        if not pending_files:
            self.logger.info("No pending files to process")
            return

        self.logger.info(f"Found {len(pending_files)} pending files to process")
        
        for file_info in pending_files:
            file_path = file_info['file_path']
            url = file_info['url']
            try:
                # Process the file
                self.process_file(file_path, url)

            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {str(e)}")
                continue

    def process_story(self, story_data: Dict[str, Any]) -> Optional[int]:
        """Process a story and add it to the database.
        
        Args:
            story_data: Dictionary containing story data with the following keys:
                - url: URL of the case study
                - publication_date: Publication date in YYYY-MM-DD format
                - full_text: Full text content
                - customer_name: Name of the customer
                - customer_city: City of the customer
                - customer_country: Country of the customer
                - customer_industry: Industry of the customer
                - persona_name: Name of the main persona
                - persona_designation: Designation of the main persona
                - use_case: Main use case
                - benefits: List of benefits
                - benefit_tags: List of benefit tags
                - technologies: List of technologies
                - partners: List of partners
                
        Returns:
            ID of the processed case study if successful, None otherwise
        """
        try:
            # Add the case study to the database
            case_study_id = self.db.add_case_study(
                url=story_data.get('url'),
                customer_name=story_data.get('customer_name'),
                customer_city=story_data.get('customer_city'),
                customer_country=story_data.get('customer_country'),
                customer_industry=story_data.get('customer_industry'),
                persona_name=story_data.get('persona_name'),
                persona_designation=story_data.get('persona_designation'),
                use_case=story_data.get('use_case'),
                benefits=story_data.get('benefits', []),
                benefit_tags=story_data.get('benefit_tags', []),
                technologies=story_data.get('technologies', []),
                partners=story_data.get('partners', [])
            )

            return case_study_id
        except Exception as e:
            self.logger.error(f"Error processing story: {str(e)}")
            return None

    def get_insights(self) -> Dict[str, Any]:
        """Generate insights from the processed stories."""
        return self.db.get_insights()

    def get_similar_stories(self, case_study_id: int, limit: int = 5) -> list:
        """Get similar case studies."""
        return self.db.get_similar_case_studies(case_study_id, limit)

    def extract_title(self, text: str) -> str:
        """Extract the title from the text."""
        return self.local_llm.extract_title(text)

    def extract_customer_name(self, text: str) -> str:
        """Extract the customer name from the text."""
        return self.local_llm.extract_customer_name(text)

    def extract_customer_location(self, text: str) -> str:
        """Extract the customer's location from the text."""
        return self.local_llm.extract_customer_location(text)

    def extract_customer_industry(self, text: str) -> str:
        """Extract the customer's industry from the text."""
        return self.local_llm.extract_customer_industry(text)

    def extract_persona_title(self, text: str) -> str:
        """Extract the persona title from the text."""
        return self.local_llm.extract_persona_title(text)

    def categorize_use_case(self, text: str) -> str:
        """Categorize the use case from the text."""
        return self.local_llm.categorize_use_case(text)

    def extract_tags(self, text: str) -> List[str]:
        """Extract relevant tags from the text."""
        return self.local_llm.extract_tags(text)

    def extract_benefits(self, text: str) -> List[str]:
        """Extract benefits from the text."""
        return self.local_llm.extract_benefits(text)

    def extract_technologies(self, text: str) -> List[str]:
        """Extract technologies from the text."""
        return self.local_llm.extract_technologies(text)

    def extract_partners(self, text: str) -> List[str]:
        """Extract partners from the text."""
        return self.local_llm.extract_partners(text)

    def rate_insight_score(self, text: str) -> float:
        """Rate the insight score from 1-5."""
        prompt = f"""Rate the level of insight and detail in this case study on a scale of 1-5, where:
1 = Basic information only
2 = Some details and metrics
3 = Good details and specific outcomes
4 = Very detailed with specific metrics and outcomes
5 = Exceptional detail with comprehensive metrics and outcomes

{text}

Score (1-5):"""
        response, stats = self._call_llm(prompt)
        try:
            score = float(response.strip())
            return max(1.0, min(5.0, score))
        except ValueError:
            return 3.0

    def _call_llm(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """Make a call to the LLM and handle any errors."""
        try:
            return self.llm_interface._call_llm(prompt)
        except Exception as e:
            logger.error(f"Error calling LLM: {str(e)}")
            raise 