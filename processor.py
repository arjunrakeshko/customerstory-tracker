import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime
import schedule
import time
from local_llm_interface import (
    extract_title as local_extract_title,
    extract_publication_date as local_extract_publication_date,
    extract_customer_name as local_extract_customer_name,
    extract_customer_location as local_extract_customer_location,
    extract_customer_industry as local_extract_customer_industry,
    extract_persona_title as local_extract_persona_title,
    categorize_use_case as local_categorize_use_case,
    extract_tags as local_extract_tags,
    extract_benefits as local_extract_benefits,
    extract_technologies as local_extract_technologies,
    extract_partners as local_extract_partners,
    embed_text as local_embed_text
)
from openai_llm_interface import OpenAILLMInterface
from database import CustomerStoryDB

logger = logging.getLogger(__name__)

class StoryProcessor:
    def __init__(self, llm_interface: OpenAILLMInterface, db: CustomerStoryDB):
        self.db = db
        self.llm_interface = llm_interface
        self.logger = logging.getLogger(__name__)

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

    def process_file(self, file_path: str, html_content: str, url: str) -> Optional[int]:
        """Process a single HTML file."""
        try:
            # Check if file is already processed
            with self.db._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT status, case_study_id, url
                    FROM file_processing_status
                    WHERE file_path = ?
                """, (file_path,))
                row = cursor.fetchone()
                if row and row['status'] == 'completed':
                    self.logger.info(f"File already processed: {file_path}")
                    return row['case_study_id']

            # Update status to processing
            self.update_file_status(file_path, 'processing', url=url)

            # Extract story data
            story_data = self.llm_interface.extract_story_from_html(html_content, url)
            if not story_data:
                self.update_file_status(file_path, 'failed', 'Failed to extract story data', url=url)
                return None

            # Process the story
            case_study_id = self.process_story(story_data)
            if case_study_id:
                self.update_file_status(file_path, 'completed', case_study_id=case_study_id, url=url)
                return case_study_id
            else:
                self.update_file_status(file_path, 'failed', 'Failed to process story', url=url)
                return None

        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {str(e)}")
            self.update_file_status(file_path, 'failed', str(e), url=url)
            return None

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
                # Read the HTML file
                with open(file_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                # Process the file
                case_study_id = self.process_file(file_path, html_content, url)
                if case_study_id:
                    self.logger.info(f"Successfully reprocessed {file_path} (ID: {case_study_id})")
                else:
                    self.logger.error(f"Failed to reprocess {file_path}")

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
                # Read the HTML file
                with open(file_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                # Process the file
                case_study_id = self.process_file(file_path, html_content, url)
                if case_study_id:
                    self.logger.info(f"Successfully processed {file_path} (ID: {case_study_id})")
                else:
                    self.logger.error(f"Failed to process {file_path}")

            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {str(e)}")
                continue

    def process_story(self, story_data: Dict[str, Any]) -> Optional[int]:
        """
        Process a story and store it in the database.
        
        Args:
            story_data: Dictionary containing story information with the following required fields:
                - url: str
                - title: str
                - full_text: str
                Optional fields:
                - customer_name: str (filled in by LLM extraction)
                - publication_date: str
                - customer_location: str
                - customer_industry: str
                - persona_title: str
                - use_case: str
                - benefits: List[str]
                - technologies: List[str]
                - partners: List[str]
                - company_id: int
        
        Returns:
            Optional[int]: The ID of the created case study if successful, None otherwise
        """
        try:
            # Validate required fields (customer_name is now optional, filled by LLM)
            required_fields = ['url', 'title', 'full_text']
            missing_fields = [field for field in required_fields if not story_data.get(field)]
            if missing_fields:
                raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

            # Check if URL already exists
            if self.db.is_url_scraped(story_data['url']):
                self.logger.info(f"URL already processed: {story_data['url']}")
                return None

            # Extract data with consistent defaults
            url = story_data['url']
            title = story_data['title']
            publication_date = story_data.get('publication_date', '')
            full_text = story_data['full_text']
            customer_name = story_data.get('customer_name', '')  # Now optional, filled by LLM
            
            # Optional fields with consistent defaults
            customer_location = story_data.get('customer_location', '')
            customer_industry = story_data.get('customer_industry', '')
            persona_title = story_data.get('persona_title', 'AI/ML Engineer')
            use_case = story_data.get('use_case', 'AI/ML Model Training')
            benefits = story_data.get('benefits', [])
            technologies = story_data.get('technologies', [])
            partners = story_data.get('partners', [])
            company_id = story_data.get('company_id', 1)

            # Generate embedding consistently
            try:
                embedding = self.llm_interface.generate_embedding(full_text)
            except Exception as e:
                self.logger.error(f"Failed to generate embedding: {str(e)}")
                return None

            # Add story to database consistently
            try:
                case_study_id = self.db.add_case_study(
                    url=url,
                    title=title,
                    publication_date=publication_date,
                    full_text=full_text,
                    customer_name=customer_name,
                    customer_location=customer_location,
                    customer_industry=customer_industry,
                    persona_title=persona_title,
                    use_case=use_case,
                    benefits=benefits,
                    technologies=technologies,
                    partners=partners,
                    company_id=company_id,
                    embedding=embedding
                )
                self.logger.info(f"Successfully processed story: {title} (ID: {case_study_id})")
                return case_study_id
            except Exception as e:
                self.logger.error(f"Failed to add case study to database: {str(e)}")
                return None

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
        prompt = f"""Extract the main title or headline from this text:

{text}

Title:"""
        response, stats = self._call_llm(prompt)
        return response.strip()

    def extract_customer_name(self, text: str) -> str:
        """Extract the customer name from the text."""
        prompt = f"""Extract the name of the customer or company being featured in this case study:

{text}

Customer name:"""
        response, stats = self._call_llm(prompt)
        return response.strip()

    def extract_customer_location(self, text: str) -> str:
        """Extract the customer's location from the text."""
        prompt = f"""Extract the location (city, country) of the customer or company being featured:

{text}

Location:"""
        response, stats = self._call_llm(prompt)
        return response.strip()

    def extract_customer_industry(self, text: str) -> str:
        """Extract the customer's industry from the text."""
        prompt = f"""Extract the industry or sector of the customer or company being featured:

{text}

Industry:"""
        response, stats = self._call_llm(prompt)
        return response.strip()

    def extract_persona_title(self, text: str) -> str:
        """Extract the persona title from the text."""
        prompt = f"""Extract the job title of the main contact or decision maker mentioned in this case study:

{text}

Persona title:"""
        response, stats = self._call_llm(prompt)
        return response.strip()

    def categorize_use_case(self, text: str) -> str:
        """Categorize the use case from the text."""
        prompt = f"""Categorize the main use case or application described in this case study:

{text}

Use case:"""
        response, stats = self._call_llm(prompt)
        return response.strip()

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

    def extract_tags(self, text: str) -> List[str]:
        """Extract relevant tags from the text."""
        prompt = f"""Extract 3-5 relevant tags that describe this case study:

{text}

Tags (one per line):"""
        response, stats = self._call_llm(prompt)
        return [tag.strip() for tag in response.split('\n') if tag.strip()]

    def _call_llm(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """Make a call to the LLM and handle any errors."""
        try:
            return self.llm_interface._call_llm(prompt)
        except Exception as e:
            logger.error(f"Error calling LLM: {str(e)}")
            raise 