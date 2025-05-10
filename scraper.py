import requests
from bs4 import BeautifulSoup
import yaml
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from datetime import datetime
import hashlib
import json
from urllib.parse import urljoin, urlparse
import time
import re
from abc import ABC, abstractmethod
from database import CustomerStoryDB
from processor import StoryProcessor

logger = logging.getLogger(__name__)

def extract_story_from_html(html_content: str, url: str) -> Optional[Dict[str, Any]]:
    """Extract story content from HTML.
    
    Args:
        html_content: Raw HTML content
        url: Source URL of the story
        
    Returns:
        Dictionary containing extracted story data or None if extraction fails
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove all media and interactive elements
        for element in soup.find_all([
            # Media elements
            'img', 'picture', 'video', 'audio', 'source', 'track',
            'canvas', 'svg', 'figure', 'figcaption',
            # Interactive elements
            'button', 'input', 'select', 'textarea', 'form',
            # Scripts and styles
            'script', 'style', 'noscript',
            # Navigation and UI elements
            'nav', 'footer', 'header', 'aside', 'sidebar', 'menu',
            # Embedding elements
            'iframe', 'embed', 'object', 'param',
            # Social media and sharing
            'div[class*="social"]', 'div[class*="share"]',
            'div[class*="related"]', 'div[class*="recommended"]',
            # Advertisement and popup elements
            'div[class*="cookie"]', 'div[class*="popup"]',
            'div[class*="banner"]', 'div[class*="ad"]',
            'div[class*="newsletter"]', 'div[class*="subscribe"]',
            # Media containers
            'div[class*="media"]', 'div[class*="gallery"]',
            'div[class*="carousel"]', 'div[class*="slider"]',
            'div[class*="video"]', 'div[class*="audio"]'
        ]):
            element.decompose()
        
        # Find the main content container
        main_content = None
        for container in ['main', 'article', 'div[class*="content"]', 'div[class*="story"]', 'div[class*="case-study"]']:
            main_content = soup.select_one(container)
            if main_content:
                break
        
        if not main_content:
            return None
        
        # Get the title
        title = soup.find('h1')
        title_text = title.text.strip() if title else ''
        
        # Extract only the main content text, preserving structure
        content_parts = []
        for element in main_content.find_all(['p', 'h2', 'h3', 'h4', 'blockquote', 'ul', 'ol']):
            if not element.text.strip():
                continue
            text = element.text.strip()
            text = re.sub(r'\s+', ' ', text)
            skip_patterns = [
                r'^share this story$', r'^related stories$', r'^download case study$',
                r'^contact us$', r'^learn more$', r'^read more$', r'^subscribe$', r'^follow us$',
                r'^click to enlarge$', r'^view image$', r'^play video$', r'^watch now$',
                r'^download media$', r'^download resources$'
            ]
            if any(re.match(pattern, text.lower()) for pattern in skip_patterns):
                continue
            content_parts.append(text)
        content = '\n'.join(content_parts)
        
        # Extract company information from URL
        parsed_url = urlparse(url)
        company_domain = parsed_url.netloc
        company_name = company_domain.split('.')[0].title()
        
        return {
            'title': title_text,
            'content': content,
            'url': url,
            'company_name': company_name,
            'company_domain': company_domain,
            'extraction_date': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error extracting story from HTML: {str(e)}")
        return None

class ScrapingStrategy(ABC):
    """Abstract base class for different scraping strategies"""
    
    def __init__(self, base_url: str, db: CustomerStoryDB, processor: StoryProcessor):
        self.base_url = base_url
        self.db = db
        self.processor = processor
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    @abstractmethod
    def scrape(self) -> List[Dict]:
        """Scrape stories using the specific strategy"""
        pass

    def _make_request(self, url: str) -> Optional[BeautifulSoup]:
        """Make HTTP request with error handling and rate limiting"""
        try:
            # Check if URL has been scraped before
            if self.db.is_url_scraped(url):
                logger.info(f"Skipping {url} - already scraped")
                return None

            # Check if URL is in processing status
            with self.db._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT status
                    FROM file_processing_status
                    WHERE url = ?
                """, (url,))
                row = cursor.fetchone()
                if row and row['status'] in ['processing', 'completed']:
                    logger.info(f"Skipping {url} - already being processed or completed")
                    return None

            time.sleep(1)  # Rate limiting
            response = self.session.get(url)
            response.raise_for_status()
            
            # Save raw HTML content with consistent naming
            html_dir = Path("data/html")
            html_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a safe filename from the URL using consistent hashing
            safe_filename = hashlib.md5(url.encode()).hexdigest()[:12] + ".html"
            html_path = html_dir / safe_filename
            
            # Save HTML with consistent encoding
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            # Add to processing status
            self.processor.update_file_status(str(html_path), 'pending', url=url)
            
            logger.info(f"Saved raw HTML to {html_path}")
            
            # Parse HTML with consistent parser
            return BeautifulSoup(response.text, 'html.parser')
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return None

class SubdirectoryCrawlStrategy(ScrapingStrategy):
    """Strategy for crawling subdirectories to find customer stories"""
    
    def scrape(self) -> List[Dict]:
        stories = []
        visited_urls = set()
        
        # Get the base domain for URL validation
        base_domain = urlparse(self.base_url).netloc
        
        # Start with the base URL
        urls_to_visit = [self.base_url]
        
        while urls_to_visit:
            current_url = urls_to_visit.pop(0)
            
            if current_url in visited_urls:
                continue
                
            visited_urls.add(current_url)
            logger.info(f"Crawling: {current_url}")
            
            soup = self._make_request(current_url)
            if not soup:
                continue
            
            # Extract all links from the page
            links = soup.find_all('a', href=True)
            
            for link in links:
                href = link['href']
                full_url = urljoin(current_url, href)
                
                # Only process URLs from the same domain and under the base path
                if (urlparse(full_url).netloc == base_domain and 
                    full_url.startswith(self.base_url) and 
                    full_url not in visited_urls):
                    
                    # Check if this is a story page (not a directory)
                    if self._is_story_page(full_url):
                        story = self._extract_story(full_url)
                        if story:
                            # Process the story using the processor
                            processed_story = self.processor.process_story(story)
                            if processed_story:
                                stories.append(processed_story)
                    else:
                        # Add to URLs to visit if it's a directory
                        urls_to_visit.append(full_url)
        
        return stories
    
    def _is_story_page(self, url: str) -> bool:
        """Determine if a URL is likely a story page rather than a directory"""
        # Common patterns that indicate a story page
        story_patterns = [
            r'/\d+$',  # Ends with numbers
            r'/[a-z-]+$',  # Ends with lowercase words and hyphens
            r'/story/',  # Contains 'story' in path
            r'/customer/',  # Contains 'customer' in path
            r'/case-study/',  # Contains 'case-study' in path
        ]
        
        return any(re.search(pattern, url) for pattern in story_patterns)
    
    def _extract_story(self, url: str) -> Optional[Dict]:
        """Extract story content from a page. Always save the input URL as the canonical story URL."""
        soup = self._make_request(url)
        if not soup:
            return None
        
        # Remove unwanted elements
        for element in soup.find_all([
            'script', 'style', 'nav', 'footer', 'header',  # Common structural elements
            'iframe', 'video', 'audio', 'img', 'picture',  # Media elements
            'aside', 'sidebar', 'menu', 'form',           # Side content
            'div[class*="cookie"]', 'div[class*="popup"]', # Popups and overlays
            'div[class*="banner"]', 'div[class*="ad"]',    # Ads and banners
            'div[class*="social"]', 'div[class*="share"]', # Social sharing
            'div[class*="related"]', 'div[class*="recommended"]' # Related content
        ]):
            element.decompose()
        
        # Find the main content container
        main_content = None
        for container in ['main', 'article', 'div[class*="content"]', 'div[class*="story"]', 'div[class*="case-study"]']:
            main_content = soup.select_one(container)
            if main_content:
                break
        
        if not main_content:
            return None
        
        # Get the title
        title = soup.find('h1')
        title_text = title.text.strip() if title else ''
        
        # Extract only the main content text, preserving structure
        content_parts = []
        for element in main_content.find_all(['p', 'h2', 'h3', 'h4', 'blockquote', 'ul', 'ol']):
            if not element.text.strip():
                continue
            text = element.text.strip()
            text = re.sub(r'\s+', ' ', text)
            skip_patterns = [
                r'^share this story$', r'^related stories$', r'^download case study$',
                r'^contact us$', r'^learn more$', r'^read more$', r'^subscribe$', r'^follow us$'
            ]
            if any(re.match(pattern, text.lower()) for pattern in skip_patterns):
                continue
            content_parts.append(text)
        content = '\n'.join(content_parts)
        
        # Extract company information from URL
        parsed_url = urlparse(url)
        company_domain = parsed_url.netloc
        company_name = company_domain.split('.')[0].title()
        
        # Always use the input URL as the canonical story URL
        return {
            'company_name': company_name,
            'company_website': f"https://{company_domain}",
            'company_industry': '',  # Will be filled by LLM
            'company_category': '',  # Will be filled by LLM
            'company_location': '',  # Will be filled by LLM
            'title': title_text,
            'publication_date': '',  # Will be filled by LLM
            'full_text': content,
            'customer_name': '',  # Will be filled by LLM
            'customer_location': '',  # Will be filled by LLM
            'customer_industry': '',  # Will be filled by LLM
            'persona_title': '',  # Will be filled by LLM
            'use_case_category': '',  # Will be filled by LLM
            'tags': [],  # Will be filled by LLM
            'benefits': [],  # Will be filled by LLM
            'technologies': [],  # Will be filled by LLM
            'partners': [],  # Will be filled by LLM
            'url': url  # Always use the input URL
        }

class Scraper:
    def __init__(self, config_path: str = "targets.yaml", db: Optional[CustomerStoryDB] = None, processor: Optional[StoryProcessor] = None):
        self.config_path = config_path
        self.config = self._load_config()
        self.db = db or CustomerStoryDB()
        self.processor = processor or StoryProcessor(llm_interface=None, db=self.db)
        self.strategies = {
            'subdirectory_crawl': SubdirectoryCrawlStrategy
        }
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/scraper.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _load_config(self) -> Dict[str, Any]:
        """Load the configuration from YAML file"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def scrape_company(self, company: Dict[str, str]) -> List[Dict[str, Any]]:
        """Scrape stories for a specific company"""
        company_name = company['name']
        base_url = company['base_url']
        strategy_type = company['type']

        self.logger.info(f"Scraping {company_name} using {strategy_type} strategy")
        
        strategy_class = self.strategies.get(strategy_type)
        if not strategy_class:
            self.logger.error(f"Unknown strategy type: {strategy_type}")
            return []

        strategy = strategy_class(base_url, self.db, self.processor)
        stories = strategy.scrape()
        
        # Add company information to each story if it's a dict
        for i, story in enumerate(stories):
            if isinstance(story, dict):
                story['company'] = company_name
            # else: it's likely an int (case study ID), do nothing
        
        self.logger.info(f"Found {len(stories)} stories for {company_name}")
        return stories

    def scrape(self) -> List[Dict[str, Any]]:
        """Scrape stories for all companies in the config"""
        all_stories = []
        for company in self.config:
            stories = self.scrape_company(company)
            all_stories.extend(stories)
        return all_stories

    def _extract_story(self, html_content: str, url: str) -> Optional[Dict[str, Any]]:
        """Extract story data from HTML content using LLM."""
        try:
            # Check if URL is already processed
            with self.db._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT f.status, f.case_study_id
                    FROM file_processing_status f
                    WHERE f.url = ? AND f.status = 'completed'
                """, (url,))
                row = cursor.fetchone()
                if row:
                    self.logger.info(f"URL already processed: {url}")
                    return None

            # Extract story data using LLM
            story_data = self.processor.llm_interface.extract_story_from_html(html_content, url)
            if not story_data:
                return None

            # Add URL to story data
            story_data['url'] = url
            return story_data

        except Exception as e:
            self.logger.error(f"Error extracting story from {url}: {str(e)}")
            return None

    def scrape_url(self, url: str) -> Optional[Dict[str, Any]]:
        """Scrape a single URL and extract story data."""
        try:
            # Check if URL is already processed
            with self.db._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT f.status, f.case_study_id
                    FROM file_processing_status f
                    WHERE f.url = ? AND f.status = 'completed'
                """, (url,))
                row = cursor.fetchone()
                if row:
                    self.logger.info(f"URL already processed: {url}")
                    return None

            # Get HTML content
            html_content = self._get_html_content(url)
            if not html_content:
                return None

            # Extract story data
            story_data = self._extract_story(html_content, url)
            if not story_data:
                return None

            # Save HTML content
            file_path = self._save_html_content(html_content, url)
            if not file_path:
                return None

            # Update file status to pending
            with self.db._get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO file_processing_status
                    (file_path, url, status, last_attempt)
                    VALUES (?, ?, 'pending', ?)
                """, (file_path, url, datetime.now().isoformat()))
                conn.commit()

            return story_data

        except Exception as e:
            self.logger.error(f"Error scraping {url}: {str(e)}")
            return None 