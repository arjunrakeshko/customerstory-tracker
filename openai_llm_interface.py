import os
import json
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
from datetime import datetime
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import tiktoken
from dotenv import load_dotenv
import logging
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Cost per 1K tokens (as of 2024)
MODEL_COSTS = {
    "gpt-4": {
        "input": 0.03,  # $0.03 per 1K input tokens
        "output": 0.06,  # $0.06 per 1K output tokens
        "max_tokens": 8192
    },
    "gpt-4-turbo": {
        "input": 0.01,  # $0.01 per 1K input tokens
        "output": 0.03,  # $0.03 per 1K output tokens
        "max_tokens": 128000
    },
    "gpt-3.5-turbo": {
        "input": 0.0005,  # $0.0005 per 1K input tokens
        "output": 0.0015,  # $0.0015 per 1K output tokens
        "max_tokens": 16385
    },
    "text-embedding-ada-002": {
        "input": 0.0001,  # $0.0001 per 1K tokens
        "output": 0.0,  # No output tokens for embeddings
        "max_tokens": 8191
    }
}

def get_token_count(text: str, model: str = "gpt-4") -> int:
    """Count the number of tokens in a text string.
    
    Args:
        text: The text to count tokens for
        model: The OpenAI model to use for token counting
        
    Returns:
        Number of tokens in the text
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception as e:
        logger.error(f"Error counting tokens: {str(e)}")
        return 0

def get_cost_estimate(input_tokens: int, output_tokens: int, model: str = "gpt-4") -> float:
    """Estimate the cost of processing a given number of tokens.
    
    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model: The OpenAI model to use for cost estimation
        
    Returns:
        Estimated cost in USD
    """
    model_costs = MODEL_COSTS.get(model, MODEL_COSTS["gpt-4"])
    input_cost = (input_tokens / 1000) * model_costs["input"]
    output_cost = (output_tokens / 1000) * model_costs["output"]
    return input_cost + output_cost

class OpenAILLMInterface:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        self.client = OpenAI(api_key=self.api_key)
        self.model = "gpt-4-turbo-preview"
        self.embedding_model = "text-embedding-3-small"
        # Token/cost tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_embedding_tokens = 0
        self.total_cost = 0.0

    def _call_llm(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """Make a call to the OpenAI API and track token usage/cost."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1000
            )
            usage = response.usage._asdict()
            input_tokens = usage.get('prompt_tokens', 0)
            output_tokens = usage.get('completion_tokens', 0)
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            cost = get_cost_estimate(input_tokens, output_tokens, model=self.model)
            self.total_cost += cost
            logger.info(f"[OpenAI LLM] Input tokens: {input_tokens}, Output tokens: {output_tokens}, Cost: ${cost:.4f}")
            return response.choices[0].message.content, {
                'model': self.model,
                'usage': usage,
                'cost': cost
            }
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {str(e)}")
            raise

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embeddings for text using OpenAI's API and track token usage/cost."""
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            # The OpenAI embeddings API may not always return a usage object
            usage = getattr(response, 'usage', None)
            input_tokens = 0
            if usage:
                # Try to access as dict or object
                if isinstance(usage, dict):
                    input_tokens = usage.get('prompt_tokens', 0) or usage.get('total_tokens', 0) or 0
                else:
                    input_tokens = getattr(usage, 'prompt_tokens', 0) or getattr(usage, 'total_tokens', 0) or 0
            self.total_embedding_tokens += input_tokens
            # Embedding cost (input only)
            cost = get_cost_estimate(input_tokens, 0, model=self.embedding_model)
            self.total_cost += cost
            logger.info(f"[OpenAI Embedding] Input tokens: {input_tokens}, Cost: ${cost:.4f}")
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

    def get_usage_stats(self) -> Dict[str, Any]:
        """Return cumulative token usage and cost for this instance."""
        return {
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_embedding_tokens': self.total_embedding_tokens,
            'total_cost': self.total_cost
        }

    def extract_title(self, text: str) -> str:
        """Extract a concise title from the text."""
        prompt = f"""Extract a concise title (max 120 characters) that captures the main outcome:

{text}

Title:"""
        title, _ = self._call_llm(prompt)
        return title.strip()[:120]

    def extract_publication_date(self, text: str) -> str:
        """Extract publication date in YYYY-MM-DD format."""
        prompt = f"""Extract the publication date in YYYY-MM-DD format. If no date is found, return an empty string:

{text}

Date:"""
        date_str, _ = self._call_llm(prompt)
        try:
            datetime.strptime(date_str.strip(), "%Y-%m-%d")
            return date_str.strip()
        except:
            return ""

    def extract_customer_name(self, text: str) -> str:
        """Extract the customer company name."""
        prompt = f"""Extract the main customer company name. Return only the company name, no additional text:

{text}

Customer name:"""
        name, _ = self._call_llm(prompt)
        return name.strip()

    def extract_customer_location(self, text: str) -> str:
        """Extract the customer's location."""
        prompt = f"""Extract the customer's location. Use format: [City], [Country] or [Region], [Country]. If no location found, return empty string:

{text}

Location:"""
        location, _ = self._call_llm(prompt)
        return location.strip()

    def extract_customer_industry(self, text: str) -> str:
        """Extract the customer's industry."""
        prompt = f"""Extract the customer's industry. Use the most specific industry name mentioned:

{text}

Industry:"""
        industry, _ = self._call_llm(prompt)
        return industry.strip()

    def extract_persona_title(self, text: str) -> str:
        """Extract the most senior job title mentioned."""
        prompt = f"""Extract the most senior job title mentioned. Include department if specified:

{text}

Job title:"""
        title, _ = self._call_llm(prompt)
        return title.strip()

    def categorize_use_case(self, text: str) -> str:
        """Categorize the main use case."""
        prompt = f"""Categorize the main use case into a single, clear label that captures the primary goal:

{text}

Use case:"""
        use_case, _ = self._call_llm(prompt)
        return use_case.strip()

    def extract_tags(self, text: str) -> List[str]:
        """Extract thematic tags."""
        prompt = f"""Extract 3-7 thematic tags that best describe the story. Format as lowercase snake_case:

{text}

Tags:"""
        tags_text, _ = self._call_llm(prompt)
        tags = tags_text.strip().split("\n")
        return [tag.strip().lower().replace(" ", "_") for tag in tags if tag.strip()]

    def extract_benefits(self, text: str) -> List[str]:
        """Extract key benefits."""
        prompt = f"""Extract 2-5 key benefits with specific metrics. Include numbers and percentages when available:

{text}

Benefits:"""
        benefits_text, _ = self._call_llm(prompt)
        benefits = benefits_text.strip().split("\n")
        return [benefit.strip() for benefit in benefits if benefit.strip()]

    def extract_technologies(self, text: str) -> List[str]:
        """Extract technology names."""
        prompt = f"""Extract all technology, tool, and product names mentioned. Include version numbers if specified:

{text}

Technologies:"""
        techs_text, _ = self._call_llm(prompt)
        techs = techs_text.strip().split("\n")
        return [tech.strip() for tech in techs if tech.strip()]

    def extract_partners(self, text: str) -> List[str]:
        """Extract partner company names."""
        prompt = f"""Extract all partner company names mentioned. Use official company names:

{text}

Partners:"""
        partners_text, _ = self._call_llm(prompt)
        partners = partners_text.strip().split("\n")
        return [partner.strip() for partner in partners if partner.strip()]

    def extract_story_from_html(self, html_content: str, url: str) -> Optional[Dict[str, Any]]:
        """Extract story data from HTML content.
        
        Args:
            html_content: The HTML content to extract from
            url: The URL of the page
            
        Returns:
            Dictionary containing extracted story data or None if extraction fails
        """
        try:
            # Parse HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup.find_all(['script', 'style', 'nav', 'footer', 'header', 'iframe', 'video', 'audio', 'img', 'button', 'input', 'select', 'form']):
                element.decompose()
            
            # Remove social media sharing divs
            for div in soup.find_all('div', class_=lambda x: x and any(term in str(x).lower() for term in ['share', 'social', 'twitter', 'facebook', 'linkedin'])):
                div.decompose()
            
            # Remove advertisement divs
            for div in soup.find_all('div', class_=lambda x: x and any(term in str(x).lower() for term in ['ad', 'advertisement', 'banner', 'sponsored'])):
                div.decompose()
            
            # Remove media containers
            for div in soup.find_all('div', class_=lambda x: x and any(term in str(x).lower() for term in ['media', 'video', 'image', 'gallery'])):
                div.decompose()
            
            # Find main content container
            main_content = None
            for tag in ['article', 'main', 'div']:
                main_content = soup.find(tag, class_=lambda x: x and any(term in str(x).lower() for term in ['content', 'article', 'post', 'story']))
                if main_content:
                    break
            
            if not main_content:
                main_content = soup
            
            # Get clean text content
            clean_text = main_content.get_text(separator='\n', strip=True)
            
            # Remove common phrases
            skip_patterns = [
                r'click to enlarge',
                r'play video',
                r'watch video',
                r'view image',
                r'click here',
                r'read more',
                r'share this',
                r'follow us',
                r'subscribe',
                r'newsletter',
                r'cookie',
                r'privacy policy',
                r'terms of use',
                r'copyright',
                r'all rights reserved'
            ]
            
            for pattern in skip_patterns:
                clean_text = re.sub(pattern, '', clean_text, flags=re.IGNORECASE)
            
            # Clean up whitespace
            clean_text = re.sub(r'\n\s*\n', '\n\n', clean_text)
            clean_text = clean_text.strip()
            
            # Extract title
            title = self.extract_title(clean_text)
            if not title:
                logger.error("Failed to extract title")
                return None
            
            # Extract customer name
            customer_name = self.extract_customer_name(clean_text)
            
            # Extract publication date
            publication_date = self.extract_publication_date(clean_text)
            
            # Extract customer location
            customer_location = self.extract_customer_location(clean_text)
            
            # Extract customer industry
            customer_industry = self.extract_customer_industry(clean_text)
            
            # Extract persona title
            persona_title = self.extract_persona_title(clean_text)
            
            # Extract use case
            use_case = self.categorize_use_case(clean_text)
            
            # Extract tags
            tags = self.extract_tags(clean_text)
            
            # Extract benefits
            benefits = self.extract_benefits(clean_text)
            
            # Extract technologies
            technologies = self.extract_technologies(clean_text)
            
            # Extract partners
            partners = self.extract_partners(clean_text)
            
            # Return extracted data
            return {
                'url': url,
                'title': title,
                'full_text': clean_text,  # Store cleaned text content
                'customer_name': customer_name,
                'publication_date': publication_date,
                'customer_location': customer_location,
                'customer_industry': customer_industry,
                'persona_title': persona_title,
                'use_case': use_case,
                'benefits': benefits,
                'technologies': technologies,
                'partners': partners
            }
            
        except Exception as e:
            logger.error(f"Error extracting story from HTML: {str(e)}")
            return None

# Create a singleton instance
openai_llm = OpenAILLMInterface()

# Export all functions at module level
extract_title = openai_llm.extract_title
extract_publication_date = openai_llm.extract_publication_date
extract_customer_name = openai_llm.extract_customer_name
extract_customer_location = openai_llm.extract_customer_location
extract_customer_industry = openai_llm.extract_customer_industry
extract_persona_title = openai_llm.extract_persona_title
categorize_use_case = openai_llm.categorize_use_case
extract_tags = openai_llm.extract_tags
extract_benefits = openai_llm.extract_benefits
extract_technologies = openai_llm.extract_technologies
extract_partners = openai_llm.extract_partners
generate_embedding = openai_llm.generate_embedding 