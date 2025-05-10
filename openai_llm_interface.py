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
            usage = dict(response.usage)
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
        prompt = f"""Extract the publication date from this text. Return only the date in YYYY-MM-DD format. If no date is found, return 'NA'.
        Look for dates in various formats and convert them to YYYY-MM-DD.
        Examples of valid dates to extract:
        - "Published on January 15, 2024" -> "2024-01-15"
        - "Posted: 2024-01-15" -> "2024-01-15"
        - "15th January 2024" -> "2024-01-15"
        - "01/15/2024" -> "2024-01-15"
        
        Text: {text}

        Date:"""
        date_str, _ = self._call_llm(prompt)
        date_str = date_str.strip()
        
        # Validate the date format
        try:
            if date_str != 'NA':
                datetime.strptime(date_str, "%Y-%m-%d")
            return date_str
        except ValueError:
            logger.warning(f"Invalid date format returned by LLM: {date_str}")
            return 'NA'

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

    def extract_title_from_html(self, html_content: str) -> str:
        """Extract a concise title from the HTML content."""
        prompt = f"""Extract a concise title (max 120 characters) that captures the main outcome from the following HTML content:

{html_content}

Title:"""
        title, _ = self._call_llm(prompt)
        return title.strip()[:120]

    def extract_story_from_html(self, text: str, url: str) -> Dict[str, Any]:
        """Extract story data from text content using LLM.
        
        Args:
            text: Cleaned text content of the page
            url: URL of the page
            
        Returns:
            Dictionary containing extracted story data with the following keys:
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
        """
        # Extract publication date using the dedicated method
        publication_date = self.extract_publication_date(text)
        
        # Extract customer name
        customer_prompt = f"""Extract the customer/company name from this text. Return only the name. Do not include the vendor company name if mentioned.
        
        Text: {text[:1000]}"""
        customer_name, _ = self._call_llm(customer_prompt)
        logger.info(f"[LLM RAW] customer_name: {repr(customer_name)}")
        customer_name = customer_name.strip()
        
        # Extract customer location
        location_prompt = f"""Extract the customer's city and country from this text. Return in format: city|country. If not found, use 'NA' for that field.
        Example: New York|USA or London|UK or NA|France
        
        Text: {text[:1000]}"""
        location, _ = self._call_llm(location_prompt)
        location = location.strip()
        customer_city, customer_country = location.split('|') if '|' in location else ('NA', 'NA')
        
        # Extract customer industry
        industry_prompt = f"""Extract the customer's industry from this text. Return only the industry name.
        
        Text: {text[:1000]}"""
        customer_industry, _ = self._call_llm(industry_prompt)
        logger.info(f"[LLM RAW] customer_industry: {repr(customer_industry)}")
        customer_industry = customer_industry.strip()
        
        # Extract persona information
        persona_prompt = f"""Extract the name and designation of the main persona mentioned in this text. Return in format: name|designation.
        Example: John Smith|CTO or Sarah Johnson|Head of AI
        
        Text: {text[:1000]}"""
        persona_info, _ = self._call_llm(persona_prompt)
        persona_info = persona_info.strip()
        persona_name, persona_designation = persona_info.split('|') if '|' in persona_info else ('NA', 'NA')
        
        # Extract use case
        use_case_prompt = f"""Extract the main use case from this text. Return a clear, concise description without quotes.
        
        Text: {text[:1000]}"""
        use_case, _ = self._call_llm(use_case_prompt)
        logger.info(f"[LLM RAW] use_case: {repr(use_case)}")
        use_case = use_case.strip().strip('"')
        
        # Extract benefits
        benefits_prompt = f"""Extract the key benefits mentioned in this text. Return as a JSON array of strings.
        Example: ["50% reduction in processing time", "Improved accuracy by 25%"]
        
        Text: {text[:1000]}"""
        benefits, _ = self._call_llm(benefits_prompt)
        try:
            benefits = json.loads(benefits)
        except json.JSONDecodeError:
            benefits = []
        
        # Extract benefit tags
        tags_prompt = f"""Extract key tags from these benefits that could be used for querying. Return as a JSON array of strings.
        Benefits: {json.dumps(benefits)}
        
        Example: ["performance", "cost-reduction", "scalability"]"""
        benefit_tags, _ = self._call_llm(tags_prompt)
        try:
            benefit_tags = json.loads(benefit_tags)
        except json.JSONDecodeError:
            benefit_tags = []
        
        # Extract technologies
        tech_prompt = f"""Extract the technologies mentioned in this text. Return as a JSON array of strings.
        Example: ["TensorFlow 2.0", "PyTorch", "CUDA"]
        
        Text: {text[:1000]}"""
        technologies, _ = self._call_llm(tech_prompt)
        try:
            technologies = json.loads(technologies)
        except json.JSONDecodeError:
            technologies = []
        
        # Extract partners
        partners_prompt = f"""Extract the partners mentioned in this text. Return as a JSON array of strings. Do not include the vendor company name if mentioned.
        Example: ["NVIDIA", "Microsoft Azure", "AWS"]
        
        Text: {text[:1000]}"""
        partners, _ = self._call_llm(partners_prompt)
        try:
            partners = json.loads(partners)
        except json.JSONDecodeError:
            partners = []
        
        return {
            'publication_date': publication_date,
            'full_text': text,
            'customer_name': customer_name,
            'customer_city': customer_city,
            'customer_country': customer_country,
            'customer_industry': customer_industry,
            'persona_name': persona_name,
            'persona_designation': persona_designation,
            'use_case': use_case,
            'benefits': benefits,
            'benefit_tags': benefit_tags,
            'technologies': technologies,
            'partners': partners
        }

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