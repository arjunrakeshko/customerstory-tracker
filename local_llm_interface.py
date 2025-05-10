import json
import requests
from typing import List, Optional, Dict, Any
import numpy as np
from datetime import datetime
import re
from bs4 import BeautifulSoup
from urllib.parse import urlparse

class LocalLLMInterface:
    def __init__(self, model_name="mistral", base_url="http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.embedding_url = f"{base_url}/api/embeddings"
        self.generate_url = f"{base_url}/api/generate"

    def _call_llm(self, prompt: str) -> str:
        """Make a call to the local LLM."""
        try:
            response = requests.post(
                self.generate_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                }
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return ""

    def _call_embedding(self, text: str) -> List[float]:
        """Get embeddings from the local LLM."""
        try:
            response = requests.post(
                self.embedding_url,
                json={
                    "model": self.model_name,
                    "prompt": text
                }
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            return []

    def extract_title(self, text: str) -> str:
        """Headline string (≤120 chars)."""
        prompt = f"""Extract a concise headline (max 120 characters) that captures the main outcome:

{text}

Headline:"""
        return self._call_llm(prompt).strip()[:120]

    def extract_publication_date(self, text: str) -> str:
        """Return ISO date (YYYY-MM-DD) or ''."""
        prompt = f"""Extract the publication date in YYYY-MM-DD format. If no date is found, return an empty string:

{text}

Date:"""
        date_str = self._call_llm(prompt).strip()
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            return date_str
        except:
            return ""

    def extract_customer_name(self, text: str) -> str:
        """Company featured, e.g. 'Stripe'."""
        prompt = f"""Extract the main customer company name. Return only the company name, no additional text:

{text}

Customer name:"""
        return self._call_llm(prompt).strip()

    def extract_customer_location(self, text: str) -> str:
        """City/region/country or ''."""
        prompt = f"""Extract the customer's location. Use format: [City], [Country] or [Region], [Country]. If no location found, return empty string:

{text}

Location:"""
        return self._call_llm(prompt).strip()

    def extract_customer_industry(self, text: str) -> str:
        """Industry label, e.g. 'Fintech'."""
        prompt = f"""Extract the customer's industry. Use the most specific industry name mentioned:

{text}

Industry:"""
        return self._call_llm(prompt).strip()

    def extract_persona_title(self, text: str) -> str:
        """Most senior quoted job title."""
        prompt = f"""Extract the most senior job title mentioned. Include department if specified:

{text}

Job title:"""
        return self._call_llm(prompt).strip()

    def categorize_use_case(self, text: str) -> str:
        """Canonical use-case label."""
        prompt = f"""Categorize the main use case into a single, clear label that captures the primary goal:

{text}

Use case:"""
        return self._call_llm(prompt).strip()

    def extract_tags(self, text: str) -> List[str]:
        """3-7 thematic tags, lowercase snake_case."""
        prompt = f"""Extract 3-7 thematic tags that best describe the story. Format as lowercase snake_case:

{text}

Tags:"""
        tags = self._call_llm(prompt).strip().split("\n")
        return [tag.strip().lower().replace(" ", "_") for tag in tags if tag.strip()]

    def extract_benefits(self, text: str) -> List[str]:
        """2-5 concise metric-driven benefit strings."""
        prompt = f"""Extract 2-5 key benefits with specific metrics. Include numbers and percentages when available:

{text}

Benefits:"""
        benefits = self._call_llm(prompt).strip().split("\n")
        return [benefit.strip() for benefit in benefits if benefit.strip()]

    def extract_technologies(self, text: str) -> List[str]:
        """Tool / product names mentioned."""
        prompt = f"""Extract all technology, tool, and product names mentioned. Include version numbers if specified:

{text}

Technologies:"""
        techs = self._call_llm(prompt).strip().split("\n")
        return [tech.strip() for tech in techs if tech.strip()]

    def extract_partners(self, text: str) -> List[str]:
        """Partner company names."""
        prompt = f"""Extract all partner company names mentioned. Use official company names:

{text}

Partners:"""
        partners = self._call_llm(prompt).strip().split("\n")
        return [partner.strip() for partner in partners if partner.strip()]

    def embed_text(self, text: str) -> List[float]:
        """≥256-dimensional float vector."""
        return self._call_embedding(text)

# Create a singleton instance
llm = LocalLLMInterface()

# Export all functions at module level
extract_title = llm.extract_title
extract_publication_date = llm.extract_publication_date
extract_customer_name = llm.extract_customer_name
extract_customer_location = llm.extract_customer_location
extract_customer_industry = llm.extract_customer_industry
extract_persona_title = llm.extract_persona_title
categorize_use_case = llm.categorize_use_case
extract_tags = llm.extract_tags
extract_benefits = llm.extract_benefits
extract_technologies = llm.extract_technologies
extract_partners = llm.extract_partners
embed_text = llm.embed_text 