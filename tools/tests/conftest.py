import pytest
import logging
from pathlib import Path
import tempfile
import shutil
from ..logger_config import setup_logging

@pytest.fixture(scope="session", autouse=True)
def setup_test_logging():
    """Set up logging for tests"""
    # Create a temporary directory for test logs
    temp_log_dir = tempfile.mkdtemp()
    setup_logging(log_dir=temp_log_dir)
    
    yield
    
    # Clean up after tests
    shutil.rmtree(temp_log_dir)

@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture(scope="session")
def sample_html():
    """Sample HTML content for testing"""
    return """
    <html>
        <body>
            <article class="customer-story">
                <h1>Customer Success Story</h1>
                <div class="content">
                    <p>This is a test customer story about how our product helped a customer achieve great results.</p>
                    <blockquote>
                        <p>"The results have been amazing!"</p>
                        <cite>- John Doe, Head of Sales</cite>
                    </blockquote>
                    <p>Based in New York, the customer was able to increase efficiency by 90%.</p>
                </div>
            </article>
        </body>
    </html>
    """

@pytest.fixture(scope="session")
def sample_config():
    """Sample configuration for testing"""
    return """
    - name: Test Company
      base_url: https://test.com/customers
    - name: Another Company
      base_url: https://another.com/case-studies
    """

@pytest.fixture(scope="session")
def mock_llm_responses():
    """Sample LLM responses for testing"""
    return {
        'summary': 'A test summary of the customer story.',
        'value_props': [
            'Increased efficiency by 90%',
            'Reduced operational costs',
            'Improved customer satisfaction'
        ],
        'tags': ['efficiency', 'cost reduction', 'customer success'],
        'persona': 'Head of Sales',
        'location': 'New York, USA',
        'use_case': 'Sales Operations',
        'embedding': [0.1] * 384,  # Mock embedding
        'insight_score': 4.5
    } 