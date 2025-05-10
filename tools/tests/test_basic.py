import pytest
from pathlib import Path
import json
import tempfile
import shutil
from datetime import datetime
from ..database import CustomerStoryDB
from ..local_llm_interface import LocalLLMInterface
from ..scraper import CustomerStoryScraper
from ..processor import CustomerStoryProcessor

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test data"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def test_db(temp_dir):
    """Create a test database"""
    db_path = Path(temp_dir) / "test.db"
    return CustomerStoryDB(db_path=str(db_path))

@pytest.fixture
def mock_llm():
    """Create a mock LLM interface"""
    return LocalLLMInterface(mock=True)

@pytest.fixture
def test_story_data():
    """Sample story data for testing"""
    return {
        'company': 'Test Company',
        'url': 'https://test.com/story1',
        'text_excerpt': 'This is a test story about how Test Company helped Customer X achieve great results.',
        'date_added': datetime.now().isoformat()
    }

def test_database_operations(test_db, test_story_data):
    """Test basic database operations"""
    # Test adding a story
    story_id = test_db.add_story(test_story_data)
    assert story_id is not None
    
    # Test retrieving the story
    story = test_db.get_story(story_id)
    assert story is not None
    assert story['company'] == test_story_data['company']
    assert story['url'] == test_story_data['url']
    
    # Test getting all stories
    all_stories = test_db.get_all_stories()
    assert len(all_stories) == 1
    
    # Test getting stories by company
    company_stories = test_db.get_stories_by_company('Test Company')
    assert len(company_stories) == 1

def test_llm_interface(mock_llm, test_story_data):
    """Test LLM interface functions"""
    text = test_story_data['text_excerpt']
    
    # Test all LLM functions
    summary = mock_llm.summarize_text(text)
    assert isinstance(summary, str)
    
    value_props = mock_llm.extract_value_props(text)
    assert isinstance(value_props, list)
    
    tags = mock_llm.classify_tags(text)
    assert isinstance(tags, list)
    
    persona = mock_llm.extract_persona(text)
    assert isinstance(persona, str)
    
    location = mock_llm.extract_geo_location(text)
    assert isinstance(location, str)
    
    use_case = mock_llm.categorize_use_case(text)
    assert isinstance(use_case, str)
    
    embedding = mock_llm.embed_text(text)
    assert isinstance(embedding, list)
    assert len(embedding) == 384  # Default embedding size
    
    score = mock_llm.rate_insight_score(text)
    assert isinstance(score, float)
    assert 1.0 <= score <= 5.0

def test_processor_integration(temp_dir, mock_llm):
    """Test the processor's integration with other components"""
    # Create a test config file
    config_path = Path(temp_dir) / "test_config.yaml"
    config_content = """
    - name: Test Company
      base_url: https://test.com/customers
    """
    config_path.write_text(config_content)
    
    # Initialize processor with mock components
    processor = CustomerStoryProcessor(
        config_path=str(config_path),
        mock_llm=True
    )
    
    # Test story processing
    test_story = {
        'company': 'Test Company',
        'url': 'https://test.com/story1',
        'text_excerpt': 'Test story content',
        'date_added': datetime.now().isoformat()
    }
    
    processed_story = processor.process_story(test_story)
    assert 'summary' in processed_story
    assert 'value_props' in processed_story
    assert 'tags' in processed_story
    assert 'persona_title' in processed_story
    assert 'geo_location' in processed_story
    assert 'use_case_category' in processed_story
    assert 'positioning_vector' in processed_story
    assert 'insight_score' in processed_story

def test_insights_generation(temp_dir, test_db, test_story_data):
    """Test insights generation"""
    # Add test stories
    test_db.add_story(test_story_data)
    
    # Create processor with test database
    processor = CustomerStoryProcessor(
        config_path=str(Path(temp_dir) / "test_config.yaml"),
        mock_llm=True
    )
    
    # Generate insights
    insights = processor.generate_insights()
    
    # Verify insights structure
    assert 'timestamp' in insights
    assert 'total_stories' in insights
    assert 'companies' in insights
    assert 'tags' in insights
    assert 'top_stories' in insights
    assert 'geo_distribution' in insights
    assert 'persona_distribution' in insights
    assert 'use_case_distribution' in insights
    
    # Verify insights content
    assert insights['total_stories'] > 0
    assert 'Test Company' in insights['companies']
    assert len(insights['top_stories']) > 0 