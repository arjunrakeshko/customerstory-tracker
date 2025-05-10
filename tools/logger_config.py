import logging
import logging.handlers
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Any
import os

class CustomJSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage()
        }
        
        # Add extra fields if they exist
        if hasattr(record, 'extra'):
            log_data.update(record.extra)
            
        # Add exception info if it exists
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
            
        return json.dumps(log_data)

def setup_logging():
    """Configure logging for the application."""
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "customer_story_processor.log"),
            logging.StreamHandler()
        ]
    )
    
    # Configure specific loggers
    loggers = {
        'scraper': logging.getLogger('scraper'),
        'processor': logging.getLogger('processor'),
        'database': logging.getLogger('database')
    }
    
    for name, logger in loggers.items():
        logger.setLevel(logging.INFO)
        # Add file handler for each logger
        fh = logging.FileHandler(log_dir / f"{name}.log")
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name"""
    return logging.getLogger(name) 