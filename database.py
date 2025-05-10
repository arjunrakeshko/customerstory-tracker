import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import hashlib
from pathlib import Path
import logging
import os
import shutil
import threading
import time

logger = logging.getLogger(__name__)

class CustomerStoryDB:
    def __init__(self, db_path: str = "data/customer_stories.db"):
        """Initialize the database connection.
        
        Args:
            db_path: Path to the SQLite database. If a 'new' version exists,
                    it will be used instead of creating a new database.
        """
        self.db_path = db_path
        self._ensure_data_dir()
        self._handle_existing_db()
        self._init_db()
        self._connection_pool = []
        self._max_connections = 5
        self._lock = threading.Lock()

    def _ensure_data_dir(self):
        """Ensure the data directory exists."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    def _handle_existing_db(self):
        """Handle existing database files to ensure consistent naming."""
        db_dir = os.path.dirname(self.db_path)
        db_name = os.path.basename(self.db_path)
        new_db_path = os.path.join(db_dir, db_name.replace('.db', '_new.db'))
        
        # If the new version exists and is not empty
        if os.path.exists(new_db_path) and os.path.getsize(new_db_path) > 0:
            # If the main db exists but is empty, remove it
            if os.path.exists(self.db_path) and os.path.getsize(self.db_path) == 0:
                os.remove(self.db_path)
            
            # If the main db doesn't exist, rename the new one
            if not os.path.exists(self.db_path):
                shutil.move(new_db_path, self.db_path)
                logger.info(f"Using existing database from {new_db_path}")
            else:
                # If both exist and main is not empty, keep both but log a warning
                logger.warning(f"Both {self.db_path} and {new_db_path} exist. Using {self.db_path}")

    def _init_db(self):
        """Initialize the database with the schema."""
        try:
            # Get the directory where this file (database.py) is located
            schema_path = "/Users/arjunrakesh/Documents/GitHub/layoftheland/customerstory-tracker/schema.sql"
            
            logger.info(f"Loading schema from {schema_path}")
            with open(schema_path, 'r') as f:
                schema = f.read()
            
            logger.info(f"Initializing database at {self.db_path}")
            with sqlite3.connect(self.db_path) as conn:
                # Execute the entire schema as a script
                try:
                    logger.info("Executing schema script")
                    conn.executescript(schema)
                    conn.commit()
                    logger.info("Schema execution completed")
                except sqlite3.OperationalError as e:
                    if "already exists" not in str(e):
                        logger.error(f"Error executing schema: {e}")
                        raise
                    logger.warning(f"Table already exists: {e}")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with proper settings."""
        with self._lock:
            # Try to get an existing connection from the pool
            while self._connection_pool:
                conn = self._connection_pool.pop()
                try:
                    # Test if connection is still valid
                    conn.execute("SELECT 1")
                    return conn
                except sqlite3.Error:
                    try:
                        conn.close()
                    except:
                        pass
                    continue
            
            # Create a new connection if pool is empty
            conn = sqlite3.connect(self.db_path, timeout=30)  # 30 second timeout
            conn.row_factory = sqlite3.Row
            return conn

    def _release_connection(self, conn: sqlite3.Connection):
        """Release a connection back to the pool."""
        with self._lock:
            if len(self._connection_pool) < self._max_connections:
                try:
                    # Test if connection is still valid
                    conn.execute("SELECT 1")
                    self._connection_pool.append(conn)
                except sqlite3.Error:
                    try:
                        conn.close()
                    except:
                        pass
            else:
                try:
                    conn.close()
                except:
                    pass

    def _execute_with_retry(self, query: str, params: tuple = None, max_retries: int = 3) -> sqlite3.Cursor:
        """Execute a query with retries for database locks."""
        for attempt in range(max_retries):
            try:
                conn = self._get_connection()
                try:
                    if params:
                        cursor = conn.execute(query, params)
                    else:
                        cursor = conn.execute(query)
                    conn.commit()
                    return cursor
                finally:
                    self._release_connection(conn)
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    time.sleep(1 * (attempt + 1))  # Exponential backoff
                    continue
                raise
            except Exception as e:
                self._release_connection(conn)
                raise

    def _generate_id(self, url: str) -> str:
        """Generate a unique ID for a story based on its URL"""
        return hashlib.md5(url.encode()).hexdigest()[:12]

    def add_company(self, name: str, website_url: str, industry: str, 
                   category: str, headquarters_country: str) -> int:
        """Add a company and return its ID."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """INSERT INTO companies (
                    name, website_url, industry, category, headquarters_country
                ) VALUES (?, ?, ?, ?, ?)""",
                (name, website_url, industry, category, headquarters_country)
            )
            conn.commit()
            return cursor.lastrowid

    def get_company_id(self, name: str) -> Optional[int]:
        """Get company ID by name."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT company_id FROM companies WHERE name = ?", (name,))
            row = cursor.fetchone()
            return row['company_id'] if row else None

    def add_industry(self, name: str, parent_id: Optional[int] = None) -> int:
        """Add an industry and return its ID."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "INSERT OR IGNORE INTO industries (name, parent_id) VALUES (?, ?)",
                (name, parent_id)
            )
            conn.commit()
            if cursor.lastrowid:
                return cursor.lastrowid
            # If industry already exists, get its ID
            cursor = conn.execute(
                "SELECT industry_id FROM industries WHERE name = ?",
                (name,)
            )
            return cursor.fetchone()['industry_id']

    def is_duplicate_content(self, title: str, full_text: str) -> bool:
        """Check if a case study with similar content already exists."""
        with self._get_connection() as conn:
            # First check exact title match
            cursor = conn.execute("SELECT case_study_id FROM case_studies WHERE title = ?", (title,))
            if cursor.fetchone():
                return True
            
            # Then check for similar content using text similarity
            # We'll use a simple approach: check if the first 100 characters match
            # This is a basic check - you might want to use more sophisticated methods
            content_preview = full_text[:100]
            cursor = conn.execute(
                "SELECT case_study_id FROM case_studies WHERE substr(full_text, 1, 100) = ?",
                (content_preview,)
            )
            return cursor.fetchone() is not None

    def _normalize_url(self, url: str) -> str:
        """Normalize a URL for duplicate checking by:
        1. Converting to lowercase
        2. Removing trailing slashes
        3. Removing common tracking parameters
        4. Removing www. prefix
        """
        url = url.lower().strip()
        url = url.rstrip('/')
        
        # Remove www. prefix
        if url.startswith('www.'):
            url = url[4:]
            
        # Remove common tracking parameters
        params_to_remove = ['utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content']
        if '?' in url:
            base_url, params = url.split('?', 1)
            param_dict = dict(param.split('=') for param in params.split('&') if '=' in param)
            filtered_params = {k: v for k, v in param_dict.items() if k not in params_to_remove}
            if filtered_params:
                url = base_url + '?' + '&'.join(f"{k}={v}" for k, v in filtered_params.items())
            else:
                url = base_url
                
        return url

    def is_url_scraped(self, url: str) -> bool:
        """Check if a URL has been scraped before"""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT 1 FROM case_studies WHERE url = ?
            """, (url,))
            return cursor.fetchone() is not None

    def add_case_study(self, url: str, customer_name: Optional[str] = None,
                      customer_industry: Optional[str] = None,
                      use_case: Optional[str] = None,
                      benefits: Optional[List[str]] = None,
                      benefit_tags: Optional[List[str]] = None,
                      technologies: Optional[List[str]] = None,
                      partners: Optional[List[str]] = None) -> int:
        """Add a new case study to the database or update if URL exists."""
        try:
            # Convert lists to JSON strings
            benefits_json = json.dumps(benefits) if benefits else None
            benefit_tags_json = json.dumps(benefit_tags) if benefit_tags else None
            technologies_json = json.dumps(technologies) if technologies else None
            partners_json = json.dumps(partners) if partners else None
            
            # Use INSERT OR REPLACE to handle existing URLs
            cursor = self._get_connection().cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO case_studies (
                    url, customer_name, customer_industry, use_case,
                    benefits, benefit_tags, technologies, partners,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """, (
                url, customer_name, customer_industry, use_case,
                benefits_json, benefit_tags_json, technologies_json, partners_json
            ))
            self._get_connection().commit()
            return cursor.lastrowid
        except Exception as e:
            self._get_connection().rollback()
            raise e

    def add_tag(self, tag_name: str) -> int:
        """Add a tag and return its ID."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "INSERT OR IGNORE INTO tags (tag_name) VALUES (?)",
                (tag_name,)
            )
            conn.commit()
            if cursor.lastrowid:
                return cursor.lastrowid
            # If tag already exists, get its ID
            cursor = conn.execute(
                "SELECT tag_id FROM tags WHERE tag_name = ?",
                (tag_name,)
            )
            return cursor.fetchone()['tag_id']

    def add_benefit(self, case_study_id: int, benefit_name: str) -> int:
        """Add a benefit to a case study."""
        with self._get_connection() as conn:
            # First, get or create the benefit
            cursor = conn.execute(
                "INSERT OR IGNORE INTO benefits (benefit_name) VALUES (?)",
                (benefit_name,)
            )
            conn.commit()
            
            if cursor.lastrowid:
                benefit_id = cursor.lastrowid
            else:
                # If benefit already exists, get its ID
                cursor = conn.execute(
                    "SELECT benefit_id FROM benefits WHERE benefit_name = ?",
                    (benefit_name,)
                )
                benefit_id = cursor.fetchone()['benefit_id']
            
            # Add the benefit to the case study
            conn.execute(
                "INSERT OR IGNORE INTO case_study_benefits (case_study_id, benefit_id) VALUES (?, ?)",
                (case_study_id, benefit_id)
            )
            conn.commit()
            return benefit_id

    def add_technology(self, case_study_id: int, technology_name: str) -> int:
        """Add a technology to a case study."""
        with self._get_connection() as conn:
            # First, get or create the technology
            cursor = conn.execute(
                "INSERT OR IGNORE INTO technologies (technology_name) VALUES (?)",
                (technology_name,)
            )
            conn.commit()
            
            if cursor.lastrowid:
                technology_id = cursor.lastrowid
            else:
                # If technology already exists, get its ID
                cursor = conn.execute(
                    "SELECT technology_id FROM technologies WHERE technology_name = ?",
                    (technology_name,)
                )
                technology_id = cursor.fetchone()['technology_id']
            
            # Add the technology to the case study
            conn.execute(
                "INSERT OR IGNORE INTO case_study_technologies (case_study_id, technology_id) VALUES (?, ?)",
                (case_study_id, technology_id)
            )
            conn.commit()
            return technology_id

    def add_partner(self, case_study_id: int, partner_name: str, role: str = 'Implementation') -> int:
        """Add a partner to a case study."""
        with self._get_connection() as conn:
            # First, get or create the partner
            cursor = conn.execute(
                """INSERT OR IGNORE INTO partners 
                   (partner_name, partner_type) 
                   VALUES (?, ?)""",
                (partner_name, 'Implementation Partner')
            )
            conn.commit()
            
            if cursor.lastrowid:
                partner_id = cursor.lastrowid
            else:
                # If partner already exists, get its ID
                cursor = conn.execute(
                    "SELECT partner_id FROM partners WHERE partner_name = ?",
                    (partner_name,)
                )
                partner_id = cursor.fetchone()['partner_id']
            
            # Add the partner to the case study with their role
            conn.execute(
                """INSERT OR IGNORE INTO case_study_partners 
                   (case_study_id, partner_id, role) 
                   VALUES (?, ?, ?)""",
                (case_study_id, partner_id, role)
            )
            conn.commit()
            return partner_id

    def get_case_study(self, case_study_id: int) -> Optional[Dict]:
        """Get a case study by ID."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT cs.*, c.name as company_name, c.website_url as company_website,
                       p.title as persona_title, p.level as persona_level,
                       uc.name as use_case_name, uc.description as use_case_description
                FROM case_studies cs
                JOIN companies c ON cs.company_id = c.company_id
                JOIN personas p ON cs.persona_id = p.persona_id
                JOIN use_cases uc ON cs.use_case_id = uc.use_case_id
                WHERE cs.case_study_id = ?
            """, (case_study_id,))
            row = cursor.fetchone()
            if not row:
                return None
                
            # Convert row to dict
            case_study = dict(row)
            
            # Get tags
            cursor = conn.execute("""
                SELECT t.tag_name
                FROM tags t
                JOIN case_study_tags cst ON t.tag_id = cst.tag_id
                WHERE cst.case_study_id = ?
            """, (case_study_id,))
            case_study['tags'] = [row['tag_name'] for row in cursor.fetchall()]
            
            # Get benefits
            cursor = conn.execute("""
                SELECT b.benefit_name, b.category
                FROM benefits b
                JOIN case_study_benefits csb ON b.benefit_id = csb.benefit_id
                WHERE csb.case_study_id = ?
            """, (case_study_id,))
            case_study['benefits'] = [dict(row) for row in cursor.fetchall()]
            
            # Get technologies
            cursor = conn.execute("""
                SELECT t.technology_name, t.category
                FROM technologies t
                JOIN case_study_technologies cst ON t.technology_id = cst.technology_id
                WHERE cst.case_study_id = ?
            """, (case_study_id,))
            case_study['technologies'] = [dict(row) for row in cursor.fetchall()]
            
            # Get partners
            cursor = conn.execute("""
                SELECT p.partner_name, p.partner_type, csp.role
                FROM partners p
                JOIN case_study_partners csp ON p.partner_id = csp.partner_id
                WHERE csp.case_study_id = ?
            """, (case_study_id,))
            case_study['partners'] = [dict(row) for row in cursor.fetchall()]
            
            return case_study

    def get_all_case_studies(self) -> List[Dict]:
        """Get all case studies."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT cs.*, c.name as company_name, c.website_url as company_website
                FROM case_studies cs
                JOIN companies c ON cs.company_id = c.company_id
            """)
            return [dict(row) for row in cursor.fetchall()]

    def get_case_studies_by_company(self, company_id: int) -> List[Dict]:
        """Get all case studies for a company."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT cs.*, c.name as company_name, c.website_url as company_website
                FROM case_studies cs
                JOIN companies c ON cs.company_id = c.company_id
                WHERE cs.company_id = ?
            """, (company_id,))
            return [dict(row) for row in cursor.fetchall()]

    def get_case_studies_by_tag(self, tag_name: str) -> List[Dict]:
        """Get all case studies with a specific tag."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT DISTINCT cs.*, c.name as company_name, c.website_url as company_website
                FROM case_studies cs
                JOIN companies c ON cs.company_id = c.company_id
                JOIN case_study_tags cst ON cs.case_study_id = cst.case_study_id
                JOIN tags t ON cst.tag_id = t.tag_id
                WHERE t.tag_name = ?
            """, (tag_name,))
            return [dict(row) for row in cursor.fetchall()]

    def get_similar_case_studies(self, case_study_id: int, limit: int = 5) -> List[Dict]:
        """Get similar case studies based on embedding similarity."""
        with self._get_connection() as conn:
            # Get the target case study's embedding
            cursor = conn.execute("""
                SELECT embedding_vector
                FROM case_studies
                WHERE case_study_id = ?
            """, (case_study_id,))
            row = cursor.fetchone()
            if not row:
                return []
            
            target_embedding = json.loads(row['embedding_vector'])
            
            # Get all other case studies
            cursor = conn.execute("""
                SELECT cs.*, c.name as company_name, c.website_url as company_website
                FROM case_studies cs
                JOIN companies c ON cs.company_id = c.company_id
                WHERE cs.case_study_id != ?
            """, (case_study_id,))
            
            # Calculate similarity scores
            similar_stories = []
            for row in cursor.fetchall():
                case_study = dict(row)
                case_study['embedding_vector'] = json.loads(case_study['embedding_vector'])
                similarity = self._cosine_similarity(target_embedding, case_study['embedding_vector'])
                case_study['similarity_score'] = similarity
                similar_stories.append(case_study)
            
            # Sort by similarity and return top N
            similar_stories.sort(key=lambda x: x['similarity_score'], reverse=True)
            return similar_stories[:limit]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        return dot_product / (norm1 * norm2)

    def get_insights(self) -> Dict:
        """Generate insights from the case studies."""
        with self._get_connection() as conn:
            # Get all case studies
            cursor = conn.execute("""
                SELECT cs.*, c.name as company_name,
                       p.title as persona_title, p.level as persona_level,
                       uc.name as use_case_name
                FROM case_studies cs
                LEFT JOIN companies c ON cs.company_id = c.company_id
                LEFT JOIN personas p ON cs.persona_id = p.persona_id
                LEFT JOIN use_cases uc ON cs.use_case_id = uc.use_case_id
            """)
            case_studies = [dict(row) for row in cursor.fetchall()]
            
            # Initialize insights
            insights = {
                'total_case_studies': len(case_studies),
                'company_distribution': {},
                'benefit_distribution': {},
                'technology_distribution': {},
                'partner_distribution': {},
                'use_case_distribution': {},
                'country_distribution': {},
                'persona_distribution': {},
                'industry_distribution': {}
            }
            
            for case_study in case_studies:
                # Company distribution
                company_name = case_study.get('company_name', 'Unknown')
                insights['company_distribution'][company_name] = insights['company_distribution'].get(company_name, 0) + 1
                
                # Benefit distribution
                cursor = conn.execute("""
                    SELECT b.benefit_name
                    FROM benefits b
                    JOIN case_study_benefits csb ON b.benefit_id = csb.benefit_id
                    WHERE csb.case_study_id = ?
                """, (case_study['case_study_id'],))
                for row in cursor.fetchall():
                    benefit = row['benefit_name']
                    insights['benefit_distribution'][benefit] = insights['benefit_distribution'].get(benefit, 0) + 1
                
                # Technology distribution
                cursor = conn.execute("""
                    SELECT t.technology_name
                    FROM technologies t
                    JOIN case_study_technologies cst ON t.technology_id = cst.technology_id
                    WHERE cst.case_study_id = ?
                """, (case_study['case_study_id'],))
                for row in cursor.fetchall():
                    tech = row['technology_name']
                    insights['technology_distribution'][tech] = insights['technology_distribution'].get(tech, 0) + 1
                
                # Partner distribution
                cursor = conn.execute("""
                    SELECT p.partner_name
                    FROM partners p
                    JOIN case_study_partners csp ON p.partner_id = csp.partner_id
                    WHERE csp.case_study_id = ?
                """, (case_study['case_study_id'],))
                for row in cursor.fetchall():
                    partner = row['partner_name']
                    insights['partner_distribution'][partner] = insights['partner_distribution'].get(partner, 0) + 1
                
                # Use case distribution
                use_case = case_study.get('use_case_name', 'Unknown')
                insights['use_case_distribution'][use_case] = insights['use_case_distribution'].get(use_case, 0) + 1
                
                # Country distribution
                if case_study.get('customer_location'):
                    country = case_study['customer_location'].split(',')[-1].strip()
                    insights['country_distribution'][country] = insights['country_distribution'].get(country, 0) + 1
                
                # Persona distribution
                persona = case_study.get('persona_title', 'Unknown')
                insights['persona_distribution'][persona] = insights['persona_distribution'].get(persona, 0) + 1
                
                # Industry distribution
                if case_study.get('customer_industry'):
                    insights['industry_distribution'][case_study['customer_industry']] = insights['industry_distribution'].get(case_study['customer_industry'], 0) + 1
            
            return insights

    def get_llm_processing_status(self, case_study_id: int, function_name: str) -> Optional[Dict[str, Any]]:
        """Get the processing status of an LLM function for a case study."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT status, error_message, timestamp
                FROM llm_processing_status
                WHERE case_study_id = ? AND function_name = ?
            """, (case_study_id, function_name))
            row = cursor.fetchone()
            return dict(row) if row else None

    def update_llm_processing_status(self, case_study_id: int, function_name: str, 
                                   status: str, error_message: Optional[str] = None):
        """Update the processing status of an LLM function for a case study."""
        with self._get_connection() as conn:
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO llm_processing_status
                    (case_study_id, function_name, status, error_message, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, (case_study_id, function_name, status, error_message, datetime.now().isoformat()))
                conn.commit()
                logger.info(f"Updated LLM processing status for case study {case_study_id}, function {function_name}: {status}")
            except Exception as e:
                logger.error(f"Error updating LLM processing status: {str(e)}")
                raise

    def needs_llm_processing(self, case_study_id: int, function_name: str) -> bool:
        """Check if a case study needs LLM processing for a specific function."""
        status = self.get_llm_processing_status(case_study_id, function_name)
        return status is None or status['status'] != 'success'

    def update_case_study(self, case_study_id: int, **kwargs):
        """Update a case study with new values."""
        if not kwargs:
            return
            
        with self._get_connection() as conn:
            # Build the SET clause
            set_clause = ', '.join(f"{k} = ?" for k in kwargs.keys())
            values = list(kwargs.values())
            values.append(case_study_id)
            
            conn.execute(f"""
                UPDATE case_studies
                SET {set_clause}
                WHERE case_study_id = ?
            """, values)
            conn.commit()

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
            
        with self._get_connection() as conn:
            # If URL is not provided, try to get it from existing record
            if not url:
                cursor = conn.execute("""
                    SELECT url FROM file_processing_status WHERE file_path = ?
                """, (file_path,))
                row = cursor.fetchone()
                if row:
                    url = row['url']
            
            # Update the file status
            conn.execute("""
                INSERT OR REPLACE INTO file_processing_status
                (file_path, url, status, error_message, last_attempt, case_study_id)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (file_path, url, status, error_message, datetime.now().isoformat(), case_study_id))
            conn.commit()
            
            # Log the status update
            logger.info(f"Updated file status: {file_path} -> {status}")
            if error_message:
                logger.info(f"Error message: {error_message}")
            if url:
                logger.info(f"URL: {url}")
            if case_study_id:
                logger.info(f"Case study ID: {case_study_id}") 