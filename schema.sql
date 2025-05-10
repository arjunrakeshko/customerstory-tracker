-- companies
CREATE TABLE companies(
  company_id INTEGER PRIMARY KEY,
  name TEXT,
  website_url TEXT,
  industry TEXT,
  category TEXT,
  headquarters_country TEXT  -- Changed from location to just country
);

-- industries
CREATE TABLE industries(
  industry_id INTEGER PRIMARY KEY,
  name TEXT UNIQUE,
  parent_id INTEGER REFERENCES industries(industry_id)  -- For hierarchical industry classification
);

-- standardized use cases
CREATE TABLE use_cases(
  use_case_id INTEGER PRIMARY KEY,
  name TEXT UNIQUE,
  description TEXT,
  parent_id INTEGER REFERENCES use_cases(use_case_id)  -- For hierarchical use case classification
);

-- standardized personas
CREATE TABLE personas(
  persona_id INTEGER PRIMARY KEY,
  title TEXT UNIQUE,
  description TEXT,
  level TEXT  -- e.g., 'C-Level', 'Director', 'Manager', 'Individual Contributor'
);

-- case studies
CREATE TABLE IF NOT EXISTS case_studies (
    case_study_id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id INTEGER,
    persona_id INTEGER,
    use_case_id INTEGER,
    url TEXT UNIQUE,
    title TEXT,
    publication_date TEXT,
    full_text TEXT,
    customer_name TEXT,
    customer_location TEXT,
    customer_industry TEXT,
    persona_title TEXT,
    use_case TEXT,
    embedding TEXT,
    date_added TEXT,
    FOREIGN KEY (company_id) REFERENCES companies(company_id),
    FOREIGN KEY (persona_id) REFERENCES personas(persona_id),
    FOREIGN KEY (use_case_id) REFERENCES use_cases(use_case_id)
);

-- Track which LLM functions have been applied to each case study
CREATE TABLE llm_processing_status(
  case_study_id INTEGER REFERENCES case_studies,
  function_name TEXT,               -- Name of the LLM function
  status TEXT,                      -- 'pending', 'processing', 'completed', 'failed'
  error_message TEXT,               -- Error message if status is 'failed'
  timestamp TEXT,                   -- ISO timestamp of last attempt
  PRIMARY KEY(case_study_id, function_name)
);

-- tags
CREATE TABLE tags(
  tag_id INTEGER PRIMARY KEY,
  tag_name TEXT UNIQUE
);

CREATE TABLE case_study_tags(
  case_study_id INTEGER REFERENCES case_studies,
  tag_id INTEGER REFERENCES tags,
  PRIMARY KEY(case_study_id, tag_id)
);

-- benefits
CREATE TABLE benefits(
  benefit_id INTEGER PRIMARY KEY,
  benefit_name TEXT UNIQUE
);

CREATE TABLE case_study_benefits(
  case_study_id INTEGER REFERENCES case_studies,
  benefit_id INTEGER REFERENCES benefits,
  PRIMARY KEY(case_study_id, benefit_id)
);

-- technologies
CREATE TABLE technologies(
  technology_id INTEGER PRIMARY KEY,
  technology_name TEXT UNIQUE
);

CREATE TABLE case_study_technologies(
  case_study_id INTEGER REFERENCES case_studies,
  technology_id INTEGER REFERENCES technologies,
  PRIMARY KEY(case_study_id, technology_id)
);

-- partners
CREATE TABLE partners(
  partner_id INTEGER PRIMARY KEY,
  partner_name TEXT UNIQUE
);

CREATE TABLE case_study_partners(
  case_study_id INTEGER REFERENCES case_studies,
  partner_id INTEGER REFERENCES partners,
  PRIMARY KEY(case_study_id, partner_id)
);

-- Track file processing status
CREATE TABLE file_processing_status(
  file_path TEXT PRIMARY KEY,
  url TEXT UNIQUE,
  status TEXT,                      -- 'pending', 'processing', 'completed', 'failed'
  error_message TEXT,               -- Error message if status is 'failed'
  last_attempt TEXT,                -- ISO timestamp of last attempt
  case_study_id INTEGER REFERENCES case_studies,  -- ID of processed case study if successful
  FOREIGN KEY (url) REFERENCES case_studies(url)
);

-- Create index for file processing status
CREATE INDEX idx_file_processing_status ON file_processing_status(status);

-- Create indexes for better query performance
CREATE INDEX idx_case_studies_url ON case_studies(url);
CREATE INDEX idx_case_studies_publication_date ON case_studies(publication_date);
CREATE INDEX idx_case_studies_customer_industry ON case_studies(customer_industry);
CREATE INDEX idx_case_studies_use_case ON case_studies(use_case);
CREATE INDEX idx_case_studies_persona ON case_studies(persona_title);
CREATE INDEX idx_case_studies_location ON case_studies(customer_location);
CREATE INDEX idx_benefits_name ON benefits(benefit_name);
CREATE INDEX idx_technologies_name ON technologies(technology_name);
CREATE INDEX idx_partners_name ON partners(partner_name);

-- Insert some common use cases
INSERT OR IGNORE INTO use_cases (name, description) VALUES
('AI/ML Model Training', 'Training and fine-tuning AI/ML models'),
('Data Processing', 'Processing and analyzing large datasets'),
('High Performance Computing', 'Running computationally intensive workloads'),
('Drug Discovery', 'Accelerating drug discovery and development'),
('Genomics Research', 'Processing and analyzing genomic data'),
('Financial Analysis', 'Financial modeling and analysis'),
('Natural Language Processing', 'Text analysis and language understanding'),
('Computer Vision', 'Image and video processing'),
('Generative AI', 'Creating new content using AI'),
('Scientific Computing', 'Scientific simulations and research');

-- Insert some common personas
INSERT OR IGNORE INTO personas (title, level, description) VALUES
('Chief Technology Officer', 'C-Level', 'Senior technology executive'),
('Chief Information Officer', 'C-Level', 'Senior IT executive'),
('Chief Data Officer', 'C-Level', 'Senior data management executive'),
('VP of Engineering', 'Executive', 'Senior engineering leader'),
('VP of Data Science', 'Executive', 'Senior data science leader'),
('Director of AI/ML', 'Director', 'AI/ML department leader'),
('Data Science Manager', 'Manager', 'Data science team leader'),
('AI/ML Engineer', 'Individual Contributor', 'AI/ML development specialist'),
('Data Scientist', 'Individual Contributor', 'Data analysis specialist'),
('Research Scientist', 'Individual Contributor', 'Research and development specialist');

-- Insert common benefits
INSERT OR IGNORE INTO benefits (benefit_name) VALUES
('Cost Reduction'),
('Performance Improvement'),
('Time Savings'),
('Scalability'),
('Reliability'),
('Security Enhancement'),
('Innovation'),
('Efficiency'),
('Quality'),
('Compliance');

-- Insert common technologies
INSERT OR IGNORE INTO technologies (technology_name) VALUES
('Machine Learning'),
('Deep Learning'),
('Data Processing'),
('Cloud Computing'),
('Big Data'),
('Natural Language Processing'),
('Computer Vision'),
('High Performance Computing'),
('Data Analytics'),
('Edge Computing'); 