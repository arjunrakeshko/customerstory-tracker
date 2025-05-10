-- companies
CREATE TABLE companies(
  company_id INTEGER PRIMARY KEY,
  name TEXT,
  website_url TEXT,
  industry TEXT,
  category TEXT,
  headquarters_country TEXT
);

-- industries
CREATE TABLE industries(
  industry_id INTEGER PRIMARY KEY,
  name TEXT UNIQUE,
  parent_id INTEGER REFERENCES industries(industry_id)
);

-- standardized use cases
CREATE TABLE use_cases(
  use_case_id INTEGER PRIMARY KEY,
  name TEXT UNIQUE,
  description TEXT,
  parent_id INTEGER REFERENCES use_cases(use_case_id)
);

-- standardized personas
CREATE TABLE personas(
  persona_id INTEGER PRIMARY KEY,
  name TEXT,
  designation TEXT,
  level TEXT
);

-- case studies
CREATE TABLE IF NOT EXISTS case_studies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT UNIQUE NOT NULL,
    publication_date TEXT,
    full_text TEXT,
    customer_name TEXT,
    customer_city TEXT DEFAULT 'NA',
    customer_country TEXT DEFAULT 'NA',
    customer_industry TEXT,
    persona_name TEXT,
    persona_designation TEXT,
    use_case TEXT,
    benefits TEXT,
    benefit_tags TEXT,
    technologies TEXT,
    partners TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Track which LLM functions have been applied to each case study
CREATE TABLE llm_processing_status(
  case_study_id INTEGER REFERENCES case_studies,
  function_name TEXT,
  status TEXT,
  error_message TEXT,
  timestamp TEXT,
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
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  file_path TEXT UNIQUE NOT NULL,
  url TEXT NOT NULL,
  status TEXT NOT NULL,
  error_message TEXT,
  last_attempt TEXT,
  case_study_id INTEGER REFERENCES case_studies(id),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better query performance
CREATE INDEX idx_file_processing_status ON file_processing_status(status);
CREATE INDEX idx_case_studies_url ON case_studies(url);
CREATE INDEX idx_case_studies_publication_date ON case_studies(publication_date);
CREATE INDEX idx_case_studies_customer_industry ON case_studies(customer_industry);
CREATE INDEX idx_case_studies_use_case ON case_studies(use_case);
CREATE INDEX idx_case_studies_persona_designation ON case_studies(persona_designation);
CREATE INDEX idx_case_studies_customer_country ON case_studies(customer_country);
CREATE INDEX idx_case_studies_customer_city ON case_studies(customer_city);
CREATE INDEX idx_case_studies_benefit_tags ON case_studies(benefit_tags);
CREATE INDEX idx_benefits_name ON benefits(benefit_name);
CREATE INDEX idx_technologies_name ON technologies(technology_name);
CREATE INDEX idx_partners_name ON partners(partner_name);

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