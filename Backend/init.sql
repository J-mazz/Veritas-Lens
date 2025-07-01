-- Veritas-Lens Database Initialization Script

-- Create database if not exists (this would be handled by the service creation)
-- CREATE DATABASE IF NOT EXISTS veritas_lens;

-- Create extension for UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL DEFAULT 'viewer' CHECK (role IN ('admin', 'annotator', 'viewer')),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- News sources table
CREATE TABLE IF NOT EXISTS news_sources (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    url VARCHAR(500) NOT NULL,
    rss_url VARCHAR(500),
    type VARCHAR(50) NOT NULL CHECK (type IN ('rss', 'scraping', 'api')),
    is_active BOOLEAN DEFAULT true,
    last_fetched_at TIMESTAMP,
    configuration JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Articles table
CREATE TABLE IF NOT EXISTS articles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    url VARCHAR(500) UNIQUE NOT NULL,
    source VARCHAR(255) NOT NULL,
    author VARCHAR(255),
    description TEXT,
    published_at TIMESTAMP,
    bias_score DECIMAL(3,2), -- -1.00 to 1.00
    bias_label VARCHAR(20) CHECK (bias_label IN ('left', 'center', 'right')),
    confidence DECIMAL(3,2) CHECK (confidence >= 0 AND confidence <= 1),
    is_labeled BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Bias predictions table
CREATE TABLE IF NOT EXISTS bias_predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    article_id UUID REFERENCES articles(id) ON DELETE CASCADE,
    bias_score DECIMAL(3,2) NOT NULL,
    bias_label VARCHAR(20) NOT NULL CHECK (bias_label IN ('left', 'center', 'right')),
    confidence DECIMAL(3,2) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    model_version VARCHAR(50) NOT NULL,
    predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model training data table
CREATE TABLE IF NOT EXISTS model_training_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    article_id UUID REFERENCES articles(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    label VARCHAR(20) NOT NULL CHECK (label IN ('left', 'center', 'right')),
    source VARCHAR(50) NOT NULL CHECK (source IN ('manual', 'active_learning', 'initial_dataset')),
    confidence DECIMAL(3,2) CHECK (confidence >= 0 AND confidence <= 1),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Active learning queries table
CREATE TABLE IF NOT EXISTS active_learning_queries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    article_id UUID REFERENCES articles(id) ON DELETE CASCADE,
    uncertainty DECIMAL(3,2) NOT NULL CHECK (uncertainty >= 0 AND uncertainty <= 1),
    query_strategy VARCHAR(50) NOT NULL CHECK (query_strategy IN ('uncertainty', 'entropy', 'margin')),
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'labeled', 'skipped')),
    labeled_by UUID REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    labeled_at TIMESTAMP
);

-- Scraping jobs table
CREATE TABLE IF NOT EXISTS scraping_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_id UUID REFERENCES news_sources(id) ON DELETE CASCADE,
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    articles_found INTEGER DEFAULT 0,
    articles_processed INTEGER DEFAULT 0,
    errors JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_articles_published_at ON articles(published_at);
CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source);
CREATE INDEX IF NOT EXISTS idx_articles_bias_label ON articles(bias_label);
CREATE INDEX IF NOT EXISTS idx_articles_is_labeled ON articles(is_labeled);
CREATE INDEX IF NOT EXISTS idx_articles_url ON articles(url);

CREATE INDEX IF NOT EXISTS idx_bias_predictions_article_id ON bias_predictions(article_id);
CREATE INDEX IF NOT EXISTS idx_bias_predictions_predicted_at ON bias_predictions(predicted_at);

CREATE INDEX IF NOT EXISTS idx_training_data_article_id ON model_training_data(article_id);
CREATE INDEX IF NOT EXISTS idx_training_data_source ON model_training_data(source);

CREATE INDEX IF NOT EXISTS idx_al_queries_status ON active_learning_queries(status);
CREATE INDEX IF NOT EXISTS idx_al_queries_uncertainty ON active_learning_queries(uncertainty);

CREATE INDEX IF NOT EXISTS idx_scraping_jobs_status ON scraping_jobs(status);
CREATE INDEX IF NOT EXISTS idx_scraping_jobs_source_id ON scraping_jobs(source_id);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_news_sources_updated_at BEFORE UPDATE ON news_sources FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_articles_updated_at BEFORE UPDATE ON articles FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert default admin user (password: admin123 - change this!)
INSERT INTO users (email, password, role) VALUES 
('admin@veritas-lens.com', '$2b$10$rQZ9fTlGxGzEJvQYqQK9ZeYn9WxSjCKZQZJkqZgZqZgZqZgZqZgZq', 'admin')
ON CONFLICT (email) DO NOTHING;

-- Insert default news sources
INSERT INTO news_sources (name, url, rss_url, type) VALUES 
('CNN', 'https://cnn.com', 'https://rss.cnn.com/rss/edition.rss', 'rss'),
('Fox News', 'https://foxnews.com', 'https://feeds.foxnews.com/foxnews/latest', 'rss'),
('NPR', 'https://npr.org', 'https://www.npr.org/rss/rss.php?id=1001', 'rss'),
('Reuters', 'https://reuters.com', 'https://feeds.reuters.com/Reuters/domesticNews', 'rss'),
('BBC', 'https://bbc.com', 'http://feeds.bbci.co.uk/news/rss.xml', 'rss'),
('The Guardian', 'https://theguardian.com', 'https://www.theguardian.com/us/rss', 'rss')
ON CONFLICT DO NOTHING;
