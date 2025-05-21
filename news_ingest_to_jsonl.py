import requests
import feedparser
import json
from datetime import datetime
from time import sleep

# ========== CONFIG ==========
# NewsAPI settings (get a free API key at https://newsapi.org/)
NEWSAPI_KEY = 'YOUR_NEWSAPI_KEY'
NEWSAPI_ENDPOINT = 'https://newsapi.org/v2/top-headlines'
NEWSAPI_SOURCES = ['bbc-news', 'cnn', 'fox-news', 'associated-press', 'the-hill', 'politico', 'npr'] # Add more as needed

# RSS feeds (add or replace with your faves)
RSS_FEEDS = [
    'http://feeds.bbci.co.uk/news/rss.xml',
    'https://rss.cnn.com/rss/cnn_topstories.rss',
    'http://feeds.foxnews.com/foxnews/latest',
    'https://feeds.npr.org/1001/rss.xml',
    'https://thehill.com/feed/',
    'https://www.politico.com/rss/politicopicks.xml',
    'https://rss.nytimes.com/services/xml/rss/nyt/Politics.xml'
]

OUTPUT_JSONL = 'news_articles.jsonl'
MAX_PER_SOURCE = 40   # Limit per API source to avoid quota issues

# ========== HELPERS ==========

def fetch_newsapi_articles():
    all_articles = []
    for source in NEWSAPI_SOURCES:
        params = {
            'apiKey': NEWSAPI_KEY,
            'sources': source,
            'pageSize': MAX_PER_SOURCE,
            'language': 'en'
        }
        try:
            r = requests.get(NEWSAPI_ENDPOINT, params=params)
            data = r.json()
            if data.get('status') == 'ok':
                for a in data['articles']:
                    article = {
                        'source': a.get('source', {}).get('name', source),
                        'author': a.get('author'),
                        'title': a.get('title'),
                        'published': a.get('publishedAt'),
                        'url': a.get('url'),
                        'content': a.get('content') or a.get('description', ''),
                        'ingest_method': 'newsapi'
                    }
                    all_articles.append(article)
            sleep(1) # be polite to the API!
        except Exception as e:
            print(f"Error fetching from NewsAPI source {source}: {e}")
    return all_articles

def fetch_rss_articles():
    all_articles = []
    for feed_url in RSS_FEEDS:
        try:
            d = feedparser.parse(feed_url)
            for entry in d.entries:
                article = {
                    'source': d.feed.get('title', feed_url),
                    'author': entry.get('author', None),
                    'title': entry.get('title'),
                    'published': entry.get('published', None),
                    'url': entry.get('link'),
                    'content': entry.get('summary', ''),
                    'ingest_method': 'rss'
                }
                all_articles.append(article)
        except Exception as e:
            print(f"Error parsing RSS feed {feed_url}: {e}")
    return all_articles

def isoformat_or_none(dt_str):
    # Attempt to parse any date string, return ISO format or None
    try:
        return datetime.fromisoformat(dt_str).isoformat()
    except Exception:
        try:
            # fallback for RSS: "Tue, 21 May 2024 12:00:00 GMT"
            from email.utils import parsedate_to_datetime
            return parsedate_to_datetime(dt_str).isoformat()
        except Exception:
            return None

# ========== MAIN SCRIPT ==========

if __name__ == "__main__":
    all_articles = []
    print("Fetching from NewsAPI...")
    all_articles += fetch_newsapi_articles()
    print("Fetching from RSS feeds...")
    all_articles += fetch_rss_articles()

    print(f"Total raw articles collected: {len(all_articles)}")

    # Standardize, filter, and write to JSONL
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
        count = 0
        for a in all_articles:
            if not a.get('content') or len(a['content']) < 40:
                continue  # skip near-empty articles
            record = {
                'source': a.get('source'),
                'author': a.get('author'),
                'title': a.get('title'),
                'published': isoformat_or_none(a.get('published')),
                'url': a.get('url'),
                'content': a.get('content'),
                'ingest_method': a.get('ingest_method')
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"Saved {count} cleaned articles to {OUTPUT_JSONL}")
