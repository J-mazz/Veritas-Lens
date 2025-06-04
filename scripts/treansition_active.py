"""
Fetch news articles using NewsAPI with debugging enabled.

Dependencies:
- requests
- json
- time (for polite API usage)

Installation:A
pip install requests
"""

import requests
import json
from time import sleep

# ========== CONFIG ==========
NEWSAPI_KEY = '67106eebc06f48d0a4afcebc5a5b11a6'  # Provided API key
NEWSAPI_ENDPOINT = 'https://newsapi.org/v2/top-headlines'
NEWSAPI_SOURCES = [
    'bbc-news', 'cnn', 'fox-news', 'associated-press',
    'the-hill', 'politico', 'npr'
]  # Add more as needed
MAX_PER_SOURCE = 40   # Limit per source to avoid quota issues
OUTPUT_JSONL = 'newsapi_articles.jsonl'

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

            # Debugging: Print API response for inspection
            print(f"Response from {source}: {json.dumps(data, indent=2)}")

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
            else:
                print(f"Error fetching articles for source {source}: {data.get('message')}")
            sleep(1)  # Be polite to the API!
        except Exception as e:
            print(f"Error fetching from NewsAPI source {source}: {e}")
    return all_articles

# ========== MAIN SCRIPT ==========
if __name__ == "__main__":
    articles = fetch_newsapi_articles()
    print(f"Fetched {len(articles)} articles from NewsAPI.")

    # Save to JSONL file
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
        for a in articles:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    print(f"Saved articles to {OUTPUT_JSONL}")
