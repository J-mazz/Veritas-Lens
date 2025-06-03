import feedparser
import pandas as pd
import re
from datetime import datetime

# List of major world/national news RSS feeds (English)
RSS_FEEDS = [
    "http://feeds.bbci.co.uk/news/world/rss.xml",
    "http://rss.cnn.com/rss/edition_world.rss",
    "https://www.reutersagency.com/feed/?best-topics=world&post_type=best",
    "https://www.theguardian.com/world/rss",
    "https://abcnews.go.com/abcnews/internationalheadlines",
    "https://www.cbc.ca/cmlink/rss-world",
    "https://www.npr.org/rss/rss.php?id=1004",
    "https://www.smh.com.au/rss/world.xml",
    "https://www.aljazeera.com/xml/rss/all.xml",
    # Add more as needed
]

def clean_text(text):
    # Remove HTML/XML tags
    text = re.sub(r"<[^>]+>", "", str(text))
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def parse_feed(url):
    feed = feedparser.parse(url)
    items = []
    for entry in feed.entries:
        title = entry.get('title', '')
        summary = entry.get('summary', '') or entry.get('description', '')
        link = entry.get('link', '')
        published = entry.get('published', '') or entry.get('updated', '')
        text = clean_text(f"{title}. {summary}")
        if text and len(text) > 50:  # Filter out non-news or very short blurbs
            items.append({
                "cleaned_text": text,
                "source": url,
                "url": link,
                "date": published
            })
    return items

def main(output_file):
    all_articles = []
    for rss in RSS_FEEDS:
        try:
            articles = parse_feed(rss)
            all_articles.extend(articles)
        except Exception as e:
            print(f"Failed to parse {rss}: {e}")
    df = pd.DataFrame(all_articles)
    # Deduplicate by text
    df = df.drop_duplicates(subset=['cleaned_text'])
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} scraped articles to {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", required=True, help="CSV file to write cleaned RSS news stories")
    args = parser.parse_args()
    main(args.output_file)
