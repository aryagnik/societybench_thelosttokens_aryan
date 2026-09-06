import requests
import trafilatura
from bs4 import BeautifulSoup
import json
from datetime import datetime
import uuid

from urllib.parse import urlparse


def get_source_name(url):
    domain = urlparse(url).netloc  # e.g. "indianexpress.com"
    # Remove "www." if present
    domain = domain.replace("www.", "")
    # Capitalize nicely
    source = domain.split(".")[0].replace("-", " ").title()
    return source


def scrape_article(url, query="NEET UG 2026 paper leak"):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }

    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()
    html = response.text
    soup = BeautifulSoup(html, "lxml")

    title = soup.title.get_text(strip=True) if soup.title else None
    text = trafilatura.extract(html, include_comments=False, include_tables=False)

    # Extract metadata
    source = get_source_name(url)
    
    published_date = None
    date_meta = soup.find("meta", {"property": "article:published_time"})
    if date_meta and date_meta.get("content"):
        published_date = date_meta["content"].split("T")[0]

    author_meta = soup.find("meta", {"name": "author"})
    author = author_meta["content"] if author_meta and author_meta.get("content") else "Unknown"

    return {
        "record_id": f"news_{uuid.uuid4().hex[:8]}",
        "source_type": "news",
        "source": source,
        "url": url,
        "title": title,
        "published_date": published_date if published_date else "Unknown",
        "author": author,
        "text": text,
        "query": query,
        "scraped_at": datetime.now().isoformat()
    }


if __name__ == "__main__":

    # url = "https://indianexpress.com/article/education/neet-ug-2026-paper-leak-nta-cancels-medical-entrance-new-dates-rajasthan-sog-neet-nta-nic-in-10685114/"
    url = "https://www.jagranjosh.com/news/neet-ug-leak-case-delhi-hc-orders-nta-to-declare-results-of-2-cbi-witnesses-187089"

    article = scrape_article(url)

    output_file = "../data/raw/news/article_001.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            article,
            f,
            ensure_ascii=False,
            indent=4
        )

    print("Article successfully saved!")
    print("Saved to:", output_file)