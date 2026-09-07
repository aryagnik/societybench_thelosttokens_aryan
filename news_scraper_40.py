import requests
import trafilatura
from bs4 import BeautifulSoup
import json
import time
from datetime import datetime
import uuid
import pandas as pd
from urllib.parse import urlparse


def get_source_name(url):
    domain = urlparse(url).netloc
    domain = domain.replace("www.", "")
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


def scrape_from_excel(excel_path, output_file="../data/raw/news/articles.json", delay=2):
    df = pd.read_excel(excel_path)

    # Basic validation
    required_cols = {"url", "query"}
    if not required_cols.issubset(set(df.columns.str.lower())):
        raise ValueError(f"Excel file must contain columns: {required_cols}")

    # Normalize column names in case of case-mismatch
    df.columns = [c.lower() for c in df.columns]

    articles = []
    failed = []

    for idx, row in df.iterrows():
        url = str(row["url"]).strip()
        query = str(row["query"]).strip() if not pd.isna(row["query"]) else "NEET UG 2026 paper leak"

        if not url or url.lower() == "nan":
            continue

        print(f"[{idx + 1}/{len(df)}] Scraping: {url}")

        try:
            article = scrape_article(url, query=query)
            articles.append(article)
            print(f"  -> Success: {article['title']}")
        except Exception as e:
            print(f"  -> FAILED: {e}")
            failed.append({"url": url, "error": str(e)})

        # Be polite to servers, avoid getting blocked
        time.sleep(delay)

    # Save all articles as a list of dicts
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=4)

    print(f"\nDone. {len(articles)} succeeded, {len(failed)} failed.")
    print("Saved to:", output_file)

    if failed:
        failed_log = output_file.replace(".json", "_failed.json")
        with open(failed_log, "w", encoding="utf-8") as f:
            json.dump(failed, f, ensure_ascii=False, indent=4)
        print("Failed URLs logged to:", failed_log)

    return articles


if __name__ == "__main__":
    excel_path = "../data/raw/urls.csv"   # adjust path as needed
    output_file = "../data/raw/news/articles.json"

    scrape_from_excel(excel_path, output_file=output_file, delay=2)
    
    
# 