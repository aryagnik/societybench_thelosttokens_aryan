# Data Collection

The .py files in this directory correspond to `skills/web-search-crawl` and `skills/media-crawl`.
**Collection requires external tools**:

| Purpose | External tool | Required env |
|---|---|---|
| Web search + article crawling | Apify (Google Search Scraper + Website Content Crawler) | `APIFY_TOKEN` |
| Social-media collection | MediaCrawlerPro-Python | `MCP_HOME`, `MCP_SIGNSRV_URL` |

## Files

- `web_search_crawl_pipeline.py` — web-search-crawl orchestrator (Step 0→7; chains the sub-scripts below via subprocess)
- `search_bulk.py` — bulk keyword search (Apify Google Search calls)
- `run_content_crawler_batch.py` — web article crawling (Apify Website Content Crawler calls)
- `web_content_crawler_local.py` — local article crawling (no Apify)
- `apify_client.py` — thin wrapper around the Apify API client
- `media_crawl.py` — thin wrapper for social-media collection (delegates to MediaCrawlerPro-Python)

## Setup

1. Register at [Apify](https://apify.com/) to obtain a token
2. Deploy [MediaCrawlerPro-Python](https://github.com/...) (only if social media is needed)
3. Copy the root `config.example.env` to `.env` and fill in the credentials
