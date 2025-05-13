import requests
import pandas as pd
import json
from tqdm import tqdm
import os
from datetime import datetime
import time

import config


def collect_election_news(api_key, search_query, max_articles: int = 800):
    """
    Collect Indian election news from newsdata.io latest endpoint for a given search query.

    Args:
        api_key (str): newsdata.io API key
        search_query (str): URL-encoded search query for qInTitle
        max_articles (int): Maximum number of articles to collect
        timeframe (str): Timeframe for news articles (e.g., '48' for 48 hours)

    Returns:
        pd.DataFrame: DataFrame containing collected articles
    """
    base_url = "https://newsdata.io/api/1/latest"

    params = {
        "apikey": api_key,
        "qInTitle": search_query,
        "country": "in",
        "language": "as,bn,gu,hi,mr",
        "size": 10,  # Free plan limit
        "removeduplicate": 1,  # Remove duplicate articles
    }

    all_articles = []
    page = None
    articles_collected = 0

    while articles_collected < max_articles:
        if page:
            params["page"] = page

        try:
            response = requests.get(base_url, params=params)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success":
                    articles = data.get("results", [])
                    print(
                        f"Collected {len(articles)} articles for page: {page or 'initial'}"
                    )

                    # Add articles until max_articles is reached
                    for article in articles:
                        if articles_collected < max_articles:
                            all_articles.append(article)
                            articles_collected += 1
                        else:
                            break

                    # Check for next page
                    next_page = data.get("nextPage")
                    if not next_page or not articles:
                        print("No more pages or articles available.")
                        break
                    page = next_page
                else:
                    print(
                        f"Error in API response: {data.get('message', 'Unknown error')}"
                    )
                    break
            else:
                print(f"Error fetching data: {response.status_code} - {response.text}")
                break

            # Sleep for 30 seconds. Rate Limit on Free plan: 30 requests/15 minutes
            for _ in tqdm(range(61)):
                time.sleep(0.5)

        except Exception as e:
            print(f"Exception occurred: {str(e)}")
            break

    # Remove duplicates based on article_id
    unique_articles = []
    seen_ids = set()

    for article in all_articles:
        article_id = article.get("article_id")
        if article_id and article_id not in seen_ids:
            unique_articles.append(article)
            seen_ids.add(article_id)

    # Save to CSV and JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("data/raw", exist_ok=True)
    query_name = (
        search_query.replace('"', "").replace(" ", "_").replace("OR", "").lower()[:50]
    )
    csv_path = f"data/raw/election_news_{query_name}_{timestamp}.csv"
    json_path = f"data/raw/election_news_{query_name}_{timestamp}.json"

    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(unique_articles)
    if not df.empty:
        df.to_csv(csv_path, index=False, encoding="utf-8")

    # Save full JSON data
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(unique_articles, f, ensure_ascii=False, indent=4)

    print(f"Collected {len(unique_articles)} unique articles for query: {search_query}")
    print(f"Data saved to {csv_path} and {json_path}")

    return df


if __name__ == "__main__":
    # newsdata.io API key
    assert config.NEWS_API_KEY, "Please set the NEWS_API_KEY environment variable."
    API_KEY = config.NEWS_API_KEY

    modi_query = (
        '"Modi" OR "BJP" OR "Narendra Modi" OR "Narendra Damodardas Modi" OR '
        '"Prime Minister India" OR "PM Modi" OR "Bharatiya Janata Party" OR '
        '"saffron wave" OR "lotus bloom" OR "NDA"'
    )
    gandhi_query = '"Rahul Gandhi" OR "Congress Party" OR "Indian National Congress"'

    # Collect 800 articles for each query
    print("Collecting articles for Modi/BJP...")
    modi_df = collect_election_news(API_KEY, modi_query, max_articles=config.MAX_ARTICLES)
    print("\nCollecting articles for Gandhi/Congress...")
    gandhi_df = collect_election_news(API_KEY, gandhi_query, max_articles=config.MAX_ARTICLES)
