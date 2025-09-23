# api_news.py
import requests
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime, timedelta

NEWS_API_KEY = "95c4865233ed409c846c9f13edd9b4f0"

def get_weekly_news():
    """Fetch oil/gas news (last 7 days on free API), compute sentiment, aggregate weekly."""
    today = datetime.today().strftime("%Y-%m-%d")
    seven_days_ago = (datetime.today() - timedelta(days=7)).strftime("%Y-%m-%d")

    url = (
        "https://newsapi.org/v2/everything?"
        "q=oil+OR+gas+OR+OPEC+OR+fuel&"
        "searchIn=title,description&"
        "language=en&"
        "sortBy=publishedAt&"
        "pageSize=100&"
        f"from={seven_days_ago}&to={today}&"
        f"apiKey={NEWS_API_KEY}"
    )

    response = requests.get(url)
    data = response.json()

    if data.get("status") != "ok":
        raise ValueError(f"❌ API error: {data}")

    articles = data.get("articles", [])
    if not articles:
        return pd.DataFrame(columns=["date", "sentiment_mean", "neg_count", "pos_count", "volume"])

    # Build dataframe
    df_news = pd.DataFrame([{
        "date": art["publishedAt"][:10],
        "title": art["title"] or ""
    } for art in articles])
    df_news["date"] = pd.to_datetime(df_news["date"])

    # Sentiment analysis on titles
    sia = SentimentIntensityAnalyzer()
    df_news["sentiment"] = df_news["title"].apply(
        lambda x: sia.polarity_scores(str(x))["compound"]
    )

    # Weekly aggregation aligned to Mondays
    df_weekly = df_news.groupby(pd.Grouper(key="date", freq="W-MON")).agg(
        sentiment_mean=("sentiment", "mean"),
        neg_count=("sentiment", lambda x: (x < -0.2).sum()),
        pos_count=("sentiment", lambda x: (x > 0.2).sum()),
        volume=("sentiment", "count")
    ).reset_index()

    # Reindex to ensure every Monday in range exists
    all_mondays = pd.date_range(
        start=(datetime.today() - timedelta(days=7)),
        end=datetime.today(),
        freq="W-MON"
    )
    df_weekly = df_weekly.set_index("date").reindex(all_mondays).fillna(0).reset_index()
    df_weekly = df_weekly.rename(columns={"index": "date"})

    return df_weekly

# Test
if __name__ == "__main__":
    df_weekly = get_weekly_news()
    print("Weekly news sentiment:")
    print(df_weekly.tail())