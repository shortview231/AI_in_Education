import praw
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import os

# --- YOU MUST REPLACE THE PLACEHOLDERS BELOW WITH YOUR ACTUAL CREDENTIALS ---
client_id = "n6R0HV920qP1cL9NKbfRHw"
client_secret = "z9w0Ry6BNCv4XHbFhrM2-qUUdQRyRwn"
username = "sdf"
password = "asdf"
user_agent = "AI_in_Education_by_Proud-Air5718"

# --- Connect to Reddit API ---
try:
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        username=username,
        password=password
    )
    print("Successfully connected to Reddit API.")
except Exception as e:
    print(f"Error connecting to Reddit: {e}")
    exit()

# --- Define Subreddits and Keywords ---
subreddits = ["Professors", "edtech", "AIethics", "education", "college", "ChatGPT"]
keywords = ["AI", "ChatGPT", "education", "lazy", "learning", "school", "cheating", "critical thinking"]

# --- Scrape and Filter Posts ---
def scrape_reddit_data():
    posts_data = []
    print("Starting data scraping...")
    for sub_name in subreddits:
        try:
            print(f"Scraping r/{sub_name}...")
            for submission in reddit.subreddit(sub_name).search(
                query=' OR '.join(keywords),
                limit=1000
            ):
                if submission.selftext:
                    posts_data.append({
                        'subreddit': sub_name,
                        'title': submission.title,
                        'selftext': submission.selftext,
                        'score': submission.score,
                        'created_utc': submission.created_utc,
                        'url': submission.url
                    })
        except Exception as e:
            print(f"Could not scrape r/{sub_name}: {e}")
            continue
    print("Scraping complete.")
    return pd.DataFrame(posts_data)

# --- Perform Sentiment Analysis ---
def add_sentiment_analysis(df):
    sia = SentimentIntensityAnalyzer()
    df['sentiment_score'] = df['selftext'].apply(lambda text: sia.polarity_scores(text)['compound'])
    df['sentiment_label'] = df['sentiment_score'].apply(
        lambda score: 'positive' if score > 0.05 else ('negative' if score < -0.05 else 'neutral')
    )
    return df

# --- Main Execution ---
if __name__ == "__main__":
    df = scrape_reddit_data()
    if not df.empty:
        df = add_sentiment_analysis(df)
        output_file_path = "ai_edu_scrape_recreated.csv"
        df.to_csv(output_file_path, index=False)
        print(f"Successfully scraped {len(df)} posts and saved them to {output_file_path}")
    else:
        print("No data was scraped. Please check your keywords and subreddit list.")