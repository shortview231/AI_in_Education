import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# --- Step 1: Download VADER lexicon once for sentiment analysis ---
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except nltk.downloader.DownloadError:
    nltk.download('vader_lexicon')

# --- Step 2: Define Sentiment Analysis Function ---
def add_sentiment_analysis(df):
    sia = SentimentIntensityAnalyzer()
    
    # Check if 'text' column exists, otherwise use 'selftext' or similar
    text_column = 'text' if 'text' in df.columns else 'selftext'
    if text_column not in df.columns:
        print("Error: The CSV file does not contain a 'text' or 'selftext' column for analysis.")
        return df

    # Calculate a compound sentiment score for each post
    df['sentiment_score'] = df[text_column].apply(lambda text: sia.polarity_scores(str(text))['compound'])
    
    # Label the sentiment based on the score
    df['sentiment_label'] = df['sentiment_score'].apply(
        lambda score: 'positive' if score > 0.05 else ('negative' if score < -0.05 else 'neutral')
    )
    return df

# --- Step 3: Main Execution ---
if __name__ == "__main__":
    # --- Load the existing CSV file ---
    # The script assumes your data is in the data/raw/ folder, relative to the project root.
    try:
        input_file_path = "data/raw/ai_edu_scrape_combined.csv"
        df = pd.read_csv(input_file_path)
        print(f"Successfully loaded {len(df)} posts for analysis.")
    except FileNotFoundError:
        print(f"Error: The file {input_file_path} was not found. Please check your file path and that you are in the project's root folder.")
        exit()

    # --- Run sentiment analysis ---
    df_with_sentiment = add_sentiment_analysis(df)
    
    # --- Save the results to a new CSV ---
    # The output file path is set to the data/processed/ folder.
    output_file_path = "data/processed/analyzed_sentiment.csv"
    df_with_sentiment.to_csv(output_file_path, index=False)
    
    print(f"Sentiment analysis complete. Results saved to {output_file_path}")
    print("\nSentiment Distribution:")
    print(df_with_sentiment['sentiment_label'].value_counts())