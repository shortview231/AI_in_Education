import os
import pandas as pd
import matplotlib.pyplot as plt

# === Dynamically locate the root of the repo ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "analyzed_sentiment.csv")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# === Create reports folder if missing ===
os.makedirs(REPORTS_DIR, exist_ok=True)

# === Load the dataset ===
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"❌ Error: File not found at {DATA_PATH}")
    exit()

# === Check for required columns ===
required_columns = ['sentiment_label', 'sentiment_score', 'created_utc', 'subreddit', 'score']
missing = [col for col in required_columns if col not in df.columns]
if missing:
    print(f"❌ Missing required columns: {missing}")
    exit()

# === Bar Chart: Sentiment Distribution ===
sentiment_counts = df['sentiment_label'].value_counts()

plt.figure(figsize=(10, 6))
sentiment_counts.plot(kind='bar', color=['#4CAF50', '#FF5722', '#607D8B'])
plt.title('Sentiment Distribution of Reddit Posts on AI in Education')
plt.xlabel('Sentiment')
plt.ylabel('Number of Posts')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/sentiment_distribution.png")
plt.show()

# === Pie Chart ===
plt.figure(figsize=(8, 8))
sentiment_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['#4CAF50', '#FF5722', '#607D8B'])
plt.title('Sentiment Proportion of Reddit Posts')
plt.ylabel('')
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/sentiment_pie_chart.png")
plt.show()

# === Line Chart: Sentiment Over Time ===
df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s')
df.set_index('created_utc', inplace=True)
sentiment_over_time = df.resample('W')['sentiment_score'].mean()

plt.figure(figsize=(12, 6))
sentiment_over_time.plot(kind='line', color='darkblue')
plt.title('Average Sentiment of Reddit Posts Over Time')
plt.xlabel('Date')
plt.ylabel('Average Sentiment Score')
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/sentiment_over_time.png")
plt.show()

# === Grouped Bar Chart: Sentiment by Subreddit ===
sentiment_by_subreddit = df.groupby('subreddit')['sentiment_label'].value_counts(normalize=True).unstack(fill_value=0)

sentiment_by_subreddit.plot(kind='bar', figsize=(12, 6), color=['#FF5722', '#607D8B', '#4CAF50'])
plt.title('Sentiment Distribution by Subreddit')
plt.xlabel('Subreddit')
plt.ylabel('Proportion of Posts')
plt.xticks(rotation=45)
plt.legend(title='Sentiment')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/sentiment_by_subreddit.png")
plt.show()

# === Scatter Plot: Post Score vs Sentiment Score ===
plt.figure(figsize=(10, 8))
plt.scatter(df['score'], df['sentiment_score'], alpha=0.5, color='darkgreen')
plt.title('Post Score vs Sentiment Score')
plt.xlabel('Post Score')
plt.ylabel('Sentiment Score')
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/score_vs_sentiment.png")
plt.show()

print("\n✅ All visualizations created and saved to the 'reports' folder:")
print(f"   {REPORTS_DIR}")
