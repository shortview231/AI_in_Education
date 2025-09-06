# Step 1: Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
import numpy as np

# --- Configuration ---
# Using the correct, full, hardcoded paths as specified.
ACT_CSV_PATH = '/home/robertsory/Desktop/DESKTOP/Projects/AI_in_Education/AI_in_Education/data/Test_Scores/raw/act_averages_test_takers_cleaned.csv'
SAT_CSV_PATH = '/home/robertsory/Desktop/DESKTOP/Projects/AI_in_Education/AI_in_Education/data/Test_Scores/raw/SAT_totals_participation_CLEAN_FIX.csv'
OUTPUT_DIR = '/home/robertsory/Desktop/DESKTOP/Projects/AI_in_Education/AI_in_Education/reports/Test_Scores/visualizations'


# --- Data Loading ---
def load_data():
    """
    Loads the ACT and SAT csv files into pandas DataFrames using their full paths.
    """
    print("--- Loading Data ---")
    try:
        act_df = pd.read_csv(ACT_CSV_PATH)
        print("✅ ACT data loaded successfully.")
        sat_df = pd.read_csv(SAT_CSV_PATH)
        print("✅ SAT data loaded successfully.")
        return act_df, sat_df
    except FileNotFoundError as e:
        print(f"❌ FATAL ERROR: Could not find a file.")
        print(f"Details: {e}")
        return None, None

# --- Visualization Functions ---
def plot_act_math_trend(df):
    """
    Creates and saves a line chart of average ACT Math scores over time.
    """
    print("\n--- Generating ACT Math Score Trend Chart ---")
    df = df.sort_values('Year')
    plt.figure(figsize=(12, 7))
    plt.plot(df['Year'], df['Math'], marker='o', linestyle='-', color='blue', linewidth=2.5, label='Math Score')
    plt.axvline(x=2014, color='black', linestyle='--', linewidth=2, label='Widespread Common Core Adoption (2014)')
    plt.title('National Average ACT Math Scores (1992-2022)', fontsize=16)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Average Math Score', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'act_math_scores_trend.png')
    plt.savefig(output_path)
    print(f"✅ Chart saved successfully to:\n{output_path}")
    plt.close()

def plot_all_act_subjects(df):
    """
    Creates a multi-line chart comparing all ACT subject scores over time.
    """
    print("\n--- Generating Comparison Chart for All ACT Subjects ---")
    df = df.sort_values('Year')
    palette = {
        'Math': {'color': 'blue', 'marker': 'o'},
        'English': {'color': 'green', 'marker': 's'},
        'Reading': {'color': '#5E35B1', 'marker': '^'},
        'Science': {'color': 'black', 'marker': 'D'}
    }
    plt.figure(figsize=(12, 7))
    for subject, style in palette.items():
        plt.plot(df['Year'], df[subject], marker=style['marker'], linestyle='-', color=style['color'], label=subject)
    plt.axvline(x=2014, color='gray', linestyle='--', linewidth=2, label='Common Core Adoption (2014)')
    plt.title('National Average ACT Scores by Subject (1992-2022)', fontsize=16)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Average Score', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'act_all_subjects_trend.png')
    plt.savefig(output_path)
    print(f"✅ Chart saved successfully to:\n{output_path}")
    plt.close()

def plot_math_scores_by_decade(df):
    """
    Creates a bar chart of average ACT Math scores grouped by decade.
    """
    print("\n--- Generating Bar Chart for Average Math Scores by Decade ---")
    df['Decade'] = (df['Year'] // 10) * 10
    decade_avg = df.groupby('Decade')['Math'].mean().reset_index()
    decade_avg['Decade'] = decade_avg['Decade'].astype(str) + 's'
    plt.figure(figsize=(10, 6))
    bars = plt.bar(decade_avg['Decade'], decade_avg['Math'], color=['#ADD8E6', '#87CEEB', '#4682B4', '#00008B'])
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}', va='bottom', ha='center', fontsize=12)
    plt.title('Average ACT Math Score by Decade', fontsize=16)
    plt.xlabel('Decade', fontsize=12)
    plt.ylabel('Average Math Score', fontsize=12)
    plt.ylim(top=plt.ylim()[1] + 0.2)
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'act_math_scores_by_decade.png')
    plt.savefig(output_path)
    print(f"✅ Chart saved successfully to:\n{output_path}")
    plt.close()

def plot_sat_total_trend(df):
    """
    Creates and saves a line chart of average SAT Total scores over time.
    """
    print("\n--- Generating SAT Total Score Trend Chart ---")
    df = df.sort_values('Year')
    plt.figure(figsize=(12, 7))
    plt.plot(df['Year'], df['Average Total Score'], marker='o', linestyle='-', color='#4A235A', linewidth=2.5)
    plt.axvline(x=2005, color='gray', linestyle=':', linewidth=2, label='Scoring change to 2400 (2005)')
    plt.axvline(x=2016, color='gray', linestyle=':', linewidth=2, label='Scoring change back to 1600 (2016)')
    plt.title('National Average SAT Total Scores (1990-2022)', fontsize=16)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Average Total Score', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'sat_total_scores_trend.png')
    plt.savefig(output_path)
    print(f"✅ Chart saved successfully to:\n{output_path}")
    plt.close()

def plot_math_score_projection(df):
    """
    Trains a linear regression model on post-2014 data to project future scores.
    """
    print("\n--- Generating ACT Math Score Future Projection Chart ---")
    trend_data = df[df['Year'] >= 2014].copy()
    X = trend_data['Year'].values.reshape(-1, 1)
    y = trend_data['Math'].values
    model = LinearRegression()
    model.fit(X, y)
    future_years = np.array(range(2023, 2041)).reshape(-1, 1)
    predicted_scores = model.predict(future_years)
    plt.figure(figsize=(12, 7))
    plt.plot(df['Year'], df['Math'], marker='o', linestyle='-', color='blue', label='Historical Average Score')
    plt.plot(future_years, predicted_scores, marker='', linestyle='--', color='red', linewidth=2.5, label='Projected Trend')
    plt.title('Projected National Average ACT Math Scores (Based on 2014-2022 Trend)', fontsize=16)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Average Math Score', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'act_math_score_projection.png')
    plt.savefig(output_path)
    print(f"✅ Projection chart saved successfully to:\n{output_path}")
    plt.close()
    print("\n--- Projected Scores ---")
    for year, score in zip(future_years.flatten(), predicted_scores):
        print(f"Year: {year}, Projected Score: {score:.2f}")


# --- Main execution block ---
if __name__ == "__main__":
    act_data, sat_data = load_data()
    
    if act_data is not None and sat_data is not None:
        print("\n--- DataFrames Created ---")
        
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            print(f"\nCreated directory for charts at: {OUTPUT_DIR}")
            
        # Call all the plotting functions
        plot_act_math_trend(act_data)
        plot_all_act_subjects(act_data)
        plot_math_scores_by_decade(act_data)
        plot_sat_total_trend(sat_data)
        plot_math_score_projection(act_data)

