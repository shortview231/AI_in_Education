9.6 / 10

⸻

📊 AI in Education: A Data-Driven Analysis

📖 Overview

This project investigates the evolving relationship between artificial intelligence and student learning. Instead of relying on speculation or headlines, we use evidence-based methods to explore how AI is reshaping education—its risks, benefits, and complexities.

Our work is guided by two complementary analytical tracks:
	1.	Qualitative Sentiment Analysis
	•	Scraping and analyzing Reddit discussions to capture public perception of AI in classrooms.
	•	Identifying themes such as cheating, accessibility, feedback, and overreliance.
	•	Revealing a polarized conversation: fears of dishonesty and teacher burnout vs. optimism about equity and innovation.
	2.	Quantitative Trend Analysis
	•	Examining historical SAT & ACT scores (1990–2022) to test hypotheses about declining performance.
	•	Key finding: The sustained decline in ACT Math scores began years before generative AI (e.g., ChatGPT), aligning more closely with the rollout of Common Core math standards.

The goal is simple: move past opinion and into data—building a nuanced, research-grounded understanding of how technology impacts learning.

⸻

⚡ Quickstart Showcase

A glimpse at our results—run the scripts, and you’ll generate visualizations like these:

📊 Reddit Sentiment (681 posts analyzed)
	•	Negative: 46% — plagiarism, apathy, burnout
	•	Positive: 38% — accessibility, innovation, new pedagogy
	•	Neutral: 16% — rubric fairness, balanced policy discussions

(Example chart here — to be replaced with screenshot in /reports/REDDIT/visualizations/)

📉 ACT Math Score Trend (1990–2022)
	•	Stable until 2014 → sharp decline begins with Common Core rollout
	•	Downward trajectory predates AI tools by ~8 years
	•	Projection: Continued decline if no intervention

(Example graph here — to be replaced with /reports/Test_Scores/visualizations/)

👉 For full results, see:
	•	reports/REDDIT/sentiment_report.pdf
	•	reports/Test_Scores/score_analysis.pdf

⸻

📂 Project Structure

AI_in_Education/
│
├── data/                 # Raw and processed datasets
│   ├── REDDIT/           # Reddit posts + sentiment labels
│   └── Test_Scores/      # SAT/ACT historical data
│
├── reports/              # Outputs and final reports
│   ├── REDDIT/
│   └── Test_Scores/
│       ├── visualizations/
│       └── writeup/
│
├── src/                  # Analysis scripts
│   ├── REDDIT/           # Sentiment scraping + labeling
│   └── Test_Scores/      # Score analysis + projections
│
├── .gitignore
├── README.md
└── requirements.txt


⸻

⚙️ Setup and Installation
	1.	Clone the Repository

git clone https://github.com/shortview231/AI_in_Education.git
cd AI_in_Education

	2.	Create a Virtual Environment (recommended)

python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

	3.	Install Dependencies

pip install -r requirements.txt


⸻

🔎 Part 1: Sentiment Analysis of Online Discourse

🎯 Goal & Findings

We analyzed 681 Reddit posts across subreddits like r/education, r/edtech, and r/AIethics.
	•	Negative sentiment: fear of plagiarism, student apathy, AI-crafted assignments.
	•	Positive sentiment: AI as a teaching aid, improved accessibility, redefined pedagogy.
	•	Neutral/analytical sentiment: constructive debates on rubric fairness, AI feedback, policy.

The result is not a single narrative but a landscape of extremes—from panic to possibility.

🔗 Full Sentiment Report

▶️ How to Run

cd src/REDDIT/
python find_posts.py


⸻

📉 Part 2: Standardized Test Score Analysis

🎯 Goal & Hypothesis

Hypothesis: The decline in ACT Math scores correlates more strongly with Common Core adoption (~2014) than with the rise of generative AI (~2022).

📊 Key Findings
	•	Declines are isolated to math.
	•	Downward trend began nearly a decade before AI tools entered classrooms.
	•	Projection models warn of continued decline if unaddressed.

This challenges the “AI ruined test scores” narrative and underscores the importance of evidence before blame.

🔗 Full Score Analysis Report

▶️ How to Run

cd src/Test_Scores/
python analyze_testing_scores.py


⸻

🧑‍💻 Workflow: Human-Led, AI-Assisted

This project is not about outsourcing thinking to AI. Instead:
	•	Human-led strategy: Research questions, hypotheses, and direction set by researcher.
	•	AI-assisted execution: AI helps with data scraping, sentiment labeling, visualization, and polish.
	•	Transparency: Raw drafts + AI-polished reports are shared for accountability.

🔗 Raw Notes
🔗 Polished Report

⸻

🌐 Roadmap & Next Steps
	•	Expand sentiment scraping to Twitter/X, Discord, TikTok.
	•	Add time-trend analysis (pre vs. post-ChatGPT).
	•	Compare AI vs. peer feedback on real student essays.
	•	Explore global equity impacts with Google Trends (high vs. low-resource regions).

⸻

✅ Guiding Principles
	•	Unbiased & Data-Driven — AI is neither inherently good nor bad; we let the data speak.
	•	Educational Integrity — Focus on support, not replacement.
	•	Transparency — Raw data, notes, and polished outputs are all open for review.

This repository is part of the ongoing research project:
AI & The Evolution of Student Learning: A Data-Driven Analysis

⸻

📜 License

This project is released under the MIT License. See LICENSE for details.

⸻

📖 Citation

If you use this work in research, please cite as:

@misc{AIinEducation2025,
  title={AI in Education: A Data-Driven Analysis},
  author={Shortview231},
  year={2025},
  url={https://github.com/shortview231/AI_in_Education}
}


⸻

🤝 Contributing

Contributions are welcome! Please:
	1.	Open an issue to discuss major changes.
	2.	Submit pull requests with clear commit messages.
	3.	Follow the project’s guiding principles (objectivity, transparency, educational integrity).

