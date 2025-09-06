AI in Education: A Data-Driven Analysis
📖 Project Overview
This project conducts a multi-faceted analysis of the impact of technology and artificial intelligence on education. Instead of relying on a single narrative, this investigation combines two distinct analytical approaches:

Qualitative Sentiment Analysis: An exploration of public discourse on Reddit to understand the feelings and perceptions surrounding AI in educational contexts.

Quantitative Trend Analysis: An examination of historical SAT and ACT scores to identify long-term trends in student performance and test specific hypotheses related to educational policy and technology.

The goal is to move beyond simple headlines and use data to have a more nuanced conversation about the complex relationship between technology and learning.

📂 Project Structure
The project is organized into a clean and scalable directory structure:

AI_in_Education/
│
├── data/
│   ├── REDDIT/
│   └── Test_Scores/
│
├── reports/
│   ├── REDDIT/
│   └── Test_Scores/
│       ├── visualizations/
│       └── writeup/
│
├── src/
│   ├── REDDIT/
│   └── Test_Scores/
│
├── .gitignore
├── README.md
└── requirements.txt

⚙️ Setup and Installation
To run this project, you will need Python 3 installed. Follow these steps to set up the environment:

Clone the Repository:

git clone [https://github.com/shortview231/AI_in_Education.git](https://github.com/shortview231/AI_in_Education.git)
cd AI_in_Education

Install Dependencies:

pip install -r requirements.txt

Part 1: Sentiment Analysis of Online Discourse
Goal & Findings
This section captures the public's emotional response to AI in education by analyzing conversations on Reddit. The analysis reveals a conversation of extremes, with negative sentiment focusing on personal fears of cheating and educator burnout, while positive sentiment explores the philosophical and innovative potential of AI.

🔗 View the Full Sentiment Analysis Report

How to Run
To regenerate the sentiment analysis report, navigate to the src/REDDIT/ directory and run the script:

python find_posts.py

Part 2: Standardized Test Score Analysis
Goal & Hypothesis
This section uses historical ACT and SAT score data (1990-2022) to investigate long-term trends in student performance. The primary hypothesis is that the widespread adoption of Common Core State Standards in mathematics (c. 2014) correlates with the beginning of a sustained decline in national average ACT Math scores, a trend that predates the recent boom in generative AI.

Key Findings & Discussion
The data provides strong visual evidence to support this hypothesis, challenging the popular narrative that AI is the primary cause of declining performance. The analysis shows a decline that is almost exclusively isolated to math and began nearly a decade before tools like ChatGPT became mainstream. A statistical projection warns that if this trend continues, scores could continue to fall significantly.

My Workflow: A Demonstration of Human-Led, AI-Assisted Analysis
My process is built on a clear division of labor: I provide the strategic direction and core insights, while leveraging AI as an advanced assistive tool for implementation and final polish. To demonstrate this workflow transparently, I have included both my raw, unedited analytical notes and the final, AI-polished report for this section.

🔗 View My Raw Analytical Notes (The 'Dirty' Draft)

🔗 View the Final Polished Report

How to Run
To regenerate all visualizations for this section, navigate to the src/Test_Scores/ directory and run the script:

python analyze_testing_scores.py
