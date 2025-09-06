AI in Education: A Data-Driven Analysis
 Project Overview
This project conducts a multi-faceted analysis of the impact of technology and artificial intelligence on education. Instead of relying on a single narrative, this investigation combines two distinct analytical approaches:

Qualitative Sentiment Analysis: An exploration of public discourse on Reddit to understand the feelings and perceptions surrounding AI in educational contexts.

Quantitative Trend Analysis: An examination of historical SAT and ACT scores to identify long-term trends in student performance and test specific hypotheses related to educational policy and technology.

The goal is to move beyond simple headlines and use data to have a more nuanced conversation about the complex relationship between technology and learning.

 Project Structure
The project is organized into a clean and scalable directory structure:

AI_in_Education/
│
├── data/                 # Contains all raw and processed data files
│   ├── REDDIT/
│   └── Test_Scores/
│
├── reports/              # Contains generated reports and visualizations
│   ├── REDDIT/
│   └── Test_Scores/
│
├── src/                  # Contains all Python source code
│   ├── REDDIT/
│   └── Test_Scores/
│
├── .gitignore            # Specifies files for Git to ignore
├── README.md             # This file
└── requirements.txt      # Lists Python dependencies

 Setup and Installation
To run this project, you will need Python 3 installed. Follow these steps to set up the environment:

Clone the Repository:

git clone <your-repository-url>
cd AI_in_Education

Install Dependencies:
Install all the necessary Python libraries using the requirements.txt file.

pip install -r requirements.txt

Part 1: Sentiment Analysis of Online Discourse
Goal
This section aims to capture the public's emotional response to AI in education by analyzing conversations on Reddit. The goal was to find real-world examples that give context to the polarized debate.

Methodology
Data Source: A dataset of Reddit posts scraped from relevant subreddits like r/education and r/AIethics.

Analysis: A sentiment analysis was performed on each post to generate a "sentiment score" from -1.0 (highly negative) to 1.0 (highly positive).

Script: The analysis is performed by src/REDDIT/find_posts.py. This script sorts the analyzed data to identify the top 3 most positive, negative, and neutral posts and generates a formatted .docx report.

Key Findings
The analysis reveals a conversation of extremes.

Negative Sentiment (-0.99): The most negative posts are deeply personal and often come from educators. They focus on fears of widespread cheating, the erosion of students' critical thinking skills, and a sense of "moral injury" caused by increased workload and institutional pressure.

Positive Sentiment (+1.00): The most positive posts are more philosophical and optimistic. They explore the potential for AI to engage in logical, ethical reasoning and to create new, innovative pedagogical approaches.

Neutral Sentiment (0.00): Neutral posts are almost exclusively factual, news-based announcements (e.g., company acquisitions, awards), indicating that emotional responses are tied to the application of AI, not its mere existence.

How to Run
To regenerate the sentiment analysis report, navigate to the src/REDDIT/ directory and run the script:

python find_posts.py

Part 2: Standardized Test Score Analysis
Goal
This section uses historical ACT and SAT score data (1990-2022) to investigate long-term trends in student performance. The primary goal was to test a specific hypothesis regarding a major educational policy shift.

Hypothesis
The widespread adoption of Common Core State Standards in mathematics, which was largely complete by 2014, correlates with the beginning of a sustained, multi-year decline in national average ACT Math scores. This suggests a systemic issue predating the recent boom in generative AI.

Methodology
Data Source: Publicly available national average score data for the ACT and SAT.

Analysis Script: The script src/Test_Scores/analyze_testing_scores.py loads the data, generates a series of visualizations, and performs a statistical projection.

Key Findings & Discussion
The data provides strong visual evidence to support the hypothesis.

A Decline Predating AI: The line chart of ACT Math scores clearly shows a consistent decline beginning around 2014. This trend started nearly a decade before generative AI tools like ChatGPT became a major factor in education, challenging the narrative that AI is the primary cause of recent performance issues.

The Problem is Isolated to Math: When comparing all four ACT subjects, it is evident that the math score is the only one showing a significant and steady decline. Reading, English, and Science scores have remained relatively stable over the same period. This isolation points towards a math-specific cause, such as a fundamental change in how math was taught nationwide (i.e., Common Core).

A Warning for the Future: A linear projection based on the 2014-2022 trend was created to illustrate the potential severity of this decline. While this is a projection and not a prediction, it serves as a powerful warning that if the underlying trend is not addressed, average scores could continue to fall significantly in the coming years.

Conclusion: While AI presents new challenges to education, the data on standardized test scores suggests that blaming it for the decline in math performance is a misdiagnosis. The evidence points to a more deeply rooted, systemic issue that began years earlier, coinciding with a major shift in educational policy.

How to Run
To regenerate all visualizations for this section, navigate to the src/Test_Scores/ directory and run the script:

python analyze_testing_scores.py
