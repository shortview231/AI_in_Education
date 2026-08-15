# AI in Education Research Landscape and Design

**Date:** 2026-08-14

## Working title

**The AI Perception Gap: How Students, Educators, and Parents Differ in Their Use, Trust, and Acceptance of Generative AI in Education**

## Primary research question

How do students, educators, and parents differ in their perceptions and use of generative AI in education, and to what extent are those differences associated with AI familiarity, actual usage, and institutional guidance?

## Core methodological decision

Do **not** merge unrelated student, educator, and parent surveys into one synthetic master dataset simply because they discuss AI. Direct statistical stakeholder comparisons should be restricted to surveys with parallel wording, compatible sampling, comparable dates, and compatible definitions of generative AI.

Comparison hierarchy:

1. Same survey, same date, parallel questions: strongest.
2. Paired parent-child survey: very strong.
3. Same organization, repeated survey wave: strong for trends.
4. Separate surveys with nearly identical wording/populations: possible with explicit warnings.
5. Separate surveys with different wording: descriptive only.
6. Different countries, different questions, different years: do not statistically pool.

## Strongest dataset candidates

### Impact Research / Walton AI in the Classroom 2024
- Stakeholders: U.S. K-12 teachers, K-12 students, undergraduates, parents
- Sample: 1,003 teachers; 1,001 K-12 students; 1,003 undergraduates; 1,003 parents
- Strength: strongest direct cross-stakeholder design found because samples were collected within the same research program and period
- Limitation: public respondent-level microdata not yet verified
- URL: https://www.waltonfamilyfoundation.org/ai-in-the-classroom

### Common Sense Media / Ipsos, The Dawn of the AI Era
- Stakeholders: U.S. teens and their parents
- Sample: 1,045 paired households
- Strength: paired parent-child design supports a parent-awareness versus student-behavior question
- Limitation: raw microdata not yet located
- URL: https://www.commonsensemedia.org/research/the-dawn-of-the-ai-era-teens-parents-and-the-adoption-of-generative-ai-at-home-and-school

### Gallup / Walton Voices of Gen Z
- Population: U.S. Gen Z
- Strength: usage, attitudes, familiarity, trends, repeated waves
- Access: downloadable data/crosstabs are available through the research hub
- URL: https://www.gallup.com/analytics/651674/gen-z-research.aspx

### Gallup / Walton K-12 Teacher Research
- Population: U.S. K-12 teachers
- Strength: teacher use, workload, guidance, perceived time savings
- URL: https://news.gallup.com/poll/691967/three-teachers-weekly-saving-six-weeks-year.aspx

### RAND American Educator Panels
- Population: U.S. teachers and education leaders
- Strength: probability-based recruitment, weighting, selected public-use data
- Best use: educator analysis of AI use, lesson planning, instruction, policy, training
- URL: https://www.rand.org/education-employment-infrastructure/survey-panels/aep.html

### Higher Education Students' Evolving Perceptions of ChatGPT, 2024-25
- Population: university students
- Sample: 22,963 respondents across 120 countries/territories
- Raw observations: yes
- Best use: student microdata, subgroup analysis, familiarity/use/attitude relationships
- URL: https://data.mendeley.com/datasets/nv2343nwsb/1

### Secondary education students' views and use of GenAI for school work
- Population: secondary students
- Sample: 1,266
- Raw observations: yes
- Best use: K-12-level student attitudes and behavior
- Limitation: Swedish context limits U.S. generalization
- URL: https://su.figshare.com/articles/dataset/Secondary_education_students_views_and_use_of_GenAI_for_school_work_-_quantitative_survey_data_with_1266_respondents/28850645

## Candidate hypotheses

| ID | Independent variable | Dependent variable | Expected relationship | Candidate test |
|---|---|---|---|---|
| H1 | Stakeholder group | Support for educational AI | Students more supportive than parents; educator position uncertain | Difference in proportions / chi-square |
| H2 | AI usage frequency | Perceived usefulness | More frequent users report greater usefulness | Ordinal/logistic regression |
| H3 | AI familiarity | Support for formal school integration | Greater familiarity predicts greater support | Regression |
| H4 | AI usage frequency | Critical-thinking concern | Frequent users may report lower concern | Chi-square / regression |
| H5 | Stakeholder group | Cheating concern | Educators/parents report greater concern than students | Chi-square |
| H6 | Teacher AI usage | Perceived workload benefit | More frequent use predicts greater benefit | Regression |
| H7 | School AI-policy clarity | Appropriate/permitted use | Clearer policies associated with more permitted use | Chi-square / logistic regression |
| H8 | Parent awareness | Student actual AI use | Parents underestimate teen use | Matched-pair analysis |
| H9 | Demographics / SES | AI adoption | Adoption differs after controls | Multivariable logistic regression |
| H10 | Survey year | AI use vs enthusiasm | Usage rises/stays high while enthusiasm need not rise | Trend analysis |

All hypotheses remain provisional until variable/codebook auditing confirms that the required fields exist.

## Core constructs to audit

- usefulness
- academic integrity / cheating
- learning
- critical thinking
- personalization / accessibility
- teacher workload / productivity
- privacy
- accuracy / misinformation
- appropriate vs inappropriate use
- policy support / restriction / integration
- actual AI use frequency
- purpose of use
- AI familiarity
- institutional guidance

## Recommended statistical approach

Use the simplest defensible method for each question:

- weighted descriptive statistics when survey weights exist
- cross-tabulations and confidence intervals
- differences in proportions
- chi-square tests for categorical variables within valid comparable samples
- logistic or ordinal regression when respondent-level data and covariates justify it
- demographic subgroup analysis with appropriate controls
- year-over-year percentage-point trend analysis only for harmonized repeated questions
- NLP/topic classification for public social-discussion data only as a separate digital-trace analysis, not as representative population polling

Example explanatory model:

`attitude ~ usage + familiarity + age + gender + SES + geography + education`

Avoid causal language unless the design genuinely supports causal inference.

## Important confounders and validity threats

- selection bias
- self-report bias
- AI familiarity
- usage frequency
- age
- education level
- socioeconomic status
- geography
- school AI policy
- teacher training
- subject taught
- grade level
- public/private school
- discipline or major
- parent education
- internet/device access
- language background
- prior academic performance
- survey year
- specific AI product
- purpose of AI use
- teacher-approved versus unapproved use
- urban/rural setting
- institutional AI availability
- ChatGPT-specific wording versus generative-AI-general wording

Purpose of use is especially important. Brainstorming and generating a finished essay should not be collapsed into the same substantive exposure simply because both count as AI use.

## Ranked research gaps

1. **Attitude-use gap across stakeholders**: strongest overall combination of originality, data availability, feasibility, academic value, and portfolio value.
2. **Parent perception vs student behavior**: highly original and academically useful if paired raw data can be obtained.
3. **Policy clarity and appropriate/permitted use**: actionable policy question with good analytical feasibility.
4. **AI familiarity as an explanation of attitude differences**: strong explanatory model with promising microdata.
5. **Critical-thinking concern vs actual use**: potentially interesting contradiction between behavior and concern.
6. **Teacher workload/productivity**: strong supporting analysis, but less novel by itself.
7. **General who-supports-AI-more comparison**: useful baseline, but not sufficient as the paper's main contribution.

## Secondary research questions

1. Which educational AI uses are considered acceptable by each stakeholder group?
2. Does frequent AI use predict stronger perceived usefulness?
3. Does familiarity predict support for formal AI integration?
4. How strongly do stakeholders associate AI with cheating or academic-integrity risks?
5. Does AI use coexist with concern about critical thinking and learning?
6. How accurately do parents understand student AI behavior?
7. Do clear school policies correspond with different student usage patterns?
8. Have attitudes and usage evolved differently between 2023 and 2026?

## Proposed dataset architecture

```text
data/
    raw/
        students/
        educators/
        parents/
        cross_stakeholder/
        social_text/
    processed/
        students_clean.csv
        educators_clean.csv
        parents_clean.csv
        stakeholder_comparison.csv
        survey_metadata.csv
    reference/
        question_crosswalk.csv
        source_registry.csv
        variable_dictionary.csv
```

`question_crosswalk.csv` should record source, survey year, stakeholder, original question, response scale, construct, technology definition, comparison group, and comparison strength.

Recommended comparison labels:

- `DIRECTLY_COMPARABLE`
- `APPROXIMATELY_COMPARABLE`
- `DESCRIPTIVE_ONLY`
- `NOT_COMPARABLE`

## Dashboard concept

Potential pages/sections:

- Stakeholder overview: Student | Educator | Parent
- AI adoption: frequency and purpose of use
- Usefulness vs concern
- Academic integrity by task/scenario
- Learning and critical-thinking perceptions
- Teacher productivity and time savings
- Parent-student awareness gap
- Policy and institutional guidance
- Attitude vs behavior quadrant
- Demographic and geographic subgroup explorer
- Trend view for harmonized repeated survey items

Recommended filters: year, age/grade, geography, AI usage, AI familiarity, demographics, survey source.

## Recommended paper structure

1. Introduction
2. Literature Review
3. Research Questions and Hypotheses
4. Data Sources
5. Dataset Comparability and Harmonization
6. Methods
7. Results
8. Stakeholder Comparison
9. Usage-Attitude Relationship
10. Demographic and Institutional Factors
11. Discussion
12. Limitations
13. Implications
14. Conclusion
15. Reproducibility and Data Appendix

The Dataset Comparability and Harmonization section should be prominent rather than hidden in a footnote.

## Immediate next phase

Do not write the final paper yet.

1. Acquire the confirmed machine-readable datasets.
2. Preserve untouched raw copies.
3. Collect survey instruments and codebooks.
4. Verify licenses and reuse restrictions.
5. Audit variables against the construct list.
6. Build `question_crosswalk.csv`.
7. Build `source_registry.csv`.
8. Finalize only the hypotheses that can actually be tested.
9. Build reproducible cleaning and EDA scripts.
10. Move from raw data -> cleaned data -> SQL/Python analysis -> statistical results -> visualizations/dashboard -> paper -> presentation.

## Research standard

Evidence first, conclusions second. Keep the following categories explicit throughout the project:

- findings directly supported by this project's data
- findings reported by previous literature
- reasonable interpretation
- speculation

A failed or reversed hypothesis is a valid result.