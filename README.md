# AI in Education: Evidence Before Conclusions

**Status:** Active research, August 2026  
**Project ID:** R-001

This project investigates generative AI in education by comparing what people believe AI is doing with measurable evidence about adoption, work, employment, wages, and educational outcomes.

The project began as a smaller sentiment and test-score analysis. It has now expanded into a structured research system combining large survey datasets, U.S. educator and public-opinion evidence, Census business data, occupational task exposure, BLS employment and wage data, and a growing long-run education-outcomes track.

## Core research direction

The current work asks several connected questions:

- How are students, educators, and the U.S. public actually using AI?
- Where do perceived benefits and concerns coexist?
- Which occupations and tasks are theoretically most exposed to generative AI?
- Did highly exposed occupations actually experience weaker employment or wage outcomes after 2022?
- When companies announce layoffs or closures, how often can those changes actually be attributed to AI rather than normal restructuring, contract loss, demand changes, mergers, offshoring, or other causes?
- If U.S. academic performance changed over time, were earlier forces such as Common Core-era instructional changes, social media, smartphones, or COVID associated with larger shifts than generative AI?

## Current evidence base

### Higher-education student survey

The acquired Global ChatGPT Student Survey contains **23,218 respondent rows and 174 variables**.

Early descriptive results from non-missing responses include:

- **71.4%** reported having used ChatGPT.
- **68.1%** of respondents rating their experience described it as good or very good.
- **63.4%** agreed ChatGPT improves access to knowledge.
- **59.4%** agreed it improves study efficiency.
- **58.0%** agreed it improves the learning experience.
- **46.9%** agreed it improves grades.
- **44.9%** agreed it encourages cheating.
- **43.9%** agreed it can mislead with inaccurate information.
- **43.5%** agreed it encourages plagiarism.
- Only **36.0%** agreed it improves critical-thinking skills.

The strongest provisional interpretation is not simply pro-AI or anti-AI. Students appear able to see AI as useful and risky at the same time. Access and efficiency are rated more positively than deeper cognitive outcomes such as critical thinking.

The survey also includes citizenship and country-of-study fields, enabling geographic analysis while preserving the important limitation that the sample is non-probability/convenience based and should not be treated as nationally representative.

### U.S. educator and public evidence

The research includes RAND educator evidence and Pew U.S. public-opinion sources. These sources currently support an **attitude-use gap** hypothesis: AI adoption is increasing while concern about educational, social, and employment consequences remains substantial.

### U.S. business AI adoption

The project has acquired a broad **Census Business Trends and Outlook Survey (BTOS)** package including national, state, metro, sector, subsector, employment-size, sector-by-size, AI supplement, response-rate, questionnaire, methodology, and supporting research files.

This gives the project a way to study where businesses report adopting AI across geography, industry, and business size instead of relying only on headlines or anecdotes.

### Occupation and task exposure

The project has acquired:

- **O*NET 30.3** full database package
- **OpenAI GPTs-are-GPTs** occupation-level exposure data
- OpenAI/O*NET **task-level exposure data**

These datasets provide a framework for asking which jobs contain tasks that current LLMs could plausibly affect.

### Employment and wages

The project has acquired full **BLS Occupational Employment and Wage Statistics (OEWS)** datasets for **2021, 2022, 2023, 2024, and 2025**.

That creates a before-and-after timeline around the public release and expansion of ChatGPT and allows exposure scores to be joined to actual employment and wage outcomes.

The first JOLTS historical layoff series has also been acquired, with industry-level labor-churn data still pending.

## First major joined analysis

The next analytical milestone is:

**OpenAI occupation exposure + O*NET tasks + BLS OEWS 2021-2025**

For each matched occupation, the analysis will calculate measures such as:

```text
occupation
AI exposure score
2021 employment
2025 employment
employment change
2021 wage
2025 wage
wage change
```

The goal is to test competing explanations rather than assume one story in advance.

## Competing labor hypotheses

1. **Displacement:** highly AI-exposed occupations experienced weaker employment growth or outright losses after 2022.
2. **Augmentation:** highly exposed occupations retained or gained employment while wages or productivity improved.
3. **Hiring suppression:** exposed occupations may show slower growth without unusually high layoffs.
4. **Little detectable effect yet:** exposure may have little relationship with 2021-2025 employment outcomes once normal economic variation is considered.
5. **Attribution problem:** many layoffs publicly described as AI-related may actually have other documented causes unless primary company evidence explicitly links them to AI or automation.

## Long-run education outcomes track

A second major expansion is designed to compare U.S. academic performance across multiple technological and policy eras.

Planned evidence includes:

- NAEP Long-Term Trend reading and mathematics, ages 9, 13, and 17
- Main NAEP grades 4, 8, and 12
- NAEP civics, history, geography, and science
- PISA
- TIMSS
- PIRLS
- SAT and ACT trend evidence as secondary sources

The central comparative hypothesis is broader than "did ChatGPT hurt test scores?"

The project will test whether score changes associated with earlier periods such as **social-media expansion, smartphone dominance, Common Core implementation, and COVID disruption** were larger or more persistent than changes observed during the generative-AI era.

These are hypotheses, not conclusions. The analysis is explicitly designed to allow the data to support, weaken, reverse, or reject them.

## Methodological safeguards

This project follows several strict rules:

- **Exposure is not replacement.** A job being technically exposed to LLMs is not evidence that AI eliminated that job.
- **Correlation is not causation.** Employment decline among exposed occupations remains an association until additional labor and company evidence supports attribution.
- **Survey attitudes are not objective outcomes.** Perceptions and measured outcomes are analyzed as separate evidence layers.
- **Headline attribution is not accepted automatically.** Layoffs and closures should be classified using primary documentation wherever possible.
- **Missing evidence should create predictions before acquisition.** Later datasets can then confirm or falsify those predictions rather than being interpreted only after the result is known.
- **Incompatible surveys are not pooled simply because they discuss similar topics.** Wording, sampling, timing, scales, and population definitions must be audited first.

## Research workflow

The project uses an evidence-first workflow:

```text
question
  ↓
source discovery
  ↓
dataset acquisition + provenance
  ↓
raw-data preservation
  ↓
variable / construct audit
  ↓
exploratory analysis
  ↓
competing hypotheses
  ↓
additional evidence
  ↓
provisional findings
  ↓
final conclusions only when supported
```

A future automation layer is planned in which a research engine writes vetted dataset links into a queue and a separate safe Python downloader validates approved URLs, downloads only permitted file types into an inbound folder, hashes files for duplicate detection, and records audit metadata. Human review remains responsible for hypotheses, interpretation, causal claims, and final evidence acceptance.

## Current project state

The project now has enough evidence to begin serious exploratory analysis while continuing targeted acquisition.

**Strong enough to analyze now:**

- student adoption and attitudes
- business AI adoption
- occupation/task AI exposure
- employment and wage outcomes from 2021-2025

**Still being strengthened:**

- industry-level layoffs, hires, quits, and separations
- business births/deaths and job creation/destruction
- verified company-level AI layoff attribution
- long-run national education outcomes
- additional directly comparable U.S. stakeholder microdata

## Repository structure

```text
AI_in_Education/
├── data/        # Raw and processed analytical inputs where appropriate
├── docs/        # Research landscape, tracking, and methodology
├── reports/     # Analysis outputs and reports
├── src/         # Reproducible analysis code
├── README.md
└── requirements.txt
```

Large source datasets are not necessarily stored directly in GitHub. Research source-of-truth files and acquisition records are maintained separately so the repository can remain focused on reproducible code, documentation, and public-safe analytical artifacts.

## Current takeaway

The evidence collected so far does **not** support a simple story that people either support or oppose AI.

A better provisional description is:

> **AI use is expanding rapidly while concern remains substantial. People appear willing to use AI for practical benefits even when they remain uncertain about its consequences.**

The next phase tests whether those perceptions line up with measurable employment, wage, business, and educational outcomes.

---

This is an active research project. Findings in this README are preliminary and will be revised as new datasets are acquired and joined.