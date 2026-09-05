# AI in Education: Evidence Before Conclusions

An evidence-first research and analytics project examining generative AI in education, work, and public attitudes without assuming the conclusion in advance.

## Project focus

This project compares perceptions of generative AI with measurable evidence across several domains:

- student and educator adoption
- perceived usefulness and risks
- business AI adoption
- occupational task exposure
- employment and wage outcomes
- long-run education trends

The goal is to separate strong claims from what the available data can actually support.

## Evidence base

The analysis draws on public or appropriately licensed sources including:

- higher-education student survey data
- RAND educator evidence
- Pew public-opinion research
- U.S. Census Business Trends and Outlook Survey data
- O*NET occupation and task data
- occupation-level AI exposure data
- Bureau of Labor Statistics employment and wage data
- national and international education-outcome sources

Large source datasets are not necessarily stored directly in this repository. The repository is focused on reproducible code, documentation, analytical outputs, and source-aware methodology.

## Example findings from the student survey

The acquired higher-education survey contains more than 23,000 respondent rows and 174 variables. Early descriptive results show a mixed picture rather than a simple pro-AI or anti-AI response.

Among non-missing responses:

- 71.4% reported having used ChatGPT
- 68.1% of respondents rating their experience described it as good or very good
- 63.4% agreed it improves access to knowledge
- 59.4% agreed it improves study efficiency
- 58.0% agreed it improves the learning experience
- 46.9% agreed it improves grades
- 44.9% agreed it encourages cheating
- 43.9% agreed it can mislead with inaccurate information
- 43.5% agreed it encourages plagiarism
- 36.0% agreed it improves critical-thinking skills

These results are descriptive and should not be treated as nationally representative because the underlying sample is non-probability based.

## Labor-market analysis

A major analytical track joins occupation exposure measures with O*NET task information and BLS Occupational Employment and Wage Statistics from 2021 through 2025.

The analysis tests several competing explanations:

1. Highly exposed occupations experienced weaker employment growth after 2022.
2. AI exposure is associated more with augmentation than displacement.
3. Exposure may suppress hiring without producing unusually high layoffs.
4. There may be little detectable labor-market effect yet after normal economic variation is considered.
5. Publicly described AI layoffs may have other documented causes unless primary evidence supports direct attribution.

The project treats technical exposure as a hypothesis-generating variable, not proof that a job was eliminated by AI.

## Education-outcomes analysis

A second track compares U.S. academic performance across multiple technological and policy periods using sources such as NAEP, PISA, TIMSS, PIRLS, SAT, and ACT trend evidence where appropriate.

The purpose is not to assign causation from timing alone. It is to compare the magnitude and persistence of changes across different periods and identify where stronger causal evidence would still be required.

## Methodological safeguards

- **Exposure is not replacement.** Technical exposure does not prove job loss.
- **Correlation is not causation.** Observed associations are labeled as such.
- **Survey attitudes are not objective outcomes.** Perceptions and measured outcomes are kept separate.
- **Headline attribution is not accepted automatically.** Primary documentation is preferred for company-level claims.
- **Survey compatibility is audited before combining sources.** Similar topics do not guarantee comparable measures.
- **Predictions and competing hypotheses are recorded before new evidence is interpreted where practical.**
- **Limitations remain visible.** Sampling, missingness, wording, timing, and licensing constraints are documented rather than hidden.

## Reproducible workflow

```text
question
  -> source discovery
  -> dataset acquisition and provenance
  -> raw-data preservation
  -> variable and construct audit
  -> exploratory analysis
  -> competing hypotheses
  -> additional evidence
  -> provisional findings
  -> supported conclusions
```

## Repository structure

```text
AI_in_Education/
├── data/        # Public-safe or appropriately licensed analytical inputs
├── docs/        # Research landscape, tracking, and methodology
├── reports/     # Analysis outputs and reports
├── src/         # Reproducible analysis code
├── README.md
└── requirements.txt
```

## Skills demonstrated

- data acquisition and provenance tracking
- exploratory data analysis
- survey-data reasoning
- Python-based reproducible analysis
- statistical hypothesis framing
- source validation
- dataset joins across public sources
- methodological documentation
- evidence-based communication
- careful treatment of uncertainty and causal claims

## Current takeaway

The evidence collected so far does not support a simple story that people either support or oppose AI. A more defensible provisional interpretation is that AI use is expanding while concern remains substantial, and the measurable consequences vary by context.

This is an active analytical research project. Findings may change as additional evidence is incorporated.