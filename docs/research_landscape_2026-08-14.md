# AI in Education: Research Design and Evidence Landscape

**Research design note, August 2026**

## Research question

How do students, educators, and parents differ in their perceptions and use of generative AI in education, and how are those differences associated with familiarity, actual usage, and institutional guidance?

The purpose of this document is to record the evidence landscape and methodological safeguards behind the public analysis. It is not a product roadmap or an internal task tracker.

## Central methodological rule

Surveys should not be merged into one synthetic dataset merely because they discuss the same topic.

Direct statistical comparisons require compatible populations, dates, technology definitions, question wording, response scales, and sampling designs. When those conditions are not met, results should remain source-specific and descriptive.

A useful comparison hierarchy is:

1. Same survey program, same period, parallel questions.
2. Paired respondent designs, such as parent-child samples.
3. Repeated waves from the same organization with harmonized questions.
4. Separate surveys with demonstrably comparable measures, accompanied by explicit caveats.
5. Different wording, populations, countries, or years: descriptive context only, not pooled inference.

## Evidence sources considered

### Cross-stakeholder and family research

- Walton Family Foundation / Impact Research, *AI in the Classroom*
- Common Sense Media / Ipsos, *The Dawn of the AI Era*

These are useful because they include multiple stakeholder groups or paired family perspectives. Respondent-level data availability and reuse terms must be verified before microdata analysis.

### Student research

- Gallup / Walton Voices of Gen Z
- Higher Education Students' Evolving Perceptions of ChatGPT, 2024-25
- Secondary education student survey data on generative AI use for school work

These sources can support questions about usage, familiarity, perceived usefulness, concern, and subgroup differences. Geographic and sampling limits should remain visible.

### Educator research

- Gallup / Walton K-12 teacher research
- RAND American Educator Panels

These sources are relevant to teacher usage, policy guidance, workload, training, and perceived productivity effects.

### Labor-market and business context

Separate public evidence from sources such as O*NET, Bureau of Labor Statistics data, Census business surveys, and occupation-level exposure research can be used to examine whether attitudes about AI align with measurable adoption or labor-market patterns.

Technical exposure is not treated as proof that a job was replaced by AI.

## Candidate analytical relationships

Potential relationships include:

- stakeholder group and support for educational AI
- usage frequency and perceived usefulness
- familiarity and support for formal integration
- usage and concern about critical thinking
- stakeholder group and academic-integrity concern
- teacher use and perceived workload benefit
- policy clarity and permitted use
- parent awareness and student-reported use
- demographic factors and adoption
- change in usage and attitudes across comparable survey waves

These are hypotheses to test only when the underlying variables and sampling designs support the comparison.

## Constructs requiring careful definition

- usefulness
- academic integrity
- perceived learning effects
- critical thinking
- personalization and accessibility
- teacher workload and productivity
- privacy
- accuracy and misinformation
- appropriate versus inappropriate use
- policy support and restriction
- actual usage frequency
- purpose of use
- AI familiarity
- institutional guidance

Purpose of use is especially important. Brainstorming, tutoring, editing, and generating a finished answer are materially different behaviors and should not automatically be collapsed into one exposure measure.

## Statistical approach

Use the simplest method that the data can defend:

- weighted descriptive statistics when valid survey weights exist
- confidence intervals
- cross-tabulations
- differences in proportions
- chi-square tests for compatible categorical variables
- logistic or ordinal regression when respondent-level data and covariates justify it
- subgroup analysis with appropriate controls
- time trends only for harmonized repeated questions

Public social-media discussion may provide qualitative context, but it is not representative population polling and should not be presented as such.

## Confounders and validity threats

Important factors include:

- selection and self-report bias
- age and education level
- socioeconomic status
- geography
- school policy and teacher training
- grade level and subject
- institution type
- prior academic performance
- device and internet access
- survey year
- specific AI product named in the question
- purpose and frequency of AI use
- teacher-approved versus unapproved use
- differences between ChatGPT-specific and generative-AI-general wording

## Comparability record

For every cross-source comparison, the analysis should record:

- source and survey year
- population
- original question wording
- response scale
- construct being measured
- technology definition
- sampling and weighting details where available
- whether the comparison is directly comparable, approximately comparable, descriptive only, or not comparable

## Causal-language boundary

The project distinguishes among:

- **description:** what a dataset reports
- **association:** variables that move together
- **causal evidence:** designs that credibly identify an effect

Timing alone is not sufficient to attribute changes in test scores, employment, attitudes, or behavior to AI, educational policy, or another intervention.

## Reproducibility and data governance

Public code should make transformations and analytical choices inspectable. Raw third-party datasets should be obtained from their original sources unless redistribution rights are clear. Credentials, private paths, working journals, and bulk user-generated content do not belong in the public repository.

The repository's `data/README.md` records this publication boundary.

## Portfolio value

This research design demonstrates:

- source evaluation
- survey comparability reasoning
- hypothesis formation
- statistical-method selection
- confounder identification
- provenance and reuse awareness
- separation of correlation from causation
- reproducible analytical planning
- communication of uncertainty and limitations
