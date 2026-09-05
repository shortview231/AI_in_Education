# Data Boundary and Provenance

This public portfolio repository does not redistribute raw third-party datasets, bulk scraped social-media text, credentials, private working journals, or machine-specific configuration.

## Reproducing the test-score analysis

`src/Test_Scores/analyze_testing_scores.py` expects locally supplied aggregate source tables under:

```text
data/Test_Scores/local_inputs/
```

Expected filenames and minimum columns:

- `act_averages_test_takers.csv`: `Year`, `Math`, `English`, `Reading`, `Science`
- `sat_totals_participation.csv`: `Year`, `Average Total Score`

Researchers should obtain source data from the original publisher or another source whose terms permit their intended use, document provenance, and verify comparability across years before analysis.

## Social-media data

Earlier exploratory work used public social-media discussion as a nonrepresentative qualitative/sentiment signal. Raw posts, processed post-level datasets, API credentials, and generated reports that reproduced post text are intentionally not republished here.

## Research rule

Public code and reports should make methods inspectable without republishing data that carries unclear redistribution rights, unnecessary user-generated content, credentials, private paths, or unrelated working materials.