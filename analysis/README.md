# Analysis Scripts – Reproducible Analyses

This folder contains the scripts used to reproduce the analyses presented in the manuscript. These are simplified scripts intended to allow full reproducibility of the reported results.

## Scripts

### 1. `demographics.py`
Performs demographic analyses of the anonymized dataset:

- **Age statistics**: median and interquartile range (overall and by tumor type)  
- **Gender distribution**: counts and percentages (overall and by tumor type)  
- **Presentation status**: first vs. follow-up presentation (overall and by tumor type)  

**Outputs:**

- **Tables**: CSVs of age, gender, and presentation statistics saved in `tables/`  

---

### 2. `analysis.py`
Performs the main evaluation of frameworks' recommendations against the anonymized tumor board dataset. The key analyses include:

- **Overall concordance** of each framework (`calculate_correct_percentages`, `calculate_correct_counts`)  
- **Concordance with tumor board** including 95% confidence intervals (`compute_overall_agreement_with_CI`)  
- **Statistical testing**: Cochran’s Q test and pairwise McNemar tests  
- **McNemar power analysis** and visualization as a heatmap  
- **Extreme correctness analysis**: identifies cases that are always wrong, mostly right (>90%), or always right (100%)  
- **Sub-analyses by category**: agreement percentages by tumor type or presentation  
- **Jaccard similarity analysis** between retrieval strategies  
- **Charts retrieved analysis**: counts of clinical charts retrieved per strategy  

**Outputs:**

- **Tables**: saved as CSVs in `tables/`  
- **Figures**: saved in `img/`  
- **Subsets**: CSVs of always wrong, mostly right, and always right cases  

---

### 3. `sensitivity_analysis_thresholds.py`
Sensitivity analysis for the 'Mostly Wrong / Mostly Correct / Fully Concordant' case classification thresholds used in the Framework Consistency Analysis (main manuscript).
The original thresholds (≤30% / >90% / 100%) were selected based on clinical reasoning to identify cases at the extremes of the cross-framework performance distribution. This script verifies that the key finding — 4 universally challenging cases failing across virtually all configurations — is robust to reasonable threshold variations.

- 'Mostly Wrong' thresholds tested: ≤20%, ≤25%, ≤30%, ≤35%, ≤40%
- 'Mostly Correct' thresholds tested: >80%, >85%, >90%, >95%
- 'Fully Concordant': 100% (threshold-independent by definition)

**Output**: printed to console only — no files saved.

### 4. `analyze_demographic_performance.py`
Exploratory analysis of concordance performance by demographic variables (age and sex) across all 16 RAG framework configurations, addressing the equity dimension raised in peer review.

- **Sex comparison**: mean concordance by sex (Mann-Whitney U test)
- **Age correlation**: Spearman correlation between age and mean concordance (all configurations and optimal configuration only)
- **Age tertile analysis**: concordance by age group (Kruskal-Wallis test)
- **Demographic distribution by tumour type**: age and sex distributions across cancer types (Kruskal-Wallis and Chi-square tests)

**Outputs:**

- **Tables**: tables/demographic_performance_summary.csv, tables/demographic_concordance_by_sex.csv, tables/demographic_age_correlation.csv, tables/demographic_by_tumour_type.csv
- **Figures**: img/demographic_performance_figure.png (3-panel: concordance by sex, age vs concordance scatter, concordance by age tertile)


### 5. `analyze_retrieval_quality.py`
Retrieval quality analysis across RAG configurations, quantifying the actionability of retrieved guideline chunks and retrieval similarity scores across full and curated corpus configurations.

- **Actionability analysis**: proportion of non-actionable chunks (epidemiology, prevention, reference lists) among top-5 retrieved chunks per case across 6 RAG configurations
- **Retrieval similarity scores**: mean and minimum cosine similarity of retrieved chunks as a proxy for retrieval confidence
- **Tumour type subgroup analysis**: non-actionable chunk rates and similarity scores by cancer type (full corpus configurations only)

**Outputs:**

- **Tables**: tables/retrieval_quality_summary.csv, tables/retrieval_quality_by_tumour_type.csv
- **Figures**: img/retrieval_quality_figure.png (3-panel: retrieval noise per configuration, mean similarity scores, per-case distribution for full corpus configurations)

## Usage

1. Ensure the anonymized dataset is placed in:  
```go
../data/anonymized_dataset/Tumorboard_ChatGPT_anomyzed_dataset.csv
```
2. Run the scripts:  
```bash
python demographics.py
python analysis.py
```
3. Results (tables and figures) will be saved in tables/ and img/ folders.

---

Reproducibility

Scripts are fully reproducible; no randomness is used.

All outputs correspond to the analyses used in the manuscript.

These are simplified scripts intended to allow reproduction of the reported results.
