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
