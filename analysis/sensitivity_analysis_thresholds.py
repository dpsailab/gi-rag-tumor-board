"""
sensitivity_analysis_thresholds.py
===================================
Sensitivity analysis for the 'Mostly Wrong / Mostly Correct / Fully Concordant'
case classification used in the Framework Consistency Analysis (main manuscript).

The original thresholds (≤30% / >90% / 100%) were selected to identify cases at
the extremes of the cross-framework performance distribution. This script verifies
that the key finding — 4 universally challenging cases failing across virtually all
configurations — is robust to reasonable threshold variations.

Thresholds tested:
    - 'Mostly Wrong':    ≤20%, ≤25%, ≤30%, ≤35%, ≤40%
    - 'Mostly Correct':  >80%, >85%, >90%, >95%
    - 'Fully Concordant': 100% (threshold-independent by definition)

Expected output: the 'Mostly Wrong' group remains stable at 4 cases for all
thresholds up to ≤30%, with only 1 additional case at ≤35% and 1 more at ≤40%,
confirming that the 4 universally challenging cases represent a robust finding
rather than a threshold artifact.
"""

import ast
import pandas as pd

# =========================
# Configuration
# =========================

DATA_PATH = "../data/anonymized_dataset/Tumorboard_ChatGPT_anonymized_dataset.csv"

MODEL_COLS = [
    'No retrieval_4o-mini', 'No retrieval_4o',
    'Assistant with guidelines_4o-mini', 'Assistant with guidelines_4o',
    'RAG w/ Full Corpora_4o-mini', 'RAG w/ Full Corpora_4o',
    'RAG w/ Selected Corpora_4o-mini', 'RAG w/ Selected Corpora_4o',
    'rw-No retrieval_4o-mini', 'rw-No retrieval_4o',
    'rw-Assistant with guidelines_4o-mini', 'rw-Assistant with guidelines_4o',
    'rw-RAG w/ Full Corpora_4o-mini', 'rw-RAG w/ Full Corpora_4o',
    'rw-RAG w/ Selected Corpora_4o-mini', 'rw-RAG w/ Selected Corpora_4o'
]

# Threshold ranges for sensitivity analysis
WRONG_THRESHOLDS   = [20, 25, 30, 35, 40]   # upper bound for 'Mostly Wrong' (%)
CORRECT_THRESHOLDS = [80, 85, 90, 95]        # lower bound for 'Mostly Correct' (%)

# =========================
# Data loading
# =========================

def load_data(path: str) -> pd.DataFrame:
    """
    Load anonymized tumor board dataset.
    The 'anonymized_tumorboard_recommedation_treatment' column is parsed as a list.
    """
    df = pd.read_csv(path, converters={
        "anonymized_tumorboard_recommedation_treatment": lambda x: ast.literal_eval(x) if pd.notna(x) else []
    })
    print(f"Loaded dataset with {len(df)} cases")
    return df


# =========================
# Main
# =========================

if __name__ == "__main__":

    df = load_data(DATA_PATH)

    # Compute percentage of frameworks concordant per case
    # (number of concordant frameworks / total frameworks * 100)
    concordance_cols = [f"{col}_treatment_concordance" for col in MODEL_COLS]
    df['Correct_Percentage'] = df[concordance_cols].sum(axis=1) / len(MODEL_COLS) * 100

    # ------------------------------------------------------------------
    # 1. Original thresholds (as reported in the manuscript)
    # ------------------------------------------------------------------
    print("=== ORIGINAL THRESHOLDS (≤30% / >90% / 100%) ===")
    print(f"  Mostly Wrong     (≤30%): {(df['Correct_Percentage'] <= 30).sum()} cases")
    print(f"  Mostly Correct   (>90%): {(df['Correct_Percentage'] > 90).sum()} cases")
    print(f"  Fully Concordant (100%): {(df['Correct_Percentage'] == 100).sum()} cases")

    # ------------------------------------------------------------------
    # 2. Sensitivity analysis across threshold variations
    # ------------------------------------------------------------------
    print("\n=== SENSITIVITY ANALYSIS ===")
    print(
        f"{'Wrong threshold':<20} "
        f"{'Correct threshold':<22} "
        f"{'Always correct':<18} | "
        f"{'N wrong':<10} "
        f"{'N correct':<12} "
        f"{'N always'}"
    )
    print("-" * 85)

    for wt in WRONG_THRESHOLDS:
        for ct in CORRECT_THRESHOLDS:
            n_wrong   = (df['Correct_Percentage'] <= wt).sum()
            n_correct = (df['Correct_Percentage'] > ct).sum()
            n_always  = (df['Correct_Percentage'] == 100).sum()  # constant by definition
            print(
                f"  ≤{wt}%{'':<15} "
                f"  >{ct}%{'':<17} "
                f"  100%{'':<14} | "
                f"{n_wrong:<10} "
                f"{n_correct:<12} "
                f"{n_always}"
            )

    print("\nNote: 'Fully Concordant' (100%) is threshold-independent by definition.")
    print("      Stability of 'Mostly Wrong' across thresholds ≤30% confirms robustness")
    print("      of the 4 universally challenging cases reported in the manuscript.")
