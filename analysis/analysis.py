# =========================
# Imports
# =========================
import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2, norm
from statsmodels.stats.proportion import proportion_confint

# =========================
# Paths & global settings
# =========================
DATA_PATH = "../data/anonymized_dataset/Tumorboard_ChatGPT_anomyzed_dataset.csv"
TABLE_DIR = "tables"
IMG_DIR = "img"

# Ensure directories exist
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# Plot styling
plt.rcParams["figure.dpi"] = 300
sns.set_style("whitegrid")

# =========================
# Column definitions
# =========================
REFERENCE_COL = "anonymized_tumorboard_recommedation"

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

# Map values to nicer labels
VALUE_RENAME = {
    "esophageal": "Oesophageal cancer",
    "pancreatic": "Pancreatic cancer",
    "gastric": "Gastric cancer",
    "colorectal": "Colorectal cancer",
    "hepatobiliary": "Hepatobiliary cancer",
    "1": "First presentation",
    "2": "Follow-up presentation"
}

# Colors for bar plots
COLOR_4O = '#1E90FF'
COLOR_4O_MINI = '#FFD700'

# =========================
# Data loading
# =========================
def load_data(path: str) -> pd.DataFrame:
    """
    Load anonymized tumor board dataset.

    The 'anonymized_tumorboard_recommedation_treatment' column is expected to contain lists.
    """
    df = pd.read_csv(path, converters={
        "anonymized_tumorboard_recommedation_treatment": lambda x: eval(x) if pd.notna(x) else []
    })
    print(f"Loaded dataset with {len(df)} cases")
    return df

# =========================
# Correctness calculations
# =========================
def calculate_correct_percentages(df: pd.DataFrame, model_cols: list) -> dict:
    """Return the percentage of concordant recommendations per model."""
    return {col: df[f"{col}_treatment_concordance"].mean() * 100 for col in model_cols}

def calculate_correct_counts(df: pd.DataFrame, model_cols: list) -> dict:
    """Return the count of concordant recommendations per model."""
    return {col: df[f"{col}_treatment_concordance"].sum() for col in model_cols}

# =========================
# Overall agreement with 95% CI
# =========================
def compute_overall_agreement_with_CI(df: pd.DataFrame, model_cols: list, alpha=0.05, save_dir=None):
    """
    Compute overall agreement percentages with 95% confidence intervals (Wilson method)
    for all models.

    Returns a dictionary and optionally saves a CSV to `save_dir`.
    """
    n_total = len(df)
    results = {}

    print("\n=== Overall Agreement Percentages with 95% CI ===")
    for col in model_cols:
        correct = df[f"{col}_treatment_concordance"].sum()
        p = correct / n_total
        ci_low, ci_up = proportion_confint(count=correct, nobs=n_total, alpha=alpha, method='wilson')
        results[col] = {"percent": p*100, "ci_lower": ci_low*100, "ci_upper": ci_up*100}
        print(f"{col}: {p*100:.2f}% ({correct}/{n_total}), 95% CI: {ci_low*100:.2f}% - {ci_up*100:.2f}%")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        ci_df = pd.DataFrame.from_dict(results, orient='index')
        save_path = os.path.join(save_dir, "overall_agreement_with_CI.csv")
        ci_df.to_csv(save_path)
        print(f"Saved overall agreement with 95% CI to {save_path}")

    return results

# =========================
# Statistical tests
# =========================
def run_mcnemar(df: pd.DataFrame, col1: str, col2: str, alpha=0.05) -> dict:
    """Perform McNemar test for two treatment concordance columns."""
    b = np.sum((df[f"{col1}_treatment_concordance"] == 1) & (df[f"{col2}_treatment_concordance"] == 0))
    c = np.sum((df[f"{col1}_treatment_concordance"] == 0) & (df[f"{col2}_treatment_concordance"] == 1))
    stat = ((abs(b - c) - 1) ** 2) / (b + c) if (b + c) > 0 else 0
    p = chi2.sf(stat, 1)
    return {"b": b, "c": c, "statistic": stat, "pvalue": p, "reject": p < alpha}

def run_cochran_q(df: pd.DataFrame, model_cols: list, alpha=0.05) -> dict:
    """Perform Cochran's Q test across all models."""
    X = df[[f"{col}_treatment_concordance" for col in model_cols]].values
    k = X.shape[1]
    row_sums = X.sum(axis=1)
    col_sums = X.sum(axis=0)
    T = X.sum()
    Q = (k * (k - 1) * np.sum((col_sums - T / k) ** 2)) / np.sum(row_sums * (k - row_sums))
    p = chi2.sf(Q, k - 1)
    return {"Q": Q, "df": k - 1, "pvalue": p, "reject": p < alpha}

# =========================
# McNemar Power
# =========================
def mcnemar_power(b, c, alpha=0.05):
    """Approximate power of McNemar's test using normal approximation."""
    n_disc = b + c
    if n_disc == 0:
        return 0.0
    delta = abs(b - c) / np.sqrt(b + c)
    z_alpha = norm.ppf(1 - alpha / 2)
    return 1 - norm.cdf(z_alpha - delta)

def mcnemar_power_from_df(df, col1, col2, alpha=0.05):
    """Compute McNemar power from two columns of treatment concordance."""
    b = np.sum((df[f"{col1}_treatment_concordance"] == 1) & (df[f"{col2}_treatment_concordance"] == 0))
    c = np.sum((df[f"{col1}_treatment_concordance"] == 0) & (df[f"{col2}_treatment_concordance"] == 1))
    power = mcnemar_power(b, c, alpha)
    return {"b": b, "c": c, "discordant_n": b + c, "power": power}

def compute_and_plot_power_matrix(df, model_cols, rename_dict, alpha=0.05, figsize=(12,12), save_dir="img"):
    """
    Compute McNemar power for all model pairs and plot as a symmetric matrix.

    Saves a CSV and PNG figure to `save_dir`.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Rename columns for display
    renamed_columns = [rename_dict.get(col, col) for col in model_cols]
    axis_labels = [f"{col.split('_')[0]} 4o-mini" if "4o-mini" in col else f"{col.split('_')[0]} 4o" for col in renamed_columns]

    n = len(model_cols)
    power_matrix = pd.DataFrame(np.zeros((n,n)), index=renamed_columns, columns=renamed_columns)

    # Compute pairwise power
    for i in range(n):
        for j in range(i, n):
            if i == j:
                power_matrix.iloc[i,j] = 1.0
            else:
                res = mcnemar_power_from_df(df, model_cols[i], model_cols[j], alpha)
                power_matrix.iloc[i,j] = res['power']
                power_matrix.iloc[j,i] = res['power']

    # Plot matrix
    plt.figure(figsize=figsize)
    im = plt.imshow(power_matrix, cmap='coolwarm', vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            plt.text(j, i, f"{power_matrix.iloc[i,j]:.2f}", ha='center', va='center', fontsize=8)
    plt.xticks(range(n), axis_labels, rotation=90)
    plt.yticks(range(n), axis_labels)
    plt.colorbar(im, label="McNemar Power")
    plt.title("McNemar Power Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "mcnemar_power_matrix.png"), dpi=300, bbox_inches='tight')
    print(f"Power matrix figure saved at {save_dir}/mcnemar_power_matrix.png")
    plt.show()

    # Save CSV
    csv_path = os.path.join('tables', "mcnemar_power_matrix.csv")
    power_matrix.to_csv(csv_path)
    print(f"Power matrix CSV saved at {csv_path}")

    return power_matrix


# =========================
# Full / Mostly Right/Wrong Analysis
# =========================
def analyze_correctness_extremes(df: pd.DataFrame, model_cols: list, comparison_col_prefix="_treatment_concordance",
                                 save_dir="tables"):
    """
    Analyze cases that are always wrong, mostly right (>90%), and always right (100%).

    Adds a 'Correct_Percentage' column to df, saves subsets to Excel,
    and prints counts & diagnosis distributions.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Calculate number of correct models per row
    correctness_counts = df[[f"{col}{comparison_col_prefix}" for col in model_cols]].sum(axis=1)

    total_models = len(model_cols)
    df['Correct_Percentage'] = (correctness_counts / total_models) * 100

    # Subsets
    always_wrong = df[df['Correct_Percentage'] <= 30].copy()
    mostly_right_90 = df[df['Correct_Percentage'] > 90].copy()
    always_right_100 = df[df['Correct_Percentage'] == 100].copy()

    # Save to Excel
    always_wrong_path = os.path.join(save_dir, "always_wrong.csv")
    mostly_right_90_path = os.path.join(save_dir, "mostly_right_90.csv")
    always_right_100_path = os.path.join(save_dir, "always_right_100.csv")

    always_wrong.to_csv(always_wrong_path, index=False)
    mostly_right_90.to_csv(mostly_right_90_path, index=False)
    always_right_100.to_csv(always_right_100_path, index=False)

    # Print counts & diagnosis distribution
    print("\n=== Always Wrong (≤30%) ===")
    print(f"Count: {len(always_wrong)}")
    print(always_wrong['tumour_type'].value_counts(), end="\n\n")

    print("=== Mostly Right (>90%) ===")
    print(f"Count: {len(mostly_right_90)}")
    print(mostly_right_90['tumour_type'].value_counts(), end="\n\n")

    print("=== Always Right (100%) ===")
    print(f"Count: {len(always_right_100)}")
    print(always_right_100['tumour_type'].value_counts(), end="\n\n")

    print(f"Excel files saved to {save_dir}:")
    print(f"- Always Wrong: {always_wrong_path}")
    print(f"- Mostly Right (>90%): {mostly_right_90_path}")
    print(f"- Always Right (100%): {always_right_100_path}")

    return always_wrong, mostly_right_90, always_right_100


# =========================
# Visualization functions
# =========================
def create_bar_plot(percentages: dict, title: str, save_path: str = None):
    """Bar plot of agreement percentages."""
    plt.figure(figsize=(12, 6))
    colors = [COLOR_4O_MINI if '4o-mini' in k else COLOR_4O for k in percentages.keys()]
    bars = plt.bar(percentages.keys(), percentages.values(), color=colors)
    plt.xticks(rotation=90)
    plt.ylim(0, 110)
    plt.ylabel("Agreement with tumor board (%)")
    plt.title(title)
    for bar, value in zip(bars, percentages.values()):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.1f}%", ha='center', va='bottom')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {save_path}")
    plt.show()

# =========================
# Sub-analysis by categorical column
# =========================
def sub_analysis_by_column(df: pd.DataFrame, column_name: str, model_cols: list, save_dir=None):
    """Plot agreement percentages by unique values of a categorical column."""
    for val in df[column_name].unique():
        sub_df = df[df[column_name] == val]
        percentages = calculate_correct_percentages(sub_df, model_cols)
        title = f"Agreement by {column_name} = {VALUE_RENAME.get(val, val)} (N={len(sub_df)})"
        save_path = os.path.join(save_dir, f"{title.replace(' ', '_')}.png") if save_dir else None
        create_bar_plot(percentages, title, save_path)


# =========================
# Jaccard Similarity Analysis
# =========================
def analyze_jaccard_similarity(df: pd.DataFrame, save_dir="tables"):
    """
    Compute Jaccard similarities between retrieval strategies.

    1. Converts the index columns (sets of integers) to Python sets.
    2. Computes pairwise Jaccard similarity per case.
    3. Computes average Jaccard similarity matrix.
    4. Saves per-case similarities and average matrix as CSVs.
    5. Plots heatmaps (optional).

    Expects columns (renamed version):
        'Chunk Indices_RAG Full Corpora',
        'Chunk Indices_RAG Selected Corpora',
        'Chunk Indices_rw_RAG Full Corpora',
        'Chunk Indices_rw_RAG Selected Corpora'
    """

    os.makedirs(save_dir, exist_ok=True)

    # Columns containing sets of indices
    index_cols = [
        'Chunk Indices_RAG Full Corpora',
        'Chunk Indices_RAG Selected Corpora',
        'Chunk Indices_rw_RAG Full Corpora',
        'Chunk Indices_rw_RAG Selected Corpora'
    ]

    # Convert string to set of ints if not already set
    def to_set(s):
        if isinstance(s, set):
            return s
        if pd.isna(s) or s == "":
            return set()
        return set(map(int, s.split(',')))

    for col in index_cols:
        df[col] = df[col].apply(to_set)

    # Define Jaccard similarity
    def jaccard(a, b):
        return len(a & b) / len(a | b) if len(a | b) > 0 else 0

    # Compute pairwise Jaccard per case
    jaccard_cols = {
        'Jaccard_RAG_Full_vs_Selected': ('Chunk Indices_RAG Full Corpora', 'Chunk Indices_RAG Selected Corpora'),
        'Jaccard_RAG_Full_vs_rw_Full': ('Chunk Indices_RAG Full Corpora', 'Chunk Indices_rw_RAG Full Corpora'),
        'Jaccard_RAG_Full_vs_rw_Selected': ('Chunk Indices_RAG Full Corpora', 'Chunk Indices_rw_RAG Selected Corpora'),
        'Jaccard_rw_Full_vs_rw_Selected': ('Chunk Indices_rw_RAG Full Corpora', 'Chunk Indices_rw_RAG Selected Corpora')
    }

    for jcol, (col1, col2) in jaccard_cols.items():
        df[jcol] = df.apply(lambda row: jaccard(row[col1], row[col2]), axis=1)

    # Save per-case similarities
    per_case_csv = os.path.join(save_dir, "jaccard_similarity_per_case.csv")
    df[list(jaccard_cols.keys())].to_csv(per_case_csv, index=False)
    print(f"Per-case Jaccard similarities saved to {per_case_csv}")

    # Compute average pairwise Jaccard similarities (4x4 matrix)
    methods = {name: df[col] for name, col in zip(
        ['RAG_Full', 'RAG_Selected', 'rw_RAG_Full', 'rw_RAG_Selected'],
        index_cols
    )}

    similarity_matrix = pd.DataFrame(index=methods.keys(), columns=methods.keys(), dtype=float)

    for name1 in methods:
        for name2 in methods:
            sims = [jaccard(a, b) for a, b in zip(methods[name1], methods[name2])]
            similarity_matrix.loc[name1, name2] = sum(sims) / len(sims)

    # Save average matrix
    matrix_csv = os.path.join(save_dir, "jaccard_similarity_matrix.csv")
    similarity_matrix.to_csv(matrix_csv)
    print(f"Average Jaccard similarity matrix saved to {matrix_csv}")

    plt.figure(figsize=(8, 6))
    sns.heatmap(similarity_matrix, annot=True, cmap='YlGnBu', vmin=0, vmax=1, fmt=".2f")
    plt.title("Average Jaccard Similarity Between Retrieval Strategies")
    plt.tight_layout()
    heatmap_path = os.path.join(IMG_DIR, "jaccard_similarity_matrix_heatmap.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Heatmap saved to {heatmap_path}")

    return df, similarity_matrix


# =========================
# Charts Retrieved Analysis
# =========================
def analyze_charts_retrieved(df: pd.DataFrame, save_dir="tables"):
    """
    Analyze how many charts were retrieved by different RAG strategies.

    Columns expected:
        - 'Charts N_RAG Full Corpora'
        - 'Charts N_RAG Selected Corpora'
        - 'Charts N_rw_RAG Full Corpora'
        - 'Charts N_rw_RAG Selected Corpora'

    Returns:
        overall_mean, mean_by_diagnosis, mean_by_presentation, mean_by_treatment
    """

    os.makedirs(save_dir, exist_ok=True)

    columns_to_analyze = [
        'Charts N_RAG Full Corpora',
        'Charts N_RAG Selected Corpora',
        'Charts N_rw_RAG Full Corpora',
        'Charts N_rw_RAG Selected Corpora'
    ]

    # Convert treatment column lists to strings to allow groupby
    df['treatment_str'] = df['anonymized_tumorboard_recommedation_treatment'].apply(lambda x: str(x))

    # Overall mean
    overall_mean = df[columns_to_analyze].mean()
    print("=== Overall Mean ===")
    print(overall_mean)

    # Mean by diagnosis
    mean_by_diagnosis = df.groupby('tumour_type')[columns_to_analyze].mean().reset_index()
    print("\n=== Mean by Diagnosis ===")
    print(mean_by_diagnosis)

    # Mean by presentation
    mean_by_presentation = df.groupby('presentation')[columns_to_analyze].mean().reset_index()
    print("\n=== Mean by Presentation ===")
    print(mean_by_presentation)

    # Save CSVs
    overall_mean.to_csv(os.path.join(save_dir, "charts_overall_mean.csv"))
    mean_by_diagnosis.to_csv(os.path.join(save_dir, "charts_mean_by_diagnosis.csv"), index=False)
    mean_by_presentation.to_csv(os.path.join(save_dir, "charts_mean_by_presentation.csv"), index=False)
    print(f"\nCharts retrieved analysis saved to folder: {save_dir}")

    return overall_mean, mean_by_diagnosis, mean_by_presentation



# =========================
# Main execution
# =========================
if __name__ == "__main__":
    df = load_data(DATA_PATH)

    # Overall percentages
    percentages = calculate_correct_percentages(df, MODEL_COLS)
    print("Overall agreement percentages:")
    for k, v in percentages.items():
        print(f"{k}: {v:.2f}%")

    # Save overall bar plot
    create_bar_plot(percentages, "Overall Agreement Across Models", os.path.join(IMG_DIR, "overall_agreement.png"))

    # Overall agreement with 95% CI
    overall_ci = compute_overall_agreement_with_CI(df, MODEL_COLS, alpha=0.05, save_dir=TABLE_DIR)

    # Cochran Q + pairwise McNemar
    print("\n=== Cochran's Q Test ===")
    res_q = run_cochran_q(df, MODEL_COLS)
    print(res_q)

    print("\n=== Pairwise McNemar Tests ===")
    for col1, col2 in itertools.combinations(MODEL_COLS, 2):
        res_m = run_mcnemar(df, col1, col2)
        print(f"{col1} vs {col2}: χ²={res_m['statistic']:.2f}, p={res_m['pvalue']:.4f}, reject={res_m['reject']}")

    # McNemar power matrix
    power_matrix = compute_and_plot_power_matrix(df, MODEL_COLS, VALUE_RENAME, alpha=0.05, figsize=(14, 14), save_dir=IMG_DIR)

    # Extreme correctness analysis
    always_wrong, mostly_right_90, always_right_100 = analyze_correctness_extremes(df, MODEL_COLS, save_dir=TABLE_DIR)

    # Sub-analyses
    sub_analysis_by_column(df, "tumour_type", MODEL_COLS, save_dir=IMG_DIR)
    sub_analysis_by_column(df, "presentation", MODEL_COLS, save_dir=IMG_DIR)

    # Save overall percentages as CSV
    pd.DataFrame.from_dict(percentages, orient='index', columns=['Agreement_Percentage']).to_csv(os.path.join(TABLE_DIR, "overall_percentages.csv"))
    print(f"Saved overall percentages table to {TABLE_DIR}/overall_percentages.csv")

    # Jaccard similarity analysis
    df, jaccard_matrix = analyze_jaccard_similarity(df, save_dir=TABLE_DIR)

    # Charts retrieved analysis
    charts_overall, charts_by_diag, charts_by_ev_wv = analyze_charts_retrieved(df, save_dir=TABLE_DIR)
